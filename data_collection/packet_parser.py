"""
packet_parser.py
----------------
Parses raw ESP32 WiFi CSI CSV files into structured numpy arrays.

Input format (2-column CSV):
    PC_Timestamp, Raw_CSI_String
    where Raw_CSI_String is the ESP-IDF CSI log line.

Output (.npz per trial):
    pc_timestamps       : (N,)   str   — wall-clock time from collection PC
    pc_time_s           : (N,)   float — seconds since first packet (for plotting)
    esp_timestamps_us   : (N,)   int   — ESP internal µs since boot
    rssi                : (N,)   int   — received signal strength, dBm
    noise_floor         : (N,)   int   — noise floor, dBm
    snr            
              : (N,)   float — rssi - noise_floor, dB
    mac                 : (N,)   str   — source MAC (which node/AP)
    amplitude           : (N, S) float — |CSI| per active subcarrier
    phase               : (N, S) float — angle(CSI) per active subcarrier, radians
    csi_real            : (N, S) int   — raw real part (kept for reproducibility)
    csi_imag            : (N, S) int   — raw imag part
    active_subcarrier_indices : (S,) int — which of the 128 pairs are non-null

    where N = number of packets, S = number of active subcarriers (typically 112)

Usage:
    python3 packet_parser.py --input walk1.csv --output walk1.npz
    python3 packet_parser.py --input walk1.csv --output walk1.npz --mac 1C:61:B4:93:EF:FA
    python3 packet_parser.py --input walk1.csv --output walk1.npz --csv-also
"""

import argparse
import csv
import re
import datetime
import numpy as np
from pathlib import Path


# ── Field positions in the comma-split ESP-IDF CSI string ──────────────────
# Format: CSI_DATA,STA,<MAC>,<RSSI>,<rate>,<sig_mode>,<mcs>,<cwb>,
#         <smoothing>,<not_sounding>,<aggregation>,<stbc>,<fec_coding>,
#         <sgi>,<noise_floor>,<ampdu_cnt>,<channel>,<secondary_channel>,
#         <esp_timestamp_us>,<ant>,<sig_len>,<rx_state>,<0>,<rx_time_s>,
#         <csi_len>,[int16 int16 ...]
_F_MAC         = 2
_F_RSSI        = 3
_F_NOISE_FLOOR = 14
_F_ESP_TS      = 18   # µs since ESP boot
_F_CSI_LEN     = 24   # number of int16 values in the bracket
_F_CSI_BRACKET = 25   # start of "[..." field

# The first 2 complex pairs in the ESP32 CSI buffer are embedded firmware
# metadata (constant across all packets). Strip them before computing
# amplitude/phase.
_METADATA_PAIRS = 2


def _parse_csi_string(raw: str):
    """
    Parse one Raw_CSI_String into a dict of scalar fields + int16 array.
    Returns None if the string is malformed or not a CSI_DATA line.
    """
    if not raw.startswith("CSI_DATA"):
        return None

    parts = raw.split(",")
    if len(parts) < _F_CSI_BRACKET + 1:
        return None

    try:
        mac         = parts[_F_MAC]
        rssi        = int(parts[_F_RSSI])
        noise_floor = int(parts[_F_NOISE_FLOOR])
        esp_ts_us   = int(parts[_F_ESP_TS])
        csi_len     = int(parts[_F_CSI_LEN])
    except (ValueError, IndexError):
        return None

    # Everything from the bracket field onwards is the CSI array
    csi_raw = ",".join(parts[_F_CSI_BRACKET:])
    int16_vals = np.array(re.findall(r"-?\d+", csi_raw), dtype=np.int16)

    if len(int16_vals) != csi_len:
        return None  # corrupted / truncated packet

    return {
        "mac":         mac,
        "rssi":        rssi,
        "noise_floor": noise_floor,
        "esp_ts_us":   esp_ts_us,
        "int16_vals":  int16_vals,
    }


def _detect_active_subcarriers(int16_vals: np.ndarray,
                                metadata_pairs: int = _METADATA_PAIRS) -> np.ndarray:
    """
    Given a flat int16 array, return the indices of complex pairs (starting
    after the metadata pairs) where at least one of real/imag is nonzero.
    This identifies the active (non-null) subcarrier positions.
    """
    n_pairs  = len(int16_vals) // 2
    real_arr = int16_vals[0::2]
    imag_arr = int16_vals[1::2]

    active = []
    for i in range(metadata_pairs, n_pairs):
        if real_arr[i] != 0 or imag_arr[i] != 0:
            active.append(i)
    return np.array(active, dtype=np.int32)


def parse_file(csv_path: str,
               target_mac: str = None,
               min_packet_rate_hz: float = 2.0) -> dict:
    """
    Parse a single trial CSV file.

    Parameters
    ----------
    csv_path        : path to the 2-column CSV
    target_mac      : if given, keep only packets from this MAC.
                      if None, auto-select the MAC with the most packets.
    min_packet_rate_hz : warn if the detected rate is below this threshold.

    Returns
    -------
    dict with arrays ready for np.savez_compressed, plus metadata fields.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(csv_path)

    # ── Read raw rows ────────────────────────────────────────────────────────
    raw_packets = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pc_ts  = row.get("PC_Timestamp", "").strip()
            raw    = row.get("Raw_CSI_String", "").strip()
            parsed = _parse_csi_string(raw)
            if parsed is not None:
                parsed["pc_timestamp"] = pc_ts
                raw_packets.append(parsed)

    if len(raw_packets) == 0:
        raise ValueError(f"No valid CSI packets found in {csv_path}")

    # ── Select MAC ───────────────────────────────────────────────────────────
    mac_counts = {}
    for p in raw_packets:
        mac_counts[p["mac"]] = mac_counts.get(p["mac"], 0) + 1

    if target_mac is None:
        target_mac = max(mac_counts, key=mac_counts.get)
        if len(mac_counts) > 1:
            print(f"  [parser] Multiple MACs found: {mac_counts}")
            print(f"  [parser] Auto-selected: {target_mac} ({mac_counts[target_mac]} packets)")
    else:
        if target_mac not in mac_counts:
            raise ValueError(f"MAC {target_mac} not found. Available: {list(mac_counts.keys())}")

    packets = [p for p in raw_packets if p["mac"] == target_mac]

    # ── Filter to consistent CSI length (drop legacy-mode fallback frames) ──
    len_counts = {}
    for p in packets:
        l = len(p["int16_vals"])
        len_counts[l] = len_counts.get(l, 0) + 1
    dominant_len = max(len_counts, key=len_counts.get)
    if len(len_counts) > 1:
        dropped = sum(v for k, v in len_counts.items() if k != dominant_len)
        print(f"  [parser] CSI lengths found: {len_counts} — keeping len={dominant_len}, "
              f"dropping {dropped} legacy/fallback packets")
    packets = [p for p in packets if len(p["int16_vals"]) == dominant_len]

    N = len(packets)

    # ── Parse PC timestamps ──────────────────────────────────────────────────
    fmt = "%Y-%m-%d %H:%M:%S.%f"
    try:
        dt_objects = [datetime.datetime.strptime(p["pc_timestamp"], fmt) for p in packets]
    except ValueError:
        fmt = "%Y-%m-%d %H:%M:%S"
        dt_objects = [datetime.datetime.strptime(p["pc_timestamp"], fmt) for p in packets]

    t0 = dt_objects[0]
    pc_time_s = np.array([(t - t0).total_seconds() for t in dt_objects], dtype=np.float64)
    pc_timestamps = np.array([p["pc_timestamp"] for p in packets], dtype=object)

    # Estimate packet rate
    duration = pc_time_s[-1] - pc_time_s[0]
    if duration > 0:
        rate_hz = (N - 1) / duration
        if rate_hz < min_packet_rate_hz:
            print(f"  [parser] WARNING: packet rate {rate_hz:.1f} Hz below minimum {min_packet_rate_hz} Hz")
        else:
            print(f"  [parser] Packet rate: {rate_hz:.1f} Hz over {duration:.2f}s ({N} packets)")
    else:
        print(f"  [parser] WARNING: zero duration — single packet or timestamp issue")

    # ── Detect active subcarriers from first packet ──────────────────────────
    active_idx = _detect_active_subcarriers(packets[0]["int16_vals"])
    S = len(active_idx)
    print(f"  [parser] Active subcarriers: {S} (of {len(packets[0]['int16_vals'])//2} total pairs)")

    # ── Build output arrays ──────────────────────────────────────────────────
    rssi_arr        = np.array([p["rssi"]        for p in packets], dtype=np.int16)
    noise_floor_arr = np.array([p["noise_floor"] for p in packets], dtype=np.int16)
    esp_ts_arr      = np.array([p["esp_ts_us"]   for p in packets], dtype=np.int64)
    snr_arr         = (rssi_arr - noise_floor_arr).astype(np.float32)
    mac_arr         = np.array([target_mac] * N, dtype=object)

    csi_real = np.zeros((N, S), dtype=np.int16)
    csi_imag = np.zeros((N, S), dtype=np.int16)

    for i, p in enumerate(packets):
        v = p["int16_vals"]
        real_all = v[0::2]
        imag_all = v[1::2]
        csi_real[i] = real_all[active_idx]
        csi_imag[i] = imag_all[active_idx]

    amplitude = np.sqrt(csi_real.astype(np.float32)**2 +
                        csi_imag.astype(np.float32)**2)
    phase = np.arctan2(csi_imag.astype(np.float32),
                       csi_real.astype(np.float32))

    return {
        # ── Per-packet scalars ──────────────────────────────────────────────
        "pc_timestamps":            pc_timestamps,
        "pc_time_s":                pc_time_s,
        "esp_timestamps_us":        esp_ts_arr,
        "rssi":                     rssi_arr,
        "noise_floor":              noise_floor_arr,
        "snr":                      snr_arr,
        "mac":                      mac_arr,

        # ── Per-packet × per-subcarrier ─────────────────────────────────────
        "amplitude":                amplitude,        # (N, S) float32
        "phase":                    phase,            # (N, S) float32, radians
        "csi_real":                 csi_real,         # (N, S) int16
        "csi_imag":                 csi_imag,         # (N, S) int16

        # ── Metadata ────────────────────────────────────────────────────────
        "active_subcarrier_indices": active_idx,      # (S,) int32
        "_source_file":             str(path.name),
        "_target_mac":              target_mac,
        "_n_packets":               N,
        "_n_subcarriers":           S,
        "_duration_s":              duration,
        "_packet_rate_hz":          float((N - 1) / duration) if duration > 0 else 0.0,
    }


def save_npz(data: dict, output_path: str):
    """Save parsed data to a compressed .npz file."""
    # Separate numpy arrays from metadata strings
    arrays   = {k: v for k, v in data.items() if isinstance(v, np.ndarray)}
    scalars  = {k: v for k, v in data.items() if not isinstance(v, np.ndarray)}

    # Store scalar metadata as 0-d arrays so they survive npz round-trip
    for k, v in scalars.items():
        arrays[k] = np.array(v)

    np.savez_compressed(output_path, **arrays)
    print(f"  [parser] Saved → {output_path}")


def save_csv(data: dict, output_path: str):
    """
    Save a human-readable CSV: one row per packet, columns are
    pc_timestamp, pc_time_s, esp_timestamp_us, rssi, noise_floor, snr,
    amp_0 .. amp_{S-1}, phase_0 .. phase_{S-1}
    """
    S = data["_n_subcarriers"]
    amp_cols   = [f"amp_{i}"   for i in range(S)]
    phase_cols = [f"phase_{i}" for i in range(S)]

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = (["pc_timestamp", "pc_time_s", "esp_timestamp_us",
                   "rssi", "noise_floor", "snr"] +
                  amp_cols + phase_cols)
        writer.writerow(header)
        N = data["_n_packets"]
        for i in range(N):
            row = ([data["pc_timestamps"][i],
                    f"{data['pc_time_s'][i]:.6f}",
                    int(data["esp_timestamps_us"][i]),
                    int(data["rssi"][i]),
                    int(data["noise_floor"][i]),
                    f"{data['snr'][i]:.1f}"] +
                   [f"{v:.4f}" for v in data["amplitude"][i]] +
                   [f"{v:.6f}" for v in data["phase"][i]])
            writer.writerow(row)

    print(f"  [parser] CSV saved → {output_path}")


def load_npz(npz_path: str) -> dict:
    """Load a previously saved .npz back into a plain dict."""
    loaded = np.load(npz_path, allow_pickle=True)
    return {k: loaded[k] for k in loaded.files}


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Parse ESP32 CSI CSV trial files.")
    ap.add_argument("--input",    required=True,  help="Input CSV file")
    ap.add_argument("--output",   required=True,  help="Output .npz file")
    ap.add_argument("--mac",      default=None,   help="Target MAC (auto-detect if omitted)")
    ap.add_argument("--csv-also", action="store_true",
                    help="Also write a human-readable CSV alongside the .npz")
    args = ap.parse_args()

    print(f"\nParsing: {args.input}")
    data = parse_file(args.input, target_mac=args.mac)
    save_npz(data, args.output)

    if args.csv_also:
        csv_out = str(Path(args.output).with_suffix(".csv"))
        save_csv(data, csv_out)

    # Quick sanity print
    print(f"\n  Shape summary:")
    print(f"    amplitude : {data['amplitude'].shape}")
    print(f"    phase     : {data['phase'].shape}")
    print(f"    rssi range: {data['rssi'].min()} to {data['rssi'].max()} dBm")
    print(f"    snr range : {data['snr'].min():.1f} to {data['snr'].max():.1f} dB")
