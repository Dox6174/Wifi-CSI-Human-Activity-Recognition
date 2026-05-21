"""
sync_all.py
===========
Processes ALL activities (walk, sit, stand, hand) in one run.

File naming per activity:
    walk  → walk1.npz, walk2.npz, ...
    sit   → sit1.npz,  sit2.npz,  ...
    stand → st1.npz,   st2.npz,   ...
    hand  → h1.npz,    h2.npz,    ...

Folder structure expected:
    nodeA/
        walk/    walk1.npz ... walk40.npz
        sit/     sit1.npz  ... sit40.npz
        stand/   st1.npz   ... st40.npz
        hand/    h1.npz    ... h40.npz
    nodeB/
        (same structure)

Output:
    processed_data/
        walk/   walk1_nodeA.npz  walk1_nodeB.npz ... sync_report.csv
        sit/    sit1_nodeA.npz   sit1_nodeB.npz  ... sync_report.csv
        stand/  st1_nodeA.npz    st1_nodeB.npz   ... sync_report.csv
        hand/   h1_nodeA.npz     h1_nodeB.npz    ... sync_report.csv

Usage:
    python sync_all.py --nodeA path/to/nodeA --nodeB path/to/nodeB

    # Custom output:
    python sync_all.py --nodeA ... --nodeB ... --out my_output

    # Override skip for specific activity:
    python sync_all.py --nodeA ... --nodeB ... --skip_sit 3.0
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
KEEP_SEC  = 10.0
TARGET_HZ = 10.0

# Skip per activity (time-based, NOT row-based)
ACTIVITY_SKIP = {
    'walk'  : 2.0,
    'sit'   : 4.0,
    'stand' : 4.0,
    'hand'  : 4.0,
}

# Actual filename prefix per activity
ACTIVITY_PREFIX = {
    'walk'  : 'walk',
    'sit'   : 'sit',
    'stand' : 'st',
    'hand'  : 'h',
}

CSI_FIELDS    = ['amplitude', 'phase', 'csi_real', 'csi_imag']
SCALAR_FIELDS = ['rssi', 'noise_floor', 'snr']
# ─────────────────────────────────────────────────────────────────────────────


def extract_trial_key(filename: str, activity: str) -> str | None:
    """
    Strip leading numeric/timestamp prefix and return trial key
    if it matches the activity's actual filename prefix.

    activity='walk'  → matches walk1, walk2  ...
    activity='sit'   → matches sit1,  sit2   ...
    activity='stand' → matches st1,   st2    ...
    activity='hand'  → matches h1,    h2     ...
    """
    prefix = ACTIVITY_PREFIX[activity]
    stem   = Path(filename).stem
    key    = re.sub(r'^[^a-zA-Z]+', '', stem)   # strip leading non-alpha
    if not re.match(rf'{re.escape(prefix)}\d+', key, re.IGNORECASE):
        return None
    return key.lower()


def load_npz(path: Path) -> dict:
    raw = np.load(path, allow_pickle=True)
    return {k: raw[k] for k in raw.files}


def get_unix_times(data: dict) -> np.ndarray:
    """Convert pc_timestamps to float64 UNIX seconds."""
    ts      = pd.to_datetime(data['pc_timestamps'])
    int64   = ts.astype('int64').values
    dtype_s = str(ts.dtype)

    if   '[ns]' in dtype_s: return int64 / 1e9
    elif '[us]' in dtype_s: return int64 / 1e6
    elif '[ms]' in dtype_s: return int64 / 1e3
    else:
        val = abs(int64[0])
        if   val > 1e17: return int64 / 1e9
        elif val > 1e14: return int64 / 1e6
        elif val > 1e11: return int64 / 1e3
        else:            return int64.astype(float)


def resample_field(values: np.ndarray,
                   t_source: np.ndarray,
                   t_grid: np.ndarray) -> np.ndarray:
    """Linear interpolation of (N,) or (N, S) array onto t_grid."""
    if values.ndim == 1:
        f = interp1d(t_source, values, kind='linear',
                     bounds_error=False, fill_value='extrapolate')
        return f(t_grid).astype(values.dtype)

    n_out, n_sub = len(t_grid), values.shape[1]
    out = np.empty((n_out, n_sub), dtype=np.float32)
    for s in range(n_sub):
        f = interp1d(t_source, values[:, s], kind='linear',
                     bounds_error=False, fill_value='extrapolate')
        out[:, s] = f(t_grid)
    return out


def sync_and_trim(data_A: dict, data_B: dict,
                  skip_sec: float) -> tuple[dict, dict, dict]:
    """
    Skip (time-based) → Common Overlap → Extract 10s → Resample.

    Skip is purely TIME-BASED:
        t >= node_own_start + skip_sec
    Never done by row index (no t_A[20:] or t_A[40:] anywhere).
    """
    t_A = get_unix_times(data_A)
    t_B = get_unix_times(data_B)

    clock_offset_ms = abs(t_A[0] - t_B[0]) * 1000

    # ── 1. Time-based skip per node independently ─────────────────────────
    t_A_valid_start = t_A[0] + skip_sec
    t_B_valid_start = t_B[0] + skip_sec

    # ── 2. Common overlap AFTER time-based skip ───────────────────────────
    t_common_start = max(t_A_valid_start, t_B_valid_start)
    t_common_end   = min(t_A[-1], t_B[-1])
    overlap_s      = t_common_end - t_common_start

    # ── 3. Graceful skip reduction if overlap is tight ────────────────────
    if overlap_s < KEEP_SEC:
        t_raw_start = max(t_A[0], t_B[0])
        t_raw_end   = min(t_A[-1], t_B[-1])
        raw_overlap = t_raw_end - t_raw_start

        if raw_overlap < KEEP_SEC:
            raise ValueError(
                f"Overlap only {raw_overlap:.2f}s even without skip — "
                f"need at least {KEEP_SEC}s. "
                f"Check both files are from the same trial."
            )

        actual_skip     = min(skip_sec, raw_overlap - KEEP_SEC)
        t_common_start  = t_raw_start + actual_skip
        t_common_end    = t_raw_end
        overlap_s       = t_common_end - t_common_start
        t_A_valid_start = t_A[0] + actual_skip
        t_B_valid_start = t_B[0] + actual_skip
    else:
        actual_skip = skip_sec

    # ── 4. Extraction window ──────────────────────────────────────────────
    t_start = t_common_start
    t_end   = t_start + KEEP_SEC

    # ── 5. Uniform target grid ────────────────────────────────────────────
    n_out  = int(KEEP_SEC * TARGET_HZ)    # = 100 samples
    t_grid = np.linspace(t_start, t_end, n_out)

    # ── 6. Resample each node ─────────────────────────────────────────────
    def process_node(data, t_raw, t_valid_start):
        out = {}

        # Pure timestamp comparisons — no row indexing
        mask = (
            (t_raw >= t_valid_start) &
            (t_raw >= t_start - 1.0) &
            (t_raw <= t_end   + 1.0)
        )
        t_m = t_raw[mask]

        if t_m.size == 0:
            raise ValueError(
                f"Empty mask after time-based windowing. "
                f"t_raw=[{t_raw[0]:.3f}, {t_raw[-1]:.3f}] "
                f"valid_start={t_valid_start:.3f} "
                f"window=[{t_start:.3f}, {t_end:.3f}]"
            )

        for field in CSI_FIELDS:
            if field in data:
                out[field] = resample_field(data[field][mask], t_m, t_grid)

        for field in SCALAR_FIELDS:
            if field in data:
                out[field] = resample_field(
                    data[field][mask].astype(np.float32), t_m, t_grid
                ).astype(data[field].dtype)

        for field in ['active_subcarrier_indices', '_target_mac',
                      '_source_file', '_n_subcarriers']:
            if field in data:
                out[field] = data[field]

        out['pc_time_s']        = (t_grid - t_grid[0]).astype(np.float64)
        out['_n_packets']       = np.int32(n_out)
        out['_duration_s']      = np.float64(KEEP_SEC)
        out['_packet_rate_hz']  = np.float64(TARGET_HZ)
        out['_sync_method']     = np.str_('pc_timestamp_ntp')
        out['_clock_offset_ms'] = np.float64(clock_offset_ms)
        out['_actual_skip_s']   = np.float64(actual_skip)
        return out

    out_A = process_node(data_A, t_A, t_A_valid_start)
    out_B = process_node(data_B, t_B, t_B_valid_start)

    skip_note = (f"reduced skip {actual_skip:.2f}s (overlap={overlap_s:.2f}s)"
                 if actual_skip < skip_sec else "ok")

    report = {
        'clock_offset_ms': round(clock_offset_ms, 2),
        'overlap_s'      : round(overlap_s, 3),
        'actual_skip_s'  : round(actual_skip, 3),
        'packets_A'      : int(data_A['_n_packets']),
        'packets_B'      : int(data_B['_n_packets']),
        'rate_A_hz'      : round(float(data_A['_packet_rate_hz']), 3),
        'rate_B_hz'      : round(float(data_B['_packet_rate_hz']), 3),
        'status'         : skip_note,
    }
    return out_A, out_B, report


def index_folder(folder: Path, activity: str) -> dict[str, Path]:
    """Return {trial_key: path} for all matching files in folder."""
    index = {}
    for f in sorted(folder.glob('*.npz')):
        key = extract_trial_key(f.name, activity)
        if key is None:
            continue
        if key in index:
            print(f"  [WARN] Duplicate key '{key}' — "
                  f"keeping {index[key].name}, skipping {f.name}")
            continue
        index[key] = f
    return index


def process_activity(activity: str,
                     dir_A: Path,
                     dir_B: Path,
                     out_dir: Path,
                     skip_sec: float):
    """Process all trial pairs for one activity class."""

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*66}")
    print(f"  Activity : {activity.upper()}   "
          f"(prefix: '{ACTIVITY_PREFIX[activity]}')")
    print(f"  Skip     : {skip_sec}s (time-based)   "
          f"Keep : {KEEP_SEC}s   Rate : {TARGET_HZ} Hz")
    print(f"  Node A   : {dir_A}")
    print(f"  Node B   : {dir_B}")
    print(f"  Output   : {out_dir}")
    print(f"{'='*66}")

    if not dir_A.exists():
        print(f"  [SKIP] Node A folder not found: {dir_A}")
        return
    if not dir_B.exists():
        print(f"  [SKIP] Node B folder not found: {dir_B}")
        return

    idx_A = index_folder(dir_A, activity)
    idx_B = index_folder(dir_B, activity)
    print(f"\n  Node A: {len(idx_A)} files — {sorted(idx_A.keys())}")
    print(f"  Node B: {len(idx_B)} files — {sorted(idx_B.keys())}")

    all_keys = sorted(set(idx_A) | set(idx_B))
    paired   = [k for k in all_keys if k in idx_A and k in idx_B]
    only_A   = [k for k in all_keys if k in idx_A and k not in idx_B]
    only_B   = [k for k in all_keys if k in idx_B and k not in idx_A]

    print(f"\n  Matched pairs : {len(paired)}")
    if only_A: print(f"  [WARN] Only in Node A: {only_A}")
    if only_B: print(f"  [WARN] Only in Node B: {only_B}")

    if not paired:
        print("  [ERROR] No matched pairs found. Check filenames.")
        return

    report_rows = []
    ok_count = fail_count = 0

    print(f"\n  {'Trial':<12} {'Offset(ms)':>10} {'Overlap(s)':>10} "
          f"{'Skip(s)':>7} {'Rate A':>8} {'Rate B':>8}  Status")
    print("  " + "─" * 70)

    for key in paired:
        row = {'trial': key,
               'file_A': idx_A[key].name,
               'file_B': idx_B[key].name}
        try:
            dA = load_npz(idx_A[key])
            dB = load_npz(idx_B[key])
            out_A, out_B, report = sync_and_trim(dA, dB, skip_sec)

            np.savez_compressed(out_dir / f"{key}_nodeA.npz", **out_A)
            np.savez_compressed(out_dir / f"{key}_nodeB.npz", **out_B)

            row.update(report)
            ok_count += 1
            status_str = f"✓ {report['status']}"

        except Exception as e:
            row['status'] = f'FAILED: {e}'
            fail_count += 1
            status_str = f"✗ FAILED: {e}"

        report_rows.append(row)
        print(f"  {key:<12} "
              f"{str(row.get('clock_offset_ms','?')):>10} "
              f"{str(row.get('overlap_s','?')):>10} "
              f"{str(row.get('actual_skip_s','?')):>7} "
              f"{str(row.get('rate_A_hz','?')):>8} "
              f"{str(row.get('rate_B_hz','?')):>8}  "
              f"{status_str}")

    report_path = out_dir / 'sync_report.csv'
    pd.DataFrame(report_rows).to_csv(report_path, index=False)

    print(f"\n  ✓ {ok_count} succeeded   ✗ {fail_count} failed")
    print(f"  Report → {report_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--nodeA', required=True,
                        help='Root folder of Node A (contains walk/, sit/, stand/, hand/)')
    parser.add_argument('--nodeB', required=True,
                        help='Root folder of Node B (contains walk/, sit/, stand/, hand/)')
    parser.add_argument('--out', default='processed_data',
                        help='Root output folder (default: processed_data/)')

    # Optional per-activity skip overrides
    parser.add_argument('--skip_walk',  type=float, default=None,
                        help='Override skip for walk  (default: 2.0s)')
    parser.add_argument('--skip_sit',   type=float, default=None,
                        help='Override skip for sit   (default: 4.0s)')
    parser.add_argument('--skip_stand', type=float, default=None,
                        help='Override skip for stand (default: 4.0s)')
    parser.add_argument('--skip_hand',  type=float, default=None,
                        help='Override skip for hand  (default: 4.0s)')

    args = parser.parse_args()

    root_A   = Path(args.nodeA)
    root_B   = Path(args.nodeB)
    out_root = Path(args.out)

    skips = dict(ACTIVITY_SKIP)
    if args.skip_walk  is not None: skips['walk']  = args.skip_walk
    if args.skip_sit   is not None: skips['sit']   = args.skip_sit
    if args.skip_stand is not None: skips['stand'] = args.skip_stand
    if args.skip_hand  is not None: skips['hand']  = args.skip_hand

    print(f"\nNode A root : {root_A}")
    print(f"Node B root : {root_B}")
    print(f"Output root : {out_root}")
    print(f"Prefixes    : {ACTIVITY_PREFIX}")
    print(f"Skip times  : { {k: f'{v}s' for k, v in skips.items()} }")

    for activity, skip_sec in skips.items():
        process_activity(
            activity = activity,
            dir_A    = root_A / activity,
            dir_B    = root_B / activity,
            out_dir  = out_root / activity,
            skip_sec = skip_sec,
        )

    print(f"\n{'='*66}")
    print(f"  All done. Output structure:")
    print(f"    {out_root}/")
    for activity, prefix in ACTIVITY_PREFIX.items():
        print(f"      {activity}/   "
              f"({prefix}1_nodeA.npz, {prefix}1_nodeB.npz, ...)")
    print(f"{'='*66}\n")


if __name__ == '__main__':
    main()