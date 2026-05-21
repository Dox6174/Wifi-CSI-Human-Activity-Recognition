"""
sync_session.py
===============
Processes continuous CSI sessions (e1, e2) from two ESP32 nodes.

Each session is ~150s of continuous data recorded simultaneously on two laptops.

Processing pipeline per session:
    1. Load Node A and Node B .npz files
    2. Skip first 10s from EACH node using timestamps (NOT row removal)
    3. Find common overlap window after skipping
    4. Extract exactly 100s from common start
    5. Resample both nodes to uniform 10 Hz grid
    6. Split into 10 trials of exactly 10s each
    7. Save as individual trial npz files

Output (20 trials per node total):
    processed_data/
        e1_trial01_nodeA.npz  ...  e1_trial10_nodeA.npz
        e1_trial01_nodeB.npz  ...  e1_trial10_nodeB.npz
        e2_trial01_nodeA.npz  ...  e2_trial10_nodeA.npz
        e2_trial01_nodeB.npz  ...  e2_trial10_nodeB.npz
        session_report.csv

Usage:
    python sync_session.py \
        --e1_nodeA path/to/e1/nodeA.npz \
        --e1_nodeB path/to/e1/nodeB.npz \
        --e2_nodeA path/to/e2/nodeA.npz \
        --e2_nodeB path/to/e2/nodeB.npz

    # Custom output folder:
    python sync_session.py ... --out my_output

    # Override parameters:
    python sync_session.py ... --skip 10.0 --keep 100.0 --trials 10
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
SKIP_SEC   = 10.0   # seconds to skip from each node's own start (time-based)
KEEP_SEC   = 100.0  # seconds of clean data to extract after sync
N_TRIALS   = 10     # number of trials to split KEEP_SEC into
TARGET_HZ  = 10.0   # resample both nodes to this uniform rate

# Derived
TRIAL_SEC  = KEEP_SEC / N_TRIALS   # = 10.0s per trial

CSI_FIELDS    = ['amplitude', 'phase', 'csi_real', 'csi_imag']
SCALAR_FIELDS = ['rssi', 'noise_floor', 'snr']
# ─────────────────────────────────────────────────────────────────────────────


def load_npz(path: Path) -> dict:
    """Load an npz file into a plain dict."""
    raw = np.load(path, allow_pickle=True)
    return {k: raw[k] for k in raw.files}


def get_unix_times(data: dict) -> np.ndarray:
    """
    Convert pc_timestamps to float64 UNIX seconds.
    Handles datetime64[ns/us/ms] across pandas versions.
    """
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
    """
    Linear interpolation of (N,) or (N, S) array onto t_grid.
    Pure timestamp-based — no row indexing.
    """
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


def sync_session(data_A: dict,
                 data_B: dict,
                 skip_sec: float,
                 keep_sec: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Synchronise one session and return aligned data arrays on a uniform grid.

    Pipeline:
        1. Skip skip_sec from each node's OWN start — time-based only,
           never by row index.
        2. Find common overlap after skipping.
        3. Extract keep_sec from common start.
        4. Resample both nodes onto a single uniform TARGET_HZ grid.

    Returns
    -------
    t_grid  : (M,) float64  — relative time in seconds [0, keep_sec]
    out_A   : dict of resampled arrays for Node A
    out_B   : dict of resampled arrays for Node B
    report  : dict of diagnostics
    """
    t_A = get_unix_times(data_A)
    t_B = get_unix_times(data_B)

    clock_offset_ms = abs(t_A[0] - t_B[0]) * 1000

    # ── 1. Time-based skip per node independently ─────────────────────────
    # We add skip_sec to each node's OWN first timestamp.
    # This is a pure timestamp operation — no row slicing (t_A[N:]) used.
    t_A_valid = t_A[0] + skip_sec
    t_B_valid = t_B[0] + skip_sec

    # ── 2. Common overlap AFTER time-based skip ───────────────────────────
    t_common_start = max(t_A_valid, t_B_valid)
    t_common_end   = min(t_A[-1], t_B[-1])
    overlap_s      = t_common_end - t_common_start

    if overlap_s < keep_sec:
        # Try without skip to give a useful error message
        raw_overlap = min(t_A[-1], t_B[-1]) - max(t_A[0], t_B[0])
        raise ValueError(
            f"After skipping {skip_sec}s per node, overlap is only "
            f"{overlap_s:.2f}s — need {keep_sec}s.\n"
            f"  Raw overlap (no skip): {raw_overlap:.2f}s\n"
            f"  Node A duration: {t_A[-1] - t_A[0]:.2f}s  "
            f"  Node B duration: {t_B[-1] - t_B[0]:.2f}s\n"
            f"  Check that the session files are long enough "
            f"(need >{skip_sec + keep_sec}s of overlap)."
        )

    # ── 3. Extraction window ──────────────────────────────────────────────
    t_start = t_common_start
    t_end   = t_start + keep_sec

    # ── 4. Uniform target grid ────────────────────────────────────────────
    # Total samples for the full keep_sec window at TARGET_HZ
    n_total = int(keep_sec * TARGET_HZ)           # = 1000 samples for 100s
    t_grid  = np.linspace(t_start, t_end, n_total)

    # ── 5. Resample each node ─────────────────────────────────────────────
    def resample_node(data, t_raw, t_valid_start):
        out = {}

        # Time-based mask — three timestamp comparisons, no row indexing:
        #   (a) t >= node's own start + skip_sec
        #   (b) t >= extraction window start - 1s buffer
        #   (c) t <= extraction window end   + 1s buffer
        mask = (
            (t_raw >= t_valid_start) &
            (t_raw >= t_start - 1.0) &
            (t_raw <= t_end   + 1.0)
        )
        t_m = t_raw[mask]

        if t_m.size == 0:
            raise ValueError(
                f"Empty mask after time-based windowing.\n"
                f"  t_raw=[{t_raw[0]:.3f}, {t_raw[-1]:.3f}]\n"
                f"  valid_start={t_valid_start:.3f}\n"
                f"  window=[{t_start:.3f}, {t_end:.3f}]"
            )

        for field in CSI_FIELDS:
            if field in data:
                out[field] = resample_field(data[field][mask], t_m, t_grid)

        for field in SCALAR_FIELDS:
            if field in data:
                out[field] = resample_field(
                    data[field][mask].astype(np.float32), t_m, t_grid
                ).astype(data[field].dtype)

        # Carry metadata through unchanged
        for field in ['active_subcarrier_indices', '_target_mac',
                      '_source_file', '_n_subcarriers']:
            if field in data:
                out[field] = data[field]

        return out

    out_A = resample_node(data_A, t_A, t_A_valid)
    out_B = resample_node(data_B, t_B, t_B_valid)

    # Relative time axis starting from 0
    t_rel = (t_grid - t_grid[0]).astype(np.float64)

    report = {
        'clock_offset_ms' : round(clock_offset_ms, 2),
        'overlap_s'       : round(overlap_s, 3),
        'skip_sec'        : skip_sec,
        'keep_sec'        : keep_sec,
        'extracted_start' : round(t_start, 3),
        'extracted_end'   : round(t_end, 3),
        'total_samples'   : n_total,
        'rate_A_hz'       : round(float(data_A['_packet_rate_hz']), 3),
        'rate_B_hz'       : round(float(data_B['_packet_rate_hz']), 3),
    }
    return t_rel, out_A, out_B, report


def split_into_trials(t_rel: np.ndarray,
                      out_A: dict,
                      out_B: dict,
                      n_trials: int,
                      trial_sec: float,
                      session_name: str,
                      out_dir: Path,
                      session_report: dict,
                      clock_offset_ms: float) -> list[dict]:
    """
    Split the synchronized 100s block into n_trials of trial_sec each.
    Saves nodeA and nodeB npz for every trial.
    Returns list of per-trial report rows.
    """
    samples_per_trial = int(trial_sec * TARGET_HZ)   # = 100 samples per trial
    report_rows = []

    print(f"\n  Splitting into {n_trials} trials of {trial_sec}s each "
          f"({samples_per_trial} samples @ {TARGET_HZ}Hz)")
    print(f"\n  {'Trial':<18}  {'Time window':>20}  {'Samples':>8}")
    print("  " + "─" * 52)

    for i in range(n_trials):
        trial_num  = i + 1
        trial_name = f"{session_name}_trial{trial_num:02d}"

        # Sample indices for this trial — pure index arithmetic on uniform grid
        idx_start = i * samples_per_trial
        idx_end   = idx_start + samples_per_trial

        t_trial   = t_rel[idx_start:idx_end]
        t_start_s = round(float(t_trial[0]),  3)
        t_end_s   = round(float(t_trial[-1]), 3)

        def save_trial_node(out_full, node_label):
            trial_out = {}

            # Slice each field by sample index on the uniform grid
            for field in CSI_FIELDS + SCALAR_FIELDS:
                if field in out_full:
                    trial_out[field] = out_full[field][idx_start:idx_end]

            for field in ['active_subcarrier_indices', '_target_mac',
                          '_source_file', '_n_subcarriers']:
                if field in out_full:
                    trial_out[field] = out_full[field]

            # Relative time restarting from 0 for each trial
            trial_out['pc_time_s']        = (t_trial - t_trial[0]).astype(np.float64)
            trial_out['_n_packets']       = np.int32(samples_per_trial)
            trial_out['_duration_s']      = np.float64(trial_sec)
            trial_out['_packet_rate_hz']  = np.float64(TARGET_HZ)
            trial_out['_sync_method']     = np.str_('pc_timestamp_ntp_session')
            trial_out['_clock_offset_ms'] = np.float64(clock_offset_ms)
            trial_out['_session']         = np.str_(session_name)
            trial_out['_trial_number']    = np.int32(trial_num)
            trial_out['_trial_t_start_s'] = np.float64(t_start_s)
            trial_out['_trial_t_end_s']   = np.float64(t_end_s)

            save_path = out_dir / f"{trial_name}_{node_label}.npz"
            np.savez_compressed(save_path, **trial_out)

        save_trial_node(out_A, 'nodeA')
        save_trial_node(out_B, 'nodeB')

        print(f"  {trial_name:<18}  "
              f"[{t_start_s:6.1f}s – {t_end_s:6.1f}s]  "
              f"{samples_per_trial:>8}")

        report_rows.append({
            'session'         : session_name,
            'trial'           : trial_name,
            'trial_number'    : trial_num,
            't_start_s'       : t_start_s,
            't_end_s'         : t_end_s,
            'samples'         : samples_per_trial,
            'clock_offset_ms' : session_report['clock_offset_ms'],
            'overlap_s'       : session_report['overlap_s'],
            'rate_A_hz'       : session_report['rate_A_hz'],
            'rate_B_hz'       : session_report['rate_B_hz'],
            'file_A'          : f"{trial_name}_nodeA.npz",
            'file_B'          : f"{trial_name}_nodeB.npz",
        })

    return report_rows


def process_session(session_name: str,
                    path_A: Path,
                    path_B: Path,
                    out_dir: Path,
                    skip_sec: float,
                    keep_sec: float,
                    n_trials: int) -> list[dict]:
    """
    Full pipeline for one session (e1 or e2).
    Returns list of per-trial report rows.
    """
    trial_sec = keep_sec / n_trials

    print(f"\n{'='*66}")
    print(f"  Session  : {session_name.upper()}")
    print(f"  Skip     : {skip_sec}s per node (time-based, not row-based)")
    print(f"  Keep     : {keep_sec}s → {n_trials} trials × {trial_sec}s")
    print(f"  Rate     : {TARGET_HZ} Hz")
    print(f"  Node A   : {path_A}")
    print(f"  Node B   : {path_B}")
    print(f"  Output   : {out_dir}")
    print(f"{'='*66}")

    if not path_A.exists():
        raise FileNotFoundError(f"Node A file not found: {path_A}")
    if not path_B.exists():
        raise FileNotFoundError(f"Node B file not found: {path_B}")

    print(f"\n  Loading Node A ... ", end='', flush=True)
    data_A = load_npz(path_A)
    t_A    = get_unix_times(data_A)
    print(f"done  ({len(t_A)} packets, "
          f"{t_A[-1]-t_A[0]:.1f}s, "
          f"rate={data_A.get('_packet_rate_hz', '?')} Hz)")

    print(f"  Loading Node B ... ", end='', flush=True)
    data_B = load_npz(path_B)
    t_B    = get_unix_times(data_B)
    print(f"done  ({len(t_B)} packets, "
          f"{t_B[-1]-t_B[0]:.1f}s, "
          f"rate={data_B.get('_packet_rate_hz', '?')} Hz)")

    print(f"\n  Clock offset between nodes : "
          f"{abs(t_A[0]-t_B[0])*1000:.1f} ms")

    # ── Sync and resample the full 100s block ─────────────────────────────
    print(f"\n  Synchronising and resampling ...")
    t_rel, out_A, out_B, report = sync_session(
        data_A, data_B, skip_sec, keep_sec
    )
    print(f"  ✓ Overlap after skip : {report['overlap_s']}s")
    print(f"  ✓ Extracted window   : "
          f"{report['extracted_start']:.2f}s → {report['extracted_end']:.2f}s "
          f"(unix timestamps)")
    print(f"  ✓ Total samples      : {report['total_samples']} "
          f"({keep_sec}s × {TARGET_HZ}Hz)")

    # ── Split into trials ─────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    report_rows = split_into_trials(
        t_rel, out_A, out_B,
        n_trials      = n_trials,
        trial_sec     = trial_sec,
        session_name  = session_name,
        out_dir       = out_dir,
        session_report= report,
        clock_offset_ms = report['clock_offset_ms'],
    )

    print(f"\n  ✓ {len(report_rows)} trials saved to {out_dir}/")
    return report_rows


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Session file paths
    parser.add_argument('--e1_nodeA', required=True,
                        help='Path to e1 Node A .npz file')
    parser.add_argument('--e1_nodeB', required=True,
                        help='Path to e1 Node B .npz file')
    parser.add_argument('--e2_nodeA', required=True,
                        help='Path to e2 Node A .npz file')
    parser.add_argument('--e2_nodeB', required=True,
                        help='Path to e2 Node B .npz file')

    # Output
    parser.add_argument('--out', default='processed_data',
                        help='Output folder (default: processed_data/)')

    # Processing parameters
    parser.add_argument('--skip',   type=float, default=SKIP_SEC,
                        help=f'Seconds to skip from each node start '
                             f'(default: {SKIP_SEC}s)')
    parser.add_argument('--keep',   type=float, default=KEEP_SEC,
                        help=f'Seconds to extract after sync '
                             f'(default: {KEEP_SEC}s)')
    parser.add_argument('--trials', type=int,   default=N_TRIALS,
                        help=f'Number of trials to split into '
                             f'(default: {N_TRIALS})')

    args = parser.parse_args()

    out_root  = Path(args.out)
    trial_sec = args.keep / args.trials

    print(f"\n{'='*66}")
    print(f"  CSI Session Sync & Split")
    print(f"  Skip per node : {args.skip}s  (time-based)")
    print(f"  Keep          : {args.keep}s")
    print(f"  Trials        : {args.trials} × {trial_sec}s each")
    print(f"  Output        : {out_root}/")
    print(f"{'='*66}")

    all_reports = []
    sessions = [
        ('e1', Path(args.e1_nodeA), Path(args.e1_nodeB)),
        ('e2', Path(args.e2_nodeA), Path(args.e2_nodeB)),
    ]

    for session_name, path_A, path_B in sessions:
        try:
            rows = process_session(
                session_name = session_name,
                path_A       = path_A,
                path_B       = path_B,
                out_dir      = out_root,
                skip_sec     = args.skip,
                keep_sec     = args.keep,
                n_trials     = args.trials,
            )
            all_reports.extend(rows)
        except Exception as e:
            print(f"\n  [ERROR] Session {session_name} failed: {e}")
            sys.exit(1)

    # ── Save master report ────────────────────────────────────────────────
    report_path = out_root / 'session_report.csv'
    pd.DataFrame(all_reports).to_csv(report_path, index=False)

    print(f"\n{'='*66}")
    print(f"  All sessions done.")
    print(f"  Total trials saved : {len(all_reports)} "
          f"({len(all_reports)//2} per node × 2 nodes)")
    print(f"\n  Output structure:")
    print(f"    {out_root}/")
    print(f"      e1_trial01_nodeA.npz  ...  e1_trial10_nodeA.npz")
    print(f"      e1_trial01_nodeB.npz  ...  e1_trial10_nodeB.npz")
    print(f"      e2_trial01_nodeA.npz  ...  e2_trial10_nodeA.npz")
    print(f"      e2_trial01_nodeB.npz  ...  e2_trial10_nodeB.npz")
    print(f"      session_report.csv")
    print(f"{'='*66}\n")


if __name__ == '__main__':
    main()