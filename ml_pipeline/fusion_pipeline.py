#!/usr/bin/env python3
"""
fusion_pipeline_fixed.py

Fixed fusion pipeline for RF-HAR.

Key fix:
- RAW WINDOWS are now the default output, so the resulting `X.npy` matches
  the DANN/CNN trainer expectation: (N, 20, 436) for concat fusion.

Use:
    python fusion_pipeline_fixed.py --data processed_data --out ml_ready
    python fusion_pipeline_fixed.py --data processed_data --out ml_ready --features
"""

from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pywt
from scipy.stats import entropy as scipy_entropy

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
TARGET_HZ = 10.0
WINDOW_SEC = 2.0
STRIDE_SEC = 0.5
MIN_PACKETS = 15

WAVELET = "db4"
WAVELET_LEVEL = 3
HAMPEL_K = 5
HAMPEL_T0 = 2.0

# Center-crop both nodes to this subcarrier width.
TARGET_SUBCARRIERS = 109

WINDOW_SAMPLES = int(WINDOW_SEC * TARGET_HZ)   # 20
STRIDE_SAMPLES = int(STRIDE_SEC * TARGET_HZ)    # 5

ACTIVITY_LABEL = {
    "walk": 0,
    "sit": 1,
    "stand": 2,
    "hand": 3,
    "empty": 4,
}

ACTIVITY_PREFIX = {
    "walk": "walk",
    "sit": "sit",
    "stand": "st",
    "hand": "h",
    "empty": "e",
}

# HT20/legacy pairs that should be skipped.
HT20_SKIP = {
    "hand": {"h13", "h19", "h23", "h31", "h33"},
    "sit": {"sit1", "sit2", "sit22", "sit27", "sit29", "sit31", "sit32", "sit5"},
    "stand": {"st33", "st8"},
}

FIELD_AMPLITUDE = "amplitude"
FIELD_PHASE = "phase"
FIELD_SNR = "snr"

# ---------------------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------------------
def _hampel_1d(x: np.ndarray) -> np.ndarray:
    n = len(x)
    out = x.copy()
    for i in range(n):
        lo = max(0, i - HAMPEL_K)
        hi = min(n, i + HAMPEL_K + 1)
        win = x[lo:hi]
        med = np.median(win)
        mad = np.median(np.abs(win - med))
        scale = 1.4826 * mad
        if scale > 0 and np.abs(x[i] - med) > HAMPEL_T0 * scale:
            out[i] = med
    return out


def hampel_filter(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return _hampel_1d(x)
    out = np.empty_like(x)
    for s in range(x.shape[1]):
        out[:, s] = _hampel_1d(x[:, s])
    return out


def wavelet_denoise(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    for s in range(x.shape[1]):
        col = x[:, s]
        coeffs = pywt.wavedec(col, WAVELET, level=WAVELET_LEVEL)
        thresh = 0.20 * np.max(np.abs(coeffs[-1]))
        if thresh < 1e-8:
            out[:, s] = col
            continue
        coeffs_t = [coeffs[0]] + [pywt.threshold(c, thresh, mode="soft") for c in coeffs[1:]]
        rec = pywt.waverec(coeffs_t, WAVELET)
        out[:, s] = rec[: x.shape[0]]
    return out


def phase_sanitize(phase: np.ndarray) -> np.ndarray:
    T, N = phase.shape
    k = np.arange(N, dtype=np.float32)
    out = np.empty_like(phase)
    for t in range(T):
        row = phase[t]
        slope, intercept = np.polyfit(k, row, 1)
        out[t] = row - (slope * k + intercept)
    return out


def zscore_normalize(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (x - mean) / std


def _check_field(data: dict, field: str, path: Path) -> np.ndarray:
    if field not in data:
        raise KeyError(
            f"Field '{field}' not found in {path.name}. Available keys: {sorted(data.keys())}."
        )
    return data[field].astype(np.float32)


def load_trial(path: Path) -> dict:
    raw = np.load(path, allow_pickle=False)
    return {k: raw[k] for k in raw.files}


def preprocess_node(data: dict, path: Path) -> tuple[np.ndarray, float]:
    amp = _check_field(data, FIELD_AMPLITUDE, path)
    pha = _check_field(data, FIELD_PHASE, path)

    if FIELD_SNR not in data:
        warnings.warn(f"{FIELD_SNR} missing in {path.name}; using equal-weight fusion.")
        mean_snr = 1.0
    else:
        mean_snr = float(np.mean(data[FIELD_SNR]))

    amp = hampel_filter(amp)
    amp = wavelet_denoise(amp)
    amp = zscore_normalize(amp)

    pha = phase_sanitize(pha)
    pha = zscore_normalize(pha)

    merged = np.concatenate([amp, pha], axis=1)  # (T, 2*N_sub)
    return merged, mean_snr


def trim_to_common(node_A: np.ndarray, node_B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    target = TARGET_SUBCARRIERS * 2

    def center_crop(x: np.ndarray) -> np.ndarray:
        D = x.shape[1]
        if D == target:
            return x
        if D < target:
            raise ValueError(
                f"Node has only {D//2} raw subcarriers after preprocessing; "
                f"expected at least {TARGET_SUBCARRIERS}. This pair is HT20/unusable."
            )
        start = (D - target) // 2
        return x[:, start : start + target]

    return center_crop(node_A), center_crop(node_B)


# ---------------------------------------------------------------------
# FUSION
# ---------------------------------------------------------------------
def fuse_concat(node_A, node_B, snr_A=1.0, snr_B=1.0):
    return np.concatenate([node_A, node_B], axis=1)  # (T, 4*TARGET_SUBCARRIERS)


def fuse_snr(node_A, node_B, snr_A=1.0, snr_B=1.0):
    total = snr_A + snr_B + 1e-8
    return (snr_A / total) * node_A + (snr_B / total) * node_B


def fuse_diff(node_A, node_B, snr_A=1.0, snr_B=1.0):
    return node_A - node_B


def fuse_dual_channel(node_A, node_B, snr_A=1.0, snr_B=1.0):
    return np.stack([node_A, node_B], axis=-1)  # (T, D, 2)


FUSION_STRATEGIES = {
    "concat": fuse_concat,
    "snr": fuse_snr,
    "diff": fuse_diff,
    "dual_chan": fuse_dual_channel,
}


def sliding_windows(x: np.ndarray) -> list[np.ndarray]:
    T = x.shape[0]
    out = []
    start = 0
    while start + WINDOW_SAMPLES <= T:
        win = x[start : start + WINDOW_SAMPLES]
        if win.shape[0] >= MIN_PACKETS:
            out.append(win)
        start += STRIDE_SAMPLES
    return out


def extract_features(window: np.ndarray) -> np.ndarray:
    if window.ndim == 3:
        T, N, C = window.shape
        window = window.reshape(T, N * C)

    T, D = window.shape
    feats = np.empty((8, D), dtype=np.float32)

    feats[0] = window.mean(axis=0)
    feats[1] = window.var(axis=0)
    feats[2] = np.abs(np.diff(window, axis=0)).mean(axis=0)
    feats[3] = np.abs(np.gradient(window, axis=0)).mean(axis=0)
    feats[4] = np.diff(window, axis=0).std(axis=0)

    for d in range(D):
        col = window[:, d]
        hist, _ = np.histogram(col, bins=10, density=True)
        feats[5, d] = scipy_entropy(hist + 1e-10)

    mean_val = window.mean(axis=0)
    std_val = window.std(axis=0)
    feats[6] = (window > mean_val).mean(axis=0)
    feats[7] = mean_val / np.where(std_val < 1e-8, 1.0, std_val)

    return feats.flatten()


def find_trial_pairs(data_root: Path, activity: str):
    prefix = ACTIVITY_PREFIX[activity]
    folder = data_root / activity
    skip_set = HT20_SKIP.get(activity, set())
    pairs = []

    if not folder.exists():
        print(f"  [WARN] Missing folder: {folder}")
        return pairs

    for f_A in sorted(folder.glob(f"{prefix}*_nodeA.npz")):
        key = f_A.stem.replace("_nodeA", "")
        f_B = folder / f"{key}_nodeB.npz"
        if not f_B.exists():
            print(f"  [WARN] nodeB missing for '{key}' — skipping.")
            continue
        if key in skip_set:
            print(f"  [SKIP-HT20] {activity}/{key} — unusable HT20 pair.")
            continue
        pairs.append((key, f_A, f_B))

    return pairs


def process_trial(key: str, path_A: Path, path_B: Path, activity: str, use_raw_windows: bool):
    data_A = load_trial(path_A)
    data_B = load_trial(path_B)

    node_A, snr_A = preprocess_node(data_A, path_A)
    node_B, snr_B = preprocess_node(data_B, path_B)
    node_A, node_B = trim_to_common(node_A, node_B)

    results = {}
    for strategy_name, fuse_fn in FUSION_STRATEGIES.items():
        fused = fuse_fn(node_A, node_B, snr_A, snr_B)
        windows = sliding_windows(fused)

        window_data = []
        for w_idx, win in enumerate(windows):
            item = win if use_raw_windows else extract_features(win)
            meta = {
                "trial": key,
                "activity": activity,
                "label": ACTIVITY_LABEL[activity],
                "window_idx": w_idx,
                "strategy": strategy_name,
                "snr_A": round(snr_A, 4),
                "snr_B": round(snr_B, 4),
            }
            window_data.append((item, meta))

        results[strategy_name] = window_data

    return results, snr_A, snr_B


def run(data_root: Path, out_root: Path, activities: list[str], use_raw_windows: bool):
    accum = {s: {"X": [], "y": [], "meta": []} for s in FUSION_STRATEGIES}
    total_trials = 0
    total_windows = 0
    total_skipped = 0

    for activity in activities:
        pairs = find_trial_pairs(data_root, activity)
        label = ACTIVITY_LABEL[activity]

        print(f"\n{'='*64}")
        print(f"  Activity : {activity.upper():<10}  label={label}  pairs found={len(pairs)}")
        print(f"{'='*64}")

        if not pairs:
            continue

        for key, path_A, path_B in pairs:
            try:
                results, snr_A, snr_B = process_trial(key, path_A, path_B, activity, use_raw_windows)
                n_windows = len(next(iter(results.values())))
                total_trials += 1
                total_windows += n_windows

                for strategy, window_data in results.items():
                    for feat, meta in window_data:
                        accum[strategy]["X"].append(feat)
                        accum[strategy]["y"].append(label)
                        accum[strategy]["meta"].append(meta)

                print(f"  ✓  {key:<22}  windows={n_windows:<4}  snr_A={snr_A:.1f}  snr_B={snr_B:.1f}")

            except ValueError as e:
                total_skipped += 1
                print(f"  ✗  {key:<22}  SKIPPED — {e}")
            except KeyError as e:
                total_skipped += 1
                print(f"  ✗  {key:<22}  FIELD ERROR — {e}")
            except Exception as e:
                total_skipped += 1
                print(f"  ✗  {key:<22}  FAILED — {type(e).__name__}: {e}")

    print(f"\n{'='*64}")
    print("  Saving ML-ready tensors ...")
    print(f"{'='*64}")

    for strategy, data in accum.items():
        if not data["X"]:
            print(f"  [SKIP] {strategy}: no data accumulated.")
            continue

        strategy_dir = out_root / strategy
        strategy_dir.mkdir(parents=True, exist_ok=True)

        X = np.asarray(data["X"])
        y = np.asarray(data["y"], dtype=np.int32)

        np.save(strategy_dir / "X.npy", X)
        np.save(strategy_dir / "y.npy", y)
        pd.DataFrame(data["meta"]).to_csv(strategy_dir / "meta.csv", index=False)

        class_dist = {int(v): int((y == v).sum()) for v in np.unique(y)}
        print(f"\n  [{strategy}]")
        print(f"    X shape    : {X.shape}")
        print(f"    y shape    : {y.shape}   classes: {np.unique(y).tolist()}")
        print(f"    Class dist : {class_dist}")
        print(f"    Saved      : {strategy_dir}/")

    print(f"\n{'='*64}")
    print("  Complete.")
    print(f"  Trials processed : {total_trials}")
    print(f"  Trials skipped   : {total_skipped}")
    print(f"  Total windows    : {total_windows}")
    print(f"  Mode             : {'RAW WINDOWS' if use_raw_windows else 'FEATURE VECTORS'}")
    print(f"  concat dim       : {TARGET_SUBCARRIERS * 4 if use_raw_windows else 8 * TARGET_SUBCARRIERS * 4}")
    print(f"{'='*64}\n")

    print_next_steps(out_root, use_raw_windows)


def print_next_steps(out_root: Path, use_raw_windows: bool):
    if use_raw_windows:
        concat_msg = f"np.load('{out_root}/concat/X.npy')  # (N, 20, 436)"
    else:
        concat_msg = f"np.load('{out_root}/concat/X.npy')  # (N, 3488)"
    print("NEXT STEPS")
    print("─" * 60)
    print("1. Train DANN/CNN:")
    print(f"   {concat_msg}")
    print("   y = np.load(.../y.npy)")
    print()
    print("2. If you need classical ML features, rerun with --features.")
    print("─" * 60)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Fixed fusion pipeline for RF-HAR.",
    )
    parser.add_argument("--data", default="processed_data", help="Root of synced data")
    parser.add_argument("--out", default="ml_ready", help="Output folder")
    parser.add_argument("--activity", default=None, choices=list(ACTIVITY_LABEL.keys()), help="Process one activity only")
    parser.add_argument("--features", action="store_true", help="Save feature vectors instead of raw windows")
    args = parser.parse_args()

    activities = [args.activity] if args.activity else list(ACTIVITY_LABEL.keys())
    use_raw_windows = not args.features

    print(f"\n  RF-HAR Fusion Pipeline (fixed)")
    print(f"  {'═'*60}")
    print(f"  Data root         : {args.data}")
    print(f"  Output            : {args.out}")
    print(f"  Activities        : {activities}")
    print(f"  Mode              : {'raw windows' if use_raw_windows else 'feature vectors'}")
    print(f"  Window            : {WINDOW_SEC}s / stride {STRIDE_SEC}s ({WINDOW_SAMPLES} samples / {STRIDE_SAMPLES} step)")
    print(f"  TARGET_SUBCARRIERS: {TARGET_SUBCARRIERS}")
    print(f"  Per-node features : {TARGET_SUBCARRIERS*2}")
    print(f"  Fusion strategies : {list(FUSION_STRATEGIES.keys())}")
    print(f"  HT20 skip pairs   : {sum(len(v) for v in HT20_SKIP.values())}")
    print(f"  {'═'*60}")

    run(Path(args.data), Path(args.out), activities, use_raw_windows)


if __name__ == "__main__":
    main()
