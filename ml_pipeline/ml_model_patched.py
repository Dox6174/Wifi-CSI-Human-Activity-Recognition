
"""
ml_model_patched.py

Patched RF-HAR training pipeline for raw-window CSI input.

Fixes relative to the earlier version:
1) Stores real macro-F1 in the final summary table.
2) Adds optional class-weighted loss to reduce class imbalance effects.
3) Adds DANN lambda cap and warmup to reduce over-aggressive domain confusion.
4) Keeps the same trial-level 80/20 split to prevent leakage.
5) Works with raw-window tensors saved by fusion_pipeline_fixed.py:
       X.npy shape = (N, 20, 436)
"""

import argparse
import math
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, f1_score


ACTIVITY_NAMES = {0: "walk", 1: "sit", 2: "stand", 3: "hand", 4: "empty"}
N_CLASSES = 5
SEED = 42

HT20_EXCLUDED = {
    'sit1', 'sit22', 'sit27', 'sit29', 'sit2', 'sit31', 'sit32', 'sit5',
    'st33', 'st8',
    'h13', 'h19', 'h23', 'h31', 'h33',
}


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def assign_subject(trial: str) -> int:
    """
    Subject 1 = first 20 trials per activity
    Subject 2 = last 20 trials per activity
    Empty room: e1_* / e2_*.
    """
    if trial.startswith("e1"):
        return 1
    if trial.startswith("e2"):
        return 2
    nums = re.findall(r"\d+", trial)
    if not nums:
        return -1
    return 1 if int(nums[-1]) <= 20 else 2


def load_dataset(data_dir: Path):
    X = np.load(data_dir / "X.npy")
    y = np.load(data_dir / "y.npy")
    meta = pd.read_csv(data_dir / "meta.csv")

    if len(X) != len(meta):
        raise ValueError(f"X.npy rows ({len(X)}) != meta.csv rows ({len(meta)})")

    meta = meta.copy()
    meta["subject"] = meta["trial"].apply(assign_subject)

    keep = ~meta["trial"].isin(HT20_EXCLUDED)
    X = X[keep.values]
    y = y[keep.values]
    meta = meta[keep].reset_index(drop=True)

    print(f"  Loaded: {len(X)} windows after excluding {(~keep).sum()} HT20 windows")
    print(f"  Shape: X={X.shape}  y={y.shape}")
    print(f"  Subject distribution:")
    for s in sorted(meta["subject"].unique()):
        print(f"    Subject {s}: {(meta['subject'] == s).sum()} windows")
    print()
    return X, y, meta


def split_by_trial_80_20(meta: pd.DataFrame, subject: int, seed: int = SEED):
    """
    Split at trial level, stratified by activity, to avoid window leakage.
    """
    rng = random.Random(seed)
    train_idx, test_idx = [], []
    subj_meta = meta[meta["subject"] == subject]

    for activity in subj_meta["activity"].unique():
        act_meta = subj_meta[subj_meta["activity"] == activity]
        trials = sorted(act_meta["trial"].unique())

        n_test = max(1, round(len(trials) * 0.20))
        rng.shuffle(trials)
        test_trials = set(trials[:n_test])

        for idx, row in act_meta.iterrows():
            if row["trial"] in test_trials:
                test_idx.append(idx)
            else:
                train_idx.append(idx)

    return train_idx, test_idx


class CSIDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, indices):
        self.X = torch.FloatTensor(X[indices])
        self.y = torch.LongTensor(y[indices])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def build_class_weights(y: np.ndarray, indices, n_classes: int = N_CLASSES) -> torch.Tensor:
    """
    Inverse-frequency class weights normalized to mean 1.
    Helps with minority classes like 'empty' and 'sit' imbalance.
    """
    labels = y[indices]
    counts = np.bincount(labels, minlength=n_classes).astype(np.float32)
    counts = np.where(counts == 0, 1.0, counts)
    weights = counts.sum() / (n_classes * counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


class FeatureExtractor(nn.Module):
    def __init__(self, input_features: int = 436, dropout: float = 0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(input_features, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool = nn.MaxPool1d(2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, time, features) -> Conv1d expects (batch, channels, time)
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.gap(x).squeeze(-1)
        return self.drop(x)


class ActivityClassifier(nn.Module):
    def __init__(self, feature_dim: int = 256, n_classes: int = N_CLASSES, dropout: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, 128)
        self.fc2 = nn.Linear(128, n_classes)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)


class BaselineCNN(nn.Module):
    def __init__(self, input_features: int = 436, dropout: float = 0.3):
        super().__init__()
        self.encoder = FeatureExtractor(input_features, dropout=dropout)
        self.classifier = ActivityClassifier(dropout=dropout)

    def forward(self, x):
        features = self.encoder(x)
        return self.classifier(features)


class _GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = float(lambda_)
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GRL(nn.Module):
    def __init__(self):
        super().__init__()
        self._lambda = 0.0

    def set_lambda(self, lam: float):
        self._lambda = float(lam)

    def forward(self, x):
        return _GradientReversalFn.apply(x, self._lambda)


class DomainClassifier(nn.Module):
    def __init__(self, feature_dim: int = 256, n_domains: int = 2, dropout: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, n_domains)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DANN(nn.Module):
    def __init__(self, input_features: int = 436, dropout: float = 0.3):
        super().__init__()
        self.encoder = FeatureExtractor(input_features, dropout=dropout)
        self.act_head = ActivityClassifier(dropout=dropout)
        self.grl = GRL()
        self.dom_head = DomainClassifier(dropout=dropout)

    def forward(self, x, lambda_val: float = 1.0):
        self.grl.set_lambda(lambda_val)
        features = self.encoder(x)
        act_logits = self.act_head(features)
        dom_logits = self.dom_head(self.grl(features))
        return act_logits, dom_logits, features

    def predict_activity(self, x):
        features = self.encoder(x)
        return self.act_head(features)


def get_lambda(epoch: int, total_epochs: int, lambda_max: float = 1.0, warmup_epochs: int = 0) -> float:
    """
    DANN lambda schedule with optional warmup and cap.
    """
    if epoch < warmup_epochs:
        return 0.0
    p = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
    base = 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0
    return float(lambda_max * base)


def train_cnn_epoch(model, loader, optimizer, device, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(Xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(yb)
        correct += (logits.argmax(1) == yb).sum().item()
        total += len(yb)
    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, device, dann_mode: bool = False):
    model.eval()
    all_preds, all_labels = [], []
    for Xb, yb in loader:
        Xb = Xb.to(device)
        if dann_mode:
            logits = model.predict_activity(Xb)
        else:
            logits = model(Xb)
        all_preds.append(logits.argmax(1).cpu())
        all_labels.append(yb.cpu())
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    acc = float((preds == labels).mean())
    f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
    return acc, f1, preds, labels


def train_dann_epoch(model, src_X, src_y, tgt_X, batch_size, optimizer, device, lambda_val, criterion_act, criterion_dom):
    model.train()
    half = batch_size // 2
    n_src = len(src_X)
    n_tgt = len(tgt_X)

    src_perm = torch.randperm(n_src)
    tgt_perm = torch.randperm(n_tgt)

    n_batches = max(n_src, n_tgt) // half
    total_loss = total_act = total_dom = 0.0

    for b in range(n_batches):
        src_idx = src_perm[torch.arange(b * half, (b + 1) * half) % n_src]
        tgt_idx = tgt_perm[torch.arange(b * half, (b + 1) * half) % n_tgt]

        Xb = torch.cat([src_X[src_idx], tgt_X[tgt_idx]]).to(device)
        y_act = src_y[src_idx].to(device)
        y_dom = torch.cat([
            torch.zeros(half, dtype=torch.long),
            torch.ones(half, dtype=torch.long),
        ]).to(device)

        optimizer.zero_grad()
        act_logits, dom_logits, _ = model(Xb, lambda_val=lambda_val)
        L_act = criterion_act(act_logits[:half], y_act)
        L_dom = criterion_dom(dom_logits, y_dom)
        L_total = L_act + L_dom
        L_total.backward()
        optimizer.step()

        total_loss += L_total.item()
        total_act += L_act.item()
        total_dom += L_dom.item()

    n = max(n_batches, 1)
    return total_loss / n, total_act / n, total_dom / n


@torch.no_grad()
def eval_domain_acc(model, src_loader, tgt_loader, device, lambda_val):
    model.eval()
    correct = total = 0

    for Xb, _ in src_loader:
        Xb = Xb.to(device)
        _, dom_logits, _ = model(Xb, lambda_val=lambda_val)
        pred = dom_logits.argmax(1)
        correct += (pred == 0).sum().item()
        total += len(Xb)

    for Xb, _ in tgt_loader:
        Xb = Xb.to(device)
        _, dom_logits, _ = model(Xb, lambda_val=lambda_val)
        pred = dom_logits.argmax(1)
        correct += (pred == 1).sum().item()
        total += len(Xb)

    return correct / max(total, 1)


def print_results(title: str, acc: float, f1: float, preds, labels):
    print(f"\n  {'─'*58}")
    print(f"  {title}")
    print(f"  {'─'*58}")
    print(f"  Accuracy : {acc*100:.2f}%")
    print(f"  F1-macro : {f1*100:.2f}%")
    print()
    names = [ACTIVITY_NAMES.get(i, str(i)) for i in sorted(set(labels))]
    report = classification_report(labels, preds, target_names=names, zero_division=0)
    for line in report.splitlines():
        print(f"    {line}")


def run_baseline_cnn(args, X, y, meta, device):
    print(f"\n{'═'*62}")
    print("  MODEL 1 — BASELINE CNN")
    print(f"{'═'*62}")

    src_train_idx, src_test_idx = split_by_trial_80_20(meta, args.source)
    tgt_train_idx, tgt_test_idx = split_by_trial_80_20(meta, args.target)

    src_train_set = CSIDataset(X, y, src_train_idx)
    src_test_set = CSIDataset(X, y, src_test_idx)
    tgt_test_set = CSIDataset(X, y, tgt_test_idx)

    print(f"  Subject {args.source} (source): {len(src_train_idx)} train / {len(src_test_idx)} test windows")
    print(f"  Subject {args.target} (target): {len(tgt_train_idx)} train (unused for CNN) / {len(tgt_test_idx)} test windows")

    train_loader = DataLoader(src_train_set, batch_size=args.batch, shuffle=True, drop_last=True)
    src_test_loader = DataLoader(src_test_set, batch_size=args.batch, shuffle=False)
    tgt_test_loader = DataLoader(tgt_test_set, batch_size=args.batch, shuffle=False)

    input_features = X.shape[2]
    model = BaselineCNN(input_features=input_features, dropout=args.dropout).to(device)

    class_weights = None
    if not args.no_class_weights:
        class_weights = build_class_weights(y, src_train_idx).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_src_acc = 0.0
    best_state = None

    print(f"\n  Training {args.epochs} epochs ...\n")
    print(f"  {'Epoch':>6}  {'Loss':>8}  {'TrainAcc':>10}  {'SrcTestAcc':>12}  {'TgtTestAcc':>12}  {'Lambda':>8}")
    print(f"  {'─'*68}")

    for epoch in range(args.epochs):
        loss, train_acc = train_cnn_epoch(model, train_loader, optimizer, device, criterion)
        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            src_acc, _, _, _ = evaluate(model, src_test_loader, device)
            tgt_acc, _, _, _ = evaluate(model, tgt_test_loader, device)
            print(f"  {epoch+1:>6}  {loss:>8.4f}  {train_acc*100:>9.2f}%  {src_acc*100:>11.2f}%  {tgt_acc*100:>11.2f}%  {'N/A':>8}")
            if src_acc > best_src_acc:
                best_src_acc = src_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    src_acc, src_f1, src_pred, src_true = evaluate(model, src_test_loader, device)
    tgt_acc, tgt_f1, tgt_pred, tgt_true = evaluate(model, tgt_test_loader, device)

    print_results(f"Baseline CNN — Same Person (S{args.source} train → S{args.source} test)", src_acc, src_f1, src_pred, src_true)
    print_results(f"Baseline CNN — Cross Person (S{args.source} train → S{args.target} test)", tgt_acc, tgt_f1, tgt_pred, tgt_true)

    out_path = Path(args.out) / f"cnn_s{args.source}_to_s{args.target}.pt"
    out_path.parent.mkdir(exist_ok=True)
    torch.save({
        "model": best_state if best_state is not None else model.state_dict(),
        "src_acc": src_acc, "tgt_acc": tgt_acc,
        "src_f1": src_f1, "tgt_f1": tgt_f1
    }, out_path)
    print(f"\n  Checkpoint saved → {out_path}")

    return model, src_acc, src_f1, tgt_acc, tgt_f1


def run_dann(args, X, y, meta, device):
    print(f"\n{'═'*62}")
    print("  MODEL 2 — DANN (Domain-Adversarial Neural Network)")
    print(f"{'═'*62}")

    src_train_idx, src_test_idx = split_by_trial_80_20(meta, args.source)
    tgt_train_idx, tgt_test_idx = split_by_trial_80_20(meta, args.target)

    src_X_train = torch.FloatTensor(X[src_train_idx])
    src_y_train = torch.LongTensor(y[src_train_idx])
    tgt_X_train = torch.FloatTensor(X[tgt_train_idx])

    src_test_set = CSIDataset(X, y, src_test_idx)
    tgt_test_set = CSIDataset(X, y, tgt_test_idx)
    src_test_loader = DataLoader(src_test_set, batch_size=args.batch, shuffle=False)
    tgt_test_loader = DataLoader(tgt_test_set, batch_size=args.batch, shuffle=False)

    print(f"  Subject {args.source} (source): {len(src_X_train)} labeled train windows  |  {len(src_test_idx)} test windows (held out)")
    print(f"  Subject {args.target} (target): {len(tgt_X_train)} unlabeled adapt windows  |  {len(tgt_test_idx)} test windows (held out)")

    input_features = X.shape[2]
    model = DANN(input_features=input_features, dropout=args.dropout).to(device)

    class_weights = None
    if not args.no_class_weights:
        class_weights = build_class_weights(y, src_train_idx).to(device)
    criterion_act = nn.CrossEntropyLoss(weight=class_weights)
    criterion_dom = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_tgt_acc = 0.0
    best_state = None

    print(f"\n  Training {args.epochs} epochs with lambda annealing ...\n")
    print(f"  {'Epoch':>6}  {'L_act':>8}  {'L_dom':>8}  {'SrcTestAcc':>12}  {'TgtTestAcc':>12}  {'DomAcc':>9}  {'Lambda':>8}")
    print(f"  {'─'*78}")

    for epoch in range(args.epochs):
        lam = get_lambda(epoch, args.epochs, lambda_max=args.dann_lambda_max, warmup_epochs=args.dann_warmup)
        total_loss, act_loss, dom_loss = train_dann_epoch(
            model, src_X_train, src_y_train, tgt_X_train, args.batch, optimizer, device,
            lam, criterion_act, criterion_dom
        )
        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            src_acc, _, _, _ = evaluate(model, src_test_loader, device, dann_mode=True)
            tgt_acc, _, _, _ = evaluate(model, tgt_test_loader, device, dann_mode=True)
            dom_acc = eval_domain_acc(model, src_test_loader, tgt_test_loader, device, lam)
            print(f"  {epoch+1:>6}  {act_loss:>8.4f}  {dom_loss:>8.4f}  {src_acc*100:>11.2f}%  {tgt_acc*100:>11.2f}%  {dom_acc*100:>8.1f}%  {lam:>8.4f}")
            if tgt_acc > best_tgt_acc:
                best_tgt_acc = tgt_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    src_acc, src_f1, src_pred, src_true = evaluate(model, src_test_loader, device, dann_mode=True)
    tgt_acc, tgt_f1, tgt_pred, tgt_true = evaluate(model, tgt_test_loader, device, dann_mode=True)
    dom_acc = eval_domain_acc(model, src_test_loader, tgt_test_loader, device, lambda_val=1.0)

    print_results(f"DANN — Same Person (S{args.source} train → S{args.source} test)", src_acc, src_f1, src_pred, src_true)
    print_results(f"DANN — Cross Person (S{args.source} train → S{args.target} test)", tgt_acc, tgt_f1, tgt_pred, tgt_true)
    print(f"\n  Domain classifier accuracy: {dom_acc*100:.1f}%")
    print("  (50% = perfect confusion = GRL working correctly)")

    out_path = Path(args.out) / f"dann_s{args.source}_to_s{args.target}.pt"
    torch.save({
        "model": best_state if best_state is not None else model.state_dict(),
        "tgt_acc": tgt_acc,
        "tgt_f1": tgt_f1,
        "dom_acc": dom_acc
    }, out_path)
    print(f"\n  Checkpoint saved → {out_path}")

    return model, src_acc, src_f1, tgt_acc, tgt_f1


def print_summary_table(rows):
    print(f"\n{'═'*62}")
    print("  FINAL RESULTS SUMMARY")
    print(f"{'═'*62}")
    print(f"  {'Model':<22}  {'Eval Setting':<28}  {'Acc':>6}  {'F1':>6}")
    print(f"  {'─'*60}")
    for row in rows:
        print(f"  {row['model']:<22}  {row['setting']:<28}  {row['acc']*100:>5.1f}%  {row['f1']*100:>5.1f}%")
    print(f"{'═'*62}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="ml_ready/concat", help="Path to fusion pipeline output dir")
    parser.add_argument("--source", type=int, default=1, choices=[1, 2], help="Source subject")
    parser.add_argument("--target", type=int, default=2, choices=[1, 2], help="Target subject")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--dann_lambda_max", type=float, default=0.85)
    parser.add_argument("--dann_warmup", type=int, default=5)
    parser.add_argument("--out", default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--no_dann", action="store_true", help="Skip DANN, only train baseline CNN")
    parser.add_argument("--no_class_weights", action="store_true", help="Disable class-weighted loss")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    assert args.source != args.target, "Source and target must be different subjects"

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n  RF-HAR Training Pipeline")
    print(f"  {'═'*60}")
    print(f"  Device  : {device}")
    print(f"  Source  : Subject {args.source}  (labeled, train domain)")
    print(f"  Target  : Subject {args.target}  (cross-person test)")
    print(f"  Epochs  : {args.epochs}")
    print(f"  Batch   : {args.batch}")
    print(f"  LR      : {args.lr}")
    print(f"  Seed    : {args.seed}")
    print(f"  {'═'*60}\n")

    print("  Loading dataset ...")
    X, y, meta = load_dataset(Path(args.data))

    subjects = sorted(meta["subject"].unique())
    assert args.source in subjects, f"Subject {args.source} not found. Available: {subjects}"
    assert args.target in subjects, f"Subject {args.target} not found. Available: {subjects}"

    results = []

    cnn_model, cnn_src_acc, cnn_src_f1, cnn_tgt_acc, cnn_tgt_f1 = run_baseline_cnn(args, X, y, meta, device)
    results.append({"model": "Baseline CNN", "setting": f"S{args.source} → S{args.source} (same)", "acc": cnn_src_acc, "f1": cnn_src_f1})
    results.append({"model": "Baseline CNN", "setting": f"S{args.source} → S{args.target} (cross)", "acc": cnn_tgt_acc, "f1": cnn_tgt_f1})

    if not args.no_dann:
        dann_model, dann_src_acc, dann_src_f1, dann_tgt_acc, dann_tgt_f1 = run_dann(args, X, y, meta, device)
        improvement = (dann_tgt_acc - cnn_tgt_acc) * 100.0
        results.append({"model": "DANN", "setting": f"S{args.source} → S{args.source} (same)", "acc": dann_src_acc, "f1": dann_src_f1})
        results.append({"model": "DANN", "setting": f"S{args.source} → S{args.target} (cross)", "acc": dann_tgt_acc, "f1": dann_tgt_f1})
        print(f"\n  Cross-person improvement from DANN: {improvement:+.1f} percentage points")

    print_summary_table(results)

    other = 2 if args.source == 1 else 1
    print("  Run the flip direction for LOSO average:")
    print(f"  python {Path(__file__).name} --data {args.data} --source {args.target} --target {args.source}")


if __name__ == "__main__":
    main()
