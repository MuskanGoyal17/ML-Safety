"""
Exercise Sheet 7 — Uncertainty Quantification & Calibration

Covers:
  7.4  ECE + reliability diagram for each of the three CARLA models
       on the in-distribution test set.
  7.5  Temperature scaling: optimise T on validation NLL (grid search),
       report ECE before / after per model.
  7.6  Cost-optimal decision threshold τ* = CFP/(CFP+CFN) ≈ 0.0099;
       compute total loss L = CFN·#FN + CFP·#FP for 2×2 table
       (uncalibrated/calibrated) × (τ=0.5 / τ=τ*).

Checkpoints: pass one or more --checkpoint paths.
  Task name is read from the "task" key inside the checkpoint.

Produces (under --out-dir, default outputs/):
  7.4_reliability_diagrams.png      — one reliability diagram per model
  7.4_ece_table.txt                 — ECE per model (uncalibrated)
  7.5_temperature_search.png        — NLL vs T curves per model
  7.5_ece_before_after.txt          — ECE and best T per model
  7.6_cost_loss_table.txt           — 2×2 cost-loss table

Run:
    python calibration.py \\
        --data-root  /content/data/MyDataset \\
        --checkpoint /content/drive/MyDrive/MyDataset/sheet3_results/checkpoints/pedestrian.pt \\
                     /content/drive/MyDrive/MyDataset/sheet3_results/checkpoints/traffic_light.pt \\
                     /content/drive/MyDrive/MyDataset/sheet3_results/checkpoints/vehicle.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models

sys.path.insert(0, '/content/sheet3')
try:
    from dataset import CarlaBinaryDataset, default_transform
except ModuleNotFoundError:
    # fallback: try relative path for non-Colab environments
    sys.path.insert(0, str(Path(__file__).parent.parent / "sheet3"))
    from dataset import CarlaBinaryDataset, default_transform

import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────

# Cost matrix from Exercise 7.3
C_FN  = 100    # cost of missing a pedestrian (false negative)
C_FP  = 1      # cost of braking unnecessarily (false positive)
TAU_STAR = C_FP / (C_FP + C_FN)   # ≈ 0.0099
TAU_STD  = 0.5

# Temperature grid for search (step 0.1 as per sheet)
T_GRID = np.round(np.arange(0.5, 3.05, 0.1), 2).tolist()

# Label parsing — handles True/False strings from CSV
def _parse_bool(val) -> int:
    if isinstance(val, (bool, np.bool_)):
        return int(val)
    return 1 if str(val).strip().lower() == "true" else 0


# ── Model helpers ─────────────────────────────────────────────────────────────

def build_model() -> torch.nn.Module:
    m = models.resnet18(weights=None)
    m.fc = torch.nn.Linear(m.fc.in_features, 2)
    return m


def load_model(ckpt_path: Path, device: torch.device):
    """Returns (model, task_name, label_column)."""
    state = torch.load(ckpt_path, map_location=device)
    model = build_model().to(device)
    model.load_state_dict(state["model_state"])
    model.eval()
    task   = state.get("task",         ckpt_path.stem)
    label  = state.get("label_column", f"has_{task}")
    return model, task, label


# ── Dataset with bool-safe label parsing ─────────────────────────────────────

class SafeDataset(torch.utils.data.Dataset):
    """
    Wraps CarlaBinaryDataset but re-reads labels.csv with explicit bool parsing
    so True/False strings are handled correctly.
    """
    EXTS = [".png", ".jpg", ".jpeg"]

    def __init__(self, folder: Path, label_col: str, transform) -> None:
        self.folder    = Path(folder)
        self.transform = transform
        df = pd.read_csv(self.folder / "labels.csv")
        df[label_col] = df[label_col].apply(_parse_bool)
        self.samples: list[tuple[Path, int]] = []
        for _, row in df.iterrows():
            frame = int(row["frame"])
            label = int(row[label_col])
            path  = self._find(frame)
            if path:
                self.samples.append((path, label))

    def _find(self, frame: int) -> Path | None:
        for ext in self.EXTS:
            for name in [f"{frame:06d}{ext}", f"{frame}{ext}"]:
                p = self.folder / name
                if p.exists():
                    return p
        hits = list(self.folder.rglob(f"*{frame}*"))
        hits = [h for h in hits if h.suffix.lower() in self.EXTS]
        return hits[0] if hits else None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from PIL import Image as PILImage
        path, label = self.samples[idx]
        img = PILImage.open(path).convert("RGB")
        return self.transform(img), torch.tensor(label, dtype=torch.long)


def make_loader(folder: Path, label_col: str,
                batch_size: int, train: bool = False) -> DataLoader:
    ds = SafeDataset(folder, label_col, default_transform(train=train))
    return DataLoader(ds, batch_size=batch_size,
                      shuffle=False, num_workers=2, pin_memory=False)


# ── Core computations ─────────────────────────────────────────────────────────

@torch.no_grad()
def collect_logits(model: torch.nn.Module,
                   loader: DataLoader,
                   device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Returns (logits [N,2], y_true [N])."""
    all_logits, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        all_logits.append(model(x).cpu().numpy())
        all_labels.append(y.numpy())
    return np.concatenate(all_logits), np.concatenate(all_labels)


def softmax_probs(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    """Softmax with temperature, returns P(class=1) for each sample."""
    scaled = logits / T
    # numerically stable softmax
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    exp     = np.exp(shifted)
    probs   = exp / exp.sum(axis=1, keepdims=True)
    return probs[:, 1]


def compute_ece(probs: np.ndarray, y_true: np.ndarray,
                n_bins: int = 10) -> float:
    """Expected Calibration Error (equal-width bins)."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n   = len(probs)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        conf = float(probs[mask].mean())
        acc  = float(y_true[mask].mean())
        ece += (mask.sum() / n) * abs(conf - acc)
    return ece


def nll(logits: np.ndarray, y_true: np.ndarray, T: float) -> float:
    """Mean negative log-likelihood at temperature T."""
    probs = softmax_probs(logits, T)
    probs = np.clip(probs, 1e-7, 1 - 1e-7)
    return float(-(y_true * np.log(probs) +
                   (1 - y_true) * np.log(1 - probs)).mean())


def best_temperature(val_logits: np.ndarray,
                     val_labels: np.ndarray) -> tuple[float, list[float]]:
    """Grid search over T_GRID; returns (best_T, nll_values)."""
    nll_vals = [nll(val_logits, val_labels, T) for T in T_GRID]
    best_T   = T_GRID[int(np.argmin(nll_vals))]
    return best_T, nll_vals


def compute_cost_loss(probs: np.ndarray, y_true: np.ndarray,
                      tau: float) -> tuple[int, int, int, float]:
    """
    Returns (n_fp, n_fn, total_loss) for a given decision threshold.
    Total loss = C_FN * #FN + C_FP * #FP
    """
    preds = (probs >= tau).astype(int)
    fn    = int(((preds == 0) & (y_true == 1)).sum())
    fp    = int(((preds == 1) & (y_true == 0)).sum())
    loss  = C_FN * fn + C_FP * fp
    return fp, fn, loss


def reliability_bins(probs: np.ndarray, y_true: np.ndarray,
                     n_bins: int = 10):
    """Returns (mean_conf, frac_pos) arrays for the reliability diagram."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    mean_conf, frac_pos = [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() > 0:
            mean_conf.append(float(probs[mask].mean()))
            frac_pos.append(float(y_true[mask].mean()))
    return np.array(mean_conf), np.array(frac_pos)


# ── Exercise 7.4 ─────────────────────────────────────────────────────────────

def run_7_4(records: list[dict], out_dir: Path) -> None:
    """
    records: list of dicts with keys task, test_logits, test_labels
    """
    print("\n══ Exercise 7.4 — ECE + Reliability Diagrams ══")

    n_models = len(records)
    fig, axes = plt.subplots(1, n_models,
                              figsize=(5.5 * n_models, 5), sharey=True)
    if n_models == 1:
        axes = [axes]

    ece_lines = [
        "Exercise 7.4 — ECE (uncalibrated, T=1.0)",
        "=" * 45,
        f"{'Model':<20s}  {'ECE':>8s}  {'Calibration'}",
        "-" * 45,
    ]

    for ax, rec in zip(axes, records):
        task   = rec["task"]
        probs  = softmax_probs(rec["test_logits"], T=1.0)
        y_true = rec["test_labels"]
        ece    = compute_ece(probs, y_true)
        mc, fp = reliability_bins(probs, y_true)

        verdict = ("overconfident" if (mc - fp).mean() > 0
                   else "underconfident")

        # Plot
        ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="perfect")
        ax.bar(mc, fp, width=0.08, alpha=0.55,
               color="#4c78a8", label="fraction of positives")
        ax.plot(mc, fp, "o-", color="#e45756", markersize=5,
                label="model calibration")
        ax.fill_between(mc, mc, fp, alpha=0.15, color="#e45756",
                        label="gap (miscalibration)")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Mean confidence")
        ax.set_ylabel("Fraction of positives")
        ax.set_title(f"{task}\nECE = {ece:.4f}  [{verdict}]", fontsize=10)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(alpha=0.3)

        ece_lines.append(
            f"{task:<20s}  {ece:>8.4f}  {verdict}")
        print(f"  {task}: ECE={ece:.4f}  [{verdict}]")
        rec["ece_uncal"] = ece

    fig.suptitle("Reliability Diagrams — CARLA models (uncalibrated)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "7.4_reliability_diagrams.png",
                dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_dir}/7.4_reliability_diagrams.png")

    txt = "\n".join(ece_lines)
    (out_dir / "7.4_ece_table.txt").write_text(txt)
    print(f"  wrote {out_dir}/7.4_ece_table.txt")
    print("\n" + txt)


# ── Exercise 7.5 ─────────────────────────────────────────────────────────────

def run_7_5(records: list[dict], out_dir: Path) -> None:
    print("\n══ Exercise 7.5 — Temperature Scaling ══")

    n_models = len(records)
    fig, axes = plt.subplots(1, n_models,
                              figsize=(5 * n_models, 4), sharey=False)
    if n_models == 1:
        axes = [axes]

    result_lines = [
        "Exercise 7.5 — Temperature Scaling Results",
        "=" * 60,
        f"{'Model':<20s}  {'Best T':>6s}  {'ECE before':>10s}  "
        f"{'ECE after':>9s}  {'ΔECE':>8s}",
        "-" * 60,
    ]

    for ax, rec in zip(axes, records):
        task       = rec["task"]
        val_logits = rec["val_logits"]
        val_labels = rec["val_labels"]
        test_logits= rec["test_logits"]
        test_labels= rec["test_labels"]

        best_T, nll_vals = best_temperature(val_logits, val_labels)
        rec["best_T"] = best_T

        probs_before = softmax_probs(test_logits, T=1.0)
        probs_after  = softmax_probs(test_logits, T=best_T)
        ece_before   = rec["ece_uncal"]
        ece_after    = compute_ece(probs_after, test_labels)
        delta        = ece_after - ece_before
        rec["ece_cal"]   = ece_after
        rec["probs_cal"] = probs_after

        result_lines.append(
            f"{task:<20s}  {best_T:>6.1f}  {ece_before:>10.4f}  "
            f"{ece_after:>9.4f}  {delta:>+8.4f}"
        )
        print(f"  {task}: best_T={best_T}  "
              f"ECE {ece_before:.4f} → {ece_after:.4f}  (Δ={delta:+.4f})")

        # NLL vs T plot
        ax.plot(T_GRID, nll_vals, color="#4c78a8", linewidth=1.5)
        ax.axvline(best_T, color="#e45756", linestyle="--",
                   label=f"best T={best_T}")
        ax.set_xlabel("Temperature T")
        ax.set_ylabel("Validation NLL")
        ax.set_title(f"{task}\nbest T={best_T}  ECE: "
                     f"{ece_before:.4f}→{ece_after:.4f}", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("Temperature search — validation NLL per model", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "7.5_temperature_search.png",
                dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_dir}/7.5_temperature_search.png")

    txt = "\n".join(result_lines)
    (out_dir / "7.5_ece_before_after.txt").write_text(txt)
    print(f"  wrote {out_dir}/7.5_ece_before_after.txt")
    print("\n" + txt)


# ── Exercise 7.6 ─────────────────────────────────────────────────────────────

def run_7_6(records: list[dict], out_dir: Path) -> None:
    print("\n══ Exercise 7.6 — Cost-Optimal Decision ══")
    print(f"  C_FN={C_FN}  C_FP={C_FP}  τ*={TAU_STAR:.4f}  τ_std={TAU_STD}")

    # Only run for the pedestrian model
    ped_records = [r for r in records if "pedestrian" in r["task"].lower()]
    if not ped_records:
        print("  ⚠ No pedestrian model found — running on first model instead")
        ped_records = records[:1]

    all_lines = [
        "Exercise 7.6 — Cost-Optimal Decision Threshold",
        f"C_FN={C_FN}  C_FP={C_FP}  τ*={TAU_STAR:.4f}",
        "=" * 65,
    ]

    for rec in ped_records:
        task        = rec["task"]
        probs_uncal = softmax_probs(rec["test_logits"], T=1.0)
        probs_cal   = rec.get("probs_cal",
                               softmax_probs(rec["test_logits"],
                                             rec.get("best_T", 1.0)))
        y_true      = rec["test_labels"]

        results = {}
        for label, probs in [("uncalibrated", probs_uncal),
                              ("calibrated",   probs_cal)]:
            for tau_label, tau in [("τ=0.5", TAU_STD),
                                   (f"τ*={TAU_STAR:.4f}", TAU_STAR)]:
                fp, fn, loss = compute_cost_loss(probs, y_true, tau)
                results[(label, tau_label)] = (fp, fn, loss)

        # 2×2 table
        rows    = ["uncalibrated", "calibrated"]
        col1    = "τ=0.5"
        col2    = f"τ*={TAU_STAR:.4f}"
        table_lines = [
            f"\nModel: {task}",
            f"{'':22s}  {col1:>22s}  {col2:>22s}",
            "─" * 70,
        ]
        for row in rows:
            c1_fp, c1_fn, c1_loss = results[(row, col1)]
            c2_fp, c2_fn, c2_loss = results[(row, col2)]
            table_lines.append(
                f"{row:<22s}  "
                f"FP={c1_fp:4d} FN={c1_fn:4d} L={c1_loss:6d}  "
                f"FP={c2_fp:4d} FN={c2_fn:4d} L={c2_loss:6d}"
            )

        # Find best combination
        best_combo = min(results, key=lambda k: results[k][2])
        best_loss  = results[best_combo][2]
        table_lines.append(
            f"\nBest: {best_combo[0]} + {best_combo[1]}  "
            f"→ total loss = {best_loss}"
        )

        all_lines.extend(table_lines)
        for line in table_lines:
            print(" ", line)

    txt = "\n".join(all_lines)
    (out_dir / "7.6_cost_loss_table.txt").write_text(txt)
    print(f"\n  wrote {out_dir}/7.6_cost_loss_table.txt")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Sheet 7 — Calibration & Temperature Scaling")
    ap.add_argument("--data-root",  required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, nargs="+", type=Path)
    ap.add_argument("--out-dir",    default=Path("outputs"), type=Path)
    ap.add_argument("--batch-size", default=64, type=int)
    ap.add_argument("--device",     default="auto")
    ap.add_argument("--exercises",  default="7.4,7.5,7.6",
                    help="Comma-separated exercises to run")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu")) if args.device == "auto" \
              else torch.device(args.device)
    print(f"device: {device}")

    to_run = {e.strip() for e in args.exercises.split(",")}

    # ── Load all models and collect logits once ───────────────────────────────
    records: list[dict] = []
    for ckpt in args.checkpoint:
        model, task, label_col = load_model(ckpt, device)
        print(f"\n  loaded '{task}'  label_col='{label_col}'  ← {ckpt.name}")

        val_loader  = make_loader(args.data_root / "validation",
                                  label_col, args.batch_size)
        test_loader = make_loader(args.data_root / "test",
                                  label_col, args.batch_size)

        print(f"    collecting validation logits …")
        val_logits, val_labels = collect_logits(model, val_loader, device)
        print(f"    collecting test logits …")
        test_logits, test_labels = collect_logits(model, test_loader, device)

        records.append({
            "task":        task,
            "label_col":   label_col,
            "val_logits":  val_logits,
            "val_labels":  val_labels,
            "test_logits": test_logits,
            "test_labels": test_labels,
            "ece_uncal":   0.0,   # filled by 7.4
            "best_T":      1.0,   # filled by 7.5
            "probs_cal":   None,  # filled by 7.5
        })

    # ── Run exercises ─────────────────────────────────────────────────────────
    if "7.4" in to_run:
        run_7_4(records, args.out_dir)
    else:
        # Still need ECE for 7.5/7.6 even if 7.4 not explicitly requested
        for rec in records:
            rec["ece_uncal"] = compute_ece(
                softmax_probs(rec["test_logits"]), rec["test_labels"])

    if "7.5" in to_run:
        run_7_5(records, args.out_dir)
    else:
        # Fill best_T and probs_cal so 7.6 works standalone
        for rec in records:
            rec["best_T"]    = 1.0
            rec["probs_cal"] = softmax_probs(rec["test_logits"], T=1.0)

    if "7.6" in to_run:
        run_7_6(records, args.out_dir)

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
