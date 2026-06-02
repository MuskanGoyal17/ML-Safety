"""
Exercise 4.7 — Per-Class Evaluation

Loads the three trained checkpoints from Sheet 3 and evaluates them on the
test split, producing:
  - precision, recall, F1 per model
  - confusion matrix plots
  - a minimum-recall threshold argument for the pedestrian model


"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, recall_score,
    precision_recall_curve,
)
from torch.utils.data import DataLoader
from torchvision import models

sys.path.insert(0, str(Path(__file__).parent.parent / "sheet3"))
from dataset import CarlaBinaryDataset, default_transform

TASKS = ("traffic_light", "pedestrian", "vehicle")


def build_model() -> torch.nn.Module:
    m = models.resnet18(weights=None)
    m.fc = torch.nn.Linear(m.fc.in_features, 2)
    return m


def load_model(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    state = torch.load(ckpt_path, map_location=device)
    model = build_model().to(device)
    model.load_state_dict(state["model_state"])
    model.eval()
    return model


@torch.no_grad()
def get_predictions(model, loader, device):
    y_true, y_pred, y_prob = [], [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs = F.softmax(logits, dim=1)[:, 1]
        pred = logits.argmax(dim=1)
        y_true.extend(y.numpy())
        y_pred.extend(pred.cpu().numpy())
        y_prob.extend(probs.cpu().numpy())
    return np.array(y_true), np.array(y_pred), np.array(y_prob)


def plot_confusion(cm: np.ndarray, task: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["pred neg", "pred pos"], fontsize=10)
    ax.set_yticklabels(["true neg", "true pos"], fontsize=10)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14, fontweight="bold")
    tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    rec = tp / max(tp + fn, 1)
    prec = tp / max(tp + fp, 1)
    ax.set_title(f"{task}\nrecall={rec:.3f}  precision={prec:.3f}", fontsize=10)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_pr_bar(metrics: list[dict], out_path: Path) -> None:
    tasks = [m["task"] for m in metrics]
    precs = [m["precision"] for m in metrics]
    recs = [m["recall"] for m in metrics]
    f1s = [m["f1"] for m in metrics]
    x = np.arange(len(tasks))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width, precs, width, label="precision", color="#4c78a8")
    ax.bar(x,          recs,  width, label="recall",    color="#e45756")
    ax.bar(x + width,  f1s,   width, label="F1",        color="#54a24b")

    # safety threshold line for pedestrian recall
    ped_idx = tasks.index("pedestrian")
    ax.axhline(y=0.90, color="red", linestyle="--", linewidth=1.2,
               label="min. required recall (0.90)")
    ax.axvline(x=ped_idx, color="grey", linestyle=":", alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("score")
    ax.set_title("Per-model metrics on test set (Sheet 4 — Exercise 4.7)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_pedestrian_pr_curve(y_true: np.ndarray, y_prob: np.ndarray,
                              out_path: Path) -> None:
    """Precision-recall curve for pedestrian to justify threshold choice."""
    prec_curve, rec_curve, thresholds = precision_recall_curve(y_true, y_prob)
    # find threshold that achieves recall >= 0.90
    valid = np.where(rec_curve >= 0.90)[0]
    if len(valid):
        idx = valid[np.argmax(prec_curve[valid])]
        best_thresh = thresholds[min(idx, len(thresholds)-1)]
        best_prec = prec_curve[idx]
        best_rec = rec_curve[idx]
    else:
        best_thresh = best_prec = best_rec = float("nan")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(rec_curve, prec_curve, color="#e45756", lw=2, label="PR curve")
    ax.axvline(x=0.90, color="red", linestyle="--", label="recall = 0.90 target")
    if not np.isnan(best_thresh):
        ax.scatter([best_rec], [best_prec], zorder=5, color="black", s=80,
                   label=f"best point at recall≥0.90\n(thresh={best_thresh:.2f}, prec={best_prec:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Pedestrian — Precision-Recall curve")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--checkpoints-dir", required=True, type=Path)
    ap.add_argument("--out-dir", default=Path("outputs"), type=Path)
    ap.add_argument("--batch-size", default=64, type=int)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"device: {device}")

    all_metrics = []
    ped_y_true = ped_y_prob = None

    for task in TASKS:
        ckpt = args.checkpoints_dir / f"{task}.pt"
        if not ckpt.exists():
            print(f"  [skip] checkpoint not found: {ckpt}")
            continue

        model = load_model(ckpt, device)
        test_ds = CarlaBinaryDataset(args.data_root / "test", task,
                                     transform=default_transform(train=False))
        loader = DataLoader(test_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=2)

        y_true, y_pred, y_prob = get_predictions(model, loader, device)
        prec  = precision_score(y_true, y_pred, zero_division=0)
        rec   = recall_score(y_true, y_pred, zero_division=0)
        f1    = f1_score(y_true, y_pred, zero_division=0)
        cm    = confusion_matrix(y_true, y_pred, labels=[0, 1])
        acc   = float((y_true == y_pred).mean())

        print(f"\n{task}: acc={acc:.3f}  prec={prec:.3f}  rec={rec:.3f}  f1={f1:.3f}")
        print(f"  confusion matrix:\n{cm}")

        plot_confusion(cm, task, args.out_dir / f"4.7_confusion_{task}.png")

        all_metrics.append({
            "task": task, "accuracy": acc,
            "precision": prec, "recall": rec, "f1": f1,
            "confusion_matrix": cm.tolist(),
        })

        if task == "pedestrian":
            ped_y_true, ped_y_prob = y_true, y_prob

    # Plots
    plot_pr_bar(all_metrics, args.out_dir / "4.7_precision_recall_bar.png")
    if ped_y_true is not None:
        plot_pedestrian_pr_curve(ped_y_true, ped_y_prob,
                                  args.out_dir / "4.7_pedestrian_pr_curve.png")

    # Summary text
    lines = [
        "Exercise 4.7 — Per-Class Evaluation Results",
        "=" * 60,
        f"{'task':18s} {'acc':>6s} {'prec':>6s} {'rec':>6s} {'F1':>6s}",
        "-" * 48,
    ]
    for m in all_metrics:
        lines.append(f"{m['task']:18s} {m['accuracy']:6.3f} "
                     f"{m['precision']:6.3f} {m['recall']:6.3f} {m['f1']:6.3f}")
    lines += [
        "",
        "Confusion matrices (TN / FP / FN / TP):",
    ]
    for m in all_metrics:
        cm = m["confusion_matrix"]
        lines.append(f"  {m['task']:18s}  TN={cm[0][0]}  FP={cm[0][1]}"
                     f"  FN={cm[1][0]}  TP={cm[1][1]}")

    summary = "\n".join(lines)
    print("\n" + summary)
    out_path = args.out_dir / "4.7_metrics_summary.txt"
    out_path.write_text(summary)
    print(f"\n  wrote {out_path}")

    # Save JSON for report
    (args.out_dir / "4.7_metrics.json").write_text(
        json.dumps(all_metrics, indent=2))


if __name__ == "__main__":
    main()
