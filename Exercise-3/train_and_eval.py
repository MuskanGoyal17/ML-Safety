"""
Exercises 3.5 and 3.6 — Train and evaluate three binary classifiers.

For each task in {traffic_light, pedestrian, vehicle}:
  1. Fine-tune ResNet-18 (ImageNet-pretrained) with a 2-way head.
  2. Train with weighted cross-entropy (class imbalance is significant).
  3. Save best-validation checkpoint to checkpoints/{task}.pt.
  4. Evaluate on test split: accuracy, precision, recall, F1, confusion matrix.

Run:
    python train_and_eval.py --data-root /path/to/dataset --epochs 5

For CPU-only:
    python train_and_eval.py --data-root /path/to/dataset --epochs 3 --batch-size 32
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
)
from torch.utils.data import DataLoader
from torchvision import models

from dataset import (
    CarlaBinaryDataset, Task, TASK_TO_COL, class_weights, default_transform,
)

TASKS: tuple[Task, ...] = ("traffic_light", "pedestrian", "vehicle")


def build_model(num_classes: int = 2) -> nn.Module:
    """ResNet-18 with ImageNet weights, last FC swapped for `num_classes`."""
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    m = models.resnet18(weights=weights)
    in_features = m.fc.in_features
    m.fc = nn.Linear(in_features, num_classes)
    return m


def pick_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             criterion: nn.Module | None = None) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Return (loss, accuracy, y_true, y_pred). loss is NaN if criterion is None."""
    model.eval()
    losses: list[float] = []
    y_true_all: list[np.ndarray] = []
    y_pred_all: list[np.ndarray] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        if criterion is not None:
            losses.append(criterion(logits, y).item())
        pred = logits.argmax(dim=1)
        y_true_all.append(y.cpu().numpy())
        y_pred_all.append(pred.cpu().numpy())
    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    acc = float((y_true == y_pred).mean())
    loss = float(np.mean(losses)) if losses else float("nan")
    return loss, acc, y_true, y_pred


def plot_loss_curves(history: dict, out_path: Path, task: Task) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(history["epoch"], history["train_loss"], "-o", label="train", color="#4c78a8")
    ax.plot(history["epoch"], history["val_loss"], "-o", label="val", color="#e45756")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title(f"{task} classifier — loss curves")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(cm: np.ndarray, out_path: Path, task: Task) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["pred neg", "pred pos"])
    ax.set_yticklabels(["true neg", "true pos"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14)
    ax.set_title(f"{task} — confusion matrix (test)")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def train_one_task(
    task: Task,
    data_root: Path,
    out_dir: Path,
    ckpt_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    num_workers: int,
    device: torch.device,
) -> dict:
    print(f"\n{'='*60}\nTraining task: {task}\n{'='*60}")
    train_ds = CarlaBinaryDataset(data_root / "train", task,
                                  transform=default_transform(train=True))
    val_ds = CarlaBinaryDataset(data_root / "validation", task,
                                transform=default_transform(train=False))
    test_ds = CarlaBinaryDataset(data_root / "test", task,
                                 transform=default_transform(train=False))
    print(f"  sizes — train: {len(train_ds)}  val: {len(val_ds)}  test: {len(test_ds)}")
    print(f"  train positive fraction: {train_ds.positive_fraction():.3f}")

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin)

    weights = class_weights(train_ds).to(device)
    print(f"  class weights [neg, pos]: {weights.cpu().numpy().round(3).tolist()}")
    criterion = nn.CrossEntropyLoss(weight=weights)

    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    ckpt_path = ckpt_dir / f"{task}.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        n_batches = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running += loss.item()
            n_batches += 1
        scheduler.step()
        train_loss = running / max(n_batches, 1)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, device, criterion)
        dt = time.time() - t0
        print(f"  epoch {epoch:2d}/{epochs}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_acc={val_acc:.3f}  ({dt:.1f}s)")
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model_state": model.state_dict(),
                        "task": task,
                        "label_column": TASK_TO_COL[task]}, ckpt_path)
            print(f"    new best val_loss — saved to {ckpt_path}")

    plot_loss_curves(history, out_dir / f"3.5_loss_{task}.png", task)

    # Final eval with best checkpoint
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state"])
    _, test_acc, y_true, y_pred = evaluate(model, test_loader, device)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plot_confusion(cm, out_dir / f"3.6_confusion_{task}.png", task)
    print(f"  TEST — acc={test_acc:.3f}  prec={prec:.3f}  rec={rec:.3f}  f1={f1:.3f}")
    print(f"  confusion matrix:\n{cm}")

    return {
        "task": task,
        "history": history,
        "test_metrics": {
            "accuracy": test_acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "confusion_matrix": cm.tolist(),
            "n_test": int(len(y_true)),
            "test_positive_fraction": float(y_true.mean()),
        },
        "checkpoint": str(ckpt_path),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, type=Path,
                    help="Folder containing train/, validation/, test/")
    ap.add_argument("--out-dir", default=Path("outputs"), type=Path)
    ap.add_argument("--ckpt-dir", default=Path("checkpoints"), type=Path)
    ap.add_argument("--epochs", default=5, type=int)
    ap.add_argument("--batch-size", default=64, type=int)
    ap.add_argument("--lr", default=1e-4, type=float)
    ap.add_argument("--num-workers", default=4, type=int)
    ap.add_argument("--device", default="auto",
                    choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--tasks", default=",".join(TASKS),
                    help="Comma-separated subset of tasks to run")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    print(f"device: {device}")
    torch.manual_seed(0)

    requested = [t.strip() for t in args.tasks.split(",") if t.strip()]
    invalid = [t for t in requested if t not in TASKS]
    if invalid:
        raise SystemExit(f"unknown tasks: {invalid}. valid: {TASKS}")

    results = []
    for task in requested:
        r = train_one_task(
            task=task,
            data_root=args.data_root,
            out_dir=args.out_dir,
            ckpt_dir=args.ckpt_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            num_workers=args.num_workers,
            device=device,
        )
        results.append(r)

    # Combined report
    report_path = args.out_dir / "3.6_test_metrics.json"
    report_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {report_path}")

    # Human-readable summary
    lines = ["Test-set metrics per model", "=" * 60,
             f"{'task':18s} {'acc':>6s} {'prec':>6s} {'rec':>6s} {'F1':>6s} {'n+':>6s}"]
    for r in results:
        m = r["test_metrics"]
        n_pos = int(m["confusion_matrix"][1][0] + m["confusion_matrix"][1][1])
        lines.append(f"{r['task']:18s} {m['accuracy']:6.3f} "
                     f"{m['precision']:6.3f} {m['recall']:6.3f} "
                     f"{m['f1']:6.3f} {n_pos:6d}")
    summary = "\n".join(lines)
    (args.out_dir / "3.6_test_summary.txt").write_text(summary)
    print("\n" + summary)


if __name__ == "__main__":
    main()
