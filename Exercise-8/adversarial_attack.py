"""
Exercise Sheet 8 — Adversarial ML: FGSM Attack on the CARLA Pedestrian Detector

Covers:
  8.4  Implement FGSM, generate adversarial examples for ε ∈ {0.01, 0.05, 0.1},
       display clean vs adversarial side-by-side.
  8.5  Evaluate recall drop on adversarial test set for each model × ε.

Expected layout under --data-root:
  <data-root>/
    test/          ← clean test split (labels.csv + images)

Checkpoints: pass one or more --checkpoint arguments, one per model.
  The script auto-detects the task name from the checkpoint "task" key
  (e.g. "pedestrian", "red_light", "speed_limit").  If the key is absent
  the filename stem is used as the model name.

Produces (under --out-dir, default outputs/):
  8.4_adversarial_examples_<task>.png    — clean vs adv side-by-side (3 eps)
  8.5_recall_drop_table.txt              — per-model x per-ε recall table
  8.5_recall_drop_plot.png               — grouped bar chart of recall

Run (single checkpoint):
    python adversarial_attack.py \\
        --data-root /content/data/MyDataset \\
        --checkpoint /content/drive/MyDrive/MyDataset/sheet3_results/checkpoints/pedestrian.pt

Run (three checkpoints):
    python adversarial_attack.py \\
        --data-root /content/data/MyDataset \\
        --checkpoint .../pedestrian.pt .../red_light.pt .../speed_limit.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import recall_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models

sys.path.insert(0, str(Path(__file__).parent.parent / "sheet3"))
from dataset import CarlaBinaryDataset, default_transform

# ── Constants ─────────────────────────────────────────────────────────────────

EPSILONS      = [0.01, 0.05, 0.1]
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# Valid normalised range per channel: [(0−mean)/std, (1−mean)/std]
NORM_MIN = (torch.zeros(3, 1, 1) - IMAGENET_MEAN) / IMAGENET_STD
NORM_MAX = (torch.ones(3, 1, 1)  - IMAGENET_MEAN) / IMAGENET_STD


# ── Model helpers ─────────────────────────────────────────────────────────────

def build_model() -> torch.nn.Module:
    m = models.resnet18(weights=None)
    m.fc = torch.nn.Linear(m.fc.in_features, 2)
    return m


def load_model(ckpt_path: Path, device: torch.device):
    """Returns (model, task_name)."""
    state = torch.load(ckpt_path, map_location=device)
    model = build_model().to(device)
    model.load_state_dict(state["model_state"])
    model.eval()
    task = state.get("task", ckpt_path.stem)
    return model, task


# ── FGSM implementation ───────────────────────────────────────────────────────

def fgsm(model: torch.nn.Module,
         x: torch.Tensor,
         y: torch.Tensor,
         eps: float,
         device: torch.device) -> torch.Tensor:
    """
    Fast Gradient Sign Method (untargeted).

    x_adv = x + ε · sign(∇_x L(y, f(x)))

    Perturbation is applied in normalised pixel space and clamped so that
    the reconstructed pixel values stay in [0, 1].

    Args:
        x   : (B, C, H, W) normalised tensor
        y   : (B,)          integer labels
        eps : perturbation budget in normalised space
    Returns:
        x_adv: perturbed tensor, same shape as x, detached from graph
    """
    if eps == 0.0:
        return x.clone().detach()

    x_adv = x.clone().detach().to(device).requires_grad_(True)
    y     = y.to(device)

    logits = model(x_adv)
    loss   = F.cross_entropy(logits, y)
    model.zero_grad()
    loss.backward()

    with torch.no_grad():
        sign  = x_adv.grad.sign()
        x_adv = x_adv + eps * sign
        lo    = NORM_MIN.to(device)
        hi    = NORM_MAX.to(device)
        x_adv = torch.max(torch.min(x_adv, hi), lo)

    return x_adv.detach()


# ── Dataset wrapper ───────────────────────────────────────────────────────────

class SubsetDataset(Dataset):
    """Returns only the first n samples of a base dataset."""
    def __init__(self, base: Dataset, n: int) -> None:
        self.base = base
        self.n    = min(n, len(base))

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx):
        return self.base[idx]


def make_loader(data_root: Path, task: str,
                batch_size: int, n: int | None = None) -> DataLoader:
    ds = CarlaBinaryDataset(data_root / "test", task,
                            transform=default_transform(train=False))
    if n is not None:
        ds = SubsetDataset(ds, n)
    return DataLoader(ds, batch_size=batch_size,
                      shuffle=False, num_workers=2, pin_memory=False)


# ── Display helper ────────────────────────────────────────────────────────────

def to_uint8(t: torch.Tensor) -> np.ndarray:
    """CHW normalised tensor → HWC uint8 numpy for imshow."""
    img = (t.cpu() * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def linf_pixels(clean: torch.Tensor, adv: torch.Tensor) -> float:
    """L∞ perturbation magnitude in [0,1] pixel space."""
    return float(((adv.cpu() - clean.cpu()) * IMAGENET_STD).abs().max())


# ── Exercise 8.4 ─────────────────────────────────────────────────────────────

def run_8_4(model: torch.nn.Module,
            task: str,
            data_root: Path,
            out_dir: Path,
            device: torch.device,
            n_vis: int = 3) -> None:
    """
    Show n_vis rows of [clean | ε=0.01 | ε=0.05 | ε=0.10] side-by-side.
    Only uses correctly-classified positive samples so attack impact is visible.
    """
    print(f"\n══ Exercise 8.4 — Adversarial examples [{task}] ══")

    ds = CarlaBinaryDataset(data_root / "test", task,
                            transform=default_transform(train=False))

    # Collect n_vis correctly-classified positives
    samples: list[tuple[torch.Tensor, int]] = []
    model.eval()
    with torch.no_grad():
        for idx in range(len(ds)):
            x, y = ds[idx]
            if int(y) != 1:
                continue
            pred = model(x.unsqueeze(0).to(device)).argmax(dim=1).item()
            if pred == 1:
                samples.append((x, int(y)))
            if len(samples) >= n_vis:
                break

    if not samples:
        print(f"  ⚠  no correctly-classified positives found for task={task}. Skipping.")
        return

    n_cols = 1 + len(EPSILONS)   # clean + one per ε
    n_rows = len(samples)
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(n_cols * 2.9, n_rows * 2.9))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    col_headers = ["clean"] + [f"ε = {e}" for e in EPSILONS]
    for c, h in enumerate(col_headers):
        axes[0, c].set_title(h, fontsize=10, fontweight="bold")

    for row, (x_clean, y_true) in enumerate(samples):
        xb = x_clean.unsqueeze(0).to(device)
        yb = torch.tensor([y_true])

        axes[row, 0].imshow(to_uint8(x_clean))
        axes[row, 0].axis("off")
        axes[row, 0].set_ylabel(f"sample {row+1}", fontsize=8)

        for col, eps in enumerate(EPSILONS, start=1):
            x_adv = fgsm(model, xb, yb, eps, device)[0]
            with torch.no_grad():
                pred_adv = model(x_adv.unsqueeze(0)).argmax(dim=1).item()
            linf = linf_pixels(x_clean, x_adv)
            fooled = pred_adv != y_true

            ax = axes[row, col]
            ax.imshow(to_uint8(x_adv))
            ax.axis("off")
            label_color = "#e45756" if fooled else "#54a24b"
            label_text  = "FOOLED" if fooled else "correct"
            ax.set_xlabel(f"L∞={linf:.3f}  [{label_text}]",
                          fontsize=7, color=label_color)

    fig.suptitle(f"FGSM adversarial examples — {task} classifier", y=1.01)
    fig.tight_layout()
    out_path = out_dir / f"8.4_adversarial_examples_{task}.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")
    print(f"  Note: perturbations become visible to humans around ε=0.05–0.10")


# ── Exercise 8.5 ─────────────────────────────────────────────────────────────

def evaluate_recall_pair(model: torch.nn.Module,
                         loader: DataLoader,
                         eps: float,
                         device: torch.device) -> tuple[float, float]:
    """
    Returns (clean_recall, adv_recall) for a given ε.
    Adversarial examples are generated on-the-fly.
    """
    model.eval()
    y_true_all, y_clean_all, y_adv_all = [], [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        with torch.no_grad():
            pred_clean = model(x).argmax(dim=1)

        x_adv = fgsm(model, x, y, eps, device)
        with torch.no_grad():
            pred_adv = model(x_adv).argmax(dim=1)

        y_true_all.extend(y.cpu().numpy())
        y_clean_all.extend(pred_clean.cpu().numpy())
        y_adv_all.extend(pred_adv.cpu().numpy())

    y_true  = np.array(y_true_all)
    y_clean = np.array(y_clean_all)
    y_adv   = np.array(y_adv_all)

    return (recall_score(y_true, y_clean, zero_division=0),
            recall_score(y_true, y_adv,   zero_division=0))


def run_8_5(models_tasks: list[tuple[torch.nn.Module, str]],
            data_root: Path,
            out_dir: Path,
            device: torch.device,
            batch_size: int,
            n_samples: int) -> None:
    print(f"\n══ Exercise 8.5 — Recall Drop (n_samples={n_samples} per model) ══")

    col_w   = 16
    eps_str = "  ".join(f"ε={e:<5} drop" for e in EPSILONS)
    header  = f"{'Model':<{col_w}}  {'Clean':>7}  {eps_str}"
    sep     = "─" * len(header)
    lines   = ["Exercise 8.5 — FGSM Recall Drop", "=" * len(sep), header, sep]

    plot_data: dict[str, dict] = {}

    for model, task in models_tasks:
        loader = make_loader(data_root, task, batch_size, n_samples)

        # Compute clean recall once (eps=0 returns the clean tensor unchanged)
        clean_recall, _ = evaluate_recall_pair(model, loader, 0.0, device)
        row = f"{task:<{col_w}}  {clean_recall:>7.4f}"
        plot_data[task] = {"clean": clean_recall, "adv": {}}

        for eps in EPSILONS:
            _, adv_recall = evaluate_recall_pair(model, loader, eps, device)
            drop = clean_recall - adv_recall
            row += f"  {adv_recall:.4f}  {drop:+.4f}"
            plot_data[task]["adv"][eps] = adv_recall
            print(f"  [{task}] ε={eps}: clean={clean_recall:.4f}  "
                  f"adv={adv_recall:.4f}  drop={drop:+.4f}")

        lines.append(row)

    summary = "\n".join(lines)
    print("\n" + summary)
    txt_path = out_dir / "8.5_recall_drop_table.txt"
    txt_path.write_text(summary)
    print(f"\n  wrote {txt_path}")

    # ── Grouped bar chart ─────────────────────────────────────────────────────
    task_names = list(plot_data.keys())
    n_models   = len(task_names)
    x_pos      = np.arange(n_models)
    n_bars     = 1 + len(EPSILONS)          # clean + one per ε
    bar_w      = 0.7 / n_bars
    colors     = ["#888888", "#4c78a8", "#f58518", "#e45756"]

    fig, ax = plt.subplots(figsize=(max(7, n_models * 3), 5))

    for i, (label, color) in enumerate(
            zip(["clean"] + [f"ε={e}" for e in EPSILONS], colors)):
        offsets = x_pos + (i - n_bars / 2 + 0.5) * bar_w
        if label == "clean":
            vals = [plot_data[t]["clean"] for t in task_names]
        else:
            eps = EPSILONS[i - 1]
            vals = [plot_data[t]["adv"][eps] for t in task_names]
        ax.bar(offsets, vals, bar_w, color=color, label=label, zorder=2)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(task_names, fontsize=11)
    ax.set_ylabel("Recall")
    ax.set_ylim(0, 1.12)
    ax.set_title("FGSM recall drop — clean vs adversarial inputs")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    fig.tight_layout()
    plot_path = out_dir / "8.5_recall_drop_plot.png"
    fig.savefig(plot_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {plot_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Sheet 8 — FGSM adversarial attacks on CARLA classifiers")
    ap.add_argument("--data-root",  required=True,  type=Path)
    ap.add_argument("--checkpoint", required=True,  nargs="+", type=Path,
                    help="One or more .pt checkpoint files (one per classifier)")
    ap.add_argument("--out-dir",    default=Path("outputs"), type=Path)
    ap.add_argument("--batch-size", default=32,  type=int)
    ap.add_argument("--n-samples",  default=100, type=int,
                    help="Max test images to use in 8.5 (default 100)")
    ap.add_argument("--device",     default="auto")
    ap.add_argument("--exercises",  default="8.4,8.5",
                    help="Comma-separated exercises to run")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu")) if args.device == "auto" \
              else torch.device(args.device)
    print(f"device: {device}")

    models_tasks: list[tuple[torch.nn.Module, str]] = []
    for ckpt in args.checkpoint:
        m, t = load_model(ckpt, device)
        models_tasks.append((m, t))
        print(f"  loaded '{t}' ← {ckpt.name}")

    to_run = {e.strip() for e in args.exercises.split(",")}

    if "8.4" in to_run:
        for m, t in models_tasks:
            run_8_4(m, t, args.data_root, args.out_dir, device)

    if "8.5" in to_run:
        run_8_5(models_tasks, args.data_root, args.out_dir,
                device, args.batch_size, args.n_samples)

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
