"""
Exercise Sheet 9 — OOD Detection for the CARLA Pedestrian Detector

Covers:
  9.4  Visualise distribution shift (sample grids + mean softmax confidence)
  9.6  MSP baseline OOD detection — score distribution plot + AUROC
  9.7  Feature-based detector (Mahalanobis distance) — AUROC comparison

Produces (outputs/):
  9.4_sample_grid.png
  9.4_confidence_table.txt
  9.6_msp_score_distribution.png
  9.6_auroc_table.txt
  9.7_mahal_score_distribution.png
  9.7_auroc_comparison.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

sys.path.insert(0, str(Path(__file__).parent.parent / "sheet3"))
from dataset import CarlaBinaryDataset, default_transform, frame_to_path

import pandas as pd

# ── Constants ────────────────────────────────────────────────────────────────

OOD_SCENARIOS = {
    "fog":   "ood_fog",
    "night": "ood_night",
    "town":  "ood_town",
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── Model helpers ─────────────────────────────────────────────────────────────

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


# ── Simple folder dataset (no labels.csv needed for raw images) ───────────────

class RawFolderDataset(Dataset):

    EXTS = {".jpg"}

    def __init__(self, folder: Path, transform=None) -> None:
        self.folder = Path(folder)
        self.transform = transform or default_transform(train=False)
        csv_path = self.folder / "labels.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            self.paths = []
            self.labels = []
            for _, row in df.iterrows():
                p = frame_to_path(self.folder, int(row["frame"]))
                if p.exists():
                    self.paths.append(p)
                    lbl = int(bool(row.get("has_pedestrian", 0)))
                    self.labels.append(lbl)
        else:
            # Fallback: glob images directly
            self.paths = sorted(
                p for p in self.folder.rglob("*")
                if p.suffix.lower() in self.EXTS
            )
            self.labels = [0] * len(self.paths)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]


# ── Feature extraction hook ────────────────────────────────────────────────────

def register_feature_hook(model: torch.nn.Module):
  
    features = {}

    def hook(module, input, output):
        features["feat"] = output.flatten(1)   # (B, 512) for ResNet-18

    handle = model.avgpool.register_forward_hook(hook)
    return features, handle


@torch.no_grad()
def extract_features_and_logits(
        model: torch.nn.Module,
        loader: DataLoader,
        device: torch.device):
    """Return (features [N,512], logits [N,2], labels [N])."""
    all_feats, all_logits, all_labels = [], [], []
    feat_store, handle = register_feature_hook(model)

    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        all_feats.append(feat_store["feat"].cpu().numpy())
        all_logits.append(logits.cpu().numpy())
        all_labels.append(y.numpy())

    handle.remove()
    return (np.concatenate(all_feats),
            np.concatenate(all_logits),
            np.concatenate(all_labels))


# ── MSP score ─────────────────────────────────────────────────────────────────

def msp_score(logits: np.ndarray) -> np.ndarray:
 
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    return probs.max(axis=1)


# ── Mahalanobis score ─────────────────────────────────────────────────────────

class MahalanobisDetector:
    """
    Fit a single Gaussian (mean + covariance) on in-distribution features,
    then score new samples by Mahalanobis distance.
    Higher distance → more OOD.
    """

    def __init__(self):
        self.mu: np.ndarray | None = None
        self.cov: EmpiricalCovariance | None = None

    def fit(self, features: np.ndarray) -> None:
        self.mu = features.mean(axis=0)
        self.cov = EmpiricalCovariance(assume_centered=False)
        self.cov.fit(features)

    def score(self, features: np.ndarray) -> np.ndarray:
    
        delta = features - self.mu
        return self.cov.mahalanobis(features) ** 0.5  # sqrt for stable scale


# ── Plotting helpers ──────────────────────────────────────────────────────────

def denormalise(tensor: torch.Tensor) -> np.ndarray:

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img  = (tensor * std + mean).clamp(0, 1)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def load_sample_images(folder: Path, n: int = 5) -> list[np.ndarray]:
    """Load n sample images from a split for display."""
    ds = RawFolderDataset(folder)
    indices = np.linspace(0, len(ds) - 1, n, dtype=int)
    imgs = []
    for i in indices:
        tensor, _ = ds[i]
        imgs.append(denormalise(tensor))
    return imgs


# ── Exercise 9.4 ──────────────────────────────────────────────────────────────

def run_9_4(model, device, data_root, out_dir, batch_size):
    print("\n══ Exercise 9.4 — Visualising Distribution Shift ══")

    # ── Sample grid ──────────────────────────────────────────────────────────
    rows = {
        "In-dist (sunny)": data_root / "test",
        "OOD — fog":       data_root / "ood_fog",
        "OOD — night":     data_root / "ood_night",
        "Different town":  data_root / "ood_town",
    }
    n_cols = 5
    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.8, n_rows * 2.4))

    for row_idx, (label, folder) in enumerate(rows.items()):
        if not folder.exists():
            print(f"  ⚠  folder not found: {folder} — skipping")
            for c in range(n_cols):
                axes[row_idx, c].axis("off")
                axes[row_idx, c].set_title("N/A", fontsize=7)
            axes[row_idx, 0].set_ylabel(label, fontsize=9, rotation=0,
                                         labelpad=80, va="center")
            continue
        imgs = load_sample_images(folder, n=n_cols)
        for c, img in enumerate(imgs):
            axes[row_idx, c].imshow(img)
            axes[row_idx, c].axis("off")
        axes[row_idx, 0].set_ylabel(label, fontsize=9, rotation=0,
                                     labelpad=80, va="center")

    fig.suptitle("Distribution shift — sample images across conditions", y=1.01)
    fig.tight_layout()
    grid_path = out_dir / "9.4_sample_grid.png"
    fig.savefig(grid_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {grid_path}")

    # ── Mean softmax confidence per condition ─────────────────────────────────
    splits = {"in_distribution": data_root / "test",
              **{k: data_root / v for k, v in OOD_SCENARIOS.items()}}
    conf_lines = [
        "Exercise 9.4 — Mean Softmax Confidence per Condition",
        "=" * 55,
        f"{'Condition':<20s}  {'N images':>8s}  {'Mean MSP':>9s}  {'Std MSP':>8s}",
        "-" * 55,
    ]
    for name, folder in splits.items():
        if not folder.exists():
            conf_lines.append(f"  {name:<20s}  (folder missing)")
            continue
        ds = RawFolderDataset(folder)
        loader = DataLoader(ds, batch_size=batch_size,
                            shuffle=False, num_workers=2)
        _, logits, _ = extract_features_and_logits(model, loader, device)
        msp = msp_score(logits)
        conf_lines.append(
            f"{name:<20s}  {len(msp):>8d}  {msp.mean():>9.4f}  {msp.std():>8.4f}"
        )
        print(f"  {name}: mean MSP = {msp.mean():.4f}  std = {msp.std():.4f}")

    conf_text = "\n".join(conf_lines)
    conf_path = out_dir / "9.4_confidence_table.txt"
    conf_path.write_text(conf_text)
    print(f"  wrote {conf_path}")
    print("\n" + conf_text)


# ── Exercise 9.6 ──────────────────────────────────────────────────────────────

def run_9_6(model, device, data_root, out_dir, batch_size):
    print("\n══ Exercise 9.6 — MSP Baseline OOD Detection ══")

    # In-distribution logits
    id_ds = RawFolderDataset(data_root / "test")
    id_loader = DataLoader(id_ds, batch_size=batch_size,
                           shuffle=False, num_workers=2)
    _, id_logits, _ = extract_features_and_logits(model, id_loader, device)
    id_msp = msp_score(id_logits)

    auroc_lines = [
        "Exercise 9.6 — MSP Baseline AUROC",
        "=" * 45,
        f"{'OOD scenario':<15s}  {'N OOD':>6s}  {'AUROC':>7s}",
        "-" * 35,
    ]

    all_ood_msp, all_ood_labels = [], []

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
    colors = {"fog": "#4c78a8", "night": "#e45756", "town": "#54a24b"}

    for ax, (scenario, subfolder) in zip(axes, OOD_SCENARIOS.items()):
        folder = data_root / subfolder
        if not folder.exists():
            print(f"  ⚠  {folder} missing — skipping {scenario}")
            ax.set_title(f"{scenario} (missing)")
            ax.axis("off")
            continue

        ood_ds = RawFolderDataset(folder)
        ood_loader = DataLoader(ood_ds, batch_size=batch_size,
                                shuffle=False, num_workers=2)
        _, ood_logits, _ = extract_features_and_logits(model, ood_loader, device)
        ood_msp = msp_score(ood_logits)

        # AUROC: label 1 = in-dist, 0 = OOD
        y_score = np.concatenate([id_msp, ood_msp])
        y_true  = np.concatenate([np.ones(len(id_msp)),
                                   np.zeros(len(ood_msp))])
        auroc = roc_auc_score(y_true, y_score)
        auroc_lines.append(
            f"{scenario:<15s}  {len(ood_msp):>6d}  {auroc:>7.4f}"
        )
        print(f"  {scenario}: AUROC = {auroc:.4f}")

        all_ood_msp.append(ood_msp)
        all_ood_labels.extend([scenario] * len(ood_msp))

        # Plot
        ax.hist(id_msp,  bins=40, alpha=0.65, density=True,
                color="#888888", label="in-dist (sunny)")
        ax.hist(ood_msp, bins=40, alpha=0.65, density=True,
                color=colors[scenario], label=f"OOD ({scenario})")
        ax.set_title(f"{scenario}\nAUROC = {auroc:.3f}", fontsize=10)
        ax.set_xlabel("MSP score")
        ax.set_ylabel("density")
        ax.legend(fontsize=8)

    # Combined AUROC (all OOD vs in-dist)
    if all_ood_msp:
        all_ood = np.concatenate(all_ood_msp)
        y_score_all = np.concatenate([id_msp, all_ood])
        y_true_all  = np.concatenate([np.ones(len(id_msp)),
                                       np.zeros(len(all_ood))])
        auroc_all = roc_auc_score(y_true_all, y_score_all)
        auroc_lines += ["-" * 35,
                        f"{'ALL combined':<15s}  {len(all_ood):>6d}  {auroc_all:>7.4f}"]
        print(f"  ALL combined: AUROC = {auroc_all:.4f}")

    fig.suptitle("MSP OOD score distribution — in-dist vs OOD", y=1.01)
    fig.tight_layout()
    dist_path = out_dir / "9.6_msp_score_distribution.png"
    fig.savefig(dist_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {dist_path}")

    auroc_text = "\n".join(auroc_lines)
    auroc_path = out_dir / "9.6_auroc_table.txt"
    auroc_path.write_text(auroc_text)
    print(f"  wrote {auroc_path}")
    print("\n" + auroc_text)


# ── Exercise 9.7 ──────────────────────────────────────────────────────────────

def run_9_7(model, device, data_root, out_dir, batch_size):
    print("\n══ Exercise 9.7 — Mahalanobis Distance OOD Detector ══")

    # Extract in-distribution TRAINING features to fit the detector
    id_train_ds = RawFolderDataset(data_root / "train")
    id_train_loader = DataLoader(id_train_ds, batch_size=batch_size,
                                 shuffle=False, num_workers=2)
    print("  extracting in-distribution train features …")
    id_train_feats, _, _ = extract_features_and_logits(
        model, id_train_loader, device)

    # Fit Mahalanobis detector on training features
    detector = MahalanobisDetector()
    detector.fit(id_train_feats)
    print(f"  fitted Mahalanobis detector on {len(id_train_feats)} train samples")

    # In-distribution TEST features + MSP (for fair comparison with 9.6)
    id_test_ds = RawFolderDataset(data_root / "test")
    id_test_loader = DataLoader(id_test_ds, batch_size=batch_size,
                                shuffle=False, num_workers=2)
    print("  extracting in-distribution test features …")
    id_test_feats, id_test_logits, _ = extract_features_and_logits(
        model, id_test_loader, device)
    id_msp   = msp_score(id_test_logits)
    # For Mahalanobis: negate distance so higher = more in-distribution (like MSP)
    id_mahal = -detector.score(id_test_feats)

    comp_lines = [
        "Exercise 9.7 — Mahalanobis vs MSP AUROC Comparison",
        "=" * 55,
        f"{'OOD scenario':<15s}  {'MSP AUROC':>10s}  {'Mahal AUROC':>12s}  {'Delta':>7s}",
        "-" * 55,
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    colors_id  = "#888888"
    colors_ood = {"fog": "#4c78a8", "night": "#e45756", "town": "#54a24b"}

    for ax, (scenario, subfolder) in zip(axes, OOD_SCENARIOS.items()):
        folder = data_root / subfolder
        if not folder.exists():
            print(f"  ⚠  {folder} missing — skipping {scenario}")
            ax.set_title(f"{scenario} (missing)")
            ax.axis("off")
            continue

        ood_ds = RawFolderDataset(folder)
        ood_loader = DataLoader(ood_ds, batch_size=batch_size,
                                shuffle=False, num_workers=2)
        ood_feats, ood_logits, _ = extract_features_and_logits(
            model, ood_loader, device)

        ood_msp   = msp_score(ood_logits)
        ood_mahal = -detector.score(ood_feats)

        y_true = np.concatenate([np.ones(len(id_test_feats)),
                                  np.zeros(len(ood_feats))])

        auroc_msp   = roc_auc_score(
            y_true, np.concatenate([id_msp,   ood_msp]))
        auroc_mahal = roc_auc_score(
            y_true, np.concatenate([id_mahal, ood_mahal]))
        delta = auroc_mahal - auroc_msp

        comp_lines.append(
            f"{scenario:<15s}  {auroc_msp:>10.4f}  {auroc_mahal:>12.4f}  "
            f"{delta:>+7.4f}"
        )
        print(f"  {scenario}: MSP={auroc_msp:.4f}  Mahal={auroc_mahal:.4f}  "
              f"Δ={delta:+.4f}")

        # Distribution plot (Mahalanobis scores)
        ax.hist(id_mahal,  bins=40, alpha=0.65, density=True,
                color=colors_id, label="in-dist (sunny)")
        ax.hist(ood_mahal, bins=40, alpha=0.65, density=True,
                color=colors_ood[scenario], label=f"OOD ({scenario})")
        ax.set_title(
            f"{scenario}\nMahal AUROC={auroc_mahal:.3f}  "
            f"MSP={auroc_msp:.3f}", fontsize=9)
        ax.set_xlabel("−Mahalanobis distance (↑ = more in-dist)")
        ax.set_ylabel("density")
        ax.legend(fontsize=8)

    fig.suptitle("Mahalanobis OOD score distribution", y=1.01)
    fig.tight_layout()
    mahal_path = out_dir / "9.7_mahal_score_distribution.png"
    fig.savefig(mahal_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {mahal_path}")

    comp_text = "\n".join(comp_lines)
    comp_path = out_dir / "9.7_auroc_comparison.txt"
    comp_path.write_text(comp_text)
    print(f"  wrote {comp_path}")
    print("\n" + comp_text)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Sheet 9 OOD detection experiments")
    ap.add_argument("--data-root",   required=True, type=Path)
    ap.add_argument("--checkpoint",  required=True, type=Path)
    ap.add_argument("--out-dir",     default=Path("outputs"), type=Path)
    ap.add_argument("--batch-size",  default=64, type=int)
    ap.add_argument("--device",      default="auto")
    ap.add_argument("--exercises",   default="9.4,9.6,9.7",
                    help="Comma-separated list of exercises to run")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu")) if args.device == "auto" \
              else torch.device(args.device)
    print(f"device: {device}")

    model = load_model(args.checkpoint, device)

    to_run = {e.strip() for e in args.exercises.split(",")}

    if "9.4" in to_run:
        run_9_4(model, device, args.data_root, args.out_dir, args.batch_size)
    if "9.6" in to_run:
        run_9_6(model, device, args.data_root, args.out_dir, args.batch_size)
    if "9.7" in to_run:
        run_9_7(model, device, args.data_root, args.out_dir, args.batch_size)



if __name__ == "__main__":
    main()
