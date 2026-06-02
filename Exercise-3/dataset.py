"""
CARLA binary-classification dataset.

The CSV has columns:
  frame, has_traffic_light, has_pedestrian, has_vehicle,
  px_traffic_light, px_pedestrian, px_vehicle

Images are assumed to live at:  <split_root>/rgb-front/{frame:06d}.png
If your filenames differ, change `frame_to_path` below.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

LABEL_COLUMNS = ("has_traffic_light", "has_pedestrian", "has_vehicle")
Task = Literal["traffic_light", "pedestrian", "vehicle"]
TASK_TO_COL = {
    "traffic_light": "has_traffic_light",
    "pedestrian": "has_pedestrian",
    "vehicle": "has_vehicle",
}


def frame_to_path(split_root: Path, frame: int) -> Path:
    """Resolve a frame id to its image path. Adjust if your layout differs.

    CARLA dataset layout: <split>/rgb-front/{frame:06d}.jpg
    """
    return split_root / "rgb-front" / f"{frame:06d}.jpg"


def default_transform(image_size: int = 224, train: bool = False):
    """ImageNet-normalized transform; mild geometric aug only if train=True.

    Note: we DO NOT use color jitter or flips that change semantics — flipping
    a scene horizontally is fine for these labels but we keep it conservative
    and let the model learn from real diversity.
    """
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


class CarlaBinaryDataset(Dataset):
    """One CARLA split, returning (image, label) for a single binary task.

    Args:
        split_root: e.g. .../train/  (the folder that contains rgb-front/ and labels.csv)
        task: which of the three classifiers we're training
        transform: torchvision transform; if None, default_transform(train=False)
        csv_name: defaults to "labels.csv"
    """

    def __init__(
        self,
        split_root: str | Path,
        task: Task,
        transform=None,
        csv_name: str = "labels.csv",
    ) -> None:
        self.split_root = Path(split_root)
        self.task = task
        self.label_col = TASK_TO_COL[task]
        self.transform = transform or default_transform(train=False)

        csv_path = self.split_root / csv_name
        if not csv_path.exists():
            raise FileNotFoundError(
                f"labels.csv not found at {csv_path}. Pass split_root pointing "
                "at the folder containing labels.csv and rgb-front/."
            )
        self.df = pd.read_csv(csv_path)

        # Quick sanity: drop rows whose image is missing on disk so training
        # doesn't crash mid-epoch.
        exists_mask = self.df["frame"].apply(
            lambda f: frame_to_path(self.split_root, int(f)).exists()
        )
        n_missing = (~exists_mask).sum()
        if n_missing:
            print(f"[{self.split_root.name}/{task}] dropping {n_missing} "
                  f"rows with missing images")
            self.df = self.df[exists_mask].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = frame_to_path(self.split_root, int(row["frame"]))
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        # Cross-entropy expects LongTensor class indices, 0/1
        label = torch.tensor(int(bool(row[self.label_col])), dtype=torch.long)
        return img, label

    def positive_fraction(self) -> float:
        return float(self.df[self.label_col].mean())


def class_weights(dataset: CarlaBinaryDataset) -> torch.Tensor:
    """Inverse-frequency weights for CE loss, shape (2,)."""
    p = dataset.positive_fraction()
    p = max(p, 1e-3)
    p = min(p, 1 - 1e-3)
    # weight_c = 1 / freq_c, normalized so they average to 1
    w_neg = 1.0 / (1.0 - p)
    w_pos = 1.0 / p
    s = (w_neg + w_pos) / 2.0
    return torch.tensor([w_neg / s, w_pos / s], dtype=torch.float32)
