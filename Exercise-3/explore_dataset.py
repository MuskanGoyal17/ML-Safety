"""
Exercise 3.4 — Dataset Exploration

Run:
    python explore_dataset.py --data-root /path/to/dataset

Where /path/to/dataset contains: train/, validation/, test/ (each with
labels.csv and rgb-front/).

Outputs:
    outputs/3.4_class_distribution.png
    outputs/3.4_example_grid.png
    outputs/3.4_summary.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from dataset import LABEL_COLUMNS, frame_to_path

SPLITS = ("train", "validation", "test")


def load_split(data_root: Path, split: str) -> pd.DataFrame | None:
    csv = data_root / split / "labels.csv"
    if not csv.exists():
        print(f"[skip] no labels.csv at {csv}")
        return None
    df = pd.read_csv(csv)
    df["__split__"] = split
    return df


def summarize(df: pd.DataFrame, name: str, out_lines: list[str]) -> None:
    out_lines.append(f"\n=== {name} (n={len(df)}) ===")
    for col in LABEL_COLUMNS:
        pos = int(df[col].sum())
        out_lines.append(f"  {col:20s} positive: {pos:6d}  ({pos/len(df):6.1%})")
    joint = df.groupby(list(LABEL_COLUMNS)).size().sort_values(ascending=False)
    out_lines.append("  joint label counts:")
    for combo, n in joint.items():
        tl, ped, veh = combo
        out_lines.append(
            f"    TL={int(tl)} PED={int(ped)} VEH={int(veh)} : {n:6d}  ({n/len(df):5.1%})"
        )


def plot_class_distribution(splits: dict[str, pd.DataFrame], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    width = 0.25
    x = np.arange(len(LABEL_COLUMNS))
    short = {"has_traffic_light": "traffic light",
             "has_pedestrian": "pedestrian",
             "has_vehicle": "vehicle"}
    for ax, (split_name, df) in zip(axes, splits.items()):
        pos = [df[c].mean() for c in LABEL_COLUMNS]
        neg = [1 - p for p in pos]
        ax.bar(x - width/2, pos, width, label="positive", color="#4c78a8")
        ax.bar(x + width/2, neg, width, label="negative", color="#e45756")
        ax.set_xticks(x)
        ax.set_xticklabels([short[c] for c in LABEL_COLUMNS], rotation=20)
        ax.set_title(f"{split_name}  (n={len(df)})")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("fraction of frames")
    axes[0].legend(loc="upper right", fontsize=9)
    fig.suptitle("Class balance per split", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_example_grid(train_df: pd.DataFrame, split_root: Path,
                      out_path: Path, rng: np.random.Generator) -> None:
    """One example per label combination, up to 8."""
    combos = (train_df.groupby(list(LABEL_COLUMNS)).size()
              .sort_values(ascending=False).head(8).index.tolist())
    n = len(combos)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 2.6))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for ax, combo in zip(axes, combos):
        tl, ped, veh = combo
        sub = train_df[(train_df["has_traffic_light"] == tl) &
                       (train_df["has_pedestrian"] == ped) &
                       (train_df["has_vehicle"] == veh)]
        if len(sub) == 0:
            continue
        row = sub.iloc[int(rng.integers(0, len(sub)))]
        path = frame_to_path(split_root, int(row["frame"]))
        try:
            img = Image.open(path).convert("RGB")
            ax.imshow(np.asarray(img))
        except FileNotFoundError:
            ax.text(0.5, 0.5, "missing", ha="center", va="center")
        tag = (f"TL={'Y' if tl else 'n'}  "
               f"PED={'Y' if ped else 'n'}  "
               f"VEH={'Y' if veh else 'n'}")
        ax.set_title(tag, fontsize=9)
    fig.suptitle("Example frame per label combination (train split)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, type=Path,
                    help="Folder containing train/, validation/, test/")
    ap.add_argument("--out-dir", default=Path("outputs"), type=Path)
    ap.add_argument("--seed", default=0, type=int)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    splits = {}
    for s in SPLITS:
        df = load_split(args.data_root, s)
        if df is not None:
            splits[s] = df

    if "train" not in splits:
        raise SystemExit("Need at least a train split.")

    summary_lines: list[str] = ["CARLA dataset exploration"]
    for name, df in splits.items():
        summarize(df, name, summary_lines)

    summary_path = args.out_dir / "3.4_summary.txt"
    summary_path.write_text("\n".join(summary_lines))
    print(f"  wrote {summary_path}")
    print("\n".join(summary_lines))

    plot_class_distribution(splits, args.out_dir / "3.4_class_distribution.png")
    plot_example_grid(splits["train"], args.data_root / "train",
                      args.out_dir / "3.4_example_grid.png", rng)


if __name__ == "__main__":
    main()
