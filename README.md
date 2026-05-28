# Sheet 3 — Fundamentals

This folder contains everything for Exercise Sheet 3: dataset exploration,
training three binary classifiers, evaluation, and the ODD gap analysis.

## Files

| File | Purpose |
|---|---|
| `dataset.py` | Shared `CarlaBinaryDataset` and transforms |
| `explore_dataset.py` | Exercise 3.4 — exploration plots and summary |
| `train_and_eval.py` | Exercises 3.5 + 3.6 — training and test metrics |
| `report.md` | Prose answers to every numbered question (this file's twin) |
| `requirements.txt` | Python deps |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Assuming your dataset is laid out like

```
<data-root>/
  train/        labels.csv + rgb-front/000000.jpg ...
  validation/   labels.csv + rgb-front/...
  test/         labels.csv + rgb-front/...
```

then:

```bash
# Exercise 3.4
python explore_dataset.py --data-root /path/to/dataset

# Exercises 3.5 + 3.6
python train_and_eval.py --data-root /path/to/dataset --epochs 5
```

Add `--device cpu --batch-size 32 --epochs 3 --num-workers 2` if you don't have a GPU.

## Outputs

- `outputs/3.4_summary.txt` — split sizes, class balance, joint counts
- `outputs/3.4_class_distribution.png` — per-split class balance bars
- `outputs/3.4_example_grid.png` — one example image per label combination
- `outputs/3.5_loss_{task}.png` — training/validation loss curves
- `outputs/3.6_confusion_{task}.png` — test confusion matrices
- `outputs/3.6_test_metrics.json` + `3.6_test_summary.txt` — final metrics
- `checkpoints/{task}.pt` — best-validation model weights (used by Sheets 4–9)


