# Sheet 4 — Model Testing and Validation

## Files

| File | Exercise | Purpose |
|---|---|---|
| `odd_coverage.py` | 4.5 | k-projection ODD coverage for k ∈ {1,2,3} |
| `per_class_eval.py` | 4.7 | Per-class precision/recall/F1/confusion matrix |
| `kprojection.py` | 4.5 | Bundled from github.com/kkirchheim/odd-coverage |




## Outputs

```
outputs/
  4.5_odd_coverage.txt        coverage numbers per k per split
  4.5_odd_coverage.png        bar chart
  4.7_metrics_summary.txt     accuracy/precision/recall/F1 table
  4.7_metrics.json            machine-readable metrics
  4.7_confusion_{task}.png    confusion matrix per model
  4.7_precision_recall_bar.png  grouped bar chart with recall threshold line
  4.7_pedestrian_pr_curve.png   PR curve with 0.90 recall target marked

