
# Sheet 5 — Testing LLMs & Agents

## Files

| File | Exercise | Purpose |
|---|---|---|
| `temperature_scaling.py` | 5.4 | Temperature scaling evaluation on pedestrian detector |
| `backdoor_attack.py` | 5.5 | Backdoor attack: poison, retrain, evaluate ASR |
| `report_sheet5.md` | all | Prose answers to all questions (5.1–5.5) |


# Exercise 5.4 — temperature scaling
!python /content/sheet5/temperature_scaling.py \
    --data-root /content/data/MyDataset \
    --checkpoint /content/drive/MyDrive/MyDataset/sheet3_results/checkpoints/pedestrian.pt
```

```python
# Exercise 5.5 — backdoor attack (~20 min on T4, retrains from scratch)
!python /content/sheet5/backdoor_attack.py \
    --data-root /content/data/MyDataset \
    --clean-checkpoint /content/drive/MyDrive/MyDataset/sheet3_results/checkpoints/pedestrian.pt \
    --epochs 5
```

## Outputs

```

  5.4_temperature_accuracy.txt      accuracy/recall/ECE table per T
  5.4_confidence_distributions.png  confidence histograms for T=0.5,1.0,2.0
  5.4_calibration_curve.png         reliability diagram per temperature
  5.4_constraint_analysis.txt       how T affects the θ=0.6 safety constraint

  5.5_trigger_example.png           clean vs triggered image side-by-side
  5.5_backdoor_results.txt          clean recall + ASR numbers
  5.5_attack_summary.png            bar chart comparing all three conditions

checkpoints/
  pedestrian_backdoored.pt          poisoned model weights
```


