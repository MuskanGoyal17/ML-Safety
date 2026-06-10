# Exercise Sheet 9

The pedestrian detector checkpoint was used for all OOD experiments.

---

## Dataset Layout

The expected dataset structure is:

```text
<data-root>/
├── train/
├── test/
├── ood_fog/
├── ood_night/
└── ood_town/
```

The pedestrian detector was used for Exercises 9.4–9.7.

---

## Outputs


outputs/
├── 9.4_sample_grid.png
├── 9.4_confidence_table.txt
├── 9.6_msp_score_distribution.png
├── 9.6_auroc_table.txt
├── 9.7_mahal_score_distribution.png
└── 9.7_auroc_comparison.txt

---

## Results Summary

### Mean Softmax Confidence

| Condition       | Mean MSP |
| --------------- | -------- |
| In-distribution | 0.7623   |
| Fog             | 0.7441   |
| Night           | 0.6498   |
| Town            | 0.7592   |

### MSP AUROC

| Scenario | AUROC  |
| -------- | ------ |
| Fog      | 0.5421 |
| Night    | 0.7243 |
| Town     | 0.5092 |
| Combined | 0.5919 |

### Mahalanobis AUROC

| Scenario | AUROC  |
| -------- | ------ |
| Fog      | 0.9818 |
| Night    | 0.9998 |
| Town     | 0.8197 |

The Mahalanobis detector substantially outperformed the MSP baseline across all OOD scenarios.
