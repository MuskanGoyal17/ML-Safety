# Sheet 6 — Explainability

## Overview

This sheet applies **Grad-CAM** (Gradient-weighted Class Activation Mapping) to the three ResNet-18 binary classifiers trained in Sheet 3 (`traffic_light`, `pedestrian`, `vehicle`). The goal is to understand *where* each model looks when making a prediction, diagnose failure modes on misclassified images, and assess how explanation quality degrades under out-of-distribution (OOD) conditions (fog, night).


## Method: Grad-CAM

Grad-CAM hooks into the last convolutional layer (`layer4[-1]` of ResNet-18). For each input image:

1. Run a forward pass → get the class score for the predicted label.
2. Backpropagate to compute gradients of that score w.r.t. every feature map channel.
3. Global-average-pool the gradients → importance weights α_k per channel.
4. Weighted sum of feature maps + ReLU → coarse 7×7 heatmap.
5. Bilinear upsample to original image resolution. Normalise to [0, 1].

**Why Grad-CAM** over the alternatives:

| Method | Pros | Cons |
|---|---|---|
| **Grad-CAM** ✓ | No architecture change, fast (1 fwd + 1 bwd), stable, class-discriminative | Low spatial resolution (7×7) |
| Saliency | Very fast | Noisy, gradient saturation in deep nets |
| Occlusion | Causal, no gradient issues | O(H×W) forward passes — very slow |
| CAM | High quality | Requires GlobalAvgPool head — needs architecture change |

---

## Requirements

```bash
pip install torch torchvision matplotlib numpy pillow scikit-learn

```
## Outputs

```
outputs/
  6.5_gradcam_correct_traffic_light.png   2 correctly classified TL images + heatmaps
  6.5_gradcam_correct_pedestrian.png      2 correctly classified pedestrian images + heatmaps
  6.5_gradcam_correct_vehicle.png         1 correctly classified vehicle image + heatmap
  6.6_gradcam_misclassified.png           3 misclassified images + heatmaps (2 ped FN, 1 veh FN)
  6.6_gradcam_ood_{name}.png              1 figure per OOD split with heatmaps + OOD accuracy
```

Each figure shows: **original image (top row) | Grad-CAM overlay (bottom row)**, with green title = correct, red title = misclassified.

---

## Key Results

### Ex 6.5 — Correctly Classified Images

| Model | Highlighted region | Matches object? |
|---|---|---|
| Traffic Light | Upper-centre / upper-left (where TL appears) |  Broadly correct |
| Pedestrian | Lower road / car bonnet (no pedestrian zone) | Correct for true negatives |
| Vehicle | Directly on the vehicle |  Best localisation |

### Ex 6.5 — Misclassified Images

All three failures share the same root cause: the model attended to **background regions** (sky, lane markings, treeline, kerb) instead of the target object. This indicates the model learned scene-level statistics rather than robust object features.

### Ex 6.6 — OOD Accuracy

| Model | In-distribution | Fog | Night |
|---|---|---|---|
| Traffic Light | ~0.85 | 0.439 | 0.270  |
| Pedestrian | ~0.87 | 0.780  | 0.485  |
| Vehicle | ~0.82 | 0.545  | 0.736 ~ |

### Ex 6.6 — Spurious Features Observed

- **Night — Traffic Light:** Model attends to streetlights and road reflections instead of traffic lights.
- **Night — Pedestrian:** Bright puddle/reflection triggers a false positive — model relies on brightness as a pedestrian proxy.
- **Fog — all models:** Heatmaps become diffuse and spread across fog texture; no focused object attention.
- **Vehicle model at night:** Only OOD success — large high-contrast vehicle preserves correct attribution.

> **Safety insight:** Diffuse or semantically incorrect Grad-CAM maps are a *leading indicator* of distribution shift, detectable before accuracy metrics flag a problem. Runtime heatmap monitoring (e.g. checking activation centroid vs. expected object region) could serve as an early warning system.

---

