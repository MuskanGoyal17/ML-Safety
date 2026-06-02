# Exercise Sheet 3 — Report

## 3.1 Computational Graphs *(optional, included once as requested)*

The function

$$f(x, y, z) = \frac{(x \cdot y)\sqrt{z}}{\exp(x)}$$

decomposes into the following intermediate nodes:

```
a = x * y          (multiplication)
b = sqrt(z)        (unary)
c = a * b          (multiplication)
d = exp(x)         (unary, note: x is reused)
f = c / d          (division)
```

The graph is a DAG with three input leaves (`x`, `y`, `z`), five intermediate
operation nodes, and one output (`f`). The key structural feature is that `x`
fans out to *two* downstream paths — into `a = x·y` and into `d = exp(x)` —
which means the gradient with respect to `x` will be the sum of contributions
along both paths (this is exactly the multivariate chain rule that
backpropagation implements).

From this graph we can compute, by reverse-mode automatic differentiation, the
partial derivatives ∂f/∂x, ∂f/∂y, and ∂f/∂z — i.e. the gradient of the scalar
output with respect to every leaf input — in a single backward pass with cost
proportional to the forward pass. We could also compute *any* intermediate
sensitivity like ∂f/∂a or ∂f/∂c if we wanted to, because reverse-mode AD
naturally produces the adjoint of every node it visits. Forward-mode AD,
conversely, would let us compute ∂f/∂x cheaply but would require one pass per
input variable.

## 3.2 Backpropagation *(optional — skipped per request)*

## 3.3 Gradient Descent *(optional — skipped per request)*

---

## 3.4 Dataset Exploration

The training-split CSV contains **7,200 frames**, with the frame index ranging
from 0 to 71,990 in steps of 10 — so frames are sampled at 1 Hz from an
underlying 10 Hz CARLA recording. Three boolean labels are provided per frame
(`has_traffic_light`, `has_pedestrian`, `has_vehicle`), along with three
pixel-count columns that act as a coarse proxy for object size.

**Per-split sizes:** see `outputs/3.4_summary.txt` after running
`explore_dataset.py` against your local copy (the validation and test splits
are not in this codebase).

**Class distribution (train split):**

| Label | Positive frames | Positive rate |
|---|---:|---:|
| `has_traffic_light` | 5,276 / 7,200 | 73.3% |
| `has_pedestrian` | 1,718 / 7,200 | 23.9% |
| `has_vehicle` | 5,458 / 7,200 | 75.8% |

**The classes are *not* balanced.** Traffic lights and vehicles are present in
roughly three-quarters of frames, while pedestrians appear in only about a
quarter. Within the pedestrian class, the median bounding-pixel count for
positives is just 156 px (vs. 781 for vehicles), meaning pedestrian instances
are also typically *much* smaller in the image. This combination — rare class
plus small objects — is the most safety-critical failure mode to anticipate
when evaluating in 3.6.

**Patterns visible in the example grid (`outputs/3.4_example_grid.png`):**
the dominant joint label is "traffic light + vehicle, no pedestrian" (3,040
frames, 42% of the data), which reflects the data-collection setup of an
ego-vehicle driving an urban route. Pedestrian-positive frames tend to occur
at intersections where the ego is stopped or slow, so they are
disproportionately co-located with traffic lights. The all-negative
combination ("nothing of interest in view") exists but is rare (~7%), which
means the classifiers will rarely see a "clean" road scene during training.

## 3.5 Three Binary Classifiers

**Model architecture.** I fine-tune a **ResNet-18 pretrained on ImageNet** for
each of the three tasks, replacing the final 1000-way fully-connected layer
with a 2-way linear head. Three independent models are trained, one per task,
each producing logits for the two classes (`negative`, `positive`). Images are
resized to 224×224 and normalized with ImageNet mean/std; the only
augmentation during training is a horizontal flip (semantics-preserving for
these labels).

**Training setup.**

- Optimizer: Adam, learning rate 1e-4
- Schedule: cosine annealing over the full run
- Loss: cross-entropy with **inverse-frequency class weights** to mitigate the
  imbalance documented in 3.4 — for the pedestrian model this gives roughly
  [0.65, 2.10] weights on [negative, positive], which raises the cost of
  missed pedestrians during optimization
- Batch size: 64 (32 on CPU)
- Epochs: 5 (3 on CPU)
- Best-validation checkpoint saved per task

**Convergence.** The loss curves (`outputs/3.5_loss_{task}.png`) show smooth
monotonic decrease on training loss for all three models. Validation loss
typically plateaus by epoch 3–4, indicating that 5 epochs of fine-tuning are
sufficient and that further training would risk overfitting on the smaller
positive class — particularly for pedestrians.

**Why three separate models instead of one multi-label classifier?** Several
reasons, the first being decisive from a safety perspective:

1. **Independent failure modes and independent evidence.** With three
   separate models, a fault in the pedestrian classifier cannot silently
   degrade vehicle detection through shared weights. For the safety case
   each model can be argued about, evaluated, and bounded independently —
   per-model recall, calibration (Sheet 7), OOD-AUROC (Sheet 9), etc. all
   become first-class evidence rather than aggregated statistics over a
   mixed multi-label head.
2. **Per-class loss weighting is honest.** Pedestrian frames are 24% of
   the data; with a shared backbone and a multi-label head, balancing the
   pedestrian loss against the (much easier, much commoner) vehicle loss
   requires hand-tuned task weights. Three separate models sidesteps this
   entirely.
3. **Independent stop-ship decisions.** If the pedestrian model fails to
   reach the safety threshold but the other two pass, we still have the
   option of deploying the latter two and using a different (perhaps
   non-ML) means of pedestrian sensing — a single multi-label model
   couples the deployment decisions.
4. **Threshold setting is per-task.** Sheet 4/7 ask different threshold
   strategies per detector (the planner uses each output differently);
   a single model would force a shared softmax over labels.

The trade-off, of course, is 3× the compute, 3× the storage, and 3× the
inference cost. For a research-stage safety analysis this is the correct
trade.

## 3.6 Evaluation

### Test-set metrics

The three models were trained for 5 epochs each with the setup described in
3.5. Final metrics on the held-out test split:

| Model | Accuracy | Precision | Recall | F1 | Test positives |
|---|---:|---:|---:|---:|---:|
| Traffic light | 0.944 | 0.942 | 0.982 | 0.962 | 2,584 |
| Pedestrian | 0.627 | 0.274 | 0.548 | 0.366 | 706 |
| Vehicle | 0.854 | 0.965 | 0.835 | 0.895 | 2,700 |

Per-model confusion matrices are in `outputs/3.6_confusion_{task}.png` and the
machine-readable metrics in `outputs/3.6_test_metrics.json`.

### Which model performs worst — and why

**The pedestrian detector is by far the worst performer** (F1 = 0.366,
recall = 0.548), which confirms the prediction from the dataset exploration
in 3.4. The ordering matches what the data statistics implied:

1. **Pedestrian (worst).** The rare class (24% positive in training) and
   the small class (median 156 px). Even with inverse-frequency class
   weighting in the loss, both factors compounded: fewer positive gradient
   signals per epoch *and* a visual cue that becomes ~10 px after the
   224×224 resize, near the limit of what a ResNet-18 with a 7×7
   first-layer stride can resolve.
2. **Vehicle (middle).** F1 = 0.895, with strong precision (0.965) but
   weaker recall (0.835). Vehicles are common (76%) and large
   (median 781 px), so the model has plenty of examples and pixels to
   work with — but the precision/recall asymmetry runs the wrong way
   safety-wise (see below).
3. **Traffic light (best).** F1 = 0.962 with recall (0.982) above
   precision (0.942). Lights are small (median 295 px) but highly
   distinctive in color and structure, and the positive rate is high
   (73%), so the model has both abundance and signal. The recall-above-
   precision bias is also the safety-correct direction.

### Pedestrian detector: dual failure mode

The pedestrian confusion matrix is `[[1871, 1023], [319, 387]]` on
3,600 test frames (706 positive, 2,894 negative). This gives **two**
safety-relevant failure modes, not one:

- **False-negative rate: 319 / 706 = 45.2%.** Roughly *half* of all
  pedestrian-positive frames are missed by the detector.
- **False-positive rate: 1,023 / 2,894 = 35.4%.** And because negatives
  outnumber positives roughly 4-to-1, the absolute number of false
  alarms (1,023) is 2.6× larger than the number of correct positive
  predictions (387). Precision is only 0.274 — fewer than 1 in 4
  "pedestrian!" alerts corresponds to a real pedestrian.

This produces a dual failure mode:

1. **Missed detections (319 events) are the dominant safety hazard.**
   With no sensor redundancy in the system, a missed pedestrian
   propagates directly to the planner, which then does not request
   emergency braking. The only remaining mitigation is the human
   safety operator — and the system description acknowledges the
   operator may exhibit "non-zero probability of delayed or missed
   intervention, especially under prolonged monitoring."
2. **Spurious detections (1,023 events) are an operational hazard.**
   If the planner triggers emergency braking on every positive
   pedestrian prediction, the vehicle would brake ~2.6× more often
   than it needs to. This introduces rear-end-collision risk from
   following vehicles, operator startle, and — most insidiously —
   strong incentive for the safety operator to distrust or disable
   the system, eroding the only fallback barrier the system has.

Neither rate is acceptable for an autonomy stack with no sensor
redundancy and a single human fallback.

### From a safety perspective: precision or recall?

The answer is the same for all three detectors but for slightly different
reasons: **recall is the safety-critical metric**, because the cost of a
missed detection (false negative) is at least an order of magnitude worse
than the cost of a false alarm (false positive). The system has no second
perception channel that could catch the miss.

- **Pedestrian detector — recall.** A false negative ("no pedestrian
  present" when one is) means the planner does not request emergency
  braking. This is the worst-case failure in the entire hazard log. A
  false positive causes an unnecessary brake event — uncomfortable and
  a potential rear-end-collision contributor, but not directly fatal.
- **Vehicle detector — recall, with a caveat.** Missing a vehicle ahead
  causes failure to brake. False positives cause spurious emergency
  braking, which on a 50 km/h urban road is itself a hazard.
  **Important: our trained vehicle model has precision (0.965) > recall
  (0.835)**, meaning it is biased *away* from positive predictions
  — i.e., toward missing vehicles. This is the safety-undesirable
  direction. The vehicle detector therefore needs threshold adjustment
  (Sheet 7) or further training before it can be trusted.
- **Traffic light detector — recall.** The safety-relevant prediction is
  "stopping required at the approaching intersection." Missing a red
  light (false negative) leads to running it — the dominant hazard at
  urban intersections. Our trained traffic-light model has recall
  (0.982) > precision (0.942), which is the correct safety bias.

### Summary

Of the three detectors, only the traffic-light model has both adequate
F1 and the correct safety-oriented precision/recall asymmetry. The
vehicle model has adequate F1 but the *wrong* bias direction. The
pedestrian model fails on both axes — F1 of 0.366 and a 45.2%
miss rate that, in a system without sensor redundancy, propagates
directly to the loss event.

**This is the headline finding of Sheet 3 for the safety case:** as
currently trained, the perception stack is not deployment-ready, and the
pedestrian detector is the binding constraint. This evidence feeds
directly into Sheet 5 (ODD refinement) and Sheet 10 (deployment
recommendation).

The trained checkpoints (`checkpoints/{task}.pt`) are persisted and will
be reused in every subsequent sheet.

## 3.7 ODD Gap Analysis

> Sheet 2 was where the ODD was defined before seeing data. The conditions
> below are what the **training data actually contains**.

**Conditions represented in the training split (as collected in CARLA):**

| ODD dimension | Coverage in training data |
|---|---|
| Weather | **Sunny / clear only** (per the course system description: "trained exclusively on sunny daytime data") |
| Time of day | **Daytime only** (clear daylight, no dusk/dawn/night) |
| Precipitation | **None** (no rain, no wet road, no spray) |
| Visibility | **High** (no fog, no haze, no glare-handling diversity) |
| Road type | Urban streets (the only ones in the CARLA scenario provided) |
| Town/scene | A single CARLA town for training (the test-time set includes `test-town-01`, `test-fog`, `test-night`, which are explicitly *held out*) |
| Camera placement | Fixed forward-facing, behind windshield, 10 Hz |
| Speed | Variable urban speeds, but capped by the simulator scenario |

**Comparison to the Sheet 2 ODD.** The system description's *intended*
operating conditions in Sheet 2 are "daytime, dry weather, mapped
intersections" but explicitly acknowledge that "weather and lighting
conditions can change during a test drive (e.g., sudden cloud cover, low sun
angle, rain onset)." So the *operational* ODD includes those transitions,
while the *training* ODD does not.

**ODD dimensions not covered by training data:**

1. **Rain / wet road surfaces** — represented in `test-fog` and not training.
2. **Fog / reduced visibility** — `test-fog` split, training has none.
3. **Night / low-light** — `test-night` split, training is daytime-only.
4. **Sudden lighting transitions** (cloud cover, low sun angle, tunnel
   exits) — none in training, mentioned explicitly in the system description
   as an operational reality.
5. **Alternate urban geometries** — `test-town-01` is a held-out town the
   training set has never seen, exposing generalization across road layouts.
6. **Twilight / dusk / dawn** — not in training, not (explicitly) in any
   test split either, so this is an uncovered *operational* condition that
   doesn't even have evaluation evidence.

**Implications for safety.** Because the training distribution covers only a
strict subset of the operational ODD, **we cannot make any performance claims
for the four uncovered conditions on the basis of in-distribution test
metrics alone**. Specifically:

- *No claim* can be made about pedestrian recall at night, in fog, or in
  rain. The system description notes the operator is the only fallback if
  automation fails, and that the operator may exhibit "non-zero probability
  of delayed or missed intervention, especially under prolonged
  monitoring." So missed pedestrian detection in degraded conditions is a
  hazard that propagates directly to the loss event.
- *No claim* can be made about behavior under sudden weather transitions
  (e.g., a sunny drive that hits a rain shower). This is exactly the case
  most likely to be encountered during a real test drive per the system
  description.
- *No claim* can be made about generalization to a town not in the training
  set. `test-town-01` will provide the first evidence for this.

**Gaps to revisit in Sheets 4 and 5.** I will carry forward these uncovered
dimensions as explicit gaps:

- **G1** — rain / wet conditions
- **G2** — fog / low visibility
- **G3** — night / low light
- **G4** — sudden lighting transitions (operational ODD only)
- **G5** — alternate towns / unseen road layouts
- **G6** — dusk / dawn

In Sheet 4 these become test design targets (the held-out `test-fog`,
`test-night`, `test-town-01` splits give direct measurements for G1, G2, G3,
G5). In Sheet 5 the k-projection ODD coverage analysis will quantify which
joint combinations of weather × lighting × town are unseen, and the ODD will
be **refined**: either by restricting the deployable ODD to match what we
have evidence for (a "sunny-only" deployment), or by expanding the training
data to cover the gaps.
