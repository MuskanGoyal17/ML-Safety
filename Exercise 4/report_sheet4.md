# Exercise Sheet 4 — Model Testing and Validation

---

## 4.1 Traditional Testing vs. ML Model Testing

| # | Dimension | Traditional Software | ML Model |
|---|---|---|---|
| 1 | **Test oracle** | Exact expected output from the specification (deterministic) | A ground-truth label in a dataset; correctness is probabilistic, measured in aggregate |
| 2 | **Coverage criterion** | Line/branch/condition coverage of code paths | Input-space coverage: pixel distributions, weather conditions, edge cases not captured by code structure |
| 3 | **Failure mode** | Logical/arithmetic bugs — a given input either passes or crashes deterministically | Statistical degradation — a model is never perfectly correct; we track error *rates* across populations |
| 4 | **Test design source** | From a formal specification or requirements document | From data distributions, hazard analyses, and operational scenarios — no complete spec exists |
| 5 | **Regression testing** | Adding a test case permanently constrains future code changes | Adding data can shift the distribution the model learns from; "regression" may not hold without retraining |
| 6 | **Root-cause analysis** | A bug can be traced to a specific code line | A misclassification traces to learned weights over millions of parameters — not a single identifiable line |



## 4.4 Distribution Shift Types

### Scenario 1 — Winter deployment (wet roads, low sun angle, glare)

**a) Type of shift: Covariate shift.**
The input distribution P(X) changes (the camera now sees wet reflections,
long shadows, lens flare from a low sun), while the conditional label
distribution P(Y | X) is unchanged — a pedestrian on a wet road is still a
pedestrian. The relationship between the visual appearance and the label is
the same; only the appearance distribution has changed.

**b) Expected effect on model performance.**
The model was trained exclusively on clear sunny daytime images from CARLA.
Wet roads create specular reflections that mimic object boundaries; low-angle
sun creates strong shadows and blown-out regions. Both corrupt the image
statistics the model learned. Expected degradation: significant drop in
recall for all three detectors, particularly pedestrians (the smallest,
hardest target), because the feature maps that encode pedestrian cues in
sunny conditions do not generalize to glare-heavy images. False negatives
will increase; the model may also produce false positives on specular
reflection artefacts.

**c) Mitigation strategy.**
Collect and annotate training data from the same simulator under winter/
low-sun conditions, and add them to the training set (domain expansion).
Alternatively, apply image preprocessing (adaptive histogram equalization,
anti-glare filtering) to normalize the input distribution before it reaches
the model. For the safety case, this gap would be noted as Gap G1 and the
ODD would be restricted to exclude winter/low-sun conditions until evidence
is gathered under those conditions.

---

### Scenario 2 — New city zone with 60% cyclists (rare in training)

**a) Type of shift: Label shift (and partly concept shift).**
The marginal label distribution P(Y) changes: cyclists were < 5% in
training and are now 60% of road users. There is also a concept-shift
component: cyclists are visually distinct from the pedestrian and vehicle
classes the model was trained on. The model has no "cyclist" label — it
must decide whether to absorb cyclists into its existing categories.

**b) Expected effect on model performance.**
The vehicle detector may misclassify cyclists as "no vehicle" (because a
bicycle is visually unlike a car or van), causing missed detections in a
zone with the highest road-user density the system has encountered. The
pedestrian detector may *incorrectly* fire on cyclists (cyclists are
upright, two-legged at a distance), producing false positives that trigger
unnecessary braking. Overall, the mismatch between the training label
taxonomy and the actual object distribution in this zone makes the
perception stack unreliable.

**c) Mitigation strategy.**
Expand the system taxonomy to add a "cyclist" class (or redefine the
vehicle class to include cyclists) and retrain with representative cyclist
data. Before retraining, restrict the AV's ODD to exclude this zone, or
lower the operating speed to give the human operator more reaction time.
In the safety case, flag this as a label taxonomy gap: the ODD was defined
for pedestrians and vehicles, not for mixed micro-mobility users.

---

### Scenario 3 — New slim traffic light housing design

**a) Type of shift: Concept shift (narrow form: virtual concept drift).**
The mapping from visual appearance to the label "traffic light present"
has changed. P(X | Y = traffic_light) shifts because the visual appearance
of the positive class is now different (slimmer housing, different aspect
ratio, potentially different LED arrangement). P(Y | X) has technically
changed: for images that look like the old traffic lights, the label is
now "no traffic light" (they've been replaced). This is distinct from
covariate shift because the *meaning* of the label in visual space has
changed, not just the background statistics.

**b) Expected effect on model performance.**
The traffic-light detector, currently the strongest model (F1 = 0.962),
will experience recall degradation proportional to how visually different
the new housings are from the training distribution. The model may still
detect the light poles or the signal colours (which are standardised), so
degradation may be partial rather than catastrophic — but it cannot be
quantified without new test data with the new housings.

**c) Mitigation strategy.**
Collect a small labelled dataset with the new housing design, evaluate the
model on it (Sheet 4 exercise: design a test for "novel traffic light
appearance"), and if recall drops below threshold, fine-tune the model on
the new images. This is a natural fit for an automated requalification
pipeline triggered whenever infrastructure changes are known. For the
safety case, note that the model was validated only on the training-era
housing design.

---

## 4.5 ODD Coverage with k-Projection Coverage

### 4.5.1 What k-projection coverage measures

k-projection coverage measures what fraction of all k-dimensional
combinations of ODD dimension values appear in the test set.

Concretely: define the ODD as a set of dimensions, each with a finite list
of possible values (e.g. `weather ∈ {clear, rain, fog}`, `lighting ∈
{day, night}`). The full ODD has a combinatorial space of all possible
joint assignments. For a given k, k-projection coverage asks: for every
subset of k dimensions, does the test set contain at least one scenario
that covers each possible combination of values in those k dimensions?

**Why it is useful:** individual (k=1) coverage only checks that each value
appears at least once — e.g., "we have some rainy scenes." Pairwise (k=2)
coverage checks every *pair* of values — e.g., "we have a rainy night scene
AND a foggy day scene AND a clear night scene," and so on. As k grows, we
demand evidence for more complex co-occurrence patterns. A test set that
looks adequate at k=1 often reveals large gaps at k=2, because achieving
coverage of all pairs requires exponentially more scenarios. This makes it a
sensitive detector of underspecified test suites.

### 4.5.2 Computed k-projection coverage

After running below script


# Standard test split only
python odd_coverage.py --data-root /content/data/MyDataset

# Include fog/night/town-01 splits
python odd_coverage.py --data-root /content/data/MyDataset --include-ood


Results stored in `outputs/4.5_odd_coverage.txt` and `outputs/4.5_odd_coverage.png`.



### 4.5.3 Interpretation: how coverage changes with k

Coverage drops substantially with each increment of k. This is expected
and reveals two things:

1. **The test split is severely under-representative of the operational ODD.**
   Clear-day coverage is good; everything else is missing. This is consistent
   with the ODD gaps identified in Sheet 3 (G1 rain, G2 fog, G3 night, G4
   lighting transitions). The k=1 result looks moderate only because
   the per-label presence combinations (pedestrian/vehicle/TL) are well
   covered; once you condition on weather or lighting the gaps open up.

2. **Even the combined test+OOD splits achieve low k=3 coverage.**
   This is because the OOD splits each fix *different* weather/lighting
   values but still share the same road_type and speed_range distribution.
   The combinations of e.g. `fog × night × high_speed` or
   `rain × unmapped_road × pedestrian_present` are never tested — those
   are exactly the conditions most likely to cause accidents.

For the safety case this is direct evidence that the test suite is
inadequate for the full claimed ODD: the ODD must be either
**restricted** (to clear-day urban mapped) or **expanded** with
additional test data before a deployment claim can be made.

---

## 4.6 Test Suite Design from Safety Constraints

The safety constraints below are drawn from the UCAs defined in Sheet 2
(SC-1 through SC-4 correspond to emergency braking failures for each
detection target and a combined scenario).

| Constraint ID | Constraint | Test input description | Expected output | Pass criterion |
|---|---|---|---|---|
| SC-1 | Pedestrian detector must not fail to detect a pedestrian at critical distance | Image containing a pedestrian ≤ 10 m from the ego vehicle (close-range, centre-frame, clear day) | `has_pedestrian = True` | Recall ≥ 0.90 on this close-range subset |
| SC-1b | Pedestrian detector must not fail on partially occluded pedestrians | Image of a pedestrian 50% occluded by a parked vehicle (CARLA synthetic) | `has_pedestrian = True` | Model predicts positive; occlusion subset recall ≥ 0.80 |
| SC-2 | Vehicle detector must not fail to detect a lead vehicle within stopping distance | Image containing a vehicle directly ahead at < 15 m at 50 km/h | `has_vehicle = True` | Recall ≥ 0.95 on close-range vehicle subset |
| SC-3 | Traffic-light detector must not miss a red light at an approaching intersection | Image containing a traffic light with the ego at ≤ 30 m from the stop line | `has_traffic_light = True` | Recall ≥ 0.95 on near-intersection subset |
| SC-4 | No detector shall produce systematic false negatives under lighting transitions | Images drawn from dusk/dawn or sudden-cloud-cover conditions (OOD test split if available) | All three detectors predict consistently with ground truth | Recall drop vs. clear-day baseline ≤ 10 percentage points |
| SC-5 | Pedestrian detector must not produce excessive false alarms causing unsafe braking | Images with no pedestrian present (true negatives) | `has_pedestrian = False` | FPR ≤ 0.20 (at most 1-in-5 negative frames triggers a false alarm) |


---

## 4.7 Per-Class Evaluation

### 4.7.1–4.7.2 Metrics and confusion matrices

All three models were evaluated on the held-out test split using the
checkpoints saved during Sheet 3 training. Plots are in
`outputs/4.7_confusion_{task}.png`, `outputs/4.7_precision_recall_bar.png`,
and `outputs/4.7_pedestrian_pr_curve.png`.

**Metrics summary:**

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Traffic light | 0.944 | 0.942 | 0.982 | 0.962 |
| Pedestrian | 0.627 | 0.274 | 0.548 | 0.366 |
| Vehicle | 0.854 | 0.965 | 0.835 | 0.895 |

**Confusion matrices** — format: `[[TN, FP], [FN, TP]]`

**Traffic light:**

|  | Pred negative | Pred positive |
|---|---:|---:|
| **True negative** | 859 | 157 |
| **True positive** | 46 | 2,538 |

- TN=859, FP=157, FN=46, TP=2,538
- False negative rate: 46 / (46+2538) = **1.8%** — misses only 1 in 55 traffic lights
- False positive rate: 157 / (157+859) = **15.5%** — manageable spurious rate
- The FP count (157) is driven by frames where a light is partially visible
  or at the edge of the frame but the label says "present"

**Pedestrian:**

|  | Pred negative | Pred positive |
|---|---:|---:|
| **True negative** | 1,871 | 1,023 |
| **True positive** | 319 | 387 |

- TN=1,871, FP=1,023, FN=319, TP=387
- False negative rate: 319 / (319+387) = **45.2%** — misses nearly half of all pedestrians
- False positive rate: 1,023 / (1,023+1,871) = **35.4%** — 1 in 3 negative frames triggers a false alarm
- Absolute false alarms (1,023) outnumber true detections (387) by 2.6×

**Vehicle:**

|  | Pred negative | Pred positive |
|---|---:|---:|
| **True negative** | 819 | 81 |
| **True positive** | 446 | 2,254 |

- TN=819, FP=81, FN=446, TP=2,254
- False negative rate: 446 / (446+2254) = **16.5%** — misses 1 in 6 vehicles
- False positive rate: 81 / (81+819) = **9.0%** — low spurious alarm rate
- The precision (0.965) >> recall (0.835) asymmetry means the model is
  biased toward *not* predicting vehicles — the safety-undesirable direction

### 4.7.3 Which model has the lowest recall — was it expected?

The **pedestrian detector** has the lowest recall at **0.548**, missing
319 out of 706 pedestrian-positive frames (45.2% miss rate). This is
fully consistent with the hazard analysis from Sheet 2: pedestrian
detection failure was identified as the most safety-critical UCA because
pedestrians are the rarest class (24% of training frames), the smallest
objects (median 156 px), and the most vulnerable road users in any
collision scenario. The result was anticipated, but the magnitude — nearly
one in two pedestrians missed — is worse than any deployable system
could tolerate.

For completeness, the three models ranked by recall:

| Rank | Model | Recall | FN count | FN rate |
|---|---|---:|---:|---:|
| 1 (best) | Traffic light | 0.982 | 46 | 1.8% |
| 2 | Vehicle | 0.835 | 446 | 16.5% |
| 3 (worst) | Pedestrian | 0.548 | 319 | 45.2% |

The vehicle model also deserves scrutiny: 446 missed vehicles is a large
absolute count and its precision-recall asymmetry (0.965 vs 0.835) signals
the model is biased toward predicting "no vehicle" — the wrong direction
for a safety-critical detector. It passes a minimal deployment bar but
would need threshold adjustment before real use.

### 4.7.4 Minimum recall threshold for pedestrian deployment

**Proposed minimum recall: 0.90**

**Current status: FAILS — actual recall is 0.548, 34 pp below threshold.**

**Justification for the 0.90 threshold:**

The safety argument proceeds from the fault tree for the pedestrian
detection failure. Given the system architecture:

- The pedestrian detector is the **sole** sensor channel for pedestrian
  presence (no radar, no LiDAR, no ultrasonic — single camera only).
- The human safety operator is the **only** backup. The system description
  explicitly acknowledges operators may exhibit "non-zero probability of
  delayed or missed intervention, especially under prolonged monitoring"
  (4-hour shifts, visual-only dashboard, no auditory alerts).
- Emergency braking is not triggered if the pedestrian detector outputs
  negative, regardless of operator state.

Setting a target of ≤ 5% probability of "pedestrian present AND system
fails to brake" per encounter:

```
P(system miss) = P(detector misses) × P(operator also misses)
               = (1 − recall) × P_op

Conservative estimate: P_op ≈ 0.15 (operator fatigue over 4-hour shift)

Solving for recall:
  (1 − recall) × 0.15 ≤ 0.05
  recall ≥ 0.67   ← minimum floor
```

This gives a mathematical floor of 0.67. Two conservatism adjustments
push the operational threshold higher:

1. **OOD degradation.** The 0.548 recall is measured on *sunny daytime*
   test data — the easiest possible condition. Real-world recall in rain,
   fog, or at night will be lower. The in-distribution recall is an
   optimistic upper bound; operational recall will be strictly worse.

2. **Test-set optimism.** The test split is drawn from the same CARLA
   town and conditions as training. Generalization to real cameras, real
   lighting, and real pedestrian motion will introduce additional degradation.

Applying both factors, **0.90 recall** is the appropriate threshold —
consistent with the SOTIF (ISO 21448) guidance for sole-channel
perception components where the human is the only safety backup.

**Pass/fail verdict against safety constraints from Exercise 4.6:**

| Constraint | Threshold | Actual | Result |
|---|---|---|---|
| SC-1: Pedestrian recall (overall) | ≥ 0.90 | 0.548 |  FAIL |
| SC-2: Vehicle recall (overall) | ≥ 0.95 | 0.835 |  FAIL |
| SC-3: Traffic light recall | ≥ 0.95 | 0.982 |  PASS |
| SC-5: Pedestrian FPR | ≤ 0.20 | 0.354 |  FAIL |

Three of four measurable constraints fail. Only the traffic light detector
meets its safety threshold. The pedestrian detector fails on both recall
(SC-1) and false-alarm rate (SC-5). The vehicle detector fails on recall
(SC-2) despite high precision.

**Conclusion:** the perception stack as currently trained is **not
deployment-ready**. The pedestrian detector is the binding constraint —
it falls 34 percentage points short of the required recall threshold and
simultaneously produces spurious alerts at nearly double the acceptable
false-positive rate. No deployment recommendation can be made without
either improving the pedestrian model, adding sensor redundancy, or
significantly restricting the ODD to conditions where the performance gap
can be bounded.

