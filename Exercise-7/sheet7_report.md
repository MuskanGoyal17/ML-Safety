# Exercise Sheet 7 — Uncertainty Quantification

## Exercise 7.1 *optional*

## Exercise 7.2: Calibration and ECE

**Calibration:** A classifier is well-calibrated if its predicted confidence matches empirical accuracy. Formally, for all confidence levels p: P(Y=1 | f(X)=p) = p. A model that outputs p=0.8 on a set of inputs should be correct on 80% of those inputs.

**Expected Calibration Error (ECE):**

ECE is computed by:
1. Partitioning predictions into M equal-width confidence bins (M=10, covering [0,1])
2. For each bin b: computing the gap between mean confidence conf(b) and mean accuracy acc(b)
3. Weighting each gap by the fraction of samples in that bin

$$\text{ECE} = \sum_{b=1}^{M} \frac{|B_b|}{N} \left| \text{conf}(B_b) - \text{acc}(B_b) \right|$$

A perfectly calibrated model has ECE = 0. Values above ~0.05 are generally considered problematic for safety-critical applications.

---

## Exercise 7.3: Cost-Optimal Downstream Decisions

**Cost matrix:**

|  | Pedestrian present | No pedestrian |
|--|-------------------|--------------|
| Brake | 0 | C_FP = 1 |
| Continue | C_FN = 100 | 0 |

### Q1 — Expected loss of each action

Let p = P(pedestrian present | x).

**E[L | brake]** = 0 · p + C_FP · (1 − p) = **(1 − p)**

**E[L | continue]** = C_FN · p + 0 · (1 − p) = **100p**

### Q2 — Threshold τ*

Set E[L | brake] = E[L | continue]:

1 − p = 100p → 1 = 101p → **τ\* = 1/101 ≈ 0.0099**

For p > τ\* the autopilot should brake; for p < τ\* it should continue.

### Q3 — Comparison to standard argmax (τ = 0.5)

τ\* ≈ 0.0099 is approximately **50× lower** than τ = 0.5. The autopilot should brake whenever the model assigns even ~1% probability to a pedestrian being present. The asymmetry is justified: a false negative costs 100× more than a false positive.

### Q4 — Why τ* only works with calibrated probabilities

τ\* is derived from the expected cost formula, which treats p as a true probability. If the model is overconfident or underconfident — for example reporting p = 0.05 when the true probability is 0.40 — applying τ\* to the raw output produces suboptimal decisions. An underconfident model (as we observed) suppresses probabilities toward 0.5, meaning many genuinely high-risk frames may sit below τ\* and be missed. Only when p is calibrated does τ\* yield cost-minimising decisions.

---

## Exercise 7.4: Measuring Calibration

### ECE per model (uncalibrated, T=1.0)

| Model | ECE | Calibration direction |
|-------|-----|-----------------------|
| pedestrian | 0.1204 | underconfident |
| traffic_light | 0.1063 | underconfident |
| vehicle | 0.1401 | underconfident |

### Reliability diagram interpretation

All three models are **underconfident**: the calibration curve lies *above* the diagonal. This means that when the model reports, say, 60% confidence, the true fraction of positives in that bin is actually higher — the model is more accurate than it claims to be. This is the opposite of the typical overconfidence pattern seen in models trained without class-balancing.

The underconfidence is consistent with the weighted cross-entropy loss used during training: the high weight placed on the minority positive class pushes the model to predict positives more conservatively, suppressing confidence below its true accuracy. This pattern holds **consistently across all three models**, which share the same architecture and training procedure.

**Does the pattern hold across all three models?** Yes — all three show underconfidence with similar ECE magnitudes (0.10–0.14), confirming this is a systematic effect of the training procedure rather than a model-specific anomaly.

---

## Exercise 7.5: Temperature Scaling

### Method

For each model, the best temperature T is found by grid search over T ∈ {0.5, 0.6, ..., 3.0} (step 0.1), minimising mean negative log-likelihood on the **validation** set. The selected T is then applied to **test** logits to compute the post-calibration ECE.

### Results

| Model | Best T | ECE before | ECE after | ΔECE |
|-------|--------|-----------|-----------|------|
| pedestrian | 2.6 | 0.1204 | 0.0783 | −0.0421 |
| traffic_light | 2.8 | 0.1063 | 0.0413 | −0.0650 |
| vehicle | 2.5 | 0.1401 | 0.1316 | −0.0085 |

### Interpretation

All three models require **T > 1.0** (ranging from 2.5 to 2.8), which softens the probability distribution by spreading confidence away from the extremes. This is consistent with the underconfidence finding in 7.4: a high temperature further softens probabilities, but here it corrects the underconfidence by pulling predictions closer to the true fraction of positives in each bin.

**Traffic light** benefits most from calibration: ECE drops from 0.1063 to 0.0413 (Δ = −0.0650), crossing below the 0.05 threshold that is generally considered acceptable.

**Pedestrian** shows meaningful improvement: ECE drops from 0.1204 to 0.0783 (Δ = −0.0421), but remains above 0.05 — temperature scaling alone is insufficient for deployment-ready calibration.

**Vehicle** shows minimal improvement: ECE drops only from 0.1401 to 0.1316 (Δ = −0.0085). The vehicle model's miscalibration is structurally resistant to a single temperature parameter, suggesting a more complex calibration pattern that temperature scaling cannot capture. A more expressive method (isotonic regression, Platt scaling) would be needed.

**Temperature scaling preserves accuracy** — it rescales logits without changing their ordering, so predictions at threshold τ = 0.5 are identical before and after scaling.

---

## Exercise 7.6: Cost-Optimal Decision in Practice

**Setup:** C_FN = 100, C_FP = 1, τ\* = 0.0099, τ_std = 0.5  
**Total loss:** L = C_FN · #FN + C_FP · #FP

### Results (pedestrian model)

|  | τ = 0.5 | τ\* ≈ 0.0099 |
|--|---------|-------------|
| **Uncalibrated** | FP=0, FN=706, L=**70,600** | FP=2894, FN=0, L=**2,894** |
| **Calibrated (T=2.6)** | FP=0, FN=706, L=**70,600** | FP=2894, FN=0, L=**2,894** |

**Best combination: τ\* = 0.0099 (either calibrated or uncalibrated) → total loss = 2,894**

### Interpretation

**τ = 0.5 (both rows):** The standard threshold produces 706 false negatives — every single pedestrian frame is missed — and zero false positives. At C_FN = 100 each, total loss = 70,600. This is catastrophic: the model never brakes for a pedestrian.

**τ\* = 0.0099 (both rows):** The near-zero threshold causes the model to predict "pedestrian present" for almost every frame — 2,894 false positives and zero false negatives. Total loss = 2,894 (one per false positive). Despite braking unnecessarily on ~80% of frames, the total cost is **24× lower** than the standard threshold.

**Why calibrated = uncalibrated at τ\*:** The pedestrian model is underconfident — most predictions are clustered in the mid-range rather than near 0 or 1. Both calibrated and uncalibrated probabilities sit well above τ\* = 0.0099 for nearly all frames, so the threshold behaviour is identical. The difference between calibrated and uncalibrated only matters at thresholds in the mid-range (e.g. τ = 0.5) where the precise probability values change the classification.

**Key finding:** The choice of threshold (τ\* vs 0.5) matters **far more** than calibration for this cost structure. Using the cost-optimal threshold reduces total loss by a factor of 24, while calibration alone at τ = 0.5 provides no improvement. This demonstrates that even without calibration, adopting τ\* is the single most impactful change for this safety-critical application.

---

## Exercise 7.7: Tracing Overconfidence Through the Safety Analysis

### Q1 — Causal Scenario

Existing hazard (Exercise 2.4): *"Vehicle fails to brake for a pedestrian who is present."*
Existing UCA (Exercise 2.6): *"The planner does not command braking while a pedestrian is in the path."*

**New causal scenario:**

> The pedestrian classifier produces a false negative (predicts "no pedestrian" for a frame containing a pedestrian) and simultaneously reports high confidence in that wrong prediction. Although the model is globally underconfident (ECE = 0.1204 uncalibrated), individual false negative predictions can still carry high confidence scores — the planner's logic registers this as a reliable negative. No low-confidence fallback is triggered. With the standard threshold τ = 0.5, this results in 706 missed pedestrians on the test set alone (total cost 70,600). The vehicle continues at speed and a collision results.

This causal scenario is structurally distinct from an OOD failure: the input is in-distribution, the model is loaded correctly, and no system monitor flags an anomaly. The failure mode is purely a miscalibration and threshold choice issue.

---

### Q2 — Safety Constraints

**Model-level constraint (calibration):**

> *The pedestrian classifier shall achieve ECE ≤ 0.05 on the in-distribution test set after post-hoc calibration (temperature scaling or equivalent), as measured by the 10-bin equal-width ECE. The current uncalibrated ECE is 0.1204; temperature scaling at T=2.6 reduces this to 0.0783, which does not yet meet the threshold. Additional calibration methods (isotonic regression, Platt scaling) must be applied until ECE ≤ 0.05 is achieved before deployment. Re-verification is required after any retraining.*

**System-level constraint (planner decision rule):**

> *The planner shall use the cost-optimal threshold τ\* = C_FP / (C_FP + C_FN) = 1/101 ≈ 0.0099 for the pedestrian classifier rather than the standard argmax threshold of 0.5. As demonstrated in Exercise 7.6, switching from τ = 0.5 to τ\* = 0.0099 reduces total cost-weighted loss from 70,600 to 2,894 — a 24× reduction. The threshold τ\* shall be recomputed whenever the cost matrix C_FN, C_FP is updated, and must be applied to calibrated probabilities to ensure the expected-cost derivation is valid.*

---

### Q3 — Verification

The evidence produced by this sheet directly verifies the model-level constraint:

- **Exercise 7.4** establishes the baseline: ECE = 0.1204 (pedestrian), 0.1063 (traffic_light), 0.1401 (vehicle) — all substantially above 0.05. The constraint is **not currently met**.
- **Exercise 7.5** shows the effect of temperature scaling: pedestrian ECE improves to 0.0783 at T=2.6, traffic_light to 0.0413, vehicle to 0.1316.

**Does the calibrated model meet the threshold?**
- Traffic light: **yes** (ECE = 0.0413 < 0.05 )
- Pedestrian: **no** (ECE = 0.0783 > 0.05 ) — additional calibration required
- Vehicle: **no** (ECE = 0.1316 > 0.05 ) — temperature scaling is insufficient; a more expressive calibration method is needed

---

### Q4 — Residual Risk

Even with a perfectly calibrated model (ECE = 0) and the cost-optimal threshold τ\*, the following risks remain:

**1. Calibration does not improve accuracy:**
Temperature scaling preserves the ranking of predictions — it cannot convert a false negative into a true positive. A calibrated model that assigns p = 0.008 to a frame with a pedestrian will correctly not trigger braking (since 0.008 < τ\* = 0.0099), but the pedestrian is still missed. The causal scenario is reduced in frequency but not eliminated.

**2. In-distribution calibration does not generalise to OOD:**
Temperature scaling is fitted on in-distribution validation data. On OOD inputs (fog, night, different town — Sheet 9), calibration breaks down. The calibrated probability p no longer reflects P(pedestrian | x) for OOD inputs, making the τ\* decision rule invalid outside the training distribution.

**3. τ\* assumes a fixed cost matrix:**
The cost matrix C_FN = 100, C_FP = 1 is a modelling assumption. In practice, costs depend on context (vehicle speed, pedestrian proximity, traffic density). A fixed τ\* cannot adapt to dynamic risk conditions.

**4. System-level fallback still required:**
The causal scenario involves confident false negatives that no confidence-based fallback can catch (the model is confident in the wrong answer). The system-level constraint — using τ\* instead of 0.5 — reduces false negatives structurally, but does not address the case where a well-calibrated model still produces the occasional confident false negative. A redundant sensor (radar, lidar) or a temporal consistency check provides a second layer of protection that calibration alone cannot.

**Conclusion:** Calibration is necessary but not sufficient. The model-level constraint (ECE ≤ 0.05) and system-level constraint (use τ\*) must both be active. The vehicle model's persistent miscalibration (ECE = 0.1316 after scaling) indicates that for that model, the system-level constraint is the primary operative defence until better calibration is achieved.
