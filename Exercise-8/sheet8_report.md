# Exercise Sheet 8 

**Course:** Introduction to Machine Learning Safety  
---

## Exercise 8.1: What Are Adversarial Examples?

An adversarial example is an input that has been deliberately modified by adding a small, carefully crafted perturbation — imperceptible or barely perceptible to a human — that causes a trained model to produce an incorrect prediction with high confidence. The perturbation is not random: it is computed by following the gradient of the loss function with respect to the input, so it maximally exploits the model's decision boundary.

**Difference from OOD examples:**
An OOD example arises from a genuinely different data distribution (e.g. a foggy image when the model was trained on sunny images). The shift is typically visible and affects global image statistics. An adversarial example, by contrast, comes from the same distribution as the training data in terms of human perception — it looks like a normal training image — but lies in a tiny adversarially chosen region of input space where the model's decision boundary is wrong. OOD detectors based on softmax confidence or feature-space distance can often flag OOD inputs, but largely fail against adversarial examples because adversarial inputs still produce high-confidence predictions and near-normal feature representations.

---

## Exercise 8.2: Attack Formulation

**Q1 — Each term in `x_{i+1} = x_i + α ∇_x L(y, f(x_i))`:**
- `x_i`: the current (perturbed) input at iteration i
- `α`: the step size controlling how far each update moves the input
- `∇_x L(y, f(x_i))`: gradient of the loss with respect to the input — points in the direction that most increases the loss, i.e. makes the model more wrong
- The full update moves the input in the direction that maximally increases the classification loss

**Q2 — Targeted vs untargeted:**
Untargeted maximises loss for the true label: `x + α ∇_x L(y_true, f(x))`. Targeted minimises loss for a specific wrong class t: `x − α ∇_x L(y_target, f(x))`. The sign flips because we want the model to predict the attacker's chosen target class rather than simply making any error.

**Q3 — Perturbation budget:**
The basic update has no constraint: after many steps the perturbation grows arbitrarily large. To enforce `‖x_0 − x_t‖_∞ ≤ ε`, after each update the input is projected back onto the ε-ball around the original: `x_{i+1} = clip(x_i + α sign(∇_x L), x_0−ε, x_0+ε)`. This is the PGD attack. FGSM is the single-step special case with α = ε.

---

## Exercise 8.3: Adversarial Training

Adversarial training augments the training set with adversarial examples on-the-fly, minimising `E[max_{‖δ‖≤ε} L(y, f(x+δ))]`. The model learns to classify correctly even on worst-case perturbed inputs within the budget ε.

**Trade-off:** Adversarial training reduces adversarial vulnerability but typically lowers clean accuracy by several percentage points, because the model must fit a harder worst-case version of each training example. Training is also significantly slower since adversarial examples must be generated for every batch. Robustness is specific to the attack norm and ε budget used during training — a model hardened against ε=0.01 L∞ perturbations is not automatically robust to larger budgets or different attack types.

---

## Exercise 8.4: Generating Adversarial Examples

### Implementation

FGSM is implemented as:

```
x_adv = x + ε · sign(∇_x CrossEntropy(y, f(x)))
```

The gradient is computed via a single forward-backward pass with `requires_grad=True` on the input tensor. After adding the signed perturbation the result is clamped to the valid normalised range so that reconstructed pixel values stay in `[0, 1]`. The perturbation budget ε is applied directly in normalised pixel space.

### Qualitative observations (from generated grids)

**ε = 0.01:** Perturbations are invisible to humans. The L∞ distance in pixel space corresponds to roughly 2–3 intensity levels on a 0–255 scale. Clean and adversarial images are visually indistinguishable. Despite invisibility, both working models are already heavily fooled at this budget.

**ε = 0.05:** A very faint structured noise pattern becomes visible on close inspection, particularly in uniform-colour regions such as sky or road surface. Most observers would not flag these images as tampered without direct comparison to the clean version.

**ε = 0.10:** The perturbation is clearly visible as a structured texture or striping artefact. Edges and flat regions show obvious noise. A careful observer would notice something is wrong.

**Threshold:** Perturbations become reliably visible to a human observer at approximately **ε = 0.05–0.10** in normalised space.

---

## Exercise 8.5: Measuring Robustness

Models were evaluated on 100 randomly sampled test images per model. Adversarial examples were generated on-the-fly using FGSM at each ε.

### Results

| Model | Clean recall | ε=0.01 recall | drop | ε=0.05 recall | drop | ε=0.10 recall | drop |
|-------|-------------|--------------|------|--------------|------|--------------|------|
| pedestrian | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| traffic_light | 0.9889 | 0.0111 | −0.9778 | 0.0000 | −0.9889 | 0.0000 | −0.9889 |
| vehicle | 0.9455 | 0.5636 | −0.3818 | 0.0727 | −0.8727 | 0.0545 | −0.8909 |

### Interpretation

**Pedestrian model:** Clean recall is 0.0000 — the model collapsed during training and always predicts "no pedestrian" regardless of input. This is a class imbalance failure: with approximately 20% positive rate in the dataset and unweighted cross-entropy loss, predicting all-negative minimises training loss. Because the model never predicts class 1, FGSM has nothing to attack and the drop is trivially zero. This is itself a critical safety finding independent of adversarial robustness: a model with zero clean recall provides no protection against missed pedestrians in any operating condition.

**Traffic light model:** Catastrophically vulnerable. Clean recall of 0.9889 collapses to 0.0111 at ε=0.01 — a drop of 0.9778 from a perturbation completely invisible to humans (roughly 2–3 intensity levels out of 255). At ε=0.05 and ε=0.10 recall reaches 0.0000. The model is effectively fully fooled by imperceptible noise.

**Vehicle model:** Degradation is more gradual but still severe. Clean recall of 0.9455 drops to 0.5636 at ε=0.01 and further to 0.0727 at ε=0.05 — an 87% relative drop at a perturbation level that is barely visible. At ε=0.10 recall is 0.0545, essentially zero.

**Overall conclusion:** All three models are highly vulnerable to FGSM. The two models with non-zero clean recall lose nearly all of it at ε=0.05, which is at the boundary of human perception. Standard ResNet-18 classifiers trained without adversarial augmentation cannot be trusted in any setting where adversarial inputs are a realistic threat.

---

## Exercise 8.6: Extending the Safety Analysis for Adversarial Robustness

### Q1 — Hazard

Existing hazard (from Exercise 2.4): *"The vehicle fails to brake for a pedestrian who is present."*

**Refined / added hazard:**
> **H-ADV:** The vehicle fails to brake for a pedestrian because the perception model's input has been adversarially perturbed, causing a false-negative output, and no detection mechanism flags this.

This is distinct from H-OOD (Sheet 9): the input is visually in-distribution and will not be flagged by an OOD monitor. The perturbation is specifically engineered to exploit the model's decision boundary, and the model outputs a high-confidence wrong prediction.

Note: the pedestrian model's collapse to zero clean recall is a separate but equally severe hazard — H-ADV is not even the binding constraint for that model since it fails on clean inputs too.

---

### Q2 — Unsafe Control Action

> **UCA-ADV:** The planner issues a "continue at speed" command while the pedestrian classifier has been fooled by an adversarial perturbation (outputting "no pedestrian" for a frame containing a pedestrian), and the perturbed input is not detected by any monitoring component.

**Linked hazard:** UCA-ADV directly causes H-ADV — no braking command is issued, the vehicle continues at speed, and a collision with the pedestrian may result.

**Context:** This UCA becomes relevant when the vehicle's camera feed can be influenced by an attacker — for example via adversarial patches placed on the road surface or on pedestrian clothing, or through a compromised image preprocessing pipeline. Given the measured results (traffic light recall collapses to 0.0111 at ε=0.01), the attack budget required is well within what a physical adversarial patch can achieve.

---

### Q3 — Safety Constraints

**Model-level constraint (adversarial robustness):**
> *The traffic light and vehicle classifiers shall maintain recall ≥ 0.80 at ε=0.01 (L∞, normalised space), as measured on a held-out evaluation set of at least 100 images using FGSM. Current models fail this constraint — traffic light recall drops to 0.0111 and vehicle recall to 0.5636 at ε=0.01 — and must undergo adversarial training before deployment. The pedestrian classifier must additionally achieve non-zero clean recall (≥ 0.80) before any adversarial robustness constraint can be evaluated; its current collapse to 0.0000 clean recall represents a prerequisite safety failure.*

Rationale: ε=0.01 is completely imperceptible to humans and represents a conservative lower bound on what a physical adversarial patch can achieve. A recall drop of 0.9778 (traffic light) at this budget is unacceptable for a safety-critical perception component.

**System-level constraint (anomaly response):**
> *When any classifier outputs a prediction confidence below θ=0.70 on any single frame, or when the prediction flips between consecutive frames without a corresponding change in scene geometry (temporal consistency check), the vehicle shall reduce speed to ≤ 15 km/h and alert the safety monitor. The system shall not resume normal speed until five consecutive frames produce high-confidence consistent predictions.*

Rationale: adversarial perturbations often produce unstable predictions across slight input variations (e.g. between consecutive video frames). A temporal consistency check exploits this instability as a detection signal even when a single-frame confidence check fails. The 15 km/h limit matches the existing safety constraint from Sheet 5 and provides a consistent fallback policy across perception failure modes.

---

### Q4 — Residual Risk

Even with adversarial training that meets the model-level constraint, significant residual risk remains:

**1. Adaptive attacks:** An adversary who knows the model has been adversarially trained can generate adaptive adversarial examples specifically designed to defeat the hardened model. Adversarial training increases robustness against the training attack type and budget but provides no provable guarantee. A stronger attacker using PGD with more iterations or a larger budget will eventually find perturbations that fool the robust model.

**2. Budget boundary:** The constraint specifies ε=0.01. A perturbation at ε=0.02 is not covered. Physical adversarial patches in the real world are not constrained to any particular ε and can be much larger in L∞ norm while still appearing innocuous as a printed sticker or road marking.

**3. Clean accuracy trade-off:** Adversarial training typically reduces clean recall by several percentage points. If clean recall drops below an operational threshold this introduces more false negatives on normal inputs — potentially creating a different safety hazard independent of any attack.

**4. Pedestrian model prerequisite failure:** The pedestrian model's zero clean recall must be resolved through retraining with class-balanced loss before adversarial robustness can even be assessed. Until then, H-ADV is dominated by the more fundamental hazard of zero-recall perception on clean inputs.

**Conclusion:** Adversarial training and temporal consistency monitoring substantially reduce UCA-ADV for the traffic light and vehicle models, but residual risk requires additional defences including certified robustness verification, input preprocessing (e.g. feature squeezing), redundant sensor modalities, and a conservative fallback policy for any anomalous perception output. The pedestrian model requires retraining as a prerequisite to all of the above.
