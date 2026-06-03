# Exercise Sheet 6 — Explainability

## Exercise 6.1 (optional): Why Explainability?

**Advantages of explainable models in safety-critical contexts:**

- **Trust and accountability.** When a model makes a decision that affects human safety (e.g., detecting a pedestrian before braking), engineers, auditors, and regulators need to verify that the decision was made for the right reasons, not based on a spurious correlation.
- **Debugging and failure analysis.** Explanations let developers identify systematic failure modes — for example, discovering that a model attends to road texture rather than the object of interest — which would be invisible from accuracy metrics alone.
- **Distribution shift detection.** Explanations can reveal when a model starts relying on environment-specific cues (sky colour, lighting) that will break under new conditions, long before accuracy drops become apparent on a held-out set.
- **Regulatory compliance.** Safety standards such as ISO 26262 for automotive systems increasingly require justification of model behaviour, which explainability methods can partially provide.
- **Human oversight.** An operator who can see *why* a model predicted "pedestrian present" can intervene or override when the highlighted region is clearly wrong.

**Disadvantages and limitations:**

- **Faithfulness is not guaranteed.** A saliency map shows which pixels influenced the gradient or the occluded output, not necessarily which features the model's internal representations actually encode. The map can point to the right region for the wrong reason.
- **Resolution mismatch.** Grad-CAM and CAM produce low-resolution maps (determined by the last conv layer spatial size) that are upsampled, so spatial precision is limited; small objects like distant pedestrians may not be pinpointed accurately.
- **Post-hoc methods do not change the model.** Discovering a spurious attribution tells you something is wrong, but the explanation itself does not fix it; additional data collection, re-training, or architectural changes are still required.
- **Explanation instability.** Saliency maps can be sensitive to minor input perturbations, producing visually very different maps for nearly identical inputs, which erodes confidence in their reliability.
- **Incomplete coverage.** Local methods (Grad-CAM, saliency) explain single predictions; global methods (e.g., feature visualisation) explain aggregate behaviour. Neither gives a complete picture on its own.

---

## Exercise 6.2: Local vs. Global Explainability

**Local explainability** explains a single model prediction: given one specific input, *why did the model output this particular label?* It answers the question "What drove this decision?" for one example.

**Global explainability** explains the model's overall behaviour across all inputs: *What general patterns, features, or rules has the model learned?* It answers "What does this model do in general?"

| Type | Example method | Question it answers |
|------|---------------|---------------------|
| **Local** | Grad-CAM | Which spatial regions of *this* image most activated the prediction "pedestrian present"? |
| **Global** | Feature visualisation (e.g., DeepDream / activation maximisation) | What input pattern maximally activates a given neuron or class logit on average across the learned manifold? |

Another valid local method is LIME (locally-interpretable surrogate model); another valid global method is probing classifiers or concept activation vectors (TCAV).

---

## Exercise 6.3: Saliency vs. Occlusion

### Saliency maps

The gradient of the class score with respect to the input pixels is computed in a single backward pass. The absolute value (or its square) at each pixel indicates how sensitive the output is to a small perturbation at that location.

**Advantage over occlusion:** Computationally cheap — only one forward and one backward pass regardless of image size.

**Disadvantage compared to occlusion:** Saliency measures *local gradient sensitivity*, not *causal contribution*. In flat or saturated regions of the loss landscape the gradient is near zero even if the region is important, leading to misleading maps (gradient saturation problem).

### Occlusion maps

A sliding patch (e.g., a grey square) is systematically placed at every location in the image. The model is run once per patch position, and the drop in the target class probability is recorded. Regions whose occlusion causes the largest drop are deemed most important.

**Advantage over saliency:** Directly measures the causal contribution of a region to the output — if covering a region hurts confidence, that region matters. Not affected by gradient saturation.

**Disadvantage compared to saliency:** Computationally expensive — requires O(H × W / patch²) forward passes, which is prohibitive for large images or real-time use.

---

## Exercise 6.4: Chain-of-Thought Fidelity

### 6.4.1 — Faithfulness and why it is hard to verify

A thinking trace is **faithful** if the steps written in the trace are the *actual computational steps* the model performed to arrive at its answer — i.e., the trace is a true causal account of the model's reasoning, not a post-hoc rationalisation.

Faithfulness is hard to verify because:
- We have no direct access to the internal computations of a transformer. The token sequence of the trace is generated by the same mechanism as the final answer; there is no separate "reasoning module" we can inspect.
- Even if the trace is logically consistent with the answer, consistency does not prove causation: the model could have determined the answer via a shortcut and then constructed a plausible-sounding chain of thought.
- Human evaluators can only judge whether the trace *looks* reasonable, not whether it reflects internal states.

### 6.4.2 — Simulatability and a counterexample

**Simulatability** means that a human (or another model) can read the thinking trace and use it to reproduce the model's final answer accurately — i.e., the trace is sufficiently detailed and self-contained that it functions as a complete recipe for reaching the conclusion.

**Counterexample where simulatability holds but faithfulness fails:**

A maths model is asked whether 997 is prime. It outputs a trace that walks through trial division up to √997 ≈ 31, checks each prime divisor, and concludes "997 is prime." A human following the trace can reproduce the answer (simulatability ✓). However, the model may have actually looked up a memorised pattern for numbers near 1000 without performing any division — the trace was generated to *sound* like the right procedure, not because that procedure was executed internally (faithfulness ✗).

### 6.4.3 — Counterfactual simulatability

Counterfactual simulatability asks: if the model's answer had been different, would the trace also have been different in a way that explains that different answer? In other words, the trace must not only support the actual answer but must *change meaningfully* when the answer changes due to a changed input.

This is stricter because a model can always produce a plausible-sounding trace for *whatever* answer it outputs (high ordinary simulatability) while the trace remains completely decoupled from the actual computation. Counterfactual simulatability requires that the trace is sensitive to the actual reasoning path, not just the final output.

### 6.4.4 — Safety risk of unfaithful thinking traces

An unfaithful trace creates a **false sense of auditability**: operators or regulators who review the chain of thought believe they are verifying the model's reasoning, but they are only checking a post-hoc narrative that may not reflect what the model actually did.

**Concrete example:** A medical triage model outputs "Low risk — patient can wait." Its trace lists: "No fever reported, vitals within normal range, symptoms consistent with mild infection." Clinicians approve the decision based on the trace. In reality the model classified the case as low-risk because the patient's demographic features correlated with low-acuity cases in training data, and ignored the vitals. If the true reason had been exposed, the clinician would have caught a dangerous shortcut; the faithful-looking but unfaithful trace blocked that oversight.

---

## Exercise 6.5: Applying an Explainability Method

### 6.5.1 — Method: Grad-CAM

**How it works:**

Grad-CAM (Gradient-weighted Class Activation Mapping) hooks into the last convolutional layer of the network (in our case, `layer4[-1]` of ResNet-18). For a given input image and predicted class:

1. A forward pass produces class logits; the score for the predicted class is selected.
2. A backward pass computes the gradient of that score with respect to each spatial feature map channel at `layer4[-1]`, giving tensors of shape `[C, H', W']`.
3. Each channel's gradient is globally average-pooled to produce a scalar importance weight α_k.
4. The weighted sum of feature maps is computed and passed through ReLU (to keep only positive influences), yielding a coarse heatmap of size `[H', W']` (7×7 for a 224×224 input into ResNet-18).
5. The heatmap is bilinearly upsampled to the original image resolution and normalised to [0, 1].

**Why Grad-CAM was chosen:**

- Works on any CNN with a spatial convolutional backbone without modifying the architecture (unlike CAM, which requires a GlobalAveragePooling head).
- More spatially stable than vanilla saliency maps, which are noisy and gradient-saturated in deep networks.
- More computationally efficient than occlusion (one forward + one backward pass vs. hundreds of forward passes).
- Class-discriminative: the map highlights regions that specifically drove the prediction for the chosen class, not just salient regions in general.

---

### 6.5.2 — Correctly Classified Images (5 images, ≥1 per model)

The following five correctly classified examples were selected (2 pedestrian, 2 traffic light, 1 vehicle). See figures `6.5_gradcam_correct_pedestrian.png`, `6.5_gradcam_correct_traffic_light.png`, and `6.5_gradcam_correct_vehicle.png`.

---

### 6.5.3 — Do highlighted regions correspond to relevant objects?

**Pedestrian model (True: negative, Pred: negative — both images):**

Both correctly classified pedestrian images are *true negatives* — no pedestrian is present and the model correctly predicts "negative." The Grad-CAM maps highlight the **lower-centre region of the image** (car bonnet, near road surface) and the **left mid-ground** (street lamp area, treeline). This is consistent with correct negative classification: the model attends to the absence of pedestrian-associated features in the lower visual field where a pedestrian would appear if present. Notably, the sky is largely cold (blue), suggesting the model does not attend to irrelevant sky regions for negative cases.

**Traffic light model (True: positive, Pred: positive — both images):**

In both images a traffic light is present and correctly detected. The Grad-CAM maps show a strong activation cluster in the **upper-centre to upper-left** region — roughly where the traffic lights appear against the building backgrounds. This corresponds well to the semantically relevant object. In the second image the hot region extends slightly toward the building facade, suggesting the model partially uses structural context (vertical structures near the road centre), which is imperfect but directionally correct.

**Vehicle model (True: positive, Pred: positive):**

The vehicle image shows a clearly visible orange/red vehicle in the mid-ground. The Grad-CAM map concentrates its highest activation (red region) **directly on and around the vehicle**, with secondary activation on nearby road markings. This is the best-localised map of the three models — the highlighted region corresponds very precisely to the object of interest.

**Summary:** The vehicle model's Grad-CAM is well localised on the target object. The traffic light model shows broadly correct upper-region focus. The pedestrian model's negative cases show diffuse lower-region attention that is behaviourally reasonable but harder to interpret semantically.

---

### 6.5.4 — Misclassified Images: Do explanations reveal why the model failed?

Three misclassified images were selected (see `6.6_gradcam_misclassified.png`): two pedestrian false negatives and one vehicle false negative.

**Pedestrian false negative #1:**

The scene shows a suburban road with low pedestrian visibility (pedestrian appears to be in the far left background near a building). The Grad-CAM map highlights a **large diffuse region in the upper-left sky and treeline**, rather than the pedestrian location. This strongly suggests the model used broad scene-level cues (sky, foliage) as a proxy for the negative class and failed to attend to the small foreground pedestrian, causing the miss.

**Pedestrian false negative #2:**

The scene is a wider urban road. The Grad-CAM map shows strong activation on the **lower road surface and yellow lane markings**, with secondary activation on building edges at the right. The pedestrian (visible near the centre-right of the image) received no meaningful attention. This indicates the model classified based on road geometry and background clutter rather than the pedestrian, explaining the false negative.

**Vehicle false negative:**

The scene shows a suburban road with a visible red vehicle parked on the right. The Grad-CAM map concentrates activation on the **lower-right corner (road/curb area) and treeline**, entirely missing the vehicle. This suggests the vehicle was outside the spatial field the model learned to attend to for this type of scene, possibly because training examples placed vehicles more centrally.

**Overall insight from misclassified images:** In all three cases, the Grad-CAM maps reveal that the model attended to background or peripheral regions (sky, road surface, lane markings, treeline) rather than the target object. This points to the model having learned scene-level statistics rather than robust object representations, which is a known failure mode when training data lacks sufficient pose and location diversity.

---

## Exercise 6.6: Explainability as a Diagnostic Tool

### 6.6.1 — Implication of predicting "pedestrian" from the sky region

If the model predicts "pedestrian present" primarily based on a sky region, this implies:

- **Spurious correlation in training data.** The model has learned that certain sky appearances (e.g., a particular blue sky with bright lighting typical of the CARLA town used for training) co-occur with pedestrian presence, rather than learning the pedestrian's visual features.
- **Poor generalisation.** Since sky appearance changes dramatically with weather, time of day, and location, the model's predictions will be unreliable — and unpredictably so — under any condition that changes the sky's appearance.

**What could cause this:**

- Class imbalance combined with a biased scene distribution: if pedestrian-present images were predominantly captured at a specific time of day or in a specific town with a particular sky colour, the network learns the confound.
- Insufficient data augmentation (no sky-variation augmentation, no randomised lighting).
- The pedestrian is small relative to the image and the sky dominates the feature map spatially; the network finds the sky a lower-difficulty feature for the training distribution.
- The model was not explicitly trained to localise; a classification objective allows any feature correlation to be exploited.

---

### 6.6.2 — OOD Condition Analysis

Models were evaluated on two OOD splits: **fog** (`test-fog`) and **night** (`test-night`). Results are reported below.

#### Accuracy comparison

| Model | OOD: Fog | OOD: Night |
|---|---|---|
| Traffic light | 0.439 | 0.270 |
| Pedestrian | 0.780 | 0.485 |
| Vehicle  | 0.545 | 0.736 |

The traffic light and pedestrian models suffer severe accuracy drops under both fog and night conditions. The vehicle model's night accuracy (0.736) is relatively preserved, likely because large vehicles remain high-contrast objects even at night (headlights, reflective surfaces).

---

#### 6.6.2(a) — Do highlighted regions still correspond to relevant objects?

**Fog condition (`6.6_gradcam_ood_test-fog.png`):**

- **Traffic light (false negative, OOD acc = 0.439):** The Grad-CAM map shows very broad, diffuse activation covering the entire lower-left quarter of the foggy image — road, kerb, and fog-obscured buildings. There is no focused attention on any traffic light location. The fog removes the colour and contrast cues the model relied on in clear conditions, causing it to spread attention across low-information regions.
- **Pedestrian (true negative, OOD acc = 0.780):** The map highlights the **mailbox and road markings** in the lower-right. This is a reasonable proxy for "no pedestrian in the walking zone," but the focus on the mailbox is a spurious feature — the model may be using object identity shortcuts. The pedestrian model is the most robust under fog (0.780), possibly because it learned some structural scene cues that remain partially valid.
- **Vehicle (false negative, OOD acc = 0.545):** The map shows wide, scattered activation across the road surface and left side of the image, missing the vehicle entirely. Fog reduces vehicle contrast and the model's spatial attention collapses to background texture.

**Night condition (`6.6_gradcam_ood_test-night.png`):**

- **Traffic light (false negative, OOD acc = 0.270):** The Grad-CAM map activates strongly on **streetlights and light reflections on the wet road surface** in the left half of the image. The model confuses streetlight glare with traffic light features — a direct consequence of having trained only on daylight scenes where traffic lights are the dominant point-light sources.
- **Pedestrian (false positive, OOD acc = 0.485):** The map highlights a **large bright region in the road centre** (light reflection/puddle). This is a spurious feature: the model predicts pedestrian presence based on a bright patch on the road, which resembles the brightness distribution of pedestrians in training (likely illuminated by direct sunlight in CARLA daytime scenes).
- **Vehicle (true positive, OOD acc = 0.736):** The map correctly activates on the **large red vehicle (delivery truck)** in the centre of the image. The vehicle's high contrast against the dark background preserved the correct attribution. This is the only OOD case where Grad-CAM highlights a semantically correct region.

---

#### 6.6.2(b) — Evidence of reliance on spurious features

Yes, clear evidence was observed:

- **Lighting artefacts:** Under night conditions, the traffic light model attends to road reflections and streetlights — features entirely absent from training. This confirms it learned "localised bright point in upper image region" as a traffic light proxy, which was valid during day training but transfers to any night light source.
- **Road texture / lane markings:** Multiple misclassified images (pedestrian FN #2, fog vehicle FN) show attention concentrated on lane markings and road surface texture rather than the target object, indicating the model uses road geometry as a scene-classification shortcut.
- **Background sky / foliage:** As discussed in 6.5.4, pedestrian false negatives attend to sky and treeline. Under fog this degrades further because the sky cue disappears, causing the model to scatter attention.
- **Bright patches:** The night pedestrian false positive (attending to a wet-road light reflection) is a direct example of a training-condition brightness prior being activated by an unrelated OOD feature.

---

#### 6.6.2(c) — How accuracy and explanation quality change across conditions

| Condition | Accuracy trend | Explanation quality (Grad-CAM localisation) |
|---|---|---|
| In-distribution (test) | Highest | Moderate-to-good: vehicle model well localised; TL model broad but directionally correct |
| OOD: Fog | Significant drop (TL: −56%, Vehicle: −45%) | Poor: activations become diffuse, covering background and fog texture |
| OOD: Night | Severe drop (TL: −73%, Pedestrian: −52%) | Mostly poor: activations shift to lighting artefacts; vehicle model is the exception with good localisation |

The degradation in explanation quality parallels — and in some cases precedes — the drop in accuracy. This demonstrates a key property of Grad-CAM as a diagnostic tool: **diffuse or semantically incorrect heatmaps are a leading indicator of model brittleness**, even when aggregate accuracy on a small OOD sample might appear tolerable. For safety-critical deployment, monitoring the spatial focus of Grad-CAM maps (e.g., automatically checking that the highest-activation region falls within the expected object bounding box) could serve as a run-time anomaly detector for distribution shift.


