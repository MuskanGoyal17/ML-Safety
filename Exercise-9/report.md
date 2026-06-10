# Exercise Sheet 9

## Exercise 9.1 - The OOD Problem

### 9.1.1 Why can a standard classifier not be trusted to signal when it receives an OOD input?

Standard classifiers are trained to predict one of the classes seen during training. 
The softmax layer always produces a probability distribution over known classes, even when the input is unlike anything encountered during training. 
As a result, the model may produce highly confident predictions for completely unfamiliar inputs.

### 9.1.2 Why is silent failure worse than uncertain failure in a safety-critical system?

Silent failures are dangerous because the system acts on incorrect information without recognising that it may be wrong. 
In autonomous driving, a confidently incorrect prediction can directly lead to unsafe actions. 
An uncertain prediction can instead trigger a fallback behaviour such as slowing down, requesting human intervention, or increasing monitoring.

## Exercise 9.2 - MSP Baseline OOD Detection

The Maximum Softmax Probability (MSP) baseline uses the largest softmax output as an OOD score.

For an input image:

- Run the classifier.
- Compute softmax probabilities.
- Take the maximum probability.
- Low MSP indicates possible OOD input.

Advantages:

- Simple to implement
- No retraining required
- Computationally inexpensive

Limitations:

- Neural networks can remain highly confident on OOD inputs.
- Confidence is often poorly calibrated.
- Performance degrades when OOD samples resemble in-distribution data.

## Exercise 9.3 - Mahalanobis Distance

Mahalanobis distance is a feature-based OOD detection method.

Instead of using softmax confidence, feature vectors are extracted from a deep layer of the network. 
A Gaussian distribution is fitted to the in-distribution features. New samples are scored according to their distance from this feature distribution.

Advantages over MSP:

- Uses internal representations rather than output probabilities.
- Better captures similarity to training data.
- More effective when confidence scores remain high for OOD samples.

## Exercise 9.4 - Visualising the Distribution Shift

### Observations

Fog and night images show clear distribution shifts relative to the sunny daytime training images.

- Fog reduces contrast and visibility.
- Night images contain significantly different illumination characteristics.
- The town images retain similar weather and lighting conditions but differ in scene layout, road geometry, buildings, and vegetation.

Compared with fog and night conditions, the town shift is more subtle because image appearance remains largely similar while semantic scene structure changes.

### Mean Softmax Confidence

| Condition       | Mean MSP | Std    |
| --------------- | -------- | ------ |
| In-distribution | 0.7623   | 0.1441 |
| Fog             | 0.7441   | 0.1159 |
| Night           | 0.6498   | 0.1039 |
| Town            | 0.7592   | 0.1295 |

### Discussion

The model becomes noticeably less confident under night conditions, indicating a significant distribution shift.

Fog causes only a small confidence reduction.

The unseen town produces confidence values almost identical to the in-distribution data, suggesting that confidence alone may not reliably detect this form of distribution shift.

## Exercise 9.5 - Is the Different Town Out-of-Distribution?

### 9.5.1 ODD Analysis

The original ODD definition focused primarily on weather and daytime conditions. It did not explicitly specify geographic layout or CARLA map identity.

Therefore, the status of the unseen town is ambiguous under the original ODD definition.

### 9.5.2 Revised ODD

The revised ODD is defined as:

Daytime driving under normal weather conditions across arbitrary CARLA towns and road layouts.

Under this definition, the unseen town is considered inside the ODD.

### 9.5.3 Implications

Because the town images are inside the ODD, the perception system should be expected to handle them correctly.

The OOD monitor should not automatically flag such inputs solely because they originate from a previously unseen town.

## Exercise 9.6 - Evaluating the MSP Baseline

### AUROC Results

| Scenario | AUROC  |
| -------- | ------ |
| Fog      | 0.5421 |
| Night    | 0.7243 |
| Town     | 0.5092 |
| Combined | 0.5919 |

### Discussion

The MSP baseline performs moderately on night images but poorly on fog and town scenarios.

The town AUROC is close to random guessing (0.5), demonstrating that confidence scores alone are insufficient for detecting subtle distribution shifts.

## Exercise 9.7 - Feature-Based OOD Detection

### AUROC Comparison

| Scenario | MSP    | Mahalanobis | Improvement |
| -------- | ------ | ----------- | ----------- |
| Fog      | 0.5421 | 0.9818      | +0.4397     |
| Night    | 0.7243 | 0.9998      | +0.2755     |
| Town     | 0.5092 | 0.8197      | +0.3104     |

### Discussion

The Mahalanobis detector substantially outperformed MSP in all scenarios.

The largest improvement occurred in the fog scenario (+0.4397 AUROC).

The Mahalanobis detector also achieved near-perfect separation for night images and strong performance on the unseen town scenario.

This suggests that feature-space methods capture distributional differences that are not reflected in classifier confidence.

## Exercise 9.8 - Extending the Safety Analysis for OOD

### Hazard

The vehicle operates using perception outputs derived from inputs outside the operational design domain, resulting in missed detections of pedestrians and unsafe driving behaviour.

### Unsafe Control Action

The planner continues normal driving behaviour while perception outputs are unreliable due to an undetected out-of-distribution input.

### Safety Constraints

#### Model-Level Constraint

The OOD monitoring system shall detect and flag inputs whose feature distribution differs significantly from the training distribution.

#### System-Level Constraint

When OOD input is detected, the vehicle shall transition into a safe fallback mode such as reducing speed, increasing following distance, or initiating a minimal-risk manoeuvre.

### Residual Risk

Even a perfect OOD detector does not eliminate risk.

Remaining risks include:

- Incorrect responses after OOD detection.
- Delayed detection.
- Novel hazards within the ODD.
- Sensor failures unrelated to distribution shift.

Therefore, OOD detection reduces risk but does not completely eliminate the underlying hazard.

## Conclusion

The experiments demonstrate that softmax confidence is often insufficient for reliable OOD detection. While MSP achieved only moderate performance, the Mahalanobis detector provided substantial improvements across all tested scenarios. The results highlight the importance of incorporating explicit OOD monitoring into safety-critical perception systems.
