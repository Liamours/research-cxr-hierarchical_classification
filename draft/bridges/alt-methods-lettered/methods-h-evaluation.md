# H. Evaluation

All metrics apply the per-finding applicability mask (Section A): a finding
not annotated for a given image does not contribute to any metric computed
over that image.

Discrimination. Area under the receiver operating characteristic curve
(AUROC) and mean average precision (mAP), reported as macro and micro
averages over findings with training-split signal (Section C) per split.

Thresholded classification. At a fixed operating threshold of 0.5, we report
precision, recall (sensitivity), specificity, accuracy, F1, balanced
accuracy, Matthews correlation coefficient, and subset accuracy (exact-match
across all findings), each as macro, micro, and support-weighted averages
over findings with training-split signal.

Calibration. Expected Calibration Error (ECE), computed with 15 equal-width
confidence bins, with reliability diagrams.

Selective prediction. Area under the risk-coverage curve (AURC), computed
with the plug-in estimator of Zhou et al. (2025). Coverage is the fraction of
predictions the model commits to; risk is the error rate among committed
decisions.

Hierarchical consistency. The fraction of positive child predictions whose
parent finding is also predicted positive (hierarchy violation rate is its
complement), computed on raw model predictions and, where the fallback gate
(Section G) is active, after the gate is applied.

Localization. Gradient-weighted Class Activation Mapping (Grad-CAM) applied
at the final convolutional block of the backbone. Where bounding-box
annotations are available, we report the pointing-game hit rate and
intersection over union between the thresholded activation map and the
annotated region.

Statistical reporting. All headline metrics are reported with 95 percent
confidence intervals from bootstrap resampling. AUROC differences between
conditions are compared with DeLong's test; thresholded-prediction
differences are compared with McNemar's test. Metrics are reported as
decimals to four places.
