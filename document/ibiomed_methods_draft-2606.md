iBioMed Paper Draft — Title and Methods
status: methods drafted; Results to be added after real-data runs.
backbone: DenseNet121-XRV (densenet121-res224-all), single backbone, 8-condition grid.
Authoritative grid: configs/grid_base.yaml + configs/grid/*.yaml


TITLE

Primary:
When Unsure, Predict Broader: Uncertainty-Gated Coarse Fallback for Hierarchical
Multi-Label Chest X-Ray Classification

Alternatives:
- Uncertainty-Gated Coarse Fallback for Hierarchical Multi-Label Chest X-Ray
  Classification with MC-Dropout
- Trustworthy Hierarchical Chest X-Ray Classification via Uncertainty-Gated
  Coarse Fallback


METHODS

A. Problem formulation and notation
We address multi-label classification of a frontal chest radiograph into N = 51
thoracic findings. For an image x, the model produces label probabilities
p in [0,1]^51, where each entry is the probability that a finding is present.
A subset of the 51 findings stand in IS-A relationships: for a parent finding g
with children L(g), the parent is clinically implied by any of its children.
The model may report at the child level, fall back to the parent level, or
abstain, depending on its estimated uncertainty.

B. Datasets and preprocessing
We combine six publicly available chest radiograph collections: CheXpert,
NIH ChestX-ray14, VinDr-CXR, PadChest, Open-I, and RSNA Pneumonia Detection
(2018). After removing duplicates and studies without usable frontal views, the
combined dataset comprises 417,136 studies. All datasets are mapped to the shared
51-label canonical set (src/data/label_space.py CANONICAL_LABELS); labels a
dataset does not annotate are stored as not-applicable and excluded from both
training and scoring for that dataset.

Each study is reduced to a single frontal image, selecting by view priority PA
over AP over other frontal views. Each image is converted to grayscale and
rescaled to the TorchXRayVision convention (pixel range [-1024, 1024]).
Images are resized so the shorter side is 256 pixels using bicubic interpolation
and center-cropped to 224 by 224 pixels. For DICOM sources, pixel values are
rescaled with the stored slope and intercept, inverted when the photometric
interpretation is MONOCHROME1, and clipped to the 0.5th and 99.5th percentiles
before scaling.

For uncertain label encoding (e.g., CheXpert's -1 "uncertain" mentions),
uncertain labels are mapped to 0 (negative); this is the U-Zero convention.
Train, validation, and test partitions follow the official patient-level splits
of each source dataset, so no patient appears in more than one partition.

Training-time augmentation, applied only to the training split, is a small
affine transform (rotation up to 10 degrees, translation up to 5 percent, scale
within [0.95, 1.05]) and mild brightness and contrast jitter (factor 0.1).
Horizontal flipping is disabled: it mirrors the cardiac silhouette and aorta,
which corresponds to the rare situs inversus anatomy.

C. Label hierarchy
The 51 canonical findings are drawn from Indonesian clinical guidelines
(PDPI, PNPK, KKI), representing the full scope of thoracic findings relevant to
the target clinical deployment context. Because public datasets annotate
radiographic findings while the canonical set is clinical diagnoses, only a
subset of the 51 labels currently receives training signal from the implemented
dataset adapters; the remainder are defined by clinical authority and supervised
wherever a dataset provides them (otherwise masked).

For the hierarchical training and inference modes, we define 13 IS-A edges over
the 51 labels, grouping more specific findings under clinically broader parents:
Pneumonia is parent of COVID-19 Pneumonia, Aspiration Pneumonia, and Other Viral
Pneumonia; ILD is parent of IPF, COP, HP, Silicosis, Asbestosis, Other
Pneumoconiosis, and Sarcoidosis; TB is parent of Post-TB Obstructive Disease;
Pleural Effusion is parent of Pleural Empyema; Pulmonary Hypertension is parent
of Cor Pulmonale. These edges are clinically grounded IS-A relationships and are
not an official taxonomy. Supplying the hierarchy as an external file allows a
different set of edges to be used without changing code.

D. Backbone and classification head
We use DenseNet121 pretrained on a multi-institution chest radiograph corpus
via TorchXRayVision (weights: densenet121-res224-all, trained on NIH ChestX-ray14,
PadChest, CheXpert, MIMIC-CXR, Open-I, and RSNA Pneumonia). The backbone
produces a 1024-dimensional pooled feature vector. We choose this backbone for
two reasons: (1) its pretraining on large, heterogeneous CXR data gives
domain-specific low-level feature detectors that are not available in ImageNet-
pretrained models; (2) it is a well-characterized convolutional architecture
whose feature maps are amenable to gradient-based localization. The backbone is
fine-tuned end-to-end on the downstream label set after a one-epoch frozen
warm-up.

A single linear classification head maps the 1024-dim feature to 51 logits,
one per canonical finding, each followed by a sigmoid. A dropout layer
(p = 0.2) precedes the head and is reused at inference for uncertainty
estimation. There is no separate coarse-group head; the parent findings are
themselves part of the 51-label output, so the parent probability at any IS-A
edge is read directly from the corresponding sigmoid output.

E. Training objective and optimization
For flat training we minimize masked binary cross-entropy. Writing sigma for
the sigmoid function, y_c for the binary target of finding c, and m_c for the
applicability mask (1 when annotated for that dataset, 0 otherwise):

  L_flat = ( sum_c m_c * BCE(sigma(z_c), y_c) ) / ( sum_c m_c )

Not-applicable findings contribute neither to the numerator nor the denominator.

For hierarchical training we add a differentiable consistency penalty
(Asadi et al. 2025, CIHMLC). For each IS-A edge (parent g, child c):

  penalty(g, c) = ReLU(p_c - 0.5) * ReLU(0.5 - p_g)

This term is positive only when the child probability exceeds 0.5 while the
parent probability is below 0.5 — a hierarchically inconsistent prediction.
The hierarchical loss is:

  L_HBCE = L_flat + lambda * sum_{(g,c) in edges} penalty(g, c)

where lambda = 0.5. The penalty is zero when predictions are already consistent,
so it does not distort well-calibrated outputs.

The model is trained with AdamW using differential learning rates: 1e-5 for the
backbone and 1e-4 for the head, with weight decay 0.01. The learning rate
schedule is cosine annealing with a linear warm-up of 1,000 steps. Training uses
bfloat16 mixed precision and gradient accumulation (effective batch size 32).
Gradients are clipped to a maximum norm of 1.0. Training stops when the
validation loss does not improve for three consecutive epochs, and the checkpoint
with the lowest validation loss is retained.

F. Uncertainty estimation
At inference we estimate per-finding uncertainty with Monte Carlo Dropout (Gal &
Ghahramani 2016). The dropout layer is kept active and the image is forwarded
through the model T = 30 times with different dropout masks. For each finding c
we record the predictive mean mu_c and the predictive standard deviation sigma_c
across the T passes. sigma_c serves as the uncertainty signal.

G. Uncertainty-Gated Coarse Fallback
For each IS-A edge (parent g, child c), we apply a gate at inference. Let tau_u
be an uncertainty threshold and tau_p a probability threshold. The rule for the
pair (g, c) is:

  1. If mu_c >= tau_p and sigma_c <= tau_u: report the child finding c.
  2. Otherwise, if mu_g >= tau_p: fall back and report the parent finding g
     (suppressing the child prediction).
  3. Otherwise, abstain for this pair.

Findings with no IS-A parent follow rule 1 only (report if mu_c >= tau_p,
abstain otherwise). The parent signal mu_g is taken directly from the same
sigmoid head that produces mu_c — no separate head or aggregation is needed
because the parent is itself one of the 51 canonical findings.

The gate trades prediction specificity for reliability: a smaller tau_u causes
more child predictions to be suppressed in favor of their parents or abstention.
We select tau_u on the validation set by identifying the smallest value for which
a target hierarchical consistency rate is met.

H. Evaluation
Classification quality: area under the receiver operating characteristic curve
(AUROC) and F1, reported as macro and micro averages over applicable findings
per split. All metrics apply the applicability mask.

Calibration: Expected Calibration Error (ECE), computed with 15 equal-width
confidence bins, with reliability diagrams.

Selective quality: area under the risk-coverage curve (AURC), computed as the
plug-in estimator of Zhou et al. (2025). Coverage is the fraction of predictions
the model commits to; risk is the error rate among committed decisions. We report
AURC for the fallback model and for a flat baseline that can only predict at the
child level or abstain, isolating the contribution of the coarse fallback step
at matched coverage. A coarse prediction is counted correct when any child in
L(g) is truly positive.

Hierarchical consistency: fraction of positive child predictions whose parent
finding is also predicted positive, reported before and after the gate is applied.

Localization: Gradient-weighted Class Activation Mapping (Grad-CAM) applied at
the final convolutional block. Where bounding-box annotations are available we
report the pointing-game hit rate and intersection over union between the
thresholded activation map and the annotated region.

I. Implementation and reproducibility
All experiments are driven by a single YAML configuration file that fixes the
random seed to 42 and records the backbone, hierarchy edges, loss, optimization,
uncertainty, and evaluation settings. The configuration is snapshotted into each
run directory, and per-epoch metrics and events are logged alongside final
metrics. The 8-condition ablation grid (2 segmentation conditioning states x
2 label structures x 2 uncertainty modes) is defined in configs/grid_base.yaml
and instantiated in configs/grid/. Code and configurations are released to
support replication.


DEFAULTS (current code; move to an experimental-setup table)
  image size 224; lambda 0.5; T (MC Dropout passes) 30; backbone lr 1e-5;
  head lr 1e-4; grad clip 1.0; early stop patience 3; seed 42; dropout p 0.2.

TODO before submission
  - real-data runs -> Results section + setup table (numbers, 4 decimals).
  - choose final tau_p, tau_u on validation; report selected values.
  - confirm mechanism name (Uncertainty-Gated Coarse Fallback) in code + paper.
  - convert to IEEE conference LaTeX template for iBioMed.
  - add Related Work section (HBCE: Asadi 2025; AURC: Zhou 2025; MC-Dropout: Gal & Ghahramani 2016; TorchXRayVision: Cohen 2022).
