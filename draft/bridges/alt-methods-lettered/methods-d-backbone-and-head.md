# D. Backbone and classification head

We use DenseNet121 pretrained on a multi-institution chest radiograph corpus
via TorchXRayVision (weights: densenet121-res224-all, trained on NIH
ChestX-ray14, PadChest, CheXpert, MIMIC-CXR, Open-I, and RSNA Pneumonia). The
backbone maps a 224 by 224 grayscale input to a 1024-dimensional pooled
feature vector (global feature extraction followed by ReLU and adaptive
average pooling). We choose this backbone for two reasons: first, its
pretraining on large, heterogeneous chest radiograph data gives domain
specific low-level feature detectors that are not available in
ImageNet-pretrained models; second, it is a well-characterized convolutional
architecture whose feature maps are amenable to gradient-based localization
(Section H). The backbone is fine-tuned end-to-end on the downstream label
set after a one-epoch frozen warm-up.

A single linear classification head maps the 1024-dimensional feature to 51
logits, one per canonical finding, each followed by a sigmoid. A dropout
layer (p = 0.2) precedes the head and is reused at inference for uncertainty
estimation (Section F). There is no separate coarse-group head: parent
findings are themselves part of the 51-label output, so the parent
probability at any IS-A edge is read directly from the corresponding sigmoid
output. The full model has 7,018,309 parameters (6,966,034 in the backbone,
52,275 in the head) and requires 2.755 GMACs per 224 by 224 image.
