Chest radiography remains the most widely available imaging modality for thoracic
disease screening in low-resource settings, and Indonesia's clinical guidelines
recognize a broad set of thoracic findings that a radiograph-based screening tool
should in principle cover. Public chest radiograph datasets, however, each
annotate a different and partial subset of findings, using different label
definitions and different acquisition protocols. A model trained on any single
public dataset inherits that dataset's narrow label vocabulary and cannot be
compared directly against a clinically grounded target list.

We assemble a canonical set of 51 thoracic findings grounded in Indonesian
clinical authority, specifically the Indonesian Society of Respirology's general
clinical practice guideline, the Ministry of Health's national clinical practice
guidelines, and the Indonesian Doctor Competency Standard. We pool seven public
chest radiograph datasets under this shared label space, using a masked training
objective so that each image only supervises the findings its source dataset
actually annotates. Twenty-seven of the 51 canonical findings receive training
signal from at least one of the seven pooled datasets; the remaining findings are
retained as defined clinical targets without present-day training data.

A subset of the 51 findings stand in clinically motivated is-a relationships,
for example COVID-19 pneumonia is a subtype of pneumonia, and idiopathic
pulmonary fibrosis is a subtype of interstitial lung disease. We compare two
training objectives on the same backbone and the same pooled corpus: a flat
masked binary cross-entropy loss that treats all 51 findings independently, and
a hierarchy-consistency-regularized loss (Hierarchical Binary Cross-Entropy,
HBCE) that penalizes predictions where a child finding is predicted positive
while its parent is predicted negative. We report both models' discrimination,
calibration, and selective prediction quality on held-out validation and test
partitions from the pooled corpus, and we report a localization sanity check using
Gradient-weighted Class Activation Mapping against the subset of images with
publicly available bounding-box annotations.

An uncertainty-gated coarse fallback mechanism, in which a model facing high
predictive uncertainty at the child-finding level falls back to reporting only
the parent finding, was designed as part of this project and is described in
the methodology. At the time of writing this mechanism has not been evaluated;
the Monte Carlo Dropout uncertainty runs and the fallback gate itself remain
future work, and no claim of its effectiveness is made in this draft.

This draft reports what was actually run: a comparison of flat and
hierarchy-consistency-regularized training on a pooled, clinically grounded
label space, together with an honest reading of where hierarchical training
helped, where it did not, and where the localization check fell short.
