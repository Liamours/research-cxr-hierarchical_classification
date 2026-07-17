# A. Problem formulation and notation

We address multi-label classification of a frontal chest radiograph into
N = 51 thoracic findings. For an image x, the model produces label
probabilities p in [0,1]^51, one entry per finding. A subset of the 51
findings stand in IS-A relationships: for a parent finding g with children
L(g), the parent is clinically implied by any of its children. The model may
report at the child level, fall back to the parent level, or abstain,
depending on its estimated uncertainty.

Each source dataset annotates only a subset of the 51 findings. For a given
image, a finding not annotated by its source dataset is not-applicable rather
than negative, and is excluded from both training and scoring for that image
(the applicability mask m_c, used throughout Sections E and H).
