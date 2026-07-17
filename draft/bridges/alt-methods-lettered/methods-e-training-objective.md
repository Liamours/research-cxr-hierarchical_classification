# E. Training objective and optimization

We compare three loss-function conditions under one fixed backbone and
otherwise identical training procedure. Writing sigma for the sigmoid
function, z_c for the raw logit of finding c, y_c for its binary target, and
m_c for the applicability mask (1 when the source dataset annotates finding
c for that image, 0 otherwise):

Flat condition. We minimize masked binary cross-entropy:

  L_flat = ( sum_c m_c * BCE(sigma(z_c), y_c) ) / ( sum_c m_c )

Not-applicable findings contribute neither to the numerator nor the
denominator.

Hierarchical condition. We add a differentiable consistency penalty (Asadi
et al. 2025, CIHMLC) over the 13 IS-A edges (Section C). For each edge
(parent g, child c):

  penalty(g, c) = ReLU(p_c - 0.5) * ReLU(0.5 - p_g)

where p_c = sigma(z_c). This term is positive only when the child probability
exceeds 0.5 while the parent probability is below 0.5, a hierarchically
inconsistent prediction, and is masked to edges where the child label is
annotated. The hierarchical loss is:

  L_HBCE = L_flat + lambda * mean_{(g,c) in edges}[ penalty(g, c) * m_c ]

with lambda = 0.5. The penalty is zero when predictions are already
consistent, so it does not distort well-calibrated outputs.

BAFL condition. We additionally evaluate a Balanced Adaptive Focal Loss,
adapted in isolation from the HP-ViT architecture (Khan et al. 2026): the
class-balanced reweighting and focal-loss mechanism of HP-ViT's loss term are
applied on top of the same masked BCE framework, without adopting HP-ViT's
attention or feature-fusion modules. For each finding c, a per-class weight
w_c is computed from the effective number of training-split positives
(Cui et al. 2019):

  w_c = (1 - beta) / (1 - beta^{n_c}),  beta = 0.999

normalized to mean 1 over findings with at least one training-split positive.
A focal term down-weights confidently correct predictions, with a focusing
exponent gamma that increases linearly with training epoch t from an initial
to a final value over a warm-up period:

  L_BAFL = ( sum_c m_c * w_c * (1 - p_t,c)^{gamma(t)} * BCE(sigma(z_c), y_c) ) / ( sum_c m_c )

where p_t,c = p_c if y_c = 1 else 1 - p_c.

Optimization is identical across all three conditions: AdamW with
differential learning rates, 1e-5 for the backbone and 1e-4 for the head,
weight decay 0.01. The learning rate schedule is one-cycle cosine annealing
with a linear warm-up of 1,000 steps. Training uses bfloat16 mixed precision,
batch size 32, and gradient clipping to a maximum norm of 1.0. The backbone
is frozen for the first epoch. Training runs for up to 15 epochs with early
stopping on validation loss (patience 3); the checkpoint with the highest
validation macro AUROC is retained for evaluation. Seed 42 is fixed across
random, NumPy, and PyTorch.
