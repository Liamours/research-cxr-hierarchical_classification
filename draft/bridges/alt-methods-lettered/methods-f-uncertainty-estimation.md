# F. Uncertainty estimation

At inference we estimate per-finding uncertainty with Monte Carlo Dropout
(Gal and Ghahramani 2016). The dropout layer preceding the classification
head (Section D) is kept active, and each image is forwarded through the
model T = 30 times with independently sampled dropout masks. For each finding
c we record the predictive mean mu_c and predictive variance sigma_c^2 across
the T passes. Training itself remains a single deterministic forward pass;
Monte Carlo Dropout is applied only at evaluation and inference time.
