On this pooled, seven-dataset corpus, hierarchy-consistency-regularized
training does not clearly outperform flat training. The flat condition wins
the large majority of overall discrimination metrics and most per-label
AUROC and average precision rows, and it also wins calibration on this
corpus, reversing a pattern observed before PadChest was added to the pool.
The hierarchical condition's advantages are narrower than the original
hypothesis anticipated: a lower hierarchy violation rate, better specificity,
and better AUROC on a handful of rare, PadChest-derived findings. Reported
honestly, hierarchy-consistency regularization traded some ranking
performance and calibration quality for a modest reduction in hierarchy
violations on this corpus.

Subsection 2.2 offers a structural reason for this outcome rather than a
purely empirical one. The hierarchy-consistency penalty can only receive
gradient signal on an is-a edge where both the parent finding and the child
finding are annotated for the same image, and only one of the 13 edges,
pneumonia to COVID-19 pneumonia, meets this condition in the current corpus.
The remaining 12 edges have at least one endpoint that is not-applicable
across the entire pooled dataset, so the penalty could not have acted on
them regardless of its weighting factor. The hierarchy violation rate
improvement observed in Table 1 is therefore plausibly concentrated on the
one edge where the penalty had any signal to act on, while violation on the
other 12 edges is left largely unaffected by training and instead reflects
whatever consistency the flat sigmoid outputs happen to exhibit. Extending
hierarchical training's measured benefit to the full 13-edge hierarchy would
require a corpus where more of those edges have co-annotated parent and
child findings, which the current seven-dataset pool does not provide.

The localization result in subsection 3.2 is a clear negative finding at
standard object-detection thresholds: zero average precision at intersection
over union 0.25 and 0.5 for every mapped finding, for both conditions. Image-level
discrimination and pixel-level localization are evidently not aligned for
either training condition on this backbone. Grad-CAM applied at the last
convolutional block of a DenseNet121 classifier appears insufficiently
precise for the pathologies evaluated here, at least with the percentile-based
thresholding used to extract a box from the activation map. Whether a
different localization method, a different convolutional block, or a
different backbone architecture would recover usable spatial signal is an
open question this corpus and this backbone alone do not answer; the
cardiomegaly ground-truth box provenance concern noted in subsection 3.2 also
means the true magnitude of the localization gap has not been independently
re-verified.

The uncertainty-gated coarse fallback mechanism, in which the model would
fall back from a child finding to its parent under high predictive
uncertainty measured by Monte Carlo Dropout, was designed as the intended
primary contribution of this line of work but has not been evaluated in the
results reported here; no organ segmentation conditioning was run either.
Whether the mechanism improves selective prediction quality over flat
abstention, the pre-registered primary hypothesis, remains open. Given how
few of the 13 hierarchy edges receive co-annotated training signal in the
current corpus, any future evaluation of the fallback mechanism should
account for the same data coverage limitation identified here: a fallback
gate can only meaningfully trade child-level for parent-level predictions on
edges where the parent finding itself carries a trained, calibrated signal.
