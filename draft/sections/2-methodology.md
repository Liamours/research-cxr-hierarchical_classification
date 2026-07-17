This section describes the label space and hierarchy, the pooled dataset and
preprocessing pipeline, the model and training procedure for the two conditions
actually trained, and the evaluation protocol. Subsection 2.1 covers the
dataset and preprocessing, subsection 2.2 the label hierarchy, subsection 2.3
the model and training objective, and subsection 2.4 the evaluation protocol.

All experiments use a single backbone, DenseNet121 pretrained on chest
radiographs via TorchXRayVision, and vary only the training loss between the
two conditions compared in this draft. Organ segmentation conditioning and
Monte Carlo Dropout uncertainty estimation were part of the original
experimental design but were not run for the results reported here; where a
designed component has not yet been evaluated, this is stated explicitly
rather than assumed.
