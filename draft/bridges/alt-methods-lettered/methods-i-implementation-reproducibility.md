# I. Implementation and reproducibility

Every experiment is driven by a single YAML configuration file that fixes the
random seed to 42 and records the backbone, label structure, loss, hierarchy
edges, optimization, and evaluation settings; unknown configuration keys are
rejected rather than silently ignored. The configuration is snapshotted into
each run's output directory alongside per-epoch metrics, training and
evaluation events, and final checkpoints, so a run's exact settings are
recoverable from its output alone.

The current experimental design compares three loss-function conditions,
flat, hierarchical, and BAFL (Section E), under one fixed backbone
(Section D), one fixed dataset and split (Section B), and identical
optimization settings, so that any difference in outcome is attributable to
the loss function alone. Organ segmentation conditioning and Monte Carlo
Dropout with uncertainty-gated fallback (Sections F and G) are implemented as
independent, optional factors in the same configuration schema and were
explored in an earlier, superseded experimental grid; they are not part of
the current three-condition comparison. Code and configurations are released
to support replication.
