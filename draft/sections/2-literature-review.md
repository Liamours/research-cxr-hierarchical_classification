Each of the seven datasets pooled in this work originates from a paper that
reported its own classification performance on its own task. Those numbers
are described below on their own terms: each paper trained and evaluated on
a single dataset, chose its own label subset, and in one case reports a
different metric entirely. None of the figures below are placed next to our
own results as a controlled comparison. A single pooled model evaluated on
27 canonical findings of varying prevalence is not measured the same way as
a specialist model trained and tested on one dataset's own narrower label
set, and averaging over more and rarer findings does not produce a number
that means the same thing as averaging over fewer, better represented ones.
This section exists to document what has already been reported on these
datasets, not to benchmark against it.

Pham and colleagues trained a convolutional network on CheXpert that
exploits hierarchical dependencies between findings together with
uncertainty-labeled training examples, reaching a mean area under the
receiver operating characteristic curve (AUROC) of 0.930 on the CheXpert
competition test set, evaluated over the five findings that competition
uses for ranking.

Kufel and colleagues fine-tuned an EfficientNet model on NIH ChestX-ray14
across its full set of 14 labeled thoracic findings, reporting a mean AUROC
of 0.843.

Pham and colleagues, in a separate paper introducing the VinDr-CXR dataset
itself, trained an explainable deep learning system on 51,485 chest
radiographs with radiologist-drawn bounding box annotations to
simultaneously classify six common thoracic diseases and localize 14
findings, reporting a mean AUROC of 0.967 on the classification task. The
same paper also measured the system's effect on interobserver agreement
among radiologists, finding a small increase in agreement when radiologists
consulted the system's suggestions.

Chen and colleagues introduced a two-stage hierarchical label-uncertainty-
aware training objective and evaluated it primarily on the PLCO dataset,
where their finetuned model reaches a mean leaf-label AUROC of 0.887, described
by the authors as the highest yet reported for that dataset. The same
method, evaluated as a supporting experiment on PadChest with a different,
larger taxonomy, reaches a mean leaf-label AUROC of 0.837, a smaller
improvement over flat baselines than on PLCO but consistent in direction.

Shanthi and Mahalingam combined a capsule network with a squeeze and
excitation attention module and hyperparameter tuning by a crossover boosted
seagull optimization algorithm, training across four combined chest
radiograph datasets. For COVIDx CXR-4 specifically, the paper reports an
accuracy of 99 percent; no AUROC or other ranking metric is reported for
this dataset in the paper.

Esbergenov and Nurimov compared several lightweight convolutional
architectures for three-class tuberculosis triage on TBX11K, distinguishing
tuberculosis, sick but not tuberculosis, and healthy cases. Their best
configuration, a MobileNet-V3 backbone, reaches a macro AUROC of 0.9998.
This is a self-deposited report on Zenodo rather than a peer-reviewed
publication, and is described here with that distinction noted.
