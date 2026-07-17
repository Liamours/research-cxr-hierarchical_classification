# B. Datasets and preprocessing

We combine seven publicly available chest radiograph collections: NIH
ChestX-ray14, CheXpert+ (2024 expert re-annotation of CheXpert), VinDr-CXR,
VinDr-PCXR (pediatric), TBX11K, COVIDx-CXR4, and PadChest. After ingestion and
frontal-view filtering, the combined dataset comprises 527,745 images. All
datasets are mapped to a shared 51-label canonical set grounded in Indonesian
clinical guidelines (PDPI, PNPK, KKI; src/data/label_space.py CANONICAL_LABELS).
A dataset that does not annotate a given finding contributes no label for that
finding (not-applicable, not negative), so the finding is excluded from both
training and scoring for that dataset's images. Of the 51 canonical findings,
27 currently receive training signal from at least one ingested dataset; the
remaining 24 are defined by clinical authority and stay masked pending
datasets that annotate them.

Each retained image is converted to grayscale and resized so the shorter side
is 256 pixels (bicubic interpolation), then center-cropped to 224 by 224
pixels and replicated to a 3-channel 8-bit PNG on disk. For DICOM sources,
pixel values are rescaled with the stored slope and intercept, inverted when
the photometric interpretation is MONOCHROME1, and clipped to the 0.5th and
99.5th percentiles before scaling (src/data/preprocess/common.py). At load
time for training, images are converted to a single grayscale channel and
normalized to the TorchXRayVision convention (pixel range [-1024, 1024]),
matching the pretraining regime of the backbone (Section D). Where a source
dataset encodes an explicit "uncertain" label category, it is mapped to
negative (U-Zero policy).

Train, validation, and test partitions are produced by a multi-label
iterative stratification (seed 42, 80/10/10 target proportions;
src/script/run_resplit_stratified.py), grouped at the patient or study level
where a group key is recoverable from the source metadata (NIH ChestX-ray14,
CheXpert+, COVIDx-CXR4, PadChest). For the remaining three datasets
(VinDr-CXR, VinDr-PCXR, TBX11K) no patient or study identifier is recoverable
from the ingested metadata, and the split is stratified at the image level.
The resulting partition sizes are 440,080 train, 43,969 validation, and 43,696
test images.

Training-time augmentation, applied only to the training split, is a small
affine transform (rotation up to 10 degrees, translation up to 5 percent,
scale within [0.95, 1.05]) and mild brightness and contrast jitter (factor
0.1). Horizontal flipping is disabled: it mirrors the cardiac silhouette and
aorta, which corresponds to the rare situs inversus anatomy.
