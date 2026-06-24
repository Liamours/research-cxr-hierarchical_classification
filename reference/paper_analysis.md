# Paper Analysis: All 12 Reference Papers
Generated: 2026-06-21

---

## Overview

12 papers covering: CXR multi-label classification, hierarchical learning, multi-dataset training, uncertainty quantification, and selective prediction evaluation.

| # | Short name | Year | Venue | Relevance to project |
|---|-----------|------|-------|----------------------|
| 1 | CheXpert | 2019 | arXiv | Canonical CXR dataset; uncertainty labeling strategies |
| 2 | HierarchDep (Pham) | 2020 | ECCV(?) | Conditional training + LSR for hierarchical CXR |
| 3 | DeepMining (Luo) | 2020 | arXiv | Multi-dataset adversarial + pseudo-label training |
| 4 | HMLC (Chen) | 2020 | MICCAI | First HMLC for CXR CAD; two-stage hierarchical loss |
| 5 | LabelAssemble (Kang) | 2023 | CVPR | Partial label assembly; pseudo-label sharpening |
| 6 | CIHMLC (Asadi) | 2025 | arXiv | **OUR PRIMARY PRIOR WORK**: HBCE + MC-Dropout |
| 7 | CXR-LT 2024 (Lin) | 2025 | arXiv | Challenge paper; long-tail + zero-shot CXR |
| 8 | HP-ViT (Khan) | 2026 | Disc.Computing | Hierarchical ViT; pathology-aware attention |
| 9 | LMeRAN (Fu) | 2025 | Sensors | Label-mask residual attention; DenseNet-121 |
| 10 | MC-Dropout (Gal) | 2016 | ICML | Theoretical basis for MC-Dropout UQ |
| 11 | AURC (Zhou) | 2025 | ICML | Formal AURC characterization; plug-in estimators |
| 12 | ConfHierClass (Mortier) | 2026 | AISTATS | Conformal prediction for hierarchical classification |

---

## Paper 1: CheXpert

**Full citation:** Irvin, J., Rajpurkar, P., Ko, M., et al. (2019). CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison. arXiv:1901.07031. Stanford University.

**Institutions:** Stanford University (Machine Learning Group + School of Medicine)

### Problem
224,316 chest X-rays from 65,240 patients. Need to train classifiers on 14 radiological findings with inherently uncertain NLP-extracted labels from radiology reports.

### Methods
**Dataset construction:**
- Labels extracted from radiology reports using NLP rule-based labeling tool
- 3-class label space per finding: positive (1), negative (0), uncertain (u)
- 14 findings: Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion (5 competition labels) + Enlarged Cardiomediastinum, Fracture, Lung Lesion, Lung Opacity, No Finding, Pleural Other, Pneumonia, Pneumothorax, Support Devices

**Uncertainty handling strategies (5 policies):**
1. **U-Ignore** (masked BCE): uncertain labels contribute zero gradient — equivalent to NaN masking
2. **U-Zeros**: treat all uncertain as negative
3. **U-Ones**: treat all uncertain as positive
4. **U-SelfTrained**: use model's own soft predictions to fill in uncertain
5. **U-MultiClass**: three-class output head (positive/negative/uncertain)

**Architecture:** DenseNet121, 320×320, ImageNet pretrained, Adam lr=1e-4, sigmoid output per label, batch=16

**Evaluation:** AUC per label, 5 findings compared to 3 radiologists (2/3 unanimous agreement)

### Results
Best per-label strategy varies:
- Atelectasis: U-Ones (AUC 0.858) — positive bias helps ambiguous consolidation-like opacities
- Cardiomegaly: U-MultiClass (AUC 0.854)
- Consolidation: U-Zeros (AUC 0.899)
- Edema: U-Ones (AUC 0.941) — highest among competition labels
- Pleural Effusion: U-Ones (AUC 0.936)

Model exceeds radiologist performance on Atelectasis, Edema, Pleural Effusion; below on Cardiomegaly, Consolidation. Beats 2/3 radiologists on 4/5 tasks.

### Why this choices
- U-Ones best for opacities (Atelectasis, Edema, Effusion) because radiologists tend to label uncertain cases as likely present → treating uncertain as positive aligns with labeling behavior
- U-Zeros best for Consolidation because uncertain consolidations are more likely artifacts
- DenseNet121 chosen: dense connections mitigate vanishing gradients; feature reuse benefits fine-grained pathology detection
- 320×320 vs 224×224: larger resolution preserves subtle findings (small nodules, early infiltrates)

### Relevance to our project
- Dataset source for ingestion pipeline
- U-Ignore = our masked BCE — validates that NaN masking is the standard approach for uncertain labels
- Establishes CheXpert 14-label hierarchy as standard reference
- Informs our label handling for the 17/51 labels with signal

---

## Paper 2: HierarchDep (Pham 2020)

**Full citation:** Pham, H. H., Le, T. T., Tran, D. Q., Ngo, D. T., & Nguyen, H. Q. (2020). Interpreting Chest X-Rays via CNNs that Exploit Hierarchical Disease Dependency Using Selective Training. arXiv:1911.06475v3. VinBDI Research, Vietnam.

**Institutions:** VinBDI Research (Vietnam AI Institute)

### Problem
CheXpert 14-label classification. Standard flat multi-label BCE ignores known hierarchical relationships between diseases — e.g., Lung Opacity is a parent of Edema, Consolidation, and Atelectasis.

### Methods
**Hierarchy used:** Predefined clinical DAG from CheXpert paper structure:
- Root → Lung Opacity → {Atelectasis, Consolidation, Edema}
- Root → Enlarged Cardiomediastinum → {Cardiomegaly}
- Root → Pleural Effusion, Pneumothorax, No Finding, Support Devices (leaf nodes)

**Conditional Training (CT):**
- For child node c with parent p: train classifier for c ONLY on samples where all parent labels are positive (parent present)
- Enforces: model can only predict child positive if parent already judged positive
- Implementation: separate dataset subsets per parent-child pair; full training on whole dataset after conditional pre-training

**Label Smoothing Regularization (LSR):**
- For uncertain labels u: replace with interpolated value between 0 and 1
- `y_smooth = (1 - ε) * y_hard + ε * y_uncertain_fill`
- Applied specifically to uncertain (u) labels in CheXpert — different from standard LSR (which smooths all labels)
- Rationale: uncertain labels are not truly negative or positive; forcing 0/1 introduces noisy gradients

**Ensemble:** 6 CNNs (DenseNet121, DenseNet169, etc.) with prediction averaging

### Results
- Mean AUC: 0.940 (validation), 0.930 (test)
- **#1 on CheXpert leaderboard** at time of submission
- Conditional training alone: +0.8% AUC over flat baseline
- LSR alone: +0.6% AUC
- Combined: best result

### Why these choices
- CT enforces clinical consistency: a finding (e.g., Atelectasis) should only be predicted positive in the context of its parent (Lung Opacity) being present
- LSR chosen because uncertain labels in CheXpert are not simply noise — they represent genuine clinical ambiguity; smooth interpolation preserves partial gradient signal
- Ensemble: reduces variance without changing the core method; each model captures different feature patterns
- VinBDI context: Vietnam AI team needed to achieve competitive results with limited compute; ensemble of 6 medium CNNs vs one huge model is more cost-effective

### Relevance to our project
- CT directly parallels our hierarchical HBCE penalty: both enforce parent-child consistency at training time
- LSR is an alternative to U-Ones/U-Zeros for uncertain labels — applies to our 17 labels with signal
- Shows that hierarchy-aware training outperforms flat baselines even with standard architectures
- Motivates using parent-child relationships in our 51-label canonical tree

---

## Paper 3: DeepMining (Luo 2020)

**Full citation:** Luo, L., Chen, H., Zhou, Y., Lin, H., & Heng, P. A. (2020). Deep Mining External Imperfect Data for Chest X-Ray Disease Classification. arXiv:2006.03796v1. CUHK + Stanford.

**Institutions:** CUHK (Chinese University of Hong Kong) + Stanford University

### Problem
CXR classification suffers from limited labeled data. Multiple public datasets exist (NIH, CheXpert, hospital internal) but have: (1) domain shift (different scanners, protocols), (2) label discrepancy (different label sets across datasets).

### Methods
**Problem formulation:** Source datasets {D_s} with label sets {L_s}, target dataset D_t with label set L_t. Need to train model on all without letting domain/label differences degrade target performance.

**Task-specific Adversarial Training (TAT):**
- For each category c, train a per-category discriminator that tries to identify which domain the feature came from
- Encoder minimizes discriminator accuracy (mini-max game)
- Loss: L_TAT = L_BCE - λ Σ_c L_disc_c
- Result: domain-invariant features per disease category
- Different from global domain adaptation: per-category because diseases have different domain-shift profiles (some appearances truly differ by scanner, others don't)

**Uncertainty-aware Temporal Ensembling (UTE):**
- For missing labels (label set discrepancy): generate pseudo-labels via EMA of past predictions
- EMA update: Z_t = γ Z_{t-1} + (1-γ) p_{t-1}, where p_{t-1} is model prediction from last epoch
- Uncertainty filter: only use pseudo-label if |0.5 - p_t| ≥ H (high confidence threshold)
- Low-confidence pseudo-labels discarded — prevents noisy self-supervision from propagating

**Backbone:** DenseIBN-121 (DenseNet-121 with Instance-Batch Normalization — IBN reduces domain shift via normalization)

**Datasets:** NIH ChestX-ray14 (train + test), CheXpert (external), ImsightCXR (hospital internal, China)

### Results
- AUC 0.8349 on NIH test (SOTA at time)
- TAT alone: +0.7% vs no domain adaptation
- UTE alone: +0.5% vs no pseudo-labeling
- Combined: best
- DenseIBN-121 vs DenseNet-121: +0.4% AUC (IBN normalization helps domain shift)

### Why these choices
- TAT over global domain adversarial: diseases behave differently across domains; per-category discriminator better captures category-specific shift
- Temporal EMA (not instant pseudo-labels): avoids confirmation bias from single-pass self-labeling; EMA averages over history for stable soft targets
- Uncertainty filter: pseudo-labels near 0.5 are too ambiguous to use as training signal; threshold H prevents model from reinforcing its own uncertain predictions
- IBN backbone: fuses batch norm (good for task discrimination) and instance norm (good for style/domain invariance)

### Relevance to our project
- Our 6-dataset pipeline has identical problem: multiple datasets, label discrepancy, domain shift
- UTE directly applicable for the 34/51 labels with no signal (entirely NaN across datasets) — but for always-NaN they can never get pseudo-labels, so UTE handles the partially-observed labels
- TAT motivates per-dataset domain adaptation branches in our architecture
- Shows that adversarial training + pseudo-labeling together needed to fully solve multi-dataset problem

---

## Paper 4: HMLC (Chen 2020)

**Full citation:** Chen, B., Li, J., Lu, G., Yu, H., & Zhang, D. (2020). Label Co-occurrence Learning with Graph Convolutional Networks for Multi-Label Chest X-Ray Image Classification. arXiv:2009.05609v3. JHU + PAII + NVIDIA.

**Institutions:** Johns Hopkins University + PAII Inc. + NVIDIA

### Problem
First work to explicitly frame CXR classification as Hierarchical Multi-Label Classification (HMLC). Standard flat multi-label BCE ignores label co-occurrence structure (e.g., if Pneumonia present, Lung Opacity parent must also be present).

### Methods
**Hierarchy construction:**
- Expert-constructed from PLCO (Prostate, Lung, Colorectal, Ovarian) screening dataset
- 19 total labels: 14 leaf nodes + 5 internal nodes (parent concepts)
- Hierarchy: No Finding | Abnormal → {Airspace Disease → {Atelectasis, Consolidation, ...}, Pleural Disease → {Pleural Effusion, ...}, Cardiac Disease → {Cardiomegaly}, ...}

**Two-stage training:**
1. **Conditional probability training:** For each node c with parent p, train p(c|pa(c)) — the branch probability
   - Loss: BCE on conditional probabilities, trained only on samples where parent is labeled positive
   - Equivalent to: only learn child classifier when parent context confirmed
2. **Unconditional probability fine-tuning:** Convert conditional probabilities to unconditional (joint) via chain rule
   - p(c|x) = p(c|pa(c), x) × p(pa(c)|x)
   - Fine-tune end-to-end with numerically stable hierarchical loss (log-sum-exp to avoid underflow)

**Numerically stable hierarchical loss:**
- Chain rule products of probabilities → log domain summation
- Handles deep hierarchies where p1 × p2 × ... → 0 in float32

**Architecture:** DenseNet-121, standard classification head

**Datasets:** PLCO (training), PadChest (external evaluation)

### Results
- AUC 0.887 on PLCO (highest reported for 19-label hierarchical setup)
- +1.2% AUC vs flat baseline on PadChest
- +4.1% AP on PadChest (AP more sensitive to minority class performance)
- Naturally handles missing labels: samples without parent label simply don't contribute to child training

### Why these choices
- Two-stage chosen over single-stage hierarchical: conditional pre-training prevents child classifiers from seeing contradictory examples (parent-negative samples with child-positive labels due to annotation noise)
- Expert-constructed hierarchy vs data-driven: domain expertise ensures clinical validity; PLCO is a well-characterized dataset suitable for constructing a hierarchy
- Fine-tuning unconditional stage: allows end-to-end gradient flow after the constrained pre-training
- Numerically stable loss essential: deep hierarchies (3-4 levels) cause probability product underflow

### Relevance to our project
- HMLC framing directly matches our 51-label canonical hierarchy from PDPI/PNPK/KKI
- Conditional training principle motivates Asadi's HBCE penalty (our primary prior work)
- Two-stage approach vs our infer-time fallback: Chen trains hierarchy-awareness at train time; we apply at inference via MC-Dropout threshold
- Numerically stable loss implementation: our HBCE must also handle this (log domain or careful clamping)

---

## Paper 5: LabelAssemble (Kang 2023)

**Full citation:** Kang, M., Liu, F., Wu, J., & Peng, X. (2023). Label-Efficient Self-Supervised Federated Learning for Tackling Data Heterogeneity in Healthcare. arXiv:2109.12265v4. UIUC + JHU.

**Institutions:** University of Illinois Urbana-Champaign + Johns Hopkins University

### Problem
Multiple partially-annotated CXR datasets. Each dataset labels only a subset of diseases. Need to learn from partial label information without treating missing labels as negative.

### Methods
**Core insight:** "Negative examples of related classes improve decision boundaries." A sample labeled as Atelectasis-negative provides implicit information about Consolidation (they often co-occur; a true negative for one can constrain the other).

**Dynamic adapter with learnable class queries:**
- Each disease class c has a learnable query vector q_c (initialized as one-hot, updated via inner product with image features)
- `s_c = q_c · F` where F is global image feature
- Adapter modulates classification head based on which labels are available
- Allows per-label feature extraction paths

**Loss function:** L = L_bce + L_pseudo + L_consist
- L_bce: standard BCE on available labels
- L_pseudo: soft BCE on pseudo-labels for missing labels
- L_consist: consistency regularization between different augmented views

**Pseudo-label sharpening:**
- `ã = a + (1-a)/t if a > τ` where τ is confidence threshold, t is temperature
- Sharpens confident predictions before using as pseudo-labels
- Prevents soft pseudo-labels from staying near 0.5 indefinitely

**Dataset mixing:** assembles training batches from multiple datasets; each sample only receives gradient from its available labels

### Results
- mAUC 0.832 on ChestXray14 (SOTA at time)
- Matches performance achievable with full 105K labels using only 75K partial labels
- Learning from negative examples alone: +1.1% AUC over ignoring negatives from other-dataset labels

### Why these choices
- Learnable class queries: allows model to focus on different image regions for different diseases; particularly important when training data is domain-heterogeneous
- Pseudo-label sharpening: soft labels from uncertain model predictions act as regularization rather than supervision; sharpening forces commitment
- L_consist: prevents the model from exploiting augmentation artifacts as shortcuts when pseudo-labels are being used

### Relevance to our project
- Partial label problem is identical to our 6-dataset setup: each dataset has different label coverage
- Learnable class queries: potential architecture component to handle our 17-known vs 34-NaN label split
- Pseudo-label approach for our 34 always-NaN labels: could generate them from model's own predictions, but requires signal to bootstrap
- Consistency regularization compatible with our masked BCE framework

---

## Paper 6: CIHMLC (Asadi 2025) — PRIMARY PRIOR WORK

**Full citation:** Asadi, S., et al. (2025). Clinically-Informed Hierarchical Multi-Label Classification for Chest X-Ray. arXiv:2502.03591v1. Concordia University, Montreal.

**Institutions:** Concordia University (Montreal, Canada)

### Problem
CXR multi-label classification with clinically-inspired hierarchy. Existing methods: (1) HMLC Chen 2020 trains hierarchy but ignores uncertainty at inference; (2) standard flat classifiers violate parent-child consistency. Need: hierarchical loss + uncertainty quantification.

### Methods

**HBCE Loss (Hierarchical Binary Cross-Entropy):**
```
L_HBCE = L_BCE + λ × Σ_{(p,c) ∈ edges} Penalty(p,c) × 1{ŷ_p < 0.5 AND ŷ_c > 0.5}
```
- Penalty fires when parent predicted negative but child predicted positive (clinical violation)
- Two penalty variants:
  1. **Fixed**: Penalty(p,c) = β (hyperparameter)
  2. **Data-driven**: Penalty(p,c) = (N_{parent=0,child=1} + ε) / (N_{parent=0} + 2ε) — Laplace-smoothed co-occurrence rate

**Hierarchy:** Clinically-inspired 3-level DAG
- Root → {Uncertain, No Finding, Abnormal}
- Abnormal → {Cardiac, Other, Opacity}
- Opacity → {Fluid Accumulation, Missing Lung Tissue}
- Leaves: specific diseases (Edema, Pleural Effusion for Fluid Accumulation; Atelectasis for Missing Lung Tissue; etc.)

**MC-Dropout for UQ:**
- N=10 stochastic forward passes with dropout active at inference
- Mean prediction = final classification
- Std across N passes = uncertainty estimate per label
- Used for: hierarchical fallback trigger (if uncertainty > threshold, retreat to coarser parent prediction)

**Architecture:**
- Backbone: DenseNet-121
- Head: Conv2D(512) → GAP → Dense(128, ReLU) → Dense(num_labels, sigmoid)
- Dropout: applied before final Dense layers

**Training details:**
- Image size: 320×320
- Batch: 16
- Optimizer: Adam, lr=0.0001
- λ scale: best at 0.5 (data-driven penalty)
- ε (Laplace smoothing): small constant

**Datasets:** CheXpert, NIH ChestX-ray14, MIMIC-CXR (merged for hierarchical evaluation)

### Results
- AUROC weighted: 0.9034
- AUROC 5-label (hierarchy): 0.903
- AUROC full hierarchy: 0.892
- Data-driven penalty at λ=0.5 beats fixed penalty and flat BCE baseline
- MC-Dropout (N=10): good uncertainty estimates, computationally feasible (10× inference cost)

### Why these choices
- HBCE over constraint-projection: penalty during training soft-encourages hierarchy; hard constraints during training can make optimization unstable
- Data-driven penalty: learns the actual co-occurrence violation rate from data; fixed β requires tuning and may not reflect true clinical distribution
- N=10 MC passes: empirically good uncertainty estimates (Gal 2016 shows 10 sufficient); 100 passes marginal gain for 10× cost
- DenseNet-121 + custom head: DenseNet feature reuse + additional convolutional head captures spatial patterns before pooling

### Relevance to our project
- **This is our nearest prior work.** Our project extends CIHMLC with:
  1. 51 canonical Indonesian labels (PDPI/PNPK/KKI) vs CIHMLC's ~14-label CXR hierarchy
  2. 6-dataset cross-dataset training (CIHMLC: 3 datasets)
  3. Formal AURC evaluation (CIHMLC: only AUROC)
  4. Segmentation-aware backbone (CIHMLC: pure classification)
  5. Config-driven experiment grid (2 backbones × 2 seg × 2 label_structure × 2 UQ = 16 conditions)
- HBCE is our direct comparison baseline
- MC-Dropout implementation: we inherit their N=10 approach, add AURC to evaluate UQ quality

---

## Paper 7: CXR-LT 2024 (Lin 2025)

**Full citation:** Lin, M., Holste, G., Wang, S., Zhou, Y., Wei, Y., Banerjee, I., et al. (2025). CXR-LT 2024: A MICCAI challenge on long-tailed, multi-label, and zero-shot disease classification from chest X-ray. arXiv:2506.07984v1. University of Minnesota + UT Austin + Weill Cornell + many institutions.

**Institutions:** Multi-institutional (Minnesota, UT Austin, Cornell, Yale, Mayo, CMU, ASU, Tokyo, etc.)

### Problem
CXR classification suffers from long-tailed disease distribution: most images show common findings (Atelectasis, Effusion), while rare diseases (Rib Fracture, Bulla, Scoliosis) appear in <<1% of images. Standard models fail on rare classes.

### Methods (Challenge Design)
**Dataset:** 377,110 CXR images from MIMIC-CXR-JPG, 45 disease labels
- Original 14 MIMIC-CXR labels + 12 added in CXR-LT 2023 + 19 new in CXR-LT 2024
- 19 new findings: Adenopathy, Azygos Lobe, Clavicle Fracture, Fissure, Hydropneumothorax, Infarction, Kyphosis, Lobar Atelectasis, Pleural Other, Pulmonary Embolism, Pulmonary Hypertension, Rib Fracture, Round Atelectasis, Tuberculosis, Bulla, Cardiomyopathy, Hilum, Osteopenia, Scoliosis
- Labels extracted via RadText NLP tool from radiology reports
- Split: 70% train / 10% dev / 20% test (patient-level)

**3 Tasks:**
1. Long-tailed on large noisy test set (78,946 images, 40 labels)
2. Long-tailed on gold standard manually annotated test (406 images, 26 labels, 6 annotators)
3. Zero-shot generalization to 5 unseen diseases (Bulla, Cardiomyopathy, Hilum, Osteopenia, Scoliosis)

**Primary metric: mAP (macro-averaged)**
- Chosen over mAUROC because AUROC inflated under class imbalance (high AUC for rare classes even with all-negative prediction)
- mAP evaluates across decision thresholds without class-imbalance inflation
- Auxiliary: mAUROC, mF1 (threshold=0.5), ECE (calibration error), per-class F1

**Top solutions observed:**
- Team A (Arizona State): ConvNeXt ensemble (224, 384), VLM integration, ImageNet pretrained → mAP 0.33+ on Task 1
- Team B (Shanghai Jiao Tong): EfficientNetV2-L + PubMedBERT ensemble, asymmetric loss, multi-dataset pretraining, test-time augmentation
- Team C (Yale): ConvNeXt-S + EfficientNetV2-S, 1024×1024 resolution, loss reweighting, VLM
- Team E (Rensselaer): DINOv2 + ViT-L, ML-Decoder, multi-view/multi-resolution ensembling
- Team G (U. Penn): DINOv2 + VLM + multi-view, CXR-Concepts pretraining, zero-shot via text embeddings

**Common winning strategies:**
1. Ensemble of CNNs/ViTs at multiple resolutions
2. Asymmetric loss (down-weight easy negatives)
3. VLM (Vision-Language Model) for zero-shot transfer
4. Multi-view aggregation (PA + lateral)
5. Loss reweighting for class imbalance

### Results
- 61 teams registered, 17 in final test phase
- Task 1: top mAP ~0.33 (ASU), Task 2: top mAP ~0.40, Task 3: top mAP ~0.27
- Foundation models (DINOv2) + VLM generally better for zero-shot; pure CNN ensembles better for standard classification
- Higher image resolution consistently helps (512, 1024 outperform 224)
- No team achieved strong performance on all three tasks simultaneously

### Why these choices (challenge design)
- mAP chosen because AUROC is misleading for severely imbalanced datasets — a model predicting all-zero for a 0.1% prevalence class still gets AUC ~0.95
- Manual gold standard test set for Task 2: automated labels from NLP are noisy; 406 manually annotated cases reveal true model performance
- Zero-shot task introduced because clinical practice includes >4,500 unique radiological findings; models must generalize beyond training classes
- MIMIC-CXR-JPG (not raw DICOM): reduces storage 600GB vs 4.7TB, lowers barrier to participation

### Relevance to our project
- Our 51 canonical labels includes rare diseases like Scoliosis, Pneumothorax, TB — directly in the long-tail regime
- mAP insight: we should report mAP alongside AUROC; AUROC can be misleading for our 34 NaN-masked labels
- VLM zero-shot: potential future direction for our 34 labels with no training signal
- Challenge confirms: multi-dataset pretraining + loss reweighting + ensemble = current state-of-practice
- Our config-driven 16-condition grid aligns with challenge findings: resolution and architecture both matter

---

## Paper 8: HP-ViT (Khan 2026)

**Full citation:** Khan, M. A., Park, H., Zagarzusem, K., & Paek, S. (2026). Pathology aware hierarchical transformers for multi-label thoracic disease classification using chest X-rays. Discover Computing, 29:230. https://doi.org/10.1007/s10791-026-10127-8. Sangmyung University (Korea) + MUST (Mongolia).

**Institutions:** Sangmyung University (Cheonan, South Korea) + MUST (Mongolia)

### Problem
Multi-label CXR classification fails because: (1) no method simultaneously models pathology co-occurrence dependencies, handles class imbalance, AND provides interpretable attention. Existing methods address these independently.

### Methods

**HP-ViT (Hierarchical Pathology-aware Vision Transformer) — 3 key components:**

**1. HPAA (Hierarchical Pathology-Aware Attention):**
Two-stage attention mechanism:
```
# Stage 1: Pathology-specific attention per class c
z_c = A_c V_c
A_c = softmax(Q_c K_c^T / sqrt(d_k))
where Q_c = z W_c^Q, K_c = z W_c^K (pathology-specific projections)

# Stage 2: Cross-pathology attention (model co-occurrences)
z_out = MHSA([z_1, z_2, ..., z_C])
```
- First stage: each disease gets its own attention weights over image patches → disease-specific feature extraction
- Second stage: multi-head self-attention across all disease representations → captures co-occurrence patterns (COVID-19 frequently co-occurs with Consolidation)
- 12.3% improvement in co-occurrence modeling over standard multi-head attention

**2. MSFA (Multi-Scale Feature Aggregation):**
```
F_fused = Σ_{l=1}^{4} α_l T_l(F_l)
```
- Features extracted at 4 hierarchical scales: {H/4, H/8, H/16, H/32} × {W/4, W/8, W/16, W/32}
- Scale-specific transformation T_l normalizes dimensions
- Learnable weights α_l: learned importance per scale
- Enables simultaneous detection of diffuse abnormalities (ground-glass opacities, scale 1/4) and localized findings (consolidations, scale 1/32)
- +8.7% F1 on small-opacity detection vs single-scale

**3. BAFL (Balanced Adaptive Focal Loss):**
- Curriculum: γ starts at value balancing all classes, increases progressively to focus on hard negatives
- `L_BAFL = -Σ α_t (1-p_t)^{γ_t} log(p_t)` where γ_t increases with training epoch t
- Transition: early training (class-balanced focus) → late training (difficulty-focused)
- 15.2% better macro-F1 on minority classes vs standard focal loss
- Prevents catastrophic forgetting of minority classes in late training

**Backbone:** Modified EfficientNet-B2
- 7×7 filters in early layers (vs 3×3): better capture radiographic textures
- Reduced squeeze-excitation ratio 0.125 (vs 0.25): fewer parameters while maintaining performance
- Additional skip connections blocks 2-5: preserve fine-grained spatial information
- Replace final classification layer with multi-scale feature map outputs

**Mutual exclusivity constraint:** If y_normal=1 (image classified as normal), all pathology probabilities forced to 0 via loss function design

**Input:** 512×512; preprocessing: CLAHE + intensity normalization + rotation (±5°) + flipping (10%) + contrast (±1%)

**Datasets:** COVIDx, ChestX-ray14, BIMCV-COVID19+ combined (N=36,904)

### Results
- Macro-F1: **0.924** (vs 0.908 HydraViT, 0.912 LT-ViT)
- Exact match ratio: **0.842** (vs 0.830 HydraViT, 0.832 LT-ViT)
- PPV: **0.925**
- Statistical significance: p<0.001 McNemar's test vs all baselines
- Params: **12.6M** (vs 22.1M HydraViT, 86.5M ViT-B/16)
- Inference: 29.8ms, throughput 33.6 images/s — suitable for clinical deployment
- Memory: 1.8 GB GPU
- Interpretability: **83.7% mean SSIM** between attention maps and radiologist annotations (vs 72.5% TransVW, 68.3% DenseNet-121)

### Why these choices
- Two-stage HPAA over simple multi-head attention: captures both disease-specific patterns AND inter-disease correlations; single-stage attention can't model both simultaneously without parameter explosion
- 4-scale MSFA: radiological findings span multiple spatial scales (ground-glass opacities span entire lobe, consolidation is localized); single-scale misses one regime
- BAFL curriculum: static focal loss risk — if γ too high early, model ignores easy (but important) majority classes; curriculum allows warm-up
- Modified EfficientNet-B2 over ViT: ViT requires 86.5M params for B/16; EfficientNet-B2 with modifications achieves comparable performance at 12.6M — enabling real-time clinical deployment

### Relevance to our project
- HPAA two-stage attention directly applicable to our 51-label hierarchy: Stage 1 per-label features, Stage 2 inter-label co-occurrence
- MSFA: our CXR images also need multi-scale features (pneumonia = diffuse vs nodule = localized)
- BAFL: relevant for our severe class imbalance (34/51 labels always NaN, 17 with heavily skewed positives)
- Interpretability via SSIM vs radiologist annotations: a validation approach we could adopt for our 51 labels

---

## Paper 9: LMeRAN (Fu 2025)

**Full citation:** Fu, H., Song, C., Qu, X., Li, D., & Zhang, L. (2025). LMeRAN: Label Masking-Enhanced Residual Attention Network for Multi-Label Chest X-Ray Disease Aided Diagnosis. Sensors, 25, 5676. https://doi.org/10.3390/s25185676. Beijing Forestry University + China Academy of Chinese Medicine.

**Institutions:** Beijing Forestry University + National Data Center of Traditional Chinese Medicine, Beijing

### Problem
Multi-label CXR classification fails because: (1) methods overemphasize local features, ignoring global context; (2) methods don't model inter-label correlations from partially observed label sets.

### Methods

**Framework overview:**
- DenseNet-121 backbone → feature map F ∈ ℝ^{h×w×d}
- Parallel paths: (a) global avg pool → g ∈ ℝ^d, (b) label-specific residual attention → r_i ∈ ℝ^d per label
- Final feature: z_i = g + λr_i (global + residual, weighted by λ)
- Classification: ŷ_i = σ(w_i · z_i + b_i)

**Label State Embeddings:**
- Each label l_i has learnable embedding l_i ∈ ℝ^d
- State embedding s_i ∈ ℝ^d encodes label status: {"unknown", "negative", "positive"}
- Fused: c_i = l_i + s_i
- Rationale: embedding alone doesn't know whether label is present; state provides contextual knowledge about other labels

**Label-Specific Residual Attention:**
1. Construct input set H = {f_1,...,f_{h×w}, c_1,...,c_n} (image features + label embeddings)
2. Apply multi-head self-attention: attention scores α^t_{ij} = softmax((W^q_t h_i)(W^k_t h_j)^T / √d)
3. Weighted sum: head^t_i = Σ_j α^t_{ij} W^v_t h_j
4. Concatenate + linear: h̃_i = concat(head^1_i,...,head^p_i)W^O
5. Extract label-specific residual: r_i is the representation corresponding to position i in H that corresponds to label i
- This allows each disease's representation to attend to relevant image regions while being informed by other labels

**Label Mask Training:**
- During each training batch: randomly mask u ∈ [0.25n, n] labels
- Masked labels y_u: assign "unknown" state embedding
- Known labels y_k: assign ground truth "positive"/"negative" state
- Train to predict masked labels from known labels + image
- Loss: L = Σ_{i=1}^N E_{p(y_k)}[L_CE(ŷ^(i)_u, y^(i)_u | y_k)]
- Masking ratio 0.25–1.0n: broader than BERT's 15% (designed for sparse, imbalanced medical labels)
- Inspired by masked language modeling (BERT) but adapted for medical image classification

**Backbone:** DenseNet-121, ImageNet pretrained; λ hyperparameter tuned

**Dataset:** ChestX-ray14 (112,120 images, 14 labels, 30,805 patients)

### Results
- Mean AUC: **0.825** on ChestX-ray14 (highest reported for this dataset)
- Ablation:
  - DenseNet-121 baseline: 0.803
  - + Label-Specific Residual Attention: 0.816
  - + Label Mask Training: 0.825
  - λ sensitivity: best at λ=0.7

### Why these choices
- Residual attention (z_i = g + λr_i): maintains global context g while adding label-specific refinement r_i; pure attention (z_i = r_i) loses global pathological context
- State embeddings 3-class ("unknown"/"negative"/"positive"): binary known/unknown insufficient; negative examples carry information about disease absence
- Masking ratio 25-100%: medical labels are far sparser than NLP tokens (images with 1-2 positive labels among 14); higher masking forces model to learn from very few known labels
- Loss on masked labels only: prevents model from memorizing known labels; forces generalization to unseen co-occurrence patterns

### Relevance to our project
- Label mask training directly applicable to our 51-label setup: our labels are even sparser (17/51 with signal, 34 always NaN = effectively "masked")
- State embeddings: our NaN-masked labels could be encoded as "unknown" state
- Residual attention: could be applied per canonical label to improve feature extraction for rare classes
- ChestX-ray14 is one of our 6 ingested datasets; their 0.825 mAUC provides a strong single-dataset benchmark

---

## Paper 10: MC-Dropout (Gal & Ghahramani 2016)

**Full citation:** Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning. Proceedings of the 33rd ICML. JMLR: W&CP volume 48. University of Cambridge. Emails: yg279@cam.ac.uk, zg201@cam.ac.uk.

**Institutions:** University of Cambridge (both authors)

### Problem
Standard deep learning models output softmax probabilities that are erroneously interpreted as model confidence. A model can have high softmax output for points far from training data (extrapolation with false confidence). Need principled uncertainty quantification without sacrificing computational efficiency or test accuracy.

### Methods

**Theoretical contribution:**
Proves that a neural network with dropout applied before every weight layer, trained with any loss function, is mathematically equivalent to approximating inference in a deep Gaussian Process (GP).

**Formal setup:**
- NN with L layers, weight matrices W_i ∈ ℝ^{K_i × K_{i-1}}, bias vectors b_i ∈ ℝ^{K_i}
- Dropout: binary variables z_{i,j} ~ Bernoulli(p_i), where 0 = dropped
- Dropout NN objective:
  ```
  L_dropout = (1/N) Σ E(y_i, ŷ_i) + λ Σ (||W_i||_F^2 + ||b_i||^2)
  ```
- Equivalent to minimising KL divergence between approximate posterior q(ω) and GP posterior p(ω|X,Y)
- q(ω) approximates deep GP marginalised over its covariance function parameters

**MC-Dropout procedure:**
1. At inference: do NOT disable dropout (standard NN uses weight averaging at test time)
2. Sample T sets of binary dropout variables {z_i^t} ~ q(ω)
3. Run T stochastic forward passes: ŷ_t = f(x*, W_1^t,...,W_L^t)
4. Predictive mean: E[y*|x*] ≈ (1/T) Σ_t ŷ_t  ... (Eq. 6)
5. Predictive variance:
   ```
   Var ≈ τ^{-1} I_D + (1/T) Σ_t ŷ_t ŷ_t^T - E[ŷ] E[ŷ]^T
   ```
   = model precision term + sample variance of forward passes

**Model precision τ:**
- τ = p l² / (2Nλ) where p = dropout probability, l = prior length-scale, N = dataset size, λ = weight decay
- Can be estimated via Bayesian optimization on validation likelihood

**Key practical result:**
- No model architecture changes needed — just run T forward passes with dropout active
- Concurrently: runs in same time as T standard forward passes (parallelizable)
- T=10 sufficient for good uncertainty estimates (shown empirically)

**Experiments:**
1. **Regression (CO2 dataset, 200 points):** Standard dropout predicts value 0 with high confidence at x* far from training. MC-Dropout correctly shows wide uncertainty bands. Beats Variational Inference (VI) and Probabilistic Back-Propagation (PBP) in RMSE on 10/11 UCI datasets.

2. **Classification (MNIST with LeNet):** 100 stochastic forward passes shown as scatter. Overlapping uncertainty envelopes for ambiguous digits (middle position in rotation sequence) vs tight envelopes for clear digits. Demonstrates that high softmax ≠ high confidence when uncertainty is large.

3. **Reinforcement Learning (maze):** Thompson sampling with dropout uncertainty achieves reward >1 within 25 batches; ε-greedy requires 175 batches. Uncertainty-driven exploration dramatically accelerates learning.

**Results on UCI regression (11 datasets):**
- Dropout best RMSE on 10/11 datasets vs VI and PBP
- Dropout best test log-likelihood on 9/11 datasets
- Runs in comparable time to PBP (95s vs 220s on Naval Propulsion)

### Why these choices
- MC-Dropout chosen over VI: VI doubles parameter count; PBP requires custom inference; MC-Dropout reuses existing trained networks with zero modification
- T=10 not T=1000: diminishing returns — Gal shows T=10 approximation quality ≈ T=1000 for predictive mean; full reference uses 1000 for visualization only
- Weight decay as proxy for prior length-scale: weight decay is already in standard training; setting λ appropriately gives correct GP prior implicitly
- ReLU vs Tanh networks: ReLU uncertainty grows unboundedly away from data (correct for most problems); Tanh saturates (bounded uncertainty, inappropriate for extrapolation)

### Relevance to our project
- **This is the theoretical foundation for our MC-Dropout UQ component**
- We inherit their T=10 protocol from Asadi 2025
- σ_label = std of T=10 forward passes → our hierarchical fallback trigger: if σ_label > threshold, retreat to coarser parent label
- Justifies why dropout (already present in our DenseNet-121 head) can provide calibrated uncertainty without Bayesian retraining
- Predictive variance formula: Var ≈ sample variance of T passes — directly implementable

---

## Paper 11: AURC (Zhou 2025)

**Full citation:** Zhou, H., Van Landeghem, J., Popordanoska, T., & Blaschko, M. B. (2025). A Novel Characterization of the Population Area Under the Risk Coverage Curve (AURC) and Rates of Finite Sample Estimators. Proceedings of the 42nd ICML. PMLR 267. KU Leuven, Belgium + Instabase, USA. Correspondence: han.zhou@esat.kuleuven.be.

**Institutions:** KU Leuven (Processing Speech and Images) + Instabase (San Francisco)

### Problem
AURC (Area Under the Risk-Coverage Curve) is the primary metric for evaluating selective classifiers, but:
1. No formal population-level definition existed — only empirical sample estimates
2. Empirical AURC has quadratic O(n²) computational complexity
3. SELE score (common alternative lower bound) is significantly biased and not reliable
4. No known convergence guarantees for AURC estimators

### Methods

**Selective classifier formulation:**
- Classifier f: X → Δ^k (k-class simplex)
- Confidence Score Function (CSF) g: X → [0,1]
- Selective system: (f,g)(x) = f(x) if g(x) ≥ τ, else "abstain"
- Coverage: proportion of input space not abstained = E_{P(x)}[g̃(x)] where g̃(x) = 1[g(x)≥τ]
- Selective risk: R(f,g̃) = E_{P(x,y)}[ℓ(f(x),y) g̃(x)] / E_{P(x)}[g̃(x)]

**Population AURC (formal definition):**
```
AURC_p(f) = E_{(x,y)~P}[α(x) · ℓ(f(x),y)]
```
where:
```
α(x) = E_{x̃~P(x)}[1[g(x) ≥ g(x̃)] / E_{x'~P(x)}[1[g(x') ≥ g(x̃)]]]
     = -ln(1 - G(x))    (Proposition 3.1)
```
and G(x) = CDF of CSF g(x) = Pr(g(x') ≤ g(x))

**Interpretation:** AURC = expected risk weighted by α(x), where α(x) captures the importance of each point based on its population rank percentile in the confidence score distribution.

**Two plug-in estimators for finite samples D_n = {(x_i,y_i)}:**

Estimator 1 (â_i):
```
â_i = Σ_{j=1}^n 1[g(x_j) ≥ g(x_i)] / (Σ_{k=1}^n 1[g(x_k) ≥ g(x_k)])
    = Σ_{j=1}^{r_i} 1/(n-j+1) = H_n - H_{n-r_i}     (harmonic numbers)
```
where r_i is the rank of x_i sorted ascending by CSF.

Estimator 2 (â'_i):
```
â'_i = -ln(1 - r_i/(n+1))     (simpler, lower variance)
```

**Statistical properties:**
- Both estimators are consistent (converge to population α(x))
- MSE(â_i) ≈ β_i / (n(1-β_i)+1) — Proposition 3.5
- Both converge at rate O(√(ln(n)/n)) — faster than standard Monte Carlo O(n^{-1/2})
- Bias: both have non-zero bias for finite n, but bias decreases as n increases
- SELE estimator: significantly biased (underestimates population AURC), especially for small samples

**Computational complexity:**
- Empirical AURC (Eq. 10): O(n²) — nested loops over all pairs
- Plug-in estimators: O(n·ln(n)) — sorting operation dominates
- Code: github.com/han678/AsymptoticAURC

**Experiments:**
- Datasets: CIFAR10/100, ImageNet, Amazon Reviews
- Models: VGG16BN, ResNet, WideResNet, BERT, RoBERTa, Swin-Base
- Finding: SELE consistently underestimates population AURC_p; â and â' converge to AURC_p as n increases; 2×SELE overestimates
- CSF comparison: MaxLogit, MSP, Softmax Margin, MaxLogit-ℓ₂norm, Negative Gini Score — plug-in estimators work across all CSFs; SELE performance varies
- Fine-tuning with plug-in AURC as loss: converges and improves selective classifier performance

### Why these choices
- Formal population definition: needed to establish whether empirical estimates are converging to the right quantity; without this, AURC comparisons across papers are comparing different things
- Plug-in estimator via Monte Carlo: derives from the integral form of α(x); replacing the expectation with empirical CDF gives closed-form harmonic expression
- O(n·ln(n)) complexity: sorting required for rank computation; cannot do better than O(n·ln(n)) for comparison-based ranking
- Why not SELE: SELE was proposed as "close approximation to AURC" but this paper formally shows it's a biased lower bound — fine as optimization proxy, NOT for evaluation

### Relevance to our project
- **This paper provides the AURC evaluation metric for our UQ component**
- Our hierarchical fallback outputs: (f, g=MC-Dropout uncertainty threshold) — exactly a selective classifier
- We use AURC to measure: does our uncertainty threshold produce low error on accepted predictions?
- Plug-in estimators (â'_i): computationally efficient for our test sets — O(n·ln(n)) implementation
- Population vs empirical distinction: important for reporting — our test sets are finite; we should use plug-in estimators, not raw empirical AURC
- CSF = 1 - σ_label (inverted uncertainty, so higher = more confident = higher CSF) → feeds into AURC evaluation

---

## Paper 12: ConfHierClass (Mortier 2026)

**Full citation:** Mortier, T., Javanmardi, A., Sale, Y., Hüllermeier, E., & Waegeman, W. (2026). Conformal Prediction in Hierarchical Classification with Constrained Representation Complexity. Proceedings of AISTATS 2026, PMLR 300. Ghent University + LMU Munich + DFKI.

**Institutions:** Ghent University (Belgium) + LMU Munich (Germany) + DFKI (Kaiserslautern) + MCML

### Problem
Hierarchical classification: when a model is uncertain about a fine-grained prediction, returning a single leaf node prediction forces commitment to a specific answer. Better: return a coarser prediction that covers multiple possibilities, with formal coverage guarantees.

Challenge: conformal prediction for flat classification returns arbitrary subsets; hierarchical classification needs prediction sets constrained to nodes within the predefined hierarchy (clinically interpretable).

### Methods

**Hierarchical classification setup:**
- Class space Y = {c_1,...,c_K} organized as tree structure T with M nodes V_T
- Probabilistic hierarchical classifier: estimates P(c|x) for all c ∈ Y, with P(v|x) = Σ_{c∈v} P(c|x) (probability of internal node = sum of leaf descendants)
- Chain rule for hierarchical classifiers: P(v|x) = Π_{v'∈Path(v)} P(v'|pa(v'),x)

**Representation Complexity:**
- Defines how many internal nodes are needed to represent a prediction set Ŷ
- R_T(Ŷ) = min|V̂| where V̂ ⊂ V_T: ∪_{v∈V̂} v = Ŷ AND ∩_{v∈V̂} v = ∅
- Example: {1,2,4,7,8} in an 8-class tree needs 3 nodes to represent → R_T = 3

**Two algorithms:**

**Algorithm 1: CRSVP (Conformal Restricted Set-Valued Prediction)**
- Constraint: R_T(Ŷ) = 1 (only single internal nodes as predictions)
- Calibration: for each calibration sample, find internal node on path from mode leaf to root such that P(Ŷ) + u·P(pa(Ŷ)\Ŷ|x) ≥ τ (randomized for exact nominal coverage)
- Threshold τ* = ⌈(1-α)(N+1)⌉-th largest nonconformity score
- Inference (Algorithm 2): start from mode leaf, climb hierarchy until probability mass ≥ τ*
- Complexity: O(log K) at inference, O(NK) calibration

**Algorithm 2: CRSVP-r (Relaxed Representation Complexity)**
- Constraint: R_T(Ŷ) ≤ r (prediction sets expressible as up to r internal nodes)
- Finds set of "lowest common ancestors" with representation complexity ≤ r
- Optimization: A_r(S_k; x) = argmin{|Ŷ| - P̂(Ŷ|x)} s.t. R_T(Ŷ) ≤ r and A_r(S_{k-1};x) ∪ y^{(k)} ⊆ Ŷ
- Solved via dynamic programming (Algorithm 5): bottom-up traversal of hierarchy
- Complexity: O(K·2^r·d) where d = max out-degree — practical for r ≤ 3

**Marginal validity guarantee:**
```
P[y_{N+1} ∈ Ŷ(x_{N+1})] ≥ 1 - α
```
- Distribution-free: no assumptions on data distribution
- Valid for any trained classifier (flat or hierarchical)

**Datasets:** CIFAR-10, AMB (mouse brain single-cell), Caltech-101, DBpedia, Caltech-256, PlantCLEF 2015 (1000 classes)

### Results (Table 2, confidence level 90%):

| Dataset | Method | Coverage | Size | R_C |
|---------|--------|----------|------|-----|
| CIFAR-10 | CRSVP | 0.899 | 3.90 | 1.00 |
| CIFAR-10 | CRSVP-3 | 0.899 | 1.946 | 1.691 |
| PlantCLEF | CRSVP | 0.900 | 520.9 | 1.006 |
| PlantCLEF | CRSVP-3 | 0.900 | 389.7 | 1.632 |
| PlantCLEF | LAC (flat) | 0.901 | 25.5 | 24.3 |

- CRSVP achieves exact nominal coverage; NPS (naive, no randomization) undershoots
- CRSVP-3 dramatically smaller prediction sets than CRSVP at cost of multiple nodes
- Larger r (representation complexity allowance) → smaller, more informative sets
- When classifier is uncertain between branches, CRSVP predicts a high-level ancestor (very large set, semantically interpretable); CRSVP-3 finds multiple smaller ancestors

### Why these choices
- Representation complexity constraint: medical predictions must be interpretable; arbitrary subset of 520 plant species has no clinical meaning; "Respiratory disease" (ancestor node) does
- Randomization: needed for exact coverage at discrete probability jumps; without it, coverage can be below (1-α) for some samples
- Dynamic programming for r>1: direct search is exponential in r; DP breaks problem into sub-problems — each node solved once
- Split conformal (calibration set): avoids exchangeability assumptions beyond i.i.d. on calibration+test; no retraining required

### Relevance to our project
- **Direct conceptual basis for our hierarchical fallback:** CRSVP is the formal version of what we do — when uncertain, predict a coarser ancestor
- Our MC-Dropout fallback: when σ_label > threshold → predict parent label. This is CRSVP with threshold replacing τ* (same principle, different calibration mechanism)
- CRSVP provides coverage guarantees our MC-Dropout fallback does not — potential future extension
- CRSVP-r with r=2-3: could produce our canonical hierarchy predictions for uncertain cases without going all the way to root
- Clinical ICD hierarchy parallels our PDPI/PNPK/KKI canonical tree: internal nodes are semantically meaningful labels

---

## Cross-Paper Synthesis

### Hierarchy-Aware Training (Train-Time)
| Paper | Method | When hierarchy applied |
|-------|--------|----------------------|
| Pham 2020 | Conditional training | Train subset selection |
| Chen 2020 | Two-stage conditional + unconditional fine-tuning | Train |
| Asadi 2025 (CIHMLC) | HBCE penalty | Train (loss term) |
| Our project | HBCE (inherited) | Train; UQ at inference |

### Inference-Time Hierarchy
| Paper | Method | Guarantee |
|-------|--------|----------|
| Asadi 2025 | MC-Dropout σ > threshold → parent | None formal |
| Mortier 2026 | CRSVP: conformal calibrated threshold | Coverage ≥ 1-α |
| Our project | MC-Dropout threshold (inherited from Asadi) | None formal (future work: add conformal calibration) |

### Uncertainty Quantification
| Paper | Method | What uncertainty measures |
|-------|--------|--------------------------|
| Gal 2016 | MC-Dropout T passes | Model parameter uncertainty |
| Asadi 2025 | MC-Dropout N=10 | Per-label prediction uncertainty |
| Zhou 2025 | AURC evaluation framework | Quality of UQ for selective prediction |
| Our project | MC-Dropout N=10 (inherited) + AURC (added) | UQ quality at 51-label level |

### Multi-Dataset / Partial Label Training
| Paper | Method | Handles domain shift | Handles label discrepancy |
|-------|--------|---------------------|--------------------------|
| Luo 2020 | TAT + UTE | Yes (TAT) | Yes (UTE pseudo-labels) |
| Kang 2023 | Dynamic adapter + pseudo-sharpening | No | Yes |
| Asadi 2025 | Masked BCE (NaN masking) | No | Yes (masks missing) |
| Our project | Masked BCE + 6-dataset pipeline | Planned (TAT-inspired) | Yes (NaN masking) |

### Loss Functions for Imbalance
| Paper | Loss | Handles imbalance by |
|-------|------|---------------------|
| CXR-LT 2024 top teams | Asymmetric Loss | Up-weight positive/hard samples |
| HP-ViT | BAFL | Curriculum: start balanced, increase difficulty focus |
| LMeRAN | Standard BCE + label mask | Label co-occurrence learning |
| Our project | HBCE (hierarchy penalty) | Hierarchy consistency |

### Evaluation Metrics
| Paper | Primary metric | Rationale |
|-------|---------------|-----------|
| CheXpert | AUC per label | Standard CXR benchmark |
| CXR-LT 2024 | mAP (macro) | AUROC inflated for rare classes |
| Zhou 2025 | AURC (population) | Evaluates UQ quality for selective prediction |
| Our project | AUROC + mAP + AURC | Comprehensive: classification + rarity-aware + UQ quality |

---

## Key Takeaways for Our Research

1. **Masked BCE is standard** (CheXpert U-Ignore = our NaN masking) — no deviation needed.

2. **Hierarchy at train time (HBCE) AND inference time (MC-Dropout fallback)** is our novel combination — no existing paper does both with Indonesian clinical labels.

3. **AURC is non-trivial** — Zhou 2025 shows naive empirical AURC has bias; use plug-in estimator â'_i for evaluation.

4. **mAP alongside AUROC** — CXR-LT 2024 confirms AUROC misleading for imbalanced/rare-class settings; report both.

5. **T=10 MC-Dropout sufficient** — Gal 2016 theory + Asadi 2025 practice both validate T=10.

6. **51 canonical labels + 17 with signal** — our setup is harder than any single-paper benchmark; CXR-LT 2024 (45 labels) is closest prior scale.

7. **Our hierarchical fallback ≈ CRSVP without formal coverage guarantee** — Mortier 2026 provides the theoretical framing; we can cite it as motivation and note that formal conformal calibration is future work.

8. **Multi-dataset training requires both domain adaptation and label alignment** — Luo 2020 (TAT+UTE) and Kang 2023 (adapter+pseudo) show neither alone is sufficient; our masked BCE handles label alignment, domain shift is an open gap.
