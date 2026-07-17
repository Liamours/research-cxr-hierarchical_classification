# Classification Results (PadChest-inclusive corpus)

Generated: 2026-07-16 22:18

Val and test performance for the two trained conditions on the current corpus:
527,745 images, 7 datasets (incl. PadChest), 27 labels with signal, stratified
80/10/10 split (seed 42). Val = 43,969 images, Test = 43,696 images.

- **Flat** — `configs/densenet121_xrv__flat.yaml`, MaskedBCE loss, DenseNet121-XRV backbone.
- **Hierarchical** — `configs/densenet121_xrv__hierarchical.yaml`, HBCE loss (lambda=0.5), same backbone.

Checkpoints: `result/20260712_densenet121_xrv__flat/`, `result/20260712_densenet121_xrv__hierarchical/`
(`best_val_auroc_macro.pt`). Bold marks the better value per row (direction-aware: AURC/ECE/HCV
lower is better, everything else higher is better). A few near-ties display identically at 4
decimals but differ at higher precision, so bold reflects the true winner even when printed
digits look equal.

---

## Overall metrics (VAL)

| Metric | Flat | Hierarchical |
|---|--:|--:|
| AUROC macro | **0.6735** | 0.6732 |
| AUROC micro | **0.9037** | 0.9014 |
| mAP macro | **0.1890** | 0.1880 |
| AURC macro (low=better) | **0.0156** | 0.0177 |
| AURC flat (low=better) | **0.0101** | 0.0106 |
| ECE (low=better) | **0.0043** | 0.0055 |
| HCV rate % (low=better) | 39.28 | **36.48** |
| F1 macro | **0.1182** | 0.1129 |
| F1 micro | **0.4716** | 0.4372 |
| precision_macro | 0.6095 | **0.6462** |
| precision_micro | **0.7670** | 0.7491 |
| precision_weighted | **0.6368** | 0.6252 |
| recall_macro | **0.1016** | 0.0924 |
| recall_micro | **0.3404** | 0.3086 |
| recall_weighted | **0.3404** | 0.3086 |
| specificity_macro | 0.9922 | **0.9933** |
| specificity_micro | 0.9922 | **0.9922** |
| specificity_weighted | 0.9661 | **0.9703** |
| accuracy_macro | **0.9616** | 0.9561 |
| accuracy_micro | **0.9463** | 0.9441 |
| accuracy_weighted | **0.9100** | 0.8875 |
| balanced_accuracy_macro | **0.5468** | 0.5427 |
| balanced_accuracy_micro | **0.6663** | 0.6504 |
| balanced_accuracy_weighted | **0.6533** | 0.6395 |
| mcc_macro | **0.2802** | 0.2798 |
| mcc_micro | **0.4891** | 0.4585 |
| mcc_weighted | **0.3647** | 0.3266 |
| subset_accuracy | **0.5960** | 0.5733 |

## Overall metrics (TEST)

| Metric | Flat | Hierarchical |
|---|--:|--:|
| AUROC macro | **0.6954** | 0.6872 |
| AUROC micro | **0.9048** | 0.9025 |
| mAP macro | **0.1999** | 0.1986 |
| AURC macro (low=better) | **0.0145** | 0.0170 |
| AURC flat (low=better) | **0.0101** | 0.0105 |
| ECE (low=better) | **0.0044** | 0.0055 |
| HCV rate % (low=better) | 39.27 | **36.53** |
| F1 macro | **0.1173** | 0.1107 |
| F1 micro | **0.4679** | 0.4339 |
| precision_macro | **0.5927** | 0.5814 |
| precision_micro | **0.7618** | 0.7443 |
| precision_weighted | **0.6015** | 0.6003 |
| recall_macro | **0.1041** | 0.0931 |
| recall_micro | **0.3376** | 0.3062 |
| recall_weighted | **0.3376** | 0.3062 |
| specificity_macro | 0.9919 | **0.9931** |
| specificity_micro | 0.9920 | **0.9920** |
| specificity_weighted | 0.9653 | **0.9698** |
| accuracy_macro | **0.9613** | 0.9556 |
| accuracy_micro | **0.9459** | 0.9437 |
| accuracy_weighted | **0.9086** | 0.8866 |
| balanced_accuracy_macro | **0.5476** | 0.5428 |
| balanced_accuracy_micro | **0.6648** | 0.6491 |
| balanced_accuracy_weighted | **0.6515** | 0.6380 |
| mcc_macro | **0.2762** | 0.2525 |
| mcc_micro | **0.4850** | 0.4549 |
| mcc_weighted | **0.3578** | 0.3236 |
| subset_accuracy | **0.5892** | 0.5665 |

---

## Per-label AUROC (VAL)

| Label | Flat | Hierarchical |
|---|--:|--:|
| Pneumonia | **0.7288** | 0.7025 |
| Tuberculosis | 0.9065 | **0.9101** |
| Bronchiolitis | 0.6572 | **0.6650** |
| Acute_Bronchitis | 0.6472 | **0.6482** |
| COVID19_Pneumonia | **0.9721** | 0.9399 |
| COPD | 0.7220 | **0.7233** |
| Bronchiectasis | **0.5051** | 0.4977 |
| Post_TB_Obstructive_Syndrome | 0.4501 | **0.4705** |
| Pleural_Effusion | **0.8782** | 0.8779 |
| Pneumothorax | 0.8753 | **0.8776** |
| Hydropneumothorax | 0.3009 | **0.3866** |
| Lung_Cancer | **0.4430** | 0.3284 |
| Pulmonary_Metastases | 0.5353 | **0.5650** |
| Mediastinal_Tumor | **0.8832** | 0.8240 |
| Solitary_Pulmonary_Nodule | **0.7856** | 0.7846 |
| ILD | **0.7142** | 0.7129 |
| IPF | 0.4978 | **0.5362** |
| Asbestosis | 0.3463 | **0.3816** |
| Pulmonary_Edema | **0.8946** | 0.8944 |
| Pulmonary_Hypertension | **0.6298** | 0.6240 |
| Cardiomegaly | **0.8257** | 0.8257 |
| Chest_Trauma | 0.6822 | **0.6831** |
| Subcutaneous_Emphysema | **0.7458** | 0.7441 |
| Diaphragmatic_Hernia | 0.7753 | **0.7842** |
| Airway_Foreign_Body | 0.3705 | **0.3775** |
| Atelectasis | **0.7376** | 0.7367 |

*Lung_Cancer appears only in VAL (0 positives on test).*

## Per-label AUROC (TEST)

| Label | Flat | Hierarchical |
|---|--:|--:|
| Pneumonia | **0.7244** | 0.6963 |
| Tuberculosis | 0.9048 | **0.9062** |
| Bronchiolitis | 0.6848 | **0.6903** |
| Acute_Bronchitis | 0.7030 | **0.7061** |
| COVID19_Pneumonia | **0.9653** | 0.9290 |
| COPD | 0.7218 | **0.7223** |
| Bronchiectasis | **0.5412** | 0.5351 |
| Post_TB_Obstructive_Syndrome | **0.5620** | 0.5483 |
| Pleural_Effusion | **0.8792** | 0.8790 |
| Pneumothorax | 0.8761 | **0.8778** |
| Hydropneumothorax | **0.3120** | 0.1979 |
| Pulmonary_Metastases | **0.6412** | 0.6288 |
| Mediastinal_Tumor | **0.5993** | 0.4984 |
| Solitary_Pulmonary_Nodule | **0.7882** | 0.7870 |
| ILD | **0.7157** | 0.7143 |
| IPF | 0.5733 | **0.5834** |
| Asbestosis | **0.6074** | 0.6001 |
| Pulmonary_Edema | **0.9044** | 0.9043 |
| Pulmonary_Hypertension | 0.5033 | **0.5551** |
| Cardiomegaly | **0.8405** | 0.8401 |
| Chest_Trauma | 0.6943 | **0.6949** |
| Subcutaneous_Emphysema | **0.6002** | 0.5975 |
| Diaphragmatic_Hernia | **0.7873** | 0.7872 |
| Airway_Foreign_Body | 0.5135 | **0.5616** |
| Atelectasis | **0.7412** | 0.7392 |

---

## Per-label AP (VAL)

| Label | Flat | Hierarchical |
|---|--:|--:|
| Pneumonia | **0.0961** | 0.0742 |
| Tuberculosis | **0.5538** | 0.5508 |
| Bronchiolitis | 0.1029 | **0.1098** |
| Acute_Bronchitis | 0.2061 | **0.2103** |
| COVID19_Pneumonia | **0.9867** | 0.9722 |
| COPD | 0.1576 | **0.1584** |
| Bronchiectasis | 0.0137 | **0.0140** |
| Post_TB_Obstructive_Syndrome | 0.0049 | **0.0051** |
| Pleural_Effusion | **0.6911** | 0.6908 |
| Pneumothorax | 0.2865 | **0.2920** |
| Hydropneumothorax | 0.0002 | **0.0002** |
| Lung_Cancer | **0.0019** | 0.0016 |
| Pulmonary_Metastases | 0.0021 | **0.0021** |
| Mediastinal_Tumor | **0.0088** | 0.0058 |
| Solitary_Pulmonary_Nodule | 0.2895 | **0.2913** |
| ILD | **0.1201** | 0.1158 |
| IPF | 0.0073 | **0.0080** |
| Asbestosis | **0.0003** | 0.0003 |
| Pulmonary_Edema | **0.5210** | 0.5208 |
| Pulmonary_Hypertension | 0.0015 | **0.0015** |
| Cardiomegaly | 0.4091 | **0.4100** |
| Chest_Trauma | **0.1136** | 0.1129 |
| Subcutaneous_Emphysema | **0.0071** | 0.0064 |
| Diaphragmatic_Hernia | 0.0727 | **0.0753** |
| Airway_Foreign_Body | 0.0004 | **0.0004** |
| Atelectasis | **0.2591** | 0.2567 |

## Per-label AP (TEST)

| Label | Flat | Hierarchical |
|---|--:|--:|
| Pneumonia | **0.0907** | 0.0687 |
| Tuberculosis | **0.5190** | 0.5035 |
| Bronchiolitis | 0.1253 | **0.1439** |
| Acute_Bronchitis | 0.2718 | **0.2750** |
| COVID19_Pneumonia | **0.9829** | 0.9642 |
| COPD | 0.1507 | **0.1523** |
| Bronchiectasis | **0.0151** | 0.0149 |
| Post_TB_Obstructive_Syndrome | **0.0070** | 0.0066 |
| Pleural_Effusion | 0.6933 | **0.6939** |
| Pneumothorax | 0.3220 | **0.3315** |
| Hydropneumothorax | 0.0002 | **0.0002** |
| Pulmonary_Metastases | **0.0032** | 0.0032 |
| Mediastinal_Tumor | **0.0027** | 0.0022 |
| Solitary_Pulmonary_Nodule | 0.2665 | **0.2677** |
| ILD | **0.1051** | 0.1011 |
| IPF | 0.0075 | **0.0081** |
| Asbestosis | **0.0007** | 0.0006 |
| Pulmonary_Edema | 0.5406 | **0.5422** |
| Pulmonary_Hypertension | 0.0012 | **0.0013** |
| Cardiomegaly | **0.4348** | 0.4340 |
| Chest_Trauma | 0.1097 | **0.1103** |
| Subcutaneous_Emphysema | **0.0070** | 0.0053 |
| Diaphragmatic_Hernia | **0.0836** | 0.0809 |
| Airway_Foreign_Body | 0.0005 | **0.0005** |
| Atelectasis | **0.2552** | 0.2531 |

---

## Reading it honestly

- Val and test are close for both models on every headline metric (e.g. flat AUROC 0.6735 val vs
  0.6954 test) -- no overfitting, the stratified split is behaving as intended.
- Flat wins the large majority of overall metrics and most per-label AUROC/AP rows, especially
  ranking metrics (AUROC, mAP, MCC).
- Hierarchical's wins cluster in specificity, HCV (hierarchy violation rate), and several
  PadChest-derived rare labels (Bronchiolitis, IPF, Hydropneumothorax on val; Pulmonary_Hypertension,
  Airway_Foreign_Body on test). Hierarchical does not clearly win calibration this time: AURC and ECE
  are both slightly worse for hierarchical than flat on this corpus, reversing an earlier
  pre-PadChest finding.
- Rare/noisy PadChest labels drag mAP down for both models (Hydropneumothorax, Asbestosis,
  Pulmonary_Hypertension all under 0.01 AP) -- expected given their tiny positive counts (3-88 total
  across the whole corpus).
- Raw accuracy is inflated by class imbalance (0.94-0.96 macro/micro). MCC (~0.28 macro) and
  balanced accuracy (~0.55 macro) are the more honest signal for this long-tailed, 27-label setup.
