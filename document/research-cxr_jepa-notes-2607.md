# JEPA Notes: Image Generation + Paper History

> Compiled 2026-07-09 from web research. Two questions: (1) can JEPA be used for image generation, (2) history of JEPA through its papers.

## Q1: JEPA for image generation — viable?

**Verdict: viable and actively researched, but never JEPA alone.** The obstacle is NOT that JEPA is non-contrastive (that only means no negative pairs, like BYOL). The obstacle is that JEPA is **non-generative by design**: it predicts in latent space and has no pixel decoder, discarding the low-level detail (texture, exact color, high frequencies) needed to render images. Every working system bolts a stochastic generative head (diffusion / flow matching) onto JEPA-style latents.

Consensus shape of the field: **JEPA for the semantics, diffusion/flow-matching for the pixels.**

### Direct JEPA-based generators

| Model | Date | What it does |
|-------|------|--------------|
| **D-JEPA** | Oct 2024 | Flagship. Reinterprets JEPA as generalized next-token prediction + per-token diffusion/flow loss. Three-ViT backbone (context enc, target enc, predictor) generates images autoregressively in continuous space. ImageNet class-conditional FID ~4.0, fast sampling (~43 ms). Extended to video/audio. [arXiv 2410.03755](https://arxiv.org/abs/2410.03755), [project](https://d-jepa.github.io/) |
| **JEPA-T** | Oct 2025 | Text-to-image on the D-JEPA line. Joint-embedding predictive Transformer over visual+text tokens, cross-attention fusion after predictor, flow matching. ImageNet-1K 256px; beats non-fusion/late-fusion baselines. [arXiv 2510.00974](https://arxiv.org/abs/2510.00974), [code](https://github.com/justin-herry/JEPA-T) |

### JEPA latents conditioning a separate generator

- **RCDM** (Bordes et al.) — diffusion decoder conditioned on frozen SSL representations; how Meta visualized I-JEPA predictions as pixels. [survey ref](https://arxiv.org/pdf/2407.00783)
- **RCG** (Li et al., "Self-conditioned Image Generation via Generating Representations") — small diffusion model samples in frozen SSL representation space, pixel generator decodes. Encoder-agnostic recipe (paper used MoCo v3; a JEPA encoder drops in). Unconditional ImageNet 256 FID 3.31. [arXiv 2312.03701](https://arxiv.org/abs/2312.03701v1)
- **JEPA-guided minority sampling** (May 2026) — JEPA encoder as "world prior" steering diffusion toward rare samples. [arXiv 2605.24631](https://arxiv.org/html/2605.24631v1)

### JEPA-flavored representations improving generators (pragmatic winner)

- **REPA** (ICLR 2025) — align diffusion transformer internal features with DINOv2 → >17.5× faster convergence, FID 1.42. Strongest evidence that semantic latent representations help generation even when the generator stays a diffusion model. [project](https://sihyun.me/REPA/), [ICLR paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/d9e42b4d7163931f3689d6d6fbaa11d0-Paper-Conference.pdf)
- Follow-ups: [Representation Entanglement](https://arxiv.org/html/2507.01467v1), [REPA for pixel-space transformers](https://arxiv.org/html/2603.14366v1)

### Relevance to CXR

If ever needed: established template = pretrain JEPA on domain images, decode with a small conditional diffusion head (D-JEPA / RCG style).

## Q2: JEPA history through papers

### Prehistory
- **1993–2006** — LeCun's Siamese networks (signature verification 1993) + energy-based learning tutorials: compare in embedding space, score with energy, don't reconstruct.
- **2020–2021** — BYOL, SimSiam, **VICReg** (Bardes/Ponce/LeCun): non-contrastive collapse avoidance via asymmetry or variance–covariance regularization. Joint-embedding architectures (JEA) — invariant embeddings of augmented views, but nothing *predicted*.
- **2022** — **data2vec** (Baevski et al., Meta): predicts teacher representations of masked inputs across modalities. JEPA sibling before the name existed.

### Founding document
- **Jun 2022** — LeCun, ["A Path Towards Autonomous Machine Intelligence"](https://openreview.net/forum?id=BZ5a1r-kVsf) (~60 pp position paper). Coins **JEPA**: predict in learned latent space, not observation space — pixel-level generative prediction wastes capacity on unpredictable detail. Sketches hierarchical JEPA (H-JEPA) for world-model agents. No experiments.

### Instantiations by modality
- **Jan 2023 — I-JEPA** (Assran et al., CVPR 2023). First real JEPA: context encoder sees patches, predictor guesses latents of masked target blocks, EMA target encoder. No augmentations/negatives/pixel loss. Beat MAE on linear probes with less compute. [code](https://github.com/facebookresearch/ijepa), [Meta blog](https://ai.meta.com/blog/yann-lecun-ai-model-i-jepa/)
- **Jul 2023 — MC-JEPA** (Bardes, Ponce, LeCun) — joint motion (optical flow) + content.
- **Nov 2023 — A-JEPA** — audio spectrograms.
- **Feb 2024 — V-JEPA** (Bardes et al.) — video, masked spatiotemporal latent prediction; strong on motion-heavy frozen eval.
- **2024 — IWM / Image World Models** (Garrido et al.) — predictor applies *transformations* in latent space → controllable world model, not just mask-filling.
- **Oct 2024 — D-JEPA** — crossover into generation (see Q1).
- **2024–2025 — domain explosion** — Point-JEPA, Graph-JEPA, Brain-JEPA (fMRI), TS-JEPA, TI-JEPA/Text-JEPA, ECG/EEG ([Laya](https://arxiv.org/html/2603.16281v1)). [Turing Post counts 14+ milestones](https://www.turingpost.com/p/jepamap). Anything maskable is JEPA-able.

### Scaling and theory (current era)
- **Jun 2025 — V-JEPA 2** (Meta) — ~1M hours video, ViT-g; **V-JEPA 2-AC** action-conditioned head → zero-shot robot planning/manipulation. First concrete "JEPA as world model for control."
- **Oct 2025 — JEPA-T** (see Q1).
- **Nov 2025 — LeJEPA** (Balestriero & LeCun; LeCun's last Meta work before leaving to found his startup). Theory paper: optimal embedding distribution for downstream risk is an **isotropic Gaussian**, enforced via **SIGReg** (sketched isotropic Gaussian regularization). Deletes anti-collapse heuristics (EMA teacher, stop-gradient, asymmetric predictor) in one ~50-line objective; stable across 60+ architectures up to 1.8B params; training loss finally correlates with downstream performance. [arXiv 2511.08544](https://arxiv.org/abs/2511.08544), [code](https://github.com/rbalestr-lab/lejepa)
- **2026 — consolidation/hybrids** — [Var-JEPA](https://arxiv.org/html/2603.20111) (variational formulation bridging predictive & generative SSL), [EPM-JEPA](https://arxiv.org/pdf/2606.12979) (experience-modulated world models), ThinkJEPA (reasoning). LeJEPA-style training spreading to applied domains.

### Arc in one paragraph
Three acts: **(1)** 2022 — position paper argues latent-space prediction beats pixel reconstruction. **(2)** 2023–2025 — Meta validates modality-by-modality (image → video → audio → robotics); community forks it everywhere; whole family runs on fragile anti-collapse heuristics. **(3)** 2025–2026 — LeJEPA replaces heuristics with theory; generative crossovers (D-JEPA, JEPA-T, Var-JEPA) blur the generative/non-generative boundary the 2022 paper drew. Still unbuilt from the manifesto: hierarchical JEPA stacks doing multi-timescale planning.

### Misc references
- [Rohit Bandaru's JEPA deep dive](https://rohitbandaru.github.io/blog/JEPA-Deep-Dive/)
- [Turing Post JEPA explainer](https://www.turingpost.com/p/jepa)
- [Turing Post LeJEPA](https://www.turingpost.com/p/lejepa)
