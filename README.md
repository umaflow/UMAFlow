# Physics-Aligned 3D Molecule Generation in Protein Pockets

> Official code repository for **"Physics-Aligned 3D Molecule Generation in Protein Pockets"**.

This repository will contain the implementation of a representation alignment (REPA) scheme that grounds pocket-conditioned 3D molecule generative models in atomic-level physics by aligning intermediate denoiser features with those of a frozen, pretrained machine-learned interatomic potential (MLIP).

---

## Overview

Structure-based generative models for 3D molecule generation are typically trained with purely geometric supervision, which lacks information about the underlying energy landscape and atomic forces. We address this by aligning the latent representations of a pocket-conditioned denoiser (DrugFlow) with those of a frozen MLIP encoder (UMA) via lightweight equivariant projectors. This provides a dense, physically grounded training signal — at **zero additional cost during sampling**.

**Key results:**
- **10× training speedup** to reach baseline performance levels
- **Improved FCD** (3.42 vs. 4.10 for DrugFlow+EMA) and structural validity
- **Superior OOD robustness** on unseen protein targets (Runs-N-Poses subset)
- **Lower ligand strain** and fewer steric clashes


## Repository structure
> *Placeholder — to do.*


## Installation

> *Placeholder — to do.*

## Data

We use:
1. **CrossDocked** — same training/validation/test splits as DrugFlow (100,000 / 100 / 100 complexes).
2. **Runs-N-Poses** (hardest subset) — 95 targets reduced to 68 after removing redundant pockets and ion-containing complexes. Used for out-of-distribution evaluation.

Preparation scripts will be provided in `data/`.

---

## Reproducing paper results

We will provide pretrained checkpoints that were used for all results in the paper, along with training scripts and evaluation and analysis scripts to reproduce all results from the paper.

---

## Acknowledgements

This work builds on:
- **DrugFlow** — pocket-conditioned flow-matching denoiser
- **UMA** — pretrained machine-learned interatomic potential
- **REPA** (Yu et al., ICLR 2025) — original representation alignment idea for diffusion models
- **MACE-REPA** (Pinede et al., ICML GenBio 2025) — REPA applied to molecular force fields

---

## License
> *Placeholder — to do.*
