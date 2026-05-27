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

## Setup

### Conda Environment

Create a conda/mamba environment 
```bash
conda env create -f environment.yaml -n drugflow
conda activate drugflow
```

and add the Gnina executable for docking score computation
```bash
wget https://github.com/gnina/gnina/releases/download/v1.1/gnina -O $CONDA_PREFIX/bin/gnina
chmod +x $CONDA_PREFIX/bin/gnina
```

## Dataset preparation

### Download pre-processed dataset
The preprocessed dataset is available on Zenodo
```bash
wget https://zenodo.org/records/14919171/files/processed_crossdocked.zip
unzip processed_crossdocked.zip
```

### Add UMA paths to the dataset
UMA embeddings are stored separately from the main dataset, so we need to add the path to the corresponding UMA file for each complex in the train and val splits. 

```bash
PATH_TO_PROCESSED_DATA=...  # e.g. /mnt/datasets/processed_crossdocked
PATH_TO_UMA_EMBEDDINGS=...  # e.g. /mnt/datasets
python scripts/python/uma_embeddings/build_uma_complex_id_mapping.py \
  --splits $PATH_TO_PROCESSED_DATA/train.pt $PATH_TO_PROCESSED_DATA/val.pt \
  --embeddings-dir $PATH_TO_UMA_EMBEDDINGS
```

Optionally verify that the mapping is correct by running:
```bash
python scripts/python/uma_embeddings/verify_uma_atom_order.py \
  --splits $PATH_TO_PROCESSED_DATA/val.with_uma.pt \ 
  --embeddings-dir $PATH_TO_UMA_EMBEDDINGS \
```
or longer: 
```bash
python scripts/python/uma_embeddings/verify_uma_atom_order.py \
  --splits $PATH_TO_PROCESSED_DATA/train.with_uma.pt \ 
  --embeddings-dir $PATH_TO_UMA_EMBEDDINGS \
```

## Training

Example config files are provided for:
- UMAFlow: `CONFIG=configs/training/repa.yml`
- DrugFlow: `CONFIG=configs/training/drugflow.yml`
- DrugFlow+EMA: `CONFIG=configs/training/drugflow_ema.yml`

Create a symlink to the processed dataset and for the output directory
```bash
LOGDIR=...  # where checkpoints, and validation outputs will be saved
ln -s $LOGDIR runs
```

To launch the training job for the DrugFlow base model, for example, run
```bash
python src/train.py --config $CONFIG
```

## Reproducing paper results
- training scripts
- evaluation and analysis scripts to reproduce all results from the paper.

### Checkpoints 
> *Placeholder — to do, provide links to pretrained checkpoints on Zenodo or elsewhere.*
```bash
# UMAFlow model
todo

# Baseline DrugFlow model
todo

# Baseline DrugFlow + EMA
todo
```

### Samples
> *Placeholder — to do, provide links to sampled datasets on Zenodo or elsewhere.*


### Sampling for all proteins in the test set

Select checkpoint, e.g. `checkpoints/drugflow.ckpt`, and specify it in `configs/sampling/sample_and_maybe_evaluate.yml`.

Furthermore, you need to update the `sample_outdir` parameter in the sampling config file or link the desired output location
```bash
SAMPLE_OUTDIR=...  # where samples will be saved
ln -s $SAMPLE_OUTDIR samples
```

For sampling, run
```bash
python src/sample_and_evaluate.py --config configs/sampling/sample_and_maybe_evaluate.yml
```
which supports parallelization across target pockets by specifying `--job_id` and `--n_jobs`.
To also evaluate the results, set `evaluate: True` in the sampling config file.

### Evaluating samples

To evaluate samples, specify:

```bash
SAMPLES_DIR=...  # Location where the sampled dataset is stored
EVALUATED_DATA_ALL=...  # Temporary directory for evaluation output
EVALUATED_DATA=...  # Evaluation output
```

Run the evaluation:
```bash
python scripts/python/evaluate_baselines.py \
       --in_dir $SAMPLES_DIR \
       --out_dir $EVALUATED_DATA_ALL

python scripts/python/postprocess_metrics.py \
       --in_dir $EVALUATED_DATA_ALL \
       --out_dir $EVALUATED_DATA
```

Per-sample evaluation results will be stored in ```EVALUATED_DATA/metrics_detailed.csv``` and aggregated metrics will be stored in ```EVALUATED_DATA/metrics_aggregated.csv```.


## OOD evaluation on Runs-N-Poses 
> *Placeholder — to do, provide instructions for reproducing the OOD evaluation on the Runs-N-Poses subset, including links to the subset and any necessary checkpoints or sampled datasets.*
Dataset: **Runs-N-Poses** (hardest subset) — 95 targets reduced to 68 after removing redundant pockets and ion-containing complexes. Used for out-of-distribution evaluation.




## Acknowledgements

This work builds on:
- **DrugFlow** (Schneuing et al., ICLR 2025) — pocket-conditioned flow-matching denoiser
- **UMA** (Wood et al., NeurIPS 2025) — pretrained machine-learned interatomic potential
- **REPA** (Yu et al., ICLR 2025) — original representation alignment idea for diffusion models
- **MACE-REPA** (Pinede et al., ICML GenBio 2025) — REPA applied to molecular force fields

