# Embedding Preparation


Precomputes per-atom UMA embeddings for every CrossDocked pocket10
complex and use them as targets for the Representation-Alignment
(REPA) auxiliary loss.


## Setup
Create the environment and install `fairchem-core` in editable mode:
```bash
conda env create -f embedding_preparation/environment.yml
conda activate uma
pip install -e external/fairchem/packages/fairchem-core   # pinned at 7ca88cc4
```
UMA itself is consumed via the public
[`fairchem-core`](https://github.com/facebookresearch/fairchem) package
— we use unmodified upstream pinned at commit
**`7ca88cc487349755f2250920d17a843256403080`**, added to this repo as a
git submodule:


## Data

Download and extract the raw CrossDocked dataset as described by the
authors of Pocket2Mol
(https://github.com/pengxingang/Pocket2Mol/tree/main/data). This gives
the `crossdocked_pocket10/` directory passed below as `--data-dir`.

## Run

Smoke test (~50 complexes):

```bash
python embedding_preparation/scripts/extract_uma_embeddings.py \
  --uma-variant uma-s-1p1 \
  --data-dir /PATH/TO/crossdocked_pocket10 \
  --output-dir /PATH/TO/embeddings/uma_s_depth_2 \
  --device cuda --block-idx 1 --num-prefetch 4 --limit 50
```

Full dataset, sharded (one GPU per shard). Drop
`--limit`, set `--num-shards`, and launch one process per `--shard-id`
(directly or as a job-array body — `--shard-id` is the only thing that
differs between tasks). All shards write into the same `--output-dir`:

```bash
python embedding_preparation/scripts/extract_uma_embeddings.py \
  --uma-variant uma-s-1p1 \
  --data-dir /PATH/TO/crossdocked_pocket10 \
  --output-dir /PATH/TO/embeddings/uma_s_depth_2 \
  --device cuda --block-idx 1 --num-prefetch 4 \
  --num-shards 16 --shard-id "$SHARD_ID"
```

Resume by rerunning the same command — already-written `.pt` files are
skipped. Completeness check:

```bash
find /PATH/TO/embeddings/uma_s_depth_2 -name '*.pt' | wc -l   # expect 183468
```

Concatenate the per-shard audit TSVs when done:

```bash
for kind in pocket ligand; do
  head -1 ${kind}_h_audit.shard_00_of_16.tsv > ${kind}_h_audit.tsv
  tail -n +2 -q ${kind}_h_audit.shard_*_of_16.tsv >> ${kind}_h_audit.tsv
done
```

## Variants

| Model | Depth | Extractor flags                         |
|-------|-------|-----------------------------------------|
| UMA-S | 2     | `--uma-variant uma-s-1p1 --block-idx 1` |
| UMA-M | 2     | `--uma-variant uma-m-1p1 --block-idx 1` |
| UMA-M | 5     | `--uma-variant uma-m-1p1 --block-idx 4` |
| UMA-M | 8     | `--uma-variant uma-m-1p1 --block-idx 7` |

Point `--output-dir` at a separate directory per variant.


# Details
For each complex (paired `*.sdf` ligand + `*_pocket10.pdb` receptor),
the script (i) parses both files with RDKit, (ii) adds hydrogens via
`Chem.AddHs(addCoords=True)` — strict-sanitize path with a
relaxed-sanitizer fallback for OpenBabel-broken SDFs and non-standard
residues (every fallback is logged), (iii) assembles the system in the
canonical layout `[pocket_heavy, ligand_heavy, pocket_H, ligand_H]` so
that ligand-heavy embedding rows align 1:1 with the original SDF
atom-block order, and (iv) forwards through UMA with
`task_name="omol"`, capturing the **raw pre-norm** output of
`backbone.blocks[--block-idx]` via a forward hook. Outputs are saved
one `.pt` per complex (resume-safe) into `--output-dir`, along with a
`metadata.json` that pins the UMA variant, lmax, channel count, block
index, and protonation policy. Run with
`python extract_uma_embeddings.py --uma-variant {uma-s-1p1,uma-m-1p1} --data-dir <CROSSDOCKED> --output-dir <OUT> --device cuda --block-idx <K>`;
the script also supports `--num-shards`/`--shard-id` for trivial
1-process-per-GPU parallelism across a SLURM array. The main paper
uses UMA-S at depth 2 (`--block-idx 1`); supplementary results sweep
UMA-M at depths 2, 5, and 8 (`--block-idx 1`, `4`, `7`).
