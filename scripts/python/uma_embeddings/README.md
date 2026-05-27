# UMA embeddings — DrugFlow integration

These scripts bridge DrugFlow's processed dataset and pretrained UMA
per-complex embeddings, which atom-level REPA training pulls in as an
auxiliary alignment target.

## Workflow

```
build_uma_complex_id_mapping.py     verify_uma_atom_order.py
  (run once per dataset)              (run once per embedding source)
        │                                       │
        ▼                                       ▼
  processed_crossdocked/             "Atom-order aligned: YES"
  {train,val}.with_uma.pt
        │
        ▼
  dataset loader at training time
```

The mapping is **source-invariant**: any UMA dir that follows the same
`<complex_id>.pt` naming convention can use the same file. Atom-order
alignment, however, depends on whether the specific UMA source includes
hydrogens, filters atoms, etc. — that's what the verify step checks.

## 1. Build the `ligand_name → complex_id` mapping

For each ligand in a DrugFlow split, finds the matching UMA `.pt` by stem
lookup + nearest-neighbour coordinate verification, then writes the split
back with a per-ligand `complex_id` field (a positional list parallel to
`ligands["name"]`) plus a `uma_meta` block of provenance + per-split match
stats.

Build the **train and val** splits (writes `train.with_uma.pt` and
`val.with_uma.pt` next to the inputs). The **test** split is intentionally
left plain — it goes through `model.sample()` and never computes the REPA
loss, so it needs no embeddings:

```bash
python scripts/python/uma_embeddings/build_uma_complex_id_mapping.py \
  --splits processed_crossdocked/train.pt processed_crossdocked/val.pt \
  --embeddings-dir /mnt/datasets/CrossDocked/embeddings_hydrogens_uma_s_depth_2
```

Flags:
- `--in-place` — overwrite each input split instead of writing a
  `<stem>.with_uma.pt` companion (atomic via tmp + rename)
- `--verify-sample N` — random ligands per split to fully load + verify by
  coords (default 200); the rest take a fast path that derives `complex_id`
  from the file path. Stem collisions are always verified.
- `--distance-tol` — NN tolerance in Å (default `1e-4`; real matches are at
  float32 noise ~1e-6)
- `--limit N` — cap examples per split (for testing); default is full split
- `--seed` — seed for the verification sampler (default 42)
- `--overwrite` — required to clobber an existing `<stem>.with_uma.pt`

The augmented splits are consumed at training time via
`train_params.dataset_suffix: '.with_uma'`, which makes the loader read
`{stage}.with_uma.pt` for the train and val splits (test always loads the
plain `test.pt`; see `_split_pt_path` / `_uses_dataset_suffix` in
`src/model/lightning.py`).

## 2. Verify atom-order alignment for a specific source

Atom-level REPA is index-by-index, so the k-th UMA ligand atom must be
the k-th DrugFlow ligand atom. Different UMA sources can differ here
(presence/absence of hydrogens, atom ordering quirks). Run once per
source you want to train against. The train split alone is enough —
alignment is a property of the embedding source and processing pipeline,
not the split, so val/test (same source) need no separate pass.

It reads each ligand's `complex_id` from the split (added by step 1, so
pass `train.with_uma.pt`) and resolves the UMA file at
`<embeddings-dir>/<complex_id>.pt`.

```bash
python scripts/python/uma_embeddings/verify_uma_atom_order.py \
  --splits processed_crossdocked/train.with_uma.pt \
  --embeddings-dir /mnt/datasets/CrossDocked/embeddings_hydrogens_uma_s_depth_2 \
  --n-check 200
```

Classifies each checked ligand into:
- `in_order` — atoms match index-by-index within tol ✓
- `permuted_same_atoms` — same atoms, different order
- `count_mismatch` — different atom counts
- `unaligned` — neither in-order nor a clean permutation
- `missing_uma` — no complex_id / no file for this ligand

Exits with code 1 if any `count_mismatch` / `permuted` / `unaligned`
cases are seen (so it's usable as a CI/sanity check).

## UMA `.pt` schema (for reference)

```
x                      shape (N_total, 3)     — coordinates
atom_embeddings        shape (N_total, K, C)  — per-atom UMA features
num_pocket_atoms       int  — heavy-atom count (alias for num_pocket_heavy_atoms)
num_ligand_atoms       int  — heavy-atom count
num_pocket_h_atoms     int  — pocket hydrogen count
num_ligand_h_atoms     int  — ligand hydrogen count
```

Row layout of `x` and `atom_embeddings`:

```
[0                : n_pocket          )   pocket heavy atoms
[n_pocket         : n_pocket+n_ligand )   ligand heavy atoms   ← what we use
[n_pocket+n_ligand : ... + n_pocket_h )   pocket H
[...                                   )   ligand H
```
