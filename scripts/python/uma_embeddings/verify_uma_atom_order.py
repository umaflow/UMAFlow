#!/usr/bin/env python
"""Verify atom-order alignment between DrugFlow splits and a UMA embedding source.

Atom-level REPA alignment is index-by-index: the k-th row of the UMA
ligand-atom embedding must be the k-th atom of ``ligand["x"]`` in the
DrugFlow sample. If the orderings disagree, the per-atom loss is
meaningless.

This script:
  1. Loads each DrugFlow split, which carries a per-ligand ``complex_id``
     (a positional list parallel to ``ligands["name"]``) added by
     ``build_uma_complex_id_mapping.py`` — e.g. ``train.with_uma.pt``.
  2. For N complexes per split, resolves the UMA ``.pt`` directly via
     ``<embeddings-dir>/<complex_id>.pt`` and classifies the alignment::

        in_order              — coords match index-by-index (within tol)
        permuted_same_atoms   — same atoms, different order (within tol)
        count_mismatch        — different atom counts on the two sides
        unaligned             — neither in-order nor a clean permutation
        missing_uma           — no complex_id / no file for this ligand

  3. Exits with code 1 if any ``count_mismatch`` / ``permuted`` /
     ``unaligned`` cases are seen (so it's usable as a CI/sanity check).

Pocket comparison is intentionally not performed — DrugFlow uses
Cα-level pockets while UMA stores full heavy-atom pockets by design.

Example::

    python scripts/python/uma_embeddings/verify_uma_atom_order.py \\
        --splits processed_crossdocked/train.with_uma.pt \\
        --embeddings-dir /mnt/datasets/CrossDocked/embeddings_hydrogens_uma_s_depth_2
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import torch


def describe(x, label):
    if isinstance(x, torch.Tensor):
        return f"{label}: Tensor{tuple(x.shape)} {x.dtype}"
    if hasattr(x, 'shape'):
        return f"{label}: array{tuple(x.shape)} dtype={getattr(x, 'dtype', '?')}"
    if isinstance(x, (list, tuple)):
        return f"{label}: {type(x).__name__}[len={len(x)}]"
    return f"{label}: {type(x).__name__}={x!r}"


def show_drugflow_schema(data, idx=0):
    print("\n=== DrugFlow .pt schema ===")
    print("Top-level keys:", list(data.keys()))
    for entity in ('ligands', 'pockets'):
        if entity not in data:
            continue
        sub = data[entity]
        print(f"\n[{entity}] keys: {list(sub.keys())}")
        for k, v in sub.items():
            if isinstance(v, (list, tuple)) and len(v) > 0 and idx < len(v):
                print("  " + describe(v[idx], f"{k}[{idx}]"))
            else:
                print("  " + describe(v, k))


def show_uma_schema(path):
    print(f"\n=== UMA schema: {path} ===")
    data = torch.load(path, weights_only=False)
    for k, v in data.items():
        if isinstance(v, dict):
            print(f"{k}: dict")
            for kk, vv in v.items():
                print("  " + describe(vv, kk))
        else:
            print(describe(v, k))
    return data


def compare_ligand(lig_x_df, uma, tol: float) -> Dict[str, Any]:
    # UMA ``x`` layout (verified):
    #   [0 : n_pocket)                          pocket heavy
    #   [n_pocket : n_pocket+n_ligand)          ligand heavy   ← slice here
    #   [n_pocket+n_ligand : ... + n_pocket_h)  pocket H
    #   [...                                 )  ligand H
    # ``num_pocket_atoms`` / ``num_ligand_atoms`` are aliases for the
    # *heavy* counts (H counts are exposed as ``num_*_h_atoms``).
    lig_xyz_df = torch.as_tensor(lig_x_df, dtype=torch.float32)
    n_p = int(uma['num_pocket_atoms'])
    n_l_uma = int(uma['num_ligand_atoms'])
    lig_xyz_uma = torch.as_tensor(uma['x'][n_p:n_p + n_l_uma], dtype=torch.float32)

    n_l_df = lig_xyz_df.shape[0]
    res: Dict[str, Any] = {
        'n_df': n_l_df, 'n_uma': n_l_uma, 'count_match': n_l_df == n_l_uma,
    }
    if not res['count_match']:
        return res

    # In-order, raw
    d_raw = (lig_xyz_df - lig_xyz_uma).norm(dim=-1)
    res['max_err_raw'] = d_raw.max().item()
    res['in_order_raw'] = res['max_err_raw'] < tol

    # In-order, centered (absorb a global translation)
    a = lig_xyz_df - lig_xyz_df.mean(0, keepdim=True)
    b = lig_xyz_uma - lig_xyz_uma.mean(0, keepdim=True)
    d_c = (a - b).norm(dim=-1)
    res['max_err_centered'] = d_c.max().item()
    res['in_order_centered'] = res['max_err_centered'] < tol

    # Greedy nearest-neighbour match — same atoms in some permutation?
    dmat = torch.cdist(a, b)
    perm = dmat.argmin(dim=1)
    res['is_permutation'] = len(set(perm.tolist())) == n_l_df
    res['max_nn_dist_centered'] = dmat.gather(1, perm.unsqueeze(1)).max().item()
    res['perm_first10'] = perm[:10].tolist()
    return res


def classify(res, tol: float) -> str:
    if not res['count_match']:
        return 'count_mismatch'
    if res.get('in_order_raw') or res.get('in_order_centered'):
        return 'in_order'
    if res.get('is_permutation') and res.get('max_nn_dist_centered', 1) < tol:
        return 'permuted_same_atoms'
    return 'unaligned'


def verify_split(
    drugflow_pt: Path,
    embeddings_dir: Path,
    n_check: int,
    tol: float,
) -> Dict[str, int]:
    print(f"\n=== Verifying split: {drugflow_pt} (source: {embeddings_dir.name}) ===")
    data = torch.load(drugflow_pt, weights_only=False)
    if 'complex_id' not in data['ligands']:
        sys.exit(
            f"{drugflow_pt} has no 'complex_id' field — regenerate it with "
            f"build_uma_complex_id_mapping.py (e.g. train.with_uma.pt)."
        )
    names = data['ligands']['name']
    lig_xs = data['ligands']['x']
    complex_ids = data['ligands']['complex_id']

    n = min(n_check, len(names))
    print(f"Checking {n} / {len(names)} complexes...")

    buckets = {'in_order': 0, 'permuted_same_atoms': 0,
               'count_mismatch': 0, 'unaligned': 0, 'missing_uma': 0}
    worst: Dict[str, list] = {
        'count_mismatch': [], 'permuted_same_atoms': [], 'unaligned': []
    }

    for i in range(n):
        name = names[i]
        cid = complex_ids[i]
        if cid is None:
            buckets['missing_uma'] += 1
            continue
        path = embeddings_dir / f'{cid}.pt'
        if not path.exists():
            buckets['missing_uma'] += 1
            continue
        uma = torch.load(path, weights_only=False)
        res = compare_ligand(lig_xs[i], uma, tol)
        label = classify(res, tol)
        buckets[label] += 1
        if label in worst and len(worst[label]) < 3:
            worst[label].append((i, name, res))

    for k, v in buckets.items():
        print(f"  {k:25s} {v}")
    for label, cases in worst.items():
        if not cases:
            continue
        print(f"  --- Examples: {label} ---")
        for i, name, res in cases:
            print(f"    [{i}] {name}")
            for k, v in res.items():
                if isinstance(v, float):
                    print(f"      {k}: {v:.6g}")
                else:
                    print(f"      {k}: {v}")

    return buckets


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--splits', nargs='+', required=True,
                    help='DrugFlow split .pt files to verify (must carry '
                         "per-ligand 'complex_id', e.g. train.with_uma.pt).")
    ap.add_argument('--embeddings-dir', required=True,
                    help='UMA embedding source to verify against (may differ from '
                         'the dir used to build the mapping).')
    ap.add_argument('--n-check', type=int, default=50,
                    help='Per-split, how many complexes to check.')
    ap.add_argument('--tol', type=float, default=1e-3,
                    help='Coordinate tolerance for in_order / permutation match (Å).')
    ap.add_argument('--schema-only', action='store_true',
                    help='Print schemas of one file on each side and exit.')
    args = ap.parse_args()

    splits = [Path(s) for s in args.splits]
    embeddings_dir = Path(args.embeddings_dir)

    if args.schema_only:
        print(f"Loading first split for schema preview: {splits[0]}")
        data = torch.load(splits[0], weights_only=False)
        show_drugflow_schema(data, idx=0)
        sample_uma_path = next(embeddings_dir.rglob('*.pt'))
        show_uma_schema(sample_uma_path)
        return

    totals = {'in_order': 0, 'permuted_same_atoms': 0,
              'count_mismatch': 0, 'unaligned': 0, 'missing_uma': 0}
    for split_pt in splits:
        b = verify_split(split_pt, embeddings_dir, args.n_check, args.tol)
        for k, v in b.items():
            totals[k] += v

    print("\n=== TOTAL ===")
    for k, v in totals.items():
        print(f"  {k:25s} {v}")

    aligned = (
        totals['in_order'] > 0
        and totals['count_mismatch'] == 0
        and totals['permuted_same_atoms'] == 0
        and totals['unaligned'] == 0
    )
    print(f"\nAtom-order aligned: {'YES' if aligned else 'NO'}")
    sys.exit(0 if aligned else 1)


if __name__ == '__main__':
    main()
