#!/usr/bin/env python
"""Augment DrugFlow split files with a UMA ``complex_id`` per ligand.

For each ligand in the requested DrugFlow split this script finds the
matching UMA ``.pt`` file by:
  1. Stem-based candidate lookup (mirrors the dataset's name -> stem rule).
  2. One-to-one nearest-neighbour coordinate matching of heavy atoms
     within ``--distance-tol``.

It then writes the split back with two added fields, in place or to a
``<stem>.with_uma.pt`` companion file::

    data["ligands"]["complex_id"]   # list[str | None], parallel to data["ligands"]["name"]
    data["uma_meta"]                # provenance + per-split match stats

Positional alignment matters: DrugFlow's ``train.pt`` contains rows
where the same ``ligand_name`` appears for *different* coordinate sets
(60 such names in the canonical split). A name-keyed mapping would
silently collapse those; the positional list does not.

Performance: unambiguous stems use a fast path that derives
``complex_id`` from the file path without loading the UMA ``.pt`` (~10
MB each). Only stem collisions and a sampled subset of ligands are
loaded for coordinate verification (controls: ``--verify-sample``).

Example::

    python scripts/python/uma_embeddings/build_uma_complex_id_mapping.py \\
        --splits processed_crossdocked/train.pt \\
        --embeddings-dir /mnt/datasets/CrossDocked/embeddings_hydrogens_uma_m_depth_2

Writes ``processed_crossdocked/train.with_uma.pt`` by default. Add
``--in-place`` to overwrite ``train.pt`` itself (atomic via tmp +
rename).
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


def ligand_name_to_stem(ligand_name: str) -> str:
    """Mirror ``ProcessedLigandPocketDataset._ligand_name_to_embedding_stem``."""
    return ligand_name.split("_", 1)[1].rsplit(".", 1)[0].replace("-", "_")


def build_index(embeddings_dir: Path) -> Dict[str, List[Path]]:
    """Group UMA .pt files by filename stem (= candidate complex_id)."""
    index: Dict[str, List[Path]] = defaultdict(list)
    for path in sorted(embeddings_dir.rglob("*.pt")):
        index[path.stem].append(path)
    return index


def path_to_complex_id(path: Path, embeddings_dir: Path) -> str:
    """Derive ``complex_id`` from the file path.

    The UMA per-complex files store ``complex_id`` equal to their relative
    path under ``embeddings_dir`` with the ``.pt`` suffix stripped (verified
    by inspecting multiple files). This lets us skip ``torch.load`` for
    unambiguous stem matches.
    """
    return path.relative_to(embeddings_dir).with_suffix("").as_posix()


def try_match_candidate(
    path: Path, ligand_x, tol: float,
) -> Tuple[Optional[str], float, int, int]:
    """Evaluate one UMA candidate against ligand_x.

    Returns ``(complex_id_or_None, max_nn_dist, n_df, n_uma)``.

    Match criterion: every DrugFlow ligand atom has a unique nearest
    neighbour in the UMA ligand block within ``tol`` Angstrom. Allows the
    UMA side to contain extra atoms (e.g. hydrogens) — only DrugFlow-side
    uniqueness is required. ``max_nn_dist`` is returned regardless of
    match outcome so callers can report distance distributions; on
    ``df_x.shape[0] > uma_x.shape[0]`` (structurally incompatible) it is
    ``inf``.

    UMA ``x`` layout (verified empirically across files):

        [0                       : n_pocket           )  pocket heavy
        [n_pocket                : n_pocket+n_ligand  )  ligand heavy   ← we slice here
        [n_pocket+n_ligand       : ... + n_pocket_h   )  pocket H
        [...                                          )  ligand H

    Note that ``num_pocket_atoms`` and ``num_ligand_atoms`` in the UMA
    schema are aliases for the *heavy*-atom counts — the per-element H
    counts are exposed separately as ``num_pocket_h_atoms`` /
    ``num_ligand_h_atoms``. The slice below therefore selects ligand
    heavies only, which is what DrugFlow's ``ligand['x']`` carries.
    """
    emb = torch.load(path, map_location="cpu", weights_only=False)
    n_pocket = int(emb["num_pocket_atoms"])
    n_ligand = int(emb["num_ligand_atoms"])
    uma_x = torch.as_tensor(emb["x"][n_pocket:n_pocket + n_ligand], dtype=torch.float64)
    df_x = torch.as_tensor(ligand_x, dtype=torch.float64)
    n_df, n_uma = int(df_x.shape[0]), int(uma_x.shape[0])

    if n_df > n_uma:
        return None, float("inf"), n_df, n_uma
    nn_dist, idx = torch.cdist(df_x, uma_x).min(dim=1)
    max_nn = float(nn_dist.max().item())
    if max_nn > tol:
        return None, max_nn, n_df, n_uma
    if len(set(idx.tolist())) != n_df:
        return None, max_nn, n_df, n_uma
    return emb["complex_id"], max_nn, n_df, n_uma


def _percentile(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    k = max(0, min(len(sorted_vals) - 1, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[k]


def _dist_summary(values: List[float]) -> Dict[str, float]:
    """min / mean / median / p95 / max of a list of floats. NaN if empty."""
    if not values:
        return {k: float("nan") for k in ("min", "mean", "median", "p95", "max")}
    s = sorted(values)
    return {
        "min": s[0],
        "mean": sum(s) / len(s),
        "median": _percentile(s, 0.50),
        "p95": _percentile(s, 0.95),
        "max": s[-1],
    }


def _fmt(x: float) -> str:
    if x != x:                # NaN
        return "n/a"
    if x == float("inf"):
        return "inf"
    return f"{x:.3e}"


def _print_split_summary(split: str, stats: Dict[str, object]) -> None:
    n_verified = stats["matched_verified"]       # type: ignore[index]
    print(f"--- {split} ---")
    print(f"  matched={stats['matched']}  "
          f"(fast={stats['matched_fast']}, "
          f"verified={n_verified})  "
          f"missing_stem={stats['missing_stem']}  "
          f"match_failed={stats['match_failed']}")

    md = stats["matched_max_nn_dist"]            # type: ignore[index]
    print(f"  verified max-NN-dist (Å, n={n_verified}):  "
          f"min={_fmt(md['min'])}  mean={_fmt(md['mean'])}  "
          f"median={_fmt(md['median'])}  p95={_fmt(md['p95'])}  "
          f"max={_fmt(md['max'])}")

    parity = stats["atom_count_parity_verified"]  # type: ignore[index]
    print(f"  atom-count parity (verified):  "
          f"|df|<|uma|={parity['df_lt_uma']}  "
          f"|df|=|uma|={parity['df_eq_uma']}  "
          f"|df|>|uma|={parity['df_gt_uma']}")

    if stats["match_failed"]:                     # type: ignore[index]
        fd = stats["failed_best_max_nn_dist"]    # type: ignore[index]
        print(f"  failed best max-NN-dist (Å):  "
              f"min={_fmt(fd['min'])}  median={_fmt(fd['median'])}  "
              f"max={_fmt(fd['max'])}")
    print()


def match_split(
    drugflow_pt: Path,
    embeddings_dir: Path,
    emb_index: Dict[str, List[Path]],
    tol: float,
    limit: Optional[int],
    verify_sample: int,
    seed: int,
) -> Tuple[dict, List[Optional[str]], Dict[str, object]]:
    """Match ligands to UMA complex_ids and return ``(data, complex_ids, stats)``.

    ``complex_ids`` is a positional list parallel to
    ``data["ligands"]["name"]`` — entry is ``None`` for ligands with no
    matching UMA file.

    Fast path (unambiguous stem): derive ``complex_id`` from the file
    path without loading the ``.pt``. Slow path (collisions + sampled
    verification): load each candidate and verify by coordinates.
    """
    print(f"Loading {drugflow_pt}")
    data = torch.load(drugflow_pt, map_location="cpu", weights_only=False)
    names = data["ligands"]["name"]
    xs = data["ligands"]["x"]

    total = len(names) if limit is None else min(limit, len(names))
    rng = random.Random(seed)
    sample_indices = set(rng.sample(range(total), k=min(verify_sample, total)))
    print(f"  verifying {len(sample_indices)} sampled ligands "
          f"(+ any with stem collisions) by coord match.")

    complex_ids: List[Optional[str]] = [None] * len(names)
    counts = {"matched": 0, "matched_fast": 0, "matched_verified": 0,
              "missing_stem": 0, "match_failed": 0}

    matched_dists: List[float] = []
    failed_best_dists: List[float] = []
    parity = {"df_lt_uma": 0, "df_eq_uma": 0, "df_gt_uma": 0}

    for i in range(total):
        stem = ligand_name_to_stem(names[i])
        candidates = emb_index.get(stem, [])
        if not candidates:
            counts["missing_stem"] += 1
            continue

        # Fast path: single candidate AND not selected for sampled verification.
        if len(candidates) == 1 and i not in sample_indices:
            complex_ids[i] = path_to_complex_id(candidates[0], embeddings_dir)
            counts["matched"] += 1
            counts["matched_fast"] += 1
        else:
            chosen: Optional[Tuple[str, float, int, int]] = None
            best_attempt_dist = float("inf")
            for candidate in candidates:
                try:
                    cid, max_nn, n_df, n_uma = try_match_candidate(
                        candidate, xs[i], tol,
                    )
                except (KeyError, RuntimeError, OSError, ValueError):
                    continue
                best_attempt_dist = min(best_attempt_dist, max_nn)
                if cid is not None:
                    chosen = (cid, max_nn, n_df, n_uma)
                    break

            if chosen is None:
                counts["match_failed"] += 1
                failed_best_dists.append(best_attempt_dist)
                continue

            cid, max_nn, n_df, n_uma = chosen
            complex_ids[i] = cid
            counts["matched"] += 1
            counts["matched_verified"] += 1
            matched_dists.append(max_nn)
            if n_df < n_uma:
                parity["df_lt_uma"] += 1
            elif n_df == n_uma:
                parity["df_eq_uma"] += 1
            else:
                parity["df_gt_uma"] += 1

        if (i + 1) % 5000 == 0 or i + 1 == total:
            print(f"  [{i + 1}/{total}] matched={counts['matched']} "
                  f"(fast={counts['matched_fast']}, "
                  f"verified={counts['matched_verified']})  "
                  f"missing_stem={counts['missing_stem']}  "
                  f"match_failed={counts['match_failed']}")

    stats: Dict[str, object] = dict(counts)
    stats["matched_max_nn_dist"] = _dist_summary(matched_dists)
    stats["failed_best_max_nn_dist"] = _dist_summary(failed_best_dists)
    stats["atom_count_parity_verified"] = parity
    return data, complex_ids, stats


def atomic_save(data: object, output: Path) -> None:
    """Save with a tmp + rename so a crash leaves the original intact."""
    tmp = output.with_name(output.name + ".tmp")
    torch.save(data, tmp)
    os.replace(tmp, output)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--splits", nargs="+", required=True,
        help="DrugFlow split .pt files to augment (e.g. train.pt val.pt test.pt).",
    )
    p.add_argument(
        "--embeddings-dir", required=True,
        help="Reference UMA embedding dir for ligand_name -> complex_id matching.",
    )
    p.add_argument(
        "--in-place", action="store_true",
        help="Overwrite each input split in place (atomic via tmp + rename). "
             "Default writes to <stem>.with_uma.pt next to each input.",
    )
    p.add_argument(
        "--distance-tol", type=float, default=1e-4,
        help="Max allowed nearest-neighbour atom distance in Angstrom.",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Optional cap on examples per split (for testing).",
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="Allow overwriting an existing output file (non-in-place case).",
    )
    p.add_argument(
        "--verify-sample", type=int, default=200,
        help="Per split, how many random ligands to fully load+verify by "
             "coords. The rest use a fast path that derives complex_id "
             "from the file path. Stem collisions are always verified. "
             "Set to 0 to disable sampling.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Seed for the verification sampler.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    splits = [Path(s) for s in args.splits]
    embeddings_dir = Path(args.embeddings_dir)

    outputs = {
        s: s if args.in_place else s.with_name(s.stem + ".with_uma" + s.suffix)
        for s in splits
    }
    if not args.in_place:
        for s, o in outputs.items():
            if o.exists() and not args.overwrite:
                sys.exit(
                    f"Refusing to overwrite existing file: {o}\n"
                    f"Pass --overwrite if you really mean to replace it."
                )

    print(f"Indexing UMA embeddings under: {embeddings_dir}")
    emb_index = build_index(embeddings_dir)
    n_files = sum(len(paths) for paths in emb_index.values())
    print(f"  {n_files} UMA files across {len(emb_index)} stems.\n")

    for split_pt in splits:
        data, complex_ids, stats = match_split(
            split_pt, embeddings_dir, emb_index,
            args.distance_tol, args.limit,
            args.verify_sample, args.seed,
        )
        _print_split_summary(split_pt.stem, stats)

        data["ligands"]["complex_id"] = complex_ids
        data["uma_meta"] = {
            "reference_embeddings_dir": str(embeddings_dir),
            "distance_tol": args.distance_tol,
            "verify_sample": args.verify_sample,
            "seed": args.seed,
            "stats": stats,
        }

        out = outputs[split_pt]
        print(f"Writing {out} ...")
        atomic_save(data, out)
        print(f"  done.\n")


if __name__ == "__main__":
    main()
