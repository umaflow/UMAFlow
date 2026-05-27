"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Resumable, prefetch-pipelined UMA embedding extractor for CrossDocked-like
protein-ligand pairs.

  1. Hydrogens are added before running UMA, with RDKit only:
       - Pocket: MolFromPDBFile + AddHs(addCoords=True). Strict-sanitize
         path first; on failure, fall back to a relaxed parse that skips
         the valence-property check. The fallback is logged per complex.
       - Ligand: MolFromMolFile + AddHs(addCoords=True). Same
         strict -> relaxed cascade for OpenBabel-broken SDFs.
     The combined system is laid out as
         [pocket_heavy, ligand_heavy, pocket_H, ligand_H].
     UMA only consumes element symbols and 3D positions, so as long as
     RDKit produces sensible heavy + H coordinates, bond-order quirks
     in the source files do not affect the embedding.
  2. The hooked block index is a CLI argument (--block-idx), and the
     block index + depth + protonation flags are recorded in
     metadata.json so downstream consumers cannot mix incompatible
     datasets.
  3. A small ThreadPoolExecutor pre-builds the next N complexes'
     AtomicData while the current one is on the model. On CUDA this
     hides ~all I/O.
"""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.io import read
from rdkit import Chem
from rdkit import RDLogger

from fairchem.core.calculate.pretrained_mlip import get_predict_unit
from fairchem.core.datasets.atomic_data import AtomicData

# RDKit logs lots of false-positive warnings; silence to keep logs clean.
RDLogger.DisableLog("rdApp.*")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(_h)

MOLECULE_CELL_SIZE = 120.0
RADIUS = 6.0
MAX_NEIGH = 300


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Extract UMA embeddings for protein-ligand complexes with hydrogens "
            "added by RDKit. Configurable hooked block. Resumable."
        )
    )
    p.add_argument("--uma-variant", required=True,
                   help="e.g. uma-s-1p1, uma-m-1p1")
    p.add_argument("--data-dir", required=True,
                   help="CrossDocked pocket10 root directory.")
    p.add_argument("--output-dir", default="embeddings_with_h",
                   help="Output directory; one .pt per complex; existing files skipped.")
    p.add_argument("--device", default="cpu", help="cpu or cuda")
    p.add_argument("--block-idx", type=int, required=True,
                   help="0-indexed backbone block to hook; raw pre-norm output.")
    p.add_argument("--num-prefetch", type=int, default=2,
                   help="Number of complexes to pre-build while the GPU is busy.")
    p.add_argument("--limit", type=int, default=None,
                   help="Optional cap on number of complexes (for debugging).")
    p.add_argument("--num-shards", type=int, default=1,
                   help="Total number of parallel shards. Each shard processes "
                        "complexes[shard_id::num_shards]. Per-.pt writes are "
                        "naturally collision-free, so all shards may share "
                        "--output-dir.")
    p.add_argument("--shard-id", type=int, default=0,
                   help="0-indexed id of THIS shard, in [0, num_shards).")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Hydrogen addition (RDKit only)
# --------------------------------------------------------------------------- #
@dataclass
class _Protonated:
    heavy_syms: list[str]
    heavy_xyz: np.ndarray   # [n_heavy, 3]
    h_syms: list[str]
    h_xyz: np.ndarray       # [n_h, 3]


def _split_atoms(mh: Chem.Mol) -> _Protonated:
    """
    Walk an RDKit Mol with a 3D conformer, splitting atoms into heavy / H lists.
    """
    heavy_syms, heavy_xyz, h_syms, h_xyz = [], [], [], []
    conf = mh.GetConformer()
    for i, atom in enumerate(mh.GetAtoms()):
        sym = atom.GetSymbol()
        p = conf.GetAtomPosition(i)
        xyz = (p.x, p.y, p.z)
        if sym == "H":
            h_syms.append(sym)
            h_xyz.append(xyz)
        else:
            heavy_syms.append(sym)
            heavy_xyz.append(xyz)
    return _Protonated(
        heavy_syms,
        np.asarray(heavy_xyz, dtype=np.float64),
        h_syms,
        np.asarray(h_xyz, dtype=np.float64) if h_xyz else np.zeros((0, 3)),
    )


def _strict_parse(path: str, *, is_sdf: bool) -> Chem.Mol | None:
    """
    Strict RDKit parse with full sanitization. May return None or raise on
    pathological inputs.
    """
    if is_sdf:
        return Chem.MolFromMolFile(path, sanitize=True, removeHs=True)
    return Chem.MolFromPDBFile(path, sanitize=True, removeHs=True)


def _relaxed_parse(path: str, *, is_sdf: bool) -> Chem.Mol:
    """
    Relaxed RDKit parse: skip sanitization on read, then run a partial
    sanitize that explicitly skips the valence-property check (which is
    what trips on OpenBabel 2.4 SDFs and on weird residues / non-standard
    bonds in PDBs). Raises only if the file truly can't be parsed.
    """
    if is_sdf:
        m = Chem.MolFromMolFile(path, sanitize=False, removeHs=True)
    else:
        m = Chem.MolFromPDBFile(path, sanitize=False, removeHs=True)
    if m is None:
        raise RuntimeError(f"RDKit could not parse {path} even with sanitize=False")
    try:
        Chem.SanitizeMol(
            m,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
            ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
        )
    except Exception:
        # Even partial sanitize can fail on truly broken inputs; AddHs may
        # still work because it only needs valence info to know how many
        # implicit Hs to place — broken atoms simply get fewer Hs.
        pass
    return m


def _protonate(path: str, *, is_sdf: bool) -> tuple[_Protonated, str, str | None]:
    """
    Add hydrogens with RDKit. Returns (Protonated, method, fallback_reason).

    method is one of:
      - "rdkit"                   -> strict-sanitize parse + AddHs worked
      - "rdkit_no_valence_check"  -> strict path failed; recovered with the
                                     relaxed (no-valence-check) sanitize.
                                     Means a small number of atoms with
                                     broken bond orders may get fewer Hs
                                     than they should — localized minor
                                     fidelity loss. Logged in audit TSV.

    Raises only when both paths fail.
    """
    # Strict path. We catch broadly because RDKit signals problems both by
    # returning None *and* by raising AtomValenceException / AtomKekulizeException
    # depending on the exact failure mode; either way we want to fall back.
    try:
        m = _strict_parse(path, is_sdf=is_sdf)
        if m is not None:
            return _split_atoms(Chem.AddHs(m, addCoords=True)), "rdkit", None
        strict_reason = "strict_parse_returned_None"
    except Exception as e:
        strict_reason = (
            f"strict_failed_{type(e).__name__}_{str(e)[:200]}"
            .replace("\t", " ")
            .replace("\n", " ")
        )

    # Relaxed fallback.
    m = _relaxed_parse(path, is_sdf=is_sdf)
    try:
        mh = Chem.AddHs(m, addCoords=True)
    except Exception as e:
        raise RuntimeError(
            f"AddHs failed on relaxed parse of {path}: "
            f"{type(e).__name__}: {e}; strict path: {strict_reason}"
        ) from e
    return _split_atoms(mh), "rdkit_no_valence_check", strict_reason


# --------------------------------------------------------------------------- #
# Discovery and per-complex preparation
# --------------------------------------------------------------------------- #
def find_complexes(data_dir: str) -> list[dict]:
    out = []
    root = Path(data_dir)
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        for sdf in sorted(sub.glob("*.sdf")):
            pdb = sub / f"{sdf.stem}_pocket10.pdb"
            if not pdb.exists():
                logger.warning(f"No pocket file for {sdf}, skipping")
                continue
            out.append({
                "complex_id": f"{sub.name}/{sdf.stem}",
                "ligand_path": str(sdf),
                "pocket_path": str(pdb),
            })
    return out


def output_path_for(output_dir: Path, complex_id: str) -> Path:
    sub, prefix = complex_id.split("/", 1)
    p = output_dir / sub / f"{prefix}.pt"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


@dataclass
class Prepared:
    """
    Everything needed for the GPU-side forward pass + save, packed up by
    the prefetch workers so the main thread never blocks on I/O.
    """
    complex_id: str
    out_path: Path
    atoms: Atoms
    n_pocket_heavy: int
    n_ligand_heavy: int
    n_pocket_h: int
    n_ligand_h: int
    pocket_h_method: str             # "rdkit" or "rdkit_no_valence_check"
    pocket_ase_heavy: int            # informational; over-counts altLoc
    pocket_fallback_reason: str | None
    ligand_h_method: str             # "rdkit" or "rdkit_no_valence_check"
    ligand_ase_heavy: int            # informational
    ligand_fallback_reason: str | None


def _prepare(complex_dict: dict, output_dir: Path) -> Prepared | None:
    """
    Read pocket+ligand, add hydrogens with RDKit, build a single ASE
    Atoms in the canonical layout: [pocket_heavy, ligand_heavy,
    pocket_H, ligand_H].

    Returns None if the output file already exists (resume path).
    Returns Prepared on success; raises on parse errors so the caller
    can record a per-complex failure without crashing the whole job.
    """
    cid = complex_dict["complex_id"]
    out = output_path_for(output_dir, cid)
    if out.exists():
        return None

    # ASE counts are kept purely for the audit log. ASE's PDB reader
    # over-counts altLoc indicators, so we do NOT use it to gate success.
    n_pH_ase = len(read(complex_dict["pocket_path"]))
    n_lH_ase = len(read(complex_dict["ligand_path"]))

    p, pocket_h_method, pocket_reason = _protonate(
        complex_dict["pocket_path"], is_sdf=False,
    )
    l, ligand_h_method, ligand_reason = _protonate(
        complex_dict["ligand_path"], is_sdf=True,
    )

    syms = p.heavy_syms + l.heavy_syms + p.h_syms + l.h_syms
    xyz = np.vstack([p.heavy_xyz, l.heavy_xyz, p.h_xyz, l.h_xyz])
    atoms = Atoms(symbols=syms, positions=xyz)
    atoms.info["charge"] = 0
    atoms.info["spin"] = 1

    return Prepared(
        complex_id=cid,
        out_path=out,
        atoms=atoms,
        n_pocket_heavy=len(p.heavy_syms),
        n_ligand_heavy=len(l.heavy_syms),
        n_pocket_h=len(p.h_syms),
        n_ligand_h=len(l.h_syms),
        pocket_h_method=pocket_h_method,
        pocket_ase_heavy=n_pH_ase,
        pocket_fallback_reason=pocket_reason,
        ligand_h_method=ligand_h_method,
        ligand_ase_heavy=n_lH_ase,
        ligand_fallback_reason=ligand_reason,
    )


# --------------------------------------------------------------------------- #
# Forward pass
# --------------------------------------------------------------------------- #
def make_atomic_data(atoms: Atoms, device: str) -> AtomicData:
    d = AtomicData.from_ase(
        atoms,
        task_name="omol",
        r_edges=True,
        r_data_keys=["spin", "charge"],
        molecule_cell_size=MOLECULE_CELL_SIZE,
        radius=RADIUS,
        max_neigh=MAX_NEIGH,
        target_dtype=torch.float32,
    )
    return d.to(device)


def compute_invariant_graph_embedding(x: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    Per-degree invariant pooling over a per-atom equivariant tensor
    [n_atoms, (lmax+1)^2, sphere_channels]. lmax inferred from shape.
    """
    n_sh = x.shape[1]
    lmax = round(n_sh ** 0.5) - 1
    parts: dict[str, torch.Tensor] = {"l0": x[:, 0, :].mean(dim=0)}
    idx = 1
    for L in range(1, lmax + 1):
        n = 2 * L + 1
        parts[f"l{L}_norm"] = x[:, idx:idx + n, :].norm(dim=1).mean(dim=0)
        idx += n
    parts["combined"] = torch.cat(list(parts.values()), dim=-1)
    return parts


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.num_shards < 1 or not (0 <= args.shard_id < args.num_shards):
        raise ValueError(
            f"Invalid sharding: shard_id={args.shard_id}, num_shards={args.num_shards}"
        )

    all_complexes = find_complexes(args.data_dir)
    if args.limit:
        all_complexes = all_complexes[: args.limit]
    # Round-robin shard slice. Stable across runs because find_complexes
    # already sorts. Round-robin (instead of contiguous chunks) tends to
    # mix small/large complexes evenly so wall-clock per shard is similar.
    complexes = all_complexes[args.shard_id :: args.num_shards]
    if not complexes:
        logger.error("No complexes found; exiting.")
        return
    logger.info(
        f"Shard {args.shard_id}/{args.num_shards}: "
        f"{len(complexes)} complexes (of {len(all_complexes)} total)."
    )

    logger.info(f"Loading UMA model: {args.uma_variant}")
    pu = get_predict_unit(args.uma_variant, inference_settings="default", device=args.device)
    backbone = pu.model.module.backbone
    num_layers = backbone.num_layers
    lmax = backbone.lmax
    sphere_channels = backbone.sphere_channels

    if args.block_idx < 0 or args.block_idx >= num_layers:
        raise ValueError(
            f"--block-idx {args.block_idx} out of range for {num_layers}-layer backbone."
        )
    extracted_depth = args.block_idx + 1
    logger.info(
        f"Backbone: {num_layers} layers, lmax={lmax}, sphere_channels={sphere_channels}. "
        f"Hooking block {args.block_idx} (depth {extracted_depth}, raw pre-norm)."
    )

    metadata = {
        "uma_variant": args.uma_variant,
        "lmax": lmax,
        "sphere_channels": sphere_channels,
        "num_layers": num_layers,
        "extracted_depth": extracted_depth,
        "extracted_block_index": args.block_idx,
        "extracted_depth_description": (
            f"raw pre-norm output of backbone.blocks[{args.block_idx}]"
        ),
        "hydrogens_added": True,
        "pocket_protonation": (
            "rdkit MolFromPDBFile + AddHs(addCoords=True); strict-sanitize "
            "first, relaxed (no valence check) fallback on failure "
            "(see pocket_h_audit.tsv)"
        ),
        "ligand_protonation": (
            "rdkit MolFromMolFile + AddHs(addCoords=True) using SDF formal "
            "valences/charges; relaxed-sanitizer fallback for "
            "OpenBabel-broken SDFs (see ligand_h_audit.tsv); no pH-aware "
            "microstate prediction"
        ),
        "atom_layout": "[pocket_heavy, ligand_heavy, pocket_H, ligand_H]",
        "invariant_keys": (
            ["l0"] + [f"l{L}_norm" for L in range(1, lmax + 1)] + ["combined"]
        ),
        "invariant_pooling_scope": "heavy atoms only (H atoms excluded from mean-pool)",
        "molecule_cell_size": MOLECULE_CELL_SIZE,
        "radius": RADIUS,
        "max_neigh": MAX_NEIGH,
    }
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Wrote {meta_path}")

    # Hook the chosen block; raw pre-norm output.
    captured: dict[str, torch.Tensor] = {}

    def _hook(module, inp, out):
        captured["x"] = out.detach().clone()

    handle = backbone.blocks[args.block_idx].register_forward_hook(_hook)

    done = skipped = failed = 0
    pocket_method_counts: dict[str, int] = {}
    ligand_method_counts: dict[str, int] = {}

    # Audit TSVs: one row per complex that did NOT use the strict path.
    # Append + flush so resumes preserve history and partial runs are
    # still inspectable. Tab-separated so `cut -f` / `awk` work cleanly.
    # Per-shard filenames: avoid concurrent appends from multiple shard
    # processes corrupting a shared TSV. After the run, just `cat` them
    # together (`cat pocket_h_audit.shard_*.tsv`).
    shard_suffix = (
        f".shard_{args.shard_id:02d}_of_{args.num_shards:02d}"
        if args.num_shards > 1 else ""
    )
    audit_path = output_dir / f"pocket_h_audit{shard_suffix}.tsv"
    audit_existing = audit_path.exists()
    audit_f = open(audit_path, "a")
    if not audit_existing:
        audit_f.write("complex_id\tmethod\tase_heavy\tused_heavy\treason\n")
        audit_f.flush()
    logger.info(f"Pocket audit log: {audit_path}")

    lig_audit_path = output_dir / f"ligand_h_audit{shard_suffix}.tsv"
    lig_audit_existing = lig_audit_path.exists()
    lig_audit_f = open(lig_audit_path, "a")
    if not lig_audit_existing:
        lig_audit_f.write("complex_id\tmethod\tase_heavy\tused_heavy\treason\n")
        lig_audit_f.flush()
    logger.info(f"Ligand audit log: {lig_audit_path}")

    pool = ThreadPoolExecutor(max_workers=max(1, args.num_prefetch))
    in_flight: list = []  # list[(complex_dict, Future[Prepared|None])]
    next_idx = 0

    def submit_next():
        nonlocal next_idx
        while len(in_flight) < max(1, args.num_prefetch) and next_idx < len(complexes):
            in_flight.append(
                (complexes[next_idx], pool.submit(_prepare, complexes[next_idx], output_dir))
            )
            next_idx += 1

    submit_next()
    processed = 0
    try:
        while in_flight:
            cplx, fut = in_flight.pop(0)
            submit_next()
            processed += 1
            cid = cplx["complex_id"]
            try:
                prep = fut.result()
            except Exception as e:
                logger.error(f"[{processed}/{len(complexes)}] PREP FAIL {cid}: {e}")
                failed += 1
                continue

            if prep is None:
                skipped += 1
                if skipped % 1000 == 0:
                    logger.info(f"Skipped {skipped} already-extracted so far.")
                continue

            try:
                data = make_atomic_data(prep.atoms, args.device)
                captured.clear()
                pu.predict(data)
                emb = captured["x"].detach().cpu()  # [N, n_sh, C]

                n_pH = prep.n_pocket_heavy
                n_lH = prep.n_ligand_heavy
                n_heavy = n_pH + n_lH
                ligand_heavy_emb = emb[n_pH:n_heavy]

                torch.save(
                    {
                        "complex_id": prep.complex_id,
                        "x": prep.atoms.get_positions().copy(),
                        "num_pocket_heavy_atoms": n_pH,
                        "num_ligand_heavy_atoms": n_lH,
                        "num_pocket_h_atoms": prep.n_pocket_h,
                        "num_ligand_h_atoms": prep.n_ligand_h,
                        "pocket_h_method": prep.pocket_h_method,
                        "ligand_h_method": prep.ligand_h_method,
                        # Backwards-compat aliases for downstream code that
                        # only knows the old schema:
                        "num_pocket_atoms": n_pH,
                        "num_ligand_atoms": n_lH,
                        "atom_embeddings": emb,
                        # Graph-level invariants over heavy atoms only —
                        # H atoms would dominate a naive mean-pool by sheer count.
                        "invariant_complex_embedding":
                            compute_invariant_graph_embedding(emb[:n_heavy]),
                        "ligand_invariant_embedding":
                            compute_invariant_graph_embedding(ligand_heavy_emb),
                    },
                    prep.out_path,
                )
                done += 1
                pocket_method_counts[prep.pocket_h_method] = (
                    pocket_method_counts.get(prep.pocket_h_method, 0) + 1
                )
                ligand_method_counts[prep.ligand_h_method] = (
                    ligand_method_counts.get(prep.ligand_h_method, 0) + 1
                )

                if prep.pocket_h_method != "rdkit":
                    audit_f.write(
                        f"{prep.complex_id}\t{prep.pocket_h_method}\t"
                        f"{prep.pocket_ase_heavy}\t{prep.n_pocket_heavy}\t"
                        f"{prep.pocket_fallback_reason or ''}\n"
                    )
                    audit_f.flush()
                if prep.ligand_h_method != "rdkit":
                    lig_audit_f.write(
                        f"{prep.complex_id}\t{prep.ligand_h_method}\t"
                        f"{prep.ligand_ase_heavy}\t{prep.n_ligand_heavy}\t"
                        f"{prep.ligand_fallback_reason or ''}\n"
                    )
                    lig_audit_f.flush()
                if done % 50 == 0 or processed <= 5:
                    logger.info(
                        f"[{processed}/{len(complexes)}] {cid} "
                        f"(N_total={emb.shape[0]}, heavy={n_heavy}, "
                        f"+H={prep.n_pocket_h + prep.n_ligand_h}) "
                        f"done={done} skipped={skipped} failed={failed}"
                    )
            except Exception as e:
                logger.error(f"[{processed}/{len(complexes)}] FORWARD FAIL {cid}: {e}")
                failed += 1
                continue
    finally:
        handle.remove()
        pool.shutdown(wait=True)
        audit_f.close()
        lig_audit_f.close()

    n_pkt_strict = pocket_method_counts.get("rdkit", 0)
    n_pkt_relaxed = pocket_method_counts.get("rdkit_no_valence_check", 0)
    pkt_pct = (100.0 * n_pkt_relaxed / done) if done else 0.0
    n_lig_strict = ligand_method_counts.get("rdkit", 0)
    n_lig_relaxed = ligand_method_counts.get("rdkit_no_valence_check", 0)
    lig_pct = (100.0 * n_lig_relaxed / done) if done else 0.0

    logger.info(
        f"Done. processed={done}, skipped={skipped}, failed={failed}.\n"
        f"  pocket H breakdown: rdkit={n_pkt_strict}, "
        f"rdkit_no_valence_check={n_pkt_relaxed} ({pkt_pct:.2f}% relaxed) "
        f"-> {audit_path}\n"
        f"  ligand H breakdown: rdkit={n_lig_strict}, "
        f"rdkit_no_valence_check={n_lig_relaxed} ({lig_pct:.2f}% relaxed) "
        f"-> {lig_audit_path}\n"
        f"  Output: {output_dir}/"
    )


if __name__ == "__main__":
    main()
