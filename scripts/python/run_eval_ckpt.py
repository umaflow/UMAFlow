#!/usr/bin/env python
"""Sample, evaluate, and postprocess a single checkpoint.

Stages (run in order):
    sample      -> src/sample_and_evaluate.py             (GPU)
    eval        -> scripts/python/evaluate_baselines.py   (--n-jobs parallel workers)
    postprocess -> scripts/python/postprocess_metrics.py

Output layout (per checkpoint):
    <experiment-base>/<run>__<ckpt>[_<tag>]/config.yml
    <experiment-base>/<run>__<ckpt>[_<tag>]/<set>/<ckpt>_T=<n>/   (samples + metrics)
    <experiment-base>/<run>__<ckpt>[_<tag>]/logs/

Example:
    python scripts/local/run_eval_ckpt.py path/to/epoch_0099.ckpt \\
        --set val --n-sampling-steps 50 --exclude gnina interactions posebusters
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import yaml

PROJECT_DIR = Path(__file__).resolve().parents[2]
SAMPLE_SCRIPT = PROJECT_DIR / "src" / "sample_and_evaluate.py"
EVAL_SCRIPT = PROJECT_DIR / "scripts" / "python" / "evaluate_baselines.py"
POSTPROCESS_SCRIPT = PROJECT_DIR / "scripts" / "python" / "postprocess_metrics.py"


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def run_name_from_ckpt(ckpt: Path) -> str:
    """Return the run directory name (parent of `checkpoints/`), or 'ckpt'."""
    for parent in ckpt.parents:
        if parent.name == "checkpoints":
            return parent.parent.name
    return "ckpt"


def resolve_samples_dir(samples_parent: Path):
    """The samples dir is <parent>/<ckpt>_T=<n>; its suffix depends on the
    checkpoint's simulation_params, so glob for it."""
    subdirs = sorted(d for d in samples_parent.glob("*") if d.is_dir())
    return subdirs[0] if subdirs else None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--mode", choices=["sample", "eval", "postprocess", "all"], default="all")
    p.add_argument("--set", dest="split", choices=["val", "test"], default="test")
    p.add_argument("--n-samples", type=int, default=1, help="samples per pocket")
    p.add_argument("--n-sampling-steps", type=int, default=500)
    p.add_argument("--eval-batch", type=int, default=10, help="pockets per dataloader batch")
    p.add_argument("--n-jobs", type=int, default=8, help="parallel workers for the eval stage")
    p.add_argument("--no-ema", action="store_true", help="use raw weights instead of EMA")
    p.add_argument("--exclude", nargs="*", default=["gnina", "interactions", "posebusters"],
                   help="evaluator IDs to skip, e.g. --exclude gnina interactions posebusters")
    p.add_argument("--datadir", default="./processed_crossdocked")
    p.add_argument("--dataset-suffix", default="", help="loads test<suffix>.pt")
    p.add_argument("--experiment-base", default="experiments/eval_ckpt")
    p.add_argument("--tag-suffix", default="", help="extra disambiguator on the output dir")
    p.add_argument("--device", default="cuda")
    p.add_argument("--gnina", default="gnina", help="gnina binary (on PATH)")
    p.add_argument("--reduce", default="reduce", help="reduce binary (on PATH)")
    p.add_argument("--reference-smiles", default=None,
                   help="default: <datadir>/train_smiles.npy")
    p.add_argument("--crossdocked-dir", default=None,
                   help="default: <datadir>/postprocessed_metrics_new")
    p.add_argument("--python", default=sys.executable,
                   help="Python interpreter for sub-stages (default: this interpreter)")
    return p.parse_args()


def write_config(path: Path, args: argparse.Namespace, ckpt: Path, out_dir: Path) -> None:
    config = {
        "checkpoint": str(ckpt),
        "set": args.split,
        "sample_outdir": str(out_dir),
        "n_samples": args.n_samples,
        "sample_with_ground_truth_size": False,
        "device": args.device,
        "seed": 42,
        "sample": True,
        "postprocess": False,
        "evaluate": False,
        "apply_ema": not args.no_ema,
        "reduce": args.reduce,
        "model_args": {
            "virtual_nodes": [0, 5],
            "train_params": {
                "datadir": args.datadir,
                "dataset_suffix": args.dataset_suffix,
                "gnina": args.gnina,
            },
            "eval_params": {
                "n_sampling_steps": args.n_sampling_steps,
                "eval_batch_size": args.eval_batch,
            },
        },
    }
    with open(path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False, default_flow_style=False)


def run_eval_workers(args, py, sdir, reference_smiles, log_dir, timestamp) -> int:
    """Launch n_jobs eval workers in parallel; return the number that failed."""
    exclude_opt = ["--exclude", *args.exclude] if args.exclude else []
    procs = []
    for j in range(args.n_jobs):
        jlog = open(log_dir / f"eval_job{j}_{timestamp}.log", "w")
        cmd = [
            py, "-u", str(EVAL_SCRIPT),
            "--in_dir", str(sdir), "--out_dir", str(sdir),
            "--reference_smiles", str(reference_smiles),
            "--n_samples", str(args.n_samples),
            "--job_id", str(j), "--n_jobs", str(args.n_jobs),
            *exclude_opt,
        ]
        procs.append((subprocess.Popen(cmd, stdout=jlog, stderr=subprocess.STDOUT,
                                       cwd=PROJECT_DIR), jlog))
    failed = 0
    for proc, jlog in procs:
        rc = proc.wait()
        jlog.close()
        failed += int(rc != 0)
    return failed


def main() -> None:
    args = parse_args()

    ckpt = args.checkpoint.resolve()
    if not ckpt.is_file():
        sys.exit(f"ERROR: checkpoint not found: {ckpt}")

    datadir = args.datadir
    reference_smiles = args.reference_smiles or f"{datadir}/train_smiles.npy"
    crossdocked_dir = args.crossdocked_dir or f"{datadir}/postprocessed_metrics_new"

    run_name = run_name_from_ckpt(ckpt)
    tag = f"{run_name}__{ckpt.stem.replace('=', '-')}"
    if args.tag_suffix:
        tag += f"_{args.tag_suffix}"

    out_dir = PROJECT_DIR / args.experiment_base / tag
    log_dir = out_dir / "logs"
    config_file = out_dir / "config.yml"
    samples_parent = out_dir / args.split
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)

    log(f"Mode       : {args.mode}")
    log(f"Checkpoint : {ckpt}")
    log(f"Datadir    : {datadir} (test{args.dataset_suffix}.pt)")
    log(f"Output     : {out_dir}")
    log(f"n_samples={args.n_samples}, n_steps={args.n_sampling_steps}, "
        f"eval_batch={args.eval_batch}, n_jobs={args.n_jobs}, set={args.split}, "
        f"apply_ema={not args.no_ema}")

    write_config(config_file, args, ckpt, out_dir)
    log(f"Config     : {config_file}")

    py = args.python

    # ----- SAMPLE -----
    if args.mode in ("sample", "all"):
        log("Sampling ...")
        subprocess.run([py, "-u", str(SAMPLE_SCRIPT), "--config", str(config_file)],
                       cwd=PROJECT_DIR, check=True)
        log("Sampling done.")

    sdir = resolve_samples_dir(samples_parent)
    if args.mode != "sample":
        if sdir is None:
            sys.exit(f"ERROR: no samples dir under {samples_parent}/ (run mode 'sample' first)")
        log(f"Samples dir: {sdir}")

    # ----- EVAL -----
    if args.mode in ("eval", "all"):
        log(f"Evaluating with {args.n_jobs} parallel worker(s)")
        failed = run_eval_workers(args, py, sdir, reference_smiles, log_dir, timestamp)
        log(f"Eval done. Failures: {failed}/{args.n_jobs} (per-worker logs in {log_dir}/)")
        if failed:
            sys.exit("ERROR: eval workers failed; not postprocessing.")

    # ----- POSTPROCESS -----
    if args.mode in ("postprocess", "all"):
        log("Postprocessing ...")
        subprocess.run([
            py, "-u", str(POSTPROCESS_SCRIPT),
            "--in_dir", str(sdir), "--out_dir", str(sdir),
            "--reference_smiles", str(reference_smiles),
            "--crossdocked_dir", str(crossdocked_dir),
        ], cwd=PROJECT_DIR, check=True)
        log("Postprocess done.")

    log(f"All done. Outputs in: {sdir or samples_parent}")


if __name__ == "__main__":
    main()
