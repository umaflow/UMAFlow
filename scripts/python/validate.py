"""Offline validation: run the expensive sampling + analyze_sample pipeline
on a saved training checkpoint.

Matches the metrics that used to be logged under `<metric>/val` during
training (via `DrugFlow.run_offline_evaluation`, which shares code with
`on_validation_epoch_end`). Optionally uploads those metrics to the original
W&B run by resuming on its run id.

EMA-aware: if the checkpoint contains EMA shadow weights (stored by
`EMACallback` in the callback state dict), they are swapped into the model
before evaluation to match in-training validation behaviour. Pass
`--no_ema` to evaluate the live weights instead.

  Usage:                                                                                                                                
  python src/validate.py --checkpoint runs/<run>/checkpoints/periodic/epoch_0090.ckpt                                                   
  python src/validate.py --checkpoint ... --no_wandb --no_viz            # local only                                                 
  python src/validate.py --checkpoint ... --wandb_run_id my_run          # explicit run id            
"""

import argparse
import sys
import yaml
from pathlib import Path

import torch
import wandb


# This copy lives in scripts/local/, so the project root is two levels up.
basedir = Path(__file__).resolve().parents[2]
sys.path.append(str(basedir))

from src import utils
from src.model.lightning import DrugFlow
from src.model.dpo import DPO
from src.model.ema import EMACallback
from src.sbdd_metrics.metrics import FullEvaluator


# Binaries are looked up on PATH by default. Pass absolute paths via
# --gnina / --reduce if they live elsewhere, or 'none' to disable that metric.
DEFAULT_GNINA = "gnina"
DEFAULT_REDUCE = "reduce"


def apply_ema_shadow(ckpt, pl_module):
    """Locate EMA shadow weights in the checkpoint's callback state dict and
    swap them into `pl_module`. Returns True if a shadow was found and applied.
    """
    cb_states = ckpt.get('callbacks', {}) or {}
    ema_key = next((k for k in cb_states if 'EMACallback' in k), None)
    if ema_key is None:
        return False
    state = cb_states[ema_key]
    if state.get('shadow') is None:
        return False
    print(f'Applying EMA shadow weights from callback state "{ema_key}"')
    dummy = EMACallback(decay=state.get('decay', 0.0))
    dummy.load_state_dict(state)
    dummy._swap_in(pl_module)
    return True


def infer_run_name(ckpt_path):
    """Walk up from the checkpoint file until the parent of a `checkpoints/`
    directory — that parent's name is the run_name used at training time
    (and therefore the W&B run id; see train.py).
    """
    current = Path(ckpt_path).resolve().parent
    while current != current.parent:
        if current.name == 'checkpoints':
            return current.parent.name
        current = current.parent
    return None


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--set', type=str, default='val', choices=['val', 'test'])
    p.add_argument('--outdir', type=str, default=None,
                   help="Directory for metrics.yaml + SDF visualizations. "
                        "Defaults to <ckpt_dir>/../offline_eval/<set>/<ckpt_stem>.")
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--n_sampling_steps', type=int, default=500,
                   help="Number of denoising steps used for sampling (overrides "
                        "eval_params.n_sampling_steps from the checkpoint).")
    p.add_argument('--n_samples', type=int, default=4,
                   help="Number of samples drawn per pocket (overrides "
                        "eval_params.n_eval_samples from the checkpoint).")
    p.add_argument('--eval_batch_size', type=int, default=None,
                   help="Override eval_params.eval_batch_size from the checkpoint "
                        "(pockets per dataloader batch). Leave unset to use the ckpt value.")
    p.add_argument('--no_ema', action='store_true',
                   help="Use live weights from the ckpt instead of the EMA shadow.")
    p.add_argument('--no_viz', action='store_true',
                   help="Skip writing molecules.sdf / pockets.sdf.")
    p.add_argument('--no_wandb', action='store_true',
                   help="Skip uploading metrics to the original W&B run.")
    p.add_argument('--wandb_run_id', type=str, default=None,
                   help="Target W&B run id. Defaults to run_name inferred from the ckpt path.")
    p.add_argument('--wandb_project', type=str, default='FlexFlow')
    p.add_argument('--wandb_entity', type=str, default=None)
    p.add_argument('--gnina', type=str, default=DEFAULT_GNINA,
                   help="Path to the gnina binary (used by the docking-score evaluator). "
                        "Pass 'none' to disable.")
    p.add_argument('--reduce', type=str, default=DEFAULT_REDUCE,
                   help="Path to the reduce binary (used by the interactions evaluator). "
                        "Pass 'none' to disable.")
    args = p.parse_args()

    utils.set_deterministic(seed=42)
    utils.disable_rdkit_logging()

    ckpt_path = Path(args.checkpoint)
    assert ckpt_path.exists(), f'Checkpoint not found: {ckpt_path}'

    ckpt = torch.load(ckpt_path, map_location='cpu')
    hp = ckpt.get('hyper_parameters', {})
    model_class = DPO if hp.get('dpo_mode', None) else DrugFlow
    print(f'Loading {model_class.__name__} from {ckpt_path}')

    pl_module = model_class.load_from_checkpoint(
        ckpt_path, map_location=args.device, strict=False
    )
    pl_module.to(args.device).eval()
    pl_module.T_sampling = args.n_sampling_steps
    pl_module.n_eval_samples = args.n_samples
    if args.eval_batch_size is not None:
        pl_module.eval_batch_size = args.eval_batch_size
    print(f'Using {pl_module.T_sampling} sampling steps, {pl_module.n_eval_samples} samples per pocket, '
          f'eval_batch_size={pl_module.eval_batch_size}')

    if not args.no_ema:
        applied = apply_ema_shadow(ckpt, pl_module)
        if not applied:
            print('No EMA shadow weights found in checkpoint; using live weights.')

    pl_module.setup(stage=args.set)

    # Rebuild the evaluator so it also runs the interactions+docking modules
    # (lightning.py's setup_metrics forwards gnina but drops reduce). Match
    # the exclude list used in setup_metrics so metrics stay comparable.
    gnina_path = None if args.gnina.lower() == 'none' else args.gnina
    reduce_path = None if args.reduce.lower() == 'none' else args.reduce
    pl_module.evaluator = FullEvaluator(
        gnina=gnina_path,
        reduce=reduce_path,
        exclude_evaluators=['geometry', 'ring_count'],
    )
    print(f'Evaluator binaries: gnina={gnina_path}, reduce={reduce_path}')

    dataloader = getattr(pl_module, f'{args.set}_dataloader')()

    if args.outdir is None:
        outdir = Path(ckpt_path.parent, '..', 'offline_eval', args.set, ckpt_path.stem).resolve()
    else:
        outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f'Writing evaluation outputs to {outdir}')

    metrics = pl_module.run_offline_evaluation(
        dataloader=dataloader,
        outdir=outdir,
        visualize=not args.no_viz,
        split=args.set,
    )

    def _to_python(v):
        if hasattr(v, 'item'):
            return float(v.item())
        try:
            return float(v)
        except (TypeError, ValueError):
            return v

    metrics_py = {k: _to_python(v) for k, v in metrics.items()}
    metrics_path = Path(outdir, 'metrics.yaml')
    with open(metrics_path, 'w') as f:
        yaml.dump(metrics_py, f, sort_keys=True)
    print(f'Wrote {metrics_path}')

    if not args.no_wandb:
        run_id = args.wandb_run_id or infer_run_name(ckpt_path)
        if run_id is None:
            print('Could not infer W&B run id from checkpoint path; '
                  'pass --wandb_run_id to upload metrics. Skipping upload.')
        else:
            epoch = int(ckpt.get('epoch', 0))
            global_step = int(ckpt.get('global_step', 0))
            prefix = args.set
            log_payload = {f'{k}/{prefix}': v for k, v in metrics_py.items()}
            log_payload['epoch'] = epoch
            log_payload['trainer/global_step'] = global_step

            print(f'Uploading to W&B run "{run_id}" at trainer/global_step={global_step}, epoch={epoch}')
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                id=run_id,
                name=run_id,
                resume='allow',
            )
            wandb.log(log_payload)
            wandb.finish()

    print('Done.')
    print(f'Metrics: {metrics_py}')
