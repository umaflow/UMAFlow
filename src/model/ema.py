"""Exponential Moving Average (EMA) of model weights as a PyTorch Lightning callback.

Usage: add an `ema` section to the training YAML config, e.g.:

    ema:
      decay: 0.9999
      start_step: 0            # optional, defaults to 0
      update_every_n_steps: 1  # optional, defaults to 1
      include_buffers: False   # optional, defaults to False

During validation the EMA shadow weights are swapped into the LightningModule
so validation metrics reflect the EMA model, then the live weights are restored
at ``on_validation_end`` -- *before* any ``ModelCheckpoint`` runs (PyTorch
Lightning reorders Checkpoint callbacks to run last among callbacks). As a
result **every saved checkpoint contains the live training weights**, while the
EMA weights live only in this callback's state dict (the ``shadow``). Resuming
therefore continues from the true training trajectory, and the EMA shadow
travels alongside it.

To evaluate or deploy the EMA model, load a checkpoint and swap the shadow into
the module -- see ``apply_ema`` in ``src/sample_and_evaluate.py`` and
``apply_ema_shadow`` in ``scripts/python/run_validate_checkpoint.py``, which
both reconstruct a callback from the saved ``shadow`` and call ``_swap_in``.
"""

from typing import Optional

import torch
import pytorch_lightning as pl


class EMACallback(pl.Callback):
    """Maintains an exponential moving average of module parameters.

    The shadow is updated after every optimizer step (honouring gradient
    accumulation via ``trainer.global_step``). During validation the shadow is
    loaded into the module so validation metrics reflect the EMA model, and the
    live weights are restored at ``on_validation_end`` before any
    ``ModelCheckpoint`` runs. Saved checkpoints therefore always contain the
    live weights; the EMA weights are kept only in this callback's state dict.
    """

    def __init__(
        self,
        decay: float,
        start_step: int = 0,
        update_every_n_steps: int = 1,
        include_buffers: bool = False,
    ):
        super().__init__()
        assert 0.0 <= decay < 1.0, f"decay must be in [0, 1), got {decay}"
        self.decay = float(decay)
        self.start_step = int(start_step)
        self.update_every_n_steps = max(1, int(update_every_n_steps))
        self.include_buffers = bool(include_buffers)

        self._shadow: Optional[dict] = None   # name -> tensor on model device
        self._backup: Optional[dict] = None   # live weights while EMA is active
        self._active: bool = False
        self._last_global_step: int = -1

    # ----- helpers -----------------------------------------------------------
    def _named_ema_tensors(self, pl_module: pl.LightningModule):
        for name, p in pl_module.named_parameters():
            if p.requires_grad:
                yield name, p
        if self.include_buffers:
            for name, b in pl_module.named_buffers():
                if b.dtype.is_floating_point:
                    yield name, b

    @torch.no_grad()
    def _init_shadow(self, pl_module: pl.LightningModule):
        self._shadow = {
            name: p.detach().clone()
            for name, p in self._named_ema_tensors(pl_module)
        }

    @torch.no_grad()
    def _update(self, pl_module: pl.LightningModule):
        assert not self._active, "Cannot update EMA while shadow weights are loaded"
        for name, p in self._named_ema_tensors(pl_module):
            shadow = self._shadow.get(name)
            if shadow is None:
                # new parameter appeared after init (shouldn't normally happen)
                self._shadow[name] = p.detach().clone()
                continue
            if shadow.device != p.device:
                shadow = shadow.to(p.device)
                self._shadow[name] = shadow
            shadow.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def _swap_in(self, pl_module: pl.LightningModule):
        if self._active or self._shadow is None:
            return
        self._backup = {}
        for name, p in self._named_ema_tensors(pl_module):
            shadow = self._shadow.get(name)
            if shadow is None:
                continue
            self._backup[name] = p.detach().clone()
            if shadow.device != p.device:
                shadow = shadow.to(p.device)
                self._shadow[name] = shadow
            p.data.copy_(shadow.data)
        self._active = True

    @torch.no_grad()
    def _swap_out(self, pl_module: pl.LightningModule):
        if not self._active:
            return
        for name, p in self._named_ema_tensors(pl_module):
            live = self._backup.get(name)
            if live is not None:
                p.data.copy_(live.data)
        self._backup = None
        self._active = False

    # ----- Lightning hooks ---------------------------------------------------
    def on_fit_start(self, trainer, pl_module):
        if self._shadow is None:
            self._init_shadow(pl_module)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        gs = int(trainer.global_step)
        # global_step only increments after an actual optimizer step, so this
        # naturally skips gradient-accumulation micro-batches.
        if gs == self._last_global_step:
            return
        self._last_global_step = gs

        if gs < self.start_step:
            # warmup phase: keep shadow pinned to live params
            self._init_shadow(pl_module)
            return
        if gs % self.update_every_n_steps != 0:
            return
        self._update(pl_module)

    def on_validation_start(self, trainer, pl_module):
        # Swap shadow weights into the module so validation metrics reflect the
        # EMA model. They are restored in on_validation_end (below), before any
        # checkpoint is written.
        self._swap_in(pl_module)

    def on_validation_end(self, trainer, pl_module):
        # Restore the live weights *before* any ModelCheckpoint saves. PyTorch
        # Lightning reorders Checkpoint callbacks to run last among callbacks at
        # on_validation_end (see _CallbackConnector._reorder_callbacks), and this
        # is an ordinary ("other") callback that runs before them. So checkpoints
        # saved here -- and `last.ckpt`, saved later at on_train_epoch_end --
        # capture the live weights, not the EMA shadow. The validation metrics
        # logged above were already computed on the EMA model.
        self._swap_out(pl_module)

    def on_train_epoch_start(self, trainer, pl_module):
        # Defensive: live weights are normally already restored in
        # on_validation_end, so this is a no-op. Kept to guarantee training
        # resumes on live weights even if a validation run was interrupted
        # before its end hook fired.
        self._swap_out(pl_module)

    def on_sanity_check_end(self, trainer, pl_module):
        # Sanity check also runs on_validation_start, so make sure we leave
        # training to start with live weights even if no train epoch start
        # fires between sanity check and first validation.
        self._swap_out(pl_module)

    def on_fit_end(self, trainer, pl_module):
        # Leave the module holding live weights at the end of fit.
        self._swap_out(pl_module)

    # ----- checkpointing of callback state ----------------------------------
    def state_dict(self):
        return {
            "decay": self.decay,
            "start_step": self.start_step,
            "update_every_n_steps": self.update_every_n_steps,
            "include_buffers": self.include_buffers,
            "last_global_step": self._last_global_step,
            "shadow": self._shadow,
        }

    def load_state_dict(self, state_dict):
        self.decay = state_dict.get("decay", self.decay)
        self.start_step = state_dict.get("start_step", self.start_step)
        self.update_every_n_steps = state_dict.get(
            "update_every_n_steps", self.update_every_n_steps
        )
        self.include_buffers = state_dict.get("include_buffers", self.include_buffers)
        self._last_global_step = state_dict.get("last_global_step", -1)
        self._shadow = state_dict.get("shadow", None)
        self._backup = None
        self._active = False
