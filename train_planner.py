"""Train Planner on top of frozen RC-aux WM.

Usage:
    python train_planner.py --config-name planner_ft data=pusht \
        planner.ckpt_path=checkpoints/rcaux_cube_object.ckpt
"""

import os
import sys

# Ensure h5py can find blosc plugin bundled with hdf5plugin
_hdf5_plugin_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    ".venv/lib/python3.10/site-packages/hdf5plugin/plugins",
)
if os.path.isdir(_hdf5_plugin_dir):
    os.environ.setdefault("HDF5_PLUGIN_PATH", _hdf5_plugin_dir)

# Project root for resolving relative paths (e.g. checkpoints/ symlink)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

import h5py as _h5py
import hydra
import lightning as pl
import lightning.pytorch.callbacks as pl_callbacks
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf

# Patch: disable SWMR mode to avoid NFS lock contention with multiple workers
_orig_open_h5_planner = swm.data.HDF5Dataset._open_h5
def _patched_open_h5_planner(self):
    if self.is_remote:
        import fsspec
        scheme = self.h5_path.split('://', 1)[0]
        fs = fsspec.filesystem(scheme, **self.storage_options)
        return _h5py.File(fs.open(self.h5_path, 'rb'), 'r')
    return _h5py.File(self.h5_path, 'r', rdcc_nbytes=256 * 1024 * 1024)
swm.data.HDF5Dataset._open_h5 = _patched_open_h5_planner

from planner import PlannerDecoder, PlannerLoss, planner_rollout
from utils import get_img_preprocessor


class GoalCache:
    """Lazy cache of episode-final goal embeddings, computed on first access."""

    def __init__(self, wm, h5_path, img_proc, offsets, lengths):
        self.wm = wm
        self.h5_path = h5_path
        self.img_proc = img_proc
        self.offsets = offsets
        self.lengths = lengths
        self._cache = {}

    def __getitem__(self, ep):
        ep = int(ep)
        if ep not in self._cache:
            with _h5py.File(self.h5_path, "r") as f:
                last_idx = int(self.offsets[ep] + self.lengths[ep] - 1)
                raw = f["pixels"][last_idx]
                t = torch.from_numpy(raw.copy()).permute(2, 0, 1)
                t = self.img_proc({"pixels": t})["pixels"]
                t = t.unsqueeze(0).unsqueeze(0).cuda()
                with torch.no_grad():
                    out = self.wm.encode({"pixels": t})
                self._cache[ep] = out["emb"][0, 0]
        return self._cache[ep]


def load_frozen_wm(ckpt_path: str):
    """Load a frozen RC-aux WM from an object checkpoint."""
    ckpt_path = os.path.expanduser(ckpt_path)
    wm = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    wm = wm.eval()
    for p in wm.parameters():
        p.requires_grad_(False)
    return wm


def planner_forward(self, batch, stage, cfg):
    """Training forward for Planner. Stage: "sft" or "ft"."""
    ctx_batch = {k: v for k, v in batch.items() if not k.startswith("goal")}
    hs = cfg.wm.history_size
    train_stage = cfg.planner.stage
    use_precomputed = "ctx_emb" in batch

    if use_precomputed:
        history_ctx = batch["ctx_emb"]
        goal_emb = batch["goal_emb"]
    else:
        with torch.no_grad():
            ctx_out = self.wm.encode(ctx_batch)
            ctx_emb = ctx_out["emb"]
        history_ctx = ctx_emb[:, :hs]
        goal_mode = cfg.planner.goal_mode
        if goal_mode == "clip_last":
            goal_emb = ctx_emb[:, -1:]
        elif goal_mode == "episode_final":
            goal_emb = self.goal_cache[batch["episode"]].unsqueeze(0).unsqueeze(0)
        else:
            raise ValueError(f"Unknown goal_mode: {goal_mode}")

    history_ctx = history_ctx.detach()
    goal_emb = goal_emb.detach()

    if train_stage == "sft":
        action = batch["action"]
        if action.dim() == 3:
            action = action[:, :hs + cfg.planner.horizon]
        pred_actions, conf = self.planner(history_ctx, goal_emb)
        loss, stats = self.planner_loss.stage_sft(pred_actions, action, history_ctx)
    elif train_stage == "ft":
        hist_actions = batch.get("hist_actions", batch["action"][:, :hs])
        pred_actions, conf = self.planner(history_ctx, goal_emb)
        info = {"pixels": ctx_batch["pixels"][:, :hs]} if "pixels" in ctx_batch else {}
        pred_embs, _ = planner_rollout(
            self.wm, pred_actions, info, history_size=hs,
            hist_actions=hist_actions, goal_emb=goal_emb,
            ctx_emb=history_ctx,
        )
        loss, stats = self.planner_loss(pred_actions, pred_embs, goal_emb, conf=conf)
    else:
        raise ValueError(f"Unknown stage: {train_stage}")

    return {"loss": loss}


@hydra.main(version_base=None, config_path="./config/train", config_name="planner_ft")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    if cfg.data.get("dataset_class"):
        dataset = hydra.utils.instantiate(cfg.data.dataset)
    else:
        dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
        dataset_name = dataset_cfg.pop("name")
        dataset_name = os.path.splitext(dataset_name)[0]
        cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
        dataset = swm.data.HDF5Dataset(
            dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
        )

    dataset.transform = get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)

    rnd_gen = torch.Generator().manual_seed(cfg.seed)

    if cfg.max_samples:
        indices = torch.randperm(len(dataset), generator=rnd_gen)[:cfg.max_samples]
        dataset = torch.utils.data.Subset(dataset, indices.tolist())

    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(
        train_set, **OmegaConf.to_container(cfg.loader, resolve=True),
        shuffle=True, drop_last=True, generator=rnd_gen,
    )
    val = torch.utils.data.DataLoader(
        val_set, **OmegaConf.to_container(cfg.loader, resolve=True),
        shuffle=False, drop_last=False,
    )

    ##############################
    ##       model / optim      ##
    ##############################

    ckpt_path = cfg.planner.ckpt_path
    if not ckpt_path or not os.path.exists(os.path.expanduser(ckpt_path)):
        raise FileNotFoundError(
            f"RC-aux checkpoint not found: {ckpt_path}\n"
            "Set planner.ckpt_path to the object checkpoint path."
        )

    wm = load_frozen_wm(ckpt_path)
    print(f"Loaded RC-aux WM from {ckpt_path}")

    act_dim = wm.action_encoder.patch_embed.in_channels
    print(f"Detected action_dim={act_dim}")

    planner = PlannerDecoder(
        embed_dim=cfg.wm.embed_dim,
        num_queries=cfg.planner.num_queries,
        horizon=cfg.planner.horizon,
        action_dim=act_dim,
        action_substeps=cfg.planner.action_substeps,
        num_layers=cfg.planner.num_layers,
        num_heads=cfg.planner.num_heads,
        mlp_dim=cfg.planner.mlp_dim,
        dropout=cfg.planner.dropout,
        action_range=cfg.planner.action_range,
    )

    planner_loss = PlannerLoss(
        diversity_weight=cfg.planner.diversity_weight,
        conf_weight=cfg.planner.conf_weight,
    )

    class PlannerLightningModule(pl.LightningModule):
        def __init__(self, wm, planner, planner_loss, cfg):
            super().__init__()
            self.automatic_optimization = False
            self.wm = wm
            self.planner = planner
            self.planner_loss = planner_loss
            self.cfg = cfg
            self.goal_cache = None

        def train(self, mode=True):
            # Lightning calls .train() at epoch start — keep WM frozen in eval
            super().train(mode)
            self.wm.eval()
            return self

        def forward(self, batch, stage):
            return planner_forward(self, batch, stage, self.cfg)

        def training_step(self, batch, batch_idx):
            output = self.forward(batch, "fit")
            loss = output["loss"]
            opt = self.optimizers()
            opt.zero_grad()
            self.manual_backward(loss)
            opt.step()
            return loss

        def validation_step(self, batch, batch_idx):
            output = self.forward(batch, "validate")
            self.log("val/loss", output["loss"].detach(), on_step=False, on_epoch=True,
                     sync_dist=True, prog_bar=True)
            return output["loss"]

        def configure_optimizers(self):
            opt_cfg = OmegaConf.to_container(self.cfg.optimizer, resolve=True)
            opt_type = opt_cfg.pop("type")
            if opt_type == "AdamW":
                opt = torch.optim.AdamW(
                    list(self.planner.parameters()) + list(self.planner_loss.parameters()),
                    **{k: v for k, v in opt_cfg.items() if k != "_target_"},
                )
            else:
                opt = hydra.utils.instantiate(
                    opt_cfg,
                    params=list(self.planner.parameters()) + list(self.planner_loss.parameters()),
                )
            return {"optimizer": opt}

    pl_module = PlannerLightningModule(wm, planner, planner_loss, cfg)

    ##########################
    ##       training       ##
    ##########################

    logger = None
    if cfg.wandb.enabled:
        from lightning.pytorch.loggers import WandbLogger
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    trainer = pl.Trainer(
        **OmegaConf.to_container(cfg.trainer, resolve=True),
        default_root_dir=PROJECT_ROOT,
        num_sanity_val_steps=0,
        logger=logger,
        enable_checkpointing=True,
        callbacks=[
            pl_callbacks.ModelCheckpoint(
                dirpath="checkpoints",
                filename="planner-{epoch:03d}-{val/loss:.4f}",
                save_top_k=3,
                monitor="val/loss",
                mode="min",
                save_last=True,
            ),
        ],
    )

    trainer.fit(pl_module, train, val)


if __name__ == "__main__":
    run()
