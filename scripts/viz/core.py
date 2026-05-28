"""Common visualization pipeline — loads WM + planner, runs rollout, builds GIF."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch, imageio
from PIL import Image

from planner import PlannerDecoder, planner_rollout
from train_planner import load_frozen_wm
from utils import get_img_preprocessor
import stable_worldmodel as swm

OUT = Path(__file__).parent.parent.parent / "output" / "viz"
OUT.mkdir(parents=True, exist_ok=True)

CTX_LEN = 3


def load_wm(ckpt_path):
    wm = load_frozen_wm(ckpt_path)
    return wm.cuda().eval()


def load_planner(weights_path):
    ckpt = torch.load(weights_path, map_location="cuda", weights_only=True)
    p_cfg = ckpt["cfg"]
    planner = PlannerDecoder(
        embed_dim=p_cfg["embed_dim"],
        num_queries=p_cfg["num_queries"],
        horizon=p_cfg["horizon"],
        action_dim=p_cfg["act_dim"],
        action_substeps=p_cfg["action_substeps"],
        num_layers=p_cfg["num_layers"],
        num_heads=p_cfg["num_heads"],
        mlp_dim=p_cfg["mlp_dim"],
    ).cuda().eval()
    planner.load_state_dict(ckpt["planner"])
    return planner, p_cfg


def load_sample(dataset_name, horizon, frameskip, idx=None):
    transform = get_img_preprocessor("pixels", "pixels", 224)
    ds_raw = swm.data.HDF5Dataset(
        dataset_name, frameskip=frameskip, num_steps=CTX_LEN + horizon, transform=None)
    ds_tf = swm.data.HDF5Dataset(
        dataset_name, frameskip=frameskip, num_steps=CTX_LEN + horizon, transform=transform)

    if idx is None:
        idx = len(ds_raw) // 4

    item_raw = ds_raw[idx]
    item_tf = ds_tf[idx]
    ds_frames = [item_raw["pixels"][i].permute(1, 2, 0).numpy().astype(np.uint8)
                 for i in range(CTX_LEN + horizon)]
    batch = {k: v.unsqueeze(0).cuda() for k, v in item_tf.items() if torch.is_tensor(v)}
    return ds_frames, batch, idx, ds_raw


def planner_evaluate(wm, planner, batch):
    with torch.no_grad():
        out = wm.encode(batch)
        ctx_emb = out["emb"][:, :CTX_LEN]
        goal_emb = out["emb"][:, -1:]

    hist_actions = batch["action"][:, :CTX_LEN]
    info = {"pixels": batch["pixels"][:, :CTX_LEN]}

    with torch.no_grad():
        actions, conf = planner(ctx_emb, goal_emb)
        pred_embs, _ = planner_rollout(
            wm, actions, info, history_size=CTX_LEN,
            hist_actions=hist_actions, goal_emb=goal_emb,
        )
        wm_costs = (pred_embs[:, :, -1:] - goal_emb.unsqueeze(1)).pow(2).mean(dim=-1).squeeze(-1)[0].cpu().numpy()
        conf_vals = torch.sigmoid(conf).squeeze().cpu().numpy()

    return actions, conf_vals, wm_costs, goal_emb


def encode_frame(wm, frame, transform):
    """Encode a single frame (HWC uint8) through WM, returns (1, 1, D) embedding."""
    f = torch.from_numpy(frame.copy()).permute(2, 0, 1)
    f = transform({"pixels": f})["pixels"].unsqueeze(1).cuda()
    with torch.no_grad():
        return wm.encode({"pixels": f})["emb"]


def build_gif(ds_frames, goal_img, sim_results, horizon, frameskip, out_path):
    """Build side-by-side comparison GIF.

    sim_results: list of dicts {name, frames}
    """
    ds_h, ds_w = ds_frames[0].shape[:2]
    combined = []

    labels = ["Dataset"] + [r["name"] for r in sim_results] + ["Goal"]
    n_cols = len(labels)
    panel_w = ds_w * n_cols
    panel_h = ds_h + 28  # + label bar

    for k in range(horizon + 1):
        ds_i = max(0, CTX_LEN - 1 + k)
        sim_step = k * frameskip

        row = [_force_size(ds_frames[ds_i], ds_h, ds_w)]
        for r in sim_results:
            f = r["frames"]
            s = min(sim_step, len(f) - 1)
            row.append(_force_size(f[s], ds_h, ds_w))
        row.append(_force_size(goal_img, ds_h, ds_w))

        panel = np.hstack(row)
        if k == 0:
            panel = _add_labels(panel, labels)
        # Ensure consistent size
        panel = _force_size(panel, panel_h, panel_w)
        combined.append(panel)

    imageio.mimsave(out_path, combined, fps=2, loop=0)
    return out_path


def _force_size(img, h, w):
    """Resize to exact (h, w) regardless of input shape."""
    if img.shape[0] == h and img.shape[1] == w:
        return img
    if img.shape[0] == h + 28 and img.shape[1] == w:
        return img  # already has label bar
    pil = Image.fromarray(img)
    return np.array(pil.resize((w, h), Image.NEAREST))


def _resize(img, h, w):  # also used by run.py
    if img.shape[:2] != (h, w):
        return np.array(Image.fromarray(img).resize((w, h), Image.NEAREST))
    return img


def _add_labels(panel, labels):
    """Add a label bar at the top of the panel."""
    from PIL import ImageDraw
    img = Image.fromarray(panel)
    bar_h = 28
    w = img.width
    bar = Image.new("RGB", (w, bar_h), (30, 30, 30))
    draw = ImageDraw.Draw(bar)
    col_w = w // max(len(labels), 1)
    for i, label in enumerate(labels):
        x = i * col_w + 4
        draw.text((x, 6), label[:30], fill=(220, 220, 220))
    return np.vstack([np.array(bar), panel])
