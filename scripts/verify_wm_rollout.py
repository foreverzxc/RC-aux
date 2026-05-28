"""Verify RC-aux WM cumulative error: WM rollout vs Sim with GT actions.

Goal = frame 10 steps ahead (the state reached by executing all GT actions).
Sim-vs-Goal at step 10 should be ~0.

Usage:
    source .venv/bin/activate
    python scripts/verify_wm_rollout.py \
        --wm models/checkpoints/pixel_control/cube_rcaux/rcaux_cube_object.ckpt \
        --horizon 10 --samples 5
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from train_planner import load_frozen_wm
from utils import get_img_preprocessor
from scripts.viz.cube import make_env, set_mujoco_state, get_state, RAW_ACT, FS, CTX_LEN, DATASET_NAME

import stable_worldmodel as swm

OUT_DIR = Path(__file__).parent.parent / "output" / "verify"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def encode_frame(wm, frame, transform):
    """Encode a single (H,W,C) uint8 frame. Returns (1,1,D)."""
    f = torch.from_numpy(frame.copy()).permute(2, 0, 1)
    f = transform({"pixels": f})["pixels"].unsqueeze(1).cuda()
    with torch.no_grad():
        return wm.encode({"pixels": f})["emb"]  # (1, 1, D)


def run_one(wm, ds_raw, ds_tf, transform, idx, horizon, device):
    """Run verification. Returns arrays of length H."""
    item_raw = ds_raw[idx]
    item_tf = ds_tf[idx]

    ctx_pixels = item_tf["pixels"][:CTX_LEN].unsqueeze(0).cuda()  # (1, CTX, C, H, W)
    gt_actions = item_raw["action"][CTX_LEN - 1:CTX_LEN - 1 + horizon].numpy()
    gt_actions_t = torch.from_numpy(gt_actions).unsqueeze(0).cuda()

    # ── Sim rollout (runs first so we can use its last frame as goal) ──
    env = make_env()
    qpos, qvel = get_state(ds_raw, idx)
    set_mujoco_state(env, qpos, qvel)
    env.step(np.zeros(RAW_ACT, dtype=np.float32))

    gt_substeps = gt_actions.reshape(horizon, FS, RAW_ACT)
    sim_frames = []
    for substeps in gt_substeps:
        for a in substeps:
            env.step(np.clip(a.astype(np.float32), -1, 1))
        sim_frames.append(env.render())

    # Goal = last sim frame (what the sim actually reached after H steps)
    goal_frame = sim_frames[-1]
    goal_emb = encode_frame(wm, goal_frame, transform)  # (1, 1, D)

    # Also get dataset clip's last frame to check alignment
    ds_goal_frame_raw = item_raw["pixels"][CTX_LEN + horizon - 1]  # (C,H,W) at clip end
    ds_goal_frame = ds_goal_frame_raw.permute(1, 2, 0).numpy().astype(np.uint8)
    ds_goal_emb = encode_frame(wm, ds_goal_frame, transform)  # (1, 1, D)

    ds_sim_goal_mse = (ds_goal_emb - goal_emb).pow(2).mean().item()

    # ── Encode context ──
    with torch.no_grad():
        ctx_out = wm.encode({"pixels": ctx_pixels})
        ctx_emb = ctx_out["emb"]  # (1, CTX, D)

    # ── WM open-loop rollout with GT actions ──
    B, HS, D_emb = ctx_emb.shape
    act_dim = gt_actions_t.shape[-1]

    emb = ctx_emb.clone()
    hist_act = torch.zeros(B, HS, act_dim, device=device)
    hist_act[:, -1:] = gt_actions_t[:, 0:1]

    wm_rollout_embs = []
    for t in range(horizon):
        cur_act = gt_actions_t[:, t:t + 1]
        if t > 0:
            hist_act = torch.cat([hist_act[:, 1:], cur_act], dim=1)
        act_emb = wm.action_encoder(hist_act)
        pred = wm.predict(emb[:, -HS:], act_emb)[:, -1:]
        emb = torch.cat([emb, pred], dim=1)
        wm_rollout_embs.append(pred.squeeze(1))

    wm_rollout_embs = torch.stack(wm_rollout_embs, dim=1)  # (1, H, D)

    # ── Encode sim frames ──
    sim_embs = []
    for frame in sim_frames:
        e = encode_frame(wm, frame, transform)  # (1, 1, D)
        sim_embs.append(e.squeeze(1))
    sim_embs = torch.stack(sim_embs, dim=1)  # (1, H, D)

    # ── MSEs ──
    wm_vs_sim  = (wm_rollout_embs - sim_embs).pow(2).mean(dim=-1).squeeze(0).cpu().numpy()
    wm_vs_goal = (wm_rollout_embs - goal_emb).pow(2).mean(dim=-1).squeeze(0).cpu().numpy()
    sim_vs_goal = (sim_embs - goal_emb).pow(2).mean(dim=-1).squeeze(0).cpu().numpy()

    return wm_vs_sim, wm_vs_goal, sim_vs_goal, ds_sim_goal_mse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wm", required=True)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--frameskip", type=int, default=FS)
    args = parser.parse_args()

    device = torch.device("cuda")

    print(f"Loading WM from {args.wm}...")
    wm = load_frozen_wm(args.wm).cuda().eval()

    transform = get_img_preprocessor("pixels", "pixels", 224)
    ds_raw = swm.data.HDF5Dataset(
        DATASET_NAME, frameskip=args.frameskip,
        num_steps=CTX_LEN + args.horizon, transform=None)
    ds_tf = swm.data.HDF5Dataset(
        DATASET_NAME, frameskip=args.frameskip,
        num_steps=CTX_LEN + args.horizon, transform=transform)

    total = len(ds_raw)
    indices = np.linspace(0, total - 1, args.samples, dtype=int).tolist()
    print(f"Running on {args.samples} samples (indices: {indices}), horizon={args.horizon}")
    print(f"Goal = sim frame at step {args.horizon} (should give Sim-Goal=0 at final step)\n")

    all_wm_sim = []
    all_wm_goal = []
    all_sim_goal = []
    ds_sim_gaps = []

    for idx in indices:
        print(f"  idx={idx}...", end=" ", flush=True)
        wm_sim, wm_goal, sim_goal, gap = run_one(
            wm, ds_raw, ds_tf, transform, idx, args.horizon, device)
        all_wm_sim.append(wm_sim)
        all_wm_goal.append(wm_goal)
        all_sim_goal.append(sim_goal)
        ds_sim_gaps.append(gap)
        print(f"DS-Sim goal MSE={gap:.6f} | sim_goal[0]={sim_goal[0]:.4f} sim_goal[-1]={sim_goal[-1]:.6f} | "
              f"wm_sim[0]={wm_sim[0]:.4f} wm_sim[-1]={wm_sim[-1]:.4f}")

    all_wm_sim  = np.array(all_wm_sim)
    all_wm_goal = np.array(all_wm_goal)
    all_sim_goal = np.array(all_sim_goal)

    print(f"\nDataset goal vs Sim goal MSE: mean={np.mean(ds_sim_gaps):.6f} std={np.std(ds_sim_gaps):.6f}")
    print(f"(If this is large, sim rendering differs from dataset pre-rendered frames)\n")

    print(f"{'Step':>5}  {'WM-Sim mean':>12}  {'WM-Sim std':>10}  {'WM-Goal':>10}  {'Sim-Goal':>12}")
    print("-" * 62)
    for t in range(args.horizon):
        print(f"{t+1:>5}  {all_wm_sim[:, t].mean():>12.6f}  {all_wm_sim[:, t].std():>10.6f}"
              f"  {all_wm_goal[:, t].mean():>10.6f}  {all_sim_goal[:, t].mean():>12.6f}")

    # ── Plot ──
    steps = np.arange(1, args.horizon + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    mu = all_wm_sim.mean(axis=0)
    sd = all_wm_sim.std(axis=0)
    ax1.fill_between(steps, mu - sd, mu + sd, alpha=0.2, color='#e74c3c')
    ax1.plot(steps, mu, 'o-', color='#e74c3c', linewidth=2, markersize=5,
             label=f"WM rollout vs Sim (n={args.samples})")
    ax1.set_xlabel("Rollout step")
    ax1.set_ylabel("MSE")
    ax1.set_title("Cumulative WM Open-Loop Error")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    ax2.plot(steps, all_wm_goal.mean(axis=0), 's--', color='#3498db', linewidth=2, markersize=5,
             label="WM rollout vs Goal")
    ax2.fill_between(steps, all_wm_goal.mean(axis=0) - all_wm_goal.std(axis=0),
                     all_wm_goal.mean(axis=0) + all_wm_goal.std(axis=0),
                     alpha=0.1, color='#3498db')
    ax2.plot(steps, all_sim_goal.mean(axis=0), 'o-', color='#2ecc71', linewidth=2, markersize=5,
             label="Sim vs Goal (~0 at end)")
    ax2.fill_between(steps, all_sim_goal.mean(axis=0) - all_sim_goal.std(axis=0),
                     all_sim_goal.mean(axis=0) + all_sim_goal.std(axis=0),
                     alpha=0.1, color='#2ecc71')
    ax2.set_xlabel("Rollout step")
    ax2.set_ylabel("MSE")
    ax2.set_title("Distance to Goal (sim last frame)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    out_path = OUT_DIR / f"wm_rollout_error_h{args.horizon}_s{args.samples}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")

    np.savez(OUT_DIR / f"wm_rollout_error_h{args.horizon}_s{args.samples}.npz",
             steps=steps, wm_sim=all_wm_sim, wm_goal=all_wm_goal,
             sim_goal=all_sim_goal, ds_sim_gaps=ds_sim_gaps)


if __name__ == "__main__":
    main()
