#!/usr/bin/env python3
"""Compare open-loop rollout quality: baseline vs direction-consistency-trained WM.

Loads two WM object checkpoints, runs open-loop GT-action rollouts at increasing
horizons, and plots per-step MSE + direction cosine similarity.

Usage:
    source .venv/bin/activate
    python scripts/viz/compare_rollout.py \
        --baseline checkpoints/baseline/ablation_baseline_epoch_10_object.ckpt \
        --dir checkpoints/dir/ablation_dir_epoch_10_object.ckpt \
        --data data/libero_goal \
        --horizon 4 --num-samples 200
"""

import argparse, sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from libero_dataset import LiberoGoalDataset
from utils import get_img_preprocessor


def load_wm(ckpt_path: str):
    """Load a frozen WM from an object checkpoint (torch.save whole model)."""
    import os
    print(f"Loading WM from {ckpt_path} ...")
    ckpt_path = os.path.expanduser(ckpt_path)
    wm = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    wm = wm.cuda().eval()
    n_params = sum(p.numel() for p in wm.parameters())
    print(f"  Loaded: {n_params/1e6:.1f}M params")
    return wm


def run_open_loop_rollout(wm, batch, history_size: int, horizon: int):
    """Open-loop rollout with GT actions. Returns per-step MSE and direction cos-sim.

    Returns:
        per_step_mse:     (horizon,)  float32
        per_step_cos_sim: (horizon-1,) float32  (empty if horizon < 2)
    """
    with torch.no_grad():
        out = wm.encode({k: v for k, v in batch.items() if torch.is_tensor(v)})
        emb = out["emb"]
        act_emb = out["act_emb"]

    ctx_emb = emb[:, :history_size]
    ctx_act = act_emb[:, :history_size]
    future_act = act_emb[:, history_size:history_size + horizon - 1]
    true_future = emb[:, history_size:history_size + horizon]

    with torch.no_grad():
        pred_future = wm.rollout_open_loop(
            ctx_emb, ctx_act, future_act,
            horizon=horizon,
            history_size=history_size,
            teacher_prob=0.0,
        )

    sq_errors = (pred_future - true_future).pow(2).mean(dim=(0, 2))
    per_step_mse = sq_errors.cpu().numpy()

    if horizon >= 2:
        pred_dirs = pred_future[:, 1:] - pred_future[:, :-1]
        true_dirs = true_future[:, 1:] - true_future[:, :-1]
        cos_sim = torch.nn.functional.cosine_similarity(
            pred_dirs.reshape(-1, pred_dirs.size(-1)),
            true_dirs.reshape(-1, true_dirs.size(-1)),
            dim=-1,
        ).view(pred_future.size(0), horizon - 1).mean(dim=0)
        per_step_cos_sim = cos_sim.cpu().numpy()
    else:
        per_step_cos_sim = np.array([])

    return per_step_mse, per_step_cos_sim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--dir", required=True)
    parser.add_argument("--data", default="data/libero_goal")
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--output", default="output/viz/rollout_compare.png")
    parser.add_argument("--sub-batch", type=int, default=8,
                        help="Eval sub-batch size (avoid OOM)")
    args = parser.parse_args()

    num_steps = args.history_size + args.horizon
    transform = get_img_preprocessor("pixels", "pixels", 224)

    # ── Load data ──
    print(f"\nLoading LiberoGoalDataset from {args.data} ...")
    dataset = LiberoGoalDataset(
        data_dir=args.data,
        frameskip=args.frameskip,
        num_steps=num_steps,
        keys_to_load=["pixels", "action", "proprio"],
        keys_to_cache=["action", "proprio"],
        transform=transform,
    )
    print(f"  Dataset size: {len(dataset)}")

    # Subsample indices
    rng = np.random.RandomState(42)
    n_total = len(dataset)
    indices = rng.choice(n_total, size=min(args.num_samples, n_total), replace=False)
    indices = sorted(indices)
    print(f"  Evaluating {len(indices)} random episodes")

    # ── Load models ──
    wm_bl = load_wm(args.baseline)
    wm_dir = load_wm(args.dir)

    # ── Run rollouts ──
    print(f"\nOpen-loop rollout (horizon={args.horizon}) ...")

    all_mse_bl, all_mse_dir = [], []
    all_cos_bl, all_cos_dir = [], []

    for batch_start in range(0, len(indices), args.sub_batch):
        batch_end = min(batch_start + args.sub_batch, len(indices))
        batch_ix = indices[batch_start:batch_end]

        # Stack samples from LiberoGoalDataset
        samples = [dataset[int(i)] for i in batch_ix]
        batch = {
            k: torch.stack([s[k] for s in samples]).cuda()
            for k in samples[0]
        }

        mse_bl, cos_bl = run_open_loop_rollout(wm_bl, batch, args.history_size, args.horizon)
        mse_dir, cos_dir = run_open_loop_rollout(wm_dir, batch, args.history_size, args.horizon)

        all_mse_bl.append(mse_bl)
        all_mse_dir.append(mse_dir)
        if len(cos_bl) > 0:
            all_cos_bl.append(cos_bl)
            all_cos_dir.append(cos_dir)

        print(f"  [{batch_start:4d}-{batch_end:4d}]  "
              f"BL step1={mse_bl[0]:.5f}  "
              f"Dir step1={mse_dir[0]:.5f}  "
              f"cos2={cos_bl[0] if len(cos_bl) else float('nan'):.4f}/{cos_dir[0] if len(cos_dir) else float('nan'):.4f}")

    # ── Aggregate ──
    mse_bl = np.stack(all_mse_bl).mean(axis=0)
    mse_dir = np.stack(all_mse_dir).mean(axis=0)
    cos_bl = np.stack(all_cos_bl).mean(axis=0) if all_cos_bl else np.array([])
    cos_dir = np.stack(all_cos_dir).mean(axis=0) if all_cos_dir else np.array([])

    # ── Print table ──
    print(f"\n{'='*70}")
    print(f"Per-step MSE  (N={len(indices)} episodes)")
    print(f"{'Horizon':<10} {'Baseline':<14} {'Dir':<14} {'Δ (Dir-BL)':<14} {'Rel Δ':<10}")
    print(f"{'-'*62}")
    for h in range(args.horizon):
        delta = mse_dir[h] - mse_bl[h]
        rel = delta / (mse_bl[h] + 1e-8) * 100
        print(f"  step {h+1:<3}    {mse_bl[h]:.6f}        {mse_dir[h]:.6f}        {delta:+.6f}      {rel:+.1f}%")

    if len(cos_bl) > 0:
        print(f"\nDirection cosine similarity (higher = better):")
        print(f"{'Step pair':<14} {'Baseline':<14} {'Dir':<14} {'Δ':<10}")
        print(f"{'-'*48}")
        for h in range(args.horizon - 1):
            delta = cos_dir[h] - cos_bl[h]
            print(f"  {h+1}→{h+2:<9} {cos_bl[h]:.6f}        {cos_dir[h]:.6f}        {delta:+.6f}")

    # ── Plot ──
    has_cos = len(cos_bl) > 0
    fig, axes = plt.subplots(1, 2 if has_cos else 1,
                             figsize=(12 if has_cos else 6, 5))
    if not has_cos:
        axes = [axes]

    steps = np.arange(1, args.horizon + 1)

    # Per-step MSE
    ax = axes[0]
    ax.plot(steps, mse_bl, 'o-', color='#2196F3', label='Baseline (no dir)', lw=2, ms=8)
    ax.plot(steps, mse_dir, 's--', color='#FF5722', label='Dir (weight=0.1)', lw=2, ms=8)
    for h in range(args.horizon):
        rel = (mse_dir[h] - mse_bl[h]) / (mse_bl[h] + 1e-8) * 100
        ax.annotate(f'{rel:+.0f}%', (steps[h], mse_dir[h]),
                    textcoords="offset points", xytext=(0, 12),
                    ha='center', fontsize=8, color='#FF5722')
    ax.set_xlabel('Rollout Step', fontsize=12)
    ax.set_ylabel('MSE (embedding space)', fontsize=12)
    ax.set_title('Open-Loop Rollout Prediction Error', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(steps)

    # Direction cosine similarity
    if has_cos:
        ax = axes[1]
        dsteps = np.arange(1, args.horizon)
        ax.plot(dsteps, cos_bl, 'o-', color='#2196F3', label='Baseline', lw=2, ms=8)
        ax.plot(dsteps, cos_dir, 's--', color='#FF5722', label='Dir', lw=2, ms=8)
        ax.set_xlabel('Step Transition', fontsize=12)
        ax.set_ylabel('Cosine Similarity', fontsize=12)
        ax.set_title('Direction Consistency (cos sim)', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(dsteps)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
