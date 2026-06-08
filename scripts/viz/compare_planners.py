#!/usr/bin/env python3
"""Compare planner_baseline vs planner_dir: WM rollout cost + LIBERO simulation.

Usage:
    source .venv/bin/activate
    python scripts/viz/compare_planners.py \
        --planner-bl checkpoints/planner_baseline/last.ckpt \
        --planner-dir checkpoints/planner_dir/last.ckpt \
        --num-samples 100
"""

import argparse, sys, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from libero_dataset import LiberoGoalDataset
from planner import PlannerDecoder, planner_rollout
from utils import get_img_preprocessor


def load_wm(ckpt_path: str):
    """Load frozen WM from object checkpoint."""
    wm = torch.load(os.path.expanduser(ckpt_path), map_location="cuda", weights_only=False)
    wm = wm.cuda().eval()
    for p in wm.parameters():
        p.requires_grad_(False)
    return wm


def load_planner(planner_ckpt_path: str, act_dim: int):
    """Load planner weights from Lightning checkpoint."""
    ckpt = torch.load(planner_ckpt_path, map_location="cuda", weights_only=True)
    state = ckpt["state_dict"]

    # Use default architecture matching planner_ft.yaml
    embed_dim = 192
    num_queries = 8
    horizon = 5
    action_substeps = 1
    num_layers = 3
    num_heads = 8
    mlp_dim = 1024
    dropout = 0.1
    action_range = 1.0

    planner = PlannerDecoder(
        embed_dim=embed_dim, num_queries=num_queries, horizon=horizon,
        action_dim=act_dim, action_substeps=action_substeps,
        num_layers=num_layers, num_heads=num_heads, mlp_dim=mlp_dim,
        dropout=dropout, action_range=action_range,
    ).cuda().eval()

    # Strip 'planner.' prefix from state dict keys
    planner_state = {}
    for k, v in state.items():
        if k.startswith("planner."):
            planner_state[k[len("planner."):]] = v
    planner.load_state_dict(planner_state, strict=True)
    for p in planner.parameters():
        p.requires_grad_(False)

    print(f"  Planner loaded: {sum(p.numel() for p in planner.parameters())/1e6:.2f}M params")
    return planner


def planner_evaluate(wm, planner, batch, history_size=3):
    """Run planner inference + WM rollout, return costs."""
    B = batch["pixels"].size(0)
    with torch.no_grad():
        out = wm.encode({k: v for k, v in batch.items() if torch.is_tensor(v)})
        ctx_emb = out["emb"][:, :history_size]
        goal_emb = out["emb"][:, -1:]
        hist_actions = batch["action"][:, :history_size]

    with torch.no_grad():
        actions, conf = planner(ctx_emb, goal_emb)
        info = {"pixels": batch["pixels"][:, :history_size]}
        pred_embs, _ = planner_rollout(
            wm, actions, info, history_size=history_size,
            hist_actions=hist_actions, goal_emb=goal_emb,
            ctx_emb=ctx_emb,
        )
        # Cost: MSE between final predicted embedding and goal embedding
        costs = (pred_embs[:, :, -1:] - goal_emb.unsqueeze(1)).pow(2).mean(dim=-1).squeeze(-1)  # (B, N)
        best_cost, best_idx = costs.min(dim=-1)
        conf_vals = torch.sigmoid(conf).squeeze(-1)  # (B, N)

    return costs, best_cost, best_idx, conf_vals, actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wm-bl", required=True, help="Baseline WM object checkpoint")
    parser.add_argument("--planner-bl", required=True, help="Baseline planner Lightning checkpoint")
    parser.add_argument("--wm-dir", required=True, help="Dir WM object checkpoint")
    parser.add_argument("--planner-dir", required=True, help="Dir planner Lightning checkpoint")
    parser.add_argument("--data", default="data/libero_goal")
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--sub-batch", type=int, default=8)
    parser.add_argument("--output", default="output/viz/planner_compare.png")
    args = parser.parse_args()

    num_steps = args.history_size + args.horizon
    transform = get_img_preprocessor("pixels", "pixels", 224)

    # ── Data ──
    print(f"Loading data from {args.data} ...")
    dataset = LiberoGoalDataset(
        data_dir=args.data, frameskip=args.frameskip,
        num_steps=num_steps, keys_to_load=["pixels", "action", "proprio"],
        keys_to_cache=["action", "proprio"], transform=transform,
    )
    rng = np.random.RandomState(42)
    n_total = len(dataset)
    indices = sorted(rng.choice(n_total, size=min(args.num_samples, n_total), replace=False))
    print(f"  Evaluating {len(indices)} samples (dataset size={n_total})")

    # ── Load WMs + planners ──
    print("\nLoading Baseline WM...")
    wm_bl = load_wm(args.wm_bl)
    act_dim = wm_bl.action_encoder.patch_embed.in_channels
    print(f"  act_dim={act_dim}")
    print("Loading Baseline Planner...")
    planner_bl = load_planner(args.planner_bl, act_dim)

    print("\nLoading Dir WM...")
    wm_dir = load_wm(args.wm_dir)
    print("Loading Dir Planner...")
    planner_dir = load_planner(args.planner_dir, act_dim)

    # ── Evaluate ──
    print(f"\nEvaluating ({len(indices)} samples)...")
    all_best_bl, all_best_dir = [], []
    bl_wins = 0
    dir_wins = 0

    for batch_start in range(0, len(indices), args.sub_batch):
        batch_end = min(batch_start + args.sub_batch, len(indices))
        batch_ix = indices[batch_start:batch_end]
        samples = [dataset[int(i)] for i in batch_ix]
        batch = {k: torch.stack([s[k] for s in samples]).cuda() for k in samples[0]}

        costs_bl, best_bl, best_idx_bl, conf_bl, acts_bl = planner_evaluate(wm_bl, planner_bl, batch, args.history_size)
        costs_dir, best_dir, best_idx_dir, conf_dir, acts_dir = planner_evaluate(wm_dir, planner_dir, batch, args.history_size)

        all_best_bl.extend(best_bl.cpu().tolist())
        all_best_dir.extend(best_dir.cpu().tolist())
        bl_wins += (best_bl < best_dir).sum().item()
        dir_wins += (best_dir < best_bl).sum().item()

        print(f"  [{batch_start:4d}-{batch_end:4d}]  "
              f"BL best={best_bl.mean():.4f}  Dir best={best_dir.mean():.4f}  "
              f"BL_wins={bl_wins}  Dir_wins={dir_wins}")

    # ── Summary ──
    best_bl_arr = np.array(all_best_bl)
    best_dir_arr = np.array(all_best_dir)
    print(f"\n{'='*60}")
    print(f"Planner comparison ({len(indices)} samples):")
    print(f"  Baseline: mean WM cost = {best_bl_arr.mean():.5f} ± {best_bl_arr.std():.5f}")
    print(f"  Dir:      mean WM cost = {best_dir_arr.mean():.5f} ± {best_dir_arr.std():.5f}")
    print(f"  Baseline wins: {bl_wins} ({bl_wins/len(indices)*100:.1f}%)")
    print(f"  Dir wins:      {dir_wins} ({dir_wins/len(indices)*100:.1f}%)")
    delta = best_dir_arr - best_bl_arr
    print(f"  Mean Δ (Dir-BL): {delta.mean():+.5f}")
    win_margin = (best_bl_arr - best_dir_arr)
    bl_better_margin = win_margin[win_margin > 0].mean() if (win_margin > 0).any() else 0
    dir_better_margin = (-win_margin[win_margin < 0]).mean() if (win_margin < 0).any() else 0
    print(f"  Avg win margin: BL={bl_better_margin:.5f}, Dir={dir_better_margin:.5f}")

    # ── Plot ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter: BL cost vs Dir cost
    ax = axes[0]
    ax.scatter(best_bl_arr, best_dir_arr, alpha=0.3, s=8)
    lims = [min(best_bl_arr.min(), best_dir_arr.min()), max(best_bl_arr.max(), best_dir_arr.max())]
    ax.plot(lims, lims, 'k--', alpha=0.3)
    ax.set_xlabel('Baseline Planner WM Cost')
    ax.set_ylabel('Dir Planner WM Cost')
    ax.set_title(f'Per-Sample Cost\nBL wins={bl_wins}, Dir wins={dir_wins}')
    ax.grid(True, alpha=0.3)

    # Histogram of cost differences
    ax = axes[1]
    ax.hist(delta, bins=40, color='gray', edgecolor='white', alpha=0.7)
    ax.axvline(0, color='k', linestyle='--')
    ax.axvline(delta.mean(), color='red', linestyle='-', label=f'Mean={delta.mean():+.5f}')
    ax.set_xlabel('Δ WM Cost (Dir - Baseline)')
    ax.set_ylabel('Count')
    ax.set_title('Cost Difference Distribution')
    ax.legend()

    plt.tight_layout()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
