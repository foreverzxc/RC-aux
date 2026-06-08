#!/usr/bin/env python3
"""Evaluate trained PlannerHead: compare planned actions vs GT on libero_goal.

Usage:
    python scripts/viz/eval_planner.py --ckpt lightning_logs/version_38/checkpoints/epoch=4-step=26285.ckpt
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
from jepa import JEPA
from module import PlannerHead, MLP, Embedder, ARPredictor
from planner import planner_rollout
from utils import get_img_preprocessor


def build_model_from_ckpt(ckpt_path: str):
    """Rebuild JEPA + PlannerHead from Lightning checkpoint state_dict."""
    ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=True)
    state = ckpt["state_dict"]

    # Get dimensions from state
    embed_dim = state["model.planner_head.input_proj.weight"].shape[1]  # 192
    query_dim = state["model.planner_head.query_embed"].shape[-1]  # 32
    act_dim = state["model.action_encoder.patch_embed.weight"].shape[1]  # 35
    horizon = 5
    num_queries = state["model.planner_head.query_embed"].shape[1]  # 8

    print(f"Config from checkpoint: embed_dim={embed_dim}, query_dim={query_dim}, "
          f"act_dim={act_dim}, num_queries={num_queries}")

    # Build encoder (ViT-tiny)
    import stable_pretraining as spt
    encoder = spt.backbone.utils.vit_hf(
        "tiny", patch_size=14, image_size=224, pretrained=False, use_mask_token=False)
    hidden_dim = encoder.config.hidden_size  # 192

    # Build WM components
    predictor = ARPredictor(
        num_frames=3, input_dim=embed_dim, hidden_dim=hidden_dim,
        output_dim=hidden_dim, depth=6, heads=16, mlp_dim=2048,
        dim_head=64, dropout=0.1, emb_dropout=0.0)

    action_encoder = Embedder(input_dim=act_dim, emb_dim=embed_dim)
    projector = MLP(input_dim=hidden_dim, output_dim=embed_dim,
                    hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d)
    pred_proj = MLP(input_dim=hidden_dim, output_dim=embed_dim,
                    hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d)
    planner_head = PlannerHead(
        embed_dim=embed_dim, query_dim=query_dim, num_queries=num_queries,
        horizon=horizon, action_dim=act_dim, head_type="mlp",
        num_layers=3, num_heads=8, mlp_dim=1024, dropout=0.1)

    model = JEPA(
        encoder=encoder, predictor=predictor, action_encoder=action_encoder,
        projector=projector, pred_proj=pred_proj, planner_head=planner_head)
    model = model.cuda().eval()

    # Load state
    model_state = {}
    for k, v in state.items():
        if k.startswith("model."):
            model_state[k[len("model."):]] = v
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    print(f"Loaded state_dict: missing={len(missing)}, unexpected={len(unexpected)}")

    return model


def evaluate_planner(model, dataset, indices, history_size=3, horizon=5):
    """Run planner inference + WM rollout on samples, return per-sample costs."""
    results = []
    for idx in indices:
        sample = dataset[int(idx)]
        batch = {k: v.unsqueeze(0).cuda() for k, v in sample.items() if torch.is_tensor(v)}

        with torch.no_grad():
            out = model.encode(batch)
            ctx_emb = out["emb"][:, :history_size]
            goal_emb = out["emb"][:, -1:]
            hist_actions = batch["action"][:, :history_size]

            # Planner inference
            actions, conf = model.plan_actions(ctx_emb, goal_emb)

            # WM rollout
            info = {"pixels": batch["pixels"][:, :history_size]}
            pred_embs, _ = planner_rollout(
                model, actions, info, history_size=history_size,
                hist_actions=hist_actions, goal_emb=goal_emb, ctx_emb=ctx_emb)

            # Costs per query
            N = actions.shape[1]
            goal = goal_emb.reshape(1, 1, 1, -1).expand(-1, N, horizon, -1)
            costs = (pred_embs - goal).pow(2).mean(dim=-1).mean(dim=-1)  # (1, N)
            best_cost, best_idx = costs.min(dim=-1)
            conf_vals = torch.sigmoid(conf).squeeze()

            # GT action comparison
            gt_actions = batch["action"][0, history_size:history_size + horizon]  # (H, A)

        results.append({
            "idx": idx,
            "best_cost": best_cost.item(),
            "best_query": best_idx.item(),
            "costs": costs.squeeze().cpu().numpy(),
            "conf": conf_vals.squeeze().cpu().numpy(),
            "planner_actions": actions[0, best_idx].cpu().numpy(),
            "gt_actions": gt_actions.cpu().numpy(),
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data", default="data/libero_goal")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--output", default="output/viz/planner_eval.png")
    args = parser.parse_args()

    # ── Load model ──
    print("Loading model...")
    model = build_model_from_ckpt(args.ckpt)

    # ── Load data ──
    print(f"\nLoading data from {args.data}...")
    transform = get_img_preprocessor("pixels", "pixels", 224)
    dataset = LiberoGoalDataset(
        data_dir=args.data, frameskip=5, num_steps=8,
        keys_to_load=["pixels", "action", "proprio"],
        keys_to_cache=["action", "proprio"], transform=transform)
    rng = np.random.RandomState(42)
    indices = sorted(rng.choice(len(dataset), size=min(args.num_samples, len(dataset)), replace=False))

    # ── Evaluate ──
    print(f"\nEvaluating {len(indices)} samples...")
    results = evaluate_planner(model, dataset, indices)

    best_costs = np.array([r["best_cost"] for r in results])
    print(f"\n{'='*50}")
    print(f"Planner evaluation ({len(results)} samples):")
    print(f"  Best cost:  mean={best_costs.mean():.5f}  std={best_costs.std():.5f}")
    print(f"  Min: {best_costs.min():.5f}  Max: {best_costs.max():.5f}")

    # ── Action comparison ──
    # Planner: (H, 35) = (H, frameskip*raw_dim). GT: (H, 7) = (H, raw_dim).
    raw_dim = 7
    all_plan = np.stack([r["planner_actions"] for r in results])[:, :, :raw_dim]
    all_gt = np.stack([r["gt_actions"] for r in results])
    print(f"  shapes: plan={all_plan.shape}, gt={all_gt.shape}")
    action_mse = float(((all_plan - all_gt) ** 2).mean())  # scalar MSE

    print(f"\nAction comparison (planner best vs GT):")
    print(f"  Overall MSE: {action_mse:.5f}")

    # ── Plot ──
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.hist(best_costs, bins=30, color='steelblue', edgecolor='white')
    ax.axvline(best_costs.mean(), color='red', linestyle='--', label=f'mean={best_costs.mean():.4f}')
    ax.set_xlabel('Best WM Cost')
    ax.set_ylabel('Count')
    ax.set_title(f'Planner Rollout Cost (action MSE vs GT={action_mse:.4f})')
    ax.legend()

    plt.tight_layout()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
