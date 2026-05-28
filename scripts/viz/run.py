"""Unified planner visualization runner.

Usage:
    python scripts/viz/run.py cube  --wm PATH --planner PATH
    python scripts/viz/run.py pusht --wm PATH --planner PATH
"""

import argparse, importlib, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch

from scripts.viz import core
from utils import get_img_preprocessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("env", choices=["pusht", "cube"])
    parser.add_argument("--wm", required=True)
    parser.add_argument("--planner", required=True)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--num-show", type=int, default=4)
    parser.add_argument("--idx", type=int, default=None)
    args = parser.parse_args()

    env = importlib.import_module(f"scripts.viz.{args.env}")
    transform = get_img_preprocessor("pixels", "pixels", 224)

    print(f"=== {args.env.upper()} Planner Visualization ===")

    # ── Load models ──
    print("Loading WM...")
    wm = core.load_wm(args.wm)
    print("Loading planner...")
    planner, p_cfg = core.load_planner(args.planner)
    horizon = p_cfg["horizon"]

    # ── Load data ──
    ds_frames, batch, ds_idx, ds_raw = core.load_sample(
        env.DATASET_NAME, horizon, env.FS, idx=args.idx)

    # ── Get initial state ──
    initial_state = env.get_state(ds_raw, ds_idx)

    # ── Planner evaluate ──
    actions, conf_vals, wm_costs, goal_emb = core.planner_evaluate(wm, planner, batch)

    print(f"\n{'Query':<8} {'Conf':>8} {'WM cost':>10}")
    print("-" * 30)
    for i in range(min(8, len(wm_costs))):
        print(f"{i:<8} {conf_vals[i]:>8.3f} {wm_costs[i]:>10.4f}")

    # ── Select queries to visualize ──
    sorted_idx = wm_costs.argsort()
    show_queries = [int(i) for i in sorted_idx[:args.num_show]]
    print(f"\nShowing queries: {show_queries}")

    # ── Shared env instance for all simulations ──
    sim_env = None
    if hasattr(env, 'make_env'):
        sim_env = env.make_env()

    # ── GT simulation ──
    item_raw = ds_raw[ds_idx]
    gt_raw = env.get_gt_actions(item_raw, horizon)
    gt_acts_2d = gt_raw.reshape(horizon, env.FS, env.RAW_ACT).reshape(-1, env.RAW_ACT)

    if initial_state is not None and sim_env is not None:
        gt_frames = env.run_sim(sim_env, *initial_state, gt_acts_2d)
    elif initial_state is not None:
        gt_frames = env.run_sim(*initial_state, gt_acts_2d)
    else:
        gt_frames = env.run_sim(gt_acts_2d)
    gt_sim_cost = core.encode_frame(wm, gt_frames[-1], transform)
    gt_sim_cost = (gt_sim_cost - goal_emb).pow(2).mean().item()

    sim_results = [{"name": "GT", "frames": gt_frames, "cost": gt_sim_cost}]
    # Verify per-step alignment: GT sim vs dataset frames
    print(f"  GT: Sim={gt_sim_cost:.4f}")
    print(f"  Per-step GT sim vs Dataset pixel diff:")
    for k in range(horizon + 1):
        ds_i = min(env.CTX_LEN - 1 + k, len(ds_frames) - 1)
        sim_step = k * env.FS
        s = min(sim_step, len(gt_frames) - 1)
        diff = np.abs(gt_frames[s].astype(float) - ds_frames[ds_i].astype(float)).mean()
        marker = " ✓" if diff < 5 else " ✗ MISALIGNED"
        print(f"    k={k}: ds_frame[{ds_i}] vs sim_frame[{s}] diff={diff:.2f}{marker}")

    # ── Planner queries ──
    for qi in show_queries:
        acts_all = actions[0, qi].cpu().numpy()
        acts_2d = acts_all.reshape(horizon, env.FS, env.RAW_ACT).reshape(-1, env.RAW_ACT)

        if initial_state is not None and sim_env is not None:
            frames = env.run_sim(sim_env, *initial_state, acts_2d)
        elif initial_state is not None:
            frames = env.run_sim(*initial_state, acts_2d)
        else:
            frames = env.run_sim(acts_2d)

        sim_cost = core.encode_frame(wm, frames[-1], transform)
        sim_cost = (sim_cost - goal_emb).pow(2).mean().item()

        name = f"Plan#{qi} wm={wm_costs[qi]:.3f} sim={sim_cost:.3f}"
        sim_results.append({"name": name, "frames": frames, "cost": sim_cost})
        print(f"  Query {qi}: WM={wm_costs[qi]:.4f}  Sim={sim_cost:.4f}")

    # ── Build GIF ──
    idx_tag = f"_idx{ds_idx}" if args.idx is not None else f"_idx{ds_idx}"
    out_name = f"planner_{args.env}{idx_tag}"
    out_path = core.build_gif(ds_frames, ds_frames[-1], sim_results,
                               horizon, env.FS, core.OUT / f"{out_name}.gif")

    # Also save a static final-frame comparison
    from PIL import Image
    final_row = [ds_frames[-1]] + [r["frames"][-1] for r in sim_results] + [ds_frames[-1]]
    ds_h, ds_w = ds_frames[0].shape[:2]
    final_parts = []
    for f in final_row:
        if f.shape[:2] != (ds_h, ds_w):
            f = np.array(Image.fromarray(f).resize((ds_w, ds_h)))
        final_parts.append(f)
    final_panel = np.hstack(final_parts)
    import imageio
    imageio.imwrite(core.OUT / f"{out_name}_final.png", final_panel)
    print(f"\nSaved: {out_path}")
    print(f"Saved: {core.OUT / f'{out_name}_final.png'}")

    if sim_env is not None:
        sim_env.close()


if __name__ == "__main__":
    main()
