"""Cube (OGBCube) simulator for planner visualization."""

import numpy as np
import mujoco
import gymnasium as gym
import stable_worldmodel.envs  # registers swm/OGBCube-v0

RAW_ACT = 5
FS = 5
CTX_LEN = 3
DATASET_NAME = "cube_single_expert"


def make_env():
    """Create and warm up a Cube env instance."""
    env = gym.make("swm/OGBCube-v0", render_mode="rgb_array")
    env.unwrapped._render_goal = False  # don't render ghost goal markers
    env.reset()
    env.render()  # warm up
    return env


def set_mujoco_state(env, qpos, qvel):
    """Set env to a specific MuJoCo state and hide target markers."""
    env.reset()
    env.unwrapped.data.qpos[:] = qpos
    env.unwrapped.data.qvel[:] = qvel
    _hide_targets(env)
    mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)


def _hide_targets(env):
    """Completely hide target geoms and mocap bodies."""
    model = env.unwrapped.model
    data = env.unwrapped.data
    # Hide target geoms: zero alpha, zero size, zero contact
    for geom_ids in getattr(env.unwrapped, '_cube_target_geom_ids_list', []):
        for gid in geom_ids:
            g = model.geom(gid)
            g.rgba[3] = 0.0
            g.contype = 0
            g.conaffinity = 0
            if hasattr(g, 'size'):
                g.size[:] = 0.0
    # Move mocap bodies far below floor
    for mocap_id in getattr(env.unwrapped, '_cube_target_mocap_ids', []):
        data.mocap_pos[mocap_id] = [0, 0, -999]


def run_sim(env, qpos, qvel, actions_2d):
    """Run actions from a specific state. Returns frames."""
    set_mujoco_state(env, qpos, qvel)
    env.step(np.zeros(RAW_ACT, dtype=np.float32))  # init internal env state
    frames = [_render_no_targets(env)]
    for a in actions_2d:
        env.step(np.clip(a.astype(np.float32), -1, 1))
        frames.append(_render_no_targets(env))
    return frames


def _render_no_targets(env):
    """Render frame with target geoms hidden."""
    env.unwrapped._render_goal = False
    _hide_targets(env)
    mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)
    return env.render()


def get_state(ds_raw, idx):
    """Get (qpos, qvel) at the last context frame, with cube pos/rot corrected."""
    base = ds_raw
    ep, local_start = base.clip_indices[idx]
    start_row = base.offsets[ep] + local_start
    state_row = start_row + (CTX_LEN - 1) * FS

    qpos = base.get_row_data(state_row)["qpos"].copy()
    qvel = base.get_row_data(state_row)["qvel"].copy()
    if qpos.ndim > 1: qpos = qpos[0]
    if qvel.ndim > 1: qvel = qvel[0]

    try:
        block_pos = base.get_row_data(state_row).get("privileged_block_0_pos")
        block_quat = base.get_row_data(state_row).get("privileged_block_0_quat")
        if block_pos is not None:
            if block_pos.ndim > 1: block_pos = block_pos[0]
            qpos[6:9] = np.array(block_pos, dtype=np.float64)
        if block_quat is not None:
            if block_quat.ndim > 1: block_quat = block_quat[0]
            bq = np.array(block_quat, dtype=np.float64)
            qpos[9] = bq[3]; qpos[10] = bq[0]; qpos[11] = bq[1]; qpos[12] = bq[2]
    except Exception:
        pass

    return np.array(qpos).astype(np.float64), np.array(qvel).astype(np.float64)


def get_gt_actions(item_raw, horizon):
    """Extract GT actions from dataset item."""
    return item_raw["action"][CTX_LEN - 1: CTX_LEN - 1 + horizon].numpy()
