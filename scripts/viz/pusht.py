"""PushT simulator for planner visualization."""

import numpy as np
import gymnasium as gym

RAW_ACT = 2
FS = 5
CTX_LEN = 3
DATASET_NAME = "pusht_expert_train"


def make_env():
    """Create a PushT env instance."""
    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    env.reset()
    env.render()  # warm up
    return env


def run_sim(env, state, actions_2d):
    """Run actions in PushT from a given state. Returns frames."""
    env.reset()
    env.unwrapped._set_state(state)
    env.step(np.zeros(RAW_ACT, dtype=np.float32))
    frames = [env.render()]
    for a in actions_2d:
        env.step(np.clip(a.astype(np.float32), -1, 1))
        frames.append(env.render())
    return frames


def get_state(ds_raw, idx):
    """Extract PushT state at the last context frame. Returns (state_array,)."""
    base = ds_raw
    ep, local_start = base.clip_indices[idx]
    start_row = base.offsets[ep] + local_start
    state_row = start_row + (CTX_LEN - 1) * FS
    state = base.get_row_data(state_row)["state"]
    if state.ndim > 1:
        state = state[0]
    return (np.array(state).astype(np.float64),)


def get_gt_actions(item_raw, horizon):
    """Extract GT actions from dataset item."""
    return item_raw["action"][CTX_LEN - 1: CTX_LEN - 1 + horizon].numpy()
