"""LiberoGoalDataset: reads LIBERO-format HDF5 files and exposes the
stable_worldmodel ``Dataset`` API for World Model training.

LIBERO HDF5 layout (one file per task, 50 demos per file)::

    <file>.hdf5
      └── data/
          └── demo_0/
              ├── actions          (T, 7)   float64
              ├── dones            (T,)     uint8
              ├── rewards          (T,)     uint8
              ├── robot_states     (T, 9)   float64
              ├── states           (T, 79)  float64
              └── obs/
                  ├── agentview_rgb     (T, 128, 128, 3)  uint8  ← primary camera
                  ├── eye_in_hand_rgb   (T, 128, 128, 3)  uint8
                  ├── ee_ori            (T, 3)  float64
                  ├── ee_pos            (T, 3)  float64
                  ├── ee_states         (T, 6)  float64
                  ├── gripper_states    (T, 2)  float64
                  └── joint_states      (T, 7)  float64

Column mapping exposed to the training pipeline::

    pixels   ← obs/agentview_rgb          (HWC uint8 → CHW float32 tensor)
    action   ← actions                    (7-dim float64)
    proprio  ← concat(joint_states, gripper_states)  (9-dim float64)
    states   ← states                     (79-dim float64, optional)
    robot_states ← robot_states           (9-dim float64, optional)
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np
import torch

from stable_worldmodel.data.dataset import Dataset


def _demo_sort_key(name: str) -> int:
    """Sort demo_0, demo_1, ..., demo_49 naturally."""
    match = re.fullmatch(r"demo_(\d+)", name)
    return int(match.group(1)) if match else 0


class LiberoGoalDataset(Dataset):
    """Dataset that reads LIBERO-goal HDF5 files in their native format.

    Each file is expected to contain ``data/demo_*/`` groups with the
    LIBERO schema above.  All demos across all ``*.hdf5`` files in
    *data_dir* are concatenated into a flat episode list.

    Args:
        data_dir: Directory containing ``*.hdf5`` files.
        frameskip: Stride between observation samples.
        num_steps: Number of observation steps per sample.
        transform: Optional dict-in / dict-out transform applied per sample.
        keys_to_load: Which columns to expose.  Supported values:
            ``pixels``, ``action``, ``proprio``, ``states``, ``robot_states``.
        keys_to_cache: Columns to pre-load into RAM (saves HDF5 reads).
            ``pixels`` is never cached because image data is too large.
    """

    # Mapping from logical column name → (hdf5_path, is_obs_group)
    # is_obs_group=True means the path is relative to demo["obs"]/
    _COLUMN_SOURCES: dict[str, tuple[list[str], bool]] = {
        "pixels": (["obs", "agentview_rgb"], True),
        "action": (["actions"], False),
        "proprio": (["obs", "joint_states", "gripper_states"], True),
        "states": (["states"], False),
        "robot_states": (["robot_states"], False),
    }

    # Image-like columns — HWC → CHW permutation + dtype conversion
    _IMAGE_COLUMNS = {"pixels"}

    def __init__(
        self,
        data_dir: str | Path,
        frameskip: int = 5,
        num_steps: int = 1,
        transform: Callable[[dict], dict] | None = None,
        keys_to_load: list[str] | None = None,
        keys_to_cache: list[str] | None = None,
        keys_to_merge: dict[str, list[str] | str] | None = None,
        **__: object,  # absorb unknown kwargs from Hydra config
    ) -> None:
        self.data_dir = Path(data_dir)
        self._handles: dict[int, h5py.File] = {}
        self._cache: dict[str, np.ndarray] = {}

        # ---- discover files -------------------------------------------------
        self.files = sorted(self.data_dir.glob("*.hdf5"))
        if not self.files:
            raise FileNotFoundError(
                f"No .hdf5 files found in {self.data_dir}"
            )
        logging.info(
            f"LiberoGoalDataset: found {len(self.files)} HDF5 file(s) "
            f"in {self.data_dir}"
        )

        # ---- enumerate demos → (file_idx, demo_name, length) ----------------
        self._demo_index: list[tuple[int, str, int]] = []
        for file_idx, path in enumerate(self.files):
            with h5py.File(path, "r") as handle:
                demos = handle["data"]
                for demo_name in sorted(demos.keys(), key=_demo_sort_key):
                    length = int(demos[demo_name]["actions"].shape[0])
                    self._demo_index.append((file_idx, demo_name, length))

        lengths = np.array([d[2] for d in self._demo_index], dtype=np.int64)
        offsets = np.zeros(len(lengths), dtype=np.int64)
        if len(lengths) > 1:
            offsets[1:] = np.cumsum(lengths[:-1])

        self._keys = keys_to_load or ["pixels", "action", "proprio"]

        # ---- pre-cache small columns for normalizer fitting -----------------
        self._precache_columns(keys_to_cache or [])

        super().__init__(lengths, offsets, frameskip, num_steps, transform)

        # ---- keys_to_merge support ------------------------------------------
        if keys_to_merge:
            for target, source in keys_to_merge.items():
                self.merge_col(source, target)

    # ------------------------------------------------------------------
    # Column metadata
    # ------------------------------------------------------------------

    @property
    def column_names(self) -> list[str]:
        return list(self._keys)

    def get_dim(self, col: str) -> int:
        """Return the per-step dimensionality of *col*."""
        data = self.get_col_data(col)
        return int(np.prod(data.shape[1:])) if data.ndim > 1 else 1

    # ------------------------------------------------------------------
    # Bulk column access (used by normalizer fitting in train.py)
    # ------------------------------------------------------------------

    def get_col_data(self, col: str) -> np.ndarray:
        """Return a flat (N, *dims) array for *col* across all episodes."""
        if col in self._cache:
            return self._cache[col]
        return self._materialize_column(col)

    def get_row_data(self, row_idx: int | list[int]) -> dict:
        """Return raw HDF5 rows (used rarely, e.g. by some callbacks)."""
        # Not performance-critical; implemented for API completeness.
        indices = np.atleast_1d(np.asarray(row_idx, dtype=int))
        result: dict[str, list] = {col: [] for col in self._keys}
        for global_idx in indices:
            ep_idx, local_step = self._global_to_episode(global_idx)
            file_idx, demo_name, _length = self._demo_index[ep_idx]
            demo = self._get_handle(file_idx)["data"][demo_name]
            for col in self._keys:
                arr = self._read_column_raw(demo, col, local_step,
                                             local_step + 1, frameskip=1)
                result[col].append(arr[0] if len(arr) else arr)
        return {k: np.stack(v) for k, v in result.items()}

    def merge_col(
        self,
        source: list[str] | str,
        target: str,
        dim: int = -1,
    ) -> None:
        """Concatenate source columns into a new cached column *target*."""
        if isinstance(source, str):
            source = [k for k in self._keys if re.match(source, k)]
        merged = np.concatenate(
            [self._materialize_column(s) for s in source], axis=dim
        )
        self._cache[target] = merged
        if target not in self._keys:
            self._keys.append(target)
        logging.info(
            f"LiberoGoalDataset: merged {source} → '{target}' (cached)"
        )

    # ------------------------------------------------------------------
    # Core read path
    # ------------------------------------------------------------------

    def _load_slice(
        self, ep_idx: int, start: int, end: int
    ) -> dict[str, torch.Tensor]:
        """Read a contiguous slice of raw steps from one episode."""
        file_idx, demo_name, _length = self._demo_index[ep_idx]
        demo = self._get_handle(file_idx)["data"][demo_name]

        steps: dict[str, torch.Tensor] = {}
        for col in self._keys:
            fs = 1 if col == "action" else self.frameskip
            arr = self._read_column_raw(demo, col, start, end, frameskip=fs)

            if col in self._IMAGE_COLUMNS:
                # arr is (T, H, W, C) uint8 → (T, C, H, W) float32
                tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).float()
            else:
                tensor = torch.from_numpy(arr.astype(np.float32))

            steps[col] = tensor

        return self.transform(steps) if self.transform else steps

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_handle(self, file_idx: int) -> h5py.File:
        """Return an open HDF5 handle (lazy, cached per process)."""
        if file_idx not in self._handles:
            self._handles[file_idx] = h5py.File(
                self.files[file_idx], "r"
            )
        return self._handles[file_idx]

    def __getstate__(self) -> dict:
        """Clear file handles for pickle compatibility (DataLoader workers)."""
        state = self.__dict__.copy()
        state["_handles"] = {}
        return state

    def _read_column_raw(
        self,
        demo: h5py.Group,
        col: str,
        start: int,
        end: int,
        frameskip: int,
    ) -> np.ndarray:
        """Read *col* from *demo* at steps [start, end) with stride."""
        source_info = self._COLUMN_SOURCES.get(col)
        if source_info is None:
            raise KeyError(
                f"Unknown column '{col}'. Supported: "
                f"{list(self._COLUMN_SOURCES)}"
            )

        paths, _is_obs = source_info

        if col == "proprio":
            # Special: concatenate joint_states + gripper_states
            joint = demo["obs"]["joint_states"][start:end:frameskip]
            gripper = demo["obs"]["gripper_states"][start:end:frameskip]
            return np.concatenate(
                [np.asarray(joint, dtype=np.float32),
                 np.asarray(gripper, dtype=np.float32)],
                axis=-1,
            )

        # General case: navigate to the value
        current = demo
        for p in paths:
            current = current[p]
        data = current[start:end:frameskip]
        return np.asarray(data)

    def _precache_columns(self, keys: list[str]) -> None:
        """Read and cache small columns (everything except pixels)."""
        for col in keys:
            if col == "pixels":
                logging.warning(
                    "LiberoGoalDataset: refusing to cache 'pixels' "
                    "(too large). Skipping."
                )
                continue
            self._cache[col] = self._materialize_column(col)
            logging.info(
                f"LiberoGoalDataset: cached '{col}' "
                f"({self._cache[col].shape})"
            )

    def _materialize_column(self, col: str) -> np.ndarray:
        """Concatenate *col* across all demos into one flat array."""
        if col in self._cache:
            return self._cache[col]

        source_info = self._COLUMN_SOURCES.get(col)
        if source_info is None:
            raise KeyError(
                f"Unknown column '{col}'. Supported: "
                f"{list(self._COLUMN_SOURCES)}"
            )

        chunks: list[np.ndarray] = []
        for file_idx in range(len(self.files)):
            with h5py.File(self.files[file_idx], "r") as handle:
                demos = handle["data"]
                for demo_name in sorted(
                    demos.keys(), key=_demo_sort_key
                ):
                    demo = demos[demo_name]
                    if col == "proprio":
                        joint = np.asarray(
                            demo["obs"]["joint_states"][:],
                            dtype=np.float32,
                        )
                        gripper = np.asarray(
                            demo["obs"]["gripper_states"][:],
                            dtype=np.float32,
                        )
                        chunks.append(
                            np.concatenate([joint, gripper], axis=-1)
                        )
                    else:
                        paths, _is_obs = source_info
                        current = demo
                        for p in paths:
                            current = current[p]
                        chunks.append(np.asarray(current, dtype=np.float32))

        return np.concatenate(chunks, axis=0)

    def _global_to_episode(
        self, global_idx: int
    ) -> tuple[int, int]:
        """Convert a flat global step index → (ep_idx, local_step)."""
        offsets = self.offsets
        ep_idx = int(np.searchsorted(offsets, global_idx, side="right") - 1)
        local_step = int(global_idx - offsets[ep_idx])
        return ep_idx, local_step


__all__ = ["LiberoGoalDataset"]
