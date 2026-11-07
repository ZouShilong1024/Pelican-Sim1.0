# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RealSource-World dataset with action value (+ optional action video).

Extends the LeRobot RSW layout used by ``RealSourceWorldDataset`` and adds:

  - parquet ``observation.state`` (follower) → ``action`` / ``action_state`` /
    ``action_delta`` (legacy key name ``action`` kept for the model interface)
  - optional ``action_video/`` mp4 aligned to the same frame indices

Values come from the first 17 dims of 71-D ``observation.state`` (not parquet
``action`` / leader commands), matching action_video FK rendering::

    [0:7]   LeftFollowerArm_Joint1..7.pos
    [7]     LeftGripper.pos
    [8:15]  RightFollowerArm_Joint1..7.pos
    [15]    RightGripper.pos
    [16]    Lift.position

Default training layout (``include_gripper=True``, ``include_lift=False``) is
the canonical **28-D** unified action (dexhand slots zero-filled)::

    left arm (7) + left gripper (1) + left dexhand (6)
    + right arm (7) + right gripper (1) + right dexhand (6)
"""

from __future__ import annotations

import json
import os
import traceback
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from decord import VideoReader, cpu
from torch.utils.data import Dataset
from torchvision import transforms as T

from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.predict2.datasets.local_datasets.action_unified_layout import (
    UNIFIED_ACTION_DIM,
    UNIFIED_ACTION_NAMES,
    expand_legacy16_to_unified,
)
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_real_source_world import (
    DEFAULT_CAMERA_KEY,
    TARGET_FPS,
    RealSourceWorldDataset,
)
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_utils import (
    ResizePreprocess,
    ToTensorVideo,
    is_excluded_video_path,
    is_too_short_for_window,
    load_exclude_video_paths,
    parallel_path_exists,
    sample_video_frame_indices,
)

# Canonical RSW pose slice from ``observation.state`` (first 17 of 71 dims).
# Index layout matches historical parquet ``action`` (leader), but values are
# follower joint positions used by the real robot / action_video renders.
RAW_ACTION_NAMES: tuple[str, ...] = (
    "LeftFollowerArm_Joint1.pos",
    "LeftFollowerArm_Joint2.pos",
    "LeftFollowerArm_Joint3.pos",
    "LeftFollowerArm_Joint4.pos",
    "LeftFollowerArm_Joint5.pos",
    "LeftFollowerArm_Joint6.pos",
    "LeftFollowerArm_Joint7.pos",
    "LeftGripper.pos",
    "RightFollowerArm_Joint1.pos",
    "RightFollowerArm_Joint2.pos",
    "RightFollowerArm_Joint3.pos",
    "RightFollowerArm_Joint4.pos",
    "RightFollowerArm_Joint5.pos",
    "RightFollowerArm_Joint6.pos",
    "RightFollowerArm_Joint7.pos",
    "RightGripper.pos",
    "Lift.position",
)

# Legacy RSW pose slice (16) before dexhand zero-pad → unified 28-D.
DEFAULT_LEGACY_ACTION_NAMES: tuple[str, ...] = RAW_ACTION_NAMES[:16]
DEFAULT_ACTION_NAMES: tuple[str, ...] = UNIFIED_ACTION_NAMES
DEFAULT_ACTION_DIM = UNIFIED_ACTION_DIM  # 28
DEFAULT_RAW_ACTION_DIM = len(RAW_ACTION_NAMES)  # 17 (pose slice width)
DEFAULT_ACTION_COLUMN = "observation.state"

# Align with OminiEWM ``wan22_5b_vace_action_robotwin_121_640_big2.py``:
# scale per-frame deltas so arm / gripper p95 land near ~0.3.
DEFAULT_ARM_DELTA_SCALER = 5.0
DEFAULT_GRIPPER_DELTA_SCALER = 4.0
DEFAULT_DEXHAND_DELTA_SCALER = 4.0
DEFAULT_LIFT_DELTA_SCALER = 1.0


def resolve_action_indices(
    *,
    include_gripper: bool = True,
    include_lift: bool = False,
) -> tuple[list[int], tuple[str, ...]]:
    """Map RSW ``observation.state`` pose-slice columns → legacy 16-D indices / names.

    Dexhand slots are not present in RSW parquet; they are zero-filled later via
    ``expand_legacy16_to_unified``.
    """
    indices: list[int] = list(range(0, 7))  # left arm
    if include_gripper:
        indices.append(7)
    indices.extend(range(8, 15))  # right arm
    if include_gripper:
        indices.append(15)
    if include_lift:
        indices.append(16)
    names = tuple(RAW_ACTION_NAMES[i] for i in indices)
    return indices, names


def build_action_delta_scaler(
    *,
    include_gripper: bool = True,
    include_lift: bool = False,
    include_dexhand: bool = True,
    arm_scaler: float = DEFAULT_ARM_DELTA_SCALER,
    gripper_scaler: float = DEFAULT_GRIPPER_DELTA_SCALER,
    dexhand_scaler: float = DEFAULT_DEXHAND_DELTA_SCALER,
    lift_scaler: float = DEFAULT_LIFT_DELTA_SCALER,
) -> list[float]:
    """Per-dim delta scalers for the unified 28-D action layout.

    Layout (default)::

        left_arm(7) * arm + left_gripper(1) * gripper + left_dexhand(6) * dex
        + right_arm(7) * arm + right_gripper(1) * gripper + right_dexhand(6) * dex

    Mirrors OminiEWM RoboTwin big2 (arm=5.0, gripper=4.0); dexhand uses the
    gripper scale by default. ``include_lift`` is kept for API compatibility
    but is unused in the canonical 28-D layout (no lift slot).
    """
    from cosmos_predict2._src.predict2.datasets.local_datasets.action_unified_layout import (
        UNIFIED_ARM_DOF,
        UNIFIED_DEXHAND_DOF,
    )

    def _one_side() -> list[float]:
        side = [float(arm_scaler)] * int(UNIFIED_ARM_DOF)
        if include_gripper:
            side.append(float(gripper_scaler))
        if include_dexhand:
            side.extend([float(dexhand_scaler)] * int(UNIFIED_DEXHAND_DOF))
        return side

    scaler = _one_side() + _one_side()
    if include_lift:
        # Legacy flag: append a trailing lift scale (not used by current datasets).
        scaler.append(float(lift_scaler))
    return scaler


def normalize_action_delta_scaler(
    action_delta_scaler: Optional[Union[float, Sequence[float]]],
    action_dim: int,
) -> Optional[torch.Tensor]:
    """Validate and convert scaler to ``float32`` tensor ``[action_dim]``."""
    if action_delta_scaler is None:
        return None
    if isinstance(action_delta_scaler, (int, float)):
        return torch.full((action_dim,), float(action_delta_scaler), dtype=torch.float32)
    scaler = torch.tensor(list(action_delta_scaler), dtype=torch.float32)
    if scaler.ndim != 1 or scaler.numel() != action_dim:
        raise ValueError(
            f"action_delta_scaler length ({scaler.numel()}) must equal action_dim ({action_dim}), "
            "or pass a scalar."
        )
    return scaler


def verify_action_triple(
    action: torch.Tensor,
    action_state: torch.Tensor,
    action_delta: torch.Tensor,
    action_delta_scaler: Optional[Union[float, Sequence[float], torch.Tensor]] = None,
    *,
    atol: float = 1e-5,
) -> None:
    """Assert Omini-style consistency between absolute action / state / scaled delta.

    Accepts unbatched ``[T,D]`` / ``[D]`` / ``[T-1,D]`` or batched
    ``[B,T,D]`` / ``[B,D]`` / ``[B,T-1,D]``.
    """
    if action.ndim == 2:
        action = action.unsqueeze(0)
        action_state = action_state.unsqueeze(0) if action_state.ndim == 1 else action_state
        action_delta = action_delta.unsqueeze(0) if action_delta.ndim == 2 else action_delta
    if action.ndim != 3 or action_state.ndim != 2 or action_delta.ndim != 3:
        raise ValueError(
            f"Bad shapes: action={tuple(action.shape)}, "
            f"action_state={tuple(action_state.shape)}, action_delta={tuple(action_delta.shape)}"
        )
    if not torch.allclose(action_state.float(), action[:, 0, :].float(), atol=atol):
        err = (action_state.float() - action[:, 0, :].float()).abs().max().item()
        raise AssertionError(f"action_state != action[:,0] (max abs err={err})")

    raw_delta = action[:, 1:, :] - action[:, :-1, :]
    if action_delta_scaler is None:
        expected = raw_delta
    else:
        scaler = normalize_action_delta_scaler(action_delta_scaler, action.shape[-1])
        assert scaler is not None
        expected = raw_delta * scaler.to(device=raw_delta.device, dtype=raw_delta.dtype)

    if not torch.allclose(action_delta.float(), expected.float(), atol=atol):
        err = (action_delta.float() - expected.float()).abs().max().item()
        # Common mistake: comparing scaled delta to unscaled raw diffs.
        raw_err = (action_delta.float() - raw_delta.float()).abs().max().item()
        raise AssertionError(
            f"action_delta != (action[:,1:]-action[:,:-1])*scaler (max abs err={err}); "
            f"vs unscaled raw max abs err={raw_err}. "
            f"Remember action_delta is scaler-applied."
        )


def action_video_path_from_rgb(video_path: str) -> str:
    """Map RGB video path to the sibling action_video render.

    RealSource-World renders use a flat layout (no LeRobot chunk dir)::

        .../videos/chunk-XXX/<camera>/episode_NNNNNN.mp4
        → .../action_video/<camera>/episode_NNNNNN.mp4
    """
    norm = video_path.replace("\\", "/")
    marker = "/videos/"
    if marker not in norm:
        raise ValueError(f"Cannot derive action_video path from {video_path!r}")

    prefix, rest = norm.split(marker, 1)
    parts = rest.split("/")
    # Drop ``chunk-XXX`` so RSW flat renders resolve correctly.
    if len(parts) >= 3 and parts[0].startswith("chunk-"):
        rel = "/".join(parts[1:])
    else:
        rel = rest
    out = f"{prefix}/action_video/{rel}"
    if os.sep != "/" and os.sep in video_path:
        out = out.replace("/", os.sep)
    return out


def parquet_path_for_episode(
    task_dir: str,
    episode_index: int,
    *,
    chunks_size: int,
    data_path_template: str,
) -> str:
    episode_chunk = int(episode_index) // int(chunks_size)
    rel = data_path_template.format(
        episode_chunk=episode_chunk,
        episode_index=int(episode_index),
    )
    return os.path.join(task_dir, rel)


def _list_column_to_numpy(column: pa.ChunkedArray | pa.Array) -> np.ndarray:
    """Convert a parquet list<float> / fixed-size list column to ``[T, D]`` float32."""
    values = column.to_pylist()
    if not values:
        return np.zeros((0, 0), dtype=np.float32)
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        # Scalar column — unexpected for RSW action.
        arr = arr.reshape(-1, 1)
    return arr


@dataclass(frozen=True)
class _ActionEpisodeSample:
    video_path: str
    action_video_path: str
    parquet_path: str
    caption: str
    length: int
    task_name: str
    episode_index: int


class RealSourceWorldActionDataset(RealSourceWorldDataset):
    """RSW video + action value (+ optional action video) for action-inject training.

    Temporal sampling (``temporal_stride``, default 4)::

        1. Sample a contiguous source window of ``num_frames * temporal_stride`` frames
           (or linspace that many when ``uniform_sample_episode=True``).
        2. Keep every ``temporal_stride``-th frame → exactly ``num_frames`` for the model.
        3. The **same** kept indices are used for RGB, ``observation.state``, and action_video.

    Output keys (in addition to ``RealSourceWorldDataset``)::

        action          float [T, action_dim]   absolute sequence (unscaled, cloned)
        action_state    float [action_dim]      frame-0 absolute state (unscaled, cloned)
        action_delta    float [T-1, action_dim] per-frame deltas (**scaled** if configured)
        action_video    uint8 [C, T, H, W]      only if ``load_action_video=True``

    Consistency (see ``verify_action_triple``)::

        action_state == action[0]
        action_delta == (action[1:] - action[:-1]) * action_delta_scaler

    ``action_delta_scaler`` multiplies **deltas only**. Pass ``None`` to disable.
    Checking ``action_delta == action[1:] - action[:-1]`` will fail when scaler != 1.
    """

    def __init__(
        self,
        dataset_dir: str,
        num_frames: int,
        video_size: tuple[int, int],
        camera_key: str = DEFAULT_CAMERA_KEY,
        verify_files: bool = True,
        target_fps: float = TARGET_FPS,
        uniform_sample_episode: bool = False,
        start_from_first_frame_prob: float = 0.0,
        exclude_video_paths: Optional[Union[str, list[str]]] = None,
        *,
        load_action_video: bool = True,
        include_gripper: bool = True,
        include_lift: bool = False,
        action_column: str = DEFAULT_ACTION_COLUMN,
        raw_action_dim: int = DEFAULT_RAW_ACTION_DIM,
        require_action_video: Optional[bool] = None,
        action_delta_scaler: Optional[Union[float, Sequence[float]]] = None,
        temporal_stride: int = 4,
    ) -> None:
        # Re-implement init (parent builds a video-only index).
        Dataset.__init__(self)
        self.dataset_dir = dataset_dir
        self.sequence_length = int(num_frames)
        self.video_size = tuple(video_size)
        self.camera_key = camera_key
        self.verify_files = verify_files
        self.target_fps = float(target_fps)
        self.uniform_sample_episode = bool(uniform_sample_episode)
        self.start_from_first_frame_prob = float(start_from_first_frame_prob)
        self.exclude_video_paths = load_exclude_video_paths(exclude_video_paths)
        self.num_failed_loads = 0

        self.load_action_video = bool(load_action_video)
        self.include_gripper = bool(include_gripper)
        self.include_lift = bool(include_lift)
        self.action_column = action_column
        self.raw_action_dim = int(raw_action_dim)
        self.require_action_video = (
            bool(require_action_video)
            if require_action_video is not None
            else bool(load_action_video)
        )
        if int(temporal_stride) < 1:
            raise ValueError(f"temporal_stride must be >= 1, got {temporal_stride}")
        self.temporal_stride = int(temporal_stride)
        # Source window length before stride subsample (num_frames * 4 by default).
        self.raw_sequence_length = self.sequence_length * self.temporal_stride

        # Select legacy 16-D pose slice from parquet, then expand to unified 28-D.
        self.action_indices, _legacy_names = resolve_action_indices(
            include_gripper=self.include_gripper,
            include_lift=self.include_lift,
        )
        self.action_dim = UNIFIED_ACTION_DIM
        self.action_names = UNIFIED_ACTION_NAMES
        self.action_delta_scaler = normalize_action_delta_scaler(action_delta_scaler, self.action_dim)

        if not os.path.isdir(self.dataset_dir):
            raise FileNotFoundError(f"RealSource-World dataset dir not found: {self.dataset_dir}")

        self.samples = self._build_index()
        if not self.samples:
            raise RuntimeError(
                f"No valid action episodes under {self.dataset_dir} "
                f"(camera_key={self.camera_key!r}, num_frames={self.sequence_length}, "
                f"raw_sequence_length={self.raw_sequence_length}, "
                f"load_action_video={self.load_action_video})."
            )

        self.video_paths = [s.video_path for s in self.samples]
        scaler_msg = (
            "None"
            if self.action_delta_scaler is None
            else [round(float(x), 4) for x in self.action_delta_scaler.tolist()]
        )
        log.info(
            f"RealSourceWorldActionDataset: {len(self.samples)} episodes from {self.dataset_dir} "
            f"(camera={self.camera_key}, num_frames={self.sequence_length}, "
            f"temporal_stride={self.temporal_stride}, raw_sequence_length={self.raw_sequence_length}, "
            f"video_size={self.video_size}, action_dim={self.action_dim}, "
            f"action_column={self.action_column!r}, "
            f"action_names={list(self.action_names)}, "
            f"action_delta_scaler={scaler_msg}, "
            f"load_action_video={self.load_action_video}, "
            f"require_action_video={self.require_action_video}, "
            f"verify_files={self.verify_files})"
        )

        self.preprocess = T.Compose(
            [ToTensorVideo(), ResizePreprocess((self.video_size[0], self.video_size[1]))]
        )

    def __str__(self) -> str:
        return f"{len(self.samples)} action samples from {self.dataset_dir}"

    def __len__(self) -> int:
        return len(self.samples)

    def _load_task_info(self, task_dir: str) -> tuple[int, str]:
        info_path = os.path.join(task_dir, "meta", "info.json")
        chunks_size = 1000
        data_path_template = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
        if os.path.isfile(info_path):
            with open(info_path, "r", encoding="utf-8") as f:
                info = json.load(f)
            chunks_size = int(info.get("chunks_size") or chunks_size)
            data_path_template = str(info.get("data_path") or data_path_template)
        return chunks_size, data_path_template

    def _build_index(self) -> list[_ActionEpisodeSample]:
        samples: list[_ActionEpisodeSample] = []
        skipped_missing = 0
        skipped_short = 0

        for task_dir in self._iter_task_dirs():
            task_name = os.path.basename(task_dir.rstrip(os.sep))
            chunks_size, data_path_template = self._load_task_info(task_dir)
            episodes_path = os.path.join(task_dir, "meta", "episodes.jsonl")
            with open(episodes_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    episode = json.loads(line)
                    length = int(episode.get("length", 0))
                    episode_index = int(episode.get("episode_index", -1))

                    videos = episode.get("videos") or {}
                    rel_video = videos.get(self.camera_key)
                    if not rel_video:
                        skipped_missing += 1
                        continue

                    video_path = os.path.join(task_dir, rel_video)
                    if is_excluded_video_path(video_path, self.exclude_video_paths):
                        skipped_missing += 1
                        continue
                    # Contiguous window must fit the denser raw window before stride subsample.
                    if is_too_short_for_window(
                        length, self.raw_sequence_length, self.uniform_sample_episode
                    ):
                        skipped_short += 1
                        continue

                    if episode_index < 0:
                        skipped_missing += 1
                        continue

                    parquet_path = parquet_path_for_episode(
                        task_dir,
                        episode_index,
                        chunks_size=chunks_size,
                        data_path_template=data_path_template,
                    )
                    try:
                        action_video_path = action_video_path_from_rgb(video_path)
                    except ValueError:
                        action_video_path = ""

                    tasks = episode.get("tasks") or []
                    if isinstance(tasks, str):
                        caption = tasks.strip()
                    else:
                        caption = ". ".join(str(t).strip() for t in tasks if str(t).strip())
                    if not caption:
                        caption = task_name.replace("_", " ")

                    samples.append(
                        _ActionEpisodeSample(
                            video_path=video_path,
                            action_video_path=action_video_path,
                            parquet_path=parquet_path,
                            caption=caption,
                            length=length,
                            task_name=task_name,
                            episode_index=episode_index,
                        )
                    )

        if self.verify_files and samples:
            paths_to_check: list[str] = []
            for s in samples:
                paths_to_check.append(s.video_path)
                paths_to_check.append(s.parquet_path)
                if self.require_action_video and s.action_video_path:
                    paths_to_check.append(s.action_video_path)

            exists = parallel_path_exists(paths_to_check)
            kept: list[_ActionEpisodeSample] = []
            cursor = 0
            before = len(samples)
            for s in samples:
                ok_video = exists[cursor]
                ok_parquet = exists[cursor + 1]
                cursor += 2
                ok_av = True
                if self.require_action_video and s.action_video_path:
                    ok_av = exists[cursor]
                    cursor += 1
                if ok_video and ok_parquet and ok_av:
                    kept.append(s)
            samples = kept
            skipped_missing += before - len(samples)

        log.info(
            f"Indexed {len(samples)} action episodes "
            f"(skipped_missing={skipped_missing}, skipped_short={skipped_short}, "
            f"verify_files={self.verify_files}, require_action_video={self.require_action_video})"
        )
        return samples

    def _sample_frame_ids(
        self, total_frames: int, src_fps: Optional[float]
    ) -> tuple[list[int], float]:
        """Sample denser raw window then keep every ``temporal_stride``-th index.

        Returns kept indices of length ``sequence_length`` (model frames) and the
        effective FPS after striding (``src_fps / temporal_stride`` when known).
        """
        raw_ids, reported_fps = sample_video_frame_indices(
            total_frames,
            self.raw_sequence_length,
            uniform_sample_episode=self.uniform_sample_episode,
            src_fps=src_fps,
            target_fps=self.target_fps,
            start_from_first_frame_prob=self.start_from_first_frame_prob,
        )
        if len(raw_ids) != self.raw_sequence_length:
            raise RuntimeError(
                f"Expected raw window of {self.raw_sequence_length} ids, got {len(raw_ids)}"
            )
        frame_ids = raw_ids[:: self.temporal_stride]
        if len(frame_ids) != self.sequence_length:
            raise RuntimeError(
                f"After temporal_stride={self.temporal_stride}, expected "
                f"{self.sequence_length} ids, got {len(frame_ids)} "
                f"(raw={len(raw_ids)})"
            )
        # Effective FPS seen by the model after keeping 1 of every stride frames.
        if self.temporal_stride > 1:
            reported_fps = float(reported_fps) / float(self.temporal_stride)
        return frame_ids, reported_fps

    def _load_rgb_with_ids(self, video_path: str) -> tuple[torch.Tensor, list[int], float]:
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file does not exist: {video_path}")
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        total_frames = len(vr)
        src_fps = self._read_fps(vr)
        frame_ids, reported_fps = self._sample_frame_ids(total_frames, src_fps)
        frame_data = vr.get_batch(frame_ids).asnumpy().astype(np.uint8)
        vr.seek(0)
        del vr

        frames = torch.from_numpy(frame_data).permute(0, 3, 1, 2)  # [T, C, H, W]
        frames = self.preprocess(frames)
        frames = torch.clamp(frames * 255.0, 0, 255).to(torch.uint8)
        return frames, frame_ids, reported_fps

    def _load_action_video_with_ids(
        self, action_video_path: str, frame_ids: Sequence[int]
    ) -> torch.Tensor:
        if not os.path.isfile(action_video_path):
            raise FileNotFoundError(f"Action video does not exist: {action_video_path}")
        vr = VideoReader(action_video_path, ctx=cpu(0), num_threads=2)
        total_frames = len(vr)
        # Clamp / remap if action_video is shorter (should match RGB in RSW).
        safe_ids = [min(int(i), total_frames - 1) for i in frame_ids]
        frame_data = vr.get_batch(safe_ids).asnumpy().astype(np.uint8)
        vr.seek(0)
        del vr

        frames = torch.from_numpy(frame_data).permute(0, 3, 1, 2)
        frames = self.preprocess(frames)
        frames = torch.clamp(frames * 255.0, 0, 255).to(torch.uint8)
        return frames

    def _load_actions(self, parquet_path: str, frame_ids: Sequence[int]) -> torch.Tensor:
        if not os.path.isfile(parquet_path):
            raise FileNotFoundError(f"Parquet file does not exist: {parquet_path}")
        table = pq.read_table(parquet_path, columns=[self.action_column])
        actions = _list_column_to_numpy(table.column(self.action_column))  # [T, D_raw]
        if actions.ndim != 2:
            raise ValueError(f"Expected action matrix [T, D], got shape {actions.shape} in {parquet_path}")
        if actions.shape[1] < self.raw_action_dim:
            raise ValueError(
                f"Action dim {actions.shape[1]} < raw_action_dim={self.raw_action_dim} in {parquet_path}"
            )

        n = actions.shape[0]
        safe_ids = [min(max(int(i), 0), n - 1) for i in frame_ids]
        selected16 = actions[safe_ids][:, self.action_indices]  # [T, 16]
        unified = expand_legacy16_to_unified(selected16)  # [T, 28]
        return torch.from_numpy(np.ascontiguousarray(unified)).float()

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        try:
            video_thw, frame_ids, fps = self._load_rgb_with_ids(sample.video_path)
            video = video_thw.permute(1, 0, 2, 3)  # [C, T, H, W]
            _, _, h, w = video.shape

            action = self._load_actions(sample.parquet_path, frame_ids)  # [T, D]
            if action.shape[0] != self.sequence_length:
                raise RuntimeError(
                    f"Action length {action.shape[0]} != num_frames {self.sequence_length}"
                )
            if action.shape[1] != self.action_dim:
                raise RuntimeError(f"Action dim {action.shape[1]} != {self.action_dim}")

            action_state = action[0]
            if action.shape[0] > 1:
                action_delta = action[1:] - action[:-1]
            else:
                action_delta = torch.zeros(1, self.action_dim, dtype=action.dtype)
            # OminiEWM delta_chunk: scale deltas only; leave absolute state untouched.
            # Invariant (batch dim optional)::
            #   action_state == action[0]
            #   action_delta == (action[1:] - action[:-1]) * action_delta_scaler
            # Do NOT expect action_delta == action[1:] - action[:-1] when scaler != 1.
            if self.action_delta_scaler is not None:
                action_delta = action_delta * self.action_delta_scaler.to(
                    dtype=action_delta.dtype, device=action_delta.device
                )

            out: dict[str, Any] = {
                "video": video,
                "ai_caption": sample.caption,
                "fps": fps,
                "image_size": torch.tensor([h, w, h, w]),
                "num_frames": self.sequence_length,
                "padding_mask": torch.zeros(1, h, w),
                "action": action,
                "action_state": action_state,
                "action_delta": action_delta,
            }

            if self.load_action_video:
                if not sample.action_video_path:
                    raise FileNotFoundError(
                        f"Missing action_video path for {sample.video_path}"
                    )
                action_video_thw = self._load_action_video_with_ids(
                    sample.action_video_path, frame_ids
                )
                out["action_video"] = action_video_thw.permute(1, 0, 2, 3)

            return out
        except Exception as e:
            self.num_failed_loads += 1
            log.warning(
                f"Failed to load action sample {sample.video_path} "
                f"(total failures: {self.num_failed_loads}): {e}\n{traceback.format_exc()}",
                rank0_only=False,
            )
            return self[int(np.random.randint(len(self.samples)))]

    @classmethod
    def render_action_video(cls, request: Any) -> np.ndarray:
        """White-bg skeleton ``[T, H, W, 3]`` for one RealSource episode."""
        import json
        import os
        from pathlib import Path

        from cosmos_predict2._src.predict2.datasets.local_datasets.action_video_render import (
            BACKEND_ROOT,
            bgr_to_rgb,
            collect_frames,
            ensure_backend_import_path,
        )

        ensure_backend_import_path(BACKEND_ROOT / "realsource")
        import batch_render_action_video_realsource as rsw

        extra = dict(request.extra or {})
        parquet = request.parquet_path
        joints = extra.get("joints")
        if joints is None and request.action_path and os.path.isfile(request.action_path):
            from cosmos_predict2._src.predict2.datasets.local_datasets.action_unified_layout import (
                split_unified_bimanual_action,
            )

            parts = split_unified_bimanual_action(np.load(request.action_path))
            joints = np.concatenate(
                [
                    parts["left_arm"],
                    parts["left_gripper"][:, None],
                    parts["right_arm"],
                    parts["right_gripper"][:, None],
                ],
                axis=1,
            )
        if (not parquet or not os.path.isfile(parquet)) and joints is None:
            raise FileNotFoundError(
                f"RealSource needs action.npy or parquet for online AV: {request.action_path!r} / {parquet!r}"
            )
        cam_entry = extra.get("camera_entry")
        if cam_entry is None:
            task_dir = None
            for raw in (parquet, request.rgb_path):
                if not raw:
                    continue
                p = Path(raw).resolve()
                for parent in [p.parent, *p.parents]:
                    if (parent / "meta" / "camera.json").is_file():
                        task_dir = parent
                        break
                if task_dir is not None:
                    break
            if task_dir is None:
                raise FileNotFoundError(f"cannot find meta/camera.json above {parquet}")
            camera = json.loads((task_dir / "meta" / "camera.json").read_text(encoding="utf-8"))
            stem = Path(parquet).stem
            cam_entry = camera.get(stem)
            if cam_entry is None:
                ei = stem.replace("episode_", "")
                try:
                    cam_entry = camera.get(f"episode_{int(ei):06d}") or camera.get(ei)
                except ValueError:
                    cam_entry = None
            if cam_entry is None:
                raise KeyError(f"no camera entry {stem} in {task_dir / 'meta' / 'camera.json'}")

        urdf = BACKEND_ROOT / "realsource" / "RS-02.urdf"
        if not urdf.is_file():
            env = os.environ.get("RSW_URDF", "")
            if env and os.path.isfile(env):
                urdf = Path(env)
            else:
                raise FileNotFoundError(f"RealSource URDF not found: {urdf}")
        rsw._init_worker(str(urdf), request.width, request.height, "veryfast", 23, 1, True, True, True)
        task = {
            "parquet_path": parquet,
            "video_path": request.rgb_path,
            "camera_entry": cam_entry,
            "max_frames": int(extra.get("max_frames") or 0),
            "fps": 30.0,
            "joints": joints,
        }
        return bgr_to_rgb(collect_frames(rsw._render_episode_frames(task, overlay=False)))
