# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-episode action-video dataset for fine-tuning, driven by an ``episodes.jsonl`` manifest.

One line per episode (written by ``scripts/prepare_robodojo.py``)::

    {
      "rgb": "/data/robodojo_ft/ep000/rgb.mp4",                   # camera video at the source fps
      "action_video": "/data/robodojo_ft/ep000/action_video.mp4", # skeleton video, frame-aligned with rgb
      "action": "/data/robodojo_ft/ep000/action.npy",             # [T, 28] unified layout
      "lang": "fold the towel",
      "frame_nums": 462,
      "fps": 25.0,
      "temporal_stride": 2,      # optional, default = dataset temporal_stride
      "start_frame": 0,          # optional, first usable frame
      "task_name": "fold_towel", # optional
      "views": {                 # multi-view manifests only (prepare_robodojo.py --multiview)
        "cam_left_wrist":  {"rgb": ".../rgb_cam_left_wrist.mp4",  "action_video": ".../action_video_cam_left_wrist.mp4"},
        "cam_right_wrist": {"rgb": ".../rgb_cam_right_wrist.mp4", "action_video": ".../action_video_cam_right_wrist.mp4"}
      }
    }

Action row t is the joint state when RGB frame t was captured; the action video is rendered from
the same rows. Windows of ``num_frames`` model frames are sampled every ``temporal_stride`` source
frames; an episode too short for its stride falls back to the largest stride in
``[min_temporal_stride, stride]`` that fits. With ``temporal_stride_choices`` every sample draws
its stride from the choices that fit the episode (weights ``temporal_stride_probs``).

With ``views`` (``MULTIVIEW_VIEWS``), ``rgb`` / ``action_video`` are the head camera and each
sample packs head + both wrists onto one canvas with ``compose_multiview``.
"""

from __future__ import annotations

import json
import os
from typing import Optional, Sequence, Union

import numpy as np
import torch

from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.predict2.datasets.local_datasets.action_unified_layout import (
    UNIFIED_ACTION_DIM,
    UNIFIED_ARM_DOF,
)
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_robotwin_action import (
    DEFAULT_TARGET_FPS,
    RobotwinActionDataset,
    _RobotwinActionSample,
)
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_robodojo_action import (
    NPZ_ACTION_KEY,
    RoboDojoActionDataset,
    compose_multiview,
)
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_utils import parallel_path_exists


class EpisodeActionDataset(RoboDojoActionDataset):
    def __init__(
        self,
        episodes_jsonl: str,
        num_frames: int = 93,
        video_size: tuple[int, int] = (480, 640),
        verify_files: bool = True,
        target_fps: float = DEFAULT_TARGET_FPS,
        start_from_first_frame_prob: float = 0.0,
        *,
        include_gripper: bool = True,
        action_delta_scaler: Optional[Union[float, Sequence[float]]] = None,
        temporal_stride: int = 2,
        min_temporal_stride: int = 1,
        temporal_stride_choices: Optional[Sequence[int]] = None,
        temporal_stride_probs: Optional[Sequence[float]] = None,
        views: Sequence[str] = (),
    ) -> None:
        if not 1 <= int(min_temporal_stride) <= int(temporal_stride):
            raise ValueError(f"min_temporal_stride must be in [1, {temporal_stride}], got {min_temporal_stride}")
        self.min_temporal_stride = int(min_temporal_stride)
        self.npz_action_key = NPZ_ACTION_KEY
        self._start_frames: dict[str, int] = {}
        self._strides: dict[str, int] = {}
        self._max_strides: dict[str, int] = {}
        self._stride_choices: Optional[tuple[int, ...]] = None
        self._stride_probs: dict[int, float] = {}
        if temporal_stride_choices:
            choices = tuple(int(s) for s in temporal_stride_choices)
            probs = [float(p) for p in (temporal_stride_probs or [1.0] * len(choices))]
            if len(probs) != len(choices) or min(choices) < 1 or min(probs) <= 0:
                raise ValueError(f"bad stride choices {temporal_stride_choices} / probs {temporal_stride_probs}")
            self._stride_choices = choices
            self._stride_probs = dict(zip(choices, probs))
            self.min_temporal_stride = min(choices)
            temporal_stride = max(choices)
        self.views = tuple(views)
        if self.views and len(self.views) != 3:
            raise ValueError(f"views must be head + two wrists, got {self.views}")
        self._wrist_rgb: dict[str, list[str]] = {}
        self._wrist_action_video: dict[str, list[str]] = {}
        # Bypass RoboDojoActionDataset.__init__, which reads its own pairs JSONL.
        RobotwinActionDataset.__init__(
            self,
            pairs_jsonl=episodes_jsonl,
            num_frames=num_frames,
            video_size=video_size,
            verify_files=verify_files,
            target_fps=target_fps,
            uniform_sample_episode=False,
            start_from_first_frame_prob=start_from_first_frame_prob,
            load_action_video=True,
            include_gripper=include_gripper,
            action_delta_scaler=action_delta_scaler,
            temporal_stride=temporal_stride,
            source_arm_dof=UNIFIED_ARM_DOF,
            skip_empty_instruction=True,
        )

    def __str__(self) -> str:
        views = f" ({'+'.join(self.views)})" if self.views else ""
        return f"{len(self.samples)} episodes{views} from {self.pairs_jsonl}"

    def _fit_episode_stride(self, usable_frames: int, max_stride: int) -> Optional[int]:
        for stride in range(max_stride, min(self.min_temporal_stride, max_stride) - 1, -1):
            if usable_frames >= self.sequence_length * stride:
                return stride
        return None

    def _build_index(self) -> list[_RobotwinActionSample]:
        samples: list[_RobotwinActionSample] = []
        skipped_missing = skipped_short = skipped_empty_cap = 0
        with open(self.pairs_jsonl, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                if not line.strip():
                    continue
                item = json.loads(line)
                rgb = str(item.get("rgb") or "").strip()
                action_video = str(item.get("action_video") or "").strip()
                action = str(item.get("action") or "").strip()
                if not (rgb and action_video and action):
                    skipped_missing += 1
                    continue
                wrists = [dict((item.get("views") or {}).get(v) or {}) for v in self.views[1:]]
                if any(not (w.get("rgb") and w.get("action_video")) for w in wrists):
                    skipped_missing += 1
                    continue
                length = int(item.get("frame_nums") or 0)
                start_frame = max(0, int(item.get("start_frame") or 0))
                max_stride = self.temporal_stride if self._stride_choices else int(
                    item.get("temporal_stride") or self.temporal_stride
                )
                stride = self._fit_episode_stride(length - start_frame, max_stride)
                if stride is None:
                    skipped_short += 1
                    continue
                caption = self._caption_from_item(item)
                if not caption:
                    skipped_empty_cap += 1
                    continue
                src_fps = item.get("fps")
                src_fps = float(src_fps) if src_fps is not None and float(src_fps) > 0 else None
                samples.append(
                    _RobotwinActionSample(
                        video_path=rgb,
                        action_video_path=action_video,
                        hdf5_path=action,
                        caption=caption,
                        length=length,
                        uuid=str(item.get("uuid") or ""),
                        task_name=str(item.get("task_name") or ""),
                        index_in_jsonl=line_idx,
                        src_fps=src_fps,
                        robot_config=str(item.get("robot") or "arx_x5"),
                    )
                )
                if start_frame:
                    self._start_frames[rgb] = start_frame
                if stride != self.temporal_stride:
                    self._strides[rgb] = stride
                self._max_strides[rgb] = stride
                if wrists:
                    self._wrist_rgb[rgb] = [w["rgb"] for w in wrists]
                    self._wrist_action_video[action_video] = [w["action_video"] for w in wrists]

        if self.verify_files and samples:
            per = 3 + 2 * len(self.views[1:])
            paths = [
                p
                for s in samples
                for p in (
                    s.video_path, s.hdf5_path, s.action_video_path,
                    *self._wrist_rgb.get(s.video_path, []),
                    *self._wrist_action_video.get(s.action_video_path, []),
                )
            ]
            exists = parallel_path_exists(paths)
            before = len(samples)
            samples = [s for i, s in enumerate(samples) if all(exists[per * i : per * i + per])]
            skipped_missing += before - len(samples)

        kept = {s.video_path for s in samples}
        self._start_frames = {k: v for k, v in self._start_frames.items() if k in kept}
        self._strides = {k: v for k, v in self._strides.items() if k in kept}
        log.info(
            f"Indexed {len(samples)} episodes from {self.pairs_jsonl} "
            f"(views={list(self.views) or ['single']}, stride choices={self._stride_choices or 'per-episode'}, "
            f"skipped: missing={skipped_missing}, too_short={skipped_short}, empty_caption={skipped_empty_cap})"
        )
        return samples

    def _draw_stride(self, video_path: str) -> int:
        fits = [s for s in self._stride_choices if s <= self._max_strides.get(video_path, self.temporal_stride)]
        weights = np.asarray([self._stride_probs[s] for s in fits], dtype=np.float64)
        return int(np.random.choice(fits, p=weights / weights.sum()))

    def _load_rgb_with_ids(self, video_path: str, meta_fps: Optional[float]):
        if self._stride_choices:
            self._strides[video_path] = self._draw_stride(video_path)
        head, frame_ids, fps = super()._load_rgb_with_ids(video_path, meta_fps)
        if not self.views:
            return head, frame_ids, fps
        wrists = [self._load_action_video_frames(p, frame_ids) for p in self._wrist_rgb[video_path]]
        return compose_multiview([head, *wrists]), frame_ids, fps

    def _load_action_video_frames(self, path: str, frame_ids: Sequence[int]) -> torch.Tensor:
        return RobotwinActionDataset._load_action_video_with_ids(self, path, frame_ids)

    def _load_action_video_with_ids(self, action_video_path: str, frame_ids: Sequence[int]) -> torch.Tensor:
        head = self._load_action_video_frames(action_video_path, frame_ids)
        if not self.views:
            return head
        wrists = [self._load_action_video_frames(p, frame_ids) for p in self._wrist_action_video[action_video_path]]
        return compose_multiview([head, *wrists])

    def _load_actions(self, hdf5_path: str, frame_ids: Sequence[int]) -> torch.Tensor:
        if not os.path.isfile(hdf5_path):
            raise FileNotFoundError(f"Action file does not exist: {hdf5_path}")
        actions = np.asarray(np.load(hdf5_path, allow_pickle=False), dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != UNIFIED_ACTION_DIM:
            raise ValueError(f"Expected [T, {UNIFIED_ACTION_DIM}] actions, got {actions.shape} in {hdf5_path}")
        safe_ids = [min(max(int(i), 0), actions.shape[0] - 1) for i in frame_ids]
        return torch.from_numpy(np.ascontiguousarray(actions[safe_ids])).float()
