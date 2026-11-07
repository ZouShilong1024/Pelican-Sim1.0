# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Robosuite OOD action-video datasets (UR5e / KUKA iiwa).

Directory layout under each root::

    index.jsonl
    <task>/                         # e.g. Lift, Stack, Door, PickPlaceCan
      episode_<id>/
        rgb.mp4
        action_video.mp4
        action.npy                  # UR5: [T, 7]=6 joints+gripper; IIWA: [T, 8]
        metadata.json

Prefers ``index.jsonl`` (absolute paths); falls back to walking ``task/episode_*``.
Missing index rows (stale paths) are skipped.

Actions pack into unified 28-D via ``to_unified_bimanual_action``:

  - UR5 (6-DoF): pad joint 7 with 0; Robotiq-85 gripper normalized
  - IIWA (7-DoF): full arm; Robotiq-140 gripper normalized
  - single-arm → left side; right / dexhand zeroed

``temporal_stride=1`` (fps resampling only), matching Libero / ManiSkill.
"""

from __future__ import annotations

import json
import os
import traceback
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union

import numpy as np
import torch
from decord import VideoReader, cpu
from torch.utils.data import Dataset
from torchvision import transforms as T

from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.predict2.datasets.local_datasets.action_unified_layout import (
    UNIFIED_ACTION_DIM,
    UNIFIED_ACTION_NAMES,
    UNIFIED_ARM_DOF,
    to_unified_bimanual_action,
)
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_real_source_world_action import (
    DEFAULT_ARM_DELTA_SCALER,
    DEFAULT_GRIPPER_DELTA_SCALER,
    build_action_delta_scaler,
    normalize_action_delta_scaler,
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

DEFAULT_TARGET_FPS = 16.0
DEFAULT_TEMPORAL_STRIDE = 1
DEFAULT_INDEX_JSONL = "index.jsonl"

# Observed open aperture (meters) slightly above dataset maxima so open → ~1.0.
# Official Robotiq stroke is smaller (85mm / 140mm); raw values can exceed that.
ROBOTIQ_85_GRIPPER_RAW_MAX = 0.14
ROBOTIQ_140_GRIPPER_RAW_MAX = 0.19

DEFAULT_UR5_DATASET_DIR = "/media/datasets/zhangshilin/ur5_robosuite_ood_ac"
DEFAULT_IIWA_DATASET_DIR = "/media/datasets/zhangshilin/iiwa_robosuite_ood_ac"
UR5_SOURCE_ARM_DOF = 6
IIWA_SOURCE_ARM_DOF = 7


@dataclass(frozen=True)
class _RobosuiteOODEpisodeSample:
    video_path: str
    action_video_path: str
    action_path: str
    caption: str
    length: int
    task_name: str
    traj_id: str
    num_arms: int
    raw_action_dim: int
    src_fps: Optional[float]
    robot_type: str = ""
    env_id: str = ""


class RobosuiteOODActionDataset(Dataset):
    """Shared Robosuite OOD RGB (+ action video) with unified 28-D actions."""

    robot_label: str = "robosuite_ood"

    def __init__(
        self,
        dataset_dir: str,
        num_frames: int = 93,
        video_size: tuple[int, int] = (480, 640),
        verify_files: bool = True,
        target_fps: float = DEFAULT_TARGET_FPS,
        uniform_sample_episode: bool = False,
        start_from_first_frame_prob: float = 0.0,
        exclude_video_paths: Optional[Union[str, list[str]]] = None,
        *,
        load_action_video: bool = True,
        include_gripper: bool = True,
        require_action_video: Optional[bool] = None,
        action_delta_scaler: Optional[Union[float, Sequence[float]]] = None,
        source_arm_dof: int = UNIFIED_ARM_DOF,
        gripper_raw_max: float = ROBOTIQ_85_GRIPPER_RAW_MAX,
        tasks: Optional[Sequence[str]] = None,
        index_jsonl: Optional[str] = None,
        temporal_stride: int = DEFAULT_TEMPORAL_STRIDE,
    ) -> None:
        Dataset.__init__(self)
        self.dataset_dir = dataset_dir
        self.sequence_length = int(num_frames)
        self.video_size = tuple(video_size)
        self.verify_files = bool(verify_files)
        self.target_fps = float(target_fps)
        self.uniform_sample_episode = bool(uniform_sample_episode)
        self.start_from_first_frame_prob = float(start_from_first_frame_prob)
        self.exclude_video_paths = load_exclude_video_paths(exclude_video_paths)
        self.num_failed_loads = 0

        self.load_action_video = bool(load_action_video)
        self.include_gripper = bool(include_gripper)
        self.require_action_video = (
            bool(require_action_video)
            if require_action_video is not None
            else bool(load_action_video)
        )
        self.source_arm_dof = int(source_arm_dof)
        self.gripper_raw_max = float(gripper_raw_max)
        self.tasks = tuple(str(t) for t in tasks) if tasks is not None else None
        self.index_jsonl = (
            str(index_jsonl)
            if index_jsonl is not None
            else os.path.join(dataset_dir, DEFAULT_INDEX_JSONL)
        )
        stride = int(temporal_stride)
        if stride != DEFAULT_TEMPORAL_STRIDE:
            log.warning(
                f"{type(self).__name__}: temporal_stride={stride} ignored; "
                f"using stride={DEFAULT_TEMPORAL_STRIDE}"
            )
        self.temporal_stride = DEFAULT_TEMPORAL_STRIDE

        self.action_dim = UNIFIED_ACTION_DIM
        self.action_names = UNIFIED_ACTION_NAMES
        if action_delta_scaler is None:
            action_delta_scaler = build_action_delta_scaler(
                include_gripper=True,
                include_lift=False,
                arm_scaler=DEFAULT_ARM_DELTA_SCALER,
                gripper_scaler=DEFAULT_GRIPPER_DELTA_SCALER,
            )
        self.action_delta_scaler = normalize_action_delta_scaler(
            action_delta_scaler, self.action_dim
        )

        if not os.path.isdir(self.dataset_dir):
            raise FileNotFoundError(
                f"{type(self).__name__} dataset dir not found: {self.dataset_dir}"
            )

        self.samples = self._build_index()
        if not self.samples:
            raise RuntimeError(
                f"No valid {type(self).__name__} episodes under {self.dataset_dir} "
                f"(num_frames={self.sequence_length}, load_action_video={self.load_action_video})."
            )

        log.info(
            f"{type(self).__name__}: {len(self.samples)} episodes from {self.dataset_dir} "
            f"(arm_dof={self.source_arm_dof}, gripper_raw_max={self.gripper_raw_max}, "
            f"action_dim={self.action_dim}, temporal_stride={self.temporal_stride}, "
            f"load_action_video={self.load_action_video}, verify_files={self.verify_files})"
        )
        self.preprocess = T.Compose(
            [ToTensorVideo(), ResizePreprocess((self.video_size[0], self.video_size[1]))]
        )

    def __str__(self) -> str:
        return f"{len(self.samples)} {type(self).__name__} samples from {self.dataset_dir}"

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve_path(self, raw: str, demo_dir: str) -> str:
        raw = str(raw or "").strip()
        if not raw:
            return ""
        return raw if os.path.isabs(raw) else os.path.join(demo_dir, raw)

    def _task_allowed(self, task_name: str) -> bool:
        if self.tasks is None:
            return True
        return task_name in self.tasks

    def _sample_from_meta(
        self,
        *,
        meta: dict[str, Any],
        demo_dir: str,
        task_name: str,
        traj_id: str,
    ) -> Optional[_RobosuiteOODEpisodeSample]:
        rgb_path = self._resolve_path(str(meta.get("rgb") or "rgb.mp4"), demo_dir)
        action_path = self._resolve_path(str(meta.get("action") or "action.npy"), demo_dir)
        action_video_path = self._resolve_path(
            str(meta.get("action_video") or "action_video.mp4"), demo_dir
        )

        if is_excluded_video_path(rgb_path, self.exclude_video_paths):
            return None
        # Always drop stale index rows (UR5 index can reference missing Door eps).
        if not os.path.isfile(rgb_path) or not os.path.isfile(action_path):
            return None
        if self.require_action_video and (
            not action_video_path or not os.path.isfile(action_video_path)
        ):
            return None

        length = int(meta.get("frames") or meta.get("frame_nums") or 0)
        if length <= 0:
            try:
                length = int(np.load(action_path, mmap_mode="r").shape[0])
            except Exception:
                return None

        if is_too_short_for_window(
            length, self.sequence_length, self.uniform_sample_episode
        ):
            return None

        caption = str(meta.get("lang") or "").strip()
        if not caption:
            caption = task_name.replace("_", " ")

        src_fps = meta.get("fps")
        src_fps_f = float(src_fps) if src_fps is not None else None
        if src_fps_f is not None and (not np.isfinite(src_fps_f) or src_fps_f <= 0):
            src_fps_f = None

        return _RobosuiteOODEpisodeSample(
            video_path=rgb_path,
            action_video_path=action_video_path if action_video_path else "",
            action_path=action_path,
            caption=caption,
            length=length,
            task_name=str(meta.get("task") or task_name),
            traj_id=str(meta.get("traj_id") or traj_id),
            num_arms=int(meta.get("num_arms") or 1),
            raw_action_dim=int(meta.get("action_dim") or 0),
            src_fps=src_fps_f,
            robot_type=str(meta.get("robot_type") or ""),
            env_id=str(meta.get("env_id") or ""),
        )

    def _iter_from_index_jsonl(self):
        if not os.path.isfile(self.index_jsonl):
            return
        with open(self.index_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                meta = json.loads(line)
                task_name = str(meta.get("task") or "")
                if not self._task_allowed(task_name):
                    continue
                rgb = str(meta.get("rgb") or "")
                demo_dir = os.path.dirname(rgb) if rgb else ""
                if not demo_dir:
                    action = str(meta.get("action") or "")
                    demo_dir = os.path.dirname(action) if action else ""
                traj_id = str(meta.get("traj_id") or os.path.basename(demo_dir))
                yield meta, demo_dir, task_name, traj_id

    def _iter_episode_dirs(self):
        for task_name in sorted(os.listdir(self.dataset_dir)):
            if task_name.startswith("_") or task_name.startswith("."):
                continue
            task_dir = os.path.join(self.dataset_dir, task_name)
            if not os.path.isdir(task_dir):
                continue
            if not self._task_allowed(task_name):
                continue
            for traj_id in sorted(os.listdir(task_dir)):
                demo_dir = os.path.join(task_dir, traj_id)
                if not os.path.isdir(demo_dir) or not traj_id.startswith("episode_"):
                    continue
                yield task_name, traj_id, demo_dir

    def _build_index(self) -> list[_RobosuiteOODEpisodeSample]:
        samples: list[_RobosuiteOODEpisodeSample] = []
        skipped_missing = 0
        skipped_short = 0
        used_index = False

        index_rows = list(self._iter_from_index_jsonl() or [])
        if index_rows:
            used_index = True
            for meta, demo_dir, task_name, traj_id in index_rows:
                length_hint = int(meta.get("frames") or meta.get("frame_nums") or 0)
                sample = self._sample_from_meta(
                    meta=meta,
                    demo_dir=demo_dir,
                    task_name=task_name,
                    traj_id=traj_id,
                )
                if sample is None:
                    if length_hint > 0 and is_too_short_for_window(
                        length_hint, self.sequence_length, self.uniform_sample_episode
                    ):
                        skipped_short += 1
                    else:
                        skipped_missing += 1
                    continue
                samples.append(sample)
        else:
            for task_name, traj_id, demo_dir in self._iter_episode_dirs():
                meta_path = os.path.join(demo_dir, "metadata.json")
                meta: dict[str, Any] = {}
                if os.path.isfile(meta_path):
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                meta.setdefault("rgb", os.path.join(demo_dir, "rgb.mp4"))
                meta.setdefault("action", os.path.join(demo_dir, "action.npy"))
                meta.setdefault("action_video", os.path.join(demo_dir, "action_video.mp4"))
                length_hint = int(meta.get("frames") or meta.get("frame_nums") or 0)
                sample = self._sample_from_meta(
                    meta=meta,
                    demo_dir=demo_dir,
                    task_name=task_name,
                    traj_id=traj_id,
                )
                if sample is None:
                    if length_hint > 0 and is_too_short_for_window(
                        length_hint, self.sequence_length, self.uniform_sample_episode
                    ):
                        skipped_short += 1
                    else:
                        skipped_missing += 1
                    continue
                samples.append(sample)

        if self.verify_files and samples:
            paths_to_check: list[str] = []
            for s in samples:
                paths_to_check.append(s.video_path)
                paths_to_check.append(s.action_path)
                if self.require_action_video and s.action_video_path:
                    paths_to_check.append(s.action_video_path)
            exists = parallel_path_exists(paths_to_check)
            kept: list[_RobosuiteOODEpisodeSample] = []
            cursor = 0
            before = len(samples)
            for s in samples:
                ok_video = exists[cursor]
                ok_action = exists[cursor + 1]
                cursor += 2
                ok_av = True
                if self.require_action_video and s.action_video_path:
                    ok_av = exists[cursor]
                    cursor += 1
                if ok_video and ok_action and ok_av:
                    kept.append(s)
            samples = kept
            skipped_missing += before - len(samples)

        log.info(
            f"Indexed {len(samples)} {type(self).__name__} episodes "
            f"(index_jsonl={'yes' if used_index else 'no'}, "
            f"skipped_missing={skipped_missing}, skipped_short={skipped_short}, "
            f"verify_files={self.verify_files})"
        )
        return samples

    def _read_fps(self, vr: VideoReader) -> Optional[float]:
        try:
            fps = float(vr.get_avg_fps())
        except Exception:
            return None
        if not np.isfinite(fps) or fps <= 0:
            return None
        return fps

    def _sample_frame_ids(
        self, total_frames: int, src_fps: Optional[float]
    ) -> tuple[list[int], float]:
        return sample_video_frame_indices(
            total_frames,
            self.sequence_length,
            uniform_sample_episode=self.uniform_sample_episode,
            src_fps=src_fps,
            target_fps=self.target_fps,
            start_from_first_frame_prob=self.start_from_first_frame_prob,
        )

    def _load_rgb_with_ids(self, video_path: str, meta_fps: Optional[float]):
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file does not exist: {video_path}")
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        total_frames = len(vr)
        src_fps = self._read_fps(vr)
        if src_fps is None:
            src_fps = meta_fps
        frame_ids, reported_fps = self._sample_frame_ids(total_frames, src_fps)
        frame_data = vr.get_batch(frame_ids).asnumpy().astype(np.uint8)
        vr.seek(0)
        del vr
        frames = torch.from_numpy(frame_data).permute(0, 3, 1, 2)
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
        safe_ids = [min(int(i), total_frames - 1) for i in frame_ids]
        frame_data = vr.get_batch(safe_ids).asnumpy().astype(np.uint8)
        vr.seek(0)
        del vr
        frames = torch.from_numpy(frame_data).permute(0, 3, 1, 2)
        frames = self.preprocess(frames)
        frames = torch.clamp(frames * 255.0, 0, 255).to(torch.uint8)
        return frames

    def _load_actions(
        self,
        action_path: str,
        frame_ids: Sequence[int],
        *,
        num_arms: int,
        raw_action_dim: int,
    ) -> torch.Tensor:
        if not os.path.isfile(action_path):
            raise FileNotFoundError(f"Action file does not exist: {action_path}")
        actions = np.asarray(np.load(action_path), dtype=np.float32)
        if actions.ndim != 2:
            raise ValueError(
                f"Expected action matrix [T, D], got shape {actions.shape} in {action_path}"
            )
        if raw_action_dim and actions.shape[1] != raw_action_dim:
            log.warning(
                f"action.npy dim {actions.shape[1]} != metadata action_dim={raw_action_dim} "
                f"in {action_path}; using array width."
            )
        n = actions.shape[0]
        safe_ids = [min(max(int(i), 0), n - 1) for i in frame_ids]
        selected = actions[safe_ids]
        unified = to_unified_bimanual_action(
            selected,
            num_arms=num_arms,
            include_gripper=self.include_gripper,
            arm_dof=self.source_arm_dof,
            gripper_raw_max=self.gripper_raw_max,
        )
        if unified.shape[-1] != self.action_dim:
            raise RuntimeError(
                f"Unified action dim {unified.shape[-1]} != {self.action_dim} from {action_path}"
            )
        return torch.from_numpy(np.ascontiguousarray(unified)).float()

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        try:
            video_thw, frame_ids, fps = self._load_rgb_with_ids(
                sample.video_path, sample.src_fps
            )
            video = video_thw.permute(1, 0, 2, 3)
            _, _, h, w = video.shape
            action = self._load_actions(
                sample.action_path,
                frame_ids,
                num_arms=sample.num_arms,
                raw_action_dim=sample.raw_action_dim,
            )
            if action.shape[0] != self.sequence_length:
                raise RuntimeError(
                    f"Action length {action.shape[0]} != num_frames {self.sequence_length}"
                )
            action_state = action[0]
            if action.shape[0] > 1:
                action_delta = action[1:] - action[:-1]
            else:
                action_delta = torch.zeros(1, self.action_dim, dtype=action.dtype)
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
                f"Failed to load {type(self).__name__} sample {sample.video_path} "
                f"(total failures: {self.num_failed_loads}): {e}\n{traceback.format_exc()}",
                rank0_only=False,
            )
            return self[int(np.random.randint(len(self.samples)))]


class UR5RobosuiteOOD(RobosuiteOODActionDataset):
    """UR5e + Robotiq-85 Robosuite OOD (6-DoF arm → pad to 7 in unified 28-D)."""

    robot_label = "ur5_robosuite_ood"

    def __init__(
        self,
        dataset_dir: str = DEFAULT_UR5_DATASET_DIR,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("source_arm_dof", UR5_SOURCE_ARM_DOF)
        kwargs.setdefault("gripper_raw_max", ROBOTIQ_85_GRIPPER_RAW_MAX)
        super().__init__(dataset_dir=dataset_dir, **kwargs)


class IIWARobosuiteOOD(RobosuiteOODActionDataset):
    """KUKA LBR iiwa + Robotiq-140 Robosuite OOD (7-DoF arm, unified 28-D)."""

    robot_label = "iiwa_robosuite_ood"

    def __init__(
        self,
        dataset_dir: str = DEFAULT_IIWA_DATASET_DIR,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("source_arm_dof", IIWA_SOURCE_ARM_DOF)
        kwargs.setdefault("gripper_raw_max", ROBOTIQ_140_GRIPPER_RAW_MAX)
        super().__init__(dataset_dir=dataset_dir, **kwargs)


__all__ = [
    "DEFAULT_IIWA_DATASET_DIR",
    "DEFAULT_TEMPORAL_STRIDE",
    "DEFAULT_UR5_DATASET_DIR",
    "IIWARobosuiteOOD",
    "IIWA_SOURCE_ARM_DOF",
    "ROBOTIQ_140_GRIPPER_RAW_MAX",
    "ROBOTIQ_85_GRIPPER_RAW_MAX",
    "RobosuiteOODActionDataset",
    "UR5RobosuiteOOD",
    "UR5_SOURCE_ARM_DOF",
]
