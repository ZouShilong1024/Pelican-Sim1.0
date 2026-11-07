# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AgiBot-World action-video dataset with unified bimanual action layout.

Label list (``labels_pkl``) is an OminiEWM caption dump with action-video join,
e.g. ``..._noBadAV_no_base_lift_head.pkl.gz``. Each item provides::

    root + camera_video_info[head_color].video  → RGB mp4
    action_video_info.video                     → rendered skeleton mp4
    root + action.parquet_path                  → LeRobot parquet
    frame_nums / fps / Caption...               → length + instruction

Parquet action fields used for the value stream (gripper AgiBot; not the full
meta ``action_dim``)::

    action.joint.position     [T, 14] = left_arm(7) + right_arm(7)
    action.effector.position  [T, 2]  = left_gripper + right_gripper

Packed into the canonical action-inject layout (dexhand zero-filled)::

    left_arm(7) + left_gripper(1) + left_dexhand(6)
    + right_arm(7) + right_gripper(1) + right_dexhand(6)

Joints match ``state.joint.position``; grippers use ``action.effector`` in
``[0, 1]`` (state effector is a different raw scale ~35–120).

Temporal sampling (``temporal_stride``, default **9**)::

    Source ~30 fps, mean episode length ≈ 1731 frames across 158 tasks.
    Stride 9 → mean effective length ≈ 192 (near the ~170–190 target),
    model rate ~3.3 fps, and keeps most episodes with
    length >= num_frames*9.
"""

from __future__ import annotations

import gzip
import os
import pickle
import traceback
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union

import numpy as np
import pyarrow.parquet as pq
import torch
from decord import VideoReader, cpu
from torch.utils.data import Dataset
from torchvision import transforms as T

from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.predict2.datasets.local_datasets.action_unified_layout import (
    UNIFIED_ACTION_DIM,
    UNIFIED_ACTION_NAMES,
    pack_unified_bimanual_action,
)
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_agibot import (
    DEFAULT_CAMERA_KEY,
    _caption_from_metadata_item,
    _is_placeholder_instruction,
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

# Prefer the curated action-video dump under /datasets/zhangshilin; keep legacy
# OminiEWM paths as fallbacks for older cluster mounts.
_AGIBOT_ACTION_VIDEO_DIR = "/datasets/zhangshilin/agibot_action_video"
_LABELS_DIR_LEGACY = "/media/datasets/OminiEWM_Data/labels/train/agibot_world"
_LABEL_PKL_CANDIDATES: tuple[str, ...] = (
    f"{_AGIBOT_ACTION_VIDEO_DIR}/agibot_world_full_with_caption_v3_vqp_clean_with_action_video_train_noBadAV_no_base_lift_head_av_v4.pkl.gz",
    f"{_LABELS_DIR_LEGACY}/agibot_world_full_with_caption_v3_vqp_clean_with_action_video_train_noBadAV_no_base_lift_head_av_v4.pkl.gz",
    f"{_LABELS_DIR_LEGACY}/agibot_world_full_with_caption_v3_vqp_clean_with_action_video_train_noBadAV_no_base_lift_head.pkl.gz",
    f"{_LABELS_DIR_LEGACY}/agibot_world_full_with_caption_v3_av_v4.pkl.gz",
)

DEFAULT_TARGET_FPS = 16.0
# mean(frame_nums)/9 ≈ 192 for the no_base_lift_head train dump.
DEFAULT_TEMPORAL_STRIDE = 9
SOURCE_ARM_DOF = 7
SOURCE_JOINT_DIM = SOURCE_ARM_DOF * 2  # 14
SOURCE_EFFECTOR_DIM = 2  # gripper-only AgiBot
SOURCE_PACKED_DIM = UNIFIED_ACTION_DIM  # 28

_JOINT_COLS = tuple(f"action.joint.position.{i}" for i in range(SOURCE_JOINT_DIM))
_EFFECTOR_COLS = tuple(f"action.effector.position.{i}" for i in range(SOURCE_EFFECTOR_DIM))


def resolve_default_labels_pkl() -> str:
    """Return the first existing AgiBot action-video label pkl."""
    for path in _LABEL_PKL_CANDIDATES:
        if os.path.isfile(path):
            return path
    return _LABEL_PKL_CANDIDATES[0]


DEFAULT_LABELS_PKL = resolve_default_labels_pkl()


def remap_datasets_mount(path: str) -> str:
    """Prefer ``/media/datasets/...`` when ``/datasets/...`` is the stored mount."""
    p = str(path or "").strip()
    if not p:
        return p
    if p.startswith("/media/datasets/"):
        return p
    if p.startswith("/datasets/"):
        alt = "/media/datasets/" + p[len("/datasets/") :]
        if os.path.isfile(alt) or not os.path.isfile(p):
            return alt
    return p


def pack_agibot_joint_effector_to_unified16(
    joint: np.ndarray,
    effector: np.ndarray,
) -> np.ndarray:
    """``joint[T,14]`` + ``effector[T,2]`` → unified ``[T, 28]`` (dexhand=0).

    Kept name ``...unified16`` for import compatibility; output width follows
    ``UNIFIED_ACTION_DIM``.
    """
    j = np.asarray(joint, dtype=np.float32)
    e = np.asarray(effector, dtype=np.float32)
    if j.ndim != 2 or j.shape[1] < SOURCE_JOINT_DIM:
        raise ValueError(f"Expected joint [T,>={SOURCE_JOINT_DIM}], got {j.shape}")
    if e.ndim != 2 or e.shape[1] < SOURCE_EFFECTOR_DIM:
        raise ValueError(f"Expected effector [T,>={SOURCE_EFFECTOR_DIM}], got {e.shape}")
    if j.shape[0] != e.shape[0]:
        raise ValueError(f"joint/effector length mismatch: {j.shape[0]} vs {e.shape[0]}")
    return pack_unified_bimanual_action(
        left_arm=j[:, :7],
        left_gripper=e[:, 0],
        right_arm=j[:, 7:14],
        right_gripper=e[:, 1],
        left_dexhand=None,
        right_dexhand=None,
    )


# Prefer the accurate name going forward.
pack_agibot_joint_effector_to_unified = pack_agibot_joint_effector_to_unified16


def load_agibot_action_from_parquet(parquet_path: str) -> np.ndarray:
    """Load absolute unified ``[T, 28]`` from a gripper AgiBot episode parquet."""
    if not os.path.isfile(parquet_path):
        raise FileNotFoundError(f"Parquet not found: {parquet_path}")
    table = pq.read_table(parquet_path, columns=list(_JOINT_COLS) + list(_EFFECTOR_COLS))
    joint = np.column_stack(
        [table.column(c).to_numpy(zero_copy_only=False) for c in _JOINT_COLS]
    ).astype(np.float32, copy=False)
    effector = np.column_stack(
        [table.column(c).to_numpy(zero_copy_only=False) for c in _EFFECTOR_COLS]
    ).astype(np.float32, copy=False)
    return pack_agibot_joint_effector_to_unified(joint, effector)


def summarize_frame_lengths(
    lengths: Sequence[int],
    *,
    task_names: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    """Overall (+ optional per-task) frame-length summary."""
    arr = np.asarray(list(lengths), dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    out: dict[str, Any] = {
        "n": int(arr.size),
        "min": float(arr.min()),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.median(arr)),
        "mean": float(arr.mean()),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }
    if task_names is not None:
        by: dict[str, list[int]] = defaultdict(list)
        for name, length in zip(task_names, lengths):
            by[str(name)].append(int(length))
        per_task = []
        for name in sorted(by.keys()):
            a = np.asarray(by[name], dtype=np.float64)
            per_task.append(
                {
                    "task_name": name,
                    "n": int(a.size),
                    "mean": float(a.mean()),
                    "p50": float(np.median(a)),
                    "min": int(a.min()),
                    "max": int(a.max()),
                }
            )
        out["n_tasks"] = len(per_task)
        out["per_task"] = per_task
    return out


@dataclass(frozen=True)
class _AgibotActionSample:
    video_path: str
    action_video_path: str
    parquet_path: str
    caption: str
    length: int
    uuid: str
    task_name: str
    episode_index: int
    src_fps: Optional[float]


class AgibotActionDataset(Dataset):
    """AgiBot RGB (+ optional action video) with unified 28-D action values.

    Output keys (aligned with ``RealSourceWorldActionDataset``)::

        video           uint8 [C, T, H, W]
        action          float [T, 28]
        action_state    float [28]
        action_delta    float [T-1, 28]   (**scaled** if ``action_delta_scaler`` set)
        action_video    uint8 [C, T, H, W]  only if ``load_action_video=True``
        ai_caption / fps / image_size / num_frames / padding_mask
    """

    def __init__(
        self,
        labels_pkl: str = DEFAULT_LABELS_PKL,
        num_frames: int = 93,
        video_size: tuple[int, int] = (480, 640),
        camera_key: str = DEFAULT_CAMERA_KEY,
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
        temporal_stride: int = DEFAULT_TEMPORAL_STRIDE,
        caption_field: str = "Caption.language_instruction.task",
        skip_empty_instruction: bool = True,
    ) -> None:
        Dataset.__init__(self)
        self.labels_pkl = labels_pkl if labels_pkl else resolve_default_labels_pkl()
        self.sequence_length = int(num_frames)
        self.video_size = tuple(video_size)
        self.camera_key = camera_key
        self.verify_files = bool(verify_files)
        self.target_fps = float(target_fps)
        self.uniform_sample_episode = bool(uniform_sample_episode)
        self.start_from_first_frame_prob = float(start_from_first_frame_prob)
        self.exclude_video_paths = load_exclude_video_paths(exclude_video_paths)
        self.num_failed_loads = 0

        self.load_action_video = bool(load_action_video)
        self.include_gripper = bool(include_gripper)
        if not self.include_gripper:
            raise ValueError("AgibotActionDataset currently requires include_gripper=True")
        self.require_action_video = (
            bool(require_action_video)
            if require_action_video is not None
            else bool(load_action_video)
        )
        self.caption_field = caption_field
        self.skip_empty_instruction = bool(skip_empty_instruction)

        if int(temporal_stride) < 1:
            raise ValueError(f"temporal_stride must be >= 1, got {temporal_stride}")
        self.temporal_stride = int(temporal_stride)
        self.raw_sequence_length = self.sequence_length * self.temporal_stride

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

        if not os.path.isfile(self.labels_pkl):
            raise FileNotFoundError(f"AgiBot labels pkl not found: {self.labels_pkl}")

        self.samples = self._build_index()
        if not self.samples:
            raise RuntimeError(
                f"No valid AgiBot action episodes from {self.labels_pkl} "
                f"(num_frames={self.sequence_length}, "
                f"temporal_stride={self.temporal_stride}, "
                f"raw_sequence_length={self.raw_sequence_length}, "
                f"load_action_video={self.load_action_video})."
            )

        lengths = [s.length for s in self.samples]
        frame_stats = summarize_frame_lengths(
            lengths, task_names=[s.task_name for s in self.samples]
        )
        self.frame_length_stats = frame_stats
        eff_mean = float(frame_stats["mean"]) / float(self.temporal_stride)

        scaler_msg = (
            "None"
            if self.action_delta_scaler is None
            else [round(float(x), 4) for x in self.action_delta_scaler.tolist()]
        )
        log.info(
            f"AgibotActionDataset: {len(self.samples)} episodes from {self.labels_pkl} "
            f"(num_frames={self.sequence_length}, temporal_stride={self.temporal_stride}, "
            f"raw_sequence_length={self.raw_sequence_length}, "
            f"video_size={self.video_size}, action_dim={self.action_dim}, "
            f"n_tasks={frame_stats.get('n_tasks')}, "
            f"frame_mean={frame_stats['mean']:.1f}, "
            f"eff_mean_after_stride≈{eff_mean:.1f}, "
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
        return f"{len(self.samples)} AgiBot action samples from {self.labels_pkl}"

    def __len__(self) -> int:
        return len(self.samples)

    def _load_label_items(self) -> list[dict[str, Any]]:
        log.info(f"Loading AgiBot action labels: {self.labels_pkl}")
        with gzip.open(self.labels_pkl, "rb") as f:
            raw = pickle.load(f)
        if isinstance(raw, dict):
            for key in ("data", "items", "samples", "dataset"):
                if key in raw and isinstance(raw[key], list):
                    raw = raw[key]
                    break
            else:
                raise TypeError(f"Unsupported label pickle dict keys: {list(raw.keys())[:20]}")
        if not isinstance(raw, list):
            raise TypeError(f"Expected list in label pickle, got {type(raw)}")
        return raw

    def _resolve_rgb_path(self, item: dict[str, Any]) -> str:
        root = str(item.get("root") or "").strip()
        cvi = item.get("camera_video_info") or {}
        view = cvi.get(self.camera_key) or {}
        rel = str((view or {}).get("video") or "").strip()
        if not rel:
            return ""
        if os.path.isabs(rel):
            return remap_datasets_mount(rel)
        if not root:
            return rel
        return os.path.join(root, rel)

    def _resolve_parquet_path(self, item: dict[str, Any]) -> str:
        root = str(item.get("root") or "").strip()
        rel = str((item.get("action") or {}).get("parquet_path") or "").strip()
        if not rel:
            return ""
        if os.path.isabs(rel):
            return remap_datasets_mount(rel)
        if not root:
            return rel
        return os.path.join(root, rel)

    def _resolve_action_video_path(self, item: dict[str, Any]) -> str:
        avi = item.get("action_video_info") or {}
        path = str((avi or {}).get("video") or "").strip()
        return remap_datasets_mount(path)

    def _caption_from_item(self, item: dict[str, Any]) -> str:
        caption = _caption_from_metadata_item(item, self.caption_field)
        if caption and not _is_placeholder_instruction(caption):
            return caption
        return ""

    def _build_index(self) -> list[_AgibotActionSample]:
        samples: list[_AgibotActionSample] = []
        skipped_missing = 0
        skipped_short = 0
        skipped_empty_cap = 0

        items = self._load_label_items()
        for item in items:
            if not isinstance(item, dict):
                skipped_missing += 1
                continue

            video_path = self._resolve_rgb_path(item)
            parquet_path = self._resolve_parquet_path(item)
            action_video_path = self._resolve_action_video_path(item)
            if not video_path or not parquet_path:
                skipped_missing += 1
                continue
            if self.require_action_video and not action_video_path:
                skipped_missing += 1
                continue
            if is_excluded_video_path(video_path, self.exclude_video_paths):
                skipped_missing += 1
                continue

            length = int(item.get("frame_nums") or 0)
            if length <= 0:
                avi = item.get("action_video_info") or {}
                length = int((avi or {}).get("frame_nums") or 0)
            if is_too_short_for_window(
                length, self.raw_sequence_length, self.uniform_sample_episode
            ):
                skipped_short += 1
                continue

            caption = self._caption_from_item(item)
            if self.skip_empty_instruction and not caption:
                skipped_empty_cap += 1
                continue

            fps_raw = item.get("fps")
            try:
                src_fps = float(fps_raw) if fps_raw is not None else None
            except (TypeError, ValueError):
                src_fps = None
            if src_fps is not None and (not np.isfinite(src_fps) or src_fps <= 0):
                src_fps = None

            # Episode index from parquet filename when possible.
            episode_index = -1
            base = os.path.basename(parquet_path)
            if base.startswith("episode_") and base.endswith(".parquet"):
                try:
                    episode_index = int(base[len("episode_") : -len(".parquet")])
                except ValueError:
                    episode_index = -1

            samples.append(
                _AgibotActionSample(
                    video_path=video_path,
                    action_video_path=action_video_path,
                    parquet_path=parquet_path,
                    caption=caption,
                    length=length,
                    uuid=str(item.get("uuid") or ""),
                    task_name=str(item.get("task_name") or ""),
                    episode_index=episode_index,
                    src_fps=src_fps,
                )
            )

        del items

        if self.verify_files and samples:
            paths_to_check: list[str] = []
            for s in samples:
                paths_to_check.append(s.video_path)
                paths_to_check.append(s.parquet_path)
                if self.require_action_video and s.action_video_path:
                    paths_to_check.append(s.action_video_path)

            exists = parallel_path_exists(paths_to_check)
            kept: list[_AgibotActionSample] = []
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
            f"Indexed {len(samples)} AgiBot action episodes "
            f"(skipped_missing={skipped_missing}, skipped_short={skipped_short}, "
            f"skipped_empty_caption={skipped_empty_cap}, "
            f"verify_files={self.verify_files}, require_action_video={self.require_action_video})"
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
        if self.temporal_stride > 1:
            reported_fps = float(reported_fps) / float(self.temporal_stride)
        return frame_ids, reported_fps

    def _load_rgb_with_ids(
        self, video_path: str, meta_fps: Optional[float]
    ) -> tuple[torch.Tensor, list[int], float]:
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
        self, parquet_path: str, frame_ids: Sequence[int]
    ) -> torch.Tensor:
        actions = load_agibot_action_from_parquet(parquet_path)  # [T, 28]
        if actions.shape[1] != self.action_dim:
            raise RuntimeError(
                f"Packed action dim {actions.shape[1]} != {self.action_dim} in {parquet_path}"
            )
        n = actions.shape[0]
        safe_ids = [min(max(int(i), 0), n - 1) for i in frame_ids]
        selected = actions[safe_ids]
        return torch.from_numpy(np.ascontiguousarray(selected)).float()

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        try:
            video_thw, frame_ids, fps = self._load_rgb_with_ids(
                sample.video_path, sample.src_fps
            )
            video = video_thw.permute(1, 0, 2, 3)
            _, _, h, w = video.shape

            action = self._load_actions(sample.parquet_path, frame_ids)
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
                f"Failed to load AgiBot action sample {sample.video_path} "
                f"(total failures: {self.num_failed_loads}): {e}\n{traceback.format_exc()}",
                rank0_only=False,
            )
            return self[int(np.random.randint(len(self.samples)))]

    @classmethod
    def render_action_video(cls, request: Any) -> np.ndarray:
        """White-bg skeleton ``[T, H, W, 3]`` for one AgiBot gripper episode."""
        from cosmos_predict2._src.predict2.datasets.local_datasets.action_video_render import (
            BACKEND_ROOT,
        )

        urdf = (
            BACKEND_ROOT
            / "agibot"
            / "official_geniesimassets"
            / "robot"
            / "curobo_robot"
            / "assets"
            / "robot"
            / "G1"
            / "G1_120s.urdf"
        )
        return render_agibot_action_video(request, urdf=str(urdf), dexhand=False)


_AGIBOT_PATCH_KEYS = (
    "load_action_arrays_agibot",
    "_resolve_render_gripper",
    "_frame_cfg",
    "_frame_cfg_batch",
    "load_task_meta",
    "GRIPPER_LINK_CHAINS",
    "GRIPPER_BASE_LINKS",
    "DEFAULT_GRIPPER_FORWARD_OFFSET",
    "DEFAULT_GRIPPER_DELAY_FRAMES",
    "ACTIVE_BONE_WIDTH",
    "ACTIVE_JOINT_RADIUS",
    "build_render_tasks_from_mapping",
    "render_one",
)


def _agibot_arrays_from_action_npy(request: Any, *, dexhand: bool) -> dict[str, dict[str, np.ndarray]] | None:
    """Build kit ``{state,action}`` arrays from unified ``action.npy`` when parquet is absent."""
    path = str(getattr(request, "action_path", "") or "")
    if not path or not os.path.isfile(path):
        return None
    from cosmos_predict2._src.predict2.datasets.local_datasets.action_unified_layout import (
        denormalize_dexhand_from_unit,
        split_unified_bimanual_action,
    )

    parts = split_unified_bimanual_action(np.load(path))
    t = int(parts["left_arm"].shape[0])
    arm = np.concatenate([parts["left_arm"], parts["right_arm"]], axis=1).astype(np.float64)
    extra = dict(getattr(request, "extra", None) or {})
    head = extra.get("head") or extra.get("head_pos")
    waist = extra.get("waist") or extra.get("waist_pos")
    head = np.zeros((t, 2), dtype=np.float64) if head is None else np.broadcast_to(
        np.asarray(head, dtype=np.float64).reshape(-1, 2)[:1], (t, 2)
    ).copy()
    waist = np.zeros((t, 2), dtype=np.float64) if waist is None else np.broadcast_to(
        np.asarray(waist, dtype=np.float64).reshape(-1, 2)[:1], (t, 2)
    ).copy()
    if dexhand:
        from cosmos_predict2._src.predict2.datasets.local_datasets.action_video_render import (
            ensure_backend_import_path,
        )

        ensure_backend_import_path()
        import batch_render_action_video_agibot_dexhand as dx

        hand = np.concatenate(
            [
                denormalize_dexhand_from_unit(parts["left_dexhand"]),
                denormalize_dexhand_from_unit(parts["right_dexhand"]),
            ],
            axis=1,
        )
        gripper = dx._calibrate_hand(hand)
    else:
        gripper = np.stack([parts["left_gripper"], parts["right_gripper"]], axis=1).astype(np.float64)
    block = {"arm": arm, "gripper": gripper, "head": head, "waist": waist}
    return {"state": block, "action": block}


def render_agibot_action_video(request: Any, *, urdf: str, dexhand: bool) -> np.ndarray:
    """Shared AgiBot / dexhand online renderer (white-bg ``[T,H,W,3]``)."""
    import os
    from pathlib import Path

    from cosmos_predict2._src.predict2.datasets.local_datasets.action_video_render import (
        bgr_to_rgb,
        collect_frames,
        ensure_backend_import_path,
        load_render_meta,
    )

    ensure_backend_import_path()
    import batch_render_action_video_agibot as ab

    extra = dict(request.extra or {})
    meta = load_render_meta(request)
    parquet = request.parquet_path
    arrays = extra.get("arrays")
    if (not parquet or not os.path.isfile(parquet)) and arrays is None:
        arrays = _agibot_arrays_from_action_npy(request, dexhand=dexhand)
    if arrays is None and (not parquet or not os.path.isfile(parquet)):
        raise FileNotFoundError(
            f"AgiBot needs action.npy or parquet for online AV: {request.action_path!r} / {parquet!r}"
        )
    if not os.path.isfile(urdf):
        raise FileNotFoundError(f"AgiBot URDF missing: {urdf}")

    cam = extra.get("camera_params") or meta.get("camera_params") or {}
    if not cam:
        mapping_csv = extra.get("camera_mapping_csv") or str(ab.DEFAULT_EPISODE_CAMERA_MAPPING)
        task = str(request.task_name or extra.get("task_name") or meta.get("task_name") or "")
        ep = extra.get("episode_index")
        if ep is None and request.parquet_path:
            base = Path(request.parquet_path).stem
            if base.startswith("episode_"):
                try:
                    ep = int(base.split("_")[-1])
                except ValueError:
                    ep = None
        if task and ep is not None and os.path.isfile(mapping_csv):
            mapping = ab.load_episode_camera_mapping(mapping_csv)
            hit = mapping.get((task, int(ep)))
            if hit:
                cam = dict(hit)
        else:
            cam = {}
    cam = dict(cam or {})

    fx = float(cam.get("fx") or ab.MEAN_FX)
    fy = float(cam.get("fy") or ab.MEAN_FY)
    cx = float(cam.get("cx") or ab.MEAN_CX)
    cy = float(cam.get("cy") or ab.MEAN_CY)
    if request.width != ab.IMG_W or request.height != ab.IMG_H:
        sx = request.width / float(ab.IMG_W)
        sy = request.height / float(ab.IMG_H)
        fx, cx = fx * sx, cx * sx
        fy, cy = fy * sy, cy * sy

    offset_xyz = cam.get("camera_offset_xyz", ab.DEFAULT_CAMERA_OFFSET_XYZ)
    offset_rpy = cam.get("camera_offset_rpy_deg", ab.DEFAULT_CAMERA_OFFSET_RPY_DEG)
    arm_l_y = float(extra.get("arm_l_base_y", ab.DEFAULT_ARM_L_BASE_Y))
    arm_r_y = float(extra.get("arm_r_base_y", ab.DEFAULT_ARM_R_BASE_Y))

    restore = None
    if dexhand:
        import batch_render_action_video_agibot_dexhand as dx

        restore = {k: getattr(ab, k) for k in _AGIBOT_PATCH_KEYS}
        dx.apply_dexhand_patches()

    try:
        ab._init_worker(
            urdf,
            request.width,
            request.height,
            fx,
            fy,
            cx,
            cy,
            offset_xyz,
            offset_rpy,
            "veryfast",
            23,
            1,
            ab.GRIPPER_120S_MAX_Q,
            ab.DEFAULT_GRIPPER_INVERT,
            arm_l_y,
            arm_r_y,
        )
        task = {
            "parquet_path": parquet,
            "video_path": request.rgb_path,
            "camera_params": cam,
            "cam_axis_mode": extra.get("cam_axis_mode", "x_fwd"),
            "cam_pitch_offset": float(extra.get("cam_pitch_offset", 0.0)),
            "cam_yaw_offset": float(extra.get("cam_yaw_offset", 0.0)),
            "left_cam_pitch_offset": extra.get("left_cam_pitch_offset"),
            "right_cam_pitch_offset": extra.get("right_cam_pitch_offset"),
            "left_cam_yaw_offset": extra.get("left_cam_yaw_offset"),
            "right_cam_yaw_offset": extra.get("right_cam_yaw_offset"),
            "arm_l_base_y": arm_l_y,
            "arm_r_base_y": arm_r_y,
            "gripper_forward_offset": float(
                extra.get(
                    "gripper_forward_offset",
                    0.0 if dexhand else ab.DEFAULT_GRIPPER_FORWARD_OFFSET,
                )
            ),
            "gripper_delay_frames": int(
                extra.get("gripper_delay_frames", 0 if dexhand else ab.DEFAULT_GRIPPER_DELAY_FRAMES)
            ),
            "overlay_alpha": 1.0,
            "max_frames": int(extra.get("max_frames") or 0),
            "fk_backend": "vectorized",
            "arrays": arrays,
        }
        task = {k: v for k, v in task.items() if v is not None}
        frames = collect_frames(ab._iter_render_frames_vectorized(task, "state", overlay=False))
    finally:
        if restore is not None:
            for k, v in restore.items():
                setattr(ab, k, v)
    return bgr_to_rgb(frames)


def default_agibot_action_delta_scaler() -> list[float]:
    return build_action_delta_scaler(
        include_gripper=True,
        include_lift=False,
        arm_scaler=DEFAULT_ARM_DELTA_SCALER,
        gripper_scaler=DEFAULT_GRIPPER_DELTA_SCALER,
    )


__all__ = [
    "DEFAULT_LABELS_PKL",
    "DEFAULT_TEMPORAL_STRIDE",
    "SOURCE_ARM_DOF",
    "SOURCE_JOINT_DIM",
    "SOURCE_PACKED_DIM",
    "AgibotActionDataset",
    "UNIFIED_ACTION_DIM",
    "UNIFIED_ACTION_NAMES",
    "default_agibot_action_delta_scaler",
    "load_agibot_action_from_parquet",
    "pack_agibot_joint_effector_to_unified16",
    "remap_datasets_mount",
    "resolve_default_labels_pkl",
    "summarize_frame_lengths",
]
