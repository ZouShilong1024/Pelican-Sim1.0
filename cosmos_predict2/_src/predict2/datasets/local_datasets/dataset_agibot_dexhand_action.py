# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AgiBot-World dexterous-hand action-video dataset (JSONL / pkl.gz labels).

Label file (default)::

    /datasets/zhangshilin/agibot_action_video/
      agibot_dexhand_fixed_base_with_manual_calib_with_action_video_ac_action_exist.pkl.gz

Also accepts the companion ``.jsonl``. Each record provides::

    root + camera_video_info[head_color].video  → RGB mp4
    root + action.parquet_path                  → LeRobot parquet
    action_video                                → rendered action mp4 (abs path)
    frame_nums / fps / caption_*                → length + instruction

Parquet fields used for the value stream::

    action.joint.position     [T, 14] = left_arm(7) + right_arm(7)
    action.effector.position  [T, 12] = left_dexhand(6) + right_dexhand(6)

Packed into the canonical unified layout (gripper slots zero-filled)::

    left_arm(7) + left_gripper(1) + left_dexhand(6)
    + right_arm(7) + right_gripper(1) + right_dexhand(6)

Temporal sampling (``temporal_stride``, default **12**)::

    Source ~30 fps. After keeping episodes with
    ``length >= num_frames * 12``, mean length ≈ 2279 → mean effective ≈ 190
    (target 160–200).
"""

from __future__ import annotations

import gzip
import json
import os
import pickle
import traceback
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
    UNIFIED_DEXHAND_DOF,
    pack_unified_bimanual_action,
)
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_agibot import (
    DEFAULT_CAMERA_KEY,
    _caption_from_metadata_item,
    _is_placeholder_instruction,
)
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_agibot_action import (
    SOURCE_ARM_DOF,
    SOURCE_JOINT_DIM,
    remap_datasets_mount,
    summarize_frame_lengths,
)
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_real_source_world_action import (
    DEFAULT_ARM_DELTA_SCALER,
    DEFAULT_DEXHAND_DELTA_SCALER,
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

_DEXHAND_LABEL_CANDIDATES: tuple[str, ...] = (
    "/datasets/zhangshilin/agibot_action_video/"
    "agibot_dexhand_fixed_base_with_manual_calib_with_action_video_ac_action_exist.pkl.gz",
    "/datasets/zhangshilin/agibot_action_video/"
    "agibot_dexhand_fixed_base_with_manual_calib_with_action_video_ac_action_exist.jsonl",
    "/media/datasets/OminiEWM_Data/labels/train/agibot_world/"
    "agibot_dexhand_fixed_base_with_manual_calib_with_action_video.jsonl",
)


def resolve_default_labels_jsonl() -> str:
    """Return the first existing dexhand label dump (pkl.gz preferred)."""
    for path in _DEXHAND_LABEL_CANDIDATES:
        if os.path.isfile(path):
            return path
    return _DEXHAND_LABEL_CANDIDATES[0]


DEFAULT_LABELS_JSONL = resolve_default_labels_jsonl()
DEFAULT_TARGET_FPS = 16.0
# Kept-set mean / 12 ≈ 190 for the fixed-base dexhand dump.
DEFAULT_TEMPORAL_STRIDE = 12
SOURCE_DEXHAND_DIM = UNIFIED_DEXHAND_DOF  # 6 per side
SOURCE_EFFECTOR_DIM = SOURCE_DEXHAND_DIM * 2  # 12

_JOINT_COLS = tuple(f"action.joint.position.{i}" for i in range(SOURCE_JOINT_DIM))
_EFFECTOR_COLS = tuple(f"action.effector.position.{i}" for i in range(SOURCE_EFFECTOR_DIM))


def pack_agibot_dexhand_joint_effector_to_unified(
    joint: np.ndarray,
    effector: np.ndarray,
) -> np.ndarray:
    """``joint[T,14]`` + ``effector[T,12]`` → unified ``[T, 28]``.

    Gripper scalars are zero (dexhand robots have no parallel-jaw gripper);
    dexhand takes the full 6-D per side from ``action.effector.position``.
    """
    j = np.asarray(joint, dtype=np.float32)
    e = np.asarray(effector, dtype=np.float32)
    if j.ndim != 2 or j.shape[1] < SOURCE_JOINT_DIM:
        raise ValueError(f"Expected joint [T,>={SOURCE_JOINT_DIM}], got {j.shape}")
    if e.ndim != 2 or e.shape[1] < SOURCE_EFFECTOR_DIM:
        raise ValueError(f"Expected effector [T,>={SOURCE_EFFECTOR_DIM}], got {e.shape}")
    if j.shape[0] != e.shape[0]:
        raise ValueError(f"joint/effector length mismatch: {j.shape[0]} vs {e.shape[0]}")
    t = j.shape[0]
    return pack_unified_bimanual_action(
        left_arm=j[:, :SOURCE_ARM_DOF],
        left_gripper=np.zeros((t,), dtype=np.float32),
        right_arm=j[:, SOURCE_ARM_DOF : SOURCE_JOINT_DIM],
        right_gripper=np.zeros((t,), dtype=np.float32),
        left_dexhand=e[:, :SOURCE_DEXHAND_DIM],
        right_dexhand=e[:, SOURCE_DEXHAND_DIM:SOURCE_EFFECTOR_DIM],
    )


def load_agibot_dexhand_action_from_parquet(parquet_path: str) -> np.ndarray:
    """Load absolute unified ``[T, 28]`` from an AgiBot dexhand episode parquet."""
    if not os.path.isfile(parquet_path):
        raise FileNotFoundError(f"Parquet not found: {parquet_path}")
    # Discover effector width: metadata JSON often says 2, but dexhand parquet is 12.
    schema_names = set(pq.read_schema(parquet_path).names)
    joint_cols = [c for c in _JOINT_COLS if c in schema_names]
    if len(joint_cols) < SOURCE_JOINT_DIM:
        raise KeyError(
            f"Missing joint columns in {parquet_path}; have "
            f"{sorted(n for n in schema_names if n.startswith('action.joint.position'))[:20]}"
        )
    effector_cols = []
    i = 0
    while f"action.effector.position.{i}" in schema_names:
        effector_cols.append(f"action.effector.position.{i}")
        i += 1
    if len(effector_cols) < SOURCE_EFFECTOR_DIM:
        raise KeyError(
            f"Expected >= {SOURCE_EFFECTOR_DIM} effector cols in {parquet_path}, "
            f"found {len(effector_cols)}"
        )
    effector_cols = effector_cols[:SOURCE_EFFECTOR_DIM]
    table = pq.read_table(parquet_path, columns=joint_cols + effector_cols)
    joint = np.column_stack(
        [table.column(c).to_numpy(zero_copy_only=False) for c in joint_cols]
    ).astype(np.float32, copy=False)
    effector = np.column_stack(
        [table.column(c).to_numpy(zero_copy_only=False) for c in effector_cols]
    ).astype(np.float32, copy=False)
    return pack_agibot_dexhand_joint_effector_to_unified(joint, effector)


def _dexhand_caption_from_item(item: dict[str, Any], caption_field: str) -> str:
    """Prefer natural captions; ``language_instruction.task`` is often ``task_XXX``."""
    # Explicit natural-caption priority for dexhand JSONL dumps.
    for key in (
        "caption_natural_short_1",
        "caption_natural_short_2",
        "caption_natural_long_1",
        "caption_natural_long_2",
    ):
        text = str(item.get(key) or "").strip()
        if text and not _is_placeholder_instruction(text):
            return text

    caption = _caption_from_metadata_item(item, caption_field)
    if caption and not _is_placeholder_instruction(caption):
        return caption

    # Fall back to action_descriptions join.
    cap = item.get("Caption") or {}
    acts = cap.get("action_descriptions") if isinstance(cap, dict) else None
    if isinstance(acts, (list, tuple)):
        parts = [str(x).strip() for x in acts if str(x).strip()]
        if parts:
            return " ".join(parts)
    return ""


@dataclass(frozen=True)
class _AgibotDexhandActionSample:
    video_path: str
    action_video_path: str
    parquet_path: str
    caption: str
    length: int
    uuid: str
    task_name: str
    episode_index: int
    src_fps: Optional[float]


class AgibotDexhandActionDataset(Dataset):
    """AgiBot dexhand RGB (+ action video) with unified 28-D actions."""

    def __init__(
        self,
        labels_jsonl: str = DEFAULT_LABELS_JSONL,
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
        remap_mounts: bool = True,
    ) -> None:
        Dataset.__init__(self)
        self.labels_jsonl = str(labels_jsonl)
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
        self.require_action_video = (
            bool(require_action_video)
            if require_action_video is not None
            else bool(load_action_video)
        )
        self.caption_field = caption_field
        self.skip_empty_instruction = bool(skip_empty_instruction)
        self.remap_mounts = bool(remap_mounts)

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
                include_dexhand=True,
                arm_scaler=DEFAULT_ARM_DELTA_SCALER,
                gripper_scaler=DEFAULT_GRIPPER_DELTA_SCALER,
                dexhand_scaler=DEFAULT_DEXHAND_DELTA_SCALER,
            )
        self.action_delta_scaler = normalize_action_delta_scaler(
            action_delta_scaler, self.action_dim
        )

        if not os.path.isfile(self.labels_jsonl):
            alt = remap_datasets_mount(self.labels_jsonl)
            if alt != self.labels_jsonl and os.path.isfile(alt):
                self.labels_jsonl = alt
            else:
                raise FileNotFoundError(
                    f"AgiBot dexhand labels not found: {labels_jsonl} "
                    f"(accepts .jsonl / .pkl / .pkl.gz)"
                )

        self.samples = self._build_index()
        if not self.samples:
            raise RuntimeError(
                f"No valid AgiBot dexhand episodes from {self.labels_jsonl} "
                f"(num_frames={self.sequence_length}, temporal_stride={self.temporal_stride})."
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
            f"AgibotDexhandActionDataset: {len(self.samples)} episodes from {self.labels_jsonl} "
            f"(num_frames={self.sequence_length}, temporal_stride={self.temporal_stride}, "
            f"raw_sequence_length={self.raw_sequence_length}, "
            f"video_size={self.video_size}, action_dim={self.action_dim}, "
            f"frame_mean={frame_stats['mean']:.1f}, eff_mean_after_stride≈{eff_mean:.1f}, "
            f"action_delta_scaler={scaler_msg}, load_action_video={self.load_action_video})"
        )

        self.preprocess = T.Compose(
            [ToTensorVideo(), ResizePreprocess((self.video_size[0], self.video_size[1]))]
        )

    def __str__(self) -> str:
        return f"{len(self.samples)} AgiBot dexhand samples from {self.labels_jsonl}"

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve_path(self, path: str) -> str:
        p = str(path or "").strip()
        if not p:
            return p
        return remap_datasets_mount(p) if self.remap_mounts else p

    def _resolve_rgb_path(self, item: dict[str, Any]) -> str:
        root = str(item.get("root") or "").strip()
        cvi = item.get("camera_video_info") or {}
        view = cvi.get(self.camera_key) or {}
        rel = str((view or {}).get("video") or "").strip()
        if not rel:
            return ""
        if os.path.isabs(rel):
            return self._resolve_path(rel)
        if not root:
            return self._resolve_path(rel)
        return self._resolve_path(os.path.join(root, rel))

    def _resolve_parquet_path(self, item: dict[str, Any]) -> str:
        root = str(item.get("root") or "").strip()
        rel = str((item.get("action") or {}).get("parquet_path") or "").strip()
        if not rel:
            return ""
        if os.path.isabs(rel):
            return self._resolve_path(rel)
        if not root:
            return self._resolve_path(rel)
        return self._resolve_path(os.path.join(root, rel))

    def _resolve_action_video_path(self, item: dict[str, Any]) -> str:
        # Dexhand JSONL stores absolute path under top-level ``action_video``.
        path = str(item.get("action_video") or "").strip()
        if path:
            return self._resolve_path(path)
        avi = item.get("action_video_info") or {}
        return self._resolve_path(str((avi or {}).get("video") or "").strip())

    def _load_label_items(self) -> list[dict[str, Any]]:
        """Load dexhand label records from ``.jsonl`` or ``.pkl(.gz)``."""
        path = self.labels_jsonl
        log.info(f"Loading AgiBot dexhand labels: {path}")
        lower = path.lower()
        if lower.endswith(".pkl.gz") or lower.endswith(".pkl"):
            opener = gzip.open if lower.endswith(".gz") else open
            with opener(path, "rb") as f:
                raw = pickle.load(f)
            if isinstance(raw, dict):
                for key in ("data", "items", "samples", "dataset"):
                    if key in raw and isinstance(raw[key], list):
                        raw = raw[key]
                        break
                else:
                    raise TypeError(
                        f"Unsupported dexhand label pickle dict keys: {list(raw.keys())[:20]}"
                    )
            if not isinstance(raw, list):
                raise TypeError(f"Expected list in dexhand label pickle, got {type(raw)}")
            return [x for x in raw if isinstance(x, dict)]

        items: list[dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if isinstance(item, dict):
                    items.append(item)
        return items

    def _build_index(self) -> list[_AgibotDexhandActionSample]:
        samples: list[_AgibotDexhandActionSample] = []
        skipped_missing = 0
        skipped_short = 0
        skipped_empty_cap = 0

        for item in self._load_label_items():
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
                length = int((item.get("action_video_info") or {}).get("frame_nums") or 0)
            if is_too_short_for_window(
                length, self.raw_sequence_length, self.uniform_sample_episode
            ):
                skipped_short += 1
                continue

            caption = _dexhand_caption_from_item(item, self.caption_field)
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

            episode_index = int(item.get("episode_index") or -1)
            if episode_index < 0:
                base = os.path.basename(parquet_path)
                if base.startswith("episode_") and base.endswith(".parquet"):
                    try:
                        episode_index = int(base[len("episode_") : -len(".parquet")])
                    except ValueError:
                        episode_index = -1

            samples.append(
                _AgibotDexhandActionSample(
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

        if self.verify_files and samples:
            paths_to_check: list[str] = []
            for s in samples:
                paths_to_check.append(s.video_path)
                paths_to_check.append(s.parquet_path)
                if self.require_action_video and s.action_video_path:
                    paths_to_check.append(s.action_video_path)
            exists = parallel_path_exists(paths_to_check)
            kept: list[_AgibotDexhandActionSample] = []
            cursor = 0
            before = len(samples)
            for s in samples:
                ok_v = exists[cursor]
                ok_p = exists[cursor + 1]
                cursor += 2
                ok_av = True
                if self.require_action_video and s.action_video_path:
                    ok_av = exists[cursor]
                    cursor += 1
                if ok_v and ok_p and ok_av:
                    kept.append(s)
            samples = kept
            skipped_missing += before - len(samples)

        log.info(
            f"Indexed {len(samples)} AgiBot dexhand episodes "
            f"(skipped_missing={skipped_missing}, skipped_short={skipped_short}, "
            f"skipped_empty_caption={skipped_empty_cap}, verify_files={self.verify_files})"
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
                f"{self.sequence_length} ids, got {len(frame_ids)}"
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

    def _load_actions(self, parquet_path: str, frame_ids: Sequence[int]) -> torch.Tensor:
        unified = load_agibot_dexhand_action_from_parquet(parquet_path)
        n = unified.shape[0]
        safe_ids = [min(max(int(i), 0), n - 1) for i in frame_ids]
        selected = unified[safe_ids]
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
                f"Failed to load AgiBot dexhand sample {sample.video_path} "
                f"(total failures: {self.num_failed_loads}): {e}\n{traceback.format_exc()}",
                rank0_only=False,
            )
            return self[int(np.random.randint(len(self.samples)))]

    @classmethod
    def render_action_video(cls, request: Any) -> np.ndarray:
        """White-bg skeleton ``[T, H, W, 3]`` for one AgiBot dexhand episode."""
        from cosmos_predict2._src.predict2.datasets.local_datasets.action_video_render import (
            BACKEND_ROOT,
        )
        from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_agibot_action import (
            render_agibot_action_video,
        )

        urdf = BACKEND_ROOT / "agibot" / "G1_revo2.urdf"
        return render_agibot_action_video(request, urdf=str(urdf), dexhand=True)


__all__ = [
    "DEFAULT_LABELS_JSONL",
    "DEFAULT_TARGET_FPS",
    "DEFAULT_TEMPORAL_STRIDE",
    "SOURCE_DEXHAND_DIM",
    "SOURCE_EFFECTOR_DIM",
    "AgibotDexhandActionDataset",
    "load_agibot_dexhand_action_from_parquet",
    "pack_agibot_dexhand_joint_effector_to_unified",
]
