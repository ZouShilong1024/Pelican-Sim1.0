# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RoboMind (Tiangong dual-arm gripper) action-video dataset.

Metadata JSONL (``pairs_jsonl``) each line::

    {
      "source_video": "<abs RGB mp4>",
      "action_video": "<abs rendered action mp4>",
      "action_path": "<abs all_fields.parquet>",
      "selected_view": "camera" | "camera_top" | ...,
      "caption_description": "<task prompt>",
      "description": "<task prompt>",
      "language_instruction": {"lang": "en", "atomic_actions": [...]},
      "status": "ok" | "exists",
      ...
    }

Parquet action columns (Tiangong dual-arm gripper)::

    control_arm_joints_robot / arm_joints_robot   [T, 14]
      = left_arm(7) + right_arm(7)
    control_hand_joints_{left,right} / hand_joints_{left,right}  [T, 1]

Packed into the canonical unified 28-D action-inject layout::

    left_arm(7) + left_gripper(1) + left_dexhand(6)
    + right_arm(7) + right_gripper(1) + right_dexhand(6)

with dexhand slots zero-filled.

Temporal sampling (``temporal_stride``, default **2**)::

    Main view ``camera`` @~26 fps has mean episode length ≈ 353 frames.
    Stride 2 → mean effective ≈ 176–180 (target ~180), model rate ~13 fps,
    and keeps ~93% of demos with ``length >= num_frames * 2``.

By default only the main view ``camera`` is indexed (``camera_top`` episodes
are much longer / different robot station setup).
"""

from __future__ import annotations

import json
import os
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
    UNIFIED_ARM_DOF,
    pack_unified_bimanual_action,
)
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_agibot_action import (
    remap_datasets_mount,
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

DEFAULT_PAIRS_JSONL = (
    "/datasets/zhangshilin/robomind_tiangong_gripper_ac_filtered_all_problems_legacy4_full_v1/"
    "action_video_list.jsonl"
)
DEFAULT_TARGET_FPS = 16.0
# Main-view camera @~26fps, mean ~353 → stride 2 ≈ mean effective 176–180.
DEFAULT_TEMPORAL_STRIDE = 2
DEFAULT_VIEWS: tuple[str, ...] = ("camera",)
SOURCE_ARM_DOF = UNIFIED_ARM_DOF  # 7
SOURCE_JOINT_DIM = SOURCE_ARM_DOF * 2  # 14


@dataclass(frozen=True)
class _RobomindActionSample:
    video_path: str
    action_video_path: str
    action_path: str
    caption: str
    length: int
    view: str
    robot: str
    source_index: int
    index_in_jsonl: int
    src_fps: Optional[float]


def _as_text(value: Any) -> str:
    """Best-effort stringify for caption / instruction fields."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in (
            "caption",
            "description",
            "text",
            "instruction",
            "language_instruction",
            "en",
            "zh",
        ):
            if key in value:
                text = _as_text(value[key])
                if text:
                    return text
        atomic = value.get("atomic_actions")
        if isinstance(atomic, (list, tuple)):
            parts = [_as_text(x) for x in atomic]
            parts = [p for p in parts if p]
            if parts:
                return "; ".join(parts)
        return ""
    if isinstance(value, (list, tuple)):
        parts = [_as_text(x) for x in value]
        parts = [p for p in parts if p]
        return "; ".join(parts) if parts else ""
    return str(value).strip()


def _caption_from_item(item: dict[str, Any]) -> str:
    """Prefer episode-level task text; fall back to atomic English steps.

    ``language_instruction.atomic_actions`` is sometimes mismatched with the
    rendered episode, so ``caption_description`` / ``description`` come first.
    """
    for key in ("caption_description", "description"):
        text = _as_text(item.get(key))
        if text:
            return text
    lang = item.get("language_instruction")
    text = _as_text(lang)
    # ``{"lang": "en", ...}`` alone is not a usable prompt.
    if text and text.lower() not in ("en", "zh", "english", "chinese"):
        return text
    for key in ("action_descriptions", "scene_description"):
        text = _as_text(item.get(key))
        if text:
            return text
    return ""


def _parquet_num_rows(path: str) -> int:
    try:
        return int(pq.read_metadata(path).num_rows)
    except Exception:
        return 0


def load_robomind_action_from_parquet(
    action_path: str,
    *,
    include_gripper: bool = True,
    prefer_control_joints: bool = True,
) -> np.ndarray:
    """Load Tiangong parquet → absolute unified ``[T, 28]`` float32.

    Packs ``control_arm_joints_robot`` / ``arm_joints_robot`` (14) + hand
    grippers (2) into unified 28-D (dexhand zero-filled).
    """
    if not os.path.isfile(action_path):
        raise FileNotFoundError(f"Action parquet does not exist: {action_path}")
    table = pq.read_table(action_path)
    names = set(table.column_names)

    if prefer_control_joints and "control_arm_joints_robot" in names:
        arm_col = "control_arm_joints_robot"
        left_g_col = (
            "control_hand_joints_left"
            if "control_hand_joints_left" in names
            else "hand_joints_left"
        )
        right_g_col = (
            "control_hand_joints_right"
            if "control_hand_joints_right" in names
            else "hand_joints_right"
        )
    elif "arm_joints_robot" in names:
        arm_col = "arm_joints_robot"
        left_g_col = "hand_joints_left"
        right_g_col = "hand_joints_right"
    else:
        raise KeyError(
            f"Missing arm joint columns in {action_path}; cols={sorted(names)}"
        )

    arms = _list_col_to_matrix(table.column(arm_col).to_pylist(), expect_width=SOURCE_JOINT_DIM)
    if arms.shape[1] < SOURCE_JOINT_DIM:
        raise ValueError(
            f"Expected arm dim >= {SOURCE_JOINT_DIM}, got {arms.shape} in {action_path}"
        )

    if include_gripper:
        if left_g_col not in names or right_g_col not in names:
            raise KeyError(
                f"Missing gripper columns ({left_g_col!r}, {right_g_col!r}) in {action_path}"
            )
        left_g = _list_col_to_matrix(
            table.column(left_g_col).to_pylist(), expect_width=1
        ).reshape(-1)
        right_g = _list_col_to_matrix(
            table.column(right_g_col).to_pylist(), expect_width=1
        ).reshape(-1)
    else:
        left_g = np.zeros((arms.shape[0],), dtype=np.float32)
        right_g = np.zeros((arms.shape[0],), dtype=np.float32)

    t = arms.shape[0]
    unified = pack_unified_bimanual_action(
        left_arm=arms[:, 0:7],
        left_gripper=left_g[:t],
        right_arm=arms[:, 7:14],
        right_gripper=right_g[:t],
        left_dexhand=None,
        right_dexhand=None,
    )
    return np.ascontiguousarray(unified)


def _list_col_to_matrix(col_values: Sequence[Any], *, expect_width: Optional[int] = None) -> np.ndarray:
    """Convert a parquet list-column (python lists) → float32 ``[T, D]``."""
    rows = [np.asarray(v, dtype=np.float32).reshape(-1) for v in col_values]
    if not rows:
        return np.zeros((0, int(expect_width or 0)), dtype=np.float32)
    width = int(expect_width) if expect_width is not None else int(max(r.shape[0] for r in rows))
    out = np.zeros((len(rows), width), dtype=np.float32)
    for i, r in enumerate(rows):
        n = min(width, int(r.shape[0]))
        if n > 0:
            out[i, :n] = r[:n]
    return out


class RobomindActionDataset(Dataset):
    """RoboMind RGB (+ action video) with unified 28-D dual-arm actions.

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
        pairs_jsonl: str = DEFAULT_PAIRS_JSONL,
        num_frames: int = 93,
        video_size: tuple[int, int] = (480, 640),
        views: Optional[Sequence[str]] = None,
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
        prefer_control_joints: bool = True,
        skip_empty_instruction: bool = True,
        remap_mounts: bool = True,
    ) -> None:
        Dataset.__init__(self)
        self.pairs_jsonl = str(pairs_jsonl)
        self.sequence_length = int(num_frames)
        self.video_size = tuple(video_size)
        self.views = tuple(views) if views is not None else DEFAULT_VIEWS
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
        self.prefer_control_joints = bool(prefer_control_joints)
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
                arm_scaler=DEFAULT_ARM_DELTA_SCALER,
                gripper_scaler=DEFAULT_GRIPPER_DELTA_SCALER,
            )
        self.action_delta_scaler = normalize_action_delta_scaler(
            action_delta_scaler, self.action_dim
        )

        if not os.path.isfile(self.pairs_jsonl):
            # Try /media/datasets ↔ /datasets remap for the JSONL itself.
            alt = remap_datasets_mount(self.pairs_jsonl)
            if alt != self.pairs_jsonl and os.path.isfile(alt):
                self.pairs_jsonl = alt
            else:
                raise FileNotFoundError(f"RoboMind pairs JSONL not found: {pairs_jsonl}")

        self.samples = self._build_index()
        if not self.samples:
            raise RuntimeError(
                f"No valid RoboMind action episodes from {self.pairs_jsonl} "
                f"(views={list(self.views)}, num_frames={self.sequence_length}, "
                f"temporal_stride={self.temporal_stride}, "
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
            f"RobomindActionDataset: {len(self.samples)} episodes from {self.pairs_jsonl} "
            f"(views={list(self.views)}, num_frames={self.sequence_length}, "
            f"temporal_stride={self.temporal_stride}, "
            f"raw_sequence_length={self.raw_sequence_length}, "
            f"video_size={self.video_size}, action_dim={self.action_dim}, "
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
        return f"{len(self.samples)} RoboMind action samples from {self.pairs_jsonl}"

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve_path(self, path: str) -> str:
        p = str(path or "").strip()
        if not p:
            return p
        if self.remap_mounts:
            p = remap_datasets_mount(p)
        return p

    def _build_index(self) -> list[_RobomindActionSample]:
        samples: list[_RobomindActionSample] = []
        skipped_missing = 0
        skipped_short = 0
        skipped_view = 0
        skipped_empty_cap = 0
        skipped_status = 0
        view_set = set(self.views)

        with open(self.pairs_jsonl, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                status = str(item.get("status") or "").strip().lower()
                if status not in ("ok", "exists", ""):
                    skipped_status += 1
                    continue

                view = str(item.get("selected_view") or "").strip()
                if view_set and view not in view_set:
                    skipped_view += 1
                    continue

                rgb = self._resolve_path(item.get("source_video") or "")
                action_video = self._resolve_path(item.get("action_video") or "")
                action_path = self._resolve_path(item.get("action_path") or "")
                if not rgb or not action_path:
                    skipped_missing += 1
                    continue
                if self.require_action_video and not action_video:
                    skipped_missing += 1
                    continue
                if item.get("source_video_exists") is False:
                    skipped_missing += 1
                    continue
                if item.get("action_exists") is False:
                    skipped_missing += 1
                    continue
                if is_excluded_video_path(rgb, self.exclude_video_paths):
                    skipped_missing += 1
                    continue

                length = _parquet_num_rows(action_path)
                if length <= 0:
                    skipped_missing += 1
                    continue
                if is_too_short_for_window(
                    length, self.raw_sequence_length, self.uniform_sample_episode
                ):
                    skipped_short += 1
                    continue

                caption = _caption_from_item(item)
                if self.skip_empty_instruction and not caption:
                    skipped_empty_cap += 1
                    continue

                samples.append(
                    _RobomindActionSample(
                        video_path=rgb,
                        action_video_path=action_video,
                        action_path=action_path,
                        caption=caption,
                        length=length,
                        view=view,
                        robot=str(item.get("robot") or ""),
                        source_index=int(item.get("source_index") or -1),
                        index_in_jsonl=line_idx,
                        src_fps=None,  # read from container at load time
                    )
                )

        if self.verify_files and samples:
            paths_to_check: list[str] = []
            for s in samples:
                paths_to_check.append(s.video_path)
                paths_to_check.append(s.action_path)
                if self.require_action_video and s.action_video_path:
                    paths_to_check.append(s.action_video_path)
            exists = parallel_path_exists(paths_to_check)
            kept: list[_RobomindActionSample] = []
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
            f"Indexed {len(samples)} RoboMind action episodes "
            f"(skipped_missing={skipped_missing}, skipped_short={skipped_short}, "
            f"skipped_view={skipped_view}, skipped_empty_cap={skipped_empty_cap}, "
            f"skipped_status={skipped_status}, verify_files={self.verify_files}, "
            f"require_action_video={self.require_action_video}, "
            f"temporal_stride={self.temporal_stride}, "
            f"raw_sequence_length={self.raw_sequence_length}, views={list(self.views)})"
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
        """Sample denser raw window then keep every ``temporal_stride``-th index."""
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
        safe_ids = [min(int(i), total_frames - 1) for i in frame_ids]
        frame_data = vr.get_batch(safe_ids).asnumpy().astype(np.uint8)
        vr.seek(0)
        del vr

        frames = torch.from_numpy(frame_data).permute(0, 3, 1, 2)
        frames = self.preprocess(frames)
        frames = torch.clamp(frames * 255.0, 0, 255).to(torch.uint8)
        return frames

    def _load_actions(self, action_path: str, frame_ids: Sequence[int]) -> torch.Tensor:
        unified = load_robomind_action_from_parquet(
            action_path,
            include_gripper=self.include_gripper,
            prefer_control_joints=self.prefer_control_joints,
        )
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
            video = video_thw.permute(1, 0, 2, 3)  # [C, T, H, W]
            _, _, h, w = video.shape

            action = self._load_actions(sample.action_path, frame_ids)
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
                f"Failed to load RoboMind action sample {sample.video_path} "
                f"(view={sample.view}, total failures: {self.num_failed_loads}): {e}\n"
                f"{traceback.format_exc()}",
                rank0_only=False,
            )
            return self[int(np.random.randint(len(self.samples)))]

    @classmethod
    def render_action_video(cls, request: Any) -> np.ndarray:
        """White-bg skeleton ``[T, H, W, 3]`` for one RoboMind gripper episode."""
        import os

        from cosmos_predict2._src.predict2.datasets.local_datasets.action_video_render import (
            ensure_backend_import_path,
        )

        ensure_backend_import_path()
        import batch_render_action_video_gripper as grip

        extra = dict(request.extra or {})
        parquet = request.parquet_path or (
            request.action_path if str(request.action_path).endswith(".parquet") else ""
        )
        item = dict(extra.get("item") or {})
        if not item:
            item = {
                "root": extra.get("root", ""),
                "action": {"all_fields": parquet} if parquet else {},
                "org_path": extra.get("org_path", ""),
                "robot": extra.get("robot", ""),
            }
        elif parquet and not item.get("action"):
            item["action"] = {"all_fields": parquet}
        urdf = grip.get_urdf_for_robot(item)
        grip._init_worker([urdf], request.width, request.height, 86.0, 55.0)
        grip._load_urdf_cached(urdf)
        robot_config = extra.get("robot_config") or grip.get_robot_config(item)
        grip._setup_camera(robot_config["head_pitch"], robot_config["cam_pitch_offset"])
        if parquet and os.path.isfile(parquet):
            arm_all, gripper_all = grip.load_action_arrays_gripper(item)
        elif request.action_path and os.path.isfile(request.action_path):
            from cosmos_predict2._src.predict2.datasets.local_datasets.action_unified_layout import (
                split_unified_bimanual_action,
            )

            parts = split_unified_bimanual_action(np.load(request.action_path))
            arm_all = np.concatenate([parts["left_arm"], parts["right_arm"]], axis=1).astype(np.float64)
            gripper_all = np.stack([parts["left_gripper"], parts["right_gripper"]], axis=1).astype(
                np.float64
            )
        else:
            raise FileNotFoundError(
                f"RoboMind needs action.npy or parquet for online AV: {request.action_path!r} / {parquet!r}"
            )
        max_frames = int(extra.get("max_frames") or 0)
        if max_frames > 0:
            arm_all = arm_all[:max_frames]
            gripper_all = gripper_all[:max_frames]
        bone_col_l, bone_col_r, joint_col_l, joint_col_r = grip.precompute_gripper_draw_colors(
            gripper_all, robot_config
        )
        vec_draws = grip._build_vectorized_draw_lists(
            arm_all,
            gripper_all,
            robot_config,
            False,
            bone_col_l,
            bone_col_r,
            joint_col_l,
            joint_col_r,
        )
        frames = [
            np.ascontiguousarray(grip._compose_skeleton_frame(vec_draws, t, transparent=False))
            for t in range(arm_all.shape[0])
        ]
        if not frames:
            raise RuntimeError(
                f"RoboMind renderer produced 0 frames for {request.action_path or parquet}"
            )
        return np.stack(frames, axis=0).astype(np.uint8)


def default_robomind_action_delta_scaler() -> list[float]:
    return build_action_delta_scaler(
        include_gripper=True,
        include_lift=False,
        arm_scaler=DEFAULT_ARM_DELTA_SCALER,
        gripper_scaler=DEFAULT_GRIPPER_DELTA_SCALER,
    )


__all__ = [
    "DEFAULT_PAIRS_JSONL",
    "DEFAULT_TARGET_FPS",
    "DEFAULT_TEMPORAL_STRIDE",
    "DEFAULT_VIEWS",
    "SOURCE_ARM_DOF",
    "SOURCE_JOINT_DIM",
    "RobomindActionDataset",
    "default_robomind_action_delta_scaler",
    "load_robomind_action_from_parquet",
]
