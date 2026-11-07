# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test dataset for nwww oven plate-phase (LeRobot v2.1).

Root (default)::

    /media/datasets/zhangshilin/oven_0810_platephase/

    meta/info.json
    meta/episodes.jsonl
    data/chunk-000/episode_000000.parquet     # action [T, 16]
    videos/chunk-000/observation.images.head_camera/episode_000000.mp4
    action_video/chunk-000/observation.images.head_camera/episode_000000.mp4

Source action is 16-D Tiangong dual-arm::

    left_arm(7) + right_arm(7) + gripper_left(1) + gripper_right(1)

Packed into the canonical unified 28-D layout (dexhand zero-filled)::

    left_arm(7) + left_gripper(1) + left_dexhand(6)
    + right_arm(7) + right_gripper(1) + right_dexhand(6)

RGB is **first frame only** (inference conditioning). Action + action-video
are loaded for the full episode (after ``temporal_stride``). Default stride=8
maps the 1760-frame (~28.6 fps) episode to 220 model frames (~3.6 fps),
which is 3 AR chunks at chunk_size=93 / overlap=1 (2-chunk band is 94–185).
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
    GRIPPER_UNIT_RAW_MAX,
    UNIFIED_ACTION_DIM,
    UNIFIED_ACTION_NAMES,
    UNIFIED_ARM_DOF,
    pack_unified_bimanual_action,
)
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_real_source_world_action import (
    DEFAULT_ARM_DELTA_SCALER,
    DEFAULT_GRIPPER_DELTA_SCALER,
    build_action_delta_scaler,
    normalize_action_delta_scaler,
)
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_robomind_action import (
    _list_col_to_matrix,
)
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_utils import (
    ResizePreprocess,
    ToTensorVideo,
    is_too_short_for_window,
)

DEFAULT_DATASET_DIR = "/media/datasets/zhangshilin/oven_0810_platephase"
DEFAULT_CAMERA_KEY = "observation.images.head_camera"
DEFAULT_TARGET_FPS = 16.0
# Native 1760 @ ~28.6 fps → stride 8 = 220 frames ≈ 3 AR loops
# (chunk=93, overlap=1; 3-chunk band is 186–277 frames).
DEFAULT_TEMPORAL_STRIDE = 8
SOURCE_ARM_DOF = UNIFIED_ARM_DOF  # 7
SOURCE_JOINT_DIM = SOURCE_ARM_DOF * 2  # 14
SOURCE_ACTION_DIM = SOURCE_JOINT_DIM + 2  # 16


@dataclass(frozen=True)
class _OvenPlatephaseSample:
    video_path: str
    action_video_path: str
    parquet_path: str
    caption: str
    length: int
    camera_key: str
    episode_index: int
    src_fps: Optional[float]


def load_oven_platephase_action_from_parquet(parquet_path: str) -> np.ndarray:
    """Load oven plate-phase parquet → absolute unified ``[T, 28]`` float32.

    Prefers ``action`` (16-D: L7+R7+gL+gR). Falls back to
    ``arm_joints_robot`` (14) + ``hand_joints_{left,right}``.
    Grippers are already unit aperture; dexhand slots stay 0.
    """
    if not os.path.isfile(parquet_path):
        raise FileNotFoundError(f"Action parquet does not exist: {parquet_path}")
    table = pq.read_table(parquet_path)
    names = set(table.column_names)

    if "action" in names:
        raw = _list_col_to_matrix(
            table.column("action").to_pylist(), expect_width=SOURCE_ACTION_DIM
        )
        if raw.shape[1] < SOURCE_ACTION_DIM:
            raise ValueError(
                f"Expected action dim >= {SOURCE_ACTION_DIM}, got {raw.shape} in {parquet_path}"
            )
        left_arm = raw[:, 0:7]
        right_arm = raw[:, 7:14]
        left_g = raw[:, 14]
        right_g = raw[:, 15]
    elif "arm_joints_robot" in names:
        arms = _list_col_to_matrix(
            table.column("arm_joints_robot").to_pylist(), expect_width=SOURCE_JOINT_DIM
        )
        if "hand_joints_left" not in names or "hand_joints_right" not in names:
            raise KeyError(
                f"Missing gripper columns in {parquet_path}; cols={sorted(names)}"
            )
        left_g = _list_col_to_matrix(
            table.column("hand_joints_left").to_pylist(), expect_width=1
        ).reshape(-1)
        right_g = _list_col_to_matrix(
            table.column("hand_joints_right").to_pylist(), expect_width=1
        ).reshape(-1)
        left_arm = arms[:, 0:7]
        right_arm = arms[:, 7:14]
    else:
        raise KeyError(
            f"Missing action / arm_joints_robot in {parquet_path}; cols={sorted(names)}"
        )

    t = int(left_arm.shape[0])
    unified = pack_unified_bimanual_action(
        left_arm=left_arm,
        left_gripper=left_g[:t],
        right_arm=right_arm,
        right_gripper=right_g[:t],
        left_dexhand=None,
        right_dexhand=None,
        gripper_raw_max=GRIPPER_UNIT_RAW_MAX,
        normalize_gripper=True,
        normalize_dexhand=False,
    )
    return np.ascontiguousarray(unified)


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _format_lerobot_path(template: str, *, episode_index: int, chunk_size: int) -> str:
    episode_chunk = int(episode_index) // max(int(chunk_size), 1)
    return template.format(episode_chunk=episode_chunk, episode_index=int(episode_index))


class OvenPlatephaseActionDataset(Dataset):
    """Oven plate-phase RGB-first-frame + full action / action-video.

    ``__getitem__`` (training-style window from t=0 after stride)::

        video           uint8 [C, T, H, W]  — first RGB frame repeated T times
        action          float [T, 28]
        action_video    uint8 [C, T, H, W]  — real action-video clip
        action_state / action_delta / ai_caption / fps / ...

    Inference should use ``.samples`` + ``load_oven_platephase_action_from_parquet``
    and decode the full action-video; RGB stays first-frame-only.
    """

    def __init__(
        self,
        dataset_dir: str = DEFAULT_DATASET_DIR,
        num_frames: int = 93,
        video_size: tuple[int, int] = (480, 640),
        target_fps: float = DEFAULT_TARGET_FPS,
        uniform_sample_episode: bool = False,
        start_from_first_frame_prob: float = 1.0,
        verify_files: bool = True,
        load_action_video: bool = True,
        include_gripper: bool = True,
        require_action_video: Optional[bool] = None,
        action_delta_scaler: Optional[Union[float, Sequence[float]]] = None,
        temporal_stride: int = DEFAULT_TEMPORAL_STRIDE,
        camera_key: str = DEFAULT_CAMERA_KEY,
        exclude_video_paths: Optional[Union[str, list[str]]] = None,  # unused, API symmetry
        source_arm_dof: Optional[int] = None,  # unused, API symmetry
    ) -> None:
        del exclude_video_paths, source_arm_dof
        self.dataset_dir = os.path.abspath(dataset_dir)
        self.sequence_length = int(num_frames)
        self.video_size = (int(video_size[0]), int(video_size[1]))
        self.target_fps = float(target_fps)
        self.uniform_sample_episode = bool(uniform_sample_episode)
        self.start_from_first_frame_prob = float(start_from_first_frame_prob)
        self.verify_files = bool(verify_files)
        self.load_action_video = bool(load_action_video)
        self.include_gripper = bool(include_gripper)
        self.require_action_video = (
            bool(load_action_video) if require_action_video is None else bool(require_action_video)
        )
        self.temporal_stride = max(int(temporal_stride), 1)
        self.camera_key = str(camera_key)
        self.action_dim = UNIFIED_ACTION_DIM
        self.num_failed_loads = 0
        self.action_delta_scaler = normalize_action_delta_scaler(
            action_delta_scaler, self.action_dim
        )

        self.preprocess = T.Compose(
            [
                ToTensorVideo(),
                ResizePreprocess(self.video_size),
            ]
        )

        info_path = os.path.join(self.dataset_dir, "meta", "info.json")
        episodes_path = os.path.join(self.dataset_dir, "meta", "episodes.jsonl")
        if not os.path.isfile(info_path):
            raise FileNotFoundError(f"Missing {info_path}")
        if not os.path.isfile(episodes_path):
            raise FileNotFoundError(f"Missing {episodes_path}")

        self.info = _read_json(info_path)
        self.samples = self._build_index(episodes_path)
        if not self.samples:
            raise RuntimeError(f"No oven plate-phase episodes indexed under {self.dataset_dir}")
        log.info(f"{self}")

    def __len__(self) -> int:
        return len(self.samples)

    def __str__(self) -> str:
        return (
            f"{len(self.samples)} oven plate-phase episode(s) from {self.dataset_dir} "
            f"(stride={self.temporal_stride}, rgb=first_frame_only)"
        )

    def _resolve(self, rel: str) -> str:
        rel = str(rel or "").strip()
        if not rel:
            return ""
        return rel if os.path.isabs(rel) else os.path.join(self.dataset_dir, rel)

    def _build_index(self, episodes_path: str) -> list[_OvenPlatephaseSample]:
        chunk_size = int(self.info.get("chunks_size") or 1000)
        data_tmpl = str(self.info.get("data_path") or "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
        video_tmpl = str(
            self.info.get("video_path")
            or "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        )
        av_tmpl = str(
            self.info.get("action_video_path")
            or "action_video/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        )
        src_fps = self.info.get("fps")
        src_fps_f = float(src_fps) if src_fps is not None else None

        samples: list[_OvenPlatephaseSample] = []
        for rec in _read_jsonl(episodes_path):
            ep_idx = int(rec.get("episode_index", 0))
            videos = rec.get("videos") or {}
            avs = rec.get("action_video") or {}
            rgb_rel = videos.get(self.camera_key) or video_tmpl.replace("{video_key}", self.camera_key)
            if "{episode" in rgb_rel:
                rgb_rel = _format_lerobot_path(rgb_rel, episode_index=ep_idx, chunk_size=chunk_size)
            av_rel = avs.get(self.camera_key) or av_tmpl.replace("{video_key}", self.camera_key)
            if "{episode" in av_rel:
                av_rel = _format_lerobot_path(av_rel, episode_index=ep_idx, chunk_size=chunk_size)
            pq_rel = _format_lerobot_path(data_tmpl, episode_index=ep_idx, chunk_size=chunk_size)

            rgb_path = self._resolve(rgb_rel)
            av_path = self._resolve(av_rel)
            pq_path = self._resolve(pq_rel)

            if self.verify_files:
                if not os.path.isfile(rgb_path):
                    log.warning(f"Skip episode {ep_idx}: missing RGB {rgb_path}")
                    continue
                if not os.path.isfile(pq_path):
                    log.warning(f"Skip episode {ep_idx}: missing parquet {pq_path}")
                    continue
                if self.require_action_video and not os.path.isfile(av_path):
                    log.warning(f"Skip episode {ep_idx}: missing action_video {av_path}")
                    continue

            length = int(rec.get("length") or 0)
            if length <= 0 and os.path.isfile(pq_path):
                length = int(pq.read_table(pq_path, columns=["frame_index"]).num_rows)
            if is_too_short_for_window(
                (length + self.temporal_stride - 1) // self.temporal_stride,
                self.sequence_length,
                self.uniform_sample_episode,
            ):
                log.warning(
                    f"Skip episode {ep_idx}: length={length} stride={self.temporal_stride} "
                    f"< num_frames={self.sequence_length}"
                )
                continue

            tasks = rec.get("tasks") or []
            caption = str(tasks[0]).strip() if tasks else "put the plate and donut into the toaster oven"

            samples.append(
                _OvenPlatephaseSample(
                    video_path=rgb_path,
                    action_video_path=av_path if os.path.isfile(av_path) else "",
                    parquet_path=pq_path,
                    caption=caption,
                    length=length,
                    camera_key=self.camera_key,
                    episode_index=ep_idx,
                    src_fps=src_fps_f,
                )
            )
        return samples

    def _load_rgb_first_frame(self, path: str) -> torch.Tensor:
        """Decode RGB frame 0 only → uint8 ``[1, C, H, W]`` (THWC preprocess path)."""
        vr = VideoReader(path, ctx=cpu(0))
        frame = torch.from_numpy(vr[0].asnumpy())  # HWC uint8
        del vr
        video = frame.unsqueeze(0).permute(0, 3, 1, 2)  # TCHW
        video = self.preprocess(video)
        if video.dtype != torch.uint8:
            video = (video * 255.0).clamp(0, 255).to(torch.uint8)
        return video

    def _load_action_video_ids(self, path: str, frame_ids: Sequence[int]) -> torch.Tensor:
        vr = VideoReader(path, ctx=cpu(0))
        n = len(vr)
        ids = [min(max(int(i), 0), n - 1) for i in frame_ids]
        frames = vr.get_batch(ids).asnumpy()  # THWC
        del vr
        video = torch.from_numpy(np.asarray(frames)).permute(0, 3, 1, 2)
        video = self.preprocess(video)
        if video.dtype != torch.uint8:
            video = (video * 255.0).clamp(0, 255).to(torch.uint8)
        return video

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        try:
            rgb = self._load_rgb_first_frame(sample.video_path)  # [1,C,H,W]
            rgb = rgb.repeat(self.sequence_length, 1, 1, 1)
            video = rgb.permute(1, 0, 2, 3)  # [C,T,H,W]
            _, _, h, w = video.shape

            unified = load_oven_platephase_action_from_parquet(sample.parquet_path)
            stride = self.temporal_stride
            unified = unified[::stride]
            if unified.shape[0] < self.sequence_length:
                raise RuntimeError(
                    f"Strided action T={unified.shape[0]} < num_frames={self.sequence_length}"
                )
            action = torch.from_numpy(np.ascontiguousarray(unified[: self.sequence_length])).float()
            frame_ids = list(range(0, self.sequence_length * stride, stride))

            action_state = action[0]
            if action.shape[0] > 1:
                action_delta = action[1:] - action[:-1]
            else:
                action_delta = torch.zeros(1, self.action_dim, dtype=action.dtype)
            if self.action_delta_scaler is not None:
                action_delta = action_delta * self.action_delta_scaler.to(
                    dtype=action_delta.dtype, device=action_delta.device
                )

            fps = float(sample.src_fps or self.target_fps) / float(stride)
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
                    raise FileNotFoundError(f"Missing action_video for {sample.video_path}")
                av = self._load_action_video_ids(sample.action_video_path, frame_ids)
                out["action_video"] = av.permute(1, 0, 2, 3)
            return out
        except Exception as e:
            self.num_failed_loads += 1
            log.warning(
                f"Failed to load oven plate-phase sample {sample.video_path} "
                f"(total failures: {self.num_failed_loads}): {e}\n{traceback.format_exc()}",
                rank0_only=False,
            )
            if len(self.samples) <= 1:
                raise
            return self[int(np.random.randint(len(self.samples)))]


def default_oven_platephase_action_delta_scaler() -> list[float]:
    return build_action_delta_scaler(
        include_gripper=True,
        include_lift=False,
        arm_scaler=DEFAULT_ARM_DELTA_SCALER,
        gripper_scaler=DEFAULT_GRIPPER_DELTA_SCALER,
    )


__all__ = [
    "DEFAULT_DATASET_DIR",
    "DEFAULT_TEMPORAL_STRIDE",
    "SOURCE_ACTION_DIM",
    "OvenPlatephaseActionDataset",
    "default_oven_platephase_action_delta_scaler",
    "load_oven_platephase_action_from_parquet",
    "UNIFIED_ACTION_DIM",
    "UNIFIED_ACTION_NAMES",
]
