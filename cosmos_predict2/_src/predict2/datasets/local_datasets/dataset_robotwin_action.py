# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RoboTwin 2.0 action-video dataset with unified bimanual action layout.

Metadata JSONL (``pairs_jsonl``) each line::

    {
      "rgb": "<abs path to head_camera mp4>",
      "action_video": "<abs path to rendered action mp4>",
      "hdf5": "<abs path to episode hdf5>",
      "lang": "<instruction>",
      "frame_nums": <int>,
      "fps": 30.0,
      ...
    }

Raw joint actions in HDF5 (Aloha AgileX)::

    joint_action/vector  [T, 14]
      = left_arm(6) + left_gripper(1) + right_arm(6) + right_gripper(1)

Packed into the canonical unified 28-D layout::

    left_arm(7) + left_gripper(1) + left_dexhand(6)
    + right_arm(7) + right_gripper(1) + right_dexhand(6)

with the 7th arm joint zero-padded (source is 6-DoF) and dexhand slots zero.

Temporal sampling (``temporal_stride``, default 2)::

    RoboTwin is ~30 fps with relatively short episodes (median ~166 frames).
    Stride 2 → model rate ~15 fps (near ``target_fps=16``) and keeps episodes
    with ``frame_nums >= num_frames * 2`` (~60k / ~141k ≈ 43% at num_frames=93).
    RSW uses stride 4 because its episodes are much longer; stride 4 here would
    drop to ~12% usable episodes (~17k), so 2 is the practical default.
"""

from __future__ import annotations

import json
import os
import traceback
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union

import h5py
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
    UNIFIED_SIDE_DIM,
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

DEFAULT_PAIRS_JSONL = (
    "/media/datasets/zhangshilin/RoboTwin2.0/robotwin_action_video_rgb_pairs.jsonl"
)
# PE rollouts: physics@30 collect → prepare downsample to src@15 pairs.
DEFAULT_PE_PAIRS_JSONL = (
    "/mnt/jackzou/Pelican-Sim/third_party/robotwin/policy/pi05/"
    "wm_compare_stack_bowls_two_orig150/policy_eval/action_video/out/"
    "orig10_ckpt10000_src15_n100/robotwin_action_video_rgb_pairs.jsonl"
)
DEFAULT_TARGET_FPS = 16.0
# Source ~30 fps; stride 2 → ~15 fps (see module docstring).
DEFAULT_TEMPORAL_STRIDE = 2
# PE src@15 pairs are already at model rate → stride 1.
DEFAULT_PE_TEMPORAL_STRIDE = 1
# Aloha AgileX dual-arm packing in joint_action/vector.
SOURCE_ARM_DOF = 6
SOURCE_ACTION_DIM = SOURCE_ARM_DOF * 2 + 2  # 14
HDF5_ACTION_VECTOR_KEY = "joint_action/vector"


def summarize_robotwin_pairs_fps(pairs_jsonl: str) -> dict[str, float]:
    """Compute per-episode fps / length stats from a Robotwin pairs JSONL.

    Prefers the ``fps`` field written by prepare; falls back to probing the
    RGB mp4 with decord when metadata is missing.
    """
    fps_vals: list[float] = []
    frame_nums: list[int] = []
    probed = 0
    with open(pairs_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            fn = int(item.get("frame_nums") or 0)
            if fn > 0:
                frame_nums.append(fn)
            fps = item.get("fps")
            try:
                fps_f = float(fps) if fps is not None else float("nan")
            except (TypeError, ValueError):
                fps_f = float("nan")
            if not np.isfinite(fps_f) or fps_f <= 0:
                rgb = item.get("rgb") or item.get("video") or ""
                if rgb and os.path.isfile(rgb):
                    try:
                        vr = VideoReader(rgb, ctx=cpu(0), num_threads=1)
                        fps_f = float(vr.get_avg_fps())
                        probed += 1
                    except Exception:
                        fps_f = float("nan")
            if np.isfinite(fps_f) and fps_f > 0:
                fps_vals.append(float(fps_f))
    if not fps_vals:
        raise RuntimeError(f"No valid fps values in {pairs_jsonl}")
    fps_arr = np.asarray(fps_vals, dtype=np.float64)
    fn_arr = np.asarray(frame_nums, dtype=np.int64) if frame_nums else np.zeros(0, dtype=np.int64)
    stats = {
        "n_episodes": float(len(fps_vals)),
        "fps_mean": float(fps_arr.mean()),
        "fps_median": float(np.median(fps_arr)),
        "fps_min": float(fps_arr.min()),
        "fps_max": float(fps_arr.max()),
        "n_probed_video_fps": float(probed),
    }
    if fn_arr.size:
        stats.update(
            {
                "frame_nums_mean": float(fn_arr.mean()),
                "frame_nums_median": float(np.median(fn_arr)),
                "frame_nums_min": float(fn_arr.min()),
                "frame_nums_max": float(fn_arr.max()),
            }
        )
    return stats


@dataclass(frozen=True)
class _RobotwinActionSample:
    video_path: str
    action_video_path: str
    hdf5_path: str
    caption: str
    length: int
    uuid: str
    task_name: str
    index_in_jsonl: int
    src_fps: Optional[float]
    # JSONL ``robot_config``, e.g. aloha-agilex_clean_500 / aloha-agilex_randomized_500.
    robot_config: str = ""


class RobotwinActionDataset(Dataset):
    """RoboTwin RGB (+ optional action video) with unified 28-D action values.

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
        source_arm_dof: int = SOURCE_ARM_DOF,
        hdf5_action_key: str = HDF5_ACTION_VECTOR_KEY,
        skip_empty_instruction: bool = True,
    ) -> None:
        Dataset.__init__(self)
        self.pairs_jsonl = pairs_jsonl
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
        self.hdf5_action_key = str(hdf5_action_key)
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

        if not os.path.isfile(self.pairs_jsonl):
            raise FileNotFoundError(f"RoboTwin pairs JSONL not found: {self.pairs_jsonl}")

        self.samples = self._build_index()
        if not self.samples:
            raise RuntimeError(
                f"No valid RoboTwin action episodes from {self.pairs_jsonl} "
                f"(num_frames={self.sequence_length}, "
                f"temporal_stride={self.temporal_stride}, "
                f"raw_sequence_length={self.raw_sequence_length}, "
                f"load_action_video={self.load_action_video})."
            )

        scaler_msg = (
            "None"
            if self.action_delta_scaler is None
            else [round(float(x), 4) for x in self.action_delta_scaler.tolist()]
        )
        log.info(
            f"RobotwinActionDataset: {len(self.samples)} episodes from {self.pairs_jsonl} "
            f"(num_frames={self.sequence_length}, temporal_stride={self.temporal_stride}, "
            f"raw_sequence_length={self.raw_sequence_length}, "
            f"video_size={self.video_size}, action_dim={self.action_dim}, "
            f"source_arm_dof={self.source_arm_dof}, "
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
        return f"{len(self.samples)} RoboTwin action samples from {self.pairs_jsonl}"

    def __len__(self) -> int:
        return len(self.samples)

    def _caption_from_item(self, item: dict[str, Any]) -> str:
        for key in ("lang", "instruction", "task_name"):
            val = item.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        return ""

    def _build_index(self) -> list[_RobotwinActionSample]:
        samples: list[_RobotwinActionSample] = []
        skipped_missing = 0
        skipped_short = 0
        skipped_empty_cap = 0

        with open(self.pairs_jsonl, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                rgb = str(item.get("rgb") or "").strip()
                action_video = str(item.get("action_video") or "").strip()
                hdf5 = str(item.get("hdf5") or "").strip()
                if not rgb or not hdf5:
                    skipped_missing += 1
                    continue
                if self.require_action_video and not action_video:
                    skipped_missing += 1
                    continue
                if is_excluded_video_path(rgb, self.exclude_video_paths):
                    skipped_missing += 1
                    continue

                length = int(item.get("frame_nums") or 0)
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

                samples.append(
                    _RobotwinActionSample(
                        video_path=rgb,
                        action_video_path=action_video,
                        hdf5_path=hdf5,
                        caption=caption,
                        length=length,
                        uuid=str(item.get("uuid") or ""),
                        task_name=str(item.get("task_name") or ""),
                        index_in_jsonl=line_idx,
                        src_fps=src_fps,
                        robot_config=str(item.get("robot_config") or ""),
                    )
                )

        if self.verify_files and samples:
            paths_to_check: list[str] = []
            for s in samples:
                paths_to_check.append(s.video_path)
                paths_to_check.append(s.hdf5_path)
                if self.require_action_video and s.action_video_path:
                    paths_to_check.append(s.action_video_path)

            exists = parallel_path_exists(paths_to_check)
            kept: list[_RobotwinActionSample] = []
            cursor = 0
            before = len(samples)
            for s in samples:
                ok_video = exists[cursor]
                ok_hdf5 = exists[cursor + 1]
                cursor += 2
                ok_av = True
                if self.require_action_video and s.action_video_path:
                    ok_av = exists[cursor]
                    cursor += 1
                if ok_video and ok_hdf5 and ok_av:
                    kept.append(s)
            samples = kept
            skipped_missing += before - len(samples)

        log.info(
            f"Indexed {len(samples)} RoboTwin action episodes "
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

    def _load_actions(
        self, hdf5_path: str, frame_ids: Sequence[int]
    ) -> torch.Tensor:
        if not os.path.isfile(hdf5_path):
            raise FileNotFoundError(f"HDF5 file does not exist: {hdf5_path}")
        with h5py.File(hdf5_path, "r") as h:
            if self.hdf5_action_key not in h:
                raise KeyError(
                    f"Missing action key {self.hdf5_action_key!r} in {hdf5_path}; "
                    f"keys sample={list(h.keys())[:20]}"
                )
            actions = np.asarray(h[self.hdf5_action_key][:], dtype=np.float32)

        if actions.ndim != 2:
            raise ValueError(
                f"Expected action matrix [T, D], got shape {actions.shape} in {hdf5_path}"
            )
        if actions.shape[1] != SOURCE_ACTION_DIM and self.source_arm_dof == SOURCE_ARM_DOF:
            log.warning(
                f"Unexpected action dim {actions.shape[1]} (expected {SOURCE_ACTION_DIM}) "
                f"in {hdf5_path}; packing with num_arms=2, arm_dof={self.source_arm_dof}."
            )

        n = actions.shape[0]
        safe_ids = [min(max(int(i), 0), n - 1) for i in frame_ids]
        selected = actions[safe_ids]
        unified = to_unified_bimanual_action(
            selected,
            num_arms=2,
            include_gripper=self.include_gripper,
            arm_dof=self.source_arm_dof,
        )
        if unified.shape[-1] != self.action_dim:
            raise RuntimeError(
                f"Unified action dim {unified.shape[-1]} != {self.action_dim} from {hdf5_path}"
            )
        return torch.from_numpy(np.ascontiguousarray(unified)).float()

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        try:
            video_thw, frame_ids, fps = self._load_rgb_with_ids(
                sample.video_path, sample.src_fps
            )
            video = video_thw.permute(1, 0, 2, 3)  # [C, T, H, W]
            _, _, h, w = video.shape

            action = self._load_actions(sample.hdf5_path, frame_ids)
            if action.shape[0] != self.sequence_length:
                raise RuntimeError(
                    f"Action length {action.shape[0]} != num_frames {self.sequence_length}"
                )
            if action.shape[1] != self.action_dim:
                raise RuntimeError(f"Action dim {action.shape[1]} != {self.action_dim}")

            # 7th joint is pad — should be ~0 for 6-DoF Aloha sources.
            # 28-D layout is L7+Lg+Ldex6+R7+Rg+Rdex6; do not use 8+dof
            # (that was the pre-dexhand 16-D offset and hits RightArm_Joint1).
            if self.source_arm_dof < UNIFIED_ARM_DOF:
                pad_l = action[:, self.source_arm_dof]
                pad_r = action[:, UNIFIED_SIDE_DIM + self.source_arm_dof]
                if not (torch.all(pad_l == 0) and torch.all(pad_r == 0)):
                    log.warning(
                        f"Non-zero padded joint-7 in {sample.hdf5_path} "
                        f"(L max={float(pad_l.abs().max()):.4g}, "
                        f"R max={float(pad_r.abs().max()):.4g})"
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
                f"Failed to load RoboTwin action sample {sample.video_path} "
                f"(total failures: {self.num_failed_loads}): {e}\n{traceback.format_exc()}",
                rank0_only=False,
            )
            return self[int(np.random.randint(len(self.samples)))]

    @classmethod
    def render_action_video(cls, request: Any) -> np.ndarray:
        """White-bg skeleton ``[T, H, W, 3]`` for one RoboTwin episode."""
        import os

        from cosmos_predict2._src.predict2.datasets.local_datasets.action_video_render import (
            BACKEND_ROOT,
            ensure_backend_import_path,
        )

        ensure_backend_import_path(BACKEND_ROOT / "robotwin")
        import batch_render_action_video_robotwin as tw

        extra = dict(request.extra or {})
        hdf5 = request.hdf5_path
        item = dict(extra.get("item") or {})
        if hdf5 and not item.get("action"):
            item = {
                "root": extra.get("hdf5_root") or extra.get("root") or "",
                "action": {"hdf5_path": hdf5},
                "hdf5_path": hdf5,
            }
        cfg = tw._load_robot_config(tw.DEFAULT_CONFIG)
        pose = tw._build_robot_pose_matrix(cfg)
        scale = tuple(cfg.get("gripper_scale") or tw.DEFAULT_GRIPPER_SCALE)
        tw._init_worker(tw.DEFAULT_URDF, request.width, request.height, pose, scale)
        if hdf5 and os.path.isfile(hdf5):
            left_arm, left_gripper, right_arm, right_gripper, intrinsic, extrinsics = tw.load_all_from_hdf5(
                item,
                extra.get("hdf5_root", ""),
                extra.get("camera_name", "head_camera"),
            )
        elif request.action_path and os.path.isfile(request.action_path):
            from cosmos_predict2._src.predict2.datasets.local_datasets.action_unified_layout import (
                split_unified_bimanual_action,
            )

            parts = split_unified_bimanual_action(np.load(request.action_path))
            left_arm = np.asarray(parts["left_arm"][:, :6], dtype=np.float64)
            right_arm = np.asarray(parts["right_arm"][:, :6], dtype=np.float64)
            left_gripper = np.asarray(parts["left_gripper"], dtype=np.float64)
            right_gripper = np.asarray(parts["right_gripper"], dtype=np.float64)
            intrinsic = np.asarray(extra.get("K") or extra.get("intrinsic"), dtype=np.float64)
            extrinsics = extra.get("E") or extra.get("E_world2cam") or extra.get("extrinsic")
            if extrinsics is None:
                raise FileNotFoundError(
                    "RoboTwin online AV from action.npy needs K and E/E_world2cam in meta.json"
                )
            extrinsics = np.asarray(extrinsics, dtype=np.float64)
        else:
            raise FileNotFoundError(
                f"RoboTwin needs action.npy+K/E or hdf5 for online AV: {request.action_path!r} / {hdf5!r}"
            )
        max_frames = int(extra.get("max_frames") or 0)
        if max_frames > 0:
            n = min(int(left_arm.shape[0]), max_frames)
            left_arm, left_gripper = left_arm[:n], left_gripper[:n]
            right_arm, right_gripper = right_arm[:n], right_gripper[:n]
            if getattr(extrinsics, "ndim", 0) == 3:
                extrinsics = extrinsics[:n]
        tw._setup_camera_from_params(intrinsic, extrinsics, request.width, request.height)
        closed_l, settled_l = tw._gripper_state_masks(1.0 - np.asarray(left_gripper, dtype=np.float64))
        closed_r, settled_r = tw._gripper_state_masks(1.0 - np.asarray(right_gripper, dtype=np.float64))
        bones, joints = tw._precompute_skeleton_draw(
            left_arm,
            left_gripper,
            right_arm,
            right_gripper,
            extrinsics,
            closed_l,
            settled_l,
            closed_r,
            settled_r,
            transparent=False,
        )
        t = int(left_arm.shape[0])
        frames = np.empty((t, request.height, request.width, 3), dtype=np.uint8)
        white = np.full((request.height, request.width, 3), 255, dtype=np.uint8)
        for i in range(t):
            frames[i] = tw._blit_skeleton_frame(white.copy(), bones, joints, i)
        return frames


class RobotwinPEActionDataset(RobotwinActionDataset):
    """RoboTwin PE rollout pairs for WM fine-tune (src@15, ``temporal_stride=1``).

    Defaults to the ``orig10_ckpt10000_src15_n100`` prepare export. Same packing /
    keys as ``RobotwinActionDataset``; only the pairs path and default stride differ.
    On construction, logs mean / median fps and frame lengths over all episodes.
    """

    def __init__(
        self,
        pairs_jsonl: str = DEFAULT_PE_PAIRS_JSONL,
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
        temporal_stride: int = DEFAULT_PE_TEMPORAL_STRIDE,
        source_arm_dof: int = SOURCE_ARM_DOF,
        hdf5_action_key: str = HDF5_ACTION_VECTOR_KEY,
        skip_empty_instruction: bool = True,
    ) -> None:
        pairs_jsonl = os.environ.get("ROBOTWIN_PE_PAIRS_JSONL", pairs_jsonl)
        if not os.path.isfile(pairs_jsonl):
            raise FileNotFoundError(f"RoboTwin PE pairs JSONL not found: {pairs_jsonl}")

        fps_stats = summarize_robotwin_pairs_fps(pairs_jsonl)
        log.info(
            "RobotwinPEActionDataset episode fps stats "
            f"(n={int(fps_stats['n_episodes'])}, probed_video={int(fps_stats['n_probed_video_fps'])}): "
            f"mean={fps_stats['fps_mean']:.4f} median={fps_stats['fps_median']:.4f} "
            f"min={fps_stats['fps_min']:.4f} max={fps_stats['fps_max']:.4f}"
        )
        if "frame_nums_mean" in fps_stats:
            log.info(
                "RobotwinPEActionDataset episode length stats: "
                f"mean={fps_stats['frame_nums_mean']:.1f} "
                f"median={fps_stats['frame_nums_median']:.1f} "
                f"min={int(fps_stats['frame_nums_min'])} max={int(fps_stats['frame_nums_max'])}"
            )
        self.episode_fps_stats = fps_stats

        super().__init__(
            pairs_jsonl=pairs_jsonl,
            num_frames=num_frames,
            video_size=video_size,
            verify_files=verify_files,
            target_fps=target_fps,
            uniform_sample_episode=uniform_sample_episode,
            start_from_first_frame_prob=start_from_first_frame_prob,
            exclude_video_paths=exclude_video_paths,
            load_action_video=load_action_video,
            include_gripper=include_gripper,
            require_action_video=require_action_video,
            action_delta_scaler=action_delta_scaler,
            temporal_stride=temporal_stride,
            source_arm_dof=source_arm_dof,
            hdf5_action_key=hdf5_action_key,
            skip_empty_instruction=skip_empty_instruction,
        )
        log.info(
            f"RobotwinPEActionDataset ready: {len(self.samples)} usable episodes "
            f"(temporal_stride={self.temporal_stride}, pairs={self.pairs_jsonl})"
        )


def default_robotwin_action_delta_scaler() -> list[float]:
    return build_action_delta_scaler(
        include_gripper=True,
        include_lift=False,
        arm_scaler=DEFAULT_ARM_DELTA_SCALER,
        gripper_scaler=DEFAULT_GRIPPER_DELTA_SCALER,
    )


__all__ = [
    "DEFAULT_PAIRS_JSONL",
    "DEFAULT_PE_PAIRS_JSONL",
    "DEFAULT_PE_TEMPORAL_STRIDE",
    "DEFAULT_TEMPORAL_STRIDE",
    "HDF5_ACTION_VECTOR_KEY",
    "SOURCE_ACTION_DIM",
    "SOURCE_ARM_DOF",
    "RobotwinActionDataset",
    "RobotwinPEActionDataset",
    "UNIFIED_ACTION_DIM",
    "UNIFIED_ACTION_NAMES",
    "UNIFIED_ARM_DOF",
    "default_robotwin_action_delta_scaler",
    "summarize_robotwin_pairs_fps",
    "to_unified_bimanual_action",
]
