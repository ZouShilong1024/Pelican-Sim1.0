# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RoboDojo (dual ARX-X5 sim) action-video dataset.

Source: LeRobot v3.0 export
``/datasets/UniEmbodied-Bench/simulation/dataset/policy_mix24_clean2400_sdk_ee20_25hz_instruction_20261029_v2``
(24 scenes x 100 episodes, 25 fps). Concatenated / mixed-codec LeRobot videos are
re-exported per episode by
``action_video_backends/robodojo/batch_render_action_video_robodojo.py``,
which also writes the pairs JSONL consumed here::

    {
      "rgb": "<abs path to per-episode cam_high mp4 (H264, 25 fps)>",
      "action_video": "<abs path to rendered skeleton mp4>",
      "actions": "<abs path to per-episode npz>",
      "hdf5": "<abs path to raw RoboDojo HDF5 (provenance only)>",
      "lang": "<instruction>",
      "frame_nums": <int>,
      "fps": 25.0,
      "start_frame": <int>,   # optional, see below
      ...
    }

``start_frame`` (imitate_sorting_sequence only): the episode opens with a scripted
Franka demonstrator whose joints are not recorded, so it cannot appear in the action
video while the ARX arms sit still. Windows are sampled from
``[start_frame, frame_nums)`` only (ARX first motion + 10 frames).

Actions come from ``npz["image_joint"]`` [T, 14]
= left_arm(6) + left_gripper01(1) + right_arm(6) + right_gripper01(1): the joint state
at the capture time of RGB frame t, the same semantics as RoboTwin ``joint_action``
(qpos read when the picture is taken). For both official and self-collected episodes
this is ``observation.joint[t]``, verified by projecting the skeleton onto the frames
(``action.joint[t]`` leads the image by one frame). gripper01: 1 = open, same
direction as RoboTwin. Packed into the unified
28-D layout exactly like RoboTwin (6-DoF arms, joint-7 / dexhand zero).

Temporal sampling (``temporal_stride``, default 2)::

    25 fps source, median episode 462 frames. Stride 2 → 12.5 fps, 7.4 s clips,
    keeps 2141 / 2400 episodes (97.5% of frames). Per-model-frame arm delta
    p95 ≈ 0.20 rad, matching RoboTwin at its stride 2, so the shared
    ACTION_DELTA_SCALER puts both on the same scale. Lost: all of the short
    find_category_outlier, most of insert_key, ~half of set_button_colors.
    Stride 1 (25 fps, 3.7 s clips, delta p95 0.10) keeps 2352; stride 3 / 4
    keep only 1741 / 1406 and drop 4-7 scenes entirely.

    ``min_temporal_stride`` (default 1): an episode too short for
    ``temporal_stride`` uses the largest stride in ``[min_temporal_stride,
    temporal_stride]`` that still fits (reported fps = 25 / stride). With the
    defaults this keeps 2352 episodes and all 24 scenes; the 211 short ones
    (find_category_outlier, insert_key, ...) train at 25 fps. ``None`` disables
    the fallback.

Multi-view (``MULTIVIEW_LAYOUT``): the three cameras are packed onto one
480x640 canvas, so a single-view checkpoint loads without a spatial-size change::

    cam_high                         (full width, top 2/3)
    cam_left_wrist | cam_right_wrist (half width each, bottom 1/3)

Each view is bilinear-resized into its cell (``compose_multiview``). RGB and
action video use the same packing; the 28-D action is shared. Wrist skeletons
are projected with that frame's own wrist extrinsic.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional, Sequence, Union

import numpy as np
import torch
from decord import VideoReader, cpu

from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.predict2.datasets.local_datasets.action_unified_layout import (
    UNIFIED_ACTION_DIM,
    UNIFIED_ACTION_NAMES,
    split_unified_bimanual_action,
    to_unified_bimanual_action,
)
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_robotwin_action import (
    DEFAULT_TARGET_FPS,
    RobotwinActionDataset,
    _RobotwinActionSample,
    default_robotwin_action_delta_scaler,
)
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_utils import (
    is_excluded_video_path,
    parallel_path_exists,
    sample_video_frame_indices,
)

DEFAULT_PAIRS_JSONL = (
    "/media/datasets/Sim20_dataset/UniEmbodied-Bench/"
    "policy_mix24_clean2400_sdk_ee20_25hz_instruction_20261029_v2_action_video/"
    "uniembodied_action_video_rgb_pairs.jsonl"
)
DEFAULT_TEMPORAL_STRIDE = 2
DEFAULT_MIN_TEMPORAL_STRIDE = 1
SOURCE_ARM_DOF = 6
SOURCE_ACTION_DIM = SOURCE_ARM_DOF * 2 + 2  # 14
NPZ_ACTION_KEY = "image_joint"
MULTIVIEW_LAYOUT = "mv_fill"
# Canvas order: the head view first, then the left / right wrist cells.
MULTIVIEW_VIEWS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def multiview_cells(height: int, width: int) -> dict[str, tuple[int, int, int, int]]:
    """Canvas cell ``(y0, y1, x0, x1)`` of each view: head on the top 2/3, wrists split the bottom 1/3."""
    head_h, left_w = (height * 2) // 3, width // 2
    return {
        MULTIVIEW_VIEWS[0]: (0, head_h, 0, width),
        MULTIVIEW_VIEWS[1]: (head_h, height, 0, left_w),
        MULTIVIEW_VIEWS[2]: (head_h, height, left_w, width),
    }


def compose_multiview(views: Sequence[torch.Tensor], height: Optional[int] = None,
                      width: Optional[int] = None) -> torch.Tensor:
    """``[head, left, right]`` uint8 ``[T, C, H, W]`` → one ``[T, C, height, width]`` canvas.

    Each view is bilinear-resized into its ``multiview_cells`` rectangle. No padding.
    """
    import torch.nn.functional as F

    if len(views) != 3:
        raise ValueError(f"multi-view canvas needs head + two wrists, got {len(views)} views")
    if len({v.shape[0] for v in views}) != 1:
        raise ValueError(f"view lengths differ: {[v.shape[0] for v in views]}")
    height = int(height or views[0].shape[-2])
    width = int(width or views[0].shape[-1])
    canvas = torch.empty((views[0].shape[0], views[0].shape[1], height, width), dtype=torch.uint8)
    for frames, (y0, y1, x0, x1) in zip(views, multiview_cells(height, width).values()):
        if frames.shape[-2:] != (y1 - y0, x1 - x0):
            frames = F.interpolate(frames.float(), size=(y1 - y0, x1 - x0), mode="bilinear", align_corners=False)
            frames = frames.clamp(0, 255).to(torch.uint8)
        canvas[:, :, y0:y1, x0:x1] = frames
    return canvas


class RoboDojoActionDataset(RobotwinActionDataset):
    """RoboDojo RGB (+ optional action video) with unified 28-D action values.

    Same output keys / packing as ``RobotwinActionDataset``; only the index (actions
    from the per-episode npz instead of HDF5) and action loading differ.
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
        min_temporal_stride: Optional[int] = DEFAULT_MIN_TEMPORAL_STRIDE,
        source_arm_dof: int = SOURCE_ARM_DOF,
        npz_action_key: str = NPZ_ACTION_KEY,
        skip_empty_instruction: bool = True,
    ) -> None:
        pairs_jsonl = os.environ.get("ROBODOJO_PAIRS_JSONL", pairs_jsonl)
        min_stride = int(temporal_stride if min_temporal_stride is None else min_temporal_stride)
        if not 1 <= min_stride <= int(temporal_stride):
            raise ValueError(
                f"min_temporal_stride must be in [1, temporal_stride={temporal_stride}], got {min_temporal_stride}"
            )
        self.min_temporal_stride = min_stride
        self.npz_action_key = str(npz_action_key)
        self._start_frames: dict[str, int] = {}
        self._strides: dict[str, int] = {}
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
            skip_empty_instruction=skip_empty_instruction,
        )

    def __str__(self) -> str:
        return f"{len(self.samples)} RoboDojo action samples from {self.pairs_jsonl}"

    def _build_index(self) -> list[_RobotwinActionSample]:
        """Like the RoboTwin index, but ``hdf5_path`` holds the per-episode actions npz."""
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
                actions = str(item.get("actions") or "").strip()
                if not rgb or not actions:
                    skipped_missing += 1
                    continue
                if self.require_action_video and not action_video:
                    skipped_missing += 1
                    continue
                if is_excluded_video_path(rgb, self.exclude_video_paths):
                    skipped_missing += 1
                    continue

                length = int(item.get("frame_nums") or 0)
                start_frame = max(0, int(item.get("start_frame") or 0))
                stride = self._fit_stride(length - start_frame)
                if stride is None:
                    skipped_short += 1
                    continue

                caption = self._caption_from_item(item)
                if self.skip_empty_instruction and not caption:
                    skipped_empty_cap += 1
                    continue

                try:
                    src_fps = float(item["fps"]) if item.get("fps") is not None else None
                except (TypeError, ValueError):
                    src_fps = None
                if src_fps is not None and (not np.isfinite(src_fps) or src_fps <= 0):
                    src_fps = None

                samples.append(
                    _RobotwinActionSample(
                        video_path=rgb,
                        action_video_path=action_video,
                        hdf5_path=actions,
                        caption=caption,
                        length=length,
                        uuid=str(item.get("uuid") or ""),
                        task_name=str(item.get("task_name") or ""),
                        index_in_jsonl=line_idx,
                        src_fps=src_fps,
                        robot_config="official" if item.get("official") else "self_collected",
                    )
                )
                if start_frame:
                    self._start_frames[rgb] = start_frame
                if stride != self.temporal_stride:
                    self._strides[rgb] = stride

        if self.verify_files and samples:
            paths_to_check: list[str] = []
            for s in samples:
                paths_to_check.append(s.video_path)
                paths_to_check.append(s.hdf5_path)
                if self.require_action_video:
                    paths_to_check.append(s.action_video_path)
            exists = parallel_path_exists(paths_to_check)
            stride = 3 if self.require_action_video else 2
            before = len(samples)
            samples = [s for i, s in enumerate(samples) if all(exists[i * stride:(i + 1) * stride])]
            skipped_missing += before - len(samples)

        kept = {s.video_path for s in samples}
        self._start_frames = {k: v for k, v in self._start_frames.items() if k in kept}
        self._strides = {k: v for k, v in self._strides.items() if k in kept}
        stride_hist = {self.temporal_stride: len(samples) - len(self._strides)}
        for v in self._strides.values():
            stride_hist[v] = stride_hist.get(v, 0) + 1
        log.info(
            f"Indexed {len(samples)} RoboDojo action episodes "
            f"(episodes per stride={dict(sorted(stride_hist.items()))}, "
            f"with start_frame={len(self._start_frames)}, "
            f"skipped_missing={skipped_missing}, skipped_short={skipped_short}, "
            f"skipped_empty_caption={skipped_empty_cap}, "
            f"verify_files={self.verify_files}, require_action_video={self.require_action_video})"
        )
        return samples

    def _fit_stride(self, usable_frames: int) -> Optional[int]:
        """Largest stride in ``[min_temporal_stride, temporal_stride]`` whose window fits."""
        if self.uniform_sample_episode:
            return self.temporal_stride
        for stride in range(self.temporal_stride, self.min_temporal_stride - 1, -1):
            if usable_frames >= self.sequence_length * stride:
                return stride
        return None

    def _sample_frame_ids_with_stride(
        self, total_frames: int, src_fps: Optional[float], stride: int
    ) -> tuple[list[int], float]:
        raw_len = self.sequence_length * stride
        raw_ids, reported_fps = sample_video_frame_indices(
            total_frames,
            raw_len,
            uniform_sample_episode=self.uniform_sample_episode,
            src_fps=src_fps,
            target_fps=self.target_fps,
            start_from_first_frame_prob=self.start_from_first_frame_prob,
        )
        frame_ids = raw_ids[::stride]
        if len(raw_ids) != raw_len or len(frame_ids) != self.sequence_length:
            raise RuntimeError(
                f"stride={stride}: expected {raw_len} raw / {self.sequence_length} kept ids, "
                f"got {len(raw_ids)} / {len(frame_ids)}"
            )
        return frame_ids, float(reported_fps) / float(stride)

    def _load_rgb_with_ids(
        self, video_path: str, meta_fps: Optional[float]
    ) -> tuple[torch.Tensor, list[int], float]:
        """Same as RoboTwin, but windows start at or after the episode's ``start_frame``
        and use the per-episode stride from ``_fit_stride``.

        Returned ids are absolute, so action video / actions stay aligned.
        """
        start = self._start_frames.get(video_path, 0)
        stride = self._strides.get(video_path, self.temporal_stride)
        if not start and stride == self.temporal_stride:
            return super()._load_rgb_with_ids(video_path, meta_fps)
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file does not exist: {video_path}")
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        total_frames = len(vr)
        src_fps = self._read_fps(vr)
        if src_fps is None:
            src_fps = meta_fps
        rel_ids, reported_fps = self._sample_frame_ids_with_stride(total_frames - start, src_fps, stride)
        frame_ids = [int(i) + start for i in rel_ids]
        frame_data = vr.get_batch(frame_ids).asnumpy().astype(np.uint8)
        vr.seek(0)
        del vr

        frames = torch.from_numpy(frame_data).permute(0, 3, 1, 2)
        frames = self.preprocess(frames)
        frames = torch.clamp(frames * 255.0, 0, 255).to(torch.uint8)
        return frames, frame_ids, reported_fps

    def _load_actions(self, hdf5_path: str, frame_ids: Sequence[int]) -> torch.Tensor:
        if not os.path.isfile(hdf5_path):
            raise FileNotFoundError(f"Actions npz does not exist: {hdf5_path}")
        with np.load(hdf5_path, allow_pickle=False) as d:
            if self.npz_action_key not in d.files:
                raise KeyError(f"Missing {self.npz_action_key!r} in {hdf5_path}; keys={d.files}")
            actions = np.asarray(d[self.npz_action_key], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != SOURCE_ACTION_DIM:
            raise ValueError(
                f"Expected action matrix [T, {SOURCE_ACTION_DIM}], got {actions.shape} in {hdf5_path}"
            )

        n = actions.shape[0]
        safe_ids = [min(max(int(i), 0), n - 1) for i in frame_ids]
        unified = to_unified_bimanual_action(
            actions[safe_ids],
            num_arms=2,
            include_gripper=self.include_gripper,
            arm_dof=self.source_arm_dof,
        )
        if unified.shape[-1] != self.action_dim:
            raise RuntimeError(
                f"Unified action dim {unified.shape[-1]} != {self.action_dim} from {hdf5_path}"
            )
        return torch.from_numpy(np.ascontiguousarray(unified)).float()

    @classmethod
    def render_action_video(cls, request: Any) -> np.ndarray:
        """White-bg ARX-X5 skeleton ``[T, H, W, 3]`` for one RoboDojo episode.

        Joints: ``action.npy`` (unified 28-D or native Joint14) at the source frame rate,
        so the gripper state machine sees the same frame spacing as the training render.
        Geometry: ``K`` / ``T_world_camera_opencv`` / ``arm_bases`` from meta.json, else the
        raw HDF5 (``hdf5``), else the default ``dual_x5.yml`` bases.

        With ``layout == MULTIVIEW_LAYOUT`` the head and both wrist skeletons are rendered
        and packed by ``compose_multiview``. Each wrist entry in ``views`` gives its ``K``
        and a ``camera_track`` npz (key = view name, ``[T, 4, 4]`` OpenCV pose per RGB frame).
        """
        from cosmos_predict2._src.predict2.datasets.local_datasets.action_video_render import (
            BACKEND_ROOT,
            ensure_backend_import_path,
            load_render_meta,
        )

        ensure_backend_import_path(BACKEND_ROOT / "robodojo")
        import batch_render_action_video_robodojo as ue

        extra = {**load_render_meta(request), **dict(request.extra or {})}
        if not request.action_path or not os.path.isfile(request.action_path):
            raise FileNotFoundError(f"RoboDojo online AV needs action.npy, got {request.action_path!r}")
        joint14 = action_to_joint14(np.load(request.action_path))
        max_frames = int(extra.get("max_frames") or 0)
        if max_frames > 0:
            joint14 = joint14[:max_frames]
        ue.init_renderer(ue.DEFAULT_URDF, request.width, request.height, ue.DEFAULT_ROBOT_CFG)
        K, T_world_camera, bases = render_geometry_from_meta(extra, ue)
        head = ue.render_action_frames(joint14, bases, K, T_world_camera)
        if extra.get("layout") != MULTIVIEW_LAYOUT:
            return head
        root = os.path.dirname(request.metadata_path or request.action_path)
        frames = [head]
        for view in MULTIVIEW_VIEWS[1:]:
            entry = dict(extra["views"][view])
            track_path = os.path.join(root, str(entry["camera_track"]))
            with np.load(track_path, allow_pickle=False) as tracks:
                track = np.asarray(tracks[view], dtype=np.float64)
            if track.shape[0] < joint14.shape[0]:
                raise ValueError(f"{track_path}[{view}] has {track.shape[0]} poses for {joint14.shape[0]} frames")
            frames.append(ue.render_action_frames(joint14, bases, entry["K"], track[: joint14.shape[0]]))
        tchw = [torch.from_numpy(f).permute(0, 3, 1, 2) for f in frames]
        return compose_multiview(tchw).permute(0, 2, 3, 1).numpy()


def action_to_joint14(action: np.ndarray) -> np.ndarray:
    """Unified ``[T, 28]`` (or native ``[T, 14]``) → Joint14 = arm(6)+gripper01 per side."""
    arr = np.asarray(action, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] not in (SOURCE_ACTION_DIM, UNIFIED_ACTION_DIM):
        raise ValueError(f"expected [T, {SOURCE_ACTION_DIM}] or [T, {UNIFIED_ACTION_DIM}], got {arr.shape}")
    if arr.shape[1] == SOURCE_ACTION_DIM:
        return arr
    parts = split_unified_bimanual_action(arr)
    return np.concatenate(
        [
            parts["left_arm"][:, :SOURCE_ARM_DOF],
            parts["left_gripper"][:, None],
            parts["right_arm"][:, :SOURCE_ARM_DOF],
            parts["right_gripper"][:, None],
        ],
        axis=1,
    ).astype(np.float64)


def render_geometry_from_meta(meta: dict[str, Any], ue: Any) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """(K 3x3 at 640x480, T_world_camera OpenCV 4x4, {left, right} arm base 4x4)."""
    K = meta.get("K") if meta.get("K") is not None else meta.get("intrinsic")
    T_wc = meta.get("T_world_camera_opencv")
    if T_wc is None and meta.get("E_world2cam") is not None:
        e = np.eye(4)
        e[:3, :] = np.asarray(meta["E_world2cam"], dtype=np.float64)[:3, :]
        T_wc = np.linalg.inv(e)
    bases = meta.get("arm_bases")
    hdf5 = str(meta.get("hdf5") or "")
    if (K is None or T_wc is None or bases is None) and hdf5 and os.path.isfile(hdf5):
        geom = ue.load_episode_geometry(hdf5, "cam_head", 0, default_bases=ue.load_default_arm_bases())
        K = geom["intrinsic"] if K is None else K
        T_wc = geom["T_world_camera_opencv"] if T_wc is None else T_wc
        bases = geom["bases"] if bases is None else bases
    if K is None or T_wc is None:
        raise FileNotFoundError("RoboDojo online AV needs K + T_world_camera_opencv in meta.json or an hdf5")
    if bases is None:
        bases = ue.load_default_arm_bases()
    bases = {side: np.asarray(bases[side], dtype=np.float64) for side in ("left", "right")}
    return np.asarray(K, dtype=np.float64), np.asarray(T_wc, dtype=np.float64), bases


def default_robodojo_action_delta_scaler() -> list[float]:
    return default_robotwin_action_delta_scaler()


__all__ = [
    "DEFAULT_MIN_TEMPORAL_STRIDE",
    "DEFAULT_PAIRS_JSONL",
    "DEFAULT_TEMPORAL_STRIDE",
    "MULTIVIEW_LAYOUT",
    "MULTIVIEW_VIEWS",
    "NPZ_ACTION_KEY",
    "SOURCE_ACTION_DIM",
    "SOURCE_ARM_DOF",
    "UNIFIED_ACTION_DIM",
    "UNIFIED_ACTION_NAMES",
    "RoboDojoActionDataset",
    "action_to_joint14",
    "compose_multiview",
    "default_robodojo_action_delta_scaler",
    "multiview_cells",
    "render_geometry_from_meta",
]
