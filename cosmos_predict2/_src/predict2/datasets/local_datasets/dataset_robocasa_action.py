# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RoboCasa dual-view action-video dataset with unified bimanual action layout.

Supports one or more render roots that share the same episode layout
(human dualview + MimicGen mg exports)::

    <root>/<TaskName>/demo_<id>/
      rgb_left.mp4
      rgb_right.mp4
      action_video_left.mp4
      action_video_right.mp4
      action.npy                 # typically [T, 8] = 7 joints + gripper
      metadata.json

Default roots::

    /media/datasets/robocasa_action_video_dualview_v2
    /media/datasets/robocasa_action_video_mg_v2

Both left and right views are indexed as separate samples, each paired with
the matching ``action_video_<view>``. ``action.npy`` is shared across views.

If ``action_video_<view>.mp4`` is missing, inference calls
``RobocasaActionDataset.render_action_video`` (Panda FK fallback).

Raw RoboCasa actions are single-arm 8-D and packed into the canonical unified
28-D layout::

    left_arm(7) + left_gripper(1) + left_dexhand(6)
    + right_arm(7) + right_gripper(1) + right_dexhand(6)

with the right side and all dexhand slots zero-filled.

Temporal sampling (``temporal_stride``, default 2)::

    RoboCasa renders at ~20 fps with median episode length ~259 frames.
    Stride 2 → model rate ~10 fps; keep demos with
    ``frames >= num_frames * 2`` (~84% at ``num_frames=93``).
"""

from __future__ import annotations

import json
import os
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union

import numpy as np
import torch
from decord import VideoReader, cpu
from torch.utils.data import Dataset
from torchvision import transforms as T

from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.predict2.datasets.local_datasets.action_unified_layout import (
    SIM_PANDA_GRIPPER_RAW_MAX,
    UNIFIED_ACTION_DIM,
    UNIFIED_ACTION_NAMES,
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

DEFAULT_DATASET_DIRS: tuple[str, ...] = (
    "/media/datasets/robocasa_action_video_dualview_v2",
    "/media/datasets/robocasa_action_video_mg_v2",
)
DEFAULT_TARGET_FPS = 16.0
DEFAULT_TEMPORAL_STRIDE = 2
DEFAULT_VIEWS: tuple[str, ...] = ("left", "right")


@dataclass(frozen=True)
class _RobocasaEpisodeSample:
    video_path: str
    action_video_path: str
    action_path: str
    caption: str
    length: int
    source_root: str
    task_name: str
    traj_id: str
    view: str
    num_arms: int
    raw_action_dim: int
    src_fps: Optional[float]


def _normalize_dataset_dirs(
    dataset_dir: Optional[Union[str, Sequence[str]]],
    dataset_dirs: Optional[Sequence[str]],
) -> tuple[str, ...]:
    """Accept ``dataset_dir`` (str/list) and/or ``dataset_dirs`` → unique roots."""
    roots: list[str] = []
    if dataset_dirs is not None:
        roots.extend(str(p) for p in dataset_dirs)
    if dataset_dir is not None:
        if isinstance(dataset_dir, (str, os.PathLike)):
            roots.append(str(dataset_dir))
        else:
            roots.extend(str(p) for p in dataset_dir)
    if not roots:
        roots = list(DEFAULT_DATASET_DIRS)
    # Preserve order, drop empties / duplicates.
    seen: set[str] = set()
    out: list[str] = []
    for r in roots:
        r = os.path.abspath(os.path.expanduser(r))
        if not r or r in seen:
            continue
        seen.add(r)
        out.append(r)
    return tuple(out)


class RobocasaActionDataset(Dataset):
    """RoboCasa RGB L/R (+ action video) with unified 28-D action values.

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
        dataset_dir: Optional[Union[str, Sequence[str]]] = None,
        num_frames: int = 93,
        video_size: tuple[int, int] = (480, 640),
        dataset_dirs: Optional[Sequence[str]] = None,
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
        # Source arm joint count excluding gripper. None → infer (RoboCasa: 7).
        source_arm_dof: Optional[int] = None,
    ) -> None:
        Dataset.__init__(self)
        self.dataset_dirs = _normalize_dataset_dirs(dataset_dir, dataset_dirs)
        self.sequence_length = int(num_frames)
        self.video_size = tuple(video_size)
        self.views = tuple(views) if views is not None else DEFAULT_VIEWS
        for v in self.views:
            if v not in ("left", "right", "center"):
                raise ValueError(f"Unsupported RoboCasa view {v!r}; expected left/right/center")
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
        self.source_arm_dof = None if source_arm_dof is None else int(source_arm_dof)

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

        missing_roots = [r for r in self.dataset_dirs if not os.path.isdir(r)]
        if missing_roots:
            raise FileNotFoundError(
                f"RoboCasa dataset dir(s) not found: {missing_roots}"
            )

        self.samples = self._build_index()
        if not self.samples:
            raise RuntimeError(
                f"No valid RoboCasa action episodes under {list(self.dataset_dirs)} "
                f"(views={list(self.views)}, num_frames={self.sequence_length}, "
                f"temporal_stride={self.temporal_stride}, "
                f"raw_sequence_length={self.raw_sequence_length}, "
                f"load_action_video={self.load_action_video})."
            )

        # Helpful for scripts that expect ``video_paths`` on the root dataset.
        self.video_paths = [s.video_path for s in self.samples]

        scaler_msg = (
            "None"
            if self.action_delta_scaler is None
            else [round(float(x), 4) for x in self.action_delta_scaler.tolist()]
        )
        log.info(
            f"RobocasaActionDataset: {len(self.samples)} view-samples from "
            f"{list(self.dataset_dirs)} "
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
        return (
            f"{len(self.samples)} RoboCasa action samples from "
            f"{list(self.dataset_dirs)}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _iter_demo_dirs(self):
        """Yield ``(root, task_name, traj_id, demo_dir)`` via ``os.scandir``."""
        for root in self.dataset_dirs:
            with os.scandir(root) as task_it:
                tasks = sorted(
                    (e for e in task_it if e.is_dir(follow_symlinks=False) and not e.name.startswith("_")),
                    key=lambda e: e.name,
                )
            for task_ent in tasks:
                with os.scandir(task_ent.path) as demo_it:
                    demos = sorted(
                        (
                            e
                            for e in demo_it
                            if e.is_dir(follow_symlinks=False) and e.name.startswith("demo_")
                        ),
                        key=lambda e: e.name,
                    )
                for demo_ent in demos:
                    yield root, task_ent.name, demo_ent.name, demo_ent.path

    @staticmethod
    def _npy_num_rows(action_path: str) -> int:
        """Read only the ``.npy`` header for the leading axis (T)."""
        try:
            with open(action_path, "rb") as f:
                version = np.lib.format.read_magic(f)
                if version == (1, 0):
                    shape, _, _ = np.lib.format.read_array_header_1_0(f)
                elif version == (2, 0):
                    shape, _, _ = np.lib.format.read_array_header_2_0(f)
                else:
                    # Fallback for uncommon versions.
                    return int(np.load(action_path, mmap_mode="r").shape[0])
            if not shape:
                return 0
            return int(shape[0])
        except Exception:
            return 0

    def _build_index(self) -> list[_RobocasaEpisodeSample]:
        """Index demos without reading every ``metadata.json`` (too slow on mg).

        Canonical on-disk names are fixed by the RoboCasa render script::

            rgb_{view}.mp4 / action_video_{view}.mp4 / action.npy

        Caption defaults to the task folder name; fps defaults to 20 (render
        default). Episode length is taken from the ``action.npy`` header only
        (threaded) so large MimicGen roots stay indexable.
        """
        demos = list(self._iter_demo_dirs())
        action_paths = [os.path.join(d, "action.npy") for _, _, _, d in demos]
        # Parallel header reads dominate cost on large NFS trees (~70k demos).
        with ThreadPoolExecutor(max_workers=64) as pool:
            lengths = list(pool.map(self._npy_num_rows, action_paths))

        samples: list[_RobocasaEpisodeSample] = []
        skipped_missing = 0
        skipped_short = 0
        default_caption_cache: dict[str, str] = {}

        for (root, task_name, traj_id, demo_dir), action_path, length in zip(
            demos, action_paths, lengths
        ):
            if length <= 0:
                skipped_missing += 1
                continue
            if is_too_short_for_window(
                length, self.raw_sequence_length, self.uniform_sample_episode
            ):
                skipped_short += 1
                continue

            caption = default_caption_cache.get(task_name)
            if caption is None:
                caption = task_name.replace("_", " ")
                default_caption_cache[task_name] = caption

            for view in self.views:
                rgb_path = os.path.join(demo_dir, f"rgb_{view}.mp4")
                action_video_path = os.path.join(demo_dir, f"action_video_{view}.mp4")
                if is_excluded_video_path(rgb_path, self.exclude_video_paths):
                    skipped_missing += 1
                    continue
                samples.append(
                    _RobocasaEpisodeSample(
                        video_path=rgb_path,
                        action_video_path=action_video_path,
                        action_path=action_path,
                        caption=caption,
                        length=length,
                        source_root=root,
                        task_name=task_name,
                        traj_id=traj_id,
                        view=view,
                        num_arms=1,
                        raw_action_dim=8,
                        src_fps=20.0,  # RoboCasa render default
                    )
                )

        if self.verify_files and samples:
            paths_to_check: list[str] = []
            for s in samples:
                paths_to_check.append(s.video_path)
                # action.npy already probed above; still re-check in batch for
                # parity with other action datasets / NFS consistency.
                paths_to_check.append(s.action_path)
                if self.require_action_video and s.action_video_path:
                    paths_to_check.append(s.action_video_path)

            exists = parallel_path_exists(paths_to_check)
            kept: list[_RobocasaEpisodeSample] = []
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
            f"Indexed {len(samples)} RoboCasa action view-samples "
            f"(skipped_missing={skipped_missing}, skipped_short={skipped_short}, "
            f"verify_files={self.verify_files}, require_action_video={self.require_action_video}, "
            f"temporal_stride={self.temporal_stride}, "
            f"raw_sequence_length={self.raw_sequence_length}, "
            f"roots={list(self.dataset_dirs)}, views={list(self.views)})"
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
        self,
        action_path: str,
        frame_ids: Sequence[int],
        *,
        num_arms: int,
        raw_action_dim: int,
    ) -> torch.Tensor:
        if not os.path.isfile(action_path):
            raise FileNotFoundError(f"Action file does not exist: {action_path}")
        actions = np.load(action_path)
        actions = np.asarray(actions, dtype=np.float32)
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
            gripper_raw_max=SIM_PANDA_GRIPPER_RAW_MAX,
        )
        if unified.shape[-1] != self.action_dim:
            raise RuntimeError(
                f"Unified action dim {unified.shape[-1]} != {self.action_dim} from {action_path}"
            )
        return torch.from_numpy(np.ascontiguousarray(unified)).float()

    def _caption_for_sample(self, sample: _RobocasaEpisodeSample) -> str:
        """Prefer ``metadata.json`` lang; fall back to task-folder caption."""
        meta_path = os.path.join(os.path.dirname(sample.action_path), "metadata.json")
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                lang = str(meta.get("lang") or "").strip()
                if lang:
                    return lang
            except Exception:
                pass
        return sample.caption

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        try:
            video_thw, frame_ids, fps = self._load_rgb_with_ids(
                sample.video_path, sample.src_fps
            )
            video = video_thw.permute(1, 0, 2, 3)  # [C, T, H, W]
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
                "ai_caption": self._caption_for_sample(sample),
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
                f"Failed to load RoboCasa action sample {sample.video_path} "
                f"(view={sample.view}, total failures: {self.num_failed_loads}): {e}\n"
                f"{traceback.format_exc()}",
                rank0_only=False,
            )
            return self[int(np.random.randint(len(self.samples)))]

    @classmethod
    def render_action_video(cls, request: Any) -> np.ndarray:
        """White-bg skeleton ``[T, H, W, 3]`` for one RoboCasa episode.

        Prefer Panda URDF + ``base_pos`` / ``base_quat_wxyz`` + per-view
        ``K`` / ``E_cam2world`` (same as ``robocasa_urdf_ac_6ep_20260911``).
        Fall back to analytic Panda FK.
        """
        from cosmos_predict2._src.predict2.datasets.local_datasets.action_video_render import (
            load_render_meta,
            render_panda_fallback,
        )

        extra = dict(request.extra or {})
        meta = load_render_meta(request)
        for key in ("base_pos", "base_quat_wxyz", "K", "E_cam2world", "E", "views", "view"):
            if extra.get(key) not in (None, "", []) and meta.get(key) in (None, "", []):
                meta[key] = extra[key]
        urdf_frames = cls._render_action_video_urdf(request, meta)
        if urdf_frames is not None:
            return urdf_frames
        return render_panda_fallback(request, meta, "robocasa")

    @classmethod
    def _render_action_video_urdf(cls, request: Any, meta: dict[str, Any]) -> Optional[np.ndarray]:
        """URDF FK + robosuite cam2world. Returns None if inputs are incomplete."""
        import os

        action_path = str(request.action_path or "")
        if not action_path or not os.path.isfile(action_path):
            return None
        if meta.get("base_pos") in (None, "", []):
            return None
        if meta.get("K") is None and not meta.get("views"):
            return None
        try:
            from cosmos_predict2._src.predict2.datasets.local_datasets.action_video_render import (
                load_backend_module,
            )

            urdf_av = load_backend_module(
                "robocasa",
                "render_action_video_from_urdf.py",
                "robocasa_render_action_video_from_urdf",
            )

            action = np.asarray(np.load(action_path), dtype=np.float64)
            if action.ndim != 2:
                return None
            if action.shape[1] > 8:
                action = action[:, :8]
            if action.shape[1] < 8:
                return None
            extra = dict(request.extra or {})
            frames = urdf_av.render_action_video_thwc(
                action,
                meta,
                height=request.height,
                width=request.width,
                base_pos=extra.get("base_pos") or meta.get("base_pos"),
                base_quat=extra.get("base_quat_wxyz") or meta.get("base_quat_wxyz"),
                view=str(extra.get("view") or meta.get("view") or "") or None,
            )
            return np.asarray(frames, dtype=np.uint8)
        except Exception as err:
            log.warning(f"RoboCasa URDF action-video render failed: {err}")
            return None


def default_robocasa_action_delta_scaler() -> list[float]:
    return build_action_delta_scaler(
        include_gripper=True,
        include_lift=False,
        arm_scaler=DEFAULT_ARM_DELTA_SCALER,
        gripper_scaler=DEFAULT_GRIPPER_DELTA_SCALER,
    )


__all__ = [
    "DEFAULT_DATASET_DIRS",
    "DEFAULT_TARGET_FPS",
    "DEFAULT_TEMPORAL_STRIDE",
    "DEFAULT_VIEWS",
    "RobocasaActionDataset",
    "default_robocasa_action_delta_scaler",
]
