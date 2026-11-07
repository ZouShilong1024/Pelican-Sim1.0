# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LIBERO action-video dataset with unified bimanual action layout.

Directory layout under ``dataset_dir``::

    <suite>/                       # e.g. libero_10, libero_90, ...
      <task_name>/
        demo_<id>/
          rgb.mp4
          action_video.mp4
          action.npy               # typically [T, 8] = 7 joints + gripper
          metadata.json

If ``action_video.mp4`` is missing, inference calls
``LiberoActionDataset.render_action_video`` (Panda URDF + camera ``K/E``,
then MuJoCo replay when HDF5+BDDL are available, otherwise analytic FK).

Raw LIBERO actions in this export are single-arm 8-D. They are packed into the
canonical unified 28-D layout::

    left_arm(7) + left_gripper(1) + left_dexhand(6)
    + right_arm(7) + right_gripper(1) + right_dexhand(6)

with the right side and all dexhand slots zero-filled. Shorter arms pad the
trailing joint dims with 0.
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
    SIM_PANDA_GRIPPER_RAW_MAX,
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

DEFAULT_DATASET_DIR = "/media/datasets/zhangshilin/libero_action_video"
DEFAULT_TARGET_FPS = 16.0
DEFAULT_SUITES: tuple[str, ...] = (
    "libero_10",
    "libero_90",
    "libero_goal",
    "libero_object",
    "libero_spatial",
)


@dataclass(frozen=True)
class _LiberoEpisodeSample:
    video_path: str
    action_video_path: str
    action_path: str
    caption: str
    length: int
    suite: str
    task_name: str
    traj_id: str
    num_arms: int
    raw_action_dim: int
    src_fps: Optional[float]


class LiberoActionDataset(Dataset):
    """LIBERO RGB (+ optional action video) with unified 28-D action values.

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
        dataset_dir: str = DEFAULT_DATASET_DIR,
        num_frames: int = 93,
        video_size: tuple[int, int] = (480, 640),
        suites: Optional[Sequence[str]] = None,
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
        # Source arm joint count excluding gripper. None → infer (LIBERO: 7).
        source_arm_dof: Optional[int] = None,
    ) -> None:
        Dataset.__init__(self)
        self.dataset_dir = dataset_dir
        self.sequence_length = int(num_frames)
        self.video_size = tuple(video_size)
        self.suites = tuple(suites) if suites is not None else DEFAULT_SUITES
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

        self.action_dim = UNIFIED_ACTION_DIM
        self.action_names = UNIFIED_ACTION_NAMES
        if action_delta_scaler is None:
            # Match RSW / OminiEWM defaults for the unified 28-D layout.
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
            raise FileNotFoundError(f"LIBERO dataset dir not found: {self.dataset_dir}")

        self.samples = self._build_index()
        if not self.samples:
            raise RuntimeError(
                f"No valid LIBERO action episodes under {self.dataset_dir} "
                f"(suites={list(self.suites)}, num_frames={self.sequence_length}, "
                f"load_action_video={self.load_action_video})."
            )

        scaler_msg = (
            "None"
            if self.action_delta_scaler is None
            else [round(float(x), 4) for x in self.action_delta_scaler.tolist()]
        )
        log.info(
            f"LiberoActionDataset: {len(self.samples)} episodes from {self.dataset_dir} "
            f"(suites={list(self.suites)}, num_frames={self.sequence_length}, "
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
        return f"{len(self.samples)} LIBERO action samples from {self.dataset_dir}"

    def __len__(self) -> int:
        return len(self.samples)

    def _iter_demo_dirs(self):
        for suite in self.suites:
            suite_dir = os.path.join(self.dataset_dir, suite)
            if not os.path.isdir(suite_dir):
                log.warning(f"LIBERO suite dir missing, skip: {suite_dir}")
                continue
            for task_name in sorted(os.listdir(suite_dir)):
                task_dir = os.path.join(suite_dir, task_name)
                if not os.path.isdir(task_dir):
                    continue
                for traj_id in sorted(os.listdir(task_dir)):
                    demo_dir = os.path.join(task_dir, traj_id)
                    if not os.path.isdir(demo_dir) or not traj_id.startswith("demo_"):
                        continue
                    yield suite, task_name, traj_id, demo_dir

    def _build_index(self) -> list[_LiberoEpisodeSample]:
        samples: list[_LiberoEpisodeSample] = []
        skipped_missing = 0
        skipped_short = 0

        for suite, task_name, traj_id, demo_dir in self._iter_demo_dirs():
            meta_path = os.path.join(demo_dir, "metadata.json")
            rgb_path = os.path.join(demo_dir, "rgb.mp4")
            action_path = os.path.join(demo_dir, "action.npy")
            action_video_path = os.path.join(demo_dir, "action_video.mp4")

            meta: dict[str, Any] = {}
            if os.path.isfile(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)

            # Prefer paths from metadata when present; fall back to convention.
            if isinstance(meta.get("rgb"), str) and meta["rgb"]:
                rgb_path = meta["rgb"] if os.path.isabs(meta["rgb"]) else os.path.join(demo_dir, meta["rgb"])
            if isinstance(meta.get("action"), str) and meta["action"]:
                action_path = (
                    meta["action"]
                    if os.path.isabs(meta["action"])
                    else os.path.join(demo_dir, meta["action"])
                )
            if isinstance(meta.get("action_video"), str) and meta["action_video"]:
                action_video_path = (
                    meta["action_video"]
                    if os.path.isabs(meta["action_video"])
                    else os.path.join(demo_dir, meta["action_video"])
                )

            if is_excluded_video_path(rgb_path, self.exclude_video_paths):
                skipped_missing += 1
                continue

            length = int(meta.get("frames") or 0)
            if length <= 0:
                # Cheap length probe without decoding full video when meta lacks frames.
                try:
                    length = int(np.load(action_path, mmap_mode="r").shape[0])
                except Exception:
                    skipped_missing += 1
                    continue

            if is_too_short_for_window(
                length, self.sequence_length, self.uniform_sample_episode
            ):
                skipped_short += 1
                continue

            caption = str(meta.get("lang") or "").strip()
            if not caption:
                caption = task_name.replace("_", " ")

            num_arms = int(meta.get("num_arms") or 1)
            raw_action_dim = int(meta.get("action_dim") or 0)
            src_fps = meta.get("fps")
            src_fps_f = float(src_fps) if src_fps is not None else None
            if src_fps_f is not None and (not np.isfinite(src_fps_f) or src_fps_f <= 0):
                src_fps_f = None

            samples.append(
                _LiberoEpisodeSample(
                    video_path=rgb_path,
                    action_video_path=action_video_path if action_video_path else "",
                    action_path=action_path,
                    caption=caption,
                    length=length,
                    suite=str(meta.get("suite") or suite),
                    task_name=str(meta.get("task") or task_name),
                    traj_id=str(meta.get("traj_id") or traj_id),
                    num_arms=num_arms,
                    raw_action_dim=raw_action_dim,
                    src_fps=src_fps_f,
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
            kept: list[_LiberoEpisodeSample] = []
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
            f"Indexed {len(samples)} LIBERO action episodes "
            f"(skipped_missing={skipped_missing}, skipped_short={skipped_short}, "
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
            raise ValueError(f"Expected action matrix [T, D], got shape {actions.shape} in {action_path}")
        if raw_action_dim and actions.shape[1] != raw_action_dim:
            # Prefer actual array width; metadata mismatch is non-fatal.
            log.warning(
                f"action.npy dim {actions.shape[1]} != metadata action_dim={raw_action_dim} "
                f"in {action_path}; using array width."
            )

        n = actions.shape[0]
        safe_ids = [min(max(int(i), 0), n - 1) for i in frame_ids]
        selected = actions[safe_ids]  # [T, D_raw]
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
                f"Failed to load LIBERO action sample {sample.video_path} "
                f"(total failures: {self.num_failed_loads}): {e}\n{traceback.format_exc()}",
                rank0_only=False,
            )
            return self[int(np.random.randint(len(self.samples)))]

    @classmethod
    def render_action_video(cls, request: Any) -> np.ndarray:
        """White-bg skeleton ``[T, H, W, 3]`` for one LIBERO episode.

        Prefer Panda URDF + camera ``K/E`` / ``base_pos`` (same as
        ``libero_urdf_ac_6ep_20260911``). Fall back to MuJoCo replay when
        HDF5+BDDL are present, then to analytic Panda FK.
        """
        import os

        from cosmos_predict2._src.predict2.datasets.local_datasets.action_video_render import (
            BACKEND_ROOT,
            ensure_backend_import_path,
            load_render_meta,
            render_panda_fallback,
        )

        extra = dict(request.extra or {})
        meta = load_render_meta(request)
        if request.camera_preset and not meta.get("camera_preset"):
            meta["camera_preset"] = request.camera_preset
        for key in ("base_pos", "camera_eye", "camera_target", "K", "suite"):
            if extra.get(key) not in (None, "", []) and meta.get(key) in (None, "", []):
                meta[key] = extra[key]

        urdf_frames = cls._render_action_video_urdf(request, meta)
        if urdf_frames is not None:
            return urdf_frames

        h5 = str(extra.get("h5_path") or meta.get("h5_path") or request.hdf5_path or "")
        bddl = str(extra.get("bddl_path") or meta.get("bddl_path") or "")
        demo_id = str(extra.get("demo_id") or extra.get("traj_id") or meta.get("traj_id") or "")
        if h5 and os.path.isfile(h5) and bddl:
            try:
                import h5py  # noqa: F401
            except ImportError:
                h5 = ""
        if h5 and bddl:
            ensure_backend_import_path(BACKEND_ROOT / "libero")
            try:
                import render_action_video_libero as lb
            except Exception:
                lb = None
            if lb is not None:
                if not demo_id:
                    demo_id = "demo_0"
                try:
                    env = lb.make_env(bddl, request.width, request.height)
                    import h5py

                    with h5py.File(h5, "r") as f:
                        if f"data/{demo_id}" not in f:
                            keys = list(f["data"].keys())
                            if demo_id not in keys:
                                demo_id = keys[0] if keys else demo_id
                        res = lb.render_episode(
                            env,
                            f,
                            demo_id,
                            request.width,
                            request.height,
                            save_rgb=False,
                            save_av=True,
                            save_ov=False,
                            camera_preset=str(
                                request.camera_preset or meta.get("camera_preset") or "default"
                            ),
                            baseline_target_dz=float(meta.get("baseline_target_dz") or 0.0),
                        )
                    frames = res.get("skel_frames") or []
                    if frames:
                        return np.stack(frames, axis=0).astype(np.uint8)
                except Exception:
                    pass
        return render_panda_fallback(request, meta, "libero")

    @classmethod
    def _render_action_video_urdf(cls, request: Any, meta: dict[str, Any]) -> Optional[np.ndarray]:
        """URDF FK + robosuite cam2world. Returns None if inputs are incomplete."""
        import os

        action_path = str(request.action_path or "")
        if not action_path or not os.path.isfile(action_path):
            return None
        if meta.get("K") is None:
            return None
        try:
            from cosmos_predict2._src.predict2.datasets.local_datasets.action_video_render import (
                load_backend_module,
            )

            urdf_av = load_backend_module(
                "libero",
                "render_action_video_from_urdf.py",
                "libero_render_action_video_from_urdf",
            )

            action = np.asarray(np.load(action_path), dtype=np.float64)
            if action.ndim != 2:
                return None
            if action.shape[1] > 8:
                # Unified 28-D: left arm (7) + left gripper (1).
                action = action[:, :8]
            if action.shape[1] < 8:
                return None
            extra = dict(request.extra or {})
            base = extra.get("base_pos") or meta.get("base_pos")
            qpos = None
            qpos_path = str(extra.get("kinematic_qpos_path") or meta.get("kinematic_qpos_path") or "")
            if not qpos_path:
                qpos_path = os.path.join(os.path.dirname(action_path), "kinematic_qpos.npy")
            if qpos_path and os.path.isfile(qpos_path):
                qpos = np.load(qpos_path)
            frames = urdf_av.render_action_video_thwc(
                action,
                meta,
                height=request.height,
                width=request.width,
                base_pos=base,
                qpos=qpos,
            )
            return np.asarray(frames, dtype=np.uint8)
        except Exception as err:
            log.warning(f"LIBERO URDF action-video render failed: {err}")
            return None


# Convenience defaults matching action-inject / RSW.
def default_libero_action_delta_scaler() -> list[float]:
    return build_action_delta_scaler(
        include_gripper=True,
        include_lift=False,
        arm_scaler=DEFAULT_ARM_DELTA_SCALER,
        gripper_scaler=DEFAULT_GRIPPER_DELTA_SCALER,
    )


__all__ = [
    "DEFAULT_DATASET_DIR",
    "DEFAULT_SUITES",
    "LiberoActionDataset",
    "UNIFIED_ACTION_DIM",
    "UNIFIED_ACTION_NAMES",
    "UNIFIED_ARM_DOF",
    "default_libero_action_delta_scaler",
    "to_unified_bimanual_action",
]
