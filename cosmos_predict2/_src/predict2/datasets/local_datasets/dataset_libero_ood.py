# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LIBERO view-generalization (OOD) action-video dataset.

Reads the nested export under ``libero_view_gen_val``::

    <difficulty>/                 # easy | medium | hard
      <camera_preset>/            # e.g. easy_left, hard_right_down
        <suite>/                  # libero_10, libero_90, ...
          <task_name>/
            demo_<id>/
              rgb.mp4
              action_video.mp4
              action.npy          # typically [T, 8] = 7 joints + gripper
              metadata.json

When ``index.jsonl`` is present at the dataset root (preferred), episodes are
indexed from that file (absolute paths already filled in). Otherwise the tree
above is walked.

Action packing / temporal sampling match ``LiberoActionDataset``:

  - unified 28-D layout via ``to_unified_bimanual_action``
  - temporal_stride = 1 (same as training Libero; fps resampling only)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union

import numpy as np

from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_libero_action import (
    DEFAULT_SUITES,
    DEFAULT_TARGET_FPS,
    LiberoActionDataset,
    _LiberoEpisodeSample,
)
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_utils import (
    is_excluded_video_path,
    is_too_short_for_window,
    parallel_path_exists,
)

DEFAULT_DATASET_DIR = "/media/datasets/zhangshilin/libero_view_gen_val"
DEFAULT_INDEX_JSONL = "index.jsonl"
DEFAULT_VIEW_DIFFICULTIES: tuple[str, ...] = ("easy", "medium", "hard")
# Same as LiberoActionDataset / action_inject training: no frame skip.
DEFAULT_TEMPORAL_STRIDE = 1


@dataclass(frozen=True)
class _LiberoOODEpisodeSample(_LiberoEpisodeSample):
    view_difficulty: str = ""
    camera_preset: str = ""


class LiberoOODDataset(LiberoActionDataset):
    """LIBERO OOD / view-gen RGB (+ action video) with unified 28-D actions.

    Reuses ``LiberoActionDataset`` decode / unify / ``__getitem__`` paths; only
    the indexer differs (nested difficulty/preset layout + optional index.jsonl).
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
        source_arm_dof: Optional[int] = None,
        view_difficulties: Optional[Sequence[str]] = None,
        camera_presets: Optional[Sequence[str]] = None,
        index_jsonl: Optional[str] = None,
        # Kept for API symmetry with other action datasets; Libero OOD is stride=1.
        temporal_stride: int = DEFAULT_TEMPORAL_STRIDE,
    ) -> None:
        self.view_difficulties = (
            tuple(view_difficulties)
            if view_difficulties is not None
            else DEFAULT_VIEW_DIFFICULTIES
        )
        self.camera_presets = (
            tuple(str(x) for x in camera_presets) if camera_presets is not None else None
        )
        self.index_jsonl = (
            str(index_jsonl)
            if index_jsonl is not None
            else os.path.join(dataset_dir, DEFAULT_INDEX_JSONL)
        )
        stride = int(temporal_stride)
        if stride != DEFAULT_TEMPORAL_STRIDE:
            log.warning(
                f"LiberoOODDataset: temporal_stride={stride} ignored; "
                f"Libero OOD matches Libero training with stride={DEFAULT_TEMPORAL_STRIDE}"
            )
        self.temporal_stride = DEFAULT_TEMPORAL_STRIDE

        # Parent builds index via overridden ``_build_index``.
        super().__init__(
            dataset_dir=dataset_dir,
            num_frames=num_frames,
            video_size=video_size,
            suites=suites if suites is not None else DEFAULT_SUITES,
            verify_files=verify_files,
            target_fps=target_fps,
            uniform_sample_episode=uniform_sample_episode,
            start_from_first_frame_prob=start_from_first_frame_prob,
            exclude_video_paths=exclude_video_paths,
            load_action_video=load_action_video,
            include_gripper=include_gripper,
            require_action_video=require_action_video,
            action_delta_scaler=action_delta_scaler,
            source_arm_dof=source_arm_dof,
        )

    def __str__(self) -> str:
        return (
            f"{len(self.samples)} LIBERO OOD action samples from {self.dataset_dir} "
            f"(difficulties={list(self.view_difficulties)})"
        )

    def _resolve_path(self, raw: str, demo_dir: str) -> str:
        raw = str(raw or "").strip()
        if not raw:
            return ""
        return raw if os.path.isabs(raw) else os.path.join(demo_dir, raw)

    def _sample_from_meta(
        self,
        *,
        meta: dict[str, Any],
        demo_dir: str,
        suite: str,
        task_name: str,
        traj_id: str,
        view_difficulty: str,
        camera_preset: str,
    ) -> Optional[_LiberoOODEpisodeSample]:
        rgb_path = self._resolve_path(str(meta.get("rgb") or "rgb.mp4"), demo_dir)
        action_path = self._resolve_path(str(meta.get("action") or "action.npy"), demo_dir)
        action_video_path = self._resolve_path(
            str(meta.get("action_video") or "action_video.mp4"), demo_dir
        )

        if is_excluded_video_path(rgb_path, self.exclude_video_paths):
            return None

        length = int(meta.get("frames") or 0)
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

        num_arms = int(meta.get("num_arms") or 1)
        raw_action_dim = int(meta.get("action_dim") or 0)
        src_fps = meta.get("fps")
        src_fps_f = float(src_fps) if src_fps is not None else None
        if src_fps_f is not None and (not np.isfinite(src_fps_f) or src_fps_f <= 0):
            src_fps_f = None

        vd = str(meta.get("view_difficulty") or view_difficulty or "").strip()
        cp = str(meta.get("camera_preset") or camera_preset or "").strip()

        return _LiberoOODEpisodeSample(
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
            view_difficulty=vd,
            camera_preset=cp,
        )

    def _iter_from_index_jsonl(self):
        """Yield (meta dict, demo_dir, suite, task, traj, difficulty, preset)."""
        if not os.path.isfile(self.index_jsonl):
            return
        suite_set = set(self.suites)
        difficulty_set = set(self.view_difficulties)
        preset_set = set(self.camera_presets) if self.camera_presets is not None else None

        with open(self.index_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                meta = json.loads(line)
                suite = str(meta.get("suite") or "")
                if suite and suite not in suite_set:
                    continue
                difficulty = str(meta.get("view_difficulty") or "")
                if difficulty and difficulty not in difficulty_set:
                    continue
                preset = str(meta.get("camera_preset") or "")
                if preset_set is not None and preset and preset not in preset_set:
                    continue

                rgb = str(meta.get("rgb") or "")
                demo_dir = os.path.dirname(rgb) if rgb else ""
                if not demo_dir:
                    action = str(meta.get("action") or "")
                    demo_dir = os.path.dirname(action) if action else ""
                task_name = str(meta.get("task") or "")
                traj_id = str(meta.get("traj_id") or os.path.basename(demo_dir))
                yield meta, demo_dir, suite, task_name, traj_id, difficulty, preset

    def _iter_demo_dirs(self):
        """Walk ``difficulty/preset/suite/task/demo_*`` (fallback when no index)."""
        suite_set = set(self.suites)
        preset_set = set(self.camera_presets) if self.camera_presets is not None else None

        for difficulty in self.view_difficulties:
            diff_dir = os.path.join(self.dataset_dir, difficulty)
            if not os.path.isdir(diff_dir):
                log.warning(f"LIBERO OOD difficulty dir missing, skip: {diff_dir}")
                continue
            for preset in sorted(os.listdir(diff_dir)):
                preset_dir = os.path.join(diff_dir, preset)
                if not os.path.isdir(preset_dir):
                    continue
                if preset_set is not None and preset not in preset_set:
                    continue
                for suite in sorted(os.listdir(preset_dir)):
                    if suite not in suite_set:
                        continue
                    suite_dir = os.path.join(preset_dir, suite)
                    if not os.path.isdir(suite_dir):
                        continue
                    for task_name in sorted(os.listdir(suite_dir)):
                        task_dir = os.path.join(suite_dir, task_name)
                        if not os.path.isdir(task_dir):
                            continue
                        for traj_id in sorted(os.listdir(task_dir)):
                            demo_dir = os.path.join(task_dir, traj_id)
                            if not os.path.isdir(demo_dir) or not traj_id.startswith("demo_"):
                                continue
                            yield suite, task_name, traj_id, demo_dir, difficulty, preset

    def _build_index(self) -> list[_LiberoEpisodeSample]:
        samples: list[_LiberoOODEpisodeSample] = []
        skipped_missing = 0
        skipped_short = 0
        used_index = False

        index_rows = list(self._iter_from_index_jsonl() or [])
        if index_rows:
            used_index = True
            for meta, demo_dir, suite, task_name, traj_id, difficulty, preset in index_rows:
                sample = self._sample_from_meta(
                    meta=meta,
                    demo_dir=demo_dir,
                    suite=suite,
                    task_name=task_name,
                    traj_id=traj_id,
                    view_difficulty=difficulty,
                    camera_preset=preset,
                )
                if sample is None:
                    # Distinguish short vs missing via frames/length probe failure.
                    length = int(meta.get("frames") or 0)
                    if length > 0 and is_too_short_for_window(
                        length, self.sequence_length, self.uniform_sample_episode
                    ):
                        skipped_short += 1
                    else:
                        skipped_missing += 1
                    continue
                samples.append(sample)
        else:
            for suite, task_name, traj_id, demo_dir, difficulty, preset in self._iter_demo_dirs():
                meta_path = os.path.join(demo_dir, "metadata.json")
                meta: dict[str, Any] = {}
                if os.path.isfile(meta_path):
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                # Fill convention paths when metadata lacks them.
                meta.setdefault("rgb", os.path.join(demo_dir, "rgb.mp4"))
                meta.setdefault("action", os.path.join(demo_dir, "action.npy"))
                meta.setdefault("action_video", os.path.join(demo_dir, "action_video.mp4"))
                meta.setdefault("view_difficulty", difficulty)
                meta.setdefault("camera_preset", preset)

                length_hint = int(meta.get("frames") or 0)
                sample = self._sample_from_meta(
                    meta=meta,
                    demo_dir=demo_dir,
                    suite=suite,
                    task_name=task_name,
                    traj_id=traj_id,
                    view_difficulty=difficulty,
                    camera_preset=preset,
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
            kept: list[_LiberoOODEpisodeSample] = []
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
            f"Indexed {len(samples)} LIBERO OOD action episodes "
            f"(index_jsonl={'yes' if used_index else 'no'}, "
            f"skipped_missing={skipped_missing}, skipped_short={skipped_short}, "
            f"verify_files={self.verify_files}, require_action_video={self.require_action_video}, "
            f"difficulties={list(self.view_difficulties)})"
        )
        return samples  # type: ignore[return-value]


__all__ = [
    "DEFAULT_DATASET_DIR",
    "DEFAULT_TEMPORAL_STRIDE",
    "DEFAULT_VIEW_DIFFICULTIES",
    "LiberoOODDataset",
]
