# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""AgiBot-World (LeRobot v2.1 per-task) video dataset for Cosmos Predict2."""

from __future__ import annotations

import gzip
import json
import os
import pickle
import re
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
import pyarrow.parquet as pq
import torch
from decord import VideoReader, cpu
from torch.utils.data import Dataset
from torchvision import transforms as T

from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_utils import (
    ResizePreprocess,
    ToTensorVideo,
    is_excluded_video_path,
    is_too_short_for_window,
    load_exclude_video_paths,
    parallel_path_exists,
    sample_video_frame_indices,
)

TARGET_FPS = 16.0
DEFAULT_CAMERA_KEY = "observation.images.head_color"

# Official / OminiEWM caption dump aligned with AgiBot-World_v2.1_per_task_fixed.
DEFAULT_CAPTION_METADATA = (
    "/media/datasets/OminiEWM_Data/labels/train/agibot_world/"
    "agibot_world_full_with_caption_v3_vqp_clean.pkl.gz"
)

# Flattened parquet columns for non-arm DOFs (head / waist / chassis).
_NON_ARM_ACTION_COLUMNS: tuple[str, ...] = (
    "action.head.position.0",
    "action.head.position.1",
    "action.waist.position.0",
    "action.waist.position.1",
    "action.robot.velocity.0",
    "action.robot.velocity.1",
)
_HEAD_COLS = _NON_ARM_ACTION_COLUMNS[0:2]
_WAIST_COLS = _NON_ARM_ACTION_COLUMNS[2:4]
_ROBOT_VEL_COLS = _NON_ARM_ACTION_COLUMNS[4:6]
DEFAULT_NON_ARM_MOTION_THRESHOLD = 1e-3
_ARM_ONLY_CACHE_VERSION = 1

_TASK_ID_RE = re.compile(r"^task[_\s-]?\d+$", re.IGNORECASE)


@dataclass(frozen=True)
class _EpisodeSample:
    video_path: str
    caption: str
    length: int
    task_name: str
    episode_index: int
    camera_key: str


def _is_placeholder_instruction(text: str) -> bool:
    """True for bare task ids like ``task_327`` / ``task 327``."""
    t = text.strip()
    if not t:
        return True
    if _TASK_ID_RE.match(t):
        return True
    # Folder-style placeholders: agibot_gripper__task_327
    if re.search(r"__task_\d+$", t, flags=re.IGNORECASE):
        return True
    return False


def _get_nested_field(item: dict[str, Any], field: str) -> Any:
    """Resolve ``item[field]`` or dotted paths like ``Caption.language_instruction.task``."""
    if "." not in field:
        return item.get(field)
    cur: Any = item
    for part in field.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _caption_from_metadata_item(item: dict[str, Any], caption_field: str) -> str:
    """Pick the best natural-language instruction from an OminiEWM caption record."""
    # Preferred field (default: Caption.language_instruction.task).
    val = _get_nested_field(item, caption_field)
    if isinstance(val, str) and val.strip() and not _is_placeholder_instruction(val):
        return val.strip()

    # Fallbacks: nested task instruction, then natural-language captions.
    for key in (
        "Caption.language_instruction.task",
        "caption_natural_short_1",
        "caption_natural_short_2",
        "task_long",
        "caption_natural_long_1",
    ):
        val = _get_nested_field(item, key)
        if isinstance(val, str) and val.strip() and not _is_placeholder_instruction(val):
            return val.strip()

    caption_obj = item.get("Caption")
    if isinstance(caption_obj, dict):
        actions = caption_obj.get("action_descriptions")
        if isinstance(actions, list):
            parts = [str(a).strip() for a in actions if str(a).strip()]
            if parts:
                return " ".join(parts)

    lang = item.get("lang")
    if isinstance(lang, str) and lang.strip() and not _is_placeholder_instruction(lang):
        return lang.strip()
    return ""


def _column_range(table: Any, columns: tuple[str, ...]) -> float:
    arr = np.column_stack([table.column(c).to_numpy(zero_copy_only=False) for c in columns]).astype(
        np.float64, copy=False
    )
    if arr.size == 0:
        return 0.0
    return float((arr.max(axis=0) - arr.min(axis=0)).max())


def _column_absmax(table: Any, columns: tuple[str, ...]) -> float:
    arr = np.column_stack([table.column(c).to_numpy(zero_copy_only=False) for c in columns]).astype(
        np.float64, copy=False
    )
    if arr.size == 0:
        return 0.0
    return float(np.abs(arr).max())


def episode_non_arm_motion(
    parquet_path: str,
    threshold: float = DEFAULT_NON_ARM_MOTION_THRESHOLD,
) -> dict[str, float]:
    """Return per-channel motion magnitudes for head / waist / chassis.

    - head / waist: max per-dim range over the episode (``max - min``)
    - robot velocity: max absolute value over the episode

    Missing files raise ``FileNotFoundError``. Missing columns are treated as 0.
    """
    del threshold  # reserved for callers; magnitudes are absolute.
    if not os.path.isfile(parquet_path):
        raise FileNotFoundError(parquet_path)

    out = {"head_range": 0.0, "waist_range": 0.0, "robot_vel_abs": 0.0}
    try:
        table = pq.read_table(parquet_path, columns=list(_NON_ARM_ACTION_COLUMNS))
    except Exception:
        # Some shards may miss a subset of columns; read whatever exists.
        schema_names = set(pq.read_schema(parquet_path).names)
        cols = [c for c in _NON_ARM_ACTION_COLUMNS if c in schema_names]
        if not cols:
            return out
        table = pq.read_table(parquet_path, columns=cols)

    names = set(table.column_names)
    head = tuple(c for c in _HEAD_COLS if c in names)
    waist = tuple(c for c in _WAIST_COLS if c in names)
    robot = tuple(c for c in _ROBOT_VEL_COLS if c in names)
    if head:
        out["head_range"] = _column_range(table, head)
    if waist:
        out["waist_range"] = _column_range(table, waist)
    if robot:
        out["robot_vel_abs"] = _column_absmax(table, robot)
    return out


def episode_has_non_arm_motion(
    parquet_path: str,
    threshold: float = DEFAULT_NON_ARM_MOTION_THRESHOLD,
) -> tuple[bool, dict[str, float]]:
    """True if head turn / waist lift / chassis velocity exceeds ``threshold``."""
    m = episode_non_arm_motion(parquet_path, threshold=threshold)
    has = (
        m["head_range"] > threshold
        or m["waist_range"] > threshold
        or m["robot_vel_abs"] > threshold
    )
    return has, m


class AgiBotDataset(Dataset):
    """AgiBot-World LeRobot-style dataset under ``<root>/<task_folder>/``.

    Expected layout (v2.1)::

        <task_folder>/
          meta/episodes.jsonl
          meta/info.json
          meta/tasks.jsonl          # optional; often only ``task_<id>``
          videos/<camera_key>/episode_XXXXXX.mp4
          data/episode_XXXXXX.parquet

    Only the **primary RGB** head camera is used by default
    (``observation.images.head_color``). Side fisheye / hand views are ignored.

    Instructions:
        On-disk ``tasks`` fields are usually placeholders (``task_327``). Prefer
        captions from ``caption_metadata_path`` (OminiEWM
        ``agibot_world_full_with_caption_v3_vqp_clean.pkl.gz``), field
        ``Caption.language_instruction.task`` by default — short task instructions
        analogous to RoboMind's ``lang``.

    Non-arm motion filter (``filter_non_arm_motion=True``):
        Drops episodes where parquet actions show head turn
        (``action.head.position``), waist lift (``action.waist.position``), or
        chassis velocity (``action.robot.velocity``) above
        ``non_arm_motion_threshold``. Results are cached under the dataset root.

    Frame sampling matches ``RoboMindDataset`` / ``RealSourceWorldDataset``:
        - ``uniform_sample_episode=False``: random contiguous ``num_frames`` window.
        - ``uniform_sample_episode=True``: linspace the full episode to ``num_frames``.

    Output keys match Video2World::

        video [C,T,H,W] uint8, ai_caption, fps, image_size, num_frames, padding_mask
    """

    def __init__(
        self,
        dataset_dir: str,
        num_frames: int,
        video_size: tuple[int, int],
        camera_key: str = DEFAULT_CAMERA_KEY,
        skip_empty_instruction: bool = True,
        verify_files: bool = True,
        target_fps: float = TARGET_FPS,
        uniform_sample_episode: bool = False,
        caption_metadata_path: Optional[str] = DEFAULT_CAPTION_METADATA,
        caption_field: str = "Caption.language_instruction.task",
        filter_non_arm_motion: bool = True,
        non_arm_motion_threshold: float = DEFAULT_NON_ARM_MOTION_THRESHOLD,
        arm_only_cache_path: Optional[str] = None,
        non_arm_filter_workers: int = 32,
        start_from_first_frame_prob: float = 0.0,
        exclude_video_paths: Optional[Union[str, list[str]]] = None,
    ) -> None:
        super().__init__()
        self.dataset_dir = dataset_dir
        self.sequence_length = int(num_frames)
        self.video_size = tuple(video_size)
        self.camera_key = camera_key
        self.skip_empty_instruction = skip_empty_instruction
        self.verify_files = verify_files
        self.target_fps = float(target_fps)
        self.uniform_sample_episode = bool(uniform_sample_episode)
        self.caption_metadata_path = caption_metadata_path
        self.caption_field = caption_field
        self.filter_non_arm_motion = bool(filter_non_arm_motion)
        self.non_arm_motion_threshold = float(non_arm_motion_threshold)
        self.non_arm_filter_workers = max(1, int(non_arm_filter_workers))
        self.start_from_first_frame_prob = float(start_from_first_frame_prob)
        self.exclude_video_paths = load_exclude_video_paths(exclude_video_paths)
        if arm_only_cache_path:
            self.arm_only_cache_path = arm_only_cache_path
        else:
            cache_name = (
                f".agibot_arm_only_cache_v{_ARM_ONLY_CACHE_VERSION}"
                f"_thr{self.non_arm_motion_threshold:g}.json.gz"
            )
            primary = os.path.join(self.dataset_dir, cache_name)
            if os.access(self.dataset_dir, os.W_OK):
                self.arm_only_cache_path = primary
            else:
                fallback_root = os.environ.get("IMAGINAIRE_OUTPUT_ROOT") or os.getcwd()
                self.arm_only_cache_path = os.path.join(fallback_root, cache_name)
                log.warning(
                    f"AgiBotDataset: dataset_dir not writable; arm-only cache → {self.arm_only_cache_path}"
                )
        self.num_failed_loads = 0

        if not os.path.isdir(self.dataset_dir):
            raise FileNotFoundError(f"AgiBot dataset dir not found: {self.dataset_dir}")

        self._caption_by_relpath = self._load_caption_index()
        self.samples = self._build_index()
        if not self.samples:
            raise RuntimeError(
                f"No valid episodes found under {self.dataset_dir} "
                f"(camera_key={self.camera_key!r}, num_frames={self.sequence_length})."
            )

        self.video_paths = [s.video_path for s in self.samples]
        log.info(
            f"AgiBotDataset: {len(self.samples)} episodes from {self.dataset_dir} "
            f"(camera={self.camera_key}, num_frames={self.sequence_length}, "
            f"video_size={self.video_size}, target_fps={self.target_fps}, "
            f"uniform_sample_episode={self.uniform_sample_episode}, "
            f"verify_files={self.verify_files}, "
            f"filter_non_arm_motion={self.filter_non_arm_motion}, "
            f"non_arm_motion_threshold={self.non_arm_motion_threshold}, "
            f"start_from_first_frame_prob={self.start_from_first_frame_prob}, "
            f"caption_map={len(self._caption_by_relpath)})"
        )
        self.preprocess = T.Compose(
            [ToTensorVideo(), ResizePreprocess((self.video_size[0], self.video_size[1]))]
        )

    def __str__(self) -> str:
        return f"{len(self.samples)} samples from {self.dataset_dir}"

    def __len__(self) -> int:
        return len(self.samples)

    def _load_caption_index(self) -> dict[str, str]:
        path = self.caption_metadata_path
        if not path:
            log.warning("AgiBotDataset: no caption_metadata_path; instructions may be placeholders")
            return {}
        if not os.path.isfile(path):
            log.warning(f"AgiBotDataset: caption metadata not found: {path}")
            return {}

        log.info(f"Loading AgiBot caption metadata: {path}")
        with gzip.open(path, "rb") as f:
            raw = pickle.load(f)
        if isinstance(raw, dict):
            for key in ("data", "items", "samples", "dataset"):
                if key in raw and isinstance(raw[key], list):
                    raw = raw[key]
                    break
            else:
                raise TypeError(f"Unsupported caption pickle dict keys: {list(raw.keys())[:20]}")
        if not isinstance(raw, list):
            raise TypeError(f"Expected list in caption pickle, got {type(raw)}")

        index: dict[str, str] = {}
        for item in raw:
            if not isinstance(item, dict):
                continue
            caption = _caption_from_metadata_item(item, self.caption_field)
            if not caption:
                continue
            cvi = item.get("camera_video_info") or {}
            view = cvi.get(self.camera_key) or cvi.get(DEFAULT_CAMERA_KEY) or {}
            if not isinstance(view, dict):
                continue
            rel = str(view.get("video") or "").strip()
            if not rel:
                continue
            # Normalize to relative key under the per-task root.
            rel = rel.lstrip("/")
            for prefix in (
                "media/datasets/AgiBot-World_v2.1_per_task_fixed/",
                "datasets/AgiBot-World_v2.1_per_task_fixed/",
                "AgiBot-World_v2.1_per_task_fixed/",
            ):
                if rel.startswith(prefix):
                    rel = rel[len(prefix) :]
                    break
            if os.path.isabs(str(view.get("video") or "")):
                # Absolute path → keep basename folder key if possible.
                marker = "AgiBot-World_v2.1_per_task_fixed/"
                abs_v = str(view.get("video"))
                if marker in abs_v:
                    rel = abs_v.split(marker, 1)[1]
            index[rel.replace("\\", "/")] = caption

        del raw
        log.info(f"AgiBot caption index: {len(index)} entries from {path}")
        return index

    def _iter_task_dirs(self) -> list[str]:
        task_dirs: list[str] = []
        for name in sorted(os.listdir(self.dataset_dir)):
            path = os.path.join(self.dataset_dir, name)
            if not os.path.isdir(path):
                continue
            episodes_path = os.path.join(path, "meta", "episodes.jsonl")
            if os.path.isfile(episodes_path):
                task_dirs.append(path)
        return task_dirs

    def _load_tasks_map(self, task_dir: str) -> dict[int, str]:
        tasks_path = os.path.join(task_dir, "meta", "tasks.jsonl")
        out: dict[int, str] = {}
        if not os.path.isfile(tasks_path):
            return out
        with open(tasks_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                idx = int(obj.get("task_index", 0))
                text = str(obj.get("task") or "").strip()
                if text:
                    out[idx] = text
        return out

    def _resolve_instruction(
        self,
        *,
        rel_video: str,
        episode: dict[str, Any],
        tasks_map: dict[int, str],
        task_name: str,
    ) -> str:
        # 1) Caption metadata (natural language).
        cap = self._caption_by_relpath.get(rel_video.replace("\\", "/"), "")
        if cap and not _is_placeholder_instruction(cap):
            return cap

        # 2) tasks.jsonl / episode tasks if they look like real text.
        task_index = int(episode.get("task_index", 0))
        for candidate in (
            tasks_map.get(task_index, ""),
            *(episode.get("tasks") or []),
            episode.get("task_name") or "",
        ):
            text = str(candidate).strip()
            if text and not _is_placeholder_instruction(text):
                return text

        # 3) No usable instruction.
        return ""

    def _parquet_path(self, task_name: str, episode_index: int) -> str:
        return os.path.join(
            self.dataset_dir,
            task_name,
            "data",
            f"episode_{episode_index:06d}.parquet",
        )

    def _arm_only_cache_key(self, task_name: str, episode_index: int) -> str:
        return f"{task_name}/episode_{episode_index:06d}"

    def _load_arm_only_cache(self) -> Optional[dict[str, bool]]:
        path = self.arm_only_cache_path
        if not path or not os.path.isfile(path):
            return None
        try:
            with gzip.open(path, "rt", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as e:
            log.warning(f"AgiBotDataset: failed to load arm-only cache {path}: {e}")
            return None
        if not isinstance(payload, dict):
            return None
        if int(payload.get("version", -1)) != _ARM_ONLY_CACHE_VERSION:
            return None
        if abs(float(payload.get("threshold", -1.0)) - self.non_arm_motion_threshold) > 1e-12:
            return None
        decisions = payload.get("decisions")
        if not isinstance(decisions, dict):
            return None
        # decisions[key] = True means KEEP (arm-only); False means reject.
        return {str(k): bool(v) for k, v in decisions.items()}

    def _save_arm_only_cache(self, decisions: dict[str, bool], reject_counts: dict[str, int]) -> None:
        path = self.arm_only_cache_path
        if not path:
            return
        payload = {
            "version": _ARM_ONLY_CACHE_VERSION,
            "threshold": self.non_arm_motion_threshold,
            "columns": list(_NON_ARM_ACTION_COLUMNS),
            "reject_counts": reject_counts,
            "decisions": decisions,
        }
        tmp = f"{path}.tmp"
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with gzip.open(tmp, "wt", encoding="utf-8") as f:
                json.dump(payload, f)
            os.replace(tmp, path)
            log.info(
                f"AgiBotDataset: wrote arm-only cache ({len(decisions)} episodes) → {path}"
            )
        except Exception as e:
            log.warning(f"AgiBotDataset: failed to write arm-only cache {path}: {e}")
            try:
                if os.path.isfile(tmp):
                    os.remove(tmp)
            except OSError:
                pass

    def _filter_non_arm_episodes(
        self, candidates: list[_EpisodeSample]
    ) -> tuple[list[_EpisodeSample], dict[str, int]]:
        """Keep arm-only episodes; drop head/waist/chassis motion."""
        reject_counts = {
            "head": 0,
            "waist": 0,
            "robot": 0,
            "any": 0,
            "missing_parquet": 0,
            "error": 0,
        }
        if not candidates:
            return [], reject_counts

        cache = self._load_arm_only_cache() or {}
        decisions: dict[str, bool] = dict(cache)
        pending: list[tuple[str, str]] = []  # (cache_key, parquet_path)
        key_by_sample: list[str] = []

        for sample in candidates:
            key = self._arm_only_cache_key(sample.task_name, sample.episode_index)
            key_by_sample.append(key)
            if key not in decisions:
                pending.append((key, self._parquet_path(sample.task_name, sample.episode_index)))

        thr = self.non_arm_motion_threshold
        if pending:
            log.info(
                f"AgiBotDataset: scanning {len(pending)} parquet files for non-arm motion "
                f"(cached={len(candidates) - len(pending)}, workers={self.non_arm_filter_workers}, "
                f"threshold={thr})"
            )

            def _eval(item: tuple[str, str]) -> tuple[str, Optional[bool], dict[str, float], str]:
                key, path = item
                if not os.path.isfile(path):
                    return key, None, {}, "missing"
                try:
                    has, metrics = episode_has_non_arm_motion(path, threshold=thr)
                    return key, (not has), metrics, "ok"
                except Exception:
                    return key, None, {}, "error"

            with ThreadPoolExecutor(max_workers=self.non_arm_filter_workers) as pool:
                futures = [pool.submit(_eval, item) for item in pending]
                for fut in as_completed(futures):
                    key, keep, metrics, status = fut.result()
                    if status == "missing":
                        reject_counts["missing_parquet"] += 1
                        decisions[key] = False
                        continue
                    if status == "error" or keep is None:
                        reject_counts["error"] += 1
                        decisions[key] = False
                        continue
                    decisions[key] = bool(keep)
                    if not keep:
                        if metrics.get("head_range", 0.0) > thr:
                            reject_counts["head"] += 1
                        if metrics.get("waist_range", 0.0) > thr:
                            reject_counts["waist"] += 1
                        if metrics.get("robot_vel_abs", 0.0) > thr:
                            reject_counts["robot"] += 1
                        reject_counts["any"] += 1

            # Persist merged cache (old + new).
            self._save_arm_only_cache(decisions, reject_counts)

        kept: list[_EpisodeSample] = []
        skipped = 0
        for sample, key in zip(candidates, key_by_sample):
            if decisions.get(key, False):
                kept.append(sample)
            else:
                skipped += 1

        reject_counts["skipped_total"] = skipped
        reject_counts["kept"] = len(kept)
        return kept, reject_counts

    def _build_index(self) -> list[_EpisodeSample]:
        samples: list[_EpisodeSample] = []
        skipped_missing = 0
        skipped_no_instruction = 0
        skipped_no_camera = 0
        skipped_short = 0
        caption_hits = 0

        for task_dir in self._iter_task_dirs():
            task_name = os.path.basename(task_dir.rstrip(os.sep))
            episodes_path = os.path.join(task_dir, "meta", "episodes.jsonl")
            tasks_map = self._load_tasks_map(task_dir)
            camera_dir = os.path.join(task_dir, "videos", self.camera_key)
            if not os.path.isdir(camera_dir):
                skipped_no_camera += 1
                continue

            with open(episodes_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    episode = json.loads(line)
                    length = int(episode.get("length", 0))

                    episode_index = int(episode.get("episode_index", -1))
                    if episode_index < 0:
                        skipped_missing += 1
                        continue

                    rel_video = (
                        f"{task_name}/videos/{self.camera_key}/"
                        f"episode_{episode_index:06d}.mp4"
                    )
                    video_path = os.path.join(self.dataset_dir, rel_video)
                    if is_excluded_video_path(video_path, self.exclude_video_paths):
                        skipped_missing += 1
                        continue
                    if is_too_short_for_window(
                        length, self.sequence_length, self.uniform_sample_episode
                    ):
                        skipped_short += 1
                        continue

                    caption = self._resolve_instruction(
                        rel_video=rel_video,
                        episode=episode,
                        tasks_map=tasks_map,
                        task_name=task_name,
                    )
                    if caption and rel_video.replace("\\", "/") in self._caption_by_relpath:
                        caption_hits += 1
                    if self.skip_empty_instruction and (
                        not caption or _is_placeholder_instruction(caption)
                    ):
                        skipped_no_instruction += 1
                        continue

                    samples.append(
                        _EpisodeSample(
                            video_path=video_path,
                            caption=caption,
                            length=length,
                            task_name=task_name,
                            episode_index=episode_index,
                            camera_key=self.camera_key,
                        )
                    )

        if self.verify_files and samples:
            exists = parallel_path_exists([s.video_path for s in samples])
            before = len(samples)
            samples = [s for s, ok in zip(samples, exists) if ok]
            skipped_missing += before - len(samples)

        skipped_non_arm = 0
        if self.filter_non_arm_motion and samples:
            before = len(samples)
            samples, reject_counts = self._filter_non_arm_episodes(samples)
            skipped_non_arm = before - len(samples)
            log.info(
                f"AgiBot non-arm motion filter: kept={len(samples)}/{before} "
                f"(skipped={skipped_non_arm}, "
                f"head={reject_counts.get('head', 0)}, "
                f"waist={reject_counts.get('waist', 0)}, "
                f"robot={reject_counts.get('robot', 0)}, "
                f"missing_parquet={reject_counts.get('missing_parquet', 0)}, "
                f"error={reject_counts.get('error', 0)})"
            )

        log.info(
            f"Indexed {len(samples)} AgiBot episodes "
            f"(skipped_missing={skipped_missing}, "
            f"skipped_no_instruction={skipped_no_instruction}, "
            f"skipped_no_camera={skipped_no_camera}, "
            f"skipped_short={skipped_short}, "
            f"skipped_non_arm_motion={skipped_non_arm}, "
            f"verify_files={self.verify_files}, "
            f"caption_hits={caption_hits})"
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

    def _sample_frame_ids(self, total_frames: int, src_fps: Optional[float]) -> tuple[list[int], float]:
        return sample_video_frame_indices(
            total_frames,
            self.sequence_length,
            uniform_sample_episode=self.uniform_sample_episode,
            src_fps=src_fps,
            target_fps=self.target_fps,
            start_from_first_frame_prob=self.start_from_first_frame_prob,
        )

    def _load_video(self, video_path: str) -> tuple[np.ndarray, float]:
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file does not exist: {video_path}")
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        total_frames = len(vr)
        src_fps = self._read_fps(vr)
        frame_ids, reported_fps = self._sample_frame_ids(total_frames, src_fps)
        frame_data = vr.get_batch(frame_ids).asnumpy()
        vr.seek(0)
        del vr
        return frame_data, reported_fps

    def _get_frames(self, video_path: str) -> tuple[torch.Tensor, float]:
        frames, fps = self._load_video(video_path)
        frames = frames.astype(np.uint8)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # [T, C, H, W]
        frames = self.preprocess(frames)
        frames = torch.clamp(frames * 255.0, 0, 255).to(torch.uint8)
        return frames, fps

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        try:
            video, fps = self._get_frames(sample.video_path)
            video = video.permute(1, 0, 2, 3)  # [C, T, H, W]
            _, _, h, w = video.shape
            return {
                "video": video,
                "ai_caption": sample.caption,
                "fps": fps,
                "image_size": torch.tensor([h, w, h, w]),
                "num_frames": self.sequence_length,
                "padding_mask": torch.zeros(1, h, w),
            }
        except Exception as e:
            self.num_failed_loads += 1
            log.warning(
                f"Failed to load video {sample.video_path} "
                f"(task={sample.task_name}, ep={sample.episode_index}, "
                f"total failures: {self.num_failed_loads}): {e}\n{traceback.format_exc()}",
                rank0_only=False,
            )
            return self[int(np.random.randint(len(self.samples)))]
