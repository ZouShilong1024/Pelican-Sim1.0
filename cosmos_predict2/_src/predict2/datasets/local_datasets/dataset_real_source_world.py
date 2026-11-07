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

"""RealSource-World (LeRobot-style) video dataset for Cosmos Predict2."""

from __future__ import annotations

import json
import os
import traceback
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
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

DEFAULT_CAMERA_KEY = "observation.images.head_camera"
TARGET_FPS = 16.0


@dataclass(frozen=True)
class _EpisodeSample:
    video_path: str
    caption: str
    length: int
    task_name: str
    episode_index: int


class RealSourceWorldDataset(Dataset):
    """Dataset for RealSource-World layout under ``<root>/<TaskName>/``.

    Expected per-task layout (LeRobot v2.1)::

        <TaskName>/
          meta/episodes.jsonl   # episode -> videos + tasks
          meta/info.json        # fps, resolution, ...
          videos/chunk-XXX/<camera_key>/episode_NNNNNN.mp4

    Frame sampling:
        - ``uniform_sample_episode=False`` (default): randomly choose a start
          so a contiguous ``num_frames`` clip fits (native source FPS; no
          target-FPS temporal resampling).
        - ``uniform_sample_episode=True``: uniformly linspace the **entire**
          episode down/up to exactly ``num_frames`` (always covers start→end).

    Missing videos:
        - ``verify_files=True`` (default): skip episodes whose mp4 is absent on
          disk during indexing (avoids decord open errors at train time).
        - Set ``verify_files=False`` to skip the existence scan (faster index on
          slow NFS; missing files are retried in ``__getitem__``).

    Output keys match ``VideoDataset`` so samples can be fed into Video2World
    training unchanged (``input_data_key=video``, ``input_caption_key=ai_caption``):

        - video: uint8 tensor [C, T, H, W]
        - ai_caption: str
        - fps: float (native source FPS when known; else ``target_fps`` fallback)
        - image_size: tensor [h, w, h, w]
        - num_frames: int
        - padding_mask: zeros [1, H, W]
    """

    def __init__(
        self,
        dataset_dir: str,
        num_frames: int,
        video_size: tuple[int, int],
        camera_key: str = DEFAULT_CAMERA_KEY,
        verify_files: bool = True,
        target_fps: float = TARGET_FPS,
        uniform_sample_episode: bool = False,
        start_from_first_frame_prob: float = 0.0,
        exclude_video_paths: Optional[Union[str, list[str]]] = None,
    ) -> None:
        super().__init__()
        self.dataset_dir = dataset_dir
        self.sequence_length = num_frames
        self.video_size = tuple(video_size)
        self.camera_key = camera_key
        self.verify_files = verify_files
        self.target_fps = float(target_fps)
        self.uniform_sample_episode = bool(uniform_sample_episode)
        self.start_from_first_frame_prob = float(start_from_first_frame_prob)
        self.exclude_video_paths = load_exclude_video_paths(exclude_video_paths)
        self.num_failed_loads = 0

        if not os.path.isdir(self.dataset_dir):
            raise FileNotFoundError(f"RealSource-World dataset dir not found: {self.dataset_dir}")

        self.samples = self._build_index()
        if not self.samples:
            raise RuntimeError(
                f"No valid episodes found under {self.dataset_dir} "
                f"(camera_key={self.camera_key!r}, num_frames={self.sequence_length})."
            )

        # Keep attribute name aligned with VideoDataset for debugging / logging.
        self.video_paths = [s.video_path for s in self.samples]
        log.info(
            f"RealSourceWorldDataset: {len(self.samples)} episodes from {self.dataset_dir} "
            f"(camera={self.camera_key}, num_frames={self.sequence_length}, "
            f"video_size={self.video_size}, target_fps={self.target_fps}, "
            f"uniform_sample_episode={self.uniform_sample_episode}, "
            f"start_from_first_frame_prob={self.start_from_first_frame_prob}, "
            f"verify_files={self.verify_files})"
        )

        self.preprocess = T.Compose(
            [ToTensorVideo(), ResizePreprocess((self.video_size[0], self.video_size[1]))]
        )

    def __str__(self) -> str:
        return f"{len(self.samples)} samples from {self.dataset_dir}"

    def __len__(self) -> int:
        return len(self.samples)

    def _iter_task_dirs(self) -> list[str]:
        task_dirs = []
        for name in sorted(os.listdir(self.dataset_dir)):
            path = os.path.join(self.dataset_dir, name)
            if not os.path.isdir(path):
                continue
            episodes_path = os.path.join(path, "meta", "episodes.jsonl")
            if os.path.isfile(episodes_path):
                task_dirs.append(path)
        return task_dirs

    def _build_index(self) -> list[_EpisodeSample]:
        samples: list[_EpisodeSample] = []
        skipped_missing = 0
        skipped_short = 0

        for task_dir in self._iter_task_dirs():
            task_name = os.path.basename(task_dir.rstrip(os.sep))
            episodes_path = os.path.join(task_dir, "meta", "episodes.jsonl")
            with open(episodes_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    episode = json.loads(line)
                    length = int(episode.get("length", 0))

                    videos = episode.get("videos") or {}
                    rel_video = videos.get(self.camera_key)
                    if not rel_video:
                        skipped_missing += 1
                        continue

                    video_path = os.path.join(task_dir, rel_video)
                    if is_excluded_video_path(video_path, self.exclude_video_paths):
                        skipped_missing += 1
                        continue
                    if is_too_short_for_window(
                        length, self.sequence_length, self.uniform_sample_episode
                    ):
                        skipped_short += 1
                        continue

                    tasks = episode.get("tasks") or []
                    if isinstance(tasks, str):
                        caption = tasks.strip()
                    else:
                        caption = ". ".join(str(t).strip() for t in tasks if str(t).strip())
                    if not caption:
                        caption = task_name.replace("_", " ")

                    samples.append(
                        _EpisodeSample(
                            video_path=video_path,
                            caption=caption,
                            length=length,
                            task_name=task_name,
                            episode_index=int(episode.get("episode_index", -1)),
                        )
                    )

        if self.verify_files and samples:
            exists = parallel_path_exists([s.video_path for s in samples])
            before = len(samples)
            samples = [s for s, ok in zip(samples, exists) if ok]
            skipped_missing += before - len(samples)

        log.info(
            f"Indexed {len(samples)} episodes "
            f"(skipped_missing={skipped_missing}, skipped_short={skipped_short}, "
            f"verify_files={self.verify_files})"
        )
        return samples

    def _read_fps(self, vr: VideoReader) -> Optional[float]:
        """Return average FPS if available and valid; otherwise None (skip alignment)."""
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
                f"(total failures: {self.num_failed_loads}): {e}\n{traceback.format_exc()}",
                rank0_only=False,
            )
            return self[int(np.random.randint(len(self.samples)))]
