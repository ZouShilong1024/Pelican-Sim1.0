# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sample action-inject inference episodes from the training action datasets.

Aligned with ``cosmos_predict2/experiments/base/action_inject.py`` (and MoE twin):

  - RealSourceWorldActionDataset (temporal_stride=4)
  - AgibotActionDataset (temporal_stride=9)
  - AgibotDexhandActionDataset (temporal_stride=12)
  - RobomindActionDataset (temporal_stride=2, main view ``camera``)
  - LiberoActionDataset (stride=1)
  - LiberoOODDataset (stride=1; view-gen val under ``libero_view_gen_val``)
  - UR5RobosuiteOOD / IIWARobosuiteOOD (stride=1; Robosuite OOD AC exports)
  - ManiSkillActionDataset (stride=1)
  - RobotwinActionDataset (temporal_stride=2)
  - RobocasaActionDataset (temporal_stride=2, left/right, dualview_v2 + mg_v2)
  - Weighted mix default matches training ``MIX_WEIGHTS``:
    ``[0.20, 0.20, 0.08, 0.20, 0.08, 0.08, 0.08, 0.08]``
    (``libero_ood`` / ``ur5_robosuite_ood`` / ``iiwa_robosuite_ood`` are
    inference-only; not part of the training mix)

Returns path + caption specs (no heavy video decode) for the inference CLI.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.predict2.action.inference.action_inject_pipeline import (
    load_action_agibot_dexhand_parquet,
    load_action_agibot_parquet,
    load_action_npy,
    load_action_robomind_parquet,
    load_action_robotwin_hdf5,
    load_action_rsw_parquet,
)
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_agibot_action import (
    DEFAULT_LABELS_PKL as AGIBOT_LABELS_PKL,
    AgibotActionDataset,
)
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_agibot_dexhand_action import (
    DEFAULT_LABELS_JSONL as AGIBOT_DEXHAND_LABELS_JSONL,
    DEFAULT_TEMPORAL_STRIDE as AGIBOT_DEXHAND_TEMPORAL_STRIDE,
    AgibotDexhandActionDataset,
)
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_libero_action import (
    DEFAULT_DATASET_DIR as LIBERO_DATASET_DIR,
    DEFAULT_SUITES as LIBERO_DEFAULT_SUITES,
    LiberoActionDataset,
)
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_libero_ood import (
    DEFAULT_DATASET_DIR as LIBERO_OOD_DATASET_DIR,
    DEFAULT_TEMPORAL_STRIDE as LIBERO_OOD_TEMPORAL_STRIDE,
    LiberoOODDataset,
)
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_maniskill_action import (
    DEFAULT_DATASET_DIR as MANISKILL_DATASET_DIR,
    ManiSkillActionDataset,
)
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_oven_platephase_action import (
    DEFAULT_DATASET_DIR as OVEN_PLATEPHASE_DATASET_DIR,
    DEFAULT_TEMPORAL_STRIDE as OVEN_PLATEPHASE_TEMPORAL_STRIDE,
    OvenPlatephaseActionDataset,
)
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_robosuite_ood import (
    DEFAULT_IIWA_DATASET_DIR,
    DEFAULT_TEMPORAL_STRIDE as ROBOSUITE_OOD_TEMPORAL_STRIDE,
    DEFAULT_UR5_DATASET_DIR,
    IIWARobosuiteOOD,
    ROBOTIQ_140_GRIPPER_RAW_MAX,
    ROBOTIQ_85_GRIPPER_RAW_MAX,
    UR5RobosuiteOOD,
)
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_real_source_world_action import (
    RealSourceWorldActionDataset,
    build_action_delta_scaler,
)
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_robocasa_action import (
    DEFAULT_DATASET_DIRS as ROBOCASA_DATASET_DIRS,
    DEFAULT_TEMPORAL_STRIDE as ROBOCASA_TEMPORAL_STRIDE,
    RobocasaActionDataset,
)
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_robomind_action import (
    DEFAULT_PAIRS_JSONL as ROBOMIND_PAIRS_JSONL,
    DEFAULT_TEMPORAL_STRIDE as ROBOMIND_TEMPORAL_STRIDE,
    RobomindActionDataset,
)
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_robotwin_action import (
    DEFAULT_PAIRS_JSONL as ROBOTWIN_PAIRS_JSONL,
    DEFAULT_TEMPORAL_STRIDE as ROBOTWIN_TEMPORAL_STRIDE,
    RobotwinActionDataset,
)
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_weighted_multi import (
    WeightedMultiDataset,
)

DEFAULT_VIDEO_SIZE = (480, 640)  # (H, W)
DEFAULT_TARGET_FPS = 16.0
DEFAULT_UNIFORM_SAMPLE_EPISODE = False
# Inference: always start from episode beginning (training uses 0.3).
DEFAULT_START_FROM_FIRST_FRAME_PROB = 1.0
DEFAULT_VERIFY_FILES = False
DEFAULT_RSW_DIR = "/media/datasets/RealSource-World"
DEFAULT_AGIBOT_LABELS_PKL = AGIBOT_LABELS_PKL
DEFAULT_AGIBOT_DEXHAND_LABELS_JSONL = AGIBOT_DEXHAND_LABELS_JSONL
DEFAULT_LIBERO_DIR = LIBERO_DATASET_DIR
DEFAULT_LIBERO_OOD_DIR = LIBERO_OOD_DATASET_DIR
DEFAULT_UR5_ROBOSUITE_OOD_DIR = DEFAULT_UR5_DATASET_DIR
DEFAULT_IIWA_ROBOSUITE_OOD_DIR = DEFAULT_IIWA_DATASET_DIR
DEFAULT_MANISKILL_DIR = MANISKILL_DATASET_DIR
DEFAULT_OVEN_PLATEPHASE_DIR = OVEN_PLATEPHASE_DATASET_DIR
DEFAULT_ROBOTWIN_PAIRS_JSONL = ROBOTWIN_PAIRS_JSONL
DEFAULT_ROBOMIND_PAIRS_JSONL = ROBOMIND_PAIRS_JSONL
DEFAULT_ROBOCASA_DIRS = list(ROBOCASA_DATASET_DIRS)
# Keep in sync with action_inject.MIX_WEIGHTS (same order / ratios).
# [real_source_world, agibot, agibot_dexhand, robomind, libero, maniskill, robotwin, robocasa]
# Real ~0.68 (dexhand lower); sim total 0.32.
DEFAULT_MIX_WEIGHTS = (0.20, 0.20, 0.08, 0.20, 0.08, 0.08, 0.08, 0.08)
DEFAULT_MIX_NAMES = (
    "real_source_world",
    "agibot",
    "agibot_dexhand",
    "robomind",
    "libero",
    "maniskill",
    "robotwin",
    "robocasa",
)
DEFAULT_RSW_TEMPORAL_STRIDE = 4
DEFAULT_AGIBOT_TEMPORAL_STRIDE = 9
DEFAULT_AGIBOT_DEXHAND_TEMPORAL_STRIDE = int(AGIBOT_DEXHAND_TEMPORAL_STRIDE)
DEFAULT_ROBOMIND_TEMPORAL_STRIDE = int(ROBOMIND_TEMPORAL_STRIDE)
DEFAULT_ROBOTWIN_TEMPORAL_STRIDE = int(ROBOTWIN_TEMPORAL_STRIDE)
DEFAULT_ROBOCASA_TEMPORAL_STRIDE = int(ROBOCASA_TEMPORAL_STRIDE)
ACTION_DELTA_SCALER = build_action_delta_scaler(include_gripper=True, include_lift=False)

SOURCE_TEMPORAL_STRIDE: dict[str, int] = {
    "real_source_world": DEFAULT_RSW_TEMPORAL_STRIDE,
    "agibot": DEFAULT_AGIBOT_TEMPORAL_STRIDE,
    "agibot_dexhand": DEFAULT_AGIBOT_DEXHAND_TEMPORAL_STRIDE,
    "robomind": DEFAULT_ROBOMIND_TEMPORAL_STRIDE,
    "libero": 1,
    "libero_ood": int(LIBERO_OOD_TEMPORAL_STRIDE),  # same as Libero (=1)
    "ur5_robosuite_ood": int(ROBOSUITE_OOD_TEMPORAL_STRIDE),
    "iiwa_robosuite_ood": int(ROBOSUITE_OOD_TEMPORAL_STRIDE),
    "maniskill": 1,
    "robotwin": DEFAULT_ROBOTWIN_TEMPORAL_STRIDE,
    "robocasa": DEFAULT_ROBOCASA_TEMPORAL_STRIDE,
    "oven_platephase": int(OVEN_PLATEPHASE_TEMPORAL_STRIDE),
    # Sample dirs carry the per-episode stride (2, or 1 for short episodes) in meta.json.
    "robodojo": 2,
}


def _oven_platephase_temporal_stride() -> int:
    env = os.environ.get("OVEN_PLATEPHASE_TEMPORAL_STRIDE")
    if env is not None and str(env).strip() != "":
        return max(int(env), 1)
    return max(int(OVEN_PLATEPHASE_TEMPORAL_STRIDE), 1)

# Dual-arm sources in the unified 28-D layout.
_BIMANUAL_SOURCES = frozenset(
    {
        "real_source_world",
        "agibot",
        "agibot_dexhand",
        "robomind",
        "robotwin",
        "oven_platephase",
        "robodojo",
    }
)


class ActionDatasetSource(str, Enum):
    REAL_SOURCE_WORLD = "real_source_world"
    AGIBOT = "agibot"
    AGIBOT_DEXHAND = "agibot_dexhand"
    ROBOMIND = "robomind"
    LIBERO = "libero"
    LIBERO_OOD = "libero_ood"
    UR5_ROBOSUITE_OOD = "ur5_robosuite_ood"
    IIWA_ROBOSUITE_OOD = "iiwa_robosuite_ood"
    MANISKILL = "maniskill"
    ROBOTWIN = "robotwin"
    ROBOCASA = "robocasa"
    OVEN_PLATEPHASE = "oven_platephase"
    MIXED = "mixed"

    def __str__(self) -> str:
        return self.value


@dataclass
class ActionInjectEpisodeSpec:
    """One sampled episode ready for action-inject inference."""

    source: str
    index: int
    name: str
    prompt: str
    rgb_path: str
    action_video_path: str
    # Full original RGB video for the comparison middle panel. Empty = rgb_path.
    gt_rgb_path: str = ""
    # RGB of the transplanted action episode (Figure-7 source sequence). Empty = same as rgb_path.
    action_rgb_path: str = ""
    action_path: str = ""  # libero / maniskill / robocasa .npy; robomind parquet
    parquet_path: str = ""  # RSW / AgiBot / RoboMind parquet
    hdf5_path: str = ""  # RoboTwin hdf5
    num_arms: int = 2  # RSW / AgiBot / RoboMind / RoboTwin; libero/maniskill/robocasa=1
    length: int = 0
    already_unified: bool = False  # True when action.npy is already unified 28-D
    # Temporal subsample applied to RGB / action / action_video (training-aligned).
    temporal_stride: int = 1
    # RoboTwin JSONL ``task_name`` (empty for other sources / older manifests).
    task_name: str = ""
    # Libero OOD view-gen metadata (empty for other sources / older manifests).
    view_difficulty: str = ""
    camera_preset: str = ""
    metadata_path: str = ""
    robot: str = ""
    org_path: str = ""
    episode_index: Optional[int] = None
    # Model-rate fps of the exported clip (sample-dir ``meta.model_fps``); 0 = use CLI --fps.
    fps: float = 0.0
    # "mv_fill" = RoboDojo multi-view canvas (head + both wrists); "" = single camera.
    layout: str = ""
    # Multi-view only: view name → {"rgb": mp4, "action_video": mp4 or ""}, canvas order.
    views: dict = field(default_factory=dict)


def temporal_stride_for_source(source: str) -> int:
    """Training-aligned temporal stride for a dataset source name."""
    if str(source) == ActionDatasetSource.ROBOTWIN.value:
        env = os.environ.get("ROBOTWIN_TEMPORAL_STRIDE")
        if env is not None and str(env).strip() != "":
            return max(int(env), 1)
    if str(source) == ActionDatasetSource.OVEN_PLATEPHASE.value:
        return _oven_platephase_temporal_stride()
    return int(SOURCE_TEMPORAL_STRIDE.get(str(source), 1))


def build_action_inference_dataset(
    source: ActionDatasetSource | str,
    *,
    num_frames: int = 93,
    video_size: tuple[int, int] = DEFAULT_VIDEO_SIZE,
    target_fps: float = DEFAULT_TARGET_FPS,
    uniform_sample_episode: bool = DEFAULT_UNIFORM_SAMPLE_EPISODE,
    start_from_first_frame_prob: float = DEFAULT_START_FROM_FIRST_FRAME_PROB,
    rsw_dir: str = DEFAULT_RSW_DIR,
    agibot_labels_pkl: str = DEFAULT_AGIBOT_LABELS_PKL,
    agibot_dexhand_labels_jsonl: str = DEFAULT_AGIBOT_DEXHAND_LABELS_JSONL,
    robomind_pairs_jsonl: str = DEFAULT_ROBOMIND_PAIRS_JSONL,
    libero_dir: str = DEFAULT_LIBERO_DIR,
    libero_ood_dir: str = DEFAULT_LIBERO_OOD_DIR,
    ur5_robosuite_ood_dir: str = DEFAULT_UR5_ROBOSUITE_OOD_DIR,
    iiwa_robosuite_ood_dir: str = DEFAULT_IIWA_ROBOSUITE_OOD_DIR,
    maniskill_dir: str = DEFAULT_MANISKILL_DIR,
    oven_platephase_dir: str = DEFAULT_OVEN_PLATEPHASE_DIR,
    robotwin_pairs_jsonl: str = DEFAULT_ROBOTWIN_PAIRS_JSONL,
    robocasa_dirs: Optional[Sequence[str]] = None,
    mix_weights: Sequence[float] = DEFAULT_MIX_WEIGHTS,
    verify_files: bool = DEFAULT_VERIFY_FILES,
    load_action_video: bool = True,
    require_action_video: Optional[bool] = None,
) -> Dataset:
    """Instantiate action datasets matching ``action_inject`` training config.

    ``require_action_video`` defaults to False so episodes without a pre-rendered
    ``action_video.mp4`` stay in the index; inference renders the skeleton online.
    Pass ``True`` to keep the old "mp4 must exist at index time" filter.
    """
    source = ActionDatasetSource(source)
    if robocasa_dirs is None:
        robocasa_dirs = list(DEFAULT_ROBOCASA_DIRS)
    if require_action_video is None:
        require_action_video = False
    common = dict(
        num_frames=int(num_frames),
        video_size=tuple(video_size),
        target_fps=float(target_fps),
        uniform_sample_episode=bool(uniform_sample_episode),
        start_from_first_frame_prob=float(start_from_first_frame_prob),
        verify_files=bool(verify_files),
        load_action_video=bool(load_action_video),
        require_action_video=bool(require_action_video),
        include_gripper=True,
        action_delta_scaler=ACTION_DELTA_SCALER,
    )

    def _rsw() -> RealSourceWorldActionDataset:
        return RealSourceWorldActionDataset(
            dataset_dir=rsw_dir,
            camera_key="observation.images.head_camera",
            include_lift=False,
            temporal_stride=DEFAULT_RSW_TEMPORAL_STRIDE,
            **common,
        )

    def _agibot() -> AgibotActionDataset:
        return AgibotActionDataset(
            labels_pkl=agibot_labels_pkl,
            camera_key="observation.images.head_color",
            temporal_stride=DEFAULT_AGIBOT_TEMPORAL_STRIDE,
            **common,
        )

    def _agibot_dexhand() -> AgibotDexhandActionDataset:
        return AgibotDexhandActionDataset(
            labels_jsonl=agibot_dexhand_labels_jsonl,
            camera_key="observation.images.head_color",
            temporal_stride=DEFAULT_AGIBOT_DEXHAND_TEMPORAL_STRIDE,
            **common,
        )

    def _robomind() -> RobomindActionDataset:
        return RobomindActionDataset(
            pairs_jsonl=robomind_pairs_jsonl,
            views=["camera"],
            temporal_stride=DEFAULT_ROBOMIND_TEMPORAL_STRIDE,
            **common,
        )

    def _libero() -> LiberoActionDataset:
        return LiberoActionDataset(
            dataset_dir=libero_dir,
            suites=list(LIBERO_DEFAULT_SUITES),
            **common,
        )

    def _libero_ood() -> LiberoOODDataset:
        return LiberoOODDataset(
            dataset_dir=libero_ood_dir,
            suites=list(LIBERO_DEFAULT_SUITES),
            temporal_stride=LIBERO_OOD_TEMPORAL_STRIDE,
            **common,
        )

    def _ur5_robosuite_ood() -> UR5RobosuiteOOD:
        return UR5RobosuiteOOD(
            dataset_dir=ur5_robosuite_ood_dir,
            temporal_stride=ROBOSUITE_OOD_TEMPORAL_STRIDE,
            **common,
        )

    def _iiwa_robosuite_ood() -> IIWARobosuiteOOD:
        return IIWARobosuiteOOD(
            dataset_dir=iiwa_robosuite_ood_dir,
            temporal_stride=ROBOSUITE_OOD_TEMPORAL_STRIDE,
            **common,
        )

    def _maniskill() -> ManiSkillActionDataset:
        return ManiSkillActionDataset(
            dataset_dir=maniskill_dir,
            **common,
        )

    def _oven_platephase() -> OvenPlatephaseActionDataset:
        return OvenPlatephaseActionDataset(
            dataset_dir=oven_platephase_dir,
            temporal_stride=_oven_platephase_temporal_stride(),
            **common,
        )

    def _robotwin() -> RobotwinActionDataset:
        # Policy-rate n800 pairs are already ~10fps; set ROBOTWIN_TEMPORAL_STRIDE=1.
        tw_stride = int(
            os.environ.get("ROBOTWIN_TEMPORAL_STRIDE", DEFAULT_ROBOTWIN_TEMPORAL_STRIDE)
        )
        return RobotwinActionDataset(
            pairs_jsonl=robotwin_pairs_jsonl,
            temporal_stride=max(tw_stride, 1),
            **common,
        )

    def _robocasa() -> RobocasaActionDataset:
        return RobocasaActionDataset(
            dataset_dirs=list(robocasa_dirs),
            views=["left", "right"],
            temporal_stride=DEFAULT_ROBOCASA_TEMPORAL_STRIDE,
            **common,
        )

    # Keep order identical to training MIX_WEIGHTS / DEFAULT_MIX_NAMES.
    builders = {
        ActionDatasetSource.REAL_SOURCE_WORLD: _rsw,
        ActionDatasetSource.AGIBOT: _agibot,
        ActionDatasetSource.AGIBOT_DEXHAND: _agibot_dexhand,
        ActionDatasetSource.ROBOMIND: _robomind,
        ActionDatasetSource.LIBERO: _libero,
        ActionDatasetSource.LIBERO_OOD: _libero_ood,
        ActionDatasetSource.UR5_ROBOSUITE_OOD: _ur5_robosuite_ood,
        ActionDatasetSource.IIWA_ROBOSUITE_OOD: _iiwa_robosuite_ood,
        ActionDatasetSource.MANISKILL: _maniskill,
        ActionDatasetSource.OVEN_PLATEPHASE: _oven_platephase,
        ActionDatasetSource.ROBOTWIN: _robotwin,
        ActionDatasetSource.ROBOCASA: _robocasa,
    }

    if source == ActionDatasetSource.MIXED:
        if len(mix_weights) != len(DEFAULT_MIX_NAMES):
            raise ValueError(
                f"mix_weights length {len(mix_weights)} != {len(DEFAULT_MIX_NAMES)} "
                f"(expected order {list(DEFAULT_MIX_NAMES)})"
            )
        ds = WeightedMultiDataset(
            datasets=[builders[ActionDatasetSource(name)]() for name in DEFAULT_MIX_NAMES],
            weights=list(mix_weights),
            names=list(DEFAULT_MIX_NAMES),
            epoch_size=None,
            seed=0,
            attach_source_name=True,
        )
    elif source in builders:
        ds = builders[source]()
    else:
        raise ValueError(f"Unsupported action dataset source: {source}")

    log.info(f"Built action-inject inference dataset {source.value}: {len(ds)} episodes")
    return ds


def _allocate_counts(weights: Sequence[float], num_samples: int) -> list[int]:
    n_src = len(weights)
    if num_samples <= 0:
        return [0] * n_src
    probs = np.asarray(weights, dtype=np.float64)
    probs = probs / probs.sum()

    if num_samples >= n_src:
        counts = np.ones(n_src, dtype=np.int64)
        remain = num_samples - n_src
        if remain > 0:
            extra = np.floor(remain * probs).astype(np.int64)
            counts += extra
            leftover = remain - int(extra.sum())
            order = np.argsort(-probs)
            for i in range(leftover):
                counts[order[i % n_src]] += 1
    else:
        order = np.argsort(-probs)
        counts = np.zeros(n_src, dtype=np.int64)
        for i in range(num_samples):
            counts[order[i]] = 1
    return counts.tolist()


def _safe_name(source: str, index: int, caption: str, max_caption_chars: int = 40) -> str:
    slug = "".join(c if c.isalnum() or c in "-_" else "_" for c in caption.strip().lower())
    slug = "_".join(part for part in slug.split("_") if part)[:max_caption_chars].strip("_")
    if not slug:
        slug = "sample"
    return f"{source}_{index:06d}_{slug}"


def _episode_from_sample(
    *,
    source: str,
    index: int,
    sample: Any,
) -> ActionInjectEpisodeSpec:
    """Convert a dataset ``.samples[i]`` dataclass into an inference spec."""
    caption = str(getattr(sample, "caption", "") or "").strip()
    rgb_path = str(getattr(sample, "video_path", "") or "")
    action_video_path = str(getattr(sample, "action_video_path", "") or "")
    length = int(getattr(sample, "length", 0) or 0)

    action_path = str(getattr(sample, "action_path", "") or "")
    parquet_path = str(getattr(sample, "parquet_path", "") or "")
    hdf5_path = str(getattr(sample, "hdf5_path", "") or "")

    # RoboMind indexes parquet under ``action_path`` (no separate parquet_path).
    if source == ActionDatasetSource.ROBOMIND.value and action_path and not parquet_path:
        parquet_path = action_path
        action_path = ""
    # AgiBot dexhand sample dataclass uses ``parquet_path``.
    if source == ActionDatasetSource.AGIBOT_DEXHAND.value:
        parquet_path = str(getattr(sample, "parquet_path", "") or parquet_path)

    if hasattr(sample, "num_arms"):
        num_arms = int(sample.num_arms)
    elif source in _BIMANUAL_SOURCES or parquet_path or hdf5_path:
        num_arms = 2
    else:
        num_arms = 1

    # Libero/ManiSkill/RoboCasa .npy is raw (e.g. 8-D); loader unifies to 28-D.
    already_unified = False

    temporal_stride = temporal_stride_for_source(source)
    # ``length`` on the index is native frames; report model-rate length after stride.
    model_length = (
        (length + temporal_stride - 1) // temporal_stride if temporal_stride > 1 else length
    )

    metadata_path = str(getattr(sample, "metadata_path", "") or "")
    if not metadata_path:
        for base in (action_path, rgb_path, action_video_path):
            if not base:
                continue
            cand = os.path.join(os.path.dirname(base), "metadata.json")
            if os.path.isfile(cand):
                metadata_path = cand
                break

    camera_preset = str(getattr(sample, "camera_preset", "") or "")
    if metadata_path and os.path.isfile(metadata_path) and not camera_preset:
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            camera_preset = str((meta or {}).get("camera_preset") or "")
        except Exception:
            camera_preset = camera_preset

    return ActionInjectEpisodeSpec(
        source=source,
        index=int(index),
        name=_safe_name(source, index, caption),
        prompt=caption if caption else source.replace("_", " "),
        rgb_path=rgb_path,
        action_video_path=action_video_path,
        action_rgb_path=str(getattr(sample, "action_rgb_path", "") or ""),
        action_path=action_path,
        parquet_path=parquet_path,
        hdf5_path=hdf5_path,
        num_arms=num_arms,
        length=model_length,
        already_unified=already_unified,
        temporal_stride=temporal_stride,
        task_name=str(getattr(sample, "task_name", "") or ""),
        view_difficulty=str(getattr(sample, "view_difficulty", "") or ""),
        camera_preset=camera_preset,
        metadata_path=metadata_path,
        robot=str(getattr(sample, "robot", "") or ""),
        org_path=str(getattr(sample, "org_path", "") or ""),
        episode_index=getattr(sample, "episode_index", None),
    )


def parse_robotwin_tasks(raw: Union[str, Sequence[str], None]) -> list[str]:
    """Parse comma/whitespace-separated RoboTwin ``task_name`` filters."""
    if raw is None:
        return []
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.replace(";", ",").split(",")]
    else:
        parts = []
        for item in raw:
            parts.extend(str(item).replace(";", ",").split(","))
        parts = [p.strip() for p in parts]
    out: list[str] = []
    seen: set[str] = set()
    for p in parts:
        if not p or p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def list_robotwin_task_names(dataset: Dataset) -> list[str]:
    """Sorted unique ``task_name`` values present in a RoboTwin (or mix) dataset."""
    names: set[str] = set()

    def _collect(ds: Dataset) -> None:
        samples = getattr(ds, "samples", None)
        if samples is None:
            return
        for s in samples:
            tn = str(getattr(s, "task_name", "") or "").strip()
            if tn:
                names.add(tn)

    if isinstance(dataset, WeightedMultiDataset):
        for child in dataset.datasets:
            _collect(child)
    else:
        _collect(dataset)
    return sorted(names)


def _robotwin_dataset_from(dataset: Dataset) -> Dataset:
    """Return the RoboTwin child dataset (or ``dataset`` itself if pure robotwin)."""
    if isinstance(dataset, WeightedMultiDataset):
        for name, child in zip(dataset.names, dataset.datasets):
            if str(name) == ActionDatasetSource.ROBOTWIN.value:
                return child
        raise ValueError(
            "Weighted mix has no 'robotwin' source; "
            "use --dataset robotwin with --robotwin-task"
        )
    samples = getattr(dataset, "samples", None)
    if samples is None:
        raise TypeError(f"Dataset {type(dataset).__name__} has no .samples index")
    # Heuristic: robotwin samples carry hdf5_path + task_name.
    if len(samples) > 0 and not hasattr(samples[0], "task_name"):
        raise TypeError(
            f"Dataset {type(dataset).__name__} samples lack task_name; "
            "expected RobotwinActionDataset"
        )
    return dataset


def select_robotwin_task_episodes(
    dataset: Dataset,
    tasks: Union[str, Sequence[str]],
    *,
    limit: int = 0,
    offset: int = 0,
    source_name: str = "robotwin",
    robot_config_substr: str | None = None,
) -> list[ActionInjectEpisodeSpec]:
    """Deterministically select RoboTwin episodes whose ``task_name`` is in ``tasks``.

    Unlike ``sample_action_episodes``, this does **not** randomly subsample: it walks
    the dataset index in order and returns every match (optionally sliced).

    Args:
        dataset: ``RobotwinActionDataset`` or a mix containing robotwin.
        tasks: one or more JSONL ``task_name`` values (e.g. ``place_object_basket``).
        limit: max episodes to return; ``<=0`` means all matches after ``offset``.
        offset: skip the first ``offset`` matches (for batched expansion runs).
        source_name: written into specs (normally ``robotwin``).
        robot_config_substr: if set, keep only episodes whose JSONL ``robot_config``
            contains this substring (case-insensitive), e.g. ``clean`` or ``randomized``.
    """
    task_list = parse_robotwin_tasks(tasks)
    if not task_list:
        raise ValueError("select_robotwin_task_episodes requires at least one task name")
    if offset < 0:
        raise ValueError(f"offset must be >= 0, got {offset}")

    rt_ds = _robotwin_dataset_from(dataset)
    samples_attr = getattr(rt_ds, "samples", None)
    if samples_attr is None:
        raise TypeError("RoboTwin dataset has no .samples index")

    wanted = set(task_list)
    cfg_key = (robot_config_substr or "").strip().lower()
    matched_idx: list[int] = []
    for i, sample in enumerate(samples_attr):
        tn = str(getattr(sample, "task_name", "") or "").strip()
        if tn not in wanted:
            continue
        if cfg_key:
            rc = str(getattr(sample, "robot_config", "") or "").lower()
            # Fallback: some indexes only expose the path.
            if not rc:
                rc = (
                    str(getattr(sample, "video_path", "") or "")
                    + " "
                    + str(getattr(sample, "uuid", "") or "")
                ).lower()
            if cfg_key not in rc:
                continue
        matched_idx.append(i)

    if not matched_idx:
        available = list_robotwin_task_names(rt_ds)
        preview = ", ".join(available[:20]) + (" ..." if len(available) > 20 else "")
        cfg_msg = f", robot_config_substr={robot_config_substr!r}" if cfg_key else ""
        raise ValueError(
            f"No RoboTwin episodes matched task(s)={task_list}{cfg_msg}. "
            f"Available task_name values ({len(available)}): {preview}"
        )

    sliced = matched_idx[int(offset) :]
    if int(limit) > 0:
        sliced = sliced[: int(limit)]
    if not sliced:
        raise ValueError(
            f"RoboTwin task filter matched {len(matched_idx)} episodes for {task_list}, "
            f"but offset={offset} / limit={limit} left an empty slice"
        )

    specs = [
        _episode_from_sample(
            source=str(source_name),
            index=idx,
            sample=samples_attr[idx],
        )
        for idx in sliced
    ]
    log.info(
        f"Selected {len(specs)}/{len(matched_idx)} RoboTwin episodes for "
        f"task(s)={task_list}"
        + (f", robot_config~={robot_config_substr!r}" if cfg_key else "")
        + f" (offset={offset}, limit={limit if limit > 0 else 'all'})"
    )
    return specs


def sample_action_episodes(
    dataset: Dataset,
    *,
    num_samples: int,
    seed: int = 0,
    source_name: str = "mixed",
) -> list[ActionInjectEpisodeSpec]:
    """Randomly sample episode specs (stratified when ``WeightedMultiDataset``)."""
    if num_samples <= 0:
        raise ValueError("num_samples must be > 0")

    specs: list[ActionInjectEpisodeSpec] = []
    rng = np.random.default_rng(int(seed))

    if isinstance(dataset, WeightedMultiDataset):
        counts = _allocate_counts(dataset.weights, num_samples)
        log.info(
            "Stratified action-mix sample counts: "
            + ", ".join(f"{dataset.names[i]}={counts[i]}" for i in range(dataset.num_sources))
        )
        for src_i, k in enumerate(counts):
            if k <= 0:
                continue
            src_ds = dataset.datasets[src_i]
            src_name = dataset.names[src_i]
            src_len = len(src_ds)
            if src_len <= 0:
                continue
            local_ids = rng.choice(src_len, size=k, replace=(k > src_len))
            samples_attr = getattr(src_ds, "samples", None)
            if samples_attr is None:
                raise TypeError(f"Dataset {src_name} has no .samples index")
            for local_idx in local_ids:
                local_idx = int(local_idx)
                specs.append(
                    _episode_from_sample(
                        source=src_name,
                        index=local_idx,
                        sample=samples_attr[local_idx],
                    )
                )
        rng.shuffle(specs)
    else:
        n = min(int(num_samples), len(dataset))
        indices = rng.choice(len(dataset), size=n, replace=False)
        samples_attr = getattr(dataset, "samples", None)
        if samples_attr is None:
            raise TypeError(f"Dataset {source_name} has no .samples index")
        for idx in sorted(int(i) for i in indices):
            specs.append(
                _episode_from_sample(
                    source=str(source_name),
                    index=idx,
                    sample=samples_attr[idx],
                )
            )

    log.info(f"Sampled {len(specs)} action episodes (seed={seed})")
    return specs


def load_episode_action(
    spec: ActionInjectEpisodeSpec,
    *,
    normalize_gripper: bool = True,
) -> torch.Tensor:
    """Load absolute unified ``[T, 28]`` action for an episode spec.

    Applies ``spec.temporal_stride`` so actions share the same kept indices as
    RGB / action_video (training-aligned per source).

    For RoboTwin, set ``normalize_gripper=False`` to keep HDF5 gripper values
    when exporting LeRobot (inference can keep the default ``True``).
    """
    source = str(spec.source)
    # Public sample packs use a Libero-style local action.npy. Prefer it when
    # the file is already in the unified 28-D layout so parquet/hdf5 are optional.
    if spec.already_unified and spec.action_path:
        action = load_action_npy(
            spec.action_path,
            num_arms=spec.num_arms,
            already_unified=True,
            gripper_raw_max=None,
        )
        stride = max(int(getattr(spec, "temporal_stride", 1) or 1), 1)
        if stride > 1:
            action = action[::stride]
        return action
    if source == ActionDatasetSource.OVEN_PLATEPHASE.value:
        parquet = spec.parquet_path or spec.action_path
        if not parquet:
            raise ValueError(f"Episode {spec.name} (oven_platephase) missing parquet_path")
        from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_oven_platephase_action import (
            load_oven_platephase_action_from_parquet,
        )

        action = torch.from_numpy(
            np.ascontiguousarray(load_oven_platephase_action_from_parquet(parquet))
        ).float()
    elif source == ActionDatasetSource.AGIBOT.value:
        if not spec.parquet_path:
            raise ValueError(f"Episode {spec.name} (agibot) missing parquet_path")
        action = load_action_agibot_parquet(spec.parquet_path)
    elif source == ActionDatasetSource.AGIBOT_DEXHAND.value:
        parquet = spec.parquet_path or spec.action_path
        if not parquet:
            raise ValueError(f"Episode {spec.name} (agibot_dexhand) missing parquet path")
        action = load_action_agibot_dexhand_parquet(parquet)
    elif source == ActionDatasetSource.ROBOMIND.value:
        parquet = spec.parquet_path or spec.action_path
        if not parquet:
            raise ValueError(f"Episode {spec.name} (robomind) missing parquet path")
        action = load_action_robomind_parquet(parquet)
    elif source == ActionDatasetSource.ROBOTWIN.value or spec.hdf5_path:
        if not spec.hdf5_path:
            raise ValueError(f"Episode {spec.name} (robotwin) missing hdf5_path")
        action = load_action_robotwin_hdf5(
            spec.hdf5_path,
            include_gripper=True,
            normalize_gripper=bool(normalize_gripper),
        )
    elif source == ActionDatasetSource.REAL_SOURCE_WORLD.value or (
        spec.parquet_path
        and not spec.hdf5_path
        and not spec.action_path
        and source not in (ActionDatasetSource.ROBOMIND.value,)
    ):
        if not spec.parquet_path:
            raise ValueError(f"Episode {spec.name} (rsw) missing parquet_path")
        action = load_action_rsw_parquet(
            spec.parquet_path, include_gripper=True, include_lift=False
        )
    elif spec.action_path:
        # Libero / ManiSkill / RoboCasa / Robosuite-OOD .npy → unified 28-D.
        from cosmos_predict2._src.predict2.datasets.local_datasets.action_unified_layout import (
            SIM_PANDA_GRIPPER_RAW_MAX,
        )

        panda_sources = {
            ActionDatasetSource.LIBERO.value,
            ActionDatasetSource.LIBERO_OOD.value,
            ActionDatasetSource.MANISKILL.value,
            ActionDatasetSource.ROBOCASA.value,
        }
        if source == ActionDatasetSource.UR5_ROBOSUITE_OOD.value:
            grip_max: Optional[float] = ROBOTIQ_85_GRIPPER_RAW_MAX
        elif source == ActionDatasetSource.IIWA_ROBOSUITE_OOD.value:
            grip_max = ROBOTIQ_140_GRIPPER_RAW_MAX
        elif source in panda_sources:
            grip_max = SIM_PANDA_GRIPPER_RAW_MAX
        else:
            grip_max = None
        action = load_action_npy(
            spec.action_path,
            num_arms=spec.num_arms,
            already_unified=True if spec.already_unified else None,
            gripper_raw_max=grip_max,
        )
    else:
        raise ValueError(
            f"Episode {spec.name} has no usable action source "
            f"(parquet={spec.parquet_path!r}, hdf5={spec.hdf5_path!r}, "
            f"action_path={spec.action_path!r})"
        )

    stride = max(int(getattr(spec, "temporal_stride", 1) or 1), 1)
    if stride > 1:
        action = action[::stride]
    return action


def pingpong_concat_time(x: torch.Tensor, time_dim: int = 0) -> torch.Tensor:
    """Forward + reverse-without-last-frame along ``time_dim``.

    ``[0, 1, ..., T-1]`` → ``[0, 1, ..., T-1, T-2, ..., 0]`` (length ``2T-1``).
    The junction is one original timestep, so the trajectory stays continuous.
    """
    if not torch.is_tensor(x):
        raise TypeError(f"pingpong_concat_time expects a tensor, got {type(x)}")
    t = int(x.shape[time_dim])
    if t < 2:
        return x
    rev = x.narrow(time_dim, 0, t - 1).flip(time_dim)
    return torch.cat([x, rev], dim=time_dim)


# RoboTwin Aloha AgileX ``joint_action/vector`` layout (source format, not unified 28-D).
ROBOTWIN_SOURCE_ACTION_DIM = 14
ROBOTWIN_SOURCE_ACTION_NAMES: tuple[str, ...] = (
    "left_arm_joint_1",
    "left_arm_joint_2",
    "left_arm_joint_3",
    "left_arm_joint_4",
    "left_arm_joint_5",
    "left_arm_joint_6",
    "left_gripper",
    "right_arm_joint_1",
    "right_arm_joint_2",
    "right_arm_joint_3",
    "right_arm_joint_4",
    "right_arm_joint_5",
    "right_arm_joint_6",
    "right_gripper",
)
ROBOTWIN_HDF5_ACTION_KEY = "joint_action/vector"

AGIBOT_SOURCE_ACTION_NAMES: tuple[str, ...] = tuple(
    [f"action.joint.position.{i}" for i in range(14)]
    + ["action.effector.position.0", "action.effector.position.1"]
)
AGIBOT_DEXHAND_SOURCE_ACTION_NAMES: tuple[str, ...] = tuple(
    [f"action.joint.position.{i}" for i in range(14)]
    + [f"action.effector.position.{i}" for i in range(12)]
)
ROBOMIND_SOURCE_ACTION_NAMES: tuple[str, ...] = (
    *(f"arm_joint_{i}" for i in range(14)),
    "hand_joints_left",
    "hand_joints_right",
)


def _stride_select(raw: np.ndarray, stride: int) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 2:
        raise ValueError(f"expected action [T,D], got {raw.shape}")
    stride = max(int(stride), 1)
    native_idx = np.arange(0, int(raw.shape[0]), stride, dtype=np.int64)
    return raw[native_idx], native_idx


def load_episode_source_action_for_export(
    spec: ActionInjectEpisodeSpec,
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    """Load **source-format** actions for LeRobot export (not unified 28-D).

    Only applies ``temporal_stride`` frame selection (e.g. indices ``0,2,4,...``
    when stride=2). Values are left as stored in the original dataset — the
    unified / normalized 28-D tensor is for train/infer only.

    Returns:
        action: ``[T, D_source]`` float32
        names: length-``D_source`` feature names
        native_frame_indices: int64 indices into the native (pre-stride) timeline
    """
    import h5py
    import pyarrow.parquet as pq

    from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_agibot_action import (
        _EFFECTOR_COLS as _AGIBOT_EFFECTOR_COLS,
        _JOINT_COLS as _AGIBOT_JOINT_COLS,
    )
    from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_real_source_world_action import (
        DEFAULT_ACTION_COLUMN,
        _list_column_to_numpy,
        resolve_action_indices,
    )
    from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_robomind_action import (
        SOURCE_JOINT_DIM as ROBOMIND_JOINT_DIM,
        _list_col_to_matrix,
    )

    stride = max(int(getattr(spec, "temporal_stride", 1) or 1), 1)
    source = str(spec.source)

    # --- RoboTwin: HDF5 joint_action/vector [T,14] ---
    if source == ActionDatasetSource.ROBOTWIN.value or (
        bool(spec.hdf5_path)
        and source
        not in {
            ActionDatasetSource.REAL_SOURCE_WORLD.value,
            ActionDatasetSource.AGIBOT.value,
            ActionDatasetSource.AGIBOT_DEXHAND.value,
            ActionDatasetSource.ROBOMIND.value,
            ActionDatasetSource.LIBERO.value,
            ActionDatasetSource.LIBERO_OOD.value,
            ActionDatasetSource.UR5_ROBOSUITE_OOD.value,
            ActionDatasetSource.IIWA_ROBOSUITE_OOD.value,
            ActionDatasetSource.MANISKILL.value,
            ActionDatasetSource.ROBOCASA.value,
        }
    ):
        if not spec.hdf5_path:
            raise ValueError(f"Episode {spec.name} (robotwin) missing hdf5_path")
        with h5py.File(spec.hdf5_path, "r") as h:
            if ROBOTWIN_HDF5_ACTION_KEY not in h:
                raise KeyError(
                    f"Missing {ROBOTWIN_HDF5_ACTION_KEY!r} in {spec.hdf5_path}"
                )
            raw = np.asarray(h[ROBOTWIN_HDF5_ACTION_KEY][:], dtype=np.float32)
        if raw.ndim != 2 or raw.shape[1] != ROBOTWIN_SOURCE_ACTION_DIM:
            raise ValueError(
                f"{spec.name}: expected RoboTwin action [T,{ROBOTWIN_SOURCE_ACTION_DIM}], "
                f"got {raw.shape} from {spec.hdf5_path}"
            )
        selected, native_idx = _stride_select(raw, stride)
        return selected, ROBOTWIN_SOURCE_ACTION_NAMES, native_idx

    # --- RealSource-World: parquet observation.state pose slice (16 w/o lift) ---
    if source == ActionDatasetSource.REAL_SOURCE_WORLD.value:
        if not spec.parquet_path:
            raise ValueError(f"Episode {spec.name} (rsw) missing parquet_path")
        indices, names = resolve_action_indices(include_gripper=True, include_lift=False)
        table = pq.read_table(spec.parquet_path, columns=[DEFAULT_ACTION_COLUMN])
        full = _list_column_to_numpy(table.column(DEFAULT_ACTION_COLUMN))
        if full.shape[1] <= max(indices):
            raise ValueError(
                f"{spec.name}: observation.state dim {full.shape[1]} too small for indices {indices}"
            )
        selected, native_idx = _stride_select(full[:, indices], stride)
        return selected, names, native_idx

    # --- AgiBot gripper: joint[14] + effector[2] ---
    if source == ActionDatasetSource.AGIBOT.value:
        if not spec.parquet_path:
            raise ValueError(f"Episode {spec.name} (agibot) missing parquet_path")
        cols = list(_AGIBOT_JOINT_COLS) + list(_AGIBOT_EFFECTOR_COLS)
        table = pq.read_table(spec.parquet_path, columns=cols)
        raw = np.column_stack(
            [table.column(c).to_numpy(zero_copy_only=False) for c in cols]
        ).astype(np.float32, copy=False)
        selected, native_idx = _stride_select(raw, stride)
        return selected, AGIBOT_SOURCE_ACTION_NAMES, native_idx

    # --- AgiBot dexhand: joint[14] + effector[12] ---
    if source == ActionDatasetSource.AGIBOT_DEXHAND.value:
        parquet = spec.parquet_path or spec.action_path
        if not parquet:
            raise ValueError(f"Episode {spec.name} (agibot_dexhand) missing parquet")
        schema_names = set(pq.read_schema(parquet).names)
        joint_cols = [c for c in _AGIBOT_JOINT_COLS if c in schema_names]
        effector_cols = []
        i = 0
        while f"action.effector.position.{i}" in schema_names:
            effector_cols.append(f"action.effector.position.{i}")
            i += 1
        effector_cols = effector_cols[:12]
        if len(joint_cols) < 14 or len(effector_cols) < 12:
            raise KeyError(
                f"{spec.name}: dexhand parquet missing joint/effector cols "
                f"(joint={len(joint_cols)}, effector={len(effector_cols)})"
            )
        table = pq.read_table(parquet, columns=joint_cols + effector_cols)
        raw = np.column_stack(
            [table.column(c).to_numpy(zero_copy_only=False) for c in joint_cols + effector_cols]
        ).astype(np.float32, copy=False)
        selected, native_idx = _stride_select(raw, stride)
        return selected, AGIBOT_DEXHAND_SOURCE_ACTION_NAMES, native_idx

    # --- RoboMind: arm[14] + left/right gripper ---
    if source == ActionDatasetSource.ROBOMIND.value:
        parquet = spec.parquet_path or spec.action_path
        if not parquet:
            raise ValueError(f"Episode {spec.name} (robomind) missing parquet")
        table = pq.read_table(parquet)
        names = set(table.column_names)
        if "control_arm_joints_robot" in names:
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
            raise KeyError(f"{spec.name}: missing RoboMind arm columns in {parquet}")
        arms = _list_col_to_matrix(
            table.column(arm_col).to_pylist(), expect_width=ROBOMIND_JOINT_DIM
        )
        left_g = _list_col_to_matrix(
            table.column(left_g_col).to_pylist(), expect_width=1
        ).reshape(-1, 1)
        right_g = _list_col_to_matrix(
            table.column(right_g_col).to_pylist(), expect_width=1
        ).reshape(-1, 1)
        raw = np.concatenate([arms[:, :ROBOMIND_JOINT_DIM], left_g, right_g], axis=-1)
        selected, native_idx = _stride_select(raw, stride)
        return selected, ROBOMIND_SOURCE_ACTION_NAMES, native_idx

    # --- Libero / ManiSkill / RoboCasa: native action.npy ---
    if spec.action_path and Path(spec.action_path).is_file():
        raw = np.asarray(np.load(spec.action_path), dtype=np.float32)
        if raw.ndim == 1:
            raw = raw[None, :]
        selected, native_idx = _stride_select(raw, stride)
        names = tuple(f"action_{i}" for i in range(int(selected.shape[1])))
        return selected, names, native_idx

    raise ValueError(
        f"{spec.name}: no source-native action loader for source={source!r} "
        f"(parquet={spec.parquet_path!r}, hdf5={spec.hdf5_path!r}, "
        f"action_path={spec.action_path!r}). Unified 28-D export is disabled."
    )


def write_episode_manifest(specs: Sequence[ActionInjectEpisodeSpec], path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for spec in specs:
            row = asdict(spec)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log.info(f"Wrote episode manifest: {path} ({len(specs)} rows)")


def read_episode_manifest(path: Union[str, Path]) -> list[ActionInjectEpisodeSpec]:
    path = Path(path)
    known = {f.name for f in fields(ActionInjectEpisodeSpec)}
    specs: list[ActionInjectEpisodeSpec] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            specs.append(
                ActionInjectEpisodeSpec(**{k: v for k, v in row.items() if k in known})
            )
    return specs


def load_sample_dir(sample_dir: Union[str, Path]) -> ActionInjectEpisodeSpec:
    """Load a public example folder (rgb + action + prompt + meta).

    Expected layout::

        sample_dir/
          meta.json          # source, prompt, fps, camera, paths…
          prompt.txt         # optional if meta.prompt is set
          action.npy         # robot-native or unified 28-D
          first_frame.png    # RGB condition (first frame)
          rgb.mp4            # original RGB (comparison only)
          kinematic_qpos.npy # optional Libero exact finger joints

    A RoboDojo multi-view folder (``meta.layout == "mv_fill"``) lists one RGB mp4 per camera
    in ``meta.views``; the condition frame and the ground truth are packed from all three.
    """
    root = Path(sample_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"sample dir not found: {root}")

    meta: dict[str, Any] = {}
    for name in ("meta.json", "metadata.json"):
        p = root / name
        if p.is_file():
            meta = json.loads(p.read_text(encoding="utf-8"))
            break
    cam_p = root / "camera.json"
    if cam_p.is_file():
        cam = json.loads(cam_p.read_text(encoding="utf-8"))
        if isinstance(cam, dict):
            meta = {**cam, **meta}

    prompt = str(meta.get("prompt") or meta.get("lang") or "").strip()
    prompt_txt = root / "prompt.txt"
    if prompt_txt.is_file():
        prompt = prompt_txt.read_text(encoding="utf-8").strip() or prompt

    def _pick(*names: str) -> str:
        for name in names:
            p = root / name
            if p.is_file():
                return str(p)
            rel = str(meta.get(name) or "").strip()
            if rel:
                cand = Path(rel) if os.path.isabs(rel) else root / rel
                if cand.is_file():
                    return str(cand)
        return ""

    rgb_path = _pick(
        "first_frame.png",
        "first_frame.jpg",
        "rgb.png",
        "rgb.mp4",
        "rgb",
        "rgb_path",
        "video_path",
    )
    if not rgb_path:
        raise FileNotFoundError(f"{root}: need rgb.mp4 or first_frame.png")
    gt_rgb_path = _pick("rgb.mp4", "rgb_video", "gt_rgb_path")
    action_path = _pick("action.npy", "action", "action_path")
    parquet_path = _pick("action.parquet", "episode.parquet", "parquet_path")
    hdf5_path = _pick("action.hdf5", "episode.hdf5", "hdf5_path")
    if not action_path and not parquet_path and not hdf5_path:
        raise FileNotFoundError(f"{root}: need action.npy, a parquet, or an hdf5")

    source = str(meta.get("source") or "libero").strip().lower()
    if source == "uniembodied":
        source = "robodojo"
    metadata_path = str(root / "meta.json") if (root / "meta.json").is_file() else ""
    if not metadata_path and (root / "metadata.json").is_file():
        metadata_path = str(root / "metadata.json")

    length = int(meta.get("frames") or meta.get("length") or 0)
    if length <= 0 and action_path:
        try:
            length = int(np.load(action_path, mmap_mode="r").shape[0])
        except Exception:
            length = 0

    already_unified = bool(meta.get("already_unified", False))
    num_arms = int(meta.get("num_arms") or (2 if source in _BIMANUAL_SOURCES else 1))
    stride = int(meta.get("temporal_stride") or temporal_stride_for_source(source))
    name = str(meta.get("name") or root.name)

    layout = str(meta.get("layout") or "")
    views: dict[str, dict[str, str]] = {}
    action_video_path = _pick("action_video.mp4", "action_video", "action_video_path")
    if layout:
        from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_robodojo_action import (
            MULTIVIEW_LAYOUT,
            MULTIVIEW_VIEWS,
        )

        if layout != MULTIVIEW_LAYOUT:
            raise ValueError(f"{root}: unknown layout {layout!r} (supported: {MULTIVIEW_LAYOUT!r})")
        for view in MULTIVIEW_VIEWS:
            entry = dict((meta.get("views") or {}).get(view) or {})
            rgb = root / str(entry.get("rgb") or "")
            if not entry.get("rgb") or not rgb.is_file():
                raise FileNotFoundError(f"{root}: multi-view sample needs meta.views.{view}.rgb")
            av = root / str(entry.get("action_video") or "")
            views[view] = {"rgb": str(rgb), "action_video": str(av) if entry.get("action_video") and av.is_file() else ""}
        rgb_path = gt_rgb_path = views[MULTIVIEW_VIEWS[0]]["rgb"]
        action_video_path = views[MULTIVIEW_VIEWS[0]]["action_video"]

    return ActionInjectEpisodeSpec(
        source=source,
        index=int(meta.get("index") or 0),
        name=name,
        prompt=prompt or source.replace("_", " "),
        rgb_path=rgb_path,
        gt_rgb_path=gt_rgb_path,
        action_video_path=action_video_path,
        action_path=action_path,
        parquet_path=parquet_path,
        hdf5_path=hdf5_path,
        num_arms=num_arms,
        length=length,
        already_unified=already_unified,
        temporal_stride=max(stride, 1),
        task_name=str(meta.get("task_name") or ""),
        view_difficulty=str(meta.get("view_difficulty") or ""),
        camera_preset=str(meta.get("camera_preset") or ""),
        metadata_path=metadata_path,
        robot=str(meta.get("robot") or ""),
        org_path=str(meta.get("org_path") or ""),
        episode_index=meta.get("episode_index"),
        fps=float(meta.get("model_fps") or 0.0),
        layout=layout,
        views=views,
    )
