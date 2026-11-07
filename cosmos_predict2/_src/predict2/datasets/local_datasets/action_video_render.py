# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Online action-video API used by inference and dataset classes.

``render_action_video_thwc`` always returns a **white-background** RGB
``[T, H, W, 3]`` uint8 tensor (model condition). Black-bg is applied only
when saving visualization videos.

Each source is rendered by the corresponding ``*ActionDataset`` class
(``LiberoActionDataset.render_action_video``, …). URDF / batch helpers live
in ``action_video_backends/`` next to this file.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

import numpy as np

BACKEND_ROOT = Path(__file__).resolve().parent / "action_video_backends"

# Official training videos were written at these sizes, then resized by the
# dataset loader. Render here first, then scale to the model resolution.
ROBOMIND_NATIVE_HW = (360, 640)  # (H, W)

SUPPORTED_RENDER_SOURCES = (
    "real_source_world",
    "agibot",
    "agibot_dexhand",
    "robomind",
    "libero",
    "libero_ood",
    "maniskill",
    "robotwin",
    "robocasa",
    "robodojo",
)


@dataclass
class RenderRequest:
    source: str
    resolution: Tuple[int, int] = (480, 640)  # (H, W)
    rgb_path: str = ""
    action_path: str = ""
    parquet_path: str = ""
    hdf5_path: str = ""
    metadata_path: str = ""
    camera_preset: str = ""
    task_name: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def height(self) -> int:
        return int(self.resolution[0])

    @property
    def width(self) -> int:
        return int(self.resolution[1])


def ensure_backend_import_path(*extra: Path) -> None:
    for p in (BACKEND_ROOT, *extra):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


def load_backend_module(subdir: str, filename: str, module_name: str):
    """Load a backend file under a unique module name (avoid sys.modules clashes)."""
    import importlib.util

    path = BACKEND_ROOT / subdir / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load backend module {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_render_meta(request: RenderRequest) -> Dict[str, Any]:
    path = request.metadata_path
    if not path:
        for raw in (request.action_path, request.parquet_path, request.rgb_path):
            if not raw:
                continue
            for name in ("meta.json", "metadata.json"):
                cand = Path(raw).parent / name
                if cand.is_file():
                    path = str(cand)
                    break
            if path:
                break
    if path and os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    return {}


def merge_render_extra(request: RenderRequest, meta: Dict[str, Any]) -> Dict[str, Any]:
    extra = dict(request.extra or {})
    for key in (
        "org_path",
        "robot",
        "camera_params",
        "camera_entry",
        "episode_index",
        "task_name",
        "h5_path",
        "bddl_path",
        "traj_id",
        "demo_id",
        "item",
        "hdf5_root",
        "camera_name",
        "view",
        "env_meta",
        "cam_axis_mode",
        "cam_pitch_offset",
        "cam_yaw_offset",
        "left_cam_pitch_offset",
        "right_cam_pitch_offset",
        "left_cam_yaw_offset",
        "right_cam_yaw_offset",
        "arm_l_base_y",
        "arm_r_base_y",
        "head_pos",
        "waist_pos",
        "head",
        "waist",
        "intrinsic",
        "extrinsic",
        "env_id",
        "env_kwargs",
        "reset_kwargs",
        "base_pos",
        "base_quat_wxyz",
        "camera_eye",
        "camera_target",
        "K",
        "E",
        "E_world2cam",
        "E_cam2world",
        "views",
        "suite",
    ):
        if key not in extra and meta.get(key) not in (None, ""):
            extra[key] = meta[key]
    return extra


def resize_thwc(frames: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize ``[T,H,W,3]`` the same way inference loads RGB / official mp4.

    Matches ``resize_input`` in ``action_inject_pipeline``: scale so the
    shorter side covers the target, then center-crop. Do **not** stretch
    512×512 Libero frames into 480×640 — that misaligns the skeleton with RGB.
    """
    import math

    import cv2

    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"expected [T,H,W,3], got {frames.shape}")
    src_h, src_w = int(frames.shape[1]), int(frames.shape[2])
    if src_h == height and src_w == width:
        return frames
    scale = max(width / float(src_w), height / float(src_h))
    mid_h = int(math.ceil(scale * src_h))
    mid_w = int(math.ceil(scale * src_w))
    y0 = max((mid_h - height) // 2, 0)
    x0 = max((mid_w - width) // 2, 0)
    out = np.empty((frames.shape[0], height, width, 3), dtype=np.uint8)
    for i, fr in enumerate(frames):
        resized = cv2.resize(fr, (mid_w, mid_h), interpolation=cv2.INTER_LINEAR)
        crop = resized[y0 : y0 + height, x0 : x0 + width]
        if crop.shape[0] != height or crop.shape[1] != width:
            crop = cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR)
        out[i] = crop
    return out


def bgr_to_rgb(frames: np.ndarray) -> np.ndarray:
    import cv2

    return np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames], axis=0)


def collect_frames(iterator: Iterator[np.ndarray]) -> np.ndarray:
    frames = [np.ascontiguousarray(fr) for fr in iterator]
    if not frames:
        raise RuntimeError("renderer produced 0 frames")
    return np.stack(frames, axis=0)


def source_native_hw(
    source: str, meta: Dict[str, Any], request: RenderRequest
) -> Tuple[int, int]:
    extra = request.extra or {}
    if extra.get("native_height") and extra.get("native_width"):
        return int(extra["native_height"]), int(extra["native_width"])
    if source == "robomind":
        return ROBOMIND_NATIVE_HW
    if source in ("agibot", "agibot_dexhand"):
        return 480, 640
    if source in ("real_source_world", "realsource", "rsw"):
        return 720, 1280
    if source in ("libero", "libero_ood", "maniskill", "robocasa", "robotwin", "robodojo"):
        mh = meta.get("height")
        mw = meta.get("width")
        if mh and mw:
            return int(mh), int(mw)
    return request.height, request.width


def normalize_render_source(source: str) -> str:
    """Map a sample ``source`` onto a renderer. ``uniembodied`` is the previous name of ``robodojo``."""
    source = str(source or "").strip().lower()
    return "robodojo" if source == "uniembodied" else source


def prepare_render_request(request: RenderRequest) -> Tuple[RenderRequest, Dict[str, Any], int, int]:
    """Merge metadata and switch to the official native render size."""
    source = normalize_render_source(request.source)
    meta = load_render_meta(request)
    if request.camera_preset:
        meta.setdefault("camera_preset", request.camera_preset)
    extra = merge_render_extra(request, meta)
    request.extra = extra
    if extra.get("task_name") and not request.task_name:
        request.task_name = str(extra["task_name"])
    target_h, target_w = request.height, request.width
    native_h, native_w = source_native_hw(source, meta, request)
    if (native_h, native_w) != (target_h, target_w):
        request = replace(request, resolution=(native_h, native_w), extra=extra)
    return request, meta, target_h, target_w


def render_panda_fallback(request: RenderRequest, meta: Dict[str, Any], source: str) -> np.ndarray:
    from cosmos_predict2._src.predict2.datasets.local_datasets.action_video_panda import (
        load_action_npy,
        render_panda_action_video,
    )

    if not request.action_path or not os.path.isfile(request.action_path):
        raise FileNotFoundError(f"{source} action.npy missing: {request.action_path}")
    action = load_action_npy(request.action_path)
    max_frames = int(meta.get("max_frames") or (request.extra or {}).get("max_frames") or 0)
    if max_frames > 0:
        action = action[:max_frames]
    return render_panda_action_video(
        action,
        height=request.height,
        width=request.width,
        meta=meta,
        source=source,
        camera_preset=request.camera_preset,
    )


def _renderer_for(source: str) -> Callable[[RenderRequest], np.ndarray]:
    if source in ("libero", "libero_ood"):
        from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_libero_action import (
            LiberoActionDataset,
        )

        return LiberoActionDataset.render_action_video
    if source == "maniskill":
        from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_maniskill_action import (
            ManiSkillActionDataset,
        )

        return ManiSkillActionDataset.render_action_video
    if source == "robocasa":
        from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_robocasa_action import (
            RobocasaActionDataset,
        )

        return RobocasaActionDataset.render_action_video
    if source == "robomind":
        from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_robomind_action import (
            RobomindActionDataset,
        )

        return RobomindActionDataset.render_action_video
    if source in ("real_source_world", "realsource", "rsw"):
        from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_real_source_world_action import (
            RealSourceWorldActionDataset,
        )

        return RealSourceWorldActionDataset.render_action_video
    if source == "agibot":
        from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_agibot_action import (
            AgibotActionDataset,
        )

        return AgibotActionDataset.render_action_video
    if source == "agibot_dexhand":
        from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_agibot_dexhand_action import (
            AgibotDexhandActionDataset,
        )

        return AgibotDexhandActionDataset.render_action_video
    if source == "robotwin":
        from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_robotwin_action import (
            RobotwinActionDataset,
        )

        return RobotwinActionDataset.render_action_video
    if source == "robodojo":
        from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_robodojo_action import (
            RoboDojoActionDataset,
        )

        return RoboDojoActionDataset.render_action_video
    raise ValueError(
        f"No online action-video renderer for source={source!r}. "
        f"Supported: {', '.join(SUPPORTED_RENDER_SOURCES)}."
    )


def render_action_video_thwc(request: RenderRequest) -> np.ndarray:
    """Render a white-background action skeleton video as RGB ``[T,H,W,3]`` uint8."""
    os.environ.pop("AC_BLACK_BACKGROUND", None)
    source = normalize_render_source(request.source)
    request, _meta, target_h, target_w = prepare_render_request(request)
    frames = np.asarray(_renderer_for(source)(request))
    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)
    return resize_thwc(frames, target_h, target_w)


def render_action_video_cthw(request: RenderRequest):
    """Same as ``render_action_video_thwc`` but ``[C,T,H,W]`` torch uint8."""
    import torch

    thwc = render_action_video_thwc(request)
    return torch.from_numpy(np.ascontiguousarray(thwc)).permute(3, 0, 1, 2).contiguous()


def render_action_video_for_dataset_sample(
    *,
    source: str,
    resolution: tuple[int, int],
    rgb_path: str = "",
    action_path: str = "",
    parquet_path: str = "",
    hdf5_path: str = "",
    metadata_path: str = "",
    camera_preset: str = "",
    task_name: str = "",
    extra: Optional[dict[str, Any]] = None,
) -> np.ndarray:
    """Return RGB ``[T, H, W, 3]`` uint8 for one dataset episode."""
    return render_action_video_thwc(
        RenderRequest(
            source=source,
            resolution=resolution,
            rgb_path=rgb_path,
            action_path=action_path,
            parquet_path=parquet_path,
            hdf5_path=hdf5_path,
            metadata_path=metadata_path,
            camera_preset=camera_preset,
            task_name=task_name,
            extra=dict(extra or {}),
        )
    )
