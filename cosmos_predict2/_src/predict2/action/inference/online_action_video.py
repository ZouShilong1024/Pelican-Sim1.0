# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve action-video tensors for inference: existing mp4, else online render."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.predict2.action.inference.action_inject_pipeline import (
    load_multiview_video,
    load_rgb_video,
)
from cosmos_predict2._src.predict2.datasets.local_datasets.action_video_render import (
    RenderRequest,
    render_action_video_thwc,
)


def spec_to_render_request(
    spec: Any,
    *,
    resolution: tuple[int, int],
) -> RenderRequest:
    extra: dict[str, Any] = {}
    view = str(getattr(spec, "view", "") or "")
    if view:
        extra["view"] = view
    episode_index = getattr(spec, "episode_index", None)
    if episode_index is not None:
        extra["episode_index"] = episode_index
    for key in ("org_path", "robot", "camera_params", "camera_entry", "item"):
        val = getattr(spec, key, None)
        if val not in (None, "", {}):
            extra[key] = val
    return RenderRequest(
        source=str(getattr(spec, "source", "") or ""),
        resolution=tuple(int(x) for x in resolution),
        rgb_path=str(getattr(spec, "rgb_path", "") or ""),
        action_path=str(getattr(spec, "action_path", "") or ""),
        parquet_path=str(getattr(spec, "parquet_path", "") or ""),
        hdf5_path=str(getattr(spec, "hdf5_path", "") or ""),
        metadata_path=str(getattr(spec, "metadata_path", "") or ""),
        camera_preset=str(getattr(spec, "camera_preset", "") or ""),
        task_name=str(getattr(spec, "task_name", "") or ""),
        extra=extra,
    )


def render_action_video_for_spec(
    spec: Any,
    *,
    resolution: tuple[int, int],
) -> torch.Tensor:
    """Online-render action video → ``[C, T, H, W]`` uint8."""
    request = spec_to_render_request(spec, resolution=resolution)
    log.info(
        f"Online-rendering action video  source={request.source}  "
        f"res={request.height}x{request.width}"
    )
    thwc = render_action_video_thwc(request)
    return torch.from_numpy(np.ascontiguousarray(thwc)).permute(3, 0, 1, 2).contiguous()


def resolve_action_video_cthw(
    spec: Any,
    *,
    resolution: tuple[int, int],
    temporal_stride: int = 1,
    render_mode: str = "auto",
) -> Optional[torch.Tensor]:
    """Load or render the action-video tensor for one episode.

    ``render_mode``:
      * ``auto`` — use an existing mp4 when present, otherwise render online
      * ``always`` — always render online
      * ``never`` — only load an existing mp4 (None if missing)
    """
    mode = str(render_mode or "auto").strip().lower()
    path = str(getattr(spec, "action_video_path", "") or "")
    # Multi-view: one mp4 per camera, packed onto the canvas like the RGB.
    view_paths = [str(v.get("action_video") or "") for v in (getattr(spec, "views", None) or {}).values()]
    paths = view_paths or [path]
    have_mp4 = all(p and Path(p).is_file() for p in paths)

    def load_existing() -> torch.Tensor:
        stride = max(int(temporal_stride), 1)
        if view_paths:
            return load_multiview_video(view_paths, resolution=resolution, temporal_stride=stride)
        return load_rgb_video(path, num_frames=None, resolution=resolution, start_frame=0, temporal_stride=stride)

    if mode == "never":
        if not have_mp4:
            return None
        return load_existing()

    if mode == "auto" and have_mp4:
        log.info(f"Using existing action-video mp4  {', '.join(paths)}")
        return load_existing()

    try:
        tensor = render_action_video_for_spec(spec, resolution=resolution)
    except Exception as err:
        if have_mp4:
            log.warning(
                f"Online action-video render failed ({err}); "
                f"falling back to {', '.join(paths)}"
            )
            return load_existing()
        raise
    stride = max(int(temporal_stride), 1)
    if stride > 1 and tensor.shape[1] > 0:
        tensor = tensor[:, ::stride]
    return tensor
