# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Action-inject Video2World inference (value + optional VACE action video).

Loads ``ac_inject_rectified_flow_2b`` checkpoints (DCP iter dirs) and generates
videos conditioned on unified 28-D actions. Supports:

  - single-chunk generation (native ``state_t`` / 93 frames)
  - autoregressive (AR) multi-chunk rollout with per-chunk action slicing
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torchvision
from megatron.core import parallel_state
from PIL import Image, ImageDraw, ImageFont

from cosmos_predict2._src.imaginaire.utils import distributed, log
from cosmos_predict2._src.imaginaire.utils.easy_io import easy_io
from cosmos_predict2._src.predict2.datasets.local_datasets.action_unified_layout import (
    UNIFIED_ACTION_DIM,
    to_unified_bimanual_action,
)
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_real_source_world_action import (
    build_action_delta_scaler,
    normalize_action_delta_scaler,
)
from cosmos_predict2._src.predict2.utils.model_loader import load_model_from_checkpoint

_DEFAULT_NEGATIVE_PROMPT = (
    "The video captures a series of frames showing ugly scenes, static with no motion, "
    "motion blur, over-saturation, shaky footage, low resolution, grainy texture, "
    "pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color "
    "balance, washed out colors, choppy sequences, jerky movements, low frame rate, "
    "artifacting, color banding, unnatural transitions, outdated special effects, fake "
    "elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, "
    "and flickering. Overall, the video is of poor quality."
)


def resize_input(video: torch.Tensor, resolution: list[int]) -> torch.Tensor:
    """Center-crop resize ``video`` [T,C,H,W] to ``resolution`` [H,W]."""
    orig_h, orig_w = video.shape[2], video.shape[3]
    target_h, target_w = resolution
    scaling_ratio = max((target_w / orig_w), (target_h / orig_h))
    resizing_shape = (int(math.ceil(scaling_ratio * orig_h)), int(math.ceil(scaling_ratio * orig_w)))
    video_resized = torchvision.transforms.functional.resize(video, resizing_shape)
    return torchvision.transforms.functional.center_crop(video_resized, resolution)

# Released Pelican-Sim 1.0 checkpoint: MoE shared-dense E8/K2.
DEFAULT_EXPERIMENT = "ac_inject_moe_shared_dense_e8_rectified_flow_2b"
DEFAULT_CONFIG_FILE = "cosmos_predict2/_src/predict2/action/configs/action_inject/config.py"
DEFAULT_JOB_DIR = "checkpoints/Pelican-Sim-1.0"
DEFAULT_CHUNK_SIZE = 93
DEFAULT_RESOLUTION = (480, 640)  # (H, W)


def resolve_experiment_and_config(
    job_dir: str | Path,
    *,
    experiment: Optional[str] = None,
    config_file: Optional[str] = None,
) -> tuple[str, str]:
    del job_dir
    return experiment or DEFAULT_EXPERIMENT, config_file or DEFAULT_CONFIG_FILE


def resolve_checkpoint_dir(
    job_dir: str | Path,
    iteration: Optional[str] = None,
    *,
    convert_if_needed: bool = True,
) -> str:
    """Resolve a loadable local checkpoint under a training job directory.

    Prefers consolidated ``model_ema_bf16.pt`` (required by the local
    ``load_model_from_checkpoint`` path). If only a DCP ``model/`` dir exists and
    ``convert_if_needed`` is True, runs ``scripts/convert_distcp_to_pt.py``.
    """
    job_dir = Path(job_dir)
    # Allow passing a .pt file or an iter_* dir directly.
    if job_dir.is_file() and str(job_dir).endswith(".pt"):
        return str(job_dir.resolve())
    if not iteration and (job_dir / "model_ema_bf16.pt").is_file():
        return str((job_dir / "model_ema_bf16.pt").resolve())

    ckpt_root = job_dir / "checkpoints" if (job_dir / "checkpoints").is_dir() else job_dir
    if iteration:
        name = iteration if str(iteration).startswith("iter_") else f"iter_{int(iteration):09d}"
        path = ckpt_root / name
    elif (ckpt_root / "latest_checkpoint.txt").is_file():
        name = (ckpt_root / "latest_checkpoint.txt").read_text(encoding="utf-8").strip()
        path = ckpt_root / name
    else:
        iters = sorted(ckpt_root.glob("iter_*"))
        if not iters:
            raise FileNotFoundError(f"No checkpoints under {ckpt_root}")
        path = iters[-1]

    if path.is_file() and str(path).endswith(".pt"):
        return str(path.resolve())
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint dir not found: {path}")

    for name in ("model_ema_bf16.pt", "model_ema_fp32.pt", "model.pt"):
        pt = path / name
        if pt.is_file():
            log.info(f"Using consolidated checkpoint: {pt}")
            return str(pt.resolve())

    model_dir = path / "model"
    if not model_dir.is_dir():
        raise FileNotFoundError(f"No model_ema_bf16.pt or DCP model/ under {path}")

    if not convert_if_needed:
        raise FileNotFoundError(
            f"Found DCP at {model_dir} but no model_ema_bf16.pt. Convert with:\n"
            f"  python scripts/convert_distcp_to_pt.py {model_dir} {path}"
        )

    log.warning(f"Converting DCP → model_ema_bf16.pt under {path} (one-time, may take a few minutes)...")
    import subprocess
    import sys

    convert_script = Path("scripts/convert_distcp_to_pt.py")
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "scripts" / "convert_distcp_to_pt.py"
        if candidate.is_file():
            convert_script = candidate
            break
    cmd = [sys.executable, str(convert_script), str(model_dir), str(path)]
    subprocess.run(cmd, check=True)
    ema_pt = path / "model_ema_bf16.pt"
    if not ema_pt.is_file():
        raise FileNotFoundError(f"Conversion finished but missing {ema_pt}")
    log.info(f"Converted checkpoint ready: {ema_pt}")
    return str(ema_pt.resolve())


def _center_crop_resize_thw(frames_thw: torch.Tensor, resolution: tuple[int, int]) -> torch.Tensor:
    """``frames_thw``: [T,C,H,W] uint8/float → resized/cropped [T,C,H',W']."""
    return resize_input(frames_thw, list(resolution))


def load_rgb_video(
    path: str,
    *,
    num_frames: Optional[int] = None,
    resolution: tuple[int, int] = DEFAULT_RESOLUTION,
    start_frame: int = 0,
    temporal_stride: int = 1,
) -> torch.Tensor:
    """Load RGB mp4/image → ``[C, T, H, W]`` uint8.

    ``temporal_stride`` keeps every N-th frame after ``start_frame`` (RSW
    training uses 4 so RGB / action / action_video share kept indices).
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in {".png", ".jpg", ".jpeg", ".webp"}:
        img = Image.open(path).convert("RGB")
        chw = torchvision.transforms.functional.to_tensor(img)  # [C,H,W] float [0,1]
        frames = (chw * 255.0).to(torch.uint8).unsqueeze(0)  # [T=1,C,H,W]
    elif ext in {".mp4", ".webm", ".avi", ".mov"}:
        video_frames, _ = easy_io.load(path)  # (T,H,W,C)
        arr = torch.from_numpy(np.asarray(video_frames)).permute(0, 3, 1, 2)  # TCHW
        if arr.dtype != torch.uint8:
            if float(arr.max()) <= 1.0:
                arr = (arr * 255.0).clamp(0, 255).to(torch.uint8)
            else:
                arr = arr.clamp(0, 255).to(torch.uint8)
        frames = arr[start_frame:]
    else:
        raise ValueError(f"Unsupported media extension: {ext}")

    stride = max(int(temporal_stride), 1)
    if stride > 1 and frames.shape[0] > 0:
        frames = frames[::stride]

    if num_frames is not None:
        if frames.shape[0] < num_frames:
            pad = frames[-1:].repeat(num_frames - frames.shape[0], 1, 1, 1)
            frames = torch.cat([frames, pad], dim=0)
        else:
            frames = frames[:num_frames]

    frames = _center_crop_resize_thw(frames, resolution)
    return frames.permute(1, 0, 2, 3).contiguous()  # CTHW


def load_multiview_video(
    paths: Sequence[str],
    *,
    num_frames: Optional[int] = None,
    resolution: tuple[int, int] = DEFAULT_RESOLUTION,
    temporal_stride: int = 1,
) -> torch.Tensor:
    """Head + left / right wrist mp4s → one packed ``[C, T, H, W]`` uint8 canvas (``compose_multiview``)."""
    from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_robodojo_action import (
        compose_multiview,
    )

    views = [
        load_rgb_video(p, num_frames=num_frames, resolution=resolution, temporal_stride=temporal_stride)
        for p in paths
    ]
    t = min(v.shape[1] for v in views)
    tchw = [v[:, :t].permute(1, 0, 2, 3) for v in views]
    return compose_multiview(tchw, *resolution).permute(1, 0, 2, 3).contiguous()


def load_rgb_first_frame(
    path: str,
    *,
    resolution: tuple[int, int] = DEFAULT_RESOLUTION,
) -> torch.Tensor:
    """Decode only frame 0 of an RGB video/image → ``[C, 1, H, W]`` uint8."""
    ext = os.path.splitext(path)[1].lower()
    if ext in {".png", ".jpg", ".jpeg", ".webp"}:
        return load_rgb_video(path, num_frames=1, resolution=resolution, start_frame=0)
    from decord import VideoReader, cpu

    vr = VideoReader(path, ctx=cpu(0))
    frame = torch.from_numpy(vr[0].asnumpy())  # HWC
    del vr
    frames = frame.unsqueeze(0).permute(0, 3, 1, 2)  # TCHW
    if frames.dtype != torch.uint8:
        if float(frames.max()) <= 1.0:
            frames = (frames * 255.0).clamp(0, 255).to(torch.uint8)
        else:
            frames = frames.clamp(0, 255).to(torch.uint8)
    frames = _center_crop_resize_thw(frames, resolution)
    return frames.permute(1, 0, 2, 3).contiguous()


def load_action_npy(
    path: str,
    *,
    num_arms: int = 1,
    include_gripper: bool = True,
    already_unified: Optional[bool] = None,
    gripper_raw_max: Optional[float] = None,
) -> torch.Tensor:
    """Load ``action.npy`` → absolute unified ``[T, 28]`` float32.

    ``gripper_raw_max`` defaults to unit aperture (1.0). Pass
    ``SIM_PANDA_GRIPPER_RAW_MAX`` (0.08) for Libero / ManiSkill / RoboCasa.
    """
    from cosmos_predict2._src.predict2.datasets.local_datasets.action_unified_layout import (
        GRIPPER_UNIT_RAW_MAX,
    )

    raw = np.asarray(np.load(path), dtype=np.float32)
    if raw.ndim != 2:
        raise ValueError(f"Expected action [T, D], got {raw.shape} from {path}")
    if already_unified is None:
        already_unified = raw.shape[1] == UNIFIED_ACTION_DIM
    if already_unified:
        if raw.shape[1] != UNIFIED_ACTION_DIM:
            raise ValueError(f"already_unified but D={raw.shape[1]} != {UNIFIED_ACTION_DIM}")
        unified = raw
    else:
        unified = to_unified_bimanual_action(
            raw,
            num_arms=num_arms,
            include_gripper=include_gripper,
            gripper_raw_max=float(
                GRIPPER_UNIT_RAW_MAX if gripper_raw_max is None else gripper_raw_max
            ),
        )
    return torch.from_numpy(np.ascontiguousarray(unified)).float()


def load_action_rsw_parquet(
    parquet_path: str,
    *,
    action_column: str = "observation.state",
    include_gripper: bool = True,
    include_lift: bool = False,
) -> torch.Tensor:
    """Load RSW episode parquet → absolute unified ``[T, 28]`` (Lift dropped; dexhand=0).

    Defaults to ``observation.state`` (follower joints), matching training /
    action_video FK. Pass ``action_column=\"action\"`` for leader commands.
    """
    import pyarrow.parquet as pq

    from cosmos_predict2._src.predict2.datasets.local_datasets.action_unified_layout import (
        expand_legacy16_to_unified,
    )
    from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_real_source_world_action import (
        DEFAULT_ACTION_COLUMN,
        DEFAULT_RAW_ACTION_DIM,
        _list_column_to_numpy,
        resolve_action_indices,
    )

    if not action_column:
        action_column = DEFAULT_ACTION_COLUMN

    if not os.path.isfile(parquet_path):
        raise FileNotFoundError(f"Parquet not found: {parquet_path}")
    table = pq.read_table(parquet_path, columns=[action_column])
    actions = _list_column_to_numpy(table.column(action_column))
    if actions.ndim != 2:
        raise ValueError(f"Expected action [T, D], got {actions.shape} from {parquet_path}")
    if actions.shape[1] < DEFAULT_RAW_ACTION_DIM:
        raise ValueError(
            f"Action dim {actions.shape[1]} < raw_action_dim={DEFAULT_RAW_ACTION_DIM} in {parquet_path}"
        )
    indices, _ = resolve_action_indices(include_gripper=include_gripper, include_lift=include_lift)
    selected16 = actions[:, indices]
    unified = expand_legacy16_to_unified(selected16)
    return torch.from_numpy(np.ascontiguousarray(unified)).float()


def load_action_agibot_parquet(parquet_path: str) -> torch.Tensor:
    """Load gripper AgiBot parquet → absolute unified ``[T, 28]`` (dexhand=0)."""
    from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_agibot_action import (
        load_agibot_action_from_parquet,
    )

    arr = load_agibot_action_from_parquet(parquet_path)
    return torch.from_numpy(np.ascontiguousarray(arr)).float()


def load_action_robotwin_hdf5(
    hdf5_path: str,
    *,
    action_key: str = "joint_action/vector",
    include_gripper: bool = True,
    source_arm_dof: int = 6,
    normalize_gripper: bool = True,
) -> torch.Tensor:
    """Load RoboTwin HDF5 ``joint_action/vector`` → absolute unified ``[T, 28]``.

    Set ``normalize_gripper=False`` to keep source gripper values (for LeRobot export).
    """
    import h5py

    if not os.path.isfile(hdf5_path):
        raise FileNotFoundError(f"HDF5 not found: {hdf5_path}")
    with h5py.File(hdf5_path, "r") as h:
        if action_key not in h:
            raise KeyError(
                f"Missing action key {action_key!r} in {hdf5_path}; "
                f"keys sample={list(h.keys())[:20]}"
            )
        raw = np.asarray(h[action_key][:], dtype=np.float32)
    if raw.ndim != 2:
        raise ValueError(f"Expected action [T, D], got {raw.shape} from {hdf5_path}")
    unified = to_unified_bimanual_action(
        raw,
        num_arms=2,
        include_gripper=include_gripper,
        arm_dof=int(source_arm_dof),
        normalize_gripper=bool(normalize_gripper),
        normalize_dexhand=False,
    )
    return torch.from_numpy(np.ascontiguousarray(unified)).float()


def load_action_robomind_parquet(
    parquet_path: str,
    *,
    include_gripper: bool = True,
    prefer_control_joints: bool = True,
) -> torch.Tensor:
    """Load RoboMind Tiangong parquet → absolute unified ``[T, 28]``."""
    from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_robomind_action import (
        load_robomind_action_from_parquet,
    )

    arr = load_robomind_action_from_parquet(
        parquet_path,
        include_gripper=include_gripper,
        prefer_control_joints=prefer_control_joints,
    )
    return torch.from_numpy(np.ascontiguousarray(arr)).float()


def load_action_agibot_dexhand_parquet(parquet_path: str) -> torch.Tensor:
    """Load AgiBot dexhand parquet → absolute unified ``[T, 28]``."""
    from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_agibot_dexhand_action import (
        load_agibot_dexhand_action_from_parquet,
    )

    arr = load_agibot_dexhand_action_from_parquet(parquet_path)
    return torch.from_numpy(np.ascontiguousarray(arr)).float()


def absolute_action_to_state_delta(
    action: torch.Tensor,
    action_delta_scaler: Optional[Union[float, Sequence[float], torch.Tensor]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``action`` [T,D] or [B,T,D] → ``action_state``, scaled ``action_delta``."""
    if action.ndim == 2:
        action_b = action.unsqueeze(0)
        squeeze = True
    elif action.ndim == 3:
        action_b = action
        squeeze = False
    else:
        raise ValueError(f"Expected action [T,D] or [B,T,D], got {tuple(action.shape)}")

    state = action_b[:, 0, :].contiguous()
    if action_b.shape[1] > 1:
        delta = action_b[:, 1:, :] - action_b[:, :-1, :]
    else:
        delta = torch.zeros(action_b.shape[0], 1, action_b.shape[-1], dtype=action.dtype)

    if action_delta_scaler is not None:
        scaler = normalize_action_delta_scaler(action_delta_scaler, action_b.shape[-1])
        assert scaler is not None
        delta = delta * scaler.to(dtype=delta.dtype)

    if squeeze:
        return state[0], delta[0]
    return state, delta


def pad_or_trim_action(action: torch.Tensor, length: int) -> torch.Tensor:
    """Pad with last row or trim absolute action to ``length`` frames."""
    t = action.shape[0]
    if t == length:
        return action
    if t > length:
        return action[:length]
    pad = action[-1:].repeat(length - t, 1)
    return torch.cat([action, pad], dim=0)


def tensor_to_uint8_video(video_btchw: torch.Tensor) -> np.ndarray:
    """Model output ``[-1,1]`` [B,C,T,H,W] → numpy uint8 [T,H,W,C]."""
    v = video_btchw[0].detach().float().cpu()
    v = ((v.clamp(-1, 1) + 1.0) * 0.5 * 255.0).to(torch.uint8)
    return v.permute(1, 2, 3, 0).numpy()


def cthw_uint8_to_thwc(video_cthw: torch.Tensor) -> np.ndarray:
    """``[C,T,H,W]`` uint8 → numpy ``[T,H,W,C]``."""
    v = video_cthw.detach().cpu()
    if v.dtype != torch.uint8:
        if float(v.max()) <= 1.0:
            v = (v.clamp(0, 1) * 255.0).to(torch.uint8)
        else:
            v = v.clamp(0, 255).to(torch.uint8)
    return v.permute(1, 2, 3, 0).contiguous().numpy()


def pad_or_trim_video_thwc(frames: np.ndarray, length: int) -> np.ndarray:
    """Pad with last frame or trim ``[T,H,W,C]`` to ``length``."""
    t = frames.shape[0]
    if t == length:
        return frames
    if t > length:
        return frames[:length]
    pad = np.repeat(frames[-1:], length - t, axis=0)
    return np.concatenate([frames, pad], axis=0)


def _load_label_font(font_size: int = 28) -> ImageFont.ImageFont:
    for name in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVuSans-Bold.ttf",
        "DejaVuSans.ttf",
        "Arial.ttf",
    ):
        try:
            return ImageFont.truetype(name, font_size)
        except OSError:
            continue
    return ImageFont.load_default()


def annotate_video_frames(
    frames: np.ndarray,
    label: str,
    *,
    bar_height: int = 40,
    font_size: int = 28,
    bar_color: tuple[int, int, int] = (20, 20, 20),
    text_color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Add a top label bar to every frame. ``frames``: ``[T,H,W,C]`` uint8."""
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected [T,H,W,3], got {frames.shape}")
    t, h, w, _ = frames.shape
    font = _load_label_font(font_size)
    out = np.empty((t, h + bar_height, w, 3), dtype=np.uint8)
    for i in range(t):
        canvas = Image.new("RGB", (w, h + bar_height), bar_color)
        canvas.paste(Image.fromarray(frames[i]), (0, bar_height))
        draw = ImageDraw.Draw(canvas)
        try:
            bbox = draw.textbbox((0, 0), label, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            tw, th = draw.textsize(label, font=font)
        draw.text(((w - tw) // 2, max(0, (bar_height - th) // 2)), label, font=font, fill=text_color)
        out[i] = np.asarray(canvas)
    return out


def stack_labeled_videos_side_by_side(
    panels: Sequence[tuple[np.ndarray, str]],
    *,
    bar_height: int = 40,
    font_size: int = 28,
) -> np.ndarray:
    """Horizontally concat labeled videos; lengths are padded to the max T."""
    if not panels:
        raise ValueError("No video panels to stack")
    max_t = max(f.shape[0] for f, _ in panels)
    labeled = []
    for frames, label in panels:
        frames = pad_or_trim_video_thwc(frames, max_t)
        labeled.append(annotate_video_frames(frames, label, bar_height=bar_height, font_size=font_size))
    return np.concatenate(labeled, axis=2)


# Premium dark comparison style (matches scripts/restyle_action_comparison_videos.py).
_PREMIUM_BG = np.array([0, 0, 0], dtype=np.float32)
_PREMIUM_GAP_W = 14
_PREMIUM_OUTER_PAD = 18
_PREMIUM_LABEL_H = 68
_PREMIUM_SKELETON_BOOST_S = 1.45
_PREMIUM_SKELETON_BOOST_V = 1.25
_PREMIUM_SKELETON_ALPHA_GAIN = 1.45


def _deepen_skeleton_rgb(rgb: np.ndarray) -> np.ndarray:
    """Keep original blue/red hue family, push saturation & brightness a bit."""
    import cv2

    u8 = np.clip(rgb, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(u8, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * _PREMIUM_SKELETON_BOOST_S + 18.0, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * _PREMIUM_SKELETON_BOOST_V + 12.0, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)


def restyle_action_frame_rgb(rgb: np.ndarray) -> np.ndarray:
    """Save-time only: swap the white/gray render canvas for black. ``rgb``: HxWx3.

    The model still sees the original white-background tensor. Near-white pixels
    become the visualization background; bone/joint colors are kept (slightly
    boosted) so the stick figure reads on a dark panel.
    """
    rgb_f = rgb.astype(np.float32)
    mean = rgb_f.mean(axis=2)
    alpha = np.clip((245.0 - mean) / 55.0, 0.0, 1.0).astype(np.float32)
    alpha = np.clip(alpha * _PREMIUM_SKELETON_ALPHA_GAIN, 0.0, 1.0)
    alpha = np.where(mean > 242.0, 0.0, alpha)
    deepened = _deepen_skeleton_rgb(rgb_f)
    a = alpha[:, :, None]
    canvas = _PREMIUM_BG * (1.0 - a) + deepened * a
    return np.clip(canvas, 0, 255).astype(np.uint8)


def restyle_action_video_thwc(frames: np.ndarray) -> np.ndarray:
    """Save-time restyle: white-bg model frames → black-bg visualization."""
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected [T,H,W,3], got {frames.shape}")
    # Already dark (e.g. re-save): leave as-is.
    if float(frames[0].mean()) < 80.0:
        return np.asarray(frames)
    return np.stack([restyle_action_frame_rgb(frames[i]) for i in range(frames.shape[0])], axis=0)


def _style_rgb_panel(rgb: np.ndarray) -> np.ndarray:
    """Mild vignette on prediction / GT panels for premium comparison."""
    rgb_f = rgb.astype(np.float32)
    h, w = rgb_f.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rad = np.sqrt(((yy - h / 2) / (h * 0.95)) ** 2 + ((xx - w / 2) / (w * 0.95)) ** 2)
    vig = np.clip(1.0 - 0.16 * np.clip(rad - 0.25, 0, None), 0.84, 1.0)[:, :, None]
    return np.clip(rgb_f * vig, 0, 255).astype(np.uint8)


def _premium_label_bar(width: int, label: str, accent: tuple[int, int, int]) -> np.ndarray:
    img = Image.new("RGB", (width, _PREMIUM_LABEL_H), (8, 8, 8))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, width, 4], fill=accent)
    draw.line([(0, _PREMIUM_LABEL_H - 1), (width, _PREMIUM_LABEL_H - 1)], fill=(55, 55, 55), width=1)
    font = _load_label_font(30)
    text = label[:1].upper() + label[1:].lower() if label else label
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        tw, th = draw.textsize(text, font=font)
    x = (width - tw) // 2
    y = max(10, (_PREMIUM_LABEL_H - th) // 2 - 2)
    draw.text((x + 2, y + 2), text, font=font, fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=(255, 255, 255))
    return np.asarray(img)


def _premium_progress_bar(width: int, height: int, t: int, total: int) -> np.ndarray:
    img = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    pad_x = 10
    y0 = height // 2
    draw.line([(pad_x, y0), (width - pad_x, y0)], fill=(40, 48, 58), width=2)
    if total > 1:
        frac = t / (total - 1)
        x1 = pad_x + int((width - 2 * pad_x) * frac)
        draw.line([(pad_x, y0), (x1, y0)], fill=(110, 220, 255), width=2)
        draw.ellipse([x1 - 3, y0 - 3, x1 + 3, y0 + 3], fill=(210, 245, 255))
    return np.asarray(img)


def compose_premium_comparison_frame(
    pred_rgb: np.ndarray,
    gt_rgb: np.ndarray,
    act_rgb: np.ndarray,
    *,
    frame_idx: int,
    num_frames: int,
    mid_label: str = "Ground truth",
    right_label: str = "Action video",
) -> np.ndarray:
    """Build one premium dark comparison frame (Prediction | mid | Action video)."""
    h = min(pred_rgb.shape[0], gt_rgb.shape[0], act_rgb.shape[0])
    panels_src = [pred_rgb[:h], gt_rgb[:h], act_rgb[:h]]
    labels = ["Prediction", mid_label, right_label]
    accents = [(40, 140, 255), (40, 200, 120), (230, 55, 70)]

    built = []
    for img, lab, acc in zip(panels_src, labels, accents):
        bar = _premium_label_bar(img.shape[1], lab, acc)
        body = np.zeros((h + 8, img.shape[1], 3), dtype=np.uint8)
        body[4 : 4 + h] = img
        body[:, 0:2] = acc
        built.append(np.concatenate([bar, body], axis=0))

    gap = np.zeros((built[0].shape[0], _PREMIUM_GAP_W, 3), dtype=np.uint8)
    row = built[0]
    for p in built[1:]:
        row = np.concatenate([row, gap, p], axis=1)

    prog = _premium_progress_bar(row.shape[1], 16, frame_idx, num_frames)
    row = np.concatenate([row, prog], axis=0)

    canvas = np.zeros(
        (row.shape[0] + 2 * _PREMIUM_OUTER_PAD, row.shape[1] + 2 * _PREMIUM_OUTER_PAD, 3),
        dtype=np.uint8,
    )
    canvas[
        _PREMIUM_OUTER_PAD : _PREMIUM_OUTER_PAD + row.shape[0],
        _PREMIUM_OUTER_PAD : _PREMIUM_OUTER_PAD + row.shape[1],
    ] = row
    return canvas


def stack_premium_comparison_thwc(
    pred: np.ndarray,
    gt: np.ndarray,
    act: np.ndarray,
    *,
    mid_label: str = "Ground truth",
    right_label: str = "Action video",
) -> np.ndarray:
    """Build premium comparison video ``[T,H,W,3]`` from panel videos."""
    n = min(pred.shape[0], gt.shape[0], act.shape[0])
    pred_s = np.stack([_style_rgb_panel(pred[i]) for i in range(n)], axis=0)
    gt_s = np.stack([_style_rgb_panel(gt[i]) for i in range(n)], axis=0)
    frames = [
        compose_premium_comparison_frame(
            pred_s[i],
            gt_s[i],
            act[i],
            frame_idx=i,
            num_frames=n,
            mid_label=mid_label,
            right_label=right_label,
        )
        for i in range(n)
    ]
    return np.stack(frames, axis=0)


def save_labeled_inference_videos(
    *,
    save_dir: Union[str, Path],
    save_name: str,
    save_fps: float,
    pred_thwc: np.ndarray,
    gt_rgb_cthw: Optional[torch.Tensor] = None,
    action_video_cthw: Optional[torch.Tensor] = None,
    transplanted_rgb_cthw: Optional[torch.Tensor] = None,
    write_comparison: bool = True,
    style: str = "premium",
) -> dict[str, str]:
    """Save prediction / GT RGB / action video (+ optional side-by-side).

    ``style``:
      - ``premium`` (default): black-bg action skeleton + dark comparison with
        labels ``Prediction | Ground truth | Action video`` (matches the
        restyled ``iter_000068000`` n400 dump).
      - ``legacy``: plain dark top bar + white labels on each clip / concat.

    Returns mapping of role → written path.
    """
    import mediapy

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    pred = np.asarray(pred_thwc)
    num_frames = int(pred.shape[0])
    written: dict[str, str] = {}
    style_norm = (style or "premium").strip().lower()

    if style_norm == "legacy":
        pred_out = annotate_video_frames(pred, "Prediction (generated)")
        pred_path = save_dir / f"{save_name}.mp4"
        mediapy.write_video(str(pred_path), pred_out, fps=save_fps)
        written["prediction"] = str(pred_path)
        log.info(f"Saved {pred_path}  shape={pred_out.shape}")

        panels: list[tuple[np.ndarray, str]] = [(pred, "Prediction (generated)")]

        if gt_rgb_cthw is not None:
            gt = pad_or_trim_video_thwc(cthw_uint8_to_thwc(gt_rgb_cthw), num_frames)
            gt_out = annotate_video_frames(gt, "Original RGB (GT)")
            gt_path = save_dir / f"{save_name}_original_rgb.mp4"
            mediapy.write_video(str(gt_path), gt_out, fps=save_fps)
            written["original_rgb"] = str(gt_path)
            log.info(f"Saved {gt_path}  shape={gt_out.shape}")
            panels.append((gt, "Original RGB (GT)"))

        if action_video_cthw is not None:
            act = pad_or_trim_video_thwc(cthw_uint8_to_thwc(action_video_cthw), num_frames)
            act_out = annotate_video_frames(act, "Action Video (VACE)")
            act_path = save_dir / f"{save_name}_action_video.mp4"
            mediapy.write_video(str(act_path), act_out, fps=save_fps)
            written["action_video"] = str(act_path)
            log.info(f"Saved {act_path}  shape={act_out.shape}")
            panels.append((act, "Action Video (VACE)"))

        if write_comparison and len(panels) > 1:
            comparison = stack_labeled_videos_side_by_side(panels)
            cmp_path = save_dir / f"{save_name}_comparison.mp4"
            mediapy.write_video(str(cmp_path), comparison, fps=save_fps)
            written["comparison"] = str(cmp_path)
            log.info(f"Saved {cmp_path}  shape={comparison.shape}")
        return written

    # Premium style: unlabeled individual clips; restyled action; dark comparison.
    pred_path = save_dir / f"{save_name}.mp4"
    mediapy.write_video(str(pred_path), pred, fps=save_fps)
    written["prediction"] = str(pred_path)
    log.info(f"Saved {pred_path}  shape={pred.shape} style=premium")

    gt = None
    if gt_rgb_cthw is not None:
        gt = pad_or_trim_video_thwc(cthw_uint8_to_thwc(gt_rgb_cthw), num_frames)
        gt_path = save_dir / f"{save_name}_original_rgb.mp4"
        mediapy.write_video(str(gt_path), gt, fps=save_fps)
        written["original_rgb"] = str(gt_path)
        log.info(f"Saved {gt_path}  shape={gt.shape} style=premium")

    act = None
    if action_video_cthw is not None:
        act = restyle_action_video_thwc(
            pad_or_trim_video_thwc(cthw_uint8_to_thwc(action_video_cthw), num_frames)
        )
        act_path = save_dir / f"{save_name}_action_video.mp4"
        mediapy.write_video(str(act_path), act, fps=save_fps)
        written["action_video"] = str(act_path)
        log.info(f"Saved {act_path}  shape={act.shape} style=premium")

    transplanted = None
    if transplanted_rgb_cthw is not None:
        transplanted = pad_or_trim_video_thwc(
            cthw_uint8_to_thwc(transplanted_rgb_cthw), num_frames
        )
        src_path = save_dir / f"{save_name}_transplanted_action_rgb.mp4"
        mediapy.write_video(str(src_path), transplanted, fps=save_fps)
        written["transplanted_action_rgb"] = str(src_path)
        log.info(f"Saved {src_path}  shape={transplanted.shape} style=premium")

    if write_comparison and act is not None:
        # Figure-7 style: Prediction | transplanted action RGB | skeleton.
        mid = transplanted if transplanted is not None else gt
        mid_label = "Transplanted action" if transplanted is not None else "Original RGB"
        if mid is not None:
            comparison = stack_premium_comparison_thwc(
                pred, mid, act, mid_label=mid_label, right_label="Action skeleton"
            )
            cmp_path = save_dir / f"{save_name}_comparison.mp4"
            mediapy.write_video(str(cmp_path), comparison, fps=save_fps)
            written["comparison"] = str(cmp_path)
            log.info(f"Saved {cmp_path}  shape={comparison.shape} style=premium")

    return written


@dataclass
class ActionInjectSample:
    """One inference sample (unbatched tensors)."""

    video: torch.Tensor  # [C,T,H,W] uint8 — conditioning filled in first frames
    action: torch.Tensor  # [T,28] absolute (unscaled)
    prompt: str
    action_video: Optional[torch.Tensor] = None  # [C,T,H,W] uint8
    fps: float = 16.0


class ActionInjectInferencePipeline:
    """Load action-inject RF model and run single / AR generation."""

    def __init__(
        self,
        ckpt_path: str,
        *,
        experiment_name: str = DEFAULT_EXPERIMENT,
        config_file: str = DEFAULT_CONFIG_FILE,
        context_parallel_size: int = 1,
        experiment_opts: Optional[list[str]] = None,
        action_delta_scaler: Optional[Union[float, Sequence[float]]] = None,
    ) -> None:
        self.experiment_name = experiment_name
        self.ckpt_path = ckpt_path
        self.config_file = config_file
        self.context_parallel_size = int(context_parallel_size)

        if action_delta_scaler is None:
            action_delta_scaler = build_action_delta_scaler(
                include_gripper=True, include_lift=False
            )
        self.action_delta_scaler = normalize_action_delta_scaler(
            action_delta_scaler, UNIFIED_ACTION_DIM
        )

        if self.context_parallel_size > 1:
            self._init_distributed()

        opts = list(experiment_opts or [])
        # Avoid constructing unused train dataloaders during inference.
        if "~data_train" not in opts and "data_train" not in " ".join(opts):
            opts.append("~data_train")

        log.info(
            f"Loading action-inject model: experiment={experiment_name}, "
            f"ckpt={ckpt_path}, config={config_file}"
        )
        from cosmos_oss.checkpoints_predict2 import register_checkpoints

        register_checkpoints()
        self.model, self.config = load_model_from_checkpoint(
            experiment_name=experiment_name,
            s3_checkpoint_dir=ckpt_path,
            config_file=config_file,
            load_ema_to_reg=True,
            experiment_opts=opts,
            to_device="cuda",
        )
        self.model.eval()
        self.chunk_size = int(self.model.tokenizer.get_pixel_num_frames(self.model.config.state_t))
        log.info(f"Native chunk_size (pixel frames) = {self.chunk_size}")

    def _init_distributed(self) -> None:
        distributed.init()
        parallel_state.initialize_model_parallel(
            context_parallel_size=self.context_parallel_size
        )
        process_group = parallel_state.get_context_parallel_group()
        self.process_group = process_group

    @property
    def is_rank0(self) -> bool:
        if self.context_parallel_size <= 1:
            return True
        return distributed.get_rank() == 0

    def _embed_text(self, prompt: str, negative_prompt: str) -> dict:
        data: dict = {"ai_caption": [prompt], "images": None}
        if self.model.text_encoder is not None:
            pos = self.model.text_encoder.compute_text_embeddings_online(
                data_batch=data, input_caption_key="ai_caption"
            )
            neg = self.model.text_encoder.compute_text_embeddings_online(
                data_batch={"ai_caption": [negative_prompt], "images": None},
                input_caption_key="ai_caption",
            )
        else:
            from cosmos_predict2._src.predict2.inference.get_t5_emb import get_text_embedding

            pos = get_text_embedding(prompt)
            neg = get_text_embedding(negative_prompt)
        return {
            "ai_caption": [prompt],
            "t5_text_embeddings": pos,
            "neg_t5_text_embeddings": neg,
            "t5_text_mask": torch.ones(1, pos.shape[1], device=pos.device),
            "neg_t5_text_mask": torch.ones(1, neg.shape[1], device=neg.device),
        }

    def build_data_batch(
        self,
        sample: ActionInjectSample,
        *,
        num_conditional_frames: int = 1,
        negative_prompt: str = _DEFAULT_NEGATIVE_PROMPT,
    ) -> dict:
        """Build a batched model input from an ``ActionInjectSample``."""
        video = sample.video
        if video.ndim != 4:
            raise ValueError(f"video must be [C,T,H,W], got {tuple(video.shape)}")
        c, t, h, w = video.shape
        action = pad_or_trim_action(sample.action.float(), t)
        state, delta = absolute_action_to_state_delta(action, self.action_delta_scaler)

        batch: dict = {
            "dataset_name": "video_data",
            "video": video.unsqueeze(0).contiguous(),  # [1,C,T,H,W] uint8
            "action": action.unsqueeze(0).contiguous(),
            "action_state": state.unsqueeze(0).contiguous(),
            "action_delta": delta.unsqueeze(0).contiguous(),
            "fps": torch.tensor([float(sample.fps)], dtype=torch.float32),
            "image_size": torch.tensor([[h, w, h, w]], dtype=torch.float32),
            "num_frames": t,
            "padding_mask": torch.zeros(1, 1, h, w),
            "num_conditional_frames": int(num_conditional_frames),
        }
        if sample.action_video is not None:
            av = sample.action_video
            if av.shape[1] != t:
                # Align temporal length with RGB window.
                if av.shape[1] > t:
                    av = av[:, :t]
                else:
                    pad = av[:, -1:].repeat(1, t - av.shape[1], 1, 1)
                    av = torch.cat([av, pad], dim=1)
            batch["action_video"] = av.unsqueeze(0).contiguous()

        batch.update(self._embed_text(sample.prompt, negative_prompt))

        for k, v in list(batch.items()):
            if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
                batch[k] = v.cuda().to(dtype=torch.bfloat16)
            elif isinstance(v, torch.Tensor) and v.dtype == torch.uint8:
                batch[k] = v.cuda()
            elif isinstance(v, torch.Tensor):
                batch[k] = v.cuda()
        return batch

    @torch.no_grad()
    def generate_chunk(
        self,
        sample: ActionInjectSample,
        *,
        guidance: float = 3.0,
        seed: int = 1,
        num_steps: int = 35,
        num_conditional_frames: int = 1,
        negative_prompt: str = _DEFAULT_NEGATIVE_PROMPT,
    ) -> torch.Tensor:
        """Generate one native chunk. Returns ``[-1,1]`` video ``[1,C,T,H,W]``."""
        data_batch = self.build_data_batch(
            sample,
            num_conditional_frames=num_conditional_frames,
            negative_prompt=negative_prompt,
        )
        # Encode action_video + get latent shape (also matches EveryNDrawSample).
        _, x0, _ = self.model.get_data_and_condition(data_batch)
        sample_latent = self.model.generate_samples_from_batch(
            data_batch,
            guidance=guidance,
            seed=seed,
            state_shape=x0.shape[1:],
            n_sample=x0.shape[0],
            num_steps=num_steps,
            is_negative_prompt=True,
        )
        video = self.model.decode(sample_latent)
        return video

    def _make_cond_video(
        self,
        cond_frames_cthw: torch.Tensor,
        *,
        total_frames: int,
        resolution: tuple[int, int],
    ) -> torch.Tensor:
        """Build ``[C,T,H,W]`` uint8 window: leading cond frames + zero pad."""
        # cond_frames_cthw: [C, Tc, H, W] uint8
        c = cond_frames_cthw.shape[0]
        h, w = resolution
        out = torch.zeros(c, total_frames, h, w, dtype=torch.uint8)
        tc = min(cond_frames_cthw.shape[1], total_frames)
        # Resize cond if needed
        thw = cond_frames_cthw.permute(1, 0, 2, 3)  # TCHW
        thw = _center_crop_resize_thw(thw, resolution)
        cthw = thw.permute(1, 0, 2, 3)
        out[:, :tc] = cthw[:, :tc]
        return out

    @torch.no_grad()
    def generate_ar(
        self,
        *,
        prompt: str,
        action: torch.Tensor,
        init_video_cthw: torch.Tensor,
        action_video: Optional[torch.Tensor] = None,
        num_output_frames: Optional[int] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: int = 1,
        guidance: float = 3.0,
        seed: int = 1,
        num_steps: int = 35,
        num_conditional_frames: int = 1,
        fps: float = 16.0,
        negative_prompt: str = _DEFAULT_NEGATIVE_PROMPT,
        resolution: Optional[tuple[int, int]] = None,
    ) -> torch.Tensor:
        """Autoregressive multi-chunk generation with per-chunk action slicing.

        Args:
            action: absolute unified actions ``[T_act, 28]`` (unscaled).
            init_video_cthw: conditioning RGB ``[C, T0, H, W]`` uint8 (image → T0=1).
            action_video: optional GT action-video ``[C, T_av, H, W]`` aligned to actions.
            num_output_frames: total pixel frames to produce. Default = ``action`` length
                (at least one chunk).
            chunk_overlap: overlapping pixel frames between chunks (usually 1).

        Returns:
            Generated video ``[-1, 1]`` with shape ``[1, C, T, H, W]``.
        """
        chunk_size = int(chunk_size or self.chunk_size)
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError(f"Invalid chunk_overlap={chunk_overlap} for chunk_size={chunk_size}")

        action = action.float()
        if action.ndim != 2 or action.shape[-1] != UNIFIED_ACTION_DIM:
            raise ValueError(f"action must be [T, {UNIFIED_ACTION_DIM}], got {tuple(action.shape)}")

        if resolution is None:
            resolution = DEFAULT_RESOLUTION

        if num_output_frames is None:
            num_output_frames = max(int(action.shape[0]), chunk_size)
        num_output_frames = int(num_output_frames)

        effective = chunk_size - chunk_overlap
        remaining = num_output_frames - chunk_size
        if remaining <= 0:
            num_chunks = 1
        else:
            num_chunks = 1 + (remaining + effective - 1) // effective

        log.info(
            f"AR generate: chunks={num_chunks}, chunk_size={chunk_size}, "
            f"overlap={chunk_overlap}, num_output_frames={num_output_frames}, "
            f"action_T={action.shape[0]}"
        )

        # Ensure action covers the full rollout horizon.
        need_action_t = chunk_size + (num_chunks - 1) * effective
        action_ext = pad_or_trim_action(action, need_action_t)
        if action_video is not None:
            # Pad/trim action video temporally the same way (repeat last frame).
            av = action_video
            if av.shape[1] < need_action_t:
                pad = av[:, -1:].repeat(1, need_action_t - av.shape[1], 1, 1)
                av = torch.cat([av, pad], dim=1)
            else:
                av = av[:, :need_action_t]
            action_video_ext = av
        else:
            action_video_ext = None

        generated_chunks: list[torch.Tensor] = []
        # Current conditioning frames (pixel). Start from init.
        cond = init_video_cthw
        frames_to_extract = 4 * (num_conditional_frames - 1) + 1

        for chunk_idx in range(num_chunks):
            start = chunk_idx * effective
            action_chunk = action_ext[start : start + chunk_size]
            av_chunk = None
            if action_video_ext is not None:
                av_chunk = action_video_ext[:, start : start + chunk_size]

            # Build RGB window: leading cond frames + zeros.
            if cond.shape[1] >= frames_to_extract:
                cond_use = cond[:, -frames_to_extract:]
            else:
                cond_use = cond
            video_window = self._make_cond_video(
                cond_use, total_frames=chunk_size, resolution=resolution
            )

            sample = ActionInjectSample(
                video=video_window,
                action=action_chunk,
                prompt=prompt,
                action_video=av_chunk,
                fps=fps,
            )
            # First chunk uses requested cond frames; later chunks keep overlap cond.
            n_cond = num_conditional_frames if chunk_idx == 0 else max(1, num_conditional_frames)
            video = self.generate_chunk(
                sample,
                guidance=guidance,
                seed=seed + chunk_idx,
                num_steps=num_steps,
                num_conditional_frames=n_cond,
                negative_prompt=negative_prompt,
            )  # [1,C,T,H,W] in [-1,1]

            generated_chunks.append(video)

            # Update cond from generated pixels (convert to uint8 CTHW).
            gen_uint8 = (
                ((video[0].detach().float().clamp(-1, 1) + 1.0) * 0.5 * 255.0)
                .to(torch.uint8)
                .cpu()
            )
            cond = gen_uint8  # [C,T,H,W]
            log.info(f"AR chunk {chunk_idx + 1}/{num_chunks} done")

        # Stitch: full first chunk + non-overlapping tails.
        parts = [generated_chunks[0]]
        for i in range(1, len(generated_chunks)):
            parts.append(generated_chunks[i][:, :, chunk_overlap:, :, :])
        out = torch.cat(parts, dim=2)
        if out.shape[2] > num_output_frames:
            out = out[:, :, :num_output_frames]
        return out

    def cleanup(self) -> None:
        if self.context_parallel_size > 1:
            import torch.distributed as dist

            if dist.is_initialized():
                dist.barrier()
            if parallel_state.is_initialized():
                parallel_state.destroy_model_parallel()
