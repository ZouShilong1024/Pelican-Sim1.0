# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataset-driven action-inject inference (single-chunk or AR).

Samples episodes from the same 8 action datasets used in dense / MoE
``action_inject`` training (RSW / AgiBot / AgiBot-dexhand / RoboMind /
Libero / ManiSkill / RoboTwin / RoboCasa / mixed), plus inference-only
``libero_ood`` / ``ur5_robosuite_ood`` / ``iiwa_robosuite_ood``, then runs
action-conditioned Video2World generation. Mix weights default to training
``MIX_WEIGHTS``.

Multi-GPU: prefer data-parallel (independent workers per GPU, same pattern as
``infer_action_inject_edited_first_frame.sh``). Context-parallel via
``--context-parallel-size`` is still available but collapses AR quality.

Released checkpoint (consolidated ``.pt``)::

    python examples/action_inject_inference_from_dataset.py \\
      --ckpt-path checkpoints/Pelican-Sim-1.0/model_ema_bf16.pt \\
      --sample-dir examples/samples/libero_spatial_top_drawer \\
      --mode ar --match-episode-length --save-dir outputs/demo

A fine-tuning job dir also works (its DCP checkpoint is converted to ``.pt`` once)::

    python examples/action_inject_inference_from_dataset.py \\
      --job-dir outputs/pelican_sim/finetune/<job_name> --iteration 2000 \\
      --sample-dir data/robodojo_ft/<episode> --save-dir outputs/ft_demo

    # Libero OOD only (view-gen val):
    python examples/action_inject_inference_from_dataset.py \\
      --dataset libero_ood --num-samples 8 --save-dir outputs/libero_ood_infer

    # UR5 / IIWA Robosuite OOD:
    python examples/action_inject_inference_from_dataset.py \\
      --dataset ur5_robosuite_ood --num-samples 8 --save-dir outputs/ur5_ood_infer

RoboDojo multi-view (head + both wrists packed on one canvas, 13-frame model; the
skeletons of all three cameras are rendered online; see ``infer_robodojo_mv.sh``)::

    python examples/action_inject_inference_from_dataset.py \\
      --ckpt-path checkpoints/<mv job>/<iter>/model_ema_bf16.pt --state-t 4 \\
      --sample-dir examples/samples/robodojo_mv_hold_button \\
      --mode ar --match-episode-length --save-dir outputs/robodojo_mv
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.predict2.action.inference.action_inject_pipeline import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CONFIG_FILE,
    DEFAULT_EXPERIMENT,
    DEFAULT_JOB_DIR,
    ActionInjectInferencePipeline,
    ActionInjectSample,
    _DEFAULT_NEGATIVE_PROMPT,
    load_multiview_video,
    load_rgb_first_frame,
    load_rgb_video,
    pad_or_trim_action,
    resolve_checkpoint_dir,
    resolve_experiment_and_config,
    save_labeled_inference_videos,
    tensor_to_uint8_video,
)
from cosmos_predict2._src.predict2.action.inference.online_action_video import (
    resolve_action_video_cthw,
)
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_robodojo_action import (
    multiview_cells,
)
from cosmos_predict2.action_inject_dataset_inference import (
    DEFAULT_AGIBOT_DEXHAND_LABELS_JSONL,
    DEFAULT_AGIBOT_LABELS_PKL,
    DEFAULT_LIBERO_DIR,
    DEFAULT_LIBERO_OOD_DIR,
    DEFAULT_IIWA_ROBOSUITE_OOD_DIR,
    DEFAULT_MANISKILL_DIR,
    DEFAULT_OVEN_PLATEPHASE_DIR,
    DEFAULT_MIX_NAMES,
    DEFAULT_MIX_WEIGHTS,
    DEFAULT_ROBOCASA_DIRS,
    DEFAULT_ROBOMIND_PAIRS_JSONL,
    DEFAULT_ROBOTWIN_PAIRS_JSONL,
    DEFAULT_RSW_DIR,
    DEFAULT_UR5_ROBOSUITE_OOD_DIR,
    ActionDatasetSource,
    build_action_inference_dataset,
    load_episode_action,
    load_sample_dir,
    pingpong_concat_time,
    read_episode_manifest,
    sample_action_episodes,
    write_episode_manifest,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--job-dir",
        type=str,
        default=DEFAULT_JOB_DIR,
        help=f"Checkpoint dir with model_ema_bf16.pt, or a job dir containing checkpoints/ "
        f"(default: {DEFAULT_JOB_DIR}).",
    )
    p.add_argument("--ckpt-path", type=str, default=None, help="Consolidated .pt, e.g. the Hugging Face release.")
    p.add_argument("--iteration", type=str, default=None)
    p.add_argument(
        "--experiment",
        type=str,
        default=None,
        help=f"Override experiment (default: {DEFAULT_EXPERIMENT})",
    )
    p.add_argument(
        "--config-file",
        type=str,
        default=None,
        help=f"Override config file (default: {DEFAULT_CONFIG_FILE})",
    )
    p.add_argument(
        "--state-t",
        type=int,
        default=None,
        help="Latent frames per chunk the checkpoint was trained with (24 = 93 frames, default; "
        "4 = 13 frames, e.g. the RoboDojo multi-view model).",
    )

    p.add_argument(
        "--dataset",
        type=str,
        default=ActionDatasetSource.MIXED.value,
        choices=[s.value for s in ActionDatasetSource],
        help="Action dataset source (aligned with action_inject training; "
        "libero_ood / ur5_robosuite_ood / iiwa_robosuite_ood are inference-only).",
    )
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--sample-seed", type=int, default=0, help="RNG seed for episode sampling")
    p.add_argument("--rsw-dir", type=str, default=DEFAULT_RSW_DIR)
    p.add_argument("--agibot-labels-pkl", type=str, default=DEFAULT_AGIBOT_LABELS_PKL)
    p.add_argument(
        "--agibot-dexhand-labels-jsonl",
        type=str,
        default=DEFAULT_AGIBOT_DEXHAND_LABELS_JSONL,
    )
    p.add_argument("--robomind-pairs-jsonl", type=str, default=DEFAULT_ROBOMIND_PAIRS_JSONL)
    p.add_argument("--libero-dir", type=str, default=DEFAULT_LIBERO_DIR)
    p.add_argument(
        "--libero-ood-dir",
        type=str,
        default=DEFAULT_LIBERO_OOD_DIR,
        help="LiberoOODDataset root (default: libero_view_gen_val)",
    )
    p.add_argument(
        "--ur5-robosuite-ood-dir",
        type=str,
        default=DEFAULT_UR5_ROBOSUITE_OOD_DIR,
        help="UR5RobosuiteOOD root (default: ur5_robosuite_ood_ac)",
    )
    p.add_argument(
        "--iiwa-robosuite-ood-dir",
        type=str,
        default=DEFAULT_IIWA_ROBOSUITE_OOD_DIR,
        help="IIWARobosuiteOOD root (default: iiwa_robosuite_ood_ac)",
    )
    p.add_argument("--maniskill-dir", type=str, default=DEFAULT_MANISKILL_DIR)
    p.add_argument(
        "--oven-platephase-dir",
        type=str,
        default=DEFAULT_OVEN_PLATEPHASE_DIR,
        help="LeRobot v2.1 oven plate-phase root (default: oven_0810_platephase)",
    )
    p.add_argument(
        "--rgb-first-frame-only",
        action="store_true",
        help="Condition AR on RGB frame 0 only (do not decode the rest of the RGB video).",
    )
    p.add_argument("--robotwin-pairs-jsonl", type=str, default=DEFAULT_ROBOTWIN_PAIRS_JSONL)
    p.add_argument(
        "--robocasa-dirs",
        type=str,
        default=",".join(DEFAULT_ROBOCASA_DIRS),
        help="Comma-separated RoboCasa roots (dualview_v2 + mg_v2 by default)",
    )
    p.add_argument(
        "--mix-weights",
        type=str,
        default=",".join(str(w) for w in DEFAULT_MIX_WEIGHTS),
        help=(
            "Comma-separated weights for "
            f"{list(DEFAULT_MIX_NAMES)} when dataset=mixed"
        ),
    )
    p.add_argument("--verify-files", action="store_true", help="Verify media paths while indexing")
    p.add_argument(
        "--episode-manifest",
        type=str,
        default=None,
        help="If set, skip dataset sampling and run exactly the episodes in this jsonl.",
    )
    p.add_argument(
        "--sample-dir",
        type=str,
        default=None,
        help=(
            "Public example folder (rgb + action.npy + prompt + meta.json). "
            "Skips dataset indexing. Repeatable via comma-separated paths."
        ),
    )
    p.add_argument(
        "--render-action-video",
        type=str,
        default="auto",
        choices=["auto", "always", "never"],
        help=(
            "auto: use action_video.mp4 when present, else render online; "
            "always: always render online; never: require a pre-rendered mp4."
        ),
    )
    p.add_argument(
        "--action-pingpong",
        action="store_true",
        help=(
            "After load, concat action / RGB / action-video as forward + "
            "reverse-without-last-frame (2T-1). Junction is continuous."
        ),
    )

    p.add_argument("--mode", choices=["single", "ar"], default="ar")
    p.add_argument("--resolution", type=str, default="480,640", help="H,W")
    p.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Model FPS passed in the generation batch (default: sample meta model_fps, else 15)",
    )
    p.add_argument(
        "--save-fps",
        type=float,
        default=None,
        help="FPS metadata when writing saved mp4s (default: the model FPS)",
    )
    p.add_argument(
        "--metrics",
        action="store_true",
        help="Record PSNR / SSIM of prediction vs original RGB (frames after the condition frame).",
    )
    p.add_argument(
        "--match-episode-length",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "AR only (default on): set num_output_frames to each episode's action "
            "length (after temporal_stride) instead of the fixed --num-output-frames. "
            "Pass --no-match-episode-length for a fixed --num-output-frames (e.g. 185)."
        ),
    )
    p.add_argument("--num-output-frames", type=int, default=185, help="AR total frames")
    p.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help=f"Pixel frames per chunk (default: the model's native chunk, {DEFAULT_CHUNK_SIZE} for state_t 24)",
    )
    p.add_argument("--chunk-overlap", type=int, default=1)
    p.add_argument("--num-conditional-frames", type=int, default=1)
    p.add_argument("--guidance", type=float, default=3.0)
    p.add_argument("--num-steps", type=int, default=35)
    p.add_argument("--seed", type=int, default=0, help="Generation seed (shared by all samples)")
    p.add_argument("--negative-prompt", type=str, default=_DEFAULT_NEGATIVE_PROMPT)
    p.add_argument(
        "--context-parallel-size",
        type=int,
        default=1,
        help="Model context-parallel size (prefer data-parallel for multi-GPU quality).",
    )
    # Independent per-GPU workers (shell sets CUDA_VISIBLE_DEVICES per process).
    p.add_argument(
        "--data-parallel",
        action="store_true",
        help="Enable episode sharding across independent single-GPU workers",
    )
    p.add_argument("--data-parallel-rank", type=int, default=None)
    p.add_argument("--data-parallel-world-size", type=int, default=0)
    p.add_argument(
        "--dp-ready-timeout-s",
        type=float,
        default=1800.0,
        help="Seconds non-leader DP workers wait for leader manifest marker",
    )

    p.add_argument("--save-dir", type=str, required=True)
    p.add_argument("--no-comparison", action="store_true")
    p.add_argument(
        "--video-style",
        type=str,
        default="premium",
        choices=["premium", "legacy"],
        help=(
            "Saved video look. premium = black-bg action skeleton + dark "
            "Prediction|Ground truth|Action video comparison (matches restyled "
            "iter_68000 n400 dump); legacy = plain dark label bars."
        ),
    )
    return p.parse_args()


def _resolve_data_parallel(args: argparse.Namespace) -> tuple[int, int]:
    """Return (dp_rank, dp_world_size) for independent per-GPU episode sharding."""
    world = int(args.data_parallel_world_size or 0)
    rank = args.data_parallel_rank
    if args.data_parallel or world > 1 or rank is not None:
        if world <= 0:
            world = int(os.environ.get("WORLD_SIZE") or os.environ.get("LOCAL_WORLD_SIZE") or 1)
        if rank is None:
            rank = int(os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or 0)
        rank = int(rank)
        world = max(1, int(world))
        if rank < 0 or rank >= world:
            raise ValueError(f"Invalid data-parallel rank={rank} world_size={world}")
        return rank, world
    return 0, 1


def _ssim_gray(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-frame SSIM on luma; ``a``/``b`` are ``[T, H, W]`` float in [0, 255]."""
    import torch.nn.functional as F

    g = torch.exp(-((torch.arange(11, dtype=torch.float32) - 5) ** 2) / (2 * 1.5**2))
    g = g / g.sum()
    win = (g[:, None] @ g[None, :])[None, None]
    a, b = a[:, None], b[:, None]
    mu_a, mu_b = F.conv2d(a, win), F.conv2d(b, win)
    var_a = F.conv2d(a * a, win) - mu_a**2
    var_b = F.conv2d(b * b, win) - mu_b**2
    cov = F.conv2d(a * b, win) - mu_a * mu_b
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ssim = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2))
    return ssim.flatten(1).mean(1)


def _psnr(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    mse = ((pred - gt) ** 2).flatten(1).mean(1).clamp_min(1e-10)
    return 10 * torch.log10(255.0**2 / mse)


def video_metrics(pred_thwc, gt_cthw: torch.Tensor, skip: int = 1, cells: dict | None = None) -> dict | None:
    """PSNR / SSIM of uint8 prediction ``[T,H,W,C]`` vs GT ``[C,T,H,W]``, skipping condition frames.

    ``cells`` (view → ``(y0, y1, x0, x1)``) adds the PSNR of each multi-view canvas cell.
    """
    pred = torch.as_tensor(pred_thwc).float()
    gt = gt_cthw.permute(1, 2, 3, 0).float()
    t = min(pred.shape[0], gt.shape[0])
    if t <= skip or pred.shape[1:] != gt.shape[1:]:
        return None
    pred, gt = pred[skip:t], gt[skip:t]
    psnr = _psnr(pred, gt)
    luma = torch.tensor([0.299, 0.587, 0.114])
    ssim = torch.cat([_ssim_gray(pred[i : i + 16] @ luma, gt[i : i + 16] @ luma) for i in range(0, len(pred), 16)])
    out = {
        "frames": int(t - skip),
        "psnr": float(psnr.mean()),
        "ssim": float(ssim.mean()),
        "psnr_per_frame": [round(float(x), 3) for x in psnr],
    }
    if cells:
        out["psnr_per_view"] = {
            view: float(_psnr(pred[:, y0:y1, x0:x1], gt[:, y0:y1, x0:x1]).mean())
            for view, (y0, y1, x0, x1) in cells.items()
        }
    return out


def _wait_for_path(
    path: Path,
    *,
    timeout_s: float = 1800.0,
    poll_s: float = 1.0,
    desc: str = "path",
) -> bool:
    """CPU/filesystem wait (no NCCL). Returns False on timeout."""
    deadline = time.time() + max(1.0, float(timeout_s))
    while time.time() < deadline:
        if path.is_file() and path.stat().st_size > 0:
            return True
        time.sleep(poll_s)
    log.error(f"Timed out waiting for {desc}: {path}")
    return False


def _run_one(
    pipe,
    *,
    mode: str,
    rgb_path: str,
    action: torch.Tensor,
    action_video_path: str,
    action_video_cthw: torch.Tensor | None = None,
    prompt: str,
    resolution: tuple[int, int],
    chunk_size: int,
    chunk_overlap: int,
    num_output_frames: int | None,
    num_conditional_frames: int,
    guidance: float,
    seed: int,
    num_steps: int,
    fps: float,
    negative_prompt: str,
    temporal_stride: int = 1,
    action_pingpong: bool = False,
    rgb_first_frame_only: bool = False,
    rgb_views: list[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Returns (pred_btchw, gt_rgb_cthw_for_save, action_video_cthw_for_save).

    ``rgb_views`` (head, left wrist, right wrist mp4s) replaces ``rgb_path`` with the packed
    multi-view canvas for both the condition frames and the ground truth.
    """
    stride = max(int(temporal_stride), 1)

    def load_rgb(num_frames: int | None) -> torch.Tensor:
        if rgb_views:
            return load_multiview_video(
                rgb_views, num_frames=num_frames, resolution=resolution, temporal_stride=stride
            )
        return load_rgb_video(
            rgb_path, num_frames=num_frames, resolution=resolution, start_frame=0, temporal_stride=stride
        )

    action_video_full = action_video_cthw
    if action_video_full is None and action_video_path and Path(action_video_path).is_file():
        action_video_full = load_rgb_video(
            action_video_path,
            num_frames=None,
            resolution=resolution,
            start_frame=0,
            temporal_stride=stride,
        )
    if rgb_first_frame_only and not rgb_views:
        gt_rgb_full = load_rgb_first_frame(rgb_path, resolution=resolution)
    else:
        gt_rgb_full = load_rgb(None)
    if action_pingpong:
        # Same time-axis concat as the action: forward + reverse-without-last.
        gt_rgb_full = pingpong_concat_time(gt_rgb_full, time_dim=1)
        if action_video_full is not None:
            action_video_full = pingpong_concat_time(action_video_full, time_dim=1)

    frames_to_extract = 4 * (num_conditional_frames - 1) + 1
    if mode == "single":
        if rgb_first_frame_only:
            rgb = gt_rgb_full
            cond = rgb[:, : min(frames_to_extract, rgb.shape[1])].clone()
        else:
            rgb = load_rgb(chunk_size)
            cond = rgb[:, :frames_to_extract].clone()
        # The generated length follows ``video``; a first-frame-only input must still span the chunk.
        video = rgb.new_zeros((rgb.shape[0], chunk_size, *rgb.shape[2:]))
        video[:, : cond.shape[1]] = cond
        action_chunk = pad_or_trim_action(action, chunk_size)
        action_video = None
        if action_video_full is not None:
            av = action_video_full
            if av.shape[1] < chunk_size:
                pad = av[:, -1:].repeat(1, chunk_size - av.shape[1], 1, 1)
                action_video = torch.cat([av, pad], dim=1)
            else:
                action_video = av[:, :chunk_size]
        sample = ActionInjectSample(
            video=video,
            action=action_chunk,
            prompt=prompt,
            action_video=action_video,
            fps=fps,
        )
        out = pipe.generate_chunk(
            sample,
            guidance=guidance,
            seed=seed,
            num_steps=num_steps,
            num_conditional_frames=num_conditional_frames,
            negative_prompt=negative_prompt,
        )
        gt_save = gt_rgb_full[:, :chunk_size]
        act_save = None if action_video_full is None else action_video_full[:, :chunk_size]
        return out, gt_save, act_save

    init = gt_rgb_full[:, : min(frames_to_extract, gt_rgb_full.shape[1])]
    out = pipe.generate_ar(
        prompt=prompt,
        action=action,
        init_video_cthw=init,
        action_video=action_video_full,
        num_output_frames=num_output_frames,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        guidance=guidance,
        seed=seed,
        num_steps=num_steps,
        num_conditional_frames=num_conditional_frames,
        fps=fps,
        negative_prompt=negative_prompt,
        resolution=resolution,
    )
    return out, gt_rgb_full, action_video_full


def main() -> None:
    args = _parse_args()
    h, w = [int(x) for x in args.resolution.split(",")]
    resolution = (h, w)
    mix_weights = tuple(float(x) for x in args.mix_weights.split(","))
    if args.dataset == ActionDatasetSource.MIXED.value and len(mix_weights) != len(DEFAULT_MIX_NAMES):
        raise ValueError(
            f"--mix-weights needs {len(DEFAULT_MIX_NAMES)} floats for "
            f"{list(DEFAULT_MIX_NAMES)}, got {mix_weights}"
        )

    dp_rank, dp_world = _resolve_data_parallel(args)
    if dp_world > 1 and int(args.context_parallel_size) > 1:
        raise ValueError(
            "Do not combine data-parallel workers with context-parallel>1; "
            "use PARALLEL_MODE=data (CP=1) for multi-GPU quality."
        )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = save_dir / "dataset_sample_manifest.jsonl"
    ready_marker = save_dir / "dp_ready.json"
    results_path = save_dir / "inference_results.jsonl"

    experiment, config_file = resolve_experiment_and_config(
        args.job_dir,
        experiment=args.experiment,
        config_file=args.config_file,
    )
    ckpt = args.ckpt_path or resolve_checkpoint_dir(args.job_dir, args.iteration)
    log.info(f"Using checkpoint: {ckpt}")
    log.info(f"experiment={experiment} config_file={config_file}")
    if dp_world > 1:
        log.info(f"Data-parallel worker rank={dp_rank}/{dp_world}")

    robocasa_dirs = [x.strip() for x in args.robocasa_dirs.split(",") if x.strip()]

    pipe = ActionInjectInferencePipeline(
        ckpt_path=ckpt,
        experiment_name=experiment,
        config_file=config_file,
        context_parallel_size=args.context_parallel_size,
        experiment_opts=[f"model.config.state_t={int(args.state_t)}"] if args.state_t else None,
    )
    chunk_size = int(args.chunk_size or pipe.chunk_size)
    log.info(f"chunk_size={chunk_size} (model native {pipe.chunk_size})")
    index_num_frames = chunk_size

    is_leader = (dp_rank == 0) if dp_world > 1 else pipe.is_rank0
    is_writer = True if dp_world > 1 else pipe.is_rank0

    if is_leader:
        if dp_world > 1:
            ready_marker.unlink(missing_ok=True)
            results_path.write_text("", encoding="utf-8")
            for p in save_dir.glob("dp_rank*_done.json"):
                p.unlink(missing_ok=True)

        if args.sample_dir:
            sample_dirs = [p.strip() for p in str(args.sample_dir).replace(";", ",").split(",") if p.strip()]
            specs = [load_sample_dir(p) for p in sample_dirs]
            write_episode_manifest(specs, manifest_path)
            log.info(f"Loaded {len(specs)} example sample(s) → {manifest_path}")
        elif args.episode_manifest:
            src_manifest = Path(args.episode_manifest)
            if not src_manifest.is_file():
                raise FileNotFoundError(f"--episode-manifest not found: {src_manifest}")
            manifest_path.write_text(src_manifest.read_text(encoding="utf-8"), encoding="utf-8")
            log.info(f"Using fixed episode manifest: {src_manifest} → {manifest_path}")
        else:
            ds = build_action_inference_dataset(
                args.dataset,
                num_frames=index_num_frames,
                video_size=resolution,
                rsw_dir=args.rsw_dir,
                agibot_labels_pkl=args.agibot_labels_pkl,
                agibot_dexhand_labels_jsonl=args.agibot_dexhand_labels_jsonl,
                robomind_pairs_jsonl=args.robomind_pairs_jsonl,
                libero_dir=args.libero_dir,
                libero_ood_dir=args.libero_ood_dir,
                ur5_robosuite_ood_dir=args.ur5_robosuite_ood_dir,
                iiwa_robosuite_ood_dir=args.iiwa_robosuite_ood_dir,
                maniskill_dir=args.maniskill_dir,
                oven_platephase_dir=args.oven_platephase_dir,
                robotwin_pairs_jsonl=args.robotwin_pairs_jsonl,
                robocasa_dirs=robocasa_dirs,
                mix_weights=mix_weights,
                verify_files=args.verify_files,
                load_action_video=True,
            )
            specs = sample_action_episodes(
                ds,
                num_samples=args.num_samples,
                seed=args.sample_seed,
                source_name=args.dataset,
            )
            write_episode_manifest(specs, manifest_path)
            del ds
        if dp_world > 1:
            ready_marker.write_text(
                json.dumps(
                    {
                        "dp_world": dp_world,
                        "manifest": str(manifest_path),
                        "dataset": args.dataset,
                        "num_samples": int(args.num_samples),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

    if not is_leader and dp_world > 1:
        if not _wait_for_path(
            ready_marker, timeout_s=args.dp_ready_timeout_s, desc="dp_ready"
        ):
            raise RuntimeError(f"Rank failed waiting for ready marker: {ready_marker}")

    if args.context_parallel_size > 1:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    specs_all = read_episode_manifest(manifest_path)
    if dp_world > 1:
        specs = [s for i, s in enumerate(specs_all) if i % dp_world == dp_rank]
        log.info(
            f"DP shard: rank {dp_rank}/{dp_world} handles {len(specs)}/{len(specs_all)} episodes"
        )
    else:
        specs = specs_all

    results_rows: list[dict] = []
    for i, spec in enumerate(specs):
        temporal_stride = max(int(getattr(spec, "temporal_stride", 1) or 1), 1)
        fps = float(args.fps) if args.fps is not None else float(getattr(spec, "fps", 0.0) or 15.0)
        save_fps = float(args.save_fps) if args.save_fps is not None else fps

        log.info(
            f"[dp{dp_rank} {i + 1}/{len(specs)}] {spec.name}  source={spec.source}  "
            f"stride={temporal_stride}  fps={fps:g}  prompt={spec.prompt!r}  rgb={spec.rgb_path}"
        )
        try:
            action = load_episode_action(spec)
        except Exception as e:
            log.error(f"Skip {spec.name}: failed to load action ({e})")
            continue
        if not Path(spec.rgb_path).is_file():
            log.error(f"Skip {spec.name}: missing rgb {spec.rgb_path}")
            continue

        t_fwd = int(action.shape[0])
        if args.action_pingpong:
            action = pingpong_concat_time(action, time_dim=0)
            log.info(
                f"action pingpong: T_fwd={t_fwd} → T={int(action.shape[0])} "
                f"(forward + reverse-without-last)"
            )

        if args.mode == "ar":
            if args.match_episode_length:
                num_output_frames = max(int(action.shape[0]), chunk_size)
            else:
                num_output_frames = int(args.num_output_frames)
        else:
            num_output_frames = None

        try:
            action_video_cthw = resolve_action_video_cthw(
                spec,
                resolution=resolution,
                temporal_stride=temporal_stride,
                render_mode=args.render_action_video,
            )
        except Exception as e:
            log.error(f"Skip {spec.name}: failed to resolve action video ({e})")
            continue
        if action_video_cthw is None and args.render_action_video != "never":
            log.error(f"Skip {spec.name}: action video missing and online render failed")
            continue

        rgb_views = [v["rgb"] for v in spec.views.values()] if spec.layout else None
        out, gt_save, act_save = _run_one(
            pipe,
            mode=args.mode,
            rgb_path=spec.rgb_path,
            action=action,
            action_video_path="" if rgb_views else spec.action_video_path,
            action_video_cthw=action_video_cthw,
            prompt=spec.prompt,
            resolution=resolution,
            chunk_size=chunk_size,
            chunk_overlap=args.chunk_overlap,
            num_output_frames=num_output_frames,
            num_conditional_frames=args.num_conditional_frames,
            guidance=args.guidance,
            seed=args.seed,
            num_steps=args.num_steps,
            fps=fps,
            negative_prompt=args.negative_prompt,
            temporal_stride=temporal_stride,
            action_pingpong=bool(args.action_pingpong),
            rgb_first_frame_only=bool(args.rgb_first_frame_only)
            or Path(spec.rgb_path).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"},
            rgb_views=rgb_views,
        )

        transplanted_rgb = None
        action_rgb_path = str(getattr(spec, "action_rgb_path", "") or "").strip()
        if action_rgb_path and Path(action_rgb_path).is_file():
            transplanted_rgb = load_rgb_video(
                action_rgb_path,
                num_frames=None,
                resolution=resolution,
                start_frame=0,
                temporal_stride=temporal_stride,
            )
            if args.action_pingpong:
                transplanted_rgb = pingpong_concat_time(transplanted_rgb, time_dim=1)

        gt_rgb_path = str(getattr(spec, "gt_rgb_path", "") or "").strip()
        if gt_rgb_path and Path(gt_rgb_path).is_file() and Path(gt_rgb_path).resolve() != Path(spec.rgb_path).resolve():
            gt_save = load_rgb_video(
                gt_rgb_path,
                num_frames=None,
                resolution=resolution,
                start_frame=0,
                temporal_stride=temporal_stride,
            )
            if args.action_pingpong:
                gt_save = pingpong_concat_time(gt_save, time_dim=1)

        if is_writer:
            frames = tensor_to_uint8_video(out)
            metrics = None
            if args.metrics and gt_save is not None:
                cells = multiview_cells(*resolution) if rgb_views else None
                metrics = video_metrics(frames, gt_save, skip=args.num_conditional_frames, cells=cells)
                if metrics:
                    per_view = "".join(
                        f" {view}={value:.3f}" for view, value in metrics.get("psnr_per_view", {}).items()
                    )
                    log.info(
                        f"{spec.name}: PSNR={metrics['psnr']:.3f} SSIM={metrics['ssim']:.4f} "
                        f"over {metrics['frames']} frames{' | PSNR per view:' + per_view if per_view else ''}"
                    )
            written = save_labeled_inference_videos(
                save_dir=save_dir,
                save_name=spec.name,
                save_fps=save_fps,
                pred_thwc=frames,
                gt_rgb_cthw=gt_save,
                action_video_cthw=act_save,
                transplanted_rgb_cthw=transplanted_rgb,
                write_comparison=not args.no_comparison,
                style=args.video_style,
            )
            results_rows.append(
                {
                    "name": spec.name,
                    "source": spec.source,
                    "index": spec.index,
                    "prompt": spec.prompt,
                    "rgb_path": spec.rgb_path,
                    "action_rgb_path": getattr(spec, "action_rgb_path", ""),
                    "action_path": spec.action_path,
                    "parquet_path": spec.parquet_path,
                    "hdf5_path": spec.hdf5_path,
                    "action_video_path": spec.action_video_path,
                    "temporal_stride": temporal_stride,
                    "action_pingpong": bool(args.action_pingpong),
                    "action_T_fwd": t_fwd,
                    "action_T": int(action.shape[0]),
                    "view_difficulty": getattr(spec, "view_difficulty", ""),
                    "camera_preset": getattr(spec, "camera_preset", ""),
                    "dp_rank": dp_rank,
                    "task_name": getattr(spec, "task_name", ""),
                    "layout": spec.layout,
                    "fps": fps,
                    "num_frames": int(frames.shape[0]),
                    "metrics": metrics,
                    "outputs": written,
                }
            )

    if is_writer and results_rows:
        if dp_world > 1:
            # Append per-rank rows; leader may have truncated the shared file earlier.
            with open(results_path, "a", encoding="utf-8") as f:
                for row in results_rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            done_path = save_dir / f"dp_rank{dp_rank}_done.json"
            done_path.write_text(
                json.dumps({"rank": dp_rank, "ok": len(results_rows)}, indent=2) + "\n",
                encoding="utf-8",
            )
            log.info(
                f"DP rank {dp_rank}: wrote {len(results_rows)} results → {results_path}"
            )
        else:
            with open(results_path, "w", encoding="utf-8") as f:
                for row in results_rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            log.info(f"Wrote {results_path} ({len(results_rows)} ok / {len(specs)} sampled)")

    pipe.cleanup()


if __name__ == "__main__":
    main()
