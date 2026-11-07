# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export RoboDojo episodes as ``--sample-dir`` folders for inference.

Reads the training pairs JSONL produced by the RoboDojo action-video render
(``uniembodied_action_video_rgb_pairs.jsonl``) and writes one folder per episode in
the same layout as ``examples/samples/robotwin_*``::

    <out>/robodojo_<scene>_ep<NNNNNN>/
      first_frame.png    RGB condition (first frame of the clip)
      rgb.mp4            ground-truth RGB at the source rate (25 fps, lossless RGB H.264)
      action.npy         [T, 28] unified action at the source rate (Joint14 image_joint → 28-D)
      prompt.txt         instruction (pairs ``lang``)
      meta.json          source=robodojo, temporal_stride, model_fps, camera K /
                         T_world_camera_opencv, arm_bases (online ARX-X5 skeleton render)
      action_video.mp4   only with --with-action-video: the training skeleton mp4 (lossless copy)

Inference renders the action video online from ``action.npy`` + meta geometry at the source
rate and then applies ``temporal_stride``, exactly like the training render (whole-episode
25 fps skeleton, frames picked by the dataset). The clip starts at ``start_frame`` (Franka
demonstration prefix of imitate_sorting_sequence); ``temporal_stride`` is the largest value
in ``[min_stride, max_stride]`` whose ``num_frames``-frame window fits (else ``min_stride``),
matching ``RoboDojoActionDataset``.

Examples::

    # one episode per scene (24 folders)
    python examples/pack_robodojo_samples.py --out-dir outputs/robodojo_samples

    # specific episodes, first 93 model frames only, keep the training action video too
    python examples/pack_robodojo_samples.py --episodes 250,1050 --max-frames 93 \\
      --with-action-video --out-dir outputs/robodojo_samples
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
from decord import VideoReader, cpu
from PIL import Image

from cosmos_predict2._src.predict2.datasets.local_datasets.action_unified_layout import (
    to_unified_bimanual_action,
)
from cosmos_predict2._src.predict2.datasets.local_datasets.action_video_render import (
    BACKEND_ROOT,
    ensure_backend_import_path,
)

DEFAULT_PAIRS_JSONL = (
    "/media/datasets/Sim20_dataset/UniEmbodied-Bench/"
    "policy_mix24_clean2400_sdk_ee20_25hz_instruction_20261029_v2_action_video/"
    "uniembodied_action_video_rgb_pairs.jsonl"
)
NPZ_ACTION_KEY = "image_joint"
SOURCE_ARM_DOF = 6
DECODE_CHUNK = 64


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pairs-jsonl", default=DEFAULT_PAIRS_JSONL)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--episodes", default="", help="Comma-separated episode_index list")
    p.add_argument("--scenes", default="", help="Comma-separated task_name filter")
    p.add_argument("--per-scene", type=int, default=1, help="Episodes per scene when --episodes is empty")
    p.add_argument("--random", action="store_true", help="Random episodes per scene instead of the middle ones")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-frames", type=int, default=93, help="Model window used to fit the stride")
    p.add_argument("--max-stride", type=int, default=2)
    p.add_argument("--min-stride", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=0, help="Cap exported model frames (0 = whole episode)")
    p.add_argument(
        "--with-action-video",
        action="store_true",
        help="Also copy the pre-rendered training skeleton mp4 (default: render online at inference)",
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def fit_stride(usable: int, num_frames: int, max_stride: int, min_stride: int) -> int:
    for s in range(max_stride, min_stride - 1, -1):
        if usable >= num_frames * s:
            return s
    return min_stride


def _episode_index(item: dict) -> int:
    if item.get("episode_index") is not None:
        return int(item["episode_index"])
    return int(Path(item["rgb"]).stem.split("_")[-1])


def select_items(items: list[dict], args: argparse.Namespace) -> list[dict]:
    if args.episodes.strip():
        wanted = [int(x) for x in args.episodes.split(",") if x.strip()]
        by_ep = {_episode_index(it): it for it in items}
        missing = [e for e in wanted if e not in by_ep]
        if missing:
            raise KeyError(f"episodes not in {args.pairs_jsonl}: {missing}")
        return [by_ep[e] for e in wanted]
    scenes = {s.strip() for s in args.scenes.split(",") if s.strip()}
    by_scene: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        if not scenes or it.get("task_name") in scenes:
            by_scene[str(it.get("task_name"))].append(it)
    rng = random.Random(args.seed)
    out: list[dict] = []
    for scene in sorted(by_scene):
        group = by_scene[scene]
        k = min(args.per_scene, len(group))
        if args.random:
            out.extend(rng.sample(group, k))
        else:
            mid = len(group) // 2
            lo = max(0, min(mid - k // 2, len(group) - k))
            out.extend(group[lo:lo + k])
    return out


def _encode_lossless(vr: VideoReader, ids: list[int], path: Path, fps: float) -> np.ndarray:
    """Pipe decoded frames into lossless RGB H.264; returns the first frame (HWC uint8)."""
    h, w = vr[0].shape[:2]
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
         "-r", f"{fps:g}", "-i", "-", "-c:v", "libx264rgb", "-crf", "0", "-preset", "veryfast",
         "-pix_fmt", "rgb24", str(path)],
        stdin=subprocess.PIPE,
    )
    first = None
    for i in range(0, len(ids), DECODE_CHUNK):
        batch = vr.get_batch(ids[i:i + DECODE_CHUNK]).asnumpy().astype(np.uint8)
        if first is None:
            first = batch[0].copy()
        proc.stdin.write(batch.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed writing {path}")
    return first


def _render_geometry(item: dict, rgb_frame_index0: int) -> dict:
    """Camera / arm-base geometry exactly as the training render reads it from the raw HDF5."""
    ensure_backend_import_path(BACKEND_ROOT / "robodojo")
    import robodojo_data as ud

    geom = ud.load_episode_geometry(
        item["hdf5"], ud.CAMERA_KEYS[item.get("camera") or "cam_high"], int(rgb_frame_index0),
        default_bases=ud.load_default_arm_bases(),
    )
    return {
        "K": np.asarray(geom["intrinsic"]).tolist(),
        "T_world_camera_opencv": np.asarray(geom["T_world_camera_opencv"]).tolist(),
        "arm_bases": {side: np.asarray(m).tolist() for side, m in geom["bases"].items()},
        "base_source": geom["base_source"],
    }


def pack_episode(item: dict, out_root: Path, args: argparse.Namespace) -> Path:
    ep = _episode_index(item)
    scene = str(item.get("task_name") or "robodojo")
    name = f"robodojo_{scene}_ep{ep:06d}"
    out = out_root / name
    if (out / "meta.json").is_file() and not args.overwrite:
        return out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    length = int(item["frame_nums"])
    start = max(0, int(item.get("start_frame") or 0))
    stride = fit_stride(length - start, args.num_frames, args.max_stride, args.min_stride)
    end = length if args.max_frames <= 0 else min(length, start + args.max_frames * stride)
    ids = list(range(start, end))
    src_fps = float(item.get("fps") or 25.0)

    rgb_vr = VideoReader(item["rgb"], ctx=cpu(0), num_threads=2)
    if len(rgb_vr) != length:
        raise RuntimeError(f"{name}: rgb frames {len(rgb_vr)} != meta {length}")
    first = _encode_lossless(rgb_vr, ids, out / "rgb.mp4", src_fps)
    Image.fromarray(first).save(out / "first_frame.png")
    del rgb_vr
    if args.with_action_video:
        av_vr = VideoReader(item["action_video"], ctx=cpu(0), num_threads=2)
        if len(av_vr) != length:
            raise RuntimeError(f"{name}: action_video frames {len(av_vr)} != meta {length}")
        _encode_lossless(av_vr, ids, out / "action_video.mp4", src_fps)
        del av_vr

    with np.load(item["actions"], allow_pickle=False) as d:
        joint14 = np.asarray(d[NPZ_ACTION_KEY], dtype=np.float32)
        rgb_frame_index0 = int(np.asarray(d["rgb_frame_index"])[0]) if "rgb_frame_index" in d.files else 0
    if joint14.shape != (length, 14):
        raise ValueError(f"{name}: {NPZ_ACTION_KEY} shape {joint14.shape} != ({length}, 14)")
    unified = to_unified_bimanual_action(
        joint14[ids], num_arms=2, include_gripper=True, arm_dof=SOURCE_ARM_DOF
    ).astype(np.float32)
    np.save(out / "action.npy", unified)

    prompt = str(item.get("lang") or item.get("instruction") or scene).strip()
    (out / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")
    meta = {
        "source": "robodojo",
        "name": name,
        "prompt": prompt,
        "frames": len(ids),
        "action_dim": 28,
        "already_unified": True,
        "num_arms": 2,
        "temporal_stride": stride,
        "model_fps": src_fps / stride,
        "width": int(first.shape[1]),
        "height": int(first.shape[0]),
        "rgb": "first_frame.png",
        "rgb_video": "rgb.mp4",
        "action": "action.npy",
        "task_name": scene,
        "episode_index": ep,
        "official": bool(item.get("official", False)),
        **_render_geometry(item, rgb_frame_index0),
        "hdf5": item.get("hdf5", ""),
        "source_fps": src_fps,
        "source_start_frame": start,
        "source_frame_range": [ids[0], ids[-1]],
        "source_length": length,
        "source_rgb": item["rgb"],
        "source_action_video": item["action_video"],
        "source_actions": item["actions"],
        "action_key": NPZ_ACTION_KEY,
    }
    if args.with_action_video:
        meta["action_video"] = "action_video.mp4"
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def main() -> None:
    args = _parse_args()
    items = [json.loads(line) for line in open(args.pairs_jsonl, encoding="utf-8") if line.strip()]
    selected = select_items(items, args)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    dirs = []
    for i, item in enumerate(selected):
        d = pack_episode(item, out_root, args)
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        print(f"[{i + 1}/{len(selected)}] {d.name}: frames={meta['frames']} stride={meta['temporal_stride']} "
              f"start={meta['source_start_frame']} fps={meta['model_fps']:g}", flush=True)
        dirs.append(str(d))
    (out_root / "sample_dirs.txt").write_text(",".join(dirs) + "\n", encoding="utf-8")
    print(f"Wrote {len(dirs)} sample dirs → {out_root / 'sample_dirs.txt'}")


if __name__ == "__main__":
    main()
