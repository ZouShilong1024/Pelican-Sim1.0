#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert RoboDojo (dual ARX-X5) episodes into a Pelican-Sim fine-tuning set.

Two sources:

  --lerobot ROOT   LeRobot v3 export (RoboDojo layout: meta/, data/, videos/,
                   SAMPLE_MAPPING.jsonl). Camera geometry is read from each episode's original
                   HDF5 listed in SAMPLE_MAPPING.jsonl.
  --hdf5 PATH      Official RoboDojo HDF5 files: one file, or a directory searched recursively
                   (e.g. RoboDojo/<task>/arx_x5/data/episode_*.hdf5).

Per episode, written to ``<out-dir>/<task>_ep<NNNNNN>/``::

    rgb.mp4            head camera, 640x480, source fps (25)
    action_video.mp4   white-background ARX-X5 skeleton rendered from action.npy, same frames
    action.npy         [T, 28] unified action: per arm 6 joints + gripper (1 = open)
    first_frame.png    first RGB frame
    prompt.txt         instruction
    meta.json          inference sample metadata (camera K / extrinsic, arm bases, stride)

and ``<out-dir>/episodes.jsonl`` (one line per episode) for ``finetune.sh``. Every folder is
also a valid ``--sample-dir`` for ``examples/action_inject_inference_from_dataset.py``.

With ``--multiview`` each folder also gets the two wrist cameras::

    rgb_cam_left_wrist.mp4 / rgb_cam_right_wrist.mp4                    wrist RGB, same frames
    action_video_cam_left_wrist.mp4 / action_video_cam_right_wrist.mp4  wrist skeletons
    camera_tracks.npz  per-frame OpenCV wrist pose [T, 4, 4] per view (online rendering)

and meta.json carries ``layout: mv_fill`` plus a ``views`` table; the manifest lines get
``views`` for ``MULTIVIEW=1 ./finetune.sh``. The wrist cameras move with the arms, so each wrist
skeleton frame is projected with the pose of that RGB frame.

Row t of action.npy is the joint state when RGB frame t was captured. In the official HDF5 the
image index lags the state by one step, so joints = ``state[:-1]`` and images = ``colors[1:]``.
For ``imitate_sorting_sequence`` the scripted Franka demonstration at the start is cut (the clip
starts 10 frames after the ARX arms first move).

    python scripts/prepare_robodojo.py --hdf5 /data/RoboDojo --scenes push_T,fold_clothes \\
        --per-scene 20 --out-dir data/robodojo_ft --workers 16
    python scripts/prepare_robodojo.py --lerobot /data/policy_mix24_lerobot --out-dir data/robodojo_ft
    python scripts/prepare_robodojo.py --lerobot /data/policy_mix24_lerobot --multiview \\
        --out-dir data/robodojo_mv_ft
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from cosmos_predict2._src.predict2.datasets.local_datasets.action_unified_layout import (  # noqa: E402
    to_unified_bimanual_action,
)
from cosmos_predict2._src.predict2.datasets.local_datasets.action_video_render import (  # noqa: E402
    BACKEND_ROOT,
    ensure_backend_import_path,
)

ensure_backend_import_path(BACKEND_ROOT / "robodojo")
import batch_render_action_video_robodojo as ue  # noqa: E402
import robodojo_data as ud  # noqa: E402

WIDTH, HEIGHT = 640, 480
NUM_FRAMES = 93
MV_NUM_FRAMES = 13
ACTION_CRF, RGB_CRF = 23, 18
WRIST_VIEWS = ("cam_left_wrist", "cam_right_wrist")

_RGB_MARKER = b"XPL-RGB1"


# ---------------------------------------------------------------------------
# Sources: each returns (joint14 [T,14], rgb frame getter, geometry, info)
# ---------------------------------------------------------------------------
def _has_rgb_marker(buf: bytes) -> bool:
    """True if the JPEG header carries the XPolicyLab ``XPL-RGB1`` COM segment (standard RGB JPEG)."""
    if len(buf) < 4 or buf[0] != 0xFF or buf[1] != 0xD8:
        return False
    off = 2
    while off + 4 <= len(buf):
        if buf[off] != 0xFF:
            return False
        marker = buf[off + 1]
        if marker == 0xFF:
            off += 1
            continue
        if marker in (0x01, 0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            off += 2
            continue
        if marker in (0x00, 0xDA):
            return False
        length = (buf[off + 2] << 8) | buf[off + 3]
        if marker == 0xFE and buf[off + 4 : off + 2 + length] == _RGB_MARKER:
            return True
        off += 2 + length
    return False


def decode_robodojo_jpeg(buf: bytes) -> np.ndarray:
    """RoboDojo ``colors`` entry → HWC uint8 RGB (legacy buffers store reversed channels)."""
    import cv2

    buf = bytes(buf).rstrip(b"\0")
    img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("failed to decode image")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if _has_rgb_marker(buf) else img


def _geometry(hdf5_path: str, frame_index: int) -> dict:
    g = ud.load_episode_geometry(hdf5_path, "cam_head", frame_index, default_bases=ud.load_default_arm_bases())
    return {"K": g["intrinsic"], "T_world_camera_opencv": g["T_world_camera_opencv"], "bases": g["bases"],
            "base_source": g["base_source"]}


def _wrist_track(hdf5_path: str, view: str, rgb_index: np.ndarray) -> dict:
    """Wrist intrinsics + the OpenCV pose of each RGB frame (``rgb_index`` = HDF5 image rows)."""
    track = ud.load_camera_track(hdf5_path, ud.CAMERA_KEYS[view])
    poses = track["T_world_camera_opencv"]
    return {"K": track["intrinsic"], "T": poses[np.clip(rgb_index, 0, len(poses) - 1)]}


def _sequential(reader, label: str) -> Callable[[int], np.ndarray]:
    state = {"next": 0, "frame": None}

    def rgb(t: int) -> np.ndarray:
        # Sequential reader: frames are requested in non-decreasing order.
        while state["next"] <= t:
            state["frame"] = reader.read()
            if state["frame"] is None:
                raise RuntimeError(f"{label}: video ended before frame {t}")
            state["next"] += 1
        return state["frame"]

    return rgb


def load_hdf5_episode(path: str, multiview: bool = False):
    import h5py

    f = h5py.File(path, "r")
    if "state/left_arm_joint_states" not in f:
        f.close()
        raise ValueError(f"{path}: not the official RoboDojo schema (convert it through the LeRobot export)")
    joint14 = np.concatenate(
        [
            f["state/left_arm_joint_states"][()],
            f["state/left_ee_joint_states"][()].reshape(-1, 1),
            f["state/right_arm_joint_states"][()],
            f["state/right_ee_joint_states"][()].reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)[:-1]
    colors = f["vision/cam_head/colors"]
    fps = float(f["additional_info/frequency"][()]) if "additional_info/frequency" in f else ud.FPS
    instruction = f["instruction"][()]
    instruction = (instruction.decode() if isinstance(instruction, bytes) else str(instruction)).strip()

    def rgb_getter(view_colors):
        return lambda t: decode_robodojo_jpeg(view_colors[t + 1])

    p = Path(path)
    scene = p.parents[2].name if p.parent.name == "data" else p.parent.name
    info = {"scene": scene, "episode": int(p.stem.split("_")[-1]), "instruction": instruction, "fps": fps,
            "hdf5": str(p), "close": f.close, "wrists": {}}
    if multiview:
        rgb_index = np.arange(1, len(joint14) + 1)
        for view in WRIST_VIEWS:
            info["wrists"][view] = {"rgb": rgb_getter(f[f"vision/{ud.CAMERA_KEYS[view]}/colors"]),
                                    **_wrist_track(path, view, rgb_index)}
    return joint14, rgb_getter(colors), _geometry(path, 1), info


def _lerobot_reader(ref: ud.EpisodeRef):
    return ue._FfmpegFrameReader(
        ref.video_file, max(0.0, ref.from_timestamp - 0.5 / ud.FPS), ref.length, WIDTH, HEIGHT
    )


def load_lerobot_episode(ref: ud.EpisodeRef, wrist_refs: Optional[dict] = None):
    frames = ud.read_episode_frames(ref)
    joint14 = frames["image_joint"].astype(np.float32)
    readers = [_lerobot_reader(ref)]
    info = {"scene": ref.scene, "episode": ref.episode_index, "instruction": ref.instruction, "fps": ud.FPS,
            "hdf5": ref.hdf5, "wrists": {}}
    for view, wref in (wrist_refs or {}).items():
        wref = ud.EpisodeRef(**wref)
        readers.append(_lerobot_reader(wref))
        info["wrists"][view] = {"rgb": _sequential(readers[-1], f"episode {ref.episode_index} {view}"),
                                **_wrist_track(ref.hdf5, view, frames["rgb_frame_index"])}
    info["close"] = lambda: [r.close() for r in readers]
    rgb = _sequential(readers[0], f"episode {ref.episode_index}")
    return joint14, rgb, _geometry(ref.hdf5, int(frames["rgb_frame_index"][0])), info


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------
def _ffmpeg_cmd(path: str, fps: float, crf: int) -> list[str]:
    return [
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{WIDTH}x{HEIGHT}", "-r", f"{fps:g}", "-i", "-",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-threads", "1",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-f", "mp4", path,
    ]


def _encode(path: Path, n: int, producer: Callable[[int], np.ndarray], fps: float, crf: int) -> None:
    tmp = str(path) + ".part"
    rc, err = ue.rt._encode_to_ffmpeg(_ffmpeg_cmd(tmp, fps, crf), n, producer)
    if rc != 0:
        raise RuntimeError(f"ffmpeg failed for {path}: {err}")
    os.replace(tmp, path)


def demo_start_frame(joint14: np.ndarray) -> int:
    moved = np.where(np.abs(joint14 - joint14[0]).max(axis=1) > 1e-3)[0]
    first = int(moved[0]) if moved.size else 0
    return min(first + ue.DEMO_SETTLE_FRAMES, len(joint14) - 1)


def fit_stride(n: int, max_stride: int, num_frames: int = NUM_FRAMES) -> int:
    for s in range(max_stride, 0, -1):
        if n >= num_frames * s:
            return s
    return 1


def write_episode(task: dict) -> Optional[dict]:
    from PIL import Image

    out_root = Path(task["out_dir"])
    multiview = bool(task.get("multiview"))
    if task["kind"] == "hdf5":
        joint14, rgb, geom, info = load_hdf5_episode(task["path"], multiview)
    else:
        joint14, rgb, geom, info = load_lerobot_episode(ud.EpisodeRef(**task["ref"]), task.get("wrist_refs"))
    name = f"{info['scene']}_ep{info['episode']:06d}"
    out = out_root / name
    try:
        if (out / "meta.json").is_file() and not task["overwrite"]:
            done = json.loads((out / "meta.json").read_text(encoding="utf-8"))
            if done.get("views") or not multiview:
                return done
        out.mkdir(parents=True, exist_ok=True)

        start = demo_start_frame(joint14) if info["scene"] in ue.DEMO_PREFIX_SCENES else 0
        joint14 = joint14[start:]
        n, fps = len(joint14), float(info["fps"])
        if not info["instruction"]:
            raise ValueError(f"{name}: empty instruction")

        first = rgb(start)
        if first.shape != (HEIGHT, WIDTH, 3):
            raise ValueError(f"{name}: expected {HEIGHT}x{WIDTH} RGB, got {first.shape}")
        Image.fromarray(first).save(out / "first_frame.png")
        _encode(out / "rgb.mp4", n, lambda t: rgb(start + t), fps, RGB_CRF)

        skeleton = ue.render_action_frames(joint14, geom["bases"], geom["K"], geom["T_world_camera_opencv"])
        _encode(out / "action_video.mp4", n, lambda t: skeleton[t], fps, ACTION_CRF)

        views = {}
        if multiview:
            views["cam_high"] = {"rgb": "rgb.mp4", "action_video": "action_video.mp4"}
            tracks = {}
            for view, wrist in info["wrists"].items():
                _encode(out / f"rgb_{view}.mp4", n, lambda t, g=wrist["rgb"]: g(start + t), fps, RGB_CRF)
                tracks[view] = wrist["T"][start:]
                wrist_skeleton = ue.render_action_frames(joint14, geom["bases"], wrist["K"], tracks[view])
                _encode(out / f"action_video_{view}.mp4", n, lambda t: wrist_skeleton[t], fps, ACTION_CRF)
                views[view] = {"rgb": f"rgb_{view}.mp4", "action_video": f"action_video_{view}.mp4",
                               "K": np.asarray(wrist["K"]).tolist(), "camera_track": "camera_tracks.npz"}
            np.savez(out / "camera_tracks.npz", **tracks)

        action = to_unified_bimanual_action(joint14, num_arms=2, include_gripper=True, arm_dof=6)
        np.save(out / "action.npy", action.astype(np.float32))
        (out / "prompt.txt").write_text(info["instruction"] + "\n", encoding="utf-8")

        stride = fit_stride(n, task["stride"], MV_NUM_FRAMES if multiview else NUM_FRAMES)
        meta = {
            "source": "robodojo",
            "name": name,
            "prompt": info["instruction"],
            "task_name": info["scene"],
            "episode_index": info["episode"],
            "frames": n,
            "source_fps": fps,
            "temporal_stride": stride,
            "model_fps": fps / stride,
            "action_dim": 28,
            "already_unified": True,
            "num_arms": 2,
            "width": WIDTH,
            "height": HEIGHT,
            "rgb": "first_frame.png",
            "rgb_video": "rgb.mp4",
            "action": "action.npy",
            "action_video": "action_video.mp4",
            "K": np.asarray(geom["K"]).tolist(),
            "T_world_camera_opencv": np.asarray(geom["T_world_camera_opencv"]).tolist(),
            "arm_bases": {side: np.asarray(m).tolist() for side, m in geom["bases"].items()},
            "base_source": geom["base_source"],
            "hdf5": info["hdf5"],
            "source_start_frame": start,
        }
        if views:
            meta.update(layout="mv_fill", views=views)
        (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return meta
    finally:
        info["close"]()


def manifest_item(folder: Path, meta: dict, max_stride: int) -> dict:
    item = {
        "uuid": f"{meta['task_name']}/episode_{meta['episode_index']:06d}",
        "task_name": meta["task_name"],
        "rgb": str((folder / "rgb.mp4").resolve()),
        "action_video": str((folder / "action_video.mp4").resolve()),
        "action": str((folder / "action.npy").resolve()),
        "lang": meta["prompt"],
        "frame_nums": meta["frames"],
        "fps": meta["source_fps"],
        "temporal_stride": max_stride,
    }
    if meta.get("views"):
        item["views"] = {
            view: {key: str((folder / entry[key]).resolve()) for key in ("rgb", "action_video")}
            for view, entry in meta["views"].items()
            if view != "cam_high"
        }
    return item


def _run(task: dict) -> tuple[dict, Optional[dict], str]:
    try:
        return task, write_episode(task), ""
    except Exception as e:  # noqa: BLE001
        return task, None, f"{type(e).__name__}: {e}"


def _init_worker() -> None:
    ue.init_renderer(ud.DEFAULT_URDF, WIDTH, HEIGHT, ud.DEFAULT_ROBOT_CFG)


def _select(items: list, scene_of: Callable, scenes: set[str], per_scene: int) -> list:
    by_scene: dict[str, list] = defaultdict(list)
    for it in items:
        if not scenes or scene_of(it) in scenes:
            by_scene[scene_of(it)].append(it)
    out = []
    for scene in sorted(by_scene):
        out.extend(by_scene[scene] if per_scene <= 0 else by_scene[scene][:per_scene])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--lerobot", help="LeRobot v3 export root")
    src.add_argument("--hdf5", help="official RoboDojo .hdf5 file or directory")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--scenes", default="", help="comma-separated task names (default: all)")
    ap.add_argument("--per-scene", type=int, default=0, help="first N episodes per task (0 = all)")
    ap.add_argument("--episodes", default="", help="--lerobot only: comma-separated episode_index list")
    ap.add_argument("--stride", type=int, default=2, help="max temporal stride (2 = 12.5 fps model frames)")
    ap.add_argument("--multiview", action="store_true", help="also export both wrist cameras (MULTIVIEW=1 finetune)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    scenes = {s.strip() for s in args.scenes.split(",") if s.strip()}
    base = {"out_dir": args.out_dir, "stride": args.stride, "overwrite": args.overwrite,
            "multiview": args.multiview}
    if args.hdf5:
        root = Path(args.hdf5)
        files = [root] if root.is_file() else sorted(
            Path(dp) / fn for dp, _, fns in os.walk(root, followlinks=True) for fn in fns if fn.endswith(".hdf5")
        )
        scene_of = lambda p: p.parents[2].name if p.parent.name == "data" else p.parent.name  # noqa: E731
        tasks = [{**base, "kind": "hdf5", "path": str(p)} for p in _select(files, scene_of, scenes, args.per_scene)]
    else:
        refs = ud.load_episode_refs(args.lerobot, camera="cam_high")
        if args.episodes.strip():
            wanted = {int(x) for x in args.episodes.split(",") if x.strip()}
            refs = [r for r in refs if r.episode_index in wanted]
        else:
            refs = _select(refs, lambda r: r.scene, scenes, args.per_scene)
        wrist_refs = {
            view: {r.episode_index: r.to_dict() for r in ud.load_episode_refs(args.lerobot, camera=view)}
            for view in (WRIST_VIEWS if args.multiview else ())
        }
        tasks = [
            {**base, "kind": "lerobot", "ref": r.to_dict(),
             "wrist_refs": {view: by_ep[r.episode_index] for view, by_ep in wrist_refs.items()}}
            for r in refs
        ]
    if not tasks:
        raise SystemExit("no episodes selected")

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Converting {len(tasks)} episodes → {out_root} ({args.workers} workers)", flush=True)
    done: list[dict] = []
    failed = 0
    with mp.get_context("spawn").Pool(args.workers, initializer=_init_worker) as pool:
        for i, (task, meta, err) in enumerate(pool.imap_unordered(_run, tasks), 1):
            src_name = task.get("path") or task["ref"]["episode_index"]
            if meta is None:
                failed += 1
                print(f"[{i}/{len(tasks)}] FAILED {src_name}: {err}", flush=True)
                continue
            done.append(meta)
            print(f"[{i}/{len(tasks)}] {meta['name']}: {meta['frames']} frames, stride {meta['temporal_stride']}",
                  flush=True)

    done.sort(key=lambda m: (m["task_name"], m["episode_index"]))
    with open(out_root / "episodes.jsonl", "w", encoding="utf-8") as f:
        for meta in done:
            f.write(json.dumps(manifest_item(out_root / meta["name"], meta, args.stride), ensure_ascii=False) + "\n")
    print(f"Wrote {len(done)} episodes to {out_root / 'episodes.jsonl'} ({failed} failed)")


if __name__ == "__main__":
    main()
