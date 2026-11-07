#!/usr/bin/env python3
"""批量渲染 RealSource-World RS-02 的 action_video（双臂 + 双夹爪火柴棍骨架）。

输入是 LeRobot v2.1 per-task 布局:
    /media/datasets/RealSource-World/
      <Task_Name>/
        meta/info.json meta/episodes.jsonl meta/camera.json
        data/chunk-000/episode_000000.parquet
        videos/chunk-000/observation.images.head_camera/episode_000000.mp4

对每条 episode:
  * 默认读 parquet 的 observation.state（follower/从臂）做 FK（realsource_calib.RS02Kinematics）；
    action_video 与 overlay 一致用从臂，对齐真机执行；可用 --action-from-action 改回 leader。
  * head 相机内参取 camera.json[episode].camera_parameters.head["720P"].MTX（按渲染分辨率缩放）；
    外参**直接用** camera.json[episode].hand_eye.{left_arm_to_eye,right_arm_to_eye}。
  * 左臂骨架点在 l_base_link 系、右臂在 r_base_link 系，各自投影到同一张 head 图。
  * 输出：action（白底骨架）/ overlay（骨架叠加真实 head 视频）；并写 metadata.json 供 merge 用。

渲染风格与 AgiBot 对齐（臂骨蓝、臂关节红、夹爪开蓝/闭橙；运动帧灰；线宽 6、关节半径 9）。

用法:
    /root/miniforge/bin/python3 visualization_urdf/realsource/batch_render_action_video_realsource.py \
        --data-root /media/datasets/RealSource-World \
        --tasks Making_steamed_potatoes \
        --output-dir /media/datasets/realsource_action_video/action \
        --metadata-output /media/datasets/realsource_action_video/metadata.json \
        --save-mode overlay_and_action --all-episodes --workers 8 --fk-backend vectorized
"""
from __future__ import annotations

import os as _os
for _e in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_e, "1")

import argparse
import fnmatch
import gzip
import json
import multiprocessing as mp
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import realsource_calib as rc

RENDER_W = 640
RENDER_H = 360
SRC_W, SRC_H = 1280, 720

# ---- worker globals ----
_kin: Optional[rc.RS02Kinematics] = None
_render_w = RENDER_W
_render_h = RENDER_H
_ffmpeg_preset = "veryfast"
_ffmpeg_crf = 23
_ffmpeg_threads = 1
_gripper_invert = rc.DEFAULT_GRIPPER_INVERT
_use_state_for_action = True
_use_state_for_overlay = True
_white: Optional[np.ndarray] = None


def _init_worker(urdf_path, render_w, render_h, preset, crf, threads, gripper_invert,
                 use_state_for_overlay=True, use_state_for_action=True):
    global _kin, _render_w, _render_h, _ffmpeg_preset, _ffmpeg_crf, _ffmpeg_threads
    global _gripper_invert, _use_state_for_overlay, _use_state_for_action, _white
    _kin = rc.RS02Kinematics(urdf_path)
    _render_w, _render_h = int(render_w), int(render_h)
    _ffmpeg_preset, _ffmpeg_crf, _ffmpeg_threads = str(preset), int(crf), max(1, int(threads))
    _gripper_invert = bool(gripper_invert)
    _use_state_for_overlay = bool(use_state_for_overlay)
    _use_state_for_action = bool(use_state_for_action)
    _white = np.full((_render_h, _render_w, 3), 255, dtype=np.uint8)
    try:
        cv2.setNumThreads(0)
    except Exception:
        pass
    print(f"[Worker {os.getpid()}] URDF loaded, render={_render_w}x{_render_h}", flush=True)


def _open_ffmpeg_writer(path: str, fps: float, w: int, h: int) -> subprocess.Popen:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-vcodec", "rawvideo",
         "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(float(fps)), "-i", "-",
         "-an", "-vcodec", "libx264", "-preset", _ffmpeg_preset, "-crf", str(_ffmpeg_crf),
         "-threads", str(_ffmpeg_threads), "-pix_fmt", "yuv420p", "-movflags", "+faststart", path],
        stdin=subprocess.PIPE)


def _compute_bone_draw(world, links, bones, base_link, view, K):
    """返回 [(u1v1(T,2), u2v2(T,2), valid(T,))] 每根 bone 的投影轨迹。"""
    inv_base = np.linalg.inv(world[base_link])  # (T,4,4)
    pos = {}
    for ln in links:
        if ln in world:
            pos[ln] = np.einsum("tij,tj->ti", inv_base, world[ln][:, :, 3])[:, :3]
    draws = []
    for a, b in bones:
        if a not in pos or b not in pos:
            continue
        uv1, d1 = rc.project_points_batch(pos[a], view, K)
        uv2, d2 = rc.project_points_batch(pos[b], view, K)
        valid = (d1 > 0.01) & (d2 > 0.01)
        draws.append((uv1, uv2, valid))
    return draws


def _load_source_array(parquet_path: str, use_state: bool) -> np.ndarray:
    """读取渲染源关节序列 (T,>=16)。
    use_state=True -> observation.state（follower/实际执行，对齐 head 视频）；
    use_state=False -> action（leader/遥操作指令，训练条件用）。
    """
    col = "observation.state" if use_state else "action"
    df = pd.read_parquet(parquet_path, columns=[col])
    return np.stack(df[col].values).astype(np.float64)


def _render_episode_frames(task: Dict[str, Any], overlay: bool):
    # 默认 action_video 与 overlay 都用 observation.state(follower/从臂)。
    use_state = bool(_use_state_for_overlay if overlay else _use_state_for_action)
    src = task.get("joints")
    if src is None:
        src = _load_source_array(task["parquet_path"], use_state=use_state)
    src = np.asarray(src, dtype=np.float64)
    n = src.shape[0]
    cap = None
    if overlay:
        cap = cv2.VideoCapture(task["video_path"])
        if not cap.isOpened():
            raise FileNotFoundError(task["video_path"])
        vn = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if vn > 0:
            n = min(n, vn)
    mf = int(task.get("max_frames") or 0)
    if mf > 0:
        n = min(n, mf)
    if n <= 0:
        if cap is not None:
            cap.release()
        return

    src = src[:n]
    cfg, close_l, close_r = rc.joints_to_cfg(src, gripper_invert=_gripper_invert)
    world = _kin.fk_batch(cfg, n)

    # 夹爪配色：settled=开蓝/闭橙（关节开红/闭绿）；运动帧骨线灰、关节点红
    closed_l, settled_l = rc.gripper_state_masks(close_l)
    closed_r, settled_r = rc.gripper_state_masks(close_r)
    bone_col_l = rc.gripper_colors_state_or_gradient(
        close_l, closed_l, settled_l, rc.COLOR_GRIPPER_OPEN_BONE, rc.COLOR_GRIPPER_CLOSED_BONE)
    bone_col_r = rc.gripper_colors_state_or_gradient(
        close_r, closed_r, settled_r, rc.COLOR_GRIPPER_OPEN_BONE, rc.COLOR_GRIPPER_CLOSED_BONE)
    joint_col_l = rc.gripper_joint_colors_state_or_motion(closed_l, settled_l)
    joint_col_r = rc.gripper_joint_colors_state_or_motion(closed_r, settled_r)

    cam = task["camera_entry"]
    K = rc.scale_K(rc.load_head_K(cam, "720P"), (SRC_W, SRC_H), (_render_w, _render_h))
    view_l = rc.hand_eye_to_view(cam["hand_eye"]["left_arm_to_eye"]["R"], cam["hand_eye"]["left_arm_to_eye"]["T"])
    view_r = rc.hand_eye_to_view(cam["hand_eye"]["right_arm_to_eye"]["R"], cam["hand_eye"]["right_arm_to_eye"]["T"])

    arm_draw = (_compute_bone_draw(world, rc.LEFT_ARM_LINKS, rc.LEFT_ARM_BONES, rc.LEFT_BASE_LINK, view_l, K)
                + _compute_bone_draw(world, rc.RIGHT_ARM_LINKS, rc.RIGHT_ARM_BONES, rc.RIGHT_BASE_LINK, view_r, K))
    grip_l = _compute_bone_draw(world, rc.LEFT_ARM_LINKS + rc.LEFT_GRIPPER_PADS, rc.LEFT_GRIPPER_BONES,
                                rc.LEFT_BASE_LINK, view_l, K)
    grip_r = _compute_bone_draw(world, rc.RIGHT_ARM_LINKS + rc.RIGHT_GRIPPER_PADS, rc.RIGHT_GRIPPER_BONES,
                                rc.RIGHT_BASE_LINK, view_r, K)

    for t in range(n):
        if overlay:
            ok, bg = cap.read()
            if not ok:
                break
            if bg.shape[1] != _render_w or bg.shape[0] != _render_h:
                bg = cv2.resize(bg, (_render_w, _render_h), interpolation=cv2.INTER_AREA)
            img = bg
        else:
            img = _white.copy()
        # 臂骨
        for uv1, uv2, valid in arm_draw:
            if not valid[t]:
                continue
            p1 = (int(round(uv1[t, 0])), int(round(uv1[t, 1])))
            p2 = (int(round(uv2[t, 0])), int(round(uv2[t, 1])))
            cv2.line(img, p1, p2, rc.COLOR_ARM_BONE, rc.ARM_BONE_WIDTH, cv2.LINE_AA)
            cv2.circle(img, p2, rc.ARM_JOINT_RADIUS, rc.COLOR_ARM_JOINT, -1, cv2.LINE_AA)
        # 夹爪：settled 开合二值色；运动帧骨线灰、关节点红
        for draws, bcols, jcols in (
            (grip_l, bone_col_l, joint_col_l),
            (grip_r, bone_col_r, joint_col_r),
        ):
            bcol = (int(bcols[t, 0]), int(bcols[t, 1]), int(bcols[t, 2]))
            jcol = (int(jcols[t, 0]), int(jcols[t, 1]), int(jcols[t, 2]))
            for uv1, uv2, valid in draws:
                if not valid[t]:
                    continue
                p1 = (int(round(uv1[t, 0])), int(round(uv1[t, 1])))
                p2 = (int(round(uv2[t, 0])), int(round(uv2[t, 1])))
                cv2.line(img, p1, p2, bcol, rc.ARM_BONE_WIDTH, cv2.LINE_AA)
                cv2.circle(img, p2, rc.ARM_JOINT_RADIUS, jcol, -1, cv2.LINE_AA)
        yield img
    if cap is not None:
        cap.release()


def _write_video_iter(path: str, frames, fps: float) -> int:
    tmp = str(Path(path).with_suffix(".tmp.mp4"))
    proc = _open_ffmpeg_writer(tmp, fps, _render_w, _render_h)
    assert proc.stdin is not None
    count = 0
    try:
        for fr in frames:
            proc.stdin.write(fr.tobytes())
            count += 1
    finally:
        proc.stdin.close()
        ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"ffmpeg rc={ret}: {tmp}")
    Path(tmp).replace(path)
    return count


def render_one(task: Dict[str, Any]) -> Dict[str, Any]:
    try:
        action_status = overlay_status = "skip"
        frames = 0
        if task["save_action"]:
            ap = Path(task["action_path"])
            if task["resume"] and ap.exists() and ap.stat().st_size > 1024:
                action_status = "exists"
            else:
                frames = _write_video_iter(str(ap), _render_episode_frames(task, overlay=False), task["fps"])
                action_status = "ok"
        if task["save_overlay"]:
            op = Path(task["overlay_path"])
            if task["resume"] and op.exists() and op.stat().st_size > 1024:
                overlay_status = "exists"
            else:
                of = _write_video_iter(str(op), _render_episode_frames(task, overlay=True), task["fps"])
                frames = max(frames, of)
                overlay_status = "ok"
        return {**task, "camera_entry": None, "status": "ok",
                "action_status": action_status, "overlay_status": overlay_status, "frames": frames}
    except Exception as exc:  # noqa: BLE001
        return {**task, "camera_entry": None, "status": "fail", "reason": repr(exc)}


def load_task_meta(task_dir: Path):
    info = json.loads((task_dir / "meta" / "info.json").read_text())
    episodes = []
    with (task_dir / "meta" / "episodes.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                episodes.append(json.loads(line))
    camera = json.loads((task_dir / "meta" / "camera.json").read_text())
    return info, episodes, camera


def build_tasks(args) -> List[Dict[str, Any]]:
    data_root = Path(args.data_root)
    if args.tasks:
        task_names = [x.strip() for x in args.tasks.split(",") if x.strip()]
    else:
        task_names = sorted(p.name for p in data_root.iterdir()
                            if p.is_dir() and fnmatch.fnmatch(p.name, args.task_glob))
    ep_filter = None
    if args.episodes:
        ep_filter = {int(x) for x in args.episodes.split(",") if x.strip()}

    # 质量筛选清单(D2 产出):{task_name: {"kept": [ep,...]}}，仅渲染保留 episode
    screen_kept: Optional[Dict[str, set]] = None
    if args.episode_filter:
        with open(args.episode_filter, encoding="utf-8") as f:
            screen = json.load(f)
        screen_kept = {tn: set(v.get("kept", [])) for tn, v in screen.get("tasks", {}).items()}
        if not args.tasks:
            task_names = [tn for tn in task_names if tn in screen_kept]

    selected: List[Dict[str, Any]] = []
    for tn in task_names:
        tdir = data_root / tn
        if not tdir.exists():
            print(f"[WARN] missing task dir: {tdir}", flush=True)
            continue
        info, episodes, camera = load_task_meta(tdir)
        dtmpl = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
        vtmpl = info.get("video_path", "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4")
        chunk_size = int(info.get("chunks_size", 1000))
        if screen_kept is not None:
            keep = screen_kept.get(tn, set())
            episodes = [e for e in episodes if int(e["episode_index"]) in keep]
        if ep_filter is not None:
            episodes = [e for e in episodes if int(e["episode_index"]) in ep_filter]
        if not args.all_episodes and ep_filter is None and screen_kept is None:
            episodes = episodes[: max(0, int(args.videos_per_task))]
        for ep in episodes:
            ei = int(ep["episode_index"])
            ck = ei // chunk_size
            pq = tdir / dtmpl.format(episode_chunk=ck, episode_index=ei)
            if not pq.exists():
                continue
            vid = tdir / vtmpl.format(episode_chunk=ck, video_key=args.video_key, episode_index=ei)
            if args.save_mode in ("overlay", "overlay_and_action") and not vid.exists():
                continue
            cam_entry = camera.get(f"episode_{ei:06d}")
            if cam_entry is None:
                print(f"[WARN] no camera params: {tn} ep{ei}", flush=True)
                continue
            selected.append({
                "task_name": tn,
                "episode_index": ei,
                "parquet_path": str(pq),
                "video_path": str(vid),
                "action_path": str(Path(args.action_output_dir or args.output_dir) / tn / f"episode_{ei:06d}.mp4"),
                "overlay_path": str(Path(args.output_dir) / tn / f"episode_{ei:06d}_overlay.mp4"),
                "camera_entry": cam_entry,
                "fps": float(args.fps),
                "save_action": args.save_mode in ("action", "overlay_and_action"),
                "save_overlay": args.save_mode in ("overlay", "overlay_and_action"),
                "max_frames": int(args.max_frames),
                "resume": bool(args.resume),
            })
            if args.limit > 0 and len(selected) >= args.limit:
                return selected
    return selected


def write_metadata(records: List[Dict[str, Any]], path: str, av_w: int, av_h: int) -> None:
    if not path:
        return
    out = []
    for r in records:
        out.append({
            "task_name": r.get("task_name"),
            "episode_index": r.get("episode_index"),
            "parquet_path": r.get("parquet_path"),
            "video": r.get("video_path"),
            "action_video": r.get("action_path") if r.get("action_status") in ("ok", "exists") else "",
            "overlay_video": r.get("overlay_path") if r.get("overlay_status") in ("ok", "exists") else "",
            "status": r.get("status"),
            "reason": r.get("reason", ""),
            "frames": r.get("frames"),
            "width": av_w,
            "height": av_h,
        })
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if path.endswith(".gz"):
        with gzip.open(path, "wb") as f:
            pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        Path(path).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Render RealSource RS-02 action/overlay videos")
    p.add_argument("--data-root", default="/media/datasets/RealSource-World")
    p.add_argument("--tasks", default="")
    p.add_argument("--task-glob", default="*")
    p.add_argument("--episodes", default="")
    p.add_argument("--episode-filter", default="",
                   help="D2 质量筛选清单 JSON;仅渲染其中 kept 的 episode")
    p.add_argument("--all-episodes", action="store_true")
    p.add_argument("--videos-per-task", type=int, default=4)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--save-mode", choices=["action", "overlay", "overlay_and_action"], default="action")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--action-output-dir", default="")
    p.add_argument("--metadata-output", default="")
    p.add_argument("--video-key", default="observation.images.head_camera")
    p.add_argument("--urdf", default=str(Path(__file__).resolve().parent / "RS-02.urdf"))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--chunksize", type=int, default=1)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--render-w", type=int, default=RENDER_W)
    p.add_argument("--render-h", type=int, default=RENDER_H)
    p.add_argument("--gripper-invert", action=argparse.BooleanOptionalAction,
                   default=rc.DEFAULT_GRIPPER_INVERT,
                   help="夹爪开/合语义：默认 True(pos 高≈开，close_frac=1-pos)；"
                        "若颜色仍反可用 --no-gripper-invert")
    p.add_argument("--use-state-for-action", dest="use_state_for_action",
                   action="store_true", default=True,
                   help="action_video 用 observation.state(follower/从臂)渲染（默认开）")
    p.add_argument("--action-from-action", dest="use_state_for_action",
                   action="store_false",
                   help="action_video 改用 action(leader)渲染")
    p.add_argument("--use-state-for-overlay", dest="use_state_for_overlay",
                   action="store_true", default=True,
                   help="overlay 用 observation.state(follower)渲染以对齐真实视频（默认开）")
    p.add_argument("--overlay-from-action", dest="use_state_for_overlay",
                   action="store_false",
                   help="overlay 也用 action(leader)渲染（会领先真实执行几帧）")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--ffmpeg-preset", default="veryfast")
    p.add_argument("--ffmpeg-crf", type=int, default=23)
    p.add_argument("--ffmpeg-threads", type=int, default=1)
    p.add_argument("--fk-backend", choices=["vectorized"], default="vectorized")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.save_mode == "overlay_and_action" and not args.action_output_dir:
        raise ValueError("--save-mode overlay_and_action requires --action-output-dir")
    if args.save_mode == "action" and not args.action_output_dir:
        args.action_output_dir = args.output_dir
    tasks = build_tasks(args)
    print(f"[Main] tasks={len(tasks)} save_mode={args.save_mode} workers={args.workers} "
          f"render={args.render_w}x{args.render_h} "
          f"action_src={'state' if args.use_state_for_action else 'action'} "
          f"overlay_src={'state' if args.use_state_for_overlay else 'action'}", flush=True)
    if not tasks:
        return 0
    initargs = (args.urdf, args.render_w, args.render_h, args.ffmpeg_preset,
                args.ffmpeg_crf, args.ffmpeg_threads, args.gripper_invert,
                args.use_state_for_overlay, args.use_state_for_action)
    nw = min(max(1, args.workers), len(tasks))
    records: List[Dict[str, Any]] = []
    t0 = time.time()
    ok = fail = 0
    with mp.Pool(nw, initializer=_init_worker, initargs=initargs) as pool:
        for rec in pool.imap_unordered(render_one, tasks, chunksize=max(1, args.chunksize)):
            records.append(rec)
            if rec.get("status") == "ok":
                ok += 1
            else:
                fail += 1
                print(f"[FAIL] {rec.get('task_name')}#{rec.get('episode_index')}: {rec.get('reason')}", flush=True)
            done = ok + fail
            if done % 20 == 0 or done == len(tasks):
                print(f"[Main] {done}/{len(tasks)} ok={ok} fail={fail} elapsed={(time.time()-t0)/60:.1f}m", flush=True)
    write_metadata(records, args.metadata_output, args.render_w, args.render_h)
    print(f"[Main] done ok={ok} fail={fail} elapsed={time.time()-t0:.1f}s", flush=True)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
