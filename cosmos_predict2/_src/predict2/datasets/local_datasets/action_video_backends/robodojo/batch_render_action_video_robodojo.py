"""
批量离线渲染 RoboDojo policy_mix24 (双臂 ARX X5 仿真) action 火柴棍视频

数据集特点见 ``robodojo_data.py``。本脚本对每条 episode：
- 导出逐 episode 的 RGB 片段 (H264 yuv420p，源视频为混合编码 H264/AV1 的 LeRobot 拼接 mp4)
- 写 ``actions/episode_XXXXXX.npz`` (Joint14 action / state、EE20、有效位、源帧索引)
- 用 X5A URDF 正运动学 + 原始 HDF5 中的头部相机参数投影，渲染白底骨架视频
  (可选再叠加到 RGB 上作检查)
- 汇总 ``uniembodied_action_video_rgb_pairs.jsonl`` 供训练 dataset 读取

默认骨架流为 ``image_joint`` (也是训练 action)：RGB 第 t 帧拍摄时刻的关节状态，与 RoboTwin
``joint_action`` (拍图时刻读取的 qpos) 同语义。官方与自采 episode 都等于 ``observation.joint[t]``
(按骨架投影与画面中机械臂的重合度逐 episode 核对；``action.joint[t]`` 领先画面一帧)。

输出目录 (``--output-root``)::

    rgb/<camera>/episode_XXXXXX.mp4
    action_video/<camera>/episode_XXXXXX.mp4
    overlay/<camera>/episode_XXXXXX.mp4          (save-mode 含 overlay 时)
    actions/episode_XXXXXX.npz
    render_manifest.jsonl                         (逐 episode 结果 + 几何 QA，追加写)
    uniembodied_action_video_rgb_pairs.jsonl      (训练索引，每次运行结束重建)
    uniembodied_action_video_rgb_pairs_stats.json

用法示例:
    # 每个场景抽 1 条，输出 overlay + action 做检查
    python batch_render_action_video_robodojo.py \
        --save-mode overlay_and_action --one-per-scene --workers 24

    # 全量 (resume 跳过已完成的 episode)
    python batch_render_action_video_robodojo.py \
        --save-mode action --workers 48 --resume

    # 只重建 pairs JSONL (不渲染)
    python batch_render_action_video_robodojo.py --index-only
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import signal
import subprocess
import sys
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, os.path.join(os.path.dirname(_SCRIPT_DIR), "robotwin"))

import batch_render_action_video_robotwin as rt  # noqa: E402
from robodojo_data import (  # noqa: E402
    CAMERA_KEYS,
    DEFAULT_DATASET_ROOT,
    DEFAULT_ROBOT_CFG,
    DEFAULT_URDF,
    FPS,
    JOINT14_NAMES,
    EpisodeRef,
    extract_episode_clip,
    load_default_arm_bases,
    load_episode_geometry,
    load_episode_refs,
    probe_video_frames,
    read_episode_frames,
    split_joint14,
)

DEFAULT_OUTPUT_ROOT = (
    "/media/datasets/Sim20_dataset/UniEmbodied-Bench/"
    "policy_mix24_clean2400_sdk_ee20_25hz_instruction_20261029_v2_action_video"
)
PAIRS_NAME = "uniembodied_action_video_rgb_pairs.jsonl"
MANIFEST_NAME = "render_manifest.jsonl"

# 这些场景开头由脚本控制的 Franka 示范臂先演示，双臂 ARX 静止；Franka 关节未记录，
# 无法渲染进 action video。pairs 里写 start_frame = ARX 首次运动帧 + DEMO_SETTLE_FRAMES，
# 训练只从该帧之后采样 (Franka 在 ARX 首次运动前 0~3 帧停止，之后约 10 帧有收臂余量)。
DEMO_PREFIX_SCENES = frozenset({"imitate_sorting_sequence"})
DEMO_SETTLE_FRAMES = 10

ARM_JOINTS = [f"joint{i}" for i in range(1, 7)]
FINGER_JOINTS = ("joint7", "joint8")
# gripper01 = (finger - lo) / (hi - lo)，X5 冻结夹爪行程
GRIPPER01_FINGER_RANGE = (-0.01, 0.044)
FINGER_JOINT_LIMIT = (0.0, 0.044)
# Wrist cameras sit on the arm; links closer than this to the lens are clipped.
NEAR_PLANE = 0.02

_worker_kin: Optional[Dict[str, Any]] = None
_worker_bones: List[Tuple[str, str, str]] = []  # (joint, parent_link, child_link)
_worker_img_w = 640
_worker_img_h = 480
_worker_ffmpeg: Dict[str, Any] = {}
_worker_default_bases: Dict[str, np.ndarray] = {}


def _init_worker(urdf_path: str, img_w: int, img_h: int, robot_cfg: str,
                 ffmpeg_preset: str, ffmpeg_threads: int, ffmpeg_crf: int, rgb_crf: int) -> None:
    global _worker_ffmpeg

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(k, "1")
    init_renderer(urdf_path, img_w, img_h, robot_cfg)
    _worker_ffmpeg = {
        "preset": ffmpeg_preset, "threads": max(1, int(ffmpeg_threads)),
        "crf": int(ffmpeg_crf), "rgb_crf": int(rgb_crf),
    }


def init_renderer(urdf_path: str = DEFAULT_URDF, img_w: int = 640, img_h: int = 480,
                  robot_cfg: str = DEFAULT_ROBOT_CFG) -> None:
    """加载 X5A 运动学与默认双臂 base (worker 与在线推理共用)。"""
    global _worker_kin, _worker_bones, _worker_img_w, _worker_img_h, _worker_default_bases
    import logging
    import warnings

    from yourdfpy import URDF

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logging.getLogger("yourdfpy").setLevel(logging.ERROR)
        robot = URDF.load(urdf_path, load_meshes=False, build_scene_graph=True)

    joints: Dict[str, Dict[str, Any]] = {}
    for jn, j in robot.joint_map.items():
        jtype = str(getattr(j, "joint_type", getattr(j, "type", "")))
        axis = np.asarray(getattr(j, "axis", None) if getattr(j, "axis", None) is not None
                          else [0.0, 0.0, 1.0], dtype=np.float64).reshape(3)
        if jtype in ("revolute", "continuous", "prismatic") and np.linalg.norm(axis) > 1e-12:
            axis = axis / np.linalg.norm(axis)
        joints[str(jn)] = {
            "type": jtype,
            "parent": rt._link_name(getattr(j, "parent", None)),
            "child": rt._link_name(getattr(j, "child", None)),
            "origin": rt._as_transform(getattr(j, "origin", None)),
            "axis": axis,
        }
    active = ARM_JOINTS + list(FINGER_JOINTS)
    missing = [j for j in active if j not in joints]
    if missing:
        raise RuntimeError(f"URDF {urdf_path} missing joints {missing}")
    _worker_kin = {"joints": joints}
    _worker_bones = [(j, joints[j]["parent"], joints[j]["child"]) for j in active]
    _worker_img_w, _worker_img_h = int(img_w), int(img_h)
    _worker_default_bases = load_default_arm_bases(robot_cfg)


def render_action_frames(
    joint14: np.ndarray, bases: Dict[str, np.ndarray], K: np.ndarray, T_world_camera: np.ndarray,
) -> np.ndarray:
    """白底骨架帧 (T,H,W,3) RGB uint8，与 ``render_one`` 写入 action_video 的帧相同 (编码前)。

    需先 ``init_renderer``；joint14 为整段 (或整段的连续后缀) 源帧率序列，夹爪状态机按帧计窗。
    """
    bones, dots = precompute_skeleton(np.asarray(joint14), bases, np.asarray(K, dtype=np.float64),
                                      np.asarray(T_world_camera, dtype=np.float64), transparent=False)
    t = int(np.asarray(joint14).shape[0])
    frames = np.empty((t, _worker_img_h, _worker_img_w, 3), dtype=np.uint8)
    for i in range(t):
        frames[i] = 255
        rt._blit_skeleton_frame(frames[i], bones, dots, i)
    return frames


def _gripper01_to_finger(g: np.ndarray) -> np.ndarray:
    lo, hi = GRIPPER01_FINGER_RANGE
    finger = lo + np.clip(np.asarray(g, dtype=np.float64), 0.0, 1.0) * (hi - lo)
    return np.clip(finger, FINGER_JOINT_LIMIT[0], FINGER_JOINT_LIMIT[1])


def _arm_link_transforms(arm_q: np.ndarray, gripper01: np.ndarray) -> Dict[str, np.ndarray]:
    """单臂 base_link 系下各 link 的 (T,4,4)。X5A 为链式结构 + 两指并联。"""
    assert _worker_kin is not None
    t = int(arm_q.shape[0])
    joints = _worker_kin["joints"]
    finger = _gripper01_to_finger(gripper01)
    q = {jn: arm_q[:, i].astype(np.float64) for i, jn in enumerate(ARM_JOINTS)}
    q.update({jn: finger for jn in FINGER_JOINTS})
    eye = np.broadcast_to(np.eye(4, dtype=np.float64), (t, 4, 4))
    world: Dict[str, np.ndarray] = {"base_link": eye}
    for jn in ARM_JOINTS + list(FINGER_JOINTS):
        info = joints[jn]
        world[info["child"]] = world[info["parent"]] @ rt._joint_local_batch(info, q[jn], t)
    return world


def fk_link6_world(arm_q: np.ndarray, base: np.ndarray) -> np.ndarray:
    """(T,3) link6 原点世界坐标；用于和 observation.state 的 EE xyz 对账。"""
    links = _arm_link_transforms(arm_q, np.ones(arm_q.shape[0]))
    return (base[None] @ links["link6"])[:, :3, 3]


def _project(view: np.ndarray, pts_world: np.ndarray, K: np.ndarray):
    t = pts_world.shape[0]
    return rt._project_batch(np.broadcast_to(view, (t, 4, 4)), pts_world,
                             float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]))


def _camera_points(view: np.ndarray, pts_world: np.ndarray) -> np.ndarray:
    """World points (T,3) through per-frame view (T,4,4) → OpenCV camera points (T,3)."""
    hom = np.concatenate([pts_world, np.ones((pts_world.shape[0], 1), dtype=np.float64)], axis=1)
    return np.einsum("tij,tj->ti", view, hom)[:, :3]


def _pixels(cam: np.ndarray, K: np.ndarray):
    z = cam[:, 2]
    zz = np.where(z > 1e-6, z, 1.0)
    u = np.round(float(K[0, 0]) * cam[:, 0] / zz + float(K[0, 2])).astype(np.int64)
    v = np.round(float(K[1, 1]) * cam[:, 1] / zz + float(K[1, 2])).astype(np.int64)
    return u, v


def _clip_segment_near(p0: np.ndarray, p1: np.ndarray, near: float = NEAR_PLANE):
    """Clip a camera-space segment to z>=near. Returns p0, p1, valid (T,)."""
    z0 = p0[:, 2]
    z1 = p1[:, 2]
    denom = z1 - z0
    alpha = np.clip(np.where(np.abs(denom) > 1e-8, (near - z0) / denom, 0.0), 0.0, 1.0)
    beta = np.clip(np.where(np.abs(denom) > 1e-8, (near - z1) / (z0 - z1), 0.0), 0.0, 1.0)
    out0 = np.where((z0 < near)[:, None], p0 + alpha[:, None] * (p1 - p0), p0)
    out1 = np.where((z1 < near)[:, None], p1 + beta[:, None] * (p0 - p1), p1)
    return out0, out1, ~((z0 < near) & (z1 < near))


def precompute_skeleton(
    joint14: np.ndarray, bases: Dict[str, np.ndarray], K: np.ndarray,
    T_world_camera: np.ndarray, transparent: bool,
) -> Tuple[list, list]:
    """整段 episode 双臂 FK + 投影 → (bones, joints) 绘制轨迹 (与 RoboTwin 同格式)。

    ``T_world_camera`` 为单帧外参 (4,4)（头部相机静止）或逐帧外参 (T,4,4)（腕部相机随臂运动，
    第 t 行是 RGB 第 t 帧拍摄时的位姿）。逐帧外参下，骨骼在近平面 ``NEAR_PLANE`` 处裁剪，
    相机背后的关节点不画。
    """
    la, lg, ra, rg = split_joint14(joint14)
    t = int(la.shape[0])
    pose = np.asarray(T_world_camera, dtype=np.float64)
    per_frame = pose.ndim == 3
    if per_frame and pose.shape != (t, 4, 4):
        raise ValueError(f"T_world_camera shape {pose.shape} != {(t, 4, 4)}")
    view = np.linalg.inv(pose)
    scale = np.array([_worker_img_w / 640.0, _worker_img_h / 480.0, 1.0])
    K = np.diag(scale) @ K
    c = 4 if transparent else 3
    arm_bone = np.tile(np.array([*rt.COLOR_ARM_BONE, 255][:c]), (t, 1))
    arm_joint = np.tile(np.array([*rt.COLOR_ARM_JOINT, 255][:c]), (t, 1))
    bones, dots = [], []
    for side, arm_q, g in (("left", la, lg), ("right", ra, rg)):
        closed, settled = rt._gripper_state_masks(1.0 - np.asarray(g, dtype=np.float64))
        g_bone, g_joint = rt._gripper_color_traj(closed, settled, transparent)
        links = _arm_link_transforms(arm_q, g)
        base = bases[side]
        if per_frame:
            cam = {name: _camera_points(view, (base[None] @ T)[:, :3, 3]) for name, T in links.items()}
            for jn, parent, child in _worker_bones:
                p0, p1, valid = _clip_segment_near(cam[parent], cam[child])
                u1, v1 = _pixels(p0, K)
                u2, v2 = _pixels(p1, K)
                du, dv = _pixels(cam[child], K)
                is_finger = jn in FINGER_JOINTS
                bones.append((u1, v1, u2, v2, valid, g_bone if is_finger else arm_bone))
                dots.append((du, dv, cam[child][:, 2] >= NEAR_PLANE, g_joint if is_finger else arm_joint))
            continue
        uv = {}
        for name, T in links.items():
            uv[name] = _project(view, (base[None] @ T)[:, :3, 3], K)
        for jn, parent, child in _worker_bones:
            u1, v1, ok1 = uv[parent]
            u2, v2, ok2 = uv[child]
            valid = ok1 & ok2
            is_finger = jn in FINGER_JOINTS
            bones.append((u1, v1, u2, v2, valid, g_bone if is_finger else arm_bone))
            dots.append((u2, v2, valid, g_joint if is_finger else arm_joint))
    return bones, dots


def _build_ffmpeg_cmd(output_path: str, fps: float) -> List[str]:
    return [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{_worker_img_w}x{_worker_img_h}", "-pix_fmt", "rgb24", "-r", f"{fps:g}",
        "-i", "-",
        "-c:v", "libx264", "-preset", _worker_ffmpeg["preset"], "-crf", str(_worker_ffmpeg["crf"]),
        "-threads", str(_worker_ffmpeg["threads"]), "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", "-f", "mp4", output_path,
    ]


class _FfmpegFrameReader:
    """ffmpeg 解码为 rgb24 管道 (支持 AV1/yuv444)，按顺序逐帧读取。"""

    def __init__(self, path: str, start: float, n: int, w: int, h: int):
        self.w, self.h, self.n = w, h, n
        cmd = ["ffmpeg", "-loglevel", "error"]
        if start > 0:
            cmd += ["-ss", f"{start:.6f}"]
        cmd += ["-i", path, "-frames:v", str(n), "-vf", f"scale={w}:{h}",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.frame_bytes = w * h * 3

    def read(self) -> Optional[np.ndarray]:
        buf = self.proc.stdout.read(self.frame_bytes) if self.proc.stdout else b""
        if len(buf) < self.frame_bytes:
            return None
        return np.frombuffer(buf, dtype=np.uint8).reshape(self.h, self.w, 3)

    def close(self) -> None:
        try:
            if self.proc.stdout:
                self.proc.stdout.close()
            self.proc.kill()
            self.proc.wait(timeout=10)
        except Exception:
            pass


def _encode(path: str, n: int, producer, fps: float) -> Tuple[str, str]:
    tmp = path + ".part"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    rc, err = rt._encode_to_ffmpeg(_build_ffmpeg_cmd(tmp, fps), n, producer)
    if rc == 0 and os.path.exists(tmp) and os.path.getsize(tmp) > 0:
        os.replace(tmp, path)
        return "ok", ""
    rt._safe_unlink(tmp)
    return "fail", f"ffmpeg rc={rc}: {err}"


def _save_actions(path: str, frames: Dict[str, np.ndarray], ref: EpisodeRef,
                  geom: Dict[str, Any]) -> None:
    meta = {
        "episode_index": ref.episode_index,
        "scene": ref.scene,
        "instruction": ref.instruction,
        "official": ref.official,
        "fps": FPS,
        "joint14_names": JOINT14_NAMES,
        "gripper01": "1=open, 0=closed",
        "action_joint": "action.joint with terminal-padding rows held from the last valid row",
        "image_joint": "Joint14 at the capture time of RGB frame k (training action) "
                       "= observation_joint[k]; action_joint[k] leads the image by one frame",
        "camera": ref.camera,
        "intrinsic": np.asarray(geom["intrinsic"]).tolist(),
        "T_world_camera_opencv": np.asarray(geom["T_world_camera_opencv"]).tolist(),
        "base_source": geom["base_source"],
    }
    tmp = path + ".part.npz"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(
        tmp,
        image_joint=frames["image_joint"],
        action_joint=frames["action_joint"],
        action_joint_raw=frames["action_joint_raw"],
        observation_joint=frames["observation_joint"],
        observation_state=frames["observation_state"],
        action_ee20=frames["action_ee20"],
        action_valid=frames["action_valid"],
        action_joint_valid=frames["action_joint_valid"],
        rgb_frame_index=frames["rgb_frame_index"],
        meta=np.array(json.dumps(meta, ensure_ascii=False)),
    )
    os.replace(tmp, path)


def render_one(task: dict) -> dict:
    ref = EpisodeRef(**task["ref"])
    out = {"episode_index": ref.episode_index, "scene": ref.scene, "status": "ok"}
    fps = float(task["fps"])
    try:
        frames = read_episode_frames(ref)
    except Exception as e:
        return {**out, "status": "skip", "reason": f"read_parquet failed: {e!r}"[:300]}
    try:
        geom = load_episode_geometry(
            ref.hdf5, CAMERA_KEYS[ref.camera], int(frames["rgb_frame_index"][0]),
            default_bases=_worker_default_bases)
    except Exception as e:
        return {**out, "status": "skip", "reason": f"load_hdf5 failed: {e!r}"[:300]}

    obs = frames["observation_joint"].astype(np.float64)
    state = frames["observation_state"].astype(np.float64)
    fk_err = []
    for side, q, xyz in (("left", obs[:, 0:6], state[:, 0:3]), ("right", obs[:, 7:13], state[:, 10:13])):
        fk_err.append(np.linalg.norm(fk_link6_world(q, geom["bases"][side]) - xyz, axis=1).max())
    out.update({
        "frames": ref.length,
        "fk_vs_state_max_mm": round(float(max(fk_err)) * 1000.0, 4),
        "intrinsic_fx_fy": [round(float(geom["intrinsic"][0, 0]), 3), round(float(geom["intrinsic"][1, 1]), 3)],
        "extrinsic_static": bool(geom["extrinsic_static"]),
        "base_source": geom["base_source"],
    })

    try:
        _save_actions(task["actions_path"], frames, ref, geom)
    except Exception as e:
        return {**out, "status": "fail", "reason": f"save npz failed: {e!r}"[:300]}

    rgb_path = task["rgb_path"]
    if task["export_rgb"]:
        rc, err = extract_episode_clip(
            ref, rgb_path, crf=_worker_ffmpeg["rgb_crf"], preset=_worker_ffmpeg["preset"],
            threads=_worker_ffmpeg["threads"], fps=fps)
        if rc != 0:
            return {**out, "status": "fail", "reason": f"rgb export rc={rc}: {err}"}
    if os.path.exists(rgb_path):
        nb = probe_video_frames(rgb_path)
        out["rgb_frames"] = nb
        if nb != ref.length:
            return {**out, "status": "fail", "reason": f"rgb frames {nb} != {ref.length}"}

    stream = frames[task["joint_stream"]]
    n = ref.length
    K = np.asarray(geom["intrinsic"])
    Tc = np.asarray(geom["T_world_camera_opencv"])
    w, h = _worker_img_w, _worker_img_h

    if task["save_overlay"]:
        bones, dots = precompute_skeleton(stream, geom["bases"], K, Tc, transparent=True)
        if os.path.exists(rgb_path):
            reader = _FfmpegFrameReader(rgb_path, 0.0, n, w, h)
        else:
            reader = _FfmpegFrameReader(ref.video_file, max(0.0, ref.from_timestamp - 0.5 / fps), n, w, h)
        alpha = float(task["overlay_alpha"])

        def _overlay(t: int) -> np.ndarray:
            bg = reader.read()
            if bg is None:
                bg = np.zeros((h, w, 3), dtype=np.uint8)
            fg = np.zeros((h, w, 4), dtype=np.uint8)
            rt._blit_skeleton_frame(fg, bones, dots, t)
            return rt._overlay_frame(bg, fg, alpha)

        try:
            out["overlay_status"], out["overlay_reason"] = _encode(task["overlay_path"], n, _overlay, fps)
        finally:
            reader.close()

    if task["save_action"]:
        bones, dots = precompute_skeleton(stream, geom["bases"], K, Tc, transparent=False)

        def _action(t: int) -> np.ndarray:
            img = np.full((h, w, 3), 255, dtype=np.uint8)
            return rt._blit_skeleton_frame(img, bones, dots, t)

        status, reason = _encode(task["action_path"], n, _action, fps)
        if status != "ok":
            return {**out, "status": "fail", "reason": reason}
    return out


def _paths(root: str, camera: str, ep: int) -> Dict[str, str]:
    name = f"episode_{ep:06d}"
    return {
        "rgb_path": os.path.join(root, "rgb", camera, f"{name}.mp4"),
        "action_path": os.path.join(root, "action_video", camera, f"{name}.mp4"),
        "overlay_path": os.path.join(root, "overlay", camera, f"{name}.mp4"),
        "actions_path": os.path.join(root, "actions", f"{name}.npz"),
    }


def _nonempty(path: str) -> bool:
    try:
        return os.path.getsize(path) > 0
    except OSError:
        return False


def demo_start_frame(actions_path: str) -> int:
    """ARX 首次运动帧 + DEMO_SETTLE_FRAMES (不超过最后一帧)。"""
    with np.load(actions_path, allow_pickle=False) as d:
        a = np.asarray(d["image_joint"], dtype=np.float32)
    moved = np.where(np.abs(a - a[0]).max(axis=1) > 1e-3)[0]
    first = int(moved[0]) if moved.size else 0
    return min(first + DEMO_SETTLE_FRAMES, len(a) - 1)


def write_pairs_index(refs: Sequence[EpisodeRef], root: str, camera: str, fps: float,
                      dataset_root: str) -> Tuple[int, int]:
    """为 rgb / action_video / actions 三者齐全的 episode 重建训练索引。"""
    pairs_path = os.path.join(root, PAIRS_NAME)
    tmp = pairs_path + ".part"
    lengths: List[int] = []
    per_scene: Counter = Counter()
    n_missing = 0
    demo_trimmed: List[int] = []
    with open(tmp, "w", encoding="utf-8") as f:
        for i, ref in enumerate(refs):
            p = _paths(root, camera, ref.episode_index)
            if not (_nonempty(p["rgb_path"]) and _nonempty(p["action_path"]) and _nonempty(p["actions_path"])):
                n_missing += 1
                continue
            item = {
                "idx": i,
                "uuid": ref.uuid,
                "dataset": os.path.basename(dataset_root.rstrip("/")),
                "episode_index": ref.episode_index,
                "task_name": ref.scene,
                "official": ref.official,
                "classification": ref.classification,
                "source_episode_id": ref.source_episode_id,
                "frame_nums": ref.length,
                "fps": fps,
                "lang": ref.instruction,
                "camera": camera,
                "rgb": p["rgb_path"],
                "action_video": p["action_path"],
                "actions": p["actions_path"],
                "hdf5": ref.hdf5,
            }
            if ref.scene in DEMO_PREFIX_SCENES:
                item["start_frame"] = demo_start_frame(p["actions_path"])
                demo_trimmed.append(item["start_frame"])
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            lengths.append(ref.length)
            per_scene[ref.scene] += 1
    os.replace(tmp, pairs_path)
    arr = np.asarray(lengths, dtype=np.int64) if lengths else np.zeros(0, dtype=np.int64)
    stats = {
        "pairs_jsonl": pairs_path,
        "dataset_root": dataset_root,
        "camera": camera,
        "fps": fps,
        "n_episodes": int(arr.size),
        "n_missing_outputs": int(n_missing),
        "frames_total": int(arr.sum()) if arr.size else 0,
        "frame_nums_min": int(arr.min()) if arr.size else 0,
        "frame_nums_median": float(np.median(arr)) if arr.size else 0.0,
        "frame_nums_max": int(arr.max()) if arr.size else 0,
        "episodes_per_scene": dict(sorted(per_scene.items())),
        "demo_prefix_scenes": sorted(DEMO_PREFIX_SCENES),
        "demo_prefix_episodes": len(demo_trimmed),
        "demo_prefix_frames_skipped": int(sum(demo_trimmed)),
        "demo_start_frame_min_median_max": (
            [int(min(demo_trimmed)), float(np.median(demo_trimmed)), int(max(demo_trimmed))]
            if demo_trimmed else []
        ),
        "action_layout": "npz image_joint [T,14] = left_arm(6)+left_gripper01+right_arm(6)+right_gripper01 "
                         "at the capture time of each RGB frame",
    }
    with open(os.path.join(root, PAIRS_NAME.replace(".jsonl", "_stats.json")), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    return int(arr.size), n_missing


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT, help="LeRobot v3.0 数据集根目录")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help="输出根目录")
    parser.add_argument("--camera", default="cam_high", choices=["cam_high"],
                        help="投影/导出相机 (腕部相机外参随手臂运动，骨架投影只按静止外参实现)")
    parser.add_argument("--save-mode", default="action", choices=["action", "overlay_and_action", "overlay"])
    parser.add_argument("--joint-stream", default="image_joint",
                        choices=["image_joint", "action_joint", "observation_joint"],
                        help="骨架使用的关节流 (默认 image_joint: 与 RGB 帧同时刻，也是训练 action)")
    parser.add_argument("--skip-rgb", action="store_true", help="不导出逐 episode RGB 片段")
    parser.add_argument("--index-only", action="store_true", help="只根据已有产物重建 pairs JSONL")
    parser.add_argument("--urdf", default=DEFAULT_URDF)
    parser.add_argument("--robot-cfg", default=DEFAULT_ROBOT_CFG)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--start", type=int, default=0, help="起始 episode 序号（含）")
    parser.add_argument("--end", type=int, default=10 ** 9, help="结束 episode 序号（不含）")
    parser.add_argument("--episodes", type=str, default="", help="逗号分隔 episode_index 列表")
    parser.add_argument("--scenes", type=str, default="", help="逗号分隔场景名过滤")
    parser.add_argument("--one-per-scene", action="store_true", help="每个 (场景, 官方/自采) 只取第一条")
    parser.add_argument("--resume", action="store_true", help="跳过产物已齐全的 episode")
    parser.add_argument("--overlay-alpha", type=float, default=0.7)
    parser.add_argument("--ffmpeg-preset", default="veryfast")
    parser.add_argument("--ffmpeg-threads", type=int, default=1)
    parser.add_argument("--ffmpeg-crf", type=int, default=23, help="action / overlay 视频 crf")
    parser.add_argument("--rgb-crf", type=int, default=18, help="RGB 片段 crf")
    parser.add_argument("--maxtasks-per-child", type=int, default=50)
    args = parser.parse_args(argv)

    args.width += args.width % 2
    args.height += args.height % 2
    fps = FPS

    print(f"[Main] 数据集: {args.dataset_root}")
    refs = load_episode_refs(args.dataset_root, camera=args.camera)
    print(f"[Main] 共 {len(refs)} 条 episode, {len({r.scene for r in refs})} 个场景")
    os.makedirs(args.output_root, exist_ok=True)

    if args.index_only:
        n_ok, n_missing = write_pairs_index(refs, args.output_root, args.camera, fps, args.dataset_root)
        print(f"[Main] pairs: {n_ok} 条写入 {os.path.join(args.output_root, PAIRS_NAME)} (缺产物 {n_missing})")
        return 0

    sel = refs[max(0, args.start):max(0, args.end)]
    if args.episodes:
        wanted = {int(x) for x in args.episodes.split(",") if x.strip()}
        sel = [r for r in refs if r.episode_index in wanted]
    if args.scenes:
        scenes = {s.strip() for s in args.scenes.split(",") if s.strip()}
        sel = [r for r in sel if r.scene in scenes]
    if args.one_per_scene:
        seen, picked = set(), []
        for r in sel:
            if (r.scene, r.official) not in seen:
                seen.add((r.scene, r.official))
                picked.append(r)
        sel = picked

    save_action = args.save_mode in ("action", "overlay_and_action")
    save_overlay = args.save_mode in ("overlay", "overlay_and_action")
    tasks: List[dict] = []
    skipped = 0
    for r in sel:
        p = _paths(args.output_root, args.camera, r.episode_index)
        if args.resume:
            done = _nonempty(p["actions_path"])
            done &= args.skip_rgb or _nonempty(p["rgb_path"])
            done &= (not save_action) or _nonempty(p["action_path"])
            done &= (not save_overlay) or _nonempty(p["overlay_path"])
            if done:
                skipped += 1
                continue
        tasks.append({
            "ref": r.to_dict(), **p, "fps": fps,
            "export_rgb": not args.skip_rgb and not (args.resume and _nonempty(p["rgb_path"])),
            "save_action": save_action, "save_overlay": save_overlay,
            "joint_stream": args.joint_stream, "overlay_alpha": args.overlay_alpha,
        })
    print(f"[Main] 选中 {len(sel)} 条, 待渲染 {len(tasks)}, resume 跳过 {skipped}; "
          f"save_mode={args.save_mode}, joint_stream={args.joint_stream}, export_rgb={not args.skip_rgb}")
    print(f"[Main] 输出: {args.output_root}")

    manifest_path = os.path.join(args.output_root, MANIFEST_NAME)
    if tasks:
        workers = max(1, min(args.workers, len(tasks)))
        # 长 episode 先跑，减少尾部只剩少数 worker 的空转
        tasks.sort(key=lambda t: -int(t["ref"]["length"]))
        ok = fail = skip = 0
        fail_samples: List[str] = []
        fk_worst = 0.0
        t0 = time.time()
        pool_kwargs = dict(
            processes=workers, initializer=_init_worker,
            initargs=(args.urdf, args.width, args.height, args.robot_cfg, args.ffmpeg_preset,
                      args.ffmpeg_threads, args.ffmpeg_crf, args.rgb_crf),
        )
        if args.maxtasks_per_child > 0:
            pool_kwargs["maxtasksperchild"] = args.maxtasks_per_child
        try:
            from tqdm import tqdm
            pbar = tqdm(total=len(tasks), desc="Rendering", unit="ep", dynamic_ncols=True)
        except ImportError:
            pbar = None
        with mp.Pool(**pool_kwargs) as pool, open(manifest_path, "a", encoding="utf-8") as mf:
            try:
                for res in pool.imap_unordered(render_one, tasks, chunksize=1):
                    res["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    mf.write(json.dumps(res, ensure_ascii=False) + "\n")
                    mf.flush()
                    st = res.get("status")
                    fk_worst = max(fk_worst, float(res.get("fk_vs_state_max_mm", 0.0)))
                    if st == "ok":
                        ok += 1
                    elif st == "skip":
                        skip += 1
                    else:
                        fail += 1
                    if st != "ok" and len(fail_samples) < 10:
                        fail_samples.append(f"ep={res['episode_index']}: {res.get('reason', '?')}")
                    if pbar is not None:
                        pbar.set_postfix(ok=ok, skip=skip, fail=fail, fk_mm=f"{fk_worst:.2f}")
                        pbar.update(1)
            except KeyboardInterrupt:
                print("\n[Main] Ctrl-C，终止 worker ...")
                pool.terminate()
                pool.join()
                return 130
        if pbar is not None:
            pbar.close()
        el = time.time() - t0
        print(f"[Main] 完成: ok={ok} skip={skip} fail={fail}, 耗时 {el:.1f}s "
              f"({el / max(1, ok):.2f}s/ep), FK 与 observation.state 最大偏差 {fk_worst:.3f} mm")
        for s in fail_samples:
            print(f"  - {s}")

    n_ok, n_missing = write_pairs_index(refs, args.output_root, args.camera, fps, args.dataset_root)
    print(f"[Main] pairs: {n_ok} 条写入 {os.path.join(args.output_root, PAIRS_NAME)} (缺产物 {n_missing})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
