"""
批量离线渲染 RoboTwin 数据集 action 火柴棍视频

RoboTwin 数据集特点：
- 双臂 ARX5 机器人，每臂 6 个 revolute 关节 + 1 个 prismatic 夹爪
- Action 14 维：left_arm(6) + left_gripper(1) + right_arm(6) + right_gripper(1)
- HDF5 中包含逐帧相机内参和外参，可做精确投影
- 元数据为 JSONL 格式
- URDF 根 frame (footprint) 需要经 robot_pose 变换到 SAPIEN 世界系再做投影

用法示例:
    # 渲染前 10 条，输出 overlay 视频（骨骼叠加到原始视频）和纯 action 视频
    python visualization_urdf/robotwin/batch_render_action_video_robotwin.py \
        --save-mode overlay_and_action \
        --output-dir /media/datasets/RoboTwin/action_video_overlay/ \
        --action-output-dir /media/datasets/RoboTwin/action_video/ \
        --end 10000000 --workers 8 --export-pkl /media/datasets/RoboTwin/robotwin_27500_50000_action_video.pkl.gz

    # 仅输出纯 action 骨骼视频（默认 resume 会预扫描目录）
    python visualization_urdf/robotwin/batch_render_action_video_robotwin.py \
        --save-mode action \
        --output-dir /tmp/robotwin_action/ \
        --end 100 --workers 64 --resume

    # 开启严格的帧数校验（用 ffprobe 验证已渲染视频，不匹配则重渲染）
    python visualization_urdf/robotwin/batch_render_action_video_robotwin.py \
        --save-mode overlay_and_action \
        --output-dir /.../action_video_overlay/ \
        --action-output-dir /.../action_video/ \
        --resume --verify-resume --verify-workers 32 \
        --workers 64

    # 输出拼接检查视频（原始 + 骨骼并排）
    python visualization_urdf/robotwin/batch_render_action_video_robotwin.py \
        --save-mode concat \
        --output-dir /tmp/robotwin_concat/ \
        --end 50 --workers 8

性能相关参数:
    --ffmpeg-preset  ffmpeg libx264 preset（默认 veryfast；ultrafast 最快、medium 最慢但小）
    --ffmpeg-threads 每个 ffmpeg 子进程线程数（默认 1，避免与 worker 竞争 CPU）
    --ffmpeg-crf     ffmpeg libx264 crf（默认 23，越低质量越高文件越大）
    --maxtasks-per-child worker 处理多少任务后重启（默认 200，防止内存泄漏）
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import copy
import gzip
import json
import multiprocessing as mp
import os
import pickle
import signal
import subprocess
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import h5py
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from visualize_action_on_urdf import (
    _as_transform,
    _link_name,
    detect_fps,
)

# ---------------------------------------------------------------------------
# ARX5 双臂关节定义（与 config.yml arm_joints_name 一致）
# ---------------------------------------------------------------------------
# fl = Front-Left arm (left), fr = Front-Right arm (right)
LEFT_ARM_JOINTS = [
    "fl_joint1", "fl_joint2", "fl_joint3",
    "fl_joint4", "fl_joint5", "fl_joint6",
]
RIGHT_ARM_JOINTS = [
    "fr_joint1", "fr_joint2", "fr_joint3",
    "fr_joint4", "fr_joint5", "fr_joint6",
]

LEFT_GRIPPER_DRIVING_JOINT = "fl_joint7"
LEFT_GRIPPER_MIMIC_JOINT = "fl_joint8"
RIGHT_GRIPPER_DRIVING_JOINT = "fr_joint7"
RIGHT_GRIPPER_MIMIC_JOINT = "fr_joint8"

GRIPPER_PRISMATIC_LIMIT = (0.0, 0.04765)

ALL_ACTION_JOINTS = frozenset(
    LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS
    + [LEFT_GRIPPER_DRIVING_JOINT, LEFT_GRIPPER_MIMIC_JOINT,
       RIGHT_GRIPPER_DRIVING_JOINT, RIGHT_GRIPPER_MIMIC_JOINT]
)

LEFT_GRIPPER_JOINT_SET = frozenset([LEFT_GRIPPER_DRIVING_JOINT, LEFT_GRIPPER_MIMIC_JOINT])
RIGHT_GRIPPER_JOINT_SET = frozenset([RIGHT_GRIPPER_DRIVING_JOINT, RIGHT_GRIPPER_MIMIC_JOINT])

HEAD_VIEW_NAMES = ["head_camera"]

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_URDF = os.path.join(_SCRIPT_DIR, "aloha-agilex", "urdf", "arx5_description_isaac.urdf")
DEFAULT_CONFIG = os.path.join(_SCRIPT_DIR, "aloha-agilex", "config.yml")
DEFAULT_JSONL = "/media/unify/RoboTwin/robotwin_27500_50000_50000.jsonl"

# 配色 (cv2 画在 RGB 帧上, 按 RGB 顺序; 对齐 RealSource / ManiSkill)
COLOR_ARM_BONE = (0, 120, 255)              # 蓝
COLOR_GRIPPER_OPEN_BONE = (0, 120, 255)     # 开: 蓝
COLOR_GRIPPER_CLOSED_BONE = (255, 140, 0)   # 合: 橙
COLOR_ARM_JOINT = (255, 50, 50)             # 红
COLOR_GRIPPER_OPEN_JOINT = (255, 50, 50)    # 开: 红
COLOR_GRIPPER_CLOSED_JOINT = (0, 200, 0)    # 合: 绿
COLOR_GRIPPER_MOTION_BONE = (128, 128, 128) # 开合过程骨线: 灰
COLOR_GRIPPER_MOTION_JOINT = (255, 50, 50)  # 开合过程关节: 红
COLOR_INACTIVE = (192, 192, 192)

# 夹爪时序状态机参数 (对齐 ManiSkill; 稳定窗口用仅未来 lookahead)
# 约定: 喂给状态机的 close_frac, HIGH=物理闭合。
# RoboTwin gripper_val: 1=开、0=闭 → close_frac = 1.0 - gripper_val
# settled: 当前+未来 lookahead 窗口跨度 < eps; 短 unsettled 空隙填回;
# early CLOSED: close_frac 很高且前向稳定时可提前确认闭合。
# 骨线: settled 二值开合色; 运动帧统一灰色。
# 关节: settled 开红/闭绿; 运动帧固定红色。
GRIPPER_CLOSED_OPEN_FRAC = 0.5
GRIPPER_STATE_LOOKAHEAD = 5
GRIPPER_STATE_LOOKBACK = 0
GRIPPER_STATE_PLATEAU_EPS = 0.03
GRIPPER_STATE_CHANGE_DELTA = 0.1
GRIPPER_STATE_SETTLED_FILL = 3
GRIPPER_EARLY_CLOSED_FRAC = 0.85

# ---------------------------------------------------------------------------
# 从 config.yml 读取的 gripper_scale（RoboTwin Robot.set_gripper 中使用）
# gripper_val ∈ [0,1] → joint_val = scale[0] + val * (scale[1] - scale[0])
# ---------------------------------------------------------------------------
DEFAULT_GRIPPER_SCALE = (-0.01, 0.045)


def _load_robot_config(config_path: str) -> dict:
    """从 config.yml 解析 robot_pose / gripper_scale 等参数。"""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def _build_robot_pose_matrix(config: dict) -> np.ndarray:
    """从 config.yml 的 robot_pose 构造 4×4 齐次变换矩阵。

    config.yml 格式: robot_pose: [[x, y, z, qw, qx, qy, qz]]
    SAPIEN quaternion 约定 (w, x, y, z)。
    """
    from scipy.spatial.transform import Rotation

    pose = config.get("robot_pose", [[0, 0, 0, 1, 0, 0, 0]])[0]
    pos = np.array(pose[:3], dtype=np.float64)
    quat_wxyz = pose[3:]
    quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
    R = Rotation.from_quat(quat_xyzw).as_matrix()
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = pos
    return T


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
def load_jsonl(path: str) -> List[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _resolve_hdf5_path(item: dict, hdf5_root: str = "") -> str:
    """Resolve HDF5 path: use *hdf5_root* (if given) instead of item['root'].

    兼容两种 JSONL action 格式:
      格式1: {"action": {"hdf5_path": "..."}}
      格式2: {"action": {"action": {"hdf5_path": "..."}}}  (双层嵌套)

    兼容磁盘路径多一层同名子目录的情况:
      JSONL:  task/robot_config/data/episode.hdf5
      磁盘:   task/robot_config/robot_config/data/episode.hdf5
    """
    root = hdf5_root or item.get("root", "")
    action_info = item.get("action", {})
    hdf5_rel = action_info.get("hdf5_path", "")
    if not hdf5_rel and isinstance(action_info.get("action"), dict):
        hdf5_rel = action_info["action"].get("hdf5_path", "")
    if not hdf5_rel:
        raise ValueError("action['hdf5_path'] 为空")

    hdf5_path = os.path.join(root, hdf5_rel) if not hdf5_rel.startswith("/") else hdf5_rel
    if not os.path.exists(hdf5_path):
        # 尝试补一层同名子目录: task/cfg/data/ep.hdf5 → task/cfg/cfg/data/ep.hdf5
        parts = hdf5_rel.split("/")
        if len(parts) >= 3:
            fixed = os.path.join(parts[0], parts[1], parts[1], *parts[2:])
            fixed_path = os.path.join(root, fixed)
            if os.path.exists(fixed_path):
                return fixed_path
        raise FileNotFoundError(f"HDF5 不存在: {hdf5_path}")
    return hdf5_path


def load_action_from_hdf5(
    item: dict, hdf5_root: str = "",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """加载一条样本的 action 数据。

    Returns:
        left_arm  (T, 6), left_gripper (T,), right_arm (T, 6), right_gripper (T,)
    """
    hdf5_path = _resolve_hdf5_path(item, hdf5_root)

    with h5py.File(hdf5_path, "r") as f:
        left_arm = f["joint_action/left_arm"][:].astype(np.float64)
        left_gripper = f["joint_action/left_gripper"][:].astype(np.float64)
        right_arm = f["joint_action/right_arm"][:].astype(np.float64)
        right_gripper = f["joint_action/right_gripper"][:].astype(np.float64)

    return left_arm, left_gripper, right_arm, right_gripper


def load_camera_params_from_hdf5(
    item: dict, camera_name: str = "head_camera", hdf5_root: str = "",
) -> Tuple[np.ndarray, np.ndarray]:
    """加载相机内参和外参。

    内参取第 0 帧（RoboTwin 各视角内参基本静态）；外参返回全时序，
    因为 left/right 腕部相机的 extrinsic_cv 随手臂运动变化。

    Returns:
        intrinsic (3, 3), extrinsic (T, 3, 4)
    """
    hdf5_path = _resolve_hdf5_path(item, hdf5_root)

    with h5py.File(hdf5_path, "r") as f:
        key_prefix = f"observation/{camera_name}"
        intrinsic = f[f"{key_prefix}/intrinsic_cv"][0].astype(np.float64)
        extrinsic = f[f"{key_prefix}/extrinsic_cv"][:].astype(np.float64)

    return intrinsic, extrinsic


def load_all_from_hdf5(
    item: dict, hdf5_root: str = "", camera_name: str = "head_camera",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray]:
    """一次打开 HDF5，同时读 action 数据和相机参数。

    对网络存储（PFS/NFS）是关键优化：把 "open + 2 次读" 合并为 "open + 1 次读"，
    减少一次 metadata 往返（对 NAS/PFS 通常 1~10ms）。

    Returns:
        left_arm (T,6), left_gripper (T,), right_arm (T,6), right_gripper (T,),
        intrinsic (3,3), extrinsic (T, 3, 4)
    """
    hdf5_path = _resolve_hdf5_path(item, hdf5_root)
    with h5py.File(hdf5_path, "r") as f:
        left_arm = f["joint_action/left_arm"][:].astype(np.float64)
        left_gripper = f["joint_action/left_gripper"][:].astype(np.float64)
        right_arm = f["joint_action/right_arm"][:].astype(np.float64)
        right_gripper = f["joint_action/right_gripper"][:].astype(np.float64)
        key_prefix = f"observation/{camera_name}"
        intrinsic = f[f"{key_prefix}/intrinsic_cv"][0].astype(np.float64)
        extrinsic = f[f"{key_prefix}/extrinsic_cv"][:].astype(np.float64)
    return left_arm, left_gripper, right_arm, right_gripper, intrinsic, extrinsic


def _resolve_video_path(root: str, rel_or_abs: str) -> str:
    if not rel_or_abs:
        return ""
    if rel_or_abs.startswith("/"):
        return rel_or_abs
    path = os.path.join(root, rel_or_abs)
    if not os.path.exists(path):
        parts = rel_or_abs.split("/")
        if len(parts) >= 3:
            fixed = os.path.join(parts[0], parts[1], parts[1], *parts[2:])
            fixed_path = os.path.join(root, fixed)
            if os.path.exists(fixed_path):
                return fixed_path
    return path


def get_head_view_video_path(item: dict) -> Tuple[str, str]:
    """获取 head 视角视频路径。返回 (video_path, view_name)。"""
    root = item.get("root", "")
    cvi = item.get("camera_video_info", {})

    for target in HEAD_VIEW_NAMES:
        if target in cvi:
            vp = cvi[target].get("video", "")
            if vp:
                return _resolve_video_path(root, vp), target

    for vn, vi in cvi.items():
        vp = vi.get("video", "")
        if vp:
            return _resolve_video_path(root, vp), vn

    raise ValueError("没有可用的视频路径")


# ---------------------------------------------------------------------------
# Worker 全局状态
# ---------------------------------------------------------------------------
_worker_robot = None
_worker_robot_pose: Optional[np.ndarray] = None  # URDF root → world 变换
_worker_gripper_scale: Tuple[float, float] = DEFAULT_GRIPPER_SCALE
_worker_intrinsic: Optional[np.ndarray] = None
_worker_extrinsic: Optional[np.ndarray] = None
_worker_view_matrix: Optional[np.ndarray] = None
_worker_fx: Optional[float] = None
_worker_fy: Optional[float] = None
_worker_cx: Optional[float] = None
_worker_cy: Optional[float] = None
_worker_img_w: int = 320
_worker_img_h: int = 240

_worker_ffmpeg_preset: str = "veryfast"
_worker_ffmpeg_threads: int = 1
_worker_ffmpeg_crf: int = 23
_worker_kin: Optional[Dict[str, Any]] = None
_worker_incoming_joint_by_link: Dict[str, str] = {}
_worker_active_bones: List[Tuple[str, str]] = []  # (joint_name, parent_link)


def _init_worker(urdf_path: str, img_w: int, img_h: int,
                 robot_pose_4x4: np.ndarray,
                 gripper_scale: Tuple[float, float],
                 ffmpeg_preset: str = "veryfast",
                 ffmpeg_threads: int = 1,
                 ffmpeg_crf: int = 23):
    global _worker_robot, _worker_img_w, _worker_img_h
    global _worker_robot_pose, _worker_gripper_scale
    global _worker_ffmpeg_preset, _worker_ffmpeg_threads, _worker_ffmpeg_crf

    import warnings
    import logging
    from yourdfpy import URDF

    # 忽略子进程的 SIGINT，让主进程统一处理 Ctrl-C（避免 KeyboardInterrupt 刷屏）
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # 限制 OpenCV 线程数，避免每个 worker 内部并发膨胀
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass

    # 避免 numpy 在 worker 里起多线程抢 CPU
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(k, "1")

    warnings.filterwarnings("ignore", message=".*mimic.*")
    logging.getLogger("yourdfpy").setLevel(logging.ERROR)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        old_level = logging.root.level
        logging.root.setLevel(logging.ERROR)
        _worker_robot = URDF.load(urdf_path, load_meshes=False, build_scene_graph=True)
        logging.root.setLevel(old_level)

    _worker_img_w = img_w
    _worker_img_h = img_h
    _worker_robot_pose = robot_pose_4x4
    _worker_gripper_scale = gripper_scale
    _worker_ffmpeg_preset = ffmpeg_preset
    _worker_ffmpeg_threads = max(1, int(ffmpeg_threads))
    _worker_ffmpeg_crf = int(ffmpeg_crf)
    _build_worker_kinematics(_worker_robot)
    # 不打印每个 worker 的初始化信息（180 核会刷屏）；仅在进程 0 打印一次
    if os.getpid() % 256 == 0:
        print(f"[Worker {os.getpid()}] ready: {img_w}x{img_h}, "
              f"preset={ffmpeg_preset}, threads={_worker_ffmpeg_threads}, "
              f"crf={_worker_ffmpeg_crf}")


def _build_worker_kinematics(robot) -> None:
    """从 yourdfpy robot 提取静态运动学树，供向量化 FK 使用。"""
    global _worker_kin, _worker_incoming_joint_by_link, _worker_active_bones
    from collections import deque

    joints: Dict[str, Dict[str, Any]] = {}
    children_by_parent: Dict[str, List[Tuple[str, str]]] = {}
    link_parent_joint: Dict[str, str] = {}
    all_child_links: set = set()
    link_set: set = set()
    incoming: Dict[str, str] = {}
    active_bones: List[Tuple[str, str]] = []

    for jn, j in robot.joint_map.items():
        name = str(jn)
        jtype = str(getattr(j, "joint_type", getattr(j, "type", "")))
        parent = _link_name(getattr(j, "parent", None))
        child = _link_name(getattr(j, "child", None))
        origin = _as_transform(getattr(j, "origin", None))
        axis_raw = getattr(j, "axis", None)
        if axis_raw is None:
            axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            axis = np.asarray(axis_raw, dtype=np.float64).reshape(3)
            nrm = float(np.linalg.norm(axis))
            if nrm > 1e-12 and jtype in ("revolute", "continuous", "prismatic"):
                axis = axis / nrm
        mimic = None
        mim = getattr(j, "mimic", None)
        if mim is not None:
            msrc = getattr(mim, "joint", None)
            if msrc is not None:
                mimic = (
                    str(msrc),
                    float(getattr(mim, "multiplier", 1.0) or 1.0),
                    float(getattr(mim, "offset", 0.0) or 0.0),
                )
        joints[name] = {
            "type": jtype,
            "parent": parent,
            "child": child,
            "origin": origin,
            "axis": axis,
            "mimic": mimic,
        }
        if child is not None:
            incoming[child] = name
        if parent is not None:
            link_set.add(parent)
        if child is not None:
            link_set.add(child)
            all_child_links.add(child)
            link_parent_joint[child] = name
            children_by_parent.setdefault(parent, []).append((name, child))
        if name in ALL_ACTION_JOINTS and jtype != "fixed" and parent is not None:
            active_bones.append((name, parent))

    roots = sorted(l for l in link_set if l not in all_child_links)
    topo: List[str] = []
    seen: set = set(roots)
    dq = deque(roots)
    while dq:
        p = dq.popleft()
        topo.append(p)
        for _jname, child in children_by_parent.get(p, []):
            if child not in seen:
                seen.add(child)
                dq.append(child)

    _worker_kin = {
        "joints": joints,
        "roots": roots,
        "topo": topo,
        "link_parent_joint": link_parent_joint,
        "mimic_joints": [n for n, info in joints.items() if info["mimic"] is not None],
    }
    _worker_incoming_joint_by_link = incoming
    _worker_active_bones = active_bones


def _axis_rotation_batch(axis: np.ndarray, q: np.ndarray) -> np.ndarray:
    ax, ay, az = float(axis[0]), float(axis[1]), float(axis[2])
    c = np.cos(q)
    s = np.sin(q)
    cc = 1.0 - c
    t = q.shape[0]
    r = np.empty((t, 3, 3), dtype=np.float64)
    r[:, 0, 0] = c + ax * ax * cc
    r[:, 0, 1] = ax * ay * cc - az * s
    r[:, 0, 2] = ax * az * cc + ay * s
    r[:, 1, 0] = ay * ax * cc + az * s
    r[:, 1, 1] = c + ay * ay * cc
    r[:, 1, 2] = ay * az * cc - ax * s
    r[:, 2, 0] = az * ax * cc - ay * s
    r[:, 2, 1] = az * ay * cc + ax * s
    r[:, 2, 2] = c + az * az * cc
    return r


def _joint_local_batch(info: Dict[str, Any], q: np.ndarray, t: int) -> np.ndarray:
    origin = info["origin"]
    jtype = info["type"]
    if jtype in ("revolute", "continuous"):
        motion = np.broadcast_to(np.eye(4, dtype=np.float64), (t, 4, 4)).copy()
        motion[:, :3, :3] = _axis_rotation_batch(info["axis"], q)
        return origin[None] @ motion
    if jtype == "prismatic":
        motion = np.broadcast_to(np.eye(4, dtype=np.float64), (t, 4, 4)).copy()
        motion[:, :3, 3] = info["axis"][None, :] * q[:, None]
        return origin[None] @ motion
    return np.broadcast_to(origin, (t, 4, 4))


def _fk_urdf_transforms_batch(
    cfg_arrays: Dict[str, np.ndarray], t: int,
) -> Dict[str, np.ndarray]:
    """URDF root 系下 link -> (T,4,4)。"""
    kin = _worker_kin
    assert kin is not None
    cfg = dict(cfg_arrays)
    for jname in kin["mimic_joints"]:
        src, mult, off = kin["joints"][jname]["mimic"]
        if src in cfg:
            cfg[jname] = mult * cfg[src] + off
    eye_t = np.broadcast_to(np.eye(4, dtype=np.float64), (t, 4, 4))
    world: Dict[str, np.ndarray] = {r: eye_t for r in kin["roots"]}
    for link in kin["topo"]:
        if link in world:
            continue
        jname = kin["link_parent_joint"].get(link)
        if jname is None:
            world[link] = eye_t
            continue
        info = kin["joints"][jname]
        q = cfg.get(jname)
        if q is None:
            q = np.zeros(t, dtype=np.float64)
        world[link] = world[info["parent"]] @ _joint_local_batch(info, q, t)
    return world


def _gripper_val_to_joint_batch(val: np.ndarray) -> np.ndarray:
    s0, s1 = _worker_gripper_scale
    real = s0 + np.clip(val, 0.0, 1.0) * (s1 - s0)
    return np.clip(real, GRIPPER_PRISMATIC_LIMIT[0], GRIPPER_PRISMATIC_LIMIT[1])


def _action_to_cfg_batch(
    left_arm: np.ndarray, left_gripper: np.ndarray,
    right_arm: np.ndarray, right_gripper: np.ndarray,
) -> Dict[str, np.ndarray]:
    cfg: Dict[str, np.ndarray] = {}
    for i, jn in enumerate(LEFT_ARM_JOINTS):
        cfg[jn] = left_arm[:, i].astype(np.float64)
    for i, jn in enumerate(RIGHT_ARM_JOINTS):
        cfg[jn] = right_arm[:, i].astype(np.float64)
    lg = _gripper_val_to_joint_batch(np.asarray(left_gripper, dtype=np.float64))
    rg = _gripper_val_to_joint_batch(np.asarray(right_gripper, dtype=np.float64))
    cfg[LEFT_GRIPPER_DRIVING_JOINT] = lg
    cfg[LEFT_GRIPPER_MIMIC_JOINT] = lg
    cfg[RIGHT_GRIPPER_DRIVING_JOINT] = rg
    cfg[RIGHT_GRIPPER_MIMIC_JOINT] = rg
    return cfg


def _extrinsics_to_view_batch(extrinsics: np.ndarray, t: int) -> np.ndarray:
    """OpenCV (3,4) 或 (T,3,4) → view (T,4,4)。"""
    ext = np.asarray(extrinsics, dtype=np.float64)
    view = np.zeros((t, 4, 4), dtype=np.float64)
    view[:, 3, 3] = 1.0
    if ext.ndim == 2:
        view[:, :3, :] = ext[None, :, :]
    else:
        n = int(ext.shape[0])
        if n == t:
            view[:, :3, :] = ext
        elif n == 1:
            view[:, :3, :] = ext[0][None, :, :]
        else:
            idx = np.minimum(np.arange(t), n - 1)
            view[:, :3, :] = ext[idx]
    return view


def _urdf_pts_to_world(pts_urdf: np.ndarray) -> np.ndarray:
    """(T,3) URDF root → (T,3) world，经 robot_pose。"""
    pose = _worker_robot_pose
    assert pose is not None
    r = pose[:3, :3]
    tr = pose[:3, 3]
    return pts_urdf @ r.T + tr


def _project_batch(
    view: np.ndarray,
    pts_world: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """view (T,4,4), pts (T,3) → u,v,valid。"""
    t = pts_world.shape[0]
    ph = np.concatenate([pts_world, np.ones((t, 1), dtype=np.float64)], axis=1)
    p = np.einsum("tij,tj->ti", view, ph)
    z = p[:, 2]
    valid = z >= 0.01
    zz = np.where(valid, z, 1.0)
    u = np.round(fx * p[:, 0] / zz + cx).astype(np.int64)
    v = np.round(fy * p[:, 1] / zz + cy).astype(np.int64)
    return u, v, valid


def _gripper_color_traj(
    closed: np.ndarray, settled: np.ndarray, transparent: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """返回 bone_col (T,C), joint_col (T,C)，C=3 或 4。"""
    t = int(closed.shape[0])
    c = 4 if transparent else 3
    bone = np.empty((t, c), dtype=np.int64)
    joint = np.empty((t, c), dtype=np.int64)
    for i in range(t):
        bc = _gripper_bone_color(bool(closed[i]), bool(settled[i]), transparent)
        jc = _gripper_joint_color(bool(closed[i]), bool(settled[i]), transparent)
        bone[i] = bc
        joint[i] = jc
    return bone, joint


def _precompute_skeleton_draw(
    left_arm: np.ndarray, left_gripper: np.ndarray,
    right_arm: np.ndarray, right_gripper: np.ndarray,
    extrinsics: np.ndarray,
    closed_l: np.ndarray, settled_l: np.ndarray,
    closed_r: np.ndarray, settled_r: np.ndarray,
    transparent: bool = False,
) -> Tuple[list, list]:
    """整段 episode 向量化 FK + 投影，返回 (bones, joints) 绘制轨迹。

    bones: list of (u1,v1,u2,v2,valid, bone_color_traj)
    joints: list of (u,v,valid, joint_color_traj)
    """
    assert _worker_kin is not None
    t = int(left_arm.shape[0])
    cfg = _action_to_cfg_batch(left_arm, left_gripper, right_arm, right_gripper)
    urdf_world = _fk_urdf_transforms_batch(cfg, t)
    view = _extrinsics_to_view_batch(extrinsics, t)
    fx, fy, cx, cy = float(_worker_fx), float(_worker_fy), float(_worker_cx), float(_worker_cy)

    # joint points in URDF frame (match per-frame _draw_frame semantics)
    joint_pts_urdf: Dict[str, np.ndarray] = {}
    for jname, parent in _worker_active_bones:
        info = _worker_kin["joints"][jname]
        jtype = info["type"]
        child = info["child"]
        if jtype == "prismatic" and child is not None and child in urdf_world:
            joint_pts_urdf[jname] = urdf_world[child][:, :3, 3]
        else:
            parent_T = urdf_world[parent]
            j_t = parent_T @ info["origin"][None]
            joint_pts_urdf[jname] = j_t[:, :3, 3]

    def link_anchor_urdf(link_name: str) -> Optional[np.ndarray]:
        inc = _worker_incoming_joint_by_link.get(link_name)
        if inc is not None and inc in joint_pts_urdf:
            return joint_pts_urdf[inc]
        if link_name in urdf_world:
            return urdf_world[link_name][:, :3, 3]
        return None

    lg_bone, lg_joint = _gripper_color_traj(closed_l, settled_l, transparent)
    rg_bone, rg_joint = _gripper_color_traj(closed_r, settled_r, transparent)
    if transparent:
        arm_bone = np.tile(np.array([*COLOR_ARM_BONE, 255], dtype=np.int64), (t, 1))
        arm_joint = np.tile(np.array([*COLOR_ARM_JOINT, 255], dtype=np.int64), (t, 1))
    else:
        arm_bone = np.tile(np.array(COLOR_ARM_BONE, dtype=np.int64), (t, 1))
        arm_joint = np.tile(np.array(COLOR_ARM_JOINT, dtype=np.int64), (t, 1))

    bones = []
    joints = []
    for jname, parent in _worker_active_bones:
        if jname not in joint_pts_urdf:
            continue
        p_anchor = link_anchor_urdf(parent)
        if p_anchor is None:
            continue
        p1w = _urdf_pts_to_world(p_anchor)
        p2w = _urdf_pts_to_world(joint_pts_urdf[jname])
        u1, v1, val1 = _project_batch(view, p1w, fx, fy, cx, cy)
        u2, v2, val2 = _project_batch(view, p2w, fx, fy, cx, cy)
        valid = val1 & val2
        if jname in LEFT_GRIPPER_JOINT_SET:
            bcol, jcol = lg_bone, lg_joint
        elif jname in RIGHT_GRIPPER_JOINT_SET:
            bcol, jcol = rg_bone, rg_joint
        else:
            bcol, jcol = arm_bone, arm_joint
        bones.append((u1, v1, u2, v2, valid, bcol))
        joints.append((u2, v2, valid, jcol))
    return bones, joints


def _blit_skeleton_frame(
    img: np.ndarray,
    bones: list,
    joints: list,
    t: int,
) -> np.ndarray:
    for u1, v1, u2, v2, valid, bcol in bones:
        if valid[t]:
            c = tuple(int(x) for x in bcol[t])
            cv2.line(img, (int(u1[t]), int(v1[t])), (int(u2[t]), int(v2[t])),
                     c, 6, cv2.LINE_AA)
    for u, v, valid, jcol in joints:
        if valid[t]:
            c = tuple(int(x) for x in jcol[t])
            cv2.circle(img, (int(u[t]), int(v[t])), 9, c, -1, cv2.LINE_AA)
    return img


def _build_ffmpeg_cmd(output_path: str, img_w: int, img_h: int,
                      fps: float) -> List[str]:
    """构造 ffmpeg rawvideo → libx264 管道命令。

    注意：我们用 .part 临时文件做原子写，但 ffmpeg 会按扩展名推断 muxer，
    而 .part 不是已知扩展名，所以必须显式 `-f mp4`。
    """
    return [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{img_w}x{img_h}",
        "-pix_fmt", "rgb24", "-r", f"{fps:g}",
        "-i", "-",
        "-c:v", "libx264",
        "-preset", _worker_ffmpeg_preset,
        "-crf", str(_worker_ffmpeg_crf),
        "-threads", str(_worker_ffmpeg_threads),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-f", "mp4",
        "-loglevel", "error",
        output_path,
    ]


def _set_view_matrix_from_extrinsic(extrinsic: np.ndarray) -> None:
    """用单帧 OpenCV world-to-camera 外参 (3,4) 更新投影 view matrix。"""
    global _worker_extrinsic, _worker_view_matrix
    ext = np.asarray(extrinsic, dtype=np.float64)
    if ext.ndim == 3:
        ext = ext[0]
    _worker_extrinsic = ext
    view_4x4 = np.eye(4, dtype=np.float64)
    view_4x4[:3, :] = ext
    _worker_view_matrix = view_4x4


def _setup_camera_from_params(intrinsic: np.ndarray, extrinsic: np.ndarray,
                              img_w: int, img_h: int):
    """从 HDF5 中读取的内参/外参设置投影参数。

    intrinsic: (3,3)  [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    extrinsic: (3,4) 或 (T,3,4)  [R | t]  —— OpenCV 约定 (world-to-camera)
               若传入时序，先用第 0 帧初始化；逐帧渲染时再调用
               _set_view_matrix_from_extrinsic 更新。
    """
    global _worker_intrinsic, _worker_extrinsic, _worker_view_matrix
    global _worker_fx, _worker_fy, _worker_cx, _worker_cy

    src_fx = intrinsic[0, 0]
    src_fy = intrinsic[1, 1]
    src_cx = intrinsic[0, 2]
    src_cy = intrinsic[1, 2]
    src_w = int(src_cx * 2)
    src_h = int(src_cy * 2)

    scale_x = img_w / src_w
    scale_y = img_h / src_h

    _worker_fx = src_fx * scale_x
    _worker_fy = src_fy * scale_y
    _worker_cx = src_cx * scale_x
    _worker_cy = src_cy * scale_y

    _worker_intrinsic = intrinsic
    _set_view_matrix_from_extrinsic(extrinsic)


def _clip(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def _gripper_val_to_joint(val: float) -> float:
    """RoboTwin gripper_val [0,1] → URDF prismatic joint position。

    与 RoboTwin Robot.set_gripper 一致：
        real = scale[0] + val * (scale[1] - scale[0])
    最后 clamp 到 URDF 关节限位。
    """
    s0, s1 = _worker_gripper_scale
    real = s0 + _clip(val, 0.0, 1.0) * (s1 - s0)
    return _clip(real, GRIPPER_PRISMATIC_LIMIT[0], GRIPPER_PRISMATIC_LIMIT[1])


def _fill_short_false_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    """把长度 <= max_gap 的 False 段填成 True (抑制 settled 闪烁)。"""
    out = np.asarray(mask, dtype=bool).copy()
    max_gap = max(0, int(max_gap))
    if max_gap <= 0 or out.size == 0:
        return out
    t_len = int(out.shape[0])
    t = 0
    while t < t_len:
        if out[t]:
            t += 1
            continue
        t0 = t
        while t < t_len and not out[t]:
            t += 1
        if (t - t0) <= max_gap:
            out[t0:t] = True
    return out


def _gripper_state_masks(
    close_frac_side: np.ndarray,
    closed_open_frac: float = GRIPPER_CLOSED_OPEN_FRAC,
    lookahead: int = GRIPPER_STATE_LOOKAHEAD,
    eps: float = GRIPPER_STATE_PLATEAU_EPS,
    change_delta: float = GRIPPER_STATE_CHANGE_DELTA,
    lookback: int = GRIPPER_STATE_LOOKBACK,
    settled_fill: int = GRIPPER_STATE_SETTLED_FILL,
    early_closed_frac: float = GRIPPER_EARLY_CLOSED_FRAC,
) -> Tuple[np.ndarray, np.ndarray]:
    """逐侧时序状态机, 返回 (closed_mask, settled_mask)。对齐 RealSource / ManiSkill。

    close_frac_side: (T,) HIGH=物理闭合。
    closed_mask: True=闭合(橙)、False=张开(蓝); 运动帧保持上一确认态(迟滞)。
    settled_mask: True=平台稳定帧(骨线用二值色); False=运动帧(骨线灰色)。
    """
    ov = np.clip(np.asarray(close_frac_side, dtype=np.float64), 0.0, 1.0)
    t_len = int(ov.shape[0])
    if t_len == 0:
        empty = np.zeros(0, dtype=bool)
        return empty, empty
    lookahead = max(0, int(lookahead))
    lookback = max(0, int(lookback))
    eps = float(eps)
    change_delta = float(change_delta)
    closed_open_frac = float(closed_open_frac)
    early_closed_frac = float(early_closed_frac)

    settled = np.zeros(t_len, dtype=bool)
    plateau_val = np.zeros(t_len, dtype=np.float64)
    for t in range(t_len):
        lo = max(0, t - lookback)
        hi = min(t_len, t + lookahead + 1)
        window = ov[lo:hi]
        settled[t] = (float(window.max()) - float(window.min())) < eps
        plateau_val[t] = float(np.median(window))
        if (not settled[t]) and ov[t] >= early_closed_frac:
            fwd = ov[t:hi]
            if (float(fwd.max()) - float(fwd.min())) < eps:
                settled[t] = True
                plateau_val[t] = float(np.median(fwd))

    settled = _fill_short_false_gaps(settled, settled_fill)

    settled_idx = np.flatnonzero(settled)
    if settled_idx.size == 0:
        return ov >= closed_open_frac, np.ones(t_len, dtype=bool)

    ref = plateau_val[int(settled_idx[0])]
    state_closed = ref >= closed_open_frac
    closed = np.zeros(t_len, dtype=bool)
    for t in range(t_len):
        if settled[t]:
            v = plateau_val[t]
            if abs(v - ref) > change_delta:
                state_closed = v > ref
            ref = v
        closed[t] = state_closed
    return closed, settled


def _gripper_bone_color(closed: bool, settled: bool, transparent: bool = False) -> tuple:
    if not settled:
        c = COLOR_GRIPPER_MOTION_BONE
    else:
        c = COLOR_GRIPPER_CLOSED_BONE if closed else COLOR_GRIPPER_OPEN_BONE
    return (*c, 255) if transparent else c


def _gripper_joint_color(closed: bool, settled: bool, transparent: bool = False) -> tuple:
    if not settled:
        c = COLOR_GRIPPER_MOTION_JOINT
    else:
        c = COLOR_GRIPPER_CLOSED_JOINT if closed else COLOR_GRIPPER_OPEN_JOINT
    return (*c, 255) if transparent else c


def _action_to_joint_cfg(
    left_arm: np.ndarray, left_gripper: float,
    right_arm: np.ndarray, right_gripper: float,
) -> Dict[str, float]:
    """将单帧 action 转为 URDF cfg。"""
    cfg: Dict[str, float] = {}

    for i, jn in enumerate(LEFT_ARM_JOINTS):
        cfg[jn] = float(left_arm[i])
    for i, jn in enumerate(RIGHT_ARM_JOINTS):
        cfg[jn] = float(right_arm[i])

    lg = _gripper_val_to_joint(float(left_gripper))
    rg = _gripper_val_to_joint(float(right_gripper))

    cfg[LEFT_GRIPPER_DRIVING_JOINT] = lg
    cfg[LEFT_GRIPPER_MIMIC_JOINT] = lg
    cfg[RIGHT_GRIPPER_DRIVING_JOINT] = rg
    cfg[RIGHT_GRIPPER_MIMIC_JOINT] = rg

    return cfg


# ---------------------------------------------------------------------------
# 透视投影
# ---------------------------------------------------------------------------
def _project(pt_urdf: np.ndarray) -> Optional[Tuple[int, int]]:
    """将 URDF 局部坐标 → robot_pose 变换到世界系 → 相机投影到 2D 像素。"""
    pt_world = _worker_robot_pose @ np.array(
        [pt_urdf[0], pt_urdf[1], pt_urdf[2], 1.0])
    p_cam = _worker_view_matrix @ pt_world
    z = p_cam[2]
    if z < 0.01:
        return None
    u = int(round(_worker_fx * p_cam[0] / z + _worker_cx))
    v = int(round(_worker_fy * p_cam[1] / z + _worker_cy))
    return (u, v)


# ---------------------------------------------------------------------------
# 单帧绘制
# ---------------------------------------------------------------------------
def _draw_frame(
    left_arm_t: np.ndarray, left_gripper_t: float,
    right_arm_t: np.ndarray, right_gripper_t: float,
    show_inactive: bool = False,
    transparent: bool = False,
    extrinsic_t: Optional[np.ndarray] = None,
    left_closed: bool = False,
    left_settled: bool = True,
    right_closed: bool = False,
    right_settled: bool = True,
) -> np.ndarray:
    robot = _worker_robot
    img_w, img_h = _worker_img_w, _worker_img_h

    if extrinsic_t is not None:
        _set_view_matrix_from_extrinsic(extrinsic_t)

    cfg = _action_to_joint_cfg(left_arm_t, left_gripper_t, right_arm_t, right_gripper_t)
    robot.update_cfg(cfg)

    incoming_joint_by_link: Dict[str, str] = {}
    for jn, j in robot.joint_map.items():
        child = getattr(j, "child", None)
        if child is not None:
            incoming_joint_by_link[str(child)] = str(jn)

    joint_points: Dict[str, np.ndarray] = {}
    for jn, j in robot.joint_map.items():
        jtype = getattr(j, "joint_type", getattr(j, "type", None))
        if str(jtype) == "fixed":
            continue
        parent_name = _link_name(getattr(j, "parent", None))
        if parent_name is None:
            continue

        if str(jtype) == "prismatic":
            child_name = _link_name(getattr(j, "child", None))
            if child_name is not None:
                child_T = _as_transform(robot.get_transform(child_name))
                joint_points[str(jn)] = child_T[:3, 3].copy()
                continue

        parent_T = _as_transform(robot.get_transform(parent_name))
        j_origin = _as_transform(getattr(j, "origin", None))
        j_T = parent_T @ j_origin
        joint_points[str(jn)] = j_T[:3, 3].copy()

    def link_anchor(link_name: str):
        inc = incoming_joint_by_link.get(link_name)
        if inc is not None and inc in joint_points:
            return joint_points[inc]
        try:
            tf = robot.get_transform(link_name)
        except Exception:
            return None
        return np.asarray(tf, dtype=np.float64)[:3, 3]

    if transparent:
        img = np.zeros((img_h, img_w, 4), dtype=np.uint8)
        color_active_bone = (*COLOR_ARM_BONE, 255)
        color_active_joint = (*COLOR_ARM_JOINT, 255)
        color_inactive = (*COLOR_INACTIVE, 180)
    else:
        img = np.full((img_h, img_w, 3), 255, dtype=np.uint8)
        color_active_bone = COLOR_ARM_BONE
        color_active_joint = COLOR_ARM_JOINT
        color_inactive = COLOR_INACTIVE

    lg_bone = _gripper_bone_color(left_closed, left_settled, transparent)
    rg_bone = _gripper_bone_color(right_closed, right_settled, transparent)
    lg_joint = _gripper_joint_color(left_closed, left_settled, transparent)
    rg_joint = _gripper_joint_color(right_closed, right_settled, transparent)

    for jn, j in robot.joint_map.items():
        jname = str(jn)
        if jname not in joint_points:
            continue
        is_active = jname in ALL_ACTION_JOINTS
        if not is_active and not show_inactive:
            continue
        parent_name = _link_name(getattr(j, "parent", None))
        if parent_name is None:
            continue
        p_anchor = link_anchor(parent_name)
        if p_anchor is None:
            continue

        uv1 = _project(p_anchor)
        uv2 = _project(joint_points[jname])
        if uv1 is not None and uv2 is not None:
            if is_active:
                bone_color = color_active_bone
                if jname in LEFT_GRIPPER_JOINT_SET:
                    bone_color = lg_bone
                elif jname in RIGHT_GRIPPER_JOINT_SET:
                    bone_color = rg_bone
                cv2.line(img, uv1, uv2, bone_color, 6, cv2.LINE_AA)
            else:
                cv2.line(img, uv1, uv2, color_inactive, 1, cv2.LINE_AA)

    for jname, pos in joint_points.items():
        is_active = jname in ALL_ACTION_JOINTS
        if not is_active and not show_inactive:
            continue
        uv = _project(pos)
        if uv is not None:
            if is_active:
                joint_color = color_active_joint
                if jname in LEFT_GRIPPER_JOINT_SET:
                    joint_color = lg_joint
                elif jname in RIGHT_GRIPPER_JOINT_SET:
                    joint_color = rg_joint
                cv2.circle(img, uv, 9, joint_color, -1, cv2.LINE_AA)
            else:
                cv2.circle(img, uv, 3, color_inactive, -1, cv2.LINE_AA)

    return img


def _overlay_frame(bg_rgb: np.ndarray, skeleton_rgba: np.ndarray,
                   overlay_alpha: float = 0.7) -> np.ndarray:
    alpha = skeleton_rgba[:, :, 3:4].astype(np.float32) / 255.0 * overlay_alpha
    fg = skeleton_rgba[:, :, :3].astype(np.float32)
    bg = bg_rgb.astype(np.float32)
    blended = fg * alpha + bg * (1.0 - alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 单条样本渲染
# ---------------------------------------------------------------------------
def _safe_unlink(path: Optional[str]) -> None:
    if not path:
        return
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def _encode_to_ffmpeg(cmd: List[str], T: int, producer) -> Tuple[int, str]:
    """统一的 ffmpeg 管道写帧逻辑（流式、低内存、死锁安全）。

    producer(t) -> np.ndarray (uint8, shape = (h, w, 3))。
    返回 (returncode, stderr_tail)。

    使用独立线程持续消费 stderr，避免 ffmpeg stderr 写满管道时死锁。
    """
    import threading

    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
        bufsize=0,
    )

    stderr_chunks: List[bytes] = []

    def _drain_stderr() -> None:
        try:
            while True:
                chunk = proc.stderr.read(4096) if proc.stderr else b""
                if not chunk:
                    break
                stderr_chunks.append(chunk)
        except Exception:
            pass

    stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
    stderr_thread.start()

    stderr_tail = ""
    try:
        try:
            for t in range(T):
                frame = producer(t)
                proc.stdin.write(frame.tobytes())
        except BrokenPipeError:
            # ffmpeg 已退出，走到 wait 拿 returncode
            pass
        try:
            proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        try:
            proc.wait(timeout=300)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
    except Exception as e:
        stderr_tail = f"exception: {e!r}"[-200:]
        try:
            proc.kill()
            proc.wait(timeout=10)
        except Exception:
            pass
    finally:
        stderr_thread.join(timeout=5.0)

    if not stderr_tail:
        err = b"".join(stderr_chunks)
        if err:
            stderr_tail = err.decode(errors="replace")[-200:]

    return (proc.returncode if proc.returncode is not None else -1, stderr_tail)


def render_one(task: dict) -> dict:
    idx = task["idx"]
    item = task["item"]
    output_path = task["output_path"]
    concat_output_path = task.get("concat_output_path")
    original_video_path = task.get("original_video_path")
    overlay_output_path = task.get("overlay_output_path")
    fps = float(task.get("fps") or 30.0)
    show_inactive = task.get("show_inactive", False)

    src_video_path = task.get("src_video_path")
    # 若主进程未预先检测 fps（task["fps_detected"] = False），worker 再 fallback 检测一次
    if not task.get("fps_detected", True):
        if src_video_path and os.path.exists(src_video_path):
            detected = detect_fps(src_video_path)
            if detected > 0:
                fps = detected

    hdf5_root = task.get("hdf5_root", "")
    camera_name = task.get("camera_name", "head_camera")

    try:
        # 一次性读完 action + 相机参数，减少对 PFS/NFS 的 open 次数
        # extrinsic 为 (T,3,4)：腕部相机需逐帧更新 view matrix
        (left_arm, left_gripper, right_arm, right_gripper,
         intrinsic, extrinsics) = load_all_from_hdf5(item, hdf5_root, camera_name)
    except Exception as e:
        return {"idx": idx, "status": "skip", "reason": f"load_hdf5 failed: {e}"}

    try:
        _setup_camera_from_params(intrinsic, extrinsics, _worker_img_w, _worker_img_h)
    except Exception as e:
        return {"idx": idx, "status": "skip", "reason": f"setup_camera failed: {e}"}

    T = int(left_arm.shape[0])
    if T == 0:
        return {"idx": idx, "status": "skip", "reason": "0 frames"}

    # RoboTwin gripper_val: 1=开、0=闭 → close_frac HIGH=闭合
    closed_l, settled_l = _gripper_state_masks(1.0 - np.asarray(left_gripper, dtype=np.float64))
    closed_r, settled_r = _gripper_state_masks(1.0 - np.asarray(right_gripper, dtype=np.float64))

    n_ext = int(extrinsics.shape[0]) if getattr(extrinsics, "ndim", 0) == 3 else 1
    use_vectorized = str(task.get("fk_backend", "vectorized")) == "vectorized"

    def _extrinsic_at(t: int) -> np.ndarray:
        if n_ext <= 1:
            return extrinsics if extrinsics.ndim == 2 else extrinsics[0]
        return extrinsics[min(t, n_ext - 1)]

    img_w = _worker_img_w
    img_h = _worker_img_h
    enable_concat = bool(task.get("enable_concat", False))
    enable_overlay = bool(task.get("enable_overlay", False))
    overlay_alpha = float(task.get("overlay_alpha", 0.7))
    skip_action_video = bool(task.get("skip_action_video", False))
    cleanup_action_after_concat = bool(task.get("cleanup_action_after_concat", False))
    concat_height = int(task.get("concat_height", img_h))

    # 原子写：先写 .part，成功后再 os.replace（避免残留半成品被 resume 误判为完成）
    action_tmp = output_path + ".part" if output_path else None
    overlay_tmp = overlay_output_path + ".part" if overlay_output_path else None
    concat_tmp = concat_output_path + ".part" if concat_output_path else None

    bones_rgba = joints_rgba = bones_rgb = joints_rgb = None
    if use_vectorized:
        if enable_overlay:
            bones_rgba, joints_rgba = _precompute_skeleton_draw(
                left_arm, left_gripper, right_arm, right_gripper, extrinsics,
                closed_l, settled_l, closed_r, settled_r, transparent=True)
        if not skip_action_video:
            bones_rgb, joints_rgb = _precompute_skeleton_draw(
                left_arm, left_gripper, right_arm, right_gripper, extrinsics,
                closed_l, settled_l, closed_r, settled_r, transparent=False)

    # ---------- overlay ----------
    overlay_status = "skip"
    overlay_reason = ""
    if enable_overlay and overlay_tmp and src_video_path:
        cap = cv2.VideoCapture(src_video_path)
        if not cap.isOpened():
            overlay_status = "fail"
            overlay_reason = f"cannot open src video: {src_video_path}"
        else:
            if use_vectorized:
                def _overlay_producer(t: int) -> np.ndarray:
                    ret, bgr_frame = cap.read()
                    if not ret:
                        bg_rgb = np.zeros((img_h, img_w, 3), dtype=np.uint8)
                    else:
                        bg_rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                        if bg_rgb.shape[0] != img_h or bg_rgb.shape[1] != img_w:
                            bg_rgb = cv2.resize(bg_rgb, (img_w, img_h),
                                                interpolation=cv2.INTER_LINEAR)
                    skeleton_rgba = np.zeros((img_h, img_w, 4), dtype=np.uint8)
                    _blit_skeleton_frame(skeleton_rgba, bones_rgba, joints_rgba, t)
                    return _overlay_frame(bg_rgb, skeleton_rgba, overlay_alpha)
            else:
                def _overlay_producer(t: int) -> np.ndarray:
                    ret, bgr_frame = cap.read()
                    if not ret:
                        bg_rgb = np.zeros((img_h, img_w, 3), dtype=np.uint8)
                    else:
                        bg_rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                        if bg_rgb.shape[0] != img_h or bg_rgb.shape[1] != img_w:
                            bg_rgb = cv2.resize(bg_rgb, (img_w, img_h),
                                                interpolation=cv2.INTER_LINEAR)
                    skeleton_rgba = _draw_frame(
                        left_arm[t], float(left_gripper[t]),
                        right_arm[t], float(right_gripper[t]),
                        show_inactive, transparent=True,
                        extrinsic_t=_extrinsic_at(t),
                        left_closed=bool(closed_l[t]), left_settled=bool(settled_l[t]),
                        right_closed=bool(closed_r[t]), right_settled=bool(settled_r[t]),
                    )
                    return _overlay_frame(bg_rgb, skeleton_rgba, overlay_alpha)

            try:
                rc, err = _encode_to_ffmpeg(
                    _build_ffmpeg_cmd(overlay_tmp, img_w, img_h, fps),
                    T, _overlay_producer,
                )
                if rc == 0 and os.path.exists(overlay_tmp) and os.path.getsize(overlay_tmp) > 0:
                    os.replace(overlay_tmp, overlay_output_path)
                    overlay_status = "ok"
                else:
                    overlay_status = "fail"
                    overlay_reason = f"ffmpeg rc={rc}: {err}"
                    _safe_unlink(overlay_tmp)
            except Exception as e:
                overlay_status = "fail"
                overlay_reason = str(e)[:200]
                _safe_unlink(overlay_tmp)
            finally:
                cap.release()

    # ---------- action (白色背景) ----------
    concat_status = "skip"
    concat_reason = ""

    if skip_action_video:
        return {
            "idx": idx, "status": "ok", "frames": T,
            "concat_status": concat_status, "concat_reason": concat_reason,
            "overlay_status": overlay_status, "overlay_reason": overlay_reason,
        }

    if use_vectorized:
        def _action_producer(t: int) -> np.ndarray:
            img = np.full((img_h, img_w, 3), 255, dtype=np.uint8)
            return _blit_skeleton_frame(img, bones_rgb, joints_rgb, t)
    else:
        def _action_producer(t: int) -> np.ndarray:
            return _draw_frame(
                left_arm[t], float(left_gripper[t]),
                right_arm[t], float(right_gripper[t]),
                show_inactive, extrinsic_t=_extrinsic_at(t),
                left_closed=bool(closed_l[t]), left_settled=bool(settled_l[t]),
                right_closed=bool(closed_r[t]), right_settled=bool(settled_r[t]),
            )

    try:
        rc, err = _encode_to_ffmpeg(
            _build_ffmpeg_cmd(action_tmp, img_w, img_h, fps),
            T, _action_producer,
        )
        if rc == 0 and os.path.exists(action_tmp) and os.path.getsize(action_tmp) > 0:
            os.replace(action_tmp, output_path)

            if enable_concat and concat_tmp and original_video_path:
                try:
                    concat_cmd = [
                        "ffmpeg", "-y",
                        "-i", original_video_path,
                        "-i", output_path,
                        "-filter_complex",
                        (
                            f"[0:v]fps={fps:g},scale=-2:{concat_height}:flags=lanczos,setsar=1[v0];"
                            f"[1:v]fps={fps:g},scale=-2:{concat_height}:flags=lanczos,setsar=1[v1];"
                            f"[v0][v1]hstack=inputs=2[v]"
                        ),
                        "-map", "[v]", "-an",
                        "-c:v", "libx264",
                        "-preset", _worker_ffmpeg_preset,
                        "-crf", str(_worker_ffmpeg_crf),
                        "-threads", str(_worker_ffmpeg_threads),
                        "-pix_fmt", "yuv420p",
                        "-r", f"{fps:g}", "-shortest",
                        "-movflags", "+faststart",
                        "-f", "mp4",
                        "-loglevel", "error",
                        concat_tmp,
                    ]
                    cproc = subprocess.run(
                        concat_cmd, capture_output=True, text=True, timeout=300,
                    )
                    if (cproc.returncode == 0
                            and os.path.exists(concat_tmp)
                            and os.path.getsize(concat_tmp) > 0):
                        os.replace(concat_tmp, concat_output_path)
                        concat_status = "ok"
                    else:
                        concat_status = "fail"
                        concat_reason = (cproc.stderr or "").strip()[:200]
                        _safe_unlink(concat_tmp)
                except subprocess.TimeoutExpired:
                    concat_status = "fail"
                    concat_reason = "concat ffmpeg timeout"
                    _safe_unlink(concat_tmp)
                except Exception as e:
                    concat_status = "fail"
                    concat_reason = str(e)[:200]
                    _safe_unlink(concat_tmp)

            if cleanup_action_after_concat and concat_status == "ok":
                _safe_unlink(output_path)

            return {
                "idx": idx, "status": "ok", "frames": T,
                "concat_status": concat_status, "concat_reason": concat_reason,
                "overlay_status": overlay_status, "overlay_reason": overlay_reason,
            }
        _safe_unlink(action_tmp)
        return {"idx": idx, "status": "fail",
                "reason": f"ffmpeg rc={rc}: {err}"}
    except Exception as e:
        _safe_unlink(action_tmp)
        return {"idx": idx, "status": "fail", "reason": str(e)[:200]}


# ---------------------------------------------------------------------------
# 导出 pkl.gz
# ---------------------------------------------------------------------------
def _export_pkl(
    all_items: List[dict],
    action_output_dir: str,
    save_mode: str,
    output_dir: str,
    img_w: int,
    img_h: int,
    export_path: str,
) -> None:
    """为每条数据添加 action_video_info 字段后保存为 pkl.gz。

    action_video_info 格式:
        {
            "video": "/absolute/path/to/000000.mp4",
            "frame_nums": <与原始帧数相同>,
            "width": <渲染宽度>,
            "height": <渲染高度>,
        }
    """
    export_items: List[dict] = []
    skipped_missing = 0

    for i, item in enumerate(all_items):
        if save_mode in ("concat_and_action", "overlay_and_action"):
            video_path = os.path.join(action_output_dir, f"{i:06d}.mp4")
        elif save_mode == "action":
            video_path = os.path.join(output_dir, f"{i:06d}.mp4")
        elif save_mode == "overlay":
            video_path = os.path.join(output_dir, f"{i:06d}_overlay.mp4")
        elif save_mode == "concat":
            video_path = os.path.join(output_dir, f"{i:06d}_concat.mp4")
        else:
            video_path = os.path.join(action_output_dir, f"{i:06d}.mp4")

        video_path = os.path.abspath(video_path)

        if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
            skipped_missing += 1
            continue

        new_item = copy.deepcopy(item)
        new_item["action_video_info"] = {
            "video": video_path,
            "frame_nums": item.get("frame_nums", 0),
            "width": img_w,
            "height": img_h,
        }
        export_items.append(new_item)

    os.makedirs(os.path.dirname(export_path) or ".", exist_ok=True)
    with gzip.open(export_path, "wb") as f:
        pickle.dump(export_items, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[Export] 已保存 {len(export_items)} 条数据到 {export_path}"
          f" (跳过 {skipped_missing} 条视频不存在)")
    if export_items:
        sample = export_items[0]
        print(f"[Export] 示例 action_video_info: {sample['action_video_info']}")


# ---------------------------------------------------------------------------
# Resume / 完整性校验工具
# ---------------------------------------------------------------------------
def _prescan_dir(dir_path: str) -> Dict[str, int]:
    """用 os.scandir 批量扫描目录中的 .mp4 文件，返回 {abs_path: size} 字典。

    对大数据集（几万条）比逐条 os.path.exists + getsize 快 10~100 倍。
    """
    result: Dict[str, int] = {}
    if not dir_path or not os.path.isdir(dir_path):
        return result
    try:
        with os.scandir(dir_path) as it:
            for entry in it:
                name = entry.name
                if not name.endswith(".mp4") or name.endswith(".part"):
                    continue
                try:
                    st = entry.stat(follow_symlinks=False)
                    result[os.path.abspath(entry.path)] = st.st_size
                except OSError:
                    pass
    except OSError:
        pass
    return result


def _probe_video_frames(video_path: str, timeout: float = 10.0) -> int:
    """用 ffprobe 读取视频帧数；失败返回 -1。

    优先读 stream.nb_frames（快），不行再 count_packets 兜底（慢但准确）。
    """
    try:
        out = subprocess.check_output(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=nb_frames,avg_frame_rate,duration",
                "-of", "default=nokey=1:noprint_wrappers=1",
                video_path,
            ],
            stderr=subprocess.DEVNULL, timeout=timeout,
        )
        lines = [x.strip() for x in out.decode().splitlines() if x.strip()]
        nb_frames_str = lines[0] if lines else ""
        if nb_frames_str and nb_frames_str != "N/A":
            return int(nb_frames_str)
        # 回退：duration × fps
        duration = float(lines[2]) if len(lines) >= 3 and lines[2] not in ("", "N/A") else 0.0
        rate = lines[1] if len(lines) >= 2 else "0/1"
        if "/" in rate:
            num, den = rate.split("/")
            fps = float(num) / float(den) if float(den) != 0 else 0.0
        else:
            fps = float(rate)
        if duration > 0 and fps > 0:
            return int(round(duration * fps))
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError, OSError):
        pass
    return -1


def _verify_video(video_path: str, expected_frames: int, tol: int = 2) -> bool:
    """验证视频帧数与期望相符。expected_frames<=0 表示跳过帧数检查。"""
    if expected_frames <= 0:
        return True
    nb = _probe_video_frames(video_path)
    if nb < 0:
        return False
    return abs(nb - expected_frames) <= tol


def _parallel_verify(paths_with_expected: List[Tuple[str, int]],
                     workers: int, tol: int,
                     desc: str = "Verify") -> Dict[str, bool]:
    """多线程并行验证视频帧数。返回 {abs_path: is_valid}。"""
    if not paths_with_expected:
        return {}
    result: Dict[str, bool] = {}
    try:
        from tqdm import tqdm
        pbar = tqdm(total=len(paths_with_expected), desc=desc,
                    unit="vid", dynamic_ncols=True)
    except ImportError:
        pbar = None

    with cf.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        fut_map = {
            pool.submit(_verify_video, p, exp, tol): p
            for p, exp in paths_with_expected
        }
        bad = 0
        for fut in cf.as_completed(fut_map):
            path = fut_map[fut]
            try:
                ok = fut.result()
            except Exception:
                ok = False
            result[path] = ok
            if not ok:
                bad += 1
            if pbar is not None:
                pbar.set_postfix(bad=bad)
                pbar.update(1)
        if pbar is not None:
            pbar.close()
    return result


def _prefetch_fps_for_tasks(tasks: List[dict], workers: int) -> int:
    """主进程并行预检测 src 视频的 fps（比每个 worker 单独 ffprobe 快）。

    将 fps 写入 task['fps']，并标记 task['fps_detected']=True。
    返回成功检测的数量。
    """
    # 按 src_video_path 去重（同一视频可能被引用多次——这里不会，但保险起见）
    unique_paths: Dict[str, List[dict]] = {}
    for t in tasks:
        sp = t.get("src_video_path")
        if sp and os.path.exists(sp):
            unique_paths.setdefault(sp, []).append(t)

    if not unique_paths:
        for t in tasks:
            t.setdefault("fps_detected", False)
        return 0

    try:
        from tqdm import tqdm
        pbar = tqdm(total=len(unique_paths), desc="Detect FPS",
                    unit="vid", dynamic_ncols=True)
    except ImportError:
        pbar = None

    ok = 0
    with cf.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        fut_map = {pool.submit(detect_fps, p): p for p in unique_paths}
        for fut in cf.as_completed(fut_map):
            path = fut_map[fut]
            try:
                fps = fut.result()
            except Exception:
                fps = 0.0
            for t in unique_paths[path]:
                if fps > 0:
                    t["fps"] = fps
                    t["fps_detected"] = True
                    ok += 1
                else:
                    t["fps_detected"] = False
            if pbar is not None:
                pbar.update(1)
    if pbar is not None:
        pbar.close()

    for t in tasks:
        t.setdefault("fps_detected", False)
    return ok


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="批量离线渲染 RoboTwin action 火柴棍视频（ARX5 双臂 + 夹爪）"
    )
    parser.add_argument(
        "--jsonl", type=str, default=DEFAULT_JSONL,
        help="JSONL 元数据路径",
    )
    parser.add_argument(
        "--urdf", type=str, default=DEFAULT_URDF,
        help="URDF 文件路径",
    )
    parser.add_argument(
        "--config", type=str, default=DEFAULT_CONFIG,
        help="RoboTwin config.yml 路径（包含 robot_pose / gripper_scale 等）",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="/tmp/robotwin_action_video/",
        help="输出目录",
    )
    parser.add_argument(
        "--action-output-dir", type=str, default="",
        help="纯 action 视频目录，仅 save-mode=concat_and_action / overlay_and_action 时使用",
    )
    parser.add_argument("--workers", type=int, default=16, help="并行 worker 数量")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="兜底 FPS（优先使用原始视频的实际 FPS）")
    parser.add_argument("--width", type=int, default=640, help="输出视频宽度")
    parser.add_argument("--height", type=int, default=480, help="输出视频高度")
    parser.add_argument("--resume", action="store_true",
                        help="跳过已存在且非空的输出文件（默认仅检查 size>0）")
    parser.add_argument("--show-inactive", action="store_true",
                        help="显示非 active 关节（轮子等）")
    parser.add_argument("--start", type=int, default=0, help="起始索引（含）")
    parser.add_argument("--end", type=int, default=1000000, help="结束索引（不含）")
    parser.add_argument(
        "--save-mode", type=str, default="overlay_and_action",
        choices=["action", "concat", "concat_and_action",
                 "overlay", "overlay_and_action"],
        help="输出模式",
    )
    parser.add_argument("--concat-height", type=int, default=480,
                        help="拼接视频统一缩放高度（像素）")
    parser.add_argument("--overlay-alpha", type=float, default=0.7,
                        help="overlay 骨骼叠加不透明度 (0~1)")
    parser.add_argument("--camera", type=str, default="head_camera",
                        help="用于投影的相机名称，默认 head_camera")
    parser.add_argument("--view-camera", type=str, default=None,
                        help="用于原始视频的相机名称（默认与 --camera 相同）")
    parser.add_argument("--robot-type", type=str, default="aloha-agilex",
                        help="只渲染指定 robot_type 的样本（空字符串表示不过滤）")
    parser.add_argument(
        "--hdf5-root", type=str, default="",
        help="HDF5 根目录（覆盖 JSONL 中的 root 字段用于解析 hdf5_path），为空则使用 item['root']",
    )
    parser.add_argument(
        "--root-remap", type=str, default="",
        help="root 路径前缀替换，格式 OLD=NEW（修复挂载点变更），空字符串禁用",
    )
    parser.add_argument(
        "--export-pkl", type=str, default="",
        help="导出 pkl.gz 路径；为每条数据添加 action_video_info 字段后保存",
    )

    parser.add_argument(
        "--fk-backend", type=str, default="vectorized",
        choices=["vectorized", "yourdfpy"],
        help="FK 后端：vectorized=整段 episode 批量化 FK（默认，更快）；"
             "yourdfpy=逐帧 update_cfg（对照用）",
    )
    parser.add_argument("--verify-resume", action="store_true",
                        help="开启后用 ffprobe 校验已渲染视频的帧数是否匹配 frame_nums；"
                             "不匹配则重渲染。对 resume 时已有视频有用。")
    parser.add_argument("--verify-tol", type=int, default=2,
                        help="帧数容差（±N 帧视为 OK），默认 2")
    parser.add_argument("--verify-workers", type=int, default=32,
                        help="并行校验线程数（ffprobe I/O 为主），默认 32")
    parser.add_argument("--no-prescan", action="store_true",
                        help="禁用目录预扫描（默认启用，大幅加速 resume 检查）")

    # ----- 渲染速度选项 -----
    parser.add_argument("--ffmpeg-preset", type=str, default="veryfast",
                        choices=["ultrafast", "superfast", "veryfast", "faster",
                                 "fast", "medium", "slow", "slower", "veryslow"],
                        help="ffmpeg libx264 preset（默认 veryfast，速度/体积平衡）")
    parser.add_argument("--ffmpeg-threads", type=int, default=1,
                        help="每个 ffmpeg 子进程线程数（默认 1，避免抢 worker 核）")
    parser.add_argument("--ffmpeg-crf", type=int, default=23,
                        help="ffmpeg libx264 crf，默认 23；越小质量越高文件越大")
    parser.add_argument("--maxtasks-per-child", type=int, default=200,
                        help="每个 worker 处理 N 个任务后重启，防止内存泄漏。0 表示不限制。")
    parser.add_argument("--chunksize", type=int, default=0,
                        help="imap_unordered chunksize；0=自动（1 最均衡，大 chunk 调度抖动大）")
    parser.add_argument("--prefetch-fps", action="store_true", default=True,
                        help="主进程并行预检测 src 视频 fps（默认开启）")
    parser.add_argument("--no-prefetch-fps", dest="prefetch_fps",
                        action="store_false", help="关闭 fps 预检测")

    args = parser.parse_args(argv)

    if args.save_mode in ("concat_and_action", "overlay_and_action"):
        if not args.action_output_dir:
            parser.error(f"save-mode={args.save_mode} 时必须指定 --action-output-dir")

    if args.width % 2 != 0:
        args.width += 1
    if args.height % 2 != 0:
        args.height += 1

    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_mode in ("concat_and_action", "overlay_and_action") and args.action_output_dir:
        os.makedirs(args.action_output_dir, exist_ok=True)

    save_action = args.save_mode in ("action", "concat", "concat_and_action",
                                     "overlay_and_action")
    save_concat = args.save_mode in ("concat", "concat_and_action")
    save_overlay = args.save_mode in ("overlay", "overlay_and_action")

    view_camera = args.view_camera or args.camera

    # ====== 0. 加载 robot config ======
    print(f"[Main] 加载 config: {args.config}")
    robot_cfg = _load_robot_config(args.config)
    robot_pose_4x4 = _build_robot_pose_matrix(robot_cfg)
    gripper_scale_raw = robot_cfg.get("gripper_scale", list(DEFAULT_GRIPPER_SCALE))
    gripper_scale = (float(gripper_scale_raw[0]), float(gripper_scale_raw[1]))
    print(f"[Main] robot_pose (translation): {robot_pose_4x4[:3, 3].tolist()}")
    print(f"[Main] gripper_scale: {gripper_scale}")

    # ====== 1. 加载元数据 ======
    print(f"[Main] 加载 JSONL: {args.jsonl}")
    all_items = load_jsonl(args.jsonl)
    total_all = len(all_items)
    print(f"[Main] 数据集共 {total_all} 条")

    # ------ root 路径重映射（修复 /media/unify → /media/datasets 等挂载点变更）------
    if args.root_remap:
        parts = args.root_remap.split("=", 1)
        if len(parts) == 2:
            old_prefix, new_prefix = parts
            remapped = 0
            for it in all_items:
                root = it.get("root", "")
                if root.startswith(old_prefix):
                    it["root"] = new_prefix + root[len(old_prefix):]
                    remapped += 1
            if remapped:
                print(f"[Main] root 重映射: '{old_prefix}' → '{new_prefix}' ({remapped}/{total_all} 条)")

    # ------ 1a. 按 robot_type 过滤 ------
    if args.robot_type:
        before = len(all_items)
        all_items = [it for it in all_items if it.get("robot_type") == args.robot_type]
        print(f"[Main] robot_type='{args.robot_type}' 过滤: {before} → {len(all_items)}")

    # ------ 1b. 按 camera 过滤（只保留有指定 camera 视频的样本） ------
    cam_name = args.camera
    before_cam = len(all_items)
    all_items = [
        it for it in all_items
        if cam_name in it.get("camera_video_info", {})
        and it["camera_video_info"][cam_name].get("video")
    ]
    if before_cam != len(all_items):
        print(f"[Main] camera='{cam_name}' 过滤: {before_cam} → {len(all_items)}")

    start_idx = max(0, min(args.start, len(all_items)))
    end_idx = max(start_idx, min(args.end, len(all_items)))
    data = all_items[start_idx:end_idx]

    # ====== 1c. 预扫描输出目录（加速 resume 判断） ======
    prescan_action: Dict[str, int] = {}
    prescan_overlay: Dict[str, int] = {}
    prescan_concat: Dict[str, int] = {}
    if args.resume and not args.no_prescan:
        t_scan = time.time()
        if args.save_mode in ("concat_and_action", "overlay_and_action"):
            prescan_action = _prescan_dir(args.action_output_dir)
        elif args.save_mode == "action":
            prescan_action = _prescan_dir(args.output_dir)
        if save_overlay:
            prescan_overlay = _prescan_dir(args.output_dir)
        if save_concat:
            prescan_concat = _prescan_dir(args.output_dir)
        total_scanned = (len(prescan_action) + len(prescan_overlay)
                         + len(prescan_concat))
        if total_scanned:
            print(f"[Main] 预扫描完成: {total_scanned} 个已有视频, "
                  f"耗时 {time.time() - t_scan:.1f}s")

    def _existing_ok(path: str, bucket: Dict[str, int]) -> bool:
        """判断 path 是否已存在且非空（优先用预扫描缓存，miss 时回退 stat）。"""
        if not path:
            return False
        abs_path = os.path.abspath(path)
        if bucket is not None:
            sz = bucket.get(abs_path)
            if sz is not None:
                return sz > 0
        # 未预扫描或 cache miss，回退到 stat
        try:
            return os.path.getsize(path) > 0
        except OSError:
            return False

    # ====== 2. 构建任务列表（第一轮：仅看文件存在/大小，不做 ffprobe） ======
    tasks: List[dict] = []
    skipped_exist = 0
    task_counter: Dict[str, int] = {}
    # 用于 verify-resume 的候选：(path, expected_frames, global_idx) 列表
    verify_candidates: List[Tuple[str, int, int]] = []
    # global_idx → {item, paths, ...}，用于校验失败后重新入队
    pending_verify_tasks: Dict[int, dict] = {}

    for i, item in enumerate(data):
        global_idx = start_idx + i
        concat_out_path = os.path.join(args.output_dir, f"{global_idx:06d}_concat.mp4")
        overlay_out_path = os.path.join(args.output_dir, f"{global_idx:06d}_overlay.mp4")

        if args.save_mode in ("concat_and_action", "overlay_and_action"):
            out_path = os.path.join(args.action_output_dir, f"{global_idx:06d}.mp4")
        elif args.save_mode == "overlay":
            out_path = os.path.join(args.output_dir, f"{global_idx:06d}_action_tmp.mp4")
        else:
            out_path = os.path.join(args.output_dir, f"{global_idx:06d}.mp4")

        try:
            cvi = item.get("camera_video_info", {})
            view_info = cvi.get(view_camera, {})
            video_rel = view_info.get("video", "")
            video_root = item.get("root", "")
            src_video = _resolve_video_path(video_root, video_rel) if video_rel else None
            if src_video and not os.path.exists(src_video) and args.hdf5_root:
                src_video = _resolve_video_path(args.hdf5_root, video_rel)
        except Exception:
            src_video = None

        expected_frames = int(item.get("frame_nums", 0) or 0)

        # ---- resume 快速检查（仅基于 size） ----
        if args.resume:
            if args.save_mode == "overlay_and_action":
                action_ok = _existing_ok(out_path, prescan_action)
                overlay_ok = _existing_ok(overlay_out_path, prescan_overlay)
                resume_hit = action_ok and overlay_ok
            elif args.save_mode == "concat_and_action":
                action_ok = _existing_ok(out_path, prescan_action)
                concat_ok = _existing_ok(concat_out_path, prescan_concat)
                resume_hit = action_ok and concat_ok
            elif args.save_mode == "overlay":
                resume_hit = _existing_ok(overlay_out_path, prescan_overlay)
            elif args.save_mode == "concat":
                resume_hit = _existing_ok(concat_out_path, prescan_concat)
            else:
                resume_hit = _existing_ok(out_path, prescan_action)

            if resume_hit:
                skipped_exist += 1
                # 若开启 verify-resume，登记该样本的所有需要校验的视频
                if args.verify_resume:
                    paths_to_verify: List[str] = []
                    if args.save_mode in ("overlay_and_action", "concat_and_action", "action"):
                        paths_to_verify.append(out_path)
                    if args.save_mode in ("overlay", "overlay_and_action"):
                        paths_to_verify.append(overlay_out_path)
                    if args.save_mode in ("concat", "concat_and_action"):
                        paths_to_verify.append(concat_out_path)
                    for p in paths_to_verify:
                        verify_candidates.append((p, expected_frames, global_idx))
                    pending_verify_tasks[global_idx] = {
                        "idx": global_idx,
                        "item": item,
                        "out_path": out_path,
                        "overlay_out_path": overlay_out_path,
                        "concat_out_path": concat_out_path,
                        "src_video": src_video,
                    }
                continue

        task_name = item.get("task_name", "unknown")
        task_counter[task_name] = task_counter.get(task_name, 0) + 1

        task_dict: dict = {
            "idx": global_idx,
            "item": item,
            "output_path": out_path,
            "fps": args.fps,
            "show_inactive": args.show_inactive,
            "src_video_path": src_video,
            "enable_concat": save_concat,
            "concat_output_path": concat_out_path if save_concat else None,
            "original_video_path": src_video if save_concat else None,
            "enable_overlay": save_overlay,
            "overlay_output_path": overlay_out_path if save_overlay else None,
            "overlay_alpha": args.overlay_alpha,
            "skip_action_video": args.save_mode == "overlay",
            "cleanup_action_after_concat": args.save_mode == "concat",
            "concat_height": args.concat_height,
            "hdf5_root": args.hdf5_root,
            "camera_name": args.camera,
            "fk_backend": args.fk_backend,
            # fps 预检测（稍后在主进程批量填充）
            "fps_detected": False,
        }
        tasks.append(task_dict)

    # ====== 2b. 可选：ffprobe 校验已 resume 的视频 ======
    if args.resume and args.verify_resume and verify_candidates:
        print(f"[Main] 启动帧数校验: {len(verify_candidates)} 个视频 "
              f"(tol=±{args.verify_tol}, workers={args.verify_workers})")
        verify_result = _parallel_verify(
            [(p, exp) for p, exp, _ in verify_candidates],
            workers=args.verify_workers,
            tol=args.verify_tol,
            desc="Verify",
        )
        # 任一产物校验失败，则整条样本都要重渲染
        bad_global_indices: set = set()
        for path, _, gidx in verify_candidates:
            if not verify_result.get(path, False):
                bad_global_indices.add(gidx)
        if bad_global_indices:
            print(f"[Main] 校验失败 {len(bad_global_indices)} 条，将重新渲染")
            for gidx in bad_global_indices:
                info = pending_verify_tasks[gidx]
                for p in (info["out_path"], info["overlay_out_path"],
                          info["concat_out_path"]):
                    _safe_unlink(p)
                item = info["item"]
                task_name = item.get("task_name", "unknown")
                task_counter[task_name] = task_counter.get(task_name, 0) + 1
                tasks.append({
                    "idx": gidx,
                    "item": item,
                    "output_path": info["out_path"],
                    "fps": args.fps,
                    "show_inactive": args.show_inactive,
                    "src_video_path": info["src_video"],
                    "enable_concat": save_concat,
                    "concat_output_path": info["concat_out_path"] if save_concat else None,
                    "original_video_path": info["src_video"] if save_concat else None,
                    "enable_overlay": save_overlay,
                    "overlay_output_path": info["overlay_out_path"] if save_overlay else None,
                    "overlay_alpha": args.overlay_alpha,
                    "skip_action_video": args.save_mode == "overlay",
                    "cleanup_action_after_concat": args.save_mode == "concat",
                    "concat_height": args.concat_height,
                    "hdf5_root": args.hdf5_root,
                    "camera_name": args.camera,
                    "fk_backend": args.fk_backend,
                    "fps_detected": False,
                })
            skipped_exist -= len(bad_global_indices)
        else:
            print(f"[Main] 校验通过，全部 {len(verify_candidates)} 个视频帧数 OK")

    print(f"[Main] 渲染范围: [{start_idx}, {end_idx}), "
          f"待渲染: {len(tasks)}, 已跳过(resume): {skipped_exist}")
    print(f"[Main] 保存模式: {args.save_mode} "
          f"(save_action={save_action}, save_concat={save_concat}, save_overlay={save_overlay})")
    print(f"[Main] URDF: {args.urdf}")
    print(f"[Main] 输出分辨率: {args.width}x{args.height}, 投影相机: {args.camera}")
    if args.hdf5_root:
        print(f"[Main] HDF5 根目录覆盖: {args.hdf5_root}")
    if save_overlay:
        print(f"[Main] Overlay alpha: {args.overlay_alpha}")
    if args.save_mode in ("concat_and_action", "overlay_and_action"):
        print(f"[Main] 双目录: {args.save_mode.split('_and_')[0]} -> {args.output_dir}")
        print(f"[Main]         action -> {args.action_output_dir}")
    if task_counter:
        top_tasks = sorted(task_counter.items(), key=lambda x: -x[1])[:10]
        print(f"[Main] 任务分布 (top-10): " +
              ", ".join(f"{tn}={tc}" for tn, tc in top_tasks))
    print(f"[Main] ffmpeg preset={args.ffmpeg_preset}, "
          f"threads={args.ffmpeg_threads}, crf={args.ffmpeg_crf}")
    print(f"[Main] FK backend: {args.fk_backend}")

    if not tasks:
        print("[Main] 无需渲染的任务")
        if args.export_pkl:
            _export_pkl(
                all_items=all_items,
                action_output_dir=(args.action_output_dir if args.action_output_dir
                                   else args.output_dir),
                save_mode=args.save_mode,
                output_dir=args.output_dir,
                img_w=args.width, img_h=args.height,
                export_path=args.export_pkl,
            )
        return 0

    # ====== 2c. 预检测 src 视频 fps（主进程并行，避免每个 worker 再 fork ffprobe） ======
    if args.prefetch_fps and tasks:
        t_fps = time.time()
        ok_fps = _prefetch_fps_for_tasks(tasks, workers=min(args.verify_workers, 64))
        print(f"[Main] FPS 预检测: {ok_fps}/{len(tasks)} 个，"
              f"耗时 {time.time() - t_fps:.1f}s")
        if ok_fps == 0:
            print(f"[Main] ⚠️  没有任何样本的源视频可访问！以下是前 3 条样本的关键路径，"
                  f"请检查 --root-remap / --hdf5-root 是否正确：")
            for t in tasks[:3]:
                item = t["item"]
                action_info = item.get("action", {})
                hdf5_rel = action_info.get("hdf5_path") or (
                    action_info.get("action", {}).get("hdf5_path") if isinstance(action_info.get("action"), dict) else ""
                )
                print(f"  idx={t['idx']}: root={item.get('root')!r}")
                print(f"           hdf5_rel={hdf5_rel!r}")
                print(f"           src_video={t.get('src_video_path')!r}"
                      f" exists={os.path.exists(t.get('src_video_path') or '')}")

    # ====== 2d. 前置数据路径体检（只检查前 5 条，避免 worker 全军覆没才发现） ======
    if tasks:
        probe_n = min(5, len(tasks))
        probe_bad = 0
        probe_reasons: List[str] = []
        for t in tasks[:probe_n]:
            try:
                _ = _resolve_hdf5_path(t["item"], t.get("hdf5_root", ""))
            except Exception as e:
                probe_bad += 1
                if len(probe_reasons) < 3:
                    probe_reasons.append(f"idx={t['idx']}: {e}")
        if probe_bad == probe_n:
            print(f"[Main] ❌ 前 {probe_n} 条样本的 HDF5 路径全部解析失败！"
                  f"基本可确定 root/remap/hdf5_root 配置有误。示例:")
            for r in probe_reasons:
                print(f"  - {r}")
            print(f"[Main] 常见原因：默认 --root-remap '/media/unify=/media/datasets' "
                  f"会把 root 前缀替换，如果数据实际仍在原路径，请改用 --root-remap \"\" 禁用。")
        elif probe_bad > 0:
            print(f"[Main] ⚠️  前 {probe_n} 条样本中 {probe_bad} 条 HDF5 路径解析失败")

    # ====== 3. 多进程渲染 ======
    num_workers = min(args.workers, len(tasks))

    # 资源使用建议（每条任务约占 1 核 Python draw + 0.1 核 ffmpeg encode ≈ 1.1 核）
    try:
        cpu_count = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpu_count = os.cpu_count() or 1
    est_usage = num_workers * 1.1
    if est_usage > cpu_count * 1.3:
        print(f"[Main] ⚠️  worker 数过高：预估占用 {est_usage:.0f} 核，"
              f"但系统只有 {cpu_count} 核。可能 oversubscription 影响效率。"
              f"建议 --workers {int(cpu_count * 0.9)}")
    elif est_usage < cpu_count * 0.7:
        print(f"[Main] 💡 算力未充分利用：预估占用 {est_usage:.0f}/{cpu_count} 核。"
              f"可尝试 --workers {int(cpu_count * 0.9)} 提升吞吐")

    print(f"[Main] 启动 {num_workers} 个 worker "
          f"(maxtasksperchild={args.maxtasks_per_child or 'unlimited'}, "
          f"chunksize={args.chunksize or 'auto=1'}) ...")

    t0 = time.time()
    ok_count = 0
    fail_count = 0
    skip_count = 0
    concat_ok = 0
    concat_fail = 0
    overlay_ok = 0
    overlay_fail = 0
    fail_samples: List[str] = []
    skip_samples: List[str] = []
    skip_reason_counter: Dict[str, int] = {}

    # chunksize 策略：
    #   - 每条任务耗时 1~3 秒（相对于 IPC 开销 ms 级），chunksize=1 最均衡
    #   - chunksize 大时末期调度不均（剩余 worker 各抱一堆）
    #   - 用户可通过 --chunksize 覆盖；0 = 自动(1)
    if args.chunksize > 0:
        chunksize = args.chunksize
    else:
        chunksize = 1

    pool_kwargs = dict(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(args.urdf, args.width, args.height,
                  robot_pose_4x4, gripper_scale,
                  args.ffmpeg_preset, args.ffmpeg_threads, args.ffmpeg_crf),
    )
    if args.maxtasks_per_child and args.maxtasks_per_child > 0:
        pool_kwargs["maxtasksperchild"] = args.maxtasks_per_child

    # Linux 下默认 fork：COW 内存共享，启动瞬时；URDF 由每个 worker 独立加载
    with mp.Pool(**pool_kwargs) as pool:
        try:
            from tqdm import tqdm
            pbar = tqdm(total=len(tasks), desc="Rendering", unit="vid",
                        dynamic_ncols=True)
        except ImportError:
            pbar = None

        try:
            for result in pool.imap_unordered(render_one, tasks, chunksize=chunksize):
                status = result.get("status", "?")
                if status == "ok":
                    ok_count += 1
                    if result.get("concat_status") == "ok":
                        concat_ok += 1
                    elif result.get("concat_status") == "fail":
                        concat_fail += 1
                        if len(fail_samples) < 10:
                            fail_samples.append(
                                f"CONCAT idx={result['idx']}: "
                                f"{result.get('concat_reason', '?')}"
                            )
                    if result.get("overlay_status") == "ok":
                        overlay_ok += 1
                    elif result.get("overlay_status") == "fail":
                        overlay_fail += 1
                        if len(fail_samples) < 10:
                            fail_samples.append(
                                f"OVERLAY idx={result['idx']}: "
                                f"{result.get('overlay_reason', '?')}"
                            )
                elif status == "skip":
                    skip_count += 1
                    reason = result.get("reason", "?")
                    # 抓取 skip 原因的第一个短句作为分类键（如 "load_action failed"）
                    short_key = reason.split(":")[0].strip()[:60] or "?"
                    skip_reason_counter[short_key] = skip_reason_counter.get(short_key, 0) + 1
                    if len(skip_samples) < 10:
                        skip_samples.append(f"idx={result['idx']}: {reason}")
                else:
                    fail_count += 1
                    if len(fail_samples) < 10:
                        fail_samples.append(
                            f"idx={result['idx']}: {result.get('reason', '?')}"
                        )

                if pbar is not None:
                    pbar.set_postfix(ok=ok_count, skip=skip_count, fail=fail_count,
                                     ov_ok=overlay_ok if save_overlay else 0,
                                     cc_ok=concat_ok if save_concat else 0)
                    pbar.update(1)
        except KeyboardInterrupt:
            print("\n[Main] Ctrl-C 收到，正在终止 worker ...")
            pool.terminate()
            pool.join()
            if pbar is not None:
                pbar.close()
            return 130

        if pbar is not None:
            pbar.close()

    elapsed = time.time() - t0
    per = elapsed / max(1, ok_count)
    print(f"\n[Main] 渲染完成: ok={ok_count}, skip={skip_count}, "
          f"fail={fail_count}, 耗时={elapsed:.1f}s ({per:.2f}s/vid)")
    if save_concat:
        print(f"[Main] 拼接视频: ok={concat_ok}, fail={concat_fail}")
    if save_overlay:
        print(f"[Main] 叠加视频: ok={overlay_ok}, fail={overlay_fail}")
    if fail_samples:
        print(f"[Main] 失败样本示例 (前 {len(fail_samples)} 条):")
        for msg in fail_samples:
            print(f"  - {msg}")
    if skip_count > 0:
        print(f"[Main] Skip 原因分布 (top-5):")
        for key, cnt in sorted(skip_reason_counter.items(),
                               key=lambda x: -x[1])[:5]:
            print(f"  - {cnt:6d}  {key}")
        if skip_samples:
            print(f"[Main] Skip 样本示例 (前 {len(skip_samples)} 条):")
            for msg in skip_samples:
                print(f"  - {msg}")
        # 如果全部 skip，很可能是路径问题
        if skip_count == len(tasks):
            print(f"[Main] ❌ 全部 {skip_count} 条任务都被 skip！"
                  f"通常是 root/remap/hdf5_root 配置错误导致 HDF5 打不开。")

    # ====== 4. 导出 pkl.gz（可选） ======
    if args.export_pkl:
        _export_pkl(
            all_items=all_items,
            action_output_dir=(args.action_output_dir if args.action_output_dir
                               else args.output_dir),
            save_mode=args.save_mode,
            output_dir=args.output_dir,
            img_w=args.width, img_h=args.height,
            export_path=args.export_pkl,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
