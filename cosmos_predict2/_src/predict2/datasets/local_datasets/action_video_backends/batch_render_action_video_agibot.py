#!/usr/bin/env python3
"""Batch render AgiBot-World `agibot_gripper` action videos (crt_120s 夹爪).

Uses the official G1 body/arms with the crt_120s end-effector (5-link fingers
per branch). The older G1_omnipicker stick-gripper renderer has been removed;
this script is the sole AgiBot gripper action/overlay renderer.

Input format is LeRobot v2.1 per-task:

    /datasets/AgiBot-World_v2.1_per_task_fixed/
      agibot_gripper__task_XXX/
        meta/info.json
        meta/episodes.jsonl
        data/episode_000000.parquet
        videos/observation.images.head_color/episode_000000.mp4

This regenerated version uses the official recovered *mean* head camera
intrinsics directly:

    fx=322.2804617342562, fy=429.14943849195873
    cx=323.4804281649203, cy=242.44413873455974

The G1_120s URDF does not contain a camera link, so the camera is
still modeled as a virtual camera mounted on `head_link2`. The mount translation
uses the mean of `-head_extrinsic.translation_vector` from recovered parameters;
the optical rotation keeps the previous URDF convention.
"""
from __future__ import annotations

# Pin math libraries to a single thread per process BEFORE importing numpy.
# Each render worker only does tiny 4x4 matrix ops, so BLAS/OpenMP threading is
# pure overhead: with many workers it oversubscribes the cores (load average
# explodes, per-core CPU stays low). One thread per worker lets the process pool
# do the real parallelism. Users can still override via the environment.
import os as _os

for _thread_env in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    _os.environ.setdefault(_thread_env, "1")

import argparse
import csv
import fnmatch
import gzip
import json
import multiprocessing as mp
import os
import pickle
import random
import shutil
import fcntl
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from visualize_action_on_urdf import _as_transform, _link_name

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_URDF = str(
    SCRIPT_DIR
    / "agibot"
    / "official_geniesimassets"
    / "robot"
    / "curobo_robot"
    / "assets"
    / "robot"
    / "G1"
    / "G1_120s.urdf"
)
DEFAULT_EPISODE_CAMERA_MAPPING = str(
    SCRIPT_DIR / "agibot_auto_calib" / "agibot_world_episode_camera_mapping_beta_parameters.csv"
)
DEFAULT_TASK_CAMERA_EXTRINSIC_OVERRIDES = str(
    SCRIPT_DIR / "agibot_auto_calib" / "task_camera_extrinsic_overrides.json"
)

IMG_W = 640
IMG_H = 480
MEAN_FX = 322.2804617342562
MEAN_FY = 429.14943849195873
MEAN_CX = 323.4804281649203
MEAN_CY = 242.44413873455974

DEFAULT_CAMERA_OFFSET_XYZ = np.array(
    [0.08930634849338406, -0.06338408536635548, -0.008881486514663101],
    dtype=np.float64,
)
DEFAULT_CAMERA_OFFSET_RPY_DEG = np.array([90.0, 0.0, 0.0], dtype=np.float64)

LEFT_ARM_JOINTS = [f"idx2{i}_arm_l_joint{i}" for i in range(1, 8)]
RIGHT_ARM_JOINTS = [f"idx6{i}_arm_r_joint{i}" for i in range(1, 8)]
WAIST_LIFT_JOINT = "idx01_body_joint1"
WAIST_PITCH_JOINT = "idx02_body_joint2"
HEAD_YAW_JOINT = "idx11_head_joint1"
HEAD_PITCH_JOINT = "idx12_head_joint2"
CAMERA_LINK = "head_link2"

# Fixed shoulder mounts on arm_base_link (URDF origin.xyz.Y, meters).
ARM_L_BASE_JOINT = "idx11_arm_l_base_joint"
ARM_R_BASE_JOINT = "idx12_arm_r_base_joint"
DEFAULT_ARM_L_BASE_Y = 0.025
DEFAULT_ARM_R_BASE_Y = -0.025

# crt_120s drive joint (idx41/idx81) limit is [0, 1.0] rad.
GRIPPER_120S_MAX_Q = 1.0
# AgiBot effector.position in mm: ~35=closed, ~120=fully open (state stream).
GRIPPER_MM_MIN = 35.0
GRIPPER_MM_SPAN = 85.0
# If the open/close sense looks flipped, set --gripper-invert (or this default).
DEFAULT_GRIPPER_INVERT = False
# crt_120s has true finger geometry in the URDF, so (unlike the omnipicker stick
# gripper) no forward push is needed; calibrated to 0 against the real videos.
DEFAULT_GRIPPER_FORWARD_OFFSET = 0.0
DEFAULT_GRIPPER_DELAY_FRAMES = 42
GRIPPER_CLOSED_THRESHOLD = 0.25
GRIPPER_OPEN_THRESHOLD = 0.75
# NOTE on `open_frac` semantics for this gripper: state.effector 35mm maps to
# open_frac=0 but is physically FULLY OPEN (fingertip gap ~106mm); 120mm maps to
# open_frac=1 but is physically FULLY CLOSED (gap ~2mm). So a HIGH open_frac means
# the gripper is closed/gripping.
#
# Coloring uses a per-side temporal state machine (see `_gripper_state_masks`):
# once the value settles onto a plateau it records that width and marks CLOSED
# (orange) / OPEN (blue). Aligned with RealSource: settled frames use binary
# open/closed colors; unsettled (motion) bone lines are solid gray (no blue↔orange
# lerp). Joint dots: settled open=red / closed=green; motion frames stay red.
# Stability is judged over a bidirectional window: past LOOKBACK + current +
# future LOOKAHEAD frames. Short unsettled gaps (<= SETTLED_FILL frames) are
# filled to avoid flicker. When open_frac is already high (>= EARLY_CLOSED_FRAC)
# and the forward window alone is stable, mark settled CLOSED early without
# waiting for lookback to clear the ramp.
GRIPPER_CLOSED_OPEN_FRAC = 0.5      # absolute threshold to classify the FIRST plateau
GRIPPER_STATE_LOOKAHEAD = 10        # future frames used for plateau stability
GRIPPER_STATE_LOOKBACK = 10         # past frames used for plateau stability
GRIPPER_STATE_PLATEAU_EPS = 0.03    # max open-fraction spread within a plateau window
GRIPPER_STATE_CHANGE_DELTA = 0.1    # min plateau-to-plateau change to flip state (> eps)
GRIPPER_STATE_SETTLED_FILL = 3      # fill unsettled gaps shorter than this (frames)
GRIPPER_EARLY_CLOSED_FRAC = 0.85    # early CLOSED if ov>=this and forward window stable

# Match RealSource / RoboTwin visual style (BGR for OpenCV/ffmpeg).
COLOR_ACTIVE_BONE = (255, 120, 0)             # visual RGB: (0, 120, 255), blue
COLOR_GRIPPER_OPEN_BONE = (255, 120, 0)       # visual RGB: (0, 120, 255), blue
COLOR_GRIPPER_CLOSED_BONE = (0, 140, 255)     # visual RGB: (255, 140, 0), orange
COLOR_ACTIVE_JOINT = (50, 50, 255)            # visual RGB: (255, 50, 50), red
COLOR_GRIPPER_OPEN_JOINT = (50, 50, 255)      # visual RGB: (255, 50, 50), red
COLOR_GRIPPER_CLOSED_JOINT = (0, 200, 0)      # visual RGB: (0, 200, 0), green
COLOR_GRIPPER_MOTION_BONE = (128, 128, 128)   # unsettled bone: gray (RealSource)
COLOR_GRIPPER_MOTION_JOINT = (50, 50, 255)    # unsettled joint: red
COLOR_INACTIVE = (192, 192, 192)
ACTIVE_BONE_WIDTH = 6
INACTIVE_BONE_WIDTH = 1
ACTIVE_JOINT_RADIUS = 9
INACTIVE_JOINT_RADIUS = 3

# crt_120s gripper joints. The drive joint (outer1) carries the commanded angle;
# inner1/outer4/inner4/outer2/inner2 are mimics of the drive joint in the URDF
# (multipliers below), and outer3/inner3 are the free coupling joints closed by
# the spherical loop_joints (kept at 0 for skeleton drawing).
GRIPPER_CONTROL_JOINTS = {
    "left": {
        "outer1": "idx41_gripper_l_outer_joint1",
        "inner1": "idx31_gripper_l_inner_joint1",
        "outer4": "idx43_gripper_l_outer_joint4",
        "inner4": "idx33_gripper_l_inner_joint4",
        "outer2": "idx49_gripper_l_outer_joint2",
        "inner2": "idx39_gripper_l_inner_joint2",
        "outer3": "idx42_gripper_l_outer_joint3",
        "inner3": "idx32_gripper_l_inner_joint3",
    },
    "right": {
        "outer1": "idx81_gripper_r_outer_joint1",
        "inner1": "idx71_gripper_r_inner_joint1",
        "outer4": "idx83_gripper_r_outer_joint4",
        "inner4": "idx73_gripper_r_inner_joint4",
        "outer2": "idx89_gripper_r_outer_joint2",
        "inner2": "idx79_gripper_r_inner_joint2",
        "outer3": "idx82_gripper_r_outer_joint3",
        "inner3": "idx72_gripper_r_inner_joint3",
    },
}
# Multiplier of the drive-joint angle for each crt_120s mimic joint (matches the
# <mimic> tags in G1_120s.urdf). Free coupling joints stay at 0.
GRIPPER_MIMIC_MULTIPLIERS = {
    "inner1": 1.0,
    "outer4": 1.0,
    "inner4": 1.0,
    "outer2": 1.0,
    "inner2": -1.0,
    "outer3": 0.0,
    "inner3": 0.0,
}

ACTIVE_JOINTS = frozenset(
    LEFT_ARM_JOINTS
    + RIGHT_ARM_JOINTS
    + [WAIST_PITCH_JOINT, WAIST_LIFT_JOINT, HEAD_YAW_JOINT, HEAD_PITCH_JOINT]
)
ACTIVE_GRIPPER_JOINTS = frozenset(
    joint_name for side in GRIPPER_CONTROL_JOINTS.values() for joint_name in side.values()
)

GRIPPER_LINK_CHAINS = {
    "left": [
        ("arm_l_link7", "gripper_l_base_link"),
        ("gripper_l_base_link", "gripper_l_outer_link1"),
        ("gripper_l_outer_link1", "gripper_l_outer_link3"),
        ("gripper_l_outer_link3", "gripper_l_outer_link4"),
        ("gripper_l_outer_link4", "gripper_l_outer_link5"),
        ("gripper_l_base_link", "gripper_l_outer_link2"),
        ("gripper_l_base_link", "gripper_l_inner_link1"),
        ("gripper_l_inner_link1", "gripper_l_inner_link3"),
        ("gripper_l_inner_link3", "gripper_l_inner_link4"),
        ("gripper_l_inner_link4", "gripper_l_inner_link5"),
        ("gripper_l_base_link", "gripper_l_inner_link2"),
    ],
    "right": [
        ("arm_r_link7", "gripper_r_base_link"),
        ("gripper_r_base_link", "gripper_r_outer_link1"),
        ("gripper_r_outer_link1", "gripper_r_outer_link3"),
        ("gripper_r_outer_link3", "gripper_r_outer_link4"),
        ("gripper_r_outer_link4", "gripper_r_outer_link5"),
        ("gripper_r_base_link", "gripper_r_outer_link2"),
        ("gripper_r_base_link", "gripper_r_inner_link1"),
        ("gripper_r_inner_link1", "gripper_r_inner_link3"),
        ("gripper_r_inner_link3", "gripper_r_inner_link4"),
        ("gripper_r_inner_link4", "gripper_r_inner_link5"),
        ("gripper_r_base_link", "gripper_r_inner_link2"),
    ],
}
GRIPPER_BASE_LINKS = {
    "left": ("arm_l_link7", "gripper_l_base_link"),
    "right": ("arm_r_link7", "gripper_r_base_link"),
}

_AXIS = {
    "+x": np.array([1.0, 0.0, 0.0]),
    "-x": np.array([-1.0, 0.0, 0.0]),
    "+y": np.array([0.0, 1.0, 0.0]),
    "-y": np.array([0.0, -1.0, 0.0]),
    "+z": np.array([0.0, 0.0, 1.0]),
    "-z": np.array([0.0, 0.0, -1.0]),
}
CAM_AXIS_MODES = {
    "x_fwd": ("-y", "-z", "+x"),
    "z_fwd": ("+x", "+y", "+z"),
    "x_fwd_yup": ("+y", "-z", "+x"),
    "y_fwd": ("+x", "-z", "+y"),
    "neg_y_fwd": ("-x", "-z", "-y"),
    "neg_z_fwd": ("+x", "-y", "-z"),
}

# The jsonl `extrinsic.rotation_matrix` uses an optical frame with x-right but
# y-up / z-back (OpenGL-style), whereas the pinhole projection in `_project`
# expects OpenCV optical axes (x-right, y-down, z-forward). This fixed flip about
# the optical x-axis converts the data convention to OpenCV so points land in
# front of the camera. (Verified: it maps the extrinsic optical frame onto the
# previously-calibrated `x_fwd` orientation.)
_DATA_TO_CV_OPTICAL = np.diag([1.0, -1.0, -1.0]).astype(np.float64)

_worker_robot = None
_worker_fx = MEAN_FX
_worker_fy = MEAN_FY
_worker_cx = MEAN_CX
_worker_cy = MEAN_CY
_worker_img_w = IMG_W
_worker_img_h = IMG_H
_worker_camera_offset_xyz = DEFAULT_CAMERA_OFFSET_XYZ.copy()
_worker_camera_offset_rpy = np.deg2rad(DEFAULT_CAMERA_OFFSET_RPY_DEG)
_worker_ffmpeg_preset = "veryfast"
_worker_ffmpeg_crf = 23
_worker_ffmpeg_threads = 1
_worker_gripper_max_q = GRIPPER_120S_MAX_Q
_worker_gripper_invert = DEFAULT_GRIPPER_INVERT
_worker_incoming_joint_by_link: Dict[str, str] = {}
_worker_active_arm_bones: List[Tuple[str, str]] = []
_worker_white_frame: Optional[np.ndarray] = None
_worker_black_frame: Optional[np.ndarray] = None
# Static kinematic spec extracted once per worker for the vectorized FK backend.
_worker_kin: Optional[Dict[str, Any]] = None


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    r, p, y = [float(v) for v in rpy]
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def _axis_vector(name: str) -> np.ndarray:
    return _AXIS[name]


def _optical_remap(cam_t: np.ndarray, mode: str) -> np.ndarray:
    axes = CAM_AXIS_MODES[mode]
    r = cam_t[:3, :3]
    out = cam_t.copy()
    out[:3, 0] = r @ _axis_vector(axes[0])
    out[:3, 1] = r @ _axis_vector(axes[1])
    out[:3, 2] = r @ _axis_vector(axes[2])
    return out


def _rot_axis(axis: str, angle_deg: float) -> np.ndarray:
    a = np.deg2rad(float(angle_deg))
    c, s = np.cos(a), np.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def _apply_pitch_yaw_offset(t_opt: np.ndarray, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    out = t_opt.copy()
    out[:3, :3] = out[:3, :3] @ _rot_axis("y", pitch_deg) @ _rot_axis("x", yaw_deg)
    return out


def _set_arm_base_y(
    left_y: float,
    right_y: float,
    robot=None,
    kin: Optional[Dict[str, Any]] = None,
) -> None:
    """Set left/right arm_base joint origin Y in the arm_base_link frame (meters).

    Mutates ``joint.origin[1, 3]`` (URDF ``T = Trans(xyz) @ R(rpy)`` so
    ``T[:3, 3] == xyz``), syncs the yourdfpy scene graph for fixed joints, and
    updates the vectorized FK kinematics cache when provided.
    """
    targets = (
        (ARM_L_BASE_JOINT, float(left_y)),
        (ARM_R_BASE_JOINT, float(right_y)),
    )
    if robot is not None:
        for jname, y in targets:
            j = robot.joint_map.get(jname)
            if j is None:
                continue
            origin = _as_transform(getattr(j, "origin", None)).copy()
            origin[1, 3] = y
            j.origin = origin
            # Fixed joints are not re-applied by update_cfg; push into the graph.
            matrix, _ = robot._forward_kinematics_joint(j, q=0.0)
            if getattr(robot, "_scene", None) is not None:
                robot._scene.graph.update(
                    frame_from=j.parent, frame_to=j.child, matrix=matrix
                )
            if getattr(robot, "_scene_collision", None) is not None:
                robot._scene_collision.graph.update(
                    frame_from=j.parent, frame_to=j.child, matrix=matrix
                )
    use_kin = kin
    if use_kin is None:
        use_kin = _worker_kin
    if use_kin is not None:
        joints = use_kin.get("joints") or {}
        for jname, y in targets:
            info = joints.get(jname)
            if info is None:
                continue
            origin = np.asarray(info["origin"], dtype=np.float64).copy()
            origin[1, 3] = y
            info["origin"] = origin


def _camera_link_transform(
    robot,
    camera_offset_xyz: Optional[Sequence[float]] = None,
    camera_offset_rpy_deg: Optional[Sequence[float]] = None,
) -> np.ndarray:
    base_t = _as_transform(robot.get_transform(CAMERA_LINK))
    offset = np.eye(4, dtype=np.float64)
    if camera_offset_xyz is None:
        xyz = _worker_camera_offset_xyz
    else:
        xyz = np.asarray(camera_offset_xyz, dtype=np.float64)
    if camera_offset_rpy_deg is None:
        rpy = _worker_camera_offset_rpy
    else:
        rpy = np.deg2rad(np.asarray(camera_offset_rpy_deg, dtype=np.float64))
    offset[:3, :3] = _rpy_matrix(rpy)
    offset[:3, 3] = xyz
    return base_t @ offset


def _project(
    view: np.ndarray,
    pt_world: np.ndarray,
    fx: Optional[float] = None,
    fy: Optional[float] = None,
    cx: Optional[float] = None,
    cy: Optional[float] = None,
) -> Optional[Tuple[int, int]]:
    p = view @ np.array([pt_world[0], pt_world[1], pt_world[2], 1.0])
    z = p[2]
    if z < 0.01:
        return None
    use_fx = _worker_fx if fx is None else float(fx)
    use_fy = _worker_fy if fy is None else float(fy)
    use_cx = _worker_cx if cx is None else float(cx)
    use_cy = _worker_cy if cy is None else float(cy)
    u = int(round(use_fx * p[0] / z + use_cx))
    v = int(round(use_fy * p[1] / z + use_cy))
    return u, v


def _build_worker_static_draw_cache(robot) -> None:
    global _worker_incoming_joint_by_link, _worker_active_arm_bones
    incoming: Dict[str, str] = {}
    active_bones: List[Tuple[str, str]] = []
    for jn, j in robot.joint_map.items():
        child = getattr(j, "child", None)
        if child is not None:
            incoming[str(child)] = str(jn)
        jname = str(jn)
        if jname not in ACTIVE_JOINTS:
            continue
        if str(getattr(j, "joint_type", getattr(j, "type", ""))) == "fixed":
            continue
        parent = _link_name(getattr(j, "parent", None))
        if parent is None:
            continue
        active_bones.append((jname, parent))
    _worker_incoming_joint_by_link = incoming
    _worker_active_arm_bones = active_bones


def _build_worker_kinematics(robot) -> None:
    """Extract a static kinematic spec for the vectorized FK backend.

    The kinematic tree is fixed; only joint angles change per frame. We cache
    each joint's origin/axis/type/parent/child/mimic plus a topological link
    order so forward kinematics can be evaluated as batched numpy matmuls over
    the whole episode instead of per-frame yourdfpy `get_transform` calls.
    """
    global _worker_kin
    from collections import deque

    joints: Dict[str, Dict[str, Any]] = {}
    children_by_parent: Dict[str, List[Tuple[str, str]]] = {}
    link_parent_joint: Dict[str, str] = {}
    all_child_links: set = set()
    link_set: set = set()

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
        if parent is not None:
            link_set.add(parent)
        if child is not None:
            link_set.add(child)
            all_child_links.add(child)
            link_parent_joint[child] = name
            children_by_parent.setdefault(parent, []).append((name, child))

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

    mimic_joints = [name for name, info in joints.items() if info["mimic"] is not None]
    _worker_kin = {
        "joints": joints,
        "roots": roots,
        "topo": topo,
        "link_parent_joint": link_parent_joint,
        "mimic_joints": mimic_joints,
    }


def _axis_rotation_batch(axis: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Batched Rodrigues rotation (T,3,3) about a unit axis for angles q (T,)."""
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


def _frame_cfg_batch(
    arm: np.ndarray,
    head: np.ndarray,
    waist: np.ndarray,
    gripper: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Vectorized version of `_frame_cfg`: joint_name -> (T,) angle array."""
    cfg: Dict[str, np.ndarray] = {}
    for i, jn in enumerate(LEFT_ARM_JOINTS):
        cfg[jn] = arm[:, i].astype(np.float64)
    for i, jn in enumerate(RIGHT_ARM_JOINTS):
        cfg[jn] = arm[:, 7 + i].astype(np.float64)
    cfg[WAIST_PITCH_JOINT] = waist[:, 0].astype(np.float64)
    cfg[WAIST_LIFT_JOINT] = np.clip(waist[:, 1], 0.0, 0.55)
    cfg[HEAD_YAW_JOINT] = head[:, 0].astype(np.float64)
    cfg[HEAD_PITCH_JOINT] = np.clip(head[:, 1], -0.349, 0.5236)
    open_frac = _gripper_open_fraction_array(gripper)
    drive_frac = _gripper_drive_fraction_array(open_frac)
    for sidx, side in enumerate(("left", "right")):
        q = _worker_gripper_max_q * drive_frac[:, sidx]
        joints = GRIPPER_CONTROL_JOINTS[side]
        cfg[joints["outer1"]] = q
        for key, mult in GRIPPER_MIMIC_MULTIPLIERS.items():
            cfg[joints[key]] = mult * q
    return cfg


def _fk_world_transforms_batch(cfg_arrays: Dict[str, np.ndarray], t: int) -> Dict[str, np.ndarray]:
    """Compute world transforms (T,4,4) for every link via batched FK."""
    kin = _worker_kin
    assert kin is not None
    # Apply mimic joints to match yourdfpy semantics: value = mult*src + offset.
    for jname in kin["mimic_joints"]:
        src, mult, off = kin["joints"][jname]["mimic"]
        if src in cfg_arrays:
            cfg_arrays[jname] = mult * cfg_arrays[src] + off
    eye_t = np.broadcast_to(np.eye(4, dtype=np.float64), (t, 4, 4))
    world: Dict[str, np.ndarray] = {}
    for r in kin["roots"]:
        world[r] = eye_t
    for link in kin["topo"]:
        if link in world:
            continue
        jname = kin["link_parent_joint"].get(link)
        if jname is None:
            world[link] = eye_t
            continue
        info = kin["joints"][jname]
        q = cfg_arrays.get(jname)
        if q is None:
            q = np.zeros(t, dtype=np.float64)
        local = _joint_local_batch(info, q, t)
        world[link] = world[info["parent"]] @ local
    return world


def _project_batch(
    view: np.ndarray,
    pts: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized projection of point trajectory (T,3) -> (u, v, valid) arrays.

    Mirrors `_project`: z < 0.01 is invalid, else round(f*p/z + c).
    """
    t = pts.shape[0]
    ph = np.concatenate([pts, np.ones((t, 1), dtype=np.float64)], axis=1)
    p = np.einsum("tij,tj->ti", view, ph)
    z = p[:, 2]
    valid = z >= 0.01
    zz = np.where(valid, z, 1.0)
    u = np.round(fx * p[:, 0] / zz + cx).astype(np.int64)
    v = np.round(fy * p[:, 1] / zz + cy).astype(np.int64)
    return u, v, valid


def _gripper_gradient_color_batch(
    open_vals: np.ndarray,
    color_open: tuple,
    color_closed: tuple,
) -> np.ndarray:
    """Vectorized `_gripper_gradient_color`: (T,) open values -> (T,3) int BGR."""
    co = np.asarray(color_open, dtype=np.float64)
    cc = np.asarray(color_closed, dtype=np.float64)
    denom = GRIPPER_OPEN_THRESHOLD - GRIPPER_CLOSED_THRESHOLD
    frac = np.clip((open_vals - GRIPPER_CLOSED_THRESHOLD) / denom, 0.0, 1.0)
    out = cc[None, :] + (co[None, :] - cc[None, :]) * frac[:, None]
    out = np.round(out)
    out[open_vals >= GRIPPER_OPEN_THRESHOLD] = co
    out[open_vals <= GRIPPER_CLOSED_THRESHOLD] = cc
    return out.astype(np.int64)


def _gripper_close_params_from_task(
    task: Dict[str, Any],
) -> Tuple[float, int, int, float, float, int, float]:
    return (
        float(task.get("gripper_closed_open_frac", GRIPPER_CLOSED_OPEN_FRAC)),
        int(task.get("gripper_state_lookahead", GRIPPER_STATE_LOOKAHEAD)),
        int(task.get("gripper_state_lookback", GRIPPER_STATE_LOOKBACK)),
        float(task.get("gripper_state_plateau_eps", GRIPPER_STATE_PLATEAU_EPS)),
        float(task.get("gripper_state_change_delta", GRIPPER_STATE_CHANGE_DELTA)),
        int(task.get("gripper_state_settled_fill", GRIPPER_STATE_SETTLED_FILL)),
        float(task.get("gripper_early_closed_frac", GRIPPER_EARLY_CLOSED_FRAC)),
    )


def _fill_short_false_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    """Set False runs shorter than/equal to `max_gap` to True (anti-flicker)."""
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
    open_vals_side: np.ndarray,
    closed_open_frac: float,
    lookahead: int = GRIPPER_STATE_LOOKAHEAD,
    eps: float = GRIPPER_STATE_PLATEAU_EPS,
    change_delta: float = GRIPPER_STATE_CHANGE_DELTA,
    lookback: int = GRIPPER_STATE_LOOKBACK,
    settled_fill: int = GRIPPER_STATE_SETTLED_FILL,
    early_closed_frac: float = GRIPPER_EARLY_CLOSED_FRAC,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-side temporal state machine: CLOSED mask + settled (plateau) mask.

    `open_vals_side` is the state open fraction where HIGH = physically closed.

    When the value settles onto a plateau (stable over past `lookback` + current +
    next `lookahead` frames) its width is recorded as the reference. A new plateau
    that differs from the reference by more than `change_delta` flips the state
    (higher -> CLOSED, lower -> OPEN). Motion frames keep the last confirmed state
    in `closed_mask` (hysteresis) but are marked unsettled so bone colors use gray.
    Short unsettled gaps (<= `settled_fill`) are filled to suppress flicker.

    Early CLOSED: if `open_frac >= early_closed_frac` and the forward window
    (current + lookahead) alone is stable, mark settled without waiting for
    lookback to clear the closing ramp, so orange/green confirm sooner.
    """
    ov = np.clip(np.asarray(open_vals_side, dtype=np.float64), 0.0, 1.0)
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

        # Early CLOSED: ignore lookback once already tightly closed and future is flat.
        if (not settled[t]) and ov[t] >= early_closed_frac:
            fwd = ov[t:hi]
            if (float(fwd.max()) - float(fwd.min())) < eps:
                settled[t] = True
                plateau_val[t] = float(np.median(fwd))

    settled = _fill_short_false_gaps(settled, settled_fill)

    settled_idx = np.flatnonzero(settled)
    if settled_idx.size == 0:
        # No stable plateau anywhere: fall back to absolute threshold, all binary.
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


def _gripper_closed_mask(
    open_vals_side: np.ndarray,
    closed_open_frac: float,
    lookahead: int = GRIPPER_STATE_LOOKAHEAD,
    eps: float = GRIPPER_STATE_PLATEAU_EPS,
    change_delta: float = GRIPPER_STATE_CHANGE_DELTA,
    lookback: int = GRIPPER_STATE_LOOKBACK,
    settled_fill: int = GRIPPER_STATE_SETTLED_FILL,
    early_closed_frac: float = GRIPPER_EARLY_CLOSED_FRAC,
) -> np.ndarray:
    """Backward-compatible wrapper: CLOSED mask only from `_gripper_state_masks`."""
    closed, _settled = _gripper_state_masks(
        open_vals_side, closed_open_frac, lookahead, eps, change_delta,
        lookback=lookback, settled_fill=settled_fill,
        early_closed_frac=early_closed_frac,
    )
    return closed


def _gripper_bone_colors_from_mask(
    closed_mask: np.ndarray,
    color_open: tuple,
    color_closed: tuple,
) -> np.ndarray:
    """Map per-frame closed mask to (T,3) BGR bone colors."""
    co = np.asarray(color_open, dtype=np.int64)
    cc = np.asarray(color_closed, dtype=np.int64)
    return np.where(closed_mask[:, None], cc[None, :], co[None, :])


def _gripper_colors_state_or_gradient(
    open_vals: np.ndarray,
    closed_mask: np.ndarray,
    settled_mask: np.ndarray,
    color_open: tuple,
    color_closed: tuple,
    color_motion: tuple = COLOR_GRIPPER_MOTION_BONE,
) -> np.ndarray:
    """Settled frames: binary state color. Motion frames: solid gray (RealSource).

    open_vals is kept for call-site compatibility; motion color no longer depends
    on its value (no blue↔orange lerp during open/close).
    """
    del open_vals  # unused: motion uses solid gray
    binary = _gripper_bone_colors_from_mask(closed_mask, color_open, color_closed)
    settled = np.asarray(settled_mask, dtype=bool)
    motion = np.asarray(color_motion, dtype=np.int64)
    return np.where(settled[:, None], binary, motion[None, :]).astype(np.int64)


def _gripper_joint_colors_state_or_motion(
    closed_mask: np.ndarray,
    settled_mask: np.ndarray,
    color_open: tuple = COLOR_GRIPPER_OPEN_JOINT,
    color_closed: tuple = COLOR_GRIPPER_CLOSED_JOINT,
    color_motion: tuple = COLOR_GRIPPER_MOTION_JOINT,
) -> np.ndarray:
    """Joint dots: settled binary (open=red / closed=green); motion = fixed red."""
    binary = _gripper_bone_colors_from_mask(closed_mask, color_open, color_closed)
    settled = np.asarray(settled_mask, dtype=bool)
    motion = np.asarray(color_motion, dtype=np.int64)
    return np.where(settled[:, None], binary, motion[None, :]).astype(np.int64)


def _gripper_color_state_or_gradient(
    open_val: float,
    closed: bool,
    settled: bool,
    color_open: tuple,
    color_closed: tuple,
    color_motion: tuple = COLOR_GRIPPER_MOTION_BONE,
) -> tuple:
    """Scalar hybrid color for the per-frame `_draw_frame` path."""
    del open_val  # unused: motion uses solid gray
    if settled:
        return color_closed if closed else color_open
    return tuple(int(x) for x in color_motion)

def _init_worker(
    urdf_path: str,
    img_w: int,
    img_h: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    camera_offset_xyz: Sequence[float],
    camera_offset_rpy_deg: Sequence[float],
    ffmpeg_preset: str,
    ffmpeg_crf: int,
    ffmpeg_threads: int,
    gripper_max_q: float = GRIPPER_120S_MAX_Q,
    gripper_invert: bool = DEFAULT_GRIPPER_INVERT,
    arm_l_base_y: float = DEFAULT_ARM_L_BASE_Y,
    arm_r_base_y: float = DEFAULT_ARM_R_BASE_Y,
) -> None:
    global _worker_robot, _worker_fx, _worker_fy, _worker_cx, _worker_cy
    global _worker_img_w, _worker_img_h, _worker_camera_offset_xyz, _worker_camera_offset_rpy
    global _worker_ffmpeg_preset, _worker_ffmpeg_crf, _worker_ffmpeg_threads
    global _worker_white_frame, _worker_black_frame
    global _worker_gripper_max_q, _worker_gripper_invert
    import logging
    import warnings
    from yourdfpy import URDF

    warnings.filterwarnings("ignore")
    logging.getLogger("yourdfpy").setLevel(logging.ERROR)
    _worker_robot = URDF.load(urdf_path, load_meshes=False, build_scene_graph=True)
    _worker_img_w = int(img_w)
    _worker_img_h = int(img_h)
    _worker_fx = float(fx)
    _worker_fy = float(fy)
    _worker_cx = float(cx)
    _worker_cy = float(cy)
    _worker_camera_offset_xyz = np.asarray(camera_offset_xyz, dtype=np.float64)
    _worker_camera_offset_rpy = np.deg2rad(np.asarray(camera_offset_rpy_deg, dtype=np.float64))
    _worker_ffmpeg_preset = str(ffmpeg_preset)
    _worker_ffmpeg_crf = int(ffmpeg_crf)
    _worker_ffmpeg_threads = max(1, int(ffmpeg_threads))
    _worker_gripper_max_q = float(gripper_max_q)
    _worker_gripper_invert = bool(gripper_invert)
    _worker_white_frame = np.full((_worker_img_h, _worker_img_w, 3), 255, dtype=np.uint8)
    _worker_black_frame = np.zeros((_worker_img_h, _worker_img_w, 3), dtype=np.uint8)
    # OpenCV defaults to a thread pool sized to the host cores. With many worker
    # processes this oversubscribes the CPU; one drawing thread per worker is
    # faster overall here.
    try:
        cv2.setNumThreads(0)
    except Exception:
        pass
    _build_worker_static_draw_cache(_worker_robot)
    _build_worker_kinematics(_worker_robot)
    _set_arm_base_y(
        float(arm_l_base_y),
        float(arm_r_base_y),
        robot=_worker_robot,
        kin=_worker_kin,
    )
    print(
        f"[Worker {os.getpid()}] URDF={os.path.basename(urdf_path)}, "
        f"fx={_worker_fx:.2f}, fy={_worker_fy:.2f}, cx={_worker_cx:.2f}, cy={_worker_cy:.2f}, "
        f"cam_xyz={np.round(_worker_camera_offset_xyz, 4)}, "
        f"cam_rpy_deg={np.round(np.rad2deg(_worker_camera_offset_rpy), 2)}, "
        f"arm_base_y=({float(arm_l_base_y):.4f},{float(arm_r_base_y):.4f})",
        flush=True,
    )


def _frame_cfg(arm: np.ndarray, head: np.ndarray, waist: np.ndarray, gripper: np.ndarray) -> Dict[str, float]:
    cfg: Dict[str, float] = {}
    for i, jn in enumerate(LEFT_ARM_JOINTS):
        cfg[jn] = float(arm[i])
    for i, jn in enumerate(RIGHT_ARM_JOINTS):
        cfg[jn] = float(arm[7 + i])
    cfg[WAIST_PITCH_JOINT] = float(waist[0])
    cfg[WAIST_LIFT_JOINT] = float(np.clip(waist[1], 0.0, 0.55))
    cfg[HEAD_YAW_JOINT] = float(head[0])
    cfg[HEAD_PITCH_JOINT] = float(np.clip(head[1], -0.349, 0.5236))
    open_frac = _gripper_open_fraction_array(np.asarray(gripper, dtype=np.float64))
    drive_frac = _gripper_drive_fraction_array(open_frac)
    for sidx, side in enumerate(("left", "right")):
        q = _worker_gripper_max_q * float(drive_frac[sidx])
        joints = GRIPPER_CONTROL_JOINTS[side]
        cfg[joints["outer1"]] = float(q)
        for key, mult in GRIPPER_MIMIC_MULTIPLIERS.items():
            cfg[joints[key]] = float(mult * q)
    return cfg


def _gripper_open_fraction_array(gripper: np.ndarray) -> np.ndarray:
    """Map effector.position to open fraction in [0, 1] where 1=open, 0=closed."""
    g = np.asarray(gripper, dtype=np.float64)
    if g.size == 0:
        return g
    if float(np.nanmax(g)) > 1.5:
        return np.clip((g - GRIPPER_MM_MIN) / GRIPPER_MM_SPAN, 0.0, 1.0)
    return np.clip(g, 0.0, 1.0)


def _gripper_drive_fraction_array(open_frac: np.ndarray) -> np.ndarray:
    """Map open fraction to crt_120s drive-joint fraction (0=open, 1=closed by default)."""
    open_frac = np.clip(np.asarray(open_frac, dtype=np.float64), 0.0, 1.0)
    closed_frac = 1.0 - open_frac
    return open_frac if _worker_gripper_invert else closed_frac


def _resolve_render_gripper(
    arrs: Dict[str, Dict[str, np.ndarray]],
    n: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (gripper_kin, open_vals) from state.effector for FK and color mask."""
    state_gripper = arrs["state"]["gripper"][:n]
    open_vals = _gripper_open_fraction_array(state_gripper)
    return state_gripper, open_vals


def _gripper_visual_value(value: float) -> float:
    """Return visual open value where 1=open and 0=closed."""
    return float(_gripper_open_fraction_array(np.asarray([value], dtype=np.float64))[0])


def _gripper_gradient_color(gripper_open_value: float, color_open: tuple, color_closed: tuple) -> tuple:
    """RoboTwin-style three-state gripper color.

    open_value >= GRIPPER_OPEN_THRESHOLD   -> open color
    open_value <= GRIPPER_CLOSED_THRESHOLD -> closed color
    otherwise                             -> linear gradient while closing
    """
    if gripper_open_value >= GRIPPER_OPEN_THRESHOLD:
        return color_open
    if gripper_open_value <= GRIPPER_CLOSED_THRESHOLD:
        return color_closed
    t = (gripper_open_value - GRIPPER_CLOSED_THRESHOLD) / (
        GRIPPER_OPEN_THRESHOLD - GRIPPER_CLOSED_THRESHOLD
    )
    return tuple(int(round(a + (b - a) * t)) for a, b in zip(color_closed, color_open))


def load_action_arrays_agibot(
    parquet_path: str,
    need_state: bool = True,
) -> Dict[str, Dict[str, np.ndarray]]:
    import pandas as pd

    prefixes = ("action", "state") if need_state else ("action",)
    cols: List[str] = []
    for prefix in prefixes:
        cols += [f"{prefix}.joint.position.{i}" for i in range(14)]
        cols += [f"{prefix}.effector.position.{i}" for i in range(2)]
        cols += [f"{prefix}.head.position.{i}" for i in range(2)]
        cols += [f"{prefix}.waist.position.{i}" for i in range(2)]
    df = pd.read_parquet(parquet_path, columns=cols)

    def block(prefix: str) -> Dict[str, np.ndarray]:
        gripper = df[[f"{prefix}.effector.position.{i}" for i in range(2)]].to_numpy(dtype=np.float64)
        if prefix == "state":
            gripper = _gripper_open_fraction_array(gripper)
        return {
            "arm": df[[f"{prefix}.joint.position.{i}" for i in range(14)]].to_numpy(dtype=np.float64),
            "gripper": gripper,
            "head": df[[f"{prefix}.head.position.{i}" for i in range(2)]].to_numpy(dtype=np.float64),
            "waist": df[[f"{prefix}.waist.position.{i}" for i in range(2)]].to_numpy(dtype=np.float64),
        }

    out = {"action": block("action")}
    if need_state:
        out["state"] = block("state")
    return out


def _draw_frame(
    arm: np.ndarray,
    gripper: np.ndarray,
    head: np.ndarray,
    waist: np.ndarray,
    cam_axis_mode: str,
    cam_pitch_offset: float,
    cam_yaw_offset: float,
    gripper_forward_offset: float,
    camera_params: Optional[Dict[str, Any]] = None,
    background: str = "white",
    left_gripper_bone: tuple = COLOR_GRIPPER_OPEN_BONE,
    right_gripper_bone: tuple = COLOR_GRIPPER_OPEN_BONE,
    left_gripper_joint: tuple = COLOR_GRIPPER_OPEN_JOINT,
    right_gripper_joint: tuple = COLOR_GRIPPER_OPEN_JOINT,
    left_cam_yaw_offset: Optional[float] = None,
    right_cam_yaw_offset: Optional[float] = None,
    left_cam_pitch_offset: Optional[float] = None,
    right_cam_pitch_offset: Optional[float] = None,
) -> np.ndarray:
    robot = _worker_robot
    robot.update_cfg(_frame_cfg(arm, head, waist, gripper))
    camera_params = camera_params or {}
    fx = camera_params.get("fx")
    fy = camera_params.get("fy")
    cx = camera_params.get("cx")
    cy = camera_params.get("cy")
    extrinsic_matrix = camera_params.get("extrinsic_matrix")
    view_matrix = camera_params.get("view_matrix")
    def make_view(pitch_offset: float, yaw_offset: float) -> np.ndarray:
        if view_matrix is not None:
            return np.asarray(view_matrix, dtype=np.float64)
        if extrinsic_matrix is not None:
            base_t = _as_transform(robot.get_transform(CAMERA_LINK))
            m = np.asarray(extrinsic_matrix, dtype=np.float64)
            cam_t = base_t @ np.linalg.inv(m)
            cam_t[:3, :3] = cam_t[:3, :3] @ _DATA_TO_CV_OPTICAL
            return np.linalg.inv(_apply_pitch_yaw_offset(cam_t, pitch_offset, yaw_offset))
        cam_t = _camera_link_transform(
            robot,
            camera_params.get("camera_offset_xyz"),
            camera_params.get("camera_offset_rpy_deg"),
        )
        t_opt = _optical_remap(cam_t, cam_axis_mode)
        return np.linalg.inv(_apply_pitch_yaw_offset(t_opt, pitch_offset, yaw_offset))

    if view_matrix is not None:
        # Caller supplies a precomputed world->optical view matrix directly
        # (e.g. beta aligned per-frame pose rigidly mapped into the FK frame).
        view = np.asarray(view_matrix, dtype=np.float64)
    elif extrinsic_matrix is not None:
        # jsonl source: use the real per-episode extrinsic. M maps the camera
        # link (head_link2) frame to the camera optical frame, so the optical
        # pose in world is base_t @ inv(M). No _optical_remap is needed because
        # the data rotation already encodes the optical convention.
        base_t = _as_transform(robot.get_transform(CAMERA_LINK))
        m = np.asarray(extrinsic_matrix, dtype=np.float64)
        cam_t = base_t @ np.linalg.inv(m)
        cam_t[:3, :3] = cam_t[:3, :3] @ _DATA_TO_CV_OPTICAL
        cam_t = _apply_pitch_yaw_offset(cam_t, cam_pitch_offset, cam_yaw_offset)
        view = np.linalg.inv(cam_t)
    else:
        cam_t = _camera_link_transform(
            robot,
            camera_params.get("camera_offset_xyz"),
            camera_params.get("camera_offset_rpy_deg"),
        )
        t_opt = _optical_remap(cam_t, cam_axis_mode)
        t_opt = _apply_pitch_yaw_offset(t_opt, cam_pitch_offset, cam_yaw_offset)
        view = np.linalg.inv(t_opt)
    left_view = make_view(
        cam_pitch_offset
        if left_cam_pitch_offset is None
        else float(left_cam_pitch_offset),
        cam_yaw_offset if left_cam_yaw_offset is None else float(left_cam_yaw_offset)
    )
    right_view = make_view(
        cam_pitch_offset
        if right_cam_pitch_offset is None
        else float(right_cam_pitch_offset),
        cam_yaw_offset if right_cam_yaw_offset is None else float(right_cam_yaw_offset)
    )
    if background == "black":
        img = _worker_black_frame.copy()
    else:
        img = _worker_white_frame.copy()

    joint_points: Dict[str, np.ndarray] = {}
    for jname, _parent in _worker_active_arm_bones:
        j = robot.joint_map[jname]
        parent_name = _link_name(getattr(j, "parent", None))
        if parent_name is None:
            continue
        parent_t = _as_transform(robot.get_transform(parent_name))
        j_t = parent_t @ _as_transform(getattr(j, "origin", None))
        joint_points[jname] = j_t[:3, 3]

    def link_anchor(link_name: str) -> Optional[np.ndarray]:
        inc = _worker_incoming_joint_by_link.get(link_name)
        if inc is not None and inc in joint_points:
            return joint_points[inc]
        try:
            return _as_transform(robot.get_transform(link_name))[:3, 3]
        except Exception:
            return None

    for jname, parent in _worker_active_arm_bones:
        if jname not in joint_points:
            continue
        p1 = link_anchor(parent)
        p2 = joint_points[jname]
        if p1 is None:
            continue
        side_view = left_view if "arm_l_" in jname else right_view
        q1 = _project(side_view, p1, fx, fy, cx, cy)
        q2 = _project(side_view, p2, fx, fy, cx, cy)
        if q1 is not None and q2 is not None:
            cv2.line(img, q1, q2, COLOR_ACTIVE_BONE, ACTIVE_BONE_WIDTH, cv2.LINE_AA)
            cv2.circle(img, q2, ACTIVE_JOINT_RADIUS, COLOR_ACTIVE_JOINT, -1, cv2.LINE_AA)

    for side, chains in GRIPPER_LINK_CHAINS.items():
        side_view = left_view if side == "left" else right_view
        end_link, base_link = GRIPPER_BASE_LINKS[side]
        try:
            base_t = _as_transform(robot.get_transform(base_link))
        except Exception:
            continue
        forward = base_t[:3, 2]
        n = np.linalg.norm(forward)
        shift = (gripper_forward_offset * forward / n) if n > 1e-9 else np.zeros(3)
        bone_color = left_gripper_bone if side == "left" else right_gripper_bone
        joint_color = left_gripper_joint if side == "left" else right_gripper_joint
        for parent, child in chains:
            try:
                p1 = _as_transform(robot.get_transform(parent))[:3, 3].copy()
                p2 = _as_transform(robot.get_transform(child))[:3, 3].copy()
            except Exception:
                continue
            if parent != end_link:
                p1 += shift
            p2 += shift
            q1 = _project(side_view, p1, fx, fy, cx, cy)
            q2 = _project(side_view, p2, fx, fy, cx, cy)
            if q1 is not None and q2 is not None:
                cv2.line(img, q1, q2, bone_color, ACTIVE_BONE_WIDTH, cv2.LINE_AA)
                cv2.circle(img, q2, ACTIVE_JOINT_RADIUS, joint_color, -1, cv2.LINE_AA)
    return img


def _enlarge_pipe(fd: int, want: int) -> None:
    """Grow a pipe's kernel buffer so a whole raw frame fits in one write.

    The default 64KB pipe forces a 640x480x3 (~900KB) frame to be handed to
    ffmpeg in ~14 chunks, each a blocking sleep/wake round-trip. That lock-step
    starves ffmpeg and leaves CPUs idle. A buffer >= one frame lets the writer
    stay a frame ahead and keeps the encoder busy.
    """
    try:
        f_setpipe_sz = getattr(fcntl, "F_SETPIPE_SZ", 1031)
        try:
            cap = int(Path("/proc/sys/fs/pipe-max-size").read_text().strip())
        except Exception:
            cap = want
        fcntl.fcntl(fd, f_setpipe_sz, min(want, cap))
    except Exception:
        pass


def _open_ffmpeg_writer(path: str, fps: float, width: int, height: int) -> subprocess.Popen:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(float(fps)),
            "-i",
            "-",
            "-an",
            "-vcodec",
            "libx264",
            "-preset",
            _worker_ffmpeg_preset,
            "-crf",
            str(_worker_ffmpeg_crf),
            "-threads",
            str(_worker_ffmpeg_threads),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            path,
        ],
        stdin=subprocess.PIPE,
    )
    if proc.stdin is not None:
        _enlarge_pipe(proc.stdin.fileno(), width * height * 3 + 4096)
    return proc


def _write_video(path: str, frames: Sequence[np.ndarray], fps: float) -> None:
    tmp = str(Path(path).with_suffix(".tmp.mp4"))
    proc = _open_ffmpeg_writer(tmp, fps, _worker_img_w, _worker_img_h)
    assert proc.stdin is not None
    try:
        for frame in frames:
            proc.stdin.write(frame.tobytes())
    finally:
        proc.stdin.close()
        ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"ffmpeg failed with code {ret}: {tmp}")
    Path(tmp).replace(path)


def _write_video_iter(path: str, frames, fps: float) -> int:
    tmp = str(Path(path).with_suffix(".tmp.mp4"))
    proc = _open_ffmpeg_writer(tmp, fps, _worker_img_w, _worker_img_h)
    assert proc.stdin is not None
    count = 0
    try:
        for frame in frames:
            proc.stdin.write(frame.tobytes())
            count += 1
    finally:
        proc.stdin.close()
        ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"ffmpeg failed with code {ret}: {tmp}")
    Path(tmp).replace(path)
    return count


def _overlay_skeleton_opaque(bg_bgr: np.ndarray, skeleton_bgr: np.ndarray, overlay_alpha: float = 1.0) -> np.ndarray:
    """Overlay skeleton pixels only, matching the non-transparent action style.

    The skeleton frame uses a black background in overlay mode. Non-black pixels
    are treated as skeleton and are copied opaquely by default, so the real
    image is not globally darkened and bones are not washed out.
    """
    mask = np.any(skeleton_bgr > 0, axis=2)
    out = bg_bgr.copy()
    alpha = float(np.clip(overlay_alpha, 0.0, 1.0))
    if alpha >= 0.999:
        out[mask] = skeleton_bgr[mask]
    else:
        out[mask] = (
            skeleton_bgr[mask].astype(np.float32) * alpha
            + out[mask].astype(np.float32) * (1.0 - alpha)
        ).astype(np.uint8)
    return out


def _camera_view_batch(
    base_world: np.ndarray,
    camera_params: Dict[str, Any],
    cam_axis_mode: str,
    cam_pitch_offset: float,
    cam_yaw_offset: float,
) -> np.ndarray:
    """Batched camera view matrices (T,4,4) replicating the per-frame chain:

    _camera_link_transform -> _optical_remap -> _apply_pitch_yaw_offset -> inv.
    """
    t = base_world.shape[0]
    extrinsic_matrix = camera_params.get("extrinsic_matrix")
    if extrinsic_matrix is not None:
        # jsonl source: cam_t = base_world @ inv(M), then optional pitch/yaw fine
        # tune, then view = inv(cam_t). Mirrors the per-frame path in _draw_frame.
        inv_m = np.linalg.inv(np.asarray(extrinsic_matrix, dtype=np.float64))
        cam_t = base_world @ inv_m[None]
        cam_t[:, :3, :3] = cam_t[:, :3, :3] @ _DATA_TO_CV_OPTICAL[None]
        rot = _rot_axis("y", cam_pitch_offset) @ _rot_axis("x", cam_yaw_offset)
        cam_t[:, :3, :3] = cam_t[:, :3, :3] @ rot[None]
        return np.linalg.inv(cam_t)
    xyz = camera_params.get("camera_offset_xyz")
    rpy_deg = camera_params.get("camera_offset_rpy_deg")
    xyz_vec = _worker_camera_offset_xyz if xyz is None else np.asarray(xyz, dtype=np.float64)
    rpy_vec = _worker_camera_offset_rpy if rpy_deg is None else np.deg2rad(
        np.asarray(rpy_deg, dtype=np.float64)
    )
    offset = np.eye(4, dtype=np.float64)
    offset[:3, :3] = _rpy_matrix(rpy_vec)
    offset[:3, 3] = xyz_vec
    cam_t = base_world @ offset[None]
    r = cam_t[:, :3, :3]
    axes = CAM_AXIS_MODES[cam_axis_mode]
    out = cam_t.copy()
    out[:, :3, 0] = np.einsum("tij,j->ti", r, _axis_vector(axes[0]))
    out[:, :3, 1] = np.einsum("tij,j->ti", r, _axis_vector(axes[1]))
    out[:, :3, 2] = np.einsum("tij,j->ti", r, _axis_vector(axes[2]))
    rot = _rot_axis("y", cam_pitch_offset) @ _rot_axis("x", cam_yaw_offset)
    out[:, :3, :3] = out[:, :3, :3] @ rot[None]
    return np.linalg.inv(out)


def _iter_render_frames_vectorized(task: Dict[str, Any], source_name: str, overlay: bool = False):
    """Fast path: batched FK + projection for the whole episode, then per-frame cv2 draw only."""
    arrs = task.get("arrays")
    if not arrs:
        arrs = load_action_arrays_agibot(task["parquet_path"], need_state=True)
    source = arrs[source_name]
    n = int(source["arm"].shape[0])
    cap = None
    if overlay:
        cap = cv2.VideoCapture(task["video_path"])
        if not cap.isOpened():
            raise FileNotFoundError(task["video_path"])
        video_n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if video_n > 0:
            n = min(n, video_n)
    max_frames = int(task.get("max_frames") or 0)
    if max_frames > 0:
        n = min(n, max_frames)
    if n <= 0:
        if cap is not None:
            cap.release()
        return

    arm = source["arm"][:n]
    head = source["head"][:n]
    waist = source["waist"][:n]
    gripper, open_vals = _resolve_render_gripper(arrs, n)

    camera_params = task.get("camera_params") or {}
    fx = camera_params.get("fx")
    fy = camera_params.get("fy")
    cx = camera_params.get("cx")
    cy = camera_params.get("cy")
    fx = _worker_fx if fx is None else float(fx)
    fy = _worker_fy if fy is None else float(fy)
    cx = _worker_cx if cx is None else float(cx)
    cy = _worker_cy if cy is None else float(cy)

    cfg = _frame_cfg_batch(arm, head, waist, gripper)
    world = _fk_world_transforms_batch(cfg, n)
    default_yaw = float(task["cam_yaw_offset"])
    default_pitch = float(task["cam_pitch_offset"])
    left_view = _camera_view_batch(
        world[CAMERA_LINK],
        camera_params,
        task["cam_axis_mode"],
        float(task.get("left_cam_pitch_offset", default_pitch)),
        float(task.get("left_cam_yaw_offset", default_yaw)),
    )
    right_view = _camera_view_batch(
        world[CAMERA_LINK],
        camera_params,
        task["cam_axis_mode"],
        float(task.get("right_cam_pitch_offset", default_pitch)),
        float(task.get("right_cam_yaw_offset", default_yaw)),
    )

    kin = _worker_kin
    joint_points: Dict[str, np.ndarray] = {}
    for jname, _parent in _worker_active_arm_bones:
        info = kin["joints"][jname]
        parent_link = info["parent"]
        if parent_link is None:
            continue
        j_t = world[parent_link] @ info["origin"][None]
        joint_points[jname] = j_t[:, :3, 3]

    def link_anchor_traj(link_name: str) -> np.ndarray:
        inc = _worker_incoming_joint_by_link.get(link_name)
        if inc is not None and inc in joint_points:
            return joint_points[inc]
        return world[link_name][:, :3, 3]

    # Precompute arm bone pixel coords / validity.
    arm_draw: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for jname, parent in _worker_active_arm_bones:
        if jname not in joint_points:
            continue
        p1 = link_anchor_traj(parent)
        p2 = joint_points[jname]
        side_view = left_view if "arm_l_" in jname else right_view
        u1, v1, val1 = _project_batch(side_view, p1, fx, fy, cx, cy)
        u2, v2, val2 = _project_batch(side_view, p2, fx, fy, cx, cy)
        arm_draw.append((u1, v1, u2, v2, val1 & val2))

    # Gripper colors: settled = state-machine binary; motion bones = gray (RealSource).
    cof, lk, lb, eps, delta, fill, early = _gripper_close_params_from_task(task)
    closed_l, settled_l = _gripper_state_masks(
        open_vals[:, 0], cof, lk, eps, delta, lookback=lb, settled_fill=fill,
        early_closed_frac=early)
    closed_r, settled_r = _gripper_state_masks(
        open_vals[:, 1], cof, lk, eps, delta, lookback=lb, settled_fill=fill,
        early_closed_frac=early)
    grip_bone_col = {
        "left": _gripper_colors_state_or_gradient(
            open_vals[:, 0], closed_l, settled_l, COLOR_GRIPPER_OPEN_BONE, COLOR_GRIPPER_CLOSED_BONE),
        "right": _gripper_colors_state_or_gradient(
            open_vals[:, 1], closed_r, settled_r, COLOR_GRIPPER_OPEN_BONE, COLOR_GRIPPER_CLOSED_BONE),
    }
    grip_joint_col = {
        "left": _gripper_joint_colors_state_or_motion(closed_l, settled_l),
        "right": _gripper_joint_colors_state_or_motion(closed_r, settled_r),
    }
    gripper_forward_offset = float(task["gripper_forward_offset"])
    grip_draw: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for side, chains in GRIPPER_LINK_CHAINS.items():
        side_view = left_view if side == "left" else right_view
        end_link, base_link = GRIPPER_BASE_LINKS[side]
        if base_link not in world:
            continue
        forward = world[base_link][:, :3, 2]
        nrm = np.linalg.norm(forward, axis=1)
        safe = nrm > 1e-9
        scale = np.where(safe, gripper_forward_offset / np.where(safe, nrm, 1.0), 0.0)
        shift = forward * scale[:, None]
        bcol = grip_bone_col[side]
        jcol = grip_joint_col[side]
        for parent, child in chains:
            if parent not in world or child not in world:
                continue
            p1 = world[parent][:, :3, 3].copy()
            p2 = world[child][:, :3, 3].copy()
            if parent != end_link:
                p1 = p1 + shift
            p2 = p2 + shift
            u1, v1, val1 = _project_batch(side_view, p1, fx, fy, cx, cy)
            u2, v2, val2 = _project_batch(side_view, p2, fx, fy, cx, cy)
            grip_draw.append((u1, v1, u2, v2, val1 & val2, bcol, jcol))

    template = (
        _worker_black_frame
        if (overlay or os.environ.get("AC_BLACK_BACKGROUND") == "1")
        else _worker_white_frame
    )
    overlay_alpha = float(task["overlay_alpha"]) if overlay else 1.0
    for t in range(n):
        img = template.copy()
        for u1, v1, u2, v2, valid in arm_draw:
            if valid[t]:
                pt1 = (int(u1[t]), int(v1[t]))
                pt2 = (int(u2[t]), int(v2[t]))
                cv2.line(img, pt1, pt2, COLOR_ACTIVE_BONE, ACTIVE_BONE_WIDTH, cv2.LINE_AA)
                cv2.circle(img, pt2, ACTIVE_JOINT_RADIUS, COLOR_ACTIVE_JOINT, -1, cv2.LINE_AA)
        for u1, v1, u2, v2, valid, bcol, jcol in grip_draw:
            if valid[t]:
                pt1 = (int(u1[t]), int(v1[t]))
                pt2 = (int(u2[t]), int(v2[t]))
                bone_color = (int(bcol[t, 0]), int(bcol[t, 1]), int(bcol[t, 2]))
                joint_color = (int(jcol[t, 0]), int(jcol[t, 1]), int(jcol[t, 2]))
                cv2.line(img, pt1, pt2, bone_color, ACTIVE_BONE_WIDTH, cv2.LINE_AA)
                cv2.circle(img, pt2, ACTIVE_JOINT_RADIUS, joint_color, -1, cv2.LINE_AA)
        if overlay:
            ok, bg = cap.read()
            if not ok:
                break
            if bg.shape[1] != _worker_img_w or bg.shape[0] != _worker_img_h:
                bg = cv2.resize(bg, (_worker_img_w, _worker_img_h), interpolation=cv2.INTER_AREA)
            img = _overlay_skeleton_opaque(bg, img, overlay_alpha)
        yield img
    if cap is not None:
        cap.release()


def _iter_render_frames(task: Dict[str, Any], source_name: str, overlay: bool = False):
    if str(task.get("fk_backend", "yourdfpy")) == "vectorized":
        yield from _iter_render_frames_vectorized(task, source_name, overlay=overlay)
        return
    arrs = load_action_arrays_agibot(task["parquet_path"], need_state=True)
    source = arrs[source_name]
    n = int(source["arm"].shape[0])
    cap = None
    if overlay:
        cap = cv2.VideoCapture(task["video_path"])
        if not cap.isOpened():
            raise FileNotFoundError(task["video_path"])
        video_n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if video_n > 0:
            n = min(n, video_n)
    max_frames = int(task.get("max_frames") or 0)
    if max_frames > 0:
        n = min(n, max_frames)
    gripper_kin, open_vals = _resolve_render_gripper(arrs, n)
    cof, lk, lb, eps, delta, fill, early = _gripper_close_params_from_task(task)
    closed_l, settled_l = _gripper_state_masks(
        open_vals[:, 0], cof, lk, eps, delta, lookback=lb, settled_fill=fill,
        early_closed_frac=early)
    closed_r, settled_r = _gripper_state_masks(
        open_vals[:, 1], cof, lk, eps, delta, lookback=lb, settled_fill=fill,
        early_closed_frac=early)
    bone_l = _gripper_colors_state_or_gradient(
        open_vals[:, 0], closed_l, settled_l, COLOR_GRIPPER_OPEN_BONE, COLOR_GRIPPER_CLOSED_BONE)
    bone_r = _gripper_colors_state_or_gradient(
        open_vals[:, 1], closed_r, settled_r, COLOR_GRIPPER_OPEN_BONE, COLOR_GRIPPER_CLOSED_BONE)
    joint_l = _gripper_joint_colors_state_or_motion(closed_l, settled_l)
    joint_r = _gripper_joint_colors_state_or_motion(closed_r, settled_r)
    for t in range(n):
        frame = _draw_frame(
            source["arm"][t],
            gripper_kin[t],
            source["head"][t],
            source["waist"][t],
            task["cam_axis_mode"],
            float(task["cam_pitch_offset"]),
            float(task["cam_yaw_offset"]),
            float(task["gripper_forward_offset"]),
            task.get("camera_params"),
            background="black" if (overlay or os.environ.get("AC_BLACK_BACKGROUND") == "1") else "white",
            left_gripper_bone=tuple(int(x) for x in bone_l[t]),
            right_gripper_bone=tuple(int(x) for x in bone_r[t]),
            left_gripper_joint=tuple(int(x) for x in joint_l[t]),
            right_gripper_joint=tuple(int(x) for x in joint_r[t]),
            left_cam_yaw_offset=float(
                task.get("left_cam_yaw_offset", task["cam_yaw_offset"])
            ),
            right_cam_yaw_offset=float(
                task.get("right_cam_yaw_offset", task["cam_yaw_offset"])
            ),
            left_cam_pitch_offset=float(
                task.get("left_cam_pitch_offset", task["cam_pitch_offset"])
            ),
            right_cam_pitch_offset=float(
                task.get("right_cam_pitch_offset", task["cam_pitch_offset"])
            ),
        )
        if overlay:
            ok, bg = cap.read()
            if not ok:
                break
            if bg.shape[1] != _worker_img_w or bg.shape[0] != _worker_img_h:
                bg = cv2.resize(bg, (_worker_img_w, _worker_img_h), interpolation=cv2.INTER_AREA)
            frame = _overlay_skeleton_opaque(bg, frame, float(task["overlay_alpha"]))
        yield frame
    if cap is not None:
        cap.release()


def render_one(task: Dict[str, Any]) -> Dict[str, Any]:
    try:
        action_status = "skip"
        overlay_status = "skip"
        frames = 0
        if task["save_action"]:
            action_path = Path(task["action_path"])
            if task["resume"] and action_path.exists() and action_path.stat().st_size > 1024:
                action_status = "exists"
            else:
                frames = _write_video_iter(
                    str(action_path),
                    _iter_render_frames(task, "action", overlay=False),
                    float(task["fps"]),
                )
                action_status = "ok"
        if task["save_overlay"]:
            overlay_path = Path(task["overlay_path"])
            if task["resume"] and overlay_path.exists() and overlay_path.stat().st_size > 1024:
                overlay_status = "exists"
            else:
                source_name = "state" if task["use_state_for_overlay"] else "action"
                overlay_frames = _write_video_iter(
                    str(overlay_path),
                    _iter_render_frames(task, source_name, overlay=True),
                    float(task["fps"]),
                )
                frames = max(frames, overlay_frames)
                overlay_status = "ok"
        return {**task, "status": "ok", "action_status": action_status, "overlay_status": overlay_status, "frames": frames}
    except Exception as exc:
        return {**task, "status": "fail", "reason": repr(exc)}


def load_task_meta(task_dir: Path) -> Tuple[dict, List[dict]]:
    meta_dir = task_dir / "meta"
    info = json.loads((meta_dir / "info.json").read_text())
    episodes = []
    with (meta_dir / "episodes.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                episodes.append(json.loads(line))
    return info, episodes


def episode_parquet_path(task_dir: Path, info: dict, episode_index: int) -> Path:
    tmpl = info.get("data_path", "data/episode_{episode_index:06d}.parquet")
    return task_dir / tmpl.format(episode_index=episode_index)


def episode_video_path(task_dir: Path, info: dict, episode_index: int, video_key: str) -> Path:
    tmpl = info.get("video_path", "videos/{video_key}/episode_{episode_index:06d}.mp4")
    return task_dir / tmpl.format(video_key=video_key, episode_index=episode_index)


def parse_episode_list(value: str) -> Optional[set[int]]:
    if not value:
        return None
    out: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if part:
            out.add(int(part))
    return out


def load_episode_camera_mapping(
    path: str,
    overrides_path: Optional[str] = None,
) -> Dict[Tuple[str, int], Dict[str, Any]]:
    if not path or not Path(path).exists():
        return {}
    mapping: Dict[Tuple[str, int], Dict[str, Any]] = {}
    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("has_head_params", "")).lower() != "true":
                continue
            task_name = str(row.get("task_name") or "")
            episode_index_text = str(row.get("episode_index") or "")
            if not task_name or not episode_index_text:
                continue
            try:
                episode_index = int(float(episode_index_text))
                fx = float(row["fx"])
                fy = float(row["fy"])
                cx = float(row["cx"])
                cy = float(row["cy"])
                camera_offset_xyz = [
                    float(row["camera_offset_x"]),
                    float(row["camera_offset_y"]),
                    float(row["camera_offset_z"]),
                ]
            except (KeyError, TypeError, ValueError):
                continue
            mapping[(task_name, episode_index)] = {
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "camera_offset_xyz": camera_offset_xyz,
                "raw_episode_id": row.get("raw_episode_id", ""),
                "head_intrinsic_path": row.get("head_intrinsic_path", ""),
                "head_extrinsic_path": row.get("head_extrinsic_path", ""),
                "source": "episode_official",
            }
    apply_task_camera_extrinsic_overrides(mapping, overrides_path=overrides_path)
    return mapping


def _mean_camera_fields(
    entries: Sequence[Dict[str, Any]],
    fields: Sequence[str],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not entries:
        return out
    if "camera_offset_xyz" in fields:
        arr = np.asarray([e["camera_offset_xyz"] for e in entries], dtype=np.float64)
        out["camera_offset_xyz"] = [float(x) for x in arr.mean(axis=0)]
    for key in ("fx", "fy", "cx", "cy"):
        if key in fields:
            vals = np.asarray([float(e[key]) for e in entries], dtype=np.float64)
            out[key] = float(vals.mean())
    return out


def apply_task_camera_extrinsic_overrides(
    mapping: Dict[Tuple[str, int], Dict[str, Any]],
    overrides_path: Optional[str] = None,
) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """Replace broken per-task camera params using a small JSON override table."""
    path = overrides_path if overrides_path is not None else DEFAULT_TASK_CAMERA_EXTRINSIC_OVERRIDES
    if not path or not Path(path).exists():
        return mapping
    try:
        overrides = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return mapping
    if not isinstance(overrides, dict):
        return mapping

    for target_task, spec in overrides.items():
        if not isinstance(spec, dict):
            continue
        strategy = str(spec.get("strategy") or "")
        fields = spec.get("fields") or ["camera_offset_xyz"]
        fields = [str(x) for x in fields]
        source_entries: List[Dict[str, Any]] = []
        source_label = strategy

        if strategy == "dexhand_mean_excluding_task":
            source_entries = [
                entry
                for (task_name, _ei), entry in mapping.items()
                if task_name.startswith("agibot_dexhand__") and task_name != target_task
            ]
            source_label = "borrowed_dexhand_mean_excluding_task"
        elif strategy == "copy_from_task":
            source_task = str(spec.get("source_task") or "")
            if not source_task:
                continue
            source_entries = [
                entry for (task_name, _ei), entry in mapping.items() if task_name == source_task
            ]
            source_label = f"borrowed_from_{source_task}"
        else:
            continue

        borrowed = _mean_camera_fields(source_entries, fields)
        if not borrowed:
            continue

        for (task_name, _ei), entry in mapping.items():
            if task_name != target_task:
                continue
            for key, val in borrowed.items():
                entry[key] = val
            entry["source"] = source_label
            entry["camera_extrinsic_override"] = target_task
    return mapping


def _read_first_jsonl(path: Path) -> Optional[dict]:
    try:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    return json.loads(line)
    except (OSError, ValueError):
        return None
    return None


def load_jsonl_camera_params(
    video_path: Path,
    extrinsic_invert: bool = False,
) -> Optional[Dict[str, Any]]:
    """Per-episode head camera params from the LeRobot jsonl sidecars.

    Reads `<stem>_intrinsic.jsonl` and `<stem>_extrinsic.jsonl` next to the
    head_color mp4. Intrinsics give fx/fy/cx(ppx)/cy(ppy); the static extrinsic
    gives a 3x3 `rotation_matrix` and 3 `translation_vector` which we pack into a
    4x4 matrix `M` (link->camera-optical), so the renderer can use the *real*
    camera orientation instead of a hard-coded rpy. The virtual camera pose is
    then `cam_t = T(head_link2) @ inv(M)` (see `_draw_frame`/`_camera_view_batch`).

    Returns None if either sidecar is missing or malformed.
    """
    video_path = Path(video_path)
    intr_path = video_path.with_name(video_path.stem + "_intrinsic.jsonl")
    extr_path = video_path.with_name(video_path.stem + "_extrinsic.jsonl")
    if not intr_path.exists() or not extr_path.exists():
        return None
    intr = _read_first_jsonl(intr_path)
    extr = _read_first_jsonl(extr_path)
    if not intr or not extr:
        return None
    intrinsic = intr.get("intrinsic") or {}
    extrinsic = extr.get("extrinsic") or {}
    try:
        fx = float(intrinsic["fx"])
        fy = float(intrinsic["fy"])
        cx = float(intrinsic["ppx"])
        cy = float(intrinsic["ppy"])
        r = np.asarray(extrinsic["rotation_matrix"], dtype=np.float64).reshape(3, 3)
        t = np.asarray(extrinsic["translation_vector"], dtype=np.float64).reshape(3)
    except (KeyError, TypeError, ValueError):
        return None
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = r
    m[:3, 3] = t
    if extrinsic_invert:
        m = np.linalg.inv(m)
    return {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "extrinsic_matrix": m.tolist(),
        "source": "episode_jsonl",
        "head_intrinsic_path": str(intr_path),
        "head_extrinsic_path": str(extr_path),
    }


def build_render_tasks_from_mapping(
    args: argparse.Namespace,
    episode_camera_mapping: Dict[Tuple[str, int], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    data_root = Path(args.data_root)
    selected: List[Dict[str, Any]] = []
    task_info_cache: Dict[str, dict] = {}
    for (task_name, ei), mapped_camera in sorted(episode_camera_mapping.items()):
        if not task_name.startswith("agibot_gripper__task_"):
            continue
        task_dir = data_root / task_name
        if not task_dir.exists():
            continue
        if task_name not in task_info_cache:
            info, _ = load_task_meta(task_dir)
            task_info_cache[task_name] = info
        info = task_info_cache[task_name]
        parquet = episode_parquet_path(task_dir, info, ei)
        if not parquet.exists():
            continue
        video = episode_video_path(task_dir, info, ei, args.video_key)
        if args.save_mode in ("overlay", "overlay_and_action") and not video.exists():
            continue
        action_path = Path(args.action_output_dir or args.output_dir) / task_name / f"episode_{ei:06d}.mp4"
        overlay_path = Path(args.output_dir) / task_name / f"episode_{ei:06d}_overlay.mp4"
        camera_params = {
            "fx": float(args.fx),
            "fy": float(args.fy),
            "cx": float(args.cx),
            "cy": float(args.cy),
            "camera_offset_xyz": [float(x) for x in args.camera_offset_xyz],
            "camera_offset_rpy_deg": [float(x) for x in args.camera_offset_rpy_deg],
            "source": "mean_default",
            "raw_episode_id": "",
            "head_intrinsic_path": "",
            "head_extrinsic_path": "",
        }
        camera_params.update(mapped_camera)
        camera_params["camera_offset_rpy_deg"] = [float(x) for x in args.camera_offset_rpy_deg]
        selected.append({
            "task_name": task_name,
            "episode_index": ei,
            "parquet_path": str(parquet),
            "video_path": str(video),
            "action_path": str(action_path),
            "overlay_path": str(overlay_path),
            "fps": float(args.fps),
            "save_action": args.save_mode in ("action", "overlay_and_action"),
            "save_overlay": args.save_mode in ("overlay", "overlay_and_action"),
            "use_state_for_overlay": bool(args.use_state_for_overlay),
            "overlay_alpha": float(args.overlay_alpha),
            "cam_axis_mode": args.cam_axis_mode,
            "cam_pitch_offset": float(args.cam_pitch_offset),
            "cam_yaw_offset": float(args.cam_yaw_offset),
            "arm_l_base_y": float(args.arm_l_base_y),
            "arm_r_base_y": float(args.arm_r_base_y),
            "gripper_forward_offset": float(args.gripper_forward_offset),
            "gripper_delay_frames": int(args.gripper_delay_frames),
            "gripper_closed_open_frac": float(args.gripper_closed_open_frac),
            "gripper_state_lookahead": int(args.gripper_state_lookahead),
            "gripper_state_lookback": int(args.gripper_state_lookback),
            "gripper_state_plateau_eps": float(args.gripper_state_plateau_eps),
            "gripper_state_change_delta": float(args.gripper_state_change_delta),
            "gripper_state_settled_fill": int(args.gripper_state_settled_fill),
            "gripper_early_closed_frac": float(args.gripper_early_closed_frac),
            "camera_params": camera_params,
            "max_frames": int(args.max_frames),
            "resume": bool(args.resume),
            "fk_backend": str(args.fk_backend),
        })
        if args.limit > 0 and len(selected) >= args.limit:
            break
    return selected


def build_render_tasks(args: argparse.Namespace) -> List[Dict[str, Any]]:
    data_root = Path(args.data_root)
    task_names: List[str]
    if args.tasks:
        task_names = [x.strip() for x in args.tasks.split(",") if x.strip()]
    else:
        task_names = sorted(p.name for p in data_root.iterdir() if p.is_dir() and fnmatch.fnmatch(p.name, args.task_glob))
    rng = random.Random(args.seed)
    selected: List[Dict[str, Any]] = []
    episode_filter = parse_episode_list(args.episodes)
    use_jsonl = args.camera_param_source == "jsonl"
    episode_camera_mapping = (
        {}
        if (args.no_episode_camera_params or use_jsonl)
        else load_episode_camera_mapping(args.episode_camera_mapping)
    )
    if args.require_episode_camera_params and not use_jsonl and not episode_camera_mapping:
        raise ValueError(
            "--require-episode-camera-params needs a valid --episode-camera-mapping "
            "with at least one official episode camera entry"
        )
    if use_jsonl:
        print("[Main] camera params source: jsonl sidecars (real extrinsic)", flush=True)
    if episode_camera_mapping:
        print(
            f"[Main] loaded episode camera params: {len(episode_camera_mapping)} "
            f"from {args.episode_camera_mapping}",
            flush=True,
        )
    if (
        args.require_episode_camera_params
        and args.all_episodes
        and episode_filter is None
        and episode_camera_mapping
    ):
        return build_render_tasks_from_mapping(args, episode_camera_mapping)
    for task_name in task_names:
        task_dir = data_root / task_name
        if not task_dir.exists():
            continue
        info, episodes = load_task_meta(task_dir)
        episodes = list(episodes)
        if episode_filter is not None:
            episodes = [ep for ep in episodes if int(ep["episode_index"]) in episode_filter]
        if args.require_episode_camera_params and not use_jsonl:
            episodes = [
                ep for ep in episodes
                if (task_name, int(ep["episode_index"])) in episode_camera_mapping
            ]
        if episode_filter is None and not args.all_episodes:
            rng.shuffle(episodes)
            episodes = episodes[: max(0, int(args.videos_per_task))]
        for ep in episodes:
            ei = int(ep["episode_index"])
            parquet = episode_parquet_path(task_dir, info, ei)
            if not parquet.exists():
                continue
            video = episode_video_path(task_dir, info, ei, args.video_key)
            if args.save_mode in ("overlay", "overlay_and_action") and not video.exists():
                continue
            action_path = Path(args.action_output_dir or args.output_dir) / task_name / f"episode_{ei:06d}.mp4"
            overlay_path = Path(args.output_dir) / task_name / f"episode_{ei:06d}_overlay.mp4"
            camera_params = {
                "fx": float(args.fx),
                "fy": float(args.fy),
                "cx": float(args.cx),
                "cy": float(args.cy),
                "camera_offset_xyz": [float(x) for x in args.camera_offset_xyz],
                "camera_offset_rpy_deg": [float(x) for x in args.camera_offset_rpy_deg],
                "source": "mean_default",
                "raw_episode_id": "",
                "head_intrinsic_path": "",
                "head_extrinsic_path": "",
            }
            if use_jsonl:
                jsonl_camera = load_jsonl_camera_params(video, args.extrinsic_invert)
                if jsonl_camera is None:
                    if args.require_episode_camera_params:
                        continue
                else:
                    camera_params.update(jsonl_camera)
            else:
                mapped_camera = episode_camera_mapping.get((task_name, ei))
                if mapped_camera is not None:
                    camera_params.update(mapped_camera)
                    camera_params["camera_offset_rpy_deg"] = [float(x) for x in args.camera_offset_rpy_deg]
            selected.append({
                "task_name": task_name,
                "episode_index": ei,
                "parquet_path": str(parquet),
                "video_path": str(video),
                "action_path": str(action_path),
                "overlay_path": str(overlay_path),
                "fps": float(args.fps),
                "save_action": args.save_mode in ("action", "overlay_and_action"),
                "save_overlay": args.save_mode in ("overlay", "overlay_and_action"),
                "use_state_for_overlay": bool(args.use_state_for_overlay),
                "overlay_alpha": float(args.overlay_alpha),
                "cam_axis_mode": args.cam_axis_mode,
                "cam_pitch_offset": float(args.cam_pitch_offset),
                "cam_yaw_offset": float(args.cam_yaw_offset),
                "arm_l_base_y": float(args.arm_l_base_y),
                "arm_r_base_y": float(args.arm_r_base_y),
                "gripper_forward_offset": float(args.gripper_forward_offset),
                "gripper_delay_frames": int(args.gripper_delay_frames),
                "gripper_closed_open_frac": float(args.gripper_closed_open_frac),
                "gripper_state_lookahead": int(args.gripper_state_lookahead),
                "gripper_state_lookback": int(args.gripper_state_lookback),
                "gripper_state_plateau_eps": float(args.gripper_state_plateau_eps),
                "gripper_state_change_delta": float(args.gripper_state_change_delta),
                "gripper_state_settled_fill": int(args.gripper_state_settled_fill),
                "gripper_early_closed_frac": float(args.gripper_early_closed_frac),
                "camera_params": camera_params,
                "max_frames": int(args.max_frames),
                "resume": bool(args.resume),
                "fk_backend": str(args.fk_backend),
            })
            if args.limit > 0 and len(selected) >= args.limit:
                return selected
    return selected


def write_metadata(records: List[Dict[str, Any]], path: str) -> None:
    if not path:
        return
    out = []
    for rec in records:
        item = {
            "task_name": rec.get("task_name"),
            "episode_index": rec.get("episode_index"),
            "parquet_path": rec.get("parquet_path"),
            "video": rec.get("video_path"),
            "action_video": rec.get("action_path") if rec.get("action_status") in ("ok", "exists") else "",
            "overlay_video": rec.get("overlay_path") if rec.get("overlay_status") in ("ok", "exists") else "",
            "status": rec.get("status"),
            "frames": rec.get("frames"),
            "camera_params": {
                **(rec.get("camera_params") or {}),
            },
        }
        out.append(item)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if path.endswith(".gz"):
        with gzip.open(path, "wb") as f:
            pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        Path(path).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render AgiBot gripper action/overlay videos")
    p.add_argument("--data-root", default="/datasets/AgiBot-World_v2.1_per_task_fixed")
    p.add_argument("--task-glob", default="agibot_gripper__task_*")
    p.add_argument("--tasks", default="")
    p.add_argument("--episodes", default="")
    p.add_argument("--all-episodes", action="store_true")
    p.add_argument("--videos-per-task", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--save-mode", choices=["action", "overlay", "overlay_and_action"], default="action")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--action-output-dir", default="")
    p.add_argument("--metadata-output", default="")
    p.add_argument("--episode-camera-mapping", default=DEFAULT_EPISODE_CAMERA_MAPPING)
    p.add_argument(
        "--camera-param-source",
        choices=["csv", "jsonl"],
        default="csv",
        help="Per-episode head camera params source: 'csv' mapping (default, "
        "backward compatible) or 'jsonl' LeRobot sidecars next to the head video "
        "(uses the real extrinsic rotation matrix).",
    )
    p.add_argument(
        "--extrinsic-invert",
        action="store_true",
        help="Invert the jsonl extrinsic matrix [R|t] (use if the convention is "
        "camera->link instead of link->camera). Only affects --camera-param-source jsonl.",
    )
    p.add_argument("--no-episode-camera-params", action="store_true")
    p.add_argument("--require-episode-camera-params", action="store_true")
    p.add_argument("--video-key", default="observation.images.head_color")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--chunksize", type=int, default=1)
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--width", type=int, default=IMG_W)
    p.add_argument("--height", type=int, default=IMG_H)
    p.add_argument("--fx", type=float, default=MEAN_FX)
    p.add_argument("--fy", type=float, default=MEAN_FY)
    p.add_argument("--cx", type=float, default=MEAN_CX)
    p.add_argument("--cy", type=float, default=MEAN_CY)
    p.add_argument("--camera-offset-xyz", type=float, nargs=3, default=DEFAULT_CAMERA_OFFSET_XYZ.tolist())
    p.add_argument("--camera-offset-rpy-deg", type=float, nargs=3, default=DEFAULT_CAMERA_OFFSET_RPY_DEG.tolist())
    p.add_argument("--cam-axis-mode", choices=sorted(CAM_AXIS_MODES), default="x_fwd")
    p.add_argument("--cam-pitch-offset", type=float, default=0.0)
    p.add_argument("--cam-yaw-offset", type=float, default=0.0)
    p.add_argument(
        "--arm-l-base-y",
        type=float,
        default=DEFAULT_ARM_L_BASE_Y,
        help="Left arm_base joint origin Y in arm_base_link frame (meters, URDF default 0.025).",
    )
    p.add_argument(
        "--arm-r-base-y",
        type=float,
        default=DEFAULT_ARM_R_BASE_Y,
        help="Right arm_base joint origin Y in arm_base_link frame (meters, URDF default -0.025).",
    )
    p.add_argument("--gripper-forward-offset", type=float, default=DEFAULT_GRIPPER_FORWARD_OFFSET)
    p.add_argument("--gripper-delay-frames", type=int, default=DEFAULT_GRIPPER_DELAY_FRAMES)
    p.add_argument("--gripper-closed-open-frac", type=float, default=GRIPPER_CLOSED_OPEN_FRAC,
                   help="Absolute open fraction used to classify the FIRST plateau as closed (>=) or open (<).")
    p.add_argument("--gripper-state-lookahead", type=int, default=GRIPPER_STATE_LOOKAHEAD,
                   help="Future frames (plus current) used to judge plateau stability in the state machine.")
    p.add_argument("--gripper-state-lookback", type=int, default=GRIPPER_STATE_LOOKBACK,
                   help="Past frames used with lookahead for bidirectional plateau stability.")
    p.add_argument("--gripper-state-plateau-eps", type=float, default=GRIPPER_STATE_PLATEAU_EPS,
                   help="Max open-fraction spread within the stability window to count as a stable plateau.")
    p.add_argument("--gripper-state-change-delta", type=float, default=GRIPPER_STATE_CHANGE_DELTA,
                   help="Minimum plateau-to-plateau change to flip the open/closed state (should exceed eps).")
    p.add_argument("--gripper-state-settled-fill", type=int, default=GRIPPER_STATE_SETTLED_FILL,
                   help="Fill unsettled gaps shorter than/equal to this many frames to reduce color flicker.")
    p.add_argument("--gripper-early-closed-frac", type=float, default=GRIPPER_EARLY_CLOSED_FRAC,
                   help="If open_frac >= this and the forward window is stable, confirm CLOSED early "
                        "(ignore lookback still containing the closing ramp).")
    p.add_argument("--gripper-max-q", type=float, default=GRIPPER_120S_MAX_Q,
                   help="crt_120s drive-joint angle at full opening (rad)")
    p.add_argument("--gripper-invert", action="store_true", default=DEFAULT_GRIPPER_INVERT,
                   help="flip the gripper open/close direction if it looks reversed")
    p.add_argument("--overlay-alpha", type=float, default=1.0)
    p.add_argument("--use-state-for-overlay", action="store_true")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--urdf", default=DEFAULT_URDF)
    p.add_argument("--ffmpeg-preset", default="veryfast")
    p.add_argument("--ffmpeg-crf", type=int, default=23)
    p.add_argument("--ffmpeg-threads", type=int, default=1)
    p.add_argument("--fk-backend", choices=["yourdfpy", "vectorized"], default="yourdfpy")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.save_mode in ("action", "overlay_and_action") and not args.action_output_dir:
        if args.save_mode == "action":
            args.action_output_dir = args.output_dir
        else:
            raise ValueError("--save-mode overlay_and_action requires --action-output-dir")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.action_output_dir:
        Path(args.action_output_dir).mkdir(parents=True, exist_ok=True)
    tasks = build_render_tasks(args)
    print(
        f"[Main] tasks={len(tasks)} save_mode={args.save_mode} workers={args.workers} "
        f"fx/fy/cx/cy={args.fx:.2f}/{args.fy:.2f}/{args.cx:.2f}/{args.cy:.2f} "
        f"cam_xyz={args.camera_offset_xyz}",
        flush=True,
    )
    if not tasks:
        return 0
    initargs = (
        args.urdf,
        args.width,
        args.height,
        args.fx,
        args.fy,
        args.cx,
        args.cy,
        args.camera_offset_xyz,
        args.camera_offset_rpy_deg,
        args.ffmpeg_preset,
        args.ffmpeg_crf,
        args.ffmpeg_threads,
        args.gripper_max_q,
        args.gripper_invert,
        float(args.arm_l_base_y),
        float(args.arm_r_base_y),
    )
    num_workers = min(max(1, args.workers), len(tasks))
    chunksize = max(1, int(args.chunksize))
    records: List[Dict[str, Any]] = []
    t0 = time.time()
    ok = fail = 0
    with mp.Pool(num_workers, initializer=_init_worker, initargs=initargs) as pool:
        for rec in pool.imap_unordered(render_one, tasks, chunksize=chunksize):
            records.append(rec)
            if rec.get("status") == "ok":
                ok += 1
            else:
                fail += 1
                print(f"[FAIL] {rec.get('task_name')}#{rec.get('episode_index')}: {rec.get('reason')}", flush=True)
            done = ok + fail
            if done % 25 == 0 or done == len(tasks):
                elapsed = time.time() - t0
                print(
                    f"[Main] done={done}/{len(tasks)} ok={ok} fail={fail} "
                    f"elapsed={elapsed/60:.1f}m last={rec.get('task_name')}#{rec.get('episode_index')}",
                    flush=True,
                )
    write_metadata(records, args.metadata_output)
    print(f"[Main] finished ok={ok} fail={fail} elapsed={time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
