"""
批量离线渲染 action 火柴棍视频 —— Head View（头部相机视角）
适配 AG95 夹爪版 URDF（final_tiangong2_EVT_with_grippers.urdf）

与灵巧手版本的区别：
- 手部 action 从 12 维 (6+6 手指关节) 变为 2 维 (左夹爪 + 右夹爪)
- 夹爪数据来源于 parquet 中的 hand_joints_left (T,1) + hand_joints_right (T,1)
- 驱动关节为 ag95_l/r_gripper_finger1_joint，其余夹爪关节为 mimic 自动传播

用法示例:
    python visualization_urdf/batch_render_action_video_gripper.py \
        --metadata /path/to/gripper_data.pkl.gz \
        --output-dir /mnt/world_foundational_model/jackzou/OminiEWM/action_video_gripper/ \
        --workers 16 --resume

    同时输出「拼接检查用」与「纯 action 训练用」到两个目录（一次渲染，双目录落盘）:
    python visualization_urdf/batch_render_action_video_gripper.py \
        --metadata /path/to/gripper_data.pkl.gz \
        --save-mode concat_and_action \
        --output-dir /path/to/concat_check/ \
        --action-output-dir /path/to/action_only/ \
        --workers 16 --resume

    输出骨骼叠加到原始视频上的 overlay 视频（用于检查对齐）:
    python visualization_urdf/batch_render_action_video_gripper.py \
        --metadata /path/to/gripper_data.pkl.gz \
        --save-mode overlay \
        --output-dir /path/to/overlay_check/ \
        --overlay-alpha 0.7 \
        --workers 16 --resume

    同时输出 overlay 视频和纯 action 视频:
    python visualization_urdf/batch_render_action_video_gripper.py \
        --metadata /path/to/gripper_data.pkl.gz \
        --save-mode overlay_and_action \
        --output-dir /path/to/overlay_check/ \
        --action-output-dir /path/to/action_only/ \
        --overlay-alpha 0.7 \
        --workers 16 --resume
"""
from __future__ import annotations

import os as _os
for _e in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_e, "1")

import argparse
import copy
import gzip
import json
import multiprocessing as mp
import os
import pickle
import random
import shutil
import sys
import time
from collections import Counter, deque
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from visualize_action_on_urdf import (
    LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS,
    detect_fps,
    _as_transform,
    _link_name,
)

# ---------------------------------------------------------------------------
# 夹爪关节定义
# ---------------------------------------------------------------------------
LEFT_GRIPPER_DRIVING_JOINT = "ag95_l_gripper_finger1_joint"
RIGHT_GRIPPER_DRIVING_JOINT = "ag95_r_gripper_finger1_joint"

LEFT_GRIPPER_MIMIC_JOINTS = {
    "ag95_l_gripper_finger2_joint": 1.0,
    "ag95_l_gripper_finger1_finger_joint": 0.4563942,
    "ag95_l_gripper_finger2_finger_joint": 0.4563942,
    "ag95_l_gripper_finger1_inner_knuckle_joint": 1.49462955,
    "ag95_l_gripper_finger2_inner_knuckle_joint": 1.49462955,
    "ag95_l_gripper_finger1_finger_tip_joint": 1.49462955,
    "ag95_l_gripper_finger2_finger_tip_joint": 1.49462955,
}
RIGHT_GRIPPER_MIMIC_JOINTS = {
    "ag95_r_gripper_finger2_joint": 1.0,
    "ag95_r_gripper_finger1_finger_joint": 0.4563942,
    "ag95_r_gripper_finger2_finger_joint": 0.4563942,
    "ag95_r_gripper_finger1_inner_knuckle_joint": 1.49462955,
    "ag95_r_gripper_finger2_inner_knuckle_joint": 1.49462955,
    "ag95_r_gripper_finger1_finger_tip_joint": 1.49462955,
    "ag95_r_gripper_finger2_finger_tip_joint": 1.49462955,
}

ALL_GRIPPER_JOINTS = (
    [LEFT_GRIPPER_DRIVING_JOINT] + list(LEFT_GRIPPER_MIMIC_JOINTS.keys())
    + [RIGHT_GRIPPER_DRIVING_JOINT] + list(RIGHT_GRIPPER_MIMIC_JOINTS.keys())
)

GRIPPER_JOINT_LIMIT = (0.0, 0.6524)

LEFT_GRIPPER_JOINT_SET = frozenset(
    [LEFT_GRIPPER_DRIVING_JOINT] + list(LEFT_GRIPPER_MIMIC_JOINTS.keys())
)
RIGHT_GRIPPER_JOINT_SET = frozenset(
    [RIGHT_GRIPPER_DRIVING_JOINT] + list(RIGHT_GRIPPER_MIMIC_JOINTS.keys())
)

# 夹爪配色（RGB，写入 ffmpeg rgb24；视觉对齐 RealSource / AgiBot）
# settled：开=蓝骨/红关节，闭=橙骨/绿关节；运动（unsettled）：骨线灰、关节红
COLOR_ARM_BONE = (0, 120, 255)
COLOR_ARM_JOINT = (255, 50, 50)
COLOR_GRIPPER_OPEN_BONE = (0, 120, 255)
COLOR_GRIPPER_CLOSED_BONE = (255, 140, 0)
COLOR_GRIPPER_OPEN_JOINT = (255, 50, 50)
COLOR_GRIPPER_CLOSED_JOINT = (0, 200, 0)
COLOR_GRIPPER_MOTION_BONE = (128, 128, 128)
COLOR_GRIPPER_MOTION_JOINT = (255, 50, 50)
COLOR_INACTIVE = (192, 192, 192)

# 夹爪时序状态机（close_frac: HIGH=物理闭合）
# 对齐 RoboCasa/LIBERO 抗闪烁：中值平滑、短 settled 岛抑制、短 unsettled 填补；
# 且仅在开合翻转时更新 ref，避免慢速闭合中途平台把状态爬成橙/灰闪烁。
GRIPPER_CLOSED_OPEN_FRAC = 0.5
GRIPPER_STATE_LOOKAHEAD = 10
GRIPPER_STATE_LOOKBACK = 10
GRIPPER_STATE_PLATEAU_EPS = 0.04
GRIPPER_STATE_CHANGE_DELTA = 0.1
GRIPPER_STATE_SETTLED_FILL = 16
GRIPPER_STATE_MIN_SETTLED_RUN = 6
GRIPPER_STATE_SMOOTH_K = 5
# 仅模糊带 (开合途中) 的短平台可降级；>=0.55 视为已闭合，不被 mid-demote 误杀
GRIPPER_STATE_MID_LO = 0.15
GRIPPER_STATE_MID_HI = 0.55
# 仅打掉短暂的模糊带平台；长于该长度的 ~0.5 平台 = 夹住物体，应显示橙色
GRIPPER_STATE_MID_MIN_KEEP = 24
# 已确认闭合后，close_frac 仍高于此值时保持橙色（容忍持物微抖，不闪灰）
GRIPPER_HOLD_ORANGE_FLOOR = 0.35
# pro2 等峰值常停在 ~0.7–0.8，低于旧 early=0.85
GRIPPER_EARLY_CLOSED_FRAC = 0.70
# 连续 N 帧 close>=thr：强制闭合着色（短平台被 lookback 窗口打散时的兜底）
GRIPPER_CONFIRM_CLOSED_FRAC = 0.45
GRIPPER_CONFIRM_CLOSED_FRAMES = 8

# ---------------------------------------------------------------------------
# humanoid_gripper.urdf 手臂关节名（运动学链顺序，与 EVT 一一对应）
# ---------------------------------------------------------------------------
HUMANOID_LEFT_ARM_JOINTS = [
    "left_joint1",              # ↔ shoulder_pitch_l_joint
    "shoulder_roll_l_joint",    # ↔ shoulder_roll_l_joint (同名)
    "left_joint3",              # ↔ shoulder_yaw_l_joint
    "elbow_l_joint",            # ↔ elbow_pitch_l_joint
    "left_joint5",              # ↔ elbow_yaw_l_joint
    "left_joint6",              # ↔ wrist_pitch_l_joint
    "left_joint7",              # ↔ wrist_roll_l_joint
]
HUMANOID_RIGHT_ARM_JOINTS = [
    "right_joint1",             # ↔ shoulder_pitch_r_joint
    "shoulder_roll_r_joint",    # ↔ shoulder_roll_r_joint (同名)
    "right_joint3",             # ↔ shoulder_yaw_r_joint
    "elbow_r_joint",            # ↔ elbow_pitch_r_joint
    "right_joint5",             # ↔ elbow_yaw_r_joint
    "right_joint6",             # ↔ wrist_pitch_r_joint
    "right_joint7",             # ↔ wrist_roll_r_joint
]

_HUMANOID_JOINT_DISPLAY: Dict[str, str] = {
    "left_joint1": "shldr_pitch_l",
    "left_joint3": "shldr_yaw_l",
    "elbow_l_joint": "elbow_pitch_l",
    "left_joint5": "elbow_yaw_l",
    "left_joint6": "wrist_pitch_l",
    "left_joint7": "wrist_roll_l",
    "right_joint1": "shldr_pitch_r",
    "right_joint3": "shldr_yaw_r",
    "elbow_r_joint": "elbow_pitch_r",
    "right_joint5": "elbow_yaw_r",
    "right_joint6": "wrist_pitch_r",
    "right_joint7": "wrist_roll_r",
}

_URDF_JOINT_DISPLAY: Dict[str, Dict[str, str]] = {
    "humanoid_gripper.urdf": _HUMANOID_JOINT_DISPLAY,
}

# URDF 文件名 → 手臂关节名的映射
_URDF_ARM_JOINTS: Dict[str, Tuple[List[str], List[str]]] = {
    "humanoid_gripper.urdf": (HUMANOID_LEFT_ARM_JOINTS, HUMANOID_RIGHT_ARM_JOINTS),
}

def _get_arm_joints_for_urdf(urdf_path: str) -> Tuple[List[str], List[str]]:
    """根据 URDF 文件路径返回左右臂关节名列表。"""
    basename = os.path.basename(urdf_path)
    if basename in _URDF_ARM_JOINTS:
        return _URDF_ARM_JOINTS[basename]
    return (LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS)

ROBOT_CONFIGS: Dict[str, dict] = {
    "tienkung_pro2_dualArm-gripper": {
        "gripper_threshold": 0.05,
        "closed_below": True,
        "head_pitch": 0.21,
        "cam_pitch_offset": 0.0,
    },
    "tienkung_station_dualArm-gripper": {
        "gripper_threshold": 0.1,
        "closed_below": True,
        "head_pitch": 0.45,
        "cam_pitch_offset": 0.0,
    },
    "tienkung_max_dualArm-gripper": {
        "gripper_threshold": 0.6,
        "closed_below": False,
        "head_pitch": 0.4,
        "cam_pitch_offset": 0.0,
    },
}
DEFAULT_ROBOT_CONFIG: dict = {
    "gripper_threshold": 0.05,
    "closed_below": True,
    "head_pitch": 0.21,
    "cam_pitch_offset": 0.0,
}

# ---------------------------------------------------------------------------
# Robot key → URDF 兜底映射
# 仅在 org_path 规则无法明确判断时使用，不是主选择逻辑。
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

ROBOT_URDF_MAP: Dict[str, str] = {
    "tienkung_pro2_dualArm-gripper": os.path.join(
        _SCRIPT_DIR, "final_tiangong2_EVT_with_grippers_2.urdf"),
    "tienkung_station_dualArm-gripper": os.path.join(
        _SCRIPT_DIR, "final_tiangong2_EVT_with_grippers.urdf"),
}
DEFAULT_URDF = os.path.join(_SCRIPT_DIR, "final_tiangong2_EVT_with_grippers.urdf")
PRO2_URDF_V2 = os.path.join(_SCRIPT_DIR, "final_tiangong2_EVT_with_grippers_2.urdf")
PRO2_URDF_V3 = os.path.join(_SCRIPT_DIR, "final_tiangong2_EVT_with_grippers_3.urdf")
PRO2_URDF_V4 = os.path.join(_SCRIPT_DIR, "final_tiangong2_EVT_with_grippers_4.urdf")
PRO2_URDF_V5 = os.path.join(_SCRIPT_DIR, "final_tiangong2_EVT_with_grippers_5.urdf")
STATION_URDF = os.path.join(_SCRIPT_DIR, "final_tiangong2_EVT_with_grippers.urdf")

_ORG_SEG_STATION_PREFIX = "tienkung_station_dualArm-gripper"
_ORG_SEG_STATION_3CAM_13 = "tienkung_station_dualArm-gripper-3cameras_13"
_ORG_SEG_PRO2_1CAM_1 = "tienkung_pro2_dualArm-gripper-1cameras_1"
_ORG_SEG_PRO2_1CAM_2 = "tienkung_pro2_dualArm-gripper-1cameras_2"
_ORG_SEG_PRO2_1CAM_6 = "tienkung_pro2_dualArm-gripper-1cameras_6"
_ORG_SEG_PRO2_PICK_BELT = "pick_up_parts_from_belt_conveyor_place_on_plate_fast"
_ORG_SEG_PRO2_PICK_PLACE = "pick_up_parts_place_on_plate"
# org_path 中匹配传送带抓取任务（含 belt_conveyor / f_coprom_belt 等变体）
_BELT_CONVEYOR_ORG_TOKENS = ("belt", "conveyor")
_FILTER_ALLOWED_LEVEL1_SEGMENTS = frozenset({
    _ORG_SEG_STATION_3CAM_13,
    _ORG_SEG_PRO2_1CAM_1,
    _ORG_SEG_PRO2_1CAM_2,
    _ORG_SEG_PRO2_1CAM_6,
})
_FILTER_SUBSAMPLE_LEVEL1_SEGMENTS = frozenset({
    _ORG_SEG_PRO2_1CAM_1,
    _ORG_SEG_PRO2_1CAM_6,
})
LEGACY4_LEVEL1_SEGMENTS = frozenset({
    _ORG_SEG_STATION_3CAM_13,
    _ORG_SEG_PRO2_1CAM_1,
    _ORG_SEG_PRO2_1CAM_2,
    _ORG_SEG_PRO2_1CAM_6,
})
_DEFAULT_SUBSAMPLE_1CAM16 = 100
_DEFAULT_FILTER_SEED = 42


def _is_belt_conveyor_task(item: dict) -> bool:
    """是否为传送带抓取任务（org_path 含 belt / conveyor）。"""
    org = _normalize_org_path(item.get("org_path")).lower()
    return any(tok in org for tok in _BELT_CONVEYOR_ORG_TOKENS)


def _select_render_items(
    items: Sequence[Any],
    selection_mode: str,
    subsample_1cam16: int,
    seed: Optional[int],
    exclude_belt_conveyor: bool = True,
) -> Tuple[List[Tuple[int, dict]], Dict[str, int]]:
    """Return deterministic (source_index, item) pairs for rendering."""
    if selection_mode == "legacy4-full":
        selected = []
        stats: Dict[str, int] = {
            "total": len(items), "missing_org_path": 0,
            "skipped_non_dict": 0, "excluded_other_level1": 0,
            "excluded_belt_conveyor": 0,
            "output_count": 0,
        }
        for source_index, item in enumerate(items):
            if not isinstance(item, dict):
                stats["skipped_non_dict"] += 1
                continue
            level1 = _get_level1_org_segment(item)
            if not level1:
                stats["missing_org_path"] += 1
                continue
            if level1 not in LEGACY4_LEVEL1_SEGMENTS:
                stats["excluded_other_level1"] += 1
                continue
            if exclude_belt_conveyor and _is_belt_conveyor_task(item):
                stats["excluded_belt_conveyor"] += 1
                continue
            selected.append((source_index, item))
            stats[level1] = stats.get(level1, 0) + 1
        stats["output_count"] = len(selected)
        return selected, stats

    filtered, stats = _filter_render_items(
        items, subsample_1cam16, seed,
        exclude_belt_conveyor=exclude_belt_conveyor)
    wanted = {id(item) for item in filtered}
    return [(i, item) for i, item in enumerate(items) if id(item) in wanted], stats


def _write_json(path: str, value: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def _write_jsonl(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _normalize_org_path(raw: object) -> str:
    return str(raw or "").replace("\\", "/").strip()


def _split_org_path(org_path: str) -> List[str]:
    parts = [p for p in org_path.strip("/").split("/") if p]
    if parts and parts[0] == "raw_data":
        parts = parts[1:]
    return parts


def _get_level1_org_segment(item: dict) -> str:
    parts = _split_org_path(_normalize_org_path(item.get("org_path")))
    return parts[0] if parts else ""


def _load_metadata(path: str) -> Any:
    if str(path).endswith(".gz"):
        with gzip.open(path, "rb") as f:
            return pickle.load(f)
    with open(path, "rb") as f:
        return pickle.load(f)


def _get_metadata_items(data: Any) -> Sequence[Any]:
    if isinstance(data, (list, tuple)):
        return data
    if isinstance(data, dict) and isinstance(data.get("data"), (list, tuple)):
        return data["data"]
    raise TypeError("metadata 必须是 list/tuple 或包含 'data' 字段的 dict")


def _rebuild_metadata(data: Any, items: Sequence[Any]) -> Any:
    if isinstance(data, list):
        return list(items)
    if isinstance(data, tuple):
        return tuple(items)
    if isinstance(data, dict) and isinstance(data.get("data"), (list, tuple)):
        output = dict(data)
        output["data"] = tuple(items) if isinstance(data["data"], tuple) else list(items)
        return output
    raise TypeError("metadata 必须是 list/tuple 或包含 'data' 字段的 dict")


def _filter_render_items(
    items: Sequence[Any],
    subsample_1cam16: int,
    seed: Optional[int],
    exclude_belt_conveyor: bool = True,
) -> Tuple[List[dict], Dict[str, int]]:
    """只保留目标一级目录，并对 1cam1/1cam6 合并后随机抽样。"""
    stats = {
        "total": len(items),
        "missing_org_path": 0,
        "skipped_non_dict": 0,
        "station_3cam13_kept": 0,
        "pro2_1cam1_candidates": 0,
        "pro2_1cam6_candidates": 0,
        "pro2_1cam16_sampled": 0,
        "pro2_1cam2_kept": 0,
        "excluded_other_level1": 0,
        "excluded_belt_conveyor": 0,
        "output_count": 0,
    }

    pool_1cam16: List[Tuple[int, dict]] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            stats["skipped_non_dict"] += 1
            continue
        if exclude_belt_conveyor and _is_belt_conveyor_task(item):
            stats["excluded_belt_conveyor"] += 1
            continue
        level1 = _get_level1_org_segment(item)
        if not level1:
            stats["missing_org_path"] += 1
            continue
        if level1 == _ORG_SEG_PRO2_1CAM_1:
            stats["pro2_1cam1_candidates"] += 1
            pool_1cam16.append((idx, item))
        elif level1 == _ORG_SEG_PRO2_1CAM_6:
            stats["pro2_1cam6_candidates"] += 1
            pool_1cam16.append((idx, item))

    rng = random.Random(seed)
    keep_k = min(max(0, subsample_1cam16), len(pool_1cam16))
    sampled_pairs = rng.sample(pool_1cam16, keep_k) if keep_k > 0 else []
    kept_1cam16_indices = {idx for idx, _ in sampled_pairs}
    stats["pro2_1cam16_sampled"] = len(kept_1cam16_indices)

    output: List[dict] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        if exclude_belt_conveyor and _is_belt_conveyor_task(item):
            continue
        level1 = _get_level1_org_segment(item)
        if not level1:
            continue
        if level1 == _ORG_SEG_STATION_3CAM_13:
            stats["station_3cam13_kept"] += 1
            output.append(item)
        elif level1 in _FILTER_SUBSAMPLE_LEVEL1_SEGMENTS:
            if idx in kept_1cam16_indices:
                output.append(item)
        elif level1 == _ORG_SEG_PRO2_1CAM_2:
            stats["pro2_1cam2_kept"] += 1
            output.append(item)
        else:
            stats["excluded_other_level1"] += 1

    stats["output_count"] = len(output)
    return output, stats


def _get_urdf_from_org_path(item: dict) -> Optional[str]:
    """根据 org_path 的目录层级优先选择 URDF。

    规则：
      1. 第一层目录以 tienkung_station_dualArm-gripper 开头 -> station URDF
      2. 第一层目录 = tienkung_pro2_dualArm-gripper-1cameras_2
         且第二层目录 = pick_up_parts_place_on_plate
         -> URDF _2
      3. 第一层目录 = tienkung_pro2_dualArm-gripper-1cameras_2
         - 若第二层目录包含 pick_up_parts_from_belt_conveyor_place_on_plate_fast -> URDF _3
         - 否则 -> URDF _2
      4. 第一层目录 = tienkung_pro2_dualArm-gripper-1cameras_1 / _6 -> URDF _3
      5. 其他情况返回 None，交由 robot 字段规则兜底
    """
    parts = _split_org_path(_normalize_org_path(item.get("org_path")))
    if not parts:
        return None

    level1 = parts[0]
    level2 = parts[1] if len(parts) > 1 else ""

    if level1.startswith(_ORG_SEG_STATION_PREFIX):
        return STATION_URDF
    if level1 == _ORG_SEG_PRO2_1CAM_2 and _ORG_SEG_PRO2_PICK_PLACE in level2:
            return PRO2_URDF_V5
    if level1 == _ORG_SEG_PRO2_1CAM_2:
        if _ORG_SEG_PRO2_PICK_BELT in level2:
            return PRO2_URDF_V3
        return PRO2_URDF_V2
    if level1 in (_ORG_SEG_PRO2_1CAM_1, _ORG_SEG_PRO2_1CAM_6):
        return PRO2_URDF_V4
    return None


def _get_fallback_urdf_from_robot_key(item: dict) -> str:
    """按解析出的 robot key 选择兜底 URDF。"""
    key = _resolve_robot_key(item)
    if key and key in ROBOT_URDF_MAP:
        return ROBOT_URDF_MAP[key]
    return DEFAULT_URDF


def _resolve_robot_key(item: dict) -> str:
    """从 item 中提取能匹配 ROBOT_CONFIGS / ROBOT_URDF_MAP 的 key。

    匹配优先级：
      1. robot 字段精确命中
      2. robot 字段子串匹配
      3. org_path 字段子串匹配（应对 robot='tiangong_dualArm' 等笼统值）
      4. 返回空串（调用方回退到默认值）
    """
    robot = str(item.get("robot", ""))
    all_keys = set(ROBOT_CONFIGS.keys()) | set(ROBOT_URDF_MAP.keys())

    if robot in all_keys:
        return robot
    for key in all_keys:
        if key in robot or robot in key:
            return key

    org_path = str(item.get("org_path", ""))
    if org_path:
        for key in all_keys:
            if key in org_path:
                return key

    return ""


def get_urdf_for_robot(item: dict, urdf_override: Optional[str] = None) -> str:
    """根据 item 信息选择 URDF 文件路径。

    选择优先级：
      1. 命令行 --urdf 强制指定
      2. org_path 分层规则
      3. robot / org_path 的通用 key 兜底映射
      4. 默认 URDF
    """
    if urdf_override is not None:
        return urdf_override
    org_path_urdf = _get_urdf_from_org_path(item)
    if org_path_urdf is not None:
        return org_path_urdf
    return _get_fallback_urdf_from_robot_key(item)


def get_robot_config(item: dict) -> dict:
    """根据 item 信息选择 robot 配置（含夹爪阈值 + 相机参数）。

    匹配策略：robot 字段精确 → 子串 → org_path 子串 → 默认配置。
    """
    key = _resolve_robot_key(item)
    if key and key in ROBOT_CONFIGS:
        return ROBOT_CONFIGS[key]
    return DEFAULT_ROBOT_CONFIG


def is_gripper_closed(value: float, config: dict) -> bool:
    """判断夹爪是否处于闭合状态（旧二值接口，保留兼容）。

    注意：渲染配色已改用时序状态机 + close_frac；此函数不再驱动绘制。
    """
    threshold = config.get("gripper_threshold", config.get("threshold", 0.05))
    if config.get("closed_below", True):
        return value < threshold
    return value > threshold


def _fill_short_false_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    """把长度 <= max_gap 的 False 段填成 True（抑制 settled 闪烁）。"""
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


def _suppress_short_true_runs(mask: np.ndarray, min_run: int) -> np.ndarray:
    """去掉长度 < min_run 的 True 段（短暂平台视为仍在运动）。"""
    out = np.asarray(mask, dtype=bool).copy()
    min_run = max(0, int(min_run))
    if min_run <= 1 or out.size == 0:
        return out
    t_len = int(out.shape[0])
    t = 0
    while t < t_len:
        if not out[t]:
            t += 1
            continue
        t0 = t
        while t < t_len and out[t]:
            t += 1
        if (t - t0) < min_run:
            out[t0:t] = False
    return out


def _median_smooth_1d(x: np.ndarray, k: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    k = int(k)
    if k <= 1 or x.size == 0:
        return x.copy()
    if k % 2 == 0:
        k += 1
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    out = np.empty_like(x)
    for i in range(int(x.shape[0])):
        out[i] = float(np.median(xp[i:i + k]))
    return out


def _demote_mid_settled_runs(
    settled: np.ndarray,
    plateau_val: np.ndarray,
    mid_lo: float,
    mid_hi: float,
    min_keep: int = GRIPPER_STATE_MID_MIN_KEEP,
) -> np.ndarray:
    """去掉短暂的模糊带平台（闭合途中闪橙）；高闭合/长持物平台保留。"""
    out = np.asarray(settled, dtype=bool).copy()
    t_len = int(out.shape[0])
    mid_lo = float(mid_lo)
    mid_hi = float(mid_hi)
    min_keep = max(0, int(min_keep))
    t = 0
    while t < t_len:
        if not out[t]:
            t += 1
            continue
        t0 = t
        while t < t_len and out[t]:
            t += 1
        med = float(np.median(plateau_val[t0:t]))
        dur = t - t0
        if mid_lo < med < mid_hi and dur < min_keep:
            out[t0:t] = False
    return out


def _sustained_high_close_mask(
    close_frac: np.ndarray,
    thr: float,
    min_frames: int,
) -> np.ndarray:
    """连续 >= min_frames 帧 close_frac>=thr 的区段掩码。"""
    ov = np.asarray(close_frac, dtype=np.float64)
    t_len = int(ov.shape[0])
    out = np.zeros(t_len, dtype=bool)
    min_frames = max(1, int(min_frames))
    thr = float(thr)
    t = 0
    while t < t_len:
        if ov[t] < thr:
            t += 1
            continue
        t0 = t
        while t < t_len and ov[t] >= thr:
            t += 1
        if (t - t0) >= min_frames:
            out[t0:t] = True
    return out


def gripper_state_masks(
    close_frac_side: np.ndarray,
    closed_open_frac: float = GRIPPER_CLOSED_OPEN_FRAC,
    lookahead: int = GRIPPER_STATE_LOOKAHEAD,
    eps: float = GRIPPER_STATE_PLATEAU_EPS,
    change_delta: float = GRIPPER_STATE_CHANGE_DELTA,
    lookback: int = GRIPPER_STATE_LOOKBACK,
    settled_fill: int = GRIPPER_STATE_SETTLED_FILL,
    early_closed_frac: float = GRIPPER_EARLY_CLOSED_FRAC,
    min_settled_run: int = GRIPPER_STATE_MIN_SETTLED_RUN,
    smooth_k: int = GRIPPER_STATE_SMOOTH_K,
    mid_lo: float = GRIPPER_STATE_MID_LO,
    mid_hi: float = GRIPPER_STATE_MID_HI,
    mid_min_keep: int = GRIPPER_STATE_MID_MIN_KEEP,
    hold_orange_floor: float = GRIPPER_HOLD_ORANGE_FLOOR,
    confirm_closed_frac: float = GRIPPER_CONFIRM_CLOSED_FRAC,
    confirm_closed_frames: int = GRIPPER_CONFIRM_CLOSED_FRAMES,
) -> Tuple[np.ndarray, np.ndarray]:
    """逐侧时序状态机，返回 (closed_mask, settled_mask)。

    close_frac_side: (T,) HIGH=物理闭合。
    settled=True 用开合二值色；settled=False（运动）骨线灰、关节红。
    夹住物体时 close_frac 常停在 ~0.5：相对翻转 + 长 mid plateau 保留；
    pro2 峰值常 ~0.7–0.8：early-closed + 持续高 close 确认，避免短闭合不变橙。
    """
    ov_raw = np.clip(np.asarray(close_frac_side, dtype=np.float64), 0.0, 1.0)
    t_len = int(ov_raw.shape[0])
    if t_len == 0:
        empty = np.zeros(0, dtype=bool)
        return empty, empty
    ov = _median_smooth_1d(ov_raw, smooth_k)
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

    # 先去掉短暂 settled 岛，再填补短 unsettled；仅打掉模糊带短平台
    settled = _suppress_short_true_runs(settled, min_settled_run)
    settled = _fill_short_false_gaps(settled, settled_fill)
    settled = _demote_mid_settled_runs(
        settled, plateau_val, mid_lo, mid_hi, min_keep=mid_min_keep)
    settled = _suppress_short_true_runs(settled, min_settled_run)
    settled = _fill_short_false_gaps(settled, max(1, settled_fill // 2))

    settled_idx = np.flatnonzero(settled)
    if settled_idx.size == 0:
        closed = ov >= closed_open_frac
        settled = np.ones(t_len, dtype=bool)
    else:
        # 仅在确认翻转时更新 ref。相对比较：从张开到 ~0.5 持物平台算闭合。
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

    # 持续高 close：强制闭合+着色（短高平台被 lookback 打散时的兜底）
    if confirm_closed_frames > 0 and confirm_closed_frac > 0.0:
        force = _sustained_high_close_mask(
            ov_raw, confirm_closed_frac, confirm_closed_frames)
        closed = closed | force
        settled = settled | force

    # 已闭合且 close 仍偏高：保持二值橙色，避免持物微抖闪灰
    hold_floor = float(hold_orange_floor)
    if hold_floor > 0.0:
        settled = settled | (closed & (ov_raw >= hold_floor))

    return closed, settled


def _gripper_bone_colors_from_mask(
    closed_mask: np.ndarray,
    color_open: tuple,
    color_closed: tuple,
) -> np.ndarray:
    co = np.asarray(color_open, dtype=np.int64)
    cc = np.asarray(color_closed, dtype=np.int64)
    return np.where(np.asarray(closed_mask, dtype=bool)[:, None], cc[None, :], co[None, :])


def gripper_colors_state_or_motion(
    closed_mask: np.ndarray,
    settled_mask: np.ndarray,
    color_open: tuple,
    color_closed: tuple,
    color_motion: tuple,
) -> np.ndarray:
    """Settled 二值开合色；运动帧统一 motion 色（RealSource：骨线灰 / 关节红）。"""
    binary = _gripper_bone_colors_from_mask(closed_mask, color_open, color_closed)
    settled = np.asarray(settled_mask, dtype=bool)
    motion = np.asarray(color_motion, dtype=np.int64)
    return np.where(settled[:, None], binary, motion[None, :]).astype(np.int64)


def gripper_close_frac_series(
    gripper_all: np.ndarray,
    robot_config: Optional[dict] = None,
) -> np.ndarray:
    """将 (T,2) 夹爪原始值映射为 close_frac（HIGH=物理闭合），供状态机使用。

    旧版 closed_below=True 把低值画成闭合色，与 AG95 视觉（低≈张开）相反。
    这里对 closed_below=True 取 close_frac = norm(g)（高=闭）；
    closed_below=False 取 close_frac = 1-norm(g)。
    """
    if robot_config is None:
        robot_config = DEFAULT_ROBOT_CONFIG
    g = np.asarray(gripper_all, dtype=np.float64)
    lo, hi = GRIPPER_JOINT_LIMIT
    span = max(float(hi - lo), 1e-6)
    norm = np.clip((g - lo) / span, 0.0, 1.0)
    if robot_config.get("closed_below", True):
        # 翻转旧二值配色极性：低开高闭
        return norm
    return 1.0 - norm


def precompute_gripper_draw_colors(
    gripper_all: np.ndarray,
    robot_config: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """返回 (bone_l, bone_r, joint_l, joint_r)，各 (T,3) RGB int。"""
    close = gripper_close_frac_series(gripper_all, robot_config)
    swap_colors = os.environ.get("AC_ROBOMIND_SWAP_COLORS") == "1"
    open_bone, closed_bone = (COLOR_GRIPPER_CLOSED_BONE, COLOR_GRIPPER_OPEN_BONE) if swap_colors else (COLOR_GRIPPER_OPEN_BONE, COLOR_GRIPPER_CLOSED_BONE)
    open_joint, closed_joint = (COLOR_GRIPPER_CLOSED_JOINT, COLOR_GRIPPER_OPEN_JOINT) if swap_colors else (COLOR_GRIPPER_OPEN_JOINT, COLOR_GRIPPER_CLOSED_JOINT)
    swap_colors = os.environ.get("AC_ROBOMIND_SWAP_COLORS") == "1"
    open_bone, closed_bone = (COLOR_GRIPPER_CLOSED_BONE, COLOR_GRIPPER_OPEN_BONE) if swap_colors else (COLOR_GRIPPER_OPEN_BONE, COLOR_GRIPPER_CLOSED_BONE)
    open_joint, closed_joint = (COLOR_GRIPPER_CLOSED_JOINT, COLOR_GRIPPER_OPEN_JOINT) if swap_colors else (COLOR_GRIPPER_OPEN_JOINT, COLOR_GRIPPER_CLOSED_JOINT)
    swap_colors = os.environ.get("AC_ROBOMIND_SWAP_COLORS") == "1"
    open_bone, closed_bone = (COLOR_GRIPPER_CLOSED_BONE, COLOR_GRIPPER_OPEN_BONE) if swap_colors else (COLOR_GRIPPER_OPEN_BONE, COLOR_GRIPPER_CLOSED_BONE)
    open_joint, closed_joint = (COLOR_GRIPPER_CLOSED_JOINT, COLOR_GRIPPER_OPEN_JOINT) if swap_colors else (COLOR_GRIPPER_OPEN_JOINT, COLOR_GRIPPER_CLOSED_JOINT)
    swap_colors = os.environ.get("AC_ROBOMIND_SWAP_COLORS") == "1"
    open_bone, closed_bone = (COLOR_GRIPPER_CLOSED_BONE, COLOR_GRIPPER_OPEN_BONE) if swap_colors else (COLOR_GRIPPER_OPEN_BONE, COLOR_GRIPPER_CLOSED_BONE)
    open_joint, closed_joint = (COLOR_GRIPPER_CLOSED_JOINT, COLOR_GRIPPER_OPEN_JOINT) if swap_colors else (COLOR_GRIPPER_OPEN_JOINT, COLOR_GRIPPER_CLOSED_JOINT)
    swap_colors = os.environ.get("AC_ROBOMIND_SWAP_COLORS") == "1"
    open_bone, closed_bone = (COLOR_GRIPPER_CLOSED_BONE, COLOR_GRIPPER_OPEN_BONE) if swap_colors else (COLOR_GRIPPER_OPEN_BONE, COLOR_GRIPPER_CLOSED_BONE)
    open_joint, closed_joint = (COLOR_GRIPPER_CLOSED_JOINT, COLOR_GRIPPER_OPEN_JOINT) if swap_colors else (COLOR_GRIPPER_OPEN_JOINT, COLOR_GRIPPER_CLOSED_JOINT)
    closed_l, settled_l = gripper_state_masks(close[:, 0])
    closed_r, settled_r = gripper_state_masks(close[:, 1])
    bone_l = gripper_colors_state_or_motion(
        closed_l, settled_l,
        open_bone, closed_bone, COLOR_GRIPPER_MOTION_BONE)
    bone_r = gripper_colors_state_or_motion(
        closed_r, settled_r,
        open_bone, closed_bone, COLOR_GRIPPER_MOTION_BONE)
    joint_l = gripper_colors_state_or_motion(
        closed_l, settled_l,
        open_joint, closed_joint, COLOR_GRIPPER_MOTION_JOINT)
    joint_r = gripper_colors_state_or_motion(
        closed_r, settled_r,
        open_joint, closed_joint, COLOR_GRIPPER_MOTION_JOINT)
    return bone_l, bone_r, joint_l, joint_r


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
# 左臂明显运动但左夹爪信号几乎不变 → 疑似左夹爪录制坏数据
LEFT_ARM_MOTION_PTP_MIN = 0.15       # rad，左臂 7 关节 peak-to-peak 最大值
LEFT_GRIPPER_STATIC_MAX = 0.02       # 左夹爪原始值全程变化幅度上限


def left_arm_moves_gripper_static(
    arm: np.ndarray,
    gripper: np.ndarray,
    arm_ptp_min: float = LEFT_ARM_MOTION_PTP_MIN,
    gripper_static_max: float = LEFT_GRIPPER_STATIC_MAX,
) -> Tuple[bool, Dict[str, float]]:
    """左臂运动但左夹爪几乎不动时返回 (True, stats)。"""
    arm = np.asarray(arm, dtype=np.float64)
    gripper = np.asarray(gripper, dtype=np.float64)
    if arm.ndim != 2 or arm.shape[1] < 7 or gripper.ndim != 2 or gripper.shape[1] < 1:
        return False, {}
    if arm.shape[0] < 2:
        return False, {}
    left_arm = arm[:, :7]
    left_g = gripper[:, 0]
    arm_ptp = float(np.max(np.ptp(left_arm, axis=0)))
    grip_span = float(np.ptp(left_g))
    stats = {
        "left_arm_ptp_max": arm_ptp,
        "left_gripper_span": grip_span,
        "arm_ptp_min": float(arm_ptp_min),
        "gripper_static_max": float(gripper_static_max),
    }
    bad = (arm_ptp >= float(arm_ptp_min)) and (grip_span <= float(gripper_static_max))
    return bad, stats


def load_action_arrays_gripper(item: dict) -> Tuple[np.ndarray, np.ndarray]:
    """从一条 metadata entry 加载 arm (T,14) 和 gripper (T,2)。

    夹爪数据优先从 hand_joints_left + hand_joints_right 加载；
    若不存在则尝试 left + right 列。
    """
    import pandas as pd

    root = item.get("root", "")
    action_info = item.get("action", {})
    all_fields_rel = action_info.get("all_fields", "")
    if not all_fields_rel:
        raise ValueError("该条数据没有 action['all_fields'] parquet 路径")
    path = os.path.join(root, all_fields_rel) if not all_fields_rel.startswith("/") else all_fields_rel
    if not os.path.exists(path):
        raise FileNotFoundError(f"Action parquet 文件不存在: {path}")

    df = pd.read_parquet(path)
    if "arm_joints_robot" not in df.columns:
        raise ValueError(f"Parquet 缺少 arm_joints_robot。columns={list(df.columns)}")
    arm = np.stack(df["arm_joints_robot"].values).astype(np.float64)

    left_gripper = None
    right_gripper = None

    if "hand_joints_left" in df.columns and "hand_joints_right" in df.columns:
        left_gripper = np.stack(df["hand_joints_left"].values).astype(np.float64)
        right_gripper = np.stack(df["hand_joints_right"].values).astype(np.float64)
    elif "left" in df.columns and "right" in df.columns:
        left_gripper = np.stack(df["left"].values).astype(np.float64)
        right_gripper = np.stack(df["right"].values).astype(np.float64)
    else:
        raise ValueError(f"Parquet 缺少夹爪列。columns={list(df.columns)}")

    if left_gripper.ndim == 1:
        left_gripper = left_gripper.reshape(-1, 1)
    if right_gripper.ndim == 1:
        right_gripper = right_gripper.reshape(-1, 1)

    gripper = np.concatenate([left_gripper[:, :1], right_gripper[:, :1]], axis=1)
    return arm, gripper


def _resolve_video_path(root: str, rel_or_abs: str) -> str:
    """将相对/绝对路径统一解析为完整路径（与 MultiViewVideoDataset 对齐）。"""
    if not rel_or_abs:
        return ""
    rel_or_abs = str(rel_or_abs)
    if rel_or_abs.startswith("/"):
        return rel_or_abs
    return os.path.join(root, rel_or_abs)


def get_head_view_video_path(item: dict) -> Tuple[str, str]:
    """获取 head 视角视频路径。

    选择策略（与 multi_view_utils.select_primary_view 对齐，但强制 head 优先）：
      1. 精确匹配 head 相关视角名
      2. 回退到 select_primary_view 主视角逻辑
      3. 回退到 action[images_camera] / action[images_*] 路径
      4. 兜底第一个可用视频

    返回: (video_path, selected_view_name)
    """
    root = item.get("root", "")
    cvi = item.get("camera_video_info", {})

    # ---------- 从 camera_video_info 中筛选有效视角 ----------
    valid_views: Dict[str, str] = {}
    if isinstance(cvi, dict):
        for vn, vi in cvi.items():
            if isinstance(vi, dict):
                vp = vi.get("video", "")
                if vp:
                    valid_views[str(vn)] = _resolve_video_path(root, vp)

    if valid_views:
        view_names = list(valid_views.keys())

        # 策略 1：精确匹配 head 视角（不做子串匹配，避免误命中）
        HEAD_EXACT_NAMES = ["camera_head", "ob_camera_head", "head"]
        for target in HEAD_EXACT_NAMES:
            for vn in view_names:
                if vn.lower() == target:
                    return valid_views[vn], vn

        # 策略 2：单相机且名为 "camera"（tienkung gripper 常见情况）
        if len(view_names) == 1 and view_names[0].lower() == "camera":
            vn = view_names[0]
            return valid_views[vn], vn

        # 策略 3：复用 multi_view_utils.select_primary_view
        try:
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if repo_root not in sys.path:
                sys.path.insert(0, repo_root)
            from view_selection import select_primary_view
            selected = select_primary_view(view_names)
            if selected:
                vn = selected[0]
                if vn in valid_views:
                    return valid_views[vn], vn
        except Exception:
            pass

        # 策略 4：返回第一个有效视角
        vn = view_names[0]
        return valid_views[vn], vn

    # ---------- camera_video_info 无效时，从 action dict 解析 ----------
    action_info = item.get("action", {})
    if isinstance(action_info, dict):
        for ak, av in action_info.items():
            if ak.startswith("images_") and isinstance(av, str) and av:
                path = _resolve_video_path(root, av)
                view_name = ak.replace("images_", "")
                return path, view_name

    raise ValueError("该条数据没有可用的视频路径")


# ---------------------------------------------------------------------------
# Worker 全局状态
# ---------------------------------------------------------------------------
_worker_robot = None
_worker_robot_cache: Dict[str, object] = {}
_worker_kin_cache: Dict[str, dict] = {}
_worker_current_urdf: Optional[str] = None
_worker_kin: Optional[dict] = None
_worker_left_arm_joints: List[str] = list(LEFT_ARM_JOINTS)
_worker_right_arm_joints: List[str] = list(RIGHT_ARM_JOINTS)
_worker_action_joint_set: frozenset = frozenset()
_worker_view_matrix = None
_worker_fx = None
_worker_fy = None
_worker_cx = None
_worker_cy = None
_worker_last_cam_key = None
_worker_img_w = None
_worker_img_h = None
_worker_white_frame: Optional[np.ndarray] = None
ACTIVE_BONE_WIDTH = 4
ACTIVE_JOINT_RADIUS = 6


def _update_arm_joint_vars(urdf_path: str):
    """根据当前 URDF 更新 worker 全局的关节名变量。"""
    global _worker_left_arm_joints, _worker_right_arm_joints, _worker_action_joint_set
    left, right = _get_arm_joints_for_urdf(urdf_path)
    _worker_left_arm_joints = left
    _worker_right_arm_joints = right
    _worker_action_joint_set = frozenset(left + right + ALL_GRIPPER_JOINTS)


def _build_kinematics(robot) -> dict:
    """从 yourdfpy robot 提取静态运动学树，供整段 episode 批量 FK。"""
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

    return {
        "joints": joints,
        "roots": roots,
        "topo": topo,
        "link_parent_joint": link_parent_joint,
        "mimic_joints": [n for n, info in joints.items() if info["mimic"] is not None],
    }


def _load_urdf_cached(urdf_path: str):
    """加载 URDF 并缓存，避免重复解析。切换时同步更新关节名映射。"""
    global _worker_robot, _worker_current_urdf, _worker_last_cam_key, _worker_kin

    if _worker_current_urdf == urdf_path:
        return

    if urdf_path in _worker_robot_cache:
        _worker_robot = _worker_robot_cache[urdf_path]
        _worker_kin = _worker_kin_cache[urdf_path]
        _worker_current_urdf = urdf_path
        _worker_last_cam_key = None
        _update_arm_joint_vars(urdf_path)
        return

    import warnings
    import logging
    from yourdfpy import URDF

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        old_level = logging.root.level
        logging.root.setLevel(logging.ERROR)
        robot = URDF.load(urdf_path, load_meshes=False, build_scene_graph=True)
        logging.root.setLevel(old_level)

    kin = _build_kinematics(robot)
    _worker_robot_cache[urdf_path] = robot
    _worker_kin_cache[urdf_path] = kin
    _worker_robot = robot
    _worker_kin = kin
    _worker_current_urdf = urdf_path
    _worker_last_cam_key = None
    _update_arm_joint_vars(urdf_path)
    print(f"[Worker {os.getpid()}] URDF loaded: {os.path.basename(urdf_path)}")


def _init_worker(urdf_paths: List[str], img_w: int, img_h: int,
                 hfov_deg: float, vfov_deg: float):
    """每个 worker 进程启动时调用：预加载所有 URDF，设置相机内参。

    相机内参基于 Orbbec Gemini 336 彩色相机规格：
      - 水平 FOV 86°, 垂直 FOV 55°, 原生分辨率 1920×1080
    fx 和 fy 分别由水平/垂直 FOV 独立计算。

    相机外参（head_pitch / cam_pitch_offset）由每个任务动态设置，
    通过 _setup_camera() 在 render_one() 中调用。
    """
    global _worker_fx, _worker_fy, _worker_cx, _worker_cy
    global _worker_img_w, _worker_img_h, _worker_white_frame

    import warnings
    import logging
    warnings.filterwarnings("ignore", message=".*mimic.*")
    logging.getLogger("yourdfpy").setLevel(logging.ERROR)

    for p in urdf_paths:
        _load_urdf_cached(p)

    _worker_img_w = img_w
    _worker_img_h = img_h

    hfov_rad = np.deg2rad(hfov_deg)
    vfov_rad = np.deg2rad(vfov_deg)
    _worker_fx = img_w / (2.0 * np.tan(hfov_rad / 2.0))
    _worker_fy = img_h / (2.0 * np.tan(vfov_rad / 2.0))
    _worker_cx = img_w / 2.0
    _worker_cy = img_h / 2.0
    _worker_white_frame = np.full((img_h, img_w, 3), 255, dtype=np.uint8)
    try:
        cv2.setNumThreads(0)
    except Exception:
        pass

    print(f"[Worker {os.getpid()}] {len(urdf_paths)} URDF(s) loaded, "
          f"fx={_worker_fx:.1f}, fy={_worker_fy:.1f}, "
          f"hfov={hfov_deg}°, vfov={vfov_deg}°, "
          f"output={img_w}x{img_h}")


def _setup_camera(head_pitch_rad: float, cam_pitch_offset_deg: float):
    """设置（或更新）头部相机外参。跳过重复调用以减少开销。

    head_pitch_rad: 数据采集时机器人头部俯仰角 (rad)，正值=向下看。
    cam_pitch_offset_deg: 额外微调的光轴俯角 (度)，正值=进一步向下。
    """
    global _worker_view_matrix, _worker_last_cam_key

    cam_key = (_worker_current_urdf, round(head_pitch_rad, 6), round(cam_pitch_offset_deg, 4))
    if _worker_last_cam_key == cam_key:
        return
    _worker_last_cam_key = cam_key

    _worker_robot.update_cfg({"head_pitch_joint": head_pitch_rad})
    cam_T = _as_transform(_worker_robot.get_transform("camera_head_link"))

    T_optical = cam_T.copy()
    T_optical[:3, 0] = -cam_T[:3, 1]
    T_optical[:3, 1] = -cam_T[:3, 2]
    T_optical[:3, 2] = cam_T[:3, 0]

    if abs(cam_pitch_offset_deg) > 1e-6:
        alpha = np.deg2rad(cam_pitch_offset_deg)
        R_pitch = np.eye(4)
        R_pitch[1, 1] = np.cos(alpha)
        R_pitch[1, 2] = -np.sin(alpha)
        R_pitch[2, 1] = np.sin(alpha)
        R_pitch[2, 2] = np.cos(alpha)
        rot_only = np.eye(4)
        rot_only[:3, :3] = T_optical[:3, :3]
        rot_only = rot_only @ R_pitch
        T_optical[:3, :3] = rot_only[:3, :3]

    _worker_view_matrix = np.linalg.inv(T_optical)

    cam_pos = cam_T[:3, 3]
    optical_z = T_optical[:3, 2]
    total_pitch = np.rad2deg(np.arcsin(-optical_z[2] / np.linalg.norm(optical_z)))
    print(f"[Worker {os.getpid()}] camera setup: pos={cam_pos.round(4)}, "
          f"optical_axis(Z)={optical_z.round(4)}, "
          f"total_pitch_down={total_pitch:.1f}°, "
          f"head_pitch={np.rad2deg(head_pitch_rad):.1f}°, "
          f"cam_pitch_offset={cam_pitch_offset_deg:.1f}°")


def _clip(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def _action_to_joint_cfg_gripper(arm: np.ndarray, gripper: np.ndarray) -> Dict[str, float]:
    """将单帧 action 转为 URDF cfg (arm 14-dim + gripper 2-dim)。

    使用 worker 全局的 _worker_left/right_arm_joints（随 URDF 切换自动更新）。
    """
    assert arm.shape == (14,), f"Expected arm shape (14,), got {arm.shape}"
    assert gripper.shape == (2,), f"Expected gripper shape (2,), got {gripper.shape}"

    cfg: Dict[str, float] = {}

    for i, jn in enumerate(_worker_left_arm_joints):
        cfg[jn] = float(arm[i])
    for i, jn in enumerate(_worker_right_arm_joints):
        cfg[jn] = float(arm[7 + i])

    left_val = _clip(float(gripper[0]), *GRIPPER_JOINT_LIMIT)
    right_val = _clip(float(gripper[1]), *GRIPPER_JOINT_LIMIT)

    cfg[LEFT_GRIPPER_DRIVING_JOINT] = left_val
    for jn, mult in LEFT_GRIPPER_MIMIC_JOINTS.items():
        cfg[jn] = left_val * mult

    cfg[RIGHT_GRIPPER_DRIVING_JOINT] = right_val
    for jn, mult in RIGHT_GRIPPER_MIMIC_JOINTS.items():
        cfg[jn] = right_val * mult

    return cfg


# ---------------------------------------------------------------------------
# 透视投影
# ---------------------------------------------------------------------------
def _project(pt_world: np.ndarray) -> Optional[Tuple[int, int]]:
    """将 3D 世界坐标点透视投影到头部相机的 2D 像素坐标。

    光学坐标系：Z 前方（光轴），X 右，Y 下。
    """
    p_cam = _worker_view_matrix @ np.array([pt_world[0], pt_world[1], pt_world[2], 1.0])
    z = p_cam[2]
    if z < 0.01:
        return None
    u = int(round(_worker_fx * p_cam[0] / z + _worker_cx))
    v = int(round(_worker_fy * p_cam[1] / z + _worker_cy))
    return (u, v)


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


def _action_cfg_batch(
    arm: np.ndarray,
    gripper: np.ndarray,
    head_pitch: float,
) -> Dict[str, np.ndarray]:
    """整段 episode 的关节 cfg：joint_name -> (T,)。"""
    t = int(arm.shape[0])
    cfg: Dict[str, np.ndarray] = {}
    for i, jn in enumerate(_worker_left_arm_joints):
        cfg[jn] = arm[:, i].astype(np.float64)
    for i, jn in enumerate(_worker_right_arm_joints):
        cfg[jn] = arm[:, 7 + i].astype(np.float64)

    lo, hi = GRIPPER_JOINT_LIMIT
    left = np.clip(gripper[:, 0].astype(np.float64), lo, hi)
    right = np.clip(gripper[:, 1].astype(np.float64), lo, hi)
    cfg[LEFT_GRIPPER_DRIVING_JOINT] = left
    for jn, mult in LEFT_GRIPPER_MIMIC_JOINTS.items():
        cfg[jn] = left * float(mult)
    cfg[RIGHT_GRIPPER_DRIVING_JOINT] = right
    for jn, mult in RIGHT_GRIPPER_MIMIC_JOINTS.items():
        cfg[jn] = right * float(mult)

    cfg["head_pitch_joint"] = np.full(t, float(head_pitch), dtype=np.float64)
    return cfg


def _fk_world_transforms_batch(
    cfg_arrays: Dict[str, np.ndarray], t: int,
) -> Dict[str, np.ndarray]:
    """批量 FK：每个 link 的世界系变换 (T,4,4)。对齐 yourdfpy mimic 语义。"""
    kin = _worker_kin
    assert kin is not None
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
        parent = info["parent"]
        if parent is None or parent not in world:
            world[link] = local
        else:
            world[link] = world[parent] @ local
    return world


def _project_batch(
    pts: np.ndarray,
    view: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """点轨迹 (T,3) -> (u, v, valid)，与 `_project` 一致。"""
    if view is None:
        view = _worker_view_matrix
    assert view is not None
    t = pts.shape[0]
    ph = np.concatenate([pts, np.ones((t, 1), dtype=np.float64)], axis=1)
    if view.ndim == 2:
        p = (view @ ph.T).T
    else:
        p = np.einsum("tij,tj->ti", view, ph)
    z = p[:, 2]
    valid = z >= 0.01
    zz = np.where(valid, z, 1.0)
    u = np.round(_worker_fx * p[:, 0] / zz + _worker_cx).astype(np.int64)
    v = np.round(_worker_fy * p[:, 1] / zz + _worker_cy).astype(np.int64)
    return u, v, valid


def _precompute_skeleton_draw_vectorized(
    world: Dict[str, np.ndarray],
    show_inactive: bool,
    bone_col_l: np.ndarray,
    bone_col_r: np.ndarray,
    joint_col_l: np.ndarray,
    joint_col_r: np.ndarray,
) -> List[Tuple[Any, ...]]:
    """预计算整段 episode 的骨线/关节点像素轨迹，供逐帧 cv2 绘制。"""
    kin = _worker_kin
    assert kin is not None
    view = _worker_view_matrix
    assert view is not None

    joint_points: Dict[str, np.ndarray] = {}
    for jname, info in kin["joints"].items():
        if info["type"] == "fixed":
            continue
        parent = info["parent"]
        if parent is None or parent not in world:
            continue
        j_t = world[parent] @ info["origin"][None]
        joint_points[jname] = j_t[:, :3, 3]

    link_parent_joint = kin["link_parent_joint"]

    def link_anchor_traj(link_name: str) -> Optional[np.ndarray]:
        inc = link_parent_joint.get(link_name)
        if inc is not None and inc in joint_points:
            return joint_points[inc]
        if link_name in world:
            return world[link_name][:, :3, 3]
        return None

    draws: List[Tuple[Any, ...]] = []
    for jname, pts in joint_points.items():
        is_active = jname in _worker_action_joint_set
        if not is_active and not show_inactive:
            continue
        parent = kin["joints"][jname]["parent"]
        if parent is None:
            continue
        p1 = link_anchor_traj(parent)
        if p1 is None:
            continue
        u1, v1, val1 = _project_batch(p1, view)
        u2, v2, val2 = _project_batch(pts, view)
        valid = val1 & val2
        if jname in LEFT_GRIPPER_JOINT_SET:
            draws.append((u1, v1, u2, v2, valid, bone_col_l, joint_col_l, "grip"))
        elif jname in RIGHT_GRIPPER_JOINT_SET:
            draws.append((u1, v1, u2, v2, valid, bone_col_r, joint_col_r, "grip"))
        elif is_active:
            draws.append((u1, v1, u2, v2, valid, None, None, "arm"))
        else:
            draws.append((u1, v1, u2, v2, valid, None, None, "inactive"))
    return draws


def _compose_skeleton_frame(
    draws: List[Tuple[Any, ...]],
    t: int,
    transparent: bool = False,
) -> np.ndarray:
    """用预计算轨迹画一帧 RGB 或 RGBA 骨骼图。"""
    img_w, img_h = _worker_img_w, _worker_img_h
    if transparent:
        img = np.zeros((img_h, img_w, 4), dtype=np.uint8)
        arm_bone = COLOR_ARM_BONE + (255,)
        arm_joint = COLOR_ARM_JOINT + (255,)
        inactive = COLOR_INACTIVE + (180,)
    else:
        if os.environ.get("AC_BLACK_BACKGROUND") == "1":
            img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        elif _worker_white_frame is not None:
            img = _worker_white_frame.copy()
        else:
            img = np.full((img_h, img_w, 3), 255, dtype=np.uint8)
        arm_bone = COLOR_ARM_BONE
        arm_joint = COLOR_ARM_JOINT
        inactive = COLOR_INACTIVE

    for u1, v1, u2, v2, valid, bcol, jcol, kind in draws:
        if not valid[t]:
            continue
        pt1 = (int(u1[t]), int(v1[t]))
        pt2 = (int(u2[t]), int(v2[t]))
        if kind == "arm":
            cv2.line(img, pt1, pt2, arm_bone, ACTIVE_BONE_WIDTH, cv2.LINE_AA)
            cv2.circle(img, pt2, ACTIVE_JOINT_RADIUS, arm_joint, -1, cv2.LINE_AA)
        elif kind == "grip":
            bone = (int(bcol[t, 0]), int(bcol[t, 1]), int(bcol[t, 2]))
            joint = (int(jcol[t, 0]), int(jcol[t, 1]), int(jcol[t, 2]))
            if transparent:
                bone = bone + (255,)
                joint = joint + (255,)
            cv2.line(img, pt1, pt2, bone, ACTIVE_BONE_WIDTH, cv2.LINE_AA)
            cv2.circle(img, pt2, ACTIVE_JOINT_RADIUS, joint, -1, cv2.LINE_AA)
        else:
            cv2.line(img, pt1, pt2, inactive, 2, cv2.LINE_AA)
            cv2.circle(img, pt2, 3, inactive, -1, cv2.LINE_AA)
    return img


def _build_vectorized_draw_lists(
    arm_all: np.ndarray,
    gripper_all: np.ndarray,
    robot_config: dict,
    show_inactive: bool,
    bone_col_l: np.ndarray,
    bone_col_r: np.ndarray,
    joint_col_l: np.ndarray,
    joint_col_r: np.ndarray,
) -> List[Tuple[Any, ...]]:
    """一次批量 FK + 投影，返回可复用的绘制列表。"""
    t = int(arm_all.shape[0])
    cfg = _action_cfg_batch(arm_all, gripper_all, float(robot_config["head_pitch"]))
    world = _fk_world_transforms_batch(cfg, t)
    return _precompute_skeleton_draw_vectorized(
        world, show_inactive, bone_col_l, bone_col_r, joint_col_l, joint_col_r)


# ---------------------------------------------------------------------------
# 单帧绘制
# ---------------------------------------------------------------------------
def _draw_frame(arm_t: np.ndarray, gripper_t: np.ndarray,
                show_inactive: bool = False,
                robot_config: Optional[dict] = None,
                transparent: bool = False,
                left_grip_bone: Optional[Sequence[int]] = None,
                right_grip_bone: Optional[Sequence[int]] = None,
                left_grip_joint: Optional[Sequence[int]] = None,
                right_grip_joint: Optional[Sequence[int]] = None) -> np.ndarray:
    """从头部相机视角绘制一帧。

    transparent=False: 返回 RGB (H, W, 3)，白色背景。
    transparent=True:  返回 RGBA (H, W, 4)，透明背景，用于 overlay 叠加。
    夹爪左右骨线/关节色由调用方按 RealSource 状态机预计算后传入。
    """
    if robot_config is None:
        robot_config = DEFAULT_ROBOT_CONFIG
    robot = _worker_robot
    img_w, img_h = _worker_img_w, _worker_img_h

    cfg = _action_to_joint_cfg_gripper(arm_t, gripper_t)
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
    else:
        bg_value = 0 if os.environ.get("AC_BLACK_BACKGROUND") == "1" else 255
        img = np.full((img_h, img_w, 3), bg_value, dtype=np.uint8)

    def _as_bgr_color(c: Sequence[int], a: Optional[int] = None):
        """numpy/list → OpenCV 可用的纯 Python int 元组。"""
        rgb = (int(c[0]), int(c[1]), int(c[2]))
        return rgb + (int(a),) if a is not None else rgb

    # 未传入预计算色时：按翻转后的单帧 close_frac 回退（高=闭）
    if left_grip_bone is None or right_grip_bone is None:
        close = gripper_close_frac_series(
            np.asarray(gripper_t, dtype=np.float64).reshape(1, 2), robot_config)[0]
        left_closed = bool(close[0] >= GRIPPER_CLOSED_OPEN_FRAC)
        right_closed = bool(close[1] >= GRIPPER_CLOSED_OPEN_FRAC)
        left_grip_bone = COLOR_GRIPPER_CLOSED_BONE if left_closed else COLOR_GRIPPER_OPEN_BONE
        right_grip_bone = COLOR_GRIPPER_CLOSED_BONE if right_closed else COLOR_GRIPPER_OPEN_BONE
        left_grip_joint = COLOR_GRIPPER_CLOSED_JOINT if left_closed else COLOR_GRIPPER_OPEN_JOINT
        right_grip_joint = COLOR_GRIPPER_CLOSED_JOINT if right_closed else COLOR_GRIPPER_OPEN_JOINT

    if transparent:
        arm_bone = _as_bgr_color(COLOR_ARM_BONE, 255)
        arm_joint = _as_bgr_color(COLOR_ARM_JOINT, 255)
        left_grip_bone = _as_bgr_color(left_grip_bone, 255)
        right_grip_bone = _as_bgr_color(right_grip_bone, 255)
        left_grip_joint = _as_bgr_color(left_grip_joint, 255)
        right_grip_joint = _as_bgr_color(right_grip_joint, 255)
        inactive = _as_bgr_color(COLOR_INACTIVE, 180)
    else:
        arm_bone = _as_bgr_color(COLOR_ARM_BONE)
        arm_joint = _as_bgr_color(COLOR_ARM_JOINT)
        left_grip_bone = _as_bgr_color(left_grip_bone)
        right_grip_bone = _as_bgr_color(right_grip_bone)
        left_grip_joint = _as_bgr_color(left_grip_joint)
        right_grip_joint = _as_bgr_color(right_grip_joint)
        inactive = _as_bgr_color(COLOR_INACTIVE)

    for jn, j in robot.joint_map.items():
        jname = str(jn)
        if jname not in joint_points:
            continue
        is_active = jname in _worker_action_joint_set
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
                if jname in LEFT_GRIPPER_JOINT_SET:
                    bone_color = left_grip_bone
                elif jname in RIGHT_GRIPPER_JOINT_SET:
                    bone_color = right_grip_bone
                else:
                    bone_color = arm_bone
                cv2.line(img, uv1, uv2, bone_color, 4, cv2.LINE_AA)
            else:
                cv2.line(img, uv1, uv2, inactive, 2, cv2.LINE_AA)

    for jname, pos in joint_points.items():
        is_active = jname in _worker_action_joint_set
        if not is_active and not show_inactive:
            continue
        uv = _project(pos)
        if uv is not None:
            if is_active:
                if jname in LEFT_GRIPPER_JOINT_SET:
                    joint_color = left_grip_joint
                elif jname in RIGHT_GRIPPER_JOINT_SET:
                    joint_color = right_grip_joint
                else:
                    joint_color = arm_joint
                cv2.circle(img, uv, 6, joint_color, -1, cv2.LINE_AA)
            else:
                cv2.circle(img, uv, 3, inactive, -1, cv2.LINE_AA)

    return img


def _overlay_frame(bg_rgb: np.ndarray, skeleton_rgba: np.ndarray,
                   overlay_alpha: float = 0.7) -> np.ndarray:
    """将 RGBA 骨骼图叠加到 RGB 背景帧上，返回 RGB (H, W, 3)。

    bg_rgb:         原始视频帧，已缩放至 (H, W, 3) RGB。
    skeleton_rgba:  骨骼帧 (H, W, 4) RGBA，透明背景。
    overlay_alpha:  骨骼不透明区域的混合强度 (0~1)，1=完全覆盖，0.7=半透明。
    """
    alpha = skeleton_rgba[:, :, 3:4].astype(np.float32) / 255.0 * overlay_alpha
    fg = skeleton_rgba[:, :, :3].astype(np.float32)
    bg = bg_rgb.astype(np.float32)
    blended = fg * alpha + bg * (1.0 - alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 单条样本完整渲染
# ---------------------------------------------------------------------------
def render_one(task: dict) -> dict:
    """渲染单条样本：逐帧投影绘制 → ffmpeg pipe 输出 mp4。"""
    import subprocess

    idx = task["idx"]
    filtered_index = task.get("filtered_index", idx)
    started_at = time.time()

    def result(status: str, **fields: Any) -> Dict[str, Any]:
        return {
            "source_index": idx,
            "filtered_index": filtered_index,
            "status": status,
            "elapsed_seconds": round(time.time() - started_at, 3),
            **fields,
        }
    item = task["item"]
    output_path = task["output_path"]
    concat_output_path = task.get("concat_output_path")
    original_video_path = task.get("original_video_path")
    overlay_output_path = task.get("overlay_output_path")
    fallback_fps = task["fps"]
    show_inactive = task.get("show_inactive", False)
    print_all_joints = bool(task.get("print_all_joints", False))

    urdf_path = task.get("urdf_path")
    if urdf_path:
        _load_urdf_cached(urdf_path)

    src_video_path = task.get("src_video_path")
    robot_config = task.get("robot_config", DEFAULT_ROBOT_CONFIG)
    fps = fallback_fps
    if src_video_path and os.path.exists(src_video_path):
        detected = detect_fps(src_video_path)
        if detected > 0:
            fps = detected

    _setup_camera(robot_config["head_pitch"], robot_config["cam_pitch_offset"])
    threshold = robot_config.get("gripper_threshold", robot_config.get("threshold", 0.05))
    print(f"[idx={idx}] fps={fps:.1f} (detected from {'src_video' if fps != fallback_fps else 'fallback'}), "
          f"gripper_threshold={threshold}, closed_below={robot_config.get('closed_below', False)}, "
          f"head_pitch={robot_config['head_pitch']:.2f}, cam_pitch_offset={robot_config['cam_pitch_offset']:.1f}")
    concat_height = int(task.get("concat_height", _worker_img_h))
    enable_concat = bool(task.get("enable_concat", True))
    enable_overlay = bool(task.get("enable_overlay", False))
    overlay_alpha = float(task.get("overlay_alpha", 0.7))
    cleanup_action_after_concat = bool(task.get("cleanup_action_after_concat", False))

    try:
        arm_all, gripper_all = load_action_arrays_gripper(item)
    except Exception as e:
        return result("fail", reason=f"load_action failed: {e}")

    T = arm_all.shape[0]
    if T == 0:
        return result("fail", reason="0 frames")

    min_gripper_max = float(task.get("min_gripper_max", 0.0))
    if min_gripper_max > 0:
        max_l = float(np.max(gripper_all[:, 0]))
        max_r = float(np.max(gripper_all[:, 1]))
        if max_l < min_gripper_max and max_r < min_gripper_max:
            return result(
                "skipped_low_gripper",
                reason=f"gripper_too_small: max_L={max_l:.4f}, max_R={max_r:.4f}, "
                       f"threshold={min_gripper_max}",
                frames=T, max_left_gripper=max_l, max_right_gripper=max_r)

    # 左臂在动但左夹爪信号几乎不变：常见坏录制，跳过
    if bool(task.get("filter_left_gripper_static", True)):
        arm_ptp_min = float(task.get("left_arm_motion_ptp_min", LEFT_ARM_MOTION_PTP_MIN))
        grip_static_max = float(task.get("left_gripper_static_max", LEFT_GRIPPER_STATIC_MAX))
        bad_lg, lg_stats = left_arm_moves_gripper_static(
            arm_all, gripper_all, arm_ptp_min, grip_static_max)
        if bad_lg:
            return result(
                "skipped_left_gripper_static",
                reason=(
                    f"left_arm_moves_but_gripper_static: "
                    f"left_arm_ptp_max={lg_stats['left_arm_ptp_max']:.4f} "
                    f"(>= {arm_ptp_min}), "
                    f"left_gripper_span={lg_stats['left_gripper_span']:.4f} "
                    f"(<= {grip_static_max})"
                ),
                frames=T,
                **lg_stats,
            )

    # RealSource 风格夹爪配色：settled 开蓝/闭橙；运动骨线灰、关节红；极性已相对旧逻辑翻转
    bone_col_l, bone_col_r, joint_col_l, joint_col_r = precompute_gripper_draw_colors(
        gripper_all, robot_config)

    fk_backend = str(task.get("fk_backend", "vectorized"))
    vec_draws = None
    if fk_backend == "vectorized":
        try:
            vec_draws = _build_vectorized_draw_lists(
                arm_all, gripper_all, robot_config, show_inactive,
                bone_col_l, bone_col_r, joint_col_l, joint_col_r)
        except Exception as e:
            print(f"[idx={idx}] vectorized FK failed ({e}); fallback to yourdfpy")
            fk_backend = "yourdfpy"
            vec_draws = None

    disp = _URDF_JOINT_DISPLAY.get(
        os.path.basename(_worker_current_urdf or ""), {})
    joint_short = [
        disp.get(jn, jn.replace("_joint", ""))
        for jn in _worker_left_arm_joints
    ]
    header = "frame  " + "  ".join(f"{s:>16s}" for s in joint_short)
    frames_to_print = T if print_all_joints else min(T, 5)
    print_mode = "all" if print_all_joints else f"first {frames_to_print}"
    print(f"[idx={idx}] LEFT ARM joints (rad) + gripper, {T} frames, "
          f"print_mode={print_mode}, fk_backend={fk_backend}")
    print(header)
    for t in range(frames_to_print):
        vals = "  ".join(f"{v:+16.6f}" for v in arm_all[t, :7])
        print(f"{t:5d}  {vals}  L_grip={gripper_all[t,0]:.4f}  R_grip={gripper_all[t,1]:.4f}")
    if not print_all_joints and T > frames_to_print:
        print(f"  ... ({T - frames_to_print} more frames)")

    img_w = _worker_img_w
    img_h = _worker_img_h

    def _skeleton_frame(t: int, transparent: bool = False) -> np.ndarray:
        if vec_draws is not None:
            return _compose_skeleton_frame(vec_draws, t, transparent=transparent)
        return _draw_frame(
            arm_all[t], gripper_all[t], show_inactive, robot_config,
            transparent=transparent,
            left_grip_bone=bone_col_l[t],
            right_grip_bone=bone_col_r[t],
            left_grip_joint=joint_col_l[t],
            right_grip_joint=joint_col_r[t])

    # ---------- overlay 模式：逐帧读取原始视频 + 叠加骨骼 ----------
    overlay_status = "skip"
    overlay_reason = ""
    if enable_overlay and overlay_output_path and src_video_path:
        cap = cv2.VideoCapture(src_video_path)
        if not cap.isOpened():
            overlay_status = "fail"
            overlay_reason = f"cannot open src video: {src_video_path}"
        else:
            overlay_ffmpeg = [
                "ffmpeg", "-y",
                "-f", "rawvideo", "-vcodec", "rawvideo",
                "-s", f"{img_w}x{img_h}",
                "-pix_fmt", "rgb24", "-r", str(fps),
                "-i", "-",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-loglevel", "error",
                overlay_output_path,
            ]
            try:
                oproc = subprocess.Popen(overlay_ffmpeg, stdin=subprocess.PIPE,
                                         stderr=subprocess.PIPE)
                for t in range(T):
                    ret, bgr_frame = cap.read()
                    if not ret:
                        bg_rgb = np.zeros((img_h, img_w, 3), dtype=np.uint8)
                    else:
                        bg_rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                        if bg_rgb.shape[0] != img_h or bg_rgb.shape[1] != img_w:
                            bg_rgb = cv2.resize(bg_rgb, (img_w, img_h),
                                                interpolation=cv2.INTER_LINEAR)

                    skeleton_rgba = _skeleton_frame(t, transparent=True)
                    overlay_rgb = _overlay_frame(bg_rgb, skeleton_rgba,
                                                 overlay_alpha)
                    oproc.stdin.write(overlay_rgb.tobytes())

                oproc.stdin.close()
                oproc.wait()
                if (oproc.returncode == 0
                        and os.path.exists(overlay_output_path)
                        and os.path.getsize(overlay_output_path) > 0):
                    overlay_status = "ok"
                else:
                    overlay_status = "fail"
                    stderr_o = (oproc.stderr.read().decode(errors="replace")[:200]
                                if oproc.stderr else "")
                    overlay_reason = f"ffmpeg rc={oproc.returncode}: {stderr_o}"
            except Exception as e:
                overlay_status = "fail"
                overlay_reason = str(e)[:200]
            finally:
                cap.release()

    # ---------- 纯 action 骨骼视频（白色背景）----------
    skip_action_video = bool(task.get("skip_action_video", False))
    concat_status = "skip"
    concat_reason = ""

    if skip_action_video:
        return result(
            "ok",
            frames=T,
            concat_status=concat_status,
            concat_reason=concat_reason,
            overlay_status=overlay_status,
            overlay_reason=overlay_reason,
            fk_backend=fk_backend,
        )

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{img_w}x{img_h}",
        "-pix_fmt", "rgb24", "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-loglevel", "error",
        output_path,
    ]

    try:
        proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        for t in range(T):
            frame = _skeleton_frame(t, transparent=False)
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        proc.wait()
        if proc.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            if enable_concat and concat_output_path and original_video_path:
                try:
                    concat_cmd = [
                        "ffmpeg", "-y",
                        "-i", original_video_path,
                        "-i", output_path,
                        "-filter_complex",
                        (
                            f"[0:v]fps={fps},scale=-2:{concat_height}:flags=lanczos,setsar=1[v0];"
                            f"[1:v]fps={fps},scale=-2:{concat_height}:flags=lanczos,setsar=1[v1];"
                            f"[v0][v1]hstack=inputs=2[v]"
                        ),
                        "-map", "[v]",
                        "-an",
                        "-c:v", "libx264",
                        "-pix_fmt", "yuv420p",
                        "-r", str(fps),
                        "-shortest",
                        "-loglevel", "error",
                        concat_output_path,
                    ]
                    cproc = subprocess.run(concat_cmd, capture_output=True, text=True)
                    if cproc.returncode == 0 and os.path.exists(concat_output_path) and os.path.getsize(concat_output_path) > 0:
                        concat_status = "ok"
                    else:
                        concat_status = "fail"
                        concat_reason = (cproc.stderr or "").strip()[:200]
                except Exception as e:
                    concat_status = "fail"
                    concat_reason = str(e)[:200]

            if cleanup_action_after_concat and concat_status == "ok":
                try:
                    if os.path.exists(output_path):
                        os.remove(output_path)
                except Exception:
                    pass

            return result(
                "ok",
                frames=T,
                concat_status=concat_status,
                concat_reason=concat_reason,
                overlay_status=overlay_status,
                overlay_reason=overlay_reason,
                fk_backend=fk_backend,
            )
        stderr_msg = proc.stderr.read().decode(errors="replace")[:200] if proc.stderr else ""
        return result("fail", reason=f"ffmpeg rc={proc.returncode}: {stderr_msg}")
    except Exception as e:
        return result("fail", reason=str(e)[:200])


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="批量离线渲染 action 火柴棍视频 —— Head View（AG95 夹爪版）"
    )
    parser.add_argument(
        "--metadata", type=str,
        # default="/mnt/world_foundational_model/jackzou/OminiEWM/robomind_1400k_all_versions_with_caption_quality_20260219-042930_max_frames_600_20260316-164204_org_path_contains_tienkung_pro2_dualArm-gripper.pkl.gz",
        # default="/mnt/world_foundational_model/jackzou/OminiEWM/robomind_1400k_all_versions_with_caption_quality_20260219-042930_max_frames_600_20260316-164204_org_path_contains_tienkung_station_dualArm-gripper-3cameras.pkl.gz",
        # default="/mnt/world_foundational_model/jackzou/OminiEWM/robomind_1400k_all_versions_with_caption_quality_20260219-042930_max_frames_600_20260316-164204_org_path_contains_tienkung_max_dualArm-gripper.pkl.gz",
        default="/mnt/world_foundational_model/jackzou/OminiEWM/robomind_1400k_all_versions_with_caption_quality_20260219-042930_max_frames_600_20260316-164204_org_path_contains_tienkung_station_dualArm-gripper_OR_tienkung_pro2_dualArm-gripper.pkl.gz",
        help="pkl.gz 元数据路径",
    )
    parser.add_argument(
        "--selection-mode", choices=["legacy", "legacy4-full"], default="legacy",
        help="legacy4-full 保留旧流程四个采集目录的全部记录，并使用源 metadata 索引命名",
    )
    parser.add_argument(
        "--exclude-belt-conveyor", action=argparse.BooleanOptionalAction, default=True,
        help="排除传送带抓取任务（org_path 含 belt/conveyor；默认开启）",
    )
    parser.add_argument(
        "--urdf", type=str, default=None,
        help="强制指定所有样本使用的 URDF 文件路径。"
             "不指定时根据每条数据的 robot 类型自动选择 URDF",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="/mnt/world_foundational_model/jackzou/OminiEWM/action_video_overlay/",
        help="输出目录。save-mode=concat 时为拼接/临时目录；save-mode=concat_and_action 时为拼接视频 "
             "(*_concat.mp4) 所在目录",
    )
    parser.add_argument(
        "--action-output-dir", type=str, default=None,
        help="纯 action 火柴棍视频目录 ({idx:06d}.mp4)。仅在 save-mode=concat_and_action 时必填",
    )
    parser.add_argument("--workers", type=int, default=16,
                        help="并行 worker 数量")
    parser.add_argument("--fps", type=float, default=18.0,
                        help="兜底 FPS（优先使用原始视频的实际 FPS，检测失败时使用此值）")
    parser.add_argument("--width", type=int, default=640,
                        help="输出视频宽度（像素），默认 640（等比缩放自 1920×1080）")
    parser.add_argument("--height", type=int, default=360,
                        help="输出视频高度（像素），默认 360（等比缩放自 1920×1080）")
    parser.add_argument("--hfov", type=float, default=86.0,
                        help="水平视场角（度），Orbbec Gemini 336 彩色相机 = 86°")
    parser.add_argument("--vfov", type=float, default=55.0,
                        help="垂直视场角（度），Orbbec Gemini 336 彩色相机 = 55°")
    parser.add_argument("--head-pitch", type=float, default=None,
                        help="数据采集时头部俯仰角 (rad)，正值=向下看。"
                             "URDF head_pitch_joint 范围 [-0.60, 0.56]。"
                             "不指定时使用 ROBOT_CONFIGS 中各 robot 类型的默认值")
    parser.add_argument("--cam-pitch-offset", type=float, default=None,
                        help="额外光轴俯角微调 (度)，正值=进一步向下看。"
                             "不指定时使用 ROBOT_CONFIGS 中各 robot 类型的默认值")
    parser.add_argument("--resume", action="store_true",
                        help="跳过已存在的输出文件")
    parser.add_argument("--show-inactive", action="store_true",
                        help="显示非 active 关节（默认只显示 active 关节）")
    parser.add_argument(
        "--fk-backend", choices=["vectorized", "yourdfpy"], default="vectorized",
        help="FK/投影后端：vectorized=整段批量 FK（默认，更快）；yourdfpy=逐帧 update_cfg",
    )
    parser.add_argument("--start", type=int, default=0,
                        help="筛选后子集的起始索引（含），用于分段渲染")
    parser.add_argument("--end", type=int, default=1000000,
                        help="筛选后子集的结束索引（不含）；默认直到子集末尾")
    parser.add_argument("--subsample-1cam16", type=int, default=_DEFAULT_SUBSAMPLE_1CAM16,
                        help="一级目录为 1cameras_1 和 1cameras_6 的样本合并后最多保留条数")
    parser.add_argument("--seed", type=int, default=_DEFAULT_FILTER_SEED,
                        help="筛选阶段随机种子；设为 -1 表示每次随机抽样")
    parser.add_argument("--no-update-pkl", action="store_true",
                        help="渲染完成后不生成新的 pkl.gz")
    parser.add_argument("--no-concat-video", action="store_true",
                        help="不输出原始视频+action skeleton 的拼接视频")
    parser.add_argument(
        "--save-mode",
        type=str,
        default="overlay_and_action",
        choices=["all", "action", "original", "concat", "concat_and_action",
                 "overlay", "overlay_and_action"],
        help="输出文件模式：all / action / original / concat / concat_and_action / "
             "overlay(骨骼叠加到原始视频上) / "
             "overlay_and_action(同时输出叠加视频和纯 action 视频)",
    )
    parser.add_argument("--concat-height", type=int, default=480,
                        help="拼接视频统一缩放高度（像素）")
    parser.add_argument("--overlay-alpha", type=float, default=0.7,
                        help="overlay 模式下骨骼叠加不透明度 (0~1)，1=完全覆盖，0.7=半透明")
    parser.add_argument("--print-all-joints", action="store_true",
                        help="打印每条 episode 的全部 frame joint 信息；默认仅打印前 5 帧")
    parser.add_argument(
        "--filter-left-gripper-static",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="过滤「左臂明显运动但左夹爪信号几乎不变」的坏样本（默认开启）",
    )
    parser.add_argument(
        "--left-arm-motion-ptp-min", type=float, default=LEFT_ARM_MOTION_PTP_MIN,
        help="判定左臂在动的阈值：左臂 7 关节 peak-to-peak 最大值 (rad)",
    )
    parser.add_argument(
        "--left-gripper-static-max", type=float, default=LEFT_GRIPPER_STATIC_MAX,
        help="判定左夹爪不动的上限：左夹爪全程变化幅度",
    )
    parser.add_argument("--min-gripper-max", type=float, default=0.12,
                        help="过滤阈值：若某条 episode 的 max(L_grip) 和 max(R_grip) "
                             "均小于此值，则跳过渲染（默认 0.12，设为 0 禁用过滤）")
    parser.add_argument(
        "--report-dir", type=str, default=None,
        help="manifest/results/summary 输出目录；legacy4-full 默认使用 action 输出目录的父目录",
    )
    parser.add_argument(
        "--final-metadata-output", type=str, default=None,
        help="完整处理后生成的新 metadata 路径；不指定时沿用旧命名规则",
    )
    parser.add_argument(
        "--finalize-metadata", action="store_true",
        help="全范围且无渲染失败时生成最终 metadata；默认不生成",
    )
    parser.add_argument(
        "--preflight-only", action="store_true",
        help="只生成 selection_manifest 并校验路径，不启动渲染",
    )
    args = parser.parse_args(argv)

    if args.save_mode in ("concat_and_action", "overlay_and_action"):
        if not args.action_output_dir:
            parser.error(f"save-mode={args.save_mode} 时必须指定 --action-output-dir")
    elif args.action_output_dir:
        parser.error("--action-output-dir 仅在与 save-mode=concat_and_action / overlay_and_action 联用时有效")

    if args.width % 2 != 0:
        args.width += 1
    if args.height % 2 != 0:
        args.height += 1

    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_mode in ("concat_and_action", "overlay_and_action"):
        os.makedirs(args.action_output_dir, exist_ok=True)

    save_action = args.save_mode in ("all", "action", "concat", "concat_and_action",
                                     "overlay_and_action")
    save_original = args.save_mode in ("all", "original")
    save_concat = args.save_mode in ("all", "concat", "concat_and_action")
    save_overlay = args.save_mode in ("overlay", "overlay_and_action")
    if args.no_concat_video:
        save_concat = False
    if args.save_mode == "concat" and not save_concat:
        raise ValueError("save-mode=concat 与 --no-concat-video 冲突")
    if args.save_mode == "concat_and_action" and not save_concat:
        raise ValueError("save-mode=concat_and_action 与 --no-concat-video 冲突")

    # ====== 1. 加载元数据 ======
    print(f"[Main] 加载元数据: {args.metadata}")
    raw_metadata = _load_metadata(args.metadata)
    raw_items = list(_get_metadata_items(raw_metadata))
    raw_total = len(raw_items)
    print(f"[Main] 原始数据集共 {raw_total} 条")

    seed = None if args.seed < 0 else args.seed
    selected_pairs, filter_stats = _select_render_items(
        raw_items, args.selection_mode, args.subsample_1cam16, seed,
        exclude_belt_conveyor=bool(args.exclude_belt_conveyor))
    source_indices = [source_index for source_index, _ in selected_pairs]
    filtered_items = [item for _, item in selected_pairs]
    filtered_metadata = _rebuild_metadata(raw_metadata, filtered_items)
    data = list(_get_metadata_items(filtered_metadata))
    total = len(data)
    print(f"[Main] 选择模式: {args.selection_mode}")
    print(f"[Main] 排除传送带抓取: {bool(args.exclude_belt_conveyor)} "
          f"(excluded={filter_stats.get('excluded_belt_conveyor', 0)})")
    if args.selection_mode == "legacy4-full":
        for segment in sorted(LEGACY4_LEVEL1_SEGMENTS):
            print(f"[Main]   - 全量保留: {segment} ({filter_stats.get(segment, 0)})")
    else:
        print(f"[Main]   - 保留: {_ORG_SEG_STATION_3CAM_13}")
        print(f"[Main]   - 保留: {_ORG_SEG_PRO2_1CAM_2}")
        print(f"[Main]   - 抽样保留: {_ORG_SEG_PRO2_1CAM_1} + {_ORG_SEG_PRO2_1CAM_6} <= {args.subsample_1cam16}")
    print(f"[Main] 过滤后数据集共 {total} 条 (seed={seed})")
    print(f"[Main] 过滤统计: {filter_stats}")

    pkl_action_dir = (
        args.action_output_dir
        if args.save_mode in ("concat_and_action", "overlay_and_action")
        else args.output_dir
    )
    report_dir = args.report_dir
    if not report_dir and args.selection_mode == "legacy4-full":
        report_dir = os.path.dirname(os.path.abspath(pkl_action_dir))

    selection_rows: List[Dict[str, Any]] = []
    level1_counts: Counter = Counter()
    urdf_counts: Counter = Counter()
    missing_action = 0
    missing_video = 0
    preflight_errors = 0
    manifest_file = None
    manifest_tmp = None
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
        manifest_path = os.path.join(report_dir, "selection_manifest.jsonl")
        manifest_tmp = f"{manifest_path}.tmp"
        manifest_file = open(manifest_tmp, "w", encoding="utf-8")
    for filtered_index, (source_index, item) in enumerate(selected_pairs):
        level1 = _get_level1_org_segment(item)
        action = item.get("action", {}) if isinstance(item.get("action"), dict) else {}
        action_rel = action.get("all_fields", "")
        action_path = _resolve_video_path(str(item.get("root", "")), action_rel)
        try:
            source_video, selected_view = get_head_view_video_path(item)
            video_error = ""
        except Exception as exc:
            source_video, selected_view, video_error = "", "", str(exc)
        robot_config = get_robot_config(item)
        urdf_path = get_urdf_for_robot(item, args.urdf)
        action_exists = bool(action_path and os.path.exists(action_path))
        source_video_exists = bool(source_video and os.path.exists(source_video))
        errors = []
        if not action_rel:
            errors.append("missing action.all_fields")
        elif not action_exists:
            errors.append("action file not found")
        if video_error:
            errors.append(video_error)
        elif not source_video_exists:
            errors.append("source video not found")
        row = {
            "source_index": source_index,
            "filtered_index": filtered_index,
            "org_path": item.get("org_path"),
            "level1": level1,
            "robot": item.get("robot"),
            "resolved_robot_key": _resolve_robot_key(item),
            "urdf": urdf_path,
            "head_pitch": robot_config["head_pitch"],
            "cam_pitch_offset": robot_config["cam_pitch_offset"],
            "gripper_threshold": robot_config["gripper_threshold"],
            "closed_below": robot_config["closed_below"],
            "action_path": action_path,
            "action_exists": action_exists,
            "selected_view": selected_view,
            "source_video": source_video,
            "source_video_exists": source_video_exists,
            "preflight_error": "; ".join(errors),
        }
        level1_counts[level1] += 1
        urdf_counts[os.path.basename(urdf_path)] += 1
        missing_action += int(not action_exists)
        missing_video += int(not source_video_exists)
        preflight_errors += int(bool(errors))
        if manifest_file:
            manifest_file.write(json.dumps(row, ensure_ascii=False) + "\n")
        if not args.preflight_only:
            selection_rows.append(row)

    if manifest_file:
        manifest_file.close()
        os.replace(manifest_tmp, manifest_path)
    if args.preflight_only:
        if report_dir:
            preflight_summary = {
                "metadata": args.metadata,
                "selection_mode": args.selection_mode,
                "selected": total,
                "level1_counts": dict(sorted(level1_counts.items())),
                "urdf_counts": dict(sorted(urdf_counts.items())),
                "missing_action": missing_action,
                "missing_video": missing_video,
                "preflight_errors": preflight_errors,
            }
            _write_json(os.path.join(report_dir, "summary.json"), preflight_summary)
        print("[Main] preflight-only 完成")
        return 0

    start_idx = args.start if args.start is not None else 0
    end_idx = args.end if args.end is not None else total
    start_idx = max(0, min(start_idx, total))
    end_idx = max(start_idx, min(end_idx, total))

    # ====== 2. 构建任务列表 ======
    tasks: List[dict] = []
    skipped_exist = 0
    copied_original = 0
    skipped_original = 0
    failed_original = 0
    copied_concat_resume_skip = 0
    selected_head_count = 0
    selected_fallback_count = 0
    robot_type_counter: Dict[str, int] = {}
    robot_resolved_key: Dict[str, str] = {}
    robot_urdf_counter: Dict[str, Dict[str, int]] = {}
    all_urdf_paths: set = set()
    run_results: List[Dict[str, Any]] = []

    for idx in range(start_idx, end_idx):
        source_index = source_indices[idx]
        file_index = source_index if args.selection_mode == "legacy4-full" else idx
        orig_out_path = os.path.join(args.output_dir, f"{file_index:06d}_original.mp4")
        concat_out_path = os.path.join(args.output_dir, f"{file_index:06d}_concat.mp4")
        overlay_out_path = os.path.join(args.output_dir, f"{file_index:06d}_overlay.mp4")
        tmp_action_for_concat = os.path.join(args.output_dir, f"{file_index:06d}_action_tmp.mp4")
        if args.save_mode in ("concat_and_action", "overlay_and_action"):
            out_path = os.path.join(args.action_output_dir, f"{file_index:06d}.mp4")
        elif args.save_mode == "overlay":
            out_path = os.path.join(args.output_dir, f"{file_index:06d}_action_tmp.mp4")
        else:
            out_path = os.path.join(args.output_dir, f"{file_index:06d}.mp4")

        try:
            src_video, selected_view = get_head_view_video_path(data[idx])
            if "head" in selected_view.lower() or selected_view.lower() == "camera":
                selected_head_count += 1
            else:
                selected_fallback_count += 1
        except Exception:
            src_video = None
            selected_view = None

        item_robot_cfg = get_robot_config(data[idx])
        if args.head_pitch is not None:
            item_robot_cfg = {**item_robot_cfg, "head_pitch": args.head_pitch}
        if args.cam_pitch_offset is not None:
            item_robot_cfg = {**item_robot_cfg, "cam_pitch_offset": args.cam_pitch_offset}
        robot_name = str(data[idx].get("robot", "unknown"))
        robot_type_counter[robot_name] = robot_type_counter.get(robot_name, 0) + 1
        if robot_name not in robot_resolved_key:
            robot_resolved_key[robot_name] = _resolve_robot_key(data[idx])

        item_urdf = get_urdf_for_robot(data[idx], args.urdf)
        all_urdf_paths.add(item_urdf)
        robot_urdf_counter.setdefault(robot_name, {})
        robot_urdf_counter[robot_name][item_urdf] = (
            robot_urdf_counter[robot_name].get(item_urdf, 0) + 1
        )

        if save_original and not (args.resume and os.path.exists(orig_out_path) and os.path.getsize(orig_out_path) > 0):
            try:
                if src_video and os.path.exists(src_video):
                    shutil.copy2(src_video, orig_out_path)
                    copied_original += 1
                else:
                    failed_original += 1
            except Exception:
                failed_original += 1
        elif save_original:
            skipped_original += 1

        if args.save_mode == "original":
            continue

        if args.save_mode == "concat_and_action":
            dual_action_ok = os.path.exists(out_path) and os.path.getsize(out_path) > 0
            dual_concat_ok = (
                os.path.exists(concat_out_path) and os.path.getsize(concat_out_path) > 0
            )
            if args.resume and dual_action_ok and dual_concat_ok:
                skipped_exist += 1
                run_results.append({"source_index": source_index, "filtered_index": idx,
                                    "status": "exists", "output_path": out_path})
                continue
        elif args.save_mode == "overlay_and_action":
            dual_action_ok = os.path.exists(out_path) and os.path.getsize(out_path) > 0
            dual_overlay_ok = (
                os.path.exists(overlay_out_path) and os.path.getsize(overlay_out_path) > 0
            )
            if args.resume and dual_action_ok and dual_overlay_ok:
                skipped_exist += 1
                run_results.append({"source_index": source_index, "filtered_index": idx,
                                    "status": "exists", "output_path": out_path})
                continue
        elif args.save_mode == "overlay":
            if (args.resume
                    and os.path.exists(overlay_out_path)
                    and os.path.getsize(overlay_out_path) > 0):
                skipped_exist += 1
                run_results.append({"source_index": source_index, "filtered_index": idx,
                                    "status": "exists", "output_path": overlay_out_path})
                continue
        else:
            target_path = concat_out_path if args.save_mode == "concat" else out_path
            if args.resume and os.path.exists(target_path) and os.path.getsize(target_path) > 0:
                skipped_exist += 1
                run_results.append({"source_index": source_index, "filtered_index": idx,
                                    "status": "exists", "output_path": target_path})
                if save_concat and os.path.exists(orig_out_path) and os.path.getsize(orig_out_path) > 0:
                    if not (os.path.exists(concat_out_path) and os.path.getsize(concat_out_path) > 0):
                        tasks.append({
                            "idx": source_index,
                            "filtered_index": idx,
                            "item": data[idx],
                            "output_path": (
                                tmp_action_for_concat if args.save_mode == "concat" else out_path
                            ),
                            "fps": args.fps,
                            "show_inactive": args.show_inactive,
                            "print_all_joints": args.print_all_joints,
                            "min_gripper_max": args.min_gripper_max,
                            "filter_left_gripper_static": bool(args.filter_left_gripper_static),
                            "left_arm_motion_ptp_min": float(args.left_arm_motion_ptp_min),
                            "left_gripper_static_max": float(args.left_gripper_static_max),
                            "src_video_path": src_video,
                            "original_video_path": orig_out_path if save_original else src_video,
                            "concat_output_path": concat_out_path,
                            "enable_concat": True,
                            "enable_overlay": False,
                            "cleanup_action_after_concat": args.save_mode == "concat",
                            "concat_height": args.concat_height,
                            "robot_config": item_robot_cfg,
                            "urdf_path": item_urdf,
                            "fk_backend": args.fk_backend,
                        })
                    else:
                        copied_concat_resume_skip += 1
                continue

        if args.save_mode == "concat":
            action_out_for_task = tmp_action_for_concat
            cleanup_after_concat = True
        elif args.save_mode == "overlay":
            action_out_for_task = out_path
            cleanup_after_concat = False
        else:
            action_out_for_task = out_path
            cleanup_after_concat = False

        tasks.append({
            "idx": source_index,
            "filtered_index": idx,
            "item": data[idx],
            "output_path": action_out_for_task,
            "fps": args.fps,
            "show_inactive": args.show_inactive,
            "print_all_joints": args.print_all_joints,
            "min_gripper_max": args.min_gripper_max,
            "filter_left_gripper_static": bool(args.filter_left_gripper_static),
            "left_arm_motion_ptp_min": float(args.left_arm_motion_ptp_min),
            "left_gripper_static_max": float(args.left_gripper_static_max),
            "src_video_path": src_video,
            "original_video_path": (orig_out_path if save_original else src_video) if save_concat else None,
            "concat_output_path": concat_out_path if save_concat else None,
            "enable_concat": save_concat,
            "enable_overlay": save_overlay,
            "overlay_output_path": overlay_out_path if save_overlay else None,
            "overlay_alpha": args.overlay_alpha,
            "skip_action_video": args.save_mode == "overlay",
            "cleanup_action_after_concat": cleanup_after_concat,
            "concat_height": args.concat_height,
            "robot_config": item_robot_cfg,
            "urdf_path": item_urdf,
            "fk_backend": args.fk_backend,
        })

    urdf_paths_list = sorted(all_urdf_paths)

    print(f"[Main] Head View 渲染 (AG95 夹爪版, camera_head_link 透视投影)")
    print(f"[Main] FK backend: {args.fk_backend}")
    print(f"[Main] 保存模式: {args.save_mode} "
          f"(save_action={save_action}, save_original={save_original}, "
          f"save_concat={save_concat}, save_overlay={save_overlay})")
    hp_str = f"{args.head_pitch:.2f} rad" if args.head_pitch is not None else "per-robot"
    cp_str = f"{args.cam_pitch_offset:.1f}°" if args.cam_pitch_offset is not None else "per-robot"
    print(f"[Main] 输出分辨率: {args.width}x{args.height}, "
          f"HFOV: {args.hfov}°, VFOV: {args.vfov}°, "
          f"head_pitch: {hp_str}, cam_pitch_offset: {cp_str}")
    if save_overlay:
        print(f"[Main] Overlay alpha: {args.overlay_alpha:.2f}")
    if args.save_mode in ("concat_and_action", "overlay_and_action"):
        print(f"[Main] 双目录: {args.save_mode.split('_and_')[0]} → {args.output_dir}")
        print(f"[Main]         action → {args.action_output_dir}")
    if args.urdf:
        print(f"[Main] URDF (强制): {args.urdf}")
    else:
        print(f"[Main] URDF (自动选择): {len(urdf_paths_list)} 个文件")
        for p in urdf_paths_list:
            print(f"[Main]   - {p}")
    print(f"[Main] 视角选择: head_like={selected_head_count}, fallback={selected_fallback_count}")
    for rname, rcnt in sorted(robot_type_counter.items()):
        rkey = robot_resolved_key.get(rname, "")
        rcfg = ROBOT_CONFIGS.get(rkey, DEFAULT_ROBOT_CONFIG)
        resolved_info = f" (resolved={rkey!r})" if rkey != rname else ""
        urdf_parts = []
        for urdf_path, ucnt in sorted(robot_urdf_counter.get(rname, {}).items()):
            urdf_parts.append(f"{os.path.basename(urdf_path)} x{ucnt}")
        urdf_summary = ", ".join(urdf_parts) if urdf_parts else os.path.basename(
            args.urdf if args.urdf else DEFAULT_URDF
        )
        print(f"[Main] robot={rname!r}{resolved_info} ({rcnt}条): "
              f"urdfs={urdf_summary}, "
              f"gripper_threshold={rcfg['gripper_threshold']}, "
              f"closed_below={rcfg['closed_below']}, "
              f"head_pitch={rcfg['head_pitch']:.2f} rad, "
              f"cam_pitch_offset={rcfg['cam_pitch_offset']:.1f}°")
    print(f"[Main] 原始视频保存: copied={copied_original}, "
          f"skip(resume)={skipped_original}, fail={failed_original}")
    if save_concat:
        print(f"[Main] 拼接视频输出: enabled=True, concat_height={args.concat_height}, "
              f"existing_skip={copied_concat_resume_skip}")
    print(f"[Main] 渲染范围(筛选后子集): [{start_idx}, {end_idx})  "
          f"待渲染: {len(tasks)}  已跳过(resume): {skipped_exist}")

    if not tasks:
        print("[Main] 无需渲染的任务")
        if args.save_mode == "original":
            return 0
        if args.finalize_metadata and not args.no_update_pkl:
            _update_pkl(args.metadata, pkl_action_dir, filtered_metadata,
                        source_indices=source_indices, results=run_results,
                        output_path=args.final_metadata_output)
        return 0

    # ====== 3. 多进程渲染 ======
    num_workers = min(args.workers, len(tasks))
    print(f"[Main] 启动 {num_workers} 个 worker ...")

    t0 = time.time()
    ok_count = 0
    fail_count = 0
    skip_count = 0
    concat_ok = 0
    concat_fail = 0
    concat_skip = 0
    overlay_ok = 0
    overlay_fail = 0
    overlay_skip = 0

    with mp.Pool(
        num_workers,
        initializer=_init_worker,
        initargs=(urdf_paths_list, args.width, args.height, args.hfov, args.vfov),
    ) as pool:
        try:
            from tqdm import tqdm
            pbar = tqdm(total=len(tasks), desc="Rendering", unit="vid",
                        dynamic_ncols=True)
        except ImportError:
            pbar = None

        for result in pool.imap_unordered(render_one, tasks, chunksize=1):
            run_results.append(result)
            status = result.get("status", "?")
            if status == "ok":
                ok_count += 1
                cst = result.get("concat_status", "skip")
                if cst == "ok":
                    concat_ok += 1
                elif cst == "fail":
                    concat_fail += 1
                    print(f"\n  [CONCAT_FAIL] idx={result['source_index']}: {result.get('concat_reason', '?')}")
                else:
                    concat_skip += 1
                ost = result.get("overlay_status", "skip")
                if ost == "ok":
                    overlay_ok += 1
                elif ost == "fail":
                    overlay_fail += 1
                    print(f"\n  [OVERLAY_FAIL] idx={result['source_index']}: {result.get('overlay_reason', '?')}")
                else:
                    overlay_skip += 1
            elif status in ("skipped_low_gripper", "skipped_left_gripper_static"):
                skip_count += 1
            else:
                fail_count += 1
                print(f"\n  [FAIL] idx={result['source_index']}: {result.get('reason', '?')}")

            if pbar is not None:
                pbar.set_postfix(ok=ok_count, skip=skip_count, fail=fail_count)
                pbar.update(1)

        if pbar is not None:
            pbar.close()

    elapsed = time.time() - t0
    print(f"\n[Main] 渲染完成: ok={ok_count}, skip={skip_count}, "
          f"fail={fail_count}, 耗时={elapsed:.1f}s")
    if save_concat:
        print(f"[Main] 拼接视频统计: ok={concat_ok}, skip={concat_skip}, fail={concat_fail}")
    if save_overlay:
        print(f"[Main] 叠加视频统计: ok={overlay_ok}, skip={overlay_skip}, fail={overlay_fail}")

    if report_dir:
        results_path = os.path.join(report_dir, "render_results.jsonl")
        previous: Dict[int, Dict[str, Any]] = {}
        if os.path.exists(results_path):
            with open(results_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        row = json.loads(line)
                        previous[int(row["source_index"])] = row
                    except (ValueError, KeyError, json.JSONDecodeError):
                        continue
        for row in run_results:
            previous[int(row["source_index"])] = row
        all_results = [previous[k] for k in sorted(previous)]
        _write_jsonl(results_path, all_results)
        _write_jsonl(os.path.join(report_dir, "failed.jsonl"),
                     [r for r in all_results if r.get("status") == "fail"])
        _write_jsonl(os.path.join(report_dir, "skipped_low_gripper.jsonl"),
                     [r for r in all_results if r.get("status") == "skipped_low_gripper"])
        _write_jsonl(
            os.path.join(report_dir, "skipped_left_gripper_static.jsonl"),
            [r for r in all_results if r.get("status") == "skipped_left_gripper_static"],
        )
        status_counts = Counter(str(r.get("status", "unknown")) for r in all_results)
        _write_json(os.path.join(report_dir, "summary.json"), {
            "metadata": args.metadata,
            "selection_mode": args.selection_mode,
            "selected": total,
            "processed_results": len(all_results),
            "status_counts": dict(sorted(status_counts.items())),
            "level1_counts": dict(sorted(level1_counts.items())),
            "urdf_counts": dict(sorted(urdf_counts.items())),
            "range": [start_idx, end_idx],
        })

    # ====== 4. 生成带 action_video 的新 pkl.gz ======
    if args.finalize_metadata and not args.no_update_pkl:
        if start_idx == 0 and end_idx == total and fail_count == 0:
            _update_pkl(args.metadata, pkl_action_dir, filtered_metadata,
                        source_indices=source_indices, results=run_results,
                        output_path=args.final_metadata_output)
        else:
            print("[UpdatePkl] 未满足全范围且 fail=0，拒绝生成最终 metadata")

    return 0


def _update_pkl(metadata_path: str, output_dir: str, data: Any,
                source_indices: Optional[Sequence[int]] = None,
                results: Optional[Sequence[Dict[str, Any]]] = None,
                output_path: Optional[str] = None):
    """为筛选后的每条数据添加 action_video 字段，生成新的 pkl.gz。"""
    items = list(_get_metadata_items(data))
    new_items = []
    found = 0
    result_by_source = {int(r["source_index"]): r for r in (results or [])
                        if "source_index" in r}
    for idx, item in enumerate(items):
        item_copy = copy.copy(item)
        source_index = source_indices[idx] if source_indices is not None else idx
        vid_path = os.path.join(output_dir, f"{source_index:06d}.mp4")
        status = result_by_source.get(source_index, {}).get("status")
        if os.path.exists(vid_path) and os.path.getsize(vid_path) > 0:
            item_copy["action_video"] = vid_path
            item_copy["action_video_info"] = {
                "status": status or "exists",
                "source_index": source_index,
                "width": 640,
                "height": 360,
                "frame_nums": result_by_source.get(source_index, {}).get("frames"),
            }
            found += 1
        else:
            item_copy.pop("action_video", None)
            item_copy["action_video_info"] = {
                "status": status or "missing",
                "source_index": source_index,
                "reason": result_by_source.get(source_index, {}).get("reason", ""),
            }
        new_items.append(item_copy)

    new_data = _rebuild_metadata(data, new_items)

    base, ext = os.path.splitext(metadata_path)
    if base.endswith(".pkl"):
        base = base[:-4]
        ext = ".pkl.gz"
    out_pkl = output_path or f"{base}_with_action_video.pkl.gz"
    os.makedirs(os.path.dirname(os.path.abspath(out_pkl)), exist_ok=True)
    with gzip.open(out_pkl, "wb") as f:
        pickle.dump(new_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[UpdatePkl] 已生成: {out_pkl}  "
          f"(含 action_video: {found}/{len(items)})")


if __name__ == "__main__":
    raise SystemExit(main())
