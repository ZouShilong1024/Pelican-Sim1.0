#!/usr/bin/env python3
"""Batch render AgiBot-World `agibot_dexhand` action videos (BrainCo Revo2 assumption).

Assumes each hand's 6-D effector maps to Revo2 active joints in BrainCo SDK order:
  [Thumb, ThumbAux, Index, Middle, Ring, Pinky]
and mounts Revo2 on G1 via `visualization_urdf/agibot/G1_revo2.urdf`.

This script monkey-patches the gripper renderer helpers so gripper production
code stays untouched. Prefer `--fk-backend yourdfpy` for smoke.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Allow `python visualization_urdf/batch_render_...py` from repo root.
_VIS_DIR = Path(__file__).resolve().parent
if str(_VIS_DIR) not in sys.path:
    sys.path.insert(0, str(_VIS_DIR))

import batch_render_action_video_agibot as base  # noqa: E402

_ORIGINAL_BASE_RENDER_ONE = base.render_one

DEFAULT_URDF = str(
    Path(__file__).resolve().parent / "agibot" / "G1_revo2.urdf"
)
DEFAULT_EPISODE_CAMERA_MAPPING = base.DEFAULT_EPISODE_CAMERA_MAPPING
def episode_calib_path(calib_dir: Path, task_name: str, episode_index: int) -> Path:
    return calib_dir / f"{task_name}__episode_{int(episode_index):06d}.json"


def load_episode_calib(path: Path, task_name: str, episode_index: int) -> Dict[str, float]:
    """Read v2 or legacy per-episode JSON and validate its identity and values."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid episode calibration {path}: {exc}") from exc
    try:
        if str(payload["task_name"]) != task_name or int(payload["episode_index"]) != int(episode_index):
            raise ValueError("task_name/episode_index does not match the filename target")
        common_yaw = float(payload.get("cam_yaw_offset", 0.0))
        common_pitch = float(payload.get("cam_pitch_offset", 0.0))
        values = {
            "cam_yaw_offset": common_yaw,
            "cam_pitch_offset": common_pitch,
            "arm_l_base_y": float(
                payload.get("arm_l_base_y", base.DEFAULT_ARM_L_BASE_Y)
            ),
            "arm_r_base_y": float(
                payload.get("arm_r_base_y", base.DEFAULT_ARM_R_BASE_Y)
            ),
        }
        values["left_cam_yaw_offset"] = float(
            payload.get("left_cam_yaw_offset", common_yaw)
        )
        values["right_cam_yaw_offset"] = float(
            payload.get("right_cam_yaw_offset", common_yaw)
        )
        values["left_cam_pitch_offset"] = float(
            payload.get("left_cam_pitch_offset", common_pitch)
        )
        values["right_cam_pitch_offset"] = float(
            payload.get("right_cam_pitch_offset", common_pitch)
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid episode calibration {path}: {exc}") from exc
    if not all(math.isfinite(v) for v in values.values()):
        raise ValueError(f"invalid episode calibration {path}: parameters must be finite")
    return values


def apply_episode_calibrations(tasks: List[Dict[str, Any]], calib_dir: str) -> Dict[str, int]:
    stats = {"loaded": 0, "missing": 0, "invalid": 0}
    if not calib_dir:
        return stats
    root = Path(calib_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"episode calibration directory not found: {root}")
    for task in tasks:
        task_name = str(task["task_name"])
        episode_index = int(task["episode_index"])
        path = episode_calib_path(root, task_name, episode_index)
        if not path.exists():
            stats["missing"] += 1
            task["episode_calib_path"] = ""
            continue
        try:
            task.update(load_episode_calib(path, task_name, episode_index))
        except ValueError:
            stats["invalid"] += 1
            raise
        task["episode_calib_path"] = str(path)
        stats["loaded"] += 1
    return stats


def render_one_dexhand(task: Dict[str, Any]) -> Dict[str, Any]:
    """Apply per-task joint origins before rendering in a reused pool worker."""
    base._set_arm_base_y(
        float(task["arm_l_base_y"]),
        float(task["arm_r_base_y"]),
        robot=base._worker_robot,
        kin=base._worker_kin,
    )
    return _ORIGINAL_BASE_RENDER_ONE(task)

# BrainCo SDK bulk order: Thumb, ThumbAux, Index, Middle, Ring, Pinky
_HAND_JOINT_SUFFIXES = (
    "thumb_proximal_joint",  # Thumb flex
    "thumb_metacarpal_joint",  # Thumb aux
    "index_proximal_joint",
    "middle_proximal_joint",
    "ring_proximal_joint",
    "pinky_proximal_joint",
)
_HAND_LIMITS = {
    "thumb_proximal_joint": (0.0, 1.03),
    "thumb_metacarpal_joint": (0.0, 1.57),
    "index_proximal_joint": (0.0, 1.41),
    "middle_proximal_joint": (0.0, 1.41),
    "ring_proximal_joint": (0.0, 1.41),
    "pinky_proximal_joint": (0.0, 1.41),
}
_HAND_LIMIT_LO = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
_HAND_LIMIT_HI = np.array([1.03, 1.57, 1.41, 1.41, 1.41, 1.41], dtype=np.float64)
# AgiBot dexhand "open" is NOT URDF zero. Dominant rest pose across tasks:
#   [+0.271, -0.165, -0.165, -0.165, -0.165, ~-1.056]
# Thumb flex (dim0) moves opposite URDF when closing; others match after offset.
_HAND_OPEN_OFFSET = np.array(
    [0.27075, -0.16456, -0.16456, -0.16456, -0.16456, -1.0558], dtype=np.float64
)
_HAND_AXIS_SIGN = np.array([-1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
# Revo2 URDF thumb aux=0 is fully abducted (flared); aux→π/2 adducts toward
# palm. Slightly inside mid (0.2/0.9); 0.6/1.5 was too tucked (QA on 749 R).
_HAND_URDF_REST = np.array([0.25, 1.05, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
# AgiBot pinky channel (effector dim5 / 11) is effectively frozen in the released
# data: across tasks, when index/middle/ring move >0.15 rad, pinky range stays
# <0.05 (usually ~0). Drive pinky from the other finger curls for overlay so it
# bends with the hand instead of staying stuck at URDF zero.
_PINKY_COUPLE_RANGE_EPS = 0.08
_HAND_MIMIC = (
    ("thumb_distal_joint", "thumb_proximal_joint", 1.0),
    ("index_distal_joint", "index_proximal_joint", 1.155),
    ("middle_distal_joint", "middle_proximal_joint", 1.155),
    ("ring_distal_joint", "ring_proximal_joint", 1.155),
    ("pinky_distal_joint", "pinky_proximal_joint", 1.155),
)


def _side_prefix(side: str) -> str:
    return "left" if side == "left" else "right"


def _active_joint_names(side: str) -> List[str]:
    p = _side_prefix(side)
    return [f"{p}_{suf}" for suf in _HAND_JOINT_SUFFIXES]


def _estimate_open_offset(hand6: np.ndarray) -> np.ndarray:
    """Per-hand open offset; pinky rest varies by episode so refine from data."""
    open_off = _HAND_OPEN_OFFSET.copy()
    h = np.asarray(hand6, dtype=np.float64).reshape(-1, 6)
    if h.shape[0] == 0:
        return open_off
    # Frames near nominal open on index/middle/ring.
    d = np.linalg.norm(h[:, 2:5] - open_off[2:5], axis=1)
    near = h[d < 0.08]
    if near.shape[0] >= max(5, h.shape[0] // 20):
        open_off[5] = float(np.median(near[:, 5]))
    else:
        # Fallback: low finger-curl frames.
        curl = h[:, 2:5].sum(axis=1)
        k = max(5, h.shape[0] // 10)
        open_off[5] = float(np.median(h[np.argsort(curl)[:k], 5]))
    return open_off


def _calibrate_hand6(hand6: np.ndarray, open_off: Optional[np.ndarray] = None) -> np.ndarray:
    """Map AgiBot effector rad → Revo2 URDF joint rad for one hand (..., 6)."""
    h = np.asarray(hand6, dtype=np.float64)
    off = _HAND_OPEN_OFFSET if open_off is None else np.asarray(open_off, dtype=np.float64)
    q = (h - off) * _HAND_AXIS_SIGN + _HAND_URDF_REST
    return np.clip(q, _HAND_LIMIT_LO, _HAND_LIMIT_HI)


def _couple_pinky_if_frozen(q6: np.ndarray, raw6: np.ndarray) -> np.ndarray:
    """If raw pinky is frozen, drive it from index/middle/ring curls.

    Released AgiBot dexhand data usually keeps pinky near a constant while the
    other three fingers close. Copying only ring fails when ring itself is quiet
    but index/middle curl; use the mean of calibrated index/middle/ring instead.
    """
    raw = np.asarray(raw6, dtype=np.float64).reshape(-1, 6)
    q = np.asarray(q6, dtype=np.float64).reshape(-1, 6).copy()
    if raw.shape[0] == 0:
        return q.reshape(np.asarray(q6).shape)
    pinky_range = float(raw[:, 5].max() - raw[:, 5].min())
    other_ranges = np.asarray(
        [float(raw[:, i].max() - raw[:, i].min()) for i in (2, 3, 4)],
        dtype=np.float64,
    )
    other_move = float(other_ranges.max())
    if pinky_range < _PINKY_COUPLE_RANGE_EPS and other_move >= _PINKY_COUPLE_RANGE_EPS:
        q[:, 5] = np.clip(
            q[:, 2:5].mean(axis=1), _HAND_LIMIT_LO[5], _HAND_LIMIT_HI[5]
        )
    elif other_move >= 0.15:
        # Soft guard: pinky has some noise range but does not track finger curl.
        curl = q[:, 2:5].mean(axis=1)
        curled = curl > 0.35
        if curled.any() and float(q[curled, 5].mean()) < 0.15:
            q[:, 5] = np.clip(curl, _HAND_LIMIT_LO[5], _HAND_LIMIT_HI[5])
    return q.reshape(np.asarray(q6).shape)


def _calibrate_hand(hand: np.ndarray) -> np.ndarray:
    """Calibrate + clip (..., 6) or (..., 12) AgiBot effector → Revo2 URDF rad."""
    h = np.asarray(hand, dtype=np.float64)
    out = h.copy()
    flat = out.reshape(-1, out.shape[-1])
    n = flat.shape[-1]
    if n == 12:
        for sl in (slice(0, 6), slice(6, 12)):
            raw = flat[:, sl].copy()
            open_off = _estimate_open_offset(raw)
            q = _calibrate_hand6(raw, open_off)
            flat[:, sl] = _couple_pinky_if_frozen(q, raw)
    elif n == 6:
        raw = flat.copy()
        open_off = _estimate_open_offset(raw)
        q = _calibrate_hand6(raw, open_off)
        flat[:] = _couple_pinky_if_frozen(q, raw)
    else:
        raise ValueError(f"Unexpected hand dim {n}")
    return out


def _clip_urdf_hand(hand: np.ndarray) -> np.ndarray:
    """Clip already-calibrated (..., 6/12) URDF joint rad into limits."""
    h = np.asarray(hand, dtype=np.float64)
    out = h.copy()
    flat = out.reshape(-1, out.shape[-1])
    n = flat.shape[-1]
    if n == 12:
        for sl in (slice(0, 6), slice(6, 12)):
            flat[:, sl] = np.clip(flat[:, sl], _HAND_LIMIT_LO, _HAND_LIMIT_HI)
    elif n == 6:
        flat[:] = np.clip(flat, _HAND_LIMIT_LO, _HAND_LIMIT_HI)
    else:
        raise ValueError(f"Unexpected hand dim {n}")
    return out


def load_action_arrays_agibot_dexhand(
    parquet_path: str,
    need_state: bool = True,
) -> Dict[str, Dict[str, np.ndarray]]:
    import pandas as pd

    prefixes = ("action", "state") if need_state else ("action",)
    cols: List[str] = []
    for prefix in prefixes:
        cols += [f"{prefix}.joint.position.{i}" for i in range(14)]
        cols += [f"{prefix}.effector.position.{i}" for i in range(12)]
        cols += [f"{prefix}.head.position.{i}" for i in range(2)]
        cols += [f"{prefix}.waist.position.{i}" for i in range(2)]
    df = pd.read_parquet(parquet_path, columns=cols)

    def block(prefix: str) -> Dict[str, np.ndarray]:
        hand = df[[f"{prefix}.effector.position.{i}" for i in range(12)]].to_numpy(
            dtype=np.float64
        )
        hand = _calibrate_hand(hand)
        return {
            "arm": df[[f"{prefix}.joint.position.{i}" for i in range(14)]].to_numpy(
                dtype=np.float64
            ),
            # Reuse gripper slot so existing render iterators keep working.
            "gripper": hand,
            "head": df[[f"{prefix}.head.position.{i}" for i in range(2)]].to_numpy(
                dtype=np.float64
            ),
            "waist": df[[f"{prefix}.waist.position.{i}" for i in range(2)]].to_numpy(
                dtype=np.float64
            ),
        }

    out = {"action": block("action")}
    if need_state:
        out["state"] = block("state")
    return out


# Index/middle/ring/pinky proximal share the same Revo2 upper limit.
_FINGER_CURL_Q_MAX = float(_HAND_LIMIT_HI[2])


def _hand_open_fraction_from_curl(hand12: np.ndarray) -> np.ndarray:
    """Map calibrated hand URDF rad → open_frac (T, 2), high = closed.

    Uses mean of index/middle/ring/pinky proximal angles / 1.41 so the existing
    gripper plateau state machine (closed_open_frac / early_closed_frac) applies
    without new thresholds. Thumb is excluded.
    """
    h = np.asarray(hand12, dtype=np.float64)
    if h.ndim == 1:
        h = h.reshape(1, -1)
    if h.shape[-1] != 12:
        raise ValueError(f"Expected hand12 with last dim 12, got {h.shape}")
    out = np.empty((h.shape[0], 2), dtype=np.float64)
    for sidx, sl in enumerate((slice(0, 6), slice(6, 12))):
        fingers = np.clip(h[:, sl][:, 2:6], 0.0, _FINGER_CURL_Q_MAX)
        out[:, sidx] = fingers.mean(axis=1) / _FINGER_CURL_Q_MAX
    return np.clip(out, 0.0, 1.0)


def _resolve_render_hand(
    arrs: Dict[str, Dict[str, np.ndarray]],
    n: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (hand12, open_vals2) from four-finger curl (high = closed)."""
    hand = arrs["state"]["gripper"][:n]
    open_vals = _hand_open_fraction_from_curl(hand)
    return hand, open_vals


def _frame_cfg_dexhand(
    arm: np.ndarray,
    head: np.ndarray,
    waist: np.ndarray,
    hand: np.ndarray,
) -> Dict[str, float]:
    cfg: Dict[str, float] = {}
    for i, jn in enumerate(base.LEFT_ARM_JOINTS):
        cfg[jn] = float(arm[i])
    for i, jn in enumerate(base.RIGHT_ARM_JOINTS):
        cfg[jn] = float(arm[7 + i])
    cfg[base.WAIST_PITCH_JOINT] = float(waist[0])
    cfg[base.WAIST_LIFT_JOINT] = float(np.clip(waist[1], 0.0, 0.55))
    cfg[base.HEAD_YAW_JOINT] = float(head[0])
    cfg[base.HEAD_PITCH_JOINT] = float(np.clip(head[1], -0.349, 0.5236))

    hand = _clip_urdf_hand(np.asarray(hand, dtype=np.float64).reshape(-1))
    if hand.shape[0] != 12:
        raise ValueError(f"Expected 12-D hand, got {hand.shape}")
    for sidx, side in enumerate(("left", "right")):
        vals = hand[sidx * 6 : (sidx + 1) * 6]
        joints = _active_joint_names(side)
        for jn, q in zip(joints, vals):
            cfg[jn] = float(q)
        p = _side_prefix(side)
        for dst_suf, src_suf, mult in _HAND_MIMIC:
            src = f"{p}_{src_suf}"
            dst = f"{p}_{dst_suf}"
            cfg[dst] = float(mult * cfg[src])
    return cfg


def _frame_cfg_batch_dexhand(
    arm: np.ndarray,
    head: np.ndarray,
    waist: np.ndarray,
    hand: np.ndarray,
) -> Dict[str, np.ndarray]:
    cfg: Dict[str, np.ndarray] = {}
    for i, jn in enumerate(base.LEFT_ARM_JOINTS):
        cfg[jn] = arm[:, i].astype(np.float64)
    for i, jn in enumerate(base.RIGHT_ARM_JOINTS):
        cfg[jn] = arm[:, 7 + i].astype(np.float64)
    cfg[base.WAIST_PITCH_JOINT] = waist[:, 0].astype(np.float64)
    cfg[base.WAIST_LIFT_JOINT] = np.clip(waist[:, 1], 0.0, 0.55)
    cfg[base.HEAD_YAW_JOINT] = head[:, 0].astype(np.float64)
    cfg[base.HEAD_PITCH_JOINT] = np.clip(head[:, 1], -0.349, 0.5236)
    hand = _clip_urdf_hand(np.asarray(hand, dtype=np.float64))
    for sidx, side in enumerate(("left", "right")):
        vals = hand[:, sidx * 6 : (sidx + 1) * 6]
        joints = _active_joint_names(side)
        for i, jn in enumerate(joints):
            cfg[jn] = vals[:, i]
        p = _side_prefix(side)
        for dst_suf, src_suf, mult in _HAND_MIMIC:
            cfg[f"{p}_{dst_suf}"] = mult * cfg[f"{p}_{src_suf}"]
    return cfg


def _hand_link_chains() -> Dict[str, List[Tuple[str, str]]]:
    """Stick bones for Revo2 hands.

    G1 URDF has no `arm_*_link7` — wrist DOF child is `arm_*_end_link`, then
    `flange_*` (+0.05m) → `{left,right}_base_link`. Connect end→base so the
    arm tip and hand are one continuous skeleton.
    """
    chains: Dict[str, List[Tuple[str, str]]] = {}
    for side, arm_end, base_link in (
        ("left", "arm_l_end_link", "left_base_link"),
        ("right", "arm_r_end_link", "right_base_link"),
    ):
        p = _side_prefix(side)
        c: List[Tuple[str, str]] = [(arm_end, base_link)]
        # Thumb
        c += [
            (base_link, f"{p}_thumb_metacarpal_link"),
            (f"{p}_thumb_metacarpal_link", f"{p}_thumb_proximal_link"),
            (f"{p}_thumb_proximal_link", f"{p}_thumb_distal_link"),
            (f"{p}_thumb_distal_link", f"{p}_thumb_tip_link"),
        ]
        for finger in ("index", "middle", "ring", "pinky"):
            c += [
                (base_link, f"{p}_{finger}_proximal_link"),
                (f"{p}_{finger}_proximal_link", f"{p}_{finger}_distal_link"),
                (f"{p}_{finger}_distal_link", f"{p}_{finger}_tip_link"),
            ]
        chains[side] = c
    return chains


def load_task_meta_dexhand(task_dir: Path):
    """Like gripper load_task_meta, but synthesize episodes from parquet if jsonl missing."""
    import json

    meta_dir = task_dir / "meta"
    info = json.loads((meta_dir / "info.json").read_text())
    episodes_path = meta_dir / "episodes.jsonl"
    episodes = []
    if episodes_path.exists():
        with episodes_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    episodes.append(json.loads(line))
    else:
        # Incomplete LeRobot export: invent episode entries from on-disk parquet.
        for pq in sorted((task_dir / "data").glob("episode_*.parquet")):
            ei = int(pq.stem.split("_")[-1])
            episodes.append({"episode_index": ei})
    return info, episodes


def apply_dexhand_patches() -> None:
    """Patch gripper renderer module in-place for Revo2 dexhand."""
    base.load_action_arrays_agibot = load_action_arrays_agibot_dexhand
    base._resolve_render_gripper = _resolve_render_hand
    base._frame_cfg = _frame_cfg_dexhand
    base._frame_cfg_batch = _frame_cfg_batch_dexhand
    base.load_task_meta = load_task_meta_dexhand
    base.GRIPPER_LINK_CHAINS = _hand_link_chains()
    base.GRIPPER_BASE_LINKS = {
        "left": ("arm_l_end_link", "left_base_link"),
        "right": ("arm_r_end_link", "right_base_link"),
    }
    base.DEFAULT_GRIPPER_FORWARD_OFFSET = 0.0
    base.DEFAULT_GRIPPER_DELAY_FRAMES = 0

    # Thinner strokes than gripper; keep gripper three-state colors
    # (open blue / closed orange / motion gray; joints red/green/red).
    base.ACTIVE_BONE_WIDTH = max(1, int(round(base.ACTIVE_BONE_WIDTH * 0.5)))  # 6 -> 3
    base.ACTIVE_JOINT_RADIUS = max(1, int(round(base.ACTIVE_JOINT_RADIUS * 0.5)))  # 9 -> 4

    _orig_from_mapping = base.build_render_tasks_from_mapping

    def build_render_tasks_from_mapping_dexhand(args, episode_camera_mapping):
        # Temporarily accept dexhand task names.
        selected = []
        data_root = Path(args.data_root)
        task_info_cache: Dict[str, dict] = {}
        for (task_name, ei), mapped_camera in sorted(episode_camera_mapping.items()):
            if not task_name.startswith("agibot_dexhand__task_"):
                continue
            task_dir = data_root / task_name
            if not task_dir.exists():
                continue
            if task_name not in task_info_cache:
                info, _ = base.load_task_meta(task_dir)
                task_info_cache[task_name] = info
            info = task_info_cache[task_name]
            parquet = base.episode_parquet_path(task_dir, info, ei)
            if not parquet.exists():
                continue
            video = base.episode_video_path(task_dir, info, ei, args.video_key)
            if args.save_mode in ("overlay", "overlay_and_action") and not video.exists():
                continue
            action_path = (
                Path(args.action_output_dir or args.output_dir)
                / task_name
                / f"episode_{ei:06d}.mp4"
            )
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
            camera_params["camera_offset_rpy_deg"] = [
                float(x) for x in args.camera_offset_rpy_deg
            ]
            selected.append(
                {
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
                }
            )
            if args.limit > 0 and len(selected) >= args.limit:
                break
        return selected

    base.build_render_tasks_from_mapping = build_render_tasks_from_mapping_dexhand
    _ = _orig_from_mapping  # keep reference for debugging

    base.render_one = render_one_dexhand


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render AgiBot dexhand (BrainCo Revo2) action/overlay videos"
    )
    p.add_argument("--data-root", default="/datasets/AgiBot-World_v2.1_per_task_fixed")
    p.add_argument("--task-glob", default="agibot_dexhand__task_*")
    p.add_argument("--tasks", default="")
    p.add_argument("--episodes", default="")
    p.add_argument("--all-episodes", action="store_true")
    p.add_argument("--videos-per-task", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument(
        "--save-mode",
        choices=["action", "overlay", "overlay_and_action"],
        default="overlay_and_action",
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--action-output-dir", default="")
    p.add_argument("--metadata-output", default="")
    p.add_argument("--episode-camera-mapping", default=DEFAULT_EPISODE_CAMERA_MAPPING)
    p.add_argument(
        "--episode-calib-dir",
        default="",
        help="Directory containing <task>__episode_<index>.json calibration files.",
    )
    p.add_argument("--camera-param-source", choices=["csv", "jsonl"], default="csv")
    p.add_argument("--extrinsic-invert", action="store_true")
    p.add_argument("--no-episode-camera-params", action="store_true")
    p.add_argument("--require-episode-camera-params", action="store_true")
    p.add_argument("--video-key", default="observation.images.head_color")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--chunksize", type=int, default=1)
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--width", type=int, default=base.IMG_W)
    p.add_argument("--height", type=int, default=base.IMG_H)
    p.add_argument("--fx", type=float, default=base.MEAN_FX)
    p.add_argument("--fy", type=float, default=base.MEAN_FY)
    p.add_argument("--cx", type=float, default=base.MEAN_CX)
    p.add_argument("--cy", type=float, default=base.MEAN_CY)
    p.add_argument(
        "--camera-offset-xyz",
        type=float,
        nargs=3,
        default=base.DEFAULT_CAMERA_OFFSET_XYZ.tolist(),
    )
    p.add_argument(
        "--camera-offset-rpy-deg",
        type=float,
        nargs=3,
        default=base.DEFAULT_CAMERA_OFFSET_RPY_DEG.tolist(),
    )
    p.add_argument("--cam-axis-mode", choices=sorted(base.CAM_AXIS_MODES), default="x_fwd")
    p.add_argument("--cam-pitch-offset", type=float, default=0.0)
    p.add_argument("--cam-yaw-offset", type=float, default=0.0)
    p.add_argument("--left-cam-pitch-offset", type=float, default=None)
    p.add_argument("--right-cam-pitch-offset", type=float, default=None)
    p.add_argument("--left-cam-yaw-offset", type=float, default=None)
    p.add_argument("--right-cam-yaw-offset", type=float, default=None)
    p.add_argument(
        "--arm-l-base-y",
        type=float,
        default=base.DEFAULT_ARM_L_BASE_Y,
        help="Left arm_base joint origin Y in arm_base_link frame (meters, URDF default 0.025).",
    )
    p.add_argument(
        "--arm-r-base-y",
        type=float,
        default=base.DEFAULT_ARM_R_BASE_Y,
        help="Right arm_base joint origin Y in arm_base_link frame (meters, URDF default -0.025).",
    )
    p.add_argument("--gripper-forward-offset", type=float, default=0.0)
    p.add_argument("--gripper-delay-frames", type=int, default=0)
    p.add_argument("--gripper-closed-open-frac", type=float, default=base.GRIPPER_CLOSED_OPEN_FRAC)
    p.add_argument("--gripper-state-lookahead", type=int, default=base.GRIPPER_STATE_LOOKAHEAD)
    p.add_argument("--gripper-state-lookback", type=int, default=base.GRIPPER_STATE_LOOKBACK)
    p.add_argument("--gripper-state-plateau-eps", type=float, default=base.GRIPPER_STATE_PLATEAU_EPS)
    p.add_argument(
        "--gripper-state-change-delta", type=float, default=base.GRIPPER_STATE_CHANGE_DELTA
    )
    p.add_argument("--gripper-state-settled-fill", type=int, default=base.GRIPPER_STATE_SETTLED_FILL)
    p.add_argument("--gripper-early-closed-frac", type=float, default=base.GRIPPER_EARLY_CLOSED_FRAC)
    p.add_argument("--gripper-max-q", type=float, default=1.0)
    p.add_argument("--gripper-invert", action="store_true", default=False)
    p.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.5,
        help="Skeleton blend alpha on RGB overlay (default 0.5 = 50%% transparent).",
    )
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
    apply_dexhand_patches()
    args = parse_args(argv)
    if args.left_cam_pitch_offset is None:
        args.left_cam_pitch_offset = args.cam_pitch_offset
    if args.right_cam_pitch_offset is None:
        args.right_cam_pitch_offset = args.cam_pitch_offset
    if args.left_cam_yaw_offset is None:
        args.left_cam_yaw_offset = args.cam_yaw_offset
    if args.right_cam_yaw_offset is None:
        args.right_cam_yaw_offset = args.cam_yaw_offset
    if args.save_mode in ("action", "overlay_and_action") and not args.action_output_dir:
        if args.save_mode == "action":
            args.action_output_dir = args.output_dir
        else:
            args.action_output_dir = str(Path(args.output_dir) / "action")
            args.output_dir = str(Path(args.output_dir) / "overlay")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.action_output_dir:
        Path(args.action_output_dir).mkdir(parents=True, exist_ok=True)

    if not Path(args.urdf).exists():
        raise FileNotFoundError(
            f"URDF not found: {args.urdf}. Run visualization_urdf/agibot/build_g1_revo2_urdf.py first."
        )

    tasks = base.build_render_tasks(args)
    for task in tasks:
        task["left_cam_pitch_offset"] = float(args.left_cam_pitch_offset)
        task["right_cam_pitch_offset"] = float(args.right_cam_pitch_offset)
        task["left_cam_yaw_offset"] = float(args.left_cam_yaw_offset)
        task["right_cam_yaw_offset"] = float(args.right_cam_yaw_offset)
    calib_stats = apply_episode_calibrations(tasks, args.episode_calib_dir)
    if args.episode_calib_dir:
        print(
            f"[Dexhand] episode calib snapshot: loaded={calib_stats['loaded']} "
            f"missing={calib_stats['missing']} invalid={calib_stats['invalid']} "
            f"dir={Path(args.episode_calib_dir).resolve()}",
            flush=True,
        )
    print(
        f"[Dexhand] tasks={len(tasks)} urdf={args.urdf} save_mode={args.save_mode} "
        f"workers={args.workers} fk={args.fk_backend}",
        flush=True,
    )
    if not tasks:
        print("[Dexhand] no tasks selected", flush=True)
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

    # Ensure patches are visible in forked workers.
    try:
        mp.set_start_method("fork")
    except RuntimeError:
        pass

    with mp.Pool(num_workers, initializer=base._init_worker, initargs=initargs) as pool:
        for rec in pool.imap_unordered(base.render_one, tasks, chunksize=chunksize):
            records.append(rec)
            if rec.get("status") == "ok":
                ok += 1
            else:
                fail += 1
                print(
                    f"[FAIL] {rec.get('task_name')}#{rec.get('episode_index')}: {rec.get('reason')}",
                    flush=True,
                )
            done = ok + fail
            if done % 5 == 0 or done == len(tasks):
                elapsed = time.time() - t0
                print(
                    f"[Dexhand] done={done}/{len(tasks)} ok={ok} fail={fail} "
                    f"elapsed={elapsed/60:.1f}m last={rec.get('task_name')}#{rec.get('episode_index')}",
                    flush=True,
                )
    base.write_metadata(records, args.metadata_output)
    print(f"[Dexhand] finished ok={ok} fail={fail} elapsed={time.time() - t0:.1f}s", flush=True)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
