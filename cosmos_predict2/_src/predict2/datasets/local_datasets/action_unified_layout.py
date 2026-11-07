# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical bimanual (+ optional dexhand) action layout shared across datasets.

Unified layout (``UNIFIED_ACTION_DIM=28``)::

    left_arm(7) + left_gripper(1) + left_dexhand(6)
    + right_arm(7) + right_gripper(1) + right_dexhand(6)

Rules for source → unified packing:

  - Prefer the richest arm DoF (``UNIFIED_ARM_DOF=7``). Shorter arms pad trailing
    joints with 0; longer arms are truncated to 7.
  - Gripper is always a scalar; missing gripper → 0.
  - Dexhand is ``UNIFIED_DEXHAND_DOF=6`` per side; missing dexhand → zeros
    (gripper-only sources such as RSW / AgiBot-gripper / RoboTwin / sim).
  - Single-arm sources fill the **left** arm (+ gripper / dexhand); right side
    is zero.

End-effector normalization (applied in ``pack_unified_bimanual_action``):

  - Gripper → ``[0, 1]`` via ``gripper / gripper_raw_max`` then clip.
    Real-robot / RoboTwin sources already use unit aperture (``raw_max=1``).
    Libero / ManiSkill / RoboCasa use Panda aperture meters (``raw_max=0.08``).
  - Dexhand → ``[0, 1]`` via affine map from
    ``[-DEXHAND_ABS_MAX, +DEXHAND_ABS_MAX]`` (AgiBot dexhand hardware range).
    Only applied when a dexhand stream is explicitly provided; padded zeros for
    gripper-only sources stay 0 (not remapped to 0.5).
  - Arm joints stay in radians (not forced into ``[0, 1]``).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

UNIFIED_ARM_DOF = 7
UNIFIED_GRIPPER_DIM = 1
UNIFIED_DEXHAND_DOF = 6
UNIFIED_SIDE_DIM = UNIFIED_ARM_DOF + UNIFIED_GRIPPER_DIM + UNIFIED_DEXHAND_DOF  # 14
UNIFIED_ACTION_DIM = UNIFIED_SIDE_DIM * 2  # 28

# Gripper unit aperture for parallel-jaw sources already in [0, 1].
GRIPPER_UNIT_RAW_MAX = 1.0
# Franka / Panda parallel-jaw open width used by Libero / ManiSkill / RoboCasa.
SIM_PANDA_GRIPPER_RAW_MAX = 0.08
# Observed AgiBot dexhand effector bound (symmetric around 0).
DEXHAND_ABS_MAX = 1.0562636

# Slice helpers within one side: [arm | gripper | dexhand]
_SIDE_ARM = slice(0, UNIFIED_ARM_DOF)
_SIDE_GRIP = UNIFIED_ARM_DOF  # index
_SIDE_DEX = slice(UNIFIED_ARM_DOF + UNIFIED_GRIPPER_DIM, UNIFIED_SIDE_DIM)

UNIFIED_ACTION_NAMES: tuple[str, ...] = (
    "LeftArm_Joint1",
    "LeftArm_Joint2",
    "LeftArm_Joint3",
    "LeftArm_Joint4",
    "LeftArm_Joint5",
    "LeftArm_Joint6",
    "LeftArm_Joint7",
    "LeftGripper",
    "LeftDexhand_1",
    "LeftDexhand_2",
    "LeftDexhand_3",
    "LeftDexhand_4",
    "LeftDexhand_5",
    "LeftDexhand_6",
    "RightArm_Joint1",
    "RightArm_Joint2",
    "RightArm_Joint3",
    "RightArm_Joint4",
    "RightArm_Joint5",
    "RightArm_Joint6",
    "RightArm_Joint7",
    "RightGripper",
    "RightDexhand_1",
    "RightDexhand_2",
    "RightDexhand_3",
    "RightDexhand_4",
    "RightDexhand_5",
    "RightDexhand_6",
)

# Legacy 16-D layout (pre-dexhand): L7+Lg+R7+Rg
LEGACY_GRIPPER_ONLY_DIM = UNIFIED_ARM_DOF * 2 + 2  # 16


def pad_or_truncate_joints(
    joints: np.ndarray,
    *,
    dof: int = UNIFIED_ARM_DOF,
) -> np.ndarray:
    """Pad trailing joints with 0 or truncate to ``dof`` → shape ``[..., dof]``."""
    arr = np.asarray(joints, dtype=np.float32)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    out = np.zeros(arr.shape[:-1] + (dof,), dtype=np.float32)
    n = min(int(arr.shape[-1]), dof)
    if n > 0:
        out[..., :n] = arr[..., :n]
    return out


def _as_T_D(x: Optional[np.ndarray], *, t: int, d: int) -> np.ndarray:
    if x is None:
        return np.zeros((t, d), dtype=np.float32)
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1) if t == 1 else arr.reshape(t, -1)
    if arr.ndim != 2:
        raise ValueError(f"Expected [T, D], got {arr.shape}")
    if arr.shape[0] == 1 and t > 1:
        arr = np.repeat(arr, t, axis=0)
    if arr.shape[0] != t:
        raise ValueError(f"T mismatch: expected {t}, got {arr.shape[0]}")
    out = np.zeros((t, d), dtype=np.float32)
    n = min(int(arr.shape[1]), d)
    if n > 0:
        out[:, :n] = arr[:, :n]
    return out


def normalize_gripper_to_unit(
    gripper: np.ndarray,
    *,
    raw_max: float = GRIPPER_UNIT_RAW_MAX,
) -> np.ndarray:
    """Map raw gripper aperture to ``[0, 1]``: ``clip(g / raw_max, 0, 1)``."""
    g = np.asarray(gripper, dtype=np.float32)
    scale = float(raw_max)
    if scale <= 0.0:
        raise ValueError(f"gripper raw_max must be > 0, got {raw_max}")
    return np.clip(g / scale, 0.0, 1.0).astype(np.float32, copy=False)


def normalize_dexhand_to_unit(
    dexhand: np.ndarray,
    *,
    abs_max: float = DEXHAND_ABS_MAX,
) -> np.ndarray:
    """Map dexhand joints from ``[-abs_max, abs_max]`` → ``[0, 1]``."""
    d = np.asarray(dexhand, dtype=np.float32)
    bound = float(abs_max)
    if bound <= 0.0:
        raise ValueError(f"dexhand abs_max must be > 0, got {abs_max}")
    return np.clip((d + bound) / (2.0 * bound), 0.0, 1.0).astype(np.float32, copy=False)


def denormalize_dexhand_from_unit(
    dexhand: np.ndarray,
    *,
    abs_max: float = DEXHAND_ABS_MAX,
) -> np.ndarray:
    """Inverse of ``normalize_dexhand_to_unit``: ``[0, 1]`` → ``[-abs_max, abs_max]``."""
    d = np.asarray(dexhand, dtype=np.float32)
    bound = float(abs_max)
    if bound <= 0.0:
        raise ValueError(f"dexhand abs_max must be > 0, got {abs_max}")
    return (d * (2.0 * bound) - bound).astype(np.float32, copy=False)


def split_unified_bimanual_action(action: np.ndarray) -> dict[str, np.ndarray]:
    """Unpack unified ``[T, 28]`` into arm / gripper / dexhand streams."""
    arr = np.asarray(action, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != UNIFIED_ACTION_DIM:
        raise ValueError(f"Expected unified [T, {UNIFIED_ACTION_DIM}], got {arr.shape}")
    return {
        "left_arm": arr[:, 0:7],
        "left_gripper": arr[:, 7],
        "left_dexhand": arr[:, 8:14],
        "right_arm": arr[:, 14:21],
        "right_gripper": arr[:, 21],
        "right_dexhand": arr[:, 22:28],
    }


def pack_unified_bimanual_action(
    left_arm: np.ndarray,
    left_gripper: np.ndarray,
    right_arm: np.ndarray,
    right_gripper: np.ndarray,
    left_dexhand: Optional[np.ndarray] = None,
    right_dexhand: Optional[np.ndarray] = None,
    *,
    gripper_raw_max: float = GRIPPER_UNIT_RAW_MAX,
    normalize_gripper: bool = True,
    normalize_dexhand: bool = True,
    dexhand_abs_max: float = DEXHAND_ABS_MAX,
) -> np.ndarray:
    """Pack per-side streams into unified ``[T, 28]``.

    Each of ``left_arm`` / ``right_arm`` is ``[T, >=arm_dof]`` (padded/truncated
    to 7). Grippers are ``[T]`` or ``[T, 1]``. Dexhand streams are optional
    ``[T, >=6]`` (missing → zeros).

    When ``normalize_gripper`` is True, grippers are scaled by ``gripper_raw_max``
    into ``[0, 1]``. When ``normalize_dexhand`` is True and a dexhand stream is
    provided, dexhand is affinely mapped into ``[0, 1]``. Arm joints are left
    unchanged (radians).
    """
    left_arm = np.asarray(left_arm, dtype=np.float32)
    if left_arm.ndim == 1:
        left_arm = left_arm[None, :]
    t = int(left_arm.shape[0])

    la = pad_or_truncate_joints(left_arm, dof=UNIFIED_ARM_DOF)
    ra = pad_or_truncate_joints(np.asarray(right_arm, dtype=np.float32), dof=UNIFIED_ARM_DOF)
    if ra.shape[0] != t:
        raise ValueError(f"right_arm T={ra.shape[0]} != left_arm T={t}")

    lg = np.asarray(left_gripper, dtype=np.float32).reshape(t)
    rg = np.asarray(right_gripper, dtype=np.float32).reshape(t)
    if normalize_gripper:
        lg = normalize_gripper_to_unit(lg, raw_max=gripper_raw_max)
        rg = normalize_gripper_to_unit(rg, raw_max=gripper_raw_max)

    # Only normalize explicitly provided dexhand streams so missing-side zeros
    # stay 0 (not remapped to mid-range 0.5).
    if left_dexhand is None:
        ld = np.zeros((t, UNIFIED_DEXHAND_DOF), dtype=np.float32)
    else:
        ld = _as_T_D(left_dexhand, t=t, d=UNIFIED_DEXHAND_DOF)
        if normalize_dexhand:
            ld = normalize_dexhand_to_unit(ld, abs_max=dexhand_abs_max)
    if right_dexhand is None:
        rd = np.zeros((t, UNIFIED_DEXHAND_DOF), dtype=np.float32)
    else:
        rd = _as_T_D(right_dexhand, t=t, d=UNIFIED_DEXHAND_DOF)
        if normalize_dexhand:
            rd = normalize_dexhand_to_unit(rd, abs_max=dexhand_abs_max)

    out = np.zeros((t, UNIFIED_ACTION_DIM), dtype=np.float32)
    # left side
    out[:, 0:7] = la
    out[:, 7] = lg
    out[:, 8:14] = ld
    # right side
    out[:, 14:21] = ra
    out[:, 21] = rg
    out[:, 22:28] = rd
    return out


def expand_legacy16_to_unified(
    action16: np.ndarray,
    *,
    gripper_raw_max: float = GRIPPER_UNIT_RAW_MAX,
) -> np.ndarray:
    """Map legacy ``[T, 16]`` (L7+Lg+R7+Rg) → unified ``[T, 28]`` (dexhand=0)."""
    arr = np.asarray(action16, dtype=np.float32)
    squeeze = False
    if arr.ndim == 1:
        arr = arr[None, :]
        squeeze = True
    if arr.ndim != 2 or arr.shape[1] != LEGACY_GRIPPER_ONLY_DIM:
        raise ValueError(f"Expected legacy [T, {LEGACY_GRIPPER_ONLY_DIM}], got {arr.shape}")
    out = pack_unified_bimanual_action(
        left_arm=arr[:, 0:7],
        left_gripper=arr[:, 7],
        right_arm=arr[:, 8:15],
        right_gripper=arr[:, 15],
        left_dexhand=None,
        right_dexhand=None,
        gripper_raw_max=gripper_raw_max,
    )
    return out[0] if squeeze else out


def _split_single_arm_row(
    row: np.ndarray,
    *,
    arm_dof: Optional[int],
    include_gripper: bool,
) -> tuple[np.ndarray, float]:
    """Parse one frame of single-arm raw action → (arm_joints[D], gripper)."""
    d = int(row.shape[0])
    if include_gripper:
        if arm_dof is None:
            if d < 2:
                raise ValueError(f"Single-arm+gripper action dim must be >= 2, got {d}")
            arm_dof_use = d - 1
        else:
            arm_dof_use = int(arm_dof)
            if d < arm_dof_use + 1:
                raise ValueError(
                    f"Single-arm+gripper expects dim >= arm_dof+1={arm_dof_use + 1}, got {d}"
                )
        arm = row[:arm_dof_use]
        grip = float(row[arm_dof_use])
    else:
        arm_dof_use = int(arm_dof) if arm_dof is not None else d
        if d < arm_dof_use:
            raise ValueError(f"Single-arm expects dim >= arm_dof={arm_dof_use}, got {d}")
        arm = row[:arm_dof_use]
        grip = 0.0
    return arm, grip


def _split_dual_arm_row(
    row: np.ndarray,
    *,
    arm_dof: Optional[int],
    include_gripper: bool,
) -> tuple[np.ndarray, float, np.ndarray, float]:
    """Parse one frame of dual-arm raw action → L_arm, L_g, R_arm, R_g.

    Expected packing (equal arms)::

        [L_arm (A), L_gripper?, R_arm (A), R_gripper?]
    """
    d = int(row.shape[0])
    if include_gripper:
        if arm_dof is None:
            if d < 4 or (d - 2) % 2 != 0:
                raise ValueError(
                    f"Dual-arm+gripper action dim must be even and >= 4 "
                    f"(2*arm_dof+2), got {d}"
                )
            arm_dof_use = (d - 2) // 2
        else:
            arm_dof_use = int(arm_dof)
            expected = 2 * arm_dof_use + 2
            if d < expected:
                raise ValueError(
                    f"Dual-arm+gripper expects dim >= {expected} (2*arm_dof+2), got {d}"
                )
        left_arm = row[:arm_dof_use]
        left_g = float(row[arm_dof_use])
        right_arm = row[arm_dof_use + 1 : arm_dof_use + 1 + arm_dof_use]
        right_g = float(row[arm_dof_use + 1 + arm_dof_use])
    else:
        if arm_dof is None:
            if d < 2 or d % 2 != 0:
                raise ValueError(f"Dual-arm action dim must be even and >= 2, got {d}")
            arm_dof_use = d // 2
        else:
            arm_dof_use = int(arm_dof)
            if d < 2 * arm_dof_use:
                raise ValueError(
                    f"Dual-arm expects dim >= {2 * arm_dof_use} (2*arm_dof), got {d}"
                )
        left_arm = row[:arm_dof_use]
        right_arm = row[arm_dof_use : 2 * arm_dof_use]
        left_g = 0.0
        right_g = 0.0
    return left_arm, left_g, right_arm, right_g


def to_unified_bimanual_action(
    raw: np.ndarray,
    *,
    num_arms: int = 1,
    include_gripper: bool = True,
    arm_dof: Optional[int] = None,
    left_dexhand: Optional[np.ndarray] = None,
    right_dexhand: Optional[np.ndarray] = None,
    gripper_raw_max: float = GRIPPER_UNIT_RAW_MAX,
    normalize_gripper: bool = True,
    normalize_dexhand: bool = True,
) -> np.ndarray:
    """Map raw per-frame gripper-arm actions to unified ``[T, 28]`` (or ``[28]``).

    Dexhand streams are optional; when omitted they are zero-filled. Pass
    ``left_dexhand`` / ``right_dexhand`` as ``[T, 6]`` (or ``[6]``) when available.
    When ``normalize_gripper`` is True, grippers are mapped to ``[0, 1]`` with
    ``gripper_raw_max`` (use ``SIM_PANDA_GRIPPER_RAW_MAX`` for Libero /
    ManiSkill / RoboCasa). Set ``normalize_gripper=False`` to keep source values.
    """
    arr = np.asarray(raw, dtype=np.float32)
    squeeze = False
    if arr.ndim == 1:
        arr = arr[None, :]
        squeeze = True
    if arr.ndim != 2:
        raise ValueError(f"Expected action [T, D] or [D], got shape {arr.shape}")

    # Already-unified (28) → pass-through (assumed already normalized at pack time).
    if arr.shape[1] == UNIFIED_ACTION_DIM:
        out = np.ascontiguousarray(arr)
        return out[0] if squeeze else out
    if arr.shape[1] == LEGACY_GRIPPER_ONLY_DIM and left_dexhand is None and right_dexhand is None:
        out = expand_legacy16_to_unified(arr, gripper_raw_max=gripper_raw_max)
        return out[0] if squeeze else out

    t = arr.shape[0]
    num_arms = int(num_arms)
    if num_arms not in (1, 2):
        raise ValueError(f"num_arms must be 1 or 2, got {num_arms}")

    left_arms = np.zeros((t, UNIFIED_ARM_DOF), dtype=np.float32)
    right_arms = np.zeros((t, UNIFIED_ARM_DOF), dtype=np.float32)
    left_g = np.zeros((t,), dtype=np.float32)
    right_g = np.zeros((t,), dtype=np.float32)

    for i in range(t):
        row = arr[i]
        if num_arms == 1:
            la, lg = _split_single_arm_row(
                row, arm_dof=arm_dof, include_gripper=include_gripper
            )
            left_arms[i] = pad_or_truncate_joints(la, dof=UNIFIED_ARM_DOF)
            left_g[i] = lg if include_gripper else 0.0
        else:
            la, lg, ra, rg = _split_dual_arm_row(
                row, arm_dof=arm_dof, include_gripper=include_gripper
            )
            left_arms[i] = pad_or_truncate_joints(la, dof=UNIFIED_ARM_DOF)
            right_arms[i] = pad_or_truncate_joints(ra, dof=UNIFIED_ARM_DOF)
            left_g[i] = lg if include_gripper else 0.0
            right_g[i] = rg if include_gripper else 0.0

    out = pack_unified_bimanual_action(
        left_arm=left_arms,
        left_gripper=left_g,
        right_arm=right_arms,
        right_gripper=right_g,
        left_dexhand=left_dexhand,
        right_dexhand=right_dexhand,
        gripper_raw_max=gripper_raw_max,
        normalize_gripper=normalize_gripper,
        normalize_dexhand=normalize_dexhand,
    )
    return out[0] if squeeze else out
