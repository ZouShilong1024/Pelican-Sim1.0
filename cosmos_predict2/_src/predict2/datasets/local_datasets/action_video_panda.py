"""Panda stick-figure used by Libero / ManiSkill / RoboCasa when sim replay is absent.

Drawing language matches ``action_video_backends/maniskill/render_action_video_maniskill.py``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

COLOR_ARM_BONE = (0, 120, 255)
COLOR_GRIPPER_OPEN_BONE = (0, 120, 255)
COLOR_GRIPPER_CLOSED_BONE = (255, 140, 0)
COLOR_ARM_JOINT = (255, 50, 50)
COLOR_GRIPPER_OPEN_JOINT = (255, 50, 50)
COLOR_GRIPPER_CLOSED_JOINT = (0, 200, 0)
COLOR_GRIPPER_MOTION_BONE = (128, 128, 128)
COLOR_GRIPPER_MOTION_JOINT = (255, 50, 50)
ARM_BONE_WIDTH = 6
ARM_JOINT_RADIUS = 9

GRIPPER_MAX_GAP = 0.08
GRIPPER_CLOSED_OPEN_FRAC = 0.5
GRIPPER_STATE_LOOKAHEAD = 5
GRIPPER_STATE_PLATEAU_EPS = 0.03
GRIPPER_STATE_SETTLED_FILL = 3
GRIPPER_EARLY_CLOSED_FRAC = 0.85

LIBERO_PANDA_BASE = np.array([-0.66, 0.0, 0.91], dtype=np.float64)
OBLIQUE45_TARGET = np.array([-0.32, 0.0, 1.00], dtype=np.float64)
OBLIQUE45_DIST = 1.75
OBLIQUE45_ELEV_DEG = 25.0
OBLIQUE45_AZIM_DEG = 45.0
OBLIQUE45_LOW_DZ = 0.55

# From the ManiSkill stick-figure renderer.
MANISKILL_BASE = np.array([0.0, 0.0, 0.0], dtype=np.float64)
MANISKILL_CLASSIC_3Q_EYE = np.array([0.6, 0.7, 0.6], dtype=np.float64)
MANISKILL_CLASSIC_3Q_TARGET = np.array([0.0, 0.0, 0.35], dtype=np.float64)

_PANDA_MDH = (
    (0.0, 0.0, 0.333),
    (0.0, -np.pi / 2, 0.0),
    (0.0, np.pi / 2, 0.316),
    (0.0825, np.pi / 2, 0.0),
    (-0.0825, -np.pi / 2, 0.384),
    (0.0, np.pi / 2, 0.0),
    (0.088, np.pi / 2, 0.107),
)

ARM_BONES = (
    ("link0", "link1"),
    ("link1", "link2"),
    ("link2", "link3"),
    ("link3", "link4"),
    ("link4", "link5"),
    ("link5", "link6"),
    ("link6", "link7"),
    ("link7", "hand"),
)
GRIPPER_BONES = (("hand", "leftfinger"), ("hand", "rightfinger"))


def _mdh(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [ct, -st, 0.0, a],
            [st * ca, ct * ca, -sa, -d * sa],
            [st * sa, ct * sa, ca, d * ca],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def panda_fk(q7: np.ndarray, gap: float, base_xyz: np.ndarray) -> Dict[str, np.ndarray]:
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = np.asarray(base_xyz, dtype=np.float64)
    pts: Dict[str, np.ndarray] = {"link0": T[:3, 3].copy()}
    q = np.asarray(q7, dtype=np.float64).reshape(-1)
    for i, (a, alpha, d) in enumerate(_PANDA_MDH):
        theta = float(q[i]) if i < q.shape[0] else 0.0
        T = T @ _mdh(a, alpha, d, theta)
        pts[f"link{i + 1}"] = T[:3, 3].copy()
    hand_T = T @ np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.058], [0, 0, 0, 1]],
        dtype=np.float64,
    )
    pts["hand"] = hand_T[:3, 3].copy()
    half = 0.5 * float(np.clip(gap, 0.0, GRIPPER_MAX_GAP))
    y = hand_T[:3, 1]
    z = hand_T[:3, 2]
    tip = hand_T[:3, 3] + 0.058 * z
    pts["leftfinger"] = tip + half * y
    pts["rightfinger"] = tip - half * y
    return pts


def lookat_cam2world(eye: np.ndarray, target: np.ndarray, up: Optional[np.ndarray] = None) -> np.ndarray:
    """OpenCV camera (Z forward, Y down). Returns cam→world 4×4."""
    eye = np.asarray(eye, dtype=np.float64).reshape(3)
    target = np.asarray(target, dtype=np.float64).reshape(3)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64) if up is None else np.asarray(up, dtype=np.float64)
    zaxis = target - eye
    n = float(np.linalg.norm(zaxis))
    if n < 1e-8:
        raise ValueError("camera eye and target are too close")
    zaxis = zaxis / n
    xaxis = np.cross(zaxis, up)
    xn = float(np.linalg.norm(xaxis))
    if xn < 1e-8:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        xaxis = np.cross(zaxis, up)
        xn = float(np.linalg.norm(xaxis))
    xaxis = xaxis / xn
    yaxis = np.cross(zaxis, xaxis)
    E = np.eye(4, dtype=np.float64)
    E[:3, 0] = xaxis
    E[:3, 1] = yaxis
    E[:3, 2] = zaxis
    E[:3, 3] = eye
    return E


def _oblique_eye_target(target: np.ndarray, dist: float, elev_deg: float, azim_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    elev = np.deg2rad(elev_deg)
    azim = np.deg2rad(azim_deg)
    offset = float(dist) * np.array(
        [np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)],
        dtype=np.float64,
    )
    tgt = np.asarray(target, dtype=np.float64).copy()
    return tgt + offset, tgt


def resolve_panda_camera(
    meta: Dict[str, Any],
    *,
    source: str,
    camera_preset: str = "",
) -> Tuple[np.ndarray, np.ndarray]:
    eye = meta.get("camera_eye")
    target = meta.get("camera_target")
    if eye is not None and target is not None:
        return np.asarray(eye, dtype=np.float64), np.asarray(target, dtype=np.float64)

    preset = str(camera_preset or meta.get("camera_preset") or "").strip()
    suite = str(meta.get("suite") or "")
    if source == "maniskill" or preset == "classic_3q":
        return MANISKILL_CLASSIC_3Q_EYE.copy(), MANISKILL_CLASSIC_3Q_TARGET.copy()

    extra_dz = 0.0
    if preset == "oblique45_low":
        extra_dz = -OBLIQUE45_LOW_DZ
    elif suite == "libero_object" and preset in ("", "oblique45", "default"):
        extra_dz = -OBLIQUE45_LOW_DZ
    target0 = OBLIQUE45_TARGET + np.array([0.0, 0.0, extra_dz], dtype=np.float64)
    return _oblique_eye_target(target0, OBLIQUE45_DIST, OBLIQUE45_ELEV_DEG, OBLIQUE45_AZIM_DEG)


def scale_K(K: np.ndarray, src_hw: Tuple[int, int], dst_hw: Tuple[int, int]) -> np.ndarray:
    K = np.asarray(K, dtype=np.float64).reshape(3, 3).copy()
    src_h, src_w = int(src_hw[0]), int(src_hw[1])
    dst_h, dst_w = int(dst_hw[0]), int(dst_hw[1])
    if src_w <= 0 or src_h <= 0:
        return K
    K[0, :] *= dst_w / float(src_w)
    K[1, :] *= dst_h / float(src_h)
    return K


def default_K(height: int, width: int, fov_deg: float = 45.0) -> np.ndarray:
    f = 0.5 * width / np.tan(np.deg2rad(fov_deg) / 2.0)
    return np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def project_points(points_world: np.ndarray, E_cam_to_world: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    E_inv = np.linalg.inv(E_cam_to_world)
    n = points_world.shape[0]
    homo = np.concatenate([points_world, np.ones((n, 1))], axis=1)
    cam = (E_inv @ homo.T).T[:, :3]
    depth = cam[:, 2].copy()
    safe = np.where(np.abs(depth) < 1e-6, 1e-6, depth)
    uv = (K @ cam.T).T
    uv = uv[:, :2] / safe[:, None]
    return uv, depth


def _fill_short_false_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    out = np.asarray(mask, dtype=bool).copy()
    n = out.size
    i = 0
    while i < n:
        if out[i]:
            i += 1
            continue
        j = i
        while j < n and not out[j]:
            j += 1
        if (j - i) <= max_gap and (i > 0 and j < n):
            out[i:j] = True
        i = j
    return out


def gripper_state_masks(close_frac: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.clip(np.asarray(close_frac, dtype=np.float64), 0.0, 1.0)
    t = x.size
    closed = x >= GRIPPER_CLOSED_OPEN_FRAC
    settled = np.zeros(t, dtype=bool)
    for i in range(t):
        j1 = min(t, i + 1 + GRIPPER_STATE_LOOKAHEAD)
        window = x[i:j1]
        if window.size and (float(window.max()) - float(window.min())) < GRIPPER_STATE_PLATEAU_EPS:
            settled[i] = True
        if x[i] >= GRIPPER_EARLY_CLOSED_FRAC and settled[i]:
            closed[i] = True
    settled = _fill_short_false_gaps(settled, GRIPPER_STATE_SETTLED_FILL)
    return closed, settled


def gripper_colors(closed: bool, settled: bool) -> Tuple[tuple, tuple]:
    if not settled:
        return COLOR_GRIPPER_MOTION_BONE, COLOR_GRIPPER_MOTION_JOINT
    if closed:
        return COLOR_GRIPPER_CLOSED_BONE, COLOR_GRIPPER_CLOSED_JOINT
    return COLOR_GRIPPER_OPEN_BONE, COLOR_GRIPPER_OPEN_JOINT


def _draw_skeleton(name2uv, name2z, img, grip_bone, grip_joint) -> np.ndarray:
    def _ok(n: str) -> bool:
        return n in name2z and name2z[n] > 0.01

    for a, b in ARM_BONES:
        if not (_ok(a) and _ok(b)):
            continue
        p1 = tuple(int(v) for v in name2uv[a])
        p2 = tuple(int(v) for v in name2uv[b])
        cv2.line(img, p1, p2, COLOR_ARM_BONE, ARM_BONE_WIDTH, cv2.LINE_AA)
        cv2.circle(img, p2, ARM_JOINT_RADIUS, COLOR_ARM_JOINT, -1, cv2.LINE_AA)
    for a, b in GRIPPER_BONES:
        if not (_ok(a) and _ok(b)):
            continue
        p1 = tuple(int(v) for v in name2uv[a])
        p2 = tuple(int(v) for v in name2uv[b])
        cv2.line(img, p1, p2, grip_bone, ARM_BONE_WIDTH, cv2.LINE_AA)
        cv2.circle(img, p2, ARM_JOINT_RADIUS, grip_joint, -1, cv2.LINE_AA)
    return img


def render_panda_action_video(
    action: np.ndarray,
    *,
    height: int,
    width: int,
    meta: Optional[Dict[str, Any]] = None,
    source: str = "libero",
    camera_preset: str = "",
) -> np.ndarray:
    """Render ``[T, 8]`` → white-bg RGB ``[T, H, W, 3]``."""
    meta = dict(meta or {})
    act = np.asarray(action, dtype=np.float64)
    if act.ndim != 2 or act.shape[1] < 7:
        raise ValueError(f"expected action [T, >=7], got {act.shape}")
    t = int(act.shape[0])
    joints = act[:, :7]
    if act.shape[1] >= 8:
        grip = act[:, 7]
        if float(np.nanmax(np.abs(grip))) <= 1.0 + 1e-6:
            gap = np.clip(grip, 0.0, 1.0) * GRIPPER_MAX_GAP
        else:
            gap = np.clip(grip, 0.0, GRIPPER_MAX_GAP)
    else:
        gap = np.full(t, GRIPPER_MAX_GAP, dtype=np.float64)
    close_frac = 1.0 - np.clip(gap / GRIPPER_MAX_GAP, 0.0, 1.0)
    closed_m, settled_m = gripper_state_masks(close_frac)

    src_h = int(meta.get("height") or height)
    src_w = int(meta.get("width") or width)
    K_raw = meta.get("K")
    K = default_K(height, width) if K_raw is None else scale_K(np.asarray(K_raw, dtype=np.float64), (src_h, src_w), (height, width))

    eye, _ = resolve_panda_camera(meta, source=source, camera_preset=camera_preset)
    base = MANISKILL_BASE if source == "maniskill" else LIBERO_PANDA_BASE
    names = [f"link{i}" for i in range(8)] + ["hand", "leftfinger", "rightfinger"]
    all_pts = [panda_fk(joints[i], float(gap[i]), base) for i in range(t)]
    centroid = np.mean(
        np.stack([np.stack([p[n] for n in ("link0", "link3", "hand")], axis=0) for p in all_pts], axis=0),
        axis=(0, 1),
    )
    E = lookat_cam2world(eye, centroid)

    frames = np.empty((t, height, width, 3), dtype=np.uint8)
    for i in range(t):
        pts = all_pts[i]
        world = np.stack([pts[n] for n in names], axis=0)
        uv, depth = project_points(world, E, K)
        name2uv = {n: uv[j] for j, n in enumerate(names)}
        name2z = {n: float(depth[j]) for j, n in enumerate(names)}
        gb, gj = gripper_colors(bool(closed_m[i]), bool(settled_m[i]))
        canvas = np.full((height, width, 3), 255, dtype=np.uint8)
        frames[i] = _draw_skeleton(name2uv, name2z, canvas, gb, gj)
    return frames


def load_action_npy(path: str) -> np.ndarray:
    arr = np.asarray(np.load(path))
    if arr.ndim != 2:
        raise ValueError(f"{path}: expected [T, D], got {arr.shape}")
    return arr
