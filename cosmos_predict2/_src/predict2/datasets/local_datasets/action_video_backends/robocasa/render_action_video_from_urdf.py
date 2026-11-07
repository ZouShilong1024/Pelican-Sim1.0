#!/usr/bin/env python3
"""Render RoboCasa AC videos from action.npy + Panda URDF + camera K/E.

Does not open MuJoCo. Per demo directory:

  * ``action.npy``  (T, 8) = 7 arm joints (rad) + fingertip gap (m)
  * ``metadata.json`` ``views.{left,right}.{K, E_cam2world}``
  * ``base_pos`` + ``base_quat_wxyz`` of ``robot0_link0`` (kitchen layouts
    share height z=0.7 but differ in xy and yaw)

Usage:
    python visualization_urdf/robocasa/render_action_video_from_urdf.py \\
        --demo-dir episodes/01_pnp_counter_to_cab/demo_1 \\
        --save-mode overlay_and_action
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import threading
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


def _find_default_urdf() -> Path:
    here = Path(__file__).resolve().parent
    cands = [
        here / "models/franka/panda.urdf",
        here.parent / "models/franka/panda.urdf",
        Path.cwd() / "models/franka/panda.urdf",
    ]
    for p in cands:
        if p.is_file():
            return p
    return cands[0]


DEFAULT_URDF = _find_default_urdf()

ARM_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]
ARM_LINKS = [f"panda_link{i}" for i in range(0, 8)] + ["panda_hand"]
GRIPPER_LINKS = ["panda_leftfinger", "panda_rightfinger"]
SKELETON_LINKS = ARM_LINKS + GRIPPER_LINKS
ARM_BONES = [(ARM_LINKS[i], ARM_LINKS[i + 1]) for i in range(len(ARM_LINKS) - 1)]
GRIPPER_BONES = [("panda_hand", "panda_leftfinger"), ("panda_hand", "panda_rightfinger")]
FINGER_JOINTS = ("panda_finger_joint1", "panda_finger_joint2")
FINGER_MAX = 0.04
GRIPPER_MAX_GAP = 0.08

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

GRIPPER_CLOSED_OPEN_FRAC = 0.5
GRIPPER_STATE_LOOKAHEAD = 5
GRIPPER_STATE_PLATEAU_EPS = 0.05
GRIPPER_STATE_CHANGE_DELTA = 0.12
GRIPPER_STATE_SETTLED_FILL = 8
GRIPPER_STATE_MIN_SETTLED_RUN = 5
GRIPPER_STATE_SMOOTH_K = 5
GRIPPER_EARLY_CLOSED_FRAC = 0.75


def quat_wxyz_to_R(q: Sequence[float]) -> np.ndarray:
    w, x, y, z = [float(v) for v in q]
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def base_T(pos: Sequence[float], quat_wxyz: Optional[Sequence[float]] = None) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = np.asarray(pos, dtype=np.float64)
    if quat_wxyz is not None:
        T[:3, :3] = quat_wxyz_to_R(quat_wxyz)
    return T


def _as_xyz(value) -> Optional[List[float]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 3 and all(v is not None for v in value):
        return [float(v) for v in value]
    return None


def _as_quat(value) -> Optional[List[float]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 4 and all(v is not None for v in value):
        return [float(v) for v in value]
    return None


def _as_E44(value) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape == (4, 4):
        return arr
    if arr.shape == (3, 4):
        out = np.eye(4)
        out[:3, :4] = arr
        return out
    return None


def resolve_base(meta: dict, cli_pos=None, cli_quat=None) -> Tuple[List[float], List[float]]:
    pos = [float(x) for x in cli_pos] if cli_pos is not None else _as_xyz(meta.get("base_pos"))
    quat = [float(x) for x in cli_quat] if cli_quat is not None else _as_quat(meta.get("base_quat_wxyz"))
    if pos is None:
        raise ValueError("metadata.base_pos (robot0_link0 xyz) is required")
    if quat is None:
        quat = [1.0, 0.0, 0.0, 0.0]
    return pos, quat


def iter_views(meta: dict) -> List[Tuple[str, dict]]:
    views = meta.get("views") or {}
    out = []
    for suf in ("left", "right", "center"):
        if suf in views:
            out.append((suf, views[suf]))
    if not out and meta.get("K") is not None:
        out.append(("", meta))
    if not out:
        raise ValueError("metadata.views.{left,right} with K and E_cam2world required")
    return out


class UrdfFk:
    def __init__(self, urdf_path: str):
        import yourdfpy

        warnings.filterwarnings("ignore")
        logging.getLogger("yourdfpy").setLevel(logging.ERROR)
        self.urdf = yourdfpy.URDF.load(
            urdf_path,
            load_meshes=False,
            build_collision_scene_graph=False,
            load_collision_meshes=False,
        )
        self.base = self.urdf.base_link

    def points_world(self, cfg: Dict[str, float], T_world_base: np.ndarray) -> Dict[str, np.ndarray]:
        self.urdf.update_cfg(cfg)
        out: Dict[str, np.ndarray] = {}
        for lk in SKELETON_LINKS:
            try:
                Tb = self.urdf.get_transform(lk, self.base)
            except Exception:
                continue
            p = T_world_base @ np.array([Tb[0, 3], Tb[1, 3], Tb[2, 3], 1.0])
            out[lk] = p[:3]
        return out


def action_to_cfg(row: np.ndarray) -> Dict[str, float]:
    cfg = {ARM_JOINTS[i]: float(row[i]) for i in range(7)}
    finger = float(np.clip(float(row[7]) * 0.5, 0.0, FINGER_MAX))
    cfg[FINGER_JOINTS[0]] = finger
    cfg[FINGER_JOINTS[1]] = finger
    return cfg


def _fill_short_false_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
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


def _gripper_state_masks(close_frac_side: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ov = np.clip(np.asarray(close_frac_side, dtype=np.float64), 0.0, 1.0)
    t_len = int(ov.shape[0])
    if t_len == 0:
        empty = np.zeros(0, dtype=bool)
        return empty, empty
    ov = _median_smooth_1d(ov, GRIPPER_STATE_SMOOTH_K)
    settled = np.zeros(t_len, dtype=bool)
    plateau_val = np.zeros(t_len, dtype=np.float64)
    for t in range(t_len):
        hi = min(t_len, t + GRIPPER_STATE_LOOKAHEAD + 1)
        window = ov[t:hi]
        settled[t] = (float(window.max()) - float(window.min())) < GRIPPER_STATE_PLATEAU_EPS
        plateau_val[t] = float(np.median(window))
        if (not settled[t]) and ov[t] >= GRIPPER_EARLY_CLOSED_FRAC:
            fwd = ov[t:hi]
            if (float(fwd.max()) - float(fwd.min())) < GRIPPER_STATE_PLATEAU_EPS:
                settled[t] = True
                plateau_val[t] = float(np.median(fwd))
    settled = _suppress_short_true_runs(settled, GRIPPER_STATE_MIN_SETTLED_RUN)
    settled = _fill_short_false_gaps(settled, GRIPPER_STATE_SETTLED_FILL)
    settled_idx = np.flatnonzero(settled)
    if settled_idx.size == 0:
        return ov >= GRIPPER_CLOSED_OPEN_FRAC, np.ones(t_len, dtype=bool)
    ref = plateau_val[int(settled_idx[0])]
    state_closed = ref >= GRIPPER_CLOSED_OPEN_FRAC
    closed = np.zeros(t_len, dtype=bool)
    for t in range(t_len):
        if settled[t]:
            v = plateau_val[t]
            if abs(v - ref) > GRIPPER_STATE_CHANGE_DELTA:
                state_closed = v > ref
                ref = v
        closed[t] = state_closed
    return closed, settled


def _gripper_bone_color(closed: bool, settled: bool) -> tuple:
    if not settled:
        return COLOR_GRIPPER_MOTION_BONE
    return COLOR_GRIPPER_CLOSED_BONE if closed else COLOR_GRIPPER_OPEN_BONE


def _gripper_joint_color(closed: bool, settled: bool) -> tuple:
    if not settled:
        return COLOR_GRIPPER_MOTION_JOINT
    return COLOR_GRIPPER_CLOSED_JOINT if closed else COLOR_GRIPPER_OPEN_JOINT


def _project(points_world: np.ndarray, E_cam_to_world: np.ndarray,
             K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    E_inv = np.linalg.inv(E_cam_to_world)
    n = points_world.shape[0]
    homo = np.concatenate([points_world, np.ones((n, 1))], axis=1)
    cam = (E_inv @ homo.T).T[:, :3]
    depth = cam[:, 2].copy()
    safe = np.where(np.abs(depth) < 1e-6, 1e-6, depth)
    uv = (K @ cam.T).T
    uv = uv[:, :2] / safe[:, None]
    return uv, depth


def _draw_skeleton(name2uv: Dict[str, np.ndarray], name2z: Dict[str, float],
                   img: np.ndarray, grip_bone, grip_joint) -> np.ndarray:
    def _ok(n):
        return n in name2z and name2z[n] > 0.01

    for a, b in ARM_BONES:
        if not (_ok(a) and _ok(b)):
            continue
        p1 = tuple(name2uv[a].astype(int))
        p2 = tuple(name2uv[b].astype(int))
        cv2.line(img, p1, p2, COLOR_ARM_BONE, ARM_BONE_WIDTH, cv2.LINE_AA)
        cv2.circle(img, p2, ARM_JOINT_RADIUS, COLOR_ARM_JOINT, -1, cv2.LINE_AA)
    for a, b in GRIPPER_BONES:
        if not (_ok(a) and _ok(b)):
            continue
        p1 = tuple(name2uv[a].astype(int))
        p2 = tuple(name2uv[b].astype(int))
        cv2.line(img, p1, p2, grip_bone, ARM_BONE_WIDTH, cv2.LINE_AA)
        cv2.circle(img, p2, ARM_JOINT_RADIUS, grip_joint, -1, cv2.LINE_AA)
    return img


def _encode_video(path: str, frames: List[np.ndarray], fps: float) -> Tuple[bool, str]:
    if not frames:
        return False, "no frames"
    h, w = frames[0].shape[:2]
    tmp = path + ".part"
    cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}", "-pix_fmt", "rgb24", "-r", f"{fps:g}", "-i", "-",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-f", "mp4", "-loglevel", "error", tmp,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    chunks: List[bytes] = []

    def _drain():
        try:
            while True:
                c = proc.stderr.read(4096) if proc.stderr else b""
                if not c:
                    break
                chunks.append(c)
        except Exception:
            pass

    th = threading.Thread(target=_drain, daemon=True)
    th.start()
    try:
        for fr in frames:
            proc.stdin.write(np.ascontiguousarray(fr, dtype=np.uint8).tobytes())
        proc.stdin.close()
        proc.wait(timeout=600)
    except Exception as e:  # noqa: BLE001
        try:
            proc.kill()
        except Exception:
            pass
        return False, repr(e)
    finally:
        th.join(timeout=5.0)
    if proc.returncode == 0 and os.path.exists(tmp) and os.path.getsize(tmp) > 0:
        os.replace(tmp, path)
        return True, ""
    err = b"".join(chunks).decode(errors="replace")[-200:]
    if os.path.exists(tmp):
        os.remove(tmp)
    return False, f"ffmpeg rc={proc.returncode}: {err}"


def read_mp4_rgb(path: Path) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def save_still(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def _view_paths(demo_dir: Path, suf: str) -> Tuple[Path, Path]:
    if suf:
        return demo_dir / f"rgb_{suf}.mp4", demo_dir / f"action_video_sim_{suf}.mp4"
    return demo_dir / "rgb.mp4", demo_dir / "action_video_sim.mp4"


def render_demo(
    demo_dir: Path,
    out_dir: Path,
    urdf_path: Path,
    save_mode: str,
    base_pos: Optional[Sequence[float]] = None,
    base_quat: Optional[Sequence[float]] = None,
    still_frac: Tuple[float, ...] = (0.0, 0.33, 0.66, 1.0),
) -> dict:
    meta = json.loads((demo_dir / "metadata.json").read_text())
    actions = np.load(demo_dir / "action.npy").astype(np.float64)
    if actions.ndim != 2 or actions.shape[1] < 8:
        raise ValueError(f"expected action (T,8), got {actions.shape}")
    T = int(actions.shape[0])
    width = int(meta.get("width") or 512)
    height = int(meta.get("height") or 512)
    fps = float(meta.get("fps") or 20.0)
    pos, quat = resolve_base(meta, base_pos, base_quat)
    Twb = base_T(pos, quat)
    views = iter_views(meta)

    fk = UrdfFk(str(urdf_path))
    close_frac = 1.0 - np.clip(actions[:, 7] / GRIPPER_MAX_GAP, 0.0, 1.0)
    closed_m, settled_m = _gripper_state_masks(close_frac)

    want_av = save_mode in ("action", "overlay_and_action", "all")
    want_ov = save_mode in ("overlay", "overlay_and_action", "all")

    view_cfg = []
    for suf, vmeta in views:
        K = np.asarray(vmeta["K"], dtype=np.float64)
        E = _as_E44(vmeta.get("E_cam2world") or vmeta.get("E"))
        if E is None:
            raise ValueError(f"view {suf or 'default'} missing E_cam2world")
        rgb_path, sim_av = _view_paths(demo_dir, suf)
        rgb_frames: List[np.ndarray] = []
        if want_ov:
            if not rgb_path.is_file():
                raise FileNotFoundError(rgb_path)
            rgb_frames = read_mp4_rgb(rgb_path)
        orig_av: List[np.ndarray] = []
        if want_av and sim_av.is_file():
            orig_av = read_mp4_rgb(sim_av)
        elif want_av:
            alt = demo_dir / (f"action_video_{suf}.mp4" if suf else "action_video.mp4")
            if alt.is_file() and alt.resolve() != (out_dir / alt.name).resolve():
                orig_av = read_mp4_rgb(alt)
        view_cfg.append((suf, K, E, rgb_frames, orig_av))

    av_by: Dict[str, List[np.ndarray]] = {suf: [] for suf, *_ in view_cfg}
    ov_by: Dict[str, List[np.ndarray]] = {suf: [] for suf, *_ in view_cfg}
    for t in range(T):
        pts = fk.points_world(action_to_cfg(actions[t]), Twb)
        names = [n for n in SKELETON_LINKS if n in pts]
        arr = np.array([pts[n] for n in names], dtype=np.float64)
        gb = _gripper_bone_color(bool(closed_m[t]), bool(settled_m[t]))
        gj = _gripper_joint_color(bool(closed_m[t]), bool(settled_m[t]))
        for suf, K, E, rgb_frames, _orig in view_cfg:
            uv, depth = _project(arr, E, K)
            name2uv = {n: uv[i] for i, n in enumerate(names)}
            name2z = {n: float(depth[i]) for i, n in enumerate(names)}
            if want_av:
                canvas = np.full((height, width, 3), 255, dtype=np.uint8)
                av_by[suf].append(_draw_skeleton(name2uv, name2z, canvas, gb, gj))
            if want_ov:
                base = rgb_frames[min(t, len(rgb_frames) - 1)].copy()
                if base.shape[0] != height or base.shape[1] != width:
                    base = cv2.resize(base, (width, height), interpolation=cv2.INTER_AREA)
                ov_by[suf].append(_draw_skeleton(name2uv, name2z, base, gb, gj))

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "demo_dir": str(demo_dir),
        "frames": T,
        "base_pos": list(map(float, pos)),
        "base_quat_wxyz": list(map(float, quat)),
        "urdf": str(urdf_path),
        "views": {},
    }
    still_dir = out_dir / "stills"
    n_ref = T
    idxs = sorted({int(np.clip(round(f * (n_ref - 1)), 0, n_ref - 1)) for f in still_frac})
    for suf, K, E, _rgb, orig_av in view_cfg:
        tag = suf or "main"
        summary["views"][tag] = {"K": K.tolist(), "E_cam2world": E.tolist()}
        np.save(out_dir / (f"E_cam2world_{tag}.npy" if suf else "E_cam2world.npy"), E)
        av_name = f"action_video_{suf}.mp4" if suf else "action_video.mp4"
        ov_name = f"overlay_{suf}.mp4" if suf else "overlay.mp4"
        if want_av:
            ok, err = _encode_video(str(out_dir / av_name), av_by[suf], fps)
            if not ok:
                raise RuntimeError(f"{av_name}: {err}")
        if want_ov:
            ok, err = _encode_video(str(out_dir / ov_name), ov_by[suf], fps)
            if not ok:
                raise RuntimeError(f"{ov_name}: {err}")
        prefix = f"{tag}_" if suf else ""
        for i in idxs:
            if want_av and i < len(av_by[suf]):
                save_still(still_dir / f"ac_{prefix}{i:03d}.jpg", av_by[suf][i])
                if i < len(orig_av):
                    pair = np.concatenate([orig_av[i], av_by[suf][i]], axis=1)
                    save_still(still_dir / f"ac_pair_{prefix}{i:03d}.jpg", pair)
            if want_ov and i < len(ov_by[suf]):
                save_still(still_dir / f"ov_{prefix}{i:03d}.jpg", ov_by[suf][i])
    (out_dir / "render_meta.json").write_text(json.dumps(summary, indent=2))
    return summary


def _scale_K(K: np.ndarray, src_hw: Tuple[int, int], dst_hw: Tuple[int, int]) -> np.ndarray:
    K = np.asarray(K, dtype=np.float64).reshape(3, 3).copy()
    src_h, src_w = int(src_hw[0]), int(src_hw[1])
    dst_h, dst_w = int(dst_hw[0]), int(dst_hw[1])
    if src_w <= 0 or src_h <= 0 or (src_h, src_w) == (dst_h, dst_w):
        return K
    K[0, :] *= dst_w / float(src_w)
    K[1, :] *= dst_h / float(src_h)
    return K


def _resolve_view_meta(meta: dict, view: Optional[str] = None) -> dict:
    views = meta.get("views") or {}
    if view and view in views:
        return views[view]
    if meta.get("K") is not None and (meta.get("E_cam2world") or meta.get("E")) is not None:
        return meta
    return iter_views(meta)[0][1]


def render_action_video_thwc(
    action: np.ndarray,
    meta: dict,
    *,
    height: int,
    width: int,
    urdf_path: Optional[Path] = None,
    base_pos: Optional[Sequence[float]] = None,
    base_quat: Optional[Sequence[float]] = None,
    view: Optional[str] = None,
) -> np.ndarray:
    """White-bg RoboCasa skeleton ``[T, H, W, 3]`` from ``action.npy`` + K/E."""
    actions = np.asarray(action, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] < 8:
        raise ValueError(f"expected action (T,8), got {actions.shape}")
    T = int(actions.shape[0])
    vmeta = _resolve_view_meta(meta, view)
    src_w = int(vmeta.get("width") or meta.get("width") or width)
    src_h = int(vmeta.get("height") or meta.get("height") or height)
    K = _scale_K(np.asarray(vmeta["K"], dtype=np.float64), (src_h, src_w), (height, width))
    E = _as_E44(vmeta.get("E_cam2world") or vmeta.get("E"))
    if E is None:
        raise ValueError("RoboCasa view missing E_cam2world")
    pos, quat = resolve_base(meta, base_pos, base_quat)
    Twb = base_T(pos, quat)
    fk = UrdfFk(str(Path(urdf_path) if urdf_path is not None else DEFAULT_URDF))
    close_frac = 1.0 - np.clip(actions[:, 7] / GRIPPER_MAX_GAP, 0.0, 1.0)
    closed_m, settled_m = _gripper_state_masks(close_frac)
    frames = np.empty((T, height, width, 3), dtype=np.uint8)
    for t in range(T):
        pts = fk.points_world(action_to_cfg(actions[t]), Twb)
        names = [n for n in SKELETON_LINKS if n in pts]
        arr = np.array([pts[n] for n in names], dtype=np.float64)
        uv, depth = _project(arr, E, K)
        name2uv = {n: uv[i] for i, n in enumerate(names)}
        name2z = {n: float(depth[i]) for i, n in enumerate(names)}
        gb = _gripper_bone_color(bool(closed_m[t]), bool(settled_m[t]))
        gj = _gripper_joint_color(bool(closed_m[t]), bool(settled_m[t]))
        canvas = np.full((height, width, 3), 255, dtype=np.uint8)
        frames[t] = _draw_skeleton(name2uv, name2z, canvas, gb, gj)
    return frames


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render RoboCasa AC from action.npy + URDF + camera E/K")
    p.add_argument("--demo-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    p.add_argument("--save-mode", choices=["action", "overlay", "overlay_and_action", "all"],
                   default="overlay_and_action")
    p.add_argument("--base-pos", type=float, nargs=3, default=None)
    p.add_argument("--base-quat", type=float, nargs=4, default=None,
                   help="robot0_link0 quaternion wxyz")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    demo_dir = args.demo_dir.resolve()
    out_dir = (args.output_dir or (demo_dir / "urdf_ac")).resolve()
    urdf = args.urdf.resolve()
    if not (demo_dir / "action.npy").is_file():
        raise FileNotFoundError(demo_dir / "action.npy")
    if not urdf.is_file():
        raise FileNotFoundError(urdf)
    summary = render_demo(
        demo_dir, out_dir, urdf, args.save_mode, args.base_pos, args.base_quat,
    )
    print(json.dumps({
        "out_dir": str(out_dir),
        "frames": summary["frames"],
        "base_pos": summary["base_pos"],
        "base_quat_wxyz": summary["base_quat_wxyz"],
        "views": list(summary["views"].keys()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
