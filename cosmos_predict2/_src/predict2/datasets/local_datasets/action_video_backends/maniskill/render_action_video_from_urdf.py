#!/usr/bin/env python3
"""Render ManiSkill AC videos from action.npy + Panda URDF + camera K/E.

Does not open SAPIEN. Per demo directory:

  * ``action.npy``  (T, 8) = 7 arm joints (rad) + fingertip gap (m)
  * ``metadata.json`` ``K`` plus either ``E_world2cam`` (OpenCV 3x4),
    ``camera_eye``/``camera_target``, or ``camera_preset`` (``classic_3q``)
  * optional ``base_pos`` (world xyz of ``panda_link0``); tabletop default
    is ``(-0.615, 0, 0)``
  * Panda URDF FK, then a rigid identity-rotation base
  * fingertip = finger-link origin + local +z * 0.0455 m (same as the sim renderer)

Usage:
    python visualization_urdf/maniskill/render_action_video_from_urdf.py \\
        --demo-dir /path/to/PickCube-v1/traj_0 \\
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

REPO = Path(__file__).resolve().parents[2]
DEFAULT_DEMO = REPO / (
    "delivery/overlay_preview_20260911/maniskill3/preview"
)

# TableSceneBuilder Panda mount (identity rotation).
DEFAULT_BASE_POS = (-0.615, 0.0, 0.0)

CLASSIC_3Q_EYE = (0.6, 0.7, 0.6)
CLASSIC_3Q_TARGET = (0.0, 0.0, 0.35)
# SAPIEN cam axes (forward, left, up) = (x, y, z) -> OpenCV (x right, y down, z fwd)
SAPIEN2OPENCV = np.array(
    [
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
FINGERTIP_LOCAL_OFFSET = np.array([0.0, 0.0, 0.0455], dtype=np.float64)


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
GRIPPER_LEFT_SYN = "grip_L"
GRIPPER_RIGHT_SYN = "grip_R"
FINGER_LINKS = {
    GRIPPER_LEFT_SYN: "panda_leftfinger",
    GRIPPER_RIGHT_SYN: "panda_rightfinger",
}
ARM_BONES = [(ARM_LINKS[i], ARM_LINKS[i + 1]) for i in range(len(ARM_LINKS) - 1)]
GRIPPER_BONES = [("panda_hand", GRIPPER_LEFT_SYN), ("panda_hand", GRIPPER_RIGHT_SYN)]
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
GRIPPER_STATE_PLATEAU_EPS = 0.03
GRIPPER_STATE_CHANGE_DELTA = 0.1
GRIPPER_STATE_SETTLED_FILL = 3
GRIPPER_EARLY_CLOSED_FRAC = 0.85


def look_at_sapien(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """cam2world in SAPIEN convention: columns = (forward, left, up)."""
    eye = np.asarray(eye, dtype=np.float64).reshape(3)
    target = np.asarray(target, dtype=np.float64).reshape(3)
    up = np.asarray(up, dtype=np.float64).reshape(3)
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    up = up / np.linalg.norm(up)
    left = np.cross(up, forward)
    left = left / np.linalg.norm(left)
    up = np.cross(forward, left)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.stack([forward, left, up], axis=1)
    T[:3, 3] = eye
    return T


def world2cam_opencv(eye, target) -> np.ndarray:
    """OpenCV world->cam 3x4, same as ManiSkill get_extrinsic_matrix()."""
    T = look_at_sapien(eye, target)
    return (SAPIEN2OPENCV @ np.linalg.inv(T))[:3, :4]


def camera_from_preset(preset: str) -> Tuple[List[float], List[float]]:
    name = (preset or "classic_3q").strip()
    if name in ("", "default", "classic_3q"):
        return list(CLASSIC_3Q_EYE), list(CLASSIC_3Q_TARGET)
    raise ValueError(f"unsupported camera_preset={preset!r}; expected classic_3q")


def _as_xyz(value) -> Optional[List[float]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 3:
        if any(v is None for v in value):
            return None
        return [float(v) for v in value]
    return None


def _as_E(value) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape == (3, 4):
        return arr
    if arr.shape == (4, 4):
        return arr[:3, :4]
    return None


def resolve_camera(meta: dict) -> Tuple[np.ndarray, List[float], List[float], str]:
    E = _as_E(meta.get("E_world2cam") or meta.get("E"))
    eye = _as_xyz(meta.get("camera_eye"))
    target = _as_xyz(meta.get("camera_target"))
    if E is not None:
        if eye is None or target is None:
            eye, target = list(CLASSIC_3Q_EYE), list(CLASSIC_3Q_TARGET)
        return E, eye, target, "metadata_E"
    if eye is not None and target is not None:
        return world2cam_opencv(eye, target), eye, target, "metadata_lookat"
    preset = meta.get("camera_preset") or "classic_3q"
    eye, target = camera_from_preset(str(preset))
    return world2cam_opencv(eye, target), eye, target, str(preset)


def resolve_base_pos(meta: dict, cli_base: Optional[Sequence[float]] = None) -> List[float]:
    if cli_base is not None:
        return [float(x) for x in cli_base]
    for key in ("base_pos", "panda_base_pos"):
        xyz = _as_xyz(meta.get(key))
        if xyz is not None:
            return xyz
    return [float(x) for x in DEFAULT_BASE_POS]


def base_T(pos: Sequence[float]) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = np.asarray(pos, dtype=np.float64)
    return T


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
        for lk in ARM_LINKS:
            try:
                Tb = self.urdf.get_transform(lk, self.base)
            except Exception:
                continue
            p = T_world_base @ np.array([Tb[0, 3], Tb[1, 3], Tb[2, 3], 1.0])
            out[lk] = p[:3]
        for syn, lk in FINGER_LINKS.items():
            try:
                Tb = self.urdf.get_transform(lk, self.base)
            except Exception:
                continue
            Tw = T_world_base @ Tb
            tip = Tw[:3, 3] + Tw[:3, :3] @ FINGERTIP_LOCAL_OFFSET
            out[syn] = tip
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


def _gripper_state_masks(close_frac_side: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ov = np.clip(np.asarray(close_frac_side, dtype=np.float64), 0.0, 1.0)
    t_len = int(ov.shape[0])
    if t_len == 0:
        empty = np.zeros(0, dtype=bool)
        return empty, empty
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


def _project(points_world: np.ndarray, E_world2cam: np.ndarray,
             K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = points_world.shape[0]
    homo = np.concatenate([points_world, np.ones((n, 1))], axis=1)
    cam = (E_world2cam @ homo.T).T
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


def render_demo(
    demo_dir: Path,
    out_dir: Path,
    urdf_path: Path,
    save_mode: str,
    base_pos: Optional[Sequence[float]] = None,
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
    K = np.asarray(meta["K"], dtype=np.float64)
    E, eye, target, cam_src = resolve_camera(meta)
    base_xyz = resolve_base_pos(meta, base_pos)
    Twb = base_T(base_xyz)

    fk = UrdfFk(str(urdf_path))
    close_frac = 1.0 - np.clip(actions[:, 7] / GRIPPER_MAX_GAP, 0.0, 1.0)
    closed_m, settled_m = _gripper_state_masks(close_frac)

    want_av = save_mode in ("action", "overlay_and_action", "all")
    want_ov = save_mode in ("overlay", "overlay_and_action", "all")
    rgb_frames: List[np.ndarray] = []
    if want_ov:
        rgb_path = demo_dir / "rgb.mp4"
        if not rgb_path.is_file():
            raise FileNotFoundError(f"overlay requested but missing {rgb_path}")
        rgb_frames = read_mp4_rgb(rgb_path)

    point_names = ARM_LINKS + [GRIPPER_LEFT_SYN, GRIPPER_RIGHT_SYN]
    av_frames: List[np.ndarray] = []
    ov_frames: List[np.ndarray] = []
    for t in range(T):
        pts = fk.points_world(action_to_cfg(actions[t]), Twb)
        names = [n for n in point_names if n in pts]
        arr = np.array([pts[n] for n in names], dtype=np.float64)
        uv, depth = _project(arr, E, K)
        name2uv = {n: uv[i] for i, n in enumerate(names)}
        name2z = {n: float(depth[i]) for i, n in enumerate(names)}
        gb = _gripper_bone_color(bool(closed_m[t]), bool(settled_m[t]))
        gj = _gripper_joint_color(bool(closed_m[t]), bool(settled_m[t]))
        if want_av:
            canvas = np.full((height, width, 3), 255, dtype=np.uint8)
            av_frames.append(_draw_skeleton(name2uv, name2z, canvas, gb, gj))
        if want_ov:
            base = rgb_frames[min(t, len(rgb_frames) - 1)].copy()
            if base.shape[0] != height or base.shape[1] != width:
                base = cv2.resize(base, (width, height), interpolation=cv2.INTER_AREA)
            ov_frames.append(_draw_skeleton(name2uv, name2z, base, gb, gj))

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "demo_dir": str(demo_dir),
        "frames": T,
        "K": K.tolist(),
        "camera_eye": eye,
        "camera_target": target,
        "camera_source": cam_src,
        "E_world2cam": E.tolist(),
        "base_pos": list(map(float, base_xyz)),
        "urdf": str(urdf_path),
    }
    np.save(out_dir / "E_world2cam.npy", E)
    (out_dir / "render_meta.json").write_text(json.dumps(summary, indent=2))

    sim_av_path = demo_dir / "action_video_sim.mp4"
    if not sim_av_path.is_file():
        sim_av_path = demo_dir / "action_video.mp4"
    orig_av: List[np.ndarray] = []
    if sim_av_path.is_file() and sim_av_path.resolve() != (out_dir / "action_video.mp4").resolve():
        orig_av = read_mp4_rgb(sim_av_path)

    if want_av:
        ok, err = _encode_video(str(out_dir / "action_video.mp4"), av_frames, fps)
        if not ok:
            raise RuntimeError(err)
    if want_ov:
        ok, err = _encode_video(str(out_dir / "overlay.mp4"), ov_frames, fps)
        if not ok:
            raise RuntimeError(err)

    n_ref = len(av_frames) or len(ov_frames)
    idxs = sorted({int(np.clip(round(f * (n_ref - 1)), 0, n_ref - 1)) for f in still_frac})
    still_dir = out_dir / "stills"
    for i in idxs:
        if want_av and i < len(av_frames):
            save_still(still_dir / f"ac_{i:03d}.jpg", av_frames[i])
            if i < len(orig_av):
                pair = np.concatenate([orig_av[i], av_frames[i]], axis=1)
                save_still(still_dir / f"ac_pair_{i:03d}.jpg", pair)
        if want_ov and i < len(ov_frames):
            save_still(still_dir / f"ov_{i:03d}.jpg", ov_frames[i])
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


def render_action_video_thwc(
    action: np.ndarray,
    meta: dict,
    *,
    height: int,
    width: int,
    urdf_path: Optional[Path] = None,
    base_pos: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """White-bg ManiSkill skeleton ``[T, H, W, 3]`` from ``action.npy`` + K/E."""
    actions = np.asarray(action, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] < 8:
        raise ValueError(f"expected action (T,8), got {actions.shape}")
    T = int(actions.shape[0])
    src_w = int(meta.get("width") or width)
    src_h = int(meta.get("height") or height)
    K = _scale_K(np.asarray(meta["K"], dtype=np.float64), (src_h, src_w), (height, width))
    E, _eye, _target, _cam_src = resolve_camera(meta)
    base_xyz = resolve_base_pos(meta, base_pos)
    Twb = base_T(base_xyz)
    fk = UrdfFk(str(Path(urdf_path) if urdf_path is not None else DEFAULT_URDF))
    close_frac = 1.0 - np.clip(actions[:, 7] / GRIPPER_MAX_GAP, 0.0, 1.0)
    closed_m, settled_m = _gripper_state_masks(close_frac)
    frames = np.empty((T, height, width, 3), dtype=np.uint8)
    for t in range(T):
        pts = fk.points_world(action_to_cfg(actions[t]), Twb)
        names = [n for n in list(ARM_LINKS) + list(FINGER_LINKS) if n in pts]
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
    p = argparse.ArgumentParser(description="Render ManiSkill AC from action.npy + URDF + camera E/K")
    p.add_argument("--demo-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=None,
                   help="default: <demo-dir>/urdf_ac")
    p.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    p.add_argument("--save-mode", choices=["action", "overlay", "overlay_and_action", "all"],
                   default="overlay_and_action")
    p.add_argument("--base-pos", type=float, nargs=3, default=None,
                   help="Panda base xyz in world; default: metadata.base_pos or -0.615 0 0")
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
    summary = render_demo(demo_dir, out_dir, urdf, args.save_mode, args.base_pos)
    keys = ("frames", "base_pos", "camera_source", "camera_eye", "camera_target", "urdf")
    print(json.dumps({"out_dir": str(out_dir), **{k: summary[k] for k in keys}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
