"""LIBERO (MuJoCo/robosuite) 回放渲染器 -> GT rgb + action 骨架 + action 数值.

对齐 ManiSkill 管线的单视角产物 (visualization_urdf/maniskill/render_action_video_maniskill.py):
  * rgb.mp4, action_video.mp4, overlay.mp4, action.npy, metadata.json
  * 布局: <output_root>/<suite>/<task>/<demo_id>/

回放思路对齐 RoboCasa (visualization_urdf/robocasa/render_action_video_robocasa.py):
  * 每条 episode 先 reset_from_xml_string(model_file) 一次
  * 逐帧 set_state_from_flattened(states[t]) + sim.forward()
  * 骨架: body xpos + robosuite camera_utils 内外参投影 (E 为 cam2world, _project 内求逆)
  * 仅 agentview 单相机

夹爪上色用 ManiSkill 时序状态机 (非 RoboCasa 渐变).

用法:
    MUJOCO_GL=egl /media/datasets/zhangshilin/envs/libero/bin/python \\
        visualization_urdf/libero/render_action_video_libero.py \\
        --suites libero_spatial --limit-per-task 2 --save-mode all

分难度视角小验证集 (7 task x 5 ep x 6 views = 210), 多进程:
    MUJOCO_GL=egl .../python visualization_urdf/libero/render_action_video_libero.py \\
        --selection-manifest visualization_urdf/libero/view_gen_val_v1.json \\
        --view-difficulty all --num-workers 4 --resume --save-mode rgb_action \\
        --output-root /media/datasets/zhangshilin/libero_view_gen_val

    # 或: bash visualization_urdf/libero/launch_view_gen_val.sh
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

# ---- Headless GL must be configured BEFORE cv2 / mujoco / OpenGL imports ----
os.environ["MUJOCO_GL"] = os.environ.get("MUJOCO_GL", "egl")
os.environ["PYOPENGL_PLATFORM"] = "osmesa" if os.environ["MUJOCO_GL"] == "osmesa" else "egl"
# X11/Wayland sessions poison PyOpenGL platform selection on cluster images.
os.environ.pop("DISPLAY", None)
os.environ.pop("WAYLAND_DISPLAY", None)
os.environ.pop("XDG_SESSION_TYPE", None)
if "MUJOCO_EGL_DEVICE_ID" not in os.environ:
    os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"

# Preload a *non-empty* EGL library. Cluster images often ship 0-byte
# libEGL_nvidia stubs when NVIDIA_DRIVER_CAPABILITIES lacks "graphics";
# loading those poisons PyOpenGL (PLATFORM.EGL is None → eglQueryString).
def _preload_egl() -> None:
    import ctypes
    from ctypes.util import find_library

    nvidia_egl = "/usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0"
    mesa_icd = "/usr/share/glvnd/egl_vendor.d/50_mesa.json"
    nvidia_stub = os.path.isfile(nvidia_egl) and os.path.getsize(nvidia_egl) == 0
    if nvidia_stub and os.path.isfile(mesa_icd) and not os.environ.get(
        "__EGL_VENDOR_LIBRARY_FILENAMES"
    ):
        # Force GLVND away from the empty NVIDIA vendor before OpenGL import.
        os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = mesa_icd

    candidates = [
        "/usr/lib/x86_64-linux-gnu/libEGL.so.1",
        find_library("EGL"),
        "libEGL.so.1",
        # Only try NVIDIA if the .so is non-empty.
        nvidia_egl if not nvidia_stub else None,
        "libEGL_nvidia.so.0" if not nvidia_stub else None,
    ]
    for name in candidates:
        if not name:
            continue
        if os.path.isabs(name) and os.path.isfile(name) and os.path.getsize(name) == 0:
            continue
        try:
            ctypes.CDLL(name, mode=ctypes.RTLD_GLOBAL)
            return
        except OSError:
            continue


if os.environ["MUJOCO_GL"] != "osmesa":
    _preload_egl()


def _ensure_torch_ld_path() -> None:
    """Make torch shared libs resolvable before libero.benchmark does `import torch`.

    Cluster launchers prepend /usr/lib/x86_64-linux-gnu for EGL; if that sits ahead of
    (or without) torch/lib, torch fails with missing libtorch.so and collect_jobs → 0.
    Changing LD_LIBRARY_PATH after process start is not enough on Linux, so also
    RTLD_GLOBAL-preload the torch lib directory.
    """
    try:
        import ctypes
        import glob as _glob
        import importlib.util

        spec = importlib.util.find_spec("torch")
        if spec is None or not spec.origin:
            return
        torch_lib = os.path.join(os.path.dirname(spec.origin), "lib")
        if not os.path.isdir(torch_lib):
            return
        cur = os.environ.get("LD_LIBRARY_PATH", "")
        parts = [p for p in cur.split(":") if p]
        parts = [p for p in parts if os.path.abspath(p) != os.path.abspath(torch_lib)]
        os.environ["LD_LIBRARY_PATH"] = ":".join([torch_lib, *parts])
        # Prefer CPU/core libs first; ignore optional CUDA stubs that may be absent.
        for name in (
            "libgomp-*.so*",
            "libcudart*.so*",
            "libc10.so",
            "libtorch_cpu.so",
            "libtorch.so",
            "libtorch_python.so",
        ):
            for path in sorted(_glob.glob(os.path.join(torch_lib, name))):
                try:
                    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    continue
    except Exception:
        return


_ensure_torch_ld_path()

import h5py  # noqa: E402
import numpy as np  # noqa: E402
# cv2 is heavy and may touch GL; import lazily in encode/draw paths.


def _ensure_libero_config(
    libero_src: str = "",
    datasets: str = "",
) -> str:
    """Write ~/.libero/config.yaml (or $LIBERO_CONFIG_PATH) without interactive prompts.

    Cluster/headless jobs have no TTY; LIBERO's first import otherwise calls input() and
    raises EOFError, leaving zero jobs.
    """
    import yaml  # local import; available via libero env

    cfg_root = os.environ.get("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero"))
    os.makedirs(cfg_root, exist_ok=True)
    cfg_file = os.path.join(cfg_root, "config.yaml")
    if not libero_src:
        import importlib.util
        spec = importlib.util.find_spec("libero")
        locations = list(spec.submodule_search_locations or []) if spec else []
        if not locations:
            raise RuntimeError("Install LIBERO and set LIBERO_SRC to its source checkout")
        # The upstream source layout is LIBERO/libero/libero/{assets,bddl_files}.
        libero_src = os.path.dirname(locations[0])
    bench_root = os.path.join(libero_src, "libero", "libero")
    if not os.path.isdir(os.path.join(bench_root, "assets")):
        raise RuntimeError("LIBERO assets missing; set LIBERO_SRC to the LIBERO checkout")
    desired = {
        "benchmark_root": bench_root,
        "bddl_files": os.path.join(bench_root, "bddl_files"),
        "init_states": os.path.join(bench_root, "init_files"),
        "datasets": os.path.abspath(datasets),
        "assets": os.path.join(bench_root, "assets"),
    }
    need_write = True
    if os.path.isfile(cfg_file):
        try:
            with open(cfg_file, "r") as f:
                cur = yaml.load(f.read(), Loader=yaml.FullLoader) or {}
            need_write = any(cur.get(k) != v for k, v in desired.items())
        except Exception:  # noqa: BLE001
            need_write = True
    if need_write:
        with open(cfg_file, "w") as f:
            yaml.dump(desired, f, default_flow_style=False)
        print(f"[libero-config] wrote {cfg_file}", flush=True)
        for k, v in desired.items():
            print(f"  {k}: {v}", flush=True)
    else:
        print(f"[libero-config] ok {cfg_file}", flush=True)
    for k, v in desired.items():
        if not os.path.exists(v):
            print(f"[libero-config] WARN missing {k}: {v}", flush=True)
    return cfg_file


CAMERA_NAME = "agentview"
# 斜上方第三人称: look-at 偏向臂中部
# (LIBERO Panda base ≈ [-0.66, 0, 0.91])
OBLIQUE45_TARGET = np.array([-0.32, 0.0, 1.00], dtype=np.float64)
OBLIQUE45_DIST = 1.75
OBLIQUE45_ELEV_DEG = 25.0
OBLIQUE45_AZIM_DEG = 45.0
# 仅降低机位高度: 俯仰/方位/距离与 oblique45 相同, look-at 与相机同步下移
OBLIQUE45_LOW_DZ = 0.55  # 相对 oblique45 下移量 (m)
OBLIQUE45_LOW_TARGET = OBLIQUE45_TARGET + np.array([0.0, 0.0, -OBLIQUE45_LOW_DZ], dtype=np.float64)
OBLIQUE45_LOW_DIST = OBLIQUE45_DIST
OBLIQUE45_LOW_ELEV_DEG = OBLIQUE45_ELEV_DEG
OBLIQUE45_LOW_AZIM_DEG = OBLIQUE45_AZIM_DEG

# 视角泛化: hard 档相机局部水平 / 世界竖直位移 (m)
VIEW_HARD_HORIZ_M = 0.20
VIEW_HARD_VERT_M = 0.15

DEFAULT_SUITES = [
    "libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90",
]
VIEW_DIFFICULTIES = ("easy", "medium", "hard", "ood")
LEGACY_CAMERA_PRESETS = ("default", "oblique45", "oblique45_low")
# Suite-specific height baseline applied *before* view-gen azimuth/shift.
# libero_object historically used oblique45_low (look-at + eye 同步下移).
SUITE_BASELINE_TARGET_DZ: Dict[str, float] = {
    "libero_object": -OBLIQUE45_LOW_DZ,
}


@dataclass(frozen=True)
class CameraPresetSpec:
    """Deterministic agentview override relative to oblique45 baseline."""

    name: str
    difficulty: Optional[str]  # easy|medium|hard|ood|None (legacy)
    azim_offset_deg: float = 0.0
    elev_offset_deg: float = 0.0
    dist_scale: float = 1.0
    horiz_shift_m: float = 0.0  # camera-local +X (right)
    vert_shift_m: float = 0.0   # world +Z
    # Legacy low-view: shift both eye and target down in world Z before spherical offset.
    target_dz: float = 0.0
    keep_default: bool = False


def _build_camera_presets() -> Dict[str, CameraPresetSpec]:
    specs = [
        CameraPresetSpec(name="default", difficulty=None, keep_default=True),
        CameraPresetSpec(name="oblique45", difficulty=None),
        CameraPresetSpec(name="oblique45_low", difficulty=None, target_dz=-OBLIQUE45_LOW_DZ),
        CameraPresetSpec(name="easy_left", difficulty="easy", azim_offset_deg=-15.0),
        CameraPresetSpec(name="easy_right", difficulty="easy", azim_offset_deg=15.0),
        CameraPresetSpec(name="medium_left", difficulty="medium", azim_offset_deg=-30.0),
        CameraPresetSpec(name="medium_right", difficulty="medium", azim_offset_deg=30.0),
        CameraPresetSpec(
            name="hard_left_up", difficulty="hard",
            azim_offset_deg=-45.0,
            horiz_shift_m=-VIEW_HARD_HORIZ_M,
            vert_shift_m=VIEW_HARD_VERT_M,
        ),
        CameraPresetSpec(
            name="hard_right_down", difficulty="hard",
            azim_offset_deg=45.0,
            horiz_shift_m=VIEW_HARD_HORIZ_M,
            vert_shift_m=-VIEW_HARD_VERT_M,
        ),
        # OOD showcase: 5 views on different axes (orbit / elev / dist)
        CameraPresetSpec(name="orbit_left", difficulty="ood", azim_offset_deg=-50.0),
        CameraPresetSpec(name="orbit_right", difficulty="ood", azim_offset_deg=50.0),
        CameraPresetSpec(name="high_down", difficulty="ood", elev_offset_deg=22.0),
        CameraPresetSpec(name="low_side", difficulty="ood", elev_offset_deg=-18.0),
        CameraPresetSpec(name="pullback", difficulty="ood", dist_scale=1.40),
    ]
    out = {s.name: s for s in specs}
    if len(out) != len(specs):
        raise RuntimeError("duplicate camera preset names")
    return out


CAMERA_PRESETS: Dict[str, CameraPresetSpec] = _build_camera_presets()
VIEW_GEN_PRESETS: Dict[str, Tuple[str, ...]] = {
    "easy": ("easy_left", "easy_right"),
    "medium": ("medium_left", "medium_right"),
    "hard": ("hard_left_up", "hard_right_down"),
    "ood": ("orbit_left", "orbit_right", "high_down", "low_side", "pullback"),
}
# --view-difficulty all stays the original 6 paired views (does not include ood).
PAIRED_VIEW_DIFFICULTIES: Tuple[str, ...] = ("easy", "medium", "hard")
ALL_VIEW_GEN_PRESETS: Tuple[str, ...] = (
    VIEW_GEN_PRESETS["easy"] + VIEW_GEN_PRESETS["medium"] + VIEW_GEN_PRESETS["hard"]
)
ALL_CAMERA_PRESET_NAMES: Tuple[str, ...] = tuple(CAMERA_PRESETS.keys())
GRIPPER_MAX_GAP = 0.08
CONTROL_FREQ = 20.0

# 臂链候选: RoboCasa 命名 + 常见 Panda 变体
ARM_LINK_CANDIDATES = [
    [
        "robot0_link0", "robot0_link1", "robot0_link2", "robot0_link3",
        "robot0_link4", "robot0_link5", "robot0_link6", "robot0_link7",
        "robot0_right_hand",
    ],
    [
        "panda_link0", "panda_link1", "panda_link2", "panda_link3",
        "panda_link4", "panda_link5", "panda_link6", "panda_link7",
        "panda_hand",
    ],
]
GRIPPER_HAND_CANDS = [
    "robot0_right_hand", "gripper0_eef", "gripper0_right_gripper", "panda_hand",
]
# 骨架绘制优先指尖; 开合信号另用 finger 体 (下方 GAP_*)。
GRIPPER_LEFT_CANDS = [
    "gripper0_finger_joint1_tip",
    "gripper0_leftfinger",
    "gripper0_right_leftfinger",
    "panda_leftfinger",
    "robot0_leftfinger",
]
GRIPPER_RIGHT_CANDS = [
    "gripper0_finger_joint2_tip",
    "gripper0_rightfinger",
    "gripper0_right_rightfinger",
    "panda_rightfinger",
    "robot0_rightfinger",
]
# 开合距离用 finger 体, tip 接触物体时更抖、易导致配色闪烁。
GRIPPER_GAP_LEFT_CANDS = [
    "gripper0_leftfinger",
    "gripper0_right_leftfinger",
    "gripper0_finger_joint1_tip",
    "panda_leftfinger",
]
GRIPPER_GAP_RIGHT_CANDS = [
    "gripper0_rightfinger",
    "gripper0_right_rightfinger",
    "gripper0_finger_joint2_tip",
    "panda_rightfinger",
]

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
GRIPPER_STATE_LOOKAHEAD = 5         # 仅未来 5 帧判定平台稳定
GRIPPER_STATE_LOOKBACK = 0
GRIPPER_STATE_PLATEAU_EPS = 0.05    # LIBERO gap 噪声略大, 放宽平台判定
GRIPPER_STATE_CHANGE_DELTA = 0.12
GRIPPER_STATE_SETTLED_FILL = 8      # 填补更长短 unsettled, 抑灰/彩闪烁
GRIPPER_STATE_MIN_SETTLED_RUN = 5   # 短于该长度的 settled 岛视为运动
GRIPPER_STATE_SMOOTH_K = 5          # close_frac 中值平滑窗口 (奇数)
GRIPPER_EARLY_CLOSED_FRAC = 0.75


@dataclass
class EpisodeJob:
    suite: str
    task: str
    task_idx: int
    demo_id: str
    h5_path: str
    bddl_path: str
    lang: str


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
    """短于 min_run 的 True 段改成 False (去掉短暂 settled 岛, 抑配色闪烁)。"""
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


def _gripper_state_masks(
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
) -> Tuple[np.ndarray, np.ndarray]:
    ov = np.clip(np.asarray(close_frac_side, dtype=np.float64), 0.0, 1.0)
    t_len = int(ov.shape[0])
    if t_len == 0:
        empty = np.zeros(0, dtype=bool)
        return empty, empty
    ov = _median_smooth_1d(ov, smooth_k)
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

    # 先去掉短暂 settled 岛, 再填补短 unsettled 空隙
    settled = _suppress_short_true_runs(settled, min_settled_run)
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


def _gripper_bone_color(closed: bool, settled: bool) -> tuple:
    if not settled:
        return COLOR_GRIPPER_MOTION_BONE
    return COLOR_GRIPPER_CLOSED_BONE if closed else COLOR_GRIPPER_OPEN_BONE


def _gripper_joint_color(closed: bool, settled: bool) -> tuple:
    if not settled:
        return COLOR_GRIPPER_MOTION_JOINT
    return COLOR_GRIPPER_CLOSED_JOINT if closed else COLOR_GRIPPER_OPEN_JOINT


def _resolve_first(names: Sequence[str], pool: set) -> Optional[str]:
    for n in names:
        if n in pool:
            return n
    return None


def _resolve_skeleton(body_names: Sequence[str]) -> Tuple[List[str], List[Tuple[str, str]], List[Tuple[str, str]], str, str]:
    pool = set(body_names)
    arm_links: List[str] = []
    for chain in ARM_LINK_CANDIDATES:
        if all(n in pool for n in chain):
            arm_links = list(chain)
            break
    if not arm_links:
        best = []
        for chain in ARM_LINK_CANDIDATES:
            hit = [n for n in chain if n in pool]
            if len(hit) > len(best):
                best = hit
        arm_links = best

    hand = _resolve_first(GRIPPER_HAND_CANDS, pool) or (arm_links[-1] if arm_links else "")
    left = _resolve_first(GRIPPER_LEFT_CANDS, pool) or ""
    right = _resolve_first(GRIPPER_RIGHT_CANDS, pool) or ""

    arm_bones = [(arm_links[i], arm_links[i + 1]) for i in range(len(arm_links) - 1)]
    gripper_bones: List[Tuple[str, str]] = []
    if hand and left:
        gripper_bones.append((hand, left))
    if hand and right:
        gripper_bones.append((hand, right))

    point_names = list(dict.fromkeys(arm_links + [x for x in (left, right) if x]))
    return point_names, arm_bones, gripper_bones, left, right


def _build_ffmpeg_cmd(output_path: str, img_w: int, img_h: int, fps: float,
                      preset: str = "veryfast", crf: int = 20, threads: int = 2) -> List[str]:
    return [
        "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{img_w}x{img_h}", "-pix_fmt", "rgb24", "-r", f"{fps:g}", "-i", "-",
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-threads", str(threads), "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", "-f", "mp4", "-loglevel", "error", output_path,
    ]


def _encode_frames(cmd: List[str], frames: List[np.ndarray]) -> Tuple[int, str]:
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
    tail = ""
    try:
        for fr in frames:
            proc.stdin.write(np.ascontiguousarray(fr, dtype=np.uint8).tobytes())
        try:
            proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        proc.wait(timeout=600)
    except Exception as e:  # noqa: BLE001
        tail = f"exception: {e!r}"[-200:]
        try:
            proc.kill()
        except Exception:
            pass
    finally:
        th.join(timeout=5.0)
    if not tail:
        err = b"".join(chunks)
        if err:
            tail = err.decode(errors="replace")[-200:]
    return (proc.returncode if proc.returncode is not None else -1, tail)


def _encode_video(path: str, frames: List[np.ndarray], fps: float,
                  crf: int, preset: str) -> Tuple[bool, str]:
    if not frames:
        return False, "no frames"
    h, w = frames[0].shape[:2]
    tmp = path + ".part"
    rc, err = _encode_frames(_build_ffmpeg_cmd(tmp, w, h, fps, preset, crf), frames)
    if rc == 0 and os.path.exists(tmp) and os.path.getsize(tmp) > 0:
        os.replace(tmp, path)
        return True, ""
    if os.path.exists(tmp):
        try:
            os.remove(tmp)
        except OSError:
            pass
    return False, f"ffmpeg rc={rc}: {err}"


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
                   img: np.ndarray, arm_bones: List[Tuple[str, str]],
                   gripper_bones: List[Tuple[str, str]],
                   grip_bone, grip_joint) -> np.ndarray:
    import cv2

    def _ok(n):
        return n in name2z and name2z[n] > 0.01

    for a, b in arm_bones:
        if not (_ok(a) and _ok(b)):
            continue
        p1 = tuple(name2uv[a].astype(int))
        p2 = tuple(name2uv[b].astype(int))
        cv2.line(img, p1, p2, COLOR_ARM_BONE, ARM_BONE_WIDTH, cv2.LINE_AA)
        cv2.circle(img, p2, ARM_JOINT_RADIUS, COLOR_ARM_JOINT, -1, cv2.LINE_AA)
    for a, b in gripper_bones:
        if not (_ok(a) and _ok(b)):
            continue
        p1 = tuple(name2uv[a].astype(int))
        p2 = tuple(name2uv[b].astype(int))
        cv2.line(img, p1, p2, grip_bone, ARM_BONE_WIDTH, cv2.LINE_AA)
        cv2.circle(img, p2, ARM_JOINT_RADIUS, grip_joint, -1, cv2.LINE_AA)
    return img


def _rewrite_model_xml(xml_str: str) -> str:
    """Rewrite mesh/texture absolute paths from the demo author's machine to local assets.

    Official demos embed paths like
    ``/Users/yifengz/.../chiliocosm/assets/...``; map the suffix after ``/assets/``
    onto ``get_libero_path("assets")``. Also remaps ``robosuite`` asset paths like
    the upstream ``postprocess_model_xml``.
    """
    import xml.etree.ElementTree as ET

    import robosuite
    from libero.libero import get_libero_path
    from robosuite.utils.mjcf_utils import find_elements

    if isinstance(xml_str, bytes):
        xml_str = xml_str.decode("utf-8")

    assets_root = os.path.abspath(get_libero_path("assets"))
    robosuite_root = os.path.split(robosuite.__file__)[0]
    robosuite_parts = robosuite_root.split(os.sep)

    tree = ET.fromstring(xml_str)
    asset = tree.find("asset")
    if asset is None:
        return xml_str

    for elem in list(asset.findall("mesh")) + list(asset.findall("texture")):
        old_path = elem.get("file")
        if not old_path:
            continue
        if os.path.isfile(old_path):
            continue

        parts = old_path.split("/")
        if "robosuite" in parts:
            ind = max(i for i, v in enumerate(parts) if v == "robosuite")
            new_path = "/".join(robosuite_parts + parts[ind + 1 :])
            if os.path.isfile(new_path):
                elem.set("file", new_path)
                continue

        marker = "/assets/"
        if marker in old_path:
            rel = old_path.split(marker, 1)[1]
            new_path = os.path.join(assets_root, rel)
            if os.path.isfile(new_path):
                elem.set("file", new_path)
                continue

        # Last resort: basename search under assets (rare missing texture variants).
        base = os.path.basename(old_path)
        hits = []
        for root, _dirs, files in os.walk(assets_root):
            if base in files:
                hits.append(os.path.join(root, base))
                if len(hits) >= 2:
                    break
        if len(hits) == 1:
            elem.set("file", hits[0])

    return ET.tostring(tree, encoding="utf8").decode("utf8")


def _lookat_cam_quat_wxyz(eye: np.ndarray, target: np.ndarray,
                          up: Optional[np.ndarray] = None) -> np.ndarray:
    """MuJoCo camera quat (wxyz): camera looks along -Z, Y is up."""
    from robosuite.utils import transform_utils as T

    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64) if up is None else np.asarray(up, dtype=np.float64)
    forward = target - eye
    n = np.linalg.norm(forward)
    if n < 1e-8:
        raise ValueError("eye and target too close")
    forward /= n
    zaxis = -forward
    xaxis = np.cross(up, zaxis)
    xn = np.linalg.norm(xaxis)
    if xn < 1e-8:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        xaxis = np.cross(up, zaxis)
        xn = np.linalg.norm(xaxis)
    xaxis /= xn
    yaxis = np.cross(zaxis, xaxis)
    R = np.stack([xaxis, yaxis, zaxis], axis=1)
    return T.convert_quat(T.mat2quat(R), to="wxyz")


def _oblique_eye_target(
    target: np.ndarray, dist: float, elev_deg: float, azim_deg: float,
) -> Tuple[np.ndarray, np.ndarray]:
    elev = np.deg2rad(elev_deg)
    azim = np.deg2rad(azim_deg)
    d = float(dist)
    offset = d * np.array([
        np.cos(elev) * np.cos(azim),
        np.cos(elev) * np.sin(azim),
        np.sin(elev),
    ], dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64).copy()
    return tgt + offset, tgt


def _camera_right_axis(eye: np.ndarray, target: np.ndarray,
                       up: Optional[np.ndarray] = None) -> np.ndarray:
    """Camera-local +X (right) for a look-at camera (MuJoCo -Z forward, +Y up)."""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64) if up is None else np.asarray(up, dtype=np.float64)
    forward = target - eye
    n = np.linalg.norm(forward)
    if n < 1e-8:
        raise ValueError("eye and target too close")
    forward /= n
    zaxis = -forward
    xaxis = np.cross(up, zaxis)
    xn = np.linalg.norm(xaxis)
    if xn < 1e-8:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        xaxis = np.cross(up, zaxis)
        xn = np.linalg.norm(xaxis)
    return xaxis / xn


def resolve_camera_preset(preset: str) -> CameraPresetSpec:
    if preset not in CAMERA_PRESETS:
        raise ValueError(
            f"unknown camera preset: {preset}; choices={list(CAMERA_PRESETS)}")
    return CAMERA_PRESETS[preset]


def presets_for_difficulty(difficulty: str) -> Tuple[str, ...]:
    d = (difficulty or "").strip().lower()
    if d == "all":
        return ALL_VIEW_GEN_PRESETS
    if d not in VIEW_GEN_PRESETS:
        raise ValueError(
            f"unknown view difficulty: {difficulty}; "
            f"choices={list(VIEW_GEN_PRESETS)}|all")
    return VIEW_GEN_PRESETS[d]


def suite_baseline_target_dz(suite: Optional[str]) -> float:
    if not suite:
        return 0.0
    return float(SUITE_BASELINE_TARGET_DZ.get(str(suite), 0.0))


def compute_camera_eye_target(
    preset: str, baseline_target_dz: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, CameraPresetSpec]:
    """Return (eye, target, spec) for a named preset. ``default`` has no override pose.

    ``baseline_target_dz`` is an extra world-Z shift of look-at (and thus eye) applied
    before spherical placement — used so libero_object sits on the oblique45_low
    baseline, then view-gen ±azim / hard shifts are applied on top.
    """
    spec = resolve_camera_preset(preset)
    if spec.keep_default:
        raise ValueError("default preset keeps demo/env camera; no eye/target override")
    total_dz = float(spec.target_dz) + float(baseline_target_dz)
    target = OBLIQUE45_TARGET + np.array([0.0, 0.0, total_dz], dtype=np.float64)
    dist = OBLIQUE45_DIST * float(spec.dist_scale)
    elev = OBLIQUE45_ELEV_DEG + float(spec.elev_offset_deg)
    azim = OBLIQUE45_AZIM_DEG + float(spec.azim_offset_deg)
    eye, tgt = _oblique_eye_target(target, dist, elev, azim)
    if abs(spec.horiz_shift_m) > 1e-12 or abs(spec.vert_shift_m) > 1e-12:
        right = _camera_right_axis(eye, tgt)
        eye = eye + float(spec.horiz_shift_m) * right
        eye = eye + np.array([0.0, 0.0, float(spec.vert_shift_m)], dtype=np.float64)
    return eye, tgt, spec


def _oblique45_eye_target() -> Tuple[np.ndarray, np.ndarray]:
    eye, tgt, _ = compute_camera_eye_target("oblique45")
    return eye, tgt


def _oblique45_low_eye_target() -> Tuple[np.ndarray, np.ndarray]:
    eye, tgt, _ = compute_camera_eye_target("oblique45_low")
    return eye, tgt


def _apply_camera_preset(
    env, preset: str, baseline_target_dz: float = 0.0,
) -> Optional[Dict[str, object]]:
    """Override agentview pose after XML load. ``default`` keeps demo/env camera.

    Returns camera pose metadata dict.
    """
    spec = resolve_camera_preset(preset)
    if spec.keep_default:
        return {
            "camera_preset": spec.name,
            "view_difficulty": None,
            "camera_eye": None,
            "camera_target": None,
            "azim_offset_deg": 0.0,
            "elev_offset_deg": 0.0,
            "dist_scale": 1.0,
            "horiz_shift_m": 0.0,
            "vert_shift_m": 0.0,
            "baseline_target_dz": float(baseline_target_dz),
            "camera_baseline": "default",
        }
    eye, target, spec = compute_camera_eye_target(
        preset, baseline_target_dz=baseline_target_dz)
    quat = _lookat_cam_quat_wxyz(eye, target)
    cid = env.sim.model.camera_name2id(CAMERA_NAME)
    env.sim.model.cam_pos[cid][:] = eye
    env.sim.model.cam_quat[cid][:] = quat
    env.sim.forward()
    baseline_name = (
        "oblique45_low" if abs(float(baseline_target_dz) + OBLIQUE45_LOW_DZ) < 1e-9
        else ("oblique45" if abs(float(baseline_target_dz)) < 1e-9 else "custom")
    )
    return {
        "camera_preset": spec.name,
        "view_difficulty": spec.difficulty,
        "camera_eye": eye.tolist(),
        "camera_target": target.tolist(),
        "azim_offset_deg": float(spec.azim_offset_deg),
        "elev_offset_deg": float(spec.elev_offset_deg),
        "dist_scale": float(spec.dist_scale),
        "horiz_shift_m": float(spec.horiz_shift_m),
        "vert_shift_m": float(spec.vert_shift_m),
        "baseline_target_dz": float(baseline_target_dz),
        "camera_baseline": baseline_name,
    }


def check_camera_presets_geometry() -> List[str]:
    """Pure geometry sanity checks for view-gen presets. Returns error strings."""
    errs: List[str] = []
    all_named = tuple(n for names in VIEW_GEN_PRESETS.values() for n in names)
    if len(set(all_named)) != len(all_named):
        errs.append("duplicate view-gen preset names")
    if "close_offset" in CAMERA_PRESETS:
        errs.append("close_offset must not exist")
    ood_names = VIEW_GEN_PRESETS.get("ood", ())
    if ood_names != ("orbit_left", "orbit_right", "high_down", "low_side", "pullback"):
        errs.append(f"ood presets mismatch: {ood_names}")
    for d in PAIRED_VIEW_DIFFICULTIES:
        pair = VIEW_GEN_PRESETS[d]
        if len(pair) != 2:
            errs.append(f"{d} should have 2 presets, got {pair}")
            continue
        left, right = resolve_camera_preset(pair[0]), resolve_camera_preset(pair[1])
        if abs(left.azim_offset_deg + right.azim_offset_deg) > 1e-9:
            errs.append(f"{d} azim not symmetric: {left.azim_offset_deg}, {right.azim_offset_deg}")
        if d == "easy" and abs(abs(left.azim_offset_deg) - 15.0) > 1e-9:
            errs.append(f"easy azim should be ±15, got {left.azim_offset_deg}")
        if d == "medium" and abs(abs(left.azim_offset_deg) - 30.0) > 1e-9:
            errs.append(f"medium azim should be ±30, got {left.azim_offset_deg}")
        if d == "hard" and abs(abs(left.azim_offset_deg) - 45.0) > 1e-9:
            errs.append(f"hard azim should be ±45, got {left.azim_offset_deg}")
    for name in all_named:
        eye, target, spec = compute_camera_eye_target(name)
        if not np.all(np.isfinite(eye)) or not np.all(np.isfinite(target)):
            errs.append(f"{name}: non-finite eye/target")
        dist = float(np.linalg.norm(eye - target))
        if abs(dist - OBLIQUE45_DIST) > 1e-6 and name.startswith(("easy_", "medium_")):
            errs.append(f"{name}: expected dist={OBLIQUE45_DIST}, got {dist}")
        if name.startswith("hard_"):
            # After horiz/vert shifts, distance changes; still require look-at target fixed.
            if float(np.linalg.norm(target - OBLIQUE45_TARGET)) > 1e-9:
                errs.append(f"{name}: target drifted from arm center")
            if abs(abs(spec.horiz_shift_m) - VIEW_HARD_HORIZ_M) > 1e-9:
                errs.append(f"{name}: bad horiz shift {spec.horiz_shift_m}")
            if abs(abs(spec.vert_shift_m) - VIEW_HARD_VERT_M) > 1e-9:
                errs.append(f"{name}: bad vert shift {spec.vert_shift_m}")
        quat = _lookat_cam_quat_wxyz(eye, target)
        qn = float(np.linalg.norm(quat))
        if not np.all(np.isfinite(quat)) or abs(qn - 1.0) > 1e-5:
            errs.append(f"{name}: bad quat norm={qn}")
    # libero_object baseline: same as oblique45_low, then view-gen on top.
    obj_dz = suite_baseline_target_dz("libero_object")
    if abs(obj_dz + OBLIQUE45_LOW_DZ) > 1e-9:
        errs.append(f"libero_object baseline_dz should be {-OBLIQUE45_LOW_DZ}, got {obj_dz}")
    else:
        low_eye, low_tgt, _ = compute_camera_eye_target("oblique45_low")
        for name in ("easy_left", "easy_right"):
            eye, tgt, _ = compute_camera_eye_target(name, baseline_target_dz=obj_dz)
            if abs(float(tgt[2] - low_tgt[2])) > 1e-9:
                errs.append(f"object/{name}: target z {tgt[2]} != low baseline {low_tgt[2]}")
            if abs(float(eye[2] - low_eye[2])) > 1e-6:
                # Pure azimuth around lowered target keeps eye z equal to low baseline.
                errs.append(f"object/{name}: eye z {eye[2]} != low baseline {low_eye[2]}")
    # Left/right pairs: same target, same height & distance for pure-azimuth tiers.
    # Note: mid-Y is NOT equal to baseline for large ±azim (sin is nonlinear) — don't check that.
    for d in PAIRED_VIEW_DIFFICULTIES:
        pair = VIEW_GEN_PRESETS[d]
        e0, t0, s0 = compute_camera_eye_target(pair[0])
        e1, t1, s1 = compute_camera_eye_target(pair[1])
        if float(np.linalg.norm(t0 - t1)) > 1e-9:
            errs.append(f"{d}: left/right targets differ")
        if d in ("easy", "medium"):
            if abs(float(e0[2] - e1[2])) > 1e-6:
                errs.append(f"{d}: left/right heights differ: {e0[2]} vs {e1[2]}")
            d0 = float(np.linalg.norm(e0 - t0))
            d1 = float(np.linalg.norm(e1 - t1))
            if abs(d0 - d1) > 1e-6:
                errs.append(f"{d}: left/right distances differ: {d0} vs {d1}")
            if abs(s0.azim_offset_deg + s1.azim_offset_deg) > 1e-9:
                errs.append(f"{d}: azim offsets not opposite")
        if d == "hard":
            if abs(s0.horiz_shift_m + s1.horiz_shift_m) > 1e-9:
                errs.append("hard: horiz shifts not opposite")
            if abs(s0.vert_shift_m + s1.vert_shift_m) > 1e-9:
                errs.append("hard: vert shifts not opposite")
    # OOD showcase: distinct axes, shared look-at; no close/offset crop.
    base_eye, base_tgt, _ = compute_camera_eye_target("oblique45")
    ol_e, ol_t, ol_s = compute_camera_eye_target("orbit_left")
    or_e, or_t, or_s = compute_camera_eye_target("orbit_right")
    hd_e, hd_t, hd_s = compute_camera_eye_target("high_down")
    ls_e, ls_t, ls_s = compute_camera_eye_target("low_side")
    pb_e, pb_t, pb_s = compute_camera_eye_target("pullback")
    for name, tgt in (
        ("orbit_left", ol_t), ("orbit_right", or_t),
        ("high_down", hd_t), ("low_side", ls_t), ("pullback", pb_t),
    ):
        if float(np.linalg.norm(tgt - base_tgt)) > 1e-9:
            errs.append(f"ood/{name}: look-at drifted from oblique45")
    if abs(ol_s.azim_offset_deg + 50.0) > 1e-9 or abs(or_s.azim_offset_deg - 50.0) > 1e-9:
        errs.append(f"ood orbit azim should be ±50, got {ol_s.azim_offset_deg}, {or_s.azim_offset_deg}")
    if abs(float(ol_e[2] - or_e[2])) > 1e-6:
        errs.append(f"ood orbit heights differ: {ol_e[2]} vs {or_e[2]}")
    if abs(hd_s.elev_offset_deg - 22.0) > 1e-9:
        errs.append(f"high_down elev should be +22, got {hd_s.elev_offset_deg}")
    if abs(ls_s.elev_offset_deg + 18.0) > 1e-9:
        errs.append(f"low_side elev should be -18, got {ls_s.elev_offset_deg}")
    if float(hd_e[2]) <= float(base_eye[2]):
        errs.append("high_down eye z should be above baseline")
    if float(ls_e[2]) >= float(base_eye[2]):
        errs.append("low_side eye z should be below baseline")
    pb_dist = float(np.linalg.norm(pb_e - pb_t))
    if abs(pb_s.dist_scale - 1.40) > 1e-9 or abs(pb_dist - OBLIQUE45_DIST * 1.40) > 1e-6:
        errs.append(f"pullback dist should be 1.40x baseline, got scale={pb_s.dist_scale} dist={pb_dist}")
    return errs


def make_env(bddl_path: str, width: int, height: int):
    # Re-assert headless EGL (in case something reset env).
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ["PYOPENGL_PLATFORM"] = "osmesa" if os.environ["MUJOCO_GL"] == "osmesa" else "egl"
    os.environ.pop("DISPLAY", None)
    os.environ.pop("WAYLAND_DISPLAY", None)
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

    def _purge_gl_modules() -> None:
        doomed = [
            m for m in list(sys.modules)
            if m == "mujoco" or m.startswith(
                ("mujoco.", "OpenGL", "robosuite", "libero.libero.envs")
            )
        ]
        for m in doomed:
            sys.modules.pop(m, None)

    last_err: Optional[BaseException] = None
    for attempt in range(1, 6):
        try:
            from libero.libero.envs import OffScreenRenderEnv

            return OffScreenRenderEnv(
                bddl_file_name=bddl_path,
                camera_heights=height,
                camera_widths=width,
                camera_names=[CAMERA_NAME],
                use_camera_obs=False,
                has_renderer=False,
                has_offscreen_renderer=True,
                control_freq=int(CONTROL_FREQ),
            )
        except BaseException as e:  # noqa: BLE001
            last_err = e
            msg = repr(e)
            print(f"[make_env] attempt {attempt}/5 failed: {msg}", flush=True)
            transient = any(
                s in msg for s in ("eglQueryString", "EGL", "OpenGL", "GLContext", "display", "PLATFORM")
            )
            if not transient or attempt >= 5:
                break
            _purge_gl_modules()
            time.sleep(2.0 * attempt)
    assert last_err is not None
    raise last_err


def _arm_joint_indexes(env) -> np.ndarray:
    robot = env.robots[0]
    arm = robot._ref_joint_pos_indexes
    if isinstance(arm, dict):
        arm = arm.get("right", list(arm.values())[0])
    return np.asarray(arm)[:7]


def _init_episode(env, model_xml: str, state0: np.ndarray,
                  camera_preset: str = "default",
                  baseline_target_dz: float = 0.0) -> Optional[Dict[str, object]]:
    xml = _rewrite_model_xml(model_xml)
    env.reset_from_xml_string(xml)
    env.sim.reset()
    env.set_state(state0)
    env.sim.forward()
    return _apply_camera_preset(
        env, camera_preset, baseline_target_dz=baseline_target_dz)


def _set_frame(env, state: np.ndarray) -> None:
    env.set_state(state)
    env.sim.forward()


def _lang_from_h5(f: h5py.File, task_lang: str) -> str:
    raw = f["data"].attrs.get("problem_info", None)
    if raw:
        try:
            info = json.loads(raw) if isinstance(raw, (str, bytes)) else {}
            lang = info.get("language_instruction", "")
            if lang:
                return lang
        except Exception:
            pass
    return task_lang or ""


def render_episode(env, f: h5py.File, demo_id: str, width: int, height: int,
                   save_rgb: bool, save_av: bool, save_ov: bool,
                   camera_preset: str = "default",
                   baseline_target_dz: float = 0.0) -> dict:
    from robosuite.utils import camera_utils

    grp = f[f"data/{demo_id}"]
    states = grp["states"][()]
    T = int(states.shape[0])
    model_xml = grp.attrs["model_file"]

    cam_meta = _init_episode(
        env, model_xml, states[0],
        camera_preset=camera_preset,
        baseline_target_dz=baseline_target_dz,
    ) or {}
    point_names, arm_bones, gripper_bones, left, right = _resolve_skeleton(env.sim.model.body_names)
    body_ids = {n: env.sim.model.body_name2id(n) for n in point_names
                if n in env.sim.model.body_names}
    names = list(body_ids.keys())
    # 开合信号用 finger 体 (比 tip 稳); 骨架仍画 tip
    pool = set(env.sim.model.body_names)
    gap_left = _resolve_first(GRIPPER_GAP_LEFT_CANDS, pool) or left
    gap_right = _resolve_first(GRIPPER_GAP_RIGHT_CANDS, pool) or right
    if gap_left and gap_left not in body_ids:
        body_ids[gap_left] = env.sim.model.body_name2id(gap_left)
    if gap_right and gap_right not in body_ids:
        body_ids[gap_right] = env.sim.model.body_name2id(gap_right)
    arm_idx = _arm_joint_indexes(env)
    K = camera_utils.get_camera_intrinsic_matrix(env.sim, CAMERA_NAME, height, width)

    rgb_frames: List[np.ndarray] = []
    rendered_frames: List[np.ndarray] = []
    skel_frames: List[np.ndarray] = []
    overlay_frames: List[np.ndarray] = []
    actions = np.zeros((T, 8), dtype=np.float32)
    closed_signal = np.zeros(T, dtype=np.float64)
    per_frame_pts: List[Dict[str, np.ndarray]] = []
    per_frame_E: List[np.ndarray] = []
    need_frame = save_rgb or save_av or save_ov
    need_skel = save_av or save_ov

    for t in range(T):
        if t > 0:
            _set_frame(env, states[t])
            # set_state can leave camera unchanged in model; re-assert preset once is enough,
            # but re-apply cheaply in case XML/state path ever resets cam.
            if t == 1:
                _apply_camera_preset(
                    env, camera_preset, baseline_target_dz=baseline_target_dz)
        jp = env.sim.data.qpos[arm_idx]
        lf = env.sim.data.body_xpos[body_ids[gap_left]] if gap_left in body_ids else None
        rf = env.sim.data.body_xpos[body_ids[gap_right]] if gap_right in body_ids else None
        gap = float(np.linalg.norm(lf - rf)) if (lf is not None and rf is not None) else 0.0
        actions[t, :7] = jp[:7]
        actions[t, 7] = gap
        open_frac = float(np.clip(gap / GRIPPER_MAX_GAP, 0.0, 1.0))
        closed_signal[t] = 1.0 - open_frac

        if not need_frame:
            continue

        if save_rgb or save_ov:
            rgb = np.ascontiguousarray(
                env.sim.render(width=width, height=height, camera_name=CAMERA_NAME)[::-1])
            rendered_frames.append(rgb.copy())
            if save_rgb:
                rgb_frames.append(rgb.copy())

        if need_skel:
            wp = {n: env.sim.data.body_xpos[body_ids[n]].copy() for n in names}
            per_frame_pts.append(wp)
            per_frame_E.append(camera_utils.get_camera_extrinsic_matrix(env.sim, CAMERA_NAME))

    if need_skel and per_frame_pts:
        closed_m, settled_m = _gripper_state_masks(closed_signal)
        for t in range(len(per_frame_pts)):
            wp = per_frame_pts[t]
            pts = np.array([wp[n] for n in names], dtype=np.float64)
            uv, depth = _project(pts, per_frame_E[t], K)
            name2uv = {n: uv[i] for i, n in enumerate(names)}
            name2z = {n: float(depth[i]) for i, n in enumerate(names)}
            closed = bool(closed_m[t])
            settled = bool(settled_m[t])
            gb = _gripper_bone_color(closed, settled)
            gj = _gripper_joint_color(closed, settled)
            if save_av:
                canvas = np.full((height, width, 3), 255, dtype=np.uint8)
                skel_frames.append(_draw_skeleton(name2uv, name2z, canvas, arm_bones,
                                                  gripper_bones, gb, gj))
            if save_ov:
                base = rendered_frames[t] if t < len(rendered_frames) else np.full(
                    (height, width, 3), 255, dtype=np.uint8)
                overlay_frames.append(_draw_skeleton(name2uv, name2z, base.copy(), arm_bones,
                                                     gripper_bones, gb, gj))

    return {
        "T": T,
        "fps": CONTROL_FREQ,
        "actions": actions,
        "num_arms": 1,
        "rgb_frames": rgb_frames,
        "skel_frames": skel_frames,
        "overlay_frames": overlay_frames,
        "camera": CAMERA_NAME,
        "camera_preset": camera_preset,
        "view_difficulty": cam_meta.get("view_difficulty"),
        "camera_baseline": cam_meta.get("camera_baseline"),
        "baseline_target_dz": cam_meta.get("baseline_target_dz"),
        "camera_eye": cam_meta.get("camera_eye"),
        "camera_target": cam_meta.get("camera_target"),
        "azim_offset_deg": cam_meta.get("azim_offset_deg"),
        "elev_offset_deg": cam_meta.get("elev_offset_deg"),
        "dist_scale": cam_meta.get("dist_scale"),
        "horiz_shift_m": cam_meta.get("horiz_shift_m"),
        "vert_shift_m": cam_meta.get("vert_shift_m"),
        "K": K.tolist(),
    }


def _benchmark_for_suite(suite: str):
    from libero.libero.benchmark import get_benchmark

    # get_benchmark returns a class; instantiate with default task order.
    return get_benchmark(suite)()


def _hdf5_task_map(benchmark, suite: str) -> Dict[str, Tuple[int, object]]:
    mapping: Dict[str, Tuple[int, object]] = {}
    for i in range(benchmark.n_tasks):
        rel = benchmark.get_task_demonstration(i)
        base = os.path.basename(rel)
        mapping[base] = (i, benchmark.get_task(i))
    return mapping


def _load_selection_manifest(path: str) -> Dict[str, object]:
    with open(path, "r") as f:
        man = json.load(f)
    if not isinstance(man, dict) or "tasks" not in man:
        raise ValueError(f"invalid selection manifest (missing tasks): {path}")
    tasks = man["tasks"]
    if not isinstance(tasks, list) or not tasks:
        raise ValueError(f"selection manifest has empty tasks: {path}")
    return man


def _resolve_task_entry(benchmark, suite: str, suite_dir: str, task_name: str
                        ) -> Tuple[int, str, str, str]:
    """Return (task_idx, lang, bddl_path, h5_path) for a suite/task."""
    h5_map = _hdf5_task_map(benchmark, suite)
    h5_candidates = [
        os.path.join(suite_dir, f"{task_name}_demo.hdf5"),
        os.path.join(suite_dir, f"{task_name}.hdf5"),
    ]
    h5_path = next((p for p in h5_candidates if os.path.isfile(p)), "")
    if not h5_path:
        # Fall back to basename match in suite dir.
        for p in sorted(glob.glob(os.path.join(suite_dir, "*.hdf5"))):
            base = os.path.basename(p)
            stem = base[:-len("_demo.hdf5")] if base.endswith("_demo.hdf5") else os.path.splitext(base)[0]
            if stem == task_name:
                h5_path = p
                break
    if not h5_path:
        raise FileNotFoundError(f"missing hdf5 for {suite}/{task_name} under {suite_dir}")

    base = os.path.basename(h5_path)
    hit = h5_map.get(base)
    if hit is not None:
        task_idx, task = hit
        lang = task.language
        bddl_path = benchmark.get_task_bddl_file_path(task_idx)
        return task_idx, lang, bddl_path, h5_path

    task_idx = -1
    lang = task_name.replace("_", " ")
    bddl_path = ""
    for i in range(benchmark.n_tasks):
        t = benchmark.get_task(i)
        if t.name == task_name:
            task_idx = i
            lang = t.language
            bddl_path = benchmark.get_task_bddl_file_path(i)
            break
    return task_idx, lang, bddl_path, h5_path


def collect_jobs_from_manifest(args, manifest: Dict[str, object]) -> List[EpisodeJob]:
    """Strict selection: fail if any listed task/demo is missing."""
    jobs: List[EpisodeJob] = []
    expected = 0
    for entry in manifest["tasks"]:
        suite = entry["suite"]
        task_name = entry["task"]
        demos = list(entry.get("demos") or [])
        if not demos:
            raise ValueError(f"manifest task {suite}/{task_name} has empty demos")
        expected += len(demos)
        suite_dir = os.path.join(args.demo_root, suite)
        if not os.path.isdir(suite_dir):
            raise FileNotFoundError(f"missing suite dir {suite_dir}")
        try:
            benchmark = _benchmark_for_suite(suite)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"benchmark {suite}: {e!r}") from e
        task_idx, lang, bddl_path, h5_path = _resolve_task_entry(
            benchmark, suite, suite_dir, task_name)
        if not bddl_path or not os.path.isfile(bddl_path):
            raise FileNotFoundError(f"no bddl for {suite}/{task_name}")
        with h5py.File(h5_path, "r") as f:
            lang = _lang_from_h5(f, lang)
            available = set(f["data"].keys())
        missing = [d for d in demos if d not in available]
        if missing:
            raise FileNotFoundError(
                f"{suite}/{task_name} missing demos {missing}; "
                f"available={sorted(available, key=lambda k: int(k.split('_')[1]))[:10]}...")
        for demo_id in demos:
            jobs.append(EpisodeJob(
                suite=suite,
                task=task_name,
                task_idx=task_idx,
                demo_id=demo_id,
                h5_path=h5_path,
                bddl_path=bddl_path,
                lang=lang,
            ))
    if len(jobs) != expected:
        raise RuntimeError(f"manifest job count mismatch: got {len(jobs)} expected {expected}")
    print(f"[Main] selection-manifest={manifest.get('name', '?')} "
          f"tasks={len(manifest['tasks'])} episodes={len(jobs)}", flush=True)
    return jobs


def collect_jobs(args) -> List[EpisodeJob]:
    if getattr(args, "selection_manifest", None):
        man = _load_selection_manifest(args.selection_manifest)
        # Preserve manifest task/demo order (do not alphabetical-sort suites).
        jobs = collect_jobs_from_manifest(args, man)
        # Allow smoke slicing after strict manifest expansion.
        if args.start or args.end < 10_000_000:
            jobs = jobs[args.start:args.end]
        if args.shard_index is not None and args.num_shards:
            n, i = int(args.num_shards), int(args.shard_index)
            jobs = [j for k, j in enumerate(jobs) if k % n == i]
        return jobs

    jobs: List[EpisodeJob] = []
    task_filter = set(args.tasks) if args.tasks else None
    limit_suite = int(args.limit_per_suite or 0)
    limit_task = int(args.limit_per_task or 0)

    for suite in args.suites:
        suite_dir = os.path.join(args.demo_root, suite)
        if not os.path.isdir(suite_dir):
            print(f"[WARN] missing suite dir {suite_dir}", flush=True)
            continue

        try:
            benchmark = _benchmark_for_suite(suite)
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] benchmark {suite}: {e!r}", flush=True)
            continue

        h5_map = _hdf5_task_map(benchmark, suite)
        h5_files = sorted(glob.glob(os.path.join(suite_dir, "*.hdf5")))
        tasks_done = 0

        for h5_path in h5_files:
            if limit_suite > 0 and tasks_done >= limit_suite:
                break
            base = os.path.basename(h5_path)
            hit = h5_map.get(base)
            if hit is None:
                task_name = base[:-len("_demo.hdf5")] if base.endswith("_demo.hdf5") else os.path.splitext(base)[0]
                task_idx = -1
                lang = task_name.replace("_", " ")
                bddl_path = ""
                for i in range(benchmark.n_tasks):
                    t = benchmark.get_task(i)
                    if t.name == task_name:
                        task_idx = i
                        lang = t.language
                        bddl_path = benchmark.get_task_bddl_file_path(i)
                        break
            else:
                task_idx, task = hit
                task_name = task.name
                lang = task.language
                bddl_path = benchmark.get_task_bddl_file_path(task_idx)

            if task_filter and task_name not in task_filter:
                continue

            if not bddl_path or not os.path.isfile(bddl_path):
                print(f"[WARN] no bddl for {suite}/{task_name} ({h5_path})", flush=True)
                continue

            with h5py.File(h5_path, "r") as f:
                lang = _lang_from_h5(f, lang)
                demos = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
            if limit_task > 0:
                demos = demos[:limit_task]

            for demo_id in demos:
                jobs.append(EpisodeJob(
                    suite=suite,
                    task=task_name,
                    task_idx=task_idx,
                    demo_id=demo_id,
                    h5_path=h5_path,
                    bddl_path=bddl_path,
                    lang=lang,
                ))
            tasks_done += 1

    jobs.sort(key=lambda j: (j.suite, j.task, int(j.demo_id.split("_")[1])))
    if args.start or args.end < 10_000_000:
        jobs = jobs[args.start:args.end]
    if args.shard_index is not None and args.num_shards:
        n, i = int(args.num_shards), int(args.shard_index)
        jobs = [j for k, j in enumerate(jobs) if k % n == i]
    return jobs


def resolve_render_presets(args) -> List[str]:
    """Resolve CLI into an ordered list of camera presets to render."""
    view_diff = getattr(args, "view_difficulty", None)
    cam = getattr(args, "camera_preset", None)
    if view_diff and cam:
        raise ValueError(
            "pass either --view-difficulty or --camera-preset, not both "
            f"(got view-difficulty={view_diff!r}, camera-preset={cam!r})")
    if view_diff:
        return list(presets_for_difficulty(view_diff))
    if cam:
        resolve_camera_preset(cam)
        return [cam]
    # Backward-compatible default.
    return ["oblique45"]


def episode_output_dir(output_root: str, job: EpisodeJob, preset: str) -> str:
    """Legacy presets stay at <root>/<suite>/<task>/<demo>; view-gen uses difficulty/preset prefix."""
    spec = resolve_camera_preset(preset)
    if spec.difficulty:
        return os.path.join(
            output_root, spec.difficulty, preset, job.suite, job.task, job.demo_id)
    return os.path.join(output_root, job.suite, job.task, job.demo_id)


def _artifact_paths(ep_dir: str) -> Dict[str, str]:
    return {
        "rgb": os.path.join(ep_dir, "rgb.mp4"),
        "action_video": os.path.join(ep_dir, "action_video.mp4"),
        "overlay": os.path.join(ep_dir, "overlay.mp4"),
        "action": os.path.join(ep_dir, "action.npy"),
        "metadata": os.path.join(ep_dir, "metadata.json"),
    }


def _nonempty(p: Optional[str]) -> bool:
    return bool(p) and os.path.exists(p) and os.path.getsize(p) > 0


def _needs_render(paths: Dict[str, str], save_rgb: bool, save_av: bool,
                  save_ov: bool, resume: bool) -> bool:
    if not resume:
        return True
    if not _nonempty(paths["metadata"]) or not _nonempty(paths["action"]):
        return True
    if save_rgb and not _nonempty(paths["rgb"]):
        return True
    if save_av and not _nonempty(paths["action_video"]):
        return True
    if save_ov and not _nonempty(paths["overlay"]):
        return True
    return False


def process_jobs(args) -> int:
    save_rgb = args.save_mode in ("rgb", "all", "rgb_action")
    save_av = args.save_mode in ("action", "all", "rgb_action")
    save_ov = args.save_mode in ("overlay", "all")

    try:
        presets = resolve_render_presets(args)
        jobs = collect_jobs(args)
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        print(f"[Main] FAIL: {e}", flush=True)
        return 2
    print(f"[Main] jobs={len(jobs)} presets={presets} mode={args.save_mode} "
          f"res={args.width}x{args.height} camera={CAMERA_NAME} "
          f"shard={args.shard_index}/{args.num_shards} out={args.output_root}", flush=True)
    if not jobs:
        # Empty shard is OK under multi-process (e.g. --end 1 with --num-workers 4).
        if args.shard_index is not None and args.num_shards:
            print("[Main] shard empty — skip", flush=True)
            return 0
        print("[Main] FAIL: jobs=0 (check LIBERO config / demo_root / suite filters / manifest)",
              flush=True)
        return 1
    os.makedirs(args.output_root, exist_ok=True)

    env = None
    env_key: Optional[Tuple[str, str]] = None
    open_files: Dict[str, h5py.File] = {}
    t0 = time.time()
    ok = fail = skipped = 0
    total_units = len(jobs) * len(presets)
    unit_i = 0

    for idx, job in enumerate(jobs):
        key = (job.suite, job.task)
        if env is None or env_key != key:
            if env is not None:
                env.close()
            env = make_env(job.bddl_path, args.width, args.height)
            env_key = key

        if job.h5_path not in open_files:
            open_files[job.h5_path] = h5py.File(job.h5_path, "r")
        f = open_files[job.h5_path]

        for preset in presets:
            unit_i += 1
            ep_dir = episode_output_dir(args.output_root, job, preset)
            paths = _artifact_paths(ep_dir)
            if not _needs_render(paths, save_rgb, save_av, save_ov, args.resume):
                skipped += 1
                continue

            try:
                res = render_episode(
                    env, f, job.demo_id, args.width, args.height,
                    save_rgb, save_av, save_ov,
                    camera_preset=preset,
                    baseline_target_dz=suite_baseline_target_dz(job.suite),
                )
            except Exception as e:  # noqa: BLE001
                fail += 1
                print(f"  [FAIL] {job.suite}/{job.task}/{job.demo_id}/{preset}: {e!r}",
                      flush=True)
                continue

            os.makedirs(ep_dir, exist_ok=True)
            np.save(paths["action"], res["actions"])
            errs: List[str] = []
            if save_rgb:
                good, err = _encode_video(paths["rgb"], res["rgb_frames"], res["fps"],
                                          args.crf, args.preset)
                if not good:
                    errs.append(f"rgb:{err}")
            if save_av:
                good, err = _encode_video(paths["action_video"], res["skel_frames"], res["fps"],
                                          args.crf, args.preset)
                if not good:
                    errs.append(f"av:{err}")
            if save_ov:
                good, err = _encode_video(paths["overlay"], res["overlay_frames"], res["fps"],
                                          args.crf, args.preset)
                if not good:
                    errs.append(f"ov:{err}")

            env_id = f"{job.suite}/{job.task}"
            meta = {
                "suite": job.suite,
                "task": job.task,
                "env_id": env_id,
                "traj_id": job.demo_id,
                "frames": res["T"],
                "fps": res["fps"],
                "lang": job.lang,
                "num_arms": res.get("num_arms", 1),
                "action_dim": 8,
                "width": args.width,
                "height": args.height,
                "camera": res["camera"],
                "camera_preset": res.get("camera_preset", preset),
                "view_difficulty": res.get("view_difficulty"),
                "camera_baseline": res.get("camera_baseline"),
                "baseline_target_dz": res.get("baseline_target_dz"),
                "camera_eye": res.get("camera_eye"),
                "camera_target": res.get("camera_target"),
                "azim_offset_deg": res.get("azim_offset_deg"),
                "elev_offset_deg": res.get("elev_offset_deg"),
                "dist_scale": res.get("dist_scale"),
                "horiz_shift_m": res.get("horiz_shift_m"),
                "vert_shift_m": res.get("vert_shift_m"),
                "K": res["K"],
                "rgb": os.path.abspath(paths["rgb"]) if _nonempty(paths["rgb"]) else None,
                "action_video": (
                    os.path.abspath(paths["action_video"])
                    if _nonempty(paths["action_video"]) else None),
                "overlay": os.path.abspath(paths["overlay"]) if _nonempty(paths["overlay"]) else None,
                "action": os.path.abspath(paths["action"]) if _nonempty(paths["action"]) else None,
                "h5_path": os.path.abspath(job.h5_path),
                "bddl_path": os.path.abspath(job.bddl_path),
                "errors": errs,
            }
            with open(paths["metadata"], "w") as mf:
                json.dump(meta, mf)

            if errs:
                fail += 1
                print(f"  [WARN] {job.suite}/{job.task}/{job.demo_id}/{preset}: {errs}",
                      flush=True)
            else:
                ok += 1
            if unit_i % 5 == 0 or unit_i == total_units:
                dt = time.time() - t0
                print(f"  [{unit_i}/{total_units}] ok={ok} fail={fail} skip={skipped} "
                      f"{dt/max(1, ok + fail):.1f}s/ep", flush=True)

    if env is not None:
        env.close()
    for fh in open_files.values():
        fh.close()
    print(f"[Main] done ok={ok} fail={fail} skip={skipped}", flush=True)
    return 0 if fail == 0 else 1


def _count_visible_gpus() -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "-L"], text=True, stderr=subprocess.DEVNULL, timeout=10)
        n = sum(1 for line in out.splitlines() if line.strip())
        return max(1, n) if n else 1
    except Exception:
        return 1


def _strip_flag(argv: List[str], flag: str) -> List[str]:
    """Remove ``flag`` and its value (supports ``--flag V`` / ``--flag=V``)."""
    out: List[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == flag:
            i += 2
            continue
        if a.startswith(flag + "="):
            i += 1
            continue
        out.append(a)
        i += 1
    return out


def run_multiprocess(argv: List[str], num_workers: int) -> int:
    """Spawn ``num_workers`` shard subprocesses (round-robin across GPUs)."""
    n = int(num_workers)
    if n <= 1:
        raise ValueError("num_workers must be > 1")
    if "--shard-index" in argv or "--num-shards" in argv:
        raise ValueError("--num-workers cannot be combined with --shard-index/--num-shards")

    child_argv = _strip_flag(list(argv), "--num-workers")
    n_gpu = _count_visible_gpus()
    script = os.path.abspath(__file__)
    py = sys.executable
    print(f"[mp] workers={n} gpus={n_gpu} python={py}", flush=True)

    procs: List[subprocess.Popen] = []
    for i in range(n):
        gpu = i % n_gpu
        cmd = [
            py, script, *child_argv,
            "--shard-index", str(i),
            "--num-shards", str(n),
        ]
        env = os.environ.copy()
        env["MUJOCO_GL"] = env.get("MUJOCO_GL", "egl")
        env["PYOPENGL_PLATFORM"] = "osmesa" if env["MUJOCO_GL"] == "osmesa" else "egl"
        env["MUJOCO_EGL_DEVICE_ID"] = str(gpu)
        env.pop("DISPLAY", None)
        env.pop("WAYLAND_DISPLAY", None)
        env.pop("CUDA_VISIBLE_DEVICES", None)
        print(f"[mp] start worker {i}/{n} gpu={gpu}", flush=True)
        procs.append(subprocess.Popen(cmd, env=env))
        # Stagger EGL init slightly to avoid thundering herd on same GPU.
        time.sleep(0.5 + 0.25 * (i % n_gpu))

    codes = [p.wait() for p in procs]
    fail = sum(1 for c in codes if c != 0)
    print(f"[mp] done codes={codes} fail={fail}", flush=True)
    return 0 if fail == 0 else 1


def main(argv: Optional[List[str]] = None) -> int:
    argv_list = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(description="LIBERO replay renderer (agentview rgb+skeleton+action)")
    p.add_argument("--demo-root", default="/media/datasets/libero")
    p.add_argument("--suites", nargs="+", default=DEFAULT_SUITES)
    p.add_argument("--output-root", default="/media/datasets/zhangshilin/libero_action_video")
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--save-mode", default="rgb_action",
                   choices=["rgb", "action", "overlay", "all", "rgb_action"])
    p.add_argument("--limit-per-suite", type=int, default=0, help="每个 suite 只渲前 N 个 task (0=all)")
    p.add_argument("--limit-per-task", type=int, default=0, help="每个 task 只渲前 N 条 demo (0=all)")
    p.add_argument("--tasks", nargs="*", default=None, help="可选 task 名过滤")
    p.add_argument("--selection-manifest", default=None,
                   help="固定验证清单 JSON (严格按 suite/task/demo 选择; 缺失则失败)")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=10_000_000)
    p.add_argument("--shard-index", type=int, default=None)
    p.add_argument("--num-shards", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=1,
                   help="多进程并行渲染: 自动按 shard 切分并轮询绑定 GPU "
                        "(>1 时父进程只负责拉起子进程; 勿与 --shard-index 同用)")
    p.add_argument("--camera-preset", default=None, choices=list(ALL_CAMERA_PRESET_NAMES),
                   help="单个 agentview preset; 与 --view-difficulty 互斥. "
                        "未指定且无 view-difficulty 时默认 oblique45")
    p.add_argument("--view-difficulty", default=None,
                   choices=["easy", "medium", "hard", "ood", "all"],
                   help="渲染该难度下全部确定性视角 "
                        "(easy/medium/hard 各 2 个; ood=5 个展示视角; all=原 6 个配对视角, 不含 ood). "
                        "与 --camera-preset 互斥")
    p.add_argument("--check-presets", action="store_true",
                   help="只做视角几何自检后退出, 不渲染")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--crf", type=int, default=20)
    p.add_argument("--preset", default="veryfast")
    args = p.parse_args(argv_list)
    if args.check_presets:
        errs = check_camera_presets_geometry()
        if errs:
            print("[check-presets] FAIL:", flush=True)
            for e in errs:
                print(f"  - {e}", flush=True)
            return 1
        print("[check-presets] OK "
              f"view_gen={list(ALL_VIEW_GEN_PRESETS)} "
              f"ood={list(VIEW_GEN_PRESETS['ood'])} "
              f"legacy={list(LEGACY_CAMERA_PRESETS)}",
              flush=True)
        return 0
    if int(args.num_workers) > 1:
        return run_multiprocess(argv_list, int(args.num_workers))
    if args.width % 2:
        args.width += 1
    if args.height % 2:
        args.height += 1
    # Must run before any `import libero` (collect_jobs / make_env).
    _ensure_libero_config(
        libero_src=os.environ.get("LIBERO_SRC", ""),
        datasets=args.demo_root,
    )
    return process_jobs(args)


if __name__ == "__main__":
    raise SystemExit(main())
