"""ManiSkill3 (SAPIEN3) 回放渲染器 -> GT rgb + action 骨架 + action 数值.

对齐 RoboCasa 的 "在仿真里回放, 一次产出 RGB + 骨架" 思路 (visualization_urdf/robocasa/
render_action_video_robocasa.py), 只把 MuJoCo 换成 ManiSkill3/SAPIEN3:

  * demo 是 state-only 的 trajectory.h5 (env_states + actions), 逐帧 set_state_dict 精确回放
  * RGB = env.render() 的 human_render_camera (第三人称, 训练目标视频)
  * 骨架 = 用 sim ground-truth 的相机内外参 + 各 link 世界坐标投影 (免 URDF / 免标定)

每条 episode 产出 (与 RoboCasa 字段风格一致):
  * rgb.mp4            -- GT RGB (训练目标)
  * action_video.mp4   -- Panda 火柴棍骨架
  * overlay.mp4        -- 骨架叠加在 RGB 上 (自检对齐, save-mode=all 时输出)
  * action.npy         -- (T, 8) = 7 臂关节 qpos + 夹爪开度 (m)
  * metadata.json      -- env_id/traj/frames/fps/K/lang/各视频路径

环境依赖 (本机已装好):
  * mani_skill 3.x + sapien 3.x (venv: /root/envs/maniskill)
  * GPU Vulkan 渲染 (需 libEGL.so.1; 见脚本顶部 env 固化)

用法:
    /root/envs/maniskill/bin/python \
        visualization_urdf/maniskill/render_action_video_maniskill.py \
        --envs PickCube-v1 --save-mode all --limit-per-env 2
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# --- 关键环境修复: SAPIEN3 走 NVIDIA Vulkan; 本容器需显式指定 ICD 且 libEGL.so.1 已装 ---
if os.path.isfile("/etc/vulkan/icd.d/nvidia_icd.json"):
    os.environ.setdefault("VK_ICD_FILENAMES", "/etc/vulkan/icd.d/nvidia_icd.json")
os.environ.setdefault("XDG_RUNTIME_DIR", "/tmp/xdg")
os.makedirs(os.environ["XDG_RUNTIME_DIR"], exist_ok=True)

import numpy as np  # noqa: E402
import cv2  # noqa: E402
import h5py  # noqa: E402

# ============================ Panda 骨架拓扑 ============================
# 臂链: base(link0) -> link1 .. link7 -> hand (所有 Panda 变体都有这些 link)
ARM_LINKS = [
    "panda_link0", "panda_link1", "panda_link2", "panda_link3",
    "panda_link4", "panda_link5", "panda_link6", "panda_link7", "panda_hand",
]
ARM_BONES = [(ARM_LINKS[i], ARM_LINKS[i + 1]) for i in range(len(ARM_LINKS) - 1)]
GRIPPER_HAND = "panda_hand"
GRIPPER_LEFT_SYN = "grip_L"
GRIPPER_RIGHT_SYN = "grip_R"
GRIPPER_MAX_GAP = 0.08  # Panda 两指最大间距 (m)
# pad(指尖) 相对 finger link 原点的固定局部偏移: 沿 finger 局部 +z 轴 0.0455m (实测)。
# panda_wristcam 无 *_pad link, 需用此偏移从 *finger 外推真实指尖。
FINGERTIP_LOCAL_OFFSET = np.array([0.0, 0.0, 0.0455])
# 夹爪末端候选 link: 优先 *_pad(直接是指尖); 退回 *finger(指根, 需 +偏移到指尖)。
GRIPPER_SIDE_CANDS = {
    GRIPPER_LEFT_SYN: ["panda_leftfinger_pad", "panda_leftfinger"],
    GRIPPER_RIGHT_SYN: ["panda_rightfinger_pad", "panda_rightfinger"],
}


def _quat_to_R(q: np.ndarray) -> np.ndarray:
    """SAPIEN 四元数 [w,x,y,z] -> 3x3 旋转矩阵。"""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _resolve_gripper(name2link: Dict[str, object]) -> List[Tuple[str, str, bool]]:
    """解析每侧夹爪末端: 返回 [(synthetic_name, link_name, needs_tip_offset), ...]。
    needs_tip_offset=True 表示该 link 是 *finger(指根), 需沿局部 z 外推到指尖。无夹爪返回 []。"""
    specs: List[Tuple[str, str, bool]] = []
    for syn, cands in GRIPPER_SIDE_CANDS.items():
        for c in cands:
            if c in name2link:
                specs.append((syn, c, not c.endswith("_pad")))
                break
    return specs


def _gripper_endpoint(link, needs_offset: bool) -> np.ndarray:
    """夹爪末端(指尖)世界坐标: 有 pad 直接用其原点; 只有 finger 则 +局部偏移。"""
    p = (link.pose.p.cpu().numpy() if hasattr(link.pose.p, "cpu") else np.asarray(link.pose.p))[0]
    if needs_offset:
        q = (link.pose.q.cpu().numpy() if hasattr(link.pose.q, "cpu") else np.asarray(link.pose.q))[0]
        p = p + _quat_to_R(q) @ FINGERTIP_LOCAL_OFFSET
    return p

# 配色 (直接画在 RGB 帧上, 故按 RGB 顺序; 对齐 RealSource 最新视觉)
COLOR_ARM_BONE = (0, 120, 255)            # 蓝
COLOR_GRIPPER_OPEN_BONE = (0, 120, 255)   # 开: 蓝
COLOR_GRIPPER_CLOSED_BONE = (255, 140, 0)  # 合: 橙
COLOR_ARM_JOINT = (255, 50, 50)           # 红
COLOR_GRIPPER_OPEN_JOINT = (255, 50, 50)  # 开: 红
COLOR_GRIPPER_CLOSED_JOINT = (0, 200, 0)  # 合: 绿
COLOR_GRIPPER_MOTION_BONE = (128, 128, 128)  # 开合过程骨线: 灰
COLOR_GRIPPER_MOTION_JOINT = (255, 50, 50)   # 开合过程关节: 红
ARM_BONE_WIDTH = 6
ARM_JOINT_RADIUS = 9

# 夹爪时序状态机参数 (配色对齐 RealSource; 稳定窗口用仅未来 lookahead)
# 约定: 喂给状态机的 close_frac, HIGH=物理闭合。
# settled: 当前+未来 lookahead 窗口跨度 < eps; 短 unsettled 空隙填回;
# early CLOSED: close_frac 很高且前向稳定时可提前确认闭合。
# 骨线: settled 二值开合色; 运动帧统一灰色。
# 关节: settled 开红/闭绿; 运动帧固定红色。
GRIPPER_CLOSED_OPEN_FRAC = 0.5
GRIPPER_STATE_LOOKAHEAD = 5         # 仅未来 5 帧判定平台稳定
GRIPPER_STATE_LOOKBACK = 0          # 不用过去帧
GRIPPER_STATE_PLATEAU_EPS = 0.03
GRIPPER_STATE_CHANGE_DELTA = 0.1
GRIPPER_STATE_SETTLED_FILL = 3
GRIPPER_EARLY_CLOSED_FRAC = 0.85


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
    """逐侧时序状态机, 返回 (closed_mask, settled_mask)。对齐 RealSource。

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
        # Early CLOSED: 已紧闭且前向窗口平坦时, 不等 lookback 清完斜坡。
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


def _gripper_bone_color(closed: bool, settled: bool) -> tuple:
    if not settled:
        return COLOR_GRIPPER_MOTION_BONE
    return COLOR_GRIPPER_CLOSED_BONE if closed else COLOR_GRIPPER_OPEN_BONE


def _gripper_joint_color(closed: bool, settled: bool) -> tuple:
    if not settled:
        return COLOR_GRIPPER_MOTION_JOINT
    return COLOR_GRIPPER_CLOSED_JOINT if closed else COLOR_GRIPPER_OPEN_JOINT

# 任务级语言指令 (ManiSkill demo 无逐 episode 语言, 用任务级描述)
ENV_INSTRUCTIONS = {
    "PickCube-v1": "pick up the cube and move it to the goal position",
    "StackCube-v1": "pick up the red cube and stack it on the green cube",
    "PushCube-v1": "push the cube to the goal region",
    "PullCube-v1": "pull the cube to the goal region",
    "PokeCube-v1": "poke the cube with the peg to the goal",
    "LiftPegUpright-v1": "lift the peg so it stands upright",
    "PegInsertionSide-v1": "insert the peg into the horizontal hole",
    "PlugCharger-v1": "plug the charger into the wall receptacle",
    "PushT-v1": "push the T-shaped block to the target pose",
    "RollBall-v1": "roll the ball to the goal position",
    "StackPyramid-v1": "stack the cubes into a pyramid",
    "DrawTriangle-v1": "draw a triangle with the end-effector",
    "PullCubeTool-v1": "use the tool to pull the cube within reach",
    "TwoRobotPickCube-v1": "two arms cooperate: hand the cube from one arm to the other and place it at the goal",
    "TwoRobotStackCube-v1": "two arms cooperate to pick and stack the cubes",
}


@dataclass
class EpisodeJob:
    env_id: str
    traj_key: str
    h5_path: str
    reset_kwargs: dict


# ============================ ffmpeg 编码 (复用 RoboCasa) ============================
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


# ============================ 投影 + 骨架绘制 ============================
def _project(points_world: np.ndarray, extrinsic_cv: np.ndarray,
             K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """ManiSkill 的 get_extrinsic_matrix() 已是 world->cam (OpenCV, 3x4),
    因此不像 RoboCasa 那样需要对 cam2world 求逆。"""
    n = points_world.shape[0]
    homo = np.concatenate([points_world, np.ones((n, 1))], axis=1)  # (n,4)
    cam = (extrinsic_cv @ homo.T).T  # (n,3)
    depth = cam[:, 2].copy()
    safe = np.where(np.abs(depth) < 1e-6, 1e-6, depth)
    uv = (K @ cam.T).T
    uv = uv[:, :2] / safe[:, None]
    return uv, depth


def _draw_skeleton(name2uv: Dict[str, np.ndarray], name2z: Dict[str, float],
                   img: np.ndarray, gripper_bones: List[Tuple[str, str]],
                   grip_bone, grip_joint) -> np.ndarray:
    def _ok(n):
        return n in name2z and name2z[n] > 0.01

    for a, b in ARM_BONES:
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


# ============================ ManiSkill 环境封装 ============================
# 经典斜上方 3/4 视角 (与 StackCube/PickCube 等多数 tabletop 任务默认一致)
CLASSIC_3Q_EYE = [0.6, 0.7, 0.6]
CLASSIC_3Q_TARGET = [0.0, 0.0, 0.35]
CLASSIC_3Q_FOV = 1.0


def make_env(env_id: str, env_kwargs: dict, width: int, height: int,
             camera_preset: str = "default"):
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401  (注册环境)
    from mani_skill.utils import sapien_utils

    ek = dict(env_kwargs)
    # 回放只需 state, 渲染用 human_render_camera; sim 用 CPU (确定性回放), 渲染用 GPU Vulkan
    ek.update(
        obs_mode="none",
        render_mode="rgb_array",
        sim_backend="physx_cpu",
        num_envs=1,
    )
    ek.pop("reward_mode", None)
    if camera_preset == "classic_3q":
        pose = sapien_utils.look_at(CLASSIC_3Q_EYE, CLASSIC_3Q_TARGET)
        ek["human_render_camera_configs"] = {
            "render_camera": {
                "pose": pose,
                "width": width,
                "height": height,
                "fov": CLASSIC_3Q_FOV,
                "mount": None,  # 覆盖 PlugCharger 等挂在场景物体上的相机
            }
        }
    else:
        try:
            ek["human_render_camera_configs"] = dict(width=width, height=height)
        except Exception:
            pass
    return gym.make(env_id, **ek)


def _to_np(x):
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


def _agent_links(agent):
    """(robot, {link_name: link}) —— link 对象跨帧持久, 只需建一次。"""
    robot = agent.robot
    return robot, {l.name: l for l in robot.get_links()}


def render_episode(env, h5file: h5py.File, traj_key: str, reset_kwargs: dict,
                   width: int, height: int, save_rgb: bool, save_av: bool,
                   save_ov: bool) -> dict:
    from mani_skill.trajectory import utils as trajectory_utils

    u = env.unwrapped
    tr = h5file[traj_key]
    states = trajectory_utils.dict_to_list_of_dicts(tr["env_states"])
    T = len(states)

    # 必须先 reset: PegInsertion/PlugCharger 等任务会在 reset 时重建 articulation,
    # 若事先缓存 robot/link 引用, 会读到已失效对象, 骨架整段冻在初始姿态。
    env.reset(**reset_kwargs)

    # 单臂: [u.agent]; 双臂(MultiAgent): [agents[0], agents[1]] —— 每臂独立 Panda
    agents = list(u.agent.agents) if hasattr(u.agent, "agents") else [u.agent]
    per_agent = [_agent_links(a) for a in agents]
    num_arms = len(agents)
    # 每臂: (robot, name2link, gripper_specs, arm_names, point_names, gripper_bones)
    # gripper_specs: [(syn_name, link_name, needs_tip_offset), ...]
    agent_meta = []
    for robot, n2l in per_agent:
        gspecs = _resolve_gripper(n2l)
        arm_names = [n for n in ARM_LINKS if n in n2l]
        gsyns = [s[0] for s in gspecs]
        point_names = arm_names + gsyns  # 骨架所有点(含合成的夹爪末端)
        gbones = [(GRIPPER_HAND, syn) for syn in gsyns]
        agent_meta.append((robot, n2l, gspecs, arm_names, point_names, gbones))
    cam = list(u.scene.human_render_cameras.values())[0]
    cam_name = list(u.scene.human_render_cameras.keys())[0]

    rgb_frames: List[np.ndarray] = []
    skel_frames: List[np.ndarray] = []
    overlay_frames: List[np.ndarray] = []
    actions = np.zeros((T, 8 * num_arms), dtype=np.float32)  # 每臂 7 关节 + 夹爪开度
    need_frame = save_rgb or save_av or save_ov
    need_skel = save_av or save_ov
    K_last: Optional[np.ndarray] = None

    # ---- 阶段1: 逐帧 set_state, 采集世界坐标点 / 夹爪闭合信号 / RGB (不上色) ----
    rendered_frames: List[np.ndarray] = []  # 渲染的 RGB (供 rgb 输出与 overlay 底图)
    per_frame_pts: List[List[Dict[str, np.ndarray]]] = []  # [t][agent] -> {name: worldpos}
    per_frame_E: List[np.ndarray] = []
    closed_signal = [np.zeros(T, dtype=np.float64) for _ in range(num_arms)]  # HIGH=闭合
    for t in range(T):
        u.set_state_dict(states[t])
        frame_worldpts: List[Dict[str, np.ndarray]] = []
        for ai, (robot, n2l, gspecs, arm_names, _pn, _gb) in enumerate(agent_meta):
            qpos = _to_np(robot.get_qpos())[0]  # (>=7,) = 7 臂 (+ 夹爪指)
            wp: Dict[str, np.ndarray] = {n: _to_np(n2l[n].pose.p)[0] for n in arm_names}
            tips: Dict[str, np.ndarray] = {}
            for syn, lname, need_off in gspecs:
                tips[syn] = _gripper_endpoint(n2l[lname], need_off)
                wp[syn] = tips[syn]
            gap = (float(np.linalg.norm(tips[GRIPPER_LEFT_SYN] - tips[GRIPPER_RIGHT_SYN]))
                   if (GRIPPER_LEFT_SYN in tips and GRIPPER_RIGHT_SYN in tips) else 0.0)
            actions[t, ai * 8:ai * 8 + 7] = qpos[:7]
            actions[t, ai * 8 + 7] = gap
            open_frac = float(np.clip(gap / GRIPPER_MAX_GAP, 0.0, 1.0))
            closed_signal[ai][t] = 1.0 - open_frac  # HIGH=闭合
            frame_worldpts.append(wp)

        if not need_frame:
            continue
        if save_rgb or save_ov:
            rendered_frames.append(np.ascontiguousarray(_to_np(env.render())[0]))
        if need_skel:
            K_last = _to_np(cam.camera.get_intrinsic_matrix())[0]  # (3,3), 相机固定
            per_frame_E.append(_to_np(cam.camera.get_extrinsic_matrix())[0])  # (3,4)
            per_frame_pts.append(frame_worldpts)

    if save_rgb:
        rgb_frames = rendered_frames

    # ---- 阶段2: RealSource 式状态机 (closed + settled) -> 骨线/关节上色 ----
    if need_skel:
        state_masks = [_gripper_state_masks(closed_signal[ai]) for ai in range(num_arms)]
        for ti in range(len(per_frame_pts)):
            E = per_frame_E[ti]
            canvas = np.full((height, width, 3), 255, dtype=np.uint8) if save_av else None
            overlay = rendered_frames[ti].copy() if save_ov else None
            for ai, (robot, n2l, gspecs, arm_names, point_names, gbones) in enumerate(agent_meta):
                wp = per_frame_pts[ti][ai]
                pts = np.array([wp[n] for n in point_names], dtype=np.float64)
                uv, depth = _project(pts, E, K_last)
                name2uv = {n: uv[i] for i, n in enumerate(point_names)}
                name2z = {n: float(depth[i]) for i, n in enumerate(point_names)}
                closed_m, settled_m = state_masks[ai]
                closed = bool(closed_m[ti])
                settled = bool(settled_m[ti])
                gb = _gripper_bone_color(closed, settled)
                gj = _gripper_joint_color(closed, settled)
                if canvas is not None:
                    _draw_skeleton(name2uv, name2z, canvas, gbones, gb, gj)
                if overlay is not None:
                    _draw_skeleton(name2uv, name2z, overlay, gbones, gb, gj)
            if canvas is not None:
                skel_frames.append(canvas)
            if overlay is not None:
                overlay_frames.append(overlay)

    fps = float(getattr(u, "control_freq", 20) or 20)
    return {
        "T": T, "fps": fps, "actions": actions, "num_arms": num_arms,
        "rgb_frames": rgb_frames, "skel_frames": skel_frames,
        "overlay_frames": overlay_frames,
        "camera": cam_name,
        "K": K_last.tolist() if K_last is not None else None,
    }


# ============================ 任务收集 / 批量 ============================
def _traj_h5_for_env(demo_root: str, env_id: str, demo_type: str) -> Optional[Tuple[str, str]]:
    d = os.path.join(demo_root, env_id, demo_type)
    h5 = os.path.join(d, "trajectory.h5")
    js = os.path.join(d, "trajectory.json")
    if os.path.isfile(h5) and os.path.isfile(js):
        return h5, js
    return None


def collect_jobs(args) -> List[EpisodeJob]:
    jobs: List[EpisodeJob] = []
    limit = int(args.limit_per_env or 0)
    for env_id in args.envs:
        found = _traj_h5_for_env(args.demo_root, env_id, args.demo_type)
        if not found:
            print(f"[WARN] no {args.demo_type}/trajectory.h5 for {env_id} under {args.demo_root}",
                  flush=True)
            continue
        h5_path, js_path = found
        meta = json.load(open(js_path))
        ep_map = {int(e["episode_id"]): e for e in meta.get("episodes", [])}
        with h5py.File(h5_path, "r") as f:
            keys = sorted(f.keys(), key=lambda k: int(k.split("_")[1]))
        if limit > 0:
            keys = keys[:limit]
        for k in keys:
            eid = int(k.split("_")[1])
            rk = ep_map.get(eid, {}).get("reset_kwargs", {"seed": eid})
            jobs.append(EpisodeJob(env_id=env_id, traj_key=k, h5_path=h5_path, reset_kwargs=rk))

    jobs.sort(key=lambda j: (j.env_id, int(j.traj_key.split("_")[1])))
    if args.start or args.end < 10_000_000:
        jobs = jobs[args.start:args.end]
    if args.shard_index is not None and args.num_shards:
        n, i = int(args.num_shards), int(args.shard_index)
        jobs = [j for k, j in enumerate(jobs) if k % n == i]
    return jobs


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

    jobs = collect_jobs(args)
    print(f"[Main] jobs={len(jobs)} mode={args.save_mode} res={args.width}x{args.height} "
          f"camera={args.camera_preset} shard={args.shard_index}/{args.num_shards} "
          f"out={args.output_root}", flush=True)
    if not jobs:
        return 0
    os.makedirs(args.output_root, exist_ok=True)

    # env_kwargs 从每个 env 的 trajectory.json 读取一次
    env_kwargs_cache: Dict[str, dict] = {}
    for j in jobs:
        if j.env_id not in env_kwargs_cache:
            js = os.path.join(args.demo_root, j.env_id, args.demo_type, "trajectory.json")
            env_kwargs_cache[j.env_id] = json.load(open(js))["env_info"].get("env_kwargs", {})

    env = None
    cur_env_id = None
    open_files: Dict[str, h5py.File] = {}
    t0 = time.time()
    ok = fail = skipped = 0

    for idx, job in enumerate(jobs):
        ep_dir = os.path.join(args.output_root, job.env_id, job.traj_key)
        paths = _artifact_paths(ep_dir)
        if not _needs_render(paths, save_rgb, save_av, save_ov, args.resume):
            skipped += 1
            continue

        if env is None or cur_env_id != job.env_id:
            if env is not None:
                env.close()
            env = make_env(job.env_id, env_kwargs_cache[job.env_id], args.width, args.height,
                           args.camera_preset)
            cur_env_id = job.env_id

        if job.h5_path not in open_files:
            open_files[job.h5_path] = h5py.File(job.h5_path, "r")
        f = open_files[job.h5_path]

        try:
            res = render_episode(env, f, job.traj_key, job.reset_kwargs,
                                 args.width, args.height, save_rgb, save_av, save_ov)
        except Exception as e:  # noqa: BLE001
            fail += 1
            print(f"  [FAIL] {job.env_id}/{job.traj_key}: {e!r}", flush=True)
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

        meta = {
            "env_id": job.env_id,
            "traj_id": job.traj_key,
            "frames": res["T"],
            "fps": res["fps"],
            "lang": ENV_INSTRUCTIONS.get(job.env_id, job.env_id),
            "num_arms": res.get("num_arms", 1),
            "action_dim": 8 * res.get("num_arms", 1),
            "width": args.width,
            "height": args.height,
            "camera": res["camera"],
            "camera_preset": args.camera_preset,
            "K": res["K"],
            "rgb": os.path.abspath(paths["rgb"]) if _nonempty(paths["rgb"]) else None,
            "action_video": os.path.abspath(paths["action_video"]) if _nonempty(paths["action_video"]) else None,
            "overlay": os.path.abspath(paths["overlay"]) if _nonempty(paths["overlay"]) else None,
            "action": os.path.abspath(paths["action"]) if _nonempty(paths["action"]) else None,
            "errors": errs,
        }
        with open(paths["metadata"], "w") as mf:
            json.dump(meta, mf)

        if errs:
            fail += 1
            print(f"  [WARN] {job.env_id}/{job.traj_key}: {errs}", flush=True)
        else:
            ok += 1
        if (idx + 1) % 5 == 0 or idx == len(jobs) - 1:
            dt = time.time() - t0
            print(f"  [{idx+1}/{len(jobs)}] ok={ok} fail={fail} skip={skipped} "
                  f"{dt/max(1,ok+fail):.1f}s/ep", flush=True)

    if env is not None:
        env.close()
    for fh in open_files.values():
        fh.close()
    print(f"[Main] done ok={ok} fail={fail} skip={skipped}", flush=True)
    return 0 if fail == 0 else 1


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="ManiSkill3 replay renderer (rgb + skeleton + action)")
    p.add_argument("--envs", nargs="+", default=["PickCube-v1"], help="ManiSkill env ids")
    p.add_argument("--demo-root", default=os.path.expanduser("~/.maniskill/demos"))
    p.add_argument("--demo-type", default="motionplanning",
                   choices=["motionplanning", "teleop", "rl"])
    p.add_argument("--output-root", default="/media/zhangshilin/maniskill_action_video")
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--save-mode", default="rgb_action",
                   choices=["rgb", "action", "overlay", "all", "rgb_action"])
    p.add_argument("--camera-preset", default="default",
                   choices=["default", "classic_3q"],
                   help="human_render_camera 预设: default=任务默认, classic_3q=统一斜上方3/4视角")
    p.add_argument("--limit-per-env", type=int, default=0, help="每个 env 只渲前 N 条 (0=all)")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=10_000_000)
    p.add_argument("--shard-index", type=int, default=None)
    p.add_argument("--num-shards", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--crf", type=int, default=20)
    p.add_argument("--preset", default="veryfast")
    args = p.parse_args(argv)
    if args.width % 2:
        args.width += 1
    if args.height % 2:
        args.height += 1
    return process_jobs(args)


if __name__ == "__main__":
    raise SystemExit(main())
