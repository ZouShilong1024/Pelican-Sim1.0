#!/usr/bin/env python3
"""RealSource-World RS-02 双臂机器人：FK + 相机投影 + 骨架定义共享模块。

对齐 AgiBot / DROID 的 action_video 渲染范式，但换成 RealSource 的 RS-02 本体：

  * 动作 17 维（见 info.json features.action.names）:
      [0:7]  LeftLeaderArm_Joint1..7   -> URDF l_joint1..7
      [7]    LeftGripper.pos           -> URDF ltool_Left_1_Joint (drive, mimic 自动)
      [8:15] RightLeaderArm_Joint1..7  -> URDF r_joint1..7
      [15]   RightGripper.pos          -> URDF rtool_Left_1_Joint (drive, mimic 自动)
      [16]   Lift.command              -> 不画骨架（仅作数值 action 注入训练）
  * 相机：head 相机。内参取 meta/camera.json[episode].camera_parameters.head["720P"].MTX；
    外参**直接用** meta/camera.json[episode].hand_eye.{left_arm_to_eye,right_arm_to_eye}
    （= 头相机相对左/右臂 base 坐标系的位姿 R,T，即 base<-cam）。
    左臂骨架点在 l_base_link 系、右臂在 r_base_link 系，各自用对应 hand_eye 投影到同一张 head 图。

FK 采用自带的向量化 numpy 实现（解析 URDF 一次，整段 episode 批量 matmul），
可用 yourdfpy 做正确性交叉校验（见 verify_extrinsics.py）。
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ============================ RS-02 关节 / link 命名 ============================
LEFT_ARM_JOINTS = [f"l_joint{i}" for i in range(1, 8)]
RIGHT_ARM_JOINTS = [f"r_joint{i}" for i in range(1, 8)]
LEFT_GRIPPER_DRIVE = "ltool_Left_1_Joint"
RIGHT_GRIPPER_DRIVE = "rtool_Left_1_Joint"

LEFT_BASE_LINK = "l_base_link"
RIGHT_BASE_LINK = "r_base_link"

LEFT_ARM_LINKS = [LEFT_BASE_LINK] + [f"L_Link_{i}" for i in range(1, 8)]
RIGHT_ARM_LINKS = [RIGHT_BASE_LINK] + [f"R_Link_{i}" for i in range(1, 8)]

# 夹爪指尖 pad（画夹爪骨架末端），从腕 link7 引出两根
LEFT_WRIST_LINK = "L_Link_7"
RIGHT_WRIST_LINK = "R_Link_7"
LEFT_GRIPPER_PADS = ["ltool_Left_Pad_Link", "ltool_Right_Pad_Link"]
RIGHT_GRIPPER_PADS = ["rtool_Left_Pad_Link", "rtool_Right_Pad_Link"]

# 骨架 bone：臂链相邻 link 连线 + 腕->每个指尖 pad
LEFT_ARM_BONES = [(LEFT_ARM_LINKS[i], LEFT_ARM_LINKS[i + 1]) for i in range(len(LEFT_ARM_LINKS) - 1)]
RIGHT_ARM_BONES = [(RIGHT_ARM_LINKS[i], RIGHT_ARM_LINKS[i + 1]) for i in range(len(RIGHT_ARM_LINKS) - 1)]
LEFT_GRIPPER_BONES = [(LEFT_WRIST_LINK, pad) for pad in LEFT_GRIPPER_PADS]
RIGHT_GRIPPER_BONES = [(RIGHT_WRIST_LINK, pad) for pad in RIGHT_GRIPPER_PADS]

# 夹爪 drive joint 关节限位（URDF: [0,1] rad）。gripper.pos 已是 [0,1] 分数。
GRIPPER_DRIVE_MAX_Q = 1.0

# 配色（与 AgiBot / RoboTwin 对齐；此处按 BGR 写给 cv2）
# 视觉 RGB: 臂骨蓝(0,120,255)、臂关节红(255,50,50)、夹爪开蓝/闭橙(255,140,0)/关节开红闭绿(0,200,0)
# 开合过渡（unsettled）统一灰色，不做蓝↔橙渐变。
COLOR_ARM_BONE = (255, 120, 0)          # RGB(0,120,255) 蓝
COLOR_ARM_JOINT = (50, 50, 255)         # RGB(255,50,50) 红
COLOR_GRIPPER_OPEN_BONE = (255, 120, 0)     # 蓝
COLOR_GRIPPER_CLOSED_BONE = (0, 140, 255)   # RGB(255,140,0) 橙
COLOR_GRIPPER_OPEN_JOINT = (50, 50, 255)    # 红
COLOR_GRIPPER_CLOSED_JOINT = (0, 200, 0)    # 绿
COLOR_GRIPPER_MOTION_BONE = (128, 128, 128)   # 开合过程骨线：灰
COLOR_GRIPPER_MOTION_JOINT = (50, 50, 255)      # 开合过程关节：红（与开态关节同色）
ARM_BONE_WIDTH = 6
ARM_JOINT_RADIUS = 9

# ---- 夹爪时序状态机参数（对齐 AgiBot beta / agibot_beta_render）----
# 约定：喂给状态机的 close_frac，HIGH = 物理闭合。
# 稳定平台：过去 lookback + 当前 + 未来 lookahead 窗口内跨度 < eps 才记参考宽度；
# 新平台与参考差 > change_delta 时翻转（更高->闭合、更低->张开）；
# closed_mask 对运动帧保持上一确认态（迟滞）；settled_mask 标记平台帧。
# 骨线：settled 用二值开合色；运动（unsettled）统一灰色（无渐变）。
# 关节点：settled 开红/闭绿；运动帧固定红色。
# 短 unsettled 空隙 (<= SETTLED_FILL) 填回 settled，避免闪烁。
# early CLOSED：close_frac 已很高且仅前向窗口稳定时，不等 lookback 清完斜坡即可确认闭合。
GRIPPER_CLOSED_OPEN_FRAC = 0.5      # 首个平台的绝对分类阈值
GRIPPER_STATE_LOOKAHEAD = 10        # 平台稳定性所用的未来帧数
GRIPPER_STATE_LOOKBACK = 10         # 平台稳定性所用的过去帧数
GRIPPER_STATE_PLATEAU_EPS = 0.03    # 平台窗口内 close_frac 最大跨度
GRIPPER_STATE_CHANGE_DELTA = 0.1    # 平台间翻转状态的最小变化（> eps）
GRIPPER_STATE_SETTLED_FILL = 3      # 填补短 unsettled 空隙（帧）
GRIPPER_EARLY_CLOSED_FRAC = 0.85    # 前向稳定且 close_frac>=此值时提前确认闭合


# ============================ 向量化 FK ============================
def _rpy_matrix(rpy: Sequence[float]) -> np.ndarray:
    r, p, y = [float(v) for v in rpy]
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def _origin_matrix(origin: Optional[ET.Element]) -> np.ndarray:
    m = np.eye(4, dtype=np.float64)
    if origin is None:
        return m
    xyz = origin.get("xyz")
    rpy = origin.get("rpy")
    if xyz:
        m[:3, 3] = [float(v) for v in xyz.split()]
    if rpy:
        m[:3, :3] = _rpy_matrix([float(v) for v in rpy.split()])
    return m


class RS02Kinematics:
    """解析 RS-02 URDF，提供整段 episode 的向量化 FK（root=base_link）。"""

    def __init__(self, urdf_path: str):
        root = ET.parse(urdf_path).getroot()
        joints: Dict[str, Dict[str, Any]] = {}
        children_by_parent: Dict[str, List[Tuple[str, str]]] = {}
        link_parent_joint: Dict[str, str] = {}
        all_children: set = set()
        link_set: set = set()

        for j in root.findall("joint"):
            name = j.get("name")
            jtype = j.get("type")
            parent = j.find("parent").get("link") if j.find("parent") is not None else None
            child = j.find("child").get("link") if j.find("child") is not None else None
            origin = _origin_matrix(j.find("origin"))
            axis_el = j.find("axis")
            if axis_el is not None and axis_el.get("xyz"):
                axis = np.array([float(v) for v in axis_el.get("xyz").split()], dtype=np.float64)
                nrm = np.linalg.norm(axis)
                if nrm > 1e-12 and jtype in ("revolute", "continuous", "prismatic"):
                    axis = axis / nrm
            else:
                axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            mimic = None
            mim = j.find("mimic")
            if mim is not None and mim.get("joint"):
                mimic = (
                    mim.get("joint"),
                    float(mim.get("multiplier", 1.0) or 1.0),
                    float(mim.get("offset", 0.0) or 0.0),
                )
            joints[name] = {
                "type": jtype, "parent": parent, "child": child,
                "origin": origin, "axis": axis, "mimic": mimic,
            }
            if parent is not None:
                link_set.add(parent)
            if child is not None:
                link_set.add(child)
                all_children.add(child)
                link_parent_joint[child] = name
                children_by_parent.setdefault(parent, []).append((name, child))

        roots = sorted(l for l in link_set if l not in all_children)
        topo: List[str] = []
        seen = set(roots)
        dq = deque(roots)
        while dq:
            p = dq.popleft()
            topo.append(p)
            for _jn, c in children_by_parent.get(p, []):
                if c not in seen:
                    seen.add(c)
                    dq.append(c)

        self.joints = joints
        self.roots = roots
        self.topo = topo
        self.link_parent_joint = link_parent_joint
        self.mimic_joints = [n for n, i in joints.items() if i["mimic"] is not None]

    @staticmethod
    def _axis_rotation_batch(axis: np.ndarray, q: np.ndarray) -> np.ndarray:
        ax, ay, az = float(axis[0]), float(axis[1]), float(axis[2])
        c, s = np.cos(q), np.sin(q)
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

    def _joint_local_batch(self, info: Dict[str, Any], q: np.ndarray, t: int) -> np.ndarray:
        origin = info["origin"]
        jtype = info["type"]
        if jtype in ("revolute", "continuous"):
            motion = np.broadcast_to(np.eye(4), (t, 4, 4)).copy()
            motion[:, :3, :3] = self._axis_rotation_batch(info["axis"], q)
            return origin[None] @ motion
        if jtype == "prismatic":
            motion = np.broadcast_to(np.eye(4), (t, 4, 4)).copy()
            motion[:, :3, 3] = info["axis"][None, :] * q[:, None]
            return origin[None] @ motion
        return np.broadcast_to(origin, (t, 4, 4))

    def fk_batch(self, cfg: Dict[str, np.ndarray], t: int) -> Dict[str, np.ndarray]:
        """cfg: joint_name -> (t,) 角度/位移。返回 link -> (t,4,4) 世界变换（root=base_link）。"""
        cfg = dict(cfg)
        for jn in self.mimic_joints:
            src, mult, off = self.joints[jn]["mimic"]
            if src in cfg:
                cfg[jn] = mult * cfg[src] + off
        eye_t = np.broadcast_to(np.eye(4), (t, 4, 4))
        world: Dict[str, np.ndarray] = {r: eye_t for r in self.roots}
        for link in self.topo:
            if link in world:
                continue
            jn = self.link_parent_joint.get(link)
            if jn is None:
                world[link] = eye_t
                continue
            info = self.joints[jn]
            q = cfg.get(jn)
            if q is None:
                q = np.zeros(t, dtype=np.float64)
            world[link] = world[info["parent"]] @ self._joint_local_batch(info, q, t)
        return world


# ============================ 关节 -> URDF cfg ============================
# action(17) 与 observation.state(71) 的前 16 维布局一致：
#   [0:7] L臂, [7] L夹爪, [8:15] R臂, [15] R夹爪。
# action 第 17 维是 Lift.command（不画骨架），state 第 17 维是 Lift.position。
_L_ARM_IDX = list(range(0, 7))
_R_ARM_IDX = list(range(8, 15))
_L_GRIP_IDX = 7
_R_GRIP_IDX = 15


DEFAULT_GRIPPER_INVERT = True


def joints_to_cfg(arr: np.ndarray, gripper_invert: bool = DEFAULT_GRIPPER_INVERT
                  ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """把 (T,>=16) 的 action 或 state 映射为 URDF joint cfg。

    返回 (cfg, close_frac_left, close_frac_right)，其中 close_frac ∈[0,1]，HIGH=物理闭合
    （喂给夹爪时序状态机）。gripper.pos 原始 ∈[0,1] 驱动 URDF 关节；
    实测与 AgiBot 类似：pos 高≈张开、低≈闭合，故默认 gripper_invert=True，
    close_frac = 1 - pos。若颜色仍反，可 --no-gripper-invert。
    """
    arr = np.asarray(arr, dtype=np.float64)
    cfg: Dict[str, np.ndarray] = {}
    for k, i in enumerate(_L_ARM_IDX):
        cfg[LEFT_ARM_JOINTS[k]] = arr[:, i]
    for k, i in enumerate(_R_ARM_IDX):
        cfg[RIGHT_ARM_JOINTS[k]] = arr[:, i]
    lg = np.clip(arr[:, _L_GRIP_IDX], 0.0, 1.0)
    rg = np.clip(arr[:, _R_GRIP_IDX], 0.0, 1.0)
    # drive joint 限位 [0,1]：几何上开合位移很小，直接用原始分数驱动骨架末端
    cfg[LEFT_GRIPPER_DRIVE] = GRIPPER_DRIVE_MAX_Q * lg
    cfg[RIGHT_GRIPPER_DRIVE] = GRIPPER_DRIVE_MAX_Q * rg
    close_left = (1.0 - lg) if gripper_invert else lg
    close_right = (1.0 - rg) if gripper_invert else rg
    return cfg, close_left, close_right


def action_to_cfg(action: np.ndarray, gripper_invert: bool = False):
    """兼容旧接口：等价于 joints_to_cfg(action[:, :16])。"""
    return joints_to_cfg(action, gripper_invert=gripper_invert)


# ============================ 夹爪时序状态机（对齐 AgiBot beta）============================
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


def gripper_state_masks(
    close_frac_side: np.ndarray,
    closed_open_frac: float = GRIPPER_CLOSED_OPEN_FRAC,
    lookahead: int = GRIPPER_STATE_LOOKAHEAD,
    eps: float = GRIPPER_STATE_PLATEAU_EPS,
    change_delta: float = GRIPPER_STATE_CHANGE_DELTA,
    lookback: int = GRIPPER_STATE_LOOKBACK,
    settled_fill: int = GRIPPER_STATE_SETTLED_FILL,
    early_closed_frac: float = GRIPPER_EARLY_CLOSED_FRAC,
) -> Tuple[np.ndarray, np.ndarray]:
    """逐侧时序状态机，返回 (closed_mask, settled_mask)。

    close_frac_side: (T,) HIGH=物理闭合。
    closed_mask: True=闭合(橙)、False=张开(蓝)；运动帧保持上一确认态（迟滞）。
    settled_mask: True=平台稳定帧（骨线用二值色）；False=运动帧（骨线按 close_frac lerp）。
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
        # Early CLOSED: 已紧闭且前向窗口平坦时，不等 lookback 清完斜坡。
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


def gripper_closed_mask(
    close_frac_side: np.ndarray,
    closed_open_frac: float = GRIPPER_CLOSED_OPEN_FRAC,
    lookahead: int = GRIPPER_STATE_LOOKAHEAD,
    eps: float = GRIPPER_STATE_PLATEAU_EPS,
    change_delta: float = GRIPPER_STATE_CHANGE_DELTA,
    lookback: int = GRIPPER_STATE_LOOKBACK,
    settled_fill: int = GRIPPER_STATE_SETTLED_FILL,
    early_closed_frac: float = GRIPPER_EARLY_CLOSED_FRAC,
) -> np.ndarray:
    """兼容旧接口：仅返回 closed_mask。"""
    closed, _settled = gripper_state_masks(
        close_frac_side, closed_open_frac, lookahead, eps, change_delta,
        lookback=lookback, settled_fill=settled_fill,
        early_closed_frac=early_closed_frac,
    )
    return closed


def gripper_bone_colors_from_mask(
    closed_mask: np.ndarray,
    color_open: tuple,
    color_closed: tuple,
) -> np.ndarray:
    """closed_mask -> (T,3) BGR 二值色。"""
    co = np.asarray(color_open, dtype=np.int64)
    cc = np.asarray(color_closed, dtype=np.int64)
    return np.where(np.asarray(closed_mask, dtype=bool)[:, None], cc[None, :], co[None, :])


def gripper_colors_state_or_gradient(
    close_frac: np.ndarray,
    closed_mask: np.ndarray,
    settled_mask: np.ndarray,
    color_open: tuple,
    color_closed: tuple,
    color_motion: tuple = COLOR_GRIPPER_MOTION_BONE,
) -> np.ndarray:
    """Settled 帧用二值开合色；运动（unsettled）帧统一灰色（无蓝↔橙渐变）。

    close_frac 参数保留以兼容旧调用签名，运动色不再依赖其数值。
    """
    del close_frac  # unused: motion uses solid gray
    binary = gripper_bone_colors_from_mask(closed_mask, color_open, color_closed)
    settled = np.asarray(settled_mask, dtype=bool)
    motion = np.asarray(color_motion, dtype=np.int64)
    out = np.where(settled[:, None], binary, motion[None, :])
    return out.astype(np.int64)


def gripper_joint_colors_state_or_motion(
    closed_mask: np.ndarray,
    settled_mask: np.ndarray,
    color_open: tuple = COLOR_GRIPPER_OPEN_JOINT,
    color_closed: tuple = COLOR_GRIPPER_CLOSED_JOINT,
    color_motion: tuple = COLOR_GRIPPER_MOTION_JOINT,
) -> np.ndarray:
    """关节点：settled 二值开合色（开红/闭绿），运动帧固定红色。"""
    binary = gripper_bone_colors_from_mask(closed_mask, color_open, color_closed)
    settled = np.asarray(settled_mask, dtype=bool)
    motion = np.asarray(color_motion, dtype=np.int64)
    return np.where(settled[:, None], binary, motion[None, :]).astype(np.int64)


# ============================ 投影 ============================
def hand_eye_to_view(R: Sequence[Sequence[float]], T: Sequence[float]) -> np.ndarray:
    """把 camera.json 的 hand_eye {R,T}（= 头相机在臂 base 系下的位姿, base<-cam）
    转成 world(臂base)->camera 的 4x4 view 矩阵（OpenCV 光学系直接用）。"""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T = np.asarray(T, dtype=np.float64).reshape(3)
    cam_t = np.eye(4, dtype=np.float64)  # cam 在 base 系下的位姿
    cam_t[:3, :3] = R
    cam_t[:3, 3] = T
    return np.linalg.inv(cam_t)  # world(base)->cam


def project_points(pts_base: np.ndarray, view: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """pts_base (N,3) 在臂 base 系；view world->cam (4,4)；K (3,3)。
    返回 (uv (N,2) int, depth (N,))。OpenCV 针孔：u=fx*x/z+cx。"""
    n = pts_base.shape[0]
    homo = np.concatenate([pts_base, np.ones((n, 1))], axis=1)
    cam = (view @ homo.T).T[:, :3]
    depth = cam[:, 2].copy()
    safe = np.where(np.abs(depth) < 1e-6, 1e-6, depth)
    uv = (K @ cam.T).T
    uv = uv[:, :2] / safe[:, None]
    return uv, depth


def project_points_batch(pts_base: np.ndarray, view: np.ndarray,
                         K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """pts_base (T,3) 轨迹（同一 base 系，静止相机）。返回 uv (T,2) float, depth (T,)。"""
    t = pts_base.shape[0]
    homo = np.concatenate([pts_base, np.ones((t, 1))], axis=1)
    cam = (view @ homo.T).T[:, :3]
    depth = cam[:, 2].copy()
    safe = np.where(np.abs(depth) < 1e-6, 1e-6, depth)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = fx * cam[:, 0] / safe + cx
    v = fy * cam[:, 1] / safe + cy
    return np.stack([u, v], axis=1), depth


def load_head_K(camera_json_entry: dict, resolution: str = "720P") -> np.ndarray:
    """从 camera.json 单 episode 条目取 head 相机内参 K (3,3)。"""
    mtx = camera_json_entry["camera_parameters"]["head"][resolution]["MTX"]
    return np.asarray(mtx, dtype=np.float64).reshape(3, 3)


def scale_K(K: np.ndarray, src_wh: Tuple[int, int], dst_wh: Tuple[int, int]) -> np.ndarray:
    """把内参从 src 分辨率缩放到 dst 分辨率。"""
    sx = dst_wh[0] / src_wh[0]
    sy = dst_wh[1] / src_wh[1]
    K2 = K.copy()
    K2[0, 0] *= sx; K2[0, 2] *= sx
    K2[1, 1] *= sy; K2[1, 2] *= sy
    return K2
