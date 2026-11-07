"""
RoboDojo policy_mix24 (LeRobot v3.0, 双臂 ARX X5 仿真) 数据读取工具

数据集特点 (policy_mix24_clean2400_sdk_ee20_25hz_instruction_20261029_v2)：
- 24 个场景 × 100 条 = 2400 条，25 Hz，三路 640x480 RGB (cam_high / cam_left_wrist / cam_right_wrist)
- LeRobot v3.0 布局：多条 episode 拼接在同一个 mp4 / parquet 中，
  ``meta/episodes`` 给出每条在视频中的 [from_timestamp, to_timestamp)
- 视频编码混合 (H264 yuv420p / H264 yuv444p / AV1)，训练前需按 episode 重新导出
- 主 action 为 EE20 (xyz + rot6d + gripper01, 每臂 10 维)；``action.joint`` 为
  Joint14 = 每臂 6 关节 + gripper01，``action.joint[t] == observation.joint[t+1]``
- gripper01: 1 = 张开, 0 = 闭合 (与 RoboTwin gripper_val 同向)
- 自采 episode 末帧 ``action_valid=False`` (terminal padding)
- ``SAMPLE_MAPPING.jsonl`` 给出每条的原始 HDF5 (官方 RoboDojo / 自采)，路径前缀
  ``/mnt/public_ckp`` 在本机挂载为 ``/datasets``

几何 (已对 24 个场景逐一核对，FK(observation.joint) 与 observation.state xyz 误差 < 0.2 mm)：
- 每臂 X5A URDF，base_link → link6；双臂 base 来自 ``arx_x5/dual_x5.yml``
  (自采 HDF5 额外记录 ``state/{side}_arm_base_pose``，优先使用)
- 头部相机静止；HDF5 ``vision/cam_head/extrinsic_matrix`` 为 USD 约定 T_world_camera，
  OpenCV 约定 = USD @ diag(1, -1, -1, 1)
- 内参逐 episode 读取：自采 pack_mixed_objects_into_box_v1 的 fy 与其他场景不同
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_DATASET_ROOT = (
    "/datasets/UniEmbodied-Bench/simulation/dataset/"
    "policy_mix24_clean2400_sdk_ee20_25hz_instruction_20261029_v2"
)
DEFAULT_PATH_REMAP: Tuple[Tuple[str, str], ...] = (
    ("/mnt/public_ckp/UniEmbodied-Bench", "/datasets/UniEmbodied-Bench"),
    ("/mnt/public_ckp/pku-data", "/datasets/pku-data"),
)

FPS = 25.0
# LeRobot 相机名 → 原始 HDF5 相机名
CAMERA_KEYS: Dict[str, str] = {
    "cam_high": "cam_head",
    "cam_left_wrist": "cam_left_wrist",
    "cam_right_wrist": "cam_right_wrist",
}
JOINT14_NAMES = [
    "left_joint1", "left_joint2", "left_joint3", "left_joint4", "left_joint5", "left_joint6",
    "left_gripper01",
    "right_joint1", "right_joint2", "right_joint3", "right_joint4", "right_joint5", "right_joint6",
    "right_gripper01",
]
ARM_DOF = 6

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_URDF = os.path.join(_SCRIPT_DIR, "arx_x5", "X5A.urdf")
DEFAULT_ROBOT_CFG = os.path.join(_SCRIPT_DIR, "arx_x5", "dual_x5.yml")

_USD_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])

_FRAME_COLUMNS = (
    "frame_index",
    "observation.joint",
    "action.joint",
    "observation.state",
    "action",
    "action_valid",
    "action.joint_valid",
    "source.rgb_frame_index",
)


@dataclass
class EpisodeRef:
    episode_index: int
    scene: str
    instruction: str
    length: int
    official: bool
    classification: str
    source_episode_id: str
    hdf5: str
    data_file: str
    camera: str
    video_file: str
    from_timestamp: float
    to_timestamp: float
    extra: Dict[str, object] = field(default_factory=dict)

    @property
    def uuid(self) -> str:
        return f"{self.scene}/episode_{self.episode_index:06d}"

    def to_dict(self) -> dict:
        return asdict(self)


def remap_path(path: str, remap: Sequence[Tuple[str, str]] = DEFAULT_PATH_REMAP) -> str:
    for old, new in remap:
        if path.startswith(old):
            return new + path[len(old):]
    return path


def _read_info(root: str) -> dict:
    with open(os.path.join(root, "meta", "info.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def _read_episodes_table(root: str, camera: str):
    import pandas as pd

    vkey = f"videos/observation.images.{camera}"
    cols = [
        "episode_index", "tasks", "length",
        "data/chunk_index", "data/file_index",
        f"{vkey}/chunk_index", f"{vkey}/file_index",
        f"{vkey}/from_timestamp", f"{vkey}/to_timestamp",
    ]
    ep_dir = os.path.join(root, "meta", "episodes")
    files = sorted(
        os.path.join(dp, fn)
        for dp, _, fns in os.walk(ep_dir)
        for fn in fns
        if fn.endswith(".parquet")
    )
    if not files:
        raise FileNotFoundError(f"no episode parquet under {ep_dir}")
    return pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)


def _read_sample_mapping(root: str) -> Dict[int, dict]:
    path = os.path.join(root, "SAMPLE_MAPPING.jsonl")
    out: Dict[int, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                out[int(item["episode_index"])] = item
    return out


def load_episode_refs(
    root: str = DEFAULT_DATASET_ROOT,
    camera: str = "cam_high",
    remap: Sequence[Tuple[str, str]] = DEFAULT_PATH_REMAP,
) -> List[EpisodeRef]:
    """按 episode_index 排序返回全部 episode 的元数据 (不读帧数据)。"""
    if camera not in CAMERA_KEYS:
        raise ValueError(f"camera must be one of {sorted(CAMERA_KEYS)}, got {camera!r}")
    info = _read_info(root)
    data_tpl = info["data_path"]
    video_tpl = info["video_path"]
    episodes = _read_episodes_table(root, camera)
    mapping = _read_sample_mapping(root)
    vkey = f"videos/observation.images.{camera}"

    refs: List[EpisodeRef] = []
    for r in episodes.to_dict("records"):
        ep = int(r["episode_index"])
        m = mapping.get(ep)
        if m is None:
            raise KeyError(f"episode {ep} missing from SAMPLE_MAPPING.jsonl")
        tasks = list(r["tasks"]) if r["tasks"] is not None else []
        instruction = str(tasks[0]) if tasks else str(m.get("training_instruction") or "")
        data_file = os.path.join(root, data_tpl.format(
            chunk_index=int(r["data/chunk_index"]), file_index=int(r["data/file_index"])))
        video_file = os.path.join(root, video_tpl.format(
            video_key=f"observation.images.{camera}",
            chunk_index=int(r[f"{vkey}/chunk_index"]),
            file_index=int(r[f"{vkey}/file_index"]),
        ))
        refs.append(EpisodeRef(
            episode_index=ep,
            scene=str(m["task"]),
            instruction=instruction,
            length=int(r["length"]),
            official=bool(m.get("official", False)),
            classification=str(m.get("classification") or ""),
            source_episode_id=str(m.get("source_episode_id") or ""),
            hdf5=remap_path(str(m.get("hdf") or ""), remap),
            data_file=data_file,
            camera=camera,
            video_file=video_file,
            from_timestamp=float(r[f"{vkey}/from_timestamp"]),
            to_timestamp=float(r[f"{vkey}/to_timestamp"]),
        ))
    refs.sort(key=lambda x: x.episode_index)
    return refs


def _stack_column(values) -> np.ndarray:
    return np.stack([np.asarray(v).reshape(-1) for v in values])


def read_episode_frames(ref: EpisodeRef) -> Dict[str, np.ndarray]:
    """读取单条 episode 的逐帧数值 (按 frame_index 排序)。"""
    import pyarrow.parquet as pq

    table = pq.read_table(
        ref.data_file,
        columns=["episode_index", *_FRAME_COLUMNS],
        filters=[("episode_index", "=", ref.episode_index)],
    )
    df = table.to_pandas().sort_values("frame_index")
    if len(df) != ref.length:
        raise ValueError(
            f"episode {ref.episode_index}: parquet rows {len(df)} != meta length {ref.length}")
    frames = {
        "frame_index": df["frame_index"].to_numpy(np.int64),
        "observation_joint": _stack_column(df["observation.joint"]).astype(np.float32),
        "action_joint_raw": _stack_column(df["action.joint"]).astype(np.float32),
        "observation_state": _stack_column(df["observation.state"]).astype(np.float32),
        "action_ee20": _stack_column(df["action"]).astype(np.float32),
        "action_valid": _stack_column(df["action_valid"]).reshape(-1).astype(bool),
        "action_joint_valid": _stack_column(df["action.joint_valid"]).reshape(-1).astype(bool),
        "rgb_frame_index": _stack_column(df["source.rgb_frame_index"]).reshape(-1).astype(np.int64),
    }
    frames["action_joint"] = sanitize_action_joint(
        frames["action_joint_raw"], frames["action_joint_valid"], frames["observation_joint"])
    frames["image_joint"] = image_time_joint(frames)
    return frames


def image_time_joint(frames: Dict[str, np.ndarray]) -> np.ndarray:
    """RGB 第 k 帧拍摄时刻的 Joint14 (与 RoboTwin ``joint_action`` 同为拍图时刻的关节状态)。

    官方与自采 episode 都是 ``observation.joint[k]``：把 observation.joint[k+o] 的骨架投影到
    RGB[k] 上、按黑色 ARX 臂上的像素亮度打分，130 条 episode 中 o=0 胜出 100 条，o=+1
    (= action.joint[k]) 只胜 5 条。官方原始 HDF5 的图像索引比 state 晚一步，LeRobot 导出
    的 ``rgb_frame_index = obs + 1`` 正是对它的校正，不能按原始索引相同推断同时刻。
    """
    return np.asarray(frames["observation_joint"], dtype=np.float32).copy()


def sanitize_action_joint(
    action_joint: np.ndarray, valid: np.ndarray, observation_joint: np.ndarray,
) -> np.ndarray:
    """无效帧 (terminal padding) 沿用上一有效 action；开头即无效时退回同帧 observation。"""
    out = np.asarray(action_joint, dtype=np.float32).copy()
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    last: Optional[np.ndarray] = None
    for t in range(out.shape[0]):
        if valid[t] and np.all(np.isfinite(out[t])):
            last = out[t]
        elif last is not None:
            out[t] = last
        else:
            out[t] = observation_joint[t]
    return out


def split_joint14(joint14: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """[T,14] → left_arm [T,6], left_gripper [T], right_arm [T,6], right_gripper [T]。"""
    j = np.asarray(joint14, dtype=np.float64)
    return j[:, 0:6], j[:, 6], j[:, 7:13], j[:, 13]


def pose7_to_matrix(pose7: Sequence[float]) -> np.ndarray:
    """[x, y, z, qw, qx, qy, qz] → 4x4。"""
    from scipy.spatial.transform import Rotation

    p = np.asarray(pose7, dtype=np.float64).reshape(7)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = Rotation.from_quat([p[4], p[5], p[6], p[3]]).as_matrix()
    out[:3, 3] = p[:3]
    return out


def load_default_arm_bases(robot_cfg: str = DEFAULT_ROBOT_CFG) -> Dict[str, np.ndarray]:
    """dual_x5.yml 中 robots[0]=左臂、robots[1]=右臂 的 root pose。"""
    import yaml

    with open(robot_cfg, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    robots = cfg["robots"]
    out = {}
    for side, r in zip(("left", "right"), robots[:2]):
        out[side] = pose7_to_matrix([*r["default_root_pos"], *r["default_root_rot"]])
    return out


def load_episode_geometry(
    hdf5_path: str,
    raw_camera: str = "cam_head",
    frame_index: int = 0,
    default_bases: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, object]:
    """从原始 HDF5 读取相机内参、OpenCV 外参 (T_world_camera) 与双臂 base。

    只读取小数组，不触碰 HDF5 中的图像字节流。
    """
    import h5py

    out: Dict[str, object] = {}
    with h5py.File(hdf5_path, "r") as f:
        g = f[f"vision/{raw_camera}"]
        out["intrinsic"] = np.asarray(g["intrinsic_matrix"][()], dtype=np.float64)
        ext = g["extrinsic_matrix_opencv"] if "extrinsic_matrix_opencv" in g else None
        if ext is not None:
            n = ext.shape[0]
            cam_T_world = np.asarray(ext[min(int(frame_index), n - 1)], dtype=np.float64)
            out["extrinsic_static"] = bool(np.allclose(ext[0], ext[n - 1], atol=1e-6))
        else:
            usd = g["extrinsic_matrix"]
            n = usd.shape[0]
            cam_T_world = np.asarray(usd[min(int(frame_index), n - 1)], dtype=np.float64) @ _USD_TO_OPENCV
            out["extrinsic_static"] = bool(np.allclose(usd[0], usd[n - 1], atol=1e-6))
        out["T_world_camera_opencv"] = cam_T_world
        bases: Dict[str, np.ndarray] = {}
        for side in ("left", "right"):
            key = f"state/{side}_arm_base_pose"
            if key in f:
                bases[side] = pose7_to_matrix(f[key][0])
        out["base_source"] = "hdf5" if len(bases) == 2 else "dual_x5.yml"
        if len(bases) != 2:
            bases = dict(default_bases or load_default_arm_bases())
        out["bases"] = bases
        shape = g["shape"][()] if "shape" in g else None
        if shape is not None:
            out["image_hw"] = (int(shape[0]), int(shape[1]))
    return out


def load_camera_track(
    hdf5_path: str,
    raw_camera: str,
    default_bases: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, object]:
    """Per-frame OpenCV camera pose plus the intrinsics and arm bases.

    ``T_world_camera_opencv`` is ``(N, 4, 4)`` and row ``rgb_frame_index[k]`` is the
    camera pose at RGB frame ``k``. Wrist cameras move with the arm, so a single
    pose would freeze the skeleton while the image moves.
    """
    import h5py

    with h5py.File(hdf5_path, "r") as f:
        g = f[f"vision/{raw_camera}"]
        intrinsic = np.asarray(g["intrinsic_matrix"][()], dtype=np.float64)
        if "extrinsic_matrix_opencv" in g:
            extrinsic = np.asarray(g["extrinsic_matrix_opencv"][()], dtype=np.float64)
        else:
            extrinsic = np.asarray(g["extrinsic_matrix"][()], dtype=np.float64) @ _USD_TO_OPENCV
        if extrinsic.ndim == 2:
            extrinsic = extrinsic[None]
        bases: Dict[str, np.ndarray] = {}
        for side in ("left", "right"):
            key = f"state/{side}_arm_base_pose"
            if key in f:
                bases[side] = pose7_to_matrix(f[key][0])
    if len(bases) != 2:
        bases = dict(default_bases or load_default_arm_bases())
    return {
        "intrinsic": intrinsic,
        "T_world_camera_opencv": np.ascontiguousarray(extrinsic),
        "bases": bases,
    }


def extract_episode_clip(
    ref: EpisodeRef,
    out_path: str,
    *,
    crf: int = 18,
    preset: str = "veryfast",
    threads: int = 1,
    fps: float = FPS,
    timeout: float = 1800.0,
) -> Tuple[int, str]:
    """把拼接视频中的 [from_timestamp, from_timestamp + length/fps) 导出为 H264 yuv420p mp4。

    ``-ss`` 放在 ``-i`` 前并重编码时 ffmpeg 会解码丢弃到精确位置；起点往前让半帧，
    避免浮点时间戳恰好落在帧边界时丢掉首帧。写 ``.part`` 后原子替换。
    """
    start = max(0.0, ref.from_timestamp - 0.5 / fps)
    tmp = out_path + ".part"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{start:.6f}", "-i", ref.video_file,
        "-frames:v", str(int(ref.length)),
        "-an", "-c:v", "libx264", "-preset", preset, "-crf", str(int(crf)),
        "-threads", str(max(1, int(threads))),
        "-pix_fmt", "yuv420p", "-r", f"{fps:g}",
        "-movflags", "+faststart", "-f", "mp4", tmp,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode == 0 and os.path.exists(tmp) and os.path.getsize(tmp) > 0:
        os.replace(tmp, out_path)
        return 0, ""
    try:
        os.remove(tmp)
    except OSError:
        pass
    return proc.returncode or -1, (proc.stderr or "").strip()[-300:]


def probe_video_frames(video_path: str, timeout: float = 30.0) -> int:
    """ffprobe 读帧数 (nb_frames, 缺失时 count_packets)；失败返回 -1。"""
    def _query(extra: List[str], key: str) -> str:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", *extra,
             "-show_entries", f"stream={key}", "-of", "default=noprint_wrappers=1", video_path],
            capture_output=True, text=True, timeout=timeout,
        ).stdout
        for line in out.splitlines():
            k, _, v = line.partition("=")
            if k.strip() == key:
                return v.strip()
        return ""

    try:
        nb = _query([], "nb_frames")
        if nb.isdigit():
            return int(nb)
        nb = _query(["-count_packets"], "nb_read_packets")
        return int(nb) if nb.isdigit() else -1
    except (subprocess.SubprocessError, OSError, ValueError):
        return -1


__all__ = [
    "ARM_DOF",
    "CAMERA_KEYS",
    "DEFAULT_DATASET_ROOT",
    "DEFAULT_PATH_REMAP",
    "DEFAULT_ROBOT_CFG",
    "DEFAULT_URDF",
    "EpisodeRef",
    "FPS",
    "JOINT14_NAMES",
    "extract_episode_clip",
    "image_time_joint",
    "load_default_arm_bases",
    "load_episode_geometry",
    "load_episode_refs",
    "pose7_to_matrix",
    "probe_video_frames",
    "read_episode_frames",
    "remap_path",
    "sanitize_action_joint",
    "split_joint14",
]
