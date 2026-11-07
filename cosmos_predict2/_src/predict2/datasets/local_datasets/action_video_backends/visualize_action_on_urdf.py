"""Shared arm constants, FPS and transform helpers; no standalone hand renderer."""
from typing import Optional
import json
import subprocess
import numpy as np

LEFT_ARM_JOINTS = [
    "shoulder_pitch_l_joint",
    "shoulder_roll_l_joint",
    "shoulder_yaw_l_joint",
    "elbow_pitch_l_joint",
    "elbow_yaw_l_joint",
    "wrist_pitch_l_joint",
    "wrist_roll_l_joint",
]

RIGHT_ARM_JOINTS = [
    "shoulder_pitch_r_joint",
    "shoulder_roll_r_joint",
    "shoulder_yaw_r_joint",
    "elbow_pitch_r_joint",
    "elbow_yaw_r_joint",
    "wrist_pitch_r_joint",
    "wrist_roll_r_joint",
]

def detect_fps(video_path: str) -> float:
    """用 ffprobe 检测视频 fps。"""
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", video_path],
            stderr=subprocess.STDOUT,
        )
        info = json.loads(out)
        for s in info.get("streams", []):
            if s.get("codec_type") == "video":
                rate = s.get("r_frame_rate", "18/1")
                if "/" in rate:
                    num, den = rate.split("/")
                    return float(num) / float(den)
                return float(rate)
    except Exception:
        pass
    return 18.0

def _as_transform(x) -> np.ndarray:
    if x is None:
        return np.eye(4, dtype=np.float64)
    arr = np.asarray(x, dtype=np.float64)
    if arr.shape == (4, 4):
        return arr
    raise ValueError(f"Unexpected transform shape: {arr.shape}")

def _link_name(x) -> Optional[str]:
    if x is None:
        return None
    if hasattr(x, "name"):
        n = getattr(x, "name")
        return str(n) if n is not None else None
    return str(x)
