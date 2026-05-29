from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from ultralytics import YOLO
except ImportError as exc:
    raise SystemExit("Ultralytics not installed: pip install ultralytics") from exc

KEYPOINT_NAMES = [
    "l_eye",        # 0
    "r_eye",        # 1
    "nose",         # 2
    "neck",         # 3
    "tail",         # 4
    "l_f_shoulder", # 5 - left front shoulder
    "l_f_elbow",    # 6 - left front elbow
    "l_f_foot",     # 7 - left front foot
    "r_f_shoulder", # 8 - right front shoulder
    "r_f_elbow",    # 9 - right front elbow
    "r_f_foot",     # 10 - right front foot
    "l_b_hip",      # 11 - left back hip
    "l_b_elbow",    # 12 - left back elbow
    "l_b_foot",     # 13 - left back foot
    "r_b_hip",      # 14 - right back hip
    "r_b_elbow",    # 15 - right back elbow
    "r_b_foot",     # 16 - right back foot
]
SKELETON_EDGES = [
    (0, 2), (1, 2), (2, 3), (3, 4),
    (3, 5), (5, 6), (6, 7),
    (3, 8), (8, 9), (9, 10),
    (4, 11), (11, 12), (12, 13),
    (4, 14), (14, 15), (15, 16),
]

NAME_TO_IDX = {name: idx for idx, name in enumerate(KEYPOINT_NAMES)}


@dataclass
class KeypointResult:
    xy: np.ndarray  # (K, 2)
    conf: np.ndarray  # K

def load_model(weights: Path, device: str):
    return YOLO(str(weights))


def extract_keypoints(result, idx: int) -> Optional[KeypointResult]:
    if result is None or result.keypoints is None:
        return None
    xy = result.keypoints.xy
    conf = result.keypoints.conf
    if xy is None or conf is None:
        return None
    if xy.shape[0] == 0:
        return None
    xy_np = xy[idx].cpu().numpy()
    conf_np = conf[idx].cpu().numpy()
    return KeypointResult(xy=xy_np, conf=conf_np)
