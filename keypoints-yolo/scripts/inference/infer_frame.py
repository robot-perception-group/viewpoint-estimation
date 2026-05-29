# Run YOLO keypoint inference on a single image (frame) and return keypoints

import sys
import os

# Add parent directory to sys.path for local import
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from yolo_keypoints_utils import KEYPOINT_NAMES, extract_keypoints, load_model

def infer_frame(img, weights_path, device="cuda:0", conf=0.5):
    """
    Run YOLO keypoint inference on a single image (numpy array) and return keypoints for all detected animals.
    Returns a list of dicts, one per detected animal.
    """
    if img is None:
        raise ValueError("Input image is None")
    model = load_model(weights_path, device)
    results = model.predict(source=img, conf=conf, device=device, verbose=False)
    result = results[0] if results else None
    if result is None or result.keypoints is None or result.keypoints.xy is None:
        return []
    all_keypoints = []
    for idx in range(result.keypoints.xy.shape[0]):
        kpts = extract_keypoints(result, idx)
        if kpts is None:
            continue
        keypoints = [
            {"name": KEYPOINT_NAMES[i], "x": float(x), "y": float(y), "conf": float(c)}
            for i, ((x, y), c) in enumerate(zip(kpts.xy, kpts.conf))
        ]
        all_keypoints.append({"keypoints": keypoints})
    return all_keypoints
