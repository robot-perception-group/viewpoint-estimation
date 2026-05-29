# Standalone normal PCK benchmark script.



from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO


MODEL_PATH = ".../keypoints-yolo/models/pose/cropped_models/quadruped-standalone-full-cropped/weights/best.pt"
IMG_DIR = ".../keypoints-yolo/data/raw/converted_merged_MINIMAL_split/images/test"
LABEL_DIR = ".../keypoints-yolo/data/raw/converted_merged_MINIMAL_split/labels/test"
THRESHOLD_FACTORS = [0.05]
OUTPUT_JSON = "pck_result.json"
SAVE_JSON = False


def calculate_pck(
    model_path: str,
    img_dir: str,
    label_dir: str,
    threshold_factor: float = 0.05,
) -> dict[str, Any]:
    model = YOLO(str(model_path))
    img_files = sorted(Path(img_dir).glob("*.jpg"))

    total_images = 0
    used_images = 0
    total_visible_keypoints = 0
    correct_keypoints = 0

    for img_path in img_files:
        total_images += 1
        label_path = Path(label_dir) / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue

        results = model(str(img_path), verbose=False)[0]
        if results.keypoints is None or results.keypoints.xy is None or len(results.keypoints.xy) == 0:
            continue

        used_images += 1

        with open(label_path, "r", encoding="utf-8") as f:
            gt_lines = f.readlines()

        img_h, img_w = results.orig_shape

        for gt_idx, gt_line in enumerate(gt_lines):
            gt_data = np.array(gt_line.split(), dtype=np.float32)
            if gt_data.shape[0] < 5 + 3:
                continue

            bbox_w, bbox_h = gt_data[3] * img_w, gt_data[4] * img_h
            diag = np.sqrt(bbox_w**2 + bbox_h**2)
            threshold = threshold_factor * diag

            gt_kpts = gt_data[5:].reshape(-1, 3)
            gt_xy = gt_kpts[:, :2] * [img_w, img_h]
            visibility = gt_kpts[:, 2]

            if gt_idx >= len(results.keypoints.xy):
                continue
            pred_kpts = results.keypoints.xy[gt_idx].cpu().numpy()

            for i in range(min(len(gt_xy), len(pred_kpts))):
                if visibility[i] > 0:
                    total_visible_keypoints += 1
                    dist = np.linalg.norm(pred_kpts[i] - gt_xy[i])
                    if dist <= threshold:
                        correct_keypoints += 1

    score = (correct_keypoints / total_visible_keypoints) if total_visible_keypoints > 0 else None

    return {
        "threshold_factor": threshold_factor,
        "score": float(score) if score is not None else None,
        "score_percent": float(score * 100) if score is not None else None,
        "total_images": total_images,
        "used_images": used_images,
        "total_visible_keypoints": total_visible_keypoints,
        "correct_keypoints": correct_keypoints,
    }


def run_pck_benchmark(
    model_path: str,
    img_dir: str,
    label_dir: str,
    threshold_factors: list[float],
) -> dict[str, Any]:
    return {
        f"pck@{thr}": calculate_pck(
            model_path=model_path,
            img_dir=img_dir,
            label_dir=label_dir,
            threshold_factor=thr,
        )
        for thr in threshold_factors
    }


def main() -> None:
    result = run_pck_benchmark(
        model_path=MODEL_PATH,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
        threshold_factors=THRESHOLD_FACTORS,
    )

    for key, value in result.items():
        print(f"{key}: {value['score_percent']}")

    if SAVE_JSON:
        output_path = Path(OUTPUT_JSON).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Saved PCK result to: {output_path}")


if __name__ == "__main__":
    main()