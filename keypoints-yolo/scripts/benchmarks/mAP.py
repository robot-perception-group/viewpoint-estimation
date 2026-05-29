# Standalone mAP benchmark script.

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from ultralytics import YOLO


MODEL_PATH = ".../keypoints-yolo/models/pose/cropped_models/quadruped-standalone-full-cropped/weights/best.pt"
DATA_YAML = ".../keypoints-yolo/config/test.yaml"
SPLIT = "test"
IMGSZ = 640
BATCH = 16
DEVICE = 0
OUTPUT_JSON = "map_result.json"
SAVE_JSON = False


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        if isinstance(value, (float, int)):
            if math.isnan(value):
                return None
            return float(value)
        parsed = float(value)
        if math.isnan(parsed):
            return None
        return parsed
    except Exception:
        return None


def _extract_val_metrics(metrics: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "pose": {
            "map50_95": None,
            "map50": None,
            "map75": None,
            "maps": None,
        },
        "box": {
            "map50_95": None,
            "map50": None,
            "map75": None,
            "maps": None,
        },
        "speed_ms_per_image": {},
    }

    if hasattr(metrics, "pose"):
        out["pose"]["map50_95"] = _to_float(getattr(metrics.pose, "map", None))
        out["pose"]["map50"] = _to_float(getattr(metrics.pose, "map50", None))
        out["pose"]["map75"] = _to_float(getattr(metrics.pose, "map75", None))
        maps = getattr(metrics.pose, "maps", None)
        if maps is not None:
            out["pose"]["maps"] = [
                _to_float(v) for v in maps.tolist() if _to_float(v) is not None
            ]

    if hasattr(metrics, "box"):
        out["box"]["map50_95"] = _to_float(getattr(metrics.box, "map", None))
        out["box"]["map50"] = _to_float(getattr(metrics.box, "map50", None))
        out["box"]["map75"] = _to_float(getattr(metrics.box, "map75", None))
        maps = getattr(metrics.box, "maps", None)
        if maps is not None:
            out["box"]["maps"] = [
                _to_float(v) for v in maps.tolist() if _to_float(v) is not None
            ]

    speed = getattr(metrics, "speed", None)
    if isinstance(speed, dict):
        out["speed_ms_per_image"] = {str(k): _to_float(v) for k, v in speed.items()}

    return out


def run_map_benchmark(
    model_path: str,
    data_yaml: str,
    split: str = "test",
    imgsz: int = 640,
    batch: int = 16,
    device: str = "cpu",
) -> dict[str, Any]:
    model = YOLO(str(model_path))
    metrics = model.val(data=data_yaml, split=split, imgsz=imgsz, batch=batch, device=device)
    return _extract_val_metrics(metrics)


def main() -> None:
    result = run_map_benchmark(
        model_path=MODEL_PATH,
        data_yaml=DATA_YAML,
        split=SPLIT,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
    )

    print(f"Pose mAP@0.5:0.95: {result['pose']['map50_95']}")
    print(f"Pose mAP@0.5: {result['pose']['map50']}")

    if SAVE_JSON:
        output_path = Path(OUTPUT_JSON).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Saved mAP result to: {output_path}")


if __name__ == "__main__":
    main()