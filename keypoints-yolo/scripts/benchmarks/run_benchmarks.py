"""
Run all benchmarks for a list of models across test domains
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure local benchmark modules are importable
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from PCK import run_pck_benchmark
from mAP import run_map_benchmark

TEST_SETS = {
    "sideview": {
        "yaml": ".../keypoints-yolo/config/DELETEbenchmark/sideview.yaml",
        "img_dir": ".../keypoints-yolo/data/processed/converted_merged_MINIMAL_split_bbox_crops/images/test",
        "label_dir": ".../keypoints-yolo/data/processed/converted_merged_MINIMAL_split_bbox_crops/labels/test",
        "split": "test",
    },
    "combined": {
        "yaml": ".../keypoints-yolo/config/DELETEbenchmark/combined.yaml",
        "img_dir": ".../keypoints-yolo/data/processed/finetune_plus_minimal_eval/images/test",
        "label_dir": ".../keypoints-yolo/data/processed/finetune_plus_minimal_eval/labels/test",
        "split": "test",
    },
    "uav": {
        "yaml": ".../keypoints-yolo/config/DELETEbenchmark/uav.yaml",
        "img_dir": ".../keypoints-yolo/data/processed/finetune_bbox/images/val",
        "label_dir": ".../keypoints-yolo/data/processed/finetune_bbox/labels/val",
        "split": "val",
    },
}

MODEL_PATHS: list[str] = [
    ".../keypoints-yolo/models/pose/cropped_models/quadruped-standalone-full-cropped/weights/best.pt",
    ".../keypoints-yolo/models/pose//cropped_models/quadruped-standalone-pruned-cropped/weights/best.pt",
    ".../keypoints-yolo/models/pose//cropped_models/quadruped-standalone-minimal-cropped/weights/best.pt",
    ".../keypoints-yolo/models/pose//finetune/minimal_cropped_finetune/weights/best.pt",
    ".../keypoints-yolo/models/pose//overfit_finetune/overfit_minimal_cropped/weights/best.pt"

]

OUTPUT_JSON = ".../keypoints-yolo/scripts/benchmarks/results/combined.json"

DEVICE = 0
IMGSZ = 224
BATCH = 16
PCK_THRESHOLDS = [0.05]
PCK_GROUP_THRESHOLD = 0.05
def _extract_model_name(weights_path: Path) -> str:
    parts = weights_path.parts
    if "weights" in parts:
        idx = parts.index("weights")
        if idx - 1 >= 0: return parts[idx - 1]
    return weights_path.stem

def run_all_benchmarks(
    model_paths: list[str],
    test_sets: dict[str, dict],
    output_json: str,
    device: str | int,
    imgsz: int,
    batch: int,
) -> dict[str, Any]:
    
    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "models": []
    }

    for model_path_str in model_paths:
        model_path = Path(model_path_str).expanduser().resolve()
        model_entry = {
            "model_name": _extract_model_name(model_path),
            "model_path": str(model_path),
            "results_by_domain": {}
        }

        print(f"\n>>> Evaluating Model: {model_entry['model_name']}")

        for domain, cfg in test_sets.items():
            print(f"  Testing Domain: {domain}")
            domain_metrics = {"benchmarks": {}, "errors": []}

            #  mAP 
            try:
                domain_metrics["benchmarks"]["map"] = run_map_benchmark(
                    model_path=str(model_path),
                    data_yaml=cfg["yaml"],
                    split=cfg.get("split", "test"),
                    imgsz=imgsz, batch=batch, device=device
                )
            except Exception as e:
                domain_metrics["errors"].append(f"mAP failed: {e}")

            # PCK 
            if cfg.get("img_dir") and cfg.get("label_dir"):
                try:
                    domain_metrics["benchmarks"]["pck"] = run_pck_benchmark(
                        model_path=str(model_path),
                        img_dir=cfg["img_dir"],
                        label_dir=cfg["label_dir"],
                        threshold_factors=PCK_THRESHOLDS
                    )
                except Exception as e:
                    domain_metrics["errors"].append(f"PCK failed: {e}")

            model_entry["results_by_domain"][domain] = domain_metrics

        report["models"].append(model_entry)

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n All benchmarks complete. Results saved to: {output_json}")
    return report

def main():
    run_all_benchmarks(
        MODEL_PATHS, TEST_SETS, OUTPUT_JSON, DEVICE, IMGSZ, BATCH
    )

if __name__ == "__main__":
    main()