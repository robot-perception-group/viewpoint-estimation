# This script retrains the best standalone models on the cropped datasets.
from pathlib import Path

from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parents[4]
CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"

RUNS = [
    {
        "name": "quadruped",
        "model": REPO_ROOT / "runs/pose/uncropped/quadruped-standalone-general/weights/best.pt",
        "data": CONFIG_DIR / "basic.yaml",
    }
    # ...
]


EPOCHS = 300
IMGSZ = 224 # RAPID specific img size (crops)
BATCH = -1
DEVICE = 0
PATIENCE = 30
WORKERS = 16
SEED = 0

# No augmentations for retrainng on croped images
MOSAIC = 0.0
MIXUP = 0.0
CUTMIX = 0.0
COPY_PASTE = 0.0
DEGREES = 0.0
SCALE = 0.0
TRANSLATE = 0.1
SHEAR = 0.0
PERSPECTIVE = 0.0
FLIPUD = 0.0
FLIPLR = 0.5
BGR = 0.0

POSE_GAIN = 25.0
KOBJ_GAIN = 3.0
BOX_GAIN = 0.1


def train_one(run_cfg: dict):
    model = YOLO(str(run_cfg["model"]))
    model.train(
        data=str(run_cfg["data"]),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        seed=SEED,
        pretrained=True,
        mosaic=MOSAIC,
        mixup=MIXUP,
        cutmix=CUTMIX,
        copy_paste=COPY_PASTE,
        degrees=DEGREES,
        translate=TRANSLATE,
        scale=SCALE,
        shear=SHEAR,
        perspective=PERSPECTIVE,
        flipud=FLIPUD,
        fliplr=FLIPLR,
        bgr=BGR,
        pose=POSE_GAIN,
        kobj=KOBJ_GAIN,
        box=BOX_GAIN,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        patience=PATIENCE,
        close_mosaic=10,
        overlap_mask=False,
        project="cropped_models",
        name=run_cfg["name"],
        save=True,
        deterministic=True,
    )


def main():
    for run_cfg in RUNS:
        if not run_cfg["model"].exists():
            raise FileNotFoundError(f"Missing checkpoint: {run_cfg['model']}")
        if not run_cfg["data"].exists():
            raise FileNotFoundError(f"Missing dataset config: {run_cfg['data']}")
        train_one(run_cfg)


if __name__ == "__main__":
    main()