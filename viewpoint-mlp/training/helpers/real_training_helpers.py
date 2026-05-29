# This file contains helper functions and classes for training on real-world data with CSV labels
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from feature_engineering import build_features


BIN_LABELS = ["back", "back_right", "right", "front_right", "front", "front_left", "left", "back_left"]
BIN_TO_IDX = {name: i for i, name in enumerate(BIN_LABELS)}
FLIP_IDX = np.array([1, 0, 2, 3, 4, 8, 9, 10, 5, 6, 7, 14, 15, 16, 11, 12, 13], dtype=np.int64)



def bin_center_deg(bin_idx: int) -> float:
    return float((bin_idx * 45.0) % 360.0)


def mirror_angle_lr(angle_deg: float) -> float:
    return float((360.0 - float(angle_deg)) % 360.0)


def parse_keypoints(raw: str) -> np.ndarray | None:
    if raw is None or not raw.strip():
        return None
    try:
        arr = np.asarray(json.loads(raw), dtype=np.float32)
    except Exception:
        return None

    if arr.ndim != 2:
        return None
    if arr.shape == (17, 3):
        return arr
    if arr.shape == (17, 2):
        conf = np.ones((17, 1), dtype=np.float32)
        return np.concatenate([arr, conf], axis=1)
    return None


def parse_bbox(raw: str) -> np.ndarray | None:
    if raw is None or not raw.strip():
        return None
    try:
        arr = np.asarray(json.loads(raw), dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if arr.size != 4:
        return None
    return arr


def canonicalize_keypoints(kpts_xyc: np.ndarray, bbox_xyxy: np.ndarray | None = None) -> np.ndarray:
    out = kpts_xyc.copy().astype(np.float32)

    if bbox_xyxy is not None and np.asarray(bbox_xyxy).size == 4:
        x1, y1, x2, y2 = [float(v) for v in np.asarray(bbox_xyxy).reshape(-1)[:4]]
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        w = max(1e-6, x2 - x1)
        h = max(1e-6, y2 - y1)
    else:
        min_xy = out[:, :2].min(axis=0)
        max_xy = out[:, :2].max(axis=0)
        cx = 0.5 * float(min_xy[0] + max_xy[0])
        cy = 0.5 * float(min_xy[1] + max_xy[1])
        w = max(1e-6, float(max_xy[0] - min_xy[0]))
        h = max(1e-6, float(max_xy[1] - min_xy[1]))

    scale = max(w, h)
    out[:, 0] = (out[:, 0] - cx) / scale
    out[:, 1] = (out[:, 1] - cy) / scale
    return out


def load_real_records(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")

    records = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 4:
                continue

            file_name = row[0].strip()
            bin_label = row[2].strip().lower()
            kpts = parse_keypoints(row[3])
            bbox = parse_bbox(row[4]) if len(row) > 4 else None

            if not file_name or kpts is None or bin_label not in BIN_TO_IDX:
                continue

            bin_idx = BIN_TO_IDX[bin_label]
            records.append(
                {
                    "file": file_name,
                    "bin_idx": int(bin_idx),
                    "angle_center": float(bin_center_deg(bin_idx)),
                    "kpts": kpts,
                    "bbox": bbox,
                }
            )

    if not records:
        raise RuntimeError("No valid labeled rows found in labels.csv")
    return records


def stratified_train_val_indices(records: list[dict], val_ratio: float, seed: int, max_samples_per_bin: int | None = None):
    rng = np.random.default_rng(seed)
    bins_to_indices = {i: [] for i in range(8)}
    for idx, rec in enumerate(records):
        bins_to_indices[rec["bin_idx"]].append(idx)

    if max_samples_per_bin is not None:
        for bin_idx in range(8):
            idxs = bins_to_indices[bin_idx]
            if len(idxs) > max_samples_per_bin:
                idxs = np.array(idxs, dtype=np.int64)
                rng.shuffle(idxs)
                bins_to_indices[bin_idx] = idxs[:max_samples_per_bin].tolist()

    train_idx, val_idx = [], []
    for bin_idx in range(8):
        idxs = bins_to_indices[bin_idx]
        if not idxs:
            continue
        idxs = np.array(idxs, dtype=np.int64)
        rng.shuffle(idxs)

        n_val = int(round(len(idxs) * val_ratio))
        if len(idxs) > 1:
            n_val = max(1, min(n_val, len(idxs) - 1))
        else:
            n_val = 0

        val_idx.extend(idxs[:n_val].tolist())
        train_idx.extend(idxs[n_val:].tolist())

    if len(train_idx) == 0 or len(val_idx) == 0:
        raise RuntimeError("Train/val split failed; check label distribution.")
    return train_idx, val_idx


def infer_dataset_name(file_name: str) -> str:
    name = (file_name or "").strip()
    low = name.lower()
    if low.startswith("train__"):
        return "gzgc_train"
    if low.startswith("test__"):
        return "gzgc_test"
    if "drone" in low:
        return "uav"
    if "_vp_" in low:
        return "gzgc_other"
    return "other"


def summarize_split_usage(records: list[dict], indices: list[int], split_name: str):
    per_bin = {label: 0 for label in BIN_LABELS}
    per_dataset = {}
    per_dataset_bin = {}

    for idx in indices:
        rec = records[idx]
        bin_label = BIN_LABELS[rec["bin_idx"]]
        dataset_name = infer_dataset_name(rec.get("file", ""))

        per_bin[bin_label] += 1
        per_dataset[dataset_name] = per_dataset.get(dataset_name, 0) + 1
        if dataset_name not in per_dataset_bin:
            per_dataset_bin[dataset_name] = {label: 0 for label in BIN_LABELS}
        per_dataset_bin[dataset_name][bin_label] += 1

    print(f"[{split_name}] total: {len(indices)}")
    print(f"[{split_name}] per-bin:")
    for label in BIN_LABELS:
        print(f"  {label}: {per_bin[label]}")

    print(f"[{split_name}] per-dataset:")
    for dataset_name in sorted(per_dataset.keys()):
        print(f"  {dataset_name}: {per_dataset[dataset_name]}")

    print(f"[{split_name}] per-dataset x per-bin:")
    for dataset_name in sorted(per_dataset_bin.keys()):
        print(f"  {dataset_name}:")
        for label in BIN_LABELS:
            count = per_dataset_bin[dataset_name][label]
            if count > 0:
                print(f"    {label}: {count}")


def compute_class_weights(records: list[dict], indices: list[int]) -> torch.Tensor:
    bin_counts = np.zeros(8, dtype=np.float32)
    for idx in indices:
        bin_counts[records[idx]["bin_idx"]] += 1

    weights = 1.0 / (bin_counts + 1e-6)
    weights = weights / weights.sum() * 8.0
    return torch.tensor(weights, dtype=torch.float32)


def get_sample_weights(records: list[dict], indices: list[int]) -> np.ndarray:
    class_weights = compute_class_weights(records, indices)
    sample_weights = np.zeros(len(indices), dtype=np.float32)
    for i, idx in enumerate(indices):
        sample_weights[i] = class_weights[records[idx]["bin_idx"]].item()
    return sample_weights


class RealCSVViewpointDataset(Dataset):
    def __init__(
        self,
        records: list[dict],
        feature_type: str,
        augment: bool,
        label_noise_std_deg: float,
        enable_flip_aug: bool,
        flip_prob: float,
        keypoint_noise_std_px: float,
    ):
        self.records = records
        self.feature_type = feature_type
        self.augment = augment
        self.label_noise_std_deg = label_noise_std_deg
        self.enable_flip_aug = enable_flip_aug
        self.flip_prob = flip_prob
        self.keypoint_noise_std_px = keypoint_noise_std_px

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        kpts = rec["kpts"].copy().astype(np.float32)
        bbox = rec.get("bbox", None)
        angle_deg = float(rec["angle_center"])

        if self.augment:
            if self.enable_flip_aug and (np.random.rand() < self.flip_prob):
                if bbox is not None:
                    x1, _, x2, _ = bbox
                    flip_axis = 0.5 * (float(x1) + float(x2))
                else:
                    min_x, max_x = float(np.min(kpts[:, 0])), float(np.max(kpts[:, 0]))
                    flip_axis = 0.5 * (min_x + max_x)

                kpts[:, 0] = (2.0 * flip_axis) - kpts[:, 0]
                kpts = kpts[FLIP_IDX]
                angle_deg = mirror_angle_lr(angle_deg)

            if self.keypoint_noise_std_px > 0.0:
                kpts[:, :2] += np.random.normal(0.0, self.keypoint_noise_std_px, size=(17, 2)).astype(np.float32)

            if self.label_noise_std_deg > 0.0:
                angle_deg += float(np.random.normal(0.0, self.label_noise_std_deg))

        kpts = canonicalize_keypoints(kpts, bbox_xyxy=bbox)
        features = build_features(kpts, self.feature_type)
        rad = np.radians(angle_deg % 360.0)
        target = np.array([np.sin(rad), np.cos(rad)], dtype=np.float32)

        return torch.tensor(features, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)