from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset

from helpers.feature_engineering import FEATURE_DIMS, build_features
from helpers.training_common import (
    ViewpointMLP,
    evaluate_mse_loss,
    make_kfold_indices,
    make_train_val_from_indices,
    resolve_first_existing_path,
    set_global_seed,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = Path(__file__).resolve().parent / "ablation_feature"

RAW_DATA_CANDIDATES = [
    REPO_ROOT / "zebra_training_dataV250k.npz",
]

FEATURE_TASKS = [
    ("basic", OUTPUT_DIR / "model_basic.pth"),
    ("geo", OUTPUT_DIR / "model_geo.pth"),
    ("full", OUTPUT_DIR / "model_full.pth"),
    ("foreshortening", OUTPUT_DIR / "model_foreshortening.pth"),
    ("stance", OUTPUT_DIR / "model_stance.pth"),
    ("symmetry", OUTPUT_DIR / "model_symmetry.pth"),
    ("torso", OUTPUT_DIR / "model_torso.pth"),
    ("expert", OUTPUT_DIR / "model_expert.pth"),
]

BATCH_SIZE = 1024
MAX_EPOCHS = 200
EARLY_STOP_PATIENCE = 20
LR = 1e-3
SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
K_FOLDS = 8

HIDDEN_DIMS = [256, 256, 256, 256]


def resolve_data_path() -> Path:
    return resolve_first_existing_path(RAW_DATA_CANDIDATES, item_name="dataset")


class ZebraDataset(Dataset):
    def __init__(self, npz_path: Path, feature_type: str = "basic", augment: bool = False):
        data = np.load(npz_path)
        self.kpts = data["kpts"].astype(np.float32)
        self.yaws = data["yaw"].astype(np.float32)
        self.feature_type = feature_type
        self.augment = augment

    def __len__(self):
        return len(self.yaws)

    def __getitem__(self, idx):
        kpts = self.kpts[idx].copy()

        if self.augment:
            noise = np.random.normal(0, 0.05, size=(17, 2))
            kpts[:, :2] += noise

        final_features = build_features(kpts, self.feature_type)

        rad = np.radians(self.yaws[idx])
        y = np.array([np.sin(rad), np.cos(rad)], dtype=np.float32)
        return torch.tensor(final_features, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def reseed_for_run(fold_id: int):
    set_global_seed(SEED + int(fold_id))


def train_one_feature(feature_name, npz_path, ckpt_path, hidden_dims, device, input_dim, fold_id, train_idx, val_idx):
    ckpt_path = ckpt_path.with_name(f"{ckpt_path.stem}_fold{fold_id}{ckpt_path.suffix}")
    reseed_for_run(fold_id)

    train_base = ZebraDataset(npz_path, feature_type=feature_name, augment=True)
    val_base = ZebraDataset(npz_path, feature_type=feature_name, augment=False)

    train_ds = Subset(train_base, train_idx)
    val_ds = Subset(val_base, val_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = ViewpointMLP(input_dim, hidden_dims).to(device)

    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=7)

    best_val_loss = float("inf")
    patience = 0

    model_tag = f"feature_{feature_name}_fold{fold_id}"

    for epoch in range(MAX_EPOCHS):
        print(f"{model_tag} | epoch {epoch + 1}/{MAX_EPOCHS}")
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()

            pred = model(x)
            loss = criterion(pred, y)

            loss.backward()
            optimizer.step()

        val_loss = evaluate_mse_loss(model, val_loader, device)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience += 1
            if patience >= EARLY_STOP_PATIENCE:
                break

    print(f"{model_tag} | finished")


def main():
    set_global_seed(SEED)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script but no CUDA device is available.")
    device = torch.device("cuda")

    data_path = resolve_data_path()
    hidden_dims = HIDDEN_DIMS
    base_ds = ZebraDataset(data_path, feature_type="basic", augment=False)
    folds = make_kfold_indices(len(base_ds), K_FOLDS, seed=SEED)

    for fold_id in range(K_FOLDS):
        trainval_idx = []
        for j in range(K_FOLDS):
            if j != fold_id:
                trainval_idx.extend(folds[j])

        train_idx, val_idx = make_train_val_from_indices(
            trainval_idx,
            val_ratio=VAL_RATIO / max(1e-12, (TRAIN_RATIO + VAL_RATIO)),
        )

        for feature_name, ckpt_path in FEATURE_TASKS:
            input_dim = FEATURE_DIMS[feature_name]
            train_one_feature(
                feature_name,
                data_path,
                ckpt_path,
                hidden_dims,
                device,
                input_dim,
                fold_id,
                train_idx,
                val_idx,
            )


if __name__ == "__main__":
    main()