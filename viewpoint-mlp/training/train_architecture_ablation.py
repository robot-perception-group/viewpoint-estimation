from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset

from helpers.training_common import (
    ViewpointMLP,
    evaluate_mse_loss,
    make_kfold_indices,
    make_train_val_from_indices,
    resolve_first_existing_path,
    set_global_seed,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_CANDIDATES = [
    REPO_ROOT / "zebra_training_dataV250k.npz",
]
OUTPUT_DIR = Path(__file__).resolve().parent / "ablation_architecture"

INPUT_DIM = 17 * 3
BATCH_SIZE = 1024
MAX_EPOCHS = 200
EARLY_STOP_PATIENCE = 20
LR = 1e-3
SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
K_FOLDS = 8


ARCHITECTURES = {
    "micro": [128, 64],
    "funnel": [256, 128, 64],
    "heavy": [512, 256, 128],
    "superheavy": [1024, 512, 256],
    "wide_funnel": [1024, 256, 64],
    "ultra_wide_compression": [2048, 256, 32],
    "deep_wide_hybrid": [1024, 512, 128, 64],
    "deep_stable": [256, 256, 256, 256],
    "heavy_plus": [512, 512, 256, 128],
    "lean_deep": [128, 128, 128, 128, 128],
    "bottleneck": [512, 64, 512],
}


def resolve_data_path() -> Path:
    return resolve_first_existing_path(DATA_CANDIDATES, item_name="dataset")


class ZebraDataset(Dataset):
    def __init__(self, npz_path: Path, augment: bool = False):
        data = np.load(npz_path)
        self.kpts = data["kpts"].astype(np.float32)
        self.yaws = data["yaw"].astype(np.float32)
        self.augment = augment

    def __len__(self):
        return len(self.yaws)

    def __getitem__(self, idx):
        kpts = self.kpts[idx].copy()
        if self.augment:
            noise = np.random.normal(0, 0.05, size=(17, 2))
            kpts[:, :2] += noise

        x = kpts.flatten()
        rad = np.radians(self.yaws[idx])
        y = np.array([np.sin(rad), np.cos(rad)], dtype=np.float32)
        return x, y


def train_one_arch(name, hidden_dims, fold_id, train_loader, val_loader, device):
    model = ViewpointMLP(INPUT_DIM, hidden_dims).to(device)

    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=7)

    best_val_loss = float("inf")
    patience = 0

    model_tag = f"arch_{name}_fold{fold_id}"
    ckpt_path = OUTPUT_DIR / f"{model_tag}.pth"

    for epoch in range(MAX_EPOCHS):
        print(f"{model_tag} | epoch {epoch + 1}/{MAX_EPOCHS}")
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()

            pred = model(x)
            print(pred)
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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = resolve_data_path()
    base_ds = ZebraDataset(data_path, augment=False)
    folds = make_kfold_indices(len(base_ds), K_FOLDS, seed=SEED)

    for name, dims in ARCHITECTURES.items():
        for fold_id in range(K_FOLDS):
            trainval_idx = []
            for j in range(K_FOLDS):
                if j != fold_id:
                    trainval_idx.extend(folds[j])

            train_idx, val_idx = make_train_val_from_indices(
                trainval_idx,
                val_ratio=VAL_RATIO / max(1e-12, (TRAIN_RATIO + VAL_RATIO)),
            )

            train_base = ZebraDataset(data_path, augment=True)
            val_base = ZebraDataset(data_path, augment=False)

            train_ds = Subset(train_base, train_idx)
            val_ds = Subset(val_base, val_idx)

            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

            train_one_arch(
                name,
                dims,
                fold_id,
                train_loader,
                val_loader,
                device,
            )


if __name__ == "__main__":
    main()
