import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def resolve_first_existing_path(candidates: list[Path], item_name: str = "path") -> Path:
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No {item_name} found. Tried: {candidates}")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_kfold_indices(n: int, k: int, seed: int) -> list[list[int]]:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    fold_sizes = [n // k] * k
    for i in range(n % k):
        fold_sizes[i] += 1

    folds = []
    start = 0
    for fs in fold_sizes:
        end = start + fs
        folds.append(perm[start:end])
        start = end
    return folds


def make_train_val_from_indices(trainval_idx: list[int], val_ratio: float) -> tuple[list[int], list[int]]:
    n_val = max(1, int(len(trainval_idx) * val_ratio))
    val_idx = trainval_idx[:n_val]
    train_idx = trainval_idx[n_val:]
    return train_idx, val_idx


def make_train_val_indices(n: int, train_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(n, generator=generator).tolist()
    n_train = max(1, int(n * train_ratio))
    n_train = min(n_train, n - 1)
    train_idx = permutation[:n_train]
    val_idx = permutation[n_train:]
    return train_idx, val_idx


def evaluate_mse_loss(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            losses.append(F.mse_loss(pred, y).item())
    return float(np.mean(losses)) if losses else float("inf")


class ViewpointMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout_rate: float = 0.15):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev = h
        layers.append(nn.Linear(prev, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return F.normalize(self.net(x), p=2, dim=1, eps=1e-8)
