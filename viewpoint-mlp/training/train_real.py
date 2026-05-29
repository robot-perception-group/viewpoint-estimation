from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from helpers.real_training_helpers import (
    RealCSVViewpointDataset,
    get_sample_weights,
    load_real_records,
    stratified_train_val_indices,
    summarize_split_usage,
)
from helpers.training_common import ViewpointMLP, evaluate_mse_loss, set_global_seed

SCRATCH_MODEL_SPECS = [
    {"name": "funnel_stance_scratch_real", "feature": "stance", "hidden_dims": [256, 128, 64], "input_dim": 56, "checkpoint_name": "real_funnel_stance_scratch.pth"},
    {"name": "heavy_stance_scratch_real", "feature": "stance", "hidden_dims": [512, 256, 128], "input_dim": 56, "checkpoint_name": "real_heavy_stance_scratch.pth"},
    {"name": "superheavy_stance_scratch_real", "feature": "stance", "hidden_dims": [1024, 512, 256], "input_dim": 56, "checkpoint_name": "real_superheavy_stance_scratch.pth"},
    {"name": "deep_stable_stance_scratch_real", "feature": "stance", "hidden_dims": [256, 256, 256, 256], "input_dim": 56, "checkpoint_name": "real_deep_stable_stance_scratch.pth"},
    {"name": "funnel_symmetry_scratch_real", "feature": "symmetry", "hidden_dims": [256, 128, 64], "input_dim": 56, "checkpoint_name": "real_funnel_symmetry_scratch.pth"},
    {"name": "heavy_symmetry_scratch_real", "feature": "symmetry", "hidden_dims": [512, 256, 128], "input_dim": 56, "checkpoint_name": "real_heavy_symmetry_scratch.pth"},
    {"name": "superheavy_symmetry_scratch_real", "feature": "symmetry", "hidden_dims": [1024, 512, 256], "input_dim": 56, "checkpoint_name": "real_superheavy_symmetry_scratch.pth"},
    {"name": "deep_stable_symmetry_scratch_real", "feature": "symmetry", "hidden_dims": [256, 256, 256, 256], "input_dim": 56, "checkpoint_name": "real_deep_stable_symmetry_scratch.pth"},
    {"name": "funnel_expert_scratch_real", "feature": "expert", "hidden_dims": [256, 128, 64], "input_dim": 60, "checkpoint_name": "real_funnel_expert_scratch.pth"},
    {"name": "heavy_expert_scratch_real", "feature": "expert", "hidden_dims": [512, 256, 128], "input_dim": 60, "checkpoint_name": "real_heavy_expert_scratch.pth"},
    {"name": "superheavy_expert_scratch_real", "feature": "expert", "hidden_dims": [1024, 512, 256], "input_dim": 60, "checkpoint_name": "real_superheavy_expert_scratch.pth"},
    {"name": "deep_stable_expert_scratch_real", "feature": "expert", "hidden_dims": [256, 256, 256, 256], "input_dim": 60, "checkpoint_name": "real_deep_stable_expert_scratch.pth"},
]

FINETUNE_MODEL_SPECS = [
    {"name": "funnel_stance_finetune_real", "feature": "stance", "hidden_dims": [256, 128, 64], "input_dim": 56, "pretrained": "full_funnel_stance.pth", "checkpoint_name": "real_funnel_stance_finetune.pth"},
    {"name": "heavy_stance_finetune_real", "feature": "stance", "hidden_dims": [512, 256, 128], "input_dim": 56, "pretrained": "full_heavy_stance.pth", "checkpoint_name": "real_heavy_stance_finetune.pth"},
    {"name": "superheavy_stance_finetune_real", "feature": "stance", "hidden_dims": [1024, 512, 256], "input_dim": 56, "pretrained": "full_superheavy_stance.pth", "checkpoint_name": "real_superheavy_stance_finetune.pth"},
    {"name": "deep_stable_stance_finetune_real", "feature": "stance", "hidden_dims": [256, 256, 256, 256], "input_dim": 56, "pretrained": "full_deep_stable_stance.pth", "checkpoint_name": "real_deep_stable_stance_finetune.pth"},
    {"name": "funnel_symmetry_finetune_real", "feature": "symmetry", "hidden_dims": [256, 128, 64], "input_dim": 56, "pretrained": "full_funnel_symmetry.pth", "checkpoint_name": "real_funnel_symmetry_finetune.pth"},
    {"name": "heavy_symmetry_finetune_real", "feature": "symmetry", "hidden_dims": [512, 256, 128], "input_dim": 56, "pretrained": "full_heavy_symmetry.pth", "checkpoint_name": "real_heavy_symmetry_finetune.pth"},
    {"name": "superheavy_symmetry_finetune_real", "feature": "symmetry", "hidden_dims": [1024, 512, 256], "input_dim": 56, "pretrained": "full_superheavy_symmetry.pth", "checkpoint_name": "real_superheavy_symmetry_finetune.pth"},
    {"name": "deep_stable_symmetry_finetune_real", "feature": "symmetry", "hidden_dims": [256, 256, 256, 256], "input_dim": 56, "pretrained": "full_deep_stable_symmetry.pth", "checkpoint_name": "real_deep_stable_symmetry_finetune.pth"},
    {"name": "funnel_expert_finetune_real", "feature": "expert", "hidden_dims": [256, 128, 64], "input_dim": 60, "pretrained": "full_funnel_expert.pth", "checkpoint_name": "real_funnel_expert_finetune.pth"},
    {"name": "heavy_expert_finetune_real", "feature": "expert", "hidden_dims": [512, 256, 128], "input_dim": 60, "pretrained": "full_heavy_expert.pth", "checkpoint_name": "real_heavy_expert_finetune.pth"},
    {"name": "superheavy_expert_finetune_real", "feature": "expert", "hidden_dims": [1024, 512, 256], "input_dim": 60, "pretrained": "full_superheavy_expert.pth", "checkpoint_name": "real_superheavy_expert_finetune.pth"},
    {"name": "deep_stable_expert_finetune_real", "feature": "expert", "hidden_dims": [256, 256, 256, 256], "input_dim": 60, "pretrained": "full_deep_stable_expert.pth", "checkpoint_name": "real_deep_stable_expert_finetune.pth"},
]

REPO_ROOT = Path(__file__).resolve().parents[3]
LABELS_CSV = Path(".../labels.csv")
FULL_MODEL_DIR = Path(__file__).resolve().parent / "full_models"
OUTPUT_DIR = Path(__file__).resolve().parent / "real_models"

SEED = 42
BATCH_SIZE = 128
MAX_EPOCHS = 250
EARLY_STOP_PATIENCE = 20
LR = 5e-4
VAL_RATIO = 0.15
MAX_SAMPLES_PER_BIN = 5000

LABEL_NOISE_STD_DEG = 3.0
ENABLE_FLIP_AUG = True
FLIP_PROB = 0.25
KEYPOINT_NOISE_STD_PX = 0.5

RUN_SCRATCH = True
RUN_FINETUNE = True



def _load_pretrained_weights(model: ViewpointMLP, pretrained_path: Path, device: torch.device) -> None:
    if not pretrained_path.exists():
        return

    ckpt = torch.load(pretrained_path, map_location=device)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    model_state = model.state_dict()
    filtered = {key: value for key, value in state_dict.items() if key in model_state and value.shape == model_state[key].shape}

    if not filtered:
        print(f"[WARN] No matching pretrained parameters found for {pretrained_path}; skipping load.")
        return

    model_state.update(filtered)
    model.load_state_dict(model_state)


def train_one_model(
    spec: dict,
    records: list[dict],
    train_idx: list[int],
    val_idx: list[int],
    device: torch.device,
    pretrained_path: Path | None,
) -> None:
    set_global_seed(SEED + (hash(spec["name"]) % 10000))

    train_base = RealCSVViewpointDataset(
        records=records,
        feature_type=spec["feature"],
        augment=True,
        label_noise_std_deg=LABEL_NOISE_STD_DEG,
        enable_flip_aug=ENABLE_FLIP_AUG,
        flip_prob=FLIP_PROB,
        keypoint_noise_std_px=KEYPOINT_NOISE_STD_PX,
    )
    val_base = RealCSVViewpointDataset(
        records=records,
        feature_type=spec["feature"],
        augment=False,
        label_noise_std_deg=0.0,
        enable_flip_aug=False,
        flip_prob=0.0,
        keypoint_noise_std_px=0.0,
    )

    train_ds = Subset(train_base, train_idx)
    val_ds = Subset(val_base, val_idx)

    weighted_sampler = WeightedRandomSampler(
        weights=get_sample_weights(records, train_idx),
        num_samples=len(train_idx),
        replacement=True,
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=weighted_sampler, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = ViewpointMLP(input_dim=spec["input_dim"], hidden_dims=spec["hidden_dims"]).to(device)

    if pretrained_path is not None:
        _load_pretrained_weights(model, pretrained_path, device)

    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=6)

    best_val = float("inf")
    patience = 0
    ckpt_path = OUTPUT_DIR / spec["checkpoint_name"]

    for epoch in range(MAX_EPOCHS):
        print(f"{spec['name']} | epoch {epoch + 1}/{MAX_EPOCHS}")
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

        if val_loss < best_val:
            best_val = val_loss
            patience = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience += 1
            if patience >= EARLY_STOP_PATIENCE:
                break


def main() -> None:
    set_global_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    records = load_real_records(LABELS_CSV)
    train_idx, val_idx = stratified_train_val_indices(records, val_ratio=VAL_RATIO, seed=SEED, max_samples_per_bin=MAX_SAMPLES_PER_BIN)

    summarize_split_usage(records, train_idx, split_name="train")
    summarize_split_usage(records, val_idx, split_name="val")

    if RUN_SCRATCH:
        for spec in SCRATCH_MODEL_SPECS:
            train_one_model(spec=spec, records=records, train_idx=train_idx, val_idx=val_idx, device=device, pretrained_path=None)

    if RUN_FINETUNE:
        for spec in FINETUNE_MODEL_SPECS:
            train_one_model(
                spec=spec,
                records=records,
                train_idx=train_idx,
                val_idx=val_idx,
                device=device,
                pretrained_path=FULL_MODEL_DIR / spec["pretrained"],
            )


if __name__ == "__main__":
    main()