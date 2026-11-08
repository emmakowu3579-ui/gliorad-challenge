"""
Training pipeline for TCGA-LGG Brain MRI U-Net baseline.
Monitors validation Dice score and saves best model checkpoint.
"""

import argparse
import os
from pathlib import Path
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import TCGALGGDataset
from model import UNet
from losses import DiceBCELoss


def compute_batch_dice(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """Compute mean Dice score across batch."""
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds = preds.contiguous().view(preds.shape[0], -1)
    targets = targets.contiguous().view(targets.shape[0], -1)

    intersection = torch.sum(preds * targets, dim=1)
    cardinality = torch.sum(preds, dim=1) + torch.sum(targets, dim=1)

    dice = (2.0 * intersection + 1e-5) / (cardinality + 1e-5)
    return float(torch.mean(dice).item())


def train_one_epoch(model, optimizer, criterion, dataloader, device, epoch, log_interval=5):
    model.train()
    total_loss = 0.0
    total_dice = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for step, (images, masks, _) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        dice = compute_batch_dice(logits, masks)
        total_loss += loss.item()
        total_dice += dice

        if (step + 1) % log_interval == 0:
            avg_loss = total_loss / (step + 1)
            avg_dice = total_dice / (step + 1)
            pbar.set_postfix({"loss": f"{avg_loss:.4f}", "dice": f"{avg_dice:.4f}"})

    return total_loss / max(1, len(dataloader)), total_dice / max(1, len(dataloader))


@torch.no_grad()
def evaluate(model, criterion, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0

    for images, masks, _ in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = criterion(logits, masks)
        dice = compute_batch_dice(logits, masks)

        total_loss += loss.item()
        total_dice += dice

    return total_loss / max(1, len(dataloader)), total_dice / max(1, len(dataloader))


def main():
    parser = argparse.ArgumentParser(description="Train U-Net on TCGA-LGG MRI")
    parser.add_argument("--config", type=str, default="configs/baseline_config.yaml", help="Path to config file")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        fallback_cfg = Path(__file__).resolve().parent.parent / "configs" / "baseline_config.yaml"
        if fallback_cfg.exists():
            cfg_path = fallback_cfg

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device_str = cfg["training"].get("device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(device_str)

    print(f"Using device: {device}")

    # Datasets
    train_dataset = TCGALGGDataset(
        manifest_csv=cfg["data"]["train_manifest"],
        images_dir=cfg["data"].get("images_dir"),
        masks_dir=cfg["data"].get("masks_dir")
    )
    val_dataset = TCGALGGDataset(
        manifest_csv=cfg["data"]["val_manifest"],
        images_dir=cfg["data"].get("images_dir"),
        masks_dir=cfg["data"].get("masks_dir")
    )

    batch_size = cfg["training"].get("batch_size", 4)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train slices: {len(train_dataset)}, Validation slices: {len(val_dataset)}")

    # Model
    model = UNet(
        in_channels=cfg["data"].get("in_channels", 3),
        num_classes=cfg["data"].get("num_classes", 1),
        init_features=cfg["model"].get("init_features", 32),
        use_batchnorm=cfg["model"].get("use_batchnorm", True),
        dropout_prob=cfg["model"].get("dropout_prob", 0.2)
    )
    model.to(device)

    criterion = DiceBCELoss(
        dice_weight=cfg["training"].get("dice_weight", 0.6),
        bce_weight=cfg["training"].get("bce_weight", 0.4)
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"].get("lr", 0.0005),
        weight_decay=cfg["training"].get("weight_decay", 0.0001)
    )

    checkpoint_dir = Path(cfg["training"].get("checkpoint_dir", "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    epochs = cfg["training"].get("epochs", 5)
    best_dice = 0.0

    for epoch in range(1, epochs + 1):
        train_loss, train_dice = train_one_epoch(
            model, optimizer, criterion, train_loader, device, epoch,
            log_interval=cfg["training"].get("log_interval", 5)
        )
        val_loss, val_dice = evaluate(model, criterion, val_loader, device)

        print(f"Epoch {epoch}/{epochs} -> Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f} | Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")

        # Checkpointing
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_dice": val_dice,
            "config": cfg
        }, checkpoint_dir / "latest_model.pt")

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_dice": val_dice,
                "config": cfg
            }, checkpoint_dir / "best_model.pt")
            print(f"[*] New best model saved with Val Dice: {best_dice:.4f}")

    print(f"Training finished! Best Validation Dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()
