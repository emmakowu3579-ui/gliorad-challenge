"""
Inference script for TCGA-LGG Brain MRI Segmentation.
Generates binary segmentation mask images for test patient slices.
"""

import argparse
import os
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

from dataset import TCGALGGDataset
from model import UNet


@torch.no_grad()
def run_inference(model, dataloader, device, output_dir: Path, threshold: float = 0.5):
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for images, _, metas in tqdm(dataloader, desc="Predicting masks"):
        images = images.to(device)
        logits = model(images)
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).cpu().numpy().squeeze(1)  # (B, H, W)

        batch_size = images.shape[0]
        for i in range(batch_size):
            pred_mask = (preds[i] * 255).astype(np.uint8)
            mask_filename = metas["mask_file"][i]
            patient_id = metas["patient_id"][i]

            out_file = output_dir / mask_filename
            Image.fromarray(pred_mask).save(out_file)

            tumor_pixels = int(np.sum(preds[i] > 0))
            records.append({
                "patient_id": patient_id,
                "mask_file": mask_filename,
                "predicted_tumor_pixels": tumor_pixels,
                "has_predicted_tumor": int(tumor_pixels > 0)
            })

    summary_df = pd.DataFrame(records)
    summary_path = output_dir.parent / "predicted_masks_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Generated {len(records)} prediction masks saved to {output_dir}")
    print(f"Summary manifest saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate TCGA-LGG Segmentation Predictions")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to checkpoint")
    parser.add_argument("--config", type=str, default="configs/baseline_config.yaml", help="Path to config file")
    parser.add_argument("--test-manifest", type=str, default=None, help="Override test manifest path")
    parser.add_argument("--output-dir", type=str, default="sample_submission/predictions", help="Output directory for predicted masks")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for binarization")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        fallback_cfg = Path(__file__).resolve().parent.parent / "configs" / "baseline_config.yaml"
        if fallback_cfg.exists():
            cfg_path = fallback_cfg

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    challenge_root = cfg_path.parent.parent
    test_manifest = args.test_manifest or cfg["data"].get("test_manifest", "data/mock_lgg/test_manifest.csv")
    if not Path(test_manifest).exists() and (challenge_root / test_manifest).exists():
        test_manifest = str(challenge_root / test_manifest)

    images_dir = cfg["data"].get("images_dir")
    if images_dir and not Path(images_dir).exists() and (challenge_root / images_dir).exists():
        images_dir = str(challenge_root / images_dir)

    masks_dir = cfg["data"].get("masks_dir")
    if masks_dir and not Path(masks_dir).exists() and (challenge_root / masks_dir).exists():
        masks_dir = str(challenge_root / masks_dir)

    test_dataset = TCGALGGDataset(
        manifest_csv=test_manifest,
        images_dir=images_dir,
        masks_dir=masks_dir,
        is_test=True
    )
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    model = UNet(
        in_channels=cfg["data"].get("in_channels", 3),
        num_classes=cfg["data"].get("num_classes", 1),
        init_features=cfg["model"].get("init_features", 32),
        use_batchnorm=cfg["model"].get("use_batchnorm", True)
    )

    ckpt_path = Path(args.checkpoint)
    if ckpt_path.exists():
        print(f"Loading checkpoint weights from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        print(f"Notice: Checkpoint {ckpt_path} not found. Running with initialized weights.")

    model.to(device)

    run_inference(model, test_loader, device, Path(args.output_dir), threshold=args.threshold)


if __name__ == "__main__":
    main()
