"""
PyTorch Dataset for TCGA-LGG Brain MRI Segmentation.
Loads 3-channel brain MRI slices and binary FLAIR abnormality masks.
"""

from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image


class TCGALGGDataset(Dataset):
    """
    Dataset for loading TCGA-LGG MRI slices from a CSV manifest file.
    Images: 3-channel (Pre-contrast, FLAIR, Post-contrast).
    Masks: 1-channel binary mask (1 for FLAIR abnormality, 0 for background).
    """

    def __init__(
        self,
        manifest_csv: str,
        images_dir: Optional[str] = None,
        masks_dir: Optional[str] = None,
        is_test: bool = False
    ):
        super().__init__()
        self.manifest_path = Path(manifest_csv)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        self.df = pd.read_csv(self.manifest_path)
        self.images_dir = Path(images_dir) if images_dir else None
        self.masks_dir = Path(masks_dir) if masks_dir else None
        self.is_test = is_test

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        row = self.df.iloc[idx]

        # Determine image path
        if "image_path" in row and pd.notna(row["image_path"]):
            img_path = Path(row["image_path"])
        elif self.images_dir and "image_file" in row:
            img_path = self.images_dir / row["image_file"]
        else:
            raise KeyError("Neither 'image_path' nor ('images_dir' and 'image_file') specified.")

        if not img_path.exists():
            raise FileNotFoundError(f"MRI Image slice not found: {img_path}")

        # Load image (3 channels)
        img_pil = Image.open(img_path).convert("RGB")
        img_np = np.array(img_pil, dtype=np.float32) / 255.0  # Normalize to [0, 1]
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # (3, H, W)

        # Determine mask path
        mask_tensor = torch.zeros((1, img_tensor.shape[1], img_tensor.shape[2]), dtype=torch.float32)
        if not self.is_test:
            if "mask_path" in row and pd.notna(row["mask_path"]):
                mask_path = Path(row["mask_path"])
            elif self.masks_dir and "mask_file" in row:
                mask_path = self.masks_dir / row["mask_file"]
            else:
                mask_path = None

            if mask_path and mask_path.exists():
                mask_pil = Image.open(mask_path).convert("L")
                mask_np = np.array(mask_pil, dtype=np.float32)
                # Binarize: any nonzero is tumor
                bin_mask = (mask_np > 127.0).astype(np.float32)
                mask_tensor = torch.from_numpy(bin_mask).unsqueeze(0)  # (1, H, W)

        meta = {
            "patient_id": str(row.get("patient_id", "")),
            "slice_idx": int(row.get("slice_idx", idx)),
            "image_file": img_path.name,
            "mask_file": str(row.get("mask_file", f"{img_path.stem}_mask.png"))
        }

        return img_tensor, mask_tensor, meta
