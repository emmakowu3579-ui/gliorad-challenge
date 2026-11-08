"""
Unit tests for TCGA-LGG Dataset loader and U-Net architecture.
"""

import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

src_dir = Path(__file__).resolve().parent.parent / "src"

spec_d = importlib.util.spec_from_file_location("tcga_dataset_mod", src_dir / "dataset.py")
tcga_ds_mod = importlib.util.module_from_spec(spec_d)
spec_d.loader.exec_module(tcga_ds_mod)
TCGALGGDataset = tcga_ds_mod.TCGALGGDataset

spec_m = importlib.util.spec_from_file_location("tcga_model_mod", src_dir / "model.py")
tcga_model_mod = importlib.util.module_from_spec(spec_m)
spec_m.loader.exec_module(tcga_model_mod)
UNet = tcga_model_mod.UNet


@pytest.fixture
def mock_mri_dir(tmp_path):
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()

    # Create dummy 3-channel slice & mask
    img_arr = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    mask_arr = np.zeros((256, 256), dtype=np.uint8)
    mask_arr[100:130, 100:130] = 255

    Image.fromarray(img_arr).save(images_dir / "test_slice.png")
    Image.fromarray(mask_arr).save(masks_dir / "test_slice_mask.png")

    manifest_df = pd.DataFrame([{
        "patient_id": "TCGA_TEST_001",
        "slice_idx": 1,
        "image_file": "test_slice.png",
        "mask_file": "test_slice_mask.png"
    }])
    manifest_path = tmp_path / "manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    return manifest_path, images_dir, masks_dir


def test_mri_dataset_loading(mock_mri_dir):
    manifest_path, images_dir, masks_dir = mock_mri_dir
    ds = TCGALGGDataset(manifest_csv=str(manifest_path), images_dir=str(images_dir), masks_dir=str(masks_dir))

    assert len(ds) == 1
    img_tensor, mask_tensor, meta = ds[0]

    assert isinstance(img_tensor, torch.Tensor)
    assert img_tensor.shape == (3, 256, 256)
    assert mask_tensor.shape == (1, 256, 256)
    assert meta["patient_id"] == "TCGA_TEST_001"
    assert mask_tensor.max().item() == 1.0


def test_unet_forward_pass():
    model = UNet(in_channels=3, num_classes=1, init_features=16)
    dummy_input = torch.randn(2, 3, 256, 256)
    out = model(dummy_input)
    assert out.shape == (2, 1, 256, 256)
