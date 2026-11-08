"""
TCGA-LGG MRI Data Acquisition & Mock Generator.
Provides instructions for obtaining official TCGA-LGG Kaggle dataset and generates realistic
synthetic multi-channel brain MRI slices with FLAIR abnormality ground truth masks for CI, testing, and dev.
"""

import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


def generate_synthetic_brain_slice(
    patient_id: str,
    slice_idx: int,
    rng: np.random.Generator,
    has_tumor: bool,
    size: int = 256
):
    """
    Generate a 3-channel brain MRI slice (pre-contrast, FLAIR, post-contrast)
    and a binary segmentation mask for the FLAIR abnormality.
    """
    # Channel 0: Pre-contrast T1 (gray brain parenchyma)
    # Channel 1: FLAIR (dark CSF, bright abnormalities)
    # Channel 2: Post-contrast T1 (contrast-enhancing areas)

    img_arr = np.zeros((size, size, 3), dtype=np.uint8)
    mask_arr = np.zeros((size, size), dtype=np.uint8)

    center_x = size // 2 + rng.integers(-5, 5)
    center_y = size // 2 + rng.integers(-5, 5)
    radius_x = rng.integers(75, 95)
    radius_y = rng.integers(85, 105)

    # 1. Generate brain parenchyma elliptical contour
    y, x = np.ogrid[:size, :size]
    brain_mask = ((x - center_x)**2 / (radius_x**2)) + ((y - center_y)**2 / (radius_y**2)) <= 1.0

    # Add brain tissue intensities
    parenchyma_t1 = rng.integers(70, 95)
    parenchyma_flair = rng.integers(60, 85)
    parenchyma_t2 = rng.integers(75, 100)

    img_arr[brain_mask, 0] = parenchyma_t1
    img_arr[brain_mask, 1] = parenchyma_flair
    img_arr[brain_mask, 2] = parenchyma_t2

    # Add skull outer ring
    skull_outer = ((x - center_x)**2 / ((radius_x + 5)**2)) + ((y - center_y)**2 / ((radius_y + 5)**2)) <= 1.0
    skull_bone = skull_outer & (~brain_mask)
    img_arr[skull_bone, :] = rng.integers(120, 160)

    # 2. Add ventricles (darker in FLAIR and T1)
    ventricle_mask = (
        (((x - center_x + 15)**2 / (10**2)) + ((y - center_y)**2 / (25**2)) <= 1.0) |
        (((x - center_x - 15)**2 / (10**2)) + ((y - center_y)**2 / (25**2)) <= 1.0)
    ) & brain_mask
    img_arr[ventricle_mask, :] = rng.integers(15, 30)

    # 3. Add tumor / FLAIR abnormality if this slice contains lesion
    if has_tumor:
        tumor_cx = center_x + rng.integers(-35, 35)
        tumor_cy = center_y + rng.integers(-35, 35)
        tumor_rx = rng.integers(12, 28)
        tumor_ry = rng.integers(12, 28)

        lesion_mask = (((x - tumor_cx)**2 / (tumor_rx**2)) + ((y - tumor_cy)**2 / (tumor_ry**2)) <= 1.0) & brain_mask

        # FLAIR hyperintensity: bright signal on channel 1
        img_arr[lesion_mask, 1] = np.clip(img_arr[lesion_mask, 1] + rng.integers(110, 150), 0, 255)
        # Contrast enhancement on channel 2
        img_arr[lesion_mask, 2] = np.clip(img_arr[lesion_mask, 2] + rng.integers(60, 110), 0, 255)

        mask_arr[lesion_mask] = 1

    # Add realistic Rician/Gaussian noise
    noise = rng.normal(0, 4, (size, size, 3)).astype(np.int16)
    img_arr = np.clip(img_arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return Image.fromarray(img_arr), Image.fromarray(mask_arr * 255)


def generate_mock_lgg_dataset(
    output_dir: str = "data/mock_lgg",
    num_patients: int = 14,
    slices_per_patient: int = 8,
    seed: int = 42
):
    """
    Generate mock TCGA-LGG brain MRI dataset partitioned strictly by Patient ID.
    """
    out_path = Path(output_dir)
    images_dir = out_path / "images"
    masks_dir = out_path / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    patient_records = []
    slice_records = []

    # Assign patients to splits: 8 train (57%), 3 val (21%), 3 test (21%)
    patients = [f"TCGA_DU_{7000 + i:04d}_19860101" for i in range(num_patients)]
    split_assignments = {}
    for i, p in enumerate(patients):
        if i < 8:
            split_assignments[p] = "train"
        elif i < 11:
            split_assignments[p] = "val"
        else:
            split_assignments[p] = "test"

    for p in patients:
        split = split_assignments[p]
        idh_status = rng.choice(["Mutant", "Wildtype"], p=[0.75, 0.25])
        codeletion = rng.choice(["codel", "non-codel"], p=[0.4, 0.6])

        # Tumor typically appears in a contiguous subset of central slices (e.g. slices 3 to 6)
        tumor_start = rng.integers(1, slices_per_patient // 2)
        tumor_end = tumor_start + rng.integers(2, 4)

        for s_idx in range(1, slices_per_patient + 1):
            has_tumor = (tumor_start <= s_idx <= tumor_end)
            slice_name = f"{p}_{s_idx:02d}"
            img_filename = f"{slice_name}.png"
            mask_filename = f"{slice_name}_mask.png"

            img_pil, mask_pil = generate_synthetic_brain_slice(p, s_idx, rng, has_tumor)
            img_pil.save(images_dir / img_filename)
            mask_pil.save(masks_dir / mask_filename)

            slice_records.append({
                "patient_id": p,
                "slice_idx": s_idx,
                "image_file": img_filename,
                "mask_file": mask_filename,
                "has_tumor": int(has_tumor),
                "split": split,
                "idh_status": idh_status,
                "codeletion_1p19q": codeletion
            })

    df = pd.DataFrame(slice_records)

    # Save manifest per split
    for split in ["train", "val", "test"]:
        split_df = df[df["split"] == split].copy()
        manifest_path = out_path / f"{split}_manifest.csv"
        split_df.to_csv(manifest_path, index=False)
        print(f"Generated {split} manifest: {len(split_df['patient_id'].unique())} patients, {len(split_df)} slices -> {manifest_path}")

    # Also create validation masks folder for direct ground truth comparison
    val_masks_dir = out_path / "masks_val"
    val_masks_dir.mkdir(parents=True, exist_ok=True)
    val_df = df[df["split"] == "val"]
    for _, row in val_df.iterrows():
        mask_file = row["mask_file"]
        src = masks_dir / mask_file
        dst = val_masks_dir / mask_file
        if src.exists():
            dst.write_bytes(src.read_bytes())

    print(f"Full mock LGG dataset initialized at {output_dir}")


def print_official_instructions():
    print("""
========================================================================
OFFICIAL TCGA-LGG DATASET DOWNLOAD INSTRUCTIONS:
========================================================================
1. Kaggle Dataset:
   mateuszbuda/lgg-mri-segmentation
   URL: https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation

2. Download using Kaggle CLI:
   kaggle datasets download -d mateuszbuda/lgg-mri-segmentation -p data/kaggle_lgg/ --unzip

3. Directory structure after unzipping:
   data/kaggle_lgg/
   ├── TCGA_CS_4941_19960909/
   │   ├── TCGA_CS_4941_19960909_1.tif
   │   ├── TCGA_CS_4941_19960909_1_mask.tif
   │   └── ...
   ├── data.csv
   └── ...

4. Run the patient-level preprocessing script:
   python data/preprocess_lgg.py --raw-dir data/kaggle_lgg/ --output-dir data/processed_lgg/
========================================================================
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download or generate mock TCGA-LGG MRI data")
    parser.add_argument("--generate-mock", action="store_true", help="Generate mock brain MRI slices and masks")
    parser.add_argument("--download-official", action="store_true", help="Print official download instructions")
    parser.add_argument("--output-dir", type=str, default="data/mock_lgg", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if args.download_official:
        print_official_instructions()
    else:
        generate_mock_lgg_dataset(output_dir=args.output_dir, seed=args.seed)
