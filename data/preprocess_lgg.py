"""
Patient-level preprocessor for official Kaggle TCGA-LGG MRI Segmentation dataset.
Guarantees strict zero-leakage patient-level stratification.
"""

import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def preprocess_kaggle_lgg(raw_dir: str, output_dir: str, seed: int = 42):
    raw_path = Path(raw_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Discover all patient folders
    patient_dirs = [p for p in raw_path.iterdir() if p.is_dir() and p.name.startswith("TCGA_")]
    if len(patient_dirs) == 0:
        raise FileNotFoundError(f"No patient folders starting with 'TCGA_' found in {raw_path}")

    print(f"Found {len(patient_dirs)} patient directories.")

    # 2. Collect all slices and masks
    records = []
    for p_dir in patient_dirs:
        patient_id = p_dir.name
        # Slices without '_mask'
        all_tifs = sorted(list(p_dir.glob("*.tif")))
        mask_files = {m.name: m for m in all_tifs if "_mask" in m.name}
        image_files = [img for img in all_tifs if "_mask" not in img.name]

        for img_path in image_files:
            expected_mask_name = f"{img_path.stem}_mask.tif"
            mask_path = mask_files.get(expected_mask_name)

            records.append({
                "patient_id": patient_id,
                "slice_id": img_path.stem,
                "image_path": str(img_path.resolve()),
                "mask_path": str(mask_path.resolve()) if mask_path else None
            })

    df = pd.DataFrame(records)
    print(f"Total discovered slice pairs: {len(df)}")

    # 3. Patient-level split
    unique_patients = df["patient_id"].unique()
    train_patients, temp_patients = train_test_split(
        unique_patients,
        test_size=0.30,
        random_state=seed
    )
    val_patients, test_patients = train_test_split(
        temp_patients,
        test_size=0.50,
        random_state=seed
    )

    split_map = {}
    for p in train_patients:
        split_map[p] = "train"
    for p in val_patients:
        split_map[p] = "val"
    for p in test_patients:
        split_map[p] = "test"

    df["split"] = df["patient_id"].map(split_map)

    for split in ["train", "val", "test"]:
        sub_df = df[df["split"] == split].copy()
        manifest_file = out_path / f"{split}_manifest.csv"
        sub_df.to_csv(manifest_file, index=False)
        print(f"Wrote {split} manifest: {len(sub_df['patient_id'].unique())} patients, {len(sub_df)} slices -> {manifest_file}")

    print("Preprocessing completed with strict patient-level separation!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Kaggle LGG MRI dataset")
    parser.add_argument("--raw-dir", type=str, default="data/kaggle_lgg", help="Path to unzipped Kaggle dataset")
    parser.add_argument("--output-dir", type=str, default="data/processed_lgg", help="Path for output manifests")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    preprocess_kaggle_lgg(args.raw_dir, args.output_dir, args.seed)
