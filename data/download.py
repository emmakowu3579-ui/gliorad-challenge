"""
TCGA-LGG MRI Data Acquisition Utility.
Downloads and verifies the official TCGA-LGG Brain MRI Segmentation dataset from Kaggle.

Dataset: mateuszbuda/lgg-mri-segmentation
Source: The Cancer Genome Atlas (TCGA) & The Cancer Imaging Archive (TCIA)
Curator: Mateusz Buda et al. (Duke University)
URL: https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Enable UTF-8 console output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

KAGGLE_DATASET = "mateuszbuda/lgg-mri-segmentation"
KAGGLE_URL = f"https://www.kaggle.com/datasets/{KAGGLE_DATASET}"


def download_official_dataset(output_dir: Path):
    """Downloads the official TCGA-LGG dataset using the Kaggle CLI."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("🧠 DOWNLOADING OFFICIAL TCGA-LGG BRAIN MRI SEGMENTATION DATASET")
    print(f"📦 Kaggle Source: {KAGGLE_DATASET}")
    print(f"🔗 URL: {KAGGLE_URL}")
    print(f"📂 Target Directory: {output_dir}")
    print("=" * 72)

    try:
        cmd = [
            "kaggle", "datasets", "download",
            "-d", KAGGLE_DATASET,
            "-p", str(output_dir),
            "--unzip"
        ]
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True)
        print("\n✅ Download and extraction complete!")
    except FileNotFoundError:
        print("\n❌ Error: 'kaggle' CLI tool not found on PATH.")
        print("Please install it with: pip install kaggle")
        print("Or download manually from: " + KAGGLE_URL)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error during download: {e}")
        print("Ensure your kaggle.json API token is placed in ~/.kaggle/ or C:\\Users\\<user>\\.kaggle\\")
        sys.exit(1)


def verify_dataset(data_dir: Path):
    """Verifies that the downloaded dataset contains the expected TCGA patient folders."""
    if not data_dir.exists():
        print(f"❌ Error: Directory does not exist: {data_dir}")
        return False

    patient_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("TCGA_")]
    if len(patient_dirs) == 0:
        # Check subfolder kaggle_3m
        sub = data_dir / "kaggle_3m"
        if sub.exists():
            patient_dirs = [d for d in sub.iterdir() if d.is_dir() and d.name.startswith("TCGA_")]

    if len(patient_dirs) > 0:
        print(f"✅ Verified: Found {len(patient_dirs)} real TCGA patient directories.")
        return True
    else:
        print(f"⚠️ Warning: No 'TCGA_*' patient folders found in {data_dir}.")
        return False


def print_instructions():
    print(f"""
========================================================================
🧠 OFFICIAL TCGA-LGG DATASET INSTRUCTIONS
========================================================================
1. Kaggle Dataset:
   {KAGGLE_DATASET}
   URL: {KAGGLE_URL}

2. Contents:
   - 110 real patient studies from The Cancer Genome Atlas (TCGA-LGG).
   - 3,929 multi-channel brain MRI slices (T1, FLAIR, T1-contrast).
   - Expert manual FLAIR abnormality tumor segmentation masks.
   - Clinical and genomic metadata (IDH mutation status, 1p/19q co-deletion).

3. To download locally with Kaggle CLI:
   kaggle datasets download -d {KAGGLE_DATASET} -p data/kaggle_lgg/ --unzip

4. To preprocess and create zero-leakage patient-level manifests:
   python data/preprocess_lgg.py --raw-dir data/kaggle_lgg/kaggle_3m --output-dir data/processed_lgg/

5. On Kaggle Notebooks:
   In the right-hand panel under 'Input', click '+ Add Input', search for:
   '{KAGGLE_DATASET}' and click Add.
========================================================================
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Acquire and verify official TCGA-LGG MRI dataset")
    parser.add_argument("--download", action="store_true", help="Download official dataset via Kaggle CLI")
    parser.add_argument("--verify", action="store_true", help="Verify downloaded dataset in target directory")
    parser.add_argument("--data-dir", type=str, default="data/kaggle_lgg", help="Path to raw dataset directory")
    args = parser.parse_args()

    target_dir = Path(args.data_dir)

    if args.download:
        download_official_dataset(target_dir)
        verify_dataset(target_dir)
    elif args.verify:
        verify_dataset(target_dir)
    else:
        print_instructions()
