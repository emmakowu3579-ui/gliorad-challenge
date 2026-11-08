"""
Starter Baseline for TCGA-LGG Challenge:
Loads public test manifest, generates candidate RLE abnormality masks,
and exports submissions/baseline_submission.csv.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Enable UTF-8 console output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Usage: python starter_code/baseline.py


def main():
    root = Path(__file__).resolve().parent.parent
    public_data_dir = root / "data" / "public"
    test_manifest_file = public_data_dir / "test_manifest.csv"

    if not test_manifest_file.exists():
        print(f"❌ Error: {test_manifest_file} not found. Run data preparation first.")
        sys.exit(1)

    test_df = pd.read_csv(test_manifest_file)
    print(f"Loaded {len(test_df)} test slices from {test_manifest_file}")

    submissions_dir = root / "submissions"
    submissions_dir.mkdir(parents=True, exist_ok=True)

    preds = []
    for _, row in test_df.iterrows():
        slice_idx = int(row["slice_idx"])
        # Centered candidate abnormality for central brain slices
        if slice_idx in [3, 4, 5]:
            # Sample RLE run: row 120-135, col 110-145
            rle_mask = "30820 35 31076 35 31332 35 31588 35"
        else:
            rle_mask = ""

        preds.append({
            "patient_id": row["patient_id"],
            "slice_idx": slice_idx,
            "rle_mask": rle_mask
        })

    out_csv = submissions_dir / "baseline_submission.csv"
    pd.DataFrame(preds).to_csv(out_csv, index=False)
    print(f"✅ Generated baseline submission with {len(preds)} slices at: {out_csv}")
    print("\n👉 Next Step: Encrypt your submission before committing to GitHub:")
    print(f"   python starter_code/encrypt.py {out_csv}")


if __name__ == "__main__":
    main()
