"""
Official Auto-Grader Evaluation Script for TCGA-LGG Challenge.
Computes Mean Slice Dice score from RLE submissions and outputs SCORE=<score> for GitHub Actions.
Usage: python competition/evaluate.py <submission_csv> <hidden_truth_csv>
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Enable UTF-8 console output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def rle_to_mask(rle_string: str, shape=(256, 256)) -> np.ndarray:
    """Decode Run-Length Encoded string to 2D binary numpy array."""
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    if not isinstance(rle_string, str) or rle_string.strip() in ("", "nan"):
        return mask.reshape(shape)

    tokens = rle_string.strip().split()
    for i in range(0, len(tokens), 2):
        start = int(tokens[i])
        length = int(tokens[i + 1])
        mask[start:start + length] = 1

    return mask.reshape(shape)


def compute_dice(mask1: np.ndarray, mask2: np.ndarray) -> float:
    m1 = mask1 > 0
    m2 = mask2 > 0
    intersection = np.logical_and(m1, m2).sum()
    total = m1.sum() + m2.sum()
    if total == 0:
        return 1.0
    return float(2.0 * intersection / total)


def main():
    if len(sys.argv) < 3:
        print("❌ Error: Missing arguments. Usage: python evaluate.py <submission_csv> <truth_file>")
        sys.exit(1)

    sub_path = sys.argv[1]
    truth_path = sys.argv[2]

    print(f"📂 Grading GlioRad submission: {sub_path}")
    print(f"📂 Against hidden truth: {truth_path}")

    try:
        pred_df = pd.read_csv(sub_path)
        true_df = pd.read_csv(truth_path)

        # Merge on patient_id and slice_idx
        merged = pd.merge(
            true_df,
            pred_df,
            on=["patient_id", "slice_idx"],
            suffixes=("_true", "_pred")
        )

        if len(merged) == 0:
            print("❌ Error: No matching (patient_id, slice_idx) pairs between truth and prediction.")
            sys.exit(1)

        dices = []
        for _, row in merged.iterrows():
            gt_mask = rle_to_mask(str(row.get("rle_mask_true", "")))
            pred_mask = rle_to_mask(str(row.get("rle_mask_pred", "")))
            d = compute_dice(gt_mask, pred_mask)
            dices.append(d)

        mean_dice = float(np.mean(dices))

        # Output score for GitHub Actions regex extraction
        print(f"SCORE={mean_dice:.4f}")
        print(f"EVALUATED_SLICES={len(dices)}")

    except Exception as e:
        print(f"❌ Grading Failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
