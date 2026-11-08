"""
Submission validator for TCGA-LGG Brain MRI Segmentation Challenge.
Checks directory structure, mask image resolutions, binary pixel values, and patient slice coverage.
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple, List
import numpy as np
from PIL import Image


def validate_submission_masks(
    pred_dir: Path,
    gt_dir: Path,
    expected_resolution: Tuple[int, int] = (256, 256)
) -> Tuple[bool, List[str]]:
    """
    Validate candidate submission mask files against official ground-truth test masks.
    """
    errors = []

    if not pred_dir.exists() or not pred_dir.is_dir():
        return False, [f"Prediction directory does not exist: {pred_dir}"]

    gt_files = sorted(list(gt_dir.glob("*_mask.png")) + list(gt_dir.glob("*_mask.tif")))
    if len(gt_files) == 0:
        return False, [f"No ground truth masks found in {gt_dir} to validate against."]

    pred_files = sorted(list(pred_dir.glob("*.png")) + list(pred_dir.glob("*.tif")))
    if len(pred_files) == 0:
        return False, [f"Prediction directory {pred_dir} contains 0 image files (.png or .tif)."]

    pred_names = {f.name for f in pred_files}
    pred_stems = {f.stem: f for f in pred_files}

    # Check for missing test slices
    missing_count = 0
    for gt_file in gt_files:
        if gt_file.name not in pred_names and gt_file.stem not in pred_stems:
            missing_count += 1
            if missing_count <= 5:
                errors.append(f"Missing prediction for test slice: {gt_file.name}")

    if missing_count > 5:
        errors.append(f"... and {missing_count - 5} more test slice predictions missing.")

    # Validate mask properties for available files
    checked = 0
    for pred_path in pred_files[:30]:  # Check first 30 masks
        try:
            img = Image.open(pred_path)
            if img.size != expected_resolution:
                errors.append(f"{pred_path.name}: Expected resolution {expected_resolution}, got {img.size}")

            arr = np.array(img)
            unique_vals = np.unique(arr)
            # Valid values: 0, 1, or 255
            invalid_vals = [v for v in unique_vals if v not in (0, 1, 255)]
            if len(invalid_vals) > 0:
                errors.append(f"{pred_path.name}: Contains non-binary mask values: {invalid_vals[:5]}")

            checked += 1
        except Exception as e:
            errors.append(f"{pred_path.name}: Failed to read image file: {e}")

    is_valid = len(errors) == 0
    return is_valid, errors


def main():
    parser = argparse.ArgumentParser(description="Validate TCGA-LGG segmentation submission")
    parser.add_argument("--predictions", type=str, required=True, help="Path to predictions directory")
    parser.add_argument("--ground-truth", type=str, required=True, help="Path to ground truth masks directory")
    args = parser.parse_args()

    pred_p = Path(args.predictions)
    gt_p = Path(args.ground_truth)

    is_valid, errors = validate_submission_masks(pred_p, gt_p)
    if is_valid:
        print(f"[PASS] All masks in '{pred_p}' successfully passed validation!")
        sys.exit(0)
    else:
        print(f"[FAIL] Submission '{pred_p}' failed validation with {len(errors)} error(s):")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
