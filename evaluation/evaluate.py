"""
Official Evaluation Script for TCGA-LGG Brain MRI Glioma Segmentation Challenge.
Validates participant submission and computes slice-wise and patient-wise 3D volumetric metrics.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from metrics import compute_segmentation_metrics
from validate_submission import validate_submission_masks


def print_scorecard(results: dict):
    """Prints formatted evaluation scoreboard to standard output."""
    print("\n" + "=" * 78)
    print("      OFFICIAL EVALUATION SCOREBOARD: TCGA-LGG BRAIN MRI GLIOMA       ")
    print("=" * 78)
    print(f" Total Evaluated Patients    : {results['total_patients']}")
    print(f" Total Evaluated MRI Slices  : {results['total_evaluated_slices']}")
    print(f" Missing Prediction Slices   : {results['missing_predictions_count']}")
    print("-" * 78)
    print(f" PRIMARY RANK METRIC (Mean Slice Dice)    : {results['mean_slice_dice']:.4f}")
    print(f" 3D VOLUMETRIC PATIENT DICE               : {results['mean_patient_3d_dice']:.4f}")
    print(f" Mean Intersection over Union (IoU)       : {results['mean_iou']:.4f}")
    print(f" 95th Percentile Hausdorff Dist (HD95)    : {results['mean_hd95_pixels']:.2f} pixels")
    print(f" Clinical Lesion Sensitivity (Recall)     : {results['mean_sensitivity']:.4f}")
    print(f" Healthy Tissue Specificity               : {results['mean_specificity']:.4f}")
    print("-" * 78)
    print(" PER-PATIENT 3D VOLUMETRIC DICE BREAKDOWN:")
    print(f" {'Patient Identifier':<35} | {'3D Volumetric Dice':<20}")
    print("-" * 78)

    for p_id, d in results.get("per_patient_dice", {}).items():
        print(f" {p_id:<35} | {d:<20.4f}")

    print("=" * 78 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Official TCGA-LGG Segmentation Evaluation")
    parser.add_argument("--predictions", type=str, required=True, help="Directory containing participant mask predictions")
    parser.add_argument("--ground-truth", type=str, required=True, help="Directory containing official ground-truth test masks")
    parser.add_argument("--output-json", type=str, default=None, help="Optional output JSON path")
    args = parser.parse_args()

    pred_dir = Path(args.predictions)
    gt_dir = Path(args.ground_truth)

    if not pred_dir.exists():
        print(f"[ERROR] Predictions directory not found: {pred_dir}")
        sys.exit(1)

    if not gt_dir.exists():
        print(f"[ERROR] Ground truth directory not found: {gt_dir}")
        sys.exit(1)

    # 1. Validation check
    is_valid, errors = validate_submission_masks(pred_dir, gt_dir)
    if not is_valid:
        print("[WARNING] Submission validation reported issues:")
        for err in errors[:5]:
            print(f"  - {err}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more.")

    # 2. Compute metrics
    results = compute_segmentation_metrics(gt_dir, pred_dir)

    # 3. Print scoreboard
    print_scorecard(results)

    # 4. Save JSON if requested
    if args.output_json:
        out_p = Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Metrics saved to {out_p}")


if __name__ == "__main__":
    main()
