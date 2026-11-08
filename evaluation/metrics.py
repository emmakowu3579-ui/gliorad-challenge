"""
Official Evaluation Metrics for TCGA-LGG Brain MRI Glioma Segmentation.
Calculates Dice Similarity Coefficient (DSC), Intersection over Union (IoU),
95th Percentile Hausdorff Distance (HD95), Sensitivity, and Specificity.
Computes both 2D slice-wise and 3D patient volumetric aggregations.
"""

from typing import Dict, List, Tuple, Any
from pathlib import Path
import numpy as np
from scipy.ndimage import distance_transform_edt
from PIL import Image


def compute_dice_score(gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    """
    Compute binary Dice Similarity Coefficient (DSC).
    Both masks must be binary boolean or 0/1 arrays.
    """
    gt = gt_mask > 0
    pred = pred_mask > 0

    intersection = np.logical_and(gt, pred).sum()
    total = gt.sum() + pred.sum()

    if total == 0:
        # Both ground truth and prediction are empty: perfect match
        return 1.0

    return float((2.0 * intersection) / total)


def compute_iou_score(gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    """Compute Intersection over Union (Jaccard Index)."""
    gt = gt_mask > 0
    pred = pred_mask > 0

    intersection = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()

    if union == 0:
        return 1.0

    return float(intersection / union)


def compute_sensitivity_specificity(gt_mask: np.ndarray, pred_mask: np.ndarray) -> Tuple[float, float]:
    """Compute Sensitivity (Recall) and Specificity."""
    gt = gt_mask > 0
    pred = pred_mask > 0

    tp = np.logical_and(gt, pred).sum()
    fn = np.logical_and(gt, ~pred).sum()
    fp = np.logical_and(~gt, pred).sum()
    tn = np.logical_and(~gt, ~pred).sum()

    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 1.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 1.0

    return sensitivity, specificity


def compute_hd95(gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    """
    Compute 95th percentile Hausdorff Distance (HD95) in pixel units.
    Returns 0.0 if both masks are empty; returns 50.0 penalty if one is empty and other is not.
    """
    gt = gt_mask > 0
    pred = pred_mask > 0

    if gt.sum() == 0 and pred.sum() == 0:
        return 0.0
    if gt.sum() == 0 or pred.sum() == 0:
        return 50.0  # Constant penalty for missed lesion or hallucinated lesion on empty slice

    # Distance transform from ground truth boundaries
    dt_gt = distance_transform_edt(~gt)
    dt_pred = distance_transform_edt(~pred)

    # Distances from pred surface to GT and vice versa
    pred_to_gt = dt_gt[pred]
    gt_to_pred = dt_pred[gt]

    hd95_1 = np.percentile(pred_to_gt, 95) if len(pred_to_gt) > 0 else 0.0
    hd95_2 = np.percentile(gt_to_pred, 95) if len(gt_to_pred) > 0 else 0.0

    return float(max(hd95_1, hd95_2))


def compute_segmentation_metrics(
    gt_dir: Path,
    pred_dir: Path
) -> Dict[str, Any]:
    """
    Compute full suite of benchmark metrics across all paired mask files in gt_dir and pred_dir.
    """
    gt_files = sorted(list(gt_dir.glob("*_mask.png")) + list(gt_dir.glob("*_mask.tif")))
    if len(gt_files) == 0:
        raise FileNotFoundError(f"No ground truth masks (*_mask.png or *_mask.tif) found in {gt_dir}")

    slice_dices = []
    slice_ious = []
    slice_hd95s = []
    slice_sens = []
    slice_specs = []

    # Patient-wise aggregation (accumulating 3D voxel volumes)
    patient_gt_voxels: Dict[str, List[np.ndarray]] = {}
    patient_pred_voxels: Dict[str, List[np.ndarray]] = {}

    missing_predictions = 0

    for gt_path in gt_files:
        mask_name = gt_path.name
        # Match corresponding prediction file (allowing .png or .tif)
        pred_path = pred_dir / mask_name
        if not pred_path.exists():
            alt_name = mask_name.replace(".tif", ".png") if mask_name.endswith(".tif") else mask_name.replace(".png", ".tif")
            pred_path = pred_dir / alt_name

        gt_img = np.array(Image.open(gt_path).convert("L")) > 127

        if pred_path.exists():
            pred_img = np.array(Image.open(pred_path).convert("L")) > 127
        else:
            pred_img = np.zeros_like(gt_img)
            missing_predictions += 1

        d = compute_dice_score(gt_img, pred_img)
        iou = compute_iou_score(gt_img, pred_img)
        hd = compute_hd95(gt_img, pred_img)
        sens, spec = compute_sensitivity_specificity(gt_img, pred_img)

        slice_dices.append(d)
        slice_ious.append(iou)
        slice_hd95s.append(hd)
        slice_sens.append(sens)
        slice_specs.append(spec)

        # Extract patient ID from mask name (e.g. TCGA_DU_7000_19860101_02_mask.png)
        parts = mask_name.split("_")
        patient_id = "_".join(parts[:4]) if len(parts) >= 4 else "unknown"

        if patient_id not in patient_gt_voxels:
            patient_gt_voxels[patient_id] = []
            patient_pred_voxels[patient_id] = []

        patient_gt_voxels[patient_id].append(gt_img)
        patient_pred_voxels[patient_id].append(pred_img)

    # Compute 3D Volumetric Patient Dice
    patient_dices = {}
    for p_id in patient_gt_voxels:
        vol_gt = np.stack(patient_gt_voxels[p_id], axis=0)
        vol_pred = np.stack(patient_pred_voxels[p_id], axis=0)
        patient_dices[p_id] = compute_dice_score(vol_gt, vol_pred)

    mean_patient_dice = float(np.mean(list(patient_dices.values())))

    return {
        "primary_rank_metric": "Mean_Slice_Dice",
        "mean_slice_dice": round(float(np.mean(slice_dices)), 4),
        "mean_patient_3d_dice": round(mean_patient_dice, 4),
        "mean_iou": round(float(np.mean(slice_ious)), 4),
        "mean_hd95_pixels": round(float(np.mean(slice_hd95s)), 2),
        "mean_sensitivity": round(float(np.mean(slice_sens)), 4),
        "mean_specificity": round(float(np.mean(slice_specs)), 4),
        "total_evaluated_slices": len(gt_files),
        "total_patients": len(patient_dices),
        "missing_predictions_count": missing_predictions,
        "per_patient_dice": {p: round(d, 4) for p, d in patient_dices.items()}
    }
