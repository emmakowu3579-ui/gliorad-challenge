"""
Unit tests for TCGA-LGG segmentation evaluation metrics.
"""

import importlib.util
from pathlib import Path
import numpy as np
import pytest

eval_dir = Path(__file__).resolve().parent.parent / "evaluation"

spec_m = importlib.util.spec_from_file_location("tcga_metrics_mod", eval_dir / "metrics.py")
tcga_metrics_mod = importlib.util.module_from_spec(spec_m)
spec_m.loader.exec_module(tcga_metrics_mod)

compute_dice_score = tcga_metrics_mod.compute_dice_score
compute_iou_score = tcga_metrics_mod.compute_iou_score
compute_hd95 = tcga_metrics_mod.compute_hd95
compute_sensitivity_specificity = tcga_metrics_mod.compute_sensitivity_specificity


def test_dice_identical():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:50, 20:50] = 1
    assert pytest.approx(compute_dice_score(mask, mask), rel=1e-5) == 1.0
    assert pytest.approx(compute_iou_score(mask, mask), rel=1e-5) == 1.0
    assert compute_hd95(mask, mask) == 0.0


def test_dice_disjoint():
    mask_a = np.zeros((100, 100), dtype=np.uint8)
    mask_b = np.zeros((100, 100), dtype=np.uint8)
    mask_a[10:20, 10:20] = 1
    mask_b[50:60, 50:60] = 1

    assert compute_dice_score(mask_a, mask_b) == 0.0
    assert compute_iou_score(mask_a, mask_b) == 0.0
    assert compute_hd95(mask_a, mask_b) > 10.0


def test_both_empty():
    empty_a = np.zeros((100, 100), dtype=np.uint8)
    empty_b = np.zeros((100, 100), dtype=np.uint8)

    assert compute_dice_score(empty_a, empty_b) == 1.0
    assert compute_iou_score(empty_a, empty_b) == 1.0
    assert compute_hd95(empty_a, empty_b) == 0.0


def test_sensitivity_specificity():
    gt = np.zeros((10, 10), dtype=np.uint8)
    pred = np.zeros((10, 10), dtype=np.uint8)
    gt[0:2, 0:2] = 1    # 4 positive pixels
    pred[0:2, 0:1] = 1  # 2 TP pixels, 2 FN pixels, 0 FP

    sens, spec = compute_sensitivity_specificity(gt, pred)
    assert sens == 0.5
    assert spec == 1.0
