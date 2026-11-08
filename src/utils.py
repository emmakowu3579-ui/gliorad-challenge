"""
Visualization and helper utilities for TCGA-LGG Brain MRI Segmentation.
"""

from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 255),
    alpha: float = 0.45
) -> np.ndarray:
    """
    Overlay a binary segmentation mask onto a 3-channel MRI image.
    Image: shape (H, W, 3) in [0, 255] or [0, 1]
    Mask: shape (H, W) binary (0 or 1)
    """
    if image.max() <= 1.0:
        img_vis = (image * 255).astype(np.uint8).copy()
    else:
        img_vis = image.astype(np.uint8).copy()

    overlay = img_vis.copy()
    mask_bool = mask > 0.5

    for c in range(3):
        overlay[mask_bool, c] = int((1.0 - alpha) * img_vis[mask_bool, c] + alpha * color[c])

    return overlay


def visualize_mri_slice(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    pred_mask: Optional[np.ndarray] = None,
    title: str = "TCGA-LGG Brain MRI Slice",
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[str] = None
):
    """
    Display MRI slice channels and ground truth/predicted masks.
    """
    num_panels = 1
    if mask is not None:
        num_panels += 1
    if pred_mask is not None:
        num_panels += 1

    fig, axes = plt.subplots(1, num_panels, figsize=figsize)
    if num_panels == 1:
        axes = [axes]

    ax_idx = 0
    # 1. Base Image (FLAIR channel or RGB)
    axes[ax_idx].imshow(image)
    axes[ax_idx].set_title("MRI (FLAIR / RGB)")
    axes[ax_idx].axis("off")
    ax_idx += 1

    # 2. Ground Truth Overlay
    if mask is not None:
        gt_overlay = overlay_mask(image, mask, color=(0, 255, 255))
        axes[ax_idx].imshow(gt_overlay)
        axes[ax_idx].set_title("Ground Truth Mask")
        axes[ax_idx].axis("off")
        ax_idx += 1

    # 3. Prediction Overlay
    if pred_mask is not None:
        pred_overlay = overlay_mask(image, pred_mask, color=(255, 60, 60))
        axes[ax_idx].imshow(pred_overlay)
        axes[ax_idx].set_title("Predicted Mask")
        axes[ax_idx].axis("off")

    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()
    else:
        plt.show()
