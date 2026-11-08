"""
TCGA-LGG Evaluation Package.
"""

from .metrics import compute_segmentation_metrics
from .validate_submission import validate_submission_masks

__all__ = ["compute_segmentation_metrics", "validate_submission_masks"]
