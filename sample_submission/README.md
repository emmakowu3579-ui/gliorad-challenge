# Sample Submission (TCGA-LGG Segmentation)

This directory contains instructions and an example predictions folder (`predictions/`) demonstrating the format required for the TCGA-LGG segmentation benchmark.

## Format Guidelines:
1. Provide a folder containing one binary prediction image for each test set slice:
   - File naming: `<patient_id>_<slice_idx>_mask.png` (or `.tif`)
   - Dimensions: $256 \times 256$ pixels
   - Bit depth: 8-bit grayscale
   - Pixel values: `0` (background) or `255` (FLAIR abnormality / tumor)
2. Ensure every slice from all test patients is included.
3. Validate your submission prior to scoring:
   ```bash
   python evaluation/validate_submission.py --predictions sample_submission/predictions/ --ground-truth data/mock_lgg/masks_val/
   ```
