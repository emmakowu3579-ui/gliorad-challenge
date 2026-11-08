# 📂 TCGA-LGG Dataset & Public Files

This directory contains the data preparation scripts, dataset manifests, and public files for the TCGA-LGG Brain MRI Glioma Segmentation Challenge.

## Public Files (`data/public/`)

These files are public and distributed directly to participants for training and local benchmarking:

| File | Description | Schema / Columns |
|---|---|---|
| `train_manifest.csv` | Training slice manifest with genomic profiles | `patient_id, slice_idx, image_file, has_tumor, idh_status, codeletion_1p19q` |
| `test_manifest.csv` | Test slice manifest without ground truth masks | `patient_id, slice_idx, image_file` |
| `sample_submission.csv` | Schema-compliant Run-Length Encoded (RLE) sample submission | `patient_id, slice_idx, rle_mask` |

---

## Zero-Leakage Patient-Level Split

Lower-grade glioma MRI volumes exhibit strong spatial correlation across contiguous slices within the same patient.
To ensure valid real-world generalization:
- **Splits are partitioned strictly at the patient level**.
- Slices belonging to a patient assigned to test appear exclusively in the test set.

---

## Run-Length Encoding (RLE) Format

Binary masks for $256 \times 256$ slices are compressed into RLE strings:
- Pairs of integers: `start length start length ...` (1-based or 0-based flattened index).
- Empty string indicates no tumor on that slice.
- Decoding utility: [`competition/evaluate.py`](../competition/evaluate.py) `rle_to_mask(rle_string)`.

---

## Generating Mock Dataset (for Rapid Dev & CI)

To generate synthetic multi-channel MRI slices and binary masks:
```bash
python data/download.py --generate-mock --output-dir data/mock_lgg
```

## Downloading Official Kaggle TCGA-LGG Dataset

1. Download dataset using Kaggle CLI:
   ```bash
   kaggle datasets download -d mateuszbuda/lgg-mri-segmentation -p data/kaggle_lgg/ --unzip
   ```
2. Run patient-level preprocessor:
   ```bash
   python data/preprocess_lgg.py --raw-dir data/kaggle_lgg/ --output-dir data/processed_lgg/
   ```
