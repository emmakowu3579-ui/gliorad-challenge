# 📂 GlioRad: TCGA-LGG Dataset & Public Files

This directory contains the dataset manifests, documentation, and data preparation utilities for the GlioRad Lower-Grade Glioma (LGG) Brain MRI Segmentation Challenge.

---

## 🏥 Official Dataset Source

- **Dataset:** [Brain MRI Segmentation (`mateuszbuda/lgg-mri-segmentation`)](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)
- **Source:** Sourced from **The Cancer Genome Atlas (TCGA)** and **The Cancer Imaging Archive (TCIA)**
- **Citation:** Buda, Saha, & Mazurowski, *"Association of genomic subtypes of lower-grade gliomas with shape features automatically extracted by a deep learning algorithm,"* Computers in Biology and Medicine, 2019.
- **Patients:** **110 real neuro-oncology patients** with available FLAIR MRI sequences and genomic cluster data.
- **Images:** **3,929 paired multi-channel MRI slices** and expert manual FLAIR abnormality masks.

---

## 📄 Public Files (`data/public/`)

These manifest files are distributed to participants for local training and benchmarking:

| File | Description | Schema / Columns |
|---|---|---|
| `train_manifest.csv` | Training slice manifest with genomic profiles | `patient_id, slice_idx, image_file, has_tumor, idh_status, codeletion_1p19q` |
| `test_manifest.csv` | Test slice manifest without ground truth masks | `patient_id, slice_idx, image_file` |
| `sample_submission.csv` | Schema-compliant Run-Length Encoded (RLE) sample submission | `patient_id, slice_idx, rle_mask` |

---

## 🔒 Zero-Leakage Patient-Level Split

Lower-grade glioma MRI volumes exhibit strong spatial correlation across contiguous slices within the same patient.
To ensure valid, clinical generalization:
- **Splits are partitioned strictly at the patient level**.
- Slices belonging to a patient assigned to test appear exclusively in the test set. No slice-level data leakage occurs across splits.

---

## 🔢 Run-Length Encoding (RLE) Format

Binary masks for $256 \times 256$ slices are compressed into RLE strings:
- Pairs of integers: `start length start length ...` (1-based flattened index).
- An empty string `""` indicates no tumor is present on that slice.
- Decoding utility: [`competition/evaluate.py`](../competition/evaluate.py) via `rle_to_mask(rle_string)`.

---

## ⬇️ How to Acquire the Dataset

### Option A: In Kaggle Notebooks (Recommended)
1. Open your notebook on Kaggle.
2. In the right-hand sidebar under **Input**, click **`+ Add Input`**.
3. Search for: `mateuszbuda/lgg-mri-segmentation`
4. Click **`+` (Add)** to mount it directly at `/kaggle/input/lgg-mri-segmentation/kaggle_3m`.

### Option B: Local Download via Kaggle CLI
```bash
kaggle datasets download -d mateuszbuda/lgg-mri-segmentation -p data/kaggle_lgg/ --unzip
```

To run the patient-level preprocessor:
```bash
python data/preprocess_lgg.py --raw-dir data/kaggle_lgg/kaggle_3m --output-dir data/processed_lgg/
```
