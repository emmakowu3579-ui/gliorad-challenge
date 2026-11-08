# 🧠 GlioRad: Glioma Radiogenomics Brain MRI Segmentation Challenge

[![Benchmark](https://img.shields.io/badge/Benchmark-GlioRad-teal?style=for-the-badge&logo=brain)](index.html)
[![Metric](https://img.shields.io/badge/Metric-Mean%20Slice%20Dice-blue?style=for-the-badge)](LEADERBOARD.md)
[![Dataset](https://img.shields.io/badge/Dataset-TCGA--LGG%20Radiogenomics-orange?style=for-the-badge)](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)
[![Submissions](https://img.shields.io/badge/Submissions-Encrypted%20PRs-blueviolet?style=for-the-badge)](submissions/)

## Welcome to the GlioRad Challenge 🚀

**GlioRad** (**Glio**ma **Rad**iogenomics) bridges **Clinical Neuro-Oncology**, **Medical Radiogenomics**, and **Computer Vision**. Your task is to build a high-fidelity semantic segmentation model that can delineate hyperintense FLAIR abnormalities (glioma tumor core + surrounding vasogenic edema) across multi-channel brain MRI scans under strict zero-leakage patient stratification.

---

## 🎯 The Task: Zero-Leakage MRI Segmentation

Given multi-sequence brain MRI slices, your model must:
1. **Segment** the full spatial extent of the FLAIR tumor abnormality.
2. **Generalize Inductively** to completely unseen patients (**Strict Patient-Level Splitting**).
3. **Generate** Run-Length Encoded (RLE) binary masks.

### 🔍 Challenge Motivation & Clinical Significance

In lower-grade gliomas, accurate tumor volumetric delineation directly informs surgical resection margins, radiotherapy targets, and molecular genomic prognosis (e.g. IDH mutation status and 1p/19q codeletion). Unlike standard vision datasets, **brain morphology is unique to each individual patient**. Mixing slices from the same patient across train and test causes severe data leakage. Our benchmark enforces strict patient stratification.

---

## 📂 The Dataset (TCGA-LGG Radiogenomics)

The data is derived from the [TCGA-LGG Radiogenomics Collection](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) (Buda et al.).

```text
[ Brain MRI Volume ]  -->  [ Multi-Sequence Slices ]  -->  [ Abnormality Segmentation ]
         🧠                         🖼️                                 🎯
   (110 Patients)           (Pre, FLAIR, Post)               (Binary Tumor Mask)
                                                                      |
                                                            (Dice / DSC Scoring)
```

### 1. MRI Input Sequences

| Channel | MRI Sequence | Tissue Characteristic |
|:---:|---|---|
| **0** | **Pre-Contrast T1** | Baseline anatomical reference; gray brain parenchyma. |
| **1** | **FLAIR** | Fluid-Attenuated Inversion Recovery; hyperintense tumor signal with CSF suppression. |
| **2** | **Post-Contrast T1 / T2** | Contrast-enhancing lesions and blood-brain barrier disruption. |

---

## 2. File Structure (`data/public/`)

| File | Description | Schema / Columns |
|---|---|---|
| `train_manifest.csv` | Training patient slices with genomic metadata | `patient_id, slice_idx, image_file, has_tumor, idh_status, codeletion_1p19q` |
| `test_manifest.csv` | Unseen test patient slices (without ground truth masks) | `patient_id, slice_idx, image_file` |
| `sample_submission.csv` | Example Run-Length Encoded (RLE) submission format | `patient_id, slice_idx, rle_mask` |

---

## 📝 Submission Format

Create a single CSV file named strictly using your team name: `submissions/<team_name>.csv` (e.g. `submissions/emmanuel_owusu.csv`).

```csv
patient_id,slice_idx,rle_mask
TCGA_DU_7008_19860101,1,
TCGA_DU_7008_19860101,2,
TCGA_DU_7008_19860101,3,30820 35 31076 35 31332 35
TCGA_DU_7008_19860101,4,30820 40 31076 40 31332 40
```

### Submission Requirements:
- Header must be exactly: `patient_id,slice_idx,rle_mask`
- Each row corresponds to one test slice.
- For slices with no tumor, `rle_mask` must be empty.
- For slices with a lesion, `rle_mask` contains space-separated pairs of `[start_pixel, length]` (1D flattened index for $256 \times 256$ image).

---

## 🚀 How to Participate

### 1️⃣ Clone the Repository & Install Dependencies
```bash
git clone https://github.com/<your-org>/cv-code-bench.git
cd cv-code-bench/challenge_02_tcga_lgg_segmentation
pip install -r starter_code/requirements.txt
```

### 2️⃣ Run the Baseline Model
A starter baseline segmentation script is provided in `starter_code/`:
```bash
python starter_code/baseline.py
```
This generates your initial submission file at `submissions/baseline_submission.csv`.

### 3️⃣ Encrypt Your Submission
To prevent test set label leakage, you **must encrypt** your submission CSV using the challenge public key before committing:

```bash
# Usage: python starter_code/encrypt.py submissions/<team_name>.csv
python starter_code/encrypt.py submissions/emmanuel_owusu.csv
```
This produces `submissions/emmanuel_owusu.csv.enc`.

### 4️⃣ Submit via GitHub Pull Request
> ⚠️ **IMPORTANT:** Do **NOT** commit raw `.csv` files. Commit **ONLY** the `.enc` file with your team name.

```bash
git checkout -b submission-emmanuel_owusu
git add submissions/emmanuel_owusu.csv.enc
git commit -m "Submit encrypted predictions for emmanuel_owusu"
git push origin submission-emmanuel_owusu
```
Then open a **Pull Request** against `main` on GitHub!

---

## 🤖 Instant Auto-Grading

Once your PR is opened:
- ✅ The **GitHub Actions Auto-Grader Bot** triggers automatically.
- 🔓 Your submission is decrypted securely using the maintainer private key.
- 📊 Official **Mean Slice Dice (DSC)** and **Patient 3D Volumetric Dice** are evaluated.
- 💬 The scorecard is posted as a **comment directly on your PR**:

```text
### 🤖 Automated Grading Report
| Challenge | Metric | Score | Status |
| :--- | :--- | :--- | :--- |
| **TCGA-LGG Segmentation** | **Mean Slice Dice** | **0.8240** | ✅ Valid |
```

When your PR is merged by a maintainer, 🎉 your score appears on the live Leaderboard!

---

## 📏 Rules & Restrictions

| Rule | Detail |
|---|---|
| **Primary Metric** | Mean Slice Dice Similarity Coefficient ($\text{DSC}$) |
| **Inductive Setting** | Test patients are completely held-out. No memorization of patient IDs. |
| **Allowed Models** | Any segmentation architecture (U-Net, Attention U-Net, nnU-Net, SegNet, Swin-UNETR) |
| **Submissions** | One official team submission per PR. |
| **Format** | Strictly encrypted `.csv.enc` via `starter_code/encrypt.py`. |

---

## 🏆 Leaderboard

- **Markdown Leaderboard Table:** [📈 View LEADERBOARD.md](LEADERBOARD.md)
- **Interactive Web Leaderboard:** [🌐 View Interactive TCGA-LGG Leaderboard](index.html)
