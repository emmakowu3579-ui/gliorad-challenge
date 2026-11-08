"""
Starter Data Visualization & Exploratory Data Analysis (EDA) for TCGA-LGG Challenge.
Visualizes multi-channel brain MRI slices (Pre-contrast, FLAIR, Post-contrast), ground-truth
FLAIR abnormality overlays, slice-level class balance, axial tumor progression, and
radiogenomic molecular subtypes (IDH1/2 & 1p/19q codeletion).

Usage:
    python starter_code/visualize_data.py
    python starter_code/visualize_data.py --manifest data/public/train_manifest.csv --output starter_code/eda_tcga_lgg_overview.png
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Enable UTF-8 console output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def synthesize_brain_mri_slice(has_tumor: bool = True, size: int = 256, seed: int = 42):
    """
    Generate a realistic synthetic 3-channel brain MRI slice (Pre-contrast, FLAIR, Post-contrast)
    and corresponding binary glioma mask when raw TIFF/PNG scans are not yet downloaded locally.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    cx, cy = size // 2, size // 2

    # Elliptical brain parenchyma
    brain_dist = ((xx - cx) / (size * 0.36)) ** 2 + ((yy - cy) / (size * 0.42)) ** 2
    brain_mask = brain_dist <= 1.0
    skull_ring = (brain_dist > 1.0) & (brain_dist <= 1.12)

    img = np.zeros((size, size, 3), dtype=np.float32)
    for c, base_intensity in enumerate([0.42, 0.50, 0.45]):
        channel = np.zeros((size, size), dtype=np.float32)
        channel[brain_mask] = base_intensity + 0.08 * np.cos(brain_dist[brain_mask] * 3.14)
        channel[skull_ring] = 0.22
        channel += rng.normal(0, 0.025, (size, size))
        img[:, :, c] = np.clip(channel, 0.0, 1.0)

    # Ventricles (dark CSF in center)
    vent_left = (((xx - (cx - 14)) / 8.0) ** 2 + ((yy - cy) / 22.0) ** 2) <= 1.0
    vent_right = (((xx - (cx + 14)) / 8.0) ** 2 + ((yy - cy) / 22.0) ** 2) <= 1.0
    img[vent_left | vent_right] *= 0.25

    mask = np.zeros((size, size), dtype=np.uint8)
    if has_tumor:
        tx = cx + rng.integers(-38, 38)
        ty = cy + rng.integers(-35, 35)
        rx = rng.integers(18, 28)
        ry = rng.integers(16, 26)
        tumor_region = (((xx - tx) / rx) ** 2 + ((yy - ty) / ry) ** 2) <= 1.0
        tumor_region = tumor_region & brain_mask
        mask[tumor_region] = 1
        # Hyperintense signal on FLAIR (Channel 1) and Post-contrast (Channel 2)
        img[tumor_region, 0] = np.clip(img[tumor_region, 0] * 0.85, 0.0, 1.0)
        img[tumor_region, 1] = np.clip(img[tumor_region, 1] + 0.38, 0.0, 1.0)
        img[tumor_region, 2] = np.clip(img[tumor_region, 2] + 0.22, 0.0, 1.0)

    return (img * 255).astype(np.uint8), mask


def load_or_synthesize_slice(row: pd.Series, images_dir: Path, masks_dir: Path):
    """Load real MRI image and mask from disk if present, otherwise synthesize realistic slice."""
    img_file = str(row.get("image_file", ""))
    mask_file = str(row.get("mask_file", ""))

    if images_dir and masks_dir and (images_dir / img_file).exists() and (masks_dir / mask_file).exists():
        img = np.array(Image.open(images_dir / img_file).convert("RGB"))
        m = np.array(Image.open(masks_dir / mask_file).convert("L"))
        mask = (m > 127).astype(np.uint8)
        return img, mask

    seed = abs(hash(f"{row.get('patient_id', 'p')}_{row.get('slice_idx', 1)}")) % 10000
    return synthesize_brain_mri_slice(has_tumor=bool(row.get("has_tumor", 1)), seed=seed)


def overlay_mask_on_mri(image: np.ndarray, mask: np.ndarray, color=(0, 255, 220), alpha: float = 0.45) -> np.ndarray:
    """Alpha-blend a binary tumor mask onto an RGB MRI scan."""
    out = image.astype(np.float32).copy()
    fg = mask > 0
    for c in range(3):
        out[fg, c] = (1.0 - alpha) * out[fg, c] + alpha * color[c]
    return np.clip(out, 0, 255).astype(np.uint8)


def create_tcga_lgg_eda_dashboard(
    manifest_csv: Path,
    images_dir: Path,
    masks_dir: Path,
    output_path: Path,
    show_plot: bool = False
):
    if not manifest_csv.exists():
        print(f"❌ Error: Manifest CSV not found at {manifest_csv}")
        sys.exit(1)

    df = pd.read_csv(manifest_csv)
    print(f"📊 Loaded {len(df)} MRI slice entries across {df['patient_id'].nunique()} patients.")

    # Pick a tumor-positive row for top-row imaging inspection
    pos_rows = df[df["has_tumor"] == 1]
    sample_row = pos_rows.iloc[0] if len(pos_rows) > 0 else df.iloc[0]
    img_rgb, mask_bin = load_or_synthesize_slice(sample_row, images_dir, masks_dir)

    fig = plt.figure(figsize=(18, 11), dpi=150)
    fig.suptitle(
        "TCGA-LGG Brain MRI Glioma Segmentation — Exploratory Data Analysis & Modality Overview",
        fontsize=16,
        fontweight="bold",
        y=0.98
    )

    # ==================== TOP-LEFT: FLAIR CHANNEL & RGB COMPOSITE ====================
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(img_rgb[:, :, 1], cmap="bone")
    ax1.set_title(
        f"FLAIR Channel (Channel 1)\nPatient: {sample_row['patient_id']} (Slice #{sample_row['slice_idx']})",
        fontsize=11.5,
        fontweight="bold"
    )
    ax1.axis("off")

    # ==================== TOP-CENTER: MULTI-MODAL RGB + CONTOUR ====================
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(img_rgb)
    if mask_bin.sum() > 0:
        ax2.contour(mask_bin, levels=[0.5], colors=["#00FFCC"], linewidths=2.0)
    ax2.set_title(
        "Multi-Channel MRI Composite\n[R: Pre-Contrast | G: FLAIR | B: Post-Contrast]",
        fontsize=11.5,
        fontweight="bold"
    )
    ax2.axis("off")

    # ==================== TOP-RIGHT: GROUND-TRUTH MASK OVERLAY ====================
    ax3 = fig.add_subplot(2, 3, 3)
    blended = overlay_mask_on_mri(img_rgb, mask_bin, color=(0, 255, 200), alpha=0.48)
    ax3.imshow(blended)
    if mask_bin.sum() > 0:
        ax3.contour(mask_bin, levels=[0.5], colors=["#FF3366"], linewidths=1.8)
    tumor_px = int(mask_bin.sum())
    ax3.set_title(
        f"Ground-Truth Glioma Overlay\nFLAIR Abnormality Area: {tumor_px:,} px²",
        fontsize=11.5,
        fontweight="bold"
    )
    ax3.axis("off")

    # ==================== BOTTOM-LEFT: SLICE POSITIVITY CLASS BALANCE ====================
    ax4 = fig.add_subplot(2, 3, 4)
    tumor_counts = df["has_tumor"].value_counts().reindex([0, 1], fill_value=0)
    labels = ["Normal Brain Slice\n(has_tumor = 0)", "FLAIR Glioma Present\n(has_tumor = 1)"]
    colors = ["#4A90E2", "#E65540"]
    bars = ax4.bar(labels, tumor_counts.values, color=colors, edgecolor="black", width=0.55, alpha=0.88)
    ax4.set_title("Slice-Level Class Balance (2D Axial Slices)", fontsize=12, fontweight="bold")
    ax4.set_ylabel("Number of MRI Slices")
    ax4.grid(axis="y", linestyle="--", alpha=0.4)

    total_slices = max(1, len(df))
    for bar, cnt in zip(bars, tumor_counts.values):
        pct = 100.0 * cnt / total_slices
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + (tumor_counts.max() * 0.02),
            f"{cnt} slices ({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold"
        )
    ax4.set_ylim(0, tumor_counts.max() * 1.18 if tumor_counts.max() > 0 else 10)

    # ==================== BOTTOM-CENTER: AXIAL SLICE TUMOR PROGRESSION ====================
    ax5 = fig.add_subplot(2, 3, 5)
    slice_prob = df.groupby("slice_idx")["has_tumor"].mean() * 100.0
    ax5.plot(slice_prob.index, slice_prob.values, marker="o", linewidth=2.5, color="#8E44AD", label="Tumor Frequency (%)")
    ax5.fill_between(slice_prob.index, slice_prob.values, color="#8E44AD", alpha=0.2)
    ax5.set_title("Spatial Tumor Probability Across Axial Slice Index", fontsize=12, fontweight="bold")
    ax5.set_xlabel("Axial Slice Index (Inferior → Superior)")
    ax5.set_ylabel("Slices Containing Tumor (%)")
    ax5.set_ylim(0, 108)
    ax5.grid(linestyle="--", alpha=0.4)
    ax5.legend(loc="upper right")

    # ==================== BOTTOM-RIGHT: RADIOGENOMIC SUBTYPES ====================
    ax6 = fig.add_subplot(2, 3, 6)
    if "idh_status" in df.columns and "codeletion_1p19q" in df.columns:
        patient_df = df.drop_duplicates(subset=["patient_id"])
        ct = pd.crosstab(patient_df["idh_status"], patient_df["codeletion_1p19q"])
        ct.plot(kind="bar", ax=ax6, color=["#2ECC71", "#F39C12"], edgecolor="black", alpha=0.88)
        ax6.set_title("Patient Radiogenomic Subtypes (IDH vs. 1p/19q)", fontsize=12, fontweight="bold")
        ax6.set_xlabel("IDH1/2 Mutation Status")
        ax6.set_ylabel("Number of Patients")
        ax6.tick_params(axis="x", rotation=0)
        ax6.legend(title="1p/19q Status")
        ax6.grid(axis="y", linestyle="--", alpha=0.4)
    else:
        per_p = df.groupby("patient_id")["has_tumor"].sum()
        ax6.hist(per_p.values, bins=8, color="#2ECC71", edgecolor="black", alpha=0.85)
        ax6.set_title("Tumor-Bearing Slices per Patient", fontsize=12, fontweight="bold")
        ax6.set_xlabel("Positive Slices Count")
        ax6.set_ylabel("Patients")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"✅ Saved TCGA-LGG EDA visualization dashboard to: {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Visualize TCGA-LGG Brain MRI dataset and radiogenomic metadata.")
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(root / "data" / "public" / "train_manifest.csv"),
        help="Path to train_manifest.csv"
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default=str(root / "data" / "mock_lgg" / "images"),
        help="Optional directory with MRI slice images"
    )
    parser.add_argument(
        "--masks-dir",
        type=str,
        default=str(root / "data" / "mock_lgg" / "masks"),
        help="Optional directory with binary mask images"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(root / "starter_code" / "eda_tcga_lgg_overview.png"),
        help="Output path for the generated EDA figure"
    )
    parser.add_argument("--show", action="store_true", help="Display interactive matplotlib window")
    args = parser.parse_args()

    create_tcga_lgg_eda_dashboard(
        manifest_csv=Path(args.manifest),
        images_dir=Path(args.images_dir),
        masks_dir=Path(args.masks_dir),
        output_path=Path(args.output),
        show_plot=args.show
    )


if __name__ == "__main__":
    main()
