"""
Exploratory Data Analysis (EDA) & Visualization for GlioRad (TCGA-LGG Challenge).
Visualizes multi-channel brain MRI scans (Pre-contrast, FLAIR, Post-contrast),
expert abnormality segmentation masks, and patient radiogenomic distributions (IDH / 1p19q).

Dataset: mateuszbuda/lgg-mri-segmentation
Source: The Cancer Genome Atlas (TCGA-LGG) & The Cancer Imaging Archive (TCIA)

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


def find_mri_slice_on_disk(row: pd.Series, search_roots):
    """Searches for real patient slice files across common local directories."""
    img_name = str(row.get("image_file", ""))
    mask_name = str(row.get("mask_file", ""))
    patient_id = str(row.get("patient_id", ""))

    for root in search_roots:
        if not root or not root.exists():
            continue

        # Check candidate locations
        candidates = [
            (root / img_name, root / mask_name),
            (root / patient_id / img_name, root / patient_id / mask_name),
            (root / "kaggle_3m" / patient_id / img_name, root / "kaggle_3m" / patient_id / mask_name),
        ]
        for img_p, mask_p in candidates:
            if img_p.exists():
                img = np.array(Image.open(img_p).convert("RGB"))
                mask = None
                if mask_p.exists():
                    m = np.array(Image.open(mask_p).convert("L"))
                    mask = (m > 127).astype(np.uint8)
                else:
                    mask = np.zeros(img.shape[:2], dtype=np.uint8)
                return img, mask

    return None, None


def overlay_mask_on_mri(image: np.ndarray, mask: np.ndarray, color=(0, 255, 220), alpha: float = 0.45) -> np.ndarray:
    """Alpha-blend a binary tumor mask onto an RGB MRI scan."""
    out = image.astype(np.float32).copy()
    fg = mask > 0
    for c in range(3):
        out[fg, c] = (1.0 - alpha) * out[fg, c] + alpha * color[c]
    return np.clip(out, 0, 255).astype(np.uint8)


def create_tcga_lgg_eda_dashboard(
    manifest_csv: Path,
    search_dirs: list,
    output_path: Path,
    show_plot: bool = False
):
    if not manifest_csv.exists():
        print(f"❌ Error: Manifest CSV not found at {manifest_csv}")
        sys.exit(1)

    df = pd.read_csv(manifest_csv)
    print(f"📊 Loaded {len(df)} MRI slice entries across {df['patient_id'].nunique()} patients.")

    # Find sample real slice if downloaded
    sample_row = df[df["has_tumor"] == 1].iloc[0] if (df["has_tumor"] == 1).any() else df.iloc[0]
    img_rgb, mask_bin = find_mri_slice_on_disk(sample_row, search_dirs)

    fig = plt.figure(figsize=(18, 11), dpi=150)
    fig.suptitle(
        "GlioRad: Lower-Grade Glioma (TCGA-LGG) Radiogenomics & Segmentation EDA",
        fontsize=16,
        fontweight="bold",
        y=0.98
    )

    gs = fig.add_gridspec(2, 4, height_ratios=[1.2, 1.0], hspace=0.35, wspace=0.28)

    # ==================== TOP ROW: MRI SCAN CHANNELS & TUMOR ABNORMALITY ====================
    if img_rgb is not None and mask_bin is not None:
        titles = [
            ("Pre-contrast T1", img_rgb[:, :, 0], "gray"),
            ("FLAIR (Tumor Hyperintensity)", img_rgb[:, :, 1], "gray"),
            ("Post-contrast T1", img_rgb[:, :, 2], "gray"),
            ("Abnormality Segmentation Overlay", overlay_mask_on_mri(img_rgb, mask_bin), None),
        ]
        for col_idx, (title, data, cmap) in enumerate(titles):
            ax = fig.add_subplot(gs[0, col_idx])
            if cmap:
                ax.imshow(data, cmap=cmap)
            else:
                ax.imshow(data)
            ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
            ax.axis("off")
    else:
        # Informative notice that scans can be downloaded from Kaggle
        ax_banner = fig.add_subplot(gs[0, :])
        ax_banner.set_facecolor("#F8FAFC")
        ax_banner.text(
            0.5, 0.55,
            "[TCGA-LGG Brain MRI Dataset: mateuszbuda/lgg-mri-segmentation]\n"
            "To view multi-channel MRI overlays locally, download the dataset using Kaggle CLI:\n\n"
            "kaggle datasets download -d mateuszbuda/lgg-mri-segmentation -p data/kaggle_lgg/ --unzip",
            ha="center", va="center", fontsize=12, color="#1E293B",
            bbox=dict(boxstyle="round,pad=1.2", facecolor="#E2E8F0", edgecolor="#CBD5E1")
        )
        ax_banner.axis("off")

    # ==================== BOTTOM ROW: RADIOGENOMIC DISTRIBUTIONS ====================
    ax1 = fig.add_subplot(gs[1, 0])
    tumor_counts = df.groupby("patient_id")["has_tumor"].mean() * 100
    ax1.hist(tumor_counts, bins=12, color="#0EA5E9", edgecolor="white", alpha=0.9)
    ax1.set_title("Patient Tumor Burden\n(% Slices with Abnormality)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("% Positive Slices", fontsize=10)
    ax1.set_ylabel("Patient Count", fontsize=10)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    ax2 = fig.add_subplot(gs[1, 1])
    if "idh_status" in df.columns:
        idh_counts = df.drop_duplicates(subset=["patient_id"])["idh_status"].value_counts()
        colors = ["#10B981" if "Mutant" in str(k) else "#EF4444" for k in idh_counts.index]
        bars = ax2.bar(idh_counts.index.astype(str), idh_counts.values, color=colors, edgecolor="black", alpha=0.85)
        ax2.set_title("IDH Mutation Status\n(Key Prognostic Biomarker)", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Patients", fontsize=10)
        for b in bars:
            ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.1, f"{int(b.get_height())}", ha="center", fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    ax3 = fig.add_subplot(gs[1, 2])
    if "codeletion_1p19q" in df.columns:
        codel_counts = df.drop_duplicates(subset=["patient_id"])["codeletion_1p19q"].value_counts()
        colors = ["#6366F1" if "codel" in str(k).lower() and "non" not in str(k).lower() else "#F59E0B" for k in codel_counts.index]
        bars = ax3.bar(codel_counts.index.astype(str), codel_counts.values, color=colors, edgecolor="black", alpha=0.85)
        ax3.set_title("1p/19q Co-deletion\n(Oligodendroglioma Hallmark)", fontsize=11, fontweight="bold")
        ax3.set_ylabel("Patients", fontsize=10)
        for b in bars:
            ax3.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.1, f"{int(b.get_height())}", ha="center", fontweight="bold")
    ax3.grid(axis="y", linestyle="--", alpha=0.5)

    ax4 = fig.add_subplot(gs[1, 3])
    slice_pos = df.groupby("slice_idx")["has_tumor"].mean() * 100
    ax4.plot(slice_pos.index, slice_pos.values, marker="o", color="#8B5CF6", linewidth=2.5)
    ax4.fill_between(slice_pos.index, slice_pos.values, color="#8B5CF6", alpha=0.25)
    ax4.set_title("Abnormality Occurrence\nby Slice Depth (Z-axis)", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Slice Index", fontsize=10)
    ax4.set_ylabel("% Slices with Tumor", fontsize=10)
    ax4.grid(True, linestyle="--", alpha=0.5)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    print(f"✅ Generated Exploratory Data Analysis Overview: {output_path}")

    if show_plot:
        plt.show()
    plt.close()


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Generate EDA visualization for GlioRad TCGA-LGG Challenge")
    parser.add_argument("--manifest", type=str, default=str(root / "data" / "public" / "train_manifest.csv"), help="Path to train manifest")
    parser.add_argument("--output", type=str, default=str(root / "starter_code" / "eda_tcga_lgg_overview.png"), help="Output path for EDA image")
    parser.add_argument("--show", action="store_true", help="Display plot interactively")
    args = parser.parse_args()

    search_roots = [
        root / "data" / "kaggle_lgg",
        root / "data" / "processed_lgg",
        root / "data" / "kaggle_lgg" / "kaggle_3m",
    ]

    create_tcga_lgg_eda_dashboard(
        manifest_csv=Path(args.manifest),
        search_dirs=search_roots,
        output_path=Path(args.output),
        show_plot=args.show
    )
