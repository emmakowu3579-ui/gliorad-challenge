"""
Updates leaderboard/leaderboard.csv from all submissions for TCGA-LGG Challenge.
"""

import pandas as pd
import os
import glob
import subprocess
from pathlib import Path
import numpy as np

SUBMISSION_DIR = "submissions"
TRUTH_FILE = "data/test_labels_hidden.csv"
LEADERBOARD_CSV = "leaderboard/leaderboard.csv"


def rle_to_mask(rle_string: str, shape=(256, 256)) -> np.ndarray:
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    if not isinstance(rle_string, str) or rle_string.strip() in ("", "nan"):
        return mask.reshape(shape)

    tokens = rle_string.strip().split()
    for i in range(0, len(tokens), 2):
        start = int(tokens[i])
        length = int(tokens[i + 1])
        mask[start:start + length] = 1

    return mask.reshape(shape)


def compute_dice(mask1: np.ndarray, mask2: np.ndarray) -> float:
    m1 = mask1 > 0
    m2 = mask2 > 0
    intersection = np.logical_and(m1, m2).sum()
    total = m1.sum() + m2.sum()
    if total == 0:
        return 1.0
    return float(2.0 * intersection / total)


def main():
    if not os.path.exists(TRUTH_FILE):
        print(f"❌ Error: Hidden truth file {TRUTH_FILE} not found.")
        return

    true_df = pd.read_csv(TRUTH_FILE)
    files = glob.glob(f"{SUBMISSION_DIR}/**/*.csv", recursive=True)
    results = []

    print(f"🔍 Found {len(files)} submission CSV files.")

    for file_path in files:
        if "sample_submission.csv" in file_path:
            continue
        try:
            pred_df = pd.read_csv(file_path)
            merged = pd.merge(
                true_df,
                pred_df,
                on=["patient_id", "slice_idx"],
                suffixes=("_true", "_pred")
            )

            if len(merged) == 0:
                continue

            dices = []
            for _, row in merged.iterrows():
                gt_mask = rle_to_mask(str(row.get("rle_mask_true", "")))
                pred_mask = rle_to_mask(str(row.get("rle_mask_pred", "")))
                dices.append(compute_dice(gt_mask, pred_mask))

            score = float(np.mean(dices))
            filename = os.path.basename(file_path)
            team_name = os.path.splitext(filename)[0]

            enc_file = file_path + ".enc"
            commit_date = ""
            if os.path.exists(enc_file):
                try:
                    cmd = f'git log -1 --format="%cd" --date=format:"%Y-%m-%d %H:%M" -- "{enc_file}"'
                    commit_date = subprocess.check_output(cmd, shell=True).decode("utf-8").strip()
                except Exception:
                    pass

            if not commit_date:
                commit_date = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")

            print(f"✅ Scored {team_name}: Dice={score:.4f}")
            results.append({
                "team": team_name,
                "score": round(score, 4),
                "date": commit_date
            })

        except Exception as e:
            print(f"⚠️ Failed to process {file_path}: {e}")

    if not results:
        print("No valid submissions found.")
        return

    df = pd.DataFrame(results)
    df = df.sort_values(by="score", ascending=False).drop_duplicates(subset=["team"], keep="first")

    os.makedirs("leaderboard", exist_ok=True)
    df.to_csv(LEADERBOARD_CSV, index=False)
    print(f"✅ Leaderboard CSV saved to {LEADERBOARD_CSV}")


if __name__ == "__main__":
    main()
