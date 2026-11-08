"""
Submission Validator for TCGA-LGG Challenge Auto-Grader.
Enforces schema requirements, patient slice alignment, and one-submission-per-team policy.
Usage: python validate_submission.py <submission_csv> <test_manifest_csv> <leaderboard_csv>
"""

import pandas as pd
import sys
import os
from pathlib import Path

# Enable UTF-8 console output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def main(submission_path: str, test_manifest_path: str, leaderboard_path: str):
    team_name = os.path.splitext(os.path.basename(submission_path))[0]
    print(f"🔍 Validating submission for Team: {team_name}")

    # 1. Team duplicate check
    if os.path.exists(leaderboard_path):
        try:
            lb_df = pd.read_csv(leaderboard_path)
            existing_teams = set(lb_df["team"].astype(str).str.lower())
            if team_name.lower() in existing_teams:
                print(f"❌ REJECTED: Team '{team_name}' has already submitted.")
                print("   Per competition rules, only one official submission is permitted.")
                sys.exit(1)
        except Exception as e:
            print(f"⚠️ Warning: Could not read leaderboard (Skipping check): {e}")

    # 2. Load submission & test manifest
    try:
        preds = pd.read_csv(submission_path)
        test_manifest = pd.read_csv(test_manifest_path)
    except Exception as e:
        print(f"❌ Error: Could not read CSV files: {e}")
        sys.exit(1)

    # 3. Required columns check
    required_cols = {"patient_id", "slice_idx", "rle_mask"}
    missing = required_cols - set(preds.columns)
    if missing:
        print(f"❌ Error: CSV headers must be 'patient_id,slice_idx,rle_mask'. Missing: {missing}")
        sys.exit(1)

    # 4. Check slice matching
    pred_pairs = set(zip(preds["patient_id"].astype(str), preds["slice_idx"].astype(int)))
    test_pairs = set(zip(test_manifest["patient_id"].astype(str), test_manifest["slice_idx"].astype(int)))

    if pred_pairs != test_pairs:
        missing_slices = test_pairs - pred_pairs
        extra_slices = pred_pairs - test_pairs
        print(f"❌ Error: Submission slices do not match test set slices.")
        if missing_slices:
            print(f"   Missing {len(missing_slices)} slices. Examples: {list(missing_slices)[:3]}")
        if extra_slices:
            print(f"   Found {len(extra_slices)} unrecognized slices. Examples: {list(extra_slices)[:3]}")
        sys.exit(1)

    # 5. RLE format validation
    for idx, row in preds.iterrows():
        rle = str(row.get("rle_mask", ""))
        if pd.isna(rle) or rle == "nan" or rle.strip() == "":
            continue
        tokens = rle.strip().split()
        if len(tokens) % 2 != 0:
            print(f"❌ Error: RLE string on slice {row['patient_id']} #{row['slice_idx']} has odd count of integers (expected start length pairs).")
            sys.exit(1)

    print(f"✅ VALID SUBMISSION for Team: {team_name} (Total slices verified: {len(preds)})")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python validate_submission.py <submission_csv> <test_manifest_csv> <leaderboard_csv>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
