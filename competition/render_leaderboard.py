"""
Renders LEADERBOARD.md for TCGA-LGG Challenge from leaderboard/leaderboard.csv.
"""

import pandas as pd
import os
import sys

# Enable UTF-8 console output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

LEADERBOARD_CSV = "leaderboard/leaderboard.csv"
LEADERBOARD_MD = "LEADERBOARD.md"


def render():
    if not os.path.exists(LEADERBOARD_CSV):
        print(f"❌ No leaderboard file found at: {LEADERBOARD_CSV}")
        return

    df = pd.read_csv(LEADERBOARD_CSV)
    df = df.sort_values(by="score", ascending=False)
    df["rank"] = df["score"].rank(method="dense", ascending=False).astype(int)

    def format_rank(r):
        if r == 1: return "🥇 1"
        if r == 2: return "🥈 2"
        if r == 3: return "🥉 3"
        return str(r)

    df["rank_display"] = df["rank"].apply(format_rank)

    md_lines = [
        "# 🏆 GlioRad: Lower-Grade Glioma MRI Segmentation Leaderboard",
        "",
        f"**Last Updated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "| Rank | Team | Mean Slice Dice | Date |",
        "| :--- | :--- | :--- | :--- |",
    ]

    for _, row in df.iterrows():
        team = str(row["team"]).replace("|", "")
        md_lines.append(f"| {row['rank_display']} | {team} | {row['score']:.4f} | {row['date']} |")

    with open(LEADERBOARD_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"✅ Leaderboard Markdown updated at {LEADERBOARD_MD}")


if __name__ == "__main__":
    render()
