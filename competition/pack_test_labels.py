"""
Admin Utility: Pack test ground-truth masks into base64-encoded gzip string.
Outputs the exact string to paste into GitHub Repository Secrets as TEST_LABELS.
"""

import gzip
import base64
import sys
from pathlib import Path

# Enable UTF-8 console output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass



def main():
    default_path = Path(__file__).resolve().parent.parent / "data" / "public" / "sample_submission.csv"
    truth_file = Path(sys.argv[1]) if len(sys.argv) > 1 else default_path

    if not truth_file.exists():
        print(f"❌ Error: File not found: {truth_file}")
        sys.exit(1)

    with open(truth_file, "rb") as f:
        data = f.read()

    compressed = gzip.compress(data)
    encoded = base64.b64encode(compressed).decode("utf-8")

    print("=" * 70)
    print("⚠️ GITHUB REPOSITORY SECRET: TEST_LABELS")
    print(f"File packed: {truth_file} ({len(data)} bytes -> {len(encoded)} base64 chars)")
    print("Go to: GitHub Repo -> Settings -> Secrets and variables -> Actions")
    print("Create a secret named: TEST_LABELS and paste the following string:")
    print("=" * 70)
    print(encoded)
    print("=" * 70)


if __name__ == "__main__":
    main()
