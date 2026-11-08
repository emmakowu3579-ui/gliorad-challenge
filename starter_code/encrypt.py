"""
Submission Encryptor for TCGA-LGG Challenge.
Encrypts participant submission CSV files using the official challenge public key
so they can be safely submitted publicly via GitHub Pull Requests.
"""

import rsa
import sys
import os
from pathlib import Path

# Enable UTF-8 console output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def encrypt_submission(file_path: str):
    key_path = Path(__file__).resolve().parent / "public_key.pem"
    if not key_path.exists():
        print(f"❌ Error: public_key.pem not found at {key_path}")
        sys.exit(1)

    with open(key_path, "rb") as f:
        public_key = rsa.PublicKey.load_pkcs1(f.read())

    sub_path = Path(file_path)
    if not sub_path.exists():
        print(f"❌ Error: Submission file not found at {sub_path}")
        sys.exit(1)

    with open(sub_path, "rb") as f:
        file_data = f.read()

    print(f"🔒 Encrypting {sub_path.name} ({len(file_data)} bytes)...")

    # Encrypt in chunks (245 bytes for 2048-bit RSA with PKCS#1 v1.5 padding)
    encrypted_data = b""
    chunk_size = 245

    try:
        for i in range(0, len(file_data), chunk_size):
            chunk = file_data[i:i + chunk_size]
            encrypted_chunk = rsa.encrypt(chunk, public_key)
            encrypted_data += encrypted_chunk

        output_path = sub_path.parent / f"{sub_path.name}.enc"
        with open(output_path, "wb") as f:
            f.write(encrypted_data)

        print(f"✅ Success! Created encrypted submission: {output_path}")
        print(f"👉 Please commit ONLY the .enc file into the 'submissions/' directory.")
        print(f"👉 Example: git add {output_path}")

    except Exception as e:
        print(f"❌ Encryption failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python encrypt.py <path_to_submission.csv>")
    else:
        encrypt_submission(sys.argv[1])
