"""
Submission Decryptor for TCGA-LGG Challenge Auto-Grader.
Decrypts participant .enc submissions using PRIVATE_KEY from GitHub Secrets.
"""

import rsa
import sys
import os
import glob
from pathlib import Path

# Enable UTF-8 console output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def decrypt_file(enc_file_path: str):
    private_key_data = os.environ.get("PRIVATE_KEY")
    if not private_key_data:
        # Fallback to local admin key if present for local testing
        candidate_paths = [
            Path("keys/private_key.pem"),
            Path("../keys/private_key.pem"),
            Path(__file__).resolve().parent / "private_key.pem",
            Path(__file__).resolve().parent.parent / "keys" / "private_key.pem",
        ]
        for cp in candidate_paths:
            if cp.exists():
                private_key_data = cp.read_text(encoding="utf-8")
                break

        if not private_key_data:
            print("❌ Error: PRIVATE_KEY environment secret is missing!")
            sys.exit(1)

    private_key = rsa.PrivateKey.load_pkcs1(private_key_data.encode("utf8"))

    if not os.path.exists(enc_file_path):
        print(f"❌ Error: The file {enc_file_path} does not exist!")
        sys.exit(1)

    print(f"🔓 Decrypting: {enc_file_path}")

    with open(enc_file_path, "rb") as f:
        encrypted_data = f.read()

    decrypted_data = b""
    chunk_size = 256

    try:
        for i in range(0, len(encrypted_data), chunk_size):
            chunk = encrypted_data[i:i + chunk_size]
            decrypted_chunk = rsa.decrypt(chunk, private_key)
            decrypted_data += decrypted_chunk

        # Strip .enc extension
        output_path = enc_file_path[:-4] if enc_file_path.endswith(".enc") else f"{enc_file_path}.dec"
        with open(output_path, "wb") as f:
            f.write(decrypted_data)

        print(f"✅ Decrypted to: {output_path}")
        return output_path

    except Exception as e:
        print(f"❌ Decryption failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        decrypt_file(sys.argv[1])
    else:
        enc_files = glob.glob("submissions/*.enc")
        if not enc_files:
            print("ℹ️ No .enc file found in submissions/")
            sys.exit(0)
        decrypt_file(enc_files[0])
