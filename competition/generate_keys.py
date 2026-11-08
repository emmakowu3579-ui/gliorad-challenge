"""
Admin Utility: Generate new RSA keypair for challenge scoring.
Generates public_key.pem for starter_code/ and outputs private key to paste into GitHub Secrets.
"""

import rsa
from pathlib import Path


def main():
    print("🔑 Generating 2048-bit RSA Keypair...")
    (public_key, private_key) = rsa.newkeys(2048)

    pub_pem = public_key.save_pkcs1()
    priv_pem = private_key.save_pkcs1().decode("utf-8")

    pub_path = Path(__file__).resolve().parent.parent / "starter_code" / "public_key.pem"
    with open(pub_path, "wb") as f:
        f.write(pub_pem)

    print(f"✅ Saved public key to: {pub_path}")
    print("\n" + "=" * 70)
    print("⚠️ ACTION REQUIRED: COPY THIS PRIVATE KEY TO GITHUB REPOSITORY SECRETS")
    print("Go to: GitHub Repo -> Settings -> Secrets and variables -> Actions")
    print("Create a new secret named: PRIVATE_KEY")
    print("=" * 70)
    print(priv_pem)
    print("=" * 70)


if __name__ == "__main__":
    main()
