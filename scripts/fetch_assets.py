"""Download the large files that are no longer stored in git.

They live as assets on the GitHub Release `v1-assets`:
  v1_models.zip         models/*.pth (v1 CNN + B-cos) and models/risk_model.pkl (state-level RF)
  gfw_tables.zip        data/raw/*.xlsx (Global Forest Watch country / state / district tables)
  v1_satellite_png.zip  data/satellite/ (the v1 RGB patch dataset, kept for reference only)

Usage:
    python scripts/fetch_assets.py              # everything
    python scripts/fetch_assets.py gfw_tables   # one archive
"""
import hashlib
import io
import os
import sys
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = "https://github.com/kavya-sajjit/deforestation-prediction-india/releases/download/v1-assets/"
ASSETS = {
    "v1_models": "0b245c2fc3fc8dbfbab53401bc1c66d6b2e9dc8a37a9e4f437fd5ce6a90214e7",
    "gfw_tables": "15c02b2e9017a49f496fb636a2ac5cc45b52fae5e4904e9d4aee81a6192f4832",
    "v1_satellite_png": "ca79f1092d268c03c0276963b885808ae7adb77105564336dd3f54d6af9fd0f1",
}


def fetch(name):
    url = BASE + name + ".zip"
    print(f"downloading {url}")
    raw = urllib.request.urlopen(url).read()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != ASSETS[name]:
        sys.exit(f"checksum mismatch for {name}.zip: got {digest}")
    zipfile.ZipFile(io.BytesIO(raw)).extractall(ROOT)
    print(f"  extracted {name}.zip")


if __name__ == "__main__":
    for name in sys.argv[1:] or ASSETS:
        fetch(name)
