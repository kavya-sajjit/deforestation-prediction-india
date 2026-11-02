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
    "v1_models": "e01a742f805684a69282528bbdec33446aa5fcb4f63bf7915f29059bfeb3cc5f",
    "gfw_tables": "d5cbb2fc37ac594e025d04b8e1ec0a06b3b270e23d6c9069a0c349a9d32852a4",
    "v1_satellite_png": "fe76bda4d7e2de3d5b606f30ce4c4a799f14ed3db9337fb791ecdbc8c8ca99c1",
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
