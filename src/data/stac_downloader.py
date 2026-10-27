"""
Sentinel-2 Satellite Imagery Acquisition and Patch Extraction for India Forest Hotspots.
Fetches real Sentinel-2 satellite data using Planetary Computer STAC API and curates
training, validation, and test datasets of forest and deforestation patches.
"""

import io
import os
import random
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from PIL import Image
import requests


# Coordinates and STAC query parameters for key Indian forest regions
HOTSPOT_REGIONS = {
    "western_ghats": {
        "name": "Western Ghats Tropical Rainforest (Karnataka/Kerala)",
        "bbox": [75.2, 12.5, 75.5, 12.8],
        "date_range": "2023-01-01T00:00:00Z/2023-04-30T00:00:00Z",
        "cloud_max": 15,
        "preview_url": "https://planetarycomputer.microsoft.com/api/data/v1/item/preview.png?collection=sentinel-2-l2a&item=S2B_MSIL2A_20230421T051649_R062_T43PEP_20240902T023432&assets=visual&asset_bidx=visual%7C1,2,3&nodata=0&format=png",
    },
    "odisha_mining_forest": {
        "name": "Odisha Forest & Mining Belt (Keonjhar/Mayurbhanj)",
        "bbox": [85.2, 21.8, 85.6, 22.2],
        "date_range": "2023-01-01T00:00:00Z/2023-04-30T00:00:00Z",
        "cloud_max": 15,
        "preview_url": "https://planetarycomputer.microsoft.com/api/data/v1/item/preview.png?collection=sentinel-2-l2a&item=S2A_MSIL2A_20230427T044701_R076_T45QUE_20240806T181206&assets=visual&asset_bidx=visual%7C1,2,3&nodata=0&format=png",
    },
    "northeast_assam": {
        "name": "Northeast India Subtropical Forest (Assam/Arunachal)",
        "bbox": [93.0, 26.5, 93.4, 27.0],
        "date_range": "2023-01-01T00:00:00Z/2023-04-30T00:00:00Z",
        "cloud_max": 15,
        "preview_url": "https://planetarycomputer.microsoft.com/api/data/v1/item/preview.png?collection=sentinel-2-l2a&item=S2A_MSIL2A_20230418T041551_R090_T46REQ_20240905T085043&assets=visual&asset_bidx=visual%7C1,2,3&nodata=0&format=png",
    },
    "central_india": {
        "name": "Central Indian Teak & Sal Corridor (Balaghat/Kanha)",
        "bbox": [80.3, 21.9, 80.7, 22.3],
        "date_range": "2023-01-01T00:00:00Z/2023-04-30T00:00:00Z",
        "cloud_max": 15,
        "preview_url": "https://planetarycomputer.microsoft.com/api/data/v1/item/preview.png?collection=sentinel-2-l2a&item=S2B_MSIL2A_20230418T050659_R019_T44QMK_20240905T094026&assets=visual&asset_bidx=visual%7C1,2,3&nodata=0&format=png",
    },
}


def fetch_stac_scene(region_key: str, region_cfg: dict) -> Tuple[Image.Image, str]:
    """Downloads a real Sentinel-2 scene preview from STAC or direct URL."""
    url = region_cfg.get("preview_url")
    granule_id = "unknown"
    if url:
        try:
            r = requests.get(url, timeout=25)
            if r.status_code == 200:
                img = Image.open(io.BytesIO(r.content)).convert("RGB")
                # extract granule id from url
                if "item=" in url:
                    granule_id = url.split("item=")[1].split("&")[0]
                return img, granule_id
        except Exception as e:
            print(f"Warning: direct URL download failed for {region_key}: {e}")

    # Fallback: Query STAC search API
    stac_url = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
    body = {
        "collections": ["sentinel-2-l2a"],
        "bbox": region_cfg["bbox"],
        "datetime": region_cfg["date_range"],
        "query": {"eo:cloud_cover": {"lt": region_cfg["cloud_max"]}},
        "limit": 1,
    }
    r = requests.post(stac_url, json=body, timeout=20)
    features = r.json().get("features", [])
    if not features:
        raise RuntimeError(f"No Sentinel-2 features found for region {region_key}")

    item = features[0]
    granule_id = item["id"]
    preview_url = item["assets"]["rendered_preview"]["href"]
    pr = requests.get(preview_url, timeout=30)
    img = Image.open(io.BytesIO(pr.content)).convert("RGB")
    return img, granule_id


def extract_labeled_patches(
    scene_img: Image.Image,
    region_key: str,
    granule_id: str,
    patch_size: int = 128,
    stride: int = 64,
) -> List[dict]:
    """
    Slices a 1024x1024 Sentinel-2 scene into overlapping patches, computes
    vegetation & soil spectral indicators, and labels patches into:
    - 0: 'intact_forest'
    - 1: 'deforested' (clear-cut, logging scars, exposed soil, degraded canopy)
    """
    arr = np.array(scene_img).astype(np.float32) / 255.0
    h, w, c = arr.shape
    patches = []

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            crop = arr[y : y + patch_size, x : x + patch_size]
            r = crop[..., 0]
            g = crop[..., 1]
            b = crop[..., 2]

            # Green Leaf Index (GLI): (2G - R - B) / (2G + R + B + eps)
            gli = (2.0 * g - r - b) / (2.0 * g + r + b + 1e-6)
            # Visible Atmospherically Resistant Index (VARI): (G - R) / (G + R - B + eps)
            vari = (g - r) / (g + r - b + 1e-6)
            # Brightness and Red-to-Green ratio (soil indicator)
            brightness = (r + g + b) / 3.0
            soil_ratio = r / (g + 1e-6)

            mean_gli = float(np.mean(gli))
            mean_vari = float(np.mean(vari))
            mean_bright = float(np.mean(brightness))
            mean_soil = float(np.mean(soil_ratio))

            # Skip uninformative water/shadow or extreme cloud patches
            if mean_bright < 0.05 or mean_bright > 0.85:
                continue

            # Class determination:
            # Forest: high green leaf index, low red/green ratio, moderate brightness
            # Deforested: lower vegetation index, exposed red/brown soil or high reflectance
            if mean_gli > 0.12 and mean_soil < 0.95:
                label = 0
                class_name = "intact_forest"
            elif mean_soil >= 1.02 or mean_gli < 0.04 or (mean_bright > 0.38 and mean_gli < 0.10):
                label = 1
                class_name = "deforested"
            else:
                # Transition / marginal patch
                if mean_gli >= 0.08:
                    label = 0
                    class_name = "intact_forest"
                else:
                    label = 1
                    class_name = "deforested"

            pil_patch = Image.fromarray((crop * 255.0).astype(np.uint8))
            patches.append({
                "region": region_key,
                "granule_id": granule_id,
                "x": x,
                "y": y,
                "mean_gli": round(mean_gli, 4),
                "mean_vari": round(mean_vari, 4),
                "mean_brightness": round(mean_bright, 4),
                "mean_soil_ratio": round(mean_soil, 4),
                "label": label,
                "class_name": class_name,
                "image": pil_patch,
            })

    return patches


def build_sentinel2_dataset(
    output_dir: str,
    patch_size: int = 128,
    stride: int = 64,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Downloads Sentinel-2 scenes for all 4 Indian forest hotspots,
    slices them into patches, splits them into train/val/test,
    saves the image files and exports metadata.csv.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    splits = ["train", "val", "test"]
    classes = ["intact_forest", "deforested"]

    for split in splits:
        for cls in classes:
            os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)

    all_records = []
    print("--- Starting Sentinel-2 Acquisition for Indian Forest Hotspots ---")

    for region_key, cfg in HOTSPOT_REGIONS.items():
        print(f"Fetching Sentinel-2 scene: {cfg['name']}...")
        scene_img, granule_id = fetch_stac_scene(region_key, cfg)
        print(f"  Downloaded granule {granule_id} ({scene_img.size})")

        patches = extract_labeled_patches(
            scene_img, region_key, granule_id, patch_size=patch_size, stride=stride
        )
        print(f"  Extracted {len(patches)} candidate patches from {region_key}")

        # Group by label to balance and split
        random.shuffle(patches)
        for i, p in enumerate(patches):
            rand = random.random()
            if rand < train_ratio:
                split = "train"
            elif rand < (train_ratio + val_ratio):
                split = "val"
            else:
                split = "test"

            filename = f"{region_key}_p{i:04d}_{p['class_name']}.png"
            rel_path = os.path.join(split, p["class_name"], filename)
            abs_path = os.path.join(output_dir, rel_path)
            p["image"].save(abs_path, format="PNG")

            all_records.append({
                "patch_id": f"{region_key}_{i:04d}",
                "file_path": rel_path,
                "region": region_key,
                "granule_id": granule_id,
                "split": split,
                "label": p["label"],
                "class_name": p["class_name"],
                "mean_gli": p["mean_gli"],
                "mean_vari": p["mean_vari"],
                "mean_brightness": p["mean_brightness"],
                "mean_soil_ratio": p["mean_soil_ratio"],
            })

    df = pd.DataFrame(all_records)
    csv_path = os.path.join(output_dir, "metadata.csv")
    df.to_csv(csv_path, index=False)
    print("\n--- Dataset Build Complete ---")
    print(f"Total patches saved: {len(df)}")
    print(df.groupby(["split", "class_name"]).size())
    print(f"Metadata exported to: {csv_path}")
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/satellite/india_sentinel2",
        help="Path where satellite dataset will be stored",
    )
    args = parser.parse_args()
    build_sentinel2_dataset(args.output_dir)
