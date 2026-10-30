"""Export everything the website needs into web/public/data.

    python -m deforest.web_export --data data/v2 --results results --out web/public/data

Writes, per region, a compact JSON with the patch grid, the forest lost in every
year 2001-2023 (from Hansen), and the model's out-of-fold risk for 2020-23.
Out-of-fold means every patch is scored by the model trained WITHOUT that
patch's spatial block, so the map shows honest predictions, not memorised ones.
Also writes summary.json (model comparison) and copies explanation galleries.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil

import numpy as np
import pandas as pd

from . import config as C
from .data.io import load_all

RISK_MODELS = ["bcos_rgbn_hist", "cnn_rgbn_hist", "gbm", "history_lr"]


# --------------------------------------------------------------------------- Hansen per year
def _hansen_rasters(data_dir: str, region: str, info: dict):
    """(treecover2000, lossyear) on the patch grid; cached as <region>_hansen.npz."""
    cache = os.path.join(data_dir, f"{region}_hansen.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return z["treecover"], z["lossyear"]
    if "transform" not in info:
        return None, None
    from affine import Affine
    from .data import hansen
    t = Affine(*info["transform"])
    shape = tuple(info["shape"])
    tc = hansen.read_layer("treecover2000", info["bbox"], t, info["crs"], shape)
    ly = hansen.read_layer("lossyear", info["bbox"], t, info["crs"], shape)
    np.savez_compressed(cache, treecover=tc, lossyear=ly)
    return tc, ly


def _yearly_loss(tc, ly, rows, cols):
    """Per patch: share of the patch that was forest in 2000, and share lost each year."""
    P = C.PATCH
    tc = tc[: rows * P, : cols * P]
    ly = ly[: rows * P, : cols * P]
    forest = (tc >= C.TREECOVER_THRESHOLD)
    f = forest.reshape(rows, P, cols, P).mean(axis=(1, 3))
    yr = np.where(forest, ly, 0).reshape(rows, P, cols, P).transpose(0, 2, 1, 3).reshape(rows, cols, P * P)
    out = np.zeros((rows, cols, 24), np.float32)
    for y in range(1, 24):
        out[..., y] = (yr == y).mean(axis=2)
    return f, out


# --------------------------------------------------------------------------- geometry
def _grid_nodes(info: dict, rows: int, cols: int):
    """lon/lat of every patch corner: arrays of shape (rows+1, cols+1)."""
    rr, cc = np.meshgrid(np.arange(rows + 1), np.arange(cols + 1), indexing="ij")
    if "transform" in info and "crs" in info:
        from affine import Affine
        from rasterio.warp import transform as warp
        t = Affine(*info["transform"]) * Affine.scale(C.PATCH)
        xs, ys = t * (cc.ravel(), rr.ravel())
        lon, lat = warp(info["crs"], "EPSG:4326", list(xs), list(ys))
        return np.array(lon).reshape(rr.shape), np.array(lat).reshape(rr.shape)
    lon0, lat0, lon1, lat1 = info.get("bbox", C.REGIONS[info["region"]]["bbox"])
    return lon0 + (lon1 - lon0) * cc / cols, lat1 - (lat1 - lat0) * rr / rows


# --------------------------------------------------------------------------- predictions
def _oof(results: str, model: str, scheme: str = "block") -> pd.DataFrame:
    files = glob.glob(os.path.join(results, "forecast", scheme, model, "*", "preds.csv"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(f) for f in files])
    return df[df.split == "test"].drop_duplicates("idx").set_index("idx")


def _summary(results: str, data_dir: str) -> dict:
    out = {"tasks": {}}
    for task in ("forecast", "detect"):
        f = os.path.join(results, f"{task}_block.csv")
        if os.path.exists(f):
            t = pd.read_csv(f)
            keep = ["experiment", "splits", "ap", "ap_sd", "roc_auc", "mcc", "capture_top10", "chance_ap",
                    "test_patches", "test_positives"]
            out["tasks"][task] = t[[k for k in keep if k in t]].round(4).to_dict(orient="records")
    ds = os.path.join(data_dir, "dataset_summary.json")
    if os.path.exists(ds):
        with open(ds) as fh:
            out["dataset"] = json.load(fh)
    return out


# --------------------------------------------------------------------------- main
def export(data_dir: str, results: str, out: str, use_hansen: bool = True):
    os.makedirs(out, exist_ok=True)
    meta, _ = load_all(data_dir)
    risk_model, oof = None, pd.DataFrame()
    for m in RISK_MODELS:
        oof = _oof(results, m)
        if len(oof):
            risk_model = m
            break
    gbm = _oof(results, "gbm")
    index = []
    for region in sorted(meta.region.unique(), key=list(C.REGIONS).index):
        with open(os.path.join(data_dir, f"{region}.json")) as fh:
            info = json.load(fh)
        info.setdefault("region", region)
        m = meta[meta.region == region]
        if "shape" in info:
            rows, cols = info["shape"][0] // C.PATCH, info["shape"][1] // C.PATCH
        else:
            rows, cols = int(m.row.max()) + 1, int(m.col.max()) + 1
        lon, lat = _grid_nodes(info, rows, cols)

        tc, ly = _hansen_rasters(data_dir, region, info) if use_hansen else (None, None)
        patches = []
        if tc is not None:
            f2000, yearly = _yearly_loss(tc, ly, rows, cols)
            lookup = {(int(a.row), int(a.col)): i for i, a in m.iterrows()}
            for r in range(rows):
                for c in range(cols):
                    if f2000[r, c] < 0.2:
                        continue
                    rec = {"r": r, "c": c, "f": round(float(f2000[r, c]), 3)}
                    loss = {str(2000 + y): round(float(yearly[r, c, y]), 4)
                            for y in range(1, 24) if yearly[r, c, y] > 0}
                    if loss:
                        rec["l"] = loss
                    i = lookup.get((r, c))
                    if i is not None:
                        rec["y"] = int(m.at[i, "y_forecast"])
                        if i in oof.index:
                            rec["p"] = round(float(oof.at[i, "p"]), 4)
                        if i in gbm.index:
                            rec["g"] = round(float(gbm.at[i, "p"]), 4)
                    patches.append(rec)
        else:  # no Hansen access: use the three label buckets from the patch table
            for i, a in m.iterrows():
                rec = {"r": int(a.row), "c": int(a.col), "f": round(float(a.forest_frac + a.hist_loss_frac), 3),
                       "l": {k: v for k, v in {"2010": round(float(a.hist_loss_frac), 4),
                                               "2019": round(float(a.loss2019_frac), 4),
                                               "2022": round(float(a.target_frac), 4)}.items() if v > 0},
                       "y": int(a.y_forecast)}
                if i in oof.index:
                    rec["p"] = round(float(oof.at[i, "p"]), 4)
                if i in gbm.index:
                    rec["g"] = round(float(gbm.at[i, "p"]), 4)
                patches.append(rec)

        doc = dict(region=region, name=region.replace("_", " ").title(),
                   biome=C.REGIONS[region]["biome"], bbox=C.REGIONS[region]["bbox"],
                   rows=rows, cols=cols, patch_m=C.PATCH * 10,
                   lon=np.round(lon, 5).tolist(), lat=np.round(lat, 5).tolist(),
                   risk_model=risk_model, yearly=tc is not None,
                   synthetic=bool(info.get("synthetic", False)), patches=patches)
        with open(os.path.join(out, f"{region}.json"), "w") as fh:
            json.dump(doc, fh, separators=(",", ":"))
        index.append(dict(region=region, name=doc["name"], biome=doc["biome"], bbox=doc["bbox"],
                          patches=len(patches), synthetic=doc["synthetic"]))
        print(f"[{region}] {len(patches)} patches, yearly={doc['yearly']}, risk model={risk_model}")

    summary = _summary(results, data_dir)
    summary["regions"] = index
    summary["risk_model"] = risk_model
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)

    img = os.path.join(os.path.dirname(out.rstrip("/")), "img")
    os.makedirs(img, exist_ok=True)
    for task, exp in (("forecast", "bcos_rgbn_hist"), ("detect", "bcos_bitemporal")):
        g = sorted(glob.glob(os.path.join(results, task, "block", exp, "*", "gallery.png")))
        if g:
            shutil.copy(g[0], os.path.join(img, f"gallery_{task}.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/v2")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="web/public/data")
    ap.add_argument("--no-hansen", action="store_true", help="skip re-reading Hansen (no yearly detail)")
    a = ap.parse_args()
    export(a.data, a.results, a.out, use_hansen=not a.no_hansen)


if __name__ == "__main__":
    main()
