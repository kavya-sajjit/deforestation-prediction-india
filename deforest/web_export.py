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
from . import risk_levels as R
from .data.io import load_all

RISK_MODELS = ["bcos_rgbn_hist", "cnn_rgbn_hist", "gbm", "history_lr"]


# --------------------------------------------------------------------------- Hansen per year
# Newest first. The website only needs yearly loss, so it can use a newer release than the one the
# models were trained on (config.HANSEN_VERSION); the training labels are unaffected.
WEB_HANSEN_VERSIONS = ("GFC-2025-v1.13", "GFC-2024-v1.12", C.HANSEN_VERSION)


def _last_year(version: str) -> int:
    return int(version.split("-")[1])                      # "GFC-2025-v1.13" -> 2025


def _hansen_rasters(data_dir: str, region: str, info: dict, versions=WEB_HANSEN_VERSIONS):
    """(treecover2000, lossyear, version) on the patch grid; cached as <region>_hansen_<version>.npz."""
    if "transform" not in info:
        return None, None, None
    for v in versions:
        cache = os.path.join(data_dir, f"{region}_hansen_{v}.npz")
        if os.path.exists(cache):
            z = np.load(cache)
            return z["treecover"], z["lossyear"], v
    old = os.path.join(data_dir, f"{region}_hansen.npz")    # cache written before versions were tracked
    from affine import Affine
    from .data import hansen
    t = Affine(*info["transform"])
    shape = tuple(info["shape"])
    for v in versions:
        try:
            C.HANSEN_VERSION, keep = v, C.HANSEN_VERSION
            tc = hansen.read_layer("treecover2000", info["bbox"], t, info["crs"], shape)
            ly = hansen.read_layer("lossyear", info["bbox"], t, info["crs"], shape)
        except Exception as e:                              # release not published or not reachable
            print(f"   {v} not available ({type(e).__name__}); trying an older release")
            continue
        finally:
            C.HANSEN_VERSION = keep
        np.savez_compressed(os.path.join(data_dir, f"{region}_hansen_{v}.npz"), treecover=tc, lossyear=ly)
        return tc, ly, v
    if os.path.exists(old):
        z = np.load(old)
        return z["treecover"], z["lossyear"], "GFC-2023-v1.11"
    return None, None, None


def _yearly_loss(tc, ly, rows, cols, last_year: int = 2023):
    """Per patch: share of the patch that was forest in 2000, and share lost each year 2001..last_year."""
    P = C.PATCH
    tc = tc[: rows * P, : cols * P]
    ly = ly[: rows * P, : cols * P]
    forest = (tc >= C.TREECOVER_THRESHOLD)
    f = forest.reshape(rows, P, cols, P).mean(axis=(1, 3))
    yr = np.where(forest, ly, 0).reshape(rows, P, cols, P).transpose(0, 2, 1, 3).reshape(rows, cols, P * P)
    n = last_year - 2000
    out = np.zeros((rows, cols, n + 1), np.float32)
    for y in range(1, n + 1):
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
    rep = None
    if risk_model:  # calibrated risk + tiers, measured on held-out squares
        cal, rep = R.run(results, risk_model)
        oof = oof.join(cal[["q", "tier"]])
        with open(os.path.join(results, "risk_report.json"), "w") as fh:
            json.dump(rep, fh, indent=1)
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

        tc, ly, version = _hansen_rasters(data_dir, region, info) if use_hansen else (None, None, None)
        last = _last_year(version) if version else 2023
        patches = []
        if tc is not None:
            f2000, yearly = _yearly_loss(tc, ly, rows, cols, last)
            lookup = {(int(a.row), int(a.col)): i for i, a in m.iterrows()}
            for r in range(rows):
                for c in range(cols):
                    if f2000[r, c] < 0.2:
                        continue
                    rec = {"r": r, "c": c, "f": round(float(f2000[r, c]), 3)}
                    loss = {str(2000 + y): round(float(yearly[r, c, y]), 4)
                            for y in range(1, last - 2000 + 1) if yearly[r, c, y] > 0}
                    if loss:
                        rec["l"] = loss
                    i = lookup.get((r, c))
                    if i is not None:
                        rec["y"] = int(m.at[i, "y_forecast"])
                        if i in oof.index:
                            rec["p"] = round(float(oof.at[i, "p"]), 4)
                            rec["q"] = round(float(oof.at[i, "q"]), 4)
                            rec["t"] = oof.at[i, "tier"]
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
                    rec["q"] = round(float(oof.at[i, "q"]), 4)
                    rec["t"] = oof.at[i, "tier"]
                if i in gbm.index:
                    rec["g"] = round(float(gbm.at[i, "p"]), 4)
                patches.append(rec)

        spots = R.hotspots([p for p in patches if "q" in p], lon, lat, rows, cols) if rep else []
        if spots:
            with open(os.path.join(out, f"{region}_hotspots.geojson"), "w") as fh:
                json.dump(R.hotspots_geojson(region, spots, lon, lat), fh)
        risk = None
        if rep:
            reg = oof[oof.region == region] if "region" in oof else oof.iloc[:0]
            risk = dict(model=risk_model, base_rate=round(rep["base_rate"], 4),
                        ece=round(rep["ece_calibrated"], 4), tiers=rep["tiers"],
                        region_tiers=R.tier_table(reg, n_boot=200) if len(reg) else None)
        doc = dict(region=region, name=region.replace("_", " ").title(),
                   biome=C.REGIONS[region]["biome"], bbox=C.REGIONS[region]["bbox"],
                   rows=rows, cols=cols, patch_m=C.PATCH * 10,
                   lon=np.round(lon, 5).tolist(), lat=np.round(lat, 5).tolist(),
                   risk_model=risk_model, yearly=tc is not None, last_observed=last, hansen=version,
                   synthetic=bool(info.get("synthetic", False)), risk=risk,
                   hotspots=[{k: v for k, v in h.items() if k != "cells"} for h in spots],
                   patches=patches)
        if tc is not None:                                  # 2026-2036 trend projection, backtested
            from .projection import run_region
            proj = run_region(doc)
            b = proj["backtest"]
            print(f"   projection to {proj['end']}: backtest AP {b['ap']} (chance {b['chance_ap']}), "
                  f"projected/actual total {b['total_ratio']}")
        with open(os.path.join(out, f"{region}.json"), "w") as fh:
            json.dump(doc, fh, separators=(",", ":"))
        index.append(dict(region=region, name=doc["name"], biome=doc["biome"], bbox=doc["bbox"],
                          patches=len(patches), synthetic=doc["synthetic"]))
        print(f"[{region}] {len(patches)} patches, Hansen {version} (observed to {last}), risk model={risk_model}")

    summary = _summary(results, data_dir)
    summary["regions"] = index
    summary["risk_model"] = risk_model
    if rep:
        summary["risk"] = {k: rep[k] for k in ("model", "base_rate", "brier_raw", "brier_calibrated",
                                                "brier_baseline", "ece_raw", "ece_calibrated", "tiers")}
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
