"""Stitch the Earth Engine exports into one all-India grid.

Earth Engine splits big GeoTIFF exports into several files (india_loss_640m-0000000000-0000000000.tif,
...). This merges them and keeps two aligned stacks:

    loss  (Y+1, H, W) float32   f2000 and the share of each square lost per year, 0-1
    cov   (C, H, W)   float32   the covariates from gee.COVARIATES

    python -m deforest.national.table --src /content/drive/MyDrive/deforestation-india --out india_grid.npz
"""
from __future__ import annotations

import argparse
import glob
import os
from typing import List, Tuple

import numpy as np


def _merge(paths: List[str]) -> Tuple[np.ndarray, object, List[str], object]:
    import rasterio
    from rasterio.merge import merge
    if not paths:
        raise FileNotFoundError("no GeoTIFFs found; are the Earth Engine exports finished?")
    srcs = [rasterio.open(p) for p in sorted(paths)]
    try:
        names = list(srcs[0].descriptions)
        arr, transform = merge(srcs) if len(srcs) > 1 else (srcs[0].read(), srcs[0].transform)
        crs = srcs[0].crs
    finally:
        for s in srcs:
            s.close()
    if any(n is None for n in names):          # GEE writes band names; fall back to positions if missing
        names = [f"b{i}" for i in range(arr.shape[0])]
    return arr, transform, names, crs


def build(src: str, out: str) -> dict:
    loss, transform, lnames, crs = _merge(glob.glob(os.path.join(src, "india_loss_640m*.tif")))
    cov, t2, cnames, _ = _merge(glob.glob(os.path.join(src, "india_cov_640m*.tif")))
    if loss.shape[1:] != cov.shape[1:] or tuple(transform)[:6] != tuple(t2)[:6]:
        raise ValueError(f"loss grid {loss.shape} and covariate grid {cov.shape} do not line up")
    years = [int(n[1:]) for n in lnames if n.startswith("l") and n[1:].isdigit()]
    if lnames[0] != "f2000" or years != list(range(2001, 2001 + len(years))):
        raise ValueError(f"unexpected loss bands {lnames[:3]}...{lnames[-2:]}")
    loss = loss.astype(np.float32) / 10000.0
    np.savez_compressed(out, loss=loss, cov=cov.astype(np.float32), years=np.array(years),
                        cov_names=np.array(cnames), transform=np.array(tuple(transform)[:6]), crs=str(crs))
    inside = cov[cnames.index("state")] > 0
    forest = loss[0] >= 0.10
    info = dict(shape=loss.shape[1:], years=f"{years[0]}-{years[-1]}", squares_in_india=int(inside.sum()),
                forest_squares=int((inside & forest).sum()),
                forest_ha_2000=round(float((loss[0] * inside).sum() * 40.96)),
                lost_ha=round(float((loss[1:].sum(0) * inside).sum() * 40.96)))
    print(info)
    return info


def load(path: str) -> dict:
    z = np.load(path, allow_pickle=False)
    return dict(loss=z["loss"], cov=z["cov"], years=[int(y) for y in z["years"]],
                cov_names=[str(c) for c in z["cov_names"]], transform=tuple(z["transform"]), crs=str(z["crs"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True, help="Drive folder with the Earth Engine exports")
    ap.add_argument("--out", default="india_grid.npz")
    a = ap.parse_args()
    build(a.src, a.out)
