"""Forest-loss risk map for a whole region from a trained forecast model.

    python -m deforest.risk_map --data data/v2 --region odisha_mining \
        --model results/forecast/block/bcos_rgbn_hist/fold0/model.pt --out maps/

Writes a GeoTIFF (probability per 640 m cell, NaN outside forest) and a PNG
overlay. Use a model whose test fold did not include the region if you want
the map to be an honest out-of-sample forecast (e.g. a leave-one-region-out model).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from . import config as C
from .data.dataset import PatchDataset, load_all
from .models.zoo import build_model
from .train import device, predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/v2")
    ap.add_argument("--region", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="maps")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    ck = torch.load(a.model, map_location="cpu")
    meta, arrays = load_all(a.data, [a.region])
    model = build_model(ck["variant"], ck["n_channels"], download=False)
    model.load_state_dict(ck["state_dict"]); model.to(device()).eval()
    ds = PatchDataset(meta, arrays, meta.index, task=ck["task"], bands=ck["bands"], history=ck["history"],
                      with_ndvi=ck["variant"] == "bcos_pre_ndvi")
    from torch.utils.data import DataLoader
    p = predict(model, DataLoader(ds, batch_size=128), ck["resize"], ck["variant"] == "bcos_pre_ndvi")

    with open(os.path.join(a.data, f"{a.region}.json")) as fh:
        info = json.load(fh)
    rows, cols = info["shape"][0] // C.PATCH, info["shape"][1] // C.PATCH
    grid = np.full((rows, cols), np.nan, np.float32)
    grid[meta.row.to_numpy(), meta.col.to_numpy()] = p

    import rasterio
    from affine import Affine
    t = Affine(*info["transform"]) * Affine.scale(C.PATCH)
    tif = os.path.join(a.out, f"{a.region}_risk.tif")
    with rasterio.open(tif, "w", driver="GTiff", height=rows, width=cols, count=1, dtype="float32",
                       crs=info["crs"], transform=t, nodata=np.nan) as dst:
        dst.write(grid, 1)

    # GeoJSON of forest cells with their risk, in WGS84, for the website map
    from rasterio.warp import transform as warp
    feats = []
    for (r_, c_), prob, y in zip(meta[["row", "col"]].to_numpy(), p, meta.y_forecast.to_numpy()):
        xs = [c_, c_ + 1, c_ + 1, c_, c_]
        ys = [r_, r_, r_ + 1, r_ + 1, r_]
        px = [t * (x, yy) for x, yy in zip(xs, ys)]
        lon, lat = warp(info["crs"], "EPSG:4326", [q[0] for q in px], [q[1] for q in px])
        feats.append(dict(type="Feature",
                          properties=dict(risk=round(float(prob), 4), row=int(r_), col=int(c_),
                                          lost_2020_23=int(y == 1), label_known=int(y >= 0)),
                          geometry=dict(type="Polygon", coordinates=[[[round(a, 6), round(b, 6)]
                                                                       for a, b in zip(lon, lat)]])))
    with open(os.path.join(a.out, f"{a.region}_risk.geojson"), "w") as fh:
        json.dump(dict(type="FeatureCollection", features=feats), fh)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r = np.load(os.path.join(a.data, f"{a.region}_rasters.npz"))
    rgb = np.clip(r["t0"][[0, 1, 2]].astype(np.float32).transpose(1, 2, 0) * 2.5, 0, 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 5.5))
    ax[0].imshow(np.nan_to_num(rgb)); ax[0].set_title(f"{a.region}: Sentinel-2, early 2019")
    ax[1].imshow(np.nan_to_num(rgb)); ax[1].imshow(np.kron(grid, np.ones((C.PATCH, C.PATCH))), cmap="magma",
                                                   alpha=0.6, vmin=0, vmax=1)
    ax[1].contour(r["loss_target"], levels=[0.5], colors="cyan", linewidths=0.4)
    ax[1].set_title("Predicted risk of loss 2020–23 (cyan: actual Hansen loss)")
    for x in ax:
        x.axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, f"{a.region}_risk.png"), dpi=130)
    print("wrote", tif)


if __name__ == "__main__":
    main()
