"""Build the v2 dataset for every region in config.REGIONS.

    python -m deforest.data.build --out data/v2

Per region this writes:
  <region>.npz   patches (T0, T1 imagery; pixel loss masks)
  <region>.csv   one row per patch: position, block, forest share, labels
  <region>.json  grid, CRS, scene counts, label statistics
  <region>_rasters.npz  full composites + Hansen layers (for risk maps)
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .. import config as C
from . import hansen, patches, sentinel2


def build_region(name: str, out: str) -> dict:
    bbox = C.REGIONS[name]["bbox"]
    print(f"[{name}] T0 composite {C.T0_WINDOW}")
    c0 = sentinel2.composite(bbox, C.T0_WINDOW)
    print(f"[{name}] T1 composite {C.T1_WINDOW}")
    c1 = sentinel2.composite(bbox, C.T1_WINDOW, crs_epsg=c0.attrs["epsg"])
    c1 = c1.reindex_like(c0, method="nearest")          # guarantee identical grids

    t0 = sentinel2.to_unit_range(np.stack([c0[b].values for b in C.S2_BANDS]).astype(np.float32))
    t1 = sentinel2.to_unit_range(np.stack([c1[b].values for b in C.S2_BANDS]).astype(np.float32))
    # keep NaNs (no clear observation) so patch filtering can see them
    t0[~np.isfinite(np.stack([c0[b].values for b in C.S2_BANDS]))] = np.nan
    t1[~np.isfinite(np.stack([c1[b].values for b in C.S2_BANDS]))] = np.nan

    geobox = c0.odc.geobox
    transform, crs, shape = geobox.transform, f"EPSG:{c0.attrs['epsg']}", t0.shape[1:]
    print(f"[{name}] Hansen {C.HANSEN_VERSION} on {shape} grid")
    tc = hansen.read_layer("treecover2000", bbox, transform, crs, shape)
    ly = hansen.read_layer("lossyear", bbox, transform, crs, shape)
    labels = hansen.label_rasters(tc, ly)

    arrays, meta = patches.extract(name, t0, t1, labels)
    info = dict(
        region=name, bbox=bbox, biome=C.REGIONS[name]["biome"], crs=crs,
        transform=list(transform)[:6], shape=list(shape),
        scenes_t0=c0.attrs["n_scenes"], scenes_t1=c1.attrs["n_scenes"],
        patches=int(len(meta)),
        forecast_pos=int((meta.y_forecast == 1).sum()), forecast_neg=int((meta.y_forecast == 0).sum()),
        detect_pos=int((meta.y_detect == 1).sum()), detect_neg=int((meta.y_detect == 0).sum()),
        hansen=C.HANSEN_VERSION,
    )
    patches.save_region(out, name, arrays, meta, info)
    np.savez_compressed(os.path.join(out, f"{name}_rasters.npz"),
                        t0=t0.astype(np.float16), t1=t1.astype(np.float16), **labels)
    print(json.dumps(info, indent=2))
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/v2")
    ap.add_argument("--regions", nargs="*", default=list(C.REGIONS))
    a = ap.parse_args()
    stats = [build_region(r, a.out) for r in a.regions]
    with open(os.path.join(a.out, "dataset_summary.json"), "w") as fh:
        json.dump(stats, fh, indent=2)


if __name__ == "__main__":
    main()
