"""Hansen et al. (2013) Global Forest Change layers, reprojected onto a Sentinel-2 grid.

Only the window that overlaps the region is read (GDAL HTTP range requests),
so nothing close to a full 10x10 degree tile is downloaded.
"""
from __future__ import annotations

import math
import os
from typing import List, Tuple

import numpy as np

from .. import config as C


def tiles_for_bbox(bbox: Tuple[float, float, float, float]) -> List[str]:
    """Hansen tiles are 10x10 degrees named by their top-left corner, e.g. 30N_080E."""
    lon0, lat0, lon1, lat1 = bbox
    names = []
    for top in range(int(math.ceil(lat1 / 10) * 10), int(math.floor(lat0 / 10) * 10), -10):
        for left in range(int(math.floor(lon0 / 10) * 10), int(math.ceil(lon1 / 10) * 10), 10):
            ns = f"{abs(top):02d}{'N' if top >= 0 else 'S'}"
            ew = f"{abs(left):03d}{'E' if left >= 0 else 'W'}"
            names.append(f"{ns}_{ew}")
    return names


def read_layer(layer: str, bbox, dst_transform, dst_crs, dst_shape) -> np.ndarray:
    """Read one GFC layer ('treecover2000' or 'lossyear') onto the destination grid."""
    import rasterio
    from rasterio.warp import Resampling, reproject

    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")
    dst = np.full(dst_shape, 255, dtype=np.uint8)
    for tile in tiles_for_bbox(bbox):
        url = C.HANSEN_URL.format(v=C.HANSEN_VERSION, layer=layer, tile=tile)
        with rasterio.open(f"/vsicurl/{url}") as src:
            tmp = np.full(dst_shape, 255, dtype=np.uint8)
            reproject(source=rasterio.band(src, 1), destination=tmp,
                      dst_transform=dst_transform, dst_crs=dst_crs,
                      dst_nodata=255, resampling=Resampling.nearest)
            dst = np.where(tmp != 255, tmp, dst)
    return dst


def label_rasters(treecover: np.ndarray, lossyear: np.ndarray) -> dict:
    """Pixel masks (uint8, 0/1) that define every label in the project."""
    forest2000 = treecover >= C.TREECOVER_THRESHOLD
    hist = np.isin(lossyear, list(C.HISTORY_LOSS_YEARS)) & forest2000
    forest_t0 = forest2000 & ~hist                         # still forest at the start of 2019
    return dict(
        forest_t0=forest_t0.astype(np.uint8),
        loss_hist=hist.astype(np.uint8),
        loss_2019=(np.isin(lossyear, list(C.AMBIGUOUS_LOSS_YEARS)) & forest_t0).astype(np.uint8),
        loss_target=(np.isin(lossyear, list(C.TARGET_LOSS_YEARS)) & forest_t0).astype(np.uint8),
        loss_detect=(np.isin(lossyear, list(C.DETECT_LOSS_YEARS)) & forest_t0).astype(np.uint8),
    )
