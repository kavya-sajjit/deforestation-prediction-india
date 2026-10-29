"""Cloud-masked Sentinel-2 L2A median composites from Microsoft Planetary Computer."""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from .. import config as C


def utm_epsg(lon: float, lat: float) -> int:
    zone = int(math.floor((lon + 180) / 6) + 1)
    return (32600 if lat >= 0 else 32700) + zone


def composite(bbox: Tuple[float, float, float, float], window: str,
              resolution: int = 10, crs_epsg: int | None = None):
    """Median composite of the least-cloudy scenes in `window`.

    Returns an xarray.Dataset with float32 reflectance (0-1) for C.S2_BANDS,
    on a UTM grid at `resolution` metres. Pixels flagged as cloud, shadow,
    cirrus, snow or no-data by the Scene Classification Layer are ignored.
    """
    import odc.stac
    import planetary_computer
    import pystac_client

    lon_c, lat_c = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    epsg = crs_epsg or utm_epsg(lon_c, lat_c)

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    items = list(catalog.search(
        collections=["sentinel-2-l2a"], bbox=bbox, datetime=window,
        query={"eo:cloud_cover": {"lt": C.MAX_CLOUD_COVER}},
    ).items())
    if not items:
        raise RuntimeError(f"No Sentinel-2 scenes for {bbox} in {window}")
    items = sorted(items, key=lambda it: it.properties["eo:cloud_cover"])[: C.MAX_SCENES * 3]

    ds = odc.stac.load(
        items, bands=C.S2_BANDS + ["SCL"], bbox=bbox, crs=f"EPSG:{epsg}",
        resolution=resolution, chunks={"x": 1024, "y": 1024}, groupby="solar_day",
    )
    # keep the MAX_SCENES dates with the most valid pixels
    valid = ds["SCL"].isin(C.SCL_VALID)
    frac = valid.mean(dim=("x", "y")).compute()
    keep = frac.sortby(frac, ascending=False).time.values[: C.MAX_SCENES]
    ds, valid = ds.sel(time=sorted(keep)), valid.sel(time=sorted(keep))

    out = {}
    cutoff = np.datetime64(C.BASELINE_04_DATE)
    offset = (ds.time >= cutoff).astype("float32") * 1000.0  # processing baseline 04.00
    for b in C.S2_BANDS:
        dn = ds[b].astype("float32")
        dn = dn.where(dn > 0)                       # 0 = no data
        refl = ((dn - offset) / 10000.0).where(valid)
        out[b] = refl.median(dim="time", skipna=True)
    import xarray as xr
    comp = xr.Dataset(out).compute()
    comp.attrs["epsg"] = epsg
    comp.attrs["n_scenes"] = int(len(keep))
    return comp


def to_unit_range(arr: np.ndarray) -> np.ndarray:
    """(4, H, W) reflectance -> fixed [0, 1] scaling per band (see config.BAND_SCALE)."""
    scale = np.array([C.BAND_SCALE[b] for b in C.S2_BANDS], dtype=np.float32)[:, None, None]
    return np.clip(arr / scale, 0.0, 1.0)
