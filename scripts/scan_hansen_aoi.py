"""Find ~30 km boxes with enough forest loss to train on.

    python scripts/scan_hansen_aoi.py --bbox 85.0 21.5 86.0 22.5 --size 0.3

Reads Hansen GFC (treecover2000, lossyear) at native 30 m for the search box
and ranks candidate windows by the number of forest pixels lost in 2020-2023.
Choosing boxes this way enriches positives; say so in the README.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deforest import config as C            # noqa: E402
from deforest.data.hansen import tiles_for_bbox  # noqa: E402


def read(layer, bbox):
    import rasterio
    from rasterio.windows import from_bounds
    out = []
    for tile in tiles_for_bbox(bbox):
        url = C.HANSEN_URL.format(v=C.HANSEN_VERSION, layer=layer, tile=tile)
        with rasterio.open(f"/vsicurl/{url}") as src:
            win = from_bounds(*bbox, transform=src.transform).round_offsets().round_lengths()
            out.append(src.read(1, window=win, boundless=True, fill_value=0))
    if len(out) > 1:
        raise SystemExit("search box crosses a Hansen tile edge; split it in two")
    return out[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("LON0", "LAT0", "LON1", "LAT1"))
    ap.add_argument("--size", type=float, default=0.3, help="window size in degrees")
    ap.add_argument("--top", type=int, default=5)
    a = ap.parse_args()
    tc, ly = read("treecover2000", a.bbox), read("lossyear", a.bbox)
    forest = tc >= C.TREECOVER_THRESHOLD
    target = np.isin(ly, list(C.TARGET_LOSS_YEARS)) & forest
    px = 1 / 4000                                  # Hansen pixel = 0.00025 deg
    w = int(a.size / px); step = w // 4
    res = []
    for r in range(0, target.shape[0] - w + 1, step):
        for c in range(0, target.shape[1] - w + 1, step):
            lost = int(target[r:r + w, c:c + w].sum())
            ff = float(forest[r:r + w, c:c + w].mean())
            lon0 = a.bbox[0] + c * px; lat1 = a.bbox[3] - r * px
            res.append((lost, ff, (round(lon0, 3), round(lat1 - a.size, 3), round(lon0 + a.size, 3), round(lat1, 3))))
    for lost, ff, box in sorted(res, reverse=True)[: a.top]:
        print(f"bbox={box}  forest share={ff:.2f}  forest pixels lost 2020-23={lost} (~{lost * 0.09:.0f} ha)")


if __name__ == "__main__":
    main()
