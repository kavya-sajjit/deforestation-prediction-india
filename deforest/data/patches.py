"""Turn region rasters into non-overlapping, labelled patches.

Used by both the real data builder and the synthetic test generator, so the
labelling rules are identical in tests and in the reported experiments.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter

from .. import config as C


def _past_loss_density(loss_hist: np.ndarray, pixel_m: float = 10.0) -> np.ndarray:
    """Share of land that lost forest before 2019 within DENSITY_RADIUS_M (100 m grid)."""
    f = max(1, int(round(100 / pixel_m)))
    H, W = loss_hist.shape
    h, w = H // f, W // f
    coarse = loss_hist[: h * f, : w * f].reshape(h, f, w, f).mean(axis=(1, 3))
    size = 2 * int(C.DENSITY_RADIUS_M / 100) + 1
    return uniform_filter(coarse.astype(np.float32), size=size, mode="nearest"), f


def extract(region: str, t0: np.ndarray, t1: np.ndarray, labels: dict,
            pixel_m: float = 10.0):
    """t0, t1: (4, H, W) float32 in [0, 1] (NaN = no valid observation).

    Returns (arrays dict, metadata DataFrame). Every patch that is at least
    MIN_FOREST_FRAC forest at T0 and MIN_VALID_FRAC cloud-free is kept; its
    forecast and detection labels are 1, 0 or -1 (ambiguous, excluded).
    """
    P = C.PATCH
    _, H, W = t0.shape
    dens, f = _past_loss_density(labels["loss_hist"], pixel_m)
    rows, cols = H // P, W // P

    keep_t0, keep_t1, keep_mt, keep_md, keep_hist, meta = [], [], [], [], [], []
    for r in range(rows):
        for c in range(cols):
            sl = (slice(r * P, (r + 1) * P), slice(c * P, (c + 1) * P))
            a0, a1 = t0[(slice(None),) + sl], t1[(slice(None),) + sl]
            valid = np.isfinite(a0).all(0) & np.isfinite(a1).all(0)
            forest = labels["forest_t0"][sl].mean()
            if valid.mean() < C.MIN_VALID_FRAC or forest < C.MIN_FOREST_FRAC:
                continue
            tgt = labels["loss_target"][sl].mean()
            det = labels["loss_detect"][sl].mean()
            amb = labels["loss_2019"][sl].mean()

            def lab(frac):
                if frac >= C.POS_LOSS_FRAC:
                    return 1
                if frac <= C.NEG_LOSS_FRAC:
                    return 0
                return -1
            y_fc = lab(tgt) if amb <= C.MAX_AMBIGUOUS_FRAC else -1
            y_det = lab(det)

            # fill the few invalid pixels with the patch median (per band)
            def fill(a):
                a = a.copy()
                for b in range(a.shape[0]):
                    m = np.nanmedian(a[b]) if np.isfinite(a[b]).any() else 0.0
                    a[b][~np.isfinite(a[b])] = m
                return a
            keep_t0.append(fill(a0).astype(np.float16))
            keep_t1.append(fill(a1).astype(np.float16))
            keep_mt.append(labels["loss_target"][sl].astype(np.uint8))
            keep_md.append(labels["loss_detect"][sl].astype(np.uint8))
            keep_hist.append(labels["loss_hist"][sl].astype(np.uint8))
            cy, cx = min((r * P + P // 2) // f, dens.shape[0] - 1), min((c * P + P // 2) // f, dens.shape[1] - 1)
            meta.append(dict(
                region=region, row=r, col=c,
                block_r=r // C.BLOCK_PATCHES, block_c=c // C.BLOCK_PATCHES,
                forest_frac=float(forest), hist_loss_frac=float(labels["loss_hist"][sl].mean()),
                hist_density=float(dens[cy, cx]), loss2019_frac=float(amb),
                target_frac=float(tgt), detect_frac=float(det),
                y_forecast=y_fc, y_detect=y_det,
            ))
    if not meta:
        raise RuntimeError(f"{region}: no usable patches; check the bbox and cloud cover")
    arrays = dict(
        t0=np.stack(keep_t0), t1=np.stack(keep_t1),
        mask_target=np.stack(keep_mt), mask_detect=np.stack(keep_md),
        loss_hist=np.stack(keep_hist),
    )
    return arrays, pd.DataFrame(meta)


def save_region(out_dir, region, arrays, meta, extra=None):
    import json
    import os
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, f"{region}.npz"), **arrays)
    meta.to_csv(os.path.join(out_dir, f"{region}.csv"), index=False)
    if extra:
        with open(os.path.join(out_dir, f"{region}.json"), "w") as fh:
            json.dump(extra, fh, indent=2)
