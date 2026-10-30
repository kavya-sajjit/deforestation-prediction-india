"""Synthetic regions with the same format as the real dataset.

Only for tests and for checking the pipeline end to end without internet.
Never report numbers from synthetic data.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter

from .. import config as C
from . import hansen, patches


def _blobs(rng, shape, sigma, q):
    f = gaussian_filter(rng.standard_normal(shape), sigma)
    return f > np.quantile(f, q)


def make_region(name: str, size: int = 1536, seed: int = 0):
    rng = np.random.default_rng(seed)
    H = W = size
    tree = np.where(_blobs(rng, (H, W), 25, 0.35), rng.integers(50, 100, (H, W)), rng.integers(0, 25, (H, W)))
    lossyear = np.zeros((H, W), np.uint8)
    forest = tree >= C.TREECOVER_THRESHOLD
    hist = _blobs(rng, (H, W), 6, 0.975) & forest
    lossyear[hist] = rng.integers(5, 19, hist.sum())
    # future loss grows next to past loss (contagion) plus some random clearing
    frontier = binary_dilation(hist, iterations=8) & forest & ~hist
    future = (frontier & _blobs(rng, (H, W), 4, 0.8)) | (_blobs(rng, (H, W), 5, 0.997) & forest & ~hist)
    lossyear[future] = rng.integers(20, 24, future.sum())
    y19 = _blobs(rng, (H, W), 3, 0.995) & forest & ~hist & ~future
    lossyear[y19] = 19
    labels = hansen.label_rasters(tree.astype(np.uint8), lossyear)

    def scene(after: bool):
        img = np.empty((4, H, W), np.float32)
        veg = forest & ~hist
        if after:
            veg = veg & ~(future | y19)
        degraded = future if not after else np.zeros_like(future)   # thinning before clearing
        refl = {  # R, G, B, NIR reflectance
            "forest": (0.03, 0.06, 0.03, 0.30), "bare": (0.12, 0.10, 0.08, 0.20),
            "degraded": (0.05, 0.07, 0.04, 0.24),
        }
        for b in range(4):
            v = np.full((H, W), refl["bare"][b], np.float32)
            v[veg] = refl["forest"][b]
            v[degraded & veg] = refl["degraded"][b]
            img[b] = v * (1 + 0.15 * rng.standard_normal((H, W)))
        scale = np.array([C.BAND_SCALE[b] for b in C.S2_BANDS], np.float32)[:, None, None]
        img = np.clip(img / scale, 0, 1)
        img[:, _blobs(rng, (H, W), 8, 0.995)] = np.nan       # a little cloud
        return img

    t0, t1 = scene(False), scene(True)
    arrays, meta = patches.extract(name, t0, t1, labels)
    return arrays, meta, tree.astype(np.uint8), lossyear


def make_dataset(out_dir: str, size: int = 1536):
    for k, name in enumerate(C.REGIONS):
        arrays, meta, tree, lossyear = make_region(name, size, seed=k)
        patches.save_region(out_dir, name, arrays, meta,
                            dict(region=name, synthetic=True, shape=[size, size],
                                 bbox=list(C.REGIONS[name]["bbox"])))
        np.savez_compressed(f"{out_dir}/{name}_hansen.npz", treecover=tree, lossyear=lossyear)
