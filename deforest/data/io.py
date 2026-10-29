"""Torch-free loading helpers (used by baselines, summaries and the dataset)."""
from __future__ import annotations

import os
from typing import Sequence

import numpy as np
import pandas as pd

from .. import config as C


def load_all(data_dir: str, regions: Sequence[str] | None = None):
    regions = regions or [r for r in C.REGIONS if os.path.exists(os.path.join(data_dir, f"{r}.csv"))]
    metas, arrs = [], {k: [] for k in ("t0", "t1", "mask_target", "mask_detect", "loss_hist")}
    for r in regions:
        m = pd.read_csv(os.path.join(data_dir, f"{r}.csv"))
        z = np.load(os.path.join(data_dir, f"{r}.npz"))
        metas.append(m)
        for k in arrs:
            arrs[k].append(z[k])
    meta = pd.concat(metas, ignore_index=True)
    arrays = {k: np.concatenate(v) for k, v in arrs.items()}
    return meta, arrays


def ndvi01(red: np.ndarray, nir: np.ndarray, scale_red=C.BAND_SCALE["B04"], scale_nir=C.BAND_SCALE["B08"]):
    """NDVI from [0,1]-scaled bands, mapped from [-1, 1] to [0, 1]."""
    r, n = red * scale_red, nir * scale_nir
    v = (n - r) / (n + r + 1e-6)
    return ((np.clip(v, -1, 1) + 1) / 2).astype(np.float32)
