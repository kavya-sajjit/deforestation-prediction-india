"""PyTorch dataset over the saved patches.

Inputs (all in [0, 1], channel-first):
  imagery  : T0 bands chosen by `bands` (subset of red, green, blue, nir)
  temporal : for detection, the same bands at T1 are appended after T0
  history  : optional, for forecasting: past-loss mask (2001-2018) and a constant
             channel with the share of land cleared within 2.5 km
  ndvi     : returned separately (for the NDVI-fusion model), rescaled to [0, 1]
"""
from __future__ import annotations

import os
from typing import List, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .. import config as C
from .io import load_all, ndvi01  # noqa: F401  (re-exported)


class PatchDataset(Dataset):
    def __init__(self, meta: pd.DataFrame, arrays: dict, idx, task: str,
                 bands: List[str] = ("red", "green", "blue", "nir"),
                 history: bool = False, with_ndvi: bool = False, augment: bool = False,
                 band_fill: dict | None = None):
        assert task in ("forecast", "detect")
        self.meta, self.arrays, self.idx = meta, arrays, np.asarray(idx)
        self.task, self.augment, self.history, self.with_ndvi = task, augment, history, with_ndvi
        self.band_ix = [C.BAND_NAMES.index(b) for b in bands]
        self.label_col = "y_forecast" if task == "forecast" else "y_detect"
        self.band_fill = band_fill or {}     # {channel_index: value} for ablations

    @property
    def n_channels(self) -> int:
        n = len(self.band_ix) * (2 if self.task == "detect" else 1)
        return n + (2 if self.history else 0)

    def __len__(self):
        return len(self.idx)

    def labels(self) -> np.ndarray:
        return self.meta.loc[self.idx, self.label_col].to_numpy().astype(np.int64)

    def __getitem__(self, i):
        j = int(self.idx[i])
        t0 = self.arrays["t0"][j].astype(np.float32)
        x = [t0[self.band_ix]]
        if self.task == "detect":
            x.append(self.arrays["t1"][j].astype(np.float32)[self.band_ix])
        if self.history:
            h = self.arrays["loss_hist"][j].astype(np.float32)[None]
            d = np.full_like(h, min(1.0, float(self.meta.at[j, "hist_density"]) * 5))  # 20% cleared -> 1
            x += [h, d]
        x = np.concatenate(x, 0)
        for ch, val in self.band_fill.items():
            x[ch] = val
        mask = self.arrays["mask_target" if self.task == "forecast" else "mask_detect"][j].astype(np.float32)
        nd = ndvi01(t0[0], t0[3])[None] if self.with_ndvi else np.zeros((1, 1, 1), np.float32)

        if self.augment:
            k = np.random.randint(4)
            flip = np.random.rand() < 0.5
            def tf(a):
                a = np.rot90(a, k, axes=(-2, -1))
                return np.flip(a, -1) if flip else a
            x, mask = tf(x), tf(mask)
            if self.with_ndvi:
                nd = tf(nd)
        y = int(self.meta.at[j, self.label_col])
        return dict(x=torch.from_numpy(np.ascontiguousarray(x)),
                    ndvi=torch.from_numpy(np.ascontiguousarray(nd)),
                    mask=torch.from_numpy(np.ascontiguousarray(mask)),
                    y=torch.tensor(y), idx=torch.tensor(j))
