"""Spatial cross-validation.

Two schemes, both reported:
  * "block": 5.12 km blocks are assigned to N_FOLDS folds (within each region),
    and training patches that touch a test block are dropped (buffer). Patches
    never overlap, so no test pixel is ever seen in training.
  * "loro":  leave-one-region-out. Train on three biomes, test on the fourth.
    This is the harder, more honest test of whether the model transfers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation

from .. import config as C


def assign_block_folds(meta: pd.DataFrame, n_folds: int = C.N_FOLDS, seed: int = C.SEED) -> pd.Series:
    rng = np.random.default_rng(seed)
    fold = pd.Series(-1, index=meta.index, dtype=int)
    for region, g in meta.groupby("region"):
        blocks = g[["block_r", "block_c"]].drop_duplicates().to_numpy()
        order = rng.permutation(len(blocks))
        ids = {tuple(blocks[i]): k % n_folds for k, i in enumerate(order)}
        fold.loc[g.index] = [ids[(r, c)] for r, c in zip(g.block_r, g.block_c)]
    return fold


def _buffer_mask(meta: pd.DataFrame, test_idx: np.ndarray, buffer: int) -> np.ndarray:
    """Boolean over meta.index: True where a patch is within `buffer` patches of a test patch."""
    near = np.zeros(len(meta), dtype=bool)
    if buffer <= 0:
        return near
    pos = {i: k for k, i in enumerate(meta.index)}
    for region, g in meta.groupby("region"):
        H, W = g.row.max() + 1, g.col.max() + 1
        grid = np.zeros((H, W), dtype=bool)
        t = g.loc[g.index.intersection(test_idx)]
        grid[t.row.to_numpy(), t.col.to_numpy()] = True
        grid = binary_dilation(grid, iterations=buffer, structure=np.ones((3, 3)))
        hit = grid[g.row.to_numpy(), g.col.to_numpy()]
        near[[pos[i] for i in g.index]] = hit
    return near


def block_split(meta: pd.DataFrame, test_fold: int, label_col: str,
                n_folds: int = C.N_FOLDS, buffer: int = C.BUFFER_PATCHES):
    """Returns (train_idx, val_idx, test_idx) as arrays of meta.index labels."""
    fold = meta["fold"]
    usable = meta[label_col] >= 0
    test = meta.index[(fold == test_fold) & usable]
    val_fold = (test_fold + 1) % n_folds
    near_test = _buffer_mask(meta, test, buffer)
    rest = meta.index[(fold != test_fold) & usable & ~near_test]
    val = rest[meta.loc[rest, "fold"].to_numpy() == val_fold]
    near_val = _buffer_mask(meta, val, buffer)
    train = rest[(meta.loc[rest, "fold"].to_numpy() != val_fold) & ~near_val[[meta.index.get_loc(i) for i in rest]]]
    return np.asarray(train), np.asarray(val), np.asarray(test)


def loro_split(meta: pd.DataFrame, test_region: str, label_col: str, n_folds: int = C.N_FOLDS):
    usable = meta[label_col] >= 0
    test = meta.index[(meta.region == test_region) & usable]
    rest = meta[(meta.region != test_region) & usable]
    val_mask = rest["fold"] == 0                           # one block-fold of the training regions
    near_val = _buffer_mask(meta, rest.index[val_mask], C.BUFFER_PATCHES)
    near_val = near_val[[meta.index.get_loc(i) for i in rest.index]]
    train = rest.index[~val_mask.to_numpy() & ~near_val]
    return np.asarray(train), np.asarray(rest.index[val_mask]), np.asarray(test)


def splits(meta: pd.DataFrame, scheme: str, label_col: str):
    """Yield (split_name, train_idx, val_idx, test_idx)."""
    if scheme == "block":
        for k in range(C.N_FOLDS):
            yield (f"fold{k}",) + block_split(meta, k, label_col)
    elif scheme == "loro":
        for r in sorted(meta.region.unique()):
            yield (f"test_{r}",) + loro_split(meta, r, label_col)
    else:
        raise ValueError(scheme)


def check_no_leak(meta: pd.DataFrame, train_idx, test_idx) -> None:
    """Assert no training patch is the same as, or adjacent to, a test patch."""
    assert len(set(train_idx) & set(test_idx)) == 0
    near = _buffer_mask(meta, np.asarray(test_idx), 1)
    pos = {i: k for k, i in enumerate(meta.index)}
    assert not any(near[pos[i]] for i in train_idx), "training patch touches a test patch"
