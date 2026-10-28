"""Run with:  pytest -q tests/

These tests use synthetic data, so they run offline in about a minute on CPU.
They check the properties the README claims: no spatial leakage, exact B-cos
explanations for every variant, and a pipeline that runs end to end.
"""
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from deforest import config as C  # noqa: E402
from deforest.data import hansen, splits, synthetic  # noqa: E402
from deforest.data.io import load_all  # noqa: E402


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("synthetic")
    synthetic.make_dataset(str(d), size=1536)
    return str(d)


def test_hansen_tile_names():
    assert hansen.tiles_for_bbox((75.75, 11.85, 76.05, 12.15)) == ["20N_070E"]
    assert hansen.tiles_for_bbox((92.6, 22.8, 92.9, 23.1)) == ["30N_090E"]
    assert set(hansen.tiles_for_bbox((79.9, 19.9, 80.1, 20.1))) == {"30N_070E", "30N_080E", "20N_070E", "20N_080E"}


def test_label_rules():
    tc = np.array([[80, 80, 80, 80, 10]], np.uint8)
    ly = np.array([[0, 10, 19, 21, 21]], np.uint8)       # none, 2010, 2019, 2021, 2021-non-forest
    lab = hansen.label_rasters(tc, ly)
    assert lab["loss_hist"].tolist() == [[0, 1, 0, 0, 0]]
    assert lab["forest_t0"].tolist() == [[1, 0, 1, 1, 0]]
    assert lab["loss_target"].tolist() == [[0, 0, 0, 1, 0]]
    assert lab["loss_2019"].tolist() == [[0, 0, 1, 0, 0]]
    assert lab["loss_detect"].tolist() == [[0, 0, 1, 1, 0]]


def test_patches_do_not_overlap_and_splits_do_not_leak(data_dir):
    meta, arrays = load_all(data_dir)
    assert not meta.duplicated(["region", "row", "col"]).any()
    meta["fold"] = splits.assign_block_folds(meta)
    for name, tr, va, te in splits.splits(meta, "block", "y_forecast"):
        splits.check_no_leak(meta, tr, te)
        assert not set(tr) & set(va)
        assert (meta.loc[te, "y_forecast"] >= 0).all()
    for name, tr, va, te in splits.splits(meta, "loro", "y_forecast"):
        assert set(meta.loc[tr, "region"]).isdisjoint(set(meta.loc[te, "region"]))


torch = pytest.importorskip("torch")
pytest.importorskip("bcos")


@pytest.mark.parametrize("variant,n", [("bcos", 4), ("bcos", 10), ("bcos_pre_rgb", 3),
                                       ("bcos_pre_inflate", 4), ("bcos_pre_project", 4),
                                       ("bcos_pre_ndvi", 3)])
def test_bcos_explanations_are_exact(variant, n):
    from deforest.models.zoo import build_model
    torch.manual_seed(0)
    m = build_model(variant, n, download=False).eval()
    size = 64 if variant == "bcos" else 128
    x = torch.rand(3, n, size, size)
    nd = torch.rand(3, 1, size, size) if variant == "bcos_pre_ndvi" else None
    r = m.contributions(x, nd)
    scale = r["logit_diff"].abs().clamp(min=1e-3)
    assert (r["completeness_error"] / scale).max() < 1e-3, r["completeness_error"]


def test_projection_commutes_with_inverse_encoding():
    from deforest.models.zoo import SumToOneProjection
    p = SumToOneProjection(4, 3)
    with torch.no_grad():
        p.delta.normal_()
    x = torch.rand(2, 4, 8, 8)
    y = p(torch.cat([x, 1 - x], 1))
    assert torch.allclose(y[:, 3:], 1 - y[:, :3], atol=1e-5)


def test_end_to_end_quick(data_dir, tmp_path):
    out = tmp_path / "results"
    cmd = [sys.executable, "-m", "deforest.experiments", "--data", data_dir, "--out", str(out),
           "--profile", "quick", "--folds", "fold0", "--no-download", "--workers", "0",
           "--only", "history_lr", "gbm", "cnn_rgbn", "bcos_rgbn_hist", "bcos_pre_project",
           "cnn_bitemporal", "bcos_bitemporal"]
    subprocess.run(cmd, check=True, cwd=ROOT)
    assert (out / "SUMMARY.md").exists()
    assert (out / "forecast" / "block" / "bcos_rgbn_hist" / "fold0" / "explanations.json").exists()
    assert (out / "forecast" / "block" / "bcos_rgbn_hist" / "fold0" / "band_ablation.csv").exists()
