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
    import pandas as pd
    pp = pd.read_csv(out / "forecast" / "block" / "bcos_rgbn_hist" / "fold0" / "explanations_per_patch.csv")
    shares = pp[[c for c in pp.columns if c.startswith("share_")]]
    assert shares.shape[1] == 6 and np.allclose(shares.sum(axis=1), 1, atol=1e-4)   # 4 bands + 2 history channels
    assert pp.loc[pp.y == 1, "deletion_gap"].notna().all() and pp.loc[pp.y == 0, "deletion_gap"].isna().all()
    # backfill: delete the per-patch file and rebuild it from the saved weights, without retraining
    from deforest import shift
    f = out / "forecast" / "block" / "bcos_pre_project" / "fold0" / "explanations_per_patch.csv"
    before = pd.read_csv(f); f.unlink()
    assert shift.backfill(data_dir, str(out)) == 1
    after = pd.read_csv(f)
    assert np.allclose(before["loc"].fillna(-1), after["loc"].fillna(-1), atol=1e-4)


def test_risk_calibration_and_hotspots():
    import pandas as pd
    from deforest import risk_levels as R
    rng = np.random.default_rng(0)
    n = 600
    p = rng.uniform(0, 1, n)
    y = (rng.uniform(0, 1, n) < p ** 2).astype(int)          # raw scores are over-confident
    oof = pd.DataFrame(dict(p=p, y=y, fold=[f"fold{i % 3}" for i in range(n)], region="r",
                            block_r=np.arange(n) // 20, block_c=0))
    oof["q"] = R.calibrate(oof)
    oof["tier"] = oof.q.map(R.tier_of)
    rep = R.report(oof, "toy")
    assert rep["ece_calibrated"] < rep["ece_raw"]              # calibration helps
    rates = {t["tier"]: t["hit_rate"] for t in rep["tiers"]}
    assert rates["high"] > rates["elevated"] > rates["low"]    # tiers are ordered by real outcome
    # hotspots: a 2x2 block of high squares plus one isolated square
    lon, lat = np.meshgrid(np.arange(6.0), np.arange(6.0)[::-1])
    cells = [dict(r=r, c=c, f=1.0, q=0.8, t="high") for r in (1, 2) for c in (1, 2)]
    cells += [dict(r=4, c=4, f=1.0, q=0.9, t="high"), dict(r=0, c=4, f=1.0, q=0.1, t="low")]
    spots = R.hotspots(cells, lon, lat, 5, 5)
    assert len(spots) == 1 and spots[0]["squares"] == 4
    assert all(c.get("h") == 1 for c in cells[:4]) and "h" not in cells[4]


def test_deletion_gap_separates_true_from_wrong_explanations():
    """A toy model that only looks at the top-left quarter of the image.

    An explanation pointing at that quarter must get a clearly positive deletion
    gap; one pointing at the opposite corner must not.
    """
    from deforest.shift import deletion_gap
    torch.manual_seed(0)
    x = torch.rand(8, 3, 16, 16) * 0.2 + 0.8                  # bright everywhere
    p_loss = lambda x, nd: torch.sigmoid(8 * (x[:, :2, :8, :8].mean(dim=(1, 2, 3)) - 0.5))
    right, wrong = torch.zeros(8, 16, 16), torch.zeros(8, 16, 16)
    right[:, :8, :8], wrong[:, 8:, 8:] = 1.0, 1.0
    fill = torch.zeros(3)
    g_right = deletion_gap(p_loss, x, right, fill, n_img=2)
    g_wrong = deletion_gap(p_loss, x, wrong, fill, n_img=2)
    assert g_right["gap"].mean() > 0.2
    assert g_wrong["gap"].mean() < 0
    assert np.all(np.diff(g_right["attr"], axis=1) <= 1e-6)    # removing more evidence never raises P(loss)
    x3 = x.clone(); x3[:, 2] = 0.123                           # channel 2 is not imagery: must be left alone
    assert torch.allclose(p_loss(x3, None), p_loss(x, None))


def test_shift_report_compares_the_same_patches(tmp_path):
    """Hand-made results: the model ranks perfectly in-distribution and at chance on the unseen region."""
    import pandas as pd
    from deforest import shift
    rng = np.random.default_rng(0)
    n = 400
    base = pd.DataFrame(dict(idx=np.arange(n), region="western_ghats", block_r=np.arange(n) // 10, block_c=0,
                             y=(np.arange(n) % 4 == 0).astype(int), split="test", threshold=0.5))
    expl = base[base.y == 1][["idx", "region", "block_r", "block_c", "y"]].assign(chance=0.05)

    def write(scheme, run, p, loc, gap, shares, rows=slice(None)):
        d = tmp_path / "forecast" / scheme / "bcos_rgbn_hist" / run
        d.mkdir(parents=True)
        base.assign(p=p).iloc[rows].to_csv(d / "preds.csv", index=False)
        e = expl.assign(loc=loc, deletion_gap=gap, p=0.5)
        for k, v in shares.items():
            e[f"share_{k}"] = v
        e[e.idx.isin(base.iloc[rows].idx)].to_csv(d / "explanations_per_patch.csv", index=False)

    good = base.y * 0.8 + 0.1
    half = slice(0, n // 2)
    write("block", "fold0", good, 0.50, 0.30, dict(red=0.5, nir=0.5), half)       # only half the region has an ID run
    write("loro", "test_western_ghats", rng.uniform(0, 1, n), 0.05, 0.0, dict(red=0.9, nir=0.1))
    text = shift.report(str(tmp_path))
    r = pd.read_csv(tmp_path / "shift.csv").iloc[0]
    assert r.patches == n // 2                                   # compared only where both runs scored the patch
    assert r.ap_in == 1.0 and r.ap_out < 0.6 and r.ap_drop_lo > 0
    assert abs(r.lift_in - 10) < 1e-6 and abs(r.lift_out - 1) < 1e-6
    assert abs(r.lift_retention) < 1e-6 and abs(r.del_retention) < 1e-6
    assert r.expl_positives == 50 and "explained" in text
    assert 0.2 < r.reliance_js < 0.5
    assert "1 of 5 spatial folds" in text
    assert (tmp_path / "shift_forecast.png").exists()


def test_no_retention_ratio_when_in_distribution_is_near_chance():
    """A ratio of two near-chance numbers is noise and must not be printed."""
    from deforest.shift import _retention
    assert np.isnan(_retention(0.014, 0.012, 0.012, 0.05, 100))        # AP barely above chance
    assert np.isnan(_retention(1.11, 2.00, 1.0, 0.5, 100))             # localisation 1.11x: was "894% kept"
    assert np.isnan(_retention(0.9, 0.5, 0.2, 0.05, 16))               # too few loss patches
    assert abs(_retention(0.6, 0.4, 0.2, 0.05, 100) - 0.5) < 1e-9
