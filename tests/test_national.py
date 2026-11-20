"""All-India pipeline on a small synthetic grid that mimics the Earth Engine exports."""
import json
import os

import numpy as np
import pandas as pd
import pytest

rasterio = pytest.importorskip("rasterio")


def _write(path, arr, names, transform):
    from rasterio.crs import CRS
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[1], width=arr.shape[2], count=arr.shape[0],
                       dtype=arr.dtype, crs=CRS.from_epsg(6933), transform=transform) as dst:
        dst.write(arr)
        for i, n in enumerate(names, 1):
            dst.set_band_description(i, n)


def make_exports(folder, H=160, W=160, last=2025, seed=0):
    """Four 'states' in quadrants. Loss is clustered and more likely near cities and outside
    protected areas, so the covariates carry real signal the history alone does not."""
    from rasterio.transform import from_origin
    from scipy.ndimage import uniform_filter
    rng = np.random.default_rng(seed)
    years = list(range(2001, last + 1))
    state = np.zeros((H, W), np.float32)
    state[:H // 2, :W // 2], state[:H // 2, W // 2:], state[H // 2:, :W // 2], state[H // 2:, W // 2:] = 11, 12, 13, 14
    state[:3, :] = 0                                                   # a strip outside India
    district = state * 100 + (np.arange(W)[None, :] // 40)
    access = uniform_filter(rng.uniform(0, 600, (H, W)), 9).astype(np.float32)
    protected = (uniform_filter(rng.uniform(0, 1, (H, W)), 15) > 0.52).astype(np.float32)
    f = np.clip(uniform_filter(rng.uniform(0, 1, (H, W)), 5) * 1.6, 0, 1).astype(np.float32)
    hot = uniform_filter((rng.uniform(0, 1, (H, W)) > 0.97).astype(float), 11)
    base = 0.0004 + 0.012 * (hot / hot.max()) ** 2 + 0.004 * (access < np.quantile(access, 0.3)) - 0.003 * protected
    base = np.clip(base, 0.0001, None)
    rem, loss = f.copy(), []
    for y in years:
        rate = base * (1.0 if y < 2011 else 2.0) * rng.uniform(0.5, 1.5, (H, W))
        l = (rem * np.clip(rate, 0, 0.5)).astype(np.float32)
        rem -= l
        loss.append(l)
    lstack = (np.stack([f] + loss) * 10000).round().astype(np.int16)
    names = ["f2000"] + [f"l{y}" for y in years]
    cov = np.stack([state, district, rng.uniform(0, 900, (H, W)), rng.uniform(0, 30, (H, W)), access,
                    rng.uniform(0, 400, (H, W)), rng.uniform(0, 5, (H, W)), rng.poisson(1, (H, W)),
                    rng.uniform(800, 3000, (H, W)), protected]).astype(np.float32)
    cnames = ["state", "district", "elevation", "slope", "access_min", "pop", "lights", "fire_days", "rain_mm", "protected"]
    t = from_origin(7_000_000, 3_000_000, 640, 640)
    # loss split into two shards like a big Earth Engine export
    _write(os.path.join(folder, "india_loss_640m-0000000000-0000000000.tif"), lstack[:, :H // 2], names, t)
    _write(os.path.join(folder, "india_loss_640m-0000080000-0000000000.tif"), lstack[:, H // 2:], names,
           from_origin(7_000_000, 3_000_000 - 640 * (H // 2), 640, 640))
    _write(os.path.join(folder, "india_cov_640m.tif"), cov, cnames, t)
    codes = sorted({int(d) for d in np.unique(district) if d > 0})
    pd.DataFrame(dict(ADM0_NAME="India", ADM1_CODE=[c // 100 for c in codes], ADM1_NAME=[f"State {c // 100}" for c in codes],
                      ADM2_CODE=codes, ADM2_NAME=[f"District {c}" for c in codes])).to_csv(
        os.path.join(folder, "india_admin_codes.csv"), index=False)
    return lstack, cov


def test_table_and_model_end_to_end(tmp_path):
    from deforest.national import model, table
    src = tmp_path / "drive"
    src.mkdir()
    lstack, cov = make_exports(str(src))
    info = table.build(str(src), str(tmp_path / "grid.npz"))
    g = table.load(str(tmp_path / "grid.npz"))
    assert g["loss"].shape == (26, 160, 160) and g["years"][-1] == 2025
    assert np.allclose(g["loss"], lstack / 10000.0, atol=1e-6)            # shards stitched in the right place
    assert info["squares_in_india"] == int((cov[0] > 0).sum())

    df = model.frame(g, 2019)
    assert set(model.FEATURES["gbm_all"]) <= set(df.columns) and (df.state > 0).all()
    assert df.remaining.min() >= 0.05 and 0 < df.y.mean() < 1
    # features use nothing after 2019: changing 2020+ loss must not change them
    g2 = dict(g, loss=g["loss"].copy())
    g2["loss"][20:] = 0
    df2 = model.frame(g2, 2019, labels=False)
    assert np.allclose(df[model.FEATURES["gbm_all"]].to_numpy(), df2[model.FEATURES["gbm_all"]].to_numpy())

    out = tmp_path / "out"
    text = model.run(str(tmp_path / "grid.npz"), str(src / "india_admin_codes.csv"), str(out))
    loso = pd.read_csv(out / "loso.csv")
    assert set(loso.model) == set(model.FEATURES) and loso.state.nunique() == 4
    w = loso.pivot(index="state", columns="model", values="ap")
    chance = loso.groupby("state").chance.first()
    assert (w.gbm_all > chance).all()
    assert w.gbm_all.mean() >= w.gbm_history.mean() - 0.01             # covariates must not make it worse
    assert "State 11" in text and "Forward in time" in text
    risk = pd.read_csv(out / "risk_2026_2029.csv.gz")
    assert risk.risk.between(0, 1).all() and len(risk) > 1000
    d = json.load(open(out / "districts.json"))
    assert d["last_observed"] == 2025 and len(d["districts"]) >= 8
    one = d["districts"][0]
    assert list(one["projected_extra_ha"])[0] == "2026" and list(one["projected_extra_ha"])[-1] == "2036"
    assert one["name"].startswith("District") and one["forest_2000_ha"] > 0
