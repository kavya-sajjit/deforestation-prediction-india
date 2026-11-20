"""All-India forecasts: do factors beyond the forest itself help on states the model never saw?

Every forested 640 m square in India becomes one row. For a forecast made at the end of year T
(default 2019) the label is: did the square lose at least 2% of its remaining forest in T+1..T+4
(2020-2023)? Features are computed from data up to T only.

Three models, each tested leave-one-state-out (trained on every other state, scored on the held-out
one), so every number is "how well does this work in a state it has never seen":

  history_lr   logistic regression on the square's own loss history (the baseline from the
               regional study that image models failed to beat on unseen regions)
  gbm_history  gradient boosting on its own history plus its neighbourhood's history
  gbm_all      the same plus the non-forest factors: terrain, remoteness, population, night
               lights, fire, rainfall and protected-area status

The comparison gbm_all vs gbm_history, state by state, answers the question. A sign test over
states says whether "better in most states" could be luck.

It also checks the time direction the website needs (train on 2016-19 outcomes, test on 2020-23),
then trains on 2020-23 and writes a forward risk score for 2026-29 from data up to 2025, a trend
projection to 2036 per state (deforest.projection), and district totals for the website.

    python -m deforest.national.model --grid india_grid.npz --admin india_admin_codes.csv --out results/national
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd

PATCH_HA = 40.96
EVENT = 0.02
HORIZON = 4
FIRST_CONSISTENT = 2011            # Hansen annual loss is comparable from 2011 on

HIST = ["f2000", "remaining", "lost_since_2011", "last1", "last3", "last5"]
NEIGHBOUR = ["nb5_last3", "nb15_last3", "nb15_remaining"]
COV = ["elevation", "slope", "access_min", "pop", "lights", "fire_days", "rain_mm", "protected"]
FEATURES = {"history_lr": HIST, "gbm_history": HIST + NEIGHBOUR, "gbm_all": HIST + NEIGHBOUR + COV}


# --------------------------------------------------------------------------- features
def _year_sum(loss, years, a, b):
    """Share of each square lost in years a..b inclusive (b < a gives zero)."""
    i0, i1 = max(a, years[0]) - years[0] + 1, min(b, years[-1]) - years[0] + 1
    return loss[i0:i1 + 1].sum(axis=0) if i1 >= i0 else np.zeros(loss.shape[1:], np.float32)


def frame(grid: dict, T: int, horizon: int = HORIZON, labels: bool = True) -> pd.DataFrame:
    """One row per forested square in India, features as of the end of year T."""
    from scipy.ndimage import uniform_filter
    loss, years, cov, names = grid["loss"], grid["years"], grid["cov"], grid["cov_names"]
    if not FIRST_CONSISTENT <= T <= years[-1]:
        raise ValueError(f"T={T} outside {FIRST_CONSISTENT}..{years[-1]}")
    f = loss[0]
    remaining = np.clip(f - _year_sum(loss, years, years[0], T), 0, None)
    last3 = _year_sum(loss, years, T - 2, T)
    g = {
        "f2000": f, "remaining": remaining,
        "lost_since_2011": _year_sum(loss, years, FIRST_CONSISTENT, T) / np.maximum(f, 1e-6),
        "last1": _year_sum(loss, years, T, T), "last3": last3, "last5": _year_sum(loss, years, T - 4, T),
        "nb5_last3": uniform_filter(last3, 5), "nb15_last3": uniform_filter(last3, 15),
        "nb15_remaining": uniform_filter(remaining, 15),
    }
    for n in names:
        g[n] = cov[names.index(n)]
    keep = (g["state"] > 0) & (remaining >= 0.05)
    rr, cc = np.nonzero(keep)
    df = pd.DataFrame({k: v[keep] for k, v in g.items()})
    df["row"], df["col"] = rr, cc
    df["pop"], df["lights"] = np.log1p(np.clip(df["pop"], 0, None)), np.log1p(np.clip(df["lights"], 0, None))
    df[["state", "district"]] = df[["state", "district"]].astype(int)
    if labels:
        if T + horizon > years[-1]:
            raise ValueError(f"labels need data to {T + horizon}; observed only to {years[-1]}")
        future = _year_sum(loss, years, T + 1, T + horizon)[keep]
        df["future"] = future
        df["y"] = (future / df["remaining"] >= EVENT).astype(int)
    return df


# --------------------------------------------------------------------------- models
def _model(name: str, seed: int = 0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler
    if name == "history_lr":
        return make_pipeline(FunctionTransformer(np.log1p), StandardScaler(), LogisticRegression(max_iter=500))
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
                                          early_stopping=True, validation_fraction=0.1, random_state=seed)


def _sample(df: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    """Keep every positive and at most `cap` rows overall (rank metrics are unaffected)."""
    if len(df) <= cap:
        return df
    pos, neg = df[df.y == 1], df[df.y == 0]
    n = max(cap - len(pos), min(len(neg), cap // 2))
    return pd.concat([pos, neg.sample(n=min(n, len(neg)), random_state=seed)])


def _metrics(y, p, future) -> Dict[str, float]:
    from sklearn.metrics import average_precision_score
    top = p >= np.quantile(p, 0.9)
    return dict(ap=float(average_precision_score(y, p)), chance=float(y.mean()),
                top10_loss_share=float(future[top].sum() / max(future.sum(), 1e-12)))


def leave_one_state_out(df: pd.DataFrame, min_pos: int = 30, cap: int = 800_000, seed: int = 0) -> pd.DataFrame:
    rows = []
    states = [s for s, g in df.groupby("state") if g.y.sum() >= min_pos and (g.y == 0).sum() >= min_pos]
    for s in states:
        tr, te = _sample(df[df.state != s], cap, seed), df[df.state == s]
        for name, feats in FEATURES.items():
            m = _model(name, seed).fit(tr[feats], tr.y)
            p = m.predict_proba(te[feats])[:, 1]
            rows.append(dict(state=s, model=name, squares=len(te), positives=int(te.y.sum()),
                             **_metrics(te.y.to_numpy(), p, te.future.to_numpy())))
        print(f"  state {s}: " + ", ".join(f"{r['model']} {r['ap']:.3f}" for r in rows[-3:])
              + f" (chance {rows[-1]['chance']:.3f})", flush=True)
    return pd.DataFrame(rows)


def sign_test(res: pd.DataFrame, a: str, b: str) -> Dict[str, float]:
    """In how many held-out states does model a beat model b, and could that be chance?"""
    from scipy.stats import binomtest
    w = res.pivot(index="state", columns="model", values="ap")
    wins, n = int((w[a] > w[b]).sum()), int(w[[a, b]].notna().all(axis=1).sum())
    return dict(a=a, b=b, wins=wins, states=n, p_value=float(binomtest(wins, n, 0.5).pvalue) if n else float("nan"),
                median_ap_gain=float((w[a] - w[b]).median()))


def importance(df: pd.DataFrame, seed: int = 0, n_eval: int = 200_000) -> pd.DataFrame:
    """Permutation importance of gbm_all on a quarter of the states it did not train on."""
    from sklearn.inspection import permutation_importance
    rng = np.random.default_rng(seed)
    states = df.state.unique()
    held = set(rng.choice(states, max(1, len(states) // 4), replace=False))
    tr, te = _sample(df[~df.state.isin(held)], 800_000, seed), df[df.state.isin(held)]
    te = te.sample(n=min(n_eval, len(te)), random_state=seed)
    feats = FEATURES["gbm_all"]
    m = _model("gbm_all", seed).fit(tr[feats], tr.y)
    r = permutation_importance(m, te[feats], te.y, scoring="average_precision", n_repeats=3, random_state=seed)
    return pd.DataFrame(dict(feature=feats, ap_drop=r.importances_mean, sd=r.importances_std)) \
        .sort_values("ap_drop", ascending=False)


# --------------------------------------------------------------------------- forward
def forward(grid: dict, train_T: int, apply_T: int, seed: int = 0):
    """Fit gbm_all on outcomes after train_T, score squares as of apply_T. Returns (frame, AP or None)."""
    feats = FEATURES["gbm_all"]
    tr = _sample(frame(grid, train_T), 1_500_000, seed)
    m = _model("gbm_all", seed).fit(tr[feats], tr.y)
    years = grid["years"]
    has_labels = apply_T + HORIZON <= years[-1]
    ap_df = frame(grid, apply_T, labels=has_labels)
    ap_df["risk"] = m.predict_proba(ap_df[feats])[:, 1]
    score = _metrics(ap_df.y.to_numpy(), ap_df.risk.to_numpy(), ap_df.future.to_numpy()) if has_labels else None
    return ap_df, score


def projection_by_state(grid: dict, end: int = 2036) -> Dict[int, dict]:
    """Trend projection (deforest.projection), run separately for each state."""
    from .. import projection as P
    loss, years, cov, names = grid["loss"], grid["years"], grid["cov"], grid["cov_names"]
    state = cov[names.index("state")]
    inside = (state > 0) & (loss[0] > 0)
    out = {}
    for s in np.unique(state[inside]):
        m = inside & (state == s)
        f, L = loss[0][m].astype(float), loss[1:, m].T.astype(float)
        T = years[-1]
        K, sh = P.choose(f, L, years[0])
        pj = P.project(f, L, years[0], T, K, sh, end)
        bt = P.score(f, L, years[0], 2018, K, sh, T) if T > 2019 else None
        out[int(s)] = dict(mask=m, pj=pj, window=K, shrink=sh,
                           backtest=None if bt is None else {k: v for k, v in bt.items() if k not in ("mse", "by_year")})
    return out


def districts(grid: dict, risk: pd.DataFrame, proj: Dict[int, dict], admin: pd.DataFrame, end: int = 2036) -> list:
    """Per-district totals for the website, in one pass over the grid (np.bincount)."""
    loss, years, cov, names = grid["loss"], grid["years"], grid["cov"], grid["cov_names"]
    dist = cov[names.index("district")].astype(np.int64)
    inside = dist > 0
    codes, idx = np.unique(dist[inside], return_inverse=True)
    n = len(codes)
    tot = lambda w: np.bincount(idx, weights=w[inside].astype(np.float64), minlength=n) * PATCH_HA
    forest = tot(loss[0])
    lost = np.stack([tot(loss[i + 1]) for i in range(len(years))])
    pj_total = np.zeros((end - years[-1],) + dist.shape, np.float32)
    for d in proj.values():
        pj_total[:, d["mask"]] = d["pj"].T
    projected = np.stack([tot(pj_total[j]) for j in range(pj_total.shape[0])])
    rdist = dist[risk.row.to_numpy(), risk.col.to_numpy()]
    ok = rdist > 0
    ridx = np.searchsorted(codes, rdist[ok])
    rv = risk.risk.to_numpy()[ok]
    top = np.quantile(risk.risk.to_numpy(), 0.9)
    cnt = np.bincount(ridx, minlength=n)
    mean_risk = np.bincount(ridx, weights=rv, minlength=n) / np.maximum(cnt, 1)
    share_top = np.bincount(ridx, weights=(rv >= top).astype(float), minlength=n) / np.maximum(cnt, 1)
    names_by = admin.set_index("ADM2_CODE")[["ADM2_NAME", "ADM1_NAME"]].to_dict("index") if len(admin) else {}
    out = []
    for k, code in enumerate(codes):
        if forest[k] < 100:                                # districts with almost no forest
            continue
        nm = names_by.get(int(code), {})
        out.append(dict(
            code=int(code), name=nm.get("ADM2_NAME", str(code)), state=nm.get("ADM1_NAME", ""),
            forest_2000_ha=round(float(forest[k])), remaining_ha=round(float(max(forest[k] - lost[:, k].sum(), 0))),
            lost_ha={int(y): round(float(lost[i, k]), 1) for i, y in enumerate(years)},
            projected_extra_ha={int(years[-1] + 1 + j): round(float(projected[j, k]), 1) for j in range(projected.shape[0])},
            mean_risk=round(float(mean_risk[k]), 4) if cnt[k] else None,
            share_top10=round(float(share_top[k]), 4) if cnt[k] else None,
            forest_squares=int(cnt[k]),
        ))
    return out


# --------------------------------------------------------------------------- report
def run(grid_path: str, admin_csv: str, out: str, T: int = 2019, seed: int = 0, skip_loso: bool = False):
    from .table import load
    os.makedirs(out, exist_ok=True)
    grid = load(grid_path)
    admin = pd.read_csv(admin_csv) if admin_csv and os.path.exists(admin_csv) else pd.DataFrame()
    state_names = admin.drop_duplicates("ADM1_CODE").set_index("ADM1_CODE")["ADM1_NAME"].to_dict() if len(admin) else {}
    last = grid["years"][-1]
    L = [f"# All-India forecast (auto-generated by `python -m deforest.national.model`)\n",
         f"Hansen loss observed 2001-{last}; history features use {FIRST_CONSISTENT} onwards. Forecast made at the "
         f"end of {T}, label = lost at least {EVENT:.0%} of remaining forest in {T + 1}-{T + HORIZON}. One seed.\n"]

    df = frame(grid, T)
    L.append(f"{len(df):,} forested squares in {df.state.nunique()} states/UTs; "
             f"{df.y.mean():.1%} lost at least {EVENT:.0%} of their forest in {T + 1}-{T + HORIZON}.\n")
    print(L[-1])
    if not skip_loso:
        res = leave_one_state_out(df, seed=seed)
        res["state_name"] = res.state.map(state_names).fillna(res.state.astype(str))
        res.to_csv(os.path.join(out, "loso.csv"), index=False)
        w = res.pivot_table(index=["state_name", "positives", "chance"], columns="model", values="ap").reset_index()
        w = w.sort_values("positives", ascending=False)
        L.append("\n## Unseen states: average precision when the state was left out of training\n")
        L.append("| State | Loss squares | Chance | history_lr | gbm_history | gbm_all |")
        L.append("|---|---|---|---|---|---|")
        for _, r in w.iterrows():
            L.append(f"| {r.state_name} | {int(r.positives):,} | {r.chance:.3f} | {r.history_lr:.3f} | "
                     f"{r.gbm_history:.3f} | {r.gbm_all:.3f} |")
        for a, b in (("gbm_all", "gbm_history"), ("gbm_history", "history_lr")):
            t = sign_test(res, a, b)
            L.append(f"\n**{a} vs {b}:** better in {t['wins']} of {t['states']} unseen states, median AP change "
                     f"{t['median_ap_gain']:+.3f}, sign-test p = {t['p_value']:.3g}.")
        imp = importance(df, seed)
        imp.to_csv(os.path.join(out, "importance.csv"), index=False)
        L.append("\n## Which inputs gbm_all relies on (permutation importance, held-out states)\n")
        L.append("| Input | AP lost when shuffled |")
        L.append("|---|---|")
        for _, r in imp.iterrows():
            L.append(f"| {r.feature} | {r.ap_drop:.4f} ± {r.sd:.4f} |")
        L.append("\nImportance is association, not cause: a factor can matter because it travels with a real driver.")

    # forward in time: the check the website's 2026-29 map relies on
    if T - HORIZON >= FIRST_CONSISTENT + 4:
        _, tcheck = forward(grid, T - HORIZON, T, seed)
        L.append(f"\n## Forward in time\n\nTrained on outcomes {T - HORIZON + 1}-{T}, scored on {T + 1}-{T + HORIZON} "
                 f"for all of India: AP {tcheck['ap']:.3f} (chance {tcheck['chance']:.3f}); the riskiest 10% of squares "
                 f"held {tcheck['top10_loss_share']:.0%} of the real loss.")
    train_T = last - HORIZON
    risk, _ = forward(grid, train_T, last, seed)
    risk[["row", "col", "state", "district", "risk"]].round({"risk": 5}).to_csv(
        os.path.join(out, f"risk_{last + 1}_{last + HORIZON}.csv.gz"), index=False)
    L.append(f"\nRisk for {last + 1}-{last + HORIZON}: model trained on outcomes {train_T + 1}-{last}, applied to data up "
             f"to {last}. This is a forecast nobody can score yet.")

    proj = projection_by_state(grid)
    bt = pd.DataFrame([dict(state=state_names.get(s, s), **(d["backtest"] or {})) for s, d in proj.items()])
    bt.to_csv(os.path.join(out, "projection_backtest.csv"), index=False)
    if "ap" in bt:
        L.append(f"\n## Trend projection to 2036\n\nPer state, backtested from 2018 on 2019-{last}: ranking AP above "
                 f"chance in {int((bt.ap > bt.chance_ap).sum())} of {bt.ap.notna().sum()} states; median projected/actual "
                 f"regional total {bt.total_ratio.median():.2f}. Full table: projection_backtest.csv.")
    d = districts(grid, risk, proj, admin)
    with open(os.path.join(out, "districts.json"), "w") as fh:
        json.dump(dict(last_observed=last, risk_years=[last + 1, last + HORIZON], districts=d), fh, separators=(",", ":"))
    L.append(f"\n{len(d)} districts with forest written to districts.json for the website.")
    text = "\n".join(L)
    with open(os.path.join(out, "NATIONAL.md"), "w") as fh:
        fh.write(text)
    print(text)
    return text


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--grid", default="india_grid.npz")
    ap.add_argument("--admin", default="", help="india_admin_codes.csv from the Earth Engine export")
    ap.add_argument("--out", default="results/national")
    ap.add_argument("--origin", type=int, default=2019)
    ap.add_argument("--skip-loso", action="store_true", help="only the forward forecast, projection and districts")
    a = ap.parse_args()
    run(a.grid, a.admin, a.out, a.origin, skip_loso=a.skip_loso)
