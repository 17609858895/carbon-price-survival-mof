"""Exp. 13: FDR-controlled carbon-price shortlists by conformal selection on the RIV scale.

Viable at price p  <=>  y = 1/nCAC >= 1/p, so a single RIV model gives a selection threshold c = 1/p for every price.
For each scenario and price, the null for MOF j is H_j: y_j < c. Conformal p-values (clipped score of Jin & Candes 2023,
JMLR 24:244) are computed from a calibration fold of MOFs that the model never saw, then Benjamini-Hochberg at level q:
    p_j = (1 + #{i in calib : y_i < c, s_i >= s_j}) / (n_calib + 1),    s = (mu - c) / sigma.
MOF folds are hash-assigned to separate identities. Theoretical selection control additionally requires within-scenario exchangeability and an independently fixed prediction rule. The model specification was selected using this library, so reported workflow errors are empirical.
Rotations: test fold r, calibration fold r+1, training on the other three folds (RIV fit of Exp. 6).
Comparators: plug-in Bayesian FDR (largest set by P(viable) with mean(1 - P) <= q) and the point rule mu >= c.
Scale misspecification: the plug-in rule is also run with sigma x 0.5 (over-confident), sigma x 2 (under-confident) and a
homoscedastic scale (one residual s.d. from the calibration fold); conformal selection is invariant to a common rescaling
of sigma and is also run with the mean-only score s = mu - c. Predictions are saved for post-hoc analysis.
The mean model follows PRISMA_RIV_MEAN (see 07_prospective_atlas.py).
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(__file__))
from common import RES, SCEN_CAT, add_physics_proxies, fold_of, load_long  # noqa: E402

warnings.filterwarnings("ignore")
_s = importlib.util.spec_from_file_location("atlas", os.path.join(os.path.dirname(__file__), "07_prospective_atlas.py"))
atlas = importlib.util.module_from_spec(_s)
_s.loader.exec_module(atlas)

PRICES = [60.0, 100.0, 150.0]
QS = [0.05, 0.10, 0.20]


def bh(p, q):
    n = len(p)
    o = np.argsort(p)
    ok = p[o] <= q * np.arange(1, n + 1) / n
    sel = np.zeros(n, bool)
    if ok.any():
        sel[o[: np.max(np.nonzero(ok)[0]) + 1]] = True
    return sel


def conformal_select(s_cal, null_cal, s_te, q):
    sc = np.sort(s_cal[null_cal])
    cnt = len(sc) - np.searchsorted(sc, s_te, side="left")       # calibration nulls with score >= s_j
    return bh((1 + cnt) / (len(s_cal) + 1), q)


def bayes_fdr_select(pv, q):
    o = np.argsort(-pv)
    fdr = np.cumsum(1 - pv[o]) / np.arange(1, len(pv) + 1)
    sel = np.zeros(len(pv), bool)
    ok = np.nonzero(fdr <= q)[0]
    if len(ok):
        sel[o[: ok.max() + 1]] = True
    return sel


def evaluate(cal, te, rows, tag):
    for s, gt in te.groupby("scenario"):
        gc = cal[cal.scenario == s]
        if len(gc) < 20:
            continue
        s_homo = np.std(gc.y_inv.values - gc.mu.values)          # misspecified (homoscedastic) scale
        mg = (gt.MOF == "MgMOF74").values
        for p in PRICES:
            c = 1.0 / p
            yt = gt.y_inv.values >= c
            null_c = gc.y_inv.values < c
            rules = []
            for q in QS:
                rules.append(("conformal", "hetero", q, conformal_select((gc.mu.values - c) / gc.sigma.values, null_c,
                                                                         (gt.mu.values - c) / gt.sigma.values, q)))
                rules.append(("conformal", "mean_only", q, conformal_select(gc.mu.values - c, null_c, gt.mu.values - c, q)))
            for scale, st_ in [("hetero", gt.sigma.values), ("hetero_x0.5", 0.5 * gt.sigma.values),
                               ("hetero_x2", 2.0 * gt.sigma.values), ("homo", s_homo)]:
                pv = 1 - norm.cdf((c - gt.mu.values) / st_)
                for q in QS:
                    rules.append(("bayes_plugin", scale, q, bayes_fdr_select(pv, q)))
            rules.append(("point", "none", np.nan, gt.mu.values >= c))
            for meth, scale, q, sel in rules:
                ns = sel.sum()
                rows.append(dict(set=tag, scenario=s, price=p, q=q, method=meth, scale=scale, n_test=len(gt),
                                 n_true=int(yt.sum()), n_sel=int(ns), fdp=float((~yt[sel]).mean()) if ns else 0.0,
                                 power=float(yt[sel].sum() / yt.sum()) if yt.sum() else np.nan,
                                 mg74_sel=bool(sel[mg].any()) if mg.any() else np.nan,
                                 mg74_viable=bool(yt[mg].any()) if mg.any() else np.nan))


if __name__ == "__main__":
    t0 = time.time()
    k = add_physics_proxies(load_long().reset_index(drop=True))
    feats = atlas.riv_features(k)
    levels = {c: sorted(k[c].unique()) for c in SCEN_CAT}
    X = atlas.design(k, feats, levels)
    k["fold"] = k.MOF.map(fold_of)
    rows, preds = [], []
    for r in range(5):
        te, ca = k.fold.values == r, k.fold.values == (r + 1) % 5
        tr = ~(te | ca)
        mods = atlas.fit_riv(X[tr], k.y_inv.values[tr], k.MOF.values[tr])
        cal = k.loc[ca, ["MOF", "scenario", "y_inv"]].copy()
        tst = k.loc[te, ["MOF", "scenario", "y_inv"]].copy()
        cal["mu"], cal["sigma"] = atlas.predict_riv(mods, X[ca])
        tst["mu"], tst["sigma"] = atlas.predict_riv(mods, X[te])
        evaluate(cal, tst, rows, f"rot{r}")
        preds += [cal.assign(rotation=r, role="calib"), tst.assign(rotation=r, role="test")]
        print("rotation", r, round(time.time() - t0), "s", flush=True)
    res = pd.DataFrame(rows)
    tag = atlas.RIV_MEAN.replace("+", "_") + ("" if os.environ.get("PRISMA_FULL", "0") == "1" else "_fast")
    res.to_csv(os.path.join(RES, f"exp13_conformal_selection_{tag}.csv"), index=False)
    pd.concat(preds).to_parquet(os.path.join(RES, f"exp13_preds_{tag}.parquet"), index=False)
    summ = res.groupby(["method", "scale", "q", "price"], dropna=False).agg(
        FDR=("fdp", "mean"), FDR_q90=("fdp", lambda x: np.quantile(x, 0.9)), power=("power", "mean"),
        n_sel=("n_sel", "mean"), n_true=("n_true", "mean"), mg74=("mg74_sel", "mean")).reset_index()
    summ.to_csv(os.path.join(RES, f"exp13_conformal_selection_{tag}_summary.csv"), index=False)
    print(summ.round(3).to_string())
    print("done", round(time.time() - t0), "s")
