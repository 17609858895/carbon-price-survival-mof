"""Exp. 14: break-even and entry carbon prices read from the RIV predictive distribution.

Uses the out-of-fold RIV predictions of Exp. 6 (final mean model, MOF-grouped folds); nothing is refitted.

1. Break-even price of a pair: p* = nCAC = 1/y when y > 0, never viable when y <= 0. The RIV distribution gives a
   median p* = 1/mu and a 90% interval [1/(mu + z sigma), 1/(mu - z sigma)] (upper end infinite when mu - z sigma <= 0).
   Because p* is a monotone map of y on y > 0, the interval covers the simulated p* exactly when the y interval
   covers y.
2. Never-viable (cure) share per scenario: simulated share of y <= 0 vs predicted mean F(0 | x) = Phi(-mu / sigma).
3. Price at which k% of the MOF library becomes viable (k = 1, 5, 10): simulated from y, predicted by solving
   mean_i S(p | x_i) = k / 100 on a fine price grid (S(p | x) = 1 - Phi((1/p - mu) / sigma)).
4. Entry price per scenario: lowest simulated nCAC among viable MOFs, and the RIV estimate 1/max(mu).
5. Humidity penalty in price terms: wet minus dry library price for scenarios matched on everything else.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(__file__))
from common import RES  # noqa: E402

Z90 = norm.ppf(0.95)
GRID = np.geomspace(20, 5000, 1200)
KS = (1, 5, 10)
PMAX = 1000.0


def library_price(y, mu, sig, k):
    """Simulated and predicted carbon price at which k% of the library is viable (nan above PMAX)."""
    q = np.quantile(y, 1 - k / 100)
    sim = 1 / q if q > 0 and 1 / q <= PMAX else np.nan
    share = np.array([norm.sf((1 / p - mu) / sig).mean() for p in GRID])
    hit = np.nonzero(share >= k / 100)[0]
    pred = GRID[hit[0]] if len(hit) and GRID[hit[0]] <= PMAX else np.nan
    return sim, pred


if __name__ == "__main__":
    o = pd.read_parquet(os.path.join(RES, "exp6_riv_oof.parquet"))
    y, mu, sig = o.y_inv.values, o.mu.values, o.sigma.values
    lo_y, hi_y = mu - Z90 * sig, mu + Z90 * sig
    viable = y > 0
    # 1. break-even price intervals (pairs with a finite simulated break-even price)
    p_sim = np.where(viable, 1 / np.where(viable, y, 1), np.inf)
    p_lo = 1 / hi_y
    p_hi = np.where(lo_y > 0, 1 / np.where(lo_y > 0, lo_y, 1), np.inf)
    cov_p = np.mean(((p_sim >= p_lo) & (p_sim <= p_hi))[viable & (hi_y > 0)])
    p_med = np.where(mu > 0, 1 / np.where(mu > 0, mu, 1), np.inf)
    sel = viable & (mu > 0) & (p_sim <= 300)
    rel_err = np.median(np.abs(p_med[sel] / p_sim[sel] - 1))
    # 2-4. scenario-level quantities
    rows = []
    for s, g in o.groupby("scenario"):
        yy, mm, ss = g.y_inv.values, g.mu.values, g.sigma.values
        r = dict(scenario=s, source=g.source.iloc[0], region=g.region.iloc[0], boiler=g.boiler.iloc[0],
                 heat_extraction=g.heat_extraction.iloc[0], process=g.process.iloc[0], humidity=g.humidity.iloc[0],
                 n=len(g), cure_sim=float((yy <= 0).mean()), cure_pred=float(norm.cdf(-mm / ss).mean()),
                 entry_sim=float(1 / yy.max()) if yy.max() > 0 else np.nan,
                 entry_pred=float(1 / mm.max()) if mm.max() > 0 else np.nan)
        for k in KS:
            r[f"p{k}_sim"], r[f"p{k}_pred"] = library_price(yy, mm, ss, k)
        rows.append(r)
    sc = pd.DataFrame(rows)
    sc.to_csv(os.path.join(RES, "exp14_entry_price.csv"), index=False)
    summ = dict(pairs=len(o), breakeven_PI90_coverage=float(cov_p), breakeven_median_rel_err_le300=float(rel_err),
                n_rel_err=int(sel.sum()), cure_MAE=float(np.abs(sc.cure_sim - sc.cure_pred).mean()),
                cure_max_abs_err=float(np.abs(sc.cure_sim - sc.cure_pred).max()))
    for k in KS:
        both = sc[[f"p{k}_sim", f"p{k}_pred"]].dropna()
        summ[f"p{k}_n_scen"] = len(both)
        summ[f"p{k}_MAE_eur"] = float(np.abs(both.iloc[:, 0] - both.iloc[:, 1]).mean())
        summ[f"p{k}_median_rel_err"] = float(np.median(np.abs(both.iloc[:, 1] / both.iloc[:, 0] - 1)))
        summ[f"p{k}_undefined_sim"] = int(sc[f"p{k}_sim"].isna().sum())
        summ[f"p{k}_undefined_pred"] = int(sc[f"p{k}_pred"].isna().sum())
    e = sc[["entry_sim", "entry_pred"]].dropna()
    summ["entry_median_rel_err"] = float(np.median(e.entry_pred / e.entry_sim - 1))
    for src, d in sc.groupby("source"):
        for k in (1, 10):
            summ[f"p{k}_sim_median_{src}"] = float(d[f"p{k}_sim"].median())
            summ[f"p{k}_sim_range_{src}"] = f"{d[f'p{k}_sim'].min():.0f}-{d[f'p{k}_sim'].max():.0f}"
        summ[f"entry_sim_range_{src}"] = f"{d.entry_sim.min():.0f}-{d.entry_sim.max():.0f}"
    # 5. humidity penalty in price terms: wet minus dry library price for matched scenarios
    sc["ctx"] = sc.scenario.str.replace(r"-(wet|dry)$", "", regex=True)
    hp = []
    for k in (1, 10):
        w = sc.pivot_table(index=["ctx", "source", "process"], columns="humidity", values=[f"p{k}_sim", f"p{k}_pred"])
        for kind in ("sim", "pred"):
            d = (w[(f"p{k}_{kind}", "wet")] - w[(f"p{k}_{kind}", "dry")]).rename("delta").reset_index()
            d["k"], d["kind"] = k, kind
            hp.append(d)
    hp = pd.concat(hp).dropna()
    hp.to_csv(os.path.join(RES, "exp14_humidity_price.csv"), index=False)
    for (k, kind, src), d in hp.groupby(["k", "kind", "source"]):
        summ[f"humid_dp{k}_{kind}_median_{src}"] = float(d.delta.median())
        summ[f"humid_dp{k}_{kind}_n_{src}"] = len(d)
    pd.Series(summ).to_csv(os.path.join(RES, "exp14_entry_price_summary.csv"), header=["value"])
    for kk, v in summ.items():
        print(f"{kk}: {v}")
