"""Exp. 6: calibrated carbon-price viability, held-out feedback-round test and atlas.

1. MOF-grouped OOF RIV predictions (mean + heteroscedastic scale) for all 132 main scenarios
   -> calibration of P(viable at p), interval coverage, per-scenario survival curves (atlas).
2. Final RIV model fitted to finite original-round responses -> retrospective evaluation on 153 published
   feedback-round MOFs (shortlisted by the original study's CatBoost loop; 9 UK wet scenarios; disjoint MOF set).
   The historical script filename is retained for compatibility; this study adds no prospective experiment.
Mean model (PRISMA_RIV_MEAN): "linear+phys" (default; linear-leaf trees + physics proxies, chosen in Exp. 12) or
"lgb" (constant-leaf LightGBM on the base features, the v2 model archived in results/v2_constleaf/).
"""
from __future__ import annotations

import os
import sys
import time
import warnings

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(__file__))
from common import PHYS, RES, SCEN_CAT, add_physics_proxies, feature_list, fold_of, lgbp, load_long  # noqa: E402

warnings.filterwarnings("ignore")
RIV_MEAN = os.environ.get("PRISMA_RIV_MEAN", "linear+phys")
LINEAR = RIV_MEAN.startswith("linear")
YS = 100.0          # fit in t per 100 EUR so that linear-leaf ridge penalties act on O(1) targets
PRICES = np.array([40, 50, 60, 75, 100, 125, 150, 200, 250, 300, 400, 500], float)


def design(df, feats, levels):
    X = df[feats].copy()
    for c in SCEN_CAT:
        for lv in levels[c]:
            X[f"{c}={lv}"] = (df[c].values == lv).astype(float)
    X.columns = [c.replace(" ", "_").replace(".", "p") for c in X.columns]
    return X


def riv_features(k):
    return feature_list(k) + (PHYS if "phys" in RIV_MEAN else [])


def _std(models, X):
    return ((X.fillna(models["med"]) - models["med"]) / models["sd"]) if models["linear"] else X


def fit_riv(X, y, groups):
    """Mean model + cross-fitted heteroscedastic scale with a cross-fitted residual-quantile 90 % scale adjustment (targets in t / 100 EUR)."""
    st = dict(med=X.median(), sd=X.std().replace(0, 1), linear=LINEAR)
    Z = _std(st, X)
    pm = lgbp(linear_tree=True, linear_lambda=1.0) if LINEAR else lgbp()
    pi = dict(pm, n_estimators=max(50, pm["n_estimators"] // 2))
    ys = y * YS
    mu = lgb.LGBMRegressor(**pm).fit(Z, ys)
    oof = np.zeros(len(y))
    for a, b in GroupKFold(3).split(Z, groups=groups):
        oof[b] = lgb.LGBMRegressor(**pi).fit(Z.iloc[a], ys[a]).predict(Z.iloc[b])
    r = np.log(np.abs(ys - oof) + 1e-5)
    sig = lgb.LGBMRegressor(**lgbp(0.5)).fit(Z, r)
    # cross-fitted scale predictions so the conformal factor is not computed on in-sample (overfitted) scales
    raw = np.zeros(len(y))
    for a, b in GroupKFold(3).split(Z, groups=groups):
        raw[b] = np.exp(lgb.LGBMRegressor(**lgbp(0.5)).fit(Z.iloc[a], r[a]).predict(Z.iloc[b])) * np.sqrt(np.pi / 2)
    z = (ys - oof) / np.maximum(raw, 1e-5)
    # Cross-fitted residual-quantile scale correction; empirical outer-fold coverage is evaluated separately.
    c90 = np.quantile(np.abs(z), 0.90) / norm.ppf(0.95)
    return dict(st, mu=mu, sig=sig, c90=c90)


def predict_riv(models, X):
    Z = _std(models, X)
    m = models["mu"].predict(Z) / YS
    s = np.maximum(np.exp(models["sig"].predict(Z)) * np.sqrt(np.pi / 2) * models["c90"], 1e-5) / YS
    return m, s


def p_viable(m, s, prices=PRICES):
    return 1 - norm.cdf((1.0 / prices[None, :] - m[:, None]) / s[:, None])  # P(y >= 1/p)


if __name__ == "__main__":
    t0 = time.time()
    k = add_physics_proxies(load_long(include_feedback=True).reset_index(drop=True))
    feats = riv_features(k)
    print("RIV mean model:", RIV_MEAN, "features", len(feats), flush=True)
    levels = {c: sorted(k[c].unique()) for c in SCEN_CAT}
    main = k[~k.feedback_round].reset_index(drop=True)
    fb = k[k.feedback_round].reset_index(drop=True)
    Xm, ym = design(main, feats, levels), main.y_inv.values
    main["fold"] = main.MOF.map(fold_of)
    mu_oof, s_oof = np.zeros(len(main)), np.zeros(len(main))
    for f in range(5):
        tr, te = main.fold.values != f, main.fold.values == f
        mods = fit_riv(Xm[tr], ym[tr], main.MOF.values[tr])
        mu_oof[te], s_oof[te] = predict_riv(mods, Xm[te])
        print("oof fold", f, round(time.time() - t0), "s", flush=True)
    main["mu"], main["sigma"] = mu_oof, s_oof
    Pv = p_viable(mu_oof, s_oof)
    for j, p in enumerate(PRICES):
        main[f"Pv@{int(p)}"] = Pv[:, j]
    keep = ["MOF", "scenario", "source", "region", "boiler", "heat_extraction", "process", "humidity", "nCAC", "y_inv",
            "purity", "mu", "sigma"] + [f"Pv@{int(p)}" for p in PRICES]
    main[keep].to_parquet(os.path.join(RES, "exp6_riv_oof.parquet"), index=False)

    # calibration + coverage
    cal = []
    for j, p in enumerate(PRICES):
        t = (main.y_inv.values >= 1 / p).astype(int)
        pr = Pv[:, j]
        bins = np.clip((pr * 10).astype(int), 0, 9)
        ece = sum(abs(pr[bins == b].mean() - t[bins == b].mean()) * (bins == b).mean() for b in range(10) if (bins == b).any())
        cal.append(dict(price=p, base_rate=t.mean(), AUC=roc_auc_score(t, pr), Brier=brier_score_loss(t, pr), ECE=ece))
    cal = pd.DataFrame(cal)
    lo, hi = mu_oof - norm.ppf(0.95) * s_oof, mu_oof + norm.ppf(0.95) * s_oof
    cov = np.mean((ym >= lo) & (ym <= hi))
    cal["PI90_coverage_y"] = cov
    cal["PI90_coverage_y_viable"] = np.mean(((ym >= lo) & (ym <= hi))[ym > 0])
    cal.to_csv(os.path.join(RES, "exp6_calibration.csv"), index=False)
    print(cal.round(4).to_string())

    # final model -> prospective feedback-round MOFs
    final = fit_riv(Xm, ym, main.MOF.values)
    joblib.dump(dict(models=final, feats=feats, levels=levels, mean_model=RIV_MEAN), os.path.join(RES, "riv_final_model.joblib"))
    fb = fb.dropna(subset=feats[:5]).reset_index(drop=True)
    Xf = design(fb, feats, levels)
    fb["mu"], fb["sigma"] = predict_riv(final, Xf)
    Pf = p_viable(fb.mu.values, fb.sigma.values)
    rows = []
    for s, g in fb.groupby("scenario"):
        idx = g.index.values
        r = dict(scenario=s, n=len(g), spearman=spearmanr(g.y_inv, g.mu).correlation,
                 viable_frac=(g.y_inv > 0).mean(), frac_le100=(g.y_inv >= 0.01).mean())
        t = (g.y_inv.values >= 0.01).astype(int)
        r["AUC@100"] = roc_auc_score(t, Pf[idx, 4]) if 0 < t.sum() < len(t) else np.nan
        r["Brier@100"] = brier_score_loss(t, Pf[idx, 4])
        r["pred_frac_le100"] = Pf[idx, 4].mean()
        top_t, top_p = set(g.nlargest(20, "y_inv").MOF), set(g.nlargest(20, "mu").MOF)
        r["P@20"] = len(top_t & top_p) / 20
        rows.append(r)
    fbres = pd.DataFrame(rows)
    fbres.to_csv(os.path.join(RES, "exp6_feedback_prospective.csv"), index=False)
    fb[["MOF", "scenario", "nCAC", "y_inv", "mu", "sigma"]].to_parquet(os.path.join(RES, "exp6_feedback_preds.parquet"), index=False)
    print(fbres.round(3).to_string())
    print("done", round(time.time() - t0), "s")
