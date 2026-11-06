"""Exp. 3: carbon-price viability as a survival problem.

"Time" = carbon price p. A material-scenario pair becomes viable ("event") at p* = nCAC when
net CO2 avoidance is positive; pairs with net avoidance <= 0 never become viable (cured fraction),
and p* > P_MAX is administratively censored. S(p | x) = P(p* > p | x) = P(not viable at price p).

Learners compared (same features, same MOF folds, same 4 evaluation scenarios):
  CoxPH (elastic-net), RSF, GBSA (sksurv), XGBoost-AFT, hurdle (classifier x log-normal AFT),
  RIV = reciprocal-viability regression: y = 1/nCAC with heteroscedastic Gaussian predictive
        S(p|x) = Phi((1/p - mu(x)) / sigma(x)), exact because {viable at p} <=> {y >= 1/p}.
Pooled (all-scenario) versions are fitted for XGB-AFT, hurdle and RIV.
Metrics: Harrell C, cumulative/dynamic AUC at 75/100/150/200 $/t, integrated Brier score (IBS) 40-500 $/t.
"""
from __future__ import annotations

import os
import sys
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import norm
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc, integrated_brier_score
from sksurv.util import Surv

sys.path.insert(0, os.path.dirname(__file__))
from common import THREADS, lgbp, RES, feature_list, fold_of, load_long, one_hot_scen  # noqa: E402
warnings.filterwarnings("ignore")
P_MAX = 1000.0
TIMES = np.array([40, 50, 60, 75, 100, 125, 150, 200, 250, 300, 400, 500], float)
SCENARIOS = ["Cement_Storage_UK_TVSA02-wet", "NGCC-onshore_Storage_UK_TVSA06-wet"]
LGB = lgbp()


def surv_target(ncac):
    event = (ncac > 0) & (ncac <= P_MAX)
    t = np.where(event, ncac, P_MAX)
    return event, t


def evaluate(name, sc, ev_tr, t_tr, ev_te, t_te, S, risk):
    """S: survival matrix (n_test x len(TIMES))."""
    ytr, yte = Surv.from_arrays(ev_tr, t_tr), Surv.from_arrays(ev_te, t_te)
    c = concordance_index_censored(ev_te, t_te, risk)[0]
    ok = (TIMES < t_te.max()) & (TIMES >= t_te[ev_te].min()) if ev_te.any() else TIMES < 0
    tt = TIMES[ok]
    try:
        auc, mean_auc = cumulative_dynamic_auc(ytr, yte, risk, tt)
        aucd = dict(zip([f"cdAUC@{int(x)}" for x in tt], auc))
    except Exception:
        aucd, mean_auc = {}, np.nan
    try:
        ibs = integrated_brier_score(ytr, yte, np.clip(S[:, ok], 0, 1), tt)
    except Exception:
        ibs = np.nan
    r = dict(scenario=sc, model=name, C_index=c, mean_cdAUC=mean_auc, IBS=ibs)
    r.update({k: v for k, v in aucd.items() if k in ("cdAUC@75", "cdAUC@100", "cdAUC@150", "cdAUC@200")})
    return r


def riv_fit_predict(Xtr, ytr, Xte, groups_tr):
    """Mean model on y=1/nCAC + scale model on inner-OOF |residual| -> Gaussian predictive."""
    from sklearn.model_selection import GroupKFold
    mu = lgb.LGBMRegressor(**LGB).fit(Xtr, ytr)
    oof = np.zeros(len(ytr))
    for a, b in GroupKFold(3).split(Xtr, groups=groups_tr):
        oof[b] = lgb.LGBMRegressor(**lgbp(0.5)).fit(Xtr[a], ytr[a]).predict(Xtr[b])
    sig = lgb.LGBMRegressor(**lgbp(0.5)).fit(Xtr, np.log(np.abs(ytr - oof) + 1e-6))
    m = mu.predict(Xte)
    s = np.exp(sig.predict(Xte)) * np.sqrt(np.pi / 2)  # E|e| = sigma*sqrt(2/pi)
    s = np.maximum(s, 1e-5)
    S = norm.cdf((1.0 / TIMES[None, :] - m[:, None]) / s[:, None])  # P(y < 1/p) = P(not viable at p)
    return S, m


def hurdle_fit_predict(Xtr, ncac_tr, Xte):
    v = ncac_tr > 0
    clf = lgb.LGBMClassifier(**LGB).fit(Xtr, v.astype(int))
    reg = lgb.LGBMRegressor(**LGB).fit(Xtr[v], np.log(ncac_tr[v]))
    oofres = np.log(ncac_tr[v]) - reg.predict(Xtr[v])
    s = np.std(oofres) * 1.5 + 1e-3  # in-sample residual inflated (no inner CV, conservative)
    pv = clf.predict_proba(Xte)[:, 1]
    mu = reg.predict(Xte)
    F = norm.cdf((np.log(TIMES)[None, :] - mu[:, None]) / s)
    S = 1 - pv[:, None] * F
    return S, pv * np.exp(-mu)


def xgb_aft_fit_predict(Xtr, ev, t, Xte):
    # survival:aft internally models log(time). Bounds must be on the original time scale.
    lo = np.asarray(t, dtype=float)
    hi = np.where(ev, t, np.inf)
    d = xgb.DMatrix(Xtr); d.set_float_info("label_lower_bound", lo); d.set_float_info("label_upper_bound", hi)
    params = dict(objective="survival:aft", eval_metric="aft-nloglik", aft_loss_distribution="normal",
                  aft_loss_distribution_scale=0.8, learning_rate=LGB["learning_rate"], max_depth=6, subsample=0.8,
                  colsample_bytree=0.7, tree_method="hist", nthread=THREADS)
    b = xgb.train(params, d, num_boost_round=LGB['n_estimators'])
    # The raw margin is log(time); requesting it avoids overflow from exp(margin).
    mu = b.predict(xgb.DMatrix(Xte), output_margin=True)
    S = 1 - norm.cdf((np.log(TIMES)[None, :] - mu[:, None]) / 0.8)
    return S, -mu


if __name__ == "__main__":
    t0 = time.time()
    k = load_long()
    feats = feature_list(k)
    k["fold"] = k.MOF.map(fold_of)
    rows = []
    # ---------------- per-scenario learners
    for sc in SCENARIOS:
        d = k[k.scenario == sc].reset_index(drop=True)
        imp = SimpleImputer(strategy="median")
        X = imp.fit_transform(d[feats].values)
        ev, t = surv_target(d.nCAC.values)
        for f in range(5):
            tr, te = d.fold.values != f, d.fold.values == f
            sc_ = StandardScaler().fit(X[tr])
            Xtr, Xte = sc_.transform(X[tr]), sc_.transform(X[te])
            ytr = Surv.from_arrays(ev[tr], t[tr])
            fits = {}
            cox = CoxnetSurvivalAnalysis(l1_ratio=0.5, alpha_min_ratio=0.01, fit_baseline_model=True).fit(Xtr, ytr)
            fits["CoxNet"] = cox
            fits["RSF"] = RandomSurvivalForest(n_estimators=300, min_samples_leaf=5, max_features=0.5, n_jobs=THREADS,
                                               random_state=0).fit(Xtr, ytr)
            fits["GBSA"] = GradientBoostingSurvivalAnalysis(n_estimators=300, learning_rate=0.05, max_depth=3,
                                                            subsample=0.8, random_state=0).fit(Xtr, ytr)
            for n, m in fits.items():
                if n == "CoxNet":
                    fns = m.predict_survival_function(Xte, alpha=m.alphas_[len(m.alphas_) // 2])
                    risk = m.predict(Xte, alpha=m.alphas_[len(m.alphas_) // 2])
                else:
                    fns = m.predict_survival_function(Xte)
                    risk = m.predict(Xte)
                S = np.vstack([np.clip([fn(min(max(x, fn.x[0]), fn.x[-1])) for x in TIMES], 0, 1) for fn in fns])
                rows.append(dict(evaluate(n, sc, ev[tr], t[tr], ev[te], t[te], S, risk), fold=f))
            S, risk = xgb_aft_fit_predict(Xtr, ev[tr], t[tr], Xte)
            rows.append(dict(evaluate("XGB-AFT", sc, ev[tr], t[tr], ev[te], t[te], S, risk), fold=f))
            S, risk = hurdle_fit_predict(Xtr, d.nCAC.values[tr], Xte)
            rows.append(dict(evaluate("Hurdle(clf x lognormal)", sc, ev[tr], t[tr], ev[te], t[te], S, risk), fold=f))
            S, risk = riv_fit_predict(Xtr, d.y_inv.values[tr], Xte, d.MOF.values[tr])
            rows.append(dict(evaluate("RIV (ours, per-scenario)", sc, ev[tr], t[tr], ev[te], t[te], S, risk), fold=f))
            print(sc, f, round(time.time() - t0), "s", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(RES, "exp3_price_survival.csv"), index=False)

    # ---------------- pooled scenario-conditioned learners (train on all 132 scenarios)
    X = pd.concat([k[feats].reset_index(drop=True), one_hot_scen(k).reset_index(drop=True)], axis=1).values
    ev, t = surv_target(k.nCAC.values)
    for f in range(5):
        tr, te_all = k.fold.values != f, k.fold.values == f
        res = {}
        res["XGB-AFT (pooled)"] = xgb_aft_fit_predict(X[tr], ev[tr], t[tr], X[te_all])
        res["Hurdle (pooled)"] = hurdle_fit_predict(X[tr], k.nCAC.values[tr], X[te_all])
        res["RIV (ours, pooled)"] = riv_fit_predict(X[tr], k.y_inv.values[tr], X[te_all], k.MOF.values[tr])
        kte = k[te_all].reset_index(drop=True)
        for sc in SCENARIOS:
            idx = np.where(kte.scenario.values == sc)[0]
            trs = tr & (k.scenario.values == sc)
            for n, (S, risk) in res.items():
                rows.append(dict(evaluate(n, sc, ev[trs], t[trs], ev[te_all][idx], t[te_all][idx], S[idx], risk[idx]), fold=f))
        # all-scenario pooled summary (C-index on every held-out row)
        for n, (S, risk) in res.items():
            rows.append(dict(evaluate(n, "ALL_132", ev[tr], t[tr], ev[te_all], t[te_all], S, risk), fold=f))
        print("pooled fold", f, round(time.time() - t0), "s", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(RES, "exp3_price_survival.csv"), index=False)
    print(res.groupby("model")[["C_index", "mean_cdAUC", "IBS", "cdAUC@100"]].mean().sort_values("IBS").round(4).to_string())
    print(res[res.scenario == "ALL_132"].groupby("model")[["C_index", "mean_cdAUC", "IBS"]].mean().round(4))
    print("done", round(time.time() - t0), "s")
