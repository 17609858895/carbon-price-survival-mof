"""Exp. 5: causal ML with ground-truth potential outcomes.

PrISMa evaluates every MOF under paired process interventions, so individual treatment effects (ITE) are
observed exactly:  humidity  (wet vs dry)  and  vacuum swing (TVSA02 vs TSA), all else fixed.
Part A  We hide one potential outcome per unit under confounded assignment
        e(x) = sigmoid(gamma * z(x)), z = standardised [log water Henry, Q_CO2],
        and test whether CATE learners recover the true ITE (PEHE, ATE bias, sign accuracy) on held-out MOFs.
Part B  Mediation of the humidity effect through process KPIs using a stage-B (process -> TEA) emulator:
        Shapley decomposition of the emulated wet-dry change over mediators.
"""
from __future__ import annotations

import itertools
import os
import sys
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

sys.path.insert(0, os.path.dirname(__file__))
from common import THREADS, lgbp, RES, feature_list, fold_of, load_long, safe_log  # noqa: E402

warnings.filterwarnings("ignore")
SCALE = 1000.0
LGB = lgbp(0.5, num_leaves=31, min_child_samples=20)
CTX = ["source", "region", "boiler", "heat_extraction"]


def paired(k, treat_col, t1, t0, keep):
    a = k[k[treat_col] == t1].set_index(["MOF"] + keep)
    b = k[k[treat_col] == t0].set_index(["MOF"] + keep)
    idx = a.index.intersection(b.index)
    d = a.loc[idx].reset_index()
    d["y1"] = a.loc[idx, "y_inv"].values * SCALE
    d["y0"] = b.loc[idx, "y_inv"].values * SCALE
    d["ite"] = d.y1 - d.y0
    return d


def learners(Xtr, T, Y, Xte, e_tr):
    out = {}
    XT = np.c_[Xtr, T]
    s = lgb.LGBMRegressor(**LGB).fit(XT, Y)
    out["S-learner"] = s.predict(np.c_[Xte, np.ones(len(Xte))]) - s.predict(np.c_[Xte, np.zeros(len(Xte))])
    m1 = lgb.LGBMRegressor(**LGB).fit(Xtr[T == 1], Y[T == 1])
    m0 = lgb.LGBMRegressor(**LGB).fit(Xtr[T == 0], Y[T == 0])
    out["T-learner"] = m1.predict(Xte) - m0.predict(Xte)
    # propensity (cross-fitted)
    from sklearn.model_selection import KFold
    eh = np.zeros(len(T)); mu1 = np.zeros(len(T)); mu0 = np.zeros(len(T)); mh = np.zeros(len(T))
    for a, b in KFold(3, shuffle=True, random_state=0).split(Xtr):
        c = lgb.LGBMClassifier(**LGB).fit(Xtr[a], T[a]); eh[b] = c.predict_proba(Xtr[b])[:, 1]
        r1 = lgb.LGBMRegressor(**LGB).fit(Xtr[a][T[a] == 1], Y[a][T[a] == 1]); mu1[b] = r1.predict(Xtr[b])
        r0 = lgb.LGBMRegressor(**LGB).fit(Xtr[a][T[a] == 0], Y[a][T[a] == 0]); mu0[b] = r0.predict(Xtr[b])
        rm = lgb.LGBMRegressor(**LGB).fit(Xtr[a], Y[a]); mh[b] = rm.predict(Xtr[b])
    eh = np.clip(eh, 0.02, 0.98)
    # X-learner
    d1 = Y[T == 1] - m0.predict(Xtr[T == 1]); d0 = m1.predict(Xtr[T == 0]) - Y[T == 0]
    t1 = lgb.LGBMRegressor(**LGB).fit(Xtr[T == 1], d1); t0 = lgb.LGBMRegressor(**LGB).fit(Xtr[T == 0], d0)
    ce = lgb.LGBMClassifier(**LGB).fit(Xtr, T).predict_proba(Xte)[:, 1]
    out["X-learner"] = ce * t0.predict(Xte) + (1 - ce) * t1.predict(Xte)
    # DR-learner (AIPW pseudo-outcome)
    psi = mu1 - mu0 + T * (Y - mu1) / eh - (1 - T) * (Y - mu0) / (1 - eh)
    out["DR-learner"] = lgb.LGBMRegressor(**LGB).fit(Xtr, psi).predict(Xte)
    # R-learner / DML (Robinson partialling-out, cross-fitted nuisances, weighted pseudo-outcome)
    rt, ry = T - eh, Y - mh
    w = rt ** 2
    ok = w > 1e-4
    out["DML (R-learner)"] = lgb.LGBMRegressor(**LGB).fit(Xtr[ok], ry[ok] / rt[ok], sample_weight=w[ok]).predict(Xte)
    # linear DML ATE (partially linear model)
    out["_ATE_DML_PLR"] = float(np.sum(rt * ry) / np.sum(rt * rt))
    out["_ATE_naive"] = float(Y[T == 1].mean() - Y[T == 0].mean())
    out["_ATE_AIPW"] = float(psi.mean())
    return out


if __name__ == "__main__":
    t0 = time.time()
    k = load_long()
    feats = feature_list(k)
    rows = []
    rng = np.random.default_rng(1)
    designs = {"humidity (wet vs dry)": ("humidity", "wet", "dry", ["source", "process"] + CTX[1:]),
               "vacuum (TVSA02 vs TSA)": ("process", "TVSA02", "TSA", ["source", "humidity"] + CTX[1:])}
    ite_store = {}
    for dname, (col, t1, t0_, keep) in designs.items():
        d = paired(k, col, t1, t0_, keep)
        # one representative context per source keeps units ~ MOF-level and the run tractable
        d = d[(d.region == "UK") & (d.boiler == "default") & (d.heat_extraction == "True")].reset_index(drop=True)
        ite_store[dname] = d
        ctx_oh = pd.get_dummies(d[[c for c in keep if c in ("source", "process", "humidity")]].astype(str), dtype=float)
        X = np.c_[d[feats].fillna(d[feats].median()).values, ctx_oh.values]
        z = np.c_[safe_log(d["w_H"].values), d["Q_CO2"].values]
        z = np.nan_to_num((z - np.nanmean(z, 0)) / np.nanstd(z, 0))
        zc = z[:, 0] - z[:, 1]  # water-loving and strongly binding MOFs more likely observed as treated
        d["fold"] = d.MOF.map(fold_of)
        for gamma in [0.0, 2.0]:
            e = 1 / (1 + np.exp(-gamma * zc))
            T = rng.binomial(1, e)
            Y = np.where(T == 1, d.y1.values, d.y0.values)
            for f in range(5):
                tr, te = d.fold.values != f, d.fold.values == f
                est = learners(X[tr], T[tr], Y[tr], X[te], e[tr])
                tau = d.ite.values[te]
                true_ate_tr = d.ite.values[tr].mean()
                for n, v in est.items():
                    if n.startswith("_"):
                        rows.append(dict(design=dname, gamma=gamma, fold=f, estimator=n[1:], ATE_bias=v - true_ate_tr,
                                         ATE_true=true_ate_tr))
                    else:
                        rows.append(dict(design=dname, gamma=gamma, fold=f, estimator=n,
                                         PEHE=np.sqrt(np.mean((v - tau) ** 2)), ITE_sd=tau.std(),
                                         spearman=spearmanr(v, tau).correlation, sign_acc=np.mean(np.sign(v) == np.sign(tau)),
                                         ATE_bias=v.mean() - tau.mean(), ATE_true=tau.mean()))
            print(dname, gamma, round(time.time() - t0), "s", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(RES, "exp5_cate_ground_truth.csv"), index=False)
    print(res.groupby(["design", "gamma", "estimator"])[["PEHE", "spearman", "sign_acc", "ATE_bias"]].mean().round(4).to_string())
    for n, d in ite_store.items():
        d[["MOF", "source"] + [c for c in d.columns if c in ("process", "humidity")] + ["y0", "y1", "ite"]].to_parquet(
            os.path.join(RES, f"exp5_true_ite_{n.split()[0]}.parquet"), index=False)

    # ---------------------------------------------------------------- Part B: mediation via stage-B emulator
    med_cols = ["purity", "recovery", "working_capacity", "productivity", "spec_heat", "el_energy", "rho_b", "CO2_captured"]

    def bX(df):
        X = pd.DataFrame({"purity": df.purity.values, "recovery": df.recovery.values,
                          "log_wc": safe_log(df.working_capacity.values + 1e-4), "productivity": df.productivity.values,
                          "log_spec_heat": safe_log(df.spec_heat.values), "el_energy": df.el_energy.values * 100,
                          "rho_b": df.rho_b.values, "log_CO2_capt": safe_log(df.CO2_captured.values),
                          "density": df.density.values, "Cp": df.Cp.values})
        for c in ["source", "process", "region", "boiler", "heat_extraction"]:
            for lv in sorted(k[c].unique()):
                X[f"{c}={lv}"] = (df[c].values == lv).astype(float)
        X["humidity=wet"] = (df.humidity.values == "wet").astype(float)
        return X

    kb = k.copy()
    kb["fold"] = kb.MOF.map(fold_of)
    tr = kb.fold.values != 0
    emu = lgb.LGBMRegressor(**lgbp(1.5, num_leaves=127, min_child_samples=20, colsample_bytree=0.8))
    emu.fit(bX(kb[tr]), kb.y_inv.values[tr] * SCALE)
    fid = r2_score(kb.y_inv.values[~tr] * SCALE, emu.predict(bX(kb[~tr])))
    print("stage-B emulator held-out R2 (TRUE process inputs):", round(fid, 4))
    # paired wet/dry rows, UK default context
    wet = k[(k.humidity == "wet") & (k.region == "UK") & (k.boiler == "default") & (k.heat_extraction == "True")]
    dry = k[(k.humidity == "dry") & (k.region == "UK") & (k.boiler == "default") & (k.heat_extraction == "True")]
    keyc = ["MOF", "source", "process"]
    m = wet.merge(dry, on=keyc, suffixes=("_w", "_d"))
    base = dry.set_index(keyc).loc[m.set_index(keyc).index].reset_index()
    wetd = wet.set_index(keyc).loc[m.set_index(keyc).index].reset_index()
    groups = {"purity/recovery": ["purity", "recovery"], "working capacity/productivity": ["working_capacity", "productivity"],
              "specific heat": ["spec_heat"], "vacuum power": ["el_energy"], "captured CO2/bed": ["CO2_captured", "rho_b"],
              "direct (TEA humidity flag)": []}
    gnames = list(groups)

    def emulate(subset):
        df = base.copy()
        for g in subset:
            for c in groups[g]:
                df[c] = wetd[c].values
        df["humidity"] = "wet" if "direct (TEA humidity flag)" in subset else "dry"
        return emu.predict(bX(df))

    cache = {}
    for r in range(len(gnames) + 1):
        for S in itertools.combinations(gnames, r):
            cache[frozenset(S)] = emulate(S)
    from math import factorial
    n = len(gnames)
    shap = {}
    for g in gnames:
        val = np.zeros(len(base))
        others = [h for h in gnames if h != g]
        for r in range(n):
            for S in itertools.combinations(others, r):
                wgt = factorial(len(S)) * factorial(n - len(S) - 1) / factorial(n)
                val += wgt * (cache[frozenset(S) | {g}] - cache[frozenset(S)])
        shap[g] = val
    med = pd.DataFrame(shap)
    med["emulated_total"] = cache[frozenset(gnames)] - cache[frozenset()]
    med["true_total"] = (wetd.y_inv.values - base.y_inv.values) * SCALE
    med[["MOF", "source", "process"]] = base[["MOF", "source", "process"]].values
    med.to_parquet(os.path.join(RES, "exp5_mediation_shapley.parquet"), index=False)
    print("mediation fidelity R2(emulated vs true wet-dry change):", round(r2_score(med.true_total, med.emulated_total), 4))
    print(med.groupby("source")[gnames + ["emulated_total", "true_total"]].mean().round(4).to_string())
    print(med[gnames].abs().mean().round(4))
    print("done", round(time.time() - t0), "s")
