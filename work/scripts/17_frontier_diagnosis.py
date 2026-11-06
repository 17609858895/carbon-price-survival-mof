"""Exp. 12b: diagnosis of the frontier material (Mg-MOF-74) used in the Results text.

1. Simulated margin over the runner-up, by process and humidity (median over contexts).
2. Position of its simulated working capacity and specific heat within each of the 18 process configurations.
3. Stage-A (material -> process output) out-of-fold prediction of those outputs (MOF-grouped folds, full settings).
4. Predicted rank under the Exp. 4 models (direct, hierarchical, oracle stage B).
5. Agreement of the physics proxies with the simulated working capacity and specific heat.
6. Minimum median Spearman correlation in the region-transfer runs (Exp. 4).
"""
from __future__ import annotations

import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(__file__))
from common import RES, add_physics_proxies, feature_list, fold_of, lgbp, load_long, safe_log  # noqa: E402

MG = "MgMOF74"

if __name__ == "__main__":
    k = add_physics_proxies(load_long().reset_index(drop=True))
    out = {}
    # 1. margin over runner-up
    rows = []
    for (p, h), d in k.groupby(["process", "humidity"]):
        g = d[d.MOF == MG].set_index("scenario").y_inv
        sec = d[d.MOF != MG].groupby("scenario").y_inv.max().reindex(g.index)
        rows.append(dict(process=p, humidity=h, margin_pct=float(np.median((g / sec - 1) * 100))))
    marg = pd.DataFrame(rows)
    out["margin_pct_min"], out["margin_pct_max"] = marg.margin_pct.min(), marg.margin_pct.max()
    best = k.loc[k.groupby("scenario").y_inv.idxmax(), "MOF"]
    out["scenarios_best"] = int((best == MG).sum())
    out["scenarios_total"] = int(k.scenario.nunique())
    # 2. position within configurations
    c = k.drop_duplicates(["MOF", "cfg"]).reset_index(drop=True)
    g = c[c.MOF == MG].set_index("cfg")
    oth = c[c.MOF != MG].groupby("cfg")
    out["wc_mg_median"] = float(g.working_capacity.median())
    out["wc_other_max_median"] = float(oth.working_capacity.max().median())
    out["wc_cfgs_above_all_others"] = int((g.working_capacity > oth.working_capacity.max().reindex(g.index)).sum())
    out["spec_heat_cfgs_lowest"] = int((g.spec_heat < oth.spec_heat.min().reindex(g.index)).sum())
    out["n_cfg"] = int(len(g))
    # 3. stage-A out-of-fold prediction of working capacity (log) and specific heat (log)
    feats = feature_list(c)
    X = pd.concat([c[feats], pd.get_dummies(c[["source", "process", "humidity"]], dtype=float)], axis=1)
    X.columns = [s.replace(" ", "_").replace(".", "p") for s in X.columns]
    fold = c.MOF.map(fold_of).values
    for t in ["working_capacity", "spec_heat"]:
        y = safe_log(c[t].values + (1e-4 if t == "working_capacity" else 0))
        oof = np.zeros(len(y))
        for f in range(5):
            tr = fold != f
            oof[~tr] = lgb.LGBMRegressor(**lgbp()).fit(X[tr], y[tr]).predict(X[~tr])
        m = (c.MOF == MG).values
        out[f"stageA_{t}_ratio_median"] = float(np.median(10 ** oof[m] / 10 ** y[m]))
    # 4. ranks under the Exp. 4 models
    o = pd.read_parquet(os.path.join(RES, "exp4_oof_preds.parquet"))
    for col in ["Direct", "Hier-oof", "Hier+direct", "Oracle-B"]:
        rk = o.groupby("scenario")[col].rank(ascending=False)
        out[f"rank_{col}"] = float(rk[o.MOF == MG].median())
    # 5. physics proxies vs simulated outputs (median within-configuration Spearman)
    sp = lambda a, b: float(c.groupby("cfg").apply(lambda d: spearmanr(d[a], d[b], nan_policy="omit").correlation).median())  # noqa: E731
    out["rho_proxyWC_WC"] = sp("phys_wc", "working_capacity")
    out["rho_uptake013_WC"] = sp("iso_CO2_0.13bar", "working_capacity")
    out["rho_proxyE_specheat"] = sp("phys_log_e", "spec_heat")
    out["mg_rank_proxyWC_median"] = float(c.groupby("cfg").phys_wc.rank(ascending=False)[c.MOF == MG].median())
    out["mg_rank_proxyE_median"] = float(c.groupby("cfg").phys_log_e.rank()[c.MOF == MG].median())
    # 6. region transfer: minimum median Spearman
    l = pd.read_csv(os.path.join(RES, "exp4_loro_fewshot.csv"))
    out["loro_min_spearman"] = float(l.median_scenario_spearman.min())
    res = pd.Series(out)
    res.to_csv(os.path.join(RES, "exp12_frontier_diagnosis.csv"), header=["value"])
    marg.to_csv(os.path.join(RES, "exp12_frontier_margin.csv"), index=False)
    print(res.round(4).to_string())
    print(marg.round(2).to_string())
