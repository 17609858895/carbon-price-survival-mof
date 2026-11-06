"""Exp. 10: carbon-price viability atlas and storage-purity gate (uses Exp. 6 OOF RIV predictions).

1. Scenario-level calibration: share of MOFs viable at price p, simulated vs predicted (mean P(viable)).
2. Best-material price per scenario: simulated min nCAC vs nCAC of the predicted-best MOF.
3. Robust candidates: mean P(viable at p) over all scenarios vs simulated fraction of viable scenarios.
4. Purity gate (storage spec purity >= 96 %): pi(x) = P(purity >= 96 | material, process configuration) is
   learned on unique MOF-configuration rows with the same MOF folds; deployable-at-p probability
   = pi(x) * P(y >= 1/p | x) (conditional independence), compared with a direct per-price classifier.
"""
from __future__ import annotations

import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from common import RES, feature_list, fold_of, lgbp, load_long  # noqa: E402

t0 = time.time()
PR = [50, 75, 100, 150, 200, 300]
o = pd.read_parquet(os.path.join(RES, "exp6_riv_oof.parquet"))
k = load_long().reset_index(drop=True)
feats = feature_list(k)
o = o.merge(k[["MOF", "scenario"] + feats], on=["MOF", "scenario"], how="left")
o["cfg"] = o.source + "|" + o.process + "|" + o.humidity

# 1. scenario-level shares
rows = []
for s, g in o.groupby("scenario"):
    r = dict(scenario=s, source=g.source.iloc[0], region=g.region.iloc[0], boiler=g.boiler.iloc[0],
             heat_extraction=g.heat_extraction.iloc[0], process=g.process.iloc[0], humidity=g.humidity.iloc[0], n=len(g))
    for p in PR:
        r[f"true_share@{p}"] = (g.y_inv >= 1 / p).mean()
        r[f"pred_share@{p}"] = g[f"Pv@{p}"].mean()
    r["true_best_nCAC"] = 1 / g.y_inv.max()
    r["pred_best_true_nCAC"] = 1 / g.loc[g.mu.idxmax(), "y_inv"] if g.loc[g.mu.idxmax(), "y_inv"] > 0 else np.inf
    r["pred_best_pred_nCAC"] = 1 / g.mu.max()
    top10 = g.nlargest(10, "mu")
    r["best_of_pred_top10_nCAC"] = 1 / top10.y_inv.max()
    rows.append(r)
sc = pd.DataFrame(rows)
sc.to_csv(os.path.join(RES, "exp10_scenario_atlas.csv"), index=False)
for p in PR:
    print(p, "scenario-share MAE", round((sc[f"true_share@{p}"] - sc[f"pred_share@{p}"]).abs().mean(), 4),
          "corr", round(np.corrcoef(sc[f"true_share@{p}"], sc[f"pred_share@{p}"])[0, 1], 4))
print("best-of-predicted-top10 regret %", round(((sc.best_of_pred_top10_nCAC / sc.true_best_nCAC) - 1).median() * 100, 2))

# 3. robust candidates across all scenarios
rob = o.groupby("MOF").agg(**{f"pred_P@{p}": (f"Pv@{p}", "mean") for p in PR},
                           **{f"true_frac@{p}": ("y_inv", (lambda s, p=p: (s >= 1 / p).mean())) for p in PR},
                           n_scen=("scenario", "nunique"))
rob.sort_values("pred_P@100", ascending=False).to_csv(os.path.join(RES, "exp10_robust_candidates.csv"))
print(rob.sort_values("pred_P@100", ascending=False).head(10).round(3).to_string())
print("robust ranking spearman (pred P@100 vs true frac@100):",
      round(rob[["pred_P@100", "true_frac@100"]].corr("spearman").iloc[0, 1], 4))

# 4. purity gate
u = o.drop_duplicates(["MOF", "cfg"]).reset_index(drop=True)
u["pur96"] = (u.purity >= 96).astype(int)
XA = pd.concat([u[feats], pd.get_dummies(u[["source", "process", "humidity"]], dtype=float)], axis=1)
XA.columns = [c.replace(" ", "_").replace(".", "p") for c in XA.columns]
fa = u.MOF.map(fold_of).values
pi = np.zeros(len(u))
for f in range(5):
    pi[fa == f] = lgb.LGBMClassifier(**lgbp()).fit(XA[fa != f], u.pur96.values[fa != f]).predict_proba(XA[fa == f])[:, 1]
u["pi"] = pi
print("purity>=96 classifier AUC", round(roc_auc_score(u.pur96, pi), 4), "Brier", round(brier_score_loss(u.pur96, pi), 4))
o = o.merge(u[["MOF", "cfg", "pi"]], on=["MOF", "cfg"], how="left")
o["pur96"] = (o.purity >= 96).astype(int)
Xd = pd.concat([o[feats], pd.get_dummies(o[["source", "process", "humidity", "region", "boiler", "heat_extraction"]].astype(str), dtype=float)], axis=1)
Xd.columns = [c.replace(" ", "_").replace(".", "p") for c in Xd.columns]
fo = o.MOF.map(fold_of).values
gate_rows = []
for p in [75, 100, 150]:
    t = ((o.pur96 == 1) & (o.y_inv >= 1 / p)).astype(int).values
    prod = o.pi.values * o[f"Pv@{p}"].values
    direct = np.zeros(len(o))
    for f in range(5):
        direct[fo == f] = lgb.LGBMClassifier(**lgbp()).fit(Xd[fo != f], t[fo != f]).predict_proba(Xd[fo == f])[:, 1]
    for name, pr in [("pi x RIV (one model for all prices)", prod), ("direct classifier at this price", direct),
                     ("RIV without gate", o[f"Pv@{p}"].values)]:
        bins = np.clip((pr * 10).astype(int), 0, 9)
        ece = sum(abs(pr[bins == b].mean() - t[bins == b].mean()) * (bins == b).mean() for b in range(10) if (bins == b).any())
        gate_rows.append(dict(price=p, method=name, base_rate=t.mean(), AUC=roc_auc_score(t, pr),
                              Brier=brier_score_loss(t, pr), ECE=ece))
    o[f"Pdeploy@{p}"] = prod
    print("gate", p, round(time.time() - t0), "s", flush=True)
gate = pd.DataFrame(gate_rows)
gate.to_csv(os.path.join(RES, "exp10_purity_gate.csv"), index=False)
print(gate.round(4).to_string())
# scenario-level deployable shares
dep = []
for s, g in o.groupby("scenario"):
    r = dict(scenario=s)
    for p in [75, 100, 150]:
        r[f"true_deploy@{p}"] = ((g.pur96 == 1) & (g.y_inv >= 1 / p)).mean()
        r[f"pred_deploy@{p}"] = g[f"Pdeploy@{p}"].mean()
    dep.append(r)
pd.DataFrame(dep).to_csv(os.path.join(RES, "exp10_scenario_deployable.csv"), index=False)
print("done", round(time.time() - t0))
