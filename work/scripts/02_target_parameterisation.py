"""Exp. 1: how should the singular ratio KPI nCAC be learned?

Compares target parameterisations with the same LightGBM learner, same features and the same
MOF-grouped 5-fold split (all scenarios of a MOF stay in one fold):
  direct      : nCAC (MSE)
  signed_log  : sign(nCAC) * log10(1+|nCAC|)
  hurdle      : P(viable) classifier + log10(nCAC) regressor on viable rows
  inverse     : y = 1/nCAC  (net avoided t CO2 per $, continuous through the singularity)
Decision metrics: viability-at-price AUC/F1, per-scenario Spearman, top-k precision, regret.
"""
from __future__ import annotations

import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(__file__))
from common import THREADS, lgbp, RES, feature_list, fold_of, load_long, one_hot_scen, topk_metrics, viability_price_metrics  # noqa: E402

t0 = time.time()
k = load_long()
feats = feature_list(k)
X = pd.concat([k[feats].reset_index(drop=True), one_hot_scen(k).reset_index(drop=True)], axis=1)
X.columns = [c.replace(" ", "_").replace(".", "p") for c in X.columns]
groups = k.MOF.values
y_inv = k.y_inv.values
ncac = k.nCAC.values
viable = k.viable.values
print("rows", len(k), "features", X.shape[1])

P = lgbp()
SEEDS = [int(x) for x in os.environ.get("PRISMA_SEEDS", "0").split(",")]
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402


def run(seed: int, targets: list[str]) -> dict:
    folds = k.MOF.map(lambda m: fold_of(m, seed=seed)).values
    pred = {m: np.zeros(len(k)) for m in targets}
    for f in range(5):
        tr, te = np.where(folds != f)[0], np.where(folds == f)[0]
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        if "direct" in targets:
            pd_ = lgb.LGBMRegressor(**P).fit(Xtr, ncac[tr]).predict(Xte)
            pred["direct"][te] = 1.0 / np.where(np.abs(pd_) < 1e-6, 1e-6, pd_)
        if "signed_log" in targets:
            sl = np.sign(ncac) * np.log10(1 + np.abs(ncac))
            p = lgb.LGBMRegressor(**P).fit(Xtr, sl[tr]).predict(Xte)
            pred["signed_log"][te] = 1.0 / (np.sign(p) * (10 ** np.abs(p) - 1) + 1e-9)
        if "hurdle" in targets:
            pv = lgb.LGBMClassifier(**P).fit(Xtr, viable[tr]).predict_proba(Xte)[:, 1]
            vtr = tr[viable[tr] == 1]
            lp = lgb.LGBMRegressor(**P).fit(X.iloc[vtr], np.log10(ncac[vtr])).predict(Xte)
            pred["hurdle"][te] = np.where(pv >= 0.5, 10 ** (-lp), -(1 - pv) * 1e-3)  # viable ranked by cost, others last
        if "inverse" in targets:
            pred["inverse"][te] = lgb.LGBMRegressor(**P).fit(Xtr, y_inv[tr]).predict(Xte)
        print("seed", seed, "fold", f, round(time.time() - t0, 1), "s", flush=True)
    return pred


rows = []
for seed in SEEDS:
    targets = ["direct", "signed_log", "hurdle", "inverse"] if seed == SEEDS[0] else ["hurdle", "inverse"]
    pred = run(seed, targets)
    if seed == SEEDS[0]:
        np.savez_compressed(os.path.join(RES, "exp1_oof_scores.npz"), **pred)
    for name, s in pred.items():
        df = k[["MOF", "scenario", "y_inv"]].copy()
        df["score"] = s
        tk = topk_metrics(df, "score")
        r = {"seed": seed, "target": name, "R2_yinv": r2_score(y_inv, np.clip(s, -0.05, 0.05)),
             "spearman_pooled": spearmanr(y_inv, s).correlation}
        r.update({c: tk[c].median() for c in ["spearman", "spearman_viable_nCAC", "P@10", "P@25", "P@50",
                                             "regret_top1_pct", "regret_best_of_top5_pct"]})
        r.update(viability_price_metrics(y_inv, s))
        rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "exp1_target_parameterisation_seeds.csv"), index=False)
res = pd.DataFrame(rows)
summ = res.groupby("target").agg(["mean", "std"])
summ.columns = [f"{a}_{b}" for a, b in summ.columns]
summ = summ.drop(columns=[c for c in summ.columns if c.startswith("seed_")])
summ.to_csv(os.path.join(RES, "exp1_target_parameterisation_summary.csv"))
# first-seed table keeps the original file layout used by the figure script
res[res.seed == SEEDS[0]].drop(columns="seed").to_csv(os.path.join(RES, "exp1_target_parameterisation.csv"), index=False)
pd.set_option("display.width", 250)
print(summ[[c for c in summ.columns if c.split("_")[0] in ("spearman", "P@25", "P@50", "F1@100", "AUC@100")]].round(4).to_string())
print("done", round(time.time() - t0, 1))
