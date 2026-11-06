"""Exp. 1b: decision-focused variants of the reciprocal-viability (RIV) learner.

Same features / MOF folds / panel as Exp. 1. Variants:
  RIV            : L2 regression on y = 1/nCAC
  RIV-band       : L2 on y with weights emphasising the policy price band (40-300 EUR/t, i.e. y in [1/300, 1/40])
  RIV-rank       : LambdaRank with one query per scenario, graded relevance from within-scenario y quantiles
                   (non-viable pairs get grade 0), linear gains and ranking truncation level 60
  RIV+rank blend : average of within-scenario rank-normalised RIV and RIV-rank scores

Response-derived relevance grades are computed for each complete scenario before
the MOF folds are fitted; this is the retrospective exploratory comparison.
"""
from __future__ import annotations

import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, os.path.dirname(__file__))
from common import RES, feature_list, fold_of, load_long, one_hot_scen, topk_metrics, viability_price_metrics  # noqa: E402

t0 = time.time()
k = load_long().sort_values(["scenario", "MOF"]).reset_index(drop=True)
feats = feature_list(k)
X = pd.concat([k[feats].reset_index(drop=True), one_hot_scen(k).reset_index(drop=True)], axis=1)
X.columns = [c.replace(" ", "_").replace(".", "p") for c in X.columns]
y = k.y_inv.values
folds = k.MOF.map(fold_of).values
P = dict(n_estimators=500, learning_rate=0.08, num_leaves=63, min_child_samples=40, subsample=0.8, subsample_freq=1,
         colsample_bytree=0.7, reg_lambda=1.0, verbose=-1, n_jobs=4)

# graded relevance within scenario: viable pairs binned into 30 quantile grades of y, non-viable = 0
q = k.groupby("scenario").y_inv.rank(pct=True).values
grade = np.where(y > 0, 1 + np.floor(q * 30).clip(0, 29), 0).astype(int)
band_w = np.where((y >= 1 / 300) & (y <= 1 / 40), 3.0, 1.0)

pred = {m: np.zeros(len(k)) for m in ["RIV", "RIV-band", "RIV-rank"]}
for f in range(5):
    tr, te = folds != f, folds == f
    pred["RIV"][te] = lgb.LGBMRegressor(**P).fit(X[tr], y[tr]).predict(X[te])
    pred["RIV-band"][te] = lgb.LGBMRegressor(**P).fit(X[tr], y[tr], sample_weight=band_w[tr]).predict(X[te])
    grp = k[tr].groupby("scenario", sort=False).size().values  # rows are sorted by scenario
    rk = lgb.LGBMRanker(**P, objective="lambdarank", label_gain=list(range(31)), lambdarank_truncation_level=60)
    rk.fit(X[tr], grade[tr], group=grp, eval_at=[50])
    pred["RIV-rank"][te] = rk.predict(X[te])
    print("fold", f, round(time.time() - t0), "s", flush=True)

sc = k.scenario.values
rn = lambda s: pd.Series(s).groupby(sc).rank(pct=True).values  # noqa: E731
pred["RIV+rank blend"] = 0.5 * rn(pred["RIV"]) + 0.5 * rn(pred["RIV-rank"])
rows = []
for name, s in pred.items():
    df = k[["MOF", "scenario", "y_inv"]].copy()
    df["score"] = s
    tk = topk_metrics(df, "score")
    r = {"variant": name}
    r.update({c: tk[c].median() for c in ["spearman", "spearman_viable_nCAC", "P@10", "P@25", "P@50",
                                         "regret_top1_pct", "regret_best_of_top5_pct"]})
    r["regret_top1_mean"] = tk["regret_top1_pct"].replace(np.inf, np.nan).mean()
    if name in ("RIV", "RIV-band"):
        vm = viability_price_metrics(y, s)
        r.update({kk: v for kk, v in vm.items() if kk in ("AUC@100", "F1@100", "F1@75", "F1@150")})
    rows.append(r)
res = pd.DataFrame(rows)
res.to_csv(os.path.join(RES, "exp1b_decision_focused.csv"), index=False)
pd.set_option("display.width", 220)
print(res.round(4).to_string())
print("done", round(time.time() - t0))
