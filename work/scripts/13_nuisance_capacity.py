"""Exp. 11: sensitivity of CATE learners to nuisance-model capacity (humidity design, confounded gamma = 2).

Re-uses the learners and design of Exp. 5 (06_causal_ground_truth.py); only the LightGBM tree count of all
nuisance/outcome models changes. Motivation: the exploratory run (300 trees) gave R-learner PEHE 4.9, the
full run (750 trees) 1.1.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import RES, feature_list, fold_of, lgbp, load_long, safe_log  # noqa: E402

spec = importlib.util.spec_from_file_location("causal", os.path.join(HERE, "06_causal_ground_truth.py"))
causal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(causal)

t0 = time.time()
k = load_long()
feats = feature_list(k)
d = causal.paired(k, "humidity", "wet", "dry", ["source", "process", "region", "boiler", "heat_extraction"])
d = d[(d.region == "UK") & (d.boiler == "default") & (d.heat_extraction == "True")].reset_index(drop=True)
ctx = pd.get_dummies(d[["source", "process"]].astype(str), dtype=float)
X = np.c_[d[feats].fillna(d[feats].median()).values, ctx.values]
z = np.c_[safe_log(d["w_H"].values), d["Q_CO2"].values]
z = np.nan_to_num((z - np.nanmean(z, 0)) / np.nanstd(z, 0))
e = 1 / (1 + np.exp(-2.0 * (z[:, 0] - z[:, 1])))
rng = np.random.default_rng(1)
T = rng.binomial(1, e)
Y = np.where(T == 1, d.y1.values, d.y0.values)
fo = d.MOF.map(fold_of).values
rows = []
for n_trees in [100, 300, 750, 1500]:
    causal.LGB = lgbp(num_leaves=31, min_child_samples=20, n_estimators=n_trees, learning_rate=0.05)
    for f in range(5):
        tr, te = fo != f, fo == f
        est = causal.learners(X[tr], T[tr], Y[tr], X[te], e[tr])
        tau = d.ite.values[te]
        for n, v in est.items():
            if not n.startswith("_"):
                rows.append(dict(n_trees=n_trees, fold=f, estimator=n, PEHE=np.sqrt(np.mean((v - tau) ** 2)),
                                 spearman=spearmanr(v, tau).correlation))
    print(n_trees, round(time.time() - t0), "s", flush=True)
res = pd.DataFrame(rows)
res.to_csv(os.path.join(RES, "exp11_nuisance_capacity.csv"), index=False)
print(res.groupby(["estimator", "n_trees"]).PEHE.mean().unstack().round(3).to_string())
