"""Exp. 9: input tiers (what must be computed before the surrogate can be used?).

Tier S  structure-only : CIF composition/charge descriptors + Zeo++ geometry (density, POAVF) + ML heat capacity
Tier A  + adsorption   : + simulated CO2/N2 Henry, uptakes, heats, isotherm points (GCMC)
Tier W  + water        : + water isotherm / competitive CO2 uptake / WRC (full feature set used elsewhere)
Same RIV learner, MOF folds and 54-scenario panel. Also reports the original-paper classification task
(Cement/UK/TVSA06-wet, nCAC <= 105/80/70 EUR/t) for each tier.
"""
from __future__ import annotations

import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, matthews_corrcoef, r2_score, roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from common import THREADS, lgbp, CIF_MAT, RES, feature_list, fold_of, load_long, one_hot_scen  # noqa: E402

P = lgbp()
k = load_long().reset_index(drop=True)
full = feature_list(k)
water = ["w_Usat", "log_w_H", "w_Qh", "w_Udes", "w_Qdes", "w_UCO2bin", "log_w_UCO2ter", "log_w_WRC"]
tiers = {"S: structure only": [c for c in CIF_MAT if c in k] + ["density", "POAVF", "Cp"]}
tiers["A: + adsorption (GCMC)"] = [c for c in full if c not in water]
tiers["W: + water (full)"] = full
y = k.y_inv.values
fo = k.MOF.map(fold_of).values
sc = one_hot_scen(k)
rows, cls = [], []
for name, cols in tiers.items():
    X = pd.concat([k[cols], sc], axis=1)
    X.columns = [c.replace(" ", "_").replace(".", "p") for c in X.columns]
    p = np.zeros(len(k))
    for f in range(5):
        p[fo == f] = lgb.LGBMRegressor(**P).fit(X[fo != f], y[fo != f]).predict(X[fo == f])
    d = k.assign(p=p)
    sp = d.groupby("scenario").apply(lambda g: spearmanr(g.y_inv, g.p).correlation).median()
    p25 = d.groupby("scenario").apply(lambda g: len(set(g.nlargest(25, "y_inv").MOF) & set(g.nlargest(25, "p").MOF)) / 25).median()
    rows.append(dict(tier=name, n_features=len(cols), R2=r2_score(y, p), median_scenario_spearman=sp, P_at_25=p25,
                     AUC100=roc_auc_score((y >= 0.01).astype(int), p)))
    g = d[d.scenario == "Cement_Storage_UK_TVSA06-wet"]
    for thr in [105, 80, 70]:
        t = ((g.nCAC > 0) & (g.nCAC <= thr)).astype(int)
        q = (g.p >= 1 / thr).astype(int)
        cls.append(dict(tier=name, thr=thr, acc=accuracy_score(t, q), MCC=matthews_corrcoef(t, q)))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(os.path.join(RES, "exp9_input_tiers.csv"), index=False)
pd.DataFrame(cls).to_csv(os.path.join(RES, "exp9_input_tiers_classification.csv"), index=False)
print(pd.DataFrame(rows).round(3).to_string())
print(pd.DataFrame(cls).round(3).to_string())
