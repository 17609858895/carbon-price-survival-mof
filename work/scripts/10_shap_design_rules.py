"""Exp. 8: interpretation (hypothesis-generating only).

a) TreeSHAP of the pooled RIV model (y = 1/nCAC) on a stratified sample: global ranking and
   material x scenario interaction (how the importance of each descriptor changes by CO2 source).
b) Drivers of the TRUE humidity penalty (observed paired ITE, wet - dry): LightGBM fitted on material
   descriptors with MOF-grouped CV R2, then TreeSHAP. Because the ITE is observed, this explains
   simulated effect heterogeneity directly instead of estimating it.
"""
from __future__ import annotations

import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import r2_score

sys.path.insert(0, os.path.dirname(__file__))
from common import THREADS, lgbp, RES, feature_list, fold_of, load_long, one_hot_scen  # noqa: E402

P = lgbp()
k = load_long().reset_index(drop=True)
feats = feature_list(k)
X = pd.concat([k[feats], one_hot_scen(k)], axis=1)
X.columns = [c.replace(" ", "_").replace(".", "p") for c in X.columns]
y = k.y_inv.values * 1000
m = lgb.LGBMRegressor(**P).fit(X, y)
idx = k.groupby("source", group_keys=False).apply(lambda g: g.sample(min(len(g), 2500), random_state=0)).index
sv = shap.TreeExplainer(m).shap_values(X.loc[idx])
S = pd.DataFrame(sv, columns=X.columns, index=idx)
S.assign(source=k.loc[idx, "source"].values).to_parquet(os.path.join(RES, "exp8_shap_riv.parquet"))
X.loc[idx].to_parquet(os.path.join(RES, "exp8_shap_riv_X.parquet"))
imp = S.abs().groupby(k.loc[idx, "source"].values).mean().T
imp["all"] = S.abs().mean()
imp.sort_values("all", ascending=False).to_csv(os.path.join(RES, "exp8_shap_importance_by_source.csv"))
print(imp.sort_values("all", ascending=False).head(15).round(3).to_string())

ite = pd.read_parquet(os.path.join(RES, "exp5_true_ite_humidity.parquet"))
mat = load_long()[["MOF", "source"] + feats].drop_duplicates(["MOF", "source"])
d = ite.merge(mat, on=["MOF", "source"], how="left")
Xi = pd.concat([d[feats], pd.get_dummies(d[["source", "process"]], dtype=float)], axis=1)
Xi.columns = [c.replace(" ", "_").replace(".", "p") for c in Xi.columns]
yi = d.ite.values
fo = d.MOF.map(fold_of).values
oof = np.zeros(len(d))
for f in range(5):
    oof[fo == f] = lgb.LGBMRegressor(**P).fit(Xi[fo != f], yi[fo != f]).predict(Xi[fo == f])
print("humidity-penalty model MOF-grouped CV R2:", round(r2_score(yi, oof), 3))
mi = lgb.LGBMRegressor(**P).fit(Xi, yi)
svi = shap.TreeExplainer(mi).shap_values(Xi)
Si = pd.DataFrame(svi, columns=Xi.columns)
Si.to_parquet(os.path.join(RES, "exp8_shap_humidity_ite.parquet"))
Xi.to_parquet(os.path.join(RES, "exp8_shap_humidity_ite_X.parquet"))
pd.Series({"cv_r2": r2_score(yi, oof)}).to_csv(os.path.join(RES, "exp8_humidity_model_cv.csv"))
print(Si.abs().mean().sort_values(ascending=False).head(12).round(4).to_string())
