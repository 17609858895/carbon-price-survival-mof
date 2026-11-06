"""Exp. 7: transfer to an unseen PROCESS CONFIGURATION (the layer that decides rank order).

Hold out a whole configuration family and predict it for unseen MOFs (MOF-grouped folds):
  LOPO : leave TVSA06 out (vacuum 0.6 bar lies between TVSA02 = 0.2 bar and TSA = no vacuum)
  LOSO : leave Coal out (flue CO2 ~13 %, between NGCC ~4 % and cement ~20 %)
Scenario encodings compared with the same learner (RIV target y = 1/nCAC):
  one-hot     : categorical source/process/humidity/region (cannot represent an unseen level)
  physics     : numeric descriptors (desorption pressure, flue CO2 fraction, wet flag) + regional one-hot +
                source-matched isotherm features (CO2 uptake at the flue partial pressure, selectivity proxy)
and k-shot adaptation with k anchor MOFs simulated in the held-out configuration.
The flue CO2 fractions and vacuum levels are taken from the scenario names / isotherm grid and must be
checked against PrISMa SI Table S-x before publication.
"""
from __future__ import annotations

import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from common import THREADS, lgbp, RES, feature_list, fold_of, load_long, safe_log  # noqa: E402

P = lgbp()
Y_CO2 = {"NGCC": 0.04, "Coal": 0.13, "Cement": 0.20}
P_DES = {"TSA": 1.0, "TVSA06": 0.6, "TVSA02": 0.2}

t0 = time.time()
k = load_long().reset_index(drop=True)
feats = feature_list(k)
k["fold"] = k.MOF.map(fold_of)
k["y_co2"] = k.source.map(Y_CO2)
k["p_des"] = k.process.map(P_DES)
k["wet"] = (k.humidity == "wet").astype(float)
grid = np.array([0.04, 0.13, 0.2])
iso = np.log(k[["iso_CO2_0.04bar", "iso_CO2_0.13bar", "iso_CO2_0.2bar"]].clip(lower=1e-6).values)
k["U_feed"] = np.exp([np.interp(np.log(y), np.log(grid), r) for y, r in zip(k.y_co2.values, iso)])
k["U_feed_over_N2"] = k.U_feed / k["iso_N2_0.8bar"].clip(lower=1e-6)
k["log_U_feed"] = safe_log(k.U_feed)

reg_oh = pd.get_dummies(k[["region"]], dtype=float)
oh_full = pd.get_dummies(k[["source", "process", "humidity", "region"]], dtype=float)
X_onehot = pd.concat([k[feats], oh_full], axis=1)
X_phys = pd.concat([k[feats + ["y_co2", "p_des", "wet", "log_U_feed", "U_feed_over_N2"]], reg_oh], axis=1)
for X in (X_onehot, X_phys):
    X.columns = [c.replace(" ", "_").replace(".", "p") for c in X.columns]
y = k.y_inv.values * 1000
rng = np.random.default_rng(0)
rows = []
for name, held in [("LOPO: TVSA06", k.process.values == "TVSA06"), ("LOSO: Coal", k.source.values == "Coal")]:
    for kshot in [0, 10, 50]:
        for f in range(5):
            test_mof = k.fold.values == f
            anchors = set()
            if kshot:
                pool = k.loc[held & ~test_mof, "MOF"].unique()
                anchors = set(rng.choice(pool, size=kshot, replace=False))
            tr = (~held & ~test_mof) | (held & k.MOF.isin(anchors).values)
            te = held & test_mof
            for enc, X in [("one-hot", X_onehot), ("physics", X_phys)]:
                p = lgb.LGBMRegressor(**P).fit(X[tr], y[tr]).predict(X[te])
                kt = k[te].assign(p=p)
                sp = kt.groupby("scenario").apply(lambda g: spearmanr(g.y_inv, g.p).correlation).median()
                top = kt.groupby("scenario").apply(
                    lambda g: len(set(g.nlargest(25, "y_inv").MOF) & set(g.nlargest(25, "p").MOF)) / 25).median()
                t = (kt.y_inv.values >= 0.01).astype(int)
                rows.append(dict(split=name, k=kshot, fold=f, encoding=enc, R2=r2_score(y[te], p),
                                 median_scenario_spearman=sp, P_at_25=top,
                                 AUC100=roc_auc_score(t, p) if 0 < t.sum() < len(t) else np.nan))
            print(name, kshot, f, round(time.time() - t0), "s", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(RES, "exp7_config_transfer.csv"), index=False)
res = pd.DataFrame(rows)
print(res.groupby(["split", "k", "encoding"])[["R2", "median_scenario_spearman", "P_at_25", "AUC100"]].mean().round(3).to_string())
print("done", round(time.time() - t0))
