"""Exp. 2: 20-algorithm benchmark + nested stacking on single-scenario tasks, versus a pooled
scenario-conditioned learner trained on all scenarios (multi-task sharing).

Target: y_inv = 1/nCAC (scaled to t CO2 per k$). MOF folds are fixed by hashing the MOF id so that
the same MOF is in the same fold in every scenario and in the pooled model.
"""
from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import (AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor,
                              HistGradientBoostingRegressor, RandomForestRegressor, StackingRegressor)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.impute import SimpleImputer
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import BayesianRidge, ElasticNet, HuberRegressor, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

sys.path.insert(0, os.path.dirname(__file__))
from common import RES, feature_list, fold_of, load_long, one_hot_scen  # noqa: E402

warnings.filterwarnings("ignore")
SCALE = 1000.0
SCENARIOS = ["Cement_Storage_UK_TVSA02-wet", "NGCC-onshore_Storage_UK_TVSA06-wet"]


def lin(est):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), est)


def kern(est):
    return make_pipeline(SimpleImputer(strategy="median"), QuantileTransformer(n_quantiles=200, output_distribution="normal"),
                         StandardScaler(), est)


def tree(est):
    return make_pipeline(SimpleImputer(strategy="median"), est)


def models():
    return {
        "Ridge": lin(Ridge(alpha=1.0)),
        "Lasso": lin(Lasso(alpha=1e-3, max_iter=20000)),
        "ElasticNet": lin(ElasticNet(alpha=1e-3, l1_ratio=0.5, max_iter=20000)),
        "BayesRidge": lin(BayesianRidge()),
        "Huber": lin(HuberRegressor(max_iter=2000)),
        "PLS": lin(PLSRegression(n_components=10)),
        "KNN": kern(KNeighborsRegressor(n_neighbors=10, weights="distance")),
        "SVR": kern(SVR(C=3.0, epsilon=0.05, gamma="scale")),
        "KRR": kern(KernelRidge(alpha=0.3, kernel="rbf", gamma=0.02)),
        "GPR": kern(GaussianProcessRegressor(ConstantKernel() * Matern(nu=2.5, length_scale=5.0) + WhiteKernel(0.1),
                                             normalize_y=True, n_restarts_optimizer=0)),
        "DT": tree(DecisionTreeRegressor(min_samples_leaf=10, max_depth=10)),
        "RF": tree(RandomForestRegressor(n_estimators=300, min_samples_leaf=2, max_features=0.5, n_jobs=2)),
        "ExtraTrees": tree(ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, max_features=0.6, n_jobs=2)),
        "GBR": tree(GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8)),
        "HistGB": HistGradientBoostingRegressor(max_iter=600, learning_rate=0.04, l2_regularization=1.0),
        "AdaBoost": tree(AdaBoostRegressor(DecisionTreeRegressor(max_depth=6), n_estimators=150, learning_rate=0.05)),
        "XGBoost": xgb.XGBRegressor(n_estimators=700, learning_rate=0.03, max_depth=5, subsample=0.8,
                                    colsample_bytree=0.7, reg_lambda=1.0, n_jobs=2),
        "LightGBM": lgb.LGBMRegressor(n_estimators=700, learning_rate=0.03, num_leaves=31, min_child_samples=15,
                                      subsample=0.8, subsample_freq=1, colsample_bytree=0.7, verbose=-1, n_jobs=2),
        "CatBoost": CatBoostRegressor(iterations=500, learning_rate=0.06, depth=6, verbose=0, thread_count=2),
        "MLP": kern(MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-3, learning_rate_init=1e-3, max_iter=800,
                                 early_stopping=True, random_state=0)),
    }


def metrics(y, p):
    t100 = (y >= SCALE / 100).astype(int)
    out = dict(R2=r2_score(y, p), RMSE=mean_squared_error(y, p) ** 0.5, Spearman=spearmanr(y, p).correlation,
               AUC100=roc_auc_score(t100, p) if 0 < t100.sum() < len(t100) else np.nan)
    top = set(np.argsort(-y)[:25]); topp = set(np.argsort(-p)[:25])
    out["P@25"] = len(top & topp) / 25
    return out


if __name__ == "__main__":
    t0 = time.time()
    k = load_long()
    feats = feature_list(k)
    k["fold"] = k.MOF.map(fold_of)
    rows, oof_store, timing = [], {}, {}
    for sc in SCENARIOS:
        d = k[k.scenario == sc].reset_index(drop=True)
        X, y, fold = d[feats].values, d.y_inv.values * SCALE, d.fold.values
        base = models()
        preds = {n: np.zeros(len(d)) for n in list(base) + ["Stacking(NNLS)"]}
        for f in range(5):
            tr, te = fold != f, fold == f
            for n, est in base.items():
                t1 = time.time()
                m = clone(est).fit(X[tr], y[tr])
                preds[n][te] = np.ravel(m.predict(X[te]))
                timing.setdefault(n, []).append(time.time() - t1)
            print(sc, "fold", f, round(time.time() - t0), "s", flush=True)
        B = np.column_stack([preds[n] for n in base])
        for f in range(5):
            tr, te = fold != f, fold == f
            meta = LinearRegression(positive=True).fit(B[tr], y[tr])
            preds["Stacking(NNLS)"][te] = meta.predict(B[te])
        meta_all = LinearRegression(positive=True).fit(B, y)
        pd.Series(meta_all.coef_, index=list(base)).to_csv(os.path.join(RES, f"exp2_stack_weights_{sc}.csv"))
        for n, p in preds.items():
            r = dict(scenario=sc, model=n); r.update(metrics(y, p)); rows.append(r)
        oof_store[sc] = pd.DataFrame(preds).assign(MOF=d.MOF, y=y)
        pd.DataFrame(rows).to_csv(os.path.join(RES, "exp2_benchmark20_stacking.csv"), index=False)

    # pooled scenario-conditioned LightGBM trained on ALL scenarios (same MOF folds), evaluated on the same scenarios
    X = pd.concat([k[feats].reset_index(drop=True), one_hot_scen(k).reset_index(drop=True)], axis=1)
    X.columns = [c.replace(" ", "_").replace(".", "p") for c in X.columns]
    yall = k.y_inv.values * SCALE
    pooled = np.zeros(len(k))
    for f in range(5):
        tr, te = k.fold.values != f, k.fold.values == f
        m = lgb.LGBMRegressor(n_estimators=1200, learning_rate=0.04, num_leaves=63, min_child_samples=40,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.7, verbose=-1, n_jobs=-1)
        m.fit(X[tr], yall[tr])
        pooled[te] = m.predict(X[te])
    k["pooled"] = pooled
    for sc in SCENARIOS:
        d = k[k.scenario == sc]
        r = dict(scenario=sc, model="Pooled-LightGBM(all scenarios)"); r.update(metrics(d.y_inv.values * SCALE, d.pooled.values))
        rows.append(r)
    pd.DataFrame({n: [np.mean(v)] for n, v in timing.items()}).T.rename(columns={0: "fit_s"}).to_csv(
        os.path.join(RES, "exp2_fit_time.csv"))
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(RES, "exp2_benchmark20_stacking.csv"), index=False)
    k[["MOF", "scenario", "y_inv", "pooled"]].to_parquet(os.path.join(RES, "exp2_pooled_oof.parquet"), index=False)
    for sc, df in oof_store.items():
        df.to_parquet(os.path.join(RES, f"exp2_oof_{sc}.parquet"), index=False)
    summ = res.groupby("model")[["R2", "RMSE", "Spearman", "AUC100", "P@25"]].mean().sort_values("R2", ascending=False)
    print(summ.round(3).to_string())
    print("done", round(time.time() - t0), "s")
