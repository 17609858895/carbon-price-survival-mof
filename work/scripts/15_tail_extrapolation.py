"""Exp. 12: tail extrapolation of the RIV mean model (Mg-MOF-74 failure case).

Question: is the under-prediction of the best materials (Mg-MOF-74 is the true best in 115/132 scenarios but ranked ~100th)
caused by missing chemistry information or by the inability of piecewise-constant trees to extrapolate?
Variants of the pooled RIV mean model (same scenario encoding as Exp. 6):
  base       LightGBM, Exp. 6 features
  +node      + metal-node descriptors (coordination, open-metal-site fraction, cation properties; script 14)
  linear     LightGBM with linear leaves (piecewise-linear trees), standardized features
  linear+node
Protocols:
  CV    MOF-grouped 5-fold over all scenarios (accuracy must not degrade)
  tail  uptake-forward split: MOFs above the 97th percentile of CO2 uptake at 0.13 bar are held out, the rest train
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from common import DP, PHYS, RES, SCEN_CAT, add_physics_proxies, feature_list, fold_of, lgbp, load_long  # noqa: E402

warnings.filterwarnings("ignore")
spec = importlib.util.spec_from_file_location("atlas", os.path.join(os.path.dirname(__file__), "07_prospective_atlas.py"))
atlas = importlib.util.module_from_spec(spec)
spec.loader.exec_module(atlas)

NODE = ["metal_per_nm3", "cn_mean", "cn_min", "cn_deficit", "oms_frac", "oms_per_nm3", "d_M_lig_min", "d_MO_mean",
        "metal_z", "metal_r_ion", "metal_en", "metal_ionic_potential", "metal_grp_alkali", "metal_grp_alkaline_earth",
        "metal_grp_tm3d", "metal_grp_tm45d", "metal_grp_p_block", "metal_grp_f_block"]
YS = 100.0          # model y in t per 100 EUR so that linear-leaf ridge penalties act on O(1) coefficients
VARIANTS = os.environ.get("PRISMA_VARIANTS", "base,+node,linear,linear+node").split(",")


def load():
    k = load_long().reset_index(drop=True)
    node = pd.read_parquet(os.path.join(DP, "metal_node.parquet"))
    k = k.merge(node[["MOF"] + NODE], on="MOF", how="left")
    k = add_physics_proxies(k)
    levels = {c: sorted(k[c].unique()) for c in SCEN_CAT}
    return k, levels


def make_X(k, levels, v):
    feats = feature_list(k) + (NODE if "node" in v else []) + (PHYS if "phys" in v else [])
    return atlas.design(k, feats, levels)


def mono_vector(cols):
    """+1 for the proxy uptake / working capacity, -1 for the proxy regeneration energy, 0 otherwise."""
    return [1 if c in ("phys_q_ads", "phys_wc") else -1 if c == "phys_log_e" else 0 for c in cols]


def fit_predict(Xtr, ytr, Xte, linear, mono=False):
    p = lgbp()
    if mono:
        p.update(monotone_constraints=mono_vector(Xtr.columns), monotone_constraints_method="advanced")
    if linear:
        med, sd = Xtr.median(), Xtr.std().replace(0, 1)
        Xtr, Xte = ((Xtr.fillna(med) - med) / sd), ((Xte.fillna(med) - med) / sd)
        p.update(linear_tree=True, linear_lambda=1.0)
    m = lgb.LGBMRegressor(**p).fit(Xtr, ytr * YS)
    return m.predict(Xte) / YS


def rank_metrics(df, col):
    rows = []
    for s, g in df.groupby("scenario"):
        t = g.y_inv.values
        r = dict(spearman=spearmanr(t, g[col]).correlation)
        for kk in (10, 25, 50):
            r[f"P@{kk}"] = len(set(g.nlargest(kk, "y_inv").MOF) & set(g.nlargest(kk, col).MOF)) / kk
        lab = (t >= 0.01).astype(int)
        r["AUC@100"] = roc_auc_score(lab, g[col]) if 0 < lab.sum() < len(lab) else np.nan
        best = t.max()
        top = g.nlargest(10, col).y_inv.values
        r["regret_top1_pct"] = (best / top[0] - 1) * 100 if top[0] > 0 else np.inf
        r["regret_best10_pct"] = (best / top.max() - 1) * 100 if top.max() > 0 else np.inf
        tt = g.nlargest(5, "y_inv")
        r["top5_bias_rel"] = float(np.median((tt.y_inv - tt[col]) / tt.y_inv))   # >0: best materials under-predicted
        rows.append(r)
    return pd.DataFrame(rows).median()


if __name__ == "__main__":
    t0 = time.time()
    k, levels = load()
    k["fold"] = k.MOF.map(fold_of)
    up = k.drop_duplicates("MOF").set_index("MOF")["iso_CO2_0.13bar"]
    tail_mofs = set(up[up >= up.quantile(0.97)].index)
    tail = k.MOF.isin(tail_mofs).values
    print("rows", len(k), "tail MOFs", len(tail_mofs), "MgMOF74 in tail", "MgMOF74" in tail_mofs, flush=True)
    out_cv, out_tail, preds = [], [], k[["MOF", "scenario", "y_inv"]].copy()
    for v in VARIANTS:
        X = make_X(k, levels, v)
        lin = v.startswith("linear")
        y = k.y_inv.values
        oof = np.zeros(len(k))
        for f in range(5):
            tr, te = k.fold.values != f, k.fold.values == f
            oof[te] = fit_predict(X[tr], y[tr], X[te], lin, "mono" in v)
            print(v, "fold", f, round(time.time() - t0), "s", flush=True)
        preds[v] = oof
        r = rank_metrics(preds.assign(p=oof), "p")
        g = preds[preds.MOF == "MgMOF74"].assign(p=oof[(preds.MOF == "MgMOF74").values])
        rk = preds.assign(p=oof).groupby("scenario").p.rank(ascending=False)[g.index]
        r["MgMOF74_pred_rank_med"] = rk.median()
        r["MgMOF74_pred_frac100"] = (g.p >= 0.01).mean()
        r["MgMOF74_true_frac100"] = (g.y_inv >= 0.01).mean()
        r["MgMOF74_ratio_med"] = float(np.median(g.p / g.y_inv))
        r["variant"] = v
        out_cv.append(r)
        # uptake-forward tail split
        pt = fit_predict(X[~tail], y[~tail], X[tail], lin, "mono" in v)
        d = k.loc[tail, ["MOF", "scenario", "y_inv"]].assign(p=pt)
        via = d.y_inv > 0
        out_tail.append(dict(variant=v, n_mof=len(tail_mofs), bias_rel_viable=float(np.median((d.y_inv - d.p)[via] / d.y_inv[via])),
                             mae_x100=float(np.mean(np.abs(d.y_inv - d.p)) * 100),
                             spearman_within_tail=float(d.groupby("scenario").apply(lambda g: spearmanr(g.y_inv, g.p).correlation).median()),
                             frac100_true=float((d.y_inv >= 0.01).mean()), frac100_pred=float((d.p >= 0.01).mean()),
                             AUC100=roc_auc_score((d.y_inv >= 0.01).astype(int), d.p),
                             MgMOF74_ratio_med=float(np.median((d.p / d.y_inv)[d.MOF == "MgMOF74"]))))
        print(pd.DataFrame(out_cv).set_index("variant").round(4).T.to_string(), flush=True)
        print(pd.DataFrame(out_tail).set_index("variant").round(4).T.to_string(), flush=True)
    tag = "" if os.environ.get("PRISMA_FULL", "0") == "1" else "_fast"
    pd.DataFrame(out_cv).to_csv(os.path.join(RES, f"exp12_tail_cv{tag}.csv"), index=False)
    pd.DataFrame(out_tail).to_csv(os.path.join(RES, f"exp12_tail_forward{tag}.csv"), index=False)
    preds.to_parquet(os.path.join(RES, f"exp12_oof{tag}.parquet"), index=False)
    print("done", round(time.time() - t0), "s")
