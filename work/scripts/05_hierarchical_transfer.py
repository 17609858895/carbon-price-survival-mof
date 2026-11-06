"""Exp. 4: platform-structured hierarchical surrogate and few-shot transfer to an unseen region.

PrISMa computes   material --(process sim, 18 configs)--> process KPIs --(TEA/LCA, region)--> nCAC.
Process KPIs are identical across regions within a config (verified: within-config CV = 0).

Part A  error budget (MOF-grouped 5-fold):
  Direct      : material + scenario -> y_inv
  Oracle-B    : TRUE process KPIs + scenario -> y_inv   (upper bound of the TEA layer emulator)
  Hier-true   : stage A (material+cfg -> process KPIs) then stage B trained on true KPIs
  Hier-oof    : stage B trained on inner out-of-fold stage-A predictions (noise-matched composition)
  Hier+direct : direct features augmented with stage-A predicted process KPIs
Part B  leave-one-region-out (LORO) few-shot transfer. Region R is unseen except k anchor MOFs
  (fully evaluated in R). Test = remaining MOFs in R, whose process KPIs / other-region outcomes are known
  (realistic: materials already screened elsewhere, new deployment region).
  Direct-kshot : pooled LightGBM (region one-hot) with the k anchors' R rows added to training
  Donor-ridge  : per R-scenario ridge of y_inv on the same-config donor-scenario outcomes (synthetic scenario)
  StageB-kshot : oracle stage B fitted on other regions + k anchors (region one-hot)
"""
from __future__ import annotations

import os
import sys
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(__file__))
from common import THREADS, lgbp, PROC_OUT, RES, feature_list, fold_of, load_long, safe_log  # noqa: E402

warnings.filterwarnings("ignore")
LGB = lgbp(min_child_samples=30)
SCALE = 1000.0


def proc_targets(df):
    t = pd.DataFrame({"purity": df.purity, "recovery": df.recovery, "log_wc": safe_log(df.working_capacity + 1e-4),
                      "productivity": df.productivity, "log_spec_heat": safe_log(df.spec_heat),
                      "el_energy": df.el_energy * 100, "rho_b": df.rho_b, "log_CO2_capt": safe_log(df.CO2_captured)})
    return t


def cat_codes(df, cols):
    return pd.get_dummies(df[cols].astype(str), dtype=float)


def summarize(name, d, pred):
    y = d.y_inv.values * SCALE
    sp = d.assign(p=pred).groupby("scenario").apply(lambda g: spearmanr(g.y_inv, g.p).correlation).median()
    v = d.y_inv.values >= 0.01  # viable at 100 $/t
    from sklearn.metrics import roc_auc_score
    return dict(model=name, R2=r2_score(y, pred), median_scenario_spearman=sp, AUC100=roc_auc_score(v, pred))


if __name__ == "__main__":
    t0 = time.time()
    k = load_long().reset_index(drop=True)
    feats = feature_list(k)
    k["fold"] = k.MOF.map(fold_of)
    PT = proc_targets(k)
    ptc = list(PT.columns)
    for c in ptc:
        k["pt_" + c] = PT[c].values
    cfg_oh = cat_codes(k, ["source", "process", "humidity"])
    tea_oh = cat_codes(k, ["region", "boiler", "heat_extraction"])
    Xmat = k[feats].reset_index(drop=True)
    X_direct = pd.concat([Xmat, cfg_oh, tea_oh], axis=1)
    X_direct.columns = [c.replace(" ", "_").replace(".", "p") for c in X_direct.columns]
    y = k.y_inv.values * SCALE
    # unique (MOF,cfg) rows for stage A
    ua = k.drop_duplicates(["MOF", "cfg"]).reset_index(drop=True)
    XA = pd.concat([ua[feats].reset_index(drop=True), cat_codes(ua, ["source", "process", "humidity"])], axis=1)
    XA.columns = [c.replace(" ", "_").replace(".", "p") for c in XA.columns]
    YA = ua[["pt_" + c for c in ptc]].values
    key = k.MOF + "||" + k.cfg
    ukey = ua.MOF + "||" + ua.cfg
    pos = pd.Series(np.arange(len(ua)), index=ukey.values)
    row2a = pos.loc[key.values].values

    def stageB_X(proc_mat, rows):
        df = pd.DataFrame(proc_mat, columns=["pt_" + c for c in ptc])
        extra = k.loc[rows, ["density", "Cp"]].reset_index(drop=True)
        return pd.concat([df, extra, cfg_oh.loc[rows].reset_index(drop=True), tea_oh.loc[rows].reset_index(drop=True)], axis=1)

    preds = {m: np.zeros(len(k)) for m in ["Direct", "Oracle-B", "Hier-true", "Hier-oof", "Hier+direct"]}
    procA_r2 = []
    for f in range(5):
        trA, teA = ua.fold.values != f, ua.fold.values == f
        # stage A: fit on train MOFs; OOF predictions on train MOFs by inner GroupKFold
        A_te = np.zeros((teA.sum(), YA.shape[1]))
        A_oof = np.zeros((trA.sum(), YA.shape[1]))
        Xa_tr, Xa_te, g_tr = XA[trA], XA[teA], ua.MOF.values[trA]
        for j in range(YA.shape[1]):
            A_te[:, j] = lgb.LGBMRegressor(**LGB).fit(Xa_tr, YA[trA, j]).predict(Xa_te)
            for a, b in GroupKFold(2).split(Xa_tr, groups=g_tr):
                A_oof[b, j] = lgb.LGBMRegressor(**lgbp(0.5, min_child_samples=30)).fit(Xa_tr.iloc[a], YA[trA][a, j]).predict(Xa_tr.iloc[b])
        procA_r2.append({c: r2_score(YA[teA, j], A_te[:, j]) for j, c in enumerate(ptc)})
        A_pred_full = np.full_like(YA, np.nan, dtype=float)
        A_pred_full[teA] = A_te
        A_pred_full[trA] = A_oof
        tr, te = k.fold.values != f, k.fold.values == f
        rtr, rte = np.where(tr)[0], np.where(te)[0]
        # direct
        preds["Direct"][te] = lgb.LGBMRegressor(**LGB).fit(X_direct[tr], y[tr]).predict(X_direct[te])
        # stage B on true process KPIs
        B_true = lgb.LGBMRegressor(**LGB).fit(stageB_X(YA[row2a[rtr]], rtr), y[tr])
        preds["Oracle-B"][te] = B_true.predict(stageB_X(YA[row2a[rte]], rte))
        preds["Hier-true"][te] = B_true.predict(stageB_X(A_pred_full[row2a[rte]], rte))
        B_oof = lgb.LGBMRegressor(**LGB).fit(stageB_X(A_pred_full[row2a[rtr]], rtr), y[tr])
        preds["Hier-oof"][te] = B_oof.predict(stageB_X(A_pred_full[row2a[rte]], rte))
        Xaug_tr = pd.concat([X_direct[tr].reset_index(drop=True), pd.DataFrame(A_pred_full[row2a[rtr]], columns=ptc)], axis=1)
        Xaug_te = pd.concat([X_direct[te].reset_index(drop=True), pd.DataFrame(A_pred_full[row2a[rte]], columns=ptc)], axis=1)
        preds["Hier+direct"][te] = lgb.LGBMRegressor(**LGB).fit(Xaug_tr, y[tr]).predict(Xaug_te)
        print("fold", f, round(time.time() - t0), "s", flush=True)
    resA = pd.DataFrame([summarize(n, k, p) for n, p in preds.items()])
    pd.DataFrame(procA_r2).mean().to_csv(os.path.join(RES, "exp4_stageA_process_R2.csv"))
    resA.to_csv(os.path.join(RES, "exp4_hierarchical_error_budget.csv"), index=False)
    pd.DataFrame(preds).assign(MOF=k.MOF, scenario=k.scenario, y_inv=k.y_inv).to_parquet(
        os.path.join(RES, "exp4_oof_preds.parquet"), index=False)
    print(resA.round(4).to_string()); print(pd.DataFrame(procA_r2).mean().round(3))

    # ---------------------------------------------------------------- Part B: LORO few-shot
    rng = np.random.default_rng(0)
    rowsB = []
    piv = k.pivot_table(index="MOF", columns="scenario", values="y_inv")
    scen_info = k.drop_duplicates("scenario").set_index("scenario")
    for R in ["US", "CN-SD", "CH", "UK2022"]:
        in_R = k.region.values == R
        mofs_R = k.loc[in_R, "MOF"].unique()
        for kshot in [0, 10, 50]:
            for rep in range(2):
                anchors = set(rng.choice(mofs_R, size=kshot, replace=False)) if kshot else set()
                is_anchor = k.MOF.isin(anchors).values
                train = (~in_R) | (in_R & is_anchor)
                test = in_R & ~is_anchor
                out = {}
                # Direct few-shot (pooled LightGBM incl. material features)
                m = lgb.LGBMRegressor(**lgbp(0.7, min_child_samples=30)).fit(X_direct[train], y[train])
                out["Direct-kshot"] = m.predict(X_direct[test])
                # Stage-B few-shot on true process KPIs (process KPIs known from other regions)
                rtr, rte = np.where(train)[0], np.where(test)[0]
                mB = lgb.LGBMRegressor(**lgbp(0.7, min_child_samples=30)).fit(stageB_X(YA[row2a[rtr]], rtr), y[train])
                out["StageB-kshot"] = mB.predict(stageB_X(YA[row2a[rte]], rte))
                # Donor ridge (synthetic scenario): y_R,s(m) ~ sum_j w_j y_j(m) over same-config donor scenarios
                if kshot >= 5:
                    pr = np.full(test.sum(), np.nan)
                    kt = k[test]
                    for s in kt.scenario.unique():
                        cfg = scen_info.loc[s, "cfg"]
                        donors = [d for d in piv.columns if d != s and scen_info.loc[d, "cfg"] == cfg
                                  and scen_info.loc[d, "region"] != R]
                        A = piv.loc[list(anchors), donors].fillna(0).values
                        b = piv.loc[list(anchors), s].values
                        okb = ~np.isnan(b)
                        if okb.sum() < 3:
                            continue
                        reg = RidgeCV(alphas=np.logspace(-6, 1, 15)).fit(A[okb], b[okb])
                        sel = (kt.scenario.values == s)
                        pr[sel] = reg.predict(piv.loc[kt.MOF.values[sel], donors].fillna(0).values) * SCALE
                    out["Donor-ridge"] = pr
                kt = k[test].reset_index(drop=True)
                for n, p in out.items():
                    okp = ~np.isnan(p)
                    sp = kt[okp].assign(p=p[okp]).groupby("scenario").apply(lambda g: spearmanr(g.y_inv, g.p).correlation).median()
                    rowsB.append(dict(region=R, k=kshot, rep=rep, model=n, R2=r2_score(y[test][okp], p[okp]),
                                      median_scenario_spearman=sp))
                print("LORO", R, kshot, rep, round(time.time() - t0), "s", flush=True)
                if kshot == 0:
                    break
        pd.DataFrame(rowsB).to_csv(os.path.join(RES, "exp4_loro_fewshot.csv"), index=False)
    resB = pd.DataFrame(rowsB)
    print(resB.groupby(["model", "k"])[["R2", "median_scenario_spearman"]].mean().round(3).to_string())
    print("done", round(time.time() - t0), "s")
