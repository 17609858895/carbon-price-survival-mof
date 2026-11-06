"""Shared data loading, feature definitions and metrics for the PrISMa ML study."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DP = os.path.join(ROOT, "data_processed")
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)

PRICE_GRID = np.array([40, 50, 60, 75, 100, 125, 150, 200, 250, 300, 400, 500], dtype=float)

LOG_COLS = ["H_CO2", "H_N2", "H_sel", "w_H", "iso_CO2_0.04bar", "iso_CO2_0.13bar", "iso_CO2_0.2bar",
            "iso_CO2_1.0bar", "iso_N2_0.8bar", "U_N2", "U_CO2", "cell_volume", "n_atoms"]
BASE_MAT = ["H_CO2", "H_N2", "H_sel", "U_CO2", "U_N2", "Q_CO2", "Q_N2", "density", "POAVF", "Cp",
            "w_Usat", "w_H", "w_Qh", "iso_CO2_0.04bar", "iso_CO2_0.13bar", "iso_CO2_0.2bar", "iso_CO2_1.0bar",
            "iso_N2_0.8bar"]
CIF_MAT = ["n_atoms", "cell_volume", "atom_density", "frac_C", "frac_H", "frac_N", "frac_O", "frac_F", "frac_Cl",
           "frac_Br", "frac_S", "frac_P", "frac_metal", "n_metal_types", "q_abs_mean", "q_std", "q_max", "q_min",
           "q_metal_mean", "q_O_mean", "q_N_mean"]
SRC_MAT = ["w_Udes", "w_Qdes", "w_UCO2bin", "w_UCO2ter", "w_WRC", "iast_CO2", "iast_N2"]  # per-source columns
SCEN_CAT = ["source", "process", "humidity", "region", "boiler", "heat_extraction"]
PROC_OUT = ["purity", "recovery", "working_capacity", "productivity", "spec_heat", "el_energy", "rho_b"]


def safe_log(x):
    return np.log10(np.clip(np.asarray(x, dtype=float), 1e-30, None))


def load_materials() -> pd.DataFrame:
    m = pd.read_parquet(os.path.join(DP, "materials.parquet"))
    for c in LOG_COLS:
        if c in m:
            m["log_" + c] = safe_log(m[c])
    return m


def mat_feature_names(m: pd.DataFrame) -> list[str]:
    base = [("log_" + c) if ("log_" + c) in m else c for c in BASE_MAT + CIF_MAT]
    return list(dict.fromkeys(base))


def load_long(include_feedback: bool = False) -> pd.DataFrame:
    """Long (MOF x scenario) table joined with material descriptors (source-specific water/IAST resolved)."""
    k = pd.read_parquet(os.path.join(DP, "kpi_long.parquet"))
    if not include_feedback:
        k = k[~k.feedback_round]
    if os.environ.get("PRISMA_PANEL", "1") == "1":
        # core scenario panel used for the exploratory runs: 3 regions x 3 sources x 3 processes x dry/wet = 54
        # (+ the 9 UK feedback scenarios when requested); set PRISMA_PANEL=0 for all 132 main scenarios
        k = k[k.region.isin(["UK", "US", "CN-SD"]) & (k.boiler == "default") & k.heat_extraction.astype(bool)]
    k = k.dropna(subset=["nCAC"]).copy()
    m = load_materials()
    k = k.merge(m, on="MOF", how="left")
    for c in SRC_MAT:
        k[c] = np.nan
        for s in ["Cement", "Coal", "NGCC"]:
            sel = k.source == s
            k.loc[sel, c] = k.loc[sel, f"{c}_{s}"]
    k["log_w_WRC"] = safe_log(k["w_WRC"])
    k["log_w_UCO2ter"] = safe_log(k["w_UCO2ter"])
    k["y_inv"] = 1.0 / k["nCAC"]          # net avoided CO2 per euro (t/EUR); sign = net avoidance
    k["viable"] = (k["nCAC"] > 0).astype(int)
    k["cfg"] = k.source + "|" + k.process + "|" + k.humidity
    k["tea"] = k.region + "|" + k.boiler + "|" + k.heat_extraction.astype(str)
    for c in SCEN_CAT:
        k[c] = k[c].astype(str)
    return k


def feature_list(k: pd.DataFrame, cif: bool = True) -> list[str]:
    m = [c for c in mat_feature_names(k) if cif or c not in CIF_MAT]
    return m + ["w_Udes", "w_Qdes", "w_UCO2bin", "log_w_UCO2ter", "log_w_WRC", "iast_CO2", "iast_N2"]


def one_hot_scen(k: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(k[SCEN_CAT], dtype=float)


# --------------------------------------------------------------------------- physics proxies (Exp. 12)
PHYS = ["phys_q_ads", "phys_wc", "phys_log_e"]
# reference-state proxy conditions (not an exact reconstruction of the fixed cycle): flue-gas CO2 partial pressure (bar),
# desorption pressure from the process name (bar), adsorption / desorption temperatures (K)
P_FEED = {"Cement": 0.20, "Coal": 0.13, "NGCC": 0.04}
P_DES = {"TSA": 1.0, "TVSA02": 0.2, "TVSA06": 0.6}
T_ADS, T_DES = 298.15, 393.15


def langmuir_fit(m: pd.DataFrame):
    """Single-site Langmuir (q_sat, b) fitted by least squares to the four GCMC CO2 uptakes at 298 K."""
    P = np.array([0.04, 0.13, 0.2, 1.0])
    q = m[["iso_CO2_0.04bar", "iso_CO2_0.13bar", "iso_CO2_0.2bar", "iso_CO2_1.0bar"]].to_numpy(float)
    bs = np.logspace(-3, 4, 500)
    f = bs[:, None] * P[None, :] / (1 + bs[:, None] * P[None, :])                    # (nb, 4)
    qsat = (q @ f.T) / (f ** 2).sum(1)[None, :]                                        # (n, nb)
    sse = ((q[:, None, :] - qsat[:, :, None] * f[None, :, :]) ** 2).sum(-1)
    j = np.argmin(np.where(np.isfinite(sse), sse, np.inf), axis=1)
    return qsat[np.arange(len(m)), j], bs[j]


def add_physics_proxies(k: pd.DataFrame) -> pd.DataFrame:
    """Langmuir + Clausius-Clapeyron working capacity and a sensible-plus-desorption regeneration energy per row
    (material descriptors and nominal scenario conditions only; no simulated process output is used)."""
    mats = k.drop_duplicates("MOF").set_index("MOF")
    qs, b = langmuir_fit(mats)
    qs, b = pd.Series(qs, mats.index).reindex(k.MOF).values, pd.Series(b, mats.index).reindex(k.MOF).values
    pf, pdes = k.source.map(P_FEED).values.astype(float), k.process.map(P_DES).values.astype(float)
    dH = np.abs(k.Q_CO2.values) * 1e3 / 8.314
    b_des = b * np.exp(-dH * (1 / T_ADS - 1 / T_DES))
    q_ads = qs * b * pf / (1 + b * pf)
    wc = q_ads - qs * b_des * pdes / (1 + b_des * pdes)
    e = k.Cp.values * (T_DES - T_ADS) / np.maximum(wc, 0.01) + np.abs(k.Q_CO2.values)   # kJ per mol CO2
    k["phys_q_ads"], k["phys_wc"], k["phys_log_e"] = q_ads, wc, np.log10(e)
    return k


# --------------------------------------------------------------------------- metrics
def viability_price_metrics(y_true_inv, y_pred_inv, prices=PRICE_GRID):
    """Classification of 'viable at carbon price p' (0 < nCAC <= p  <=>  y_inv >= 1/p)."""
    from sklearn.metrics import f1_score, roc_auc_score
    out = {}
    for p in prices:
        t = (y_true_inv >= 1.0 / p).astype(int)
        if t.min() == t.max():
            continue
        out[f"AUC@{int(p)}"] = roc_auc_score(t, y_pred_inv)
        out[f"F1@{int(p)}"] = f1_score(t, (y_pred_inv >= 1.0 / p).astype(int))
    return out


def topk_metrics(df, score_col, true_col="y_inv", group="scenario", ks=(10, 25, 50)):
    """Per-scenario top-k retrieval of the lowest-nCAC (highest y_inv) materials."""
    from scipy.stats import spearmanr
    rows = []
    for s, g in df.groupby(group):
        r = {"scenario": s, "spearman": spearmanr(g[true_col], g[score_col]).correlation}
        tv = g[g[true_col] > 0]
        r["spearman_viable_nCAC"] = spearmanr(tv[true_col], tv[score_col]).correlation if len(tv) > 10 else np.nan
        for kk in ks:
            top_t = set(g.nlargest(kk, true_col).MOF)
            top_p = set(g.nlargest(kk, score_col).MOF)
            r[f"P@{kk}"] = len(top_t & top_p) / kk
            # regret: true nCAC of best predicted vs true best (relative)
        best_true = 1 / g[true_col].max()
        pick = g.loc[g[score_col].idxmax()]
        r["regret_top1_pct"] = ((1 / pick[true_col]) / best_true - 1) * 100 if pick[true_col] > 0 else np.inf
        top5 = g.nlargest(5, score_col)
        r["regret_best_of_top5_pct"] = ((1 / top5[true_col].max()) / best_true - 1) * 100 if top5[true_col].max() > 0 else np.inf
        rows.append(r)
    return pd.DataFrame(rows)


def fold_of(mof: str, n: int = 5, seed: int = 0) -> int:
    """Deterministic MOF-level fold id: a MOF is in the same fold in every scenario and every experiment.
    seed > 0 gives an independent repetition of the MOF partition (seed 0 keeps the original assignment)."""
    import hashlib
    key = str(mof) if seed == 0 else f"{seed}|{mof}"
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % n


# --------------------------------------------------------------------------- model settings
FULL = os.environ.get("PRISMA_FULL", "0") == "1"      # 1 = final settings; 0 = fast exploratory settings
THREADS = int(os.environ.get("PRISMA_THREADS", "4"))


def lgbp(scale: float = 1.0, **kw) -> dict:
    """LightGBM parameters shared by all experiments; `scale` shrinks the tree count for inner/nuisance fits."""
    n, lr = (1500, 0.03) if FULL else (500, 0.08)
    p = dict(n_estimators=max(50, int(n * scale)), learning_rate=lr, num_leaves=63, min_child_samples=40,
             subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0, verbose=-1, n_jobs=THREADS)
    p.update(kw)
    return p
