"""Data-landscape statistics that motivate the method (no model fitting).

- singularity: share of material-scenario pairs with nCAC < 0 (net CO2 avoided <= 0) per scenario factor
- invariance: within-configuration coefficient of variation of process vs TEA/LCA KPIs across regions
- rank instability: cross-scenario Spearman of y = 1/nCAC and top-20 overlap
"""
from __future__ import annotations

import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import RES, load_long  # noqa: E402

k = load_long()
k["negative_nCAC"] = (k.nCAC < 0).astype(int)
rows = []
for fac in ["source", "process", "humidity", "region", "boiler", "heat_extraction"]:
    g = k.groupby(fac).agg(neg_share=("negative_nCAC", "mean"), viable100=("y_inv", lambda s: (s >= 0.01).mean()),
                           n=("MOF", "size"))
    g["factor"] = fac
    rows.append(g.reset_index().rename(columns={fac: "level"}))
pd.concat(rows).to_csv(os.path.join(RES, "data_viability_by_factor.csv"), index=False)

sc = k.groupby(["scenario", "source", "region", "boiler", "heat_extraction", "process", "humidity"]).agg(
    neg_share=("negative_nCAC", "mean"), viable100=("y_inv", lambda s: (s >= 0.01).mean()),
    min_nCAC=("nCAC", lambda s: s[s > 0].min()), n=("MOF", "size")).reset_index()
sc.to_csv(os.path.join(RES, "data_viability_by_scenario.csv"), index=False)

inv = []
for col in ["purity", "recovery", "working_capacity", "productivity", "spec_heat", "el_energy", "CAPEX", "OPEX",
            "eff_recovery", "Climate_Change", "nCAC"]:
    g = k.groupby(["cfg", "MOF"])[col]
    cv = (g.std() / g.mean().abs()).replace([np.inf, -np.inf], np.nan)
    inv.append(dict(kpi=col, layer="process" if col in ("purity", "recovery", "working_capacity", "productivity",
                                                        "spec_heat", "el_energy") else "TEA/LCA",
                    median_cv=cv.median(), p90_cv=cv.quantile(0.9)))
pd.DataFrame(inv).to_csv(os.path.join(RES, "data_invariance_cv.csv"), index=False)

piv = k.pivot_table(index="MOF", columns="scenario", values="y_inv")
c = piv.corr(method="spearman")
c.to_csv(os.path.join(RES, "data_cross_scenario_spearman.csv"))
tops = {s: set(piv[s].nlargest(20).index) for s in piv}
ov = pd.DataFrame([(a, b, len(tops[a] & tops[b]) / 20) for a, b in itertools.combinations(piv.columns, 2)],
                  columns=["a", "b", "top20_overlap"])
ov.to_csv(os.path.join(RES, "data_top20_overlap.csv"), index=False)
print(pd.DataFrame(inv).round(4).to_string())
print("negative nCAC share", k.negative_nCAC.mean().round(3), "| median cross-scenario rho",
      np.nanmedian(c.values[np.triu_indices(len(c), 1)]).round(3), "| median top-20 overlap", ov.top20_overlap.median())
