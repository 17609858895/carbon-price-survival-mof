"""Build analysis tables from the PrISMa v3 Zenodo release.

Outputs (work/data_processed):
  materials.parquet  - one row per MOF: adsorption, geometry, water, IAST flags, CIF descriptors
  kpi_long.parquet   - one row per (MOF, scenario) with scalar KPIs and parsed scenario factors
  scenarios.csv      - scenario factor table
"""
from __future__ import annotations

import ast
import glob
import os
import re

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data_raw")
OUT = os.path.join(ROOT, "data_processed")
os.makedirs(OUT, exist_ok=True)

SOURCES = {"Cement": "Cement", "Coal": "Coal", "NGCC-onshore": "NGCC"}


def parse_scenario(name: str) -> dict:
    stem = name.replace(".csv", "")
    feedback = stem.startswith("Feedback_")
    stem = stem.replace("Feedback_", "")
    proc_hum = stem.split("_")[-1]
    process, humidity = proc_hum.split("-")
    parts = stem.split("_")
    source = SOURCES[parts[0]]
    region = parts[2]
    rest = parts[3:-1]
    heat_extraction = "NoHeatExtraction" not in rest
    boiler = "ELboiler" if "ELboiler" in rest else ("NGboiler" if "NGboiler" in rest else "default")
    return dict(scenario=stem if not feedback else "Feedback_" + stem, source=source, region=region,
                heat_extraction=heat_extraction, boiler=boiler, process=process, humidity=humidity,
                feedback_round=feedback)


def to_float_list(s):
    if isinstance(s, str):
        try:
            v = ast.literal_eval(s)  # literal_eval only parses literals, never executes code
            return np.asarray(v, dtype=float)
        except Exception:
            return None
    return None


# ----------------------------------------------------------------------------- materials
mp = os.path.join(RAW, "Material_Properties")
mk = pd.read_csv(os.path.join(mp, "Material_KPIs-Simulated.csv"))
mk.columns = ["MOF", "H_CO2", "H_N2", "H_sel", "U_CO2", "U_N2", "Q_CO2", "Q_N2"]
zeo = pd.read_csv(os.path.join(mp, "Zeo++-Simulated.csv")).drop(columns="Molecule")
zeo.columns = ["MOF", "density", "POAVF", "Cp", "round"]
mat = mk.merge(zeo, on="MOF")
for src, fn in [("Cement", "Water_Cement"), ("Coal", "Water_Coal"), ("NGCC", "Water_NGCC-onshore")]:
    w = pd.read_csv(os.path.join(mp, f"{fn}-Simulated.csv")).drop(columns="Molecule")
    w.columns = ["MOF"] + [f"w_{c}_{src}" for c in ["Usat", "H", "Qh", "Udes", "Qdes", "UCO2bin", "UCO2ter", "WRC"]]
    mat = mat.merge(w, on="MOF")
# water saturation/Henry/heat are source independent? keep only what differs
for c in ["Usat", "H", "Qh"]:
    same = np.allclose(mat[f"w_{c}_Cement"], mat[f"w_{c}_Coal"], equal_nan=True)
    if same:
        mat[f"w_{c}"] = mat[f"w_{c}_Cement"]
        mat = mat.drop(columns=[f"w_{c}_{s}" for s in ["Cement", "Coal", "NGCC"]])

# isotherm curves -> uptake at fixed partial pressures (log-interp)
iso = pd.read_csv(os.path.join(mp, "Isotherm-Simulated.csv"))
iso.columns = ["MOF", "Molecule", "Henry", "P", "U", "Heat", "T"]
grid = {"CO2": [0.04, 0.13, 0.2, 1.0], "N2": [0.8]}
rows = {}
for _, r in iso.iterrows():
    P, U = to_float_list(r.P), to_float_list(r.U)
    if P is None or U is None or len(P) != len(U):
        continue
    o = np.argsort(P)
    P, U = P[o], U[o]
    d = rows.setdefault(r.MOF, {})
    d[f"T_ref_{r.Molecule}"] = r["T"]
    for p in grid.get(r.Molecule, []):
        d[f"iso_{r.Molecule}_{p}bar"] = float(np.interp(np.log(p), np.log(P + 1e-12), U))
iso_df = pd.DataFrame.from_dict(rows, orient="index").rename_axis("MOF").reset_index()
mat = mat.merge(iso_df, on="MOF", how="left")

# IAST validity flags (per source)
for src, fn in [("Cement", "IAST-Flag_Cement"), ("Coal", "IAST-Flag_Coal"), ("NGCC", "IAST-Flag_NGCC-onshore")]:
    f = pd.read_csv(os.path.join(RAW, "Flags", f"{fn}.csv"))
    f.columns = ["MOF", f"iast_CO2_{src}", f"iast_N2_{src}"]
    mat = mat.merge(f, on="MOF", how="left")

# ----------------------------------------------------------------------------- CIF descriptors
METALS = set("Li Be Na Mg Al K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Rb Sr Y Zr Nb Mo Ru Rh Pd Ag Cd In Sn Cs Ba La Ce "
             "Pr Nd Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Th U".split())
ELEMS = ["C", "H", "N", "O", "F", "Cl", "Br", "S", "P", "B", "Si", "I"]


def cif_desc(path):
    txt = open(path, encoding="utf-8", errors="ignore").read().splitlines()
    cell = {}
    atoms, charges = [], []
    cols, in_loop = [], False
    for line in txt:
        s = line.strip()
        m = re.match(r"_cell_(length_[abc]|angle_(alpha|beta|gamma))\s+([-\d.]+)", s)
        if m:
            cell[m.group(1)] = float(m.group(3))
            continue
        if s == "loop_":
            cols, in_loop = [], True
            continue
        if in_loop and s.startswith("_"):
            cols.append(s.split()[0])
            continue
        if in_loop and cols and "_atom_site_type_symbol" in cols and s and not s.startswith("_"):
            tok = s.split()
            if len(tok) < len(cols):
                continue
            atoms.append(re.sub(r"[^A-Za-z]", "", tok[cols.index("_atom_site_type_symbol")]))
            if "_atom_site_charge" in cols:
                try:
                    charges.append(float(tok[cols.index("_atom_site_charge")]))
                except ValueError:
                    pass
        elif in_loop and s == "":
            in_loop = False if cols and "_atom_site_type_symbol" not in cols else in_loop
    a, b, c = cell.get("length_a"), cell.get("length_b"), cell.get("length_c")
    al, be, ga = [np.radians(cell.get(k, 90.0)) for k in ("angle_alpha", "angle_beta", "angle_gamma")]
    vol = a * b * c * np.sqrt(1 - np.cos(al) ** 2 - np.cos(be) ** 2 - np.cos(ga) ** 2
                              + 2 * np.cos(al) * np.cos(be) * np.cos(ga)) if a else np.nan
    n = len(atoms)
    s = pd.Series(atoms)
    d = {"n_atoms": n, "cell_volume": vol, "atom_density": n / vol if vol else np.nan}
    for e in ELEMS:
        d[f"frac_{e}"] = (s == e).mean() if n else np.nan
    metals = s[s.isin(METALS)]
    d["frac_metal"] = len(metals) / n if n else np.nan
    d["metal_main"] = metals.value_counts().index[0] if len(metals) else "none"
    d["n_metal_types"] = metals.nunique()
    q = np.asarray(charges)
    if len(q) == n and n:
        d.update(q_abs_mean=np.abs(q).mean(), q_std=q.std(), q_max=q.max(), q_min=q.min(),
                 q_metal_mean=q[s.isin(METALS).values].mean() if len(metals) else np.nan,
                 q_O_mean=q[(s == "O").values].mean() if (s == "O").any() else 0.0,
                 q_N_mean=q[(s == "N").values].mean() if (s == "N").any() else 0.0,
                 q_has=1)
    else:
        d.update(q_abs_mean=np.nan, q_std=np.nan, q_max=np.nan, q_min=np.nan, q_metal_mean=np.nan,
                 q_O_mean=np.nan, q_N_mean=np.nan, q_has=0)
    return d


cif_dir = os.path.join(RAW, "Cifs")
cif_map = {}
for f in glob.glob(os.path.join(cif_dir, "*.cif")):
    cif_map[os.path.basename(f)[:-4]] = f
cif_rows = []
for m in mat.MOF:
    f = cif_map.get(m)
    if f is None:  # try loose matching
        cand = [k for k in cif_map if k.lower() == str(m).lower()]
        f = cif_map[cand[0]] if cand else None
    d = cif_desc(f) if f else {}
    d["MOF"] = m
    d["cif_found"] = f is not None
    cif_rows.append(d)
cif_df = pd.DataFrame(cif_rows)
mat = mat.merge(cif_df, on="MOF", how="left")


def family(m: str) -> str:
    m = str(m)
    if re.match(r"^\d+_[A-Za-z\-]+\+N\d+\+\d+", m):
        return "ToBaCCo-like:" + m.split("_")[1].split("+")[0]
    if m.startswith("ddmof_"):
        return "ddmof"
    if m.endswith("_clean") or re.match(r"^[A-Z]{6}", m):
        return "CoRE/CSD"
    if m.isdigit():
        return "hMOF-numeric"
    return "named/other"


mat["family"] = mat.MOF.map(family)
mat["family_coarse"] = mat.family.str.split(":").str[0]
mat.to_parquet(os.path.join(OUT, "materials.parquet"), index=False)
print("materials", mat.shape, "cif_found", mat.cif_found.sum())
print(mat.family_coarse.value_counts())

# ----------------------------------------------------------------------------- KPI long table
SCALAR = ["purity", "recovery", "working_capacity", "productivity", "spec_heat", "spec_cool", "el_energy",
          "rho_b", "selectivity", "alpha", "CAPEX", "OPEX", "CAC", "CCC", "LCOC", "spec_heat_tot", "spec_cool_tot",
          "spec_power_tot", "purity_c", "productivity_tea", "CO2_avoided", "CO2_captured", "spec_CO2_emission",
          "SPECCA", "var_OPEX", "Climate Change", "Water Use", "Energy Resources: Non-Renewable",
          "Material Resources: Metals/Minerals", "Land Use", "Acidification", "Eutrophication: Freshwater",
          "Eutrophication: Marine", "Ecotoxicity: Freshwater", "Human Toxicity: Carcinogenic",
          "Human Toxicity: Non-Carcinogenic", "Particulate Matter Formation", "Ozone Depletion",
          "Photochemical Ozone Formation: Human Health", "Ionising Radiation: Human Health", "nCAC",
          "eff_recovery", "OPEX_system", "nCAC_system"]
frames = []
for f in sorted(glob.glob(os.path.join(RAW, "KPIs_Simulated", "*.csv"))):
    d = pd.read_csv(f, usecols=lambda c: c in ["MOF"] + SCALAR + ["CAC_bd", "nCAC_system_bd"])
    meta = parse_scenario(os.path.basename(f).replace("NGCC-onshore_storage", "NGCC-onshore_Storage"))
    for k, v in meta.items():
        d[k] = v
    # CAC breakdown (dict literal) -> components
    if "nCAC_system_bd" in d:
        bd = d["nCAC_system_bd"].map(lambda s: ast.literal_eval(s) if isinstance(s, str) else {})
        for comp in ["CAPEX", "FixedOPEX", "VariableOPEX"]:
            d[f"nCAC_{comp}"] = bd.map(lambda x: x.get(comp, np.nan))
        d = d.drop(columns=["nCAC_system_bd", "CAC_bd"], errors="ignore")
    frames.append(d)
kpi = pd.concat(frames, ignore_index=True)
kpi = kpi.replace([np.inf, -np.inf], np.nan)
kpi.columns = [c.replace(": ", "_").replace(" ", "_").replace(",", "").replace("/", "_") for c in kpi.columns]
kpi.to_parquet(os.path.join(OUT, "kpi_long.parquet"), index=False)
sc = kpi.drop_duplicates("scenario")[["scenario", "source", "region", "heat_extraction", "boiler", "process",
                                      "humidity", "feedback_round"]]
sc["n_rows"] = kpi.groupby("scenario").size().reindex(sc.scenario).values
sc["n_nCAC"] = kpi.groupby("scenario")["nCAC"].count().reindex(sc.scenario).values
sc.to_csv(os.path.join(OUT, "scenarios.csv"), index=False)
print("kpi", kpi.shape, "scenarios", len(sc))
