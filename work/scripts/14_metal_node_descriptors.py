"""Metal-node descriptors from the (P1) CIFs: metal coordination, open-metal-site fraction and cation properties.

Coordination counts non-H, non-metal neighbours within r_M + r_X + 0.3 A (Cordero covalent radii), using all 27 periodic
images. A metal atom is counted as an open metal site (OMS) when its coordination number is below a reference value for
that cation (6 by default; 4 for Zn, 5 for Cu, 7 for Ca/Sr/Ba, 8 for lanthanides/actinides, 2 for Ag/Hg/Au).
Output: data_processed/metal_node.parquet (one row per MOF).
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CIF_DIR = os.path.join(ROOT, "data_raw", "Cifs")
OUT = os.path.join(ROOT, "data_processed", "metal_node.parquet")

COV = dict(H=0.31, B=0.84, C=0.76, N=0.71, O=0.66, F=0.57, Si=1.11, P=1.07, S=1.05, Cl=1.02, Br=1.20, I=1.39,
           Li=1.28, Be=0.96, Na=1.66, Mg=1.41, Al=1.21, K=2.03, Ca=1.76, Sc=1.70, Ti=1.60, V=1.53, Cr=1.39, Mn=1.50,
           Fe=1.42, Co=1.38, Ni=1.24, Cu=1.32, Zn=1.22, Ga=1.22, Rb=2.20, Sr=1.95, Y=1.90, Zr=1.75, Nb=1.64, Mo=1.54,
           Ru=1.46, Rh=1.42, Pd=1.39, Ag=1.45, Cd=1.44, In=1.42, Sn=1.39, Cs=2.44, Ba=2.15, La=2.07, Ce=2.04, Pr=2.03,
           Nd=2.01, Sm=1.98, Eu=1.98, Gd=1.96, Tb=1.94, Dy=1.92, Ho=1.92, Er=1.89, Tm=1.90, Yb=1.87, Lu=1.87, Hf=1.75,
           Ta=1.70, W=1.62, Re=1.51, Os=1.44, Ir=1.41, Pt=1.36, Au=1.36, Hg=1.32, Tl=1.45, Pb=1.46, Bi=1.48, Th=2.06,
           U=1.96)
# typical oxidation state, Shannon ionic radius (A, CN 6 where available), Pauling electronegativity, group label
ION = {
    "Li": (1, 0.76, 0.98, "alkali"), "Na": (1, 1.02, 0.93, "alkali"), "K": (1, 1.38, 0.82, "alkali"),
    "Rb": (1, 1.52, 0.82, "alkali"), "Cs": (1, 1.67, 0.79, "alkali"),
    "Be": (2, 0.45, 1.57, "alkaline_earth"), "Mg": (2, 0.72, 1.31, "alkaline_earth"),
    "Ca": (2, 1.00, 1.00, "alkaline_earth"), "Sr": (2, 1.18, 0.95, "alkaline_earth"),
    "Ba": (2, 1.35, 0.89, "alkaline_earth"),
    "Sc": (3, 0.745, 1.36, "tm3d"), "Ti": (4, 0.605, 1.54, "tm3d"), "V": (3, 0.64, 1.63, "tm3d"),
    "Cr": (3, 0.615, 1.66, "tm3d"), "Mn": (2, 0.83, 1.55, "tm3d"), "Fe": (3, 0.645, 1.83, "tm3d"),
    "Co": (2, 0.745, 1.88, "tm3d"), "Ni": (2, 0.69, 1.91, "tm3d"), "Cu": (2, 0.73, 1.90, "tm3d"),
    "Zn": (2, 0.74, 1.65, "tm3d"),
    "Y": (3, 0.90, 1.22, "tm45d"), "Zr": (4, 0.72, 1.33, "tm45d"), "Nb": (5, 0.64, 1.60, "tm45d"),
    "Mo": (6, 0.59, 2.16, "tm45d"), "Ru": (3, 0.68, 2.20, "tm45d"), "Rh": (3, 0.665, 2.28, "tm45d"),
    "Pd": (2, 0.86, 2.20, "tm45d"), "Ag": (1, 1.15, 1.93, "tm45d"), "Cd": (2, 0.95, 1.69, "tm45d"),
    "Hf": (4, 0.71, 1.30, "tm45d"), "Ta": (5, 0.64, 1.50, "tm45d"), "W": (6, 0.60, 2.36, "tm45d"),
    "Re": (7, 0.53, 1.90, "tm45d"), "Os": (4, 0.63, 2.20, "tm45d"), "Ir": (3, 0.68, 2.20, "tm45d"),
    "Pt": (2, 0.80, 2.28, "tm45d"), "Au": (1, 1.37, 2.54, "tm45d"), "Hg": (2, 1.02, 2.00, "tm45d"),
    "Al": (3, 0.535, 1.61, "p_block"), "Ga": (3, 0.62, 1.81, "p_block"), "In": (3, 0.80, 1.78, "p_block"),
    "Sn": (4, 0.69, 1.96, "p_block"), "Tl": (1, 1.50, 1.62, "p_block"), "Pb": (2, 1.19, 2.33, "p_block"),
    "Bi": (3, 1.03, 2.02, "p_block"),
    "La": (3, 1.032, 1.10, "f_block"), "Ce": (3, 1.01, 1.12, "f_block"), "Pr": (3, 0.99, 1.13, "f_block"),
    "Nd": (3, 0.983, 1.14, "f_block"), "Sm": (3, 0.958, 1.17, "f_block"), "Eu": (3, 0.947, 1.20, "f_block"),
    "Gd": (3, 0.938, 1.20, "f_block"), "Tb": (3, 0.923, 1.10, "f_block"), "Dy": (3, 0.912, 1.22, "f_block"),
    "Ho": (3, 0.901, 1.23, "f_block"), "Er": (3, 0.89, 1.24, "f_block"), "Tm": (3, 0.88, 1.25, "f_block"),
    "Yb": (3, 0.868, 1.10, "f_block"), "Lu": (3, 0.861, 1.27, "f_block"), "Th": (4, 0.94, 1.30, "f_block"),
    "U": (6, 0.73, 1.38, "f_block"),
}
CN_REF = {"Zn": 4, "Cu": 5, "Ca": 7, "Sr": 7, "Ba": 7, "Ag": 2, "Hg": 2, "Au": 2}
GROUPS = ["alkali", "alkaline_earth", "tm3d", "tm45d", "p_block", "f_block"]
SHIFTS = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)], float)


def read_p1(path):
    lines = open(path, encoding="utf-8", errors="ignore").read().splitlines()
    cell, cols, rows, in_loop = {}, [], [], False
    for line in lines:
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
        if in_loop and "_atom_site_fract_x" in cols and s and not s.startswith("_"):
            tok = s.split()
            if len(tok) >= len(cols):
                el = tok[cols.index("_atom_site_type_symbol")] if "_atom_site_type_symbol" in cols else tok[0]
                xyz = [float(re.sub(r"\(.*\)", "", tok[cols.index(c)])) for c in
                       ("_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z")]
                rows.append((re.sub(r"[^A-Za-z]", "", el), *xyz))
    a, b, c = cell["length_a"], cell["length_b"], cell["length_c"]
    al, be, ga = (np.radians(cell.get(k, 90.0)) for k in ("angle_alpha", "angle_beta", "angle_gamma"))
    v = np.sqrt(1 - np.cos(al) ** 2 - np.cos(be) ** 2 - np.cos(ga) ** 2 + 2 * np.cos(al) * np.cos(be) * np.cos(ga))
    M = np.array([[a, b * np.cos(ga), c * np.cos(be)],
                  [0, b * np.sin(ga), c * (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)],
                  [0, 0, c * v / np.sin(ga)]])
    el = np.array([r[0] for r in rows])
    frac = np.array([r[1:] for r in rows], float)
    return el, frac, M, a * b * c * v


def node_desc(path):
    el, frac, M, vol = read_p1(path)
    is_m = np.isin(el, list(ION))
    lig = ~is_m & (el != "H")
    out = dict(n_metal_atoms=int(is_m.sum()), metal_per_nm3=1000 * is_m.sum() / vol)
    if not is_m.any() or not lig.any():
        return out
    fm, fl = frac[is_m], frac[lig]
    rm = np.array([COV.get(e, 1.5) for e in el[is_m]])
    rl = np.array([COV.get(e, 0.8) for e in el[lig]])
    cn, dmin, d_mo = np.zeros(len(fm)), np.full(len(fm), np.nan), []
    for i0 in range(0, len(fm), 16):
        d = fl[None, :, :] - fm[i0:i0 + 16, None, :]
        d -= np.round(d)
        dd = d[:, :, None, :] + SHIFTS[None, None, :, :]                     # (m, l, 27, 3)
        r = np.linalg.norm(dd @ M.T, axis=-1).min(axis=2)                     # (m, l)
        cut = rm[i0:i0 + 16, None] + rl[None, :] + 0.3
        bond = r < cut
        cn[i0:i0 + 16] = bond.sum(1)
        dmin[i0:i0 + 16] = np.where(bond, r, np.inf).min(1)
        o = el[lig] == "O"
        if o.any():
            ro = np.where(bond[:, o], r[:, o], np.nan)
            d_mo.extend(ro[np.isfinite(ro)].tolist())
    mets = el[is_m]
    ref = np.array([CN_REF.get(e, 8 if ION[e][3] == "f_block" else 6) for e in mets])
    main = pd.Series(mets).value_counts().index[0]
    z, rad, en, grp = ION[main]
    out.update(cn_mean=cn.mean(), cn_min=cn.min(), cn_deficit=float(np.mean(np.clip(ref - cn, 0, None))),
               oms_frac=float(np.mean(cn < ref)), oms_per_nm3=1000 * float(np.sum(cn < ref)) / vol,
               d_M_lig_min=float(np.nanmin(np.where(np.isfinite(dmin), dmin, np.nan))) if np.isfinite(dmin).any() else np.nan, d_MO_mean=float(np.mean(d_mo)) if d_mo else np.nan,
               metal_z=z, metal_r_ion=rad, metal_en=en, metal_ionic_potential=z / rad)
    for g in GROUPS:
        out[f"metal_grp_{g}"] = float(grp == g)
    return out


if __name__ == "__main__":
    mat = pd.read_parquet(os.path.join(ROOT, "data_processed", "materials.parquet"))[["MOF"]]
    files = {os.path.basename(f)[:-4].lower(): f for f in glob.glob(os.path.join(CIF_DIR, "*.cif"))}
    rows = []
    for i, m in enumerate(mat.MOF):
        f = files.get(str(m).lower())
        d = node_desc(f) if f else {}
        d["MOF"] = m
        rows.append(d)
        if i % 200 == 0:
            print(i, m, {k: round(v, 3) for k, v in d.items() if isinstance(v, float)}, flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(OUT, index=False)
    print(df.describe().T.round(3).to_string())
    print(df[df.MOF == "MgMOF74"].T)
