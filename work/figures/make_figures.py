"""Ten compact main figures (full 132-scenario results).

Every panel is a function drawing on a given axis. Each figure is rendered once as a combined canvas with panel
letters, and every panel is rendered again on its own standalone canvas into <FigXX>/subfigures/ (no cropping, no
letter). Text uses regular weight; letters sit at the top left of each panel footprint, aligned by grid row and column.
Usage:  python figures/make_figures.py [Fig01 Fig02 ...]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PRISMA_PANEL", "0")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import json  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Patch, Polygon, Rectangle  # noqa: E402
from scipy.stats import gaussian_kde, norm  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))
from common import RES, load_long  # noqa: E402
from ml_publication_style import configure_publication_style, numeric_ticks  # noqa: E402
from ml_publication_style import style_axis as _style_axis  # noqa: E402
from scientific_palettes import STYLE_COLORS, get_palette  # noqa: E402

configure_publication_style()
plt.rcParams.update({"axes.labelweight": "normal", "font.weight": "normal", "axes.labelsize": 15.0,
                     "legend.fontsize": 11})
PAL = get_palette("au_nimno_soft_8")
INK, MUTED, LIGHT = STYLE_COLORS["ink"], STYLE_COLORS["muted"], STYLE_COLORS["light"]
OURS = PAL[1]
SRC = {"Cement": PAL[0], "Coal": PAL[1], "NGCC": PAL[2]}
METH = {"direct": PAL[3], "signed_log": PAL[5], "hurdle": PAL[0], "inverse": OURS}
METH_LAB = {"direct": "nCAC", "signed_log": "Signed log", "hurdle": "Hurdle", "inverse": "RIV (1/nCAC)"}
W2R = LinearSegmentedColormap.from_list("w2r", ["#FFFFFF", OURS])
W2B = LinearSegmentedColormap.from_list("w2b", ["#FFFFFF", PAL[0]])
LETTER_PT = 22
EUR = "€ t$^{-1}$"
R = lambda f: os.path.join(RES, f)  # noqa: E731
_cache: dict = {}


def style_axis(ax, xlabel=None, ylabel=None, tick_size=13.0, label_size=15.0):
    """Shared axis style in regular weight (ticks visible, top/right spines off)."""
    tick_size = max(tick_size, 11.8)
    _style_axis(ax, xlabel, ylabel, tick_size=tick_size, label_size=label_size)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("normal")
    ax.xaxis.label.set_fontweight("normal")
    ax.yaxis.label.set_fontweight("normal")


def kdata():
    if "k" not in _cache:
        k = load_long()
        k["CC"] = 1 - k.eff_recovery / k.recovery
        _cache["k"] = k
    return _cache["k"]


def oof():
    if "o" not in _cache:
        _cache["o"] = pd.read_parquet(R("exp6_riv_oof.parquet"))
    return _cache["o"]


def edge(ax, lw=0.8):
    for p in ax.patches:
        p.set_edgecolor(INK)
        p.set_linewidth(lw)


def hcbar(im, ax, label, ticks=None, ticklabels=None, pad=0.2, fraction=0.05):
    cb = plt.colorbar(im, ax=ax, orientation="horizontal", fraction=fraction, pad=pad, aspect=30)
    cb.set_label(label, fontsize=12)
    cb.ax.tick_params(labelsize=11, length=4, width=1.1)
    cb.outline.set_linewidth(1.1)
    if ticks is not None:
        cb.set_ticks(ticks)
    if ticklabels is not None:
        cb.set_ticklabels(ticklabels)
    return cb


def add_letters(fig, placed):
    """Letters at the top left of each panel footprint; aligned by grid-row top and grid-column left."""
    if hasattr(fig,"_panel_targets"):
        for letter, target in fig._panel_targets.items():
            fig.text(target.x0-.004,target.y1,letter,fontsize=LETTER_PT,ha="right",va="top",color=INK)
        return
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    info = []
    for letter, ax, r0, c0 in placed:
        bb = ax.get_tightbbox(rend).transformed(inv)
        info.append((letter, bb, r0, c0))
    for letter, bb, r0, c0 in info:
        y = max(b.y1 for _, b, rr, _ in info if rr == r0)
        x = min(b.x0 for _, b, _, cc in info if cc == c0)
        fig.text(x - 0.004, y, letter, fontsize=LETTER_PT, ha="right", va="top", color=INK)


def combined(ax):
    return bool(getattr(ax.figure, "_publication_combined", False))


def _readable_combined_type(fig, stem):
    """Apply an explicit physical print-size target without resizing the canvas."""
    from matplotlib.text import Text
    # A physical target is recomputed for the current compact canvas.  The former
    # frozen-width table is not applicable after panels are regrouped.
    target=8.4*fig.get_figwidth()/5.76
    for ax in fig.axes:
        for axis in (ax.xaxis,ax.yaxis):
            axis.set_tick_params(labelsize=target)
        saved=getattr(ax,"_publication_legend",None)
        if saved:
            args,kwargs,original=saved
            kwargs["fontsize"]=max(float(kwargs.get("fontsize",target)),target)
            kwargs["handlelength"]=1.15
            kwargs["handletextpad"]=0.5
            kwargs["columnspacing"]=0.9
            kwargs["labelspacing"]=0.25
            original(*args,**kwargs)
    for item in fig.findobj(Text):
        item.set_fontsize(max(item.get_fontsize(),target))
    fig._publication_minimum_native=target


def _footprint(fig, group):
    renderer = fig.canvas.get_renderer()
    boxes = [a.get_tightbbox(renderer) for a in group if a.get_visible()]
    return Bbox.union(boxes).transformed(fig.transFigure.inverted())


def _fit_footprints(fig, entries, iterations=7):
    """Calibrate each complete panel, including labels and colourbars, to its grid cell."""
    fig.canvas.draw()
    for _ in range(iterations):
        worst = 0
        for group, target in entries:
            bb = _footprint(fig, group)
            err = max(abs(bb.x0-target.x0), abs(bb.y0-target.y0), abs(bb.x1-target.x1), abs(bb.y1-target.y1))
            worst = max(worst,err)
            sx, sy = target.width/bb.width, target.height/bb.height
            for ax in group:
                pos = ax.get_position()
                ax.set_position([target.x0+(pos.x0-bb.x0)*sx, target.y0+(pos.y0-bb.y0)*sy, pos.width*sx, pos.height*sy])
        fig.canvas.draw()
        if worst < 0.001:
            break
    return [dict(target=list(t.bounds), actual=list(_footprint(fig,g).bounds),
                 max_error=float(max(abs(v-w) for v,w in zip(t.extents,_footprint(fig,g).extents)))) for g,t in entries]


def _audit_actual_palette(fig, stem):
    """Inspect base RGB values of actual coloured marks; numerical colormaps are separate."""
    from matplotlib.colors import to_rgba, to_hex
    from matplotlib.lines import Line2D
    from matplotlib.collections import Collection
    from matplotlib.patches import Patch
    palette={to_hex(c).upper() for c in PAL}
    neutral={to_hex(c).upper() for c in [INK,MUTED,LIGHT,"white","black","none"]}
    checks=[]; unexpected=[]
    for ai,ax in enumerate(fig.axes):
        for ar in list(ax.lines)+list(ax.collections)+list(ax.patches):
            colors=[]
            if isinstance(ar,Line2D):
                colors=[ar.get_color()]
                if ar.get_marker() not in [None,'None','none','',' ']:
                    colors += [ar.get_markerfacecolor(),ar.get_markeredgecolor()]
            elif isinstance(ar,Collection):
                if ar.get_array() is not None:
                    continue  # continuously mapped numerical value, not a category
                colors=list(ar.get_facecolors())+list(ar.get_edgecolors())
            elif isinstance(ar,Patch):
                colors=[ar.get_facecolor()]
            for c in colors:
                try:
                    rgba=to_rgba(c)
                    if rgba[3] == 0:continue
                    hx=to_hex(rgba).upper()
                    if hx in neutral:continue
                    checks.append(dict(axis=ai,artist=type(ar).__name__,colour=hx))
                    if hx not in palette:unexpected.append(checks[-1])
                except (ValueError,TypeError):continue
    result=dict(passed=not unexpected,palette="au_nimno_soft_8",coloured_mark_checks=len(checks),unexpected=unexpected,
                scope="Actual coloured lines, marks, bars and bands; neutral structural guides and continuous numeric maps excluded. Raster illustration assessed visually.",checks=checks)
    (HERE.parent/"qa_20261107_compact"/f"{stem}_palette.json").write_text(json.dumps(result,indent=2),encoding="utf8")
    if unexpected:raise ValueError(f"Out-of-palette marks in {stem}: {unexpected}")




def build(stem: str, grid: tuple, panels: list, figsize: tuple, wspace=0.42, hspace=0.45, height_ratios=None,
          width_ratios=None, rectangles=None):
    """Render combined figure and re-render every panel on its own canvas."""
    preview=os.environ.get("PRISMA_PREVIEW")=="1"
    qadir=HERE.parent/"qa_20261107_compact"
    qadir.mkdir(exist_ok=True)
    folder = (qadir/"previews" if preview else HERE) / stem
    sub = folder / "subfigures"
    sub.mkdir(parents=True, exist_ok=True)
    for old in list(folder.glob("*.png")) + list(folder.glob("*.pdf")) + list(sub.glob("*.png")) + list(sub.glob("*.pdf")):
        old.unlink()
    fig = plt.figure(figsize=figsize, dpi=90)
    fig._publication_combined = True
    gs = fig.add_gridspec(*grid, left=0.025, right=0.995, bottom=0.025, top=0.975,
                         wspace=.085, hspace=.065, height_ratios=height_ratios, width_ratios=width_ratios)
    placed, entries = [], []
    for letter, fn, sl, _ in panels:
        ss = gs[sl]
        previous = set(fig.axes)
        target=Bbox.from_bounds(*rectangles[letter]) if rectangles else ss.get_position(fig)
        # add_axes must receive a copied bounds tuple; reusing the target Bbox
        # aliases it to the axes position and makes iterative fitting collapse.
        ax = fig.add_axes(tuple(target.bounds)) if rectangles else fig.add_subplot(ss)
        original_legend = ax.legend
        def capture_legend(*args, _ax=ax, _original=original_legend, **kwargs):
            _ax._publication_legend = (args, dict(kwargs), _original)
            return _original(*args, **kwargs)
        ax.legend = capture_legend
        fn(ax)
        ax.legend = original_legend
        group = [a for a in fig.axes if a not in previous]
        entries.append((group,target))
        placed.append((letter, ax, ss.rowspan.start, ss.colspan.start))
    _readable_combined_type(fig, stem)
    alignment = _fit_footprints(fig, entries, iterations=10)
    fig._panel_targets={p[0]:entry[1] for p,entry in zip(panels,entries)}
    if len(placed) > 1:
        add_letters(fig, placed)
    _audit_actual_palette(fig,stem)
    fig.savefig(folder / f"{stem}.png", dpi=144 if preview else 600, bbox_inches="tight", facecolor="white", metadata={"Software":"Scientific figure"}, pil_kwargs={"compress_level":1})
    if not preview:fig.savefig(folder / f"{stem}.pdf", bbox_inches="tight", facecolor="white", metadata={"Author":"CHONG LIU","Creator":"CHONG LIU","Producer":"Matplotlib"})
    if True:
        from PIL import Image
        im=Image.open(folder/f"{stem}.png")
        image_width=im.width/(144 if preview else 600.); image_height=im.height/(144 if preview else 600.)
        word_scale=min(5.76/image_width,(17.7/2.54)/image_height)
        geometry={"image_width_in":image_width,"image_height_in":image_height,"height_to_width":image_height/image_width,"word_scale":word_scale,"printed_width_in":image_width*word_scale,"printed_height_in":image_height*word_scale,"panels":{}}
        for letter,ax,_,_ in placed:
            pos=ax.get_position(); hi=pos.height*fig.get_figheight()
            entry={"data_height_native_in":hi,"data_height_word_px144":hi*word_scale*144,"ylim":list(ax.get_ylim())}
            if (stem=="Fig08_viability_atlas" and letter=="d") or (stem=="Fig09_humidity_mechanism" and letter=="b"):
                entry["row_pitch_word_px144"]=entry["data_height_word_px144"]/abs(ax.get_ylim()[1]-ax.get_ylim()[0])
            geometry["panels"][letter]=entry
        from matplotlib.text import Text
        fs=[t.get_fontsize() for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
        geometry["minimum_ordinary_font_word_pt"]=min(fs)*word_scale
        geometry["panels_count"]=len(panels)
        geometry["native_minimum_pt"]=fig._publication_minimum_native
        geometry["alignment"]=alignment
        (qadir/f"{stem}_geometry.json").write_text(json.dumps(geometry,indent=2),encoding="utf8")
    plt.close(fig)
    if preview:
        print(stem,"preview",len(panels),"panels",flush=True)
        return
    lines = ["# Subfigures", "", "Each panel is re-drawn on its own standalone canvas by the same plotting function "
             "(not cropped from the combined figure); panel letters appear only in the combined figure.", ""]
    for letter, fn, _, size in panels:
        f2 = plt.figure(figsize=size)
        fn(f2.add_subplot(111))
        f2.savefig(sub / f"{stem}_{letter}.png", dpi=600, bbox_inches="tight", facecolor="white", metadata={"Software":"Scientific figure"}, pil_kwargs={"compress_level":1})
        f2.savefig(sub / f"{stem}_{letter}.pdf", bbox_inches="tight", facecolor="white", metadata={"Author":"CHONG LIU","Creator":"CHONG LIU","Producer":"Matplotlib"})
        plt.close(f2)
        lines.append(f"- {letter}: {stem}_{letter}.png and {stem}_{letter}.pdf")
    (sub / "README.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    qadir = HERE.parent / "qa_20261107_compact"
    qadir.mkdir(exist_ok=True)
    (qadir/f"{stem}_alignment.json").write_text(json.dumps(alignment,indent=2),encoding="utf8")
    print(stem, "panels", len(panels), "alignment max",max([a['max_error'] for a in alignment] or [0]),flush=True)


def _arrow(ax, x0, y0, x1, y1, lw=1.8, ms=18, color=INK):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=ms, lw=lw, color=color, zorder=2))


def _cube(ax, cx, cy, s, alpha=1.0, z=3):
    f = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * s * 0.5 + [cx - 0.17 * s, cy - 0.14 * s]
    b = f + [0.34 * s, 0.28 * s]
    for P in (b, f):
        ax.add_patch(Polygon(P, closed=True, fc="none", ec=INK, lw=1.3, alpha=alpha, zorder=z))
    for i in range(4):
        ax.plot([f[i, 0], b[i, 0]], [f[i, 1], b[i, 1]], color=INK, lw=1.3, alpha=alpha, zorder=z)
    for P in (b, f):
        for i in range(4):
            m = (P[i] + P[(i + 1) % 4]) / 2
            ax.add_patch(Circle(m, 0.06 * s, fc=PAL[2], ec=INK, lw=0.7, alpha=alpha, zorder=z + 1))
        for x, y in P:
            ax.add_patch(Rectangle((x - 0.085 * s, y - 0.085 * s), 0.17 * s, 0.17 * s, fc=PAL[0], ec=INK, lw=0.9,
                                   alpha=alpha, zorder=z + 2))


def _co2(ax, cx, cy, r, z=8):
    ax.plot([cx - 1.7 * r, cx + 1.7 * r], [cy, cy], color=INK, lw=1.2, zorder=z)
    for dx, c in [(-1.7 * r, OURS), (0, MUTED), (1.7 * r, OURS)]:
        ax.add_patch(Circle((cx + dx, cy), r, fc=c, ec=INK, lw=0.8, zorder=z + 1))


def _column(ax, cx, cy, w, h, seed):
    ax.add_patch(Ellipse((cx, cy - h / 2), w, w * 0.36, fc=PAL[4], ec=INK, lw=1.3, zorder=2))
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, fc=PAL[4], ec="none", zorder=3))
    ax.plot([cx - w / 2, cx - w / 2], [cy - h / 2, cy + h / 2], color=INK, lw=1.3, zorder=4)
    ax.plot([cx + w / 2, cx + w / 2], [cy - h / 2, cy + h / 2], color=INK, lw=1.3, zorder=4)
    ax.add_patch(Ellipse((cx, cy + h / 2), w, w * 0.36, fc=PAL[3], ec=INK, lw=1.3, zorder=5))
    rng = np.random.default_rng(seed)
    xs = rng.uniform(cx - w / 2 + 0.35, cx + w / 2 - 0.35, 30)
    ys = rng.uniform(cy - h / 2 + 0.3, cy + h / 2 - 0.7, 30)
    ax.scatter(xs, ys, s=11, color=PAL[0], edgecolor=INK, linewidth=0.3, zorder=6)


def _thermo(ax, x, y):
    ax.add_patch(FancyBboxPatch((x - 0.35, y), 0.7, 3.2, boxstyle="round,pad=0.02,rounding_size=0.35", fc="white",
                                ec=INK, lw=1.1, zorder=4))
    ax.add_patch(Rectangle((x - 0.16, y + 0.1), 0.32, 2.0, fc=OURS, ec="none", zorder=5))
    ax.add_patch(Circle((x, y), 0.6, fc=OURS, ec=INK, lw=1.1, zorder=6))


def _gauge(ax, x, y, r=1.15):
    ax.add_patch(Circle((x, y), r, fc="white", ec=INK, lw=1.2, zorder=4))
    for a in np.linspace(200, -20, 5):
        t = np.deg2rad(a)
        ax.plot([x + 0.72 * r * np.cos(t), x + 0.92 * r * np.cos(t)], [y + 0.72 * r * np.sin(t), y + 0.92 * r * np.sin(t)],
                color=INK, lw=0.9, zorder=5)
    t = np.deg2rad(140)
    ax.plot([x, x + 0.7 * r * np.cos(t)], [y, y + 0.7 * r * np.sin(t)], color=OURS, lw=1.8, zorder=6)
    ax.add_patch(Circle((x, y), 0.13 * r, fc=INK, ec=INK, zorder=7))


def _smoke(ax, x, y):
    for i, (dx, dy, r) in enumerate([(0.2, 0.7, 0.45), (0.8, 1.5, 0.6), (1.6, 2.3, 0.75)]):
        ax.add_patch(Circle((x + dx, y + dy), r, fc=LIGHT, ec=MUTED, lw=0.8, zorder=2))


def _plant(ax, x0, y0, kind):
    if kind == "cement":
        ax.add_patch(Rectangle((x0 + 0.3, y0 + 2.6), 1.3, 3.6, fc=PAL[3], ec=INK, lw=1.0, zorder=3))
        ax.add_patch(Ellipse((x0 + 0.95, y0 + 6.2), 1.3, 0.5, fc=PAL[3], ec=INK, lw=1.0, zorder=4))
        ax.add_patch(Polygon([[x0 + 1.9, y0 + 2.6], [x0 + 4.6, y0 + 3.6], [x0 + 4.6, y0 + 4.4], [x0 + 1.9, y0 + 3.4]],
                             closed=True, fc=PAL[6], ec=INK, lw=1.0, zorder=3))
        ax.add_patch(Rectangle((x0 + 3.5, y0 + 4.0), 0.55, 2.4, fc=PAL[5], ec=INK, lw=1.0, zorder=3))
        _smoke(ax, x0 + 3.6, y0 + 6.4)
    elif kind == "coal":
        ax.add_patch(Rectangle((x0 + 3.0, y0 + 2.6), 0.8, 5.0, fc=PAL[5], ec=INK, lw=1.0, zorder=3))
        ax.add_patch(Polygon([[x0 + 0.4, y0 + 2.6], [x0 + 0.9, y0 + 5.2], [x0 + 2.3, y0 + 5.2], [x0 + 2.8, y0 + 2.6]],
                             closed=True, fc=PAL[3], ec=INK, lw=1.0, zorder=3))
        _smoke(ax, x0 + 3.1, y0 + 7.6)
    else:
        ax.add_patch(Rectangle((x0 + 3.1, y0 + 2.6), 0.65, 4.2, fc=PAL[5], ec=INK, lw=1.0, zorder=3))
        ax.add_patch(Circle((x0 + 1.5, y0 + 4.1), 1.15, fc=PAL[7], ec=INK, lw=1.0, zorder=3))
        for a in range(0, 360, 60):
            t = np.deg2rad(a)
            ax.plot([x0 + 1.5, x0 + 1.5 + 0.95 * np.cos(t)], [y0 + 4.1, y0 + 4.1 + 0.95 * np.sin(t)], color=INK,
                    lw=0.8, zorder=4)
        _smoke(ax, x0 + 3.2, y0 + 6.8)
    ax.add_patch(Rectangle((x0, y0), 4.8, 2.6, fc=PAL[4], ec=INK, lw=1.1, zorder=4))
    for i in range(3):
        ax.add_patch(Rectangle((x0 + 0.5 + 1.4 * i, y0 + 0.8), 0.7, 0.9, fc="white", ec=INK, lw=0.7, zorder=5))


def _card(ax, x, y, w, h, fc="white", z=3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.6", fc=fc, ec=INK, lw=1.2,
                                zorder=z))


def f1_schematic_legacy(ax):
    ax.set_xlim(0, 120); ax.set_ylim(4.6, 29.6); ax.set_aspect("equal"); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.6, 4.8), 118.8, 24.6, boxstyle="round,pad=0,rounding_size=1.5", fc="#FAFBFC",
                                ec="none", zorder=0))
    cy = 19.0
    # 1 MOF library
    _cube(ax, 7.4, cy + 1.2, 5.4, alpha=0.3, z=2)
    _cube(ax, 10.0, cy - 0.6, 6.4, z=4)
    _co2(ax, 10.0, cy - 0.5, 0.42)
    # 2 process
    _column(ax, 27.5, cy - 0.3, 3.4, 9.0, 1)
    _column(ax, 32.3, cy - 0.3, 3.4, 9.0, 2)
    _thermo(ax, 36.6, cy - 3.6)
    _gauge(ax, 36.7, cy + 3.3)
    ax.add_patch(FancyArrowPatch((23.0, cy - 4.0), (25.6, cy - 4.0), arrowstyle="-|>", mutation_scale=11, lw=1.3,
                                 color=MUTED, zorder=3))
    _co2(ax, 30.0, cy + 6.6, 0.38)
    # 3 sources
    _plant(ax, 43.5, cy - 4.4, "cement")
    _plant(ax, 49.4, cy - 4.4, "coal")
    _plant(ax, 55.3, cy - 4.4, "ngcc")
    # 4 TEA + LCA
    for i in range(4):
        ax.add_patch(Ellipse((70.0, cy - 3.2 + 0.75 * i), 4.2, 1.3, fc=PAL[3], ec=INK, lw=1.0, zorder=3 + i))
    ax.add_patch(Circle((70.0, cy + 2.8), 2.1, fc=PAL[3], ec=INK, lw=1.2, zorder=8))
    ax.text(70.0, cy + 2.75, "€", ha="center", va="center", fontsize=17, color=INK, zorder=9)
    leaf = Ellipse((75.6, cy + 0.4), 2.6, 5.4, angle=-35, fc=PAL[2], ec=INK, lw=1.1, zorder=4)
    ax.add_patch(leaf)
    ax.plot([74.3, 76.9], [cy - 1.4, cy + 2.2], color=INK, lw=1.0, zorder=5)
    # 5 RIV model card
    _card(ax, 82.4, cy - 5.3, 13.0, 11.4, fc=PAL[6])
    xs = np.linspace(-3.2, 3.2, 120)
    dens = np.exp(-0.5 * xs ** 2)
    px, py = 88.9 + xs * 1.8, cy - 3.2 + dens * 6.4
    ax.plot(px, py, color=INK, lw=1.5, zorder=6)
    m0, m1 = xs < -1.3, xs > 0.35
    ax.fill_between(px[m0], cy - 3.2, py[m0], color=MUTED, alpha=0.45, lw=0, zorder=5)
    ax.fill_between(px[m1], cy - 3.2, py[m1], color=OURS, alpha=0.75, lw=0, zorder=5)
    ax.plot([83.6, 94.2], [cy - 3.2, cy - 3.2], color=INK, lw=1.1, zorder=6)
    ax.plot([88.9 + 0.35 * 1.8] * 2, [cy - 3.2, cy + 3.9], color=INK, lw=1.0, ls="--", zorder=6)
    ax.text(89.55, cy + 4.15, "1/p", ha="center", va="bottom", fontsize=11, color=INK, zorder=7)
    # 6 outputs
    for j, yy in enumerate([cy + 2.6, cy - 1.6, cy - 5.8]):
        _card(ax, 102.0, yy, 15.0, 3.6)
    t = np.linspace(0, 1, 60)
    ax.plot(103.2 + t * 6.0, cy + 3.0 + 2.4 * (1 - np.exp(-4 * t)) * 0.9, color=PAL[0], lw=1.8, zorder=6)
    ax.plot([103.2, 109.2], [cy + 3.0 + 2.16, cy + 3.0 + 2.16], color=PAL[0], lw=1.0, ls=":", zorder=6)
    ax.text(110.0, cy + 4.4, "V(p | x)", ha="left", va="center", fontsize=11, color=INK, zorder=6)
    for i in range(3):
        yy = cy - 0.75 + i * 0.95 - 0.6
        ax.add_patch(Rectangle((103.4, yy), 4.6, 0.55, fc=PAL[4], ec=INK, lw=0.6, zorder=6))
        ax.plot([108.5, 108.9, 109.6], [yy + 0.3, yy + 0.05, yy + 0.6], color=OURS, lw=1.5, zorder=6)
    ax.text(110.0, cy - 0.0, "FDR list", ha="left", va="center", fontsize=11, color=INK, zorder=6)
    vals = np.array([[0.2, 0.5, 0.8, 0.4], [0.1, 0.3, 0.6, 0.9]])
    for i in range(2):
        for jj in range(4):
            ax.add_patch(Rectangle((103.4 + jj * 1.35, cy - 5.25 + i * 1.25), 1.25, 1.15, fc=W2R(vals[i, jj]), ec=INK,
                                   lw=0.5, zorder=6))
    ax.text(110.0, cy - 4.0, "Atlas", ha="left", va="center", fontsize=11, color=INK, zorder=6)
    # arrows and labels
    xs_c = [10.0, 31.0, 51.6, 72.4, 88.9, 109.5]
    for a, b in [(15.6, 23.2), (39.0, 42.8), (61.0, 66.6), (78.6, 82.0), (95.8, 101.6)]:
        _arrow(ax, a, cy, b, cy)
    labels = ["1,175 MOFs\nstructure · isotherms", "18 process configs\nTSA · TVSA · dry · wet",
              "3 CO$_2$ sources\n× 10 TEA/LCA contexts", "TEA + LCA\n150,482 nCAC values",
              "One RIV model\ny | x ~ N(μ, σ)", "At every carbon price\nprice · shortlist · atlas"]
    for x, lab in zip(xs_c, labels):
        ax.text(x, 9.6, lab, ha="center", va="top", fontsize=13, color=INK, linespacing=1.35)


def f1_schematic(ax):
    """Original pictorial layer; all scientific text is deterministic and editable."""
    from PIL import Image
    art = Image.open(HERE / "assets" / "workflow_artwork.png")
    ax.imshow(art, extent=(0, 3, 0, 1), aspect="equal")
    ax.set_xlim(0, 3); ax.set_ylim(0.035, 0.865); ax.axis("off")
    centres = [0.080, 0.250, 0.445, 0.620, 0.786, 0.940]
    labels = ["MOF library", "Fixed cycles", "Deployment", "TEA + LCA", "RIV model", "Price decisions"]
    details = ["1,175 MOFs", "18 configurations", "132 scenarios", "150,482 pairs", "y = 1/nCAC", "V(p) · FDR · atlas"]
    for x, lab, detail in zip(centres, labels, details):
        ax.text(x*3, 0.165, lab, ha="center", va="center", fontsize=18.0, color=INK)
        ax.text(x*3, 0.085, detail, ha="center", va="center", fontsize=17.2, color=INK)


def f1_pole(ax):
    cc = np.linspace(0.02, 3.0, 800)
    n = 60.0 / (1 - cc)
    n[np.abs(1 - cc) < 0.01] = np.nan
    ax.axvspan(1, 3.0, color=PAL[6], zorder=0, lw=0)
    ax.plot(cc, n, color=INK, lw=2.4)
    ax.axvline(1, color=OURS, lw=1.6, ls="--")
    ax.set_yscale("symlog", linthresh=10, linscale=0.4)
    ax.set_ylim(-3e4, 3e4)
    ax.set_yticks([-1e4, -1e2, 0, 1e2, 1e4])
    ax.set_xticks([0,1,2,3])
    ax.text(2.0, 1e3, "Net emitter", ha="center", fontsize=11.5, color=INK)
    style_axis(ax,"CC" if combined(ax) else "Life-cycle ratio CC",f"nCAC ({EUR})")


def f1_riv(ax):
    cc = np.linspace(0.02, 3.0, 400)
    y = (1 - cc) / 60.0 * 1000
    ax.axvspan(1, 3.0, color=PAL[6], zorder=0, lw=0)
    ax.plot(cc, y, color=OURS, lw=2.6)
    for p, ls in [(75, ":"), (100, "--"), (150, "-.")]:
        ax.axhline(1000 / p, color=MUTED, lw=1.3, ls=ls)
        if not combined(ax):ax.text(2.95,1000/p+.5,f"p = {p}",ha="right",fontsize=11.5,color=INK)
    if combined(ax):ax.legend([Line2D([],[],color=MUTED,ls=ls,lw=1.3) for ls in [":","--","-."]],["p = 75","p = 100","p = 150"],loc="lower left",fontsize=11.5)
    ax.axhline(0,color=INK,lw=1)
    ax.axvline(1,color=OURS,lw=1.2,ls="--")
    style_axis(ax,"CC" if combined(ax) else "Life-cycle ratio CC","y (t k€$^{-1}$)" if combined(ax) else "1/nCAC (t k€$^{-1}$)")
    numeric_ticks(ax, n=4)


def f1_density(ax):
    mu, sd = 9.0, 5.0
    x = np.linspace(-9, 27, 500)
    d = norm.pdf(x, mu, sd)
    ax.fill_between(x[x <= 0], 0, d[x <= 0], fc=LIGHT, ec=MUTED, hatch="///", lw=0.6)
    ax.fill_between(x[x >= 10], 0, d[x >= 10], color=OURS, alpha=0.55, lw=0)
    ax.plot(x, d, color=INK, lw=2.2)
    ax.axvline(10, color=MUTED, lw=1.2, ls="--", ymax=0.86)
    ax.text(10, 0.1, "p = 100", ha="center", va="bottom", fontsize=11.5, color=INK)
    ax.axvline(0, color=INK, lw=1.0)
    ax.text(-4.5,.045 if combined(ax) else .02,"F(0)" if combined(ax) else "F(0 | x)",ha="center",va="bottom",rotation=0 if combined(ax) else 90,fontsize=11.5,color=INK)
    ax.text(20 if combined(ax) else 16.5,.065 if combined(ax) else .036,"V(100)" if combined(ax) else "V(100 | x)",ha="center" if combined(ax) else "left",va="bottom",fontsize=11.5,color=INK)
    ax.set_ylim(0, 0.115); ax.set_xlim(-9, 27)
    ax.set_xticks([0,10,20])
    style_axis(ax,"y (t k€$^{-1}$)" if combined(ax) else "1/nCAC (t k€$^{-1}$)","Density" if combined(ax) else "Predictive density")
    ax.set_yticks([0, 0.04, 0.08])


def _example_mofs():
    o = oof()
    g = o[o.scenario == "Coal_Storage_UK_TVSA02-dry"].copy()
    g["cure"] = norm.cdf(-g.mu / g.sigma)
    g["pmed"] = 1 / g.mu.where(g.mu > 0)
    mg = g[g.MOF == "MgMOF74"]
    mid = g[(g.cure < 0.02) & (g.y_inv > 0)].assign(d=lambda d: (d.pmed - 110).abs()).nsmallest(1, "d")
    risky = g[(g.cure.between(0.2, 0.5)) & (g.pmed.between(80, 250))].assign(d=lambda d: (d.cure - 0.35).abs()).nsmallest(1, "d")
    return pd.concat([mg, mid, risky])


def f1_curves(ax):
    ex = _example_mofs()
    p = np.geomspace(10, 1000, 400)
    z = norm.ppf(0.95)
    for (_, r), col, lab, yb in zip(ex.iterrows(), [PAL[0], OURS, PAL[2]], ["Mg-MOF-74", "MOF 2", "MOF 3"],
                                     [0.5, 0.5, 0.5]):
        S = norm.sf((1 / p - r.mu) / r.sigma)
        display_lab = {"Mg-MOF-74":"Mg", "MOF 2":"2", "MOF 3":"3"}.get(lab,lab) if combined(ax) else lab
        ax.plot(p, S, color=col, lw=2.4, label=display_lab)
        ax.axhline(1 - r.cure, color=col, lw=1.0, ls=":")
        lo = 1 / (r.mu + z * r.sigma)
        hi = 1 / (r.mu - z * r.sigma) if r.mu - z * r.sigma > 0 else 1000
        ax.plot([lo, hi], [yb, yb], color=col, lw=3.5, alpha=0.5, solid_capstyle="butt")
        if r.mu - z*r.sigma <= 0:
            ax.annotate("", xy=(1000,yb), xytext=(700,yb), arrowprops=dict(arrowstyle="-|>",color=col,lw=1.8,mutation_scale=9))
        ax.scatter([1 / r.mu], [yb], s=60, color=col, edgecolor=INK, lw=0.9, zorder=4)
        if r.y_inv > 0:
            ax.scatter([1 / r.y_inv], [norm.sf((r.y_inv - r.mu) / r.sigma)], marker="x", s=55, color=PAL[5], lw=1.6,
                       zorder=5)
    ax.set_xscale("log"); ax.set_xticks([30, 100, 300, 1000]); ax.set_xticklabels(["30", "100", "300", "1000"])
    ax.minorticks_off()
    if combined(ax):
        ax.set_xlim(10,1300);ax.set_xticks([10,100,1000]);ax.set_xticklabels(["10",r"$10^{2}$",r"$10^{3}$"])
    ax.set_ylim(0,1.03)
    style_axis(ax,f"p ({EUR})" if combined(ax) else f"Carbon price p ({EUR})","V(p | x)" if combined(ax) else "V(p | x) = P(viable at p)")
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    h, l = ax.get_legend_handles_labels()
    ax.legend(h, l, loc="lower center", bbox_to_anchor=(0.5,1.02), ncol=3 if combined(ax) else 1,
              fontsize=9.0 if combined(ax) else 11.5, handlelength=.55 if combined(ax) else 1.2, columnspacing=.25,
              handletextpad=.25, labelspacing=0.15, borderaxespad=0.1)




# =============================================================================== Fig. 2 data landscape
def f2_pole_data(ax):
    k = kdata().sample(12000, random_state=0)
    for s, g in k.groupby("source"):
        ax.scatter(g.CC, g.nCAC, s=6, facecolors="none", edgecolors=SRC[s], linewidths=0.6, label=s)
    ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=100)
    ax.axvline(1, color=INK, lw=1.2, ls="--"); ax.set_ylim(-1e5, 1e5)
    ax.set_yticks([-1e4, -1e2, 0, 1e2, 1e4])
    ax.set_xticks([1e-1, 1e1, 1e3, 1e5])
    style_axis(ax,"CC" if combined(ax) else "Life-cycle ratio CC",f"nCAC ({EUR})")
    ax.legend(loc="upper right" if combined(ax) else "lower right", markerscale=2.5, handletextpad=0.2, fontsize=11.5)


def f2_riv_data(ax):
    k = kdata().sample(12000, random_state=0)
    for s, g in k.groupby("source"):
        ax.scatter(g.CC, g.y_inv * 1000, s=6, facecolors="none", edgecolors=SRC[s], linewidths=0.6)
    ax.set_xscale("log"); ax.axvline(1, color=INK, lw=1.2, ls="--"); ax.axhline(0, color=INK, lw=1.0)
    ax.set_ylim(-12, 32); ax.set_xticks([1e-1, 1e1, 1e3, 1e5])
    style_axis(ax,"CC" if combined(ax) else "Life-cycle ratio CC","y (t k€$^{-1}$)" if combined(ax) else "1/nCAC (t k€$^{-1}$)")
    numeric_ticks(ax, x=False, n=4)


def f2_share(ax):
    k = kdata()
    prices = np.linspace(20, 600, 250)
    for s, g in k.groupby("source"):
        plat = (g.y_inv > 0).mean()
        ax.plot(prices, [(g.y_inv >= 1 / p).mean() for p in prices], color=SRC[s], lw=2.4, label=s)
        ax.axhline(plat, color=SRC[s], lw=1.1, ls=":")
        right = s == "NGCC"
        ax.text(595 if right else 30, plat - 0.015 if right else plat + 0.012, f"{1 - plat:.0%} never viable",
                ha="right" if right else "left", va="top" if right else "bottom", fontsize=11.5, color=INK)
    style_axis(ax, f"Carbon price p ({EUR})", "Share viable at p")
    ax.set_ylim(0, 1.02); numeric_ticks(ax, n=4)
    ax.legend(loc="center right", fontsize=11.5)


def f2_invariance(ax):
    inv = pd.read_csv(R("data_invariance_cv.csv"))
    lab = {"working_capacity": "Work. capacity", "spec_heat": "Regeneration heat", "el_energy": "Vacuum power",
           "eff_recovery": "Eff. recovery", "Climate_Change": "Climate impact", "productivity": "Productivity",
           "purity": "Purity", "recovery": "Recovery"}
    cols = [PAL[0] if l == "process" else OURS for l in inv.layer]
    y = np.arange(len(inv))
    ax.hlines(y, 0, inv.median_cv, color=cols, lw=2.4)
    ax.scatter(inv.median_cv, y, s=70, c=cols, edgecolor=INK, linewidth=0.9, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels([{"Regeneration heat":"Regen. heat","Vacuum power":"Vac. power","Eff. recovery":"Eff. rec.","Climate impact":"Climate"}.get(lab.get(c,c),lab.get(c,c)) if combined(ax) else lab.get(c,c) for c in inv.kpi])
    ax.invert_yaxis()
    ax.set_xlim(-0.02, 0.6)
    style_axis(ax, "Context CV" if combined(ax) else "CV across contexts", None, tick_size=11.5)
    numeric_ticks(ax,y=False,n=4)
    if combined(ax):ax.set_xticks([0,.3,.6])
    hh = [Line2D([], [], lw=0, marker="o", ms=8, mfc=c, mec=INK) for c in (PAL[0], OURS)]
    ax.legend(hh, ["Process", "TEA/LCA"], loc="upper right", fontsize=11.5, handletextpad=0.3)


def f2_heat(ax):
    c = pd.read_csv(R("data_cross_scenario_spearman.csv"), index_col=0)
    key = lambda s: (s.split("_")[0], s.split("_")[-1].split("-")[0], s.split("-")[-1])  # noqa: E731
    order = sorted(c.index, key=key)
    im = ax.imshow(c.loc[order, order].values, cmap=W2B, vmin=0.5, vmax=1, interpolation="nearest", aspect="auto")
    src = [key(s)[0] for s in order]
    edges = [i for i in range(1, len(src)) if src[i] != src[i - 1]]
    for e in edges:
        ax.axhline(e - 0.5, color=INK, lw=1.0); ax.axvline(e - 0.5, color=INK, lw=1.0)
    mids = [(a + b) / 2 for a, b in zip([0] + edges, edges + [len(src)])]
    names = [src[int(m)].replace("NGCC-onshore", "NGCC") for m in mids]
    labels=["Cem." if t=="Cement" else t for t in names] if combined(ax) else names
    ax.set_xticks(mids);ax.set_xticklabels(labels);ax.set_yticks(mids);ax.set_yticklabels(labels,rotation=90,va="center")
    style_axis(ax, None, None, tick_size=11.5)
    hcbar(im,ax,"Spearman ρ" if combined(ax) else "Spearman ρ between scenarios",ticks=[.5,.75,1],pad=.18 if combined(ax) else .1)


def f2_overlap(ax):
    ov = pd.read_csv(R("data_top20_overlap.csv"))
    ax.hist(ov.top20_overlap, bins=np.linspace(0, 1, 21), color=PAL[3])
    edge(ax)
    med = ov.top20_overlap.median()
    ax.axvline(med, color=OURS, lw=2.0, ls="--")
    ax.text(med + 0.03, ax.get_ylim()[1] * 0.93, f"Median {med:.2f}", ha="left", va="top", fontsize=11, color=INK)
    style_axis(ax, "Top-20 overlap", "Scenario pairs")
    numeric_ticks(ax, n=4)




# =============================================================================== Fig. 3 method comparison
def f3_targets(ax):
    s = pd.read_csv(R("exp1_target_parameterisation_summary.csv"), index_col=0)
    mets = [("spearman", "ρ"), ("P@25", "P@25"), ("P@50", "P@50"), ("F1@100", "F1")]
    M = np.array([[s.loc[t, f"{m}_mean"] for m, _ in mets] for t in METH])
    im = ax.imshow(np.clip(M, 0, 1), cmap=W2R, vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            value = (f"{M[i,j]:.3f}" if j==0 else f"{M[i,j]:.2f}")
            if combined(ax):
                value = (f"{M[i,j]:.2f}" if i==0 and j==0 else value).replace("0.",".")
            ax.text(j, i, value.replace("-", "−"), ha="center", va="center", fontsize=13.0, color=INK)
    ax.set_xticks(range(len(mets))); ax.set_xticklabels([l for _, l in mets])
    ax.set_yticks(range(len(METH))); ax.set_yticklabels(["nCAC","Signed\nlog","Hurdle","RIV"] if combined(ax) else [METH_LAB[t] for t in METH])
    style_axis(ax, None, None, tick_size=12)
    if combined(ax): ax.tick_params(axis="x", labelrotation=30)
    hcbar(im, ax, "Median score" if combined(ax) else "Median over scenarios", ticks=[0, 0.5, 1.0], pad=0.22)


def f3_f1(ax):
    e1 = pd.read_csv(R("exp1_target_parameterisation.csv")).set_index("target")
    prices = [40, 50, 60, 75, 100, 125, 150, 200, 250, 300, 400, 500]
    for t in METH:
        ax.plot(prices, [e1.loc[t, f"F1@{p}"] for p in prices], marker="o", ms=5.5, lw=2.2, color=METH[t],
                label=METH_LAB[t], mec=INK, mew=0.6)
    ax.set_xscale("log"); ax.set_xticks([50, 100, 200, 500]); ax.set_xticklabels(["50", "100", "200", "500"])
    ax.minorticks_off()
    style_axis(ax, f"Carbon price ({EUR})", "F1" if combined(ax) else "F1, viable at price")
    ax.set_ylim(0, 1.02)
    if combined(ax):
        h,l=ax.get_legend_handles_labels()
        l=["nCAC","Signed\nlog","Hurdle","RIV"]
        ax.legend(h,l,loc="upper center",bbox_to_anchor=(.5,-.35),ncol=2,fontsize=11.5)
    else:
        ax.legend(loc="center right",bbox_to_anchor=(1,.56),fontsize=11.5)


def f3_survival(ax):
    e3 = pd.read_csv(R("exp3_price_survival.csv"))
    s3 = e3[e3.scenario != "ALL_132"].groupby("model")[["C_index", "IBS"]].mean()
    mk = {"RIV (ours, pooled)": ("o", OURS, "RIV pooled"), "RIV (ours, per-scenario)": ("s", OURS, "RIV single"),
          "Hurdle (pooled)": ("o", PAL[0], "Hurdle pooled"), "Hurdle(clf x lognormal)": ("s", PAL[0], "Hurdle single"),
          "RSF": ("^", PAL[2], "RSF"), "GBSA": ("D", PAL[7], "GBSA"), "CoxNet": ("v", PAL[5], "CoxNet")}
    for m, (sym, col, lab) in mk.items():
        if m in s3.index:
            r = s3.loc[m]
            ax.scatter(r.C_index, r.IBS, marker=sym, s=110, color=col, edgecolor=INK, linewidth=0.9, zorder=3)
            offsets = {"RIV pooled": (10,4), "RIV single": (10,4), "Hurdle pooled": (-10,4), "Hurdle single": (10,4), "RSF": (-28,-14), "GBSA": (10,-5), "CoxNet": (10,-3)}
            if combined(ax):
                label={"RIV pooled":"RIV-P","RIV single":"RIV-S","Hurdle pooled":"H-P","Hurdle single":"H-S"}.get(lab,lab)
                positions={"CoxNet":(.15,.88),"Hurdle single":(.18,.62),"Hurdle pooled":(.10,.48),"RSF":(.08,.35),"GBSA":(.80,.29),"RIV single":(.10,.22),"RIV pooled":(.38,.07)}
                ax.annotate(label,(r.C_index,r.IBS),xytext=positions[lab],textcoords="axes fraction",fontsize=11.2,ha="left",va="center",arrowprops=dict(arrowstyle="-",color=MUTED,lw=.8,shrinkA=2,shrinkB=4))
            else:
                ax.annotate(lab,(r.C_index,r.IBS),xytext=offsets[lab],textcoords="offset points",fontsize=11.2,ha="right" if lab=="Hurdle pooled" else "left")
    ax.set_xlim(0.905, 0.966)
    style_axis(ax, "C-index", "Integrated Brier score")
    numeric_ticks(ax, n=4)
    if combined(ax): ax.set_ylim(.019,.049)


def f3_calibration(ax):
    o = oof()
    for p, col in [(75, PAL[0]), (100, OURS), (150, PAL[2])]:
        pr, t = o[f"Pv@{p}"].values, (o.y_inv.values >= 1 / p).astype(float)
        idx = np.clip((pr * 10).astype(int), 0, 9)
        xs = [pr[idx == i].mean() for i in range(10) if (idx == i).sum() > 50]
        ys = [t[idx == i].mean() for i in range(10) if (idx == i).sum() > 50]
        ax.plot(xs, ys, marker="o", ms=6, lw=2.0, color=col, label=f"{p} {EUR}", mec=INK, mew=0.6)
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1.2, ls="--")
    style_axis(ax, "Predicted P(viable)", "Observed frequency")
    numeric_ticks(ax, n=5)
    if combined(ax):
        ax.set_xticks([0,.5,1]);ax.set_yticks([0,.5,1])
    if combined(ax):
        h,l=ax.get_legend_handles_labels()
        ax.legend(h,["75","100","150"],loc="upper center",bbox_to_anchor=(.5,-.3),ncol=3,title=f"p ({EUR})",fontsize=11.5)
    else:ax.legend(loc="upper left",fontsize=11.5)


def f3_cure(ax):
    e = pd.read_csv(R("exp14_entry_price.csv"))
    for s, d in e.groupby("source"):
        ax.scatter(d.cure_sim, d.cure_pred, s=40, facecolors="none", edgecolors=SRC[s], linewidths=1.4, label=s)
    ax.plot([0, 0.8], [0, 0.8], color=MUTED, lw=1.2, ls="--")
    ax.text(0.97, 0.05, f"MAE {np.abs(e.cure_sim - e.cure_pred).mean():.3f}", ha="right", fontsize=11.5,
            transform=ax.transAxes)
    ax.set_xlim(0, 0.8); ax.set_ylim(0, 0.8)
    style_axis(ax,"Simulated cure share" if combined(ax) else "Simulated never-viable share","Predicted cure share" if combined(ax) else "Predicted mean F(0 | x)")
    numeric_ticks(ax, n=4)
    if combined(ax): ax.set_xticks([0,.4,.8]);ax.set_yticks([0,.4,.8])
    ax.legend(loc="upper left", fontsize=11.5)


def f3_breakeven(ax):
    o = oof()
    m = (o.y_inv > 0) & (o.mu > 0)
    ps, pp = 1 / o.y_inv[m], 1 / o.mu[m]
    k = (ps.between(20, 1000)) & (pp.between(20, 1000))
    hb = ax.hexbin(ps[k], pp[k], gridsize=55, xscale="log", yscale="log", bins="log", cmap=W2B, mincnt=1,
                   extent=(np.log10(20), 3, np.log10(20), 3), linewidths=0.15, edgecolors="face")
    ax.plot([20, 1000], [20, 1000], color=INK, lw=1.1, ls="--")
    for a in (ax.set_xticks, ax.set_yticks):
        a([30, 100, 300, 1000])
    ax.set_xticklabels(["30", "100", "300", "1000"]); ax.set_yticklabels(["30", "100", "300", "1000"])
    ax.minorticks_off()
    style_axis(ax, f"Simulated price ({EUR})", "Predicted 1/μ")
    hcbar(hb, ax, "Pairs per hexagon", pad=0.24)




# =============================================================================== Fig. 4 one model, transfer
def f4_budget(ax):
    e4 = pd.read_csv(R("exp4_hierarchical_error_budget.csv")).set_index("model")
    order = ["Direct", "Hier-oof", "Hier+direct", "Oracle-B"]
    lab = ["Direct","Hier.","Combined","Process\nKPIs"] if combined(ax) else ["Direct","Hierarchical","Hier. + direct","Simulated\nprocess outputs"]
    y = np.arange(len(order))[::-1]
    r2, sp = e4.loc[order, "R2"].values, e4.loc[order, "median_scenario_spearman"].values
    ax.hlines(y, r2, sp, color=MUTED, lw=1.6, zorder=1)
    ax.scatter(r2, y, marker="o", s=85, color=PAL[3], edgecolor=INK, lw=0.9, zorder=3, label="R$^{2}$")
    ax.scatter(sp, y, marker="D", s=75, color=OURS, edgecolor=INK, lw=0.9, zorder=3, label="ρ" if combined(ax) else "Spearman")
    ax.set_yticks(y); ax.set_yticklabels(lab)
    ax.set_xlim(0.925, 1.006); ax.set_ylim(-0.6, len(order) - 0.2)
    style_axis(ax,"Unseen-MOF score" if combined(ax) else "Score, unseen MOFs",None,tick_size=11.5)
    numeric_ticks(ax, y=False, n=4)
    if combined(ax):ax.set_xticks([.94,.97,1]);ax.set_xticklabels([".94",".97","1.00"])
    ax.legend(loc="lower center",bbox_to_anchor=(.5,1),ncol=2,fontsize=11.5,handletextpad=.2)


def f4_stageA(ax):
    e4a = pd.read_csv(R("exp4_stageA_process_R2.csv"), index_col=0).iloc[:, 0]
    nm = {"purity": "Purity", "recovery": "Recovery", "log_wc": "Work. capacity", "productivity": "Productivity",
          "log_spec_heat": "Regeneration heat", "el_energy": "Vacuum power", "rho_b": "Bed density", "log_CO2_capt": "CO$_2$ captured"}
    s = e4a.rename(nm).sort_values()
    y = np.arange(len(s))
    ax.hlines(y, 0.8, s.values, color=PAL[0], lw=2.4)
    ax.scatter(s.values, y, s=70, color=PAL[0], edgecolor=INK, lw=0.9, zorder=3)
    for i, v in enumerate(s.values):
        ax.text(v + 0.006, i, f"{v:.2f}", va="center", fontsize=11.5, color=INK)
    ax.set_yticks(y);ax.set_yticklabels([{"Regeneration heat":"Regen. heat","Vacuum power":"Vac. power","Work. capacity":"Work. cap."}.get(t,t) for t in s.index] if combined(ax) else s.index)
    ax.set_xlim(0.8, 1.03)
    style_axis(ax, "Stage-A R$^{2}$" if combined(ax) else "Stage-A R$^{2}$, unseen MOFs", None, tick_size=11.5)
    ax.set_xticks([0.8, 0.9, 1.0])


def f4_tiers(ax):
    t = pd.read_csv(R("exp9_input_tiers.csv"))
    x = np.arange(3)
    for col, c, mk, lab in [("median_scenario_spearman", PAL[0], "o", "Spearman"), ("P_at_25", OURS, "D", "P@25")]:
        ax.plot(x, t[col], color=c, lw=2.2, marker=mk, ms=8, mec=INK, mew=0.9, label=lab)
        for xi, v in zip(x, t[col]):
            ax.text(xi, v + 0.045, f"{v:.2f}", ha="center", fontsize=11.5, color=INK)
    ax.set_xticks(x);ax.set_xticklabels(["Struct.","+\nGCMC","+\nWater"] if combined(ax) else ["Structure","+ Adsorption","+ Water"])
    ax.set_xlim(-0.3, 2.3); ax.set_ylim(0, 1.12)
    style_axis(ax, "Input tier", "Median per scenario")
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.legend(loc="lower right", fontsize=11.5)


def f4_region(ax):
    raw = pd.read_csv(R("exp4_loro_fewshot.csv"))
    cm = {"Direct-kshot": (PAL[3], "Direct", "s"), "StageB-kshot": (PAL[0], "Stage B", "^"),
          "Donor-ridge": (OURS, "Donor ridge", "o")}
    for m, d in raw.groupby("model"):
        g = d.groupby("k").R2
        ax.fill_between(g.mean().index, g.min(), g.max(), color=cm[m][0], alpha=0.18, lw=0)
        ax.plot(g.mean().index, g.mean(), marker=cm[m][2], ms=7, lw=2.2, color=cm[m][0], label=cm[m][1], mec=INK,
                mew=0.8)
    ax.set_xticks([0, 10, 50])
    style_axis(ax, "Anchor MOFs" if combined(ax) else "Anchor MOFs, new region", "R$^{2}$")
    numeric_ticks(ax, x=False, n=4)
    ax.legend(loc="lower right", fontsize=11.5)


def f4_config(ax):
    raw = pd.read_csv(R("exp7_config_transfer.csv"))
    for (sp, enc), d in raw.groupby(["split", "encoding"]):
        g = d.groupby("k").R2
        c = OURS if enc == "physics" else PAL[0]
        ls = "-" if sp.startswith("LOSO") else "--"
        ax.fill_between(g.mean().index, g.min(), g.max(), color=c, alpha=0.12, lw=0)
        ax.plot(g.mean().index, g.mean(), marker="o" if enc == "physics" else "s", ms=6.5, lw=2.2, color=c, ls=ls,
                mec=INK, mew=0.8, label=f"{'New source' if sp.startswith('LOSO') else 'New vacuum'}, "
                                        f"{'physical' if enc == 'physics' else 'indicators'}")
    ax.set_xticks([0, 10, 50])
    style_axis(ax, "Anchor MOFs" if combined(ax) else "Anchor MOFs, new configuration", "R$^{2}$, unseen MOFs")
    numeric_ticks(ax, x=False, n=4)
    if combined(ax):
        h,l=ax.get_legend_handles_labels()
        l=[t.replace("New vacuum, indicators","Vac-ind.").replace("New vacuum, physical","Vac-phys.").replace("New source, indicators","Src-ind.").replace("New source, physical","Src-phys.") for t in l]
        ax.legend(h,l,loc="upper center",bbox_to_anchor=(.5,-.3),ncol=2,fontsize=10.8)
    else:ax.legend(loc="lower right",fontsize=10.8)


def f4_feedback(ax):
    fb = pd.read_parquet(R("exp6_feedback_preds.parquet"))
    fb["src"] = fb.scenario.str.split("_").str[1].str.replace("NGCC-onshore", "NGCC")
    for s, d in fb.groupby("src"):
        ax.scatter(d.y_inv * 1000, d.mu * 1000, s=16, facecolors="none", edgecolors=SRC[s], linewidths=0.9, label=s)
    lim = [min(fb.y_inv.min(), fb.mu.min()) * 1000, max(fb.y_inv.max(), fb.mu.max()) * 1000]
    ax.plot(lim, lim, color=MUTED, lw=1.2)
    style_axis(ax,"Simulated RIV\n(t k€$^{-1}$)" if combined(ax) else "Simulated 1/nCAC (t k€$^{-1}$)","Predicted RIV\n(t k€$^{-1}$)" if combined(ax) else "Predicted 1/nCAC (t k€$^{-1}$)")
    numeric_ticks(ax, n=4)
    ax.legend(loc="upper left", markerscale=1.6, fontsize=11.5)




# =============================================================================== Fig. 7 causal + mechanism (function fig5)
LEARN = ["S-learner", "X-learner", "T-learner", "DML (R-learner)", "DR-learner"]


def f5_pehe(ax):
    e5 = pd.read_csv(R("exp5_cate_ground_truth.csv"))
    ite = pd.read_parquet(R("exp5_true_ite_humidity.parquet"))
    c = e5[~e5.estimator.str.startswith("ATE") & e5.design.str.startswith("humidity")]
    pe = c.groupby(["estimator", "gamma"]).PEHE.mean().unstack().loc[LEARN]
    x = np.arange(len(LEARN))
    ax.vlines(x, pe[0.0], pe[2.0], color=MUTED, lw=1.6, zorder=1)
    ax.scatter(x, pe[0.0], s=80, marker="o", color=PAL[0], edgecolor=INK, lw=0.9, zorder=3, label="Randomised")
    ax.scatter(x, pe[2.0], s=75, marker="D", color=OURS, edgecolor=INK, lw=0.9, zorder=3, label="Confounded")
    ax.axhline(ite.ite.std(), color=MUTED, ls="--", lw=1.2)
    ax.text(-0.35, ite.ite.std() + 0.12, "SD of true effects", fontsize=11.5, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(["S", "X", "T", "R", "DR"]); ax.set_xlim(-0.5, 4.5); ax.set_ylim(0, 6.4)
    style_axis(ax, "CATE learner", "PEHE (t k€$^{-1}$)")
    numeric_ticks(ax, x=False, n=4)
    ax.legend(loc="upper left", fontsize=11.5, bbox_to_anchor=(0.0, 0.92))


def f5_ate(ax):
    e5 = pd.read_csv(R("exp5_cate_ground_truth.csv"))
    a = e5[e5.estimator.str.startswith("ATE") & e5.design.str.startswith("humidity")]
    ab = a.groupby(["estimator", "gamma"]).ATE_bias.mean().unstack().loc[["ATE_naive", "ATE_AIPW", "ATE_DML_PLR"]]
    x = np.arange(3)
    ax.axhspan(-a.ATE_true.mean(), 4.2, color=PAL[6], lw=0, zorder=0)
    ax.text(2.45, 3.9, "Sign reversed", ha="right", va="top", fontsize=11.5, color=INK)
    ax.axhline(0, color=INK, lw=1.0)
    ax.vlines(x, ab[0.0], ab[2.0], color=MUTED, lw=1.6, zorder=1)
    ax.scatter(x, ab[0.0], s=80, marker="o", color=PAL[0], edgecolor=INK, lw=0.9, zorder=3, label="Randomised")
    ax.scatter(x, ab[2.0], s=75, marker="D", color=OURS, edgecolor=INK, lw=0.9, zorder=3, label="Confounded")
    ax.set_xticks(x); ax.set_xticklabels(["Naive","AIPW","DML\nPLR"] if combined(ax) else ["Naive","AIPW","DML-PLR"]); ax.set_xlim(-0.5, 2.5); ax.set_ylim(-1.0, 4.2)
    style_axis(ax, "Effect estimator", "Deviation from\nmean effect (t k€$^{-1}$)" if combined(ax) else "Deviation from mean effect (t k€$^{-1}$)")
    numeric_ticks(ax, x=False, n=5)
    ax.legend(loc="center right", fontsize=11.5)


def f5_ite(ax):
    ite = pd.read_parquet(R("exp5_true_ite_humidity.parquet"))
    xs = np.linspace(-8, 2, 400)
    for s, d in ite.groupby("source"):
        kde = gaussian_kde(d.ite.values, bw_method=0.12)
        ax.fill_between(xs, kde(xs), color=SRC[s], alpha=0.35, lw=0)
        ax.plot(xs, kde(xs), color=SRC[s], lw=2.0, label=s)
    ax.axvline(0, color=INK, lw=1.0)
    ax.set_ylim(0, None)
    style_axis(ax, "Humidity effect\n(t k€$^{-1}$)" if combined(ax) else "True humidity effect (t k€$^{-1}$)", "Density")
    numeric_ticks(ax, n=5)
    ax.legend(loc="upper left", fontsize=11.5)


MED_G = ["specific heat", "working capacity/productivity", "direct (TEA humidity flag)", "captured CO2/bed",
         "purity/recovery", "vacuum power"]
MED_L = ["Regeneration heat", "Work. cap./productivity", "Humidity indicator", "Captured CO$_2$/bed", "Purity/recovery", "Vacuum power"]
MED_C = [OURS, PAL[0], PAL[2], PAL[3], PAL[5], PAL[7]]


def f5_mediation(ax):
    med = pd.read_parquet(R("exp5_mediation_shapley.parquet"))
    mm = med.groupby("source")[MED_G].mean()
    left = np.zeros(len(mm))
    for g_,l_,c_ in zip(MED_G,["Heat","Wcap/prod","Humidity","CO$_2$/bed","Pur/rec","Vacuum"] if combined(ax) else MED_L,MED_C):
        ax.barh(np.arange(len(mm)), mm[g_].values, left=left, height=0.62, color=c_, edgecolor=INK, lw=0.8, label=l_)
        left += mm[g_].values
    ax.set_yticks(np.arange(len(mm))); ax.set_yticklabels(mm.index)
    ax.axvline(0, color=INK, lw=1.0)
    ax.invert_xaxis()
    style_axis(ax, "Attribution (t k€$^{-1}$)" if combined(ax) else "Process-group attribution (t k€$^{-1}$)", None)
    numeric_ticks(ax,y=False,n=4)
    if combined(ax):ax.set_xticks([0,-.9,-1.8]);ax.set_xticklabels(["0","−.9","−1.8"])
    ax.legend(loc="upper center",bbox_to_anchor=(.45,-.4 if combined(ax) else -.22), ncol=2, fontsize=10, handlelength=1.2,
              columnspacing=0.8)


def f5_shap(ax):
    shi = pd.read_parquet(R("exp8_shap_humidity_ite.parquet"))
    rn = {"w_UCO2bin": "CO$_2$ uptake, humid", "w_Qdes": "Water desorption heat", "log_w_WRC": "Water resistance",
          "log_iso_CO2_0p13bar": "CO$_2$ uptake, 0.13 bar", "log_iso_CO2_0p2bar": "CO$_2$ uptake, 0.2 bar",
          "log_iso_CO2_0p04bar": "CO$_2$ uptake, 0.04 bar", "log_H_CO2": "CO$_2$ Henry", "w_Udes": "Water desorbed",
          "log_w_UCO2ter": "CO$_2$ uptake, ternary", "Q_CO2": "CO$_2$ heat", "Cp": "Heat capacity", "density": "Density",
          "log_H_sel": "Henry selectivity", "w_Qh": "Water Henry heat", "log_w_H": "Water Henry"}
    cols = [c for c in shi.columns if not c.startswith(("process_", "source_"))]
    imp = shi[cols].abs().mean().sort_values(ascending=False).head(8)[::-1]
    y = np.arange(len(imp))
    ax.hlines(y, 0, imp.values, color=PAL[0], lw=2.4)
    ax.scatter(imp.values, y, s=70, color=PAL[0], edgecolor=INK, lw=0.9, zorder=3)
    short={"w_UCO2bin":"q, humid","w_Qdes":"H$_2$O des. heat","log_w_WRC":"Water resist.","log_iso_CO2_0p13bar":"q, 0.13 bar","log_iso_CO2_0p2bar":"q, 0.2 bar","log_iso_CO2_0p04bar":"q, 0.04 bar","log_H_CO2":"CO$_2$ Henry","Q_CO2":"CO$_2$ heat"}
    ax.set_yticks(y);ax.set_yticklabels([short.get(c,rn.get(c,c)) if combined(ax) else rn.get(c,c) for c in imp.index])
    ax.set_xlim(0, imp.max() * 1.1)
    style_axis(ax, "Mean |SHAP|\n(t k€$^{-1}$)" if combined(ax) else "Mean |SHAP| (t k€$^{-1}$)", None, tick_size=11)
    numeric_ticks(ax, y=False, n=4)


def f5_humid_price(ax):
    e = pd.read_csv(R("exp14_entry_price.csv"))
    e["ctx"] = e.scenario.str.replace(r"-(wet|dry)$", "", regex=True)
    hp = pd.read_csv(R("exp14_humidity_price.csv"))
    hp = hp[hp.k == 1]
    rows = []
    for (s, p), d in e.groupby(["source", "process"]):
        w = d.pivot_table(index="ctx", columns="humidity", values="p1_sim").dropna()
        dd = hp[(hp.source == s) & (hp.process == p)]
        rows.append(dict(lab=f"{s} {p.replace('TVSA0', 'TVSA0.')}", dry=w.dry.median(), wet=w.wet.median(),
                         dsim=dd[dd.kind == "sim"].delta.median(), dpred=dd[dd.kind == "pred"].delta.median()))
    r = pd.DataFrame(rows)[::-1].reset_index(drop=True)
    y = np.arange(len(r))
    ax.hlines(y, r.dry, r.wet, color=MUTED, lw=1.6, zorder=1)
    ax.scatter(r.dry, y, s=70, marker="o", color=PAL[0], edgecolor=INK, lw=0.9, zorder=3, label="Dry")
    ax.scatter(r.wet, y, s=70, marker="D", color=OURS, edgecolor=INK, lw=0.9, zorder=3, label="Wet")
    for i, q in r.iterrows():
        ax.text(1150, i, f"+{q.dsim:.0f} ({q.dpred:.0f})", va="center", ha="left", fontsize=10, color=INK)
    ax.set_xscale("log"); ax.set_xlim(30, 1100)
    ax.set_xticks([30, 100, 300, 1000]); ax.set_xticklabels(["30", "100", "300", "1000"]); ax.minorticks_off()
    ax.set_yticks(y);ax.set_yticklabels([t.replace("Cement","Cem.").replace("TVSA","TV") for t in r.lab] if combined(ax) else r.lab)
    if combined(ax):ax.tick_params(axis="x",labelrotation=45)
    style_axis(ax, f"Price for 1% viable ({EUR})", None, tick_size=11)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=11.5, handletextpad=0.2)




# =============================================================================== Fig. 6 viability atlas
def f6_curves(ax):
    o = oof()
    prices = np.array([40, 50, 60, 75, 100, 125, 150, 200, 250, 300, 400, 500])
    for s, g in o.groupby("source"):
        ax.plot(prices, [(g.y_inv >= 1 / p).mean() for p in prices], color=SRC[s], lw=2.4, label=f"{s}")
        ax.plot(prices, [g[f"Pv@{int(p)}"].mean() for p in prices], color=SRC[s], lw=0, marker="o", ms=6,
                markerfacecolor="white", markeredgewidth=1.6)
    ax.set_xscale("log"); ax.set_xticks([50, 100, 200, 500]); ax.set_xticklabels(["50", "100", "200", "500"])
    ax.minorticks_off()
    style_axis(ax, f"Carbon price ({EUR})", "Share of MOFs viable")
    ax.set_ylim(0, 1)
    numeric_ticks(ax, x=False, n=5)
    h, l = ax.get_legend_handles_labels()
    if not combined(ax):h += [Line2D([], [], color=INK, lw=2.2),Line2D([], [], color=INK,lw=0,marker="o",markerfacecolor="white")]
    if not combined(ax):l += ["Simulated","Predicted"]
    ax.legend(h, l, loc="upper left", fontsize=11.2)


def f6_scen_cal(ax):
    sc = pd.read_csv(R("exp10_scenario_atlas.csv"))
    for p, col in [(75, PAL[0]), (100, OURS), (150, PAL[2])]:
        ax.scatter(sc[f"true_share@{p}"], sc[f"pred_share@{p}"], s=26, facecolors="none", edgecolors=col, linewidths=1.2,
                   label=f"{p} {EUR}")
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1.2, ls="--")
    mae = np.mean([np.abs(sc[f"true_share@{p}"] - sc[f"pred_share@{p}"]).mean() for p in (75, 100, 150)])
    ax.text(0.97, 0.05, f"MAE {mae:.3f}", ha="right", fontsize=11.5, transform=ax.transAxes)
    style_axis(ax,"Simulated share" if combined(ax) else "Simulated share viable","Predicted share" if combined(ax) else "Predicted share viable")
    numeric_ticks(ax, n=5)
    if combined(ax):ax.set_xticks([0,.5,1]);ax.set_yticks([0,.5,1])
    ax.legend(loc="upper left", fontsize=11.5)


def _ctx(r):
    reg = r.region
    if r.boiler != "default":
        reg += "-" + ("EL" if r.boiler == "ELboiler" else "NG")
    if str(r.heat_extraction) == "False":
        reg += "*"
    return reg


def f6_price_heat(ax):
    e = pd.read_csv(R("exp14_entry_price.csv"))
    e["row"] = e.source + " " + e.process.str.replace("TVSA0", "TVSA0.") + " " + e.humidity
    e["col"] = e.apply(_ctx, axis=1)
    piv = e.pivot_table(index="row", columns="col", values="p1_pred")
    exist = e.pivot_table(index="row", columns="col", values="n")
    rows = sorted(exist.index, key=lambda s: (s.split()[0], s.split()[1], s.split()[2]))
    cols = sorted(exist.columns, key=lambda s: (s.replace("*", "~")))
    M = piv.reindex(index=rows, columns=cols).values
    E = exist.reindex(index=rows, columns=cols).notna().values
    norm_ = LogNorm(30, 700)
    im = ax.imshow(np.ma.masked_invalid(M), cmap=W2R, norm=norm_, aspect="auto")
    ax.set_facecolor("white")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=13.0, color=INK)
            elif E[i, j]:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fc=LIGHT, ec=MUTED, hatch="///", lw=0.4))
                ax.text(j, i, ">10$^{3}$", ha="center", va="center", fontsize=8.5, color=INK)
            else:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fc=LIGHT, ec="none"))
    for i in (5.5, 11.5):
        ax.axhline(i, color=INK, lw=1.0)
    ax.set_xticks(np.arange(len(cols))); ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(rows))); ax.set_yticklabels(rows)
    style_axis(ax, None, None, tick_size=10.5)
    hcbar(im, ax, f"Price for 1% viability ({EUR})", ticks=[30, 100, 300],
          ticklabels=["30","100","300"],pad=.18 if combined(ax) else .16,fraction=.035)


def f6_robust(ax):
    allr = pd.read_csv(R("exp10_robust_candidates.csv"))
    rob = allr.head(11)
    best = allr.loc[[allr["true_frac@100"].idxmax()]]  # simulated most robust MOF, shown even if under-predicted
    if best.MOF.iloc[0] not in set(rob.MOF):
        rob = pd.concat([rob, best])
    rob = rob[::-1]
    y = np.arange(len(rob))
    isb = [m == best.MOF.iloc[0] for m in rob.MOF]
    ax.hlines(y, rob["pred_P@100"], rob["true_frac@100"], color=MUTED, lw=1.4, zorder=1)
    ax.scatter(rob["pred_P@100"], y, s=70, marker="o", color=[PAL[2] if b else PAL[0] for b in isb], edgecolor=INK,
               lw=0.9, zorder=3)
    ax.scatter(rob["true_frac@100"], y, s=60, marker="D", color=OURS, edgecolor=INK, lw=0.9, zorder=3)
    ax.set_yticks(y)
    if combined(ax):
        label_map={m:f"M{i+1:02d}" for i,m in enumerate(allr.head(11).MOF)}
        label_map[best.MOF.iloc[0]]="Mg-MOF-74"
        ax.set_yticklabels([label_map[m] for m in rob.MOF])
        identifiers_path = HERE.parent/"qa_20261106"/"readability_final"/"Table_S6b_MOF_identifiers.csv"
        identifiers_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"Short code":label_map[m],"Full material identifier":m} for m in rob.MOF]).to_csv(identifiers_path,index=False)
    else:ax.set_yticklabels(["Mg-MOF-74" if b else str(m)[:22] for m,b in zip(rob.MOF,isb)])
    ax.set_xlim(0.4, 0.8)
    style_axis(ax, f"Mean viability\nat 100 {EUR}" if combined(ax) else f"Viable at 100 {EUR}, mean", None, tick_size=10)
    numeric_ticks(ax,y=False,n=4)
    if combined(ax):ax.set_xticks([.4,.6,.8])
    hh = [Line2D([], [], lw=0, marker="o", markerfacecolor=PAL[0], markeredgecolor=INK, ms=8),
          Line2D([], [], lw=0, marker="D", markerfacecolor=OURS, markeredgecolor=INK, ms=7)]
    ax.legend(hh,["Pred.","Sim."] if combined(ax) else ["Predicted P(viable)","Simulated share"], loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2,
              fontsize=10, handletextpad=0.3, columnspacing=0.8)


def f6_gate(ax):
    g = pd.read_csv(R("exp10_purity_gate.csv"))
    order = ["RIV without gate", "pi x RIV (one model for all prices)", "direct classifier at this price"]
    lab = ["No purity gate","π × RIV","Price classifier"] if combined(ax) else ["RIV, no gate","π × RIV, one model","Classifier per price"]
    sty = [(PAL[3], "s", "--"), (OURS, "D", "-"), (PAL[0], "o", "-")]
    prices = sorted(g.price.unique())
    for m, l, (c, mk, ls) in zip(order, lab, sty):
        d = g[g.method == m].set_index("price").loc[prices]
        ax.plot(prices, d.Brier, color=c, marker=mk, ms=7.5, lw=2.0, ls=ls, mec=INK, mew=0.8, label=l)
    ax.set_yscale("log")
    ax.set_xticks(prices); ax.set_xticklabels([f"{int(p)}" for p in prices]); ax.minorticks_off()
    ax.set_yticks([0.005, 0.01, 0.03, 0.1]); ax.set_yticklabels(["0.005", "0.01", "0.03", "0.1"])
    style_axis(ax, f"Carbon price ({EUR})", "Brier score")
    ax.legend(loc="center right", fontsize=10)




# =============================================================================== Fig. 5 tail extrapolation + shortlists (function fig7)
V7 = ["base", "+node", "+phys", "linear", "linear+phys"]
V7_ALL = ["base", "+node", "+phys", "linear", "linear+node", "linear+phys", "linear+phys+mono"]
V7_LAB = {"base": "Const.\nleaves", "+node": "+ Metal\nnodes", "+phys": "+ Phys.\nproxy", "linear": "Linear\nleaves",
          "linear+node": "Linear\n+ node", "linear+phys": "Linear\n+ proxy", "linear+phys+mono": "Linear + proxy\n+ monotone"}
V7_COL = {"base": PAL[3], "+node": PAL[5], "+phys": PAL[2], "linear": PAL[0], "linear+node": PAL[7], "linear+phys": OURS,
          "linear+phys+mono": PAL[6]}
MG = "MgMOF74"


def tail_labels(ax):
    return ["C","C+M","C+P","L","L+P"] if combined(ax) else [V7_LAB[v] for v in V7]


def _tail_cv():
    return pd.read_csv(R("exp12_tail_cv.csv")).set_index("variant")


def f7_ceiling(ax):
    o = pd.read_parquet(R("exp12_oof.parquet"))
    o["rk"] = o.groupby("scenario").y_inv.rank(ascending=False)
    top = o[o.rk <= 20]
    bins = np.linspace(top.y_inv.min(), top.y_inv.max(), 13)
    mid = 0.5 * (bins[1:] + bins[:-1]) * 1000
    for v, c in [("base", V7_COL["base"]), ("linear+phys", V7_COL["linear+phys"])]:
        d = top[top.MOF != MG]
        ax.scatter(d.y_inv * 1000, d[v] * 1000, s=7, color=c, alpha=0.22, lw=0)
        med = top.groupby(pd.cut(top.y_inv, bins), observed=False)[v].median().values * 1000
        ax.plot(mid, med, color=c, lw=2.6, label=("C" if v=="base" else "L+P") if combined(ax) else V7_LAB[v].replace(chr(10)," ")+", bin median")
    for v, fc in [("base", "white"), ("linear+phys", OURS)]:
        d = top[top.MOF == MG]
        ax.scatter(d.y_inv * 1000, d[v] * 1000, s=44, marker="D", facecolors=fc, edgecolors=INK, linewidths=0.9, zorder=4,
                   label="_nolegend_" if combined(ax) else "Mg-MOF-74, "+("constant leaves" if v=="base" else "linear + proxy"))
    lim = [top.y_inv.min() * 1000, top.y_inv.max() * 1000 * 1.04]
    ax.plot(lim, lim, color=MUTED, lw=1.2, ls="--")
    ax.set_xlim(lim); ax.set_ylim(0, lim[1])
    style_axis(ax,"Simulated RIV\n(t k€$^{-1}$)" if combined(ax) else "Simulated 1/nCAC (t k€$^{-1}$)","Predicted RIV\n(t k€$^{-1}$)" if combined(ax) else "Predicted 1/nCAC (t k€$^{-1}$)")
    numeric_ticks(ax, n=4)
    ax.legend(loc="upper left",ncol=2 if combined(ax) else 1,fontsize=10.4,handletextpad=.3)


def f7_mg_rank(ax):
    o = pd.read_parquet(R("exp12_oof.parquet"))
    data = []
    for v in V7:
        rk = o.groupby("scenario")[v].rank(ascending=False)
        data.append(rk[o.MOF == MG].values)
    bp = ax.boxplot(data, positions=np.arange(len(V7)), widths=0.55, patch_artist=True, showfliers=False,
                    medianprops=dict(color=INK, lw=1.6), whiskerprops=dict(color=INK), capprops=dict(color=INK))
    for p_, v in zip(bp["boxes"], V7):
        p_.set_facecolor(V7_COL[v]); p_.set_edgecolor(INK)
    for i, d in enumerate(data):
        ax.scatter(i + np.random.default_rng(i).uniform(-0.18, 0.18, len(d)), d, s=7, color=V7_COL[V7[i]], edgecolor=INK, linewidth=0.2, alpha=0.55, zorder=3)
    ax.set_yscale("log"); ax.invert_yaxis()
    ax.axhline(10, color=MUTED, lw=1.1, ls="--")
    ax.set_yticks([1, 10, 100, 1000]); ax.set_yticklabels(["1", "10", "100", "1000"]); ax.minorticks_off()
    ax.set_xticks(np.arange(len(V7))); ax.set_xticklabels(tail_labels(ax),fontsize=10,rotation=75 if combined(ax) else 0,ha="right" if combined(ax) else "center")
    style_axis(ax, None, "Mg-MOF-74 rank" if combined(ax) else "Predicted rank of Mg-MOF-74", tick_size=10.5)


def f7_precision(ax):
    t = _tail_cv().loc[V7]
    x = np.arange(len(V7))
    for col, c, mk in [("P@25", PAL[0], "o"), ("P@50", OURS, "D")]:
        ax.plot(x, t[col], color=c, lw=2.0, marker=mk, ms=8, mec=INK, mew=0.9, label=col)
    ax.set_xticks(x); ax.set_xticklabels(tail_labels(ax),fontsize=10,rotation=75 if combined(ax) else 0,ha="right" if combined(ax) else "center")
    ax.set_ylim(0.6, 0.8); ax.set_xlim(-0.4, len(V7) - 0.6)
    style_axis(ax, None, "Median top-k precision", tick_size=10.5)
    numeric_ticks(ax, x=False, n=4)
    ax.legend(loc="lower center" if combined(ax) else "upper left",bbox_to_anchor=(.5,1.01) if combined(ax) else None,ncol=2,fontsize=11.2)


def f7_regret(ax):
    t = _tail_cv().loc[V7]
    x = np.arange(len(V7))
    ax.vlines(x, t["regret_best10_pct"], t["regret_top1_pct"], color=MUTED, lw=1.6, zorder=1)
    ax.scatter(x, t["regret_top1_pct"], s=75, marker="s", color=PAL[3], edgecolor=INK, lw=0.9, zorder=3, label="Top 1" if combined(ax) else "Top-1 pick")
    ax.scatter(x, t["regret_best10_pct"], s=80, marker="o", color=OURS, edgecolor=INK, lw=0.9, zorder=3,
               label="Best 10" if combined(ax) else "Best of top 10")
    ax.set_xticks(x); ax.set_xticklabels(tail_labels(ax),fontsize=10,rotation=75 if combined(ax) else 0,ha="right" if combined(ax) else "center")
    ax.set_ylim(-1.5, 1.3 * t[["regret_top1_pct", "regret_best10_pct"]].to_numpy().max()); ax.set_xlim(-0.4, len(V7) - 0.6)
    style_axis(ax, None, "Median nCAC regret (%)", tick_size=10.5)
    numeric_ticks(ax, x=False, n=4)
    ax.legend(loc="lower center" if combined(ax) else "upper center",bbox_to_anchor=(.5,1.01) if combined(ax) else None,ncol=2,fontsize=11.2)


def f7_forward(ax):
    f = pd.read_csv(R("exp12_tail_forward.csv")).set_index("variant").loc[V7]
    x = np.arange(len(V7))
    v_ = 100 * f.bias_rel_viable
    ax.vlines(x, 0, v_, color=[V7_COL[v] for v in V7], lw=3.0)
    ax.scatter(x, v_, s=90, color=[V7_COL[v] for v in V7], edgecolor=INK, lw=0.9, zorder=3)
    ax.axhline(0, color=INK, lw=1.0)
    for i, v in enumerate(V7):
        ax.text(i, v_.iloc[i] + 0.35, f"ρ {f.loc[v, 'spearman_within_tail']:.2f}", ha="center", va="bottom",rotation=90 if combined(ax) else 0,
                fontsize=10,color=INK)
    ax.set_xticks(x); ax.set_xticklabels(tail_labels(ax),fontsize=10,rotation=75 if combined(ax) else 0,ha="right" if combined(ax) else "center")
    ax.set_ylim(min(0, v_.min()) - 0.5, v_.max() + 1.8); ax.set_xlim(-0.5, len(V7) - 0.5)
    style_axis(ax, None, "Tail underprediction (%)" if combined(ax) else "Held-out tail under-prediction (%)", tick_size=10.5)
    numeric_ticks(ax, x=False, n=4)


def _sel(variant):
    return pd.read_csv(R(f"exp13_conformal_selection_{variant.replace('+', '_')}.csv"))


SELECTION_ROWS = [
    ("linear+phys","conformal","hetero","Conformal, σ"),
    ("linear+phys","conformal","mean_only","Conformal, mean"),
    ("linear+phys","bayes_plugin","hetero","Plug-in, σ"),
    ("linear+phys","bayes_plugin","hetero_x0.5","Plug-in, σ/2"),
    ("linear+phys","bayes_plugin","hetero_x2","Plug-in, 2σ"),
    ("linear+phys","bayes_plugin","homo","Plug-in, common σ"),
    ("linear+phys","point","none","Point"),
    ("lgb","conformal","hetero","Conformal, C"),
]


def selection_matrix_values(metric):
    values=[]
    for variant,method,scale,_ in SELECTION_ROWS:
        d=_sel(variant)
        d=d[(d.method==method)&(d.scale==scale)&(d.q.isna()|np.isclose(d.q,.10))]
        values.append(d.groupby("price")[metric].mean().reindex([60,100,150]).to_numpy())
    matrix=np.asarray(values)
    if not np.isfinite(matrix).all():raise ValueError(f'Missing selection cells: {metric}')
    return matrix


def selection_matrix(ax, metric, row_labels):
    power=metric=="power"
    values=selection_matrix_values(metric)*(100 if power else 1)
    im=ax.imshow(values,cmap=W2B if power else W2R,vmin=0,vmax=100 if power else .10,
                 interpolation="nearest",aspect="auto")
    for (i,j),value in np.ndenumerate(values):
        label=f'{value:.1f}' if power else f'{value:.3f}'.lstrip('0')
        ax.text(j,i,label,ha="center",va="center",fontsize=11.0,color=INK)
    ax.set_xticks([0,1,2]);ax.set_xticklabels(["60","100","150"])
    ax.set_yticks(np.arange(len(SELECTION_ROWS)))
    ax.set_yticklabels([r[3] for r in SELECTION_ROWS] if row_labels or not combined(ax) else [])
    style_axis(ax,f'Carbon price ({EUR})',None,tick_size=11.0)
    ax.tick_params(axis="y",left=row_labels or not combined(ax),length=3)
    for spine in ax.spines.values():spine.set_visible(False)
    hcbar(im,ax,"Power (%)" if power else "Mean FDR",ticks=[0,50,100] if power else [0,.05,.10],
          ticklabels=["0","50","100"] if power else ["0",".05",".10"],pad=.25,fraction=.045)


def selection_fdr_matrix(ax):
    selection_matrix(ax,"fdp",True)


def selection_power_matrix(ax):
    selection_matrix(ax,"power",False)


def f7_fdr(ax):
    s = _sel("linear+phys")
    s = s[s.price == 100]
    lines = [("conformal", "hetero", OURS, "-", "o", "Conformal" if combined(ax) else "Conformal (any σ scale)"),
             ("bayes_plugin", "hetero", PAL[0], "-", "s", "Plug-in, σ"),
             ("bayes_plugin", "hetero_x0.5", PAL[3], "--", "v", "Plug-in, σ/2"),
             ("bayes_plugin", "hetero_x2", PAL[2], "--", "^", "Plug-in, 2σ")]
    for m, sc, c, ls, mk, lab in lines:
        d = s[(s.method == m) & (s.scale == sc)].groupby("q").fdp.mean()
        ax.plot(d.index, d.values, marker=mk, ms=6.5, lw=2.0, color=c, ls=ls, label=lab,
                markeredgecolor=INK, markeredgewidth=0.8)
    ax.plot([0, 0.22], [0, 0.22], color=MUTED, lw=1.2, ls=":")
    ax.set_xlim(0, 0.22); ax.set_ylim(0, None)
    style_axis(ax, "Target FDR q", f"FDR at 100 {EUR}" if combined(ax) else f"Realised FDR at 100 {EUR}")
    numeric_ticks(ax, n=4)
    ax.legend(loc="upper left", fontsize=10.6)




# =============================================================================== Fig. S3 tail ablations + selection power
def s3_all_variants(ax):
    t = _tail_cv().loc[V7_ALL]
    x = np.arange(len(V7_ALL))
    ax.bar(x - 0.27, t["P@10"], 0.27, color=PAL[3], label="P@10")
    ax.bar(x, t["P@25"], 0.27, color=PAL[0], label="P@25")
    ax.bar(x + 0.27, t["P@50"], 0.27, color=OURS, label="P@50")
    edge(ax)
    for i, v in enumerate(V7_ALL):
        ax.text(i, 0.505, f"ρ {t.loc[v, 'spearman']:.3f}", ha="center", va="bottom", fontsize=9, color=INK, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(["C","C+M","C+P","L","L+M","L+P","L+P+mono"] if combined(ax) else [V7_LAB[v].replace(chr(10), " ") for v in V7_ALL], fontsize=10.8, rotation=30, ha="right")
    ax.set_ylim(0.5, 0.85)
    style_axis(ax, None, "Median top-k precision", tick_size=10.5)
    numeric_ticks(ax, x=False, n=4)
    ax.legend(loc="upper left", ncol=3, fontsize=10)


def s3_power(ax):
    s = _sel("linear+phys")
    s = s[s.q.isna() | (s.q == 0.10)]
    lines = [("conformal", "hetero", OURS, "-", "o", "Conformal"), ("bayes_plugin", "hetero", PAL[0], "-", "s", "Plug-in, σ"),
             ("bayes_plugin", "hetero_x2", PAL[2], "--", "^", "Plug-in, 2σ"), ("bayes_plugin", "homo", PAL[5], "--", "D", "Plug-in, homosc. σ"),
             ("point", "none", PAL[7], ":", "x", "Point rule (no FDR control)")]
    for m, sc, c, ls, mk, lab in lines:
        d = s[(s.method == m) & (s.scale == sc)].groupby("price").power.mean()
        ax.plot(d.index, d.values, marker=mk, ms=6, lw=2.0, ls=ls, color=c, label=lab, markeredgecolor=INK if mk != "x" else c,
                markeredgewidth=0.8 if mk != "x" else 1.6)
    d = _sel("lgb")
    d = d[(d.method == "conformal") & (d.scale == "hetero") & (d.q == 0.10)].groupby("price").power.mean()
    ax.plot(d.index, d.values, marker="o", ms=6, lw=1.6, ls="--", color=OURS, markerfacecolor="white",
            label="Conformal, constant leaves")
    ax.set_xticks([60, 100, 150])
    style_axis(ax, f"Carbon price ({EUR})", "Power (q = 0.10)")
    numeric_ticks(ax, x=False, n=4)
    if combined(ax):
        h,l=ax.get_legend_handles_labels()
        names={"Plug-in, homosc. σ":"Plug-in common σ","Point rule (no FDR control)":"Point","Conformal, constant leaves":"Conformal const."}
        ax.legend(h,[names.get(t,t).replace("Plug-in, ","Plug-in ") for t in l],loc="upper center",bbox_to_anchor=(.5,-.25),ncol=1,fontsize=9.2)
    else:ax.legend(loc="lower right",fontsize=9.2)


def s3_point_fdr(ax):
    rows = []
    s = _sel("linear+phys")
    for (m, sc), g in s[s.q.isna() | (s.q == 0.10)].groupby(["method", "scale"]):
        rows.append(dict(rule=f"{m}|{sc}", fdr=g.groupby("price").fdp.mean().reindex([60, 100, 150]).values))
    order = [("conformal|hetero", "Conformal"), ("conformal|mean_only", "Conformal,\nmean only"),
             ("bayes_plugin|hetero", "Plug-in"), ("bayes_plugin|hetero_x0.5", "Plug-in,\nσ/2"),
             ("bayes_plugin|homo", "Plug-in,\nhomosc. σ"), ("point|none", "Point\nrule")]
    d = pd.DataFrame(rows)
    x = np.arange(len(order))
    for j, (p, c) in enumerate(zip([60, 100, 150], [PAL[0], OURS, PAL[2]])):
        vals = [d[d.rule == r].fdr.iloc[0][j] for r, _ in order]
        ax.bar(x + (j - 1) * 0.26,vals,0.26,color=c,label=str(p) if combined(ax) else f"{p} {EUR}")
    edge(ax)
    ax.axhline(0.10, color=INK, lw=1.1, ls="--")
    ax.set_xticks(x);ax.set_xticklabels(["Conformal","Conformal mean","Plug-in σ","Plug-in σ/2","Plug-in common σ","Point"] if combined(ax) else [l.replace(chr(10)," ") for _,l in order],fontsize=10.8,rotation=75 if combined(ax) else 30,ha="right")
    ax.set_ylim(0, 0.135)
    style_axis(ax, None, "Realised FDR (q = 0.10)", tick_size=10.5)
    numeric_ticks(ax, x=False, n=4)
    ax.legend(loc="lower center" if combined(ax) else "upper left",bbox_to_anchor=(.5,1.02) if combined(ax) else None,ncol=3,fontsize=10.6,columnspacing=.8,title=f"Carbon price ({EUR})" if combined(ax) else None)




# =============================================================================== Fig. S1 benchmark + ablations
def s1_bench(ax):
    e2 = pd.read_csv(R("exp2_benchmark20_stacking.csv"))
    b = e2.groupby("model")[["R2", "Spearman"]].mean().sort_values("R2", ascending=False)
    cols = [OURS if m.startswith("Stacking") else (PAL[0] if m.startswith("Pooled") else PAL[3]) for m in b.index]
    x = np.arange(len(b))
    ax.bar(x, b.R2, color=cols, width=0.7); edge(ax)
    ax.plot(x, b.Spearman, "o-", color=PAL[2], ms=5, lw=1.4, label="Spearman")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("Pooled-LightGBM(all scenarios)", "Pooled" if combined(ax) else "Pooled, 132 scen.").replace("Stacking(NNLS)", "Stacking")
                        for m in b.index], rotation=75 if combined(ax) else 45, ha="right")
    ax.set_ylim(0.6, 1.0)
    style_axis(ax, None, "R$^{2}$ (bars) / Spearman", tick_size=11)
    ax.legend(loc="upper right", fontsize=11.5)


def s1_weights(ax):
    w = None
    for f in sorted(Path(RES).glob("exp2_stack_weights_*.csv")):
        s = pd.read_csv(f, index_col=0).iloc[:, 0].rename(f.stem.replace("exp2_stack_weights_", "").split("_")[0].replace("NGCC-onshore", "NGCC"))
        w = s.to_frame() if w is None else w.join(s)
    w = w[(w > 0.01).any(axis=1)]
    w = w.loc[w.sum(axis=1).sort_values().index]
    y = np.arange(len(w))
    for i, c in enumerate(w.columns):
        ax.barh(y + (i - 0.5) * 0.38, w[c], 0.38, color=[PAL[0], PAL[2]][i], label=c)
    edge(ax)
    ax.set_yticks(y); ax.set_yticklabels(w.index)
    style_axis(ax, "NNLS stacking weight", None, tick_size=11)
    numeric_ticks(ax, y=False, n=4)
    ax.legend(loc="lower right", fontsize=11.5)


def s1_ablation(ax):
    d = pd.read_csv(R("panel54_prelim/exp1b_decision_focused.csv")).set_index("variant")
    order = ["RIV", "RIV-band", "RIV-rank", "RIV+rank blend"]
    lab = ["RIV", "Band", "Rank", "Blend"] if combined(ax) else ["RIV", "Price-band\nweights", "LambdaRank", "RIV + rank\nblend"]
    ax.bar(np.arange(4) - 0.18, d.loc[order, "spearman"], 0.36, color=PAL[0], label="Spearman")
    ax.bar(np.arange(4) + 0.18, d.loc[order, "P@25"], 0.36, color=OURS, label="P@25")
    edge(ax)
    ax.set_xticks(np.arange(4)); ax.set_xticklabels(lab, fontsize=11.5)
    ax.set_ylim(0, 1.1)
    style_axis(ax, None, "Median score" if combined(ax) else "Median per scenario", tick_size=11)
    ax.legend(loc="lower center" if combined(ax) else "upper right",bbox_to_anchor=(.5,1.02) if combined(ax) else None,ncol=2,fontsize=11.5)


def s1_nuisance(ax):
    d = pd.read_csv(R("exp11_nuisance_capacity.csv")).groupby(["estimator", "n_trees"]).PEHE.mean().unstack(0)
    cm = {"S-learner": PAL[0], "X-learner": PAL[2], "T-learner": PAL[3], "DML (R-learner)": OURS, "DR-learner": PAL[5]}
    for e, c in cm.items():
        ax.plot(d.index, d[e], marker="o", ms=6, lw=2.2, color=c, label=e.replace("DML (R-learner)", "R (DML)"))
    ax.set_xscale("log"); ax.set_xticks([100, 300, 750, 1500]); ax.set_xticklabels(["100", "300", "750", "1500"])
    ax.minorticks_off()
    style_axis(ax, "Trees per nuisance/outcome model", "PEHE (t k€$^{-1}$)")
    numeric_ticks(ax, x=False, n=4)
    ax.legend(loc="upper left", fontsize=11.2)




def compact(stem, panels, size, rectangles):
    """Explicit content footprints; no extra cell padding or equal-cell constraint."""
    items=[(letter,fn,(i,0),standalone) for i,(letter,fn,standalone) in enumerate(panels)]
    build(stem,(len(items),1),items,size,rectangles=rectangles)


def main01():
    compact("Fig01_concept_framework",[
        ("a",f1_schematic,(15,4.2)),("b",f1_pole,(4.4,4)),("c",f1_riv,(4.4,4)),
        ("d",f1_density,(4.4,4)),("e",f1_curves,(4.6,4))],(12.8,7.5),
        {"a":(.024,.545,.971,.435),"b":(.024,.025,.223,.487),"c":(.273,.025,.223,.487),
         "d":(.522,.025,.223,.487),"e":(.771,.025,.224,.487)})


def six_grid(stem, panels, size=(12.8,7.65)):
    rect={l:(.024+c*.331,.025+(1-r)*.491,.309,.459) for r,row in enumerate(["abc","def"]) for c,l in enumerate(row)}
    compact(stem,panels,size,rect)


def main02():
    six_grid("Fig02_data_landscape",[("a",f2_pole_data,(4.6,4)),("b",f2_riv_data,(4.6,4)),
        ("c",f2_share,(4.6,4)),("d",f2_invariance,(4.8,4.2)),("e",f2_heat,(4.6,4.8)),("f",f2_overlap,(4.6,4))])


def main03():
    compact("Fig03_target_objectives",[("a",f3_targets,(4.8,4.4)),("b",f3_f1,(4.6,4)),("c",s1_ablation,(6,4.6))],
        (12.8,4.7),{"a":(.024,.025,.321,.95),"b":(.371,.025,.290,.95),"c":(.687,.025,.308,.95)})


def main04():
    compact("Fig04_distribution_validation",[("a",f3_survival,(4.6,4)),("b",f3_calibration,(4.6,4)),
        ("c",f3_cure,(4.6,4)),("d",f3_breakeven,(4.6,4.6))],(12.0,8.3),
        {"a":(.024,.516,.460,.459),"b":(.510,.516,.485,.459),"c":(.024,.025,.460,.459),"d":(.510,.025,.485,.459)})


def main05():
    six_grid("Fig05_one_model_transfer",[("a",f4_budget,(4.8,4.2)),("b",f4_stageA,(4.6,4)),
        ("c",f4_tiers,(4.6,4)),("d",f4_region,(4.6,4)),("e",f4_config,(4.6,4)),("f",f4_feedback,(4.6,4))])


def main06():
    compact("Fig06_frontier_ablation",[("a",f7_ceiling,(4.8,4.2)),("b",f7_mg_rank,(5,4.2)),
        ("c",s3_all_variants,(7.4,4.6)),("d",f7_regret,(5,4)),("e",f7_forward,(5,4))],(13,8.3),
        {"a":(.024,.516,.310,.459),"b":(.024,.025,.310,.459),"c":(.360,.516,.635,.459),
         "d":(.360,.025,.304,.459),"e":(.690,.025,.305,.459)})


def main07():
    compact("Fig07_error_controlled_selection",[("a",f7_fdr,(4.8,4)),("b",selection_fdr_matrix,(6.8,5)),
        ("c",selection_power_matrix,(6.8,5))],(13.4,5.5),
        {"a":(.024,.025,.264,.950),"b":(.314,.025,.386,.950),"c":(.726,.025,.269,.950)})


def main08():
    compact("Fig08_viability_atlas",[("a",f6_curves,(4.6,4)),("b",f6_scen_cal,(4.6,4)),("c",f6_gate,(4.6,4)),
        ("d",f6_price_heat,(9.2,6.2)),("e",f6_robust,(5.2,5.6))],(13.4,12.6),
        {"a":(.024,.705,.309,.270),"b":(.355,.705,.309,.270),"c":(.686,.705,.309,.270),
         "d":(.024,.025,.653,.653),"e":(.703,.025,.292,.653)})


def main09():
    compact("Fig09_humidity_mechanism",[("a",f5_ite,(4.6,4)),("b",f5_humid_price,(5.2,4.4)),
        ("c",f5_mediation,(4.8,4.6)),("d",f5_shap,(4.8,4))],(13.2,6.6),
        {"a":(.024,.553,.310,.422),"b":(.360,.025,.337,.950),
         "c":(.024,.025,.310,.495),"d":(.723,.025,.272,.950)})


def main10():
    compact("Fig10_causal_validation",[("a",f5_ate,(4.6,4)),("b",f5_pehe,(4.6,4)),("c",s1_nuisance,(6,4.6))],
        (12.6,7.3),{"a":(.024,.516,.367,.459),"b":(.024,.025,.367,.459),"c":(.417,.025,.578,.950)})


ALL={f"Fig{i:02d}":fn for i,fn in enumerate([main01,main02,main03,main04,main05,main06,main07,main08,main09,main10],1)}
if __name__ == "__main__":
    for name in (sys.argv[1:] or list(ALL)):
        ALL[name]()
