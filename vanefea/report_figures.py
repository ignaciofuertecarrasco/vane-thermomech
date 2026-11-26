"""Explanatory figures for the technical report (docs/report).

Usage::

    python -m vanefea.report_figures --case /tmp/vanefea/baseline_L3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from vanefea.analysis import CFD_DIR, VaneCase, mesh_vane  # noqa: E402
from vanefea.cad import export  # noqa: E402
from vanefea.geometry import (CoolingLayout, camber_line, load_profile, local_thickness,  # noqa: E402
                              point_on_camber, wall_distance)
from vanefea.loads import Coolant, ExternalBC  # noqa: E402
from vanefea.plots import _surface  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "report"
BASE = CoolingLayout(n=7, ratio=0.40, f_start=0.10, f_end=0.80)


def geometry():
    prof = load_profile() * 1e3
    cam = camber_line() * 1e3
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw=dict(width_ratios=[1, 1.15, 1]))
    a = ax[0]
    a.fill(prof[:, 0], prof[:, 1], c="0.9", ec="k", lw=0.8)
    a.plot(cam[:, 0], cam[:, 1], "C0--", lw=1, label="camber line")
    for f in (0.1, 0.3, 0.5, 0.7, 0.9):
        c = point_on_camber(f) * 1e3
        r = 0.5 * local_thickness(f) * 1e3
        a.add_patch(plt.Circle(c, r, fill=False, ec="C1", lw=0.8))
        a.text(c[0] + r + 0.6, c[1] - 1, f"f = {f}", fontsize=7, color="C1")
    a.plot([], [], "o", mfc="none", mec="C1", label="largest inscribed circle")
    a.set_aspect("equal")
    a.set_title("Section, camber line and local thickness", fontsize=10)
    a.legend(frameon=False, fontsize=8, loc="lower right")
    a.set_xlabel("x [mm]")
    a.set_ylabel("y [mm]")

    f = np.linspace(0.02, 0.98, 97)
    t = np.array([local_thickness(x) for x in f]) * 1e3
    a = ax[1]
    a.plot(f, t, "k-", lw=1.5, label="local thickness t(f)")
    for ratio, c in ((0.3, "C2"), (0.4, "C0"), (0.5, "C3")):
        d = np.clip(ratio * t, 0.6, 3.0)
        a.plot(f, d, c=c, lw=1.2, label=f"passage diameter, ratio {ratio}")
        wall = 0.5 * (t - d)
        a.plot(f, wall, c=c, lw=0.8, ls=":")
    a.axhline(0.6, c="gray", ls="--", lw=0.8)
    a.text(0.02, 0.68, "minimum wall 0.6 mm / smallest drill 0.6 mm", fontsize=7, color="gray")
    a.set_xlabel("position along the camber line f (0 = LE, 1 = TE)")
    a.set_ylabel("[mm]")
    a.set_title("Thickness, passage diameter (solid) and remaining wall (dotted)", fontsize=10)
    a.legend(frameon=False, fontsize=7)
    a.grid(alpha=0.3)

    a = ax[2]
    a.fill(prof[:, 0], prof[:, 1], c="0.9", ec="k", lw=0.8)
    for c, d in zip(BASE.centres() * 1e3, BASE.diameters() * 1e3):
        a.add_patch(plt.Circle(c, d / 2, fc="w", ec="C3"))
        wd = wall_distance(np.array([c / 1e3]))[0] * 1e3 - d / 2
        a.text(c[0] + d / 2 + 0.5, c[1] - 1.6, f"{d:.1f} mm\nwall {wd:.1f}", fontsize=6.5)
    a.set_aspect("equal")
    a.set_title("Baseline layout: 7 passages, ratio 0.40, f = 0.10-0.80", fontsize=10)
    a.set_xlabel("x [mm]")
    fig.tight_layout()
    fig.savefig(OUT / "geometry.png", dpi=150)
    plt.close(fig)


def mesh_pictures():
    from vanefea.mesh_study import LEVELS
    step = export(BASE, Path("/tmp/vanefea/report/vane.step"))
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for k, (name, zoom) in enumerate((("L1", None), ("L3", None), ("L3", (24, 36, 18, 33)))):
        h, hh, hmin, nc = LEVELS[name]
        m = mesh_vane(step, VaneCase(layout=BASE, h_size=h, h_hole=hh, h_min=hmin, n_curv=nc))
        tris = m.surfaces["hub"][:, :3]
        xy = m.nodes[:, :2] * 1e3
        ax[k].triplot(xy[:, 0], xy[:, 1], tris, lw=0.25, c="k")
        ax[k].set_aspect("equal")
        ttl = f"{name}: {m.n_nodes:,} nodes, {len(m.tets):,} tetrahedra (hub face shown)"
        if zoom:
            ax[k].set_xlim(*zoom[:2])
            ax[k].set_ylim(*zoom[2:])
            ttl = f"{name}, zoom on the rear passages"
        ax[k].set_title(ttl, fontsize=9)
        ax[k].set_xlabel("x [mm]")
    fig.tight_layout()
    fig.savefig(OUT / "mesh.png", dpi=170)
    plt.close(fig)


def bc_mapping(case_dir: Path):
    ext = ExternalBC.from_dir(CFD_DIR)
    f = np.load(case_dir / "fields.npz")
    xyz, tris = f["xyz"].astype(float), f["outer_tris"]
    p, T_aw, h = ext.at(xyz, 0.05)
    h_eff = 1 / (1 / h + 0.3e-3 / 1.0)
    fig = plt.figure(figsize=(15, 5.3))
    a = fig.add_subplot(1, 3, 1)
    tab = np.genfromtxt(CFD_DIR / "htc.csv", delimiter=",", names=True)
    s = tab["s"] * 1e3
    a.plot(s, tab["h_Tw1100"] / 1e3, "C0", lw=1.4, label="h from CFD")
    a.plot(s, 1 / (1 / tab["h_Tw1100"] + 3e-4) / 1e3, "C0--", lw=1.2, label="h with 0.3 mm coating")
    a.set_xlabel("s [mm] (PS < 0 < SS)")
    a.set_ylabel("h [kW/(m² K)]")
    a2 = a.twinx()
    a2.plot(s, tab["T_aw"], "C3", lw=1.2, label="T_aw")
    a2.set_ylabel("T_aw [K]", color="C3")
    h1, l1 = a.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
    a.set_title("Hot-gas table from the 2D CFD", fontsize=10)
    a.grid(alpha=0.3)
    ax = fig.add_subplot(1, 3, 2, projection="3d")
    _surface(ax, xyz, tris, h_eff / 1e3, "viridis", "effective h [kW/(m² K)]")
    ax.view_init(elev=24, azim=-62)
    ax.set_title("Mapped onto the 3D faces (with coating)", fontsize=10)
    ax = fig.add_subplot(1, 3, 3, projection="3d")
    _surface(ax, xyz, tris, p / 1e5, "cividis", "static pressure [bar]")
    ax.view_init(elev=24, azim=-62)
    ax.set_title("Gas pressure load", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "bc_mapping.png", dpi=150)
    plt.close(fig)


def modes(case_dir: Path):
    f = np.load(case_dir / "fields.npz")
    if "modes" not in f or len(f["modes"]) == 0:
        print("no mode shapes stored in", case_dir)
        return
    xyz, tris = f["xyz"].astype(float), f["outer_tris"]
    fig = plt.figure(figsize=(15, 5))
    for k, (shape, freq) in enumerate(zip(f["modes"], f["mode_freqs"])):
        u = shape.astype(float)
        u /= np.abs(u).max()
        mag = np.linalg.norm(u, axis=1)
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        _surface(ax, xyz + 0.006 * u, tris, mag, "plasma", "relative displacement")
        ax.view_init(elev=18, azim=-62)
        ax.set_title(f"Mode {k + 1}: {freq:.0f} Hz", fontsize=10)
    fig.suptitle("Natural modes of the hot vane, hub clamped (displacements exaggerated)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "modes.png", dpi=150)
    plt.close(fig)


def design_plan(name="sweep_v1"):
    tab = np.genfromtxt(ROOT / "results" / "dataset" / name / "scalars.csv", delimiter=",", names=True)
    keys = ["n", "ratio", "f_start", "f_end", "coolant"]
    fig, ax = plt.subplots(len(keys), len(keys), figsize=(11, 11))
    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            a = ax[i, j]
            if i == j:
                a.hist(tab[ki], bins=12, color="C0", alpha=0.7)
            else:
                a.scatter(tab[kj], tab[ki], s=7, c=tab["T_mean"], cmap="inferno")
            if i == len(keys) - 1:
                a.set_xlabel(kj)
            if j == 0:
                a.set_ylabel(ki)
            a.tick_params(labelsize=6)
    fig.suptitle("Design plan: 110 Sobol samples (colour: mean metal temperature)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "design_plan.png", dpi=130)
    plt.close(fig)


def coolant_model():
    c = Coolant(m_total=0.022, T_in=800.0)
    d = BASE.diameters()
    m = c.split(d)
    info = [c.passage(mi, di, 800.0) for mi, di in zip(m, d)]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    idx = np.arange(1, len(d) + 1)
    ax[0].bar(idx - 0.2, d * 1e3, 0.4, label="diameter [mm]")
    ax[0].bar(idx + 0.2, m / m.sum() * 100 / 10, 0.4, label="share of coolant [% / 10]")
    ax[0].set_xlabel("passage (1 = leading edge)")
    ax[0].legend(frameon=False, fontsize=8)
    ax[0].set_title("Equal pressure drop: big passages take most of the flow", fontsize=10)
    ax[1].bar(idx - 0.2, [2 * i["h"] / 1e3 for i in info], 0.4, label="h (ribbed) [kW/(m² K)]")
    ax[1].bar(idx + 0.2, [i["Mach"] * 10 for i in info], 0.4, label="inlet Mach × 10")
    ax[1].set_xlabel("passage (1 = leading edge)")
    ax[1].legend(frameon=False, fontsize=8)
    ax[1].set_title("Heat transfer coefficient and Mach number per passage", fontsize=10)
    for a in ax:
        a.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "coolant_model.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--case", default="/tmp/vanefea/baseline_L3")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    geometry()
    coolant_model()
    design_plan()
    bc_mapping(Path(a.case))
    modes(Path(a.case))
    mesh_pictures()
    print(json.dumps(sorted(p.name for p in OUT.glob("*.png"))))
