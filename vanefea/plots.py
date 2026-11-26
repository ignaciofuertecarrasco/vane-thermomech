"""Figures of the vane analysis: fields, coolant, mesh study, design sweep, modal interference.

Usage::

    python -m vanefea.plots fields --case /tmp/vanefea/baseline --out results/figures
    python -m vanefea.plots study --out results/figures
    python -m vanefea.plots sweep --name sweep_v1 --out results/figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from vanefea.geometry import load_profile  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _surface(ax, xyz, tris, values, cmap, label, vmin=None, vmax=None):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    face_val = values[tris].mean(1)
    pc = Poly3DCollection(xyz[tris] * 1e3, linewidths=0)
    norm = plt.Normalize(vmin if vmin is not None else face_val.min(), vmax if vmax is not None else face_val.max())
    pc.set_facecolor(plt.get_cmap(cmap)(norm(face_val)))
    ax.add_collection3d(pc)
    lo, hi = xyz.min(0) * 1e3, xyz.max(0) * 1e3
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi - lo)
    ax.view_init(elev=22, azim=-128)
    ax.set_xlabel("x [mm]", fontsize=8)
    ax.set_ylabel("y [mm]", fontsize=8)
    ax.set_zlabel("z (span) [mm]", fontsize=8)
    ax.tick_params(labelsize=7)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.08, label=label)


def _section(ax, xy, val, prof, hole_xy, hole_d, cmap, label):
    """Filled contours over the section, masking triangles outside the profile or inside a passage."""
    import matplotlib.tri as mtri
    from matplotlib.path import Path as MPath
    tri = mtri.Triangulation(xy[:, 0], xy[:, 1])
    cen = xy[tri.triangles].mean(1)
    outside = ~MPath(prof).contains_points(cen)
    for c, d in zip(hole_xy, hole_d):
        outside |= np.linalg.norm(cen - c, axis=1) < 0.5 * d
    tri.set_mask(outside)
    cs = ax.tricontourf(tri, val, levels=24, cmap=cmap)
    ax.plot(*np.vstack([prof, prof[:1]]).T, "k-", lw=0.6)
    for c, d in zip(hole_xy, hole_d):
        ax.add_patch(plt.Circle(c, 0.5 * d, fc="w", ec="k", lw=0.5))
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    plt.colorbar(cs, ax=ax, shrink=0.8, label=label)


def fields(case_dir: Path, out: Path):
    f = np.load(case_dir / "fields.npz")
    q = json.loads((case_dir / "qois.json").read_text())
    xyz, tris = f["xyz"].astype(float), f["outer_tris"]
    fig = plt.figure(figsize=(15, 5.2))
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    _surface(ax1, xyz, tris, f["T"], "inferno", "metal temperature [K]")
    ax1.view_init(elev=24, azim=-62)
    ax1.set_title("Metal temperature on the hot-gas surface", fontsize=10)
    prof = load_profile() * 1e3
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)
    if "mid_xy" in f:
        _section(ax2, f["mid_xy"] * 1e3, f["mid_T"], prof, f["hole_xy"] * 1e3, f["hole_d"] * 1e3, "inferno", "T [K]")
        vm_mid = np.minimum(f["mid_vm"] / 1e6, np.percentile(f["mid_vm"] / 1e6, 99.5))
        _section(ax3, f["mid_xy"] * 1e3, vm_mid, prof, f["hole_xy"] * 1e3, f["hole_d"] * 1e3, "viridis",
                 "von Mises [MPa]")
    ax2.set_title("Mid-span section: temperature", fontsize=10)
    ax3.set_title("Mid-span section: von Mises stress", fontsize=10)
    qq = q["qois"]
    fig.suptitle(f"T_max {qq['T_max']:.0f} K (trailing edge), T_mean {qq['T_mean']:.0f} K, "
                 f"von Mises p99 {qq['vm_p99_core'] / 1e6:.0f} MPa, f1 {qq['f1']:.0f} Hz, "
                 f"coolant {q['case']['coolant_fraction'] * 100:.1f} % of the main flow", fontsize=10)
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "vane_fields.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for i, tc in enumerate(f["T_c"]):
        ax.plot(f["z_bins"] * 1e3, tc, label=f"passage {i + 1} (d = {qq['diameters'][i] * 1e3:.2f} mm)")
    ax.set_xlabel("span position z [mm] (coolant enters at the hub)")
    ax.set_ylabel("coolant bulk temperature [K]")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "coolant.png", dpi=150)
    plt.close(fig)


def study(out: Path):
    d = json.loads((ROOT / "results" / "mesh_study" / "mesh_study.json").read_text())
    rows = d["results"]
    names = list(rows)
    n = np.array([rows[k]["n_nodes"] for k in names]) / 1e3
    show = [("T_max", "peak temperature"), ("T_mean", "mean temperature"), ("Q_gas_W", "heat load"),
            ("vm_p99_core", "von Mises, 99th percentile"), ("f1", "1st natural frequency"),
            ("f3", "3rd natural frequency")]
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for k, label in show:
        v = np.array([rows[m][k] for m in names])
        ax.plot(n, 100 * (v / v[-1] - 1), "o-", label=label)
    ax.axhline(0, c="k", lw=0.6)
    ax.set_xlabel("mesh size [thousand nodes]  (L1 = design-sweep mesh)")
    ax.set_ylabel("difference from the finest mesh [%]")
    ax.set_xticks(n, [f"{x:.0f} ({name})" for x, name in zip(n, names)])
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.24))
    ax.set_title("Mesh convergence of the vane analysis (quadratic tetrahedra)", fontsize=10)
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "mesh_study.png", dpi=150)
    plt.close(fig)


def sweep(name: str, out: Path, T_limit: float | None = None):
    d = ROOT / "results" / "dataset" / name
    tab = np.genfromtxt(d / "scalars.csv", delimiter=",", names=True)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    s = ax[0].scatter(tab["coolant"] * 100, tab["T_max"], c=tab["f_end"], cmap="viridis", s=18)
    plt.colorbar(s, ax=ax[0], label="last passage position f_end")
    ax[0].set_xlabel("coolant flow [% of main flow]")
    ax[0].set_ylabel("peak metal temperature T_max [K]")
    ax[0].set_title("Peak temperature is set by the trailing edge", fontsize=10)
    s = ax[1].scatter(tab["coolant"] * 100, tab["T_mean"], c=tab["vm_p99_core"] / 1e6, cmap="plasma", s=18)
    plt.colorbar(s, ax=ax[1], label="von Mises p99 [MPa]")
    ax[1].set_xlabel("coolant flow [% of main flow]")
    ax[1].set_ylabel("mean metal temperature [K]")
    ax[1].set_title("Bulk temperature vs stress", fontsize=10)
    # interference (Campbell-type) diagram: vane modes vs downstream rotor blade passing
    krpm = np.linspace(0, 15, 50)
    for nb in (48, 64, 80):
        ax[2].plot(krpm, nb * krpm * 1e3 / 60, "k-", lw=0.6)
        ax[2].text(krpm[-1] + 0.2, nb * krpm[-1] * 1e3 / 60, f"{nb} blades", fontsize=7, va="center", ha="left")
    for k, c in (("f1", "C0"), ("f2", "C1"), ("f3", "C2")):
        lo, hi = np.percentile(tab[k], [2, 98])
        ax[2].axhspan(lo, hi, color=c, alpha=0.25, label=f"{k} range over the sweep")
    ax[2].set_xlim(0, 18.5)
    ax[2].set_ylim(0, 22000)
    ax[2].set_xlabel("rotor speed [1000 rpm] (illustrative)")
    ax[2].set_ylabel("frequency [Hz]")
    ax[2].set_title("Interference diagram: vane modes vs rotor passing", fontsize=10)
    ax[2].legend(frameon=False, fontsize=7, loc="upper left")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "sweep.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("what", choices=["fields", "study", "sweep"])
    ap.add_argument("--case", default="/tmp/vanefea/baseline")
    ap.add_argument("--name", default="sweep_v1")
    ap.add_argument("--out", default=str(ROOT / "results" / "figures"))
    a = ap.parse_args()
    if a.what == "fields":
        fields(Path(a.case), Path(a.out))
    elif a.what == "study":
        study(Path(a.out))
    else:
        sweep(a.name, Path(a.out))
