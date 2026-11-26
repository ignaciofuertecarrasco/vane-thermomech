"""Verification of the CalculiX set-up against closed-form solutions.

Before the vane is analysed, the same element type (C3D10), the same deck writer and the same load
types (face films, prescribed nodal temperatures for the thermal stress, eigenfrequencies) are run
on two problems with exact answers:

1. **Thick-walled cylinder with convection on both faces** (quarter model, symmetry planes, free
   axial expansion). Temperature from the thermal-resistance network and the logarithmic profile;
   radial, hoop and axial thermal stresses from the Timoshenko-Goodier solution for a long hollow
   cylinder with free ends. This checks ``*FILM``, the conduction, ``*TEMPERATURE`` and the
   thermal-strain bookkeeping (reference temperature, secant expansion).
2. **Cantilever beam, natural frequencies** against Euler-Bernoulli theory (first two bending
   modes). This checks the mass matrix, the eigenvalue extraction and the frequency parser.

Each case is run on three meshes so that the convergence of the error is visible.

Usage::

    python -m vanefea.verification            # writes results/verification/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gmsh
import numpy as np

from vanefea import ccx
from vanefea.materials import STEEL_CONST as MAT
from vanefea.mesh import volume_mesh

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "verification"

# cylinder problem
A, B, L = 5e-3, 10e-3, 40e-3            # inner/outer radius, length (half of a 2L cylinder) [m]
H_IN, T_IN = 2000.0, 500.0              # inner film coefficient [W/m2K], coolant temperature [K]
H_OUT, T_OUT = 1000.0, 1200.0           # outer film coefficient, gas temperature
T_REF = 293.15


def cylinder_exact(r):
    k = MAT.k[0]
    R = 1 / (2 * np.pi * A * H_IN) + np.log(B / A) / (2 * np.pi * k) + 1 / (2 * np.pi * B * H_OUT)
    qp = (T_OUT - T_IN) / R                                 # heat per unit length [W/m]
    Ta, Tb = T_IN + qp / (2 * np.pi * A * H_IN), T_OUT - qp / (2 * np.pi * B * H_OUT)
    T = lambda rr: Ta + (Tb - Ta) * np.log(rr / A) / np.log(B / A)   # noqa: E731
    E, nu, al = MAT.E[0], MAT.nu, MAT.alpha[0]
    c = E * al / (1 - nu)
    rr = np.linspace(A, B, 4001)
    Tr = (T(rr) - T_REF) * rr
    I_ab = np.trapezoid(Tr, rr)

    def I_ar(x):
        return np.interp(x, rr, np.r_[0, np.cumsum(0.5 * (Tr[1:] + Tr[:-1]) * np.diff(rr))])

    dT = T(r) - T_REF
    sr = c / r ** 2 * ((r ** 2 - A ** 2) / (B ** 2 - A ** 2) * I_ab - I_ar(r))
    st = c / r ** 2 * ((r ** 2 + A ** 2) / (B ** 2 - A ** 2) * I_ab + I_ar(r) - dT * r ** 2)
    sz = c * (2 / (B ** 2 - A ** 2) * I_ab - dT)
    return dict(T=T(r), s_r=sr, s_t=st, s_z=sz, q_per_length=qp)


def _cylinder_model(h):
    gmsh.model.add("cylinder")
    occ = gmsh.model.occ
    outer = occ.addCylinder(0, 0, 0, 0, 0, L, B)
    inner = occ.addCylinder(0, 0, 0, 0, 0, L, A)
    ring, _ = occ.cut([(3, outer)], [(3, inner)])
    box = occ.addBox(0, 0, 0, 2 * B, 2 * B, L)
    q, _ = occ.intersect(ring, [(3, box)])
    occ.synchronize()
    groups = {"inner": [], "outer": [], "x0": [], "y0": [], "z0": [], "zL": []}
    for _, s in gmsh.model.getBoundary(q, oriented=False):
        x0, y0, z0, x1, y1, z1 = gmsh.model.getBoundingBox(2, s)
        cx, cy, cz = gmsh.model.occ.getCenterOfMass(2, s)
        if x1 - x0 < 1e-6:
            groups["x0"].append(s)
        elif y1 - y0 < 1e-6:
            groups["y0"].append(s)
        elif z1 - z0 < 1e-6:
            groups["z0" if cz < L / 2 else "zL"].append(s)
        else:
            groups["inner" if np.hypot(cx, cy) < 0.5 * (A + B) else "outer"].append(s)
    for name, ents in groups.items():
        assert ents, f"no surface found for group {name}"
        gmsh.model.setPhysicalName(2, gmsh.model.addPhysicalGroup(2, ents), name)
    gmsh.model.addPhysicalGroup(3, [q[0][1]])
    gmsh.option.setNumber("Mesh.MeshSizeMax", h)
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.5 * h)


def run_cylinder(work: Path, h: float) -> dict:
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        _cylinder_model(h)
        m = volume_mesh()
    finally:
        gmsh.finalize()
    work.mkdir(parents=True, exist_ok=True)
    idx = ccx.face_index(m.tets)
    f_in, f_out = (ccx.faces_of_triangles(m.surfaces[s], idx) for s in ("inner", "outer"))
    # thermal analysis
    d = ccx.Deck(m.nodes, m.tets).mesh().material(MAT, mechanical=False).initial_temperature(800.0)
    d.raw("*STEP", "*HEAT TRANSFER, STEADY STATE")
    d.film(f_in, np.full(len(f_in), T_IN), np.full(len(f_in), H_IN))
    d.film(f_out, np.full(len(f_out), T_OUT), np.full(len(f_out), H_OUT))
    d.raw("*NODE FILE", "NT", "*END STEP").write(work / "thermal.inp")
    T = ccx.last_block(ccx.read_frd(ccx.run(work, "thermal")), "NDTEMP").full(m.n_nodes)[:, 0]
    # thermal stress
    d = ccx.Deck(m.nodes, m.tets).mesh().material(MAT, thermal=False)
    for name, comp in (("x0", 1), ("y0", 2), ("z0", 3)):
        d.nset(name.upper(), m.surface_nodes(name)).boundary(name.upper(), comp)
    d.initial_temperature(T_REF).raw("*STEP", "*STATIC").temperatures(T)
    d.raw("*NODE FILE", "U", "*EL FILE", "S", "*END STEP").write(work / "stress.inp")
    S = ccx.last_block(ccx.read_frd(ccx.run(work, "stress")), "STRESS").full(m.n_nodes)

    # compare on the symmetry plane z = 0, away from the free end
    x, y, z = m.nodes.T
    r = np.hypot(x, y)
    th = np.arctan2(y, x)
    # compare at element vertices on the symmetry plane: with straight-sided elements the mid-edge nodes of
    # curved faces lie on the chord, slightly off the true surface, where the exact solution does not apply
    vertex = np.zeros(m.n_nodes, bool)
    vertex[np.unique(m.tets[:, :4])] = True
    sel = (z < 1e-9) & vertex
    ex = cylinder_exact(r[sel])
    s_t = S[sel, 0] * np.sin(th[sel]) ** 2 + S[sel, 1] * np.cos(th[sel]) ** 2 - 2 * S[sel, 3] * np.sin(th[sel]) * np.cos(th[sel])
    s_r = S[sel, 0] * np.cos(th[sel]) ** 2 + S[sel, 1] * np.sin(th[sel]) ** 2 + 2 * S[sel, 3] * np.sin(th[sel]) * np.cos(th[sel])
    s_scale = np.abs(ex["s_t"]).max()
    return dict(h=h, n_nodes=m.n_nodes, n_elements=len(m.tets),
                T_max_error_K=float(np.abs(T[sel] - ex["T"]).max()),
                T_range_K=float(np.ptp(ex["T"])),
                hoop_max_error_rel=float(np.abs(s_t - ex["s_t"]).max() / s_scale),
                radial_max_error_rel=float(np.abs(s_r - ex["s_r"]).max() / s_scale),
                axial_max_error_rel=float(np.abs(S[sel, 2] - ex["s_z"]).max() / s_scale),
                hoop_inner_MPa=dict(fe=float(s_t[np.argmin(r[sel])] / 1e6), exact=float(ex["s_t"][np.argmin(r[sel])] / 1e6)),
                hoop_outer_MPa=dict(fe=float(s_t[np.argmax(r[sel])] / 1e6), exact=float(ex["s_t"][np.argmax(r[sel])] / 1e6)),
                profile=dict(r=r[sel].tolist(), T=T[sel].tolist(), s_t=s_t.tolist(), s_z=S[sel, 2].tolist()))


# cantilever beam --------------------------------------------------------------------------------
LB, HB = 0.2, 0.01        # length and square side [m]
BETA_L = (1.875104, 4.694091)


def beam_exact(timoshenko: bool = True):
    """Cantilever bending frequencies [Hz]: Euler-Bernoulli, optionally with the first-order Timoshenko
    correction for shear deformation and rotary inertia,
    f_T = f_EB [1 - (beta_n r)^2 (1 + E / (kappa G)) / 2],  r^2 = I / A,  kappa = 5/6.
    For this slender beam (L/h = 20) the correction is -0.2 % on mode 1 and -0.9 % on mode 2."""
    E, rho, nu = MAT.E[0], MAT.rho, MAT.nu
    inertia, Ar = HB ** 4 / 12, HB ** 2
    f = np.array([b ** 2 / (2 * np.pi * LB ** 2) * np.sqrt(E * inertia / (rho * Ar)) for b in BETA_L])
    if timoshenko:
        G, kappa = E / (2 * (1 + nu)), 5 / 6
        r2 = inertia / Ar
        f = f * np.array([1 - 0.5 * (b / LB) ** 2 * r2 * (1 + E / (kappa * G)) for b in BETA_L])
    return f


def run_beam(work: Path, h: float) -> dict:
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add("beam")
        v = gmsh.model.occ.addBox(0, 0, 0, LB, HB, HB)
        gmsh.model.occ.synchronize()
        root = [s for _, s in gmsh.model.getBoundary([(3, v)], oriented=False)
                if gmsh.model.getBoundingBox(2, s)[3] < 1e-6]
        gmsh.model.setPhysicalName(2, gmsh.model.addPhysicalGroup(2, root), "root")
        gmsh.model.addPhysicalGroup(3, [v])
        gmsh.option.setNumber("Mesh.MeshSizeMax", h)
        m = volume_mesh()
    finally:
        gmsh.finalize()
    work.mkdir(parents=True, exist_ok=True)
    d = ccx.Deck(m.nodes, m.tets).mesh().material(MAT, thermal=False)
    d.nset("ROOT", m.surface_nodes("root")).boundary("ROOT", 1, 3)
    d.raw("*STEP", "*FREQUENCY", "6", "*END STEP").write(work / "modal.inp")
    ccx.run(work, "modal")
    f = ccx.read_eigenfrequencies(work / "modal.dat")
    ex = beam_exact()
    # modes come in pairs (square section): bending 1 = modes 1-2, bending 2 = modes 3-4
    fe = np.array([f[:2].mean(), f[2:4].mean()])
    eb = beam_exact(timoshenko=False)
    return dict(h=h, n_nodes=m.n_nodes, f_fe_Hz=fe.tolist(), f_timoshenko_Hz=ex.tolist(), f_euler_bernoulli_Hz=eb.tolist(),
                rel_error=((fe - ex) / ex).tolist(), rel_error_vs_euler_bernoulli=((fe - eb) / eb).tolist(),
                all_modes_Hz=f.tolist())


def plot(cyl, beam, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(14, 5.0))
    best = cyl[-1]["profile"]
    r = np.array(best["r"])
    o = np.argsort(r)
    rr = np.linspace(A, B, 200)
    ex = cylinder_exact(rr)
    ax[0].plot(rr * 1e3, ex["T"], "k-", lw=1, label="exact")
    ax[0].plot(r[o] * 1e3, np.array(best["T"])[o], "C3.", ms=3, label="CalculiX C3D10")
    ax[0].set_xlabel("r [mm]")
    ax[0].set_ylabel("T [K]")
    ax[0].set_title("Cylinder, two-sided convection")
    ax[1].plot(rr * 1e3, ex["s_t"] / 1e6, "k-", lw=1, label="exact hoop")
    ax[1].plot(r[o] * 1e3, np.array(best["s_t"])[o] / 1e6, "C0.", ms=3, label="FE hoop")
    ax[1].plot(rr * 1e3, ex["s_z"] / 1e6, "k--", lw=1, label="exact axial")
    ax[1].plot(r[o] * 1e3, np.array(best["s_z"])[o] / 1e6, "C2.", ms=3, label="FE axial")
    ax[1].set_xlabel("r [mm]")
    ax[1].set_ylabel("thermal stress [MPa]")
    ax[1].set_title("Thermal stress, free ends")
    n = [c["n_nodes"] for c in cyl]
    ax[2].loglog(n, [c["hoop_max_error_rel"] for c in cyl], "C0o-", label="cylinder, hoop stress")
    nb = [b["n_nodes"] for b in beam]
    ax[2].loglog(nb, [abs(b["rel_error"][0]) for b in beam], "C1s-", label="beam, 1st bending frequency (vs Timoshenko)")
    ax[2].loglog(nb, [abs(b["rel_error"][1]) for b in beam], "C1^--", label="beam, 2nd bending frequency (vs Timoshenko)")
    ax[2].set_xlabel("nodes")
    ax[2].set_ylabel("relative error")
    ax[2].set_title("Mesh convergence")
    for a in ax:
        a.grid(alpha=0.3, which="both")
    ax[0].legend(frameon=False, fontsize=8)
    ax[1].legend(frameon=False, fontsize=8)
    ax[2].legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1)
    fig.tight_layout()
    fig.savefig(png, dpi=150)
    plt.close(fig)


def main(work_root: Path):
    OUT.mkdir(parents=True, exist_ok=True)
    cyl = [run_cylinder(work_root / f"cyl_{h * 1e3:.2f}mm", h) for h in (2.0e-3, 1.0e-3, 0.5e-3)]
    beam = [run_beam(work_root / f"beam_{h * 1e3:.1f}mm", h) for h in (5e-3, 2.5e-3, 1.25e-3)]
    plot(cyl, beam, OUT / "verification.png")
    summary = dict(cylinder=[{k: v for k, v in c.items() if k != "profile"} for c in cyl], beam=beam,
                   exact=dict(beam_timoshenko_Hz=beam_exact().tolist(), beam_euler_bernoulli_Hz=beam_exact(False).tolist(), cylinder_q_W_per_m=cylinder_exact(np.array([A]))["q_per_length"]))
    (OUT / "verification.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--work", default="/tmp/vanefea/verification")
    a = ap.parse_args()
    print(json.dumps(main(Path(a.work)), indent=2))
