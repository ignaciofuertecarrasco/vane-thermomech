"""Thermo-mechanical and modal analysis of the cooled vane: CAD -> mesh -> CalculiX.

Workflow of ``run_vane``:

1. **CAD**: CadQuery solid of the vane with its cooling passages, exported to STEP.
2. **Mesh**: the STEP file is imported in gmsh and meshed with quadratic tetrahedra (C3D10), refined
   around the passages and by curvature (leading and trailing edges). Surfaces are grouped into
   ``outer`` (hot gas), ``hole<i>`` (passage i), ``hub`` and ``tip``.
3. **Thermal** (steady conduction, temperature-dependent conductivity): film conditions on the outer
   surface from the CFD (h, T_aw) and on each passage from the coolant model. The coolant
   temperature rise along each passage depends on the heat it picks up, so the conduction solve and
   the 1D coolant march are iterated (Picard) until the coolant temperatures change by < 0.5 K.
   End walls are adiabatic. Two standard features of a convection-cooled vane enter as engineering
   models rather than geometry: a ceramic thermal-barrier coating (TBC), as a thermal resistance
   t/k in series with the gas-side film (h_eff = 1 / (1/h + t/k)), and rib turbulators in the
   passages, as a multiplier of the smooth-duct Nusselt number (typically 2-3 for ribbed channels,
   Han, Dutta and Ekkad, Gas Turbine Heat Transfer and Cooling Technology, 2012).
4. **Static**: thermal strains of the computed temperature field (stress-free at 293 K) plus the
   hot-gas pressure from the CFD on the outer surface and the coolant pressure in the passages.
   Supports: the vane is cantilevered from the hub (hub face fixed); the tip is free (default) or,
   with ``tip_support="radial"``, held in-plane but free to grow radially.
5. **Modal**: natural frequencies of the loaded vane (perturbation step about the static state,
   so stress stiffening and the hot Young's modulus are included).

Results: scalar quantities of interest (QoIs) in ``qois.json`` and the surface fields in
``fields.npz`` (node coordinates, temperature, von Mises stress), which are the inputs of the
design-sweep data set and of the figures.

Usage::

    python -m vanefea.analysis --work /tmp/vane_baseline
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import gmsh
import numpy as np

from vanefea import ccx
from vanefea.cad import SPAN, export
from vanefea.geometry import CoolingLayout
from vanefea.loads import Coolant, ExternalBC
from vanefea.materials import NI_SUPERALLOY as MAT
from vanefea.mesh import triangle_areas_centroids, tet_volumes, volume_mesh

ROOT = Path(__file__).resolve().parents[1]
CFD_DIR = Path(__file__).resolve().parent / "data" / "aerothermal" / "engine"


@dataclass
class VaneCase:
    layout: CoolingLayout = field(default_factory=CoolingLayout)
    coolant_fraction: float = 0.03     # coolant mass flow / main-stream mass flow through one passage
    T_coolant_in: float = 800.0        # [K]
    p_coolant: float = 10.5e5          # [Pa], above the inlet total pressure of the hot gas
    radial_peak: float = 0.0           # mid-span excess of the gas temperature [K]
    span: float = SPAN
    h_size: float = 1.6e-3             # global element size [m]
    h_hole: float = 0.45e-3            # element size on the passage walls [m]
    h_min: float = 0.25e-3             # smallest element (trailing edge) [m]
    n_curv: int = 24                   # elements per 2*pi of surface curvature (LE/TE resolution)
    n_modes: int = 6
    n_bins: int = 20                   # coolant march: bins along the span
    tbc_thickness: float = 0.3e-3      # thermal-barrier coating [m] (0 = uncoated)
    tbc_conductivity: float = 1.0      # [W/(m K)], yttria-stabilised zirconia
    rib_factor: float = 2.0            # Nusselt-number enhancement of the ribbed passages (1 = smooth)
    tip_support: str = "free"          # "free" (cantilever) or "radial" (in-plane fixed, radially free)
    cfd_dir: str = str(CFD_DIR)

    def to_dict(self):
        d = asdict(self)
        d["layout"] = self.layout.to_dict()
        return d


# meshing ------------------------------------------------------------------------------------------
def mesh_vane(step: Path, case: VaneCase):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        # CadQuery labels the STEP file in millimetres while the model is built in metres: keep the
        # numbers as written (no unit conversion) and check the span
        gmsh.option.setString("Geometry.OCCTargetUnit", "MM")
        vols = gmsh.model.occ.importShapes(str(step))
        gmsh.model.occ.synchronize()
        zmax = gmsh.model.getBoundingBox(-1, -1)[5]
        if abs(zmax - case.span) > 1e-6:
            raise RuntimeError(f"unexpected CAD units: span {zmax} instead of {case.span}")
        centres = case.layout.centres()
        groups = {"outer": [], "hub": [], "tip": [], **{f"hole{i}": [] for i in range(len(centres))}}
        for _, s in gmsh.model.getBoundary(vols, oriented=False):
            z0, z1 = gmsh.model.getBoundingBox(2, s)[2], gmsh.model.getBoundingBox(2, s)[5]
            if z1 - z0 < 1e-6:
                groups["hub" if z0 < 0.5 * case.span else "tip"].append(s)
            elif gmsh.model.getType(2, s) == "Cylinder":
                c = np.array(gmsh.model.occ.getCenterOfMass(2, s)[:2])
                groups[f"hole{int(np.argmin(np.linalg.norm(centres - c, axis=1)))}"].append(s)
            else:
                groups["outer"].append(s)
        for name, ents in groups.items():
            if not ents:
                raise RuntimeError(f"surface group {name} is empty")
            gmsh.model.setPhysicalName(2, gmsh.model.addPhysicalGroup(2, ents), name)
        gmsh.model.addPhysicalGroup(3, [v for _, v in vols])
        # sizes: global, curvature-based, and refined near the passages
        hole_surfs = [s for k, v in groups.items() if k.startswith("hole") for s in v]
        fd = gmsh.model.mesh.field
        dist = fd.add("Distance")
        fd.setNumbers(dist, "SurfacesList", hole_surfs)
        thr = fd.add("Threshold")
        fd.setNumber(thr, "InField", dist)
        fd.setNumber(thr, "SizeMin", min(case.h_hole, 0.35 * float(case.layout.diameters().min())))
        fd.setNumber(thr, "SizeMax", case.h_size)
        fd.setNumber(thr, "DistMin", 0.3e-3)
        fd.setNumber(thr, "DistMax", 3e-3)
        fd.setAsBackgroundMesh(thr)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", case.n_curv)
        gmsh.option.setNumber("Mesh.MeshSizeMax", case.h_size)
        gmsh.option.setNumber("Mesh.MeshSizeMin", case.h_min)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.Algorithm3D", 10)          # HXT, multi-threaded Delaunay
        gmsh.option.setNumber("General.NumThreads", 1)
        return volume_mesh()
    finally:
        gmsh.finalize()


# analysis ---------------------------------------------------------------------------------------
def _face_data(m, name, idx):
    tris = m.surfaces[name]
    faces = ccx.faces_of_triangles(tris, idx)
    area, cen = triangle_areas_centroids(m.nodes, tris)
    return tris, faces, area, cen


def run_vane(case: VaneCase, work: Path, threads: int = 1, keep_fields: bool = True,
             external: ExternalBC | None = None) -> dict:
    """Run the whole chain for one design; ``external`` overrides the hot-gas table read from ``case.cfd_dir``."""
    t_start = time.time()
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    lay = case.layout
    if not lay.valid():
        raise ValueError(f"invalid cooling layout: {lay.check()}")
    step = export(lay, work / "vane.step", span=case.span)
    m = mesh_vane(step, case)
    t_mesh = time.time()
    idx = ccx.face_index(m.tets)

    ext = external if external is not None else ExternalBC.from_dir(Path(case.cfd_dir))
    _, f_out, a_out, c_out = _face_data(m, "outer", idx)
    p_gas, T_aw, h_gas = ext.at(c_out, case.span, case.radial_peak)
    if case.tbc_thickness > 0:      # coating resistance in series with the gas-side film
        h_gas = 1.0 / (1.0 / h_gas + case.tbc_thickness / case.tbc_conductivity)
    m_main = ext.mdot_per_m * case.span
    cool = Coolant(m_total=case.coolant_fraction * m_main, T_in=case.T_coolant_in, p_in=case.p_coolant)
    diam = lay.diameters()
    m_pass = cool.split(diam)

    holes = []
    z_edges = np.linspace(0, case.span, case.n_bins + 1)
    for i in range(lay.n):
        tris, faces, area, cen = _face_data(m, f"hole{i}", idx)
        holes.append(dict(tris=tris, faces=faces, area=area, d=diam[i], m=m_pass[i],
                          bin=np.clip(np.digitize(cen[:, 2], z_edges) - 1, 0, case.n_bins - 1),
                          T_c=np.full(case.n_bins, case.T_coolant_in)))

    def h_cool(hl):
        return case.rib_factor * cool.passage(hl["m"], hl["d"], hl["T_c"])["h"]

    # thermal problem coupled with the coolant march -------------------------------------------------
    history = []
    for _ in range(8):
        d = ccx.Deck(m.nodes, m.tets).mesh().material(MAT, mechanical=False).initial_temperature(1000.0)
        d.raw("*STEP", "*HEAT TRANSFER, STEADY STATE")
        d.film(f_out, T_aw, h_gas)
        for hl in holes:
            d.film(hl["faces"], hl["T_c"][hl["bin"]], h_cool(hl)[hl["bin"]])
        d.raw("*NODE FILE", "NT", "*END STEP").write(work / "thermal.inp")
        T = ccx.last_block(ccx.read_frd(ccx.run(work, "thermal", threads)), "NDTEMP").full(m.n_nodes)[:, 0]
        change = 0.0
        for hl in holes:
            T_face = T[hl["tris"][:, :3]].mean(1)
            q = h_cool(hl)[hl["bin"]] * (T_face - hl["T_c"][hl["bin"]]) * hl["area"]
            q_bin = np.bincount(hl["bin"], weights=q, minlength=case.n_bins)
            new = cool.march(hl["m"], z_edges, q_bin)
            change = max(change, float(np.abs(new - hl["T_c"]).max()))
            hl["T_c"], hl["q"] = new, q_bin
        history.append(change)
        if change < 0.5:
            break
    t_thermal = time.time()

    # energy balance: heat entering from the gas = heat taken by the coolant (adiabatic end walls)
    q_in = float(np.sum(h_gas * (T_aw - T[m.surfaces["outer"][:, :3]].mean(1)) * a_out))
    q_cool = float(sum(hl["q"].sum() for hl in holes))

    # static analysis: thermal strains + gas and coolant pressure ----------------------------------------
    # Thermally free support: the hub face is held radially (uz = 0, which also prevents rigid rotation
    # out of the plane) but may expand in its own plane, as the hot platform does; two hub nodes remove
    # the in-plane rigid-body motion (3-2-1 support). Clamping the hub in-plane instead would restrain the
    # free thermal expansion and create a spurious stress of order E alpha dT ~ 2 GPa near the hub.
    hub, tip = m.surface_nodes("hub"), m.surface_nodes("tip")
    hub_xy = m.nodes[hub, :2]
    a = hub[np.argmin(np.linalg.norm(hub_xy - hub_xy.mean(0), axis=1))]
    b = hub[np.argmax(np.linalg.norm(hub_xy - m.nodes[a, :2], axis=1))]
    ab = m.nodes[b, :2] - m.nodes[a, :2]
    nrm = np.array([-ab[1], ab[0]]) / np.linalg.norm(ab)
    d = ccx.Deck(m.nodes, m.tets).mesh().material(MAT, thermal=False)
    d.nset("HUB", hub).boundary("HUB", 3).nset("HUBA", [a]).boundary("HUBA", 1, 2)
    d.raw("*EQUATION", "2", f"{b + 1}, 1, {nrm[0]:.10f}, {b + 1}, 2, {nrm[1]:.10f}")
    if case.tip_support == "radial":
        d.nset("TIP", tip).boundary("TIP", 1, 2)
    d.initial_temperature(MAT.alpha_ref_T)
    d.raw("*STEP", "*STATIC").temperatures(T).pressure(f_out, p_gas)
    for hl in holes:
        d.pressure(hl["faces"], np.full(len(hl["faces"]), case.p_coolant))
    d.raw("*NODE FILE", "U", "*EL FILE", "S", "*END STEP").write(work / "mech.inp")
    blocks = ccx.read_frd(ccx.run(work, "mech", threads))
    stress = ccx.last_block(blocks, "STRESS").full(m.n_nodes)
    U = ccx.last_block(blocks, "DISP").full(m.n_nodes)[:, :3]

    # modal analysis: hub clamped (stiff platform), Young's modulus at the operating temperatures
    d = ccx.Deck(m.nodes, m.tets).mesh().material(MAT, thermal=False)
    d.nset("HUB", hub).boundary("HUB", 1, 3).initial_temperature(T)
    # (initial = final temperature: material data at T, no thermal strain)
    d.raw("*STEP", "*FREQUENCY", str(case.n_modes)).temperatures(T)
    if keep_fields:
        d.raw("*NODE FILE", "U")                                  # mode shapes, one DISP block per mode
    d.raw("*END STEP").write(work / "modal.inp")
    modal_frd = ccx.run(work, "modal", threads)
    freqs = ccx.read_eigenfrequencies(work / "modal.dat")
    modes = ([b.full(m.n_nodes)[:, :3] for b in ccx.read_frd(modal_frd) if b.name == "DISP"][:3]
             if keep_fields else [])
    t_mech = time.time()

    vm = ccx.von_mises(stress)
    z = m.nodes[:, 2]
    core = (z > 0.1 * case.span) & (z < 0.9 * case.span)     # away from the idealised supports
    vol = np.abs(tet_volumes(m.nodes, m.tets))
    T_el = T[m.tets[:, :4]].mean(1)
    i_tmax = int(np.nanargmax(T))
    i_vm = int(np.nanargmax(np.where(core, vm, -np.inf)))
    inlet = [cool.passage(hl["m"], hl["d"], case.T_coolant_in) for hl in holes]
    ff = 3 * case.rib_factor if case.rib_factor > 1 else 1.0
    qois = dict(
        T_max=float(np.nanmax(T)), T_mean=float(np.sum(T_el * vol) / vol.sum()), T_min=float(np.nanmin(T)),
        T_max_xyz=m.nodes[i_tmax].tolist(),
        vm_max_core=float(vm[i_vm]), vm_max_core_xyz=m.nodes[i_vm].tolist(),
        vm_p99_core=float(np.nanpercentile(vm[core], 99)), vm_max_all=float(np.nanmax(vm)),
        U_max=float(np.nanmax(np.linalg.norm(U, axis=1))),
        **{f"f{i + 1}": float(f) for i, f in enumerate(freqs)},
        T_coolant_out=float(np.sum([hl["m"] * hl["T_c"][-1] for hl in holes]) / cool.m_total),
        Q_gas_W=q_in, Q_coolant_W=q_cool, energy_balance_rel=(q_in - q_cool) / q_in,
        coolant_mdot=cool.m_total, coolant_Re_min=float(min(c["Re"] for c in inlet)),
        coolant_Mach=float(max(c["Mach"] for c in inlet)),
        coolant_dp=max(cool.pressure_drop(hl["m"], hl["d"], case.span, case.T_coolant_in, ff) for hl in holes),
        metal_volume=float(vol.sum()), mass=float(vol.sum() * MAT.rho),
        n_nodes=int(m.n_nodes), n_elements=int(len(m.tets)), picard_iterations=len(history),
        picard_history=history, time_mesh_s=t_mesh - t_start, time_thermal_s=t_thermal - t_mesh,
        time_mech_s=t_mech - t_thermal, diameters=diam.tolist(), passage_mdot=m_pass.tolist(),
        **{f"geom_{k}": v for k, v in lay.check().items()})
    (work / "qois.json").write_text(json.dumps(dict(case=case.to_dict(), qois=qois), indent=2))
    if keep_fields:
        surf = np.unique(np.vstack([m.surfaces[k][:, :3] for k in m.surfaces]))
        mid = np.abs(z - 0.5 * case.span) < 0.5 * case.h_size      # volume nodes near mid-span
        np.savez_compressed(work / "fields.npz", xyz=m.nodes[surf].astype(np.float32), T=T[surf].astype(np.float32),
                            vm=vm[surf].astype(np.float32), U=U[surf].astype(np.float32),
                            outer_tris=_reindex(m.surfaces["outer"][:, :3], surf),
                            T_c=np.array([hl["T_c"] for hl in holes]), z_bins=0.5 * (z_edges[1:] + z_edges[:-1]),
                            mid_xy=m.nodes[mid, :2].astype(np.float32), mid_T=T[mid].astype(np.float32),
                            mid_vm=vm[mid].astype(np.float32), hole_xy=lay.centres(), hole_d=diam,
                            modes=np.array([mo[surf] for mo in modes], dtype=np.float32),
                            mode_freqs=np.array(freqs[:len(modes)]))
    return qois


def _reindex(tris, keep):
    lut = np.full(keep.max() + 1, -1)
    lut[keep] = np.arange(len(keep))
    return lut[tris]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Thermo-mechanical and modal analysis of the cooled vane.")
    ap.add_argument("--work", default="/tmp/vanefea/baseline")
    ap.add_argument("--n", type=int, default=7)
    ap.add_argument("--ratio", type=float, default=0.40, help="passage diameter / local thickness")
    ap.add_argument("--f-start", type=float, default=0.10)
    ap.add_argument("--f-end", type=float, default=0.80)
    ap.add_argument("--coolant", type=float, default=0.03, help="coolant fraction of the main flow")
    ap.add_argument("--radial-peak", type=float, default=0.0)
    ap.add_argument("--h", type=float, default=2.5e-3)
    ap.add_argument("--h-hole", type=float, default=0.7e-3)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    c = VaneCase(layout=CoolingLayout(n=a.n, ratio=a.ratio, f_start=a.f_start, f_end=a.f_end),
                 coolant_fraction=a.coolant, radial_peak=a.radial_peak, h_size=a.h, h_hole=a.h_hole)
    print(json.dumps(run_vane(c, Path(a.work), a.threads), indent=2))
