"""Mesh convergence of the vane analysis (baseline cooling design).

The same design is analysed on three (optionally four) meshes of quadratic tetrahedra. For
each quantity of interest the observed order of convergence and the grid convergence index (GCI,
Roache, safety factor 1.25) are computed from the three finest meshes, using the cube root of the
node count as the representative size. The design-sweep mesh is chosen from this study.

Usage::

    python -m vanefea.mesh_study --work /tmp/vanefea/mesh_study
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vanefea.analysis import VaneCase, run_vane
from vanefea.geometry import CoolingLayout

ROOT = Path(__file__).resolve().parents[1]
LEVELS = {  # name: (h_size, h_hole, h_min, n_curv)
    "L1": (3.0e-3, 0.9e-3, 0.5e-3, 10),
    "L2": (2.5e-3, 0.7e-3, 0.4e-3, 12),
    "L3": (2.0e-3, 0.55e-3, 0.3e-3, 16),
    "L4": (1.5e-3, 0.4e-3, 0.2e-3, 24),
}
QOIS = ["T_max", "T_mean", "vm_max_core", "vm_p99_core", "f1", "f2", "f3", "T_coolant_out", "Q_gas_W", "U_max"]
BASELINE = CoolingLayout(n=7, ratio=0.40, f_start=0.10, f_end=0.80)


def gci(f1, f2, f3, r21, r32, fs=1.25):
    """Fine-grid GCI and observed order from three solutions (f1 finest)."""
    e21, e32 = f2 - f1, f3 - f2
    if e21 == 0 or e32 == 0 or np.sign(e21) != np.sign(e32):
        return dict(order=float("nan"), gci_fine=float(abs(e21 / f1)) * fs, monotonic=False)
    p = abs(np.log(abs(e32 / e21))) / np.log(r21)
    for _ in range(50):     # fixed-point iteration for unequal refinement ratios
        q = np.log((r21 ** p - np.sign(e32 / e21)) / (r32 ** p - np.sign(e32 / e21)))
        p = abs(np.log(abs(e32 / e21)) + q) / np.log(r21)
    return dict(order=float(p), gci_fine=float(fs * abs(e21 / f1) / (r21 ** p - 1)), monotonic=True,
                extrapolated=float(f1 + (f1 - f2) / (r21 ** p - 1)))


def main(work: Path, threads: int = 4, levels=("L1", "L2", "L3")):
    rows = {}
    for name in levels:
        h, hh, hmin, nc = LEVELS[name]
        case = VaneCase(layout=BASELINE, h_size=h, h_hole=hh, h_min=hmin, n_curv=nc)
        q = run_vane(case, work / name, threads=threads, keep_fields=(name == levels[-1]))
        rows[name] = q
        print(name, q["n_nodes"], {k: round(q[k], 4) for k in QOIS}, flush=True)
    names = list(levels)[-3:][::-1]                     # finest first
    n = np.array([rows[k]["n_nodes"] for k in names], float)
    r21, r32 = (n[0] / n[1]) ** (1 / 3), (n[1] / n[2]) ** (1 / 3)
    study = {k: gci(*[rows[m][k] for m in names], r21, r32) for k in QOIS}
    out = ROOT / "results" / "mesh_study"
    out.mkdir(parents=True, exist_ok=True)
    (out / "mesh_study.json").write_text(json.dumps(dict(levels=LEVELS, results=rows, gci=study), indent=2))
    return study


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--work", default="/tmp/vanefea/mesh_study")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--levels", nargs="+", default=["L1", "L2", "L3"])
    a = ap.parse_args()
    print(json.dumps(main(Path(a.work), a.threads, tuple(a.levels)), indent=2))
