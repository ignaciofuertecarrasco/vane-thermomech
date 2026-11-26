"""Design sweep of the cooling layout and its delivery as a machine-learning data set.

Design variables (``BOUNDS``):

==============  ============================================  =================
``n``           number of radial passages                      4 - 9 (integer)
``ratio``       passage diameter / local section thickness     0.30 - 0.50
``f_start``     first passage position on the camber line      0.06 - 0.20
``f_end``       last passage position                          0.65 - 0.90
``coolant``     coolant / main-stream mass-flow ratio          1.5 - 4.5 %
==============  ============================================  =================

Samples come from a scrambled Sobol sequence; layouts that violate the wall-thickness or ligament
constraints are rejected *before* any CAD is built (``CoolingLayout.check``) and replaced by the
next point of the sequence, so the plan is reproducible. Cases run in parallel, each in its own
directory; a finished case is skipped when the campaign is restarted (resumable), and a failed one
leaves a ``FAILED`` file with the error instead of stopping the campaign.

``collect`` writes the data set in a layout an ML team can consume without this code:

* ``dataset.h5``: ``inputs`` (N x 5), ``outputs`` (N x K) with column names as attributes, the
  per-case surface fields (padded point clouds with temperature and von Mises stress), and the
  train/test split;
* ``dataset_card.md``: what every column means, units, the solver set-up, the verification
  status and the known limitations;
* a SHA-256 of the HDF5 file in the card, so that a model can cite the exact data version.

Usage::

    python -m vanefea.campaign plan --name sweep_v1 --n 120
    python -m vanefea.campaign run --name sweep_v1 --workers 3
    python -m vanefea.campaign collect --name sweep_v1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from vanefea.geometry import CoolingLayout

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path(os.environ.get("VANEFEA_RUNS", "~/vanefea-runs")).expanduser()
BOUNDS = {"n": (4, 9), "ratio": (0.30, 0.50), "f_start": (0.06, 0.20), "f_end": (0.65, 0.90),
          "coolant": (0.015, 0.045)}
INPUTS = list(BOUNDS)
OUTPUTS = ["T_max", "T_mean", "T_min", "vm_max_core", "vm_p99_core", "U_max", "f1", "f2", "f3", "T_coolant_out",
           "Q_gas_W", "coolant_Mach", "coolant_dp", "mass", "energy_balance_rel"]
# campaign mesh: level L1 of the mesh study (vanefea.mesh_study), whose error is quantified there
MESH = dict(h_size=3.0e-3, h_hole=0.9e-3, h_min=0.5e-3, n_curv=10)


def _layout(row) -> CoolingLayout:
    return CoolingLayout(n=int(row["n"]), ratio=row["ratio"], f_start=row["f_start"], f_end=row["f_end"])


def plan(name: str, n: int, seed: int = 0, n_test: int = 20):
    sob = qmc.Sobol(d=len(BOUNDS), scramble=True, seed=seed)
    rows, rejected = [], 0
    while len(rows) < n:
        u = sob.random(1)[0]
        row = {k: lo + ui * (hi - lo) for ui, (k, (lo, hi)) in zip(u, BOUNDS.items())}
        row["n"] = int(np.clip(np.floor(row["n"] + 0.5), *BOUNDS["n"]))
        if not _layout(row).valid():
            rejected += 1
            continue
        row["id"] = f"c{len(rows):04d}"
        rows.append(row)
    rng = np.random.default_rng(seed)
    test = set(rng.choice(len(rows), size=n_test, replace=False).tolist())
    for i, r in enumerate(rows):
        r["split"] = "test" if i in test else "train"
    d = RUNS / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "plan.json").write_text(json.dumps(dict(bounds=BOUNDS, mesh=MESH, rejected=rejected, cases=rows), indent=2))
    print(f"{name}: {len(rows)} cases ({rejected} infeasible layouts rejected)")


def run_one(row: dict, case_dir: str, threads: int = 2):
    from vanefea.analysis import VaneCase, run_vane
    d = Path(case_dir)
    if (d / "qois.json").exists():
        return row["id"], "skipped", 0.0
    t0 = time.time()
    try:
        case = VaneCase(layout=_layout(row), coolant_fraction=row["coolant"], **MESH)
        run_vane(case, d, threads=threads)
        (d / "FAILED").unlink(missing_ok=True)
        # the large solver files are not needed once the results are extracted
        for f in d.glob("*.inp"):
            f.unlink()
        for f in d.glob("*.frd"):
            f.unlink()
        return row["id"], "ok", time.time() - t0
    except Exception:
        d.mkdir(parents=True, exist_ok=True)
        (d / "FAILED").write_text(traceback.format_exc())
        return row["id"], "FAILED", time.time() - t0


def run(name: str, workers: int = 3, threads: int = 2):
    d = RUNS / name
    rows = json.loads((d / "plan.json").read_text())["cases"]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(run_one, r, str(d / "cases" / r["id"]), threads) for r in rows]
        for k, f in enumerate(as_completed(futs)):
            cid, status, dt = f.result()
            print(f"[{k + 1:4d}/{len(rows)}] {cid} {status} ({dt:.0f}s)  elapsed {(time.time() - t0) / 60:.1f} min", flush=True)


def collect(name: str, n_points: int = 4096):
    import h5py
    d = RUNS / name
    rows = json.loads((d / "plan.json").read_text())["cases"]
    X, Y, split, ids, clouds, failed = [], [], [], [], [], []
    rng = np.random.default_rng(0)
    for r in rows:
        q = d / "cases" / r["id"] / "qois.json"
        if not q.exists():
            failed.append(r["id"])
            continue
        qo = json.loads(q.read_text())["qois"]
        X.append([r[k] for k in INPUTS])
        Y.append([qo[k] for k in OUTPUTS])
        split.append(r["split"])
        ids.append(r["id"])
        f = np.load(d / "cases" / r["id"] / "fields.npz")
        sel = rng.choice(len(f["xyz"]), size=min(n_points, len(f["xyz"])), replace=False)
        cloud = np.full((n_points, 5), np.nan, np.float32)
        cloud[:len(sel)] = np.c_[f["xyz"][sel], f["T"][sel], f["vm"][sel]]
        clouds.append(cloud)
    out = ROOT / "results" / "dataset" / name
    out.mkdir(parents=True, exist_ok=True)
    h5 = out / "dataset.h5"
    with h5py.File(h5, "w") as fh:
        fh.create_dataset("inputs", data=np.array(X)).attrs["columns"] = INPUTS
        fh.create_dataset("outputs", data=np.array(Y)).attrs["columns"] = OUTPUTS
        fh.create_dataset("split", data=np.array(split, dtype="S5"))
        fh.create_dataset("case_id", data=np.array(ids, dtype="S8"))
        g = fh.create_dataset("surface_cloud", data=np.array(clouds), compression="gzip")
        g.attrs["columns"] = ["x", "y", "z", "T", "von_mises"]
        fh.attrs["generator"] = "vanefea.campaign"
        fh.attrs["bounds"] = json.dumps(BOUNDS)
    sha = hashlib.sha256(h5.read_bytes()).hexdigest()
    np.savetxt(out / "scalars.csv", np.c_[np.array(X), np.array(Y)], delimiter=",",
               header=",".join(INPUTS + OUTPUTS), comments="", fmt="%.8g")
    (out / "manifest.json").write_text(json.dumps(dict(name=name, n_cases=len(ids), n_failed=len(failed), failed=failed,
                                                       sha256=sha, inputs=INPUTS, outputs=OUTPUTS), indent=2))
    (out / "dataset_card.md").write_text(dataset_card(name, len(ids), split, failed, sha, np.array(Y)))
    print(f"{name}: {len(ids)} cases collected, {len(failed)} missing; sha256 {sha[:12]}")
    return out


UNITS = {
    "n": ("-", "number of radial cooling passages"),
    "ratio": ("-", "passage diameter / local section thickness (diameters clipped to 0.6-3.0 mm)"),
    "f_start": ("-", "camber-line position of the first passage (0 = leading edge, 1 = trailing edge)"),
    "f_end": ("-", "camber-line position of the last passage"),
    "coolant": ("-", "coolant mass flow / main-stream mass flow through one vane passage"),
    "T_max": ("K", "peak metal temperature (in practice at the trailing edge)"),
    "T_mean": ("K", "volume-averaged metal temperature"),
    "T_min": ("K", "lowest metal temperature"),
    "vm_max_core": ("Pa", "peak von Mises stress, 10-90 % span (away from the idealised supports)"),
    "vm_p99_core": ("Pa", "99th percentile of the nodal von Mises stress, 10-90 % span"),
    "U_max": ("m", "largest displacement"),
    "f1": ("Hz", "first natural frequency (hub clamped, hot Young's modulus)"),
    "f2": ("Hz", "second natural frequency"),
    "f3": ("Hz", "third natural frequency"),
    "T_coolant_out": ("K", "mass-averaged coolant temperature at the passage exits"),
    "Q_gas_W": ("W", "heat entering the vane from the hot gas"),
    "coolant_Mach": ("-", "largest passage-inlet Mach number"),
    "coolant_dp": ("Pa", "largest passage friction pressure drop"),
    "mass": ("kg", "vane mass"),
    "energy_balance_rel": ("-", "(heat from gas - heat to coolant) / heat from gas: a solution check"),
}


def dataset_card(name, n_cases, split, failed, sha, Y):
    """Human-readable description of the data set, written next to it."""
    rows = []
    for k in INPUTS:
        lo, hi = BOUNDS[k]
        rows.append(f"| `{k}` | input | {UNITS[k][0]} | {lo:g} - {hi:g} | {UNITS[k][1]} |")
    for j, k in enumerate(OUTPUTS):
        rows.append(f"| `{k}` | output | {UNITS[k][0]} | {Y[:, j].min():.4g} - {Y[:, j].max():.4g} | {UNITS[k][1]} |")
    n_test = sum(s == "test" for s in split)
    return f"""# Data set `{name}`: cooled LS-89 vane, thermal / static / modal FEA

* **Cases**: {n_cases} ({n_cases - n_test} train, {n_test} test; split fixed in the plan), {len(failed)} missing.
* **Version**: SHA-256 of `dataset.h5` = `{sha}`. Cite it when reporting a model trained on these data.
* **Generator**: `python -m vanefea.campaign plan|run|collect --name {name}` (scrambled Sobol plan, infeasible
  layouts rejected before meshing; every case independent and reproducible from its row of `plan.json`).

## Contents of `dataset.h5`

| Dataset | Shape | Meaning |
|---|---|---|
| `inputs` | (N, {len(INPUTS)}) | design variables, columns in the attribute `columns` |
| `outputs` | (N, {len(OUTPUTS)}) | quantities of interest, columns in the attribute `columns` |
| `split` | (N,) | `train` / `test` |
| `case_id` | (N,) | case directory name |
| `surface_cloud` | (N, 4096, 5) | random sample of surface nodes: x, y, z [m], temperature [K], von Mises [Pa] (NaN padding) |

`scalars.csv` holds the same inputs and outputs as plain text.

## Columns

| Name | Role | Unit | Range in the data | Meaning |
|---|---|---|---|---|
{chr(10).join(rows)}

## How the data were produced

Hot-gas boundary conditions from 2D RANS of the LS-89 cascade at P0 = 10 bar, T0 = 1600 K, exit Mach 0.9
(OpenFOAM, k-omega SST, viscous work included), uniform along the 50 mm span; 0.3 mm thermal-barrier coating;
coolant air at 800 K, 10.5 bar, rib-enhanced radial passages; CalculiX quadratic tetrahedra (mesh level L1 of
`vanefea.mesh_study`, about 90k nodes); steady conduction iterated with a 1D coolant march; static analysis with
thermal and pressure loads on a thermally free support; modal analysis with the hub clamped.

## Verification status

* Closed-form benchmarks of the solver set-up (`vanefea.verification`): conduction with convection and thermal
  stress in a thick cylinder, cantilever natural frequencies.
* Mesh study on the baseline design (`vanefea.mesh_study`): campaign mesh vs. finer meshes; see the README.
* Every case: energy balance between the gas-side and coolant-side heat flows (column `energy_balance_rel`).

## Known limitations

* The hot gas is two-dimensional: no secondary flows, end-wall effects or combustor hot streaks.
* No film cooling or trailing-edge ejection: the uncooled trailing edge sets `T_max`.
* Material data are representative of a cast nickel superalloy, not certified values; no creep, plasticity or life.
* The passage count `n` is an integer; models treating it as continuous should be evaluated at integer values.
"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Cooling-layout design sweep and ML data set.")
    ap.add_argument("cmd", choices=["plan", "run", "collect"])
    ap.add_argument("--name", default="sweep_v1")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--n-test", type=int, default=20)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    if a.cmd == "plan":
        plan(a.name, a.n, n_test=a.n_test)
    elif a.cmd == "run":
        run(a.name, a.workers, a.threads)
    else:
        collect(a.name)
