"""A small Python interface to the CalculiX solver (``ccx``): input decks, execution and result files.

CalculiX reads Abaqus-style ``.inp`` decks and writes nodal results to an ASCII ``.frd`` file and
printed output (eigenvalues, reaction forces) to a ``.dat`` file. This module

* writes decks from NumPy arrays (``Deck``): nodes, quadratic tetrahedra (C3D10), node sets,
  temperature-dependent materials, film and pressure loads on element faces;
* runs ``ccx`` in a working directory and checks that the job really finished;
* parses ``.frd`` result blocks and the eigenvalue table of the ``.dat`` file.

Face numbering of the C3D10 element (same as C3D4, 1-based corner nodes):
F1 = (1, 2, 3), F2 = (1, 4, 2), F3 = (2, 4, 3), F4 = (3, 4, 1).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from vanefea.materials import Material

# corner nodes (0-based) of the four faces of a tetrahedron, CalculiX/Abaqus convention
TET_FACES = ((0, 1, 2), (0, 3, 1), (1, 3, 2), (2, 3, 0))


class CalculixError(RuntimeError):
    pass


def ccx_executable() -> str:
    exe = os.environ.get("CCX", "ccx")
    if shutil.which(exe) is None:
        raise CalculixError(f"CalculiX executable '{exe}' not found (install calculix-ccx or set CCX)")
    return exe


def face_index(tets: np.ndarray) -> dict:
    """Map each tetrahedron face (sorted corner node ids) to (element number, face number 1-4)."""
    idx = {}
    for e, t in enumerate(tets[:, :4]):
        for f, (a, b, c) in enumerate(TET_FACES):
            idx[tuple(sorted((t[a], t[b], t[c])))] = (e + 1, f + 1)
    return idx


def faces_of_triangles(tris: np.ndarray, index: dict) -> np.ndarray:
    """Element faces (n x 2: element, face) matching boundary triangles given by their corner nodes."""
    out = np.empty((len(tris), 2), dtype=int)
    for i, t in enumerate(tris[:, :3]):
        out[i] = index[tuple(sorted(t))]
    return out


@dataclass
class Deck:
    """Builder for a CalculiX input deck. Node and element numbers are 1-based in the deck."""
    nodes: np.ndarray                  # (n, 3) coordinates [m]
    tets: np.ndarray                   # (m, 10) node indices, 0-based, CalculiX C3D10 order
    lines: list = field(default_factory=list)

    def mesh(self, elset="EALL"):
        self.lines.append("*NODE, NSET=NALL")
        self.lines += [f"{i + 1}, {x:.10g}, {y:.10g}, {z:.10g}" for i, (x, y, z) in enumerate(self.nodes)]
        self.lines.append(f"*ELEMENT, TYPE=C3D10, ELSET={elset}")
        for e, t in enumerate(self.tets + 1):
            self.lines.append(f"{e + 1}, " + ", ".join(map(str, t[:8])) + ",")
            self.lines.append(", ".join(map(str, t[8:])))
        return self

    def nset(self, name, ids0):
        self.lines.append(f"*NSET, NSET={name}")
        ids = np.asarray(ids0) + 1
        for k in range(0, len(ids), 12):
            self.lines.append(", ".join(map(str, ids[k:k + 12])))
        return self

    def material(self, m: Material, elset="EALL", name="MAT", thermal=True, mechanical=True):
        self.lines += [f"*MATERIAL, NAME={name}"]
        if mechanical:
            self.lines += ["*ELASTIC"] + [f"{E:.6g}, {m.nu:.4g}, {T:.6g}" for E, T in zip(m.E, m.T)]
            self.lines += [f"*EXPANSION, ZERO={m.alpha_ref_T:.6g}"] + [f"{a:.6g}, {T:.6g}" for a, T in zip(m.alpha, m.T)]
            self.lines += ["*DENSITY", f"{m.rho:.6g}"]
        if thermal:
            self.lines += ["*CONDUCTIVITY"] + [f"{k:.6g}, {T:.6g}" for k, T in zip(m.k, m.T)]
        self.lines += [f"*SOLID SECTION, ELSET={elset}, MATERIAL={name}"]
        return self

    def raw(self, *text):
        self.lines += list(text)
        return self

    def initial_temperature(self, T):
        """Initial temperature: a scalar (thermal start value, or stress-free state of a mechanical run)
        or one value per node (material properties at operating temperature in a modal analysis)."""
        if np.ndim(T) == 0:
            return self.raw("*INITIAL CONDITIONS, TYPE=TEMPERATURE", f"NALL, {float(T):.6g}")
        self.lines.append("*INITIAL CONDITIONS, TYPE=TEMPERATURE")
        self.lines += [f"{i + 1}, {t:.6g}" for i, t in enumerate(T)]
        return self

    def film(self, faces: np.ndarray, sink: np.ndarray, h: np.ndarray):
        """Convective heat flux q = h (T_sink - T) on element faces (one line per face)."""
        self.lines.append("*FILM")
        self.lines += [f"{e}, F{f}, {ts:.6g}, {hh:.6g}" for (e, f), ts, hh in zip(faces, sink, h)]
        return self

    def pressure(self, faces: np.ndarray, p: np.ndarray):
        """Distributed pressure on element faces, positive into the element."""
        self.lines.append("*DLOAD")
        self.lines += [f"{e}, P{f}, {pp:.8g}" for (e, f), pp in zip(faces, p)]
        return self

    def temperatures(self, T: np.ndarray):
        """Prescribed nodal temperatures for a mechanical step (from a previous thermal analysis)."""
        self.lines.append("*TEMPERATURE")
        self.lines += [f"{i + 1}, {t:.6g}" for i, t in enumerate(T)]
        return self

    def boundary(self, nset, first, last=None, value=None):
        last = first if last is None else last
        self.lines += ["*BOUNDARY", f"{nset}, {first}, {last}" + ("" if value is None else f", {value:.8g}")]
        return self

    def write(self, path: Path):
        Path(path).write_text("\n".join(self.lines) + "\n")
        return path


def run(job_dir: Path, job: str, threads: int = 1, timeout: float = 3600) -> Path:
    """Run ``ccx -i job`` in ``job_dir``; raises CalculixError unless the job ended normally."""
    env = {**os.environ, "OMP_NUM_THREADS": str(threads), "CCX_NPROC_EQUATION_SOLVER": str(threads)}
    with open(job_dir / f"{job}.log", "w") as log:
        rc = subprocess.call([ccx_executable(), "-i", job], cwd=job_dir, stdout=log, stderr=subprocess.STDOUT,
                             env=env, timeout=timeout)
    text = (job_dir / f"{job}.log").read_text(errors="replace")
    if rc != 0 or "*ERROR" in text or "Job finished" not in text:
        tail = "\n".join(text.splitlines()[-15:])
        raise CalculixError(f"ccx failed for {job_dir / job} (rc={rc}):\n{tail}")
    return job_dir / f"{job}.frd"


# result files -------------------------------------------------------------------------------------
@dataclass
class FrdBlock:
    name: str            # e.g. NDTEMP, DISP, STRESS
    step: int
    value: float         # time / frequency / load factor of the block
    components: list
    nodes: np.ndarray    # node numbers (1-based)
    data: np.ndarray     # (n_nodes, n_components)

    def full(self, n_nodes: int) -> np.ndarray:
        """Values for every node of the mesh (0-based rows); nodes without output are NaN."""
        out = np.full((n_nodes, self.data.shape[1]), np.nan)
        out[self.nodes - 1] = self.data
        return out


def read_frd(path: Path) -> list[FrdBlock]:
    """Parse the nodal result blocks of an ASCII .frd file."""
    blocks, lines = [], Path(path).read_text().splitlines()
    i, step, value = 0, 0, 0.0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("    1PSTEP"):
            step = int(ln.split()[2]) if len(ln.split()) > 2 else step
        elif ln.startswith("  100C"):
            try:
                value = float(ln[12:24])
            except ValueError:
                value = 0.0
        elif ln.startswith(" -4"):
            name = ln.split()[1]
            ncomp = int(ln.split()[2])
            comps, i = [], i + 1
            while lines[i].startswith(" -5"):
                comps.append(lines[i].split()[1])
                i += 1
            nodes, rows, cur = [], [], None
            while not lines[i].startswith(" -3"):
                ln2 = lines[i]
                vals = [float(ln2[k:k + 12]) for k in range(13, len(ln2), 12) if ln2[k:k + 12].strip()]
                if ln2.startswith(" -1"):
                    if cur is not None:
                        rows.append(cur)
                    nodes.append(int(ln2[3:13]))
                    cur = vals
                elif ln2.startswith(" -2"):
                    cur += vals
                i += 1
            if cur is not None:
                rows.append(cur)
            ncol = min(len(r) for r in rows) if rows else 0
            data = np.array([r[:ncol] for r in rows])
            blocks.append(FrdBlock(name, step, value, comps[:max(ncol, ncomp)][:ncol], np.array(nodes), data))
        i += 1
    return blocks


def last_block(blocks, name) -> FrdBlock:
    sel = [b for b in blocks if b.name == name]
    if not sel:
        raise CalculixError(f"no {name} block in the results")
    return sel[-1]


def read_eigenfrequencies(dat: Path) -> np.ndarray:
    """Natural frequencies [Hz] from the eigenvalue table of a .dat file."""
    text = Path(dat).read_text()
    m = re.search(r"E I G E N V A L U E   O U T P U T(.*?)(?:\n\s*\n\s*\n|P A R T I C I P|$)", text, re.S)
    if not m:
        raise CalculixError(f"no eigenvalue output in {dat}")
    freqs = []
    for ln in m.group(1).splitlines():
        parts = ln.split()
        if len(parts) >= 4 and parts[0].isdigit():
            freqs.append(float(parts[3]))
    return np.array(freqs)


def von_mises(S: np.ndarray) -> np.ndarray:
    """Von Mises stress from (sxx, syy, szz, sxy, syz, szx)."""
    sx, sy, sz, txy, tyz, tzx = S.T[:6]
    return np.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2) + 3 * (txy ** 2 + tyz ** 2 + tzx ** 2))
