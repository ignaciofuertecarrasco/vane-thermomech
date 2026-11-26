"""CAM for the cooled vane: finishing toolpath of the aerofoil, drilling of the passages, G-code.

The vane is prismatic, so its aerofoil can be finished by 2.5-axis contouring with a flat end mill
(side milling at successive depths). This module

1. offsets the section outward by the tool radius (tool-centre path, no controller compensation);
2. checks for **gouging**: in a concave region the tool radius must be smaller than the local
   radius of curvature of the surface, otherwise the offset curve folds and the tool cuts into the
   part (the LS-89 pressure side is concave);
3. linearises the path into G1 segments with a **chordal tolerance** (adaptive subdivision);
4. **verifies** the program by reconstructing the machined section as the envelope of the tool
   positions and measuring its deviation from the nominal surface (form error from the
   linearisation, the scallop between moves is zero for side milling with a flat tool);
5. writes the radial cooling passages as peck-drilling cycles (G83). With length/diameter ratios
   of 20-35 these are electro-chemical (STEM) or EDM drilled in production; the cycle defines the
   positions and depths whatever the process.

Units in the G-code: millimetres (G21), absolute coordinates (G90), XY plane (G17).

Usage::

    python -m vanefea.cam --tool-d 6 --tol 0.005 --out results/cam/vane.nc
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vanefea.cad import SPAN
from vanefea.geometry import CoolingLayout, load_profile, resample_closed


def curvature(pts: np.ndarray) -> np.ndarray:
    """Signed curvature of a closed counter-clockwise polyline (positive = convex)."""
    d1 = (np.roll(pts, -1, 0) - np.roll(pts, 1, 0)) / 2
    d2 = np.roll(pts, -1, 0) - 2 * pts + np.roll(pts, 1, 0)
    return (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / np.linalg.norm(d1, axis=1) ** 3


def outward_normals(pts: np.ndarray) -> np.ndarray:
    t = np.roll(pts, -1, 0) - np.roll(pts, 1, 0)
    t /= np.linalg.norm(t, axis=1)[:, None]
    return np.c_[t[:, 1], -t[:, 0]]          # right of the tangent for a counter-clockwise loop


def gouge_check(section: np.ndarray, tool_r: float) -> dict:
    k = curvature(section)
    concave = k < 0
    r_min = float(1 / -k[concave].min()) if concave.any() else np.inf
    return dict(min_concave_radius=r_min, tool_radius=tool_r, ok=bool(tool_r < r_min))


def linearise(path_fn, n_dense: int, tol: float) -> np.ndarray:
    """Adaptive subdivision of a closed parametric path so that every chord deviates < tol from the curve."""
    u = np.linspace(0, 1, n_dense, endpoint=False)
    dense = path_fn(u)
    keep = [0]
    i = 0
    while i < n_dense - 1:
        j = i + 2
        while j <= n_dense:
            a, b = dense[i], dense[j % n_dense]
            seg = dense[i:j + 1] if j < n_dense else np.vstack([dense[i:], dense[:1]])
            ab = b - a
            L = np.linalg.norm(ab)
            rel = seg - a
            dev = np.abs(ab[0] * rel[:, 1] - ab[1] * rel[:, 0]) / L if L > 0 else np.zeros(len(seg))
            if dev.max() > tol:
                break
            j += 1
        i = max(j - 1, i + 1)
        if i < n_dense:
            keep.append(i)
    return dense[sorted(set(keep))]


def toolpath(tool_d: float, tol: float, n_dense: int = 6000):
    """Tool-centre path (closed, mm) and the checks; the section is in mm."""
    section = resample_closed(load_profile() * 1e3, n_dense)
    r = 0.5 * tool_d
    g = gouge_check(section, r)
    offset = section + r * outward_normals(section)

    def path_fn(u):
        idx = (u * n_dense).astype(int) % n_dense
        return offset[idx]

    pts = linearise(path_fn, n_dense, tol)
    return section, pts, g


def machined_error(section: np.ndarray, centres: np.ndarray, tool_r: float, n_sub: int = 20) -> np.ndarray:
    """Distance from the nominal surface to the swept tool boundary (positive = material left)."""
    from scipy.spatial import cKDTree
    seg = np.vstack([np.linspace(centres[i], centres[(i + 1) % len(centres)], n_sub, endpoint=False)
                     for i in range(len(centres))])
    d, _ = cKDTree(seg).query(section)
    return d - tool_r


def gcode(centres: np.ndarray, layout: CoolingLayout, span_mm: float, tool_d: float, step_down: float,
          feed: float = 400.0, rpm: int = 9000, peck: float = 2.0) -> str:
    safe = 5.0
    lines = ["%", "(cooled vane LS-89 section - finishing and passage drilling)", "G21 G90 G17 G40 G49 G80",
             f"(T1: flat end mill D{tool_d:.2f}, side finishing)", "T1 M6", f"S{rpm} M3", "M8",
             f"G0 Z{safe:.3f}"]
    x0, y0 = centres[0]
    depths = np.arange(step_down, span_mm + 1e-9, step_down)
    if depths[-1] < span_mm:
        depths = np.r_[depths, span_mm]
    for z in depths:
        lines += [f"G0 X{x0:.4f} Y{y0:.4f}", f"G1 Z{-z:.4f} F{feed / 4:.0f}"]
        lines += [f"G1 X{x:.4f} Y{y:.4f} F{feed:.0f}" for x, y in centres[1:]]
        lines += [f"G1 X{x0:.4f} Y{y0:.4f}", f"G0 Z{safe:.3f}"]
    # one drill per passage diameter (rounded to 0.05 mm), largest first
    diam = np.round(layout.diameters() * 1e3 / 0.05) * 0.05
    xy = layout.centres() * 1e3
    lines.append("M9")
    for k, dd in enumerate(sorted(set(diam.tolist()), reverse=True)):
        tool = k + 2
        lines += [f"(T{tool}: drill D{dd:.2f}, radial cooling passages, peck {peck:.1f} mm)", f"T{tool} M6",
                  f"S{int(min(12000, 9000 / max(dd, 0.5)))} M3", "M8", f"G0 Z{safe:.3f}"]
        for i, (x, y) in enumerate(xy[diam == dd]):
            cyc = "G98 G83 " if i == 0 else ""
            lines.append(f"{cyc}X{x:.4f} Y{y:.4f} Z{-span_mm - 0.5:.4f} R{2.0:.3f} Q{min(peck, dd):.3f} F40")
        lines += ["G80", "M9"]
    lines += ["M5", f"G0 Z{safe + 45:.3f}", "M30", "%"]
    return "\n".join(lines) + "\n"


def plot(section, centres, err, layout, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.4), gridspec_kw=dict(width_ratios=[1.1, 1]))
    ax[0].fill(section[:, 0], section[:, 1], c="0.85", label="section")
    ax[0].plot(*np.vstack([centres, centres[:1]]).T, "C0-", lw=0.8, label=f"tool-centre path ({len(centres)} G1 moves)")
    for (x, y), d in zip(layout.centres() * 1e3, layout.diameters()):
        ax[0].add_patch(plt.Circle((x, y), d * 500, fc="w", ec="C3"))
    ax[0].plot([], [], "o", mfc="w", mec="C3", label="drilled passages (G83)")
    ax[0].set_aspect("equal")
    ax[0].set_xlabel("x [mm]")
    ax[0].set_ylabel("y [mm]")
    ax[0].legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=1)
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(section, axis=0), axis=1))]
    ax[1].plot(arc, err * 1e3, lw=0.8)
    ax[1].set_xlabel("surface distance [mm]")
    ax[1].set_ylabel("machined - nominal surface [µm]")
    ax[1].set_title("form error of the program (envelope of the tool)")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(png, dpi=150)
    plt.close(fig)


def main(tool_d, tol, step_down, out: Path, layout: CoolingLayout):
    section, centres, g = toolpath(tool_d, tol)
    err = machined_error(section, centres, 0.5 * tool_d)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(gcode(centres, layout, SPAN * 1e3, tool_d, step_down))
    plot(section, centres, err, layout, out.with_suffix(".png"))
    report = dict(gouge=g, n_moves=len(centres), chord_tolerance_mm=tol, max_overcut_um=float(-err.min() * 1e3),
                  max_material_left_um=float(err.max() * 1e3), passages=layout.n, passage_L_over_d_max=float(SPAN / layout.diameters().min()),
                  gcode_lines=len(out.read_text().splitlines()))
    out.with_suffix(".json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Finishing toolpath, drilling cycles and G-code for the vane.")
    ap.add_argument("--tool-d", type=float, default=6.0, help="end-mill diameter [mm]")
    ap.add_argument("--tol", type=float, default=0.005, help="chordal tolerance [mm]")
    ap.add_argument("--step-down", type=float, default=10.0, help="axial depth per pass [mm]")
    ap.add_argument("--out", default="results/cam/vane.nc")
    ap.add_argument("--n", type=int, default=7)
    ap.add_argument("--ratio", type=float, default=0.40)
    ap.add_argument("--f-start", type=float, default=0.10)
    ap.add_argument("--f-end", type=float, default=0.80)
    a = ap.parse_args()
    lay = CoolingLayout(n=a.n, ratio=a.ratio, f_start=a.f_start, f_end=a.f_end)
    print(json.dumps(main(a.tool_d, a.tol, a.step_down, Path(a.out), lay), indent=2))
