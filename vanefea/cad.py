"""Parametric CAD model of the cooled vane (CadQuery / OpenCASCADE), exported to STEP and STL.

The solid is the LS-89 section extruded radially over the span ``span``, with the radial cooling
passages of a ``CoolingLayout`` cut through it. The section is a periodic B-spline through the
profile points, so the STEP file carries an exact smooth surface rather than facets.

Coordinates are in metres in the TurboAI/CFD frame (x axial, y tangential, z radial from the hub),
so that the CFD boundary conditions can be mapped onto the CAD faces without any transformation.

Usage::

    python -m vanefea.cad --n 7 --ratio 0.4 --out results/cad/vane.step
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cadquery as cq
import numpy as np

from vanefea.geometry import CoolingLayout, load_profile, resample_closed

SPAN = 0.050        # radial height of the vane [m]


def section_points(n: int = 160) -> list[tuple[float, float]]:
    """Profile points for the CAD spline (fewer than for the analysis, to keep the B-spline smooth)."""
    pts = resample_closed(load_profile(), n)
    return [tuple(p) for p in pts]


def vane_solid(layout: CoolingLayout | None, span: float = SPAN) -> cq.Workplane:
    """Vane body with its cooling passages."""
    body = cq.Workplane("XY").spline(section_points(), periodic=True).close().extrude(span)
    if layout is not None and layout.n > 0:
        for c, d in zip(layout.centres(), layout.diameters()):
            body = body.cut(cq.Workplane("XY").center(*c).circle(0.5 * d).extrude(span))
    return body


def export(layout: CoolingLayout | None, step: Path, stl: Path | None = None, span: float = SPAN) -> Path:
    """Write the STEP file (and optionally an STL for visualisation or additive manufacturing)."""
    solid = vane_solid(layout, span)
    step = Path(step)
    step.parent.mkdir(parents=True, exist_ok=True)
    cq.exporters.export(solid, str(step))
    if stl is not None:
        cq.exporters.export(solid, str(stl), tolerance=1e-4, angularTolerance=0.3)
    return step


def properties(layout: CoolingLayout | None, span: float = SPAN) -> dict:
    """Volume, mass-relevant properties and the outer wetted area of the CAD solid."""
    s = vane_solid(layout, span).val()
    return dict(volume_m3=s.Volume(), area_m2=s.Area(), n_faces=len(s.Faces()),
                centroid=list(np.round(s.Center().toTuple(), 7)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Export the cooled vane CAD model.")
    ap.add_argument("--n", type=int, default=7)
    ap.add_argument("--ratio", type=float, default=0.40)
    ap.add_argument("--f-start", type=float, default=0.10)
    ap.add_argument("--f-end", type=float, default=0.80)
    ap.add_argument("--span", type=float, default=SPAN)
    ap.add_argument("--out", default="results/cad/vane.step")
    a = ap.parse_args()
    lay = CoolingLayout(n=a.n, ratio=a.ratio, f_start=a.f_start, f_end=a.f_end)
    print("layout check:", lay.check())
    out = export(lay, Path(a.out), Path(a.out).with_suffix(".stl"), a.span)
    print("wrote", out, properties(lay, a.span))
