"""2D vane section and the layout of the radial cooling passages.

The section is the VKI LS-89 high-pressure turbine nozzle guide vane (axial chord 36.97 mm, true
chord 67.65 mm), read from ``data/ls89_profile.dat`` in metres, leading edge at x = 0.

The vane is cooled by ``n`` radial cylindrical passages of diameter ``d`` placed on the camber line
between the fractions ``f_start`` and ``f_end`` of its length. A layout is valid when

* every passage keeps a metal wall of at least ``t_wall_min`` to the outer surface, and
* the ligament between neighbouring passages is at least ``ligament_min``.

Both are casting/drilling constraints that the design sweep must respect; ``CoolingLayout.check``
returns the margins so that infeasible designs are rejected before any CAD or meshing is done.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.interpolate import splev, splprep
from scipy.spatial import cKDTree

DATA = Path(__file__).resolve().parent / "data"
PROFILE = DATA / "ls89_profile.dat"


def resample_closed(pts: np.ndarray, n: int) -> np.ndarray:
    """Periodic cubic spline through a closed outline, resampled with n points equally spaced in arc length."""
    P = np.vstack([pts, pts[:1]])
    tck, _ = splprep([P[:, 0], P[:, 1]], s=0, per=True)
    u = np.linspace(0, 1, 40001)[:-1]
    x, y = splev(u, tck)
    arc = np.r_[0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
    ui = np.interp(np.linspace(0, arc[-1], n, endpoint=False), arc, u)
    return np.column_stack(splev(ui, tck))


@lru_cache(maxsize=4)
def load_profile(path: str | Path = PROFILE, n: int = 400) -> np.ndarray:
    """Closed section outline (n x 2, metres), counter-clockwise, starting at the leading edge."""
    raw = np.loadtxt(path)
    pts = resample_closed(raw, n)
    if _signed_area(pts) < 0:
        pts = pts[::-1]
    i_le = int(pts[:, 0].argmin())
    return np.roll(pts, -i_le, axis=0)


def _signed_area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def sides(profile: np.ndarray):
    """Split the outline at the trailing edge (the point farthest from the LE) into two LE -> TE curves."""
    i_te = int(np.linalg.norm(profile - profile[0], axis=1).argmax())
    a = profile[: i_te + 1]
    b = np.vstack([profile[i_te:], profile[:1]])[::-1]
    return a, b


@lru_cache(maxsize=4)
def camber_line(n: int = 200) -> np.ndarray:
    """Approximate camber line: midpoints of the two sides at equal fractional arc length (n x 2)."""
    a, b = sides(load_profile())

    def by_fraction(c):
        arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))]
        t = np.linspace(0, 1, n)
        return np.c_[np.interp(t, arc / arc[-1], c[:, 0]), np.interp(t, arc / arc[-1], c[:, 1])]

    return 0.5 * (by_fraction(a) + by_fraction(b))


def point_on_camber(f: float) -> np.ndarray:
    """Point at the fraction f (0 = LE, 1 = TE) of the camber-line length."""
    c = camber_line()
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))]
    return np.array([np.interp(f * arc[-1], arc, c[:, 0]), np.interp(f * arc[-1], arc, c[:, 1])])


@lru_cache(maxsize=2)
def _surface_tree(n: int = 4000):
    return cKDTree(resample_closed(np.loadtxt(PROFILE), n))


def wall_distance(xy: np.ndarray) -> np.ndarray:
    """Distance from interior points to the outer surface [m]."""
    d, _ = _surface_tree().query(np.atleast_2d(xy))
    return d


def local_thickness(f: float) -> float:
    """Diameter of the largest circle centred on the camber line at fraction f."""
    return float(2 * wall_distance(point_on_camber(f))[0])


@dataclass
class CoolingLayout:
    """Radial cooling passages on the camber line.

    Passage i sits at the camber fraction f_i (evenly spaced from ``f_start`` to ``f_end``) and its
    diameter is ``ratio`` times the local section thickness, clipped to [d_min, d_max]: large
    passages in the thick front part, small STEM-drilled holes towards the trailing edge, as in
    radial-hole cooled vanes.
    """
    n: int = 7
    ratio: float = 0.40        # passage diameter / local thickness
    f_start: float = 0.10      # first passage, fraction of camber length
    f_end: float = 0.80        # last passage
    d_min: float = 0.6e-3      # smallest drillable passage [m]
    d_max: float = 3.0e-3
    t_wall_min: float = 0.6e-3
    ligament_min: float = 0.8e-3
    extra: dict = field(default_factory=dict)

    def fractions(self) -> np.ndarray:
        return np.linspace(self.f_start, self.f_end, self.n) if self.n > 1 else np.array([self.f_start])

    def centres(self) -> np.ndarray:
        return np.array([point_on_camber(fi) for fi in self.fractions()])

    def diameters(self) -> np.ndarray:
        t = np.array([local_thickness(fi) for fi in self.fractions()])
        return np.clip(self.ratio * t, self.d_min, self.d_max)

    def check(self) -> dict:
        """Margins of the manufacturing constraints [m]; the layout is valid if both are >= 0."""
        c, d = self.centres(), self.diameters()
        wall = wall_distance(c) - 0.5 * d
        if self.n > 1:
            lig = np.linalg.norm(np.diff(c, axis=0), axis=1) - 0.5 * (d[1:] + d[:-1])
        else:
            lig = np.array([np.inf])
        return dict(wall_margin=float(wall.min() - self.t_wall_min),
                    ligament_margin=float(lig.min() - self.ligament_min),
                    min_wall=float(wall.min()), min_ligament=float(lig.min()),
                    d_front=float(d[0]), d_rear=float(d[-1]))

    def valid(self) -> bool:
        m = self.check()
        return m["wall_margin"] >= 0 and m["ligament_margin"] >= 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("extra")
        return d


def section_area(profile: np.ndarray | None = None) -> float:
    return abs(_signed_area(load_profile() if profile is None else profile))
