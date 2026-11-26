"""Boundary conditions of the vane: the hot-gas side from CFD and the coolant side from a 1D model.

Hot gas (external surface)
--------------------------
The aero-thermal CFD of the cascade (TurboAI, ``turboai.cfd.heat_transfer``) provides, at every wall
face of the 2D blade-to-blade mesh, the static pressure p, the adiabatic wall temperature T_aw and
the heat transfer coefficient h. ``ExternalBC`` maps them onto the faces of the 3D FE mesh by
nearest neighbour in the (x, y) plane: the CFD is two-dimensional, so the values are uniform along
the span unless a radial gas-temperature profile is imposed (``radial_peak``, a parabolic
combustor-exit profile that keeps the span-averaged T_aw unchanged).

Coolant (radial passages)
-------------------------
Each passage carries an equal share of the coolant mass flow, entering at the hub at ``T_in``.
The passage is a smooth circular duct:

* heat transfer: Dittus-Boelter, Nu = 0.023 Re^0.8 Pr^0.4 (fully developed turbulent flow, fluid
  being heated), properties at the local bulk temperature;
* friction: Blasius, f = 0.316 Re^-0.25, for the pressure drop;
* energy balance: the bulk temperature rises along the passage with the heat picked up from the
  wall, m cp dT_c/dz = q'(z). Because q' depends on the metal temperature, the coolant and the
  conduction problem are coupled; ``vanefea.analysis`` iterates between them.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

R_AIR = 287.05
CP_COOLANT = 1100.0       # air at ~850 K [J/(kg K)]
PR_COOLANT = 0.70


def sutherland(T):
    return 1.458e-6 * T ** 1.5 / (T + 110.4)


@dataclass
class ExternalBC:
    """Hot-gas boundary-condition table from the aero-thermal CFD."""
    x: np.ndarray
    y: np.ndarray
    p: np.ndarray
    T_aw: np.ndarray
    h: np.ndarray
    meta: dict

    @classmethod
    def from_dir(cls, d: Path, h_column: str | None = None):
        d = Path(d)
        tab = np.genfromtxt(d / "htc.csv", delimiter=",", names=True)
        meta = json.loads((d / "summary.json").read_text())
        if (d / "scalars.json").exists():
            meta["cfd_scalars"] = json.loads((d / "scalars.json").read_text())
        h_column = h_column or f"h_Tw{int(meta['wall_T'][0])}"
        return cls(tab["x"], tab["y"], tab["p"], tab["T_aw"], tab[h_column], meta)

    def at(self, xyz: np.ndarray, span: float, radial_peak: float = 0.0):
        """(p, T_aw, h) at face centroids; ``radial_peak`` is the mid-span T_aw excess over the mean."""
        _, j = cKDTree(np.c_[self.x, self.y]).query(xyz[:, :2])
        T_aw = self.T_aw[j].copy()
        if radial_peak:
            # parabolic profile with zero span average: +radial_peak at mid-span, -2 radial_peak at the end walls
            eta = xyz[:, 2] / span
            T_aw += radial_peak * 2 * (6 * eta * (1 - eta) - 1.0)
        return self.p[j], T_aw, self.h[j]

    @classmethod
    def from_arrays(cls, x, y, p, T_aw, h, mdot_per_m: float):
        """Table handed over in memory, e.g. by the CFD component of a coupled pipeline."""
        a = [np.asarray(v, dtype=float) for v in (x, y, p, T_aw, h)]
        return cls(*a, meta=dict(cfd_scalars=dict(mdot_per_m=float(mdot_per_m))))

    @property
    def mdot_per_m(self) -> float:
        return float(self.meta.get("cfd_scalars", {}).get("mdot_per_m", np.nan))


@dataclass
class Coolant:
    m_total: float            # coolant mass flow of the vane [kg/s]
    T_in: float = 800.0       # coolant temperature at the passage inlet [K]
    p_in: float = 10.5e5      # coolant pressure [Pa]

    def split(self, d: np.ndarray) -> np.ndarray:
        """Mass flow of each passage when all are fed from one plenum and exhaust to one sink.

        Equal friction pressure drop with the Blasius law, dp ~ m^1.75 d^-4.75, gives m_i ~ d_i^(19/7).
        """
        w = np.asarray(d) ** (19 / 7)
        return self.m_total * w / w.sum()

    def passage(self, m: float, d: float, T_bulk: float | np.ndarray):
        """Reynolds number, heat transfer coefficient (smooth duct), velocity and Mach number of one passage."""
        mu = sutherland(T_bulk)
        k = mu * CP_COOLANT / PR_COOLANT
        Re = 4 * m / (np.pi * d * mu)
        Nu = 0.023 * Re ** 0.8 * PR_COOLANT ** 0.4
        rho = self.p_in / (R_AIR * T_bulk)
        V = m / (rho * np.pi * d ** 2 / 4)
        M = V / np.sqrt(1.4 * R_AIR * T_bulk)
        return dict(m=m, Re=Re, h=Nu * k / d, V=V, Mach=M, rho=rho)

    def pressure_drop(self, m: float, d: float, length: float, T_bulk: float, friction_factor: float = 1.0) -> float:
        """Friction pressure drop; ``friction_factor`` multiplies the smooth-duct value (ribs raise it)."""
        c = self.passage(m, d, T_bulk)
        f = friction_factor * 0.316 * c["Re"] ** -0.25
        return float(f * length / d * 0.5 * c["rho"] * c["V"] ** 2)

    def march(self, m: float, z_edges: np.ndarray, q_line: np.ndarray) -> np.ndarray:
        """Bulk temperature at the bin centres of one passage of mass flow m given the heat per bin q_line [W]."""
        T_edges = self.T_in + np.r_[0, np.cumsum(q_line)] / (m * CP_COOLANT)
        return 0.5 * (T_edges[1:] + T_edges[:-1])
