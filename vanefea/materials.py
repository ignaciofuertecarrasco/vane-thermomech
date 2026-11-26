"""Temperature-dependent material data (SI units).

``NI_SUPERALLOY`` holds representative values for a cast nickel-base superalloy of the kind used
for nozzle guide vanes, assembled from public datasheets and handbook trends. It is suitable for a
methodology demonstration, **not** certified design data: replace it with the supplier's values
for any real assessment.

The thermal expansion coefficient is the *secant* (mean) value from the reference temperature
``alpha_ref_T``, which is what CalculiX expects with ``*EXPANSION, ZERO=...``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Material:
    name: str
    T: tuple            # temperature grid [K]
    E: tuple            # Young's modulus [Pa]
    nu: float           # Poisson's ratio [-]
    alpha: tuple        # secant expansion coefficient from alpha_ref_T [1/K]
    k: tuple            # thermal conductivity [W/(m K)]
    rho: float          # density [kg/m^3]
    alpha_ref_T: float = 293.15

    def at(self, prop: str, T):
        return np.interp(T, self.T, getattr(self, prop))


NI_SUPERALLOY = Material(
    name="Ni-base superalloy (representative)",
    T=(293.15, 573.15, 873.15, 1073.15, 1273.15),
    E=(205e9, 190e9, 172e9, 158e9, 140e9),
    nu=0.30,
    alpha=(12.8e-6, 13.6e-6, 14.6e-6, 15.6e-6, 16.8e-6),
    k=(10.5, 15.0, 20.0, 23.5, 27.0),
    rho=8200.0,
)

# constant-property material for the analytical verification cases
STEEL_CONST = Material(name="steel (constant properties)", T=(0.0, 3000.0), E=(200e9, 200e9), nu=0.3,
                       alpha=(12e-6, 12e-6), k=(40.0, 40.0), rho=7850.0)
