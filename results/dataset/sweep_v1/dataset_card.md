# Data set `sweep_v1`: cooled LS-89 vane, thermal / static / modal FEA

* **Cases**: 110 (90 train, 20 test; split fixed in the plan), 0 missing.
* **Version**: SHA-256 of `dataset.h5` = `7855c58fc0bc6a16b615224877fbfc8458100e7b45c93b35ff51bd13aaff88fb`. Cite it when reporting a model trained on these data.
* **Generator**: `python -m vanefea.campaign plan|run|collect --name sweep_v1` (scrambled Sobol plan, infeasible
  layouts rejected before meshing; every case independent and reproducible from its row of `plan.json`).

## Contents of `dataset.h5`

| Dataset | Shape | Meaning |
|---|---|---|
| `inputs` | (N, 5) | design variables, columns in the attribute `columns` |
| `outputs` | (N, 15) | quantities of interest, columns in the attribute `columns` |
| `split` | (N,) | `train` / `test` |
| `case_id` | (N,) | case directory name |
| `surface_cloud` | (N, 4096, 5) | random sample of surface nodes: x, y, z [m], temperature [K], von Mises [Pa] (NaN padding) |

`scalars.csv` holds the same inputs and outputs as plain text.

## Columns

| Name | Role | Unit | Range in the data | Meaning |
|---|---|---|---|---|
| `n` | input | - | 4 - 9 | number of radial cooling passages |
| `ratio` | input | - | 0.3 - 0.5 | passage diameter / local section thickness (diameters clipped to 0.6-3.0 mm) |
| `f_start` | input | - | 0.06 - 0.2 | camber-line position of the first passage (0 = leading edge, 1 = trailing edge) |
| `f_end` | input | - | 0.65 - 0.9 | camber-line position of the last passage |
| `coolant` | input | - | 0.015 - 0.045 | coolant mass flow / main-stream mass flow through one vane passage |
| `T_max` | output | K | 1545 - 1583 | peak metal temperature (in practice at the trailing edge) |
| `T_mean` | output | K | 1105 - 1264 | volume-averaged metal temperature |
| `T_min` | output | K | 899.5 - 1083 | lowest metal temperature |
| `vm_max_core` | output | Pa | 6.208e+08 - 1.099e+09 | peak von Mises stress, 10-90 % span (away from the idealised supports) |
| `vm_p99_core` | output | Pa | 4.981e+08 - 8.299e+08 | 99th percentile of the nodal von Mises stress, 10-90 % span |
| `U_max` | output | m | 0.001108 - 0.001321 | largest displacement |
| `f1` | output | Hz | 3318 - 3467 | first natural frequency (hub clamped, hot Young's modulus) |
| `f2` | output | Hz | 4630 - 4779 | second natural frequency |
| `f3` | output | Hz | 7870 - 7984 | third natural frequency |
| `T_coolant_out` | output | K | 868.3 - 958.5 | mass-averaged coolant temperature at the passage exits |
| `Q_gas_W` | output | W | 1747 - 2894 | heat entering the vane from the hot gas |
| `coolant_Mach` | output | - | 0.1023 - 0.6787 | largest passage-inlet Mach number |
| `coolant_dp` | output | Pa | 1.997e+04 - 5.477e+05 | largest passage friction pressure drop |
| `mass` | output | kg | 0.2264 - 0.2408 | vane mass |
| `energy_balance_rel` | output | - | -0.006524 - -0.001836 | (heat from gas - heat to coolant) / heat from gas: a solution check |

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
