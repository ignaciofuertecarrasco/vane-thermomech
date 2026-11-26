import shutil

import numpy as np
import pytest

from vanefea import ccx
from vanefea.loads import Coolant


def test_face_index_matches_corner_triples():
    tets = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    idx = ccx.face_index(tets)
    assert idx[(0, 1, 2)] == (1, 1)
    assert idx[(0, 1, 3)] == (1, 2)
    assert idx[(1, 2, 3)] == (1, 3)
    assert idx[(0, 2, 3)] == (1, 4)


def test_von_mises_of_uniaxial_and_shear_states():
    assert np.isclose(ccx.von_mises(np.array([[100.0, 0, 0, 0, 0, 0]]))[0], 100.0)
    assert np.isclose(ccx.von_mises(np.array([[0, 0, 0, 50.0, 0, 0]]))[0], 50.0 * np.sqrt(3))


def test_frd_parser(tmp_path):
    frd = tmp_path / "x.frd"
    frd.write_text("\n".join([
        "    1PSTEP                         1           1           1",
        "  100CL  101 1.000000000           2                     0    1           1",
        " -4  NDTEMP      1    1",
        " -5  T           1    1    0    0",
        " -1         1 1.00000E+03",
        " -1         2 1.10000E+03",
        " -3",
    ]))
    b = ccx.last_block(ccx.read_frd(frd), "NDTEMP")
    assert b.nodes.tolist() == [1, 2]
    assert np.allclose(b.full(3)[:2, 0], [1000.0, 1100.0]) and np.isnan(b.full(3)[2, 0])


def test_coolant_energy_balance():
    c = Coolant(m_total=0.015, T_in=800.0)
    m = c.split(np.array([3e-3, 3e-3, 1e-3]))
    assert np.isclose(m.sum(), 0.015) and m[0] > m[2]           # bigger passages take more flow
    z = np.linspace(0, 0.05, 11)
    T = c.march(m[0], z, np.full(10, 10.0))                      # 100 W along one passage
    assert np.isclose(T[-1] - 800.0, 95.0 / (m[0] * 1100.0), rtol=1e-6)
    assert 0.05 < c.passage(m[0], 3e-3, 800.0)["Mach"] < 0.5


@pytest.mark.calculix
@pytest.mark.skipif(shutil.which("ccx") is None, reason="CalculiX not installed")
def test_cantilever_frequency_against_beam_theory(tmp_path):
    from vanefea.verification import run_beam
    r = run_beam(tmp_path, 5e-3)
    assert abs(r["rel_error"][0]) < 0.01
