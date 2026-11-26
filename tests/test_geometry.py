import numpy as np

from vanefea.geometry import CoolingLayout, camber_line, load_profile, local_thickness, section_area


def test_profile_is_the_ls89_section():
    p = load_profile()
    assert np.isclose(np.ptp(p[:, 0]), 36.97e-3, rtol=3e-3)        # axial chord
    assert 5.5e-4 < section_area() < 6.5e-4                         # ~605 mm2


def test_camber_line_runs_from_le_to_te():
    c = camber_line()
    assert c[0, 0] < 1e-3 and c[-1, 0] > 0.035


def test_thickness_distribution():
    t = [local_thickness(f) for f in (0.2, 0.5, 0.9)]
    assert t[0] > t[1] > t[2] > 0
    assert 14e-3 < t[0] < 18e-3


def test_layout_constraints():
    lay = CoolingLayout(n=7, ratio=0.40, f_start=0.10, f_end=0.80)
    assert lay.valid()
    d = lay.diameters()
    assert d[0] >= d[-1] and d.min() >= lay.d_min and d.max() <= lay.d_max   # passages shrink towards the TE
    too_close_to_te = CoolingLayout(n=7, ratio=0.45, f_start=0.10, f_end=0.97)
    assert too_close_to_te.check()["wall_margin"] < 0
    crowded = CoolingLayout(n=12, ratio=0.5, f_start=0.10, f_end=0.30)
    assert crowded.check()["ligament_margin"] < 0
