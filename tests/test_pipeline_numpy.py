"""Framework-free tests of the BAV-DQS pipeline (no Qiskit required).

They exercise the methodology through the NumPy connector, the same code path used with any other
framework, plus the cross-engine equivalence with Qiskit when it is installed.
"""
import subprocess
import sys

import numpy as np
import pytest
from unittest.mock import MagicMock

from bav_dqs.core.connectors import available_frameworks, get_connector, get_model_builder
from bav_dqs.core.detectors.boundary_detector import BoundaryDetector, BoundaryDetectorCfg
from bav_dqs.core.engines.numpy_engine import NumpyEngine
from bav_dqs.core.models.dirac_geometry import excluded_sites, max_edge_window, source_index
from bav_dqs.core.operators.labels import z_label
from bav_dqs.utils.helpers.resolution import (
    align_half_to_full, compute_n_safe_full_grid, richardson_extrapolate, safe_window,
)
from bav_dqs.utils.plugins.dirac_simulation import (
    _build_result, _run_simulation_loop, run_boundary_detection,
)

MODEL = {"m": 0.5, "w": 0.5, "dt": 0.02}
NUMPY_BACKEND = {"mode": "ideal", "framework": "numpy"}


def _run(n, thr, window=1, persistence=5, steps=400, framework_cfg=None, **validity):
    return run_boundary_detection(
        n_qubits=n,
        model_cfg=MODEL,
        detector_cfg={"threshold": thr, "edge_window": window, "edge_persistence": persistence},
        backend_cfg=framework_cfg or NUMPY_BACKEND,
        validity_cfg={"p_min": 32, "stricted": False, **validity},
        max_steps=steps,
    )


# --------------------------------------------------------------------------- architecture
def test_plugin_import_does_not_pull_in_any_quantum_framework():
    code = (
        "import sys; import bav_dqs.utils.plugins.dirac_simulation; "
        "bad = [m for m in sys.modules if m.split('.')[0] in ('qiskit', 'qiskit_aer')]; "
        "sys.exit(1 if bad else 0)"
    )
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def test_registry_lists_frameworks_and_rejects_unknown():
    assert {"qiskit", "numpy"} <= set(available_frameworks())
    with pytest.raises(ValueError, match="Unknown framework"):
        get_connector("does-not-exist")
    with pytest.raises(ValueError, match="No model builder"):
        get_model_builder("dirac", "does-not-exist")


def test_numpy_engine_only_supports_ideal_mode():
    with pytest.raises(ValueError, match="only supports mode='ideal'"):
        NumpyEngine({"mode": "aer"})


# --------------------------------------------------------------------------- use-case geometry
@pytest.mark.parametrize("n, src, limit", [(4, 2, 1), (6, 3, 2), (8, 4, 3), (12, 6, 5), (20, 10, 9)])
def test_dirac_geometry(n, src, limit):
    assert source_index(n) == src
    assert max_edge_window(n) == limit == BoundaryDetector.max_valid_edge_window(n, excluded_sites(n))


def test_build_result_geometry_uses_innermost_window_site():
    """Max-based marker: effective distance runs from the source to the innermost window site."""
    n, W = 8, 3  # source = 4; left window sites 0-2 (inner = 2), right window sites 5-7 (inner = 5)
    det = BoundaryDetector(BoundaryDetectorCfg(0.1, W, 5), vector_size=n)
    det.results.step_left = 10
    det.results.first_hit_step = 10
    det.results.first_side = "left"
    res = {"hits": [10, None], "history": np.zeros((2, n)), "correlation": np.zeros((2, n)),
           "dL": [0.0], "dR": [0.0]}
    out = _build_result(n, 0.02, 0.5, 0.5, det, {"edge_window": W, "edge_persistence": 5}, "ideal", {}, res)
    assert out.source_index == 4
    assert (out.x_left_eff, out.x_right_eff) == (2.0, 5.0)
    assert (out.d_left_eff, out.d_right_eff, out.d_hit_eff) == (2.0, 1.0, 2.0)
    assert out.v_hit_est == pytest.approx(2.0 / (10 * 0.02))


def test_build_result_geometry_w1_is_the_edge_site():
    n = 8
    det = BoundaryDetector(BoundaryDetectorCfg(0.1, 1, 5), vector_size=n)
    res = {"hits": [None, None], "history": np.zeros((1, n)), "correlation": np.zeros((1, n)),
           "dL": [0.0], "dR": [0.0]}
    out = _build_result(n, 0.02, 0.5, 0.5, det, {"edge_window": 1, "edge_persistence": 5}, "ideal", {}, res)
    assert (out.x_left_eff, out.x_right_eff) == (0.0, 7.0)


def test_run_boundary_detection_rejects_window_that_contains_the_source():
    with pytest.raises(ValueError, match="contain excluded site"):
        _run(4, 0.1, window=3, steps=5)


# --------------------------------------------------------------------------- pipeline behaviour
def test_auto_threshold_is_applied_to_the_detector():
    """auto_threshold must change the threshold actually used by the detector (single source of truth)."""
    n, p_min, thr_fixed = 4, 4, 1e-6
    det = BoundaryDetector(BoundaryDetectorCfg(thr_fixed, 1, 3), vector_size=n)
    engine = MagicMock()
    # occupancy of site 0 drifts by 0.001 per step -> edge deviation 0.001 * k
    engine.compute_step.side_effect = lambda step_idx, *a, **k: np.array([-0.002 * step_idx, 0.0, 0.0, 0.0])
    _run_simulation_loop(
        n=n, T=6, engine=engine, init_def=None, step_def=None,
        obs_map={"occupancy": [None] * n}, det=det, logger=None,
        backend_cfg={"mode": "ideal"},
        validity_cfg={"stricted": False, "p_min": p_min},
        detector_cfg={"auto_threshold": True, "threshold": thr_fixed, "edge_persistence": 3},
    )
    warm = 0.001 * np.arange(p_min)
    expected = warm.mean() + 5 * warm.std()
    assert expected > thr_fixed
    assert det.cfg.threshold == pytest.approx(expected)


def test_correlation_channel_is_zz_not_a_copy_of_z():
    """At t=0 the source site holds the excitation (Z=-1): <Z_ref Z_j> = -1 for every j != ref."""
    res = _run(4, 0.1, steps=5, observation_mode=["occupation", "correlation"])  # legacy spelling accepted
    assert res.history.shape == (6, 4)
    assert res.correlation.shape == (6, 3)
    np.testing.assert_allclose(res.correlation[0], -1.0)


@pytest.mark.parametrize("thr, n, expected", [
    # Published Tables I (theta=0.10) and II (theta=0.02): edge_window=1, m=w=0.5, dt=0.02, 400 steps
    (0.10, 4, 16), (0.10, 6, 46), (0.10, 8, 162), (0.10, 12, 213), (0.10, 16, 348), (0.10, 20, None),
    (0.02, 4, 7), (0.02, 6, 27), (0.02, 8, 52), (0.02, 12, 110), (0.02, 16, 235), (0.02, 20, 286),
])
def test_published_tables_are_reproduced_through_the_numpy_connector(thr, n, expected):
    """Regression of the published p_hit values with the framework-free connector (first-order circuit)."""
    assert _run(n, thr).first_hit_step == expected


def test_qiskit_and_numpy_connectors_agree():
    """Same methodology, two frameworks: every channel must match, site by site (index j == qubit j)."""
    pytest.importorskip("qiskit")
    modes = ["occupancy", "correlation"]
    a = _run(6, 0.1, steps=30, framework_cfg={"mode": "ideal", "framework": "qiskit"}, observation_mode=modes)
    b = _run(6, 0.1, steps=30, framework_cfg={"mode": "ideal", "framework": "numpy"}, observation_mode=modes)
    np.testing.assert_allclose(a.history, b.history, atol=1e-9)
    np.testing.assert_allclose(a.correlation, b.correlation, atol=1e-9)
    assert a.history[0].argmax() == b.history[0].argmax() == source_index(6)
    assert a.first_hit_step == b.first_hit_step
    assert a.first_hit_side == b.first_hit_side
    assert a.d_hit_eff == b.d_hit_eff


# --------------------------------------------------------------------------- labels (site j == qubit j)
def test_z_label_is_little_endian_with_site_equal_to_qubit():
    assert z_label(4, [0]) == "IIIZ"
    assert z_label(4, [3]) == "ZIII"
    assert z_label(4, [2, 0]) == "IZIZ"
    with pytest.raises(ValueError, match="out of range"):
        z_label(4, [4])


# --------------------------------------------------------------------------- Δt / Δt/2 resolution handling
def test_refined_grid_hit_is_mapped_with_floor_and_never_discarded():
    occ_half = np.zeros((9, 3))
    _, a = align_half_to_full(occ_half, 181)   # odd index (previously dropped)
    _, b = align_half_to_full(occ_half, 180)
    assert (a, b) == (90, 90)
    assert align_half_to_full(occ_half, None)[1] is None
    assert align_half_to_full(occ_half, -1)[1] is None
    assert align_half_to_full(occ_half, 0)[1] == 0      # index 0 is a valid hit
    assert align_half_to_full(occ_half, 4)[0].shape[0] == 5   # even steps live on the base grid


def test_n_safe_is_the_earliest_hit_across_resolutions():
    assert compute_n_safe_full_grid(n_steps_full=401, first_hit_full=162, first_hit_half_aligned=90) == 90
    assert compute_n_safe_full_grid(n_steps_full=401, first_hit_full=50, first_hit_half_aligned=90) == 50
    assert compute_n_safe_full_grid(n_steps_full=401, first_hit_full=None, first_hit_half_aligned=None) == 401


def test_safe_window_compares_hits_in_physical_time():
    sw = safe_window(162, 181, 0.02)            # base hit at 3.24, refined hit at 181 * 0.01 = 1.81
    assert sw.t_full == pytest.approx(3.24) and sw.t_half == pytest.approx(1.81)
    assert sw.limiting == "dt/2" and sw.p_safe_base == 90
    assert safe_window(90, 180, 0.02).limiting == "both"          # same physical instant
    assert safe_window(-1, -1, 0.02).t_safe is None               # sentinel: no hit
    assert safe_window(0, -1, 0.02).t_safe == 0.0                 # index 0 is a hit, not 'no hit'


def test_richardson_combines_dt_and_dt_over_2_with_the_integrator_order():
    x, c = 1.0, 0.3
    full, half = np.array([x + c * 0.02]), np.array([x + c * 0.01])   # first-order error: e(h) = c*h
    assert richardson_extrapolate(full, half, 1)[0] == pytest.approx(x)       # p = 1 removes it
    assert richardson_extrapolate(full, half, 2)[0] != pytest.approx(x)       # p = 2 does not
    with pytest.raises(ValueError):
        richardson_extrapolate(full, half, 0)


def test_refined_grid_odd_hit_flows_through_the_pipeline():
    """Real series on both grids: an odd refined-grid hit index must still bound the safe prefix."""
    full = _run(8, 0.10, steps=400)
    half = run_boundary_detection(
        n_qubits=8, model_cfg={"m": 0.5, "w": 0.5, "dt": 0.01},
        detector_cfg={"threshold": 0.10, "edge_window": 1, "edge_persistence": 5},
        backend_cfg=NUMPY_BACKEND, validity_cfg={"p_min": 32, "stricted": False}, max_steps=800,
    )
    assert half.first_hit_step % 2 == 1
    _, fh = align_half_to_full(half.history, half.first_hit_step)
    n_safe = compute_n_safe_full_grid(n_steps_full=full.history.shape[0],
                                      first_hit_full=full.first_hit_step, first_hit_half_aligned=fh)
    assert fh == half.first_hit_step // 2 and n_safe == min(full.first_hit_step, fh)
