import numpy as np
import pytest

from bav_dqs.core.models.dirac_scheme import INTEGRATOR_ORDER, SCHEME
from bav_dqs.utils.helpers.resolution import check_richardson_order
from bav_dqs.utils.runtime import verify_trotter_order as vto

M, W, N = 0.5, 0.5, 6


def _err(occ, exact):
    return float(np.max(np.abs(occ - exact)))


def _order(scheme, t=1.0, dts=(0.02, 0.01)):
    ex = vto.exact_occupations(N, M, W, t)
    e = [_err(vto.occupations(scheme, N, M, W, dt, t), ex) for dt in dts]
    return float(np.log2(e[0] / e[1]))


def test_reduction_matches_full_space_gate_sequence():
    """The 1-excitation reduction reproduces the 2^N simulation of the gate sequence."""
    assert vto.check_full_space(4, M, W, 0.02) < 1e-12


def test_step_circuit_is_first_order():
    """build_step_circuit is a Lie-Trotter formula: occupation error scales as dt^1."""
    assert _order("lie") == pytest.approx(1.0, abs=0.1)


def test_strang_symmetrization_is_second_order():
    assert _order("strang") == pytest.approx(2.0, abs=0.1)


def test_richardson_order_must_match_the_scheme():
    """Lie-Trotter needs p=1 (result becomes O(dt^2)); p=2 only removes ~2/3 of a first-order error."""
    t = 1.0
    ex = vto.exact_occupations(N, M, W, t)
    full = vto.occupations("lie", N, M, W, 0.02, t)
    half = vto.occupations("lie", N, M, W, 0.01, t)
    e_full = _err(full, ex)
    e_p1 = _err(vto.richardson(full, half, 1), ex)
    e_p2 = _err(vto.richardson(full, half, 2), ex)
    assert e_p1 < 0.05 * e_full      # second-order residual
    assert e_p2 > 0.2 * e_full       # still first order

    sfull = vto.occupations("strang", N, M, W, 0.02, t)
    shalf = vto.occupations("strang", N, M, W, 0.01, t)
    assert _err(vto.richardson(sfull, shalf, 2), ex) < 1e-6  # p=2 is correct for a 2nd-order scheme


def test_declared_integrator_order_matches_the_measured_order():
    """INTEGRATOR_ORDER (used to validate richardson.order_p) is tied to the measured convergence order."""
    assert SCHEME == "lie"
    assert _order(SCHEME) == pytest.approx(INTEGRATOR_ORDER, abs=0.1)


def test_inconsistent_richardson_order_is_reported():
    assert check_richardson_order(INTEGRATOR_ORDER, INTEGRATOR_ORDER) is None
    assert "does not match" in check_richardson_order(2, INTEGRATOR_ORDER)


def test_replica_matches_the_real_qiskit_circuit():
    """Ties the NumPy replica to bav_dqs.core.models.dirac_circuits (skipped without qiskit)."""
    pytest.importorskip("qiskit")
    from qiskit.quantum_info import Statevector
    from bav_dqs.core.models.dirac_circuits import build_initial_circuit, build_step_circuit
    from bav_dqs.utils.types.dirac_simulation import DiracSimulationModelCfg

    dt, steps = 0.02, 7
    sv = Statevector(build_initial_circuit(N))
    step = build_step_circuit(N, DiracSimulationModelCfg(m=M, w=W, dt=dt))
    for _ in range(steps):
        sv = sv.evolve(step)
    p_qiskit = np.array([sv.probabilities([i])[1] for i in range(N)])
    p_replica = vto.occupations("lie", N, M, W, dt, steps * dt)
    assert np.max(np.abs(p_qiskit - p_replica)) < 1e-10
