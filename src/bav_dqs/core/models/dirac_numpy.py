"""NumPy reference model of the Dirac lattice (framework-free).

The circuit built by ``dirac_circuits.build_step_circuit`` conserves the number of excitations and the
initial state has a single excitation, so the dynamics lives in an N-dimensional subspace. This module
provides that exact reduction:

    H = m * sum_q (-1)^q Z_q  +  w * sum_q (X_q X_{q+1} + Y_q Y_{q+1})

    step_lie    : the same product formula as the Qiskit circuit (mass layer, then bonds q=0..N-2) - first order
    step_strang : palindromic version of the same gates - second order
    exact_propagator : exp(-i H t) by eigendecomposition (reference)

It is used (i) by the NumPy connector (see ``core/connectors.py``), which runs the full BAV-DQS pipeline
without any quantum framework, and (ii) by ``verify_trotter_order`` to verify convergence orders.
"""
from __future__ import annotations
from typing import Any, Callable, Dict

import numpy as np

from bav_dqs.core.models.dirac_geometry import source_index


def hamiltonian_1exc(n: int, m: float, w: float) -> np.ndarray:
    s = np.array([1.0 if q % 2 == 0 else -1.0 for q in range(n)])
    H = np.diag(m * (s.sum() - 2.0 * s)).astype(complex)  # Z_q = +1 for q != i, -1 for q == i
    for q in range(n - 1):
        H[q, q + 1] = H[q + 1, q] = 2.0 * w  # (XX + YY)|01> = 2|10>
    return H


def exact_propagator(H: np.ndarray, t: float) -> np.ndarray:
    ev, V = np.linalg.eigh(H)
    return (V * np.exp(-1j * ev * t)) @ V.conj().T


def _mass(n: int, m: float, dt: float) -> np.ndarray:
    s = np.array([1.0 if q % 2 == 0 else -1.0 for q in range(n)])
    return np.diag(np.exp(-1j * dt * m * (s.sum() - 2.0 * s)))


def _bond(n: int, q: int, w: float, dt: float) -> np.ndarray:
    B = np.eye(n, dtype=complex)
    c, sn = np.cos(2.0 * w * dt), -1j * np.sin(2.0 * w * dt)
    B[q, q] = B[q + 1, q + 1] = c
    B[q, q + 1] = B[q + 1, q] = sn
    return B


def step_lie(n: int, m: float, w: float, dt: float) -> np.ndarray:
    """Exactly what build_step_circuit does (first-order Lie-Trotter)."""
    U = _mass(n, m, dt)
    for q in range(n - 1):
        U = _bond(n, q, w, dt) @ U
    return U


def step_strang(n: int, m: float, w: float, dt: float) -> np.ndarray:
    """Symmetrized (palindromic) version of the same gates: second order."""
    U = _mass(n, m, dt / 2)
    for q in range(n - 2):
        U = _bond(n, q, w, dt / 2) @ U
    U = _bond(n, n - 2, w, dt) @ U
    for q in reversed(range(n - 2)):
        U = _bond(n, q, w, dt / 2) @ U
    return _mass(n, m, dt / 2) @ U


SCHEMES: Dict[str, Callable[[int, float, float, float], np.ndarray]] = {"lie": step_lie, "strang": step_strang}


def evolve(U: np.ndarray, psi0: np.ndarray, steps: int) -> np.ndarray:
    psi = psi0.copy()
    for _ in range(steps):
        psi = U @ psi
    return psi


# ---- builders with the same contract as the Qiskit model (init_def, step_def) -----------------
def build_initial_state(n_qubits: int) -> np.ndarray:
    psi = np.zeros(int(n_qubits), dtype=complex)
    psi[source_index(n_qubits)] = 1.0
    return psi


def build_step_matrix(n_qubits: int, cfg: Any) -> np.ndarray:
    """Step unitary for the model parameters ``cfg`` (needs .m, .w, .dt)."""
    return step_lie(int(n_qubits), float(cfg.m), float(cfg.w), float(cfg.dt))
