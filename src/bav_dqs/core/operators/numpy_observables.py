"""Framework-free observables for the NumPy connector.

In the single-excitation basis |i> (excitation at site i) every Pauli-Z string is diagonal, so an
observable is represented by its diagonal (a real vector). Same contract as the Qiskit builders:
``build_z_observables`` -> one Z_j per site; ``build_correlation_observables`` -> Z_ref Z_j for j != ref.
"""
from __future__ import annotations
from typing import List

import numpy as np


def _z_diag(n: int, j: int) -> np.ndarray:
    d = np.ones(n)
    d[j] = -1.0
    return d


def build_z_observables(n_qubits: int) -> List[np.ndarray]:
    n = int(n_qubits)
    return [_z_diag(n, j) for j in range(n)]


def build_correlation_observables(n_qubits: int, reference_qubit: int) -> List[np.ndarray]:
    n, ref = int(n_qubits), int(reference_qubit)
    return [_z_diag(n, ref) * _z_diag(n, j) for j in range(n) if j != ref]
