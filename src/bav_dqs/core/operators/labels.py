"""Pauli labels with an explicit, framework-independent convention.

Qiskit labels are little-endian: the LEFTMOST character acts on the HIGHEST qubit index. The BAV-DQS
pipeline needs the opposite guarantee: element j of every observable vector is the lattice site j, i.e.
qubit j of the circuit (site 0 = left edge, site N-1 = right edge, source at ``dirac_geometry.source_index``).
Building labels through this helper keeps that mapping in one place.
"""
from __future__ import annotations
from typing import Iterable


def z_label(n_qubits: int, qubits: Iterable[int]) -> str:
    """Little-endian Pauli label with Z on the given qubit indices (qubit j = site j) and I elsewhere."""
    n = int(n_qubits)
    chars = ["I"] * n
    for q in qubits:
        q = int(q)
        if not 0 <= q < n:
            raise ValueError(f"qubit index {q} out of range for {n} qubits")
        chars[n - 1 - q] = "Z"
    return "".join(chars)
