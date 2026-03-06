from __future__ import annotations
from typing import List
from qiskit.quantum_info import SparsePauliOp


def build_z_observables(n_qubits: int) -> List[SparsePauliOp]:
    """Creates a list of Z observables for each qubit (local measurement mechanism)."""
    n = int(n_qubits)
    obs: List[SparsePauliOp] = []
    for j in range(n):
        # Generates Pauli strings like 'IIZII'
        pauli = ("I" * j) + "Z" + ("I" * (n - j - 1))
        obs.append(SparsePauliOp.from_list([(pauli, 1.0)]))
    return obs