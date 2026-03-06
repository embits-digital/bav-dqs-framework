from __future__ import annotations
from typing import Any, Dict, List
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector


def build_z_observables(n_qubits: int) -> List[SparsePauliOp]:
    """Cria uma lista de observáveis Z para cada qubit (mecanismo de medida local)."""
    n = int(n_qubits)
    obs: List[SparsePauliOp] = []
    for j in range(n):
        # Gera strings Pauli tipo 'IIZII'
        pauli = ("I" * j) + "Z" + ("I" * (n - j - 1))
        obs.append(SparsePauliOp.from_list([(pauli, 1.0)]))
    return obs