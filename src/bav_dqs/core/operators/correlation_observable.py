from __future__ import annotations
from typing import List
from qiskit.quantum_info import SparsePauliOp

def build_correlation_observables(n_qubits: int, reference_qubit: int) -> List[SparsePauliOp]:
    """
    Cria observáveis de correlação ZZ entre um qubit de referência (ex: centro)
    e todos os outros: <Z_ref Z_j> - <Z_ref><Z_j>
    """
    obs = []
    for j in range(n_qubits):
        if j == reference_qubit:
            continue 
        # String para Z_ref Z_j
        chars = ["I"] * n_qubits
        chars[reference_qubit] = "Z"
        chars[j] = "Z"
        pauli = "".join(chars)
        obs.append(SparsePauliOp.from_list([(pauli, 1.0)]))
    return obs
