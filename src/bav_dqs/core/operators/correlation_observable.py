from __future__ import annotations
from typing import List
from qiskit.quantum_info import SparsePauliOp

from bav_dqs.core.operators.labels import z_label


def build_correlation_observables(n_qubits: int, reference_qubit: int) -> List[SparsePauliOp]:
    """
    ZZ correlation observables <Z_ref Z_j> between a reference qubit (e.g., the source)
    and all other qubits j != ref. Element order follows the site index j (ref skipped).

    These are the RAW two-point values; subtract <Z_ref><Z_j> (available from the occupation
    channel) to obtain the connected correlation.
    """
    n, ref = int(n_qubits), int(reference_qubit)
    return [
        SparsePauliOp.from_list([(z_label(n, [ref, j]), 1.0)])
        for j in range(n) if j != ref
    ]
