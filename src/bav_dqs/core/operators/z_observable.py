from __future__ import annotations
from typing import List
from qiskit.quantum_info import SparsePauliOp

from bav_dqs.core.operators.labels import z_label


def build_z_observables(n_qubits: int) -> List[SparsePauliOp]:
    """Z observables for each qubit: element j measures qubit j (= lattice site j).

    Qiskit labels are little-endian, so the label is built through ``z_label`` (previously element j
    measured qubit n-1-j, mirroring left/right and the source position in the occupation vector).
    """
    n = int(n_qubits)
    return [SparsePauliOp.from_list([(z_label(n, [j]), 1.0)]) for j in range(n)]
