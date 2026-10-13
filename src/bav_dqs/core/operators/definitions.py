from __future__ import annotations
from typing import List, Dict

from bav_dqs.core.models.dirac_geometry import source_index
from bav_dqs.core.operators.labels import z_label


def get_dirac_observables(n_qubits: int) -> Dict[str, List[str]]:
    """Operator labels for Mass (Z) and Causality (ZZ); element j refers to qubit j (= site j).

    The correlation list has one entry per site: Z_ref Z_j for j != ref and Z_ref alone for j == ref.
    """
    n = int(n_qubits)
    z_ops = [z_label(n, [i]) for i in range(n)]
    ref = source_index(n)
    zz_ops = [z_label(n, [ref] if j == ref else [ref, j]) for j in range(n)]
    return {"occupancy": z_ops, "correlation": zz_ops}
