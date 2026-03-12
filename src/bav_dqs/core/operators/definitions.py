from __future__ import annotations
from typing import List, Dict

def get_dirac_observables(n_qubits: int) -> Dict[str, List[str]]:
    """Define os operadores para Massa (Z) e Causalidade (ZZ)."""
    
    # Operadores de Ocupação (Z_i)
    z_ops = []
    for i in range(n_qubits):
        chars = ["I"] * n_qubits
        chars[i] = "Z"
        z_ops.append("".join(chars))
        
    # Operadores de Correlação (Z_ref Z_j)
    ref = n_qubits // 2
    zz_ops = []
    for j in range(n_qubits):
        chars = ["I"] * n_qubits
        chars[ref] = "Z"
        if j != ref:
            chars[j] = "Z"
        zz_ops.append("".join(chars))
        
    return {"occupancy": z_ops, "correlation": zz_ops}