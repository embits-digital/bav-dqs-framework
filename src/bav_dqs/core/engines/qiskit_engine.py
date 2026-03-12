from __future__ import annotations
from typing import Any, Dict
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from bav_dqs.core.engines.base import BaseEngine

class QiskitEngine(BaseEngine):
    def __init__(self, backend_cfg: Dict[str, Any]):
        self.mode = str(backend_cfg.get("mode", "ideal")).lower()
        self._estimator, self._meta = self._make_estimator(backend_cfg)

    def _make_estimator(self, cfg):
        mode = str(cfg.get("mode", "")).strip().lower()
        
        if mode == "ideal":
            from qiskit.primitives import StatevectorEstimator
            return StatevectorEstimator(), {"mode": mode}

        if mode in ["aer", "mps"]:
            from qiskit_aer.primitives import EstimatorV2
            from qiskit_aer import AerSimulator
            
            precision = float(cfg.get("precision", 0.0))
            options = {"default_precision": precision} if precision > 0.0 else {}
            
            backend_options = {}
            if mode == "mps":
                backend_options["method"] = "matrix_product_state"
                for key in ["matrix_product_state_max_bond_dimension", 
                            "matrix_product_state_truncation_threshold"]:
                    if key in cfg:
                        backend_options[key] = cfg[key]
            
            sim = AerSimulator(**backend_options)
            estimator = EstimatorV2.from_backend(sim, options=options)
            
            return estimator, {"mode": mode, "precision": precision, **backend_options}

        raise ValueError(f"Modo de backend inválido: '{mode}'. Use 'ideal', 'aer' or 'mps'.")


    def compute_step(self, step_idx, n, init_qc, step_qc, obs_defs, ctx) -> np.ndarray:
        ops = []
        for s in obs_defs:
            pauli_str = s[0] if isinstance(s, (list, tuple)) else s
            label = pauli_str[0].paulis.to_labels()[0]
            ops.append(SparsePauliOp.from_list([(label, 1.0)]))

        
        if self.mode == "ideal":
            if step_idx == 0: ctx["state"] = Statevector.from_instruction(init_qc)
            else: ctx["state"] = ctx["state"].evolve(step_qc)
            return np.array([ctx["state"].expectation_value(o).real for o in ops])
        
        if "qc" not in ctx: ctx["qc"] = QuantumCircuit(n)
        ctx["qc"].compose(init_qc if step_idx == 0 else step_qc, inplace=True)
        return np.asarray(self._estimator.run([(ctx["qc"], ops)]).result()[0].data.evs)

    @property
    def metadata(self): return self._meta
