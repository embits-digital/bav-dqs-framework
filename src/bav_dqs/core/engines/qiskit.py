from __future__ import annotations
from typing import Any, Dict, List
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from bav_dqs.utils.config_manager import ConfigManager

def make_estimator(backend_cfg: Dict[str, Any]):
    """
    Factory for instantiating the Qiskit Estimator (V2).
    Supports: ideal (Statevector), aer (Standard Simulation), and mps (Matrix Product State).
    """
    mode = str(ConfigManager.require_key(backend_cfg, "mode")).strip().lower()

    if mode == "ideal":
        from qiskit.primitives import StatevectorEstimator
        return mode, StatevectorEstimator(), {}

    if mode == "aer":
        from qiskit_aer.primitives import EstimatorV2
        precision = float(ConfigManager.require_key(backend_cfg, "precision"))
        return mode, EstimatorV2(options={"default_precision": precision}), {"precision": precision}

    if mode == "mps":
        from qiskit_aer import AerSimulator
        from qiskit_aer.primitives import EstimatorV2
        
        precision = float(backend_cfg.get("precision", 0.0) or 0.0)
        backend_options: Dict[str, Any] = {"method": "matrix_product_state"}
        
        if "matrix_product_state_max_bond_dimension" in backend_cfg:
            backend_options["matrix_product_state_max_bond_dimension"] = int(backend_cfg["matrix_product_state_max_bond_dimension"])
        if "matrix_product_state_truncation_threshold" in backend_cfg:
            backend_options["matrix_product_state_truncation_threshold"] = float(backend_cfg["matrix_product_state_truncation_threshold"])

        sim = AerSimulator(**backend_options)
        opts: Dict[str, Any] = {"default_precision": precision} if precision > 0.0 else {}
        
        return mode, EstimatorV2.from_backend(sim, options=opts), {"precision": precision, **backend_options}

    raise ValueError("backend.mode deve ser: 'ideal' | 'aer' | 'mps'.")

def estimate_evs(estimator, circuit: QuantumCircuit, observables: List[SparsePauliOp]) -> np.ndarray:
    """Performs the estimation of expected values ​​synchronously."""
    job = estimator.run([(circuit, observables)])
    result = job.result()
    data = getattr(result[0], "data", None)
    evs = getattr(data, "evs", None) if data is not None else None
    if evs is None:
        raise RuntimeError("The Estimator result does not contain 'data.evs'.")
    return np.asarray(evs, dtype=float)

def ideal_expectation_values_real(
    state: Statevector,
    observables: List[SparsePauliOp],
    *,
    abort_on_complex_evs: bool,
    complex_evs_imag_tol: float,
) -> np.ndarray:
    """Exact calculation via state vector, ignoring/validating imaginary parts."""
    vals: List[float] = []
    for obs in observables:
        ev = state.expectation_value(obs)
        if abs(ev.imag) > complex_evs_imag_tol and abort_on_complex_evs:
            raise RuntimeError(f"Excessive imaginary part: {ev.imag}")
        vals.append(float(ev.real))
    return np.asarray(vals, dtype=float)

def _get_occupation(
    step_idx: int,
    mode: str,
    estimator: Any,
    step_qc: QuantumCircuit,
    init_qc: QuantumCircuit,
    observables: Any,
    backend_cfg: Dict[str, Any],
    ctx: Dict[str, Any]
) -> np.ndarray:
    """It calculates occupancy based on the simulation mode (Ideal or Estimator)."""
    if mode == "ideal":
        # Evolution via Statevector
        if step_idx == 0:
            ctx["state"] = Statevector.from_instruction(init_qc)
        else:
            ctx["state"] = ctx["state"].evolve(step_qc)
            
        evs = ideal_expectation_values_real(
            ctx["state"],
            observables,
            abort_on_complex_evs=bool(backend_cfg.get("abort_on_complex_evs", False)),
            complex_evs_imag_tol=float(backend_cfg.get("complex_evs_imag_tol", 0.0)),
        )
    else:
        # Evolution via Circuit and Estimator
        if step_idx == 0:
            ctx["qc"].compose(init_qc, inplace=True)
        else:
            ctx["qc"].compose(step_qc, inplace=True)
            
        evs = estimate_evs(estimator, ctx["qc"], observables)

    return (1.0 - np.asarray(evs)) / 2.0