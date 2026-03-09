from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import yaml
from qiskit import QuantumCircuit

from bav_dqs.core.detectors.boundary_detector import (
    BoundaryDetector,
    BoundaryDetectorCfg,
    _get_d_hit_eff,
    _update_hit_logic,
)
from bav_dqs.core.engines.qiskit import _get_occupation, make_estimator
from bav_dqs.core.models.dirac_circuits import build_initial_circuit, build_step_circuit
from bav_dqs.core.operators.correlation_observable import build_correlation_observables
from bav_dqs.core.operators.z_observable import build_z_observables
from bav_dqs.utils.config_manager import ConfigManager
from bav_dqs.utils.types.dirac_simulation import DiracSimulationModelCfg, DiracSimulationResult

def build_initial_circuit(n_qubits: int) -> QuantumCircuit:
    """Prepares the initial state: single excitation at the center of the 1D network."""
    n = int(n_qubits)
    if n < 2: raise ValueError("n_qubits >= 2")
    qc = QuantumCircuit(n)
    qc.x(n // 2) 
    return qc

def load_dirac_simulation_yaml(path: Path) -> Dict[str, Any]:
    ConfigManager.require_path(str(path))
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a YAML mapping at the root.")
    _ = ConfigManager.require_path_key(cfg, "experiment.id")
    _ = ConfigManager.require_path_key(cfg, "experiment.schema_version")
    _ = ConfigManager.require_path_key(cfg, "lattice.widths")
    return cfg

def parse_widths(cfg: Dict[str, Any]) -> List[int]:
    widths = list(ConfigManager.require_path_key(cfg, "lattice.widths"))
    if not widths:
        raise ValueError("lattice.widths must be a non-empty list")
    return [int(w) for w in widths]

def parse_backend_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    backend: Dict[str, Any] = dict(ConfigManager.require_path_key(cfg, "backend"))
    mode = str(ConfigManager.require_path_key(cfg, "backend.mode")).strip().lower()

    out: Dict[str, Any] = {"mode": mode}

    if mode == "aer":
        out["precision"] = float(ConfigManager.require_path_key(cfg, "backend.precision"))

    if "shots" in backend:
        out["shots"] = backend["shots"]
    if "optimization_level" in backend:
        out["optimization_level"] = backend["optimization_level"]

    out["abort_on_complex_evs"] = bool(backend.get("abort_on_complex_evs", False))
    out["complex_evs_imag_tol"] = float(backend.get("complex_evs_imag_tol", 0.0))

    log_cfg = backend.get("logging", {})
    if isinstance(log_cfg, dict):
        out["logging_enabled"] = bool(log_cfg.get("enabled", False))
        out["log_every_steps"] = int(log_cfg.get("every_steps", 0))
    else:
        out["logging_enabled"] = False
        out["log_every_steps"] = 0

    return out

def parse_detector_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "threshold": float(ConfigManager.require_path_key(cfg, "lattice.threshold")),
        "edge_window": int(ConfigManager.require_path_key(cfg, "lattice.edge_window")),
        "edge_persistence": int(ConfigManager.require_path_key(cfg, "lattice.edge_persistence")),
    }

def parse_model_cfg(cfg: Dict[str, Any]) -> Tuple[float, float, float, int]:
    m = float(ConfigManager.require_path_key(cfg, "physics.m"))
    w = float(ConfigManager.require_path_key(cfg, "physics.w"))
    dt = float(ConfigManager.require_path_key(cfg, "physics.dt"))
    max_steps = int(ConfigManager.require_path_key(cfg, "physics.max_steps"))
    return m, w, dt, max_steps

def parse_richardson_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    r = cfg.get("richardson", None)
    if not isinstance(r, dict):
        return {"enabled": False}
    enabled = bool(r.get("enabled", False))
    if not enabled:
        return {"enabled": False}
    order_p = int(ConfigManager.require_path_key(cfg, "richardson.order_p"))
    return {"enabled": True, "order_p": order_p}

def dirac_simulation(
    *,
    n_qubits: int,
    m: float,
    w: float,
    dt: float,
    max_steps: int,
    threshold: float,
    edge_window: int,
    edge_persistence: int,
    backend_cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Optional[int]]:
    res = run_boundary_detection(
        n_qubits=int(n_qubits),
        model_cfg={"m": float(m), "w": float(w), "dt": float(dt)},
        detector_cfg={
            "threshold": float(threshold),
            "edge_window": int(edge_window),
            "edge_persistence": int(edge_persistence),
        },
        backend_cfg=dict(backend_cfg),
        max_steps=int(max_steps),
        logger=None,
    )
    return res.history, res.first_hit_step
def _calculate_side(h_left, h_right) -> Optional[str]:
    if h_left is None and h_right is None: return None
    if h_left == h_right: return "both"
    if h_left is None: return "right"
    if h_right is None: return "left"
    return "left" if h_left < h_right else "right"

def _build_result(n, dt, m, w, det: BoundaryDetector, d_cfg, mode, meta, res) -> DiracSimulationResult:
    """
    Constrói o resultado final integrando Massa (Z) e Informação (ZZ).
    Inclui os metadados de 5-sigma para blindagem científica.
    """
    h_left, h_right = res["hits"]
    side = _calculate_side(h_left, h_right)

    # Geometria Efetiva da Rede
    src = n // 2
    x_l = (d_cfg["edge_window"] - 1) / 2.0
    x_r = (n - 1) - (d_cfg["edge_window"] - 1) / 2.0
    d_l_eff, d_r_eff = src - x_l, x_r - src
    
    # Cálculo de t_hit (Físico/Massa)
    t_hit = float(det.results.first_hit_step) * dt if det.results.first_hit_step else None
    d_hit_eff = _get_d_hit_eff(side, d_l_eff, d_r_eff)
    v_est = (d_hit_eff / t_hit) if t_hit and t_hit > 0 else None
    h_causal = res.get("causal_hits", [None, None])
    first_causal_hit = min([h for h in h_causal if h is not None], default=None)

    return DiracSimulationResult(
        history=np.array(res["history"]),
        correlations=np.array(res.get("correlations", [])),
        first_hit_step=det.results.first_hit_step,
        first_hit_step_left=h_left,
        first_hit_step_right=h_right,
        first_hit_side=side,
        d_left=np.array(res["dL"]),
        d_right=np.array(res["dR"]),
        first_causal_hit_step=first_causal_hit,
        dt=dt, t_hit=t_hit, source_index=src,
        x_left_eff=x_l, x_right_eff=x_r, 
        d_left_eff=d_l_eff, d_right_eff=d_r_eff,
        d_hit_eff=d_hit_eff, v_hit_est=v_est, 
        n_qubits=n, m=m, w=w,
        detector_threshold=det.cfg.threshold, 
        detector_edge_window=d_cfg["edge_window"],
        detector_persistence=d_cfg["edge_persistence"],
        backend_mode=mode,
        backend_meta=meta
    )

def _validate_inputs(n: int, T: int) -> None:
    if n < 2:
        raise ValueError("n_qubits must be >= 2.")
    if T < 1:
        raise ValueError("max_steps must be >= 1.")

def _get_model_params(model_cfg: Dict[str, Any]) -> Tuple[float, float, float]:
    m = float(ConfigManager.require_key(model_cfg, "m"))
    w = float(ConfigManager.require_key(model_cfg, "w"))
    dt = float(ConfigManager.require_key(model_cfg, "dt"))
    return m, w, dt

def _setup_detector(detector_cfg: Dict[str, Any], n: int) -> Tuple[BoundaryDetector, float, int]:
    thr = float(ConfigManager.require_key(detector_cfg, "threshold"))
    edge_window = int(ConfigManager.require_key(detector_cfg, "edge_window"))
    edge_persistence = int(ConfigManager.require_key(detector_cfg, "edge_persistence"))
    
    det = BoundaryDetector(
        BoundaryDetectorCfg(threshold=thr, edge_window=edge_window, edge_persistence=edge_persistence),
        vector_size=n,
    )
    return det, thr, edge_persistence

def _calibrate_multi_threshold(estimator, circuit, obs_map: dict) -> dict:
    thresholds = {}
    
    for label, observables in obs_map.items():
        all_values = []
        
        for _ in range(5):
            # No Qiskit 2.3.0, passamos a lista de tuplas (PUBs)
            pubs = [(circuit, obs) for obs in observables]
            job = estimator.run(pubs)
            result = job.result()
            step_values = [pub.data.evs for pub in result]
            all_values.append(step_values)
        
        data = np.array(all_values, dtype=float)
        
        std_dev = np.std(data)
        
        thresholds[label] = max(5.0 * std_dev, 1e-4)
        
        print(f"[BAV-DQS] {label} calibrated at 5-sigma: {thresholds[label]:.5f}")
        
    return thresholds

def run_boundary_detection(
    *,
    n_qubits: int,
    model_cfg: Dict[str, Any],
    detector_cfg: Dict[str, Any],
    backend_cfg: Dict[str, Any],
    max_steps: int,
    diagnostic_modes: List[str] = ["occupancy", "correlation"], 
    logger=None,
) -> DiracSimulationResult:
    """Main orchestrator of the BAV-DQS framework."""
    n, T = int(n_qubits), int(max_steps)
    _validate_inputs(n, T)
    
    m, w, dt = _get_model_params(model_cfg)

    det, thr_yaml, edge_persistence = _setup_detector(detector_cfg, n)
    mode, estimator, backend_meta = make_estimator(dict(backend_cfg))
    
    model_params = DiracSimulationModelCfg(m=m, w=w, dt=dt)
    init_qc = build_initial_circuit(n)
    
    obs_map = {}
    if "occupancy" in diagnostic_modes:
        obs_map["occupancy"] = build_z_observables(n)
    if "correlation" in diagnostic_modes:
        obs_map["correlation"] = build_correlation_observables(n, reference_qubit=n // 2)

    if detector_cfg.get("auto_threshold", False):
        thr_dict = _calibrate_multi_threshold(estimator, init_qc, obs_map)
        thr_main = thr_dict.get("occupancy", thr_yaml)
        det.update_threshold(thr_main)
    else:
        thr_dict = {"occupancy": thr_yaml, "correlation": thr_yaml}

    step_qc = build_step_circuit(n, model_params)
    
    history_data = _run_simulation_loop(
        n=n, T=T, mode=mode, estimator=estimator, 
        step_qc=step_qc, init_qc=init_qc, 
        backend_cfg=backend_cfg, det=det, 
        thr_dict=thr_dict, obs_map=obs_map,
        edge_persistence=edge_persistence, logger=logger
    )

    return _build_result(n, dt, m, w, det, detector_cfg, mode, backend_meta, history_data)

def _run_simulation_loop(
    n: int, T: int, mode: str, estimator, step_qc, init_qc, 
    backend_cfg, det: BoundaryDetector, thr_dict: Dict[str, float], 
    obs_map: Dict[str, List], edge_persistence: int, logger
):
    """Executa a evolução usando _get_occupation para todos os observáveis."""
    log_every = int(backend_cfg.get("log_every_steps", 0))
    state_ctx = {"state": None, "qc": QuantumCircuit(n)}
    
    all_obs = obs_map.get("occupancy", []) + obs_map.get("correlation", [])
    split_idx = len(obs_map.get("occupancy", []))

    res = {
        "history": [], "correlations": [], 
        "dL": [], "dR": [], "hits": [None, None], "counts": [0, 0]
    }

    for step_idx in range(T + 1):
        combined_values = _get_occupation(
            step_idx, mode, estimator, step_qc, init_qc, all_obs, backend_cfg, state_ctx
        )
        
        occ = np.array(combined_values[:split_idx])
        corr = np.array(combined_values[split_idx:])

        dl, dr = det.update(occ, step=step_idx)
        _update_hit_logic(res, dl, dr, thr_dict["occupancy"], edge_persistence, step_idx)
        
        res["history"].append(occ)
        res["correlations"].append(corr)
        res["dL"].append(dl)
        res["dR"].append(dr)
        
        if logger and log_every > 0 and (step_idx % log_every == 0 or step_idx == T):
            logger.info(f"Step {step_idx}/{T} | dL={dl:.6f} dR={dr:.6f}")

    return res
