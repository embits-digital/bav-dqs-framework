from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import yaml

from bav_dqs.core.detectors.boundary_detector import (
    BoundaryDetector,
    BoundaryDetectorCfg
)
from bav_dqs.core.engines.base import BaseEngine
from bav_dqs.core.engines.qiskit_engine import QiskitEngine
from bav_dqs.core.models.dirac_circuits import build_initial_circuit, build_step_circuit
from bav_dqs.core.operators.correlation_observable import build_correlation_observables
from bav_dqs.core.operators.z_observable import build_z_observables
from bav_dqs.utils.helpers.config_manager import ConfigManager
from bav_dqs.utils.types.dirac_simulation import DiracSimulationModelCfg, DiracSimulationResult

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

def parse_validity_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    validity = cfg.get("validity", {})
    return {
        "p_min": int(validity.get("p_min", 32)),
        "stricted": bool(validity.get("stricted", False)),
        "reference_qubit": validity.get("reference_qubit", "center"),
        "observation_mode": validity.get("observation_mode", ["occupation", "correlation"])
    }

def parse_analysis_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    analysis = cfg.get("analysis", {})
    return {
        "fft_window": analysis.get("fft_window", "hann"),
        "figure_dpi": int(analysis.get("figure_dpi", 300))
    }

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
        "auto_threshold" : bool(ConfigManager.require_path_key(cfg, "lattice.auto_threshold"))
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
    d_hit_eff = det._get_d_hit_eff(side, d_l_eff, d_r_eff)
    v_est = (d_hit_eff / t_hit) if t_hit and t_hit > 0 else None
    h_causal = res.get("causal_hits", [None, None])
    first_causal_hit = min([h for h in h_causal if h is not None], default=None)

    return DiracSimulationResult(
        history=np.array(res["history"]),
        correlation=np.array(res["correlation"]),
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

def run_boundary_detection(
    *,
    n_qubits: int,
    model_cfg: Dict[str, Any],
    detector_cfg: Dict[str, Any],
    backend_cfg: Dict[str, Any],
    validity_cfg: Dict[str, Any],
    max_steps: int,
    logger=None,
) -> DiracSimulationResult:
    """Orquestrador principal agnóstico a framework (BAV-DQS)."""
    n, T = int(n_qubits), int(max_steps)
    _validate_inputs(n, T)
    
    # 1. Setup de Física e Detecção
    m, w, dt = _get_model_params(model_cfg)
    det, thr_yaml, edge_persistence = _setup_detector(detector_cfg, n)
    
    # 2. Inicialização do Engine (Injeção de Dependência)
    # No futuro, uma Factory pode escolher entre QiskitEngine, PennyLaneEngine, etc.
    engine: BaseEngine = QiskitEngine(backend_cfg)
    
    # 3. Definição Abstrata de Circuitos (Gerados pelo Model)
    model_params = DiracSimulationModelCfg(m=m, w=w, dt=dt)
    init_def = build_initial_circuit(n)
    step_def = build_step_circuit(n, model_params)
    
    # 4. Definição de Operadores (Strings de Pauli)
    obs_map = {}
    observation_modes = validity_cfg.get("observation_mode", ["occupancy"])
    ref_val = validity_cfg.get("reference_qubit", "center")
    if ref_val == "center":
        ref_idx = n_qubits // 2
    else:
        ref_idx = int(ref_val)
    
    if "occupancy" in observation_modes:
        obs_map["occupancy"] = build_z_observables(n)
    if "correlation" in observation_modes:
        obs_map["correlation"] = build_correlation_observables(n, reference_qubit=ref_idx)

    # 5. Execução do Loop de Simulação
    history_data = _run_simulation_loop(
        n=n, T=T, engine=engine, 
        init_def=init_def, step_def=step_def,
        obs_map=obs_map, det=det,
        backend_cfg=backend_cfg,
        validity_cfg=validity_cfg,
        detector_cfg=detector_cfg,
        logger=logger
    )

    if history_data is None:
        return None

    return _build_result(
        n, dt, m, w, det, detector_cfg, 
        engine.mode, engine.metadata, history_data
    )

def _run_simulation_loop(
    n: int, T: int, engine: BaseEngine, 
    init_def: Any, step_def: Any,
    obs_map: Dict[str, List[str]], 
    det: BoundaryDetector,
    backend_cfg: Dict[str, Any],
    validity_cfg: Dict[str, Any],
    detector_cfg: Dict[str, Any],
    logger: Any
) -> Optional[Dict[str, Any]]:
    auto_thr = detector_cfg.get("auto_threshold", False)
    thr_fixed = float(detector_cfg.get("threshold", 0.01))
    edge_persistence = detector_cfg.get("edge_persistence", 3)
    p_min=validity_cfg["p_min"]
    stricted=validity_cfg["stricted"]
    z_ops = obs_map.get("occupancy", [])
    corr_ops = obs_map.get("correlation", [])
    all_ops = z_ops + corr_ops
    split_idx = len(z_ops)
    ctx = {}
    
    res = {k: [] for k in ["history", "correlation", "dL", "dR", "hits", "counts"]}
    res["hits"] = [None, None]
    res["counts"] = [0, 0]
    log_every = int(backend_cfg.get("log_every_steps", 0))
    warmup_steps = p_min
    calibration_buffer = []
    final_thr = thr_fixed
    flag = True
    for step_idx in range(T + 1):
        # A. Engine e Física
        raw_evs = engine.compute_step(step_idx, n, init_def, step_def, all_ops, ctx)
        occ = (1.0 - raw_evs[:split_idx]) / 2.0
        corr = raw_evs[:split_idx]
        dl, dr = det.update(occ, step=step_idx)
        max_signal = max(dl, dr)

        res["detector_threshold"] = final_thr
        res["history"].append(occ)
        res["correlation"].append(corr)
        res["dL"].append(dl)
        res["dR"].append(dr)
    

        if step_idx < warmup_steps:
            calibration_buffer.append(max_signal)

        elif step_idx == warmup_steps and auto_thr:
            mu = np.mean(calibration_buffer)
            sigma = np.std(calibration_buffer)
            final_thr = mu + 5 * sigma
            if final_thr <= thr_fixed:
                final_thr = thr_fixed
            if logger: 
                logger.info(f"Auto-Threshold Calibrated: {final_thr} (5-sigma) at step {step_idx}")

        if step_idx >= warmup_steps:
            det._update_hit_logic(res, dl, dr, final_thr, edge_persistence, step_idx)

        # C. Registro de dados
       
        
        # D. Lógica de Early Exit (Validação de Fronteira)
        if det.results.first_hit_step is not None:
            status = _handle_collision(det.results.first_hit_step, p_min, stricted, flag, logger)
            flag = False
            if status == "ABORT": return None
            if status == "STOP":  break

        if logger and log_every > 0 and (step_idx % log_every == 0 or step_idx == T):
            logger.info(f"Step {step_idx}/{T} | dL={dl:.4f} dR={dr:.4f}")

    # E. Finalização
    final_res = _finalize_results(res, det)
    return final_res

def _handle_collision(hit_step: int, p_min: int, stricted: bool, flag: bool, logger: Any) -> str:
    """Centraliza a lógica de decisão sobre colisões detectadas."""
    if (int(hit_step) < int(p_min)) and flag:
        if logger:
            logger.warning(f"early colision at t={hit_step} (p_min={p_min}).")
        if stricted:
            if logger: logger.error("[STRICT] Abort:insuficient data.")
            return "ABORT"
        return "CONTINUE"

    if stricted:
        if logger: logger.info(f"Boundary confirmed at t={hit_step}. Ended.")
        return "STOP"
    
    return "CONTINUE"

def _finalize_results(res, det):
    final_res = {k: np.array(v) for k, v in res.items()}
    final_res["hits"] = [det.results.step_left, det.results.step_right]
    return final_res

def _get_model_params(model_cfg: Dict[str, Any]) -> Tuple[float, float, float]:
    m = float(ConfigManager.require_key(model_cfg, "m"))
    w = float(ConfigManager.require_key(model_cfg, "w"))
    dt = float(ConfigManager.require_key(model_cfg, "dt"))
    return m, w, dt

def _setup_detector(detector_cfg: Dict[str, Any], n: int) -> Tuple[BoundaryDetector, float, int]:
    thr = float(ConfigManager.require_key(detector_cfg, "threshold"))
    edge_window = int(ConfigManager.require_key(detector_cfg, "edge_window"))
    persistence = int(ConfigManager.require_key(detector_cfg, "edge_persistence"))
    det = BoundaryDetector(
        BoundaryDetectorCfg(threshold=thr, edge_window=edge_window, edge_persistence=persistence),
        vector_size=n
    )
    return det, thr, persistence

def _validate_inputs(n: int, T: int) -> None:
    if n < 2: raise ValueError("n_qubits must be >= 2")
    if T < 1: raise ValueError("max_steps must be >= 1")
