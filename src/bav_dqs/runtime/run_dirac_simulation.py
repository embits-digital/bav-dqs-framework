from __future__ import annotations
import argparse
import logging
from pathlib import Path
from typing import Optional
import datetime as _date
import numpy as np

from bav_dqs.io.data_manager import DataManager
from bav_dqs.plugins.dirac_simulation import load_dirac_simulation_yaml, parse_detector_cfg, parse_model_cfg, parse_richardson_cfg, parse_widths, run_boundary_detection, parse_backend_cfg

def _setup_logger(enabled: bool) -> Optional[logging.Logger]:
    if not enabled:
        return None
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(name)s: %(message)s",
    )
    return logging.getLogger("bav_dqs.plugins.dirac_simulation")

def _align_half_to_full(occ_half_raw: np.ndarray, first_hit_half_raw: Optional[int]):
    occ_half_aligned = occ_half_raw[0::2]

    if first_hit_half_raw is None:
        return occ_half_aligned, None

    fh = int(first_hit_half_raw)
    if fh < 0:
        return occ_half_aligned, None
    if (fh % 2) != 0:
        return occ_half_aligned, None

    return occ_half_aligned, fh // 2

def _compute_n_safe_full_grid(
    *,
    n_steps_full: int,
    first_hit_full: Optional[int],
    first_hit_half_aligned: Optional[int],
) -> int:
    candidates = []
    for fh in (first_hit_full, first_hit_half_aligned):
        if fh is None:
            continue
        try:
            k = int(fh)
        except Exception:
            continue
        if k >= 0:
            candidates.append(k)

    if not candidates:
        return int(n_steps_full)

    kmin = max(0, min(candidates))
    return int(min(kmin, n_steps_full))

def _richardson_extrapolate_aligned(
    occ_full_prefix: np.ndarray,
    occ_half_aligned_prefix: np.ndarray,
    order_p: int,
) -> np.ndarray:
    p = int(order_p)
    if p < 1:
        raise ValueError("richardson.order_p must be >= 1")
    factor = 2.0 ** p
    denom = factor - 1.0
    return (factor * occ_half_aligned_prefix - occ_full_prefix) / denom
        
def main() -> None:
    parser = argparse.ArgumentParser(description="Run Dirac Simulation pipeline")
    parser.add_argument("--config", type=Path, default=Path("configs/dirac_simulation.yaml"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    cfg = load_dirac_simulation_yaml(args.config)
    widths = parse_widths(cfg)
    backend_cfg = parse_backend_cfg(cfg)
    detector_cfg = parse_detector_cfg(cfg)
    m, w, dt_full, max_steps_full = parse_model_cfg(cfg)
    rich_cfg = parse_richardson_cfg(cfg)

    logger = _setup_logger(bool(backend_cfg.get("logging_enabled", False)))

    if logger is not None:
        dt_half = dt_full / 2.0
        logger.info("Starting Simulation 101 execution")
        logger.info("Physics: m=%.6g w=%.6g", m, w)
        logger.info(
            "Time grids: dt=%.6g (steps=%d), dt/2=%.6g (steps=%d)",
            dt_full,
            max_steps_full,
            dt_half,
            max_steps_full * 2,
        )
        logger.info(
            "Boundary detector: threshold=%.6g window=%d edge_persistence=%d",
            detector_cfg["threshold"],
            detector_cfg["edge_window"],
            detector_cfg["edge_persistence"],
        )
        logger.info("Lattice sweep: widths=%s", widths)
        logger.info("Backend mode: %s", backend_cfg["mode"])

    exp_id = str(cfg["experiment"]["id"])
    exp_desc = str(cfg["experiment"]["description"])
    schema_version = str(cfg["experiment"]["schema_version"])

    # Setup do Gerenciador de Dados
    args.results_dir.mkdir(parents=True, exist_ok=True)
    stamp = _date.datetime.now(_date.timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = args.results_dir / f"{exp_id}_{stamp}.h5"
    manager = DataManager(
        file_path=out_path,
        config=str(args.config),
        schema_version=schema_version,
        experiment_id=exp_id,
        description=exp_desc
    )

    for width in widths:
        _run_and_save_width(width, m, w, dt_full, max_steps_full, detector_cfg, backend_cfg, rich_cfg, manager, logger, cfg, stamp)

    if logger:
        logger.info("Simulation pipeline complete. HDF5: %s", manager.file_path)

def _run_and_save_width(n_qubits, m, w, dt_full, max_steps_full, detector_cfg, backend_cfg, rich_cfg, manager, logger, cfg, stamp):
    """Executa as simulações e salva os resultados para uma largura específica."""
    if logger: logger.info("Running lattice N=%d", n_qubits)

    # 1. Simulação FULL
    res_full = run_boundary_detection(
        n_qubits=n_qubits, model_cfg={"m": m, "w": w, "dt": dt_full},
        detector_cfg=dict(detector_cfg), backend_cfg=dict(backend_cfg),
        max_steps=max_steps_full, logger=logger
    )
    _log_sim_status(logger, res_full, dt_full, "FULL")

    # 2. Simulação HALF
    dt_half = dt_full / 2.0
    res_half = run_boundary_detection(
        n_qubits=n_qubits, model_cfg={"m": m, "w": w, "dt": dt_half},
        detector_cfg=dict(detector_cfg), backend_cfg=dict(backend_cfg),
        max_steps=max_steps_full * 2, logger=logger
    )
    _log_sim_status(logger, res_half, dt_half, "HALF")

    # 3. Alinhamento e Richardson
    occ_full = np.asarray(res_full.history, dtype=float)
    occ_half_raw = np.asarray(res_half.history, dtype=float)
    occ_half_aligned, fh_half_aligned = _align_half_to_full(np.asarray(res_half.history, dtype=float), res_half.first_hit_step)
    n_safe = _compute_n_safe_full_grid(
            n_steps_full=occ_full.shape[0],
            first_hit_full=res_full.first_hit_step,
            first_hit_half_aligned=fh_half_aligned
        )
    rich_enabled = bool(rich_cfg.get("enabled", False))
    occ_rich, m_rich_l, m_rich_r = _apply_richardson_if_enabled(
        rich_cfg, detector_cfg, occ_full, occ_half_aligned, n_safe
    )

    writer = manager.get_writer()

    datasets = {
        "occ_full": occ_full,
        "occ_half": occ_half_raw,
        "occ_rich": occ_rich,
        "metric_full_left": np.asarray(res_full.d_left, dtype=float),
        "metric_full_right": np.asarray(res_full.d_right, dtype=float),
        "metric_half_left": np.asarray(res_half.d_left, dtype=float),
        "metric_half_right": np.asarray(res_half.d_right, dtype=float),
        "metric_rich_left": m_rich_l,
        "metric_rich_right": m_rich_r,
    }

    datasets = {k: v for k, v in datasets.items() if v is not None}

    attributes = {
        "m": float(m),
        "w": float(w),
        "n_qubits": int(n_qubits),
        "dt_full": float(dt_full),
        "dt_half": float(dt_half),
        "max_steps_full": int(max_steps_full), 
        "first_hit_full": res_full.first_hit_step if res_full.first_hit_step is not None else -1,
        "first_hit_half": res_half.first_hit_step if res_half.first_hit_step is not None else -1,
        "threshold": float(detector_cfg["threshold"]),
        "backend_mode": str(backend_cfg["mode"]),
        "richardson_enabled": bool(rich_enabled),
    }

    for k, v in cfg.get("experiment", {}).items():
        if isinstance(v, (list, dict)) or v is None:
            attributes[f"meta_{k}"] = str(v)
        else:
            attributes[f"meta_{k}"] = v

    writer.save_run(
        group_name="dirac_simulation",
        run_id=f"n{n_qubits}_{stamp}",
        datasets=datasets,
        attributes=attributes
    )


    if logger is not None:
        logger.info("Simulation pipeline complete. HDF5: %s", manager.file_path)

def _apply_richardson_if_enabled(rich_cfg, detector_cfg, occ_full, occ_half_aligned, n_safe):
    """Encapsula a lógica de Richardson para limpar o fluxo principal."""
    if not bool(rich_cfg.get("enabled", False)):
        return None, None, None
        
    occ_rich = _richardson_extrapolate_aligned(occ_full[:n_safe], occ_half_aligned[:n_safe], order_p=int(rich_cfg.get("order_p", 1)))
    w_edge = int(detector_cfg["edge_window"])
    m_rich_l = np.abs(np.mean(occ_full[:n_safe, :w_edge], axis=1) - np.mean(occ_half_aligned[:n_safe, :w_edge], axis=1))
    m_rich_r = np.abs(np.mean(occ_full[:n_safe, -w_edge:], axis=1) - np.mean(occ_half_aligned[:n_safe, -w_edge:], axis=1))
    return occ_rich, m_rich_l, m_rich_r

def _log_sim_status(logger, res, dt, label):
    if not logger: return
    fh = res.first_hit_step if res.first_hit_step is not None else "None"
    status = f"completed. First hit at step: {fh}" if res.first_hit_step is not None else "completed. No hits were detected."
    logger.info(f"Simulation {label} (dt={dt:.6g}) {status}")

if __name__ == "__main__":
    main()