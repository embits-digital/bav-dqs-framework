from __future__ import annotations
import argparse
import logging
from pathlib import Path
from typing import Optional
import datetime as _date
import warnings
import numpy as np

from bav_dqs.core.models.dirac_geometry import max_edge_window
from bav_dqs.core.models.dirac_scheme import INTEGRATOR_ORDER, SCHEME
from bav_dqs.utils.helpers.provenance import collect_provenance
from bav_dqs.utils.io.data_manager import DataManager
from bav_dqs.utils.plugins.dirac_simulation import load_dirac_simulation_yaml, parse_detector_cfg, parse_model_cfg, parse_richardson_cfg, parse_validity_cfg, parse_widths, run_boundary_detection, parse_backend_cfg

def _setup_logger(enabled: bool) -> Optional[logging.Logger]:
    if not enabled:
        return None
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(name)s: %(message)s",
    )
    return logging.getLogger("[BAV-DQS]")

from bav_dqs.utils.helpers.resolution import (
    align_half_to_full as _align_half_to_full,
    check_richardson_order as _check_richardson_order,
    compute_n_safe_full_grid as _compute_n_safe_full_grid,
    richardson_extrapolate as _richardson_extrapolate_aligned,
)


def richardson_attributes(rich_cfg, integrator_order: int = INTEGRATOR_ORDER) -> dict:
    """HDF5 provenance of the Richardson step, derived from the SAME resolved config used to compute it.

    ``rich_cfg`` must come from ``parse_richardson_cfg`` (order_p validated, no defaults). Conventions:
      - enabled:  richardson_order_p = the p actually used in the extrapolation;
                  richardson_order_consistent = (p == integrator_order).
      - disabled: richardson_order_p = -1 (sentinel: no extrapolation was computed);
                  richardson_order_consistent = True (no inconsistent extrapolation was applied).
    """
    enabled = bool(rich_cfg.get("enabled", False))
    if not enabled:
        return {"richardson_enabled": False, "richardson_order_p": -1, "richardson_order_consistent": True}
    p = int(rich_cfg["order_p"])  # KeyError on an unvalidated config: never fall back to a default
    return {"richardson_enabled": True, "richardson_order_p": p,
            "richardson_order_consistent": bool(p == int(integrator_order))}


def _warn_if_richardson_order_inconsistent(rich_cfg, logger=None) -> Optional[str]:
    """Warns (does not abort) when order_p != INTEGRATOR_ORDER; the mismatch is also recorded in HDF5."""
    if not bool(rich_cfg.get("enabled", False)):
        return None
    msg = _check_richardson_order(int(rich_cfg["order_p"]), INTEGRATOR_ORDER)
    if msg:
        warnings.warn(msg)
        if logger:
            logger.warning(msg)
    return msg


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Dirac Simulation pipeline")
    parser.add_argument("--config", type=Path, default=Path("configs/dirac_simulation.yaml"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    cfg = load_dirac_simulation_yaml(args.config)
    widths = parse_widths(cfg)
    backend_cfg = parse_backend_cfg(cfg)
    validity_cfg = parse_validity_cfg(cfg)
    detector_cfg = parse_detector_cfg(cfg)
    m, w, dt_full, max_steps_full = parse_model_cfg(cfg)
    rich_cfg = parse_richardson_cfg(cfg)  # validated once; order_p is mandatory when enabled

    logger = _setup_logger(bool(backend_cfg.get("logging_enabled", False)))
    _warn_if_richardson_order_inconsistent(rich_cfg, logger)

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
        # Full resolved config embedded (not just a relative path) + config hash and Git commit.
        config=cfg,
        schema_version=schema_version,
        experiment_id=exp_id,
        experiment_desc=exp_desc,
        extra_metadata=collect_provenance(args.config),
    )

    for width in widths:
        _run_and_save_width(width, m, w, dt_full, max_steps_full, detector_cfg, backend_cfg, validity_cfg, rich_cfg, manager, logger, cfg, stamp)
        

    if logger:
        logger.info("Simulation pipeline complete. HDF5: %s", manager.file_path)

def _run_and_save_width(n_qubits, m, w, dt_full, max_steps_full, detector_cfg, backend_cfg, validity_cfg, rich_cfg, manager, logger, cfg, stamp):
    """Executa as simulações e salva os resultados para uma largura específica."""
    # Skip widths whose edge windows would contain the source site (marker would be invalid).
    limit = max_edge_window(int(n_qubits))
    if int(detector_cfg["edge_window"]) > limit:
        msg = (f"Width N={n_qubits} skipped: edge_window={detector_cfg['edge_window']} "
               f"overlaps the source site (max valid edge_window = {limit}).")
        warnings.warn(msg)
        if logger: logger.warning(msg)
        return

    if logger: logger.info("Running lattice N=%d", n_qubits)

    # 1. Simulação FULL
    res_full = run_boundary_detection(
        n_qubits=n_qubits, model_cfg={"m": m, "w": w, "dt": dt_full},
        detector_cfg=dict(detector_cfg), backend_cfg=dict(backend_cfg), validity_cfg=dict(validity_cfg),
        max_steps=max_steps_full, logger=logger
    )

    if res_full is None:
        if logger: logger.error(f"Width N={n_qubits} aborted due to causal isolation violation (FULL).")
        return
    
    _log_sim_status(logger, res_full, dt_full, "FULL")

    # 2. Simulação HALF
    dt_half = dt_full / 2.0
    res_half = run_boundary_detection(
        n_qubits=n_qubits, model_cfg={"m": m, "w": w, "dt": dt_half},
        detector_cfg=dict(detector_cfg), backend_cfg=dict(backend_cfg), validity_cfg=dict(validity_cfg),
        max_steps=max_steps_full * 2, logger=logger
    )

    if res_half is None:
        if logger: logger.error(f"Width N={n_qubits} aborted due to causal isolation violation (HALF)..")
        return
    
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
    occ_rich, m_gap_l, m_gap_r = _apply_richardson_if_enabled(
        rich_cfg, detector_cfg, occ_full, occ_half_aligned, n_safe
    )

    writer = manager.get_writer()
    datasets = {
        "correlation":  np.asarray(res_full.correlation, dtype=float),
        "occ_full": occ_full,
        "occ_half": occ_half_raw,
        "occ_rich": occ_rich,
        "metric_full_left": np.asarray(res_full.d_left, dtype=float),
        "metric_full_right": np.asarray(res_full.d_right, dtype=float),
        "metric_half_left": np.asarray(res_half.d_left, dtype=float),
        "metric_half_right": np.asarray(res_half.d_right, dtype=float),
        # Resolution gap (full vs dt/2) using the window MEAN; NOT the boundary marker.
        "metric_res_gap_left": m_gap_l,
        "metric_res_gap_right": m_gap_r,
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
        "threshold": float(res_full.detector_threshold),
        "threshold_occupancy": float(res_full.detector_threshold),
        "threshold_correlation": float(detector_cfg["threshold"]),
        "backend_mode": str(backend_cfg["mode"]),
        # --- Provenance of detector/Richardson hyperparameters ---
        "edge_marker": "max_abs_deviation_from_t0",
        "framework": str(backend_cfg.get("framework", "qiskit")).lower(),
        "edge_window": int(detector_cfg["edge_window"]),
        "edge_persistence": int(detector_cfg["edge_persistence"]),
        "auto_threshold": bool(detector_cfg.get("auto_threshold", False)),
        "threshold_configured": float(detector_cfg["threshold"]),
        **richardson_attributes(rich_cfg),  # richardson_enabled / _order_p / _order_consistent
        "integrator_scheme": SCHEME,
        "integrator_order": int(INTEGRATOR_ORDER),
        "p_min": int(validity_cfg["p_min"]),
        "stricted": bool(validity_cfg["stricted"]),
        "source_index": int(res_full.source_index),
        "first_hit_side": res_full.first_hit_side if res_full.first_hit_side else "none",
        "d_hit_eff": float(res_full.d_hit_eff) if res_full.d_hit_eff is not None else float("nan"),
        "metric_res_gap_definition": "abs(mean_W(occ_full) - mean_W(occ_half_aligned))",
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

def _apply_richardson_if_enabled(rich_cfg, detector_cfg, occ_full, occ_half_aligned, n_safe):
    """It encapsulates Richardson's logic to clean the main stream."""
    if not bool(rich_cfg.get("enabled", False)):
        return None, None, None
        
    # Same p that richardson_attributes() records in the HDF5 file (validated by parse_richardson_cfg).
    occ_rich = _richardson_extrapolate_aligned(occ_full[:n_safe], occ_half_aligned[:n_safe], order_p=int(rich_cfg["order_p"]))
    w_edge = int(detector_cfg["edge_window"])
    m_gap_l = np.abs(np.mean(occ_full[:n_safe, :w_edge], axis=1) - np.mean(occ_half_aligned[:n_safe, :w_edge], axis=1))
    m_gap_r = np.abs(np.mean(occ_full[:n_safe, -w_edge:], axis=1) - np.mean(occ_half_aligned[:n_safe, -w_edge:], axis=1))
    return occ_rich, m_gap_l, m_gap_r

def _log_sim_status(logger, res, dt, label):
    if not logger: return
    fh = res.first_hit_step if res.first_hit_step is not None else "None"
    status = f"completed. First hit at step: {fh}" if res.first_hit_step is not None else "completed. No hits detected."
    logger.info(f"Simulation {label} (dt={dt:.6g}) {status}")

if __name__ == "__main__":
    main()