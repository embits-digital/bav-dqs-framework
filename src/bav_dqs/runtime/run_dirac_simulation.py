from __future__ import annotations
import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Optional
import datetime as _date
import numpy as np

from bav_dqs.io.data_manager import DataManager
from bav_dqs.utils.config_manager import ConfigManager
from bav_dqs.core.engines.qiskit import make_estimator, estimate_evs
from bav_dqs.core.operators.z_observable import build_z_observables
from bav_dqs.plugins.dirac_simulation import build_initial_circuit, load_simulation_101_yaml, parse_detector_cfg, parse_model_cfg, parse_richardson_cfg, parse_widths, run_boundary_detection, run_dd, parse_backend_cfg

def _setup_logger(enabled: bool) -> Optional[logging.Logger]:
    if not enabled:
        return None
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(name)s: %(message)s",
    )
    return logging.getLogger("dirac_qed_qc.simulation_101")

def _align_half_to_full(occ_half_raw: np.ndarray, first_hit_half_raw: Optional[int]):
    # Downsample half-grid (dt/2) to full-grid (dt) by taking even indices
    occ_half_aligned = occ_half_raw[0::2]

    if first_hit_half_raw is None:
        return occ_half_aligned, None

    fh = int(first_hit_half_raw)
    if fh < 0:
        return occ_half_aligned, None
    if (fh % 2) != 0:
        # If the hit happens on an odd half-step, it does not map cleanly to the full grid.
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
    parser = argparse.ArgumentParser(description="Run Dirac-QED-QC Simulation 101 pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("dirac_qed_qc/config/simulation_101.yaml"),
        help="Path to Simulation 101 YAML config",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory where the HDF5 output will be written",
    )
    args = parser.parse_args()

    cfg = load_simulation_101_yaml(args.config)

    widths = parse_widths(cfg)
    backend_cfg = parse_backend_cfg(cfg)
    detector_cfg = parse_detector_cfg(cfg)
    m, w, dt_full, max_steps_full = parse_model_cfg(cfg)
    rich_cfg = parse_richardson_cfg(cfg)

    logger = _setup_logger(bool(backend_cfg.get("logging_enabled", False)))

    logger.info("Wrote M = %s", str(m))

    logger.info("Wrote W = %s", str(w))

    logger.info("Wrote DT_FULL = %s", str(dt_full))

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
            "Boundary detector: threshold=%.6g window=%d persistence=%d",
            detector_cfg["threshold"],
            detector_cfg["edge_window"],
            detector_cfg["edge_persistence"],
        )
        logger.info("Lattice sweep: widths=%s", widths)
        logger.info("Backend mode: %s", backend_cfg["mode"])

    exp_id = str(cfg["experiment"]["id"])
    schema_version = str(cfg["experiment"]["schema_version"])

    args.results_dir.mkdir(parents=True, exist_ok=True)
    stamp = _date.datetime.now(_date.timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = args.results_dir / f"{exp_id}_{stamp}.h5"

    manager = DataManager(
        file_path=out_path,
        config_path=args.config,
        schema_version=schema_version,
        experiment_id=exp_id,
    )

    for width in widths:
        n_qubits = int(width)

        if logger is not None:
            logger.info("Running lattice N=%d", n_qubits)

        # FULL (dt)
        res_full = run_boundary_detection(
            n_qubits=n_qubits,
            model_cfg={"m": m, "w": w, "dt": dt_full},
            detector_cfg=dict(detector_cfg),
            backend_cfg=dict(backend_cfg),
            max_steps=max_steps_full,
            logger=logger,
        )
        occ_full = np.asarray(res_full.history, dtype=float)
        first_hit_full = res_full.first_hit_step
        metric_full_left = np.asarray(res_full.d_left, dtype=float)
        metric_full_right = np.asarray(res_full.d_right, dtype=float)

        # HALF RAW (dt/2) — persisted fully
        dt_half = dt_full / 2.0
        max_steps_half_raw = max_steps_full * 2

        res_half = run_dd(
            n_qubits=n_qubits,
            model_cfg={"m": m, "w": w, "dt": dt_half},
            detector_cfg=dict(detector_cfg),
            backend_cfg=dict(backend_cfg),
            max_steps=max_steps_half_raw,
            logger=logger,
        )
        occ_half_raw = np.asarray(res_half.history, dtype=float)
        first_hit_half_raw = res_half.first_hit_step
        metric_half_left_raw = np.asarray(res_half.d_left, dtype=float)
        metric_half_right_raw = np.asarray(res_half.d_right, dtype=float)

        # ALIGNED HALF (dt grid) — for markers/optional derived datasets
        occ_half_aligned, first_hit_half_aligned = _align_half_to_full(
            occ_half_raw, first_hit_half_raw
        )
        metric_half_left_aligned = metric_half_left_raw[0::2]
        metric_half_right_aligned = metric_half_right_raw[0::2]

        n_steps_full = int(occ_full.shape[0])
        n_steps_half_raw = int(occ_half_raw.shape[0])
        n_steps_aligned = int(min(occ_half_aligned.shape[0], n_steps_full))

        # Marker on FULL grid (not used to truncate persistence)
        n_safe = _compute_n_safe_full_grid(
            n_steps_full=n_steps_full,
            first_hit_full=first_hit_full,
            first_hit_half_aligned=first_hit_half_aligned,
        )

        # Time grids (persist full and raw-half completely)
        t_full = np.arange(n_steps_full, dtype=float) * float(dt_full)
        t_half_raw = np.arange(n_steps_half_raw, dtype=float) * float(dt_half)
        # Aligned time grid (dt)
        t_half_aligned = t_half_raw[0::2]

        # Baselines (t=0, from FULL)
        w_edge = int(detector_cfg["edge_window"])
        baseline_left = float(np.mean(occ_full[0, :w_edge]))
        baseline_right = float(np.mean(occ_full[0, n_qubits - w_edge :]))

        # Optional Richardson: compute on aligned common prefix only, but do NOT truncate persistence
        occ_rich = None
        metric_rich_left = None
        metric_rich_right = None

        rich_enabled = bool(rich_cfg.get("enabled", False))
        rich_order_p = int(rich_cfg.get("order_p", 0) or 0)

        if rich_enabled:
            # Use common aligned prefix length (n_steps_aligned) for extrapolation
            occ_rich = _richardson_extrapolate_aligned(
                occ_full[:n_steps_aligned, :],
                occ_half_aligned[:n_steps_aligned, :],
                order_p=rich_order_p,
            )
            # Diagnostics for rich (aligned grid)
            left_full = np.mean(occ_full[:n_steps_aligned, :w_edge], axis=1)
            right_full = np.mean(occ_full[:n_steps_aligned, n_qubits - w_edge :], axis=1)
            left_half = np.mean(occ_half_aligned[:n_steps_aligned, :w_edge], axis=1)
            right_half = np.mean(occ_half_aligned[:n_steps_aligned, n_qubits - w_edge :], axis=1)
            metric_rich_left = np.abs(left_full - left_half)
            metric_rich_right = np.abs(right_full - right_half)

        meta: Dict[str, Any] = {
            "dt_full": float(dt_full),
            "dt_half": float(dt_half),

            "n_steps_full": int(n_steps_full),
            "n_steps_half_raw": int(n_steps_half_raw),
            "n_steps_aligned": int(n_steps_aligned),

            # Marker (FULL grid)
            "n_safe": int(n_safe),

            "first_hit_full": first_hit_full,
            "first_hit_half_raw": first_hit_half_raw,
            "first_hit_half_aligned": first_hit_half_aligned,

            "half_hit_on_full_grid": (first_hit_half_raw is not None) and (int(first_hit_half_raw) % 2 == 0),

            "threshold": float(detector_cfg["threshold"]),
            "edge_window": int(detector_cfg["edge_window"]),
            "edge_persistence": int(detector_cfg["edge_persistence"]),

            "backend_mode": str(backend_cfg["mode"]),

            # Persisted HALF is RAW (dt/2)
            "dt_half_is_aligned": False,

            "baseline_left_t0": baseline_left,
            "baseline_right_t0": baseline_right,

            "richardson_enabled": rich_enabled,
            "richardson_order_p": rich_order_p,
        }

        # Persist ALL steps (full + half raw). Provide aligned + rich as optional derived artifacts.
        run_id = manager.writer.save_simulation_101(
            n_qubits=n_qubits,
            run_id=None,  # ou um str; None é aceito e o Writer gera um run_id
            t_full=t_full,
            occ_full=occ_full,
            metric_full_left=metric_full_left,
            metric_full_right=metric_full_right,
            t_half=t_half_raw,
            occ_half=occ_half_raw,
            metric_half_left=metric_half_left_raw,
            metric_half_right=metric_half_right_raw,
            metadata=meta,
            occ_rich=occ_rich,
            metric_rich_left=metric_rich_left,
            metric_rich_right=metric_rich_right,
        )

    if logger is not None:
        logger.info("Wrote HDF5 to %s", str(manager.get_file_path()))

if __name__ == "__main__":
    main()