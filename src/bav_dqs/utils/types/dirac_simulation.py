from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np

@dataclass(frozen=True)
class DiracSimulationModelCfg:
    """Setting the physical parameters of the discretized Dirac model.

    Attributes:
        m: Mass parameter (rest energy). Defines the dispersion of the wave packet.
        w: Hopping frequency. Defines the propagation speed in the network.
        dt: Temporal discretization step of the simulation.
    """
    m: float
    w: float
    dt: float

@dataclass(frozen=True)
class DiracSimulationResult:
    """Encapsulates mass (Z) and information (ZZ) metrics for DQS validation.

    Attributes:
        history: Matrix (T+1, N) with site occupancy <Z_j>.
        correlations: Matrix (T+1, N-1) with ZZ correlations from center.
        first_hit_step: First step where mass (occupancy) hit an edge.
        first_causal_hit_step: First step where information (correlation) hit an edge.
        detector_threshold: The calibrated 5-sigma floor for occupancy.
        detector_threshold_corr: The calibrated 5-sigma floor for correlations.
    """
    # --- Core Evolution Data ---
    history: np.ndarray
    correlations: Optional[np.ndarray] = None  # Dados ZZ para Lieb-Robinson
    
    # --- Mass Edge Metrics (Physical) ---
    first_hit_step: Optional[int] = None
    d_left: Optional[np.ndarray] = None
    d_right: Optional[np.ndarray] = None
    first_hit_step_left: Optional[int] = None
    first_hit_step_right: Optional[int] = None
    first_hit_side: Optional[str] = None

    # --- Causal Edge Metrics (Information/Lieb-Robinson) ---
    first_causal_hit_step: Optional[int] = None # NOVIDADE 2026
    
    # --- Temporal & Geometric Parameters ---
    dt: Optional[float] = None
    t_hit: Optional[float] = None
    source_index: Optional[float] = None
    x_left_eff: Optional[float] = None
    x_right_eff: Optional[float] = None
    d_left_eff: Optional[float] = None
    d_right_eff: Optional[float] = None
    d_hit_eff: Optional[float] = None
    v_hit_est: Optional[float] = None

    # --- Model & Hardware Context ---
    n_qubits: Optional[int] = None
    m: Optional[float] = None
    w: Optional[float] = None
    
    # --- Admissibility Parameters (The "Shield") ---
    detector_threshold: Optional[float] = None       # Theta_Z (5-sigma)
    detector_threshold_corr: Optional[float] = None  # Theta_ZZ (5-sigma)
    detector_edge_window: Optional[int] = None
    detector_persistence: Optional[int] = None
    
    backend_mode: Optional[str] = None
    backend_meta: Optional[Dict[str, Any]] = None
