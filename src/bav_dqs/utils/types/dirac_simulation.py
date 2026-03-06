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
    """It encapsulates all the data and metrics resulting from a Dirac simulation.

    This structure stores everything from the raw evolution of probabilities 
    to metrics derived from edge detection and velocity estimates.

    Attributes:
        history: Matrix (T+1, N) with the occupancy (probability) of each qubit at each step.
        first_hit_step: First temporal step where any edge was hit.
        d_left: Vector of the left edge detection metric over time.
        d_right: Vector of the right edge detection metric over time.
        first_hit_step_left: Step of the first confirmed detection on the left.
        first_hit_step_right: Step of the first confirmed detection on the right.
        first_hit_side: Side of the first impact ('left', 'right', 'both' or None).
        dt: Time step used in the simulation.
        t_hit: Physical time of the impact (first_hit_step * dt).
        source_index: Initial position (center) of the excitation in the qubit chain.
        x_left_eff: Effective coordinate of the left detector.
        x_right_eff: Effective coordinate of the right detector.
        d_left_eff: Effective distance from the center to the left detector.
        d_right_eff: Effective distance from the center to the right detector.
        d_hit_eff: Distance traveled to the edge that triggered the first hit.
        v_hit_est: Estimated propagation speed (d_hit_eff / t_hit).
        n_qubits: Number of qubits in the simulated chain.
        m: Mass used in the model.
        w: Coupling used in the model.
        detector_threshold: Probability threshold used for detection.
        detector_edge_window: Size of the qubit window monitored at each edge.
        detector_persistence: Number of steps required to confirm a detection.
        backend_mode: Execution mode ('ideal' or 'estimator').
        backend_meta: Additional backend metadata (e.g., shots, chip name).
    """
    history: np.ndarray
    first_hit_step: Optional[int]
    d_left: np.ndarray
    d_right: np.ndarray

    first_hit_step_left: Optional[int] = None
    first_hit_step_right: Optional[int] = None
    first_hit_side: Optional[str] = None  # 'left' | 'right' | 'both' | None

    dt: Optional[float] = None
    t_hit: Optional[float] = None
    source_index: Optional[float] = None
    x_left_eff: Optional[float] = None
    x_right_eff: Optional[float] = None
    d_left_eff: Optional[float] = None
    d_right_eff: Optional[float] = None
    d_hit_eff: Optional[float] = None
    v_hit_est: Optional[float] = None

    n_qubits: Optional[int] = None
    m: Optional[float] = None
    w: Optional[float] = None
    detector_threshold: Optional[float] = None
    detector_edge_window: Optional[int] = None
    detector_persistence: Optional[int] = None
    backend_mode: Optional[str] = None
    backend_meta: Optional[Dict[str, Any]] = None