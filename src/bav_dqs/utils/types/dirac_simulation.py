from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np

@dataclass(frozen=True)
class DiracSimulationModelCfg:
    m: float
    w: float
    dt: float

@dataclass(frozen=True)
class DiracSimulationResult:
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