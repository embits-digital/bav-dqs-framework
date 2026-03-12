from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List
import numpy as np

class BaseEngine(ABC):
    """Required interface for any simulator (Qiskit, PennyLane, NumPy)."""

    @abstractmethod
    def __init__(self, backend_cfg: Dict[str, Any]):
        """Initializes the backend and the accuracy/shot options."""
        pass

    @abstractmethod
    def compute_step(
        self,
        step_idx: int,
        n_qubits: int,
        init_def: Any,
        step_def: Any,
        obs_defs: List[str],
        ctx: Dict[str, Any]
    ) -> np.ndarray:
        """
        Evolves the state and calculates the raw expected values ​​<O>.
        Returns an array with the results of all observables.
        """
        pass

    @property
    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """Returns backend metadata (mode, precision, etc.)."""
        pass
