from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List
import numpy as np

class BaseEngine(ABC):
    """Interface obrigatória para qualquer simulador (Qiskit, PennyLane, NumPy)."""

    @abstractmethod
    def __init__(self, backend_cfg: Dict[str, Any]):
        """Inicializa o backend e as opções de precisão/shots."""
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
        Evolui o estado e calcula os valores esperados brutos <O>.
        Retorna um array com os resultados de todos os observáveis.
        """
        pass

    @property
    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """Retorna metadados do backend (modo, precisão, etc)."""
        pass
