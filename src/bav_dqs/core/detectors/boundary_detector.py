from __future__ import annotations
from typing import Optional, Tuple, Dict
import numpy as np

from bav_dqs.utils.types.boundary_detector import BoundaryDetectorCfg, DetectionResult

class BoundaryDetector:
    """
    Implementa o critério operacional de gating inferencial (Seção IV-B).
    Agnóstico ao modelo físico: opera sobre vetores de ocupação/densidade.
    """
    def __init__(self, cfg: BoundaryDetectorCfg, vector_size: int):
        self.cfg = cfg
        self.size = int(vector_size)
        self._validate_config()

        # Estado interno de processamento
        self._baselines: Dict[str, float] = {"left": None, "right": None}
        self._counters: Dict[str, int] = {"left": 0, "right": 0}
        self.results = DetectionResult()

    def _validate_config(self):
        if self.size < 2: raise ValueError("vector_size must be >= 2")
        if not (1 <= self.cfg.edge_window <= self.size):
            raise ValueError(f"Invalid edge_window: {self.cfg.edge_window}")
        if self.cfg.persistence < 1: raise ValueError("persistence must be >= 1")

    def _calculate_edge_means(self, data: np.ndarray) -> Tuple[float, float]:
        w = self.cfg.edge_window
        left = float(np.mean(data[:w]))
        right = float(np.mean(data[-w:]))
        return left, right

    def update(self, data: np.ndarray, step: int) -> Tuple[float, float]:
        """
        Processa um novo passo temporal e atualiza o estado de detecção.
        Retorna as derivações (delta) calculadas para log/análise.
        """
        data = np.asarray(data, dtype=float)
        if data.shape != (self.size,):
            raise ValueError(f"Data shape mismatch. Expected ({self.size},), got {data.shape}")

        curr_left, curr_right = self._calculate_edge_means(data)

        # Inicialização do Baseline (Primeira observação)
        if self._baselines["left"] is None:
            self._baselines["left"] = curr_left
            self._baselines["right"] = curr_right

        # Cálculo das derivações (Seção IV-B, Eq. 2)
        d_left = abs(curr_left - self._baselines["left"])
        d_right = abs(curr_right - self._baselines["right"])

        # Processamento de Persistência e Hits
        self._process_side("left", d_left, step)
        self._process_side("right", d_right, step)

        return d_left, d_right

    def _process_side(self, side: str, delta: float, step: int):
        # Lógica de Persistência (Eq. 3)
        if delta >= self.cfg.threshold:
            self._counters[side] += 1
        else:
            self._counters[side] = 0

        # Verificação de Hit Confirmado
        if self._counters[side] >= self.cfg.persistence:
            hit_index = step - self.cfg.persistence
            
            # Registra hit específico do lado
            if side == "left" and self.results.step_left is None:
                self.results.step_left = hit_index
            elif side == "right" and self.results.step_right is None:
                self.results.step_right = hit_index

            # Atualiza Hit Global (o primeiro que ocorrer)
            if self.results.first_hit_step is None:
                self.results.first_hit_step = hit_index
                self.results.first_side = side
            elif self.results.first_hit_step == hit_index and self.results.first_side != side:
                self.results.first_side = "both"

    def reset(self):
        """Limpa baselines, contadores e resultados para uma nova run."""
        self._baselines = {"left": None, "right": None}
        self._counters = {"left": 0, "right": 0}
        self.results = DetectionResult()

def _update_hit_logic(res, dl, dr, thr, persistence, step_idx):
    # Lógica de persistência simplificada para esquerda (0) e direita (1)
    for i, val in enumerate([dl, dr]):
        if val >= thr:
            res["counts"][i] += 1
        else:
            res["counts"][i] = 0
            
        if res["hits"][i] is None and res["counts"][i] >= persistence:
            res["hits"][i] = step_idx - persistence

def _get_d_hit_eff(side: Optional[str], d_l_eff: float, d_r_eff: float) -> Optional[float]:
    """Calcula a distância efetiva percorrida baseada no lado detectado."""
    if side is None:
        return None
    if side == "left":
        return d_l_eff
    if side == "right":
        return d_r_eff
    # Caso 'both', retorna a menor distância (impacto mais próximo)
    return min(d_l_eff, d_r_eff)
