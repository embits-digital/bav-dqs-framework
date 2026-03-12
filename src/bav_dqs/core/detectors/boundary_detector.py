from __future__ import annotations
from dataclasses import replace
from typing import Optional, Tuple, Dict
import numpy as np

from bav_dqs.utils.types.boundary_detector import BoundaryDetectorCfg, DetectionResult

class BoundaryDetector:
    """
    Implements the operational criterion of inferential gating.
    Agnostic to the physical model: operates on occupancy/density vectors.
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
        if self.cfg.edge_persistence < 1: raise ValueError("persistence must be >= 1")

    def update_threshold(self, new_threshold: float):
        """
        Atualiza o limite de detecção (theta) dinamicamente.
        Essencial para calibração de 5-sigma em hardware real.
        """
        self.cfg = replace(self.cfg, threshold=float(new_threshold))
        print(f"[Detector] Threshold updated to: {self.cfg.threshold:.6f}")

    def _calculate_edge_means(self, data: np.ndarray) -> Tuple[float, float]:
        w = self.cfg.edge_window
        left = float(np.max(np.abs(data[:w]))) # Usar MAX em vez de MEAN para correlações
        right = float(np.max(np.abs(data[-w:])))
        return left, right

    def update(self, data: np.ndarray, step: int) -> Tuple[float, float]:
        """
        Processes a new temporal step and updates the detection state.
        Returns the calculated derivations (delta) for log/analysis.
        """
        data = np.asarray(data, dtype=float)
        if data.shape != (self.size,):
            raise ValueError(f"Data shape mismatch. Expected ({self.size},), got {data.shape}")

        curr_left, curr_right = self._calculate_edge_means(data)

        # Baseline Initialization (First Observation)
        if self._baselines["left"] is None:
            self._baselines["left"] = curr_left
            self._baselines["right"] = curr_right

        # Calculation of derivations (Section IV-B, Eq. 2)
        d_left = abs(curr_left - self._baselines["left"])
        d_right = abs(curr_right - self._baselines["right"])

        # Persistence and Hit Processing
        self._process_side("left", d_left, step)
        self._process_side("right", d_right, step)

        return d_left, d_right

    def _process_side(self, side: str, delta: float, step: int):
        # Persistence Logic (Eq. 3)
        if delta >= self.cfg.threshold:
            self._counters[side] += 1
        else:
            self._counters[side] = 0

        # Hit Verification Confirmed
        if self._counters[side] >= self.cfg.edge_persistence:
            hit_index = step - self.cfg.edge_persistence
            
            # Record a specific hit on the side.
            if side == "left" and self.results.step_left is None:
                self.results.step_left = hit_index
            elif side == "right" and self.results.step_right is None:
                self.results.step_right = hit_index

            # Update Hit Global (the first one to occur)
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

    @staticmethod
    def _update_hit_logic(res, dl, dr, thr, edge_persistence, step_idx):
        """Simplified edge_persistence logic for left (0) and right (1)."""
        for i, val in enumerate([dl, dr]):
            if val >= thr:
                res["counts"][i] += 1
            else:
                res["counts"][i] = 0
                
            if res["hits"][i] is None and res["counts"][i] >= edge_persistence:
                res["hits"][i] = step_idx - edge_persistence

    @staticmethod
    def _get_d_hit_eff(side: Optional[str], d_l_eff: float, d_r_eff: float) -> Optional[float]:
        """It calculates the actual distance traveled based on the detected side."""
        if side is None:
            return None
        if side == "left":
            return d_l_eff
        if side == "right":
            return d_r_eff
        # If 'both', returns the shortest distance (closest impact).
        return min(d_l_eff, d_r_eff)
