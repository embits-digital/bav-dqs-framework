from __future__ import annotations
from dataclasses import replace
from typing import Iterable, Optional, Tuple, Dict
import numpy as np

from bav_dqs.utils.types.boundary_detector import BoundaryDetectorCfg, DetectionResult

class BoundaryDetector:
    """
    Implements the operational criterion of inferential gating.
    Agnostic to the physical model: operates on occupancy/density vectors.

    Boundary marker (peak-deviation criterion)
    ------------------------------------------
    For each edge window W (the ``edge_window`` outermost sites), the marker is the
    MAXIMUM per-site deviation from the initial (t=0) profile:

        Delta_L(t) = max_{i in W_left}  | n_i(t) - n_i(0) |
        Delta_R(t) = max_{i in W_right} | n_i(t) - n_i(0) |

    A side is flagged when Delta >= threshold for ``edge_persistence`` consecutive steps.

    Design rationale: the maximum is deliberately preferred over the window average.
    Since mean <= max <= W * mean, the max-based marker never triggers later than a
    mean-based one at the same threshold, so the safe window can only shrink (the
    conservative direction for an admissibility gate). It is also insensitive to
    dilution of a localized front by quiet sites inside the window.

    Index convention: when the persistence counter reaches ``edge_persistence`` at
    step ``s``, the reported hit index is ``s - edge_persistence`` (one step before
    the first above-threshold step).

    Geometry note: with a max-based marker, the front is detected when it reaches the
    innermost site of the window (index W-1 on the left, N-W on the right).
    The edge windows must not contain sites where the signal is non-zero at t=0 by construction
    (e.g. the support of the initial excitation); the use case declares those sites and calls
    ``validate_edge_windows``. The detector itself knows nothing about any particular use case.
    """
    def __init__(self, cfg: BoundaryDetectorCfg, vector_size: int):
        self.cfg = cfg
        self.size = int(vector_size)
        self._validate_config()

        # Internal processing state. Baselines are per-site profiles (arrays of size edge_window).
        self._baselines: Dict[str, Optional[np.ndarray]] = {"left": None, "right": None}
        self._counters: Dict[str, int] = {"left": 0, "right": 0}
        self.results = DetectionResult()

    def _validate_config(self):
        if self.size < 2: raise ValueError("vector_size must be >= 2")
        if not (1 <= self.cfg.edge_window <= self.size):
            raise ValueError(f"Invalid edge_window: {self.cfg.edge_window}")
        if self.cfg.edge_persistence < 1: raise ValueError("persistence must be >= 1")

    @staticmethod
    def max_valid_edge_window(n: int, excluded_sites: Iterable[int]) -> int:
        """Largest edge_window whose left/right windows both exclude ``excluded_sites``.

        ``excluded_sites`` is declared by the use case (e.g. the support of the initial excitation).
        Windows that contain such sites would measure the evolution of the excitation itself,
        not the arrival of a front at the edge. With no excluded sites any window is valid.
        """
        sites = sorted({int(s) for s in excluded_sites})
        if not sites:
            return int(n)
        return max(0, min(sites[0], int(n) - 1 - sites[-1]))

    @staticmethod
    def validate_edge_windows(n: int, edge_window: int, excluded_sites: Iterable[int]) -> None:
        """Raises ValueError if an edge window would contain any of ``excluded_sites``."""
        sites = sorted({int(s) for s in excluded_sites})
        limit = BoundaryDetector.max_valid_edge_window(n, sites)
        if edge_window > limit:
            raise ValueError(
                f"edge_window={edge_window} is invalid for vector size {n}: an edge window would "
                f"contain excluded site(s) {sites} (max valid edge_window = {limit})."
            )

    def update_threshold(self, new_threshold: float):
        """
        Updates the detection limit (theta) dynamically.
        Essential for 5-sigma calibration on real hardware.
        """
        self.cfg = replace(self.cfg, threshold=float(new_threshold))
        print(f"[Detector] Threshold updated to: {self.cfg.threshold:.6f}")

    def _edge_windows(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        w = self.cfg.edge_window
        return data[:w], data[-w:]

    def _calculate_edge_deviations(self, data: np.ndarray) -> Tuple[float, float]:
        """Peak deviation from the t=0 profile on each edge (max over the window)."""
        left_win, right_win = self._edge_windows(data)
        if self._baselines["left"] is None:
            self._baselines["left"] = left_win.copy()
            self._baselines["right"] = right_win.copy()
        d_left = float(np.max(np.abs(left_win - self._baselines["left"])))
        d_right = float(np.max(np.abs(right_win - self._baselines["right"])))
        return d_left, d_right

    def update(self, data: np.ndarray, step: int) -> Tuple[float, float]:
        """
        Processes a new temporal step and updates the detection state.
        Returns the calculated derivations (delta) for log/analysis.
        """
        data = np.asarray(data, dtype=float)
        if data.shape != (self.size,):
            raise ValueError(f"Data shape mismatch. Expected ({self.size},), got {data.shape}")

        d_left, d_right = self._calculate_edge_deviations(data)

        self._process_side("left", d_left, step)
        self._process_side("right", d_right, step)

        return d_left, d_right

    def _process_side(self, side: str, delta: float, step: int):
        if delta >= self.cfg.threshold:
            self._counters[side] += 1
        else:
            self._counters[side] = 0

        if self._counters[side] >= self.cfg.edge_persistence:
            hit_index = step - self.cfg.edge_persistence
            
            if side == "left" and self.results.step_left is None:
                self.results.step_left = hit_index
            elif side == "right" and self.results.step_right is None:
                self.results.step_right = hit_index

            if self.results.first_hit_step is None:
                self.results.first_hit_step = hit_index
                self.results.first_side = side
            elif self.results.first_hit_step == hit_index and self.results.first_side != side:
                self.results.first_side = "both"

    def reset(self):
        """Clears baselines, counters, and results for a new run."""
        self._baselines = {"left": None, "right": None}
        self._counters = {"left": 0, "right": 0}
        self.results = DetectionResult()

    @staticmethod
    def _get_d_hit_eff(side: Optional[str], d_l_eff: float, d_r_eff: float) -> Optional[float]:
        if side is None:
            return None
        if side == "left":
            return d_l_eff
        if side == "right":
            return d_r_eff
        return min(d_l_eff, d_r_eff)
