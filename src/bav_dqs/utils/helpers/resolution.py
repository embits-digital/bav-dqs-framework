"""Time-resolution utilities: base grid (step Δt) and refined grid (step Δt/2, twice as many steps).

Conventions
-----------
* Step k of the base grid is physical time t = k·Δt; step j of the refined grid is t = j·Δt/2.
  The refined series sampled at even j (``[0::2]``) lives on the base grid.
* A hit index h is the value reported by the detector (``step - edge_persistence``). Hits from the two
  grids are compared in PHYSICAL TIME, never by raw index.
* Mapping a refined-grid hit to the base grid uses the floor, ``h_half // 2``: the mapped index never lies
  after the physical hit, so the safe window can only shrink (conservative). An odd refined index is
  therefore not discarded (the previous implementation dropped it and silently fell back to the base hit).
* Richardson combines the two resolutions as (2^p·f(Δt/2) − f(Δt)) / (2^p − 1), where p is the global
  error order of the integrator (p = 1 for the first-order Lie-Trotter circuit). The two resolutions are
  always Δt and Δt/2; p is NOT a number of resolutions.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

_TOL = 1e-9


def align_half_to_full(occ_half_raw: np.ndarray, first_hit_half_raw: Optional[int]) -> Tuple[np.ndarray, Optional[int]]:
    """Refined series sampled on the base grid and the refined-grid hit mapped to base steps (floor)."""
    occ_half_aligned = np.asarray(occ_half_raw)[0::2]
    if first_hit_half_raw is None:
        return occ_half_aligned, None
    fh = int(first_hit_half_raw)
    if fh < 0:
        return occ_half_aligned, None
    return occ_half_aligned, fh // 2


def compute_n_safe_full_grid(*, n_steps_full: int, first_hit_full: Optional[int],
                             first_hit_half_aligned: Optional[int]) -> int:
    """Length (base steps) of the safe prefix: the earliest hit across both resolutions."""
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
    return int(min(max(0, min(candidates)), n_steps_full))


def check_richardson_order(order_p: int, integrator_order: int) -> Optional[str]:
    """Returns a warning message when ``richardson.order_p`` differs from the integrator's error order.

    With the wrong p the extrapolation does not cancel the leading error term (e.g. p=2 on a first-order
    scheme leaves about one third of it), so the combined series is not the improvement it claims to be.
    """
    if int(order_p) == int(integrator_order):
        return None
    return (f"richardson.order_p={order_p} does not match the integrator's global error order "
            f"({integrator_order}): the extrapolation will not cancel the leading error term.")


def richardson_extrapolate(occ_full_prefix: np.ndarray, occ_half_aligned_prefix: np.ndarray, order_p: int) -> np.ndarray:
    p = int(order_p)
    if p < 1:
        raise ValueError("richardson.order_p must be >= 1")
    factor = 2.0 ** p
    return (factor * occ_half_aligned_prefix - occ_full_prefix) / (factor - 1.0)


@dataclass(frozen=True)
class SafeWindow:
    t_full: Optional[float]      # physical time of the hit on the base grid (Δt)
    t_half: Optional[float]      # physical time of the hit on the refined grid (Δt/2)
    t_safe: Optional[float]      # earliest of both
    limiting: Optional[str]      # "dt", "dt/2" or "both"
    p_safe_base: Optional[int]   # t_safe expressed in base steps (floor)


def safe_window(first_hit_full: int, first_hit_half: int, dt_full: float) -> SafeWindow:
    """Safe window from the hits stored as attributes (-1 means no hit; index 0 is a valid hit)."""
    t_full = int(first_hit_full) * dt_full if int(first_hit_full) >= 0 else None
    t_half = int(first_hit_half) * (dt_full / 2.0) if int(first_hit_half) >= 0 else None
    times = [t for t in (t_full, t_half) if t is not None]
    if not times:
        return SafeWindow(t_full, t_half, None, None, None)
    t_safe = min(times)
    at_full = t_full is not None and abs(t_full - t_safe) <= _TOL * max(1.0, abs(t_safe))
    at_half = t_half is not None and abs(t_half - t_safe) <= _TOL * max(1.0, abs(t_safe))
    limiting = "both" if (at_full and at_half) else ("dt" if at_full else "dt/2")
    return SafeWindow(t_full, t_half, t_safe, limiting, int(np.floor(t_safe / dt_full + _TOL)))
