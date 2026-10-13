from __future__ import annotations
from typing import Any, Dict

import numpy as np

from bav_dqs.core.engines.base import BaseEngine


class NumpyEngine(BaseEngine):
    """Exact, noiseless engine in the single-excitation subspace (no quantum framework required).

    Contract (same as every BaseEngine):
      init_def : initial state vector
      step_def : one-step unitary matrix
      obs_defs : diagonal observables (see ``core.operators.numpy_observables``)
    """

    def __init__(self, backend_cfg: Dict[str, Any]):
        mode = str(backend_cfg.get("mode", "ideal")).lower()
        if mode != "ideal":
            raise ValueError(f"NumpyEngine only supports mode='ideal' (exact, noiseless); got '{mode}'.")
        self.mode = mode
        self._meta = {"mode": mode, "framework": "numpy", "sector": "single-excitation"}

    def compute_step(self, step_idx, n, init_def, step_def, obs_defs, ctx) -> np.ndarray:
        if step_idx == 0:
            ctx["state"] = np.asarray(init_def, dtype=complex)
        else:
            ctx["state"] = np.asarray(step_def, dtype=complex) @ ctx["state"]
        prob = np.abs(ctx["state"]) ** 2
        return np.array([float(np.dot(np.asarray(d, dtype=float), prob)) for d in obs_defs])

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._meta
