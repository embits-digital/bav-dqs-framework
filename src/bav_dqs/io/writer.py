from __future__ import annotations

import datetime as _date
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import h5py
import numpy as np
import yaml

def _now_utc_z() -> str:
    return _date.datetime.now(_date.timezone.utc).replace(microsecond=0).isoformat() + "Z"

def _git_state() -> Dict[str, Any]:
    state = {"git_commit": "unknown", "git_dirty": False}
    try:
        state["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"]).decode().strip()
        state["git_dirty"] = bool(dirty)
    except Exception:
        pass
    return state

def _yaml_snapshot(cfg: Dict[str, Any]) -> str:
    return yaml.safe_dump(cfg, sort_keys=True)

def _require_1d(*arrs: np.ndarray) -> int:
    if len(arrs) == 0:
        raise ValueError("Internal error: no arrays provided")
    n: Optional[int] = None
    for a in arrs:
        if a is None:
            raise ValueError("Internal error: required array is None")
        a = np.asarray(a)
        if a.ndim != 1:
            raise ValueError("Expected 1D arrays for time/metric series")
        if n is None:
            n = int(a.shape[0])
        elif int(a.shape[0]) != n:
            raise ValueError("1D series length mismatch")
    return int(n if n is not None else 0)

def _require_2d_occ(occ: np.ndarray, T: int, n_qubits: int, name: str) -> None:
    occ = np.asarray(occ)
    if occ.ndim != 2:
        raise ValueError(f"{name} must be 2D (T, N)")
    if int(occ.shape[0]) != int(T):
        raise ValueError(f"{name} time length mismatch: {occ.shape[0]} != {T}")
    if int(occ.shape[1]) != int(n_qubits):
        raise ValueError(f"{name} n_qubits mismatch: {occ.shape[1]} != {n_qubits}")

def _as_int_or_neg1(x: Optional[int]) -> int:
    return -1 if x is None else int(x)

def _create_or_require_group(parent: h5py.Group, name: str) -> h5py.Group:
    if name in parent:
        obj = parent[name]
        if not isinstance(obj, h5py.Group):
            raise TypeError(f"Expected group at {name}")
        return obj
    return parent.create_group(name)

def _write_or_replace_dataset(grp: h5py.Group, name: str, data: np.ndarray) -> None:
    if name in grp:
        del grp[name]
    grp.create_dataset(name, data=data, compression="gzip", compression_opts=4)

def _is_scalar_ok(v: Any) -> bool:
    return isinstance(v, (str, bytes, int, float, bool, np.integer, np.floating, np.bool_))

def _attr_value(v: Any) -> Any:
    if v is None:
        return "none"

    if isinstance(v, (np.integer, np.floating, np.bool_)):
        return v.item()

    if _is_scalar_ok(v):
        return v

    # Common containers: serialize for safety
    if isinstance(v, (dict, list, tuple)):
        try:
            return yaml.safe_dump(v, sort_keys=True)
        except Exception:
            return str(v)

    return str(v)

def _make_run_id() -> str:
    # High uniqueness without relying on wall-clock granularity
    # Example: 20260211_101530_123456_ab12cd34
    ts = _date.datetime.now(_date.timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    suf = uuid.uuid4().hex[:8]
    return f"{ts}_{suf}"

@dataclass(frozen=True)
class Writer:
    file_path: Path
    config: Dict[str, Any]
    schema_version: str
    experiment_id: str

    def initialize_file(self) -> None:
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        if self.file_path.exists():
            return
        with h5py.File(self.file_path, "w") as f:
            f.attrs["schema_version"] = str(self.schema_version)
            f.attrs["experiment_id"] = str(self.experiment_id)
            f.attrs["start_time_utc"] = _now_utc_z()
            f.attrs["config_snapshot"] = _yaml_snapshot(self.config)
            git = _git_state()
            f.attrs["git_commit"] = str(git["git_commit"])
            f.attrs["git_dirty"] = bool(git["git_dirty"])

    def save_simulation_101(
        self,
        *,
        n_qubits: int,
        run_id: Optional[str],

        # FULL (dt) — complete
        t_full: np.ndarray,
        occ_full: np.ndarray,
        metric_full_left: np.ndarray,
        metric_full_right: np.ndarray,

        # HALF RAW (dt/2) — complete
        t_half: np.ndarray,
        occ_half: np.ndarray,
        metric_half_left: np.ndarray,
        metric_half_right: np.ndarray,

        # Metadata / markers (attrs)
        metadata: Dict[str, Any],

        # Optional Richardson on aligned grid
        occ_rich: Optional[np.ndarray] = None,
        metric_rich_left: Optional[np.ndarray] = None,
        metric_rich_right: Optional[np.ndarray] = None,
    ) -> str:
        n_qubits = int(n_qubits)
        run_id = str(run_id) if run_id else _make_run_id()

        # --- validate full/raw lengths
        t_full = np.asarray(t_full, dtype=float)
        t_half = np.asarray(t_half, dtype=float)

        metric_full_left = np.asarray(metric_full_left, dtype=float)
        metric_full_right = np.asarray(metric_full_right, dtype=float)
        metric_half_left = np.asarray(metric_half_left, dtype=float)
        metric_half_right = np.asarray(metric_half_right, dtype=float)

        Tf = _require_1d(t_full, metric_full_left, metric_full_right)
        Th = _require_1d(t_half, metric_half_left, metric_half_right)

        occ_full = np.asarray(occ_full, dtype=float)
        occ_half = np.asarray(occ_half, dtype=float)

        _require_2d_occ(occ_full, Tf, n_qubits, "occ_full")
        _require_2d_occ(occ_half, Th, n_qubits, "occ_half")

        # --- deterministic aligned derivations (always persisted)
        t_half_aligned = t_half[0::2]
        occ_half_aligned = occ_half[0::2, :]
        metric_half_left_aligned = metric_half_left[0::2]
        metric_half_right_aligned = metric_half_right[0::2]

        Ta = _require_1d(t_half_aligned, metric_half_left_aligned, metric_half_right_aligned)
        _require_2d_occ(occ_half_aligned, Ta, n_qubits, "occ_half_aligned")

        # --- richardson validation (aligned grid)
        if occ_rich is not None:
            if metric_rich_left is None or metric_rich_right is None:
                raise ValueError("If occ_rich is provided, metric_rich_left and metric_rich_right are required.")
            occ_rich = np.asarray(occ_rich, dtype=float)
            metric_rich_left = np.asarray(metric_rich_left, dtype=float)
            metric_rich_right = np.asarray(metric_rich_right, dtype=float)
            Tr = _require_1d(metric_rich_left, metric_rich_right)
            if occ_rich.ndim != 2:
                raise ValueError("occ_rich must be 2D (T, N)")
            if int(occ_rich.shape[0]) != int(Tr) or int(occ_rich.shape[1]) != int(n_qubits):
                raise ValueError("occ_rich shape mismatch vs metric_rich_* or n_qubits")

        # --- write
        with h5py.File(self.file_path, "a") as f:
            # width group
            grp_name = f"n_qubits_{n_qubits}"
            g = _create_or_require_group(f, grp_name)
            g.attrs["n_qubits"] = n_qubits
            g.attrs["schema_version"] = str(self.schema_version)

            runs = _create_or_require_group(g, "runs")

            # Do not overwrite an existing run_id
            if run_id in runs:
                raise ValueError(f"run_id collision: {run_id} already exists under {grp_name}/runs")

            rgrp = runs.create_group(run_id)

            # run attrs (self-descriptive)
            rgrp.attrs["run_id"] = run_id
            rgrp.attrs["run_start_time_utc"] = _now_utc_z()
            rgrp.attrs["schema_version"] = str(self.schema_version)
            rgrp.attrs["experiment_id"] = str(self.experiment_id)
            rgrp.attrs["config_snapshot"] = _yaml_snapshot(self.config)

            # canonical hits
            rgrp.attrs["first_hit_full"] = _as_int_or_neg1(metadata.get("first_hit_full"))
            rgrp.attrs["first_hit_half_raw"] = _as_int_or_neg1(metadata.get("first_hit_half_raw"))
            rgrp.attrs["first_hit_half_aligned"] = _as_int_or_neg1(metadata.get("first_hit_half_aligned"))

            # other metadata (sanitized)
            for k, v in metadata.items():
                if k in ("first_hit_full", "first_hit_half_raw", "first_hit_half_aligned"):
                    continue
                rgrp.attrs[str(k)] = _attr_value(v)

            # datasets: full raw
            _write_or_replace_dataset(rgrp, "t_full", t_full)
            _write_or_replace_dataset(rgrp, "occ_full", occ_full)
            _write_or_replace_dataset(rgrp, "metric_full_left", metric_full_left)
            _write_or_replace_dataset(rgrp, "metric_full_right", metric_full_right)

            # datasets: half raw
            _write_or_replace_dataset(rgrp, "t_half", t_half)
            _write_or_replace_dataset(rgrp, "occ_half", occ_half)
            _write_or_replace_dataset(rgrp, "metric_half_left", metric_half_left)
            _write_or_replace_dataset(rgrp, "metric_half_right", metric_half_right)

            # datasets: aligned (always)
            _write_or_replace_dataset(rgrp, "t_half_aligned", t_half_aligned)
            _write_or_replace_dataset(rgrp, "occ_half_aligned", occ_half_aligned)
            _write_or_replace_dataset(rgrp, "metric_half_left_aligned", metric_half_left_aligned)
            _write_or_replace_dataset(rgrp, "metric_half_right_aligned", metric_half_right_aligned)

            # datasets: rich (optional)
            if occ_rich is not None:
                _write_or_replace_dataset(rgrp, "occ_rich", occ_rich)
                _write_or_replace_dataset(rgrp, "metric_rich_left", metric_rich_left)
                _write_or_replace_dataset(rgrp, "metric_rich_right", metric_rich_right)

            # record "latest" pointer for convenience (Reader uses it)
            g.attrs["latest_run_id"] = run_id

        return run_id
