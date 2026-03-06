from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import h5py
import numpy as np


class Reader:
    def __init__(self, path: Path) -> None:
        self._path = str(path)
        self._f: Optional[h5py.File] = None

    def __enter__(self) -> "Reader":
        self._f = h5py.File(self._path, "r")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None

    def _require_open(self) -> h5py.File:
        if self._f is None:
            raise RuntimeError("Reader is not open. Use `with Reader(path) as r:`")
        return self._f
    
    def _json_safe(self, v: Any) -> Any:
        if isinstance(v, (np.integer, np.floating, np.bool_)):
            return v.item()
        if isinstance(v, (bytes, bytearray)):
            try:
                return v.decode()
            except Exception:
                return v.hex()
        return v
    
    def root_metadata(self) -> Dict[str, Any]:
        f = self._require_open()
        return {str(k): self._json_safe(v) for k, v in dict(f.attrs.items()).items()}
    
    def get_config(self) -> Dict[str, Any]:
        """
        Retrieves the snapshot of the YAML configuration saved in the H5 file
        and transforms it back into a Python dictionary.
        """
        import yaml # Import local to avoid dependency if not used.
        
        f = self._require_open()
        snapshot = f.attrs.get("config_snapshot")
        
        if snapshot is None:
            raise KeyError("The H5 file does not contain a 'config_snapshot' in the root directory.")
            
        # The snapshot can be delivered as bytes or a string, depending on the h5py version.
        if isinstance(snapshot, bytes):
            snapshot = snapshot.decode("utf-8")
            
        config = yaml.safe_load(snapshot)
        
        if not isinstance(config, dict):
            raise ValueError("The retrieved config_snapshot is not a valid YAML mapping.")
            
        return config

    def iter_widths(self) -> Iterator[int]:
        f = self._require_open()
        for k in f.keys():
            if k.startswith("n_qubits_"):
                yield int(k.split("_")[-1])

    def _width_group(self, n_qubits: int) -> h5py.Group:
        f = self._require_open()
        return f[f"n_qubits_{int(n_qubits)}"]

    def iter_runs(self, n_qubits: int) -> Iterator[str]:
        g = self._width_group(n_qubits)
        if "runs" not in g:
            return iter(())
        runs = g["runs"]
        for rid in runs.keys():
            yield str(rid)

    def _resolve_run_id(self, n_qubits: int) -> str:
        g = self._width_group(n_qubits)
        rid = g.attrs.get("latest_run_id", None)
        if rid is not None:
            return str(rid)

        if "runs" in g:
            runs = g["runs"]
            ids = list(runs.keys())
            if len(ids) == 1:
                return str(ids[0])
            if len(ids) == 0:
                raise KeyError(f"No runs found for n_qubits={int(n_qubits)}")
            raise KeyError(
                f"Multiple runs found for n_qubits={int(n_qubits)} but latest_run_id is missing"
            )

        raise KeyError(f"Missing runs group and latest_run_id for n_qubits={int(n_qubits)}")

    def group(self, n_qubits: int, run_id: Optional[str] = None) -> h5py.Group:
        g = self._width_group(n_qubits)

        # Canonical layout: /n_qubits_N/runs/<run_id>/...
        if "runs" in g:
            rid = str(run_id) if run_id else self._resolve_run_id(n_qubits)
            return g["runs"][rid]

        # Legacy fallback: /n_qubits_N/<datasets>
        if run_id is not None:
            raise KeyError(f"Legacy layout does not support run_id for n_qubits={int(n_qubits)}")
        return g

    # FULL
    def t_full(self, n_qubits: int, run_id: Optional[str] = None) -> np.ndarray:
        return np.asarray(self.group(n_qubits, run_id)["t_full"])

    def occ_full(self, n_qubits: int, run_id: Optional[str] = None) -> np.ndarray:
        return np.asarray(self.group(n_qubits, run_id)["occ_full"])

    def metric_full_left(self, n_qubits: int, run_id: Optional[str] = None) -> np.ndarray:
        return np.asarray(self.group(n_qubits, run_id)["metric_full_left"])

    def metric_full_right(self, n_qubits: int, run_id: Optional[str] = None) -> np.ndarray:
        return np.asarray(self.group(n_qubits, run_id)["metric_full_right"])

    # HALF RAW
    def t_half(self, n_qubits: int, run_id: Optional[str] = None) -> np.ndarray:
        return np.asarray(self.group(n_qubits, run_id)["t_half"])

    def occ_half(self, n_qubits: int, run_id: Optional[str] = None) -> np.ndarray:
        return np.asarray(self.group(n_qubits, run_id)["occ_half"])

    def metric_half_left(self, n_qubits: int, run_id: Optional[str] = None) -> np.ndarray:
        return np.asarray(self.group(n_qubits, run_id)["metric_half_left"])

    def metric_half_right(self, n_qubits: int, run_id: Optional[str] = None) -> np.ndarray:
        return np.asarray(self.group(n_qubits, run_id)["metric_half_right"])

    # HALF ALIGNED
    def has_half_aligned(self, n_qubits: int, run_id: Optional[str] = None) -> bool:
        grp = self.group(n_qubits, run_id)
        return "occ_half_aligned" in grp and "t_half_aligned" in grp

    def t_half_aligned(self, n_qubits: int, run_id: Optional[str] = None) -> np.ndarray:
        return np.asarray(self.group(n_qubits, run_id)["t_half_aligned"])

    def occ_half_aligned(self, n_qubits: int, run_id: Optional[str] = None) -> np.ndarray:
        return np.asarray(self.group(n_qubits, run_id)["occ_half_aligned"])

    def metric_half_left_aligned(self, n_qubits: int, run_id: Optional[str] = None) -> np.ndarray:
        return np.asarray(self.group(n_qubits, run_id)["metric_half_left_aligned"])

    def metric_half_right_aligned(self, n_qubits: int, run_id: Optional[str] = None) -> np.ndarray:
        return np.asarray(self.group(n_qubits, run_id)["metric_half_right_aligned"])

    # RICH
    def has_richardson(self, n_qubits: int, run_id: Optional[str] = None) -> bool:
        return "occ_rich" in self.group(n_qubits, run_id)

    def occ_rich(self, n_qubits: int, run_id: Optional[str] = None) -> np.ndarray:
        return np.asarray(self.group(n_qubits, run_id)["occ_rich"])

    def metric_rich_left(self, n_qubits: int, run_id: Optional[str] = None) -> np.ndarray:
        return np.asarray(self.group(n_qubits, run_id)["metric_rich_left"])

    def metric_rich_right(self, n_qubits: int, run_id: Optional[str] = None) -> np.ndarray:
        return np.asarray(self.group(n_qubits, run_id)["metric_rich_right"])

    # Metadata / Hits
    def metadata(self, n_qubits: int, run_id: Optional[str] = None) -> Dict[str, Any]:
        grp = self.group(n_qubits, run_id)
        return {str(k): self._json_safe(v) for k, v in dict(grp.attrs.items()).items()}

    def first_hit(self, n_qubits: int, run_id: Optional[str] = None) -> Dict[str, Optional[int]]:
        md = self.metadata(n_qubits, run_id)

        def norm(v: Any) -> Optional[int]:
            try:
                iv = int(v)
            except Exception:
                return None
            return None if iv < 0 else iv

        return {
            "first_hit_full": norm(md.get("first_hit_full", -1)),
            "first_hit_half_raw": norm(md.get("first_hit_half_raw", -1)),
            "first_hit_half_aligned": norm(md.get("first_hit_half_aligned", -1)),
        }
