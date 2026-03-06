from __future__ import annotations
import datetime as _date
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import h5py
import numpy as np
import yaml

@dataclass(frozen=True)
class Writer:
    file_path: Path
    metadata: Dict[str, Any]  # Initialization metadata (schema, exp_id, etc.)

    def initialize_file(self) -> None:
        """Creates the file and writes global attributes if it doesn't exist."""
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        if self.file_path.exists():
            return
            
        with h5py.File(self.file_path, "w") as f:
            for k, v in self.metadata.items():
                f.attrs[k] = self._serialize_attr(v)
            f.attrs["created_at_utc"] = _date.datetime.now(_date.timezone.utc).isoformat()

    def save_run(
        self,
        group_name: str,
        run_id: str,
        datasets: Dict[str, np.ndarray],
        attributes: Dict[str, Any]
    ) -> None:
        """
        A dataset persists under a generic hierarchy.
        Structure: /group_name/runs/run_id/datasets
        """
        with h5py.File(self.file_path, "a") as f:
            # Navigation/Group Creation
            grp = f.require_group(group_name)
            runs_grp = grp.require_group("runs")
            
            if run_id in runs_grp:
                raise ValueError(f"Collision with run_id: '{run_id}' already exists in '{group_name}'")
            
            run_grp = runs_grp.create_group(run_id)

            # Attribute Recording (Run Metadata)
            for k, v in attributes.items():
                run_grp.attrs[k] = self._serialize_attr(v)

            # Dataset Recording (Numerical Data)
            for name, data in datasets.items():
                run_grp.create_dataset(
                    name, 
                    data=np.asarray(data), 
                    compression="gzip", 
                    compression_opts=4
                )

    @staticmethod
    def _serialize_attr(v: Any) -> Any:
        """Converts complex data types to formats accepted by HDF5."""
        if isinstance(v, (dict, list, tuple)):
            return yaml.safe_dump(v, sort_keys=True)
        if v is None: return "none"
        return v
