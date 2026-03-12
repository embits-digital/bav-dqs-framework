from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional
from contextlib import contextmanager

from ..helpers.config_manager import ConfigManager
from .writer import Writer
from .reader import Reader

class DataManager:
    """
    Central I/O point of the bav_dqs framework.
    Manages edge_persistence in HDF5, ensuring traceability and integrity.
    """
    def __init__(
        self,
        file_path: Path | str,
        config: Optional[Dict[str, Any]] = None,
        schema_version: str = "1.0.7.preprint",
        experiment_id: str = "default_exp",
        experiment_desc: str = "default_exp",
    ) -> None:
        self.file_path = Path(file_path)
        self.config = config or {}
        self.schema_version = schema_version
        self.experiment_id = experiment_id
        self.experiment_desc = experiment_desc
        
        # Writer is maintained as internal property.
        self._writer: Optional[Writer] = None

    @classmethod
    def from_yaml_config(cls, h5_path: Path | str, yaml_path: Path | str) -> DataManager:
        """Instantiate the Manager by validating the YAML file through the ConfigManager."""
        cfg_mngr = ConfigManager.from_yaml(str(yaml_path))
        
        # Retrieve YAML metadata or use defaults.
        exp_id = cfg_mngr.get("experiment.id")
        exp_desc = cfg_mngr.get("experiment.description")
        schema = cfg_mngr.get("experiment.schema_version")
        
        return cls(
            file_path=h5_path,
            config=cfg_mngr._config,
            experiment_id=exp_id,
            experiment_desc=exp_desc,
            schema_version=schema
        )

    def get_writer(self) -> Writer:
        if self._writer is None:
            # Groups the global metadata into the dictionary expected by Writer.
            full_metadata = {
                "config": self.config,
                "schema_version": self.schema_version,
                "experiment_id": self.experiment_id,
                "experiment_desc": self.experiment_desc,
            }
            
            self._writer = Writer(
                file_path=self.file_path,
                metadata=full_metadata  # Now it coincides with the signing of the Writer.
            )
            self._writer.initialize_file()
        return self._writer


    def get_reader(self) -> Reader:
        """Returns the Reader to access the H5 data."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.file_path}")
        return Reader(path=self.file_path)
    
    def open_reader(self) -> Reader:
        """
        Returns a Reader instance ready for use with Context Manager.
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.file_path}")
        return Reader(self.file_path)

    def __repr__(self) -> str:
        return f"<DataManager(exp={self.experiment_id}, path={self.file_path.name})>"

    @contextmanager
    def session(self):
        """Context Manager for fast read/write operations."""
        reader = self.open_reader()
        with reader as r:
            yield r

    def fetch_run_slice(self, group: str, run_id: str, dataset: str, start: int, end: int):
        """
        Example of Lazy Loading Real: Reads only a time interval from HDF5.
        Prevents memory overflow in very long simulation datasets.
        """
        with self.session() as r:
            ds = r.get_dataset_lazy(group, dataset, run_id)
            return ds[start:end] 

    def stream_run_data(self, group: str, run_id: str, dataset: str, chunk_size: int = 1000):
        """Generator for processing large files into chunks (streaming)."""
        with self.session() as r:
            ds = r.get_dataset_lazy(group, dataset, run_id)
            total_size = ds.shape[0]
            for i in range(0, total_size, chunk_size):
                yield ds[i : i + chunk_size]
    
    @classmethod
    def from_h5_file(cls, h5_path: Path | str) -> DataManager:
        """
        Instancia o DataManager a partir de um arquivo HDF5 existente,
        tentando recuperar metadados básicos se disponíveis.
        """
        path = Path(h5_path)
        with Reader(path) as reader:
            meta = reader.get_global_metadata()
        if not path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado em: {path}")

        return cls(
            file_path=path,
            config=meta.get("config", {}),
            schema_version=meta.get("schema_version", "1.0.7.preprint"),
            experiment_id=meta.get("experiment_id", "default_exp"),
            experiment_desc=meta.get("experiment_desc", "default_exp")
        )
