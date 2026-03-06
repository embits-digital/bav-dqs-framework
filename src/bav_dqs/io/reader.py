from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import h5py
import numpy as np
import yaml

class Reader:
    def __init__(self, file_path: Union[Path, str]) -> None:
        self.file_path = Path(file_path)
        self._f: Optional[h5py.File] = None

    def __enter__(self) -> "Reader":
        if not self.file_path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {self.file_path}")
        self._f = h5py.File(self.file_path, "r")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._f:
            self._f.close()
            self._f = None

    def _require_open(self) -> h5py.File:
        if self._f is None:
            raise RuntimeError("Reader fechado. Utilize o bloco 'with'.")
        return self._f

    def _unserialize_attr(self, v: Any) -> Any:
        """Processo inverso do Writer._serialize_attr."""
        if isinstance(v, str):
            # Tenta carregar como YAML se parecer uma estrutura complexa
            if v.startswith(("{", "[", "config:", "physics:", "experiment:")) or "\n" in v:
                if v == "none": return None
                try:
                    return yaml.safe_load(v)
                except yaml.YAMLError:
                    return v
        return v

    def get_global_metadata(self) -> Dict[str, Any]:
        """Lê os atributos de inicialização gravados pelo Writer.initialize_file."""
        f = self._require_open()
        return {k: self._unserialize_attr(v) for k, v in f.attrs.items()}

    def list_groups(self) -> List[str]:
        """Lista os group_names (ex: nomes dos experimentos ou configurações)."""
        f = self._require_open()
        return list(f.keys())

    def list_runs(self, group_name: str) -> List[str]:
        """Lista todos os run_ids dentro de um grupo específico."""
        f = self._require_open()
        path = f"{group_name}/runs"
        return list(f[path].keys()) if path in f else []

    def get_run_data(
        self, 
        group_name: str, 
        run_id: str
    ) -> Dict[str, np.ndarray]:
        """
        Extrai todos os datasets de uma run específica.
        Compatível com Writer.save_run(datasets=...)
        """
        f = self._require_open()
        run_path = f"{group_name}/runs/{run_id}"
        
        if run_path not in f:
            raise KeyError(f"Run '{run_id}' não encontrada no grupo '{group_name}'")
            
        group = f[run_path]
        return {name: np.asarray(group[name]) for name in group.keys()}

    def get_run_attributes(
        self, 
        group_name: str, 
        run_id: str
    ) -> Dict[str, Any]:
        """Lê os metadados específicos de uma execução (attributes no Writer)."""
        f = self._require_open()
        run_path = f"{group_name}/runs/{run_id}"
        attrs = f[run_path].attrs
        return {k: self._unserialize_attr(v) for k, v in attrs.items()}

    def find_runs_with_attribute(self, group_name: str, key: str, value: Any) -> List[str]:
        """Utilitário para filtrar runs por um metadado específico."""
        matching_runs = []
        for rid in self.list_runs(group_name):
            attrs = self.get_run_attributes(group_name, rid)
            if attrs.get(key) == value:
                matching_runs.append(rid)
        return matching_runs
