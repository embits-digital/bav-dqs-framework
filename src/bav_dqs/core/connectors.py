"""Registry of framework connectors and use-case model builders.

The BAV-DQS methodology (boundary detector + admissibility gating) only sees occupation series; it never
imports a quantum framework. A *connector* adapts one framework to the ``BaseEngine`` contract, and a
*model builder* provides the use-case dynamics for that framework. Both are loaded lazily, so the
framework is imported only when it is actually selected (``backend.framework`` in the YAML, default ``qiskit``).

Adding another framework (Cirq, PennyLane, tensor networks, ...) means registering a loader here; the
detector, the pipeline and the reports stay untouched:

    register_connector("myfw", _load_myfw)            # -> Connector(...)
    register_model_builder("dirac", "myfw", _load_dirac_myfw)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

from bav_dqs.core.engines.base import BaseEngine


@dataclass(frozen=True)
class Connector:
    name: str
    create_engine: Callable[[Dict[str, Any]], BaseEngine]
    build_occupancy_observables: Callable[[int], List[Any]]
    build_correlation_observables: Callable[[int, int], List[Any]]


_CONNECTORS: Dict[str, Callable[[], Connector]] = {}
_MODELS: Dict[Tuple[str, str], Callable[[], Callable[[int, Any], Tuple[Any, Any]]]] = {}


def register_connector(name: str, loader: Callable[[], Connector]) -> None:
    _CONNECTORS[name.strip().lower()] = loader


def register_model_builder(use_case: str, framework: str, loader: Callable[[], Callable]) -> None:
    _MODELS[(use_case.strip().lower(), framework.strip().lower())] = loader


def available_frameworks() -> List[str]:
    return sorted(_CONNECTORS)


def get_connector(framework: str = "qiskit") -> Connector:
    key = str(framework).strip().lower()
    if key not in _CONNECTORS:
        raise ValueError(f"Unknown framework '{framework}'. Available: {available_frameworks()}")
    return _CONNECTORS[key]()


def get_model_builder(use_case: str, framework: str = "qiskit") -> Callable[[int, Any], Tuple[Any, Any]]:
    """Returns ``build(n_qubits, model_cfg) -> (init_def, step_def)`` for the use case on that framework."""
    key = (use_case.strip().lower(), str(framework).strip().lower())
    if key not in _MODELS:
        raise ValueError(f"No model builder for use case '{key[0]}' on framework '{key[1]}'.")
    return _MODELS[key]()


# ----------------------------------------------------------------------------- Qiskit
def _load_qiskit() -> Connector:
    from bav_dqs.core.engines.qiskit_engine import QiskitEngine
    from bav_dqs.core.operators.correlation_observable import build_correlation_observables
    from bav_dqs.core.operators.z_observable import build_z_observables
    return Connector("qiskit", QiskitEngine, build_z_observables, build_correlation_observables)


def _load_dirac_qiskit():
    from bav_dqs.core.models.dirac_circuits import build_initial_circuit, build_step_circuit

    def build(n_qubits, model_cfg):
        return build_initial_circuit(n_qubits), build_step_circuit(n_qubits, model_cfg)
    return build


# ----------------------------------------------------------------------------- NumPy (framework-free)
def _load_numpy() -> Connector:
    from bav_dqs.core.engines.numpy_engine import NumpyEngine
    from bav_dqs.core.operators.numpy_observables import build_correlation_observables, build_z_observables
    return Connector("numpy", NumpyEngine, build_z_observables, build_correlation_observables)


def _load_dirac_numpy():
    from bav_dqs.core.models.dirac_numpy import build_initial_state, build_step_matrix

    def build(n_qubits, model_cfg):
        return build_initial_state(n_qubits), build_step_matrix(n_qubits, model_cfg)
    return build


register_connector("qiskit", _load_qiskit)
register_connector("numpy", _load_numpy)
register_model_builder("dirac", "qiskit", _load_dirac_qiskit)
register_model_builder("dirac", "numpy", _load_dirac_numpy)
