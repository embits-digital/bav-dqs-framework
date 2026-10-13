"""Geometry of the Dirac-lattice use case.

Pure Python on purpose: it must be importable without any quantum framework, so the
use-case knowledge (where the initial excitation sits) lives here and NOT in the
framework-agnostic boundary detector.
"""
from __future__ import annotations
from typing import Tuple


def source_index(n_qubits: int) -> int:
    """Site of the initial excitation (center of the lattice)."""
    return int(n_qubits) // 2


def excluded_sites(n_qubits: int) -> Tuple[int, ...]:
    """Sites that an edge window must not contain for the boundary marker to be meaningful."""
    return (source_index(n_qubits),)


def max_edge_window(n_qubits: int) -> int:
    """Largest edge window (per side) that excludes the source: min(src, N-1-src)."""
    n, src = int(n_qubits), source_index(n_qubits)
    return max(0, min(src, n - 1 - src))
