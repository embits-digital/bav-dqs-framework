from __future__ import annotations
from qiskit import QuantumCircuit
from qiskit.circuit.library import RXXGate, RYYGate, RZGate

from bav_dqs.utils.types.dirac_simulation import DiracSimulationModelCfg

def build_initial_circuit(n_qubits: int) -> QuantumCircuit:
    """Prepares the initial state: single excitation at the center of the 1D network."""
    n = int(n_qubits)
    if n < 2: raise ValueError("n_qubits >= 2")
    qc = QuantumCircuit(n)
    qc.x(n // 2) 
    return qc

def build_step_circuit(n_qubits: int, cfg: DiracSimulationModelCfg) -> QuantumCircuit:
    n = int(n_qubits)
    if n < 2:
        raise ValueError("n_qubits must be >= 2")

    m = float(cfg.m)
    w = float(cfg.w)
    dt = float(cfg.dt)

    qc = QuantumCircuit(n)

    qc.x(0)

    angle_z = m * dt
    for q in range(n):
        sign = 1.0 if q % 2 == 0 else -1.0
        qc.append(RZGate(sign * angle_z), [q])

    angle_xy = w * dt
    for q in range(n - 1):
        qc.append(RXXGate(angle_xy), [q, q + 1])
        qc.append(RYYGate(angle_xy), [q, q + 1])

    return qc