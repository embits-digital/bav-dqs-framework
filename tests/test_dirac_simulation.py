import pytest
import numpy as np
from unittest.mock import MagicMock
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from bav_dqs.core.engines.qiskit import _get_occupation
from bav_dqs.core.detectors.boundary_detector import _update_hit_logic, _get_d_hit_eff
from bav_dqs.core.operators.z_observable import build_z_observables
from bav_dqs.plugins.dirac_simulation import _calculate_side, _validate_inputs, run_boundary_detection

@pytest.fixture
def basic_configs():
    """Minimum settings for scientific consistency."""
    return {
        "model": {"m": 1.0, "w": 1.0, "dt": 0.1},
        "detector": {"threshold": 0.5, "edge_window": 1, "edge_persistence": 2},
        "backend": {"log_every_steps": 0, "abort_on_complex_evs": True}
    }

# --- Unit Tests in Scientific Logic ---

def test_validate_inputs_raises():
    """Ensures physical integrity: system needs minimum dimensions."""
    with pytest.raises(ValueError, match="n_qubits must be >= 2"):
        _validate_inputs(n=1, T=10)
    with pytest.raises(ValueError, match="max_steps must be >= 1"):
        _validate_inputs(n=4, T=0)

@pytest.mark.parametrize("side, dl, dr, expected", [
    ("left", 10.0, 20.0, 10.0),
    ("right", 10.0, 20.0, 20.0),
    ("both", 10.0, 20.0, 10.0), # Shortest distance (first hit)
    (None, 10.0, 20.0, None),
])
def test_d_hit_eff_calculation(side, dl, dr, expected):
    """Validates the accuracy of the effective distance metric for the v_hit calculation."""
    assert _get_d_hit_eff(side, dl, dr) == expected

def test_update_hit_logic_persistence():
    """Validates whether the persistence filter avoids false positives (stability)."""
    res = {"hits": [None, None], "counts": [0, 0]}
    thr, edge_persistence = 0.8, 3
    
    # Steps 1 and 2: Below persistence
    for i in range(1, 3):
        _update_hit_logic(res, dl=0.9, dr=0.1, thr=thr, edge_persistence=edge_persistence, step_idx=i)
        assert res["hits"][0] is None 
        
    # Step 3: Achieve persistence
    _update_hit_logic(res, dl=0.9, dr=0.1, thr=thr, edge_persistence=edge_persistence, step_idx=3)
    assert res["hits"][0] == 0 # (3 - 3) original step at the start of detection

# --- Interoperability and Backend Testing ---

def test_get_occupation_ideal_mode(basic_configs):
    """Validates whether occupancy extraction preserves the quantum norm (sum <= 1.0)."""
    n = 2
    # Mock context to simulate state persistence.
    ctx = {"state": None, "qc": QuantumCircuit(n)}
    init_qc = QuantumCircuit(n) # Estado |00>
    step_qc = QuantumCircuit(n) # Identidade
    obs = build_z_observables(n)
    
    # Occupation should return an array of size n.
    occ = _get_occupation(
        step_idx=0, mode="ideal", estimator=None, 
        step_qc=step_qc, init_qc=init_qc, observables=obs,
        backend_cfg=basic_configs["backend"], ctx=ctx
    )
    
    assert len(occ) == n
    assert np.all(occ >= 0) and np.all(occ <= 1.0)
    assert isinstance(ctx["state"], Statevector)

# --- Integration Testing (End-to-End) ---

def test_full_simulation_run(basic_configs):
    """Smoke test: full integration of the Dirac plugin."""
    # Simplified 3-qubit configuration
    result = run_boundary_detection(
        n_qubits=4,
        model_cfg=basic_configs["model"],
        detector_cfg=basic_configs["detector"],
        backend_cfg={"mode": "ideal"},
        max_steps=5
    )
    
    # Integrity checks of the Result Object
    assert result.n_qubits == 4
    assert result.history.shape == (6, 4) # T+1 steps, n qubits
    assert hasattr(result, 'v_hit_est')
    assert isinstance(result.d_left_eff, float)

def test_calculate_side_logic():
    """Ensures the correct classification of the direction of propagation."""
    assert _calculate_side(None, None) is None
    assert _calculate_side(5, 5) == "both"
    assert _calculate_side(2, 8) == "left"
    assert _calculate_side(None, 4) == "right"
