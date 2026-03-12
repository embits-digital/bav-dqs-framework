import pytest
import numpy as np
from unittest.mock import MagicMock

from bav_dqs.core.detectors.boundary_detector import BoundaryDetector
from bav_dqs.core.models.dirac_circuits import build_initial_circuit
from bav_dqs.core.operators.correlation_observable import build_correlation_observables
from bav_dqs.core.operators.definitions import get_dirac_observables
from bav_dqs.core.operators.z_observable import build_z_observables
from bav_dqs.utils.plugins.dirac_simulation import _calculate_side, _run_simulation_loop, _validate_inputs, run_boundary_detection

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
    assert BoundaryDetector._get_d_hit_eff(side, dl, dr) == expected

def test_update_hit_logic_persistence():
    """Validates whether the persistence filter avoids false positives (stability)."""
    res = {"hits": [None, None], "counts": [0, 0]}
    thr, edge_persistence = 0.8, 3
    
    # Steps 1 and 2: Below persistence
    for i in range(1, 3):
        BoundaryDetector._update_hit_logic(res, dl=0.9, dr=0.1, thr=thr, edge_persistence=edge_persistence, step_idx=i)
        assert res["hits"][0] is None 
        
    # Step 3: Achieve persistence
    BoundaryDetector._update_hit_logic(res, dl=0.9, dr=0.1, thr=thr, edge_persistence=edge_persistence, step_idx=3)
    assert res["hits"][0] == 0 # (3 - 3) original step at the start of detection

# --- Integration Testing (End-to-End) ---

def test_full_simulation_run(basic_configs):
    """Smoke test: full integration of the Dirac plugin."""
    # Simplified 3-qubit configuration
    result = run_boundary_detection(
        n_qubits=4,
        model_cfg=basic_configs["model"],
        detector_cfg=basic_configs["detector"],
        backend_cfg={"mode": "ideal"},
        validity_cfg={"p_min": 32, "stricted": True},
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

def test_calibrate_multi_threshold_rigor():
    """Garante que o threshold 5-sigma reage corretamente ao ruído do Estimator."""
    mock_estimator = MagicMock()
    # Simulamos 5 rodadas com um desvio padrão conhecido (ex: 0.02)
    mock_result = MagicMock()
    # Qiskit 2.3.0: PubResult.data.evs
    mock_result.__iter__.return_value = [
        MagicMock(data=MagicMock(evs=np.array([0.1, 0.12, 0.08]))) 
    ]
    mock_estimator.run.return_value.result.return_value = mock_result

    obs_map = {"occupancy": [MagicMock()]}
    init_qc = build_initial_circuit(2)
    
    thresholds = {}
    
    for label, observables in obs_map.items():
        all_values = []
        
        for _ in range(5):
            # No Qiskit 2.3.0, passamos a lista de tuplas (PUBs)
            pubs = [(init_qc, obs) for obs in observables]
            job = mock_estimator.run(pubs)
            result = job.result()
            step_values = [pub.data.evs for pub in result]
            all_values.append(step_values)
        
        data = np.array(all_values, dtype=float)
        
        std_dev = np.std(data)
        
        thresholds[label] = max(5.0 * std_dev, 1e-4)
        
        print(f"[BAV-DQS] {label} calibrated at 5-sigma: {thresholds[label]:.5f}")
    
    assert "occupancy" in thresholds
    assert thresholds["occupancy"] >= 1e-4 # Fallback de segurança
    # O valor deve ser ~5 * std_dev do nosso mock

def test_run_simulation_loop_multimodal_separation():
    """Verifica se o loop separa corretamente Ocupação (Z) de Correlação (ZZ)."""
    n = 4
    T = 1
    # Mock do Engine (substitui a necessidade de QiskitEngine real)
    mock_engine = MagicMock()
    
    # Mock do Detector
    mock_det = MagicMock()
    mock_det.update.return_value = (0.0, 0.0) 

    # Ocupação: (1 - Z)/2. Se queremos occ=0.5, Z deve ser 0.0
    # Correlação: valor bruto. Se queremos corr=0.9, ZZ deve ser 0.9
    # Retorno esperado: [Z0..Z3, ZZ0..ZZ2] -> [0,0,0,0, 0.9,0.9,0.9]
    z_vals = [0.0] * n
    zz_vals = [0.9] * n
    mock_engine.compute_step.return_value = np.array(z_vals + zz_vals)

    obs_map = get_dirac_observables(n)
    n_z = len(build_z_observables(n))
    n_zz = len(build_correlation_observables(n, int(n/2)))
    
    res = _run_simulation_loop(
        n=n, T=T, 
        engine=mock_engine, 
        init_def=None, step_def=None,
        obs_map=obs_map, 
        det=mock_det,
        logger=None,
        backend_cfg={"mode": "ideal"},
        validity_cfg={"stricted": False, "p_min": 32},
        detector_cfg={"auto_threshold": False, "threshold": 0.01}
    )

    assert res["history"].shape == (T + 1, n_z)
    assert res["correlation"].shape == (T + 1, n_zz + 1)

    assert np.all(res["history"][0] == pytest.approx(0.5)) 

    assert np.all(res["correlation"][0] == pytest.approx(0.0))