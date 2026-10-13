import pytest
import numpy as np
from bav_dqs.core.detectors.boundary_detector import BoundaryDetector, BoundaryDetectorCfg

# --- Fixtures ---

@pytest.fixture
def base_cfg():
    return BoundaryDetectorCfg(threshold=0.5, edge_window=2, edge_persistence=3)

@pytest.fixture
def detector(base_cfg):
    return BoundaryDetector(cfg=base_cfg, vector_size=10)

# --- Unit Tests ---

def test_initialization_and_validation():
    """Ensures that invalid configurations throw appropriate errors."""
    with pytest.raises(ValueError, match="vector_size must be >= 2"):
        BoundaryDetector(BoundaryDetectorCfg(0.5, 1, 1), vector_size=1)
        
    with pytest.raises(ValueError, match="Invalid edge_window"):
        BoundaryDetector(BoundaryDetectorCfg(0.5, 11, 1), vector_size=10)

    with pytest.raises(ValueError, match="persistence must be >= 1"):
        BoundaryDetector(BoundaryDetectorCfg(0.5, 2, 0), vector_size=10)

def test_baseline_initialization(detector: BoundaryDetector):
    """The first step should not trigger any hits, only define the baseline."""
    size = detector.size
    data = np.full(size, 0.1, dtype=float)
    d_left, d_right = detector.update(data, step=0)
    
    assert d_left ==pytest.approx(0.0)
    assert d_right ==pytest.approx(0.0)
    # Baselines are per-site profiles (size = edge_window), not scalars.
    np.testing.assert_allclose(detector._baselines["left"], [0.1, 0.1])
    np.testing.assert_allclose(detector._baselines["right"], [0.1, 0.1])

    assert detector.results.first_hit_step is None
    assert detector.results.first_side is None

def test_persistence_logic(detector: BoundaryDetector):
    """Validates that the hit is only confirmed after N consecutive steps (Eq. 3)."""
    detector.update(np.zeros(10), step=0)
    
    # Steps 1 and 2: Above the threshold (0.5), but persistence is 3.
    data_high = np.ones(10)
    detector.update(data_high, step=1)
    detector.update(data_high, step=2)
    assert detector.results.first_hit_step is None
    
    # Step 3: Hit confirmed
    detector.update(data_high, step=3)
    # hit_index = step - edge_persistence -> 3 - 3 = 0
    assert detector.results.first_hit_step == 0
    assert detector.results.first_side == "both"

def test_noise_reset_persistence(detector: BoundaryDetector):
    """Ensures that a value below the threshold resets the persistence counter."""
    detector.update(np.zeros(10), step=0) # Baseline
    
    # 2 steps above
    detector.update(np.ones(10), step=1)
    detector.update(np.ones(10), step=2)
    
    # 1 noise step (return to baseline)
    detector.update(np.zeros(10), step=3)
    assert detector._counters["left"] == 0
    
    # 3 more steps needed now.
    detector.update(np.ones(10), step=4)
    detector.update(np.ones(10), step=5)
    detector.update(np.ones(10), step=6)
    
    assert detector.results.first_hit_step == 6 - 3 # 3

def test_side_specific_detection(base_cfg):
    """Tests when the edge only touches one side of the vector."""
    detector = BoundaryDetector(base_cfg, vector_size=10)
    detector.update(np.zeros(10), step=0) # Baseline
    
    # It only changes the left side (the first 2 elements of the window).
    left_spike = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    for i in range(1, 4):
        detector.update(left_spike, step=i)
        
    assert detector.results.step_left == 0
    assert detector.results.step_right is None
    assert detector.results.first_side == "left"

def test_simultaneous_hit_assignment(base_cfg):
    """Tests the 'both' assignment logic for detections in the same step."""
    detector = BoundaryDetector(base_cfg, vector_size=10)
    detector.update(np.zeros(10), step=0)
    
    # It creates a hit on both sides at the same time.
    data = np.ones(10)
    for i in range(1, 4):
        detector.update(data, step=i)
        
    assert detector.results.first_side == "both"
    assert detector.results.step_left == detector.results.step_right

def test_data_shape_mismatch(detector: BoundaryDetector):
    """Ensures that the detector validates the size of the input vector."""
    with pytest.raises(ValueError, match="Data shape mismatch"):
        detector.update(np.zeros(5), step=1) # Expected 10

def test_dynamic_threshold_update(detector: BoundaryDetector):
    """Garante que o detector respeita a recalibração de threshold (5-sigma)."""
    detector.update(np.zeros(10), step=0) # Baseline em 0.0
    
    # Caso 1: Com threshold original (0.5), um sinal de 0.4 NÃO deve disparar contador
    data_low = np.full(10, 0.4)
    detector.update(data_low, step=1)
    assert detector._counters["left"] == 0
    
    # Caso 2: Atualizamos dinamicamente para 0.3 (Simulando calibração em hardware ruidoso)
    detector.update_threshold(0.3)
    detector.update(data_low, step=2)
    
    # Agora o contador deve subir, pois 0.4 > 0.3
    assert detector._counters["left"] == 1

def test_edge_window_sensitivity_max(base_cfg):
    """Valida se o detector usa o valor MÁXIMO da janela (sensibilidade a picos)."""
    # Janela de tamanho 2 (base_cfg.edge_window=2)
    detector = BoundaryDetector(base_cfg, vector_size=10)
    detector.update(np.zeros(10), step=0) # Baseline
    
    # Apenas o SEGUNDO qubit da janela esquerda sobe (índice 1).
    # Se usasse média (mean), o valor seria 0.4 (não dispararia thr=0.5).
    # Como usa MÁX, o valor é 0.8 (deve disparar!).
    spike = np.zeros(10)
    spike[1] = 0.8 
    
    detector.update(spike, step=1)
    assert detector._counters["left"] == 1

def test_max_of_deviations_with_nonzero_baseline():
    """Marker is max_i |n_i(t) - n_i(0)|, NOT |max_i n_i(t) - max_i n_i(0)|.

    Baseline [0.8, 0.0]; later [0.8, 0.6]: the max of the window is unchanged (0.8),
    so a deviation-of-max marker would stay at 0 and miss the front, while the
    max-of-deviations marker sees 0.6.
    """
    cfg = BoundaryDetectorCfg(threshold=0.5, edge_window=2, edge_persistence=1)
    det = BoundaryDetector(cfg, vector_size=10)
    base = np.zeros(10)
    base[0] = 0.8
    det.update(base, step=0)
    later = base.copy()
    later[1] = 0.6
    d_left, _ = det.update(later, step=1)
    assert d_left == pytest.approx(0.6)
    assert det._counters["left"] == 1


def test_decay_below_baseline_is_also_a_deviation():
    """Absolute deviation: a drop relative to t=0 counts as a deviation as well."""
    cfg = BoundaryDetectorCfg(threshold=0.5, edge_window=1, edge_persistence=1)
    det = BoundaryDetector(cfg, vector_size=6)
    start = np.zeros(6)
    start[0] = 0.9
    det.update(start, step=0)
    d_left, _ = det.update(np.zeros(6), step=1)
    assert d_left == pytest.approx(0.9)


@pytest.mark.parametrize("n, excluded, expected", [
    (4, (2,), 1), (6, (3,), 2), (8, (4,), 3), (20, (10,), 9),   # single excluded site
    (10, (3, 6), 3),                                              # excluded support spans several sites
    (10, (), 10),                                                 # nothing to exclude
])
def test_max_valid_edge_window(n, excluded, expected):
    assert BoundaryDetector.max_valid_edge_window(n, excluded) == expected


def test_validate_edge_windows_rejects_excluded_sites():
    BoundaryDetector.validate_edge_windows(8, 3, excluded_sites=(4,))  # ok
    with pytest.raises(ValueError, match="contain excluded site"):
        BoundaryDetector.validate_edge_windows(4, 3, excluded_sites=(2,))
    with pytest.raises(ValueError, match="contain excluded site"):
        BoundaryDetector.validate_edge_windows(6, 3, excluded_sites=(3,))


def test_detector_reset_integrity(detector: BoundaryDetector):
    """Garante que o reset limpa todos os estados internos para uma nova run."""
    detector.update(np.zeros(10), step=0)
    detector.update(np.ones(10), step=1)
    detector.update(np.ones(10), step=2)
    detector.update(np.ones(10), step=3) # Hit confirmado
    
    assert detector.results.first_hit_step is not None
    
    detector.reset()
    
    assert detector._baselines["left"] is None
    assert detector._counters["left"] == 0
    assert detector.results.first_hit_step is None
    assert detector.results.step_left is None

