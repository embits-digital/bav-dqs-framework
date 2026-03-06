import pytest
import numpy as np
from bav_dqs.core.detectors.boundary_detector import BoundaryDetector, BoundaryDetectorCfg

# --- Fixtures ---

@pytest.fixture
def base_cfg():
    return BoundaryDetectorCfg(threshold=0.5, edge_window=2, persistence=3)

@pytest.fixture
def detector(base_cfg):
    return BoundaryDetector(cfg=base_cfg, vector_size=10)

# --- Testes Unitários ---

def test_initialization_and_validation():
    """Garante que configurações inválidas lancem erros apropriados."""
    with pytest.raises(ValueError, match="vector_size must be >= 2"):
        BoundaryDetector(BoundaryDetectorCfg(0.5, 1, 1), vector_size=1)
        
    with pytest.raises(ValueError, match="Invalid edge_window"):
        BoundaryDetector(BoundaryDetectorCfg(0.5, 11, 1), vector_size=10)

    with pytest.raises(ValueError, match="persistence must be >= 1"):
        BoundaryDetector(BoundaryDetectorCfg(0.5, 2, 0), vector_size=10)

def test_baseline_initialization(detector):
    """O primeiro passo não deve disparar hits, apenas definir o baseline."""
    data = np.zeros(10)
    d_left, d_right = detector.update(data, step=0)
    
    assert d_left == 0.0
    assert d_right == 0.0
    assert detector._baselines["left"] == 0.0
    assert detector.results.first_hit_step is None

def test_persistence_logic(detector):
    """Valida que o hit só é confirmado após N passos consecutivos (Eq. 3)."""
    # Baseline em 0.0
    detector.update(np.zeros(10), step=0)
    
    # Passo 1 e 2: Acima do threshold (0.5), mas persistência é 3
    data_high = np.ones(10)
    detector.update(data_high, step=1)
    detector.update(data_high, step=2)
    assert detector.results.first_hit_step is None
    
    # Passo 3: Hit confirmado
    detector.update(data_high, step=3)
    # hit_index = step - persistence -> 3 - 3 = 0
    assert detector.results.first_hit_step == 0
    assert detector.results.first_side == "both"

def test_noise_reset_persistence(detector):
    """Garante que um valor abaixo do threshold reseta o contador de persistência."""
    detector.update(np.zeros(10), step=0) # Baseline
    
    # 2 passos acima
    detector.update(np.ones(10), step=1)
    detector.update(np.ones(10), step=2)
    
    # 1 passo de ruído (volta para o baseline)
    detector.update(np.zeros(10), step=3)
    assert detector._counters["left"] == 0
    
    # Precisa de mais 3 passos agora
    detector.update(np.ones(10), step=4)
    detector.update(np.ones(10), step=5)
    detector.update(np.ones(10), step=6)
    
    assert detector.results.first_hit_step == 6 - 3 # 3

def test_side_specific_detection(base_cfg):
    """Testa quando a borda atinge apenas um lado do vetor."""
    detector = BoundaryDetector(base_cfg, vector_size=10)
    detector.update(np.zeros(10), step=0) # Baseline
    
    # Altera apenas o lado esquerdo (primeiros 2 elementos da janela)
    left_spike = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    for i in range(1, 4):
        detector.update(left_spike, step=i)
        
    assert detector.results.step_left == 0
    assert detector.results.step_right is None
    assert detector.results.first_side == "left"

def test_simultaneous_hit_assignment(base_cfg):
    """Testa a lógica de atribuição 'both' para detecções no mesmo passo."""
    detector = BoundaryDetector(base_cfg, vector_size=10)
    detector.update(np.zeros(10), step=0)
    
    # Provoca hit nos dois lados ao mesmo tempo
    data = np.ones(10)
    for i in range(1, 4):
        detector.update(data, step=i)
        
    assert detector.results.first_side == "both"
    assert detector.results.step_left == detector.results.step_right

def test_data_shape_mismatch(detector):
    """Garante que o detector valide o tamanho do vetor de entrada."""
    with pytest.raises(ValueError, match="Data shape mismatch"):
        detector.update(np.zeros(5), step=1) # Esperado 10
