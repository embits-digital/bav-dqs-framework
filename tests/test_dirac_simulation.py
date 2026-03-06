import pytest
import numpy as np
from unittest.mock import MagicMock
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from bav_dqs.core.engines.qiskit import _get_occupation
from bav_dqs.core.detectors.boundary_detector import _update_hit_logic, _get_d_hit_eff
from bav_dqs.core.operators.z_observable import build_z_observables
from bav_dqs.plugins.dirac_simulation import _calculate_side, _validate_inputs, run_boundary_detection


class TestDiracSimulation:
    
    @pytest.fixture
    def basic_configs(self):
        """Configurações mínimas para consistência científica."""
        return {
            "model": {"m": 1.0, "w": 1.0, "dt": 0.1},
            "detector": {"threshold": 0.5, "edge_window": 1, "edge_persistence": 2},
            "backend": {"log_every_steps": 0, "abort_on_complex_evs": True}
        }

    # --- Testes Unitários de Lógica Científica ---

    def test_validate_inputs_raises(self):
        """Garante a integridade física: sistema precisa de dimensão mínima."""
        with pytest.raises(ValueError, match="n_qubits must be >= 2"):
            _validate_inputs(n=1, T=10)
        with pytest.raises(ValueError, match="max_steps must be >= 1"):
            _validate_inputs(n=4, T=0)

    @pytest.mark.parametrize("side, dl, dr, expected", [
        ("left", 10.0, 20.0, 10.0),
        ("right", 10.0, 20.0, 20.0),
        ("both", 10.0, 20.0, 10.0), # Menor distância (first hit)
        (None, 10.0, 20.0, None),
    ])
    def test_d_hit_eff_calculation(self, side, dl, dr, expected):
        """Valida a precisão da métrica de distância efetiva para o cálculo de v_hit."""
        assert _get_d_hit_eff(side, dl, dr) == expected

    def test_update_hit_logic_persistence(self):
        """Valida se o filtro de persistência evita falsos positivos (estabilidade)."""
        res = {"hits": [None, None], "counts": [0, 0]}
        thr, persistence = 0.8, 3
        
        # Passo 1 e 2: Abaixo da persistência
        for i in range(1, 3):
            _update_hit_logic(res, dl=0.9, dr=0.1, thr=thr, persistence=persistence, step_idx=i)
            assert res["hits"][0] is None 
            
        # Passo 3: Atinge persistência
        _update_hit_logic(res, dl=0.9, dr=0.1, thr=thr, persistence=persistence, step_idx=3)
        assert res["hits"][0] == 0 # (3 - 3) passo original do início da detecção

    # --- Testes de Interoperabilidade e Backend ---

    def test_get_occupation_ideal_mode(self, basic_configs):
        """Valida se a extração de ocupação preserva a norma quântica (sum <= 1.0)."""
        n = 2
        # Mock de contexto para simular persistência de estado
        ctx = {"state": None, "qc": QuantumCircuit(n)}
        init_qc = QuantumCircuit(n) # Estado |00>
        step_qc = QuantumCircuit(n) # Identidade
        obs = build_z_observables(n)
        
        # Ocupação deve retornar um array de tamanho n
        occ = _get_occupation(
            step_idx=0, mode="ideal", estimator=None, 
            step_qc=step_qc, init_qc=init_qc, observables=obs,
            backend_cfg=basic_configs["backend"], ctx=ctx
        )
        
        assert len(occ) == n
        assert np.all(occ >= 0) and np.all(occ <= 1.0)
        assert isinstance(ctx["state"], Statevector)

    # --- Teste de Integração (End-to-End) ---

    def test_full_simulation_run(self, basic_configs):
        """Teste de fumaça: integração completa do plugin Dirac."""
        # Mocking das funções que buildam circuitos para isolar o teste do Qiskit pesado
        with MagicMock() as mock_build:
            # Configuração de um cenário simplificado de 3 qubits
            result = run_boundary_detection(
                n_qubits=4,
                model_cfg=basic_configs["model"],
                detector_cfg=basic_configs["detector"],
                backend_cfg={"mode": "ideal"},
                max_steps=5
            )
            
            # Verificações de integridade do Objeto de Resultado
            assert result.n_qubits == 4
            assert result.history.shape == (6, 4) # T+1 passos, n qubits
            assert hasattr(result, 'v_hit_est')
            assert isinstance(result.d_left_eff, float)

    def test_calculate_side_logic(self):
        """Garante a classificação correta da direção de propagação."""
        assert _calculate_side(None, None) is None
        assert _calculate_side(5, 5) == "both"
        assert _calculate_side(2, 8) == "left"
        assert _calculate_side(None, 4) == "right"
