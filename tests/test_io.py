import pytest
import numpy as np
from pathlib import Path
from bav_dqs.io.data_manager import DataManager

# --- Fixtures Atualizadas ---

@pytest.fixture
def mock_config():
    return {
        "physics": {"mass_bare": 0.5},
        "experiment": {"id": "TEST_EXP_001", "schema_version": "1.0.0"}
    }

@pytest.fixture
def generic_sim_data():
    """Dados genéricos seguindo a nova estrutura de dicionários."""
    return {
        "group_name": "n_qubits_4",
        "run_id": "run_001",
        "datasets": {
            "t_full": np.linspace(0, 1, 10),
            "occ_full": np.random.rand(10, 4),
            "metrics": np.array([0.1, 0.2, 0.3])
        },
        "attributes": {
            "first_hit_step": 5,
            "status": "completed",
            "params": {"theta": 0.1}
        }
    }

# --- Testes de Integração Refatorados ---

def test_full_io_cycle(tmp_path, mock_config, generic_sim_data):
    """Testa o ciclo completo: DataManager -> Writer Genérico -> Reader Genérico."""
    h5_path = tmp_path / "test_refactored.h5"
    
    # 1. Inicialização e Escrita Genérica
    dm = DataManager(
        file_path=h5_path, 
        config=mock_config,
        experiment_id=mock_config["experiment"]["id"]
    )
    
    writer = dm.get_writer()
    writer.save_run(**generic_sim_data)
    
    # 2. Verificação de Leitura
    with dm.open_reader() as reader:
        # Testa metadados globais (recuperados do snapshot de inicialização)
        global_meta = reader.get_global_metadata()
        assert global_meta["experiment_id"] == "TEST_EXP_001"
        assert global_meta["config"]["physics"]["mass_bare"] == 0.5
        
        # Testa recuperação de Datasets (Numéricos)
        data = reader.get_run_data(
            group_name=generic_sim_data["group_name"], 
            run_id=generic_sim_data["run_id"]
        )
        assert "t_full" in data
        assert data["t_full"].shape == (10,)
        assert data["occ_full"].shape == (10, 4)
        
        # Testa recuperação de Atributos (Metadados da Run)
        attrs = reader.get_run_attributes(
            group_name=generic_sim_data["group_name"], 
            run_id=generic_sim_data["run_id"]
        )
        assert attrs["first_hit_step"] == 5
        assert attrs["params"]["theta"] == 0.1  # Verificando desserialização YAML

def test_writer_collision_prevention(tmp_path, mock_config, generic_sim_data):
    """Garante que o Writer impede sobrescrever o mesmo run_id no mesmo grupo."""
    h5_path = tmp_path / "collision.h5"
    dm = DataManager(h5_path, config=mock_config)
    writer = dm.get_writer()
    
    writer.save_run(**generic_sim_data)
    
    # Segunda tentativa com mesmo group e run_id deve falhar
    with pytest.raises(ValueError, match="Colisão de run_id"):
        writer.save_run(**generic_sim_data)

def test_reader_file_not_found():
    """Garante erro claro ao tentar ler arquivo inexistente via DataManager."""
    dm = DataManager("non_existent.h5")
    with pytest.raises(FileNotFoundError):
        dm.open_reader()

def test_complex_attribute_serialization(tmp_path, mock_config):
    """Testa se estruturas complexas (dict/list) são preservadas via YAML no HDF5."""
    h5_path = tmp_path / "complex_attrs.h5"
    dm = DataManager(h5_path, config=mock_config)
    writer = dm.get_writer()
    
    complex_attrs = {"nested": {"list": [1, 2, 3], "flag": True}}
    
    writer.save_run(
        group_name="test_group",
        run_id="run_1",
        datasets={"dummy": np.array([1])},
        attributes=complex_attrs
    )
    
    with dm.open_reader() as reader:
        recovered = reader.get_run_attributes("test_group", "run_1")
        assert recovered["nested"]["list"] == [1, 2, 3]
        assert recovered["nested"]["flag"] is True

def test_list_helpers(tmp_path, mock_config, generic_sim_data):
    """Testa os métodos de listagem de grupos e runs."""
    h5_path = tmp_path / "lists.h5"
    dm = DataManager(h5_path, config=mock_config)
    writer = dm.get_writer()
    
    writer.save_run(**generic_sim_data)
    
    with dm.open_reader() as reader:
        groups = reader.list_groups()
        assert generic_sim_data["group_name"] in groups
        
        runs = reader.list_runs(generic_sim_data["group_name"])
        assert generic_sim_data["run_id"] in runs
