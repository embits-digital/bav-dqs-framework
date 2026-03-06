import pytest
import numpy as np
import h5py
from pathlib import Path
from bav_dqs.io.data_manager import DataManager
from bav_dqs.io.writer import Writer
from bav_dqs.io.reader import Reader

# --- Fixtures de Dados de Simulação ---

@pytest.fixture
def mock_config():
    return {
        "physics": {"mass_bare": 0.5, "time_step_dt": 0.01},
        "experiment": {"id": "TEST_EXP_001", "schema_version": "1.0.0"}
    }

@pytest.fixture
def sample_sim_data():
    """Generates arrays that meet the size requirements of your Writer."""
    t = np.linspace(0, 1, 10)
    return {
        "n_qubits": 4,
        "run_id": "run_unit_test",
        "t_full": t,
        "occ_full": np.zeros((10, 4)),
        "metric_full_left": np.random.rand(10),
        "metric_full_right": np.random.rand(10),
        "t_half": t,
        "occ_half": np.zeros((10, 4)),
        "metric_half_left": np.random.rand(10),
        "metric_half_right": np.random.rand(10),
        "metadata": {"first_hit_full": 5}
    }

# --- Testes de Integração ---

def test_full_io_cycle(tmp_path, mock_config, sample_sim_data):
    """Test the complete flow: Creates DataManager -> Write -> Read."""
    h5_path = tmp_path / "test_data.h5"
    
    # 1. Initialization and Writing
    dm = DataManager(
        file_path=h5_path, 
        config=mock_config,
        experiment_id=mock_config["experiment"]["id"]
    )
    
    writer = dm.get_writer()
    writer.save_simulation_101(**sample_sim_data)
    
    # 2. Reading Verification
    with dm.open_reader() as reader:
        # Testing Configuration Recovery (the new method)
        recovered_cfg = reader.get_config()
        assert recovered_cfg["physics"]["mass_bare"] == 0.5
        
        # Test root metadata
        root_meta = reader.root_metadata()
        assert root_meta["experiment_id"] == "TEST_EXP_001"
        
        # Testing numerical data
        widths = list(reader.iter_widths())
        assert 4 in widths
        
        t_recovered = reader.t_full(n_qubits=4, run_id="run_unit_test")
        assert len(t_recovered) == 10
        assert isinstance(t_recovered, np.ndarray)

def test_writer_collision_prevention(tmp_path, mock_config, sample_sim_data):
    """Ensures that Writer prevents overwriting the same run_id."""
    h5_path = tmp_path / "collision.h5"
    dm = DataManager(h5_path, config=mock_config)
    writer = dm.get_writer()
    
    # First writing
    writer.save_simulation_101(**sample_sim_data)
    
    # A second attempt with the same run_id should fail.
    with pytest.raises(ValueError, match="run_id collision"):
        writer.save_simulation_101(**sample_sim_data)

def test_reader_file_not_found():
    """It guarantees a clear error if you try to read a non-existent file."""
    dm = DataManager("do_not_exists.h5")
    with pytest.raises(FileNotFoundError):
        dm.open_reader()

def test_writer_validation_logic(tmp_path, mock_config, sample_sim_data):
    """Tests whether the Writer's internal validation functions (_require_1d) works."""
    h5_path = tmp_path / "validation.h5"
    dm = DataManager(h5_path, config=mock_config)
    writer = dm.get_writer()
    
    # Change data to become invalid (different size)
    invalid_data = sample_sim_data.copy()
    invalid_data["t_full"] = np.array([1, 2, 3]) # mismatch with the other arrays of 10 positions
    
    with pytest.raises(ValueError, match="1D series length mismatch"):
        writer.save_simulation_101(**invalid_data)

def test_reader_iteration_helpers(tmp_path, mock_config, sample_sim_data):
    """Tests iter_widths and iter_runs."""
    h5_path = tmp_path / "iter.h5"
    dm = DataManager(h5_path, config=mock_config)
    writer = dm.get_writer()
    
    writer.save_simulation_101(**sample_sim_data)
    
    with dm.open_reader() as reader:
        widths = list(reader.iter_widths())
        assert widths == [4]
        
        runs = list(reader.iter_runs(4))
        assert "run_unit_test" in runs

