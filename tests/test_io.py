import pytest
import numpy as np
from bav_dqs.utils.io.data_manager import DataManager

# --- Fixtures ---

@pytest.fixture
def mock_config():
    """Simulated configuration for HDF5 global metadata."""
    return {
        "physics": {"m": 0.5},
        "experiment": {"id": "TEST_EXPERIMENT", "schema_version": "1.0.7.preprint"}
    }

@pytest.fixture
def generic_sim_data():
    """Simulation data following the framework's standard dictionary structure."""
    return {
        "group_name": "n_qubits_4",
        "run_id": "run_001",
        "datasets": {
            "t_full": np.linspace(0, 1, 10),
            "occ_full": np.random.default_rng(seed = 137).random((10, 4)),
            "metrics": np.array([0.1, 0.2, 0.3])
        },
        "attributes": {
            "first_hit_step": 5,
            "status": "completed",
            "params": {"theta": 0.1}
        }
    }

# --- Integration Tests ---

def test_full_io_cycle(tmp_path, mock_config, generic_sim_data):
    """
    Validates the complete persistence cycle: DataManager -> Writer -> Reader.
    Ensures that metadata, numeric datasets, and complex attributes are preserved.
    """
    h5_path = tmp_path / "test_refactored.h5"
    
    # DataManager Initialization
    dm = DataManager(
        file_path=h5_path, 
        config=mock_config,
        experiment_id=mock_config["experiment"]["id"]
    )
    
    # Data writing
    writer = dm.get_writer()
    writer.save_run(**generic_sim_data)
    
    # Verification via Reader
    with dm.open_reader() as reader:
        # Global Metadata
        global_meta = reader.get_global_metadata()
        assert global_meta["experiment_id"] == "TEST_EXPERIMENT"
        assert global_meta["config"]["physics"]["m"] == pytest.approx(0.5)
        
        # Datasets (NumPy Arrays)
        data = reader.get_run_data(
            group_name=generic_sim_data["group_name"], 
            run_id=generic_sim_data["run_id"]
        )
        np.testing.assert_array_equal(data["t_full"], generic_sim_data["datasets"]["t_full"])
        assert data["occ_full"].shape == (10, 4)
        
        # Attributes (YAML/Complex Serialization)
        attrs = reader.get_run_attributes(
            group_name=generic_sim_data["group_name"], 
            run_id=generic_sim_data["run_id"]
        )
        assert attrs["first_hit_step"] == 5
        assert attrs["params"]["theta"] == pytest.approx(0.1)

def test_writer_collision_prevention(tmp_path, mock_config, generic_sim_data):
    """Ensures that Writer throws an error when attempting to overwrite an existing run_id."""
    h5_path = tmp_path / "collision.h5"
    dm = DataManager(h5_path, config=mock_config)
    writer = dm.get_writer()
    
    writer.save_run(**generic_sim_data)
    
    run_id = generic_sim_data["run_id"]
    group = generic_sim_data["group_name"]

    expected_error = f"Collision with run_id: '{run_id}' already exists in '{group}'"   
    with pytest.raises(ValueError, match=expected_error):
        writer.save_run(**generic_sim_data)

def test_reader_file_not_found():
    """Checks if the DataManager raises a FileNotFoundError for non-existent files."""
    dm = DataManager("non_existent_file_999.h5")
    with pytest.raises(FileNotFoundError):
        dm.open_reader()

def test_complex_attribute_serialization(tmp_path, mock_config):
    """
    Checks if nested dictionaries and lists are correctly serialized
    in HDF5 format (usually via an internal YAML or JSON string).
    """
    h5_path = tmp_path / "complex_attrs.h5"
    dm = DataManager(h5_path, config=mock_config)
    writer = dm.get_writer()
    
    complex_attrs = {
        "nested": {"list": [1, 2, 3], "flag": True},
        "metadata": "scientific_data"
    }
    
    writer.save_run(
        group_name="test_group",
        run_id="run_1",
        datasets={"dummy": np.array([1.0])},
        attributes=complex_attrs
    )
    
    with dm.open_reader() as reader:
        recovered = reader.get_run_attributes("test_group", "run_1")
        assert recovered["nested"]["list"] == [1, 2, 3]
        assert recovered["nested"]["flag"] is True
        assert isinstance(recovered["nested"]["list"], list)

def test_listing_methods(tmp_path, mock_config, generic_sim_data):
    """Tests the Reader's ability to list the file hierarchy (groups and runs)."""
    h5_path = tmp_path / "lists.h5"
    dm = DataManager(h5_path, config=mock_config)
    writer = dm.get_writer()
    
    writer.save_run(**generic_sim_data)
    
    with dm.open_reader() as reader:
        groups = reader.list_groups()
        assert generic_sim_data["group_name"] in groups
        
        runs = reader.list_runs(generic_sim_data["group_name"])
        assert generic_sim_data["run_id"] in runs
