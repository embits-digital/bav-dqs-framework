import pytest
import yaml
from pathlib import Path
from bav_dqs.utils.config_manager import ConfigManager

# --- Fixtures ---

@pytest.fixture
def mock_yaml_content():
    return {
        "experiment": {"id": "TEST_001"},
        "physics": {"mass_bare": 0.5, "widths": [20, 40]},
        "backend": {"logging": {"enabled": True}}
    }

@pytest.fixture
def temp_yaml_file(tmp_path, mock_yaml_content):
    """Creates a real YAML file in a temporary system directory."""
    cfg_path = tmp_path / "test_config.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(mock_yaml_content, f)
    return str(cfg_path)

# --- Unit tests ---

def test_config_loading(temp_yaml_file):
    """Checks if the ConfigManager loads correctly from disk."""
    cfg = ConfigManager.from_yaml(temp_yaml_file)
    assert cfg.get("experiment.id") == "TEST_001"
    assert cfg.get("physics.widths")[0] == 20

def test_nested_key_access(mock_yaml_content):
    """Validates the resolution of paths with dots (dotted paths)."""
    cfg = ConfigManager(mock_yaml_content)
    assert cfg.get("backend.logging.enabled") is True

def test_path_validation_success():
    """Validates that existing directories (like the current '.') pass the test."""
    # The '.' directory aways exists, then it must return to the absolute path.
    path = ConfigManager.require_path(".")
    assert Path(path).is_absolute()
    assert Path(path).exists()

def test_path_validation_failure():
    """Ensures that non-existent paths trigger a file error."""
    with pytest.raises(FileNotFoundError):
        ConfigManager.require_path("/tmp/do_not_exists_bav_dqs")

def test_cli_update_override(mock_yaml_content):
    """Validates whether updating via the dictionary (CLI) overwrites the YAML."""
    cfg = ConfigManager(mock_yaml_content)
    # Simulate --physics.mass_bare 0.9 via CLI
    cli_updates = {"physics": {"mass_bare": 0.9}}
    
    cfg.update_from_args({"manual_override": "active"})
    assert cfg.get("manual_override") == "active"

def test_wrong_type_access(mock_yaml_content):
    """Guarantees an error when attempting to access the depth of a value that is not a dict."""
    cfg = ConfigManager(mock_yaml_content)
    # mass_bare is a float, it cannot have subkeys.
    with pytest.raises(TypeError, match="Expected dict at path segment"):
        cfg.get("physics.mass_bare.invalid_sub_key")

def test_missing_key_error(mock_yaml_content):
    """Ensures that missing keys will trigger a KeyError."""
    cfg = ConfigManager(mock_yaml_content)
    with pytest.raises(KeyError, match="Missing required key path"):
        cfg.get("physics.gravity")
