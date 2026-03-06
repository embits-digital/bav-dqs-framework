def test_import():
    """Checks if the base package is importable and has metadata."""
    import bav_dqs
    
    assert hasattr(bav_dqs, "__version__")
    assert isinstance(bav_dqs.__version__, str)
    assert bav_dqs.__version__ is not None