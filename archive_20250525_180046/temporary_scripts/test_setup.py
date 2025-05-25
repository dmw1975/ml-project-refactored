"""Test script to verify project setup."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

from config import settings
from utils import io, helpers

def test_configuration():
    """Test that configuration is correctly set up."""
    print("Testing configuration setup...")
    print(f"Project root: {settings.ROOT_DIR}")
    print(f"Data directories:")
    print(f"  - Raw: {settings.RAW_DATA_DIR}")
    print(f"  - Processed: {settings.PROCESSED_DATA_DIR}")
    
    print(f"\nOutput directories:")
    print(f"  - Models: {settings.MODEL_DIR}")
    print(f"  - Visualizations: {settings.VISUALIZATION_DIR}")
    
    print("\nAll directories exist!")
    return True

def test_io_utilities():
    """Test I/O utilities."""
    print("\nTesting I/O utilities...")
    
    # Test directory creation
    test_dir = settings.OUTPUT_DIR / "test"
    io.ensure_dir(test_dir)
    print(f"Created test directory: {test_dir}")
    
    # Test saving and loading a dummy object
    dummy_data = {"test": "data", "value": 42}
    save_path = io.save_model(dummy_data, "test_data.pkl", test_dir)
    print(f"Saved test data to: {save_path}")
    
    loaded_data = io.load_model("test_data.pkl", test_dir)
    print(f"Loaded test data: {loaded_data}")
    
    assert dummy_data == loaded_data, "Saved and loaded data don't match!"
    print("I/O utilities working correctly!")
    return True

if __name__ == "__main__":
    print("Testing project setup...")
    
    # Test configuration
    config_ok = test_configuration()
    
    # Test IO utilities
    io_ok = test_io_utilities()
    
    if config_ok and io_ok:
        print("\nAll tests passed! Project is correctly set up.")
    else:
        print("\nSome tests failed. Check the output above for details.")