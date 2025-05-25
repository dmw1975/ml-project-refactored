"""Test script to verify sector model functionality."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

from config import settings

def test_sector_models():
    """Test sector model training and evaluation."""
    print("Testing sector-specific model functionality...")
    
    # Import and run sector model training with a small subset
    from models.sector_models import run_sector_models
    
    # Run with a smaller test size for faster testing
    print("\nTraining sector models with test parameters...")
    try:
        sector_models = run_sector_models(test_size=0.5)
        print("Sector model training successful.")
        
        # Test evaluation
        print("\nTesting sector model evaluation...")
        from models.sector_models import evaluate_sector_models
        eval_results = evaluate_sector_models(sector_models)
        print("Sector model evaluation successful.")
        
        # Test importance analysis
        print("\nTesting sector model feature importance analysis...")
        from models.sector_models import analyze_sector_importance
        # Use fewer repeats for faster testing
        importance_results, consolidated = analyze_sector_importance(sector_models, n_repeats=2)
        print("Sector model importance analysis successful.")
        
        # Test visualization
        print("\nTesting sector model visualization...")
        from visualization.sector_plots import visualize_sector_models
        visualize_sector_models(run_all=False)
        print("Sector model visualization successful.")
        
        print("\nAll sector model tests completed successfully!")
        
    except Exception as e:
        print(f"\nError during sector model testing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    test_sector_models()