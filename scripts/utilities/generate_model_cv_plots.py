"""Generate cross-validation distribution plots for all models.

This is a wrapper module that bridges the main pipeline to the new visualization architecture.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.io import load_all_models
from src.visualization.plots.cv_distributions import plot_cv_distributions
from src.visualization.core.interfaces import VisualizationConfig
from src.config import settings


def main():
    """Generate CV distribution plots for all trained models."""
    print("Generating cross-validation distribution plots...")
    
    # Load all trained models
    all_models = load_all_models()
    
    if not all_models:
        print("No models found to generate CV plots")
        return
    
    # Filter models that have CV results
    cv_models = []
    for model_name, model_data in all_models.items():
        if isinstance(model_data, dict) and 'cv_scores' in model_data:
            cv_models.append(model_data)
    
    if not cv_models:
        print("No models with cross-validation results found")
        return
    
    print(f"Found {len(cv_models)} models with CV results")
    
    # Generate CV distribution plots
    try:
        output_dir = settings.VISUALIZATION_DIR / "performance" / "cv_distributions"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        config = VisualizationConfig(
            output_dir=output_dir,
            format="png",
            dpi=300,
            save=True,
            show=False
        )
        
        # Call the actual function that exists
        plot_cv_distributions(cv_models, config)
        print("✓ CV distribution plots generated successfully")
    except Exception as e:
        print(f"Error generating CV distribution plots: {e}")
        import traceback
        traceback.print_exc()
        # Don't fail the entire pipeline for visualization errors
        pass


if __name__ == "__main__":
    main()