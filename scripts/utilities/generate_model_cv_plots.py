"""Generate cross-validation distribution plots for all models.

This is a wrapper module that bridges the main pipeline to the new visualization architecture.
"""

from pathlib import Path
from utils.io import load_all_models
from visualization_new.viz_factory import VisualizationFactory
from config import settings


def main():
    """Generate CV distribution plots for all trained models."""
    print("Generating cross-validation distribution plots...")
    
    # Load all trained models
    all_models = load_all_models()
    
    if not all_models:
        print("No models found to generate CV plots")
        return
    
    # Filter models that have CV results
    cv_models = {}
    for model_name, model_data in all_models.items():
        if isinstance(model_data, dict) and 'cv_scores' in model_data:
            cv_models[model_name] = model_data
    
    if not cv_models:
        print("No models with cross-validation results found")
        return
    
    print(f"Found {len(cv_models)} models with CV results")
    
    # Create visualization factory
    viz_factory = VisualizationFactory()
    
    # Generate CV distribution plots
    try:
        viz_factory.create_cv_distributions(
            models=cv_models,
            output_dir=settings.VISUALS_DIR / "cv_analysis"
        )
        print("✓ CV distribution plots generated successfully")
    except Exception as e:
        print(f"Error generating CV distribution plots: {e}")
        # Don't fail the entire pipeline for visualization errors
        pass


if __name__ == "__main__":
    main()