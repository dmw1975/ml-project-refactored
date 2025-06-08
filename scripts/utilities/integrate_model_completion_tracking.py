#!/usr/bin/env python3
"""Integrate model completion tracking into training functions."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


def update_file_with_tracking(file_path, model_type, function_patterns):
    """Update a model file to include completion tracking."""
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Add import if not present
    if "from src.pipelines.state_manager import get_state_manager" not in content:
        # Find imports section
        import_end = content.find("\n\n\ndef")
        if import_end > 0:
            new_import = "from src.pipelines.state_manager import get_state_manager\n"
            content = content[:import_end] + new_import + content[import_end:]
    
    # Add tracking to each function
    for func_name, save_pattern in function_patterns:
        # Find the function
        func_start = content.find(f"def {func_name}(")
        if func_start < 0:
            continue
            
        # Find where model is saved
        save_index = content.find(save_pattern, func_start)
        if save_index < 0:
            continue
            
        # Find the end of the save line
        save_line_end = content.find("\n", save_index)
        
        # Check if tracking already exists
        if "get_state_manager().increment_completed_models" not in content[save_index:save_line_end+100]:
            # Insert tracking code
            indent = "    "  # Assuming standard indentation
            tracking_code = f"\n{indent}{indent}# Report model completion\n{indent}{indent}get_state_manager().increment_completed_models('{model_type}')"
            content = content[:save_line_end] + tracking_code + content[save_line_end:]
    
    # Write back
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"✓ Updated {file_path.name}")


def main():
    """Update all model files with completion tracking."""
    
    models_dir = project_root / "src" / "models"
    
    # Define model files and their patterns
    updates = [
        (
            models_dir / "linear_regression.py",
            "linear_regression",
            [
                ("train_linear_model", "save_model(model,"),
                ("train_all_models", "save_model(")
            ]
        ),
        (
            models_dir / "elastic_net.py",
            "elasticnet",
            [
                ("train_elasticnet_model", "save_model(model,"),
                ("train_elasticnet_models", "save_model(")
            ]
        ),
        (
            models_dir / "xgboost_categorical.py",
            "xgboost",
            [
                ("train_xgboost_model", "save_model(xgb_model,"),
                ("train_xgboost_categorical_model", "save_model(xgb_model,"),
                ("train_xgboost_models", "save_model("),
                ("train_xgboost_categorical_models", "save_model(")
            ]
        ),
        (
            models_dir / "lightgbm_categorical.py",
            "lightgbm",
            [
                ("train_lightgbm_model", "save_model(lgb_model,"),
                ("train_lightgbm_categorical_model", "save_model(lgb_model,"),
                ("train_lightgbm_models", "save_model("),
                ("train_lightgbm_categorical_models", "save_model(")
            ]
        ),
        (
            models_dir / "catboost_categorical.py",
            "catboost",
            [
                ("train_catboost_model", "save_model(cb_model,"),
                ("train_catboost_categorical_model", "save_model(cb_model,"),
                ("train_catboost_models", "save_model("),
                ("train_catboost_categorical_models", "save_model(")
            ]
        )
    ]
    
    print("Integrating model completion tracking...\n")
    
    for file_path, model_type, patterns in updates:
        if file_path.exists():
            update_file_with_tracking(file_path, model_type, patterns)
        else:
            print(f"⚠️  {file_path.name} not found")
    
    print("\n✓ Model completion tracking integrated")
    print("\nModels will now report completion to the pipeline state manager.")
    print("This ensures metrics table generation waits for all models to complete.")


if __name__ == "__main__":
    main()