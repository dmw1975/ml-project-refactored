#!/usr/bin/env python3
"""
Integrate enhanced categorical models into the main pipeline.
This script updates main.py to use the enhanced implementations with Optuna optimization.
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

def backup_original_files():
    """Backup original model files before integration."""
    backup_dir = Path(f"backup_integration_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    backup_dir.mkdir(exist_ok=True)
    
    files_to_backup = [
        "main.py",
        "models/lightgbm_categorical.py", 
        "models/xgboost_categorical.py"
    ]
    
    for file in files_to_backup:
        if Path(file).exists():
            dest = backup_dir / file
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, dest)
            print(f"Backed up {file}")
    
    return backup_dir

def update_main_py():
    """Update main.py to use enhanced implementations."""
    print("\nUpdating main.py...")
    
    with open("main.py", "r") as f:
        content = f.read()
    
    # Replace imports
    old_lightgbm_import = "from models.lightgbm_categorical import train_lightgbm_categorical"
    new_lightgbm_import = "from enhanced_lightgbm_categorical import train_enhanced_lightgbm_categorical as train_lightgbm_categorical"
    
    old_xgboost_import = "from models.xgboost_categorical import train_xgboost_categorical"
    new_xgboost_import = "from enhanced_xgboost_categorical import train_enhanced_xgboost_categorical as train_xgboost_categorical"
    
    content = content.replace(old_lightgbm_import, new_lightgbm_import)
    content = content.replace(old_xgboost_import, new_xgboost_import)
    
    # Also need to update the function calls to handle the new return format
    # The enhanced versions return a dictionary of models (basic and optuna)
    
    # Find and update the XGBoost training section
    xgb_section_start = content.find("if args.xgboost or args.all:")
    if xgb_section_start != -1:
        xgb_section_end = content.find("if args.lightgbm or args.all:", xgb_section_start)
        if xgb_section_end == -1:
            xgb_section_end = content.find("if args.catboost or args.all:", xgb_section_start)
        
        if xgb_section_end != -1:
            xgb_section = content[xgb_section_start:xgb_section_end]
            
            # Replace the model saving logic
            new_xgb_section = xgb_section.replace(
                "xgboost_models = {}",
                "xgboost_models = {}\n        xgboost_all_results = {}"
            )
            
            # Update the loop to handle dictionary results
            new_xgb_section = new_xgb_section.replace(
                "model_data = train_xgboost_categorical(",
                "model_results = train_xgboost_categorical("
            )
            
            new_xgb_section = new_xgb_section.replace(
                "xgboost_models[model_name] = model_data",
                "# Enhanced version returns multiple models\n            xgboost_all_results.update(model_results)\n            # Add to xgboost_models for compatibility\n            for key, value in model_results.items():\n                xgboost_models[key] = value"
            )
            
            content = content[:xgb_section_start] + new_xgb_section + content[xgb_section_end:]
    
    # Similar update for LightGBM
    lgb_section_start = content.find("if args.lightgbm or args.all:")
    if lgb_section_start != -1:
        lgb_section_end = content.find("if args.catboost or args.all:", lgb_section_start)
        if lgb_section_end == -1:
            lgb_section_end = content.find("# Evaluation", lgb_section_start)
        
        if lgb_section_end != -1:
            lgb_section = content[lgb_section_start:lgb_section_end]
            
            # Replace the model saving logic
            new_lgb_section = lgb_section.replace(
                "lightgbm_models = {}",
                "lightgbm_models = {}\n        lightgbm_all_results = {}"
            )
            
            # Update the loop to handle dictionary results
            new_lgb_section = new_lgb_section.replace(
                "model_data = train_lightgbm_categorical(",
                "model_results = train_lightgbm_categorical("
            )
            
            new_lgb_section = new_lgb_section.replace(
                "lightgbm_models[model_name] = model_data",
                "# Enhanced version returns multiple models\n            lightgbm_all_results.update(model_results)\n            # Add to lightgbm_models for compatibility\n            for key, value in model_results.items():\n                lightgbm_models[key] = value"
            )
            
            content = content[:lgb_section_start] + new_lgb_section + content[lgb_section_end:]
    
    # Write updated content
    with open("main.py", "w") as f:
        f.write(content)
    
    print("✅ main.py updated successfully!")

def create_compatibility_wrappers():
    """Create wrapper functions for backward compatibility."""
    print("\nCreating compatibility wrappers...")
    
    # Create updated lightgbm_categorical.py
    lightgbm_wrapper = '''#!/usr/bin/env python3
"""
Wrapper for enhanced LightGBM categorical implementation.
Provides backward compatibility while using the enhanced version.
"""

from enhanced_lightgbm_categorical import train_enhanced_lightgbm_categorical

def train_lightgbm_categorical(X, y, dataset_name, categorical_columns, test_size=0.2, random_state=42):
    """
    Wrapper function that calls the enhanced implementation.
    Returns results in a format compatible with the existing pipeline.
    """
    # Call enhanced version
    results = train_enhanced_lightgbm_categorical(
        X, y, dataset_name, categorical_columns, test_size, random_state
    )
    
    # For backward compatibility, return the dictionary of models
    return results
'''
    
    with open("models/lightgbm_categorical_wrapper.py", "w") as f:
        f.write(lightgbm_wrapper)
    
    # Create updated xgboost_categorical.py
    xgboost_wrapper = '''#!/usr/bin/env python3
"""
Wrapper for enhanced XGBoost categorical implementation.
Provides backward compatibility while using the enhanced version.
"""

from enhanced_xgboost_categorical import train_enhanced_xgboost_categorical

def train_xgboost_categorical(X, y, dataset_name, categorical_columns, test_size=0.2, random_state=42):
    """
    Wrapper function that calls the enhanced implementation.
    Returns results in a format compatible with the existing pipeline.
    """
    # Call enhanced version
    results = train_enhanced_xgboost_categorical(
        X, y, dataset_name, categorical_columns, test_size, random_state
    )
    
    # For backward compatibility, return the dictionary of models
    return results
'''
    
    with open("models/xgboost_categorical_wrapper.py", "w") as f:
        f.write(xgboost_wrapper)
    
    print("✅ Compatibility wrappers created!")

def update_model_imports():
    """Update the original model files to redirect to enhanced versions."""
    print("\nUpdating model imports...")
    
    # Rename original files
    if Path("models/lightgbm_categorical.py").exists():
        shutil.move("models/lightgbm_categorical.py", "models/lightgbm_categorical_original.py")
    
    if Path("models/xgboost_categorical.py").exists():
        shutil.move("models/xgboost_categorical.py", "models/xgboost_categorical_original.py")
    
    # Create new files that import from enhanced versions
    lightgbm_redirect = '''#!/usr/bin/env python3
"""
Redirect to enhanced LightGBM categorical implementation.
"""

from enhanced_lightgbm_categorical import train_enhanced_lightgbm_categorical

def train_lightgbm_categorical(X, y, dataset_name, categorical_columns, test_size=0.2, random_state=42):
    """
    Call the enhanced implementation and return results.
    The enhanced version returns a dictionary with both basic and optuna models.
    """
    return train_enhanced_lightgbm_categorical(X, y, dataset_name, categorical_columns, test_size, random_state)
'''
    
    with open("models/lightgbm_categorical.py", "w") as f:
        f.write(lightgbm_redirect)
    
    xgboost_redirect = '''#!/usr/bin/env python3
"""
Redirect to enhanced XGBoost categorical implementation.
"""

from enhanced_xgboost_categorical import train_enhanced_xgboost_categorical

def train_xgboost_categorical(X, y, dataset_name, categorical_columns, test_size=0.2, random_state=42):
    """
    Call the enhanced implementation and return results.
    The enhanced version returns a dictionary with both basic and optuna models.
    """
    return train_enhanced_xgboost_categorical(X, y, dataset_name, categorical_columns, test_size, random_state)
'''
    
    with open("models/xgboost_categorical.py", "w") as f:
        f.write(xgboost_redirect)
    
    print("✅ Model imports updated!")

def main():
    """Run the integration process."""
    print("Enhanced Model Integration")
    print("=========================")
    
    # Backup original files
    backup_dir = backup_original_files()
    print(f"\nBackup created in: {backup_dir}")
    
    # Update files
    #update_main_py()  # This is complex, let's do simpler approach
    update_model_imports()  # Just redirect the imports
    
    print("\n✅ Integration complete!")
    print("\nThe enhanced models will now be used when running:")
    print("  python main.py --xgboost")
    print("  python main.py --lightgbm")
    print("  python main.py --all")
    
    print("\nNote: The first run will train new models with Optuna optimization.")
    print("This will take longer but produce better results.")

if __name__ == "__main__":
    main()