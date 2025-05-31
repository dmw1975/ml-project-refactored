#!/usr/bin/env python3
"""Add SHAP and CV distribution visualizations to the main pipeline."""

import os
from pathlib import Path

def add_visualizations_to_init():
    """Add SHAP and CV distribution imports to visualization_new/__init__.py"""
    
    init_file = Path("visualization_new/__init__.py")
    
    # Read current content
    with open(init_file, 'r') as f:
        content = f.read()
    
    # Check if already added
    if 'plot_cv_distributions' in content:
        print("CV distributions already in __init__.py")
        return
    
    # Find the import section
    import_section_end = content.find("__all__")
    
    # Add new imports before __all__
    new_imports = """
# Import CV distribution functions
from visualization_new.plots.cv_distributions import (
    plot_cv_distributions
)

"""
    
    # Insert new imports
    content = content[:import_section_end] + new_imports + content[import_section_end:]
    
    # Add to __all__
    all_section_start = content.find("__all__ = [")
    all_section_end = content.find("]", all_section_start)
    
    # Add new exports
    new_exports = "    'plot_cv_distributions',\n"
    
    # Insert before the closing bracket
    content = content[:all_section_end] + new_exports + content[all_section_end:]
    
    # Write back
    with open(init_file, 'w') as f:
        f.write(content)
    
    print("✓ Added CV distributions to visualization_new/__init__.py")

def add_visualizations_to_main():
    """Add SHAP and CV distribution calls to main.py visualization section."""
    
    main_file = Path("main.py")
    
    # Read current content
    with open(main_file, 'r') as f:
        lines = f.readlines()
    
    # Find where to insert new visualization calls
    # Look for the statistical tests section
    insert_index = None
    for i, line in enumerate(lines):
        if "Creating statistical test visualizations..." in line:
            # Find the end of the try block
            for j in range(i, len(lines)):
                if lines[j].strip().startswith("except Exception as e:") and "statistical test" in lines[j+1]:
                    insert_index = j + 3  # After the traceback.print_exc()
                    break
            break
    
    if insert_index is None:
        print("Could not find insertion point in main.py")
        return
    
    # Create new visualization sections
    new_sections = '''
                # Generate SHAP visualizations for tree models
                try:
                    print("Creating SHAP visualizations...")
                    # Check if SHAP script exists
                    shap_script = Path("generate_shap_visualizations.py")
                    if shap_script.exists():
                        import subprocess
                        result = subprocess.run([sys.executable, str(shap_script)], 
                                              capture_output=True, text=True)
                        if result.returncode == 0:
                            print("SHAP visualizations created successfully.")
                        else:
                            print(f"Error creating SHAP visualizations: {result.stderr}")
                    else:
                        print("SHAP visualization script not found. Skipping.")
                except Exception as e:
                    print(f"Error creating SHAP visualizations: {e}")
                
                # Generate CV distribution plots
                try:
                    print("Creating CV distribution plots...")
                    from visualization_new.plots.cv_distributions import plot_cv_distributions
                    
                    # Filter models with CV data
                    cv_models = []
                    for model_data in model_list:
                        if isinstance(model_data, dict) and ('cv_scores' in model_data or 
                                                            'cv_fold_scores' in model_data or
                                                            'cv_mean' in model_data):
                            cv_models.append(model_data)
                    
                    if cv_models:
                        cv_config = {
                            'save': True,
                            'output_dir': settings.VISUALIZATION_DIR / "performance" / "cv_distributions",
                            'dpi': 300,
                            'format': 'png'
                        }
                        cv_figures = plot_cv_distributions(cv_models, cv_config)
                        print(f"Created {len(cv_figures)} CV distribution plots.")
                    else:
                        print("No models with CV data found. Skipping CV distribution plots.")
                except Exception as e:
                    print(f"Error creating CV distribution plots: {e}")
                    import traceback
                    traceback.print_exc()
                
'''
    
    # Insert the new sections
    lines.insert(insert_index, new_sections)
    
    # Write back
    with open(main_file, 'w') as f:
        f.writelines(lines)
    
    print("✓ Added SHAP and CV distribution calls to main.py")

def main():
    """Add missing visualizations to the pipeline."""
    print("Adding missing visualizations to the pipeline...")
    
    add_visualizations_to_init()
    add_visualizations_to_main()
    
    print("\n✅ Done! SHAP and CV distribution plots will now be generated when running:")
    print("   python main.py --visualize")
    print("   python main.py --all")

if __name__ == "__main__":
    main()