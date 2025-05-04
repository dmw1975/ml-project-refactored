#!/usr/bin/env python3
"""
Script to update main.py to use only the new visualization architecture.

This script updates the main.py file to:
1. Redirect all --visualize flag calls to use visualization_new
2. Add appropriate fallbacks for missing features
3. Add debug logging for transition period
"""

import re
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

def update_main_py():
    """Update main.py to use only the new visualization architecture."""
    main_py_path = project_root / "main.py"
    
    if not main_py_path.exists():
        print(f"Error: {main_py_path} not found")
        return
    
    # Read the main.py file
    with open(main_py_path, "r") as f:
        content = f.read()
    
    # Create a backup
    backup_path = main_py_path.with_suffix(".py.bak")
    with open(backup_path, "w") as f:
        f.write(content)
    print(f"Created backup of main.py at {backup_path}")
    
    # Update visualization flags
    updated_content = content
    
    # Pattern for updating the visualize flag
    visualize_flag_pattern = r"(parser\.add_argument\('--visualize'.*?help=')([^']*?)('.*?\))"
    visualize_flag_replacement = r"\1Generate visualizations (uses new architecture)\3"
    updated_content = re.sub(visualize_flag_pattern, visualize_flag_replacement, updated_content)
    
    # Pattern for updating the visualize-new flag
    visualize_new_flag_pattern = r"(parser\.add_argument\('--visualize-new'.*?help=')([^']*?)('.*?\))"
    visualize_new_flag_replacement = r"\1Generate visualizations using new architecture (same as --visualize)\3"
    updated_content = re.sub(visualize_new_flag_pattern, visualize_new_flag_replacement, updated_content)
    
    # Update the visualization section
    # Find the old visualization block
    old_viz_pattern = r"if args\.all or args\.visualize:.*?# Generate dataset-centric model comparisons.*?create_all_dataset_comparisons\(\)"
    
    # Create the new visualization code that redirects to visualization_new
    new_viz_code = """if args.all or args.visualize or args.visualize_new:
            print("\\nGenerating visualizations using unified architecture...")
            
            # Import from visualization_new architecture
            import visualization_new as viz
            from visualization_new.utils.io import load_all_models
            
            try:
                # Load all models once
                models = load_all_models()
                model_list = list(models.values())
                
                # Create all visualizations
                try:
                    print("Creating residual plots...")
                    viz.create_all_residual_plots()
                except Exception as e:
                    print(f"Error creating residual plots: {e}")
                    # Fallback to legacy visualization if needed
                    print("Falling back to legacy residual plots...")
                    from visualization.create_residual_plots import create_all_residual_plots
                    create_all_residual_plots()
                
                try:
                    print("Creating model comparison visualizations...")
                    viz.create_model_comparison_plot(model_list)
                except Exception as e:
                    print(f"Error creating model comparison: {e}")
                    # Fallback to legacy visualization if needed
                    print("Falling back to legacy model comparison...")
                    from visualization.metrics_plots import plot_model_comparison
                    plot_model_comparison()
                
                try:
                    print("Creating metrics summary table...")
                    viz.create_metrics_table(model_list)
                except Exception as e:
                    print(f"Error creating metrics table: {e}")
                    # Fallback to legacy visualization if needed
                    print("Falling back to legacy metrics table...")
                    from visualization.metrics_plots import plot_metrics_summary_table
                    plot_metrics_summary_table()
                
                try:
                    print("Creating feature importance visualizations...")
                    for model in model_list:
                        viz.create_feature_importance_plot(model)
                except Exception as e:
                    print(f"Error creating feature importance plots: {e}")
                    # Fallback to legacy visualization if needed
                    print("Falling back to legacy feature importance plots...")
                    from visualization.feature_plots import plot_top_features, plot_feature_importance_by_model
                    plot_top_features()
                    plot_feature_importance_by_model()
                
                try:
                    print("Creating statistical test visualizations...")
                    viz.create_comparative_dashboard(model_list)
                except Exception as e:
                    print(f"Error creating statistical visualizations: {e}")
                    # Fallback to legacy visualization if needed
                    print("Falling back to legacy statistical tests...")
                    from visualization.statistical_tests import visualize_statistical_tests
                    visualize_statistical_tests()
                
                # ElasticNet specific visualization
                try:
                    print("Creating ElasticNet visualizations...")
                    elasticnet_models = {name: model for name, model in models.items() 
                                      if 'elasticnet' in name.lower()}
                    
                    if elasticnet_models:
                        for name, model in elasticnet_models.items():
                            print(f"Creating visualizations for {name}...")
                            model_adapter = viz.core.registry.get_adapter_for_model(model)
                            viz.visualize_model(model_adapter)
                    else:
                        print("No ElasticNet models found.")
                except Exception as e:
                    print(f"Error creating ElasticNet visualizations: {e}")
                    # Fallback to legacy visualization if needed
                    print("Falling back to legacy ElasticNet visualizations...")
                    from visualization.elasticnet_plots import plot_elasticnet_feature_importance
                    plot_elasticnet_feature_importance()
                
                # Dataset comparison visualizations
                try:
                    print("Creating dataset comparison visualizations...")
                    viz.plots.dataset_comparison.create_all_dataset_comparisons()
                except Exception as e:
                    print(f"Error creating dataset comparisons: {e}")
                    # Fallback to legacy visualization if needed
                    print("Falling back to legacy dataset comparisons...")
                    from visualization.dataset_comparison import create_all_dataset_comparisons
                    create_all_dataset_comparisons()
                
                print("Visualization completed successfully.")
            except Exception as e:
                print(f"Error in visualization pipeline: {e}")
                print("Falling back to legacy visualization...")
                
                # Import legacy visualization as a last resort
                print("\\nFalling back to legacy visualization module...")
                from visualization.metrics_plots import plot_model_comparison, plot_residuals, plot_statistical_tests_filtered
                from visualization.create_residual_plots import create_all_residual_plots
                from visualization.feature_plots import plot_top_features, plot_feature_importance_by_model
                from visualization.statistical_tests import visualize_statistical_tests
                
                print("Creating model performance visualizations...")
                plot_model_comparison()
                create_all_residual_plots()
                plot_residuals()
                plot_statistical_tests_filtered()
                visualize_statistical_tests()
                
                print("Creating feature importance visualizations...")
                plot_top_features()
                plot_feature_importance_by_model()"""
    
    # Replace the old visualization block with the new code
    updated_content = re.sub(old_viz_pattern, new_viz_code, updated_content, flags=re.DOTALL)
    
    # Remove the old visualize-new block since we merged it
    old_visualize_new_pattern = r"if args\.all or args\.visualize_new:.*?except Exception as e:.*?print\(f\"Error in comparative dashboard: \{e\}\"\)"
    updated_content = re.sub(old_visualize_new_pattern, "", updated_content, flags=re.DOTALL)
    
    # Write the updated content
    with open(main_py_path, "w") as f:
        f.write(updated_content)
    
    print(f"Updated {main_py_path} to use visualization_new architecture")
    print("Please review the changes and ensure all functionality works as expected.")
    print(f"A backup of the original file was saved to {backup_path}")

if __name__ == "__main__":
    update_main_py()