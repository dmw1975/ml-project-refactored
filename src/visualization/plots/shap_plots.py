#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SHAP visualization module for ML models.
Integrates SHAP analysis into the unified visualization framework.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import shap

# Handle seaborn import with fallback
try:
    import seaborn as sns
except ImportError:
    sns = None
    print("Warning: seaborn not available. Using matplotlib fallbacks.")

warnings.filterwarnings('ignore')


class SHAPVisualizer:
    """SHAP visualization class for creating model interpretability plots."""
    
    def __init__(self, model_data: Dict[str, Any], style: Optional[Dict] = None):
        """
        Initialize SHAP visualizer.
        
        Args:
            model_data: Dictionary containing model information
            style: Optional style configuration
        """
        self.model_data = model_data
        self.model = model_data.get('model')
        self.model_name = model_data.get('model_name', 'Unknown')
        self.X_test = model_data.get('X_test')
        self.feature_names = None
        
        # Get feature names from various possible sources
        if self.X_test is not None:
            if hasattr(self.X_test, 'columns'):
                self.feature_names = list(self.X_test.columns)
            elif 'feature_names' in model_data:
                self.feature_names = model_data['feature_names']
        
        self.style = style or {
            'figure_size': (12, 8),
            'dpi': 300,
            'colors': {
                'primary': '#3498db',
                'secondary': '#2ecc71',
                'tertiary': '#e67e22',
                'quaternary': '#9b59b6'
            }
        }
    
    def create_shap_summary_plot(self, shap_values: np.ndarray, X_sample: pd.DataFrame, 
                                output_path: Path) -> None:
        """
        Create SHAP summary plot showing feature importance and impact.
        
        Args:
            shap_values: SHAP values array
            X_sample: Sample data for SHAP analysis
            output_path: Path to save the plot
        """
        plt.figure(figsize=self.style['figure_size'])
        
        # Create SHAP summary plot
        shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
        
        plt.title(f'SHAP Feature Importance - {self.model_name}', fontsize=14, pad=20)
        plt.xlabel('Mean |SHAP Value| (Average Impact on Model Output)', fontsize=12)
        plt.tight_layout()
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=self.style['dpi'], bbox_inches='tight')
        plt.close()
    
    def create_shap_waterfall_plot(self, shap_values: np.ndarray, X_sample: pd.DataFrame,
                                   instance_idx: int, output_path: Path) -> None:
        """
        Create SHAP waterfall plot for a single prediction.
        
        Args:
            shap_values: SHAP values array
            X_sample: Sample data
            instance_idx: Index of instance to explain
            output_path: Path to save the plot
        """
        plt.figure(figsize=(10, 6))
        
        # Create explanation object
        if hasattr(shap, 'Explanation'):
            # For newer SHAP versions
            explanation = shap.Explanation(
                values=shap_values[instance_idx],
                base_values=0,  # Will be set by waterfall plot
                data=X_sample.iloc[instance_idx].values,
                feature_names=list(X_sample.columns)
            )
            shap.waterfall_plot(explanation, show=False)
        else:
            # Fallback for older versions - create manual waterfall
            self._create_manual_waterfall(shap_values[instance_idx], X_sample.iloc[instance_idx], output_path)
            return
        
        plt.title(f'SHAP Waterfall Plot - {self.model_name} (Instance {instance_idx})', fontsize=14)
        plt.tight_layout()
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=self.style['dpi'], bbox_inches='tight')
        plt.close()
    
    def create_shap_dependence_plot(self, shap_values: np.ndarray, X_sample: pd.DataFrame,
                                    feature_name: str, output_path: Path) -> None:
        """
        Create SHAP dependence plot for a specific feature.
        
        Args:
            shap_values: SHAP values array
            X_sample: Sample data
            feature_name: Name of feature to plot
            output_path: Path to save the plot
        """
        plt.figure(figsize=self.style['figure_size'])
        
        # Get feature index
        feature_idx = list(X_sample.columns).index(feature_name)
        
        # Create dependence plot
        shap.dependence_plot(feature_idx, shap_values, X_sample, show=False)
        
        plt.title(f'SHAP Dependence Plot - {feature_name} ({self.model_name})', fontsize=14)
        plt.tight_layout()
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=self.style['dpi'], bbox_inches='tight')
        plt.close()
    
    def create_categorical_shap_plot(self, shap_values: np.ndarray, X_sample: pd.DataFrame,
                                    feature_name: str, output_path: Path) -> None:
        """
        Create specialized SHAP plot for categorical features.
        
        Args:
            shap_values: SHAP values array
            X_sample: Sample data
            feature_name: Name of categorical feature
            output_path: Path to save the plot
        """
        # Get feature column index
        feature_idx = list(X_sample.columns).index(feature_name)
        
        # Extract SHAP values for this feature
        feature_shap_values = shap_values[:, feature_idx]
        
        # Get the feature values
        feature_values = X_sample[feature_name].values
        
        # Create DataFrame for plotting
        plot_df = pd.DataFrame({
            'Category': feature_values,
            'SHAP Value': feature_shap_values
        })
        
        # Create the plot
        plt.figure(figsize=self.style['figure_size'])
        
        if sns is not None:
            # Use violin plot to show distribution
            sns.violinplot(data=plot_df, x='Category', y='SHAP Value', inner='box')
        else:
            # Fallback to boxplot
            categories = plot_df['Category'].unique()
            data_by_category = [plot_df[plot_df['Category'] == cat]['SHAP Value'].values 
                               for cat in categories]
            plt.boxplot(data_by_category, labels=categories)
            plt.xlabel('Category')
            plt.ylabel('SHAP Value')
        
        plt.title(f'SHAP Value Distribution by {feature_name} - {self.model_name}', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        
        # Add horizontal line at y=0
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        
        # Add annotation
        plt.text(0.02, 0.98, 'Above 0 = increases prediction\nBelow 0 = decreases prediction', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=self.style['dpi'], bbox_inches='tight')
        plt.close()
    
    def _create_manual_waterfall(self, shap_values: np.ndarray, instance_data: pd.Series, 
                                output_path: Path) -> None:
        """Create manual waterfall plot for older SHAP versions."""
        # Sort features by absolute SHAP value
        feature_names = list(instance_data.index)
        sorted_idx = np.argsort(np.abs(shap_values))[::-1][:10]  # Top 10 features
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate cumulative sum
        cumsum = 0
        for i, idx in enumerate(sorted_idx):
            value = shap_values[idx]
            feature = feature_names[idx]
            
            # Draw bar
            if value > 0:
                ax.barh(i, value, left=cumsum, color=self.style['colors']['secondary'])
            else:
                ax.barh(i, value, left=cumsum + value, color=self.style['colors']['primary'])
            
            cumsum += value
            
            # Add feature name
            ax.text(-0.1, i, f"{feature} = {instance_data.iloc[idx]:.2f}", 
                   ha='right', va='center', fontsize=10)
        
        ax.set_yticks(range(len(sorted_idx)))
        ax.set_yticklabels([''] * len(sorted_idx))
        ax.set_xlabel('SHAP Value')
        ax.set_title(f'Feature Contributions - {self.model_name}', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.style['dpi'], bbox_inches='tight')
        plt.close()


def create_shap_visualizations(model_data: Dict[str, Any], output_dir: Path, 
                              sample_size: int = 100) -> List[Path]:
    """
    Create all SHAP visualizations for a model.
    
    Args:
        model_data: Dictionary containing model information
        output_dir: Directory to save visualizations
        sample_size: Number of samples to use for SHAP analysis
        
    Returns:
        List of paths to created visualizations
    """
    created_paths = []
    
    try:
        model = model_data.get('model')
        X_test = model_data.get('X_test')
        model_name = model_data.get('model_name', 'Unknown')
        
        if model is None or X_test is None:
            print(f"Skipping SHAP analysis for {model_name}: Missing model or test data")
            return created_paths
        
        # Ensure X_test is a DataFrame
        if not isinstance(X_test, pd.DataFrame):
            if hasattr(X_test, 'columns'):
                # It might be a numpy array with column info
                X_test = pd.DataFrame(X_test)
            else:
                print(f"Skipping SHAP analysis for {model_name}: X_test is not a DataFrame")
                return created_paths
        
        # Sample data for SHAP analysis
        n_samples = min(sample_size, len(X_test))
        sample_indices = np.random.choice(len(X_test), n_samples, replace=False)
        X_sample = X_test.iloc[sample_indices]
        
        # Create explainer based on model type
        model_type = model_data.get('model_type', '').lower()
        
        if 'xgboost' in model_type or 'xgb' in str(type(model)).lower():
            explainer = shap.TreeExplainer(model)
        elif 'lightgbm' in model_type or 'lgb' in str(type(model)).lower():
            explainer = shap.TreeExplainer(model)
        elif 'catboost' in model_type or 'cat' in str(type(model)).lower():
            explainer = shap.TreeExplainer(model)
        elif 'elastic' in model_type or 'linear' in model_type:
            explainer = shap.LinearExplainer(model, X_sample)
        else:
            # Try tree explainer first, fall back to kernel
            try:
                explainer = shap.TreeExplainer(model)
            except:
                explainer = shap.KernelExplainer(model.predict, X_sample)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_sample)
        
        # Handle multi-class output
        if isinstance(shap_values, list):
            shap_values = shap_values[0]  # Use first class for regression
        
        # Create visualizer
        visualizer = SHAPVisualizer(model_data)
        
        # Create output directory
        model_output_dir = output_dir / model_name.replace(' ', '_')
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Summary plot
        summary_path = model_output_dir / f"{model_name}_shap_summary.png"
        visualizer.create_shap_summary_plot(shap_values, X_sample, summary_path)
        created_paths.append(summary_path)
        
        # 2. Waterfall plot for first instance
        waterfall_path = model_output_dir / f"{model_name}_shap_waterfall.png"
        visualizer.create_shap_waterfall_plot(shap_values, X_sample, 0, waterfall_path)
        created_paths.append(waterfall_path)
        
        # 3. Dependence plots for top features
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_features_idx = np.argsort(mean_abs_shap)[::-1][:5]  # Top 5 features
        
        for idx in top_features_idx:
            feature_name = X_sample.columns[idx]
            dependence_path = model_output_dir / f"{model_name}_shap_dependence_{feature_name}.png"
            visualizer.create_shap_dependence_plot(shap_values, X_sample, feature_name, dependence_path)
            created_paths.append(dependence_path)
        
        # 4. Categorical feature plots (if any)
        categorical_features = identify_categorical_features(X_sample, model_data)
        for cat_feature in categorical_features[:3]:  # Top 3 categorical features
            if cat_feature in X_sample.columns:
                cat_path = model_output_dir / f"{model_name}_shap_categorical_{cat_feature}.png"
                visualizer.create_categorical_shap_plot(shap_values, X_sample, cat_feature, cat_path)
                created_paths.append(cat_path)
        
        print(f"Created {len(created_paths)} SHAP visualizations for {model_name}")
        
    except Exception as e:
        print(f"Error creating SHAP visualizations for {model_data.get('model_name', 'Unknown')}: {e}")
        import traceback
        traceback.print_exc()
    
    return created_paths


def identify_categorical_features(X_sample: pd.DataFrame, model_data: Dict[str, Any]) -> List[str]:
    """
    Identify categorical features in the dataset.
    
    Args:
        X_sample: Sample data
        model_data: Model information dictionary
        
    Returns:
        List of categorical feature names
    """
    categorical_features = []
    
    # Method 1: Check data types
    for col in X_sample.columns:
        if X_sample[col].dtype == 'object' or X_sample[col].dtype.name == 'category':
            categorical_features.append(col)
    
    # Method 2: Check for integer columns with few unique values
    for col in X_sample.columns:
        if col not in categorical_features:
            if X_sample[col].dtype in ['int64', 'int32', 'float64', 'float32']:
                n_unique = X_sample[col].nunique()
                # Less than 20 unique values or 5% of data
                if n_unique < 20 and n_unique < len(X_sample) * 0.05:
                    categorical_features.append(col)
    
    # Method 3: Known categorical columns
    known_categorical = ['gics_sector', 'issuer_cntry_domicile', 'moodys_rating', 
                        'sp_rating', 'fitch_rating', 'tier']
    for col in known_categorical:
        if col in X_sample.columns and col not in categorical_features:
            categorical_features.append(col)
    
    # Method 4: Check model's categorical features if available
    if 'categorical_features' in model_data:
        model_cats = model_data['categorical_features']
        for col in model_cats:
            if col in X_sample.columns and col not in categorical_features:
                categorical_features.append(col)
    
    return categorical_features


def create_all_shap_visualizations(models: Dict[str, Any], output_dir: Path) -> Dict[str, List[Path]]:
    """
    Create SHAP visualizations for all models.
    
    Args:
        models: Dictionary of models
        output_dir: Directory to save visualizations
        
    Returns:
        Dictionary mapping model names to lists of created paths
    """
    all_paths = {}
    shap_dir = output_dir / "shap"
    shap_dir.mkdir(parents=True, exist_ok=True)
    
    for model_name, model_data in models.items():
        print(f"\nCreating SHAP visualizations for {model_name}...")
        paths = create_shap_visualizations(model_data, shap_dir)
        all_paths[model_name] = paths
    
    # Create README
    readme_path = shap_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write("# SHAP Visualizations\n\n")
        f.write("This directory contains SHAP (SHapley Additive exPlanations) visualizations for all models.\n\n")
        f.write("## Visualization Types\n\n")
        f.write("- **Summary Plots**: Show feature importance and impact distribution\n")
        f.write("- **Waterfall Plots**: Explain individual predictions\n")
        f.write("- **Dependence Plots**: Show relationship between feature values and SHAP values\n")
        f.write("- **Categorical Plots**: Special plots for categorical features\n\n")
        f.write("## Models Analyzed\n\n")
        for model_name, paths in all_paths.items():
            if paths:
                f.write(f"- {model_name}: {len(paths)} visualizations\n")
    
    # After creating individual model SHAP visualizations, create comparison plot
    print("\nCreating model comparison SHAP plot...")
    comparison_path = create_model_comparison_shap_plot(models, shap_dir)
    if comparison_path:
        all_paths['model_comparison'] = [comparison_path]
    
    return all_paths


def create_model_comparison_shap_plot(models: Dict[str, Any], output_dir: Path) -> Optional[Path]:
    """
    Create a heatmap comparing SHAP feature importance across different models.
    
    Args:
        models: Dictionary of models
        output_dir: Directory to save the plot
        
    Returns:
        Path to the created plot, or None if failed
    """
    try:
        # Collect SHAP values for each model
        # We'll organize by model type (not including Random variations)
        model_shap_by_type = {}
        all_feature_names = {}
        
        for model_name, model_data in models.items():
            # Skip non-tree models for now
            model_type = model_data.get('model_type', '').lower()
            if not any(tree_type in model_type for tree_type in ['xgboost', 'lightgbm', 'catboost', 'xgb', 'lgb']):
                continue
            
            # Skip Random models for cleaner comparison
            if 'Random' in model_name:
                continue
                
            model = model_data.get('model')
            X_test = model_data.get('X_test')
            
            if model is None or X_test is None:
                continue
            
            # Ensure X_test is a DataFrame
            if not isinstance(X_test, pd.DataFrame):
                continue
            
            # Sample data for SHAP analysis
            n_samples = min(100, len(X_test))
            sample_indices = np.random.choice(len(X_test), n_samples, replace=False)
            X_sample = X_test.iloc[sample_indices]
            
            # Extract model type for display
            if 'xgboost' in model_type or 'xgb' in model_type:
                display_name = 'XGBoost'
            elif 'lightgbm' in model_type or 'lgb' in model_type:
                display_name = 'LightGBM'
            elif 'catboost' in model_type:
                display_name = 'CatBoost'
            else:
                display_name = model_type.upper()
            
            # Store feature names per model type
            all_feature_names[display_name] = list(X_sample.columns)
            
            # Create explainer and calculate SHAP values
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)
                
                # Handle multi-class output
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]
                
                # Calculate mean absolute SHAP values
                mean_abs_shap = np.abs(shap_values).mean(axis=0)
                
                # Store the first valid result for each model type
                if display_name not in model_shap_by_type:
                    model_shap_by_type[display_name] = {
                        'shap_values': mean_abs_shap,
                        'feature_names': list(X_sample.columns)
                    }
                
            except Exception as e:
                print(f"  Error calculating SHAP values for {model_name}: {e}")
                continue
        
        if len(model_shap_by_type) < 2:
            print("  Not enough models with SHAP values for comparison")
            return None
        
        # Prepare data for DataFrame
        # Use the feature names from the first model type as reference
        reference_features = None
        model_shap_data = {}
        
        for display_name, data in model_shap_by_type.items():
            if reference_features is None:
                reference_features = data['feature_names']
            model_shap_data[display_name] = data['shap_values']
        
        # Create DataFrame for comparison
        shap_df = pd.DataFrame(model_shap_data, index=reference_features)
        
        # Select top features based on maximum importance across models
        max_importance = shap_df.max(axis=1)
        top_features = max_importance.nlargest(15).index
        shap_df_top = shap_df.loc[top_features]
        
        # Normalize values for better comparison (0-1 scale for each model)
        shap_df_normalized = shap_df_top.copy()
        for col in shap_df_normalized.columns:
            max_val = shap_df_normalized[col].max()
            if max_val > 0:
                shap_df_normalized[col] = shap_df_normalized[col] / max_val
        
        # Create the heatmap
        plt.figure(figsize=(10, 8))
        
        # Use a colormap that starts from white
        cmap = plt.cm.viridis
        
        # Create heatmap with annotations
        ax = plt.gca()
        im = ax.imshow(shap_df_normalized.values, cmap=cmap, aspect='auto')
        
        # Set ticks
        ax.set_xticks(np.arange(len(shap_df_normalized.columns)))
        ax.set_yticks(np.arange(len(shap_df_normalized.index)))
        ax.set_xticklabels(shap_df_normalized.columns)
        ax.set_yticklabels(shap_df_normalized.index)
        
        # Rotate the tick labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Mean |SHAP Value|', rotation=270, labelpad=20)
        
        # Add text annotations
        for i in range(len(shap_df_normalized.index)):
            for j in range(len(shap_df_normalized.columns)):
                text = ax.text(j, i, f'{shap_df_normalized.iloc[i, j]:.3f}',
                             ha="center", va="center", color="white" if shap_df_normalized.iloc[i, j] > 0.5 else "black")
        
        # Add grid
        ax.set_xticks(np.arange(len(shap_df_normalized.columns) + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(shap_df_normalized.index) + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
        ax.tick_params(which="minor", size=0)
        
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.title('Feature Importance Comparison Across Models', fontsize=14)
        plt.tight_layout()
        
        # Save the plot
        output_path = output_dir / "model_comparison_shap.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Created model comparison SHAP plot: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"  Error creating model comparison SHAP plot: {e}")
        import traceback
        traceback.print_exc()
        return None


# Make functions available at module level
__all__ = ['create_shap_visualizations', 'create_all_shap_visualizations', 
           'SHAPVisualizer', 'identify_categorical_features', 'create_model_comparison_shap_plot']