#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SHAP visualization module for ML models.
Integrates SHAP analysis into the unified visualization framework.

ENHANCED: Added dot summary, force plots, interaction plots, and safety features - 2025-01-13
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import shap
import gc
import psutil

# Handle seaborn import with fallback
try:
    import seaborn as sns
except ImportError:
    sns = None
    print("Warning: seaborn not available. Using matplotlib fallbacks.")

warnings.filterwarnings('ignore')

# Import settings for default paths
from src.config import settings


def check_memory_usage():
    """Check current memory usage in GB."""
    process = psutil.Process()
    mem_info = process.memory_info()
    return mem_info.rss / 1024 / 1024 / 1024  # GB


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
    
    def create_shap_dot_summary_plot(self, shap_values: np.ndarray, X_sample: pd.DataFrame,
                                    output_path: Path) -> None:
        """
        Create SHAP summary plot with dot style showing distributions.
        
        Args:
            shap_values: SHAP values array
            X_sample: Sample data for SHAP analysis
            output_path: Path to save the plot
        """
        plt.figure(figsize=self.style['figure_size'])
        
        # Create SHAP summary plot with dots
        shap.summary_plot(shap_values, X_sample, plot_type="dot", show=False)
        
        plt.title(f'SHAP Summary Plot - {self.model_name}', fontsize=14, pad=20)
        plt.tight_layout()
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=self.style['dpi'], bbox_inches='tight')
        plt.close()
    
    def create_shap_force_plots(self, explainer, shap_values: np.ndarray, X_sample: pd.DataFrame,
                               output_dir: Path, n_instances: int = 3) -> List[Path]:
        """
        Create multiple SHAP force plots for individual instances.
        
        Args:
            explainer: SHAP explainer object
            shap_values: SHAP values array
            X_sample: Sample data
            output_dir: Directory to save plots
            n_instances: Number of instances to plot
            
        Returns:
            List of paths to created plots
        """
        created_paths = []
        n_to_plot = min(n_instances, len(X_sample))
        
        for i in range(n_to_plot):
            try:
                plt.figure(figsize=(20, 3))
                
                # Create force plot
                shap.force_plot(
                    explainer.expected_value,
                    shap_values[i],
                    X_sample.iloc[i],
                    matplotlib=True,
                    show=False
                )
                
                plt.title(f'SHAP Force Plot - {self.model_name} (Instance {i+1})', fontsize=14)
                
                # Save plot
                output_path = output_dir / f"{self.model_name}_shap_force_instance_{i+1}.png"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(output_path, dpi=self.style['dpi'], bbox_inches='tight')
                plt.close()
                
                created_paths.append(output_path)
                
            except Exception as e:
                print(f"  Force plot for instance {i+1} failed: {e}")
                plt.close()
        
        return created_paths
    
    def create_shap_interaction_plot(self, shap_values: np.ndarray, X_sample: pd.DataFrame,
                                    output_path: Path, top_n: int = 2) -> None:
        """
        Create SHAP interaction plot for top features.
        
        Args:
            shap_values: SHAP values array
            X_sample: Sample data
            output_path: Path to save the plot
            top_n: Number of top features to use
        """
        # Get top features by importance
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_features_idx = np.argsort(mean_abs_shap)[::-1][:top_n]
        
        if len(top_features_idx) >= 2:
            plt.figure(figsize=self.style['figure_size'])
            
            # Create interaction plot
            shap.dependence_plot(
                top_features_idx[0],
                shap_values,
                X_sample,
                interaction_index=top_features_idx[1],
                show=False
            )
            
            feature1 = X_sample.columns[top_features_idx[0]]
            feature2 = X_sample.columns[top_features_idx[1]]
            plt.title(f'SHAP Interaction: {feature1} vs {feature2} - {self.model_name}', fontsize=14)
            plt.tight_layout()
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=self.style['dpi'], bbox_inches='tight')
            plt.close()


def _create_safe_explainer(model, model_name: str, X_sample: pd.DataFrame):
    """
    Create SHAP explainer with model-specific handling and safety features.
    
    Args:
        model: The model object
        model_name: Name of the model
        X_sample: Sample data for explainer initialization
        
    Returns:
        SHAP explainer object or None if creation fails
    """
    try:
        if "CatBoost" in model_name:
            # CatBoost specific handling with multiple fallback strategies
            try:
                # Try with tree_path_dependent for categorical support
                explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
                print(f"  ✓ Created CatBoost TreeExplainer with tree_path_dependent")
            except Exception as e:
                print(f"  ⚠ CatBoost TreeExplainer with tree_path_dependent failed: {str(e)[:100]}")
                try:
                    # Try standard TreeExplainer without tree_path_dependent
                    explainer = shap.TreeExplainer(model)
                    print(f"  ✓ Created CatBoost TreeExplainer (standard mode)")
                except Exception as e2:
                    print(f"  ⚠ CatBoost standard TreeExplainer failed: {str(e2)[:100]}")
                    try:
                        # Use sampling-based Explainer for CatBoost (slower but more robust)
                        # Reduce sample size for memory efficiency
                        sample_size = min(20, len(X_sample))
                        explainer = shap.Explainer(model.predict, X_sample[:sample_size])
                        print(f"  ✓ Created CatBoost sampling Explainer with {sample_size} samples")
                    except Exception as e3:
                        print(f"  ✗ All CatBoost explainer methods failed: {str(e3)[:100]}")
                        # Last resort: return None, but log it properly
                        print(f"  ✗ WARNING: Skipping SHAP for {model_name} - all methods failed")
                        return None
        elif "LightGBM" in model_name or "lightgbm" in str(type(model)).lower():
            try:
                explainer = shap.TreeExplainer(model)
                print(f"  ✓ Created LightGBM TreeExplainer")
            except Exception as e:
                print(f"  ⚠ LightGBM TreeExplainer failed: {str(e)[:100]}")
                try:
                    # Fallback to sampling-based explainer
                    sample_size = min(30, len(X_sample))
                    explainer = shap.Explainer(model.predict, X_sample[:sample_size])
                    print(f"  ✓ Created LightGBM sampling Explainer with {sample_size} samples")
                except Exception as e2:
                    print(f"  ✗ LightGBM explainer failed: {str(e2)[:100]}")
                    return None
        elif "XGBoost" in model_name or "xgboost" in str(type(model)).lower():
            explainer = shap.TreeExplainer(model)
            print(f"  ✓ Created XGBoost TreeExplainer")
        elif "elastic" in model_name.lower() or "linear" in model_name.lower():
            explainer = shap.LinearExplainer(model, X_sample)
            print(f"  ✓ Created LinearExplainer")
        else:
            # Generic fallback
            try:
                explainer = shap.TreeExplainer(model)
                print(f"  ✓ Created generic TreeExplainer")
            except:
                # Last resort: kernel explainer (slow)
                print(f"  ⚠ Falling back to KernelExplainer (slow)")
                explainer = shap.KernelExplainer(model.predict, X_sample[:50])
        
        return explainer
        
    except Exception as e:
        print(f"  ✗ Failed to create explainer: {str(e)[:200]}")
        return None


def create_shap_visualizations_safe(model_data: Dict[str, Any], output_dir: Path, 
                                   max_samples: int = 30, skip_existing: bool = True) -> List[Path]:
    """
    Safe wrapper around create_shap_visualizations with memory management and safety features.
    
    Args:
        model_data: Dictionary containing model information
        output_dir: Directory to save visualizations
        max_samples: Maximum samples for SHAP computation (default 30 for safety)
        skip_existing: Skip if visualizations already exist
        
    Returns:
        List of paths to created visualizations
    """
    model_name = model_data.get('model_name', 'Unknown')
    model_output_dir = output_dir / model_name.replace(' ', '_')
    
    # Check if outputs already exist
    if skip_existing and model_output_dir.exists() and len(list(model_output_dir.glob("*.png"))) >= 5:
        print(f"  ⚠ SHAP plots already exist for {model_name}, skipping...")
        return list(model_output_dir.glob("*.png"))
    
    # Memory monitoring
    initial_memory = check_memory_usage()
    print(f"  Initial memory: {initial_memory:.2f} GB")
    
    try:
        # Call enhanced version with safety limits
        paths = create_shap_visualizations(
            model_data, 
            output_dir, 
            sample_size=max_samples
        )
        return paths
    finally:
        # Cleanup
        plt.close('all')
        gc.collect()
        final_memory = check_memory_usage()
        print(f"  Memory after cleanup: {final_memory:.2f} GB (delta: {final_memory - initial_memory:.2f} GB)")


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
        
        # Create explainer using safe method
        explainer = _create_safe_explainer(model, model_name, X_sample)
        if explainer is None:
            print(f"  ✗ Could not create explainer for {model_name}, skipping SHAP analysis")
            return created_paths
        
        # Calculate SHAP values
        print(f"  Computing SHAP values for {n_samples} samples...")
        try:
            shap_values = explainer.shap_values(X_sample)
            
            # Handle multi-class output
            if isinstance(shap_values, list):
                shap_values = shap_values[0]  # Use first class for regression
                
            print(f"  ✓ SHAP values computed: shape {shap_values.shape}")
        except Exception as e:
            print(f"  ✗ Failed to compute SHAP values: {str(e)[:200]}")
            return created_paths
        
        # Create visualizer
        visualizer = SHAPVisualizer(model_data)
        
        # Extract output_dir from config if it's a dict/object
        if isinstance(output_dir, dict):
            output_dir_path = Path(output_dir.get('output_dir', settings.VISUALIZATION_DIR / 'shap'))
        elif hasattr(output_dir, 'output_dir'):
            output_dir_path = Path(output_dir.output_dir)
        else:
            output_dir_path = Path(output_dir)
        
        # Create output directory
        model_output_dir = output_dir_path / model_name.replace(' ', '_')
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create plots with granular error handling
        plot_functions = [
            # (plot_name, function, description)
            ('summary_bar', lambda: visualizer.create_shap_summary_plot(
                shap_values, X_sample, 
                model_output_dir / f"{model_name}_shap_summary_bar.png"
            ), "Summary bar plot"),
            
            ('summary_dot', lambda: visualizer.create_shap_dot_summary_plot(
                shap_values, X_sample,
                model_output_dir / f"{model_name}_shap_summary_dot.png"
            ), "Summary dot plot"),
            
            ('waterfall', lambda: visualizer.create_shap_waterfall_plot(
                shap_values, X_sample, 0,
                model_output_dir / f"{model_name}_shap_waterfall.png"
            ), "Waterfall plot"),
            
            ('interaction', lambda: visualizer.create_shap_interaction_plot(
                shap_values, X_sample,
                model_output_dir / f"{model_name}_shap_interaction.png"
            ), "Interaction plot"),
        ]
        
        # Execute each plot type with error handling
        for plot_name, plot_func, description in plot_functions:
            try:
                plot_func()
                plot_path = model_output_dir / f"{model_name}_shap_{plot_name}.png"
                if plot_path.exists():
                    created_paths.append(plot_path)
                    print(f"    ✓ {description} created")
            except Exception as e:
                print(f"    ✗ {description} failed: {str(e)[:100]}")
        
        # Force plots (returns list)
        try:
            force_paths = visualizer.create_shap_force_plots(
                explainer, shap_values, X_sample, model_output_dir, n_instances=3
            )
            created_paths.extend(force_paths)
            if force_paths:
                print(f"    ✓ Force plots created ({len(force_paths)} instances)")
        except Exception as e:
            print(f"    ✗ Force plots failed: {str(e)[:100]}")
        
        # Dependence plots for top features
        try:
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            top_features_idx = np.argsort(mean_abs_shap)[::-1][:5]  # Top 5 features
            
            dependence_created = 0
            for idx in top_features_idx:
                try:
                    feature_name = X_sample.columns[idx]
                    dependence_path = model_output_dir / f"{model_name}_shap_dependence_{feature_name}.png"
                    visualizer.create_shap_dependence_plot(shap_values, X_sample, feature_name, dependence_path)
                    created_paths.append(dependence_path)
                    dependence_created += 1
                except Exception as e:
                    print(f"    ✗ Dependence plot for feature {idx} failed: {str(e)[:50]}")
            
            if dependence_created > 0:
                print(f"    ✓ Dependence plots created ({dependence_created} features)")
        except Exception as e:
            print(f"    ✗ Dependence plots failed: {str(e)[:100]}")
        
        # Categorical feature plots
        try:
            categorical_features = identify_categorical_features(X_sample, model_data)
            cat_created = 0
            for cat_feature in categorical_features[:3]:  # Top 3 categorical features
                if cat_feature in X_sample.columns:
                    try:
                        cat_path = model_output_dir / f"{model_name}_shap_categorical_{cat_feature}.png"
                        visualizer.create_categorical_shap_plot(shap_values, X_sample, cat_feature, cat_path)
                        created_paths.append(cat_path)
                        cat_created += 1
                    except Exception as e:
                        print(f"    ✗ Categorical plot for {cat_feature} failed: {str(e)[:50]}")
            
            if cat_created > 0:
                print(f"    ✓ Categorical plots created ({cat_created} features)")
        except Exception as e:
            print(f"    ✗ Categorical plots failed: {str(e)[:100]}")
        
        print(f"  Total SHAP visualizations created: {len(created_paths)}")
        
    except Exception as e:
        print(f"  ✗ Error creating SHAP visualizations for {model_data.get('model_name', 'Unknown')}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up matplotlib
        plt.close('all')
    
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


def create_all_shap_visualizations(models: Dict[str, Any], output_dir: Path, 
                                  use_safe_mode: bool = True) -> Dict[str, List[Path]]:
    """
    Create SHAP visualizations for all models.
    
    Args:
        models: Dictionary of models
        output_dir: Directory to save visualizations
        use_safe_mode: Use memory-safe version with limits (default True)
        
    Returns:
        Dictionary mapping model names to lists of created paths
    """
    all_paths = {}
    shap_dir = output_dir / "shap"
    shap_dir.mkdir(parents=True, exist_ok=True)
    
    # Track memory if in safe mode
    if use_safe_mode:
        print(f"Starting SHAP generation in SAFE MODE (memory limits enabled)")
        print(f"Initial memory usage: {check_memory_usage():.2f} GB")
    
    for model_name, model_data in models.items():
        print(f"\nCreating SHAP visualizations for {model_name}...")
        
        if use_safe_mode:
            # Use safe version with memory management
            paths = create_shap_visualizations_safe(model_data, shap_dir)
        else:
            # Use standard version
            paths = create_shap_visualizations(model_data, shap_dir)
            
        all_paths[model_name] = paths
    
    # Create README
    readme_path = shap_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write("# SHAP Visualizations\n\n")
        f.write("This directory contains SHAP (SHapley Additive exPlanations) visualizations for all models.\n\n")
        f.write("## Visualization Types\n\n")
        f.write("- **Summary Plots (Bar)**: Show feature importance rankings\n")
        f.write("- **Summary Plots (Dot)**: Show feature importance and impact distribution\n")
        f.write("- **Waterfall Plots**: Explain individual predictions\n")
        f.write("- **Force Plots**: Show feature contributions for specific instances\n")
        f.write("- **Dependence Plots**: Show relationship between feature values and SHAP values\n")
        f.write("- **Interaction Plots**: Show interactions between top features\n")
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
    
    # Final memory report and cleanup
    if use_safe_mode:
        gc.collect()
        final_memory = check_memory_usage()
        print(f"\nFinal memory usage: {final_memory:.2f} GB")
        print(f"Total SHAP visualizations created: {sum(len(paths) for paths in all_paths.values())}")
    
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
    # Call the new function that creates separate Base and Yeo plots
    paths = create_separated_model_comparison_shap_plots(models, output_dir)
    # Return the first path for backward compatibility
    return paths[0] if paths else None


def create_separated_model_comparison_shap_plots(models: Dict[str, Any], output_dir: Path) -> List[Path]:
    """
    Create separate SHAP comparison plots for Base and Yeo datasets.
    
    Args:
        models: Dictionary of models
        output_dir: Directory to save the plots
        
    Returns:
        List of paths to created plots
    """
    created_paths = []
    
    # Create separate plots for Base and Yeo datasets
    for dataset_type in ['Base', 'Yeo']:
        plot_path = _create_dataset_specific_shap_plot(models, output_dir, dataset_type)
        if plot_path:
            created_paths.append(plot_path)
    
    return created_paths


def _create_dataset_specific_shap_plot(models: Dict[str, Any], output_dir: Path, dataset_type: str) -> Optional[Path]:
    """
    Create a SHAP comparison plot for a specific dataset type (Base or Yeo).
    
    Args:
        models: Dictionary of models
        output_dir: Directory to save the plot
        dataset_type: 'Base' or 'Yeo'
        
    Returns:
        Path to the created plot, or None if failed
    """
    try:
        print(f"\nCreating SHAP comparison plot for {dataset_type} models...")
        
        # Collect SHAP values for each model
        # Include ALL Optuna-optimized models (not just first valid)
        model_shap_data = {}
        
        # Track features separately for each model to identify common features
        model_features = {}
        
        for model_name, model_data in models.items():
            # Skip non-tree models
            model_type = model_data.get('model_type', '').lower()
            if not any(tree_type in model_type for tree_type in ['xgboost', 'lightgbm', 'catboost', 'xgb', 'lgb']):
                continue
            
            # Skip Random models for main pipeline plots
            if 'Random' in model_name:
                continue
            
            # Filter by dataset type
            if dataset_type not in model_name:
                continue
            
            # Include only Optuna-optimized models
            if 'optuna' not in model_name.lower():
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
                display_name = f'XGBoost_{dataset_type}'
            elif 'lightgbm' in model_type or 'lgb' in model_type:
                display_name = f'LightGBM_{dataset_type}'
            elif 'catboost' in model_type:
                display_name = f'CatBoost_{dataset_type}'
            else:
                display_name = model_name
            
            # Create explainer and calculate SHAP values
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)
                
                # Handle multi-class output
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]
                
                # Calculate mean absolute SHAP values
                mean_abs_shap = np.abs(shap_values).mean(axis=0)
                
                # Store results with feature names
                model_shap_data[display_name] = {
                    'shap_values': mean_abs_shap,
                    'feature_names': list(X_sample.columns)
                }
                
                # Track features for this model
                model_features[display_name] = list(X_sample.columns)
                
                print(f"  Calculated SHAP values for {model_name} - {len(X_sample.columns)} features")
                
            except Exception as e:
                print(f"  Error calculating SHAP values for {model_name}: {e}")
                continue
        
        if len(model_shap_data) < 2:
            print(f"  Not enough {dataset_type} models with SHAP values for comparison")
            return None
        
        # Find common features across all models
        common_features = set(model_features[list(model_features.keys())[0]])
        for features in model_features.values():
            common_features = common_features.intersection(set(features))
        
        print(f"  Found {len(common_features)} common features across {dataset_type} models")
        
        # Create DataFrame for comparison using only common features
        # This ensures fair comparison across models
        shap_df = pd.DataFrame(index=sorted(common_features))
        
        for display_name, data in model_shap_data.items():
            # Create a Series for common features only
            feature_values = pd.Series(index=sorted(common_features), dtype=float)
            
            # Fill in the values for common features
            for i, feat in enumerate(data['feature_names']):
                if feat in common_features:
                    feature_values[feat] = data['shap_values'][i]
            
            shap_df[display_name] = feature_values
        
        # Remove any rows with all NaN values (shouldn't happen with common features)
        shap_df = shap_df.dropna(how='all')
        
        # Select top features based on maximum importance across models
        max_importance = shap_df.max(axis=1)
        top_n = 15  # Fixed number of top features to show
        
        # Ensure we don't try to select more features than available
        n_features_to_show = min(top_n, len(shap_df))
        top_features = max_importance.nlargest(n_features_to_show).index
        shap_df_top = shap_df.loc[top_features]
        
        print(f"  Selected top {n_features_to_show} features for visualization")
        
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
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Normalized |SHAP Value| (0-1)', rotation=270, labelpad=20)
        
        # Add text annotations
        for i in range(len(shap_df_normalized.index)):
            for j in range(len(shap_df_normalized.columns)):
                text = ax.text(j, i, f'{shap_df_normalized.iloc[i, j]:.3f}',
                             ha="center", va="center", 
                             color="white" if shap_df_normalized.iloc[i, j] > 0.5 else "black",
                             fontsize=8)
        
        # Add grid
        ax.set_xticks(np.arange(len(shap_df_normalized.columns) + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(shap_df_normalized.index) + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
        ax.tick_params(which="minor", size=0)
        
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        # Updated title to include feature count information
        plt.title(f'SHAP Feature Importance Comparison - {dataset_type} Dataset (Top {n_features_to_show} Features)\n(Normalized 0-1 Scale per Model)', 
                  fontsize=14)
        plt.tight_layout()
        
        # Save the plot
        output_path = output_dir / f"model_comparison_shap_{dataset_type.lower()}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Created {dataset_type} model comparison SHAP plot: {output_path}")
        print(f"  Displayed {n_features_to_show} features from {len(common_features)} common features")
        return output_path
        
    except Exception as e:
        print(f"  Error creating {dataset_type} model comparison SHAP plot: {e}")
        import traceback
        traceback.print_exc()
        return None


# Make functions available at module level
__all__ = [
    'create_shap_visualizations', 
    'create_shap_visualizations_safe',
    'create_all_shap_visualizations', 
    'SHAPVisualizer', 
    'identify_categorical_features', 
    'create_model_comparison_shap_plot',
    'create_separated_model_comparison_shap_plots',
    '_create_dataset_specific_shap_plot',
    'check_memory_usage'
]