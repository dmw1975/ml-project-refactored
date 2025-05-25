"""Cross-validation score distribution plots."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

from visualization_new.core.interfaces import ModelData, VisualizationConfig
from visualization_new.core.base import ModelViz
from visualization_new.core.registry import get_adapter_for_model
from visualization_new.components.formats import format_figure_for_export, save_figure
from config import settings


class CVDistributionPlot(ModelViz):
    """Plot cross-validation score distributions for models."""
    
    def __init__(
        self, 
        model_data: Union[ModelData, Dict[str, Any]], 
        config: Optional[Union[Dict[str, Any], VisualizationConfig]] = None
    ):
        """
        Initialize CV distribution plot.
        
        Args:
            model_data: Model data or adapter
            config: Visualization configuration
        """
        # Convert model data to adapter if needed
        if not isinstance(model_data, ModelData):
            model_data = get_adapter_for_model(model_data)
            
        super().__init__(model_data, config)
        
    def plot(self) -> Optional[plt.Figure]:
        """
        Create CV score distribution plot.
        
        Returns:
            matplotlib.figure.Figure: The created figure, or None if no CV data available
        """
        # Get raw model data to access CV results
        raw_data = self.model.get_raw_model_data()
        
        # Try to find CV scores in various locations
        cv_scores = None
        
        # Check for ElasticNet parameter results (stored separately)
        if 'param_results' in raw_data:
            # This is likely from ElasticNet parameter search
            param_results = raw_data['param_results']
            if param_results and isinstance(param_results, list):
                # Extract all CV RMSE scores from all parameter combinations
                all_cv_scores = []
                for result in param_results:
                    if 'cv_results' in result and isinstance(result['cv_results'], pd.DataFrame):
                        cv_df = result['cv_results']
                        if 'rmse_folds' in cv_df.columns:
                            # Each row has a list of CV fold scores
                            for fold_scores in cv_df['rmse_folds']:
                                all_cv_scores.extend(fold_scores)
                if all_cv_scores:
                    cv_scores = np.array(all_cv_scores)
                    
        # Check for direct CV scores in model data
        elif 'cv_scores' in raw_data:
            cv_scores = np.array(raw_data['cv_scores'])
            
        # Check for rmse_folds (from parameter search)
        elif 'rmse_folds' in raw_data:
            cv_scores = np.array(raw_data['rmse_folds'])
            
        # Check if this is from a stored parameter search
        elif hasattr(self.model.model, 'cv_results_') and hasattr(self.model.model.cv_results_, 'get'):
            # Scikit-learn GridSearchCV or similar
            cv_results = self.model.model.cv_results_
            # Look for split scores
            split_keys = [k for k in cv_results.keys() if k.startswith('split') and k.endswith('_test_score')]
            if split_keys:
                all_scores = []
                for key in split_keys:
                    all_scores.extend(cv_results[key])
                cv_scores = -np.array(all_scores)  # Convert negative MSE to RMSE
                cv_scores = np.sqrt(cv_scores)
        
        if cv_scores is None or len(cv_scores) == 0:
            print(f"No cross-validation scores found for {self.model.model_name}")
            return None
            
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create distribution plot
        sns.histplot(cv_scores, kde=True, ax=ax, bins=30, alpha=0.7, color='#3498db')
        
        # Add vertical lines for mean and median
        mean_score = np.mean(cv_scores)
        median_score = np.median(cv_scores)
        std_score = np.std(cv_scores)
        
        ax.axvline(mean_score, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_score:.4f}')
        ax.axvline(median_score, color='green', linestyle='--', linewidth=2, label=f'Median: {median_score:.4f}')
        
        # Add shaded region for ±1 std
        ax.axvspan(mean_score - std_score, mean_score + std_score, 
                   alpha=0.2, color='red', label=f'±1 SD: {std_score:.4f}')
        
        # Labels and title
        model_name = self.model.get_metadata().get('model_name', 'Model')
        ax.set_title(f'Cross-Validation RMSE Distribution: {model_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('RMSE', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        
        # Add statistics text box
        stats_text = f'N = {len(cv_scores)}\nMean = {mean_score:.4f}\nStd = {std_score:.4f}\nMin = {np.min(cv_scores):.4f}\nMax = {np.max(cv_scores):.4f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Legend
        ax.legend(loc='upper right')
        
        # Grid
        ax.grid(True, alpha=0.3)
        
        # Format figure
        format_figure_for_export(fig)
        
        return fig
        
    def save(self, fig: plt.Figure) -> Optional[str]:
        """Save the CV distribution plot."""
        # Determine output directory based on model type
        model_name = self.model.get_metadata().get('model_name', 'model')
        
        if 'elasticnet' in model_name.lower():
            output_dir = settings.VISUALIZATION_DIR / "performance" / "elasticnet"
        elif 'xgb' in model_name.lower():
            output_dir = settings.VISUALIZATION_DIR / "performance" / "xgboost"
        elif 'lightgbm' in model_name.lower():
            output_dir = settings.VISUALIZATION_DIR / "performance" / "lightgbm"
        elif 'catboost' in model_name.lower():
            output_dir = settings.VISUALIZATION_DIR / "performance" / "catboost"
        else:
            output_dir = settings.VISUALIZATION_DIR / "performance"
            
        # Update config with output directory
        self.config['output_dir'] = output_dir
        
        # Generate filename
        clean_name = model_name.lower().replace(' ', '_')
        filename = f"{clean_name}_cv_rmse_distribution"
        
        return save_figure(fig, filename, self.config)


def plot_cv_distribution_for_elasticnet(param_results_path: Optional[Path] = None) -> Optional[str]:
    """
    Create CV RMSE distribution plot specifically for ElasticNet models.
    
    This function loads ElasticNet parameter search results and creates
    a distribution plot of all CV RMSE scores across all parameter combinations.
    
    Args:
        param_results_path: Path to elasticnet_params.pkl file. If None, uses default path.
        
    Returns:
        Path to saved plot or None if failed
    """
    # Load parameter results
    if param_results_path is None:
        param_results_path = settings.MODEL_DIR / "elasticnet_params.pkl"
        
    if not param_results_path.exists():
        print(f"ElasticNet parameter results not found at {param_results_path}")
        return None
        
    try:
        from utils.io import load_model
        
        # Load from the specified path
        if param_results_path.parent.name == 'old':
            param_results = load_model("elasticnet_params.pkl", param_results_path.parent)
        else:
            param_results = load_model("elasticnet_params.pkl", settings.MODEL_DIR)
        
        if not param_results:
            print("No parameter results found in file")
            return None
            
        # Extract all CV scores
        all_cv_scores = []
        dataset_scores = {}  # Store scores by dataset
        
        for result in param_results:
            dataset_name = result.get('dataset', 'Unknown')
            cv_results_df = result.get('cv_results')
            
            if cv_results_df is not None and 'rmse_folds' in cv_results_df.columns:
                dataset_cv_scores = []
                for fold_scores in cv_results_df['rmse_folds']:
                    dataset_cv_scores.extend(fold_scores)
                    all_cv_scores.extend(fold_scores)
                    
                dataset_scores[dataset_name] = np.array(dataset_cv_scores)
        
        if not all_cv_scores:
            print("No CV scores found in parameter results")
            return None
            
        # Create figure with subplots for overall and per-dataset distributions
        n_datasets = len(dataset_scores)
        fig_height = 6 + (n_datasets - 1) * 2  # Adjust height based on number of datasets
        
        fig, axes = plt.subplots(n_datasets + 1, 1, figsize=(10, fig_height))
        
        # Ensure axes is always a list
        if n_datasets == 0:
            axes = [axes]
        
        # Overall distribution
        ax = axes[0]
        all_scores = np.array(all_cv_scores)
        
        sns.histplot(all_scores, kde=True, ax=ax, bins=30, alpha=0.7, color='#3498db')
        
        mean_score = np.mean(all_scores)
        median_score = np.median(all_scores)
        std_score = np.std(all_scores)
        
        ax.axvline(mean_score, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_score:.4f}')
        ax.axvline(median_score, color='green', linestyle='--', linewidth=2, label=f'Median: {median_score:.4f}')
        ax.axvspan(mean_score - std_score, mean_score + std_score, 
                   alpha=0.2, color='red', label=f'±1 SD: {std_score:.4f}')
        
        ax.set_title('Overall Cross-Validation RMSE Distribution (All ElasticNet Models)', fontsize=14, fontweight='bold')
        ax.set_xlabel('RMSE', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f'N = {len(all_scores)}\nMean = {mean_score:.4f}\nStd = {std_score:.4f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Per-dataset distributions
        colors = ['#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
        for i, (dataset_name, scores) in enumerate(dataset_scores.items()):
            ax = axes[i + 1]
            color = colors[i % len(colors)]
            
            sns.histplot(scores, kde=True, ax=ax, bins=20, alpha=0.7, color=color)
            
            mean_score = np.mean(scores)
            ax.axvline(mean_score, color='black', linestyle='--', linewidth=2)
            
            ax.set_title(f'{dataset_name} Dataset', fontsize=12)
            ax.set_xlabel('RMSE', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Add mean text
            ax.text(0.02, 0.98, f'Mean: {mean_score:.4f}', transform=ax.transAxes, 
                    verticalalignment='top', fontsize=9, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save figure
        output_dir = settings.VISUALIZATION_DIR / "performance" / "elasticnet"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / "elasticnet_cv_rmse_distribution.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Created ElasticNet CV RMSE distribution plot: {output_path}")
        return str(output_path)
        
    except Exception as e:
        print(f"Error creating ElasticNet CV distribution plot: {e}")
        return None