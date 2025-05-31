"""DEPRECATED: Sector stratification visualizations for model train/test splits.

This module has been replaced by sector_weights.py which provides focused
visualizations of sector weight distributions.

DO NOT USE THIS MODULE FOR NEW VISUALIZATIONS.
"""

# This module is deprecated in favor of sector_weights.py
# The code is kept for reference only but won't generate files
_DEPRECATED_MODULE = True

import sys
import os
from pathlib import Path

# Add project root to the path so Python can find the modules
project_root = Path(__file__).parent.parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except (ImportError, SyntaxError) as e:
    print(f"Warning: Could not import seaborn: {e}")
    sns = None
from typing import Dict, Any, Optional, Union, List, Tuple
from scipy.stats import entropy

from src.visualization.core.interfaces import ModelData, VisualizationConfig
from src.visualization.core.base import BaseViz, ComparativeViz
from src.visualization.core.registry import get_adapter_for_model
from src.visualization.components.annotations import add_value_labels as original_add_value_labels

# Custom version that supports decimal_places
def add_value_labels(bars, decimal_places=2, fontsize=8, color='black', vertical_offset=0.01):
    """
    Add value labels to bars in bar chart with customizable decimal places.

    Args:
        bars: Bar container
        decimal_places: Number of decimal places
        fontsize: Font size
        color: Text color
        vertical_offset: Vertical offset as fraction of bar height
    """
    ax = bars[0].axes
    for bar in bars:
        height = bar.get_height()
        if height == 0 or np.isnan(height):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + (abs(height) * vertical_offset),
            f"{height:.{decimal_places}f}",
            ha='center',
            va='bottom',
            fontsize=fontsize,
            color=color
        )
from src.visualization.components.formats import save_figure
from src.visualization.utils.io import ensure_dir, load_all_models

# Import project settings
from config import settings


class SectorStratificationPlot(BaseViz):
    """DEPRECATED: Sector stratification visualization to compare train-test distribution.

    This class is deprecated. Use functions from sector_weights.py instead.
    """
    
    def __init__(
        self, 
        model_data_list: Optional[List[Dict[str, Any]]] = None,
        config: Optional[Union[Dict[str, Any], VisualizationConfig]] = None
    ):
        """
        Initialize sector stratification visualization.
        
        Args:
            model_data_list: List of model data dictionaries
            config: Visualization configuration
        """
        super().__init__(config)

        # Force saving to be disabled
        if _DEPRECATED_MODULE:
            self.config.update(save=False)

        # Set default output directory if not provided (but won't be used)
        if self.config.get('output_dir') is None:
            self.config.update(output_dir=settings.VISUALIZATION_DIR / "sectors" / "stratification")

        # Only ensure directory exists if not deprecated
        if not _DEPRECATED_MODULE:
            ensure_dir(self.config.get('output_dir'))
        
        # Load models if not provided
        if model_data_list is None:
            try:
                model_data_list = []

                # Import utils.io directly (not from visualization_new)
                import sys
                from pathlib import Path

                # Add project root to path if needed
                project_root = Path(__file__).parent.parent.parent.absolute()
                if str(project_root) not in sys.path:
                    sys.path.append(str(project_root))

                # Import the actual io module from the main project
                from utils import io
                from config import settings

                model_files = {
                    'xgboost': 'xgboost_models.pkl',
                    'lightgbm': 'lightgbm_models.pkl',
                    'catboost': 'catboost_models.pkl',
                    'elasticnet': 'elasticnet_models.pkl',
                    'linear': 'linear_regression_models.pkl',
                    'sector': 'sector_models.pkl'
                }

                # Load each model type
                for model_type, filename in model_files.items():
                    try:
                        models = io.load_model(filename, settings.MODEL_DIR)
                        if models:
                            # Add model_type field if not present
                            if isinstance(models, dict):
                                for name, model in models.items():
                                    model_data = model.copy()
                                    if 'model_type' not in model_data:
                                        model_data['model_type'] = model_type
                                    model_data_list.append(model_data)
                                print(f"Loaded {len(models)} {model_type} models")
                    except Exception as e:
                        print(f"Could not load {model_type} models: {e}")

                if not model_data_list:
                    raise ValueError("No models could be loaded")

            except Exception as e:
                raise ValueError(f"Failed to load models: {e}")
        
        self.model_data_list = model_data_list
        print(f"Initialized with {len(self.model_data_list)} models")
        
        # Process models to extract sector distributions
        self.sector_distributions = self._extract_all_sector_distributions()
    
    def _extract_all_sector_distributions(self) -> Dict[str, Dict[str, Any]]:
        """
        Extract sector distributions from all models.
        
        Returns:
            Dictionary mapping model names to their sector distribution data
        """
        distributions = {}
        
        for model_data in self.model_data_list:
            model_name = model_data.get('model_name', 'unknown')
            try:
                distribution = self._extract_sector_distribution(model_data)
                if distribution:
                    distributions[model_name] = distribution
            except Exception as e:
                print(f"Could not extract sector distribution for {model_name}: {e}")
        
        print(f"Extracted sector distributions for {len(distributions)} models")
        return distributions
    
    def _extract_sector_distribution(self, model_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Extract sector distribution from a single model.
        
        Args:
            model_data: Model data dictionary
            
        Returns:
            Dictionary with sector distribution information or None if not available
        """
        model_name = model_data.get('model_name', 'unknown')
        model_type = model_data.get('model_type', 'unknown')
        
        # Basic distribution data
        distribution = {
            'model_name': model_name,
            'model_type': model_type,
            'n_companies': model_data.get('n_companies', 0),
            'n_companies_train': model_data.get('n_companies_train', 0),
            'n_companies_test': model_data.get('n_companies_test', 0),
        }
        
        # Calculate train/test split ratio
        if distribution['n_companies'] > 0:
            distribution['test_ratio'] = distribution['n_companies_test'] / distribution['n_companies']
            distribution['train_ratio'] = distribution['n_companies_train'] / distribution['n_companies']
        
        # Get sector information
        if 'X_test' in model_data:
            X_test = model_data['X_test']
            
            # Find sector columns
            sector_cols = [col for col in X_test.columns if col.startswith('gics_sector_') or col.startswith('sector_')]
            
            if sector_cols:
                # Calculate sector distribution in test set
                test_sector_dist = {}
                for sector_col in sector_cols:
                    sector_name = sector_col.replace('gics_sector_', '').replace('sector_', '')
                    n_companies_in_sector_test = X_test[sector_col].sum()
                    pct_in_sector_test = n_companies_in_sector_test / len(X_test) * 100
                    test_sector_dist[sector_name] = {
                        'count': n_companies_in_sector_test,
                        'percentage': pct_in_sector_test
                    }
                
                distribution['test_sector_distribution'] = test_sector_dist
                
                # Calculate expected sector distribution in train set based on overall counts
                # Note: We don't have actual train set data, so we estimate
                train_sector_dist = {}
                for sector_name, test_data in test_sector_dist.items():
                    # For each sector, we need to calculate how many companies should be in train set
                    # based on the test count and the overall train/test ratio
                    test_count = test_data['count']
                    
                    # Calculate total sector count using test percentage and total test count
                    sector_pct = test_data['percentage'] / 100
                    total_in_sector = int(test_count / sector_pct * 100) if sector_pct > 0 else 0
                    
                    # Train count is total minus test count
                    train_count = total_in_sector - test_count
                    
                    # Calculate percentage in train set
                    train_pct = (train_count / distribution['n_companies_train']) * 100 if distribution['n_companies_train'] > 0 else 0
                    
                    train_sector_dist[sector_name] = {
                        'count': train_count,
                        'percentage': train_pct
                    }
                
                distribution['train_sector_distribution'] = train_sector_dist
                
                # Calculate distribution similarity metrics
                test_dist = np.array([test_sector_dist[sector]['percentage'] for sector in test_sector_dist])
                train_dist = np.array([train_sector_dist[sector]['percentage'] for sector in train_sector_dist])
                
                # Normalize distributions to sum to 1 for KL divergence
                test_dist_norm = test_dist / np.sum(test_dist) if np.sum(test_dist) > 0 else np.ones_like(test_dist) / len(test_dist)
                train_dist_norm = train_dist / np.sum(train_dist) if np.sum(train_dist) > 0 else np.ones_like(train_dist) / len(train_dist)
                
                # Calculate KL divergence (add small value to avoid division by zero)
                epsilon = 1e-10
                test_dist_norm = test_dist_norm + epsilon
                train_dist_norm = train_dist_norm + epsilon
                
                # Re-normalize
                test_dist_norm = test_dist_norm / np.sum(test_dist_norm)
                train_dist_norm = train_dist_norm / np.sum(train_dist_norm)
                
                # Calculate KL divergence in both directions and take average
                kl_train_test = entropy(train_dist_norm, test_dist_norm)
                kl_test_train = entropy(test_dist_norm, train_dist_norm)
                
                distribution['kl_divergence'] = (kl_train_test + kl_test_train) / 2
                
                return distribution
        
        # For sector-specific models
        if 'sector' in model_data:
            specific_sector = model_data['sector']
            distribution['specific_sector'] = specific_sector
            return distribution
            
        # If no sector information found
        print(f"No sector information found for model {model_name}")
        return None
    
    def plot_sector_distribution_comparison(self, model_name: str) -> Optional[plt.Figure]:
        """
        Create stacked bar chart comparing train vs test sector distributions.
        
        Args:
            model_name: Name of the model to visualize
            
        Returns:
            matplotlib.figure.Figure: The created figure or None if not available
        """
        if model_name not in self.sector_distributions:
            print(f"No sector distribution data for model {model_name}")
            return None
        
        model_dist = self.sector_distributions[model_name]
        
        if 'test_sector_distribution' not in model_dist or 'train_sector_distribution' not in model_dist:
            print(f"Missing sector distribution data for model {model_name}")
            return None
        
        # Prepare data for plotting
        sectors = list(model_dist['test_sector_distribution'].keys())
        test_pcts = [model_dist['test_sector_distribution'][s]['percentage'] for s in sectors]
        train_pcts = [model_dist['train_sector_distribution'][s]['percentage'] for s in sectors]
        
        # Sort sectors by test percentage
        sort_idx = np.argsort(test_pcts)[::-1]  # Descending
        sectors = [sectors[i] for i in sort_idx]
        test_pcts = [test_pcts[i] for i in sort_idx]
        train_pcts = [train_pcts[i] for i in sort_idx]
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.config.get('figsize', (12, 8)))
        
        # Set width of bars
        barWidth = 0.35
        
        # Set position of bars on X axis
        r1 = np.arange(len(sectors))
        r2 = [x + barWidth for x in r1]
        
        # Create stacked bar chart
        bars1 = ax.bar(r1, train_pcts, width=barWidth, label='Train', color='#3498db', alpha=0.7)
        bars2 = ax.bar(r2, test_pcts, width=barWidth, label='Test', color='#e74c3c', alpha=0.7)
        
        # Add labels, title and axis ticks
        ax.set_xlabel('Sector')
        ax.set_ylabel('Percentage of Companies')
        ax.set_title(f'Sector Distribution Comparison for {model_name}', 
                    fontsize=self.config.get('title_fontsize', 14))
        ax.set_xticks([r + barWidth/2 for r in range(len(sectors))])
        ax.set_xticklabels(sectors, rotation=45, ha='right')
        
        # Add KL divergence as a measure of distribution similarity
        if 'kl_divergence' in model_dist:
            kl_div = model_dist['kl_divergence']
            ax.annotate(f"KL Divergence: {kl_div:.4f}\n(lower is better)",
                       xy=(0.95, 0.95), xycoords='axes fraction',
                       ha='right', va='top',
                       bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3))
        
        # Add grid
        ax.grid(axis='y', alpha=0.3)
        
        # Add legend
        ax.legend()
        
        # Add value labels
        for bars in [bars1, bars2]:
            add_value_labels(bars, decimal_places=1)
        
        plt.tight_layout()
        
        # Save figure if requested
        if self.config.get('save', True):
            save_figure(
                fig=fig,
                filename=f"{model_name}_sector_distribution",
                output_dir=self.config.get('output_dir'),
                dpi=self.config.get('dpi', 300),
                format=self.config.get('format', 'png')
            )
        
        # Show figure if requested
        if self.config.get('show', False):
            plt.show()
        
        return fig
    
    def plot_sector_distribution_heatmap(self) -> plt.Figure:
        """
        Create heatmap of sector distributions across models.
        
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        # Prepare data for heatmap
        models = []
        sectors = set()
        
        # Collect all sectors and models
        for model_name, dist in self.sector_distributions.items():
            if 'test_sector_distribution' in dist:
                models.append(model_name)
                sectors.update(dist['test_sector_distribution'].keys())
        
        sectors = sorted(sectors)
        
        # Create dataframes for train and test distributions
        train_data = []
        test_data = []
        
        for model_name in models:
            dist = self.sector_distributions[model_name]
            
            train_row = {'model': model_name}
            test_row = {'model': model_name}
            
            for sector in sectors:
                train_row[sector] = dist['train_sector_distribution'].get(sector, {'percentage': 0})['percentage']
                test_row[sector] = dist['test_sector_distribution'].get(sector, {'percentage': 0})['percentage']
            
            train_data.append(train_row)
            test_data.append(test_row)
        
        train_df = pd.DataFrame(train_data).set_index('model')
        test_df = pd.DataFrame(test_data).set_index('model')
        
        # Calculate difference between train and test
        diff_df = train_df - test_df
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=self.config.get('figsize', (14, 18)))
        
        # Train distribution heatmap
        sns.heatmap(train_df, annot=True, fmt='.1f', cmap='Blues', ax=axes[0], cbar_kws={'label': 'Percentage'})
        axes[0].set_title('Training Set Sector Distribution (%)', fontsize=self.config.get('title_fontsize', 14))
        axes[0].set_ylabel('Model')
        
        # Test distribution heatmap
        sns.heatmap(test_df, annot=True, fmt='.1f', cmap='Reds', ax=axes[1], cbar_kws={'label': 'Percentage'})
        axes[1].set_title('Testing Set Sector Distribution (%)', fontsize=self.config.get('title_fontsize', 14))
        axes[1].set_ylabel('Model')
        
        # Difference heatmap
        sns.heatmap(diff_df, annot=True, fmt='.1f', cmap='RdBu_r', center=0, ax=axes[2], 
                  cbar_kws={'label': 'Difference (% points)'})
        axes[2].set_title('Difference (Train - Test)', fontsize=self.config.get('title_fontsize', 14))
        axes[2].set_ylabel('Model')
        axes[2].set_xlabel('Sector')
        
        plt.tight_layout()
        
        # Save figure if requested
        if self.config.get('save', True):
            save_figure(
                fig=fig,
                filename="sector_distribution_heatmap",
                output_dir=self.config.get('output_dir'),
                dpi=self.config.get('dpi', 300),
                format=self.config.get('format', 'png')
            )
        
        # Show figure if requested
        if self.config.get('show', False):
            plt.show()
        
        return fig
    
    def plot_stratification_quality_metrics(self) -> plt.Figure:
        """
        Plot metrics for stratification quality (e.g., KL divergence).
        
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        # Collect KL divergence metrics
        model_names = []
        kl_values = []
        model_types = []
        
        for model_name, dist in self.sector_distributions.items():
            if 'kl_divergence' in dist:
                model_names.append(model_name)
                kl_values.append(dist['kl_divergence'])
                model_types.append(dist.get('model_type', 'unknown'))
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.config.get('figsize', (12, 6)))
        
        # Sort by KL divergence (lower is better)
        sort_idx = np.argsort(kl_values)
        model_names = [model_names[i] for i in sort_idx]
        kl_values = [kl_values[i] for i in sort_idx]
        model_types = [model_types[i] for i in sort_idx]
        
        # Create bar chart with color mapping by model type
        model_type_colors = {
            'xgboost': '#3498db',
            'lightgbm': '#2ecc71',
            'catboost': '#e74c3c',
            'elasticnet': '#9b59b6',
            'linear': '#f39c12',
            'sector': '#1abc9c',
            'unknown': '#7f8c8d'
        }
        
        colors = [model_type_colors.get(mt, '#7f8c8d') for mt in model_types]
        bars = ax.bar(model_names, kl_values, color=colors, alpha=0.7)
        
        # Add labels, title and axis ticks
        ax.set_xlabel('Model')
        ax.set_ylabel('KL Divergence (lower is better)')
        ax.set_title('Stratification Quality by Model', 
                    fontsize=self.config.get('title_fontsize', 14))
        plt.xticks(rotation=45, ha='right')
        
        # Add grid
        ax.grid(axis='y', alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=model_type) 
                          for model_type, color in model_type_colors.items()
                          if model_type in model_types]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Add value labels
        add_value_labels(bars, decimal_places=4)
        
        plt.tight_layout()
        
        # Save figure if requested
        if self.config.get('save', True):
            save_figure(
                fig=fig,
                filename="stratification_quality_metrics",
                output_dir=self.config.get('output_dir'),
                dpi=self.config.get('dpi', 300),
                format=self.config.get('format', 'png')
            )
        
        # Show figure if requested
        if self.config.get('show', False):
            plt.show()
        
        return fig
    
    def plot_all_models_sector_balance(self) -> plt.Figure:
        """
        Plot sector balance for all models in a single figure.
        
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        # Prepare data
        model_data = []
        
        for model_name, dist in self.sector_distributions.items():
            if 'test_sector_distribution' in dist and 'train_sector_distribution' in dist:
                # Calculate overall balance metric (average absolute difference)
                sectors = list(dist['test_sector_distribution'].keys())
                test_pcts = [dist['test_sector_distribution'][s]['percentage'] for s in sectors]
                train_pcts = [dist['train_sector_distribution'][s]['percentage'] for s in sectors]
                
                # Calculate average absolute difference
                avg_abs_diff = np.mean([abs(t - tr) for t, tr in zip(test_pcts, train_pcts)])
                
                model_data.append({
                    'model': model_name,
                    'model_type': dist.get('model_type', 'unknown'),
                    'avg_diff': avg_abs_diff,
                    'kl_divergence': dist.get('kl_divergence', np.nan)
                })
        
        if not model_data:
            print("No suitable models found for sector balance visualization")
            return None
        
        # Convert to DataFrame and sort
        df = pd.DataFrame(model_data)
        df = df.sort_values('avg_diff')
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.config.get('figsize', (12, 6)))
        
        # Create bar chart with color mapping by model type
        model_type_colors = {
            'xgboost': '#3498db',
            'lightgbm': '#2ecc71',
            'catboost': '#e74c3c',
            'elasticnet': '#9b59b6',
            'linear': '#f39c12',
            'sector': '#1abc9c',
            'unknown': '#7f8c8d'
        }
        
        colors = [model_type_colors.get(mt, '#7f8c8d') for mt in df['model_type']]
        bars = ax.bar(df['model'], df['avg_diff'], color=colors, alpha=0.7)
        
        # Add labels, title and axis ticks
        ax.set_xlabel('Model')
        ax.set_ylabel('Average Absolute Difference (%)')
        ax.set_title('Sector Balance by Model', 
                    fontsize=self.config.get('title_fontsize', 14))
        plt.xticks(rotation=45, ha='right')
        
        # Add grid
        ax.grid(axis='y', alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=model_type) 
                          for model_type, color in model_type_colors.items()
                          if model_type in df['model_type'].values]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Add value labels
        add_value_labels(bars, decimal_places=2)
        
        plt.tight_layout()
        
        # Save figure if requested
        if self.config.get('save', True):
            save_figure(
                fig=fig,
                filename="all_models_sector_balance",
                output_dir=self.config.get('output_dir'),
                dpi=self.config.get('dpi', 300),
                format=self.config.get('format', 'png')
            )
        
        # Show figure if requested
        if self.config.get('show', False):
            plt.show()
        
        return fig
    
    def plot(self) -> Dict[str, plt.Figure]:
        """
        Create all sector stratification visualizations.
        
        Returns:
            Dict[str, matplotlib.figure.Figure]: Dictionary of created figures
        """
        figures = {}
        
        # 1. Create individual distribution plots for each model
        for model_name in self.sector_distributions.keys():
            try:
                fig = self.plot_sector_distribution_comparison(model_name)
                if fig:
                    figures[f"{model_name}_sector_distribution"] = fig
            except Exception as e:
                print(f"Error creating distribution plot for {model_name}: {e}")
        
        # 2. Create heatmap of sector distributions
        try:
            figures['sector_distribution_heatmap'] = self.plot_sector_distribution_heatmap()
        except Exception as e:
            print(f"Error creating sector distribution heatmap: {e}")
        
        # 3. Create stratification quality metrics plot
        try:
            figures['stratification_quality'] = self.plot_stratification_quality_metrics()
        except Exception as e:
            print(f"Error creating stratification quality metrics plot: {e}")
        
        # 4. Create all models sector balance plot
        try:
            balance_fig = self.plot_all_models_sector_balance()
            if balance_fig:
                figures['all_models_sector_balance'] = balance_fig
        except Exception as e:
            print(f"Error creating all models sector balance plot: {e}")
        
        print(f"Created {len(figures)} sector stratification visualizations")
        return figures


# Helper functions
def plot_stratification_for_model(
    model_data: Dict[str, Any],
    config: Optional[Union[Dict[str, Any], VisualizationConfig]] = None
) -> Optional[plt.Figure]:
    """
    Plot stratification for a single model.
    
    Args:
        model_data: Model data dictionary
        config: Visualization configuration
        
    Returns:
        matplotlib.figure.Figure: The created figure or None if not available
    """
    plot = SectorStratificationPlot([model_data], config)
    model_name = model_data.get('model_name', 'unknown')
    return plot.plot_sector_distribution_comparison(model_name)


def plot_all_stratification_visualizations(
    model_data_list: Optional[List[Dict[str, Any]]] = None,
    config: Optional[Union[Dict[str, Any], VisualizationConfig]] = None
) -> Dict[str, plt.Figure]:
    """
    DEPRECATED: Create all stratification visualizations.

    This function is deprecated. Use functions from sector_weights.py instead.

    Args:
        model_data_list: List of model data dictionaries
        config: Visualization configuration

    Returns:
        Dict[str, matplotlib.figure.Figure]: Dictionary of created figures
    """
    if _DEPRECATED_MODULE:
        print("WARNING: This module is deprecated. Please use sector_weights.py instead.")
        print("No files will be generated in the sectors/stratification directory.")

        # Override config to prevent saving files
        if config is None:
            config = {}

        if isinstance(config, dict):
            config = dict(config)
            config['save'] = False
        else:
            config.config['save'] = False

    plot = SectorStratificationPlot(model_data_list, config)
    return plot.plot()


# Command-line execution
if __name__ == "__main__":
    print("WARNING: This module is deprecated. Please use sector_weights.py instead.")
    print("For sector weight distribution visualizations, run:")
    print("python -m visualization_new.plots.sector_weights")

    # Don't run the deprecated functionality
    import sys
    sys.exit(0)