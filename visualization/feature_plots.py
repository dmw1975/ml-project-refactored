"""Visualization functions for feature importance and analysis."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

from config import settings
from visualization.style import setup_visualization_style, save_figure
from utils import io

def plot_top_features(importance_results=None, top_n=20):
    """
    Plot top features by importance across models.
    
    Parameters:
    -----------
    importance_results : dict, optional
        Dictionary of feature importance DataFrames. If None, it will be loaded.
    top_n : int, default=20
        Number of top features to display.
    """
    # Set up style
    style = setup_visualization_style()
    
    # Load importance results if not provided
    if importance_results is None:
        try:
            importance_results = io.load_model("feature_importance.pkl", 
                                             settings.FEATURE_IMPORTANCE_DIR)
        except:
            print("No feature importance data found. Please run feature importance analysis first.")
            return
    
    # Set up output directory
    output_dir = settings.VISUALIZATION_DIR / "features"
    io.ensure_dir(output_dir)
    
    # Create consolidated importance table
    all_features = set()
    for model_name, importance_df in importance_results.items():
        all_features.update(importance_df['Feature'])
    
    # Create DataFrame
    consolidated = pd.DataFrame(index=list(all_features))
    
    # Add importance values for each model
    for model_name, importance_df in importance_results.items():
        # Convert to dictionary for easier lookup
        importance_dict = dict(zip(importance_df['Feature'], importance_df['Importance']))
        
        # Add to consolidated DataFrame
        consolidated[model_name] = consolidated.index.map(lambda x: importance_dict.get(x, 0))
    
    # Add average importance
    consolidated['avg_importance'] = consolidated.mean(axis=1)
    
    # Sort by average importance
    consolidated = consolidated.sort_values('avg_importance', ascending=False)
    
    # 1. Plot top features by average importance
    fig, ax = plt.subplots(figsize=(12, 10))
    
    top_df = consolidated.head(top_n)
    
    # Plot horizontal bar chart
    bars = ax.barh(top_df.index[::-1], top_df['avg_importance'][::-1], 
                   color='#3498db', alpha=0.7)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                f'{width:.4f}', va='center')
    
    ax.set_xlabel('Average Importance')
    ax.set_title(f'Top {top_n} Features by Average Importance Across Models', fontsize=14)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, f"top_{top_n}_features_avg_importance", output_dir)
    
    # 2. Create heatmap of top features across models
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Get top 30 features for heatmap
    heatmap_df = consolidated.head(30).drop('avg_importance', axis=1)
    
    # Create heatmap
    sns.heatmap(heatmap_df, cmap=style.get('blue_cmap', 'Blues'), annot=True,
               fmt='.3f', linewidths=0.5, ax=ax)
    
    plt.title('Top 30 Features Importance Across Models', fontsize=14)
    plt.ylabel('Features')
    plt.xlabel('Models')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    save_figure(fig, "top_features_heatmap", output_dir)
    
    # 3. Plot feature importance distributions for top features
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Select top 10 features
    top10_df = consolidated.head(10)
    
    # Calculate statistics for each feature
    stats_df = pd.DataFrame({
        'feature': top10_df.index,
        'mean': top10_df.drop('avg_importance', axis=1).mean(axis=1),
        'std': top10_df.drop('avg_importance', axis=1).std(axis=1),
        'min': top10_df.drop('avg_importance', axis=1).min(axis=1),
        'max': top10_df.drop('avg_importance', axis=1).max(axis=1)
    })
    
    # Plot error bars
    ax.errorbar(
        x=stats_df['feature'],
        y=stats_df['mean'],
        yerr=[
            stats_df['mean'] - stats_df['min'],  # lower error
            stats_df['max'] - stats_df['mean']   # upper error
        ],
        fmt='o',
        capsize=5,
        ecolor='gray',
        markersize=8,
        color='#3498db',
        alpha=0.7
    )
    
    ax.set_title('Feature Importance Distribution (Top 10 Features)', fontsize=14)
    ax.set_ylabel('Mean Importance')
    ax.set_xlabel('Features')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    save_figure(fig, "top_features_distribution", output_dir)
    
    # 4. Plot random feature performance if available
    random_csv = settings.FEATURE_IMPORTANCE_DIR / "random_feature_stats.csv"
    if random_csv.exists():
        random_df = pd.read_csv(random_csv)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(random_df['model_name'], random_df['random_rank'], 
                     color='#e74c3c', alpha=0.7)
        
        # Add value labels
        for bar, total in zip(bars, random_df['total_features']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                    f'{height}/{total}', ha='center', va='bottom')
        
        ax.set_title('Random Feature Rank Across Models (Higher Rank = Less Important)', 
                    fontsize=14)
        ax.set_ylabel('Rank')
        ax.set_xlabel('Model')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        save_figure(fig, "random_feature_rank", output_dir)
    
    print(f"Feature importance visualizations saved to {output_dir}")
    return consolidated

def plot_feature_importance_by_model(importance_results=None, top_n=15):
    """
    Create individual feature importance plots for each model.
    
    Parameters:
    -----------
    importance_results : dict, optional
        Dictionary of feature importance DataFrames. If None, it will be loaded.
    top_n : int, default=15
        Number of top features to display per model.
    """
    # Set up style
    style = setup_visualization_style()
    
    # Load importance results if not provided
    if importance_results is None:
        try:
            importance_results = io.load_model("feature_importance.pkl", 
                                             settings.FEATURE_IMPORTANCE_DIR)
        except:
            print("No feature importance data found. Please run feature importance analysis first.")
            return
    
    # Set up output directory
    output_dir = settings.VISUALIZATION_DIR / "features"
    io.ensure_dir(output_dir)
    
    # Create individual plots for each model
    for model_name, importance_df in importance_results.items():
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Sort and get top features
        sorted_df = importance_df.sort_values('Importance', ascending=False).head(top_n)
        
        # Create horizontal bar chart
        bars = ax.barh(sorted_df['Feature'][::-1], sorted_df['Importance'][::-1],
                       xerr=sorted_df['Std'][::-1], color='#3498db', alpha=0.7, 
                       capsize=5, ecolor='gray')
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            label_x = max(width + 0.01, 0.01)  # Handle negative importances
            ax.text(label_x, bar.get_y() + bar.get_height()/2,
                    f'{width:.4f}', va='center')
        
        ax.set_xlabel('Importance')
        ax.set_title(f'Top {top_n} Features for {model_name}', fontsize=14)
        ax.grid(axis='x', alpha=0.3)
        
        # Add vertical line at x=0
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        save_figure(fig, f"{model_name}_top_features", output_dir)
    
    print(f"Individual model feature importance plots saved to {output_dir}")
    return importance_results

def plot_feature_correlations(top_features=None, n_features=20):
    """
    Plot correlation matrix for top features.
    
    Parameters:
    -----------
    top_features : list, optional
        List of top feature names. If None, they will be loaded from consolidated importance.
    n_features : int, default=20
        Number of top features to include in correlation analysis.
    """
    # Set up style
    style = setup_visualization_style()
    
    # Load top features if not provided
    if top_features is None:
        consolidated_file = settings.FEATURE_IMPORTANCE_DIR / "consolidated_importance.csv"
        if consolidated_file.exists():
            consolidated = pd.read_csv(consolidated_file, index_col=0)
            top_features = consolidated.index[:n_features].tolist()
        else:
            # Try to get them from feature importance analysis
            try:
                from evaluation.importance import create_consolidated_importance_table, analyze_feature_importance
                importance_results = io.load_model("feature_importance.pkl", 
                                                 settings.FEATURE_IMPORTANCE_DIR)
                consolidated = create_consolidated_importance_table(importance_results)
                top_features = consolidated.index[:n_features].tolist()
            except:
                print("No feature importance data found. Please run feature importance analysis first.")
                return
    
    # Set up output directory
    output_dir = settings.VISUALIZATION_DIR / "features"
    io.ensure_dir(output_dir)
    
    # Load original data
    try:
        from data import load_features_data
        feature_df = load_features_data()
    except:
        print("Could not load feature data.")
        return
    
    # Filter for top features
    # Make sure all features exist in the dataframe
    valid_features = [f for f in top_features if f in feature_df.columns]
    if len(valid_features) < len(top_features):
        print(f"Warning: {len(top_features) - len(valid_features)} features not found in data.")
    
    # Calculate correlation matrix
    corr_matrix = feature_df[valid_features].corr()
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 12))
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
               square=True, linewidths=.5, annot=True, fmt='.2f')
    
    ax.set_title(f'Correlation Matrix of Top {len(valid_features)} Features', fontsize=15)
    
    plt.tight_layout()
    save_figure(fig, "top_features_correlation", output_dir)
    
    # Cluster features by correlation and visualize
    try:
        from scipy.cluster import hierarchy
        from scipy.spatial.distance import squareform
        
        # Convert correlation to distance
        corr_condensed = squareform(1 - abs(corr_matrix))
        
        # Perform hierarchical clustering
        z = hierarchy.linkage(corr_condensed, method='average')
        
        # Create dendrogram
        fig, ax = plt.subplots(figsize=(14, 10))
        
        hierarchy.dendrogram(
            z,
            labels=corr_matrix.index,
            orientation='right',
            leaf_font_size=10
        )
        
        ax.set_title('Feature Clustering by Correlation', fontsize=15)
        plt.xlabel('Distance (1 - |Correlation|)')
        
        plt.tight_layout()
        save_figure(fig, "feature_correlation_clustering", output_dir)
        
    except:
        print("Could not create feature clustering visualization.")
    
    print(f"Feature correlation visualizations saved to {output_dir}")
    return corr_matrix

if __name__ == "__main__":
    # Run all visualizations
    top_features = plot_top_features()
    plot_feature_importance_by_model()
    if top_features is not None:
        plot_feature_correlations(top_features.index[:20].tolist())