"""Multicollinearity analysis for features."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Try to import seaborn, but continue if it fails
try:
    import seaborn as sns
except (ImportError, SyntaxError) as e:
    print(f"Warning: Could not import seaborn: {e}")
    sns = None

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

from src.config import settings
from src.data import load_features_data, get_base_and_yeo_features
from src.utils import io
from src.visualization.core.style import setup_visualization_style, save_figure

def calculate_vif(X):
    """
    Calculate Variance Inflation Factor for each feature in a dataframe.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Feature dataframe
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with VIF values for each feature
    """
    # Create a dataframe to store VIF values
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    
    # Calculate VIF for each feature
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    
    # Sort by VIF value (descending)
    vif_data = vif_data.sort_values("VIF", ascending=False)
    
    return vif_data

def analyze_multicollinearity():
    """
    Analyze multicollinearity in feature sets and save results.
    """
    print("Analyzing multicollinearity in feature sets...")
    
    # Load feature data
    feature_df = load_features_data()
    
    # Get base and yeo feature sets
    LR_Base, LR_Yeo, base_columns, yeo_columns = get_base_and_yeo_features(feature_df)
    
    # Create versions with random features - these are the ones we will use
    from src.data import add_random_feature
    LR_Base_random = add_random_feature(LR_Base)
    LR_Yeo_random = add_random_feature(LR_Yeo)
    
    # Explicitly list the base financial KPIs
    base_financial_kpis = [
        'shares_outstanding',
        'shares_float',
        'hist_ev_usd',
        'hist_net_debt_usd',
        'market_cap_usd',
        'net_income_usd',
        'hist_gross_profit_usd',
        'hist_book_px',
        'hist_ebitda_ev',
        'hist_fcf_yld',
        'vola',
        'top_2_shareholder_percentage',
        'beta',
        'top_3_shareholder_percentage',
        'hist_roa',
        'top_1_shareholder_percentage',
        'hist_pe',
        'hist_asset_turnover',
        'hist_roe',
        'hist_capex_sales',
        'return_usd',
        'hist_capex_depr',
        'hist_net_chg_lt_debt_usd',
        'hist_rd_exp_usd',
        'hist_eps_usd',
        'hist_roic',
        'random_feature'  # Added random feature here
    ]
    
    # Define the Yeo-Johnson transformed KPIs (same as base but with prefix)
    yeo_financial_kpis = ['yeo_joh_' + kpi for kpi in base_financial_kpis if kpi != 'random_feature']
    yeo_financial_kpis.append('random_feature')  # Add random feature separately for Yeo
    
    # Verify all KPIs exist in the datasets
    valid_base_kpis = [col for col in base_financial_kpis if col in LR_Base_random.columns]
    if len(valid_base_kpis) < len(base_financial_kpis):
        missing = set(base_financial_kpis) - set(valid_base_kpis)
        print(f"Warning: Some base financial KPIs not found in dataset: {missing}")
    
    valid_yeo_kpis = [col for col in yeo_financial_kpis if col in LR_Yeo_random.columns]
    if len(valid_yeo_kpis) < len(yeo_financial_kpis):
        missing = set(yeo_financial_kpis) - set(valid_yeo_kpis)
        print(f"Warning: Some Yeo-Johnson financial KPIs not found in dataset: {missing}")
    
    print(f"Base financial KPIs count: {len(valid_base_kpis)}")
    print(f"Yeo financial KPIs count: {len(valid_yeo_kpis)}")
    
    # Calculate VIF for Base features (financial KPIs only)
    print("Calculating VIF for Base financial KPIs...")
    base_vif = calculate_vif(LR_Base_random[valid_base_kpis])
    
    # Calculate VIF for Yeo-Johnson transformed features
    print("Calculating VIF for Yeo-Johnson transformed features...")
    yeo_vif = calculate_vif(LR_Yeo_random[valid_yeo_kpis])
    
    # Save results
    output_dir = settings.METRICS_DIR
    io.ensure_dir(output_dir)
    base_vif.to_csv(f"{output_dir}/base_vif_values.csv", index=False)
    yeo_vif.to_csv(f"{output_dir}/yeo_vif_values.csv", index=False)
    
    # Create visual comparison
    visualize_vif_comparison(base_vif, yeo_vif)
    
    print("Multicollinearity analysis complete.")
    return base_vif, yeo_vif

def visualize_vif_comparison(base_vif, yeo_vif):
    """
    Create visual comparison of VIF values between Base and Yeo-Johnson datasets.
    
    Parameters:
    -----------
    base_vif : pandas.DataFrame
        VIF values for Base features
    yeo_vif : pandas.DataFrame
        VIF values for Yeo-Johnson transformed features
    """
    # Set up style
    style = setup_visualization_style()
    # Change from features to vif directory for better organization
    output_dir = settings.VISUALIZATION_DIR / "vif"
    io.ensure_dir(output_dir)
    
    # Create separate VIF plots for Base and Yeo
    create_vif_plot(base_vif, "Base Financial KPIs", output_dir)
    create_vif_plot(yeo_vif, "Yeo-Johnson Transformed KPIs", output_dir)
    
    # Create a combined plot
    create_combined_vif_plot(base_vif, yeo_vif, output_dir)
    
    print(f"VIF visualizations saved to the dedicated VIF directory: {output_dir}")

def create_vif_plot(vif_data, title_suffix, output_dir, top_n=30):
    """
    Create a VIF bar plot for a single dataset.
    
    Parameters:
    -----------
    vif_data : pandas.DataFrame
        VIF values for features
    title_suffix : str
        Suffix for plot title
    output_dir : Path
        Directory to save the plot
    top_n : int, default=30
        Number of top features to display (by VIF value)
    """
    plt.figure(figsize=(12, 10))
    
    # Limit to top_n features with highest VIF
    plot_data = vif_data.head(top_n) if len(vif_data) > top_n else vif_data
    
    # Create horizontal bar chart
    plt.barh(plot_data["Feature"], plot_data["VIF"], color='#3498db')
    
    # Add VIF = 10 threshold line
    plt.axvline(x=10, color='gray', linestyle='--', label='VIF = 10 Threshold')
    
    # Set labels and title
    plt.xlabel('Variance Inflation Factor (VIF)')
    plt.ylabel('Feature')
    plt.title(f'Top {len(plot_data)} of {len(vif_data)} VIF Values: {title_suffix} + Random')
    
    # Add legend
    plt.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    save_figure(plt.gcf(), f"vif_values_{title_suffix.replace(' ', '_').lower()}", output_dir)
    
    plt.close()

def create_combined_vif_plot(base_vif, yeo_vif, output_dir, top_n=30):
    """
    Create a combined plot comparing VIF values between Base and Yeo datasets.
    
    Parameters:
    -----------
    base_vif : pandas.DataFrame
        VIF values for Base features
    yeo_vif : pandas.DataFrame
        VIF values for Yeo-Johnson transformed features
    output_dir : Path
        Directory to save the plot
    top_n : int, default=30
        Number of top features to display (by Base VIF value)
    """
    # Create a combined DataFrame for plotting
    # First, rename columns to indicate dataset
    base_vif_copy = base_vif.copy()
    base_vif_copy.columns = ["Feature", "Base_VIF"]
    
    yeo_vif_copy = yeo_vif.copy()
    yeo_vif_copy.columns = ["Feature", "Yeo_VIF"]
    
    # For Yeo features, remove the 'yeo_joh_' prefix for better comparison
    yeo_vif_copy["Feature"] = yeo_vif_copy["Feature"].apply(
        lambda x: x.replace('yeo_joh_', '') if isinstance(x, str) and x.startswith('yeo_joh_') else x
    )
    
    # Merge the two dataframes on Feature
    combined_vif = pd.merge(base_vif_copy, yeo_vif_copy, on="Feature", how="outer")
    
    # Sort by Base VIF value
    combined_vif = combined_vif.sort_values("Base_VIF", ascending=False)
    
    # Limit to top_n features with highest Base VIF
    plot_data = combined_vif.head(top_n) if len(combined_vif) > top_n else combined_vif
    
    # Set up figure
    plt.figure(figsize=(14, 12))
    
    # Define width of bars
    bar_width = 0.4
    
    # Set up positions for bars
    y_pos = np.arange(len(plot_data))
    
    # Create bars
    plt.barh(y_pos - bar_width/2, plot_data["Base_VIF"], bar_width, 
             color='#3498db', label='Base Features + Random')
    plt.barh(y_pos + bar_width/2, plot_data["Yeo_VIF"], bar_width, 
             color='#2ecc71', label='Yeo-Johnson Transformed + Random')
    
    # Add VIF = 10 threshold line
    plt.axvline(x=10, color='gray', linestyle='--', label='VIF = 10 Threshold')
    
    # Set y-ticks at bar positions with feature names
    plt.yticks(y_pos, plot_data["Feature"])
    
    # Set labels and title
    plt.xlabel('Variance Inflation Factor (VIF)')
    plt.ylabel('Feature')
    plt.title(f'Top {len(plot_data)} Features: Base vs. Yeo-Johnson VIF Comparison (With Random Feature)')
    
    # Add legend
    plt.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    save_figure(plt.gcf(), "vif_comparison_base_vs_yeo", output_dir)
    
    plt.close()
    
    # Create an alternative stacked bar visualization for better comparison
    create_stacked_vif_plot(plot_data, output_dir)

def create_stacked_vif_plot(combined_vif, output_dir):
    """
    Create a stacked plot comparing VIF values between Base and Yeo datasets.
    
    Parameters:
    -----------
    combined_vif : pandas.DataFrame
        Combined VIF values for both datasets (already limited to top features)
    output_dir : Path
        Directory to save the plot
    """
    # Melt the dataframe for easier plotting
    melted_vif = pd.melt(combined_vif, id_vars=["Feature"], 
                         value_vars=["Base_VIF", "Yeo_VIF"],
                         var_name="Dataset", value_name="VIF")
    
    # Replace dataset names for better labeling
    melted_vif["Dataset"] = melted_vif["Dataset"].replace({
        "Base_VIF": "Base Features + Random",
        "Yeo_VIF": "Yeo-Johnson Transformed + Random"
    })
    
    # Set up figure
    plt.figure(figsize=(16, 14))
    
    # Create stacked bar chart using seaborn if available, otherwise use matplotlib
    if sns is not None:
        sns.barplot(x="VIF", y="Feature", hue="Dataset", data=melted_vif, 
                    palette=["#3498db", "#2ecc71"])
    else:
        # Fallback to matplotlib if seaborn is not available
        datasets = melted_vif["Dataset"].unique()
        colors = {"Base Features + Random": "#3498db", "Yeo-Johnson Transformed + Random": "#2ecc71"}
        
        for i, dataset in enumerate(datasets):
            data = melted_vif[melted_vif["Dataset"] == dataset]
            plt.barh(data["Feature"], data["VIF"], 
                    color=colors.get(dataset, "#999999"), 
                    label=dataset, alpha=0.8)
    
    # Add VIF = 10 threshold line
    plt.axvline(x=10, color='gray', linestyle='--', label='VIF = 10 Threshold')
    
    # Set labels and title
    plt.xlabel('Variance Inflation Factor (VIF)')
    plt.ylabel('Feature')
    plt.title(f'Top {len(combined_vif)} Features: VIF Comparison (With Random Feature)')
    
    # Adjust legend
    plt.legend(title="Dataset")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    save_figure(plt.gcf(), "vif_stacked_comparison", output_dir)
    
    plt.close()

if __name__ == "__main__":
    # Run multicollinearity analysis
    analyze_multicollinearity()