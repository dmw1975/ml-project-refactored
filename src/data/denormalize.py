"""Temporary denormalization utilities to fix pre-normalized data issue."""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, Optional

# Estimated statistics for common financial features
# These are rough estimates based on typical financial data ranges
FEATURE_STATS = {
    # Market metrics
    'market_cap_usd': {'mean': 10000, 'std': 50000},  # in millions
    'hist_ev_usd': {'mean': 12000, 'std': 60000},     # enterprise value
    'net_income_usd': {'mean': 500, 'std': 2500},     # net income
    'hist_gross_profit_usd': {'mean': 2000, 'std': 10000},
    
    # Ratios and multiples
    'hist_pe': {'mean': 18, 'std': 15},               # P/E ratio
    'hist_book_px': {'mean': 2.5, 'std': 3},          # Price to book
    'hist_fcf_yld': {'mean': 0.05, 'std': 0.08},      # FCF yield
    'hist_ebitda_ev': {'mean': 0.08, 'std': 0.06},    # EBITDA/EV
    
    # Returns
    'hist_roe': {'mean': 0.12, 'std': 0.15},          # ROE
    'hist_roic': {'mean': 0.10, 'std': 0.12},         # ROIC
    'hist_roa': {'mean': 0.06, 'std': 0.08},          # ROA
    
    # Other metrics
    'hist_asset_turnover': {'mean': 0.8, 'std': 0.6},
    'hist_capex_sales': {'mean': 0.05, 'std': 0.08},
    'hist_capex_depr': {'mean': 1.2, 'std': 0.8},
    'hist_rd_exp_usd': {'mean': 200, 'std': 1000},
    'hist_eps_usd': {'mean': 3, 'std': 10},
    'return_usd': {'mean': 0.10, 'std': 0.25},
    'vol_total_usd': {'mean': 0.25, 'std': 0.15},
    
    # Governance
    'top_1_shareholder_percentage': {'mean': 0.15, 'std': 0.12},
    'top_2_shareholder_percentage': {'mean': 0.08, 'std': 0.06},
    'top_3_shareholder_percentage': {'mean': 0.05, 'std': 0.04},
    
    # Additional financial metrics
    'hist_net_debt_usd': {'mean': 5000, 'std': 25000},
    'hist_net_chg_lt_debt_usd': {'mean': 100, 'std': 2000},
    'beta': {'mean': 1.0, 'std': 0.5},
}


def denormalize_dataframe(df: pd.DataFrame, 
                         stats: Optional[Dict[str, Dict[str, float]]] = None,
                         exclude_patterns: Optional[list] = None) -> pd.DataFrame:
    """
    Denormalize a dataframe that has been standardized.
    
    Parameters
    ----------
    df : pd.DataFrame
        Normalized dataframe
    stats : dict, optional
        Dictionary of feature statistics. If None, uses FEATURE_STATS
    exclude_patterns : list, optional
        List of column patterns to exclude from denormalization
        
    Returns
    -------
    pd.DataFrame
        Denormalized dataframe
    """
    if stats is None:
        stats = FEATURE_STATS
        
    if exclude_patterns is None:
        exclude_patterns = ['gics_', 'cntry_', 'issuer_', 'top_', '_location', 'yeo_joh_']
    
    df_denorm = df.copy()
    
    # Track what we denormalize
    denormalized_cols = []
    
    for col in df.columns:
        # Skip categorical and already transformed columns
        if any(pattern in col for pattern in exclude_patterns):
            continue
            
        # Check if this looks like a normalized column
        if col in df_denorm.columns:
            col_mean = df_denorm[col].mean()
            col_std = df_denorm[col].std()
            
            # If it looks normalized (mean ~0, std ~1)
            if abs(col_mean) < 0.1 and 0.8 < col_std < 1.2:
                if col in stats:
                    # Denormalize using known stats
                    original_mean = stats[col]['mean']
                    original_std = stats[col]['std']
                    df_denorm[col] = df_denorm[col] * original_std + original_mean
                    denormalized_cols.append(col)
                else:
                    # Try to infer from similar columns
                    for known_col, known_stats in stats.items():
                        if known_col in col or col in known_col:
                            original_mean = known_stats['mean']
                            original_std = known_stats['std']
                            df_denorm[col] = df_denorm[col] * original_std + original_mean
                            denormalized_cols.append(col)
                            break
    
    print(f"Denormalized {len(denormalized_cols)} columns")
    if denormalized_cols:
        print(f"Examples: {denormalized_cols[:5]}")
    
    return df_denorm


def check_if_normalized(df: pd.DataFrame, threshold: float = 0.1) -> bool:
    """
    Check if a dataframe appears to be normalized.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to check
    threshold : float
        Threshold for mean to be considered "near zero"
        
    Returns
    -------
    bool
        True if dataframe appears normalized
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Exclude categorical columns
    numeric_cols = [col for col in numeric_cols 
                   if not any(p in col for p in ['gics_', 'cntry_', 'issuer_', 'top_', '_location'])]
    
    if len(numeric_cols) == 0:
        return False
    
    # Check a sample of columns
    sample_cols = numeric_cols[:10] if len(numeric_cols) > 10 else numeric_cols
    
    normalized_count = 0
    for col in sample_cols:
        col_mean = df[col].mean()
        col_std = df[col].std()
        
        if abs(col_mean) < threshold and 0.8 < col_std < 1.2:
            normalized_count += 1
    
    # If more than half appear normalized
    return normalized_count > len(sample_cols) / 2


def save_denormalization_stats(stats: Dict[str, Dict[str, float]], 
                              filepath: str = "data/metadata/denormalization_stats.json"):
    """Save denormalization statistics to JSON file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved denormalization stats to {filepath}")


def load_denormalization_stats(filepath: str = "data/metadata/denormalization_stats.json") -> Dict:
    """Load denormalization statistics from JSON file."""
    if Path(filepath).exists():
        with open(filepath, 'r') as f:
            return json.load(f)
    else:
        print(f"No custom stats found at {filepath}, using defaults")
        return FEATURE_STATS


# Temporary fix for data loading
def load_and_denormalize_features(model_type='linear'):
    """Load features and denormalize if needed."""
    from src.data.data import load_features_data
    
    df = load_features_data(model_type=model_type)
    
    if check_if_normalized(df):
        print("Data appears to be normalized - applying denormalization")
        df = denormalize_dataframe(df)
    
    return df