"""Data validation utilities for ML pipeline."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def validate_tree_models_data(df: pd.DataFrame, target: pd.Series) -> Tuple[bool, List[str]]:
    """
    Validate tree models dataset structure and content.
    
    Parameters
    ----------
    df : pd.DataFrame
        Features dataframe
    target : pd.Series
        Target series
        
    Returns
    -------
    Tuple[bool, List[str]]
        (is_valid, list_of_issues)
    """
    issues = []
    
    # Expected feature counts
    expected_numerical = 52  # 26 raw + 26 Yeo
    expected_categorical = 7
    expected_total = expected_numerical + expected_categorical
    
    # Check shape
    if df.shape[1] != expected_total:
        issues.append(f"Expected {expected_total} features, got {df.shape[1]}")
    
    # Check for required categorical features
    required_categorical = [
        'gics_sector', 'gics_sub_ind', 'issuer_cntry_domicile_name',
        'cntry_of_risk', 'top_1_shareholder_location',
        'top_2_shareholder_location', 'top_3_shareholder_location'
    ]
    
    missing_categorical = [col for col in required_categorical if col not in df.columns]
    if missing_categorical:
        issues.append(f"Missing categorical features: {missing_categorical}")
    
    # Check for Yeo features
    yeo_features = [col for col in df.columns if col.startswith('yeo_joh_')]
    if len(yeo_features) != 26:
        issues.append(f"Expected 26 Yeo features, found {len(yeo_features)}")
    
    # Check data types
    categorical_dtypes = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_dtypes) != expected_categorical:
        issues.append(f"Expected {expected_categorical} categorical columns, found {len(categorical_dtypes)}")
    
    # Check for target alignment
    if len(df) != len(target):
        issues.append(f"Feature rows ({len(df)}) != target rows ({len(target)})")
    
    # Check index alignment
    if not df.index.equals(target.index):
        issues.append("Feature and target indices do not match")
    
    # Check for NaN in target
    if target.isna().any():
        issues.append(f"Found {target.isna().sum()} NaN values in target")
    
    is_valid = len(issues) == 0
    
    if is_valid:
        logger.info("Tree models data validation passed")
    else:
        logger.warning(f"Tree models data validation failed with {len(issues)} issues")
        for issue in issues:
            logger.warning(f"  - {issue}")
    
    return is_valid, issues


def validate_linear_models_data(df: pd.DataFrame, target: pd.Series) -> Tuple[bool, List[str]]:
    """
    Validate linear models dataset structure and content.
    
    Parameters
    ----------
    df : pd.DataFrame
        Features dataframe with one-hot encoded features
    target : pd.Series
        Target series
        
    Returns
    -------
    Tuple[bool, List[str]]
        (is_valid, list_of_issues)
    """
    issues = []
    
    # Check minimum expected columns (52 numerical + one-hot encoded)
    min_expected_cols = 52 + 50  # At least 50 one-hot columns
    if df.shape[1] < min_expected_cols:
        issues.append(f"Expected at least {min_expected_cols} features, got {df.shape[1]}")
    
    # Check for one-hot pattern
    one_hot_patterns = [
        'gics_sector_', 'gics_sub_ind_', 'issuer_cntry_domicile_name_',
        'cntry_of_risk_', 'top_1_shareholder_location_',
        'top_2_shareholder_location_', 'top_3_shareholder_location_'
    ]
    
    for pattern in one_hot_patterns:
        pattern_cols = [col for col in df.columns if col.startswith(pattern)]
        if len(pattern_cols) == 0:
            issues.append(f"No one-hot columns found for pattern: {pattern}")
    
    # Check all values are numeric
    non_numeric = df.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric) > 0:
        issues.append(f"Found non-numeric columns: {list(non_numeric)[:5]}...")
    
    # Check one-hot encoding validity (0 or 1 values)
    for pattern in one_hot_patterns:
        pattern_cols = [col for col in df.columns if col.startswith(pattern)]
        if pattern_cols:
            values = df[pattern_cols].values.flatten()
            unique_vals = np.unique(values[~np.isnan(values)])
            if not np.array_equal(unique_vals, [0., 1.]) and not np.array_equal(unique_vals, [0.]) and not np.array_equal(unique_vals, [1.]):
                issues.append(f"One-hot columns for {pattern} contain values other than 0 and 1")
    
    # Check for target alignment
    if len(df) != len(target):
        issues.append(f"Feature rows ({len(df)}) != target rows ({len(target)})")
    
    # Check index alignment
    if not df.index.equals(target.index):
        issues.append("Feature and target indices do not match")
    
    is_valid = len(issues) == 0
    
    if is_valid:
        logger.info("Linear models data validation passed")
    else:
        logger.warning(f"Linear models data validation failed with {len(issues)} issues")
        for issue in issues:
            logger.warning(f"  - {issue}")
    
    return is_valid, issues


def validate_data_files_exist() -> Tuple[bool, List[str]]:
    """
    Validate that all required data files exist.
    
    Returns
    -------
    Tuple[bool, List[str]]
        (all_exist, list_of_missing_files)
    """
    from src.config.settings import PROCESSED_DATA_DIR, RAW_DATA_DIR
    
    required_files = {
        'processed': [
            PROCESSED_DATA_DIR / 'tree_models_dataset.csv',
            PROCESSED_DATA_DIR / 'linear_models_dataset.csv',
            PROCESSED_DATA_DIR / 'categorical_mappings.pkl',
            PROCESSED_DATA_DIR / 'datasets_metadata.json'
        ],
        'raw': [
            RAW_DATA_DIR / 'score.csv',
            RAW_DATA_DIR / 'combined_df_for_ml_models.csv',
            RAW_DATA_DIR / 'combined_df_for_tree_models.csv'
        ],
        'pkl': [
            Path('data/pkl/base_columns.pkl'),
            Path('data/pkl/yeo_columns.pkl')
        ]
    }
    
    missing_files = []
    
    for category, files in required_files.items():
        for file_path in files:
            if not file_path.exists():
                missing_files.append(str(file_path))
                logger.warning(f"Missing required file: {file_path}")
    
    all_exist = len(missing_files) == 0
    
    if all_exist:
        logger.info("All required data files exist")
    else:
        logger.error(f"Missing {len(missing_files)} required files")
    
    return all_exist, missing_files


def validate_score_coverage(features_df: pd.DataFrame, scores_path: Path) -> Tuple[bool, List[str]]:
    """
    Validate that all companies in features have corresponding scores.
    
    Parameters
    ----------
    features_df : pd.DataFrame
        Features dataframe
    scores_path : Path
        Path to scores CSV file
        
    Returns
    -------
    Tuple[bool, List[str]]
        (is_valid, list_of_issues)
    """
    issues = []
    
    try:
        scores_df = pd.read_csv(scores_path)
        scores_df = scores_df.set_index('issuer_name')
        
        # Check for companies without scores
        feature_companies = set(features_df.index)
        score_companies = set(scores_df.index)
        
        missing_scores = feature_companies - score_companies
        if missing_scores:
            issues.append(f"{len(missing_scores)} companies in features lack scores: {list(missing_scores)[:5]}...")
        
        # Check for duplicate scores
        if scores_df.index.duplicated().any():
            duplicates = scores_df.index[scores_df.index.duplicated()].unique()
            issues.append(f"Duplicate companies in scores: {list(duplicates)[:5]}...")
        
    except Exception as e:
        issues.append(f"Error reading scores file: {str(e)}")
    
    is_valid = len(issues) == 0
    
    if is_valid:
        logger.info("Score coverage validation passed")
    else:
        logger.warning(f"Score coverage validation failed with {len(issues)} issues")
    
    return is_valid, issues


def run_full_validation() -> Dict[str, bool]:
    """
    Run complete data validation suite.
    
    Returns
    -------
    Dict[str, bool]
        Validation results for each check
    """
    logger.info("Running full data validation suite...")
    
    results = {}
    
    # Check file existence first
    files_exist, missing = validate_data_files_exist()
    results['files_exist'] = files_exist
    
    if not files_exist:
        logger.error("Cannot proceed with validation - required files missing")
        return results
    
    # Load and validate tree models data
    try:
        from src.data.data_categorical import load_tree_models_data
        features, target = load_tree_models_data()
        is_valid, issues = validate_tree_models_data(features, target)
        results['tree_models_valid'] = is_valid
    except Exception as e:
        logger.error(f"Error validating tree models data: {str(e)}")
        results['tree_models_valid'] = False
    
    # Load and validate linear models data
    try:
        from src.data.data_categorical import load_linear_models_data
        features, target = load_linear_models_data()
        is_valid, issues = validate_linear_models_data(features, target)
        results['linear_models_valid'] = is_valid
    except Exception as e:
        logger.error(f"Error validating linear models data: {str(e)}")
        results['linear_models_valid'] = False
    
    # Validate score coverage
    try:
        from src.config.settings import RAW_DATA_DIR
        scores_path = RAW_DATA_DIR / 'score.csv'
        is_valid, issues = validate_score_coverage(features, scores_path)
        results['score_coverage_valid'] = is_valid
    except Exception as e:
        logger.error(f"Error validating score coverage: {str(e)}")
        results['score_coverage_valid'] = False
    
    # Summary
    all_valid = all(results.values())
    if all_valid:
        logger.info("✓ All data validation checks passed")
    else:
        failed_checks = [k for k, v in results.items() if not v]
        logger.error(f"✗ Data validation failed: {failed_checks}")
    
    return results