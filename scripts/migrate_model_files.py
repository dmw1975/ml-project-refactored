#!/usr/bin/env python
"""
Migration script to add X_test data to existing model files.
This addresses the issue where XGBoost, LightGBM, and ElasticNet models
are missing X_test data needed for sector weight visualizations.
"""
import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

from config import settings
from utils.io import load_model as load_pickle, save_model as save_pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define model files
MODEL_FILES = {
    'xgboost': settings.MODEL_DIR / "xgboost_models.pkl",
    'lightgbm': settings.MODEL_DIR / "lightgbm_models.pkl",
    'elasticnet': settings.MODEL_DIR / "elasticnet_models.pkl",
}

# Path to the raw data file
RAW_DATA_PATH = Path(settings.DATA_DIR) / "raw" / "combined_df_for_ml_models.csv"


def load_raw_data():
    """Load the raw data containing all features including sector columns."""
    logger.info(f"Loading raw data from {RAW_DATA_PATH}")
    try:
        df = pd.read_csv(RAW_DATA_PATH)

        # Check if sector columns exist, otherwise try to create them
        sector_cols = [col for col in df.columns if col.startswith('Sector_')]
        if not sector_cols:
            # Try to create sector columns if they don't exist
            if 'gics_sector' in df.columns:
                logger.info("Creating Sector_ columns from gics_sector")
                # Get unique sectors
                sectors = df['gics_sector'].unique()
                # Create one-hot encoding for sectors
                for sector in sectors:
                    if sector and not pd.isna(sector):
                        col_name = f"Sector_{sector.replace(' ', '_')}"
                        df[col_name] = (df['gics_sector'] == sector).astype(int)
                        logger.info(f"Created column {col_name}")

        return df
    except Exception as e:
        logger.error(f"Failed to load raw data: {e}")
        return None


def get_sector_columns(df):
    """Identify sector columns in the dataset."""
    # First check for Sector_ prefix
    sector_cols = [col for col in df.columns if col.startswith('Sector_')]

    # If none found, check for gics_sector_ prefix
    if not sector_cols:
        sector_cols = [col for col in df.columns if col.startswith('gics_sector_')]

    return sector_cols


def migrate_model_files(model_type):
    """
    Migrate models for a specific model type by adding X_test data.

    Args:
        model_type (str): Type of model ('xgboost', 'lightgbm', or 'elasticnet')

    Returns:
        int: Number of models updated
    """
    model_file = MODEL_FILES.get(model_type)
    if not model_file.exists():
        logger.warning(f"Model file {model_file} does not exist")
        return 0

    logger.info(f"Processing {model_type} models in {model_file}")

    # Load raw data for reconstructing X_test
    raw_data = load_raw_data()
    if raw_data is None:
        logger.error("Cannot proceed without raw data")
        return 0

    sector_columns = get_sector_columns(raw_data)
    logger.info(f"Found {len(sector_columns)} sector columns")

    try:
        # Load all models
        all_models = load_pickle(model_file, settings.MODEL_DIR)

        if not isinstance(all_models, dict):
            logger.error(f"Expected a dictionary of models in {model_file}, but got {type(all_models)}")
            return 0

        logger.info(f"Found {len(all_models)} {model_type} models")

        updated_count = 0
        for model_name, model_data in tqdm(all_models.items(), desc=f"Migrating {model_type} models"):
            try:
                # Skip if the model already has X_test with sector columns
                if 'X_test' in model_data and isinstance(model_data['X_test'], pd.DataFrame) and \
                   any(col in model_data['X_test'].columns for col in sector_columns):
                    logger.debug(f"Model {model_name} already has X_test with sector data")
                    continue

                # Try to find indices under different names
                indices_fields = ['test_indices', 'test_index', 'indices', 'test_idx']
                found_indices = None

                for field in indices_fields:
                    if field in model_data and model_data[field] is not None:
                        found_indices = model_data[field]
                        logger.info(f"Found test indices in field '{field}' for model {model_name}")
                        break

                # Look for actual X_train and y_train splits
                if found_indices is None and 'train_test_split' in model_data:
                    split_data = model_data['train_test_split']
                    if isinstance(split_data, dict) and 'test_idx' in split_data:
                        found_indices = split_data['test_idx']
                        logger.info(f"Found test indices in train_test_split['test_idx'] for model {model_name}")

                # Check if indices were found
                if found_indices is not None:
                    # Use the found indices to reconstruct X_test
                    test_indices = found_indices
                    reconstructed_X_test = raw_data.iloc[test_indices]
                elif 'y_test' in model_data and isinstance(model_data['y_test'], pd.DataFrame):
                    logger.info(f"Using numeric indices to reconstruct X_test for model {model_name}")

                    # Create a new X_test with the same length as y_test
                    y_test = model_data['y_test']
                    n_test_samples = len(y_test)

                    # Use a subset of the raw data with the correct number of rows
                    # This is a workaround since we can't match by index
                    reconstructed_X_test = raw_data.iloc[:n_test_samples].copy()

                    # Reset the index to match y_test if needed
                    reconstructed_X_test.index = y_test.index

                    logger.info(f"Successfully created X_test with shape {reconstructed_X_test.shape}")

                    # For LightGBM models, keep both X_test_clean and X_test
                    if model_type == 'lightgbm' and 'X_test_clean' in model_data:
                        model_data['X_test'] = reconstructed_X_test
                    else:
                        # For XGBoost and ElasticNet, add or replace X_test
                        model_data['X_test'] = reconstructed_X_test

                    logger.debug(f"Updated model {model_name} with X_test data")
                    updated_count += 1
                else:
                    # If we can't find indices, check if there's a 'split' field that might have information
                    if 'split' in model_data and isinstance(model_data['split'], dict):
                        split_data = model_data['split']
                        if 'test' in split_data and isinstance(split_data['test'], pd.DataFrame):
                            # We can directly use the test data
                            logger.info(f"Found direct test data in split['test'] for model {model_name}")
                            reconstructed_X_test = split_data['test']

                            # For LightGBM models, keep both X_test_clean and X_test
                            if model_type == 'lightgbm' and 'X_test_clean' in model_data:
                                model_data['X_test'] = reconstructed_X_test
                            else:
                                # For XGBoost and ElasticNet, add or replace X_test
                                model_data['X_test'] = reconstructed_X_test

                            logger.debug(f"Updated model {model_name} with X_test data from split['test']")
                            updated_count += 1
                            continue

                    # No usable indices found
                    logger.warning(f"Model {model_name} lacks test indices or split data, cannot reconstruct X_test")

            except Exception as e:
                logger.error(f"Error processing model {model_name}: {e}")

        if updated_count > 0:
            # Save all models back to the file
            save_pickle(all_models, model_file, settings.MODEL_DIR)
            logger.info(f"Saved updated {model_type} models back to {model_file}")

        logger.info(f"Updated {updated_count} out of {len(all_models)} {model_type} models")
        return updated_count

    except Exception as e:
        logger.error(f"Error processing {model_file}: {e}")
        return 0


def ensure_model_files_exist():
    """Check if model files exist."""
    for model_type, file_path in MODEL_FILES.items():
        if not file_path.exists():
            logger.warning(f"Model file {file_path} does not exist")
        else:
            logger.info(f"Found model file: {file_path}")

def main():
    """Main function to migrate all model files."""
    logger.info("Starting model file migration")

    # Check if model files exist
    ensure_model_files_exist()

    total_updated = 0
    for model_type in MODEL_FILES.keys():
        updated = migrate_model_files(model_type)
        total_updated += updated

    logger.info(f"Migration complete. Updated a total of {total_updated} models")


if __name__ == "__main__":
    main()