#!/usr/bin/env python
"""
Script to check if the raw data has sector columns and add them if not.
"""
import sys
from pathlib import Path
import pandas as pd
import pickle

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

from config import settings

def check_and_add_sector_columns():
    """Check if raw data has sector columns and add them if not."""
    # Path to the raw data file
    raw_data_path = Path(settings.DATA_DIR) / "raw" / "combined_df_for_ml_models.csv"
    
    print(f"Reading raw data from {raw_data_path}...")
    
    # Load raw data
    try:
        raw_data = pd.read_csv(raw_data_path)
        print(f"Raw data loaded successfully with shape {raw_data.shape}")
        
        # Check for sector columns
        sector_cols = [col for col in raw_data.columns if col.startswith('Sector_')]
        print(f"Found {len(sector_cols)} sector columns: {sector_cols}")
        
        if not sector_cols:
            # Look for sector-related columns
            sector_related_cols = [col for col in raw_data.columns if 'sector' in col.lower()]
            print(f"Found sector-related columns: {sector_related_cols}")
            
            # Try to create sector columns from gics_sector if available
            if 'gics_sector' in raw_data.columns:
                print("\nCreating one-hot encoded sector columns from gics_sector...")
                
                # Get unique sectors
                sectors = raw_data['gics_sector'].unique()
                print(f"Found {len(sectors)} unique sectors")
                
                # Create one-hot encoding for sectors
                for sector in sectors:
                    if sector and not pd.isna(sector):
                        col_name = f"Sector_{sector.replace(' ', '_')}"
                        raw_data[col_name] = (raw_data['gics_sector'] == sector).astype(int)
                        print(f"Created column {col_name}")
                
                # Save the modified data
                output_path = raw_data_path.with_name("combined_df_with_sectors.csv")
                raw_data.to_csv(output_path, index=False)
                print(f"\nSaved modified data with sector columns to {output_path}")
            else:
                print("\nNo 'gics_sector' column found, cannot create sector columns")
        else:
            print("\nSector columns already exist in the raw data")
        
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    check_and_add_sector_columns()