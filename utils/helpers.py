"""Helper functions used across modules."""
import pandas as pd
import sys
from pathlib import Path

def add_project_root_to_path():
    """Add project root to Python path."""
    project_root = Path(__file__).parent.parent.absolute()
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    return project_root

def format_column_names(df):
    """Format column names to be consistent."""
    return df.columns.str.lower().str.replace(' ', '_')

def safe_float(value):
    """Safely convert any value to float."""
    try:
        if hasattr(value, 'item'):  # numpy value
            return float(value.item())
        elif isinstance(value, pd.Series):
            if len(value) == 1:
                return float(value.iloc[0])
            else:
                return float(value.values[0])
        else:
            return float(value)
    except (TypeError, ValueError):
        return 0.0