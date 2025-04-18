"""I/O utilities for saving and loading data."""

import pickle
import pandas as pd
import os
from pathlib import Path

def save_model(model, filename, directory):
    """Save a model to a pickle file."""
    path = Path(directory) / filename
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    return path

def load_model(filename, directory):
    """Load a model from a pickle file."""
    path = Path(directory) / filename
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_metrics(metrics_dict, filename, directory):
    """Save metrics to a CSV file."""
    path = Path(directory) / filename
    pd.DataFrame(metrics_dict).to_csv(path)
    return path

def ensure_dir(directory):
    """Ensure a directory exists."""
    os.makedirs(directory, exist_ok=True)
    return directory