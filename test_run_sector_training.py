"""Script to run sector model training, evaluation, and visualization."""

from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

print("Training sector models...")
from models.sector_models import run_sector_models
sector_models = run_sector_models()

print("\nEvaluating sector models...")
from models.sector_models import evaluate_sector_models
sector_eval_results = evaluate_sector_models(sector_models)

print("\nSaving metrics to CSV...")
import pandas as pd
from config import settings
metrics_df = pd.DataFrame([
    {
        'model_name': name,
        'sector': metrics['sector'],
        'type': metrics['type'],
        'RMSE': metrics['RMSE'],
        'MAE': metrics['MAE'],
        'MSE': metrics['MSE'],
        'R2': metrics['R2'],
        'n_companies': metrics['n_companies']
    }
    for name, metrics in sector_models.items()
])
metrics_df.to_csv(f"{settings.METRICS_DIR}/sector_models_metrics.csv", index=False)
print(f"Metrics saved to {settings.METRICS_DIR}/sector_models_metrics.csv")