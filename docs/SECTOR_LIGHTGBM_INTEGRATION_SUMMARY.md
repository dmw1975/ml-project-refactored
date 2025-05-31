# Sector LightGBM Models - Pipeline Integration Summary

## Date: 2025-05-31

### Integration Completed

Successfully integrated sector LightGBM models into the main pipeline's `--all` flag.

### Changes Made

#### 1. main.py Updates (Lines 1246-1260)
```python
# LightGBM Sector model pipeline (including when --all is used)
if args.train_sector_lightgbm or args.all:
    print("\nTraining sector-specific LightGBM models...")
    from src.models.sector_lightgbm_models import run_sector_lightgbm_models
    sector_lightgbm_models = run_sector_lightgbm_models()

if args.evaluate_sector_lightgbm or args.all:
    print("\nEvaluating sector-specific LightGBM models...")
    from src.models.sector_lightgbm_models import evaluate_sector_lightgbm_models
    sector_lightgbm_eval_results = evaluate_sector_lightgbm_models()
    
    print("\nAnalyzing sector LightGBM model feature importance...")
    from src.models.sector_lightgbm_models import analyze_sector_lightgbm_importance
    sector_lightgbm_importance_results = analyze_sector_lightgbm_importance()

if args.visualize_sector_lightgbm or args.all:
    print("\nGenerating sector-specific LightGBM visualizations...")
```

#### 2. comprehensive.py Already Included Support (Lines 234-256)
The comprehensive visualization pipeline already checks for and generates LightGBM sector visualizations:
- Checks for `sector_lightgbm_metrics.csv`
- Calls `visualize_lightgbm_sector_plots()` if metrics exist
- Properly handles the generated figures

### How It Works Now

#### With --all flag:
```bash
python main.py --all
```
This will now:
1. Train all standard models (Linear, ElasticNet, XGBoost, LightGBM, CatBoost)
2. **Train sector-specific LightGBM models** (NEW)
3. Evaluate all models including sector LightGBM
4. Generate all visualizations including sector LightGBM plots

#### With specific flags (unchanged):
```bash
python main.py --train-sector-lightgbm
python main.py --evaluate-sector-lightgbm  
python main.py --visualize-sector-lightgbm
```

### Output Files

When sector LightGBM models are trained and visualized:

1. **Models**: `outputs/models/sector_lightgbm_models.pkl`
2. **Metrics**: `outputs/metrics/sector_lightgbm_metrics.csv`
3. **Feature Importance**: `outputs/metrics/feature_importance/sector_lightgbm_<sector>_<type>_importance.csv`
4. **Visualizations**: `outputs/visualizations/sectors/lightgbm/`
   - `sector_performance_comparison.png`
   - `sector_model_type_heatmap.png`
   - `overall_vs_sector_comparison.png`
   - `sector_performance_boxplots.png`
   - `sector_metrics_summary_table.png`
   - `sector_train_test_distribution.png`

### Benefits

1. **Convenience**: Users can now train and visualize all models (including sector LightGBM) with a single `--all` flag
2. **Completeness**: Ensures sector LightGBM models aren't forgotten in comprehensive analyses
3. **Consistency**: Sector LightGBM models are treated the same as other model types in the pipeline

### Notes

- The integration preserves backward compatibility with individual flags
- Sector LightGBM training adds approximately 2-5 minutes to the full pipeline
- Each sector needs at least 50 companies to be included in modeling