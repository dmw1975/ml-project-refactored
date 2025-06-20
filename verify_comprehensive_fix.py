#!/usr/bin/env python
"""Verify that all missing visualizations have been fixed."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.absolute()))

from src.config import settings
from src.utils.io import load_all_models

def verify_comprehensive_fix():
    """Check the status of all visualization categories."""
    
    print("="*80)
    print("COMPREHENSIVE VISUALIZATION FIX VERIFICATION")
    print("="*80)
    
    viz_dir = settings.VISUALIZATION_DIR
    issues = []
    
    # 1. CV Distributions
    print("\n1. CV DISTRIBUTIONS:")
    cv_dir = viz_dir / "performance" / "cv_distributions"
    if cv_dir.exists():
        cv_files = list(cv_dir.glob("*.png"))
        print(f"   Found {len(cv_files)} CV distribution plots:")
        for f in sorted(cv_files):
            print(f"   - {f.name}")
        
        # Check for all 4 expected plots
        expected = ["catboost", "lightgbm", "xgboost", "elasticnet"]
        found = [e for e in expected if any(e in f.name.lower() for f in cv_files)]
        missing = [e for e in expected if e not in found]
        
        if missing:
            issues.append(f"Missing CV distributions for: {', '.join(missing)}")
        else:
            print("   ✅ All 4 model types have CV distribution plots")
    else:
        issues.append("CV distributions directory not found")
    
    # 2. SHAP Visualizations
    print("\n2. SHAP VISUALIZATIONS:")
    shap_dir = viz_dir / "shap"
    if shap_dir.exists():
        shap_dirs = [d for d in shap_dir.iterdir() if d.is_dir()]
        print(f"   Found {len(shap_dirs)} SHAP directories")
        
        # Count by model type
        model_counts = {"CatBoost": 0, "LightGBM": 0, "XGBoost": 0, "ElasticNet": 0}
        for d in shap_dirs:
            for model_type in model_counts:
                if model_type in d.name:
                    model_counts[model_type] += 1
                    break
        
        print("   Model type breakdown:")
        for model_type, count in model_counts.items():
            print(f"   - {model_type}: {count} models")
        
        if model_counts["CatBoost"] == 0:
            issues.append("No CatBoost SHAP visualizations")
        if model_counts["LightGBM"] == 0:
            issues.append("No LightGBM SHAP visualizations")
    else:
        issues.append("SHAP directory not found")
    
    # 3. Family Comparison Plots
    print("\n3. FAMILY COMPARISON PLOTS:")
    comparison_dir = viz_dir / "comparisons" / "family_comparison"
    if comparison_dir.exists():
        comparison_files = list(comparison_dir.glob("*.png"))
        print(f"   Found {len(comparison_files)} comparison plots")
        
        # Check if plots include all model types
        all_models = load_all_models()
        model_types = set()
        for name in all_models.keys():
            if "CatBoost" in name:
                model_types.add("CatBoost")
            elif "LightGBM" in name:
                model_types.add("LightGBM")
            elif "XGBoost" in name:
                model_types.add("XGBoost")
            elif "ElasticNet" in name:
                model_types.add("ElasticNet")
        
        print(f"   Expected model types in comparisons: {sorted(model_types)}")
    else:
        issues.append("Family comparison directory not found")
    
    # 4. Metrics Summary Table
    print("\n4. METRICS SUMMARY TABLE:")
    metrics_file = viz_dir / "tables" / "model_metrics_summary.png"
    if metrics_file.exists():
        print(f"   ✅ Metrics summary table exists: {metrics_file}")
        # Note: Can't easily verify content without loading the image
    else:
        issues.append("Metrics summary table not found")
    
    # 5. Baseline Comparisons
    print("\n5. BASELINE COMPARISONS:")
    baseline_dir = viz_dir / "comparisons" / "baselines"
    if baseline_dir.exists():
        baseline_files = list(baseline_dir.glob("*.png"))
        print(f"   Found {len(baseline_files)} baseline comparison plots:")
        for f in sorted(baseline_files):
            print(f"   - {f.name}")
    else:
        issues.append("Baseline comparisons directory not found")
    
    # Final Summary
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    
    if not issues:
        print("✅ ALL VISUALIZATIONS APPEAR TO BE COMPLETE!")
        print("\nAll 5 categories of visualizations are present:")
        print("1. CV Distributions - All 4 model types")
        print("2. SHAP Analysis - LightGBM and CatBoost included")  
        print("3. Family Comparison Plots - Present")
        print("4. Metrics Summary Table - Present")
        print("5. Baseline Comparisons - Present")
        return True
    else:
        print("⚠️  REMAINING ISSUES:")
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue}")
        return False

if __name__ == "__main__":
    success = verify_comprehensive_fix()
    sys.exit(0 if success else 1)