#!/bin/bash
# Comprehensive Pipeline Output Validation Script

echo "============================================"
echo "ML Pipeline Output Validation Report"
echo "Generated: $(date)"
echo "============================================"

# Color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check file existence
check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✓${NC} $2"
        return 0
    else
        echo -e "${RED}✗${NC} $2"
        return 1
    fi
}

# Function to count files in directory
count_files() {
    if [ -d "$1" ]; then
        count=$(find "$1" -name "*.png" -type f 2>/dev/null | wc -l)
        echo -e "${GREEN}✓${NC} $2: $count files"
        return $count
    else
        echo -e "${RED}✗${NC} $2: Directory not found"
        return 0
    fi
}

echo -e "\n1. TRAINED MODELS CHECK"
echo "------------------------"
models_ok=0
check_file "outputs/models/catboost_models.pkl" "CatBoost models" && ((models_ok++))
check_file "outputs/models/lightgbm_models.pkl" "LightGBM models" && ((models_ok++))
check_file "outputs/models/xgboost_models.pkl" "XGBoost models" && ((models_ok++))
check_file "outputs/models/elasticnet_models.pkl" "ElasticNet models" && ((models_ok++))
check_file "outputs/models/linear_regression_models.pkl" "Linear Regression models" && ((models_ok++))
echo "Models complete: $models_ok/5"

echo -e "\n2. SHAP VISUALIZATIONS CHECK"
echo "----------------------------"
shap_dir="outputs/visualizations/shap"
if [ -d "$shap_dir" ]; then
    echo "SHAP directories found:"
    catboost_count=$(find "$shap_dir" -name "CatBoost_*" -type d 2>/dev/null | wc -l)
    lightgbm_count=$(find "$shap_dir" -name "LightGBM_*" -type d 2>/dev/null | wc -l)
    xgboost_count=$(find "$shap_dir" -name "XGBoost_*" -type d 2>/dev/null | wc -l)
    elasticnet_count=$(find "$shap_dir" -name "ElasticNet_*" -type d 2>/dev/null | wc -l)
    
    if [ $catboost_count -eq 0 ]; then
        echo -e "${RED}✗${NC} CatBoost: $catboost_count directories (MISSING - Expected 8)"
    else
        echo -e "${GREEN}✓${NC} CatBoost: $catboost_count directories"
    fi
    
    if [ $lightgbm_count -eq 0 ]; then
        echo -e "${RED}✗${NC} LightGBM: $lightgbm_count directories (MISSING - Expected 8)"
    else
        echo -e "${GREEN}✓${NC} LightGBM: $lightgbm_count directories"
    fi
    
    echo -e "${GREEN}✓${NC} XGBoost: $xgboost_count directories"
    echo -e "${GREEN}✓${NC} ElasticNet: $elasticnet_count directories"
    
    total_shap=$((catboost_count + lightgbm_count + xgboost_count + elasticnet_count))
    echo "Total SHAP model directories: $total_shap/28 expected"
else
    echo -e "${RED}✗${NC} SHAP directory not found"
fi

echo -e "\n3. CV DISTRIBUTION PLOTS CHECK"
echo "------------------------------"
cv_dir="outputs/visualizations/performance/cv_distribution"
count_files "$cv_dir" "CV distribution plots"

echo -e "\n4. BASELINE COMPARISONS CHECK"
echo "-----------------------------"
baseline_dir="outputs/visualizations/statistical_tests"
if [ -d "$baseline_dir" ]; then
    baseline_count=$(find "$baseline_dir" -name "baseline_*.png" -type f 2>/dev/null | wc -l)
    echo -e "${GREEN}✓${NC} Baseline comparison plots: $baseline_count files"
    
    # Check for specific model comparisons
    echo "Checking model-specific baselines:"
    for model in "mean" "median" "random"; do
        check_file "$baseline_dir/baseline_comparison_${model}.png" "Baseline ${model} comparison"
    done
else
    echo -e "${RED}✗${NC} Statistical tests directory not found"
fi

echo -e "\n5. FEATURE IMPORTANCE PLOTS CHECK"
echo "---------------------------------"
feature_dir="outputs/visualizations/features"
if [ -d "$feature_dir" ]; then
    # Count plots by model type
    for model in catboost lightgbm xgboost elasticnet linear_regression; do
        count=$(find "$feature_dir" -name "*${model}*" -type f 2>/dev/null | wc -l)
        if [ $count -gt 0 ]; then
            echo -e "${GREEN}✓${NC} ${model}: $count plots"
        else
            echo -e "${YELLOW}⚠${NC} ${model}: $count plots"
        fi
    done
else
    echo -e "${RED}✗${NC} Feature importance directory not found"
fi

echo -e "\n6. RESIDUAL PLOTS CHECK"
echo "-----------------------"
residual_dir="outputs/visualizations/residuals"
count_files "$residual_dir" "Residual plots"

echo -e "\n7. METRICS AND REPORTS CHECK"
echo "----------------------------"
check_file "outputs/visualizations/performance/metrics_summary_table.png" "Metrics summary table"
check_file "outputs/reports/model_performance.csv" "Model performance CSV"
check_file "outputs/metrics/cross_validation_results.json" "Cross-validation results"

echo -e "\n8. PIPELINE STATE CHECK"
echo "-----------------------"
if [ -f "outputs/pipeline_state.json" ]; then
    echo -e "${YELLOW}⚠${NC} Pipeline state file exists"
    # Extract key information
    status=$(grep -o '"status": "[^"]*"' outputs/pipeline_state.json | head -1 | cut -d'"' -f4)
    echo "  Last pipeline status: $status"
else
    echo -e "${GREEN}✓${NC} No pipeline state file (clean state)"
fi

echo -e "\n============================================"
echo "SUMMARY"
echo "============================================"

# Calculate missing components
missing_components=0
[ $catboost_count -eq 0 ] && ((missing_components++)) && echo -e "${RED}CRITICAL:${NC} CatBoost SHAP visualizations missing"
[ $lightgbm_count -eq 0 ] && ((missing_components++)) && echo -e "${RED}CRITICAL:${NC} LightGBM SHAP visualizations missing"

if [ $missing_components -eq 0 ]; then
    echo -e "${GREEN}✓ All critical components present${NC}"
else
    echo -e "${RED}✗ $missing_components critical components missing${NC}"
    echo -e "\n${YELLOW}Recommended Action:${NC}"
    echo "1. Run SHAP generation separately:"
    echo "   python scripts/utilities/generate_shap_visualizations.py"
    echo "2. Or use the safe runner:"
    echo "   python run_pipeline_safe.py --visualize --non-interactive"
fi

echo -e "\n============================================"