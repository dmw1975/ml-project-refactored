"""Test script to verify evaluation functionality."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

from config import settings
from evaluation.metrics import evaluate_models
from evaluation.importance import analyze_feature_importance

def main():
    """Run evaluation tests."""
    print("Testing model evaluation functionality...")
    
    # Run model evaluation
    eval_results = evaluate_models()
    
    if eval_results:
        # Run feature importance analysis
        importance_results, consolidated = analyze_feature_importance(eval_results['all_models'])
        
        print("\nEvaluation test completed successfully.")
    else:
        print("\nNo models found. Please train models first.")

if __name__ == "__main__":
    main()