"""Evaluation pipeline for ML models."""

import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.pipelines.base import BasePipeline
from src.evaluation.metrics import evaluate_models
from src.evaluation.baselines import evaluate_baselines
from src.evaluation.importance import analyze_feature_importance
from src.evaluation.multicollinearity import analyze_multicollinearity


class EvaluationPipeline(BasePipeline):
    """Pipeline for evaluating ML models."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the evaluation pipeline.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.evaluation_results = {}
        
    def run(
        self,
        evaluate_metrics: bool = True,
        evaluate_baselines: bool = True,
        analyze_importance: bool = True,
        analyze_vif: bool = False,
        **kwargs
    ):
        """
        Run the evaluation pipeline.
        
        Args:
            evaluate_metrics: Whether to evaluate model metrics
            evaluate_baselines: Whether to evaluate against baselines
            analyze_importance: Whether to analyze feature importance
            analyze_vif: Whether to analyze multicollinearity
            **kwargs: Additional arguments
        """
        self.start_timing()
        
        if evaluate_metrics:
            self._evaluate_metrics()
            
        if evaluate_baselines:
            self._evaluate_baselines()
            
        if analyze_importance:
            self._analyze_feature_importance()
            
        if analyze_vif:
            self._analyze_multicollinearity()
            
        self.report_timing()
        return self.evaluation_results
    
    def _evaluate_metrics(self):
        """Evaluate model metrics."""
        print("\nEvaluating model metrics...")
        with self.time_step("Model Metrics Evaluation"):
            try:
                metrics = evaluate_models()
                self.evaluation_results['metrics'] = metrics
                print("✓ Model metrics evaluation completed")
            except Exception as e:
                print(f"✗ Error evaluating metrics: {e}")
                
    def _evaluate_baselines(self):
        """Evaluate models against baselines."""
        print("\nEvaluating baselines...")
        with self.time_step("Baseline Evaluation"):
            try:
                baselines = evaluate_baselines()
                self.evaluation_results['baselines'] = baselines
                print("✓ Baseline evaluation completed")
            except Exception as e:
                print(f"✗ Error evaluating baselines: {e}")
                
    def _analyze_feature_importance(self):
        """Analyze feature importance."""
        print("\nAnalyzing feature importance...")
        with self.time_step("Feature Importance Analysis"):
            try:
                importance = analyze_feature_importance()
                self.evaluation_results['feature_importance'] = importance
                print("✓ Feature importance analysis completed")
            except Exception as e:
                print(f"✗ Error analyzing feature importance: {e}")
                
    def _analyze_multicollinearity(self):
        """Analyze multicollinearity using VIF."""
        print("\nAnalyzing multicollinearity (VIF)...")
        with self.time_step("VIF Analysis"):
            try:
                vif_results = analyze_multicollinearity()
                self.evaluation_results['vif'] = vif_results
                print("✓ Multicollinearity analysis completed")
            except Exception as e:
                print(f"✗ Error analyzing multicollinearity: {e}")