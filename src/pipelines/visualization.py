"""Visualization pipeline for ML models."""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.pipelines.base import BasePipeline
from src.pipelines.state_manager import PipelineStage
from src.utils.io import load_all_models
from src.config import settings


class VisualizationPipeline(BasePipeline):
    """Pipeline for generating visualizations."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the visualization pipeline.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.generated_plots = []
        
    def run(
        self,
        plot_types: Optional[List[str]] = None,
        models: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Run the visualization pipeline.
        
        Args:
            plot_types: Types of plots to generate (None = all)
            models: Pre-loaded models (None = load from disk)
            **kwargs: Additional arguments
        """
        # Check if we can start visualization
        if not self.state_manager.can_start_stage(PipelineStage.VISUALIZATION):
            raise RuntimeError("Cannot start visualization: evaluation not completed")
        
        # Additional check: ensure all models are completed before generating metrics table
        if plot_types is None or 'metrics_table' in plot_types:
            if not self.state_manager.all_models_completed():
                print("⚠️ Warning: Not all expected models have completed training.")
                print("  Metrics table generation may be incomplete.")
                # Get status summary
                expected = self.state_manager._state["model_counts"]["expected"]
                completed = self.state_manager._state["model_counts"]["completed"]
                for model_type, exp_count in expected.items():
                    comp_count = completed.get(model_type, 0)
                    if comp_count < exp_count:
                        print(f"  - {model_type}: {comp_count}/{exp_count} completed")
        
        # Start visualization stage
        self.state_manager.start_stage(
            PipelineStage.VISUALIZATION,
            details={
                "plot_types": plot_types,
                "model_count": len(models) if models else "unknown"
            }
        )
        
        self.start_timing()
        
        # Load models if not provided
        if models is None:
            print("Loading models for visualization...")
            models = load_all_models()
            model_list = list(models.values())
        else:
            model_list = list(models.values()) if isinstance(models, dict) else models
            
        # Default to all plot types if none specified
        if plot_types is None:
            plot_types = [
                'residuals', 'comparison', 'metrics_table', 'dashboard',
                'dataset_comparison', 'statistical_tests', 'baselines',
                'feature_importance', 'cv_distributions', 'additional'
            ]
            
        # Import visualization module
        import src.visualization as viz
        
        # Generate each plot type
        if 'residuals' in plot_types:
            self._create_residual_plots(viz)
            
        if 'comparison' in plot_types:
            self._create_comparison_plots(viz, model_list)
            
        if 'metrics_table' in plot_types:
            self._create_metrics_table(viz, model_list)
            
        if 'dashboard' in plot_types:
            self._create_dashboard(viz)
            
        if 'dataset_comparison' in plot_types:
            self._create_dataset_comparisons(viz)
            
        if 'statistical_tests' in plot_types:
            self._create_statistical_tests(viz)
            
        if 'baselines' in plot_types:
            self._create_baseline_plots(viz)
            
        if 'feature_importance' in plot_types:
            self._create_feature_importance(viz)
            
        if 'cv_distributions' in plot_types:
            self._create_cv_distributions(viz)
            
        if 'additional' in plot_types:
            self._create_additional_visualizations()
            
        try:
            self.report_timing()
            
            # Mark visualization as completed
            outputs = {
                "plots_generated": self.generated_plots,
                "plot_count": len(self.generated_plots),
                "visualizations": self.generated_plots
            }
            self.state_manager.complete_stage(PipelineStage.VISUALIZATION, outputs)
            
        except Exception as e:
            self.state_manager.fail_stage(PipelineStage.VISUALIZATION, str(e))
            raise
            
        return self.generated_plots
    
    def _create_residual_plots(self, viz):
        """Create residual plots."""
        print("\nCreating residual plots...")
        with self.time_step("Residual Plots"):
            try:
                viz.create_all_residual_plots()
                self.generated_plots.append('residuals')
                print("✓ Residual plots created")
            except Exception as e:
                print(f"✗ Error creating residual plots: {e}")
                
    def _create_comparison_plots(self, viz, model_list):
        """Create model comparison plots."""
        print("\nCreating model comparison visualizations...")
        with self.time_step("Model Comparison"):
            try:
                viz.create_model_comparison_plot(model_list)
                self.generated_plots.append('comparison')
                print("✓ Model comparison plots created")
            except Exception as e:
                print(f"✗ Error creating model comparison: {e}")
                
    def _create_metrics_table(self, viz, model_list):
        """Create metrics summary table."""
        print("\nCreating metrics summary table...")
        with self.time_step("Metrics Table"):
            # Final check before creating metrics table
            if not self.state_manager.all_models_completed():
                print("⚠️ Warning: Creating metrics table with incomplete model set")
            
            try:
                viz.create_metrics_table(model_list)
                self.generated_plots.append('metrics_table')
                print("✓ Metrics table created")
            except Exception as e:
                print(f"✗ Error creating metrics table: {e}")
                
    def _create_dashboard(self, viz):
        """Create visualization dashboard."""
        print("\nCreating visualization dashboard...")
        with self.time_step("Dashboard"):
            try:
                viz.create_comparative_dashboard()
                self.generated_plots.append('dashboard')
                print("✓ Dashboard created")
            except Exception as e:
                print(f"✗ Error creating dashboard: {e}")
                
    def _create_dataset_comparisons(self, viz):
        """Create dataset comparison plots."""
        print("\nCreating dataset-centric model comparisons...")
        with self.time_step("Dataset Comparisons"):
            try:
                viz.create_all_dataset_comparisons()
                self.generated_plots.append('dataset_comparison')
                print("✓ Dataset comparisons created")
            except Exception as e:
                print(f"✗ Error creating dataset comparisons: {e}")
                
    def _create_statistical_tests(self, viz):
        """Create statistical test visualizations."""
        print("\nCreating statistical test visualizations...")
        with self.time_step("Statistical Tests"):
            try:
                # Get the explicit path to the model comparison tests file
                output_path = settings.OUTPUT_DIR / 'evaluation' / 'model_comparison_tests.csv'
                if output_path.exists():
                    viz.create_statistical_test_visualizations(output_path)
                    self.generated_plots.append('statistical_tests')
                    print("✓ Statistical test visualizations created")
                else:
                    print("⚠️  No statistical test results found. Run evaluation first.")
            except Exception as e:
                print(f"✗ Error creating statistical tests: {e}")
                
    def _create_baseline_plots(self, viz):
        """Create baseline comparison plots."""
        print("\nCreating baseline comparison visualizations...")
        with self.time_step("Baseline Comparisons"):
            try:
                from src.visualization.plots.consolidated_baselines import create_consolidated_baseline_comparison
                create_consolidated_baseline_comparison()
                self.generated_plots.append('baselines')
                print("✓ Baseline comparisons created")
            except Exception as e:
                print(f"✗ Error creating baseline comparisons: {e}")
                
    def _create_feature_importance(self, viz):
        """Create feature importance plots."""
        print("\nCreating feature importance plots...")
        with self.time_step("Feature Importance"):
            try:
                from src.visualization.plots.features import create_cross_model_feature_importance
                create_cross_model_feature_importance()
                self.generated_plots.append('feature_importance')
                print("✓ Feature importance plots created")
            except Exception as e:
                print(f"✗ Error creating feature importance: {e}")
                
    def _create_cv_distributions(self, viz):
        """Create CV distribution plots."""
        print("\nCreating CV distribution plots...")
        with self.time_step("CV Distributions"):
            try:
                from src.visualization.viz_factory import VisualizationFactory
                viz_factory = VisualizationFactory()
                models = load_all_models()
                viz_factory.create_cv_distributions(
                    models=models,
                    output_dir=settings.VISUALS_DIR / "cv_analysis"
                )
                self.generated_plots.append('cv_distributions')
                print("✓ CV distribution plots created")
            except Exception as e:
                print(f"✗ Error creating CV distributions: {e}")
                
    def _create_additional_visualizations(self):
        """Create additional visualizations."""
        print("\nGenerating additional visualizations...")
        with self.time_step("Additional Visualizations"):
            try:
                # Cross-validation plots
                print("  Generating cross-validation plots...")
                from scripts.utilities.generate_model_cv_plots import main as generate_cv_plots
                generate_cv_plots()
                
                # ElasticNet CV plots
                print("  Generating ElasticNet CV plots...")
                from scripts.utilities.generate_elasticnet_cv_plots import generate_elasticnet_cv_plots
                generate_elasticnet_cv_plots()
                
                # Sector stratification plots
                print("  Generating sector stratification plots...")
                from scripts.utilities.create_sector_stratification_plot import create_sector_stratification_plot
                create_sector_stratification_plot()
                
                # SHAP visualizations
                print("  Generating SHAP visualizations...")
                try:
                    from scripts.utilities.generate_shap_visualizations import main as generate_shap_viz
                    generate_shap_viz()
                except Exception as e:
                    print(f"    Warning: SHAP generation timed out or failed: {e}")
                
                # Ensure model comparison SHAP plot is created
                print("  Generating model comparison SHAP plot...")
                try:
                    from scripts.utilities.generate_model_comparison_shap_only import main as generate_comparison
                    if generate_comparison():
                        print("    ✓ Model comparison SHAP plot created")
                    else:
                        print("    ✗ Model comparison SHAP plot failed")
                except Exception as e:
                    print(f"    ✗ Error creating model comparison SHAP plot: {e}")
                
                self.generated_plots.append('additional')
                print("✓ Additional visualizations created")
            except Exception as e:
                print(f"✗ Error creating additional visualizations: {e}")