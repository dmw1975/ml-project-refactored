"""Pipeline state management for tracking stage completion and dependencies."""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from enum import Enum
import threading
import atexit


class PipelineStage(Enum):
    """Enumeration of pipeline stages."""
    INITIALIZATION = "initialization"
    TRAINING = "training"
    EVALUATION = "evaluation"
    STATISTICAL_TESTS = "statistical_tests"
    VISUALIZATION = "visualization"
    COMPLETION = "completion"


class StageStatus(Enum):
    """Status of a pipeline stage."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class PipelineStateManager:
    """Manages pipeline state and stage completion tracking."""
    
    def __init__(self, state_file: Optional[Path] = None):
        """
        Initialize the pipeline state manager.
        
        Args:
            state_file: Path to state file. If None, uses default location.
        """
        if state_file is None:
            from src.config import settings
            state_file = settings.OUTPUT_DIR / "pipeline_state.json"
        
        self.state_file = Path(state_file)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        
        self._lock = threading.Lock()
        self._state = self._load_or_create_state()
        
        # Register cleanup on exit
        atexit.register(self._cleanup)
    
    def _load_or_create_state(self) -> Dict[str, Any]:
        """Load existing state or create new one."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                print(f"Loaded existing pipeline state from {self.state_file}")
                
                # Validate loaded state - check for aborted runs
                if self._is_state_stale(state):
                    print("Warning: Detected stale/aborted pipeline state. Creating fresh state.")
                    return self._create_new_state()
                
                return state
            except Exception as e:
                print(f"Warning: Could not load state file: {e}. Creating new state.")
        
        # Create new state
        return {
            "pipeline_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "stages": {
                stage.value: {
                    "status": StageStatus.NOT_STARTED.value,
                    "start_time": None,
                    "end_time": None,
                    "details": {},
                    "errors": []
                }
                for stage in PipelineStage
            },
            "model_counts": {
                "expected": {},
                "completed": {}
            },
            "outputs": {
                "models": [],
                "visualizations": [],
                "reports": []
            }
        }
    
    def _is_state_stale(self, state: Dict[str, Any]) -> bool:
        """Check if a loaded state is stale or from an aborted run."""
        # Check if pipeline has end_time but stages are incomplete
        if state.get("end_time") and state["stages"]["initialization"]["status"] == "in_progress":
            return True
        
        # Check if any stage has been in_progress for too long (>1 hour)
        for stage_name, stage_info in state["stages"].items():
            if stage_info["status"] == "in_progress" and stage_info["start_time"]:
                start_time = datetime.fromisoformat(stage_info["start_time"])
                if datetime.now() - start_time > timedelta(hours=1):
                    return True
        
        # Check if the pipeline is from a different day
        if state.get("start_time"):
            start_time = datetime.fromisoformat(state["start_time"])
            if start_time.date() != datetime.now().date():
                return True
                
        return False
    
    def _create_new_state(self) -> Dict[str, Any]:
        """Create a fresh pipeline state."""
        return {
            "pipeline_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "stages": {
                stage.value: {
                    "status": StageStatus.NOT_STARTED.value,
                    "start_time": None,
                    "end_time": None,
                    "details": {},
                    "errors": []
                }
                for stage in PipelineStage
            },
            "model_counts": {
                "expected": {},
                "completed": {}
            },
            "outputs": {
                "models": [],
                "visualizations": [],
                "reports": []
            }
        }
    
    def reset_state(self):
        """Reset the pipeline state to start fresh."""
        print("Resetting pipeline state...")
        with self._lock:
            self._state = self._create_new_state()
            self._save_state()
        print("Pipeline state reset successfully")

    def _save_state(self):
        """Save current state to file (assumes lock is already held)."""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self._state, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save state file: {e}")
    
    def start_stage(self, stage: PipelineStage, details: Optional[Dict] = None):
        """
        Mark a stage as started.
        
        Args:
            stage: The pipeline stage to start
            details: Optional details about the stage
        """
        with self._lock:
            stage_info = self._state["stages"][stage.value]
            stage_info["status"] = StageStatus.IN_PROGRESS.value
            stage_info["start_time"] = datetime.now().isoformat()
            if details:
                stage_info["details"].update(details)
            
            print(f"\n{'='*60}")
            print(f"Starting pipeline stage: {stage.value.upper()}")
            print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*60}")
            
            self._save_state()
    
    def complete_stage(self, stage: PipelineStage, outputs: Optional[Dict] = None):
        """
        Mark a stage as completed.
        
        Args:
            stage: The pipeline stage that completed
            outputs: Optional outputs from the stage
        """
        with self._lock:
            stage_info = self._state["stages"][stage.value]
            stage_info["status"] = StageStatus.COMPLETED.value
            stage_info["end_time"] = datetime.now().isoformat()
            
            if outputs:
                stage_info["details"]["outputs"] = outputs
                
                # Update global outputs
                if "models" in outputs:
                    self._state["outputs"]["models"].extend(outputs["models"])
                if "visualizations" in outputs:
                    self._state["outputs"]["visualizations"].extend(outputs["visualizations"])
                if "reports" in outputs:
                    self._state["outputs"]["reports"].extend(outputs["reports"])
            
            # Calculate duration
            start = datetime.fromisoformat(stage_info["start_time"])
            end = datetime.fromisoformat(stage_info["end_time"])
            duration = (end - start).total_seconds()
            
            print(f"\n✅ Completed stage: {stage.value.upper()}")
            print(f"Duration: {duration:.1f} seconds")
            
            self._save_state()
    
    def fail_stage(self, stage: PipelineStage, error: str):
        """
        Mark a stage as failed.
        
        Args:
            stage: The pipeline stage that failed
            error: Error message
        """
        with self._lock:
            stage_info = self._state["stages"][stage.value]
            stage_info["status"] = StageStatus.FAILED.value
            stage_info["end_time"] = datetime.now().isoformat()
            stage_info["errors"].append({
                "time": datetime.now().isoformat(),
                "message": error
            })
            
            print(f"\n❌ Failed stage: {stage.value.upper()}")
            print(f"Error: {error}")
            
            self._save_state()
    
    def get_stage_status(self, stage: PipelineStage) -> StageStatus:
        """Get the current status of a stage."""
        with self._lock:
            status_str = self._state["stages"][stage.value]["status"]
            return StageStatus(status_str)
    
    def can_start_stage(self, stage: PipelineStage) -> bool:
        """
        Check if a stage can be started based on dependencies.
        
        Args:
            stage: The stage to check
            
        Returns:
            True if the stage can be started
        """
        # Define stage dependencies
        dependencies = {
            PipelineStage.INITIALIZATION: [],
            PipelineStage.TRAINING: [PipelineStage.INITIALIZATION],
            PipelineStage.EVALUATION: [PipelineStage.TRAINING],
            PipelineStage.STATISTICAL_TESTS: [PipelineStage.EVALUATION],
            PipelineStage.VISUALIZATION: [PipelineStage.EVALUATION],
            PipelineStage.COMPLETION: [
                PipelineStage.STATISTICAL_TESTS,
                PipelineStage.VISUALIZATION
            ]
        }
        
        # Check if all dependencies are completed
        for dep in dependencies.get(stage, []):
            if self.get_stage_status(dep) != StageStatus.COMPLETED:
                print(f"Cannot start {stage.value}: dependency {dep.value} not completed")
                return False
        
        return True
    
    def set_expected_models(self, model_type: str, count: int):
        """Set expected model count for tracking."""
        with self._lock:
            self._state["model_counts"]["expected"][model_type] = count
            self._save_state()
    
    def increment_completed_models(self, model_type: str):
        """Increment completed model count."""
        with self._lock:
            if model_type not in self._state["model_counts"]["completed"]:
                self._state["model_counts"]["completed"][model_type] = 0
            self._state["model_counts"]["completed"][model_type] += 1
            self._save_state()
    
    def all_models_completed(self) -> bool:
        """Check if all expected models are completed."""
        with self._lock:
            expected = self._state["model_counts"]["expected"]
            completed = self._state["model_counts"]["completed"]
            
            for model_type, expected_count in expected.items():
                if completed.get(model_type, 0) < expected_count:
                    return False
            
            return True
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get a summary of the pipeline state."""
        with self._lock:
            summary = {
                "pipeline_id": self._state["pipeline_id"],
                "start_time": self._state["start_time"],
                "stages": {}
            }
            
            for stage_name, stage_info in self._state["stages"].items():
                summary["stages"][stage_name] = {
                    "status": stage_info["status"],
                    "duration": None
                }
                
                if stage_info["start_time"] and stage_info["end_time"]:
                    start = datetime.fromisoformat(stage_info["start_time"])
                    end = datetime.fromisoformat(stage_info["end_time"])
                    summary["stages"][stage_name]["duration"] = (end - start).total_seconds()
            
            return summary
    
    def print_summary(self):
        """Print a summary of the pipeline execution."""
        summary = self.get_pipeline_summary()
        
        print(f"\n{'='*60}")
        print("PIPELINE EXECUTION SUMMARY")
        print(f"{'='*60}")
        print(f"Pipeline ID: {summary['pipeline_id']}")
        print(f"Start Time: {summary['start_time']}")
        print(f"\nStage Status:")
        
        for stage_name, stage_info in summary["stages"].items():
            status = stage_info["status"]
            duration = stage_info["duration"]
            
            status_symbol = {
                StageStatus.NOT_STARTED.value: "⏸",
                StageStatus.IN_PROGRESS.value: "🔄",
                StageStatus.COMPLETED.value: "✅",
                StageStatus.FAILED.value: "❌",
                StageStatus.SKIPPED.value: "⏭"
            }.get(status, "?")
            
            print(f"  {status_symbol} {stage_name:.<30} {status}")
            if duration:
                print(f"     Duration: {duration:.1f}s")
        
        print(f"{'='*60}")
    
    def _cleanup(self):
        """Cleanup on exit."""
        if hasattr(self, '_state') and self._state:
            # Mark any in-progress stages as failed
            for stage_name, stage_info in self._state["stages"].items():
                if stage_info["status"] == "in_progress":
                    stage_info["status"] = "failed"
                    stage_info["end_time"] = datetime.now().isoformat()
                    stage_info["errors"].append({
                        "time": datetime.now().isoformat(),
                        "message": "Pipeline interrupted"
                    })
            
            # Mark pipeline as completed if not already
            if self._state["end_time"] is None:
                self._state["end_time"] = datetime.now().isoformat()
                self._save_state()


# Global state manager instance
_state_manager = None


def get_state_manager() -> PipelineStateManager:
    """Get or create the global state manager instance."""
    global _state_manager
    if _state_manager is None:
        _state_manager = PipelineStateManager()
    return _state_manager