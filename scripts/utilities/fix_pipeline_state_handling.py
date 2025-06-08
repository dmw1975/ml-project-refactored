#!/usr/bin/env python3
"""Fix pipeline state handling for aborted runs and force-retune."""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


def update_state_manager():
    """Update state manager to handle aborted runs."""
    
    state_manager_path = project_root / "src" / "pipelines" / "state_manager.py"
    
    with open(state_manager_path, 'r') as f:
        content = f.read()
    
    # Find the _load_or_create_state method
    method_start = content.find("def _load_or_create_state(self)")
    if method_start < 0:
        print("Could not find _load_or_create_state method")
        return
    
    # Find the end of the try block where state is loaded
    try_block_end = content.find("return state", method_start)
    if try_block_end < 0:
        print("Could not find state return")
        return
    
    # Insert validation code before returning state
    validation_code = """
                # Validate loaded state - check for aborted runs
                if self._is_state_stale(state):
                    print("Warning: Detected stale/aborted pipeline state. Creating fresh state.")
                    return self._create_new_state()
                """
    
    # Insert before return state
    content = content[:try_block_end] + validation_code + "\n                " + content[try_block_end:]
    
    # Add the validation methods after _load_or_create_state
    method_end = content.find("\n    def ", method_start + 1)
    if method_end < 0:
        method_end = content.find("\n    def _save_state", method_start)
    
    new_methods = """
    def _is_state_stale(self, state: Dict[str, Any]) -> bool:
        \"\"\"Check if a loaded state is stale or from an aborted run.\"\"\"
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
        \"\"\"Create a fresh pipeline state.\"\"\"
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
        \"\"\"Reset the pipeline state to start fresh.\"\"\"
        print("Resetting pipeline state...")
        self._state = self._create_new_state()
        self._save_state()
        print("Pipeline state reset successfully")
"""
    
    content = content[:method_end] + new_methods + content[method_end:]
    
    # Write back
    with open(state_manager_path, 'w') as f:
        f.write(content)
    
    print("✓ Updated state manager with stale state detection")


def update_main_for_force_retune():
    """Update main.py to reset state when using --force-retune."""
    
    main_path = project_root / "main.py"
    
    with open(main_path, 'r') as f:
        content = f.read()
    
    # Find where state manager is initialized
    init_start = content.find("state_manager = get_state_manager()")
    if init_start < 0:
        print("Could not find state manager initialization")
        return
    
    # Find the line after state manager initialization
    init_end = content.find("\n", init_start)
    
    # Check if force-retune handling already exists
    if "args.force_retune" not in content[init_start:init_start + 500]:
        # Add force-retune handling
        force_retune_code = """
        
        # Reset state if force-retune is requested
        if hasattr(args, 'force_retune') and args.force_retune:
            print("Force retune requested - resetting pipeline state")
            state_manager.reset_state()"""
        
        # Find where to insert - after state_manager.start_stage
        start_stage_pos = content.find("state_manager.start_stage(PipelineStage.INITIALIZATION)", init_start)
        if start_stage_pos > 0:
            insert_pos = content.find("\n", start_stage_pos) + 1
        else:
            insert_pos = init_end + 1
        
        content = content[:insert_pos] + force_retune_code + "\n" + content[insert_pos:]
    
    # Write back
    with open(main_path, 'w') as f:
        f.write(content)
    
    print("✓ Updated main.py to reset state on --force-retune")


def add_cleanup_on_exit():
    """Add proper cleanup when pipeline is interrupted."""
    
    state_manager_path = project_root / "src" / "pipelines" / "state_manager.py"
    
    with open(state_manager_path, 'r') as f:
        content = f.read()
    
    # Update the _cleanup method to handle interrupted pipelines better
    cleanup_start = content.find("def _cleanup(self):")
    if cleanup_start > 0:
        cleanup_end = content.find("\n\n", cleanup_start)
        
        new_cleanup = '''    def _cleanup(self):
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
                self._save_state()'''
        
        content = content[:cleanup_start] + new_cleanup + content[cleanup_end:]
        
        with open(state_manager_path, 'w') as f:
            f.write(content)
        
        print("✓ Updated cleanup handler for interrupted pipelines")


def main():
    """Apply all fixes for pipeline state handling."""
    
    print("Fixing pipeline state handling...\n")
    
    # First, delete any existing corrupted state
    state_file = project_root / "outputs" / "pipeline_state.json"
    if state_file.exists():
        print(f"Removing existing pipeline state: {state_file}")
        state_file.unlink()
    
    # Apply fixes
    update_state_manager()
    update_main_for_force_retune()
    add_cleanup_on_exit()
    
    print("\n✓ Pipeline state handling fixed")
    print("\nThe pipeline will now:")
    print("1. Detect and handle stale/aborted states")
    print("2. Reset state when using --force-retune")
    print("3. Properly cleanup when interrupted")
    print("4. Start fresh if previous run was incomplete")


if __name__ == "__main__":
    main()