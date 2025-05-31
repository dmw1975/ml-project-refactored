"""Base pipeline class for all ML pipelines."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional
import time
import datetime


class BasePipeline(ABC):
    """Base class for all ML pipelines."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the pipeline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.step_times = {}
        self.start_time = None
        
    @abstractmethod
    def run(self, **kwargs):
        """Run the pipeline."""
        pass
    
    def time_step(self, step_name: str):
        """Context manager to time a pipeline step."""
        class StepTimer:
            def __init__(self, pipeline, name):
                self.pipeline = pipeline
                self.name = name
                self.start = None
                
            def __enter__(self):
                self.start = time.time()
                return self
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                self.pipeline.step_times[self.name] = time.time() - self.start
                
        return StepTimer(self, step_name)
    
    def start_timing(self):
        """Start timing the pipeline execution."""
        self.start_time = time.time()
        
    def report_timing(self):
        """Report timing information for the pipeline."""
        if not self.start_time:
            return
            
        total_time = time.time() - self.start_time
        time_formatted = str(datetime.timedelta(seconds=int(total_time)))
        
        print(f"\nTotal execution time: {time_formatted}")
        
        if self.step_times:
            print("\nExecution time breakdown by step:")
            print("-" * 50)
            print(f"{'Step':<35} | {'Time (sec)':<10} | {'Time %':<10}")
            print("-" * 50)
            
            sorted_steps = sorted(self.step_times.items(), key=lambda x: x[1], reverse=True)
            
            for step, step_time in sorted_steps:
                percent = (step_time / total_time) * 100
                time_formatted = str(datetime.timedelta(seconds=int(step_time)))
                print(f"{step:<35} | {time_formatted:<10} | {percent:6.2f}%")