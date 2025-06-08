#\!/usr/bin/env python3
"""Debug script to find where the pipeline is hanging."""

import sys
import os
from pathlib import Path
import traceback

# Add parent directory to path
project_dir = Path(__file__).parent
print(f"1. Adding project directory to path: {project_dir}")
sys.path.append(str(project_dir))

try:
    print("2. Importing datetime...")
    from datetime import datetime
    print("   OK datetime imported successfully")
    
    print("3. Importing logging...")
    import logging
    print("   OK logging imported successfully")
    
    print("4. Importing argparse...")
    import argparse
    print("   OK argparse imported successfully")
    
    print("5. Creating argparse parser...")
    parser = argparse.ArgumentParser(description='ML Pipeline')
    parser.add_argument('--visualize', action='store_true', help='Run visualization')
    parser.add_argument('--non-interactive', action='store_true', help='Non-interactive mode')
    args = parser.parse_args()
    print(f"   OK Parser created and args parsed: {args}")
    
    print("6. Importing state manager...")
    from src.pipelines.state_manager import get_state_manager, PipelineStage
    print("   OK state_manager imported successfully")
    
    print("7. Getting state manager instance...")
    state_manager = get_state_manager()
    print("   OK state_manager instance created")
    
    print("8. Starting INITIALIZATION stage...")
    state_manager.start_stage(PipelineStage.INITIALIZATION)
    print("   OK INITIALIZATION stage started")
    
    print("9. Creating logs directory...")
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    print(f"   OK Logs directory created/confirmed: {log_dir}")
    
    print("10. Creating log filename...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = log_dir / f"debug_{timestamp}.log"
    print(f"   OK Log filename created: {log_filename}")
    
    print("11. Configuring logging...")
    print("    - Creating FileHandler...")
    file_handler = logging.FileHandler(log_filename)
    print("    OK FileHandler created")
    
    print("    - Creating StreamHandler...")
    stream_handler = logging.StreamHandler()
    print("    OK StreamHandler created")
    
    print("    - Calling logging.basicConfig...")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[file_handler, stream_handler]
    )
    print("   OK logging.basicConfig completed")
    
    print("12. Testing logging...")
    logging.info("Test log message")
    print("   OK Test log message sent")
    
    print("\nALL TESTS PASSED\! The hang is NOT in the basic initialization code.")
    
except Exception as e:
    print(f"\nERROR occurred: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)
