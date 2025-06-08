#!/usr/bin/env python3
"""Safe pipeline runner with timeout and signal handling."""

import signal
import sys
import subprocess
from pathlib import Path
import argparse

def signal_handler(sig, frame):
    print('\nPipeline interrupted by user')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Parse arguments to determine timeout
parser = argparse.ArgumentParser(description='Safe pipeline runner')
parser.add_argument('--timeout', type=int, default=3600, help='Timeout in seconds (default: 3600)')
parser.add_argument('--extended-timeout', action='store_true', help='Use extended timeout of 4 hours for full pipeline')
args, remaining_args = parser.parse_known_args()

# Determine timeout
if args.extended_timeout:
    timeout = 14400  # 4 hours
    print(f"Using extended timeout: 4 hours")
else:
    timeout = args.timeout
    print(f"Using timeout: {timeout} seconds ({timeout/3600:.1f} hours)")

# Run main with timeout
try:
    print(f"Running: python main.py {' '.join(remaining_args)}")
    result = subprocess.run(
        [sys.executable, "main.py"] + remaining_args,
        timeout=timeout
    )
    sys.exit(result.returncode)
except subprocess.TimeoutExpired:
    print(f"\nPipeline timed out after {timeout} seconds ({timeout/3600:.1f} hours)")
    sys.exit(1)
except KeyboardInterrupt:
    print("\nPipeline interrupted by user")
    sys.exit(0)
