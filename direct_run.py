#!/usr/bin/env python3
"""Direct pipeline runner that imports and runs main."""

import sys
import os

# Ensure unbuffered output
os.environ['PYTHONUNBUFFERED'] = '1'

# Set up arguments
sys.argv = ['main.py', '--visualize', '--non-interactive']

# Import and run
from main import main
main()