#\!/usr/bin/env python3
"""Fixed pipeline runner that bypasses shell issues."""

import sys
import os
os.environ['PYTHONUNBUFFERED'] = '1'  # Force unbuffered output

# Disable any potential shell aliases or functions
if 'BASH_FUNC_neofetch%%' in os.environ:
    del os.environ['BASH_FUNC_neofetch%%']

# Import after environment setup
from pathlib import Path
project_dir = Path(__file__).parent
sys.path.append(str(project_dir))

# Direct import and run
if __name__ == "__main__":
    # Pass arguments to main
    sys.argv = ['main.py', '--visualize', '--non-interactive']
    
    # Import and run main directly
    import main
    main.main()
ENDFILE < /dev/null
