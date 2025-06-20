#!/bin/bash

# Script to set up the # shortcut for adding to CLAUDE.md

echo "Setting up # shortcut for CLAUDE.md..."

# Add the function to .bashrc
cat >> ~/.bashrc << 'EOF'

# Shortcut to add to CLAUDE.md
function # {
    if [ $# -eq 0 ]; then
        echo "Usage: # \"Your note to add to CLAUDE.md\""
        return 1
    fi
    
    # Get the CLAUDE.md path (adjust this if your project is elsewhere)
    CLAUDE_MD="/mnt/d/ml_project_refactored/CLAUDE.md"
    
    if [ ! -f "$CLAUDE_MD" ]; then
        echo "Error: CLAUDE.md not found at $CLAUDE_MD"
        return 1
    fi
    
    # Add timestamp and note to CLAUDE.md
    echo "" >> "$CLAUDE_MD"
    echo "## Note added on $(date '+%Y-%m-%d %H:%M:%S')" >> "$CLAUDE_MD"
    echo "$*" >> "$CLAUDE_MD"
    
    echo "✓ Added to CLAUDE.md: $*"
}
EOF

echo "✓ Function added to ~/.bashrc"
echo ""
echo "To activate the shortcut, run:"
echo "  source ~/.bashrc"
echo ""
echo "Then you can use it like:"
echo "  # \"Remember to check sklearn version before running tree models\""
echo ""
echo "The note will be appended to CLAUDE.md with a timestamp."