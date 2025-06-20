#!/bin/bash

# Script to set up shortcuts for adding to CLAUDE.md

echo "Setting up CLAUDE.md shortcuts..."

# Create the shortcuts in ~/.bashrc
cat >> ~/.bashrc << 'EOF'

# Shortcut to add notes to CLAUDE.md
alias claude='function _claude() {
    CLAUDE_MD="/mnt/d/ml_project_refactored/CLAUDE.md"
    
    if [ ! -f "$CLAUDE_MD" ]; then
        echo "Error: CLAUDE.md not found at $CLAUDE_MD"
        return 1
    fi
    
    if [ $# -eq 0 ]; then
        echo "Usage: claude \"Your note to add to CLAUDE.md\""
        echo "   or: claude-todo \"A todo item\""
        echo "   or: claude-fix \"A fix that was applied\""
        return 1
    fi
    
    echo "" >> "$CLAUDE_MD"
    echo "## Note ($(date +%Y-%m-%d))" >> "$CLAUDE_MD"
    echo "$*" >> "$CLAUDE_MD"
    
    echo "✓ Added to CLAUDE.md: $*"
}; _claude'

# Shortcut to add TODO items
alias claude-todo='function _claude_todo() {
    CLAUDE_MD="/mnt/d/ml_project_refactored/CLAUDE.md"
    
    if [ ! -f "$CLAUDE_MD" ]; then
        echo "Error: CLAUDE.md not found at $CLAUDE_MD"
        return 1
    fi
    
    # Check if TODO section exists
    if ! grep -q "## TODO" "$CLAUDE_MD"; then
        echo "" >> "$CLAUDE_MD"
        echo "## TODO" >> "$CLAUDE_MD"
    fi
    
    # Add the todo item
    sed -i "/## TODO/a - [ ] $*" "$CLAUDE_MD"
    
    echo "✓ Added TODO to CLAUDE.md: $*"
}; _claude_todo'

# Shortcut to add fixes/updates
alias claude-fix='function _claude_fix() {
    CLAUDE_MD="/mnt/d/ml_project_refactored/CLAUDE.md"
    
    if [ ! -f "$CLAUDE_MD" ]; then
        echo "Error: CLAUDE.md not found at $CLAUDE_MD"
        return 1
    fi
    
    # Add to Recent Fixes section
    sed -i "/## Recent Fixes and Updates/a - **$(date +%Y-%m-%d)**: $*" "$CLAUDE_MD"
    
    echo "✓ Added fix to CLAUDE.md: $*"
}; _claude_fix'

# Short alias using 'cl' instead of '#'
alias cl=claude
alias clt=claude-todo
alias clf=claude-fix

EOF

echo "✅ Shortcuts added to ~/.bashrc"
echo ""
echo "To activate the shortcuts, run:"
echo "  source ~/.bashrc"
echo ""
echo "Available shortcuts:"
echo "  claude \"Note to remember\"     # Add a general note"
echo "  cl \"Note to remember\"         # Short version" 
echo "  claude-todo \"Implement X\"     # Add a TODO item"
echo "  clt \"Implement X\"             # Short version"
echo "  claude-fix \"Fixed Y issue\"    # Add to fixes section"
echo "  clf \"Fixed Y issue\"           # Short version"
echo ""
echo "Examples:"
echo "  cl \"Remember to run with --force-retune when changing hyperparameters\""
echo "  clt \"Add support for neural network models\""
echo "  clf \"Fixed memory leak in visualization pipeline\""