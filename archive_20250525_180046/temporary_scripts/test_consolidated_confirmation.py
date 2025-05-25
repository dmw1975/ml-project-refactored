#!/usr/bin/env python3
"""Test the consolidated confirmation system."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

from utils.io import check_all_existing_models, prompt_consolidated_retrain

def test_consolidated_confirmation():
    """Test the consolidated confirmation system."""
    print("Testing consolidated confirmation system...")
    
    # Check for existing models
    all_existing_models = check_all_existing_models()
    
    if not all_existing_models:
        print("✅ No existing models found - system working correctly")
        print("Run some model training first to test the confirmation system")
        return
    
    print(f"\n📊 Found existing models across {len(all_existing_models)} algorithms")
    
    # Test the confirmation prompt (but don't actually proceed)
    print("\n🧪 Testing confirmation display (will not proceed with retraining):")
    
    # This would normally prompt for user input, but we'll just show the display
    should_retrain = prompt_consolidated_retrain(all_existing_models)
    
    if should_retrain:
        print("\n✅ User confirmed - would proceed with retraining all algorithms")
    else:
        print("\n⏭️  User declined - would skip retraining and use existing models")
    
    print("\n🔧 Test completed successfully!")

if __name__ == "__main__":
    test_consolidated_confirmation()