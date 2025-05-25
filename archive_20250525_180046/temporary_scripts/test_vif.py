"""Test script to analyze multicollinearity using VIF."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

from evaluation.multicollinearity import analyze_multicollinearity

def main():
    """Run VIF analysis."""
    print("Testing Variance Inflation Factor (VIF) analysis...")
    
    # Run VIF analysis
    base_vif, yeo_vif = analyze_multicollinearity()
    
    # Print summary
    print("\nTop 5 Base features with highest VIF:")
    print(base_vif.head(5))
    
    print("\nTop 5 Yeo-Johnson transformed features with highest VIF:")
    print(yeo_vif.head(5))
    
    print("\nVIF analysis test completed successfully.")

if __name__ == "__main__":
    main()