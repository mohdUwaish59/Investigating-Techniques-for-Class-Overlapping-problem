#!/usr/bin/env python3
"""
Simple integration test for DeviOCSVM
Tests basic import and instantiation
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

def test_import():
    """Test if DeviOCSVM can be imported"""
    try:
        from techniques.devi_ocsvm import DeviOCSVM
        print("✓ DeviOCSVM import successful")
        return DeviOCSVM
    except ImportError as e:
        print(f"✗ DeviOCSVM import failed: {e}")
        return None

def test_instantiation(DeviOCSVM):
    """Test if DeviOCSVM can be instantiated"""
    try:
        devi = DeviOCSVM(nu=0.5, K1=1, K2=5, K3=1, verbose=False)
        print("✓ DeviOCSVM instantiation successful")
        return devi
    except Exception as e:
        print(f"✗ DeviOCSVM instantiation failed: {e}")
        return None

def test_basic_functionality(devi):
    """Test basic functionality with simple data"""
    try:
        # Create simple test data
        np.random.seed(42)
        X = np.random.rand(50, 3)
        y = np.array([0]*40 + [1]*10)  # Imbalanced
        
        X_res, y_res = devi.fit_resample(X, y)
        
        print(f"✓ Basic functionality test passed")
        print(f"  Original: {X.shape}, {np.bincount(y)}")
        print(f"  Resampled: {X_res.shape}, {np.bincount(y_res)}")
        return True
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

def main():
    """Run integration tests"""
    print("DeviOCSVM Integration Test")
    print("="*40)
    
    # Test import
    DeviOCSVM = test_import()
    if DeviOCSVM is None:
        return
    
    # Test instantiation
    devi = test_instantiation(DeviOCSVM)
    if devi is None:
        return
    
    # Test basic functionality
    success = test_basic_functionality(devi)
    
    if success:
        print("\n✓ All integration tests passed!")
        print("DeviOCSVM is ready to use.")
    else:
        print("\n✗ Integration tests failed!")

if __name__ == "__main__":
    main()