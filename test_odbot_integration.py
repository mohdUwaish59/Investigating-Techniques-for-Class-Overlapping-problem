#!/usr/bin/env python3
"""
Simple integration test for ODBOT
Tests basic import and instantiation
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

def test_import():
    """Test if ODBOT can be imported"""
    try:
        from techniques.odbot import ODBOT
        print("✓ ODBOT import successful")
        return ODBOT
    except ImportError as e:
        print(f"✗ ODBOT import failed: {e}")
        return None

def test_instantiation(ODBOT):
    """Test if ODBOT can be instantiated"""
    try:
        odbot = ODBOT(k=2, percentage=None, verbose=False)
        print("✓ ODBOT instantiation successful")
        return odbot
    except Exception as e:
        print(f"✗ ODBOT instantiation failed: {e}")
        return None

def test_basic_functionality(odbot):
    """Test basic functionality with simple data"""
    try:
        # Create simple test data
        np.random.seed(42)
        X = np.random.rand(50, 3)
        y = np.array([0]*40 + [1]*10)  # Imbalanced
        
        X_res, y_res = odbot.fit_resample(X, y)
        
        print(f"✓ Basic functionality test passed")
        print(f"  Original: {X.shape}, {np.bincount(y)}")
        print(f"  Resampled: {X_res.shape}, {np.bincount(y_res)}")
        return True
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

def test_parameter_validation(ODBOT):
    """Test parameter validation"""
    try:
        # Test invalid k parameter
        try:
            odbot = ODBOT(k=1)  # Should raise ValueError
            print("✗ Should have raised ValueError for k=1")
            return False
        except ValueError:
            print("✓ Parameter validation working (k > 1 required)")
            return True
    except Exception as e:
        print(f"✗ Parameter validation test failed: {e}")
        return False

def main():
    """Run integration tests"""
    print("ODBOT Integration Test")
    print("="*40)
    
    # Test import
    ODBOT = test_import()
    if ODBOT is None:
        return
    
    # Test parameter validation
    param_valid = test_parameter_validation(ODBOT)
    if not param_valid:
        return
    
    # Test instantiation
    odbot = test_instantiation(ODBOT)
    if odbot is None:
        return
    
    # Test basic functionality
    success = test_basic_functionality(odbot)
    
    if success:
        print("\n✓ All integration tests passed!")
        print("ODBOT is ready to use.")
    else:
        print("\n✗ Integration tests failed!")

if __name__ == "__main__":
    main()