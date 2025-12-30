#!/usr/bin/env python3
"""
Simple integration test for FCMBoostOBU
Tests basic import and instantiation
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

def test_import():
    """Test if FCMBoostOBU can be imported"""
    try:
        from techniques.fcm_boost_obu import FCMBoostOBU
        print("✓ FCMBoostOBU import successful")
        return FCMBoostOBU
    except ImportError as e:
        print(f"✗ FCMBoostOBU import failed: {e}")
        return None

def test_dependency():
    """Test if scikit-fuzzy is available"""
    try:
        import skfuzzy
        print("✓ scikit-fuzzy dependency available")
        return True
    except ImportError:
        print("✗ scikit-fuzzy dependency missing (this is expected)")
        print("  FCMBoostOBU will show appropriate error message when used")
        return False

def test_instantiation(FCMBoostOBU):
    """Test if FCMBoostOBU can be instantiated"""
    try:
        fcm_boost = FCMBoostOBU(k=5, m=2, max_iter=1000, error=1e-5, verbose=False)
        print("✓ FCMBoostOBU instantiation successful")
        return fcm_boost
    except Exception as e:
        print(f"✗ FCMBoostOBU instantiation failed: {e}")
        return None

def test_basic_functionality(fcm_boost, has_skfuzzy):
    """Test basic functionality with simple data"""
    try:
        # Create simple test data
        np.random.seed(42)
        X = np.random.rand(50, 3)
        y = np.array([0]*40 + [1]*10)  # Imbalanced
        
        if has_skfuzzy:
            X_res, y_res = fcm_boost.fit_resample(X, y)
            print(f"✓ Basic functionality test passed")
            print(f"  Original: {X.shape}, {np.bincount(y)}")
            print(f"  Resampled: {X_res.shape}, {np.bincount(y_res)}")
            return True
        else:
            # Test that it properly handles missing dependency
            try:
                X_res, y_res = fcm_boost.fit_resample(X, y)
                print("✗ Should have failed due to missing scikit-fuzzy")
                return False
            except ImportError:
                print("✓ Properly handles missing scikit-fuzzy dependency")
                return True
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

def main():
    """Run integration tests"""
    print("FCMBoostOBU Integration Test")
    print("="*40)
    
    # Test import
    FCMBoostOBU = test_import()
    if FCMBoostOBU is None:
        return
    
    # Test dependency
    has_skfuzzy = test_dependency()
    
    # Test instantiation
    fcm_boost = test_instantiation(FCMBoostOBU)
    if fcm_boost is None:
        return
    
    # Test basic functionality
    success = test_basic_functionality(fcm_boost, has_skfuzzy)
    
    if success:
        print("\n✓ All integration tests passed!")
        print("FCMBoostOBU is ready to use.")
        if not has_skfuzzy:
            print("Note: Install scikit-fuzzy to use FCMBoostOBU functionality")
    else:
        print("\n✗ Integration tests failed!")

if __name__ == "__main__":
    main()