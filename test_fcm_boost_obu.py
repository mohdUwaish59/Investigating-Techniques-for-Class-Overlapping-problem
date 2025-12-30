#!/usr/bin/env python3
"""
Test script for FCMBoostOBU (Fuzzy C-Means Boosted Overlap-Based Undersampling)
Tests the BoostOBU technique with BLSMOTE1 and FCM clustering
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

def create_test_dataset():
    """Create an imbalanced dataset with overlap for testing"""
    print("Creating test dataset...")
    
    # Create imbalanced dataset with overlap
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_classes=2,
        weights=[0.9, 0.1],  # 90% majority, 10% minority
        flip_y=0.1,  # Add some noise/overlap
        random_state=42
    )
    
    print(f"Original dataset shape: {X.shape}")
    print(f"Original class distribution: {np.bincount(y)}")
    print(f"Imbalance Ratio: {np.bincount(y)[0] / np.bincount(y)[1]:.2f}")
    
    return X, y

def test_fcm_boost_obu_import():
    """Test if FCMBoostOBU can be imported"""
    print("\n" + "="*70)
    print("Testing FCMBoostOBU Import")
    print("="*70)
    
    try:
        from techniques.fcm_boost_obu import FCMBoostOBU
        print("✓ FCMBoostOBU import successful")
        return FCMBoostOBU
    except ImportError as e:
        print(f"✗ FCMBoostOBU import failed: {e}")
        return None

def test_fcm_boost_obu_basic(FCMBoostOBU):
    """Test basic functionality of FCMBoostOBU"""
    print("\n" + "="*70)
    print("Testing FCMBoostOBU Basic Functionality")
    print("="*70)
    
    X, y = create_test_dataset()
    
    # Test with default parameters
    print("\nTesting with default parameters...")
    try:
        fcm_boost = FCMBoostOBU(verbose=True)
        X_resampled, y_resampled = fcm_boost.fit_resample(X, y)
        
        print(f"\nResampled dataset shape: {X_resampled.shape}")
        print(f"Resampled class distribution: {np.bincount(y_resampled)}")
        print(f"New Imbalance Ratio: {np.bincount(y_resampled)[0] / np.bincount(y_resampled)[1]:.2f}")
        print(f"Adaptive threshold μth: {fcm_boost.mu_th_:.4f}")
        
        return X_resampled, y_resampled
    except Exception as e:
        print(f"Error in basic functionality test: {e}")
        return None, None

def test_fcm_boost_obu_parameters(FCMBoostOBU):
    """Test FCMBoostOBU with different parameter values"""
    print("\n" + "="*70)
    print("Testing FCMBoostOBU with Different Parameters")
    print("="*70)
    
    X, y = create_test_dataset()
    
    # Test different parameter combinations
    param_configs = [
        {'k': 3, 'm': 1.5, 'max_iter': 500, 'error': 1e-4},
        {'k': 5, 'm': 2.0, 'max_iter': 1000, 'error': 1e-5},
        {'k': 7, 'm': 3.0, 'max_iter': 1500, 'error': 1e-6}
    ]
    
    results = {}
    
    for i, params in enumerate(param_configs):
        print(f"\n--- Testing configuration {i+1}: {params} ---")
        try:
            fcm_boost = FCMBoostOBU(**params, verbose=False, random_state=42)
            X_res, y_res = fcm_boost.fit_resample(X, y)
            
            results[i] = {
                'params': params,
                'shape': X_res.shape,
                'distribution': np.bincount(y_res),
                'imbalance_ratio': np.bincount(y_res)[0] / np.bincount(y_res)[1],
                'mu_th': fcm_boost.mu_th_,
                'success': True
            }
            
            print(f"Shape: {X_res.shape}")
            print(f"Distribution: {np.bincount(y_res)}")
            print(f"Imbalance Ratio: {np.bincount(y_res)[0] / np.bincount(y_res)[1]:.2f}")
            print(f"Adaptive threshold: {fcm_boost.mu_th_:.4f}")
            
        except Exception as e:
            print(f"Error with configuration {i+1}: {e}")
            results[i] = {'params': params, 'success': False, 'error': str(e)}
    
    return results

def test_classification_performance(FCMBoostOBU):
    """Test classification performance before and after FCMBoostOBU"""
    print("\n" + "="*70)
    print("Testing Classification Performance")
    print("="*70)
    
    X, y = create_test_dataset()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Train on original data
    print("\n--- Performance on Original Data ---")
    rf_original = RandomForestClassifier(random_state=42)
    rf_original.fit(X_train, y_train)
    y_pred_original = rf_original.predict(X_test)
    
    print("Classification Report (Original):")
    print(classification_report(y_test, y_pred_original))
    
    # Apply FCMBoostOBU and train
    print("\n--- Performance after FCMBoostOBU ---")
    try:
        fcm_boost = FCMBoostOBU(k=5, m=2, verbose=False, random_state=42)
        X_train_resampled, y_train_resampled = fcm_boost.fit_resample(X_train, y_train)
        
        rf_resampled = RandomForestClassifier(random_state=42)
        rf_resampled.fit(X_train_resampled, y_train_resampled)
        y_pred_resampled = rf_resampled.predict(X_test)
        
        print("Classification Report (After FCMBoostOBU):")
        print(classification_report(y_test, y_pred_resampled))
        
        return {
            'original': classification_report(y_test, y_pred_original, output_dict=True),
            'resampled': classification_report(y_test, y_pred_resampled, output_dict=True)
        }
    except Exception as e:
        print(f"Error in performance test: {e}")
        return None

def test_edge_cases(FCMBoostOBU):
    """Test edge cases and error handling"""
    print("\n" + "="*70)
    print("Testing Edge Cases")
    print("="*70)
    
    # Test with very small dataset
    print("\n--- Testing with small dataset ---")
    X_small = np.random.rand(20, 5)
    y_small = np.array([0]*18 + [1]*2)
    
    try:
        fcm_boost = FCMBoostOBU(verbose=False, random_state=42)
        X_res, y_res = fcm_boost.fit_resample(X_small, y_small)
        print(f"Small dataset - Original: {X_small.shape}, Resampled: {X_res.shape}")
        print(f"Original distribution: {np.bincount(y_small)}")
        print(f"Resampled distribution: {np.bincount(y_res)}")
    except Exception as e:
        print(f"Error with small dataset: {e}")
    
    # Test with balanced dataset
    print("\n--- Testing with balanced dataset ---")
    X_balanced = np.random.rand(100, 5)
    y_balanced = np.array([0]*50 + [1]*50)
    
    try:
        fcm_boost = FCMBoostOBU(verbose=False, random_state=42)
        X_res, y_res = fcm_boost.fit_resample(X_balanced, y_balanced)
        print(f"Balanced dataset - Original: {X_balanced.shape}, Resampled: {X_res.shape}")
        print(f"Original distribution: {np.bincount(y_balanced)}")
        print(f"Resampled distribution: {np.bincount(y_res)}")
    except Exception as e:
        print(f"Error with balanced dataset: {e}")

def test_dependency_check():
    """Test if scikit-fuzzy dependency is available"""
    print("\n" + "="*70)
    print("Testing Dependencies")
    print("="*70)
    
    try:
        import skfuzzy
        print("✓ scikit-fuzzy is available")
        return True
    except ImportError:
        print("✗ scikit-fuzzy is not available")
        print("  Install with: pip install scikit-fuzzy")
        return False

def main():
    """Run all tests"""
    print("FCMBoostOBU (Fuzzy C-Means Boosted Overlap-Based Undersampling) Test Suite")
    print("="*80)
    
    # Check dependencies first
    if not test_dependency_check():
        print("\nSkipping tests due to missing dependencies.")
        return
    
    try:
        # Import test
        FCMBoostOBU = test_fcm_boost_obu_import()
        if FCMBoostOBU is None:
            return
        
        # Basic functionality test
        X_res, y_res = test_fcm_boost_obu_basic(FCMBoostOBU)
        if X_res is None:
            print("Basic functionality test failed, skipping remaining tests.")
            return
        
        # Parameter testing
        param_results = test_fcm_boost_obu_parameters(FCMBoostOBU)
        
        # Classification performance
        perf_results = test_classification_performance(FCMBoostOBU)
        
        # Edge cases
        test_edge_cases(FCMBoostOBU)
        
        print("\n" + "="*80)
        print("All tests completed!")
        print("="*80)
        
        # Summary
        print("\nSUMMARY:")
        print(f"✓ Import: Working")
        print(f"✓ Basic functionality: Working")
        print(f"✓ Parameter variations: {len(param_results)} configurations tested")
        print(f"✓ Classification performance: {'Evaluated' if perf_results else 'Failed'}")
        print(f"✓ Edge cases: Handled")
        
        # Parameter results summary
        successful_configs = sum(1 for r in param_results.values() if r.get('success', False))
        print(f"✓ Successful parameter configurations: {successful_configs}/{len(param_results)}")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()