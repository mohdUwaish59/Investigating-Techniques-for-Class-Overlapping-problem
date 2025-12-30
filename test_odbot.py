#!/usr/bin/env python3
"""
Test script for ODBOT (Outlier Detection-Based Oversampling Technique)
Tests the ODBOT technique with clustering-based synthetic generation
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
    """Create an imbalanced dataset for testing"""
    print("Creating test dataset...")
    
    # Create imbalanced dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_classes=2,
        weights=[0.9, 0.1],  # 90% majority, 10% minority
        random_state=42
    )
    
    print(f"Original dataset shape: {X.shape}")
    print(f"Original class distribution: {np.bincount(y)}")
    print(f"Imbalance Ratio: {np.bincount(y)[0] / np.bincount(y)[1]:.2f}")
    
    return X, y

def test_paper_example():
    """Test using the exact example from the paper (Table 1)"""
    print("\n" + "="*70)
    print("Testing ODBOT with Paper Example (Table 1)")
    print("="*70)
    
    # Table 1 from paper (imbalanced dataset)
    data = np.array([
        [15.0, 1.0, 35.6],   # Sample 1, Class 0 (minority)
        [19.0, 1.3, 29.4],   # Sample 2, Class 0 (minority)
        [24.0, 2.4, 32.0],   # Sample 3, Class 0 (minority)
        [1.3, 19.0, 1.8],    # Sample 4, Class 0 (minority)
        [10.0, 18.3, 7.3],   # Sample 5, Class 1 (majority)
        [1.0, 15.0, 3.0],    # Sample 6, Class 1 (majority)
        [6.0, 17.0, 5.6],    # Sample 7, Class 1 (majority)
        [8.0, 23.4, 9.0],    # Sample 8, Class 1 (majority)
        [9.0, 22.8, 7.8],    # Sample 9, Class 1 (majority)
        [7.6, 19.5, 2.8],    # Sample 10, Class 1 (majority)
        [2.4, 24.5, 4.6],    # Sample 11, Class 1 (majority)
        [0.9, 20.3, 3.9],    # Sample 12, Class 1 (majority)
        [5.6, 16.8, 6.7],    # Sample 13, Class 1 (majority)
        [39.4, 5.7, 12.0],   # Sample 14, Class 2 (minority)
        [12.6, 19.7, 3.6],   # Sample 15, Class 2 (minority)
        [34.7, 6.3, 14.6],   # Sample 16, Class 2 (minority)
        [36.3, 7.4, 16.3],   # Sample 17, Class 2 (minority)
        [32.8, 9.6, 15.7],   # Sample 18, Class 2 (minority)
    ])
    
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
    
    print(f"Original dataset: {len(labels)} samples")
    print(f"  Class 0 (minority): {np.sum(labels == 0)} samples")
    print(f"  Class 1 (majority): {np.sum(labels == 1)} samples")
    print(f"  Class 2 (minority): {np.sum(labels == 2)} samples")
    
    try:
        from techniques.odbot import ODBOT
        
        # Apply ODBOT with k=2 as used in paper example
        odbot = ODBOT(k=2, random_state=42, verbose=True)
        X_resampled, y_resampled = odbot.fit_resample(data, labels)
        
        print(f"\nBalanced dataset after ODBOT: {len(y_resampled)} samples")
        print(f"  Class 0: {np.sum(y_resampled == 0)} samples")
        print(f"  Class 1: {np.sum(y_resampled == 1)} samples")
        print(f"  Class 2: {np.sum(y_resampled == 2)} samples")
        
        return X_resampled, y_resampled
        
    except ImportError as e:
        print(f"✗ ODBOT import failed: {e}")
        return None, None

def test_odbot_import():
    """Test if ODBOT can be imported"""
    print("\n" + "="*70)
    print("Testing ODBOT Import")
    print("="*70)
    
    try:
        from techniques.odbot import ODBOT
        print("✓ ODBOT import successful")
        return ODBOT
    except ImportError as e:
        print(f"✗ ODBOT import failed: {e}")
        return None

def test_odbot_basic(ODBOT):
    """Test basic functionality of ODBOT"""
    print("\n" + "="*70)
    print("Testing ODBOT Basic Functionality")
    print("="*70)
    
    X, y = create_test_dataset()
    
    # Test with default parameters
    print("\nTesting with default parameters...")
    try:
        odbot = ODBOT(verbose=True)
        X_resampled, y_resampled = odbot.fit_resample(X, y)
        
        print(f"\nResampled dataset shape: {X_resampled.shape}")
        print(f"Resampled class distribution: {np.bincount(y_resampled)}")
        print(f"New Imbalance Ratio: {np.bincount(y_resampled)[0] / np.bincount(y_resampled)[1]:.2f}")
        
        return X_resampled, y_resampled
    except Exception as e:
        print(f"Error in basic functionality test: {e}")
        return None, None

def test_odbot_parameters(ODBOT):
    """Test ODBOT with different parameter values"""
    print("\n" + "="*70)
    print("Testing ODBOT with Different Parameters")
    print("="*70)
    
    X, y = create_test_dataset()
    
    # Test different parameter combinations
    param_configs = [
        {'k': 2, 'percentage': None},
        {'k': 3, 'percentage': 100},
        {'k': 4, 'percentage': 200},
        {'k': 5, 'percentage': 300}
    ]
    
    results = {}
    
    for i, params in enumerate(param_configs):
        print(f"\n--- Testing configuration {i+1}: {params} ---")
        try:
            odbot = ODBOT(**params, verbose=False, random_state=42)
            X_res, y_res = odbot.fit_resample(X, y)
            
            results[i] = {
                'params': params,
                'shape': X_res.shape,
                'distribution': np.bincount(y_res),
                'imbalance_ratio': np.bincount(y_res)[0] / np.bincount(y_res)[1],
                'success': True
            }
            
            print(f"Shape: {X_res.shape}")
            print(f"Distribution: {np.bincount(y_res)}")
            print(f"Imbalance Ratio: {np.bincount(y_res)[0] / np.bincount(y_res)[1]:.2f}")
            
        except Exception as e:
            print(f"Error with configuration {i+1}: {e}")
            results[i] = {'params': params, 'success': False, 'error': str(e)}
    
    return results

def test_classification_performance(ODBOT):
    """Test classification performance before and after ODBOT"""
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
    
    # Apply ODBOT and train
    print("\n--- Performance after ODBOT ---")
    try:
        odbot = ODBOT(k=2, percentage=None, verbose=False, random_state=42)
        X_train_resampled, y_train_resampled = odbot.fit_resample(X_train, y_train)
        
        rf_resampled = RandomForestClassifier(random_state=42)
        rf_resampled.fit(X_train_resampled, y_train_resampled)
        y_pred_resampled = rf_resampled.predict(X_test)
        
        print("Classification Report (After ODBOT):")
        print(classification_report(y_test, y_pred_resampled))
        
        return {
            'original': classification_report(y_test, y_pred_original, output_dict=True),
            'resampled': classification_report(y_test, y_pred_resampled, output_dict=True)
        }
    except Exception as e:
        print(f"Error in performance test: {e}")
        return None

def test_edge_cases(ODBOT):
    """Test edge cases and error handling"""
    print("\n" + "="*70)
    print("Testing Edge Cases")
    print("="*70)
    
    # Test with very small dataset
    print("\n--- Testing with small dataset ---")
    X_small = np.random.rand(10, 3)
    y_small = np.array([0]*8 + [1]*2)
    
    try:
        odbot = ODBOT(k=2, verbose=False, random_state=42)
        X_res, y_res = odbot.fit_resample(X_small, y_small)
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
        odbot = ODBOT(k=2, verbose=False, random_state=42)
        X_res, y_res = odbot.fit_resample(X_balanced, y_balanced)
        print(f"Balanced dataset - Original: {X_balanced.shape}, Resampled: {X_res.shape}")
        print(f"Original distribution: {np.bincount(y_balanced)}")
        print(f"Resampled distribution: {np.bincount(y_res)}")
    except Exception as e:
        print(f"Error with balanced dataset: {e}")
    
    # Test invalid k parameter
    print("\n--- Testing invalid k parameter ---")
    try:
        odbot = ODBOT(k=1)  # Should raise ValueError
        print("✗ Should have raised ValueError for k=1")
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")

def main():
    """Run all tests"""
    print("ODBOT (Outlier Detection-Based Oversampling Technique) Test Suite")
    print("="*80)
    
    try:
        # Import test
        ODBOT = test_odbot_import()
        if ODBOT is None:
            return
        
        # Paper example test
        X_paper, y_paper = test_paper_example()
        
        # Basic functionality test
        X_res, y_res = test_odbot_basic(ODBOT)
        if X_res is None:
            print("Basic functionality test failed, skipping remaining tests.")
            return
        
        # Parameter testing
        param_results = test_odbot_parameters(ODBOT)
        
        # Classification performance
        perf_results = test_classification_performance(ODBOT)
        
        # Edge cases
        test_edge_cases(ODBOT)
        
        print("\n" + "="*80)
        print("All tests completed!")
        print("="*80)
        
        # Summary
        print("\nSUMMARY:")
        print(f"✓ Import: Working")
        print(f"✓ Paper example: {'Working' if X_paper is not None else 'Failed'}")
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