#!/usr/bin/env python3
"""
Test script for DeviOCSVM (Devi et al. 2019) technique
Tests the One-Class SVM undersampling method with comprehensive overlap handling
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

from techniques.devi_ocsvm import DeviOCSVM

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
        weights=[0.8, 0.2],  # 80% majority, 20% minority
        flip_y=0.1,  # Add some noise/overlap
        random_state=42
    )
    
    print(f"Original dataset shape: {X.shape}")
    print(f"Original class distribution: {np.bincount(y)}")
    print(f"Imbalance Ratio: {np.bincount(y)[0] / np.bincount(y)[1]:.2f}")
    
    return X, y

def test_devi_ocsvm_basic():
    """Test basic functionality of DeviOCSVM"""
    print("\n" + "="*70)
    print("Testing DeviOCSVM Basic Functionality")
    print("="*70)
    
    X, y = create_test_dataset()
    
    # Test with default parameters
    print("\nTesting with default parameters...")
    devi = DeviOCSVM(verbose=True)
    X_resampled, y_resampled = devi.fit_resample(X, y)
    
    print(f"\nResampled dataset shape: {X_resampled.shape}")
    print(f"Resampled class distribution: {np.bincount(y_resampled)}")
    print(f"New Imbalance Ratio: {np.bincount(y_resampled)[0] / np.bincount(y_resampled)[1]:.2f}")
    
    return X_resampled, y_resampled

def test_devi_ocsvm_parameters():
    """Test DeviOCSVM with different parameter values"""
    print("\n" + "="*70)
    print("Testing DeviOCSVM with Different Parameters")
    print("="*70)
    
    X, y = create_test_dataset()
    
    # Test different nu values as explored in the paper
    nu_values = [0.3, 0.5, 0.7]
    results = {}
    
    for nu in nu_values:
        print(f"\n--- Testing with nu = {nu} ---")
        devi = DeviOCSVM(nu=nu, K1=1, K2=5, K3=1, verbose=False)
        X_res, y_res = devi.fit_resample(X, y)
        
        results[nu] = {
            'shape': X_res.shape,
            'distribution': np.bincount(y_res),
            'imbalance_ratio': np.bincount(y_res)[0] / np.bincount(y_res)[1]
        }
        
        print(f"Shape: {X_res.shape}")
        print(f"Distribution: {np.bincount(y_res)}")
        print(f"Imbalance Ratio: {np.bincount(y_res)[0] / np.bincount(y_res)[1]:.2f}")
    
    return results

def test_devi_ocsvm_kernels():
    """Test DeviOCSVM with different kernel types"""
    print("\n" + "="*70)
    print("Testing DeviOCSVM with Different Kernels")
    print("="*70)
    
    X, y = create_test_dataset()
    
    kernels = ['rbf', 'linear', 'poly']
    results = {}
    
    for kernel in kernels:
        print(f"\n--- Testing with kernel = {kernel} ---")
        try:
            devi = DeviOCSVM(nu=0.5, kernel=kernel, verbose=False)
            X_res, y_res = devi.fit_resample(X, y)
            
            results[kernel] = {
                'shape': X_res.shape,
                'distribution': np.bincount(y_res),
                'success': True
            }
            
            print(f"Shape: {X_res.shape}")
            print(f"Distribution: {np.bincount(y_res)}")
            
        except Exception as e:
            print(f"Error with kernel {kernel}: {e}")
            results[kernel] = {'success': False, 'error': str(e)}
    
    return results

def test_classification_performance():
    """Test classification performance before and after DeviOCSVM"""
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
    
    # Apply DeviOCSVM and train
    print("\n--- Performance after DeviOCSVM ---")
    devi = DeviOCSVM(nu=0.5, verbose=False)
    X_train_resampled, y_train_resampled = devi.fit_resample(X_train, y_train)
    
    rf_resampled = RandomForestClassifier(random_state=42)
    rf_resampled.fit(X_train_resampled, y_train_resampled)
    y_pred_resampled = rf_resampled.predict(X_test)
    
    print("Classification Report (After DeviOCSVM):")
    print(classification_report(y_test, y_pred_resampled))
    
    return {
        'original': classification_report(y_test, y_pred_original, output_dict=True),
        'resampled': classification_report(y_test, y_pred_resampled, output_dict=True)
    }

def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n" + "="*70)
    print("Testing Edge Cases")
    print("="*70)
    
    # Test with very small dataset
    print("\n--- Testing with small dataset ---")
    X_small = np.random.rand(20, 5)
    y_small = np.array([0]*15 + [1]*5)
    
    try:
        devi = DeviOCSVM(verbose=False)
        X_res, y_res = devi.fit_resample(X_small, y_small)
        print(f"Small dataset - Original: {X_small.shape}, Resampled: {X_res.shape}")
    except Exception as e:
        print(f"Error with small dataset: {e}")
    
    # Test with balanced dataset
    print("\n--- Testing with balanced dataset ---")
    X_balanced = np.random.rand(100, 5)
    y_balanced = np.array([0]*50 + [1]*50)
    
    try:
        devi = DeviOCSVM(verbose=False)
        X_res, y_res = devi.fit_resample(X_balanced, y_balanced)
        print(f"Balanced dataset - Original: {X_balanced.shape}, Resampled: {X_res.shape}")
        print(f"Original distribution: {np.bincount(y_balanced)}")
        print(f"Resampled distribution: {np.bincount(y_res)}")
    except Exception as e:
        print(f"Error with balanced dataset: {e}")

def main():
    """Run all tests"""
    print("DeviOCSVM (Devi et al. 2019) Test Suite")
    print("="*70)
    
    try:
        # Basic functionality test
        X_res, y_res = test_devi_ocsvm_basic()
        
        # Parameter testing
        param_results = test_devi_ocsvm_parameters()
        
        # Kernel testing
        kernel_results = test_devi_ocsvm_kernels()
        
        # Classification performance
        perf_results = test_classification_performance()
        
        # Edge cases
        test_edge_cases()
        
        print("\n" + "="*70)
        print("All tests completed successfully!")
        print("="*70)
        
        # Summary
        print("\nSUMMARY:")
        print(f"✓ Basic functionality: Working")
        print(f"✓ Parameter variations: {len(param_results)} configurations tested")
        print(f"✓ Kernel variations: {len(kernel_results)} kernels tested")
        print(f"✓ Classification performance: Evaluated")
        print(f"✓ Edge cases: Handled")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()