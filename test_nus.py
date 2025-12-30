"""
Test script for NUS (Neighbourhood-based Under-Sampling)
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'src'))

import pandas as pd
import numpy as np
from techniques import NUS
from data_loader import DataLoader

print("="*70)
print("Testing NUS (Neighbourhood-based Under-Sampling)")
print("="*70)

# Load the imbalanced dataset
df = pd.read_csv('data/data_imbalanced.csv')

# Initialize data loader
loader = DataLoader(standardize=True, random_state=42)
X, y = loader.load_from_dataframe(df, target_column='stabf')

print(f"\nOriginal dataset:")
print(f"  Total samples: {len(X)}")
print(f"  Features: {X.shape[1]}")
print(f"  Class distribution: {np.bincount(y)}")
print(f"  Imbalance Ratio: {np.sum(y == 0) / np.sum(y == 1):.2f}")

# Test 1: NUS with auto k and median threshold (default)
print("\n" + "="*70)
print("Test 1: NUS with Auto k and Median Threshold")
print("="*70)

nus_default = NUS(k_neighbors=None, distance_threshold='median', min_membership=2, verbose=True)
X_res1, y_res1 = nus_default.fit_resample(X, y)

print(f"\nResults:")
print(f"  k value used: {nus_default.k_value_}")
print(f"  Samples removed: {len(X) - len(X_res1)}")
print(f"  Final samples: {len(X_res1)}")
print(f"  Final class distribution: {np.bincount(y_res1)}")
print(f"  Final IR: {np.sum(y_res1 == 0) / np.sum(y_res1 == 1):.2f}")

# Test 2: NUS with manual k=15
print("\n" + "="*70)
print("Test 2: NUS with Manual k=15")
print("="*70)

nus_manual = NUS(k_neighbors=15, distance_threshold='median', min_membership=2, verbose=True)
X_res2, y_res2 = nus_manual.fit_resample(X, y)

print(f"\nResults:")
print(f"  Samples removed: {len(X) - len(X_res2)}")
print(f"  Final samples: {len(X_res2)}")
print(f"  Final class distribution: {np.bincount(y_res2)}")
print(f"  Final IR: {np.sum(y_res2 == 0) / np.sum(y_res2 == 1):.2f}")

# Test 3: NUS with mean threshold
print("\n" + "="*70)
print("Test 3: NUS with Mean Threshold")
print("="*70)

nus_mean = NUS(k_neighbors=None, distance_threshold='mean', min_membership=2, verbose=True)
X_res3, y_res3 = nus_mean.fit_resample(X, y)

print(f"\nResults:")
print(f"  Samples removed: {len(X) - len(X_res3)}")
print(f"  Final samples: {len(X_res3)}")
print(f"  Final class distribution: {np.bincount(y_res3)}")
print(f"  Final IR: {np.sum(y_res3 == 0) / np.sum(y_res3 == 1):.2f}")

# Test 4: NUS with min_membership=3
print("\n" + "="*70)
print("Test 4: NUS with Min Membership = 3")
print("="*70)

nus_mem3 = NUS(k_neighbors=None, distance_threshold='median', min_membership=3, verbose=True)
X_res4, y_res4 = nus_mem3.fit_resample(X, y)

print(f"\nResults:")
print(f"  Samples removed: {len(X) - len(X_res4)}")
print(f"  Final samples: {len(X_res4)}")
print(f"  Final class distribution: {np.bincount(y_res4)}")
print(f"  Final IR: {np.sum(y_res4 == 0) / np.sum(y_res4 == 1):.2f}")

# Test 5: NUS with min_membership=1 (most aggressive)
print("\n" + "="*70)
print("Test 5: NUS with Min Membership = 1 (Most Aggressive)")
print("="*70)

nus_mem1 = NUS(k_neighbors=None, distance_threshold='median', min_membership=1, verbose=True)
X_res5, y_res5 = nus_mem1.fit_resample(X, y)

print(f"\nResults:")
print(f"  Samples removed: {len(X) - len(X_res5)}")
print(f"  Final samples: {len(X_res5)}")
print(f"  Final class distribution: {np.bincount(y_res5)}")
print(f"  Final IR: {np.sum(y_res5 == 0) / np.sum(y_res5 == 1):.2f}")

# Summary comparison
print("\n" + "="*70)
print("SUMMARY COMPARISON")
print("="*70)

print(f"\n{'Configuration':<35} {'k':<8} {'Nominated':<12} {'Eliminated':<12} {'Final IR':<10}")
print("-"*80)

print(f"{'Original':<35} {'-':<8} {'-':<12} {'-':<12} {np.sum(y == 0) / np.sum(y == 1):<10.2f}")

configs = [
    ("Auto k, median, min_mem=2", nus_default),
    ("Manual k=15, median, min_mem=2", nus_manual),
    ("Auto k, mean, min_mem=2", nus_mean),
    ("Auto k, median, min_mem=3", nus_mem3),
    ("Auto k, median, min_mem=1", nus_mem1)
]

for name, nus_obj in configs:
    print(f"{name:<35} {nus_obj.k_value_:<8} "
          f"{nus_obj.stats_['nominated']:<12} "
          f"{nus_obj.stats_['eliminated']:<12} "
          f"{nus_obj.stats_['IR_after']:<10.2f}")

# Detailed statistics
print("\n" + "="*70)
print("DETAILED STATISTICS")
print("="*70)

for name, nus_obj in configs:
    print(f"\n{name}:")
    print(f"  k value: {nus_obj.k_value_}")
    print(f"  Nominated: {nus_obj.stats_['nominated']}")
    print(f"  Eliminated: {nus_obj.stats_['eliminated']}")
    print(f"  IR reduction: {nus_obj.stats_['IR_reduction_pct']:.2f}%")
    print(f"  Size reduction: {nus_obj.stats_['size_reduction_pct']:.2f}%")
    print(f"  Final majority: {nus_obj.stats_['final_majority']}")
    print(f"  Final minority: {nus_obj.stats_['final_minority']}")

print("\n✅ NUS test completed successfully!")
print("\nKey Insights:")
print("  - Auto k = min(sqrt(n_majority), 50)")
print("  - Median threshold: more conservative than mean")
print("  - Higher min_membership = less aggressive removal")
print("  - NUS uses colonial neighbours (k-NN from majority class)")
print("  - Only removes instances with multiple memberships")
