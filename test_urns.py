"""
Test script for URNS (Undersampling based on Recursive Neighbourhood Search)
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'src'))

import pandas as pd
import numpy as np
from techniques import URNS
from data_loader import DataLoader

print("="*70)
print("Testing URNS (Recursive Neighbourhood Search)")
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

# Test 1: URNS with adaptive k and 2 rounds (default)
print("\n" + "="*70)
print("Test 1: URNS with Adaptive k and 2 Rounds")
print("="*70)

urns_adaptive = URNS(k='adaptive', min_frequency=2, rounds=2, verbose=True)
X_res1, y_res1 = urns_adaptive.fit_resample(X, y)

print(f"\nResults:")
print(f"  Samples removed: {len(X) - len(X_res1)}")
print(f"  Final samples: {len(X_res1)}")
print(f"  Final class distribution: {np.bincount(y_res1)}")
print(f"  Final IR: {np.sum(y_res1 == 0) / np.sum(y_res1 == 1):.2f}")

# Test 2: URNS with manual k=10
print("\n" + "="*70)
print("Test 2: URNS with Manual k=10")
print("="*70)

urns_manual = URNS(k=10, min_frequency=2, rounds=2, verbose=True)
X_res2, y_res2 = urns_manual.fit_resample(X, y)

print(f"\nResults:")
print(f"  Samples removed: {len(X) - len(X_res2)}")
print(f"  Final samples: {len(X_res2)}")
print(f"  Final class distribution: {np.bincount(y_res2)}")
print(f"  Final IR: {np.sum(y_res2 == 0) / np.sum(y_res2 == 1):.2f}")

# Test 3: URNS with 1 round only
print("\n" + "="*70)
print("Test 3: URNS with 1 Round Only")
print("="*70)

urns_1round = URNS(k='adaptive', min_frequency=2, rounds=1, verbose=True)
X_res3, y_res3 = urns_1round.fit_resample(X, y)

print(f"\nResults:")
print(f"  Samples removed: {len(X) - len(X_res3)}")
print(f"  Final samples: {len(X_res3)}")
print(f"  Final class distribution: {np.bincount(y_res3)}")
print(f"  Final IR: {np.sum(y_res3 == 0) / np.sum(y_res3 == 1):.2f}")

# Test 4: URNS with different min_frequency
print("\n" + "="*70)
print("Test 4: URNS with Min Frequency = 3")
print("="*70)

urns_freq3 = URNS(k='adaptive', min_frequency=3, rounds=2, verbose=True)
X_res4, y_res4 = urns_freq3.fit_resample(X, y)

print(f"\nResults:")
print(f"  Samples removed: {len(X) - len(X_res4)}")
print(f"  Final samples: {len(X_res4)}")
print(f"  Final class distribution: {np.bincount(y_res4)}")
print(f"  Final IR: {np.sum(y_res4 == 0) / np.sum(y_res4 == 1):.2f}")

# Summary comparison
print("\n" + "="*70)
print("SUMMARY COMPARISON")
print("="*70)

print(f"\n{'Configuration':<30} {'k value':<10} {'Removed':<10} {'Final':<10} {'Final IR':<10}")
print("-"*70)

print(f"{'Original':<30} {'-':<10} {'-':<10} {len(X):<10} {np.sum(y == 0) / np.sum(y == 1):<10.2f}")

print(f"{'Adaptive k, 2 rounds':<30} {urns_adaptive.k_value_:<10} "
      f"{len(X) - len(X_res1):<10} {len(X_res1):<10} "
      f"{np.sum(y_res1 == 0) / np.sum(y_res1 == 1):<10.2f}")

print(f"{'Manual k=10, 2 rounds':<30} {10:<10} "
      f"{len(X) - len(X_res2):<10} {len(X_res2):<10} "
      f"{np.sum(y_res2 == 0) / np.sum(y_res2 == 1):<10.2f}")

print(f"{'Adaptive k, 1 round':<30} {urns_1round.k_value_:<10} "
      f"{len(X) - len(X_res3):<10} {len(X_res3):<10} "
      f"{np.sum(y_res3 == 0) / np.sum(y_res3 == 1):<10.2f}")

print(f"{'Adaptive k, min_freq=3':<30} {urns_freq3.k_value_:<10} "
      f"{len(X) - len(X_res4):<10} {len(X_res4):<10} "
      f"{np.sum(y_res4 == 0) / np.sum(y_res4 == 1):<10.2f}")

# Statistics comparison
print("\n" + "="*70)
print("DETAILED STATISTICS")
print("="*70)

configs = [
    ("Adaptive k, 2 rounds", urns_adaptive),
    ("Manual k=10, 2 rounds", urns_manual),
    ("Adaptive k, 1 round", urns_1round),
    ("Adaptive k, min_freq=3", urns_freq3)
]

for name, urns_obj in configs:
    print(f"\n{name}:")
    print(f"  k value: {urns_obj.k_value_}")
    if 'round_1_removed' in urns_obj.stats_:
        print(f"  Round 1 removed: {urns_obj.stats_['round_1_removed']}")
    if 'round_2_removed' in urns_obj.stats_:
        print(f"  Round 2 removed: {urns_obj.stats_['round_2_removed']}")
    print(f"  Total removed: {urns_obj.stats_['total_removed']}")
    print(f"  Final majority: {urns_obj.stats_['final_majority']}")
    print(f"  Final minority: {urns_obj.stats_['final_minority']}")

print("\n✅ URNS test completed successfully!")
print("\nKey Insights:")
print("  - Adaptive k adjusts based on dataset size and imbalance ratio")
print("  - More rounds = more aggressive removal")
print("  - Higher min_frequency = less aggressive removal")
print("  - URNS specifically targets overlapping majority class instances")
