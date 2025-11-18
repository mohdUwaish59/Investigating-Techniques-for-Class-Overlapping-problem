"""
Test script for OSM (Overlap-Separating Model) implementation
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'src'))

import pandas as pd
import numpy as np
from techniques import OSM
from data_loader import DataLoader

print("="*70)
print("Testing OSM (Overlap-Separating Model) Implementation")
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

# Test OSM with default parameters
print("\n" + "="*70)
print("Testing OSM with Default Parameters")
print("="*70)

osm = OSM(
    n_clusters=2,
    n_features=None,  # Auto-select
    outlier_removal=True,
    svm_optimization=True,
    rose_sampling=True,
    tomek_removal=True,
    feature_selection=True,
    overlap_threshold=0.3,
    random_state=42,
    verbose=True
)

X_resampled, y_resampled = osm.fit_resample(X, y)

print(f"\n{'='*70}")
print("RESULTS")
print(f"{'='*70}")
print(f"Original samples: {len(X)}")
print(f"Resampled samples: {len(X_resampled)}")
print(f"Samples removed: {len(X) - len(X_resampled)}")
print(f"\nOriginal features: {X.shape[1]}")
print(f"Selected features: {X_resampled.shape[1]}")
print(f"\nOriginal class distribution: {np.bincount(y)}")
print(f"Resampled class distribution: {np.bincount(y_resampled)}")
print(f"\nOriginal IR: {np.sum(y == 0) / np.sum(y == 1):.2f}")
print(f"Resampled IR: {np.sum(y_resampled == 0) / np.sum(y_resampled == 1):.2f}")

# Print statistics
print(f"\n{'='*70}")
print("PREPROCESSING STATISTICS")
print(f"{'='*70}")
for key, value in osm.stats_.items():
    print(f"{key}: {value}")

# Test with different configurations
print("\n" + "="*70)
print("Testing OSM with Minimal Pipeline")
print("="*70)

osm_minimal = OSM(
    n_clusters=3,
    rose_sampling=False,
    tomek_removal=False,
    feature_selection=False,
    outlier_removal=False,
    svm_optimization=False,
    overlap_threshold=0.5,
    random_state=42,
    verbose=True
)

X_minimal, y_minimal = osm_minimal.fit_resample(X, y)

print(f"\nMinimal pipeline results:")
print(f"  Samples: {len(X)} → {len(X_minimal)}")
print(f"  Features: {X.shape[1]} → {X_minimal.shape[1]}")
print(f"  Class distribution: {np.bincount(y_minimal)}")

print("\n✅ OSM test completed successfully!")
