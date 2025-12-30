"""
Test script for Complexity Measures (N3 and T1)
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'src'))

import pandas as pd
import numpy as np
from complexity_measures import ComplexityMeasures, compare_pre_post_overlap
from data_loader import DataLoader
from techniques import RFCL, OSM

print("="*70)
print("Testing Complexity Measures (N3 & T1)")
print("="*70)

# Load the imbalanced dataset
df = pd.read_csv('data/data_imbalanced.csv')

# Initialize data loader
loader = DataLoader(standardize=True, random_state=42)
X, y = loader.load_from_dataframe(df, target_column='stabf')

print(f"\nOriginal dataset:")
print(f"  Total samples: {len(X)}")
print(f"  Class distribution: {np.bincount(y)}")
print(f"  Imbalance Ratio: {np.sum(y == 0) / np.sum(y == 1):.2f}")

# Calculate complexity measures for original data
print("\n" + "="*70)
print("ORIGINAL DATA COMPLEXITY ANALYSIS")
print("="*70)

cm_original = ComplexityMeasures(X, y)
results_original = cm_original.analyze_overlap(by_class=True)

print("\n1. N3 (Instance Overlap) Results:")
print(f"   Overall N3: {results_original['n3']['overall']:.4f}")
print(f"   Misclassified: {results_original['n3']['misclassified_count']} / {len(X)}")
if 'by_class' in results_original['n3']:
    print("   By class:")
    for cls, value in results_original['n3']['by_class'].items():
        print(f"      Class {cls}: {value:.4f}")

print("\n2. T1 (Structural Overlap) Results:")
print(f"   Overall T1: {results_original['t1']['overall']} hyperspheres")
print(f"   Normalized T1: {results_original['t1']['normalized']:.4f}")
if 'by_class' in results_original['t1']:
    print("   By class:")
    for cls, value in results_original['t1']['by_class'].items():
        print(f"      Class {cls}: {value['count']} hyperspheres (normalized: {value['normalized']:.4f})")

print("\n3. Interpretation:")
print(f"   Overall Complexity: {results_original['interpretation']['overall_complexity'].upper()}")
print(f"   Instance Overlap: {results_original['interpretation']['instance_overlap'].upper()}")
print(f"   Structural Overlap: {results_original['interpretation']['structural_overlap'].upper()}")

if results_original['interpretation']['recommendations']:
    print("\n   Recommendations:")
    for i, rec in enumerate(results_original['interpretation']['recommendations'], 1):
        print(f"      {i}. {rec}")

# Test with RFCL
print("\n" + "="*70)
print("TESTING WITH T1: RFCL")
print("="*70)

rfcl = RFCL(random_state=42, verbose=False)
X_rfcl, y_rfcl = rfcl.fit_resample(X, y)

print(f"\nAfter RFCL:")
print(f"  Samples: {len(X)} → {len(X_rfcl)}")
print(f"  Class distribution: {np.bincount(y_rfcl)}")

# Compare complexity
comparison_rfcl = compare_pre_post_overlap(X, y, X_rfcl, y_rfcl)

print("\nComplexity Improvements:")
print(f"  N3: {comparison_rfcl['pre_processing']['n3']['overall']:.4f} → "
      f"{comparison_rfcl['post_processing']['n3']['overall']:.4f} "
      f"({comparison_rfcl['improvements']['n3']['absolute']:+.4f}, "
      f"{comparison_rfcl['improvements']['n3']['relative']:+.2f}%)")

print(f"  T1: {comparison_rfcl['pre_processing']['t1']['normalized']:.4f} → "
      f"{comparison_rfcl['post_processing']['t1']['normalized']:.4f} "
      f"({comparison_rfcl['improvements']['t1']['absolute']:+.4f}, "
      f"{comparison_rfcl['improvements']['t1']['relative']:+.2f}%)")

# Test with OSM
print("\n" + "="*70)
print("TESTING WITH T6: OSM")
print("="*70)

osm = OSM(n_clusters=2, verbose=False)
X_osm, y_osm = osm.fit_resample(X, y)

print(f"\nAfter OSM:")
print(f"  Samples: {len(X)} → {len(X_osm)}")
print(f"  Class distribution: {np.bincount(y_osm)}")

# Compare complexity
comparison_osm = compare_pre_post_overlap(X, y, X_osm, y_osm)

print("\nComplexity Improvements:")
print(f"  N3: {comparison_osm['pre_processing']['n3']['overall']:.4f} → "
      f"{comparison_osm['post_processing']['n3']['overall']:.4f} "
      f"({comparison_osm['improvements']['n3']['absolute']:+.4f}, "
      f"{comparison_osm['improvements']['n3']['relative']:+.2f}%)")

print(f"  T1: {comparison_osm['pre_processing']['t1']['normalized']:.4f} → "
      f"{comparison_osm['post_processing']['t1']['normalized']:.4f} "
      f"({comparison_osm['improvements']['t1']['absolute']:+.4f}, "
      f"{comparison_osm['improvements']['t1']['relative']:+.2f}%)")

# Summary comparison
print("\n" + "="*70)
print("SUMMARY COMPARISON")
print("="*70)

print(f"\n{'Technique':<15} {'N3 Before':<12} {'N3 After':<12} {'N3 Change':<12} {'T1 Before':<12} {'T1 After':<12} {'T1 Change':<12}")
print("-"*90)

print(f"{'Original':<15} {results_original['n3']['overall']:<12.4f} {'-':<12} {'-':<12} "
      f"{results_original['t1']['normalized']:<12.4f} {'-':<12} {'-':<12}")

print(f"{'RFCL':<15} {comparison_rfcl['pre_processing']['n3']['overall']:<12.4f} "
      f"{comparison_rfcl['post_processing']['n3']['overall']:<12.4f} "
      f"{comparison_rfcl['improvements']['n3']['absolute']:<12.4f} "
      f"{comparison_rfcl['pre_processing']['t1']['normalized']:<12.4f} "
      f"{comparison_rfcl['post_processing']['t1']['normalized']:<12.4f} "
      f"{comparison_rfcl['improvements']['t1']['absolute']:<12.4f}")

print(f"{'OSM':<15} {comparison_osm['pre_processing']['n3']['overall']:<12.4f} "
      f"{comparison_osm['post_processing']['n3']['overall']:<12.4f} "
      f"{comparison_osm['improvements']['n3']['absolute']:<12.4f} "
      f"{comparison_osm['pre_processing']['t1']['normalized']:<12.4f} "
      f"{comparison_osm['post_processing']['t1']['normalized']:<12.4f} "
      f"{comparison_osm['improvements']['t1']['absolute']:<12.4f}")

print("\n✅ Complexity measures test completed successfully!")
print("\nNote: Positive changes indicate improvement (reduced overlap)")
