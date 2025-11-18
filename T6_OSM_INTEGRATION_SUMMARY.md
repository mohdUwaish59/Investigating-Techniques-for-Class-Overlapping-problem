# ✅ T6: OSM Successfully Integrated!

## Overview
**T6: OSM (Overlap-Separating Model)** is now fully integrated into your imbalanced learning system. This is a comprehensive preprocessing and resampling technique based on the paper "Optimising Prediction in Overlapping and Non-Overlapping Regions" by Sumana B.V. and Punithavalli M. (2020).

## What Was Done

### 1. Created OSM Implementation (`src/techniques/osm.py`)
A complete implementation with three phases:

**Phase 1: Balancing Phase**
- Missing value handling (KNN imputation)
- Min-Max normalization
- ROSE balancing (SMOTE + Random Undersampling)
- Tomek link removal
- Random Forest feature selection
- Boxplot outlier removal (IQR method)

**Phase 2: Overlap Separation Phase**
- K-means clustering to separate overlap/non-overlap regions
- SVM optimization to remove misclassified instances

**Phase 3: Result Combination**
- Combines processed overlap and non-overlap regions

### 2. System Integration
- ✅ Added to `src/techniques/__init__.py`
- ✅ Added to `src/resampling_techniques.py`
- ✅ Updated `app.py` with comprehensive UI controls
- ✅ Updated `src/run.py`
- ✅ Updated `README.md`
- ✅ Created `test_osm.py`

## UI Features

When user selects "T6: OSM" in the Streamlit app:

### Parameters Available:
1. **Number of clusters (K-means)**: Slider (2-10, default=2)
   - Controls K-means clustering for overlap separation

2. **Number of features to select**: Slider (2-20, default=6)
   - Features to keep after Random Forest selection

3. **Overlap threshold**: Slider (0.1-0.9, default=0.3)
   - Lower = more samples classified as overlap region
   - Higher = more samples classified as non-overlap region

### Pipeline Step Toggles:
Users can enable/disable each preprocessing step:

**Column 1:**
- ☑️ **ROSE balancing** (SMOTE + Random Undersampling)
- ☑️ **Tomek link removal** (Remove borderline examples)
- ☑️ **Feature selection** (Random Forest importance)

**Column 2:**
- ☑️ **Outlier removal** (Boxplot IQR method)
- ☑️ **SVM optimization** (Remove misclassified samples)
- ☑️ **Show detailed progress** (Verbose output)

## How OSM Works

### The Pipeline:
```
Original Data
    ↓
[1] Handle Missing Values (KNN)
    ↓
[2] Normalize (Min-Max to [0,1])
    ↓
[3] Balance with ROSE (SMOTE + Undersampling)
    ↓
[4] Remove Tomek Links
    ↓
[5] Select Features (Random Forest)
    ↓
[6] Remove Outliers (Boxplot IQR)
    ↓
[7] Separate Overlap/Non-Overlap (K-means)
    ↓
[8] Optimize Non-Overlap (SVM)
    ↓
Resampled Data
```

### Key Concepts:

**Overlap Region:**
- Samples where both classes are mixed
- Identified by K-means clustering
- Samples close to decision boundaries
- More difficult to classify

**Non-Overlap Region:**
- Samples in pure class clusters
- Far from decision boundaries
- Easier to classify
- Optimized with SVM to remove misclassified instances

## Testing

Run the test script:
```bash
python test_osm.py
```

This will:
1. Test OSM with default (full pipeline) parameters
2. Test OSM with minimal pipeline (only overlap separation)
3. Show detailed statistics for each phase
4. Compare before/after class distributions

## Your System Now Has

**8 Techniques** with **16+ Total Variants**:

1. **T1: RFCL** (1 variant)
   - Random Forest Cleaning Rule

2. **T2: SVDDWSMOTE** (1 variant)
   - SVDD-based overlap handler

3. **T4: NBUS** (4 variants)
   - NB-Basic, NB-Tomek, NB-Comm, NB-Rec

4. **T5: KMeans** (4 variants)
   - HKM, FCM, RKM, FRKM

5. **T6: OSM** (1 variant) ⭐ **NEW**
   - Comprehensive preprocessing pipeline
   - Highly configurable (6 toggleable steps)

6. **EHSO** (1 variant)
   - Evolutionary Hybrid Sampling

7. **ROS** (1 variant)
   - Random Oversampling

8. **RUS** (1 variant)
   - Random Undersampling

## Advantages of OSM

1. **Comprehensive**: Handles multiple preprocessing steps in one technique
2. **Flexible**: Each step can be toggled on/off
3. **Intelligent**: Separates overlap/non-overlap regions for targeted processing
4. **Research-Based**: Implements published methodology
5. **Feature Selection**: Automatically reduces dimensionality
6. **Outlier Handling**: Removes noisy samples
7. **Balance-Aware**: Includes ROSE balancing

## Use Cases

**Use OSM when:**
- You have significant class overlap
- You want comprehensive preprocessing
- You need feature selection
- Your data has outliers
- You want a research-backed approach
- You need fine-grained control over preprocessing steps

**OSM is especially good for:**
- High-dimensional datasets (feature selection helps)
- Noisy datasets (outlier removal helps)
- Datasets with both overlap and non-overlap regions
- When you want to understand which preprocessing steps help

## Configuration Examples

### Full Pipeline (Default):
```python
OSM(
    n_clusters=2,
    n_features=None,  # Auto-select
    rose_sampling=True,
    tomek_removal=True,
    feature_selection=True,
    outlier_removal=True,
    svm_optimization=True,
    overlap_threshold=0.3,
    random_state=42,
    verbose=True
)
```

### Minimal (Only Overlap Separation):
```python
OSM(
    n_clusters=3,
    rose_sampling=False,
    tomek_removal=False,
    feature_selection=False,
    outlier_removal=False,
    svm_optimization=False,
    overlap_threshold=0.5,
    random_state=42
)
```

### Balanced + Overlap Separation:
```python
OSM(
    n_clusters=2,
    rose_sampling=True,
    tomek_removal=True,
    feature_selection=False,
    outlier_removal=False,
    svm_optimization=True,
    overlap_threshold=0.3,
    random_state=42
)
```

## Statistics Tracked

OSM tracks detailed statistics during processing:
- `original_samples`: Starting sample count
- `original_features`: Starting feature count
- `after_rose`: Samples after ROSE balancing
- `tomek_removed`: Number of Tomek links removed
- `selected_features`: Number of features selected
- `outliers_removed`: Number of outliers removed
- `overlap_samples`: Samples in overlap region
- `non_overlap_samples`: Samples in non-overlap region
- `svm_removed`: Samples removed by SVM optimization
- `final_samples`: Final sample count
- `final_features`: Final feature count

## Integration Complete! 🎉

Your system now has one of the most comprehensive preprocessing pipelines available for imbalanced learning with class overlap. OSM combines multiple state-of-the-art techniques into a single, configurable pipeline.

The UI provides full control over every aspect of the preprocessing, making it easy to experiment and find the best configuration for your specific dataset.
