"""
T6: OSM - Overlap-Separating Model
Implementation of "Optimising Prediction in Overlapping and Non-Overlapping Regions"
by Sumana B.V. and Punithavalli M. (2020)

This technique handles class imbalance with class overlap using a separating scheme.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks, RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from .base_sampler import BaseSampler


class OSM(BaseSampler):
    """
    T6: Overlap-Separating Model (OSM)
    
    A comprehensive preprocessing and resampling technique that:
    1. Balances data using ROSE (SMOTE + Random Undersampling)
    2. Removes Tomek links
    3. Performs feature selection using Random Forest
    4. Removes outliers using boxplot method
    5. Separates overlap/non-overlap regions using K-means
    6. Optimizes with SVM
    
    Parameters
    ----------
    n_clusters : int, default=2
        Number of clusters for K-means separation
    
    n_features : int or None, default=None
        Number of features to select. If None, selects top 50%
    
    outlier_removal : bool, default=True
        Whether to remove outliers using boxplot method
    
    svm_optimization : bool, default=True
        Whether to optimize non-overlap region with SVM
    
    rose_sampling : bool, default=True
        Whether to apply ROSE balancing (SMOTE + undersampling)
    
    tomek_removal : bool, default=True
        Whether to remove Tomek links
    
    feature_selection : bool, default=True
        Whether to perform Random Forest feature selection
    
    overlap_threshold : float, default=0.3
        Threshold for determining overlap region (0-1)
        Lower = more samples in overlap region
    
    random_state : int, default=42
        Random seed for reproducibility
    
    verbose : bool, default=False
        Whether to print detailed progress
    """
    
    def __init__(self, n_clusters=2, n_features=None, outlier_removal=True,
                 svm_optimization=True, rose_sampling=True, tomek_removal=True,
                 feature_selection=True, overlap_threshold=0.3,
                 random_state=42, verbose=False):
        super().__init__()
        self.n_clusters = n_clusters
        self.n_features = n_features
        self.outlier_removal = outlier_removal
        self.svm_optimization = svm_optimization
        self.rose_sampling = rose_sampling
        self.tomek_removal = tomek_removal
        self.feature_selection = feature_selection
        self.overlap_threshold = overlap_threshold
        self.random_state = random_state
        self.verbose = verbose
        
        # Internal components
        self.scaler = MinMaxScaler()
        self.selected_features = None
        self.kmeans = None
        self.stats_ = {}
    
    def _log(self, message):
        """Print message if verbose"""
        if self.verbose:
            print(message)
    
    def _handle_missing_values(self, X):
        """Handle missing values using weighted KNN"""
        if np.isnan(X).any():
            self._log("Handling missing values with KNN imputer...")
            imputer = KNNImputer(n_neighbors=5, weights='distance')
            X = imputer.fit_transform(X)
        return X
    
    def _normalize_data(self, X, fit=True):
        """Normalize data using Min-Max scaling"""
        self._log("Normalizing data to [0, 1]...")
        if fit:
            X_norm = self.scaler.fit_transform(X)
        else:
            X_norm = self.scaler.transform(X)
        return X_norm
    
    def _balance_rose(self, X, y):
        """Balance data using ROSE (SMOTE + Random Undersampling)"""
        self._log(f"Balancing with ROSE - Original: {len(X)} samples")
        
        over = SMOTE(sampling_strategy='auto', random_state=self.random_state)
        under = RandomUnderSampler(sampling_strategy='auto', random_state=self.random_state)
        pipeline = ImbPipeline(steps=[('over', over), ('under', under)])
        
        X_balanced, y_balanced = pipeline.fit_resample(X, y)
        self._log(f"After ROSE: {len(X_balanced)} samples")
        self.stats_['after_rose'] = len(X_balanced)
        
        return X_balanced, y_balanced
    
    def _remove_tomek(self, X, y):
        """Remove Tomek links"""
        self._log(f"Removing Tomek links - Before: {len(X)} samples")
        
        tomek = TomekLinks(sampling_strategy='majority')
        X_clean, y_clean = tomek.fit_resample(X, y)
        
        removed = len(X) - len(X_clean)
        self._log(f"Tomek links removed: {removed}, After: {len(X_clean)} samples")
        self.stats_['tomek_removed'] = removed
        
        return X_clean, y_clean
    
    def _select_features(self, X, y):
        """Feature selection using Random Forest importance"""
        self._log(f"Selecting features - Original: {X.shape[1]} features")
        
        rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        rf.fit(X, y)
        
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        if self.n_features is None:
            n_select = max(6, X.shape[1] // 2)
        else:
            n_select = min(self.n_features, X.shape[1])
        
        self.selected_features = indices[:n_select]
        X_selected = X[:, self.selected_features]
        
        self._log(f"Selected {n_select} features")
        self.stats_['selected_features'] = n_select
        
        return X_selected
    
    def _remove_outliers(self, X, y):
        """Remove outliers using boxplot method (IQR)"""
        self._log(f"Removing outliers - Before: {len(X)} samples")
        
        Q1 = np.percentile(X, 25, axis=0)
        Q3 = np.percentile(X, 75, axis=0)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        mask = ~((X < lower_bound) | (X > upper_bound)).any(axis=1)
        X_clean = X[mask]
        y_clean = y[mask]
        
        removed = len(X) - len(X_clean)
        self._log(f"Outliers removed: {removed}, After: {len(X_clean)} samples")
        self.stats_['outliers_removed'] = removed
        
        return X_clean, y_clean
    
    def _separate_overlap(self, X, y):
        """Separate overlap and non-overlap regions using K-means"""
        self._log(f"Separating overlap/non-overlap regions with K-means (k={self.n_clusters})...")
        
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        cluster_labels = self.kmeans.fit_predict(X)
        
        overlap_mask = np.zeros(len(X), dtype=bool)
        
        for cluster_id in range(self.n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_classes = np.unique(y[cluster_mask])
            
            if len(cluster_classes) > 1:
                # Mixed cluster = overlap region
                overlap_mask[cluster_mask] = True
            else:
                # Check distance to boundary
                cluster_samples = X[cluster_mask]
                distances_to_own = np.linalg.norm(
                    cluster_samples - self.kmeans.cluster_centers_[cluster_id], axis=1
                )
                
                other_centroids = [c for i, c in enumerate(self.kmeans.cluster_centers_) if i != cluster_id]
                if other_centroids:
                    distances_to_other = np.min(
                        [np.linalg.norm(cluster_samples - c, axis=1) for c in other_centroids], axis=0
                    )
                    boundary_mask = distances_to_own / (distances_to_other + 1e-10) > 0.7
                    cluster_indices = np.where(cluster_mask)[0]
                    overlap_mask[cluster_indices[boundary_mask]] = True
        
        # Ensure we have samples in both regions
        if overlap_mask.sum() == 0 or (~overlap_mask).sum() == 0:
            distances = self.kmeans.transform(X)
            confidence = np.abs(distances[:, 0] - distances[:, 1])
            threshold = np.percentile(confidence, self.overlap_threshold * 100)
            overlap_mask = confidence <= threshold
        
        X_overlap = X[overlap_mask]
        y_overlap = y[overlap_mask]
        X_non_overlap = X[~overlap_mask]
        y_non_overlap = y[~overlap_mask]
        
        self._log(f"Overlap region: {len(X_overlap)} samples")
        self._log(f"Non-overlap region: {len(X_non_overlap)} samples")
        
        self.stats_['overlap_samples'] = len(X_overlap)
        self.stats_['non_overlap_samples'] = len(X_non_overlap)
        
        return X_overlap, y_overlap, X_non_overlap, y_non_overlap
    
    def _optimize_svm(self, X_non_overlap, y_non_overlap):
        """Optimize non-overlap region using SVM"""
        self._log(f"Optimizing with SVM - Before: {len(X_non_overlap)} samples")
        
        if len(X_non_overlap) < 10 or len(np.unique(y_non_overlap)) < 2:
            self._log("Skipping SVM: insufficient samples or classes")
            self.stats_['svm_removed'] = 0
            return X_non_overlap, y_non_overlap
        
        svm = SVC(kernel='rbf', random_state=self.random_state)
        svm.fit(X_non_overlap, y_non_overlap)
        
        predictions = svm.predict(X_non_overlap)
        correct_mask = predictions == y_non_overlap
        
        X_optimized = X_non_overlap[correct_mask]
        y_optimized = y_non_overlap[correct_mask]
        
        removed = len(X_non_overlap) - len(X_optimized)
        self._log(f"SVM removed: {removed} misclassified, After: {len(X_optimized)} samples")
        self.stats_['svm_removed'] = removed
        
        return X_optimized, y_optimized
    
    def fit_resample(self, X, y):
        """
        Apply OSM preprocessing and resampling
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
        
        Returns
        -------
        X_resampled : ndarray of shape (n_samples_new, n_features)
            Resampled data
        y_resampled : ndarray of shape (n_samples_new,)
            Resampled target values
        """
        self._log("\n" + "="*70)
        self._log("T6: OSM - Overlap-Separating Model")
        self._log("="*70)
        
        X = np.array(X, dtype=float)
        y = np.array(y)
        
        self.stats_['original_samples'] = len(X)
        self.stats_['original_features'] = X.shape[1]
        
        # Phase 1: Balancing Phase
        self._log("\n=== Phase 1: Balancing Phase ===")
        X = self._handle_missing_values(X)
        X = self._normalize_data(X, fit=True)
        
        if self.rose_sampling:
            X, y = self._balance_rose(X, y)
        
        if self.tomek_removal:
            X, y = self._remove_tomek(X, y)
        
        if self.feature_selection:
            X = self._select_features(X, y)
        
        if self.outlier_removal:
            X, y = self._remove_outliers(X, y)
        
        # Phase 2: Separating Class Overlap Phase
        self._log("\n=== Phase 2: Overlap Separation Phase ===")
        X_overlap, y_overlap, X_non_overlap, y_non_overlap = self._separate_overlap(X, y)
        
        if self.svm_optimization:
            X_non_overlap, y_non_overlap = self._optimize_svm(X_non_overlap, y_non_overlap)
        
        # Combine regions
        X_resampled = np.vstack([X_overlap, X_non_overlap])
        y_resampled = np.hstack([y_overlap, y_non_overlap])
        
        self.stats_['final_samples'] = len(X_resampled)
        self.stats_['final_features'] = X_resampled.shape[1]
        
        self._log("\n" + "="*70)
        self._log(f"OSM Complete: {self.stats_['original_samples']} → {self.stats_['final_samples']} samples")
        self._log(f"Features: {self.stats_['original_features']} → {self.stats_['final_features']}")
        self._log("="*70)
        
        return X_resampled, y_resampled
