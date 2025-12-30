"""
T1.3: NUS - Neighbourhood-based Under-Sampling

Implementation based on:
"Handling Class-Imbalance with KNN (Neighbourhood) Under-Sampling for Software Defect Prediction"
by Goyal, S. (2021), Artificial Intelligence Review

This implementation uses colonial neighbours and membership counting.
"""
import numpy as np
from collections import Counter
from sklearn.neighbors import NearestNeighbors
from .base_sampler import BaseSampler


class NUS(BaseSampler):
    """
    T1.3: Neighbourhood-based Under-Sampling (NUS)
    
    Uses colonial neighbours (k-NN from majority class) to identify and remove
    majority class instances that are close to minority instances.
    
    Algorithm:
    1. For each minority instance, find k nearest majority neighbours
    2. Nominate majority instances within median distance for elimination
    3. Eliminate only instances with multiple memberships (appear for multiple minority instances)
    
    Parameters
    ----------
    k_neighbors : int or None, default=None
        Number of neighbors to consider.
        If None, automatically set to min(sqrt(n_majority), 50)
    
    distance_threshold : str or float, default='median'
        Threshold for nominating neighbours for elimination.
        - 'median': Use median distance (paper default)
        - 'mean': Use mean distance
        - float: Use specific distance value
    
    min_membership : int, default=2
        Minimum number of memberships required for elimination.
        Instances appearing >= min_membership times are removed.
        Paper uses 2 (multiple memberships).
    
    random_state : int, default=42
        Random seed for reproducibility
    
    verbose : bool, default=False
        Whether to print detailed progress
    """
    
    def __init__(self, k_neighbors=None, distance_threshold='median', 
                 min_membership=2, random_state=42, verbose=False):
        super().__init__()
        self.k_neighbors = k_neighbors
        self.distance_threshold = distance_threshold
        self.min_membership = min_membership
        self.random_state = random_state
        self.verbose = verbose
        
        self.k_value_ = None
        self.stats_ = {}
    
    def _log(self, message):
        """Print message if verbose"""
        if self.verbose:
            print(message)
    
    def fit_resample(self, X, y):
        """
        Apply NUS algorithm to undersample the dataset.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target labels (0 for majority, 1 for minority)
        
        Returns
        -------
        X_resampled : ndarray of shape (n_samples_new, n_features)
            Resampled features
        y_resampled : ndarray of shape (n_samples_new,)
            Resampled labels
        """
        X = np.array(X, dtype=float)
        y = np.array(y)
        
        self._log(f"\n{'='*70}")
        self._log(f"T1.3: NUS - Neighbourhood-based Under-Sampling")
        self._log(f"{'='*70}")
        
        # Identify minority and majority classes
        unique_classes = np.unique(y)
        class_counts = Counter(y)
        
        # Assume minority class is the one with fewer samples
        minority_class = min(class_counts, key=class_counts.get)
        majority_class = max(class_counts, key=class_counts.get)
        
        # Get indices
        minority_indices = np.where(y == minority_class)[0]
        majority_indices = np.where(y == majority_class)[0]
        
        n_minority = len(minority_indices)
        n_majority = len(majority_indices)
        IR_before = n_majority / n_minority
        
        self._log(f"Initial IR: {IR_before:.2f}")
        self._log(f"Minority instances: {n_minority}")
        self._log(f"Majority instances: {n_majority}")
        
        self.stats_['original_samples'] = len(X)
        self.stats_['original_minority'] = n_minority
        self.stats_['original_majority'] = n_majority
        self.stats_['IR_before'] = IR_before
        
        # Set k_neighbors if not specified
        if self.k_neighbors is None:
            self.k_value_ = min(int(np.sqrt(n_majority)), 50)
        else:
            self.k_value_ = int(self.k_neighbors)
        
        # Ensure k doesn't exceed available majority samples
        self.k_value_ = min(self.k_value_, n_majority)
        
        self._log(f"Using k = {self.k_value_} neighbors")
        
        # Fit KNN on majority instances only
        knn = NearestNeighbors(n_neighbors=self.k_value_)
        knn.fit(X[majority_indices])
        
        # Storage for nominated points
        nominated_for_elimination = []
        
        # Process each minority instance
        self._log(f"\nFinding colonial neighbours for {n_minority} minority instances...")
        
        for minority_idx in minority_indices:
            minority_point = X[minority_idx].reshape(1, -1)
            
            # Find k nearest majority neighbors (Colonial Neighbours)
            distances, neighbour_indices = knn.kneighbors(minority_point)
            
            # Convert to actual majority indices
            majority_neighbour_indices = majority_indices[neighbour_indices[0]]
            neighbour_distances = distances[0]
            
            # Determine threshold for nomination
            if self.distance_threshold == 'median':
                threshold = np.median(neighbour_distances)
            elif self.distance_threshold == 'mean':
                threshold = np.mean(neighbour_distances)
            else:
                threshold = float(self.distance_threshold)
            
            # Nominate neighbours within threshold
            closest_mask = neighbour_distances <= threshold
            closest_neighbours = majority_neighbour_indices[closest_mask]
            
            # Add to nominated list
            nominated_for_elimination.extend(closest_neighbours)
        
        # Count memberships
        membership_counts = Counter(nominated_for_elimination)
        
        # Select for elimination based on min_membership
        to_eliminate = [idx for idx, count in membership_counts.items() 
                       if count >= self.min_membership]
        
        self._log(f"\nMajority points nominated: {len(set(nominated_for_elimination))}")
        self._log(f"Majority points with >= {self.min_membership} memberships: {len(to_eliminate)}")
        
        self.stats_['nominated'] = len(set(nominated_for_elimination))
        self.stats_['eliminated'] = len(to_eliminate)
        
        # Create undersampled dataset
        # Keep all minority instances and majority instances not in elimination list
        majority_to_keep = [idx for idx in majority_indices if idx not in to_eliminate]
        indices_to_keep = np.concatenate([minority_indices, majority_to_keep])
        
        X_resampled = X[indices_to_keep]
        y_resampled = y[indices_to_keep]
        
        # Compute new IR
        n_majority_after = len(majority_to_keep)
        IR_after = n_majority_after / n_minority
        IR_reduction = ((IR_before - IR_after) / IR_before * 100) if IR_before > 0 else 0
        size_reduction = ((len(X) - len(X_resampled)) / len(X) * 100)
        
        self.stats_['final_samples'] = len(X_resampled)
        self.stats_['final_minority'] = n_minority
        self.stats_['final_majority'] = n_majority_after
        self.stats_['IR_after'] = IR_after
        self.stats_['IR_reduction_pct'] = IR_reduction
        self.stats_['size_reduction_pct'] = size_reduction
        
        self._log(f"\nFinal IR: {IR_after:.2f}")
        self._log(f"Minority instances: {n_minority}")
        self._log(f"Majority instances: {n_majority_after}")
        self._log(f"IR reduction: {IR_reduction:.2f}%")
        self._log(f"Dataset size reduction: {size_reduction:.2f}%")
        self._log(f"{'='*70}")
        
        return X_resampled, y_resampled
