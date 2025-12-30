"""
T1.1: URNS - Undersampling based on Recursive Neighbourhood Search

Implementation based on:
"Overlap-Based Undersampling Method for Classification of Imbalanced Medical Datasets"
by Pattaramon Vuttipittayamongkol and Eyad Elyan

This implementation follows Algorithm 1 from the paper.
"""
import numpy as np
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors
from .base_sampler import BaseSampler


class URNS(BaseSampler):
    """
    T1.1: Undersampling based on Recursive Neighbourhood Search (URNS)
    
    Maximizes visibility of minority class by eliminating majority class instances
    from overlapping regions through recursive neighbourhood searching.
    
    The algorithm performs two rounds:
    1. Find common majority class neighbours of minority instances
    2. Find common majority class neighbours of instances found in round 1
    3. Remove all common neighbours from both rounds
    
    Parameters
    ----------
    k : int or 'adaptive', default='adaptive'
        Number of nearest neighbours to search.
        If 'adaptive', k is computed as: k = sqrt(N) + sqrt(IR)
        where N is dataset size and IR is imbalance ratio
    
    min_frequency : int, default=2
        Minimum frequency for an instance to be considered a common neighbour.
        Instances appearing >= min_frequency times are removed.
    
    rounds : int, default=2
        Number of recursive rounds to perform (1 or 2).
        Paper uses 2 rounds for better overlap removal.
    
    random_state : int, default=42
        Random seed for reproducibility
    
    verbose : bool, default=False
        Whether to print detailed progress
    """
    
    def __init__(self, k='adaptive', min_frequency=2, rounds=2, 
                 random_state=42, verbose=False):
        super().__init__()
        self.k = k
        self.min_frequency = min_frequency
        self.rounds = rounds
        self.random_state = random_state
        self.verbose = verbose
        
        self.k_value_ = None
        self.stats_ = {}
    
    def _log(self, message):
        """Print message if verbose"""
        if self.verbose:
            print(message)
    
    def _calculate_adaptive_k(self, X, y):
        """
        Calculate adaptive k value based on Equation 1 from the paper:
        k = sqrt(N) + sqrt(IR)
        
        where:
        - N is the total number of instances
        - IR is the imbalance ratio (majority_size / minority_size)
        """
        N = len(X)
        
        # Count class distributions
        unique, counts = np.unique(y, return_counts=True)
        class_counts = dict(zip(unique, counts))
        
        # Assuming binary classification with 1 as minority class
        minority_count = class_counts.get(1, min(counts))
        majority_count = class_counts.get(0, max(counts))
        
        IR = majority_count / minority_count
        k = int(np.sqrt(N) + np.sqrt(IR))
        
        return k
    
    def _common_neighbour(self, X, y, queries_indices, k):
        """
        CommonNeighbour function from Algorithm 1
        
        Identifies majority class instances that are common neighbours
        of multiple query instances.
        
        Parameters
        ----------
        X : array-like
            Training data
        y : array-like
            Target labels
        queries_indices : array-like
            Indices of query instances
        k : int
            Number of nearest neighbours
        
        Returns
        -------
        common_neighbours : set
            Set of indices of majority class instances that are common neighbours
        """
        # Initialize frequency table
        frequency_table = defaultdict(int)
        
        # Fit k-NN model
        knn = NearestNeighbors(n_neighbors=k+1, algorithm='auto')
        knn.fit(X)
        
        # For each query instance
        for q_idx in queries_indices:
            # Find k nearest neighbours
            distances, indices = knn.kneighbors(X[q_idx].reshape(1, -1))
            
            # Get the neighbours (excluding the query itself)
            neighbours = indices[0][1:]  # Skip first as it's the query itself
            
            # Filter for majority class neighbours (class 0)
            majority_neighbours = [idx for idx in neighbours if y[idx] == 0]
            
            # Update frequency count for each majority class neighbour
            for neighbour_idx in majority_neighbours:
                frequency_table[neighbour_idx] += 1
        
        # Select instances that appear >= min_frequency times
        common_neighbours = set()
        for instance_idx, freq in frequency_table.items():
            if freq >= self.min_frequency:
                common_neighbours.add(instance_idx)
        
        return common_neighbours
    
    def fit_resample(self, X, y):
        """
        Resample the dataset using URNS method following Algorithm 1.
        
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
        
        # Determine k value
        if self.k == 'adaptive':
            self.k_value_ = self._calculate_adaptive_k(X, y)
        else:
            self.k_value_ = int(self.k)
        
        self._log(f"\n{'='*70}")
        self._log(f"T1.1: URNS - Recursive Neighbourhood Search")
        self._log(f"{'='*70}")
        self._log(f"Using k = {self.k_value_}")
        
        # Get indices of minority class instances
        minority_indices = np.where(y == 1)[0]
        majority_indices = np.where(y == 0)[0]
        
        self._log(f"Original dataset size: {len(X)}")
        self._log(f"Minority class size: {len(minority_indices)}")
        self._log(f"Majority class size: {len(majority_indices)}")
        
        self.stats_['original_samples'] = len(X)
        self.stats_['original_minority'] = len(minority_indices)
        self.stats_['original_majority'] = len(majority_indices)
        
        # Perform recursive rounds
        all_instances_to_remove = set()
        query_indices = minority_indices
        
        for round_num in range(1, self.rounds + 1):
            self._log(f"\nRound {round_num}: Finding common neighbours...")
            
            round_common = self._common_neighbour(X, y, query_indices, self.k_value_)
            self._log(f"Found {len(round_common)} common neighbours in Round {round_num}")
            
            self.stats_[f'round_{round_num}_removed'] = len(round_common)
            
            all_instances_to_remove.update(round_common)
            
            # For next round, use current round's results as queries
            query_indices = list(round_common)
            
            # If no common neighbours found, stop
            if len(round_common) == 0:
                self._log(f"No common neighbours found in Round {round_num}. Stopping.")
                break
        
        # Create mask for instances to keep
        mask = np.ones(len(X), dtype=bool)
        mask[list(all_instances_to_remove)] = False
        
        # Apply undersampling
        X_resampled = X[mask]
        y_resampled = y[mask]
        
        self.stats_['total_removed'] = len(all_instances_to_remove)
        self.stats_['final_samples'] = len(X_resampled)
        self.stats_['final_minority'] = np.sum(y_resampled == 1)
        self.stats_['final_majority'] = np.sum(y_resampled == 0)
        
        self._log(f"\nTotal instances removed: {len(all_instances_to_remove)}")
        self._log(f"Resampled dataset size: {len(X_resampled)}")
        self._log(f"Minority class size: {np.sum(y_resampled == 1)}")
        self._log(f"Majority class size: {np.sum(y_resampled == 0)}")
        self._log(f"{'='*70}")
        
        return X_resampled, y_resampled
