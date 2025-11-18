"""
T4: NBUS - Neighbourhood-Based Undersampling Methods
Implementation of 4 neighbourhood-based undersampling techniques

Reference: Vuttipittayamongkol, P., & Elyan, E. (2020). 
Neighbourhood-based undersampling approach for handling imbalanced and overlapped data.
Information Sciences, 509, 47-70.
"""

import numpy as np
from typing import Tuple, Optional
from sklearn.neighbors import NearestNeighbors
from .base_sampler import BaseSampler


class NBUSBase(BaseSampler):
    """
    Base class for Neighbourhood-Based Undersampling methods
    """
    
    def __init__(self, k=None, random_state=None, verbose=True):
        """
        Parameters:
        -----------
        k : int, optional
            Number of nearest neighbors. If None, calculated as sqrt(N) + imb
        random_state : int, optional
            Random seed for reproducibility
        verbose : bool, default=True
            Print progress information
        """
        self.k = k
        self.random_state = random_state
        self.verbose = verbose
        self.stats_ = {}
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _calculate_k(self, y):
        """Calculate k using formula from paper: k = sqrt(N) + imb"""
        N = len(y)
        n_neg = np.sum(y == 0)
        n_pos = np.sum(y == 1)
        imb = n_neg / n_pos if n_pos > 0 else 1
        k = int(np.sqrt(N) + imb)
        return k


class NBBasic(NBUSBase):
    """
    T4.1: NB-Basic - Basic Neighbourhood Search Undersampling
    
    Algorithm 1 from the paper:
    Removes any negative instance that has at least one positive neighbor.
    """
    
    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform undersampling based on NB-Basic algorithm
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target vector (0 for negative/majority, 1 for positive/minority)
        
        Returns:
        --------
        X_resampled : np.ndarray
            Resampled features
        y_resampled : np.ndarray
            Resampled labels
        """
        X = np.array(X)
        y = np.array(y)
        
        # Calculate k if not provided
        k = self.k if self.k is not None else self._calculate_k(y)
        
        # Store initial statistics
        self.stats_['original'] = {
            'n_negative': int(np.sum(y == 0)),
            'n_positive': int(np.sum(y == 1)),
            'k_used': k
        }
        
        if self.verbose:
            print(f"NB-Basic: Using k={k}")
            print(f"NB-Basic: Original (Neg={self.stats_['original']['n_negative']}, "
                  f"Pos={self.stats_['original']['n_positive']})")
        
        # Separate positive and negative instances
        neg_mask = y == 0
        pos_mask = y == 1
        X_neg = X[neg_mask]
        y_neg = y[neg_mask]
        X_pos = X[pos_mask]
        y_pos = y[pos_mask]
        
        # Fit k-NN on entire training set
        nbrs = NearestNeighbors(n_neighbors=k + 1)
        nbrs.fit(X)
        
        # Find neighbors for each negative instance
        X_to_remove = []
        for i, x in enumerate(X_neg):
            distances, indices = nbrs.kneighbors([x], n_neighbors=k + 1)
            neighbor_indices = indices[0][1:]  # Exclude self
            neighbor_labels = y[neighbor_indices]
            
            # Check if 'positive' in NN
            if 1 in neighbor_labels:
                X_to_remove.append(i)
        
        # Remove identified instances
        mask_to_keep = np.ones(len(X_neg), dtype=bool)
        mask_to_keep[X_to_remove] = False
        X_neg_resampled = X_neg[mask_to_keep]
        y_neg_resampled = y_neg[mask_to_keep]
        
        # Combine with positive instances
        X_resampled = np.vstack([X_pos, X_neg_resampled])
        y_resampled = np.hstack([y_pos, y_neg_resampled])
        
        # Store final statistics
        self.stats_['final'] = {
            'n_negative': int(np.sum(y_resampled == 0)),
            'n_positive': int(np.sum(y_resampled == 1)),
            'n_removed': len(X_to_remove)
        }
        
        if self.verbose:
            print(f"NB-Basic: Removed {self.stats_['final']['n_removed']} negative instances")
            print(f"NB-Basic: Final (Neg={self.stats_['final']['n_negative']}, "
                  f"Pos={self.stats_['final']['n_positive']})")
        
        return X_resampled, y_resampled


class NBTomek(NBUSBase):
    """
    T4.2: NB-Tomek - Modified Tomek Link Search Undersampling
    
    Algorithm 2 from the paper:
    Extension of NB-Basic - removes negative instance only if it appears 
    within the k nearest neighbors of its positive neighbor (bidirectional).
    """
    
    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform undersampling based on NB-Tomek algorithm"""
        X = np.array(X)
        y = np.array(y)
        
        k = self.k if self.k is not None else self._calculate_k(y)
        
        self.stats_['original'] = {
            'n_negative': int(np.sum(y == 0)),
            'n_positive': int(np.sum(y == 1)),
            'k_used': k
        }
        
        if self.verbose:
            print(f"NB-Tomek: Using k={k}")
            print(f"NB-Tomek: Original (Neg={self.stats_['original']['n_negative']}, "
                  f"Pos={self.stats_['original']['n_positive']})")
        
        neg_mask = y == 0
        pos_mask = y == 1
        X_neg = X[neg_mask]
        y_neg = y[neg_mask]
        X_pos = X[pos_mask]
        y_pos = y[pos_mask]
        
        nbrs = NearestNeighbors(n_neighbors=k + 1)
        nbrs.fit(X)
        
        neg_indices_to_remove = set()
        
        for neg_idx, x_neg in enumerate(X_neg):
            distances, indices = nbrs.kneighbors([x_neg], n_neighbors=k + 1)
            neighbor_indices = indices[0][1:]
            
            for neighbor_idx in neighbor_indices:
                if y[neighbor_idx] == 1:
                    y_pos_neighbor = X[neighbor_idx]
                    distances_pos, indices_pos = nbrs.kneighbors([y_pos_neighbor], n_neighbors=k + 1)
                    neighbors_of_pos = indices_pos[0][1:]
                    
                    original_neg_idx = np.where(neg_mask)[0][neg_idx]
                    
                    if original_neg_idx in neighbors_of_pos:
                        neg_indices_to_remove.add(neg_idx)
                        break
        
        mask_to_keep = np.ones(len(X_neg), dtype=bool)
        mask_to_keep[list(neg_indices_to_remove)] = False
        X_neg_resampled = X_neg[mask_to_keep]
        y_neg_resampled = y_neg[mask_to_keep]
        
        X_resampled = np.vstack([X_pos, X_neg_resampled])
        y_resampled = np.hstack([y_pos, y_neg_resampled])
        
        self.stats_['final'] = {
            'n_negative': int(np.sum(y_resampled == 0)),
            'n_positive': int(np.sum(y_resampled == 1)),
            'n_removed': len(neg_indices_to_remove)
        }
        
        if self.verbose:
            print(f"NB-Tomek: Removed {self.stats_['final']['n_removed']} negative instances")
            print(f"NB-Tomek: Final (Neg={self.stats_['final']['n_negative']}, "
                  f"Pos={self.stats_['final']['n_positive']})")
        
        return X_resampled, y_resampled


class NBComm(NBUSBase):
    """
    T4.3: NB-Comm - Common Nearest Neighbours Search Undersampling
    
    Algorithm 3 from the paper:
    Uses positive queries - removes common negative neighbors of any two positive instances.
    """
    
    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform undersampling based on NB-Comm algorithm"""
        X = np.array(X)
        y = np.array(y)
        
        k = self.k if self.k is not None else self._calculate_k(y)
        
        self.stats_['original'] = {
            'n_negative': int(np.sum(y == 0)),
            'n_positive': int(np.sum(y == 1)),
            'k_used': k
        }
        
        if self.verbose:
            print(f"NB-Comm: Using k={k}")
            print(f"NB-Comm: Original (Neg={self.stats_['original']['n_negative']}, "
                  f"Pos={self.stats_['original']['n_positive']})")
        
        neg_mask = y == 0
        pos_mask = y == 1
        X_neg = X[neg_mask]
        y_neg = y[neg_mask]
        X_pos = X[pos_mask]
        y_pos = y[pos_mask]
        
        nbrs = NearestNeighbors(n_neighbors=k + 1)
        nbrs.fit(X)
        
        neg_original_indices = np.where(neg_mask)[0]
        frequency_table = {idx: 0 for idx in neg_original_indices}
        
        for x_pos in X_pos:
            distances, indices = nbrs.kneighbors([x_pos], n_neighbors=k + 1)
            neighbor_indices = indices[0][1:]
            
            for neighbor_idx in neighbor_indices:
                if y[neighbor_idx] == 0:
                    frequency_table[neighbor_idx] += 1
        
        indices_to_remove = [idx for idx, freq in frequency_table.items() if freq > 1]
        
        original_to_neg_idx = {orig_idx: neg_idx for neg_idx, orig_idx in enumerate(neg_original_indices)}
        neg_indices_to_remove = [original_to_neg_idx[idx] for idx in indices_to_remove if idx in original_to_neg_idx]
        
        mask_to_keep = np.ones(len(X_neg), dtype=bool)
        mask_to_keep[neg_indices_to_remove] = False
        X_neg_resampled = X_neg[mask_to_keep]
        y_neg_resampled = y_neg[mask_to_keep]
        
        X_resampled = np.vstack([X_pos, X_neg_resampled])
        y_resampled = np.hstack([y_pos, y_neg_resampled])
        
        self.stats_['final'] = {
            'n_negative': int(np.sum(y_resampled == 0)),
            'n_positive': int(np.sum(y_resampled == 1)),
            'n_removed': len(neg_indices_to_remove)
        }
        
        if self.verbose:
            print(f"NB-Comm: Removed {self.stats_['final']['n_removed']} negative instances")
            print(f"NB-Comm: Final (Neg={self.stats_['final']['n_negative']}, "
                  f"Pos={self.stats_['final']['n_positive']})")
        
        return X_resampled, y_resampled


class NBRec(NBUSBase):
    """
    T4.4: NB-Rec - Recursive Search Undersampling
    
    Algorithm 4 from the paper:
    Extension of NB-Comm with recursive search using secondary queries.
    """
    
    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform undersampling based on NB-Rec algorithm"""
        X = np.array(X)
        y = np.array(y)
        
        k = self.k if self.k is not None else self._calculate_k(y)
        
        self.stats_['original'] = {
            'n_negative': int(np.sum(y == 0)),
            'n_positive': int(np.sum(y == 1)),
            'k_used': k
        }
        
        if self.verbose:
            print(f"NB-Rec: Using k={k}")
            print(f"NB-Rec: Original (Neg={self.stats_['original']['n_negative']}, "
                  f"Pos={self.stats_['original']['n_positive']})")
        
        # First, run NB-Comm
        nb_comm = NBComm(k=k, verbose=False)
        X_temp, y_temp = nb_comm.fit_resample(X, y)
        
        # Identify which negative instances were removed by NB-Comm
        neg_mask = y == 0
        X_neg = X[neg_mask]
        neg_original_indices = np.where(neg_mask)[0]
        
        remaining_set = set()
        for i, x_remaining in enumerate(X_temp):
            for j, x_original in enumerate(X):
                if np.allclose(x_remaining, x_original):
                    remaining_set.add(j)
                    break
        
        X_set_indices = [idx for idx in neg_original_indices if idx not in remaining_set]
        X_set = X[X_set_indices]
        
        # Recursive search
        nbrs = NearestNeighbors(n_neighbors=k + 1)
        nbrs.fit(X)
        
        frequency_table_2 = {idx: 0 for idx in neg_original_indices}
        
        for x1 in X_set:
            distances, indices = nbrs.kneighbors([x1], n_neighbors=k + 1)
            neighbor_indices = indices[0][1:]
            
            for neighbor_idx in neighbor_indices:
                if y[neighbor_idx] == 0:
                    frequency_table_2[neighbor_idx] += 1
        
        X2_indices = [idx for idx, freq in frequency_table_2.items() if freq > 1]
        
        all_indices_to_remove = set(X_set_indices) | set(X2_indices)
        
        original_to_neg_idx = {orig_idx: neg_idx for neg_idx, orig_idx in enumerate(neg_original_indices)}
        neg_indices_to_remove = [original_to_neg_idx[idx] for idx in all_indices_to_remove if idx in original_to_neg_idx]
        
        pos_mask = y == 1
        X_pos = X[pos_mask]
        y_pos = y[pos_mask]
        
        mask_to_keep = np.ones(len(X_neg), dtype=bool)
        mask_to_keep[neg_indices_to_remove] = False
        X_neg_resampled = X_neg[mask_to_keep]
        y_neg_resampled = y[neg_mask][mask_to_keep]
        
        X_resampled = np.vstack([X_pos, X_neg_resampled])
        y_resampled = np.hstack([y_pos, y_neg_resampled])
        
        self.stats_['final'] = {
            'n_negative': int(np.sum(y_resampled == 0)),
            'n_positive': int(np.sum(y_resampled == 1)),
            'n_removed': len(neg_indices_to_remove)
        }
        
        if self.verbose:
            print(f"NB-Rec: Removed {self.stats_['final']['n_removed']} negative instances")
            print(f"NB-Rec: Final (Neg={self.stats_['final']['n_negative']}, "
                  f"Pos={self.stats_['final']['n_positive']})")
        
        return X_resampled, y_resampled


# Wrapper class for easy access
class NBUS(BaseSampler):
    """
    T4: NBUS - Neighbourhood-Based Undersampling
    
    Wrapper class that provides access to all 4 NBUS variants:
    - NB-Basic: Basic neighbourhood search
    - NB-Tomek: Modified Tomek link search
    - NB-Comm: Common nearest neighbours search
    - NB-Rec: Recursive search
    """
    
    def __init__(self, method='NB-Basic', k=None, random_state=None, verbose=True):
        """
        Parameters:
        -----------
        method : str, default='NB-Basic'
            Which NBUS variant to use: 'NB-Basic', 'NB-Tomek', 'NB-Comm', 'NB-Rec'
        k : int, optional
            Number of nearest neighbors
        random_state : int, optional
            Random seed
        verbose : bool, default=True
            Print progress
        """
        self.method = method
        self.k = k
        self.random_state = random_state
        self.verbose = verbose
        
        # Initialize the appropriate method
        if method == 'NB-Basic':
            self.sampler = NBBasic(k=k, random_state=random_state, verbose=verbose)
        elif method == 'NB-Tomek':
            self.sampler = NBTomek(k=k, random_state=random_state, verbose=verbose)
        elif method == 'NB-Comm':
            self.sampler = NBComm(k=k, random_state=random_state, verbose=verbose)
        elif method == 'NB-Rec':
            self.sampler = NBRec(k=k, random_state=random_state, verbose=verbose)
        else:
            raise ValueError(f"Unknown method: {method}. Choose from: NB-Basic, NB-Tomek, NB-Comm, NB-Rec")
        
        self.stats_ = {}
    
    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply the selected NBUS method"""
        X_res, y_res = self.sampler.fit_resample(X, y)
        self.stats_ = self.sampler.stats_
        return X_res, y_res
