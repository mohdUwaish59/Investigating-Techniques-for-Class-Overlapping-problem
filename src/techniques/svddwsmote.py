"""
T2: SVDDWSMOTE - SVDD-based Overlap Handler
Implements SVDD-based class overlap handling for imbalanced learning

Reference: SVDD-based overlap handling technique
"""

import numpy as np
import time
from typing import Tuple
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from cvxopt import matrix, solvers
import warnings
warnings.filterwarnings('ignore')

from .base_sampler import BaseSampler


class SVDDWSMOTE(BaseSampler):
    """
    T2: SVDDWSMOTE - SVDD-based Class Overlap Handler
    
    Fast optimized version with configurable parameter search space.
    Uses Support Vector Data Description (SVDD) to identify and remove
    overlapped and noisy instances from imbalanced datasets.
    """
    
    def __init__(self, rho_threshold=0.045, delta_threshold=0.25, 
                 n_C1_candidates=5, n_sigma_candidates=5,
                 random_state=None, verbose=True):
        """
        Parameters:
        -----------
        rho_threshold : float, default=0.045
            Normalized local density threshold
        delta_threshold : float, default=0.25
            Normalized relative distance threshold
        n_C1_candidates : int, default=5
            Number of C1 values to try (paper uses 13)
        n_sigma_candidates : int, default=5
            Number of sigma values to try (paper uses 16)
        random_state : int, optional
            Random seed for reproducibility
        verbose : bool, default=True
            Print progress information
        """
        self.rho_threshold = rho_threshold
        self.delta_threshold = delta_threshold
        self.n_C1_candidates = n_C1_candidates
        self.n_sigma_candidates = n_sigma_candidates
        self.random_state = random_state
        self.verbose = verbose
        
        # Generate parameter candidates
        self.C1_candidates = np.logspace(-3, 2, n_C1_candidates)
        self.sigma_candidates = np.linspace(0.5, 10, n_sigma_candidates)
        
        if random_state is not None:
            np.random.seed(random_state)
        
        # Store results
        self.stats_ = {}
        self.best_C1_ = None
        self.best_sigma_ = None
        
        if self.verbose:
            print(f"SVDDWSMOTE: Using {len(self.C1_candidates)} C1 candidates and {len(self.sigma_candidates)} sigma candidates")
            print(f"SVDDWSMOTE: Total combinations: {len(self.C1_candidates) * len(self.sigma_candidates)}")
    
    def _calculate_cutoff_distance(self, X_min):
        """Calculate cutoff distance dc (2% of instances on average)"""
        n = len(X_min)
        if n < 100:
            distances = cdist(X_min, X_min, metric='euclidean')
            sorted_distances = np.sort(distances, axis=1)
            percent_idx = max(1, int(0.02 * n))
            dc = np.mean(sorted_distances[:, percent_idx])
        else:
            sample_size = min(100, n)
            sample_idx = np.random.choice(n, sample_size, replace=False)
            X_sample = X_min[sample_idx]
            distances = cdist(X_sample, X_sample, metric='euclidean')
            sorted_distances = np.sort(distances, axis=1)
            percent_idx = max(1, int(0.02 * sample_size))
            dc = np.mean(sorted_distances[:, percent_idx])
        return dc
    
    def _calculate_local_density(self, X_min, dc):
        """Calculate local density using Gaussian kernel"""
        distances = cdist(X_min, X_min, metric='euclidean')
        rho = np.sum(np.exp(-np.power(distances / dc, 2)), axis=1)
        return rho
    
    def _calculate_relative_distance(self, X_min, rho):
        """Calculate relative distance δᵢ"""
        n = len(X_min)
        distances = cdist(X_min, X_min, metric='euclidean')
        delta = np.zeros(n)
        
        sorted_indices = np.argsort(-rho)
        for i in range(n):
            idx = sorted_indices[i]
            if i == 0:
                delta[idx] = np.max(distances[idx])
            else:
                higher_density_indices = sorted_indices[:i]
                delta[idx] = np.min(distances[idx, higher_density_indices])
        
        return delta
    
    def _normalize(self, values):
        """Normalize values to [0, 1]"""
        min_val = np.min(values)
        max_val = np.max(values)
        if max_val - min_val == 0:
            return np.zeros_like(values)
        return (values - min_val) / (max_val - min_val)
    
    def _identify_candidate_noisy_minority(self, X_min):
        """Identify candidate noisy minority instances"""
        dc = self._calculate_cutoff_distance(X_min)
        rho = self._calculate_local_density(X_min, dc)
        delta = self._calculate_relative_distance(X_min, rho)
        
        rho_norm = self._normalize(rho)
        delta_norm = self._normalize(delta)
        
        candidate_noisy = (rho_norm <= self.rho_threshold) & (delta_norm >= self.delta_threshold)
        return np.where(candidate_noisy)[0], rho_norm, delta_norm
    
    def _gaussian_kernel(self, X1, X2, sigma):
        """Gaussian kernel k(x,y) = exp(-||x-y||²/σ²)"""
        distances_sq = cdist(X1, X2, metric='sqeuclidean')
        return np.exp(-distances_sq / (sigma ** 2))
    
    def _train_svdd(self, X_maj, X_min, candidate_noisy_idx, C1, sigma):
        """Train SVDD model using dual formulation"""
        n_maj = len(X_maj)
        n_min = len(X_min)
        n = n_maj + n_min
        
        X = np.vstack([X_maj, X_min])
        y = np.array([1] * n_maj + [-1] * n_min)
        
        IM = n_maj / n_min
        C2 = IM * C1
        C = np.array([C1] * n_maj + [C2] * n_min)
        
        for idx in candidate_noisy_idx:
            C[n_maj + idx] = C1
        
        K = self._gaussian_kernel(X, X, sigma)
        K += np.eye(n) * 1e-8  # Regularization for numerical stability
        
        P = matrix(np.outer(y, y) * K)
        q = matrix(-np.diag(K) * y)
        G = matrix(np.vstack([-np.eye(n), np.eye(n)]))
        h = matrix(np.hstack([np.zeros(n), C]))
        A = matrix(y.reshape(1, -1).astype(float))
        b = matrix([1.0])
        
        solvers.options['show_progress'] = False
        solvers.options['maxiters'] = 100
        
        try:
            solution = solvers.qp(P, q, G, h, A, b)
            alpha = np.array(solution['x']).flatten()
            sv_indices = np.where(alpha > 1e-5)[0]
            return alpha, sv_indices
        except:
            return None, None
    
    def _evaluate_svdd_params(self, X_maj, X_min, candidate_noisy_idx, C1, sigma):
        """Evaluate SVDD parameters using EM metric"""
        alpha, support_vectors = self._train_svdd(X_maj, X_min, candidate_noisy_idx, C1, sigma)
        if alpha is None:
            return float('inf')
        
        IM = len(X_maj) / len(X_min)
        n_maj = len(X_maj)
        num_majority_outlier = np.sum(alpha[:n_maj] >= C1 - 1e-5)
        C2 = IM * C1
        num_minority_outlier = np.sum(alpha[n_maj:] >= C2 - 1e-5)
        
        num_sv = np.sum((alpha[:n_maj] > 1e-5) & (alpha[:n_maj] < C1 - 1e-5))
        num_sv += np.sum((alpha[n_maj:] > 1e-5) & (alpha[n_maj:] < C2 - 1e-5))
        
        EM = num_majority_outlier + IM * num_minority_outlier + num_sv
        return EM
    
    def _select_best_svdd_params(self, X_maj, X_min, candidate_noisy_idx):
        """Select best C1 and sigma using EM metric"""
        best_EM = float('inf')
        best_C1 = None
        best_sigma = None
        
        total_combinations = len(self.C1_candidates) * len(self.sigma_candidates)
        current = 0
        
        for C1 in self.C1_candidates:
            for sigma in self.sigma_candidates:
                current += 1
                if self.verbose and (current % 5 == 0 or current == total_combinations):
                    print(f"SVDDWSMOTE: Progress: {current}/{total_combinations} parameter combinations tested", end='\r')
                
                EM = self._evaluate_svdd_params(X_maj, X_min, candidate_noisy_idx, C1, sigma)
                if EM < best_EM:
                    best_EM = EM
                    best_C1 = C1
                    best_sigma = sigma
        
        if self.verbose:
            print()  # New line after progress
        
        return best_C1, best_sigma
    
    def _compute_distance_to_boundary(self, X, X_train, alpha, y, sigma):
        """Compute distance to SVDD boundary"""
        n_test = len(X)
        K_xx = np.ones(n_test)
        K_train = self._gaussian_kernel(X_train, X_train, sigma)
        term2 = np.sum(np.outer(alpha * y, alpha * y) * K_train)
        K_cross = self._gaussian_kernel(X, X_train, sigma)
        term3 = 2 * np.dot(K_cross, alpha * y)
        distances_sq = K_xx + term2 - term3
        return distances_sq
    
    def _compute_radius(self, X_train, alpha, y, C1, sigma):
        """Compute SVDD radius from support vectors on the boundary"""
        n_maj = np.sum(y == 1)
        n_min = np.sum(y == -1)
        IM = n_maj / n_min
        C2 = IM * C1
        
        on_boundary = np.zeros(len(alpha), dtype=bool)
        maj_mask = (alpha[:n_maj] > 1e-5) & (alpha[:n_maj] < C1 - 1e-5)
        on_boundary[:n_maj] = maj_mask
        min_mask = (alpha[n_maj:] > 1e-5) & (alpha[n_maj:] < C2 - 1e-5)
        on_boundary[n_maj:] = min_mask
        
        if np.sum(on_boundary) == 0:
            on_boundary = alpha > 1e-5
        
        if np.sum(on_boundary) == 0:
            distances_sq = self._compute_distance_to_boundary(X_train, X_train, alpha, y, sigma)
            return np.mean(distances_sq)
        
        X_boundary = X_train[on_boundary]
        distances_sq = self._compute_distance_to_boundary(X_boundary, X_train, alpha, y, sigma)
        R_sq = np.mean(distances_sq)
        return R_sq
    
    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove overlapped and noisy instances from the dataset
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target vector (0 for minority, 1 for majority)
        
        Returns:
        --------
        X_resampled : np.ndarray
            Cleaned feature matrix
        y_resampled : np.ndarray
            Cleaned target vector
        """
        start_time = time.time()
        
        X = np.array(X)
        y = np.array(y)
        
        # Normalize data
        scaler = StandardScaler()
        X_normalized = scaler.fit_transform(X)
        
        # Separate classes
        minority_mask = y == 0
        majority_mask = y == 1
        X_min = X_normalized[minority_mask]
        X_maj = X_normalized[majority_mask]
        
        # Store initial statistics
        self.stats_['original'] = {
            'n_majority': len(X_maj),
            'n_minority': len(X_min),
            'imbalance_ratio': len(X_maj) / len(X_min) if len(X_min) > 0 else 0
        }
        
        if self.verbose:
            print(f"SVDDWSMOTE: Original IR={self.stats_['original']['imbalance_ratio']:.2f} "
                  f"(Maj={len(X_maj)}, Min={len(X_min)})")
        
        # Step 1: Identify candidate noisy minority instances
        if self.verbose:
            print("SVDDWSMOTE: Step 1 - Identifying candidate noisy minority instances...")
        
        candidate_noisy_idx, rho_norm, delta_norm = self._identify_candidate_noisy_minority(X_min)
        
        if self.verbose:
            print(f"SVDDWSMOTE: Identified {len(candidate_noisy_idx)} candidate noisy minority instances")
        
        # Step 2: Select best SVDD parameters
        if self.verbose:
            print("SVDDWSMOTE: Step 2 - Selecting best SVDD parameters...")
        
        best_C1, best_sigma = self._select_best_svdd_params(X_maj, X_min, candidate_noisy_idx)
        self.best_C1_ = best_C1
        self.best_sigma_ = best_sigma
        
        if self.verbose:
            print(f"SVDDWSMOTE: Best parameters: C1={best_C1:.4f}, sigma={best_sigma:.4f}")
        
        # Step 3: Train final SVDD model
        if self.verbose:
            print("SVDDWSMOTE: Step 3 - Training final SVDD model...")
        
        alpha, sv_indices = self._train_svdd(X_maj, X_min, candidate_noisy_idx, best_C1, best_sigma)
        
        if alpha is None:
            if self.verbose:
                print("SVDDWSMOTE: SVDD training failed, returning original data")
            return X, y
        
        # Step 4: Identify instances to remove
        if self.verbose:
            print("SVDDWSMOTE: Step 4 - Identifying instances to remove...")
        
        n_maj = len(X_maj)
        n_min = len(X_min)
        IM = n_maj / n_min
        C2 = IM * best_C1
        
        X_combined = np.vstack([X_maj, X_min])
        y_combined = np.array([1] * n_maj + [-1] * n_min)
        
        R_sq = self._compute_radius(X_combined, alpha, y_combined, best_C1, best_sigma)
        distances_sq = self._compute_distance_to_boundary(X_combined, X_combined, alpha, y_combined, best_sigma)
        
        remove_mask = np.zeros(len(X_combined), dtype=bool)
        
        # Remove majority instances with α >= C1 (outside boundary)
        remove_mask[:n_maj] = (alpha[:n_maj] >= best_C1 - 1e-5)
        
        # Remove minority instances with α >= C2 AND in candidate noisy set (inside boundary)
        for idx in candidate_noisy_idx:
            global_idx = n_maj + idx
            if alpha[global_idx] >= C2 - 1e-5 and distances_sq[global_idx] < R_sq:
                remove_mask[global_idx] = True
        
        # Keep instances
        keep_mask = ~remove_mask
        X_clean = X_combined[keep_mask]
        y_clean = y_combined[keep_mask]
        
        # Convert y back to 0/1 format
        y_clean = (y_clean == 1).astype(int)
        
        # Inverse transform to original scale
        X_clean = scaler.inverse_transform(X_clean)
        
        # Store final statistics
        n_maj_removed = np.sum(remove_mask[:n_maj])
        n_min_removed = np.sum(remove_mask[n_maj:])
        elapsed_time = time.time() - start_time
        
        self.stats_['final'] = {
            'n_majority': int(np.sum(y_clean == 1)),
            'n_minority': int(np.sum(y_clean == 0)),
            'imbalance_ratio': np.sum(y_clean == 1) / np.sum(y_clean == 0) if np.sum(y_clean == 0) > 0 else 0,
            'n_maj_removed': int(n_maj_removed),
            'n_min_removed': int(n_min_removed),
            'execution_time': elapsed_time
        }
        
        if self.verbose:
            print(f"\nSVDDWSMOTE: Results:")
            print(f"SVDDWSMOTE: Removed {n_maj_removed} majority instances (overlapped/noisy)")
            print(f"SVDDWSMOTE: Removed {n_min_removed} minority instances (noisy)")
            print(f"SVDDWSMOTE: Final IR={self.stats_['final']['imbalance_ratio']:.2f} "
                  f"(Maj={self.stats_['final']['n_majority']}, Min={self.stats_['final']['n_minority']})")
            print(f"SVDDWSMOTE: Total execution time: {elapsed_time:.2f} seconds")
        
        return X_clean, y_clean
