"""
Implementation of Boosted Overlap-Based Undersampling (BoostOBU)

Reference:
Vuttipittayamongkol, P., & Elyan, E. (2020). Improved overlap-based undersampling 
for imbalanced dataset classification with application to epilepsy and Parkinson's disease. 
International Journal of Neural Systems, 30(8), 2050043.

This implements Algorithm 3: BoostOBU with BLSMOTE1 and FCM clustering.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import warnings
from .base_sampler import BaseSampler

warnings.filterwarnings('ignore')

try:
    from skfuzzy.cluster import cmeans
    SKFUZZY_AVAILABLE = True
except ImportError:
    SKFUZZY_AVAILABLE = False


class BLSMOTE1:
    """
    Borderline-SMOTE variant 1 for BoostOBU
    
    Identifies DANGER samples (minority samples with 0 < m' < k majority neighbors)
    and generates synthetic samples using only minority class nearest neighbors.
    """
    
    def __init__(self, k=5, random_state=None):
        """
        Parameters:
        -----------
        k : int, default=5
            Number of nearest neighbors
        random_state : int
            Random seed
        """
        self.k = k
        self.random_state = random_state

    def fit_resample(self, X, y):
        """Apply BLSMOTE1"""
        np.random.seed(self.random_state)
        X = np.array(X)
        y = np.array(y)
        
        X_min = X[y == 1]
        X_maj = X[y == 0]
        
        if len(X_min) == 0:
            return X, y
        
        # Find k nearest neighbors for each minority sample
        nbrs = NearestNeighbors(n_neighbors=min(self.k + 1, len(X))).fit(X)
        distances, indices = nbrs.kneighbors(X_min)
        
        # Identify DANGER samples
        # DANGER: 0 < m' < k (where m' is number of majority neighbors)
        danger_samples = []
        danger_indices = []
        
        for i, neighbors in enumerate(indices):
            # Exclude self (first neighbor)
            neighbor_labels = y[neighbors[1:min(len(neighbors), self.k + 1)]]
            n_majority = np.sum(neighbor_labels == 0)
            
            # DANGER condition
            if 0 < n_majority < min(self.k, len(neighbor_labels)):
                danger_samples.append(X_min[i])
                danger_indices.append(i)
        
        if len(danger_samples) == 0:
            # No danger samples, return original
            return X, y
        
        danger_samples = np.array(danger_samples)
        
        # Generate synthetic samples from DANGER samples
        # Only using minority class nearest neighbors (BLSMOTE1)
        synthetic = []
        
        for danger_idx in danger_indices:
            if len(X_min) > 1:
                # Find k minority nearest neighbors of this danger sample
                min_nbrs = NearestNeighbors(n_neighbors=min(self.k + 1, len(X_min))).fit(X_min)
                _, min_indices = min_nbrs.kneighbors([X_min[danger_idx]])
                
                # Randomly select one minority neighbor (excluding self)
                available_neighbors = min_indices[0][1:]
                if len(available_neighbors) > 0:
                    nn_idx = np.random.choice(available_neighbors)
                    nn = X_min[nn_idx]
                    
                    # Generate synthetic sample (linear interpolation)
                    diff = nn - X_min[danger_idx]
                    gap = np.random.random()
                    synthetic_sample = X_min[danger_idx] + gap * diff
                    synthetic.append(synthetic_sample)
        
        if len(synthetic) == 0:
            return X, y
        
        synthetic = np.array(synthetic)
        
        # Combine original and synthetic
        X_new = np.vstack([X, synthetic])
        y_new = np.hstack([y, np.ones(len(synthetic))])
        
        return X_new, y_new


class FCMBoostOBU(BaseSampler):
    """
    Boosted Overlap-Based Undersampling (Algorithm 3)
    
    The method combines:
    1. BLSMOTE1 to emphasize minority borderline samples
    2. Fuzzy C-Means clustering with 2 clusters on boosted dataset
    3. Adaptive threshold calculation: μth = min(μ̄neg, μ̄pos)
    4. Elimination of majority instances where μineg ≥ μth
    
    Parameters:
    -----------
    k : int, default=5
        Number of neighbors for BLSMOTE1
    m : float, default=2
        Fuzziness parameter for FCM (1 ≤ m ≤ ∞)
    max_iter : int, default=1000
        Maximum iterations for FCM
    error : float, default=1e-5
        Convergence criterion for FCM
    random_state : int, optional
        Random seed for reproducibility
    verbose : bool, default=False
        Whether to print progress information
    """
    
    def __init__(self, k=5, m=2, max_iter=1000, error=1e-5, random_state=None, verbose=False):
        self.k = k
        self.m = m
        self.max_iter = max_iter
        self.error = error
        self.random_state = random_state
        self.verbose = verbose
        self.mu_th_ = None  # Adaptive threshold (computed during fit)

    def fit_resample(self, X, y):
        """
        Apply BoostOBU algorithm
        
        Algorithm Steps:
        1. Apply BLSMOTE1 to emphasize minority borderline
        2. Apply FCM clustering with 2 clusters on boosted dataset
        3. Calculate adaptive threshold: μth = min(μ̄neg, μ̄pos)
        4. Eliminate majority instances where μineg ≥ μth
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values (0 for majority, 1 for minority)
            
        Returns:
        --------
        X_resampled : array, shape (n_samples_new, n_features)
            Resampled feature matrix
        y_resampled : array, shape (n_samples_new,)
            Resampled target values
        """
        if not SKFUZZY_AVAILABLE:
            raise ImportError("scikit-fuzzy is required for FCMBoostOBU. Install with: pip install scikit-fuzzy")
        
        X = np.array(X)
        y = np.array(y)
        
        # Identify minority and majority classes
        unique, counts = np.unique(y, return_counts=True)
        minority_class = unique[np.argmin(counts)]
        majority_class = unique[np.argmax(counts)]
        
        # Convert to binary (0 for majority, 1 for minority)
        y_binary = (y == minority_class).astype(int)
        
        if self.verbose:
            print(f"Original dataset: {len(X)} samples ({np.sum(y_binary==0)} majority, {np.sum(y_binary==1)} minority)")
        
        # Step 1: Apply BLSMOTE1 (Line 2 in Algorithm 3)
        if self.verbose:
            print("Step 1: Applying BLSMOTE1 to emphasize minority borderline...")
        
        blsmote = BLSMOTE1(k=self.k, random_state=self.random_state)
        X_bs, y_bs = blsmote.fit_resample(X, y_binary)
        
        if self.verbose:
            print(f"  After BLSMOTE1: {len(X_bs)} samples ({np.sum(y_bs==0)} majority, {np.sum(y_bs==1)} minority)")
        
        T_neg = X_bs[y_bs == 0]  # Majority class
        T_pos_new = X_bs[y_bs == 1]  # Minority class (includes synthetic)
        
        if len(T_neg) == 0 or len(T_pos_new) == 0:
            # Edge case: no samples in one class
            return X, y
        
        # Step 2: Apply FCM on boosted dataset (Line 3)
        if self.verbose:
            print("Step 2: Applying Fuzzy C-means clustering...")
        
        try:
            cntr, u, u0, d, jm, p, fpc = cmeans(
                X_bs.T, 
                c=2, 
                m=self.m, 
                error=self.error, 
                maxiter=self.max_iter
            )
        except Exception as e:
            if self.verbose:
                print(f"FCM clustering failed: {e}. Returning original data.")
            return X, y
        
        neg_indices = np.where(y_bs == 0)[0]
        pos_indices = np.where(y_bs == 1)[0]
        
        # Identify which cluster is the "positive cluster"
        # (cluster with higher average membership from positive samples)
        u_pos_samples = u[:, pos_indices]
        avg_membership_per_cluster = u_pos_samples.mean(axis=1)
        pos_cluster_idx = np.argmax(avg_membership_per_cluster)
        neg_cluster_idx = 1 - pos_cluster_idx
        
        if self.verbose:
            print(f"  Positive cluster identified as cluster {pos_cluster_idx}")
        
        # Step 3-6: Calculate adaptive threshold (Lines 4-6)
        if self.verbose:
            print("Step 3: Calculating adaptive threshold...")
        
        u_neg = u[:, neg_indices]  # Membership matrix for negative instances
        
        # μ̄neg = mean membership of negative instances in negative cluster
        mu_ineg_neg = u_neg[neg_cluster_idx, :]
        mu_bar_neg = np.mean(mu_ineg_neg)
        
        # μ̄pos = mean membership of negative instances in positive cluster
        mu_ineg_pos = u_neg[pos_cluster_idx, :]
        mu_bar_pos = np.mean(mu_ineg_pos)
        
        # Adaptive threshold: μth = min(μ̄neg, μ̄pos)
        self.mu_th_ = min(mu_bar_neg, mu_bar_pos)
        
        if self.verbose:
            print(f"  μ̄neg = {mu_bar_neg:.4f}")
            print(f"  μ̄pos = {mu_bar_pos:.4f}")
            print(f"  Adaptive threshold μth = {self.mu_th_:.4f}")
        
        # Step 7: Eliminate majority instances (Line 7)
        if self.verbose:
            print("Step 4: Eliminating overlapped majority instances...")
        
        keep_mask = mu_ineg_pos < self.mu_th_
        T_neg_new = T_neg[keep_mask]
        removed_count = len(T_neg) - len(T_neg_new)
        
        if self.verbose:
            print(f"  Removed {removed_count} majority instances ({removed_count/len(T_neg)*100:.1f}%)")
        
        # Step 8: Combine resampled data (Line 8)
        X_resampled = np.vstack([T_neg_new, T_pos_new])
        y_resampled_binary = np.hstack([np.zeros(len(T_neg_new)), np.ones(len(T_pos_new))])
        
        # Convert back to original class labels
        y_resampled = np.where(y_resampled_binary == 1, minority_class, majority_class)
        
        if self.verbose:
            print(f"Final dataset: {len(X_resampled)} samples ({len(T_neg_new)} majority, {len(T_pos_new)} minority)")
            if len(T_pos_new) > 0:
                print(f"Imbalance ratio: {len(T_neg_new)/len(T_pos_new):.2f}")
        
        return X_resampled, y_resampled