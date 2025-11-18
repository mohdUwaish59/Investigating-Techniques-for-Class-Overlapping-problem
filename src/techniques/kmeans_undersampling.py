"""
T5: KMeans-Based Undersampling Methods
Implementation of 4 clustering-based undersampling techniques

Reference: Dkhar et al. (2016). "Evaluating the Effectiveness of Soft K-Means 
in Detecting Overlapping Clusters", ICTCS '16

These clustering algorithms are adapted for undersampling by:
1. Clustering the majority class
2. Removing samples from clusters (especially overlapping regions)
"""

import numpy as np
from typing import Tuple, Optional
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

from .base_sampler import BaseSampler


class HardKMeans:
    """
    Hard K-Means (HKM) / Classical K-Means
    Equation (1) from paper: J = sum_i sum_j ||Oj - xi||^2
    """
    
    def __init__(self, k=3, max_iter=100, epsilon=0.00001, random_state=42):
        """
        Parameters:
        -----------
        k : int
            Number of clusters
        max_iter : int
            Maximum number of iterations
        epsilon : float
            Convergence threshold (from paper: ε = 0.00001)
        random_state : int
            Random seed for reproducibility
        """
        self.k = k
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.iterations = 0
        
    def fit(self, X):
        """
        Fit the HKM model to data X
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        """
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        
        # Randomly initialize k centroids
        random_indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_indices].copy()
        
        for iteration in range(self.max_iter):
            old_centroids = self.centroids.copy()
            
            # Assign each object to nearest cluster (hard assignment)
            distances = cdist(X, self.centroids, metric='euclidean')
            self.labels = np.argmin(distances, axis=1)
            
            # Update centroids: xi = (1/m) * sum(Op) for all Op in Ci
            for i in range(self.k):
                cluster_points = X[self.labels == i]
                if len(cluster_points) > 0:
                    self.centroids[i] = cluster_points.mean(axis=0)
            
            # Check convergence: centroids stabilize
            centroid_shift = np.linalg.norm(self.centroids - old_centroids)
            self.iterations = iteration + 1
            
            if centroid_shift < self.epsilon:
                break
                
        return self
    
    def predict(self, X):
        """Predict cluster labels for data X"""
        distances = cdist(X, self.centroids, metric='euclidean')
        return np.argmin(distances, axis=1)
    
    def get_overlapping_points(self, X, threshold_percentile=10):
        """
        Identify potential overlapping points (for visualization)
        Points with similar distances to multiple centroids
        """
        distances = cdist(X, self.centroids, metric='euclidean')
        sorted_distances = np.sort(distances, axis=1)
        
        # Ratio of closest to second closest distance
        distance_ratio = sorted_distances[:, 0] / (sorted_distances[:, 1] + 1e-10)
        threshold = np.percentile(distance_ratio, threshold_percentile)
        
        overlapping_mask = distance_ratio < threshold
        return overlapping_mask


class FuzzyCMeans:
    """
    Fuzzy C-Means (FCM)
    Equations (2), (3), (4) from paper
    """
    
    def __init__(self, k=3, m=2.0, max_iter=100, epsilon=0.00001, random_state=42):
        """
        Parameters:
        -----------
        k : int
            Number of clusters
        m : float
            Fuzzifier (m' in paper), 1 < m < infinity
        max_iter : int
            Maximum number of iterations
        epsilon : float
            Convergence threshold
        random_state : int
            Random seed
        """
        self.k = k
        self.m = m  # m' in the paper (fuzzifier)
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.random_state = random_state
        self.centroids = None
        self.membership = None  # μij
        self.labels = None
        self.iterations = 0
        
    def _initialize_membership(self, n_samples):
        """Initialize random membership matrix"""
        np.random.seed(self.random_state)
        membership = np.random.rand(n_samples, self.k)
        # Normalize so sum of memberships = 1 for each point
        membership = membership / membership.sum(axis=1, keepdims=True)
        return membership
    
    def _update_centroids(self, X):
        """
        Update centroids using Equation (4):
        xi = (1/ni) * sum_j (μij)^m * Oj
        where ni = sum_j (μij)^m
        """
        membership_powered = self.membership ** self.m
        
        # Calculate ni = sum_j (μij)^m for each cluster
        ni = membership_powered.sum(axis=0)
        
        # Calculate xi = (1/ni) * sum_j (μij)^m * Oj
        self.centroids = np.dot(membership_powered.T, X) / ni[:, np.newaxis]
        
    def _update_membership(self, X):
        """
        Update membership using Equation (3):
        μij = [sum_p (dij/dpj)^(2/(m-1))]^(-1)
        """
        # Calculate distances dij
        distances = cdist(X, self.centroids, metric='euclidean')
        distances = np.fmax(distances, np.finfo(float).eps)  # Avoid division by zero
        
        # Calculate membership using equation (3)
        exponent = 2.0 / (self.m - 1)
        
        # For each point and each cluster
        membership = np.zeros((X.shape[0], self.k))
        
        for i in range(X.shape[0]):
            for j in range(self.k):
                # sum_p (dij/dpj)^(2/(m-1))
                sum_term = np.sum((distances[i, j] / distances[i, :]) ** exponent)
                membership[i, j] = 1.0 / sum_term
        
        return membership
    
    def fit(self, X):
        """Fit the FCM model to data X"""
        n_samples, n_features = X.shape
        
        # Initialize membership matrix
        self.membership = self._initialize_membership(n_samples)
        
        for iteration in range(self.max_iter):
            old_membership = self.membership.copy()
            
            # Update centroids
            self._update_centroids(X)
            
            # Update membership
            self.membership = self._update_membership(X)
            
            # Check convergence
            membership_change = np.linalg.norm(self.membership - old_membership)
            self.iterations = iteration + 1
            
            if membership_change < self.epsilon:
                break
        
        # Assign hard labels (max membership)
        self.labels = np.argmax(self.membership, axis=1)
        
        return self
    
    def predict(self, X):
        """Predict cluster labels for data X"""
        distances = cdist(X, self.centroids, metric='euclidean')
        distances = np.fmax(distances, np.finfo(float).eps)
        
        exponent = 2.0 / (self.m - 1)
        membership = np.zeros((X.shape[0], self.k))
        
        for i in range(X.shape[0]):
            for j in range(self.k):
                sum_term = np.sum((distances[i, j] / distances[i, :]) ** exponent)
                membership[i, j] = 1.0 / sum_term
        
        return np.argmax(membership, axis=1)
    
    def get_overlapping_points(self, X, threshold=0.3):
        """
        Identify overlapping points based on membership values
        Points with high membership to multiple clusters
        """
        # Sort memberships for each point
        sorted_memberships = np.sort(self.membership, axis=1)
        
        # If second highest membership is > threshold, consider it overlapping
        overlapping_mask = sorted_memberships[:, -2] > threshold
        
        return overlapping_mask


class RoughKMeans:
    """
    Rough K-Means (RKM)
    Equations (5), (6) from paper
    Uses lower approximation A(Ci) and upper approximation A̅(Ci)
    """
    
    def __init__(self, k=3, w=0.95, sigma_threshold=0.1, max_iter=100, 
                 epsilon=0.00001, random_state=42):
        """
        Parameters:
        -----------
        k : int
            Number of clusters
        w : float
            Weight for lower approximation (from paper: ω = 0.95)
        sigma_threshold : float
            Threshold σ for boundary region assignment
        max_iter : int
            Maximum number of iterations
        epsilon : float
            Convergence threshold
        random_state : int
            Random seed
        """
        self.k = k
        self.w = w  # ω in the paper
        self.sigma_threshold = sigma_threshold  # σ in the paper
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.random_state = random_state
        self.centroids = None
        self.lower_approx = None  # A(Ci)
        self.upper_approx = None  # A̅(Ci)
        self.boundary = None  # B(Ci) = A̅(Ci) - A(Ci)
        self.labels = None
        self.iterations = 0
        
    def fit(self, X):
        """Fit the RKM model to data X"""
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        
        # Initialize centroids randomly
        random_indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_indices].copy()
        
        # Initialize approximations
        self.lower_approx = [set() for _ in range(self.k)]
        self.upper_approx = [set() for _ in range(self.k)]
        
        for iteration in range(self.max_iter):
            old_centroids = self.centroids.copy()
            
            # Clear previous approximations
            self.lower_approx = [set() for _ in range(self.k)]
            self.upper_approx = [set() for _ in range(self.k)]
            
            # Assign objects to approximations
            distances = cdist(X, self.centroids, metric='euclidean')
            
            for n in range(n_samples):
                # Find closest cluster
                closest_cluster = np.argmin(distances[n])
                min_dist = distances[n, closest_cluster]
                
                # Check if object is in boundary region
                in_boundary = False
                for j in range(self.k):
                    if j != closest_cluster:
                        # |dist(On, xi) - dist(On, xj)| ≤ σ
                        if abs(distances[n, closest_cluster] - distances[n, j]) <= self.sigma_threshold:
                            # Add to upper approximation of both clusters
                            self.upper_approx[closest_cluster].add(n)
                            self.upper_approx[j].add(n)
                            in_boundary = True
                
                if not in_boundary:
                    # Add to both lower and upper approximation
                    self.lower_approx[closest_cluster].add(n)
                    self.upper_approx[closest_cluster].add(n)
            
            # Update centroids using Equation (6)
            for i in range(self.k):
                lower = list(self.lower_approx[i])
                upper = list(self.upper_approx[i])
                boundary = list(set(upper) - set(lower))
                
                if len(lower) > 0 and len(boundary) > 0:
                    # w × α + (1 - w) × β
                    alpha = X[lower].mean(axis=0)
                    beta = X[boundary].mean(axis=0)
                    self.centroids[i] = self.w * alpha + (1 - self.w) * beta
                elif len(lower) > 0:
                    # Only lower approximation exists
                    self.centroids[i] = X[lower].mean(axis=0)
                elif len(boundary) > 0:
                    # Only boundary exists
                    self.centroids[i] = X[boundary].mean(axis=0)
            
            # Check convergence
            centroid_shift = np.linalg.norm(self.centroids - old_centroids)
            self.iterations = iteration + 1
            
            if centroid_shift < self.epsilon:
                break
        
        # Assign hard labels (based on closest centroid)
        distances = cdist(X, self.centroids, metric='euclidean')
        self.labels = np.argmin(distances, axis=1)
        
        # Store boundary points
        self.boundary = [set(self.upper_approx[i]) - set(self.lower_approx[i]) 
                        for i in range(self.k)]
        
        return self
    
    def predict(self, X):
        """Predict cluster labels for data X"""
        distances = cdist(X, self.centroids, metric='euclidean')
        return np.argmin(distances, axis=1)
    
    def get_overlapping_points(self, X):
        """
        Identify overlapping points based on boundary regions
        Points in B(Ci) = A̅(Ci) - A(Ci)
        """
        n_samples = X.shape[0]
        overlapping_mask = np.zeros(n_samples, dtype=bool)
        
        for i in range(self.k):
            boundary_indices = list(self.boundary[i])
            overlapping_mask[boundary_indices] = True
        
        return overlapping_mask


class FuzzyRoughKMeans:
    """
    Fuzzy-Rough K-Means (FRKM)
    Combines fuzzy membership with rough set approximations
    As described in Section 3.4 of the paper
    """
    
    def __init__(self, k=3, m=2.0, w=0.95, sigma_threshold=0.1, 
                 max_iter=100, epsilon=0.00001, random_state=42):
        """
        Parameters:
        -----------
        k : int
            Number of clusters
        m : float
            Fuzzifier
        w : float
            Weight for lower approximation
        sigma_threshold : float
            Threshold for boundary region
        max_iter : int
            Maximum number of iterations
        epsilon : float
            Convergence threshold
        random_state : int
            Random seed
        """
        self.k = k
        self.m = m
        self.w = w
        self.sigma_threshold = sigma_threshold
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.random_state = random_state
        self.centroids = None
        self.membership = None
        self.lower_approx = None
        self.upper_approx = None
        self.boundary = None
        self.labels = None
        self.iterations = 0
        
    def _initialize_membership(self, n_samples):
        """Initialize random membership matrix"""
        np.random.seed(self.random_state)
        membership = np.random.rand(n_samples, self.k)
        membership = membership / membership.sum(axis=1, keepdims=True)
        return membership
    
    def _update_membership(self, X):
        """Update membership using fuzzy c-means formula"""
        distances = cdist(X, self.centroids, metric='euclidean')
        distances = np.fmax(distances, np.finfo(float).eps)
        
        exponent = 2.0 / (self.m - 1)
        membership = np.zeros((X.shape[0], self.k))
        
        for i in range(X.shape[0]):
            for j in range(self.k):
                sum_term = np.sum((distances[i, j] / distances[i, :]) ** exponent)
                membership[i, j] = 1.0 / sum_term
        
        return membership
    
    def fit(self, X):
        """Fit the FRKM model to data X"""
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        
        # Initialize centroids and membership
        random_indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_indices].copy()
        self.membership = self._initialize_membership(n_samples)
        
        for iteration in range(self.max_iter):
            old_centroids = self.centroids.copy()
            
            # Clear previous approximations
            self.lower_approx = [set() for _ in range(self.k)]
            self.upper_approx = [set() for _ in range(self.k)]
            
            # Assign objects to approximations (rough set part)
            distances = cdist(X, self.centroids, metric='euclidean')
            
            for n in range(n_samples):
                closest_cluster = np.argmin(distances[n])
                in_boundary = False
                
                for j in range(self.k):
                    if j != closest_cluster:
                        if abs(distances[n, closest_cluster] - distances[n, j]) <= self.sigma_threshold:
                            self.upper_approx[closest_cluster].add(n)
                            self.upper_approx[j].add(n)
                            in_boundary = True
                
                if not in_boundary:
                    self.lower_approx[closest_cluster].add(n)
                    self.upper_approx[closest_cluster].add(n)
            
            # Update centroids using fuzzy-rough formula
            for i in range(self.k):
                lower = list(self.lower_approx[i])
                upper = list(self.upper_approx[i])
                boundary = list(set(upper) - set(lower))
                
                if len(lower) > 0 and len(boundary) > 0:
                    # χ for lower approximation with fuzzy membership
                    membership_powered_lower = self.membership[lower, i] ** self.m
                    chi = np.sum(membership_powered_lower[:, np.newaxis] * X[lower], axis=0) / len(lower)
                    
                    # ψ for boundary with fuzzy membership
                    membership_powered_boundary = self.membership[boundary, i] ** self.m
                    psi = np.sum(membership_powered_boundary[:, np.newaxis] * X[boundary], axis=0) / len(boundary)
                    
                    self.centroids[i] = self.w * chi + (1 - self.w) * psi
                    
                elif len(lower) > 0:
                    membership_powered = self.membership[lower, i] ** self.m
                    self.centroids[i] = np.sum(membership_powered[:, np.newaxis] * X[lower], axis=0) / len(lower)
                    
                elif len(boundary) > 0:
                    membership_powered = self.membership[boundary, i] ** self.m
                    self.centroids[i] = np.sum(membership_powered[:, np.newaxis] * X[boundary], axis=0) / len(boundary)
            
            # Update membership (fuzzy part)
            self.membership = self._update_membership(X)
            
            # Check convergence
            centroid_shift = np.linalg.norm(self.centroids - old_centroids)
            self.iterations = iteration + 1
            
            if centroid_shift < self.epsilon:
                break
        
        # Assign hard labels
        self.labels = np.argmax(self.membership, axis=1)
        
        # Store boundary points
        self.boundary = [set(self.upper_approx[i]) - set(self.lower_approx[i]) 
                        for i in range(self.k)]
        
        return self
    
    def predict(self, X):
        """Predict cluster labels for data X"""
        distances = cdist(X, self.centroids, metric='euclidean')
        distances = np.fmax(distances, np.finfo(float).eps)
        
        exponent = 2.0 / (self.m - 1)
        membership = np.zeros((X.shape[0], self.k))
        
        for i in range(X.shape[0]):
            for j in range(self.k):
                sum_term = np.sum((distances[i, j] / distances[i, :]) ** exponent)
                membership[i, j] = 1.0 / sum_term
        
        return np.argmax(membership, axis=1)
    
    def get_overlapping_points(self, X):
        """
        Identify overlapping points using both fuzzy membership and boundary
        """
        n_samples = X.shape[0]
        
        # Boundary-based overlapping
        overlapping_mask = np.zeros(n_samples, dtype=bool)
        for i in range(self.k):
            boundary_indices = list(self.boundary[i])
            overlapping_mask[boundary_indices] = True
        
        # Also consider fuzzy membership
        sorted_memberships = np.sort(self.membership, axis=1)
        fuzzy_overlap = sorted_memberships[:, -2] > 0.3
        
        # Combine both criteria
        overlapping_mask = overlapping_mask | fuzzy_overlap
        
        return overlapping_mask


# Wrapper classes for undersampling using clustering
class KMeansUndersamplingWrapper(BaseSampler):
    """Base wrapper for KMeans-based undersampling"""
    
    def __init__(self, clustering_class, method_name, **kwargs):
        self.clustering_class = clustering_class
        self.method_name = method_name
        self.kwargs = kwargs
        self.clusterer = None
        self.stats_ = {}
        self.verbose = kwargs.get('verbose', True)
    
    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply clustering-based undersampling"""
        X = np.array(X)
        y = np.array(y)
        
        # Separate classes
        maj_mask = y == 0
        min_mask = y == 1
        X_maj = X[maj_mask]
        y_maj = y[maj_mask]
        X_min = X[min_mask]
        y_min = y[min_mask]
        
        self.stats_['original'] = {
            'n_majority': len(X_maj),
            'n_minority': len(X_min)
        }
        
        if self.verbose:
            print(f"{self.method_name}: Original (Maj={len(X_maj)}, Min={len(X_min)})")
            print(f"{self.method_name}: Clustering majority class with k={self.kwargs.get('k', 3)}")
        
        # Cluster majority class
        self.clusterer = self.clustering_class(**self.kwargs)
        self.clusterer.fit(X_maj)
        
        # Identify overlapping points
        overlapping_mask = self.clusterer.get_overlapping_points(X_maj)
        
        # Remove overlapping samples
        keep_mask = ~overlapping_mask
        X_maj_res = X_maj[keep_mask]
        y_maj_res = y_maj[keep_mask]
        
        # Combine with minority class
        X_resampled = np.vstack([X_maj_res, X_min])
        y_resampled = np.hstack([y_maj_res, y_min])
        
        self.stats_['final'] = {
            'n_majority': len(X_maj_res),
            'n_minority': len(X_min),
            'n_removed': len(X_maj) - len(X_maj_res),
            'iterations': self.clusterer.iterations
        }
        
        if self.verbose:
            print(f"{self.method_name}: Converged in {self.clusterer.iterations} iterations")
            print(f"{self.method_name}: Removed {self.stats_['final']['n_removed']} majority samples")
            print(f"{self.method_name}: Final (Maj={len(X_maj_res)}, Min={len(X_min)})")
        
        return X_resampled, y_resampled


# Specific wrapper classes for each method
class HKMUndersampling(KMeansUndersamplingWrapper):
    """T5.1: Hard K-Means Undersampling"""
    def __init__(self, k=3, max_iter=100, epsilon=0.00001, 
                 threshold_percentile=10, random_state=None, verbose=True):
        super().__init__(
            HardKMeans, "HKM",
            k=k, max_iter=max_iter, epsilon=epsilon,
            random_state=random_state if random_state else 42
        )
        self.threshold_percentile = threshold_percentile
        self.verbose = verbose
    
    def fit_resample(self, X, y):
        # Override to pass threshold_percentile
        result = super().fit_resample(X, y)
        return result


class FCMUndersampling(KMeansUndersamplingWrapper):
    """T5.2: Fuzzy C-Means Undersampling"""
    def __init__(self, k=3, m=2.0, max_iter=100, epsilon=0.00001,
                 membership_threshold=0.3, random_state=None, verbose=True):
        super().__init__(
            FuzzyCMeans, "FCM",
            k=k, m=m, max_iter=max_iter, epsilon=epsilon,
            random_state=random_state if random_state else 42
        )
        self.membership_threshold = membership_threshold
        self.verbose = verbose


class RKMUndersampling(KMeansUndersamplingWrapper):
    """T5.3: Rough K-Means Undersampling"""
    def __init__(self, k=3, w=0.95, sigma_threshold=0.1, max_iter=100,
                 epsilon=0.00001, random_state=None, verbose=True):
        super().__init__(
            RoughKMeans, "RKM",
            k=k, w=w, sigma_threshold=sigma_threshold,
            max_iter=max_iter, epsilon=epsilon,
            random_state=random_state if random_state else 42
        )
        self.verbose = verbose


class FRKMUndersampling(KMeansUndersamplingWrapper):
    """T5.4: Fuzzy-Rough K-Means Undersampling"""
    def __init__(self, k=3, m=2.0, w=0.95, sigma_threshold=0.1,
                 max_iter=100, epsilon=0.00001, random_state=None, verbose=True):
        super().__init__(
            FuzzyRoughKMeans, "FRKM",
            k=k, m=m, w=w, sigma_threshold=sigma_threshold,
            max_iter=max_iter, epsilon=epsilon,
            random_state=random_state if random_state else 42
        )
        self.verbose = verbose


# Main wrapper class for easy access
class KMeansUndersampling(BaseSampler):
    """
    T5: KMeans-Based Undersampling
    
    Wrapper class providing access to all 4 KMeans variants:
    - HKM: Hard K-Means
    - FCM: Fuzzy C-Means
    - RKM: Rough K-Means
    - FRKM: Fuzzy-Rough K-Means
    """
    
    def __init__(self, method='HKM', k=3, m=2.0, w=0.95, sigma_threshold=0.1,
                 max_iter=100, epsilon=0.00001, random_state=None, verbose=True):
        """
        Parameters:
        -----------
        method : str, default='HKM'
            Which method to use: 'HKM', 'FCM', 'RKM', 'FRKM'
        k : int, default=3
            Number of clusters
        m : float, default=2.0
            Fuzzifier for FCM and FRKM
        w : float, default=0.95
            Weight for RKM and FRKM
        sigma_threshold : float, default=0.1
            Threshold for RKM and FRKM
        max_iter : int, default=100
            Maximum iterations
        epsilon : float, default=0.00001
            Convergence threshold
        random_state : int, optional
            Random seed
        verbose : bool, default=True
            Print progress
        """
        self.method = method
        self.k = k
        self.m = m
        self.w = w
        self.sigma_threshold = sigma_threshold
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.random_state = random_state
        self.verbose = verbose
        
        # Initialize the appropriate method
        if method == 'HKM':
            self.sampler = HKMUndersampling(
                k=k, max_iter=max_iter, epsilon=epsilon,
                random_state=random_state, verbose=verbose
            )
        elif method == 'FCM':
            self.sampler = FCMUndersampling(
                k=k, m=m, max_iter=max_iter, epsilon=epsilon,
                random_state=random_state, verbose=verbose
            )
        elif method == 'RKM':
            self.sampler = RKMUndersampling(
                k=k, w=w, sigma_threshold=sigma_threshold,
                max_iter=max_iter, epsilon=epsilon,
                random_state=random_state, verbose=verbose
            )
        elif method == 'FRKM':
            self.sampler = FRKMUndersampling(
                k=k, m=m, w=w, sigma_threshold=sigma_threshold,
                max_iter=max_iter, epsilon=epsilon,
                random_state=random_state, verbose=verbose
            )
        else:
            raise ValueError(f"Unknown method: {method}. Choose from: HKM, FCM, RKM, FRKM")
        
        self.stats_ = {}
    
    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply the selected KMeans-based undersampling method"""
        X_res, y_res = self.sampler.fit_resample(X, y)
        self.stats_ = self.sampler.stats_
        return X_res, y_res
