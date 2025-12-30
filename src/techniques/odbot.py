"""
ODBOT: Outlier Detection-Based Oversampling Technique

Implementation based on Ibrahim, M. H. (2021). Neural Computing and Applications.
This implementation follows the exact methodology described in the paper.

Reference:
Ibrahim, M. H. (2021). ODBOT: Outlier detection-based oversampling technique for 
imbalanced datasets learning. Neural Computing and Applications, 33(22), 15781-15806.
"""

import numpy as np
from sklearn.cluster import KMeans
import warnings
from .base_sampler import BaseSampler

warnings.filterwarnings('ignore')


class ODBOT(BaseSampler):
    """
    Outlier Detection-Based Oversampling Technique (ODBOT)
    
    As described in the paper, ODBOT follows 4 main steps:
    1. Determining the number of synthetic samples (NSS)
    2. Finding dissimilarity relationships through clustering
    3. Detecting outliers using SMCD (Sum of Minority Cluster Distance)
    4. Generating synthetic samples within boundaries of best cluster
    
    Parameters:
    -----------
    k : int, default=2
        Number of clusters (must be > 1 as per paper)
    percentage : float or None
        Percentage value for oversampling. If None, calculated automatically
    random_state : int, default=42
        Random state for reproducibility
    verbose : bool, default=False
        Whether to print progress information
    """
    
    def __init__(self, k=2, percentage=None, random_state=42, verbose=False):
        if k <= 1:
            raise ValueError("k must be greater than 1 as per ODBOT methodology")
        
        self.k = k
        self.percentage = percentage
        self.random_state = random_state
        self.verbose = verbose
        
        np.random.seed(random_state)

    def fit_resample(self, X, y):
        """
        Resample the dataset using ODBOT methodology.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
            
        Returns:
        --------
        X_resampled : array-like
            Resampled features
        y_resampled : array-like
            Resampled target values
        """
        X = np.array(X)
        y = np.array(y)
        
        # Identify majority and minority classes
        unique_classes, counts = np.unique(y, return_counts=True)
        
        if self.verbose:
            print(f"Original dataset: {len(X)} samples")
            for cls, count in zip(unique_classes, counts):
                print(f"  Class {cls}: {count} samples")
        
        # For binary classification: identify majority and minority
        if len(unique_classes) == 2:
            majority_class = unique_classes[np.argmax(counts)]
            minority_class = unique_classes[np.argmin(counts)]
            X_resampled, y_resampled = self._resample_binary(X, y, majority_class, minority_class)
        else:
            # For multi-class: treat each non-majority class as minority
            majority_class = unique_classes[np.argmax(counts)]
            X_resampled = X.copy()
            y_resampled = y.copy()
            
            for minority_class in unique_classes:
                if minority_class != majority_class:
                    if self.verbose:
                        print(f"\nProcessing minority class {minority_class}...")
                    X_resampled, y_resampled = self._resample_binary(
                        X_resampled, y_resampled, majority_class, minority_class
                    )
        
        if self.verbose:
            print(f"\nFinal resampled dataset: {len(X_resampled)} samples")
            unique_final, counts_final = np.unique(y_resampled, return_counts=True)
            for cls, count in zip(unique_final, counts_final):
                print(f"  Class {cls}: {count} samples")
        
        return X_resampled, y_resampled

    def _resample_binary(self, X, y, majority_class, minority_class):
        """
        Resample for one minority class against majority class.
        Follows the exact 4-step ODBOT algorithm from the paper.
        """
        # Extract majority and minority samples
        X_maj = X[y == majority_class]
        X_min = X[y == minority_class]
        
        if self.verbose:
            print(f"  Majority class {majority_class}: {len(X_maj)} samples")
            print(f"  Minority class {minority_class}: {len(X_min)} samples")
        
        # STEP 1: Determining the number of synthetic samples (NSS)
        # Equation (1) and (2) from paper
        NSS = self._calculate_nss(X_maj, X_min)
        
        if self.verbose:
            print(f"  Number of synthetic samples to generate: {NSS}")
        
        if NSS == 0:
            return X, y
        
        # STEP 2: Finding the dissimilarity relationships
        # Using clustering (paper uses WBBA-KM, we use standard K-means as approximation)
        if self.verbose:
            print("  Step 2: Clustering data...")
        
        maj_clusters, maj_centers = self._cluster_data(X_maj)
        min_clusters, min_centers = self._cluster_data(X_min)
        
        # STEP 3: Detecting outliers
        # Calculate SMCD for each minority cluster using Equation (3)
        if self.verbose:
            print("  Step 3: Finding best cluster using SMCD...")
        
        best_cluster_idx = self._find_best_cluster(min_centers, maj_centers)
        
        # Get samples from the best cluster
        best_cluster_samples = X_min[min_clusters == best_cluster_idx]
        
        if len(best_cluster_samples) == 0:
            # Fallback: use all minority samples if best cluster is empty
            best_cluster_samples = X_min
            if self.verbose:
                print("    Warning: Best cluster is empty, using all minority samples")
        
        if self.verbose:
            print(f"    Best cluster {best_cluster_idx} has {len(best_cluster_samples)} samples")
        
        # STEP 4: Generating synthetic samples
        # Using Equation (5) from paper
        if self.verbose:
            print("  Step 4: Generating synthetic samples...")
        
        synthetic_samples = self._generate_synthetic_samples(best_cluster_samples, NSS)
        
        # Combine original and synthetic data
        X_resampled = np.vstack([X, synthetic_samples])
        y_resampled = np.hstack([y, np.full(NSS, minority_class)])
        
        return X_resampled, y_resampled

    def _calculate_nss(self, X_maj, X_min):
        """
        Step 1: Calculate Number of Synthetic Samples (NSS)
        
        From paper Equation (1) and (2):
        NSS = percentage_value × |minority_class| / 100
        percentage_value = (|majority_class| / |minority_class| - 1) × 100
        """
        n_majority = len(X_maj)
        n_minority = len(X_min)
        
        if self.percentage is None:
            # Equation (2) from paper
            percentage_value = ((n_majority / n_minority) - 1) * 100
        else:
            percentage_value = self.percentage
        
        # Equation (1) from paper
        NSS = int((percentage_value * n_minority) / 100)
        
        return NSS

    def _cluster_data(self, X):
        """
        Step 2: Cluster data using k-means
        
        Paper uses WBBA-KM (Weight-Based Bat Algorithm with K-means)
        For practical implementation, we use standard K-means
        
        Returns:
        --------
        labels : cluster assignments for each sample
        centers : cluster centers
        """
        n_samples = len(X)
        
        # If fewer samples than clusters, reduce k
        actual_k = min(self.k, n_samples)
        
        if actual_k < 2:
            # If only 1 cluster possible, return dummy clustering
            return np.zeros(n_samples, dtype=int), X.mean(axis=0, keepdims=True)
        
        kmeans = KMeans(n_clusters=actual_k, random_state=self.random_state, n_init=10)
        labels = kmeans.fit_predict(X)
        centers = kmeans.cluster_centers_
        
        return labels, centers

    def _find_best_cluster(self, min_centers, maj_centers):
        """
        Step 3: Find the best cluster by calculating SMCD
        
        From paper Equation (3) and (4):
        SMCD_{i,j} = Σ |MinCen_{i,j} - MajCen_l| for l=1 to k
        Best cluster = Max(SMCD_{i,1}, SMCD_{i,2}, ..., SMCD_{i,k})
        
        The maximum SMCD indicates the cluster with maximum dissimilarity
        from majority class (contains fewer outliers).
        """
        n_min_clusters = len(min_centers)
        smcd_values = np.zeros(n_min_clusters)
        
        for j in range(n_min_clusters):
            # Calculate sum of distances from this minority center to all majority centers
            # Equation (3) from paper
            smcd = 0
            for l in range(len(maj_centers)):
                smcd += np.linalg.norm(min_centers[j] - maj_centers[l])
            smcd_values[j] = smcd
        
        # Equation (4): Maximum SMCD indicates best cluster
        best_cluster_idx = np.argmax(smcd_values)
        
        if self.verbose:
            print(f"    SMCD values: {smcd_values}")
            print(f"    Best cluster index: {best_cluster_idx} (SMCD: {smcd_values[best_cluster_idx]:.3f})")
        
        return best_cluster_idx

    def _generate_synthetic_samples(self, best_cluster_samples, NSS):
        """
        Step 4: Generate synthetic samples
        
        From paper Equation (5):
        S = r × (max_minor - min_minor) + min_minor
        
        where:
        - r is random number between 0 and 1
        - min_minor is minimum value in best cluster
        - max_minor is maximum value in best cluster
        """
        n_features = best_cluster_samples.shape[1]
        synthetic_samples = np.zeros((NSS, n_features))
        
        # Calculate min and max for each feature in the best cluster
        min_minor = best_cluster_samples.min(axis=0)
        max_minor = best_cluster_samples.max(axis=0)
        
        # Generate NSS synthetic samples using Equation (5)
        for i in range(NSS):
            # r is random number between 0 and 1
            r = np.random.random(n_features)
            # Equation (5): S = r × (max_minor - min_minor) + min_minor
            synthetic_samples[i] = r * (max_minor - min_minor) + min_minor
        
        return synthetic_samples