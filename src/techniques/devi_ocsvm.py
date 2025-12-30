"""
Implementation of: "Learning in presence of class imbalance and class overlapping by using one-class SVM and undersampling technique"
Authors: Debashree Devi, Saroj K. Biswas & Biswajit Purkayastha (2019)
Connection Science, DOI: 10.1080/09540091.2018.1560394

This implements the exact methodology as described in the paper.
"""

import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from .base_sampler import BaseSampler


class DeviOCSVM(BaseSampler):
    """
    Implementation of the proposed method from Devi et al. (2019).
    
    The method combines:
    1. One-class SVM for overlapping region detection
    2. Abnormal Tomek-Link (ATL) pairs detection
    3. Sparse neighbourhood estimation
    4. Normal Tomek-Link (NTL) pairs with redundancy and intra-cluster significance
    
    Parameters:
    -----------
    nu : float, default=0.5
        Parameter for one-class SVM (ϑ in the paper)
        Controls the fraction of outliers (ranges from 0.3, 0.5, 0.7 in paper)
    K1 : int, default=1
        K-NN of minority instance (set to 1 in paper)
    K2 : int, default=5
        K-NN of majority instances (set to 5 in paper)
    K3 : int, default=1
        K-NN of overlapped majority instance (set to 1 in paper)
    kernel : str, default='rbf'
        Kernel type for one-class SVM
    gamma : str or float, default='scale'
        Kernel coefficient for one-class SVM
    """
    
    def __init__(self, nu=0.5, K1=1, K2=5, K3=1, kernel='rbf', gamma='scale', verbose=False):
        self.nu = nu
        self.K1 = K1
        self.K2 = K2
        self.K3 = K3
        self.kernel = kernel
        self.gamma = gamma
        self.verbose = verbose
        
        # Storage for intermediate results
        self.ovr_regmin = None
        self.ovr_regmaj = None
        self.nov_regmin = None
        self.nov_regmaj = None
        self.ATL = None
        self.NTL = None
        self.spNN = None

    def _detect_overlapping_region(self, X_min, X_maj):
        """
        Stage 1: Detection of overlapping region using one-class SVM
        As described in Section 4.1.1 and Algorithm 1 of the paper.
        
        Parameters:
        -----------
        X_min : array-like, shape (n_minority, n_features)
            Minority class instances
        X_maj : array-like, shape (n_majority, n_features)
            Majority class instances
            
        Returns:
        --------
        ovr_regmin : array-like
            Overlapping minority instances (outliers)
        ovr_regmaj : array-like
            Overlapping majority instances (outliers)
        nov_regmin : array-like
            Novel (non-overlapping) minority instances
        nov_regmaj : array-like
            Novel (non-overlapping) majority instances
        """
        # Step 1: Apply one-class SVM to majority class
        ocsvm_maj = OneClassSVM(nu=self.nu, kernel=self.kernel, gamma=self.gamma)
        ocsvm_maj.fit(X_maj)
        
        # Step 2: Predict outliers for majority class
        # f(x) < 0 means outlier (overlapping)
        # f(x) > 0 means novel (non-overlapping)
        pred_maj = ocsvm_maj.predict(X_maj)
        ovr_regmaj_mask = pred_maj == -1  # Outliers (Equation 3)
        nov_regmaj_mask = pred_maj == 1   # Novel instances (Equation 4)
        
        self.ovr_regmaj = X_maj[ovr_regmaj_mask]
        self.nov_regmaj = X_maj[nov_regmaj_mask]
        
        # Step 3: Apply one-class SVM to minority class
        ocsvm_min = OneClassSVM(nu=self.nu, kernel=self.kernel, gamma=self.gamma)
        ocsvm_min.fit(X_min)
        
        # Step 4: Predict outliers for minority class
        pred_min = ocsvm_min.predict(X_min)
        ovr_regmin_mask = pred_min == -1  # Outliers (Equation 5)
        nov_regmin_mask = pred_min == 1   # Novel instances (Equation 6)
        
        self.ovr_regmin = X_min[ovr_regmin_mask]
        self.nov_regmin = X_min[nov_regmin_mask]
        
        # Create indices for tracking
        self.ovr_regmin_idx = np.where(ovr_regmin_mask)[0]
        self.ovr_regmaj_idx = np.where(ovr_regmaj_mask)[0]
        self.nov_regmin_idx = np.where(nov_regmin_mask)[0]
        self.nov_regmaj_idx = np.where(nov_regmaj_mask)[0]
        
        return self.ovr_regmin, self.ovr_regmaj, self.nov_regmin, self.nov_regmaj

    def _compute_tomek_links(self, X_min, X_maj):
        """
        Compute Tomek-link pairs as described in Section 3.2 and Algorithm 2.
        
        Two instances xi and xj form a Tomek-link pair if there is no other instance xk such that 
        d(xi, xk) ≤ d(xk, xj), provided class(xi) ≠ class(xj).
        
        Parameters:
        -----------
        X_min : array-like
            Minority class instances
        X_maj : array-like
            Majority class instances
            
        Returns:
        --------
        TL : list of tuples
            Tomek-link pairs (minority_idx, majority_idx)
        """
        # Step 1: For ∀ xi ∈ Smin, determine its K1-NN from Sn×l (K1=1)
        # Step 2: Determine NNmaj (nearest neighbor from majority class)
        TL = []
        
        for i, x_min in enumerate(X_min):
            # Find nearest neighbor in majority class
            min_dist = float('inf')
            nn_maj_idx = -1
            
            for j, x_maj in enumerate(X_maj):
                dist = euclidean(x_min, x_maj)
                if dist < min_dist:
                    min_dist = dist
                    nn_maj_idx = j
            
            # Verify it's a Tomek-link (no closer instance exists)
            # Check if x_maj's nearest neighbor is x_min
            x_maj = X_maj[nn_maj_idx]
            
            # Find nearest neighbor of x_maj in minority class
            min_dist_reverse = float('inf')
            nn_min_idx = -1
            
            for k, x_min_check in enumerate(X_min):
                dist = euclidean(x_maj, x_min_check)
                if dist < min_dist_reverse:
                    min_dist_reverse = dist
                    nn_min_idx = k
            
            # If they are mutual nearest neighbors, it's a Tomek-link
            if nn_min_idx == i:
                TL.append((i, nn_maj_idx))
        
        return TL

    def _detect_abnormal_tomek_links(self, TL, X_min, X_maj):
        """
        Step 3-4 of Algorithm 2: Extract Abnormal Tomek-Link pairs (ATL)
        
        ATL pairs are Tomek-link pairs where BOTH associating instances reside in the overlapping region.
        ATL = {(xc, xd) | xc, xd ∈ TL; xc ∈ ovr_regmin, xd ∈ ovr_regmaj}
        (Equation 8)
        """
        ATL = []
        
        for min_idx, maj_idx in TL:
            # Check if minority instance is in overlapping region
            min_in_ovr = min_idx in self.ovr_regmin_idx
            
            # Check if majority instance is in overlapping region
            maj_in_ovr = maj_idx in self.ovr_regmaj_idx
            
            # Both must be in overlapping region
            if min_in_ovr and maj_in_ovr:
                ATL.append((min_idx, maj_idx))
        
        self.ATL = ATL
        return ATL

    def _update_after_atl_removal(self, ATL, X_min, X_maj):
        """
        Steps 4-6 of Algorithm 2: Update data after ATL removal
        
        Remove both minority and majority instances from ATL pairs.
        Update ovr_regmin, ovr_regmaj, Smin, Smaj, and TL.
        """
        # Extract indices to remove
        atl_min_indices = [pair[0] for pair in ATL]
        atl_maj_indices = [pair[1] for pair in ATL]
        
        # Update minority set (Equation 12)
        mask_min = np.ones(len(X_min), dtype=bool)
        mask_min[atl_min_indices] = False
        X_min_updated = X_min[mask_min]
        
        # Update majority set (Equation 13)
        mask_maj = np.ones(len(X_maj), dtype=bool)
        mask_maj[atl_maj_indices] = False
        X_maj_updated = X_maj[mask_maj]
        
        # Update overlapping regions (Equations 9, 10)
        ovr_min_mask = np.ones(len(self.ovr_regmin), dtype=bool)
        ovr_maj_mask = np.ones(len(self.ovr_regmaj), dtype=bool)
        
        # Remove ATL instances from overlapping regions
        for min_idx in atl_min_indices:
            if min_idx in self.ovr_regmin_idx:
                local_idx = np.where(self.ovr_regmin_idx == min_idx)[0]
                if len(local_idx) > 0:
                    ovr_min_mask[local_idx[0]] = False
        
        for maj_idx in atl_maj_indices:
            if maj_idx in self.ovr_regmaj_idx:
                local_idx = np.where(self.ovr_regmaj_idx == maj_idx)[0]
                if len(local_idx) > 0:
                    ovr_maj_mask[local_idx[0]] = False
        
        self.ovr_regmin = self.ovr_regmin[ovr_min_mask]
        self.ovr_regmaj = self.ovr_regmaj[ovr_maj_mask]
        
        # Recompute TL' (Equation 14)
        TL_prime = self._compute_tomek_links(X_min_updated, X_maj_updated)
        
        return X_min_updated, X_maj_updated, TL_prime

    def _estimate_sparse_neighbourhood(self, ovrout_maj, X_min, X_maj):
        """
        Steps 8-9 of Algorithm 2: Detect sparse neighbourhood
        
        Sparse neighbourhood is K-NN estimation where most neighbors have different class labels.
        For ∀xovr_d ∈ ovrout_maj:
        - Determine K2-NN from reg_ovr (K2=5)
        - Calculate NNhits (neighbors with same class)
        - Calculate NNmiss (neighbors with different class)
        - If NNmiss > NNhits, instance has sparse neighbourhood
        
        Returns instances to remove (Equation 16)
        """
        spNN = []
        spNN_indices = []
        
        # Combine overlapping regions for neighbourhood search
        reg_ovr = np.vstack([self.ovr_regmin, self.ovr_regmaj])
        
        for idx, x_ovr in enumerate(ovrout_maj):
            # Step 9(a): Determine K2-NN from reg_ovr
            nbrs = NearestNeighbors(n_neighbors=min(self.K2 + 1, len(reg_ovr)))
            nbrs.fit(reg_ovr)
            distances, indices = nbrs.kneighbors([x_ovr])
            
            # Exclude self if present
            knn_indices = indices[0][1:self.K2 + 1] if len(indices[0]) > 1 else indices[0]
            
            # Step 9(b-c): Determine NNhits and NNmiss
            nn_hits = 0  # Neighbors from majority class
            nn_miss = 0  # Neighbors from minority class
            
            for knn_idx in knn_indices:
                # Check if neighbor is from minority or majority class
                if knn_idx < len(self.ovr_regmin):
                    nn_miss += 1  # Minority class
                else:
                    nn_hits += 1  # Majority class
            
            # Step 9(d-e): If NNmiss > NNhits, add to sparse neighbourhood
            if nn_miss > nn_hits:
                spNN.append(x_ovr)
                spNN_indices.append(idx)
        
        self.spNN = np.array(spNN) if len(spNN) > 0 else np.array([]).reshape(0, ovrout_maj.shape[1])
        return self.spNN, spNN_indices

    def _detect_normal_tomek_links(self, TL, X_min, X_maj):
        """
        Step 9 (Case 3) of Algorithm 2: Detect Normal Tomek-Link pairs
        
        NTL pairs are Tomek-link pairs where BOTH instances reside OUTSIDE the overlapping region.
        NTL = {(x'c, x'd) | x'c ← xc, x'd ← xc, x'c, x'd ∉ reg_ovr}
        (Equation 21)
        """
        NTL = []
        
        for min_idx, maj_idx in TL:
            # Check if minority instance is NOT in overlapping region
            min_not_in_ovr = min_idx not in self.ovr_regmin_idx
            
            # Check if majority instance is NOT in overlapping region
            maj_not_in_ovr = maj_idx not in self.ovr_regmaj_idx
            
            # Both must be outside overlapping region
            if min_not_in_ovr and maj_not_in_ovr:
                NTL.append((min_idx, maj_idx))
        
        self.NTL = NTL
        return NTL

    def _check_redundancy(self, NTL, X_maj):
        """
        Step 10 of Algorithm 2: Redundancy check
        
        For ∀x'd ∈ NTL, determine its redundant pair xe from S'maj
        using K3-NN rule (K3=1).
        REDN = {(x'd, xe) | x'd ∈ NTL, S'maj; xe ∈ S'maj; d(x'd, xe) is minimum}
        (Equation 22)
        """
        REDN = []
        
        # Extract majority indices from NTL
        ntl_maj_indices = [pair[1] for pair in NTL]
        
        for ntl_maj_idx in ntl_maj_indices:
            x_d = X_maj[ntl_maj_idx]
            
            # Find nearest neighbor in majority class (excluding self)
            min_dist = float('inf')
            redundant_idx = -1
            
            for j, x_e in enumerate(X_maj):
                if j != ntl_maj_idx:
                    dist = euclidean(x_d, x_e)
                    if dist < min_dist:
                        min_dist = dist
                        redundant_idx = j
            
            if redundant_idx != -1:
                REDN.append((ntl_maj_idx, redundant_idx))
        
        return REDN

    def _compute_intra_cluster_significance(self, X_maj):
        """
        Steps 11-12 of Algorithm 2: Check intra-cluster significance
        
        Compute centroid of S'maj (Equation 23):
        cenmaj = (1/q) * Σ(x'd)
        
        Then compute intra-centroid distance for each instance (Equation 24):
        dintra_cen(x'd) = (1/q) * √[Σ((x'd - cenmaj)²)]
        """
        # Compute centroid (Equation 23)
        cen_maj = np.mean(X_maj, axis=0)
        
        # Compute intra-centroid distances (Equation 24)
        dintra_cen = {}
        for idx, x_d in enumerate(X_maj):
            dist = euclidean(x_d, cen_maj)
            dintra_cen[idx] = dist
        
        return cen_maj, dintra_cen

    def _select_least_significant_redundant(self, REDN, dintra_cen):
        """
        Step 13 of Algorithm 2: Detection of least significant redundant instance
        
        For each (x'd, xe) pair:
        - If dintra_cen(x'd) > dintra_cen(xe), then x'd is less significant
        - Select less significant instance for elimination
        """
        to_remove = []
        
        for x_d_idx, x_e_idx in REDN:
            # Compare intra-centroid distances
            if dintra_cen[x_d_idx] > dintra_cen[x_e_idx]:
                to_remove.append(x_d_idx)
            else:
                to_remove.append(x_e_idx)
        
        return list(set(to_remove))  # Remove duplicates

    def fit_resample(self, X, y):
        """
        Main method to fit and resample the dataset.
        
        Implements the complete algorithm as described in Section 4 of the paper.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target labels (binary: 0 for majority, 1 for minority)
            
        Returns:
        --------
        X_resampled : array-like
            Resampled feature matrix
        y_resampled : array-like
            Resampled target labels
        """
        # Identify minority and majority classes
        unique, counts = np.unique(y, return_counts=True)
        minority_class = unique[np.argmin(counts)]
        majority_class = unique[np.argmax(counts)]
        
        # Separate minority and majority instances
        X_min = X[y == minority_class]
        X_maj = X[y == majority_class]
        
        if self.verbose:
            print(f"Initial distribution - Minority: {len(X_min)}, Majority: {len(X_maj)}")
        
        # ==================================================================
        # STAGE 1: Detection of Overlapping Region (Section 4.1.1)
        # ==================================================================
        if self.verbose:
            print("\n=== Stage 1: Detecting overlapping region ===")
        
        ovr_regmin, ovr_regmaj, nov_regmin, nov_regmaj = self._detect_overlapping_region(X_min, X_maj)
        
        if self.verbose:
            print(f"Overlapping minority instances: {len(ovr_regmin)}")
            print(f"Overlapping majority instances: {len(ovr_regmaj)}")
            print(f"Novel minority instances: {len(nov_regmin)}")
            print(f"Novel majority instances: {len(nov_regmaj)}")
        
        # ==================================================================
        # STAGE 2: Cleaning up overlapping instances and undersampling
        # (Section 4.1.2)
        # ==================================================================
        if self.verbose:
            print("\n=== Stage 2: Cleaning up and undersampling ===")
        
        # Step 1-2: Compute Tomek-Links
        if self.verbose:
            print("Computing Tomek-Links...")
        TL = self._compute_tomek_links(X_min, X_maj)
        if self.verbose:
            print(f"Total Tomek-Link pairs: {len(TL)}")
        
        # Step 3-6: Detect and remove Abnormal Tomek-Links (ATL)
        if self.verbose:
            print("Detecting Abnormal Tomek-Links (ATL)...")
        ATL = self._detect_abnormal_tomek_links(TL, X_min, X_maj)
        if self.verbose:
            print(f"Abnormal Tomek-Link pairs: {len(ATL)}")
        
        if len(ATL) > 0:
            X_min, X_maj, TL = self._update_after_atl_removal(ATL, X_min, X_maj)
            if self.verbose:
                print(f"After ATL removal - Minority: {len(X_min)}, Majority: {len(X_maj)}")
        
        # Step 7-9: Handle remaining cases based on TL'
        if self.verbose:
            print("\nProcessing remaining Tomek-Links...")
        
        # Case 2: Sparse neighbourhood detection
        ovrout_maj = []
        ovrout_maj_indices = []
        
        for min_idx, maj_idx in TL:
            # If majority in overlapping but minority not
            if maj_idx in self.ovr_regmaj_idx and min_idx not in self.ovr_regmin_idx:
                ovrout_maj.append(X_maj[maj_idx])
                ovrout_maj_indices.append(maj_idx)
        
        if len(ovrout_maj) > 0:
            if self.verbose:
                print(f"Overlapping majority instances for sparse neighbourhood check: {len(ovrout_maj)}")
            ovrout_maj = np.array(ovrout_maj)
            spNN, spNN_indices = self._estimate_sparse_neighbourhood(ovrout_maj, X_min, X_maj)
            if self.verbose:
                print(f"Instances with sparse neighbourhood: {len(spNN)}")
            
            # Remove sparse neighbourhood instances
            if len(spNN_indices) > 0:
                actual_indices = [ovrout_maj_indices[i] for i in spNN_indices]
                mask = np.ones(len(X_maj), dtype=bool)
                mask[actual_indices] = False
                X_maj = X_maj[mask]
                if self.verbose:
                    print(f"After sparse neighbourhood removal - Majority: {len(X_maj)}")
            
            # Recompute TL
            TL = self._compute_tomek_links(X_min, X_maj)
        
        # Case 3: Normal Tomek-Links (NTL) with redundancy check
        if self.verbose:
            print("\nDetecting Normal Tomek-Links (NTL)...")
        NTL = self._detect_normal_tomek_links(TL, X_min, X_maj)
        if self.verbose:
            print(f"Normal Tomek-Link pairs: {len(NTL)}")
        
        if len(NTL) > 0:
            # Step 10: Redundancy check
            if self.verbose:
                print("Checking redundancy...")
            REDN = self._check_redundancy(NTL, X_maj)
            if self.verbose:
                print(f"Redundant pairs found: {len(REDN)}")
            
            # Step 11-12: Intra-cluster significance
            if self.verbose:
                print("Computing intra-cluster significance...")
            cen_maj, dintra_cen = self._compute_intra_cluster_significance(X_maj)
            
            # Step 13: Select least significant redundant instances
            to_remove = self._select_least_significant_redundant(REDN, dintra_cen)
            if self.verbose:
                print(f"Least significant instances to remove: {len(to_remove)}")
            
            # Remove selected instances
            if len(to_remove) > 0:
                mask = np.ones(len(X_maj), dtype=bool)
                mask[to_remove] = False
                X_maj = X_maj[mask]
        
        # ==================================================================
        # Final step: Combine refined datasets (Equation 26)
        # ==================================================================
        if self.verbose:
            print(f"\nFinal distribution - Minority: {len(X_min)}, Majority: {len(X_maj)}")
        
        X_resampled = np.vstack([X_min, X_maj])
        y_resampled = np.hstack([np.full(len(X_min), minority_class),
                                np.full(len(X_maj), majority_class)])
        
        return X_resampled, y_resampled