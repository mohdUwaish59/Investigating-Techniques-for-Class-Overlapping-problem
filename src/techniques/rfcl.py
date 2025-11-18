"""
RFCL: Random Forest Cleaning Rule
Handles class overlap in imbalanced datasets using Random Forest margins

Reference: Zhang et al. (2021) "RFCL: A new under-sampling method of 
reducing the degree of imbalance and overlap"
"""

import numpy as np
from typing import Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.base import clone
from .base_sampler import BaseSampler


class RFCL(BaseSampler):
    """
    T1: RFCL - Random Forest Cleaning Rule
    
    Uses Random Forest to identify and remove overlapping majority class samples
    based on margin scores computed from tree predictions.
    
    FIXED VERSION: Properly identifies majority/minority classes
    """
    
    def __init__(self, final_classifier=None, random_state=42, verbose=True):
        """
        Parameters:
        -----------
        final_classifier : classifier object, optional
            The classifier to use for threshold optimization.
            If None, uses RandomForestClassifier with default parameters.
        random_state : int, default=42
            Random seed for reproducibility
        verbose : bool, default=True
            Print progress information
        """
        self.random_state = random_state
        self.final_classifier = final_classifier
        self.verbose = verbose
        self.threshold_ = None
        self.rf_margin_ = None
        self.stats_ = {}
        self.majority_class_ = None
        self.minority_class_ = None
    
    def _identify_classes(self, y):
        """Identify which class is majority and which is minority."""
        unique, counts = np.unique(y, return_counts=True)
        if counts[0] > counts[1]:
            self.majority_class_ = unique[0]
            self.minority_class_ = unique[1]
        else:
            self.majority_class_ = unique[1]
            self.minority_class_ = unique[0]
        
        if self.verbose:
            print(f"RFCL: Identified majority class = {self.majority_class_}, "
                  f"minority class = {self.minority_class_}")
    
    def _build_rf_and_compute_margins(self, X, y):
        """
        Build Random Forest and compute modified margins using EXACT paper formula.
        
        Paper's formula (Equation 3):
        mmargin(xi) = (vi1 - vi0) / (vi1 + vi0)
        
        where vi1 = votes for majority class, vi0 = votes for minority class
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
        
        Returns:
        --------
        margins : array of shape (n_samples,)
            Modified margin for each instance
        rf : RandomForestClassifier
            Fitted Random Forest model
        """
        n_features = X.shape[1]
        n_estimators = 500
        max_features = int(np.floor(np.log2(n_features)) + 1)
        
        if self.verbose:
            print(f"RFCL: Building Random Forest (n_estimators={n_estimators}, "
                  f"max_features={max_features})")
        
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_features=max_features,
            bootstrap=True,
            oob_score=True,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        rf.fit(X, y)
        
        # Compute margins using EXACT paper formula with vote counting
        margins = np.zeros(len(X))
        
        for i in range(len(X)):
            # Count votes from each tree
            vi_majority = 0  # votes for majority class
            vi_minority = 0  # votes for minority class
            
            for tree in rf.estimators_:
                prediction = tree.predict(X[i:i+1])[0]
                
                if prediction == self.majority_class_:
                    vi_majority += 1
                else:
                    vi_minority += 1
            
            # Apply paper's formula (Equation 3)
            total_votes = vi_majority + vi_minority
            if total_votes > 0:
                margins[i] = (vi_majority - vi_minority) / total_votes
            else:
                margins[i] = 0
        
        return margins, rf
    
    def _define_search_range(self, margins, y):
        """
        Define search range R for threshold as specified in Algorithm 2.
        
        R = {-1} ∪ {v | v is interpolation point from min(mmargin(SM))
        to max(median(mmargin(SM)), median(mmargin(Sm)))
        with step size 0.05}
        
        Parameters:
        -----------
        margins : array of shape (n_samples,)
            Computed margins
        y : array of shape (n_samples,)
            Target values
        
        Returns:
        --------
        search_range : array
            Candidate threshold values
        """
        # Separate margins by class
        margins_majority = margins[y == self.majority_class_]
        margins_minority = margins[y == self.minority_class_]
        
        # Compute boundaries
        min_majority = np.min(margins_majority)
        median_majority = np.median(margins_majority)
        median_minority = np.median(margins_minority)
        
        # Upper bound
        upper_bound = max(median_majority, median_minority)
        
        # Create range with step size 0.05
        step_size = 0.05
        interpolation_points = np.arange(min_majority, upper_bound + step_size, step_size)
        
        # Combine with -1
        search_range = np.concatenate([[-1.0], interpolation_points])
        search_range = np.unique(search_range)
        
        if self.verbose:
            print(f"RFCL: Search range: min={search_range.min():.3f}, "
                  f"max={search_range.max():.3f}, n_candidates={len(search_range)}")
        
        return search_range
    
    def _search_threshold(self, X, y, search_range):
        """
        Search for optimal threshold using 3-fold cross-validation
        as specified in Algorithm 2.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
        search_range : array
            Candidate threshold values
        
        Returns:
        --------
        best_threshold : float
            Optimal threshold value
        """
        # Precompute margins once
        margins, _ = self._build_rf_and_compute_margins(X, y)
        
        best_f1 = -np.inf
        best_threshold = -1
        
        # Get final classifier
        if self.final_classifier is None:
            n_features = X.shape[1]
            max_features = int(np.floor(np.log2(n_features)) + 1)
            base_classifier = RandomForestClassifier(
                n_estimators=500,
                max_features=max_features,
                random_state=self.random_state,
                n_jobs=-1
            )
        else:
            base_classifier = self.final_classifier
        
        if self.verbose:
            print(f"RFCL: Searching optimal threshold from {len(search_range)} candidates")
        
        # Try each threshold value
        for v in search_range:
            f1_scores = []
            
            # 3-fold stratified cross-validation
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
            
            for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                # Split data
                X_train_fold = X[train_idx]
                y_train_fold = y[train_idx]
                X_val_fold = X[val_idx]
                y_val_fold = y[val_idx]
                margins_train = margins[train_idx]
                
                # Apply RFCL: Remove majority instances with margin <= v
                # Keep: all minority instances + majority instances with margin > v
                mask = (y_train_fold == self.minority_class_) | (margins_train > v)
                X_train_cleaned = X_train_fold[mask]
                y_train_cleaned = y_train_fold[mask]
                
                # Skip if no samples remain or only one class
                if len(X_train_cleaned) == 0 or len(np.unique(y_train_cleaned)) < 2:
                    f1_scores.append(0.0)
                    continue
                
                # Train classifier on cleaned data
                clf = clone(base_classifier)
                clf.fit(X_train_cleaned, y_train_cleaned)
                
                # Predict on validation set
                y_pred = clf.predict(X_val_fold)
                
                # Calculate F1-score (minority class is positive)
                f1 = f1_score(y_val_fold, y_pred, pos_label=self.minority_class_, zero_division=0)
                f1_scores.append(f1)
            
            # Average F1-score across folds
            avg_f1 = np.mean(f1_scores)
            
            # Update best threshold
            if avg_f1 > best_f1:
                best_f1 = avg_f1
                best_threshold = v
        
        if self.verbose:
            print(f"RFCL: Optimal threshold = {best_threshold:.3f} (F1 = {best_f1:.4f})")
        
        return best_threshold
    
    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply RFCL to resample the dataset.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
        
        Returns:
        --------
        X_resampled : array of shape (n_samples_new, n_features)
            Resampled features
        y_resampled : array of shape (n_samples_new,)
            Resampled target values
        """
        X = np.array(X)
        y = np.array(y)
        
        # Identify majority and minority classes
        self._identify_classes(y)
        
        n_majority_before = np.sum(y == self.majority_class_)
        n_minority_before = np.sum(y == self.minority_class_)
        
        if self.verbose:
            ir_before = n_majority_before / n_minority_before
            print(f"RFCL: Original IR={ir_before:.2f} "
                  f"(Maj={n_majority_before}, Min={n_minority_before})")
        
        # Step 1: Build RF and compute margins
        margins, self.rf_margin_ = self._build_rf_and_compute_margins(X, y)
        
        # Step 2: Define search range
        search_range = self._define_search_range(margins, y)
        
        # Step 3: Search for optimal threshold
        self.threshold_ = self._search_threshold(X, y, search_range)
        
        # Step 4: Apply RFCL with optimal threshold
        # Remove majority class instances with margin <= threshold
        # Keep: all minority + majority with margin > threshold
        mask = (y == self.minority_class_) | (margins > self.threshold_)
        X_resampled = X[mask]
        y_resampled = y[mask]
        
        n_removed = np.sum(~mask)
        n_majority_after = np.sum(y_resampled == self.majority_class_)
        n_minority_after = np.sum(y_resampled == self.minority_class_)
        
        # Store final statistics
        self.stats_['final'] = {
            'n_majority': int(n_majority_after),
            'n_minority': int(n_minority_after),
            'imbalance_ratio': n_majority_after / n_minority_after if n_minority_after > 0 else 0,
            'n_removed': int(n_removed)
        }
        
        if self.verbose:
            print(f"RFCL: Removed {n_removed} majority samples")
            ir_after = n_majority_after / n_minority_after if n_minority_after > 0 else 0
            print(f"RFCL: Final IR={ir_after:.2f} "
                  f"(Maj={n_majority_after}, Min={n_minority_after})")
        
        return X_resampled, y_resampled
