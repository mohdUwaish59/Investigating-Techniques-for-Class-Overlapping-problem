"""
Random Oversampling Technique
Simple random duplication of minority class samples
"""

import numpy as np
from typing import Tuple
from .base_sampler import BaseSampler


class RandomOverSampler(BaseSampler):
    """
    Random Oversampling (ROS)
    
    Randomly duplicates minority class samples to balance the dataset.
    Simple and fast baseline technique.
    """
    
    def __init__(self, sampling_strategy='auto', random_state=None):
        """
        Parameters:
        -----------
        sampling_strategy : str or float, default='auto'
            If 'auto', balance to majority class size
            If float, multiply minority class by this factor
        random_state : int, optional
            Random seed for reproducibility
        """
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)
    
    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply random oversampling
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target vector
            
        Returns:
        --------
        X_resampled : np.ndarray
            Resampled feature matrix
        y_resampled : np.ndarray
            Resampled target vector
        """
        X_maj, y_maj, X_min, y_min, maj_class, min_class = self._separate_classes(X, y)
        
        # Determine target size
        if self.sampling_strategy == 'auto':
            target_size = len(X_maj)
        else:
            target_size = int(len(X_min) * self.sampling_strategy)
        
        n_samples_needed = target_size - len(X_min)
        
        if n_samples_needed > 0:
            # Randomly duplicate minority samples
            indices = np.random.choice(len(X_min), n_samples_needed, replace=True)
            X_min_ros = np.vstack([X_min, X_min[indices]])
            y_min_ros = np.hstack([y_min, y_min[indices]])
        else:
            X_min_ros, y_min_ros = X_min, y_min
        
        # Combine majority and oversampled minority
        X_resampled = np.vstack([X_maj, X_min_ros])
        y_resampled = np.hstack([y_maj, y_min_ros])
        
        return X_resampled, y_resampled
