"""
Random Undersampling Technique
Simple random removal of majority class samples
"""

import numpy as np
from typing import Tuple
from .base_sampler import BaseSampler


class RandomUnderSampler(BaseSampler):
    """
    Random Undersampling (RUS)
    
    Randomly removes majority class samples to balance the dataset.
    Simple and fast baseline technique.
    """
    
    def __init__(self, sampling_strategy='auto', random_state=None):
        """
        Parameters:
        -----------
        sampling_strategy : str or float, default='auto'
            If 'auto', balance to minority class size
            If float, keep this fraction of majority class
        random_state : int, optional
            Random seed for reproducibility
        """
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)
    
    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply random undersampling
        
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
            target_size = len(X_min)
        else:
            target_size = int(len(X_maj) * self.sampling_strategy)
        
        if target_size < len(X_maj):
            # Randomly select majority samples
            indices = np.random.choice(len(X_maj), target_size, replace=False)
            X_maj_rus = X_maj[indices]
            y_maj_rus = y_maj[indices]
        else:
            X_maj_rus, y_maj_rus = X_maj, y_maj
        
        # Combine undersampled majority and minority
        X_resampled = np.vstack([X_maj_rus, X_min])
        y_resampled = np.hstack([y_maj_rus, y_min])
        
        return X_resampled, y_resampled
