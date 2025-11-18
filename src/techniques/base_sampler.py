"""
Base Sampler Class
Abstract base class for all resampling techniques
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any


class BaseSampler(ABC):
    """
    Abstract base class for all resampling techniques
    """
    
    @abstractmethod
    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resample the dataset
        
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
        pass
    
    def get_params(self) -> Dict[str, Any]:
        """Get parameters of the sampler"""
        return self.__dict__.copy()
    
    def _separate_classes(self, X, y):
        """Utility method to separate majority and minority classes"""
        unique_classes, counts = np.unique(y, return_counts=True)
        majority_class = unique_classes[np.argmax(counts)]
        minority_class = unique_classes[np.argmin(counts)]
        
        X_maj = X[y == majority_class]
        y_maj = y[y == majority_class]
        X_min = X[y == minority_class]
        y_min = y[y == minority_class]
        
        return X_maj, y_maj, X_min, y_min, majority_class, minority_class
