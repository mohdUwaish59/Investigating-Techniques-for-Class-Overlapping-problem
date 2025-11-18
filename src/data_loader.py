"""
Data Loader Module
Handles loading data from various sources including DataFrames, CSV files, and synthetic generation
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.datasets import make_classification
from typing import Tuple, Optional, Union


class DataLoader:
    """
    Modular data loader for imbalanced learning experiments
    """
    
    def __init__(self, standardize=True, random_state=42):
        """
        Parameters:
        -----------
        standardize : bool, default=True
            Whether to standardize features
        random_state : int, default=42
            Random state for reproducibility
        """
        self.standardize = standardize
        self.random_state = random_state
        self.scaler = StandardScaler() if standardize else None
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.original_classes = None
        
    def load_from_dataframe(self, df: pd.DataFrame, 
                           target_column: str,
                           feature_columns: Optional[list] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data from a pandas DataFrame
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        target_column : str
            Name of the target column
        feature_columns : list, optional
            List of feature column names. If None, uses all columns except target
            
        Returns:
        --------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target vector (encoded as 0 for majority, 1 for minority)
        """
        # Get feature columns if not specified
        if feature_columns is None:
            feature_columns = [col for col in df.columns if col != target_column]
        
        self.feature_names = feature_columns
        
        # Extract features and target
        X = df[feature_columns].values
        y_raw = df[target_column].values
        
        # Encode target to ensure minority class is 1
        y_encoded = self.label_encoder.fit_transform(y_raw)
        
        # Ensure minority class is labeled as 1
        unique_classes, counts = np.unique(y_encoded, return_counts=True)
        if len(unique_classes) != 2:
            raise ValueError(f"Expected binary classification, got {len(unique_classes)} classes")
        
        # If class 0 is minority, swap labels
        if counts[0] < counts[1]:
            y_encoded = 1 - y_encoded
            self.original_classes = self.label_encoder.classes_[::-1]
        else:
            self.original_classes = self.label_encoder.classes_
        
        # Standardize features if requested
        if self.standardize:
            X = self.scaler.fit_transform(X)
        
        print(f"Data loaded successfully:")
        print(f"  Features: {X.shape[1]} dimensions")
        print(f"  Samples: {X.shape[0]} total")
        print(f"  Majority class (0): {np.sum(y_encoded == 0)} samples")
        print(f"  Minority class (1): {np.sum(y_encoded == 1)} samples")
        print(f"  Imbalance Ratio: {np.sum(y_encoded == 0) / np.sum(y_encoded == 1):.2f}")
        
        return X, y_encoded
    
    def load_from_csv(self, filepath: str, 
                      target_column: str,
                      feature_columns: Optional[list] = None,
                      **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data from a CSV file
        
        Parameters:
        -----------
        filepath : str
            Path to CSV file
        target_column : str
            Name of the target column
        feature_columns : list, optional
            List of feature column names
        **kwargs : additional arguments for pd.read_csv()
        
        Returns:
        --------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target vector
        """
        df = pd.read_csv(filepath, **kwargs)
        df['stabf'] = df['stabf'].apply(lambda x: 1 if x == 'stable' else 0)
        return self.load_from_dataframe(df, target_column, feature_columns)
    
    def create_synthetic_data(self, n_samples: int = 500,
                             n_features: int = 10,
                             n_informative: Optional[int] = None,
                             imbalance_ratio: float = 5.0,
                             overlap_degree: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create synthetic imbalanced data with controlled overlap
        
        Parameters:
        -----------
        n_samples : int
            Total number of samples
        n_features : int
            Number of features
        n_informative : int, optional
            Number of informative features
        imbalance_ratio : float
            Ratio of majority to minority samples
        overlap_degree : float
            Degree of class overlap (0-1)
        
        Returns:
        --------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target vector
        """
        if n_informative is None:
            n_informative = max(2, n_features - 2)
        
        n_minority = int(n_samples / (imbalance_ratio + 1))
        n_majority = n_samples - n_minority
        
        weights = [n_majority/n_samples, n_minority/n_samples]
        
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_features - n_informative,
            n_clusters_per_class=2,
            weights=weights,
            flip_y=overlap_degree,
            random_state=self.random_state
        )
        
        # Ensure minority class is labeled as 1
        unique_classes, counts = np.unique(y, return_counts=True)
        if counts[0] < counts[1]:
            y = 1 - y
        
        # Create synthetic feature names
        self.feature_names = [f"Feature_{i+1}" for i in range(n_features)]
        
        if self.standardize:
            X = self.scaler.fit_transform(X)
        
        print(f"Synthetic data created:")
        print(f"  Features: {n_features} ({n_informative} informative)")
        print(f"  Majority class (0): {np.sum(y == 0)} samples")
        print(f"  Minority class (1): {np.sum(y == 1)} samples")
        print(f"  Imbalance Ratio: {np.sum(y == 0) / np.sum(y == 1):.2f}")
        print(f"  Overlap degree: {overlap_degree:.1%}")
        
        return X, y
    
    def get_class_distribution(self, y: np.ndarray) -> dict:
        """
        Get detailed class distribution statistics
        
        Parameters:
        -----------
        y : np.ndarray
            Target vector
            
        Returns:
        --------
        dict : Class distribution statistics
        """
        unique, counts = np.unique(y, return_counts=True)
        
        stats = {
            'n_samples': len(y),
            'n_classes': len(unique),
            'class_counts': dict(zip(unique, counts)),
            'class_ratios': dict(zip(unique, counts / len(y))),
            'imbalance_ratio': max(counts) / min(counts) if len(unique) == 2 else None,
            'majority_class': unique[np.argmax(counts)],
            'minority_class': unique[np.argmin(counts)]
        }
        
        return stats
    
    def transform_new_data(self, X: np.ndarray) -> np.ndarray:
        """
        Transform new data using fitted scaler
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix to transform
            
        Returns:
        --------
        X_transformed : np.ndarray
            Transformed feature matrix
        """
        if self.standardize and self.scaler is not None:
            return self.scaler.transform(X)
        return X


# Example usage functions
def load_iris_imbalanced():
    """
    Example: Load iris dataset and make it imbalanced
    """
    from sklearn.datasets import load_iris
    
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    
    # Make it binary and imbalanced (class 0 vs rest)
    df['target'] = (df['target'] != 0).astype(int)
    
    # Further imbalance it
    minority_indices = df[df['target'] == 1].sample(n=20, random_state=42).index
    majority_indices = df[df['target'] == 0].index
    df_imbalanced = df.loc[list(majority_indices) + list(minority_indices)]
    
    return df_imbalanced


def load_credit_card_fraud_sample():
    """
    Example: Create a sample credit card fraud-like dataset
    """
    np.random.seed(42)
    
    n_legitimate = 500
    n_fraud = 50
    
    # Create synthetic credit card transaction data
    legitimate = np.random.normal(0, 1, (n_legitimate, 10))
    fraud = np.random.normal(0.5, 1.5, (n_fraud, 10))
    
    X = np.vstack([legitimate, fraud])
    y = np.hstack([np.zeros(n_legitimate), np.ones(n_fraud)])
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    # Create DataFrame
    columns = [f"V{i+1}" for i in range(10)]
    df = pd.DataFrame(X, columns=columns)
    df['Class'] = y.astype(int)
    
    return df


if __name__ == "__main__":
    # Test the data loader
    print("Testing DataLoader Module")
    print("="*60)
    
    # Test 1: Synthetic data
    print("\n1. Creating synthetic data:")
    loader = DataLoader(standardize=True)
    X_synth, y_synth = loader.create_synthetic_data(
        n_samples=500, 
        n_features=10,
        imbalance_ratio=5.0,
        overlap_degree=0.3
    )
    
    # Test 2: Load from DataFrame
    print("\n2. Loading from DataFrame (Iris example):")
    df_iris = load_iris_imbalanced()
    loader_iris = DataLoader(standardize=True)
    X_iris, y_iris = loader_iris.load_from_dataframe(df_iris, target_column='target')
    
    # Test 3: Credit card fraud example
    print("\n3. Loading credit card fraud-like data:")
    df_fraud = load_credit_card_fraud_sample()
    loader_fraud = DataLoader(standardize=True)
    X_fraud, y_fraud = loader_fraud.load_from_dataframe(df_fraud, target_column='Class')
    
    print("\nData loader tests completed successfully!")