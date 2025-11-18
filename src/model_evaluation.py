"""
Model Training and Evaluation Module
Provides consistent model training and evaluation across different resampling techniques
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    balanced_accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """
    Unified model training and evaluation for imbalanced learning
    """
    
    def __init__(self, test_size: float = 0.3, random_state: int = 42,
                 cv_folds: int = 5, verbose: bool = True):
        """
        Parameters:
        -----------
        test_size : float
            Proportion of data to use for testing
        random_state : int
            Random state for reproducibility
        cv_folds : int
            Number of cross-validation folds
        verbose : bool
            Whether to print progress information
        """
        self.test_size = test_size
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.verbose = verbose
        
        # Store results
        self.results_ = {}
        self.models_ = {}
        self.predictions_ = {}
    
    def get_classifier(self, classifier_name: str = 'decision_tree', **kwargs):
        """
        Get a classifier instance
        
        Parameters:
        -----------
        classifier_name : str
            Name of the classifier
        **kwargs : additional parameters for the classifier
        
        Returns:
        --------
        classifier : sklearn classifier instance
        """
        classifiers = {
            'decision_tree': DecisionTreeClassifier(
                max_depth=kwargs.get('max_depth', 5),
                random_state=self.random_state
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 5),
                random_state=self.random_state
            ),
            'naive_bayes': GaussianNB(),
            'svm': SVC(
                kernel=kwargs.get('kernel', 'rbf'),
                probability=True,
                random_state=self.random_state
            ),
            'knn': KNeighborsClassifier(
                n_neighbors=kwargs.get('n_neighbors', 5)
            ),
            'logistic_regression': LogisticRegression(
                max_iter=kwargs.get('max_iter', 1000),
                random_state=self.random_state
            )
        }
        
        if classifier_name not in classifiers:
            raise ValueError(f"Unknown classifier: {classifier_name}")
        
        return classifiers[classifier_name]
    
    def calculate_gmean(self, y_true, y_pred):
        """Calculate geometric mean of sensitivity and specificity"""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        return np.sqrt(sensitivity * specificity)
    
    def evaluate_single_technique(self, X_original: np.ndarray, y_original: np.ndarray,
                                 X_resampled: np.ndarray, y_resampled: np.ndarray,
                                 technique_name: str, classifier_name: str = 'decision_tree',
                                 **classifier_kwargs) -> Dict[str, float]:
        """
        Evaluate a single resampling technique
        
        Parameters:
        -----------
        X_original : np.ndarray
            Original feature matrix
        y_original : np.ndarray
            Original target vector
        X_resampled : np.ndarray
            Resampled feature matrix
        y_resampled : np.ndarray
            Resampled target vector
        technique_name : str
            Name of the resampling technique
        classifier_name : str
            Name of the classifier to use
        **classifier_kwargs : additional classifier parameters
        
        Returns:
        --------
        metrics : dict
            Evaluation metrics
        """
        # Split original data for testing
        X_train, X_test, y_train, y_test = train_test_split(
            X_original, y_original, test_size=self.test_size,
            random_state=self.random_state, stratify=y_original
        )
        
        # Train on resampled data
        clf = self.get_classifier(classifier_name, **classifier_kwargs)
        clf.fit(X_resampled, y_resampled)
        
        # Predict on test set
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, 'predict_proba') else y_pred
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'specificity': recall_score(y_test, y_pred, pos_label=0),
            'g_mean': self.calculate_gmean(y_test, y_pred),
            'auc_roc': roc_auc_score(y_test, y_proba) if hasattr(clf, 'predict_proba') else None
        }
        
        # Store results
        self.results_[technique_name] = metrics
        self.models_[technique_name] = clf
        self.predictions_[technique_name] = {
            'y_true': y_test,
            'y_pred': y_pred,
            'y_proba': y_proba
        }
        
        if self.verbose:
            print(f"\n{technique_name} Results:")
            print("-" * 40)
            for metric, value in metrics.items():
                if value is not None:
                    print(f"{metric:20s}: {value:.4f}")
        
        return metrics
    
    def compare_techniques(self, X_original: np.ndarray, y_original: np.ndarray,
                          techniques: Dict[str, Tuple[np.ndarray, np.ndarray]],
                          classifier_name: str = 'decision_tree',
                          include_baseline: bool = True,
                          **classifier_kwargs) -> pd.DataFrame:
        """
        Compare multiple resampling techniques
        
        Parameters:
        -----------
        X_original : np.ndarray
            Original feature matrix
        y_original : np.ndarray
            Original target vector
        techniques : dict
            Dictionary of {name: (X_resampled, y_resampled)} pairs
        classifier_name : str
            Name of the classifier to use
        include_baseline : bool
            Whether to include baseline (no resampling) results
        **classifier_kwargs : additional classifier parameters
        
        Returns:
        --------
        results_df : pd.DataFrame
            Comparison results
        """
        if self.verbose:
            print(f"Evaluating {len(techniques)} techniques using {classifier_name}")
            print("="*60)
        
        # Evaluate baseline if requested
        if include_baseline:
            self.evaluate_single_technique(
                X_original, y_original, X_original, y_original,
                "Baseline (No Resampling)", classifier_name, **classifier_kwargs
            )
        
        # Evaluate each technique
        for name, (X_res, y_res) in techniques.items():
            self.evaluate_single_technique(
                X_original, y_original, X_res, y_res,
                name, classifier_name, **classifier_kwargs
            )
        
        # Create results DataFrame
        results_df = pd.DataFrame(self.results_).T
        results_df = results_df.round(4)
        
        # Sort by G-mean (good overall metric for imbalanced data)
        results_df = results_df.sort_values('g_mean', ascending=False)
        
        if self.verbose:
            print("\n" + "="*60)
            print("SUMMARY COMPARISON")
            print("="*60)
            print(results_df.to_string())
        
        return results_df
    
    def plot_confusion_matrices(self, technique_names: Optional[List[str]] = None,
                               figsize: Optional[Tuple[int, int]] = None):
        """
        Plot confusion matrices for specified techniques
        
        Parameters:
        -----------
        technique_names : list, optional
            Names of techniques to plot. If None, plots all
        figsize : tuple, optional
            Figure size
        """
        if technique_names is None:
            technique_names = list(self.predictions_.keys())
        
        n_techniques = len(technique_names)
        if figsize is None:
            figsize = (5 * n_techniques, 4)
        
        fig, axes = plt.subplots(1, n_techniques, figsize=figsize)
        if n_techniques == 1:
            axes = [axes]
        
        for ax, name in zip(axes, technique_names):
            y_true = self.predictions_[name]['y_true']
            y_pred = self.predictions_[name]['y_pred']
            
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f'{name}\nG-mean: {self.results_[name]["g_mean"]:.3f}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        
        plt.tight_layout()
        plt.show()
    
    def plot_metrics_comparison(self, metrics: Optional[List[str]] = None,
                               figsize: Tuple[int, int] = (12, 6)):
        """
        Plot bar chart comparing metrics across techniques
        
        Parameters:
        -----------
        metrics : list, optional
            Metrics to compare. If None, uses default set
        figsize : tuple
            Figure size
        """
        if metrics is None:
            metrics = ['precision', 'recall', 'f1_score', 'g_mean']
        
        results_df = pd.DataFrame(self.results_).T
        
        # Filter metrics
        results_df = results_df[metrics]
        
        # Create plot
        ax = results_df.plot(kind='bar', figsize=figsize, rot=45)
        ax.set_xlabel('Technique')
        ax.set_ylabel('Score')
        ax.set_title('Performance Metrics Comparison')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', fontsize=8)
        
        plt.tight_layout()
        plt.show()
    
    def get_best_technique(self, metric: str = 'g_mean') -> str:
        """
        Get the best performing technique based on a specific metric
        
        Parameters:
        -----------
        metric : str
            Metric to use for comparison
        
        Returns:
        --------
        best_technique : str
            Name of the best technique
        """
        results_df = pd.DataFrame(self.results_).T
        return results_df[metric].idxmax()
    
    def cross_validate_technique(self, X: np.ndarray, y: np.ndarray,
                                resampler, classifier_name: str = 'decision_tree',
                                scoring: str = 'balanced_accuracy',
                                **classifier_kwargs) -> Dict[str, float]:
        """
        Perform cross-validation for a single technique
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target vector
        resampler : resampling object with fit_resample method
        classifier_name : str
            Name of the classifier
        scoring : str
            Scoring metric for cross-validation
        **classifier_kwargs : additional classifier parameters
        
        Returns:
        --------
        cv_results : dict
            Cross-validation results
        """
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, 
                             random_state=self.random_state)
        
        scores = []
        
        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Apply resampling to training data
            X_train_res, y_train_res = resampler.fit_resample(X_train, y_train)
            
            # Train and evaluate
            clf = self.get_classifier(classifier_name, **classifier_kwargs)
            clf.fit(X_train_res, y_train_res)
            y_pred = clf.predict(X_val)
            
            if scoring == 'balanced_accuracy':
                score = balanced_accuracy_score(y_val, y_pred)
            elif scoring == 'g_mean':
                score = self.calculate_gmean(y_val, y_pred)
            else:
                score = accuracy_score(y_val, y_pred)
            
            scores.append(score)
        
        cv_results = {
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'scores': scores
        }
        
        if self.verbose:
            print(f"Cross-validation ({self.cv_folds} folds):")
            print(f"  Mean {scoring}: {cv_results['mean_score']:.4f} "
                  f"(+/- {cv_results['std_score']:.4f})")
        
        return cv_results


# Convenience function for quick evaluation
def quick_evaluate(X_original, y_original, X_resampled, y_resampled, 
                  technique_name="Resampling", classifier="decision_tree"):
    """Quick evaluation function for convenience"""
    evaluator = ModelEvaluator(verbose=True)
    return evaluator.evaluate_single_technique(
        X_original, y_original, X_resampled, y_resampled,
        technique_name, classifier
    )


if __name__ == "__main__":
    # Test the evaluation module
    from data_loader import DataLoader
    from resampling_techniques import EHSO, RandomOverSampler, RandomUnderSampler
    
    print("Testing Model Evaluation Module")
    print("="*60)
    
    # Create synthetic data
    loader = DataLoader()
    X, y = loader.create_synthetic_data(n_samples=500, n_features=10, 
                                       imbalance_ratio=5, overlap_degree=0.3)
    
    # Apply different resampling techniques
    techniques = {}
    
    # EHSO
    ehso = EHSO(verbose=False)
    X_ehso, y_ehso = ehso.fit_resample(X, y)
    techniques['EHSO'] = (X_ehso, y_ehso)
    
    # Random Oversampling
    ros = RandomOverSampler()
    X_ros, y_ros = ros.fit_resample(X, y)
    techniques['ROS'] = (X_ros, y_ros)
    
    # Random Undersampling
    rus = RandomUnderSampler()
    X_rus, y_rus = rus.fit_resample(X, y)
    techniques['RUS'] = (X_rus, y_rus)
    
    # Initialize evaluator and compare techniques
    evaluator = ModelEvaluator()
    results_df = evaluator.compare_techniques(X, y, techniques, 
                                             classifier_name='decision_tree',
                                             include_baseline=True)
    
    # Plot comparisons
    print("\nPlotting confusion matrices...")
    evaluator.plot_confusion_matrices()
    
    print("\nPlotting metrics comparison...")
    evaluator.plot_metrics_comparison()
    
    # Get best technique
    best = evaluator.get_best_technique(metric='g_mean')
    print(f"\nBest technique based on G-mean: {best}")
    
    print("\nEvaluation module tests completed!")