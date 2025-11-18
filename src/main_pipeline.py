"""
Main Pipeline Script
Demonstrates how to use all modular components together for imbalanced learning experiments
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Optional, List

# Import all modules
from data_loader import DataLoader
from resampling_techniques import EHSO, RandomOverSampler, RandomUnderSampler
from visualization import ImbalancedDataVisualizer
from model_evaluation import ModelEvaluator


class ImbalancedLearningPipeline:
    """
    Complete pipeline for imbalanced learning experiments
    """
    
    def __init__(self, random_state: int = 42, verbose: bool = True):
        """
        Parameters:
        -----------
        random_state : int
            Random state for reproducibility
        verbose : bool
            Whether to print progress information
        """
        self.random_state = random_state
        self.verbose = verbose
        
        # Initialize components
        self.data_loader = DataLoader(standardize=True, random_state=random_state)
        self.visualizer = ImbalancedDataVisualizer()
        self.evaluator = ModelEvaluator(random_state=random_state, verbose=verbose)
        
        # Store data and results
        self.X = None
        self.y = None
        self.resampled_data = {}
        self.evaluation_results = None
    
    def load_data(self, data_source: str = 'synthetic', **kwargs):
        """
        Load data from various sources
        
        Parameters:
        -----------
        data_source : str
            Source of data ('synthetic', 'dataframe', 'csv')
        **kwargs : additional parameters for data loading
        
        Returns:
        --------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target vector
        """
        if self.verbose:
            print("\n" + "="*70)
            print("LOADING DATA")
            print("="*70)
        
        if data_source == 'synthetic':
            self.X, self.y = self.data_loader.create_synthetic_data(**kwargs)
        elif data_source == 'dataframe':
            df = kwargs.get('dataframe')
            target_column = kwargs.get('target_column')
            feature_columns = kwargs.get('feature_columns', None)
            self.X, self.y = self.data_loader.load_from_dataframe(
                df, target_column, feature_columns
            )
        elif data_source == 'csv':
            filepath = kwargs.get('filepath')
            target_column = kwargs.get('target_column')
            self.X, self.y = self.data_loader.load_from_csv(filepath, target_column)
        else:
            raise ValueError(f"Unknown data source: {data_source}")
        
        # Get and display statistics
        stats = self.data_loader.get_class_distribution(self.y)
        if self.verbose:
            print(f"\nClass Distribution:")
            print(f"  Total samples: {stats['n_samples']}")
            print(f"  Class counts: {stats['class_counts']}")
            print(f"  Imbalance ratio: {stats['imbalance_ratio']:.2f}")
        
        return self.X, self.y
    
    def visualize_original_data(self, show_overlap: bool = True, 
                               k_neighbors: int = 5):
        """
        Visualize the original data distribution
        
        Parameters:
        -----------
        show_overlap : bool
            Whether to show overlapping regions
        k_neighbors : int
            Number of neighbors for overlap detection
        """
        if self.X is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if self.verbose:
            print("\n" + "="*70)
            print("VISUALIZING ORIGINAL DATA")
            print("="*70)
        
        if show_overlap:
            overlapping_indices = self.visualizer.plot_overlap_detection(
                self.X, self.y, k_neighbors=k_neighbors
            )
            print(f"Detected {len(overlapping_indices)} overlapping samples")
        else:
            self.visualizer.plot_data_distribution(
                self.X, self.y, title="Original Data Distribution"
            )
    
    def apply_resampling_techniques(self, techniques: Optional[Dict] = None):
        """
        Apply multiple resampling techniques
        
        Parameters:
        -----------
        techniques : dict, optional
            Dictionary of {name: resampler} pairs. If None, uses default set
        
        Returns:
        --------
        resampled_data : dict
            Dictionary of {name: (X_resampled, y_resampled)} pairs
        """
        if self.X is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if self.verbose:
            print("\n" + "="*70)
            print("APPLYING RESAMPLING TECHNIQUES")
            print("="*70)
        
        # Default techniques if not specified
        if techniques is None:
            techniques = {
                'EHSO': EHSO(k_neighbors=5, alpha=0.1, verbose=False, 
                           random_state=self.random_state),
                'Random Oversampling': RandomOverSampler(random_state=self.random_state),
                'Random Undersampling': RandomUnderSampler(random_state=self.random_state)
            }
        
        # Apply each technique
        for name, resampler in techniques.items():
            if self.verbose:
                print(f"\nApplying {name}...")
            
            X_res, y_res = resampler.fit_resample(self.X, self.y)
            self.resampled_data[name] = (X_res, y_res)
            
            # Display statistics
            unique, counts = np.unique(y_res, return_counts=True)
            ir = max(counts) / min(counts)
            if self.verbose:
                print(f"  Resampled: {len(X_res)} samples, IR={ir:.2f}")
        
        return self.resampled_data
    
    def visualize_resampling_results(self, technique_names: Optional[List[str]] = None):
        """
        Visualize results of resampling techniques
        
        Parameters:
        -----------
        technique_names : list, optional
            Names of techniques to visualize. If None, visualizes all
        """
        if not self.resampled_data:
            raise ValueError("No resampled data. Call apply_resampling_techniques() first.")
        
        if self.verbose:
            print("\n" + "="*70)
            print("VISUALIZING RESAMPLING RESULTS")
            print("="*70)
        
        if technique_names is None:
            technique_names = list(self.resampled_data.keys())
        
        for name in technique_names:
            if name in self.resampled_data:
                X_res, y_res = self.resampled_data[name]
                self.visualizer.plot_before_after_comparison(
                    self.X, self.y, X_res, y_res,
                    technique_name=name
                )
    
    def evaluate_techniques(self, classifier_name: str = 'decision_tree',
                           include_baseline: bool = True,
                           **classifier_kwargs):
        """
        Evaluate all resampling techniques
        
        Parameters:
        -----------
        classifier_name : str
            Name of the classifier to use
        include_baseline : bool
            Whether to include baseline (no resampling) results
        **classifier_kwargs : additional classifier parameters
        
        Returns:
        --------
        results : pd.DataFrame
            Evaluation results
        """
        if not self.resampled_data:
            raise ValueError("No resampled data. Call apply_resampling_techniques() first.")
        
        if self.verbose:
            print("\n" + "="*70)
            print("EVALUATING TECHNIQUES")
            print("="*70)
        
        self.evaluation_results = self.evaluator.compare_techniques(
            self.X, self.y,
            self.resampled_data,
            classifier_name=classifier_name,
            include_baseline=include_baseline,
            **classifier_kwargs
        )
        
        return self.evaluation_results
    
    def visualize_evaluation_results(self):
        """
        Visualize evaluation results
        """
        if self.evaluation_results is None:
            raise ValueError("No evaluation results. Call evaluate_techniques() first.")
        
        if self.verbose:
            print("\n" + "="*70)
            print("VISUALIZING EVALUATION RESULTS")
            print("="*70)
        
        # Plot confusion matrices
        self.evaluator.plot_confusion_matrices()
        
        # Plot metrics comparison
        self.evaluator.plot_metrics_comparison()
        
        # Plot class distribution comparison
        datasets_for_comparison = {'Original': self.y}
        for name, (_, y_res) in self.resampled_data.items():
            datasets_for_comparison[name] = y_res
        
        self.visualizer.plot_class_distribution_bar(datasets_for_comparison)
    
    def get_best_technique(self, metric: str = 'g_mean'):
        """
        Get the best performing technique
        
        Parameters:
        -----------
        metric : str
            Metric to use for comparison
        
        Returns:
        --------
        best_technique : str
            Name of the best technique
        """
        if self.evaluation_results is None:
            raise ValueError("No evaluation results. Call evaluate_techniques() first.")
        
        best = self.evaluator.get_best_technique(metric=metric)
        
        if self.verbose:
            print(f"\nBest technique based on {metric}: {best}")
            print(f"{metric} score: {self.evaluation_results.loc[best, metric]:.4f}")
        
        return best
    
    def run_complete_pipeline(self, data_source: str = 'synthetic',
                            data_kwargs: Optional[Dict] = None,
                            techniques: Optional[Dict] = None,
                            classifier_name: str = 'decision_tree',
                            visualize: bool = True):
        """
        Run the complete pipeline from data loading to evaluation
        
        Parameters:
        -----------
        data_source : str
            Source of data
        data_kwargs : dict
            Parameters for data loading
        techniques : dict
            Resampling techniques to apply
        classifier_name : str
            Classifier for evaluation
        visualize : bool
            Whether to show visualizations
        
        Returns:
        --------
        results : dict
            Complete pipeline results
        """
        if self.verbose:
            print("\n" + "="*70)
            print("RUNNING COMPLETE IMBALANCED LEARNING PIPELINE")
            print("="*70)
        
        # Step 1: Load data
        data_kwargs = data_kwargs or {
            'n_samples': 500,
            'n_features': 10,
            'imbalance_ratio': 5,
            'overlap_degree': 0.3
        }
        self.load_data(data_source, **data_kwargs)
        
        # Step 2: Visualize original data
        if visualize:
            self.visualize_original_data(show_overlap=True)
        
        # Step 3: Apply resampling techniques
        self.apply_resampling_techniques(techniques)
        
        # Step 4: Visualize resampling results
        if visualize:
            # Show comparison for best technique (we'll determine after evaluation)
            pass
        
        # Step 5: Evaluate techniques
        evaluation_results = self.evaluate_techniques(
            classifier_name=classifier_name,
            include_baseline=True
        )
        
        # Step 6: Visualize evaluation results
        if visualize:
            self.visualize_evaluation_results()
        
        # Step 7: Identify best technique
        best_technique = self.get_best_technique(metric='g_mean')
        
        # Show best technique visualization
        if visualize and best_technique != "Baseline (No Resampling)":
            print(f"\nShowing before/after comparison for best technique: {best_technique}")
            X_best, y_best = self.resampled_data[best_technique]
            self.visualizer.plot_before_after_comparison(
                self.X, self.y, X_best, y_best,
                technique_name=f"Best: {best_technique}"
            )
        
        # Return complete results
        results = {
            'data_stats': self.data_loader.get_class_distribution(self.y),
            'evaluation_results': evaluation_results,
            'best_technique': best_technique,
            'resampled_data': self.resampled_data
        }
        
        if self.verbose:
            print("\n" + "="*70)
            print("PIPELINE COMPLETED SUCCESSFULLY")
            print("="*70)
        
        return results


# Example usage functions
def example_with_dataframe():
    """Example: Using the pipeline with a DataFrame"""
    # Create sample DataFrame
    np.random.seed(42)
    n_samples = 500
    n_features = 5
    
    # Create imbalanced data
    X = np.random.randn(n_samples, n_features)
    y = np.zeros(n_samples)
    y[:50] = 1  # 10% minority class
    
    # Add some overlap
    X[y == 1] += np.random.randn(50, n_features) * 0.5
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    df['target'] = y.astype(int)
    
    # Run pipeline
    pipeline = ImbalancedLearningPipeline(verbose=True)
    
    # Load data from DataFrame
    pipeline.load_data(
        data_source='dataframe',
        dataframe=df,
        target_column='target'
    )
    
    # Apply only EHSO for this example
    techniques = {
        'EHSO': EHSO(k_neighbors=5, alpha=0.1, verbose=False)
    }
    pipeline.apply_resampling_techniques(techniques)
    
    # Evaluate
    results = pipeline.evaluate_techniques(classifier_name='decision_tree')
    
    return results


if __name__ == "__main__":
    print("="*70)
    print("IMBALANCED LEARNING PIPELINE DEMONSTRATION")
    print("="*70)
    
    # Example 1: Complete pipeline with synthetic data
    print("\n" + "-"*70)
    print("Example 1: Complete Pipeline with Synthetic Data")
    print("-"*70)
    
    pipeline = ImbalancedLearningPipeline(verbose=True)
    results = pipeline.run_complete_pipeline(
        data_source='synthetic',
        data_kwargs={
            'n_samples': 400,
            'n_features': 10,
            'imbalance_ratio': 5,
            'overlap_degree': 0.35
        },
        classifier_name='decision_tree',
        visualize=True
    )
    
    # Example 2: Using with DataFrame
    print("\n" + "-"*70)
    print("Example 2: Using Pipeline with DataFrame")
    print("-"*70)
    
    df_results = example_with_dataframe()
    
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETED")
    print("="*70)