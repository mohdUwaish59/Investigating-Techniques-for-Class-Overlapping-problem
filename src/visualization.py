"""
Visualization Module
Provides various visualization functions for imbalanced learning analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from typing import Optional, Tuple, List
import warnings
warnings.filterwarnings('ignore')


class ImbalancedDataVisualizer:
    """
    Visualization tools for imbalanced data analysis
    """
    
    def __init__(self, figsize_single=(10, 8), figsize_comparison=(15, 6), 
                 style='seaborn-v0_8', dpi=100):
        """
        Parameters:
        -----------
        figsize_single : tuple
            Figure size for single plots
        figsize_comparison : tuple
            Figure size for comparison plots
        style : str
            Matplotlib style
        dpi : int
            DPI for saving figures
        """
        self.figsize_single = figsize_single
        self.figsize_comparison = figsize_comparison
        self.dpi = dpi
        
        if style:
            try:
                plt.style.use(style)
            except:
                # Fallback to default if style not found
                pass
        
        # Color schemes
        self.colors_majority = ['#3498db', '#2980b9', '#1f618d']  # Blues
        self.colors_minority = ['#e74c3c', '#c0392b', '#922b21']  # Reds
        self.colors_overlap = ['#f39c12', '#d68910', '#9c6d0e']   # Oranges
    
    def reduce_dimensions(self, X: np.ndarray, method: str = 'pca', 
                         n_components: int = 2, **kwargs) -> np.ndarray:
        """
        Reduce dimensions for visualization
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        method : str
            Reduction method ('pca' or 'tsne')
        n_components : int
            Number of components
        **kwargs : additional parameters for the reduction method
        
        Returns:
        --------
        X_reduced : np.ndarray
            Reduced feature matrix
        """
        if X.shape[1] <= n_components:
            return X[:, :n_components]
        
        if method == 'pca':
            reducer = PCA(n_components=n_components, **kwargs)
        elif method == 'tsne':
            reducer = TSNE(n_components=n_components, **kwargs)
        else:
            raise ValueError(f"Unknown reduction method: {method}")
        
        X_reduced = reducer.fit_transform(X)
        
        if method == 'pca' and hasattr(reducer, 'explained_variance_ratio_'):
            print(f"Variance explained: {reducer.explained_variance_ratio_.sum():.2%}")
        
        return X_reduced
    
    def plot_data_distribution(self, X: np.ndarray, y: np.ndarray, 
                              title: str = "Data Distribution",
                              overlapping_indices: Optional[np.ndarray] = None,
                              reduction_method: str = 'pca',
                              save_path: Optional[str] = None,
                              show_legend: bool = True,
                              show_stats: bool = True):
        """
        Plot data distribution with optional overlapping region highlighting
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target vector
        title : str
            Plot title
        overlapping_indices : np.ndarray, optional
            Indices of overlapping samples
        reduction_method : str
            Dimension reduction method
        save_path : str, optional
            Path to save figure
        show_legend : bool
            Whether to show legend
        show_stats : bool
            Whether to show statistics on plot
        """
        # Reduce dimensions if necessary
        X_vis = self.reduce_dimensions(X, method=reduction_method)
        
        # Identify classes
        unique_classes = np.unique(y)
        counts = [np.sum(y == c) for c in unique_classes]
        maj_class = unique_classes[np.argmax(counts)]
        min_class = unique_classes[np.argmin(counts)]
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize_single)
        
        # Plot majority class
        maj_mask = y == maj_class
        
        if overlapping_indices is not None and len(overlapping_indices) > 0:
            # Split majority into overlapping and non-overlapping
            overlap_mask = np.zeros(len(X), dtype=bool)
            overlap_mask[overlapping_indices] = True
            overlap_mask = overlap_mask & maj_mask
            
            # Non-overlapping majority
            non_overlap_maj = maj_mask & ~overlap_mask
            ax.scatter(X_vis[non_overlap_maj, 0], X_vis[non_overlap_maj, 1],
                      c=self.colors_majority[0], 
                      label=f'Majority (safe, n={np.sum(non_overlap_maj)})',
                      alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
            
            # Overlapping majority
            ax.scatter(X_vis[overlap_mask, 0], X_vis[overlap_mask, 1],
                      c=self.colors_overlap[0], 
                      label=f'Majority (overlap, n={np.sum(overlap_mask)})',
                      alpha=0.8, s=50, marker='s', edgecolors='yellow', linewidth=2)
        else:
            ax.scatter(X_vis[maj_mask, 0], X_vis[maj_mask, 1],
                      c=self.colors_majority[0], 
                      label=f'Majority (n={np.sum(maj_mask)})',
                      alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
        
        # Plot minority class
        min_mask = y == min_class
        ax.scatter(X_vis[min_mask, 0], X_vis[min_mask, 1],
                  c=self.colors_minority[0], 
                  label=f'Minority (n={np.sum(min_mask)})',
                  alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
        
        # Labels and title
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_title(title)
        
        if show_legend:
            ax.legend()
        
        ax.grid(True, alpha=0.3)
        
        # Add statistics box
        if show_stats:
            ir = max(counts) / min(counts)
            stats_text = f'IR = {ir:.2f}'
            if overlapping_indices is not None:
                overlap_ratio = len(overlapping_indices) / np.sum(maj_mask) * 100
                stats_text += f'\nOverlap = {overlap_ratio:.1f}%'
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=11, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        plt.show()
    
    def plot_before_after_comparison(self, X_original: np.ndarray, y_original: np.ndarray,
                                    X_resampled: np.ndarray, y_resampled: np.ndarray,
                                    technique_name: str = "Resampling",
                                    reduction_method: str = 'pca',
                                    save_path: Optional[str] = None):
        """
        Plot side-by-side comparison of data before and after resampling
        
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
        reduction_method : str
            Dimension reduction method
        save_path : str, optional
            Path to save figure
        """
        # Fit reducer on original data and apply to both
        if X_original.shape[1] > 2:
            if reduction_method == 'pca':
                reducer = PCA(n_components=2)
            else:
                reducer = TSNE(n_components=2, random_state=42)
            
            X_orig_vis = reducer.fit_transform(X_original)
            X_res_vis = reducer.transform(X_resampled) if reduction_method == 'pca' else reducer.fit_transform(X_resampled)
        else:
            X_orig_vis = X_original[:, :2]
            X_res_vis = X_resampled[:, :2]
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize_comparison)
        
        # Identify classes
        unique_classes = np.unique(y_original)
        
        # Plot original data
        for i, cls in enumerate(unique_classes):
            mask = y_original == cls
            color = self.colors_majority[0] if i == 0 else self.colors_minority[0]
            ax1.scatter(X_orig_vis[mask, 0], X_orig_vis[mask, 1],
                       c=color, label=f'Class {cls} (n={np.sum(mask)})',
                       alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
        
        # Calculate and display IR for original
        counts_orig = [np.sum(y_original == c) for c in unique_classes]
        ir_orig = max(counts_orig) / min(counts_orig)
        ax1.set_title(f'Original Data\nIR = {ir_orig:.2f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel('Component 1')
        ax1.set_ylabel('Component 2')
        
        # Plot resampled data
        for i, cls in enumerate(unique_classes):
            mask = y_resampled == cls
            color = self.colors_majority[0] if i == 0 else self.colors_minority[0]
            ax2.scatter(X_res_vis[mask, 0], X_res_vis[mask, 1],
                       c=color, label=f'Class {cls} (n={np.sum(mask)})',
                       alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
        
        # Calculate and display IR for resampled
        counts_res = [np.sum(y_resampled == c) for c in unique_classes]
        ir_res = max(counts_res) / min(counts_res)
        ax2.set_title(f'After {technique_name}\nIR = {ir_res:.2f}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel('Component 1')
        ax2.set_ylabel('Component 2')
        
        plt.suptitle(f'{technique_name} Results', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        plt.show()
    
    def plot_class_distribution_bar(self, datasets: dict, 
                                   save_path: Optional[str] = None):
        """
        Plot bar chart comparing class distributions across multiple datasets
        
        Parameters:
        -----------
        datasets : dict
            Dictionary of {name: y_array} pairs
        save_path : str, optional
            Path to save figure
        """
        fig, ax = plt.subplots(figsize=self.figsize_single)
        
        names = list(datasets.keys())
        n_datasets = len(names)
        
        # Calculate statistics for each dataset
        stats = []
        for name, y in datasets.items():
            unique, counts = np.unique(y, return_counts=True)
            stats.append({
                'name': name,
                'majority': max(counts),
                'minority': min(counts),
                'ir': max(counts) / min(counts)
            })
        
        # Create bars
        x = np.arange(n_datasets)
        width = 0.35
        
        majority_counts = [s['majority'] for s in stats]
        minority_counts = [s['minority'] for s in stats]
        
        bars1 = ax.bar(x - width/2, majority_counts, width, 
                      label='Majority', color=self.colors_majority[0])
        bars2 = ax.bar(x + width/2, minority_counts, width,
                      label='Minority', color=self.colors_minority[0])
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{int(height)}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom')
        
        # Add IR values above
        for i, s in enumerate(stats):
            ax.text(i, max(s['majority'], s['minority']) * 1.1,
                   f"IR={s['ir']:.2f}", ha='center', fontweight='bold')
        
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Number of Samples')
        ax.set_title('Class Distribution Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        plt.show()
    
    def plot_overlap_detection(self, X: np.ndarray, y: np.ndarray,
                              k_neighbors: int = 5,
                              reduction_method: str = 'pca',
                              save_path: Optional[str] = None):
        """
        Visualize overlap detection process
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target vector
        k_neighbors : int
            Number of neighbors for overlap detection
        reduction_method : str
            Dimension reduction method
        save_path : str, optional
            Path to save figure
        """
        # Detect overlapping samples
        unique_classes = np.unique(y)
        counts = [np.sum(y == c) for c in unique_classes]
        maj_class = unique_classes[np.argmax(counts)]
        min_class = unique_classes[np.argmin(counts)]
        
        X_maj = X[y == maj_class]
        X_min = X[y == min_class]
        
        # Find overlapping majority samples
        X_combined = np.vstack([X_maj, X_min])
        nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1)
        nbrs.fit(X_combined)
        
        overlapping_indices = []
        for i, x in enumerate(X_maj):
            distances, indices = nbrs.kneighbors([x])
            neighbor_indices = indices[0][1:]
            if np.any(neighbor_indices >= len(X_maj)):
                overlapping_indices.append(i)
        
        # Convert to global indices
        maj_global_indices = np.where(y == maj_class)[0]
        overlapping_global = maj_global_indices[overlapping_indices]
        
        # Visualize
        self.plot_data_distribution(X, y, 
                                   title=f"Overlap Detection (k={k_neighbors})",
                                   overlapping_indices=overlapping_global,
                                   reduction_method=reduction_method,
                                   save_path=save_path)
        
        return overlapping_global


# Utility function for quick visualization
def quick_plot(X, y, title="Data Distribution"):
    """Quick plot function for convenience"""
    viz = ImbalancedDataVisualizer()
    viz.plot_data_distribution(X, y, title=title)


if __name__ == "__main__":
    # Test the visualization module
    from data_loader import DataLoader
    
    print("Testing Visualization Module")
    print("="*60)
    
    # Create synthetic data
    loader = DataLoader()
    X, y = loader.create_synthetic_data(n_samples=300, n_features=10, 
                                       imbalance_ratio=3, overlap_degree=0.3)
    
    # Initialize visualizer
    viz = ImbalancedDataVisualizer()
    
    # Test different visualizations
    print("\n1. Basic data distribution plot:")
    viz.plot_data_distribution(X, y, title="Test Data Distribution")
    
    print("\n2. Overlap detection visualization:")
    overlapping_indices = viz.plot_overlap_detection(X, y, k_neighbors=5)
    print(f"Found {len(overlapping_indices)} overlapping samples")
    
    print("\nVisualization tests completed!")