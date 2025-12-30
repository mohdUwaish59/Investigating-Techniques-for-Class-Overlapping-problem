"""
Complexity Measures for Class Overlap Analysis
Using local complexity.py implementation (copied from pycol)
"""
import numpy as np
import pandas as pd
import tempfile
import os
from typing import Dict, List, Optional

try:
    from complexity import Complexity
    COMPLEXITY_AVAILABLE = True
except ImportError as e:
    COMPLEXITY_AVAILABLE = False
    print(f"Warning: complexity.py import failed: {e}")
    print("Using fallback implementation.")


class ComplexityMeasures:
    """
    Comprehensive class overlap analysis using local complexity implementation.
    Falls back to basic implementation if complexity.py is not available.
    """
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Initialize complexity measures calculator.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
        y : np.ndarray
            Target labels (n_samples,)
        """
        self.X = X
        self.y = y
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        
        # Initialize based on complexity.py availability
        if COMPLEXITY_AVAILABLE:
            self._init_complexity()
        else:
            self._init_fallback()
    
    def _init_complexity(self):
        """Initialize with local complexity.py implementation."""
        try:
            # Create temporary CSV file for complexity.py
            self._temp_file = None
            self._complexity_obj = None
            self._create_temp_csv()
            self.use_complexity = True
        except Exception as e:
            print(f"Warning: Failed to initialize complexity.py: {e}")
            self._init_fallback()
    
    def _init_fallback(self):
        """Initialize with fallback implementation."""
        self.use_complexity = False
        self._temp_file = None
        self._complexity_obj = None
        print("Using fallback complexity measures. Check complexity.py file.")
    
    def _create_temp_csv(self):
        """Create temporary CSV file for complexity.py."""
        # Create DataFrame
        df = pd.DataFrame(self.X, columns=[f'feature_{i}' for i in range(self.n_features)])
        df['class'] = self.y
        
        # Create temporary CSV file
        self._temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        
        # Write CSV data
        df.to_csv(self._temp_file.name, index=False)
        self._temp_file.close()
        
        # Initialize complexity object
        self._complexity_obj = Complexity(
            file_name=self._temp_file.name,
            distance_func="default",
            file_type="csv"
        )
    
    def __del__(self):
        """Clean up temporary file."""
        if self._temp_file and os.path.exists(self._temp_file.name):
            os.unlink(self._temp_file.name)
    
    # Feature Overlap Measures
    def calculate_feature_overlap(self) -> Dict:
        """
        Calculate Feature Overlap measures.
        
        Returns
        -------
        dict : Feature overlap measures
        """
        try:
            return {
                'F1': self._complexity_obj.F1(),           # Maximum Fisher's Discriminant Ratio
                'F1v': self._complexity_obj.F1v(),         # Directional Vector Maximum Fisher's
                'F2': self._complexity_obj.F2(),           # Volume of Overlapping Region
                'F3': self._complexity_obj.F3(),           # Maximum Individual Feature Efficiency
                'F4': self._complexity_obj.F4(),           # Collective Feature Efficiency
                'IN': self._complexity_obj.input_noise()   # Input Noise
            }
        except Exception as e:
            print(f"Warning: Error calculating feature overlap measures: {e}")
            return {}
    
    # Instance Overlap Measures
    def calculate_instance_overlap(self) -> Dict:
        """
        Calculate Instance Overlap measures.
        
        Returns
        -------
        dict : Instance overlap measures
        """
        try:
            measures = {
                'R_value': self._complexity_obj.R_value(),     # R-value
                'N3': self._complexity_obj.N3(),               # Error Rate of 1-NN Classifier
                'SI': self._complexity_obj.SI(),               # Separability Index
                'N4': self._complexity_obj.N4(),               # Non-Linearity of 1-NN Classifier
                'CM': self._complexity_obj.CM(),               # Complexity Metric (k-NN based)
                'kDN': self._complexity_obj.kDN(),             # K-Disagreeing Neighbours
                'D3': self._complexity_obj.D3_value(),         # Class Density in Overlap Region
                'deg_overlap': self._complexity_obj.deg_overlap(),  # Degree of Overlap
                'borderline': self._complexity_obj.borderline()     # Borderline examples
            }
            
            return measures
        except Exception as e:
            print(f"Warning: Error calculating instance overlap measures: {e}")
            return {}
    
    # Structural Overlap Measures
    def calculate_structural_overlap(self) -> Dict:
        """
        Calculate Structural Overlap measures.
        
        Returns
        -------
        dict : Structural overlap measures
        """
        try:
            measures = {
                'N1': self._complexity_obj.N1(),               # Fraction of Borderline Points
                'T1': self._complexity_obj.T1(),               # Fraction of Hyperspheres
                'N2': self._complexity_obj.N2(),               # Ratio of Intra/Extra Class NN Distance
                'LSC': self._complexity_obj.LSC(),             # Local Set Cardinality
                'Clst': self._complexity_obj.Clust(),          # Number of Clusters
                'ONB': self._complexity_obj.ONB(),             # Overlap Number of Balls
                'DBC': self._complexity_obj.DBC(),             # Decision Boundary Complexity
                'NSG': self._complexity_obj.NSG(),             # Number of samples per group
                'ICSV': self._complexity_obj.ICSV()            # Inter-class scale variation
            }
            
            return measures
        except Exception as e:
            print(f"Warning: Error calculating structural overlap measures: {e}")
            return {}
    
    # Multiresolution Overlap Measures
    def calculate_multiresolution_overlap(self) -> Dict:
        """
        Calculate Multiresolution Overlap measures.
        
        Returns
        -------
        dict : Multiresolution overlap measures
        """
        try:
            return {
                'MRCA': self._complexity_obj.MRCA(),           # Multiresolution Complexity Analysis
                'C1': self._complexity_obj.C1(),               # Case Base Complexity Profile
                'C2': self._complexity_obj.C2(),               # Similarity-Weighted Case Base
                'purity': self._complexity_obj.purity(),       # Purity
                'neighbourhood_separability': self._complexity_obj.neighbourhood_separability()
            }
        except Exception as e:
            print(f"Warning: Error calculating multiresolution overlap measures: {e}")
            return {}
    
    def calculate_n3(self) -> Dict:
        """
        Calculate N3: Error Rate of the Nearest Neighbour Classifier.
        
        Returns
        -------
        dict : N3 results with overall score
        """
        try:
            n3_score = self._complexity_obj.N3()
            return {
                'overall': n3_score,
                'interpretation': self._interpret_n3(n3_score)
            }
        except Exception as e:
            print(f"Warning: Error calculating N3: {e}")
            return {'overall': 0.0, 'interpretation': 'unknown'}
    
    def calculate_t1(self) -> Dict:
        """
        Calculate T1: Fraction of Hyperspheres Covering Data.
        
        Returns
        -------
        dict : T1 results with normalized score
        """
        try:
            t1_score = self._complexity_obj.T1()
            return {
                'normalized': t1_score,
                'interpretation': self._interpret_t1(t1_score)
            }
        except Exception as e:
            print(f"Warning: Error calculating T1: {e}")
            return {'normalized': 0.0, 'interpretation': 'unknown'}
    
    def analyze_overlap(self, include_all: bool = False) -> Dict:
        """
        Comprehensive overlap analysis using local complexity implementation.
        
        Parameters
        ----------
        include_all : bool, default=False
            If True, calculate all available measures (slower)
            If False, calculate only key measures (N3, T1, N1, F1)
        
        Returns
        -------
        dict : Dictionary containing complexity measures and interpretation
        """
        if self.use_complexity and self._complexity_obj:
            return self._analyze_with_complexity(include_all)
        else:
            return self._analyze_with_fallback()
    
    def _analyze_with_complexity(self, include_all: bool = False) -> Dict:
        """Analyze using local complexity.py implementation."""
        if include_all:
            # Calculate all measures
            results = {
                'feature_overlap': self.calculate_feature_overlap(),
                'instance_overlap': self.calculate_instance_overlap(),
                'structural_overlap': self.calculate_structural_overlap(),
                'multiresolution_overlap': self.calculate_multiresolution_overlap()
            }
        else:
            # Calculate key measures only
            try:
                results = {
                    'n3': {'overall': self._complexity_obj.N3()},
                    't1': {'normalized': self._complexity_obj.T1()},
                    'n1': self._complexity_obj.N1(),  # Fraction of borderline points
                    'f1': self._complexity_obj.F1(),  # Fisher's discriminant ratio
                    'n2': self._complexity_obj.N2(),  # Intra/Extra class NN distance ratio
                    'si': self._complexity_obj.SI()   # Separability Index
                }
            except Exception as e:
                print(f"Warning: Error calculating key measures with complexity.py: {e}")
                return self._analyze_with_fallback()
        
        # Add interpretation
        results['interpretation'] = self._interpret_results(results)
        return results
    
    def _analyze_with_fallback(self) -> Dict:
        """Analyze using fallback implementation."""
        try:
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.model_selection import LeaveOneOut
            
            # Simple N3 calculation
            knn = KNeighborsClassifier(n_neighbors=1)
            loo = LeaveOneOut()
            errors = 0
            
            for train_idx, test_idx in loo.split(self.X):
                X_train, X_test = self.X[train_idx], self.X[test_idx]
                y_train, y_test = self.y[train_idx], self.y[test_idx]
                
                knn.fit(X_train, y_train)
                pred = knn.predict(X_test)
                
                if pred != y_test:
                    errors += 1
            
            n3_score = errors / len(self.X)
            
            # Simple T1 approximation (placeholder)
            t1_score = min(0.8, n3_score * 2)  # Rough approximation
            
            results = {
                'n3': {'overall': n3_score},
                't1': {'normalized': t1_score},
                'n1': n3_score * 0.8,  # Approximation
                'f1': max(0.1, 1.0 - n3_score),  # Approximation
                'n2': 1.0,  # Default
                'si': max(0.1, 1.0 - n3_score)  # Approximation
            }
            
            # Add interpretation
            results['interpretation'] = self._interpret_results(results)
            return results
            
        except Exception as e:
            print(f"Warning: Error in fallback complexity calculation: {e}")
            return {
                'n3': {'overall': 0.0},
                't1': {'normalized': 0.0},
                'interpretation': {
                    'overall_complexity': 'unknown',
                    'instance_overlap': 'unknown',
                    'structural_overlap': 'unknown',
                    'recommendations': ['Error calculating complexity measures']
                }
            }
    
    def _interpret_n3(self, n3_score: float) -> str:
        """Interpret N3 score."""
        if n3_score < 0.1:
            return 'low'
        elif n3_score < 0.3:
            return 'medium'
        else:
            return 'high'
    
    def _interpret_t1(self, t1_score: float) -> str:
        """Interpret T1 score."""
        if t1_score < 0.2:
            return 'low'
        elif t1_score < 0.4:
            return 'medium'
        else:
            return 'high'
    
    def _interpret_results(self, results: Dict) -> Dict:
        """
        Interpret complexity results and provide recommendations.
        
        Parameters
        ----------
        results : dict
            Complexity measure results
        
        Returns
        -------
        dict : Interpretation and recommendations
        """
        interpretation = {
            'overall_complexity': 'low',
            'instance_overlap': 'low',
            'structural_overlap': 'low',
            'recommendations': []
        }
        
        # Extract key measures
        n3_score = 0.0
        t1_score = 0.0
        
        if 'n3' in results and 'overall' in results['n3']:
            n3_score = results['n3']['overall']
        elif 'instance_overlap' in results and 'N3' in results['instance_overlap']:
            n3_score = results['instance_overlap']['N3']
        
        if 't1' in results and 'normalized' in results['t1']:
            t1_score = results['t1']['normalized']
        elif 'structural_overlap' in results and 'T1' in results['structural_overlap']:
            t1_score = results['structural_overlap']['T1']
        
        # Thresholds based on literature
        n3_threshold = 0.2  # 20% error rate
        t1_threshold = 0.3  # 30% fragmentation
        
        # Interpret N3 (Instance Overlap)
        if n3_score > n3_threshold:
            interpretation['instance_overlap'] = 'high'
            interpretation['recommendations'].append(
                'High instance overlap detected (N3={:.3f}). Consider instance-based techniques: NBUS, RFCL, URNS, or OSM.'.format(n3_score)
            )
        elif n3_score > n3_threshold/2:
            interpretation['instance_overlap'] = 'medium'
        
        # Interpret T1 (Structural Overlap)
        if t1_score > t1_threshold:
            interpretation['structural_overlap'] = 'high'
            interpretation['recommendations'].append(
                'High structural overlap detected (T1={:.3f}). Consider clustering-based approaches: KMeans variants or OSM.'.format(t1_score)
            )
        elif t1_score > t1_threshold/2:
            interpretation['structural_overlap'] = 'medium'
        
        # Overall complexity
        if n3_score > n3_threshold or t1_score > t1_threshold:
            interpretation['overall_complexity'] = 'high'
            if len(interpretation['recommendations']) == 0:
                interpretation['recommendations'].append(
                    'High complexity detected. Consider comprehensive approaches: OSM or EHSO.'
                )
        elif n3_score > n3_threshold/2 or t1_score > t1_threshold/2:
            interpretation['overall_complexity'] = 'medium'
        
        # Additional recommendations based on other measures
        if 'n1' in results:
            n1_val = results['n1']
            if isinstance(n1_val, (list, np.ndarray)):
                n1_val = np.mean(n1_val) if len(n1_val) > 0 else 0
            if n1_val > 0.5:
                interpretation['recommendations'].append(
                    'High fraction of borderline points (N1={:.3f}). Borderline-focused techniques recommended.'.format(n1_val)
                )
        
        if 'f1' in results:
            f1_val = results['f1']
            if isinstance(f1_val, (list, np.ndarray)):
                f1_val = np.mean(f1_val) if len(f1_val) > 0 else 1.0
            if f1_val < 1.0:
                interpretation['recommendations'].append(
                    'Low feature discriminability (F1={:.3f}). Feature selection or transformation may help.'.format(f1_val)
                )
        
        return interpretation


def compare_pre_post_overlap(X_pre: np.ndarray, y_pre: np.ndarray,
                             X_post: np.ndarray, y_post: np.ndarray,
                             include_all: bool = False) -> Dict:
    """
    Compare overlap before and after applying resampling technique.
    
    Parameters
    ----------
    X_pre : np.ndarray
        Features before preprocessing
    y_pre : np.ndarray
        Labels before preprocessing
    X_post : np.ndarray
        Features after preprocessing
    y_post : np.ndarray
        Labels after preprocessing
    include_all : bool, default=False
        If True, calculate all available measures (slower)
    
    Returns
    -------
    dict : Comparison results showing improvement in overlap measures
    """
    try:
        # Calculate measures pre-preprocessing
        cm_pre = ComplexityMeasures(X_pre, y_pre)
        results_pre = cm_pre.analyze_overlap(include_all=include_all)
        
        # Calculate measures post-preprocessing
        cm_post = ComplexityMeasures(X_post, y_post)
        results_post = cm_post.analyze_overlap(include_all=include_all)
        
        # Calculate improvements for key measures
        improvements = {}
        
        # N3 improvement
        n3_pre = results_pre.get('n3', {}).get('overall', 0.0)
        n3_post = results_post.get('n3', {}).get('overall', 0.0)
        if n3_pre > 0:
            improvements['n3'] = {
                'absolute': n3_pre - n3_post,
                'relative': (n3_pre - n3_post) / n3_pre * 100
            }
        else:
            improvements['n3'] = {'absolute': 0.0, 'relative': 0.0}
        
        # T1 improvement
        t1_pre = results_pre.get('t1', {}).get('normalized', 0.0)
        t1_post = results_post.get('t1', {}).get('normalized', 0.0)
        if t1_pre > 0:
            improvements['t1'] = {
                'absolute': t1_pre - t1_post,
                'relative': (t1_pre - t1_post) / t1_pre * 100
            }
        else:
            improvements['t1'] = {'absolute': 0.0, 'relative': 0.0}
        
        # Additional measures if available
        for measure in ['n1', 'f1', 'n2', 'si']:
            if measure in results_pre and measure in results_post:
                pre_val = results_pre[measure]
                post_val = results_post[measure]
                if isinstance(pre_val, (int, float)) and isinstance(post_val, (int, float)) and pre_val > 0:
                    improvements[measure] = {
                        'absolute': pre_val - post_val,
                        'relative': (pre_val - post_val) / pre_val * 100
                    }
        
        comparison = {
            'pre_processing': results_pre,
            'post_processing': results_post,
            'improvements': improvements
        }
        
        return comparison
        
    except Exception as e:
        print(f"Warning: Error in complexity comparison: {e}")
        # Return empty comparison with default structure
        return {
            'pre_processing': {'n3': {'overall': 0.0}, 't1': {'normalized': 0.0}},
            'post_processing': {'n3': {'overall': 0.0}, 't1': {'normalized': 0.0}},
            'improvements': {'n3': {'absolute': 0.0}, 't1': {'absolute': 0.0}}
        }


def get_available_measures() -> Dict[str, List[str]]:
    """
    Get list of available complexity measures by category.
    
    Returns
    -------
    dict : Available measures by category
    """
    if not COMPLEXITY_AVAILABLE:
        return {}
    
    return {
        'feature_overlap': [
            'F1 - Maximum Fisher\'s Discriminant Ratio',
            'F1v - Directional Vector Maximum Fisher\'s',
            'F2 - Volume of Overlapping Region',
            'F3 - Maximum Individual Feature Efficiency',
            'F4 - Collective Feature Efficiency',
            'IN - Input Noise'
        ],
        'instance_overlap': [
            'R_value - R-value',
            'N3 - Error Rate of 1-NN Classifier',
            'SI - Separability Index',
            'N4 - Non-Linearity of 1-NN Classifier',
            'kDN - K-Disagreeing Neighbours',
            'D3 - Class Density in Overlap Region',
            'CM - Complexity Metric (k-NN based)',
            'deg_overlap - Degree of Overlap',
            'borderline - Borderline Examples'
        ],
        'structural_overlap': [
            'N1 - Fraction of Borderline Points',
            'T1 - Fraction of Hyperspheres',
            'N2 - Ratio of Intra/Extra Class NN Distance',
            'Clst - Number of Clusters',
            'ONB - Overlap Number of Balls',
            'LSC - Local Set Cardinality',
            'DBC - Decision Boundary Complexity',
            'NSG - Number of samples per group',
            'ICSV - Inter-class scale variation'
        ],
        'multiresolution_overlap': [
            'MRCA - Multiresolution Complexity Analysis',
            'C1 - Case Base Complexity Profile',
            'C2 - Similarity-Weighted Case Base',
            'purity - Purity',
            'neighbourhood_separability - Neighbourhood Separability'
        ]
    }


