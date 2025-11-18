"""
EHSO: Evolutionary Hybrid Sampling in Overlapping Scenarios
Based on Zhu et al. (2020) - Neurocomputing 417 (2020) 333-346
"""

import numpy as np
from typing import Tuple, Optional
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.model_selection import train_test_split
from .base_sampler import BaseSampler


class EHSO(BaseSampler):
    """
    EHSO: Evolutionary Hybrid Sampling in overlapping scenarios
    
    Combines evolutionary undersampling of overlapping majority samples
    with random oversampling of minority samples to achieve balance.
    
    Reference: Zhu et al. (2020) - Neurocomputing 417 (2020) 333-346
    """
    
    def __init__(self, 
                 k_neighbors: int = 5,
                 alpha: float = 0.1,
                 population_size: int = 10,
                 max_iterations: int = 30,
                 hux_threshold: float = 0.25,
                 mutation_ratio: float = 0.35,
                 random_state: Optional[int] = None,
                 verbose: bool = True):
        """
        Parameters:
        -----------
        k_neighbors : int, default=5
            Number of nearest neighbors for overlapping detection
        alpha : float, default=0.1
            Weight parameter in fitness function (paper finds 0.1 optimal)
        population_size : int, default=10
            CHC population size
        max_iterations : int, default=30
            Maximum CHC iterations
        hux_threshold : float, default=0.25
            Initial HUX crossover threshold
        mutation_ratio : float, default=0.35
            Mutation ratio for cataclysmic mutation
        random_state : int, optional
            Random state for reproducibility
        verbose : bool, default=True
            Whether to print progress information
        """
        self.k_neighbors = k_neighbors
        self.alpha = alpha
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.hux_threshold = hux_threshold
        self.mutation_ratio = mutation_ratio
        self.random_state = random_state
        self.verbose = verbose
        
        if random_state is not None:
            np.random.seed(random_state)
        
        # Store results
        self.overlapping_indices_ = None
        self.removed_indices_ = None
        self.stats_ = {}
    
    def detect_overlapping_region(self, X_maj, X_min, y_min):
        """Detect overlapping region using k-NN (Algorithm 1 from paper)"""
        X_combined = np.vstack([X_maj, X_min])
        
        nbrs = NearestNeighbors(n_neighbors=self.k_neighbors + 1)
        nbrs.fit(X_combined)
        
        overlapping_indices = []
        
        for i, x in enumerate(X_maj):
            distances, indices = nbrs.kneighbors([x])
            neighbor_indices = indices[0][1:]  # Remove self
            
            # Check if any neighbor is from minority class
            neighbor_is_minority = neighbor_indices >= len(X_maj)
            if np.any(neighbor_is_minority):
                overlapping_indices.append(i)
        
        return np.array(overlapping_indices)
    
    def calculate_IR(self, n_maj, n_min):
        """Calculate Imbalanced Ratio (Equation 3)"""
        return n_maj / n_min if n_min > 0 else float('inf')
    
    def calculate_OR(self, X_maj_overlap, X_min, selected_mask):
        """Calculate Overlapping Ratio (Equation 4)"""
        if not np.any(selected_mask):
            return 0
        
        selected_X = X_maj_overlap[selected_mask]
        nbrs = NearestNeighbors(n_neighbors=self.k_neighbors)
        nbrs.fit(X_min)
        
        total_or = 0
        for x in selected_X:
            distances, indices = nbrs.kneighbors([x])
            total_or += 1
            
        return total_or / len(selected_X) if len(selected_X) > 0 else 0
    
    def calculate_gmean(self, X_train, y_train, selected_mask):
        """Calculate G-mean using 1-NN classifier"""
        if not np.any(selected_mask):
            return 0
        
        # Create dataset with selected majority samples removed
        remaining_indices = np.ones(len(X_train), dtype=bool)
        if len(self.overlapping_indices_) > 0:
            remaining_indices[self.overlapping_indices_[selected_mask]] = False
        
        X_remaining = X_train[remaining_indices]
        y_remaining = y_train[remaining_indices]
        
        if len(X_remaining) < 4:
            return 0
            
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_remaining, y_remaining, test_size=0.3, 
                stratify=y_remaining, random_state=42
            )
            
            clf = KNeighborsClassifier(n_neighbors=1)
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_te)
            
            tp = np.sum((y_te == 1) & (y_pred == 1))
            tn = np.sum((y_te == 0) & (y_pred == 0))
            fn = np.sum((y_te == 1) & (y_pred == 0))
            fp = np.sum((y_te == 0) & (y_pred == 1))
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            gmean = np.sqrt(tpr * tnr)
            return gmean
        except:
            return 0
    
    def fitness_function(self, chromosome, X_maj_overlap, X_min, X_all, y_all):
        """Fitness function (Equation 7)"""
        selected_mask = chromosome.astype(bool)
        
        n_maj_remaining = np.sum(~selected_mask) + (len(X_all) - len(X_maj_overlap))
        n_min = len(X_min)
        
        IR = self.calculate_IR(n_maj_remaining, n_min)
        OR = self.calculate_OR(X_maj_overlap, X_min, selected_mask)
        GM = self.calculate_gmean(X_all, y_all, selected_mask)
        
        fitness = self.alpha * ((1 - OR) / IR) + (1 - self.alpha) * GM
        return fitness
    
    def hux_crossover(self, parent1, parent2, threshold):
        """HUX (Half Uniform Crossover)"""
        diff_positions = np.where(parent1 != parent2)[0]
        
        if len(diff_positions) < threshold:
            return None, None
        
        n_swap = len(diff_positions) // 2
        swap_positions = np.random.choice(diff_positions, n_swap, replace=False)
        
        offspring1 = parent1.copy()
        offspring2 = parent2.copy()
        
        offspring1[swap_positions] = parent2[swap_positions]
        offspring2[swap_positions] = parent1[swap_positions]
        
        return offspring1, offspring2
    
    def cataclysmic_mutation(self, population, best_individual):
        """Cataclysmic mutation"""
        new_population = [best_individual.copy()]
        
        for _ in range(len(population) - 1):
            mutated = best_individual.copy()
            mutation_mask = np.random.random(len(mutated)) < self.mutation_ratio
            mutated[mutation_mask] = 1 - mutated[mutation_mask]
            new_population.append(mutated)
            
        return np.array(new_population)
    
    def evolutionary_undersampling(self, X_maj_overlap, X_min, X_all, y_all):
        """CHC evolutionary algorithm for selecting majority samples to remove"""
        if len(self.overlapping_indices_) == 0:
            return np.array([])
        
        chromosome_length = len(self.overlapping_indices_)
        threshold = chromosome_length * self.hux_threshold
        threshold_gradient = 1
        
        # Initialize population
        population = np.random.randint(0, 2, (self.population_size, chromosome_length))
        
        # Evaluate initial population
        fitness_scores = np.array([
            self.fitness_function(ind, X_maj_overlap, X_min, X_all, y_all) 
            for ind in population
        ])
        
        best_fitness = np.max(fitness_scores)
        best_individual = population[np.argmax(fitness_scores)].copy()
        generations_without_improvement = 0
        
        for generation in range(self.max_iterations):
            # Crossover
            new_population = []
            
            for i in range(0, self.population_size - 1, 2):
                parent1 = population[i]
                parent2 = population[i + 1]
                
                offspring1, offspring2 = self.hux_crossover(parent1, parent2, threshold)
                
                if offspring1 is not None:
                    new_population.extend([offspring1, offspring2])
                else:
                    new_population.extend([parent1.copy(), parent2.copy()])
            
            # Combine populations
            combined_population = np.vstack([population, new_population[:self.population_size]])
            
            # Evaluate
            combined_fitness = np.array([
                self.fitness_function(ind, X_maj_overlap, X_min, X_all, y_all) 
                for ind in combined_population
            ])
            
            # Elitist selection
            elite_indices = np.argsort(combined_fitness)[-self.population_size:]
            population = combined_population[elite_indices]
            fitness_scores = combined_fitness[elite_indices]
            
            # Update best
            if np.max(fitness_scores) > best_fitness:
                best_fitness = np.max(fitness_scores)
                best_individual = population[np.argmax(fitness_scores)].copy()
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1
            
            threshold = max(0, threshold - threshold_gradient)
            
            # Cataclysmic mutation if converged
            if generations_without_improvement >= 10 or threshold == 0:
                population = self.cataclysmic_mutation(population, best_individual)
                threshold = chromosome_length * self.hux_threshold
                generations_without_improvement = 0
        
        return best_individual
    
    def random_oversampling(self, X_min, y_min, target_size):
        """Random oversampling to balance the dataset"""
        n_samples_needed = target_size - len(X_min)
        
        if n_samples_needed <= 0:
            return X_min, y_min
        
        indices = np.random.choice(len(X_min), n_samples_needed, replace=True)
        X_oversampled = np.vstack([X_min, X_min[indices]])
        y_oversampled = np.hstack([y_min, y_min[indices]])
        
        return X_oversampled, y_oversampled
    
    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Main EHSO algorithm implementation
        
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
        # Separate classes
        X_maj, y_maj, X_min, y_min, maj_class, min_class = self._separate_classes(X, y)
        
        # Store initial statistics
        self.stats_['original'] = {
            'n_majority': len(X_maj),
            'n_minority': len(X_min),
            'imbalance_ratio': len(X_maj) / len(X_min)
        }
        
        if self.verbose:
            print(f"EHSO: Original IR={self.stats_['original']['imbalance_ratio']:.2f} "
                  f"(Maj={len(X_maj)}, Min={len(X_min)})")
        
        # Step 1: Detect overlapping region
        self.overlapping_indices_ = self.detect_overlapping_region(X_maj, X_min, y_min)
        
        if self.verbose:
            print(f"EHSO: Found {len(self.overlapping_indices_)} overlapping majority samples")
        
        if len(self.overlapping_indices_) == 0:
            if self.verbose:
                print("EHSO: No overlap detected, applying only ROS")
            X_min_ros, y_min_ros = self.random_oversampling(X_min, y_min, len(X_maj))
            return np.vstack([X_maj, X_min_ros]), np.hstack([y_maj, y_min_ros])
        
        # Step 2: Evolutionary undersampling
        X_maj_overlap = X_maj[self.overlapping_indices_]
        best_chromosome = self.evolutionary_undersampling(X_maj_overlap, X_min, X, y)
        
        # Remove selected majority samples
        samples_to_remove = self.overlapping_indices_[best_chromosome.astype(bool)]
        self.removed_indices_ = samples_to_remove
        
        mask = np.ones(len(X_maj), dtype=bool)
        mask[samples_to_remove] = False
        X_maj_reduced = X_maj[mask]
        y_maj_reduced = y_maj[mask]
        
        if self.verbose:
            print(f"EHSO: Removed {len(samples_to_remove)} majority samples")
        
        # Step 3: Random oversampling
        X_min_ros, y_min_ros = self.random_oversampling(X_min, y_min, len(X_maj_reduced))
        
        # Store final statistics
        self.stats_['final'] = {
            'n_majority': len(X_maj_reduced),
            'n_minority': len(X_min_ros),
            'imbalance_ratio': 1.0,
            'n_removed': len(samples_to_remove),
            'n_oversampled': len(X_min_ros) - len(X_min)
        }
        
        if self.verbose:
            print(f"EHSO: Final balanced dataset (Maj={len(X_maj_reduced)}, Min={len(X_min_ros)})")
        
        # Combine results
        X_resampled = np.vstack([X_maj_reduced, X_min_ros])
        y_resampled = np.hstack([y_maj_reduced, y_min_ros])
        
        return X_resampled, y_resampled
