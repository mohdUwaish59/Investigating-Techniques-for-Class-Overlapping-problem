# Resampling Techniques

This folder contains modular implementations of various resampling techniques for imbalanced learning.

## Structure

```
techniques/
├── __init__.py              # Package initialization, exports all techniques
├── base_sampler.py          # Abstract base class for all techniques
├── ehso.py                  # EHSO: Evolutionary Hybrid Sampling
├── rfcl.py                  # RFCL: Random Forest Cleaning Rule
├── random_oversampler.py    # Random Oversampling
├── random_undersampler.py   # Random Undersampling
└── README.md                # This file
```

## Available Techniques

### 1. BaseSampler (base_sampler.py)
Abstract base class that all resampling techniques inherit from.

**Key Methods:**
- `fit_resample(X, y)` - Abstract method to be implemented by subclasses
- `get_params()` - Returns technique parameters
- `_separate_classes(X, y)` - Utility to separate majority/minority classes

### 2. RFCL (rfcl.py)
**T1: Random Forest Cleaning Rule**

Handles class overlap using Random Forest margins.

**Reference:** Zhang et al. (2021)

**Parameters:**
- `final_classifier` - Classifier for threshold optimization
- `random_state` - Random seed
- `verbose` - Print progress

**Key Features:**
- Identifies overlapping majority samples
- Uses RF vote-based margins
- Optimizes threshold via 3-fold CV
- Preserves all minority samples

### 3. EHSO (ehso.py)
**Evolutionary Hybrid Sampling in Overlapping Scenarios**

Combines evolutionary undersampling with random oversampling.

**Reference:** Zhu et al. (2020) - Neurocomputing 417

**Parameters:**
- `k_neighbors` - Neighbors for overlap detection
- `alpha` - Fitness function weight
- `population_size` - CHC population size
- `max_iterations` - Maximum iterations
- `hux_threshold` - Crossover threshold
- `mutation_ratio` - Mutation ratio
- `random_state` - Random seed
- `verbose` - Print progress

**Key Features:**
- Detects overlapping regions with k-NN
- CHC evolutionary algorithm
- HUX crossover
- Cataclysmic mutation
- Balances to IR=1.0

### 4. RandomOverSampler (random_oversampler.py)
**Random Oversampling (ROS)**

Simple random duplication of minority samples.

**Parameters:**
- `sampling_strategy` - 'auto' or float
- `random_state` - Random seed

**Key Features:**
- Fast and simple
- Good baseline
- Preserves all original data

### 5. RandomUnderSampler (random_undersampler.py)
**Random Undersampling (RUS)**

Simple random removal of majority samples.

**Parameters:**
- `sampling_strategy` - 'auto' or float
- `random_state` - Random seed

**Key Features:**
- Fast and simple
- Good baseline
- Reduces dataset size

## Adding New Techniques

To add a new resampling technique:

### Step 1: Create New File

Create `src/techniques/your_technique.py`:

```python
"""
Your Technique Name
Brief description
"""

import numpy as np
from typing import Tuple
from .base_sampler import BaseSampler


class YourTechnique(BaseSampler):
    """
    Your Technique
    
    Detailed description of what it does.
    
    Reference: Author et al. (Year)
    """
    
    def __init__(self, param1=default1, param2=default2, random_state=None, verbose=True):
        """
        Parameters:
        -----------
        param1 : type
            Description
        param2 : type
            Description
        random_state : int, optional
            Random seed
        verbose : bool
            Print progress
        """
        self.param1 = param1
        self.param2 = param2
        self.random_state = random_state
        self.verbose = verbose
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply your technique
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target vector
            
        Returns:
        --------
        X_resampled : np.ndarray
            Resampled features
        y_resampled : np.ndarray
            Resampled targets
        """
        # Your implementation here
        
        # Use utility method to separate classes
        X_maj, y_maj, X_min, y_min, maj_class, min_class = self._separate_classes(X, y)
        
        # Your resampling logic
        
        return X_resampled, y_resampled
```

### Step 2: Update __init__.py

Add your technique to `src/techniques/__init__.py`:

```python
from .your_technique import YourTechnique

__all__ = [
    'BaseSampler',
    'EHSO',
    'RFCL',
    'RandomOverSampler',
    'RandomUnderSampler',
    'YourTechnique'  # Add this
]
```

### Step 3: Update UI (Optional)

If you want it in the Streamlit UI, update `app.py`:

```python
# Add to available_techniques
available_techniques = {
    "Your Technique": "Description",
    # ... existing techniques
}

# Add instantiation logic
elif technique == "Your Technique":
    sampler = YourTechnique(param1=value1, param2=value2)
```

### Step 4: Test

Create a test script:

```python
from techniques import YourTechnique

# Test your technique
technique = YourTechnique()
X_res, y_res = technique.fit_resample(X, y)
```

## Design Principles

1. **Inheritance**: All techniques inherit from `BaseSampler`
2. **Consistency**: All implement `fit_resample(X, y)`
3. **Modularity**: Each technique in its own file
4. **Documentation**: Comprehensive docstrings
5. **Parameters**: Configurable with sensible defaults
6. **Verbosity**: Optional progress printing
7. **Reproducibility**: Random state support

## Usage Examples

### Basic Usage

```python
from techniques import RFCL, EHSO

# RFCL
rfcl = RFCL(random_state=42, verbose=True)
X_rfcl, y_rfcl = rfcl.fit_resample(X, y)

# EHSO
ehso = EHSO(k_neighbors=5, alpha=0.1, verbose=True)
X_ehso, y_ehso = ehso.fit_resample(X, y)
```

### With Pipeline

```python
from main_pipeline import ImbalancedLearningPipeline
from techniques import RFCL, EHSO

pipeline = ImbalancedLearningPipeline()
pipeline.load_data(...)

techniques = {
    'T1: RFCL': RFCL(random_state=42),
    'EHSO': EHSO(k_neighbors=5)
}
pipeline.apply_resampling_techniques(techniques)
```

### Accessing Results

```python
# RFCL
print(f"Threshold: {rfcl.threshold_}")
print(f"Stats: {rfcl.stats_}")

# EHSO
print(f"Overlapping indices: {ehso.overlapping_indices_}")
print(f"Stats: {ehso.stats_}")
```

## Testing

Test individual techniques:

```bash
python -c "from techniques import RFCL; print('RFCL imported successfully')"
python -c "from techniques import EHSO; print('EHSO imported successfully')"
```

Test all techniques:

```bash
python test_rfcl.py
python src/run.py
```

## References

1. **RFCL**: Zhang et al. (2021). "RFCL: A new under-sampling method of reducing the degree of imbalance and overlap"

2. **EHSO**: Zhu, T., Lin, Y., & Liu, Y. (2020). Synthetic minority oversampling technique for multiclass imbalance problems. Pattern Recognition, 72, 327-340.

## Contributing

When adding new techniques:
1. Follow the existing code structure
2. Include comprehensive docstrings
3. Add type hints
4. Support random_state for reproducibility
5. Include verbose option
6. Add tests
7. Update documentation

---

**Maintainer**: Research Team
**Last Updated**: 2025-11-18
