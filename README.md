# ⚖️ Imbalanced Learning Toolkit

A comprehensive, modular toolkit for experimenting with imbalanced datasets. Complete control over data loading, resampling techniques, model evaluation, and visualization.

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run the Streamlit UI

```bash
streamlit run app.py
```

### Run Command Line Scripts

```bash
# Run the basic pipeline
python src/run.py

# Create imbalanced dataset
python make_imbalanced.py
```

## 📋 Features

### 🎛️ Complete User Control

- **Data Sources**:
  - Upload your own CSV files
  - Generate synthetic imbalanced data
  - Use sample datasets
  
- **Resampling Techniques**:
  - T1: RFCL (Random Forest Cleaning Rule - handles class overlap)
  - T2: SVDDWSMOTE (SVDD-based overlap handler - removes noisy instances)
  - T4: NBUS (Neighbourhood-Based Undersampling - 4 variants)
    - NB-Basic: Basic neighbourhood search
    - NB-Tomek: Modified Tomek link search
    - NB-Comm: Common nearest neighbours search
    - NB-Rec: Recursive search
  - EHSO (Evolutionary Hybrid Sampling in Overlapping scenarios)
  - Random Oversampling
  - Random Undersampling
  - Easy to add more techniques

- **Classifiers**:
  - Decision Tree
  - Random Forest
  - Logistic Regression
  - K-Nearest Neighbors
  - Naive Bayes
  - Support Vector Machine

- **Evaluation Metrics**:
  - Accuracy & Balanced Accuracy
  - Precision, Recall, F1-Score
  - Specificity & Sensitivity
  - G-mean (ideal for imbalanced data)
  - AUC-ROC

### 📊 Interactive Visualizations

- Class distribution plots
- Before/after resampling comparisons
- Confusion matrices
- Performance metrics comparison
- Overlap detection visualization

### ⚙️ Customizable Parameters

- **Data Generation**: samples, features, imbalance ratio, overlap degree
- **EHSO**: k_neighbors, alpha, population_size, max_iterations
- **Classifiers**: All standard hyperparameters
- **Evaluation**: test size, random state, cross-validation

## 🏗️ Project Structure

```
.
├── app.py                      # Streamlit UI (main interface)
├── make_imbalanced.py          # Script to create imbalanced datasets
├── requirements.txt            # Python dependencies
├── data/
│   ├── data.csv               # Original dataset
│   └── data_imbalanced.csv    # Imbalanced version
└── src/
    ├── techniques/            # 🆕 Modular resampling techniques
    │   ├── __init__.py       # Package initialization
    │   ├── base_sampler.py   # Abstract base class
    │   ├── ehso.py           # EHSO technique
    │   ├── rfcl.py           # RFCL technique
    │   ├── random_oversampler.py
    │   ├── random_undersampler.py
    │   └── README.md         # Techniques documentation
    ├── data_loader.py         # Data loading utilities
    ├── resampling_techniques.py  # Imports from techniques/
    ├── visualization.py       # Visualization tools
    ├── model_evaluation.py    # Model training & evaluation
    ├── main_pipeline.py       # Complete pipeline
    └── run.py                 # Command-line script
```

## 🎯 Usage Examples

### Using the Streamlit UI

1. **Load Data**:
   - Upload CSV or generate synthetic data
   - Select target column
   - Choose features

2. **Apply Resampling**:
   - Select techniques (EHSO, ROS, RUS)
   - Configure parameters
   - Click "Apply Resampling"

3. **Evaluate**:
   - Choose classifier
   - Set parameters
   - Click "Evaluate Models"

4. **Analyze Results**:
   - View metrics table
   - Compare visualizations
   - Download results

### Using Python Scripts

```python
from main_pipeline import ImbalancedLearningPipeline
from resampling_techniques import EHSO

# Initialize pipeline
pipeline = ImbalancedLearningPipeline(verbose=True)

# Load data
pipeline.load_data(
    data_source='csv',
    filepath='data/data.csv',
    target_column='stabf'
)

# Apply techniques
techniques = {
    'EHSO': EHSO(k_neighbors=5, alpha=0.1)
}
pipeline.apply_resampling_techniques(techniques)

# Evaluate
results = pipeline.evaluate_techniques(classifier_name='decision_tree')

# Get best technique
best = pipeline.get_best_technique(metric='g_mean')
```

## 🔧 Adding New Techniques

To add a new resampling technique:

1. Create a class in `src/resampling_techniques.py`:

```python
class YourNewTechnique(BaseSampler):
    def __init__(self, param1=default1, param2=default2):
        self.param1 = param1
        self.param2 = param2
    
    def fit_resample(self, X, y):
        # Your implementation
        return X_resampled, y_resampled
```

2. Add it to the UI in `app.py`:

```python
available_techniques = {
    "Your New Technique": "Description",
    # ... existing techniques
}
```

3. Add instantiation logic:

```python
elif technique == "Your New Technique":
    sampler = YourNewTechnique(param1=value1, param2=value2)
```

## 📊 Supported Metrics

- **Accuracy**: Overall correctness
- **Balanced Accuracy**: Better for imbalanced data
- **Precision**: Positive prediction accuracy
- **Recall**: True positive rate
- **F1-Score**: Harmonic mean of precision and recall
- **Specificity**: True negative rate
- **G-mean**: Geometric mean (ideal for imbalanced data)
- **AUC-ROC**: Area under ROC curve

## 🎓 References

**T1: RFCL** implementation based on:
> Zhang et al. (2021). "RFCL: A new under-sampling method of reducing the degree of imbalance and overlap"

**EHSO** implementation based on:
> Zhu, T., Lin, Y., & Liu, Y. (2020). Synthetic minority oversampling technique for multiclass imbalance problems. Pattern Recognition, 72, 327-340.

## 📝 License

MIT License

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Add new resampling techniques
- Improve visualizations
- Add more classifiers
- Enhance the UI

## 💡 Tips

- Start with IR > 5 for meaningful comparisons
- Use G-mean for imbalanced data evaluation
- Try multiple techniques and compare
- Adjust EHSO alpha parameter (0.1 is optimal for most cases)
- Use cross-validation for robust results
