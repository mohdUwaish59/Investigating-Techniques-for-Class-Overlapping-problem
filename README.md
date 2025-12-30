# Investigating Techniques for Class Overlapping Problem

A comprehensive, modular toolkit for experimenting with imbalanced datasets and class overlap analysis. Complete control over data loading, resampling techniques, model evaluation, and visualization.

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
python misc/make_imbalanced.py
```

## 📋 Features

### 🎛️ Complete User Control

- **Data Sources**:
  - Upload your own CSV files
  - Generate synthetic imbalanced data
  - Use sample datasets
  
- **Resampling Techniques**:
  - T1: RFCL (Random Forest Cleaning Rule - handles class overlap)
  - T1.1: URNS (Recursive Neighbourhood Search - overlap-based undersampling)
  - T1.3: NUS (Neighbourhood Under-Sampling - colonial neighbours method)
  - T1.4: DeviOCSVM (Devi et al. One-Class SVM - comprehensive overlap handling with Tomek links)
  - T1.5: FCMBoostOBU (Fuzzy C-Means Boosted Overlap-Based Undersampling - BLSMOTE1 + FCM clustering)
  - T2: SVDDWSMOTE (SVDD-based overlap handler - removes noisy instances)
  - T2.1: ODBOT (Outlier Detection-Based Oversampling - clustering-based synthetic generation)
  - T3: EHSO (Evolutionary Hybrid Sampling in Overlapping scenarios)
  - T4: NBUS (Neighbourhood-Based Undersampling - 4 variants)
    - NB-Basic: Basic neighbourhood search
    - NB-Tomek: Modified Tomek link search
    - NB-Comm: Common nearest neighbours search
    - NB-Rec: Recursive search
  - T5: KMeans (Clustering-Based Undersampling - 4 variants)
    - HKM: Hard K-Means clustering
    - FCM: Fuzzy C-Means clustering
    - RKM: Rough K-Means clustering
    - FRKM: Fuzzy-Rough K-Means clustering
  - T6: OSM (Overlap-Separating Model - comprehensive preprocessing)
    - ROSE balancing, Tomek removal, RF feature selection
    - Boxplot outlier removal, K-means overlap separation, SVM optimization
  - T7: ROS (Random Oversampling - randomly duplicate minority samples)
  - T8: RUS (Random Undersampling - randomly remove majority samples)

- **Complexity Analysis** (Global Feature):
  - **30+ complexity measures** using local implementation (copied from pycol)
  - **N3**: Error Rate of 1-NN Classifier (instance-level overlap)
  - **T1**: Fraction of Hyperspheres (structural overlap)
  - **N1**: Fraction of Borderline Points
  - **F1**: Maximum Fisher's Discriminant Ratio
  - Before/after comparison for all techniques
  - Visual comparison charts
  - Automatic recommendations based on complexity

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
- Complexity analysis charts

### ⚙️ Customizable Parameters

- **Data Generation**: samples, features, imbalance ratio, overlap degree
- **All Techniques**: Comprehensive parameter controls for each technique
- **Classifiers**: All standard hyperparameters
- **Evaluation**: test size, random state, cross-validation

## 🏗️ Project Structure

```
.
├── app.py                      # Streamlit UI (main interface)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── PROJECT_DETAILED_REPORT.md  # Comprehensive project documentation
├── COMPLEXITY_ANALYSIS_FEATURE.md  # Complexity analysis documentation
├── data/
│   ├── data.csv               # Original dataset
│   ├── data_imbalanced.csv    # Imbalanced version
│   ├── contraceptive+method+choice/  # Sample dataset
│   └── working/
│       └── contraceptive_method_choice.csv
├── misc/
│   ├── data.ipynb            # Data exploration notebook
│   ├── make_imbalanced.py    # Script to create imbalanced datasets
│   ├── reduce_dataset.py     # Dataset reduction utilities
│   ├── QUICK_START.md        # Quick start guide
│   └── test_*.py             # Individual technique tests
└── src/
    ├── techniques/            # 🆕 Modular resampling techniques
    │   ├── __init__.py       # Package initialization
    │   ├── base_sampler.py   # Abstract base class
    │   ├── rfcl.py           # T1: RFCL technique
    │   ├── urns.py           # T1.1: URNS technique
    │   ├── nus.py            # T1.3: NUS technique
    │   ├── svddwsmote.py     # T2: SVDDWSMOTE technique
    │   ├── ehso.py           # T3: EHSO technique
    │   ├── nbus.py           # T4: NBUS variants
    │   ├── kmeans_undersampling.py  # T5: KMeans variants
    │   ├���─ osm.py            # T6: OSM technique
    │   ├── random_oversampler.py   # T7: ROS
    │   ├── random_undersampler.py  # T8: RUS
    │   └── README.md         # Techniques documentation
    ├── complexity.py          # Local complexity measures implementation
    ├── complexity_measures.py # Complexity analysis interface
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
   - Select techniques (multiple options available)
   - Configure parameters for each technique
   - Enable complexity analysis
   - Click "Apply Resampling"

3. **Evaluate**:
   - Choose classifier
   - Set parameters
   - Click "Evaluate Models"

4. **Analyze Results**:
   - View metrics table (focus on G-mean)
   - Compare complexity improvements
   - View visualizations
   - Download results

### Using Python Scripts

```python
from src.main_pipeline import ImbalancedLearningPipeline
from src.resampling_techniques import RFCL, URNS, OSM

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
    'RFCL': RFCL(random_state=42),
    'URNS': URNS(k='adaptive', rounds=2),
    'OSM': OSM()
}
pipeline.apply_resampling_techniques(techniques)

# Evaluate
results = pipeline.evaluate_techniques(classifier_name='random_forest')

# Get best technique
best = pipeline.get_best_technique(metric='g_mean')
```

## 🔧 Complexity Analysis

The toolkit includes comprehensive complexity analysis using **30+ measures**:

### Key Measures
- **N3**: Error Rate of 1-NN Classifier (instance overlap)
- **T1**: Fraction of Hyperspheres (structural overlap)  
- **N1**: Fraction of Borderline Points
- **F1**: Maximum Fisher's Discriminant Ratio

### Automatic Recommendations
- **High N3** → Instance-based techniques (NBUS, RFCL, URNS)
- **High T1** → Clustering-based techniques (KMeans, OSM)
- **High N1** → Borderline-focused techniques
- **Low F1** → Feature selection/transformation needed

## 📊 Supported Metrics

- **Accuracy**: Overall correctness
- **Balanced Accuracy**: Better for imbalanced data
- **Precision**: Positive prediction accuracy
- **Recall**: True positive rate
- **F1-Score**: Harmonic mean of precision and recall
- **Specificity**: True negative rate
- **G-mean**: Geometric mean (ideal for imbalanced data) ⭐
- **AUC-ROC**: Area under ROC curve

## 🎓 References

**Complexity Measures** based on:
> Ho, T. K., & Basu, M. (2002). "Complexity measures of supervised classification problems." IEEE Transactions on Pattern Analysis and Machine Intelligence.

**Individual Techniques** - See PROJECT_DETAILED_REPORT.md for complete references.

## 📝 License

MIT License

## 🤝 Contributing

Contributions are welcome! The modular architecture makes it easy to:
- Add new resampling techniques
- Improve visualizations
- Add more classifiers
- Enhance the UI

## 💡 Tips

- Start with IR > 3 for meaningful comparisons
- Use G-mean for imbalanced data evaluation
- Enable complexity analysis for technique selection guidance
- Try multiple techniques and compare results
- Use cross-validation for robust results

## 📚 Documentation

- **PROJECT_DETAILED_REPORT.md**: Comprehensive 50+ page documentation
- **COMPLEXITY_ANALYSIS_FEATURE.md**: Detailed complexity analysis guide
- **misc/QUICK_START.md**: Quick start guide
- **src/techniques/README.md**: Techniques documentation
