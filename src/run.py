import pandas as pd
from main_pipeline import ImbalancedLearningPipeline
from resampling_techniques import EHSO, RandomOverSampler, RandomUnderSampler, RFCL, SVDDWSMOTE, NBUS, OSM, URNS, NUS, DeviOCSVM, FCMBoostOBU, ODBOT

# Create or load your DataFrame
df = pd.read_csv('data/data_imbalanced.csv')

# Initialize pipeline
pipeline = ImbalancedLearningPipeline(verbose=True)

# Load data
pipeline.load_data(
    data_source='dataframe',
    dataframe=df,
    target_column='stabf'
)

# Visualize original data with overlap detection
print("\nVisualizing original data...")
pipeline.visualize_original_data(show_overlap=True, k_neighbors=5)

# Apply EHSO and other techniques
techniques = {
    'T1: RFCL': RFCL(random_state=42, verbose=True),
    'T1.1: URNS': URNS(k='adaptive', rounds=2, verbose=True),
    'T1.3: NUS': NUS(k_neighbors=None, distance_threshold='median', verbose=True),
    'T1.4: DeviOCSVM': DeviOCSVM(nu=0.5, K1=1, K2=5, K3=1, kernel='rbf', gamma='scale', verbose=True),
    'T1.5: FCMBoostOBU': FCMBoostOBU(k=5, m=2, max_iter=1000, error=1e-5, random_state=42, verbose=True),
    'T2: SVDDWSMOTE': SVDDWSMOTE(
        rho_threshold=0.045, 
        delta_threshold=0.25,
        n_C1_candidates=5,
        n_sigma_candidates=5,
        verbose=True
    ),
    'T2.1: ODBOT': ODBOT(k=2, percentage=None, random_state=42, verbose=True),
    'T3: EHSO': EHSO(k_neighbors=5, alpha=0.1, verbose=False),
    'T4: NBUS-Basic': NBUS(method='NB-Basic', k=None, random_state=42, verbose=True),
    'T6: OSM': OSM(n_clusters=2, verbose=True),
    'T7: ROS': RandomOverSampler(random_state=42),
    'T8: RUS': RandomUnderSampler(random_state=42)
}
pipeline.apply_resampling_techniques(techniques)

# Visualize resampling results
print("\nVisualizing resampling results...")
pipeline.visualize_resampling_results()

# Evaluate all techniques
results = pipeline.evaluate_techniques(classifier_name='decision_tree')

# Visualize evaluation results
print("\nVisualizing evaluation results...")
pipeline.visualize_evaluation_results()

# Get best technique
best = pipeline.get_best_technique(metric='g_mean')
print(f"\n{'='*70}")
print(f"Best Technique: {best}")
print(f"{'='*70}")