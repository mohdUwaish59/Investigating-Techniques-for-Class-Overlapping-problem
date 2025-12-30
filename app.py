import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from data_loader import DataLoader
from resampling_techniques import (
    EHSO, RandomOverSampler, RandomUnderSampler, RFCL, SVDDWSMOTE, NBUS, KMeansUndersampling, OSM, URNS, NUS, DeviOCSVM, FCMBoostOBU, ODBOT
)
from visualization import ImbalancedDataVisualizer
from model_evaluation import ModelEvaluator
from complexity_measures import ComplexityMeasures, compare_pre_post_overlap

# Page config
st.set_page_config(
    page_title="Imbalanced Learning Toolkit",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498db;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'resampling_done' not in st.session_state:
    st.session_state.resampling_done = False
if 'evaluation_done' not in st.session_state:
    st.session_state.evaluation_done = False

# Title
st.markdown('<div class="main-header">⚖️ Investigating Techniques for Class Overlapping problem</div>', unsafe_allow_html=True)
st.markdown("**A comparative approach**")

# Sidebar - Configuration
st.sidebar.title("🎛️ Configuration")

# ============================================================================
# SECTION 1: DATA LOADING
# ============================================================================
st.sidebar.markdown("### 📊 Data Source")
data_source = st.sidebar.radio(
    "Choose data source:",
    ["Upload CSV", "Generate Synthetic Data", "Use Sample Data"]
)

df = None
target_column = None

if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success(f"✅ Loaded {len(df)} rows")
        
        # Target column selection
        target_column = st.sidebar.selectbox(
            "Select target column:",
            options=df.columns.tolist()
        )
        
        # Feature columns selection
        all_columns = df.columns.tolist()
        all_columns.remove(target_column)
        feature_columns = st.sidebar.multiselect(
            "Select feature columns (leave empty for all):",
            options=all_columns,
            default=all_columns
        )
        
        if not feature_columns:
            feature_columns = all_columns

elif data_source == "Generate Synthetic Data":
    st.sidebar.markdown("#### Synthetic Data Parameters")
    n_samples = st.sidebar.slider("Number of samples", 100, 5000, 500, 100)
    n_features = st.sidebar.slider("Number of features", 2, 50, 10, 1)
    imbalance_ratio = st.sidebar.slider("Imbalance ratio", 1.0, 20.0, 5.0, 0.5)
    overlap_degree = st.sidebar.slider("Overlap degree", 0.0, 1.0, 0.3, 0.05)
    
    if st.sidebar.button("Generate Data"):
        loader = DataLoader(standardize=True, random_state=42)
        X, y = loader.create_synthetic_data(
            n_samples=n_samples,
            n_features=n_features,
            imbalance_ratio=imbalance_ratio,
            overlap_degree=overlap_degree
        )
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
        df['target'] = y
        target_column = 'target'
        feature_columns = [f'feature_{i}' for i in range(n_features)]
        st.sidebar.success("✅ Synthetic data generated!")

else:  # Use Sample Data
    if st.sidebar.button("Load Sample Data"):
        try:
            df = pd.read_csv('data/data_imbalanced.csv')
            target_column = 'stabf'
            feature_columns = [col for col in df.columns if col != target_column]
            st.sidebar.success("✅ Sample data loaded!")
        except FileNotFoundError:
            st.sidebar.error("Sample data not found. Please upload a CSV or generate synthetic data.")
            df = None

# ============================================================================
# SECTION 2: DATA PREPROCESSING
# ============================================================================
if df is not None and target_column is not None:
    st.session_state.data_loaded = True
    
    st.sidebar.markdown("### ⚙️ Preprocessing")
    standardize = st.sidebar.checkbox("Standardize features", value=True)
    random_state = st.sidebar.number_input("Random state", 0, 1000, 42, 1)
    
    # Load data
    try:
        loader = DataLoader(standardize=standardize, random_state=random_state)
        X, y = loader.load_from_dataframe(df, target_column, feature_columns)
        
        # Store in session state
        st.session_state.X = X
        st.session_state.y = y
        st.session_state.loader = loader
    except ValueError as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.info("💡 This toolkit currently supports binary classification only. Please ensure your target column has exactly 2 unique values.")
        st.session_state.data_loaded = False
        st.stop()
    
    # Display data info
    st.markdown('<div class="section-header">📊 Dataset Overview</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    unique, counts = np.unique(y, return_counts=True)
    ir = max(counts) / min(counts)
    
    with col1:
        st.metric("Total Samples", len(X))
    with col2:
        st.metric("Features", X.shape[1])
    with col3:
        st.metric("Majority Class", int(max(counts)))
    with col4:
        st.metric("Minority Class", int(min(counts)))
    
    col5, col6 = st.columns(2)
    with col5:
        st.metric("Imbalance Ratio", f"{ir:.2f}")
    with col6:
        st.metric("Minority %", f"{(min(counts)/len(y)*100):.1f}%")
    
    # Show data preview
    with st.expander("📋 View Data Preview"):
        st.dataframe(df.head(20))
    
    # Show class distribution
    with st.expander("📊 Class Distribution"):
        fig, ax = plt.subplots(figsize=(8, 4))
        class_names = [f"Class {c}" for c in unique]
        ax.bar(class_names, counts, color=['#3498db', '#e74c3c'])
        ax.set_ylabel('Count')
        ax.set_title('Class Distribution')
        for i, (name, count) in enumerate(zip(class_names, counts)):
            ax.text(i, count, str(count), ha='center', va='bottom')
        st.pyplot(fig)
        plt.close()

# ============================================================================
# SECTION 3: RESAMPLING TECHNIQUES
# ============================================================================
if st.session_state.data_loaded:
    st.markdown('<div class="section-header">🔄 Resampling Techniques</div>', unsafe_allow_html=True)
    
    st.sidebar.markdown("### 🔄 Resampling Methods")
    
    # Global Complexity Analysis Toggle
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Complexity Analysis")
    enable_complexity = st.sidebar.checkbox(
        "Calculate N3 & T1 measures",
        value=True,
        help="Measure class overlap before/after resampling using N3 and T1 complexity measures"
    )
    if enable_complexity:
        complexity_by_class = st.sidebar.checkbox(
            "Show per-class breakdown",
            value=True,
            help="Calculate N3 and T1 separately for each class"
        )
    st.sidebar.markdown("---")
    
    # Technique selection
    available_techniques = {
        "T1: RFCL": "Random Forest Cleaning Rule - handles class overlap",
        "T1.1: URNS": "Recursive Neighbourhood Search - overlap-based undersampling",
        "T1.3: NUS": "Neighbourhood Under-Sampling - colonial neighbours method",
        "T1.4: DeviOCSVM": "Devi et al. One-Class SVM - comprehensive overlap handling with Tomek links",
        "T1.5: FCMBoostOBU": "Fuzzy C-Means Boosted Overlap-Based Undersampling - BLSMOTE1 + FCM clustering",
        "T2: SVDDWSMOTE": "SVDD-based overlap handler - removes noisy instances",
        "T2.1: ODBOT": "Outlier Detection-Based Oversampling - clustering-based synthetic generation",
        "T3: EHSO": "Evolutionary Hybrid Sampling in Overlapping scenarios",
        "T4: NBUS": "Neighbourhood-Based Undersampling - 4 variants available",
        "T5: KMeans": "KMeans-Based Undersampling - 4 clustering variants",
        "T6: OSM": "Overlap-Separating Model - comprehensive preprocessing pipeline",
        "T7: ROS": "Random Oversampling - randomly duplicate minority samples",
        "T8: RUS": "Random Undersampling - randomly remove majority samples",
        "SMOTE": "Synthetic Minority Over-sampling Technique (Coming Soon)",
        "ADASYN": "Adaptive Synthetic Sampling (Coming Soon)",
    }
    
    selected_techniques = st.sidebar.multiselect(
        "Select techniques to apply:",
        options=list(available_techniques.keys())[:13],  # Only available ones
        default=["T1: RFCL", "T1.3: NUS"],
        help="Select one or more resampling techniques"
    )
    
    # Technique-specific parameters
    technique_params = {}
    
    if "T1: RFCL" in selected_techniques:
        with st.sidebar.expander("⚙️ RFCL Parameters"):
            st.info("RFCL uses Random Forest to identify and remove overlapping majority samples")
            rfcl_verbose = st.checkbox("Show detailed progress", value=False, key="rfcl_verbose")
            random_state = 42
            technique_params["T1: RFCL"] = {
                'random_state': random_state,
                'verbose': rfcl_verbose
            }
    
    if "T1.1: URNS" in selected_techniques:
        with st.sidebar.expander("⚙️ URNS Parameters"):
            st.info("URNS: Recursive neighbourhood search to remove overlapping majority samples")
            
            # K parameter
            urns_k_mode = st.radio("k selection mode", ["Adaptive", "Manual"], key="urns_k_mode",
                                  help="Adaptive: k = sqrt(N) + sqrt(IR)")
            if urns_k_mode == "Manual":
                urns_k = st.slider("k (neighbors)", 3, 50, 10, 1, key="urns_k")
            else:
                urns_k = 'adaptive'
            
            # Min frequency
            urns_min_freq = st.slider("Minimum frequency", 1, 5, 2, 1, key="urns_min_freq",
                                     help="Min times an instance must appear to be removed")
            
            # Rounds
            urns_rounds = st.slider("Recursive rounds", 1, 3, 2, 1, key="urns_rounds",
                                   help="Number of recursive neighbourhood search rounds")
            
            urns_verbose = st.checkbox("Show detailed progress", value=False, key="urns_verbose")
            
            technique_params["T1.1: URNS"] = {
                'k': urns_k,
                'min_frequency': urns_min_freq,
                'rounds': urns_rounds,
                'random_state': random_state,
                'verbose': urns_verbose
            }
    
    if "T1.3: NUS" in selected_techniques:
        with st.sidebar.expander("⚙️ NUS Parameters"):
            st.info("NUS: Colonial neighbours method for overlap-based undersampling")
            
            # K parameter
            nus_k_mode = st.radio("k selection mode", ["Auto", "Manual"], key="nus_k_mode",
                                 help="Auto: k = min(sqrt(n_majority), 50)")
            if nus_k_mode == "Manual":
                nus_k = st.slider("k (neighbors)", 3, 50, 10, 1, key="nus_k")
            else:
                nus_k = None
            
            # Distance threshold
            nus_dist_mode = st.radio("Distance threshold", ["Median", "Mean", "Custom"], key="nus_dist_mode",
                                    help="Threshold for nominating neighbours")
            if nus_dist_mode == "Custom":
                nus_dist = st.number_input("Custom distance", 0.01, 10.0, 1.0, 0.1, key="nus_dist")
            else:
                nus_dist = nus_dist_mode.lower()
            
            # Min membership
            nus_min_mem = st.slider("Minimum membership", 1, 5, 2, 1, key="nus_min_mem",
                                   help="Min memberships required for elimination")
            
            nus_verbose = st.checkbox("Show detailed progress", value=False, key="nus_verbose")
            
            technique_params["T1.3: NUS"] = {
                'k_neighbors': nus_k,
                'distance_threshold': nus_dist,
                'min_membership': nus_min_mem,
                'random_state': random_state,
                'verbose': nus_verbose
            }
    
    if "T1.4: DeviOCSVM" in selected_techniques:
        with st.sidebar.expander("⚙️ DeviOCSVM Parameters"):
            st.info("DeviOCSVM: Comprehensive One-Class SVM method with Tomek links (Devi et al. 2019)")
            devi_nu = st.slider("nu", 0.1, 0.9, 0.5, 0.1, key="devi_nu",
                               help="Parameter for one-class SVM (ϑ in paper, tested with 0.3, 0.5, 0.7)")
            devi_k1 = st.slider("K1", 1, 5, 1, 1, key="devi_k1",
                               help="K-NN of minority instance (set to 1 in paper)")
            devi_k2 = st.slider("K2", 3, 10, 5, 1, key="devi_k2",
                               help="K-NN of majority instances (set to 5 in paper)")
            devi_k3 = st.slider("K3", 1, 5, 1, 1, key="devi_k3",
                               help="K-NN of overlapped majority instance (set to 1 in paper)")
            devi_kernel = st.selectbox("kernel", ["rbf", "linear", "poly", "sigmoid"], 
                                      index=0, key="devi_kernel")
            devi_gamma = st.selectbox("gamma", ["scale", "auto"], index=0, key="devi_gamma")
            devi_verbose = st.checkbox("Show detailed progress", value=False, key="devi_verbose")
            
            technique_params["T1.4: DeviOCSVM"] = {
                'nu': devi_nu,
                'K1': devi_k1,
                'K2': devi_k2,
                'K3': devi_k3,
                'kernel': devi_kernel,
                'gamma': devi_gamma,
                'verbose': devi_verbose
            }
    
    if "T1.5: FCMBoostOBU" in selected_techniques:
        with st.sidebar.expander("⚙️ FCMBoostOBU Parameters"):
            st.info("FCMBoostOBU: Fuzzy C-Means Boosted Overlap-Based Undersampling (Vuttipittayamongkol & Elyan 2020)")
            fcm_k = st.slider("k", 3, 15, 5, 1, key="fcm_k",
                             help="Number of neighbors for BLSMOTE1")
            fcm_m = st.slider("m", 1.1, 5.0, 2.0, 0.1, key="fcm_m",
                             help="Fuzziness parameter for FCM (1 ≤ m ≤ ∞)")
            fcm_max_iter = st.slider("max_iter", 100, 2000, 1000, 100, key="fcm_max_iter",
                                    help="Maximum iterations for FCM")
            fcm_error = st.selectbox("error", [1e-3, 1e-4, 1e-5, 1e-6], index=2, key="fcm_error",
                                    help="Convergence criterion for FCM")
            fcm_random_state = st.number_input("random_state", value=42, key="fcm_random_state")
            fcm_verbose = st.checkbox("Show detailed progress", value=False, key="fcm_verbose")
            
            technique_params["T1.5: FCMBoostOBU"] = {
                'k': fcm_k,
                'm': fcm_m,
                'max_iter': fcm_max_iter,
                'error': fcm_error,
                'random_state': int(fcm_random_state),
                'verbose': fcm_verbose
            }
    
    if "T2: SVDDWSMOTE" in selected_techniques:
        with st.sidebar.expander("⚙️ SVDDWSMOTE Parameters"):
            st.info("SVDDWSMOTE uses SVDD to identify and remove overlapped/noisy instances")
            svdd_rho = st.slider("rho_threshold", 0.01, 0.1, 0.045, 0.005, key="svdd_rho")
            svdd_delta = st.slider("delta_threshold", 0.1, 0.5, 0.25, 0.05, key="svdd_delta")
            svdd_n_c1 = st.slider("n_C1_candidates", 3, 13, 5, 1, key="svdd_n_c1", 
                                 help="Number of C1 values to try (paper uses 13)")
            svdd_n_sigma = st.slider("n_sigma_candidates", 3, 16, 5, 1, key="svdd_n_sigma",
                                    help="Number of sigma values to try (paper uses 16)")
            svdd_verbose = st.checkbox("Show detailed progress", value=False, key="svdd_verbose")
            technique_params["T2: SVDDWSMOTE"] = {
                'rho_threshold': svdd_rho,
                'delta_threshold': svdd_delta,
                'n_C1_candidates': svdd_n_c1,
                'n_sigma_candidates': svdd_n_sigma,
                'random_state': random_state,
                'verbose': svdd_verbose
            }
    
    if "T4: NBUS" in selected_techniques:
        with st.sidebar.expander("⚙️ NBUS Parameters"):
            st.info("NBUS: Neighbourhood-Based Undersampling with 4 variants")
            
            # Sub-technique selection
            nbus_methods = st.multiselect(
                "Select NBUS variant(s):",
                options=['NB-Basic', 'NB-Tomek', 'NB-Comm', 'NB-Rec'],
                default=['NB-Basic'],
                help="Select one or more NBUS variants to apply and compare",
                key="nbus_methods"
            )
            
            # K parameter
            nbus_k_auto = st.checkbox("Auto-calculate k (sqrt(N) + imb)", value=True, key="nbus_k_auto")
            if not nbus_k_auto:
                nbus_k = st.slider("k (neighbors)", 1, 50, 5, 1, key="nbus_k")
            else:
                nbus_k = None
            
            nbus_verbose = st.checkbox("Show detailed progress", value=False, key="nbus_verbose")
            
            random_state = 42
            technique_params["T4: NBUS"] = {
                'methods': nbus_methods,
                'k': nbus_k,
                'random_state': random_state,
                'verbose': nbus_verbose
            }
    
    if "T5: KMeans" in selected_techniques:
        with st.sidebar.expander("⚙️ KMeans Parameters"):
            st.info("KMeans: Clustering-based undersampling with 4 variants")
            
            # Sub-technique selection
            kmeans_methods = st.multiselect(
                "Select KMeans variant(s):",
                options=['HKM', 'FCM', 'RKM', 'FRKM'],
                default=['HKM'],
                help="HKM=Hard, FCM=Fuzzy, RKM=Rough, FRKM=Fuzzy-Rough",
                key="kmeans_methods"
            )
            
            # Common parameters
            kmeans_k = st.slider("k (clusters)", 2, 10, 3, 1, key="kmeans_k",
                                help="Number of clusters for majority class")
            kmeans_max_iter = st.slider("max_iter", 50, 200, 100, 10, key="kmeans_max_iter")
            kmeans_epsilon = st.number_input("epsilon", 0.00001, 0.001, 0.00001, 
                                            format="%.5f", key="kmeans_epsilon")
            
            # FCM and FRKM specific
            if 'FCM' in kmeans_methods or 'FRKM' in kmeans_methods:
                kmeans_m = st.slider("m (fuzzifier)", 1.1, 5.0, 2.0, 0.1, key="kmeans_m",
                                    help="Fuzzifier for FCM/FRKM")
            else:
                kmeans_m = 2.0
            
            # RKM and FRKM specific
            if 'RKM' in kmeans_methods or 'FRKM' in kmeans_methods:
                kmeans_w = st.slider("w (weight)", 0.5, 1.0, 0.95, 0.05, key="kmeans_w",
                                    help="Weight for lower approximation")
                kmeans_sigma = st.slider("sigma_threshold", 0.05, 0.5, 0.1, 0.05, key="kmeans_sigma",
                                        help="Threshold for boundary region")
            else:
                kmeans_w = 0.95
                kmeans_sigma = 0.1
            
            kmeans_verbose = st.checkbox("Show detailed progress", value=False, key="kmeans_verbose")
            
            technique_params["T5: KMeans"] = {
                'methods': kmeans_methods,
                'k': kmeans_k,
                'm': kmeans_m,
                'w': kmeans_w,
                'sigma_threshold': kmeans_sigma,
                'max_iter': kmeans_max_iter,
                'epsilon': kmeans_epsilon,
                'random_state': random_state,
                'verbose': kmeans_verbose
            }
    
    if "T6: OSM" in selected_techniques:
        with st.sidebar.expander("⚙️ OSM Parameters"):
            st.info("OSM: Comprehensive overlap-separating preprocessing pipeline")
            
            # Clustering parameters
            osm_n_clusters = st.slider("Number of clusters (K-means)", 2, 10, 2, 1, key="osm_n_clusters",
                                      help="Number of clusters for overlap separation")
            
            # Feature selection
            osm_n_features = st.slider("Number of features to select", 2, 20, 6, 1, key="osm_n_features",
                                      help="Features to keep after RF selection (None = auto)")
            osm_n_features = None if osm_n_features == 6 else osm_n_features
            
            # Overlap threshold
            osm_overlap_threshold = st.slider("Overlap threshold", 0.1, 0.9, 0.3, 0.05, key="osm_overlap_threshold",
                                             help="Lower = more samples in overlap region")
            
            # Pipeline toggles
            st.markdown("**Pipeline Steps:**")
            col1, col2 = st.columns(2)
            with col1:
                osm_rose = st.checkbox("ROSE balancing", value=True, key="osm_rose",
                                      help="SMOTE + Random Undersampling")
                osm_tomek = st.checkbox("Tomek link removal", value=True, key="osm_tomek")
                osm_feature_sel = st.checkbox("Feature selection", value=True, key="osm_feature_sel")
            with col2:
                osm_outlier = st.checkbox("Outlier removal", value=True, key="osm_outlier",
                                         help="Boxplot IQR method")
                osm_svm = st.checkbox("SVM optimization", value=True, key="osm_svm",
                                     help="Remove misclassified samples")
                osm_verbose = st.checkbox("Show detailed progress", value=False, key="osm_verbose")
            
            technique_params["T6: OSM"] = {
                'n_clusters': osm_n_clusters,
                'n_features': osm_n_features,
                'overlap_threshold': osm_overlap_threshold,
                'rose_sampling': osm_rose,
                'tomek_removal': osm_tomek,
                'feature_selection': osm_feature_sel,
                'outlier_removal': osm_outlier,
                'svm_optimization': osm_svm,
                'random_state': random_state,
                'verbose': osm_verbose
            }
    
    if "T3: EHSO" in selected_techniques:
        with st.sidebar.expander("⚙️ EHSO Parameters"):
            ehso_k = st.slider("k_neighbors", 3, 15, 5, 1, key="ehso_k")
            ehso_alpha = st.slider("alpha", 0.0, 1.0, 0.1, 0.05, key="ehso_alpha")
            ehso_pop = st.slider("population_size", 5, 20, 10, 1, key="ehso_pop")
            ehso_iter = st.slider("max_iterations", 10, 50, 30, 5, key="ehso_iter")
            technique_params["T3: EHSO"] = {
                'k_neighbors': ehso_k,
                'alpha': ehso_alpha,
                'population_size': ehso_pop,
                'max_iterations': ehso_iter,
                'verbose': False,
                'random_state': random_state
            }
    
    if "T2.1: ODBOT" in selected_techniques:
        with st.sidebar.expander("⚙️ ODBOT Parameters"):
            st.info("ODBOT: Outlier Detection-Based Oversampling Technique (Ibrahim 2021)")
            odbot_k = st.slider("k", 2, 10, 2, 1, key="odbot_k",
                               help="Number of clusters (must be > 1 as per paper)")
            odbot_percentage = st.selectbox("percentage", [None, 100, 200, 300, 400, 500], 
                                           index=0, key="odbot_percentage",
                                           help="Percentage for oversampling. None = automatic calculation")
            odbot_random_state = st.number_input("random_state", value=42, key="odbot_random_state")
            odbot_verbose = st.checkbox("Show detailed progress", value=False, key="odbot_verbose")
            
            technique_params["T2.1: ODBOT"] = {
                'k': odbot_k,
                'percentage': odbot_percentage,
                'random_state': int(odbot_random_state),
                'verbose': odbot_verbose
            }
    
    # Apply resampling button
    if st.sidebar.button("🚀 Apply Resampling", type="primary"):
        with st.spinner("Applying resampling techniques..."):
            resampled_data = {}
            
            for technique in selected_techniques:
                if technique == "T1: RFCL":
                    sampler = RFCL(**technique_params["T1: RFCL"])
                    X_res, y_res = sampler.fit_resample(st.session_state.X, st.session_state.y)
                    resampled_data[technique] = (X_res, y_res)
                
                elif technique == "T1.1: URNS":
                    sampler = URNS(**technique_params["T1.1: URNS"])
                    X_res, y_res = sampler.fit_resample(st.session_state.X, st.session_state.y)
                    resampled_data[technique] = (X_res, y_res)
                
                elif technique == "T1.3: NUS":
                    sampler = NUS(**technique_params["T1.3: NUS"])
                    X_res, y_res = sampler.fit_resample(st.session_state.X, st.session_state.y)
                    resampled_data[technique] = (X_res, y_res)
                    
                elif technique == "T1.4: DeviOCSVM":
                    sampler = DeviOCSVM(**technique_params["T1.4: DeviOCSVM"])
                    X_res, y_res = sampler.fit_resample(st.session_state.X, st.session_state.y)
                    resampled_data[technique] = (X_res, y_res)
                    
                elif technique == "T1.5: FCMBoostOBU":
                    sampler = FCMBoostOBU(**technique_params["T1.5: FCMBoostOBU"])
                    X_res, y_res = sampler.fit_resample(st.session_state.X, st.session_state.y)
                    resampled_data[technique] = (X_res, y_res)
                    
                elif technique == "T2: SVDDWSMOTE":
                    sampler = SVDDWSMOTE(**technique_params["T2: SVDDWSMOTE"])
                    X_res, y_res = sampler.fit_resample(st.session_state.X, st.session_state.y)
                    resampled_data[technique] = (X_res, y_res)
                    
                elif technique == "T4: NBUS":
                    # Apply each selected NBUS variant
                    params = technique_params["T4: NBUS"]
                    for method in params['methods']:
                        sampler = NBUS(
                            method=method,
                            k=params['k'],
                            random_state=params['random_state'],
                            verbose=params['verbose']
                        )
                        X_res, y_res = sampler.fit_resample(st.session_state.X, st.session_state.y)
                        resampled_data[f"T4: NBUS-{method}"] = (X_res, y_res)
                
                elif technique == "T5: KMeans":
                    # Apply each selected KMeans variant
                    params = technique_params["T5: KMeans"]
                    for method in params['methods']:
                        sampler = KMeansUndersampling(
                            method=method,
                            k=params['k'],
                            m=params['m'],
                            w=params['w'],
                            sigma_threshold=params['sigma_threshold'],
                            max_iter=params['max_iter'],
                            epsilon=params['epsilon'],
                            random_state=params['random_state'],
                            verbose=params['verbose']
                        )
                        X_res, y_res = sampler.fit_resample(st.session_state.X, st.session_state.y)
                        resampled_data[f"T5: KMeans-{method}"] = (X_res, y_res)
                
                elif technique == "T6: OSM":
                    sampler = OSM(**technique_params["T6: OSM"])
                    X_res, y_res = sampler.fit_resample(st.session_state.X, st.session_state.y)
                    resampled_data[technique] = (X_res, y_res)
                    
                elif technique == "T3: EHSO":
                    sampler = EHSO(**technique_params["T3: EHSO"])
                    X_res, y_res = sampler.fit_resample(st.session_state.X, st.session_state.y)
                    resampled_data[technique] = (X_res, y_res)
                    
                elif technique == "T2.1: ODBOT":
                    sampler = ODBOT(**technique_params["T2.1: ODBOT"])
                    X_res, y_res = sampler.fit_resample(st.session_state.X, st.session_state.y)
                    resampled_data[technique] = (X_res, y_res)
                    
                elif technique == "T7: ROS":
                    sampler = RandomOverSampler(random_state=random_state)
                    X_res, y_res = sampler.fit_resample(st.session_state.X, st.session_state.y)
                    resampled_data[technique] = (X_res, y_res)
                    
                elif technique == "T8: RUS":
                    sampler = RandomUnderSampler(random_state=random_state)
                    X_res, y_res = sampler.fit_resample(st.session_state.X, st.session_state.y)
                    resampled_data[technique] = (X_res, y_res)
                    
                else:
                    continue
            
            st.session_state.resampled_data = resampled_data
            st.session_state.resampling_done = True
            
            # Calculate complexity measures if enabled
            if enable_complexity:
                with st.spinner("Calculating complexity measures using local complexity.py..."):
                    complexity_results = {}
                    
                    # Calculate for original data
                    try:
                        cm_original = ComplexityMeasures(st.session_state.X, st.session_state.y)
                        complexity_results['Original'] = cm_original.analyze_overlap(include_all=False)
                    except Exception as e:
                        st.warning(f"Could not calculate complexity for original data: {e}")
                        complexity_results['Original'] = {'n3': {'overall': 0.0}, 't1': {'normalized': 0.0}}
                    
                    # Calculate for each resampled dataset
                    for name, (X_res, y_res) in resampled_data.items():
                        try:
                            comparison = compare_pre_post_overlap(
                                st.session_state.X, st.session_state.y,
                                X_res, y_res,
                                include_all=False
                            )
                            complexity_results[name] = comparison
                        except Exception as e:
                            st.warning(f"Could not calculate complexity for {name}: {e}")
                            complexity_results[name] = {
                                'post_processing': {'n3': {'overall': 0.0}, 't1': {'normalized': 0.0}},
                                'improvements': {'n3': {'absolute': 0.0}, 't1': {'absolute': 0.0}}
                            }
                    
                    st.session_state.complexity_results = complexity_results
                    st.session_state.complexity_enabled = True
            else:
                st.session_state.complexity_enabled = False
            
            st.success(f"✅ Applied {len(selected_techniques)} resampling technique(s)!")
    
    # Display resampling results
    if st.session_state.resampling_done:
        st.markdown("#### 📈 Resampling Results")
        
        # Create comparison table
        comparison_data = []
        comparison_data.append({
            'Technique': 'Original',
            'Total Samples': len(st.session_state.y),
            'Majority': int(np.sum(st.session_state.y == 0)),
            'Minority': int(np.sum(st.session_state.y == 1)),
            'IR': f"{np.sum(st.session_state.y == 0) / np.sum(st.session_state.y == 1):.2f}"
        })
        
        for name, (X_res, y_res) in st.session_state.resampled_data.items():
            unique, counts = np.unique(y_res, return_counts=True)
            comparison_data.append({
                'Technique': name,
                'Total Samples': len(y_res),
                'Majority': int(max(counts)),
                'Minority': int(min(counts)),
                'IR': f"{max(counts) / min(counts):.2f}"
            })
        
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
        
        # Visualization
        with st.expander("📊 Visualize Resampling Results"):
            viz = ImbalancedDataVisualizer()
            
            for name, (X_res, y_res) in st.session_state.resampled_data.items():
                st.markdown(f"**{name}**")
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                
                # Original data
                X_vis_orig = viz.reduce_dimensions(st.session_state.X, method='pca')
                for cls in np.unique(st.session_state.y):
                    mask = st.session_state.y == cls
                    color = '#3498db' if cls == 0 else '#e74c3c'
                    ax1.scatter(X_vis_orig[mask, 0], X_vis_orig[mask, 1],
                              c=color, label=f'Class {cls}', alpha=0.6, s=30)
                ax1.set_title('Original Data')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Resampled data
                X_vis_res = viz.reduce_dimensions(X_res, method='pca')
                for cls in np.unique(y_res):
                    mask = y_res == cls
                    color = '#3498db' if cls == 0 else '#e74c3c'
                    ax2.scatter(X_vis_res[mask, 0], X_vis_res[mask, 1],
                              c=color, label=f'Class {cls}', alpha=0.6, s=30)
                ax2.set_title(f'After {name}')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                plt.close()
        
        # Complexity Analysis Results
        if st.session_state.get('complexity_enabled', False):
            st.markdown("#### 📊 Complexity Analysis (N3 & T1 Measures)")
            
            with st.expander("ℹ️ About N3 and T1 Measures", expanded=False):
                st.markdown("""
                **Complexity Measures using local complexity.py implementation**
                
                **N3 (Error Rate of 1-NN Classifier)**
                - Measures instance-level overlap (local characteristics)
                - Uses Leave-One-Out cross-validation with 1-NN
                - Lower values = less overlap
                - Range: 0 (no overlap) to 1 (complete overlap)
                
                **T1 (Fraction of Hyperspheres)**
                - Measures structural overlap (global characteristics)
                - Counts hyperspheres needed to cover data
                - Lower values = less overlap
                - Normalized by number of samples
                
                **Additional Measures Available:**
                - N1: Fraction of Borderline Points
                - F1: Maximum Fisher's Discriminant Ratio
                - N2: Ratio of Intra/Extra Class NN Distance
                - SI: Separability Index
                
                **Interpretation:**
                - High N3: Instance overlap - consider NBUS, RFCL, OSM
                - High T1: Structural overlap - consider KMeans, OSM
                - High N1: Many borderline points - consider borderline techniques
                """)
            
            # Create complexity comparison table
            complexity_comparison = []
            
            # Original data
            orig_results = st.session_state.complexity_results['Original']
            n3_orig = orig_results.get('n3', {}).get('overall', 0.0)
            t1_orig = orig_results.get('t1', {}).get('normalized', 0.0)
            complexity_level = orig_results.get('interpretation', {}).get('overall_complexity', 'unknown').upper()
            
            complexity_comparison.append({
                'Technique': 'Original',
                'N3 (Instance Overlap)': f"{n3_orig:.4f}",
                'T1 (Structural Overlap)': f"{t1_orig:.4f}",
                'Complexity Level': complexity_level
            })
            
            # Resampled data
            for name in st.session_state.resampled_data.keys():
                if name in st.session_state.complexity_results:
                    comp_results = st.session_state.complexity_results[name]
                    
                    # Handle both direct results and comparison results
                    if 'post_processing' in comp_results:
                        post_results = comp_results['post_processing']
                        n3_post = post_results.get('n3', {}).get('overall', 0.0)
                        t1_post = post_results.get('t1', {}).get('normalized', 0.0)
                        n3_improvement = comp_results.get('improvements', {}).get('n3', {}).get('absolute', 0.0)
                        t1_improvement = comp_results.get('improvements', {}).get('t1', {}).get('absolute', 0.0)
                        complexity_level = post_results.get('interpretation', {}).get('overall_complexity', 'unknown').upper()
                        
                        complexity_comparison.append({
                            'Technique': name,
                            'N3 (Instance Overlap)': f"{n3_post:.4f} ({n3_improvement:+.4f})",
                            'T1 (Structural Overlap)': f"{t1_post:.4f} ({t1_improvement:+.4f})",
                            'Complexity Level': complexity_level
                        })
                    else:
                        # Direct results
                        n3_direct = comp_results.get('n3', {}).get('overall', 0.0)
                        t1_direct = comp_results.get('t1', {}).get('normalized', 0.0)
                        complexity_level = comp_results.get('interpretation', {}).get('overall_complexity', 'unknown').upper()
                        
                        complexity_comparison.append({
                            'Technique': name,
                            'N3 (Instance Overlap)': f"{n3_direct:.4f}",
                            'T1 (Structural Overlap)': f"{t1_direct:.4f}",
                            'Complexity Level': complexity_level
                        })
            
            st.dataframe(pd.DataFrame(complexity_comparison), use_container_width=True)
            
            # Visualize complexity improvements
            with st.expander("📈 Visualize Complexity Improvements"):
                # Prepare data for plotting
                techniques = ['Original'] + list(st.session_state.resampled_data.keys())
                n3_values = [n3_orig]
                t1_values = [t1_orig]
                
                for name in st.session_state.resampled_data.keys():
                    if name in st.session_state.complexity_results:
                        comp_results = st.session_state.complexity_results[name]
                        if 'post_processing' in comp_results:
                            n3_values.append(comp_results['post_processing'].get('n3', {}).get('overall', 0.0))
                            t1_values.append(comp_results['post_processing'].get('t1', {}).get('normalized', 0.0))
                        else:
                            n3_values.append(comp_results.get('n3', {}).get('overall', 0.0))
                            t1_values.append(comp_results.get('t1', {}).get('normalized', 0.0))
                
                # Create comparison plots
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                
                # N3 comparison
                colors = ['#e74c3c'] + ['#2ecc71'] * (len(techniques) - 1)
                bars1 = ax1.barh(techniques, n3_values, color=colors, alpha=0.7)
                ax1.set_xlabel('N3 Score (Lower is Better)', fontsize=12)
                ax1.set_title('N3: Instance Overlap Comparison', fontsize=14, fontweight='bold')
                ax1.axvline(x=0.2, color='orange', linestyle='--', alpha=0.5, label='High Overlap Threshold')
                ax1.legend()
                ax1.grid(True, alpha=0.3, axis='x')
                
                # Add value labels
                for i, (bar, val) in enumerate(zip(bars1, n3_values)):
                    ax1.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                            f'{val:.4f}', va='center', fontsize=9)
                
                # T1 comparison
                bars2 = ax2.barh(techniques, t1_values, color=colors, alpha=0.7)
                ax2.set_xlabel('T1 Score (Lower is Better)', fontsize=12)
                ax2.set_title('T1: Structural Overlap Comparison', fontsize=14, fontweight='bold')
                ax2.axvline(x=0.3, color='orange', linestyle='--', alpha=0.5, label='High Overlap Threshold')
                ax2.legend()
                ax2.grid(True, alpha=0.3, axis='x')
                
                # Add value labels
                for i, (bar, val) in enumerate(zip(bars2, t1_values)):
                    ax2.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                            f'{val:.4f}', va='center', fontsize=9)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # Per-class breakdown if enabled
                if complexity_by_class:
                    st.markdown("**Additional Complexity Measures**")
                    
                    # Show additional measures if available
                    additional_measures = []
                    for name in ['Original'] + list(st.session_state.resampled_data.keys()):
                        if name == 'Original':
                            results = orig_results
                        elif name in st.session_state.complexity_results:
                            comp_results = st.session_state.complexity_results[name]
                            results = comp_results.get('post_processing', comp_results)
                        else:
                            continue
                        
                        measure_data = {'Technique': name}
                        
                        # Add available measures
                        for measure in ['n1', 'f1', 'n2', 'si']:
                            if measure in results:
                                val = results[measure]
                                # Handle array results by taking the mean
                                if isinstance(val, (list, np.ndarray)):
                                    val = np.mean(val) if len(val) > 0 else 0.0
                                measure_data[measure.upper()] = f"{val:.4f}"
                        
                        if len(measure_data) > 1:  # More than just technique name
                            additional_measures.append(measure_data)
                    
                    if additional_measures:
                        st.dataframe(pd.DataFrame(additional_measures), use_container_width=True)
                    else:
                        st.info("Additional measures not available. Check complexity.py file.")

# ============================================================================
# SECTION 4: MODEL EVALUATION
# ============================================================================
if st.session_state.resampling_done:
    st.markdown('<div class="section-header">🎯 Model Evaluation</div>', unsafe_allow_html=True)
    
    st.sidebar.markdown("### 🎯 Evaluation Settings")
    
    # Classifier selection
    classifier_name = st.sidebar.selectbox(
        "Select classifier:",
        ["decision_tree", "random_forest", "logistic_regression", "knn", "naive_bayes", "svm"]
    )
    
    # Classifier parameters
    classifier_params = {}
    with st.sidebar.expander("⚙️ Classifier Parameters"):
        if classifier_name == "decision_tree":
            max_depth = st.slider("max_depth", 1, 20, 5, 1)
            classifier_params['max_depth'] = max_depth
        elif classifier_name == "random_forest":
            n_estimators = st.slider("n_estimators", 10, 200, 100, 10)
            max_depth = st.slider("max_depth", 1, 20, 5, 1)
            classifier_params['n_estimators'] = n_estimators
            classifier_params['max_depth'] = max_depth
        elif classifier_name == "knn":
            n_neighbors = st.slider("n_neighbors", 1, 20, 5, 1)
            classifier_params['n_neighbors'] = n_neighbors
    
    # Evaluation settings
    test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.3, 0.05)
    include_baseline = st.sidebar.checkbox("Include baseline (no resampling)", value=True)
    
    # Evaluate button
    if st.sidebar.button("📊 Evaluate Models", type="primary"):
        with st.spinner("Evaluating models..."):
            evaluator = ModelEvaluator(
                test_size=test_size,
                random_state=random_state,
                verbose=False
            )
            
            results = evaluator.compare_techniques(
                st.session_state.X,
                st.session_state.y,
                st.session_state.resampled_data,
                classifier_name=classifier_name,
                include_baseline=include_baseline,
                **classifier_params
            )
            
            st.session_state.evaluation_results = results
            st.session_state.evaluator = evaluator
            st.session_state.evaluation_done = True
            st.success("✅ Evaluation complete!")
    
    # Display evaluation results
    if st.session_state.evaluation_done:
        st.markdown("#### 📊 Performance Metrics")
        
        # Display results table
        results_df = st.session_state.evaluation_results
        
        # Color code the best values
        st.dataframe(
            results_df.style.highlight_max(axis=0, color='lightgreen'),
            use_container_width=True
        )
        
        # Best technique
        best_technique = results_df['g_mean'].idxmax()
        st.success(f"🏆 **Best Technique (by G-mean):** {best_technique} ({results_df.loc[best_technique, 'g_mean']:.4f})")
        
        # Metrics visualization
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 Metrics Comparison**")
            metrics_to_plot = ['precision', 'recall', 'f1_score', 'g_mean']
            fig, ax = plt.subplots(figsize=(10, 6))
            results_df[metrics_to_plot].plot(kind='bar', ax=ax, rot=45)
            ax.set_ylabel('Score')
            ax.set_title('Performance Metrics Comparison')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.markdown("**🎯 Confusion Matrices**")
            selected_technique = st.selectbox(
                "Select technique to view confusion matrix:",
                options=results_df.index.tolist()
            )
            
            if selected_technique in st.session_state.evaluator.predictions_:
                from sklearn.metrics import confusion_matrix
                import seaborn as sns
                
                y_true = st.session_state.evaluator.predictions_[selected_technique]['y_true']
                y_pred = st.session_state.evaluator.predictions_[selected_technique]['y_pred']
                
                cm = confusion_matrix(y_true, y_pred)
                
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_title(f'{selected_technique}\nG-mean: {results_df.loc[selected_technique, "g_mean"]:.3f}')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                st.pyplot(fig)
                plt.close()
        
        # Detailed metrics
        with st.expander("📋 Detailed Metrics Explanation"):
            st.markdown("""
            - **Accuracy**: Overall correctness of predictions
            - **Balanced Accuracy**: Average of recall for each class (better for imbalanced data)
            - **Precision**: Proportion of positive predictions that are correct
            - **Recall (Sensitivity)**: Proportion of actual positives correctly identified
            - **F1-Score**: Harmonic mean of precision and recall
            - **Specificity**: Proportion of actual negatives correctly identified
            - **G-mean**: Geometric mean of sensitivity and specificity (ideal for imbalanced data)
            - **AUC-ROC**: Area under the ROC curve
            """)
        
        # Download results
        st.markdown("#### 💾 Export Results")
        
        csv = results_df.to_csv()
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name="evaluation_results.csv",
            mime="text/csv"
        )

# ============================================================================
# FOOTER
# ============================================================================
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 About")
st.sidebar.info("""
**Imbalanced Learning Toolkit**

A comprehensive tool for experimenting with imbalanced datasets.

Features:
- Multiple data sources
- Various resampling techniques
- Multiple classifiers
- Comprehensive evaluation
- Interactive visualizations
""")

# Instructions if no data loaded
if not st.session_state.data_loaded:
    st.info("👈 **Get started:** Select a data source from the sidebar to begin your experiment!")
