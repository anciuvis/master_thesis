import pandas as pd
import numpy as np
import glob
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)

# =============================================================================
# 1. DATA LOADING
# =============================================================================

def load_taxi_data(file_path, sample_size):
    """
    Load larger NYC taxi sample for robust pattern detection.
    """
    print("\n" + "="*80)
    print("STEP 1: LOADING DATA")
    print("="*80)
    print(f"Loading NYC Yellow Taxi data from: {file_path}")

    # If file_path is provided, use it. Otherwise use default logic.
    if file_path and os.path.exists(file_path):
        full_path = file_path
    else:
        input_path = 'C:/Users/Anya/master_thesis/output'
        full_path = os.path.join(input_path, 'taxi_data_cleaned_full.parquet')

    # Load parquet file
    print(f"Reading parquet file...")
    data = pd.read_parquet(full_path)
    print(f"[+] Total records in dataset: {len(data):,}")
    print(f"[+] Columns in dataset: {list(data.columns)}")
    print(f"[+] Data shape: {data.shape}")
    print(f"[+] Memory usage: {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    # sample the dataset
    print(f"\nSampling {sample_size:,} records for optimization...")
    df_sample = data.sample(n=min(sample_size, len(data)), random_state=42)
    print(f"[+] Sample size: {len(df_sample):,} records")
    
    # Check data quality
    print(f"\nData Quality Check:")
    print(f"  - Missing values: {df_sample.isnull().sum().sum()}")
    print(f"  - Latitude range: [{df_sample['pickup_latitude'].min():.4f}, {df_sample['pickup_latitude'].max():.4f}]")
    print(f"  - Longitude range: [{df_sample['pickup_longitude'].min():.4f}, {df_sample['pickup_longitude'].max():.4f}]")
    print(f"  - Hour range: [{df_sample['pickup_hour'].min():.2f}, {df_sample['pickup_hour'].max():.2f}]")
    print(f"  - Weekday range: [{df_sample['pickup_weekday'].min()}, {df_sample['pickup_weekday'].max()}]")

    return df_sample

# =============================================================================
# 2. FEATURE ENGINEERING WITH OPTIMIZED WEIGHTS
# =============================================================================

def create_spatiotemporal_features(df, spatial_weight=1.0, temporal_weight=1.0):
    """Full 6D spatiotemporal features with cyclical encoding."""
    features = df[['pickup_latitude', 'pickup_longitude', 'pickup_hour', 'pickup_weekday']].copy()
    
    # Spatial
    features['lat'] = df['pickup_latitude'] * spatial_weight
    features['lon'] = df['pickup_longitude'] * spatial_weight
    
    # Cyclical temporal (hour: 24h cycle, weekday: 7d cycle)
    features['hour_sin'] = np.sin(2 * np.pi * features['pickup_hour'] / 24) * temporal_weight
    features['hour_cos'] = np.cos(2 * np.pi * features['pickup_hour'] / 24) * temporal_weight
    features['weekday_sin'] = np.sin(2 * np.pi * features['pickup_weekday'] / 7) * temporal_weight
    features['weekday_cos'] = np.cos(2 * np.pi * features['pickup_weekday'] / 7) * temporal_weight
    
    feature_cols = ['lat', 'lon', 'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos']
    X = features[feature_cols].values
    
    return X, feature_cols, features

# =============================================================================
# 3. ENHANCED ST-DBSCAN
# =============================================================================

class STDBSCAN:
    def __init__(self, eps1=0.008, eps2=0.25, min_samples=20):
        self.eps1 = eps1    # Spatial (~800m)
        self.eps2 = eps2    # Temporal (~1.5h equivalent)
        self.min_samples = min_samples
        
    def fit(self, X):
        n_samples = X.shape[0]
        self.labels_ = np.full(n_samples, -1, dtype=int)
        cluster_id = 0
        visited = np.zeros(n_samples, dtype=bool)
        
        for i in range(n_samples):
            if visited[i]:
                continue
                
            spatial_neighbors = self._spatial_neighbors(X[i], X)
            temporal_neighbors = self._temporal_neighbors(X[i], X)
            neighbors = np.intersect1d(spatial_neighbors, temporal_neighbors)
            
            if len(neighbors) < self.min_samples:
                self.labels_[i] = -1
                visited[i] = True
                continue
                
            self._expand_cluster(i, neighbors, X, cluster_id)
            cluster_id += 1
            visited[i] = True
        
        return self
    
    def _spatial_neighbors(self, point, X):
        spatial_dist = np.sqrt((X[:, 0] - point[0])**2 + (X[:, 1] - point[1])**2)
        return np.where(spatial_dist <= self.eps1)[0]
    
    def _temporal_neighbors(self, point, X):
        # Cyclical distance in sin/cos space
        t_dist_hour = np.minimum(np.abs(X[:, 2] - point[2]), 2 - np.abs(X[:, 2] - point[2]))
        t_dist_day = np.minimum(np.abs(X[:, 4] - point[4]), 2 - np.abs(X[:, 4] - point[4]))
        temporal_dist = np.sqrt(t_dist_hour**2 + t_dist_day**2)
        return np.where(temporal_dist <= self.eps2)[0]
    
    def _expand_cluster(self, point_idx, neighbors, X, cluster_id):
        cluster = [point_idx]
        i = 0
        
        while i < len(cluster):
            current = cluster[i]
            spatial_n = self._spatial_neighbors(X[current], X)
            temporal_n = self._temporal_neighbors(X[current], X)
            new_neighbors = np.intersect1d(spatial_n, temporal_n)
            
            unvisited_neighbors = np.setdiff1d(new_neighbors, cluster)
            cluster.extend(unvisited_neighbors)
            
            self.labels_[current] = cluster_id
            i += 1

# =============================================================================
# 4. HIERARCHICAL PATTERN MERGING
# =============================================================================

def merge_weekday_patterns(df, st_labels, max_distance=1.5):
    """
    Merge ST-DBSCAN clusters with similar temporal profiles across weekdays.
    """
    print("  Extracting temporal profiles for each cluster...")
    
    # Create 7×24 temporal profile for each cluster
    cluster_profiles = []
    valid_clusters = []
    
    # Updated to use your new column names
    group_cols = ['pickup_weekday', 'pickup_hour']
    
    initial_clusters = len(np.unique(st_labels[st_labels >= 0]))
    print(f"    - Initial ST-DBSCAN clusters before merging: {initial_clusters}")
    
    for cluster_id in np.unique(st_labels[st_labels >= 0]):
        mask = st_labels == cluster_id
        cluster_size = mask.sum()
        
        if cluster_size < 50:  # Skip tiny clusters
            print(f"    - Skipping cluster {cluster_id} (size: {cluster_size} < 50)")
            continue
            
        profile = df[mask].groupby(group_cols).size().unstack(fill_value=0)
        profile = profile.reindex(columns=np.arange(24), fill_value=0).values.flatten()
        cluster_profiles.append(profile)
        valid_clusters.append(cluster_id)
    
    if len(cluster_profiles) == 0:
        print("    [!] No valid clusters to merge!")
        return st_labels
    
    print(f"    - Valid clusters after filtering: {len(valid_clusters)}")
    print(f"    - Cluster IDs being merged: {valid_clusters}")
    
    profiles = np.array(cluster_profiles)
    
    # Hierarchical clustering on temporal profiles
    print(f"  Running hierarchical clustering on temporal profiles...")
    print(f"    - Method: Ward linkage")
    print(f"    - Max distance threshold: {max_distance}")
    
    Z = linkage(profiles, method='ward')
    pattern_labels = fcluster(Z, t=max_distance, criterion='distance')
    
    unique_patterns = len(np.unique(pattern_labels))
    print(f"    - Final merged patterns: {unique_patterns}")
    print(f"    - Pattern distribution: {dict(zip(*np.unique(pattern_labels, return_counts=True)))}")
    
    # Create final hierarchical labels
    final_labels = st_labels.copy()
    for i, (cluster_id, pattern_id) in enumerate(zip(valid_clusters, pattern_labels)):
        final_labels[st_labels == cluster_id] = f"{cluster_id}_P{pattern_id}"
    
    return final_labels.astype(str)

# =============================================================================
# 5. PARAMETER OPTIMIZATION
# =============================================================================

def optimize_stdbscan_parameters(df, param_grid=None):
    """Grid search for optimal eps1, eps2, weights using silhouette score."""
    
    print("\n" + "="*80)
    print("STEP 2: PARAMETER OPTIMIZATION (GRID SEARCH)")
    print("="*80)
    
    if param_grid is None:
        param_grid = {
            'spatial_weight': [0.8, 1.0, 1.2],
            'temporal_weight': [0.6, 0.8, 1.0],
            'eps1': [0.006, 0.008, 0.010],
            'eps2': [0.20, 0.25, 0.30],
            'min_samples': [15, 20, 25]
        }
    
    print("\nParameter Grid Configuration:")
    for param_name, values in param_grid.items():
        print(f"  - {param_name}: {values}")
    
    # Calculate total combinations
    total_combinations = 1
    for values in param_grid.values():
        total_combinations *= len(values)
    print(f"\nTotal parameter combinations to test: {total_combinations}")
    
    best_score = -1
    best_params = {}
    tested_count = 0
    skipped_count = 0
    
    # Sample for faster optimization
    sample_df = df.sample(n=min(50000, len(df)), random_state=42).reset_index(drop=True)
    print(f"\nOptimization sample size: {len(sample_df):,} records")
    print("\nStarting grid search...\n")
    
    for sw in param_grid['spatial_weight']:
        for tw in param_grid['temporal_weight']:
            X, _, _ = create_spatiotemporal_features(sample_df, sw, tw)
            X_scaled = StandardScaler().fit_transform(X)
            
            for eps1 in param_grid['eps1']:
                for eps2 in param_grid['eps2']:
                    for ms in param_grid['min_samples']:
                        tested_count += 1
                        
                        st = STDBSCAN(eps1=eps1, eps2=eps2, min_samples=ms)
                        labels = st.fit(X_scaled).labels_
                        
                        n_clusters = len(np.unique(labels[labels >= 0]))
                        n_noise = (labels == -1).sum()
                        
                        if n_clusters < 3:
                            skipped_count += 1
                            print(f"  [{tested_count}/{total_combinations}] SKIP | "
                                  f"sw={sw}, tw={tw}, eps1={eps1}, eps2={eps2}, ms={ms} | "
                                  f"Clusters={n_clusters} (< 3 minimum)")
                            continue
                            
                        score = silhouette_score(X_scaled, labels)
                        
                        is_best = score > best_score
                        
                        print(f"  [{tested_count}/{total_combinations}] {'[*] BEST' if is_best else ' .'} | "
                              f"sw={sw}, tw={tw}, eps1={eps1}, eps2={eps2}, ms={ms} | "
                              f"Score={score:.4f} | Clusters={n_clusters} | Noise={n_noise}")
                        
                        if is_best:
                            best_score = score
                            best_params = {'spatial_weight': sw, 'temporal_weight': tw, 
                                         'eps1': eps1, 'eps2': eps2, 'min_samples': ms}
    
    print("\n" + "-"*80)
    print("OPTIMIZATION RESULTS:")
    print("-"*80)
    print(f"Total combinations tested: {tested_count}")
    print(f"Combinations skipped (< 3 clusters): {skipped_count}")
    print(f"Valid combinations: {tested_count - skipped_count}")
    print(f"\n[+] BEST PARAMETERS FOUND:")
    print(f"  - Spatial Weight: {best_params['spatial_weight']}")
    print(f"  - Temporal Weight: {best_params['temporal_weight']}")
    print(f"  - Spatial Epsilon (eps1): {best_params['eps1']} (~{best_params['eps1']*111:.0f}m)")
    print(f"  - Temporal Epsilon (eps2): {best_params['eps2']}")
    print(f"  - Min Samples: {best_params['min_samples']}")
    print(f"  - Best Silhouette Score: {best_score:.4f}")
    print("-"*80)
    
    return best_params

# =============================================================================
# 6. MAIN PIPELINE WITH MODEL SAVING
# =============================================================================

def spatiotemporal_clustering_pipeline(file_path, save_path='st_clusters_model.pkl'):
    """
    Complete pipeline: optimization -> clustering -> model saving.
    """
    # 1. Load larger sample
    df = load_taxi_data(file_path, sample_size=250_000)
    
    # 2. Optimize parameters
    best_params = optimize_stdbscan_parameters(df)
    
    # 3. Apply optimal parameters to full sample
    print("\n" + "="*80)
    print("STEP 3: APPLYING OPTIMIZED ST-DBSCAN TO SAMPLE")
    print("="*80)
    print(f"\nApplying ST-DBSCAN with optimized parameters to {len(df):,} sample records...")
    
    X, feature_cols, features_df = create_spatiotemporal_features(
        df, best_params['spatial_weight'], best_params['temporal_weight']
    )
    
    print(f"[+] Feature matrix created: shape {X.shape}")
    print(f"  - Features: {feature_cols}")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"[+] Features standardized")
    
    st_dbscan = STDBSCAN(
        eps1=best_params['eps1'],
        eps2=best_params['eps2'],
        min_samples=best_params['min_samples']
    )
    
    print(f"\nRunning ST-DBSCAN clustering...")
    st_labels = st_dbscan.fit(X_scaled).labels_
    print(f"[+] Clustering complete")
    
    # 4. Hierarchical pattern merging
    print("\n" + "="*80)
    print("STEP 4: HIERARCHICAL PATTERN MERGING")
    print("="*80 + "\n")
    final_labels = merge_weekday_patterns(df, st_labels)
    df['st_cluster'] = st_labels
    df['final_cluster'] = final_labels
    
    # 5. Save complete model
    print("\n" + "="*80)
    print("STEP 5: SAVING MODEL")
    print("="*80)
    
    model = {
        'scaler': scaler,
        'st_dbscan': st_dbscan,
        'best_params': best_params,
        'feature_cols': feature_cols,
        'cluster_mapping': dict(zip(np.unique(final_labels), range(len(np.unique(final_labels)))))
    }
    joblib.dump(model, save_path)
    print(f"[+] Model saved to: {save_path}")
    print(f"  - Scaler: StandardScaler")
    print(f"  - ST-DBSCAN: eps1={best_params['eps1']}, eps2={best_params['eps2']}, min_samples={best_params['min_samples']}")
    print(f"  - Features: {feature_cols}")
    
    # 6. Results summary
    print("\n" + "="*80)
    print("CLUSTERING RESULTS (SAMPLE)")
    print("="*80)
    
    n_st_clusters = len(np.unique(st_labels[st_labels >= 0]))
    n_final_patterns = len(np.unique(final_labels))
    n_noise = (st_labels == -1).sum()
    silhouette = silhouette_score(X_scaled, st_labels)
    
    print(f"\nCluster Statistics:")
    print(f"  - ST-DBSCAN clusters (before merging): {n_st_clusters}")
    print(f"  - Final patterns (after merging): {n_final_patterns}")
    print(f"  - Noise points: {n_noise:,} ({100*n_noise/len(st_labels):.2f}%)")
    print(f"  - Silhouette Score: {silhouette:.4f}")
    
    print(f"\nCluster Size Distribution:")
    cluster_sizes = df['final_cluster'].value_counts().sort_values(ascending=False)
    for idx, (cluster_id, size) in enumerate(cluster_sizes.head(10).items(), 1):
        percentage = 100 * size / len(df)
        print(f"  {idx:2d}. Pattern {cluster_id:15s}: {size:7,} records ({percentage:5.2f}%)")
    
    if len(cluster_sizes) > 10:
        print(f"  ... and {len(cluster_sizes) - 10} more patterns")
    
    return df, model

# =============================================================================
# 7. APPLY TO FULL DATASET
# =============================================================================

def apply_to_full_dataset(full_file_path, model_path='st_clusters_model.pkl'):
    """Apply trained model to complete NYC taxi dataset (Parquet)."""
    
    print("\n" + "="*80)
    print("STEP 6: APPLYING MODEL TO FULL DATASET")
    print("="*80)
    
    print(f"\nLoading full dataset from: {full_file_path}")
    
    # Load model
    print(f"Loading model from: {model_path}")
    model = joblib.load(model_path)
    print(f"[+] Model loaded successfully")
    print(f"  - Model parameters: {model['best_params']}")
    
    # Load full parquet data
    print(f"\nReading full parquet file...")
    df_full = pd.read_parquet(full_file_path)
    print(f"[+] Full dataset loaded: {len(df_full):,} records")
    print(f"  - Memory usage: {df_full.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print(f"\nCreating feature matrix for full dataset...")
    
    # Create features using SAME transformation logic and weights as training
    features = df_full[['pickup_latitude', 'pickup_longitude', 'pickup_hour', 'pickup_weekday']].copy()
    
    for col in model['feature_cols']:
        if col == 'lat':
            features[col] = df_full['pickup_latitude'] * model['best_params']['spatial_weight']
        elif col == 'lon':
            features[col] = df_full['pickup_longitude'] * model['best_params']['spatial_weight']
        else:
            # Cyclical features
            if 'hour' in col:
                features[col] = (np.sin(2 * np.pi * features['pickup_hour'] / 24) * model['best_params']['temporal_weight'] 
                            if 'sin' in col 
                            else np.cos(2 * np.pi * features['pickup_hour'] / 24) * model['best_params']['temporal_weight'])
            else:
                features[col] = (np.sin(2 * np.pi * features['pickup_weekday'] / 7) * model['best_params']['temporal_weight'] 
                            if 'sin' in col 
                            else np.cos(2 * np.pi * features['pickup_weekday'] / 7) * model['best_params']['temporal_weight'])
    
    X = features[model['feature_cols']].values
    print(f"[+] Feature matrix created: shape {X.shape}")
    
    print(f"\nScaling features using trained scaler...")
    X_scaled = model['scaler'].transform(X)
    print(f"[+] Features scaled")
    
    # Predict clusters
    print(f"\nRunning ST-DBSCAN clustering on full dataset...")
    print(f"  (This may take several minutes for {len(df_full):,} records)")
    
    st_full = STDBSCAN(**{k: v for k, v in model['st_dbscan'].__dict__.items() 
                          if k in ['eps1', 'eps2', 'min_samples']})
    
    labels = st_full.fit(X_scaled).labels_
    print(f"[+] Clustering complete")
    
    n_st_clusters = len(np.unique(labels[labels >= 0]))
    n_noise = (labels == -1).sum()
    print(f"  - ST-DBSCAN clusters found: {n_st_clusters}")
    print(f"  - Noise points: {n_noise:,} ({100*n_noise/len(labels):.2f}%)")
    
    # Merge patterns
    print(f"\nMerging temporal patterns...")
    final_labels = merge_weekday_patterns(df_full, labels, max_distance=1.5)
    print(f"[+] Pattern merging complete")
    
    df_full['st_cluster'] = labels
    df_full['final_cluster'] = final_labels
    
    print(f"\nFinal Result Statistics:")
    n_final_patterns = len(np.unique(final_labels))
    print(f"  - Final patterns: {n_final_patterns}")
    
    print(f"\nTop 10 Pattern Size Distribution:")
    cluster_sizes = df_full['final_cluster'].value_counts().sort_values(ascending=False)
    for idx, (cluster_id, size) in enumerate(cluster_sizes.head(10).items(), 1):
        percentage = 100 * size / len(df_full)
        print(f"  {idx:2d}. Pattern {cluster_id:15s}: {size:8,} records ({percentage:5.2f}%)")
    
    if len(cluster_sizes) > 10:
        print(f"  ... and {len(cluster_sizes) - 10} more patterns")
    
    return df_full

# =============================================================================
# 8. VISUALIZATION AND EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Define paths
    BASE_DIR = 'C:/Users/Anya/master_thesis/output'
    INPUT_FILE = os.path.join(BASE_DIR, 'taxi_data_cleaned_full.parquet')
    OUTPUT_FILE = os.path.join(BASE_DIR, 'taxi_data_cleaned_full_with_clusters.parquet')
    MODEL_PATH = os.path.join(BASE_DIR, 'models', 'nyc_taxi_st_model.pkl')
    VIZ_PATH = os.path.join(BASE_DIR, 'optimized_st_clustering_results.png')
    
    # Ensure model dir exists
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    print("\n" + "="*80)
    print("NYC YELLOW TAXI - SPATIOTEMPORAL CLUSTERING PIPELINE")
    print("="*80)
    
    # 1. Train Pipeline on Sample
    print("\n[PHASE 1: TRAINING]")
    df_sample_results, model = spatiotemporal_clustering_pipeline(INPUT_FILE, MODEL_PATH)
    
    # 2. Quick Visualization of Sample Results
    print("\n" + "="*80)
    print("STEP 7: GENERATING VISUALIZATIONS (SAMPLE)")
    print("="*80)
    print(f"\nGenerating visualization plots...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Spatial clusters
    unique_labels = df_sample_results['final_cluster'].unique()
    label_map = {label: i for i, label in enumerate(unique_labels)}
    c_codes = df_sample_results['final_cluster'].map(label_map)
    
    axes[0].scatter(df_sample_results['pickup_longitude'], df_sample_results['pickup_latitude'], 
                   c=c_codes, cmap='tab20', s=1, alpha=0.6)
    axes[0].set_title('Final Spatiotemporal Patterns (Sample)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    axes[0].grid(True, alpha=0.3)
    
    # Temporal patterns
    axes[1].scatter(df_sample_results['pickup_hour'], df_sample_results['pickup_weekday'], 
                   c=c_codes, cmap='tab20', s=1, alpha=0.6)
    axes[1].set_title('Temporal Distribution', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Hour')
    axes[1].set_ylabel('Weekday')
    axes[1].grid(True, alpha=0.3)
    
    # Cluster sizes
    cluster_sizes = df_sample_results['final_cluster'].value_counts()
    axes[2].bar(range(len(cluster_sizes)), cluster_sizes.values)
    axes[2].set_title('Cluster Sizes', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Pattern ID')
    axes[2].set_ylabel('Count')
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(VIZ_PATH, dpi=300, bbox_inches='tight')
    print(f"[+] Visualization saved to: {VIZ_PATH}")
    # plt.show()
    
    # 3. Apply to Full Dataset and Save
    print("\n[PHASE 2: FULL DATASET INFERENCE]")
    df_full_results = apply_to_full_dataset(INPUT_FILE, MODEL_PATH)
    
    # 4. Save Results
    print("\n" + "="*80)
    print("STEP 8: SAVING FULL DATASET WITH CLUSTERS")
    print("="*80)
    print(f"\nSaving full dataset with clusters...")
    print(f"  Output file: {OUTPUT_FILE}")
    print(f"  Records to save: {len(df_full_results):,}")
    
    df_full_results.to_parquet(OUTPUT_FILE, index=False)
    
    output_size = os.path.getsize(OUTPUT_FILE) / 1024**2
    print(f"[+] Successfully saved!")
    print(f"  File size: {output_size:.2f} MB")
    print(f"  New columns added: 'st_cluster', 'final_cluster'")
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    print(f"\nSummary:")
    print(f"  - Sample dataset: {len(df_sample_results):,} records")
    print(f"  - Full dataset: {len(df_full_results):,} records")
    print(f"  - Model saved: {MODEL_PATH}")
    print(f"  - Full results saved: {OUTPUT_FILE}")
    print(f"  - Visualization saved: {VIZ_PATH}")
    print("="*80 + "\n")
