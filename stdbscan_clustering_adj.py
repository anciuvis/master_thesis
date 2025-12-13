import pandas as pd
import numpy as np
import os
import json
import pickle
import joblib
import warnings
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from scipy.cluster.hierarchy import linkage, fcluster

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)

# =============================================================================
# CONFIGURATION AND PATHS
# =============================================================================

CONFIG = {
    'BASE_DIR': 'C:/Users/Anya/master_thesis/output',
    'CHECKPOINTS_DIR': 'C:/Users/Anya/master_thesis/output/checkpoints',
    'INPUT_FILE': 'C:/Users/Anya/master_thesis/output/taxi_data_cleaned_full.parquet',
    'OUTPUT_FILE': 'C:/Users/Anya/master_thesis/output/taxi_data_cleaned_full_with_clusters.parquet',
    
    'SAMPLE_SIZE': 2_500_000,         # For final training
    'OPTIMIZATION_SAMPLE': 500_000,   # Large base sample for grid search
    'PER_COMBO_SAMPLE': 60_000,       # Subset size for each grid iteration (Speedup)
    'BATCH_SIZE': 1_000_000,          # For full dataset inference
}

# Create checkpoint directory
Path(CONFIG['CHECKPOINTS_DIR']).mkdir(parents=True, exist_ok=True)

# =============================================================================
# CHECKPOINT MANAGEMENT
# =============================================================================

class CheckpointManager:
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def save_checkpoint(self, name, data, data_type='pkl'):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if data_type == 'pkl':
            path = self.checkpoint_dir / f"{name}_{timestamp}.pkl"
            with open(path, 'wb') as f: pickle.dump(data, f)
        elif data_type == 'joblib':
            path = self.checkpoint_dir / f"{name}_{timestamp}.joblib"
            joblib.dump(data, path)
        elif data_type == 'json':
            path = self.checkpoint_dir / f"{name}_{timestamp}.json"
            with open(path, 'w') as f: json.dump(data, f, indent=4)
        elif data_type == 'parquet':
            path = self.checkpoint_dir / f"{name}_{timestamp}.parquet"
            data.to_parquet(path, index=False)
        print(f"[CHECKPOINT] Saved: {path}")
        return path
    
    def load_latest_checkpoint(self, name, data_type='pkl'):
        pattern = f"{name}_*.{data_type.replace('pkl', 'pkl').replace('joblib', 'joblib').replace('json', 'json').replace('parquet', 'parquet')}"
        checkpoints = sorted(self.checkpoint_dir.glob(pattern))
        if not checkpoints:
            print(f"[!] No checkpoint found for {name}")
            return None
        latest = checkpoints[-1]
        print(f"[CHECKPOINT] Loading: {latest}")
        if data_type == 'pkl':
            with open(latest, 'rb') as f: return pickle.load(f)
        elif data_type == 'joblib': return joblib.load(latest)
        elif data_type == 'json':
            with open(latest, 'r') as f: return json.load(f)
        elif data_type == 'parquet': return pd.read_parquet(latest)

checkpoint_mgr = CheckpointManager(CONFIG['CHECKPOINTS_DIR'])

# =============================================================================
# 1. DATA LOADING
# =============================================================================

def load_taxi_data(sample_size=None, use_checkpoint=True):
    print("\n" + "="*80)
    print("STEP 1: LOADING DATA")
    print("="*80)
    
    if use_checkpoint:
        df_sample = checkpoint_mgr.load_latest_checkpoint('df_sample', 'parquet')
        if df_sample is not None:
            print(f"[+] Loaded sample from checkpoint: {len(df_sample):,} records")
            return df_sample
    
    print(f"Loading from: {CONFIG['INPUT_FILE']}")
    data = pd.read_parquet(CONFIG['INPUT_FILE'])
    print(f"[+] Total records: {len(data):,}")
    
    if sample_size is None:
        sample_size = CONFIG['SAMPLE_SIZE']
    
    print(f"\nSampling {sample_size:,} records...")
    df_sample = data.sample(n=min(sample_size, len(data)), random_state=42)
    print(f"[+] Sample size: {len(df_sample):,} records")
    
    checkpoint_mgr.save_checkpoint('df_sample', df_sample, 'parquet')
    return df_sample

# =============================================================================
# 2. FEATURE ENGINEERING (KM PROJECTION)
# =============================================================================

def create_spatiotemporal_features(df, spatial_weight=1.0, temporal_weight=1.0):
    # NYC Center (approx)
    NYC_LAT = 40.7128
    NYC_LON = -74.0060
    LAT_KM = 111
    LON_KM = 85
    
    features = pd.DataFrame()
    # Convert Lat/Lon to Kilometers from center
    features['lat_km'] = (df['pickup_latitude'] - NYC_LAT) * LAT_KM * spatial_weight
    features['lon_km'] = (df['pickup_longitude'] - NYC_LON) * LON_KM * spatial_weight
    
    # Scale time (5.0 factor makes 12 hours ≈ 10km distance)
    time_scale = 5.0 * temporal_weight
    features['hour_sin'] = np.sin(2 * np.pi * df['pickup_hour'] / 24) * time_scale
    features['hour_cos'] = np.cos(2 * np.pi * df['pickup_hour'] / 24) * time_scale
    features['weekday_sin'] = np.sin(2 * np.pi * df['pickup_weekday'] / 7) * time_scale
    features['weekday_cos'] = np.cos(2 * np.pi * df['pickup_weekday'] / 7) * time_scale
    
    X = features.values
    return X, list(features.columns), features

# =============================================================================
# 3. ST-DBSCAN (OPTIMIZED EUCLIDEAN)
# =============================================================================

class STDBSCAN:
    def __init__(self, eps1=0.5, eps2=1.0, min_samples=30):
        self.eps1 = eps1    # Spatial Radius (km)
        self.eps2 = eps2    # Temporal Radius (distance units)
        self.min_samples = min_samples
        
    def fit(self, X):
        n_samples = X.shape[0]
        self.labels_ = np.full(n_samples, -1, dtype=int)
        cluster_id = 0
        visited = np.zeros(n_samples, dtype=bool)
        
        # X columns: [lat_km, lon_km, h_sin, h_cos, w_sin, w_cos]
        spatial_data = X[:, :2]
        temporal_data = X[:, 2:]
        
        # print(f"Running ST-DBSCAN on {n_samples:,} samples...")
        
        for i in range(n_samples):
            if visited[i]: continue
            
            # Spatial Neighbors (Euclidean in KM)
            s_dist = np.sqrt(np.sum((spatial_data - spatial_data[i])**2, axis=1))
            spatial_neighbors = np.where(s_dist <= self.eps1)[0]
            
            # Temporal Neighbors (Euclidean in Scaled Sin/Cos space)
            t_dist = np.sqrt(np.sum((temporal_data - temporal_data[i])**2, axis=1))
            temporal_neighbors = np.where(t_dist <= self.eps2)[0]
            
            neighbors = np.intersect1d(spatial_neighbors, temporal_neighbors)
            
            if len(neighbors) < self.min_samples:
                self.labels_[i] = -1
                visited[i] = True
                continue
                
            self._expand_cluster(i, neighbors, spatial_data, temporal_data, cluster_id, visited)
            cluster_id += 1
            visited[i] = True
            
        # print(f"Clustering complete. Found {cluster_id} clusters.")
        return self

    def _expand_cluster(self, seed_idx, neighbors, s_data, t_data, cluster_id, visited):
        cluster = [seed_idx]
        visited[seed_idx] = True
        self.labels_[seed_idx] = cluster_id
        
        i = 0
        while i < len(cluster):
            current = cluster[i]
            
            s_dist = np.sqrt(np.sum((s_data - s_data[current])**2, axis=1))
            s_n = np.where(s_dist <= self.eps1)[0]
            
            t_dist = np.sqrt(np.sum((t_data - t_data[current])**2, axis=1))
            t_n = np.where(t_dist <= self.eps2)[0]
            
            new_neighbors = np.intersect1d(s_n, t_n)
            
            if len(new_neighbors) >= self.min_samples:
                unvisited = np.where(~visited[new_neighbors])[0]
                actual_indices = new_neighbors[unvisited]
                visited[actual_indices] = True
                self.labels_[actual_indices] = cluster_id
                cluster.extend(actual_indices)
            i += 1

# =============================================================================
# 4. HIERARCHICAL PATTERN MERGING
# =============================================================================

def merge_weekday_patterns(df, st_labels, max_distance=1.5, use_checkpoint=True):
    if use_checkpoint:
        final_labels = checkpoint_mgr.load_latest_checkpoint('final_labels', 'pkl')
        if final_labels is not None:
            print(f"[+] Loaded final_labels from checkpoint")
            return final_labels
    
    print("  Extracting temporal profiles for each cluster...")
    cluster_profiles = []
    valid_clusters = []
    
    initial_clusters = len(np.unique(st_labels[st_labels >= 0]))
    print(f"    - Initial clusters: {initial_clusters}")
    
    for cluster_id in np.unique(st_labels[st_labels >= 0]):
        mask = st_labels == cluster_id
        if mask.sum() < 50: continue
        
        cluster_data = df[mask]
        
        # Circular statistics for profiles
        hour_sin = np.sin(2 * np.pi * cluster_data['pickup_hour'] / 24)
        hour_cos = np.cos(2 * np.pi * cluster_data['pickup_hour'] / 24)
        weekday_sin = np.sin(2 * np.pi * cluster_data['pickup_weekday'] / 7)
        weekday_cos = np.cos(2 * np.pi * cluster_data['pickup_weekday'] / 7)
        
        profile = np.array([
            hour_sin.mean(), hour_cos.mean(), np.sqrt(hour_sin.mean()**2 + hour_cos.mean()**2),
            weekday_sin.mean(), weekday_cos.mean(), np.sqrt(weekday_sin.mean()**2 + weekday_cos.mean()**2)
        ])
        
        cluster_profiles.append(profile)
        valid_clusters.append(cluster_id)
    
    if len(cluster_profiles) == 0:
        print("    [!] No valid clusters!")
        return st_labels
    
    print(f"    - Valid clusters: {len(valid_clusters)}")
    print(f"  Running hierarchical clustering...")
    
    profiles = np.array(cluster_profiles)
    Z = linkage(profiles, method='ward')
    pattern_labels = fcluster(Z, t=max_distance, criterion='distance')
    
    unique_patterns = len(np.unique(pattern_labels))
    print(f"    - Final merged patterns: {unique_patterns}")
    
    final_labels = st_labels.astype(str).copy()
    for i, (cluster_id, pattern_id) in enumerate(zip(valid_clusters, pattern_labels)):
        # Assign all points in this cluster to the merged pattern
        # The array can now hold these string values
        final_labels[st_labels == cluster_id] = f"{cluster_id}_P{pattern_id}"
    
    checkpoint_mgr.save_checkpoint('final_labels', final_labels, 'pkl')
    return final_labels

# =============================================================================
# 5. PARAMETER OPTIMIZATION
# =============================================================================

def optimize_stdbscan_parameters(df, param_grid=None, use_checkpoint=True):
    print("\n" + "="*80)
    print("STEP 2: PARAMETER OPTIMIZATION (SUB-SAMPLED GRID SEARCH)")
    print("="*80)
    
    if use_checkpoint:
        best_params = checkpoint_mgr.load_latest_checkpoint('best_params', 'json')
        if best_params is not None:
            print(f"[+] Loaded best_params from checkpoint: {best_params}")
            return best_params
    
    if param_grid is None:
        # param_grid = {
        #     'spatial_weight': [1.0],
        #     'temporal_weight': [1.0],
        #     'eps1': [0.5, 0.8],      
        #     'eps2': [1.0, 1.5, 2.0],      
        #     'min_samples': [10, 15]   
        # }
        param_grid = {
            'spatial_weight': [1.0],
            'temporal_weight': [1.0],
            'eps1': [0.1, 0.3, 0.4],
            'eps2': [1.0, 1.5],
            'min_samples': [50, 100]
        }
    
    print("\nParameter Grid:")
    for param_name, values in param_grid.items():
        print(f"  - {param_name}: {values}")
    
    total_combinations = np.prod([len(v) for v in param_grid.values()])
    print(f"\nTotal combinations: {total_combinations}")
    
    # 1. Load large base sample for density
    sample_df = df.sample(n=min(CONFIG['OPTIMIZATION_SAMPLE'], len(df)), random_state=42).reset_index(drop=True)
    print(f"Base Optimization sample: {len(sample_df):,}")
    print(f"Per-Combo Processing Subset: {CONFIG['PER_COMBO_SAMPLE']:,} (Speedup)")
    
    best_score = -1
    best_params = {}
    tested_count = 0
    
    for sw in param_grid['spatial_weight']:
        for tw in param_grid['temporal_weight']:
            # Create features once for the full sample
            X_full, _, _ = create_spatiotemporal_features(sample_df, sw, tw)
            
            for eps1 in param_grid['eps1']:
                for eps2 in param_grid['eps2']:
                    for ms in param_grid['min_samples']:
                        tested_count += 1
                        
                        # Sub-sample for this iteration to speed up clustering loop
                        subset_idx = np.random.choice(len(sample_df), CONFIG['PER_COMBO_SAMPLE'], replace=False)
                        X_subset = X_full[subset_idx]
                        
                        st = STDBSCAN(eps1=eps1, eps2=eps2, min_samples=ms)
                        labels = st.fit(X_subset).labels_
                        
                        n_clusters = len(np.unique(labels[labels >= 0]))
                        noise_ratio = (labels == -1).mean()
                        
                        if n_clusters < 3 or noise_ratio > 0.95:
                            print(f"  [{tested_count}/{total_combinations}] SKIP | "
                                  f"eps1={eps1}, eps2={eps2}, ms={ms} | "
                                  f"Clusters={n_clusters}, Noise={noise_ratio:.2f}")
                            continue
                        
                        # Silhouette on even smaller subset for speed
                        eval_n = 5000
                        eval_idx = np.random.choice(len(X_subset), min(len(X_subset), eval_n), replace=False)
                        X_eval = X_subset[eval_idx]
                        labels_eval = labels[eval_idx]
                        
                        if len(np.unique(labels_eval[labels_eval >= 0])) < 2:
                             continue

                        score = silhouette_score(X_eval, labels_eval)
                        is_best = score > best_score
                        
                        print(f"  [{tested_count}/{total_combinations}] {'[*] BEST' if is_best else ' .'} | "
                              f"eps1={eps1}, eps2={eps2}, ms={ms} | "
                              f"Score={score:.4f} | Clusters={n_clusters}")
                        
                        if is_best:
                            best_score = score
                            best_params = {
                                'spatial_weight': float(sw), 'temporal_weight': float(tw), 
                                'eps1': float(eps1), 'eps2': float(eps2), 'min_samples': int(ms)
                            }
    
    print(f"\n[+] BEST PARAMETERS:")
    for k, v in best_params.items(): print(f"  - {k}: {v}")
    print(f"  - Best Score: {best_score:.4f}")
    
    checkpoint_mgr.save_checkpoint('best_params', best_params, 'json')
    return best_params

# =============================================================================
# 6. TRAINING PIPELINE
# =============================================================================

def train_clustering_model(use_checkpoint=True):
    print("\n[PHASE 1: TRAINING]")
    df = load_taxi_data(use_checkpoint=use_checkpoint)
    best_params = optimize_stdbscan_parameters(df, use_checkpoint=False)
    
    print("\n" + "="*80)
    print("STEP 3: APPLYING OPTIMIZED ST-DBSCAN TO SAMPLE")
    print("="*80 + "\n")
    
    X, feature_cols, features_df = create_spatiotemporal_features(
        df, best_params['spatial_weight'], best_params['temporal_weight']
    )
    st_dbscan = STDBSCAN(
        eps1=best_params['eps1'], eps2=best_params['eps2'], min_samples=best_params['min_samples']
    )

    # Try to load from checkpoint first
    if use_checkpoint:
        st_labels = checkpoint_mgr.load_latest_checkpoint('st_labels_sample', 'pkl')
        if st_labels is not None:
            print("[+] Loaded st_labels_sample from checkpoint")
        else:
            print("Computing ST-DBSCAN clustering...")
            st_labels = st_dbscan.fit(X).labels_
            checkpoint_mgr.save_checkpoint('st_labels_sample', st_labels, 'pkl')
    else:
        print("Computing ST-DBSCAN clustering...")
        st_labels = st_dbscan.fit(X).labels_
        checkpoint_mgr.save_checkpoint('st_labels_sample', st_labels, 'pkl')
    
    print("\n" + "="*80)
    print("STEP 4: HIERARCHICAL PATTERN MERGING")
    print("="*80 + "\n")

    print("\n" + "="*80)
    print("HIERARCHICAL MERGING DIAGNOSTICS")
    print("="*80)

    # Get all ST-DBSCAN clusters (before merging)
    st_clusters = np.unique(st_labels[st_labels >= 0])
    print(f"\nTotal ST-DBSCAN clusters: {len(st_clusters)}")

    # Build profiles (copied from merge_weekday_patterns)
    cluster_profiles = []
    valid_clusters = []

    for cluster_id in st_clusters:
        mask = st_labels == cluster_id
        if mask.sum() < 50: 
            continue
        
        cluster_data = df[mask]
        
        hour_sin = np.sin(2 * np.pi * cluster_data['pickup_hour'] / 24)
        hour_cos = np.cos(2 * np.pi * cluster_data['pickup_hour'] / 24)
        weekday_sin = np.sin(2 * np.pi * cluster_data['pickup_weekday'] / 7)
        weekday_cos = np.cos(2 * np.pi * cluster_data['pickup_weekday'] / 7)
        
        profile = np.array([
            hour_sin.mean(), hour_cos.mean(), np.sqrt(hour_sin.mean()**2 + hour_cos.mean()**2),
            weekday_sin.mean(), weekday_cos.mean(), np.sqrt(weekday_sin.mean()**2 + weekday_cos.mean()**2)
        ])
        
        cluster_profiles.append(profile)
        valid_clusters.append(cluster_id)

    profiles = np.array(cluster_profiles)

    # Try different max_distance values
    from scipy.cluster.hierarchy import linkage, fcluster

    Z = linkage(profiles, method='ward')

    print("\nNumber of clusters at different distance thresholds:")
    for max_dist in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
        pattern_labels = fcluster(Z, t=max_dist, criterion='distance')
        n_patterns = len(np.unique(pattern_labels))
        print(f"  max_distance={max_dist}: {n_patterns} patterns")

    
    final_labels = merge_weekday_patterns(df, st_labels, use_checkpoint=use_checkpoint)
    df['st_cluster'] = st_labels
    df['final_cluster'] = final_labels

    checkpoint_mgr.save_checkpoint('df_sample', df, 'parquet')
    print("[+] Updated df_sample checkpoint with cluster columns")
    
    print("\n" + "="*80)
    print("STEP 5: SAVING MODEL")
    print("="*80)
    
    scaler = StandardScaler()
    # Note: We fit the scaler just to have it in the model object for KNN, 
    # even though we didn't use it for DBSCAN. KNN works better with scaling 
    # IF we decide to use it, but since features are already in KM/Distance units, 
    # we can actually skip scaling or use Identity.
    scaler.fit(X)
    unique_labels = np.unique(final_labels.astype(str))
    
    model = {
        'scaler': scaler,
        'st_dbscan': st_dbscan,
        'best_params': best_params,
        'feature_cols': feature_cols,
        'cluster_mapping': dict(zip(unique_labels, range(len(unique_labels))))
    }
    checkpoint_mgr.save_checkpoint('trained_model', model, 'joblib')
    
    # Results
    n_st_clusters = len(np.unique(st_labels[st_labels >= 0]))
    n_final = len(np.unique(final_labels))
    n_noise = (st_labels == -1).sum()
    
    print("\nStatistics:")
    print(f"  - ST-DBSCAN clusters: {n_st_clusters}")
    print(f"  - Final patterns: {n_final}")
    print(f"  - Noise points: {n_noise:,} ({100*n_noise/len(st_labels):.2f}%)")
    
    return df, model

# =============================================================================
# 7. INFERENCE WITH KNN
# =============================================================================

def apply_to_full_dataset_knn(use_checkpoint=True):
    print("\n[PHASE 2: FULL DATASET INFERENCE (KNN CLASSIFIER)]")
    
    model = checkpoint_mgr.load_latest_checkpoint('trained_model', 'joblib')
    if model is None: raise ValueError("No trained model found.")
    
    print(f"\nLoading full dataset...")
    df_full = pd.read_parquet(CONFIG['INPUT_FILE'])
    print(f"[+] Full dataset: {len(df_full):,} records")
    
    df_sample = checkpoint_mgr.load_latest_checkpoint('df_sample', 'parquet')
    final_labels_sample = checkpoint_mgr.load_latest_checkpoint('final_labels', 'pkl')
    
    print(f"\nTraining KNN classifier (k=5)...")
    X_train, _, _ = create_spatiotemporal_features(
        df_sample, model['best_params']['spatial_weight'], model['best_params']['temporal_weight']
    )
    
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1, weights='distance')
    knn.fit(X_train, final_labels_sample)
    print(f"[+] KNN classifier trained")
    
    print(f"\nProcessing full dataset in batches...")
    batch_size = CONFIG['BATCH_SIZE']
    n_batches = int(np.ceil(len(df_full) / batch_size))
    all_clusters = []
    
    for batch_idx in range(n_batches):
        print(f"  Batch {batch_idx + 1}/{n_batches}...", end=' ')
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(df_full))
        df_batch = df_full.iloc[start_idx:end_idx].copy()
        
        X_batch, _, _ = create_spatiotemporal_features(
            df_batch, model['best_params']['spatial_weight'], model['best_params']['temporal_weight']
        )
        
        batch_clusters = knn.predict(X_batch)
        df_batch['final_cluster'] = batch_clusters
        all_clusters.append(df_batch[['final_cluster']])
        print(f"OK")
    
    print(f"\nMerging batches...")
    df_full['final_cluster'] = pd.concat(all_clusters, ignore_index=True)['final_cluster'].values
    return df_full

# =============================================================================
# 8. VISUALIZATION
# =============================================================================

def visualize_sample_results():
    print("\n" + "="*80)
    print("STEP 7: VISUALIZATIONS")
    print("="*80 + "\n")
    
    df_sample = checkpoint_mgr.load_latest_checkpoint('df_sample', 'parquet')
    if df_sample is None:
        print("[!] No df_sample checkpoint found. Skipping visualization.")
        return
    
    print(f"[+] Loaded df_sample: {len(df_sample):,} records")
    
    # Create color mapping for clusters
    unique_labels = df_sample['final_cluster'].unique()
    label_map = {label: i for i, label in enumerate(unique_labels)}
    c_codes = df_sample['final_cluster'].map(label_map)
    
    # ========================================================================
    # VISUALIZATION 1: SPATIAL DISTRIBUTION
    # ========================================================================
    print("\n[*] Creating spatial distribution map...")
    fig, ax = plt.subplots(figsize=(16, 14))
    scatter = ax.scatter(
        df_sample['pickup_longitude'], 
        df_sample['pickup_latitude'], 
        c=c_codes, 
        cmap='tab20b',
        s=5,
        alpha=0.7,
        edgecolors='none'
    )
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title('Spatiotemporal Clusters - NYC Taxi Pickup Locations', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Cluster ID', fontsize=11)
    
    # Add NYC landmarks for reference
    landmarks = {
        'Manhattan Center': (40.7580, -73.9855),
        'JFK Airport': (40.6413, -73.7781),
        'LaGuardia': (40.7769, -73.8740),
    }
    for name, (lat, lon) in landmarks.items():
        ax.plot(lon, lat, marker='*', markersize=20, color='red', alpha=0.7)
        ax.text(lon + 0.01, lat + 0.01, name, fontsize=10, fontweight='bold', color='red')
    
    viz_path_1 = os.path.join(CONFIG['BASE_DIR'], 'viz_01_spatial_distribution.png')
    plt.tight_layout()
    plt.savefig(viz_path_1, dpi=300, bbox_inches='tight')
    print(f"[+] Saved: {viz_path_1}")
    plt.close()
    
    # ========================================================================
    # VISUALIZATION 2: TEMPORAL HEATMAP
    # ========================================================================
    print("[*] Creating temporal heatmap...")
    fig, ax = plt.subplots(figsize=(14, 8))
    
    temporal_pivot = df_sample.pivot_table(
        values='final_cluster', 
        index='pickup_hour',
        columns='pickup_weekday',
        aggfunc='count'
    )
    
    im = ax.imshow(temporal_pivot.fillna(0), cmap='YlOrRd', aspect='auto')
    ax.set_xlabel('Day of Week (0=Monday, 6=Sunday)', fontsize=12)
    ax.set_ylabel('Hour of Day (0-23)', fontsize=12)
    ax.set_title('Taxi Activity Heatmap: Hour × Day of Week', fontsize=14, fontweight='bold')
    ax.set_xticks(range(7))
    ax.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Number of Pickups', fontsize=11)
    
    viz_path_2 = os.path.join(CONFIG['BASE_DIR'], 'viz_02_temporal_heatmap.png')
    plt.tight_layout()
    plt.savefig(viz_path_2, dpi=300, bbox_inches='tight')
    print(f"[+] Saved: {viz_path_2}")
    plt.close()
    
    # ========================================================================
    # VISUALIZATION 3: CLUSTER SIZE DISTRIBUTION
    # ========================================================================
    print("[*] Creating cluster size distribution...")
    fig, ax = plt.subplots(figsize=(14, 6))
    
    cluster_sizes = df_sample['final_cluster'].value_counts().sort_values(ascending=False)
    
    n_clusters_to_show = min(30, len(cluster_sizes))
    top_clusters = cluster_sizes.head(n_clusters_to_show)
    
    colors_bar = plt.cm.tab20b(np.linspace(0, 1, n_clusters_to_show))
    bars = ax.bar(range(len(top_clusters)), top_clusters.values, color=colors_bar)
    
    ax.set_xlabel('Cluster ID (ranked by size)', fontsize=12)
    ax.set_ylabel('Number of Points', fontsize=12)
    ax.set_title(f'Cluster Size Distribution (Top {n_clusters_to_show} of {len(cluster_sizes)} clusters)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, (bar, val) in enumerate(zip(bars, top_clusters.values)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100, 
                f'{int(val)}', ha='center', va='bottom', fontsize=9)
    
    viz_path_3 = os.path.join(CONFIG['BASE_DIR'], 'viz_03_cluster_sizes.png')
    plt.tight_layout()
    plt.savefig(viz_path_3, dpi=300, bbox_inches='tight')
    print(f"[+] Saved: {viz_path_3}")
    plt.close()
    
    # ========================================================================
    # ADDITIONAL ANALYSIS: Cluster statistics
    # ========================================================================
    print("\n" + "="*80)
    print("CLUSTER STATISTICS")
    print("="*80)
    print(f"\nTotal clusters: {len(cluster_sizes)}")
    print(f"Median cluster size: {cluster_sizes.median():.0f} points")
    print(f"Mean cluster size: {cluster_sizes.mean():.0f} points")
    print(f"\nTop 10 clusters:")
    for i, (cluster_id, size) in enumerate(cluster_sizes.head(10).items(), 1):
        pct = 100 * size / len(df_sample)
        print(f"  {i:2d}. Cluster {cluster_id}: {size:6,} points ({pct:5.2f}%)")
    
    print("\nBottom 10 clusters (smallest):")
    for i, (cluster_id, size) in enumerate(cluster_sizes.tail(10).items(), 1):
        pct = 100 * size / len(df_sample)
        print(f"  {i:2d}. Cluster {cluster_id}: {size:6,} points ({pct:5.2f}%)")


    # ========================================================================
    # ANALYSIS: TOP 8 CLUSTERS (DETAILED)
    # ========================================================================
    print("\n" + "="*80)
    print("DETAILED ANALYSIS: TOP 8 CLUSTERS")
    print("="*80)
    
    n_top = 8
    cluster_sizes = df_sample['final_cluster'].value_counts()
    top_clusters = cluster_sizes.head(n_top).index.tolist()
    
    # Create analysis dataframe
    cluster_analysis = []
    
    for rank, cluster_id in enumerate(top_clusters, 1):
        cluster_data = df_sample[df_sample['final_cluster'] == cluster_id]
        
        # Geographic characteristics
        center_lat = cluster_data['pickup_latitude'].mean()
        center_lon = cluster_data['pickup_longitude'].mean()
        lat_std = cluster_data['pickup_latitude'].std()
        lon_std = cluster_data['pickup_longitude'].std()
        
        # Temporal characteristics
        peak_hour = cluster_data['pickup_hour'].mode()[0] if len(cluster_data['pickup_hour'].mode()) > 0 else cluster_data['pickup_hour'].mean()
        peak_weekday = cluster_data['pickup_weekday'].mode()[0] if len(cluster_data['pickup_weekday'].mode()) > 0 else cluster_data['pickup_weekday'].mean()
        
        # Weekday vs Weekend split
        weekday_pct = (cluster_data['pickup_weekday'] < 5).mean() * 100
        
        # Trip characteristics
        avg_distance = cluster_data['trip_distance'].mean()
        avg_duration = cluster_data['trip_duration_min'].mean()
        avg_fare = cluster_data['fare_amount'].mean()
        
        size = len(cluster_data)
        pct = 100 * size / len(df_sample)
        
        cluster_analysis.append({
            'Rank': rank,
            'Cluster_ID': cluster_id,
            'Size': size,
            'Pct': pct,
            'Center_Lat': center_lat,
            'Center_Lon': center_lon,
            'Spatial_Spread_km': np.sqrt(lat_std**2 + lon_std**2) * 111,
            'Peak_Hour': int(peak_hour),
            'Peak_Weekday': int(peak_weekday),
            'Weekday_Pct': weekday_pct,
            'Avg_Distance_km': avg_distance,
            'Avg_Duration_min': avg_duration,
            'Avg_Fare': avg_fare,
        })
    
    analysis_df = pd.DataFrame(cluster_analysis)
    
    # Print summary table
    print("\n" + "-"*140)
    print("CLUSTER SUMMARY TABLE")
    print("-"*140)
    
    print(f"{'Rank':>4} | {'Cluster_ID':>20} | {'Size':>8} | {'Pct':>6} | {'Lat':>9} | {'Lon':>10} | "
          f"{'Spread_km':>10} | {'Peak_Hr':>8} | {'Day':>10} | {'WD%':>6}")
    print("-"*140)
    
    for idx, row in analysis_df.iterrows():
        day_name = 'Mon-Fri' if row['Peak_Weekday'] < 5 else 'Sat-Sun'
        print(f"{row['Rank']:4d} | {str(row['Cluster_ID']):>20} | {row['Size']:>8,d} | {row['Pct']:>6.2f} | "
              f"{row['Center_Lat']:>9.4f} | {row['Center_Lon']:>10.4f} | {row['Spatial_Spread_km']:>10.2f} | "
              f"{row['Peak_Hour']:>8} | {day_name:>10} | {row['Weekday_Pct']:>6.1f}")
    
    # Print detailed descriptions
    print("\n" + "-"*140)
    print("DETAILED DESCRIPTIONS")
    print("-"*140)
    
    weekday_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 
                     4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    
    for idx, row in analysis_df.iterrows():
        cluster_id = row['Cluster_ID']
        cluster_data = df_sample[df_sample['final_cluster'] == cluster_id]
        
        peak_day_name = weekday_names[int(row['Peak_Weekday'])]
        
        print(f"\n[CLUSTER {row['Rank']}] ID: {cluster_id} | {row['Size']:,} points ({row['Pct']:.2f}%)")
        print(f"  Location: Center at ({row['Center_Lat']:.4f}, {row['Center_Lon']:.4f}) | "
              f"Spread: ~{row['Spatial_Spread_km']:.1f} km")
        
        # Identify likely location based on coordinates
        lat, lon = row['Center_Lat'], row['Center_Lon']
        if 40.7 <= lat <= 40.8 and -74.0 <= lon <= -73.9:
            location_hint = "Manhattan (core)"
        elif 40.6 <= lat <= 40.7 and -74.0 <= lon <= -73.8:
            location_hint = "Lower Manhattan / Financial District"
        elif lat > 40.75:
            location_hint = "Upper Manhattan / Harlem"
        elif lat < 40.64:
            location_hint = "Outer boroughs / Airports"
        elif lon < -73.95:
            location_hint = "Queens / Eastern outer boroughs"
        else:
            location_hint = "Brooklyn / Downtown Brooklyn"
        
        print(f"  Likely area: {location_hint}")
        print(f"  Temporal: Peak at {int(row['Peak_Hour']):02d}:00 on {peak_day_name} | "
              f"{row['Weekday_Pct']:.1f}% weekday traffic")
        print(f"  Trip profile: Avg {row['Avg_Distance_km']:.1f} km, {row['Avg_Duration_min']:.0f} min, "
              f"${row['Avg_Fare']:.2f} fare")
        
        # Rush hour analysis
        morning_rush = (cluster_data['pickup_hour'].isin([7,8,9])).mean() * 100
        evening_rush = (cluster_data['pickup_hour'].isin([17,18,19])).mean() * 100
        night = (cluster_data['pickup_hour'].isin([22,23,0,1,2,3,4,5])).mean() * 100
        
        print(f"  Pattern: {morning_rush:.1f}% morning rush, {evening_rush:.1f}% evening rush, {night:.1f}% night")
    
    # ========================================================================
    # VISUALIZATION 4: TOP 8 CLUSTERS - GEOGRAPHIC MAPS
    # ========================================================================
    print(f"\n[*] Creating individual geographic maps for top {n_top} clusters...")
    
    n_rows = (n_top + 3) // 4
    n_cols = min(4, n_top)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    axes = axes.flatten()
    
    for idx, cluster_id in enumerate(top_clusters):
        cluster_data = df_sample[df_sample['final_cluster'] == cluster_id]
        
        ax = axes[idx]
        
        # Plot cluster points
        ax.scatter(cluster_data['pickup_longitude'], 
                  cluster_data['pickup_latitude'],
                  c='#1f77b4', s=10, alpha=0.6, label='Pickups', edgecolors='none')
        
        # Plot center
        center_lat = cluster_data['pickup_latitude'].mean()
        center_lon = cluster_data['pickup_longitude'].mean()
        ax.scatter(center_lon, center_lat, c='red', s=300, marker='*', 
                  edgecolors='darkred', linewidth=2, label='Cluster Center', zorder=5)
        
        # Draw bounding box (±1.5 sigma)
        lat_std = cluster_data['pickup_latitude'].std()
        lon_std = cluster_data['pickup_longitude'].std()
        rect = plt.Rectangle(
            (center_lon - 1.5*lon_std, center_lat - 1.5*lat_std),
            3*lon_std, 3*lat_std,
            fill=False, edgecolor='red', linewidth=2, linestyle='--', alpha=0.7, label='±1.5σ bounds'
        )
        ax.add_patch(rect)
        
        ax.set_xlabel('Longitude', fontsize=10)
        ax.set_ylabel('Latitude', fontsize=10)
        
        size = len(cluster_data)
        pct = 100 * size / len(df_sample)
        peak_hour = int(cluster_data['pickup_hour'].mode()[0]) if len(cluster_data['pickup_hour'].mode()) > 0 else int(cluster_data['pickup_hour'].mean())
        
        ax.set_title(f'Cluster {idx+1}: {cluster_id}\n{size:,} points ({pct:.2f}%) | Peak: {peak_hour:02d}:00',
                    fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='best')
    
    # Hide extra subplots
    for idx in range(n_top, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    viz_path_4 = os.path.join(CONFIG['BASE_DIR'], 'viz_04_top_clusters_geographic.png')
    plt.savefig(viz_path_4, dpi=300, bbox_inches='tight')
    print(f"[+] Saved: {viz_path_4}")
    plt.close()
    
    # ========================================================================
    # VISUALIZATION 5: TOP 8 CLUSTERS - TEMPORAL HEATMAPS
    # ========================================================================
    print(f"[*] Creating temporal heatmaps for top {n_top} clusters...")
    
    n_rows = (n_top + 3) // 4
    n_cols = min(4, n_top)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    axes = axes.flatten()
    
    for idx, cluster_id in enumerate(top_clusters):
        cluster_data = df_sample[df_sample['final_cluster'] == cluster_id]
        
        # Create hour × weekday heatmap
        temporal_pivot = cluster_data.pivot_table(
            values='final_cluster',
            index='pickup_hour',
            columns='pickup_weekday',
            aggfunc='count'
        )
        
        ax = axes[idx]
        im = ax.imshow(temporal_pivot.fillna(0), cmap='YlOrRd', aspect='auto')
        
        ax.set_xlabel('Day of Week', fontsize=9)
        ax.set_ylabel('Hour of Day', fontsize=9)
        ax.set_xticks(range(7))
        ax.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], fontsize=8)
        ax.set_yticks(range(0, 24, 4))
        ax.set_yticklabels(range(0, 24, 4), fontsize=8)
        
        size = len(cluster_data)
        pct = 100 * size / len(df_sample)
        ax.set_title(f'Cluster {idx+1}: {cluster_id}\n{size:,} pickups ({pct:.2f}%)',
                    fontsize=11, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax, label='Count')
    
    # Hide extra subplots
    for idx in range(n_top, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    viz_path_5 = os.path.join(CONFIG['BASE_DIR'], 'viz_05_top_clusters_temporal.png')
    plt.savefig(viz_path_5, dpi=300, bbox_inches='tight')
    print(f"[+] Saved: {viz_path_5}")
    plt.close()

    print("\n" + "="*80)
    print("CLUSTER SPATIAL ANALYSIS")
    print("="*80)

    cluster_sizes = df_sample['final_cluster'].value_counts()
    for cluster_id in cluster_sizes.head(8).index:
        cluster_data = df_sample[df_sample['final_cluster'] == cluster_id]
        
        lat = cluster_data['pickup_latitude'].values
        lon = cluster_data['pickup_longitude'].values
        
        # Haversine or simple approximation
        lat_range_km = (lat.max() - lat.min()) * 111
        lon_range_km = (lon.max() - lon.min()) * 85
        
        print(f"\nCluster {cluster_id}:")
        print(f"  Lat range: {lat_range_km:.2f} km")
        print(f"  Lon range: {lon_range_km:.2f} km")
        print(f"  Diagonal: ~{np.sqrt(lat_range_km**2 + lon_range_km**2):.2f} km")
        print(f"  Center: ({lat.mean():.4f}, {lon.mean():.4f})")
        
    print("\n[+] All visualizations complete!")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import sys
    
    # Get command line argument
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else 'all'
    
    # 1. Training only
    if mode in ['all', 'train']:
        df_sample, model = train_clustering_model(use_checkpoint=True)
        visualize_sample_results()
    
    # 2. Inference only
    if mode in ['all', 'infer_knn']:
        df_full = apply_to_full_dataset_knn(use_checkpoint=True)
        print(f"\nSaving to: {CONFIG['OUTPUT_FILE']}")
        df_full.to_parquet(CONFIG['OUTPUT_FILE'], index=False)
        print(f"[+] Success!")
    
    # 3. VISUALIZATION ONLY
    if mode == 'viz':
        print("\n" + "="*80)
        print("[*] VISUALIZATION ONLY MODE")
        print("="*80)
        visualize_sample_results()
        print("[+] Visualization complete!")
