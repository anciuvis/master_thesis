# ========================================================================
# HIERARCHICAL SPATIOTEMPORAL CLUSTERING FOR NYC TAXI DATA
# ENHANCED: Visualizations + Statistics Export
# OPTIMIZED: Fine-grained spatial clustering with HDBSCAN
# ========================================================================


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
import seaborn as sns
import geopandas as gpd
import contextily as ctx
import time
import sys


from sklearn.cluster import KMeans
# CHANGED: Replaced DBSCAN with HDBSCAN for density-adaptive clustering
from hdbscan import HDBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score


warnings.filterwarnings('ignore')
np.random.seed(42)


# Set matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10


# ========================================================================
# CONFIGURATION
# ========================================================================


CONFIG = {
    'BASE_DIR': 'C:/Users/Anya/master_thesis/output',
    'CHECKPOINTS_DIR': 'C:/Users/Anya/master_thesis/output/checkpoints_hierarch',
    'INPUT_FILE': 'C:/Users/Anya/master_thesis/output/taxi_data_cleaned_full.parquet',
    'SAMPLE_SIZE': 1_500_000,
    'PLOTS_DIR': 'C:/Users/Anya/master_thesis/output/plots_clustering',
    'STATS_DIR': 'C:/Users/Anya/master_thesis/output/clustering_stats',
    'OUTPUT_DIR': 'C:/Users/Anya/master_thesis/output/plots_clustering',
    
    # HDBSCAN parameters
    # HDBSCAN is density-adaptive and prevents large bridging clusters
    'HDBSCAN_MIN_CLUSTER_SIZE': 20,      # Minimum points per cluster
    'HDBSCAN_MIN_SAMPLES': 10,
    'HDBSCAN_CLUSTER_SELECTION_EPSILON': 0.15,  # Controls cluster separation (km)
    
}


# Create directories
Path(CONFIG['CHECKPOINTS_DIR']).mkdir(parents=True, exist_ok=True)
Path(CONFIG['PLOTS_DIR']).mkdir(parents=True, exist_ok=True)
Path(CONFIG['STATS_DIR']).mkdir(parents=True, exist_ok=True)


# ========================================================================
# PROGRESS TRACKING
# ========================================================================


class ProgressTracker:
    def __init__(self, task_name):
        self.start_time = None
        self.step_times = {}
    
    def start(self, step_name):
        self.start_time = time.time()
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] STARTING: {step_name}")
        sys.stdout.flush()
    
    def update(self, message, percentage=None):
        if self.start_time is None:
            self.start_time = time.time()
        elapsed = time.time() - self.start_time
        if percentage is not None:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {percentage:3d}% | {message} | {elapsed:.1f}s", end='\r')
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {message} | {elapsed:.1f}s")
        sys.stdout.flush()
    
    def end(self, step_name):
        elapsed = time.time() - self.start_time
        self.step_times[step_name] = elapsed
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] COMPLETE: {step_name} ({elapsed:.1f}s)\n")
        sys.stdout.flush()


progress = ProgressTracker('main script')


# ========================================================================
# CUSTOM JSON ENCODER
# ========================================================================


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ========================================================================
# CHECKPOINT MANAGEMENT
# ========================================================================


class CheckpointManager:
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def save_checkpoint(self, name, data, data_type='pkl'):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if data_type == 'pkl':
            path = self.checkpoint_dir / f"{name}_{timestamp}.pkl"
            with open(path, 'wb') as f:
                pickle.dump(data, f)
        elif data_type == 'joblib':
            path = self.checkpoint_dir / f"{name}_{timestamp}.joblib"
            joblib.dump(data, path)
        elif data_type == 'json':
            path = self.checkpoint_dir / f"{name}_{timestamp}.json"
            with open(path, 'w') as f:
                json.dump(data, f, indent=4, cls=NumpyEncoder)
        elif data_type == 'parquet':
            path = self.checkpoint_dir / f"{name}_{timestamp}.parquet"
            data.to_parquet(path, index=False)
        print(f"[CHECKPOINT] Saved: {path}")
        return path
    
    def load_latest_checkpoint(self, name, data_type='pkl'):
        pattern = f"{name}_*.{data_type}"
        checkpoints = sorted(self.checkpoint_dir.glob(pattern))
        if not checkpoints:
            return None
        latest = checkpoints[-1]
        if data_type == 'pkl':
            with open(latest, 'rb') as f:
                return pickle.load(f)
        elif data_type == 'joblib':
            return joblib.load(latest)
        elif data_type == 'json':
            with open(latest, 'r') as f:
                return json.load(f)
        elif data_type == 'parquet':
            return pd.read_parquet(latest)
    def checkpoint_exists(self, name):
        """Check if a checkpoint exists"""
        patterns = [f"{name}_*.pkl", f"{name}_*.joblib", f"{name}_*.json", f"{name}_*.parquet"]
        for pattern in patterns:
            if list(self.checkpoint_dir.glob(pattern)):
                return True
        return False

    def load_checkpoint(self, name, data_type):
        """Load the latest checkpoint of given type"""
        pattern = f"{name}_*.{data_type}"
        checkpoints = sorted(self.checkpoint_dir.glob(pattern))
        if not checkpoints:
            return None
        latest = checkpoints[-1]
        
        if data_type == 'pkl':
            with open(latest, 'rb') as f:
                return pickle.load(f)
        elif data_type == 'joblib':
            return joblib.load(latest)
        elif data_type == 'json':
            with open(latest, 'r') as f:
                return json.load(f)
        elif data_type == 'parquet':
            return pd.read_parquet(latest)

checkpoint_mgr = CheckpointManager(CONFIG['CHECKPOINTS_DIR'])


# ========================================================================
# DATA LOADING
# ========================================================================


def load_taxi_data(sample_size=None, use_checkpoint=True):
    """Load taxi data from file or checkpoint"""
    progress.start("DATA LOADING")
    
    if use_checkpoint:
        df_sample = checkpoint_mgr.load_latest_checkpoint('df_sample', 'parquet')
        if df_sample is not None:
            progress.update(f"Loaded sample from checkpoint: {len(df_sample):,} records")
            progress.end("DATA LOADING")
            return df_sample
    
    progress.update("Loading parquet file...")
    data = pd.read_parquet(CONFIG['INPUT_FILE'])
    progress.update(f"Total records loaded: {len(data):,}")
    
    if sample_size is None:
        sample_size = CONFIG['SAMPLE_SIZE']
    
    progress.update(f"Sampling {sample_size:,} records...")
    df_sample = data.sample(n=min(sample_size, len(data)), random_state=42)
    
    progress.update("Saving sample checkpoint...")
    checkpoint_mgr.save_checkpoint('df_sample', df_sample, 'parquet')
    
    progress.end("DATA LOADING")
    return df_sample

def load_and_prepare_data(sample_size=None, use_checkpoint=True):
    """Load and prepare taxi data for clustering"""
    progress.start("DATA PREPARATION")
    
    df = load_taxi_data(sample_size=sample_size, use_checkpoint=use_checkpoint)
    
    progress.update("Checking required columns...")
    required_cols = ['pickup_hour', 'pickup_weekday', 'pickup_longitude', 'pickup_latitude', 
                     'trip_distance', 'trip_duration_min', 'fare_amount']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    progress.update("Cleaning coordinates...")
    df = df[
        (df['pickup_longitude'].notna()) & 
        (df['pickup_latitude'].notna()) &
        (df['pickup_longitude'] != 0) & 
        (df['pickup_latitude'] != 0)
    ]
    
    progress.update("Converting data types...")
    df['pickup_hour'] = df['pickup_hour'].astype(int)
    df['pickup_weekday'] = df['pickup_weekday'].astype(int)
    df['pickup_longitude'] = df['pickup_longitude'].astype(float)
    df['pickup_latitude'] = df['pickup_latitude'].astype(float)
    df['trip_distance'] = df['trip_distance'].astype(float)
    df['trip_duration_min'] = df['trip_duration_min'].astype(float)
    df['fare_amount'] = df['fare_amount'].astype(float)
    
    progress.end("DATA PREPARATION")
    return df


# ========================================================================
# TEMPORAL CLUSTERING
# ========================================================================


def temporal_clustering(df, n_temporal_clusters=6):
    """
    K-Means on temporal features with visualization and statistics
    """
    progress.start(f"TEMPORAL CLUSTERING (K={n_temporal_clusters})")
    
    # Create temporal features
    progress.update("Creating temporal features...")
    temporal_features = pd.DataFrame({
        'hour_sin': np.sin(2 * np.pi * df['pickup_hour'] / 24),
        'hour_cos': np.cos(2 * np.pi * df['pickup_hour'] / 24),
        'weekday_sin': np.sin(2 * np.pi * df['pickup_weekday'] / 7),
        'weekday_cos': np.cos(2 * np.pi * df['pickup_weekday'] / 7),
    })
    
    # K-Means
    progress.update("Fitting K-Means...")
    kmeans_temporal = KMeans(
        n_clusters=n_temporal_clusters, 
        random_state=42, 
        n_init=20,
        max_iter=300,
        verbose=0
    )
    temporal_labels = kmeans_temporal.fit_predict(temporal_features.values)
    
    df = df.copy()
    df['temporal_cluster'] = temporal_labels
    
    # Analyze clusters
    progress.update("Analyzing temporal clusters...")
    temporal_info = {}
    
    for t_id in range(n_temporal_clusters):
        mask = df['temporal_cluster'] == t_id
        cluster_df = df[mask]
        
        size = mask.sum()
        pct = 100 * size / len(df)
        
        peak_hour = cluster_df['pickup_hour'].mode()[0] if len(cluster_df['pickup_hour'].mode()) > 0 else cluster_df['pickup_hour'].mean()
        peak_weekday = cluster_df['pickup_weekday'].mode()[0] if len(cluster_df['pickup_weekday'].mode()) > 0 else cluster_df['pickup_weekday'].mean()
        
        weekday_pct = (cluster_df['pickup_weekday'] < 5).mean() * 100
        morning_rush = (cluster_df['pickup_hour'].isin([7,8,9])).mean() * 100
        midday = (cluster_df['pickup_hour'].isin([11,12,13,14])).mean() * 100
        evening_rush = (cluster_df['pickup_hour'].isin([17,18,19])).mean() * 100
        night = (cluster_df['pickup_hour'].isin([22,23,0,1,2,3,4,5])).mean() * 100
        
        # Pattern classification
        if weekday_pct > 80 and (morning_rush > 20 or evening_rush > 20):
            pattern_name = "Weekday Evening Rush" if evening_rush > morning_rush else "Weekday Morning Rush"
        elif weekday_pct > 80:
            pattern_name = "Weekday Daytime"
        elif weekday_pct < 20:
            pattern_name = "Weekend/Night"
        else:
            pattern_name = "Mixed Pattern"
        
        temporal_info[t_id] = {
            'size': int(size),
            'pct': float(pct),
            'pattern_name': pattern_name,
            'peak_hour': int(peak_hour),
            'peak_weekday': int(peak_weekday),
            'weekday_pct': float(weekday_pct),
            'morning_rush': float(morning_rush),
            'midday': float(midday),
            'evening_rush': float(evening_rush),
            'night': float(night),
        }
        
        print(f"Temporal Cluster {t_id}: {pattern_name} ({size:,} points, {pct:.1f}%)")
    
    # Create visualizations
    progress.update("Creating temporal visualizations...")
    create_temporal_visualizations(df, temporal_info)
    
    # Save statistics
    progress.update("Saving temporal statistics...")
    save_temporal_statistics(temporal_info, n_temporal_clusters)
    
    progress.end(f"TEMPORAL CLUSTERING (K={n_temporal_clusters})")
    return df, kmeans_temporal, temporal_info


# ========================================================================
# SPATIAL CLUSTERING - HDBSCAN (DENSITY-ADAPTIVE)
# ========================================================================

def get_adaptive_hdbscan_params(coords_scaled_clean, t_id):
    """Adjust HDBSCAN params based on SPATIAL DENSITY, not just point count"""
    from scipy.spatial.distance import pdist
    import numpy as np
    
    n_points = len(coords_scaled_clean)
    
    # Sample for density calculation (avoid computing on millions of points)
    sample_size = min(10000, n_points)
    sample_indices = np.random.choice(n_points, sample_size, replace=False)
    coords_sample = coords_scaled_clean[sample_indices]
    
    # Calculate spatial density metrics
    pairwise_distances = pdist(coords_sample, metric='euclidean')
    median_distance = np.median(pairwise_distances)
    q25_distance = np.percentile(pairwise_distances, 25)
    q75_distance = np.percentile(pairwise_distances, 75)
    
    # Density classification
    density_score = 1.0 / (median_distance + 1e-6)  # Higher = denser
    
    print(f"  T{t_id}: n_points={n_points}, median_dist={median_distance:.4f}, density_score={density_score:.4f}")
    
    if density_score > 5.0:  # Very dense (T1, T2, T3 - rush hours)
        return {
            'min_cluster_size': max(50, int(0.0003 * n_points)),   # Very lenient
            'min_samples': 10,
            'cluster_selection_epsilon': 0.020,  # Large epsilon catches tight clusters
        }, "VERY_DENSE"
    
    elif density_score > 2.0:  # Dense (normal rush)
        return {
            'min_cluster_size': max(100, int(0.0005 * n_points)),
            'min_samples': 15,
            'cluster_selection_epsilon': 0.010,
        }, "DENSE"
    
    elif density_score > 1.0:  # Normal (T0, T5 - moderate)
        return {
            'min_cluster_size': max(150, int(0.001 * n_points)),
            'min_samples': 25,
            'cluster_selection_epsilon': 0.008,
        }, "NORMAL"
    
    else:  # Sparse (T4 - night/weekend)
        return {
            'min_cluster_size': max(50, int(0.0002 * n_points)),
            'min_samples': 8,
            'cluster_selection_epsilon': 0.015,
        }, "SPARSE"

def spatial_clustering_within_temporal(df):
    """
    HYBRID APPROACH: HDBSCAN (find natural clusters) + KMeans (refine)
    
    This combines:
    1. HDBSCAN: Discovers natural density-based hotspots
    2. KMeans: Uses HDBSCAN centroids as init to refine and filter noise
    
    ADVANTAGES:
    + Finds ALL natural clusters HDBSCAN discovers
    + Filters noise points effectively
    + Much finer granularity than HDBSCAN alone
    + Better geographic coverage with small zones
    
    PARAMETERS TUNED FOR NYC TAXI DATA:
    - HDBSCAN min_cluster_size=400: Conservative, finds major hotspots
    - HDBSCAN min_samples: Reasonable density threshold
    - HDBSCAN cluster_selection_epsilon: Scaled coordinates unit
    - StandardScaler: Normalize coordinates to [-1, 1] range for HDBSCAN
    """
    progress.start("SPATIAL CLUSTERING (HDBSCAN + KMeans Hybrid)")
    
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import silhouette_score
    
    progress.update("Preparing coordinates...")
    
    df = df.copy()
    df['hierarchical_cluster'] = '-1'
    
    spatial_info = {}
    
    n_temporal = len(df['temporal_cluster'].unique())
    
    print("\n" + "="*100)
    print("SPATIAL CLUSTERING WITH HYBRID HDBSCAN + KMEANS (FIND 377+ NATURAL CLUSTERS)")
    print("="*100)
    print(f"Approach: HDBSCAN discovers natural hotspots, KMeans refines with those as init\n")
    
    manhattan_center_lat = 40.7128
    manhattan_center_lon = -74.0060
    
    for idx, t_id in enumerate(sorted(df['temporal_cluster'].unique()), 1):
        progress.update(f"Processing temporal cluster {idx}/{n_temporal}...", percentage=int(50 + idx*5))
        
        temporal_mask = df['temporal_cluster'] == t_id
        temporal_subset = df[temporal_mask]
        
        # Prepare coordinates
        clustering_coords = temporal_subset[['pickup_longitude', 'pickup_latitude']].copy()
        clustering_coords = clustering_coords.dropna()
        
        if len(clustering_coords) < 400:
            print(f"\nTemporal Cluster {t_id}: Too few points ({len(clustering_coords)}), skipping spatial clustering")
            continue
        
        # Step 1: STANDARDIZE coordinates for HDBSCAN
        scaler = StandardScaler()
        coords_scaled = scaler.fit_transform(clustering_coords)
        
        # Step 2: REMOVE OUTLIERS (top 1% distance from center)
        dist_pickup = np.sqrt(
            (clustering_coords['pickup_longitude'].values - manhattan_center_lon)**2 + 
            (clustering_coords['pickup_latitude'].values - manhattan_center_lat)**2
        )
        q99 = np.percentile(dist_pickup, 99)
        outlier_mask = dist_pickup < q99
        coords_scaled_clean = coords_scaled[outlier_mask]
        #clustering_coords_clean = clustering_coords[outlier_mask]
        
        # Step 3: HDBSCAN to find natural density-based hotspots
        progress.update(f"  T{t_id}: HDBSCAN finding natural clusters...")
        hdbscan_params, density_mode = get_adaptive_hdbscan_params(coords_scaled_clean, t_id)
        print(f"  T{t_id}: Using {density_mode} parameters")

        hdbscan = HDBSCAN(
            min_cluster_size=hdbscan_params['min_cluster_size'],
            min_samples=hdbscan_params['min_samples'],
            cluster_selection_epsilon=hdbscan_params['cluster_selection_epsilon'],
            allow_single_cluster=False
        )
        hdbscan_labels = hdbscan.fit_predict(coords_scaled_clean)
        
        # Extract HDBSCAN cluster centers as KMeans initial centroids
        valid_mask = hdbscan_labels != -1
        hdbscan_centroids = []
        
        for cluster_id in np.unique(hdbscan_labels[valid_mask]):
            cluster_points = coords_scaled_clean[valid_mask & (hdbscan_labels == cluster_id)]
            centroid = cluster_points.mean(axis=0)
            hdbscan_centroids.append(centroid)
        
        hdbscan_centroids = np.array(hdbscan_centroids)
        n_clusters = len(hdbscan_centroids)
        
        # Step 4: KMeans with HDBSCAN centroids as init (refines and filters noise)
        progress.update(f"  T{t_id}: KMeans refining {n_clusters} HDBSCAN clusters...")
        kmeans_hybrid = MiniBatchKMeans(
            n_clusters=n_clusters,
            init=hdbscan_centroids,
            n_init=1,
            max_iter=500,
            random_state=42,
            batch_size=1000
        )
        hybrid_labels = kmeans_hybrid.fit_predict(coords_scaled_clean)
        
        print(f"\nTemporal Cluster {t_id}: HDBSCAN found {n_clusters} natural hotspots")
        print(f"  After outlier removal: {len(coords_scaled_clean):,} points")
        
        # Map back to original dataframe
        # Create mapping for clean indices
        clean_idx = clustering_coords.index[outlier_mask]
        
        for s_id in np.unique(hybrid_labels):
            mask_cluster = temporal_mask & df.index.isin(clean_idx[hybrid_labels == s_id])
            
            if mask_cluster.sum() > 0:
                cluster_name = f"T{t_id}_S{s_id}"
                df.loc[mask_cluster, 'hierarchical_cluster'] = cluster_name
                
                cluster_data = df[mask_cluster]
                
                # Detailed statistics
                spatial_info[cluster_name] = {
                    'temporal_id': int(t_id),
                    'spatial_id': int(s_id),
                    'size': int(mask_cluster.sum()),
                    'pct': float(100 * mask_cluster.sum() / len(df)),
                    'center_lat': float(cluster_data['pickup_latitude'].mean()),
                    'center_lon': float(cluster_data['pickup_longitude'].mean()),
                    'spread_km': float(np.sqrt(
                        ((cluster_data['pickup_latitude'] - cluster_data['pickup_latitude'].mean()) * 111)**2 +
                        ((cluster_data['pickup_longitude'] - cluster_data['pickup_longitude'].mean()) * 85)**2
                    ).std() or 0.1),
                    'peak_hour': int(cluster_data['pickup_hour'].mode()[0]) if len(cluster_data['pickup_hour'].mode()) > 0 else int(cluster_data['pickup_hour'].mean()),
                    'avg_distance_km': float(cluster_data['trip_distance'].mean()),
                    'avg_duration_min': float(cluster_data['trip_duration_min'].mean()),
                    'avg_fare': float(cluster_data['fare_amount'].mean()),
                    'lat_min': float(cluster_data['pickup_latitude'].min()),
                    'lat_max': float(cluster_data['pickup_latitude'].max()),
                    'lon_min': float(cluster_data['pickup_longitude'].min()),
                    'lon_max': float(cluster_data['pickup_longitude'].max()),
                }
    
    # Create visualizations
    progress.update("Creating spatial visualizations...")
    create_spatial_visualizations(df, spatial_info)
    
    # Save statistics
    progress.update("Saving spatial statistics...")
    save_spatial_statistics(spatial_info)
    
    progress.end("SPATIAL CLUSTERING (HDBSCAN + KMeans Hybrid)")
    return df, spatial_info


# ========================================================================
# VISUALIZATION FUNCTIONS
# ========================================================================


def create_temporal_visualizations(df, temporal_info):
    """Create comprehensive temporal cluster visualizations"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax = axes[0, 0]
    for t_id in sorted(df['temporal_cluster'].unique()):
        hours = df[df['temporal_cluster'] == t_id]['pickup_hour']
        ax.hist(hours, bins=24, alpha=0.5, label=f"T{t_id}: {temporal_info[t_id]['pattern_name'][:15]}")
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Count')
    ax.set_title('Hour Distribution by Temporal Cluster')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    
    ax = axes[0, 1]
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    for t_id in sorted(df['temporal_cluster'].unique()):
        day_counts = df[df['temporal_cluster'] == t_id]['pickup_weekday'].value_counts().sort_index()
        ax.plot(day_counts.index, day_counts.values, marker='o', label=f"T{t_id}")
    ax.set_xticks(range(7))
    ax.set_xticklabels(days)
    ax.set_xlabel('Day of Week')
    ax.set_ylabel('Count')
    ax.set_title('Weekday Distribution by Temporal Cluster')
    ax.legend()
    ax.grid(alpha=0.3)
    
    ax = axes[1, 0]
    sizes = [temporal_info[t_id]['size'] for t_id in sorted(temporal_info.keys())]
    colors = plt.cm.viridis(np.linspace(0, 1, len(sizes)))
    bars = ax.bar(range(len(sizes)), sizes, color=colors)
    ax.set_xlabel('Temporal Cluster')
    ax.set_ylabel('Number of Trips')
    ax.set_title('Temporal Cluster Sizes')
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([f"T{i}" for i in range(len(sizes))])
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height/1000)}k', ha='center', va='bottom', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    ax = axes[1, 1]
    patterns_data = {
        'Morning Rush': [temporal_info[t_id]['morning_rush'] for t_id in sorted(temporal_info.keys())],
        'Midday': [temporal_info[t_id]['midday'] for t_id in sorted(temporal_info.keys())],
        'Evening Rush': [temporal_info[t_id]['evening_rush'] for t_id in sorted(temporal_info.keys())],
        'Night': [temporal_info[t_id]['night'] for t_id in sorted(temporal_info.keys())],
    }
    x = np.arange(len(temporal_info))
    width = 0.2
    for i, (label, values) in enumerate(patterns_data.items()):
        ax.bar(x + i*width, values, width, label=label)
    ax.set_xlabel('Temporal Cluster')
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Time-of-Day Composition by Cluster')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f"T{i}" for i in range(len(temporal_info))])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['PLOTS_DIR'], 'temporal_clusters_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: temporal_clusters_analysis.png")


def create_spatial_visualizations(df, spatial_info):
    """Create comprehensive spatial cluster visualizations"""
    
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: Geographic map of clusters (FINE-GRAINED with HDBSCAN)
    ax1 = plt.subplot(2, 2, 1)
    
    unique_clusters = df['hierarchical_cluster'].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))
    color_map = {cluster: colors[i % len(colors)] for i, cluster in enumerate(unique_clusters)}
    
    for cluster in sorted(df['hierarchical_cluster'].unique()):
        if cluster != '-1':
            mask = df['hierarchical_cluster'] == cluster
            ax1.scatter(df[mask]['pickup_longitude'], df[mask]['pickup_latitude'], 
                       s=5, alpha=0.7, color=color_map[cluster])
    
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_title(f'Geographic Distribution (Fine-Grained Spatial Clusters - HDBSCAN)\nTotal clusters: {len(spatial_info)}')
    ax1.grid(alpha=0.3)
    
    # Plot 2: Top clusters by size
    ax2 = plt.subplot(2, 2, 2)
    sorted_clusters = sorted(spatial_info.items(), key=lambda x: x[1]['size'], reverse=True)[:20]
    cluster_names = [name for name, _ in sorted_clusters]
    sizes = [info['size'] for _, info in sorted_clusters]
    colors_bar = plt.cm.viridis(np.linspace(0, 1, len(sizes)))
    bars = ax2.barh(cluster_names, sizes, color=colors_bar)
    ax2.set_xlabel('Number of Trips')
    ax2.set_title(f'Top 20 Clusters by Size (out of {len(spatial_info)} total)')
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax2.text(width, bar.get_y() + bar.get_height()/2.,
                f' {int(width/1000)}k', ha='left', va='center', fontsize=8)
    ax2.grid(axis='x', alpha=0.3)
    
    # Plot 3: Cluster density and spread
    ax3 = plt.subplot(2, 2, 3)
    spreads = [info['spread_km'] for info in spatial_info.values()]
    sizes_all = [info['size'] for info in spatial_info.values()]
    scatter = ax3.scatter(sizes_all, spreads, s=100, alpha=0.6, c=range(len(sizes_all)), cmap='viridis')
    ax3.set_xlabel('Cluster Size (trips)')
    ax3.set_ylabel('Geographic Spread (km)')
    ax3.set_title(f'Cluster Characteristics ({len(spatial_info)} clusters - HDBSCAN)')
    ax3.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax3, label='Cluster ID')
    
    # Plot 4: Trip characteristics
    ax4 = plt.subplot(2, 2, 4)
    sorted_clusters_char = sorted(spatial_info.items(), key=lambda x: x[1]['size'], reverse=True)[:15]
    cluster_names_char = [name for name, _ in sorted_clusters_char]
    distances = [info['avg_distance_km'] for _, info in sorted_clusters_char]
    fares = [info['avg_fare'] for _, info in sorted_clusters_char]
    
    x = np.arange(len(cluster_names_char))
    width = 0.35
    ax4_twin = ax4.twinx()
    
    bars1 = ax4.bar(x - width/2, distances, width, label='Avg Distance (km)', color='steelblue')
    bars2 = ax4_twin.bar(x + width/2, fares, width, label='Avg Fare ($)', color='coral')
    
    ax4.set_xlabel('Cluster')
    ax4.set_ylabel('Distance (km)', color='steelblue')
    ax4_twin.set_ylabel('Fare ($)', color='coral')
    ax4.set_title('Trip Characteristics (Top 15 Clusters)')
    ax4.set_xticks(x)
    ax4.set_xticklabels(cluster_names_char, rotation=45, ha='right', fontsize=8)
    ax4.tick_params(axis='y', labelcolor='steelblue')
    ax4_twin.tick_params(axis='y', labelcolor='coral')
    ax4.grid(axis='y', alpha=0.3)
    
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['PLOTS_DIR'], 'spatial_clusters_analysis_finegrained.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: spatial_clusters_analysis_finegrained.png")

def create_temporal_geographic_visualizations(df, spatial_info):
    """
    Create 6 separate geographic distribution plots (one per temporal cluster)
    showing spatial clusters colored by temporal pattern with OpenStreetMap background
    """
    progress = ProgressTracker("Temporal Geographic Visualizations")
    progress.update("Creating temporal-specific geographic visualizations...")
    
    temporal_clusters = sorted(df['temporal_cluster'].unique())
    
    # Create one plot per temporal cluster
    for t_id in temporal_clusters:
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Get data for this temporal cluster
        temporal_data = df[df['temporal_cluster'] == t_id].copy()
        
        # Convert to GeoDataFrame for contextily
        gdf = gpd.GeoDataFrame(
            temporal_data,
            geometry=gpd.points_from_xy(temporal_data['pickup_longitude'], temporal_data['pickup_latitude']),
            crs="EPSG:4326"
        )
        
        # Project to Web Mercator for contextily compatibility
        gdf = gdf.to_crs(epsg=3857)
        
        # Get all spatial clusters for this temporal pattern
        spatial_clusters_temporal = temporal_data['hierarchical_cluster'].unique()
        
        ax.scatter = gdf.plot(
            ax=ax,
            column='hierarchical_cluster',
            markersize=2,
            cmap='tab20',
            alpha=0.6
        )

        # Use OpenStreetMap
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
        
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        
        # Get temporal pattern name
        n_clusters_t = len(spatial_clusters_temporal) - 1  # Exclude noise
        n_trips_t = len(temporal_data)
        
        ax.set_title(
            f'Geographic Distribution - Temporal Cluster {t_id}\n'
            f'({n_clusters_t} spatial zones, {n_trips_t:,} trips)',
            fontsize=14,
            fontweight='bold'
        )
        
        ax.grid(alpha=0.3)
        
        # Save figure
        plot_filename = os.path.join(
            CONFIG['OUTPUT_DIR'],
            f'geographic_distribution_temporal_T{t_id}.png'
        )
        plt.tight_layout()
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    ✓ Saved: geographic_distribution_temporal_T{t_id}.png")
    
    progress.update("Temporal geographic visualizations complete")
    progress.end("Temporal Geographic Visualizations")


# ========================================================================
# STATISTICS EXPORT FUNCTIONS
# ========================================================================


def save_temporal_statistics(temporal_info, n_temporal_clusters):
    """Save temporal cluster statistics"""
    progress.update("Exporting temporal statistics...")
    
    df_temporal = pd.DataFrame(temporal_info).T
    df_temporal.index.name = 'temporal_cluster_id'
    
    csv_path = os.path.join(CONFIG['STATS_DIR'], 'temporal_clusters_statistics.csv')
    df_temporal.to_csv(csv_path)
    print(f"Saved: temporal_clusters_statistics.csv")
    
    json_path = os.path.join(CONFIG['STATS_DIR'], 'temporal_clusters_statistics.json')
    with open(json_path, 'w') as f:
        json.dump(temporal_info, f, indent=4, cls=NumpyEncoder)
    print(f"Saved: temporal_clusters_statistics.json")
    
    report_path = os.path.join(CONFIG['STATS_DIR'], 'temporal_clustering_report.txt')
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("TEMPORAL CLUSTERING REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Number of temporal clusters: {n_temporal_clusters}\n")
        f.write(f"Total trips analyzed: {sum(info['size'] for info in temporal_info.values()):,}\n\n")
        
        f.write("CLUSTER SUMMARIES:\n")
        f.write("-"*80 + "\n")
        
        for t_id in sorted(temporal_info.keys()):
            info = temporal_info[t_id]
            f.write(f"\nTemporal Cluster {t_id}: {info['pattern_name']}\n")
            f.write(f"  Size: {info['size']:,} trips ({info['pct']:.2f}%)\n")
            f.write(f"  Peak hour: {info['peak_hour']:02d}:00\n")
            f.write(f"  Weekday percentage: {info['weekday_pct']:.1f}%\n")
            f.write(f"  Time composition:\n")
            f.write(f"    - Morning rush (7-9): {info['morning_rush']:.1f}%\n")
            f.write(f"    - Midday (11-14): {info['midday']:.1f}%\n")
            f.write(f"    - Evening rush (17-19): {info['evening_rush']:.1f}%\n")
            f.write(f"    - Night (22-5): {info['night']:.1f}%\n")
    
    print(f"Saved: temporal_clustering_report.txt")


def save_spatial_statistics(spatial_info):
    """Save spatial cluster statistics with dispatcher-friendly formatting"""
    progress.update("Exporting spatial statistics...")
    
    df_spatial = pd.DataFrame(spatial_info).T
    df_spatial.index.name = 'hierarchical_cluster_id'
    
    csv_path = os.path.join(CONFIG['STATS_DIR'], 'spatial_clusters_statistics.csv')
    df_spatial.to_csv(csv_path)
    print(f"Saved: spatial_clusters_statistics.csv")
    
    json_path = os.path.join(CONFIG['STATS_DIR'], 'spatial_clusters_statistics.json')
    with open(json_path, 'w') as f:
        json.dump(spatial_info, f, indent=4, cls=NumpyEncoder)
    print(f"Saved: spatial_clusters_statistics.json")
    
    report_path = os.path.join(CONFIG['STATS_DIR'], 'spatial_clustering_report.txt')
    with open(report_path, 'w') as f:
        f.write("="*100 + "\n")
        f.write("SPATIAL CLUSTERING REPORT - FINE-GRAINED ZONES (HDBSCAN)\n")
        f.write("="*100 + "\n\n")
        f.write(f"CLUSTERING ALGORITHM: HDBSCAN (Hierarchical Density-Based Spatial Clustering)\n")
        f.write(f"(Replaces DBSCAN - density-adaptive, prevents large bridging clusters)\n\n")
        f.write(f"CLUSTERING PARAMETERS:\n")
        f.write(f"  min_cluster_size: {CONFIG['HDBSCAN_MIN_CLUSTER_SIZE']} points\n")
        f.write(f"  min_samples: {CONFIG['HDBSCAN_MIN_SAMPLES']} points\n")
        f.write(f"  cluster_selection_epsilon: {CONFIG['HDBSCAN_CLUSTER_SELECTION_EPSILON']} km\n")
        f.write(f"  metric: haversine (geographic distance)\n")
        f.write(f"  Granularity: ~{CONFIG['HDBSCAN_CLUSTER_SELECTION_EPSILON']*1000:.0f} meters per zone\n\n")
        f.write(f"RESULTS:\n")
        f.write(f"  Total hierarchical clusters: {len(spatial_info)}\n")
        f.write(f"  Total classified trips: {sum(info['size'] for info in spatial_info.values()):,}\n\n")
        
        f.write("DISPATCHER-FRIENDLY ZONE DESCRIPTIONS (Top 20):\n")
        f.write("-"*100 + "\n")
        f.write(f"{'Zone_ID':>20} | {'Trips':>8} | {'Lat':>9} | {'Lon':>10} | {'Spread':>7} | {'Avg_Dist':>8} | {'Avg_Fare':>8}\n")
        f.write("-"*100 + "\n")
        
        sorted_clusters = sorted(spatial_info.items(), key=lambda x: x[1]['size'], reverse=True)
        for cluster_name, info in sorted_clusters[:20]:
            f.write(f"{cluster_name:>20} | {info['size']:>8,d} | {info['center_lat']:>9.4f} | {info['center_lon']:>10.4f} | {info['spread_km']:>7.2f} | {info['avg_distance_km']:>8.2f} | ${info['avg_fare']:>7.2f}\n")
        
        f.write("\n\nZONE BOUNDARIES (for routing systems):\n")
        f.write("-"*100 + "\n")
        for cluster_name, info in sorted_clusters[:10]:
            f.write(f"\n{cluster_name}:\n")
            f.write(f"  Latitude range: {info['lat_min']:.4f} to {info['lat_max']:.4f}\n")
            f.write(f"  Longitude range: {info['lon_min']:.4f} to {info['lon_max']:.4f}\n")
            f.write(f"  Center: ({info['center_lat']:.4f}, {info['center_lon']:.4f})\n")
            f.write(f"  Spread: {info['spread_km']:.2f} km\n")
            f.write(f"  Trip count: {info['size']:,} ({info['pct']:.2f}%)\n")
    
    print(f"Saved: spatial_clustering_report.txt")


def save_combined_statistics(df, temporal_info, spatial_info):
    """Save combined statistics and summary"""
    progress.update("Exporting combined statistics...")
    
    report_path = os.path.join(CONFIG['STATS_DIR'], 'clustering_summary_finegrained.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("HIERARCHICAL SPATIOTEMPORAL CLUSTERING SUMMARY (HDBSCAN - DENSITY-ADAPTIVE)\n")
        f.write("="*100 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("ALGORITHM UPGRADE:\n")
        f.write("-"*100 + "\n")
        f.write("Previous algorithm: DBSCAN (fixed eps=1.0 km then eps=0.3 km)\n")
        f.write("New algorithm: HDBSCAN (Hierarchical Density-Based Spatial Clustering)\n")
        f.write("Key improvement: Density-adaptive - prevents large bridging clusters\n\n")
        
        f.write("SPATIAL GRANULARITY OPTIMIZATION:\n")
        f.write("-"*100 + "\n")
        f.write("Previous DBSCAN params: eps=1.0 km → sparse clusters, large bridging zones\n")
        f.write("Intermediate DBSCAN params: eps=0.3 km → still had large sparse clusters (e.g., S0 spanning Manhattan)\n")
        f.write("New HDBSCAN params:\n")
        f.write(f"  - min_cluster_size: {CONFIG['HDBSCAN_MIN_CLUSTER_SIZE']}\n")
        f.write(f"  - min_samples: {CONFIG['HDBSCAN_MIN_SAMPLES']}\n")
        f.write(f"  - cluster_selection_epsilon: {CONFIG['HDBSCAN_CLUSTER_SELECTION_EPSILON']} km\n")
        f.write(f"Result: {len(spatial_info)} distinct zones vs previous ~18 zones\n\n")
        
        f.write("OVERVIEW:\n")
        f.write("-"*100 + "\n")
        f.write(f"Total data points: {len(df):,}\n")
        f.write(f"Temporal clusters: {len(temporal_info)}\n")
        f.write(f"Spatial zones (NEW with HDBSCAN): {len(spatial_info)}\n")
        f.write(f"Classified points: {(df['hierarchical_cluster'] != '-1').sum():,} ({100 * (df['hierarchical_cluster'] != '-1').sum() / len(df):.2f}%)\n")
        f.write(f"Noise points: {(df['hierarchical_cluster'] == '-1').sum():,} ({100 * (df['hierarchical_cluster'] == '-1').sum() / len(df):.2f}%)\n\n")
        
        f.write("TEMPORAL PATTERNS:\n")
        f.write("-"*100 + "\n")
        for t_id in sorted(temporal_info.keys()):
            info = temporal_info[t_id]
            f.write(f"T{t_id}: {info['pattern_name']} ({info['size']:,} trips, {info['pct']:.2f}%)\n")
        
        f.write("\nTOP SPATIAL ZONES (Dispatcher-Friendly):\n")
        f.write("-"*100 + "\n")
        sorted_clusters = sorted(spatial_info.items(), key=lambda x: x[1]['size'], reverse=True)
        for i, (cluster_name, info) in enumerate(sorted_clusters[:15], 1):
            f.write(f"{i:2d}. {cluster_name}: {info['size']:,} trips ({info['pct']:.2f}%) - ")
            f.write(f"Center: ({info['center_lat']:.4f}, {info['center_lon']:.4f}), ")
            f.write(f"Dist: {info['avg_distance_km']:.1f}km, Fare: ${info['avg_fare']:.2f}\n")
    
    print(f"Saved: clustering_summary_finegrained.txt")

# ========================================================================
# PARAMETER TESTING & TUNING
# ========================================================================

def test_temporal_parameters(df, n_clusters_range=range(4, 12)):
    """Test different K values for temporal clustering"""
    progress.start("TESTING TEMPORAL CLUSTERING PARAMETERS")
    
    results = []
    
    for n_clusters in n_clusters_range:
        progress.update(f"Testing K={n_clusters}...")
        
        temporal_features = pd.DataFrame({
            'hour_sin': np.sin(2 * np.pi * df['pickup_hour'] / 24),
            'hour_cos': np.cos(2 * np.pi * df['pickup_hour'] / 24),
            'weekday_sin': np.sin(2 * np.pi * df['pickup_weekday'] / 7),
            'weekday_cos': np.cos(2 * np.pi * df['pickup_weekday'] / 7),
        })
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20, max_iter=300)
        labels = kmeans.fit_predict(temporal_features.values)
        
        # Calculate metrics
        silhouette = silhouette_score(temporal_features.values, labels)
        calinski = calinski_harabasz_score(temporal_features.values, labels)
        inertia = kmeans.inertia_
        
        results.append({
            'n_clusters': n_clusters,
            'silhouette': silhouette,
            'calinski_harabasz': calinski,
            'inertia': inertia,
            'wcss': inertia  # Within-cluster sum of squares
        })
        
        print(f"K={n_clusters}: Silhouette={silhouette:.4f}, Calinski-Harabasz={calinski:.2f}")
    
    progress.end("TESTING TEMPORAL CLUSTERING PARAMETERS")
    return pd.DataFrame(results)


def test_spatial_parameters(df, temporal_cluster_id=0, 
                           min_cluster_size_range=[50, 100, 200, 400, 600],
                           min_samples_range=[10, 20, 50]):
    """Test different HDBSCAN parameters for spatial clustering"""
    progress.start(f"TESTING SPATIAL CLUSTERING PARAMETERS (Temporal Cluster {temporal_cluster_id})")
    
    # Get subset
    temporal_mask = df['temporal_cluster'] == temporal_cluster_id
    temporal_subset = df[temporal_mask]
    clustering_coords = temporal_subset[['pickup_longitude', 'pickup_latitude']].copy().dropna()
    
    # Standardize
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(clustering_coords)
    
    # Remove outliers
    manhattan_center_lat, manhattan_center_lon = 40.7128, -74.0060
    dist_pickup = np.sqrt(
        (clustering_coords['pickup_longitude'].values - manhattan_center_lon)**2 + 
        (clustering_coords['pickup_latitude'].values - manhattan_center_lat)**2
    )
    q99 = np.percentile(dist_pickup, 99)
    outlier_mask = dist_pickup < q99
    coords_scaled_clean = coords_scaled[outlier_mask]
    
    results = []
    
    for min_cluster_size in min_cluster_size_range:
        for min_samples in min_samples_range:
            try:
                progress.update(f"Testing min_cluster_size={min_cluster_size}, min_samples={min_samples}...")
                
                hdbscan = HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    cluster_selection_epsilon=0.005,
                    allow_single_cluster=False
                )
                labels = hdbscan.fit_predict(coords_scaled_clean)
                
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)
                cluster_pct = 100 * (len(labels) - n_noise) / len(labels)
                
                results.append({
                    'min_cluster_size': min_cluster_size,
                    'min_samples': min_samples,
                    'n_clusters': n_clusters,
                    'n_noise': n_noise,
                    'cluster_pct': cluster_pct,
                    'params_str': f"size={min_cluster_size}, samples={min_samples}"
                })
                
                print(f"  -> {n_clusters} clusters, {n_noise} noise points ({cluster_pct:.1f}% classified)")
            
            except Exception as e:
                print(f"  ERROR: {e}")
    
    progress.end("TESTING SPATIAL CLUSTERING PARAMETERS")
    return pd.DataFrame(results)


def visualize_parameter_test_results(results_df, metric='silhouette', title='Parameter Test Results'):
    """Visualize parameter testing results"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Metric vs parameter
    ax = axes
    ax.plot(results_df.iloc[:, 0], results_df[metric], marker='o', linewidth=2, markersize=8)
    ax.set_xlabel('Parameter Value')
    ax.set_ylabel(metric.title())
    ax.set_title(f'{metric.title()} vs Parameter')
    ax.grid(alpha=0.3)
    
    # Plot 2: All metrics normalized
    ax = axes
    for col in results_df.columns[1:]:
        normalized = (results_df[col] - results_df[col].min()) / (results_df[col].max() - results_df[col].min())
        ax.plot(results_df.iloc[:, 0], normalized, marker='o', label=col, linewidth=2)
    ax.set_xlabel('Parameter Value')
    ax.set_ylabel('Normalized Score')
    ax.set_title('All Metrics (Normalized)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['PLOTS_DIR'], f'parameter_test_{title}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: parameter_test_{title}.png")


# ========================================================================
# PARAMETER TESTING ENTRY POINT
# ========================================================================

def test_parameters():
    """Test various clustering parameters to find optimal values"""
    print("\n" + "="*100)
    print("PARAMETER TESTING MODE")
    print("="*100 + "\n")
    
    # Load data
    df = load_and_prepare_data()
    
    # Test temporal parameters
    print(" Testing Temporal Clustering Parameters...")
    print("-"*100)
    temporal_results = test_temporal_parameters(df, n_clusters_range=range(3, 11))
    
    # Save results
    temporal_results.to_csv(os.path.join(CONFIG['STATS_DIR'], 'temporal_parameter_test.csv'), index=False)
    print(f"\nTemporal parameter test results:\n{temporal_results}")
    
    # Visualize
    visualize_parameter_test_results(temporal_results, metric='silhouette', title='temporal')
    
    # Test spatial parameters for T0
    print("\n Testing Spatial Clustering Parameters (Temporal Cluster 0)...")
    print("-"*100)
    
    # First do temporal clustering
    temporal_features = pd.DataFrame({
        'hour_sin': np.sin(2 * np.pi * df['pickup_hour'] / 24),
        'hour_cos': np.cos(2 * np.pi * df['pickup_hour'] / 24),
        'weekday_sin': np.sin(2 * np.pi * df['pickup_weekday'] / 7),
        'weekday_cos': np.cos(2 * np.pi * df['pickup_weekday'] / 7),
    })
    kmeans_temporal = KMeans(n_clusters=6, random_state=42, n_init=20, max_iter=300)
    df['temporal_cluster'] = kmeans_temporal.fit_predict(temporal_features.values)
    
    # Test spatial params on T0
    spatial_results = test_spatial_parameters(
        df, 
        temporal_cluster_id=0,
        min_cluster_size_range=[100, 150, 200, 400, 600],
        min_samples_range=[20, 30, 50, 80]
    )
    
    # Save results
    spatial_results.to_csv(os.path.join(CONFIG['STATS_DIR'], 'spatial_parameter_test.csv'), index=False)
    print(f"\nSpatial parameter test results:\n{spatial_results}")
    
    # Visualize
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create pivot table for heatmap
    pivot_df = spatial_results.pivot_table(
        values='n_clusters',
        index='min_cluster_size',
        columns='min_samples'
    )
    
    sns.heatmap(pivot_df, annot=True, fmt='g', cmap='viridis', ax=ax, cbar_kws={'label': 'Number of Clusters'})
    ax.set_title('Number of Clusters by HDBSCAN Parameters')
    ax.set_xlabel('min_samples')
    ax.set_ylabel('min_cluster_size')
    
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['PLOTS_DIR'], 'parameter_test_spatial_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: parameter_test_spatial_heatmap.png")
    
    print("\n" + "="*100)
    print("PARAMETER TESTING COMPLETE")
    print("="*100 + "\n")


# ========================================================================
# MAIN FUNCTION
# ========================================================================


def train_hierarchical_clustering(n_temporal_clusters=6):
    """
    Train hierarchical spatiotemporal clustering with checkpoint recovery.
    Skips steps if checkpoints already exist (saves 45+ minutes!)
    """
    print("\n" + "="*100)
    print("HIERARCHICAL SPATIOTEMPORAL CLUSTERING WITH CHECKPOINT RECOVERY")
    print("="*100 + "\n")
    
    checkpoint_mgr = CheckpointManager(CONFIG['CHECKPOINTS_DIR'])
    
    # ========================================================================
    # STEP 1: TEMPORAL CLUSTERING
    # ========================================================================
    
    print("[STEP 1] TEMPORAL CLUSTERING")
    print("-"*100)
    
    # Check if already computed
    if checkpoint_mgr.checkpoint_exists('temporal_info'):
        print("✓ Temporal clustering checkpoint found - LOADING...")
        df = checkpoint_mgr.load_checkpoint('df_temporal', 'parquet')
        temporal_info = checkpoint_mgr.load_checkpoint('temporal_info', 'json')
        print(f"✓ Loaded: {len(temporal_info)} temporal clusters")
    else:
        print("✗ Temporal clustering checkpoint NOT found - COMPUTING...")
        df = load_and_prepare_data()
        df, temporal_info = temporal_clustering(df, n_clusters=n_temporal_clusters)
        
        checkpoint_mgr.save_checkpoint('df_temporal', df, 'parquet')
        checkpoint_mgr.save_checkpoint('temporal_info', temporal_info, 'json')
        print(f"✓ Saved: {len(temporal_info)} temporal clusters to checkpoint")
    
    # ========================================================================
    # STEP 2: SPATIAL CLUSTERING (HDBSCAN + KMEANS)
    # ========================================================================
    
    print("\n[STEP 2] SPATIAL CLUSTERING (HDBSCAN + KMEANS HYBRID)")
    print("-"*100)
    
    # Check if already computed
    if checkpoint_mgr.checkpoint_exists('spatial_info'):
        print("✓ Spatial clustering checkpoint found - LOADING...")
        df = checkpoint_mgr.load_checkpoint('df_hierarchical', 'parquet')
        spatial_info = checkpoint_mgr.load_checkpoint('spatial_info', 'json')
        print(f"✓ Loaded: {len(spatial_info)} spatial clusters")
    else:
        print("✗ Spatial clustering checkpoint NOT found - COMPUTING...")
        df, spatial_info = spatial_clustering_within_temporal(df)
        
        checkpoint_mgr.save_checkpoint('spatial_info', spatial_info, 'json')
        checkpoint_mgr.save_checkpoint('df_hierarchical', df, 'parquet')
        print(f"✓ Saved: {len(spatial_info)} spatial clusters to checkpoint")
    
    # ========================================================================
    # STEP 3: VISUALIZATIONS (Always regenerate - fast)
    # ========================================================================
    
    print("\n[STEP 3] GENERATING VISUALIZATIONS")
    print("-"*100)
    
    print("Creating overall spatial distribution plot...")
    create_spatial_visualizations(df, spatial_info)
    
    print("Creating 6 temporal-specific geographic plots...")
    create_temporal_geographic_visualizations(df, spatial_info)
    
    # ========================================================================
    # STEP 4: STATISTICS & REPORTS (Always regenerate - fast)
    # ========================================================================
    
    print("\n[STEP 4] EXPORTING STATISTICS")
    print("-"*100)
    
    # Temporal stats
    print("Exporting temporal statistics...")
    temporal_clusters = df.groupby('temporal_cluster').size()
    temporal_stats = {
        f'T{t_id}': {
            'size': int(size),
            'pct': float(100 * size / len(df)),
            'pattern': temporal_info.get(t_id, {}).get('pattern_name', f'Temporal {t_id}')
        }
        for t_id, size in temporal_clusters.items()
    }
    with open(os.path.join(CONFIG['STATS_DIR'], 'temporal_clusters_statistics.json'), 'w') as f:
        json.dump(temporal_stats, f, indent=2)
    print(f"✓ Saved: temporal_clusters_statistics.json")
    
    # Spatial stats
    print("Exporting spatial statistics...")
    with open(os.path.join(CONFIG['STATS_DIR'], 'spatial_clusters_statistics.json'), 'w') as f:
        json.dump(spatial_info, f, indent=2, default=str)
    print(f"✓ Saved: spatial_clusters_statistics.json")
    
    # Combined report
    print("Generating combined report...")
    report_path = os.path.join(CONFIG['STATS_DIR'], 'clustering_summary_finegrained.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("HIERARCHICAL SPATIOTEMPORAL CLUSTERING SUMMARY\n")
        f.write("="*100 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("OVERVIEW:\n")
        f.write("-"*100 + "\n")
        f.write(f"Total trips: {len(df):,}\n")
        f.write(f"Temporal clusters: {df['temporal_cluster'].nunique()}\n")
        f.write(f"Spatial zones: {len(spatial_info)}\n")
        f.write(f"Classified points: {(df['hierarchical_cluster'] != '-1').sum():,} ")
        f.write(f"({100 * (df['hierarchical_cluster'] != '-1').sum() / len(df):.2f}%)\n")
        f.write(f"Noise points: {(df['hierarchical_cluster'] == '-1').sum():,} ")
        f.write(f"({100 * (df['hierarchical_cluster'] == '-1').sum() / len(df):.2f}%)\n\n")
        
        f.write("TEMPORAL PATTERNS:\n")
        f.write("-"*100 + "\n")
        for t_id in sorted(temporal_stats.keys()):
            info = temporal_stats[t_id]
            f.write(f"{t_id}: {info['pattern']} ({info['size']:,} trips, {info['pct']:.2f}%)\n")
    
    print(f"✓ Saved: clustering_summary_finegrained.txt")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    
    print("\n" + "="*100)
    print("CLUSTERING COMPLETE!")
    print("="*100)
    print(f"Total spatial zones: {len(spatial_info)}")
    print(f"Average zones per temporal: {len(spatial_info) / df['temporal_cluster'].nunique():.0f}")
    print(f"Output directory: {CONFIG['OUTPUT_DIR']}")
    print("="*100 + "\n")
    
    return {
        'df': df,
        'temporal_info': temporal_info,
        'spatial_info': spatial_info,
        'n_temporal': df['temporal_cluster'].nunique(),
        'n_spatial': len(spatial_info),
    }


# ========================================================================
# ENTRY POINT
# ========================================================================


if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "train_6":
            results = train_hierarchical_clustering(n_temporal_clusters=6)
            
        elif command == "train_custom":
            n_clusters = int(sys.argv[2]) if len(sys.argv) > 2 else 6
            results = train_hierarchical_clustering(n_temporal_clusters=n_clusters)
            
        elif command == "test_params":
            test_parameters()
            
        else:
            print("Usage:")
            print("  python hierarchical_clustering.py train_6")
            print("  python hierarchical_clustering.py train_custom 8")
            print("  python hierarchical_clustering.py test_params")
    else:
        results = train_hierarchical_clustering(n_temporal_clusters=6)
