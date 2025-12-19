import pandas as pd
import numpy as np
import kagglehub
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
import glob
import os
import joblib
import warnings
import sys


# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)


# ============================================================================
# MODE SELECTION - RUN VISUALIZATION ONLY OR FULL PIPELINE
# ============================================================================
# Usage:
#   Full pipeline:     python script.py --mode full
#   Visualization only: python script.py --mode viz
#   Default (full):     python script.py
# ============================================================================

def get_execution_mode():
    """Parse command line arguments to determine execution mode."""
    if len(sys.argv) > 1 and sys.argv[1] == '--mode':
        if len(sys.argv) > 2:
            mode = sys.argv[2].lower()
            if mode in ['full', 'viz']:
                return mode
    return 'full'  # Default to full pipeline


# ============================================================================
# 0. UTILITY FUNCTIONS
# ============================================================================


def format_number(value):
    """
    Convert numbers to human-readable format (k, m, b, etc.)
    """
    if pd.isna(value) or value == 0:
        return "0"

    abs_val = abs(value)

    if abs_val >= 1e9:
        return f"{value / 1e9:.1f}b"
    elif abs_val >= 1e6:
        return f"{value / 1e6:.1f}m"
    elif abs_val >= 1e3:
        return f"{value / 1e3:.1f}k"
    elif abs_val >= 1:
        return f"{value:.2f}"
    else:
        return f"{value:.4f}"


# ============================================================================
# 1. DATA INGESTION AND PREPROCESSING
# ============================================================================


def load_and_preprocess_data(input_pattern):
    """
    Loads multiple parquet files, concatenates them, and drops unnecessary columns.
    Performs initial filtering of zero-distance/duration trips to prevent division errors.
    """
    print(f"[INFO] Searching for files matching: {input_pattern}")
    parquet_files = sorted(glob.glob(input_pattern))

    if not parquet_files:
        raise FileNotFoundError("No parquet files found. Please check the input path.")

    print(f"[INFO] Found {len(parquet_files)} parquet files. Loading...")

    # Load all chunks
    dfs = [pd.read_parquet(f) for f in parquet_files]
    df = pd.concat(dfs, ignore_index=True)

    print(f"[INFO] Raw combined data shape: ({format_number(df.shape[0])}, {df.shape[1]})")
    print(f"[INFO] Column names: {df.columns.tolist()}")

    # Drop administrative/irrelevant columns
    columns_to_drop = [
        'VendorID', 'RateCodeID', 'RatecodeID', 'store_and_fwd_flag', 
        'extra', 'mta_tax', 'tolls_amount', 'improvement_surcharge',
        'passenger_count', 'payment_type', 'tip_amount', 'total_amount'
    ]

    # Only drop columns that actually exist in the dataframe
    existing_cols_to_drop = [c for c in columns_to_drop if c in df.columns]
    if existing_cols_to_drop:
        print(f"[INFO] Dropping columns: {existing_cols_to_drop}")
        df = df.drop(columns=existing_cols_to_drop)

    # re-filter zero distances and durations to prevent division by zero later
    initial_rows = len(df)
    if 'trip_distance' in df.columns:
        df = df[df['trip_distance'] > 0]
        print(f"[INFO] Removed {format_number(initial_rows - len(df))} rows with trip_distance <= 0")

    current_rows = len(df)
    df = df[df['tpep_dropoff_datetime'] > df['tpep_pickup_datetime']]
    print(f"[INFO] Removed {format_number(current_rows - len(df))} rows with 0 or negative duration")

    if 'pickup_latitude' in df.columns and 'dropoff_latitude' in df.columns:
         before_coord_filter = len(df)
         df = df[~((df['pickup_latitude'] == df['dropoff_latitude']) & 
                   (df['pickup_longitude'] == df['dropoff_longitude']))]
         print(f"[INFO] Removed {format_number(before_coord_filter - len(df))} rows with identical pickup/dropoff coordinates")

    print(f"[INFO] Data loaded and preprocessed. Shape: ({format_number(df.shape[0])}, {df.shape[1]})")
    return df


# ============================================================================
# 2. FEATURE ENGINEERING
# ============================================================================


def engineer_features(df, output_path, create_viz=True):
    """
    Calculates derived spatiotemporal and pricing features.
    """
    print("\n[PROCESS] Calculating derived features...")

    # Ensure datetime format
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])

    # Trip duration (minutes)
    df['trip_duration_min'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60

    # Trip distance (in km)
    if 'trip_distance' in df.columns:
        print("[INFO] Using 'trip_distance' column (from meter/GPS) for distance calculation")
        df['trip_distance_km'] = df['trip_distance'] * 1.60934
    else:
        print("[WARNING] 'trip_distance' column missing - falling back to Haversine distance (straight line)")
        def haversine_distance(lat1, lon1, lat2, lon2):
            R = 6371  # Earth radius in km
            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            return R * c

        df['trip_distance_km'] = haversine_distance(
            df['pickup_latitude'], df['pickup_longitude'],
            df['dropoff_latitude'], df['dropoff_longitude']
        )

    # Average speed (km/h) - safe division guaranteed by pre-filtering
    df['avg_speed_kmh'] = (df['trip_distance_km'] / df['trip_duration_min']) * 60

    # Temporal features
    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour + df['tpep_pickup_datetime'].dt.minute / 60.0
    df['pickup_weekday'] = df['tpep_pickup_datetime'].dt.weekday
    df['pickup_date'] = df['tpep_pickup_datetime'].dt.date

    # Price-related features - safe division guaranteed
    df['price_per_km'] = df['fare_amount'] / df['trip_distance_km']
    df['price_per_min'] = df['fare_amount'] / df['trip_duration_min']

    print("[INFO] Derived features created")

    # Visualization: Feature distributions after engineering (P0-P95 RANGE) - only if flag is True
    if create_viz:
        visualize_feature_distributions(df, output_path, stage="01_After_Feature_Engineering", orientation='vertical')

    return df


def safe_histogram(ax, data, title, bins=100, color='steelblue', xlabel='Value'):
    """
    Creates histogram showing P0-P95 range (0/min to 95th percentile).
    This excludes only extreme outliers (top 5%) while showing full lower distribution.
    """
    # Remove inf and nan values
    data_clean = data.replace([np.inf, -np.inf], np.nan).dropna()

    if len(data_clean) == 0:
        ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title, fontweight='bold', fontsize=9, pad=2)
        return

    median = data_clean.median()
    p95 = data_clean.quantile(0.95)

    # Filter data to P0-P95 range (0 to 95th percentile)
    data_clipped = data_clean[(data_clean >= 0) & (data_clean <= p95)]

    # Plot P0-P95 histogram
    if len(data_clipped) > 0:
        ax.hist(data_clipped, bins=bins, alpha=0.7, color=color, edgecolor='black')
        ax.axvline(median, color='red', linestyle='--', linewidth=2,
                   label=f'Median: {median:.2f}')

    # Set x-axis to P0-P95 range
    ax.set_xlim(left=0, right=p95)

    ax.set_title(f'{title} - Min: {data_clean.min():.2f}, Max: {data_clean.max():.2f}, P95: {p95:.2f}',
                 fontweight='bold', fontsize=12, pad=4, linespacing=1.0)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.legend(fontsize=12, loc='upper right')
    ax.margins(x=0.02, y=0.08)


def visualize_feature_distributions(df, output_path, stage="raw", orientation='horizontal'):
    """
    Creates comprehensive distribution plots showing P0-P95 range of data.
    Used after feature engineering AND after all cleaning stages.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with features
    output_path : str
        Directory to save visualizations
    stage : str
        Stage name for the visualization title and filename
    orientation : str
        'horizontal' (2x3, default, 18x12) or 'vertical' (3x2, 12x18)
    """
    # Determine layout based on orientation
    if orientation == 'vertical':
        figsize = (12, 18)
        layout = (3, 2)
    else:  # horizontal (default)
        figsize = (18, 12)
        layout = (2, 3)
    
    fig, axes = plt.subplots(*layout, figsize=figsize)
    plt.tight_layout(rect=[0, 0, 1, 0.94], h_pad=1.5, w_pad=1.0)
    fig.suptitle(f'Feature Distributions - {stage} Showing P0-P95 range',
                 fontsize=16, fontweight='bold', y=0.96)

    # Flatten axes for consistent indexing regardless of orientation
    axes_flat = axes.flatten()

    # Trip Duration
    safe_histogram(axes_flat[0], df['trip_duration_min'], 'Trip Duration (min)',
                  color='steelblue', xlabel='Minutes')

    # Trip Distance
    safe_histogram(axes_flat[1], df['trip_distance_km'], 'Trip Distance (km)',
                  color='darkorange', xlabel='Distance (km)', bins=50)

    # Average Speed
    safe_histogram(axes_flat[2], df['avg_speed_kmh'], 'Avg Speed (km/h)',
                  color='forestgreen', xlabel='Speed (km/h)')

    # Fare Amount
    safe_histogram(axes_flat[3], df['fare_amount'], 'Fare Amount ($)',
                  color='crimson', xlabel='Fare ($)')

    # Price per KM
    safe_histogram(axes_flat[4], df['price_per_km'], 'Price/KM ($/km)',
                  color='mediumpurple', xlabel='Price ($/km)')

    # Price per Minute
    safe_histogram(axes_flat[5], df['price_per_min'], 'Price per Minute ($/min)',
                  color='teal', xlabel='Price ($/min)')
    
    # Add orientation suffix to filename for clarity
    orientation_suffix = '_vertical' if orientation == 'vertical' else ''
    save_figure(fig, output_path, f'{stage}_distributions{orientation_suffix}.png')
    plt.close()


def visualize_boxplot_single_metric(df_before, df_after, output_path, stage_name, 
                                   metric_name, feature_key, title):
    """
    Creates 2-boxplot figure showing Before vs After for each metric.

    Left: Before (orange, full Y-axis range showing all outliers)
    Right: After (green, independent Y-axis - smaller range due to fewer outliers)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # Extract and clean data
    before = df_before[feature_key].replace([np.inf, -np.inf], np.nan).dropna()
    after = df_after[feature_key].replace([np.inf, -np.inf], np.nan).dropna()

    if len(before) == 0 or len(after) == 0:
        print(f"[WARNING] No valid data for {feature_key} in {stage_name}")
        return

    # ================================================================
    # LEFT: Before boxplot (orange)
    # Y-axis will auto-scale to full range including outliers
    # ================================================================
    bp_before = axes[0].boxplot(before, vert=True, patch_artist=True, widths=0.6)
    bp_before['boxes'][0].set_facecolor('orange')
    bp_before['boxes'][0].set_alpha(0.7)

    axes[0].set_title(f'Before\n{format_number(len(before))} records', fontweight='bold', fontsize=11)
    axes[0].set_ylabel(metric_name, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].set_xticklabels([])

    # ================================================================
    # RIGHT: After boxplot (green)
    # Y-axis will auto-scale to cleaner range
    # ================================================================
    bp_after = axes[1].boxplot(after, vert=True, patch_artist=True, widths=0.6)
    bp_after['boxes'][0].set_facecolor('green')
    bp_after['boxes'][0].set_alpha(0.7)

    axes[1].set_title(f'After\n{format_number(len(after))} records', fontweight='bold', fontsize=11)
    axes[1].set_ylabel(metric_name, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].set_xticklabels([])

    plt.tight_layout()
    save_figure(fig, output_path, f'{stage_name}_{feature_key}.png')
    plt.close()

    print(f"[CHART] Saved: {stage_name}_{feature_key}.png")
    print(f"[DATA] Before: {format_number(len(before))} records, range: {before.min():.2f} to {before.max():.2f}")
    print(f"[DATA] After:  {format_number(len(after))} records, range: {after.min():.2f} to {after.max():.2f}")


def save_figure(fig, output_path, filename):
    """Helper function to save figures."""
    os.makedirs(output_path, exist_ok=True)
    filepath = os.path.join(output_path, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')


def calculate_and_print_statistics(df, stage_name, output_path=None):
    """
    Calculates and prints comprehensive statistics for the dataset.
    """
    # Remove inf and nan for statistics
    df_clean = df.copy()
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)

    stats_data = {
        'Metric': ['Trip Duration (min)', 'Distance (km)', 'Avg Speed (km/h)', 'Fare ($)', 'Price/km ($)', 'Price/min ($)'],
        'Mean': [
            format_number(df_clean['trip_duration_min'].mean()),
            format_number(df_clean['trip_distance_km'].mean()),
            format_number(df_clean['avg_speed_kmh'].mean()),
            format_number(df_clean['fare_amount'].mean()),
            format_number(df_clean['price_per_km'].mean()),
            format_number(df_clean['price_per_min'].mean())
        ],
        'Median': [
            format_number(df_clean['trip_duration_min'].median()),
            format_number(df_clean['trip_distance_km'].median()),
            format_number(df_clean['avg_speed_kmh'].median()),
            format_number(df_clean['fare_amount'].median()),
            format_number(df_clean['price_per_km'].median()),
            format_number(df_clean['price_per_min'].median())
        ],
        'Std Dev': [
            format_number(df_clean['trip_duration_min'].std()),
            format_number(df_clean['trip_distance_km'].std()),
            format_number(df_clean['avg_speed_kmh'].std()),
            format_number(df_clean['fare_amount'].std()),
            format_number(df_clean['price_per_km'].std()),
            format_number(df_clean['price_per_min'].std())
        ],
        'Min': [
            format_number(df_clean['trip_duration_min'].min()),
            format_number(df_clean['trip_distance_km'].min()),
            format_number(df_clean['avg_speed_kmh'].min()),
            format_number(df_clean['fare_amount'].min()),
            format_number(df_clean['price_per_km'].min()),
            format_number(df_clean['price_per_min'].min())
        ],
        'Max': [
            format_number(df_clean['trip_duration_min'].max()),
            format_number(df_clean['trip_distance_km'].max()),
            format_number(df_clean['avg_speed_kmh'].max()),
            format_number(df_clean['fare_amount'].max()),
            format_number(df_clean['price_per_km'].max()),
            format_number(df_clean['price_per_min'].max())
        ]
    }

    stats_df = pd.DataFrame(stats_data)
    print(f"\n{stage_name}")
    print("=" * 100)
    print(stats_df.to_string(index=False))
    print(f"Dataset size: {format_number(len(df))} rows")
    print("=" * 100)

    return stats_df


# ============================================================================
# 3. OUTLIER DETECTION PIPELINE
# ============================================================================


def apply_cleaning_filters(df, output_path):
    """
    Applies multi-stage outlier detection with FOCUSED single-metric visualizations.
    """
    initial_rows = len(df)

    # Print PRE-PROCESSED statistics
    print("\n" + "=" * 100)
    pre_process_stats = calculate_and_print_statistics(df, "[STATS] PRE-PROCESSED STATISTICS (After loading + feature engineering)")

    # Visualization: Raw data initial boxplot
    print("\n[PROCESS] Creating raw data visualization...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Raw Data - All Metrics (Full Range)', fontsize=16, fontweight='bold')

    features = ['trip_duration_min', 'trip_distance_km', 'avg_speed_kmh',
                'fare_amount', 'price_per_km', 'price_per_min']
    colors = ['steelblue', 'darkorange', 'forestgreen', 'crimson', 'mediumpurple', 'teal']

    for idx, (ax, feature, color) in enumerate(zip(axes.flatten(), features, colors)):
        data = df[feature].replace([np.inf, -np.inf], np.nan).dropna()

        if len(data) == 0:
            ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
            continue

        bp = ax.boxplot(data, vert=True, patch_artist=True, widths=0.5)
        for patch in bp['boxes']:
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_title(feature.replace('_', ' ').title(), fontweight='bold')
        ax.set_ylabel('Value')
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_figure(fig, output_path, '00_Raw_Data_boxplots.png')
    plt.close()

    print("\n[STAGE_1] Basic validity checks...")
    df_stage1 = df.copy()

    cols_to_check = ['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'fare_amount']
    if 'pickup_latitude' in df_stage1.columns:
        cols_to_check.extend(['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude'])

    df_stage1 = df_stage1.dropna(subset=cols_to_check)
    df_stage1 = df_stage1[df_stage1['fare_amount'] > 0]
    df_stage1 = df_stage1.replace([np.inf, -np.inf], np.nan)
    df_stage1 = df_stage1.dropna(subset=['trip_duration_min', 'trip_distance_km', 'avg_speed_kmh', 'price_per_km', 'price_per_min'])

    rows_after_stage1 = len(df_stage1)
    print(f"[INFO] After basic cleaning: {format_number(rows_after_stage1)} ({100*rows_after_stage1/initial_rows:.1f}%)")
    visualize_boxplot_single_metric(df, df_stage1, output_path, '01_stage1',
                                   'Trip Duration (minutes)', 'trip_duration_min',
                                   'Stage 1: Trip Duration Before vs After (Basic Validity Checks)')

    print("\n[STAGE_2] Geographic outliers (NYC bounding box)...")
    df_stage2 = df_stage1.copy()

    if 'pickup_latitude' in df_stage2.columns:
        nyc_mask = (
            ((df_stage2['pickup_longitude'] >= -74.15) & (df_stage2['pickup_longitude'] <= -73.65) &
             (df_stage2['pickup_latitude'] >= 40.49) & (df_stage2['pickup_latitude'] <= 40.91)) |
            ((df_stage2['dropoff_longitude'] >= -74.15) & (df_stage2['dropoff_longitude'] <= -73.65) &
             (df_stage2['dropoff_latitude'] >= 40.49) & (df_stage2['dropoff_latitude'] <= 40.91))
        )
        df_stage2 = df_stage2[nyc_mask]

    df_stage2 = df_stage2[df_stage2['trip_distance_km'] > 0.01]

    rows_after_stage2 = len(df_stage2)
    print(f"[INFO] After geographic filtering: {format_number(rows_after_stage2)} (removed {format_number(rows_after_stage1 - rows_after_stage2)})")
    visualize_boxplot_single_metric(df_stage1, df_stage2, output_path, '02_stage2',
                                   'Trip Distance (km)', 'trip_distance_km',
                                   'Stage 2: Trip Distance Before vs After (Geographic Filtering)')

    if 'pickup_latitude' in df_stage2.columns:
        visualize_geographic_distribution(df_stage1, df_stage2, output_path)

    print("\n[STAGE_3] Speed-based outliers (1-120 km/h)...")
    df_stage3 = df_stage2.copy()

    speed_mask = (df_stage3['avg_speed_kmh'] >= 1) & (df_stage3['avg_speed_kmh'] <= 120)
    df_stage3 = df_stage3[speed_mask]

    rows_after_stage3 = len(df_stage3)
    print(f"[INFO] After speed filtering: {format_number(rows_after_stage3)} (removed {format_number(rows_after_stage2 - rows_after_stage3)})")
    visualize_boxplot_single_metric(df_stage2, df_stage3, output_path, '03_stage3',
                                   'Average Speed (km/h)', 'avg_speed_kmh',
                                   'Stage 3: Average Speed Before vs After (Speed Filtering)')

    print("\n[STAGE_4] Trip duration outliers (IQR method)...")
    df_stage4 = df_stage3.copy()

    Q1_duration = df_stage4['trip_duration_min'].quantile(0.25)
    Q3_duration = df_stage4['trip_duration_min'].quantile(0.75)
    IQR_duration = Q3_duration - Q1_duration

    duration_mask = (df_stage4['trip_duration_min'] >= Q1_duration - 1.5*IQR_duration) & \
                    (df_stage4['trip_duration_min'] <= Q3_duration + 3*IQR_duration)
    df_stage4 = df_stage4[duration_mask]

    rows_after_stage4 = len(df_stage4)
    print(f"[INFO] After duration filtering: {format_number(rows_after_stage4)} (removed {format_number(rows_after_stage3 - rows_after_stage4)})")
    visualize_boxplot_single_metric(df_stage3, df_stage4, output_path, '04_stage4',
                                   'Trip Duration (minutes)', 'trip_duration_min',
                                   'Stage 4: Trip Duration Before vs After (IQR-based Filtering)')

    print("\n[STAGE_5] Distance outliers (IQR method)...")
    df_stage5 = df_stage4.copy()

    Q1_dist = df_stage5['trip_distance_km'].quantile(0.25)
    Q3_dist = df_stage5['trip_distance_km'].quantile(0.75)
    IQR_dist = Q3_dist - Q1_dist

    dist_mask = df_stage5['trip_distance_km'] <= (Q3_dist + 3*IQR_dist)
    df_stage5 = df_stage5[dist_mask]

    rows_after_stage5 = len(df_stage5)
    print(f"[INFO] After distance filtering: {format_number(rows_after_stage5)} (removed {format_number(rows_after_stage4 - rows_after_stage5)})")
    visualize_boxplot_single_metric(df_stage4, df_stage5, output_path, '05_stage5',
                                   'Trip Distance (km)', 'trip_distance_km',
                                   'Stage 5: Trip Distance Before vs After (IQR-based Filtering)')

    print("\n[STAGE_6] Pricing outliers...")
    df_stage6 = df_stage5.copy()

    price_per_km_mask = (df_stage6['price_per_km'] >= 0.5) & (df_stage6['price_per_km'] <= 20)
    price_per_min_mask = (df_stage6['price_per_min'] >= 0.5) & (df_stage6['price_per_min'] <= 100)
    df_stage6 = df_stage6[price_per_km_mask & price_per_min_mask]

    rows_after_stage6 = len(df_stage6)
    print(f"[INFO] After pricing filtering: {format_number(rows_after_stage6)} (removed {format_number(rows_after_stage5 - rows_after_stage6)})")
    visualize_boxplot_single_metric(df_stage5, df_stage6, output_path, '06_stage6a',
                                   'Price per KM ($/km)', 'price_per_km',
                                   'Stage 6a: Price per KM Before vs After (Pricing Filtering)')
    visualize_boxplot_single_metric(df_stage5, df_stage6, output_path, '06_stage6b',
                                   'Price per Minute ($/min)', 'price_per_min',
                                   'Stage 6b: Price per Minute Before vs After (Pricing Filtering)')

    # Initialize scaler
    robust_scaler = RobustScaler()
    features = ['trip_duration_min', 'trip_distance_km', 'avg_speed_kmh', 'fare_amount']
    X = df_stage6[features].values
    robust_scaler.fit(X)

    return df_stage6, robust_scaler, pre_process_stats


def visualize_geographic_distribution(df_before, df_after, output_path):
    """Visualizes pickup locations before and after geographic filtering."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Geographic Distribution of Pickups: Before vs After Filtering', fontsize=14, fontweight='bold')

    axes[0].scatter(df_before['pickup_longitude'], df_before['pickup_latitude'],
                   alpha=0.3, s=1, color='red')
    axes[0].set_title(f'Before: {format_number(len(df_before))} trips', fontweight='bold')
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    axes[0].grid(alpha=0.3)

    axes[1].scatter(df_after['pickup_longitude'], df_after['pickup_latitude'],
                   alpha=0.3, s=1, color='green')
    axes[1].set_title(f'After: {format_number(len(df_after))} trips', fontweight='bold')
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    save_figure(fig, output_path, '02_Geographic_Distribution.png')
    plt.close()


# ============================================================================
# 4. VISUALIZATION-ONLY MODE - Load cleaned data and visualize
# ============================================================================


def visualize_cleaned_data_only(data_path, output_path, featured_data_path=None):
    """
    Load pre-cleaned dataset and generate visualizations of BOTH:
    1. Before cleaning (01_After_Feature_Engineering) - if featured data available
    2. After cleaning (02_After_All_Cleaning_Stages)
    
    This is the key function for visualization-only mode.
    
    Usage: 
        python script.py --mode viz
    
    Parameters:
    -----------
    data_path : str
        Path to the cleaned parquet file (taxi_data_cleaned_full.parquet)
    output_path : str
        Directory where visualizations will be saved
    featured_data_path : str
        Optional path to featured (before cleaning) data
    """
    print("\n" + "=" * 100)
    print("[PROCESS] VISUALIZATION-ONLY MODE: Generating BEFORE and AFTER distributions")
    print("=" * 100)
    
    # Try to load featured (pre-cleaned) data
    df_featured = None
    if featured_data_path and os.path.exists(featured_data_path):
        print(f"\n[OK] Loading featured (pre-cleaned) dataset from: {featured_data_path}")
        df_featured = pd.read_parquet(featured_data_path)
        print(f"[INFO] Loaded {format_number(len(df_featured))} records (BEFORE cleaning)")
    else:
        print(f"\n[WARNING] Featured data not found at: {featured_data_path}")
        print("[INFO] SOLUTION: Run full pipeline again with --mode full")
        print("[INFO] Command: python script.py --mode full")
        print("\n[INFO] This will generate the featured data file for subsequent viz-mode runs.")
    
    # Load the cleaned dataset
    print(f"\n[PROCESS] Loading cleaned dataset from: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"[ERROR] Cleaned data file not found at {data_path}")
        print("[INFO] Please run the full pipeline first (python script.py --mode full)")
        return
    
    df_cleaned = pd.read_parquet(data_path)
    print(f"[OK] Loaded {format_number(len(df_cleaned))} records (AFTER cleaning)")
    
    # Generate BEFORE visualizations if featured data available
    if df_featured is not None:
        print("\n" + "=" * 100)
        print("[PROCESS] Generating BEFORE-cleaning visualizations (vertical)...")
        print("=" * 100)
        visualize_feature_distributions(df_featured, output_path, 
                                       stage="01_After_Feature_Engineering", 
                                       orientation='vertical')
        print("[OK] Saved: 01_After_Feature_Engineering_distributions_vertical.png")
        print("\n[PROCESS] BEFORE-cleaning statistics:")
        calculate_and_print_statistics(df_featured, "[STATS] BEFORE-CLEANING DATA STATISTICS")
    
    # Generate AFTER visualizations
    print("\n" + "=" * 100)
    print("[PROCESS] Generating AFTER-cleaning visualizations (vertical)...")
    print("=" * 100)
    visualize_feature_distributions(df_cleaned, output_path, 
                                   stage="02_After_All_Cleaning_Stages", 
                                   orientation='vertical')
    print("[OK] Saved: 02_After_All_Cleaning_Stages_distributions_vertical.png")
    
    # Print AFTER statistics
    print("\n[PROCESS] AFTER-cleaning statistics:")
    calculate_and_print_statistics(df_cleaned, "[STATS] AFTER-CLEANING DATA STATISTICS")
    
    # Summary comparison
    if df_featured is not None:
        print("\n" + "=" * 100)
        print("[SUMMARY] CLEANING IMPACT SUMMARY")
        print("=" * 100)
        print(f"Records BEFORE cleaning: {format_number(len(df_featured))}")
        print(f"Records AFTER cleaning:  {format_number(len(df_cleaned))}")
        retention = 100 * len(df_cleaned) / len(df_featured)
        print(f"Retention rate: {retention:.1f}%")
        print(f"Records removed: {100 - retention:.1f}%")
        print("=" * 100)


# ============================================================================
# 5. REPORTING AND SAVING
# ============================================================================


def save_and_report(df, original_size, output_path, scaler, pre_process_stats):
    """Saves the cleaned dataframe and cleaning artifacts."""
    final_size = len(df)
    retention = 100 * final_size / original_size

    print(f"\n" + "=" * 100)
    print("[SUMMARY] CLEANING SUMMARY")
    print("=" * 100)
    print(f"Original size (after loading):  {format_number(original_size)}")
    print(f"Final size (after cleaning):    {format_number(final_size)}")
    print(f"Retained:                       {retention:.1f}%")
    print(f"Removed as outliers:            {100 - retention:.1f}%")
    print("=" * 100)

    # Calculate and print post-processed statistics
    post_process_stats = calculate_and_print_statistics(df, "[STATS] POST-PROCESSED STATISTICS (After all 6 cleaning stages)")

    # ================================================================
    # Generate final visualization of cleaned data (P0-P95 RANGE)
    # ================================================================
    print("\n[PROCESS] Creating final visualization of cleaned data (vertical)...")
    visualize_feature_distributions(df, output_path, stage="02_After_All_Cleaning_Stages", orientation='vertical')

    os.makedirs(output_path, exist_ok=True)
    file_name = 'taxi_data_cleaned_full.parquet'
    full_output_path = os.path.join(output_path, file_name)

    print(f"\n[PROCESS] Saving to {full_output_path}...")
    df.to_parquet(full_output_path, compression='snappy', index=False)

    stats_file = os.path.join(output_path, 'cleaning_statistics.csv')
    post_process_stats.to_csv(stats_file, index=False)
    print(f"[OK] Statistics saved to: {stats_file}")

    # Save comparison of pre vs post statistics
    comparison_file = os.path.join(output_path, 'statistics_comparison.csv')
    comparison_df = pd.DataFrame({
        'Metric': pre_process_stats['Metric'],
        'Pre-Processed Mean': pre_process_stats['Mean'],
        'Post-Processed Mean': post_process_stats['Mean'],
        'Pre-Processed Median': pre_process_stats['Median'],
        'Post-Processed Median': post_process_stats['Median'],
        'Pre-Processed Std Dev': pre_process_stats['Std Dev'],
        'Post-Processed Std Dev': post_process_stats['Std Dev'],
    })
    comparison_df.to_csv(comparison_file, index=False)
    print(f"[OK] Comparison saved to: {comparison_file}")

    report_path = os.path.join(output_path, 'cleaning_report.pkl')
    scaler_path = os.path.join(output_path, 'robust_scaler.pkl')

    cleaning_report = {
        'original_size': original_size,
        'final_size': final_size,
        'retention_rate': retention,
        'pre_process_statistics': pre_process_stats,
        'post_process_statistics': post_process_stats,
    }

    joblib.dump(cleaning_report, report_path)
    joblib.dump(scaler, scaler_path)

    print("[INFO] All files saved successfully.")
    print(f"\n[INFO] Output files in: {output_path}")
    print(f"[IMPORTANT] * taxi_data_featured.parquet (BEFORE cleaning - for viz mode)")
    print(f"[FILE]      * taxi_data_cleaned_full.parquet (AFTER cleaning)")
    print(f"[FILE]      * cleaning_report.pkl (cleaning metadata)")
    print(f"[FILE]      * robust_scaler.pkl (preprocessing scaler)")
    print(f"[FILE]      * cleaning_statistics.csv (post-processed stats)")
    print(f"[FILE]      * statistics_comparison.csv (pre vs post)")
    print(f"[FEATURED]  * 01_After_Feature_Engineering_distributions_vertical.png (BEFORE cleaning)")
    print(f"[FEATURED]  * 02_After_All_Cleaning_Stages_distributions_vertical.png (AFTER cleaning)")
    print(f"[FILE]      * FOCUSED 2-boxplot PNG files per stage")


# ============================================================================
# MAIN EXECUTION BLOCK
# ============================================================================


if __name__ == "__main__":
    INPUT_PATTERN = "C:/Users/Anya/master_thesis/tmp/chunk_*.parquet"
    OUTPUT_PATH = "C:/Users/Anya/master_thesis/output"
    CLEANED_DATA_PATH = os.path.join(OUTPUT_PATH, 'taxi_data_cleaned_full.parquet')
    FEATURED_DATA_PATH = os.path.join(OUTPUT_PATH, 'taxi_data_featured.parquet')

    # Determine execution mode
    execution_mode = get_execution_mode()

    print("=" * 100)
    print("[STARTUP] NYC TAXI DATA CLEANING & VISUALIZATION PIPELINE")
    print("=" * 100)
    print(f"[INFO] Execution Mode: {execution_mode.upper()}")
    print("=" * 100)

    try:
        if execution_mode == 'viz':
            # ================================================================
            # VISUALIZATION-ONLY MODE
            # ================================================================
            # Load pre-cleaned data and generate visualizations
            visualize_cleaned_data_only(CLEANED_DATA_PATH, OUTPUT_PATH, FEATURED_DATA_PATH)
            
        else:
            # ================================================================
            # FULL PIPELINE MODE
            # ================================================================
            print("\n")
            df_raw = load_and_preprocess_data(INPUT_PATTERN)
            original_size = len(df_raw)

            df_featured = engineer_features(df_raw, OUTPUT_PATH, create_viz=True)

            # ================================================================
            # CRITICAL: SAVE featured data for later use in viz mode
            # ================================================================
            os.makedirs(OUTPUT_PATH, exist_ok=True)
            featured_save_path = os.path.join(OUTPUT_PATH, 'taxi_data_featured.parquet')
            print(f"\n[CRITICAL] Saving featured data to {featured_save_path}")
            print(f"[INFO]     This enables BEFORE-cleaning visualizations in --mode viz")
            df_featured.to_parquet(featured_save_path, compression='snappy', index=False)
            print(f"[OK]       Saved {format_number(len(df_featured))} records successfully")

            df_cleaned, scaler, pre_process_stats = apply_cleaning_filters(df_featured, OUTPUT_PATH)

            save_and_report(df_cleaned, original_size, OUTPUT_PATH, scaler, pre_process_stats)

        print("\n" + "=" * 100)
        if execution_mode == 'viz':
            print("[COMPLETE] VISUALIZATION COMPLETE - Ready for analysis")
        else:
            print("[COMPLETE] PIPELINE COMPLETE - Ready for clustering/analysis phase")
        print("=" * 100)

    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
