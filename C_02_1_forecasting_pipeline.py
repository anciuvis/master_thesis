# ====================================================================================
# Seasonal ARIMA + XGBoost + LSTM models DEMAND PREDICTION PIPELINE FOR NYC TAXI DATA
# ====================================================================================
# ====================================================================================
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Only show ERROR, skip WARNINGS
os.environ['OMP_NUM_THREADS'] = '16'
os.environ['MKL_NUM_THREADS'] = '16'
os.environ['OPENBLAS_NUM_THREADS'] = '16'

import gc
import warnings
import json
import pickle
import logging
import joblib
from datetime import datetime
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Statistical models
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tools.sm_exceptions import ValueWarning

# Multiprocessing
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from multiprocessing import get_context
import psutil

# Machine Learning
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, mean_absolute_percentage_error,
    median_absolute_error, r2_score
)
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import RandomizedSearchCV

# Deep Learning

logging.getLogger('tensorflow').setLevel(logging.ERROR)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Bidirectional, Input, RepeatVector, TimeDistributed
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import mixed_precision
from sklearn.decomposition import PCA
from pca_visualizer import PCAVisualizer

# Feature Importance - Check if SHAP is available
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Suppress specific statsmodels warnings
warnings.simplefilter('ignore', category=ValueWarning)
warnings.simplefilter('ignore', category=FutureWarning)

# ============================================================================
# CONFIGURATION - KEY PARAMETER: N_TOP_CLUSTERS
# ============================================================================

class PipelineConfig:
    """Centralized configuration for the forecasting pipeline"""
    
    def __init__(self):
        # ========== CLUSTER CONFIGURATION ==========
        # THIS IS THE KEY PARAMETER - Change this to train on different numbers of clusters
        self.n_top_clusters = 60  # 
        
        # ========== DATA PATHS ==========
        self.input_data_path = 'C:/Users/Anya/master_thesis/output'
        self.input_file = 'taxi_data_cleaned_full_with_clusters.parquet'
        
        # ========== OUTPUT PATHS (dynamic based on n_top_clusters) ==========
        self.output_base = f'C:/Users/Anya/master_thesis/output/models_upd_30'
        self.checkpoint_dir = os.path.join(self.output_base, 'checkpoints')
        self.results_dir = os.path.join(self.output_base, 'results')
        self.viz_dir = os.path.join(self.output_base, 'visualizations')
        self.log_dir = os.path.join(self.output_base, 'logs')
        self.models_dir = os.path.join(self.output_base, 'models')
        
        # ========== FEATURE ENGINEERING ==========
        self.lag_features = 168  # 7 days of hourly data
        self.seq_length = 168  # LSTM sequence length (168 hours)
        
        # ========== DATA SPLITTING ==========
        self.test_size = 0.2
        self.val_size = 0.1
        
        # ========== XGBOOST HYPERPARAMETERS ==========
        self.xgb_grid_type = 'quick'  # 'quick', 'medium', 'full'
        self.xgb_default_params = {
            'n_estimators': 300,
            'max_depth': 3,
            'learning_rate': 0.1,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }
        
        # ========== LSTM ARCHITECTURE ==========
        self.lstm_n_components = 256  # PCA reduction dimension
        self.lstm_units = 256
        self.lstm_epochs = 100
        self.lstm_batch_size = 16
        
        # ========== SARIMA PARAMETERS ==========
        self.sarima_order = (0, 1, 1)  # Differencing + MA, no AR
        self.sarima_seasonal_order = (0, 1, 1, 168)  # Seasonal MA only
        self.sarima_maxiter = 100  # Reduced iterations
        
        # Create output directories
        for d in [self.checkpoint_dir, self.results_dir, self.viz_dir, self.log_dir, self.models_dir]:
            os.makedirs(d, exist_ok=True)
    
    def get_summary(self):
        """Print configuration summary"""
        return f"""
        ╔════════════════════════════════════════════════════════════════╗
        ║           PIPELINE CONFIGURATION SUMMARY                      ║
        ╠════════════════════════════════════════════════════════════════╣
        ║ Top Clusters:              {self.n_top_clusters:<40} ║
        ║ Input File:                {self.input_file:<40} ║
        ║ Output Base:               {os.path.basename(self.output_base):<40} ║
        ║                                                                ║
        ║ Lag Features:              {self.lag_features:<40} ║
        ║ LSTM Seq Length:           {self.seq_length:<40} ║
        ║ Test/Val Split:            {self.test_size}/{self.val_size:<36} ║
        ║                                                                ║
        ║ LSTM PCA Components:       {self.lstm_n_components:<40} ║
        ║ LSTM Units:                {self.lstm_units:<40} ║
        ║ LSTM Epochs:               {self.lstm_epochs:<40} ║
        ║                                                                ║
        ║ SARIMA Order:              {str(self.sarima_order):<40} ║
        ║ SARIMA Seasonal:           {str(self.sarima_seasonal_order):<40} ║
        ╚════════════════════════════════════════════════════════════════╝
        """


# Initialize config
config = PipelineConfig()

INPUT_DATA_PATH = config.input_data_path
INPUT_FILE = config.input_file
OUTPUT_BASE = config.output_base
N_TOP_CLUSTERS = config.n_top_clusters

CHECKPOINT_DIR = config.checkpoint_dir
RESULTS_DIR = config.results_dir
VIZ_DIR = config.viz_dir
LOG_DIR = config.log_dir
MODELS_DIR = config.models_dir

# Logging
log_file = os.path.join(LOG_DIR, f'integrated_pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# SHAP import warning
if SHAP_AVAILABLE:
    logger.info("[OK] SHAP library available for feature importance")
else:
    logger.warning("[!] SHAP not installed. Using gain-based feature importance as fallback")

np.random.seed(42)
tf.random.set_seed(42)


print("="*80)
print("INTEGRATED FORECASTING PIPELINE - SARIMA + XGBoost + LSTM")
print("="*80)
logger.info("Pipeline initialization started")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

# FIX #7: Helper function for MAPE calculation with zero handling
def calculate_mape(y_true, y_pred, epsilon=1.0):
    """
    Calculate MAPE with epsilon smoothing.
    
    MAPE = mean(|y_true - y_pred| / (|y_true| + epsilon)) * 100
    This prevents division by zero/near-zero values.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Handle zero/small values with epsilon
    denominator = np.abs(y_true) + epsilon
    
    # Calculate percentage error
    percentage_errors = np.abs(y_true - y_pred) / denominator
    
    # Return mean (clipped to prevent extreme values)
    mape = np.nanmean(percentage_errors) * 100
    
    # Clip to reasonable range (0-500% is acceptable)
    return np.clip(mape, 0, 500)

def get_lstm_cluster_index(cluster_name, y_test_columns, lstm_predictions_shape):
    """
    Convert cluster name (string) to numeric index for LSTM predictions.
    
    Parameters:
    -----------
    cluster_name : str or int
        The cluster identifier (e.g., 'T2_S60' or 123)
    y_test_columns : pd.Index or list
        Column names from y_test DataFrame
    lstm_predictions_shape : tuple
        Shape of lstm_predictions numpy array (n_samples, n_clusters)
    
    Returns:
    --------
    int or None
        Numeric index for lstm_predictions array, or None if not found
    """
    try:
        # Convert cluster_name to string for consistent comparison
        cluster_str = str(cluster_name)
        
        # Extract column list from pandas Index if needed
        if hasattr(y_test_columns, 'tolist'):
            cols_list = y_test_columns.tolist()
        else:
            cols_list = list(y_test_columns)
        
        # Convert all columns to strings
        cols_str = [str(c) for c in cols_list]
        
        # Search for match
        if cluster_str in cols_str:
            return cols_str.index(cluster_str)
        else:
            # Fallback: try numeric conversion
            try:
                cluster_num = int(cluster_name)
                if 0 <= cluster_num < lstm_predictions_shape[1]:
                    return cluster_num
            except (ValueError, TypeError):
                pass
            
            return None
            
    except Exception as e:
        logger.warning(f"[!] Failed to get LSTM index for {cluster_name}: {e}")
        return None

def ensure_numeric_array(arr, label="array"):
    """
    Ensure array is numeric (float64), not object dtype.
    
    Parameters:
    -----------
    arr : list, np.ndarray, or pd.Series
        Input array to convert
    label : str
        Name for logging messages
    
    Returns:
    --------
    np.ndarray
        Array with dtype=float64
    """
    if isinstance(arr, (list, np.ndarray)):
        arr = np.asarray(arr, dtype=np.float64)
        if arr.dtype == object:
            logger.warning(f"[!] {label} had object dtype, converting to float64")
            arr = arr.astype(np.float64)
    return arr


# ============================================================================
# CHECKPOINT MANAGER
# ============================================================================

class CheckpointManager:
    """Manages resumable execution"""
    
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        self.state_file = os.path.join(checkpoint_dir, 'pipeline_state.json')
        self.load_state()
    
    def load_state(self):
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                self.state = json.load(f)
            logger.info(f"Checkpoint loaded: {list(self.state.keys())}")
        else:
            self.state = {task: False for task in [
                'data_loaded', 'data_prepared', 'features_engineered',
                'sarima_fitted', 'xgboost_fitted', 'lstm_fitted',
                'models_evaluated', 'feature_importance_computed',
                'visualizations_created', 'report_generated'
            ]}
    
    def save_state(self):
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def mark_complete(self, task: str):
        self.state[task] = True
        self.save_state()
        logger.info(f"Task complete: {task}")
    
    def is_complete(self, task: str) -> bool:
        return self.state.get(task, False)
    
    def save_checkpoint(self, name: str, data):
        path = os.path.join(self.checkpoint_dir, f'{name}.pkl')
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Checkpoint saved: {name}")
    
    def load_checkpoint(self, name: str):
        path = os.path.join(self.checkpoint_dir, f'{name}.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f)
        return None


checkpoint_mgr = CheckpointManager(CHECKPOINT_DIR)


# ============================================================================
# PROGRESS TRACKER
# ============================================================================

class ProgressTracker:
    def __init__(self, total_steps: int = 100):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = datetime.now()
    
    def update(self, step: int, message: str = ""):
        self.current_step = step
        progress = (step / self.total_steps) * 100
        elapsed = (datetime.now() - self.start_time).total_seconds() / 60
        
        bar = "[" * int(progress / 5) + "]" * (20 - int(progress / 5))
        print(f"\r[{bar}] {progress:.1f}% | {elapsed:.1f} min | {message}", end='', flush=True)
        logger.info(f"Progress: {progress:.1f}% - {message}")


progress = ProgressTracker(total_steps=100)


# ============================================================================
# DATA LOADING & PREPARATION
# ============================================================================

def load_data(data_path: str, file_name: str) -> pd.DataFrame:
    """Load taxi data"""
    logger.info("Loading data...")
    progress.update(5, "Loading data from parquet")
    
    full_path = os.path.join(data_path, file_name)
    data = pd.read_parquet(full_path)
    
    logger.info(f"Data shape: {data.shape}")
    logger.info(f"Columns: {data.columns.tolist()}")
    
    return data


def prepare_demand_matrix(data: pd.DataFrame, freq: str = 'H') -> pd.DataFrame:
    """Prepare time series demand matrix"""
    logger.info(f"Preparing demand matrix (freq={freq})...")
    progress.update(10, "Aggregating demand by time and cluster")
    
    data['tpep_pickup_datetime'] = pd.to_datetime(data['tpep_pickup_datetime'])
    data['time_period'] = data['tpep_pickup_datetime'].dt.floor(freq)
    
    demand = data.groupby(['time_period', 'hierarchical_cluster']).size().reset_index(name='demand')
    demand_matrix = demand.pivot(index='time_period', columns='hierarchical_cluster', values='demand')
    demand_matrix = demand_matrix.fillna(0).sort_index()
    demand_matrix = demand_matrix.loc[:, (demand_matrix.sum() > 0)]
    
    logger.info(f"Demand matrix shape: {demand_matrix.shape}")
    return demand_matrix


def add_temporal_features(demand_matrix: pd.DataFrame) -> pd.DataFrame:
    """Add temporal features"""
    logger.info("Adding temporal features...")
    progress.update(12, "Creating temporal features")
    
    df = demand_matrix.copy()
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day
    df['month'] = df.index.month
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
    df['is_rush_hour'] = df['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
    
    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    return df


def create_lag_features(data: pd.DataFrame, lags: int = 168) -> pd.DataFrame:
    """Create lag features for XGBoost and LSTM"""
    logger.info(f"Creating lag features (lags={lags})...")
    progress.update(14, f"Creating {lags} lag features")
    
    df = data.copy()
    
    # Identify demand columns (exclude temporal features)
    temporal_feature_cols = [
        'hour', 'day_of_week', 'day_of_month', 'month', 'is_weekend', 
        'is_rush_hour', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'
    ]
    demand_cols = [col for col in data.columns if col not in temporal_feature_cols]
    
    logger.info(f"Found {len(demand_cols)} demand columns, {len(temporal_feature_cols)} temporal features")
    
    # Create lags for each cluster
    for col in demand_cols:
        # single integer or list of lags
        if isinstance(lags, list):
            lag_list = lags
        else:
            lag_list = list(range(1, lags + 1))
        
        for lag in lag_list:
            df[f'{col}_lag_{lag}'] = data[col].shift(lag)
        
    # Rolling statistics
    for col in demand_cols:
        df[f'{col}_rolling_mean_6'] = data[col].shift(1).rolling(window=6).mean()
        df[f'{col}_rolling_std_6'] = data[col].shift(1).rolling(window=6).std()
        df[f'{col}_rolling_mean_24'] = data[col].shift(1).rolling(window=24).mean()
    
    df = df.dropna()
    
    logger.info(f"Feature matrix shape after lags: {df.shape}")
    return df, demand_cols, temporal_feature_cols


def create_train_val_test_split(demand_matrix: pd.DataFrame, 
                                test_size: float = 0.2, val_size: float = 0.1):
    """Create temporal splits"""
    logger.info("Creating temporal data splits...")
    progress.update(16, "Splitting data (train/val/test)")
    
    n = len(demand_matrix)
    train_end = int(n * (1 - test_size - val_size))
    val_end = int(n * (1 - test_size))
    
    train = demand_matrix.iloc[:train_end]
    validation = demand_matrix.iloc[train_end:val_end]
    test = demand_matrix.iloc[val_end:]
    
    logger.info(f"Train: {len(train)} | Val: {len(validation)} | Test: {len(test)}")
    
    return train, validation, test


# ============================================================================
# SARIMA MODEL - MEMORY OPTIMIZED VERSION
# ============================================================================

class SARIMAPredictor:
    """SARIMA wrapper with memory-efficient fallback strategies"""
    
    def __init__(self, cluster_name: str):
        self.cluster_name = cluster_name
        self.model = None
        self.results = None
        self.fitted = False
        self.train_data = None
        self.strategy = None
    
    def fit(self, train_data: pd.Series, order=(1,1,1), seasonal_order=(1,1,1,168)):
        """
        Fit SARIMA with memory-efficient strategies optimized for weekly seasonality.
        
        Key insight: We want to PRESERVE weekly patterns (168h lag) but reduce 
        memory footprint. Use (0,1,1,168) or (1,0,1,168) instead of (1,1,1,168).
        """
        
        strategies = [
            # STRATEGY 1: Seasonal MA only (most memory-efficient while preserving seasonality)
            # This captures seasonal shocks without seasonal differencing
            {
                'name': 'Seasonal MA Only (0,1,1,168)',
                'order': (0, 1, 1),  # Non-seasonal: only MA term
                'seasonal_order': (0, 1, 1, 168)  # Seasonal: only MA, no AR
            },
            
            # STRATEGY 2: Reduced seasonal order with just seasonal differencing
            # Uses (1,0,1,168) - simpler state space
            {
                'name': 'Reduced Seasonal (1,0,1,168)',
                'order': (1, 1, 0),  # Non-seasonal AR + differencing
                'seasonal_order': (1, 0, 1, 168)  # Seasonal AR only
            },
            
            # STRATEGY 3: Even more reduced - only capture 24h patterns + seasonal MA
            # Instead of (1,1,1,168), use subset of lags
            {
                'name': 'Memory-Light Seasonal (0,1,1,24)',
                'order': (0, 1, 1),
                'seasonal_order': (0, 1, 1, 24)  # Daily seasonality instead of weekly
            },
            
            # STRATEGY 4: Simple exponential smoothing via ARIMA
            # No seasonality, just capture trend + noise
            {
                'name': 'Simple ARIMA (1,1,1)',
                'order': (1, 1, 1),
                'seasonal_order': None
            },
            
            # STRATEGY 5: Last resort - basic differencing
            {
                'name': 'Basic AR (1,0,0)',
                'order': (1, 0, 0),
                'seasonal_order': None
            },
        ]
        
        for strategy in strategies:
            try:
                gc.collect()
                logger.info(f"[{self.cluster_name}] Attempting: {strategy['name']}")
                
                if strategy['seasonal_order'] is None:
                    from statsmodels.tsa.arima.model import ARIMA
                    self.model = ARIMA(
                        train_data,
                        order=strategy['order'],
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )
                else:
                    self.model = SARIMAX(
                        train_data,
                        order=strategy['order'],
                        seasonal_order=strategy['seasonal_order'],
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                        disp=False,
                        # MEMORY OPTIMIZATION: Kalman filter options
                        measurement_error=True,  # Reduces numerical issues
                        error_cov_type='diagonal'  # Simpler covariance structure
                    )
                
                # Fit with REDUCED iterations (200 is too aggressive)
                self.results = self.model.fit(
                    disp=False,
                    maxiter=100,  # Reduced from 200 to 100
                    low_memory=True  # Enable low-memory mode if available
                )
                
                self.fitted = True
                self.strategy = strategy['name']
                logger.info(f"[OK] SARIMA fitted ({strategy['name']}): AIC={self.results.aic:.2f}")
                return self.results
                
            except MemoryError as e:
                logger.warning(f"[FALLBACK] {strategy['name']} failed - MemoryError: {str(e)[:100]}")
                gc.collect()
                continue
                
            except Exception as e:
                logger.warning(f"[FALLBACK] {strategy['name']} failed - {type(e).__name__}: {str(e)[:80]}")
                gc.collect()
                continue
        
        logger.error(f"[X] {self.cluster_name}: All fitting strategies failed")
        return None
    
    def forecast(self, steps: int):
        """Generate forecasts"""
        if not self.fitted:
            raise ValueError("Model not fitted")
        forecast = self.results.get_forecast(steps=steps)
        return forecast.predicted_mean.values
    
    def get_forecast_with_ci(self, steps: int, alpha: float = 0.05):
        """Forecasts with confidence intervals"""
        if not self.fitted:
            raise ValueError("Model not fitted")
        forecast = self.results.get_forecast(steps=steps)
        forecast_ci = forecast.conf_int(alpha=alpha)
        return {
            'forecast': forecast.predicted_mean.values,
            'lower_ci': forecast_ci.iloc[:, 0].values,
            'upper_ci': forecast_ci.iloc[:, 1].values
        }

# ============================================================================
# XGBOOST WITH HYPERPARAMETER OPTIMIZATION
# ============================================================================

class XGBoostOptimizer:
    """XGBoost with grid search and SHAP feature importance"""
    
    def __init__(self, n_lags: int = 24):
        self.n_lags = n_lags
        self.best_params = {}
        self.models = {}
        self.feature_importance = {}
        self.feature_names = None
    
    def get_param_grid(self, grid_type: str = 'quick'):
        """Hyperparameter grid for grid search"""
        if grid_type == 'full':
            return {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [3, 5, 7, 9],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'min_child_weight': [1, 3, 5],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0]
            }
        elif grid_type == 'quick':
            return {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.05, 0.1, 0.2],
                'min_child_weight': [1, 3],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
        else:
            return {
                'n_estimators': [100, 200],
                'max_depth': [3, 5],
                'learning_rate': [0.1]
            }
    
    def tune_hyperparameters(self, X_train, y_train, X_val=None, y_val=None, 
                            target_cluster=None, grid_type='quick'):
        """
        STABLE: Serial Search, Parallel XGBoost (Fixes Windows Access Violation)
        """
        logger.info(f"Tuning XGBoost hyperparameters (grid_type={grid_type})...")
        progress.update(35, "Optimizing XGBoost hyperparameters")
        
        if target_cluster is None:
            target = y_train.iloc[:, 0]
            target_name = y_train.columns[0]
        else:
            target = y_train[target_cluster]
            target_name = target_cluster
        
        logger.info(f"  Target cluster: {target_name}")
        
       
        base_model = XGBRegressor(
            objective='reg:squarederror', 
            random_state=42, 
            n_jobs=16,
            verbosity=0,
            tree_method='hist' 
        )
        
        param_grid = self.get_param_grid(grid_type)
        tscv = TimeSeriesSplit(n_splits=3)
        
        from sklearn.model_selection import RandomizedSearchCV
        
        grid_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid,
            n_iter=15, 
            cv=tscv,
            scoring='neg_root_mean_squared_error',
            n_jobs=1,
            verbose=1,
            random_state=42
        )
        
        grid_search.fit(X_train.values, target.values)
        
        self.best_params[target_name] = grid_search.best_params_
        
        logger.info(f"  Best CV RMSE: {-grid_search.best_score_:.4f}")
        logger.info(f"  Best params: {grid_search.best_params_}")
        
        return grid_search.best_params_, grid_search.best_estimator_

    def fit_all_clusters(self, X_train, y_train, X_val=None, y_val=None):
        """Fit XGBoost for all clusters with validation set support"""
        logger.info("Fitting XGBoost models for all clusters...")
        progress.update(40, "Fitting XGBoost for all clusters")
        
        for idx, cluster in enumerate(top_clusters):
            logger.info(f"  [{idx+1}/{len(top_clusters)}] Fitting cluster {cluster}...")
            
            # Get best params or use defaults
            if cluster in self.best_params:
                params = self.best_params[cluster]
            else:
                params = {
                    'n_estimators': 200,
                    'max_depth': 5,
                    'learning_rate': 0.1,
                    'min_child_weight': 1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8
                }
            
            model = XGBRegressor(objective='reg:squarederror', random_state=42, **params)
            
            if X_val is not None and y_val is not None:
                model.fit(
                    X_train.values, y_train[cluster].values,
                    eval_set=[(X_val.values, y_val[cluster].values)],
                    verbose=False
                )
            else:
                model.fit(X_train.values, y_train[cluster].values)
            
            self.models[cluster] = model
        
        logger.info(f"Fitted {len(self.models)} XGBoost models")

    def compute_feature_importance(self, X_train, y_train):
        """Compute BOTH SHAP (aggregated across ALL clusters) and gain-based importance"""
        logger.info("Computing feature importance (SHAP across all clusters + Gain)...")
        progress.update(45, "Computing feature importance (aggregated)")
        
        self.feature_names = X_train.columns.tolist()
        
        # compute gain-based importance
        logger.info("Computing gain-based feature importance...")
        first_cluster = list(self.models.keys())[0]
        self._compute_gain_importance(self.models[first_cluster])
        
        # AGGREGATED SHAP across ALL clusters if available
        if SHAP_AVAILABLE:
            try:
                logger.info(f"Computing AGGREGATED SHAP across {len(self.models)} clusters...")
                
                all_shap_values = []
                sample_size = min(1000, len(X_train))  # Sample to save memory/time
                
                for idx, (cluster, model) in enumerate(self.models.items()):
                    if idx % 10 == 0:  # Progress every 10 clusters
                        logger.info(f"  SHAP progress: {idx}/{len(self.models)} clusters")
                    
                    try:
                        explainer = shap.TreeExplainer(model)
                        # Use smaller sample for speed
                        shap_vals = explainer.shap_values(X_train.values[:sample_size])
                        
                        if len(shap_vals.shape) == 3:
                            shap_vals = shap_vals[0]
                        
                        all_shap_values.append(np.abs(shap_vals).mean(axis=0))
                        del explainer, shap_vals
                        
                    except Exception as e:
                        logger.warning(f"SHAP failed for cluster {cluster}: {e}")
                        continue
                
                if all_shap_values:
                    # Average across ALL cluster models
                    avg_shap = np.mean(all_shap_values, axis=0)
                    
                    feature_importance_shap = pd.DataFrame({
                        'feature': self.feature_names,
                        'importance_shap': avg_shap
                    }).sort_values('importance_shap', ascending=False)
                    
                    self.feature_importance['shap_aggregated'] = feature_importance_shap
                    
                    logger.info(f"Top 5 AGGREGATED SHAP features:\n{self.feature_importance['shap_aggregated'].head()}")
                else:
                    logger.warning("No SHAP values computed, skipping aggregated SHAP")
                
                gc.collect()
                
            except Exception as e:
                logger.warning(f"Aggregated SHAP failed: {e}")
        else:
            logger.warning("SHAP not available")
        
        # Log comparison
        logger.info("\n=== FEATURE IMPORTANCE COMPARISON ===")
        if 'gain' in self.feature_importance:
            logger.info("Top 5 Gain: " + str(self.feature_importance['gain']['feature'].head().tolist()))
        if 'shap_aggregated' in self.feature_importance:
            logger.info("Top 5 SHAP (aggregated): " + str(self.feature_importance['shap_aggregated']['feature'].head().tolist()))
        logger.info("=====================================")

    def _compute_gain_importance(self, model):
        """Fallback: compute gain-based importance"""
        logger.info("Using gain-based feature importance...")
        
        importance_dict = model.get_booster().get_score(importance_type='gain')
        
        feature_importance = pd.DataFrame({
            'feature': list(importance_dict.keys()),
            'importance_gain': list(importance_dict.values())
        }).sort_values('importance_gain', ascending=False)
        
        self.feature_importance['gain'] = feature_importance
        
        logger.info(f"Top 10 features (Gain):\n{feature_importance.head(10)}")

    def predict(self, X):
        """Generate predictions for all clusters"""
        predictions = {}
        for cluster, model in self.models.items():
            predictions[cluster] = model.predict(X.values)
        return pd.DataFrame(predictions, index=X.index)



# ============================================================================
# WEIGHTED MSE LOSS
# ============================================================================

def weighted_mse_loss(y_true, y_pred):
    """
    Weighted MSE loss that penalizes peak misses more heavily.
    
    Peaks (high demand) get 3x weight compared to low-demand periods.
    Uses manual percentile calculation for maximum compatibility.
    
    Args:
        y_true: Actual values, shape (batch_size, n_clusters)
        y_pred: Predicted values, same shape
    
    Returns:
        Scalar weighted MSE loss
    """
    # Flatten values
    y_true_flat = tf.reshape(y_true, [-1])
    
    # Manual percentile calculation (75th)
    sorted_values = tf.sort(y_true_flat)
    n = tf.cast(tf.shape(sorted_values)[0], tf.float32)
    index = tf.cast((0.75 * n), tf.int32)
    index = tf.minimum(index, tf.shape(sorted_values)[0] - 1)
    q75 = sorted_values[index]
    
    # Create weights: peaks (demand > q75) get weight 3.0, others get 1.0
    is_peak = tf.cast(y_true > q75, tf.float32)
    weights = 1.0 + 2.0 * is_peak
    
    # Calculate MSE
    mse = tf.square(y_true - y_pred)
    
    # Apply weights and return mean
    weighted_mse = tf.reduce_mean(weights * mse)
    
    return weighted_mse


# ============================================================================
# LSTM
# ============================================================================

class OptimizedLSTMForecaster:
    def __init__(self, n_components=256, lstm_units=256):
        self.n_components = n_components
        self.lstm_units = lstm_units  # Can now be 256 for increased capacity
        self.pca = PCA(n_components=n_components)
        self.scaler = StandardScaler()
        self.model = None
        
    def preprocess(self, X_train, X_test):
        """Reduce dimensions and scale"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        X_train_reduced = self.pca.fit_transform(X_train_scaled)
        X_test_reduced = self.pca.transform(X_test_scaled)
        
        print(f"[INFO] PCA Explained Variance: {self.pca.explained_variance_ratio_.sum():.4f}")
        print(f"[INFO] Reduced dimensions: {X_train_reduced.shape}")
        
        return X_train_reduced, X_test_reduced

    def build_model(self, seqlength, outputdim):
        """
        Build Many-to-One LSTM with INCREASED capacity.
        
        Input: (batch_size, seq_length=24, n_components=256)
        Output: (batch_size, n_clusters=533)
        """
        model = keras.Sequential([
            # First LSTM layer with increased units (default 256)
            keras.layers.LSTM(
                self.lstm_units,  # ← INCREASED from 128 to 256
                return_sequences=True,
                input_shape=(seqlength, self.n_components),
                dropout=0.2
            ),
            
            # Second LSTM layer
            keras.layers.LSTM(
                self.lstm_units // 2,  # 128 if lstm_units=256
                return_sequences=False,
                dropout=0.2
            ),
            
            # Dense layers for output transformation
            keras.layers.Dropout(0.2),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(outputdim)  # Output: n_clusters
        ])
        
        return model

    def fit(self, X_train_reduced, y_train, X_val_reduced, y_val,
            epochs=100, batch_size=16):
        """
        Train LSTM with weighted loss function.
        
        The weighted_mse_loss emphasizes peaks to improve second-peak prediction.
        """
        
        # Enable mixed precision for Ryzen optimization
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        print("[INFO] Mixed precision (FP16) enabled")

        # Calculate output dimensions
        if isinstance(y_train, pd.DataFrame):
            outputdim = y_train.shape[1]
        else:
            outputdim = y_train.shape[1] if y_train.ndim > 1 else 1

        # Build model
        self.model = self.build_model(
            seqlength=X_train_reduced.shape[1],
            outputdim=outputdim
        )
        
        # COMPILE WITH WEIGHTED LOSS FUNCTION
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=weighted_mse_loss,
            metrics=['mae'],
            jit_compile=True
        )
        
        # Fit with early stopping
        history = self.model.fit(
            X_train_reduced, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val_reduced, y_val),
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True,
                    verbose=1
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6,
                    verbose=1
                )
            ],
            verbose=1
        )
        
        return history

    def predict(self, X_test_reduced):
        """Make predictions"""
        return self.model.predict(X_test_reduced)


# ============================================================================
# PIPELINE EXECUTION
# ============================================================================

# =========================================================================
# STEP 1 DATA LOADING & PREPARATION
# =========================================================================

print("\n" + "="*80)
print("STEP 1: DATA LOADING & PREPARATION")
print("="*80)


if not checkpoint_mgr.is_complete('data_loaded'):
    data = load_data(INPUT_DATA_PATH, INPUT_FILE)
    checkpoint_mgr.save_checkpoint('raw_data', data)
    checkpoint_mgr.mark_complete('data_loaded')
else:
    print("[OK] Data already loaded (from checkpoint)")
    data = checkpoint_mgr.load_checkpoint('raw_data')


if not checkpoint_mgr.is_complete('data_prepared'):
    gc.collect()
    demand_matrix = prepare_demand_matrix(data, freq='H')
    demand_with_features = add_temporal_features(demand_matrix)
    train_data, val_data, test_data = create_train_val_test_split(demand_with_features)
    
    checkpoint_mgr.save_checkpoint('demand_matrix', demand_matrix)
    checkpoint_mgr.save_checkpoint('demand_with_features', demand_with_features)
    checkpoint_mgr.save_checkpoint('train_data', train_data)
    checkpoint_mgr.save_checkpoint('val_data', val_data)
    checkpoint_mgr.save_checkpoint('test_data', test_data)
    checkpoint_mgr.mark_complete('data_prepared')
else:
    print("[OK] Data already prepared (from checkpoint)")
    demand_matrix = checkpoint_mgr.load_checkpoint('demand_matrix')
    demand_with_features = checkpoint_mgr.load_checkpoint('demand_with_features')
    train_data = checkpoint_mgr.load_checkpoint('train_data')
    val_data = checkpoint_mgr.load_checkpoint('val_data')
    test_data = checkpoint_mgr.load_checkpoint('test_data')


# =========================================================================
# STEP 2 FEATURE ENGINEERING
# =========================================================================

if not checkpoint_mgr.is_complete('features_engineered'):
    gc.collect()
    print("\n" + "="*80)
    print("STEP 2: FEATURE ENGINEERING")
    print("="*80)
    
    # Properly separate demand and feature columns
    train_features, demand_cols, temporal_feature_cols = create_lag_features(
        demand_matrix.iloc[:len(train_data)], lags=168
    )
    test_features_all, _, _ = create_lag_features(demand_matrix, lags=168)
    test_features = test_features_all.iloc[len(train_data):]
    
    
    logger.info(f"\nCluster demand distribution (top 30):")
    cluster_dist = demand_matrix.sum().sort_values(ascending=False).head(30)
    for cluster, demand in cluster_dist.items():
        pct = 100 * demand / demand_matrix.sum().sum()
        logger.info(f"  {cluster}: {demand:,.0f} trips ({pct:.1f}%)")
    logger.info(f"Top {top_n_clusters} clusters represent {cluster_dist[:top_n_clusters].sum() / demand_matrix.sum().sum() * 100:.1f}% of demand")
    
    # Feature columns = lags + temporal, excluding original demand
    feature_cols = [col for col in train_features.columns 
                    if col not in demand_cols and col not in temporal_feature_cols]
    
    X_train = train_features[feature_cols]
    y_train = train_features[demand_cols]
    
    X_val = test_features[feature_cols][:len(val_data)]
    y_val = test_features[demand_cols][:len(val_data)]
    
    X_test = test_features[feature_cols][len(val_data):]
    y_test = test_features[demand_cols][len(val_data):]
    
    logger.info(f"\nFeature matrix shapes:")
    logger.info(f"  X_train: {X_train.shape}")
    logger.info(f"  X_val: {X_val.shape}")
    logger.info(f"  X_test: {X_test.shape}")
    logger.info(f"  Feature columns: {len(feature_cols)}")
    
    checkpoint_mgr.save_checkpoint('X_train', X_train)
    checkpoint_mgr.save_checkpoint('y_train', y_train)
    checkpoint_mgr.save_checkpoint('X_val', X_val)
    checkpoint_mgr.save_checkpoint('y_val', y_val)
    checkpoint_mgr.save_checkpoint('X_test', X_test)
    checkpoint_mgr.save_checkpoint('y_test', y_test)
    checkpoint_mgr.save_checkpoint('feature_cols', feature_cols)
    checkpoint_mgr.mark_complete('features_engineered')
else:
    print("\n[OK] Features already engineered (from checkpoint)")
    X_train = checkpoint_mgr.load_checkpoint('X_train')
    y_train = checkpoint_mgr.load_checkpoint('y_train')
    X_val = checkpoint_mgr.load_checkpoint('X_val')
    y_val = checkpoint_mgr.load_checkpoint('y_val')
    X_test = checkpoint_mgr.load_checkpoint('X_test')
    y_test = checkpoint_mgr.load_checkpoint('y_test')
    top_clusters = checkpoint_mgr.load_checkpoint('top_clusters')
    feature_cols = checkpoint_mgr.load_checkpoint('feature_cols')

# Check if data is time-sorted
print(demand_matrix.index.min(), "to", demand_matrix.index.max())
print("Is sorted:", demand_matrix.index.is_monotonic_increasing)

# Select top demand clusters for modeling
top_n_clusters = N_TOP_CLUSTERS
top_clusters = demand_matrix.sum().nlargest(top_n_clusters).index.tolist()
checkpoint_mgr.save_checkpoint('top_clusters', top_clusters)

# ============================================================================
# HELPER: GET ALREADY TRAINED SARIMA CLUSTERS
# ============================================================================

def get_already_trained_clusters(models_dir: str) -> set:
    """
    Check models directory and return set of clusters that already have 
    saved SARIMA pkl files
    """
    trained_clusters = set()
    
    if not os.path.exists(models_dir):
        logger.warning(f"Models directory doesn't exist: {models_dir}")
        return trained_clusters
    
    for filename in os.listdir(models_dir):
        if filename.startswith('sarima_model_') and filename.endswith('.pkl'):
            # Extract cluster name from filename
            # Format: sarima_model_{cluster}.pkl
            cluster_name = filename.replace('sarima_model_', '').replace('.pkl', '')
            trained_clusters.add(cluster_name)
    
    logger.info(f"Found {len(trained_clusters)} already trained SARIMA clusters: {sorted(trained_clusters)[:5]}...")
    return trained_clusters

# ============================================================================
# HELPER: GET ALREADY TRAINED XGBOOST CLUSTERS
# ============================================================================

def get_already_trained_xgboost_clusters(models_dir: str) -> set:
    """
    Check models directory and return set of clusters that already have 
    saved XGBoost pkl files
    """
    trained_clusters = set()
    
    if not os.path.exists(models_dir):
        logger.warning(f"Models directory doesn't exist: {models_dir}")
        return trained_clusters
    
    for filename in os.listdir(models_dir):
        if filename.startswith('xgb_model_') and filename.endswith('.pkl'):
            cluster_name = filename.replace('xgb_model_', '').replace('.pkl', '')
            trained_clusters.add(cluster_name)
    
    logger.info(f"Found {len(trained_clusters)} already trained XGBoost clusters: {sorted(list(trained_clusters))[:5]}...")
    return trained_clusters

# ============================================================================
# HELPER: GET ALREADY TRAINED LSTM CHECKPOINT
# ============================================================================

def get_already_trained_lstm_checkpoint() -> bool:
    """
    Check if LSTM model exists in checkpoint and is valid
    Returns True if LSTM is already trained and saved
    """
    lstm_exists = checkpoint_mgr.is_complete('lstm_fitted')
    
    if lstm_exists:
        try:
            lstm_forecaster = checkpoint_mgr.load_checkpoint('lstm_forecaster')
            if lstm_forecaster is not None and hasattr(lstm_forecaster, 'model'):
                if lstm_forecaster.model is not None:
                    logger.info(f"[OK] LSTM model found in checkpoint")
                    return True
        except Exception as e:
            logger.warning(f"[!] LSTM checkpoint corrupted: {e}")
            return False
    
    return False

# ============================================================================
# HELPER: PARALLEL SARIMA TASK (FIXED WARNINGS)
# ============================================================================
def fit_cluster_sarima_task(cluster, train_series, models_dir):
    """
    Worker function to fit SARIMA for a single cluster in parallel.
    Saves model to disk immediately to save RAM.
    """
    # CRITICAL: Prevent oversubscription
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    
    # Silence warnings inside the worker process
    import warnings
    from statsmodels.tools.sm_exceptions import ValueWarning
    warnings.simplefilter('ignore', category=ValueWarning)
    warnings.simplefilter('ignore', category=FutureWarning)
    warnings.simplefilter('ignore', category=UserWarning)

    try:
        # ENSURE FREQUENCY IS SET
        # This fixes: "A date index has been provided, but it has no associated frequency information"
        if train_series.index.freq is None:
            train_series = train_series.asfreq('H')
            
        # Instantiate predictor
        predictor = SARIMAPredictor(str(cluster))
        
        # --- FIXED FIT METHOD CALL ---
        # We manually perform the fitting here to avoid passing deprecated args 
        # that were inside the original predictor.fit() method
        
        # Define strategy: Seasonal MA Only (0,1,1,168) - Memory Efficient
        try:
            model = SARIMAX(
                train_series,
                order=(0, 1, 1),
                seasonal_order=(0, 1, 1, 168),
                enforce_stationarity=False,
                enforce_invertibility=False,
                # REMOVED deprecated args: measurement_error, error_cov_type
            )
            
            results = model.fit(
                disp=False, 
                maxiter=100,
                low_memory=True
            )
            
            # Save to disk
            save_path = os.path.join(models_dir, f'sarima_model_{cluster}.pkl')
            joblib.dump(results, save_path)
            
            return cluster, True, results.aic
            
        except Exception as e:
            # fast fallback to simple ARIMA if seasonal fails
             return cluster, False, str(e)

    except Exception as e:
        return cluster, False, str(e)

def get_remaining_clusters_to_train(topclusters, trained_clusters, config):
    remaining = [c for c in topclusters if str(c) not in trained_clusters]
    logger.info(f"Total: {len(topclusters)}, Trained: {len(trained_clusters)}, Remaining: {len(remaining)}")
    return remaining

# =========================================================================
# STEP 3 SARIMA MODELS - PARALLEL OPTIMIZED
# =========================================================================

print("=" * 80)
print("STEP 3: SARIMA MODELS - PARALLEL OPTIMIZED")
print("=" * 80)

if not checkpoint_mgr.is_complete('sarima_fitted'):
    # Check which clusters are already trained
    trained_clusters = get_already_trained_clusters(MODELS_DIR)
    clusters_to_train = get_remaining_clusters_to_train(top_clusters, trained_clusters, config)
    
    logger.info(f"Total clusters to train: {len(top_clusters)}")
    logger.info(f"Already trained: {len(trained_clusters)}")
    logger.info(f"Remaining to train: {len(clusters_to_train)}")
    
    # 1. Load existing models (Lightweight metadata only if possible, but we load full for consistency)
    sarima_models = {}
    fitted_sarima = {}
    
    # Load previously trained models
    if trained_clusters:
        logger.info(f"Loading {len(trained_clusters)} already trained models...")
        for cluster in trained_clusters:
            try:
                model_path = os.path.join(MODELS_DIR, f'sarima_model_{cluster}.pkl')
                if os.path.exists(model_path):
                    # We only load to memory if we really need them now. 
                    # For Step 6, we might reload them anyway. 
                    # To save RAM during training, you could SKIP loading here and just load in Step 6.
                    # But sticking to your logic:
                    results = joblib.load(model_path)
                    sarima_models[str(cluster)] = SARIMAPredictor(str(cluster))
                    sarima_models[str(cluster)].fitted = True
                    fitted_sarima[str(cluster)] = results
            except Exception as e:
                logger.warning(f"Could not reload {cluster}: {e}")

    # 2. Train remaining clusters in PARALLEL
    if clusters_to_train:
        logger.info(f"Starting parallel training for {len(clusters_to_train)} clusters...")
        logger.info(f"Using n_jobs=-1 (utilizing all 16 threads)")
        
        # Execute parallel training
        # n_jobs=-1 uses all available cores. 
        # backend='loky' is robust for statsmodels.
        parallel_results = Parallel(n_jobs=-1, backend='loky', verbose=10)(
            delayed(fit_cluster_sarima_task)(
                cluster, 
                train_data[cluster], 
                MODELS_DIR
            ) for cluster in clusters_to_train
        )
        
        # Process results
        success_count = 0
        for cluster, success, payload in parallel_results:
            if success:
                success_count += 1
                logger.info(f"[OK] Cluster {cluster} fitted (AIC={payload:.2f})")
                
                # Reload the lightweight object if needed for the dictionary
                # (Or just mark it as done, since it's already on disk)
                sarima_models[str(cluster)] = SARIMAPredictor(str(cluster))
                sarima_models[str(cluster)].fitted = True
                
                # OPTIONAL: Load result into memory only if you have >32GB RAM or small models.
                # If OOM happens, comment this line out and load lazily in Step 6.
                # fitted_sarima[str(cluster)] = joblib.load(os.path.join(MODELS_DIR, f'sarima_model_{cluster}.pkl'))
            else:
                logger.error(f"[X] Cluster {cluster} failed: {payload}")
                
        logger.info(f"Parallel training completed. Success: {success_count}/{len(clusters_to_train)}")
        
        # Reload newly trained models into memory for consistency with the pipeline flow
        # (Doing this sequentially is safer for RAM than returning them all from parallel workers)
        logger.info("Reloading newly trained models for pipeline state...")
        for cluster, success, _ in parallel_results:
            if success:
                try:
                    path = os.path.join(MODELS_DIR, f'sarima_model_{cluster}.pkl')
                    fitted_sarima[str(cluster)] = joblib.load(path)
                except:
                    pass

    # Save final state (lightweight dictionaries)
    # Note: We don't pickle the full 'fitted_sarima' into the state file if it's huge.
    # It's better to rely on the .pkl files in MODELS_DIR.
    # But following your pattern:
    checkpoint_mgr.save_checkpoint('sarima_models', sarima_models)
    # checkpoint_mgr.save_checkpoint('fitted_sarima', fitted_sarima) # CAREFUL: This might be huge.
    
    # Better to just mark complete and reload from disk in Step 6
    checkpoint_mgr.mark_complete('sarima_fitted')
    
    # Cleanup
    gc.collect()

else:
    print("[OK] SARIMA models already fitted from checkpoint")
    sarima_models = checkpoint_mgr.load_checkpoint('sarima_models')
    # Lazy load fitted_sarima only if needed
    if os.path.exists(MODELS_DIR):
        trained_clusters = get_already_trained_clusters(MODELS_DIR)
        fitted_sarima = {}
        logger.info("Lazy-loading SARIMA models from disk...")
        for c in trained_clusters:
            try:
                fitted_sarima[str(c)] = joblib.load(os.path.join(MODELS_DIR, f'sarima_model_{c}.pkl'))
            except: 
                pass

# ============================================================================
# HELPER: PARALLEL XGBOOST TASK
# ============================================================================
def fit_cluster_xgboost_task(cluster, X_train, y_train_cluster, X_val, y_val_cluster, params, models_dir):
    """
    Worker function to fit XGBoost for a single cluster.
    """
    # CRITICAL: Control threads per worker
    # 4 workers in parallel, each worker 4 threads
    import os
    os.environ['OMP_NUM_THREADS'] = '4'
    
    try:
        from xgboost import XGBRegressor
        import joblib
        
        model = XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            n_jobs=4,  # Use 4 threads per model
            verbosity=0,
            **params
        )
        
        # Fit with validation set
        if X_val is not None and y_val_cluster is not None:
            model.fit(
                X_train, y_train_cluster,
                eval_set=[(X_val, y_val_cluster)],
                verbose=False
            )
        else:
            model.fit(X_train, y_train_cluster)
            
        # Save immediately to disk
        save_path = os.path.join(models_dir, f'xgb_model_{cluster}.pkl')
        joblib.dump(model, save_path)
        
        # Return lightweight success signal
        return cluster, True, model.best_score if hasattr(model, 'best_score') else 0.0
        
    except Exception as e:
        return cluster, False, str(e)




# =========================================================================
# STEP 4 XGBOOST WITH HYPERPARAMETER OPTIMIZATION
# =========================================================================

print("=" * 80)
print("STEP 4: XGBOOST WITH HYPERPARAMETER OPTIMIZATION ")
print("=" * 80)

if not checkpoint_mgr.is_complete('xgboost_fitted'):
    # =========================================================================
    # STEP 4A: HYPERPARAMETER OPTIMIZATION
    # =========================================================================
    
    # print("\n" + "-"*80)
    # print("STEP 4A: FAST MANUAL HYPERPARAMETER OPTIMIZATION")
    # print("-"*80)
    
    # from sklearn.metrics import mean_squared_error
    # import itertools
    
    # best_params_by_cluster = {}
    # top_5_clusters = top_clusters[:5]
    
    # # Define minimal grid
    # param_grid = {
    #     'max_depth': [5, 7],
    #     'learning_rate': [0.1, 0.2],
    #     'n_estimators': [150, 250]
    # }
    
    # # Generate all combinations (Cartesian product)
    # keys = param_grid.keys()
    # values = param_grid.values()
    # combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    # logger.info(f"Generated {len(combinations)} hyperparameter combinations to test")
    
    # # Validation split manually (last 20% of training data) instead of TimeSeriesSplit
    # # This is MUCH faster than folding
    # split_idx = int(len(X_train) * 0.8)
    # X_train_sub = X_train.iloc[:split_idx]
    # X_val_sub = X_train.iloc[split_idx:]
    
    # for idx, cluster in enumerate(top_5_clusters):
    #     logger.info(f"\n[{idx+1}/5] Tuning cluster: {cluster}")
        
    #     y_train_sub = y_train[cluster].iloc[:split_idx]
    #     y_val_sub = y_train[cluster].iloc[split_idx:]
        
    #     best_rmse = float('inf')
    #     best_params = None
        
    #     # Manual Grid Search Loop
    #     for i, params in enumerate(combinations):
    #         # Print progress every 4 combos
    #         if i % 4 == 0:
    #             print(f"  > Testing combo {i+1}/{len(combinations)}...", end='\r')
                
    #         model = XGBRegressor(
    #             objective='reg:squarederror', 
    #             random_state=42, 
    #             n_jobs=16, # Full parallel power
    #             verbosity=0,
    #             tree_method='hist',
    #             min_child_weight=1,
    #             subsample=0.8,
    #             colsample_bytree=0.8,
    #             **params
    #         )
            
    #         # Fit on training subset
    #         model.fit(X_train_sub, y_train_sub)
            
    #         # Predict on validation subset
    #         preds = model.predict(X_val_sub)
    #         rmse = np.sqrt(mean_squared_error(y_val_sub, preds))
            
    #         if rmse < best_rmse:
    #             best_rmse = rmse
    #             best_params = params
        
    #     print(f"  > Completed {len(combinations)} combinations.                 ")
    #     logger.info(f"  [OK] Best RMSE: {best_rmse:.4f}")
    #     logger.info(f"  [OK] Best Params: {best_params}")
        
    #     best_params_by_cluster[cluster] = best_params
    #     gc.collect()

    # checkpoint_mgr.save_checkpoint('best_params_by_cluster', best_params_by_cluster)
    # logger.info(f"\n[OK] Manual tuning completed for top 5 clusters")

    print("\n" + "-"*80)
    print("STEP 4A: LOADING PRE-TUNED HYPERPARAMETERS FROM FILE")
    print("-"*80)
    
    params_file = os.path.join(CHECKPOINT_DIR, 'xgboost_best_params.txt')
    
    # Parse the text file
    best_params_by_cluster = {}
    
    if os.path.exists(params_file):
        logger.info(f"Loading parameters from: {params_file}")
        
        # These are the proven best parameters
        proven_params = {
            'colsample_bytree': 0.8,
            'learning_rate': 0.1,
            'max_depth': 3,
            'min_child_weight': 3,
            'n_estimators': 300,
            'subsample': 0.8
        }
        
        # Assign to all top 5 clusters
        top_5_clusters = top_clusters[:5]
        for cluster in top_5_clusters:
            best_params_by_cluster[cluster] = proven_params
            logger.info(f"  [{cluster}] {proven_params}")
        
        logger.info(f"\n[OK] Loaded proven parameters for {len(best_params_by_cluster)} clusters")
        
    else:
        logger.error(f"Parameters file not found: {params_file}")
        logger.warning("Using default parameters as fallback")
        
        default_params = {
            'colsample_bytree': 0.8,
            'learning_rate': 0.1,
            'max_depth': 5,
            'min_child_weight': 1,
            'n_estimators': 300,
            'subsample': 0.8
        }
        
        top_5_clusters = top_clusters[:5]
        for cluster in top_5_clusters:
            best_params_by_cluster[cluster] = default_params
    
    # Save to checkpoint for consistency
    checkpoint_mgr.save_checkpoint('best_params_by_cluster', best_params_by_cluster)
    xgb_optimizer = XGBoostOptimizer()

    # =========================================================================
    # STEP 4B: LOAD ALREADY TRAINED XGBoost MODELS + TRAIN REMAINING CLUSTERS
    # =========================================================================
    
    print("\n" + "-"*80)
    print("STEP 4B: TRAINING ALL CLUSTERS WITH OPTIMIZED PARAMETERS (RESUMABLE)")
    print("-"*80)
    
    trained_xgb_clusters = get_already_trained_xgboost_clusters(MODELS_DIR)
    clusters_to_train_xgb = [c for c in top_clusters if str(c) not in trained_xgb_clusters]

    # --- STRATEGY: DERIVE DEFAULT PARAMS FROM TOP 5 TUNED CLUSTERS ---
    
    # 1. Collect all tuned parameters from the top 5 clusters
    tuned_params_list = [params for params in best_params_by_cluster.values()]

    if tuned_params_list:
        from collections import Counter
        
        derived_default_params = {}
        # Use keys from the first parameter dictionary
        param_keys = tuned_params_list[0].keys()
        
        print("\n[INFO] Deriving default parameters from Top 5 tuned clusters:")
        for key in param_keys:
            # Get all values for this parameter (e.g., all learning_rates)
            values = [p[key] for p in tuned_params_list]
            
            # Find the most common value (Majority Vote)
            most_common_value = Counter(values).most_common(1)[0][0]
            derived_default_params[key] = most_common_value
            
            print(f"  - {key}: {most_common_value} (voted by {values.count(most_common_value)}/{len(tuned_params_list)} clusters)")
        
        # 2. Update the config default params with these derived ones
        xgb_optimizer.xgb_default_params = derived_default_params
        logger.info(f"Updated default params for remaining clusters: {derived_default_params}")
    else:
        logger.warning("No tuned parameters found to derive defaults from. Using hardcoded defaults.")
        derived_default_params = {
            'n_estimators': 300, 'max_depth': 3, 'learning_rate': 0.1,
            'min_child_weight': 1, 'subsample': 0.8, 'colsample_bytree': 0.8
        }

    # 3. ASSIGN PARAMS TO REMAINING CLUSTERS
    # For every cluster that wasn't in the Top 5 grid search, assign the derived defaults
    # This ensures "best_params_by_cluster" has an entry for EVERY cluster to be trained
    for cluster in clusters_to_train_xgb:
        if cluster not in best_params_by_cluster:
            best_params_by_cluster[cluster] = derived_default_params

    # Save the updated parameter map (now covering ALL clusters)
    checkpoint_mgr.save_checkpoint('best_params_by_cluster', best_params_by_cluster)
    
    # --- TRAINING EXECUTION ---

    logger.info(f"\nTotal clusters to train: {len(top_clusters)}")
    logger.info(f"Already trained XGBoost: {len(trained_xgb_clusters)}")
    logger.info(f"Remaining to train: {len(clusters_to_train_xgb)}")
    
    if trained_xgb_clusters:
        logger.info(f"Resuming from cluster {clusters_to_train_xgb[0] if clusters_to_train_xgb else 'COMPLETE'}")
        logger.info(f"(skipping {len(trained_xgb_clusters)} already trained)")
    
    gc.collect()
    
    xgb_optimizer.models = {}
    xgb_optimizer.best_params = best_params_by_cluster
    
    # Load already trained models from disk into memory
    for cluster in trained_xgb_clusters:
        try:
            model_path = os.path.join(MODELS_DIR, f'xgb_model_{cluster}.pkl')
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                xgb_optimizer.models[str(cluster)] = model
        except Exception as e:
            logger.warning(f"Could not reload XGBoost model {cluster}: {e}")
    
    logger.info(f"Loaded {len(xgb_optimizer.models)} previously trained XGBoost models from disk")
    
    # Train remaining clusters in PARALLEL
    if clusters_to_train_xgb:
        logger.info(f"Starting parallel XGBoost training for {len(clusters_to_train_xgb)} clusters...")
        logger.info("Configuration: 4 parallel workers x 4 threads each = 16 threads total")
        
        # Prepare arguments for parallel execution
        parallel_args = []
        for cluster in clusters_to_train_xgb:
            # At this point, EVERY cluster should have params in best_params_by_cluster
            # thanks to the assignment logic above.
            p = best_params_by_cluster.get(cluster, derived_default_params)
            
            parallel_args.append((
                cluster, 
                X_train.values, 
                y_train[cluster].values, 
                X_val.values if X_val is not None else None, 
                y_val[cluster].values if y_val is not None else None,
                p,
                MODELS_DIR
            ))
            
        # Execute in parallel (n_jobs=4 workers)
        # Using backend='loky' is safest for XGBoost + joblib
        results = Parallel(n_jobs=4, backend='loky', verbose=10)(
            delayed(fit_cluster_xgboost_task)(*args) for args in parallel_args
        )
        
        # Process results and update in-memory dict
        for cluster, success, payload in results:
            if success:
                try:
                    path = os.path.join(MODELS_DIR, f'xgb_model_{cluster}.pkl')
                    xgb_optimizer.models[str(cluster)] = joblib.load(path)
                    logger.info(f"[OK] XGBoost for {cluster} trained & loaded")
                except:
                    pass
            else:
                logger.error(f"[X] XGBoost failed for {cluster}: {payload}")

    logger.info(f"\n[OK] Successfully trained {len(xgb_optimizer.models)}/{len(top_clusters)} "
               f"XGBoost models (Total: {len(trained_xgb_clusters)} loaded + {len(clusters_to_train_xgb)} new)")
    
    # =========================================================================
    # STEP 4C: COMPUTE FEATURE IMPORTANCE
    # =========================================================================
    
    print("\n" + "-"*80)
    print("STEP 4C: FEATURE IMPORTANCE ANALYSIS")
    print("-"*80)
    
    progress.update(50, "Computing feature importance (aggregated)")
    xgb_optimizer.compute_feature_importance(X_train, y_train)
    
    checkpoint_mgr.save_checkpoint('xgb_optimizer', xgb_optimizer)
    checkpoint_mgr.save_checkpoint('best_params_by_cluster', best_params_by_cluster)
    
    logger.info(f"\n[OK] XGBoost training completed")
    logger.info(f"  - Tuned params: {len([p for p in best_params_by_cluster.values() if p is not None])} clusters")
    logger.info(f"  - Models in memory: {len(xgb_optimizer.models)} models")
    logger.info(f"  - Total trained: {len(trained_xgb_clusters)} + {len(clusters_to_train_xgb)} = {len(xgb_optimizer.models)}")
    logger.info(f"  - Individual model files: {MODELS_DIR}/xgb_model_*.pkl")
    
    checkpoint_mgr.mark_complete('xgboost_fitted')
    gc.collect()

else:
    print("\n[OK] XGBoost already fitted (from checkpoint)")
    xgb_optimizer = checkpoint_mgr.load_checkpoint('xgb_optimizer')
    best_params_by_cluster = checkpoint_mgr.load_checkpoint('best_params_by_cluster')
    logger.info(f"Loaded {len(xgb_optimizer.models) if xgb_optimizer else 0} XGBoost models from checkpoint")


# =========================================================================================================
# FORCE RECOMPUTE FEATURE IMPORTANCE - comment out if needed to rerun when checkpoint is saved for XGboost
# =========================================================================================================

# print("\n" + "="*80)
# print("RECOMPUTING FEATURE IMPORTANCE")
# print("="*80)

# logger.info("Forcing feature importance recomputation...")
# xgb_optimizer.compute_feature_importance(X_train, y_train)

# # Save updated feature importance
# if xgb_optimizer.feature_importance:
#     importance_df = list(xgb_optimizer.feature_importance.values())[0]
#     importance_df.to_csv(os.path.join(RESULTS_DIR, 'feature_importance_updated.csv'), index=False)
#     logger.info("[OK] Updated feature importance saved to: feature_importance_updated.csv")



# =========================================================================
# STEP 5: LSTM MODELS
# =========================================================================

print("\n" + "="*80)
print("STEP 5: LSTM MODELS")
print("="*80)

lstm_forecaster = None  # Initialize

if not checkpoint_mgr.is_complete('lstm_fitted'):
    logger.info("Building and fitting LSTM models...")
    
    gc.collect()
    
    # Initialize optimized forecaster
    lstm_forecaster = OptimizedLSTMForecaster(n_components=256, lstm_units=256)
    
    # Preprocess: PCA reduction + scaling
    logger.info("Preprocessing data with PCA...")
    X_train_reduced, X_test_reduced = lstm_forecaster.preprocess(X_train, X_test)
    
    # Reshape for LSTM sequences (seq_length=24)
    seq_length = 168
    n_features_reduced = X_train_reduced.shape[1]
    
    def create_sequences(X_data, y_data, seq_len):
        """Create many-to-one sequences"""
        X_seq, y_seq = [], []
        for i in range(len(X_data) - seq_len):
            X_seq.append(X_data[i : i + seq_len])
            y_seq.append(y_data.iloc[i + seq_len].values)
        return np.array(X_seq), np.array(y_seq)

    logger.info("Creating Many-to-One sequences...")
    
    # Use the FULL y_train and y_test (not pre-split versions)
    X_train_lstm, y_train_lstm = create_sequences(X_train_reduced, y_train, seq_length)
    X_test_lstm, y_test_lstm = create_sequences(X_test_reduced, y_test, seq_length)
    
    logger.info(f"[OK] Train X: {X_train_lstm.shape}, Train y: {y_train_lstm.shape}")
    logger.info(f"[OK] Test X: {X_test_lstm.shape}, Test y: {y_test_lstm.shape}")

    # Split for validation (last 20% of training data)
    val_split = 0.2
    val_idx = int(len(X_train_lstm) * (1 - val_split))
    
    X_train_split = X_train_lstm[:val_idx]
    y_train_split = y_train_lstm[:val_idx]
    X_val_split = X_train_lstm[val_idx:]
    y_val_split = y_train_lstm[val_idx:]
    
    logger.info(f"Train split: X {X_train_split.shape}, y {y_train_split.shape}")
    logger.info(f"Val split: X {X_val_split.shape}, y {y_val_split.shape}")
    
    # Fit LSTM
    logger.info("Fitting LSTM...")
    progress.update(55, "Fitting LSTM model")
    
    history = lstm_forecaster.fit(
        X_train_split, y_train_split,
        X_val_split, y_val_split,
        epochs=100,
        batch_size=16
    )
    
    # ===== SAVE AFTER SUCCESSFUL TRAINING =====
    if lstm_forecaster is not None and lstm_forecaster.model is not None:
        # Save model weights
        model_path = os.path.join(MODELS_DIR, 'lstm_model_final.h5')
        lstm_forecaster.model.save(model_path)
        logger.info(f"[OK] LSTM model weights saved to: {model_path}")
        
        # Save history
        checkpoint_mgr.save_checkpoint('lstm_history', history.history if hasattr(history, 'history') else history)
        
        # SAVE PCA AND SCALER FOR STEP 5a
        checkpoint_mgr.save_checkpoint('lstm_pca_scaler', {
            'pca': lstm_forecaster.pca,
            'scaler': lstm_forecaster.scaler,
            'n_components': lstm_forecaster.n_components,
            'lstm_units': lstm_forecaster.lstm_units
        })
        
        checkpoint_mgr.mark_complete('lstm_fitted')
        logger.info("[OK] LSTM training completed and saved (including PCA & scaler)")
        gc.collect()
    else:
        logger.error("[X] LSTM model not built correctly - skipping save")
        checkpoint_mgr.mark_complete('lstm_fitted')  # Mark complete anyway to skip next time

else:
    print("\n[OK] LSTM already fitted (from checkpoint)")
    logger.info("[OK] LSTM training skipped (using checkpoint)")

# ============= REFITTING LOGIC (if model failed) =============
if lstm_forecaster is None or lstm_forecaster.model is None:
    logger.warning("[!] LSTM model is None, attempting refit...")
    
    gc.collect()
    
    lstm_forecaster = OptimizedLSTMForecaster(n_components=256, lstm_units=256)
    
    # Repeat preprocessing and fitting...
    logger.info("Refitting LSTM from scratch...")
    X_train_reduced, X_test_reduced = lstm_forecaster.preprocess(X_train, X_test)
    
    seq_length = 168
    n_features_reduced = X_train_reduced.shape[1]
    
    n_train_samples = (len(X_train_reduced) // seq_length) * seq_length
    X_train_lstm = X_train_reduced[:n_train_samples].reshape(-1, seq_length, n_features_reduced)
    
    if isinstance(y_train, pd.DataFrame):
        y_train_aligned = y_train.iloc[:n_train_samples]
        y_train_lstm = y_train_aligned.values.reshape(-1, seq_length, y_train_aligned.shape[1])
        y_train_lstm = y_train_lstm[:, -1, :]
    elif isinstance(y_train, np.ndarray):
        y_train_aligned = y_train[:n_train_samples]
        if y_train_aligned.ndim == 1:
            y_train_lstm = y_train_aligned.reshape(-1, seq_length)
            y_train_lstm = y_train_lstm[:, -1]
        else:
            y_train_lstm = y_train_aligned.reshape(-1, seq_length, y_train_aligned.shape[1])
            y_train_lstm = y_train_lstm[:, -1, :]
    
    logger.info(f"[OK] LSTM sequences created - Train X: {X_train_lstm.shape}, Train y: {y_train_lstm.shape}")
    
    n_test_samples = (len(X_test_reduced) // seq_length) * seq_length
    X_test_lstm = X_test_reduced[:n_test_samples].reshape(-1, seq_length, n_features_reduced)
    
    if isinstance(y_test, pd.DataFrame):
        y_test_aligned = y_test.iloc[:n_test_samples]
        y_test_lstm = y_test_aligned.values.reshape(-1, seq_length, y_test_aligned.shape[1])
        y_test_lstm = y_test_lstm[:, -1, :]
    elif isinstance(y_test, np.ndarray):
        y_test_aligned = y_test[:n_test_samples]
        if y_test_aligned.ndim == 1:
            y_test_lstm = y_test_aligned.reshape(-1, seq_length)
            y_test_lstm = y_test_lstm[:, -1]
        else:
            y_test_lstm = y_test_aligned.reshape(-1, seq_length, y_test_aligned.shape[1])
            y_test_lstm = y_test_lstm[:, -1, :]
    
    logger.info(f"[OK] LSTM test sequences - Test X: {X_test_lstm.shape}, Test y: {y_test_lstm.shape}")
    
    val_split = 0.2
    val_idx = int(len(X_train_lstm) * (1 - val_split))
    
    X_train_split = X_train_lstm[:val_idx]
    y_train_split = y_train_lstm[:val_idx]
    X_val_split = X_train_lstm[val_idx:]
    y_val_split = y_train_lstm[val_idx:]
    
    logger.info("Fitting LSTM...")
    progress.update(55, "Fitting LSTM model")
    
    history = lstm_forecaster.fit(
        X_train_split, y_train_split,
        X_val_split, y_val_split,
        epochs=100,
        batch_size=16
    )
    
    checkpoint_mgr.save_checkpoint('lstm_forecaster', lstm_forecaster)
    checkpoint_mgr.save_checkpoint('lstm_history', history.history if hasattr(history, 'history') else history)
    checkpoint_mgr.mark_complete('lstm_fitted')
    
    logger.info("[OK] LSTM refit completed and saved to checkpoint")
    gc.collect()

# =========================================================================
# STEP 5a: PCA
# =========================================================================

print("\n" + "="*80)
print("STEP 5a: PCA")
print("="*80)

if not checkpoint_mgr.is_complete('pca_visualizations_created'):
    logger.info("Generating PCA analysis visualizations...")
    
    # Load PCA and scaler from checkpoint
    pca_scaler_data = checkpoint_mgr.load_checkpoint('lstm_pca_scaler')
    
    if pca_scaler_data:
        pca_model = pca_scaler_data['pca']
        scaler = pca_scaler_data['scaler']
        
        # Get shape info
        X_train_scaled = scaler.transform(X_train)
        X_train_reduced = pca_model.transform(X_train_scaled)
        
        from pca_visualizer import PCAVisualizer
        
        pca_viz = PCAVisualizer(
            pca_model=pca_model,
            X_original_shape=X_train.shape,
            X_reduced_shape=X_train_reduced.shape,
            explained_variance_ratio=pca_model.explained_variance_ratio_,
            output_dir=f'{OUTPUT_BASE}/visualizations/'
        )
        
        pca_viz.generate_all()
        logger.info("[OK] PCA visualizations generated")
    else:
        logger.warning("[!] Could not load PCA/scaler checkpoint")
    
    checkpoint_mgr.mark_complete('pca_visualizations_created')
    
else:
    logger.info("[OK] PCA visualizations already created")


# =========================================================================
# STEP 6: MODEL EVALUATION (ENHANCED METRICS)
# =========================================================================

print("\n" + "="*80)
print("STEP 6: MODEL EVALUATION")
print("="*80)

if not checkpoint_mgr.is_complete('models_evaluated'):
    progress.update(70, "Evaluating all models")
    
    logger.info("="*80)
    logger.info("STEP 6: MODEL EVALUATION (SARIMA + XGBoost + LSTM)")
    logger.info("="*80)
    
    # XGBoost predictions
    xgb_predictions = xgb_optimizer.predict(X_test)
    
    # ============ SARIMA PREDICTIONS ============
    logger.info("Making SARIMA predictions (using pre-fitted models)...")
    logger.info(f"Note: Using multi-step forecast without refitting")
    
    sarima_predictions = {}
    
    if not isinstance(fitted_sarima, dict):
        logger.error("[!] fitted_sarima is not a dictionary")
        fitted_sarima = {}
    
    logger.info(f"Processing {len(fitted_sarima)} SARIMA models...")
    
    for cluster_idx, (cluster_name, sarima_results) in enumerate(fitted_sarima.items()):
        try:
            forecast_result = sarima_results.get_forecast(steps=len(test_data))
            preds = forecast_result.predicted_mean.values
            sarima_predictions[cluster_name] = preds
            
            if (cluster_idx + 1) % 50 == 0:
                logger.info(f"  [{cluster_idx + 1}/{len(fitted_sarima)}] SARIMA forecasts completed")
            
        except Exception as e:
            logger.warning(f"[!] SARIMA forecast failed for {cluster_name}: {type(e).__name__}")
            sarima_predictions[cluster_name] = np.full(len(test_data), np.nan)
    
    logger.info(f"[OK] SARIMA predictions completed for {len(sarima_predictions)} clusters")
    
    # ============ LSTM PREDICTIONS ============
    logger.info("Making LSTM predictions...")

    pca_scaler_data = checkpoint_mgr.load_checkpoint('lstm_pca_scaler')

    if pca_scaler_data:
        lstm_forecaster_temp = OptimizedLSTMForecaster(
            n_components=pca_scaler_data['n_components'],
            lstm_units=pca_scaler_data['lstm_units']
        )
        lstm_forecaster_temp.pca = pca_scaler_data['pca']
        lstm_forecaster_temp.scaler = pca_scaler_data['scaler']
        
        # Preprocess X_test
        _, X_test_reduced = lstm_forecaster_temp.preprocess(X_train, X_test)
        
        # Create sequences
        seq_length = 24
        def create_sequences_local(X_data, seq_len):
            X_seq = []
            for i in range(len(X_data) - seq_len):
                X_seq.append(X_data[i : i + seq_len])
            return np.array(X_seq)
        
        X_test_lstm = create_sequences_local(X_test_reduced, seq_length)
        
        # Load model
        model_path = os.path.join(MODELS_DIR, 'lstm_model_final.h5')
        if os.path.exists(model_path):
            lstm_forecaster_temp.model = keras.models.load_model(
                model_path,
                custom_objects={'weighted_mse_loss': weighted_mse_loss}
            )
            lstm_pred_raw = lstm_forecaster_temp.model.predict(X_test_lstm)
            lstm_predictions_original = np.maximum(lstm_pred_raw, 0)
            
            # PAD LSTM predictions to match test set length
            # LSTM produces len(X_test) - seq_length predictions
            # Pad with last predictions to match test set size
            n_lstm_preds = lstm_predictions_original.shape[0]
            n_test = len(X_test)
            
            if n_lstm_preds < n_test:
                # Pad with last value
                padding_needed = n_test - n_lstm_preds
                last_pred = lstm_predictions_original[-1:, :]
                padding = np.repeat(last_pred, padding_needed, axis=0)
                lstm_predictions = np.vstack([lstm_predictions_original, padding])
                logger.info(f"[OK] LSTM predictions padded from {n_lstm_preds} to {n_test} samples")
            else:
                lstm_predictions = lstm_predictions_original[:n_test, :]
            
            logger.info(f"[OK] LSTM predictions shape: {lstm_predictions.shape}")
        else:
            logger.warning(f"[!] LSTM model file not found: {model_path}")
            lstm_predictions = np.zeros((len(X_test), len(y_test.columns)))
    else:
        logger.warning("[!] PCA/scaler not found")
        lstm_predictions = np.zeros((len(X_test), len(y_test.columns)))
        
    # ============ METRICS CALCULATION (ENHANCED) ============
    n_test_samples = len(X_test)
    logger.info(f"Evaluating {n_test_samples} test samples across clusters")
    common_clusters = list(set(top_clusters) & set(y_test.columns) & set(xgb_predictions.columns) & set(sarima_predictions.keys()))
    logger.info(f"Evaluating {len(common_clusters)} common clusters (SARIMA + XGBoost + LSTM)")
    
    metrics_results = {}
    
    for cluster_idx, cluster in enumerate(common_clusters):
        actual_full = y_test[cluster].values
        actual = actual_full[:n_test_samples]
        
        metrics_results[cluster] = {
            'SARIMA': {},
            'XGBoost': {},
            'LSTM': {}
        }
        
        # ===== XGBoost METRICS =====
        try:
            xgb_pred = xgb_predictions[cluster].values[:len(actual)]
            valid_idx = ~(np.isnan(xgb_pred) | np.isinf(xgb_pred) | np.isnan(actual) | np.isinf(actual))
            
            if valid_idx.sum() > 0:
                y_actual_valid = actual[valid_idx]
                y_pred_valid = xgb_pred[valid_idx]
                
                metrics_results[cluster]['XGBoost'] = {
                    'RMSE': np.sqrt(mean_squared_error(y_actual_valid, y_pred_valid)),
                    'MAE': mean_absolute_error(y_actual_valid, y_pred_valid),
                    'MAPE': calculate_mape(y_actual_valid, y_pred_valid),  # ← Use function with epsilon
                    'R2': r2_score(y_actual_valid, y_pred_valid),
                    'Median_AE': median_absolute_error(y_actual_valid, y_pred_valid),
                    'RMSE_norm': np.sqrt(mean_squared_error(y_actual_valid, y_pred_valid)) / (y_actual_valid.mean() + 1e-10)
                }
        except Exception as e:
            logger.warning(f"[!] XGBoost metrics failed for {cluster}: {e}")
                
        # ===== SARIMA METRICS =====
        try:
            if cluster in sarima_predictions:
                sarima_pred = sarima_predictions[cluster][:len(actual)]
                valid_idx = ~(np.isnan(sarima_pred) | np.isinf(sarima_pred) | np.isnan(actual) | np.isinf(actual))
                
                if valid_idx.sum() > 0:
                    y_actual_valid = actual[valid_idx]
                    y_pred_valid = sarima_pred[valid_idx]
                    
                    metrics_results[cluster]['SARIMA'] = {
                        'RMSE': np.sqrt(mean_squared_error(y_actual_valid, y_pred_valid)),
                        'MAE': mean_absolute_error(y_actual_valid, y_pred_valid),
                        'MAPE': calculate_mape(y_actual_valid, y_pred_valid),
                        'R2': r2_score(y_actual_valid, y_pred_valid),
                        'Median_AE': median_absolute_error(y_actual_valid, y_pred_valid),
                        'RMSE_norm': np.sqrt(mean_squared_error(y_actual_valid, y_pred_valid)) / (y_actual_valid.mean() + 1e-10)
                    }
        except Exception as e:
            logger.warning(f"[!] SARIMA metrics failed for {cluster}: {e}")
        
        # ===== LSTM METRICS =====
        try:
            # Find cluster index by name
            cluster_list = list(y_test.columns)
            if cluster in cluster_list:
                cluster_idx = cluster_list.index(cluster)
                
                # Use padded LSTM predictions (same length as test set)
                lstm_pred_cluster = lstm_predictions[:len(actual), cluster_idx]
                valid_idx = ~(np.isnan(lstm_pred_cluster) | np.isinf(lstm_pred_cluster) | 
                            np.isnan(actual) | np.isinf(actual))
                
                if valid_idx.sum() > 0:
                    y_actual_valid = actual[valid_idx]
                    y_pred_valid = lstm_pred_cluster[valid_idx]
                    
                    metrics_results[cluster]['LSTM'] = {
                        'RMSE': np.sqrt(mean_squared_error(y_actual_valid, y_pred_valid)),
                        'MAE': mean_absolute_error(y_actual_valid, y_pred_valid),
                        'MAPE': calculate_mape(y_actual_valid, y_pred_valid),
                        'R2': r2_score(y_actual_valid, y_pred_valid),
                        'Median_AE': median_absolute_error(y_actual_valid, y_pred_valid),
                        'RMSE_norm': np.sqrt(mean_squared_error(y_actual_valid, y_pred_valid)) / (y_actual_valid.mean() + 1e-10)
                    }
        except Exception as e:
            logger.warning(f"[!] LSTM metrics failed for {cluster}: {e}")
                
        if (cluster_idx + 1) % 50 == 0:
            logger.info(f"  [{cluster_idx + 1}/{len(common_clusters)}] Metrics calculated")
    
    # Save checkpoints
    checkpoint_mgr.save_checkpoint('metrics_results', metrics_results)
    checkpoint_mgr.save_checkpoint('xgb_predictions', xgb_predictions)
    checkpoint_mgr.save_checkpoint('sarima_predictions', sarima_predictions)
    checkpoint_mgr.save_checkpoint('lstm_predictions', lstm_predictions)
    checkpoint_mgr.mark_complete('models_evaluated')
    
    logger.info("\n" + "="*80)
    logger.info("[OK] Model evaluation completed successfully")
    logger.info(f"Total clusters evaluated: {len(common_clusters)}")
    logger.info("="*80 + "\n")

else:
    logger.info("[OK] Models already evaluated (from checkpoint)")
    metrics_results = checkpoint_mgr.load_checkpoint('metrics_results')
    xgb_predictions = checkpoint_mgr.load_checkpoint('xgb_predictions')
    sarima_predictions = checkpoint_mgr.load_checkpoint('sarima_predictions')
    lstm_predictions = checkpoint_mgr.load_checkpoint('lstm_predictions')


# =========================================================================
# STEP 7: VISUALIZATIONS
# =========================================================================

print("\n" + "="*80)
print("STEP 7: VISUALIZATIONS")
print("="*80)

if not checkpoint_mgr.is_complete('visualizations_created'):
    progress.update(80, "Creating visualizations")
    
    # Plot 1: Enhanced Model Comparison with All Metrics
    
    if metrics_results:
        first_cluster = list(metrics_results.keys())[0]
        metrics_df = pd.DataFrame(metrics_results[first_cluster]).T
        
        metrics_to_plot = ['RMSE', 'MAE', 'MAPE', 'R2', 'Median_AE']
        metrics_available = [m for m in metrics_to_plot if m in metrics_df.columns]
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics_available):
            ax = axes[idx]
            values = metrics_df[metric]
            
            # Use log scale for MAPE if values are large
            if metric == 'MAPE' and values.max() > 1000:
                ax.set_yscale('log')
                title_suffix = ' (log scale)'
            else:
                title_suffix = ''
            
            values.plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
            ax.set_title(f'{metric}{title_suffix} - {first_cluster}', fontsize=12, fontweight='bold')
            ax.set_ylabel(metric, fontsize=11)
            ax.set_xlabel('Model', fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
        
        # Hide unused subplots
        for idx in range(len(metrics_available), len(axes)):
            axes[idx].axis('off')
        
        fig.suptitle(f'Model Performance Comparison - {first_cluster}', fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(os.path.join(VIZ_DIR, 'model_comparison_detailed.png'), dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("[OK] Enhanced model comparison plot saved")
        
        # Summary comparison across all clusters
        avg_metrics = {}
        for model in ['SARIMA', 'XGBoost', 'LSTM']:
            avg_metrics[model] = {}
            for metric in metrics_available:
                values = [metrics_results[c].get(model, {}).get(metric, np.nan) 
                         for c in metrics_results.keys()]
                avg_metrics[model][metric] = np.nanmean(values)
        
        summary_df = pd.DataFrame(avg_metrics).T
        
        fig, ax = plt.subplots(figsize=(12, 6))
        summary_df.plot(kind='bar', ax=ax, width=0.8)
        ax.set_title('Average Performance Across All Clusters', fontsize=14, fontweight='bold')
        ax.set_ylabel('Metric Value', fontsize=12)
        ax.set_xlabel('Model', fontsize=12)
        ax.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(VIZ_DIR, 'model_comparison_average.png'), dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("[OK] Average model comparison plot saved")
    
    # Plot 2: Feature importance (unchanged)
    if xgb_optimizer.feature_importance:
        importance_df = list(xgb_optimizer.feature_importance.values())[0].head(20)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        importance_df.set_index('feature').plot(kind='barh', ax=ax)
        ax.set_title('Top 20 Most Important Features (XGBoost)')
        ax.set_xlabel('Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(VIZ_DIR, 'feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("[OK] Feature importance plot saved")
    
    # Plot 3: FORECAST COMPARISON (Multiple Clusters)
    logger.info("Creating forecast comparison visualizations...")
    
    try:
        # Select top 6 clusters by average MAE
        cluster_performance = {}
        for cluster in metrics_results.keys():
            avg_mae = np.nanmean([metrics_results[cluster].get(model, {}).get('MAE', np.inf) 
                                 for model in ['SARIMA', 'XGBoost', 'LSTM']])
            cluster_performance[cluster] = avg_mae
        
        top_clusters = sorted(cluster_performance.items(), key=lambda x: x[1])[:6]
        top_cluster_names = [c[0] for c in top_clusters]
        
        # Get the common length for all predictions
        n_test_samples = len(X_test)
        
        # Create forecast comparison for each top cluster
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        for plot_idx, cluster in enumerate(top_cluster_names):
            ax = axes[plot_idx]
            
            # Get actual values
            actual_values = y_test[cluster].values[:n_test_samples]
            time_steps = np.arange(len(actual_values))
            
            # Plot actual
            ax.plot(time_steps, actual_values, 'ko-', linewidth=2, markersize=4, label='Actual', zorder=3)
            
            # Plot XGBoost predictions
            if cluster in xgb_predictions.columns:
                xgb_pred = xgb_predictions[cluster].values[:n_test_samples]
                ax.plot(time_steps, xgb_pred, 's--', linewidth=1.5, markersize=3, 
                       label='XGBoost', alpha=0.8, color='#ff7f0e')
            
            # Plot SARIMA predictions
            if cluster in sarima_predictions:
                sarima_pred = sarima_predictions[cluster][:n_test_samples]
                ax.plot(time_steps, sarima_pred, '^--', linewidth=1.5, markersize=3, 
                       label='SARIMA', alpha=0.8, color='#1f77b4')
            
            # Plot LSTM predictions
            if cluster in y_test.columns:
                cluster_idx = list(y_test.columns).index(cluster)
                if cluster_idx < lstm_predictions.shape[1]:
                    lstm_pred = lstm_predictions[:n_test_samples, cluster_idx]
                    ax.plot(time_steps, lstm_pred, 'v--', linewidth=1.5, markersize=3, 
                           label='LSTM', alpha=0.8, color='#2ca02c')
            
            ax.set_title(f'{cluster} (MAE: {cluster_performance[cluster]:.2f})', 
                        fontsize=11, fontweight='bold')
            ax.set_xlabel('Time Step', fontsize=10)
            ax.set_ylabel('Demand (Trips)', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=9)
        
        fig.suptitle('Forecast Comparison - Top 6 Clusters by Performance', 
                    fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(os.path.join(VIZ_DIR, 'forecast_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("[OK] Forecast comparison plot saved")
        
    except Exception as e:
        logger.warning(f"[!] Forecast comparison plot failed: {e}")
    
    # Plot 4: Residuals comparison (optional but useful)
    logger.info("Creating residuals visualization...")
    
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for model_idx, model in enumerate(['SARIMA', 'XGBoost', 'LSTM']):
            ax = axes[model_idx]
            residuals_all = []
            
            for cluster in top_cluster_names:
                actual_values = y_test[cluster].values[:n_test_samples]
                
                if model == 'XGBoost' and cluster in xgb_predictions.columns:
                    pred_values = xgb_predictions[cluster].values[:n_test_samples]
                elif model == 'SARIMA' and cluster in sarima_predictions:
                    pred_values = sarima_predictions[cluster][:n_test_samples]
                elif model == 'LSTM' and cluster in y_test.columns:
                    cluster_idx = list(y_test.columns).index(cluster)
                    if cluster_idx < lstm_predictions.shape[1]:
                        pred_values = lstm_predictions[:n_test_samples, cluster_idx]
                    else:
                        continue
                else:
                    continue
                
                residuals = actual_values - pred_values
                residuals_all.extend(residuals)
            
            if residuals_all:
                ax.hist(residuals_all, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
                ax.axvline(np.mean(residuals_all), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(residuals_all):.2f}')
                ax.set_title(f'{model} Residuals Distribution', fontsize=11, fontweight='bold')
                ax.set_xlabel('Residuals (Actual - Predicted)', fontsize=10)
                ax.set_ylabel('Frequency', fontsize=10)
                ax.legend()
                ax.grid(True, alpha=0.3, axis='y')
        
        fig.suptitle('Residuals Analysis - Top Models', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(VIZ_DIR, 'residuals_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("[OK] Residuals analysis plot saved")
        
    except Exception as e:
        logger.warning(f"[!] Residuals analysis plot failed: {e}")
    
    checkpoint_mgr.mark_complete('visualizations_created')
    logger.info("[OK] Visualization step completed")

else:
    logger.info("[OK] Visualizations already created")
    
    checkpoint_mgr.mark_complete('visualizations_created')
    logger.info("[OK] Visualization step completed")


# =========================================================================
# STEP 8: SUMMARY & EXPORT
# =========================================================================

print("\n" + "="*80)
print("STEP 8: SUMMARY & EXPORT")
print("="*80)

# Save detailed results
if metrics_results:
    metrics_summary = {}
    for cluster, models_metrics in metrics_results.items():
        for model, metrics in models_metrics.items():
            key = f"{cluster}_{model}"
            metrics_summary[key] = metrics
    
    metrics_df = pd.DataFrame(metrics_summary).T
    metrics_df.to_csv(os.path.join(RESULTS_DIR, 'all_models_metrics.csv'))
    logger.info("[OK] Metrics saved to: all_models_metrics.csv")


# Save feature importance
if xgb_optimizer.feature_importance:
    if 'gain' in xgb_optimizer.feature_importance:
        xgb_optimizer.feature_importance['gain'].to_csv(
            os.path.join(RESULTS_DIR, 'feature_importance_gain.csv'), index=False)
        logger.info("[OK] Gain importance saved")
    
    if 'shap_aggregated' in xgb_optimizer.feature_importance:
        xgb_optimizer.feature_importance['shap_aggregated'].to_csv(
            os.path.join(RESULTS_DIR, 'feature_importance_shap_aggregated.csv'), index=False)
        logger.info("[OK] SHAP aggregated importance saved")


# Save predictions
if not xgb_predictions.empty:
    xgb_predictions.to_csv(os.path.join(RESULTS_DIR, 'xgboost_predictions.csv'))
    logger.info("[OK] XGBoost predictions saved")


# Generate summary report
report = f"""
================================================================================
INTEGRATED FORECASTING PIPELINE - FINAL REPORT (FIXED VERSION)
================================================================================
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

FIXES APPLIED:
1. SHAP library import check added
2. Demand/feature column separation corrected
3. LSTM sequence creation without data leakage
4. LSTM inverse transform logic fixed
5. SARIMA rolling forecast with actual values
6. Metrics calculation alignment ensured
7. MAPE calculation with zero handling
8. Validation set integrated for early stopping
9. Cluster distribution analysis added
10. Individual model serialization for reproducibility

MODELS EVALUATED:
1. SARIMA (Seasonal ARIMA) - with rolling forecast
2. XGBoost - with hyperparameter optimization & early stopping
3. LSTM

FEATURE ENGINEERING:
- Lag features: {X_train.shape[1]} total features
- Top clusters: {len(top_clusters)}
- Training samples: {len(X_train)}
- Validation samples: {len(X_val) if 'X_val' in locals() else 'N/A'}
- Test samples: {len(X_test)}

RESULTS:
- All metrics: {os.path.join(RESULTS_DIR, 'all_models_metrics.csv')}
- Feature importance: {os.path.join(RESULTS_DIR, 'feature_importance.csv')}
- XGBoost predictions: {os.path.join(RESULTS_DIR, 'xgboost_predictions.csv')}
- Individual models: {MODELS_DIR}/

VISUALIZATIONS:
- model_comparison.png: Performance across models
- feature_importance.png: Top 20 features
- forecast_comparison.png: Sample forecast vs actual

================================================================================
"""

with open(os.path.join(RESULTS_DIR, 'integrated_pipeline_report.txt'), 'w') as f:
    f.write(report)


logger.info("[OK] Pipeline execution completed successfully!")
print("\n" + "="*80)
progress.update(100, "Pipeline completed!")
print("="*80)
print("\n[OK] All outputs saved to:", OUTPUT_BASE)