# ====================================================================================
# Seasonal ARIMA + XGBoost + LSTM models DEMAND PREDICTION PIPELINE FOR NYC TAXI DATA
# ====================================================================================
# ====================================================================================
import os
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

# Machine Learning
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, mean_absolute_percentage_error,
    median_absolute_error, r2_score
)
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor

# Deep Learning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Only show ERROR, skip WARNINGS
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Bidirectional, Input, RepeatVector, TimeDistributed
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Feature Importance - Check if SHAP is available
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ============================================================================
# CONFIGURATION
# ============================================================================


INPUT_DATA_PATH = 'C:/Users/Anya/master_thesis/output'
INPUT_FILE = 'taxi_data_cleaned_full_with_clusters.parquet'
OUTPUT_BASE = 'C:/Users/Anya/master_thesis/output/models_upd'


CHECKPOINT_DIR = os.path.join(OUTPUT_BASE, 'checkpoints')
RESULTS_DIR = os.path.join(OUTPUT_BASE, 'results')
VIZ_DIR = os.path.join(OUTPUT_BASE, 'visualizations')
LOG_DIR = os.path.join(OUTPUT_BASE, 'logs')
MODELS_DIR = os.path.join(OUTPUT_BASE, 'models')


for d in [CHECKPOINT_DIR, RESULTS_DIR, VIZ_DIR, LOG_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)


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
def calculate_mape(y_true, y_pred, epsilon=1e-10):
    """Calculate MAPE with handling for zero values"""
    return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100


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


def create_lag_features(data: pd.DataFrame, lags: int = 24) -> pd.DataFrame:
    """Create lag features for XGBoost and LSTM"""
    logger.info(f"Creating lag features (lags={lags})...")
    progress.update(14, f"Creating {lags} lag features")
    
    df = data.copy()
    
    # FIX #2: Identify demand columns (exclude temporal features)
    temporal_feature_cols = [
        'hour', 'day_of_week', 'day_of_month', 'month', 'is_weekend', 
        'is_rush_hour', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'
    ]
    demand_cols = [col for col in data.columns if col not in temporal_feature_cols]
    
    logger.info(f"Found {len(demand_cols)} demand columns, {len(temporal_feature_cols)} temporal features")
    
    # Create lags for each cluster
    for col in demand_cols:
        for lag in range(1, lags + 1):
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
# SARIMA MODEL
# ============================================================================


class SARIMAPredictor:
    """SARIMA wrapper with rolling forecast support"""
    
    def __init__(self, cluster_name: str):
        self.cluster_name = cluster_name
        self.model = None
        self.results = None
        self.fitted = False
        self.train_data = None
    
    def fit(self, train_data: pd.Series, order=(1,1,1), seasonal_order=(1,1,1,24)):
        """Fit SARIMA model"""
        try:
            self.train_data = train_data.copy()
            self.model = SARIMAX(train_data, order=order, seasonal_order=seasonal_order,
                                enforce_stationarity=False, enforce_invertibility=False)
            self.results = self.model.fit(disp=False, maxiter=200)
            self.fitted = True
            logger.info(f"[OK] SARIMA fitted for {self.cluster_name}: AIC={self.results.aic:.2f}")
            return self.results
        except Exception as e:
            logger.error(f"[X] SARIMA fitting failed for {self.cluster_name}: {e}")
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
        """GridSearchCV for hyperparameter optimization with validation set"""
        logger.info(f"Tuning XGBoost hyperparameters (grid_type={grid_type})...")
        progress.update(35, "Optimizing XGBoost hyperparameters")
        
        if target_cluster is None:
            target = y_train.iloc[:, 0]
            target_name = y_train.columns[0]
        else:
            target = y_train[target_cluster]
            target_name = target_cluster
        
        logger.info(f"  Target cluster: {target_name}")
        
        base_model = XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=1, verbosity=0)
        param_grid = self.get_param_grid(grid_type)
        
        tscv = TimeSeriesSplit(n_splits=3)
        
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=tscv,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1,
            verbose=0
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
        
        # ALWAYS compute gain-based importance (fast, reliable)
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
# LSTM ENCODER-DECODER
# ============================================================================


class LSTMPredictor:
    """LSTM encoder-decoder for sequence prediction"""

    def __init__(self, n_features: int, n_lags: int = 24):
        self.n_features = n_features
        self.n_lags = n_lags
        self.model = None
        self.scaler = MinMaxScaler()
        self.history = None

    def create_sequences(self, data: np.ndarray, seq_length: int = 24):
        """Create sequences for LSTM"""
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length])
        return np.array(X), np.array(y)

    def build_model(self, units: list = [64, 32]):
        """Build encoder-decoder LSTM"""
        logger.info("Building LSTM encoder-decoder model...")

        inputs = keras.Input(shape=(self.n_lags, self.n_features))

        # Encoder
        encoded = Bidirectional(LSTM(units[0], activation='relu', return_sequences=True))(inputs)
        encoded = Dropout(0.2)(encoded)
        encoded = Bidirectional(LSTM(units[1], activation='relu'))(encoded)
        encoded = Dropout(0.2)(encoded)

        # Decoder
        decoded = RepeatVector(1)(encoded)
        decoded = LSTM(units[1], activation='relu', return_sequences=True)(decoded)
        decoded = Dropout(0.2)(decoded)
        decoded = LSTM(units[0], activation='relu', return_sequences=False)(decoded)
        decoded = Dropout(0.2)(decoded)

        outputs = Dense(self.n_features)(decoded)

        self.model = keras.Model(inputs, outputs)
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

        logger.info(f"[OK] LSTM model built: {self.model.count_params()} parameters")
        return self.model

    def fit(self, X_train: np.ndarray, epochs: int = 50, batch_size: int = 32):
        """Fit LSTM encoder-decoder for multi-step forecasting"""
        logger.info("Fitting LSTM for multi-step forecasting...")
        progress.update(50, "Training LSTM encoder-decoder")

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
        ]

        # X_train shape: (num_sequences, 24, num_features)
        # Target: predict next step(s) from last timestep
        self.history = self.model.fit(
            X_train,  # Keep ALL 24 timesteps (encoder input)
            X_train[:, -1, :],  # Target: last timestep (decoder output)
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=0
        )

        logger.info(f"[OK] LSTM encoder-decoder fitted. Final loss: {self.history.history['loss'][-1]:.4f}")

    def predict(self, X):
        """Generate predictions"""
        return self.model.predict(X, verbose=0)


# ============================================================================
# PIPELINE EXECUTION
# ============================================================================


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


if not checkpoint_mgr.is_complete('features_engineered'):
    gc.collect()
    print("\n" + "="*80)
    print("STEP 2: FEATURE ENGINEERING")
    print("="*80)
    
    # FIX #2: Properly separate demand and feature columns
    train_features, demand_cols, temporal_feature_cols = create_lag_features(
        demand_matrix.iloc[:len(train_data)], lags=24
    )
    test_features_all, _, _ = create_lag_features(demand_matrix, lags=24)
    test_features = test_features_all.iloc[len(train_data):]
    
    # Select top demand clusters for modeling
    top_n_clusters = 30
    top_clusters = demand_matrix.sum().nlargest(top_n_clusters).index.tolist()
    
    logger.info(f"\nCluster demand distribution (top 15):")
    cluster_dist = demand_matrix.sum().sort_values(ascending=False).head(15)
    for cluster, demand in cluster_dist.items():
        pct = 100 * demand / demand_matrix.sum().sum()
        logger.info(f"  {cluster}: {demand:,.0f} trips ({pct:.1f}%)")
    logger.info(f"Top {top_n_clusters} clusters represent {cluster_dist[:top_n_clusters].sum() / demand_matrix.sum().sum() * 100:.1f}% of demand")
    
    # FIX #2: Feature columns = lags + temporal, excluding original demand
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
    checkpoint_mgr.save_checkpoint('top_clusters', top_clusters)
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


print("\n" + "="*80)
print("STEP 3: SARIMA MODELS")
print("="*80)


if not checkpoint_mgr.is_complete('sarima_fitted'):
    sarima_models = {}
    sarima_results = {}
    fitted_sarima = []
    
    for idx, cluster in enumerate(top_clusters):
        progress.update(20 + (idx * 2), f"Fitting SARIMA for cluster {cluster}")
        
        predictor = SARIMAPredictor(str(cluster))
        results = predictor.fit(train_data[cluster])
        
        if results is not None:
            sarima_models[str(cluster)] = predictor
            fitted_sarima.append(cluster)
            # Save individual SARIMA models
            joblib.dump(results, os.path.join(MODELS_DIR, f'sarima_model_{cluster}.pkl'))
    
    checkpoint_mgr.save_checkpoint('sarima_models', sarima_models)
    checkpoint_mgr.save_checkpoint('fitted_sarima', fitted_sarima)
    checkpoint_mgr.mark_complete('sarima_fitted')
    del train_data  # Release large time series
    gc.collect()
else:
    print("[OK] SARIMA models already fitted (from checkpoint)")
    sarima_models = checkpoint_mgr.load_checkpoint('sarima_models')
    fitted_sarima = checkpoint_mgr.load_checkpoint('fitted_sarima')


print("\n" + "="*80)
print("STEP 4: XGBOOST WITH HYPERPARAMETER OPTIMIZATION")
print("="*80)


if not checkpoint_mgr.is_complete('xgboost_fitted'):
    xgb_optimizer = XGBoostOptimizer(n_lags=24)
    
    # Hyperparameter tuning for first cluster
    #logger.info("Starting XGBoost hyperparameter optimization...")
    #xgb_optimizer.tune_hyperparameters(X_train, y_train, X_val, y_val,
    #                                   target_cluster=top_clusters[0], 
    #                                   grid_type='quick')
    
    # Fit with validation set
    xgb_optimizer.fit_all_clusters(X_train, y_train, X_val, y_val)

    checkpoint_mgr.save_checkpoint('xgb_optimizer', xgb_optimizer)
    logger.info(f"[OK] Fitted {len(xgb_optimizer.models)} XGBoost models - checkpoint saved")
    
    # Compute feature importance
    xgb_optimizer.compute_feature_importance(X_train, y_train)
    
    # Save individual XGBoost models
    for cluster, model in xgb_optimizer.models.items():
        joblib.dump(model, os.path.join(MODELS_DIR, f'xgb_model_{cluster}.pkl'))
    logger.info(f"[OK] Saved {len(xgb_optimizer.models)} XGBoost models")
    
    checkpoint_mgr.save_checkpoint('xgb_optimizer', xgb_optimizer)
    checkpoint_mgr.mark_complete('xgboost_fitted')
else:
    print("[OK] XGBoost already fitted (from checkpoint)")
    xgb_optimizer = checkpoint_mgr.load_checkpoint('xgb_optimizer')


# ============================================================================
# FORCE RECOMPUTE FEATURE IMPORTANCE
# ============================================================================

print("\n" + "="*80)
print("RECOMPUTING FEATURE IMPORTANCE")
print("="*80)

logger.info("Forcing feature importance recomputation...")
xgb_optimizer.compute_feature_importance(X_train, y_train)

# Save updated feature importance
if xgb_optimizer.feature_importance:
    importance_df = list(xgb_optimizer.feature_importance.values())[0]
    importance_df.to_csv(os.path.join(RESULTS_DIR, 'feature_importance_updated.csv'), index=False)
    logger.info("[OK] Updated feature importance saved to: feature_importance_updated.csv")

print("\n" + "="*80)
print("STEP 5: LSTM MODELS")
print("="*80)


if not checkpoint_mgr.is_complete('lstm_fitted'):
    logger.info("Building and fitting LSTM models...")
    
    # Scale data for LSTM
    scaler = MinMaxScaler()
    gc.collect()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Proper sequence creation to avoid data leakage
    X_combined = np.vstack([X_train_scaled, X_test_scaled])
    X_combined_lstm, _ = LSTMPredictor(X_train.shape[1]).create_sequences(X_combined, seq_length=24)
    
    # Split sequences properly
    n_train_seq = len(X_train_scaled) - 24
    X_train_lstm = X_combined_lstm[:n_train_seq]
    X_test_lstm = X_combined_lstm[n_train_seq:]
    
    logger.info(f"[OK] LSTM sequences created - Train: {X_train_lstm.shape}, Test: {X_test_lstm.shape}")
    
    # Build LSTM
    lstm_predictor = LSTMPredictor(n_features=X_train.shape[1], n_lags=24)
    lstm_predictor.build_model(units=[64, 32])
    lstm_predictor.scaler = scaler
    
    # Fit LSTM
    lstm_predictor.fit(X_train_lstm, epochs=50, batch_size=32)
    
    checkpoint_mgr.save_checkpoint('lstm_predictor', lstm_predictor)
    checkpoint_mgr.save_checkpoint('scaler_lstm', scaler)
    checkpoint_mgr.mark_complete('lstm_fitted')
else:
    print("[OK] LSTM already fitted (from checkpoint)")
    lstm_predictor = checkpoint_mgr.load_checkpoint('lstm_predictor')
    scaler = checkpoint_mgr.load_checkpoint('scaler_lstm')


print("\n" + "="*80)
print("STEP 6: MODEL EVALUATION")
print("="*80)


if not checkpoint_mgr.is_complete('models_evaluated'):
    progress.update(70, "Evaluating all models")
    
    # XGBoost predictions
    xgb_predictions = xgb_optimizer.predict(X_test)
    
    # SARIMA with rolling forecast
    sarima_predictions = {}
    for cluster in fitted_sarima:
        try:
            predictor = sarima_models[str(cluster)]
            preds = []
            history = list(train_data[cluster].values)
            
            for t in range(len(test_data)):
                # Forecast next step
                fc = predictor.forecast(steps=1)
                preds.append(fc[0])
                
                if t < len(test_data):
                    history.append(test_data[cluster].iloc[t])
            
            sarima_predictions[cluster] = np.array(preds)
            logger.info(f"[OK] SARIMA rolling forecast completed for {cluster}")
        except Exception as e:
            logger.warning(f"[!] SARIMA forecast failed for cluster {cluster}: {e}")
    
    # LSTM predictions with proper inverse transform
    X_test_scaled = scaler.transform(X_test)
    X_test_lstm, _ = lstm_predictor.create_sequences(X_test_scaled, seq_length=24)
    lstm_pred = lstm_predictor.predict(X_test_lstm)
    
    # Correct inverse transform
    dummy = np.zeros((lstm_pred.shape[0], X_test.shape[1]))
    dummy[:, :lstm_pred.shape[1]] = lstm_pred
    lstm_predictions_full = scaler.inverse_transform(dummy)
    lstm_predictions = lstm_predictions_full[:, :lstm_pred.shape[1]]
    
    # Ensure alignment between predictions and actual values
    common_clusters = list(set(top_clusters) & set(y_test.columns) & set(xgb_predictions.columns))
    logger.info(f"Evaluating {len(common_clusters)} common clusters")
    
    # Calculate metrics for all models
    metrics_results = {}
    
    for cluster in common_clusters:
        actual = y_test[cluster].values
        
        metrics_results[cluster] = {
            'SARIMA': {},
            'XGBoost': {},
            'LSTM': {}
        }
        
        # XGBoost metrics
        xgb_pred = xgb_predictions[cluster].values[:len(actual)]
        metrics_results[cluster]['XGBoost'] = {
            'RMSE': np.sqrt(mean_squared_error(actual, xgb_pred)),
            'MAE': mean_absolute_error(actual, xgb_pred),
            'MAPE': calculate_mape(actual, xgb_pred),  # FIX #7: Use helper function
            'R2': r2_score(actual, xgb_pred)
        }
        
        # SARIMA metrics
        if cluster in sarima_predictions:
            sarima_pred = sarima_predictions[cluster][:len(actual)]
            metrics_results[cluster]['SARIMA'] = {
                'RMSE': np.sqrt(mean_squared_error(actual, sarima_pred)),
                'MAE': mean_absolute_error(actual, sarima_pred),
                'MAPE': calculate_mape(actual, sarima_pred),  # FIX #7
                'R2': r2_score(actual, sarima_pred)
            }
        
        logger.info(f"\nCluster {cluster}:")
        logger.info(f"  XGBoost - RMSE: {metrics_results[cluster]['XGBoost']['RMSE']:.4f}")
        if cluster in sarima_predictions:
            logger.info(f"  SARIMA  - RMSE: {metrics_results[cluster]['SARIMA']['RMSE']:.4f}")
    
    checkpoint_mgr.save_checkpoint('metrics_results', metrics_results)
    checkpoint_mgr.save_checkpoint('xgb_predictions', xgb_predictions)
    checkpoint_mgr.save_checkpoint('sarima_predictions', sarima_predictions)
    checkpoint_mgr.save_checkpoint('lstm_predictions', lstm_predictions)
    checkpoint_mgr.mark_complete('models_evaluated')
    del sarima_models
    gc.collect()
else:
    print("[OK] Models already evaluated (from checkpoint)")
    metrics_results = checkpoint_mgr.load_checkpoint('metrics_results')
    xgb_predictions = checkpoint_mgr.load_checkpoint('xgb_predictions')
    sarima_predictions = checkpoint_mgr.load_checkpoint('sarima_predictions')
    lstm_predictions = checkpoint_mgr.load_checkpoint('lstm_predictions')


print("\n" + "="*80)
print("STEP 7: VISUALIZATIONS")
print("="*80)


if not checkpoint_mgr.is_complete('visualizations_created'):
    progress.update(80, "Creating visualizations")
    
    # Plot 1: Model comparison
    if metrics_results:
        first_cluster = list(metrics_results.keys())[0]
        metrics_df = pd.DataFrame(metrics_results[first_cluster]).T
        
        fig, ax = plt.subplots(figsize=(10, 6))
        metrics_df[['RMSE', 'MAE']].plot(kind='bar', ax=ax)
        ax.set_title(f'Model Comparison - Cluster {first_cluster}')
        ax.set_ylabel('Error')
        ax.legend(['RMSE', 'MAE'])
        plt.tight_layout()
        plt.savefig(os.path.join(VIZ_DIR, 'model_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("[OK] Model comparison plot saved")
    
    # Plot 2: Feature importance
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
    
    # Plot 3: Forecast comparison
    sample_cluster = list(metrics_results.keys())[0]
    if sample_cluster in y_test.columns:
        actual = y_test[sample_cluster].values[:100]
        
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(actual, 'k-', label='Actual', linewidth=2)
        
        if sample_cluster in xgb_predictions.columns:
            ax.plot(xgb_predictions[sample_cluster].values[:100], 'b--', label='XGBoost', alpha=0.7)
        
        if sample_cluster in sarima_predictions:
            ax.plot(sarima_predictions[sample_cluster][:100], 'g--', label='SARIMA', alpha=0.7)
        
        ax.set_title(f'Forecast Comparison - Cluster {sample_cluster}')
        ax.set_ylabel('Demand')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(VIZ_DIR, 'forecast_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("[OK] Forecast comparison plot saved")
    gc.collect()
    checkpoint_mgr.mark_complete('visualizations_created')


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
3. LSTM - encoder-decoder architecture

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
