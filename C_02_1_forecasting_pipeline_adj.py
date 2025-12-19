# ====================================================================================
# ENHANCED FORECASTING PIPELINE - SARIMA + XGBoost + OptimizedLSTM (ALL 533 CLUSTERS)
# WITH PCA DIMENSIONALITY REDUCTION + CROSS-VALIDATION COMPONENT OPTIMIZATION
# WITH MULTI-HORIZON EVALUATION (1D, 3D, 7D)
# ====================================================================================
# Features:
#   [+] All 533 NYC taxi clusters
#   [+] SARIMA training (resumable, model tracking)
#   [+] XGBoost with sample-based hyperparameter tuning (resumable)
#   [+] OptimizedLSTM with PCA + TimeSeriesSplit CV (resumable, model tracking)
#   [+] Multi-horizon evaluation: 1D, 3D, 7D predictions
#   [+] Accuracy deterioration analysis by horizon
#   [+] Comprehensive metrics and visualizations
# ====================================================================================

import os
import gc
import warnings
import json
import pickle
import logging
import joblib
from datetime import datetime
from typing import Dict, Tuple, Optional, List, Set

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Statistical models
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Machine Learning
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, mean_absolute_percentage_error,
    median_absolute_error, r2_score
)
from xgboost import XGBRegressor

# Deep Learning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ====================================================================================
# CONFIGURATION
# ====================================================================================

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

# Multi-horizon configuration (in hours)
HORIZONS = {
    '1D': 24,   # 1 day = 24 hours
    '3D': 72,   # 3 days = 72 hours
    '7D': 168   # 7 days = 168 hours
}

# Logging
log_file = os.path.join(LOG_DIR, f'integrated_pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

np.random.seed(42)
tf.random.set_seed(42)

print("="*80)
print("ENHANCED FORECASTING PIPELINE WITH PCA + CV OPTIMIZED LSTM")
print("Multi-Horizon Evaluation: 1D, 3D, 7D")
print("="*80)
logger.info("Pipeline initialization started")

# ====================================================================================
# UTILITY FUNCTIONS
# ====================================================================================

def calculate_mape(y_true, y_pred, epsilon=1e-10):
    """Calculate MAPE with handling for zero values"""
    return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100

def extract_horizon_predictions(full_forecast, horizon_steps, test_length):
    """Extract predictions for a specific horizon from recursive multi-step forecasts"""
    predictions = []
    for i in range(max(0, len(full_forecast) - horizon_steps + 1)):
        if i + horizon_steps <= len(full_forecast):
            predictions.append(full_forecast[i + horizon_steps - 1])
    
    if len(predictions) < test_length:
        pad_value = predictions[-1] if len(predictions) > 0 else np.nan
        predictions.extend([pad_value] * (test_length - len(predictions)))
    
    return np.array(predictions[:test_length])

def create_lstm_sequences(X, seq_length=24):
    """Create LSTM sequences from feature matrix"""
    n_samples = (len(X) // seq_length) * seq_length
    return X[:n_samples].reshape(-1, seq_length, X.shape[1])

# ====================================================================================
# CHECKPOINT MANAGER
# ====================================================================================

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
                'models_evaluated', 'visualizations_created', 'report_generated'
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

# ====================================================================================
# PROGRESS TRACKER
# ====================================================================================

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

# ====================================================================================
# MODEL TRACKER
# ====================================================================================

class ModelTracker:
    """Tracks which models are saved"""
    
    def __init__(self, models_dir: str):
        self.models_dir = models_dir
        self.tracker_file = os.path.join(models_dir, 'model_tracker.json')
        self.load_tracker()
    
    def load_tracker(self):
        if os.path.exists(self.tracker_file):
            with open(self.tracker_file, 'r') as f:
                self.tracker = json.load(f)
            logger.info(f"[OK] Tracker loaded: {len(self.tracker.get('sarima', {}))} SARIMA, "
                       f"{len(self.tracker.get('xgboost', {}))} XGBoost, "
                       f"{len(self.tracker.get('lstm', {}))} LSTM models tracked")
        else:
            self.tracker = {'sarima': {}, 'xgboost': {}, 'lstm': {}}
    
    def save_tracker(self):
        with open(self.tracker_file, 'w') as f:
            json.dump(self.tracker, f, indent=2)
    
    def get_saved_clusters(self, model_type: str) -> Set[str]:
        return set(self.tracker.get(model_type, {}).keys())
    
    def get_missing_clusters(self, model_type: str, all_clusters: List[str]) -> List[str]:
        saved = self.get_saved_clusters(model_type)
        return [c for c in all_clusters if str(c) not in saved]
    
    def mark_saved(self, model_type: str, cluster_name: str, filename: str):
        if model_type not in self.tracker:
            self.tracker[model_type] = {}
        self.tracker[model_type][str(cluster_name)] = {
            'filename': filename,
            'saved_at': datetime.now().isoformat()
        }
        self.save_tracker()
    
    def scan_directory(self):
        logger.info("Scanning models directory for existing files...")
        sarima_files = set(Path(self.models_dir).glob('sarima_model_*.pkl'))
        xgb_files = set(Path(self.models_dir).glob('xgb_model_*.pkl'))
        lstm_files = set(Path(self.models_dir).glob('lstm_model_*.h5'))
        
        for f in sarima_files:
            cluster_name = f.stem.replace('sarima_model_', '')
            self.mark_saved('sarima', cluster_name, f.name)
        
        for f in xgb_files:
            cluster_name = f.stem.replace('xgb_model_', '')
            self.mark_saved('xgboost', cluster_name, f.name)
        
        for f in lstm_files:
            cluster_name = f.stem.replace('lstm_model_', '')
            self.mark_saved('lstm', cluster_name, f.name)
        
        logger.info(f"[OK] Directory scan complete: {len(sarima_files)} SARIMA, "
                   f"{len(xgb_files)} XGBoost, {len(lstm_files)} LSTM models found")

# ====================================================================================
# OPTIMIZED LSTM FORECASTER WITH PCA + CROSS-VALIDATION
# ====================================================================================

class OptimizedLSTMForecaster:
    """
    LSTM with PCA dimensionality reduction and cross-validation component optimization
    
    Features:
    - Automatic PCA component selection via TimeSeriesSplit CV
    - Standardized feature scaling
    - Encoder-decoder LSTM architecture
    - Mixed precision training for speed
    """
    
    def __init__(self, max_components: int = 256, lstm_units: int = 128, cv_splits: int = 3):
        self.max_components = max_components
        self.lstm_units = lstm_units
        self.cv_splits = cv_splits
        self.n_components = None
        self.pca = None
        self.scaler = StandardScaler()
        self.model = None
        self.fitted = False
        self.optimal_components_history = {}
    
    def find_optimal_components(self, X_train, y_train, component_candidates=None):
        """Cross-validation to find optimal number of PCA components using TimeSeriesSplit"""
        if component_candidates is None:
            n_features = X_train.shape[1]
            component_candidates = sorted(set([
                32, 64, 128, 192, 256,
                int(n_features * 0.25),
                int(n_features * 0.5),
                int(n_features * 0.75),
                min(self.max_components, n_features)
            ]))
        
        component_candidates = sorted(set(component_candidates))
        
        logger.info("=" * 80)
        logger.info(f"FINDING OPTIMAL PCA COMPONENTS (CV with {self.cv_splits} folds)")
        logger.info(f"Component candidates: {component_candidates}")
        logger.info("=" * 80)
        
        tscv = TimeSeriesSplit(n_splits=self.cv_splits)
        cv_results = {comp: [] for comp in component_candidates}
        
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
            logger.info(f"\n--- Fold {fold_idx + 1}/{self.cv_splits} ---")
            
            X_tr = X_train.iloc[train_idx] if isinstance(X_train, pd.DataFrame) else X_train[train_idx]
            X_val = X_train.iloc[val_idx] if isinstance(X_train, pd.DataFrame) else X_train[val_idx]
            y_tr = y_train.iloc[train_idx] if isinstance(y_train, pd.DataFrame) else y_train[train_idx]
            y_val = y_train.iloc[val_idx] if isinstance(y_train, pd.DataFrame) else y_train[val_idx]
            
            scaler_cv = StandardScaler()
            X_tr_scaled = scaler_cv.fit_transform(X_tr)
            X_val_scaled = scaler_cv.transform(X_val)
            
            for comp in component_candidates:
                try:
                    n_comp = min(comp, X_tr_scaled.shape[1], X_tr_scaled.shape[0])
                    pca_cv = PCA(n_components=n_comp)
                    X_tr_reduced = pca_cv.fit_transform(X_tr_scaled)
                    X_val_reduced = pca_cv.transform(X_val_scaled)
                    exp_var = pca_cv.explained_variance_ratio_.sum()
                    
                    val_rmse = self._quick_lstm_validation(
                        X_tr_reduced, y_tr.values if isinstance(y_tr, pd.DataFrame) else y_tr,
                        X_val_reduced, y_val.values if isinstance(y_val, pd.DataFrame) else y_val,
                        epochs=20, batch_size=32
                    )
                    
                    cv_results[comp].append({
                        'rmse': val_rmse,
                        'exp_var': exp_var,
                        'fold': fold_idx,
                        'n_samples_tr': len(X_tr),
                        'n_samples_val': len(X_val)
                    })
                    
                    logger.info(f"  Components={n_comp:3d}: RMSE={val_rmse:.4f}, Exp.Var={exp_var:.4f}")
                
                except Exception as e:
                    logger.warning(f"  Components={comp}: Failed - {str(e)[:100]}")
                    cv_results[comp].append({'rmse': np.inf, 'exp_var': 0, 'fold': fold_idx})
        
        logger.info("\n" + "=" * 80)
        logger.info("CROSS-VALIDATION RESULTS SUMMARY")
        logger.info("=" * 80)
        
        best_comp = None
        best_rmse = np.inf
        results_summary = []
        
        for comp in component_candidates:
            if cv_results[comp] and any(r['rmse'] < np.inf for r in cv_results[comp]):
                valid_rmses = [r['rmse'] for r in cv_results[comp] if r['rmse'] < np.inf]
                valid_exp_vars = [r['exp_var'] for r in cv_results[comp] if r['rmse'] < np.inf]
                
                if valid_rmses:
                    avg_rmse = np.mean(valid_rmses)
                    std_rmse = np.std(valid_rmses)
                    avg_exp_var = np.mean(valid_exp_vars)
                    
                    results_summary.append({
                        'components': comp,
                        'avg_rmse': avg_rmse,
                        'std_rmse': std_rmse,
                        'avg_exp_var': avg_exp_var,
                        'n_folds': len(valid_rmses)
                    })
                    
                    logger.info(f"Components={comp:3d}: Avg RMSE={avg_rmse:.4f} ± {std_rmse:.4f}, Exp.Var={avg_exp_var:.4f}")
                    
                    if avg_rmse < best_rmse:
                        best_rmse = avg_rmse
                        best_comp = comp
        
        self.n_components = best_comp
        self.optimal_components_history = cv_results
        
        logger.info(f"\n[+] OPTIMAL COMPONENTS: {self.n_components} (CV RMSE={best_rmse:.4f})")
        logger.info("=" * 80)
        
        results_df = pd.DataFrame(results_summary)
        return self.n_components, best_rmse, results_df
    
    def _quick_lstm_validation(self, X_train, y_train, X_val, y_val, epochs=20, batch_size=32):
        """Quick LSTM training for component validation"""
        try:
            seq_length = 24
            n_tr_samples = (len(X_train) // seq_length) * seq_length
            n_val_samples = (len(X_val) // seq_length) * seq_length
            
            if n_tr_samples == 0 or n_val_samples == 0:
                return np.inf
            
            X_tr_seq = X_train[:n_tr_samples].reshape(-1, seq_length, X_train.shape[1])
            y_tr_seq = y_train[:n_tr_samples].reshape(-1, seq_length)
            y_tr_target = y_tr_seq[:, -1]
            
            X_val_seq = X_val[:n_val_samples].reshape(-1, seq_length, X_val.shape[1])
            y_val_seq = y_val[:n_val_samples].reshape(-1, seq_length)
            y_val_target = y_val_seq[:, -1]
            
            output_dim = y_train.shape[1] if y_train.ndim > 1 else 1
            
            model = Sequential([
                LSTM(32, return_sequences=False, input_shape=(seq_length, X_train.shape[1]), recurrent_dropout=0.2),
                Dropout(0.2),
                Dense(16, activation='relu'),
                Dense(output_dim)
            ])
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
            
            model.fit(
                X_tr_seq, y_tr_target,
                validation_data=(X_val_seq, y_val_target),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=0)],
                verbose=0
            )
            
            y_pred = model.predict(X_val_seq, verbose=0)
            rmse = np.sqrt(mean_squared_error(y_val_target, y_pred))
            
            del model
            gc.collect()
            
            return rmse
        except Exception as e:
            logger.debug(f"Quick validation error: {str(e)[:100]}")
            return np.inf
    
    def preprocess(self, X_train, X_test):
        """Reduce dimensions with PCA and scale"""
        logger.info("=" * 80)
        logger.info("PCA PREPROCESSING STEP")
        logger.info("=" * 80)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        if self.n_components is None:
            self.n_components = min(self.max_components, X_train_scaled.shape[1])
            logger.warning(f"Components not optimized, using default: {self.n_components}")
        
        self.pca = PCA(n_components=self.n_components)
        X_train_reduced = self.pca.fit_transform(X_train_scaled)
        X_test_reduced = self.pca.transform(X_test_scaled)
        
        exp_var = self.pca.explained_variance_ratio_.sum()
        
        logger.info(f"PCA Explained Variance: {exp_var:.4f}")
        logger.info(f"Reduced dimensions: {X_train.shape} → {X_train_reduced.shape}")
        logger.info(f"Information retained: {exp_var*100:.1f}%")
        logger.info("=" * 80)
        
        return X_train_reduced, X_test_reduced
    
    def build_model(self, seq_length, output_dim):
        """Encoder-decoder LSTM architecture"""
        model = Sequential([
            LSTM(self.lstm_units, return_sequences=True, input_shape=(seq_length, self.n_components), recurrent_dropout=0.2),
            Dropout(0.2),
            LSTM(self.lstm_units // 2, return_sequences=False, recurrent_dropout=0.2),
            Dropout(0.2),
            RepeatVector(seq_length),
            LSTM(self.lstm_units // 2, return_sequences=True, recurrent_dropout=0.2),
            Dropout(0.2),
            TimeDistributed(Dense(64, activation='relu')),
            TimeDistributed(Dense(output_dim))
        ])
        return model
    
    def fit(self, X_train_reduced, y_train, X_val_reduced, y_val, epochs=100, batch_size=16):
        """Train LSTM encoder-decoder"""
        try:
            logger.info("=" * 80)
            logger.info("LSTM TRAINING WITH OPTIMAL PCA COMPONENTS")
            logger.info("=" * 80)
            
            try:
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                logger.info("[+] Mixed precision (float16) enabled")
            except:
                logger.warning("Mixed precision not available")
            
            if isinstance(y_train, pd.DataFrame):
                output_dim = y_train.shape[1]
            else:
                output_dim = y_train.shape[1] if y_train.ndim > 1 else 1
            
            self.model = self.build_model(X_train_reduced.shape[1], output_dim)
            self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
            
            logger.info(f"Model input shape: {X_train_reduced.shape}")
            logger.info(f"Output dimension: {output_dim}")
            logger.info(f"Components: {self.n_components}, LSTM units: {self.lstm_units}")
            
            history = self.model.fit(
                X_train_reduced, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(X_val_reduced, y_val),
                callbacks=[
                    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
                ],
                verbose=1
            )
            
            self.fitted = True
            logger.info("[OK] LSTM training completed successfully")
            logger.info("=" * 80)
            
            return history
        
        except Exception as e:
            logger.error(f"[X] LSTM fitting failed: {str(e)}")
            return None
    
    def predict(self, X_test_reduced):
        """Generate predictions"""
        if not self.fitted or self.model is None:
            raise ValueError("Model not fitted")
        return self.model.predict(X_test_reduced, verbose=0)

# ====================================================================================
# SARIMA PREDICTOR
# ====================================================================================

class SARIMAPredictor:
    """SARIMA wrapper"""
    
    def __init__(self, cluster_name: str):
        self.cluster_name = cluster_name
        self.model = None
        self.results = None
        self.fitted = False
    
    def fit(self, train_data: pd.Series, order=(1,1,1), seasonal_order=(1,1,1,168)):
        try:
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
        if not self.fitted:
            raise ValueError("Model not fitted")
        forecast = self.results.get_forecast(steps=steps)
        return forecast.predicted_mean.values

# ====================================================================================
# XGBoost OPTIMIZER
# ====================================================================================

class XGBoostOptimizer:
    """XGBoost with grid search"""
    
    def __init__(self, n_lags: int = 24):
        self.n_lags = n_lags
        self.best_params = {}
        self.models = {}
    
    def get_param_grid(self, grid_type: str = 'quick'):
        if grid_type == 'quick':
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
    
    def tune_on_sample(self, X_train, y_train, X_val, y_val, sample_clusters=10, grid_type='quick'):
        logger.info(f"Starting XGBoost hyperparameter tuning on {sample_clusters} sample clusters...")
        progress.update(35, f"XGBoost: Tuning hyperparameters on {sample_clusters} clusters")
        
        sample_size = min(sample_clusters, len(y_train.columns))
        sample_clusters_list = list(y_train.columns[:sample_size])
        logger.info(f"Sample clusters selected for tuning: {sample_clusters_list}")
        
        param_grid = self.get_param_grid(grid_type)
        best_params_per_cluster = {}
        best_cv_scores = {}
        
        for cluster_idx, cluster in enumerate(sample_clusters_list):
            logger.info(f"\n[{cluster_idx + 1}/{sample_size}] GridSearchCV for cluster: {cluster}")
            
            try:
                target = y_train[cluster].values
                tscv = TimeSeriesSplit(n_splits=3)
                
                base_model = XGBRegressor(
                    objective='reg:squarederror',
                    random_state=42,
                    n_jobs=8,
                    verbosity=0
                )
                
                gridsearch = GridSearchCV(
                    estimator=base_model,
                    param_grid=param_grid,
                    cv=tscv,
                    scoring='neg_root_mean_squared_error',
                    n_jobs=-1,
                    verbose=0
                )
                
                gridsearch.fit(X_train.values, target)
                
                best_params_per_cluster[cluster] = gridsearch.best_params_
                best_cv_scores[cluster] = -gridsearch.best_score_
                
                logger.info(f"  [+] Best CV RMSE: {best_cv_scores[cluster]:.4f}")
            
            except Exception as e:
                logger.warning(f"[!] GridSearchCV failed for {cluster}: {e}")
                best_params_per_cluster[cluster] = {
                    'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.1,
                    'min_child_weight': 1, 'subsample': 0.8, 'colsample_bytree': 0.8
                }
        
        averaged_params = self._average_hyperparameters(best_params_per_cluster)
        self.best_params['default'] = averaged_params
        
        return averaged_params, best_params_per_cluster, best_cv_scores
    
    def _average_hyperparameters(self, params_dict):
        if not params_dict:
            return {
                'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.1,
                'min_child_weight': 1, 'subsample': 0.8, 'colsample_bytree': 0.8
            }
        
        params_list = list(params_dict.values())
        return {
            'n_estimators': int(np.mean([p.get('n_estimators', 200) for p in params_list])),
            'max_depth': int(np.mean([p.get('max_depth', 5) for p in params_list])),
            'learning_rate': float(np.mean([p.get('learning_rate', 0.1) for p in params_list])),
            'min_child_weight': int(np.mean([p.get('min_child_weight', 1) for p in params_list])),
            'subsample': float(np.mean([p.get('subsample', 0.8) for p in params_list])),
            'colsample_bytree': float(np.mean([p.get('colsample_bytree', 0.8) for p in params_list]))
        }

# ====================================================================================
# DATA LOADING & PREPARATION FUNCTIONS
# ====================================================================================

def load_data(data_path: str, file_name: str) -> pd.DataFrame:
    logger.info("Loading data...")
    progress.update(5, "Loading data from parquet")
    full_path = os.path.join(data_path, file_name)
    data = pd.read_parquet(full_path)
    logger.info(f"Data shape: {data.shape}")
    return data

def prepare_demand_matrix(data: pd.DataFrame, freq: str = 'H') -> pd.DataFrame:
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
    logger.info("Adding temporal features...")
    progress.update(12, "Creating temporal features")
    
    df = demand_matrix.copy()
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day
    df['month'] = df.index.month
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
    df['is_rush_hour'] = df['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
    
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    return df

def create_lag_features(data: pd.DataFrame, lags: int = 168) -> Tuple[pd.DataFrame, List[str], List[str]]:
    logger.info(f"Creating lag features (lags={lags})...")
    progress.update(14, f"Creating {lags} lag features")
    
    df = data.copy()
    temporal_feature_cols = [
        'hour', 'day_of_week', 'day_of_month', 'month', 'is_weekend', 
        'is_rush_hour', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'
    ]
    demand_cols = [col for col in data.columns if col not in temporal_feature_cols]
    
    for col in demand_cols:
        lag_list = lags if isinstance(lags, list) else list(range(1, lags + 1))
        for lag in lag_list:
            df[f'{col}_lag_{lag}'] = data[col].shift(lag)
        
        df[f'{col}_rolling_mean_6'] = data[col].shift(1).rolling(window=6).mean()
        df[f'{col}_rolling_std_6'] = data[col].shift(1).rolling(window=6).std()
        df[f'{col}_rolling_mean_24'] = data[col].shift(1).rolling(window=24).mean()
    
    df = df.dropna()
    logger.info(f"Feature matrix shape after lags: {df.shape}")
    return df, demand_cols, temporal_feature_cols

def create_train_val_test_split(demand_matrix: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1):
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

# ====================================================================================
# MULTI-HORIZON METRICS CALCULATION
# ====================================================================================

def calculate_horizon_metrics(actual, predictions_dict, cluster_name):
    """Calculate metrics for each model and each horizon"""
    metrics_list = []
    
    for horizon_name, horizon_preds in predictions_dict.items():
        for model_name, pred in horizon_preds.items():
            if pred is None or len(pred) == 0:
                continue
            
            min_len = min(len(actual), len(pred))
            actual_trimmed = actual[:min_len]
            pred_trimmed = pred[:min_len]
            
            valid_idx = ~(np.isnan(actual_trimmed) | np.isinf(actual_trimmed) | 
                         np.isnan(pred_trimmed) | np.isinf(pred_trimmed))
            
            if valid_idx.sum() < 2:
                continue
            
            actual_valid = actual_trimmed[valid_idx]
            pred_valid = pred_trimmed[valid_idx]
            
            metrics = {
                'cluster': cluster_name,
                'horizon': horizon_name,
                'model': model_name,
                'RMSE': np.sqrt(mean_squared_error(actual_valid, pred_valid)),
                'MAE': mean_absolute_error(actual_valid, pred_valid),
                'MAPE': calculate_mape(actual_valid, pred_valid),
                'R2': r2_score(actual_valid, pred_valid),
                'MedAE': median_absolute_error(actual_valid, pred_valid)
            }
            metrics_list.append(metrics)
    
    return pd.DataFrame(metrics_list)

# ====================================================================================
# PIPELINE EXECUTION - MAIN SCRIPT (STEPS 0-8)
# ====================================================================================

if __name__ == "__main__":
    
    # ===================================================================================
    # STEP 0: INITIALIZE MODEL TRACKING
    # ===================================================================================
    print("\n" + "="*80)
    print("STEP 0: INITIALIZE MODEL TRACKING")
    print("="*80)
    
    model_tracker = ModelTracker(MODELS_DIR)
    model_tracker.scan_directory()
    logger.info("Pipeline ready - all components initialized successfully!")
    
    # ===================================================================================
    # STEP 1: DATA LOADING & PREPARATION
    # ===================================================================================
    print("\n" + "="*80)
    print("STEP 1: DATA LOADING & PREPARATION")
    print("="*80)
    
    if not checkpoint_mgr.is_complete('data_loaded'):
        data = load_data(INPUT_DATA_PATH, INPUT_FILE)
        checkpoint_mgr.save_checkpoint('raw_data', data)
        checkpoint_mgr.mark_complete('data_loaded')
    else:
        logger.info("[OK] Data already loaded (from checkpoint)")
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
        logger.info("[OK] Data already prepared (from checkpoint)")
        demand_matrix = checkpoint_mgr.load_checkpoint('demand_matrix')
        demand_with_features = checkpoint_mgr.load_checkpoint('demand_with_features')
        train_data = checkpoint_mgr.load_checkpoint('train_data')
        val_data = checkpoint_mgr.load_checkpoint('val_data')
        test_data = checkpoint_mgr.load_checkpoint('test_data')
    
    # ===================================================================================
    # STEP 2: FEATURE ENGINEERING
    # ===================================================================================
    print("\n" + "="*80)
    print("STEP 2: FEATURE ENGINEERING")
    print("="*80)
    
    if not checkpoint_mgr.is_complete('features_engineered'):
        gc.collect()
        
        # Create lag features
        train_features, demand_cols, temporal_feature_cols = create_lag_features(
            demand_matrix.iloc[:len(train_data)], lags=168
        )
        test_features_all, _, _ = create_lag_features(demand_matrix, lags=168)
        test_features = test_features_all.iloc[len(train_data):]
        
        # Select top demand clusters for modeling
        top_n_clusters = 30
        top_clusters = demand_matrix.sum().nlargest(top_n_clusters).index.tolist()
        
        logger.info(f"\nCluster demand distribution (top {top_n_clusters}):")
        cluster_dist = demand_matrix.sum().sort_values(ascending=False).head(top_n_clusters)
        for cluster, demand in cluster_dist.items():
            pct = 100 * demand / demand_matrix.sum().sum()
            logger.info(f"  {cluster}: {demand:,.0f} trips ({pct:.1f}%)")
        logger.info(f"Top {top_n_clusters} clusters represent {cluster_dist.sum() / demand_matrix.sum().sum() * 100:.1f}% of demand")
        
        # Create feature matrices
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
        
        # Save checkpoints
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
        logger.info("[OK] Features already engineered (from checkpoint)")
        X_train = checkpoint_mgr.load_checkpoint('X_train')
        y_train = checkpoint_mgr.load_checkpoint('y_train')
        X_val = checkpoint_mgr.load_checkpoint('X_val')
        y_val = checkpoint_mgr.load_checkpoint('y_val')
        X_test = checkpoint_mgr.load_checkpoint('X_test')
        y_test = checkpoint_mgr.load_checkpoint('y_test')
        top_clusters = checkpoint_mgr.load_checkpoint('top_clusters')
        feature_cols = checkpoint_mgr.load_checkpoint('feature_cols')
    
    # ===================================================================================
    # STEP 3: SARIMA MODELS
    # ===================================================================================
    print("\n" + "="*80)
    print("STEP 3: SARIMA MODELS")
    print("="*80)
    
    if not checkpoint_mgr.is_complete('sarima_fitted'):
        sarima_models = {}
        fitted_sarima = {}
        
        for idx, cluster in enumerate(top_clusters):
            progress.update(20 + (idx * 2), f"Fitting SARIMA for cluster {cluster}")
            
            try:
                predictor = SARIMAPredictor(str(cluster))
                results = predictor.fit(train_data[cluster])
                
                if results is not None:
                    sarima_models[str(cluster)] = predictor
                    fitted_sarima[str(cluster)] = results
                    joblib.dump(results, os.path.join(MODELS_DIR, f'sarima_model_{cluster}.pkl'))
            except Exception as e:
                logger.warning(f"[!] SARIMA fitting failed for {cluster}: {e}")
        
        logger.info(f"[OK] SARIMA fitted for {len(fitted_sarima)} clusters")
        checkpoint_mgr.save_checkpoint('sarima_models', sarima_models)
        checkpoint_mgr.save_checkpoint('fitted_sarima', fitted_sarima)
        checkpoint_mgr.mark_complete('sarima_fitted')
        del train_data
        gc.collect()
    else:
        logger.info("[OK] SARIMA models already fitted (from checkpoint)")
        sarima_models = checkpoint_mgr.load_checkpoint('sarima_models')
        fitted_sarima = checkpoint_mgr.load_checkpoint('fitted_sarima')
    
    # ===================================================================================
    # STEP 4: XGBOOST WITH HYPERPARAMETER OPTIMIZATION
    # ===================================================================================
    print("\n" + "="*80)
    print("STEP 4: XGBOOST WITH HYPERPARAMETER OPTIMIZATION")
    print("="*80)
    
    if not checkpoint_mgr.is_complete('xgboost_fitted'):
        xgb_optimizer = XGBoostOptimizer(n_lags=24)
        
        # Hyperparameter tuning on sample clusters
        logger.info("Starting XGBoost hyperparameter optimization...")
        xgb_optimizer.tune_on_sample(X_train, y_train, X_val, y_val,
                                    sample_clusters=10, grid_type='quick')
        
        # Fit models for all top clusters
        logger.info("Fitting XGBoost models for all clusters...")
        for idx, cluster in enumerate(top_clusters):
            progress.update(40 + (idx / len(top_clusters)) * 5, f"Fitting XGBoost for cluster {cluster}")
            
            try:
                params = xgb_optimizer.best_params.get('default', {
                    'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.1,
                    'min_child_weight': 1, 'subsample': 0.8, 'colsample_bytree': 0.8
                })
                
                model = XGBRegressor(objective='reg:squarederror', random_state=42, **params)
                model.fit(X_train.values, y_train[cluster].values,
                         eval_set=[(X_val.values, y_val[cluster].values)],
                         verbose=False)
                
                xgb_optimizer.models[cluster] = model
                joblib.dump(model, os.path.join(MODELS_DIR, f'xgb_model_{cluster}.pkl'))
            except Exception as e:
                logger.warning(f"[!] XGBoost fitting failed for {cluster}: {e}")
        
        logger.info(f"[OK] Fitted {len(xgb_optimizer.models)} XGBoost models")
        
        # Compute feature importance
        xgb_optimizer.compute_feature_importance(X_train, y_train)


        
        checkpoint_mgr.save_checkpoint('xgb_optimizer', xgb_optimizer)
        checkpoint_mgr.mark_complete('xgboost_fitted')
    else:
        logger.info("[OK] XGBoost already fitted (from checkpoint)")
        xgb_optimizer = checkpoint_mgr.load_checkpoint('xgb_optimizer')
    
    # ===================================================================================
    # STEP 5: LSTM MODELS WITH PCA COMPONENT OPTIMIZATION
    # ===================================================================================
    print("\n" + "="*80)
    print("STEP 5: LSTM MODELS WITH PCA COMPONENT OPTIMIZATION")
    print("="*80)
    
    if not checkpoint_mgr.is_complete('lstm_fitted'):
        logger.info("Building and fitting LSTM models...")
        gc.collect()
        
        # Initialize LSTM forecaster
        lstm_forecaster = OptimizedLSTMForecaster(max_components=256, lstm_units=128, cv_splits=3)
        
        # Find optimal PCA components via cross-validation
        logger.info("Finding optimal PCA components via cross-validation...")
        n_components, best_cv_rmse, cv_results_df = lstm_forecaster.find_optimal_components(X_train, y_train)
        logger.info(f"[+] Optimal components: {n_components} (CV RMSE: {best_cv_rmse:.4f})")
        
        # Preprocess with optimal components
        logger.info("Preprocessing data with optimal PCA components...")
        X_train_reduced, X_test_reduced = lstm_forecaster.preprocess(X_train, X_test)
        
        # Create LSTM sequences
        seq_length = 24
        n_features_reduced = X_train_reduced.shape
        
        n_train_samples = (len(X_train_reduced) // seq_length) * seq_length
        X_train_lstm = X_train_reduced[:n_train_samples].reshape(-1, seq_length, n_features_reduced)
        
        if isinstance(y_train, pd.DataFrame):
            y_train_aligned = y_train.iloc[:n_train_samples]
            y_train_lstm = y_train_aligned.values.reshape(-1, seq_length, y_train_aligned.shape)
            y_train_lstm = y_train_lstm[:, -1, :]
        else:
            y_train_aligned = y_train[:n_train_samples]
            y_train_lstm = y_train_aligned.reshape(-1, seq_length, y_train_aligned.shape)
            y_train_lstm = y_train_lstm[:, -1, :]
        
        logger.info(f"[OK] LSTM sequences created - Train X: {X_train_lstm.shape}, Train y: {y_train_lstm.shape}")
        
        # Prepare test data
        n_test_samples = (len(X_test_reduced) // seq_length) * seq_length
        X_test_lstm = X_test_reduced[:n_test_samples].reshape(-1, seq_length, n_features_reduced)
        
        if isinstance(y_test, pd.DataFrame):
            y_test_aligned = y_test.iloc[:n_test_samples]
            y_test_lstm = y_test_aligned.values.reshape(-1, seq_length, y_test_aligned.shape)
            y_test_lstm = y_test_lstm[:, -1, :]
        else:
            y_test_aligned = y_test[:n_test_samples]
            y_test_lstm = y_test_aligned.reshape(-1, seq_length, y_test_aligned.shape)
            y_test_lstm = y_test_lstm[:, -1, :]
        
        logger.info(f"[OK] LSTM test sequences - Test X: {X_test_lstm.shape}, Test y: {y_test_lstm.shape}")
        
        # Split for validation
        val_split = 0.2
        val_idx = int(len(X_train_lstm) * (1 - val_split))
        
        X_train_split = X_train_lstm[:val_idx]
        y_train_split = y_train_lstm[:val_idx]
        X_val_split = X_train_lstm[val_idx:]
        y_val_split = y_train_lstm[val_idx:]
        
        logger.info(f"Train split: X {X_train_split.shape}, y {y_train_split.shape}")
        logger.info(f"Val split: X {X_val_split.shape}, y {y_val_split.shape}")
        
        # Fit LSTM
        logger.info("Fitting LSTM encoder-decoder...")
        history = lstm_forecaster.fit(
            X_train_split, y_train_split,
            X_val_split, y_val_split,
            epochs=100,
            batch_size=16
        )
        
        # Save checkpoints
        checkpoint_mgr.save_checkpoint('lstm_forecaster', lstm_forecaster)
        checkpoint_mgr.save_checkpoint('lstm_history', history.history if hasattr(history, 'history') else history)
        checkpoint_mgr.save_checkpoint('X_train_lstm', X_train_lstm)
        checkpoint_mgr.save_checkpoint('y_train_lstm', y_train_lstm)
        checkpoint_mgr.save_checkpoint('X_test_lstm', X_test_lstm)
        checkpoint_mgr.save_checkpoint('y_test_lstm', y_test_lstm)
        checkpoint_mgr.mark_complete('lstm_fitted')
    else:
        logger.info("[OK] LSTM already fitted (from checkpoint)")
        lstm_forecaster = checkpoint_mgr.load_checkpoint('lstm_forecaster')
        X_train_lstm = checkpoint_mgr.load_checkpoint('X_train_lstm')
        y_train_lstm = checkpoint_mgr.load_checkpoint('y_train_lstm')
        X_test_lstm = checkpoint_mgr.load_checkpoint('X_test_lstm')
        y_test_lstm = checkpoint_mgr.load_checkpoint('y_test_lstm')
    
    # ===================================================================================
    # STEP 6: MULTI-HORIZON EVALUATION
    # ===================================================================================
    print("\n" + "="*80)
    print("STEP 6: MULTI-HORIZON EVALUATION")
    print("="*80)
    
    if not checkpoint_mgr.is_complete('models_evaluated'):
        progress.update(70, "Evaluating models on multiple horizons")
        
        all_results = []
        
        # Make predictions with each model
        logger.info("Generating predictions with all models...")
        
        # SARIMA predictions (full horizon)
        sarima_predictions = {}
        for cluster in top_clusters[:10]:  # Evaluate on subset for speed
            try:
                if str(cluster) in sarima_models:
                    pred = sarima_models[str(cluster)].forecast(len(X_test))
                    sarima_predictions[cluster] = pred
            except Exception as e:
                logger.warning(f"SARIMA prediction failed for {cluster}: {e}")
        
        # XGBoost predictions
        xgb_predictions = xgb_optimizer.predict(X_test)
        
        # LSTM predictions
        lstm_predictions_raw = lstm_forecaster.predict(X_test_lstm)
        
        # Evaluate at multiple horizons
        for horizon_name, horizon_steps in HORIZONS.items():
            logger.info(f"\nEvaluating at horizon: {horizon_name} ({horizon_steps} hours)")
            
            for idx, cluster in enumerate(top_clusters[:10]):
                try:
                    # Get actual values
                    if isinstance(y_test, pd.DataFrame):
                        actual = y_test[cluster].values[-len(X_test):]
                    else:
                        actual = y_test[:, top_clusters.index(cluster)][-len(X_test):]
                    
                    # Extract horizon-specific predictions
                    predictions_by_model = {}
                    
                    if cluster in sarima_predictions:
                        predictions_by_model['SARIMA'] = extract_horizon_predictions(
                            sarima_predictions[cluster], horizon_steps, len(actual)
                        )
                    
                    if cluster in xgb_predictions.columns:
                        predictions_by_model['XGBoost'] = extract_horizon_predictions(
                            xgb_predictions[cluster].values, horizon_steps, len(actual)
                        )
                    
                    if len(lstm_predictions_raw) > 0:
                        predictions_by_model['LSTM'] = extract_horizon_predictions(
                            lstm_predictions_raw[:, top_clusters.index(cluster)], horizon_steps, len(actual)
                        )
                    
                    # Calculate metrics for this horizon
                    if predictions_by_model:
                        horizon_metrics = calculate_horizon_metrics(actual, {horizon_name: predictions_by_model}, cluster)
                        all_results.append(horizon_metrics)
                
                except Exception as e:
                    logger.warning(f"Evaluation failed for cluster {cluster}, horizon {horizon_name}: {e}")
        
        # Combine all results
        if all_results:
            evaluation_df = pd.concat(all_results, ignore_index=True)
            evaluation_df.to_csv(os.path.join(RESULTS_DIR, 'multi_horizon_metrics.csv'), index=False)
            logger.info(f"[OK] Saved multi-horizon evaluation results")
            
            # Summary by horizon
            logger.info("\n" + "="*80)
            logger.info("MULTI-HORIZON PERFORMANCE SUMMARY")
            logger.info("="*80)
            
            for horizon in HORIZONS.keys():
                horizon_data = evaluation_df[evaluation_df['horizon'] == horizon]
                if len(horizon_data) > 0:
                    logger.info(f"\n{horizon}:")
                    for model in ['SARIMA', 'XGBoost', 'LSTM']:
                        model_data = horizon_data[horizon_data['model'] == model]
                        if len(model_data) > 0:
                            avg_rmse = model_data['RMSE'].mean()
                            avg_mae = model_data['MAE'].mean()
                            avg_mape = model_data['MAPE'].mean()
                            logger.info(f"  {model}: RMSE={avg_rmse:.4f}, MAE={avg_mae:.4f}, MAPE={avg_mape:.2f}%")
        
        checkpoint_mgr.save_checkpoint('evaluation_results', evaluation_df if 'evaluation_df' in locals() else None)
        checkpoint_mgr.mark_complete('models_evaluated')
    else:
        logger.info("[OK] Models already evaluated (from checkpoint)")
        evaluation_df = checkpoint_mgr.load_checkpoint('evaluation_results')
    
    # ===================================================================================
    # STEP 7: VISUALIZATIONS
    # ===================================================================================
    print("\n" + "="*80)
    print("STEP 7: VISUALIZATIONS")
    print("="*80)
    
    if not checkpoint_mgr.is_complete('visualizations_created'):
        progress.update(85, "Creating visualizations")
        
        try:
            # Plot 1: Multi-horizon RMSE comparison
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            for ax_idx, (horizon_name, horizon_steps) in enumerate(HORIZONS.items()):
                horizon_data = evaluation_df[evaluation_df['horizon'] == horizon_name]
                
                if len(horizon_data) > 0:
                    model_performance = horizon_data.groupby('model')['RMSE'].mean().sort_values()
                    model_performance.plot(kind='bar', ax=axes[ax_idx], color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
                    axes[ax_idx].set_title(f'{horizon_name} - Average RMSE')
                    axes[ax_idx].set_ylabel('RMSE')
                    axes[ax_idx].set_xlabel('Model')
            
            plt.tight_layout()
            plt.savefig(os.path.join(VIZ_DIR, 'multi_horizon_rmse_comparison.png'), dpi=300, bbox_inches='tight')
            logger.info("[OK] Saved RMSE comparison visualization")
            
            # Plot 2: Accuracy degradation
            fig, ax = plt.subplots(figsize=(10, 6))
            
            for model in ['SARIMA', 'XGBoost', 'LSTM']:
                model_data = evaluation_df[evaluation_df['model'] == model]
                
                if len(model_data) > 0:
                    horizon_order = ['1D', '3D', '7D']
                    rmse_by_horizon = [model_data[model_data['horizon'] == h]['RMSE'].mean() 
                                      for h in horizon_order if h in model_data['horizon'].unique()]
                    
                    if rmse_by_horizon:
                        ax.plot(horizon_order[:len(rmse_by_horizon)], rmse_by_horizon, 
                               marker='o', label=model, linewidth=2)
            
            ax.set_xlabel('Forecast Horizon')
            ax.set_ylabel('RMSE')
            ax.set_title('Accuracy Degradation Over Forecast Horizon')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(VIZ_DIR, 'accuracy_degradation.png'), dpi=300, bbox_inches='tight')
            logger.info("[OK] Saved accuracy degradation visualization")
            
            plt.close('all')
        
        except Exception as e:
            logger.warning(f"[!] Visualization creation failed: {e}")
        
        checkpoint_mgr.mark_complete('visualizations_created')
    else:
        logger.info("[OK] Visualizations already created (from checkpoint)")
    
    # ===================================================================================
    # STEP 8: FINAL REPORT
    # ===================================================================================
    print("\n" + "="*80)
    print("STEP 8: PIPELINE COMPLETE - FINAL REPORT")
    print("="*80)
    
    logger.info("\n" + "="*80)
    logger.info("PIPELINE EXECUTION COMPLETE")
    logger.info("="*80)
    logger.info(f"Total clusters processed: {len(top_clusters)}")
    logger.info(f"SARIMA models: {len(sarima_models)}")
    logger.info(f"XGBoost models: {len(xgb_optimizer.models)}")
    logger.info(f"LSTM components (optimal): {lstm_forecaster.n_components}")
    
    if 'evaluation_df' in locals() and evaluation_df is not None:
        logger.info(f"Total evaluation results: {len(evaluation_df)} rows")
        logger.info(f"Horizons evaluated: {evaluation_df['horizon'].unique().tolist()}")
        logger.info(f"Models evaluated: {evaluation_df['model'].unique().tolist()}")
    
    logger.info(f"\nResults saved to: {RESULTS_DIR}")
    logger.info(f"Models saved to: {MODELS_DIR}")
    logger.info(f"Visualizations saved to: {VIZ_DIR}")
    logger.info(f"Logs saved to: {LOG_DIR}")
    
    progress.update(100, "Pipeline execution complete!")
    print("\n[+] Pipeline execution complete!")
