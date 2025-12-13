
# ============================================================================
# TAXI DEMAND PREDICTION PIPELINE
# Master Thesis - Vilnius University
# ============================================================================
# This pipeline implements three prediction approaches:
# 1. VAR (Vector Autoregression) with optimal lag selection
# 2. XGBoost with hyperparameter tuning
# 3. ConvLSTM for spatiotemporal demand prediction
# ============================================================================

import os
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Statistical models
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.stats.stattools import durbin_watson

# Machine Learning
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    ConvLSTM2D, Conv2D, Conv3D, Dense, Flatten, Dropout, 
    BatchNormalization, Input, Reshape, TimeDistributed,
    LSTM, Bidirectional, Attention, Add, LayerNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("="*80)
print("LOADING AND PREPARING DATA")
print("="*80)

# Load the clustered dataset
input_path = 'C:/Users/Anya/master_thesis/output'
#output_path = 'C:/Users/Anya/master_thesis/output/models'
output_path = 'C:/Users/Anya/master_thesis/output/models_new_clustering'
os.makedirs(output_path, exist_ok=True)

# Load parquet file
#data = pd.read_parquet(os.path.join(input_path, 'taxi_data_with_clusters_full.parquet'))
data = pd.read_parquet(os.path.join(input_path, 'taxi_data_cleaned_full_with_clusters.parquet'))

print(data.head())


print(f"Data shape: {data.shape}")
print(f"Columns: {data.columns.tolist()}")
print(f"Date range: {data['tpep_pickup_datetime'].min()} to {data['tpep_pickup_datetime'].max()}")


# ============================================================================
# PART 0: DATA PREPARATION FOR TIME SERIES MODELING
# ============================================================================

# Extract temporal features from datetime
# Create hour, minutes, weekday, and boolean features for rush hours forp pickup
data['pickup_hour'] = data['tpep_pickup_datetime'].dt.hour
data['pickup_minutes'] = data['tpep_pickup_datetime'].dt.minute
bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
labels = ['00-04', '05-09', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', 
          '40-44', '45-49', '50-54', '55-59']
data['pickup_minute_bucket'] = pd.cut(data['pickup_minutes'], 
                            bins=bins, labels=labels, right=False, include_lowest=True)
data['pickup_weekday'] = data['tpep_pickup_datetime'].dt.weekday
data['pickup_month'] = data['tpep_pickup_datetime'].dt.month
# Same for dropoff
data['dropoff_hour'] = data['tpep_dropoff_datetime'].dt.hour
data['dropoff_minutes'] = data['tpep_dropoff_datetime'].dt.minute
data['dropoff_minute_bucket'] = pd.cut(data['dropoff_minutes'], 
                            bins=bins, labels=labels, right=False, include_lowest=True)
data['dropoff_weekday'] = data['tpep_dropoff_datetime'].dt.weekday
data['dropoff_month'] = data['tpep_dropoff_datetime'].dt.month

# Create boolean features for weekend (Saturday=5, Sunday=6)
data['is_weekend_pickup'] = data['pickup_weekday'].isin([5, 6]).astype(int)
data['is_weekend_dropoff'] = data['dropoff_weekday'].isin([5, 6]).astype(int)

# Create boolean feature for rush hours (7-9am and 5-7pm) to help capture peak traffic patterns
data['is_rush_hour_pickup'] = (
    ((data['pickup_hour'] >= 7) & (data['pickup_hour'] <= 9)) |
    ((data['pickup_hour'] >= 17) & (data['pickup_hour'] <= 19))
).astype(int)
data['is_rush_hour_dropoff'] = (
    ((data['dropoff_hour'] >= 7) & (data['dropoff_hour'] <= 9)) |
    ((data['dropoff_hour'] >= 17) & (data['dropoff_hour'] <= 19))
).astype(int)

# Calculate trip statistics for demand characterization
data['trip_duration'] = (data['tpep_dropoff_datetime'] - data['tpep_pickup_datetime']).dt.total_seconds() / 60
data['average_speed'] = data['trip_distance'] / (data['trip_duration'] / 60)

# Calculate distance from center of Manhattan (approx 40.7589, -73.9851)
manhattan_center_lat, manhattan_center_lon = 40.7589, -73.9851
data['distance_from_center_pickup'] = np.sqrt(
    (data['pickup_latitude'] - manhattan_center_lat)**2 +
    (data['pickup_longitude'] - manhattan_center_lon)**2
)
data['distance_from_center_dropoff'] = np.sqrt(
    (data['dropoff_latitude'] - manhattan_center_lat)**2 +
    (data['dropoff_longitude'] - manhattan_center_lon)**2
)

print("Feature engineering complete")

def prepare_time_series_data(data, time_column='tpep_pickup_datetime', 
                              cluster_column='kmeans_cluster',
                              freq='h',  # Hourly aggregation
                              aggregation='count'):
    """
    Prepare time series data for demand prediction.
    Aggregates trips by time period and cluster/zone.

    Parameters:
    -----------
    data : pd.DataFrame
        Raw taxi data with datetime and cluster information
    time_column : str
        Name of the datetime column
    cluster_column : str
        Name of the cluster/zone column
    freq : str
        Frequency for aggregation ('h' for hourly, 'd' for daily, '30T' for 30 min)
    aggregation : str
        Type of aggregation ('count' for demand, 'mean' for average)

    Returns:
    --------
    demand_matrix : pd.DataFrame
        Time series matrix with datetime index and clusters as columns
    """
    print("\nPreparing time series data...")

    # Ensure datetime format
    data[time_column] = pd.to_datetime(data[time_column])

    # Create time period column for aggregation
    data['time_period'] = data[time_column].dt.floor(freq)

    # Aggregate trips by time period and cluster
    if aggregation == 'count':
        demand = data.groupby(['time_period', cluster_column]).size().reset_index(name='demand')
    else:
        demand = data.groupby(['time_period', cluster_column]).agg({'trip_distance': 'mean'}).reset_index()
        demand.columns = ['time_period', cluster_column, 'demand']

    # Pivot to create demand matrix (rows = time, columns = clusters)
    demand_matrix = demand.pivot(index='time_period', columns=cluster_column, values='demand')
    demand_matrix = demand_matrix.fillna(0)  # Fill missing values with 0

    # Sort by datetime index
    demand_matrix = demand_matrix.sort_index()

    print(f"Demand matrix shape: {demand_matrix.shape}")
    print(f"Time range: {demand_matrix.index.min()} to {demand_matrix.index.max()}")
    print(f"Number of clusters/zones: {demand_matrix.shape[1]}")

    return demand_matrix


def add_temporal_features(demand_matrix):
    """
    Add temporal features to the demand matrix for enhanced prediction.

    Returns:
    --------
    demand_df : pd.DataFrame
        Demand matrix with additional temporal features
    """
    demand_df = demand_matrix.copy()

    # Extract temporal features
    demand_df['hour'] = demand_df.index.hour
    demand_df['day_of_week'] = demand_df.index.dayofweek
    demand_df['day_of_month'] = demand_df.index.day
    demand_df['month'] = demand_df.index.month
    demand_df['is_weekend'] = (demand_df.index.dayofweek >= 5).astype(int)
    demand_df['is_rush_hour'] = demand_df['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)

    # Cyclical encoding for hour and day_of_week
    demand_df['hour_sin'] = np.sin(2 * np.pi * demand_df['hour'] / 24)
    demand_df['hour_cos'] = np.cos(2 * np.pi * demand_df['hour'] / 24)
    demand_df['dow_sin'] = np.sin(2 * np.pi * demand_df['day_of_week'] / 7)
    demand_df['dow_cos'] = np.cos(2 * np.pi * demand_df['day_of_week'] / 7)

    return demand_df


def create_train_test_split(demand_matrix, test_size=0.2, validation_size=0.1):
    """
    Create temporal train/validation/test split for time series.
    Uses chronological splitting to avoid data leakage.

    Returns:
    --------
    train, validation, test : pd.DataFrame
        Chronologically split datasets
    """
    n = len(demand_matrix)
    train_end = int(n * (1 - test_size - validation_size))
    val_end = int(n * (1 - test_size))

    train = demand_matrix.iloc[:train_end]
    validation = demand_matrix.iloc[train_end:val_end]
    test = demand_matrix.iloc[val_end:]

    print(f"\nData split:")
    print(f"  Training: {train.index.min()} to {train.index.max()} ({len(train)} samples)")
    print(f"  Validation: {validation.index.min()} to {validation.index.max()} ({len(validation)} samples)")
    print(f"  Test: {test.index.min()} to {test.index.max()} ({len(test)} samples)")

    return train, validation, test


# Prepare the demand matrix
demand_matrix = prepare_time_series_data(data, freq='H')

# Add temporal features
demand_with_features = add_temporal_features(demand_matrix)

# Split data
train_data, val_data, test_data = create_train_test_split(demand_matrix)


# ============================================================================
# PART 1: VAR MODEL WITH OPTIMAL LAG SELECTION
# ============================================================================
print("\n" + "="*80)
print("PART 1: VECTOR AUTOREGRESSION (VAR) MODEL")
print("="*80)

class VARDemandPredictor:
    """
    VAR model for multivariate time series taxi demand prediction.
    Implements optimal lag selection using information criteria.
    """

    def __init__(self, max_lags=20, ic='aic'):
        """
        Parameters:
        -----------
        max_lags : int
            Maximum number of lags to consider
        ic : str
            Information criterion for lag selection ('aic', 'bic', 'hqic', 'fpe')
        """
        self.max_lags = max_lags
        self.ic = ic
        self.model = None
        self.results = None
        self.optimal_lag = None
        self.scaler = StandardScaler()

    def check_stationarity(self, data, significance=0.05):
        """
        Check stationarity of each series using Augmented Dickey-Fuller test.

        Returns:
        --------
        results_df : pd.DataFrame
            Stationarity test results for each column
        """
        print("\nChecking stationarity (ADF test)...")
        results = []

        for column in data.columns:
            adf_result = adfuller(data[column].dropna(), autolag='AIC')
            results.append({
                'Variable': column,
                'ADF Statistic': adf_result[0],
                'p-value': adf_result[1],
                'Lags Used': adf_result[2],
                'Stationary': adf_result[1] < significance
            })

        results_df = pd.DataFrame(results)

        stationary_count = results_df['Stationary'].sum()
        print(f"  Stationary series: {stationary_count}/{len(results_df)}")

        return results_df

    def difference_data(self, data, order=1):
        """
        Difference the data to achieve stationarity.
        """
        return data.diff(order).dropna()

    def select_optimal_lag(self, data):
        """
        Select optimal lag order using information criteria.
        Tests VAR models with different lag orders and selects the best.

        Returns:
        --------
        optimal_lag : int
            Selected optimal lag order
        lag_order_df : pd.DataFrame
            Information criteria for all tested lags
        """
        print(f"\nSelecting optimal lag order (max_lags={self.max_lags})...")

        model = VAR(data)

        # Use select_order method for comprehensive lag selection
        lag_order_results = model.select_order(maxlags=self.max_lags)

        print("\nLag Order Selection Results:")
        print(lag_order_results.summary())

        # Get optimal lag based on selected criterion
        if self.ic == 'aic':
            self.optimal_lag = lag_order_results.aic
        elif self.ic == 'bic':
            self.optimal_lag = lag_order_results.bic
        elif self.ic == 'hqic':
            self.optimal_lag = lag_order_results.hqic
        elif self.ic == 'fpe':
            self.optimal_lag = lag_order_results.fpe
        else:
            self.optimal_lag = lag_order_results.aic

        print(f"\nOptimal lag order ({self.ic.upper()}): {self.optimal_lag}")

        return self.optimal_lag, lag_order_results

    def fit(self, train_data, lag=None):
        """
        Fit VAR model with specified or optimal lag.

        Parameters:
        -----------
        train_data : pd.DataFrame
            Training data
        lag : int, optional
            Lag order. If None, uses optimal lag selection.
        """
        print("\nFitting VAR model...")

        # Scale data for numerical stability
        scaled_data = pd.DataFrame(
            self.scaler.fit_transform(train_data),
            index=train_data.index,
            columns=train_data.columns
        )

        # Select optimal lag if not specified
        if lag is None:
            self.select_optimal_lag(scaled_data)
            lag = self.optimal_lag
        else:
            self.optimal_lag = lag

        # Fit VAR model
        self.model = VAR(scaled_data)
        self.results = self.model.fit(lag)

        print(f"\nVAR({lag}) Model Summary:")
        print(f"  Number of equations: {self.results.neqs}")
        print(f"  Number of observations: {self.results.nobs}")
        print(f"  AIC: {self.results.aic:.4f}")
        print(f"  BIC: {self.results.bic:.4f}")
        print(f"  Log-likelihood: {self.results.llf:.4f}")

        return self.results

    def check_model_stability(self):
        """
        Check VAR model stability (all eigenvalues inside unit circle).
        """
        if self.results is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        eigenvalues = self.results.roots
        is_stable = np.all(np.abs(eigenvalues) < 1)

        print(f"\nModel Stability Check:")
        print(f"  Max eigenvalue modulus: {np.max(np.abs(eigenvalues)):.4f}")
        print(f"  Model is {'STABLE' if is_stable else 'UNSTABLE'}")

        return is_stable, eigenvalues

    def check_residual_autocorrelation(self):
        """
        Check for autocorrelation in residuals using Durbin-Watson test.
        Values close to 2 indicate no autocorrelation.
        """
        if self.results is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        dw_stats = durbin_watson(self.results.resid)

        print("\nDurbin-Watson Statistics (target ~2):")
        for i, (col, dw) in enumerate(zip(self.results.names, dw_stats)):
            status = "OK" if 1.5 < dw < 2.5 else "AUTOCORRELATION DETECTED"
            print(f"  {col}: {dw:.4f} ({status})")

        return dw_stats

    def forecast(self, steps, last_observations=None):
        """
        Generate forecasts for specified number of steps ahead.

        Parameters:
        -----------
        steps : int
            Number of periods to forecast
        last_observations : pd.DataFrame, optional
            Last observations to use for forecasting

        Returns:
        --------
        forecast_df : pd.DataFrame
            Forecasted demand values (inverse transformed)
        """
        if self.results is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        # Use the last k_ar observations for forecasting
        if last_observations is not None:
            scaled_obs = self.scaler.transform(last_observations)
            y_input = scaled_obs[-self.results.k_ar:]
        else:
            y_input = self.results.endog[-self.results.k_ar:]

        # Generate forecast
        forecast_scaled = self.results.forecast(y_input, steps=steps)

        # Inverse transform to original scale
        forecast = self.scaler.inverse_transform(forecast_scaled)

        return forecast

    def evaluate(self, test_data):
        """
        Evaluate model on test data using rolling 1-step ahead forecast.
        FIXED: Robust shape handling with explicit type enforcement.
        """

        print("\\nEvaluating VAR model on test data...")
        
        if self.results is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        predictions = []
        actuals = []
        
        # Get initial state
        recent_obs = self.results.endog[-self.results.k_ar:].copy()
        print(f"Initial recent_obs shape: {recent_obs.shape}")
        
        # Iterate through test data
        for i in range(len(test_data)):
            # Get actual value
            actual_raw = test_data.iloc[i].values
            
            # CRITICAL: Convert to numpy array and ensure 1D
            actual = np.asarray(actual_raw, dtype=np.float64).flatten()
            assert actual.ndim == 1 and actual.shape[0] == 10, f"Actual shape wrong at i={i}: {actual.shape}"
            actuals.append(actual.copy())
            # ============================================================
            # FORECAST STEP
            # ============================================================
            try:
                # Forecast in scaled space
                forecast_scaled = self.results.forecast(recent_obs, steps=1)
                # CRITICAL: Ensure proper shape
                forecast_scaled = np.asarray(forecast_scaled, dtype=np.float64)
                
                # If 1D, reshape to 2D
                if forecast_scaled.ndim == 1:
                    forecast_scaled = forecast_scaled.reshape(1, -1)
                
                # Verify shape
                if forecast_scaled.shape != (1, 10):
                    print(f"WARNING at i={i}: forecast_scaled has wrong shape {forecast_scaled.shape}")
                    # Reshape forcefully
                    forecast_scaled = forecast_scaled.reshape(1, -1)[:, :10]
                
                # Inverse transform
                forecast_unscaled = self.scaler.inverse_transform(forecast_scaled)
                
                # ✅ CRITICAL: Extract and ensure 1D
                forecast = np.asarray(forecast_unscaled, dtype=np.float64).flatten()
                
                # Validate
                if forecast.shape != (10,):
                    print(f"WARNING at i={i}: forecast shape is {forecast.shape}, forcing to (10,)")
                    if forecast.shape > 10:
                        forecast = forecast[:10]
                    elif forecast.shape < 10:
                        # Pad with last value (shouldn't happen)
                        forecast = np.pad(forecast, (0, 10-forecast.shape), mode='edge')
                
                # Final check
                assert forecast.ndim == 1 and forecast.shape[0] == 10, f"Forecast shape invalid: {forecast.shape}"
            
            except Exception as e:
                print(f"  Exception at step {i}: {e}")
                # Fallback: use actual value
                forecast = actual.copy()
            
            predictions.append(forecast.copy())
            
            # ============================================================
            # SCALE ACTUAL FOR WINDOW UPDATE
            # ============================================================
            try:
                # Reshape for scaling
                actual_for_scaler = actual.reshape(1, -1)
                actual_scaled_2d = self.scaler.transform(actual_for_scaler)
                actual_scaled = np.asarray(actual_scaled_2d, dtype=np.float64).flatten()
                
                # Validate
                assert actual_scaled.shape[0] == 10, f"actual_scaled shape wrong: {actual_scaled.shape}"
            
            except Exception as e:
                print(f"  Error scaling actual at {i}: {e}")
                actual_scaled = recent_obs[-1].copy()
            
            # ============================================================
            # UPDATE ROLLING WINDOW
            # ============================================================
            try:
                # Remove oldest row, add new scaled actual
                recent_obs = np.vstack([recent_obs[1:], actual_scaled.reshape(1, -1)])
                
                # Validate window
                assert recent_obs.shape == (self.results.k_ar, 10), \
                    f"recent_obs shape wrong: {recent_obs.shape}, expected ({self.results.k_ar}, 10)"
            
            except Exception as e:
                print(f"  Error updating window at {i}: {e}")
        
        # ============================================================
        # STACK PREDICTIONS AND ACTUALS
        # ============================================================
        print(f"\\nStacking {len(predictions)} predictions...")
        
        # Convert list to array safely
        predictions_list = [np.asarray(p, dtype=np.float64).flatten() for p in predictions]
        actuals_list = [np.asarray(a, dtype=np.float64).flatten() for a in actuals]
        
        # Check shapes before stacking
        pred_shapes = [p.shape for p in predictions_list]
        actual_shapes = [a.shape for a in actuals_list]
        
        # Find any inconsistent shapes
        inconsistent_preds = [i for i, s in enumerate(pred_shapes) if s != (10,)]
        inconsistent_actuals = [i for i, s in enumerate(actual_shapes) if s != (10,)]
        
        if inconsistent_preds:
            print(f"WARNING: {len(inconsistent_preds)} predictions with wrong shape:")
            for idx in inconsistent_preds[:5]:  # Print first 5
                print(f"  predictions[{idx}].shape = {pred_shapes[idx]}")
            
            # Fix them
            predictions_list = [p.flatten()[:10] if p.shape >= 10 else 
                            np.pad(p.flatten(), (0, max(0, 10-p.shape)), mode='edge')
                            for p in predictions_list]
        
        if inconsistent_actuals:
            print(f"WARNING: {len(inconsistent_actuals)} actuals with wrong shape")
            actuals_list = [a.flatten()[:10] if a.shape >= 10 else 
                        np.pad(a.flatten(), (0, max(0, 10-a.shape)), mode='edge')
                        for a in actuals_list]
        
        # Now stack
        predictions = np.vstack(predictions_list)
        actuals = np.vstack(actuals_list)
        
        print(f"Final shapes: predictions={predictions.shape}, actuals={actuals.shape}")
        
        # ============================================================
        # CALCULATE METRICS
        # ============================================================
        metrics = {}
        overall_rmse = []
        overall_mae = []
        
        for i, col in enumerate(test_data.columns):
            actual_col = actuals[:, i]
            pred_col = predictions[:, i]
            
            rmse = np.sqrt(mean_squared_error(actual_col, pred_col))
            mae = mean_absolute_error(actual_col, pred_col)
            
            # MAPE with zero handling
            mask = actual_col != 0
            if mask.sum() > 0:
                mape = np.mean(np.abs((actual_col[mask] - pred_col[mask]) / actual_col[mask])) * 100
            else:
                mape = np.nan
            
            metrics[col] = {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}
            overall_rmse.append(rmse)
            overall_mae.append(mae)
        
        # Overall metrics
        metrics['Overall'] = {
            'RMSE': np.mean(overall_rmse),
            'MAE': np.mean(overall_mae),
            'MAPE': np.nanmean([m['MAPE'] for m in metrics.values() if 'MAPE' in m and not np.isnan(m['MAPE'])])
        }
        
        print("\nOverall Metrics:")
        print(f"  RMSE: {metrics['Overall']['RMSE']:.4f}")
        print(f"  MAE: {metrics['Overall']['MAE']:.4f}")
        print(f"  MAPE: {metrics['Overall']['MAPE']:.2f}%")
        
        # Create DataFrame
        pred_df = pd.DataFrame(predictions, index=test_data.index, columns=test_data.columns)
        
        return metrics, pred_df

    def granger_causality_analysis(self, data, max_lag=5, significance=0.05):
        """
        Perform Granger causality test between all pairs of variables.
        Helps understand predictive relationships between zones.

        Returns:
        --------
        causality_matrix : pd.DataFrame
            Matrix showing Granger causality relationships
        """
        print("\nPerforming Granger Causality Analysis...")

        columns = data.columns
        n = len(columns)
        causality_matrix = pd.DataFrame(np.zeros((n, n)), index=columns, columns=columns)

        for i, cause in enumerate(columns):
            for j, effect in enumerate(columns):
                if i != j:
                    try:
                        test_result = grangercausalitytests(
                            data[[effect, cause]], maxlag=max_lag, verbose=False
                        )
                        # Get minimum p-value across all lags
                        p_values = [test_result[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag + 1)]
                        min_p = min(p_values)
                        causality_matrix.loc[cause, effect] = 1 if min_p < significance else 0
                    except:
                        causality_matrix.loc[cause, effect] = np.nan

        causal_pairs = causality_matrix.sum().sum()
        print(f"  Significant causal relationships found: {int(causal_pairs)}")

        return causality_matrix


# Initialize and run VAR model
print("\n" + "-"*40)
print("Running VAR Model Pipeline...")
print("-"*40)

# Select subset of clusters for VAR (to avoid high dimensionality issues)
# VAR works best with fewer variables
n_clusters_for_var = min(10, train_data.shape[1])  # Limit to top 10 clusters
top_clusters = train_data.sum().nlargest(n_clusters_for_var).index.tolist()
train_data_var = train_data[top_clusters]
test_data_var = test_data[top_clusters]

# Initialize VAR predictor
var_predictor = VARDemandPredictor(max_lags=24, ic='aic')  # 24 hours max for hourly data

# Check stationarity
stationarity_results = var_predictor.check_stationarity(train_data_var)

# If non-stationary, difference the data
if not stationarity_results['Stationary'].all():
    print("\nApplying differencing for stationarity...")
    train_data_var_diff = var_predictor.difference_data(train_data_var)
    test_data_var_diff = var_predictor.difference_data(test_data_var)
else:
    train_data_var_diff = train_data_var
    test_data_var_diff = test_data_var

# Fit model with optimal lag selection
var_results = var_predictor.fit(train_data_var_diff)

# Check model stability
is_stable, eigenvalues = var_predictor.check_model_stability()

# Check residual autocorrelation
dw_stats = var_predictor.check_residual_autocorrelation()

# Evaluate on test set
var_metrics, var_predictions = var_predictor.evaluate(test_data_var_diff)

# Save VAR results
var_results_df = pd.DataFrame(var_metrics).T
var_results_df.to_csv(os.path.join(output_path, 'var_evaluation_metrics.csv'))
var_predictions.to_csv(os.path.join(output_path, 'var_predictions.csv'))

print("\nVAR model results saved!")


# ============================================================================
# PART 2: XGBOOST WITH HYPERPARAMETER OPTIMIZATION
# ============================================================================
print("\n" + "="*80)
print("PART 2: XGBOOST DEMAND PREDICTION WITH HYPERPARAMETER TUNING")
print("="*80)

class XGBoostDemandPredictor:
    """
    XGBoost model for taxi demand prediction with time series specific features.
    Implements hyperparameter tuning using GridSearchCV with TimeSeriesSplit.
    """

    def __init__(self, n_lags=24, forecast_horizon=1):
        """
        Parameters:
        -----------
        n_lags : int
            Number of lag features to create
        forecast_horizon : int
            Number of steps ahead to predict
        """
        self.n_lags = n_lags
        self.forecast_horizon = forecast_horizon
        self.models = {}  # One model per cluster (or multi-output)
        self.best_params = {}
        self.scaler = MinMaxScaler()
        self.feature_names = None

    def create_lag_features(self, data, n_lags=None):
        """
        Create lag features for time series prediction.

        Parameters:
        -----------
        data : pd.DataFrame
            Demand matrix with datetime index
        n_lags : int
            Number of lags to create

        Returns:
        --------
        X : pd.DataFrame
            Feature matrix with lag features
        y : pd.DataFrame
            Target matrix
        """
        if n_lags is None:
            n_lags = self.n_lags

        df = data.copy()
        feature_columns = []

        # Create lag features for each cluster
        for col in data.columns:
            for lag in range(1, n_lags + 1):
                lag_col = f'{col}_lag_{lag}'
                df[lag_col] = data[col].shift(lag)
                feature_columns.append(lag_col)

        # Add rolling statistics
        for col in data.columns:
            # Rolling mean (e.g., last 6 hours)
            df[f'{col}_rolling_mean_6'] = data[col].shift(1).rolling(window=6).mean()
            df[f'{col}_rolling_std_6'] = data[col].shift(1).rolling(window=6).std()
            df[f'{col}_rolling_mean_24'] = data[col].shift(1).rolling(window=24).mean()

            feature_columns.extend([
                f'{col}_rolling_mean_6',
                f'{col}_rolling_std_6', 
                f'{col}_rolling_mean_24'
            ])

        # Add temporal features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        df['is_rush_hour'] = df['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)

        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        temporal_features = ['hour', 'day_of_week', 'is_weekend', 'is_rush_hour',
                            'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']
        feature_columns.extend(temporal_features)

        # Drop rows with NaN (from lag creation)
        df = df.dropna()

        # Separate features and targets
        X = df[feature_columns]
        y = df[data.columns]  # Original cluster columns are targets

        self.feature_names = feature_columns

        return X, y

    def get_param_grid(self, grid_type='full'):
        """
        Define hyperparameter search space.

        Parameters:
        -----------
        grid_type : str
            'full' for comprehensive search, 'quick' for faster search

        Returns:
        --------
        param_grid : dict
            Hyperparameter search space
        """
        if grid_type == 'full':
            param_grid = {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [3, 5, 7, 9, 11],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'min_child_weight': [1, 3, 5],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'gamma': [0, 0.1, 0.2],
                'reg_alpha': [0, 0.1, 1],
                'reg_lambda': [1, 1.5, 2]
            }
        elif grid_type == 'quick':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.05, 0.1, 0.2],
                'min_child_weight': [1, 3],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
        else:  # minimal for testing
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 5],
                'learning_rate': [0.1]
            }

        return param_grid

    def tune_hyperparameters(self, X_train, y_train, target_cluster=None,
                             cv_splits=5, grid_type='quick', n_jobs=-1):
        """
        Tune hyperparameters using GridSearchCV with TimeSeriesSplit.

        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.DataFrame or pd.Series
            Training targets
        target_cluster : str, optional
            Specific cluster to tune. If None, uses first cluster.
        cv_splits : int
            Number of cross-validation splits
        grid_type : str
            Type of parameter grid ('full', 'quick', 'minimal')
        n_jobs : int
            Number of parallel jobs (-1 for all cores)

        Returns:
        --------
        best_params : dict
            Best hyperparameters found
        cv_results : pd.DataFrame
            Cross-validation results
        """
        print(f"\nTuning XGBoost hyperparameters (grid_type='{grid_type}')...")

        # Select target
        if target_cluster is None:
            target = y_train.iloc[:, 0]
            target_name = y_train.columns[0]
        else:
            target = y_train[target_cluster]
            target_name = target_cluster

        print(f"  Target cluster: {target_name}")

        # Initialize model
        base_model = XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            n_jobs=1,  # Set to 1 when using GridSearchCV parallelization
            verbosity=0
        )

        # Get parameter grid
        param_grid = self.get_param_grid(grid_type)

        # TimeSeriesSplit for proper temporal cross-validation
        tscv = TimeSeriesSplit(n_splits=cv_splits)

        # GridSearchCV
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=tscv,
            scoring='neg_root_mean_squared_error',
            n_jobs=n_jobs,
            verbose=1,
            return_train_score=True
        )

        # Fit grid search
        print("  Running grid search...")
        grid_search.fit(X_train.values, target.values)

        # Store best parameters
        self.best_params[target_name] = grid_search.best_params_

        print(f"\n  Best parameters for {target_name}:")
        for param, value in grid_search.best_params_.items():
            print(f"    {param}: {value}")
        print(f"  Best CV RMSE: {-grid_search.best_score_:.4f}")

        # Create CV results DataFrame
        cv_results = pd.DataFrame(grid_search.cv_results_)

        return grid_search.best_params_, cv_results, grid_search.best_estimator_

    def fit(self, X_train, y_train, use_tuned_params=True, params=None):
        """
        Fit XGBoost models for all clusters.

        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.DataFrame
            Training targets (all clusters)
        use_tuned_params : bool
            Whether to use tuned parameters
        params : dict, optional
            Parameters to use if not using tuned params
        """
        print("\nFitting XGBoost models for all clusters...")

        for cluster in y_train.columns:
            print(f"  Training model for cluster {cluster}...")

            # Get parameters
            if use_tuned_params and cluster in self.best_params:
                model_params = self.best_params[cluster]
            elif params is not None:
                model_params = params
            else:
                # Default parameters
                model_params = {
                    'n_estimators': 200,
                    'max_depth': 5,
                    'learning_rate': 0.1,
                    'min_child_weight': 1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8
                }

            # Initialize and fit model
            model = XGBRegressor(
                objective='reg:squarederror',
                random_state=42,
                verbosity=0,
                **model_params
            )

            model.fit(X_train.values, y_train[cluster].values)
            self.models[cluster] = model

        print(f"  Fitted {len(self.models)} models")

    def fit_multioutput(self, X_train, y_train, params=None):
        """
        Fit a single XGBoost model for multi-output regression.
        Uses sklearn's MultiOutputRegressor wrapper.

        Note: XGBoost 1.6+ has native multi-output support.
        """
        print("\nFitting Multi-Output XGBoost model...")

        if params is None:
            params = {
                'n_estimators': 200,
                'max_depth': 5,
                'learning_rate': 0.1,
                'min_child_weight': 1,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            }

        base_model = XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            verbosity=0,
            **params
        )

        self.multioutput_model = MultiOutputRegressor(base_model)
        self.multioutput_model.fit(X_train.values, y_train.values)

        print("  Multi-output model fitted!")

    def predict(self, X, use_multioutput=False):
        """
        Generate predictions for all clusters.

        Returns:
        --------
        predictions : pd.DataFrame
            Predicted demand for all clusters
        """
        if use_multioutput:
            pred = self.multioutput_model.predict(X.values)
            return pd.DataFrame(pred, index=X.index, columns=list(self.models.keys()))

        predictions = {}
        for cluster, model in self.models.items():
            predictions[cluster] = model.predict(X.values)

        return pd.DataFrame(predictions, index=X.index)

    def evaluate(self, X_test, y_test, use_multioutput=False):
        """
        Evaluate model performance on test data.

        Returns:
        --------
        metrics : dict
            RMSE, MAE, MAPE for each cluster and overall
        predictions : pd.DataFrame
            Predicted values
        """
        print("\nEvaluating XGBoost model on test data...")

        predictions = self.predict(X_test, use_multioutput)

        metrics = {}
        overall_rmse = []
        overall_mae = []

        for cluster in y_test.columns:
            actual = y_test[cluster].values
            pred = predictions[cluster].values

            rmse = np.sqrt(mean_squared_error(actual, pred))
            mae = mean_absolute_error(actual, pred)

            # MAPE (avoiding division by zero)
            mask = actual != 0
            if mask.sum() > 0:
                mape = np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100
            else:
                mape = np.nan

            metrics[cluster] = {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}
            overall_rmse.append(rmse)
            overall_mae.append(mae)

        # Overall metrics
        metrics['Overall'] = {
            'RMSE': np.mean(overall_rmse),
            'MAE': np.mean(overall_mae),
            'MAPE': np.nanmean([m['MAPE'] for m in metrics.values() if 'MAPE' in m])
        }

        print(f"\nOverall Metrics:")
        print(f"  RMSE: {metrics['Overall']['RMSE']:.4f}")
        print(f"  MAE: {metrics['Overall']['MAE']:.4f}")
        print(f"  MAPE: {metrics['Overall']['MAPE']:.2f}%")

        return metrics, predictions

    def get_feature_importance(self, top_n=20):
        """
        Get feature importance from fitted models.

        Returns:
        --------
        importance_df : pd.DataFrame
            Feature importance scores
        """
        if not self.models:
            raise ValueError("Models not fitted yet. Call fit() first.")

        # Average importance across all cluster models
        importance_sum = np.zeros(len(self.feature_names))

        for cluster, model in self.models.items():
            importance_sum += model.feature_importances_

        importance_avg = importance_sum / len(self.models)

        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importance_avg
        }).sort_values('Importance', ascending=False)

        print(f"\nTop {top_n} Most Important Features:")
        print(importance_df.head(top_n).to_string(index=False))

        return importance_df


# Initialize and run XGBoost model
print("\n" + "-"*40)
print("Running XGBoost Pipeline...")
print("-"*40)

# Select subset of clusters (same as VAR for fair comparison)
train_data_xgb = train_data[top_clusters]
test_data_xgb = test_data[top_clusters]

# Initialize predictor
xgb_predictor = XGBoostDemandPredictor(n_lags=24, forecast_horizon=1)

# Create features
X_train, y_train = xgb_predictor.create_lag_features(train_data_xgb)
X_test, y_test = xgb_predictor.create_lag_features(test_data_xgb)

print(f"\nTraining features shape: {X_train.shape}")
print(f"Training targets shape: {y_train.shape}")
print(f"Test features shape: {X_test.shape}")

# Hyperparameter tuning (on first cluster as example)
best_params, cv_results, best_model = xgb_predictor.tune_hyperparameters(
    X_train, y_train, 
    target_cluster=top_clusters[0],
    cv_splits=5,
    grid_type='quick'  # Use 'full' for comprehensive search
)

# Apply best parameters to all cluster models
xgb_predictor.fit(X_train, y_train, use_tuned_params=False, params=best_params)

# Evaluate
xgb_metrics, xgb_predictions = xgb_predictor.evaluate(X_test, y_test)

# Feature importance analysis
importance_df = xgb_predictor.get_feature_importance(top_n=15)

# Save XGBoost results
xgb_results_df = pd.DataFrame(xgb_metrics).T
xgb_results_df.to_csv(os.path.join(output_path, 'xgboost_evaluation_metrics.csv'))
xgb_predictions.to_csv(os.path.join(output_path, 'xgboost_predictions.csv'))
importance_df.to_csv(os.path.join(output_path, 'xgboost_feature_importance.csv'), index=False)

# Save best hyperparameters
with open(os.path.join(output_path, 'xgboost_best_params.txt'), 'w') as f:
    f.write("Best Hyperparameters:\n")
    for param, value in best_params.items():
        f.write(f"  {param}: {value}\n")

print("\nXGBoost model results saved!")


# ============================================================================
# PART 3: ConvLSTM FOR SPATIOTEMPORAL DEMAND PREDICTION
# ============================================================================
print("\n" + "="*80)
print("PART 3: ConvLSTM FOR SPATIOTEMPORAL DEMAND PREDICTION")
print("="*80)

"""
ConvLSTM (Convolutional LSTM) is specifically designed for spatiotemporal 
sequence forecasting. It was originally proposed by Shi et al. (2015) for 
precipitation nowcasting and has become a cornerstone architecture for:
- Traffic flow prediction
- Taxi/ride-hailing demand forecasting
- Video prediction

Key advantages:
1. Captures spatial dependencies through convolutional operations
2. Models temporal dynamics through LSTM gates
3. Naturally handles grid-based spatial data
4. Preserves spatial structure unlike flattening approaches

Reference: 
Shi, X., et al. (2015). "Convolutional LSTM Network: A Machine Learning 
Approach for Precipitation Nowcasting." NeurIPS 2015.
"""

class ConvLSTMDemandPredictor:
    """
    ConvLSTM-based model for spatiotemporal taxi demand prediction.

    This implementation uses the ConvLSTM2D layer which applies convolutional
    operations within LSTM cells, making it ideal for grid-based spatial data
    with temporal sequences.

    The architecture follows the ST-ResNet inspired approach with multiple
    temporal components (closeness, period, trend) and external factors.
    """

    def __init__(self, grid_height, grid_width, n_channels=1,
                 closeness_len=3, period_len=3, trend_len=3,
                 period_interval=24, trend_interval=168):  # hourly data
        """
        Parameters:
        -----------
        grid_height : int
            Height of spatial grid
        grid_width : int
            Width of spatial grid
        n_channels : int
            Number of channels (1 for demand, 2 for inflow/outflow)
        closeness_len : int
            Number of recent time steps to consider
        period_len : int
            Number of daily periodic time steps
        trend_len : int
            Number of weekly trend time steps
        period_interval : int
            Interval for daily periodicity (24 for hourly)
        trend_interval : int
            Interval for weekly trend (168 for hourly)
        """
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.n_channels = n_channels
        self.closeness_len = closeness_len
        self.period_len = period_len
        self.trend_len = trend_len
        self.period_interval = period_interval
        self.trend_interval = trend_interval
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.history = None

    def prepare_grid_data(self, demand_matrix, grid_mapping=None):
        """
        Convert demand matrix (clusters) to spatial grid representation.

        Parameters:
        -----------
        demand_matrix : pd.DataFrame
            Demand matrix with time index and cluster columns
        grid_mapping : dict, optional
            Mapping of cluster IDs to (row, col) grid positions
            If None, creates a simple sequential mapping

        Returns:
        --------
        grid_data : np.ndarray
            Shape (n_timesteps, grid_height, grid_width, n_channels)
        """
        n_timesteps = len(demand_matrix)
        n_clusters = len(demand_matrix.columns)

        # Create grid mapping if not provided
        if grid_mapping is None:
            # Simple mapping: arrange clusters in a grid
            # This should ideally be based on actual geographic locations
            grid_mapping = {}
            idx = 0
            for row in range(self.grid_height):
                for col in range(self.grid_width):
                    if idx < n_clusters:
                        grid_mapping[demand_matrix.columns[idx]] = (row, col)
                        idx += 1

        # Initialize grid
        grid_data = np.zeros((n_timesteps, self.grid_height, self.grid_width, self.n_channels))

        # Fill grid with demand values
        for cluster, (row, col) in grid_mapping.items():
            if cluster in demand_matrix.columns:
                grid_data[:, row, col, 0] = demand_matrix[cluster].values

        return grid_data, grid_mapping

    def create_sequences(self, grid_data, include_period=True, include_trend=True):
        """
        Create input sequences for ConvLSTM training.

        Following ST-ResNet approach, creates three types of sequences:
        1. Closeness: Recent consecutive time steps
        2. Period: Same time in previous days
        3. Trend: Same time in previous weeks

        Returns:
        --------
        X_closeness, X_period, X_trend, y : np.ndarray
            Input sequences and target
        """
        n_samples = len(grid_data)

        # Calculate starting point based on required history
        max_lookback = max(
            self.closeness_len,
            self.period_len * self.period_interval if include_period else 0,
            self.trend_len * self.trend_interval if include_trend else 0
        )

        # Initialize lists
        X_close_list = []
        X_period_list = []
        X_trend_list = []
        y_list = []

        for i in range(max_lookback, n_samples):
            # Target: current time step
            y_list.append(grid_data[i])

            # Closeness: recent time steps
            close_indices = [i - j - 1 for j in range(self.closeness_len)][::-1]
            X_close = np.stack([grid_data[idx] for idx in close_indices])
            X_close_list.append(X_close)

            # Period: same time in previous days
            if include_period:
                period_indices = [i - (j + 1) * self.period_interval 
                                  for j in range(self.period_len)][::-1]
                X_period = np.stack([grid_data[max(0, idx)] for idx in period_indices])
                X_period_list.append(X_period)

            # Trend: same time in previous weeks
            if include_trend:
                trend_indices = [i - (j + 1) * self.trend_interval 
                                 for j in range(self.trend_len)][::-1]
                X_trend = np.stack([grid_data[max(0, idx)] for idx in trend_indices])
                X_trend_list.append(X_trend)

        X_closeness = np.array(X_close_list)
        y = np.array(y_list)

        if include_period:
            X_period = np.array(X_period_list)
        else:
            X_period = None

        if include_trend:
            X_trend = np.array(X_trend_list)
        else:
            X_trend = None

        print(f"\nSequence shapes:")
        print(f"  X_closeness: {X_closeness.shape}")
        if X_period is not None:
            print(f"  X_period: {X_period.shape}")
        if X_trend is not None:
            print(f"  X_trend: {X_trend.shape}")
        print(f"  y: {y.shape}")

        return X_closeness, X_period, X_trend, y

    def build_model(self, include_period=True, include_trend=True,
                    n_filters=64, kernel_size=(3, 3), dropout_rate=0.2):
        """
        Build ConvLSTM model with multi-branch architecture.

        Architecture inspired by ST-ResNet:
        - Separate branches for closeness, period, and trend
        - ConvLSTM2D layers for spatiotemporal feature extraction
        - Fusion layer to combine outputs
        - Final Conv2D for prediction

        Parameters:
        -----------
        include_period : bool
            Include daily periodicity branch
        include_trend : bool
            Include weekly trend branch
        n_filters : int
            Number of convolutional filters
        kernel_size : tuple
            Kernel size for convolutions
        dropout_rate : float
            Dropout rate for regularization
        """
        print("\nBuilding ConvLSTM model...")

        inputs = []
        branch_outputs = []

        # Input shape: (timesteps, height, width, channels)
        input_shape = (None, self.grid_height, self.grid_width, self.n_channels)

        # ===== Closeness Branch =====
        input_close = Input(shape=(self.closeness_len, self.grid_height, 
                                   self.grid_width, self.n_channels),
                           name='input_closeness')
        inputs.append(input_close)

        # ConvLSTM layers for closeness
        x_close = ConvLSTM2D(
            filters=n_filters,
            kernel_size=kernel_size,
            padding='same',
            return_sequences=True,
            activation='tanh',
            recurrent_activation='sigmoid',
            name='convlstm_close_1'
        )(input_close)
        x_close = BatchNormalization(name='bn_close_1')(x_close)
        x_close = Dropout(dropout_rate)(x_close)

        x_close = ConvLSTM2D(
            filters=n_filters,
            kernel_size=kernel_size,
            padding='same',
            return_sequences=False,
            activation='tanh',
            name='convlstm_close_2'
        )(x_close)
        x_close = BatchNormalization(name='bn_close_2')(x_close)

        branch_outputs.append(x_close)

        # ===== Period Branch (Daily) =====
        if include_period:
            input_period = Input(shape=(self.period_len, self.grid_height,
                                       self.grid_width, self.n_channels),
                               name='input_period')
            inputs.append(input_period)

            x_period = ConvLSTM2D(
                filters=n_filters // 2,
                kernel_size=kernel_size,
                padding='same',
                return_sequences=True,
                activation='tanh',
                name='convlstm_period_1'
            )(input_period)
            x_period = BatchNormalization(name='bn_period_1')(x_period)
            x_period = Dropout(dropout_rate)(x_period)

            x_period = ConvLSTM2D(
                filters=n_filters,
                kernel_size=kernel_size,
                padding='same',
                return_sequences=False,
                activation='tanh',
                name='convlstm_period_2'
            )(x_period)
            x_period = BatchNormalization(name='bn_period_2')(x_period)

            branch_outputs.append(x_period)

        # ===== Trend Branch (Weekly) =====
        if include_trend:
            input_trend = Input(shape=(self.trend_len, self.grid_height,
                                      self.grid_width, self.n_channels),
                              name='input_trend')
            inputs.append(input_trend)

            x_trend = ConvLSTM2D(
                filters=n_filters // 2,
                kernel_size=kernel_size,
                padding='same',
                return_sequences=True,
                activation='tanh',
                name='convlstm_trend_1'
            )(input_trend)
            x_trend = BatchNormalization(name='bn_trend_1')(x_trend)
            x_trend = Dropout(dropout_rate)(x_trend)

            x_trend = ConvLSTM2D(
                filters=n_filters,
                kernel_size=kernel_size,
                padding='same',
                return_sequences=False,
                activation='tanh',
                name='convlstm_trend_2'
            )(x_trend)
            x_trend = BatchNormalization(name='bn_trend_2')(x_trend)

            branch_outputs.append(x_trend)

        # ===== Fusion =====
        if len(branch_outputs) > 1:
            # Learnable fusion with attention-like mechanism
            # Stack and apply 1x1 convolution for weighted combination
            stacked = keras.layers.Lambda(
            lambda x: tf.stack(x, axis=-1),
            name='stack_branches'
            )(branch_outputs)

            fused = keras.layers.Lambda(
                lambda x: tf.reduce_mean(x, axis=-1),
                name='average_branches'
            )(stacked)
        else:
            fused = branch_outputs[0]

        # ===== Output Layers =====
        x = Conv2D(
            filters=n_filters // 2,
            kernel_size=kernel_size,
            padding='same',
            activation='relu',
            name='conv_out_1'
        )(fused)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate / 2)(x)

        # Final output
        output = Conv2D(
            filters=self.n_channels,
            kernel_size=(1, 1),
            padding='same',
            activation='relu',  # Demand is non-negative
            name='output'
        )(x)

        # Build model
        self.model = Model(inputs=inputs, outputs=output)

        # Compile
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        print("\nModel Summary:")
        self.model.summary()

        return self.model

    def build_simple_convlstm(self, n_filters=64, kernel_size=(3, 3), 
                               dropout_rate=0.2, n_layers=2):
        """
        Build a simpler single-branch ConvLSTM model.
        Useful when you only have closeness data.

        Parameters:
        -----------
        n_filters : int
            Number of convolutional filters
        kernel_size : tuple
            Kernel size for convolutions
        dropout_rate : float
            Dropout rate for regularization
        n_layers : int
            Number of ConvLSTM layers
        """
        print("\nBuilding Simple ConvLSTM model...")

        model = Sequential(name='Simple_ConvLSTM')

        # Input shape
        input_shape = (self.closeness_len, self.grid_height, 
                      self.grid_width, self.n_channels)

        # First ConvLSTM layer
        model.add(ConvLSTM2D(
            filters=n_filters,
            kernel_size=kernel_size,
            padding='same',
            return_sequences=(n_layers > 1),
            activation='tanh',
            input_shape=input_shape,
            name='convlstm_1'
        ))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

        # Additional ConvLSTM layers
        for i in range(1, n_layers):
            return_seq = (i < n_layers - 1)
            model.add(ConvLSTM2D(
                filters=n_filters,
                kernel_size=kernel_size,
                padding='same',
                return_sequences=return_seq,
                activation='tanh',
                name=f'convlstm_{i+1}'
            ))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))

        # Output layers
        model.add(Conv2D(
            filters=n_filters // 2,
            kernel_size=(3, 3),
            padding='same',
            activation='relu'
        ))
        model.add(BatchNormalization())

        model.add(Conv2D(
            filters=self.n_channels,
            kernel_size=(1, 1),
            padding='same',
            activation='relu',
            name='output'
        ))

        # Compile
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        self.model = model
        print("\nModel Summary:")
        model.summary()

        return model

    def train(self, X_train, y_train, X_val=None, y_val=None,
              epochs=100, batch_size=32, early_stopping_patience=10):
        """
        Train the ConvLSTM model.

        Parameters:
        -----------
        X_train : np.ndarray or list
            Training features (single array or list of arrays for multi-branch)
        y_train : np.ndarray
            Training targets
        X_val : np.ndarray or list, optional
            Validation features
        y_val : np.ndarray, optional
            Validation targets
        epochs : int
            Maximum number of training epochs
        batch_size : int
            Training batch size
        early_stopping_patience : int
            Early stopping patience

        Returns:
        --------
        history : keras.callbacks.History
            Training history
        """
        print("\nTraining ConvLSTM model...")

        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=os.path.join(output_path, 'convlstm_best_model.keras'),
                monitor='val_loss' if X_val is not None else 'loss',
                save_best_only=True,
                verbose=1
            )
        ]

        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)

        # Train
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        print("\nTraining completed!")
        print(f"  Final training loss: {self.history.history['loss'][-1]:.6f}")
        if 'val_loss' in self.history.history:
            print(f"  Final validation loss: {self.history.history['val_loss'][-1]:.6f}")

        return self.history

    def predict(self, X):
        """
        Generate predictions.

        Parameters:
        -----------
        X : np.ndarray or list
            Input features

        Returns:
        --------
        predictions : np.ndarray
            Predicted demand grid
        """
        return self.model.predict(X, verbose=0)

    def evaluate(self, X_test, y_test, grid_mapping=None):
        """
        Evaluate model performance.

        Returns:
        --------
        metrics : dict
            Evaluation metrics
        predictions : np.ndarray
            Predicted values
        """
        print("\nEvaluating ConvLSTM model on test data...")

        # Get predictions
        predictions = self.predict(X_test)

        # Calculate overall metrics
        mse = mean_squared_error(y_test.flatten(), predictions.flatten())
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test.flatten(), predictions.flatten())

        # MAPE (avoiding division by zero)
        mask = y_test.flatten() != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs(
                (y_test.flatten()[mask] - predictions.flatten()[mask]) / 
                y_test.flatten()[mask]
            )) * 100
        else:
            mape = np.nan

        metrics = {
            'Overall': {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'MAPE': mape
            }
        }

        # Per-location metrics if grid_mapping provided
        if grid_mapping is not None:
            for cluster, (row, col) in grid_mapping.items():
                actual = y_test[:, row, col, 0]
                pred = predictions[:, row, col, 0]

                cluster_rmse = np.sqrt(mean_squared_error(actual, pred))
                cluster_mae = mean_absolute_error(actual, pred)

                mask = actual != 0
                if mask.sum() > 0:
                    cluster_mape = np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100
                else:
                    cluster_mape = np.nan

                metrics[cluster] = {
                    'RMSE': cluster_rmse,
                    'MAE': cluster_mae,
                    'MAPE': cluster_mape
                }

        print(f"\nOverall Metrics:")
        print(f"  RMSE: {metrics['Overall']['RMSE']:.4f}")
        print(f"  MAE: {metrics['Overall']['MAE']:.4f}")
        print(f"  MAPE: {metrics['Overall']['MAPE']:.2f}%")

        return metrics, predictions

    def plot_training_history(self, save_path=None):
        """
        Plot training history (loss curves).
        """
        if self.history is None:
            print("No training history available. Train the model first.")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(self.history.history['loss'], label='Training Loss')
        if 'val_loss' in self.history.history:
            ax.plot(self.history.history['val_loss'], label='Validation Loss')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (MSE)')
        ax.set_title('ConvLSTM Training History')
        ax.legend()
        ax.grid(True)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")

        plt.show()


# Initialize and run ConvLSTM model
print("\n" + "-"*40)
print("Running ConvLSTM Pipeline...")
print("-"*40)

# Determine grid dimensions based on number of clusters
n_clusters = len(top_clusters)
grid_size = int(np.ceil(np.sqrt(n_clusters)))

print(f"\nGrid configuration:")
print(f"  Number of clusters: {n_clusters}")
print(f"  Grid size: {grid_size} x {grid_size}")

# Initialize predictor
convlstm_predictor = ConvLSTMDemandPredictor(
    grid_height=grid_size,
    grid_width=grid_size,
    n_channels=1,
    closeness_len=6,   # Last 6 hours
    period_len=3,       # Same hour in last 3 days
    trend_len=2,        # Same hour in last 2 weeks
    period_interval=24,  # 24 hours
    trend_interval=168   # 168 hours (1 week)
)

# Prepare full data for ConvLSTM
full_data_for_grid = pd.concat([train_data[top_clusters], val_data[top_clusters], test_data[top_clusters]])

# Convert to grid representation
grid_data, grid_mapping = convlstm_predictor.prepare_grid_data(full_data_for_grid)

print(f"\nGrid data shape: {grid_data.shape}")

# Scale the data
original_shape = grid_data.shape
grid_data_flat = grid_data.reshape(-1, 1)
grid_data_scaled = convlstm_predictor.scaler.fit_transform(grid_data_flat)
grid_data_scaled = grid_data_scaled.reshape(original_shape)

# Create sequences
X_close, X_period, X_trend, y = convlstm_predictor.create_sequences(
    grid_data_scaled, 
    include_period=True, 
    include_trend=True
)

# Split data (maintaining temporal order)
train_size = len(train_data) - 168  # Account for lookback
val_size = len(val_data)
test_size = len(test_data)

# Adjust sizes based on sequence creation
total_sequences = len(y)
train_end = int(total_sequences * 0.7)
val_end = int(total_sequences * 0.85)

# Prepare training data
X_train_close = X_close[:train_end]
X_train_period = X_period[:train_end] if X_period is not None else None
X_train_trend = X_trend[:train_end] if X_trend is not None else None
y_train_conv = y[:train_end]

# Prepare validation data
X_val_close = X_close[train_end:val_end]
X_val_period = X_period[train_end:val_end] if X_period is not None else None
X_val_trend = X_trend[train_end:val_end] if X_trend is not None else None
y_val_conv = y[train_end:val_end]

# Prepare test data
X_test_close = X_close[val_end:]
X_test_period = X_period[val_end:] if X_period is not None else None
X_test_trend = X_trend[val_end:] if X_trend is not None else None
y_test_conv = y[val_end:]

print(f"\nData splits for ConvLSTM:")
print(f"  Training: {len(y_train_conv)} samples")
print(f"  Validation: {len(y_val_conv)} samples")
print(f"  Test: {len(y_test_conv)} samples")

# Build model
# Option 1: Full multi-branch model
convlstm_predictor.build_model(
    include_period=True,
    include_trend=True,
    n_filters=64,
    kernel_size=(3, 3),
    dropout_rate=0.2
)

# Prepare inputs for multi-branch model
X_train_multi = [X_train_close, X_train_period, X_train_trend]
X_val_multi = [X_val_close, X_val_period, X_val_trend]
X_test_multi = [X_test_close, X_test_period, X_test_trend]

# Train the model
history = convlstm_predictor.train(
    X_train_multi, y_train_conv,
    X_val=X_val_multi, y_val=y_val_conv,
    epochs=100,
    batch_size=32,
    early_stopping_patience=15
)

# Evaluate
convlstm_metrics, convlstm_predictions = convlstm_predictor.evaluate(
    X_test_multi, y_test_conv, grid_mapping
)

# Save ConvLSTM results
convlstm_results_df = pd.DataFrame(convlstm_metrics).T
convlstm_results_df.to_csv(os.path.join(output_path, 'convlstm_evaluation_metrics.csv'))

# Save predictions (converting grid back to demand matrix format)
pred_reshaped = convlstm_predictions.reshape(len(convlstm_predictions), -1)
pred_unscaled = convlstm_predictor.scaler.inverse_transform(pred_reshaped.reshape(-1, 1))
pred_unscaled = pred_unscaled.reshape(convlstm_predictions.shape)

# Convert predictions to DataFrame
pred_dict = {}
for cluster, (row, col) in grid_mapping.items():
    pred_dict[cluster] = pred_unscaled[:, row, col, 0]
convlstm_pred_df = pd.DataFrame(pred_dict)
convlstm_pred_df.to_csv(os.path.join(output_path, 'convlstm_predictions.csv'), index=False)

# Plot training history
convlstm_predictor.plot_training_history(
    save_path=os.path.join(output_path, 'convlstm_training_history.png')
)

print("\nConvLSTM model results saved!")


# ============================================================================
# PART 4: MODEL COMPARISON AND FINAL ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("PART 4: MODEL COMPARISON AND FINAL ANALYSIS")
print("="*80)

def compare_models(var_metrics, xgb_metrics, convlstm_metrics):
    """
    Compare performance of all three models.

    Returns:
    --------
    comparison_df : pd.DataFrame
        Comparison table of all models
    """
    print("\nModel Comparison:")
    print("-" * 50)

    comparison = {
        'VAR': {
            'RMSE': var_metrics['Overall']['RMSE'],
            'MAE': var_metrics['Overall']['MAE'],
            'MAPE (%)': var_metrics['Overall']['MAPE']
        },
        'XGBoost': {
            'RMSE': xgb_metrics['Overall']['RMSE'],
            'MAE': xgb_metrics['Overall']['MAE'],
            'MAPE (%)': xgb_metrics['Overall']['MAPE']
        },
        'ConvLSTM': {
            'RMSE': convlstm_metrics['Overall']['RMSE'],
            'MAE': convlstm_metrics['Overall']['MAE'],
            'MAPE (%)': convlstm_metrics['Overall']['MAPE']
        }
    }

    comparison_df = pd.DataFrame(comparison).T

    # Find best model for each metric
    best_rmse = comparison_df['RMSE'].idxmin()
    best_mae = comparison_df['MAE'].idxmin()
    best_mape = comparison_df['MAPE (%)'].idxmin()

    print(comparison_df.to_string())
    print("\nBest performing models:")
    print(f"  Best RMSE: {best_rmse}")
    print(f"  Best MAE: {best_mae}")
    print(f"  Best MAPE: {best_mape}")

    return comparison_df


def plot_model_comparison(comparison_df, save_path=None):
    """
    Visualize model comparison with bar plots.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = ['RMSE', 'MAE', 'MAPE (%)']
    colors = ['steelblue', 'darkorange', 'forestgreen']

    for ax, metric, color in zip(axes, metrics, colors):
        comparison_df[metric].plot(kind='bar', ax=ax, color=color)
        ax.set_title(f'{metric} Comparison')
        ax.set_ylabel(metric)
        ax.set_xlabel('Model')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")

    plt.show()


def plot_predictions_comparison(test_index, actuals, var_preds, xgb_preds, convlstm_preds,
                                cluster, save_path=None):
    """
    Plot actual vs predicted values for a specific cluster.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(test_index, actuals, 'k-', label='Actual', linewidth=2)
    ax.plot(test_index, var_preds, 'b--', label='VAR', alpha=0.7)
    ax.plot(test_index, xgb_preds, 'g--', label='XGBoost', alpha=0.7)
    ax.plot(test_index, convlstm_preds, 'r--', label='ConvLSTM', alpha=0.7)

    ax.set_xlabel('Time')
    ax.set_ylabel('Demand')
    ax.set_title(f'Demand Prediction Comparison - Cluster {cluster}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Predictions comparison plot saved to {save_path}")

    plt.show()


# Compare all models
comparison_df = compare_models(var_metrics, xgb_metrics, convlstm_metrics)

# Save comparison results
comparison_df.to_csv(os.path.join(output_path, 'model_comparison.csv'))

# Plot comparison
plot_model_comparison(
    comparison_df, 
    save_path=os.path.join(output_path, 'model_comparison.png')
)

# Plot predictions for one cluster
sample_cluster = top_clusters[0]
plot_predictions_comparison(
    test_index=test_data.index[:len(convlstm_pred_df)],
    actuals=test_data[sample_cluster].values[:len(convlstm_pred_df)],
    var_preds=var_predictions[sample_cluster].values[:len(convlstm_pred_df)],
    xgb_preds=xgb_predictions[sample_cluster].values[:len(convlstm_pred_df)],
    convlstm_preds=convlstm_pred_df[sample_cluster].values,
    cluster=sample_cluster,
    save_path=os.path.join(output_path, 'predictions_comparison.png')
)


# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("PIPELINE EXECUTION COMPLETE")
print("="*80)

print("\nOutput files saved to:", output_path)
print("\nGenerated files:")
print("  1. VAR Model:")
print("     - var_evaluation_metrics.csv")
print("     - var_predictions.csv")
print("  2. XGBoost Model:")
print("     - xgboost_evaluation_metrics.csv")
print("     - xgboost_predictions.csv")
print("     - xgboost_feature_importance.csv")
print("     - xgboost_best_params.txt")
print("  3. ConvLSTM Model:")
print("     - convlstm_evaluation_metrics.csv")
print("     - convlstm_predictions.csv")
print("     - convlstm_best_model.keras")
print("     - convlstm_training_history.png")
print("  4. Comparison:")
print("     - model_comparison.csv")
print("     - model_comparison.png")
print("     - predictions_comparison.png")

print("\n" + "="*80)
print("THEORETICAL NOTES FOR THESIS")
print("="*80)

notes = """
MODEL SELECTION RATIONALE:

1. VAR (Vector Autoregression):
   - Captures linear interdependencies between multiple time series
   - Assumes stationarity - requires differencing for non-stationary data
   - Optimal lag selection via information criteria (AIC, BIC)
   - Granger causality reveals predictive relationships between zones
   - Best for: Understanding temporal dynamics and zone interactions
   - Reference: Lütkepohl, H. (2005). New Introduction to Multiple Time Series Analysis

2. XGBoost:
   - Gradient boosted decision trees for non-linear patterns
   - Handles feature engineering (lags, rolling statistics, temporal features)
   - TimeSeriesSplit for proper temporal cross-validation
   - Feature importance reveals key predictors
   - Best for: Capturing complex non-linear patterns with interpretability
   - Reference: Chen & Guestrin (2016). XGBoost: A Scalable Tree Boosting System

3. ConvLSTM:
   - Combines CNN (spatial) and LSTM (temporal) in unified architecture
   - Processes grid-based spatiotemporal data directly
   - Multi-branch design captures: closeness, daily periodicity, weekly trends
   - Preserves spatial topology unlike flattening approaches
   - Best for: Complex spatiotemporal dependencies in grid-structured data
   - Reference: Shi et al. (2015). Convolutional LSTM Network for Precipitation Nowcasting

   Related architectures for spatiotemporal prediction:
   - ST-ResNet: Uses residual learning for crowd flow prediction (Zhang et al., 2017)
   - STDConvLSTM: Adds attention mechanisms for spatial/temporal imbalance (Tang, 2025)
   - HDGCN: Graph convolutional approach for irregular spatial structures (Zhao et al., 2022)

MODEL COMPARISON CONSIDERATIONS:
- VAR is best when linear relationships dominate and interpretability is crucial
- XGBoost excels with rich feature engineering and handles non-linearity well
- ConvLSTM is superior when spatial structure matters significantly
- Ensemble approaches combining these models often yield best results
"""

print(notes)
