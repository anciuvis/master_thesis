# ============================================================================
# SARIMA MODEL - MEMORY OPTIMIZED VERSION
# ============================================================================

from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
import gc

class SARIMAPredictor:
    """SARIMA wrapper with auto_arima parameter optimization"""
    
    def __init__(self, cluster_name):
        self.cluster_name = cluster_name
        self.model = None
        self.results = None
        self.fitted = False
        self.order = None
        self.seasonal_order = None
    
    
    def grid_search_params(self, train_data, d_fixed=0, D_fixed=1, m=168):
        """
        Grid search SARIMA parameters with d=1, D=1 fixed (from ADF analysis).
        Searches p∈[0,2], q∈[0,2], P∈[0,1], Q∈[0,1] = 36 combinations.
        """        
        # Data preparation
        if train_data.index.freq is None:
            train_data = train_data.asfreq('h')
        train_data = train_data.ffill().bfill().dropna()
        
        # Limit to 30 days
        if len(train_data) > 720:
            train_data = train_data.iloc[-720:]
        
        best_aic = np.inf
        best_params = None
        
        print(f"[{self.cluster_name}] Grid search: 36 combinations (d={d_fixed},D={D_fixed},m={m})")
        
        # Grid search: p,q [0,2], P,Q [0,1]
        for p in range(0, 3):
            for q in range(0, 3):
                for P in range(0, 2):
                    for Q in range(0, 2):
                        try:
                            model = SARIMAX(
                                train_data,
                                order=(p, d_fixed, q),
                                seasonal_order=(P, D_fixed, Q, m),
                                enforce_stationarity=False,
                                enforce_invertibility=False,
                                measurement_error=True
                            )
                            
                            results = model.fit(
                                disp=False,
                                maxiter=50,
                                low_memory=True,
                                method='lbfgs'
                            )
                            
                            aic = results.aic
                            if aic < best_aic:
                                best_aic = aic
                                best_params = {'p': p, 'q': q, 'P': P, 'Q': Q}

                            logger.info(f"Cluster: {self.cluster_name}, order=({p}, {d_fixed}, {q}), seasonal_order=({P}, {D_fixed}, {Q}, {m}), AIC: {aic}, best AIC: {best_aic}")
                        
                        except:
                            continue  # Skip failed combinations
        
        gc.collect()
        
        if best_params is None:
            raise ValueError(f"Grid search failed for {self.cluster_name}")
        
        # Update class attributes
        self.order = (best_params['p'], d_fixed, best_params['q'])
        self.seasonal_order = (best_params['P'], D_fixed, best_params['Q'], m)
        
        print(f"[{self.cluster_name}] Best: {self.order}×{self.seasonal_order}, AIC={best_aic:.2f}")
        
        return {
            'order': self.order,
            'seasonal_order': self.seasonal_order,
            'aic': best_aic
        }

    def fit_with_params(self, train_data, order, seasonal_order):
        """
        Fit SARIMA with SPECIFIED parameters (no search).
        Used for Phase 2: Apply best params from grid search to all clusters.
        """
        
        # Data preparation (same as grid_search_params)
        if train_data.index.freq is None:
            train_data = train_data.asfreq('h')
        train_data = train_data.ffill().bfill().dropna()
        
        # Limit to 30 days (memory safe)
        if len(train_data) > 720:
            train_data = train_data.iloc[-720:]
        
        print(f"[{self.cluster_name}] Fitting with {order}×{seasonal_order}...")
        
        # Fit with EXACT parameters provided (no search)
        self.model = SARIMAX(
            train_data,
            order=order,                    # (p,d,q) from grid search
            seasonal_order=seasonal_order,  # (P,D,Q,m) from grid search
            enforce_stationarity=False,
            enforce_invertibility=False,
            measurement_error=True
        )
        
        self.results = self.model.fit(
            disp=False,
            maxiter=200,
            low_memory=True,
            method='lbfgs'
        )
        
        # Update class attributes
        self.order = order
        self.seasonal_order = seasonal_order
        self.fitted = True
        
        print(f"[{self.cluster_name}] Fitted: AIC={self.results.aic:.2f}")
        gc.collect()
        
        return self.results
    
    def forecast(self, steps):
        """Generate forecasts"""
        if not self.fitted or self.results is None:
            raise ValueError("Model not fitted")
        
        forecast = self.results.get_forecast(steps=steps)
        return forecast.predicted_mean.values