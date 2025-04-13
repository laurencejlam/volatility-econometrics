"""
Classical time series volatility forecasting models.

This module implements traditional time series volatility models including:
- EWMA (Exponentially Weighted Moving Average)
- GARCH(1,1) and variants
- HAR (Heterogeneous Autoregressive) models
"""

import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.tsa.statespace.sarimax import SARIMAX


class EWMAModel:
    """
    Exponentially Weighted Moving Average volatility model.
    
    This is a simple and historically important volatility model that forms the 
    basis of RiskMetrics methodology. It can be seen as a restricted GARCH(1,1) model.
    """
    
    def __init__(self, lambda_param=0.94):
        """
        Initialize EWMA model with decay factor.
        
        Parameters
        ----------
        lambda_param : float, default=0.94
            Decay factor for EWMA model, typically around 0.94 for daily data
        """
        self.lambda_param = lambda_param
        self.is_fitted = False
        self.last_var = None
    
    def fit(self, returns, initial_variance=None):
        """
        Fit the EWMA model to the return series.
        
        Parameters
        ----------
        returns : array-like
            Series of asset returns
        initial_variance : float, optional
            Initial variance estimate. If None, use squared first return.
        
        Returns
        -------
        self : object
            Returns self
        """
        returns = np.asarray(returns)
        
        if initial_variance is None:
            # Use squared first return as initial variance
            initial_variance = returns[0]**2
        
        # Initialize variance series
        var_series = np.zeros_like(returns)
        var_series[0] = initial_variance
        
        # Compute EWMA variance recursively
        for t in range(1, len(returns)):
            var_series[t] = (self.lambda_param * var_series[t-1] + 
                            (1 - self.lambda_param) * returns[t-1]**2)
        
        self.variances = var_series
        self.last_var = var_series[-1]
        self.is_fitted = True
        
        return self
    
    def forecast(self, horizon=1):
        """
        Forecast volatility for specified horizon.
        
        For EWMA, the multi-step forecast is equal to the 
        last estimated variance.
        
        Parameters
        ----------
        horizon : int, default=1
            Forecast horizon
            
        Returns
        -------
        forecasts : ndarray
            Array of variance forecasts
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        # For EWMA, multi-period forecast is the same as one-period
        forecasts = np.full(horizon, self.last_var)
        
        return forecasts
    
    def simulate(self, n_days, rng=None):
        """
        Simulate returns from the EWMA model.
        
        Parameters
        ----------
        n_days : int
            Number of days to simulate
        rng : numpy.random.Generator, optional
            Random number generator
            
        Returns
        -------
        tuple
            Returns and conditional variances
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before simulation")
            
        if rng is None:
            rng = np.random.default_rng()
            
        returns = np.zeros(n_days)
        variances = np.zeros(n_days)
        
        # Start with last variance from fitted model
        variances[0] = self.last_var
        
        # Generate first return
        returns[0] = np.sqrt(variances[0]) * rng.standard_normal()
        
        # Generate subsequent returns and variances
        for t in range(1, n_days):
            variances[t] = (self.lambda_param * variances[t-1] + 
                           (1 - self.lambda_param) * returns[t-1]**2)
            returns[t] = np.sqrt(variances[t]) * rng.standard_normal()
            
        return returns, variances


class GARCHModel:
    """
    GARCH model wrapper using arch_model from the arch package.
    
    Provides a consistent interface for GARCH-type models.
    """
    
    def __init__(self, p=1, q=1, mean='Zero', vol='GARCH', dist='normal'):
        """
        Initialize GARCH model.
        
        Parameters
        ----------
        p : int, default=1
            Lag order of the symmetric innovation
        q : int, default=1
            Lag order of lagged volatility
        mean : str, default='Zero'
            Name of mean model
        vol : str, default='GARCH'
            Name of volatility model
        dist : str, default='normal'
            Name of distribution for innovations
        """
        self.p = p
        self.q = q
        self.mean = mean
        self.vol = vol
        self.dist = dist
        self.model = None
        self.result = None
        
    def fit(self, returns, update_freq=1):
        """
        Fit the GARCH model to return series.
        
        Parameters
        ----------
        returns : array-like
            Series of asset returns
        update_freq : int, default=1
            Frequency for iteration updates
            
        Returns
        -------
        self : object
            Returns self
        """
        self.model = arch_model(
            returns, 
            p=self.p, 
            q=self.q, 
            mean=self.mean, 
            vol=self.vol, 
            dist=self.dist
        )
        self.result = self.model.fit(disp='off', update_freq=update_freq)
        
        return self
    
    def forecast(self, horizon=1, reindex=True):
        """
        Forecast volatility for specified horizon.
        
        Parameters
        ----------
        horizon : int, default=1
            Forecast horizon
        reindex : bool, default=True
            Whether to reindex the forecast to start after the sample
            
        Returns
        -------
        forecasts : DataFrame
            Variance forecasts
        """
        if self.result is None:
            raise ValueError("Model must be fitted before forecasting")
            
        forecast = self.result.forecast(horizon=horizon, reindex=reindex)
        
        return forecast.variance
    
    def summary(self):
        """
        Return summary of fitted model.
        
        Returns
        -------
        summary : Summary
            Summary of model results
        """
        if self.result is None:
            raise ValueError("Model must be fitted before summary")
            
        return self.result.summary()
    
    def simulate(self, n_days, burn=500, initial_value=None):
        """
        Simulate from the fitted GARCH model.
        
        Parameters
        ----------
        n_days : int
            Number of days to simulate
        burn : int, default=500
            Number of burn-in periods
        initial_value : array or float, optional
            Initial values for the time series process
            
        Returns
        -------
        tuple
            Returns and conditional variances
        """
        if self.result is None:
            raise ValueError("Model must be fitted before simulation")
            
        simulation = self.result.simulate(
            n_days, 
            burn=burn, 
            initial_value=initial_value
        )
        
        return simulation.data, simulation.volatility**2


class HARModel:
    """
    Heterogeneous Autoregressive (HAR) model for volatility forecasting.
    
    HAR models capture the long-memory property of volatility through
    a cascade of different time horizons (e.g., daily, weekly, monthly).
    """
    
    def __init__(self, lags=(1, 5, 22)):
        """
        Initialize HAR model with specified lags.
        
        Parameters
        ----------
        lags : tuple, default=(1, 5, 22)
            Lags for different components (e.g., daily, weekly, monthly)
        """
        self.lags = lags
        self.model = None
        self.result = None
        
    def _prepare_features(self, realized_vol):
        """
        Prepare features for HAR model based on realized volatility.
        
        Parameters
        ----------
        realized_vol : array-like
            Series of realized volatility
            
        Returns
        -------
        X : DataFrame
            Features for HAR model
        y : Series
            Target variable (next-day realized volatility)
        """
        df = pd.DataFrame({'rv': realized_vol})
        
        # Create lag features
        for lag in self.lags:
            if lag == 1:
                df[f'rv_d'] = df['rv'].shift(1)
            else:
                # Average over the lag period
                df[f'rv_{lag}'] = df['rv'].rolling(window=lag).mean().shift(1)
        
        # Remove missing values
        df = df.dropna()
        
        # Prepare X and y
        X = df.drop(columns=['rv'])
        y = df['rv']
        
        return X, y
    
    def fit(self, realized_vol):
        """
        Fit the HAR model to realized volatility.
        
        Parameters
        ----------
        realized_vol : array-like
            Series of realized volatility
            
        Returns
        -------
        self : object
            Returns self
        """
        X, y = self._prepare_features(realized_vol)
        
        # Fit OLS model
        self.model = SARIMAX(y, exog=X, order=(0, 0, 0))
        self.result = self.model.fit(disp=False)
        
        return self
    
    def forecast(self, realized_vol, horizon=1):
        """
        Forecast volatility for specified horizon.
        
        Parameters
        ----------
        realized_vol : array-like
            Series of realized volatility
        horizon : int, default=1
            Forecast horizon
            
        Returns
        -------
        forecasts : ndarray
            Array of volatility forecasts
        """
        if self.result is None:
            raise ValueError("Model must be fitted before forecasting")
        
        # Prepare data including the latest observations
        X, _ = self._prepare_features(realized_vol)
        
        # Get the last row for initial forecast
        last_X = X.iloc[-1:].values
        
        # For multi-step forecasts, we need to recursively forecast
        forecasts = np.zeros(horizon)
        temp_x = last_X.copy().flatten()
        
        # Map from lag indices to X columns
        lag_indices = {lag: i for i, lag in enumerate(self.lags)}
        
        for h in range(horizon):
            # Forecast one step
            forecasts[h] = self.result.params[0] + np.dot(self.result.params[1:], temp_x)
            
            # Update temp_x for next forecast
            if h < horizon - 1:
                # Update daily lag
                temp_x[lag_indices[1]] = forecasts[h]
                
                # Update other lags as needed
                for lag in self.lags:
                    if lag > 1:
                        # Approximate update for longer lags
                        # This is a simplification and could be improved
                        temp_x[lag_indices[lag]] = ((lag - 1) * temp_x[lag_indices[lag]] + forecasts[h]) / lag
        
        return forecasts
    
    def summary(self):
        """
        Return summary of fitted model.
        
        Returns
        -------
        summary : Summary
            Summary of model results
        """
        if self.result is None:
            raise ValueError("Model must be fitted before summary")
            
        return self.result.summary() 