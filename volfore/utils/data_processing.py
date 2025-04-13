"""
Data processing utilities for volatility forecasting.

This module provides functions for loading financial data,
calculating returns, and computing realized volatility.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import pandas_datareader as pdr
from datetime import datetime, timedelta


def fetch_stock_data(ticker, start_date, end_date=None, source='yahoo'):
    """
    Fetch stock price data from various sources.
    
    Parameters
    ----------
    ticker : str
        Ticker symbol
    start_date : str or datetime
        Start date in 'YYYY-MM-DD' format
    end_date : str or datetime, optional
        End date in 'YYYY-MM-DD' format. If None, uses current date.
    source : str, default='yahoo'
        Data source ('yahoo', 'fred', 'stooq', etc.)
        
    Returns
    -------
    DataFrame
        Historical stock data
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
        
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
    
    if source.lower() == 'yahoo':
        # Use yfinance for Yahoo Finance data
        data = yf.download(ticker, start=start_date, end=end_date)
    else:
        # Use pandas_datareader for other sources
        data = pdr.DataReader(ticker, source, start_date, end_date)
    
    return data


def fetch_market_data(tickers, start_date, end_date=None, source='yahoo'):
    """
    Fetch market data for multiple tickers.
    
    Parameters
    ----------
    tickers : list
        List of ticker symbols
    start_date : str or datetime
        Start date in 'YYYY-MM-DD' format
    end_date : str or datetime, optional
        End date in 'YYYY-MM-DD' format. If None, uses current date.
    source : str, default='yahoo'
        Data source ('yahoo', 'fred', 'stooq', etc.)
        
    Returns
    -------
    DataFrame
        Historical market data with MultiIndex columns
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
        
    if source.lower() == 'yahoo':
        # Use yfinance for Yahoo Finance data
        data = yf.download(tickers, start=start_date, end=end_date)
    else:
        # Use pandas_datareader for other sources
        data_dict = {}
        for ticker in tickers:
            data_dict[ticker] = pdr.DataReader(ticker, source, start_date, end_date)
        
        # Combine data into MultiIndex DataFrame
        data = pd.concat(data_dict, axis=1)
    
    return data


def fetch_vix_data(start_date, end_date=None):
    """
    Fetch VIX data from FRED.
    
    Parameters
    ----------
    start_date : str or datetime
        Start date in 'YYYY-MM-DD' format
    end_date : str or datetime, optional
        End date in 'YYYY-MM-DD' format. If None, uses current date.
        
    Returns
    -------
    Series
        VIX time series
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
        
    # Fetch VIX data from FRED (ticker: VIXCLS)
    vix = pdr.DataReader('VIXCLS', 'fred', start_date, end_date)
    
    return vix['VIXCLS']


def fetch_macro_data(indicators, start_date, end_date=None):
    """
    Fetch macroeconomic data from FRED.
    
    Parameters
    ----------
    indicators : list
        List of FRED indicator codes
    start_date : str or datetime
        Start date in 'YYYY-MM-DD' format
    end_date : str or datetime, optional
        End date in 'YYYY-MM-DD' format. If None, uses current date.
        
    Returns
    -------
    DataFrame
        Macroeconomic time series
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
        
    # Fetch indicators from FRED
    macro_data = pdr.DataReader(indicators, 'fred', start_date, end_date)
    
    return macro_data


def calculate_returns(prices, method='log', fill_na=True):
    """
    Calculate returns from price series.
    
    Parameters
    ----------
    prices : Series or DataFrame
        Price series or DataFrame
    method : str, default='log'
        Return calculation method: 'log' for log returns or 'simple' for simple returns
    fill_na : bool, default=True
        Whether to fill NaN values (first row) with zeros
        
    Returns
    -------
    Series or DataFrame
        Return series or DataFrame
    """
    if method.lower() == 'log':
        # Calculate log returns
        returns = np.log(prices / prices.shift(1))
    else:
        # Calculate simple returns
        returns = prices / prices.shift(1) - 1
    
    if fill_na:
        returns = returns.fillna(0)
        
    return returns


def calculate_realized_volatility(returns, window=5, annualize=True, trading_days=252):
    """
    Calculate realized volatility from return series.
    
    Parameters
    ----------
    returns : Series or DataFrame
        Return series or DataFrame
    window : int, default=5
        Rolling window size
    annualize : bool, default=True
        Whether to annualize volatility
    trading_days : int, default=252
        Number of trading days in a year
        
    Returns
    -------
    Series or DataFrame
        Realized volatility series or DataFrame
    """
    # Calculate realized volatility as rolling standard deviation
    realized_vol = returns.rolling(window=window).std()
    
    if annualize:
        # Annualize volatility
        realized_vol = realized_vol * np.sqrt(trading_days)
        
    return realized_vol


def calculate_range_volatility(high, low, close_prev, method='parkinson', annualize=True, trading_days=252):
    """
    Calculate range-based volatility estimators.
    
    Parameters
    ----------
    high : Series or DataFrame
        High price series
    low : Series or DataFrame
        Low price series
    close_prev : Series or DataFrame
        Previous close price series
    method : str, default='parkinson'
        Volatility estimator method: 'parkinson', 'garman_klass', or 'rogers_satchell'
    annualize : bool, default=True
        Whether to annualize volatility
    trading_days : int, default=252
        Number of trading days in a year
        
    Returns
    -------
    Series or DataFrame
        Range-based volatility series or DataFrame
    """
    if method.lower() == 'parkinson':
        # Parkinson estimator
        log_hl = np.log(high / low)
        vol = np.sqrt(log_hl**2 / (4 * np.log(2)))
    elif method.lower() == 'garman_klass':
        # Garman-Klass estimator
        log_hl = np.log(high / low)
        log_co = np.log(close_prev)
        vol = np.sqrt(0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2)
    elif method.lower() == 'rogers_satchell':
        # Rogers-Satchell estimator
        log_ho = np.log(high / close_prev)
        log_lo = np.log(low / close_prev)
        vol = np.sqrt(log_ho * (log_ho - log_lo) + log_lo * (log_lo - log_ho))
    else:
        raise ValueError(f"Unknown volatility estimator method: {method}")
    
    if annualize:
        # Annualize volatility
        vol = vol * np.sqrt(trading_days)
        
    return vol


def prepare_volatility_features(returns, realized_vol, window_sizes=None, lags=None):
    """
    Prepare features for volatility forecasting models.
    
    Parameters
    ----------
    returns : Series
        Return series
    realized_vol : Series
        Realized volatility series
    window_sizes : list, optional
        List of rolling window sizes. If None, uses [5, 10, 22, 63]
    lags : list, optional
        List of lag sizes. If None, uses [1, 2, 3, 5, 10, 22]
        
    Returns
    -------
    DataFrame
        Feature matrix
    """
    if window_sizes is None:
        window_sizes = [5, 10, 22, 63]  # 1w, 2w, 1m, 3m
    if lags is None:
        lags = [1, 2, 3, 5, 10, 22]  # Various lags
    
    features = pd.DataFrame(index=returns.index)
    
    # Add basic features
    features['return'] = returns
    features['abs_return'] = np.abs(returns)
    features['return_squared'] = returns**2
    
    # Add lagged returns
    for lag in lags:
        features[f'return_lag_{lag}'] = returns.shift(lag)
        features[f'abs_return_lag_{lag}'] = np.abs(returns).shift(lag)
        features[f'return_squared_lag_{lag}'] = returns.shift(lag)**2
    
    # Add rolling window features
    for window in window_sizes:
        # Return-based features
        features[f'return_mean_{window}d'] = returns.rolling(window=window).mean()
        features[f'return_std_{window}d'] = returns.rolling(window=window).std()
        features[f'abs_return_mean_{window}d'] = np.abs(returns).rolling(window=window).mean()
        
        # Realized vol features
        features[f'realized_vol_{window}d'] = realized_vol
        
        for lag in [1, 5, 10]:
            if lag < window:
                features[f'realized_vol_{window}d_lag_{lag}'] = realized_vol.shift(lag)
    
    # Add realized volatility lags
    for lag in lags:
        features[f'realized_vol_lag_{lag}'] = realized_vol.shift(lag)
    
    # Add volatility of volatility (vol-of-vol)
    for window in window_sizes:
        features[f'vol_of_vol_{window}d'] = realized_vol.rolling(window=window).std()
    
    # EWMA volatility (RiskMetrics)
    for lambda_val in [0.94, 0.97]:
        ewma_vol = returns.ewm(alpha=1-lambda_val).std()
        features[f'ewma_vol_{lambda_val}'] = ewma_vol
    
    # Drop NaN values
    features = features.dropna()
    
    return features


def time_series_split(data, n_splits=5, test_size=0.2):
    """
    Create time series cross-validation splits.
    
    Parameters
    ----------
    data : DataFrame
        Data to split
    n_splits : int, default=5
        Number of splits
    test_size : float, default=0.2
        Fraction of data to include in test set
        
    Returns
    -------
    list
        List of (train_idx, test_idx) tuples
    """
    n = len(data)
    test_n = int(n * test_size)
    
    # Start with initial training data
    train_end = n - test_n * n_splits
    
    splits = []
    for i in range(n_splits):
        train_idx = np.arange(0, train_end + i * test_n)
        test_idx = np.arange(train_end + i * test_n, min(train_end + (i + 1) * test_n, n))
        splits.append((train_idx, test_idx))
    
    return splits 