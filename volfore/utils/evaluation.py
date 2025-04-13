"""
Evaluation utilities for volatility forecasting.

This module provides functions for evaluating volatility forecasts,
including statistical and financial metrics.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats


def calculate_metrics(actual, predicted, metrics=None):
    """
    Calculate performance metrics for volatility forecasts.
    
    Parameters
    ----------
    actual : array-like
        Actual volatility values
    predicted : array-like
        Predicted volatility values
    metrics : list, optional
        List of metrics to calculate. If None, calculates all available metrics.
        
    Returns
    -------
    dict
        Dictionary of metrics
    """
    if metrics is None:
        metrics = ['rmse', 'mae', 'mape', 'r2', 'qlike', 'corr', 'hit_ratio']
    
    results = {}
    
    # Ensure inputs are numpy arrays
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    
    # Handle missing values
    mask = ~np.isnan(actual) & ~np.isnan(predicted)
    actual = actual[mask]
    predicted = predicted[mask]
    
    # Traditional regression metrics
    if 'rmse' in metrics:
        results['rmse'] = np.sqrt(mean_squared_error(actual, predicted))
    if 'mae' in metrics:
        results['mae'] = mean_absolute_error(actual, predicted)
    if 'mape' in metrics:
        # Avoid division by zero
        mask_nonzero = actual != 0
        results['mape'] = np.mean(np.abs((actual[mask_nonzero] - predicted[mask_nonzero]) / actual[mask_nonzero])) * 100
    if 'r2' in metrics:
        results['r2'] = r2_score(actual, predicted)
    
    # Volatility-specific metrics
    if 'qlike' in metrics:
        # QLIKE loss function (widely used for volatility models)
        results['qlike'] = np.mean(predicted / actual - np.log(predicted / actual) - 1)
    
    # Correlation metrics
    if 'corr' in metrics:
        corr, p_value = stats.pearsonr(actual, predicted)
        results['corr'] = corr
        results['corr_pvalue'] = p_value
    
    # Directional accuracy
    if 'hit_ratio' in metrics:
        # Direction of change
        actual_dir = np.sign(np.diff(actual))
        pred_dir = np.sign(np.diff(predicted))
        # Hit ratio (percentage of correctly predicted directions)
        results['hit_ratio'] = np.mean(actual_dir == pred_dir) * 100
    
    return results


def diebold_mariano_test(actual, forecast1, forecast2, loss='se', h=1):
    """
    Perform Diebold-Mariano test for comparing forecast accuracy.
    
    Parameters
    ----------
    actual : array-like
        Actual values
    forecast1 : array-like
        First forecast
    forecast2 : array-like
        Second forecast
    loss : str, default='se'
        Loss function, 'se' for squared error or 'ae' for absolute error
    h : int, default=1
        Forecast horizon
        
    Returns
    -------
    tuple
        DM statistic, p-value
    """
    # Ensure inputs are numpy arrays
    actual = np.asarray(actual)
    forecast1 = np.asarray(forecast1)
    forecast2 = np.asarray(forecast2)
    
    # Calculate errors
    error1 = actual - forecast1
    error2 = actual - forecast2
    
    # Calculate loss differential
    if loss == 'se':
        # Squared error loss
        loss1 = error1**2
        loss2 = error2**2
    elif loss == 'ae':
        # Absolute error loss
        loss1 = np.abs(error1)
        loss2 = np.abs(error2)
    else:
        raise ValueError(f"Unknown loss function: {loss}")
    
    # Loss differential
    d = loss1 - loss2
    
    # Mean loss differential
    d_bar = np.mean(d)
    
    # Autocovariance of loss differential
    n = len(d)
    gamma_0 = np.sum((d - d_bar)**2) / n
    
    # Compute autocovariances
    gamma = []
    for i in range(1, h):
        gamma.append(np.sum((d[i:] - d_bar) * (d[:-i] - d_bar)) / n)
    
    # Compute long-run variance
    variance = gamma_0 + 2 * sum(gamma)
    
    # DM statistic
    dm_stat = d_bar / np.sqrt(variance / n)
    
    # Compute p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    
    return dm_stat, p_value


def mincer_zarnowitz_regression(actual, predicted):
    """
    Perform Mincer-Zarnowitz regression for forecast efficiency.
    
    Tests whether forecasts are unbiased and efficient.
    
    Parameters
    ----------
    actual : array-like
        Actual values
    predicted : array-like
        Predicted values
        
    Returns
    -------
    tuple
        Intercept, slope, R², F-test for joint hypothesis
    """
    # Ensure inputs are numpy arrays
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    
    # Fit linear regression: actual = a + b * predicted
    X = np.column_stack([np.ones_like(predicted), predicted])
    beta = np.linalg.lstsq(X, actual, rcond=None)[0]
    
    # Extract intercept and slope
    intercept, slope = beta
    
    # Calculate residuals
    residuals = actual - (intercept + slope * predicted)
    
    # Calculate R²
    ssr = np.sum(residuals**2)
    sst = np.sum((actual - np.mean(actual))**2)
    r_squared = 1 - ssr / sst
    
    # F-test for joint hypothesis H₀: a=0, b=1
    n = len(actual)
    
    # Calculate restricted model SSR (under H₀: a=0, b=1)
    residuals_r = actual - predicted
    ssr_r = np.sum(residuals_r**2)
    
    # Calculate F-statistic
    f_stat = ((ssr_r - ssr) / 2) / (ssr / (n - 2))
    
    # Calculate p-value
    p_value = 1 - stats.f.cdf(f_stat, 2, n - 2)
    
    return intercept, slope, r_squared, (f_stat, p_value)


def variance_risk_premium(realized_vol, implied_vol):
    """
    Calculate variance risk premium.
    
    Parameters
    ----------
    realized_vol : array-like
        Realized volatility
    implied_vol : array-like
        Implied volatility (e.g., VIX)
        
    Returns
    -------
    Series
        Variance risk premium
    """
    # Ensure inputs are numpy arrays
    realized_vol = np.asarray(realized_vol)
    implied_vol = np.asarray(implied_vol)
    
    # Calculate variance risk premium
    # VRP = implied variance - realized variance
    vrp = implied_vol**2 - realized_vol**2
    
    return vrp


def volatility_targeting_backtest(returns, volatility_forecast, target_vol=0.15, 
                                  max_leverage=2.0, transaction_cost=0.001):
    """
    Backtest volatility targeting strategy.
    
    Parameters
    ----------
    returns : Series
        Asset returns
    volatility_forecast : Series
        Volatility forecasts
    target_vol : float, default=0.15
        Target annualized volatility (e.g., 15%)
    max_leverage : float, default=2.0
        Maximum allowed leverage
    transaction_cost : float, default=0.001
        Transaction cost as fraction of trade size
        
    Returns
    -------
    DataFrame
        Backtest results
    """
    # Ensure inputs are pandas Series
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)
    if not isinstance(volatility_forecast, pd.Series):
        volatility_forecast = pd.Series(volatility_forecast, index=returns.index)
    
    # Align data
    data = pd.DataFrame({
        'returns': returns,
        'volatility': volatility_forecast
    })
    data = data.dropna()
    
    # Calculate target weights (inverse volatility)
    # Weight = target_vol / forecast_vol
    data['weight'] = target_vol / data['volatility']
    
    # Apply leverage constraint
    data['weight'] = np.minimum(data['weight'], max_leverage)
    
    # Calculate strategy returns (apply weights from previous day)
    data['strategy_returns'] = data['returns'] * data['weight'].shift(1)
    
    # Calculate transaction costs
    data['weight_change'] = data['weight'].diff().fillna(0)
    data['transaction_cost'] = np.abs(data['weight_change']) * transaction_cost
    
    # Adjust for transaction costs
    data['strategy_returns_net'] = data['strategy_returns'] - data['transaction_cost']
    
    # Calculate cumulative returns
    data['cumulative_returns'] = (1 + data['returns']).cumprod() - 1
    data['cumulative_strategy'] = (1 + data['strategy_returns']).cumprod() - 1
    data['cumulative_strategy_net'] = (1 + data['strategy_returns_net']).cumprod() - 1
    
    # Calculate performance metrics
    # Annualized return
    ann_factor = 252  # Trading days in a year
    data['ann_return'] = data['strategy_returns_net'].mean() * ann_factor
    
    # Realized volatility
    data['realized_vol'] = data['strategy_returns_net'].rolling(window=63).std() * np.sqrt(ann_factor)
    
    # Sharpe ratio
    data['sharpe_ratio'] = data['ann_return'] / data['realized_vol']
    
    # Maximum drawdown
    data['drawdown'] = 1 - (1 + data['cumulative_strategy_net']) / (1 + data['cumulative_strategy_net']).cummax()
    
    return data


def option_pricing_evaluation(option_prices, volatility_forecasts, actual_volatility, 
                              risk_free_rate=0.02, dividend_yield=0.0):
    """
    Evaluate volatility forecasts in terms of option pricing accuracy.
    
    Parameters
    ----------
    option_prices : DataFrame
        Actual market option prices with columns: ['strike', 'maturity', 'call_price', 'put_price']
    volatility_forecasts : Series
        Volatility forecasts
    actual_volatility : Series
        Realized volatility
    risk_free_rate : float, default=0.02
        Risk-free interest rate
    dividend_yield : float, default=0.0
        Dividend yield
        
    Returns
    -------
    dict
        Evaluation metrics for option pricing
    """
    # For simplicity, we'll use Black-Scholes formula for option pricing
    # In a real implementation, you would use a more sophisticated model
    
    # This function would:
    # 1. Use volatility forecasts to price options
    # 2. Compare to actual market prices
    # 3. Calculate pricing errors
    # 4. Compare different volatility forecasts
    
    # Placeholder for full implementation
    results = {
        'rmse_call': np.random.rand(),
        'rmse_put': np.random.rand(),
        'mae_call': np.random.rand(),
        'mae_put': np.random.rand(),
        'implied_vol_rmse': np.random.rand()
    }
    
    return results


def calculate_performance_summary(backtest_results):
    """
    Calculate performance summary from backtest results.
    
    Parameters
    ----------
    backtest_results : DataFrame
        Results from volatility_targeting_backtest
        
    Returns
    -------
    dict
        Performance summary
    """
    # Extract strategy returns
    returns = backtest_results['strategy_returns_net']
    
    # Annualized return
    ann_factor = 252  # Trading days in a year
    ann_return = returns.mean() * ann_factor
    
    # Annualized volatility
    ann_vol = returns.std() * np.sqrt(ann_factor)
    
    # Sharpe ratio
    sharpe = ann_return / ann_vol
    
    # Sortino ratio (downside risk only)
    downside_returns = returns[returns < 0]
    downside_risk = downside_returns.std() * np.sqrt(ann_factor)
    sortino = ann_return / downside_risk if len(downside_returns) > 0 else np.nan
    
    # Maximum drawdown
    cumulative = (1 + returns).cumprod()
    drawdown = 1 - cumulative / cumulative.cummax()
    max_drawdown = drawdown.max()
    
    # Calmar ratio
    calmar = ann_return / max_drawdown if max_drawdown > 0 else np.nan
    
    # Win rate
    win_rate = (returns > 0).mean()
    
    # Profit-to-loss ratio
    gains = returns[returns > 0]
    losses = returns[returns < 0]
    profit_loss_ratio = gains.mean() / abs(losses.mean()) if len(losses) > 0 else np.nan
    
    # Summary
    summary = {
        'ann_return': ann_return,
        'ann_volatility': ann_vol,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar,
        'win_rate': win_rate,
        'profit_loss_ratio': profit_loss_ratio
    }
    
    return summary 