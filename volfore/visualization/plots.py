"""
Plotting utilities for volatility forecasting.

This module provides functions for visualizing volatility forecasts,
model performance, and model diagnostics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from itertools import cycle


def plot_volatility_time_series(realized_vol, forecast_vols=None, title=None, 
                               figsize=(12, 6), style='seaborn-v0_8-whitegrid'):
    """
    Plot realized volatility and forecasts over time.
    
    Parameters
    ----------
    realized_vol : Series
        Realized volatility series
    forecast_vols : dict, optional
        Dictionary of forecast series {model_name: forecast_series}
    title : str, optional
        Plot title
    figsize : tuple, default=(12, 6)
        Figure size
    style : str, default='seaborn-v0_8-whitegrid'
        Matplotlib style
        
    Returns
    -------
    tuple
        Figure and axes objects
    """
    with plt.style.context(style):
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot realized volatility
        realized_vol.plot(ax=ax, label='Realized Volatility', linewidth=2, color='black')
        
        # Plot forecasts if provided
        if forecast_vols is not None:
            # Create color cycle
            colors = cycle(plt.cm.tab10.colors)
            
            for model_name, forecast in forecast_vols.items():
                color = next(colors)
                forecast.plot(ax=ax, label=f'{model_name} Forecast', linewidth=1.5, 
                             linestyle='--', color=color, alpha=0.8)
        
        # Set title and labels
        if title is not None:
            ax.set_title(title, fontsize=14)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Volatility', fontsize=12)
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.1%}'))
        
        # Add legend
        ax.legend(loc='upper left')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig, ax


def plot_volatility_forecast_comparison(realized_vol, forecast_vols, 
                                       start_date=None, end_date=None,
                                       figsize=(14, 8), style='seaborn-v0_8-whitegrid'):
    """
    Create a comprehensive comparison of volatility forecasts.
    
    Parameters
    ----------
    realized_vol : Series
        Realized volatility series
    forecast_vols : dict
        Dictionary of forecast series {model_name: forecast_series}
    start_date : str or datetime, optional
        Start date for plotting
    end_date : str or datetime, optional
        End date for plotting
    figsize : tuple, default=(14, 8)
        Figure size
    style : str, default='seaborn-v0_8-whitegrid'
        Matplotlib style
        
    Returns
    -------
    tuple
        Figure and axes objects
    """
    with plt.style.context(style):
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True, 
                                gridspec_kw={'height_ratios': [3, 1]})
        
        # Filter by date range if provided
        if start_date is not None or end_date is not None:
            if start_date is not None:
                realized_vol = realized_vol[realized_vol.index >= start_date]
                forecast_vols = {k: v[v.index >= start_date] for k, v in forecast_vols.items()}
            if end_date is not None:
                realized_vol = realized_vol[realized_vol.index <= end_date]
                forecast_vols = {k: v[v.index <= end_date] for k, v in forecast_vols.items()}
        
        # Plot realized volatility and forecasts
        realized_vol.plot(ax=axes[0], label='Realized Volatility', 
                         linewidth=2.5, color='black')
        
        # Create color cycle
        colors = cycle(plt.cm.tab10.colors)
        
        # Calculate errors for bottom plot
        errors = {}
        
        for model_name, forecast in forecast_vols.items():
            color = next(colors)
            forecast.plot(ax=axes[0], label=f'{model_name}', 
                         linewidth=1.5, linestyle='--', color=color)
            
            # Calculate forecast error
            aligned_rv = realized_vol.reindex(forecast.index).dropna()
            aligned_fc = forecast.reindex(aligned_rv.index).dropna()
            
            if not aligned_rv.empty and not aligned_fc.empty:
                error = aligned_fc - aligned_rv
                errors[model_name] = error
        
        # Plot errors
        for model_name, error in errors.items():
            error.plot(ax=axes[1], label=f'{model_name} Error', alpha=0.7)
        
        # Add horizontal line at zero for errors
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        # Set titles and labels
        axes[0].set_title('Volatility Forecast Comparison', fontsize=16)
        axes[0].set_ylabel('Volatility', fontsize=12)
        axes[0].legend(loc='upper left')
        
        axes[1].set_title('Forecast Errors', fontsize=14)
        axes[1].set_xlabel('Date', fontsize=12)
        axes[1].set_ylabel('Error', fontsize=12)
        axes[1].legend(loc='upper left')
        
        # Format y-axis as percentage
        for ax in axes:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.1%}'))
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig, axes


def plot_feature_importance(model_results, top_n=20, figsize=(10, 8), 
                           style='seaborn-v0_8-whitegrid'):
    """
    Plot feature importance from volatility forecasting models.
    
    Parameters
    ----------
    model_results : dict
        Dictionary with model feature importance DataFrames {model_name: feature_importance_df}
    top_n : int, default=20
        Number of top features to display
    figsize : tuple, default=(10, 8)
        Figure size
    style : str, default='seaborn-v0_8-whitegrid'
        Matplotlib style
        
    Returns
    -------
    tuple
        Figure and axes objects
    """
    with plt.style.context(style):
        n_models = len(model_results)
        fig, axes = plt.subplots(n_models, 1, figsize=figsize)
        
        # Handle single model case
        if n_models == 1:
            axes = [axes]
        
        for i, (model_name, importance_df) in enumerate(model_results.items()):
            # Sort by importance and take top_n
            sorted_df = importance_df.sort_values('Importance', ascending=False).head(top_n)
            
            # Create horizontal bar plot
            sns.barplot(x='Importance', y='Feature', data=sorted_df, ax=axes[i])
            
            # Set title and labels
            axes[i].set_title(f'{model_name} Feature Importance', fontsize=14)
            axes[i].set_xlabel('Importance', fontsize=12)
            axes[i].set_ylabel('Feature', fontsize=12)
            
            # Format y-tick labels
            for label in axes[i].get_yticklabels():
                label.set_fontsize(10)
        
        plt.tight_layout()
        
        return fig, axes


def plot_forecast_evaluation(metrics_dict, figsize=(14, 10), style='seaborn-v0_8-whitegrid'):
    """
    Plot comprehensive evaluation of volatility forecasts.
    
    Parameters
    ----------
    metrics_dict : dict
        Dictionary with model evaluation metrics {model_name: metrics_dict}
    figsize : tuple, default=(14, 10)
        Figure size
    style : str, default='seaborn-v0_8-whitegrid'
        Matplotlib style
        
    Returns
    -------
    tuple
        Figure and axes objects
    """
    with plt.style.context(style):
        # Extract metrics for comparison
        metrics_to_plot = ['rmse', 'mae', 'mape', 'r2', 'qlike', 'hit_ratio']
        
        # Create DataFrame for plotting
        plot_data = []
        for model_name, metrics in metrics_dict.items():
            for metric in metrics_to_plot:
                if metric in metrics:
                    plot_data.append({
                        'Model': model_name,
                        'Metric': metric,
                        'Value': metrics[metric]
                    })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        # Plot each metric
        for i, metric in enumerate(metrics_to_plot):
            metric_data = plot_df[plot_df['Metric'] == metric]
            
            if not metric_data.empty:
                # Sort by value (lower is better for most metrics, except r2 and hit_ratio)
                if metric in ['r2', 'hit_ratio']:
                    metric_data = metric_data.sort_values('Value', ascending=False)
                else:
                    metric_data = metric_data.sort_values('Value', ascending=True)
                
                # Create bar plot
                bar_plot = sns.barplot(x='Model', y='Value', data=metric_data, ax=axes[i])
                
                # Set title and labels
                metric_name = metric.upper() if metric == 'rmse' or metric == 'mae' else metric.capitalize()
                axes[i].set_title(f'{metric_name}', fontsize=14)
                axes[i].set_xlabel('')
                axes[i].set_ylabel('')
                
                # Rotate x-tick labels
                axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')
                
                # Add value labels on bars
                for p in bar_plot.patches:
                    bar_plot.annotate(f'{p.get_height():.4f}', 
                                     (p.get_x() + p.get_width() / 2., p.get_height()), 
                                     ha='center', va='bottom', fontsize=9, rotation=90)
        
        fig.suptitle('Volatility Forecast Evaluation Metrics', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        return fig, axes


def plot_rolling_metrics(actual, forecasts, window=252, metrics=None, 
                        figsize=(14, 10), style='seaborn-v0_8-whitegrid'):
    """
    Plot rolling forecast evaluation metrics.
    
    Parameters
    ----------
    actual : Series
        Actual realized volatility
    forecasts : dict
        Dictionary of forecast series {model_name: forecast_series}
    window : int, default=252
        Rolling window size (typically 1 year for financial data)
    metrics : list, optional
        List of metrics to calculate. If None, uses ['rmse', 'mae', 'corr']
    figsize : tuple, default=(14, 10)
        Figure size
    style : str, default='seaborn-v0_8-whitegrid'
        Matplotlib style
        
    Returns
    -------
    tuple
        Figure and axes objects
    """
    if metrics is None:
        metrics = ['rmse', 'mae', 'corr']
    
    with plt.style.context(style):
        # Create figure with subplots
        fig, axes = plt.subplots(len(metrics), 1, figsize=figsize, sharex=True)
        
        # Handle single metric case
        if len(metrics) == 1:
            axes = [axes]
        
        # Calculate rolling metrics for each model
        for model_name, forecast in forecasts.items():
            # Align series
            aligned_actual = actual.reindex(forecast.index).dropna()
            aligned_forecast = forecast.reindex(aligned_actual.index).dropna()
            
            # Create rolling window
            rolling_result = {}
            
            for i, metric in enumerate(metrics):
                values = []
                
                # Calculate metric for each rolling window
                for j in range(window, len(aligned_actual)):
                    window_actual = aligned_actual.iloc[j-window:j]
                    window_forecast = aligned_forecast.iloc[j-window:j]
                    
                    if metric == 'rmse':
                        value = np.sqrt(np.mean((window_actual - window_forecast) ** 2))
                    elif metric == 'mae':
                        value = np.mean(np.abs(window_actual - window_forecast))
                    elif metric == 'corr':
                        value = np.corrcoef(window_actual, window_forecast)[0, 1]
                    else:
                        continue
                    
                    values.append(value)
                
                # Create series for the rolling metric
                rolling_result[metric] = pd.Series(
                    values, 
                    index=aligned_actual.index[window:len(aligned_actual)]
                )
                
                # Plot on appropriate subplot
                rolling_result[metric].plot(
                    ax=axes[i], 
                    label=model_name,
                    linewidth=1.5
                )
                
                # Set title and labels
                metric_name = metric.upper() if metric == 'rmse' or metric == 'mae' else metric.capitalize()
                axes[i].set_title(f'Rolling {metric_name} (Window: {window} days)', fontsize=14)
                axes[i].set_ylabel(metric_name, fontsize=12)
                axes[i].legend(loc='upper left')
                axes[i].grid(True, alpha=0.3)
        
        # Set x-axis label on bottom subplot
        axes[-1].set_xlabel('Date', fontsize=12)
        
        plt.tight_layout()
        
        return fig, axes


def plot_volatility_regime(returns, realized_vol, window=22, n_regimes=3, 
                          figsize=(14, 10), style='seaborn-v0_8-whitegrid'):
    """
    Plot volatility regimes and regime transitions.
    
    Parameters
    ----------
    returns : Series
        Return series
    realized_vol : Series
        Realized volatility series
    window : int, default=22
        Rolling window for volatility analysis
    n_regimes : int, default=3
        Number of volatility regimes to identify
    figsize : tuple, default=(14, 10)
        Figure size
    style : str, default='seaborn-v0_8-whitegrid'
        Matplotlib style
        
    Returns
    -------
    tuple
        Figure and axes objects
    """
    with plt.style.context(style):
        # Create figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True, 
                                gridspec_kw={'height_ratios': [2, 1, 1]})
        
        # Plot returns
        returns.plot(ax=axes[0], alpha=0.7, linewidth=0.8)
        axes[0].set_title('Asset Returns', fontsize=14)
        axes[0].set_ylabel('Return', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        
        # Plot realized volatility
        realized_vol.plot(ax=axes[1], color='blue', linewidth=1.5)
        axes[1].set_title('Realized Volatility', fontsize=14)
        axes[1].set_ylabel('Volatility', fontsize=12)
        axes[1].yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.1%}'))
        axes[1].grid(True, alpha=0.3)
        
        # Identify volatility regimes
        # Sort volatility into n_regimes quantiles
        vol_quantiles = realized_vol.quantile(np.linspace(0, 1, n_regimes+1))
        
        # Assign regimes based on quantiles
        regimes = pd.cut(realized_vol, bins=vol_quantiles, labels=False, include_lowest=True)
        
        # Create a DataFrame for regimes
        regime_data = pd.DataFrame({
            'Volatility': realized_vol,
            'Regime': regimes
        })
        
        # Plot regime transitions
        regime_colors = plt.cm.viridis(np.linspace(0, 1, n_regimes))
        
        for regime in range(n_regimes):
            regime_periods = regime_data[regime_data['Regime'] == regime]
            axes[2].fill_between(
                regime_periods.index,
                0, 1,
                color=regime_colors[regime],
                alpha=0.7,
                label=f'Regime {regime+1}'
            )
        
        axes[2].set_title('Volatility Regimes', fontsize=14)
        axes[2].set_ylabel('Regime', fontsize=12)
        axes[2].set_yticks([])
        axes[2].legend(loc='upper right')
        axes[2].set_xlabel('Date', fontsize=12)
        
        plt.tight_layout()
        
        return fig, axes


def plot_backtest_results(backtest_results, benchmark_returns=None, 
                         figsize=(14, 12), style='seaborn-v0_8-whitegrid'):
    """
    Plot volatility targeting strategy backtest results.
    
    Parameters
    ----------
    backtest_results : DataFrame
        Results from volatility_targeting_backtest function
    benchmark_returns : Series, optional
        Benchmark returns for comparison
    figsize : tuple, default=(14, 12)
        Figure size
    style : str, default='seaborn-v0_8-whitegrid'
        Matplotlib style
        
    Returns
    -------
    tuple
        Figure and axes objects
    """
    with plt.style.context(style):
        # Create figure with subplots
        fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
        
        # Plot cumulative returns
        backtest_results['cumulative_returns'].plot(
            ax=axes[0], label='Buy & Hold', linewidth=1.5
        )
        backtest_results['cumulative_strategy_net'].plot(
            ax=axes[0], label='Vol Targeting (Net)', linewidth=1.5
        )
        
        if benchmark_returns is not None:
            # Calculate benchmark cumulative returns
            bench_cum = (1 + benchmark_returns).cumprod() - 1
            bench_cum.plot(ax=axes[0], label='Benchmark', linewidth=1.5)
        
        axes[0].set_title('Cumulative Returns', fontsize=14)
        axes[0].set_ylabel('Return', fontsize=12)
        axes[0].legend(loc='upper left')
        axes[0].grid(True, alpha=0.3)
        axes[0].yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.1%}'))
        
        # Plot volatility
        backtest_results['volatility'].plot(
            ax=axes[1], label='Forecast Volatility', linewidth=1.5
        )
        backtest_results['realized_vol'].plot(
            ax=axes[1], label='Realized Volatility', linewidth=1.5
        )
        
        axes[1].set_title('Volatility', fontsize=14)
        axes[1].set_ylabel('Volatility', fontsize=12)
        axes[1].legend(loc='upper left')
        axes[1].grid(True, alpha=0.3)
        axes[1].yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.1%}'))
        
        # Plot weights
        backtest_results['weight'].plot(
            ax=axes[2], label='Position Size', linewidth=1.5
        )
        
        axes[2].set_title('Strategy Weights', fontsize=14)
        axes[2].set_ylabel('Weight', fontsize=12)
        axes[2].legend(loc='upper left')
        axes[2].grid(True, alpha=0.3)
        
        # Plot drawdowns
        backtest_results['drawdown'].plot(
            ax=axes[3], label='Drawdown', linewidth=1.5, color='red'
        )
        
        axes[3].set_title('Drawdown', fontsize=14)
        axes[3].set_ylabel('Drawdown', fontsize=12)
        axes[3].set_xlabel('Date', fontsize=12)
        axes[3].legend(loc='upper left')
        axes[3].grid(True, alpha=0.3)
        axes[3].yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.1%}'))
        
        plt.tight_layout()
        
        return fig, axes


def create_interactive_volatility_dashboard(realized_vol, forecasts, returns=None):
    """
    Create an interactive Plotly dashboard for volatility forecasting.
    
    Parameters
    ----------
    realized_vol : Series
        Realized volatility series
    forecasts : dict
        Dictionary of forecast series {model_name: forecast_series}
    returns : Series, optional
        Return series for additional context
        
    Returns
    -------
    go.Figure
        Plotly figure object
    """
    # Set up subplot
    if returns is not None:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('Volatility Forecasts', 'Asset Returns'),
            row_heights=[0.7, 0.3]
        )
    else:
        fig = make_subplots(
            rows=1, cols=1,
            subplot_titles=('Volatility Forecasts',)
        )
    
    # Add realized volatility
    fig.add_trace(
        go.Scatter(
            x=realized_vol.index,
            y=realized_vol,
            mode='lines',
            name='Realized Volatility',
            line=dict(width=2, color='black')
        ),
        row=1, col=1
    )
    
    # Add forecasts
    colors = px.colors.qualitative.Plotly
    for i, (model_name, forecast) in enumerate(forecasts.items()):
        fig.add_trace(
            go.Scatter(
                x=forecast.index,
                y=forecast,
                mode='lines',
                name=f'{model_name} Forecast',
                line=dict(width=1.5, dash='dash', color=colors[i % len(colors)])
            ),
            row=1, col=1
        )
    
    # Add returns if provided
    if returns is not None:
        fig.add_trace(
            go.Scatter(
                x=returns.index,
                y=returns,
                mode='lines',
                name='Returns',
                line=dict(width=1, color='blue')
            ),
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        title='Volatility Forecast Dashboard',
        height=800,
        width=1200,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template='plotly_white'
    )
    
    # Format y-axis as percentage
    fig.update_yaxes(tickformat='.1%', row=1, col=1)
    
    # Add range slider
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    
    return fig 