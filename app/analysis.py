import pandas as pd
import numpy as np
import json
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import IsolationForest
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder

def perform_basic_analysis(df):
    """
    Perform basic statistical analysis on time series data.
    
    Args:
        df (DataFrame): Pandas DataFrame containing time series data
    
    Returns:
        dict: Dictionary containing analysis results and visualizations
    """
    # Identify datetime columns
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    # If no datetime column is found, try to convert the first column
    if not datetime_cols and len(df.columns) > 0:
        try:
            df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])
            datetime_cols = [df.columns[0]]
        except:
            pass
    
    # Set the first datetime column as index if available
    if datetime_cols:
        df = df.set_index(datetime_cols[0])
    
    # Get numeric columns for analysis
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if not numeric_cols:
        return {
            'error': 'No numeric columns found for analysis'
        }
    
    # Basic statistics
    stats = df[numeric_cols].describe().to_dict()
    
    # Check for missing values
    missing_values = df[numeric_cols].isnull().sum().to_dict()
    
    # Create time series plot for each numeric column
    plots = {}
    for col in numeric_cols:
        fig = px.line(df, y=col, title=f'Time Series Plot: {col}')
        plots[col] = json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))
    
    # Autocorrelation analysis
    autocorrelation = {}
    for col in numeric_cols:
        if len(df[col].dropna()) > 1:  # Need at least 2 points
            acf_values = pd.Series(df[col]).autocorr(lag=1)
            autocorrelation[col] = acf_values
    
    # Seasonality detection (simple approach)
    seasonality = {}
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) > 2:  # Need at least 3 points
            # Check if index is datetime for proper resampling
            if isinstance(df.index, pd.DatetimeIndex):
                # Try to detect daily, weekly, monthly patterns
                daily_mean = series.resample('D').mean()
                daily_std = daily_mean.std()
                weekly_mean = series.resample('W').mean()
                weekly_std = weekly_mean.std()
                monthly_mean = series.resample('M').mean()
                monthly_std = monthly_mean.std()
                
                seasonality[col] = {
                    'daily_variation': daily_std,
                    'weekly_variation': weekly_std,
                    'monthly_variation': monthly_std
                }
    
    return {
        'statistics': stats,
        'missing_values': missing_values,
        'autocorrelation': autocorrelation,
        'seasonality': seasonality,
        'plots': plots
    }

def detect_anomalies(df, method='zscore', threshold=3):
    """
    Detect anomalies in time series data.
    
    Args:
        df (DataFrame): Pandas DataFrame containing time series data
        method (str): Method to use for anomaly detection ('zscore', 'iqr', 'isolation_forest')
        threshold (float): Threshold for anomaly detection
    
    Returns:
        dict: Dictionary containing anomaly detection results and visualizations
    """
    # Identify datetime columns
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    # If no datetime column is found, try to convert the first column
    if not datetime_cols and len(df.columns) > 0:
        try:
            df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])
            datetime_cols = [df.columns[0]]
        except:
            pass
    
    # Set the first datetime column as index if available
    if datetime_cols:
        df = df.set_index(datetime_cols[0])
    
    # Get numeric columns for analysis
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if not numeric_cols:
        return {
            'error': 'No numeric columns found for analysis'
        }
    
    results = {}
    plots = {}
    
    for col in numeric_cols:
        series = df[col].dropna()
        
        if len(series) < 3:
            results[col] = {'error': 'Not enough data points'}
            continue
        
        anomalies = pd.Series(False, index=series.index)
        
        if method == 'zscore':
            # Z-score method
            mean = series.mean()
            std = series.std()
            z_scores = (series - mean) / std
            anomalies = abs(z_scores) > threshold
            
        elif method == 'iqr':
            # IQR method
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - (threshold * iqr)
            upper_bound = q3 + (threshold * iqr)
            anomalies = (series < lower_bound) | (series > upper_bound)
            
        elif method == 'isolation_forest':
            # Isolation Forest method
            model = IsolationForest(contamination=0.05, random_state=42)
            anomalies = pd.Series(
                model.fit_predict(series.values.reshape(-1, 1)),
                index=series.index
            )
            anomalies = anomalies == -1  # -1 indicates anomaly
        
        # Create visualization
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=series.index, 
            y=series.values,
            mode='lines+markers',
            name=col
        ))
        
        # Add anomalies
        fig.add_trace(go.Scatter(
            x=series[anomalies].index,
            y=series[anomalies].values,
            mode='markers',
            marker=dict(color='red', size=10),
            name='Anomalies'
        ))
        
        fig.update_layout(title=f'Anomaly Detection: {col} (Method: {method})')
        
        # Store results
        results[col] = {
            'total_points': len(series),
            'anomalies_count': anomalies.sum(),
            'anomalies_percentage': (anomalies.sum() / len(series)) * 100,
            'anomalies_indices': anomalies[anomalies].index.tolist()
        }
        
        plots[col] = json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))
    
    return {
        'results': results,
        'plots': plots,
        'method': method
    }

def perform_forecasting(df, model='arima', periods=10):
    """
    Perform time series forecasting.
    
    Args:
        df (DataFrame): Pandas DataFrame containing time series data
        model (str): Forecasting model to use ('arima', 'sarima', 'exponential_smoothing')
        periods (int): Number of periods to forecast
    
    Returns:
        dict: Dictionary containing forecasting results and visualizations
    """
    # Identify datetime columns
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    # If no datetime column is found, try to convert the first column
    if not datetime_cols and len(df.columns) > 0:
        try:
            df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])
            datetime_cols = [df.columns[0]]
        except:
            pass
    
    # Set the first datetime column as index if available
    if datetime_cols:
        df = df.set_index(datetime_cols[0])
    
    # Get numeric columns for analysis
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if not numeric_cols:
        return {
            'error': 'No numeric columns found for analysis'
        }
    
    results = {}
    plots = {}
    
    for col in numeric_cols:
        series = df[col].dropna()
        
        if len(series) < 5:
            results[col] = {'error': 'Not enough data points for forecasting'}
            continue
        
        # Generate future dates for forecasting
        if isinstance(series.index, pd.DatetimeIndex):
            # Calculate the average frequency
            if len(series.index) > 1:
                avg_freq = (series.index[-1] - series.index[0]) / (len(series.index) - 1)
                future_dates = [series.index[-1] + (i + 1) * avg_freq for i in range(periods)]
            else:
                # Default to daily if only one data point
                future_dates = pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), periods=periods)
        else:
            # Use integer indices if not datetime
            future_dates = range(len(series), len(series) + periods)
        
        forecast_values = None
        confidence_intervals = None
        
        try:
            if model == 'arima':
                # ARIMA model (p,d,q) = (1,1,1) as default
                arima_model = ARIMA(series, order=(1, 1, 1))
                arima_result = arima_model.fit()
                forecast = arima_result.forecast(steps=periods)
                forecast_values = forecast
                
                # Get confidence intervals
                pred_conf = arima_result.get_forecast(steps=periods).conf_int()
                confidence_intervals = {
                    'lower': pred_conf.iloc[:, 0].values,
                    'upper': pred_conf.iloc[:, 1].values
                }
                
            elif model == 'sarima':
                # SARIMA model with seasonal component
                sarima_model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
                sarima_result = sarima_model.fit(disp=False)
                forecast = sarima_result.forecast(steps=periods)
                forecast_values = forecast
                
                # Get confidence intervals
                pred_conf = sarima_result.get_forecast(steps=periods).conf_int()
                confidence_intervals = {
                    'lower': pred_conf.iloc[:, 0].values,
                    'upper': pred_conf.iloc[:, 1].values
                }
                
            elif model == 'exponential_smoothing':
                # Exponential Smoothing
                exp_model = ExponentialSmoothing(series, trend='add', seasonal=None)
                exp_result = exp_model.fit()
                forecast = exp_result.forecast(periods)
                forecast_values = forecast
                
                # No built-in confidence intervals for ExponentialSmoothing
                confidence_intervals = None
            
            # Create visualization
            fig = go.Figure()
            
            # Add historical data
            fig.add_trace(go.Scatter(
                x=series.index, 
                y=series.values,
                mode='lines',
                name='Historical'
            ))
            
            # Add forecast
            fig.add_trace(go.Scatter(
                x=future_dates, 
                y=forecast_values,
                mode='lines',
                line=dict(dash='dash'),
                name='Forecast'
            ))
            
            # Add confidence intervals if available
            if confidence_intervals:
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=confidence_intervals['upper'],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=confidence_intervals['lower'],
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor='rgba(0, 176, 246, 0.2)',
                    name='Confidence Interval'
                ))
            
            fig.update_layout(title=f'Forecasting: {col} (Model: {model})')
            
            # Store results
            results[col] = {
                'forecast_values': forecast_values.tolist() if hasattr(forecast_values, 'tolist') else forecast_values,
                'model': model,
                'periods': periods
            }
            
            if confidence_intervals:
                results[col]['confidence_intervals'] = {
                    'lower': confidence_intervals['lower'].tolist() if hasattr(confidence_intervals['lower'], 'tolist') else confidence_intervals['lower'],
                    'upper': confidence_intervals['upper'].tolist() if hasattr(confidence_intervals['upper'], 'tolist') else confidence_intervals['upper']
                }
            
            plots[col] = json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))
            
        except Exception as e:
            results[col] = {'error': str(e)}
    
    return {
        'results': results,
        'plots': plots,
        'model': model,
        'periods': periods
    }

def get_available_models():
    """
    Get a list of available forecasting models.
    
    Returns:
        list: List of available models
    """
    return [
        {'id': 'arima', 'name': 'ARIMA', 'description': 'AutoRegressive Integrated Moving Average'},
        {'id': 'sarima', 'name': 'SARIMA', 'description': 'Seasonal ARIMA'},
        {'id': 'exponential_smoothing', 'name': 'Exponential Smoothing', 'description': 'Exponential Smoothing with trend'}
    ]
