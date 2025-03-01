import pandas as pd
import numpy as np
import json
import traceback
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import IsolationForest
from datetime import date, datetime
import logging

def numpy_to_python(obj):
    """
    Convert numpy types to standard Python types for JSON serialization.
    
    Args:
        obj: Object to convert
        
    Returns:
        Converted object
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (date, datetime)):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [numpy_to_python(i) for i in obj]
    else:
        return obj

def safe_json_serialize(obj):
    """
    Safely serialize an object to JSON, handling numpy types
    
    Args:
        obj: Object to serialize
        
    Returns:
        JSON-serializable object
    """
    import numpy as np
    import pandas as pd
    import json
    from datetime import datetime, date
    
    def _serialize(obj):
        """Helper function to serialize objects"""
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        elif isinstance(obj, dict):
            return {k: _serialize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_serialize(item) for item in obj]
        else:
            try:
                # Try to convert to a basic type
                return json.loads(json.dumps(obj))
            except:
                # If all else fails, convert to string
                return str(obj)
    
    return _serialize(obj)

def perform_basic_analysis(df):
    """
    Perform basic statistical analysis on time series data.
    
    Args:
        df (DataFrame): Pandas DataFrame containing time series data
    
    Returns:
        dict: Dictionary containing analysis results and visualizations
    """
    try:
        # Identify datetime columns
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # If no datetime column is found, try to convert the first column
        if not datetime_cols and len(df.columns) > 0:
            try:
                df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])
                datetime_cols = [df.columns[0]]
            except Exception as e:
                print(f"Error converting to datetime: {str(e)}")
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
        
        # Autocorrelation analysis
        autocorrelation = {}
        for col in numeric_cols:
            try:
                if len(df[col].dropna()) > 1:  # Need at least 2 points
                    acf_values = pd.Series(df[col]).autocorr(lag=1)
                    autocorrelation[col] = float(acf_values)  # Convert to float for JSON serialization
            except Exception as e:
                print(f"Error calculating autocorrelation for {col}: {str(e)}")
                autocorrelation[col] = None
        
        # Seasonality detection (simple approach)
        seasonality = {}
        for col in numeric_cols:
            try:
                series = df[col].dropna()
                if len(series) > 2:  # Need at least 3 points
                    # Check if index is datetime for proper resampling
                    if isinstance(df.index, pd.DatetimeIndex):
                        # Try to detect daily, weekly, monthly patterns
                        daily_mean = series.resample('D').mean()
                        daily_std = float(daily_mean.std()) if not pd.isna(daily_mean.std()) else 0
                        weekly_mean = series.resample('W').mean()
                        weekly_std = float(weekly_mean.std()) if not pd.isna(weekly_mean.std()) else 0
                        monthly_mean = series.resample('M').mean()
                        monthly_std = float(monthly_mean.std()) if not pd.isna(monthly_mean.std()) else 0
                        
                        seasonality[col] = {
                            'daily_variation': daily_std,
                            'weekly_variation': weekly_std,
                            'monthly_variation': monthly_std
                        }
            except Exception as e:
                print(f"Error calculating seasonality for {col}: {str(e)}")
                seasonality[col] = {"error": f"Could not calculate seasonality: {str(e)}"}
        
        return {
            'statistics': stats,
            'missing_values': missing_values,
            'autocorrelation': autocorrelation,
            'seasonality': seasonality
        }
    except Exception as e:
        print(f"Error in perform_basic_analysis: {str(e)}")
        print(traceback.format_exc())
        return {
            'error': f"Analysis error: {str(e)}"
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
    try:
        # Identify datetime columns
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # If no datetime column is found, try to convert the first column
        if not datetime_cols and len(df.columns) > 0:
            try:
                df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])
                datetime_cols = [df.columns[0]]
            except Exception as e:
                print(f"Error converting to datetime: {str(e)}")
                pass
        
        # Set the first datetime column as index if available
        if datetime_cols:
            df = df.set_index(datetime_cols[0])
        
        # Get numeric columns for analysis
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if not numeric_cols:
            return {
                'error': 'No numeric columns found for anomaly detection'
            }
        
        results = {}
        
        for col in numeric_cols:
            series = df[col].dropna()
            
            if len(series) < 3:  # Need at least 3 points for anomaly detection
                continue
            
            anomalies = None
            
            try:
                if method == 'zscore':
                    # Z-score method
                    mean = series.mean()
                    std = series.std()
                    z_scores = (series - mean) / std
                    anomalies = series[abs(z_scores) > threshold]
                
                elif method == 'iqr':
                    # IQR method
                    q1 = series.quantile(0.25)
                    q3 = series.quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - (threshold * iqr)
                    upper_bound = q3 + (threshold * iqr)
                    anomalies = series[(series < lower_bound) | (series > upper_bound)]
                
                elif method == 'isolation_forest':
                    # Isolation Forest method
                    model = IsolationForest(contamination=0.05, random_state=42)
                    series_reshaped = series.values.reshape(-1, 1)
                    preds = model.fit_predict(series_reshaped)
                    anomalies = series[preds == -1]
                
                else:
                    return {
                        'error': f'Invalid anomaly detection method: {method}'
                    }
                
                # Store results
                results[col] = {
                    'anomalies': numpy_to_python(anomalies.values) if anomalies is not None else [],
                    'threshold': numpy_to_python(threshold) if threshold is not None else None,
                    'anomaly_indices': numpy_to_python(anomalies.index) if anomalies is not None else []
                }
            except Exception as e:
                print(f"Error detecting anomalies for {col}: {str(e)}")
                results[col] = {'error': f"Could not detect anomalies: {str(e)}"}
        
        return {
            'method': method,
            'threshold': threshold,
            'results': results
        }
    except Exception as e:
        print(f"Error in detect_anomalies: {str(e)}")
        print(traceback.format_exc())
        return {
            'error': f"Anomaly detection error: {str(e)}"
        }

def arima_forecast(series, periods=10):
    """
    Perform ARIMA forecasting on a time series
    
    Parameters:
    -----------
    series : pandas.Series
        The time series data
    periods : int
        Number of periods to forecast
    
    Returns:
    --------
    dict
        Dictionary containing forecast results
    """
    try:
        # Import statsmodels only when needed
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tsa.stattools import adfuller
        import numpy as np
        import pandas as pd
        import logging
        
        logger = logging.getLogger(__name__)
        
        # Convert series to float to ensure compatibility
        series = series.astype(float)
        
        # Check if the series is constant (all values are the same)
        if series.std() == 0:
            logger.warning("Series is constant, cannot apply ARIMA. Using simple constant forecast.")
            # Generate a constant forecast
            forecast_values = np.full(periods, series.iloc[0])
            
            # Create dummy dates for the forecast
            if isinstance(series.index[0], pd.Timestamp):
                # If the index is a datetime, generate future dates
                freq = pd.infer_freq(series.index)
                if freq is None:
                    # If frequency cannot be inferred, use day as default
                    freq = 'D'
                future_dates = pd.date_range(start=series.index[-1], periods=periods+1, freq=freq)[1:]
            else:
                # If the index is not a datetime, just use integers
                future_dates = range(len(series), len(series) + periods)
            
            # Return a simple constant forecast
            return {
                'historical_dates': series.index.tolist(),
                'historical_values': series.values.tolist(),
                'dates': list(future_dates) if isinstance(future_dates, range) else future_dates.tolist(),
                'forecast': forecast_values.tolist(),
                'lower_bounds': forecast_values.tolist(),
                'upper_bounds': forecast_values.tolist(),
                'model_info': {
                    'name': 'Constant',
                    'order': (0, 0, 0),
                    'aic': 0,
                    'is_stationary': True,
                    'p_value': 1.0,
                    'metrics': {},
                    'description': "Constant forecast (input data has no variation)"
                }
            }
        
        # Check for stationarity using Augmented Dickey-Fuller test
        adf_result = adfuller(series.dropna())
        is_stationary = adf_result[1] < 0.05  # p-value < 0.05 indicates stationarity
        
        # Determine differencing parameter (d)
        d = 0 if is_stationary else 1
        
        # Initialize test as an empty Series to avoid the variable reference error
        test = pd.Series()
        
        # For large datasets, use a simpler approach
        if len(series) > 10000:
            logger.info(f"Large dataset detected ({len(series)} points). Using simplified ARIMA approach.")
            
            # Sample the data to improve performance
            sample_size = min(5000, len(series))
            sampled_series = series.sample(n=sample_size)
            sampled_series = sampled_series.sort_index()
            
            # Use a simple ARIMA model with fixed parameters
            model = ARIMA(sampled_series, order=(1, d, 1))
            model_fit = model.fit()
            
            best_order = (1, d, 1)
            best_aic = model_fit.aic
            
            logger.info(f"Using ARIMA{best_order} for large dataset")
        else:
            # Split data for training and testing
            train_size = int(len(series) * 0.8)
            train = series[:train_size]
            test = series[train_size:]
            
            # Auto-select best ARIMA parameters based on AIC
            best_aic = float('inf')
            best_order = (1, d, 1)
            
            # Try different combinations of p and q
            # Limit the search to fewer combinations for large datasets
            max_order = 1 if len(series) > 5000 else 2
            
            for p in range(0, max_order + 1):
                for q in range(0, max_order + 1):
                    try:
                        model = ARIMA(train, order=(p, d, q))
                        model_fit = model.fit()
                        aic = model_fit.aic
                        
                        if aic < best_aic:
                            best_aic = aic
                            best_order = (p, d, q)
                            
                        logger.info(f"ARIMA({p},{d},{q}) AIC: {aic}")
                    except Exception as e:
                        logger.warning(f"Error fitting ARIMA({p},{d},{q}): {str(e)}")
                        continue
            
            logger.info(f"Best ARIMA order: {best_order} with AIC: {best_aic}")
        
        # Fit the best model on the full dataset
        model = ARIMA(series, order=best_order)
        model_fit = model.fit()
        
        # Generate forecast
        forecast = model_fit.forecast(steps=periods)
        
        # Calculate prediction intervals (95% confidence)
        pred_intervals = model_fit.get_forecast(steps=periods).conf_int(alpha=0.05)
        lower_bounds = pred_intervals.iloc[:, 0]
        upper_bounds = pred_intervals.iloc[:, 1]
        
        # Calculate model evaluation metrics if we have test data
        metrics = {}
        if len(test) > 0:
            # Generate in-sample predictions for test set
            in_sample_pred = model_fit.predict(start=train_size, end=len(series)-1)
            
            # Calculate metrics
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            mse = mean_squared_error(test, in_sample_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(test, in_sample_pred)
            
            # Calculate MAPE safely to avoid division by zero
            if (test == 0).any():
                mape = None  # Skip MAPE if there are zeros in the test data
            else:
                mape = np.mean(np.abs((test - in_sample_pred) / test)) * 100
            
            metrics = {
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae),
                'mape': float(mape) if mape is not None and not np.isinf(mape) else None
            }
        
        # Get dates for the forecast
        last_date = series.index[-1]
        
        # Generate future dates
        if isinstance(last_date, pd.Timestamp):
            # If the index is a datetime, generate future dates
            freq = pd.infer_freq(series.index)
            if freq is None:
                # If frequency cannot be inferred, use day as default
                freq = 'D'
            future_dates = pd.date_range(start=last_date, periods=periods+1, freq=freq)[1:]
        else:
            # If the index is not a datetime, just use integers
            future_dates = range(len(series), len(series) + periods)
        
        # Return results
        return {
            'historical_dates': series.index.tolist(),
            'historical_values': series.values.tolist(),
            'dates': list(future_dates) if isinstance(future_dates, range) else future_dates.tolist(),
            'forecast': forecast.tolist(),
            'lower_bounds': lower_bounds.tolist(),
            'upper_bounds': upper_bounds.tolist(),
            'model_info': {
                'name': 'ARIMA',
                'order': best_order,
                'aic': float(best_aic),
                'is_stationary': is_stationary,
                'p_value': float(adf_result[1]),
                'metrics': metrics,
                'description': f"ARIMA({best_order[0]},{best_order[1]},{best_order[2]}) model with auto-optimized parameters"
            }
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'error': str(e)}

def sarima_forecast(series, periods=10):
    """
    Perform SARIMA forecasting on a time series
    
    Parameters:
    -----------
    series : pandas.Series
        The time series data
    periods : int
        Number of periods to forecast
    
    Returns:
    --------
    dict
        Dictionary containing forecast results
    """
    try:
        # Import statsmodels only when needed
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        
        # Convert series to float to ensure compatibility
        series = series.astype(float)
        
        # Fit SARIMA model
        model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        model_fit = model.fit(disp=False)
        
        # Generate forecast
        forecast = model_fit.forecast(steps=periods)
        
        # Get dates for the forecast
        last_date = series.index[-1]
        
        # Generate future dates
        import pandas as pd
        if isinstance(last_date, pd.Timestamp):
            # If the index is a datetime, generate future dates
            freq = pd.infer_freq(series.index)
            if freq is None:
                # If frequency cannot be inferred, use day as default
                freq = 'D'
            future_dates = pd.date_range(start=last_date, periods=periods+1, freq=freq)[1:]
        else:
            # If the index is not a datetime, just use integers
            future_dates = range(len(series), len(series) + periods)
        
        # Return results
        return {
            'historical_dates': series.index.tolist(),
            'historical_values': series.values.tolist(),
            'dates': list(future_dates) if isinstance(future_dates, range) else future_dates.tolist(),
            'forecast': forecast.tolist()
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'error': str(e)}

def exponential_smoothing_forecast(series, periods=10):
    """
    Perform Exponential Smoothing forecasting on a time series
    
    Parameters:
    -----------
    series : pandas.Series
        The time series data
    periods : int
        Number of periods to forecast
    
    Returns:
    --------
    dict
        Dictionary containing forecast results
    """
    try:
        # Import statsmodels only when needed
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        
        # Convert series to float to ensure compatibility
        series = series.astype(float)
        
        # Fit Exponential Smoothing model
        model = ExponentialSmoothing(series, trend='add', seasonal=None)
        model_fit = model.fit()
        
        # Generate forecast
        forecast = model_fit.forecast(periods)
        
        # Get dates for the forecast
        last_date = series.index[-1]
        
        # Generate future dates
        import pandas as pd
        if isinstance(last_date, pd.Timestamp):
            # If the index is a datetime, generate future dates
            freq = pd.infer_freq(series.index)
            if freq is None:
                # If frequency cannot be inferred, use day as default
                freq = 'D'
            future_dates = pd.date_range(start=last_date, periods=periods+1, freq=freq)[1:]
        else:
            # If the index is not a datetime, just use integers
            future_dates = range(len(series), len(series) + periods)
        
        # Return results
        return {
            'historical_dates': series.index.tolist(),
            'historical_values': series.values.tolist(),
            'dates': list(future_dates) if isinstance(future_dates, range) else future_dates.tolist(),
            'forecast': forecast.tolist()
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'error': str(e)}

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
    try:
        import pandas as pd
        import numpy as np
        import logging
        
        logger = logging.getLogger(__name__)
        
        # Limit data size for performance
        max_rows = 5000
        if len(df) > max_rows:
            logger.warning(f"Dataset too large ({len(df)} rows), sampling to {max_rows} rows for forecasting")
            df = df.sample(n=max_rows) if len(df) > max_rows else df
            df = df.sort_index()
        
        # Get numeric columns for forecasting
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            return {'error': 'No numeric columns found for forecasting'}
        
        logger.info(f"Found {len(numeric_cols)} numeric columns for forecasting: {numeric_cols}")
        
        # Identify datetime columns
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # If no datetime column is found, try to convert the first column
        if not datetime_cols and len(df.columns) > 0:
            try:
                print(f"No datetime column found, attempting to convert first column: {df.columns[0]}")
                df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])
                datetime_cols = [df.columns[0]]
                print(f"Successfully converted {df.columns[0]} to datetime")
            except Exception as e:
                print(f"Error converting to datetime: {str(e)}")
                return {'error': f"Could not identify or convert datetime column: {str(e)}"}
        
        # Set the first datetime column as index if available
        if datetime_cols:
            print(f"Setting index to datetime column: {datetime_cols[0]}")
            df = df.set_index(datetime_cols[0])
        else:
            return {'error': "No datetime column found for forecasting"}
        
        # Check if index is DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            return {'error': "Index must be a DatetimeIndex for forecasting"}
        
        results = {}
        
        for col in numeric_cols:
            try:
                print(f"Processing column: {col}")
                series = df[col].dropna()
                
                if len(series) < 10:  # Need sufficient data for forecasting
                    print(f"Not enough data points for {col}: {len(series)}")
                    results[col] = {'error': 'Not enough data points for forecasting (minimum 10 required)'}
                    continue
                
                # Determine forecast frequency
                freq = pd.infer_freq(series.index)
                if freq is None:
                    # Try to infer frequency from the first few observations
                    freq = pd.infer_freq(series.index[:10])
                
                if freq is None:
                    # If still can't infer, default to daily
                    freq = 'D'
                    print(f"Warning: Could not infer frequency for {col}, defaulting to daily (D)")
                
                print(f"Using frequency: {freq}")
                
                # Create future dates for forecasting
                last_date = series.index[-1]
                future_dates = pd.date_range(start=last_date, periods=periods+1, freq=freq)[1:]
                future_dates_str = future_dates.strftime('%Y-%m-%d').tolist()
                
                print(f"Future dates: {future_dates_str}")
                
                forecast_values = None
                lower_ci = None
                upper_ci = None
                
                if model == 'arima':
                    forecast_result = arima_forecast(series, periods=periods)
                    forecast_values = forecast_result['forecast']
                    historical_dates = forecast_result['historical_dates']
                    historical_values = forecast_result['historical_values']
                    dates = forecast_result['dates']
                
                elif model == 'sarima':
                    forecast_result = sarima_forecast(series, periods=periods)
                    forecast_values = forecast_result['forecast']
                    historical_dates = forecast_result['historical_dates']
                    historical_values = forecast_result['historical_values']
                    dates = forecast_result['dates']
                
                elif model == 'exponential_smoothing':
                    forecast_result = exponential_smoothing_forecast(series, periods=periods)
                    forecast_values = forecast_result['forecast']
                    historical_dates = forecast_result['historical_dates']
                    historical_values = forecast_result['historical_values']
                    dates = forecast_result['dates']
                
                else:
                    results[col] = {'error': f'Invalid forecasting model: {model}'}
                    continue
                
                # Store results
                results[col] = {
                    'forecast': forecast_values,
                    'dates': dates,
                    'historical_dates': historical_dates,
                    'historical_values': historical_values
                }
                
                print(f"Completed forecasting for column: {col}")
                
            except Exception as e:
                print(f"Error forecasting for {col}: {str(e)}")
                print(traceback.format_exc())
                results[col] = {'error': f"Forecasting failed: {str(e)}"}
        
        print("Forecasting complete, returning results")
        return {
            'model': model,
            'periods': periods,
            'results': results
        }
    except Exception as e:
        print(f"Error in perform_forecasting: {str(e)}")
        print(traceback.format_exc())
        return {
            'error': f"Forecasting error: {str(e)}"
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
