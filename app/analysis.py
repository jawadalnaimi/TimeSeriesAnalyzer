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
import math

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
    Safely serialize an object to JSON, handling numpy types and other complex objects
    
    Args:
        obj: Object to serialize
        
    Returns:
        JSON-serializable object
    """
    import numpy as np
    import pandas as pd
    import json
    from datetime import datetime, date
    import logging
    
    logger = logging.getLogger(__name__)
    
    def _serialize(obj):
        """Helper function to serialize objects"""
        # Handle None values
        if obj is None:
            return None
            
        # Handle basic Python types directly
        if isinstance(obj, str):
            return obj
        elif isinstance(obj, (int, bool)):
            return obj
        elif isinstance(obj, float):
            # Handle NaN and Infinity values for Python floats
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj
            
        # Handle numpy numeric types
        elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
            
        # Handle numpy arrays
        elif isinstance(obj, (np.ndarray,)):
            try:
                return [_serialize(x) for x in obj.tolist()]
            except Exception as e:
                logger.warning(f"Error serializing numpy array: {str(e)}")
                return None
                
        # Handle datetime objects
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
            
        # Handle pandas Series
        elif isinstance(obj, pd.Series):
            try:
                return [_serialize(x) for x in obj.tolist()]
            except Exception as e:
                logger.warning(f"Error serializing pandas Series: {str(e)}")
                return None
                
        # Handle pandas DataFrame
        elif isinstance(obj, pd.DataFrame):
            try:
                return obj.to_dict(orient='records')
            except Exception as e:
                logger.warning(f"Error serializing pandas DataFrame: {str(e)}")
                return None
                
        # Handle dictionaries
        elif isinstance(obj, dict):
            try:
                return {str(k): _serialize(v) for k, v in obj.items()}
            except Exception as e:
                logger.warning(f"Error serializing dictionary: {str(e)}")
                return None
                
        # Handle lists and tuples
        elif isinstance(obj, (list, tuple)):
            try:
                return [_serialize(item) for item in obj]
            except Exception as e:
                logger.warning(f"Error serializing list/tuple: {str(e)}")
                return None
                
        # Try to convert to a basic type
        else:
            try:
                # Try to convert to a basic type via JSON
                json_str = json.dumps(obj)
                return json.loads(json_str)
            except Exception as e:
                logger.warning(f"Could not JSON serialize object of type {type(obj)}: {str(e)}")
                # If all else fails, convert to string
                try:
                    return str(obj)
                except Exception as e:
                    logger.error(f"Failed to convert object to string: {str(e)}")
                    return "Unserializable object"
    
    try:
        serialized = _serialize(obj)
        
        # Final validation - ensure the object is actually JSON serializable
        try:
            # Use a custom JSON encoder to handle any remaining NaN or Infinity values
            class CustomJSONEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, float):
                        if math.isnan(obj) or math.isinf(obj):
                            return None
                    return super().default(obj)
            
            # Test serialization with custom encoder
            json.dumps(serialized, cls=CustomJSONEncoder)
            
            # Replace any NaN values that might have been missed
            def replace_nan_values(obj):
                if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
                    return None
                elif isinstance(obj, dict):
                    return {k: replace_nan_values(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [replace_nan_values(item) for item in obj]
                return obj
            
            # Apply the NaN replacement
            serialized = replace_nan_values(serialized)
            
            return serialized
        except Exception as e:
            logger.error(f"Final JSON validation failed: {str(e)}")
            return {"error": f"JSON validation error: {str(e)}"}
            
    except Exception as e:
        logger.error(f"Error in safe_json_serialize: {str(e)}")
        logger.error(traceback.format_exc())
        return {"error": f"Serialization error: {str(e)}"}

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

def prophet_forecast(series, periods=10):
    """
    Perform Prophet forecasting on a time series
    
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
        # Import required libraries
        import pandas as pd
        import numpy as np
        try:
            from prophet import Prophet
        except ImportError:
            # If Prophet is not installed, return a helpful error message
            return {
                'error': 'Prophet is not installed. Please install it with: pip install prophet',
                'forecast': None,
                'model_info': {
                    'name': 'Prophet',
                    'status': 'Not available'
                }
            }
        import logging
        
        logger = logging.getLogger(__name__)
        
        # Check if the series is constant
        if series.std() == 0:
            logger.warning("Series is constant, cannot apply Prophet. Using simple constant forecast.")
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
                    'aic': 0,
                    'is_stationary': True,
                    'p_value': 1.0,
                    'metrics': {},
                    'description': "Constant forecast (input data has no variation)"
                }
            }
        
        # Prepare data for Prophet
        df = pd.DataFrame({'ds': series.index, 'y': series.values})
        
        # Initialize test as an empty DataFrame
        test_df = pd.DataFrame()
        
        # Split data for training and testing
        if len(series) > 5:
            train_size = int(len(series) * 0.8)
            train_df = df.iloc[:train_size]
            test_df = df.iloc[train_size:]
        else:
            train_df = df
        
        # Initialize and fit Prophet model
        model = Prophet(
            yearly_seasonality='auto',
            weekly_seasonality='auto',
            daily_seasonality='auto',
            seasonality_mode='additive',
            interval_width=0.95  # 95% prediction intervals
        )
        
        # Add additional seasonality if enough data
        if len(train_df) >= 30:
            model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        
        # Fit the model
        model.fit(train_df)
        
        # Create future dataframe for forecasting
        if isinstance(series.index[0], pd.Timestamp):
            # If the index is a datetime, use Prophet's make_future_dataframe
            future = model.make_future_dataframe(periods=periods, freq=pd.infer_freq(series.index) or 'D')
        else:
            # If the index is not a datetime, create a custom future dataframe
            last_date = pd.to_datetime('today')
            future = pd.DataFrame({'ds': pd.date_range(start=last_date, periods=len(df) + periods, freq='D')})
        
        # Generate forecast
        forecast = model.predict(future)
        
        # Calculate metrics if test data is available
        metrics = {}
        if len(test_df) > 0:
            # Generate predictions for test set
            test_forecast = model.predict(test_df[['ds']])
            
            # Calculate metrics
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            
            # Extract actual and predicted values
            y_true = test_df['y'].values
            y_pred = test_forecast['yhat'].values
            
            # Calculate metrics
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            
            # Calculate MAPE safely
            if np.all(y_true != 0):
                mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            else:
                # Handle zeros in test data
                non_zero_indices = y_true != 0
                if np.any(non_zero_indices):
                    mape = np.mean(np.abs((y_true[non_zero_indices] - y_pred[non_zero_indices]) / y_true[non_zero_indices])) * 100
                else:
                    mape = None
            
            metrics = {
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae),
                'mape': float(mape) if mape is not None and not np.isinf(mape) else None
            }
        
        # Extract forecast components
        forecast_values = forecast['yhat'].iloc[-periods:].values
        lower_bounds = forecast['yhat_lower'].iloc[-periods:].values
        upper_bounds = forecast['yhat_upper'].iloc[-periods:].values
        forecast_dates = forecast['ds'].iloc[-periods:].values
        
        # Return results
        return {
            'historical_dates': series.index.tolist(),
            'historical_values': series.values.tolist(),
            'dates': forecast_dates.tolist(),
            'forecast': forecast_values.tolist(),
            'lower_bounds': lower_bounds.tolist(),
            'upper_bounds': upper_bounds.tolist(),
            'model_info': {
                'name': 'Prophet',
                'metrics': metrics,
                'components': {
                    'trend': forecast['trend'].iloc[-periods:].values.tolist(),
                    'yearly': forecast['yearly'].iloc[-periods:].values.tolist() if 'yearly' in forecast.columns else None,
                    'weekly': forecast['weekly'].iloc[-periods:].values.tolist() if 'weekly' in forecast.columns else None,
                    'daily': forecast['daily'].iloc[-periods:].values.tolist() if 'daily' in forecast.columns else None,
                    'monthly': forecast['monthly'].iloc[-periods:].values.tolist() if 'monthly' in forecast.columns else None
                },
                'description': "Facebook Prophet model with auto-detected seasonality"
            }
        }
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'error': str(e)}

def perform_forecasting(df, model='arima', periods=10, forecast_periods=None):
    """
    Perform time series forecasting.
    
    Args:
        df (DataFrame): Pandas DataFrame containing time series data
        model (str): Forecasting model to use ('arima', 'sarima', 'exponential_smoothing', 'prophet', 'lstm')
        periods (int): Number of periods to forecast (default)
        forecast_periods (list): Optional list of different forecast horizons (e.g. [10, 30, 90])
    
    Returns:
        dict: Dictionary containing forecasting results and visualizations
    """
    try:
        import pandas as pd
        import numpy as np
        import logging
        
        logger = logging.getLogger(__name__)
        
        # Initialize results dictionary
        results = {}
        
        # Set default forecast periods if not provided
        if forecast_periods is None:
            forecast_periods = [periods]
        
        # Ensure forecast_periods is a list
        if not isinstance(forecast_periods, list):
            forecast_periods = [forecast_periods]
        
        # Sort and limit to 3 periods maximum
        forecast_periods = sorted(forecast_periods)[:3]
        
        # Find numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if not numeric_cols:
            return {'error': 'No numeric columns found in the dataset'}
        
        # Find datetime columns
        datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
        
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
        
        # For each numeric column, perform forecasting
        for col in numeric_cols:
            # Skip columns with all NaN values
            if df[col].isna().all():
                results[col] = {'error': 'Column contains only NaN values'}
                continue
            
            # Get the series
            series = df[col].dropna()
            
            # Skip if not enough data
            if len(series) < 5:
                results[col] = {'error': 'Not enough data points for forecasting (minimum 5 required)'}
                continue
            
            # Initialize forecasts dictionary for this column
            col_results = {'forecasts': {}}
            
            # For each forecast period
            for period in forecast_periods:
                logger.info(f"Forecasting {col} with {model} model for {period} periods")
                
                # Perform forecasting based on the selected model
                if model == 'arima':
                    forecast_result = arima_forecast(series, periods=period)
                    
                elif model == 'sarima':
                    forecast_result = sarima_forecast(series, periods=period)
                    
                elif model == 'exponential_smoothing':
                    forecast_result = exponential_smoothing_forecast(series, periods=period)
                    
                elif model == 'prophet':
                    forecast_result = prophet_forecast(series, periods=period)
                    
                elif model == 'lstm':
                    forecast_result = lstm_forecast(series, periods=period)
                    
                else:
                    results[col] = {'error': f'Invalid forecasting model: {model}'}
                    continue
                
                # Check for errors
                if 'error' in forecast_result:
                    col_results['forecasts'][str(period)] = {'error': forecast_result['error']}
                    continue
                
                # Store forecast result for this period
                col_results['forecasts'][str(period)] = forecast_result
                
                # Store common information only once
                if 'historical_dates' not in col_results:
                    col_results['historical_dates'] = forecast_result['historical_dates']
                    col_results['historical_values'] = forecast_result['historical_values']
                    col_results['model_info'] = forecast_result['model_info']
            
            # Add to results
            results[col] = col_results
        
        return results
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'error': str(e)}

def get_available_models():
    """
    Get a list of available forecasting models.
    
    Returns:
        list: List of available models
    """
    return [
        {'id': 'arima', 'name': 'ARIMA', 'description': 'AutoRegressive Integrated Moving Average'},
        {'id': 'sarima', 'name': 'SARIMA', 'description': 'Seasonal ARIMA'},
        {'id': 'exponential_smoothing', 'name': 'Exponential Smoothing', 'description': 'Exponential Smoothing with trend'},
        {'id': 'prophet', 'name': 'Prophet', 'description': 'Facebook Prophet'},
        {'id': 'lstm', 'name': 'LSTM', 'description': 'Long Short-Term Memory'}
    ]

def perform_enhanced_eda(df):
    """
    Perform enhanced Exploratory Data Analysis (EDA) on time series data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The time series data
    
    Returns:
    --------
    dict
        Dictionary containing enhanced EDA results
    """
    try:
        import pandas as pd
        import numpy as np
        from scipy import stats
        import statsmodels.api as sm
        from statsmodels.tsa.seasonal import seasonal_decompose
        from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
        import logging
        import traceback
        
        logger = logging.getLogger(__name__)
        logger.info("Starting enhanced EDA analysis")
        
        # Validate input
        if not isinstance(df, pd.DataFrame):
            logger.error("Input is not a pandas DataFrame")
            return {'error': 'Input must be a pandas DataFrame'}
        
        if df.empty:
            logger.error("Input DataFrame is empty")
            return {'error': 'Input DataFrame is empty'}
        
        # Store original DataFrame before any index modifications
        original_df = df.copy()
        
        # Find numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if not numeric_cols:
            logger.error("No numeric columns found in the dataset")
            return {
                'error': 'No numeric columns found in the dataset',
                'error_details': 'The dataset must contain at least one numeric column for time series analysis',
                'error_suggestions': [
                    'Check if your data contains numeric values',
                    'Ensure numeric columns are properly formatted',
                    'Try converting string columns to numeric if appropriate'
                ]
            }
        
        logger.info(f"Found {len(numeric_cols)} numeric columns: {', '.join(numeric_cols)}")
        
        # Find potential datetime columns
        datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
        
        # If no datetime column is found, try to convert the first column
        if not datetime_cols and len(df.columns) > 0:
            try:
                logger.info(f"No datetime columns found, attempting to convert column '{df.columns[0]}'")
                df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])
                datetime_cols = [df.columns[0]]
                logger.info(f"Successfully converted '{df.columns[0]}' to datetime")
            except Exception as e:
                logger.warning(f"Failed to convert column to datetime: {str(e)}")
        
        # Set the first datetime column as index if available
        if datetime_cols:
            logger.info(f"Using '{datetime_cols[0]}' as datetime index")
            df = df.set_index(datetime_cols[0])
        else:
            logger.warning("No datetime column found or created, using default index")
        
        # Results dictionary
        results = {
            'summary_stats': {},
            'stationarity_tests': {},
            'autocorrelation': {},
            'distribution_analysis': {},
            'seasonal_decomposition': {},
            'trend_analysis': {}
        }
        
        # Process each numeric column
        for col in numeric_cols:
            logger.info(f"Processing column: {col}")
            
            try:
                # Get the series data, handling the case where it might be in the index
                if col in df.columns:
                    series = df[col].dropna()
                elif col in original_df.columns and col not in df.columns:
                    # The column might have been set as index
                    if col in datetime_cols:
                        # Skip if it's a datetime column that was set as index
                        continue
                    # Use the original dataframe to access the column
                    series = original_df[col].dropna()
                else:
                    logger.warning(f"Column {col} not found in DataFrame")
                    continue
                
                # Skip if not enough data
                if len(series) < 5:
                    logger.warning(f"Column {col} has insufficient data points (< 5)")
                    results['summary_stats'][col] = {
                        'error': 'Not enough data points',
                        'count': int(series.count())
                    }
                    continue
                
                # Calculate summary statistics
                try:
                    stats_dict = {
                        'count': int(series.count()),
                        'mean': float(series.mean()),
                        'std': float(series.std()),
                        'min': float(series.min()),
                        'q1': float(series.quantile(0.25)),
                        'median': float(series.median()),
                        'q3': float(series.quantile(0.75)),
                        'max': float(series.max()),
                        'skewness': float(stats.skew(series.dropna())),
                        'kurtosis': float(stats.kurtosis(series.dropna())),
                        'missing_values': int(series.isna().sum()),
                        'missing_percentage': float(series.isna().mean() * 100)
                    }
                    
                    # Calculate rolling statistics
                    if len(series) >= 10:
                        rolling_mean = series.rolling(window=min(5, len(series)//2)).mean()
                        rolling_std = series.rolling(window=min(5, len(series)//2)).std()
                        stats_dict['rolling_mean_last'] = float(rolling_mean.iloc[-1]) if not pd.isna(rolling_mean.iloc[-1]) else None
                        stats_dict['rolling_std_last'] = float(rolling_std.iloc[-1]) if not pd.isna(rolling_std.iloc[-1]) else None
                    
                    results['summary_stats'][col] = stats_dict
                    logger.info(f"Completed summary statistics for {col}")
                except Exception as e:
                    logger.error(f"Error calculating summary statistics for {col}: {str(e)}")
                    results['summary_stats'][col] = {'error': f"Error in summary statistics: {str(e)}"}
                
                # Stationarity tests
                try:
                    # Only perform if we have enough data
                    if len(series.dropna()) < 8:
                        logger.warning(f"Column {col} has insufficient data for stationarity tests (< 8)")
                        results['stationarity_tests'][col] = {'error': 'Not enough data points for stationarity tests'}
                        continue
                        
                    # Augmented Dickey-Fuller test
                    adf_result = adfuller(series.dropna())
                    adf_dict = {
                        'test_statistic': float(adf_result[0]),
                        'p_value': float(adf_result[1]),
                        'critical_values': {str(key): float(val) for key, val in adf_result[4].items()},
                        'is_stationary': bool(adf_result[1] < 0.05)
                    }
                    
                    # KPSS test
                    kpss_result = kpss(series.dropna())
                    kpss_dict = {
                        'test_statistic': float(kpss_result[0]),
                        'p_value': float(kpss_result[1]),
                        'critical_values': {str(key): float(val) for key, val in kpss_result[3].items()},
                        'is_stationary': bool(kpss_result[1] > 0.05)
                    }
                    
                    results['stationarity_tests'][col] = {
                        'adf_test': adf_dict,
                        'kpss_test': kpss_dict
                    }
                    logger.info(f"Completed stationarity tests for {col}")
                except Exception as e:
                    logger.warning(f"Error in stationarity tests for column {col}: {str(e)}")
                    results['stationarity_tests'][col] = {'error': f"Error in stationarity tests: {str(e)}"}
                
                # Autocorrelation analysis
                try:
                    # Only perform if we have enough data
                    if len(series.dropna()) < 8:
                        logger.warning(f"Column {col} has insufficient data for autocorrelation analysis (< 8)")
                        results['autocorrelation'][col] = {'error': 'Not enough data points for autocorrelation analysis'}
                        continue
                        
                    # Calculate ACF and PACF
                    max_lags = min(20, len(series.dropna()) // 2)
                    if max_lags < 2:
                        logger.warning(f"Column {col} has insufficient data for autocorrelation analysis (max_lags < 2)")
                        results['autocorrelation'][col] = {'error': 'Not enough data points for autocorrelation analysis'}
                        continue
                        
                    acf_values = acf(series.dropna(), nlags=max_lags)
                    pacf_values = pacf(series.dropna(), nlags=max_lags)
                    
                    results['autocorrelation'][col] = {
                        'acf': acf_values.tolist(),
                        'pacf': pacf_values.tolist(),
                        'lags': list(range(len(acf_values)))
                    }
                    logger.info(f"Completed autocorrelation analysis for {col}")
                except Exception as e:
                    logger.warning(f"Error in autocorrelation analysis for column {col}: {str(e)}")
                    results['autocorrelation'][col] = {'error': f"Error in autocorrelation analysis: {str(e)}"}
                
                # Distribution analysis
                try:
                    # Shapiro-Wilk test for normality
                    if len(series.dropna()) < 5:
                        logger.warning(f"Column {col} has insufficient data for distribution analysis (< 5)")
                        results['distribution_analysis'][col] = {'error': 'Not enough data points for distribution analysis'}
                        continue
                        
                    if len(series.dropna()) < 5000:  # Shapiro-Wilk is only valid for n < 5000
                        shapiro_test = stats.shapiro(series.dropna())
                        normality_test = {
                            'test_name': 'Shapiro-Wilk',
                            'test_statistic': float(shapiro_test[0]),
                            'p_value': float(shapiro_test[1]),
                            'is_normal': bool(shapiro_test[1] > 0.05)
                        }
                    else:
                        # Use D'Agostino's K^2 test for larger samples
                        k2_test = stats.normaltest(series.dropna())
                        normality_test = {
                            'test_name': "D'Agostino's K^2",
                            'test_statistic': float(k2_test[0]),
                            'p_value': float(k2_test[1]),
                            'is_normal': bool(k2_test[1] > 0.05)
                        }
                    
                    # Create histogram with appropriate number of bins
                    n_bins = min(20, max(5, len(series.dropna()) // 10))
                    hist_data = np.histogram(series.dropna(), bins=n_bins)
                    
                    results['distribution_analysis'][col] = {
                        'normality_test': normality_test,
                        'histogram_data': {
                            'bin_edges': hist_data[1].tolist(),
                            'bin_counts': hist_data[0].tolist()
                        }
                    }
                    logger.info(f"Completed distribution analysis for {col}")
                except Exception as e:
                    logger.warning(f"Error in distribution analysis for column {col}: {str(e)}")
                    results['distribution_analysis'][col] = {'error': f"Error in distribution analysis: {str(e)}"}
                
                # Seasonal decomposition (if datetime index)
                if isinstance(df.index, pd.DatetimeIndex) and len(series.dropna()) >= 10:
                    try:
                        logger.info(f"Attempting seasonal decomposition for {col}")
                        # Try to infer frequency
                        if df.index.inferred_freq is None:
                            # Try common frequencies
                            decomposition_success = False
                            for freq in ['D', 'W', 'M', 'Q', 'Y']:
                                try:
                                    # Resample to handle missing values
                                    resampled = series.resample(freq).mean()
                                    if len(resampled.dropna()) >= 10:  # Need enough data points
                                        # Fill missing values for decomposition
                                        resampled_filled = resampled.interpolate(method='linear')
                                        decomposition = seasonal_decompose(resampled_filled, model='additive')
                                        
                                        results['seasonal_decomposition'][col] = {
                                            'frequency': freq,
                                            'trend': decomposition.trend.dropna().tolist(),
                                            'seasonal': decomposition.seasonal.dropna().tolist(),
                                            'residual': decomposition.resid.dropna().tolist(),
                                            'dates': decomposition.trend.dropna().index.strftime('%Y-%m-%d').tolist()
                                        }
                                        decomposition_success = True
                                        logger.info(f"Successful seasonal decomposition with frequency '{freq}' for {col}")
                                        break
                                except Exception as e:
                                    logger.debug(f"Failed decomposition with freq {freq} for {col}: {str(e)}")
                                    continue
                            
                            if not decomposition_success:
                                logger.warning(f"Could not perform seasonal decomposition for {col} with any frequency")
                                results['seasonal_decomposition'][col] = {
                                    'error': 'Could not perform seasonal decomposition with any frequency',
                                    'attempted_frequencies': ['D', 'W', 'M', 'Q', 'Y']
                                }
                        else:
                            # Use inferred frequency
                            try:
                                freq = df.index.inferred_freq
                                logger.info(f"Using inferred frequency '{freq}' for {col}")
                                decomposition = seasonal_decompose(series.interpolate(), model='additive')
                                
                                results['seasonal_decomposition'][col] = {
                                    'frequency': freq,
                                    'trend': decomposition.trend.dropna().tolist(),
                                    'seasonal': decomposition.seasonal.dropna().tolist(),
                                    'residual': decomposition.resid.dropna().tolist(),
                                    'dates': decomposition.trend.dropna().index.strftime('%Y-%m-%d').tolist()
                                }
                                logger.info(f"Completed seasonal decomposition for {col}")
                            except Exception as e:
                                logger.warning(f"Error in seasonal decomposition with inferred freq for {col}: {str(e)}")
                                results['seasonal_decomposition'][col] = {'error': f"Error in seasonal decomposition: {str(e)}"}
                    except Exception as e:
                        logger.warning(f"Error in seasonal decomposition for column {col}: {str(e)}")
                        results['seasonal_decomposition'][col] = {'error': f"Error in seasonal decomposition: {str(e)}"}
                else:
                    if not isinstance(df.index, pd.DatetimeIndex):
                        logger.warning(f"Cannot perform seasonal decomposition for {col}: No datetime index")
                    else:
                        logger.warning(f"Cannot perform seasonal decomposition for {col}: Insufficient data points")
                    results['seasonal_decomposition'][col] = {
                        'error': 'Cannot perform seasonal decomposition',
                        'reason': 'No datetime index' if not isinstance(df.index, pd.DatetimeIndex) else 'Insufficient data points'
                    }
                
                # Trend analysis
                try:
                    if len(series.dropna()) < 5:
                        logger.warning(f"Column {col} has insufficient data for trend analysis (< 5)")
                        results['trend_analysis'][col] = {'error': 'Not enough data points for trend analysis'}
                        continue
                        
                    # Simple linear trend
                    x = np.arange(len(series.dropna()))
                    y = series.dropna().values
                    
                    # Add constant for statsmodels
                    X = sm.add_constant(x)
                    
                    # Fit linear model
                    model = sm.OLS(y, X).fit()
                    
                    results['trend_analysis'][col] = {
                        'slope': float(model.params[1]),
                        'intercept': float(model.params[0]),
                        'r_squared': float(model.rsquared),
                        'p_value': float(model.pvalues[1]),
                        'significant_trend': bool(model.pvalues[1] < 0.05)
                    }
                    logger.info(f"Completed trend analysis for {col}")
                except Exception as e:
                    logger.warning(f"Error in trend analysis for column {col}: {str(e)}")
                    results['trend_analysis'][col] = {'error': f"Error in trend analysis: {str(e)}"}
            except Exception as e:
                logger.error(f"Error processing column {col}: {str(e)}")
                results['summary_stats'][col] = {'error': f"Error processing column: {str(e)}"}
        
        logger.info("Enhanced EDA analysis completed successfully")
        return results
    
    except Exception as e:
        logger.error(f"Error in perform_enhanced_eda: {str(e)}")
        logger.error(traceback.format_exc())
        return {'error': f"Error in enhanced EDA analysis: {str(e)}"}

def lstm_forecast(series, periods=10):
    """
    Perform LSTM (Long Short-Term Memory) forecasting on a time series
    
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
        # Import required libraries
        import pandas as pd
        import numpy as np
        from sklearn.preprocessing import MinMaxScaler
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, LSTM
        except ImportError:
            # If TensorFlow is not installed, return a helpful error message
            return {
                'error': 'TensorFlow is not installed. Please install it with: pip install tensorflow',
                'forecast': None,
                'model_info': {
                    'name': 'LSTM',
                    'status': 'Not available'
                }
            }
        import logging
        
        logger = logging.getLogger(__name__)
        
        # Check if the series is constant
        if series.std() == 0:
            logger.warning("Series is constant, cannot apply LSTM. Using simple constant forecast.")
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
                    'metrics': {},
                    'description': "Constant forecast (input data has no variation)"
                }
            }
        
        # Check if we have enough data
        if len(series) < 10:
            logger.warning("Not enough data for LSTM forecasting. Need at least 10 data points.")
            return {'error': 'Not enough data for LSTM forecasting. Need at least 10 data points.'}
        
        # Normalize the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(series.values.reshape(-1, 1))
        
        # Create sequences for LSTM
        def create_sequences(data, seq_length):
            X, y = [], []
            for i in range(len(data) - seq_length):
                X.append(data[i:i + seq_length, 0])
                y.append(data[i + seq_length, 0])
            return np.array(X), np.array(y)
        
        # Use a sequence length of 5 or 1/4 of the data, whichever is smaller
        seq_length = min(5, len(scaled_data) // 4)
        if seq_length < 2:
            seq_length = 2  # Minimum sequence length
        
        X, y = create_sequences(scaled_data, seq_length)
        
        # Reshape X to be [samples, time steps, features]
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        # Split into train and test sets
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Build the LSTM model
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, 1)))
        model.add(LSTM(50))
        model.add(Dense(1))
        
        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Fit the model
        model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
        
        # Calculate metrics on test set
        metrics = {}
        if len(X_test) > 0:
            y_pred = model.predict(X_test, verbose=0)
            
            # Inverse transform for actual metrics
            y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
            y_pred_inv = scaler.inverse_transform(y_pred).flatten()
            
            # Calculate metrics
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            
            # Extract actual and predicted values
            y_true = y_test_inv
            y_pred = y_pred_inv
            
            # Calculate metrics
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            
            # Calculate MAPE safely
            if np.all(y_true != 0):
                mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            else:
                # Handle zeros in test data
                non_zero_indices = y_true != 0
                if np.any(non_zero_indices):
                    mape = np.mean(np.abs((y_true[non_zero_indices] - y_pred[non_zero_indices]) / y_true[non_zero_indices])) * 100
                else:
                    mape = None
            
            metrics = {
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae),
                'mape': float(mape) if mape is not None and not np.isinf(mape) else None
            }
        
        # Generate forecast
        # Start with the last sequence from the original data
        curr_seq = scaled_data[-seq_length:].reshape(1, seq_length, 1)
        
        # Generate predictions
        forecast_scaled = []
        for _ in range(periods):
            # Get prediction (next step)
            next_pred = model.predict(curr_seq, verbose=0)[0, 0]
            forecast_scaled.append(next_pred)
            
            # Update sequence
            curr_seq = np.append(curr_seq[:, 1:, :], [[next_pred]], axis=1)
        
        # Convert back to original scale
        forecast_values = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1)).flatten()
        
        # Create confidence intervals (simple approach)
        # Using RMSE from test set to create intervals
        if 'rmse' in metrics:
            rmse_value = metrics['rmse']
            lower_bounds = forecast_values - 1.96 * rmse_value
            upper_bounds = forecast_values + 1.96 * rmse_value
        else:
            # If no test metrics, use a percentage of the forecast
            lower_bounds = forecast_values * 0.9
            upper_bounds = forecast_values * 1.1
        
        # Create future dates
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
        
        # Return results
        return {
            'historical_dates': series.index.tolist(),
            'historical_values': series.values.tolist(),
            'dates': list(future_dates) if isinstance(future_dates, range) else future_dates.tolist(),
            'forecast': forecast_values.tolist(),
            'lower_bounds': lower_bounds.tolist(),
            'upper_bounds': upper_bounds.tolist(),
            'model_info': {
                'name': 'LSTM',
                'sequence_length': seq_length,
                'metrics': metrics,
                'description': "Long Short-Term Memory neural network"
            }
        }
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'error': str(e)}
