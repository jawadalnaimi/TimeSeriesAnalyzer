"""
Routes for the Time Series Analyzer application.

This module defines the URL routes and request handlers for the application.

Copyright (c) 2025 Jawad Alnaimi
Time Series Analyzer Project
All rights reserved.

This code is the intellectual property of Jawad Alnaimi.
Unauthorized copying, distribution, or use is strictly prohibited.
"""

import os
import uuid
import pandas as pd
import numpy as np
from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, current_app, send_from_directory
from werkzeug.utils import secure_filename
from app.analysis import (
    perform_basic_analysis, 
    detect_anomalies, 
    perform_forecasting,
    get_available_models,
    numpy_to_python,
    safe_json_serialize
)

main_bp = Blueprint('main', __name__)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

@main_bp.route('/')
def index():
    return render_template('index.html')

@main_bp.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    try:
        # Check if the post request has the file part
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # Check file extension
        allowed_extensions = {'csv', 'xls', 'xlsx', 'json'}
        if '.' not in file.filename or \
           file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
            return jsonify({'error': 'Invalid file extension'}), 400
        
        # Generate a unique ID for the file
        import uuid
        file_id = uuid.uuid4().hex[:32]
        
        # Save the file with the ID as prefix
        filename = f"{file_id}_{file.filename}"
        
        # Ensure upload folder exists
        upload_folder = current_app.config['UPLOAD_FOLDER']
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
            current_app.logger.warning(f"Created missing upload folder: {upload_folder}")
            
        file_path = os.path.join(upload_folder, filename)
        file.save(file_path)
        
        current_app.logger.info(f"File uploaded successfully: {file.filename} (ID: {file_id})")
        
        return jsonify({
            'success': True,
            'file_id': file_id,
            'filename': file.filename
        })
    
    except Exception as e:
        current_app.logger.error(f"Error uploading file: {str(e)}")
        return jsonify({'error': f'Error uploading file: {str(e)}'}), 500

@main_bp.route('/analyze/<file_id>', methods=['GET', 'POST'])
def analyze(file_id):
    # Find the file
    for filename in os.listdir(current_app.config['UPLOAD_FOLDER']):
        if filename.startswith(file_id):
            file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            break
    else:
        flash('File not found')
        return redirect(url_for('main.index'))
    
    # Process the file based on its extension
    file_extension = filename.rsplit('.', 1)[1].lower()
    
    try:
        if file_extension == 'csv':
            df = pd.read_csv(file_path)
        elif file_extension in ['xls', 'xlsx']:
            df = pd.read_excel(file_path)
        elif file_extension == 'json':
            df = pd.read_json(file_path)
        else:
            flash('Unsupported file format')
            return redirect(url_for('main.index'))
        
        # Basic analysis
        if request.method == 'POST':
            analysis_type = request.form.get('analysis_type', 'basic')
            
            if analysis_type == 'basic':
                result = perform_basic_analysis(df)
                result = numpy_to_python(result)
            elif analysis_type == 'anomaly':
                method = request.form.get('method', 'zscore')
                result = detect_anomalies(df, method=method)
                result = numpy_to_python(result)
            elif analysis_type == 'forecast':
                model = request.form.get('model', 'arima')
                periods = int(request.form.get('periods', 10))
                try:
                    result = perform_forecasting(df, model=model, periods=periods)
                    result = numpy_to_python(result)
                    if 'error' in result:
                        flash(result['error'])
                        return jsonify(result)
                except Exception as e:
                    return jsonify({'error': f"Forecasting error: {str(e)}"})
            else:
                result = {'error': 'Invalid analysis type'}
            
            return jsonify(result)
        
        # For GET requests, just return the file info and available options
        return render_template('analyze.html', 
                              filename=filename,
                              file_id=file_id,
                              columns=df.columns.tolist(),
                              models=get_available_models())
    
    except Exception as e:
        return jsonify({'error': str(e)})

@main_bp.route('/download/<file_id>/<result_type>', methods=['GET'])
def download_result(file_id, result_type):
    # Check if the processed file exists
    result_filename = f"{file_id}_{result_type}.csv"
    file_path = os.path.join(current_app.config['PROCESSED_FOLDER'], result_filename)
    
    if os.path.exists(file_path):
        return send_from_directory(current_app.config['PROCESSED_FOLDER'], 
                                   result_filename, 
                                   as_attachment=True)
    else:
        flash('Result file not found')
        return redirect(url_for('main.index'))

@main_bp.route('/api/forecast', methods=['POST'])
def api_forecast():
    """API endpoint for forecasting"""
    try:
        current_app.logger.info("Forecast API called")
        
        if not request.json:
            return jsonify({'error': 'No data provided'}), 400
        
        # Import pandas at the beginning of the function to ensure it's available
        import pandas as pd
        import numpy as np
        
        data = request.json
        file_id = data.get('file_id')
        model = data.get('model', 'arima')
        periods = int(data.get('periods', 10))
        
        # Get multiple forecast periods if provided
        forecast_periods = data.get('forecast_periods')
        if forecast_periods:
            try:
                # Convert to list of integers
                forecast_periods = [int(p) for p in forecast_periods]
                # Limit to 3 periods maximum
                forecast_periods = sorted(forecast_periods)[:3]
            except (ValueError, TypeError):
                return jsonify({'error': 'Invalid forecast_periods format. Must be a list of integers.'}), 400
        
        if not file_id:
            return jsonify({'error': 'file_id is required'}), 400
        
        # Find the file
        file_path = None
        upload_folder = current_app.config['UPLOAD_FOLDER']
        
        # Ensure upload folder exists
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
            current_app.logger.warning(f"Created missing upload folder: {upload_folder}")
        
        # Check if the file exists
        for filename in os.listdir(upload_folder):
            if filename.startswith(file_id):
                file_path = os.path.join(upload_folder, filename)
                break
        
        if not file_path:
            return jsonify({'error': f'File not found for ID: {file_id}'}), 404
        
        # Process the file based on its extension
        file_extension = filename.rsplit('.', 1)[1].lower()
        
        try:
            # Use the imported pandas (pd) to read the file
            if file_extension == 'csv':
                df = pd.read_csv(file_path)
            elif file_extension in ['xls', 'xlsx']:
                df = pd.read_excel(file_path)
            elif file_extension == 'json':
                df = pd.read_json(file_path)
            else:
                return jsonify({'error': 'Unsupported file format'}), 400
            
            # Import the analysis module
            from app import analysis
            
            # Perform forecasting
            result = analysis.perform_forecasting(df, model=model, periods=periods, forecast_periods=forecast_periods)
            
            # Return the results
            from app.analysis import safe_json_serialize
            return jsonify(safe_json_serialize({
                'success': True,
                'model': model,
                'periods': periods,
                'forecast_periods': forecast_periods,
                'results': result
            }))
            
        except Exception as e:
            current_app.logger.error(f"Error in forecast API: {str(e)}")
            current_app.logger.error(traceback.format_exc())
            return jsonify({'error': f'Error in forecast API: {str(e)}'}), 500
        
    except Exception as e:
        import traceback
        current_app.logger.error(f"Error in forecast API: {str(e)}")
        current_app.logger.error(traceback.format_exc())
        return jsonify({'error': f'Error in forecast API: {str(e)}'}), 500

@main_bp.route('/api/data/<file_id>', methods=['GET'])
def api_get_data(file_id):
    """API endpoint to get data from an uploaded file"""
    try:
        # Find the file
        for filename in os.listdir(current_app.config['UPLOAD_FOLDER']):
            if filename.startswith(file_id):
                file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
                break
        else:
            return jsonify({'error': 'File not found'}), 404
        
        # Process the file based on its extension
        file_extension = filename.rsplit('.', 1)[1].lower()
        
        if file_extension == 'csv':
            df = pd.read_csv(file_path)
        elif file_extension in ['xls', 'xlsx']:
            df = pd.read_excel(file_path)
        elif file_extension == 'json':
            df = pd.read_json(file_path)
        else:
            return jsonify({'error': 'Unsupported file format'}), 400
        
        # Convert DataFrame to dict
        data_dict = df.to_dict(orient='records')
        
        return jsonify({
            'file_id': file_id,
            'data': data_dict
        })
    
    except Exception as e:
        current_app.logger.error(f"Error getting data: {str(e)}")
        current_app.logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@main_bp.route('/api/test', methods=['GET'])
def api_test():
    """Simple test endpoint that returns a basic JSON response"""
    return jsonify({
        'message': 'Test successful',
        'status': 'ok',
        'data': {
            'number': 42,
            'string': 'hello world',
            'list': [1, 2, 3, 4, 5],
            'boolean': True
        }
    })

@main_bp.route('/test')
def test_page():
    """Test page to isolate the issue"""
    return render_template('test.html')

@main_bp.route('/minimal-forecast')
def minimal_forecast_page():
    """Minimal forecast page to isolate the issue"""
    return render_template('minimal-forecast.html')

@main_bp.route('/api/minimal-forecast', methods=['POST'])
def api_minimal_forecast():
    """API endpoint for minimal forecasting with less data processing"""
    try:
        current_app.logger.info("Minimal forecast API called")
        
        if not request.json:
            return jsonify({'error': 'No data provided'}), 400
        
        # Import pandas at the beginning of the function to ensure it's available
        import pandas as pd
        import numpy as np
        
        data = request.json
        file_id = data.get('file_id')
        time_series = data.get('time_series')
        model = data.get('model', 'arima')
        periods = int(data.get('periods', 10))
        
        # Get multiple forecast periods if provided
        forecast_periods = data.get('forecast_periods')
        if forecast_periods:
            try:
                # Convert to list of integers
                forecast_periods = [int(p) for p in forecast_periods]
                # Limit to 3 periods maximum
                forecast_periods = sorted(forecast_periods)[:3]
            except (ValueError, TypeError):
                return jsonify({'error': 'Invalid forecast_periods format. Must be a list of integers.'}), 400
        
        # Check if we have file_id or time_series
        if not file_id and not time_series:
            return jsonify({'error': 'Either file_id or time_series data is required'}), 400
        
        # If we have a file_id, load the data from the file
        if file_id:
            # Find the file
            file_path = None
            upload_folder = current_app.config['UPLOAD_FOLDER']
            
            # Ensure upload folder exists
            if not os.path.exists(upload_folder):
                os.makedirs(upload_folder)
                current_app.logger.warning(f"Created missing upload folder: {upload_folder}")
            
            # Check if the file exists
            for filename in os.listdir(upload_folder):
                if filename.startswith(file_id):
                    file_path = os.path.join(upload_folder, filename)
                    break
            
            if not file_path:
                return jsonify({'error': f'File not found for ID: {file_id}'}), 404
            
            # Process the file based on its extension
            file_extension = filename.rsplit('.', 1)[1].lower()
            
            try:
                # Use the imported pandas (pd) to read the file
                if file_extension == 'csv':
                    df = pd.read_csv(file_path)
                elif file_extension in ['xls', 'xlsx']:
                    df = pd.read_excel(file_path)
                elif file_extension == 'json':
                    df = pd.read_json(file_path)
                else:
                    return jsonify({'error': 'Unsupported file format'}), 400
                
                # Find datetime column
                datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
                
                # If no datetime column is found, try to convert the first column
                if not datetime_cols and len(df.columns) > 0:
                    try:
                        df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])
                        datetime_cols = [df.columns[0]]
                    except Exception as e:
                        current_app.logger.error(f"Error converting to datetime: {str(e)}")
                        return jsonify({'error': f'Could not find or convert date column: {str(e)}'}), 400
                
                if not datetime_cols:
                    return jsonify({'error': 'No date column found in the file'}), 400
                
                # Get the first numeric column
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                if not numeric_cols:
                    return jsonify({'error': 'No numeric columns found in the file'}), 400
                
                # Create a time series from the first datetime column and first numeric column
                series = pd.Series(df[numeric_cols[0]].values, index=df[datetime_cols[0]])
                
            except Exception as e:
                current_app.logger.error(f"Error reading file: {str(e)}")
                return jsonify({'error': f'Error reading file: {str(e)}'}), 500
        
        # If we have time_series data directly
        elif time_series:
            try:
                # Convert to pandas Series
                dates = [pd.to_datetime(item['date']) for item in time_series]
                values = [float(item['value']) for item in time_series]
                series = pd.Series(values, index=dates)
            except Exception as e:
                current_app.logger.error(f"Error processing time series data: {str(e)}")
                return jsonify({'error': f'Error processing time series data: {str(e)}'}), 400
        
        # Sort by date
        series = series.sort_index()
        
        # Check if we have enough data
        if len(series) < 5:
            return jsonify({'error': 'Not enough data points for forecasting (minimum 5 required)'}), 400
        
        # Import the analysis module
        from app import analysis
        
        # Get available models
        available_models = analysis.get_available_models()
        available_model_ids = [m['id'] for m in available_models]
        
        if model not in available_model_ids:
            return jsonify({'error': f'Model {model} not available. Available models: {", ".join(available_model_ids)}'}), 400
        
        # Perform forecasting based on the selected model
        if model == 'arima':
            result = analysis.arima_forecast(series, periods)
        elif model == 'sarima':
            result = analysis.sarima_forecast(series, periods)
        elif model == 'exponential_smoothing':
            result = analysis.exponential_smoothing_forecast(series, periods)
        elif model == 'prophet':
            result = analysis.prophet_forecast(series, periods)
        elif model == 'lstm':
            result = analysis.lstm_forecast(series, periods)
        else:
            # For multiple forecast periods, use the perform_forecasting function
            result = analysis.perform_forecasting(pd.DataFrame({'value': series}), model=model, periods=periods, forecast_periods=forecast_periods)
        
        # Return the results
        from app.analysis import safe_json_serialize
        return jsonify(safe_json_serialize({
            'success': True,
            'model': model,
            'periods': periods,
            'forecast_periods': forecast_periods if forecast_periods else [periods],
            'results': result
        }))
        
    except Exception as e:
        import traceback
        current_app.logger.error(f"Error in minimal forecast API: {str(e)}")
        current_app.logger.error(traceback.format_exc())
        return jsonify({
            'error': True,
            'error_message': f'Error in minimal forecast API: {str(e)}'
        }), 500

@main_bp.route('/api/enhanced-eda', methods=['POST'])
def api_enhanced_eda():
    """API endpoint for enhanced exploratory data analysis"""
    try:
        current_app.logger.info("Enhanced EDA API called")
        
        if not request.json:
            return jsonify({'error': 'No data provided'}), 400
        
        # Import pandas at the beginning of the function to ensure it's available
        import pandas as pd
        import numpy as np
        import json
        import traceback
        
        data = request.json
        file_id = data.get('file_id')
        
        if not file_id:
            return jsonify({'error': 'file_id is required'}), 400
        
        # Find the file
        file_path = None
        upload_folder = current_app.config['UPLOAD_FOLDER']
        
        # Ensure upload folder exists
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
            current_app.logger.warning(f"Created missing upload folder: {upload_folder}")
        
        # Check if the file exists
        for filename in os.listdir(upload_folder):
            if filename.startswith(file_id):
                file_path = os.path.join(upload_folder, filename)
                current_app.logger.info(f"Found file for ID {file_id}: {filename}")
                break
        
        if not file_path:
            # For testing purposes, if file not found, use test data
            current_app.logger.warning(f"File not found for ID: {file_id}, using test data")
            
            # Generate synthetic data
            import numpy as np
            import pandas as pd
            
            # Create synthetic time series with multiple patterns
            np.random.seed(42)
            dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
            
            # Create multiple columns with different patterns
            trend = np.linspace(0, 10, 100)
            seasonal = 5 * np.sin(np.linspace(0, 4*np.pi, 100))
            noise = np.random.normal(0, 1, 100)
            
            values1 = trend + seasonal + noise
            values2 = trend * 2 + seasonal * 0.5 + noise * 1.5
            
            # Create a pandas DataFrame
            df = pd.DataFrame({
                'date': dates,
                'value': values1,
                'value2': values2
            })
            
            # Set date as index
            df = df.set_index('date')
            
            current_app.logger.info(f"Created synthetic test data with shape: {df.shape}")
        else:
            # Process the file based on its extension
            file_extension = filename.rsplit('.', 1)[1].lower()
            
            try:
                # Use the imported pandas (pd) to read the file
                if file_extension == 'csv':
                    df = pd.read_csv(file_path)
                    current_app.logger.info(f"Read CSV file with shape: {df.shape}")
                elif file_extension in ['xls', 'xlsx']:
                    df = pd.read_excel(file_path)
                    current_app.logger.info(f"Read Excel file with shape: {df.shape}")
                elif file_extension == 'json':
                    df = pd.read_json(file_path)
                    current_app.logger.info(f"Read JSON file with shape: {df.shape}")
                else:
                    return jsonify({'error': 'Unsupported file format'}), 400
                
                # Check if dataframe is empty
                if df.empty:
                    return jsonify({'error': 'File contains no data'}), 400
                
                # Check for datetime columns
                datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
                
                # If no datetime column is found, try to convert the first column
                if not datetime_cols and len(df.columns) > 0:
                    try:
                        df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])
                        datetime_cols = [df.columns[0]]
                        current_app.logger.info(f"Converted column {df.columns[0]} to datetime")
                    except Exception as e:
                        current_app.logger.warning(f"Could not convert column to datetime: {str(e)}")
                
                # Set the first datetime column as index if available
                if datetime_cols:
                    df = df.set_index(datetime_cols[0])
                    current_app.logger.info(f"Set index to datetime column: {datetime_cols[0]}")
                
            except Exception as e:
                current_app.logger.error(f"Error reading file: {str(e)}")
                current_app.logger.error(traceback.format_exc())
                return jsonify({
                    'error': True,
                    'error_type': 'data_processing_error',
                    'error_title': 'File Processing Error',
                    'error_message': f'Error reading file: {str(e)}',
                    'error_details': traceback.format_exc(),
                    'error_suggestions': [
                        'Check if the file is a valid data file',
                        'Ensure the file contains properly formatted data',
                        'Try uploading the file again'
                    ],
                    'error_icon': 'file-excel'
                }), 500
        
        # Import the analysis module
        from app import analysis
        
        # Perform enhanced EDA
        try:
            current_app.logger.info("Starting enhanced EDA analysis")
            result = analysis.perform_enhanced_eda(df)
            current_app.logger.info("Completed enhanced EDA analysis")
            
            # Check if result contains an error
            if isinstance(result, dict) and 'error' in result:
                current_app.logger.error(f"Error in EDA analysis: {result['error']}")
                return jsonify({
                    'error': True,
                    'error_type': 'analysis_error',
                    'error_title': 'Analysis Error',
                    'error_message': result['error'],
                    'error_details': 'The analysis module encountered an error processing your data',
                    'error_suggestions': [
                        'Check if your data contains valid time series',
                        'Ensure your data has a proper date/time column',
                        'Try with a different dataset or model'
                    ],
                    'error_icon': 'chart-line'
                }), 400
            
            # Log the structure of the result for debugging
            current_app.logger.info(f"Result keys: {list(result.keys())}")
            for key in result.keys():
                if isinstance(result[key], dict):
                    current_app.logger.info(f"{key} contains: {list(result[key].keys())}")
            
            # Serialize the result step by step with detailed logging
            try:
                current_app.logger.info("Serializing EDA results")
                
                # First, serialize the summary stats section
                if 'summary_stats' in result:
                    current_app.logger.info("Serializing summary_stats section")
                    summary_stats = analysis.safe_json_serialize(result['summary_stats'])
                    current_app.logger.info("Successfully serialized summary_stats section")
                else:
                    summary_stats = {}
                
                # Next, serialize the stationarity tests section
                if 'stationarity_tests' in result:
                    current_app.logger.info("Serializing stationarity_tests section")
                    stationarity_tests = analysis.safe_json_serialize(result['stationarity_tests'])
                    current_app.logger.info("Successfully serialized stationarity_tests section")
                else:
                    stationarity_tests = {}
                
                # Next, serialize the autocorrelation section
                if 'autocorrelation' in result:
                    current_app.logger.info("Serializing autocorrelation section")
                    autocorrelation = analysis.safe_json_serialize(result['autocorrelation'])
                    current_app.logger.info("Successfully serialized autocorrelation section")
                else:
                    autocorrelation = {}
                
                # Next, serialize the distribution analysis section
                if 'distribution_analysis' in result:
                    current_app.logger.info("Serializing distribution_analysis section")
                    distribution_analysis = analysis.safe_json_serialize(result['distribution_analysis'])
                    current_app.logger.info("Successfully serialized distribution_analysis section")
                else:
                    distribution_analysis = {}
                
                # Next, serialize the seasonal decomposition section
                if 'seasonal_decomposition' in result:
                    current_app.logger.info("Serializing seasonal_decomposition section")
                    seasonal_decomposition = analysis.safe_json_serialize(result['seasonal_decomposition'])
                    current_app.logger.info("Successfully serialized seasonal_decomposition section")
                else:
                    seasonal_decomposition = {}
                
                # Finally, serialize the trend analysis section
                if 'trend_analysis' in result:
                    current_app.logger.info("Serializing trend_analysis section")
                    trend_analysis = analysis.safe_json_serialize(result['trend_analysis'])
                    current_app.logger.info("Successfully serialized trend_analysis section")
                else:
                    trend_analysis = {}
                
                # Combine all serialized sections
                serialized_result = {
                    'success': True,
                    'results': {
                        'summary_stats': summary_stats,
                        'stationarity_tests': stationarity_tests,
                        'autocorrelation': autocorrelation,
                        'distribution_analysis': distribution_analysis,
                        'seasonal_decomposition': seasonal_decomposition,
                        'trend_analysis': trend_analysis
                    }
                }
                
                # Try to validate the serialized result
                try:
                    current_app.logger.info("Validating final JSON result")
                    
                    # Use a custom JSON encoder to handle any NaN or Infinity values
                    class CustomJSONEncoder(json.JSONEncoder):
                        def default(self, obj):
                            import math
                            if isinstance(obj, float):
                                if math.isnan(obj) or math.isinf(obj):
                                    return None
                            return super().default(obj)
                    
                    json_str = json.dumps(serialized_result, cls=CustomJSONEncoder)
                    current_app.logger.info(f"Serialized result length: {len(json_str)} characters")
                    # Return the validated result
                    return json_str, 200, {'Content-Type': 'application/json'}
                except Exception as e:
                    current_app.logger.error(f"JSON dumps validation error: {str(e)}")
                    current_app.logger.error(traceback.format_exc())
                    return jsonify({
                        'error': True,
                        'error_type': 'json_validation_error',
                        'error_title': 'JSON Validation Error',
                        'error_message': f"JSON validation error: {str(e)}",
                        'error_details': traceback.format_exc(),
                        'error_suggestions': [
                            'Try with a simpler dataset',
                            'The analysis results contain data that cannot be converted to JSON'
                        ],
                        'error_icon': 'exclamation-circle'
                    }), 500
            except Exception as e:
                current_app.logger.error(f"Serialization error: {str(e)}")
                current_app.logger.error(traceback.format_exc())
                return jsonify({
                    'error': True,
                    'error_type': 'serialization_error',
                    'error_title': 'Data Serialization Error',
                    'error_message': f"Could not serialize results: {str(e)}",
                    'error_details': traceback.format_exc(),
                    'error_suggestions': [
                        'Try with a simpler dataset',
                        'Check if your data contains complex objects that cannot be serialized'
                    ],
                    'error_icon': 'exclamation-circle'
                }), 500
            
        except Exception as e:
            current_app.logger.error(f"Error in EDA analysis: {str(e)}")
            current_app.logger.error(traceback.format_exc())
            return jsonify({
                'error': True,
                'error_type': 'analysis_error',
                'error_title': 'Analysis Error',
                'error_message': f'Error in EDA analysis: {str(e)}',
                'error_details': traceback.format_exc(),
                'error_suggestions': [
                    'Check if your data is suitable for time series analysis',
                    'Ensure your data has a proper date/time column',
                    'Try with a different dataset or model'
                ],
                'error_icon': 'chart-line'
            }), 500
        
    except Exception as e:
        import traceback
        current_app.logger.error(f"Error in enhanced EDA API: {str(e)}")
        current_app.logger.error(traceback.format_exc())
        return jsonify({
            'error': True,
            'error_type': 'system_error',
            'error_title': 'System Error',
            'error_message': f'Error in enhanced EDA API: {str(e)}',
            'error_details': traceback.format_exc(),
            'error_suggestions': [
                'Try again later',
                'Contact system administrator if the problem persists'
            ],
            'error_icon': 'exclamation-triangle'
        }), 500

@main_bp.route('/api/debug-test', methods=['GET', 'POST'])
def api_debug_test():
    """Debug endpoint to test JSON serialization and error handling"""
    import numpy as np
    import pandas as pd
    from datetime import datetime, timedelta
    import logging
    import traceback
    from app.error_handling import format_error_response, json_error_response
    from app.analysis import safe_json_serialize
    
    try:
        # Check if this is a test error request
        if request.method == 'POST' and request.json and request.json.get('test_error'):
            error_type = request.json.get('error_type', 'system_error')
            current_app.logger.info(f"Testing error handling with error type: {error_type}")
            
            # Generate different types of errors based on the requested type
            if error_type == 'validation_error':
                error_data = format_error_response(
                    "Invalid parameter values", 
                    error_type="validation_error",
                    details="The parameters provided do not meet the required validation rules.",
                    suggestions=[
                        "Check that all required parameters are provided",
                        "Ensure parameter values are within acceptable ranges",
                        "Verify data types match the expected format"
                    ]
                )
                return jsonify(error_data), 400
                
            elif error_type == 'data_error':
                error_data = format_error_response(
                    "Data processing error", 
                    error_type="data_error",
                    details="The system encountered an error while processing your data.",
                    suggestions=[
                        "Check that your data contains the required columns",
                        "Ensure your data does not contain invalid values",
                        "Verify your data is in the correct format (e.g., dates are valid)"
                    ]
                )
                return jsonify(error_data), 422
                
            elif error_type == 'model_error':
                error_data = format_error_response(
                    "Model fitting error", 
                    error_type="model_error",
                    details="The forecasting model could not be fitted to your data.",
                    suggestions=[
                        "Try a different forecasting model",
                        "Check if your data has sufficient history for forecasting",
                        "Ensure your time series data is properly formatted",
                        "Consider preprocessing your data to handle outliers or missing values"
                    ]
                )
                return jsonify(error_data), 500
                
            elif error_type == 'system_error':
                error_data = format_error_response(
                    "System processing error", 
                    error_type="system_error",
                    details="An unexpected system error occurred while processing your request.",
                    suggestions=[
                        "Try again later",
                        "Contact support if the problem persists",
                        "Check system status for any ongoing issues"
                    ]
                )
                return jsonify(error_data), 500
                
            elif error_type == 'dependency_error':
                error_data = format_error_response(
                    "Missing dependency", 
                    error_type="dependency_error",
                    details="The requested operation requires libraries that are not installed.",
                    suggestions=[
                        "Install the required dependencies",
                        "Check the documentation for installation instructions",
                        "Use a different feature that doesn't require the missing dependencies",
                        "Contact your system administrator to install the required packages"
                    ]
                )
                return jsonify(error_data), 503
                
            else:
                # Default to system error if unknown error type
                error_data = format_error_response(
                    f"Unknown error type: {error_type}", 
                    error_type="system_error"
                )
                return jsonify(error_data), 400
        
        # Check if this is a dependency test
        if request.method == 'POST' and request.json and request.json.get('test_dependency') and request.json.get('simulate_import'):
            dependency_name = request.json.get('test_dependency')
            current_app.logger.info(f"Testing dependency import for: {dependency_name}")
            
            try:
                # Try to import the dependency
                __import__(dependency_name)
                return jsonify({
                    "success": True,
                    "message": f"Successfully imported {dependency_name}"
                })
            except ImportError as e:
                # Return a dependency error
                error_data = format_error_response(
                    f"Missing dependency: {dependency_name}", 
                    error_type="dependency_error",
                    details=f"The operation requires {dependency_name} which is not installed: {str(e)}",
                    suggestions=[
                        f"Install {dependency_name} using pip: pip install {dependency_name}",
                        "Check the documentation for installation instructions",
                        "Use a different feature that doesn't require this dependency"
                    ]
                )
                return jsonify(error_data), 503
        
        # Create test data with various types
        current_app.logger.info("Creating test data for debug endpoint")
        
        # Create a date range
        dates = pd.date_range(start=datetime.now(), periods=5, freq='D')
        
        # Create a DataFrame with various data types
        df = pd.DataFrame({
            'date': dates,
            'int_values': np.array([1, 2, 3, 4, 5], dtype=np.int64),
            'float_values': np.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype=np.float64),
            'string_values': ['a', 'b', 'c', 'd', 'e'],
            'bool_values': [True, False, True, False, True],
            'complex_values': [complex(1, 2), complex(2, 3), complex(3, 4), complex(4, 5), complex(5, 6)],
            'nan_values': [np.nan, 1.0, 2.0, 3.0, np.nan],
            'inf_values': [np.inf, 1.0, 2.0, 3.0, -np.inf]
        })
        
        # Set the date column as index
        df = df.set_index('date')
        
        # Create a dictionary with various nested structures
        test_data = {
            'scalar_values': {
                'int': np.int64(42),
                'float': np.float64(3.14159),
                'string': 'test',
                'bool': np.bool_(True),
                'none': None,
                'nan': np.nan,
                'inf': np.inf,
                'neg_inf': -np.inf
            },
            'array_values': {
                'int_array': np.array([1, 2, 3, 4, 5], dtype=np.int64),
                'float_array': np.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype=np.float64),
                'mixed_array': [1, 'two', 3.0, True, None]
            },
            'dataframe': df.reset_index().to_dict(orient='records'),
            'series': df['int_values'].to_dict(),
            'timestamp': pd.Timestamp('2025-01-01'),
            'timedelta': pd.Timedelta(days=1),
            'datetime': datetime.now(),
            'date': datetime.now().date(),
        }
        
        # If POST request, use the provided data
        if request.method == 'POST' and request.json:
            current_app.logger.info("Using provided data for debug test")
            user_data = request.json
            
            # Merge with some test data to ensure we test serialization
            test_data = {
                'user_data': user_data,
                'test_data': test_data
            }
        
        # Safely serialize the data
        current_app.logger.info("Serializing test data")
        safe_data = safe_json_serialize(test_data)
        
        # Try to serialize to JSON string first to catch any issues
        import json
        try:
            json_str = json.dumps(safe_data)
            current_app.logger.info(f"Successfully serialized debug test data (length: {len(json_str)})")
        except Exception as e:
            current_app.logger.error(f"Error serializing debug test data: {str(e)}")
            error_data = format_error_response(
                f"Error serializing results: {str(e)}", 
                error_type="system_error"
            )
            return jsonify(error_data), 500
        
        # Return the results
        current_app.logger.info("Returning debug test results")
        return jsonify(safe_data)
    
    except Exception as e:
        current_app.logger.error(f"Error in debug test API: {str(e)}")
        current_app.logger.error(traceback.format_exc())
        error_data = format_error_response(e)
        return jsonify(error_data), 500

@main_bp.route('/api/debug-eda', methods=['POST'])
def api_debug_eda():
    """Debug endpoint for enhanced exploratory data analysis"""
    try:
        current_app.logger.info("Debug EDA API called")
        
        # Import pandas at the beginning of the function to ensure it's available
        import pandas as pd
        import numpy as np
        import json
        import traceback
        import math
        
        # Generate synthetic data
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        
        # Create multiple columns with different patterns
        trend = np.linspace(0, 10, 100)
        seasonal = 5 * np.sin(np.linspace(0, 4*np.pi, 100))
        noise = np.random.normal(0, 1, 100)
        
        values1 = trend + seasonal + noise
        values2 = trend * 2 + seasonal * 0.5 + noise * 1.5
        
        # Add some NaN values to test handling
        values1[10:15] = np.nan
        values2[80:85] = np.nan
        
        # Create a pandas DataFrame
        df = pd.DataFrame({
            'date': dates,
            'value': values1,
            'value2': values2
        })
        
        # Set date as index
        df = df.set_index('date')
        
        current_app.logger.info(f"Created synthetic test data with shape: {df.shape}")
        
        # Import the analysis module
        from app import analysis
        
        # Perform enhanced EDA
        try:
            current_app.logger.info("Starting enhanced EDA analysis")
            result = analysis.perform_enhanced_eda(df)
            current_app.logger.info("Completed enhanced EDA analysis")
            
            # Check if result contains an error
            if isinstance(result, dict) and 'error' in result:
                current_app.logger.error(f"Error in EDA analysis: {result['error']}")
                return jsonify({
                    'error': True,
                    'error_message': result['error']
                }), 400
            
            # Log the structure of the result for debugging
            current_app.logger.info(f"Result keys: {list(result.keys())}")
            
            # Serialize the result step by step with detailed logging
            try:
                current_app.logger.info("Serializing EDA results")
                
                # First, serialize the summary stats section
                if 'summary_stats' in result:
                    current_app.logger.info("Serializing summary_stats section")
                    summary_stats = analysis.safe_json_serialize(result['summary_stats'])
                    current_app.logger.info("Successfully serialized summary_stats section")
                else:
                    summary_stats = {}
                
                # Next, serialize the stationarity tests section
                if 'stationarity_tests' in result:
                    current_app.logger.info("Serializing stationarity_tests section")
                    stationarity_tests = analysis.safe_json_serialize(result['stationarity_tests'])
                    current_app.logger.info("Successfully serialized stationarity_tests section")
                else:
                    stationarity_tests = {}
                
                # Next, serialize the autocorrelation section
                if 'autocorrelation' in result:
                    current_app.logger.info("Serializing autocorrelation section")
                    autocorrelation = analysis.safe_json_serialize(result['autocorrelation'])
                    current_app.logger.info("Successfully serialized autocorrelation section")
                else:
                    autocorrelation = {}
                
                # Next, serialize the distribution analysis section
                if 'distribution_analysis' in result:
                    current_app.logger.info("Serializing distribution_analysis section")
                    distribution_analysis = analysis.safe_json_serialize(result['distribution_analysis'])
                    current_app.logger.info("Successfully serialized distribution_analysis section")
                else:
                    distribution_analysis = {}
                
                # Next, serialize the seasonal decomposition section
                if 'seasonal_decomposition' in result:
                    current_app.logger.info("Serializing seasonal_decomposition section")
                    seasonal_decomposition = analysis.safe_json_serialize(result['seasonal_decomposition'])
                    current_app.logger.info("Successfully serialized seasonal_decomposition section")
                else:
                    seasonal_decomposition = {}
                
                # Finally, serialize the trend analysis section
                if 'trend_analysis' in result:
                    current_app.logger.info("Serializing trend_analysis section")
                    trend_analysis = analysis.safe_json_serialize(result['trend_analysis'])
                    current_app.logger.info("Successfully serialized trend_analysis section")
                else:
                    trend_analysis = {}
                
                # Combine all serialized sections
                serialized_result = {
                    'success': True,
                    'results': {
                        'summary_stats': summary_stats,
                        'stationarity_tests': stationarity_tests,
                        'autocorrelation': autocorrelation,
                        'distribution_analysis': distribution_analysis,
                        'seasonal_decomposition': seasonal_decomposition,
                        'trend_analysis': trend_analysis
                    }
                }
                
                # Try to validate the serialized result
                try:
                    current_app.logger.info("Validating final JSON result")
                    
                    # Use a custom JSON encoder to handle any NaN or Infinity values
                    class CustomJSONEncoder(json.JSONEncoder):
                        def default(self, obj):
                            if isinstance(obj, float):
                                if math.isnan(obj) or math.isinf(obj):
                                    return None
                            return super().default(obj)
                    
                    json_str = json.dumps(serialized_result, cls=CustomJSONEncoder)
                    current_app.logger.info(f"Serialized result length: {len(json_str)} characters")
                    
                    # Return the validated result
                    return json_str, 200, {'Content-Type': 'application/json'}
                except Exception as e:
                    current_app.logger.error(f"JSON dumps validation error: {str(e)}")
                    current_app.logger.error(traceback.format_exc())
                    return jsonify({
                        'error': True,
                        'error_message': f"JSON validation error: {str(e)}"
                    }), 500
            except Exception as e:
                current_app.logger.error(f"Serialization error: {str(e)}")
                current_app.logger.error(traceback.format_exc())
                return jsonify({
                    'error': True,
                    'error_message': f"Could not serialize results: {str(e)}"
                }), 500
            
        except Exception as e:
            current_app.logger.error(f"Error in EDA analysis: {str(e)}")
            current_app.logger.error(traceback.format_exc())
            return jsonify({
                'error': True,
                'error_message': f'Error in EDA analysis: {str(e)}'
            }), 500
        
    except Exception as e:
        import traceback
        current_app.logger.error(f"Error in debug EDA API: {str(e)}")
        current_app.logger.error(traceback.format_exc())
        return jsonify({
            'error': True,
            'error_message': f'Error in debug EDA API: {str(e)}'
        }), 500

@main_bp.route('/api/debug-json', methods=['GET'])
def api_debug_json():
    """Debug endpoint to test JSON serialization with various data types"""
    try:
        import pandas as pd
        import numpy as np
        import json
        import math
        from app.analysis import safe_json_serialize
        import logging
        
        logger = logging.getLogger(__name__)
        logger.info("Debug JSON endpoint called")
        
        # Create a dictionary with various data types including NaN values
        test_data = {
            'int_value': 42,
            'float_value': 3.14159,
            'str_value': 'Hello, world!',
            'bool_value': True,
            'none_value': None,
            'nan_value': float('nan'),
            'inf_value': float('inf'),
            'neg_inf_value': float('-inf'),
            'list_with_nan': [1, 2, float('nan'), 4, 5],
            'dict_with_nan': {
                'a': 1,
                'b': float('nan'),
                'c': 3
            },
            'numpy_array': np.array([1, 2, 3, 4, 5]),
            'numpy_array_with_nan': np.array([1, 2, np.nan, 4, 5]),
            'pandas_series': pd.Series([1, 2, 3, 4, 5]),
            'pandas_series_with_nan': pd.Series([1, 2, np.nan, 4, 5]),
            'nested_dict': {
                'a': {
                    'b': {
                        'c': float('nan')
                    }
                }
            }
        }
        
        logger.info("Created test data with NaN values")
        
        # Serialize the test data
        serialized_data = safe_json_serialize(test_data)
        logger.info("Serialized test data with safe_json_serialize")
        
        # Use a custom JSON encoder to handle any remaining NaN or Infinity values
        class CustomJSONEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, float):
                    if math.isnan(obj) or math.isinf(obj):
                        return None
                return super().default(obj)
        
        # Convert to JSON string with the custom encoder
        json_str = json.dumps(serialized_data, cls=CustomJSONEncoder)
        logger.info("Converted to JSON string with custom encoder")
        
        # Return the JSON string
        return json_str, 200, {'Content-Type': 'application/json'}
    
    except Exception as e:
        import traceback
        current_app.logger.error(f"Error in debug JSON API: {str(e)}")
        current_app.logger.error(traceback.format_exc())
        return jsonify({
            'error': True,
            'error_message': f'Error in debug JSON API: {str(e)}'
        }), 500

@main_bp.route('/debug-test')
def debug_test_page():
    """Debug test page for JSON serialization"""
    return render_template('debug-test.html')

@main_bp.route('/minimal-eda-test')
def minimal_eda_test_page():
    """Minimal page to test enhanced EDA functionality"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Minimal EDA Test</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css">
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js"></script>
    </head>
    <body>
        <div class="container mt-5">
            <h1>Minimal Enhanced EDA Test</h1>
            <div class="row">
                <div class="col-md-12">
                    <div class="card">
                        <div class="card-header">
                            <h5>Test Enhanced EDA</h5>
                        </div>
                        <div class="card-body">
                            <div class="mb-3">
                                <label for="file-id" class="form-label">File ID:</label>
                                <input type="text" class="form-control" id="file-id" value="test">
                            </div>
                            <button id="test-eda-btn" class="btn btn-primary">Test Enhanced EDA</button>
                            <div id="result" class="mt-3"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            document.getElementById('test-eda-btn').addEventListener('click', function() {
                const fileId = document.getElementById('file-id').value;
                const resultDiv = document.getElementById('result');
                
                resultDiv.innerHTML = '<div class="alert alert-info">Testing enhanced EDA... Please wait.</div>';
                
                console.log("Calling enhanced EDA API with file_id:", fileId);
                
                fetch('/api/enhanced-eda', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        file_id: fileId
                    })
                })
                .then(response => {
                    console.log("EDA API response status:", response.status);
                    if (!response.ok) {
                        throw new Error(`HTTP error! Status: ${response.status}`);
                    }
                    return response.text().then(text => {
                        console.log("Raw API response:", text);
                        try {
                            return JSON.parse(text);
                        } catch (e) {
                            console.error("JSON parse error:", e);
                            throw new Error(`Failed to parse JSON response: ${e.message}`);
                        }
                    });
                })
                .then(data => {
                    console.log("EDA API parsed response:", data);
                    if (data.error === true || (typeof data.error === 'string' && data.error)) {
                        console.error("EDA error:", data.error);
                        throw new Error(typeof data.error === 'string' ? data.error : JSON.stringify(data.error));
                    }
                    
                    // Display success message
                    resultDiv.innerHTML = '<div class="alert alert-success">Enhanced EDA completed successfully!</div>';
                    
                    // Add a pre element with the JSON response
                    const pre = document.createElement('pre');
                    pre.style.maxHeight = '400px';
                    pre.style.overflow = 'auto';
                    pre.textContent = JSON.stringify(data, null, 2);
                    resultDiv.appendChild(pre);
                })
                .catch(error => {
                    console.error("Enhanced EDA error:", error);
                    resultDiv.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
                });
            });
        </script>
    </body>
    </html>
    '''
