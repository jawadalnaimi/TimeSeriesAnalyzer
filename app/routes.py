import os
import uuid
import pandas as pd
from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, current_app, send_from_directory
from werkzeug.utils import secure_filename
from app.analysis import (
    perform_basic_analysis, 
    detect_anomalies, 
    perform_forecasting,
    get_available_models
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
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        # Generate a unique filename
        original_filename = secure_filename(file.filename)
        file_extension = original_filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{uuid.uuid4().hex}.{file_extension}"
        
        # Save the file
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        
        # Store original filename for display purposes
        session_data = {
            'original_filename': original_filename,
            'file_path': file_path,
            'file_id': unique_filename.split('.')[0]
        }
        
        return jsonify({
            'success': True,
            'file_id': session_data['file_id'],
            'filename': session_data['original_filename']
        })
    
    return jsonify({
        'success': False,
        'error': 'Invalid file type'
    })

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
            elif analysis_type == 'anomaly':
                method = request.form.get('method', 'zscore')
                result = detect_anomalies(df, method=method)
            elif analysis_type == 'forecast':
                model = request.form.get('model', 'arima')
                periods = int(request.form.get('periods', 10))
                result = perform_forecasting(df, model=model, periods=periods)
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
