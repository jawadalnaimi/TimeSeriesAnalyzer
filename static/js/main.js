// Main JavaScript for Time Series Analyzer

document.addEventListener('DOMContentLoaded', function() {
    // File upload form handling
    const uploadForm = document.getElementById('upload-form');
    if (uploadForm) {
        uploadForm.addEventListener('submit', handleFileUpload);
    }
    
    // Analysis form handling
    const basicAnalysisForm = document.getElementById('basic-analysis-form');
    if (basicAnalysisForm) {
        basicAnalysisForm.addEventListener('submit', handleBasicAnalysis);
    }
    
    const anomalyDetectionForm = document.getElementById('anomaly-detection-form');
    if (anomalyDetectionForm) {
        anomalyDetectionForm.addEventListener('submit', handleAnomalyDetection);
    }
    
    const forecastingForm = document.getElementById('forecasting-form');
    if (forecastingForm) {
        forecastingForm.addEventListener('submit', handleForecasting);
    }
    
    // Column selection
    const columnItems = document.querySelectorAll('.column-item');
    if (columnItems.length > 0) {
        columnItems.forEach(item => {
            item.addEventListener('click', function() {
                this.classList.toggle('active');
            });
        });
    }
});

/**
 * Handle file upload
 * @param {Event} event - Form submit event
 */
function handleFileUpload(event) {
    event.preventDefault();
    
    const fileInput = document.getElementById('file');
    const file = fileInput.files[0];
    
    if (!file) {
        showUploadStatus('Please select a file to upload.', 'danger');
        return;
    }
    
    // Check file extension
    const fileName = file.name;
    const fileExt = fileName.split('.').pop().toLowerCase();
    const allowedExts = ['csv', 'xls', 'xlsx', 'json', 'txt'];
    
    if (!allowedExts.includes(fileExt)) {
        showUploadStatus('Invalid file type. Please upload a CSV, Excel, or JSON file.', 'danger');
        return;
    }
    
    // Create FormData object
    const formData = new FormData();
    formData.append('file', file);
    
    // Show loading status
    showUploadStatus('<div class="loading"></div> Uploading file...', 'info');
    
    // Send AJAX request
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showUploadStatus('File uploaded successfully!', 'success');
            // Redirect to analysis page
            window.location.href = `/analyze/${data.file_id}`;
        } else {
            showUploadStatus(`Error: ${data.error}`, 'danger');
        }
    })
    .catch(error => {
        showUploadStatus(`Error: ${error.message}`, 'danger');
    });
}

/**
 * Display upload status message
 * @param {string} message - Status message
 * @param {string} type - Message type (success, danger, info)
 */
function showUploadStatus(message, type) {
    const statusDiv = document.getElementById('upload-status');
    statusDiv.innerHTML = `<div class="alert alert-${type}">${message}</div>`;
    statusDiv.style.display = 'block';
}

/**
 * Handle basic analysis form submission
 * @param {Event} event - Form submit event
 */
function handleBasicAnalysis(event) {
    event.preventDefault();
    
    const form = event.target;
    const formData = new FormData(form);
    const fileId = formData.get('file_id');
    
    // Get selected columns
    const selectedColumns = getSelectedColumns();
    if (selectedColumns.length > 0) {
        formData.append('columns', JSON.stringify(selectedColumns));
    }
    
    // Show loading
    const resultsDiv = document.getElementById('basic-results');
    resultsDiv.innerHTML = '<div class="text-center"><div class="loading"></div> Analyzing data...</div>';
    
    // Send AJAX request
    fetch(`/analyze/${fileId}`, {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        displayBasicAnalysisResults(data, resultsDiv);
    })
    .catch(error => {
        resultsDiv.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
    });
}

/**
 * Handle anomaly detection form submission
 * @param {Event} event - Form submit event
 */
function handleAnomalyDetection(event) {
    event.preventDefault();
    
    const form = event.target;
    const formData = new FormData(form);
    const fileId = formData.get('file_id');
    
    // Get selected columns
    const selectedColumns = getSelectedColumns();
    if (selectedColumns.length > 0) {
        formData.append('columns', JSON.stringify(selectedColumns));
    }
    
    // Show loading
    const resultsDiv = document.getElementById('anomaly-results');
    resultsDiv.innerHTML = '<div class="text-center"><div class="loading"></div> Detecting anomalies...</div>';
    
    // Send AJAX request
    fetch(`/analyze/${fileId}`, {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        displayAnomalyResults(data, resultsDiv);
    })
    .catch(error => {
        resultsDiv.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
    });
}

/**
 * Handle forecasting form submission
 * @param {Event} event - Form submit event
 */
function handleForecasting(event) {
    event.preventDefault();
    
    const form = event.target;
    const formData = new FormData(form);
    const fileId = formData.get('file_id');
    
    // Get selected columns
    const selectedColumns = getSelectedColumns();
    if (selectedColumns.length > 0) {
        formData.append('columns', JSON.stringify(selectedColumns));
    }
    
    // Show loading
    const resultsDiv = document.getElementById('forecast-results');
    resultsDiv.innerHTML = '<div class="text-center"><div class="loading"></div> Generating forecast...</div>';
    
    // Send AJAX request
    fetch(`/analyze/${fileId}`, {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        displayForecastResults(data, resultsDiv);
    })
    .catch(error => {
        resultsDiv.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
    });
}

/**
 * Get selected columns from the column list
 * @returns {Array} Array of selected column names
 */
function getSelectedColumns() {
    const selectedColumns = [];
    const columnItems = document.querySelectorAll('.column-item.active');
    
    columnItems.forEach(item => {
        selectedColumns.push(item.dataset.column);
    });
    
    return selectedColumns;
}

/**
 * Display basic analysis results
 * @param {Object} data - Analysis results data
 * @param {HTMLElement} container - Container element for results
 */
function displayBasicAnalysisResults(data, container) {
    if (data.error) {
        container.innerHTML = `<div class="alert alert-danger">Error: ${data.error}</div>`;
        return;
    }
    
    let html = '<div class="results-section">';
    
    // Statistics
    if (data.statistics) {
        html += '<div class="results-title">Statistical Summary</div>';
        html += '<div class="table-responsive"><table class="stats-table">';
        html += '<thead><tr><th>Statistic</th>';
        
        // Get all column names
        const columns = Object.keys(data.statistics);
        columns.forEach(col => {
            html += `<th>${col}</th>`;
        });
        
        html += '</tr></thead><tbody>';
        
        // Get all statistic types (count, mean, std, etc.)
        if (columns.length > 0 && data.statistics[columns[0]]) {
            const statTypes = Object.keys(data.statistics[columns[0]]);
            
            statTypes.forEach(stat => {
                html += `<tr><td>${stat}</td>`;
                
                columns.forEach(col => {
                    const value = data.statistics[col][stat];
                    html += `<td>${typeof value === 'number' ? value.toFixed(4) : value}</td>`;
                });
                
                html += '</tr>';
            });
        }
        
        html += '</tbody></table></div>';
    }
    
    // Missing values
    if (data.missing_values) {
        html += '<div class="results-title mt-4">Missing Values</div>';
        html += '<div class="table-responsive"><table class="stats-table">';
        html += '<thead><tr><th>Column</th><th>Missing Count</th></tr></thead><tbody>';
        
        Object.entries(data.missing_values).forEach(([col, count]) => {
            html += `<tr><td>${col}</td><td>${count}</td></tr>`;
        });
        
        html += '</tbody></table></div>';
    }
    
    // Autocorrelation
    if (data.autocorrelation) {
        html += '<div class="results-title mt-4">Autocorrelation (Lag 1)</div>';
        html += '<div class="table-responsive"><table class="stats-table">';
        html += '<thead><tr><th>Column</th><th>Autocorrelation</th></tr></thead><tbody>';
        
        Object.entries(data.autocorrelation).forEach(([col, value]) => {
            html += `<tr><td>${col}</td><td>${typeof value === 'number' ? value.toFixed(4) : value}</td></tr>`;
        });
        
        html += '</tbody></table></div>';
    }
    
    // Plots
    if (data.plots) {
        html += '<div class="results-title mt-4">Time Series Plots</div>';
        
        Object.entries(data.plots).forEach(([col, plotData]) => {
            html += `<div class="plot-container" id="plot-${col.replace(/\s+/g, '-')}"></div>`;
        });
    }
    
    html += '</div>';
    
    // Display the HTML
    container.innerHTML = html;
    
    // Create the plots
    if (data.plots) {
        Object.entries(data.plots).forEach(([col, plotData]) => {
            const plotElement = document.getElementById(`plot-${col.replace(/\s+/g, '-')}`);
            if (plotElement) {
                Plotly.newPlot(plotElement, plotData.data, plotData.layout);
            }
        });
    }
}

/**
 * Display anomaly detection results
 * @param {Object} data - Anomaly detection results data
 * @param {HTMLElement} container - Container element for results
 */
function displayAnomalyResults(data, container) {
    if (data.error) {
        container.innerHTML = `<div class="alert alert-danger">Error: ${data.error}</div>`;
        return;
    }
    
    let html = '<div class="results-section">';
    html += `<div class="alert alert-info">Method: ${data.method}</div>`;
    
    // Results summary
    if (data.results) {
        html += '<div class="results-title">Anomaly Detection Results</div>';
        html += '<div class="table-responsive"><table class="stats-table">';
        html += '<thead><tr><th>Column</th><th>Total Points</th><th>Anomalies</th><th>Percentage</th></tr></thead><tbody>';
        
        Object.entries(data.results).forEach(([col, result]) => {
            if (result.error) {
                html += `<tr><td>${col}</td><td colspan="3">${result.error}</td></tr>`;
            } else {
                html += `<tr>
                    <td>${col}</td>
                    <td>${result.total_points}</td>
                    <td>${result.anomalies_count}</td>
                    <td>${result.anomalies_percentage.toFixed(2)}%</td>
                </tr>`;
            }
        });
        
        html += '</tbody></table></div>';
    }
    
    // Plots
    if (data.plots) {
        html += '<div class="results-title mt-4">Anomaly Detection Plots</div>';
        
        Object.entries(data.plots).forEach(([col, plotData]) => {
            html += `<div class="plot-container" id="anomaly-plot-${col.replace(/\s+/g, '-')}"></div>`;
        });
    }
    
    html += '</div>';
    
    // Display the HTML
    container.innerHTML = html;
    
    // Create the plots
    if (data.plots) {
        Object.entries(data.plots).forEach(([col, plotData]) => {
            const plotElement = document.getElementById(`anomaly-plot-${col.replace(/\s+/g, '-')}`);
            if (plotElement) {
                Plotly.newPlot(plotElement, plotData.data, plotData.layout);
            }
        });
    }
}

/**
 * Display forecasting results
 * @param {Object} data - Forecasting results data
 * @param {HTMLElement} container - Container element for results
 */
function displayForecastResults(data, container) {
    if (data.error) {
        container.innerHTML = `<div class="alert alert-danger">Error: ${data.error}</div>`;
        return;
    }
    
    let html = '<div class="results-section">';
    html += `<div class="alert alert-info">Model: ${data.model}, Periods: ${data.periods}</div>`;
    
    // Results summary
    if (data.results) {
        html += '<div class="results-title">Forecasting Results</div>';
        
        Object.entries(data.results).forEach(([col, result]) => {
            if (result.error) {
                html += `<div class="alert alert-warning">${col}: ${result.error}</div>`;
            } else {
                html += `<h5 class="mt-3">${col}</h5>`;
                html += '<div class="table-responsive"><table class="stats-table">';
                html += '<thead><tr><th>Period</th><th>Forecast</th>';
                
                if (result.confidence_intervals) {
                    html += '<th>Lower Bound</th><th>Upper Bound</th>';
                }
                
                html += '</tr></thead><tbody>';
                
                for (let i = 0; i < result.forecast_values.length; i++) {
                    html += `<tr><td>${i + 1}</td><td>${result.forecast_values[i].toFixed(4)}</td>`;
                    
                    if (result.confidence_intervals) {
                        html += `<td>${result.confidence_intervals.lower[i].toFixed(4)}</td>`;
                        html += `<td>${result.confidence_intervals.upper[i].toFixed(4)}</td>`;
                    }
                    
                    html += '</tr>';
                }
                
                html += '</tbody></table></div>';
            }
        });
    }
    
    // Plots
    if (data.plots) {
        html += '<div class="results-title mt-4">Forecasting Plots</div>';
        
        Object.entries(data.plots).forEach(([col, plotData]) => {
            html += `<div class="plot-container" id="forecast-plot-${col.replace(/\s+/g, '-')}"></div>`;
        });
    }
    
    // Download links
    html += '<div class="mt-4">';
    html += `<a href="/download/${fileId}/forecast" class="btn btn-success">Download Forecast Results</a>`;
    html += '</div>';
    
    html += '</div>';
    
    // Display the HTML
    container.innerHTML = html;
    
    // Create the plots
    if (data.plots) {
        Object.entries(data.plots).forEach(([col, plotData]) => {
            const plotElement = document.getElementById(`forecast-plot-${col.replace(/\s+/g, '-')}`);
            if (plotElement) {
                Plotly.newPlot(plotElement, plotData.data, plotData.layout);
            }
        });
    }
}
