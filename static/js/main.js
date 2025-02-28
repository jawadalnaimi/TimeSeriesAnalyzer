// Main JavaScript for Time Series Analyzer

document.addEventListener('DOMContentLoaded', function() {
    // Set up event listeners
    const uploadForm = document.getElementById('upload-form');
    if (uploadForm) {
        uploadForm.addEventListener('submit', handleUpload);
    }
    
    const forecastButton = document.getElementById('forecast-button');
    if (forecastButton) {
        forecastButton.addEventListener('click', handleForecasting);
    }
    
    // File input change handler to show selected file name
    const fileInput = document.getElementById('file');
    if (fileInput) {
        fileInput.addEventListener('change', function() {
            const fileName = this.files[0] ? this.files[0].name : 'No file selected';
            const fileLabel = this.nextElementSibling;
            if (fileLabel) {
                fileLabel.textContent = fileName;
            }
            
            // Check file size
            if (this.files[0] && this.files[0].size > 100 * 1024 * 1024) {
                alert('File is too large! Maximum size is 100MB.');
                this.value = '';
                if (fileLabel) {
                    fileLabel.textContent = 'No file selected';
                }
            }
        });
    }
    
    // Forecast form submission handler
    document.getElementById('forecast-form').addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Show loading indicator
        document.getElementById('loading-indicator').style.display = 'block';
        document.getElementById('error-message').style.display = 'none';
        document.getElementById('results-container').style.display = 'none';
        
        try {
            // Get form data
            const fileId = document.getElementById('file-id').value;
            const model = document.getElementById('model').value;
            const periods = document.getElementById('periods').value;
            
            console.log(`Generating forecast for file ${fileId} using model ${model} for ${periods} periods`);
            
            // Make API request
            const response = await fetch('/api/forecast', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    file_id: fileId,
                    model: model,
                    periods: parseInt(periods)
                })
            });
            
            // Check if response is ok
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            
            // Parse response
            const data = await response.json();
            
            if (data.success) {
                // Display results
                displayResults(data);
            } else {
                throw new Error(data.error || 'Unknown error occurred');
            }
        } catch (error) {
            console.error('Error:', error);
            document.getElementById('error-message').textContent = error.message;
            document.getElementById('error-message').style.display = 'block';
        } finally {
            // Hide loading indicator
            document.getElementById('loading-indicator').style.display = 'none';
        }
    });
});

/**
 * Handle file upload
 * @param {Event} event - Form submit event
 */
function handleUpload(event) {
    event.preventDefault();
    
    const fileInput = document.getElementById('file');
    if (!fileInput.files[0]) {
        alert('Please select a file first');
        return;
    }
    
    // Check file size again
    if (fileInput.files[0].size > 100 * 1024 * 1024) {
        alert('File is too large! Maximum size is 100MB.');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    
    // Show loading status
    const statusDiv = document.getElementById('upload-status');
    statusDiv.style.display = 'block';
    statusDiv.innerHTML = '<div class="alert alert-info">Uploading file... Please wait.</div>';
    
    // Upload the file
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            if (response.status === 413) {
                throw new Error('File is too large! Maximum size is 100MB.');
            }
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        console.log('Upload successful:', data);
        
        if (data.error) {
            statusDiv.innerHTML = `<div class="alert alert-danger">Error: ${data.error}</div>`;
            return;
        }
        
        // Show success message
        statusDiv.innerHTML = `<div class="alert alert-success">
            <strong>File uploaded successfully!</strong><br>
            File ID: ${data.file_id}<br>
            Filename: ${data.filename}
        </div>`;
        
        // Store the file ID for later use
        document.getElementById('forecast-file-id').value = data.file_id;
        
        // Enable analysis buttons
        enableAnalysisButtons();
    })
    .catch(error => {
        console.error('Error:', error);
        statusDiv.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
    });
}

/**
 * Enable analysis buttons after file upload
 */
function enableAnalysisButtons() {
    const analysisButtons = document.querySelectorAll('.analysis-btn');
    analysisButtons.forEach(button => {
        button.disabled = false;
    });
}

/**
 * Handle forecasting form submission
 */
function handleForecasting() {
    const fileId = document.getElementById('forecast-file-id').value;
    const forecastModel = document.getElementById('forecast-model').value;
    const forecastPeriods = parseInt(document.getElementById('forecast-periods').value);
    const resultDiv = document.getElementById('forecast-result');
    
    console.log("Starting forecast with fileId:", fileId, "model:", forecastModel, "periods:", forecastPeriods);
    
    if (!fileId) {
        resultDiv.innerHTML = '<div class="alert alert-warning">Please upload a file first</div>';
        return;
    }
    
    resultDiv.innerHTML = '<div class="alert alert-info">Processing... Please wait.</div>';
    
    forecastData(fileId, forecastModel, forecastPeriods)
        .then(result => {
            console.log("Forecast result:", result);
            displayForecastResults(result);
        })
        .catch(error => {
            console.error("Forecast error:", error);
            resultDiv.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
        });
}

/**
 * Call the forecast API
 */
function forecastData(fileId, model, periods) {
    console.log("Calling forecast API with:", {file_id: fileId, model: model, periods: periods});
    
    return fetch('/api/forecast', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            file_id: fileId,
            model: model,
            periods: periods
        })
    })
    .then(response => {
        console.log("Received response with status:", response.status);
        if (!response.ok) {
            if (response.status === 413) {
                throw new Error('Response data is too large!');
            }
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.text().then(text => {
            console.log("Raw response text:", text);
            try {
                return JSON.parse(text);
            } catch (e) {
                console.error("JSON parse error:", e);
                throw new Error(`Failed to parse JSON response: ${e.message}`);
            }
        });
    })
    .then(data => {
        console.log("Parsed data:", data);
        if (data.error) {
            throw new Error(data.error);
        }
        return data;
    });
}

/**
 * Display forecast results
 */
function displayForecastResults(data) {
    const resultDiv = document.getElementById('forecast-result');
    
    console.log("Displaying forecast results:", data);
    
    if (!data || data.error) {
        resultDiv.innerHTML = `<div class="alert alert-danger">Error: ${data.error || 'Unknown error'}</div>`;
        return;
    }
    
    // Display the results header with model information
    let html = `
        <div class="alert alert-success">
            <h4>Forecast Complete!</h4>
            <p><strong>Model:</strong> ${data.model_name || data.model}</p>
            <p><strong>Periods:</strong> ${data.periods}</p>
            <p><strong>Date Range:</strong> ${data.forecast_start_date || 'N/A'} to ${data.forecast_end_date || 'N/A'}</p>
        </div>
    `;
    
    // Add results for each column
    if (data.results) {
        Object.keys(data.results).forEach(column => {
            const result = data.results[column];
            
            if (result.error) {
                html += `
                    <div class="card mb-3">
                        <div class="card-header">Column: ${column}</div>
                        <div class="card-body">
                            <div class="alert alert-warning">${result.error}</div>
                        </div>
                    </div>
                `;
                return;
            }
            
            // Create a card for each column result
            html += `
                <div class="card mb-4">
                    <div class="card-header">
                        <h5 class="mb-0">Forecast for: ${column}</h5>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-8">
                                <!-- Plotly chart container -->
                                <div id="chart-${column.replace(/[^a-zA-Z0-9]/g, '-')}" class="forecast-chart" style="height: 400px;"></div>
                            </div>
                            <div class="col-md-4">
                                <!-- Model information -->
                                <div class="card">
                                    <div class="card-header">Model Information</div>
                                    <div class="card-body">
                                        <p><strong>Model:</strong> ${result.model_info?.name || data.model}</p>
                                        <p><strong>Description:</strong> ${result.model_info?.description || 'N/A'}</p>
                                        <p><strong>Forecast Start:</strong> ${result.model_info?.forecast_start || 'N/A'}</p>
                                        <p><strong>Forecast End:</strong> ${result.model_info?.forecast_end || 'N/A'}</p>
                                        
                                        ${result.model_info?.order ? 
                                            `<p><strong>Model Order:</strong> (${result.model_info.order.join(',')})</p>` : ''}
                                        
                                        ${result.model_info?.is_stationary !== undefined ? 
                                            `<p><strong>Stationary:</strong> ${result.model_info.is_stationary ? 'Yes' : 'No'}</p>` : ''}
                                        
                                        ${result.model_info?.metrics ? 
                                            `<div class="mt-3">
                                                <h6>Model Metrics:</h6>
                                                <ul class="list-group list-group-flush">
                                                    ${result.model_info.metrics.rmse !== undefined ? 
                                                        `<li class="list-group-item p-1">RMSE: ${result.model_info.metrics.rmse.toFixed(3)}</li>` : ''}
                                                    ${result.model_info.metrics.mae !== undefined ? 
                                                        `<li class="list-group-item p-1">MAE: ${result.model_info.metrics.mae.toFixed(3)}</li>` : ''}
                                                    ${result.model_info.metrics.mape !== undefined ? 
                                                        `<li class="list-group-item p-1">MAPE: ${result.model_info.metrics.mape.toFixed(2)}%</li>` : ''}
                                                </ul>
                                            </div>` : ''}
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="mt-4">
                            <h6>Forecast Values</h6>
                            <div class="table-responsive">
                                <table class="table table-striped table-sm">
                                    <thead>
                                        <tr>
                                            <th>Date</th>
                                            <th>Forecast</th>
                                        </tr>
                                    </thead>
                                    <tbody>
            `;
            
            // Add rows for each forecast point
            if (result.dates && result.forecast) {
                for (let i = 0; i < result.dates.length; i++) {
                    const formattedValue = typeof result.forecast[i] === 'number' 
                        ? result.forecast[i].toFixed(2)
                        : result.forecast[i];
                        
                    html += `
                        <tr>
                            <td>${result.dates[i]}</td>
                            <td>${formattedValue}</td>
                        </tr>
                    `;
                }
            }
            
            html += `
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        });
    }
    
    // Update the result div
    resultDiv.innerHTML = html;
    
    // Create Plotly charts for each column
    if (data.results) {
        Object.keys(data.results).forEach(column => {
            const result = data.results[column];
            
            if (result.error) {
                return;
            }
            
            try {
                const chartId = `chart-${column.replace(/[^a-zA-Z0-9]/g, '-')}`;
                const chartDiv = document.getElementById(chartId);
                
                if (chartDiv) {
                    // Create a simple plot from the data
                    const traces = [];
                    
                    // Add historical data if available
                    if (result.historical_dates && result.historical_values) {
                        traces.push({
                            x: result.historical_dates,
                            y: result.historical_values,
                            mode: 'lines',
                            name: 'Historical',
                            line: {color: 'blue'}
                        });
                    }
                    
                    // Add forecast data
                    if (result.dates && result.forecast) {
                        traces.push({
                            x: result.dates,
                            y: result.forecast,
                            mode: 'lines',
                            name: 'Forecast',
                            line: {color: 'red', dash: 'dash'}
                        });
                        
                        // Add confidence intervals if available
                        if (result.lower_bounds && result.upper_bounds) {
                            // Create a filled area for the confidence interval
                            traces.push({
                                x: result.dates.concat(result.dates.slice().reverse()),
                                y: result.upper_bounds.concat(result.lower_bounds.slice().reverse()),
                                fill: 'toself',
                                fillcolor: 'rgba(255, 0, 0, 0.1)',
                                line: {color: 'transparent'},
                                name: '95% Confidence Interval',
                                showlegend: true,
                                hoverinfo: 'skip'
                            });
                        }
                    }
                    
                    if (traces.length > 0) {
                        const layout = {
                            title: `Forecast for ${column}`,
                            xaxis: {title: 'Date'},
                            yaxis: {title: 'Value'},
                            legend: {title: 'Data Type'},
                            hovermode: 'x unified'
                        };
                        
                        console.log(`Creating chart for ${column} with traces:`, traces);
                        Plotly.newPlot(chartId, traces, layout);
                    }
                }
            } catch (error) {
                console.error(`Error creating chart for ${column}:`, error);
            }
        });
    }
}

/**
 * Function to display results
 */
function displayResults(data) {
    console.log('Displaying results:', data);
    
    const resultsContainer = document.getElementById('results-container');
    resultsContainer.innerHTML = '';
    resultsContainer.style.display = 'block';
    
    // Create a container for each result
    for (const [column, result] of Object.entries(data.results)) {
        // Skip if there's an error for this column
        if (result.error) {
            console.error(`Error for column ${column}:`, result.error);
            continue;
        }
        
        // Create a container for this column
        const columnContainer = document.createElement('div');
        columnContainer.className = 'column-result';
        columnContainer.innerHTML = `<h3>Forecast for ${column}</h3>`;
        
        // Create a div for the chart
        const chartDiv = document.createElement('div');
        chartDiv.id = `chart-${column.replace(/\s+/g, '-')}`;
        chartDiv.className = 'chart-container';
        columnContainer.appendChild(chartDiv);
        
        // Create a div for model info
        const modelInfoDiv = document.createElement('div');
        modelInfoDiv.className = 'model-info';
        
        // Add model information
        if (result.model_info) {
            const modelInfo = result.model_info;
            let modelInfoHTML = `<h4>Model Information</h4>`;
            modelInfoHTML += `<p><strong>Model:</strong> ${modelInfo.name || data.model}</p>`;
            
            if (modelInfo.order) {
                modelInfoHTML += `<p><strong>Order:</strong> (${modelInfo.order.join(', ')})</p>`;
            }
            
            if (modelInfo.description) {
                modelInfoHTML += `<p><strong>Description:</strong> ${modelInfo.description}</p>`;
            }
            
            if (modelInfo.is_stationary !== undefined) {
                modelInfoHTML += `<p><strong>Stationary:</strong> ${modelInfo.is_stationary ? 'Yes' : 'No'}</p>`;
            }
            
            if (modelInfo.aic !== undefined) {
                modelInfoHTML += `<p><strong>AIC:</strong> ${modelInfo.aic.toFixed(2)}</p>`;
            }
            
            // Add metrics if available
            if (modelInfo.metrics) {
                modelInfoHTML += `<h4>Performance Metrics</h4>`;
                modelInfoHTML += `<ul>`;
                if (modelInfo.metrics.rmse !== undefined) {
                    modelInfoHTML += `<li><strong>RMSE:</strong> ${modelInfo.metrics.rmse.toFixed(2)}</li>`;
                }
                if (modelInfo.metrics.mae !== undefined) {
                    modelInfoHTML += `<li><strong>MAE:</strong> ${modelInfo.metrics.mae.toFixed(2)}</li>`;
                }
                if (modelInfo.metrics.mape !== undefined) {
                    modelInfoHTML += `<li><strong>MAPE:</strong> ${modelInfo.metrics.mape.toFixed(2)}%</li>`;
                }
                modelInfoHTML += `</ul>`;
            }
            
            modelInfoDiv.innerHTML = modelInfoHTML;
            columnContainer.appendChild(modelInfoDiv);
        }
        
        // Add the column container to the results container
        resultsContainer.appendChild(columnContainer);
        
        // Create the chart
        createChart(
            chartDiv.id, 
            result.historical_dates, 
            result.historical_values,
            result.dates, 
            result.forecast,
            result.lower_bounds,
            result.upper_bounds
        );
    }
}

/**
 * Function to create a chart
 */
function createChart(chartId, historicalDates, historicalValues, forecastDates, forecastValues, lowerBounds, upperBounds) {
    // Create historical data trace
    const historicalTrace = {
        x: historicalDates,
        y: historicalValues,
        type: 'scatter',
        mode: 'lines',
        name: 'Historical Data',
        line: {
            color: 'blue',
            width: 2
        }
    };
    
    // Create forecast trace
    const forecastTrace = {
        x: forecastDates,
        y: forecastValues,
        type: 'scatter',
        mode: 'lines',
        name: 'Forecast',
        line: {
            color: 'red',
            width: 2
        }
    };
    
    // Create data array
    const data = [historicalTrace, forecastTrace];
    
    // Add confidence intervals if available
    if (lowerBounds && upperBounds) {
        // Create upper bound trace
        const upperBoundTrace = {
            x: forecastDates,
            y: upperBounds,
            type: 'scatter',
            mode: 'lines',
            name: 'Upper Bound (95%)',
            line: {
                color: 'rgba(255, 0, 0, 0.3)',
                width: 0
            },
            showlegend: false
        };
        
        // Create lower bound trace
        const lowerBoundTrace = {
            x: forecastDates,
            y: lowerBounds,
            type: 'scatter',
            mode: 'lines',
            name: 'Lower Bound (95%)',
            line: {
                color: 'rgba(255, 0, 0, 0.3)',
                width: 0
            },
            fill: 'tonexty',
            fillcolor: 'rgba(255, 0, 0, 0.1)',
            showlegend: false
        };
        
        // Add confidence interval traces
        data.push(upperBoundTrace);
        data.push(lowerBoundTrace);
    }
    
    // Create layout
    const layout = {
        title: 'Time Series Forecast',
        xaxis: {
            title: 'Date'
        },
        yaxis: {
            title: 'Value'
        },
        legend: {
            orientation: 'h',
            y: -0.2
        },
        margin: {
            l: 50,
            r: 50,
            b: 100,
            t: 50,
            pad: 4
        },
        hovermode: 'closest'
    };
    
    // Create chart
    Plotly.newPlot(chartId, data, layout, {responsive: true});
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
                try {
                    // Check if plotData is already in the correct format
                    if (plotData.data && plotData.layout) {
                        Plotly.newPlot(plotElement, plotData.data, plotData.layout);
                    } else {
                        // If it's a raw JSON string or object, try to parse it
                        console.log("Attempting to parse plot data for", col);
                        Plotly.newPlot(plotElement, plotData);
                    }
                } catch (error) {
                    console.error("Error rendering plot for", col, error);
                    plotElement.innerHTML = `<div class="alert alert-warning">Error rendering plot: ${error.message}</div>`;
                }
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
                try {
                    // Check if plotData is already in the correct format
                    if (plotData.data && plotData.layout) {
                        Plotly.newPlot(plotElement, plotData.data, plotData.layout);
                    } else {
                        // If it's a raw JSON string or object, try to parse it
                        console.log("Attempting to parse plot data for", col);
                        Plotly.newPlot(plotElement, plotData);
                    }
                } catch (error) {
                    console.error("Error rendering plot for", col, error);
                    plotElement.innerHTML = `<div class="alert alert-warning">Error rendering plot: ${error.message}</div>`;
                }
            }
        });
    }
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

// Handle file upload
document.getElementById('upload-form').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    // Show loading indicator
    document.getElementById('upload-loading').style.display = 'block';
    document.getElementById('upload-error').style.display = 'none';
    
    try {
        const formData = new FormData(this);
        
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.success) {
            // Set the file ID in the hidden input
            document.getElementById('file-id').value = data.file_id;
            
            // Show success message
            document.getElementById('upload-success').textContent = `File "${data.filename}" uploaded successfully!`;
            document.getElementById('upload-success').style.display = 'block';
            
            // Show the forecast form
            document.getElementById('forecast-section').style.display = 'block';
            
            // Scroll to the forecast section
            document.getElementById('forecast-section').scrollIntoView({ behavior: 'smooth' });
        } else {
            throw new Error(data.error || 'Unknown error occurred');
        }
    } catch (error) {
        console.error('Error:', error);
        document.getElementById('upload-error').textContent = error.message;
        document.getElementById('upload-error').style.display = 'block';
    } finally {
        // Hide loading indicator
        document.getElementById('upload-loading').style.display = 'none';
    }
});

// Add event listener for the test model
document.getElementById('model').addEventListener('change', function() {
    const periodsInput = document.getElementById('periods');
    const modelInfoText = document.getElementById('model-info-text');
    
    if (this.value === 'test') {
        modelInfoText.textContent = 'Test model provides quick results with synthetic data for UI testing.';
        modelInfoText.style.display = 'block';
    } else if (this.value === 'arima') {
        modelInfoText.textContent = 'ARIMA (AutoRegressive Integrated Moving Average) is a statistical model for time series forecasting.';
        modelInfoText.style.display = 'block';
    } else if (this.value === 'sarima') {
        modelInfoText.textContent = 'SARIMA adds seasonal components to the ARIMA model for data with seasonal patterns.';
        modelInfoText.style.display = 'block';
    } else if (this.value === 'exponential_smoothing') {
        modelInfoText.textContent = 'Exponential smoothing gives more weight to recent observations for forecasting.';
        modelInfoText.style.display = 'block';
    } else {
        modelInfoText.style.display = 'none';
    }
});
