// Minimal forecasting script to isolate the issue

document.addEventListener('DOMContentLoaded', function() {
    const forecastButton = document.getElementById('forecast-button');
    if (forecastButton) {
        forecastButton.addEventListener('click', runForecast);
    }
    
    const fileInput = document.getElementById('file-input');
    if (fileInput) {
        fileInput.addEventListener('change', function() {
            document.getElementById('file-name').textContent = this.files[0] ? this.files[0].name : 'No file selected';
            
            // Check file size
            if (this.files[0] && this.files[0].size > 100 * 1024 * 1024) {
                alert('File is too large! Maximum size is 100MB.');
                this.value = '';
                document.getElementById('file-name').textContent = 'No file selected';
            }
        });
    }
    
    // Initialize the multiple forecast periods input
    const addPeriodButton = document.getElementById('add-period-button');
    if (addPeriodButton) {
        addPeriodButton.addEventListener('click', addForecastPeriod);
    }
    
    // Initialize enhanced EDA button
    const edaButton = document.getElementById('enhanced-eda-button');
    if (edaButton) {
        edaButton.addEventListener('click', runEnhancedEDA);
    }
});

/**
 * Add a new forecast period input field
 */
function addForecastPeriod() {
    const periodsContainer = document.getElementById('forecast-periods-container');
    const periodCount = periodsContainer.getElementsByClassName('forecast-period-input').length;
    
    // Limit to 3 periods
    if (periodCount >= 3) {
        alert('Maximum 3 forecast periods allowed');
        return;
    }
    
    const newInput = document.createElement('div');
    newInput.className = 'input-group mb-2 forecast-period-input';
    newInput.innerHTML = `
        <input type="number" class="form-control forecast-period" min="1" max="1000" value="30">
        <div class="input-group-append">
            <button class="btn btn-outline-danger remove-period-button" type="button">Remove</button>
        </div>
    `;
    
    periodsContainer.appendChild(newInput);
    
    // Add event listener to the remove button
    const removeButton = newInput.querySelector('.remove-period-button');
    removeButton.addEventListener('click', function() {
        periodsContainer.removeChild(newInput);
    });
}

/**
 * Run enhanced exploratory data analysis
 */
function runEnhancedEDA() {
    const fileInput = document.getElementById('file-input');
    const resultDiv = document.getElementById('forecast-result');
    
    if (!fileInput.files[0]) {
        resultDiv.innerHTML = '<div class="alert alert-warning">Please select a file first</div>';
        return;
    }
    
    // Check file size
    if (fileInput.files[0].size > 100 * 1024 * 1024) {
        resultDiv.innerHTML = '<div class="alert alert-danger">File is too large! Maximum size is 100MB.</div>';
        return;
    }
    
    resultDiv.innerHTML = '<div class="alert alert-info">Processing enhanced EDA... Please wait.</div>';
    console.log("Starting enhanced EDA analysis");
    
    // Create form data for file upload
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    
    // First upload the file
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            if (response.status === 413) {
                throw new Error('File is too large! Maximum size is 100MB.');
            }
            return response.json().then(data => {
                // If the response contains structured error data, use it
                if (data.error === true && data.error_type) {
                    throw data;
                }
                // Otherwise create a generic error
                throw new Error(`HTTP error! Status: ${response.status}`);
            }).catch(err => {
                // If JSON parsing fails, throw the original error
                if (err instanceof SyntaxError) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                throw err;
            });
        }
        return response.json();
    })
    .then(data => {
        console.log("Upload response:", data);
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        const fileId = data.file_id;
        
        // Now run enhanced EDA
        return fetch('/api/enhanced-eda', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                file_id: fileId
            })
        });
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(data => {
                // If the response contains structured error data, use it
                if (data.error === true && data.error_type) {
                    throw data;
                }
                // Otherwise create a generic error
                throw new Error(`HTTP error! Status: ${response.status}`);
            }).catch(err => {
                // If JSON parsing fails, throw the original error
                if (err instanceof SyntaxError) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                throw err;
            });
        }
        return response.json();
    })
    .then(data => {
        console.log("Enhanced EDA response:", data);
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        // Display the results
        displayEDAResults(data);
    })
    .catch(error => {
        console.error("Enhanced EDA error:", error);
        
        // Check if this is a structured error response
        if (error.error === true && error.error_type) {
            displayErrorMessage(error, resultDiv);
        } else {
            // Fallback for non-structured errors
            resultDiv.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
        }
    });
}

/**
 * Display EDA results in a user-friendly format
 */
function displayEDAResults(results) {
    const resultDiv = document.getElementById('forecast-result');
    
    let html = `
        <div class="alert alert-success">
            <h4>Enhanced Exploratory Data Analysis Complete!</h4>
        </div>
    `;
    
    // Process each column's results
    if (results) {
        Object.keys(results).forEach(column => {
            const columnResults = results[column];
            
            html += `
                <div class="card mb-3">
                    <div class="card-header bg-primary text-white">
                        <h5 class="mb-0">Column: ${column}</h5>
                    </div>
                    <div class="card-body">
            `;
            
            // Summary statistics
            if (columnResults.summary_stats) {
                html += `
                    <h6>Summary Statistics</h6>
                    <div class="table-responsive mb-3">
                        <table class="table table-sm table-bordered">
                            <thead>
                                <tr>
                                    <th>Statistic</th>
                                    <th>Value</th>
                                </tr>
                            </thead>
                            <tbody>
                `;
                
                Object.entries(columnResults.summary_stats).forEach(([stat, value]) => {
                    html += `
                        <tr>
                            <td>${stat}</td>
                            <td>${typeof value === 'number' ? value.toFixed(4) : value}</td>
                        </tr>
                    `;
                });
                
                html += `
                            </tbody>
                        </table>
                    </div>
                `;
            }
            
            // Stationarity tests
            if (columnResults.stationarity_tests) {
                html += `
                    <h6>Stationarity Tests</h6>
                    <div class="table-responsive mb-3">
                        <table class="table table-sm table-bordered">
                            <thead>
                                <tr>
                                    <th>Test</th>
                                    <th>Statistic</th>
                                    <th>p-value</th>
                                    <th>Result</th>
                                </tr>
                            </thead>
                            <tbody>
                `;
                
                Object.entries(columnResults.stationarity_tests).forEach(([test, results]) => {
                    html += `
                        <tr>
                            <td>${test}</td>
                            <td>${results.statistic ? results.statistic.toFixed(4) : 'N/A'}</td>
                            <td>${results.p_value ? results.p_value.toFixed(4) : 'N/A'}</td>
                            <td>${results.is_stationary ? 
                                '<span class="badge bg-success">Stationary</span>' : 
                                '<span class="badge bg-warning">Non-stationary</span>'}</td>
                        </tr>
                    `;
                });
                
                html += `
                            </tbody>
                        </table>
                    </div>
                `;
            }
            
            // Add other sections as needed
            if (columnResults.seasonality) {
                html += `
                    <h6>Seasonality Analysis</h6>
                    <p>Detected seasonality: ${columnResults.seasonality.detected ? 
                        `Yes (Period: ${columnResults.seasonality.period})` : 'No'}</p>
                `;
            }
            
            html += `
                    </div>
                </div>
            `;
        });
    }
    
    resultDiv.innerHTML = html;
}

/**
 * Display a structured error message with appropriate styling based on error type
 * @param {Object} errorData - The error data object from the API
 * @param {HTMLElement} container - The container element to display the error in
 */
function displayErrorMessage(errorData, container) {
    console.error("Error:", errorData);
    
    // Default values if not provided
    const errorType = errorData.error_type || 'system_error';
    const errorTitle = errorData.error_title || 'Error';
    const errorMessage = errorData.error_message || 'An unknown error occurred';
    const errorIcon = errorData.error_icon || 'exclamation-circle';
    const errorDetails = errorData.error_details || '';
    const errorSuggestions = errorData.error_suggestions || [];
    
    // Map error types to Bootstrap alert classes
    const alertClassMap = {
        'validation_error': 'alert-warning',
        'data_error': 'alert-danger',
        'model_error': 'alert-primary',
        'system_error': 'alert-danger',
        'dependency_error': 'alert-info'
    };
    
    // Get the appropriate alert class based on error type
    const alertClass = alertClassMap[errorType] || 'alert-danger';
    
    // Build the error message HTML
    let errorHTML = `
        <div class="alert ${alertClass} error-container">
            <div class="d-flex align-items-center mb-2">
                <i class="bi bi-${errorIcon} fs-3 me-2"></i>
                <h4 class="mb-0">${errorTitle}</h4>
            </div>
            <p class="mb-2">${errorMessage}</p>
    `;
    
    // Add error details if available
    if (errorDetails) {
        errorHTML += `<div class="error-details mb-2"><strong>Details:</strong> ${errorDetails}</div>`;
    }
    
    // Add suggestions if available
    if (errorSuggestions && errorSuggestions.length > 0) {
        errorHTML += `
            <div class="error-suggestions">
                <strong>Suggestions:</strong>
                <ul class="mb-0">
        `;
        
        errorSuggestions.forEach(suggestion => {
            errorHTML += `<li>${suggestion}</li>`;
        });
        
        errorHTML += `
                </ul>
            </div>
        `;
    }
    
    errorHTML += `</div>`;
    
    // Set the error message in the container
    container.innerHTML = errorHTML;
    
    // Scroll to the error message
    container.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/**
 * Run a minimal forecast test
 */
function runForecast() {
    const fileInput = document.getElementById('file-input');
    const resultDiv = document.getElementById('forecast-result');
    const forecastModel = document.getElementById('forecast-model').value;
    const forecastPeriods = parseInt(document.getElementById('forecast-periods').value);
    
    // Validate inputs
    if (!fileInput.files[0]) {
        resultDiv.innerHTML = '<div class="alert alert-warning">Please select a file first</div>';
        return;
    }
    
    if (isNaN(forecastPeriods) || forecastPeriods < 1) {
        resultDiv.innerHTML = '<div class="alert alert-warning">Please enter a valid number of forecast periods</div>';
        return;
    }
    
    // Check file size
    if (fileInput.files[0].size > 100 * 1024 * 1024) {
        resultDiv.innerHTML = '<div class="alert alert-danger">File is too large! Maximum size is 100MB.</div>';
        return;
    }
    
    // Get additional forecast periods
    const additionalPeriods = [];
    const periodInputs = document.getElementsByClassName('forecast-period');
    for (let i = 0; i < periodInputs.length; i++) {
        const period = parseInt(periodInputs[i].value);
        if (!isNaN(period) && period > 0) {
            additionalPeriods.push(period);
        }
    }
    
    resultDiv.innerHTML = '<div class="alert alert-info">Processing forecast... Please wait.</div>';
    console.log(`Starting forecast with model ${forecastModel} for ${forecastPeriods} periods`);
    
    // Create form data for file upload
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    
    // First upload the file
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            if (response.status === 413) {
                throw new Error('File is too large! Maximum size is 100MB.');
            }
            return response.json().then(data => {
                // If the response contains structured error data, use it
                if (data.error === true && data.error_type) {
                    throw data;
                }
                // Otherwise create a generic error
                throw new Error(`HTTP error! Status: ${response.status}`);
            }).catch(err => {
                // If JSON parsing fails, throw the original error
                if (err instanceof SyntaxError) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                throw err;
            });
        }
        return response.json();
    })
    .then(data => {
        console.log("Upload response:", data);
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        const fileId = data.file_id;
        
        // Now run forecast
        const requestBody = {
            file_id: fileId,
            model: forecastModel,
            periods: forecastPeriods
        };
        
        // Add additional periods if any
        if (additionalPeriods.length > 0) {
            requestBody.additional_periods = additionalPeriods;
        }
        
        return fetch('/api/minimal-forecast', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestBody)
        });
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(data => {
                // If the response contains structured error data, use it
                if (data.error === true && data.error_type) {
                    throw data;
                }
                // Otherwise create a generic error
                throw new Error(`HTTP error! Status: ${response.status}`);
            }).catch(err => {
                // If JSON parsing fails, throw the original error
                if (err instanceof SyntaxError) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                throw err;
            });
        }
        return response.json();
    })
    .then(data => {
        console.log("Forecast response:", data);
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        // Display the results
        let html = '<div class="card mb-4">';
        html += '<div class="card-header bg-primary text-white">Forecast Results</div>';
        html += '<div class="card-body">';
        
        // Add model info
        html += `<p><strong>Model:</strong> ${data.model}</p>`;
        html += `<p><strong>Periods:</strong> ${data.periods}</p>`;
        
        // Add metrics
        if (data.metrics) {
            html += '<h5 class="mt-4">Model Metrics</h5>';
            html += '<table class="table table-sm">';
            html += '<thead><tr><th>Metric</th><th>Value</th></tr></thead>';
            html += '<tbody>';
            
            for (const [key, value] of Object.entries(data.metrics)) {
                html += `<tr><td>${key}</td><td>${value !== null && value !== undefined ? value.toFixed(4) : 'N/A'}</td></tr>`;
            }
            
            html += '</tbody></table>';
        }
        
        // Add forecast plot
        if (data.plot_data) {
            html += '<h5 class="mt-4">Forecast Plot</h5>';
            html += '<div id="forecast-plot" style="width:100%; height:400px;"></div>';
            
            // Store plot data for later use
            window.forecastPlotData = data.plot_data;
            
            // Schedule plot creation after the HTML is inserted
            setTimeout(() => {
                const plotData = [];
                
                // Add historical data
                plotData.push({
                    x: data.plot_data.historical_dates,
                    y: data.plot_data.historical_values,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Historical',
                    line: { color: 'blue' }
                });
                
                // Add forecast data
                plotData.push({
                    x: data.plot_data.forecast_dates,
                    y: data.plot_data.forecast_values,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Forecast',
                    line: { color: 'red' }
                });
                
                // Add confidence intervals if available
                if (data.plot_data.lower_bounds && data.plot_data.upper_bounds) {
                    plotData.push({
                        x: data.plot_data.forecast_dates,
                        y: data.plot_data.upper_bounds,
                        type: 'scatter',
                        mode: 'lines',
                        name: 'Upper Bound',
                        line: { color: 'rgba(255, 0, 0, 0.2)' },
                        showlegend: false
                    });
                    
                    plotData.push({
                        x: data.plot_data.forecast_dates,
                        y: data.plot_data.lower_bounds,
                        type: 'scatter',
                        mode: 'lines',
                        name: 'Lower Bound',
                        line: { color: 'rgba(255, 0, 0, 0.2)' },
                        fill: 'tonexty',
                        showlegend: false
                    });
                }
                
                const layout = {
                    title: 'Time Series Forecast',
                    xaxis: { title: 'Date' },
                    yaxis: { title: 'Value' },
                    hovermode: 'closest'
                };
                
                Plotly.newPlot('forecast-plot', plotData, layout);
            }, 100);
        }
        
        html += '</div></div>';
        
        resultDiv.innerHTML = html;
    })
    .catch(error => {
        console.error("Forecast error:", error);
        
        // Check if this is a structured error response
        if (error.error === true && error.error_type) {
            displayErrorMessage(error, resultDiv);
        } else {
            // Fallback for non-structured errors
            resultDiv.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
        }
    });
}
