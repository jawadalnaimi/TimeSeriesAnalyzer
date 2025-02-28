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
});

/**
 * Run a minimal forecast test
 */
function runForecast() {
    const fileInput = document.getElementById('file-input');
    const forecastModel = document.getElementById('forecast-model').value;
    const forecastPeriods = parseInt(document.getElementById('forecast-periods').value);
    const resultDiv = document.getElementById('forecast-result');
    
    if (!fileInput.files[0]) {
        resultDiv.innerHTML = '<div class="alert alert-warning">Please select a file first</div>';
        return;
    }
    
    // Check file size again
    if (fileInput.files[0].size > 100 * 1024 * 1024) {
        resultDiv.innerHTML = '<div class="alert alert-danger">File is too large! Maximum size is 100MB.</div>';
        return;
    }
    
    resultDiv.innerHTML = '<div class="alert alert-info">Processing... Please wait.</div>';
    console.log("Starting forecast with model:", forecastModel, "periods:", forecastPeriods);
    
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
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
    })
    .then(uploadResult => {
        console.log("Upload result:", uploadResult);
        if (uploadResult.error) {
            throw new Error(uploadResult.error);
        }
        
        // Now call the minimal forecast endpoint
        return fetch('/api/minimal-forecast', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                file_id: uploadResult.file_id,
                model: forecastModel,
                periods: forecastPeriods
            })
        });
    })
    .then(response => {
        if (!response.ok) {
            if (response.status === 413) {
                throw new Error('Response data is too large!');
            }
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
    })
    .then(forecastResult => {
        console.log("Forecast result:", forecastResult);
        if (forecastResult.error) {
            throw new Error(forecastResult.error);
        }
        
        // Display the results as simple text/tables
        let html = `
            <div class="alert alert-success">
                <h4>Forecast Complete!</h4>
                <p>Model: ${forecastResult.model}</p>
                <p>Periods: ${forecastResult.periods}</p>
            </div>
        `;
        
        // Add results for each column
        if (forecastResult.results) {
            Object.keys(forecastResult.results).forEach(column => {
                const result = forecastResult.results[column];
                
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
                
                // Create a table for the forecast values
                html += `
                    <div class="card mb-3">
                        <div class="card-header">Column: ${column}</div>
                        <div class="card-body">
                            <div class="table-responsive">
                                <table class="table table-striped">
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
                        html += `
                            <tr>
                                <td>${result.dates[i]}</td>
                                <td>${result.forecast[i]}</td>
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
                `;
            });
        }
        
        resultDiv.innerHTML = html;
    })
    .catch(error => {
        console.error("Forecast error:", error);
        resultDiv.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
    });
}
