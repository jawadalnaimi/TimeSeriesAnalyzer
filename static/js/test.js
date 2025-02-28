// Test script to isolate the issue

document.addEventListener('DOMContentLoaded', function() {
    const testButton = document.getElementById('test-button');
    if (testButton) {
        testButton.addEventListener('click', runTest);
    }
});

/**
 * Run a simple test to check if the API is working
 */
function runTest() {
    const resultDiv = document.getElementById('test-result');
    resultDiv.innerHTML = '<div class="alert alert-info">Testing API...</div>';
    
    console.log("Running test...");
    
    // Call the test API
    fetch('/api/test')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log("Test response:", data);
            resultDiv.innerHTML = `
                <div class="alert alert-success">
                    <h4>Test Successful!</h4>
                    <p>Message: ${data.message}</p>
                    <p>Status: ${data.status}</p>
                    <pre>${JSON.stringify(data.data, null, 2)}</pre>
                </div>
            `;
        })
        .catch(error => {
            console.error("Test error:", error);
            resultDiv.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
        });
}
