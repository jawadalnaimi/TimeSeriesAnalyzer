# Time Series Analyzer

A web-based tool for analyzing and visualizing time series data.

## Features

- Upload and process time series data in various formats (CSV, Excel, JSON)
- Interactive visualization of time series data
- Statistical analysis and anomaly detection
- Forecasting using various models (ARIMA, SARIMA, Exponential Smoothing, Prophet, LSTM)
- Export results and visualizations
- Robust error handling and data validation
- Comprehensive JSON serialization for complex data structures

## Installation

1. Clone this repository
2. Create a virtual environment:
   ```
   python -m venv venv
   ```
3. Activate the virtual environment:
   - On Windows: `venv\Scripts\activate`
   - On macOS/Linux: `source venv/bin/activate`
4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Activate the virtual environment
2. Run the application:
   ```
   python run.py
   ```
3. Open your browser and navigate to `http://localhost:5001`

## API Endpoints

The application provides several API endpoints:

- `/api/enhanced-eda` - Perform enhanced exploratory data analysis
- `/api/minimal-forecast` - Generate time series forecasts with various models
- `/api/debug-json` - Test JSON serialization with various data types
- `/upload` - Upload time series data files

## Project Structure

```
TimeSeriesAnalyzer/
├── app/                    # Application package
│   ├── __init__.py         # Initialize the app
│   ├── routes.py           # Define routes
│   ├── models.py           # Data models
│   ├── analysis.py         # Analysis functions
│   ├── error_handling.py   # Error handling utilities
│   └── dependency_check.py # Dependency validation
├── data/                   # Sample data and user uploads
├── static/                 # Static files (CSS, JS)
│   ├── css/                # CSS files
│   └── js/                 # JavaScript files
├── templates/              # HTML templates
├── tests/                  # Unit tests
│   ├── test_enhanced_eda.py       # Tests for EDA functionality
│   ├── test_debug_json.py         # Tests for JSON serialization
│   └── test_error_handling.py     # Tests for error handling
├── docs/                   # Documentation
│   ├── model_documentation.md     # Model documentation
│   └── error_handling.md          # Error handling guide
├── venv/                   # Virtual environment
├── .gitignore              # Git ignore file
├── requirements.txt        # Project dependencies
└── run.py                  # Application entry point
```

## Key Components

### JSON Serialization

The application includes a robust JSON serialization system that handles:
- NaN and Infinity values
- NumPy arrays and data types
- Pandas Series and DataFrames
- Nested data structures
- Custom objects

### Error Handling

Comprehensive error handling includes:
- Structured error responses
- Detailed logging
- User-friendly error messages
- Dependency validation

### Forecasting Models

The following forecasting models are available:
- ARIMA (Auto-Regressive Integrated Moving Average)
- SARIMA (Seasonal ARIMA)
- Exponential Smoothing
- Prophet (Facebook's time series forecasting tool)
- LSTM (Long Short-Term Memory neural networks)

## License

MIT
