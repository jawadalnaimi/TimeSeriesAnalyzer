# Time Series Analyzer

A web-based tool for analyzing and visualizing time series data.

## Features

- Upload and process time series data in various formats (CSV, Excel, JSON)
- Interactive visualization of time series data
- Statistical analysis and anomaly detection
- Forecasting using various models (ARIMA, Prophet, etc.)
- Export results and visualizations

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
3. Open your browser and navigate to `http://localhost:5000`

## Project Structure

```
TimeSeriesAnalyzer/
├── app/                    # Application package
│   ├── __init__.py         # Initialize the app
│   ├── routes.py           # Define routes
│   ├── models.py           # Data models
│   └── analysis.py         # Analysis functions
├── data/                   # Sample data and user uploads
├── static/                 # Static files (CSS, JS)
│   ├── css/                # CSS files
│   └── js/                 # JavaScript files
├── templates/              # HTML templates
├── tests/                  # Unit tests
├── venv/                   # Virtual environment
├── .gitignore              # Git ignore file
├── requirements.txt        # Project dependencies
└── run.py                  # Application entry point
```

## License

MIT
