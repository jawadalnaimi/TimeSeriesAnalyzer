"""
Test script for the debug EDA endpoint.
This script sends a test request to the debug EDA endpoint and verifies that NaN values are handled correctly.
"""
import requests
import json
import sys
import os

# Add the parent directory to the path so we can import from app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_debug_eda():
    """Test the debug EDA endpoint to verify NaN handling."""
    url = "http://localhost:5001/api/debug-eda"
    
    print("Sending request to debug EDA endpoint...")
    response = requests.post(url, json={})
    
    print(f"Status code: {response.status_code}")
    
    if response.status_code == 200:
        # Try to parse the response as JSON
        try:
            result = response.json()
            print("Response successfully parsed as JSON")
            
            # Check if the response contains the expected keys
            if 'success' in result and result['success'] and 'results' in result:
                print("Response contains expected keys")
                
                # Print the keys in the results
                results = result['results']
                print(f"Results contains the following sections: {list(results.keys())}")
                
                # Check each section
                for section, data in results.items():
                    if data:
                        print(f"Section '{section}' contains data")
                    else:
                        print(f"Section '{section}' is empty")
                
                # Specifically check for NaN values in the summary_stats section
                if 'summary_stats' in results and results['summary_stats']:
                    print("\nChecking for NaN values in summary_stats:")
                    for column, stats in results['summary_stats'].items():
                        print(f"  Column: {column}")
                        if stats:
                            # In a properly serialized response, NaN values should be converted to null (None in Python)
                            null_values = [k for k, v in stats.items() if v is None]
                            if null_values:
                                print(f"    Contains null values for: {', '.join(null_values)}")
                            else:
                                print(f"    No null values found")
                        else:
                            print(f"    No statistics available")
            else:
                print("Response does not contain expected keys")
                print(json.dumps(result, indent=2))
        except json.JSONDecodeError as e:
            print(f"Response is not valid JSON: {e}")
            print(response.text[:1000])  # Print first 1000 chars of response
    else:
        print("Error response:")
        try:
            print(json.dumps(response.json(), indent=2))
        except:
            print(response.text[:1000])  # Print first 1000 chars of response

if __name__ == "__main__":
    test_debug_eda()
