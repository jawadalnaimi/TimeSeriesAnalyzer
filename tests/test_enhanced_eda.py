"""
Test script for the enhanced EDA endpoint.
This script sends a test request to the enhanced EDA endpoint and prints the response.
"""
import requests
import json
import sys
import os

# Add the parent directory to the path so we can import from app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_enhanced_eda():
    """Test the enhanced EDA endpoint with a sample request."""
    url = "http://localhost:5001/api/enhanced-eda"
    
    # Use a test file ID - this will trigger the synthetic data generation
    payload = {
        "file_id": "test_file_id"
    }
    
    print("Sending request to enhanced EDA endpoint...")
    response = requests.post(url, json=payload)
    
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
            else:
                print("Response does not contain expected keys")
                print(json.dumps(result, indent=2))
        except json.JSONDecodeError:
            print("Response is not valid JSON")
            print(response.text[:1000])  # Print first 1000 chars of response
    else:
        print("Error response:")
        try:
            print(json.dumps(response.json(), indent=2))
        except:
            print(response.text[:1000])  # Print first 1000 chars of response

if __name__ == "__main__":
    test_enhanced_eda()
