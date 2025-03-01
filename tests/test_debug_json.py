"""
Test script for the debug-json endpoint.
This script sends a request to the debug-json endpoint and verifies that NaN values are handled correctly.
"""
import requests
import json
import sys
import os

# Add the parent directory to the path so we can import from app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_debug_json():
    """Test the debug-json endpoint to verify NaN handling."""
    url = "http://localhost:5001/api/debug-json"
    
    print("Sending request to debug-json endpoint...")
    response = requests.get(url)
    
    print(f"Status code: {response.status_code}")
    
    if response.status_code == 200:
        # Try to parse the response as JSON
        try:
            result = response.json()
            print("Response successfully parsed as JSON")
            
            # Print the entire response
            print("\nFull response:")
            print(json.dumps(result, indent=2))
            
            # Check for NaN values
            print("\nChecking NaN handling:")
            
            # Check direct NaN value
            if 'nan_value' in result:
                print(f"  nan_value = {result['nan_value']} (should be null)")
            
            # Check infinity values
            if 'inf_value' in result:
                print(f"  inf_value = {result['inf_value']} (should be null)")
            if 'neg_inf_value' in result:
                print(f"  neg_inf_value = {result['neg_inf_value']} (should be null)")
            
            # Check list with NaN
            if 'list_with_nan' in result:
                print(f"  list_with_nan = {result['list_with_nan']} (should have null as third element)")
            
            # Check dict with NaN
            if 'dict_with_nan' in result:
                print(f"  dict_with_nan['b'] = {result['dict_with_nan'].get('b')} (should be null)")
            
            # Check numpy array with NaN
            if 'numpy_array_with_nan' in result:
                print(f"  numpy_array_with_nan = {result['numpy_array_with_nan']} (should have null as third element)")
            
            # Check pandas series with NaN
            if 'pandas_series_with_nan' in result:
                print(f"  pandas_series_with_nan = {result['pandas_series_with_nan']} (should have null as third element)")
            
            # Check nested dict with NaN
            if 'nested_dict' in result and 'a' in result['nested_dict'] and 'b' in result['nested_dict']['a']:
                print(f"  nested_dict['a']['b']['c'] = {result['nested_dict']['a']['b'].get('c')} (should be null)")
            
            print("\nAll NaN values should be converted to null (None in Python)")
            print("If you see 'NaN' in any of the outputs above, the serialization is not working correctly")
            
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
    test_debug_json()
