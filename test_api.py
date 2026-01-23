"""
Simple test script to test the API endpoint.
"""
import requests
from pathlib import Path

# Test image path - adjust to a real image in your data folder
test_image_path = "data/test/Cat"  # or "data/test/Dog"

def test_api():
    """Test the prediction endpoint."""
    # Find a test image
    test_dir = Path(test_image_path)
    if not test_dir.exists():
        print(f"Test directory not found: {test_image_path}")
        return
    
    image_files = list(test_dir.glob("*.*"))
    if not image_files:
        print(f"No images found in {test_image_path}")
        return
    
    # Use the first image
    image_path = image_files[0]
    print(f"Testing with image: {image_path}")
    
    # Make request
    with open(image_path, "rb") as f:
        files = {"image": f}
        response = requests.post("http://localhost:5000/predict", files=files)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")

if __name__ == "__main__":
    # Test health endpoint first
    try:
        response = requests.get("http://localhost:5000/health")
        print(f"Health Check: {response.json()}")
    except Exception as e:
        print(f"Error connecting to server: {e}")
        print("Make sure the API server is running!")
        exit(1)
    
    # Test prediction
    test_api()
