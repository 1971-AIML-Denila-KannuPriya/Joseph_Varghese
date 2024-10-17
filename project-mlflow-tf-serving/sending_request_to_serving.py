import requests
import json

# Example input data (replace with real input features)
# Ensure the input shape matches the model's expected input shape
data = json.dumps({"instances": [[6.0, 148.0, 72.0, 35.0, 0.0, 33.6, 0.627, 50.0]]})  # Update the values as needed
headers = {"content-type": "application/json"}

# Send a request to TensorFlow Serving
model_name = 'my_model'  # Change to your model's name if needed
url = f'http://localhost:8501/v1/models/{model_name}:predict'

try:
    response = requests.post(url, data=data, headers=headers)
    response.raise_for_status()  # Raise an error for bad responses
    print(response.json())  # Print the prediction result
except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")
