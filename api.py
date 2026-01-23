import torch
import torch.nn as nn
from torchvision import models, transforms
from flask import Flask, request, jsonify, render_template
from PIL import Image
import io
import os

# Initialize Flask app with template folder
app = Flask(__name__, template_folder='templates')

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
def load_model():
    """Load the trained ResNet18 model."""
    model = models.resnet18(pretrained=False)
    # Replace the final fc layer for binary classification (Cat vs Dog)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    
    # Load the saved weights
    model_path = os.path.join(os.path.dirname(__file__), "cats_vs_dogs_resnet18.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Initialize model
model = load_model()

# Define transforms (must match training transforms)
inference_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Class mapping
class_names = {0: "Cat", 1: "Dog"}

@app.route("/", methods=["GET"])
def home():
    """Serve the frontend HTML."""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint to predict whether an uploaded image is a cat or dog.
    
    Expected: multipart/form-data with 'image' file field
    Returns: JSON with prediction (Cat/Dog) and confidence scores
    """
    try:
        # Check if image is in the request
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        file = request.files["image"]
        
        if file.filename == "":
            return jsonify({"error": "No image selected"}), 400
        
        # Open image
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        
        # Preprocess image
        image_tensor = inference_transforms(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][prediction].item()
        
        # Prepare response
        response = {
            "prediction": class_names[prediction],
            "confidence": round(confidence * 100, 2),
            "probabilities": {
                "cat": round(probabilities[0][0].item() * 100, 2),
                "dog": round(probabilities[0][1].item() * 100, 2)
            }
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "Model is ready"}), 200

if __name__ == "__main__":
    print(f"Running on device: {device}")
    print("Starting Flask server...")
    app.run(debug=True, host="0.0.0.0", port=5000)
