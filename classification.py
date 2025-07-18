import os
import cv2
import numpy as np
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

# --- Configuration ---
# Directory where your model components are saved
MODEL_DIR = 'deployed_model_cats_dogs'
# Expected image size (width, height) - MUST match your training preprocessing
IMG_SIZE = (64, 64)

# --- Load the saved model components ---
model = None
scaler = None
pca = None
try:
    model_path = os.path.join(MODEL_DIR, 'svc_cat_dog_model.pkl')
    scaler_path = os.path.join(MODEL_DIR, 'minmax_scaler.pkl')
    pca_path = os.path.join(MODEL_DIR, 'pca_transformer.pkl')

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    pca = joblib.load(pca_path)

    print("INFO: Model, scaler, and PCA transformer loaded successfully!")
except Exception as e:
    print(f"ERROR: Could not load model components: {e}")
    print("Please ensure 'deployed_model_cats_dogs' folder exists and contains 'svc_cat_dog_model.pkl', 'minmax_scaler.pkl', 'pca_transformer.pkl'.")


# --- API Routes ---

@app.route('/')
def home():
    print("DEBUG: / (Home) route hit - GET request received.")
    return "Cat vs Dog Classification API. Send POST requests to /predict."

@app.route('/predict', methods=['POST'])
def predict():
    print("DEBUG: /predict route hit - POST request received.")

    if model is None or scaler is None or pca is None:
        print("DEBUG: Model components not available for prediction.")
        return jsonify({"error": "Server is not ready: Model components not loaded."}), 500

    if 'image' not in request.files:
        print("DEBUG: No 'image' file key found in request.files.")
        return jsonify({"error": "No image file provided. Please upload an image with the key 'image'."}), 400

    file = request.files['image']
    if file.filename == '':
        print("DEBUG: Empty filename provided for 'image' file.")
        return jsonify({"error": "No selected file."}), 400

    try:
        # Read the image file directly from the request stream
        np_image = np.frombuffer(file.read(), np.uint8)
        # Decode the image in color
        image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

        if image is None:
            print("DEBUG: Could not decode image data.")
            return jsonify({"error": "Could not decode image. Ensure it's a valid image file (e.g., JPG, PNG)."}), 400

        # Resize image to expected dimensions (64x64)
        image = cv2.resize(image, IMG_SIZE)
        
        # Flatten the image for model input
        image_flattened = image.flatten().reshape(1, -1) # Reshape for a single sample

        # Apply the exact same preprocessing steps as during training
        # 1. Scale the image data using the fitted scaler
        image_scaled = scaler.transform(image_flattened)
        
        # 2. Apply PCA transformation using the fitted pca object
        image_pca = pca.transform(image_scaled)

        # Make prediction using the loaded SVC model
        prediction_proba = model.predict_proba(image_pca) # Get probabilities
        predicted_class_index = np.argmax(prediction_proba) # Get the class with highest probability

        class_names = {0: 'Cat', 1: 'Dog'} # Map indices to class names
        predicted_class_name = class_names[predicted_class_index]
        confidence = prediction_proba[0][predicted_class_index] * 100

        print(f"DEBUG: Prediction made: {predicted_class_name} with {confidence:.2f}% confidence.")
        return jsonify({
            "prediction": predicted_class_name,
            "confidence": f"{confidence:.2f}%",
            "probabilities": {
                "Cat": f"{prediction_proba[0][0]*100:.2f}%",
                "Dog": f"{prediction_proba[0][1]*100:.2f}%"
            }
        })

    except Exception as e:
        print(f"ERROR: An exception occurred during prediction: {str(e)}")
        return jsonify({"error": f"An internal server error occurred: {str(e)}"}), 500

# --- Run the Flask App ---
if __name__ == '__main__':
    # Use 0.0.0.0 to make it accessible from other devices on your network
    # debug=True provides helpful error messages during development (turn off for production)
    app.run(debug=True, host='0.0.0.0', port=5000)