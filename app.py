from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import numpy as np
import cv2

app = Flask(__name__)

# Load the model (from your converted notebook code)
model = load_model("braintumor.h5")

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Process uploaded image
        file = request.files["file"]
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (150, 150))  # Resize as needed
        img = np.expand_dims(img, axis=0)

        # Predict using the loaded model
        predictions = model.predict(img)
        class_index = np.argmax(predictions)

        # Map index to tumor type
        classes = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
        result = classes[class_index]

        return f"The uploaded image is classified as: {result}"

if __name__ == "__main__":
    app.run(debug=True)
