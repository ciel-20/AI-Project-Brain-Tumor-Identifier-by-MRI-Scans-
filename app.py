from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Define the upload folder path
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create the upload folder if it does not exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the trained model
model = load_model("braintumor.h5")

@app.route("/")
def home():
    return render_template("home.html")  # Make sure `home.html` exists in the templates folder

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        file = request.files.get("file")
        if not file:
            return "No file uploaded", 400

        try:
            # Save uploaded file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Read and process the image
            img = cv2.imread(file_path)
            if img is None:
                return "The uploaded file is not a valid image.", 400

            img = cv2.resize(img, (150, 150))
            img = np.expand_dims(img, axis=0)

            # Predict
            predictions = model.predict(img)
            class_index = np.argmax(predictions)

            # Map index to tumor type
            classes = ["GliomaTumor", "MeningiomaTumor", "NoTumor", "PituitaryTumor"]
            result = classes[class_index]

            return render_template(
                "result.html",
                classification=result,
                uploaded_image=f"/static/uploads/{filename}"
            )

        except Exception as e:
            return f"Error processing the image: {str(e)}", 500

if __name__ == "__main__":
    app.run(debug=True)
