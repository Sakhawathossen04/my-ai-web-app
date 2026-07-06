from flask import Flask, render_template, request, jsonify
from huggingface_hub import hf_hub_download
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# === Configuration (auto-generated) ===
HF_REPO_ID = "Sakhawathossen/my-ai-models"
MODEL_FILES = ["efficientnetb3-Eye Disease-weights.keras"]
IMAGE_SIZE = (128, 128)
CLASS_NAMES = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']

# === Model cache (একবার download হলে আর করবে না) ===
loaded_models = {}

def get_model(model_file_name):
    if model_file_name not in loaded_models:
        print(f"[INFO] Downloading {model_file_name} from HF Hub...")
        local_path = hf_hub_download(repo_id=HF_REPO_ID, filename=model_file_name)
        print("[INFO] Loading model...")
        model = tf.keras.models.load_model(local_path)
        loaded_models[model_file_name] = model
        print(f"[INFO] {model_file_name} loaded!")
    return loaded_models[model_file_name]

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert("RGB")
    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        model_name = request.form.get("model_name", MODEL_FILES[0])
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded!"}), 400
        image_file = request.files["image"]
        if image_file.filename == "":
            return jsonify({"error": "No image selected!"}), 400
        image_bytes = image_file.read()
        processed_image = preprocess_image(image_bytes)
        model = get_model(model_name)
        predictions = model.predict(processed_image)
        predicted_class_index = int(np.argmax(predictions[0]))
        predicted_class_name = CLASS_NAMES[predicted_class_index]
        confidence = float(np.max(predictions[0])) * 100
        all_probabilities = {}
        for i, class_name in enumerate(CLASS_NAMES):
            all_probabilities[class_name] = round(float(predictions[0][i]) * 100, 2)
        return jsonify({
            "success": True,
            "predicted_class": predicted_class_name,
            "confidence": round(confidence, 2),
            "all_probabilities": all_probabilities,
            "model_used": model_name
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/models", methods=["GET"])
def get_models():
    return jsonify({"models": MODEL_FILES, "classes": CLASS_NAMES})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
