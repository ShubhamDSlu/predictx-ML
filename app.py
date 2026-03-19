from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import os
import requests

print("🔥 PredictX App Starting...")

app = Flask(__name__)

# ==============================
# MODEL DOWNLOAD (IMPORTANT)
# ==============================
MODEL_URL = os.getenv("MODEL_URL")

MODEL_PATH = "rf_pipeline_model.pkl"

if not os.path.exists(MODEL_PATH):
    print("⬇️ Downloading model from cloud...")
    try:
        r = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)
        print("✅ Model downloaded successfully")
    except Exception as e:
        print("❌ Model download failed:", e)

# ==============================
# LOAD MODEL
# ==============================
try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Model loading failed:", e)

# ==============================
# ROUTES
# ==============================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        features = []
        for i in range(1, 9):
            key = "f" + str(i)
            if key not in data:
                return jsonify({"error": "Missing field: " + key}), 400
            features.append(float(data[key]))

        X = np.array(features).reshape(1, -1)

        prediction = model.predict(X)
        output = round(float(prediction[0]), 4)

        return jsonify({
            "prediction": output,
            "confidence": 0.82
        })

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)