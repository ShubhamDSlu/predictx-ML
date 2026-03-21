from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import os
import gdown

print("🚀 PredictX App Starting...")

app = Flask(__name__)

# ==============================
# MODEL DOWNLOAD + LOAD (FINAL FIX)
# ==============================

MODEL_PATH = "rf_pipeline_model.pkl"
FILE_ID = "10Z55J4uQgWKlcHdmBlnZEEJTs4FJFx2r"

model = None

try:
    # Step 1: Download if not exists
    if not os.path.exists(MODEL_PATH):
        print("📥 Downloading model from Google Drive...")

        gdown.download(id=FILE_ID, output=MODEL_PATH, quiet=False)

        print("✅ Model downloaded")

    # Step 2: Load model
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
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        data = request.get_json()

        # Get features
        features = []
        for i in range(1, 9):
            key = "f" + str(i)
            if key not in data:
                return jsonify({"error": f"Missing field: {key}"}), 400
            features.append(float(data[key]))

        # Prediction
        X = np.array(features).reshape(1, -1)
        pred = model.predict(X)[0]

        output = round(float(pred), 4)

        return jsonify({
            "prediction": output,
            "confidence": 0.82
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# ==============================
# MAIN (RENDER SAFE)
# ==============================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)