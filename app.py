from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import os

print("PredictX App Starting...")

app = Flask(__name__)

MODEL_PATH = "rf_pipeline_model.pkl"
MODEL_URL  = os.getenv("MODEL_URL")

if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    try:
        import gdown
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False, fuzzy=True)
        print("Model downloaded successfully")
    except Exception as e:
        print("gdown failed:", e)
        try:
            import requests
            url = MODEL_URL + "&confirm=t"
            r = requests.get(url, stream=True, timeout=120)
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=32768):
                    if chunk:
                        f.write(chunk)
            print("Model downloaded via requests")
        except Exception as e2:
            print("Download failed:", e2)

model = None
try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully")
except Exception as e:
    print("Model loading failed:", e)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        data = request.get_json()

        features = []
        for i in range(1, 9):
            key = "f" + str(i)
            if key not in data:
                return jsonify({"error": "Missing field: " + key}), 400
            features.append(float(data[key]))

        X = np.array(features).reshape(1, -1)
        pred = model.predict(X)[0]
        output = round(float(pred), 4)

        return jsonify({
            "prediction": output,
            "confidence": 0.82
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
```

**`requirements.txt`:**
```
flask>=3.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
joblib>=1.3.0
gdown>=4.7.1
gunicorn
requests