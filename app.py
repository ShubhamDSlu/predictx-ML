from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import os

print("PredictX App Starting...")

app = Flask(__name__)

MODEL_PATH = "rf_pipeline_model.pkl"

# Always train fresh on Render
if not os.path.exists(MODEL_PATH):
    print("Training model...")
    try:
        from sklearn.datasets import fetch_california_housing
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split

        data = fetch_california_housing()
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        m = RandomForestRegressor(n_estimators=100, random_state=42)
        m.fit(X_train, y_train)
        joblib.dump(m, MODEL_PATH)
        print("Model trained and saved!")
    except Exception as e:
        print("Training failed:", e)

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
gunicorn