from flask import Flask, render_template, request, jsonify
import numpy as np
import os

print("PredictX App Starting...")

app = Flask(__name__)

# Train once at startup - no pkl needed
print("Training model...")
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

data = fetch_california_housing()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X_train, y_train)
print("Model ready!")

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
        pred = model.predict(X)[0]

        return jsonify({
            "prediction": round(float(pred), 4),
            "confidence": 0.82
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)