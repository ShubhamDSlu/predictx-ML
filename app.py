from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib

print("PredictX App Starting...")

app = Flask(__name__)

model = joblib.load("rf_pipeline_model.pkl")
print("Model loaded successfully")

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