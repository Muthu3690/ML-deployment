from flask import Flask, request, jsonify, render_template_string
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# =========================
# Load trained artifacts
# =========================
model = joblib.load("logistic_regression_model.joblib")
scaler = joblib.load("scaler.joblib")

# Columns used during training
categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# EXACT column order from training
model_columns = [
    'age', 'trestbps', 'chol', 'thalach', 'oldpeak',
    'sex_1',
    'cp_1', 'cp_2', 'cp_3',
    'fbs_1',
    'restecg_1', 'restecg_2',
    'exang_1',
    'slope_1', 'slope_2',
    'ca_1', 'ca_2', 'ca_3', 'ca_4',
    'thal_1', 'thal_2', 'thal_3'
]

# =========================
# Preprocessing
# =========================
def preprocess_input(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])

    # Force categorical columns to string
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # One-hot encode
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Ensure all expected columns exist
    df_encoded = df_encoded.reindex(columns=model_columns, fill_value=0)

    # Convert to float
    df_encoded = df_encoded.astype(float)

    # Scale numerical columns
    df_encoded[numerical_cols] = scaler.transform(df_encoded[numerical_cols])

    return df_encoded

# =========================
# UI Route (Browser)
# =========================
@app.route("/", methods=["GET"])
def home():
    return render_template_string("""
<!DOCTYPE html>
<html>
<head>
    <title>Heart Disease AI</title>
    <style>
        body { font-family: Arial; background: #f4f6f8; }
        .card {
            width: 420px;
            margin: 40px auto;
            padding: 20px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        input, button {
            width: 100%;
            margin: 6px 0;
            padding: 8px;
        }
        button {
            background: #007bff;
            color: white;
            border: none;
            cursor: pointer;
        }
        pre {
            background: #eee;
            padding: 10px;
            white-space: pre-wrap;
        }
    </style>
</head>
<body>

<div class="card">
    <h2>Heart Disease AI Predictor</h2>

    <input id="age" placeholder="Age">
    <input id="sex" placeholder="Sex (0=female, 1=male)">
    <input id="cp" placeholder="Chest Pain (0-3)">
    <input id="trestbps" placeholder="Resting BP">
    <input id="chol" placeholder="Cholesterol">
    <input id="fbs" placeholder="FBS (0/1)">
    <input id="restecg" placeholder="Rest ECG (0-2)">
    <input id="thalach" placeholder="Max Heart Rate">
    <input id="exang" placeholder="Exercise Angina (0/1)">
    <input id="oldpeak" placeholder="Oldpeak">
    <input id="slope" placeholder="Slope (0-2)">
    <input id="ca" placeholder="CA (0-4)">
    <input id="thal" placeholder="Thal (1-3)">

    <button onclick="predict()">Predict</button>

    <h3>AI Response</h3>
    <pre id="result">Waiting for input...</pre>
</div>

<script>
function predict() {
    const data = {
        age: Number(age.value),
        sex: Number(sex.value),
        cp: Number(cp.value),
        trestbps: Number(trestbps.value),
        chol: Number(chol.value),
        fbs: Number(fbs.value),
        restecg: Number(restecg.value),
        thalach: Number(thalach.value),
        exang: Number(exang.value),
        oldpeak: Number(oldpeak.value),
        slope: Number(slope.value),
        ca: Number(ca.value),
        thal: Number(thal.value)
    };

    fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data)
    })
    .then(res => res.json())
    .then(json => {
        document.getElementById("result").textContent =
            JSON.stringify(json, null, 2);
    })
    .catch(err => {
        document.getElementById("result").textContent = err;
    });
}
</script>

</body>
</html>
""")

# =========================
# Prediction API
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Empty JSON body"}), 400

        processed_data = preprocess_input(data)

        prediction = int(model.predict(processed_data)[0])
        probability = float(model.predict_proba(processed_data)[0][1])

        return jsonify({
            "prediction": prediction,
            "risk_probability": round(probability, 4)
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =========================
# App Entry Point
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
