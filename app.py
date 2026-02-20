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
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Heart Disease AI Predictor</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <style>
        body {
            font-family: Arial, sans-serif;
            background: #f4f6f8;
        }
        .card {
            max-width: 480px;
            margin: 40px auto;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        h2 {
            text-align: center;
        }
        label {
            font-weight: bold;
            margin-top: 10px;
            display: block;
        }
        input, select, button {
            width: 100%;
            padding: 8px;
            margin-top: 5px;
        }
        button {
            margin-top: 15px;
            background: #007bff;
            color: white;
            border: none;
            font-size: 16px;
            cursor: pointer;
            border-radius: 5px;
        }
        button:hover {
            background: #0056b3;
        }
        pre {
            background: #eee;
            padding: 10px;
            margin-top: 10px;
            border-radius: 5px;
            white-space: pre-wrap;
        }
    </style>
</head>

<body>

<div class="card">
    <h2>❤️ Heart Disease AI Predictor</h2>

    <label>Age</label>
    <input type="number" id="age" value="55" min="1" max="120">

    <label>Sex</label>
    <select id="sex">
        <option value="0">Female</option>
        <option value="1" selected>Male</option>
    </select>

    <label>Chest Pain Type</label>
    <select id="cp">
        <option value="0">0 - Typical Angina</option>
        <option value="1">1 - Atypical Angina</option>
        <option value="2" selected>2 - Non-anginal Pain</option>
        <option value="3">3 - Asymptomatic</option>
    </select>

    <label>Resting Blood Pressure (mm Hg)</label>
    <input type="number" id="trestbps" value="120" min="80" max="250">

    <label>Cholesterol (mg/dl)</label>
    <input type="number" id="chol" value="200" min="100" max="600">

    <label>Fasting Blood Sugar</label>
    <select id="fbs">
        <option value="0" selected>≤ 120 mg/dl</option>
        <option value="1">> 120 mg/dl</option>
    </select>

    <label>Rest ECG</label>
    <select id="restecg">
        <option value="0" selected>Normal</option>
        <option value="1">ST-T Abnormality</option>
        <option value="2">Left Ventricular Hypertrophy</option>
    </select>

    <label>Max Heart Rate</label>
    <input type="number" id="thalach" value="150" min="60" max="250">

    <label>Exercise Induced Angina</label>
    <select id="exang">
        <option value="0" selected>No</option>
        <option value="1">Yes</option>
    </select>

    <label>Oldpeak (ST Depression)</label>
    <input type="number" step="0.1" id="oldpeak" value="1.0" min="0" max="10">

    <label>Slope</label>
    <select id="slope">
        <option value="0">Upsloping</option>
        <option value="1" selected>Flat</option>
        <option value="2">Downsloping</option>
    </select>

    <label>Number of Major Vessels (CA)</label>
    <select id="ca">
        <option value="0" selected>0</option>
        <option value="1">1</option>
        <option value="2">2</option>
        <option value="3">3</option>
        <option value="4">4</option>
    </select>

    <label>Thalassemia</label>
    <select id="thal">
        <option value="1">Normal</option>
        <option value="2" selected>Fixed Defect</option>
        <option value="3">Reversible Defect</option>
    </select>

    <button onclick="predict()">Predict</button>

    <h3>AI Response</h3>
    <pre id="result">Waiting for prediction...</pre>
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
