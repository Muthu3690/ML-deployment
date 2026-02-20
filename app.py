from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load trained artifacts
model = joblib.load("logistic_regression_model.joblib")
scaler = joblib.load("scaler.joblib")

# Columns used during training
categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# EXACT column order used during training (X_encoded.columns.tolist())
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


def preprocess_input(data: dict) -> pd.DataFrame:
    """
    Convert raw JSON input into model-ready dataframe
    """

    # Convert to DataFrame
    df = pd.DataFrame([data])

    # Ensure categorical columns are treated correctly
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # One-hot encode categorical features
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Ensure ALL expected columns exist (missing ones set to 0)
    df_encoded = df_encoded.reindex(columns=model_columns, fill_value=0)

    # Convert all data to numeric
    df_encoded = df_encoded.astype(float)

    # Scale numerical columns
    df_encoded[numerical_cols] = scaler.transform(df_encoded[numerical_cols])

    return df_encoded


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid or empty JSON input"}), 400

        processed_data = preprocess_input(data)

        prediction = model.predict(processed_data)[0]
        probability = model.predict_proba(processed_data)[0][1]

        return jsonify({
            "prediction": int(prediction),
            "risk_probability": round(float(probability), 4)
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
