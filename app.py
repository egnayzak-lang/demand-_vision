from flask import Flask, render_template, request, jsonify
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "model"

app = Flask(__name__)

def load_artifacts():
    names = ["ai_model.pkl", "encoder.pkl", "scaler.pkl", "feature_columns.pkl",
             "categorical_cols.pkl", "data.pkl"]
    missing = [n for n in names if not (MODEL_DIR / n).exists()]
    if missing:
        return None, missing
    return {
        "model": joblib.load(MODEL_DIR / "ai_model.pkl"),
        "encoder": joblib.load(MODEL_DIR / "encoder.pkl"),
        "scaler": joblib.load(MODEL_DIR / "scaler.pkl"),
        "features": joblib.load(MODEL_DIR / "feature_columns.pkl"),
        "categorical": joblib.load(MODEL_DIR / "categorical_cols.pkl"),
        "df": joblib.load(MODEL_DIR / "data.pkl"),
    }, []

ARTIFACTS, MISSING = load_artifacts()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/health")
def health():
    return jsonify({
        "online": ARTIFACTS is not None,
        "missing_files": MISSING
    })

@app.route("/api/stats")
def stats():
    if ARTIFACTS is None:
        return jsonify({"error": "Model files are missing.", "missing_files": MISSING}), 503

    df = ARTIFACTS["df"]
    return jsonify({
        "rows": int(len(df)),
        "products": int(df["Product ID"].nunique()),
        "avg_demand": round(float(df["Demand Forecast"].mean()), 2),
        "max_demand": round(float(df["Demand Forecast"].max()), 2),
        "model_r2": 0.9935,
        "mae": 7.5639,
        "rmse": 8.8319
    })

@app.route("/api/products")
def products():
    if ARTIFACTS is None:
        return jsonify({"error": "Model files are missing.", "missing_files": MISSING}), 503

    df = ARTIFACTS["df"]
    ids = df["Product ID"].astype(str).drop_duplicates().tolist()
    return jsonify({"products": ids[:500]})

def make_features(row):
    encoder = ARTIFACTS["encoder"]
    categorical_cols = ARTIFACTS["categorical"]
    feature_columns = ARTIFACTS["features"]
    scaler = ARTIFACTS["scaler"]
    model = ARTIFACTS["model"]

    product_data = row.drop(labels=["Demand Forecast"]).to_frame().T

    product_data["Date"] = pd.to_datetime(product_data["Date"], errors="coerce")
    product_data["Year"] = product_data["Date"].dt.year
    product_data["Month"] = product_data["Date"].dt.month
    product_data["Day"] = product_data["Date"].dt.day
    product_data["DayOfWeek"] = product_data["Date"].dt.dayofweek
    product_data = product_data.drop(columns=["Date"])

    for col in categorical_cols:
        if product_data[col].isna().any():
            # The training notebook learns categorical fill values from Train.
            # Saved encoder categories handle the known vocabulary safely.
            product_data[col] = product_data[col].fillna("Unknown")

    encoded = encoder.transform(product_data[categorical_cols])
    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(categorical_cols),
        index=product_data.index
    )

    product_data = pd.concat(
        [product_data.drop(columns=categorical_cols), encoded_df],
        axis=1
    )
    product_data = product_data.reindex(columns=feature_columns, fill_value=0)

    scaled = scaler.transform(product_data)
    return float(model.predict(scaled)[0])

@app.route("/predict", methods=["POST"])
def predict():
    if ARTIFACTS is None:
        return jsonify({
            "error": "Model files are missing. Run prepare_model.py first.",
            "missing_files": MISSING
        }), 503

    body = request.get_json(silent=True) or {}
    product_id = str(body.get("product_id", "")).strip()

    if not product_id:
        return jsonify({"error": "Please enter a Product ID."}), 400

    df = ARTIFACTS["df"]
    matches = df[df["Product ID"].astype(str).str.lower() == product_id.lower()]

    if matches.empty:
        examples = df["Product ID"].astype(str).drop_duplicates().head(10).tolist()
        return jsonify({
            "error": "Product not found.",
            "examples": examples
        }), 404

    row = matches.iloc[0]
    predicted = make_features(row)
    actual = float(row["Demand Forecast"])
    inventory = float(row["Inventory Level"])

    if inventory < predicted:
        recommendation = "Consider restocking."
        status = "RESTOCK"
    else:
        recommendation = "Current inventory may be sufficient."
        status = "STABLE"

    return jsonify({
        "product_id": str(row["Product ID"]),
        "predicted_demand": round(predicted, 2),
        "actual_demand": round(actual, 2),
        "inventory": round(inventory, 2),
        "units_sold": round(float(row["Units Sold"]), 2),
        "units_ordered": round(float(row["Units Ordered"]), 2),
        "price": round(float(row["Price"]), 2),
        "status": status,
        "recommendation": recommendation
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
