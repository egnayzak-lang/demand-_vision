import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def main():
    parser = argparse.ArgumentParser(description="Build DemandVision model artifacts.")
    parser.add_argument("--data", default="retail_store_inventory.csv",
                        help="Path to retail_store_inventory.csv")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}\n"
            "Put retail_store_inventory.csv beside this script or pass --data PATH."
        )

    df = pd.read_csv(data_path)

    target = "Demand Forecast"
    if target not in df.columns:
        raise ValueError(f"Missing target column: {target}")

    X = df.drop(columns=[target]).copy()
    y = df[target].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Match the notebook: derive date features before encoding.
    for data in [X_train, X_test]:
        data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
        data["Year"] = data["Date"].dt.year
        data["Month"] = data["Date"].dt.month
        data["Day"] = data["Date"].dt.day
        data["DayOfWeek"] = data["Date"].dt.dayofweek

    X_train = X_train.drop(columns=["Date"])
    X_test = X_test.drop(columns=["Date"])

    # Train-derived categorical imputation.
    categorical_cols = X_train.select_dtypes(
        include=["object", "str"]
    ).columns.tolist()

    for col in categorical_cols:
        mode = X_train[col].mode()
        fill_value = mode.iloc[0] if not mode.empty else "Unknown"
        X_train[col] = X_train[col].fillna(fill_value)
        X_test[col] = X_test[col].fillna(fill_value)

    # Numeric safety if the source contains missing numeric values.
    for col in X_train.select_dtypes(include=np.number).columns:
        median = X_train[col].median()
        X_train[col] = X_train[col].fillna(median)
        X_test[col] = X_test[col].fillna(median)

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    train_encoded = encoder.fit_transform(X_train[categorical_cols])
    test_encoded = encoder.transform(X_test[categorical_cols])

    train_encoded = pd.DataFrame(
        train_encoded,
        columns=encoder.get_feature_names_out(categorical_cols),
        index=X_train.index
    )
    test_encoded = pd.DataFrame(
        test_encoded,
        columns=encoder.get_feature_names_out(categorical_cols),
        index=X_test.index
    )

    X_train = pd.concat(
        [X_train.drop(columns=categorical_cols), train_encoded], axis=1
    )
    X_test = pd.concat(
        [X_test.drop(columns=categorical_cols), test_encoded], axis=1
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    out = Path(__file__).resolve().parent / "model"
    out.mkdir(exist_ok=True)

    joblib.dump(model, out / "ai_model.pkl")
    joblib.dump(encoder, out / "encoder.pkl")
    joblib.dump(scaler, out / "scaler.pkl")
    joblib.dump(X_train.columns.tolist(), out / "feature_columns.pkl")
    joblib.dump(categorical_cols, out / "categorical_cols.pkl")
    joblib.dump(df, out / "data.pkl")

    print("\n=== DemandVision ===")
    print("Model trained and saved.")
    print(f"R2   : {r2:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"Features after encoding: {X_train.shape[1]}")
    print(f"Rows: {len(df)}")

if __name__ == "__main__":
    main()
