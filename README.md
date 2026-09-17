# DemandVision

A futuristic open-source frontend + Flask API for the DemandVision retail demand forecasting model.

## Architecture

Frontend (HTML/CSS/JS)
        ↓
Flask API
        ↓
Preprocessing artifacts
        ↓
RandomForestRegressor
        ↓
Demand prediction + inventory recommendation

## 1. Install

```bash
pip install -r requirements.txt
```

## 2. Put the dataset in the project folder

The training notebook uses:

`retail_store_inventory.csv`

Expected target:

`Demand Forecast`

## 3. Build the model artifacts

```bash
python prepare_model.py --data retail_store_inventory.csv
```

This creates:

- model/ai_model.pkl
- model/encoder.pkl
- model/scaler.pkl
- model/feature_columns.pkl
- model/categorical_cols.pkl
- model/data.pkl

## 4. Start the website

```bash
python app.py
```

Open:

http://127.0.0.1:5000

## Test

Try Product ID:

`P0020`

The API endpoint is:

`POST /predict`

JSON body:

```json
{"product_id":"P0020"}
```

## Model

The project follows the uploaded DemandVision notebook structure:

- train/test split with test_size=0.2 and random_state=42
- Date → Year / Month / Day / DayOfWeek
- train-derived categorical filling
- OneHotEncoder(handle_unknown="ignore")
- StandardScaler
- RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)

The notebook's reported evaluation was approximately:

- R²: 0.9935
- MAE: 7.5639
- RMSE: 8.8319

## License

MIT
