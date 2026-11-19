
import json
import mlflow
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Read registered model information
with open(BASE_DIR / "outputs" / "metrics_summary.json") as file:
    summary = json.load(file)

model_name = summary["registered_model_name"]

# Load Champion directly from MLflow Registry
model_uri = f"models:/{model_name}@champion"

print("Loading Champion:", model_uri)

champion = mlflow.pyfunc.load_model(model_uri)

# Load new, unseen sentences
new_data = pd.read_csv(BASE_DIR / "data" / "new_data.csv")

# Generate predictions
predictions = champion.predict(new_data)

# Save results
output_path = BASE_DIR / "outputs" / "new_data_predictions.csv"
predictions.to_csv(output_path, index=False)

print("\nNEW DATA PREDICTIONS")
print(predictions.to_string(index=False))

print("\nSUCCESS! Champion loaded from MLflow.")
print("Predictions saved to:", output_path)
