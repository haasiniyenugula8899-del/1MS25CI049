
import json
import mlflow
from pathlib import Path
from mlflow import MlflowClient

BASE_DIR = Path(__file__).resolve().parent
SUMMARY_FILE = BASE_DIR / "outputs" / "metrics_summary.json"

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

MODEL_NAME = "sentiment-model"

with open(SUMMARY_FILE, "r") as file:
    summary = json.load(file)

registered = {}

# Register both model versions under ONE registered model name.
for result in summary["results"]:

    role = result["role"]

    print(f"\nRegistering Model {role}...")

    version = mlflow.register_model(
        model_uri=result["model_uri"],
        name=MODEL_NAME
    )

    registered[role] = version.version

# Determine the winning role.
champion_role = next(
    r["role"] for r in summary["results"]
    if r["model_id"] == summary["champion"]
)

challenger_role = "B" if champion_role == "A" else "A"

# Assign aliases to the correct versions.
client.set_registered_model_alias(
    MODEL_NAME, "champion", registered[champion_role]
)

client.set_registered_model_alias(
    MODEL_NAME, "challenger", registered[challenger_role]
)

summary["registered_model_name"] = MODEL_NAME
summary["registered_versions"] = registered

with open(SUMMARY_FILE, "w") as file:
    json.dump(summary, file, indent=4)

print("\nRegistered model:", MODEL_NAME)
print("Champion version:", registered[champion_role])
print("Challenger version:", registered[challenger_role])
print("\nREGISTRATION COMPLETED!")
