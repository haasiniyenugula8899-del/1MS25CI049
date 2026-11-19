
import os
import json
import time
import random
import pandas as pd
import mlflow
import torch
import transformers
import sklearn
import matplotlib.pyplot as plt

from pathlib import Path
from transformers import pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay
)

# ---------------- CONFIGURATION ----------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

SEED = 42
TASK = "sentiment-analysis"
DEVICE = -1

STUDENT_USN = "1MS25CI049"

MODELS = {
    "A": "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    "B": "textattack/bert-base-uncased-SST-2"
}

# Champion rule declared before evaluation:
# Model B replaces Model A only if its macro-F1 is at least
# 0.01 higher. Otherwise, Model A remains Champion.

IMPROVEMENT_MARGIN = 0.01

random.seed(SEED)
torch.manual_seed(SEED)

# SQLite database supports MLflow Model Registry and aliases.
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Champion_Challenger_Sentiment")


# ---------------- LABEL CONVERSION ----------------

def convert_label(label, model_config):
    label = label.upper()

    if label in ["POSITIVE", "LABEL_1"]:
        return 1

    if label in ["NEGATIVE", "LABEL_0"]:
        return 0

    for key, value in model_config.id2label.items():
        if str(value).upper() == label:
            return int(key)

    raise ValueError(f"Unknown model label: {label}")


# ---------------- EVALUATION ----------------

def evaluate_model(role, model_id, texts, actual):

    print(f"\nLoading model {role}: {model_id}")

    classifier = pipeline(
        TASK,
        model=model_id,
        device=DEVICE
    )

    predictions = []
    latencies = []

    for sentence in texts:
        start = time.perf_counter()
        result = classifier(sentence, truncation=True)[0]
        latency = time.perf_counter() - start

        latencies.append(latency)

        prediction = convert_label(
            result["label"],
            classifier.model.config
        )

        predictions.append(prediction)

    total_time = sum(latencies)

    metrics = {
        "accuracy": float(accuracy_score(actual, predictions)),
        "precision": float(precision_score(actual, predictions, average="macro", zero_division=0)),
        "recall": float(recall_score(actual, predictions, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(actual, predictions, average="macro", zero_division=0)),
        "average_latency_seconds": float(total_time / len(texts)),
        "total_inference_time_seconds": float(total_time)
    }

    run_name = f"{role}_sentiment_" + (
        "distilbert" if role == "A" else "textattack"
    )

    model_dir = OUTPUT_DIR / f"model_{role}"
    model_dir.mkdir(exist_ok=True)

    classifier.model.save_pretrained(model_dir)
    classifier.tokenizer.save_pretrained(model_dir)

    with mlflow.start_run(run_name=run_name, nested=True) as run:

        mlflow.log_params({
            "hf_model_id": model_id,
            "task": TASK,
            "role": role,
            "device": "CPU",
            "seed": SEED,
            "eval_sample_count": len(texts),
            "inference_settings": "truncation=True",
            "improvement_margin": IMPROVEMENT_MARGIN
        })

        mlflow.set_tags({
            "student_usn": STUDENT_USN,
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "mlflow_version": mlflow.__version__,
            "sklearn_version": sklearn.__version__,
            "dataset_name": "custom_sentiment_evaluation"
        })

        mlflow.log_metrics(metrics)

        prediction_df = pd.DataFrame({
            "text": texts,
            "actual": actual,
            "predicted": predictions
        })

        prediction_file = OUTPUT_DIR / f"predictions_{role}.csv"
        prediction_df.to_csv(prediction_file, index=False)

        mlflow.log_artifact(str(prediction_file))

        # Save model in MLflow as a loadable PyFunc model.
        class SentimentModel(mlflow.pyfunc.PythonModel):

            def load_context(self, context):
                self.classifier = pipeline(
                    TASK,
                    model=context.artifacts["model_dir"],
                    device=-1
                )

            def predict(self, context, model_input, params=None):
                texts = model_input["text"].tolist()
                results = self.classifier(texts, truncation=True)

                return pd.DataFrame([
                    {
                        "text": text,
                        "prediction": result["label"],
                        "confidence": float(result["score"])
                    }
                    for text, result in zip(texts, results)
                ])

        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=SentimentModel(),
            artifacts={"model_dir": str(model_dir.resolve())},
            pip_requirements=[
                "mlflow",
                "transformers",
                "torch",
                "pandas"
            ]
        )

        return {
            "role": role,
            "model_id": model_id,
            "metrics": metrics,
            "run_id": run.info.run_id,
            "model_uri": f"runs:/{run.info.run_id}/model",
            "predictions": predictions
        }


# ---------------- MAIN EXPERIMENT ----------------

def main():

    print("CHAMPION-CHALLENGER EXPERIMENT")

    data = pd.read_csv(DATA_DIR / "eval_inputs.csv")

    texts = data["text"].astype(str).tolist()
    actual = data["label"].astype(int).tolist()

    results = []

    with mlflow.start_run(run_name="compare_sentiment") as parent_run:

        mlflow.log_params({
            "task": TASK,
            "seed": SEED,
            "dataset": "custom_sentiment_evaluation",
            "champion_rule": "B replaces A only if macro-F1 improves by at least 0.01"
        })

        # Same evaluation inputs and order for both models.
        for role, model_id in MODELS.items():
            results.append(
                evaluate_model(role, model_id, texts, actual)
            )

        # Fixed Champion rule.
        model_a = next(r for r in results if r["role"] == "A")
        model_b = next(r for r in results if r["role"] == "B")

        if model_b["metrics"]["macro_f1"] >= (
            model_a["metrics"]["macro_f1"] + IMPROVEMENT_MARGIN
        ):
            champion = model_b
            challenger = model_a
        else:
            champion = model_a
            challenger = model_b

        # Comparison table.
        comparison = pd.DataFrame([
            {
                "role": r["role"],
                "model": r["model_id"],
                **r["metrics"]
            }
            for r in results
        ])

        comparison_file = OUTPUT_DIR / "comparison_results.csv"
        comparison.to_csv(comparison_file, index=False)
        mlflow.log_artifact(str(comparison_file))

        # Disagreement examples.
        disagreements = []

        for i in range(len(texts)):
            if model_a["predictions"][i] != model_b["predictions"][i]:
                disagreements.append({
                    "text": texts[i],
                    "actual": actual[i],
                    "model_A_prediction": model_a["predictions"][i],
                    "model_B_prediction": model_b["predictions"][i]
                })

        disagreement_file = OUTPUT_DIR / "disagreement_examples.csv"

        pd.DataFrame(
            disagreements[:5],
            columns=[
                "text", "actual",
                "model_A_prediction", "model_B_prediction"
            ]
        ).to_csv(disagreement_file, index=False)

        mlflow.log_artifact(str(disagreement_file))

        # Confusion matrix for Champion.
        champion_result = confusion_matrix(
            actual, champion["predictions"], labels=[0, 1]
        )

        display = ConfusionMatrixDisplay(
            confusion_matrix=champion_result,
            display_labels=["Negative", "Positive"]
        )

        display.plot()
        plt.title("Champion Confusion Matrix")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "confusion_matrix.png")
        plt.close()

        # Model comparison plot.
        comparison.plot(
            x="role",
            y=["accuracy", "macro_f1"],
            kind="bar"
        )

        plt.title("Champion-Challenger Model Comparison")
        plt.ylabel("Score")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "model_comparison.png")
        plt.close()

        mlflow.log_artifact(str(OUTPUT_DIR / "confusion_matrix.png"))
        mlflow.log_artifact(str(OUTPUT_DIR / "model_comparison.png"))

        # Save final experiment information.
        summary = {
            "champion": champion["model_id"],
            "challenger": challenger["model_id"],
            "champion_run_id": champion["run_id"],
            "challenger_run_id": challenger["run_id"],
            "champion_model_uri": champion["model_uri"],
            "challenger_model_uri": challenger["model_uri"],
            "champion_rule": "Model B replaces Model A only if macro-F1 improves by at least 0.01",
            "improvement_margin": IMPROVEMENT_MARGIN,
            "results": [
                {
                    "role": r["role"],
                    "model_id": r["model_id"],
                    "run_id": r["run_id"],
                    "model_uri": r["model_uri"],
                    "metrics": r["metrics"]
                }
                for r in results
            ]
        }

        summary_file = OUTPUT_DIR / "metrics_summary.json"

        with open(summary_file, "w") as file:
            json.dump(summary, file, indent=4)

        mlflow.log_artifact(str(summary_file))

        mlflow.set_tags({
            "student_usn": STUDENT_USN,
            "selected_champion": champion["model_id"]
        })

        print("\nCOMPARISON RESULTS")
        print(comparison.to_string(index=False))

        print("\nSELECTED CHAMPION:", champion["model_id"])
        print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main()
