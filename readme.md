# Champion–Challenger Model Comparison

## 1. Project Overview

This project compares two Hugging Face sentiment-analysis models using the same labelled evaluation dataset. MLflow is used to track experiments, compare model performance, register models and load the selected Champion for predictions on new data.

## 2. Models Used

- Model A: `distilbert/distilbert-base-uncased-finetuned-sst-2-english`
- Model B: `textattack/bert-base-uncased-SST-2`

Task: Sentiment classification  
Dataset: Custom sentiment evaluation dataset  
Evaluation samples: 20  
Device: CPU  
Random seed: 42

## 3. Champion Selection Rule

The selection rule was declared before evaluation.

Model B replaces Model A only if its Macro-F1 score is at least 0.01 higher. Otherwise, Model A remains Champion.

## 4. Experimental Results

| Metric | DistilBERT | TextAttack BERT |
|---|---:|---:|
| Accuracy | 1.00 | 1.00 |
| Precision | 1.00 | 1.00 |
| Recall | 1.00 | 1.00 |
| Macro-F1 | 1.00 | 1.00 |
| Average latency | 0.0358 s | 0.0792 s |
| Total inference time | 0.7163 s | 1.5848 s |

Selected Champion: DistilBERT  
Challenger: TextAttack BERT

## 5. Champion Decision

Both models achieved an accuracy, precision, recall and Macro-F1 score of 1.00 on the evaluation dataset. Therefore, neither model achieved the required 0.01 Macro-F1 improvement over the other.

According to the selection rule declared before the experiment, Model A remained the Champion. DistilBERT also had a lower average inference latency of approximately 0.0358 seconds, compared with 0.0792 seconds for TextAttack BERT. Its total inference time was also lower.

Based on these results, DistilBERT was selected as the Champion because it provided the same measured classification performance with faster inference on the tested inputs. Both models were registered using MLflow, and Champion and Challenger aliases were assigned.

## 6. New Data Prediction

The registered Champion was loaded from MLflow using its Champion alias. It was then used to predict sentiment for eight new sentences that were not part of the evaluation dataset.

Predictions were saved in `outputs/new_data_predictions.csv`.

## 7. Limitations

The evaluation used only 20 custom-labelled sentences. The results may not represent performance on a larger or more varied real-world dataset. The experiment was conducted on CPU, so latency may differ on other hardware.

## 8. How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the evaluation:

```bash
python evaluate.py
```

Register the models:

```bash
python register_champion.py
```

Predict using the saved Champion:

```bash
python predict_from_mlflow.py
```

## 9. AI Declaration

AI assistance was used to help develop and organise parts of the project code and documentation. The experiments were executed and the reported results were obtained from the actual program output.
