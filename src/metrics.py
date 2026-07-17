import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score, 
    classification_report, 
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    balanced_accuracy_score
)
import torch

import config

def classification_metrics(result):
    probs = torch.sigmoid(result["logits"]).numpy().ravel()
    preds = (probs >= config.CLASSIFICATION_THRESHOLD).astype(float)
    targets = result["targets"].numpy().ravel()

    metrics = {
        "avg_loss": result["avg_loss"],
        "accuracy": accuracy_score(targets, preds),
        "balanced_accuracy": balanced_accuracy_score(targets, preds),
        "precision": precision_score(targets, preds, zero_division=0),
        "recall": recall_score(targets, preds, zero_division=0),
        "f1": f1_score(targets, preds, zero_division=0),
        "mcc": matthews_corrcoef(targets, preds),
        "baseline": max(targets.mean(), 1 - targets.mean()),
    }

    try:
        metrics["auc"] = roc_auc_score(targets, probs)
    except ValueError:
        metrics["auc"] = np.nan

    metrics["predictions"] = preds
    metrics["probabilities"] = probs

    return metrics

def print_metrics(result, metrics, checkpoint):


    print("\n--- FINAL METRICS ---")
    print(f"BEST CHECKPOINT: Epoch {checkpoint["epoch"]} | avg_train_loss {checkpoint["train_loss"]} | avg_valid_loss {checkpoint["valid_loss"]}")
    print(
        classification_report(
            result["targets"].numpy().ravel(),
            metrics["predictions"],
            digits=4,
        )
    )