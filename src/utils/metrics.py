import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix
)
from typing import Dict


def compute_classification_metrics(y_true, y_pred, average: str = "weighted") -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }


def compute_regression_metrics(y_true, y_pred) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2_score": float(r2_score(y_true, y_pred)),
        "mape": float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100),
    }


def compute_forecast_metrics(y_true, y_pred) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "mape": float(mape),
        "mse": float(mse),
    }


def compute_ner_metrics(y_true, y_pred) -> Dict[str, float]:
    from seqeval.metrics import precision_score, recall_score, f1_score
    return {
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred)),
    }
