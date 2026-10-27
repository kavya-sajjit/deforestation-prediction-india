"""
Evaluation Metrics for Satellite Deforestation Classification.
"""

from typing import Dict, List, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_probs: Optional[List[float]] = None,
) -> Dict[str, float]:
    """Computes comprehensive binary classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
    }

    if y_probs is not None:
        try:
            auc = roc_auc_score(y_true, y_probs)
            metrics["roc_auc"] = float(auc)
        except Exception:
            metrics["roc_auc"] = 0.0

    return metrics


def get_confusion_matrix(y_true: List[int], y_pred: List[int]) -> np.ndarray:
    """Returns the 2x2 confusion matrix."""
    return confusion_matrix(y_true, y_pred)
