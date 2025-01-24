from abc import ABC, abstractmethod
from typing import Any
import numpy as np

METRICS = [
    "mean_squared_error",
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "mean_absolute_error",
]

def get_metric(name: str):
    """
    Factory function to get a metric by name.

    Args:
        name (str): Name of the metric.

    Returns:
        Metric: Instance of a Metric class.

    Raises:
        ValueError: If the metric name is not found.
    """
    metrics_map = {
        "mean_squared_error": MeanSquaredError(),
        "accuracy": Accuracy(),
        "precision": Precision(),
        "recall": Recall(),
        "f1_score": F1Score(),
        "mean_absolute_error": MeanAbsoluteError(),
    }
    if name not in metrics_map:
        raise ValueError(f"Metric '{name}' is not found. Available metrics: {list(metrics_map.keys())}")
    return metrics_map[name]

class Metric(ABC):
    """
    Base class for all metrics.

    Methods:
        __call__: Abstract method to compute the metric.
    """
    @abstractmethod
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the metric.

        Args:
            y_true (np.ndarray): Ground truth values.
            y_pred (np.ndarray): Predicted values.

        Returns:
            float: The computed metric value.
        """
        pass
    
class MeanSquaredError(Metric):
    """
    Mean Squared Error (MSE).
    """
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean((y_true - y_pred) ** 2)

class Accuracy(Metric):
    """
    Accuracy for classification.
    """
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(y_true == y_pred)

# Extended Metrics

class Precision(Metric):
    """
    Precision.

    Measures the proportion of true positives out of all positive predictions.
    For binary or multi-class classification tasks.
    """
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

class Recall(Metric):
    """
    Recall.

    Measures the proportion of true positives out of all actual positives.
    For binary or multi-class classification tasks.
    """
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

class F1Score(Metric):
    """
    F1 Score.

    Harmonic mean of precision and recall.
    For binary or multi-class classification tasks.
    """
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        precision = Precision()(y_true, y_pred)
        recall = Recall()(y_true, y_pred)
        return (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

class MeanAbsoluteError(Metric):
    """
    Mean Absolute Error (MAE).

    Calculates the average absolute difference between ground truth and predictions.
    For regression tasks.
    """
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(np.abs(y_true - y_pred))