import numpy as np
from sklearn.metrics import f1_score


def optimize_f1(x: float, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return -f1_score(y_true, y_pred >= x)
