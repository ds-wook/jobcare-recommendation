import logging
from typing import List

import numpy as np
from scipy.optimize import Bounds, minimize
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold


def get_score(
    weights: np.ndarray, train_idx: List[int], oofs: List[np.ndarray], preds: np.ndarray
) -> float:
    """
    Blending Models
    Parameters:
        weights: Each parameters
        train_idx: Fold's train index
        oofs: oof preds
        preds: real preds
    return:
        F1 Score
    """
    blending = np.zeros_like(oofs[0][train_idx])

    for oof, weights in zip(oofs[:-1], weights):
        blending += weights * oof[train_idx]

    blending += (1 - np.sum(weights)) * oofs[-1][train_idx]
    scores = -f1_score(preds[train_idx], blending > 0.4)

    return scores


def get_best_weights(oofs: List[np.ndarray], preds: List[np.ndarray]) -> np.ndarray:
    """
    Optimized weight with Gradient method
    Parameters:
        oofs: oof preds
        preds: real preds
    Return:
        weight's values array
    """
    weight_list = []
    weights = np.array([1 / len(oofs) for _ in range(len(oofs) - 1)])

    logging.info("Blending Start")

    kf = KFold(n_splits=5)
    for fold, (train_idx, valid_idx) in enumerate(kf.split(oofs[0])):
        result = minimize(
            get_score,
            weights,
            args=(train_idx, oofs, preds),
            method="Nelder-Mead",
            bounds=Bounds(0.01, 0.99),
            tol=1e-06,
        )

        logging.info(f"fold: {fold} weights: {result.x}")
        weight_list.append(result.x)

    mean_weight = np.mean(weight_list, axis=0)
    mean_weight = np.insert(mean_weight, len(mean_weight), 1 - np.sum(mean_weight))
    logging.info(f"Optimized weight: {mean_weight}\n")

    return mean_weight
