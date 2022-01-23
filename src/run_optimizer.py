import logging

import hydra
import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from scipy.optimize import minimize
from sklearn.metrics import f1_score

from optimization.optimize import optimize_f1


@hydra.main(config_path="../config/train/", config_name="lgbm.yaml")
def _main(cfg: DictConfig):
    path = to_absolute_path(cfg.dataset.path) + "/"
    submit_path = to_absolute_path(cfg.submit.path) + "/"
    submission = pd.read_csv(path + cfg.dataset.submit)

    # lgbm_oof = pd.read_csv(submit_path + "train_oof_5fold_lightgbm.csv")
    # lgbm_preds = pd.read_csv(submit_path + "proba_5fold_lightgbm.csv")

    cb_oof = pd.read_csv(submit_path + "train_oof_5fold_catboost.csv")
    cb_preds = pd.read_csv(submit_path + "proba_5fold_catboost.csv")

    # tabnet_oof = pd.read_csv(submit_path + "train_oof_5fold_tabnet.csv")
    # tabnet_preds = pd.read_csv(submit_path + "proba_5fold_tabnet.csv")

    y_true = cb_oof["target"].to_numpy()
    y_pred = cb_oof["oof_preds"].to_numpy()

    result = minimize(
        lambda x: optimize_f1(x, y_true, y_pred),
        x0=np.array([0.5]),
        method="Nelder-Mead",
    )

    best_threshold = result["x"].item()
    best_score = f1_score(y_true, y_pred >= best_threshold)

    logging.info(f"best threshold: {best_threshold}, best score: {best_score}")

    submission["target"] = np.where(cb_preds["proba_1"] < best_threshold, 0, 1)

    submission.to_csv(submit_path + cfg.submit.name, index=False)


if __name__ == "__main__":
    _main()
