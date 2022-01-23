import hydra
import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig


@hydra.main(config_path="../config/train/", config_name="model.yaml")
def _main(cfg: DictConfig):
    path = to_absolute_path(cfg.dataset.path) + "/"
    submit_path = to_absolute_path(cfg.submit.path) + "/"
    submission = pd.read_csv(path + cfg.dataset.submit)

    lightgbm = pd.read_csv(submit_path + "proba_5fold_lightgbm.csv")
    tabnet = pd.read_csv(submit_path + "proba_5fold_tabnet.csv")
    catboost = pd.read_csv(submit_path + "proba_5fold_catboost.csv")
    elo = pd.read_csv(submit_path + "proba_elo.csv")
    submission["target"] = (
        lightgbm["proba_1"] * 0.1
        + tabnet["proba_1"] * 0.05
        + catboost["proba_1"] * 0.8
        + elo["target"] * 0.05
    )
    submission["target"] = np.where(submission["target"] < 0.38, 0, 1)
    submission.to_csv(submit_path + cfg.submit.name, index=False)


if __name__ == "__main__":
    _main()
