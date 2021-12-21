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

    elo = pd.read_csv(submit_path + "elo_probability.csv")
    lightgbm = pd.read_csv(submit_path + "5fold_lightgbm_proba.csv")
    catboost = pd.read_csv(submit_path + "5fold_catboost_proba.csv")
    submission["target"] = (
        elo.target * 0.7 + (lightgbm.target * 0.9 + catboost.target * 0.1) * 0.3
    )
    submission["target"] = np.where(submission.target < 0.4, 0, 1)
    submission.to_csv(submit_path + cfg.submit.name, index=False)


if __name__ == "__main__":
    _main()
