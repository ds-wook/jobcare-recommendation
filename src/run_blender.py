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

    lightgbm = pd.read_csv(submit_path + "5fold_lightgbm_threshold_0.36.csv")
    catboost = pd.read_csv(submit_path + "5fold_catboost_threshold_0.4.csv")
    elo = pd.read_csv(submit_path + "elo_pred.csv")
    submission["submit_lgb"] = lightgbm.target
    submission["submit_ctb"] = catboost.target
    submission["submit_elo"] = elo.target

    submission["target"] = (
        submission[
            [col for col in submission.columns if col.startswith("submit_")]
        ].sum(axis=1)
        >= 2
    ).astype(int)

    submission.drop(
        [col for col in submission.columns if col.startswith("submit_")],
        axis=1,
        inplace=True,
    )
    submission.to_csv(submit_path + cfg.submit.name, index=False)


if __name__ == "__main__":
    _main()
