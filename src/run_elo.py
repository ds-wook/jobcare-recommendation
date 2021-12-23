import hydra
import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from data.elo_features import estimate_parameters, estimate_probas


@hydra.main(config_path="../config/train/", config_name="model.yaml")
def _main(cfg: DictConfig):
    path = to_absolute_path(cfg.dataset.path) + "/"
    submit_path = to_absolute_path(cfg.submit.path) + "/"
    submission = pd.read_csv(path + cfg.dataset.submit)

    train = pd.read_csv(path + cfg.dataset.train)
    test = pd.read_csv(path + cfg.dataset.test)

    train["left_asymptote"] = 0.25
    person_parameters, content_parameters = estimate_parameters(train)

    test["left_asymptote"] = 0.25
    submission["target"] = estimate_probas(test, person_parameters, content_parameters)
    submission["target"] = np.where(submission.target < cfg.model.threshold, 0, 1)
    submission.to_csv(submit_path + cfg.submit.name, index=False)


if __name__ == "__main__":
    _main()
