from functools import partial

import hydra
import neptune.new as neptune
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from data.dataset import load_dataset
from data.features import kfold_mean_encoding, select_features
from tuning.bayesian import BayesianOptimizer, lgbm_objective, xgb_objective


@hydra.main(config_path="../config/optimization/", config_name="model.yaml")
def _main(cfg: DictConfig):
    path = to_absolute_path(cfg.dataset.path) + "/"
    model_name = cfg.model.select
    train_x, test_x, train_y = load_dataset(path)
    train_x, test_x, train_y = kfold_mean_encoding(train_x, test_x, train_y)
    train_x, test_x = select_features(train_x, train_y, test_x)

    if model_name == "lightgbm":
        run = neptune.init(
            project=cfg.experiment.project,
            tags=list(cfg.experiment.tags.lightgbm),
        )

        objective = partial(
            lgbm_objective,
            X=train_x,
            y=train_y,
            fold=cfg.model.fold,
            threshold=cfg.model.threshold,
        )

        bayesian_optim = BayesianOptimizer(
            objective_function=objective,
            run=run,
            trials=cfg.experiment.trials,
            direction=cfg.experiment.direction,
        )
        study = bayesian_optim.build_study()
        bayesian_optim.save_as_yaml_lgbm_params(study, cfg.experiment.params)

    elif model_name == "xgboost":
        run = neptune.init(
            project=cfg.experiment.project,
            tags=list(cfg.experiment.tags.xgboost),
        )

        objective = partial(
            xgb_objective,
            X=train_x,
            y=train_y,
            fold=cfg.model.fold,
            threshold=cfg.model.threshold,
        )

        bayesian_optim = BayesianOptimizer(
            objective_function=objective,
            run=run,
            trials=cfg.experiment.trials,
            direction=cfg.experiment.direction,
        )
        study = bayesian_optim.build_study()
        bayesian_optim.save_as_yaml_xgb_params(study, cfg.experiment.params)

    else:
        raise NotImplementedError


if __name__ == "__main__":
    _main()
