import logging
import warnings
from typing import Any, Callable, Dict, Sequence, Union

import neptune.new.integrations.optuna as optuna_utils
import optuna
import pandas as pd
import yaml
from hydra.utils import to_absolute_path
from neptune.new import Run
from neptune.new.exceptions import NeptuneMissingApiTokenException
from optuna.integration import LightGBMPruningCallback, XGBoostPruningCallback
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.study import Study
from optuna.trial import FrozenTrial, Trial
from sklearn.metrics import f1_score

from models.gbdt import LightGBMTrainer, XGBoostTrainer

warnings.filterwarnings("ignore")


class BayesianOptimizer:
    """
    Search Hyperparameter with Bayesian TPE Sampling
    Parameters:
        objective_function: Each model objective function
        run: run neptune server
    """

    def __init__(
        self,
        objective_function: Callable[[Trial], Union[float, Sequence[float]]],
        run: Run,
        trials: FrozenTrial = 100,
        direction: str = "maximize",
    ):
        self.objective_function = objective_function
        self.run = run
        self.trials = trials
        self.direction = direction

    def build_study(self, verbose: bool = False):
        try:
            neptune_callback = optuna_utils.NeptuneCallback(
                self.run,
                plots_update_freq=1,
                log_plot_slice=False,
                log_plot_contour=False,
            )
            sampler = TPESampler(seed=42)

            study = optuna.create_study(
                study_name="TPE Optimization",
                direction=self.direction,
                sampler=sampler,
                pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5),
            )

            study.optimize(
                self.objective_function,
                n_trials=self.trials,
                callbacks=[neptune_callback],
            )
            self.run.stop()

        except NeptuneMissingApiTokenException:
            sampler = TPESampler(seed=42)
            study = optuna.create_study(
                study_name="optimization", direction=self.direction, sampler=sampler
            )
            study.optimize(self.objective_function, n_trials=self.trials)

        if verbose:
            self.display_study_statistics(study)

        return study

    @staticmethod
    def display_study_statistics(study: Study):
        """
        Display best metric score and hyperparameters
        Parameters:
            study: study best hyperparameter object.
        """
        logging.info("Best trial:")
        trial = study.best_trial
        logging.info(f"  Value: {trial.value}")
        logging.info("  Params: ")
        for key, value in trial.params.items():
            logging.info(f"    '{key}': {value},")

    @staticmethod
    def save_as_yaml_lgbm_params(study: Study, params_name: str):
        """
        Save LightGBM hyperparameter
        Parameter:
            study: study best hyperparameter object.
            params_name: .yaml names
        """
        params = study.best_trial.params
        params["n_estimators"] = 10000
        params["boosting_type"] = "gbdt"
        params["objective"] = "binary"
        params["random_state"] = 42
        params["n_jobs"] = -1

        with open(to_absolute_path("../config/train/model.yaml")) as f:
            train_dict = yaml.load(f, Loader=yaml.FullLoader)

        train_dict["model"]["lightgbm"]["params"] = params

        with open(to_absolute_path("../config/train/" + params_name), "w") as p:
            yaml.dump(train_dict, p)

    @staticmethod
    def save_as_yaml_xgb_params(study: Study, params_name: str):
        """
        Save XGBoost hyperparameter
        Parameter:
            study: study best hyperparameter object.
            params_name: .yaml names
        """
        params = study.best_trial.params
        params["random_state"] = 42
        params["n_estimators"] = 10000
        params["n_jobs"] = -1
        params["objective"] = "binary:logistic"

        with open(to_absolute_path("../config/train/model.yaml")) as f:
            train_dict = yaml.load(f, Loader=yaml.FullLoader)
        train_dict["model"]["xgboost"]["params"] = params

        with open(to_absolute_path("../config/train/" + params_name), "w") as p:
            yaml.dump(train_dict, p)


def lgbm_objective(
    trial: FrozenTrial,
    params: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    fold: int,
    threshold: float,
) -> float:
    """
    LightGBM objective function
    Parameters:
        trial: Experiment times
        X: train dataset
        y: target values
        n_fold: model fold count n
    Return:
        f1 score
    """
    params = {
        "n_estimators": 10000,
        "objective": "binary",
        "boosting_type": "gbdt",
        "n_jobs": -1,
        "learning_rate": trial.suggest_float("learning_rate", *params.learning_rate),
        "num_leaves": trial.suggest_int("num_leaves", *params.num_leaves),
        "max_depth": trial.suggest_int("max_depth", *params.max_depth),
        "reg_alpha": trial.suggest_float("reg_alpha", *params.reg_alpha),
        "reg_lambda": trial.suggest_float("reg_lambda", *params.reg_lambda),
        "subsample": trial.suggest_float("subsample", *params.subsample),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", *params.colsample_bytree
        ),
    }
    pruning_callback = LightGBMPruningCallback(trial, "f1", valid_name="valid_1")

    lgbm_trainer = LightGBMTrainer(
        params=params,
        run=pruning_callback,
        fold=fold,
        threshold=threshold,
        metric=f1_score,
        search=True,
    )

    result = lgbm_trainer.train(X, y)
    score = f1_score(y, result.oof_preds > threshold)

    return score


def xgb_objective(
    trial: FrozenTrial,
    params: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    fold: int,
    threshold: float,
) -> float:
    """
    XGBoost objective function
    Parameters:
        trial: Experiment times
        X: train dataset
        y: target values
        n_fold: model fold count n
    Return:
        f1 score
    """
    params = {
        "random_state": 42,
        "n_estimators": 10000,
        "objective": "reg:squarederror",
        "n_jobs": -1,
        "learning_rate": trial.suggest_float("learning_rate", *params.learning_rate),
        "reg_alpha": trial.suggest_float("reg_alpha", *params.reg_alpha),
        "reg_lambda": trial.suggest_float("reg_lambda", *params.reg_lambda),
        "max_depth": trial.suggest_int("max_depth", *params.max_depth),
        "subsample": trial.suggest_float("subsample", *params.subsample),
        "gamma": trial.suggest_float("gamma", *params.gamma),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", *params.colsample_bytree
        ),
        "min_child_weight": trial.suggest_int(
            "min_child_weight", *params.min_child_weight
        ),
    }

    pruning_callback = XGBoostPruningCallback(trial, "validation_1-f1")

    xgb_trainer = XGBoostTrainer(
        params=params,
        run=pruning_callback,
        search=True,
        fold=fold,
        threshold=threshold,
        metric=f1_score,
    )

    result = xgb_trainer.train(X, y)
    score = f1_score(y, result.oof_preds)

    return score
