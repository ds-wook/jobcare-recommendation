import warnings
from typing import Callable, Sequence, Union

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
                timeout=21600,
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
        print("Best trial:")
        trial = study.best_trial
        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    '{key}': {value},")

    @staticmethod
    def lgbm_save_params(study: Study, params_name: str):
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
    def xgb_save_params(study: optuna.create_study, params_name: str):
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
        "objective": "regression",
        "boosting_type": "gbdt",
        "n_jobs": -1,
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.05),
        "num_leaves": trial.suggest_int("num_leaves", 4, 64),
        "max_depth": trial.suggest_int("max_depth", 4, 16),
        "subsample": trial.suggest_float("subsample", 0.1, 1.0),
        "colsample_bytree": trial.suggest_float("subsample", 0.1, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 0.1),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 0.1),
    }
    pruning_callback = LightGBMPruningCallback(trial, "f1", valid_name="valid_1")

    lgbm_trainer = LightGBMTrainer(
        params=params,
        run=pruning_callback,
        search=True,
        fold=fold,
        threshold=threshold,
        metric=f1_score,
    )

    result = lgbm_trainer.train(X, y)
    score = f1_score(y, result.oof_preds)

    return score


def xgb_objective(
    trial: FrozenTrial,
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
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.01),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 0.5),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.3, 1.0),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.3, 1.0),
        "gamma": trial.suggest_float("gamma", 0.01, 0.1),
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
