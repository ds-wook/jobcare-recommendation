import warnings
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier
from neptune.new import Run
from neptune.new.integrations import lightgbm, xgboost
from neptune.new.integrations.lightgbm import create_booster_summary
from xgboost import XGBClassifier

from models.base import BaseModel
from utils.utils import f1_eval, xgb_f1

warnings.filterwarnings("ignore")


class LightGBMTrainer(BaseModel):
    def __init__(
        self,
        params: Optional[Dict[str, Any]],
        run: Optional[Run],
        seed: int = 42,
        **kwargs,
    ):
        self.params = params
        self.run = run
        self.seed = seed
        super().__init__(**kwargs)

    def _get_default_params(self) -> Dict[str, Any]:
        """
        setting default parameters
        Return:
            LightGBM default parameter
        """

        return {
            "n_estimators": 10000,
            "boosting_type": "gbdt",
            "objective": "binary",
            "learning_rate": 0.05,
            "num_leaves": 5,
            "max_bin": 55,
            "subsample": 0.8,
            "min_child_sample": 6,
            "min_child_weight": 11,
        }

    def _train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
        fold: int,
        threshold: float = 0.4,
        verbose: Union[int, bool] = False,
    ) -> LGBMClassifier:
        """method train"""

        neptune_callback = (
            lightgbm.NeptuneCallback(run=self.run, base_namespace=f"fold_{fold}")
            if not self.search
            else self.run
        )

        model = (
            LGBMClassifier(random_state=self.seed, **self.params)
            if self.params is not None
            else LGBMClassifier(random_state=self.seed, **self._get_default_params())
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            early_stopping_rounds=100,
            eval_metric=lambda y_true, y_pred: f1_eval(y_true, y_pred, threshold),
            verbose=verbose,
            callbacks=[neptune_callback],
        )

        if not self.search:
            # Log summary metadata to the same run under the "lgbm_summary" namespace
            self.run[f"lgbm_summary/fold_{fold}"] = create_booster_summary(
                booster=model,
                y_pred=model.predict(X_valid),
                y_true=y_valid,
            )

        return model


class XGBoostTrainer(BaseModel):
    def __init__(
        self,
        params: Optional[Dict[str, Any]],
        run: Optional[Run],
        seed: int = 42,
        search: bool = False,
        **kwargs,
    ):
        self.params = params
        self.run = run
        self.seed = seed
        self.search = search
        super().__init__(**kwargs)

    def _get_default_params(self) -> Dict[str, Any]:
        """
        setting default parameters
        Return:
            XGBoost default parameter
        """

        return {
            "objective": "binary:logistic",
            "n_estimators": 10000,
            "random_state": 42,
            "learning_rate": 0.05,
            "max_depth": 3,
            "gamma": 0.0468,
            "min_child_weight": 1.7817,
            "reg_alpha": 0.4640,
            "reg_lambda": 0.8571,
            "subsample": 0.5213,
        }

    def _train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
        fold: int,
        threshold: float = 0.4,
        verbose: Union[int, bool] = False,
    ) -> XGBClassifier:
        """method train"""

        neptune_callback = (
            xgboost.NeptuneCallback(
                run=self.run,
                base_namespace=f"fold_{fold}",
                log_tree=[0, 1, 2, 3],
                max_num_features=10,
            )
            if not self.search
            else self.run
        )

        model = (
            XGBClassifier(random_state=self.seed, **self.params)
            if self.params is not None
            else XGBClassifier(random_state=self.seed, **self._get_default_params())
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            early_stopping_rounds=100,
            eval_metric=lambda y_true, y_pred: xgb_f1(y_true, y_pred, threshold),
            verbose=verbose,
            callbacks=[neptune_callback],
        )

        return model


class CatBoostTrainer(BaseModel):
    def __init__(
        self,
        params: Optional[Dict[str, Any]],
        cat_features: Optional[List[str]],
        seed: int = 42,
        search: bool = False,
        **kwargs,
    ):
        self.params = params
        self.search = search
        self.cat_features = cat_features
        self.seed = seed
        super().__init__(**kwargs)

    def _get_default_params(self) -> Dict[str, Any]:
        """
        setting default parameters
        Return:
            NGBoost default parameter
        """
        return {
            "learning_rate": 0.03,
            "l2_leaf_reg": 20,
            "depth": 4,
            "bootstrap_type": "Bernoulli",
            "subsample": 0.6,
            "random_state": 42,
            "eval_metric": "RMSE",
            "loss_function": "RMSE",
            "od_type": "Iter",
            "od_wait": 45,
            "iterations": 10000,
        }

    def _train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
        fold: int,
        threshold: float = 0.5,
        is_search: Union[bool] = False,
        verbose: Union[bool] = False,
    ) -> CatBoostClassifier:
        """method train"""
        train_data = Pool(data=X_train, label=y_train, cat_features=self.cat_features)
        valid_data = Pool(data=X_valid, label=y_valid, cat_features=self.cat_features)

        model = (
            CatBoostClassifier(
                random_state=self.seed, cat_features=self.cat_features, **self.params
            )
            if self.params is not None
            else CatBoostClassifier(
                random_state=self.seed,
                cat_features=self.cat_features,
                **self._get_default_params(),
            )
        )

        model.fit(
            train_data,
            eval_set=valid_data,
            early_stopping_rounds=100,
            use_best_model=True,
            verbose=verbose,
        )

        return model
