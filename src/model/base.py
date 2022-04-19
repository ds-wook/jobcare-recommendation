import gc
import logging
import pickle
from abc import ABCMeta, abstractclassmethod
from pathlib import Path
from typing import Any, Callable, Dict, NamedTuple

import numpy as np
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedKFold


class ModelResult(NamedTuple):
    oof_preds: np.ndarray
    models: Dict[str, Any]
    scores: Dict[str, float]


class BaseModel(metaclass=ABCMeta):
    def __init__(self, config: DictConfig, metric: Callable, search: bool = False):
        self.config = config
        self.metric = metric
        self.search = search
        self.result = None

    @abstractclassmethod
    def _train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
        fold: int,
    ):
        raise NotImplementedError

    def save_model(self):
        """
        Save model
        """
        model_path = Path(get_original_cwd()) / self.config.model.path

        with open(model_path, "wb") as output:
            pickle.dump(self.result, output, pickle.HIGHEST_PROTOCOL)

    def train(self, train_x: pd.DataFrame, train_y: pd.Series) -> ModelResult:
        """
        Train data
            Parameter:
                train_x: train dataset
                train_y: target dataset
            Return:
                True: Finish Training
        """

        models = dict()
        scores = dict()

        str_kf = StratifiedKFold(
            n_splits=self.config.model.fold, shuffle=True, random_state=42
        )
        splits = str_kf.split(train_x, train_y)

        oof_preds = np.zeros(train_x.shape[0])

        for fold, (train_idx, valid_idx) in enumerate(splits, 1):
            X_train, y_train = train_x.iloc[train_idx], train_y.iloc[train_idx]
            X_valid, y_valid = train_x.iloc[valid_idx], train_y.iloc[valid_idx]

            # model
            model = self._train(
                X_train,
                y_train,
                X_valid,
                y_valid,
                fold=fold,
            )
            models[f"fold_{fold}"] = model

            # validation
            oof_preds[valid_idx] = model.predict_proba(X_valid)[:, 1]

            score = self.metric(
                y_valid.values, oof_preds[valid_idx] > self.config.model.thershold
            )
            scores[f"fold_{fold}"] = score

            if not self.search:
                logging.info(f"Fold {fold}: {score}")

            gc.collect()

            del X_train, X_valid, y_train, y_valid

        oof_score = self.metric(train_y.values, oof_preds > self.config.model.thershold)
        self.result = ModelResult(
            oof_preds=oof_preds,
            models=models,
            preds=None,
            scores={"oof_score": oof_score, "KFold_scores": scores},
        )

        return self.result
