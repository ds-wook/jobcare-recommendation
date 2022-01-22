import gc
import logging
from abc import abstractclassmethod
from typing import Any, Callable, Dict, NamedTuple, Optional, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm


class ModelResult(NamedTuple):
    oof_preds: np.ndarray
    preds: Optional[np.ndarray]
    models: Dict[str, Any]
    scores: Dict[str, float]


class BaseModel:
    def __init__(
        self, fold: int, threshold: float, metric: Callable, search: bool = False
    ):
        self.fold = fold
        self.threshold = threshold
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
        verbose: Union[bool] = False,
    ):
        raise NotImplementedError

    def train(
        self,
        train_x: pd.DataFrame,
        train_y: pd.Series,
        thershold: float = 0.4,
        verbose: Union[bool] = False,
    ) -> ModelResult:
        """
        Train data
            Parameter:
                train_x: train dataset
                train_y: target dataset
                groups: group fold parameters
                params: lightgbm' parameters
                verbose: log lightgbm' training
            Return:
                True: Finish Training
        """

        models = dict()
        scores = dict()

        str_kf = StratifiedKFold(n_splits=self.fold, shuffle=True, random_state=42)
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
                threshold=thershold,
                verbose=verbose,
            )
            models[f"fold_{fold}"] = model

            # validation
            oof_preds[valid_idx] = model.predict_proba(X_valid)[:, 1]

            score = self.metric(y_valid.values, oof_preds[valid_idx] > thershold)
            scores[f"fold_{fold}"] = score

            if not self.search:
                logging.info(f"Fold {fold}: {score}")

            gc.collect()

            del X_train, X_valid, y_train, y_valid

        oof_score = self.metric(train_y.values, oof_preds > thershold)
        self.result = ModelResult(
            oof_preds=oof_preds,
            models=models,
            preds=None,
            scores={"oof_score": oof_score, "KFold_scores": scores},
        )

        return self.result

    def predict(self, test_x: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Predict data
            Parameter:
                test_x: test dataset
            Return:
                preds: inference prediction
        """
        folds = self.fold
        preds = np.zeros(test_x.shape[0])

        logging.info(f"oof score: {self.result.scores['oof_score']}")
        logging.info("Inference Start!")

        for fold in tqdm(range(1, folds + 1)):
            model = self.result.models[f"fold_{fold}"]
            preds += model.predict_proba(test_x)[:, 1] / folds

        preds = np.where(preds < threshold, 0, 1)
        assert len(preds) == len(test_x)
        logging.info("Inference Finish!\n")

        return preds

    def predict_proba(self, test_x: pd.DataFrame) -> np.ndarray:
        """
        Predict data
            Parameter:
                test_x: test dataset
            Return:
                preds: inference prediction
        """
        folds = self.fold
        preds_proba = np.zeros((test_x.shape[0], 2))

        for fold in tqdm(range(1, folds + 1)):
            model = self.result.models[f"fold_{fold}"]
            preds_proba += model.predict_proba(test_x) / folds

        assert len(preds_proba) == len(test_x)

        return preds_proba
