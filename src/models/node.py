import gc
from typing import Union

import numpy as np
import pandas as pd
from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular.models import NodeConfig
from pytorch_tabular.utils import get_class_weighted_cross_entropy
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import tqdm

from models.base import BaseModel, ModelResult
from utils.utils import LoggerFactory

logger = LoggerFactory().getLogger(__name__)


class NodeTrainer(BaseModel):
    def __init__(self, cat_features, **kwargs):
        self.cat_features = cat_features
        super().__init__(**kwargs)

    def _train(
        self,
        train: pd.DataFrame,
        valid: pd.DataFrame,
        fold: int,
        thershold: float = 0.4,
        verbose: Union[bool] = False,
    ):
        data_config = DataConfig(
            target=["target"],
            categorical_cols=self.cat_features,
            continuous_feature_transform="quantile_normal",
            normalize_continuous_features=True,
        )
        trainer_config = TrainerConfig(
            auto_lr_find=True,
            batch_size=1024,
            max_epochs=1000,
            auto_select_gpus=False,
            gpus=0,
        )

        optimizer_config = OptimizerConfig()

        model_config = NodeConfig(
            task="classification",
            num_layers=2,
            num_trees=1024,
            learning_rate=1,
            embed_categorical=False,
            metrics=["accuracy", "f1"],
            # target_range=(train['block_0'].min().item(), train['block_0'].max().item())
        )
        tabular_model = TabularModel(
            data_config=data_config,
            model_config=model_config,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config,
        )
        weighted_loss = get_class_weighted_cross_entropy(
            train["target"].values.ravel(), mu=0.1
        )
        tabular_model.fit(
            train=train, validation=valid, max_epochs=100, loss=weighted_loss
        )

        return tabular_model

    def train(
        self,
        data: pd.DataFrame,
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
        splits = str_kf.split(data, data.target.values)

        oof_preds = np.zeros(data.shape[0])

        for fold, (train_idx, valid_idx) in enumerate(splits, 1):
            train, test = data.iloc[train_idx], data.iloc[valid_idx]
            train, val = train_test_split(train, random_state=42)

            # model
            model = self._train(
                train,
                val,
                fold=fold,
                thershold=thershold,
                verbose=verbose,
            )
            models[f"fold_{fold}"] = model

            # validation
            oof_preds[valid_idx] = model.predict(test)[:, ["prediction"]]

            score = self.metric(test.target.values, oof_preds[valid_idx] > thershold)
            scores[f"fold_{fold}"] = score
            gc.collect()

            del train, test, val

        oof_score = self.metric(data.target.values, oof_preds > thershold)
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

        logger.info(f"oof score: {self.result.scores['oof_score']}")
        logger.info("Inference Start!")

        for fold in tqdm(range(1, folds + 1)):
            model = self.result.models[f"fold_{fold}"]
            preds += model.predict(test_x)[:, 1] / folds

        preds = np.where(preds < threshold, 0, 1)
        assert len(preds) == len(test_x)
        logger.info("Inference Finish!\n")

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
        preds_proba = np.zeros(test_x.shape[0])

        logger.info(f"oof score: {self.result.scores['oof_score']}")
        logger.info("Inference Start!")

        for fold in tqdm(range(1, folds + 1)):
            model = self.result.models[f"fold_{fold}"]
            preds_proba += model.predict_proba(test_x)[:, 1] / folds

        assert len(preds_proba) == len(test_x)
        logger.info("Inference Finish!\n")

        return preds_proba
