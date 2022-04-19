import pickle
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from tqdm import tqdm

from data.dataset import load_test_dataset
from model.base import ModelResult


def load_model(model_name: str) -> ModelResult:
    """
    Load model
    Args:
        model_name: model name
    Returns:
        ModelResult object
    """
    model_path = Path(get_original_cwd()) / model_name

    with open(model_path, "rb") as output:
        model_result = pickle.load(output)

    return model_result


def predict(
    result: ModelResult, test_x: pd.DataFrame, threshold: float = 0.5
) -> np.ndarray:
    """
    Predict data
        Parameter:
            test_x: test dataset
        Return:
            preds: inference prediction
    """
    folds = len(result.model)
    preds = np.zeros(test_x.shape[0])

    for model in tqdm(result.models.values(), total=folds):
        preds += model.predict_proba(test_x)[:, 1] / folds

    preds = np.where(preds < threshold, 0, 1)

    assert len(preds) == len(test_x)

    return preds


@hydra.main(config_path="../config/", config_name="predict.yaml")
def _main(cfg: DictConfig):
    dataset_path = Path(get_original_cwd()) / cfg.dataset.path
    submit_path = Path(get_original_cwd()) / cfg.output.path

    test = load_test_dataset(cfg)

    ignore = ["sequence", "subject"]
    features = [feat for feat in test.columns if feat not in ignore]
    test_x = test[features]

    # model load
    results = load_model(cfg.model.catboost)

    # infer test
    preds = predict(results, test_x)

    # Save test predictions
    submission = pd.read_csv(dataset_path / cfg.dataset.submit)
    submission[cfg.dataset.target] = preds
    submission.to_csv(submit_path / cfg.output.name, index=False)


if __name__ == "__main__":
    _main()
