import hydra
import neptune.new as neptune
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from sklearn.metrics import f1_score

from data.dataset import load_dataset
from data.features import kfold_mean_encoding, select_features
from models.gbdt import LightGBMTrainer


@hydra.main(config_path="../config/train/", config_name="lgbm.yaml")
def _main(cfg: DictConfig):
    path = to_absolute_path(cfg.dataset.path) + "/"
    submit_path = to_absolute_path(cfg.submit.path) + "/"
    submission = pd.read_csv(path + cfg.dataset.submit)
    train = pd.read_csv(path + "train.csv")
    train_x, test_x, train_y = load_dataset(path)
    train_x, test_x, train_y = kfold_mean_encoding(train_x, test_x, train_y)
    train_x, test_x = select_features(train_x, train_y, test_x)

    # make experiment tracking
    run = neptune.init(
        project=cfg.experiment.project,
        tags=list(cfg.experiment.tags.lightgbm),
        capture_hardware_metrics=False,
    )

    # model train
    lgbm_trainer = LightGBMTrainer(
        params=cfg.model.params,
        run=run,
        seed=cfg.model.seed,
        fold=cfg.model.fold,
        threshold=cfg.model.threshold,
        metric=f1_score,
    )
    lgbm_result = lgbm_trainer.train(
        train_x, train_y, cfg.model.threshold, cfg.model.verbose
    )
    train["oof_preds"] = lgbm_result.oof_preds
    train[["id", "target", "oof_preds"]].to_csv(
        submit_path + f"train_oof_{cfg.submit.name}"
    )
    lgbm_preds = lgbm_trainer.predict(test_x, threshold=cfg.model.threshold)
    lgbm_preds_proba = lgbm_trainer.predict_proba(test_x)

    # Save test predictions
    submission[cfg.dataset.target] = lgbm_preds
    submission.to_csv(submit_path + cfg.submit.name, index=False)

    submission[["proba_0", "proba_1"]] = lgbm_preds_proba
    submission.to_csv(submit_path + f"proba_{cfg.submit.name}", index=False)


if __name__ == "__main__":
    _main()
