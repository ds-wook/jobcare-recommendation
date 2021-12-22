import hydra
import neptune.new as neptune
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from sklearn.metrics import f1_score

from data.dataset import load_dataset
from models.gbdt import CatBoostTrainer, LightGBMTrainer, XGBoostTrainer


@hydra.main(config_path="../config/train/", config_name="model.yaml")
def _main(cfg: DictConfig):
    path = to_absolute_path(cfg.dataset.path) + "/"
    submit_path = to_absolute_path(cfg.submit.path) + "/"

    submission = pd.read_csv(path + cfg.dataset.submit)
    model_name = cfg.model.select
    train_x, test_x, train_y = load_dataset(path)

    if model_name == "lightgbm":
        # make experiment tracking
        run = neptune.init(
            project=cfg.experiment.project, tags=list(cfg.experiment.tags.lightgbm)
        )

        # model train
        lgbm_trainer = LightGBMTrainer(
            params=cfg.model.lightgbm.params,
            run=run,
            fold=cfg.model.fold,
            threshold=cfg.model.threshold,
            metric=f1_score,
        )
        lgbm_oof = lgbm_trainer.train(
            train_x, train_y, cfg.model.threshold, cfg.model.verbose
        )
        lgbm_preds = lgbm_trainer.predict(test_x, threshold=cfg.model.threshold)
        lgbm_preds_proba = lgbm_trainer.predict_proba(test_x)

        # Save test predictions
        submission[cfg.dataset.target] = lgbm_preds
        submission.to_csv(submit_path + cfg.submit.name, index=False)
        train = pd.read_csv(path + "train.csv")
        train["oof_preds"] = lgbm_oof.oof_preds
        train[["id", "target", "oof_preds"]].to_csv(path + "lgbm_oof.csv", index=False)
        submission[cfg.dataset.target] = lgbm_preds_proba
        submission.to_csv(
            submit_path
            + f"{cfg.model.fold}fold_{cfg.model.select}_proba_{cfg.model.threshold}.csv",
            index=False,
        )

    elif model_name == "xgboost":
        # make experiment tracking
        run = neptune.init(
            project=cfg.experiment.project, tags=list(cfg.experiment.tags.xgboost)
        )

        # model train
        xgb_trainer = XGBoostTrainer(
            params=cfg.model.xgboost.params,
            run=run,
            fold=cfg.model.fold,
            threshold=cfg.model.threshold,
            metric=f1_score,
        )
        xgb_trainer.train(train_x, train_y, cfg.model.threshold, cfg.model.verbose)
        xgb_preds = xgb_trainer.predict(test_x, threshold=cfg.model.threshold)
        xgb_preds_proba = xgb_trainer.predict_proba(test_x)

        # Save test predictions
        submission[cfg.dataset.target] = xgb_preds
        submission.to_csv(submit_path + cfg.submit.name, index=False)
        submission[cfg.dataset.target] = xgb_preds_proba
        submission.to_csv(
            submit_path
            + f"{cfg.model.fold}fold_{cfg.model.select}_proba_{cfg.model.threshold}.csv",
            index=False,
        )

    elif model_name == "catboost":
        # model train
        cb_trainer = CatBoostTrainer(
            params=cfg.model.catboost.params,
            cat_features=cfg.dataset.cat_features,
            fold=cfg.model.fold,
            threshold=cfg.model.threshold,
            metric=f1_score,
        )
        cb_oof = cb_trainer.train(
            train_x, train_y, cfg.model.threshold, cfg.model.verbose
        )

        cb_preds = cb_trainer.predict(test_x, threshold=cfg.model.threshold)
        cb_preds_proba = cb_trainer.predict_proba(test_x)
        train["oof_preds"] = cb_oof.oof_preds
        train[["id", "target", "oof_preds"]].to_csv(path + "cb_oof.csv", index=False)
        # Save test predictions
        submission[cfg.dataset.target] = cb_preds
        submission.to_csv(submit_path + cfg.submit.name, index=False)
        submission[cfg.dataset.target] = cb_preds_proba
        submission.to_csv(
            submit_path
            + f"{cfg.model.fold}fold_{cfg.model.select}_proba_{cfg.model.threshold}.csv",
            index=False,
        )

    else:
        raise NotImplementedError


if __name__ == "__main__":
    _main()
