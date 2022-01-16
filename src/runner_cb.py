import hydra
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from sklearn.metrics import f1_score

from data.dataset import load_dataset
from models.gbdt import CatBoostTrainer
from utils.utils import reduce_mem_usage


@hydra.main(config_path="../config/train/", config_name="cb.yaml")
def _main(cfg: DictConfig):
    path = to_absolute_path(cfg.dataset.path) + "/"
    submit_path = to_absolute_path(cfg.submit.path) + "/"
    train = pd.read_csv(path + "train.csv")
    submission = pd.read_csv(path + cfg.dataset.submit)

    train_x, test_x, train_y = load_dataset(path)
    train_x = reduce_mem_usage(train_x)
    test_x = reduce_mem_usage(test_x)

    # model train
    cb_trainer = CatBoostTrainer(
        params=cfg.model.params,
        seed=cfg.model.seed,
        cat_features=cfg.dataset.cat_features,
        fold=cfg.model.fold,
        threshold=cfg.model.threshold,
        metric=f1_score,
    )
    cb_oof = cb_trainer.train(train_x, train_y, cfg.model.threshold, cfg.model.verbose)

    cb_preds = cb_trainer.predict(test_x, threshold=cfg.model.threshold)
    cb_preds_proba = cb_trainer.predict_proba(test_x)

    train["oof_preds"] = cb_oof.oof_preds
    train[["id", "target", "oof_preds"]].to_csv(path + "cb_oof.csv", index=False)
    # Save test predictions
    submission[cfg.dataset.target] = cb_preds
    submission.to_csv(submit_path + cfg.submit.name, index=False)
    submission[["proba_0", "proba_1"]] = cb_preds_proba
    submission.to_csv(
        submit_path + f"{cfg.model.fold}fold_catboost_proba_{cfg.model.threshold}.csv",
        index=False,
    )


if __name__ == "__main__":
    _main()
