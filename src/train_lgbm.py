import hydra
import neptune.new as neptune
from omegaconf import DictConfig
from sklearn.metrics import f1_score

from data.dataset import load_test_dataset, load_train_dataset
from features.build import kfold_mean_encoding, select_features
from model.boosting import LightGBMTrainer


@hydra.main(config_path="../config/train/", config_name="lgbm.yaml")
def _main(cfg: DictConfig):
    train_x, train_y = load_train_dataset(cfg)
    test_x = load_test_dataset(cfg)
    train_x, test_x, train_y = kfold_mean_encoding(
        train_x, test_x, train_y, cfg.dataset.cat_features
    )
    train_x, test_x = select_features(train_x, train_y, test_x)

    # make experiment tracking
    run = neptune.init(
        project=cfg.experiment.project,
        tags=list(cfg.experiment.tags),
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
    lgbm_trainer.train(train_x, train_y)
    lgbm_trainer.save_model()


if __name__ == "__main__":
    _main()
