import hydra
import neptune.new as neptune
from omegaconf import DictConfig

from data.dataset import load_test_dataset, load_train_dataset
from features.build import kfold_mean_encoding, select_features
from tuning.boosting import LightGBMTuner


@hydra.main(config_path="../config/tuning/", config_name="model.yaml")
def _main(cfg: DictConfig):
    model_name = cfg.model.select
    train_x, train_y = load_train_dataset(cfg)
    test_x = load_test_dataset(cfg)
    train_x, test_x, train_y = kfold_mean_encoding(
        train_x, test_x, train_y, cfg.dataset.cat_features
    )
    train_x, test_x = select_features(train_x, train_y, test_x)

    if model_name == "lightgbm":
        run = neptune.init(
            project=cfg.experiment.project,
            tags=list(cfg.experiment.tags.lightgbm),
        )

        lgbm_tuner = LightGBMTuner(
            train_x=train_x, train_y=train_y, config=cfg, run=run
        )
        study = lgbm_tuner.build_study()
        lgbm_tuner.save_hyperparameters(study)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    _main()
