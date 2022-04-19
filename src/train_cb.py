import hydra
import neptune.new as neptune
from omegaconf import DictConfig
from sklearn.metrics import f1_score

from data.dataset import load_train_dataset
from model.boosting import CatBoostTrainer
from utils.utils import reduce_mem_usage


@hydra.main(config_path="../config/modeling/", config_name="cb.yaml")
def _main(cfg: DictConfig):
    train_x, train_y = load_train_dataset(cfg)

    train_x = reduce_mem_usage(train_x)

    run = neptune.init(
        project=cfg.experiment.project,
        tags=list(cfg.experiment.tags),
        capture_hardware_metrics=False,
    )

    cb_trainer = CatBoostTrainer(config=cfg, run=run, metric=f1_score)
    cb_trainer.train(train_x, train_y)
    cb_trainer.save_model()


if __name__ == "__main__":
    _main()
