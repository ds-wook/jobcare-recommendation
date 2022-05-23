import hydra
from omegaconf import DictConfig
from sklearn.metrics import f1_score

from data.dataset import load_train_dataset
from model.boosting import EBMTrainer
from utils.utils import reduce_mem_usage


@hydra.main(config_path="../config/modeling/", config_name="ebm.yaml")
def _main(cfg: DictConfig):
    train_x, train_y = load_train_dataset(cfg)

    train_x = reduce_mem_usage(train_x)

    ebm_trainer = EBMTrainer(config=cfg, metric=f1_score)
    ebm_trainer.train(train_x, train_y)


if __name__ == "__main__":
    _main()
