import logging
import logging.handlers
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONASHSEED"] = str(seed)
    np.random.seed(seed)


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)

        return cls._instances[cls]


class LoggerFactory(metaclass=Singleton):
    def __init__(self, log_path: Optional[str] = None, loglevel=logging.INFO):
        self.loglevel = loglevel
        if log_path is None:
            self.log_path = Path("../log/log")
        else:
            self.log_path = Path(log_path)
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def getLogger(self, log_name: str) -> logging.getLogger:
        fmt = "%(asctime)s [%(name)s|%(levelname)s] %(message)s"
        formatter = logging.Formatter(fmt)
        logger = logging.getLogger(log_name)

        # add stream Handler
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # add file Handler
        handler = logging.handlers.RotatingFileHandler(
            filename=self.log_path, maxBytes=2 * 1024 * 1024 * 1024, backupCount=10
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(self.loglevel)

        return logger


@contextmanager
def timer(name: Any, logger: logging.getLogger):
    t0 = time.time()
    logging.debug(f"[{name}] start")
    yield
    logger.debug(f"[{name}] done in {time.time() - t0:.0f} s")


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024 ** 2

    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024 ** 2

    if verbose:
        print(
            "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df


def f1_eval(
    y_true: np.ndarray, y_pred: np.ndarray, thershold: float = 0.4
) -> Tuple[Union[str, float, bool]]:
    y_labels = (y_pred > thershold).astype(np.int8)
    return "f1", f1_score(y_labels, y_true), True


def xgb_f1(
    pred: np.ndarray, dtrain: np.ndarray, threshold: float = 0.4
) -> Tuple[Union[str, float]]:
    y_true = dtrain.get_label()
    y_pred = (pred > threshold).astype(np.int8)
    return "f1", f1_score(y_pred, y_true)
