import logging
import logging.handlers
import os
import random
import time
from contextlib import contextmanager
from typing import Any, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike
from pandas import DataFrame
from sklearn.metrics import f1_score


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONASHSEED"] = str(seed)
    np.random.seed(seed)


@contextmanager
def timer(name: Any, logger: logging.getLogger) -> None:
    t0 = time.time()
    logging.debug(f"[{name}] start")
    yield
    logger.debug(f"[{name}] done in {time.time() - t0:.0f} s")


def reduce_mem_usage(df: DataFrame, verbose: bool = True) -> DataFrame:
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024**2

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

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose:
        logging.info(
            "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df


def f1_eval(
    y_true: ArrayLike, y_pred: ArrayLike, thershold: float = 0.4
) -> Tuple[Union[str, float, bool]]:
    y_labels = (y_pred > thershold).astype(np.int8)
    return "f1", f1_score(y_labels, y_true), True


def xgb_f1(
    pred: ArrayLike, dtrain: ArrayLike, threshold: float = 0.4
) -> Tuple[Union[str, float]]:
    y_true = dtrain.get_label()
    y_pred = (pred > threshold).astype(np.int8)
    return "f1", f1_score(y_pred, y_true)
