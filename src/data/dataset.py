from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from tqdm import tqdm

tqdm.pandas()


def preprocess_data(
    df: pd.DataFrame,
    cols_merge: List[Tuple[str, pd.DataFrame]],
    cols_equi: List[Tuple[str, str]],
    cols_drop: List[str],
    is_train: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray]:
    df = df.copy()

    y_data = None

    if is_train:
        y_data = df["target"]
        df = df.drop(columns="target")

    for col, df_code in cols_merge:
        df = merge_codes(df, df_code, col)

    cols = df.select_dtypes(bool).columns.tolist()
    df[cols] = df[cols].astype(int)

    for col1, col2 in cols_equi:
        df[f"{col1}_{col2}"] = (df[col1] == df[col2]).astype(np.int8)

    df = df.drop(columns=cols_drop)

    return df, y_data


def merge_codes(df: pd.DataFrame, df_code: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    df_code = df_code.copy()
    df_code = df_code.add_prefix(f"{col}_")
    df_code.columns.values[0] = col
    return pd.merge(df, df_code, how="left", on=col)


def load_train_dataset(config: DictConfig) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load dataset

    Args:
        config: config object
    Returns:
        train_x, train_y
    """
    path = Path(get_original_cwd()) / config.dataset.path
    train = pd.read_csv(path / config.dataset.train)

    code_d = pd.read_csv(path / config.dataset.code_d)
    code_h = pd.read_csv(path / config.dataset.code_h)
    code_l = pd.read_csv(path / config.dataset.code_l)

    code_d.columns = [
        "attribute_d",
        "attribute_d_d",
        "attribute_d_s",
        "attribute_d_m",
        "attribute_d_l",
    ]
    code_h.columns = ["attribute_h", "attribute_h_l", "attribute_h_m"]
    code_l.columns = [
        "attribute_l",
        "attribute_l_d",
        "attribute_l_s",
        "attribute_l_m",
        "attribute_l_l",
    ]
    # 소분류 중분류 대분류 속성코드 merge 컬럼명 및 데이터 프레임 리스트
    cols_merge = [
        ("person_prefer_d_1", code_d),
        ("person_prefer_d_2", code_d),
        ("person_prefer_d_3", code_d),
        ("contents_attribute_d", code_d),
        ("person_prefer_h_1", code_h),
        ("person_prefer_h_2", code_h),
        ("person_prefer_h_3", code_h),
        ("contents_attribute_h", code_h),
        ("contents_attribute_l", code_l),
    ]

    # 회원 속성과 콘텐츠 속성의 동일한 코드 여부에 대한 컬럼명 리스트
    cols_equi = [
        ("contents_attribute_c", "person_prefer_c"),
        ("contents_attribute_e", "person_prefer_e"),
        ("person_prefer_d_2_attribute_d_s", "contents_attribute_d_attribute_d_s"),
        ("person_prefer_d_2_attribute_d_m", "contents_attribute_d_attribute_d_m"),
        ("person_prefer_d_2_attribute_d_l", "contents_attribute_d_attribute_d_l"),
        ("person_prefer_d_3_attribute_d_s", "contents_attribute_d_attribute_d_s"),
        ("person_prefer_d_3_attribute_d_m", "contents_attribute_d_attribute_d_m"),
        ("person_prefer_d_3_attribute_d_l", "contents_attribute_d_attribute_d_l"),
        ("person_prefer_h_1_attribute_h_m", "contents_attribute_h_attribute_h_m"),
        ("person_prefer_h_2_attribute_h_m", "contents_attribute_h_attribute_h_m"),
        ("person_prefer_h_3_attribute_h_m", "contents_attribute_h_attribute_h_m"),
        ("person_prefer_h_1_attribute_h_l", "contents_attribute_h_attribute_h_l"),
        ("person_prefer_h_2_attribute_h_l", "contents_attribute_h_attribute_h_l"),
        ("person_prefer_h_3_attribute_h_l", "contents_attribute_h_attribute_h_l"),
    ]

    # 학습에 필요없는 컬럼 리스트
    cols_drop = [
        "id",
        "person_prefer_f",
        "person_prefer_g",
        "contents_open_dt",
        "person_rn",
        "contents_rn",
    ]

    train, target = preprocess_data(
        train, cols_merge=cols_merge, cols_equi=cols_equi, cols_drop=cols_drop
    )

    return train, target


def load_test_dataset(config: DictConfig) -> pd.DataFrame:
    """
    Load dataset

    Args:
        config: config object
    Returns:
        test_x
    """
    path = Path(get_original_cwd()) / config.dataset.path
    test = pd.read_csv(path / config.dataset.test)
    code_d = pd.read_csv(path / config.dataset.code_d)
    code_h = pd.read_csv(path / config.dataset.code_h)
    code_l = pd.read_csv(path / config.dataset.code_l)

    code_d.columns = [
        "attribute_d",
        "attribute_d_d",
        "attribute_d_s",
        "attribute_d_m",
        "attribute_d_l",
    ]
    code_h.columns = ["attribute_h", "attribute_h_l", "attribute_h_m"]
    code_l.columns = [
        "attribute_l",
        "attribute_l_d",
        "attribute_l_s",
        "attribute_l_m",
        "attribute_l_l",
    ]
    # 소분류 중분류 대분류 속성코드 merge 컬럼명 및 데이터 프레임 리스트
    cols_merge = [
        ("person_prefer_d_1", code_d),
        ("person_prefer_d_2", code_d),
        ("person_prefer_d_3", code_d),
        ("contents_attribute_d", code_d),
        ("person_prefer_h_1", code_h),
        ("person_prefer_h_2", code_h),
        ("person_prefer_h_3", code_h),
        ("contents_attribute_h", code_h),
        ("contents_attribute_l", code_l),
    ]

    # 회원 속성과 콘텐츠 속성의 동일한 코드 여부에 대한 컬럼명 리스트
    cols_equi = [
        ("contents_attribute_c", "person_prefer_c"),
        ("contents_attribute_e", "person_prefer_e"),
        ("person_prefer_d_2_attribute_d_s", "contents_attribute_d_attribute_d_s"),
        ("person_prefer_d_2_attribute_d_m", "contents_attribute_d_attribute_d_m"),
        ("person_prefer_d_2_attribute_d_l", "contents_attribute_d_attribute_d_l"),
        ("person_prefer_d_3_attribute_d_s", "contents_attribute_d_attribute_d_s"),
        ("person_prefer_d_3_attribute_d_m", "contents_attribute_d_attribute_d_m"),
        ("person_prefer_d_3_attribute_d_l", "contents_attribute_d_attribute_d_l"),
        ("person_prefer_h_1_attribute_h_m", "contents_attribute_h_attribute_h_m"),
        ("person_prefer_h_2_attribute_h_m", "contents_attribute_h_attribute_h_m"),
        ("person_prefer_h_3_attribute_h_m", "contents_attribute_h_attribute_h_m"),
        ("person_prefer_h_1_attribute_h_l", "contents_attribute_h_attribute_h_l"),
        ("person_prefer_h_2_attribute_h_l", "contents_attribute_h_attribute_h_l"),
        ("person_prefer_h_3_attribute_h_l", "contents_attribute_h_attribute_h_l"),
    ]

    # 학습에 필요없는 컬럼 리스트
    cols_drop = [
        "id",
        "person_prefer_f",
        "person_prefer_g",
        "contents_open_dt",
        "person_rn",
        "contents_rn",
    ]

    test, _ = preprocess_data(
        test,
        cols_merge=cols_merge,
        cols_equi=cols_equi,
        cols_drop=cols_drop,
        is_train=False,
    )

    return test
