import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from pandas import DataFrame, Series
from shap import TreeExplainer
from sklearn.model_selection import KFold
from tqdm import tqdm


def select_features(
    train: DataFrame, label: Series, test: DataFrame
) -> Tuple[DataFrame, DataFrame]:

    model = LGBMClassifier(random_state=42)
    logging.info(f"{model.__class__.__name__} Train Start!")
    model.fit(train, label)
    explainer = TreeExplainer(model)

    shap_values = explainer.shap_values(test)
    shap_sum = np.abs(shap_values).mean(axis=1).sum(axis=0)

    importance_df = pd.DataFrame([test.columns.tolist(), shap_sum.tolist()]).T
    importance_df.columns = ["column_name", "shap_importance"]

    importance_df = importance_df.sort_values("shap_importance", ascending=False)
    importance_df = importance_df.query("shap_importance != 0")
    boosting_shap_col = importance_df.column_name.values.tolist()

    logging.info(f"Total {len(train.columns)} Select {len(boosting_shap_col)}")

    shap_train = train.loc[:, boosting_shap_col]
    shap_test = test.loc[:, boosting_shap_col]

    logging.info(f"Train: {shap_train.shape} Test: {shap_test.shape}")

    return shap_train, shap_test


def kfold_mean_encoding(
    train_x: DataFrame,
    test_x: DataFrame,
    train_y: Series,
    cat_features: List[str],
) -> DataFrame:
    for c in tqdm(cat_features):
        data_tmp = pd.DataFrame({c: train_x[c], "target": train_y})
        target_mean = data_tmp.groupby(c)["target"].mean()

        # 테스트 데이터의 카테고리 변경
        test_x[c] = test_x[c].map(target_mean)

        # 학습 데이터 변환 후 값을 저장하는 배열 준비
        tmp = np.repeat(np.nan, train_x.shape[0])

        kf = KFold(n_splits=4, shuffle=True, random_state=42)

        for train_idx, valid_idx in kf.split(train_x):
            # out of fold 로 각 범주형 목적변수 평균 계산
            target_mean = data_tmp.iloc[train_idx].groupby(c)["target"].mean()
            # 변환 후의 값을 날짜 배열에 저장
            tmp[valid_idx] = train_x[c].iloc[valid_idx].map(target_mean)

        train_x[c] = tmp

    return train_x, test_x, train_y
