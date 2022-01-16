# %%
import numpy as np
import pandas as pd

# %%
path = "../submit/"
lgbm_preds = pd.read_csv(path + "5fold_lightgbm_proba_0.4.csv")
lgbm_preds.head()
# %%
cb_preds = pd.read_csv(path + "5fold_catboost_proba_0.38.csv")
cb_preds.head()
# %%
(lgbm_preds["proba_1"].rank() / lgbm_preds["proba_1"].rank().sum()).sum()
# %%
from scipy.stats import rankdata

np.average(
    [rankdata(lgbm_preds["proba_1"]), rankdata(cb_preds["proba_1"])],
    weights=[0.3, 0.7],
    axis=0,
)
# %%
(lgbm_preds["proba_1"].rank(method="min") + cb_preds["proba_1"].rank(method="min")) / 2
# %%
