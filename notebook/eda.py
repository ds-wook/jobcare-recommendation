# %%
import numpy as np
import pandas as pd

# %%
lgbm_oof = pd.read_csv("../input/jobcare-recommendation/lgbm_oof.csv")
lgbm_oof.head()
# %%
from sklearn.metrics import f1_score

print(f1_score(lgbm_oof.target, lgbm_oof.oof_preds > 0.36))
# %%
lgbm_preds = pd.read_csv("../submit/5fold_lightgbm_proba.csv")
submission = pd.read_csv("../input/jobcare-recommendation/sample_submission.csv")

submission["target"] = np.where(lgbm_preds.target > 0.36, 1, 0)
submission.to_csv("../submit/5fold_lightgbm_threshold_0.36.csv", index=False)
# %%
