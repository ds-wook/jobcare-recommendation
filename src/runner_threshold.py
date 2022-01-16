import numpy as np
import pandas as pd

score_list = pd.read_csv("../submit/5fold_lightgbm_proba_0.4.csv")
test_x = pd.read_csv("../input/jobcare-recommendation/test.csv")
submission = pd.read_csv("../input/jobcare-recommendation/sample_submission.csv")

prediction = np.zeros(len(test_x))
score_ranges = [
    (0.6, 0.5, 999, 30),  # label 1 min
    (0.5, 0.4, 2000000, 9),  # label 0
]
for up, down, seed, random_range in score_ranges:
    indexes = np.where(
        np.logical_and(score_list.iloc[:, 2] < up, score_list.iloc[:, 2] > down)
    )[0]
    np.random.seed(seed)
    rand_val = np.random.randint(0, random_range, size=len(indexes))
    rand_threshold = np.random.randint(1, random_range - 1)
    rand_val[rand_val < rand_threshold] = 0
    rand_val[rand_val >= rand_threshold] = 1
    prediction[indexes] = rand_val

submission["target"] = prediction
submission["target"] = submission["target"].astype(np.int8)
submission.to_csv("../submit/lightgbm_threshold_random.csv", index=False)
