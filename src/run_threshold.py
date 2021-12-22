import numpy as np
import pandas as pd

score_list = pd.read_csv("../submit/5fold_lightgbm_proba.csv")
test_x = pd.read_csv("../input/jobcare-recommendation/test.csv")
submission = pd.read_csv("../input/jobcare-recommendation/sample_submission.csv")

prediction = np.zeros(len(test_x))
score_ranges = [
    (0.55, 0.50, 999, 30),  # label 1 min
    (0.50, 0.45, 2000000, 9),  # label 0
]
for up, down, seed, random_range in score_ranges:
    indexes = np.where(
        np.logical_and(score_list.target < up, score_list.target > down)
    )[0]
    print(indexes)
    np.random.seed(seed)
    rand_val = np.random.randint(0, random_range, size=len(indexes))
    print(rand_val)
    rand_threshold = np.random.randint(1, random_range - 1)
    print(rand_threshold)
    rand_val[rand_val < rand_threshold] = 0
    rand_val[rand_val >= rand_threshold] = 1
    prediction[indexes] = rand_val

submission["target"] = prediction
submission.to_csv("../submit/threshold_random.csv", index=False)
