# %%
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# %%
path = "../input/jobcare-recommendation/"
train = pd.read_csv(path + "train.csv")
# %%
train["contents_open_dt"].head()
# %%
train["contents_open_dt"] = pd.to_datetime(train["contents_open_dt"])
# %%
train[["id", "contents_open_dt"]].sort_values(by="contents_open_dt")
# %%
train.groupby(["id", "contents_open_dt"])["target"].sum().to_frame() >= 2
# %%
