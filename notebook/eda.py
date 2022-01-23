# %%
import numpy as np
import pandas as pd


# %%
path = "../input/jobcare-recommendation/"
train = pd.read_csv(path + "train.csv")
test = pd.read_csv(path + "test.csv")
# %%
train.head()
# %%
test.head()
# %%
train_person_rn = train["person_rn"].unique().tolist()
test_person_rn = test["person_rn"].unique().tolist()
# %%
set(test_person_rn) - set(train_person_rn)
# %%
