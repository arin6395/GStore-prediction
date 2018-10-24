import pandas as pd
import numpy as np
import time
import gc
import json
import os
from datetime import datetime	
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
from sklearn import linear_model

np.set_printoptions(threshold='nan')


train_df = pd.read_csv("all/proccessed_train2.csv")
test_df=pd.read_csv("all/proccessed_test2.csv")


test_df=test_df[['fullVisitorId']].astype(str)
print(len(test_df))
print(test_df.head())
test_df2=test_df.groupby(["fullVisitorId"],as_index=False)
print(len(test_df2))
print(test_df2.head())
