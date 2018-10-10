import pandas as pd
import numpy as np
import time
import gc
import json
import os
from datetime import datetime
from pandas.io.json import json_normalize

np.set_printoptions(threshold='nan')


train_df = pd.read_csv("all/proccessed_train.csv")
test_df=pd.read_csv("all/proccessed_test.csv")

print(train_df.shape)

print(train_df.describe())
