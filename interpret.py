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

print(train_df.info())
print(train_df['totals.transactionRevenue'].describe())

target_sums = train_df.groupby("fullVisitorId")["totals.transactionRevenue"].sum().reset_index()
plt.scatter(range(target_sums.shape[0]), np.sort(np.log1p(target_sums["totals.transactionRevenue"].values)))
plt.xlabel('index')
plt.ylabel('TransactionRevenue')
plt.show()

plt.scatter(range(target_sums.shape[0]), np.sort(target_sums["totals.transactionRevenue"].values/10000000000))
plt.xlabel('index')
plt.ylabel('TransactionRevenue')
plt.show()