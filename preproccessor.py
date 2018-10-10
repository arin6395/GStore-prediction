import pandas as pd
import numpy as np
import time
import gc
import json
import os
from datetime import datetime
from pandas.io.json import json_normalize
#import matplotlib.pyplot as plt
#from sklearn import linear_model

#json breaking
def load_df(csv_path='all/train.csv', nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    
    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     parse_dates=['date'],
                     dtype={'fullVisitorId': 'str'}, # Important!!
                     nrows=nrows)
    
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
    return df

train_df = load_df()
test_df = load_df("all/test.csv")

dropcols = [c for c in train_df.columns if train_df[c].nunique(dropna=True)==1]
dropcols.remove('totals.bounces')
dropcols.remove('totals.newVisits')
print(dropcols)

train_df.drop(dropcols,axis=1,inplace=True,errors='ignore')
test_df.drop(dropcols,axis=1,inplace=True,errors='ignore')

train_df.to_csv("all/proccessed_train.csv")
test_df.to_csv("all/proccessed_test.csv")
