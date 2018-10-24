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


#cleaning
dropcols = [c for c in train_df.columns if train_df[c].nunique(dropna=True)==1]
dropcols.extend(['totals.newVisits','device.isMobile',"visitId","sessionId","geoNetwork.region","geoNetwork.networkDomain","trafficSource.adContent",'trafficSource.adwordsClickInfo.page'])
print(dropcols)

train_df.drop(dropcols,axis=1,inplace=True,errors='ignore')
test_df.drop(dropcols,axis=1,inplace=True,errors='ignore')


test_df['totals.pageviews'].fillna(0,inplace=True)
test_df['totals.hits'].fillna(0,inplace=True)
test_df[['totals.hits','totals.pageviews']] = test_df[['totals.hits','totals.pageviews']].astype(np.int)

train_df['totals.pageviews'].fillna(0,inplace=True)
train_df['totals.hits'].fillna(0,inplace=True)
train_df[['totals.hits','totals.pageviews']] = train_df[['totals.hits','totals.pageviews']].astype(np.int)

cols = ['trafficSource.adwordsClickInfo.adNetworkType','trafficSource.adwordsClickInfo.gclId','trafficSource.adwordsClickInfo.slot']
train_df[cols] = train_df[cols].fillna("No_Ad")
train_df['trafficSource.referralPath'].fillna("No_Path",inplace=True)
train_df['trafficSource.keyword'].fillna("(not provided)",inplace=True)
train_df['trafficSource.medium'].replace(to_replace=["(None)"],value="(not set)",inplace=True)

test_df[cols] = test_df[cols].fillna("No_Ad")
test_df['trafficSource.referralPath'].fillna("No_Path",inplace=True)
test_df['trafficSource.keyword'].fillna("(not provided)",inplace=True)
test_df['trafficSource.medium'].replace(to_replace=["(None)"],value="(not set)",inplace=True)


#adding date attributes
train_df['year'] = train_df.date.dt.year
train_df['month'] = train_df.date.dt.month
train_df['dayofmonth'] = train_df.date.dt.day
train_df['dayofweek'] = train_df.date.dt.dayofweek
train_df['dayofyear'] = train_df.date.dt.dayofyear
train_df['weekofyear'] = train_df.date.dt.weekofyear
train_df['is_month_start'] = (train_df.date.dt.is_month_start).astype(int)
train_df['is_month_end'] = (train_df.date.dt.is_month_end).astype(int)
train_df['quarter'] = train_df.date.dt.quarter

test_df['year'] = test_df.date.dt.year
test_df['month'] = test_df.date.dt.month
test_df['dayofmonth'] = test_df.date.dt.day
test_df['dayofweek'] = test_df.date.dt.dayofweek
test_df['dayofyear'] = test_df.date.dt.dayofyear
test_df['weekofyear'] = test_df.date.dt.weekofyear
test_df['is_month_start'] = (test_df.date.dt.is_month_start).astype(int)
test_df['is_month_end'] = (test_df.date.dt.is_month_end).astype(int)
test_df['quarter'] = test_df.date.dt.quarter

#adding time attributes
train_df['visitStartTime']=pd.to_datetime(train_df['visitStartTime'],unit='s')
train_df['hour_of_day']=train_df['visitStartTime'].dt.hour
test_df['visitStartTime']=pd.to_datetime(test_df['visitStartTime'],unit='s')
test_df['hour_of_day']=test_df['visitStartTime'].dt.hour


#adding custom KPIs
train_df['hitsPerPage']=round(train_df['totals.hits']/train_df['totals.pageviews'],2)
test_df['hitsPerPage']=round(test_df['totals.hits']/test_df['totals.pageviews'],2)

train_df["totals.transactionRevenue"].fillna(0,inplace=True)
train_df["totals.transactionRevenue"] = train_df["totals.transactionRevenue"].astype(np.float)

dropcols = ['visitStartTime']
train_df.drop(dropcols,axis=1,inplace=True,errors='ignore')
test_df.drop(dropcols,axis=1,inplace=True,errors='ignore')

train_df['hitsPerPage'].replace(to_replace=[float('inf')],value=0,inplace=True)
test_df['hitsPerPage'].replace(to_replace=[float('inf')],value=0,inplace=True)

train_df.to_csv("all/proccessed_train2.csv",index=False)
test_df.to_csv("all/proccessed_test2.csv",index=False)
