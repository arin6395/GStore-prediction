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
from sklearn import metrics
import seaborn as sns
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
sns.set_style("dark")

train_df = pd.read_csv("all/proccessed_train2.csv",parse_dates=['date'])
test_df=pd.read_csv("all/proccessed_test2.csv",parse_dates=['date'])
#test_df["totals.transactionRevenue"]=0

print(train_df.shape)
print(test_df.shape)

cat_many_label_cols = ["channelGrouping", "device.browser", "device.operatingSystem", 
            "geoNetwork.city", "geoNetwork.continent", 
            "geoNetwork.country", "geoNetwork.metro",
            "geoNetwork.subContinent","trafficSource.adwordsClickInfo.gclId", 
            "trafficSource.campaign",
            "trafficSource.keyword", "trafficSource.medium", 
            "trafficSource.referralPath", "trafficSource.source"]

cat_few_label_cols = ["device.deviceCategory","trafficSource.adwordsClickInfo.adNetworkType",
                     "trafficSource.adwordsClickInfo.slot"]


for col in cat_many_label_cols:
    print(col)
    lbl1 = LabelEncoder()
    lbl1.fit(list(train_df[col].values.astype('str')))
    train_df[col] = lbl1.transform(list(train_df[col].values.astype('str')))
    lbl2 = LabelEncoder()
    lbl2.fit(list(test_df[col].values.astype('str')))
    test_df[col] = lbl2.transform(list(test_df[col].values.astype('str')))
    
train_df = pd.get_dummies(train_df,columns=cat_few_label_cols)
test_df = pd.get_dummies(test_df,columns=cat_few_label_cols)

train_df['totals.transactionRevenue']=train_df['totals.transactionRevenue'].astype(float)

val_df = train_df[train_df['date']>datetime(2017,6,1)]
dropcols = ['fullVisitorId']
train_x = train_df.drop(dropcols,axis=1)
test_x = test_df.drop(dropcols,axis=1)


dev_x = train_x[train_df['date']<=datetime(2017,6,1)]
val_x = train_x[train_df['date']>datetime(2017,6,1)]
dev_y = dev_x["totals.transactionRevenue"].values/10000000000
val_y = val_x["totals.transactionRevenue"].values/10000000000
dev_x.drop(["totals.transactionRevenue",'date'],axis=1,inplace=True)
val_x.drop(["totals.transactionRevenue",'date'],axis=1,inplace=True)
test_x.drop(["date"],axis=1,inplace=True)

print(train_df.shape)
print(val_df.shape)
print(test_df.shape)



lgb_params = {
        "objective" : "regression",
        "metric" : "rmse", 
        "num_leaves" : 128,
        'max_depth': 16,  
        'max_bin': 255,
        "min_child_samples" : 50,
        "learning_rate" : 0.03,
        'verbose': 0,
        "bagging_fraction" : 0.7,
        "feature_fraction" : 0.7,
        "bagging_seed" : 2018
    }

dtrain = lgb.Dataset(dev_x, label=dev_y)
dvalid = lgb.Dataset(val_x, label=val_y)

evals_results = {}
print("Training the model...")

start = datetime.now()
lgb_model = lgb.train(lgb_params, 
                 dtrain, 
                 valid_sets=[dtrain, dvalid], 
                 valid_names=['train','valid'], 
                 evals_result=evals_results, 
                 num_boost_round=500,
                 early_stopping_rounds=100,
                 verbose_eval=50, 
                 feval=None)
print("Total time taken : ", datetime.now()-start)

pred_test_lgb = lgb_model.predict(test_x, num_iteration=lgb_model.best_iteration)
pred_val_lgb = lgb_model.predict(val_x, num_iteration=lgb_model.best_iteration)



pred_val_lgb[pred_val_lgb<0] = 0
val_pred_df = pd.DataFrame({"fullVisitorId":val_df["fullVisitorId"].values})
val_pred_df["transactionRevenue"] = val_df["totals.transactionRevenue"].values
val_pred_df["PredictedRevenue"] = pred_val_lgb*10000000000
val_pred_df = val_pred_df.groupby("fullVisitorId")["transactionRevenue", "PredictedRevenue"].sum().reset_index()
print(np.sqrt(metrics.mean_squared_error(val_pred_df["transactionRevenue"].values, val_pred_df["PredictedRevenue"].values)))



fold_importance_df = pd.DataFrame()
fold_importance_df["feature"] = val_x.columns
fold_importance_df["importance"] = lgb_model.feature_importance()
plt.figure(figsize=(18,20))
sns.barplot(x='importance',y='feature',data=fold_importance_df.sort_values(by="importance", ascending=False))
plt.show()

train_id = train_df["fullVisitorId"].values
test_id = test_df["fullVisitorId"].values
sub_df = pd.DataFrame({"fullVisitorId":test_id})
sub_df["fullVisitorId"]=sub_df["fullVisitorId"].astype('float')
pred_test_lgb[pred_test_lgb<0] = 0
sub_df["PredictedLogRevenue"] = pred_test_lgb*10000000000
sub_df = sub_df.groupby(["fullVisitorId"])["PredictedLogRevenue"].sum().reset_index()
sub_df.columns = ["fullVisitorId", "PredictedLogRevenue"]
sub_df["PredictedLogRevenue"] = np.log1p(sub_df["PredictedLogRevenue"])
sub_df.to_csv("baseline_lgb.csv", index=False)
print(sub_df.describe())