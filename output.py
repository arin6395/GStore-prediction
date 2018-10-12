import pandas as pd
import numpy as np
import time


df=pd.read_csv("baseline_lgb.csv")
print(df.describe())
df["fullVisitorId"]=df["fullVisitorId"].astype('str')
df["PredictedLogRevenue"]=df["PredictedLogRevenue"].astype(float)
df1=df.groupby(["fullVisitorId"])['PredictedLogRevenue'].sum().reset_index()
df1["PredictedLogRevenue"] = np.log1p(df1["PredictedLogRevenue"])
df1.to_csv("baseline_lgb2.csv", index=False)
print(df1.describe())