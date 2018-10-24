import pandas as pd
import numpy as np
import time


df=pd.read_csv("baseline_lgb.csv")
print(df.describe())
df["fullVisitorId"]=df["fullVisitorId"].astype('str')
df["PredictedLogRevenue"]=df["PredictedLogRevenue"].astype(float)
#df["PredictedLogRevenue"]=np.expm1(df["PredictedLogRevenue"])


df.to_csv("baseline_lgb2.csv", index=False)

df2=pd.read_csv("all/sample_submission.csv")
df2['PredictedLogRevenue']=df["PredictedLogRevenue"]
print(df2.head(10))
print(df2.describe())

df2.to_csv("baseline_lgb3.csv", index=False)
