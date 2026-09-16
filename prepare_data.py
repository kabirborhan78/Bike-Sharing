import pandas as pd
# Place the official UCI hour.csv beside this script before running.
df=pd.read_csv("hour.csv")
cols=["season","mnth","holiday","workingday","temp","hum","windspeed","cnt"]
valid=df[cols].dropna().drop_duplicates()
valid=valid[(valid.season.between(1,4))&(valid.mnth.between(1,12))&(valid.holiday.isin([0,1]))&(valid.workingday.isin([0,1]))&(valid.temp.between(0,1))&(valid.hum.between(0,1))&(valid.windspeed>=0)&(valid.cnt>=0)]
out=valid.sample(n=2000,random_state=42).reset_index(drop=True)
out.to_csv("../data/raw_bike_sharing_2000.csv",index=False)
print(out.shape)
