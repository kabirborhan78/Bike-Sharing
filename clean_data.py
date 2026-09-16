import pandas as pd
df=pd.read_csv("../data/raw_bike_sharing_2000.csv").drop_duplicates()
for c in df.columns: df[c]=pd.to_numeric(df[c],errors="coerce")
df=df.dropna()
df=df[(df.season.between(1,4))&(df.mnth.between(1,12))&(df.holiday.isin([0,1]))&(df.workingday.isin([0,1]))&(df.temp.between(0,1))&(df.hum.between(0,1))&(df.windspeed>=0)&(df.cnt>=0)]
assert len(df)==2000
df.to_csv("../data/clean_bike_sharing_2000.csv",index=False)
print(df.shape)
