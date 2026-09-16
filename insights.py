import pandas as pd
df=pd.read_csv("../data/clean_bike_sharing_2000.csv")
print("Average by season:
",df.groupby("season").cnt.mean())
print("Average by month:
",df.groupby("mnth").cnt.mean())
print("Average by workingday:
",df.groupby("workingday").cnt.mean())
print("Correlations:
",df[["temp","hum","windspeed","cnt"]].corr()["cnt"].sort_values(ascending=False))
