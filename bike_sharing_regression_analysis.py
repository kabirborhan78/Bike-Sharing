import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 130

DATA_PATH = "clean_bike_sharing_2000.csv"  # CSV must be in the SAME folder as this script
CHART_DIR = "charts"  # charts folder will be created automatically here

import os
os.makedirs(CHART_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
print(df.describe(include="all"))
print(df.isnull().sum())

summary = {}
summary["n_rows"] = int(df.shape[0])
summary["n_cols"] = int(df.shape[1])
summary["missing_values"] = int(df.isnull().sum().sum())
summary["cnt_mean"] = float(df["cnt"].mean())
summary["cnt_median"] = float(df["cnt"].median())
summary["cnt_std"] = float(df["cnt"].std())
summary["cnt_min"] = float(df["cnt"].min())
summary["cnt_max"] = float(df["cnt"].max())

# ---------- Charts ----------

# 1. Distribution of cnt
plt.figure(figsize=(7, 4.5))
sns.histplot(df["cnt"], bins=40, kde=True, color="#2E5395")
plt.title("Distribution of Bike Rental Count (cnt)")
plt.xlabel("cnt (rentals)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/01_cnt_distribution.png")
plt.close()

# 2. Average cnt by season
season_map = {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}
df["season_label"] = df["season"].map(season_map)
plt.figure(figsize=(7, 4.5))
order = ["Spring", "Summer", "Fall", "Winter"]
sns.barplot(data=df, x="season_label", y="cnt", order=order, palette="Blues_d", errorbar=None)
plt.title("Average Rentals by Season")
plt.xlabel("Season")
plt.ylabel("Average cnt")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/02_avg_by_season.png")
plt.close()

# 3. Average cnt by month
plt.figure(figsize=(8, 4.5))
monthly = df.groupby("mnth")["cnt"].mean().reset_index()
sns.barplot(data=monthly, x="mnth", y="cnt", color="#4472C4")
plt.title("Average Rentals by Month")
plt.xlabel("Month")
plt.ylabel("Average cnt")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/03_avg_by_month.png")
plt.close()

# 4. temp vs cnt scatter
plt.figure(figsize=(7, 4.5))
sns.scatterplot(data=df, x="temp", y="cnt", alpha=0.4, color="#2E5395", s=18)
plt.title("Temperature vs Rental Count")
plt.xlabel("Normalized Temperature")
plt.ylabel("cnt")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/04_temp_vs_cnt.png")
plt.close()

# 5. Working day comparison
plt.figure(figsize=(6, 4.5))
sns.boxplot(data=df, x="workingday", y="cnt", palette=["#8FAADC", "#2E5395"])
plt.title("Rental Count: Working Day vs Non-Working Day")
plt.xlabel("Working Day (0 = No, 1 = Yes)")
plt.ylabel("cnt")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/05_workingday_boxplot.png")
plt.close()

# 6. Correlation heatmap
plt.figure(figsize=(7, 5.5))
num_cols = ["season", "mnth", "holiday", "workingday", "temp", "hum", "windspeed", "cnt"]
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", cbar=True)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/06_correlation_heatmap.png")
plt.close()

summary["corr_with_cnt"] = corr["cnt"].drop("cnt").round(3).to_dict()

# ---------- Modeling ----------
feature_cols = ["season", "mnth", "holiday", "workingday", "temp", "hum", "windspeed"]
categorical = ["season", "mnth"]
numeric = ["holiday", "workingday", "temp", "hum", "windspeed"]

X = df[feature_cols]
y = df["cnt"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first"), categorical),
    ],
    remainder="passthrough",
)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=300, random_state=42, max_depth=None),
    "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42),
}

results = {}
predictions = {}
feature_importance = {}

for name, model in models.items():
    pipe = Pipeline(steps=[("prep", preprocess), ("model", model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    predictions[name] = preds

    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    results[name] = {"R2": round(float(r2), 4), "RMSE": round(float(rmse), 2), "MAE": round(float(mae), 2)}

    if name in ("Random Forest Regressor", "Gradient Boosting Regressor"):
        feat_names = pipe.named_steps["prep"].get_feature_names_out()
        importances = pipe.named_steps["model"].feature_importances_
        fi = sorted(zip(feat_names, importances), key=lambda x: -x[1])[:8]
        feature_importance[name] = [(f, round(float(v), 4)) for f, v in fi]

print(json.dumps(results, indent=2))
print(json.dumps(feature_importance, indent=2))

summary["results"] = results
summary["feature_importance"] = feature_importance

best_model_name = max(results.items(), key=lambda kv: kv[1]["R2"])[0]
summary["best_model"] = best_model_name

# 7. Actual vs predicted for best model
plt.figure(figsize=(6.5, 6))
best_preds = predictions[best_model_name]
plt.scatter(y_test, best_preds, alpha=0.4, color="#2E5395", s=18)
lims = [0, max(y_test.max(), best_preds.max()) + 10]
plt.plot(lims, lims, "r--", linewidth=1.5)
plt.xlabel("Actual cnt")
plt.ylabel("Predicted cnt")
plt.title(f"Actual vs Predicted ({best_model_name})")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/07_actual_vs_predicted.png")
plt.close()

# 8. Model comparison bar chart (R2)
plt.figure(figsize=(7, 4.5))
names = list(results.keys())
r2s = [results[n]["R2"] for n in names]
sns.barplot(x=names, y=r2s, palette="Blues_d")
plt.ylabel("R\u00b2 Score")
plt.title("Model Comparison (R\u00b2 Score on Test Set)")
plt.xticks(rotation=10)
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/08_model_comparison.png")
plt.close()

# 9. Feature importance chart for best tree model
tree_model_for_chart = "Random Forest Regressor" if "Random Forest Regressor" in feature_importance else best_model_name
if tree_model_for_chart in feature_importance:
    fi = feature_importance[tree_model_for_chart]
    labels = [f.replace("cat__", "").replace("remainder__", "") for f, v in fi]
    vals = [v for f, v in fi]
    plt.figure(figsize=(7.5, 4.5))
    sns.barplot(x=vals, y=labels, color="#2E5395")
    plt.title(f"Feature Importance ({tree_model_for_chart})")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(f"{CHART_DIR}/09_feature_importance.png")
    plt.close()

with open("summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("DONE")
print("Best model:", best_model_name)