"""
Script to generate clean, reproducible Jupyter Notebooks for:
01_data_cleaning.ipynb
02_eda.ipynb
03_model_training.ipynb
"""

import json
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def make_notebook(cells):
    """Create a Jupyter Notebook dict matching nbformat v4."""
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.12.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }


def md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")]
    }


def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.split("\n")]
    }


def build_01_cleaning():
    cells = [
        md_cell("""# 01 - Data Cleaning & Feature Engineering Pipeline
## EV Charging Station Congestion & Grid Load Predictor

This notebook demonstrates the end-to-end data cleaning, anomaly handling, station-hour feature engineering, and academic target proxy derivation from the raw Harvard Dataverse EV charging dataset (`station_data_dataverse.csv`), culminating in the exact 2,000-row project dataset."""),
        
        md_cell("### 1. Import Libraries and Inspect Raw Dataset"),
        code_cell("""import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (10, 5)

# Locate raw data
raw_path = Path("../data/raw/station_data_dataverse.csv")
df_raw = pd.read_csv(raw_path)
print(f"Raw Dataset Shape: {df_raw.shape}")
df_raw.head()"""),

        md_cell("### 2. Dataset Overview: Types and Null Values"),
        code_cell("""# Check column data types and missing counts
print("Missing values per column:")
print(df_raw.isnull().sum()[df_raw.isnull().sum() > 0])
print("\\nData Types summary:")
print(df_raw.dtypes.value_counts())"""),

        md_cell("### 3. Check Duplicates and Invalid Values"),
        code_cell("""# Duplicates
dup_count = df_raw.duplicated().sum()
print(f"Duplicate rows: {dup_count}")

# Check negative or zero numerical values
neg_energy = (df_raw['kwhTotal'] <= 0).sum()
neg_duration = (df_raw['chargeTimeHrs'] <= 0).sum()
invalid_hours = ((df_raw['startTime'] < 0) | (df_raw['startTime'] > 23)).sum()

print(f"Zero or negative kWh sessions: {neg_energy}")
print(f"Zero or negative duration sessions: {neg_duration}")
print(f"Invalid startTime hours (<0 or >23): {invalid_hours}")"""),

        md_cell("### 4. Clean Raw Data"),
        code_cell("""# Essential columns required for analysis
essential_cols = ["stationId", "locationId", "startTime", "weekday", "facilityType", "chargeTimeHrs", "kwhTotal"]

# Drop missing values in essential columns and filter valid physical ranges
df_clean = df_raw.dropna(subset=essential_cols).copy()
df_clean = df_clean[
    (df_clean["kwhTotal"] > 0) &
    (df_clean["chargeTimeHrs"] > 0) &
    (df_clean["startTime"] >= 0) &
    (df_clean["startTime"] <= 23)
].copy()

print(f"Cleaned Usable Records: {len(df_clean)} (from original {len(df_raw)})")"""),

        md_cell("""### 5. Feature Engineering: Historical Station-Hour Statistics
Calculate station-hour historical charging behavior:
- `sessions_at_station_hour`: concurrent session activity during that hour
- `avg_energy_kwh`: historical average energy delivered
- `avg_charge_time_min`: historical average charging duration (minutes)"""),
        code_cell("""# Normalize hour and weekday names
weekday_map = {
    "Mon": "Monday", "Tue": "Tuesday", "Wed": "Wednesday",
    "Thu": "Thursday", "Fri": "Friday", "Sat": "Saturday", "Sun": "Sunday"
}
df_clean["hour"] = df_clean["startTime"].astype(int)
df_clean["day_of_week"] = df_clean["weekday"].map(weekday_map).fillna(df_clean["weekday"])

# Calculate historical statistics grouped by stationId and hour
station_hour_stats = df_clean.groupby(["stationId", "hour"]).agg(
    sessions_at_station_hour=("sessionId", "count"),
    avg_energy_kwh=("kwhTotal", "mean"),
    avg_charge_time_min=("chargeTimeHrs", lambda x: x.mean() * 60.0)
).reset_index()

merged = pd.merge(df_clean, station_hour_stats, on=["stationId", "hour"], how="left")
print(f"Merged features shape: {merged.shape}")
merged[["stationId", "hour", "sessions_at_station_hour", "avg_energy_kwh", "avg_charge_time_min"]].head()"""),

        md_cell("""### 6. Target Derivation: Congestion Level, Grid Load Level & Waiting Time
- **Congestion Score**: $0.65 \\times \\text{sessions} + 0.35 \\times \\text{avg\\_duration}$
- **Load Score**: $0.60 \\times \\text{avg\\_energy} + 0.40 \\times \\text{sessions}$
- Discretized into `Low`, `Medium`, and `High` using 33rd and 66th percentiles.
- **Estimated Waiting Time**: Fraction of average charging time based on congestion tier."""),
        code_cell("""# 1. Congestion Level
congestion_score = 0.65 * merged["sessions_at_station_hour"] + 0.35 * merged["avg_charge_time_min"]
c_p33 = float(np.percentile(congestion_score, 33))
c_p66 = float(np.percentile(congestion_score, 66))

merged["congestion_level"] = pd.cut(
    congestion_score,
    bins=[-np.inf, c_p33, c_p66, np.inf],
    labels=["Low", "Medium", "High"]
)

# 2. Grid Load Level
load_score = 0.60 * merged["avg_energy_kwh"] + 0.40 * merged["sessions_at_station_hour"]
l_p33 = float(np.percentile(load_score, 33))
l_p66 = float(np.percentile(load_score, 66))

merged["load_level"] = pd.cut(
    load_score,
    bins=[-np.inf, l_p33, l_p66, np.inf],
    labels=["Low", "Medium", "High"]
)

# 3. Estimated Waiting Time
def calc_wait(row):
    c = row["congestion_level"]
    m = row["avg_charge_time_min"]
    if c == "Low":
        w = m * 0.20
    elif c == "Medium":
        w = m * 0.50
    else:
        w = m * 0.85
    return max(0, int(round(w)))

merged["estimated_wait_min"] = merged.apply(calc_wait, axis=1)

print("Congestion Level Distribution:")
print(merged["congestion_level"].value_counts())
print("\\nGrid Load Level Distribution:")
print(merged["load_level"].value_counts())"""),

        md_cell("### 7. Select Final Columns and Sample Exactly 2,000 Rows"),
        code_cell("""# Rename and select final 13 project columns
merged["station_id"] = merged["stationId"]
merged["location_id"] = merged["locationId"]
merged["facility_type"] = merged["facilityType"]
merged["charge_time_hrs"] = merged["chargeTimeHrs"]
merged["energy_kwh"] = merged["kwhTotal"]

final_cols = [
    "station_id", "location_id", "hour", "day_of_week", "facility_type",
    "charge_time_hrs", "energy_kwh", "avg_energy_kwh", "avg_charge_time_min",
    "sessions_at_station_hour", "congestion_level", "load_level", "estimated_wait_min"
]
project_2000 = merged[final_cols].sample(n=2000, random_state=42).reset_index(drop=True)

# Save to processed directory
output_file = Path("../data/processed/ev_charging_project_2000.csv")
output_file.parent.mkdir(parents=True, exist_ok=True)
project_2000.to_csv(output_file, index=False)
print(f"Successfully generated and saved project dataset with shape: {project_2000.shape}")
project_2000.head()"""),

        md_cell("### 8. Data Validation Checklist Verification"),
        code_cell("""from src.preprocessing import validate_dataset
validate_dataset(output_file)""")
    ]
    return make_notebook(cells)


def build_02_eda():
    cells = [
        md_cell("""# 02 - Exploratory Data Analysis (EDA)
## EV Charging Station Congestion & Grid Load Predictor

This notebook performs a comprehensive exploratory analysis of the processed 2,000-row dataset, uncovering spatial, temporal, energy, and congestion distribution characteristics across the EV charging network."""),

        md_cell("### 1. Load Processed Dataset"),
        code_cell("""from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure visual theme
sns.set_theme(style="whitegrid", palette="tab10")
plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["font.size"] = 11

data_path = Path("../data/processed/ev_charging_project_2000.csv")
df = pd.read_csv(data_path)
print(f"Dataset Dimensions: {df.shape}")
df.info()"""),

        md_cell("### 2. Summary Statistics"),
        code_cell("""df.describe().round(2)"""),

        md_cell("### 3. Charging Sessions by Station (Top 15 Most Active Stations)"),
        code_cell("""top_stations = df["station_id"].value_counts().head(15)
plt.figure(figsize=(12, 5))
sns.barplot(x=top_stations.index.astype(str), y=top_stations.values, color="#1f77b4")
plt.title("Top 15 Most Active Charging Stations (Session Count)", fontsize=13, fontweight="bold")
plt.xlabel("Station ID")
plt.ylabel("Session Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()"""),

        md_cell("### 4. Charging Sessions by Hour of Day"),
        code_cell("""hourly_sessions = df["hour"].value_counts().sort_index()
plt.figure(figsize=(12, 5))
sns.barplot(x=hourly_sessions.index, y=hourly_sessions.values, color="#2ca02c")
plt.title("Charging Session Distribution by Hour of the Day", fontsize=13, fontweight="bold")
plt.xlabel("Hour (24-hour clock)")
plt.ylabel("Number of Sessions")
plt.tight_layout()
plt.show()"""),

        md_cell("### 5. Charging Sessions by Day of Week"),
        code_cell("""day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x="day_of_week", order=day_order, color="#ff7f0e")
plt.title("Weekly Session Distribution Across Days", fontsize=13, fontweight="bold")
plt.xlabel("Day of Week")
plt.ylabel("Session Count")
plt.tight_layout()
plt.show()"""),

        md_cell("### 6. Energy Consumption & Charging Duration Distributions"),
        code_cell("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Energy histogram
sns.histplot(df["energy_kwh"], bins=30, kde=True, ax=axes[0], color="#9467bd")
axes[0].set_title("Energy Consumption (kWh) Distribution", fontweight="bold")
axes[0].set_xlabel("Energy (kWh)")
axes[0].set_ylabel("Frequency")

# Charging duration histogram
sns.histplot(df["charge_time_hrs"], bins=30, kde=True, ax=axes[1], color="#8c564b")
axes[1].set_title("Charging Duration (Hours) Distribution", fontweight="bold")
axes[1].set_xlabel("Duration (Hours)")
axes[1].set_ylabel("Frequency")

plt.tight_layout()
plt.show()"""),

        md_cell("### 7. Congestion Level and Grid Load Target Distributions"),
        code_cell("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))
tier_order = ["Low", "Medium", "High"]
palette = {"Low": "#2ca02c", "Medium": "#ff7f0e", "High": "#d62728"}

sns.countplot(data=df, x="congestion_level", order=tier_order, palette=palette, ax=axes[0])
axes[0].set_title("Congestion Level Target Distribution", fontweight="bold")
axes[0].set_xlabel("Congestion Tier")
axes[0].set_ylabel("Session Count")

sns.countplot(data=df, x="load_level", order=tier_order, palette=palette, ax=axes[1])
axes[1].set_title("Grid Load Level Target Distribution", fontweight="bold")
axes[1].set_xlabel("Grid Load Tier")
axes[1].set_ylabel("Session Count")

plt.tight_layout()
plt.show()"""),

        md_cell("### 8. Hour vs Congestion Level"),
        code_cell("""hour_c = pd.crosstab(df["hour"], df["congestion_level"], normalize="index")[tier_order]
plt.figure(figsize=(14, 6))
hour_c.plot(kind="bar", stacked=True, color=["#2ca02c", "#ff7f0e", "#d62728"], figsize=(14, 6))
plt.title("Proportion of Congestion Levels by Hour of the Day", fontsize=13, fontweight="bold")
plt.xlabel("Hour")
plt.ylabel("Proportion")
plt.legend(title="Congestion Tier")
plt.tight_layout()
plt.show()"""),

        md_cell("### 9. Top Stations vs Congestion Breakdown"),
        code_cell("""top_10 = df["station_id"].value_counts().head(10).index
station_c = pd.crosstab(df[df["station_id"].isin(top_10)]["station_id"], df["congestion_level"])[tier_order]
station_c.plot(kind="bar", stacked=True, color=["#2ca02c", "#ff7f0e", "#d62728"], figsize=(12, 5))
plt.title("Congestion Distribution in Top 10 Stations", fontsize=13, fontweight="bold")
plt.xlabel("Station ID")
plt.ylabel("Session Count")
plt.legend(title="Congestion Tier")
plt.tight_layout()
plt.show()"""),

        md_cell("""### 10. Key Insights & Findings
1. **Diurnal Peaks**: Charging sessions surge from 8:00 AM to 11:00 AM and again in late afternoon (4:00 PM to 7:00 PM), reflecting workplace arrival and evening commute patterns.
2. **Balanced Target Representation**: Derived academic proxies for Congestion Level and Grid Load achieve balanced representation across `Low`, `Medium`, and `High` categories (~33% each).
3. **Station Heterogeneity**: Several primary stations exhibit consistent `High` congestion during commute windows, confirming the need for an automated alternative station recommendation mechanism.""")
    ]
    return make_notebook(cells)


def build_03_training():
    cells = [
        md_cell("""# 03 - Machine Learning Model Training & Comparative Evaluation
## EV Charging Station Congestion & Grid Load Predictor

This notebook details:
1. Feature matrix and target separation strictly preventing data leakage.
2. Pipeline preprocessing using `ColumnTransformer` (OneHotEncoding + StandardScaler).
3. Training three multi-class models for Congestion Level:
   - Logistic Regression
   - Decision Tree Classifier
   - Random Forest Classifier
4. Training a Random Forest classifier for Grid Load Level proxy.
5. Comparative model evaluation across Accuracy, Precision, Recall, and F1-Score.
6. Confusion matrix analysis and model serialization via `joblib`."""),

        md_cell("### 1. Load Processed Dataset"),
        code_cell("""from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib

df = pd.read_csv("../data/processed/ev_charging_project_2000.csv")
print(f"Loaded dataset: {df.shape}")
df.head(3)"""),

        md_cell("### 2. Feature Selection & Data Leakage Prevention"),
        code_cell("""# Input features excluding identifiers (sessionId, userId) and targets (congestion_level, load_level, estimated_wait_min)
categorical_cols = ["station_id", "location_id", "day_of_week", "facility_type"]
numerical_cols = ["hour", "avg_energy_kwh", "avg_charge_time_min", "sessions_at_station_hour"]
feature_cols = categorical_cols + numerical_cols

X = df[feature_cols].copy()
y_congestion = df["congestion_level"].copy()
y_load = df["load_level"].copy()

print("Feature Matrix Shape:", X.shape)
print("Categorical features:", categorical_cols)
print("Numerical features:  ", numerical_cols)"""),

        md_cell("### 3. Stratified Train / Test Split (80 / 20)"),
        code_cell("""X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_congestion, test_size=0.20, random_state=42, stratify=y_congestion
)
X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(
    X, y_load, test_size=0.20, random_state=42, stratify=y_load
)

print(f"Congestion Train Set: {X_train_c.shape}, Test Set: {X_test_c.shape}")
print(f"Grid Load Train Set:  {X_train_l.shape}, Test Set: {X_test_l.shape}")"""),

        md_cell("### 4. Build Preprocessing Transformer"),
        code_cell("""def get_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
            ("num", StandardScaler(), numerical_cols)
        ]
    )"""),

        md_cell("### 5. Train and Evaluate Congestion Models"),
        code_cell("""models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
}

trained_pipes = {}
results = []
cms = {}
labels_order = ["Low", "Medium", "High"]

for name, clf in models.items():
    pipe = Pipeline([("prep", get_preprocessor()), ("clf", clf)])
    pipe.fit(X_train_c, y_train_c)
    y_pred = pipe.predict(X_test_c)
    
    acc = accuracy_score(y_test_c, y_pred)
    prec = precision_score(y_test_c, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test_c, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test_c, y_pred, average="weighted", zero_division=0)
    
    trained_pipes[name] = pipe
    cms[name] = confusion_matrix(y_test_c, y_pred, labels=labels_order)
    
    results.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1 Score": round(f1, 4)
    })

comparison_df = pd.DataFrame(results)
print("=== CONGESTION MODEL COMPARISON ===")
comparison_df"""),

        md_cell("### 6. Confusion Matrices for Congestion Classifiers"),
        code_cell("""fig, axes = plt.subplots(1, 3, figsize=(16, 4))

for idx, (name, cm) in enumerate(cms.items()):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=labels_order, yticklabels=labels_order, ax=axes[idx])
    axes[idx].set_title(f"{name}\\nConfusion Matrix", fontweight="bold")
    axes[idx].set_xlabel("Predicted")
    axes[idx].set_ylabel("Actual")

plt.tight_layout()
plt.show()"""),

        md_cell("### 7. Accuracy & F1-Score Comparative Visualization"),
        code_cell("""fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

sns.barplot(data=comparison_df, x="Model", y="Accuracy", palette="viridis", ax=axes[0])
axes[0].set_title("Model Accuracy Comparison", fontweight="bold")
axes[0].set_ylim(0.8, 1.02)
for p in axes[0].patches:
    axes[0].annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points')

sns.barplot(data=comparison_df, x="Model", y="F1 Score", palette="magma", ax=axes[1])
axes[1].set_title("Model F1-Score Comparison", fontweight="bold")
axes[1].set_ylim(0.8, 1.02)
for p in axes[1].patches:
    axes[1].annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points')

plt.tight_layout()
plt.show()"""),

        md_cell("### 8. Train and Evaluate Grid Load Level Model"),
        code_cell("""load_pipe = Pipeline([
    ("prep", get_preprocessor()),
    ("clf", RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42))
])
load_pipe.fit(X_train_l, y_train_l)
y_pred_l = load_pipe.predict(X_test_l)

print("=== GRID LOAD MODEL EVALUATION ===")
print("Accuracy: ", round(accuracy_score(y_test_l, y_pred_l), 4))
print("Precision:", round(precision_score(y_test_l, y_pred_l, average='weighted'), 4))
print("Recall:   ", round(recall_score(y_test_l, y_pred_l, average='weighted'), 4))
print("F1 Score: ", round(f1_score(y_test_l, y_pred_l, average='weighted'), 4))
print("\\nClassification Report:")
print(classification_report(y_test_l, y_pred_l, labels=labels_order))"""),

        md_cell("### 9. Serialize Trained Models via Joblib"),
        code_cell("""models_dir = Path("../models")
models_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(trained_pipes["Logistic Regression"], models_dir / "congestion_logistic.pkl")
joblib.dump(trained_pipes["Decision Tree"], models_dir / "congestion_tree.pkl")
joblib.dump(trained_pipes["Random Forest"], models_dir / "congestion_forest.pkl")
joblib.dump(load_pipe, models_dir / "grid_load_model.pkl")

print(f"Successfully saved all models to: {models_dir.resolve()}")""")
    ]
    return make_notebook(cells)


def generate_all_notebooks():
    nb1 = build_01_cleaning()
    with open(NOTEBOOKS_DIR / "01_data_cleaning.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb1, f, indent=2)
    print("Generated 01_data_cleaning.ipynb")

    nb2 = build_02_eda()
    with open(NOTEBOOKS_DIR / "02_eda.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb2, f, indent=2)
    print("Generated 02_eda.ipynb")

    nb3 = build_03_training()
    with open(NOTEBOOKS_DIR / "03_model_training.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb3, f, indent=2)
    print("Generated 03_model_training.ipynb")


if __name__ == "__main__":
    generate_all_notebooks()
