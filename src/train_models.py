"""
Model Training and Evaluation Module
EV Charging Station Congestion & Grid Load Predictor

This module:
1. Loads the processed 2,000-row dataset.
2. Prepares features and target labels (preventing data leakage).
3. Fits the feature preprocessor (OneHotEncoder + StandardScaler).
4. Trains 3 multi-class ML models for Congestion Level:
   - Logistic Regression
   - Decision Tree Classifier
   - Random Forest Classifier
5. Trains 1 Random Forest model for Grid Load Level.
6. Evaluates all models (Accuracy, Precision, Recall, F1, Confusion Matrix).
7. Serializes trained models and artifacts to models/ directory using joblib.
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import joblib

# Reconfigure stdout/stderr for Unicode paths on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "ev_charging_project_2000.csv"
MODELS_DIR = PROJECT_ROOT / "models"


def get_feature_columns():
    """Return feature column groupings avoiding data leakage."""
    categorical_cols = ["station_id", "location_id", "day_of_week", "facility_type"]
    numerical_cols = ["hour", "avg_energy_kwh", "avg_charge_time_min", "sessions_at_station_hour"]
    all_features = categorical_cols + numerical_cols
    return categorical_cols, numerical_cols, all_features


def build_preprocessor(categorical_cols, numerical_cols):
    """Build ColumnTransformer for categorical and numerical features."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            ),
            (
                "num",
                StandardScaler(),
                numerical_cols,
            ),
        ]
    )
    return preprocessor


def train_and_evaluate_all():
    """Train all congestion and grid-load models, evaluate, and save artifacts."""
    if not PROCESSED_DATA_PATH.exists():
        from src.preprocessing import create_project_dataset
        create_project_dataset()

    df = pd.read_csv(PROCESSED_DATA_PATH)
    print("Dataset loaded successfully.\n")
    print("Training models...\n")

    categorical_cols, numerical_cols, feature_cols = get_feature_columns()
    X = df[feature_cols].copy()
    y_congestion = df["congestion_level"].copy()
    y_load = df["load_level"].copy()

    # Train / Test split for Congestion Level (80/20 stratified)
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X, y_congestion, test_size=0.20, random_state=42, stratify=y_congestion
    )

    # Train / Test split for Grid Load Level (80/20 stratified)
    X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(
        X, y_load, test_size=0.20, random_state=42, stratify=y_load
    )

    # Congestion models dictionary
    congestion_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    }

    congestion_pipelines = {}
    metrics_summary = {}

    labels_order = ["Low", "Medium", "High"]

    for name, clf in congestion_models.items():
        preprocessor = build_preprocessor(categorical_cols, numerical_cols)
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
        pipe.fit(X_train_c, y_train_c)
        y_pred = pipe.predict(X_test_c)

        acc = float(accuracy_score(y_test_c, y_pred))
        prec = float(precision_score(y_test_c, y_pred, average="weighted", zero_division=0))
        rec = float(recall_score(y_test_c, y_pred, average="weighted", zero_division=0))
        f1 = float(f1_score(y_test_c, y_pred, average="weighted", zero_division=0))
        cm = confusion_matrix(y_test_c, y_pred, labels=labels_order).tolist()

        congestion_pipelines[name] = pipe
        metrics_summary[name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1,
            "Confusion Matrix": cm,
            "Labels": labels_order,
        }

        print(name)
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1 Score:  {f1:.4f}\n")

    # Grid Load Model (Random Forest Classifier)
    load_preprocessor = build_preprocessor(categorical_cols, numerical_cols)
    load_pipe = Pipeline(
        steps=[
            ("preprocessor", load_preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)),
        ]
    )
    load_pipe.fit(X_train_l, y_train_l)
    y_pred_l = load_pipe.predict(X_test_l)

    load_acc = float(accuracy_score(y_test_l, y_pred_l))
    load_prec = float(precision_score(y_test_l, y_pred_l, average="weighted", zero_division=0))
    load_rec = float(recall_score(y_test_l, y_pred_l, average="weighted", zero_division=0))
    load_f1 = float(f1_score(y_test_l, y_pred_l, average="weighted", zero_division=0))
    load_cm = confusion_matrix(y_test_l, y_pred_l, labels=labels_order).tolist()

    metrics_summary["Grid Load Model (Random Forest)"] = {
        "Accuracy": load_acc,
        "Precision": load_prec,
        "Recall": load_rec,
        "F1 Score": load_f1,
        "Confusion Matrix": load_cm,
        "Labels": labels_order,
    }

    # Ensure models directory exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Save models using joblib
    joblib.dump(congestion_pipelines["Logistic Regression"], MODELS_DIR / "congestion_logistic.pkl")
    joblib.dump(congestion_pipelines["Decision Tree"], MODELS_DIR / "congestion_tree.pkl")
    joblib.dump(congestion_pipelines["Random Forest"], MODELS_DIR / "congestion_forest.pkl")
    joblib.dump(load_pipe, MODELS_DIR / "grid_load_model.pkl")

    # Save metrics JSON for the app and notebooks
    with open(MODELS_DIR / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=4)

    # Save station and profile metadata for fast runtime inference and recommendations
    station_meta = {
        "unique_stations": sorted([int(x) for x in df["station_id"].unique()]),
        "unique_days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        "unique_hours": sorted([int(x) for x in df["hour"].unique()]),
        "categorical_cols": categorical_cols,
        "numerical_cols": numerical_cols,
        "feature_cols": feature_cols,
    }
    with open(MODELS_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(station_meta, f, indent=4)

    print("Models saved successfully.")
    return metrics_summary


if __name__ == "__main__":
    train_and_evaluate_all()
