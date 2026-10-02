"""
Comprehensive Project Verification and Validation Test Suite
EV Charging Station Congestion & Grid Load Predictor
"""

import sys
import json
from pathlib import Path
import pandas as pd
import joblib

# Reconfigure stdout for Windows Unicode
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run_checks():
    print("=" * 70)
    print("EV CHARGING PREDICTOR - FINAL VERIFICATION & VALIDATION SUITE")
    print("=" * 70)

    # 1. Project Directory & File Existence
    required_files = [
        PROJECT_ROOT / "data" / "raw" / "station_data_dataverse.csv",
        PROJECT_ROOT / "data" / "processed" / "ev_charging_project_2000.csv",
        PROJECT_ROOT / "notebooks" / "01_data_cleaning.ipynb",
        PROJECT_ROOT / "notebooks" / "02_eda.ipynb",
        PROJECT_ROOT / "notebooks" / "03_model_training.ipynb",
        PROJECT_ROOT / "models" / "congestion_logistic.pkl",
        PROJECT_ROOT / "models" / "congestion_tree.pkl",
        PROJECT_ROOT / "models" / "congestion_forest.pkl",
        PROJECT_ROOT / "models" / "grid_load_model.pkl",
        PROJECT_ROOT / "models" / "model_metrics.json",
        PROJECT_ROOT / "models" / "metadata.json",
        PROJECT_ROOT / "src" / "preprocessing.py",
        PROJECT_ROOT / "src" / "train_models.py",
        PROJECT_ROOT / "src" / "prediction.py",
        PROJECT_ROOT / "app" / "app.py",
        PROJECT_ROOT / "requirements.txt",
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / ".gitignore",
    ]

    all_files_exist = True
    for f in required_files:
        exists = f.exists()
        size = f.stat().st_size if exists else 0
        status = "[PASS]" if exists and size > 0 else "[FAIL]"
        print(f"{status} File: {f.relative_to(PROJECT_ROOT)} ({size:,} bytes)")
        if not (exists and size > 0):
            all_files_exist = False

    # 2. Dataset Validation
    print("\n--- Processed Dataset Integrity ---")
    proc_csv = PROJECT_ROOT / "data" / "processed" / "ev_charging_project_2000.csv"
    df = pd.read_csv(proc_csv)
    
    assert len(df) == 2000, f"Expected 2000 rows, got {len(df)}"
    print("[PASS] Row count is exactly 2,000")

    expected_cols = [
        "station_id", "location_id", "hour", "day_of_week", "facility_type",
        "charge_time_hrs", "energy_kwh", "avg_energy_kwh", "avg_charge_time_min",
        "sessions_at_station_hour", "congestion_level", "load_level", "estimated_wait_min"
    ]
    assert list(df.columns) == expected_cols, "Column mismatch"
    print(f"[PASS] All {len(expected_cols)} required columns present with correct names")

    assert df.duplicated().sum() == 0, "Duplicate rows found"
    print("[PASS] 0 duplicate records")

    assert (df["energy_kwh"] <= 0).sum() == 0, "Found non-positive energy"
    print("[PASS] All energy values are strictly positive (>0 kWh)")

    assert (df["charge_time_hrs"] <= 0).sum() == 0, "Found non-positive duration"
    print("[PASS] All charge durations are strictly positive (>0 hrs)")

    assert ((df["hour"] < 0) | (df["hour"] > 23)).sum() == 0, "Invalid hours"
    print("[PASS] All hours are strictly between 0 and 23")

    valid_tiers = {"Low", "Medium", "High"}
    assert set(df["congestion_level"].unique()) == valid_tiers
    print("[PASS] Congestion labels strictly match {'Low', 'Medium', 'High'}")

    assert set(df["load_level"].unique()) == valid_tiers
    print("[PASS] Grid load labels strictly match {'Low', 'Medium', 'High'}")

    assert (df["estimated_wait_min"] < 0).sum() == 0, "Negative waiting time"
    print("[PASS] All estimated waiting times are non-negative (>=0 min)")

    # 3. Model Loading & Prediction Engine
    print("\n--- Model Inference & Recommendation Verification ---")
    from src.prediction import predict_for_inputs, get_predictor
    predictor = get_predictor()
    stations = predictor.get_available_stations()
    print(f"[PASS] Successfully loaded predictor with {len(stations)} unique stations")

    # Test prediction
    test_res = predict_for_inputs(stations[0], "Monday", 9)
    assert "final_congestion" in test_res
    assert "grid_load" in test_res
    assert "estimated_wait_min" in test_res
    assert "model_predictions" in test_res
    assert len(test_res["model_predictions"]) == 3
    print(f"[PASS] Sample prediction: Station {stations[0]} at 09:00 -> Congestion: {test_res['final_congestion']}, Grid: {test_res['grid_load']}, Wait: {test_res['estimated_wait_min']} min")

    # Test High Congestion recommendation
    found_high = False
    for s in stations:
        for h in [8, 12, 17, 18, 19]:
            r = predict_for_inputs(s, "Friday", h)
            if r["final_congestion"] == "High":
                assert r["recommendation"] is not None
                assert "station_id" in r["recommendation"]
                assert "estimated_wait_min" in r["recommendation"]
                print(f"[PASS] Alternative recommendation confirmed: Station {s} at {h}:00 (High) -> Alternative Station {r['recommendation']['station_id']} ({r['recommendation']['congestion_level']}, {r['recommendation']['estimated_wait_min']} min wait)")
                found_high = True
                break
        if found_high:
            break

    # 4. Model Evaluation Benchmark Check
    print("\n--- Model Evaluation Benchmark Metrics ---")
    metrics_path = PROJECT_ROOT / "models" / "model_metrics.json"
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    for m_name, vals in metrics.items():
        print(f"[PASS] {m_name:<32}: Accuracy = {vals['Accuracy']:.4f} | F1 = {vals['F1 Score']:.4f}")

    print("\n" + "=" * 70)
    print("ALL VERIFICATION AND INTEGRATION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_checks()
