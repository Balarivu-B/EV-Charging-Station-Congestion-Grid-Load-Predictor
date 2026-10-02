"""
Data Preprocessing and Feature Engineering Module
EV Charging Station Congestion & Grid Load Predictor

This module handles:
1. Loading raw EV charging session data.
2. Cleaning invalid, duplicate, and anomalous records.
3. Deriving academic proxy features:
   - sessions_at_station_hour
   - avg_energy_kwh
   - avg_charge_time_min
4. Deriving project targets:
   - congestion_level (Low, Medium, High)
   - load_level (Low, Medium, High)
   - estimated_wait_min (non-negative integer)
5. Exporting the final 2,000-row reproducible dataset.
6. Validation checks for dataset integrity.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Reconfigure stdout/stderr for Unicode paths on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Default project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "station_data_dataverse.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "ev_charging_project_2000.csv"

# Weekday normalization mapping
WEEKDAY_MAPPING = {
    "Mon": "Monday",
    "Tue": "Tuesday",
    "Wed": "Wednesday",
    "Thu": "Thursday",
    "Fri": "Friday",
    "Sat": "Saturday",
    "Sun": "Sunday",
    # If already full name, retain
    "Monday": "Monday",
    "Tuesday": "Tuesday",
    "Wednesday": "Wednesday",
    "Thursday": "Thursday",
    "Friday": "Friday",
    "Saturday": "Saturday",
    "Sunday": "Sunday",
}


def load_raw_data(filepath: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw EV charging session dataset."""
    if not filepath.exists():
        raise FileNotFoundError(f"Raw data file not found at: {filepath}")
    df = pd.read_csv(filepath)
    return df


def clean_raw_data(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Clean the raw EV charging dataset.
    
    Reports data cleaning counts before record removal:
    - Original row count
    - Duplicate count
    - Missing-value count in essential fields
    - Invalid-value count (negative or zero kWh / duration, invalid hours)
    - Final usable row count
    """
    original_rows = len(df)
    duplicate_count = df.duplicated().sum()
    
    essential_cols = ["stationId", "locationId", "startTime", "weekday", "facilityType", "chargeTimeHrs", "kwhTotal"]
    missing_count = df[essential_cols].isnull().any(axis=1).sum()
    
    # Invalid values: negative or zero energy / charge duration, or hour out of [0, 23]
    invalid_energy = (df["kwhTotal"] <= 0).sum()
    invalid_time = (df["chargeTimeHrs"] <= 0).sum()
    invalid_hours = ((df["startTime"] < 0) | (df["startTime"] > 23)).sum()
    invalid_total = ((df["kwhTotal"] <= 0) | (df["chargeTimeHrs"] <= 0) | (df["startTime"] < 0) | (df["startTime"] > 23)).sum()

    if verbose:
        print("=" * 60)
        print("DATA CLEANING REPORT")
        print("=" * 60)
        print(f"Original row count:        {original_rows}")
        print(f"Duplicate count:           {duplicate_count}")
        print(f"Missing-value count:       {missing_count}")
        print(f"Invalid energy (<=0 kWh):  {invalid_energy}")
        print(f"Invalid duration (<=0 hrs):{invalid_time}")
        print(f"Invalid hours (<0 or >23): {invalid_hours}")
        print(f"Total invalid rows:        {invalid_total}")

    # Remove duplicates
    cleaned = df.drop_duplicates().copy()

    # Drop missing in essential fields
    cleaned = cleaned.dropna(subset=essential_cols).copy()

    # Filter valid positive values and valid hours
    cleaned = cleaned[
        (cleaned["kwhTotal"] > 0)
        & (cleaned["chargeTimeHrs"] > 0)
        & (cleaned["startTime"] >= 0)
        & (cleaned["startTime"] <= 23)
    ].copy()

    usable_rows = len(cleaned)
    if verbose:
        print(f"Final usable row count:    {usable_rows}")
        print("=" * 60)

    return cleaned


def engineer_features_and_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive station-hour historical statistics and academic target proxies.
    
    Derived Features:
    - hour: integer hour extracted from startTime
    - day_of_week: normalized day string
    - sessions_at_station_hour: historical session count for (stationId, hour)
    - avg_energy_kwh: historical mean(kwhTotal) for (stationId, hour)
    - avg_charge_time_min: historical mean(chargeTimeHrs) * 60 for (stationId, hour)
    
    Derived Targets:
    - congestion_score = 0.65 * sessions_at_station_hour + 0.35 * avg_charge_time_min
      Categorized into Low, Medium, High via 33rd & 66th percentiles.
    - load_score = 0.60 * avg_energy_kwh + 0.40 * sessions_at_station_hour
      Categorized into Low, Medium, High via 33rd & 66th percentiles.
    - estimated_wait_min = round(factor * avg_charge_time_min) where factor depends on congestion.
    """
    data = df.copy()

    # Normalize hour and day of week
    data["hour"] = data["startTime"].astype(int)
    data["day_of_week"] = data["weekday"].map(WEEKDAY_MAPPING).fillna(data["weekday"])

    # Calculate historical station-hour statistics across all usable sessions
    station_hour_stats = (
        data.groupby(["stationId", "hour"])
        .agg(
            sessions_at_station_hour=("sessionId", "count"),
            avg_energy_kwh=("kwhTotal", "mean"),
            avg_charge_time_min=("chargeTimeHrs", lambda x: x.mean() * 60.0),
        )
        .reset_index()
    )

    merged = pd.merge(data, station_hour_stats, on=["stationId", "hour"], how="left")

    # 1. Congestion Score & Level
    congestion_score = (
        0.65 * merged["sessions_at_station_hour"]
        + 0.35 * merged["avg_charge_time_min"]
    )
    c_p33 = float(np.percentile(congestion_score, 33))
    c_p66 = float(np.percentile(congestion_score, 66))

    def categorize_congestion(score):
        if score <= c_p33:
            return "Low"
        elif score <= c_p66:
            return "Medium"
        else:
            return "High"

    merged["congestion_level"] = congestion_score.apply(categorize_congestion)

    # 2. Grid Load Score & Level
    load_score = (
        0.60 * merged["avg_energy_kwh"]
        + 0.40 * merged["sessions_at_station_hour"]
    )
    l_p33 = float(np.percentile(load_score, 33))
    l_p66 = float(np.percentile(load_score, 66))

    def categorize_load(score):
        if score <= l_p33:
            return "Low"
        elif score <= l_p66:
            return "Medium"
        else:
            return "High"

    merged["load_level"] = load_score.apply(categorize_load)

    # 3. Estimated Waiting Time (minutes)
    def calculate_estimated_wait(row):
        c_level = row["congestion_level"]
        charge_min = row["avg_charge_time_min"]
        if c_level == "Low":
            wait = charge_min * 0.20
        elif c_level == "Medium":
            wait = charge_min * 0.50
        else:
            wait = charge_min * 0.85
        return max(0, int(round(wait)))

    merged["estimated_wait_min"] = merged.apply(calculate_estimated_wait, axis=1)

    # Standardize column names to snake_case
    merged["station_id"] = merged["stationId"]
    merged["location_id"] = merged["locationId"]
    merged["facility_type"] = merged["facilityType"]
    merged["charge_time_hrs"] = merged["chargeTimeHrs"]
    merged["energy_kwh"] = merged["kwhTotal"]

    # Final project columns specified in requirements
    final_cols = [
        "station_id",
        "location_id",
        "hour",
        "day_of_week",
        "facility_type",
        "charge_time_hrs",
        "energy_kwh",
        "avg_energy_kwh",
        "avg_charge_time_min",
        "sessions_at_station_hour",
        "congestion_level",
        "load_level",
        "estimated_wait_min",
    ]

    return merged[final_cols].copy()


def create_project_dataset(
    raw_path: Path = RAW_DATA_PATH,
    output_path: Path = PROCESSED_DATA_PATH,
    target_rows: int = 2000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Execute full pipeline to generate exactly 2,000 reproducible rows.
    """
    raw_df = load_raw_data(raw_path)
    cleaned_df = clean_raw_data(raw_df, verbose=True)
    featured_df = engineer_features_and_targets(cleaned_df)

    if len(featured_df) >= target_rows:
        project_df = featured_df.sample(n=target_rows, random_state=random_state).copy()
    else:
        project_df = featured_df.copy()

    # Sort index for clean sequential ordering
    project_df = project_df.reset_index(drop=True)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    project_df.to_csv(output_path, index=False)
    print(f"\n[SUCCESS] Saved processed project dataset ({len(project_df)} rows) to: {output_path}")

    return project_df


def validate_dataset(filepath: Path = PROCESSED_DATA_PATH) -> bool:
    """
    Perform automated validation checks on the processed project dataset.
    """
    print("\n" + "=" * 60)
    print("DATA VALIDATION CHECK")
    print("=" * 60)

    if not filepath.exists():
        print(f"[FAIL] File does not exist at {filepath}")
        return False

    df = pd.read_csv(filepath)
    checks = []

    # 1. Exact row count check
    rows_ok = len(df) == 2000
    checks.append(("Row count == 2000", rows_ok, f"Actual: {len(df)}"))

    # 2. Required columns check
    expected_cols = [
        "station_id", "location_id", "hour", "day_of_week", "facility_type",
        "charge_time_hrs", "energy_kwh", "avg_energy_kwh", "avg_charge_time_min",
        "sessions_at_station_hour", "congestion_level", "load_level", "estimated_wait_min"
    ]
    cols_ok = list(df.columns) == expected_cols
    checks.append(("Required columns present & ordered", cols_ok, f"{len(df.columns)} columns"))

    # 3. Duplicates check
    dups = df.duplicated().sum()
    dups_ok = dups == 0
    checks.append(("No duplicate rows", dups_ok, f"Found: {dups}"))

    # 4. Negative values check
    neg_energy = (df["energy_kwh"] <= 0).sum()
    energy_ok = neg_energy == 0
    checks.append(("Energy kWh > 0", energy_ok, f"Violations: {neg_energy}"))

    neg_time = (df["charge_time_hrs"] <= 0).sum()
    time_ok = neg_time == 0
    checks.append(("Charge time hrs > 0", time_ok, f"Violations: {neg_time}"))

    # 5. Hour range [0, 23]
    hour_invalid = ((df["hour"] < 0) | (df["hour"] > 23)).sum()
    hour_ok = hour_invalid == 0
    checks.append(("Hour within [0, 23]", hour_ok, f"Violations: {hour_invalid}"))

    # 6. Valid labels
    valid_labels = {"Low", "Medium", "High"}
    c_labels_ok = set(df["congestion_level"].unique()).issubset(valid_labels)
    checks.append(("Congestion labels valid", c_labels_ok, f"{df['congestion_level'].unique()}"))

    l_labels_ok = set(df["load_level"].unique()).issubset(valid_labels)
    checks.append(("Load labels valid", l_labels_ok, f"{df['load_level'].unique()}"))

    # 7. Estimated wait min >= 0
    wait_neg = (df["estimated_wait_min"] < 0).sum()
    wait_ok = wait_neg == 0
    checks.append(("Estimated wait min >= 0", wait_ok, f"Violations: {wait_neg}"))

    all_passed = True
    for name, status, detail in checks:
        mark = "[PASS]" if status else "[FAIL]"
        print(f"{mark:<7} {name:<35} ({detail})")
        if not status:
            all_passed = False

    print("=" * 60)
    print(f"Overall Validation Result: {'PASSED' if all_passed else 'FAILED'}")
    print("=" * 60 + "\n")
    return all_passed


if __name__ == "__main__":
    create_project_dataset()
    validate_dataset()
