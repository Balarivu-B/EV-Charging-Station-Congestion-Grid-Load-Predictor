"""
Inference and Alternative Station Recommendation Module
EV Charging Station Congestion & Grid Load Predictor

This module provides reusable functions to:
1. Load serialized models from disk.
2. Construct input features from user parameters (station_id, day_of_week, hour).
3. Query all three congestion classifiers (Logistic Regression, Decision Tree, Random Forest).
4. Perform majority voting with Random Forest tie-breaker.
5. Extract prediction probabilities/confidence for each model.
6. Generate 24-hour comparative trends across the 3 models.
7. Predict grid load level proxy.
8. Compute estimated waiting time based on historical charge duration and predicted congestion.
9. Recommend a nearby/co-located or less-congested alternative station when high congestion is detected.
"""

import sys
import json
from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np
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


class EVChargingPredictor:
    """End-to-end inference and recommendation engine."""

    def __init__(self, models_dir: Path = MODELS_DIR, data_path: Path = PROCESSED_DATA_PATH):
        self.models_dir = Path(models_dir)
        self.data_path = Path(data_path)
        self.models = {}
        self.metadata = {}
        self.metrics = {}
        self.dataset = None
        self.station_profiles = None
        self.station_hour_profiles = None
        self._load_resources()

    def _load_resources(self):
        """Load datasets, metadata, and saved joblib models with error handling."""
        if self.data_path.exists():
            self.dataset = pd.read_csv(self.data_path)
            self._build_historical_profiles()
        else:
            raise FileNotFoundError(f"Processed dataset not found at {self.data_path}")

        model_files = {
            "logistic": self.models_dir / "congestion_logistic.pkl",
            "tree": self.models_dir / "congestion_tree.pkl",
            "forest": self.models_dir / "congestion_forest.pkl",
            "grid_load": self.models_dir / "grid_load_model.pkl",
        }

        for key, path in model_files.items():
            if path.exists():
                self.models[key] = joblib.load(path)
            else:
                raise FileNotFoundError(f"Required model artifact not found: {path}")

        meta_path = self.models_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

        metrics_path = self.models_dir / "model_metrics.json"
        if metrics_path.exists():
            with open(metrics_path, "r", encoding="utf-8") as f:
                self.metrics = json.load(f)

    def _build_historical_profiles(self):
        """Precompute station-level and station-hour profiles for fast lookup."""
        df = self.dataset

        self.station_hour_profiles = (
            df.groupby(["station_id", "hour"])
            .agg(
                location_id=("location_id", "first"),
                facility_type=("facility_type", "first"),
                sessions_at_station_hour=("sessions_at_station_hour", "first"),
                avg_energy_kwh=("avg_energy_kwh", "first"),
                avg_charge_time_min=("avg_charge_time_min", "first"),
            )
            .reset_index()
        )

        self.station_profiles = (
            df.groupby("station_id")
            .agg(
                location_id=("location_id", "first"),
                facility_type=("facility_type", "first"),
                sessions_at_station_hour=("sessions_at_station_hour", "median"),
                avg_energy_kwh=("avg_energy_kwh", "mean"),
                avg_charge_time_min=("avg_charge_time_min", "mean"),
            )
            .reset_index()
        )

    def get_available_stations(self):
        """Return sorted list of unique station IDs."""
        return sorted([int(x) for x in self.dataset["station_id"].unique()])

    def get_available_hours(self):
        """Return available hour range 0 to 23."""
        return list(range(24))

    def get_available_days(self):
        """Return list of standard weekdays."""
        return ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    def prepare_input(self, station_id: int, day_of_week: str, hour: int):
        """Assemble feature row for the ML pipelines avoiding data leakage."""
        exact_match = True
        hour = int(hour)
        station_id = int(station_id)

        sh_row = self.station_hour_profiles[
            (self.station_hour_profiles["station_id"] == station_id)
            & (self.station_hour_profiles["hour"] == hour)
        ]

        if not sh_row.empty:
            rec = sh_row.iloc[0]
            loc_id = int(rec["location_id"])
            fac_type = int(rec["facility_type"])
            sessions = int(rec["sessions_at_station_hour"])
            avg_kwh = float(rec["avg_energy_kwh"])
            avg_min = float(rec["avg_charge_time_min"])
        else:
            s_row = self.station_profiles[self.station_profiles["station_id"] == station_id]
            if not s_row.empty:
                exact_match = False
                rec = s_row.iloc[0]
                loc_id = int(rec["location_id"])
                fac_type = int(rec["facility_type"])
                sessions = max(1, int(round(rec["sessions_at_station_hour"])))
                avg_kwh = float(rec["avg_energy_kwh"])
                avg_min = float(rec["avg_charge_time_min"])
            else:
                exact_match = False
                loc_id = int(self.dataset["location_id"].mode()[0])
                fac_type = int(self.dataset["facility_type"].mode()[0])
                sessions = int(self.dataset["sessions_at_station_hour"].median())
                avg_kwh = float(self.dataset["avg_energy_kwh"].mean())
                avg_min = float(self.dataset["avg_charge_time_min"].mean())

        feature_df = pd.DataFrame(
            [
                {
                    "station_id": station_id,
                    "location_id": loc_id,
                    "hour": hour,
                    "day_of_week": day_of_week,
                    "facility_type": fac_type,
                    "avg_energy_kwh": avg_kwh,
                    "avg_charge_time_min": avg_min,
                    "sessions_at_station_hour": sessions,
                }
            ]
        )

        return feature_df, exact_match

    def predict_congestion(self, feature_df: pd.DataFrame):
        """
        Generate predictions and class probabilities from Logistic Regression,
        Decision Tree, and Random Forest. Computes majority vote with RF tie-breaker.
        """
        pred_lr = str(self.models["logistic"].predict(feature_df)[0])
        pred_dt = str(self.models["tree"].predict(feature_df)[0])
        pred_rf = str(self.models["forest"].predict(feature_df)[0])

        individual_preds = {
            "Logistic Regression": pred_lr,
            "Decision Tree": pred_dt,
            "Random Forest": pred_rf,
        }

        # Probabilities
        probs_dict = {}
        for name, key in [
            ("Logistic Regression", "logistic"),
            ("Decision Tree", "tree"),
            ("Random Forest", "forest"),
        ]:
            model = self.models[key]
            prob_vals = model.predict_proba(feature_df)[0]
            classes = list(model.classes_)
            # Map probabilities to standardized ordering: Low, Medium, High
            p_map = {cls: float(prob_vals[idx]) for idx, cls in enumerate(classes)}
            probs_dict[name] = {
                "Low": round(p_map.get("Low", 0.0), 4),
                "Medium": round(p_map.get("Medium", 0.0), 4),
                "High": round(p_map.get("High", 0.0), 4),
            }

        # Majority voting
        votes = [pred_lr, pred_dt, pred_rf]
        vote_counts = Counter(votes)
        most_common = vote_counts.most_common()

        if most_common[0][1] >= 2:
            final_pred = most_common[0][0]
        else:
            final_pred = pred_rf

        return individual_preds, final_pred, probs_dict

    def predict_grid_load(self, feature_df: pd.DataFrame):
        """Predict grid load proxy level using trained Random Forest."""
        return str(self.models["grid_load"].predict(feature_df)[0])

    def estimate_wait_time(self, avg_charge_time_min: float, congestion_level: str) -> int:
        """Calculate estimated waiting time (minutes)."""
        if congestion_level == "Low":
            factor = 0.20
        elif congestion_level == "Medium":
            factor = 0.50
        else:
            factor = 0.85

        wait_min = int(round(avg_charge_time_min * factor))
        return max(0, wait_min)

    def find_alternative_station(
        self, current_station_id: int, current_location_id: int, current_wait_min: int, day_of_week: str, hour: int
    ):
        """
        Identify a nearby/co-located or less-congested alternative station.
        
        Prioritizes:
        1. Peer stations located at the EXACT SAME LOCATION ID (co-located physical facility).
        2. Stations predicted as 'Low' congestion.
        3. Stations with the lowest estimated waiting time.
        """
        current_station_id = int(current_station_id)
        current_location_id = int(current_location_id)
        candidate_stations = [s for s in self.get_available_stations() if s != current_station_id]

        candidates = []
        for cand_id in candidate_stations:
            feat_df, _ = self.prepare_input(cand_id, day_of_week, hour)
            loc_id = int(feat_df["location_id"].iloc[0])
            fac_type = int(feat_df["facility_type"].iloc[0])
            _, final_c, _ = self.predict_congestion(feat_df)
            avg_min = float(feat_df["avg_charge_time_min"].iloc[0])
            wait_m = self.estimate_wait_time(avg_min, final_c)
            sessions = int(feat_df["sessions_at_station_hour"].iloc[0])

            candidates.append(
                {
                    "station_id": cand_id,
                    "location_id": loc_id,
                    "facility_type": fac_type,
                    "is_same_location": (loc_id == current_location_id),
                    "congestion_level": final_c,
                    "estimated_wait_min": wait_m,
                    "wait_time_saved_min": max(0, current_wait_min - wait_m),
                    "sessions_at_station_hour": sessions,
                    "avg_charge_time_min": avg_min,
                    "avg_energy_kwh": float(feat_df["avg_energy_kwh"].iloc[0]),
                }
            )

        cand_df = pd.DataFrame(candidates)
        if cand_df.empty:
            return None

        # Priority 1: Co-located stations (same facility) with Low congestion
        co_low = cand_df[(cand_df["is_same_location"]) & (cand_df["congestion_level"] == "Low")].sort_values(
            by=["estimated_wait_min", "sessions_at_station_hour"]
        )
        if not co_low.empty:
            best = co_low.iloc[0].to_dict()
            best["proximity"] = "Co-located (Same Facility / Location ID)"
            best["reason"] = (
                f"Station {best['station_id']} is located in the exact same facility (Location ID {current_location_id}) "
                f"and is predicted to have Low congestion, saving an estimated {best['wait_time_saved_min']} minutes of waiting time."
            )
            return best

        # Priority 2: Co-located stations with Medium congestion
        co_med = cand_df[(cand_df["is_same_location"]) & (cand_df["congestion_level"] == "Medium")].sort_values(
            by=["estimated_wait_min"]
        )
        if not co_med.empty:
            best = co_med.iloc[0].to_dict()
            best["proximity"] = "Co-located (Same Facility / Location ID)"
            best["reason"] = (
                f"Station {best['station_id']} is at the same facility with moderate congestion, "
                f"saving an estimated {best['wait_time_saved_min']} minutes compared to current station."
            )
            return best

        # Priority 3: Network-wide Low congestion station with lowest wait
        net_low = cand_df[cand_df["congestion_level"] == "Low"].sort_values(
            by=["estimated_wait_min", "sessions_at_station_hour"]
        )
        if not net_low.empty:
            best = net_low.iloc[0].to_dict()
            best["proximity"] = f"Nearby Network Station (Location ID {best['location_id']})"
            best["reason"] = (
                f"Station {best['station_id']} is predicted to have Low congestion and the shortest queue "
                f"({best['estimated_wait_min']} min wait), saving an estimated {best['wait_time_saved_min']} minutes."
            )
            return best

        # Priority 4: Lowest wait time overall
        best = cand_df.sort_values(by="estimated_wait_min").iloc[0].to_dict()
        best["proximity"] = f"Alternative Network Station (Location ID {best['location_id']})"
        best["reason"] = f"Station with minimal queue time ({best['estimated_wait_min']} min wait)."
        return best

    def get_24h_model_comparison(self, station_id: int, day_of_week: str):
        """
        Generate 24-hour simulation across all 3 models for this station.
        Useful for plotting model changes and diurnal patterns.
        """
        records = []
        tier_to_val = {"Low": 1, "Medium": 2, "High": 3}

        for h in range(24):
            feat_df, _ = self.prepare_input(station_id, day_of_week, h)
            ind_preds, final_c, probs = self.predict_congestion(feat_df)
            grid_l = self.predict_grid_load(feat_df)
            avg_m = float(feat_df["avg_charge_time_min"].iloc[0])
            wait_m = self.estimate_wait_time(avg_m, final_c)

            records.append(
                {
                    "hour": h,
                    "hour_label": f"{h:02d}:00",
                    "Logistic Regression": ind_preds["Logistic Regression"],
                    "Decision Tree": ind_preds["Decision Tree"],
                    "Random Forest": ind_preds["Random Forest"],
                    "Final Consensus": final_c,
                    "LR_Numeric": tier_to_val[ind_preds["Logistic Regression"]],
                    "DT_Numeric": tier_to_val[ind_preds["Decision Tree"]],
                    "RF_Numeric": tier_to_val[ind_preds["Random Forest"]],
                    "Consensus_Numeric": tier_to_val[final_c],
                    "Grid Load": grid_l,
                    "Estimated Wait (min)": wait_m,
                    "LR_Prob_High": probs["Logistic Regression"]["High"],
                    "DT_Prob_High": probs["Decision Tree"]["High"],
                    "RF_Prob_High": probs["Random Forest"]["High"],
                }
            )

        return pd.DataFrame(records)

    def run_full_prediction(self, station_id: int, day_of_week: str, hour: int):
        """Comprehensive prediction pipeline returning all metrics and recommendations."""
        feat_df, exact_match = self.prepare_input(station_id, day_of_week, hour)
        loc_id = int(feat_df["location_id"].iloc[0])
        fac_type = int(feat_df["facility_type"].iloc[0])
        individual_preds, final_congestion, model_probabilities = self.predict_congestion(feat_df)
        grid_load = self.predict_grid_load(feat_df)

        avg_charge_time = float(feat_df["avg_charge_time_min"].iloc[0])
        avg_energy = float(feat_df["avg_energy_kwh"].iloc[0])
        sessions = int(feat_df["sessions_at_station_hour"].iloc[0])
        wait_min = self.estimate_wait_time(avg_charge_time, final_congestion)

        recommendation = None
        if final_congestion == "High":
            recommendation = self.find_alternative_station(
                station_id, loc_id, wait_min, day_of_week, hour
            )

        return {
            "station_id": station_id,
            "location_id": loc_id,
            "facility_type": fac_type,
            "day_of_week": day_of_week,
            "hour": hour,
            "exact_historical_match": exact_match,
            "historical_sessions": sessions,
            "avg_energy_kwh": round(avg_energy, 2),
            "avg_charge_time_min": round(avg_charge_time, 1),
            "model_predictions": individual_preds,
            "model_probabilities": model_probabilities,
            "final_congestion": final_congestion,
            "grid_load": grid_load,
            "estimated_wait_min": wait_min,
            "recommendation": recommendation,
        }


# Module level convenience functions
_PREDICTOR_INSTANCE = None


def get_predictor():
    """Singleton getter for EVChargingPredictor."""
    global _PREDICTOR_INSTANCE
    if _PREDICTOR_INSTANCE is None:
        _PREDICTOR_INSTANCE = EVChargingPredictor()
    return _PREDICTOR_INSTANCE


def predict_for_inputs(station_id: int, day_of_week: str, hour: int):
    """Convenience function for Streamlit app and scripts."""
    predictor = get_predictor()
    return predictor.run_full_prediction(station_id, day_of_week, hour)
