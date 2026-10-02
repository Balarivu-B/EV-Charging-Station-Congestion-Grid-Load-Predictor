# EV Charging Station Congestion & Grid Load Predictor

An academic machine-learning and data science project designed to predict EV charging station congestion levels, estimate electrical grid-load proxy levels, calculate estimated waiting times, and recommend alternative less-congested charging stations based on historical charging session patterns.

---

## 1. Problem Statement

Rapid electric vehicle (EV) adoption poses challenges for both EV drivers and power distribution networks:
- **EV Drivers** face unpredictable queue lengths, prolonged waiting times, and station congestion, especially during peak commute hours.
- **Power Grid Operators** face localized demand spikes when multiple high-power charging sessions coincide at specific facilities.

Real-time physical sensors and IoT queue monitoring are often unavailable or expensive to deploy across public charging infrastructure. This project addresses the challenge by leveraging **historical charging session data** to forecast congestion and grid load patterns, offering intelligent alternatives to optimize station utilization.

---

## 2. Proposed Solution

This system uses machine learning trained on historical EV session patterns to provide:
1. **Congestion Level Prediction (`Low`, `Medium`, `High`)**: Evaluated via three distinct ML classifiers (Logistic Regression, Decision Tree, and Random Forest) with majority-voting consensus.
2. **Grid-Load Level Proxy Prediction (`Low`, `Medium`, `High`)**: Predicting energy demand stress based on session activity and energy volume.
3. **Estimated Waiting Time (minutes)**: Derived heuristically from historical station charging duration and predicted congestion.
4. **Smart Alternative Station Recommendation**: When high congestion is predicted, the system automatically identifies less-congested alternative stations operating at the same time period.

---

## 3. Dataset & Academic Derived Targets

### Original Public Dataset
The project utilizes real-world EV charging session data from the Harvard Dataverse / public repository (`station_data_dataverse.csv`), containing recorded charging events with fields such as:
- `sessionId`, `kwhTotal`, `dollars`, `created`, `ended`, `startTime`, `endTime`, `chargeTimeHrs`, `weekday`, `platform`, `distance`, `userId`, `stationId`, `locationId`, `facilityType`.

### Critical Clarification on Derived Targets
> **Important Note:** The original dataset contains historical session records and does **not** directly measure live physical queue lengths, traffic, or electrical substation telemetry. Therefore, **congestion level, grid-load level, and estimated waiting time are derived academic proxies** calculated from historical charging session frequency, duration, and energy consumption patterns.

- **Congestion Level**: Derived from a composite historical score:
  $$\text{congestion\_score} = 0.65 \times \text{sessions\_at\_station\_hour} + 0.35 \times \text{avg\_charge\_time\_min}$$
  Discretized into *Low*, *Medium*, and *High* using the 33rd and 66th percentiles.
- **Grid Load Level**: Derived from a composite energy-activity score:
  $$\text{load\_score} = 0.60 \times \text{avg\_energy\_kwh} + 0.40 \times \text{sessions\_at\_station\_hour}$$
  Discretized into *Low*, *Medium*, and *High* using the 33rd and 66th percentiles.
- **Estimated Waiting Time**: Calculated as an empirical fraction of average charging duration:
  - *Low Congestion*: $0.20 \times \text{avg\_charge\_time\_min}$
  - *Medium Congestion*: $0.50 \times \text{avg\_charge\_time\_min}$
  - *High Congestion*: $0.85 \times \text{avg\_charge\_time\_min}$

---

## 4. Technologies Used

- **Language**: Python 3.12+
- **Data Manipulation**: Pandas, NumPy
- **Data Visualization**: Matplotlib, Seaborn
- **Machine Learning**: Scikit-Learn (`LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`, `ColumnTransformer`, `Pipeline`)
- **Model Serialization**: Joblib
- **Web Dashboard**: Streamlit
- **Exploration**: Jupyter Notebooks

---

## 5. Machine Learning Models

1. **Logistic Regression (Multinomial)**: Linear baseline classifier with L2 regularization. Provides calibrated probability baselines and rapid inference.
2. **Decision Tree Classifier**: Non-linear, rule-based tree model capturing hierarchical feature interactions (e.g., station location vs. hour).
3. **Random Forest Classifier**: Ensemble of bagged decision trees reducing variance and overfitting, providing robust generalization and tie-breaking capability.

---

## 6. End-to-End Workflow

```text
station_data_dataverse.csv (Raw Data)
                ↓
    Data Cleaning & Validation
  (Missing values, negatives, ranges)
                ↓
 Feature Engineering & Target Derivation
    (Exact 2,000 reproducible rows)
                ↓
    Exploratory Data Analysis (EDA)
                ↓
 Train/Test Split (Stratified 80/20)
                ↓
 Model Training (LR, DT, RF)
                ↓
 Evaluation (Accuracy, Precision, Recall, F1)
                ↓
 Model Persistence (.pkl via Joblib)
                ↓
 Inference & Recommender Engine
                ↓
 Streamlit Interactive Web Application
```

---

## 7. Project Architecture

```text
EV-Charging-Predictor/
│
├── data/
│   ├── raw/
│   │   └── station_data_dataverse.csv      # Original Harvard Dataverse dataset
│   └── processed/
│       └── ev_charging_project_2000.csv    # Final 2,000-row cleaned project dataset
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb              # Cleaning, feature derivation, export
│   ├── 02_eda.ipynb                        # Complete exploratory data analysis
│   └── 03_model_training.ipynb             # ML training, evaluation, comparison
│
├── models/
│   ├── congestion_logistic.pkl             # Trained Logistic Regression model
│   ├── congestion_tree.pkl                 # Trained Decision Tree model
│   ├── congestion_forest.pkl               # Trained Random Forest model
│   ├── grid_load_model.pkl                 # Trained Grid Load Random Forest model
│   └── preprocessor.pkl                    # Fitted ColumnTransformer pipeline
│
├── src/
│   ├── preprocessing.py                    # Data cleaning, feature pipeline, targets
│   ├── train_models.py                     # CLI script to train and evaluate all models
│   └── prediction.py                       # Modular inference & alternative recommender
│
├── app/
│   └── app.py                              # Streamlit dashboard application
│
├── requirements.txt                        # Project dependencies
├── README.md                               # Project documentation & viva guide
└── .gitignore                              # Git exclusion rules
```

---

## 8. Setup and Execution Guide

### Step 1: Clone or Navigate to Directory
```powershell
cd c:\Users\balar\OneDrive\ドキュメント\EV-Charging-Predictor
```

### Step 2: Create and Activate Virtual Environment (Optional / Recommended)
```powershell
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Run Data Processing & Model Training
```powershell
python src/train_models.py
```

### Step 5: Launch Streamlit Dashboard
**Option A: Double-click script**
- Double-click [`run_app.bat`](file:///c:/Users/balar/OneDrive/ドキュメント/EV-Charging-Predictor/run_app.bat) in the project folder.

**Option B: Terminal (PowerShell)**
```powershell
.\venv\Scripts\python.exe -m streamlit run app/app.py
```
*(Using `python -m streamlit` avoids Windows Application Control / policy execution errors on `streamlit.exe`).*


---

## 9. Viva / Academic Presentation Guide

### 1. What problem does the project solve?
It predicts EV charging station congestion, grid stress proxies, and estimated waiting times using historical session data, helping drivers avoid congested stations and helping operators anticipate demand.

### 2. Why EV charging congestion?
As EV adoption surges, public charging infrastructure experiences peak-hour clustering. Congestion prediction mitigates range anxiety and minimizes wait times.

### 3. What dataset was used?
A public EV charging session dataset (`station_data_dataverse.csv`) containing historical charging sessions with energy, duration, timestamps, and station facility attributes.

### 4. What are the original dataset columns?
Key columns include `sessionId`, `stationId`, `locationId`, `kwhTotal`, `chargeTimeHrs`, `startTime`, `endTime`, `weekday`, `facilityType`, and platform identifiers.

### 5. Which features were created?
Aggregated station-hour historical metrics: `avg_energy_kwh`, `avg_charge_time_min`, and `sessions_at_station_hour`, combined with temporal features (`hour`, `day_of_week`), station identifiers, and `facility_type`.

### 6. How was congestion calculated?
Through a weighted composite score combining session volume (65%) and average charge duration (35%), discretized into Low, Medium, and High using 33rd/66th percentiles.

### 7. How was grid load calculated?
Through a weighted proxy combining average energy consumed (60%) and concurrent session volume (40%), categorized into Low, Medium, and High using tertile thresholds.

### 8. Why Logistic Regression?
It provides an interpretable, rapid linear baseline model for multi-class classification.

### 9. Why Decision Tree?
It captures non-linear relationships, threshold splits, and categorical interaction rules without requiring feature scaling.

### 10. Why Random Forest?
As an ensemble of decorrelated decision trees (bagging), it mitigates single-tree variance and overfitting, yielding superior generalization.

### 11. Why compare three models?
In academic ML, comparing multiple paradigms (linear vs. tree vs. ensemble) validates whether algorithmic complexity delivers measurable performance gains across precision, recall, and F1-score.

### 12. What metrics were used?
Accuracy, Macro/Weighted Precision, Macro/Weighted Recall, Macro/Weighted F1 Score, and Confusion Matrices to evaluate per-class misclassifications.

### 13. What is data leakage?
Data leakage occurs when information from outside the training dataset or from the target label itself inadvertently enters model training, artificially inflating performance.

### 14. How did you prevent data leakage?
Target variables (`congestion_level`, `load_level`, `estimated_wait_min`) and arbitrary identifiers (`sessionId`, `userId`) were strictly excluded from input features. Preprocessing transformers were fit strictly on the training set.

### 15. How does the alternative-station recommendation work?
When a station is predicted to have `High` congestion, rule-based logic queries historical station profiles at the same time period and identifies stations with `Low` predicted congestion and minimal estimated waiting time.

### 16. Is waiting time real or estimated?
It is strictly an **estimated** proxy calculated from average session duration and congestion severity, not a physical queue sensor measurement.

### 17. Is grid load directly measured?
No, it is a derived academic indicator based on historical kWh draw and station session activity, not direct telemetry from grid substations.

### 18. Why Streamlit?
Streamlit enables rapid, pure-Python dashboard development with interactive widgets, state management, and clear visualization without needing frontend frameworks or external APIs.

### 19. What are the limitations?
The system relies on historical patterns rather than real-time hardware telemetry, weather factors, or sudden traffic disruptions.

### 20. What could be improved in future?
Incorporating real-time IoT plug status, dynamic time-of-use electricity pricing, driver cancellation patterns, and weather-dependent energy consumption models.
