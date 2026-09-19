# Zephyra: Intelligent Energy & Equipment Monitoring

### AI-Powered Contextual Anomaly Detection & Diagnostic Intelligence for Chiller Systems
**Yukthi 2026 National-Level Hackathon Project**

---

## 1. Problem Statement
Chillers and central cooling plants consume up to 40%–50% of electrical energy in modern commercial and industrial buildings. Traditional monitoring solutions rely heavily on simplistic static thresholds (e.g. *“Energy > 400 kWh = Anomaly”*). 

Such thresholding fails in industrial reality:
- **False Negatives:** Moderate energy draw during a cool evening with low building load may indicate severe chiller degradation or valve bypass, yet passes fixed thresholds unnoticed.
- **False Positives:** High energy draw during scorching afternoon peak building loads is thermodynamically expected and normal, but triggers false alarms.

**Zephyra** addresses this challenge by replacing fixed engineering thresholds with **Contextual + Multivariate Machine Learning**, learning the baseline thermodynamic behavior of multi-unit chillers under varying operating loads, hydraulic flows, and ambient weather conditions.

---

## 2. Core Architecture Pipeline

```
CSV Telemetry (25k+ records, multi-chiller)
                     ↓
        Data Validation & Quality Profiler
                     ↓
        Chronological Sorting & Preprocessing
          (Gap isolation, non-fault imputation)
                     ↓
         Domain Feature Engineering
   (Cyclical time, lags, rolling stats, lift, kW/RT)
                     ↓
    Contextual Expected-Behaviour Regressor
   (HistGradientBoosting per equipment → Expected kWh)
                     ↓
    Unsupervised Multivariate ML Anomaly Detector
   (Isolation Forest on residuals + multivariate manifold)
                     ↓
         Data-Driven Severity Engine
       (NORMAL, LOW, MEDIUM, HIGH, CRITICAL)
                     ↓
    Persistence & Episode Clustering Engine
   (Clusters consecutive anomalies into multi-hour episodes)
                     ↓
      Feature-Level Evidence & Explainability
     (Z-score deviation, physical hypothesis separation)
                     ↓
     Evidence-Grounded Operational Recommendations
   (P1 Urgent, P2 High, P3 Medium, P4 Routine Checklists)
                     ↓
    Industrial Streamlit Web Monitoring Platform
```

---

## 3. Key Innovations & Machine Learning Methodology

### A. Contextual Expected-Behaviour Model (`HistGradientBoostingRegressor`)
- Rather than evaluating raw kilowatt-hours in isolation, Zephyra predicts the **expected energy consumption** $\hat{y}$ given instantaneous operating conditions:
  - **Thermodynamic Drivers:** Building Load (RT), Chilled Water Flow Rate (L/sec), Cooling Water Supply Temperature (°C).
  - **Meteorological Drivers:** Outside Dry-Bulb Temperature (°F), Dew Point (°F), Relative Humidity (%), Wind Speed, Barometric Pressure.
  - **Temporal & Lag Dynamics:** Cyclical sine/cosine of hour and month, operational lags ($t-1, t-2, t-4$), and rolling historical averages.
- **Residual Calculation:**
  $$\text{Energy Residual} = y_{\text{actual}} - \hat{y}_{\text{expected}}$$
  $$\text{Relative Deviation (\%)} = \frac{y_{\text{actual}} - \hat{y}_{\text{expected}}}{\hat{y}_{\text{expected}}} \times 100$$

### B. Unsupervised Multivariate Anomaly Detection (`IsolationForest`)
- Because industrial operational data lacks ground-truth fault labels, an unsupervised manifold learning approach is necessary.
- An ensemble of Isolation Trees isolates points exhibiting abnormal multidimensional trajectories across energy residuals, specific power ratio ($kWh / RT$), chilled water flow per ton ($L/sec / RT$), compressor lift ($\Delta T$), and rolling volatility.
- Produces a continuous, calibrated `anomaly_score` $\in [0.0, 1.0]$ and binary `anomaly_flag`.

### C. Multi-Criteria Severity Engine
Severity is determined by combining:
1. **Multivariate Anomaly Score Weight (40%)**: Degree of outlierness in high-dimensional space.
2. **Relative Energy Deviation (35%)**: Thermodynamic magnitude of excess energy ($|Actual - Expected|$).
3. **Persistence & Trend Factor (25%)**: Escalation for repeating, sustained, or worsening episodes.

| Severity Tier | Composite Score | Action Threshold |
|---|---|---|
| **CRITICAL** | $S \ge 75$ | Immediate field technician dispatch |
| **HIGH** | $55 \le S < 75$ | Urgent investigation within 24 hours |
| **MEDIUM** | $35 \le S < 55$ | Shift inspection & logging |
| **LOW** | $S < 35$ | Routine supervisory monitoring |
| **NORMAL** | $S = 0$ | Operating within standard envelope |

### D. Persistence Detection (Abnormal Episode Clustering)
Single isolated spikes can occur due to electrical switching transients or intermittent telemetry packet drops. Zephyra clusters anomalous points within rolling temporal windows into distinct **Abnormal Episodes**:
- **Isolated:** Single isolated interval.
- **Repeated:** 2 sporadic abnormal observations.
- **Persistent:** $\ge 3$ consecutive cycles ($\ge 1.5$ hours) of sustained off-design behavior.
- **Increasing Trend:** Positive residual slope indicating accelerating performance degradation.

### E. Transparent Equipment Health Score ($0 - 100$)
Health scores are computed dynamically with itemized deductions:
$$\text{Health Score} = 100 - (\text{Penalty}_{\text{freq}} + \text{Penalty}_{\text{sev}} + \text{Penalty}_{\text{persist}} + \text{Penalty}_{\text{dev}})$$
Every point deducted is explicitly explained to facility managers.

### F. Evidence Generation & Engineering Recommendations
Every detected anomaly produces an **Evidence Package** with:
- Observed vs. Expected physical metrics.
- Top contributing variables evaluated via standardized Z-score deviations from the normal equipment baseline.
- **Empirical Observation vs. Mechanical Root-Cause Separation:** The platform clearly distinguishes factual operational measurements from possible mechanical hypotheses (e.g. condenser fouling vs sensor drift) to prevent false assumptions.

---

## 4. Zero Hardcoding Guarantee (Reusability Contract)
Zephyra complies strictly with the universal data contract:
- **No hard-coded equipment IDs:** Automatically discovers any set of chillers (`CHILLER-01`, `CHILLER-02`, etc.).
- **No hard-coded timestamps or row counts:** Functions identically on 1,000 rows or 100,000 rows.
- **No hard-coded date ranges or anomaly rows.**
- **Telemetry Discontinuity Awareness:** Tracks intervals $> 60$ minutes as acquisition gaps and never treats missing data or time gaps as equipment faults.

---

## 5. Quick Start & Installation

### Prerequisites
- Python 3.10+ (Tested on Python 3.13)

### Installation
```bash
# Clone or navigate to the repository
cd zephyra

# Install dependencies
python3 -m pip install -r requirements.txt
```

### Running the Application
```bash
# Launch the Streamlit Industrial Dashboard
python3 -m streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### Running Automated Test Suite
```bash
python3 tests/test_pipeline.py
```

---

## 6. Project Directory Structure

```
zephyra/
│
├── app.py                     # Streamlit Industrial Monitoring Platform
├── requirements.txt           # Dependency specifications
├── README.md                  # Comprehensive documentation & presentation guide
├── run.bat                    # Windows quick-launch script
├── run.ps1                    # PowerShell quick-launch script
│
├── data/
│   └── sample/
│       └── development_dataset.csv # 25,003-row historical chiller dataset
│
├── src/
│   ├── __init__.py
│   ├── config.py              # Schema definitions, constants, model parameters
│   ├── data_loader.py         # CSV ingestion, validation, synthetic demo generator
│   ├── preprocessing.py       # Cleaning, gap detection, non-fault imputation
│   ├── feature_engineering.py # Lags, rolling stats, cyclical time, lift, kW/RT
│   ├── baseline_model.py      # Contextual expected energy regressor (HistGradientBoosting)
│   ├── anomaly_detection.py   # Unsupervised Isolation Forest contextual detector
│   ├── severity.py            # Multi-criteria severity engine & episode clustering
│   ├── explainability.py      # Feature-level evidence generation & deviation analysis
│   ├── recommendations.py     # Evidence-grounded operational engineering recommendations
│   ├── health_score.py        # Dynamic 0-100 equipment health index with itemized audit
│   └── visualization.py       # Industrial dark Plotly charts & UI components
│
└── tests/
    └── test_pipeline.py       # End-to-end integration and unit verification suite
```

---

## 7. Interactive Dashboard Walkthrough

1. **Executive Overview**: High-level fleet KPIs, active unit count, total abnormal episodes, estimated financial waste from excess energy ($), and health status matrix.
2. **Equipment Health**: Semicircular gauge meters showing 0-100 health index for each chiller, accompanied by itemized penalty deduction audit panels.
3. **Anomaly Timeline**: Zoomable Plotly time series displaying actual energy vs expected baseline curves with color-coded severity markers.
4. **Energy vs Load**: Thermodynamic load-line scatter plot illustrating the operational envelope and off-design outliers.
5. **Contextual Analysis**: 4-quadrant multi-variable cross plots against Chilled Water Flow, Cooling Water Temperature, Outside Temperature, and Specific Power (kW/RT).
6. **Anomaly Investigation & Evidence**: Interactive table of abnormal episodes; selecting an episode loads its telemetry metrics, top deviating variables, and engineering hypotheses.
7. **Recommendations & Actions**: Prioritized engineering action cards (P1 Critical to P4 Routine) complete with field inspection checklists and impact estimations.
8. **Data Quality Profiler**: Ingestion diagnostics, missing value distributions, timestamp discontinuity tracking, and sampling interval verification.

---

## 8. Hackathon Judging Checklist

| Requirement | Implementation Evidence |
|---|---|
| **Meaningful ML Component** | Gradient Boosted contextual regressor + unsupervised Isolation Forest |
| **Contextual Anomaly Detection** | Detects excessive energy under low load & normal energy under peak load |
| **Multivariate Analysis** | Evaluates energy, cooling temp, chilled water flow, and weather simultaneously |
| **Equipment-Level Isolation** | Separate models, health scores, and lag baselines per chiller |
| **Time-Series Discipline** | Strictly backward-looking lags & rolling windows; zero future data leakage |
| **Time Gap & Missing Data Handling** | Tracks data acquisition gaps without false alarm fault triggers |
| **Severity & Persistence** | Clusters multi-hour episodes and scores NORMAL through CRITICAL |
| **Transparent Explainability** | Z-score feature deviations + physical causality separation disclaimer |
| **Actionable Recommendations** | Evidence-grounded engineering checklists with savings impact |
| **Zero Hardcoding** | Universal schema contract; works on any valid chiller dataset |
| **Demo Mode** | Synthetic physical generator with clear `[DEMO DATA]` labeling |
