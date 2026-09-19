"""
Validation test on the real 25,003-row historical development dataset.
"""

import time
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import load_dataset
from src.preprocessing import preprocess_data
from src.feature_engineering import engineer_all_features
from src.baseline_model import ContextualExpectedEnergyModel
from src.anomaly_detection import ContextualAnomalyDetector
from src.severity import compute_anomaly_episodes, assign_severity_levels
from src.explainability import AnomalyExplainer
from src.recommendations import generate_recommendations
from src.health_score import calculate_equipment_health_scores


def main():
    data_path = os.path.join("data", "sample", "development_dataset.csv")
    print(f"Loading {data_path}...")
    start_t = time.time()

    df_raw, report, errors = load_dataset(data_path)
    if errors:
        print("Errors loading:", errors)
        sys.exit(1)

    print(f"Loaded {len(df_raw)} rows across equipment: {report['discovered_equipment']}")

    print("Running preprocessing...")
    df_clean, gaps, prep_summary = preprocess_data(df_raw)
    print(f"Preprocessed in {time.time() - start_t:.2f}s | Gaps detected: {len(gaps)}")

    print("Running feature engineering...")
    t1 = time.time()
    df_feat, feat_cols = engineer_all_features(df_clean)
    print(f"Feature engineering completed in {time.time() - t1:.2f}s ({len(feat_cols)} engineered features)")

    print("Training Contextual Expected Energy Model (HistGradientBoosting)...")
    t2 = time.time()
    reg = ContextualExpectedEnergyModel()
    df_exp, reg_metrics = reg.fit_predict(df_feat)
    print(f"Regression completed in {time.time() - t2:.2f}s")
    for eq, m in reg_metrics.items():
        print(f"  {eq} -> R2: {m['r2_score']}, MAE: {m['mae_kwh']} kWh, RMSE: {m['rmse_kwh']} kWh")

    print("Running Unsupervised Isolation Forest Anomaly Detector...")
    t3 = time.time()
    det = ContextualAnomalyDetector(contamination=0.05)
    df_anom, det_metrics = det.fit_predict(df_exp)
    print(f"Anomaly detection completed in {time.time() - t3:.2f}s")
    for eq, m in det_metrics.items():
        print(f"  {eq} -> Anomaly count: {m['anomaly_points']}/{m['total_points']} ({m['anomaly_rate_pct']}%)")

    print("Clustering persistent abnormal episodes & assigning multi-criteria severity...")
    t4 = time.time()
    df_ep, episodes = compute_anomaly_episodes(df_anom)
    df_sev = assign_severity_levels(df_ep, episodes)
    print(f"Identified {len(episodes)} abnormal episodes in {time.time() - t4:.2f}s")

    print("Calculating Equipment Health Scores...")
    health = calculate_equipment_health_scores(df_sev, episodes)
    for eq, h in health.items():
        print(f"  {eq} -> Health Score: {h['health_score']}/100 ({h['status']})")
        for r in h["reasons"]:
            print(f"    • {r}")

    print("Generating Explainability Evidence & Recommendations...")
    explainer = AnomalyExplainer()
    explainer.fit_baselines(df_sev)
    recs = generate_recommendations(df_sev, episodes, health)
    print(f"Generated {len(recs)} prioritized operational recommendations.")
    for rec in recs[:3]:
        print(f"  [{rec['priority']}] {rec['id']}: {rec['title']} ({rec['equipment_id']})")

    total_time = time.time() - start_t
    print(f"\n=======================================================")
    print(f"SUCCESS! Entire 25,003-row pipeline completed in {total_time:.2f} seconds!")
    print(f"=======================================================")


if __name__ == "__main__":
    main()
