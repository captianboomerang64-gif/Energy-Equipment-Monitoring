"""
Comprehensive Integration Test Suite for Intelligent Energy Monitoring Pipeline.
Tests data validation, preprocessing, feature engineering, ML models, severity,
persistence, explainability, and recommendations end-to-end.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import generate_synthetic_demo_data, validate_schema, load_dataset
from src.preprocessing import preprocess_data
from src.feature_engineering import engineer_all_features
from src.baseline_model import ContextualExpectedEnergyModel
from src.anomaly_detection import ContextualAnomalyDetector
from src.severity import compute_anomaly_episodes, assign_severity_levels
from src.explainability import AnomalyExplainer
from src.recommendations import generate_recommendations
from src.health_score import calculate_equipment_health_scores
from src.visualization import (
    plot_anomaly_timeline,
    plot_energy_vs_load_scatter,
    plot_contextual_scatter,
    plot_health_gauge
)


class TestEnergyMonitoringPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Generate 10-day synthetic test dataset
        cls.df_synth = generate_synthetic_demo_data(num_days=10, seed=42)
        
        # Check if real development dataset is present
        cls.real_data_path = os.path.join("data", "sample", "development_dataset.csv")

    def test_01_schema_validation(self):
        is_valid, errors, warnings = validate_schema(self.df_synth)
        self.assertTrue(is_valid, f"Synthetic schema validation failed: {errors}")
        self.assertEqual(len(errors), 0)

    def test_02_preprocessing(self):
        cleaned_df, gaps, summary = preprocess_data(self.df_synth)
        self.assertEqual(len(cleaned_df), len(self.df_synth))
        self.assertFalse(cleaned_df.isna().any().any(), "Cleaned dataset contains NaNs")
        self.assertIn("initial_rows", summary)

    def test_03_feature_engineering(self):
        cleaned_df, _, _ = preprocess_data(self.df_synth)
        featured_df, feat_cols = engineer_all_features(cleaned_df)
        self.assertIn("sin_hour", featured_df.columns)
        self.assertIn("energy_per_load", featured_df.columns)
        self.assertIn("energy_lag_1", featured_df.columns)
        self.assertIn("energy_roll_mean_3", featured_df.columns)
        self.assertFalse(featured_df[feat_cols].isna().any().any(), "Engineered features have NaNs")

    def test_04_baseline_expected_energy_model(self):
        cleaned_df, _, _ = preprocess_data(self.df_synth)
        featured_df, _ = engineer_all_features(cleaned_df)
        regressor = ContextualExpectedEnergyModel()
        df_expected, metrics = regressor.fit_predict(featured_df)

        self.assertIn("expected_energy", df_expected.columns)
        self.assertIn("energy_residual", df_expected.columns)
        self.assertIn("relative_deviation", df_expected.columns)

        for eq, m in metrics.items():
            self.assertGreater(m["r2_score"], 0.65, f"R2 score too low for {eq}: {m['r2_score']}")

    def test_05_unsupervised_anomaly_detection(self):
        cleaned_df, _, _ = preprocess_data(self.df_synth)
        featured_df, _ = engineer_all_features(cleaned_df)
        regressor = ContextualExpectedEnergyModel()
        df_expected, _ = regressor.fit_predict(featured_df)

        detector = ContextualAnomalyDetector(contamination=0.05)
        df_anom, det_metrics = detector.fit_predict(df_expected)

        self.assertIn("anomaly_score", df_anom.columns)
        self.assertIn("anomaly_flag", df_anom.columns)
        self.assertTrue((df_anom["anomaly_score"] >= 0.0).all())
        self.assertTrue((df_anom["anomaly_score"] <= 1.0).all())
        self.assertGreater(df_anom["anomaly_flag"].sum(), 0)

    def test_06_severity_and_persistence(self):
        cleaned_df, _, _ = preprocess_data(self.df_synth)
        featured_df, _ = engineer_all_features(cleaned_df)
        regressor = ContextualExpectedEnergyModel()
        df_expected, _ = regressor.fit_predict(featured_df)
        detector = ContextualAnomalyDetector(contamination=0.05)
        df_anom, _ = detector.fit_predict(df_expected)

        df_episodes, episodes = compute_anomaly_episodes(df_anom)
        df_severity = assign_severity_levels(df_episodes, episodes)

        self.assertIn("severity", df_severity.columns)
        self.assertIn("severity_score", df_severity.columns)
        valid_sevs = {"NORMAL", "LOW", "MEDIUM", "HIGH", "CRITICAL"}
        self.assertTrue(set(df_severity["severity"].unique()).issubset(valid_sevs))

    def test_07_health_score_and_explainability(self):
        cleaned_df, _, _ = preprocess_data(self.df_synth)
        featured_df, _ = engineer_all_features(cleaned_df)
        regressor = ContextualExpectedEnergyModel()
        df_expected, _ = regressor.fit_predict(featured_df)
        detector = ContextualAnomalyDetector(contamination=0.05)
        df_anom, _ = detector.fit_predict(df_expected)
        df_episodes, episodes = compute_anomaly_episodes(df_anom)
        df_severity = assign_severity_levels(df_episodes, episodes)

        health_scores = calculate_equipment_health_scores(df_severity, episodes)
        self.assertGreater(len(health_scores), 0)
        for eq, h in health_scores.items():
            self.assertGreaterEqual(h["health_score"], 0.0)
            self.assertLessEqual(h["health_score"], 100.0)
            self.assertGreater(len(h["reasons"]), 0)

        explainer = AnomalyExplainer()
        explainer.fit_baselines(df_severity)
        if episodes:
            ev = explainer.explain_episode(df_severity, episodes[0])
            self.assertIn("telemetry", ev)
            self.assertIn("top_deviating_features", ev)
            self.assertIn("disclaimer", ev)

    def test_08_recommendations(self):
        cleaned_df, _, _ = preprocess_data(self.df_synth)
        featured_df, _ = engineer_all_features(cleaned_df)
        regressor = ContextualExpectedEnergyModel()
        df_expected, _ = regressor.fit_predict(featured_df)
        detector = ContextualAnomalyDetector(contamination=0.05)
        df_anom, _ = detector.fit_predict(df_expected)
        df_episodes, episodes = compute_anomaly_episodes(df_anom)
        df_severity = assign_severity_levels(df_episodes, episodes)
        health_scores = calculate_equipment_health_scores(df_severity, episodes)

        recs = generate_recommendations(df_severity, episodes, health_scores)
        self.assertIsInstance(recs, list)
        if recs:
            self.assertIn("priority", recs[0])
            self.assertIn("action_checklist", recs[0])

    def test_09_visualizations_render(self):
        cleaned_df, _, _ = preprocess_data(self.df_synth)
        featured_df, _ = engineer_all_features(cleaned_df)
        regressor = ContextualExpectedEnergyModel()
        df_expected, _ = regressor.fit_predict(featured_df)
        detector = ContextualAnomalyDetector(contamination=0.05)
        df_anom, _ = detector.fit_predict(df_expected)
        df_episodes, episodes = compute_anomaly_episodes(df_anom)
        df_severity = assign_severity_levels(df_episodes, episodes)

        fig1 = plot_anomaly_timeline(df_severity)
        fig2 = plot_energy_vs_load_scatter(df_severity)
        fig3 = plot_health_gauge(85.5, "CHILLER-01")

        self.assertIsNotNone(fig1)
        self.assertIsNotNone(fig2)
        self.assertIsNotNone(fig3)


if __name__ == "__main__":
    unittest.main()
