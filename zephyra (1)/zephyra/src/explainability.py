"""
Evidence Generation and Transparent Feature-Level Explainability Engine.
Generates comprehensive evidence packages for anomalous episodes, detailing
why the model flagged the observation, which variables contributed most,
and strictly separates statistical evidence from physical causation hypotheses.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from src.config import (
    EQUIPMENT_COL,
    TIMESTAMP_COL,
    TARGET_ENERGY_COL,
    BUILDING_LOAD_COL,
    CHILLED_WATER_COL,
    COOLING_WATER_COL,
    OUTSIDE_TEMP_COL,
    DEW_POINT_COL
)


class AnomalyExplainer:
    """
    Computes feature-level standardized deviations and generates
    transparent evidence reports for anomalies and episodes.
    """

    def __init__(self):
        self.baseline_stats: Dict[str, Dict[str, Dict[str, float]]] = {}

    def fit_baselines(self, df: pd.DataFrame):
        """
        Calculates normal operating baselines (mean, std, 5th, 95th percentiles)
        for each equipment unit on non-anomalous points.
        """
        normal_df = df[df["anomaly_flag"] == 0]
        eval_cols = [
            TARGET_ENERGY_COL,
            BUILDING_LOAD_COL,
            CHILLED_WATER_COL,
            COOLING_WATER_COL,
            OUTSIDE_TEMP_COL,
            "energy_per_load",
            "energy_per_chw_flow",
            "cooling_vs_outside_diff"
        ]
        available_cols = [c for c in eval_cols if c in df.columns]

        for eq, eq_df in normal_df.groupby(EQUIPMENT_COL):
            self.baseline_stats[str(eq)] = {}
            for col in available_cols:
                vals = eq_df[col].dropna()
                if len(vals) > 0:
                    mean_val = float(vals.mean())
                    std_val = float(vals.std()) if vals.std() > 1e-4 else 1.0
                    self.baseline_stats[str(eq)][col] = {
                        "mean": mean_val,
                        "std": std_val,
                        "p05": float(vals.quantile(0.05)),
                        "p95": float(vals.quantile(0.95))
                    }

    def explain_episode(self, df: pd.DataFrame, episode: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates a comprehensive evidence report for an abnormal episode.
        """
        eq = episode["equipment_id"]
        indices = episode["indices"]
        ep_df = df.loc[indices]

        # Key aggregates during episode
        actual_energy = round(float(ep_df[TARGET_ENERGY_COL].mean()), 1)
        expected_energy = round(float(ep_df["expected_energy"].mean()), 1)
        residual = round(float(ep_df["energy_residual"].mean()), 1)
        rel_dev = round(float(ep_df["relative_deviation"].mean()), 1)

        load = round(float(ep_df[BUILDING_LOAD_COL].mean()), 1)
        chw_rate = round(float(ep_df[CHILLED_WATER_COL].mean()), 1)
        cw_temp = round(float(ep_df[COOLING_WATER_COL].mean()), 1)
        outside_temp = round(float(ep_df[OUTSIDE_TEMP_COL].mean()), 1) if OUTSIDE_TEMP_COL in ep_df.columns else None

        avg_score = episode["avg_anomaly_score"]
        persistence_count = episode["observations_count"]
        duration_hrs = episode["duration_hours"]
        severity = episode.get("max_severity", "MEDIUM")

        # Feature contribution analysis (Z-score deviation vs normal baseline)
        feature_contributions = []
        eq_baselines = self.baseline_stats.get(str(eq), {})

        for col, stats in eq_baselines.items():
            if col in ep_df.columns:
                obs_mean = float(ep_df[col].mean())
                z_score = (obs_mean - stats["mean"]) / stats["std"]
                feature_contributions.append({
                    "feature": col,
                    "observed_mean": round(obs_mean, 2),
                    "baseline_mean": round(stats["mean"], 2),
                    "baseline_range": f"[{round(stats['p05'], 1)}, {round(stats['p95'], 1)}]",
                    "deviation_z_score": round(z_score, 2),
                    "abs_z_score": round(abs(z_score), 2)
                })

        # Sort features by highest absolute Z-score deviation
        feature_contributions.sort(key=lambda x: x["abs_z_score"], reverse=True)
        top_anomalous_features = feature_contributions[:4]

        # Formulate contextual behavior patterns and hypotheses
        patterns_detected = []
        hypotheses = []

        if rel_dev > 15.0 and residual > 20.0:
            patterns_detected.append(
                f"Elevated Energy Consumption: Consuming {actual_energy} kWh vs {expected_energy} kWh expected (+{rel_dev}% deviation) under {load} RT load."
            )
            hypotheses.append("Possible compressor inefficiency, refrigerant subcooling deficit, or mechanical resistance.")

        if cw_temp > 32.0 or (cw_temp > eq_baselines.get(COOLING_WATER_COL, {}).get("p95", 33.0)):
            patterns_detected.append(
                f"High Condenser Water Temperature: Recorded at {cw_temp}°C (normal baseline ~{eq_baselines.get(COOLING_WATER_COL, {}).get('mean', 29.0):.1f}°C)."
            )
            hypotheses.append("Possible cooling tower heat rejection limitation, condenser tube scaling/fouling, or high ambient wet-bulb approach.")

        if chw_rate < eq_baselines.get(CHILLED_WATER_COL, {}).get("p05", 40.0):
            patterns_detected.append(
                f"Reduced Chilled Water Flow Rate: {chw_rate} L/sec is substantially below nominal range."
            )
            hypotheses.append("Possible evaporator valve restriction, strainer clogging, or flow sensor decalibration.")

        if not patterns_detected:
            patterns_detected.append(
                "Multivariate operational signature deviates significantly from the multidimensional historical manifold."
            )

        if not hypotheses:
            hypotheses.append(
                "Abnormal operating behaviour detected. Sensor telemetry or local operational conditions warrant on-site verification."
            )

        evidence_panel = {
            "episode_id": episode["episode_id"],
            "equipment_id": eq,
            "period": f"{episode['start_time']} → {episode['end_time']}",
            "duration_hours": duration_hrs,
            "severity": severity,
            "observations_count": persistence_count,
            "ml_anomaly_score": avg_score,
            "telemetry": {
                "actual_energy_kwh": actual_energy,
                "expected_energy_kwh": expected_energy,
                "residual_kwh": residual,
                "relative_deviation_pct": rel_dev,
                "building_load_rt": load,
                "chilled_water_rate_lps": chw_rate,
                "cooling_water_temp_c": cw_temp,
                "outside_temp_f": outside_temp
            },
            "top_deviating_features": top_anomalous_features,
            "detected_behavior": patterns_detected,
            "engineering_hypotheses": hypotheses,
            "disclaimer": (
                "NOTE: Detected signatures describe empirical operational deviations. "
                "Data correlation does not definitively prove physical mechanical causality. "
                "Physical inspection and maintenance logs are required to confirm root causes."
            )
        }

        return evidence_panel
