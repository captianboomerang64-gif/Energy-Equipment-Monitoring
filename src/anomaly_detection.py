"""
Unsupervised Contextual & Multivariate Anomaly Detection using Isolation Forest.
Trained per equipment unit on multidimensional operational, environmental,
and thermodynamic residual features. Generates continuous anomaly_score and binary anomaly_flag.
"""

from typing import Tuple, Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from src.config import (
    EQUIPMENT_COL,
    TARGET_ENERGY_COL,
    BUILDING_LOAD_COL,
    CHILLED_WATER_COL,
    COOLING_WATER_COL,
    OUTSIDE_TEMP_COL,
    HUMIDITY_COL,
    IFOREST_CONTAMINATION,
    IFOREST_ESTIMATORS,
    RANDOM_STATE
)

ANOMALY_FEATURE_CANDIDATES = [
    "energy_residual",
    "relative_deviation",
    "energy_per_load",
    "energy_per_chw_flow",
    "chw_flow_per_load",
    BUILDING_LOAD_COL,
    CHILLED_WATER_COL,
    COOLING_WATER_COL,
    OUTSIDE_TEMP_COL,
    HUMIDITY_COL,
    "approx_lift_c",
    "cooling_vs_outside_diff",
    "dew_point_spread",
    "energy_diff_1",
    "energy_pct_change",
    "energy_roll_std_3",
    "energy_roll_mean_3"
]


class ContextualAnomalyDetector:
    """
    Unsupervised Multivariate Isolation Forest Anomaly Detector.
    Learns normal operational manifold and flags points deviating in high-dimensional feature space.
    """

    def __init__(
        self,
        contamination: float = IFOREST_CONTAMINATION,
        n_estimators: int = IFOREST_ESTIMATORS,
        random_state: int = RANDOM_STATE
    ):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.models: Dict[str, IsolationForest] = {}
        self.feature_columns: List[str] = []

    def fit_predict(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Fits Isolation Forest per equipment and computes:
        - anomaly_score: calibrated float in [0.0, 1.0] (higher = more anomalous)
        - anomaly_flag: int (1 = anomaly, 0 = normal)
        """
        df_out = df.copy()
        df_out["anomaly_score"] = 0.0
        df_out["anomaly_flag"] = 0

        # Select available features
        features_to_use = [f for f in ANOMALY_FEATURE_CANDIDATES if f in df_out.columns]
        self.feature_columns = features_to_use

        equipment_summaries = {}

        for eq, eq_indices in df_out.groupby(EQUIPMENT_COL).groups.items():
            eq_subset = df_out.loc[eq_indices]
            X = eq_subset[features_to_use].fillna(0.0)

            # Fit Isolation Forest
            iforest = IsolationForest(
                n_estimators=self.n_estimators,
                contamination=self.contamination,
                random_state=self.random_state,
                n_jobs=-1
            )
            iforest.fit(X)
            self.models[eq] = iforest

            # Predictions: 1 for inlier, -1 for outlier
            raw_preds = iforest.predict(X)
            flags = (raw_preds == -1).astype(int)

            # Score: score_samples returns opposite of anomaly score (lower is more anomalous)
            # Scikit-learn score is typically in range [-0.7, -0.3]
            raw_scores = iforest.score_samples(X)
            # Calibrate: invert so higher = more anomalous
            # Normalize to [0.0, 1.0] range within this equipment dataset
            min_s, max_s = raw_scores.min(), raw_scores.max()
            if max_s > min_s:
                calibrated_scores = 1.0 - ((raw_scores - min_s) / (max_s - min_s))
            else:
                calibrated_scores = np.zeros(len(raw_scores))

            # Store results
            df_out.loc[eq_indices, "anomaly_score"] = np.round(calibrated_scores, 4)
            df_out.loc[eq_indices, "anomaly_flag"] = flags

            anomaly_count = int(flags.sum())
            total_count = len(flags)
            anomaly_rate = round((anomaly_count / total_count) * 100, 2)

            equipment_summaries[str(eq)] = {
                "total_points": total_count,
                "anomaly_points": anomaly_count,
                "anomaly_rate_pct": anomaly_rate,
                "mean_anomaly_score": round(float(np.mean(calibrated_scores)), 4)
            }

        return df_out, equipment_summaries
