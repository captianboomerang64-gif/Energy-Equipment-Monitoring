"""
Contextual Expected Energy Consumption Baseline Model.
Uses non-linear Gradient Boosting to predict expected energy under varying
operational, thermodynamic, and ambient meteorological conditions.
Calculates expected_energy, energy_residual, and relative_deviation.
"""

from typing import Tuple, Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from src.config import (
    EQUIPMENT_COL,
    TARGET_ENERGY_COL,
    BUILDING_LOAD_COL,
    CHILLED_WATER_COL,
    COOLING_WATER_COL,
    OUTSIDE_TEMP_COL,
    DEW_POINT_COL,
    HUMIDITY_COL,
    WIND_SPEED_COL,
    PRESSURE_COL,
    RANDOM_STATE
)

# Feature set for predicting thermodynamic expected energy consumption
BASELINE_INPUT_FEATURES = [
    BUILDING_LOAD_COL,
    CHILLED_WATER_COL,
    COOLING_WATER_COL,
    OUTSIDE_TEMP_COL,
    DEW_POINT_COL,
    HUMIDITY_COL,
    WIND_SPEED_COL,
    PRESSURE_COL,
    "sin_hour",
    "cos_hour",
    "day_of_week",
    "is_weekend",
    "approx_lift_c",
    "cooling_vs_outside_diff",
    "dew_point_spread",
    "load_lag_1",
    "flow_lag_1",
    "cooling_temp_lag_1",
    "load_roll_mean_3"
]


class ContextualExpectedEnergyModel:
    """
    Fits and manages equipment-specific expected energy regressors.
    Each chiller unit possesses distinct compressor curves and heat exchanger geometry,
    so models are trained per equipment to learn individual thermodynamic baselines.
    """

    def __init__(self, random_state: int = RANDOM_STATE):
        self.random_state = random_state
        self.models: Dict[str, HistGradientBoostingRegressor] = {}
        self.metrics: Dict[str, Dict[str, float]] = {}
        self.feature_names: List[str] = BASELINE_INPUT_FEATURES

    def fit_predict(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Trains baseline model per equipment and generates expected_energy,
        energy_residual, and relative_deviation.
        """
        df_out = df.copy()
        df_out["expected_energy"] = np.nan
        df_out["energy_residual"] = np.nan
        df_out["relative_deviation"] = np.nan

        available_features = [f for f in self.feature_names if f in df_out.columns]

        model_summaries = {}

        for eq, eq_indices in df_out.groupby(EQUIPMENT_COL).groups.items():
            eq_subset = df_out.loc[eq_indices]
            X = eq_subset[available_features].fillna(0.0)
            y = eq_subset[TARGET_ENERGY_COL].values

            # Initialize HistGradientBoostingRegressor with robust loss
            model = HistGradientBoostingRegressor(
                loss="squared_error",
                max_iter=120,
                learning_rate=0.08,
                max_depth=7,
                min_samples_leaf=15,
                random_state=self.random_state
            )
            model.fit(X, y)
            self.models[eq] = model

            # In-sample expected predictions
            y_pred = model.predict(X)
            # Ensure physical floor (energy cannot be negative)
            y_pred = np.clip(y_pred, a_min=1.0, a_max=None)

            residuals = y - y_pred
            # Relative percentage deviation: (Actual - Expected) / Expected * 100
            eps = 1e-3
            rel_dev = (residuals / (y_pred + eps)) * 100.0

            df_out.loc[eq_indices, "expected_energy"] = np.round(y_pred, 2)
            df_out.loc[eq_indices, "energy_residual"] = np.round(residuals, 2)
            df_out.loc[eq_indices, "relative_deviation"] = np.round(rel_dev, 2)

            # Performance metrics (R2, MAE, RMSE)
            mae = float(np.mean(np.abs(residuals)))
            rmse = float(np.sqrt(np.mean(residuals ** 2)))
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = float(1 - (np.sum(residuals ** 2) / (ss_tot + 1e-8)))

            model_summaries[str(eq)] = {
                "r2_score": round(r2, 4),
                "mae_kwh": round(mae, 2),
                "rmse_kwh": round(rmse, 2),
                "observations": len(X)
            }

        self.metrics = model_summaries
        return df_out, model_summaries
