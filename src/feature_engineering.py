"""
Feature Engineering for Contextual and Multivariate Equipment Monitoring.
Computes time, lag, rolling, change, and thermodynamic context features per equipment.
Strictly avoids future data leakage (only past observations used).
"""

from typing import Tuple, List
import numpy as np
import pandas as pd
from src.config import (
    TIMESTAMP_COL,
    EQUIPMENT_COL,
    TARGET_ENERGY_COL,
    BUILDING_LOAD_COL,
    CHILLED_WATER_COL,
    COOLING_WATER_COL,
    OUTSIDE_TEMP_COL,
    DEW_POINT_COL,
    HUMIDITY_COL,
    WIND_SPEED_COL,
    PRESSURE_COL
)


def compute_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts chronological and cyclical calendar features.
    """
    df = df.copy()
    ts = df[TIMESTAMP_COL]

    df["hour"] = ts.dt.hour
    df["day"] = ts.dt.day
    df["day_of_week"] = ts.dt.dayofweek
    df["month"] = ts.dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Cyclical hour encodings
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24.0)

    # Cyclical month encodings
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12.0)

    return df


def compute_contextual_thermodynamic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes thermodynamic ratios, temperature differentials, and chiller efficiency indicators.
    """
    df = df.copy()
    eps = 1e-4

    # 1. Energy per building load (kW/RT specific power)
    df["energy_per_load"] = df[TARGET_ENERGY_COL] / (df[BUILDING_LOAD_COL].clip(lower=1.0) + eps)

    # 2. Energy per chilled water flow rate
    df["energy_per_chw_flow"] = df[TARGET_ENERGY_COL] / (df[CHILLED_WATER_COL].clip(lower=1.0) + eps)

    # 3. Chilled water flow per building load (L/sec per RT)
    df["chw_flow_per_load"] = df[CHILLED_WATER_COL] / (df[BUILDING_LOAD_COL].clip(lower=1.0) + eps)

    # 4. Outside temperature converted to Celsius for physical differential alignment
    outside_temp_c = (df[OUTSIDE_TEMP_COL] - 32.0) * (5.0 / 9.0)
    df["outside_temp_c"] = outside_temp_c

    # 5. Cooling water temperature minus outside ambient temperature (C)
    df["cooling_vs_outside_diff"] = df[COOLING_WATER_COL] - outside_temp_c

    # 6. Dew point spread (F) - key indicator of cooling tower wet-bulb approach limit
    df["dew_point_spread"] = df[OUTSIDE_TEMP_COL] - df[DEW_POINT_COL]

    # 7. Approximate thermodynamic lift (Cooling water supply temp - estimated chilled water supply temp 7C)
    df["approx_lift_c"] = df[COOLING_WATER_COL] - 7.0

    return df


def compute_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes lag, rolling statistics, and rate-of-change features strictly per equipment.
    All rolling statistics use shift(1) to strictly ensure NO future data leakage.
    """
    df = df.copy()
    feature_dfs = []

    for eq, eq_df in df.groupby(EQUIPMENT_COL):
        eq_sorted = eq_df.sort_values(by=TIMESTAMP_COL).copy()

        # Target variable lag features
        eq_sorted["energy_lag_1"] = eq_sorted[TARGET_ENERGY_COL].shift(1)
        eq_sorted["energy_lag_2"] = eq_sorted[TARGET_ENERGY_COL].shift(2)
        eq_sorted["energy_lag_4"] = eq_sorted[TARGET_ENERGY_COL].shift(4)

        # Operational lag features
        eq_sorted["load_lag_1"] = eq_sorted[BUILDING_LOAD_COL].shift(1)
        eq_sorted["flow_lag_1"] = eq_sorted[CHILLED_WATER_COL].shift(1)
        eq_sorted["cooling_temp_lag_1"] = eq_sorted[COOLING_WATER_COL].shift(1)

        # Rate of change features (first differences and percentage changes)
        eq_sorted["energy_diff_1"] = eq_sorted[TARGET_ENERGY_COL] - eq_sorted["energy_lag_1"]
        eq_sorted["load_diff_1"] = eq_sorted[BUILDING_LOAD_COL] - eq_sorted["load_lag_1"]
        eq_sorted["energy_pct_change"] = (
            eq_sorted["energy_diff_1"] / (eq_sorted["energy_lag_1"].abs() + 1e-3)
        ).clip(lower=-2.0, upper=2.0)

        # Rolling statistics strictly on historical observations (using shift to prevent leakage)
        hist_energy = eq_sorted[TARGET_ENERGY_COL].shift(1)
        hist_load = eq_sorted[BUILDING_LOAD_COL].shift(1)

        # Rolling 3 observations (~1.5 hours)
        eq_sorted["energy_roll_mean_3"] = hist_energy.rolling(window=3, min_periods=1).mean()
        eq_sorted["energy_roll_std_3"] = hist_energy.rolling(window=3, min_periods=1).std().fillna(0.0)
        eq_sorted["energy_roll_median_3"] = hist_energy.rolling(window=3, min_periods=1).median()

        # Rolling 6 observations (~3.0 hours)
        eq_sorted["energy_roll_mean_6"] = hist_energy.rolling(window=6, min_periods=1).mean()
        eq_sorted["energy_roll_std_6"] = hist_energy.rolling(window=6, min_periods=1).std().fillna(0.0)

        # Rolling load mean
        eq_sorted["load_roll_mean_3"] = hist_load.rolling(window=3, min_periods=1).mean()

        # Backfill initial lag rows with current values to maintain dataset size without losing rows
        fill_cols = [
            "energy_lag_1", "energy_lag_2", "energy_lag_4",
            "load_lag_1", "flow_lag_1", "cooling_temp_lag_1",
            "energy_diff_1", "load_diff_1", "energy_pct_change",
            "energy_roll_mean_3", "energy_roll_std_3", "energy_roll_median_3",
            "energy_roll_mean_6", "energy_roll_std_6", "load_roll_mean_3"
        ]
        eq_sorted[fill_cols] = eq_sorted[fill_cols].bfill().fillna(0.0)

        feature_dfs.append(eq_sorted)

    df_featured = pd.concat(feature_dfs, ignore_index=True)
    df_featured = df_featured.sort_values(by=[EQUIPMENT_COL, TIMESTAMP_COL]).reset_index(drop=True)
    return df_featured


def engineer_all_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Executes full feature engineering pipeline.
    Returns:
        (featured_dataframe, list_of_engineered_feature_columns)
    """
    df_step1 = compute_time_features(df)
    df_step2 = compute_contextual_thermodynamic_features(df_step1)
    df_final = compute_lag_and_rolling_features(df_step2)

    engineered_cols = [
        # Time
        "hour", "day", "day_of_week", "month", "is_weekend",
        "sin_hour", "cos_hour", "sin_month", "cos_month",
        # Thermodynamic context
        "energy_per_load", "energy_per_chw_flow", "chw_flow_per_load",
        "outside_temp_c", "cooling_vs_outside_diff", "dew_point_spread", "approx_lift_c",
        # Lags & changes
        "energy_lag_1", "energy_lag_2", "energy_lag_4",
        "load_lag_1", "flow_lag_1", "cooling_temp_lag_1",
        "energy_diff_1", "load_diff_1", "energy_pct_change",
        # Rolling
        "energy_roll_mean_3", "energy_roll_std_3", "energy_roll_median_3",
        "energy_roll_mean_6", "energy_roll_std_6", "load_roll_mean_3"
    ]

    return df_final, engineered_cols
