"""
Data ingestion, schema validation, quality profiling, and synthetic demo dataset generation.
Ensures no assumptions about filename, row count, or hardcoded equipment names.
"""

from typing import Tuple, Dict, Any, Optional
import io
import numpy as np
import pandas as pd
from src.config import (
    REQUIRED_COLUMNS,
    NUMERIC_COLUMNS,
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


def validate_schema(df: pd.DataFrame) -> Tuple[bool, list, list]:
    """
    Validate that the DataFrame satisfies the required schema contract.
    Returns:
        (is_valid, error_messages, warning_messages)
    """
    errors = []
    warnings = []

    # Clean whitespace from column names
    df.columns = [c.strip() for c in df.columns]

    # Check for missing required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {', '.join(missing_cols)}")
        return False, errors, warnings

    # Check for empty dataframe
    if len(df) == 0:
        errors.append("Uploaded dataset is empty.")
        return False, errors, warnings

    # Check equipment_id existence and values
    unique_equipment = df[EQUIPMENT_COL].dropna().unique()
    if len(unique_equipment) == 0:
        errors.append("No valid equipment identifiers found in equipment_id column.")
        return False, errors, warnings

    # Check if timestamp can be parsed
    sample_timestamps = df[TIMESTAMP_COL].dropna().head(20)
    try:
        pd.to_datetime(sample_timestamps)
    except Exception as e:
        errors.append(f"Timestamp column could not be parsed as date/time: {str(e)}")
        return False, errors, warnings

    return True, errors, warnings


def generate_quality_report(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate an exhaustive data quality report.
    """
    df_clean = df.copy()
    df_clean.columns = [c.strip() for c in df_clean.columns]

    total_rows = len(df_clean)
    total_cols = len(df_clean.columns)
    duplicate_count = int(df_clean.duplicated().sum())
    
    # Missing values analysis
    missing_per_col = df_clean.isna().sum().to_dict()
    total_missing_cells = int(df_clean.isna().sum().sum())
    missing_pct = round((total_missing_cells / (total_rows * total_cols)) * 100, 2) if total_rows > 0 else 0

    # Equipment breakdown
    if EQUIPMENT_COL in df_clean.columns:
        equipment_counts = df_clean[EQUIPMENT_COL].value_counts().to_dict()
        discovered_equipment = list(equipment_counts.keys())
    else:
        equipment_counts = {}
        discovered_equipment = []

    # Timestamp analysis
    time_range = None
    median_interval_mins = None
    if TIMESTAMP_COL in df_clean.columns:
        try:
            ts_series = pd.to_datetime(df_clean[TIMESTAMP_COL])
            time_range = {
                "start": str(ts_series.min()),
                "end": str(ts_series.max()),
                "span_days": round((ts_series.max() - ts_series.min()).total_seconds() / 86400, 1)
            }
            if len(ts_series) > 1:
                # Calculate diffs on sample of unique sorted times
                sample_diffs = ts_series.drop_duplicates().sort_values().diff().dropna()
                if len(sample_diffs) > 0:
                    median_interval_mins = round(sample_diffs.median().total_seconds() / 60, 1)
        except Exception:
            pass

    return {
        "total_rows": total_rows,
        "total_cols": total_cols,
        "duplicate_rows": duplicate_count,
        "missing_cells": total_missing_cells,
        "missing_percentage": missing_pct,
        "missing_by_column": missing_per_col,
        "discovered_equipment": discovered_equipment,
        "equipment_counts": equipment_counts,
        "time_range": time_range,
        "median_interval_minutes": median_interval_mins
    }


def load_dataset(file_or_path) -> Tuple[Optional[pd.DataFrame], Dict[str, Any], list]:
    """
    Loads dataset from file path, StringIO, BytesIO, or Streamlit UploadedFile.
    Validates schema and generates data quality profile.
    Returns:
        (dataframe or None, quality_report, errors)
    """
    errors = []
    try:
        if hasattr(file_or_path, "seek"):
            file_or_path.seek(0)
        df = pd.read_csv(file_or_path)
    except Exception as e:
        errors.append(f"Failed to read CSV file: {str(e)}")
        return None, {}, errors

    # Clean and normalize column names with case-insensitive and whitespace-tolerant matching
    col_mapping = {}
    actual_cols_lower = {c.strip().lower(): c for c in df.columns}
    for req_col in REQUIRED_COLUMNS:
        req_lower = req_col.strip().lower()
        if req_lower in actual_cols_lower:
            orig_col = actual_cols_lower[req_lower]
            if orig_col != req_col:
                col_mapping[orig_col] = req_col

    if col_mapping:
        df = df.rename(columns=col_mapping)

    df.columns = [c.strip() for c in df.columns]

    is_valid, validation_errors, _ = validate_schema(df)
    if not is_valid:
        return None, {}, validation_errors

    report = generate_quality_report(df)
    return df, report, []


def generate_synthetic_demo_data(num_days: int = 45, seed: int = 42) -> pd.DataFrame:
    """
    Generates a thermodynamically realistic synthetic chiller dataset
    conforming strictly to the 11-column schema.
    Clearly designated as DEMO DATA.
    """
    np.random.seed(seed)
    periods = num_days * 48  # 48 intervals of 30 mins per day
    timestamps = pd.date_range("2026-06-01 00:00:00", periods=periods, freq="30min")

    records = []
    equipment_names = ["CHILLER-01", "CHILLER-02", "CHILLER-03"]

    for eq in equipment_names:
        # Equipment specific baseline efficiency factor
        eff_factor = 0.22 if eq == "CHILLER-01" else (0.24 if eq == "CHILLER-02" else 0.23)

        for i, ts in enumerate(timestamps):
            hour = ts.hour
            doy = ts.dayofyear
            is_workday = 1 if ts.weekday() < 5 else 0

            # Ambient weather dynamics
            diurnal_temp = 75.0 + 12.0 * np.sin((hour - 8) * np.pi / 12) + np.random.normal(0, 1.5)
            outside_temp = max(60.0, diurnal_temp)
            dew_point = outside_temp - np.random.uniform(5.0, 15.0)
            humidity = np.clip(100 - (outside_temp - dew_point) * 4.5 + np.random.normal(0, 3), 35, 98)
            wind_speed = np.clip(np.random.rayleigh(8.0), 1.0, 32.0)
            pressure = 29.85 + np.random.normal(0, 0.08)

            # Building Load profile (higher on workdays 8am - 6pm and warmer days)
            workday_boost = 250.0 if (is_workday and 8 <= hour <= 18) else 80.0
            weather_load = max(0.0, (outside_temp - 65.0) * 8.5)
            load = np.clip(220.0 + workday_boost + weather_load + np.random.normal(0, 25), 150.0, 650.0)

            # Chilled water rate follows load proportionally (approx 0.18 L/sec per RT)
            flow_rate = np.clip(load * 0.18 + np.random.normal(0, 2.5), 30.0, 130.0)

            # Cooling water temp depends on ambient wet bulb / outdoor temp and chiller heat rejection
            cooling_temp = np.clip(26.0 + (outside_temp - 70) * 0.22 + (load / 600.0) * 3.5 + np.random.normal(0, 0.6), 22.0, 36.5)

            # Baseline expected thermodynamic energy consumption:
            # Power (kW) approx = Load * (baseline_eff + lift_penalty)
            lift = max(0.0, cooling_temp - 7.0)  # chilled water delivery ~7C
            baseline_power = load * (eff_factor + 0.0028 * lift) + np.random.normal(0, 2.0)

            # Inject realistic contextual anomalies for demonstration:
            # 1. CHILLER-02: Condenser fouling episode (high cooling temp causing excessive power)
            if eq == "CHILLER-02" and 500 <= i <= 525:
                cooling_temp += 4.5
                baseline_power *= 1.38  # High energy anomaly under moderate load

            # 2. CHILLER-03: Sensor drift / severe valve hunting episode
            if eq == "CHILLER-03" and 900 <= i <= 918:
                flow_rate *= 0.55
                baseline_power *= 1.45  # Inefficient low-flow surge

            # 3. CHILLER-01: Low load penalty (running high energy during night hours)
            if eq == "CHILLER-01" and 1200 <= i <= 1215:
                baseline_power += 65.0  # Moderate load, unexplainably high consumption

            energy = max(15.0, baseline_power)

            records.append({
                TIMESTAMP_COL: str(ts),
                EQUIPMENT_COL: eq,
                CHILLED_WATER_COL: round(flow_rate, 1),
                COOLING_WATER_COL: round(cooling_temp, 1),
                BUILDING_LOAD_COL: round(load, 1),
                TARGET_ENERGY_COL: round(energy, 1),
                OUTSIDE_TEMP_COL: round(outside_temp, 1),
                DEW_POINT_COL: round(dew_point, 1),
                HUMIDITY_COL: round(humidity, 1),
                WIND_SPEED_COL: round(wind_speed, 1),
                PRESSURE_COL: round(pressure, 2)
            })

    df_synth = pd.DataFrame(records)
    return df_synth
