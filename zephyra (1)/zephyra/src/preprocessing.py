"""
Data preprocessing, chronological ordering, missing-value imputation,
and timestamp discontinuity (gap) tracking.
"""

from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd
from src.config import (
    TIMESTAMP_COL,
    EQUIPMENT_COL,
    NUMERIC_COLUMNS,
    NOMINAL_INTERVAL_MINUTES,
    GAP_THRESHOLD_MINUTES
)


def detect_timestamp_gaps(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Detects temporal gaps greater than GAP_THRESHOLD_MINUTES within each equipment unit's timeline.
    Crucially, gaps are isolated as maintenance or communication outages and NEVER flagged as equipment faults.
    """
    gaps = []
    if TIMESTAMP_COL not in df.columns or EQUIPMENT_COL not in df.columns:
        return gaps

    for eq, eq_df in df.groupby(EQUIPMENT_COL):
        sorted_times = eq_df[TIMESTAMP_COL].sort_values()
        diffs = sorted_times.diff()

        gap_mask = diffs > pd.Timedelta(minutes=GAP_THRESHOLD_MINUTES)
        for idx in sorted_times[gap_mask].index:
            end_time = sorted_times.loc[idx]
            # Find the previous observation time
            prev_idx = sorted_times.index[sorted_times.index.get_loc(idx) - 1]
            start_time = sorted_times.loc[prev_idx]
            duration_hours = round((end_time - start_time).total_seconds() / 3600.0, 2)

            gaps.append({
                "equipment_id": eq,
                "gap_start": str(start_time),
                "gap_end": str(end_time),
                "duration_hours": duration_hours,
                "type": "Data Acquisition / Downtime Discontinuity"
            })

    return gaps


def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Preprocesses raw dataset:
    1. Converts timestamps & sorts chronologically per equipment.
    2. Removes duplicate timestamp records per equipment.
    3. Coerces numeric columns and isolates invalid values.
    4. Handles missing values via time-aware linear interpolation and forward-fill.
    5. Detects and logs timestamp gaps separately.

    Returns:
        (cleaned_df, detected_gaps, preprocessing_summary)
    """
    df_clean = df.copy()
    df_clean.columns = [c.strip() for c in df_clean.columns]

    initial_row_count = len(df_clean)

    # 1. Parse timestamps
    df_clean[TIMESTAMP_COL] = pd.to_datetime(df_clean[TIMESTAMP_COL])

    # 2. Coerce numeric columns
    for col in NUMERIC_COLUMNS:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    # 3. Deduplicate by equipment_id and timestamp
    df_clean = df_clean.drop_duplicates(subset=[EQUIPMENT_COL, TIMESTAMP_COL], keep="first")
    dedup_row_count = len(df_clean)
    duplicates_removed = initial_row_count - dedup_row_count

    # 4. Chronological sorting per equipment
    df_clean = df_clean.sort_values(by=[EQUIPMENT_COL, TIMESTAMP_COL]).reset_index(drop=True)

    # 5. Detect time gaps before interpolation
    detected_gaps = detect_timestamp_gaps(df_clean)

    # 6. Missing value imputation per equipment
    # Missing values must NOT be treated as anomalies!
    # Interpolate linearly for continuous physical measurements, then ffill/bfill
    processed_groups = []
    missing_imputed_count = 0

    for eq, eq_df in df_clean.groupby(EQUIPMENT_COL):
        eq_sorted = eq_df.sort_values(by=TIMESTAMP_COL).copy()
        
        # Count missing values before
        pre_missing = eq_sorted[NUMERIC_COLUMNS].isna().sum().sum()

        # Interpolate numeric features along the time index
        eq_sorted[NUMERIC_COLUMNS] = eq_sorted[NUMERIC_COLUMNS].interpolate(
            method="linear", limit=6, limit_direction="both"
        )
        # Fallback forward fill and backfill for edges
        eq_sorted[NUMERIC_COLUMNS] = eq_sorted[NUMERIC_COLUMNS].ffill().bfill()

        post_missing = eq_sorted[NUMERIC_COLUMNS].isna().sum().sum()
        missing_imputed_count += int(pre_missing - post_missing)

        processed_groups.append(eq_sorted)

    df_processed = pd.concat(processed_groups, ignore_index=True)
    df_processed = df_processed.sort_values(by=[EQUIPMENT_COL, TIMESTAMP_COL]).reset_index(drop=True)

    summary = {
        "initial_rows": initial_row_count,
        "processed_rows": len(df_processed),
        "duplicates_removed": duplicates_removed,
        "missing_imputed_cells": missing_imputed_count,
        "total_gaps_detected": len(detected_gaps)
    }

    return df_processed, detected_gaps, summary
