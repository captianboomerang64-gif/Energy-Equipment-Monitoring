"""
Data-Driven Severity Engine & Persistent Anomaly Episode Clustering.
Classifies anomalies into NORMAL, LOW, MEDIUM, HIGH, CRITICAL based on
multivariate score, relative deviation, magnitude, and temporal persistence.
"""

from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd
from src.config import (
    TIMESTAMP_COL,
    EQUIPMENT_COL,
    TARGET_ENERGY_COL
)


def compute_anomaly_episodes(
    df: pd.DataFrame,
    max_gap_hours: float = 2.0
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Groups nearby anomalous timestamps into contiguous abnormal periods/episodes.
    Computes:
    - equipment
    - start time, end time, duration
    - number of anomalous observations
    - episode type (ISOLATED, REPEATED, PERSISTENT, INCREASING TREND)
    - maximum severity, average anomaly score, average deviation, total excess kWh
    """
    df = df.copy()
    df["episode_id"] = -1
    df["is_persistent"] = False

    episodes = []
    episode_counter = 0

    for eq, eq_df in df.groupby(EQUIPMENT_COL):
        anomalies = eq_df[eq_df["anomaly_flag"] == 1].sort_values(by=TIMESTAMP_COL)
        if len(anomalies) == 0:
            continue

        times = anomalies[TIMESTAMP_COL].values
        indices = anomalies.index.tolist()

        current_episode_indices = [indices[0]]

        for i in range(1, len(indices)):
            curr_time = pd.to_datetime(times[i])
            prev_time = pd.to_datetime(times[i - 1])
            gap_hours = (curr_time - prev_time).total_seconds() / 3600.0

            if gap_hours <= max_gap_hours:
                current_episode_indices.append(indices[i])
            else:
                # Close current episode
                episode_counter += 1
                ep_info = _summarize_episode(df, current_episode_indices, eq, episode_counter)
                episodes.append(ep_info)
                df.loc[current_episode_indices, "episode_id"] = episode_counter
                if ep_info["is_persistent"]:
                    df.loc[current_episode_indices, "is_persistent"] = True

                current_episode_indices = [indices[i]]

        # Close the final episode
        if current_episode_indices:
            episode_counter += 1
            ep_info = _summarize_episode(df, current_episode_indices, eq, episode_counter)
            episodes.append(ep_info)
            df.loc[current_episode_indices, "episode_id"] = episode_counter
            if ep_info["is_persistent"]:
                df.loc[current_episode_indices, "is_persistent"] = True

    return df, episodes


def _summarize_episode(
    df: pd.DataFrame,
    indices: List[int],
    equipment: str,
    ep_id: int
) -> Dict[str, Any]:
    """
    Summarizes statistical and thermodynamic metrics of an abnormal episode.
    """
    ep_df = df.loc[indices]
    start_time = ep_df[TIMESTAMP_COL].min()
    end_time = ep_df[TIMESTAMP_COL].max()
    obs_count = len(ep_df)

    duration_mins = max(30.0, (end_time - start_time).total_seconds() / 60.0)
    duration_hours = round(duration_mins / 60.0, 2)

    avg_score = round(float(ep_df["anomaly_score"].mean()), 4)
    max_score = round(float(ep_df["anomaly_score"].max()), 4)

    avg_deviation = round(float(ep_df["relative_deviation"].mean()), 2)
    max_deviation = round(float(ep_df["relative_deviation"].max()), 2)

    # Total excess energy consumed (sum of positive residuals in kWh)
    # Assuming nominal 30-min (0.5 hour) interval energy accumulation
    positive_residuals = ep_df["energy_residual"].clip(lower=0.0)
    total_excess_kwh = round(float(positive_residuals.sum()), 1)

    # Persistence classification
    if obs_count >= 3 or duration_hours >= 1.5:
        is_persistent = True
        ep_type = "PERSISTENT"
    elif obs_count == 2:
        is_persistent = False
        ep_type = "REPEATED"
    else:
        is_persistent = False
        ep_type = "ISOLATED"

    # Check for increasing trend
    if obs_count >= 3:
        residuals = ep_df["energy_residual"].values
        slope = (residuals[-1] - residuals[0]) / len(residuals)
        if slope > 3.0:
            ep_type = "INCREASING_TREND"

    return {
        "episode_id": ep_id,
        "equipment_id": equipment,
        "start_time": str(start_time),
        "end_time": str(end_time),
        "duration_hours": duration_hours,
        "observations_count": obs_count,
        "is_persistent": is_persistent,
        "episode_type": ep_type,
        "avg_anomaly_score": avg_score,
        "max_anomaly_score": max_score,
        "avg_relative_deviation_pct": avg_deviation,
        "max_relative_deviation_pct": max_deviation,
        "total_excess_energy_kwh": total_excess_kwh,
        "indices": indices
    }


def assign_severity_levels(df: pd.DataFrame, episodes: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Computes transparent multi-criteria severity for each row.
    Combines:
    1. Anomaly score weight (40%)
    2. Energy relative deviation magnitude (35%)
    3. Persistence bonus (25%)

    Outputs:
    - severity: NORMAL, LOW, MEDIUM, HIGH, CRITICAL
    - severity_score: 0 - 100
    """
    df = df.copy()
    df["severity"] = "NORMAL"
    df["severity_score"] = 0.0

    anomaly_mask = df["anomaly_flag"] == 1
    if not anomaly_mask.any():
        return df

    # Map episode info to points
    ep_lookup = {ep["episode_id"]: ep for ep in episodes}

    for idx in df[anomaly_mask].index:
        score = df.loc[idx, "anomaly_score"]  # 0.0 to 1.0
        rel_dev = abs(df.loc[idx, "relative_deviation"])  # percentage
        ep_id = df.loc[idx, "episode_id"]

        is_persistent = False
        ep_type = "ISOLATED"
        if ep_id in ep_lookup:
            is_persistent = ep_lookup[ep_id]["is_persistent"]
            ep_type = ep_lookup[ep_id]["episode_type"]

        # 1. Anomaly score component: 0 to 40 pts
        score_pts = score * 40.0

        # 2. Relative deviation component: 0 to 35 pts (clamped at 70% deviation = 35 pts)
        dev_pts = min(35.0, (rel_dev / 70.0) * 35.0)

        # 3. Persistence component: 0 to 25 pts
        if ep_type == "INCREASING_TREND":
            persist_pts = 25.0
        elif is_persistent:
            persist_pts = 20.0
        elif ep_type == "REPEATED":
            persist_pts = 10.0
        else:
            persist_pts = 2.0

        total_sev = round(score_pts + dev_pts + persist_pts, 1)
        total_sev = min(100.0, total_sev)

        # Determine level
        if total_sev >= 75.0:
            level = "CRITICAL"
        elif total_sev >= 55.0:
            level = "HIGH"
        elif total_sev >= 35.0:
            level = "MEDIUM"
        else:
            level = "LOW"

        df.loc[idx, "severity_score"] = total_sev
        df.loc[idx, "severity"] = level

    # Update episode max severity
    for ep in episodes:
        ep_indices = ep["indices"]
        ep_severities = df.loc[ep_indices, "severity"].values
        # Rank: CRITICAL > HIGH > MEDIUM > LOW
        for lvl in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
            if lvl in ep_severities:
                ep["max_severity"] = lvl
                break
        else:
            ep["max_severity"] = "LOW"

    return df
