"""
Dynamic, Transparent Equipment Health Scoring System.
Calculates a 0-100 composite health index based on anomaly frequency,
severity distribution, persistent episode duration, and thermodynamic deviation.
Provides transparent itemized explanations for every deducted point.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from src.config import EQUIPMENT_COL, MAX_HEALTH_SCORE


def calculate_equipment_health_scores(
    df: pd.DataFrame,
    episodes: List[Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    Computes transparent, data-driven health scores for all discovered equipment.
    Returns:
        Dict mapping equipment_id -> health_profile
    """
    results = {}

    for eq, eq_df in df.groupby(EQUIPMENT_COL):
        eq_str = str(eq)
        total_obs = len(eq_df)
        if total_obs == 0:
            continue

        anomalies = eq_df[eq_df["anomaly_flag"] == 1]
        anomaly_count = len(anomalies)
        anomaly_rate_pct = (anomaly_count / total_obs) * 100.0

        # Severity breakdown
        sev_counts = eq_df["severity"].value_counts().to_dict()
        crit_count = sev_counts.get("CRITICAL", 0)
        high_count = sev_counts.get("HIGH", 0)
        med_count = sev_counts.get("MEDIUM", 0)
        low_count = sev_counts.get("LOW", 0)

        # Persistent episodes for this equipment
        eq_episodes = [ep for ep in episodes if ep["equipment_id"] == eq]
        persistent_episodes = [ep for ep in eq_episodes if ep.get("is_persistent", False)]
        persist_count = len(persistent_episodes)

        # Deviation magnitude on anomalies
        if anomaly_count > 0 and "relative_deviation" in anomalies.columns:
            mean_rel_dev = float(anomalies["relative_deviation"].abs().mean())
        else:
            mean_rel_dev = 0.0

        # Data-driven deduction calculation
        # 1. Frequency deduction (max 30 pts)
        freq_deduction = min(30.0, anomaly_rate_pct * 3.5)

        # 2. High/Critical Severity deduction (max 35 pts)
        sev_deduction = min(35.0, (high_count * 0.15) + (crit_count * 0.45))

        # 3. Persistence deduction (max 20 pts)
        persist_deduction = min(20.0, persist_count * 4.0)

        # 4. Energy deviation deduction (max 15 pts)
        dev_deduction = min(15.0, mean_rel_dev * 0.35)

        total_deduction = freq_deduction + sev_deduction + persist_deduction + dev_deduction
        final_score = round(max(10.0, MAX_HEALTH_SCORE - total_deduction), 1)

        # Health tier & classification
        if final_score >= 85.0:
            status = "Normal / Optimal"
            color_tier = "success"
        elif final_score >= 70.0:
            status = "Satisfactory (Monitor)"
            color_tier = "info"
        elif final_score >= 50.0:
            status = "Attention Required"
            color_tier = "warning"
        else:
            status = "Critical Condition"
            color_tier = "danger"

        # Itemized transparent explanations
        reasons = []
        if anomaly_rate_pct < 3.0:
            reasons.append(f"Low anomaly occurrence frequency ({anomaly_rate_pct:.1f}% of operating time).")
        else:
            reasons.append(f"Deducted {freq_deduction:.1f} pts: Elevated anomaly rate ({anomaly_rate_pct:.1f}% of total data).")

        if (crit_count + high_count) == 0:
            reasons.append("Zero high or critical severity events recorded.")
        else:
            reasons.append(
                f"Deducted {sev_deduction:.1f} pts: Detected {crit_count} Critical and {high_count} High severity intervals."
            )

        if persist_count == 0:
            reasons.append("No prolonged persistent abnormal episodes detected.")
        else:
            reasons.append(
                f"Deducted {persist_deduction:.1f} pts: Identified {persist_count} persistent multi-hour operational episodes."
            )

        if mean_rel_dev < 10.0:
            reasons.append("Energy consumption aligns tightly with contextual thermodynamic expectation.")
        else:
            reasons.append(
                f"Deducted {dev_deduction:.1f} pts: Mean abnormal energy deviation is {mean_rel_dev:.1f}% off-design."
            )

        results[eq_str] = {
            "equipment_id": eq_str,
            "health_score": final_score,
            "status": status,
            "color_tier": color_tier,
            "anomaly_count": anomaly_count,
            "anomaly_rate_pct": round(anomaly_rate_pct, 2),
            "high_critical_count": crit_count + high_count,
            "persistent_episode_count": persist_count,
            "mean_relative_deviation_pct": round(mean_rel_dev, 1),
            "deductions": {
                "frequency": round(freq_deduction, 1),
                "severity": round(sev_deduction, 1),
                "persistence": round(persist_deduction, 1),
                "deviation": round(dev_deduction, 1)
            },
            "reasons": reasons
        }

    return results
