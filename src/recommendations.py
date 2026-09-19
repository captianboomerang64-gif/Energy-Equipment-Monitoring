"""
Evidence-Grounded Operational Recommendations Engine.
Translates detected anomalies, severity levels, and thermodynamic deviations
into prioritized engineering actions for facility managers and chiller technicians.
"""

from typing import List, Dict, Any
import pandas as pd
from src.config import (
    COOLING_WATER_COL,
    CHILLED_WATER_COL,
    BUILDING_LOAD_COL,
    TARGET_ENERGY_COL
)


def generate_recommendations(
    df: pd.DataFrame,
    episodes: List[Dict[str, Any]],
    health_summaries: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Synthesizes fleet telemetry, abnormal episodes, and health status into
    a prioritized action matrix. Every recommendation is directly grounded
    in empirical observation.
    """
    recommendations = []
    rec_id = 1

    # 1. Critical and High Severity Persistent Episodes (P1 Immediate Action)
    critical_or_high_episodes = [
        ep for ep in episodes if ep.get("max_severity") in ["CRITICAL", "HIGH"]
    ]

    for ep in critical_or_high_episodes:
        eq = ep["equipment_id"]
        rel_dev = ep["avg_relative_deviation_pct"]
        excess_kwh = ep.get("total_excess_energy_kwh", 0.0)
        duration = ep["duration_hours"]

        rec = {
            "id": f"REC-{rec_id:03d}",
            "priority": "P1 - CRITICAL",
            "equipment_id": eq,
            "category": "Compressor & Lift Optimization",
            "title": f"Mitigate Severe Persistent Energy Deviation (+{rel_dev}% vs Expected)",
            "evidence_basis": (
                f"Sustained abnormal episode lasting {duration} hours ({ep['observations_count']} cycles) "
                f"with peak {ep.get('max_severity')} severity, accumulating ~{excess_kwh} kWh in excess consumption."
            ),
            "action_checklist": [
                "Perform physical inspection of compressor motor current draw and vibration levels.",
                "Verify refrigerant suction/discharge pressures against manufacturer P-T charts to rule out internal leakage.",
                "Review chiller staging logic to shift peak tonnage to higher-efficiency companion units if operational."
            ],
            "estimated_impact": f"Eliminate up to {excess_kwh:.0f} kWh wasteful draw per occurrence and prevent compressor motor overheating."
        }
        recommendations.append(rec)
        rec_id += 1

    # 2. Condenser Water & Heat Rejection Inefficiencies (P2 Priority)
    for eq, h_data in health_summaries.items():
        eq_df = df[df["equipment_id"] == eq]
        anom_subset = eq_df[eq_df["anomaly_flag"] == 1]

        if len(anom_subset) > 0 and COOLING_WATER_COL in anom_subset.columns:
            high_cw_count = (anom_subset[COOLING_WATER_COL] > 32.5).sum()
            if high_cw_count >= 3:
                max_cw = round(anom_subset[COOLING_WATER_COL].max(), 1)
                rec = {
                    "id": f"REC-{rec_id:03d}",
                    "priority": "P2 - HIGH",
                    "equipment_id": eq,
                    "category": "Condenser & Cooling Tower Heat Rejection",
                    "title": "Inspect Condenser Tube Cleanliness & Cooling Tower Approach",
                    "evidence_basis": (
                        f"Detected {high_cw_count} anomalous intervals where cooling water temperature surged up to {max_cw}°C, "
                        f"elevating compressor lift and degrading system COP."
                    ),
                    "action_checklist": [
                        "Measure cooling tower temperature approach (Condenser Water Supply minus Ambient Wet-Bulb).",
                        "Inspect cooling tower fan variable-frequency drives (VFD) and water distribution nozzles for scale/debris.",
                        "Schedule eddy-current testing or mechanical condenser tube brushing during upcoming maintenance outage."
                    ],
                    "estimated_impact": "Lowering condenser water temperature by 1°C recovers ~2.5% to 3.5% chiller electrical efficiency."
                }
                recommendations.append(rec)
                rec_id += 1

    # 3. Chilled Water Flow & Hydraulic Distribution (P3 Priority)
    for eq, h_data in health_summaries.items():
        eq_df = df[df["equipment_id"] == eq]
        anom_subset = eq_df[eq_df["anomaly_flag"] == 1]

        if len(anom_subset) > 0 and CHILLED_WATER_COL in anom_subset.columns:
            low_flow_count = (anom_subset[CHILLED_WATER_COL] < 45.0).sum()
            if low_flow_count >= 3:
                min_flow = round(anom_subset[CHILLED_WATER_COL].min(), 1)
                rec = {
                    "id": f"REC-{rec_id:03d}",
                    "priority": "P3 - MEDIUM",
                    "equipment_id": eq,
                    "category": "Hydraulic & Evaporator Flow",
                    "title": "Audit Evaporator Chilled Water Flow Dynamics",
                    "evidence_basis": (
                        f"Observed {low_flow_count} intervals with throttled chilled water flow as low as {min_flow} L/sec, "
                        f"inducing laminar flow conditions and low Delta-T syndrome."
                    ),
                    "action_checklist": [
                        "Inspect secondary chilled water 2-way and 3-way control valve modulation.",
                        "Inspect evaporator differential pressure sensors and inline Y-strainers for blockage.",
                        "Recalibrate ultrasonic or electromagnetic flow transmitters."
                    ],
                    "estimated_impact": "Restores turbulent heat transfer in evaporator bundles and avoids nuisance freeze-stat alarms."
                }
                recommendations.append(rec)
                rec_id += 1

    # 4. Routine Baseline Maintenance & Monitoring
    for eq, h_data in health_summaries.items():
        score = h_data.get("health_score", 100)
        if score >= 85 and not any(r["equipment_id"] == eq for r in recommendations):
            rec = {
                "id": f"REC-{rec_id:03d}",
                "priority": "P4 - ROUTINE",
                "equipment_id": eq,
                "category": "Preventive Maintenance",
                "title": "Maintain Continuous Predictive Telemetry & Optimal Dispatch",
                "evidence_basis": f"Equipment demonstrates high operational health index ({score}/100) with low anomaly frequency.",
                "action_checklist": [
                    "Continue automated baseline monitoring.",
                    "Verify quarterly lubrication and oil filter differential pressure as scheduled.",
                    "Verify accuracy of outside ambient weather feed sensors."
                ],
                "estimated_impact": "Maintains design COP and extends expected compressor operating life."
            }
            recommendations.append(rec)
            rec_id += 1

    # Sort recommendations by priority (P1 > P2 > P3 > P4)
    priority_order = {"P1 - CRITICAL": 1, "P2 - HIGH": 2, "P3 - MEDIUM": 3, "P4 - ROUTINE": 4}
    recommendations.sort(key=lambda x: priority_order.get(x["priority"], 99))

    return recommendations
