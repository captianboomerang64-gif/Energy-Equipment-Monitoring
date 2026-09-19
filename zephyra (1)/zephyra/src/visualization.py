"""
Visualization Engine for Industrial Intelligent Energy Monitoring.
Generates Plotly interactive figures with an industrial dark theme,
clear severity color encodings, and rich contextual tooltips.
"""

from typing import Dict, Any, Optional
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from src.config import (
    TIMESTAMP_COL,
    EQUIPMENT_COL,
    TARGET_ENERGY_COL,
    BUILDING_LOAD_COL,
    CHILLED_WATER_COL,
    COOLING_WATER_COL,
    OUTSIDE_TEMP_COL
)

# Polished Industrial Color Palette
SEVERITY_COLORS = {
    "NORMAL": "#10B981",    # Emerald green
    "LOW": "#38BDF8",       # Bright sky blue
    "MEDIUM": "#F59E0B",    # Warm amber
    "HIGH": "#F97316",      # Vivid orange
    "CRITICAL": "#EF4444"   # Crimson alert
}

DARK_TEMPLATE = "plotly_dark"
CHART_BG = "#0f172a"        # Deep slate navy
PLOT_BG = "#162032"         # Container slate
GRID_COLOR = "#26334D"      # Subtle grid line
TEXT_COLOR = "#F8FAFC"      # Clean readable white
MUTED_TEXT = "#94A3B8"      # Muted gray text


def plot_anomaly_timeline(
    df: pd.DataFrame,
    equipment_id: Optional[str] = None
) -> go.Figure:
    """
    Plots interactive time series showing:
    - Actual energy consumption curve
    - Contextual expected energy baseline (dashed)
    - Highlighted anomaly points colored by severity tier
    """
    plot_df = df.copy()
    if equipment_id and equipment_id != "ALL":
        plot_df = plot_df[plot_df[EQUIPMENT_COL] == equipment_id]

    plot_df = plot_df.sort_values(by=TIMESTAMP_COL)

    fig = go.Figure()

    # Expected Energy Baseline trace
    if "expected_energy" in plot_df.columns:
        fig.add_trace(go.Scatter(
            x=plot_df[TIMESTAMP_COL],
            y=plot_df["expected_energy"],
            mode="lines",
            name="Expected Energy (ML Baseline)",
            line=dict(color="#818CF8", width=1.8, dash="dash"),
            hovertemplate="<b>Expected Baseline</b>: %{y:.1f} kWh<br><b>Timestamp</b>: %{x}<extra></extra>"
        ))

    # Actual Energy trace
    fig.add_trace(go.Scatter(
        x=plot_df[TIMESTAMP_COL],
        y=plot_df[TARGET_ENERGY_COL],
        mode="lines",
        name="Actual Energy Consumption",
        line=dict(color="#38BDF8", width=2.0),
        hovertemplate="<b>Actual Energy</b>: %{y:.1f} kWh<br><b>Timestamp</b>: %{x}<extra></extra>"
    ))

    # Anomaly points grouped by severity
    anomalies = plot_df[plot_df["anomaly_flag"] == 1]
    for sev in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
        sev_subset = anomalies[anomalies["severity"] == sev]
        if len(sev_subset) > 0:
            hover_text = [
                f"<b style='color:{SEVERITY_COLORS[sev]};'>SEVERITY: {sev}</b><br>"
                f"<b>Equipment</b>: {row[EQUIPMENT_COL]}<br>"
                f"<b>Actual</b>: {row[TARGET_ENERGY_COL]:.1f} kWh<br>"
                f"<b>Expected</b>: {row.get('expected_energy', 0.0):.1f} kWh<br>"
                f"<b>Deviation</b>: {row.get('relative_deviation', 0.0):+.1f}%<br>"
                f"<b>Building Load</b>: {row[BUILDING_LOAD_COL]:.1f} RT<br>"
                f"<b>Cooling Water</b>: {row[COOLING_WATER_COL]:.1f} °C<br>"
                f"<b>ML Anomaly Score</b>: {row.get('anomaly_score', 0.0):.3f}"
                for _, row in sev_subset.iterrows()
            ]

            marker_size = 8 if sev in ["LOW", "MEDIUM"] else 12
            fig.add_trace(go.Scatter(
                x=sev_subset[TIMESTAMP_COL],
                y=sev_subset[TARGET_ENERGY_COL],
                mode="markers",
                name=f"Anomaly ({sev})",
                marker=dict(
                    color=SEVERITY_COLORS[sev],
                    size=marker_size,
                    line=dict(color="#ffffff", width=1.5 if sev in ["HIGH", "CRITICAL"] else 1.0)
                ),
                text=hover_text,
                hoverinfo="text"
            ))

    title_text = f"({equipment_id or 'All Equipment'})"
    fig.update_layout(
        title=dict(text=title_text, font=dict(color=TEXT_COLOR, size=15, family="Outfit, sans-serif")),
        template=DARK_TEMPLATE,
        paper_bgcolor=CHART_BG,
        plot_bgcolor=PLOT_BG,
        hovermode="closest",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            font=dict(color=TEXT_COLOR, size=11)
        ),
        xaxis=dict(
            title="Timestamp",
            gridcolor=GRID_COLOR,
            color=TEXT_COLOR,
            rangeslider=dict(visible=False)
        ),
        yaxis=dict(
            title="Chiller Energy Consumption (kWh)",
            gridcolor=GRID_COLOR,
            color=TEXT_COLOR
        ),
        margin=dict(l=50, r=30, t=50, b=40)
    )

    return fig


def plot_energy_vs_load_scatter(
    df: pd.DataFrame,
    equipment_id: Optional[str] = None
) -> go.Figure:
    """
    Scatter plot: Building Load (RT) vs Chiller Energy Consumption (kWh).
    Highlights anomalies against normal thermodynamic operating envelope.
    """
    plot_df = df.copy()
    if equipment_id and equipment_id != "ALL":
        plot_df = plot_df[plot_df[EQUIPMENT_COL] == equipment_id]

    fig = go.Figure()

    # Normal points
    normal_subset = plot_df[plot_df["anomaly_flag"] == 0]
    fig.add_trace(go.Scatter(
        x=normal_subset[BUILDING_LOAD_COL],
        y=normal_subset[TARGET_ENERGY_COL],
        mode="markers",
        name="Normal Operation",
        marker=dict(
            color="#475569",
            size=5,
            opacity=0.45
        ),
        hovertemplate="<b>Normal Operation</b><br>Load: %{x:.1f} RT<br>Energy: %{y:.1f} kWh<extra></extra>"
    ))

    # Anomalies
    anom_subset = plot_df[plot_df["anomaly_flag"] == 1]
    for sev in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
        sev_data = anom_subset[anom_subset["severity"] == sev]
        if len(sev_data) > 0:
            fig.add_trace(go.Scatter(
                x=sev_data[BUILDING_LOAD_COL],
                y=sev_data[TARGET_ENERGY_COL],
                mode="markers",
                name=f"Anomaly: {sev}",
                marker=dict(
                    color=SEVERITY_COLORS[sev],
                    size=9 if sev in ["LOW", "MEDIUM"] else 12,
                    line=dict(color="#ffffff", width=1.2)
                ),
                hovertemplate=f"<b>Anomaly ({sev})</b><br>Load: %{{x:.1f}} RT<br>Energy: %{{y:.1f}} kWh<br>Deviation: %{{text}}<extra></extra>",
                text=[f"{dev:+.1f}%" for dev in sev_data.get("relative_deviation", 0.0)]
            ))

    fig.update_layout(
        title=dict(text="Thermodynamic Operating Envelope: Building Load vs Energy", font=dict(color=TEXT_COLOR, size=15, family="Outfit, sans-serif")),
        template=DARK_TEMPLATE,
        paper_bgcolor=CHART_BG,
        plot_bgcolor=PLOT_BG,
        xaxis=dict(title="Building Load (RT)", gridcolor=GRID_COLOR, color=TEXT_COLOR),
        yaxis=dict(title="Chiller Energy Consumption (kWh)", gridcolor=GRID_COLOR, color=TEXT_COLOR),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        margin=dict(l=50, r=30, t=50, b=40)
    )

    return fig


def plot_contextual_scatter(
    df: pd.DataFrame,
    x_col: str,
    x_title: str,
    equipment_id: Optional[str] = None
) -> go.Figure:
    """
    Plots Energy Consumption vs an independent contextual driver (Flow, CW Temp, Ambient Temp).
    """
    plot_df = df.copy()
    if equipment_id and equipment_id != "ALL":
        plot_df = plot_df[plot_df[EQUIPMENT_COL] == equipment_id]

    if x_col not in plot_df.columns:
        return go.Figure()

    fig = go.Figure()

    # Normal points
    normal_subset = plot_df[plot_df["anomaly_flag"] == 0]
    fig.add_trace(go.Scatter(
        x=normal_subset[x_col],
        y=normal_subset[TARGET_ENERGY_COL],
        mode="markers",
        name="Normal",
        marker=dict(color="#475569", size=4, opacity=0.45),
        hovertemplate=f"{x_title}: %{{x:.1f}}<br>Energy: %{{y:.1f}} kWh<extra></extra>"
    ))

    # Anomalies
    anom_subset = plot_df[plot_df["anomaly_flag"] == 1]
    if len(anom_subset) > 0:
        fig.add_trace(go.Scatter(
            x=anom_subset[x_col],
            y=anom_subset[TARGET_ENERGY_COL],
            mode="markers",
            name="Anomaly",
            marker=dict(
                color=anom_subset["severity"].map(SEVERITY_COLORS).fillna("#F59E0B"),
                size=8,
                line=dict(color="#ffffff", width=1)
            ),
            hovertemplate=f"<b>Anomaly</b><br>{x_title}: %{{x:.1f}}<br>Energy: %{{y:.1f}} kWh<extra></extra>"
        ))

    fig.update_layout(
        title=dict(text=f"Contextual Analysis: Energy vs {x_title}", font=dict(color=TEXT_COLOR, size=14, family="Outfit, sans-serif")),
        template=DARK_TEMPLATE,
        paper_bgcolor=CHART_BG,
        plot_bgcolor=PLOT_BG,
        xaxis=dict(title=x_title, gridcolor=GRID_COLOR, color=TEXT_COLOR),
        yaxis=dict(title="Energy (kWh)", gridcolor=GRID_COLOR, color=TEXT_COLOR),
        margin=dict(l=40, r=20, t=45, b=35),
        showlegend=False
    )
    return fig


def plot_health_gauge(score: float, equipment_id: str) -> go.Figure:
    """
    Semicircular gauge indicator showing equipment health index (0 to 100).
    """
    status_color = "#10B981" if score >= 85 else ("#38BDF8" if score >= 70 else ("#F59E0B" if score >= 50 else "#EF4444"))
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Health Index: <b>{equipment_id}</b>", 'font': {'size': 15, 'color': TEXT_COLOR, 'family': 'Outfit, sans-serif'}},
        number={'font': {'size': 34, 'color': status_color, 'family': 'JetBrains Mono, monospace'}, 'suffix': "/100"},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': TEXT_COLOR},
            'bar': {'color': status_color, 'thickness': 0.25},
            'bgcolor': PLOT_BG,
            'borderwidth': 1,
            'bordercolor': GRID_COLOR,
            'steps': [
                {'range': [0, 50], 'color': 'rgba(239, 68, 68, 0.25)'},
                {'range': [50, 70], 'color': 'rgba(249, 115, 22, 0.25)'},
                {'range': [70, 85], 'color': 'rgba(56, 189, 248, 0.2)'},
                {'range': [85, 100], 'color': 'rgba(16, 185, 129, 0.25)'}
            ],
            'threshold': {
                'line': {'color': "#FFFFFF", 'width': 3},
                'thickness': 0.75,
                'value': score
            }
        }
    ))

    fig.update_layout(
        template=DARK_TEMPLATE,
        paper_bgcolor=CHART_BG,
        margin=dict(l=25, r=25, t=40, b=15),
        height=210
    )
    return fig

