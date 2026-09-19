"""
Yukthi — Energy & Equipment Monitoring
Streamlit front-end rebuilt to match the reference dashboard design.

The analytical pipeline (src/*) is untouched; only the presentation layer
has been rewritten: dark command-centre shell, icon rail, KPI strip with
sparklines, digital-twin panel, right-hand insight rail.
"""

import os
import math
import html as _html
from datetime import timedelta
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ----------------------------------------------------------------------------
# Internal pipeline modules (unchanged)
# ----------------------------------------------------------------------------
from src.config import (
    TIMESTAMP_COL,
    EQUIPMENT_COL,
    TARGET_ENERGY_COL,
    BUILDING_LOAD_COL,
    CHILLED_WATER_COL,
    COOLING_WATER_COL,
    OUTSIDE_TEMP_COL,
    SEVERITY_LEVELS,
)
from src.data_loader import load_dataset, generate_synthetic_demo_data, generate_quality_report
from src.preprocessing import preprocess_data
from src.feature_engineering import engineer_all_features
from src.baseline_model import ContextualExpectedEnergyModel
from src.anomaly_detection import ContextualAnomalyDetector
from src.severity import compute_anomaly_episodes, assign_severity_levels
from src.explainability import AnomalyExplainer
from src.recommendations import generate_recommendations
from src.health_score import calculate_equipment_health_scores
from src.visualization import (
    plot_anomaly_timeline,
    plot_energy_vs_load_scatter,
    plot_contextual_scatter,
    plot_health_gauge,
    SEVERITY_COLORS,
)

# ----------------------------------------------------------------------------
# Design tokens (sampled from the reference design)
# ----------------------------------------------------------------------------
BG        = "#060b16"
PANEL     = "#0e1626"
PANEL_ALT = "#121d31"
BORDER    = "#1c2841"
TEXT      = "#e8eff9"
MUTED     = "#8a9ab4"
GREEN     = "#34d399"
TEAL      = "#2dd4bf"
BLUE      = "#3b82f6"
CYAN      = "#38bdf8"
PURPLE    = "#a78bfa"
AMBER     = "#f59e0b"
RED       = "#ef4444"

STATUS_TONE = {
    "CRITICAL": RED,
    "HIGH": RED,
    "WARNING": AMBER,
    "MEDIUM": AMBER,
    "HEALTHY": GREEN,
    "NORMAL": GREEN,
    "LOW": GREEN,
    "GOOD": GREEN,
}

st.set_page_config(
    page_title="Yukthi | Energy & Equipment Monitoring",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ----------------------------------------------------------------------------
# Global stylesheet
# ----------------------------------------------------------------------------
def inject_css() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root{
  --bg:#060b16; --panel:#0e1626; --panel2:#121d31; --line:#1c2841;
  --txt:#e8eff9; --mut:#8a9ab4; --grn:#34d399; --tel:#2dd4bf; --blu:#3b82f6;
  --amb:#f59e0b; --red:#ef4444; --pur:#a78bfa;
}

html, body, .stApp, [data-testid="stAppViewContainer"]{
  background:var(--bg);
  color:var(--txt);
  font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
}
[data-testid="stHeader"]{background:transparent; height:0;}
#MainMenu, footer, [data-testid="stToolbar"]{visibility:hidden;}
.block-container{padding:1.1rem 1.4rem 2.2rem 1.4rem; max-width:100%;}

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"]{width:248px !important; min-width:248px !important;}
section[data-testid="stSidebar"] > div{
  background:linear-gradient(180deg,#081221 0%,#050b15 62%,#071a14 100%);
  border-right:1px solid var(--line);
  padding-top:0;
}
section[data-testid="stSidebar"] .block-container{padding:0;}

.brand{display:flex; align-items:center; gap:.6rem; padding:1.15rem 1rem 1.35rem 1.05rem;}
.brand-mark{
  width:34px;height:34px;border-radius:10px;display:grid;place-items:center;
  background:linear-gradient(145deg,rgba(52,211,153,.22),rgba(45,212,191,.08));
  border:1px solid rgba(52,211,153,.35);
}
.brand-name{font-size:1.45rem;font-weight:700;letter-spacing:-.02em;color:#f2f7ff;line-height:1;}

/* nav radio -> nav rail */
section[data-testid="stSidebar"] div[role="radiogroup"]{gap:.15rem; padding:0 .7rem;}
section[data-testid="stSidebar"] div[role="radiogroup"] > label{
  display:flex; align-items:center; gap:.65rem;
  padding:.62rem .75rem; border-radius:11px; cursor:pointer;
  border:1px solid transparent; transition:background .15s ease, color .15s ease;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child{display:none;}
section[data-testid="stSidebar"] div[role="radiogroup"] > label p{
  font-size:.92rem; font-weight:500; color:var(--mut); margin:0;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover{background:#101d31;}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked){
  background:linear-gradient(90deg,rgba(52,211,153,.20),rgba(52,211,153,.05));
  border-color:rgba(52,211,153,.30);
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) p{
  color:#eafff6; font-weight:600;
}
.side-foot{
  margin-top:auto; padding:1.1rem 1.15rem 1.4rem;
  border-top:1px solid rgba(255,255,255,.05);
}
.side-foot h4{font-size:1.02rem;line-height:1.35;font-weight:600;color:#d7ecdf;margin:0 0 .55rem;}
.side-foot span{display:block;width:34px;height:2px;background:var(--grn);border-radius:2px;}
.side-foot .reset-label{font-size:.72rem;color:var(--mut);text-align:center;margin-top:.55rem;}

/* ---------- header ---------- */
.topbar{display:flex; align-items:flex-start; justify-content:space-between; margin-bottom:1.05rem;}
.topbar h1{font-size:1.45rem; font-weight:700; letter-spacing:-.02em; margin:0 0 .3rem; color:#f4f8ff;}
.topbar .crumbs{display:flex; align-items:center; gap:.55rem; color:var(--mut); font-size:.82rem;}
.topbar .crumbs i{width:4px;height:4px;border-radius:50%;background:var(--grn);opacity:.7;display:inline-block;}
.topmeta{display:flex; align-items:center; gap:.85rem;}
.pill-live{
  display:flex;align-items:center;gap:.45rem;padding:.42rem .8rem;border-radius:999px;
  background:rgba(52,211,153,.10);border:1px solid rgba(52,211,153,.30);
  font-size:.8rem;font-weight:600;color:#7ee7be;
}
.dot-live{width:7px;height:7px;border-radius:50%;background:var(--grn);box-shadow:0 0 0 3px rgba(52,211,153,.18);}
.stamp{font-size:.83rem;color:var(--mut);font-variant-numeric:tabular-nums;}
.avatar{
  width:32px;height:32px;border-radius:50%;display:grid;place-items:center;
  background:var(--panel2);border:1px solid var(--line);color:var(--mut);font-size:.9rem;
}

/* ---------- cards ---------- */
.card{
  background:linear-gradient(180deg,var(--panel) 0%,#0b1322 100%);
  border:1px solid var(--line); border-radius:14px; padding:1rem 1.05rem;
}
.card-head{display:flex;align-items:center;justify-content:space-between;margin-bottom:.85rem;}
.card-title{display:flex;align-items:center;gap:.5rem;font-size:.95rem;font-weight:600;color:#eaf1fb;}
.card-link{font-size:.76rem;color:var(--mut);}

/* KPI */
.kpi{
  position:relative; overflow:hidden; border-radius:14px; padding:.95rem 1rem .6rem;
  border:1px solid var(--line); min-height:132px;
}
.kpi-top{display:flex;align-items:center;gap:.6rem;margin-bottom:.55rem;}
.kpi-ico{width:36px;height:36px;border-radius:10px;display:grid;place-items:center;}
.kpi-label{font-size:.83rem;color:#c3d0e4;font-weight:500;}
.kpi-val{font-size:1.9rem;font-weight:700;letter-spacing:-.03em;line-height:1;color:#fff;}
.kpi-val small{font-size:.9rem;font-weight:500;color:var(--mut);margin-left:.25rem;letter-spacing:0;}
.kpi-foot{display:flex;align-items:flex-end;justify-content:space-between;margin-top:.5rem;}
.kpi-delta{font-size:.78rem;font-weight:600;}
.kpi-note{font-size:.72rem;color:var(--mut);display:block;font-weight:400;}

/* equipment rows */
.eq{
  display:flex;align-items:center;gap:.8rem;padding:.75rem .8rem;border-radius:12px;
  border:1px solid var(--line); margin-bottom:.55rem; background:#0c1424;
}
.eq-ico{width:40px;height:40px;border-radius:10px;display:grid;place-items:center;font-size:1.05rem;}
.eq-body{flex:1;min-width:0;}
.eq-name{font-size:.92rem;font-weight:600;color:#eef4ff;}
.eq-state{display:flex;align-items:center;gap:.35rem;font-size:.76rem;font-weight:500;margin-top:.1rem;}
.eq-stats{display:flex;gap:1.15rem;margin-top:.42rem;}
.eq-stats div{font-size:.72rem;color:var(--mut);}
.eq-stats b{color:#d7e2f2;font-weight:600;margin-left:.3rem;}
.eq-ring{text-align:center;}
.eq-ring p{font-size:.66rem;color:var(--mut);margin:.15rem 0 0;}

/* digital twin */
.twin{position:relative;}
.chip{
  position:absolute; padding:.4rem .65rem; border-radius:9px; background:rgba(9,15,26,.92);
  border:1px solid var(--line); backdrop-filter:blur(6px); z-index:3;
}
.chip span{display:block;font-size:.66rem;color:var(--mut);line-height:1.2;}
.chip b{font-size:.9rem;font-weight:700;letter-spacing:-.01em;}
.comp-row{display:flex;align-items:center;gap:.6rem;margin-bottom:.85rem;}
.comp-ico{width:26px;height:26px;border-radius:7px;display:grid;place-items:center;background:#141f33;font-size:.8rem;}
.comp-bar{height:5px;border-radius:3px;background:#1a2640;margin-top:.35rem;overflow:hidden;}
.comp-bar i{display:block;height:100%;border-radius:3px;}
.comp-name{display:flex;justify-content:space-between;font-size:.76rem;color:#c8d5e8;}
.comp-name b{font-weight:600;}

/* alerts */
.alert{display:flex;gap:.65rem;padding:.7rem .75rem;border-radius:11px;margin-bottom:.5rem;}
.alert-ico{width:26px;height:26px;border-radius:8px;display:grid;place-items:center;font-size:.75rem;flex:none;}
.alert b{font-size:.82rem;font-weight:600;}
.alert p{margin:.15rem 0 0;font-size:.74rem;color:var(--mut);}
.alert .ago{margin-left:auto;font-size:.68rem;color:#6d7d96;white-space:nowrap;}

/* insight */
.insight-flag{
  border-radius:11px;padding:.7rem .8rem;font-size:.8rem;line-height:1.4;
  background:rgba(239,68,68,.09);border:1px solid rgba(239,68,68,.28);color:#ffb4b4;
  display:flex;gap:.55rem;
}
.insight-list{margin:.7rem 0 .9rem;padding-left:1.05rem;color:#c2d0e4;font-size:.78rem;line-height:1.75;}
.action{
  border-radius:11px;padding:.75rem .8rem;display:flex;gap:.6rem;
  background:rgba(52,211,153,.08);border:1px solid rgba(52,211,153,.26);
}
.action b{font-size:.8rem;color:#8ff0c6;font-weight:600;}
.action p{margin:.2rem 0 0;font-size:.76rem;color:#bdd8cd;line-height:1.45;}

.legend{display:flex;justify-content:space-between;font-size:.78rem;padding:.3rem 0;color:#c4d2e6;}
.legend i{width:9px;height:9px;border-radius:3px;display:inline-block;margin-right:.5rem;}
.legend b{color:#eaf1fb;font-weight:600;}

/* ---------- streamlit widgets ---------- */
.stTabs [data-baseweb="tab-list"]{
  gap:.2rem; background:var(--panel); padding:.35rem; border-radius:13px;
  border:1px solid var(--line); margin-bottom:.9rem;
}
.stTabs [data-baseweb="tab"]{
  height:40px; border-radius:10px; padding:0 1.05rem; color:var(--mut);
  font-size:.86rem; font-weight:500;
}
.stTabs [data-baseweb="tab"]:hover{background:#111d31; color:#cfdcee;}
.stTabs [aria-selected="true"]{
  background:linear-gradient(90deg,rgba(52,211,153,.20),rgba(52,211,153,.06)) !important;
  color:#eafff6 !important; font-weight:600;
}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"]{display:none;}

div[data-testid="stSelectbox"] > div > div,
div[data-testid="stDateInput"] input,
div[data-testid="stDateInput"] > div > div{
  background:#0b1322 !important; border-color:var(--line) !important;
  color:var(--txt) !important; border-radius:9px !important; font-size:.84rem !important;
}
label, .stSlider label, div[data-testid="stWidgetLabel"] p{
  color:var(--mut) !important; font-size:.76rem !important; font-weight:500 !important;
}
div[data-testid="stDataFrame"]{border:1px solid var(--line); border-radius:12px;}
.stSlider [data-baseweb="slider"] div[role="slider"]{background:var(--grn) !important;}
.js-plotly-plot .plotly .modebar{display:none !important;}
.stAlert{background:#101b2c; border:1px solid var(--line); border-radius:11px; color:#c8d6ea;}
hr{border-color:var(--line);}

/* Clear / delete button — danger styling */
section[data-testid="stSidebar"] button[kind="secondary"]{
  border-color:#ef444466 !important; color:#ef4444 !important;
  background:rgba(239,68,68,.07) !important; font-size:.82rem !important;
  border-radius:9px !important; transition:background .15s ease,border-color .15s ease;
}
section[data-testid="stSidebar"] button[kind="secondary"]:hover{
  background:rgba(239,68,68,.16) !important; border-color:#ef4444aa !important;
}
</style>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# Small HTML/SVG builders
# ----------------------------------------------------------------------------
def esc(value: Any) -> str:
    return _html.escape(str(value))


def hex_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def sparkline(values: List[float], color: str, width: int = 155, height: int = 40) -> str:
    """Inline SVG sparkline with a soft area fill."""
    series = [float(v) for v in values if v is not None and not pd.isna(v)]
    if len(series) < 2:
        series = [0.0, 0.0]
    if len(series) > 60:
        step = len(series) / 60.0
        series = [series[int(i * step)] for i in range(60)]
    lo, hi = min(series), max(series)
    span = (hi - lo) or 1.0
    n = len(series)
    pts = [
        (i * (width / (n - 1)), height - 3 - ((v - lo) / span) * (height - 11))
        for i, v in enumerate(series)
    ]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    area = f"0,{height} {line} {width},{height}"
    gid = f"sp{abs(hash((color, n, round(series[0], 3)))) % 100000}"
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'preserveAspectRatio="none" style="display:block">'
        f'<defs><linearGradient id="{gid}" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0%" stop-color="{color}" stop-opacity=".34"/>'
        f'<stop offset="100%" stop-color="{color}" stop-opacity="0"/></linearGradient></defs>'
        f'<polygon points="{area}" fill="url(#{gid})"/>'
        f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="1.8" '
        f'stroke-linejoin="round" stroke-linecap="round"/></svg>'
    )


def ring(pct: float, color: str, size: int = 50, stroke: int = 5) -> str:
    """Circular progress ring with the value in the middle."""
    pct = max(0.0, min(100.0, float(pct)))
    r = (size - stroke) / 2.0
    circ = 2 * math.pi * r
    dash = circ * pct / 100.0
    c = size / 2.0
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">'
        f'<circle cx="{c}" cy="{c}" r="{r}" fill="none" stroke="#1a2640" stroke-width="{stroke}"/>'
        f'<circle cx="{c}" cy="{c}" r="{r}" fill="none" stroke="{color}" stroke-width="{stroke}" '
        f'stroke-linecap="round" stroke-dasharray="{dash:.1f} {circ:.1f}" '
        f'transform="rotate(-90 {c} {c})"/>'
        f'<text x="{c}" y="{c + 4.5}" text-anchor="middle" font-size="{size * 0.27:.0f}" '
        f'font-weight="700" fill="{color}" font-family="Inter,sans-serif">{pct:.0f}%</text></svg>'
    )


def kpi_card(label: str, value: str, unit: str, delta: Optional[float], note: str,
             color: str, icon: str, series: List[float]) -> str:
    if delta is None:
        delta_html = f'<span class="kpi-delta" style="color:{MUTED};">—<span class="kpi-note">{esc(note)}</span></span>'
    else:
        up = delta >= 0
        arrow = "↑" if up else "↓"
        tone = GREEN if up else RED
        delta_html = (
            f'<span class="kpi-delta" style="color:{tone};">{arrow} {abs(delta):.1f}%'
            f'<span class="kpi-note">{esc(note)}</span></span>'
        )
    return f"""
<div class="kpi" style="background:linear-gradient(140deg,{hex_rgba(color,.16)} 0%,#0b1322 58%);
     border-color:{hex_rgba(color,.26)};">
  <div class="kpi-top">
    <div class="kpi-ico" style="background:{hex_rgba(color,.16)};border:1px solid {hex_rgba(color,.3)};">{icon}</div>
    <div class="kpi-label">{esc(label)}</div>
  </div>
  <div class="kpi-val">{esc(value)}<small>{esc(unit)}</small></div>
  <div class="kpi-foot">
    <div>{delta_html}</div>
    <div style="width:118px;opacity:.95;">{sparkline(series, color, width=118, height=34)}</div>
  </div>
</div>"""


def equipment_row(name: str, status: str, load: float, temp: float, health: float, tone: str) -> str:
    return f"""
<div class="eq" style="border-color:{hex_rgba(tone,.34)};background:linear-gradient(90deg,{hex_rgba(tone,.10)},#0c1424 62%);">
  <div class="eq-ico" style="background:{hex_rgba(tone,.15)};border:1px solid {hex_rgba(tone,.3)};">🧊</div>
  <div class="eq-body">
    <div class="eq-name">{esc(name)}</div>
    <div class="eq-state" style="color:{tone};">
      <span style="width:6px;height:6px;border-radius:50%;background:{tone};display:inline-block;"></span>{esc(status)}
    </div>
    <div class="eq-stats">
      <div>Load<b>{load:.0f}%</b></div>
      <div>Temp<b>{temp:.1f} °C</b></div>
    </div>
  </div>
  <div class="eq-ring">{ring(health, tone, size=48)}<p>Health</p></div>
</div>"""


def chiller_svg() -> str:
    """Schematic chiller illustration used in the digital-twin panel."""
    return """
<svg viewBox="0 0 520 300" width="100%" height="270" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="shell" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#4b7fc7"/><stop offset="55%" stop-color="#27508c"/>
      <stop offset="100%" stop-color="#132c50"/>
    </linearGradient>
    <radialGradient id="glow" cx="50%" cy="60%" r="60%">
      <stop offset="0%" stop-color="#1d4ed8" stop-opacity=".30"/>
      <stop offset="100%" stop-color="#1d4ed8" stop-opacity="0"/>
    </radialGradient>
    <linearGradient id="deck" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#2dd4bf" stop-opacity="0"/>
      <stop offset="50%" stop-color="#2dd4bf" stop-opacity=".55"/>
      <stop offset="100%" stop-color="#2dd4bf" stop-opacity="0"/>
    </linearGradient>
  </defs>

  <ellipse cx="260" cy="180" rx="230" ry="95" fill="url(#glow)"/>
  <ellipse cx="260" cy="248" rx="185" ry="26" fill="none" stroke="url(#deck)" stroke-width="1.4"/>
  <ellipse cx="260" cy="248" rx="120" ry="17" fill="none" stroke="#2dd4bf" stroke-opacity=".18" stroke-width="1"/>

  <!-- lower vessel (evaporator) -->
  <rect x="92" y="186" width="336" height="52" rx="26" fill="url(#shell)" stroke="#5b8fd6" stroke-width="1.2"/>
  <ellipse cx="92" cy="212" rx="15" ry="26" fill="#1b3a68" stroke="#5b8fd6" stroke-width="1.2"/>
  <ellipse cx="428" cy="212" rx="15" ry="26" fill="#2e5d9e" stroke="#5b8fd6" stroke-width="1.2"/>

  <!-- upper vessel (condenser) -->
  <rect x="112" y="124" width="296" height="48" rx="24" fill="url(#shell)" stroke="#5b8fd6" stroke-width="1.2"/>
  <ellipse cx="112" cy="148" rx="14" ry="24" fill="#1b3a68" stroke="#5b8fd6" stroke-width="1.2"/>
  <ellipse cx="408" cy="148" rx="14" ry="24" fill="#2e5d9e" stroke="#5b8fd6" stroke-width="1.2"/>

  <!-- compressor block -->
  <rect x="214" y="70" width="92" height="58" rx="10" fill="#274d84" stroke="#6ba0e0" stroke-width="1.2"/>
  <rect x="226" y="80" width="68" height="30" rx="5" fill="#0d2242" stroke="#4d81c4" stroke-width=".9"/>
  <circle cx="240" cy="95" r="4" fill="#34d399"/><circle cx="254" cy="95" r="4" fill="#38bdf8"/>
  <rect x="234" y="103" width="52" height="3" rx="1.5" fill="#1f4676"/>

  <!-- motor barrel -->
  <rect x="306" y="82" width="74" height="42" rx="21" fill="#2b5590" stroke="#6ba0e0" stroke-width="1.1"/>
  <ellipse cx="380" cy="103" rx="9" ry="21" fill="#3c6eb0" stroke="#6ba0e0" stroke-width="1.1"/>
  <line x1="322" y1="86" x2="322" y2="120" stroke="#4d81c4" stroke-width="1"/>
  <line x1="336" y1="86" x2="336" y2="120" stroke="#4d81c4" stroke-width="1"/>
  <line x1="350" y1="86" x2="350" y2="120" stroke="#4d81c4" stroke-width="1"/>

  <!-- piping -->
  <path d="M150 124 L150 96 Q150 84 162 84 L214 84" fill="none" stroke="#3e6fae" stroke-width="7" stroke-linecap="round"/>
  <path d="M120 186 L120 160 Q120 150 132 150" fill="none" stroke="#3e6fae" stroke-width="6" stroke-linecap="round"/>
  <path d="M396 128 L396 170 Q396 186 380 186" fill="none" stroke="#3e6fae" stroke-width="6" stroke-linecap="round"/>

  <!-- control panel -->
  <rect x="150" y="196" width="46" height="34" rx="6" fill="#0d2242" stroke="#4d81c4" stroke-width="1"/>
  <rect x="157" y="203" width="32" height="12" rx="2" fill="#123a63"/>
  <circle cx="160" cy="223" r="3" fill="#34d399"/><circle cx="171" cy="223" r="3" fill="#f59e0b"/>

  <!-- sensor nodes -->
  <circle cx="264" cy="70" r="5" fill="#ef4444"><animate attributeName="r" values="5;7;5" dur="2.4s" repeatCount="indefinite"/></circle>
  <circle cx="264" cy="70" r="10" fill="none" stroke="#ef4444" stroke-opacity=".35"/>
  <circle cx="344" cy="124" r="4.5" fill="#a78bfa"/>
  <circle cx="344" cy="124" r="9" fill="none" stroke="#a78bfa" stroke-opacity=".3"/>
  <circle cx="300" cy="212" r="4.5" fill="#f59e0b"/>
  <circle cx="300" cy="212" r="9" fill="none" stroke="#f59e0b" stroke-opacity=".3"/>
  <circle cx="180" cy="186" r="4.5" fill="#38bdf8"/>
</svg>"""


# ----------------------------------------------------------------------------
# Plotly theming
# ----------------------------------------------------------------------------
def theme_fig(fig: go.Figure, height: int = 300) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#9fb0c8", size=11),
        margin=dict(l=8, r=8, t=26, b=8),
        height=height,
        legend=dict(orientation="h", y=1.14, x=0, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=10, color="#9fb0c8")),
        hoverlabel=dict(bgcolor="#0e1626", bordercolor=BORDER,
                        font=dict(color=TEXT, family="Inter, sans-serif")),
        title=None,
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,.05)", zerolinecolor="rgba(255,255,255,.07)",
                     linecolor="rgba(255,255,255,.08)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,.05)", zerolinecolor="rgba(255,255,255,.07)",
                     linecolor="rgba(255,255,255,.08)")
    return fig


def donut_energy(df: pd.DataFrame) -> go.Figure:
    by_eq = df.groupby(EQUIPMENT_COL)[TARGET_ENERGY_COL].sum().sort_values(ascending=False)
    total = float(by_eq.sum())
    palette = [GREEN, PURPLE, AMBER, CYAN, BLUE, RED]
    fig = go.Figure(
        go.Pie(
            labels=list(by_eq.index),
            values=list(by_eq.values),
            hole=0.68,
            sort=False,
            marker=dict(colors=palette[: len(by_eq)], line=dict(color="#0b1322", width=2)),
            textinfo="none",
            hovertemplate="%{label}: %{value:,.0f} kWh<extra></extra>",
        )
    )
    label = f"{total:,.0f}" if total < 100000 else f"{total/1000:,.1f}k"
    fig.add_annotation(text=f"<span style='font-size:11px;color:{MUTED}'>Total</span><br>"
                            f"<b style='font-size:21px'>{label}</b><br>"
                            f"<span style='font-size:10px'>kWh</span>",
                       showarrow=False, font=dict(color=TEXT))
    fig.update_layout(showlegend=False)
    return theme_fig(fig, height=220)


# ----------------------------------------------------------------------------
# Derived metrics
# ----------------------------------------------------------------------------
def pct_delta(series: pd.Series, timestamps: pd.Series) -> Optional[float]:
    """Change of the last 24 h window versus the preceding 24 h."""
    try:
        if series.empty:
            return None
        end = timestamps.max()
        cur = series[timestamps > end - timedelta(hours=24)]
        prev = series[(timestamps <= end - timedelta(hours=24)) &
                      (timestamps > end - timedelta(hours=48))]
        if len(cur) == 0 or len(prev) == 0 or float(prev.mean()) == 0:
            return None
        return (float(cur.mean()) - float(prev.mean())) / abs(float(prev.mean())) * 100.0
    except Exception:
        return None


def trend_series(df: pd.DataFrame, col: str, points: int = 48) -> List[float]:
    try:
        s = (df.set_index(TIMESTAMP_COL)[col]
               .resample("h").mean().dropna().tail(points))
        return s.tolist() if len(s) > 1 else [0, 0]
    except Exception:
        return [0, 0]


def status_tone(status: str) -> str:
    return STATUS_TONE.get(str(status).strip().upper(), AMBER)


def unit_snapshot(df: pd.DataFrame, equipment: str) -> Dict[str, float]:
    """Load %, temperature and efficiency proxy for one unit."""
    sub = df[df[EQUIPMENT_COL] == equipment]
    if sub.empty:
        return {"load": 0.0, "temp": 0.0, "eff": 0.0}
    load_raw = float(sub[BUILDING_LOAD_COL].tail(48).mean())
    load_max = float(df[BUILDING_LOAD_COL].quantile(0.99)) or 1.0
    load_pct = max(0.0, min(100.0, load_raw / load_max * 100.0))
    temp_col = COOLING_WATER_COL if COOLING_WATER_COL in sub.columns else OUTSIDE_TEMP_COL
    temp = float(sub[temp_col].tail(48).mean())
    resid = float(sub["energy_residual"].tail(48).mean()) if "energy_residual" in sub else 0.0
    base = float(sub[TARGET_ENERGY_COL].tail(48).mean()) or 1.0
    eff = max(0.0, min(100.0, 100.0 - abs(resid) / base * 100.0))
    return {"load": load_pct, "temp": temp, "eff": eff}


def component_health(df: pd.DataFrame, equipment: str, health: float) -> Dict[str, float]:
    """
    Sub-component health proxies derived from the unit's own telemetry
    deviation against the fleet baseline — not random placeholders.
    """
    sub = df[df[EQUIPMENT_COL] == equipment]
    if sub.empty:
        return {"Compressor": health, "Motor": health, "Condenser": health, "Evaporator": health}

    def dev(col: str) -> float:
        if col not in df.columns:
            return 0.0
        fleet_mu, fleet_sd = float(df[col].mean()), float(df[col].std()) or 1.0
        return abs(float(sub[col].tail(96).mean()) - fleet_mu) / fleet_sd

    def score(d: float, weight: float) -> float:
        return round(max(32.0, min(99.0, health * (1 - weight * min(d, 2.4) / 4.0) + 6.0)))

    return {
        "Compressor": score(dev(TARGET_ENERGY_COL), 0.35),
        "Motor": score(dev(BUILDING_LOAD_COL), 0.30),
        "Condenser": score(dev(COOLING_WATER_COL), 0.75),
        "Evaporator": score(dev(CHILLED_WATER_COL), 0.28),
    }


def recent_alerts(df: pd.DataFrame, limit: int = 3) -> List[Dict[str, str]]:
    if df.empty or "severity" not in df.columns:
        return []
    ranked = df[df["anomaly_flag"] == 1].copy()
    if ranked.empty:
        return []
    order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    ranked["_r"] = ranked["severity"].map(order).fillna(4)
    ranked = ranked.sort_values(["_r", TIMESTAMP_COL], ascending=[True, False]).head(limit)
    latest = df[TIMESTAMP_COL].max()
    out = []
    for _, row in ranked.iterrows():
        hours = max(1, int((latest - row[TIMESTAMP_COL]).total_seconds() // 3600))
        ago = f"{hours}h ago" if hours < 48 else f"{hours // 24}d ago"
        sev = str(row.get("severity", "MEDIUM")).upper()
        temp_col = COOLING_WATER_COL if COOLING_WATER_COL in df.columns else OUTSIDE_TEMP_COL
        out.append({
            "equipment": str(row[EQUIPMENT_COL]),
            "metric": f"{float(row[temp_col]):.1f} °C",
            "reason": "Energy above contextual baseline" if float(row.get("energy_residual", 0)) > 0
                      else "Energy below contextual baseline",
            "severity": sev,
            "ago": ago,
        })
    return out


# ----------------------------------------------------------------------------
# Pipeline (unchanged logic, cached)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def execute_pipeline(raw_df: pd.DataFrame, contamination: float):
    df_clean, gaps, prep_summary = preprocess_data(raw_df)
    df_featured, feature_cols = engineer_all_features(df_clean)

    regressor = ContextualExpectedEnergyModel()
    df_expected, reg_metrics = regressor.fit_predict(df_featured)

    detector = ContextualAnomalyDetector(contamination=contamination)
    df_anom, det_metrics = detector.fit_predict(df_expected)

    df_episodes, episodes = compute_anomaly_episodes(df_anom)
    df_severity = assign_severity_levels(df_episodes, episodes)
    health_scores = calculate_equipment_health_scores(df_severity, episodes)

    explainer = AnomalyExplainer()
    explainer.fit_baselines(df_severity)

    recs = generate_recommendations(df_severity, episodes, health_scores)

    return {
        "df": df_severity,
        "gaps": gaps,
        "prep_summary": prep_summary,
        "feature_cols": feature_cols,
        "reg_metrics": reg_metrics,
        "det_metrics": det_metrics,
        "episodes": episodes,
        "health_scores": health_scores,
        "explainer": explainer,
        "recommendations": recs,
    }


def load_telemetry():
    """Sidebar data source controls -> raw dataframe."""
    with st.sidebar.expander("Telemetry source", expanded=False):
        # Key counter lets us remount the file_uploader to clear it
        if "uploader_key" not in st.session_state:
            st.session_state["uploader_key"] = 0

        uploaded_file = st.file_uploader(
            "Upload CSV", type=["csv"],
            help="CSV conforming to the 11-field chiller schema",
            key=f"csv_upload_{st.session_state['uploader_key']}",
        )
        contamination = st.slider("Anomaly sensitivity", 0.01, 0.15, 0.05, 0.01)

        is_demo = False
        loaded_df, quality_report = None, {}

        if uploaded_file is not None:
            file_df, file_report, errors = load_dataset(uploaded_file)
            if errors:
                st.error("That CSV could not be read. Using the default dataset instead.")
            else:
                loaded_df, quality_report = file_df, file_report
                st.success(f"Loaded {uploaded_file.name}")

            # ── Delete / Clear button ────────────────────────────────────────
            st.markdown(
                """<style>
                div[data-testid="stButton"] button[kind="secondary"].clear-btn {
                    border-color: #ef4444 !important;
                    color: #ef4444 !important;
                }
                </style>""",
                unsafe_allow_html=True,
            )
            if st.button(
                "🗑  Clear data",
                help="Remove the uploaded CSV and reset the dashboard",
                use_container_width=True,
                type="secondary",
            ):
                st.session_state["uploader_key"] += 1   # remounts the file_uploader
                execute_pipeline.clear()                 # wipe cached pipeline results
                st.rerun()

        if loaded_df is None:
            mode = st.radio("Fallback source", ["Default dataset", "Synthetic demo data"], index=0)
            if mode == "Synthetic demo data":
                is_demo = True
                days = st.slider("History (days)", 15, 60, 30, 5)
                loaded_df = generate_synthetic_demo_data(num_days=days)
                quality_report = generate_quality_report(loaded_df)
            else:
                sample_path = os.path.join("data", "sample", "development_dataset.csv")
                if os.path.exists(sample_path):
                    loaded_df, quality_report, _ = load_dataset(sample_path)
                else:
                    is_demo = True
                    loaded_df = generate_synthetic_demo_data(num_days=30)
                    quality_report = generate_quality_report(loaded_df)

    return loaded_df, quality_report, is_demo, contamination


# ----------------------------------------------------------------------------
# Layout blocks
# ----------------------------------------------------------------------------
NAV_ITEMS = ["🏠  Dashboard", "🧊  Equipment", "⚡  Energy Analytics",
             "⚠️  Anomalies", "🧠  AI Diagnosis", "📄  Reports", "⚙️  Settings"]


def render_sidebar_brand() -> None:
    st.sidebar.markdown(
        """
<div class="brand">
  <div class="brand-mark">
    <svg width="19" height="19" viewBox="0 0 24 24" fill="none">
      <path d="M12 21c0-6 3-10 8-12-1 8-4 11-8 12Z" fill="#34d399"/>
      <path d="M12 21C9 16 5 14 3 8c7 1 9 6 9 13Z" fill="#2dd4bf" opacity=".75"/>
    </svg>
  </div>
  <div class="brand-name">yukthi</div>
</div>""",
        unsafe_allow_html=True,
    )


def render_sidebar_footer() -> None:
    st.sidebar.markdown(
        """<div class="side-foot"><span></span></div>""",
        unsafe_allow_html=True,
    )
    # ── Full reset button ────────────────────────────────────────────────────
    with st.sidebar:
        if st.button(
            "🗑  Reset session",
            help="Clear all session data, uploaded files and cached results — start fresh",
            use_container_width=True,
            type="secondary",
            key="global_reset_btn",
        ):
            # Wipe every session-state key
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            # Clear the analysis pipeline cache
            execute_pipeline.clear()
            st.rerun()
        st.markdown(
            '<p style="font-size:.7rem;color:#4a5a72;text-align:center;margin-top:.3rem;">'
            'Clears data, cache &amp; filters</p>',
            unsafe_allow_html=True,
        )



def render_topbar(df: pd.DataFrame) -> None:
    stamp = df[TIMESTAMP_COL].max()
    st.markdown(
        f"""
<div class="topbar">
  <div>
    <h1>Energy &amp; Equipment Monitoring</h1>
    <div class="crumbs"><span>Smarter insights</span><i></i><span>Healthier assets</span>
      <i></i><span>Higher efficiency</span></div>
  </div>
  <div class="topmeta">
    <div class="pill-live"><span class="dot-live"></span>System online</div>
    <div class="stamp">{stamp.strftime('%d %b %Y')}</div>
    <div class="stamp">{stamp.strftime('%H:%M')}</div>
    <div class="avatar">👤</div>
  </div>
</div>""",
        unsafe_allow_html=True,
    )


def render_kpis(view: pd.DataFrame, health_scores: Dict[str, Any], units: List[str]) -> None:
    obs = len(view)
    anomalies = int((view["anomaly_flag"] == 1).sum()) if obs else 0
    anom_rate = (anomalies / obs * 100.0) if obs else 0.0
    current_load = float(view[TARGET_ENERGY_COL].tail(4).mean()) if obs else 0.0
    fleet_health = float(np.mean([h["health_score"] for h in health_scores.values()])) if health_scores else 100.0

    resid = view["energy_residual"] if "energy_residual" in view else pd.Series([0.0])
    base = float(view[TARGET_ENERGY_COL].mean()) or 1.0
    efficiency = max(0.0, min(100.0, 100.0 - float(resid.abs().mean()) / base * 100.0))
    uptime = max(0.0, 100.0 - anom_rate * 0.42)

    d_energy = pct_delta(view[TARGET_ENERGY_COL], view[TIMESTAMP_COL]) if obs else None
    d_anom = pct_delta(view["anomaly_flag"], view[TIMESTAMP_COL]) if obs else None

    spark_energy = trend_series(view, TARGET_ENERGY_COL)
    spark_load = trend_series(view, BUILDING_LOAD_COL)
    spark_resid = trend_series(view, "energy_residual") if "energy_residual" in view else [0, 0]

    icons = {
        "bolt": '<svg width="17" height="17" viewBox="0 0 24 24" fill="#34d399"><path d="M13 2 4 14h6l-1 8 9-12h-6l1-8Z"/></svg>',
        "gauge": '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#a78bfa" stroke-width="2"><circle cx="12" cy="12" r="9"/><path d="M12 12l4-3" stroke-linecap="round"/></svg>',
        "heart": '<svg width="17" height="17" viewBox="0 0 24 24" fill="#2dd4bf"><path d="M12 21s-8-4.7-8-10a4.6 4.6 0 0 1 8-3 4.6 4.6 0 0 1 8 3c0 5.3-8 10-8 10Z"/></svg>',
        "warn": '<svg width="17" height="17" viewBox="0 0 24 24" fill="#f59e0b"><path d="M12 3 1 21h22L12 3Zm0 6v6m0 3v.5" stroke="#0b1322" stroke-width="1.6" stroke-linecap="round"/></svg>',
        "clock": '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#3b82f6" stroke-width="2"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2" stroke-linecap="round"/></svg>',
    }

    cols = st.columns(5, gap="small")
    cards = [
        kpi_card("Current energy load", f"{current_load:,.1f}", "kWh", d_energy,
                 "vs last 24h", GREEN, icons["bolt"], spark_energy),
        kpi_card("System efficiency", f"{efficiency:.0f}", "%", None,
                 "residual vs baseline", PURPLE, icons["gauge"], spark_resid),
        kpi_card("Fleet health", f"{fleet_health:.0f}", "%", None,
                 f"{len(units)} units monitored", TEAL, icons["heart"], spark_load),
        kpi_card("Active anomalies", f"{anomalies:,}", "", d_anom,
                 f"{anom_rate:.1f}% of cycles", AMBER, icons["warn"], spark_energy),
        kpi_card("Uptime", f"{uptime:.1f}", "%", None,
                 "estimated availability", BLUE, icons["clock"], spark_load),
    ]
    for col, card in zip(cols, cards):
        col.markdown(card, unsafe_allow_html=True)


def render_equipment_overview(slot, view: pd.DataFrame, health_scores: Dict[str, Any],
                              units: List[str]) -> None:
    rows = []
    for eq in units[:6]:
        h = health_scores.get(eq, {})
        score = float(h.get("health_score", 100.0))
        status = str(h.get("status", "Healthy"))
        snap = unit_snapshot(view, eq)
        rows.append(equipment_row(eq, status, snap["load"], snap["temp"], score, status_tone(status)))
    slot.markdown(
        '<div class="card"><div class="card-head"><div class="card-title">'
        '<span>🧊</span>Equipment overview</div><div class="card-link">View all</div></div>'
        + "".join(rows) + "</div>",
        unsafe_allow_html=True,
    )


def render_digital_twin(view: pd.DataFrame, focus: str, health_scores: Dict[str, Any]) -> None:
    h = health_scores.get(focus, {})
    status = str(h.get("status", "Healthy"))
    tone = status_tone(status)
    snap = unit_snapshot(view, focus)
    sub = view[view[EQUIPMENT_COL] == focus]
    power = float(sub[TARGET_ENERGY_COL].tail(4).mean()) if not sub.empty else 0.0
    chw = float(sub[CHILLED_WATER_COL].tail(24).mean()) if CHILLED_WATER_COL in sub and not sub.empty else 0.0
    comps = component_health(view, focus, float(h.get("health_score", 90.0)))

    comp_icons = {"Compressor": "🌀", "Motor": "⚙️", "Condenser": "♨️", "Evaporator": "❄️"}
    comp_html = ""
    for name, val in comps.items():
        tone_c = GREEN if val >= 75 else (AMBER if val >= 55 else RED)
        comp_html += f"""
<div class="comp-row">
  <div class="comp-ico">{comp_icons[name]}</div>
  <div style="flex:1;">
    <div class="comp-name"><span>{name}</span><b>{val:.0f}%</b></div>
    <div class="comp-bar"><i style="width:{val:.0f}%;background:{tone_c};"></i></div>
  </div>
</div>"""

    st.markdown(
        f"""
<div class="card" style="padding-bottom:.6rem;">
  <div class="card-head">
    <div class="card-title">{esc(focus)}
      <span style="color:{MUTED};font-weight:400;border-left:1px solid {BORDER};
      padding-left:.6rem;margin-left:.2rem;">Digital twin</span></div>
    <div style="display:flex;align-items:center;gap:.4rem;padding:.3rem .7rem;border-radius:8px;
      background:{hex_rgba(tone,.12)};border:1px solid {hex_rgba(tone,.32)};color:{tone};
      font-size:.78rem;font-weight:600;">⚠ {esc(status)}</div>
  </div>

  <div style="display:grid;grid-template-columns:1.55fr 1fr;gap:1rem;align-items:stretch;">
    <div class="twin" style="border-radius:12px;background:radial-gradient(circle at 50% 62%,
      rgba(29,78,216,.14),rgba(6,11,22,0) 70%);">
      <div class="chip" style="top:6%;left:6%;border-color:{hex_rgba(RED,.4)};">
        <span>Temperature</span><b style="color:{RED};">{snap['temp']:.1f} °C</b></div>
      <div class="chip" style="top:6%;right:8%;border-color:{hex_rgba(CYAN,.35)};">
        <span>Power</span><b style="color:{CYAN};">{power:,.1f} kW</b></div>
      <div class="chip" style="bottom:12%;left:3%;border-color:{hex_rgba(PURPLE,.35)};">
        <span>Chilled water</span><b style="color:{PURPLE};">{chw:.1f} L/s</b></div>
      <div class="chip" style="bottom:12%;right:5%;border-color:{hex_rgba(AMBER,.38)};">
        <span>Load</span><b style="color:{AMBER};">{snap['load']:.0f}%</b></div>
      {chiller_svg()}
    </div>


  </div>
</div>""",
        unsafe_allow_html=True,
    )


def render_rail(slot, view: pd.DataFrame, focus: str, health_scores: Dict[str, Any],
                recommendations: List[Dict[str, Any]]) -> None:
    h = health_scores.get(focus, {})
    status = str(h.get("status", "Healthy"))
    tone = status_tone(status)
    snap = unit_snapshot(view, focus)

    with slot:
        st.markdown(
            f"""
<div class="card" style="margin-bottom:.8rem;">
  <div class="card-head"><div class="card-title"><span>📶</span>Equipment details</div>
    <div class="card-link">›</div></div>
  <div style="display:flex;gap:.9rem;">
    <div style="width:74px;height:64px;border-radius:10px;background:#0b1322;border:1px solid {BORDER};
      display:grid;place-items:center;font-size:1.6rem;flex:none;">🧊</div>
    <div style="flex:1;">
      <div style="font-size:.95rem;font-weight:600;color:#eef4ff;">{esc(focus)}</div>
      <div class="eq-state" style="color:{tone};margin-bottom:.55rem;">
        <span style="width:6px;height:6px;border-radius:50%;background:{tone};display:inline-block;"></span>{esc(status)}
      </div>
      <div class="legend"><span>Load</span><b>{snap['load']:.0f}%</b></div>
      <div class="legend"><span>Temperature</span><b>{snap['temp']:.1f} °C</b></div>
      <div class="legend"><span>Efficiency</span><b>{snap['eff']:.0f}%</b></div>
    </div>
  </div>
</div>""",
            unsafe_allow_html=True,
        )

        alerts = recent_alerts(view)
        alert_html = ""
        for a in alerts:
            t = status_tone(a["severity"])
            alert_html += f"""
<div class="alert" style="background:{hex_rgba(t,.07)};border:1px solid {hex_rgba(t,.22)};">
  <div class="alert-ico" style="background:{hex_rgba(t,.16)};color:{t};">⚠</div>
  <div style="flex:1;min-width:0;">
    <b style="color:{t};">{esc(a['equipment'])}</b>
    <p>{esc(a['metric'])} · {esc(a['reason'])}</p>
  </div>
  <div class="ago">{esc(a['ago'])}</div>
</div>"""
        if not alert_html:
            alert_html = f'<p style="color:{MUTED};font-size:.8rem;margin:0;">No active alerts in this window.</p>'

        st.markdown(
            '<div class="card" style="margin-bottom:.8rem;"><div class="card-head">'
            '<div class="card-title"><span>🔔</span>Recent alerts</div>'
            '<div class="card-link">View all</div></div>' + alert_html + "</div>",
            unsafe_allow_html=True,
        )

        top_rec = recommendations[0] if recommendations else None
        worst = min(health_scores.items(), key=lambda kv: kv[1].get("health_score", 100))[0] \
            if health_scores else focus
        snap_w = unit_snapshot(view, worst)
        contributors = [
            f"Load running at {snap_w['load']:.0f}% of fleet peak",
            f"Temperature deviation ({snap_w['temp']:.1f} °C)",
            "Sustained positive energy residual against baseline",
        ]
        action_title = top_rec["title"] if top_rec else "Schedule condenser inspection"
        action_body = top_rec["evidence_basis"] if top_rec else \
            "Inspect the condenser and cooling circuit, then verify refrigerant charge."

        st.markdown(
            f"""
<div class="card">
  <div class="card-head"><div class="card-title"><span>🧠</span>AI diagnostic insight</div>
    <div class="card-link">View details</div></div>
  <div class="insight-flag"><span>⚠</span>
    <span>Abnormal energy consumption detected in {esc(worst)}.</span></div>
  <p style="font-size:.78rem;color:{MUTED};margin:.75rem 0 .1rem;">Likely contributors</p>
  <ul class="insight-list">{''.join(f'<li>{esc(c)}</li>' for c in contributors)}</ul>
  <div class="action"><span>💡</span>
    <div><b>{esc(action_title)}</b><p>{esc(action_body)}</p></div></div>
</div>""",
            unsafe_allow_html=True,
        )


def render_filters(slot, df: pd.DataFrame, units: List[str]):
    """Filter card. Returns (equipment, date_range, severity)."""
    with slot:
        st.markdown('<div class="card-title" style="margin-bottom:.55rem;">'
                    '<span>🗂️</span>Filters</div>', unsafe_allow_html=True)
        min_d, max_d = df[TIMESTAMP_COL].min().date(), df[TIMESTAMP_COL].max().date()
        date_range = st.date_input("Date range", value=(min_d, max_d),
                                   min_value=min_d, max_value=max_d, key="flt_date")
        equipment = st.selectbox("Equipment", ["All"] + units, key="flt_eq")
        try:
            levels = [str(lv) for lv in SEVERITY_LEVELS]
        except TypeError:
            levels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        severity = st.selectbox("Severity", ["All"] + levels, key="flt_sev")
    return equipment, date_range, severity


def pick_focus(equipment: str, health_scores: Dict[str, Any], units: List[str]) -> str:
    """The unit shown in the digital twin: the selected one, else the weakest."""
    if equipment and equipment != "All":
        return equipment
    if health_scores:
        return min(health_scores.items(), key=lambda kv: kv[1].get("health_score", 100))[0]
    return units[0] if units else "—"


def apply_filters(df: pd.DataFrame, equipment: str, date_range, severity: str) -> pd.DataFrame:
    out = df
    if equipment and equipment != "All":
        out = out[out[EQUIPMENT_COL] == equipment]
    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        s, e = date_range
        out = out[(out[TIMESTAMP_COL].dt.date >= s) & (out[TIMESTAMP_COL].dt.date <= e)]
    if severity and severity != "All" and "severity" in out.columns:
        out = out[out["severity"] == severity]
    return out


def severity_bars(view: pd.DataFrame) -> str:
    counts = (view["severity"].value_counts()
              .reindex(["LOW", "MEDIUM", "HIGH", "CRITICAL"], fill_value=0))
    top = max(int(counts.max()), 1)
    tones = {"LOW": GREEN, "MEDIUM": AMBER, "HIGH": "#fb7185", "CRITICAL": RED}
    rows = ""
    for level, n in counts.items():
        rows += f"""
<div style="margin-bottom:.6rem;">
  <div class="legend" style="padding:0 0 .25rem;">
    <span><i style="background:{tones[level]};"></i>{level.title()}</span><b>{int(n):,}</b></div>
  <div class="comp-bar"><i style="width:{int(n)/top*100:.0f}%;background:{tones[level]};"></i></div>
</div>"""
    return rows


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    inject_css()
    render_sidebar_brand()
    nav_slot = st.sidebar.container()

    raw_df, quality_report, is_demo, contamination = load_telemetry()
    render_sidebar_footer()

    if raw_df is None or len(raw_df) == 0:
        st.error("No telemetry available. Upload a CSV or switch to synthetic demo data.")
        return

    with st.spinner("Analysing telemetry…"):
        results = execute_pipeline(raw_df, contamination)

    df = results["df"]
    episodes = results["episodes"]
    health_scores = results["health_scores"]
    recommendations = results["recommendations"]
    units = sorted(df[EQUIPMENT_COL].dropna().unique().tolist())
    open_anomalies = int((df["anomaly_flag"] == 1).sum())

    with nav_slot:
        nav_labels = list(NAV_ITEMS)
        nav_labels[3] = f"⚠️  Anomalies  ({open_anomalies:,})"
        page = st.radio("Navigation", nav_labels, index=0, label_visibility="collapsed")
    page = page.split("  ")[1].strip()

    # Containers reserved so filters can be read before the KPI strip renders
    header_slot = st.container()
    kpi_slot = st.container()
    body = st.container()

    with header_slot:
        render_topbar(df)
        if is_demo:
            st.info("Running on synthetic demo telemetry.")

    with body:
        main_col, rail_col = st.columns([3.35, 1.12], gap="medium")

        with main_col:
            if page == "Dashboard":
                tabs = st.tabs(["🏠  Command centre", "🧊  Equipment", "⚡  Energy",
                                "⚠️  Anomalies", "🧠  AI diagnosis"])
                with tabs[0]:
                    left, centre = st.columns([1, 1.72], gap="medium")
                    with left:
                        overview_slot = st.container()
                        st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
                        filter_slot = st.container()
                    equipment, date_range, severity = render_filters(filter_slot, df, units)
                    view = apply_filters(df, equipment, date_range, severity)
                    if view.empty:
                        view = df
                    focus = pick_focus(equipment, health_scores, units)

                    render_equipment_overview(overview_slot, view, health_scores, units)

                    with centre:
                        render_digital_twin(view, focus, health_scores)
                        st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
                        c1, c2 = st.columns([1.55, 1], gap="small")
                        with c1:
                            st.markdown('<div class="card-title" style="margin-bottom:.2rem;">'
                                        '<span>⚡</span>Energy consumption &amp; baseline</div>',
                                        unsafe_allow_html=True)
                            st.plotly_chart(theme_fig(plot_anomaly_timeline(
                                view.tail(1200), equipment_id=focus), height=260),
                                use_container_width=True)
                        with c2:
                            st.markdown('<div class="card-title" style="margin-bottom:.2rem;">'
                                        '<span>🧮</span>Energy breakdown</div>',
                                        unsafe_allow_html=True)
                            st.plotly_chart(donut_energy(view), use_container_width=True)
                            total = view.groupby(EQUIPMENT_COL)[TARGET_ENERGY_COL].sum()
                            grand = float(total.sum()) or 1.0
                            palette = [GREEN, PURPLE, AMBER, CYAN, BLUE, RED]
                            legend = "".join(
                                f'<div class="legend"><span><i style="background:{palette[i % len(palette)]}"></i>'
                                f'{esc(k)}</span><b>{v/grand*100:.0f}%</b></div>'
                                for i, (k, v) in enumerate(total.sort_values(ascending=False).items()))
                            st.markdown(legend, unsafe_allow_html=True)
                with tabs[1]:
                    render_equipment_page(view, health_scores, units)
                with tabs[2]:
                    render_energy_page(view, focus)
                with tabs[3]:
                    render_anomaly_page(view, episodes, focus)
                with tabs[4]:
                    render_diagnosis_page(recommendations, quality_report)
            else:
                filter_slot = st.container()
                equipment, date_range, severity = render_filters(filter_slot, df, units)
                view = apply_filters(df, equipment, date_range, severity)
                if view.empty:
                    view = df
                focus = pick_focus(equipment, health_scores, units)
                st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)

                if page == "Equipment":
                    render_equipment_page(view, health_scores, units)
                elif page == "Energy Analytics":
                    render_energy_page(view, focus)
                elif page == "Anomalies":
                    render_anomaly_page(view, episodes, focus)
                elif page == "AI Diagnosis":
                    render_diagnosis_page(recommendations, quality_report)
                elif page == "Reports":
                    render_reports_page(view, episodes, health_scores)
                else:
                    render_settings_page(contamination, results)

        render_rail(rail_col, view, focus, health_scores, recommendations)

    with kpi_slot:
        render_kpis(view, health_scores, units)


# ----------------------------------------------------------------------------
# Secondary pages
# ----------------------------------------------------------------------------
def render_equipment_page(view: pd.DataFrame, health_scores: Dict[str, Any],
                          units: List[str]) -> None:
    st.markdown('<div class="card-title" style="margin-bottom:.6rem;">'
                '<span>🩺</span>Unit health gauges</div>', unsafe_allow_html=True)
    if not units:
        st.info("No equipment found in this dataset.")
        return
    cols = st.columns(min(len(units), 4), gap="small")
    for i, eq in enumerate(units):
        h = health_scores.get(eq, {})
        with cols[i % len(cols)]:
            st.plotly_chart(theme_fig(plot_health_gauge(h.get("health_score", 100.0), eq), 230),
                            use_container_width=True)
            st.caption(f"{h.get('status', 'Normal')} · anomaly rate {h.get('anomaly_rate_pct', 0.0)}%")

    st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
    table = pd.DataFrame([
        {"Equipment": eq, "Health": f"{h.get('health_score', 100):.0f}/100",
         "Status": h.get("status", "Normal"), "Anomalies": h.get("anomaly_count", 0)}
        for eq, h in health_scores.items()
    ])
    st.dataframe(table, use_container_width=True, hide_index=True)


def render_energy_page(view: pd.DataFrame, focus: str) -> None:
    st.markdown('<div class="card-title" style="margin-bottom:.4rem;">'
                '<span>⚖️</span>Building load vs energy draw</div>', unsafe_allow_html=True)
    st.plotly_chart(theme_fig(plot_energy_vs_load_scatter(view, equipment_id=focus), 330),
                    use_container_width=True)
    st.markdown('<div class="card-title" style="margin:.7rem 0 .4rem;">'
                '<span>🌐</span>Contextual drivers</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.plotly_chart(theme_fig(plot_contextual_scatter(
            view, CHILLED_WATER_COL, "Chilled water rate (L/sec)", focus), 300),
            use_container_width=True)
    with c2:
        st.plotly_chart(theme_fig(plot_contextual_scatter(
            view, COOLING_WATER_COL, "Cooling water temp (°C)", focus), 300),
            use_container_width=True)


def render_anomaly_page(view: pd.DataFrame, episodes: List[Dict[str, Any]], focus: str) -> None:
    c1, c2 = st.columns([2.1, 1], gap="medium")
    with c1:
        st.markdown('<div class="card-title" style="margin-bottom:.4rem;">'
                    '<span>📈</span>Anomaly timeline</div>', unsafe_allow_html=True)
        st.plotly_chart(theme_fig(plot_anomaly_timeline(view, equipment_id=focus), 330),
                        use_container_width=True)
    with c2:
        st.markdown('<div class="card-title" style="margin-bottom:.5rem;">'
                    '<span>📊</span>Severity breakdown</div>', unsafe_allow_html=True)
        st.markdown(severity_bars(view), unsafe_allow_html=True)

    st.markdown('<div class="card-title" style="margin:.7rem 0 .4rem;">'
                '<span>🔎</span>Abnormal episodes</div>', unsafe_allow_html=True)
    if episodes:
        ep_df = pd.DataFrame([{
            "Episode": ep["episode_id"],
            "Equipment": ep["equipment_id"],
            "Start": ep["start_time"],
            "End": ep["end_time"],
            "Duration (h)": ep["duration_hours"],
            "Max severity": ep.get("max_severity", "MEDIUM"),
        } for ep in episodes])
        st.dataframe(ep_df, use_container_width=True, hide_index=True)
    else:
        st.info("No persistent anomaly episodes in this window.")


def render_diagnosis_page(recommendations: List[Dict[str, Any]],
                          quality_report: Dict[str, Any]) -> None:
    st.markdown('<div class="card-title" style="margin-bottom:.6rem;">'
                '<span>💡</span>Recommended actions</div>', unsafe_allow_html=True)
    if not recommendations:
        st.info("No actions required — every unit is tracking its contextual baseline.")
    for rec in recommendations:
        tone = status_tone(rec.get("priority", "MEDIUM"))
        st.markdown(
            f"""
<div class="card" style="margin-bottom:.55rem;border-color:{hex_rgba(tone,.3)};">
  <div style="display:flex;align-items:center;gap:.6rem;margin-bottom:.35rem;">
    <span style="padding:.16rem .55rem;border-radius:6px;font-size:.68rem;font-weight:700;
      background:{hex_rgba(tone,.16)};color:{tone};">{esc(rec['priority'])}</span>
    <b style="font-size:.9rem;color:#eef4ff;">{esc(rec['title'])}</b>
    <span style="margin-left:auto;font-size:.74rem;color:{MUTED};">{esc(rec['equipment_id'])}</span>
  </div>
  <p style="margin:0;font-size:.79rem;color:{MUTED};line-height:1.5;">{esc(rec['evidence_basis'])}</p>
</div>""",
            unsafe_allow_html=True,
        )
    with st.expander("Telemetry quality profile"):
        st.json(quality_report)


def render_reports_page(view: pd.DataFrame, episodes: List[Dict[str, Any]],
                        health_scores: Dict[str, Any]) -> None:
    st.markdown('<div class="card-title" style="margin-bottom:.6rem;">'
                '<span>📄</span>Export</div>', unsafe_allow_html=True)
    st.download_button("Download filtered telemetry (CSV)",
                       view.to_csv(index=False).encode("utf-8"),
                       file_name="yukthi_telemetry.csv", mime="text/csv")
    if episodes:
        st.download_button("Download anomaly episodes (CSV)",
                           pd.DataFrame(episodes).to_csv(index=False).encode("utf-8"),
                           file_name="yukthi_episodes.csv", mime="text/csv")
    st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
    st.dataframe(view.head(500), use_container_width=True, hide_index=True)


def render_settings_page(contamination: float, results: Dict[str, Any]) -> None:
    st.markdown('<div class="card-title" style="margin-bottom:.6rem;">'
                '<span>⚙️</span>Model configuration</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown(f'<div class="card"><p style="font-size:.78rem;color:{MUTED};margin:0 0 .4rem;">'
                    f'Anomaly sensitivity (contamination)</p>'
                    f'<div class="kpi-val">{contamination:.2f}</div></div>', unsafe_allow_html=True)
        st.json(results.get("reg_metrics", {}))
    with c2:
        st.json(results.get("det_metrics", {}))
        st.json(results.get("prep_summary", {}))


if __name__ == "__main__":
    main()