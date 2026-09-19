"""
Configuration settings, schema definitions, and parameters for Intelligent Energy Monitoring.
Adheres to zero hard-coding of equipment names, rows, timestamps, or static thresholds.
"""

from typing import List

# Core Column Names matching dataset schema
TIMESTAMP_COL = "timestamp"
EQUIPMENT_COL = "equipment_id"
TARGET_ENERGY_COL = "Chiller Energy Consumption (kWh)"

BUILDING_LOAD_COL = "Building Load (RT)"
CHILLED_WATER_COL = "Chilled Water Rate (L/sec)"
COOLING_WATER_COL = "Cooling Water Temperature (C)"

OUTSIDE_TEMP_COL = "Outside Temperature (F)"
DEW_POINT_COL = "Dew Point (F)"
HUMIDITY_COL = "Humidity (%)"
WIND_SPEED_COL = "Wind Speed (mph)"
PRESSURE_COL = "Pressure (in)"

REQUIRED_COLUMNS: List[str] = [
    TIMESTAMP_COL,
    EQUIPMENT_COL,
    CHILLED_WATER_COL,
    COOLING_WATER_COL,
    BUILDING_LOAD_COL,
    TARGET_ENERGY_COL,
    OUTSIDE_TEMP_COL,
    DEW_POINT_COL,
    HUMIDITY_COL,
    WIND_SPEED_COL,
    PRESSURE_COL
]

OPERATIONAL_COLUMNS: List[str] = [
    BUILDING_LOAD_COL,
    CHILLED_WATER_COL,
    COOLING_WATER_COL
]

ENVIRONMENTAL_COLUMNS: List[str] = [
    OUTSIDE_TEMP_COL,
    DEW_POINT_COL,
    HUMIDITY_COL,
    WIND_SPEED_COL,
    PRESSURE_COL
]

NUMERIC_COLUMNS: List[str] = OPERATIONAL_COLUMNS + ENVIRONMENTAL_COLUMNS + [TARGET_ENERGY_COL]

# Nominal Sampling & Time Discontinuity Tracking
NOMINAL_INTERVAL_MINUTES = 30
GAP_THRESHOLD_MINUTES = 60  # Discontinuities > 60 mins tracked as time gaps, not anomalies

# Model Parameters
RANDOM_STATE = 42
IFOREST_CONTAMINATION = 0.05  # Expected approximate baseline anomaly contamination
IFOREST_ESTIMATORS = 150

# Severity Levels
SEVERITY_LEVELS = ["NORMAL", "LOW", "MEDIUM", "HIGH", "CRITICAL"]

# Persistence Clustering Parameters
PERSISTENCE_MIN_OBSERVATIONS = 2  # At least 2 abnormal intervals in window to flag persistent
PERSISTENCE_WINDOW_HOURS = 4      # Group anomalies occurring within a 4-hour rolling span

# Health Score Base Points
MAX_HEALTH_SCORE = 100.0
