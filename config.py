"""
Configuration settings for the MSCI inclusion and digital transformation analysis
"""

import os
from pathlib import Path

# Project directory paths
ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = ROOT_DIR / "dataset"
RESULTS_DIR = ROOT_DIR / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

# Create directories if they don't exist
for dir_path in [RESULTS_DIR, TABLES_DIR, FIGURES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Data file paths
DATA_FILE = DATA_DIR / "msci_dt_processed_2010_2023.csv"

# Analysis parameters
TREATMENT_YEAR = 2018  # Year of MSCI inclusion (post-2018)
PLACEBO_YEAR = 2015    # Year for placebo tests

# Variable groups
DT_MEASURES = [
    "Digital_transformationA",
    "Digital_transformationB",
    "Digital_transformation_rapidA",
    "Digital_transformation_rapidB"
]

CONTROL_VARS = [
    "age",
    "TFP_OP",
    "SA_index",
    "WW_index",
    "F050501B",  # Return on assets (ROA)
    "F060101B"   # Asset turnover ratio
]

FINANCIAL_ACCESS_VARS = [
    "SA_index",
    "WW_index"
]

CORP_GOV_VARS = [
    "Top3DirectorSumSalary2",
    "DirectorHoldSum2",
    "DirectorUnpaidNo2"
]

INVESTOR_SCRUTINY_VARS = [
    "ESG_Score_mean"
]

# Plot settings
PLOT_DPI = 300
PLOT_FIGSIZE = (10, 6)
COLORS = {
    'treated': '#1f77b4',  # blue
    'control': '#ff7f0e',  # orange
    'highlight': '#2ca02c'  # green
}

# Model specifications
RANDOM_SEED = 42
