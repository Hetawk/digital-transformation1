"""
Configuration settings for the digital transformation project
"""

import os
from pathlib import Path
import logging
import sys

# Project directories
BASE_DIR = Path(__file__).parent.parent.parent
RESULTS_DIR = BASE_DIR / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

# Data settings
DATA_DIR = BASE_DIR / "dataset"
DATA_FILE = DATA_DIR / "msci_dt_processed_2010_2023.csv"
PREPROCESSED_FILE = DATA_DIR / "preprocessed_data.pkl"

# Analysis settings
TREATMENT_YEAR = 2020  # MSCI inclusion event
SAMPLE_YEARS = list(range(2010, 2024))  # Years included in analysis
PLACEBO_YEAR = 2015    # Year for placebo test

# Plot settings
PLOT_FIGSIZE = (10, 6)
PLOT_DPI = 300
COLORS = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "treated": "#2ca02c",
    "control": "#d62728",
    "highlight": "#9467bd"
}

# Configure logging
def configure_logging():
    """Configure logging for the project"""
    logger = logging.getLogger('digital_transformation')
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    logger.setLevel(logging.INFO)
    
    # Create console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', 
                                 datefmt='%Y-%m-%d %H:%M:%S')
    console.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console)
    
    return logger

# Initialize logger
logger = configure_logging()

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

# Model specifications
RANDOM_SEED = 42
