# MSCI Inclusion and Digital Transformation Analysis

This repository contains code for analyzing the relationship between MSCI inclusion (capital market liberalization) and digital transformation of firms.

## Project Overview

This research examines how inclusion in the MSCI Emerging Markets Index affects companies' digital transformation efforts. The analysis uses difference-in-differences and other econometric approaches to identify causal relationships and underlying mechanisms.

## Project Structure

```
patie_preprocess/
├── config.py                      # Configuration settings
├── main.py                        # Main entry point
├── src/
│   ├── data_preprocessing.p       # Data preparation
│   ├── analysis/                  # Analysis modules
│   │   ├── descriptive.py         # Descriptive statistics
│   │   ├── hypothesis.py          # Hypothesis testing
│   │   ├── models.py              # Econometric models
│   │   └── mechanisms.py          # Mechanism analysis
│   └── visualization/             # Visualization modules
│       ├── descriptive_plots.py   # Descriptive visualizations
│       ├── model_plots.py         # Model result visualizations
│       └── mechanism_plots.py     # Mechanism visualizations
├── dataset/                          # Data directory
└── results/                       # Generated figures

```

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the full analysis:

```
python main.py
```

The script will:

1. Preprocess the data
2. Run descriptive analysis
3. Perform hypothesis testing
4. Conduct model analysis
5. Analyze potential mechanisms
6. Generate visualizations for all analyses

## Analysis Workflow

1. **Data Preprocessing**: Cleans, transforms, and prepares data for analysis
2. **Descriptive Analysis**: Generates summary statistics and correlation matrices
3. **Hypothesis Testing**: Tests key hypotheses about MSCI inclusion and digital transformation
4. **Model Analysis**: Implements DiD models, event studies, matching methods, and robustness checks
5. **Mechanism Analysis**: Explores potential causal mechanisms including financial access, corporate governance, and investor scrutiny

## Requirements

- Python 3.7+
- pandas
- numpy
- matplotlib
- statsmodels
- scikit-learn
- seaborn
- pathlib

## Author


## License

[License Information]
