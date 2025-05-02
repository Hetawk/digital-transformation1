"""
Main script for MSCI inclusion and digital transformation analysis
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import project modules
import config
from src.data_preprocessing import DataPreprocessor
from src.analysis.descriptive import DescriptiveAnalysis
from src.analysis.hypothesis import HypothesisTesting
from src.analysis.models import ModelAnalysis
from src.analysis.mechanisms import MechanismAnalysis
from src.visualization.descriptive_plots import DescriptivePlots
from src.visualization.model_plots import ModelPlots
from src.visualization.mechanism_plots import MechanismPlots


def setup_directories():
    """Create necessary directories if they don't exist"""
    for dir_path in [config.RESULTS_DIR, config.TABLES_DIR, config.FIGURES_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Directory exists or created: {dir_path}")


def preprocess_data(data_file=None):
    """
    Load and preprocess the data

    Parameters:
    -----------
    data_file : str or Path, optional
        Path to data file

    Returns:
    --------
    pd.DataFrame: Preprocessed data
    """
    print("\n" + "="*80)
    print("DATA PREPROCESSING".center(80))
    print("="*80)

    try:
        preprocessor = DataPreprocessor(data_file)
        data = preprocessor.run_all()
        print(f"Data preprocessing complete. Shape: {data.shape}")
        return data
    except Exception as e:
        print(f"\nError in data preprocessing: {e}")
        print("\nAttempting to continue with synthetic data...")
        # Create an instance that will auto-generate synthetic data
        preprocessor = DataPreprocessor(None)
        data = preprocessor._generate_synthetic_data()
        data = preprocessor.run_all()  # Run all preprocessing steps on synthetic data
        print(f"Continuing with synthetic data. Shape: {data.shape}")
        return data


def run_descriptive_analysis(data):
    """
    Run descriptive analysis

    Parameters:
    -----------
    data : pd.DataFrame
        Preprocessed data

    Returns:
    --------
    DescriptiveAnalysis: The descriptive analysis object
    """
    print("\n" + "="*80)
    print("DESCRIPTIVE ANALYSIS".center(80))
    print("="*80)

    descriptive = DescriptiveAnalysis(data)

    print("\nGenerating summary statistics...")
    summary = descriptive.summarize_variables()
    print(summary.head())

    print("\nGenerating correlation matrix...")
    corr = descriptive.correlation_analysis()
    print(corr.head())

    print("\nChecking treatment balance...")
    try:
        balance = descriptive.treatment_balance()
        print(balance.head())
    except Exception as e:
        print(f"Error generating treatment balance: {e}")

    print("\nCreating treatment distribution tables...")
    try:
        dist_tables = descriptive.create_treatment_distribution_table()
        for name, table in dist_tables.items():
            print(f"\n{name}:\n{table}")
    except Exception as e:
        print(f"Error generating treatment distribution: {e}")

    print("\nExporting descriptive statistics...")
    descriptive.export_results()

    return descriptive


def create_descriptive_plots(data):
    """
    Create descriptive plots

    Parameters:
    -----------
    data : pd.DataFrame
        Preprocessed data

    Returns:
    --------
    dict: Dictionary of created figures
    """
    print("\n" + "="*80)
    print("DESCRIPTIVE VISUALIZATIONS".center(80))
    print("="*80)

    plotter = DescriptivePlots(data)

    print("\nGenerating all descriptive plots...")
    figures = plotter.plot_all_descriptives(save=True)

    print(f"Created {len(figures)} descriptive visualizations.")
    return figures


def run_hypothesis_testing(data):
    """
    Run hypothesis testing

    Parameters:
    -----------
    data : pd.DataFrame
        Preprocessed data

    Returns:
    --------
    HypothesisTesting: The hypothesis testing object
    """
    print("\n" + "="*80)
    print("HYPOTHESIS TESTING".center(80))
    print("="*80)

    hypothesis = HypothesisTesting(data)

    print("\nTesting Hypothesis 1: Capital market liberalization positively influences digital transformation")
    h1_model = hypothesis.test_h1(controls=True, year_fe=True)

    print("\nTesting Hypothesis 2: MSCI inclusion leads to increased adoption of digital technologies")
    h2_model = hypothesis.test_h2(controls=True, year_fe=True)

    print("\nRunning placebo test...")
    placebo_model = hypothesis.run_placebo_test(controls=True, year_fe=True)

    print("\nComparing hypothesis test results...")
    comparison = hypothesis.compare_hypotheses()
    print(comparison)

    print("\nExporting hypothesis testing results...")
    hypothesis.export_results()

    return hypothesis


def run_model_analysis(data):
    """
    Run model analysis

    Parameters:
    -----------
    data : pd.DataFrame
        Preprocessed data

    Returns:
    --------
    ModelAnalysis: The model analysis object
    """
    print("\n" + "="*80)
    print("MODEL ANALYSIS".center(80))
    print("="*80)

    models = ModelAnalysis(data)

    print("\nRunning Pooled OLS DiD models...")
    did_model = models.run_pooled_ols_did(controls=True, year_fe=True)
    did_model_b = models.run_pooled_ols_did(
        dv="Digital_transformationB", controls=True, year_fe=True)

    print("\nRunning Direct MSCI Effect model (FE)...")
    try:
        direct_model = models.run_direct_msci_effect(
            controls=True, year_fe=True)
    except Exception as e:
        print(f"Error running direct MSCI effect model: {e}")

    print("\nRunning Post-Period Analysis...")
    try:
        post_model = models.run_post_period_analysis(
            controls=True, year_fe=True)
    except Exception as e:
        print(f"Error running post-period analysis: {e}")

    print("\nRunning Matched Sample Analysis...")
    try:
        matched_data, ate, ate_se = models.run_matched_sample_analysis(
            matching_vars=["age", "TFP_OP"]
        )
        print(f"Matched Sample ATE: {ate:.4f} (SE: {ate_se:.4f})")
    except Exception as e:
        print(f"Error running matched sample analysis: {e}")

    print("\nRunning First Differences Analysis...")
    try:
        fd_model = models.run_first_differences(controls=True, year_fe=True)
    except Exception as e:
        print(f"Error running first differences analysis: {e}")

    print("\nRunning Event Study Analysis...")
    try:
        event_model = models.run_event_study(window=(-3, 3), controls=True)
    except Exception as e:
        print(f"Error running event study analysis: {e}")

    print("\nComparing model results...")
    comparison = models.compare_models()
    print(comparison)

    print("\nExporting model analysis results...")
    models.export_results()

    return models


def create_model_plots(models, hypothesis):
    """
    Create model plots

    Parameters:
    -----------
    models : ModelAnalysis
        The model analysis object
    hypothesis : HypothesisTesting
        The hypothesis testing object

    Returns:
    --------
    dict: Dictionary of created figures
    """
    print("\n" + "="*80)
    print("MODEL VISUALIZATIONS".center(80))
    print("="*80)

    plotter = ModelPlots()
    figures = {}

    print("\nCreating coefficient comparison plot...")
    try:
        comparison = models.compare_models()
        fig = plotter.plot_coefficient_comparison(
            comparison,
            coefficient_col='Estimate',
            stderr_col='Std. Error',
            model_col='Model',
            title='Comparison of Alternative Model Estimates'
        )
        figures['coefficient_comparison'] = fig
    except Exception as e:
        print(f"Error creating coefficient comparison plot: {e}")

    print("\nCreating event study plot...")
    try:
        if 'event_study_Digital_transformationA_controls' in models.models:
            fig = plotter.plot_event_study(
                models.models['event_study_Digital_transformationA_controls'],
                coef_prefix='time_',
                title='Event Study: Dynamic Effects of MSCI Inclusion'
            )
            figures['event_study'] = fig
    except Exception as e:
        print(f"Error creating event study plot: {e}")

    print("\nCreating placebo comparison plot...")
    try:
        # Extract coefficients from hypothesis testing results
        h1_model = hypothesis.results.get('h1')
        placebo_model = hypothesis.results.get('placebo')

        if h1_model is not None and placebo_model is not None:
            h1_coef = h1_model.params.get('TreatPost')
            h1_stderr = h1_model.bse.get('TreatPost')
            placebo_coef = placebo_model.params.get('Placebo_TreatPost')
            placebo_stderr = placebo_model.bse.get('Placebo_TreatPost')

            fig = plotter.plot_placebo_comparison(
                h1_coef, h1_stderr, placebo_coef, placebo_stderr
            )
            figures['placebo_comparison'] = fig
    except Exception as e:
        print(f"Error creating placebo comparison plot: {e}")

    return figures


def run_mechanism_analysis(data):
    """
    Run mechanism analysis

    Parameters:
    -----------
    data : pd.DataFrame
        Preprocessed data

    Returns:
    --------
    MechanismAnalysis: The mechanism analysis object
    """
    print("\n" + "="*80)
    print("MECHANISM ANALYSIS".center(80))
    print("="*80)

    mechanisms = MechanismAnalysis(data)

    print("\nAnalyzing Financial Access Mechanisms...")
    try:
        fin_models = mechanisms.analyze_financial_access(post_only=True)
    except Exception as e:
        print(f"Error analyzing financial access mechanisms: {e}")

    print("\nAnalyzing Corporate Governance Mechanisms...")
    try:
        gov_models = mechanisms.analyze_corporate_governance(post_only=True)
    except Exception as e:
        print(f"Error analyzing corporate governance mechanisms: {e}")

    print("\nAnalyzing Investor Scrutiny Mechanisms...")
    try:
        inv_models = mechanisms.analyze_investor_scrutiny(post_only=True)
    except Exception as e:
        print(f"Error analyzing investor scrutiny mechanisms: {e}")

    print("\nExtracting mechanism effects...")
    try:
        effects = mechanisms.extract_mechanism_effects()
        print(effects)
    except Exception as e:
        print(f"Error extracting mechanism effects: {e}")

    print("\nExporting mechanism analysis results...")
    mechanisms.export_results()

    return mechanisms


def create_mechanism_plots(mechanisms):
    """
    Create mechanism plots

    Parameters:
    -----------
    mechanisms : MechanismAnalysis
        The mechanism analysis object

    Returns:
    --------
    dict: Dictionary of created figures
    """
    print("\n" + "="*80)
    print("MECHANISM VISUALIZATIONS".center(80))
    print("="*80)

    plotter = MechanismPlots()
    figures = {}

    print("\nCreating mechanism effects plot...")
    try:
        effects = mechanisms.extract_mechanism_effects()
        fig = plotter.plot_mechanism_effects(
            effects,
            mech_col='Mechanism',
            var_col='Variable',
            effect_col='Interaction Effect',
            stderr_col='Std. Error'
        )
        figures['mechanism_effects'] = fig
    except Exception as e:
        print(f"Error creating mechanism effects plot: {e}")

    print("\nCreating mechanism heatmap...")
    try:
        effects = mechanisms.extract_mechanism_effects()
        fig = plotter.plot_mechanism_heatmap(
            effects,
            mech_col='Mechanism',
            var_col='Variable',
            effect_col='Interaction Effect',
            pval_col='p-value'
        )
        figures['mechanism_heatmap'] = fig
    except Exception as e:
        print(f"Error creating mechanism heatmap: {e}")

    return figures


def run_full_analysis(data_file=None):
    """
    Run the full analysis workflow

    Parameters:
    -----------
    data_file : str or Path, optional
        Path to data file
    """
    # Setup directories
    setup_directories()

    # Preprocess data
    data = preprocess_data(data_file)

    # Run descriptive analysis
    descriptive = run_descriptive_analysis(data)

    # Create descriptive plots
    descriptive_figures = create_descriptive_plots(data)

    # Run hypothesis testing
    hypothesis = run_hypothesis_testing(data)

    # Run model analysis
    models = run_model_analysis(data)

    # Create model plots
    model_figures = create_model_plots(models, hypothesis)

    # Run mechanism analysis
    mechanisms = run_mechanism_analysis(data)

    # Create mechanism plots
    mechanism_figures = create_mechanism_plots(mechanisms)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE".center(80))
    print("="*80)
    print(f"Results saved to: {config.RESULTS_DIR}")
    print(f"Tables saved to: {config.TABLES_DIR}")
    print(f"Figures saved to: {config.FIGURES_DIR}")


def main():
    """Main function to parse command line arguments and run analysis"""
    parser = argparse.ArgumentParser(
        description="MSCI inclusion and digital transformation analysis")
    parser.add_argument('--data', type=str,
                        help='Path to data file (optional)')
    args = parser.parse_args()

    # Set matplotlib backend to non-interactive
    plt.switch_backend('agg')

    print("\n" + "="*80)
    print("MSCI INCLUSION AND DIGITAL TRANSFORMATION ANALYSIS".center(80))
    print("="*80)
    print(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}")
    print(f"Treatment year: {config.TREATMENT_YEAR}")

    try:
        run_full_analysis(args.data)
    except Exception as e:
        print(f"\nError in analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
