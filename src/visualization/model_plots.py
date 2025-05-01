"""
Model visualization module for MSCI inclusion and digital transformation analysis
"""

import config
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))


class ModelPlots:
    def __init__(self, data=None):
        """
        Initialize the ModelPlots

        Parameters:
        -----------
        data : pd.DataFrame or None
            The preprocessed dataset, optional as some plots don't need the full dataset
        """
        self.data = data
        self.setup_plot_style()

    def setup_plot_style(self):
        """Set up the matplotlib/seaborn plotting style"""
        sns.set_theme(style="whitegrid")
        plt.rcParams['figure.figsize'] = config.PLOT_FIGSIZE
        plt.rcParams['savefig.dpi'] = config.PLOT_DPI

    def plot_coefficient_comparison(self, results_df, coefficient_col='Estimate',
                                    stderr_col='Std. Error', model_col='Model',
                                    title=None, save=True, filename='coefficient_comparison.png'):
        """
        Plot coefficient comparison across different models

        Parameters:
        -----------
        results_df : pd.DataFrame
            DataFrame with model results (one row per model)
        coefficient_col : str
            Column name with coefficient estimates
        stderr_col : str
            Column name with standard errors
        model_col : str
            Column name with model names
        title : str or None
            Custom title for plot
        save : bool
            Whether to save the plot
        filename : str
            Filename to save the plot

        Returns:
        --------
        matplotlib.figure.Figure: The figure object
        """
        # Create figure
        fig, ax = plt.subplots()

        # Get model names and coefficients
        models = results_df[model_col].values
        coefs = results_df[coefficient_col].values

        # Calculate confidence intervals
        ci_low = coefs - 1.96 * results_df[stderr_col].values
        ci_high = coefs + 1.96 * results_df[stderr_col].values

        # Create y-positions for the models
        y_pos = np.arange(len(models))

        # Horizontal bar plot
        ax.barh(y_pos, coefs, height=0.5,
                color=config.COLORS['treated'], alpha=0.7)

        # Add error bars
        ax.errorbar(coefs, y_pos, xerr=np.vstack([coefs - ci_low, ci_high - coefs]),
                    fmt='none', ecolor='black', capsize=3)

        # Add zero reference line
        ax.axvline(x=0, color='gray', linestyle='--')

        # Set labels and title
        ax.set_yticks(y_pos)
        ax.set_yticklabels(models)
        ax.set_xlabel('Coefficient Estimate')
        ax.set_title(title or 'Model Coefficient Comparison')

        # Add note about confidence intervals
        plt.figtext(0.5, 0.01, "Note: Error bars represent 95% confidence intervals.",
                    ha="center", fontsize=8, style='italic')

        plt.tight_layout()

        # Save the figure if requested
        if save:
            output_path = config.FIGURES_DIR / filename
            plt.savefig(output_path)
            print(f"Plot saved to {output_path}")

        return fig

    def plot_event_study(self, event_study_results, coef_prefix='time_',
                         title='Event Study: Dynamic Effects of MSCI Inclusion',
                         save=True):
        """
        Plot event study results

        Parameters:
        -----------
        event_study_results : statsmodels regression result
            Regression results from event study analysis
        coef_prefix : str
            Prefix of event time dummy variables in regression
        title : str
            Title for the plot
        save : bool
            Whether to save the plot

        Returns:
        --------
        matplotlib.figure.Figure: The figure object
        """
        # Extract coefficients and standard errors for event time dummies
        coefs = []
        periods = []
        ci_low = []
        ci_high = []

        # Get all coefficients that start with the prefix
        for name in event_study_results.params.index:
            if name.startswith(coef_prefix):
                # Extract period number from variable name
                try:
                    period = int(name[len(coef_prefix):])
                    periods.append(period)
                    coefs.append(event_study_results.params[name])
                    stderr = event_study_results.bse[name]
                    ci_low.append(coefs[-1] - 1.96 * stderr)
                    ci_high.append(coefs[-1] + 1.96 * stderr)
                except:
                    continue

        # Sort by period
        period_order = np.argsort(periods)
        periods = [periods[i] for i in period_order]
        coefs = [coefs[i] for i in period_order]
        ci_low = [ci_low[i] for i in period_order]
        ci_high = [ci_high[i] for i in period_order]

        # Create figure
        fig, ax = plt.subplots()

        # Plot coefficients
        ax.plot(periods, coefs, marker='o', linestyle='-',
                color=config.COLORS['treated'])

        # Add confidence intervals
        ax.fill_between(periods, ci_low, ci_high, alpha=0.2,
                        color=config.COLORS['treated'])

        # Add zero reference line
        ax.axhline(y=0, color='gray', linestyle='--')

        # Add vertical line at event time = 0
        ax.axvline(x=0, color='gray', linestyle='--')

        # Set labels and title
        ax.set_xlabel('Years Relative to MSCI Inclusion')
        ax.set_ylabel('Effect on Digital Transformation')
        ax.set_title(title)

        # Add note
        plt.figtext(0.5, 0.01, "Note: Coefficients are relative to t-1. Shaded area represents 95% CI.",
                    ha="center", fontsize=8, style='italic')

        plt.tight_layout()

        # Save the figure if requested
        if save:
            output_path = config.FIGURES_DIR / "event_study_plot.png"
            plt.savefig(output_path)
            print(f"Plot saved to {output_path}")

        return fig

    def plot_heterogeneity(self, results_df, model_col='Interaction', coef_col='Estimate',
                           stderr_col='Std. Error', title='Heterogeneity Analysis',
                           save=True, filename='heterogeneity_effects.png'):
        """
        Plot heterogeneity in treatment effects

        Parameters:
        -----------
        results_df : pd.DataFrame
            DataFrame with heterogeneity results
        model_col : str
            Column name with interaction terms
        coef_col : str
            Column name with coefficient estimates
        stderr_col : str
            Column name with standard errors
        title : str
            Title for the plot
        save : bool
            Whether to save the plot
        filename : str
            Filename to save the plot

        Returns:
        --------
        matplotlib.figure.Figure: The figure object
        """
        # Create figure
        fig, ax = plt.subplots()

        # Get model names and coefficients
        models = results_df[model_col].values
        coefs = results_df[coef_col].values

        # Calculate confidence intervals
        ci_low = coefs - 1.96 * results_df[stderr_col].values
        ci_high = coefs + 1.96 * results_df[stderr_col].values

        # Create y-positions
        y_pos = np.arange(len(models))

        # Horizontal bar plot
        ax.barh(y_pos, coefs, height=0.5,
                color=config.COLORS['highlight'], alpha=0.7)

        # Add error bars
        ax.errorbar(coefs, y_pos, xerr=np.vstack([coefs - ci_low, ci_high - coefs]),
                    fmt='none', ecolor='black', capsize=3)

        # Add zero reference line
        ax.axvline(x=0, color='gray', linestyle='--')

        # Set labels and title
        ax.set_yticks(y_pos)
        ax.set_yticklabels(models)
        ax.set_xlabel('Heterogeneous Effect')
        ax.set_title(title)

        # Add note about confidence intervals
        plt.figtext(0.5, 0.01, "Note: Error bars represent 95% confidence intervals.",
                    ha="center", fontsize=8, style='italic')

        plt.tight_layout()

        # Save the figure if requested
        if save:
            output_path = config.FIGURES_DIR / filename
            plt.savefig(output_path)
            print(f"Plot saved to {output_path}")

        return fig

    def plot_placebo_comparison(self, real_coef, real_stderr, placebo_coef, placebo_stderr,
                                save=True):
        """
        Plot comparison between real and placebo effects

        Parameters:
        -----------
        real_coef : float
            Real treatment effect coefficient
        real_stderr : float
            Standard error of real effect
        placebo_coef : float
            Placebo effect coefficient
        placebo_stderr : float
            Standard error of placebo effect
        save : bool
            Whether to save the plot

        Returns:
        --------
        matplotlib.figure.Figure: The figure object
        """
        # Create figure
        fig, ax = plt.subplots()

        # Set up data
        effects = ['Real Effect', 'Placebo Effect']
        coefs = [real_coef, placebo_coef]
        ci_low = [real_coef - 1.96 * real_stderr,
                  placebo_coef - 1.96 * placebo_stderr]
        ci_high = [real_coef + 1.96 * real_stderr,
                   placebo_coef + 1.96 * placebo_stderr]

        # Create y-positions
        y_pos = np.arange(len(effects))

        # Horizontal bar plot
        ax.barh(y_pos, coefs, height=0.5,
                color=[config.COLORS['treated'], config.COLORS['control']])

        # Add error bars
        for i, (coef, low, high) in enumerate(zip(coefs, ci_low, ci_high)):
            ax.errorbar(coef, y_pos[i], xerr=[[coef - low], [high - coef]],
                        fmt='none', ecolor='black', capsize=3)

        # Add zero reference line
        ax.axvline(x=0, color='gray', linestyle='--')

        # Set labels and title
        ax.set_yticks(y_pos)
        ax.set_yticklabels(effects)
        ax.set_xlabel('Coefficient Estimate')
        ax.set_title('Comparison of Real and Placebo Effects')

        # Add note about confidence intervals
        plt.figtext(0.5, 0.01, "Note: Error bars represent 95% confidence intervals.",
                    ha="center", fontsize=8, style='italic')

        plt.tight_layout()

        # Save the figure if requested
        if save:
            output_path = config.FIGURES_DIR / "placebo_comparison.png"
            plt.savefig(output_path)
            print(f"Plot saved to {output_path}")

        return fig

    def plot_matched_sample_balance(self, balance_df, var_col='Variable',
                                    before_col='Before_Std_Diff', after_col='After_Std_Diff',
                                    save=True):
        """
        Plot standardized differences before and after matching

        Parameters:
        -----------
        balance_df : pd.DataFrame
            DataFrame with balance statistics
        var_col : str
            Column name with variable names
        before_col : str
            Column name with standardized differences before matching
        after_col : str
            Column name with standardized differences after matching
        save : bool
            Whether to save the plot

        Returns:
        --------
        matplotlib.figure.Figure: The figure object
        """
        # Create figure
        fig, ax = plt.subplots()

        # Get variables and differences
        variables = balance_df[var_col].values
        before = balance_df[before_col].values
        after = balance_df[after_col].values

        # Create y-positions
        y_pos = np.arange(len(variables))

        # Line plot with markers
        ax.plot(before, y_pos, marker='o', linestyle='--',
                label='Before Matching', color=config.COLORS['control'])
        ax.plot(after, y_pos, marker='s', linestyle='-',
                label='After Matching', color=config.COLORS['treated'])

        # Add reference lines
        ax.axvline(x=0.1, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=-0.1, color='gray', linestyle='--', alpha=0.5)

        # Set labels and title
        ax.set_yticks(y_pos)
        ax.set_yticklabels(variables)
        ax.set_xlabel('Standardized Difference')
        ax.set_title('Covariate Balance Before and After Matching')
        ax.legend()

        # Add note
        plt.figtext(0.5, 0.01, "Note: Values within ±0.1 are typically considered balanced.",
                    ha="center", fontsize=8, style='italic')

        plt.tight_layout()

        # Save the figure if requested
        if save:
            output_path = config.FIGURES_DIR / "matching_balance.png"
            plt.savefig(output_path)
            print(f"Plot saved to {output_path}")

        return fig
