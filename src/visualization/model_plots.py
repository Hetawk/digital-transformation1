"""
Model visualization module for MSCI inclusion and digital transformation analysis
"""

import loader.config as config
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

    def plot_event_study(self, model, coef_prefix='time_', title='Event Study', save=True):
        """
        Create event study plot from regression results
        
        Parameters:
        -----------
        model : statsmodels.regression.linear_model.RegressionResults
            Event study model results
        coef_prefix : str
            Prefix for coefficient names
        title : str
            Plot title
        save : bool
            Whether to save the plot
            
        Returns:
        --------
        matplotlib.figure.Figure: Figure object
        """
        # Extract coefficients and standard errors
        coefs = []
        errors = []
        times = []
        
        # Handle special naming of negative time periods (time_m1, time_m2, etc)
        for param in model.params.index:
            if param.startswith(coef_prefix):
                if param.startswith(coef_prefix + 'm'):
                    # Extract the number after 'm'
                    t = -int(param[len(coef_prefix + 'm'):])
                else:
                    # Extract the number after the prefix
                    t = int(param[len(coef_prefix):])
                
                times.append(t)
                coefs.append(model.params[param])
                errors.append(model.bse[param])
        
        # Convert to numpy arrays and sort by time
        times = np.array(times)
        coefs = np.array(coefs)
        errors = np.array(errors)
        
        # Sort by time
        idx = np.argsort(times)
        times = times[idx]
        coefs = coefs[idx]
        errors = errors[idx]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot coefficients and confidence intervals
        ax.plot(times, coefs, 'o-', color='blue')
        ax.fill_between(times, coefs - 1.96 * errors, coefs + 1.96 * errors, 
                       alpha=0.2, color='blue')
        
        # Add horizontal line at y=0
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add vertical line at x=0 (treatment)
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        
        # Customize plot
        ax.set_xlabel('Time relative to MSCI inclusion')
        ax.set_ylabel('Coefficient estimate')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Save if requested
        if save:
            plt.tight_layout()
            fig.savefig(config.FIGURES_DIR / 'event_study.png', dpi=300)
        
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
