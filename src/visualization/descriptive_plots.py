"""
Descriptive visualizations for MSCI inclusion and digital transformation analysis
"""

import config
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))


class DescriptivePlots:
    def __init__(self, data):
        """
        Initialize the DescriptivePlots

        Parameters:
        -----------
        data : pd.DataFrame
            The preprocessed dataset
        """
        self.data = data
        self.setup_plot_style()

    def setup_plot_style(self):
        """Set up the matplotlib/seaborn plotting style"""
        sns.set_theme(style="whitegrid")
        plt.rcParams['figure.figsize'] = config.PLOT_FIGSIZE
        plt.rcParams['savefig.dpi'] = config.PLOT_DPI

    def plot_dt_trends(self, dt_var="Digital_transformationA", save=True):
        """
        Plot digital transformation trends over time by treatment status

        Parameters:
        -----------
        dt_var : str
            Digital transformation variable to plot
        save : bool
            Whether to save the plot

        Returns:
        --------
        matplotlib.figure.Figure: The figure object
        """
        if dt_var not in self.data.columns:
            raise ValueError(f"Variable {dt_var} not found in dataset")

        # Create a copy of data for aggregation
        plot_data = self.data.copy()

        # Calculate mean, std error, and count by year and treatment
        agg_data = (plot_data.groupby(['year', 'Treat'])
                    .agg({dt_var: ['mean', 'std', 'count']})
                    .reset_index())

        # Flatten the hierarchical column names
        agg_data.columns = ['year', 'Treat', 'mean', 'std', 'count']

        # Calculate confidence intervals
        agg_data['ci'] = 1.96 * agg_data['std'] / np.sqrt(agg_data['count'])
        agg_data['lower'] = agg_data['mean'] - agg_data['ci']
        agg_data['upper'] = agg_data['mean'] + agg_data['ci']

        # Create the plot
        fig, ax = plt.subplots()

        # Plot treated and control separately
        for treat, label, color, marker in zip(
            [1, 0],
            ['MSCI Included Firms (Ever)', 'Non-MSCI Firms'],
            [config.COLORS['treated'], config.COLORS['control']],
            ['o', 's']
        ):
            group_data = agg_data[agg_data['Treat'] == treat]
            ax.plot(group_data['year'], group_data['mean'], marker=marker,
                    label=label, color=color)
            ax.fill_between(group_data['year'], group_data['lower'], group_data['upper'],
                            alpha=0.2, color=color)

        # Add vertical line at treatment year
        ax.axvline(x=config.TREATMENT_YEAR, linestyle='--', color='gray')

        # Add labels and legend
        ax.set_xlabel('Year')
        ax.set_ylabel(f'Mean {dt_var}')
        ax.set_title(f'Digital Transformation Trends Over Time by MSCI Status')
        ax.legend(loc='best')

        # Add note
        plt.figtext(0.5, 0.01, "Note: Shaded areas represent 95% confidence intervals.",
                    ha="center", fontsize=8, style='italic')

        plt.tight_layout()

        # Save the figure if requested
        if save:
            output_path = config.FIGURES_DIR / f"dt_trends_{dt_var}.png"
            plt.savefig(output_path)
            print(f"Plot saved to {output_path}")

        return fig

    def plot_variable_distributions(self, variables=None, by_treatment=True, save=True):
        """
        Plot distributions of key variables, optionally by treatment status

        Parameters:
        -----------
        variables : list or None
            Variables to plot, defaults to DT measures
        by_treatment : bool
            Whether to split by treatment status
        save : bool
            Whether to save the plot

        Returns:
        --------
        dict: Dictionary of figure objects
        """
        if variables is None:
            variables = config.DT_MEASURES

        # Filter to variables that exist in the dataset
        variables = [var for var in variables if var in self.data.columns]

        if not variables:
            raise ValueError("No valid variables found in dataset")

        figures = {}

        for var in variables:
            fig, ax = plt.subplots()

            if by_treatment:
                # Plot separate distributions by treatment status
                for treat, label, color in zip(
                    [0, 1],
                    ['Non-MSCI Firms', 'MSCI Included Firms (Ever)'],
                    [config.COLORS['control'], config.COLORS['treated']]
                ):
                    group_data = self.data[self.data['Treat']
                                           == treat][var].dropna()
                    sns.kdeplot(group_data, label=label, ax=ax,
                                color=color, fill=True, alpha=0.3)
            else:
                # Plot single distribution
                sns.kdeplot(self.data[var].dropna(), ax=ax,
                            color=config.COLORS['highlight'], fill=True)

            # Add labels
            ax.set_xlabel(var)
            ax.set_ylabel('Density')
            ax.set_title(f'Distribution of {var}')

            if by_treatment:
                ax.legend(loc='best')

            plt.tight_layout()

            # Save the figure if requested
            if save:
                suffix = "_by_treatment" if by_treatment else ""
                output_path = config.FIGURES_DIR / \
                    f"distribution_{var}{suffix}.png"
                plt.savefig(output_path)
                print(f"Plot saved to {output_path}")

            figures[var] = fig

        return figures

    def plot_correlation_heatmap(self, variables=None, save=True):
        """
        Plot correlation heatmap for key variables

        Parameters:
        -----------
        variables : list or None
            Variables to include, defaults to DT measures and key controls
        save : bool
            Whether to save the plot

        Returns:
        --------
        matplotlib.figure.Figure: The figure object
        """
        if variables is None:
            variables = (config.DT_MEASURES[:2] +
                         ['MSCI', 'MSCI_clean', 'TreatPost'] +
                         config.CONTROL_VARS[:4])

        # Filter to variables that exist in the dataset
        variables = [var for var in variables if var in self.data.columns]

        if not variables:
            raise ValueError("No valid variables found in dataset")

        # Calculate correlation matrix
        corr_matrix = self.data[variables].corr()

        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))

        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f",
                    cmap="coolwarm", square=True, ax=ax, cbar_kws={"shrink": .8})

        ax.set_title("Correlation Heatmap")
        plt.tight_layout()

        # Save the figure if requested
        if save:
            output_path = config.FIGURES_DIR / "correlation_heatmap.png"
            plt.savefig(output_path)
            print(f"Plot saved to {output_path}")

        return fig

    def plot_treatment_distribution(self, save=True):
        """
        Plot treatment distribution over time

        Parameters:
        -----------
        save : bool
            Whether to save the plot

        Returns:
        --------
        matplotlib.figure.Figure: The figure object
        """
        if 'year' not in self.data.columns or 'MSCI' not in self.data.columns:
            raise ValueError(
                "Required variables (year, MSCI) not found in dataset")

        # Count MSCI firms by year
        msci_counts = self.data.groupby(['year'])['MSCI'].sum().reset_index()
        msci_counts.rename(columns={'MSCI': 'MSCI_count'}, inplace=True)

        # Count total firms by year
        total_counts = self.data.groupby(
            ['year']).size().reset_index(name='total')

        # Merge counts
        count_data = pd.merge(msci_counts, total_counts, on='year')

        # Calculate percentage
        count_data['msci_percentage'] = 100 * \
            count_data['MSCI_count'] / count_data['total']

        # Create plot
        fig, ax1 = plt.subplots()

        # Plot count on left axis
        ax1.bar(count_data['year'], count_data['MSCI_count'], color=config.COLORS['treated'],
                alpha=0.7, label='MSCI Included Firms (Count)')
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Number of MSCI Included Firms')

        # Create right axis for percentage
        ax2 = ax1.twinx()
        ax2.plot(count_data['year'], count_data['msci_percentage'], color=config.COLORS['highlight'],
                 marker='o', linestyle='-', linewidth=2, label='MSCI Percentage')
        ax2.set_ylabel('Percentage of Firms (%)')

        # Add vertical line at treatment year
        ax1.axvline(x=config.TREATMENT_YEAR, linestyle='--', color='gray')

        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        ax1.set_title('MSCI Inclusion Distribution Over Time')

        plt.tight_layout()

        # Save the figure if requested
        if save:
            output_path = config.FIGURES_DIR / "msci_inclusion_distribution.png"
            plt.savefig(output_path)
            print(f"Plot saved to {output_path}")

        return fig

    def plot_all_descriptives(self, save=True):
        """
        Generate all descriptive plots

        Parameters:
        -----------
        save : bool
            Whether to save the plots

        Returns:
        --------
        dict: Dictionary of figure objects
        """
        figures = {}

        # Plot DT trends for each DT measure
        for dt_var in [var for var in config.DT_MEASURES if var in self.data.columns]:
            try:
                fig = self.plot_dt_trends(dt_var, save)
                figures[f'dt_trends_{dt_var}'] = fig
            except Exception as e:
                print(f"Error plotting DT trends for {dt_var}: {e}")

        # Plot variable distributions
        try:
            dist_figs = self.plot_variable_distributions(save=save)
            figures.update(dist_figs)
        except Exception as e:
            print(f"Error plotting variable distributions: {e}")

        # Plot correlation heatmap
        try:
            corr_fig = self.plot_correlation_heatmap(save=save)
            figures['correlation_heatmap'] = corr_fig
        except Exception as e:
            print(f"Error plotting correlation heatmap: {e}")

        # Plot treatment distribution
        try:
            dist_fig = self.plot_treatment_distribution(save=save)
            figures['treatment_distribution'] = dist_fig
        except Exception as e:
            print(f"Error plotting treatment distribution: {e}")

        return figures
