"""
Mechanism visualization module for MSCI inclusion and digital transformation analysis
"""

import config
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))


class MechanismPlots:
    def __init__(self, data=None):
        """
        Initialize the MechanismPlots

        Parameters:
        -----------
        data : pd.DataFrame or None
            The preprocessed dataset, optional for some plots
        """
        self.data = data
        self.setup_plot_style()

    def setup_plot_style(self):
        """Set up the matplotlib/seaborn plotting style"""
        sns.set_theme(style="whitegrid")
        plt.rcParams['figure.figsize'] = config.PLOT_FIGSIZE
        plt.rcParams['savefig.dpi'] = config.PLOT_DPI

    def plot_mechanism_effects(self, mechanism_df, mech_col='Mechanism', var_col='Variable',
                               effect_col='Interaction Effect', stderr_col='Std. Error',
                               title='Mechanisms of MSCI Inclusion Effect on Digital Transformation',
                               save=True):
        """
        Plot mechanism interaction effects

        Parameters:
        -----------
        mechanism_df : pd.DataFrame
            DataFrame with mechanism results
        mech_col : str
            Column name with mechanism type
        var_col : str
            Column name with variable names
        effect_col : str
            Column name with interaction effect estimates
        stderr_col : str
            Column name with standard errors
        title : str
            Title for the plot
        save : bool
            Whether to save the plot

        Returns:
        --------
        matplotlib.figure.Figure: The figure object
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(11, 7))

        # Group by mechanism type
        mechanisms = mechanism_df[mech_col].unique()

        # Define colors for different mechanism types
        mech_colors = {
            'financial_access': config.COLORS['treated'],
            'corporate_governance': config.COLORS['control'],
            'investor_scrutiny': config.COLORS['highlight']
        }

        # Default color if mechanism not in dictionary
        default_color = 'darkgray'

        # Track position for plotting
        all_labels = []
        all_effects = []
        all_stderr = []
        all_colors = []
        all_mechs = []

        # Process each mechanism type
        for i, mech in enumerate(mechanisms):
            # Filter data for this mechanism
            mech_data = mechanism_df[mechanism_df[mech_col] == mech].copy()

            # Get variables and effects
            variables = mech_data[var_col].values
            effects = mech_data[effect_col].values
            stderr = mech_data[stderr_col].values

            # Create clean labels
            labels = [
                f"{mech.replace('_', ' ').title()}: {var}" for var in variables]

            # Append to lists
            all_labels.extend(labels)
            all_effects.extend(effects)
            all_stderr.extend(stderr)
            all_colors.extend(
                [mech_colors.get(mech, default_color)] * len(variables))
            all_mechs.extend([mech] * len(variables))

        # Create y-positions
        y_pos = np.arange(len(all_labels))

        # Horizontal bar plot
        bars = ax.barh(y_pos, all_effects, height=0.6,
                       color=all_colors, alpha=0.7)

        # Add error bars
        for i, (effect, stderr) in enumerate(zip(all_effects, all_stderr)):
            ci_low = effect - 1.96 * stderr
            ci_high = effect + 1.96 * stderr
            ax.errorbar(effect, y_pos[i], xerr=[[effect - ci_low], [ci_high - effect]],
                        fmt='none', ecolor='black', capsize=3)

        # Add zero reference line
        ax.axvline(x=0, color='gray', linestyle='--')

        # Set labels and title
        ax.set_yticks(y_pos)
        ax.set_yticklabels(all_labels)
        ax.set_xlabel('Interaction Effect (MSCI × Mechanism)')
        ax.set_title(title)

        # Add legend for mechanism types
        legend_handles = []
        for mech in np.unique(all_mechs):
            color = mech_colors.get(mech, default_color)
            legend_handles.append(plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.7,
                                                label=mech.replace('_', ' ').title()))
        ax.legend(handles=legend_handles, loc='best')

        # Add note about confidence intervals
        plt.figtext(0.5, 0.01, "Note: Error bars represent 95% confidence intervals.",
                    ha="center", fontsize=8, style='italic')

        plt.tight_layout()

        # Save the figure if requested
        if save:
            output_path = config.FIGURES_DIR / "mechanism_effects.png"
            plt.savefig(output_path)
            print(f"Plot saved to {output_path}")

        return fig

    def plot_mechanism_heatmap(self, mechanism_df, mech_col='Mechanism', var_col='Variable',
                               effect_col='Interaction Effect', pval_col='p-value',
                               title='Mechanism Interaction Effects Heatmap', save=True):
        """
        Plot heatmap of mechanism interaction effects

        Parameters:
        -----------
        mechanism_df : pd.DataFrame
            DataFrame with mechanism results
        mech_col : str
            Column name with mechanism type
        var_col : str
            Column name with variable names
        effect_col : str
            Column name with interaction effect estimates
        pval_col : str
            Column name with p-values
        title : str
            Title for the plot
        save : bool
            Whether to save the plot

        Returns:
        --------
        matplotlib.figure.Figure: The figure object
        """
        # Create pivot table for heatmap
        pivot_data = mechanism_df.pivot_table(
            index=var_col,
            columns=mech_col,
            values=effect_col,
            aggfunc='first'
        ).fillna(0)

        # Create pivot table for p-values
        pval_data = mechanism_df.pivot_table(
            index=var_col,
            columns=mech_col,
            values=pval_col,
            aggfunc='first'
        ).fillna(1)

        # Create significance markers
        sig_markers = pval_data.copy()
        sig_markers = sig_markers.applymap(
            lambda x: '***' if x < 0.01 else ('**' if x < 0.05 else ('*' if x < 0.1 else '')))

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Create heatmap
        sns.heatmap(pivot_data, cmap="coolwarm", center=0,
                    annot=True, fmt=".3f", ax=ax)

        # Add significance markers
        for i in range(pivot_data.shape[0]):
            for j in range(pivot_data.shape[1]):
                text = ax.texts[i * pivot_data.shape[1] + j]
                text.set_text(f"{text.get_text()}{sig_markers.iloc[i, j]}")

        ax.set_title(title)

        # Add note about significance levels
        plt.figtext(0.5, 0.01, "Note: * p<0.1, ** p<0.05, *** p<0.01",
                    ha="center", fontsize=8, style='italic')

        plt.tight_layout()

        # Save the figure if requested
        if save:
            output_path = config.FIGURES_DIR / "mechanism_heatmap.png"
            plt.savefig(output_path)
            print(f"Plot saved to {output_path}")

        return fig

    def plot_mediation_analysis(self, direct_effect, indirect_effects,
                                title='Mediation Analysis: Direct and Indirect Effects',
                                save=True):
        """
        Plot mediation analysis results

        Parameters:
        -----------
        direct_effect : tuple
            Tuple of (direct effect estimate, standard error)
        indirect_effects : dict
            Dictionary of mediators with tuples (indirect effect estimate, standard error)
        title : str
            Title for the plot
        save : bool
            Whether to save the plot

        Returns:
        --------
        matplotlib.figure.Figure: The figure object
        """
        # Create figure
        fig, ax = plt.subplots()

        # Prepare data
        labels = ['Direct Effect'] + list(indirect_effects.keys())
        effects = [direct_effect[0]] + [effect[0]
                                        for effect in indirect_effects.values()]
        stderr = [direct_effect[1]] + [effect[1]
                                       for effect in indirect_effects.values()]

        # Calculate confidence intervals
        ci_low = [e - 1.96 * se for e, se in zip(effects, stderr)]
        ci_high = [e + 1.96 * se for e, se in zip(effects, stderr)]

        # Create y-positions
        y_pos = np.arange(len(labels))

        # Horizontal bar plot with different color for direct effect
        colors = [config.COLORS['highlight']] + \
            [config.COLORS['treated']] * len(indirect_effects)
        ax.barh(y_pos, effects, height=0.6, color=colors, alpha=0.7)

        # Add error bars
        for i, (effect, low, high) in enumerate(zip(effects, ci_low, ci_high)):
            ax.errorbar(effect, y_pos[i], xerr=[[effect - low], [high - effect]],
                        fmt='none', ecolor='black', capsize=3)

        # Add zero reference line
        ax.axvline(x=0, color='gray', linestyle='--')

        # Set labels and title
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel('Effect Size')
        ax.set_title(title)

        # Add legend
        ax.legend([plt.Rectangle((0, 0), 1, 1, color=config.COLORS['highlight'], alpha=0.7),
                  plt.Rectangle((0, 0), 1, 1, color=config.COLORS['treated'], alpha=0.7)],
                  ['Direct Effect', 'Indirect Effect'], loc='best')

        # Add note about confidence intervals
        plt.figtext(0.5, 0.01, "Note: Error bars represent 95% confidence intervals.",
                    ha="center", fontsize=8, style='italic')

        plt.tight_layout()

        # Save the figure if requested
        if save:
            output_path = config.FIGURES_DIR / "mediation_analysis.png"
            plt.savefig(output_path)
            print(f"Plot saved to {output_path}")

        return fig

    def plot_subgroup_mechanisms(self, mech_data, title='Mechanism Effects by Subgroup',
                                 save=True):
        """
        Plot mechanism effects by subgroup

        Parameters:
        -----------
        mech_data : dict
            Dictionary with subgroups as keys and DataFrames with mechanism results as values
        title : str
            Title for the plot
        save : bool
            Whether to save the plot

        Returns:
        --------
        matplotlib.figure.Figure: The figure object
        """
        # Identify common mechanisms across subgroups
        all_mechanisms = set()
        for subgroup, df in mech_data.items():
            if 'Mechanism' in df.columns and 'Variable' in df.columns:
                for _, row in df.iterrows():
                    all_mechanisms.add(
                        f"{row['Mechanism']}: {row['Variable']}")

        # Convert to list and sort
        all_mechanisms = sorted(list(all_mechanisms))

        # Create figure with subplots for each subgroup
        n_subgroups = len(mech_data)
        fig, axes = plt.subplots(n_subgroups, 1, figsize=(10, 4 * n_subgroups),
                                 sharex=True)

        # Handle single subplot case
        if n_subgroups == 1:
            axes = [axes]

        # Plot each subgroup
        for i, (subgroup, df) in enumerate(mech_data.items()):
            ax = axes[i]

            # Create dictionary to look up effects and stderr
            effect_dict = {}
            stderr_dict = {}
            for _, row in df.iterrows():
                if 'Mechanism' in df.columns and 'Variable' in df.columns:
                    key = f"{row['Mechanism']}: {row['Variable']}"
                    effect_dict[key] = row.get('Interaction Effect', 0)
                    stderr_dict[key] = row.get('Std. Error', 0)

            # Prepare data for plotting
            effects = [effect_dict.get(mech, 0) for mech in all_mechanisms]
            stderr = [stderr_dict.get(mech, 0) for mech in all_mechanisms]
            y_pos = np.arange(len(all_mechanisms))

            # Horizontal bar plot
            ax.barh(y_pos, effects, height=0.6,
                    color=config.COLORS['treated'], alpha=0.7)

            # Add error bars
            for j, (effect, se) in enumerate(zip(effects, stderr)):
                if se > 0:  # Only add error bars if stderr is available
                    ci_low = effect - 1.96 * se
                    ci_high = effect + 1.96 * se
                    ax.errorbar(effect, y_pos[j], xerr=[[effect - ci_low], [ci_high - effect]],
                                fmt='none', ecolor='black', capsize=3)

            # Add zero reference line
            ax.axvline(x=0, color='gray', linestyle='--')

            # Set labels
            ax.set_yticks(y_pos)
            ax.set_yticklabels(all_mechanisms)
            ax.set_ylabel('')
            ax.set_title(f"Subgroup: {subgroup}")

            if i == n_subgroups - 1:  # Only add xlabel to bottom subplot
                ax.set_xlabel('Interaction Effect')

        # Add main title
        fig.suptitle(title, fontsize=16, y=1.02)

        # Add note about confidence intervals
        plt.figtext(0.5, 0.01, "Note: Error bars represent 95% confidence intervals.",
                    ha="center", fontsize=8, style='italic')

        plt.tight_layout()

        # Save the figure if requested
        if save:
            output_path = config.FIGURES_DIR / "subgroup_mechanisms.png"
            plt.savefig(output_path)
            print(f"Plot saved to {output_path}")

        return fig
