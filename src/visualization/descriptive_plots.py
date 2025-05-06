"""
Descriptive plots for MSCI inclusion and digital transformation analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import loader.config as config
import logging
import sys
import traceback
from pathlib import Path

# Set up logger
logger = logging.getLogger('digital_transformation')
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', 
                                 datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class DescriptivePlots:
    def __init__(self, data, balance_df=None):
        """
        Initialize descriptive plots class
        
        Parameters:
        -----------
        data : pd.DataFrame
            The dataset for visualization
        balance_df : pd.DataFrame or None
            Balance check results from check_balance()
        """
        self.data = data
        self.balance_df = balance_df
        
        # Define directories for saving figures
        self.figures_dir = config.FIGURES_DIR
        
        # Define primary variables for distribution plots
        self.primary_vars = [
            'Digital_transformationA', 
            'Digital_transformationB', 
            'age', 
            'TFP_OP', 
            'SA_index', 
            'WW_index', 
            'F050501B'
        ]
        
        # Check if we have a staggered adoption design
        self.staggered_adoption = False
        if balance_df is not None and hasattr(balance_df, 'attrs'):
            self.staggered_adoption = balance_df.attrs.get('staggered_adoption', False)

    def plot_dt_trends(self, variables=None, save=True):
        """
        Plot digital transformation trends over time

        Parameters:
        -----------
        variables : list or None
            Variables to plot, defaults to DT measures
        save : bool
            Whether to save the plot

        Returns:
        --------
        dict: Dictionary of figure objects
        """
        if variables is None:
            variables = config.DT_MEASURES

        figures = {}

        for var in variables:
            if var not in self.data.columns:
                logger.warning(f"Variable {var} not found in dataset, skipping trend plot")
                continue

            try:
                logger.info(f"Plotting trend for {var}")
                
                # Create figure
                fig, ax = plt.subplots(figsize=config.PLOT_FIGSIZE)
                
                # Calculate means by year and treatment status
                trend_data = self.data.groupby(['year', 'Treat'])[var].mean().reset_index()
                
                # Convert to numeric if not already
                trend_data['year'] = pd.to_numeric(trend_data['year'], errors='coerce')
                trend_data = trend_data.dropna(subset=['year'])
                
                # Convert specifically for plotting to avoid multi-dimensional indexing error
                for treat_val, label, color in zip(
                    [0, 1],
                    ['Control', 'Treated'],
                    [config.COLORS['control'], config.COLORS['treated']]
                ):
                    group_data = trend_data[trend_data['Treat'] == treat_val]
                    
                    # Convert to numpy explicitly before plotting
                    x_vals = group_data['year'].to_numpy()
                    y_vals = group_data[var].to_numpy()
                    
                    if len(x_vals) > 0:  # Only plot if we have data
                        ax.plot(x_vals, y_vals, marker='o', linestyle='-', 
                               color=color, label=label)
                
                # Add vertical line at treatment year
                ax.axvline(x=float(config.TREATMENT_YEAR), color='r', 
                          linestyle='--', alpha=0.5,
                          label='Treatment Year')
                
                # Customize plot
                ax.set_xlabel('Year')
                ax.set_ylabel(var)
                ax.set_title(f'{var} Trend by Treatment Status')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Save if requested
                if save:
                    plt.tight_layout()
                    filename = f"trend_{var}.png"
                    fig.savefig(config.FIGURES_DIR / filename)
                    logger.info(f"Plot saved to {config.FIGURES_DIR / filename}")
                
                figures[var] = fig
                
            except Exception as e:
                logger.error(f"Error plotting DT trends for {var}: {str(e)}")
                
                # Try one more time with alternative approach
                try:
                    logger.info(f"Retrying plot for {var} with alternative approach")
                    
                    fig, ax = plt.subplots(figsize=config.PLOT_FIGSIZE)
                    
                    # Different approach using simpler aggregation
                    for treat_val, label, color in zip(
                        [0, 1],
                        ['Control', 'Treated'],
                        [config.COLORS['control'], config.COLORS['treated']]
                    ):
                        # Filter and convert to numpy immediately 
                        mask = self.data['Treat'] == treat_val
                        subset = self.data[mask]
                        
                        # Group and convert to numpy arrays
                        grouped = subset.groupby('year')[var].mean().reset_index()
                        x_vals = grouped['year'].to_numpy()
                        y_vals = grouped[var].to_numpy()
                        
                        if len(x_vals) > 0:
                            ax.plot(x_vals, y_vals, marker='o', linestyle='-',
                                  color=color, label=label)
                    
                    # Add vertical line at treatment year
                    ax.axvline(x=float(config.TREATMENT_YEAR), color='r', 
                              linestyle='--', alpha=0.5,
                              label='Treatment Year')
                    
                    # Customize plot
                    ax.set_xlabel('Year')
                    ax.set_ylabel(var)
                    ax.set_title(f'{var} Trend by Treatment Status')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    # Save if requested
                    if save:
                        plt.tight_layout()
                        filename = f"trend_{var}_alternative.png"
                        fig.savefig(config.FIGURES_DIR / filename)
                        logger.info(f"Alternative plot saved to {config.FIGURES_DIR / filename}")
                    
                    figures[var] = fig
                    
                except Exception as e2:
                    logger.error(f"Second attempt at plotting {var} failed: {str(e2)}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    # We'll continue with other plots rather than exiting

        return figures

    def plot_variable_distributions(self, vars_to_plot, by_treatment=True, title=None, save=True):
        """
        Plot distributions of variables, optionally by treatment status.
        Useful for staggered adoption designs where conventional balance tests aren't possible.
        
        Parameters:
        -----------
        vars_to_plot : list
            List of variable names to plot
        by_treatment : bool, default=True
            Whether to separate distributions by treatment status
        title : str or None
            Optional title for the figure
        save : bool, default=True
            Whether to save the figure to file
            
        Returns:
        --------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Make sure we have some valid variables
        valid_vars = [var for var in vars_to_plot if var in self.data.columns]
        if not valid_vars:
            print("No valid variables for distribution plots")
            return None
            
        fig, axes = plt.subplots(len(valid_vars), 1, figsize=(10, 3*len(valid_vars)))
        if len(valid_vars) == 1:
            axes = [axes]  # Ensure axes is always a list for consistent indexing
        
        for i, var in enumerate(valid_vars):
            # Create a clean subset of data for this variable
            data_to_plot = self.data[[var]].copy().dropna()
            
            if by_treatment and 'Treat' in self.data.columns:
                data_to_plot['Treat'] = self.data['Treat']
                
                # Create violin plot with treatment groups
                sns.violinplot(x='Treat', y=var, data=data_to_plot, ax=axes[i])
                axes[i].set_title(f"Distribution of {var} by Treatment Status")
                axes[i].set_xlabel("Treatment Status")
            else:
                # Create distribution plot without treatment groups
                sns.histplot(data_to_plot[var], kde=True, ax=axes[i])
                axes[i].set_title(f"Distribution of {var}")
            
            axes[i].set_ylabel(var)
        
        plt.tight_layout()
        
        if title:
            fig.suptitle(title, fontsize=16, y=1.02)
        
        if save:
            filename = "variable_distributions.png"
            filepath = self.figures_dir / filename
            plt.savefig(filepath, bbox_inches='tight', dpi=300)
            print(f"Saved distribution plot to {filepath}")
        
        return fig

    def plot_distribution(self, variable, bins=30, save=True):
        """
        Plot distribution of a variable
        
        Parameters:
        -----------
        variable : str
            Name of the variable to plot
        bins : int, default 30
            Number of bins for histogram
        save : bool, default True
            Whether to save the plot
            
        Returns:
        --------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        
        if variable not in self.data.columns:
            raise ValueError(f"Variable {variable} not found in dataset")
            
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Convert to numpy array before manipulation to avoid pandas indexing issues
        data_values = self.data[variable].dropna().to_numpy().flatten()
        
        if len(data_values) == 0:
            logger.warning(f"No non-NA values for {variable}")
            ax.text(0.5, 0.5, f"No data available for {variable}", 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=14)
            return fig
        
        # Plot distribution
        sns.histplot(data_values, kde=True, bins=bins, ax=ax)
        
        # Add vertical line for mean
        mean_val = np.mean(data_values)
        ax.axvline(mean_val, color='red', linestyle='--', 
                   label=f'Mean: {mean_val:.2f}')
        
        # Add vertical line for median
        median_val = np.median(data_values)
        ax.axvline(median_val, color='green', linestyle='-', 
                 label=f'Median: {median_val:.2f}')
        
        # Add title and labels
        ax.set_title(f'Distribution of {variable}')
        ax.set_xlabel(variable)
        ax.set_ylabel('Frequency')
        ax.legend()
        
        # Save if requested
        if save:
            filename = f"{variable}_distribution.png"
            filepath = self.figures_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved distribution plot to {filepath}")
            
        return fig
        
    def plot_time_trends(self, variable, group_var='Treat', save=True):
        """
        Plot variable trends over time, grouped by treatment status
        
        Parameters:
        -----------
        variable : str
            Name of the variable to plot
        group_var : str
            Grouping variable (usually treatment status)
        save : bool
            Whether to save the plot
            
        Returns:
        --------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt
        
        if variable not in self.data.columns:
            raise ValueError(f"Variable {variable} not found in dataset")
        if group_var not in self.data.columns:
            raise ValueError(f"Group variable {group_var} not found in dataset")
        if 'year' not in self.data.columns:
            raise ValueError("Year variable not found in dataset")
            
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Calculate means by year and treatment status
        # Convert data to numpy arrays to avoid indexing issues
        grouped_data = []
        
        for treat_val in sorted(self.data[group_var].dropna().unique()):
            # Filter data for this treatment group
            mask = self.data[group_var] == treat_val
            group_data = self.data[mask]
            
            # Get years and calculate mean for each year
            years = []
            means = []
            
            for year in sorted(group_data['year'].dropna().unique()):
                year_mask = group_data['year'] == year
                year_values = group_data.loc[year_mask, variable].dropna()
                
                if len(year_values) > 0:
                    years.append(year)
                    means.append(year_values.mean())
            
            # Plot if we have data
            if years and means:
                label = f"Treated" if treat_val == 1 else "Control"
                color = "blue" if treat_val == 1 else "gray"
                ax.plot(years, means, marker='o', linestyle='-', color=color, label=label)
        
        # Add vertical line at treatment year if configured
        if hasattr(config, 'TREATMENT_YEAR'):
            treatment_year = float(config.TREATMENT_YEAR)
            ax.axvline(x=treatment_year, color='red', linestyle='--', alpha=0.7, 
                      label=f'Treatment Year ({int(treatment_year)})')
        
        # Customize plot
        ax.set_xlabel('Year')
        ax.set_ylabel(variable)
        ax.set_title(f'{variable} Trends Over Time by Treatment Status')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save if requested
        if save:
            filename = f"{variable}_time_trends.png"
            filepath = self.figures_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved time trends plot to {filepath}")
            
        return fig
        
    def plot_treatment_effect(self, outcome_var, save=True):
        """
        Plot treatment effect visualization
        
        Parameters:
        -----------
        outcome_var : str
            Name of the outcome variable
        save : bool
            Whether to save the plot
            
        Returns:
        --------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt
        
        if outcome_var not in self.data.columns:
            raise ValueError(f"Outcome variable {outcome_var} not found in dataset")
        if 'Treat' not in self.data.columns:
            raise ValueError("Treatment variable 'Treat' not found in dataset")
        if 'Post' not in self.data.columns:
            raise ValueError("Post variable 'Post' not found in dataset")
            
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate means for each treatment-period group
        means = {}
        for treat in [0, 1]:
            for post in [0, 1]:
                mask = (self.data['Treat'] == treat) & (self.data['Post'] == post)
                values = self.data.loc[mask, outcome_var].dropna().to_numpy()
                if len(values) > 0:
                    means[(treat, post)] = np.mean(values)
                else:
                    means[(treat, post)] = np.nan
        
        # Handle staggered adoption case where (1,0) doesn't exist
        if np.isnan(means.get((1, 0), np.nan)):
            print("Staggered adoption detected. Modifying treatment effect visualization.")
            
            # Set width for bars
            width = 0.35
            
            # Plot only post-period comparison
            ax.bar([0], [means.get((0, 1), 0)], width, label='Control (Post)')
            ax.bar([1], [means.get((1, 1), 0)], width, label='Treated (Post)')
            
            ax.set_ylabel(outcome_var)
            ax.set_title(f'Post-Period Comparison of {outcome_var}')
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Control', 'Treated'])
            ax.legend()
            
        else:
            # Standard DiD case with all groups
            width = 0.35
            x = np.arange(2)  # Pre and Post
            
            # Plot control group
            ax.bar(x - width/2, [means.get((0, 0), 0), means.get((0, 1), 0)], 
                 width, label='Control')
            
            # Plot treatment group
            ax.bar(x + width/2, [means.get((1, 0), 0), means.get((1, 1), 0)], 
                 width, label='Treated')
            
            ax.set_ylabel(outcome_var)
            ax.set_title(f'Treatment Effect on {outcome_var}')
            ax.set_xticks(x)
            ax.set_xticklabels(['Pre', 'Post'])
            ax.legend()
        
        # Save if requested
        if save:
            filename = f"{outcome_var}_treatment_effect.png"
            filepath = self.figures_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved treatment effect plot to {filepath}")
            
        return fig

    def plot_correlation_matrix(self, variables=None, save=True):
        """
        Plot correlation matrix
        
        Parameters:
        -----------
        variables : list or None
            List of variables to include, if None use all
        save : bool, default True
            Whether to save the plot
            
        Returns:
        --------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # If no variables specified, use numeric columns
        if variables is None:
            # Get numeric columns
            numeric_cols = self.data.select_dtypes(include=['number']).columns.tolist()
            # Filter out id columns and other non-meaningful columns
            variables = [col for col in numeric_cols 
                        if 'id' not in col.lower() and 'code' not in col.lower()]
            # Limit to 10 variables to avoid cluttered plot
            variables = variables[:10]
        
        # Calculate correlation
        corr_data = self.data[variables].corr()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot heatmap
        sns.heatmap(corr_data, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                   ax=ax, fmt='.2f', linewidths=0.5)
        
        # Add title
        ax.set_title('Correlation Matrix')
        
        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right')
        
        # Save if requested
        if save:
            filename = "correlation_matrix.png"
            filepath = self.figures_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved correlation matrix to {filepath}")
            
        return fig

    def create_all_plots(self, save=True):
        """
        Create all descriptive plots

        Parameters:
        -----------
        save : bool
            Whether to save plots

        Returns:
        --------
        dict: Dictionary of all figure objects
        """
        logger.info("Generating all descriptive plots...")
        all_figures = {}

        # Digital transformation trends
        dt_trends = self.plot_dt_trends(save=save)
        all_figures.update(dt_trends)

        # Variable distributions
        distributions = self.plot_variable_distributions(save=save)
        all_figures.update(distributions)

        # Correlation heatmap
        heatmap = self.plot_correlation_heatmap(save=save)
        if heatmap is not None:
            all_figures['heatmap'] = heatmap

        # Treatment distribution
        treat_dist = self.plot_treatment_distribution(save=save)
        if treat_dist is not None:
            all_figures['treatment_distribution'] = treat_dist

        logger.info(f"Created {len(all_figures)} descriptive visualizations.")
        return all_figures

    def plot_all_descriptives(self, save=True):
        """
        Generate all descriptive plots in a single call.
        
        Parameters:
        -----------
        save : bool, default=True
            Whether to save the plots to file
            
        Returns:
        --------
        dict : Dictionary of generated figure objects
        """
        figures = {}
        
        # Create distribution plots
        print("Generating distribution plots...")
        for var in self.primary_vars:
            if var in self.data.columns:
                try:
                    figures[f'{var}_dist'] = self.plot_distribution(var, save=save)
                except Exception as e:
                    print(f"Error creating distribution plot for {var}: {e}")
        
        # Create time trend plots
        print("Generating time trend plots...")
        try:
            if 'Digital_transformationA' in self.data.columns and 'Treat' in self.data.columns and 'year' in self.data.columns:
                figures['dt_time_trends'] = self.plot_time_trends('Digital_transformationA', group_var='Treat', save=save)
            else:
                print("Missing required variables for time trend plot")
        except Exception as e:
            print(f"Error creating time trend plot: {e}")
            
        # Create treatment effect visualization
        print("Generating treatment effect visualization...")
        try:
            if 'Digital_transformationA' in self.data.columns and 'Treat' in self.data.columns and 'Post' in self.data.columns:
                figures['treat_effect'] = self.plot_treatment_effect('Digital_transformationA', save=save)
            else:
                print("Missing required variables for treatment effect plot")
        except Exception as e:
            print(f"Error creating treatment effect plot: {e}")
        
        # Create balance visualization if available
        print("Generating balance visualization...")
        if hasattr(self, 'balance_df') and self.balance_df is not None:
            try:
                # Check if we're in a staggered adoption case
                if self.staggered_adoption:
                    print("Staggered adoption design detected. Creating alternative balance visualization...")
                    key_vars = ['age', 'TFP_OP', 'F050501B']
                    valid_vars = [var for var in key_vars if var in self.data.columns]
                    if valid_vars:
                        figures['balance'] = self.plot_variable_distributions(
                            valid_vars, 
                            by_treatment=False, 
                            title="Pre-Treatment Variable Distributions",
                            save=save
                        )
                else:
                    # Standard balance plot if not staggered adoption
                    std_cols = [col for col in self.balance_df.columns if 'Std' in col]
                    if std_cols:
                        figures['balance'] = self.plot_balance(
                            self.balance_df, 
                            var_col='Variable', 
                            std_diff_col=std_cols[0], 
                            save=save
                        )
            except Exception as e:
                print(f"Error creating balance plot: {e}")
        
        # Create correlation heatmap
        print("Generating correlation heatmap...")
        try:
            corr_vars = [var for var in self.primary_vars if var in self.data.columns]
            if len(corr_vars) >= 2:
                figures['correlation'] = self.plot_correlation_matrix(
                    variables=corr_vars[:5],  # Limit to first 5 to avoid too large heatmap
                    save=save
                )
            else:
                print("Not enough variables for correlation heatmap")
        except Exception as e:
            print(f"Error creating correlation heatmap: {e}")
        
        print(f"Generated {len(figures)} descriptive plots")
        return figures
