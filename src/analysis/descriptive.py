"""
Descriptive statistics for MSCI inclusion and digital transformation analysis
"""

from src.utils import save_results_to_file, check_balance
import config
import pandas as pd
import numpy as np
import os
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))


class DescriptiveAnalysis:
    def __init__(self, data):
        """
        Initialize the DescriptiveAnalysis

        Parameters:
        -----------
        data : pd.DataFrame
            The preprocessed dataset
        """
        self.data = data

    def summarize_variables(self, variables=None, by_group=None):
        """
        Generate summary statistics for variables

        Parameters:
        -----------
        variables : list or None
            List of variables to summarize, default to DT measures and controls
        by_group : str or None
            Variable to group by (e.g., 'Treat')

        Returns:
        --------
        pd.DataFrame : Summary statistics
        """
        if variables is None:
            variables = config.DT_MEASURES + config.CONTROL_VARS

        # Filter to variables that exist in the dataset
        variables = [var for var in variables if var in self.data.columns]

        if not variables:
            raise ValueError("No valid variables found for summary")

        if by_group is None:
            # Simple summary stats without grouping
            summary = self.data[variables].describe().T

            # Add additional statistics
            summary['median'] = self.data[variables].median()
            summary['skew'] = self.data[variables].skew()
            summary['kurtosis'] = self.data[variables].kurtosis()

            return summary
        else:
            # Group by specified variable
            if by_group not in self.data.columns:
                raise ValueError(
                    f"Group variable {by_group} not found in dataset")

            # Get unique groups
            groups = self.data[by_group].unique()

            # Create summary for each group
            result = {}
            for group in groups:
                group_data = self.data[self.data[by_group] == group]
                summary = group_data[variables].describe().T

                # Add additional statistics
                summary['median'] = group_data[variables].median()

                result[f"{by_group}={group}"] = summary

            return result

    def correlation_analysis(self, variables=None):
        """
        Perform correlation analysis

        Parameters:
        -----------
        variables : list or None
            List of variables, default to DT measures and key variables

        Returns:
        --------
        pd.DataFrame : Correlation matrix
        """
        if variables is None:
            variables = config.DT_MEASURES[:2] + ['MSCI',
                                                  'MSCI_clean'] + config.CONTROL_VARS[:3]

        # Filter to variables that exist in the dataset
        variables = [var for var in variables if var in self.data.columns]

        if not variables:
            raise ValueError(
                "No valid variables found for correlation analysis")

        # Compute correlation
        corr_matrix = self.data[variables].corr()

        return corr_matrix

    def treatment_balance(self):
        """
        Check balance between treatment and control groups

        Returns:
        --------
        pd.DataFrame : Balance check results
        """
        balance_vars = config.CONTROL_VARS + config.DT_MEASURES[:2]

        # Filter to pre-treatment period if possible
        if 'Post' in self.data.columns:
            pre_data = self.data[self.data['Post'] == 0]
        else:
            pre_data = self.data

        # Perform balance check
        balance_results = check_balance(
            pre_data,
            treatment_var='Treat',
            balance_vars=balance_vars,
            standardized=True
        )

        return balance_results

    def create_treatment_distribution_table(self):
        """
        Create a table showing the distribution of treatment status over time

        Returns:
        --------
        pd.DataFrame : Treatment distribution table
        """
        # Check if required variables exist
        if not all(var in self.data.columns for var in ['Treat', 'Post', 'year']):
            raise ValueError(
                "Required variables (Treat, Post, year) not found")

        # Create cross-tabulation
        treat_post_table = pd.crosstab(
            self.data['Treat'],
            self.data['Post'],
            rownames=['Treat'],
            colnames=['Post'],
            margins=True
        )

        # Create year-wise distribution
        yearly_table = pd.crosstab(
            self.data['year'],
            self.data['Treat'],
            rownames=['Year'],
            colnames=['Treat'],
            margins=True
        )

        # MSCI inclusion by year if available
        if 'MSCI' in self.data.columns:
            msci_yearly = pd.crosstab(
                self.data['year'],
                self.data['MSCI'],
                rownames=['Year'],
                colnames=['MSCI'],
                margins=True
            )

            return {'treat_post': treat_post_table, 'yearly': yearly_table, 'msci_yearly': msci_yearly}
        else:
            return {'treat_post': treat_post_table, 'yearly': yearly_table}

    def export_results(self, output_dir=None):
        """
        Export descriptive statistics results to files

        Parameters:
        -----------
        output_dir : str or Path or None
            Directory to save results, default to config.TABLES_DIR
        """
        if output_dir is None:
            output_dir = config.TABLES_DIR
        else:
            output_dir = Path(output_dir)

        # Create directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate and save summary statistics
        summary = self.summarize_variables()
        summary.to_csv(output_dir / "summary_statistics.csv")

        # Generate and save summary by treatment group
        treat_summary = self.summarize_variables(by_group='Treat')
        for group, df in treat_summary.items():
            df.to_csv(output_dir / f"summary_{group}.csv")

        # Generate and save correlation matrix
        corr = self.correlation_analysis()
        corr.to_csv(output_dir / "correlation_matrix.csv")

        # Generate and save treatment balance
        try:
            balance = self.treatment_balance()
            balance.to_csv(output_dir / "treatment_balance.csv")
        except Exception as e:
            print(f"Error generating treatment balance: {e}")

        # Generate and save treatment distribution
        try:
            dist_tables = self.create_treatment_distribution_table()
            for name, table in dist_tables.items():
                table.to_csv(output_dir / f"treatment_dist_{name}.csv")
        except Exception as e:
            print(f"Error generating treatment distribution: {e}")

        print(f"Descriptive statistics results saved to {output_dir}")
