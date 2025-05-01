"""
Mechanism analysis for MSCI inclusion and digital transformation research
"""

from src.utils import create_formula, format_regression_table, save_results_to_file
import config
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
import os
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))


class MechanismAnalysis:
    def __init__(self, data):
        """
        Initialize the MechanismAnalysis

        Parameters:
        -----------
        data : pd.DataFrame
            The preprocessed dataset
        """
        self.data = data
        self.mechanism_results = {}

    def analyze_financial_access(self, dv="Digital_transformationA", post_only=True):
        """
        Analyze financial access mechanism

        Parameters:
        -----------
        dv : str
            Dependent variable name
        post_only : bool
            Whether to analyze post-period only

        Returns:
        --------
        dict : Dictionary of regression models
        """
        # Filter data if post_only
        if post_only:
            data = self.data[self.data['Post'] == 1].copy()
        else:
            data = self.data.copy()

        # Check mechanism variables
        fin_vars = [
            var for var in config.FINANCIAL_ACCESS_VARS if var in data.columns]

        if not fin_vars:
            raise ValueError(
                "No financial access mechanism variables found in dataset")

        # Run regressions
        models = {}

        for fin_var in fin_vars:
            # Create interaction term
            data[f'MSCI_clean_{fin_var}'] = data['MSCI_clean'] * data[fin_var]

            # Define regression variables
            x_vars = ['MSCI_clean', fin_var, f'MSCI_clean_{fin_var}']

            # Add control variables (excluding the current mechanism variable)
            control_vars = [
                var for var in config.CONTROL_VARS if var in data.columns and var != fin_var]
            x_vars.extend(control_vars)

            # Add year fixed effects
            fe_vars = ["year"]

            # Create formula
            formula, _ = create_formula(dv, x_vars, fe_vars)

            # Run regression with entity fixed effects
            try:
                # Add entity (firm) fixed effects
                model = smf.ols(
                    f"{formula} + entity",
                    data=data
                ).fit(cov_type='cluster', cov_kwds={'groups': data['stkcd']})

                # Store results
                models[fin_var] = model

                # Print results
                print(f"\nFinancial Access Mechanism: {fin_var}")
                print(format_regression_table(
                    model,
                    title=f"Effect of MSCI inclusion on {dv} through {fin_var}"
                ))
            except Exception as e:
                print(f"Error estimating model for {fin_var}: {e}")

        # Store all models in mechanism results
        self.mechanism_results['financial_access'] = models

        return models

    def analyze_corporate_governance(self, dv="Digital_transformationA", post_only=True):
        """
        Analyze corporate governance mechanism

        Parameters:
        -----------
        dv : str
            Dependent variable name
        post_only : bool
            Whether to analyze post-period only

        Returns:
        --------
        dict : Dictionary of regression models
        """
        # Filter data if post_only
        if post_only:
            data = self.data[self.data['Post'] == 1].copy()
        else:
            data = self.data.copy()

        # Check mechanism variables
        gov_vars = [var for var in config.CORP_GOV_VARS if var in data.columns]

        if not gov_vars:
            print("Warning: No corporate governance variables found in dataset")
            return {}

        # Run regressions
        models = {}

        for gov_var in gov_vars:
            # Create interaction term
            data[f'MSCI_clean_{gov_var}'] = data['MSCI_clean'] * data[gov_var]

            # Define regression variables
            x_vars = ['MSCI_clean', gov_var, f'MSCI_clean_{gov_var}']

            # Add control variables (excluding the current mechanism variable)
            control_vars = [
                var for var in config.CONTROL_VARS if var in data.columns and var != gov_var]
            x_vars.extend(control_vars)

            # Add year fixed effects
            fe_vars = ["year"]

            # Create formula
            formula, _ = create_formula(dv, x_vars, fe_vars)

            # Run regression with entity fixed effects
            try:
                # Add entity (firm) fixed effects
                model = smf.ols(
                    f"{formula} + entity",
                    data=data
                ).fit(cov_type='cluster', cov_kwds={'groups': data['stkcd']})

                # Store results
                models[gov_var] = model

                # Print results
                print(f"\nCorporate Governance Mechanism: {gov_var}")
                print(format_regression_table(
                    model,
                    title=f"Effect of MSCI inclusion on {dv} through {gov_var}"
                ))
            except Exception as e:
                print(f"Error estimating model for {gov_var}: {e}")

        # Store all models in mechanism results
        self.mechanism_results['corporate_governance'] = models

        return models

    def analyze_investor_scrutiny(self, dv="Digital_transformationA", post_only=True):
        """
        Analyze investor scrutiny mechanism

        Parameters:
        -----------
        dv : str
            Dependent variable name
        post_only : bool
            Whether to analyze post-period only

        Returns:
        --------
        dict : Dictionary of regression models
        """
        # Filter data if post_only
        if post_only:
            data = self.data[self.data['Post'] == 1].copy()
        else:
            data = self.data.copy()

        # Check mechanism variables
        is_vars = [
            var for var in config.INVESTOR_SCRUTINY_VARS if var in data.columns]

        if not is_vars:
            print("Warning: No investor scrutiny variables found in dataset")
            return {}

        # Run regressions
        models = {}

        for is_var in is_vars:
            # Create interaction term
            data[f'MSCI_clean_{is_var}'] = data['MSCI_clean'] * data[is_var]

            # Define regression variables
            x_vars = ['MSCI_clean', is_var, f'MSCI_clean_{is_var}']

            # Add control variables (excluding the current mechanism variable)
            control_vars = [
                var for var in config.CONTROL_VARS if var in data.columns and var != is_var]
            x_vars.extend(control_vars)

            # Add year fixed effects
            fe_vars = ["year"]

            # Create formula
            formula, _ = create_formula(dv, x_vars, fe_vars)

            # Run regression with entity fixed effects
            try:
                # Add entity (firm) fixed effects
                model = smf.ols(
                    f"{formula} + entity",
                    data=data
                ).fit(cov_type='cluster', cov_kwds={'groups': data['stkcd']})

                # Store results
                models[is_var] = model

                # Print results
                print(f"\nInvestor Scrutiny Mechanism: {is_var}")
                print(format_regression_table(
                    model,
                    title=f"Effect of MSCI inclusion on {dv} through {is_var}"
                ))
            except Exception as e:
                print(f"Error estimating model for {is_var}: {e}")

        # Store all models in mechanism results
        self.mechanism_results['investor_scrutiny'] = models

        return models

    def extract_mechanism_effects(self):
        """
        Extract mechanism interaction effects from all models

        Returns:
        --------
        pd.DataFrame : Table of mechanism interaction effects
        """
        if not self.mechanism_results:
            raise ValueError(
                "No mechanism analysis results available. Run analysis methods first.")

        results = []

        for mechanism_type, models in self.mechanism_results.items():
            for var_name, model in models.items():
                # Extract the interaction term coefficient
                interaction_term = f'MSCI_clean_{var_name}'
                if interaction_term in model.params:
                    coef = model.params[interaction_term]
                    pval = model.pvalues[interaction_term]
                    stderr = model.bse[interaction_term]

                    results.append({
                        'Mechanism': mechanism_type,
                        'Variable': var_name,
                        'Interaction Effect': coef,
                        'Std. Error': stderr,
                        'p-value': pval,
                        'Significant': pval < 0.05
                    })

        return pd.DataFrame(results)

    def run_all_mechanisms(self, dv="Digital_transformationA", post_only=True):
        """
        Run all mechanism analyses

        Parameters:
        -----------
        dv : str
            Dependent variable name
        post_only : bool
            Whether to analyze post-period only

        Returns:
        --------
        pd.DataFrame : Table of mechanism interaction effects
        """
        # Run all mechanism analyses
        self.analyze_financial_access(dv, post_only)
        self.analyze_corporate_governance(dv, post_only)
        self.analyze_investor_scrutiny(dv, post_only)

        # Extract and return mechanism effects
        return self.extract_mechanism_effects()

    def export_results(self, output_dir=None):
        """
        Export mechanism analysis results to files

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

        # Export mechanism effects table
        try:
            effects = self.extract_mechanism_effects()
            effects.to_csv(output_dir / "mechanism_effects.csv", index=False)
        except Exception as e:
            print(f"Error exporting mechanism effects: {e}")

        # Export detailed regression tables for each mechanism
        for mechanism_type, models in self.mechanism_results.items():
            for var_name, model in models.items():
                # Create regression table
                table = format_regression_table(
                    model,
                    title=f"{mechanism_type.replace('_', ' ').title()} Mechanism: {var_name}"
                )

                # Save to file
                save_results_to_file(
                    table,
                    f"mechanism_{mechanism_type}_{var_name}",
                    'txt'
                )

        # Create summary report
        report_content = "===============================================================\n"
        report_content += "             MECHANISM ANALYSIS SUMMARY REPORT                \n"
        report_content += "===============================================================\n\n"

        # Add summary for each mechanism type
        for mechanism_type, models in self.mechanism_results.items():
            report_content += f"\n{mechanism_type.replace('_', ' ').upper()} MECHANISMS:\n"
            report_content += "-" * 60 + "\n"

            for var_name, model in models.items():
                # Extract interaction term
                interaction_term = f'MSCI_clean_{var_name}'
                if interaction_term in model.params:
                    coef = model.params[interaction_term]
                    pval = model.pvalues[interaction_term]
                    significant = "significant" if pval < 0.05 else "not significant"

                    report_content += f"- {var_name}: Interaction effect = {coef:.4f} (p={pval:.4f}), {significant}\n"

            report_content += "\n"

        # Save summary report
        save_results_to_file(report_content, "mechanism_summary_report", 'txt')

        print(f"Mechanism analysis results saved to {output_dir}")
