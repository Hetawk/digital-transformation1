"""
Hypothesis testing module for MSCI inclusion and digital transformation analysis
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


class HypothesisTesting:
    def __init__(self, data):
        """
        Initialize the HypothesisTesting

        Parameters:
        -----------
        data : pd.DataFrame
            The preprocessed dataset
        """
        self.data = data
        self.results = {}

    def test_h1(self, controls=True, year_fe=True):
        """
        Test Hypothesis 1: Capital market liberalization positively influences digital transformation

        Parameters:
        -----------
        controls : bool, default True
            Whether to include control variables
        year_fe : bool, default True
            Whether to include year fixed effects

        Returns:
        --------
        statsmodels regression result
        """
        # Dependent variable
        y_var = "Digital_transformationA"

        # Main independent variables
        x_vars = ["Treat", "Post", "TreatPost"]

        # Control variables
        if controls:
            x_vars += config.CONTROL_VARS

        # Fixed effects
        fe_vars = ["year"] if year_fe else []

        # Create formula
        formula, _ = create_formula(y_var, x_vars, fe_vars)

        # Run regression
        model = smf.ols(formula, data=self.data).fit(
            cov_type='cluster', cov_kwds={'groups': self.data['stkcd']})

        # Store results
        self.results['h1'] = model

        # Format and print results
        print(format_regression_table(
            model, title="Hypothesis 1: MSCI Inclusion Effect on Digital Transformation"))

        return model

    def test_h2(self, controls=True, year_fe=True):
        """
        Test Hypothesis 2: MSCI inclusion leads to increased adoption of specific digital technologies

        Parameters:
        -----------
        controls : bool, default True
            Whether to include control variables
        year_fe : bool, default True
            Whether to include year fixed effects

        Returns:
        --------
        statsmodels regression result
        """
        # Dependent variable - use the alternative measure
        y_var = "Digital_transformationB"

        # Main independent variables
        x_vars = ["Treat", "Post", "TreatPost"]

        # Control variables
        if controls:
            x_vars += config.CONTROL_VARS

        # Fixed effects
        fe_vars = ["year"] if year_fe else []

        # Create formula
        formula, _ = create_formula(y_var, x_vars, fe_vars)

        # Run regression
        model = smf.ols(formula, data=self.data).fit(
            cov_type='cluster', cov_kwds={'groups': self.data['stkcd']})

        # Store results
        self.results['h2'] = model

        # Format and print results
        print(format_regression_table(
            model, title="Hypothesis 2: MSCI Inclusion Effect on Digital Technology Adoption"))

        return model

    def run_placebo_test(self, placebo_year=None, controls=True, year_fe=True):
        """
        Run placebo test with alternative treatment year

        Parameters:
        -----------
        placebo_year : int or None
            Placebo treatment year, default to config.PLACEBO_YEAR
        controls : bool, default True
            Whether to include control variables
        year_fe : bool, default True
            Whether to include year fixed effects

        Returns:
        --------
        statsmodels regression result
        """
        if placebo_year is None:
            placebo_year = config.PLACEBO_YEAR

        # Create placebo treatment variables
        self.data['Placebo_Post'] = (
            self.data['year'] >= placebo_year).astype(int)
        self.data['Placebo_TreatPost'] = self.data['Treat'] * \
            self.data['Placebo_Post']

        # Dependent variable
        y_var = "Digital_transformationA"

        # Main independent variables
        x_vars = ["Treat", "Placebo_Post", "Placebo_TreatPost"]

        # Control variables
        if controls:
            x_vars += config.CONTROL_VARS

        # Fixed effects
        fe_vars = ["year"] if year_fe else []

        # Create formula
        formula, _ = create_formula(y_var, x_vars, fe_vars)

        # Run regression
        model = smf.ols(formula, data=self.data).fit(
            cov_type='cluster', cov_kwds={'groups': self.data['stkcd']})

        # Store results
        self.results['placebo'] = model

        # Format and print results
        print(format_regression_table(
            model, title=f"Placebo Test: Assuming Treatment in {placebo_year}"))

        return model

    def compare_hypotheses(self):
        """
        Compare hypothesis test results

        Returns:
        --------
        pd.DataFrame : Comparison of coefficients and p-values
        """
        if 'h1' not in self.results or 'h2' not in self.results:
            raise ValueError("Run test_h1() and test_h2() first")

        # Extract coefficients and p-values
        results = []

        models = {
            'H1 (Digital_transformationA)': self.results['h1'],
            'H2 (Digital_transformationB)': self.results['h2']
        }

        if 'placebo' in self.results:
            models['Placebo Test'] = self.results['placebo']

        for name, model in models.items():
            if name == 'Placebo Test':
                coef = model.params.get('Placebo_TreatPost', np.nan)
                pval = model.pvalues.get('Placebo_TreatPost', np.nan)
                stderr = model.bse.get('Placebo_TreatPost', np.nan)
            else:
                coef = model.params.get('TreatPost', np.nan)
                pval = model.pvalues.get('TreatPost', np.nan)
                stderr = model.bse.get('TreatPost', np.nan)

            results.append({
                'Hypothesis': name,
                'Coefficient': coef,
                'Std. Error': stderr,
                'p-value': pval,
                'Significant': pval < 0.05 if not np.isnan(pval) else None
            })

        return pd.DataFrame(results)

    def export_results(self):
        """
        Export hypothesis testing results to files

        Returns:
        --------
        None
        """
        # Create directory if it doesn't exist
        output_dir = config.TABLES_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        # Export comparison table
        comparison = self.compare_hypotheses()
        comparison.to_csv(
            output_dir / "hypothesis_comparison.csv", index=False)

        # Export detailed results for each model
        for name, model in self.results.items():
            if name == 'h1':
                title = "Hypothesis 1: MSCI Inclusion Effect on Digital Transformation"
            elif name == 'h2':
                title = "Hypothesis 2: MSCI Inclusion Effect on Digital Technology Adoption"
            elif name == 'placebo':
                title = f"Placebo Test: Assuming Treatment in {config.PLACEBO_YEAR}"
            else:
                title = f"Model: {name}"

            table = format_regression_table(model, title=title)
            save_results_to_file(table, f"hypothesis_{name}", 'txt')

        # Create a summary report
        report_content = "===============================================================\n"
        report_content += "             HYPOTHESIS TESTING SUMMARY REPORT                \n"
        report_content += "===============================================================\n\n"

        # Add H1 results
        if 'h1' in self.results:
            model = self.results['h1']
            coef = model.params.get('TreatPost', np.nan)
            pval = model.pvalues.get('TreatPost', np.nan)
            significant = "significant" if pval < 0.05 else "not significant"

            report_content += "H1: Capital market liberalization positively influences digital transformation\n"
            report_content += f"   - Effect (TreatPost): {coef:.3f}, p-value: {pval:.3f}\n"
            report_content += f"   - Conclusion: The effect is {significant} at the 5% level.\n\n"

        # Add H2 results
        if 'h2' in self.results:
            model = self.results['h2']
            coef = model.params.get('TreatPost', np.nan)
            pval = model.pvalues.get('TreatPost', np.nan)
            significant = "significant" if pval < 0.05 else "not significant"

            report_content += "H2: MSCI inclusion leads to increased adoption of digital technologies\n"
            report_content += f"   - Effect (TreatPost): {coef:.3f}, p-value: {pval:.3f}\n"
            report_content += f"   - Conclusion: The effect is {significant} at the 5% level.\n\n"

        # Add placebo results
        if 'placebo' in self.results:
            model = self.results['placebo']
            coef = model.params.get('Placebo_TreatPost', np.nan)
            pval = model.pvalues.get('Placebo_TreatPost', np.nan)
            significant = "significant" if pval < 0.05 else "not significant"

            report_content += f"Placebo Test (Treatment year = {config.PLACEBO_YEAR}):\n"
            report_content += f"   - Effect (Placebo_TreatPost): {coef:.3f}, p-value: {pval:.3f}\n"
            report_content += f"   - Conclusion: The placebo effect is {significant} at the 5% level.\n"

            if 'h1' in self.results:
                h1_coef = self.results['h1'].params.get('TreatPost', np.nan)
                report_content += f"   - Comparison: Placebo effect is {abs(coef/h1_coef):.2f} times the size of the actual effect.\n\n"

        # Save summary report
        save_results_to_file(
            report_content, "hypothesis_summary_report", 'txt')

        print(f"Hypothesis testing results saved to {output_dir}")
