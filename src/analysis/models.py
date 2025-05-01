"""
Model estimation and analysis for MSCI inclusion and digital transformation research
"""

from src.utils import create_formula, format_regression_table, save_results_to_file, match_nearest_neighbor, prepare_event_study_data
import config
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
import os
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))


class ModelAnalysis:
    def __init__(self, data):
        """
        Initialize the ModelAnalysis

        Parameters:
        -----------
        data : pd.DataFrame
            The preprocessed dataset
        """
        self.data = data
        self.models = {}
        self.matched_data = None

    def run_pooled_ols_did(self, dv="Digital_transformationA", controls=True, year_fe=True):
        """
        Run pooled OLS DiD regression

        Parameters:
        -----------
        dv : str
            Dependent variable name
        controls : bool
            Whether to include control variables
        year_fe : bool
            Whether to include year fixed effects

        Returns:
        --------
        statsmodels regression result
        """
        # Main independent variables
        x_vars = ["Treat", "Post", "TreatPost"]

        # Add control variables
        if controls:
            control_vars = [
                var for var in config.CONTROL_VARS if var in self.data.columns]
            x_vars.extend(control_vars)

        # Add fixed effects
        fe_vars = ["year"] if year_fe else []

        # Create formula
        formula, _ = create_formula(dv, x_vars, fe_vars)

        # Run regression
        model = smf.ols(formula, data=self.data).fit(cov_type='cluster',
                                                     cov_kwds={'groups': self.data['stkcd']})

        # Store model
        model_name = f"pooled_ols_did_{dv}"
        if controls:
            model_name += "_controls"
        if year_fe:
            model_name += "_yearfe"

        self.models[model_name] = model

        return model

    def run_direct_msci_effect(self, dv="Digital_transformationA", controls=True, year_fe=True):
        """
        Run direct effect of MSCI inclusion with firm fixed effects

        Parameters:
        -----------
        dv : str
            Dependent variable name
        controls : bool
            Whether to include control variables
        year_fe : bool
            Whether to include year fixed effects

        Returns:
        --------
        statsmodels regression result
        """
        # Check if MSCI_clean exists
        if 'MSCI_clean' not in self.data.columns:
            raise ValueError("MSCI_clean variable not found in dataset")

        # Main independent variables
        x_vars = ["MSCI_clean"]

        # Add control variables
        if controls:
            control_vars = [
                var for var in config.CONTROL_VARS if var in self.data.columns]
            x_vars.extend(control_vars)

        # Add fixed effects
        fe_vars = ["year"] if year_fe else []

        # Create formula for entity_effects model
        model = smf.ols(
            f"{dv} ~ {' + '.join(x_vars + ['entity'])}",
            data=self.data
        ).fit(cov_type='cluster', cov_kwds={'groups': self.data['stkcd']})

        # Store model
        model_name = f"direct_msci_effect_{dv}"
        if controls:
            model_name += "_controls"
        if year_fe:
            model_name += "_yearfe"

        self.models[model_name] = model

        return model

    def run_post_period_analysis(self, dv="Digital_transformationA", controls=True, year_fe=True):
        """
        Run analysis on post-period only

        Parameters:
        -----------
        dv : str
            Dependent variable name
        controls : bool
            Whether to include control variables
        year_fe : bool
            Whether to include year fixed effects

        Returns:
        --------
        statsmodels regression result
        """
        # Filter to post period
        post_data = self.data[self.data['Post'] == 1].copy()

        if len(post_data) == 0:
            raise ValueError("No observations in post-period")

        # Main independent variables
        x_vars = ["MSCI_clean"]

        # Add control variables
        if controls:
            control_vars = [
                var for var in config.CONTROL_VARS if var in post_data.columns]
            x_vars.extend(control_vars)

        # Add fixed effects
        fe_vars = ["year"] if year_fe else []

        # Create formula for entity_effects model
        model = smf.ols(
            f"{dv} ~ {' + '.join(x_vars + ['entity'])}",
            data=post_data
        ).fit(cov_type='cluster', cov_kwds={'groups': post_data['stkcd']})

        # Store model
        model_name = f"post_period_{dv}"
        if controls:
            model_name += "_controls"
        if year_fe:
            model_name += "_yearfe"

        self.models[model_name] = model

        return model

    def run_matched_sample_analysis(self, dv="Digital_transformationA", matching_vars=None):
        """
        Run matched sample analysis

        Parameters:
        -----------
        dv : str
            Dependent variable name
        matching_vars : list or None
            Variables to use for matching, default to age and productivity

        Returns:
        --------
        tuple : (matched_data, ate, ate_se)
            matched_data: DataFrame of matched data
            ate: Average treatment effect
            ate_se: Standard error of ATE
        """
        if matching_vars is None:
            # Use age and productivity for matching
            matching_vars = ["age", "TFP_OP"]

            # Filter to variables that exist in the dataset
            matching_vars = [
                var for var in matching_vars if var in self.data.columns]

            if not matching_vars:
                raise ValueError("No valid matching variables found")

        # Filter to post period
        post_data = self.data[self.data['Post'] == 1].copy()

        # Perform matching
        matched_data, ate, ate_se = match_nearest_neighbor(
            post_data,
            treatment_var="MSCI_clean",
            outcome_var=dv,
            matching_vars=matching_vars
        )

        # Store results
        self.matched_data = matched_data
        self.matched_ate = ate
        self.matched_ate_se = ate_se

        # Print results
        print(f"Matched Sample Analysis Results:")
        print(f"Average Treatment Effect (ATE): {ate:.4f}")
        print(f"Standard Error: {ate_se:.4f}")
        print(f"t-statistic: {ate/ate_se:.4f}")
        print(f"Number of matched pairs: {len(matched_data)}")

        return matched_data, ate, ate_se

    def run_first_differences(self, dv="Digital_transformationA", controls=True, year_fe=True):
        """
        Run first-differences model on post-period changes

        Parameters:
        -----------
        dv : str
            Dependent variable name
        controls : bool
            Whether to include control variables
        year_fe : bool
            Whether to include year fixed effects

        Returns:
        --------
        statsmodels regression result
        """
        # Create first-differenced dependent variable if it doesn't exist
        diff_var = f"D_{dv}"
        if diff_var not in self.data.columns:
            self.data[diff_var] = self.data.groupby('stkcd')[dv].diff()

        # Filter to post period
        post_data = self.data[self.data['Post'] == 1].copy()

        # Main independent variables
        x_vars = ["MSCI_clean"]

        # Add control variables
        if controls:
            control_vars = [
                var for var in config.CONTROL_VARS if var in post_data.columns]
            x_vars.extend(control_vars)

        # Add fixed effects
        fe_vars = ["year"] if year_fe else []

        # Create formula
        formula, _ = create_formula(diff_var, x_vars, fe_vars)

        # Run regression
        model = smf.ols(formula, data=post_data).fit(cov_type='cluster',
                                                     cov_kwds={'groups': post_data['stkcd']})

        # Store model
        model_name = f"first_diff_{dv}"
        if controls:
            model_name += "_controls"
        if year_fe:
            model_name += "_yearfe"

        self.models[model_name] = model

        return model

    def run_event_study(self, dv="Digital_transformationA", window=(-5, 5), controls=True):
        """
        Run event study analysis

        Parameters:
        -----------
        dv : str
            Dependent variable name
        window : tuple
            Event window (pre, post) periods
        controls : bool
            Whether to include control variables

        Returns:
        --------
        statsmodels regression result
        """
        # Prepare event study data
        event_data = prepare_event_study_data(
            self.data,
            event_time_var="Event_time",
            outcome_var=dv,
            window=window
        )

        # Get event time dummy variables
        time_dummies = [
            col for col in event_data.columns if col.startswith('time_')]

        # Main independent variables
        x_vars = time_dummies

        # Add control variables
        if controls:
            control_vars = [
                var for var in config.CONTROL_VARS if var in event_data.columns]
            x_vars.extend(control_vars)

        # Add year fixed effects (already included in time dummies)

        # Create formula for entity_effects model
        model = smf.ols(
            f"{dv} ~ {' + '.join(x_vars + ['entity'])}",
            data=event_data
        ).fit(cov_type='cluster', cov_kwds={'groups': event_data['stkcd']})

        # Store model
        model_name = f"event_study_{dv}"
        if controls:
            model_name += "_controls"

        self.models[model_name] = model

        return model

    def compare_models(self):
        """
        Compare results from different models

        Returns:
        --------
        pd.DataFrame : Comparison of coefficients and p-values
        """
        if not self.models:
            raise ValueError("No models have been estimated")

        results = []

        # Extract coefficient of interest from each model
        for name, model in self.models.items():
            # Determine which coefficient to extract based on model type
            if 'pooled_ols_did' in name:
                coef_name = 'TreatPost'
            elif 'direct_msci_effect' in name or 'post_period' in name or 'first_diff' in name:
                coef_name = 'MSCI_clean'
            elif 'event_study' in name:
                # Use the latest post-treatment period
                time_coefs = [c for c in model.params.index if c.startswith(
                    'time_') and int(c.split('_')[1]) > 0]
                coef_name = max(time_coefs, key=lambda x: int(
                    x.split('_')[1])) if time_coefs else None
            else:
                coef_name = None

            # Extract coefficient and statistics
            if coef_name and coef_name in model.params:
                coef = model.params[coef_name]
                pval = model.pvalues[coef_name]
                stderr = model.bse[coef_name]

                results.append({
                    'Model': name,
                    'Coefficient': coef_name,
                    'Estimate': coef,
                    'Std. Error': stderr,
                    'p-value': pval,
                    'Significant': pval < 0.05
                })

        # Add matched sample results if available
        if hasattr(self, 'matched_ate') and hasattr(self, 'matched_ate_se'):
            results.append({
                'Model': 'matched_sample',
                'Coefficient': 'ATE',
                'Estimate': self.matched_ate,
                'Std. Error': self.matched_ate_se,
                'p-value': 2 * (1 - abs(self.matched_ate / self.matched_ate_se)),
                'Significant': 2 * (1 - abs(self.matched_ate / self.matched_ate_se)) < 0.05
            })

        return pd.DataFrame(results)

    def export_results(self):
        """
        Export model results to files

        Returns:
        --------
        None
        """
        # Create directory if it doesn't exist
        output_dir = config.TABLES_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        # Export comparison table
        comparison = self.compare_models()
        comparison.to_csv(output_dir / "model_comparison.csv", index=False)

        # Export detailed results for each model
        for name, model in self.models.items():
            table = format_regression_table(model, title=f"Model: {name}")
            save_results_to_file(table, f"model_{name}", 'txt')

        # Export matched sample results if available
        if self.matched_data is not None:
            self.matched_data.to_csv(
                output_dir / "matched_sample_data.csv", index=False)

            # Create summary report
            report_content = "===============================================================\n"
            report_content += "             MATCHED SAMPLE ANALYSIS SUMMARY                  \n"
            report_content += "===============================================================\n\n"
            report_content += f"Average Treatment Effect (ATE): {self.matched_ate:.4f}\n"
            report_content += f"Standard Error: {self.matched_ate_se:.4f}\n"
            report_content += f"t-statistic: {self.matched_ate/self.matched_ate_se:.4f}\n"
            report_content += f"p-value: {2 * (1 - abs(self.matched_ate/self.matched_ate_se)):.4f}\n"
            report_content += f"Number of matched pairs: {len(self.matched_data)}\n"

            save_results_to_file(
                report_content, "matched_sample_summary", 'txt')

        print(f"Model analysis results saved to {output_dir}")
