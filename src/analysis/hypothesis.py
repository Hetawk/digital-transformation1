"""
Hypothesis testing module for MSCI inclusion and digital transformation analysis
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from pathlib import Path
import sys
import logging
import traceback

from utils.util import create_formula, format_regression_table, save_results_to_file
import loader.config as config

# Set up logger
logger = logging.getLogger('digital_transformation')
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', 
                                 datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class HypothesisTesting:
    def __init__(self, data):
        """
        Initialize the HypothesisTesting class

        Parameters:
        -----------
        data : pd.DataFrame
            The preprocessed dataset
        """
        self.data = data
        self.results = {}

    def test_h1(self, controls=True, year_fe=True):
        """
        Test Hypothesis 1: MSCI inclusion leads to increased digital transformation

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
        control_vars = []
        if controls:
            control_vars = [var for var in config.CONTROL_VARS if var in self.data.columns]
            x_vars += control_vars

        # Fixed effects
        fe_vars = []
        if year_fe:
            fe_vars = ["year"]

        # Create formula
        formula, all_vars = create_formula(y_var, x_vars, fe_vars)
        
        # Collect all variables needed for the model
        all_model_vars = [y_var] + x_vars
        if "year" in fe_vars:
            all_model_vars.append("year")
        
        # Add cluster variable
        cluster_var = "stkcd"
        all_model_vars.append(cluster_var)
        
        # Check data before fitting
        logger.info(f"Preparing to fit H1 model with formula: {formula}")
        
        # Drop rows with missing values in any required variables
        original_rows = len(self.data)
        df_sub = self.data.dropna(subset=all_model_vars)
        dropped_rows = original_rows - len(df_sub)
        
        logger.info(f"Fitting H1 on N={len(df_sub)} obs (dropped {dropped_rows} rows with missing values)")
        
        # Check if we have enough data
        n_params = len(x_vars) + len(fe_vars) + 1  # +1 for intercept
        if len(df_sub) == 0:
            logger.error("No complete observations available for H1 model after dropping missing values")
            sys.exit(1)
        elif len(df_sub) < n_params + 10:
            logger.error(f"Insufficient observations ({len(df_sub)}) for H1 model with {n_params} parameters")
            sys.exit(1)
            
        # Ensure cluster groups are aligned with data
        cluster_groups = df_sub[cluster_var].astype(str)
        if len(cluster_groups) != len(df_sub):
            logger.error(f"Cluster variable length ({len(cluster_groups)}) != data length ({len(df_sub)})")
            sys.exit(1)

        # Run regression with cluster-robust standard errors
        try:
            model = smf.ols(formula, data=df_sub).fit(
                cov_type='cluster', cov_kwds={'groups': cluster_groups})
            
            # Store results
            self.results['h1'] = model

            # Format and print results
            result_str = format_regression_table(
                model, title="Hypothesis 1: MSCI Inclusion Effect on Digital Transformation")
            print(result_str)
            
            return model
            
        except Exception as e:
            logger.error(f"Error in H1 regression: {str(e)}")
            logger.error(f"Data shape: {df_sub.shape}, formula: {formula}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            sys.exit(1)

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
        control_vars = []
        if controls:
            control_vars = [var for var in config.CONTROL_VARS if var in self.data.columns]
            x_vars += control_vars

        # Fixed effects
        fe_vars = []
        if year_fe:
            fe_vars = ["year"]

        # Create formula
        formula, all_vars = create_formula(y_var, x_vars, fe_vars)
        
        # Collect all variables needed for the model
        all_model_vars = [y_var] + x_vars
        if "year" in fe_vars:
            all_model_vars.append("year")
        
        # Add cluster variable
        cluster_var = "stkcd"
        all_model_vars.append(cluster_var)
        
        # Check data before fitting
        logger.info(f"Preparing to fit H2 model with formula: {formula}")
        
        # Drop rows with missing values in any required variables
        original_rows = len(self.data)
        df_sub = self.data.dropna(subset=all_model_vars)
        dropped_rows = original_rows - len(df_sub)
        
        logger.info(f"Fitting H2 on N={len(df_sub)} obs (dropped {dropped_rows} rows with missing values)")
        
        # Check if we have enough data
        n_params = len(x_vars) + len(fe_vars) + 1  # +1 for intercept
        if len(df_sub) == 0:
            logger.error("No complete observations available for H2 model after dropping missing values")
            sys.exit(1)
        elif len(df_sub) < n_params + 10:
            logger.error(f"Insufficient observations ({len(df_sub)}) for H2 model with {n_params} parameters")
            sys.exit(1)
            
        # Ensure cluster groups are aligned with data
        cluster_groups = df_sub[cluster_var].astype(str)
        if len(cluster_groups) != len(df_sub):
            logger.error(f"Cluster variable length ({len(cluster_groups)}) != data length ({len(df_sub)})")
            sys.exit(1)

        # Run regression with cluster-robust standard errors
        try:
            model = smf.ols(formula, data=df_sub).fit(
                cov_type='cluster', cov_kwds={'groups': cluster_groups})
            
            # Store results
            self.results['h2'] = model

            # Format and print results
            result_str = format_regression_table(
                model, title="Hypothesis 2: MSCI Inclusion Effect on Digital Technology Adoption")
            print(result_str)
            
            return model
            
        except Exception as e:
            logger.error(f"Error in H2 regression: {str(e)}")
            logger.error(f"Data shape: {df_sub.shape}, formula: {formula}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            sys.exit(1)

    def run_placebo_test(self, placebo_year=None, controls=True, year_fe=True):
        """
        Run placebo test using different treatment year

        Parameters:
        -----------
        placebo_year : int or None
            Placebo treatment year, default is 3 years before actual treatment
        controls : bool, default True
            Whether to include control variables
        year_fe : bool, default True
            Whether to include year fixed effects

        Returns:
        --------
        statsmodels regression result
        """
        if placebo_year is None:
            placebo_year = int(config.TREATMENT_YEAR) - 3

        # Store original data
        original_data = self.data.copy()

        # Create placebo Post variable
        self.data['Placebo_Post'] = (self.data['year'] >= placebo_year).astype(int)
        
        # Create placebo Treat variable - define as 0 for all observations
        # to ensure a proper placebo test
        self.data['Placebo_Treat'] = 0
        
        # If we have non-zero Treat values, randomly assign some control units to be "placebo treated"
        # but ONLY among those that are never actually treated in the real study
        if 1 in self.data['Treat'].unique():
            # Identify control units (never treated in the actual study)
            control_firms = self.data[self.data['Treat'] == 0]['stkcd'].unique()
            
            # If we have enough control firms, randomly select ~20% of them to be "placebo treated"
            if len(control_firms) >= 10:
                import random
                random.seed(42)  # For reproducibility
                placebo_treated = random.sample(list(control_firms), k=max(1, int(len(control_firms) * 0.2)))
                
                # Assign Placebo_Treat = 1 to these randomly selected control firms
                self.data.loc[self.data['stkcd'].isin(placebo_treated), 'Placebo_Treat'] = 1
                
                logger.info(f"Assigned {len(placebo_treated)} control firms to placebo treatment group")
            else:
                logger.warning("Not enough control firms to create meaningful placebo treatment group")
        
        # Create placebo interaction
        self.data['Placebo_TreatPost'] = self.data['Placebo_Treat'] * self.data['Placebo_Post']

        # Dependent variable
        y_var = "Digital_transformationA"

        # Main independent variables (including placebo)
        x_vars = ["Placebo_Treat", "Placebo_Post", "Placebo_TreatPost"]
        
        # Control variables
        control_vars = []
        if controls:
            control_vars = [var for var in config.CONTROL_VARS if var in self.data.columns]
            x_vars += control_vars

        # Fixed effects
        fe_vars = []
        if year_fe:
            fe_vars = ["year"]

        # Create formula
        formula, all_vars = create_formula(y_var, x_vars, fe_vars)
        
        # Collect all variables needed for the model
        all_model_vars = [y_var] + x_vars
        if "year" in fe_vars:
            all_model_vars.append("year")
        
        # Add cluster variable
        cluster_var = "stkcd"
        all_model_vars.append(cluster_var)
        
        # Check data before fitting
        logger.info(f"Preparing to fit placebo test (year={placebo_year}) with formula: {formula}")
        
        # Drop rows with missing values in any required variables
        original_rows = len(self.data)
        df_sub = self.data.dropna(subset=all_model_vars)
        dropped_rows = original_rows - len(df_sub)
        
        logger.info(f"Fitting placebo test on N={len(df_sub)} obs (dropped {dropped_rows} rows with missing values)")
        
        # Check if we have enough data
        n_params = len(x_vars) + len(fe_vars) + 1  # +1 for intercept
        if len(df_sub) == 0:
            logger.error("No complete observations available for placebo test after dropping missing values")
            sys.exit(1)
        elif len(df_sub) < n_params + 10:
            logger.error(f"Insufficient observations ({len(df_sub)}) for placebo test with {n_params} parameters")
            sys.exit(1)
            
        # Ensure cluster groups are aligned with data
        cluster_groups = df_sub[cluster_var].astype(str)
        if len(cluster_groups) != len(df_sub):
            logger.error(f"Cluster variable length ({len(cluster_groups)}) != data length ({len(df_sub)})")
            sys.exit(1)

        # Run regression with cluster-robust standard errors
        try:
            model = smf.ols(formula, data=df_sub).fit(
                cov_type='cluster', cov_kwds={'groups': cluster_groups})
            
            # Store results
            self.results['placebo'] = model

            # Format and print results
            result_str = format_regression_table(
                model, title=f"Placebo Test: Assuming Treatment in {placebo_year}")
            print(result_str)
            
            # Restore original data
            self.data = original_data
            
            return model
            
        except Exception as e:
            logger.error(f"Error in placebo test: {str(e)}")
            logger.error(f"Data shape: {df_sub.shape}, formula: {formula}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Restore original data
            self.data = original_data
            sys.exit(1)

    def compare_hypotheses(self):
        """
        Compare results from different hypotheses

        Returns:
        --------
        pd.DataFrame: Comparison of results
        """
        results = []

        for name, model in self.results.items():
            if name == 'h1' or name == 'h2':
                coef = model.params.get('TreatPost', np.nan)
                pval = model.pvalues.get('TreatPost', np.nan)
                stderr = model.bse.get('TreatPost', np.nan)
            elif name == 'placebo':
                coef = model.params.get('Placebo_TreatPost', np.nan)
                pval = model.pvalues.get('Placebo_TreatPost', np.nan)
                stderr = model.bse.get('Placebo_TreatPost', np.nan)
            else:
                coef = np.nan
                pval = np.nan
                stderr = np.nan

            results.append({
                'Hypothesis': name,
                'Coefficient': coef,
                'Std. Error': stderr,
                'p-value': pval,
                'Significant': pval < 0.05 if not np.isnan(pval) else None
            })

        return pd.DataFrame(results)

    def export_results(self, output_dir=None):
        """
        Export hypothesis testing results to files

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

        # Export hypothesis testing results
        if self.results:
            # Export comparison of hypotheses
            comparison = self.compare_hypotheses()
            comparison.to_csv(output_dir / "hypothesis_comparison.csv", index=False)

            # Export individual hypothesis results
            for name, model in self.results.items():
                # Save model summary
                with open(output_dir / f"{name}_results.txt", "w") as f:
                    f.write(str(model.summary()))

            logger.info(f"Hypothesis testing results saved to {output_dir}")
        else:
            logger.warning("No hypothesis testing results to export")
