"""
Model estimation and analysis for MSCI inclusion and digital transformation research
"""

from utils.util import create_formula, format_regression_table, save_results_to_file, match_nearest_neighbor, prepare_event_study_data
import loader.config as config
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
import os
from pathlib import Path
import sys
import logging
import traceback

sys.path.append(str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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

    def _build_formula(self, dv, controls=False, year_fe=False, entity_fe=False):
        """
        Build regression formula based on parameters
        
        Parameters:
        -----------
        dv : str
            Dependent variable
        controls : bool
            Whether to include control variables
        year_fe : bool
            Whether to include year fixed effects
        entity_fe : bool
            Whether to include entity fixed effects
            
        Returns:
        --------
        str: Regression formula
        """
        # Start with basic DiD specification
        formula = f"{dv} ~ TreatPost"
        
        # Add controls if specified
        if controls and hasattr(self, 'control_vars') and self.control_vars:
            formula += " + " + " + ".join(self.control_vars)
        
        # Add fixed effects
        if year_fe:
            formula += " + C(year)"
        
        if entity_fe and 'panel_id' in self.data.columns:
            formula += " + C(panel_id)"
        
        return formula

    def run_pooled_ols_did(self, dv="Digital_transformationA", controls=True, year_fe=True, entity_fe=True):
        """
        Run pooled OLS DiD model for staggered adoption design
        
        Parameters:
        -----------
        dv : str
            Dependent variable name
        controls : bool
            Whether to include control variables
        year_fe : bool
            Whether to include year fixed effects
        entity_fe : bool
            Whether to include entity fixed effects
            
        Returns:
        --------
        statsmodels regression result
        """
        # Store model name for identification
        model_name = f"pooled_ols_did_{dv}"
        if controls:
            model_name += "_controls"
        if year_fe:
            model_name += "_yearfe"
        if entity_fe:
            model_name += "_entityfe"
        
        try:
            # Define control variables if not yet defined
            if controls and not hasattr(self, 'control_vars'):
                self.control_vars = ['age', 'TFP_OP', 'SA_index', 'WW_index', 'F050501B', 'F060101B']
                
            # Build formula
            formula = f"{dv} ~ Treat + Post + TreatPost"
            
            # Add controls if specified
            if controls and hasattr(self, 'control_vars'):
                # Only include controls that exist in the dataset
                valid_controls = [var for var in self.control_vars if var in self.data.columns]
                if valid_controls:
                    formula += " + " + " + ".join(valid_controls)
            
            # Add fixed effects
            if year_fe:
                formula += " + C(year)"
            
            if entity_fe and 'panel_id' in self.data.columns:
                formula += " + C(panel_id)"
            
            # Create a clean subset of data for analysis - drop missing values in key columns
            # First identify all variables in the formula
            all_vars = [dv, 'Treat', 'Post', 'TreatPost', 'stkcd']
            if controls and hasattr(self, 'control_vars'):
                all_vars.extend([var for var in self.control_vars if var in self.data.columns])
            
            # Drop rows with NA in any of the required variables
            analysis_data = self.data.dropna(subset=all_vars)
            
            # Ensure the clustering variable is properly prepared
            if 'stkcd' in analysis_data.columns:
                # Convert to string to avoid numeric binning issues
                cluster_var = analysis_data['stkcd'].astype(str)
            else:
                # Use panel_id if available, otherwise create a dummy
                if 'panel_id' in analysis_data.columns:
                    cluster_var = analysis_data['panel_id'].astype(str)
                else:
                    cluster_var = pd.Series(['1'] * len(analysis_data))
            
            # Log the analysis details
            logger.info(f"Running {model_name} on {len(analysis_data)} observations")
            logger.info(f"Formula: {formula}")
            
            # Run the model with the clean data
            model = smf.ols(formula, data=analysis_data).fit(
                cov_type='cluster',
                cov_kwds={'groups': cluster_var}
            )
            
            # Store and return the model
            self.models[model_name] = model
            return model
            
        except Exception as e:
            logger.error(f"Error in pooled OLS DiD: {str(e)}")
            logger.error(f"Details: {traceback.format_exc()}")
            return None

    def run_direct_msci_effect(self, dv="Digital_transformationA", controls=True, year_fe=True, entity_fe=True):
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
        entity_fe : bool
            Whether to include entity fixed effects
            
        Returns:
        --------
        statsmodels regression result
        """
        try:
            # Check if MSCI_clean exists
            if 'MSCI_clean' not in self.data.columns:
                logger.error("MSCI_clean variable not found in dataset")
                return None
                
            # Define model name
            model_name = f"direct_msci_effect_{dv}"
            if controls:
                model_name += "_controls"
            if year_fe:
                model_name += "_yearfe"
            if entity_fe:
                model_name += "_entityfe"
                
            # Define control variables if not yet defined
            if controls and not hasattr(self, 'control_vars'):
                self.control_vars = ['age', 'TFP_OP', 'SA_index', 'WW_index', 'F050501B', 'F060101B']
                
            # Build formula
            formula = f"{dv} ~ MSCI_clean"
            
            # Add controls
            if controls and hasattr(self, 'control_vars'):
                valid_controls = [var for var in self.control_vars if var in self.data.columns]
                if valid_controls:
                    formula += " + " + " + ".join(valid_controls)
            
            # Add fixed effects
            if year_fe:
                formula += " + C(year)"
            
            if entity_fe and 'panel_id' in self.data.columns:
                formula += " + C(panel_id)"
                
            # Create a clean subset of data for analysis
            all_vars = [dv, 'MSCI_clean', 'stkcd']
            if controls and hasattr(self, 'control_vars'):
                all_vars.extend([var for var in self.control_vars if var in self.data.columns])
            
            # Drop rows with NA in any of the required variables
            analysis_data = self.data.dropna(subset=all_vars)
            
            # Prepare clustering variable
            if 'stkcd' in analysis_data.columns:
                cluster_var = analysis_data['stkcd'].astype(str)
            else:
                if 'panel_id' in analysis_data.columns:
                    cluster_var = analysis_data['panel_id'].astype(str)
                else:
                    cluster_var = pd.Series(['1'] * len(analysis_data))
                
            # Log the analysis
            logger.info(f"Running {model_name} on {len(analysis_data)} observations")
            logger.info(f"Formula: {formula}")
                
            # Run the model
            model = smf.ols(formula, data=analysis_data).fit(
                cov_type='cluster',
                cov_kwds={'groups': cluster_var}
            )
            
            # Store the model
            self.models[model_name] = model
            return model
        except Exception as e:
            logger.error(f"Error running direct MSCI effect model: {str(e)}")
            logger.error(f"Details: {traceback.format_exc()}")
            return None

    def run_post_period_analysis(self, dv="Digital_transformationA", controls=True, year_fe=True):
        """
        Run post-period analysis (only post-treatment period)
        
        Parameters:
        -----------
        dv : str, default 'Digital_transformationA'
            Dependent variable
        controls : bool, default True
            Whether to include control variables
        year_fe : bool, default True
            Whether to include year fixed effects
            
        Returns:
        --------
        statsmodels.regression.linear_model.RegressionResults
        """
        # Subset to post-treatment period
        if 'Post' not in self.data.columns:
            raise ValueError("Post variable not found in dataset")
            
        # FIX: Create a copy first to avoid SettingWithCopyWarning
        post_data = self.data[self.data['Post'] == 1].copy()
        
        # Build model formula
        x_vars = ['MSCI']
        
        # Add control variables
        if controls and hasattr(self, 'control_vars'):
            for var in self.control_vars:
                if var in post_data.columns:
                    x_vars.append(var)
        
        # Add fixed effects
        fe_vars = []
        if year_fe:
            fe_vars.append('year')
        
        # Build formula
        from src.utils.util import create_formula
        formula, subset = create_formula(dv, x_vars, fe_vars)
        
        # Drop missing values in variables used in the formula
        all_vars = [dv] + x_vars + fe_vars
        # FIX: Check if clustering variable exists before adding it
        if 'stkcd' in post_data.columns: 
            all_vars.append('stkcd')
        
        # Remove duplicates
        all_vars = list(set(all_vars))
        
        # Drop rows with missing values
        analysis_data = post_data.dropna(subset=all_vars)
        
        if len(analysis_data) == 0:
            raise ValueError("No complete observations for post-period analysis")
        
        # Fit model
        print(f"Fitting post-period model with formula: {formula}")
        print(f"N={len(analysis_data)} observations")
        
        # FIX: Only use cluster robust SE if stkcd exists
        if 'stkcd' in analysis_data.columns:
            model = smf.ols(formula, data=analysis_data).fit(
                cov_type='cluster', 
                cov_kwds={'groups': analysis_data['stkcd'].astype(str)}
            )
        else:
            model = smf.ols(formula, data=analysis_data).fit()
        
        # Store model
        model_name = f"post_period_{dv}"
        if controls:
            model_name += "_controls"
        if year_fe:
            model_name += "_yearfe"
        
        self.models[model_name] = model
        
        return model

    def run_matched_sample_analysis(self, matching_vars=None, outcome_var="Digital_transformationA"):
        """
        Run matched sample analysis
        
        Parameters:
        -----------
        matching_vars : list or None
            Variables to match on, defaults to ['age', 'TFP_OP']
        outcome_var : str, default 'Digital_transformationA'
            Outcome variable
            
        Returns:
        --------
        tuple: (matched_data, ate, ate_se)
        """
        if matching_vars is None:
            matching_vars = ['age', 'TFP_OP']
                    
        # Verify variables exist
        missing_vars = [var for var in matching_vars + [outcome_var, 'Treat'] 
                       if var not in self.data.columns]
        if missing_vars:
            raise ValueError(f"Missing variables for matching: {missing_vars}")
        
        # Create a clean subset for matching - Drop NaN values in relevant columns
        match_cols = matching_vars + [outcome_var, 'Treat']
        match_data = self.data[match_cols].copy()
        match_data = match_data.dropna(subset=match_cols)
        
        if len(match_data) == 0:
            raise ValueError("No complete cases available for matching")
        
        from src.utils.util import match_nearest_neighbor
        matched_data, ate, ate_se = match_nearest_neighbor(
            match_data, 
            treatment_var='Treat',
            outcome_var=outcome_var,
            matching_vars=matching_vars
        )
        
        # Store matched data results for later use
        self.matched_data = matched_data
        self.matched_ate = ate
        self.matched_ate_se = ate_se
        
        print(f"Matched sample analysis results:")
        print(f"  Treatment effect: {ate:.4f}")
        print(f"  Standard error: {ate_se:.4f}")
        
        return matched_data, ate, ate_se

    def run_first_differences(self, dv="Digital_transformationA", controls=True, year_fe=True, entity_fe=True):
        """
        Run first differences model
        
        Parameters:
        -----------
        dv : str
            Dependent variable name
        controls : bool
            Whether to include control variables
        year_fe : bool
            Whether to include year fixed effects
        entity_fe : bool
            Whether to include entity fixed effects
            
        Returns:
        --------
        statsmodels regression result
        """
        # Create first-differenced dependent variable if it doesn't exist
        diff_var = f"D_{dv}"
        if diff_var not in self.data.columns:
            self.data[diff_var] = self.data.groupby('stkcd')[dv].diff()
            
        # Filter to post-treatment period
        post_data = self.data[self.data['Post'] == 1].copy()
        
        # Build the model formula
        formula = self._build_formula(diff_var, controls, year_fe, entity_fe)
        
        # Run the model
        model = smf.ols(formula, data=post_data).fit(
            cov_type='cluster',
            cov_kwds={'groups': post_data['stkcd'].astype(str)}
        )
        
        # Store the model
        model_name = f"first_diff_{dv}"
        if controls:
            model_name += "_controls"
        if year_fe:
            model_name += "_yearfe"
        if entity_fe:
            model_name += "_entityfe"
            
        self.models[model_name] = model
        return model

    def run_event_study(self, dv="Digital_transformationA", window=(-5, 5), controls=True, entity_fe=True):
        """
        Run event study analysis
        
        Parameters:
        -----------
        dv : str
            Dependent variable name
        window : tuple
            Time window around event, e.g. (-5, 5) for 5 years before and after
        controls : bool
            Whether to include control variables
        entity_fe : bool
            Whether to include entity fixed effects
            
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
        
        # Build and run the model
        formula = self._build_formula(dv, controls, False, entity_fe)
        model = smf.ols(formula, data=event_data).fit(
            cov_type='cluster',
            cov_kwds={'groups': event_data['stkcd'].astype(str)}
        )
        
        # Store the model
        model_name = f"event_study_{dv}"
        if controls:
            model_name += "_controls"
        if entity_fe:
            model_name += "_entityfe"
            
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

    def validate_data_for_models(self):
        """
        Validate data for model estimation
        
        Returns:
        --------
        dict: Validation results
        """
        validation = {
            "is_valid": True,
            "warnings": [],
            "critical_issues": []
        }
        
        # Check if we have the basic required variables
        required_vars = ['Treat', 'Post', 'year', 'stkcd']
        missing_vars = [var for var in required_vars if var not in self.data.columns]
        
        if missing_vars:
            validation["is_valid"] = False
            validation["critical_issues"].append(f"Missing critical variables: {missing_vars}")
            
        # Check if we have outcome variables
        outcome_vars = [var for var in self.data.columns if var.startswith('Digital_transformation')]
        if not outcome_vars:
            validation["is_valid"] = False
            validation["critical_issues"].append("No Digital_transformation variables found")
        
        # Check if we have treatment variation
        if 'Treat' in self.data.columns:
            treat_values = self.data['Treat'].unique()
            if len(treat_values) < 2:
                validation["warnings"].append(f"Only found treatment values: {treat_values}, expected both 0 and 1")
        
        # Check for staggered adoption design (no treated units pre-treatment)
        if 'Treat' in self.data.columns and 'Post' in self.data.columns:
            pre_treat = self.data[(self.data['Post'] == 0) & (self.data['Treat'] == 1)]
            if len(pre_treat) == 0:
                validation["warnings"].append("Staggered adoption design detected (no treated units pre-treatment)")
        
        # Check for high missing values
        for var in required_vars + ['Digital_transformationA', 'Digital_transformationB']:
            if var in self.data.columns:
                pct_missing = self.data[var].isna().mean() * 100
                if pct_missing > 0:
                    validation["warnings"].append(f"Missing values in {var}: {pct_missing:.1f}%")
        
        return validation