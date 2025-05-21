"""
Mechanism analysis for MSCI inclusion and digital transformation research
"""

import traceback
from utils.util import create_formula, format_regression_table, save_results_to_file
import loader.config as config
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
import os
from pathlib import Path
import sys
from loader.config import logger

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
        # Store results for each mechanism type
        self.financial_access_results = {}
        self.corporate_governance_results = {}
        self.investor_scrutiny_results = {}

    def _run_mechanism_model(self, dv, mechanism_var, post_only=False, controls=True, year_fe=True, entity_fe=True):
        """
        Run model to test a specific mechanism
        
        Parameters:
        -----------
        dv : str
            Dependent variable
        mechanism_var : str
            Mechanism variable
        post_only : bool
            Include only post-treatment periods
        controls : bool
            Include control variables
        year_fe : bool
            Include year fixed effects
        entity_fe : bool
            Include entity fixed effects
            
        Returns:
        --------
        statsmodels.regression.linear_model.RegressionResults: Model results
        """
        try:
            # Filter to post-treatment if specified
            if post_only:
                data = self.data[self.data['Post'] == 1].copy()
            else:
                data = self.data.copy()
            
            # Check if variables exist
            if dv not in data.columns:
                logger.error(f"Dependent variable {dv} not found in dataset")
                return None
                
            if mechanism_var not in data.columns:
                logger.error(f"Mechanism variable {mechanism_var} not found in dataset")
                return None
                
            if 'MSCI_clean' not in data.columns:
                logger.error("MSCI_clean variable not found in dataset")
                return None
            
            # Create interaction term
            interaction_name = f"MSCI_clean_{mechanism_var}"
            data[interaction_name] = data['MSCI_clean'] * data[mechanism_var]
            
            # Build formula
            formula = f"{dv} ~ MSCI_clean + {mechanism_var} + {interaction_name}"
            
            # Define control variables if not yet defined
            if controls and not hasattr(self, 'control_vars'):
                self.control_vars = ['age', 'TFP_OP', 'SA_index', 'WW_index', 'F050501B', 'F060101B']
            
            # Add controls (excluding the mechanism variable if it's a control)
            if controls and hasattr(self, 'control_vars'):
                other_controls = [var for var in self.control_vars 
                                if var in data.columns and var != mechanism_var]
                if other_controls:
                    formula += " + " + " + ".join(other_controls)
            
            # Add fixed effects
            if year_fe:
                formula += " + C(year)"
            
            if entity_fe:
                # Use panel_id if available, otherwise stkcd
                if 'panel_id' in data.columns:
                    formula += " + C(panel_id)"
                    cluster_var = data['panel_id'].astype(str)
                elif 'stkcd' in data.columns:
                    formula += " + C(stkcd)"
                    cluster_var = data['stkcd'].astype(str)
                else:
                    cluster_var = pd.Series(['1'] * len(data))
            else:
                # Default clustering
                if 'stkcd' in data.columns:
                    cluster_var = data['stkcd'].astype(str)
                elif 'panel_id' in data.columns:
                    cluster_var = data['panel_id'].astype(str)
                else:
                    cluster_var = pd.Series(['1'] * len(data))
            
            # Prepare data by dropping NA values
            vars_in_formula = [v for v in formula.split() if v in data.columns]
            vars_in_formula.extend([dv, 'MSCI_clean', mechanism_var, interaction_name])
            if 'stkcd' in data.columns:
                vars_in_formula.append('stkcd')
            if 'panel_id' in data.columns:
                vars_in_formula.append('panel_id')
            
            # Create a clean subset
            analysis_data = data.dropna(subset=vars_in_formula)
            
            if len(analysis_data) == 0:
                logger.error(f"No valid observations after dropping NA values for mechanism {mechanism_var}")
                return None
                
            # Log analysis details
            logger.info(f"Running mechanism analysis for {mechanism_var} on {len(analysis_data)} observations")
            logger.info(f"Formula: {formula}")
            
            # Run the model
            model = smf.ols(formula, data=analysis_data).fit(
                cov_type='cluster',
                cov_kwds={'groups': cluster_var}
            )
            
            return model
        except Exception as e:
            logger.error(f"Error estimating model for {mechanism_var}: {str(e)}")
            logger.error(f"Details: {traceback.format_exc()}")
            return None

    def analyze_financial_access(self, post_only=False):
        """
        Analyze financial access mechanisms
        
        Parameters:
        -----------
        post_only : bool, default False
            Whether to analyze only post-treatment period
            
        Returns:
        --------
        dict: Dictionary of model results
        """
        results = {}
        
        # Subset data if needed
        data = self.data.copy()
        if post_only and 'Post' in data.columns:
            data = data[data['Post'] == 1].copy()
        
        # Run models for each financial access variable
        for var in config.FINANCIAL_ACCESS_VARS:
            if var in data.columns:
                try:
                    # Create interaction term
                    data[f'MSCI_clean_{var}'] = data['MSCI_clean'] * data[var]
                    
                    # Create model formula
                    formula = f"Digital_transformationA ~ MSCI_clean + {var} + MSCI_clean_{var}"
                    
                    # Only add control variables that exist in the data
                    control_vars = []
                    for control in ["age", "TFP_OP"]:
                        if control in data.columns and control != var:
                            control_vars.append(control)
                    
                    # Add financial access variables (except the current one)
                    for fa_var in config.FINANCIAL_ACCESS_VARS:
                        if fa_var != var and fa_var in data.columns:
                            control_vars.append(fa_var)
                    
                    # Add other controls if they exist
                    for other_var in ["F050501B", "F060101B"]:
                        if other_var in data.columns:
                            control_vars.append(other_var)
                    
                    # Add controls to formula
                    if control_vars:
                        formula += " + " + " + ".join(control_vars)
                    
                    # Add year fixed effects if we have year column
                    if 'year' in data.columns:
                        formula += " + C(year)"
                    
                    print(f"Estimating model for {var}")
                    
                    # Drop missing values in all variables used in the model
                    all_vars = [v for v in formula.split(" ~ ")[1].replace("C(year)", "year").split(" + ")]
                    if 'Digital_transformationA' not in all_vars:
                        all_vars.append('Digital_transformationA')
                    if 'stkcd' in data.columns:
                        all_vars.append('stkcd')
                    
                    # Remove duplicates and ensure all variables exist
                    all_vars = list(set([v for v in all_vars if v in data.columns]))
                    
                    # Create clean subset for analysis
                    model_data = data.dropna(subset=all_vars)
                    
                    if len(model_data) == 0:
                        print(f"  No complete observations for {var} model after dropping missing values")
                        continue
                    
                    # Ensure cluster groups align with filtered data
                    if 'stkcd' in model_data.columns:
                        cluster_groups = model_data['stkcd'].astype(str)
                        model = smf.ols(formula, data=model_data).fit(
                            cov_type='cluster', 
                            cov_kwds={'groups': cluster_groups}
                        )
                    else:
                        # Run without clustering if stkcd not available
                        model = smf.ols(formula, data=model_data).fit()
                    
                    results[var] = model
                    self.financial_access_results[var] = model
                    
                    # Print key coefficients
                    print(f"  MSCI_clean coefficient: {model.params.get('MSCI_clean', 'N/A')}")
                    print(f"  {var} coefficient: {model.params.get(var, 'N/A')}")
                    print(f"  Interaction coefficient: {model.params.get(f'MSCI_clean_{var}', 'N/A')}")
                    
                except Exception as e:
                    print(f"Error estimating model for {var}: {e}")
                    
        return results

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
        cg_vars = [var for var in config.CORP_GOV_VARS if var in data.columns]

        if not cg_vars:
            print("Warning: No corporate governance variables found in dataset")
            return {}

        # Run regressions
        models = {}

        for cg_var in cg_vars:
            # Create interaction term
            data[f'MSCI_clean_{cg_var}'] = data['MSCI_clean'] * data[cg_var]

            # Define regression variables
            x_vars = ['MSCI_clean', cg_var, f'MSCI_clean_{cg_var}']

            # Add control variables (excluding the current mechanism variable)
            control_vars = [var for var in config.CONTROL_VARS 
                           if var in data.columns and var != cg_var]
            x_vars.extend(control_vars)
            
            # Add year fixed effects
            fe_vars = ["year"]

            # Create formula
            formula, _ = create_formula(dv, x_vars, fe_vars)
            
            # Run regression with entity fixed effects
            try:
                # Add entity (firm) fixed effects
                if 'stkcd' in data.columns:
                    formula += " + C(stkcd)"
                    model = smf.ols(formula, data=data).fit(
                        cov_type='cluster', 
                        cov_kwds={'groups': data['stkcd'].astype(str)}
                    )
                    
                    # Store results
                    models[cg_var] = model
                    self.corporate_governance_results[cg_var] = model
                    
                    # Print results
                    print(f"\nCorporate Governance Mechanism: {cg_var}")
                    print(format_regression_table(
                        model,
                        title=f"Effect of MSCI inclusion on {dv} through {cg_var}"
                    ))
                else:
                    print(f"Error: stkcd variable not found for clustering in {cg_var} model")
            except Exception as e:
                print(f"Error estimating model for {cg_var}: {e}")

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
        is_vars = [var for var in config.INVESTOR_SCRUTINY_VARS if var in data.columns]

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
            control_vars = [var for var in config.CONTROL_VARS 
                           if var in data.columns and var != is_var]
            x_vars.extend(control_vars)

            # Add year fixed effects
            fe_vars = ["year"]

            # Create formula
            formula, _ = create_formula(dv, x_vars, fe_vars)

            # Run regression with entity fixed effects
            try:
                # Add entity (firm) fixed effects
                if 'stkcd' in data.columns:
                    formula += " + C(stkcd)"
                    model = smf.ols(formula, data=data).fit(
                        cov_type='cluster', 
                        cov_kwds={'groups': data['stkcd'].astype(str)}
                    )
                    
                    # Store results
                    models[is_var] = model
                    self.investor_scrutiny_results[is_var] = model
                    
                    # Print results
                    print(f"\nInvestor Scrutiny Mechanism: {is_var}")
                    print(format_regression_table(
                        model,
                        title=f"Effect of MSCI inclusion on {dv} through {is_var}"
                    ))
                else:
                    print(f"Error: stkcd variable not found for clustering in {is_var} model")
            except Exception as e:
                print(f"Error estimating model for {is_var}: {e}")

        # Store all models in mechanism results
        self.mechanism_results['investor_scrutiny'] = models
        return models

    def extract_mechanism_effects(self):
        """
        Extract mechanism effects from all mechanisms analyses
        
        Returns:
        --------
        pd.DataFrame: Combined results of all mechanism analyses
        """
        results = []
        
        # Collect results from all mechanism analyses
        mechanism_results = {
            'financial_access': self.financial_access_results,
            'corporate_governance': self.corporate_governance_results,
            'investor_scrutiny': self.investor_scrutiny_results
        }
        
        # Check if we have any results
        total_models = sum(len(models) for models in mechanism_results.values())
        if total_models == 0:
            print("No mechanism analysis results available. Run analysis methods first.")
            # Return an empty DataFrame with appropriate columns instead of raising an error
            return pd.DataFrame(columns=['Mechanism', 'Variable', 'Interaction Effect', 'Std. Error', 'p-value'])
        
        # Process each mechanism type
        for mech_type, models in mechanism_results.items():
            for var, model in models.items():
                try:
                    # Extract interaction coefficient
                    interaction_var = f"MSCI_clean_{var}"
                    if interaction_var in model.params:
                        interaction_coef = model.params[interaction_var]
                        interaction_se = model.bse[interaction_var]
                        interaction_pval = model.pvalues[interaction_var]
                        
                        results.append({
                            'Mechanism': mech_type,
                            'Variable': var,
                            'Interaction Effect': interaction_coef,
                            'Std. Error': interaction_se,
                            'p-value': interaction_pval,
                            'Significant': interaction_pval < 0.05
                        })
                except Exception as e:
                    print(f"Error extracting results for {mech_type}:{var}: {e}")
        
        # Convert to DataFrame
        if results:
            df = pd.DataFrame(results)
            return df
        else:
            print("No mechanism interaction effects found in models.")
            return pd.DataFrame(columns=['Mechanism', 'Variable', 'Interaction Effect', 'Std. Error', 'p-value'])

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
        self.analyze_financial_access(post_only)
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
