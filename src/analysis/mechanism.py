
from src.utils import create_formula, format_regression_table, save_results_to_file
import config
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
import os
from pathlib import Path
import sys
from config import logger

sys.path.append(str(Path(__file__).parent.parent.parent))

def run_mechanism_analysis(self, dv, mechanism_var, controls=True, year_fe=True, entity_fe=True):
    """
    Run analysis of mechanism variables
    
    Parameters:
    -----------
    dv : str
        Dependent variable
    mechanism_var : str
        Mechanism variable to test
    controls : bool
        Whether to include control variables
    year_fe : bool
        Whether to include year fixed effects
    entity_fe : bool
        Whether to include entity fixed effects
        
    Returns:
    --------
    statsmodels regression results
    """
    try:
        # Check if variables exist
        if dv not in self.data.columns:
            logger.error(f"Dependent variable {dv} not found in dataset")
            return None
            
        if mechanism_var not in self.data.columns:
            logger.error(f"Mechanism variable {mechanism_var} not found in dataset")
            return None
            
        if 'MSCI_clean' not in self.data.columns:
            logger.error("MSCI_clean variable not found in dataset")
            return None
            
        # Create interaction term
        interaction_name = f"MSCI_clean_{mechanism_var}"
        self.data[interaction_name] = self.data['MSCI_clean'] * self.data[mechanism_var]
        
        # Build formula
        formula = f"{dv} ~ MSCI_clean + {mechanism_var} + {interaction_name}"
        
        # Define control variables if not yet defined
        if controls and not hasattr(self, 'control_vars'):
            self.control_vars = ['age', 'TFP_OP', 'SA_index', 'WW_index', 'F050501B', 'F060101B']
        
        # Add controls (excluding the mechanism variable if it's a control)
        if controls and hasattr(self, 'control_vars'):
            other_controls = [var for var in self.control_vars 
                              if var in self.data.columns and var != mechanism_var]
            if other_controls:
                formula += " + " + " + ".join(other_controls)
        
        # Add fixed effects
        if year_fe:
            formula += " + C(year)"
        
        if entity_fe:
            # Use panel_id if available
            if 'panel_id' in self.data.columns:
                formula += " + C(panel_id)"
            elif 'stkcd' in self.data.columns:
                formula += " + C(stkcd)"
        
        # Prepare data by dropping NA values
        vars_in_formula = [v for v in formula.split() if v in self.data.columns]
        vars_in_formula.extend([dv, 'MSCI_clean', mechanism_var, interaction_name])
        if 'stkcd' in self.data.columns:
            vars_in_formula.append('stkcd')
        
        # Create a clean subset
        analysis_data = self.data.dropna(subset=vars_in_formula)
        
        if len(analysis_data) == 0:
            logger.error(f"No valid observations after dropping NA values for mechanism {mechanism_var}")
            return None
            
        # Prepare clustering variable
        if 'stkcd' in analysis_data.columns:
            cluster_var = analysis_data['stkcd'].astype(str)
        else:
            if 'panel_id' in analysis_data.columns:
                cluster_var = analysis_data['panel_id'].astype(str)
            else:
                cluster_var = pd.Series(['1'] * len(analysis_data))
        
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
        logger.error(f"Formula: {formula}")
        logger.error(f"Details: {traceback.format_exc()}")
        return None
