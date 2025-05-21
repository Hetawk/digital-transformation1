"""
Utility functions for the MSCI inclusion and digital transformation analysis
"""

import src.loader.config as config
import pandas as pd
import numpy as np
import os
from pathlib import Path
import statsmodels.formula.api as smf
import statsmodels.api as sm
import sys

sys.path.append(str(Path(__file__).parent.parent))


def save_results_to_file(content, filename, file_format='txt'):
    """
    Save analysis results to a file

    Parameters:
    -----------
    content : str
        Content to save to the file
    filename : str
        Name of the file without extension
    file_format : str, default 'txt'
        Format of the file ('txt', 'csv', 'tex', or 'rtf')
    """
    formats = {
        'txt': config.TABLES_DIR / f"{filename}.txt",
        'csv': config.TABLES_DIR / f"{filename}.csv",
        'tex': config.TABLES_DIR / f"{filename}.tex",
        'rtf': config.TABLES_DIR / f"{filename}.rtf"
    }

    if file_format not in formats:
        raise ValueError(f"Unsupported file format: {file_format}")

    file_path = formats[file_format]

    with open(file_path, 'w') as f:
        f.write(content)

    print(f"Results saved to {file_path}")


def create_formula(y_var, x_vars=None, fe_vars=None, interact_vars=None, subset_condition=None):
    """
    Create a regression formula for statsmodels

    Parameters:
    -----------
    y_var : str
        Dependent variable
    x_vars : list or None
        Independent variables
    fe_vars : list or None
        Fixed effect variables to include as categorical (will be converted to C(var))
    interact_vars : list or None
        List of tuples (var1, var2) for interaction terms
    subset_condition : str or None
        Condition for subsetting data (e.g., "Post == 1")

    Returns:
    --------
    formula : str
        Regression formula
    subset : str or None
        Subset condition
    """
    x_terms = []

    # Add main independent variables
    if x_vars:
        x_terms.extend(x_vars)

    # Add fixed effects
    if fe_vars:
        x_terms.extend([f"C({var})" for var in fe_vars])

    # Add interaction terms
    if interact_vars:
        for var1, var2 in interact_vars:
            if var2.startswith("C("):
                # If var2 is already a categorical term
                x_terms.append(f"{var1}:{var2}")
            else:
                x_terms.append(f"{var1}*{var2}")

    # Construct the formula
    formula = f"{y_var} ~ {' + '.join(x_terms)}"

    return formula, subset_condition


def cluster_robust_se(model, cluster_var):
    """
    Calculate cluster-robust standard errors

    Parameters:
    -----------
    model : statsmodels regression model
        The fitted regression model
    cluster_var : array-like
        The variable to cluster on

    Returns:
    --------
    cov_matrix : ndarray
        Cluster-robust covariance matrix
    """
    u = model.resid
    X = model.model.exog

    # Unique cluster IDs
    clusters = np.unique(cluster_var)
    n_clusters = len(clusters)

    # Initialize meat of the sandwich
    meat = np.zeros((X.shape[1], X.shape[1]))

    # Compute meat for each cluster
    for cluster in clusters:
        cluster_mask = (cluster_var == cluster)
        X_cluster = X[cluster_mask]
        u_cluster = u[cluster_mask].reshape(-1, 1)
        meat += X_cluster.T @ u_cluster @ u_cluster.T @ X_cluster

    # Compute bread of the sandwich
    bread = np.linalg.inv(X.T @ X)

    # Compute the sandwich
    cov_matrix = bread @ meat @ bread

    # Adjustment factor
    n = X.shape[0]
    k = X.shape[1]
    dfc = (n - 1) / (n - k) * n_clusters / (n_clusters - 1)

    # Return adjusted covariance matrix
    return cov_matrix * dfc


def format_regression_table(model, cluster_var=None, title=None):
    """
    Format regression results as a text table

    Parameters:
    -----------
    model : statsmodels regression model
        The fitted regression model
    cluster_var : array-like or None
        The variable to cluster on for cluster-robust standard errors
    title : str or None
        Title for the regression table

    Returns:
    --------
    str : Formatted table as a string
    """
    # Get model summary
    if cluster_var is not None:
        # Compute cluster-robust standard errors
        cov_matrix = cluster_robust_se(model, cluster_var)
        se = np.sqrt(np.diag(cov_matrix))
        t_stats = model.params / se
        p_values = 2 * (1 - abs(np.minimum(t_stats, -t_stats)).cdf(0))

        # Create summary table with cluster-robust SEs
        summary = pd.DataFrame({
            'Coefficient': model.params,
            'Std. Error': se,
            't-stat': t_stats,
            'p-value': p_values
        })
    else:
        summary = pd.DataFrame({
            'Coefficient': model.params,
            'Std. Error': model.bse,
            't-stat': model.tvalues,
            'p-value': model.pvalues
        })

    # Add stars for significance
    summary[''] = ''
    summary.loc[summary['p-value'] < 0.1, ''] = '*'
    summary.loc[summary['p-value'] < 0.05, ''] = '**'
    summary.loc[summary['p-value'] < 0.01, ''] = '***'

    # Format numbers
    summary['Coefficient'] = summary['Coefficient'].map('{:.3f}'.format)
    summary['Std. Error'] = summary['Std. Error'].map('({:.3f})'.format)

    # Create formatted string table
    result = []

    if title:
        result.append('=' * 80)
        result.append(title.center(80))
        result.append('=' * 80)

    result.append('-' * 80)
    result.append(
        f"{'Variable':<30}{'Coefficient':<15}{'Std. Error':<15}{'t-stat':<10}{'p-value':<10}{'':>5}")
    result.append('-' * 80)

    for idx, row in summary.iterrows():
        var_name = idx
        coef = row['Coefficient']
        se = row['Std. Error']
        t = f"{row['t-stat']:.2f}"
        p = f"{row['p-value']:.3f}"
        stars = row['']
        result.append(
            f"{var_name:<30}{coef:<15}{se:<15}{t:<10}{p:<10}{stars:>5}")

    result.append('-' * 80)
    result.append(f"Observations: {model.nobs}")
    if hasattr(model, 'rsquared'):
        result.append(f"R-squared: {model.rsquared:.3f}")
    if hasattr(model, 'rsquared_adj'):
        result.append(f"Adjusted R-squared: {model.rsquared_adj:.3f}")

    if cluster_var is not None:
        num_clusters = len(np.unique(cluster_var))
        result.append(f"Number of clusters: {num_clusters}")

    return '\n'.join(result)


def check_balance(data, treatment_var, balance_vars, standardized=False, allow_missing_groups=False):
    """
    Check balance between treatment and control groups
    
    Supports staggered adoption designs where treated units may only exist
    in the post-treatment period. For such designs, this function provides
    descriptive statistics and recommends appropriate estimation methods
    following current academic standards.

    Parameters:
    -----------
    data : pd.DataFrame
        The dataset
    treatment_var : str
        Name of the treatment variable (0/1)
    balance_vars : list
        List of variables to check for balance
    standardized : bool, default False
        If True, compute standardized differences
    allow_missing_groups : bool, default False
        If True, continue with available groups instead of exiting when
        one group is missing (important for staggered adoption designs)

    Returns:
    --------
    pd.DataFrame : Balance check results with metadata on design type
    """
    import logging
    import sys
    import numpy as np
    
    logger = logging.getLogger('digital_transformation')
    
    results = []

    # First check if there are both treatment and control groups
    treat_groups = data[treatment_var].dropna().unique()
    
    # Detect design type
    design_type = "complete_did" if (0 in treat_groups and 1 in treat_groups) else "staggered_adoption"
    if design_type == "staggered_adoption":
        logger.info(f"Detected staggered adoption design with treatment groups: {treat_groups}")
        if not allow_missing_groups:
            logger.error(f"Staggered adoption design detected. Set allow_missing_groups=True to proceed with analysis.")
            sys.exit(1)
    
    if len(treat_groups) < 2:
        message = f"Found only {treat_groups} for {treatment_var}. This indicates a staggered adoption design."
        if allow_missing_groups:
            logger.warning(message + " Proceeding with limited comparison using available groups.")
            logger.warning("Note: Standard balance tests between treated and control units in pre-treatment")
            logger.warning("period cannot be performed due to absence of treated units pre-treatment.")
        else:
            logger.error(message + " Set allow_missing_groups=True to proceed with alternative analysis.")
            sys.exit(1)
    
    # Check if we have enough observations in each group
    treat_counts = data[treatment_var].value_counts()
    
    # Check for control group
    control_exists = 0 in treat_counts.index and treat_counts[0] >= 5
    if not control_exists:
        message = f"Insufficient control observations (Treat=0): {0 if 0 not in treat_counts.index else treat_counts[0]}"
        if allow_missing_groups:
            logger.warning(message + " Balance comparison will be limited.")
        else:
            logger.error(message)
            sys.exit(1)
    
    # Check for treatment group
    treatment_exists = 1 in treat_counts.index and treat_counts[1] >= 5
    if not treatment_exists:
        message = f"Insufficient treatment observations (Treat=1): {0 if 1 not in treat_counts.index else treat_counts[1]}"
        if allow_missing_groups:
            logger.warning(message + " Balance comparison will be limited.")
            if design_type == "staggered_adoption":
                logger.warning("For staggered adoption designs, consider these state-of-the-art approaches:")
                logger.warning("1. Callaway & Sant'Anna (2021): estimator for staggered treatment timing")
                logger.warning("2. Sun & Abraham (2021): interaction-weighted estimator for heterogeneous effects")
                logger.warning("3. de Chaisemartin & D'Haultfœuille (2020): estimator robust to treatment effect heterogeneity")
                logger.warning("4. Event study approach with binned endpoints (Schmidheiny & Siegloch, 2020)")
        else:
            logger.error(message)
            sys.exit(1)

    for var in balance_vars:
        if var not in data.columns:
            logger.warning(f"Variable {var} not found in dataset, skipping in balance check")
            continue

        # Get treatment and control groups
        treat_data = data[data[treatment_var] == 1][var].dropna() if treatment_exists else pd.Series()
        control_data = data[data[treatment_var] == 0][var].dropna() if control_exists else pd.Series()

        # Calculate statistics based on available data
        treat_mean = treat_data.mean() if len(treat_data) > 0 else np.nan
        control_mean = control_data.mean() if len(control_data) > 0 else np.nan
        treat_sd = treat_data.std() if len(treat_data) > 1 else np.nan
        control_sd = control_data.std() if len(control_data) > 1 else np.nan
        
        # Calculate difference if both means are available
        difference = treat_mean - control_mean if not np.isnan(treat_mean) and not np.isnan(control_mean) else np.nan

        # Calculate t-test only if both groups have data and non-zero variance
        tstat, pvalue = np.nan, np.nan
        if len(treat_data) > 1 and len(control_data) > 1 and not np.isnan(treat_sd) and not np.isnan(control_sd) and treat_sd > 0 and control_sd > 0:
            try:
                from scipy import stats
                tstat, pvalue = stats.ttest_ind(
                    treat_data, control_data, equal_var=False)
            except Exception as e:
                logger.warning(f"Error calculating t-test for {var}: {e}")

        # Calculate standardized difference if both SDs are available
        std_difference = np.nan
        if not np.isnan(treat_sd) and not np.isnan(control_sd) and not np.isnan(difference):
            pooled_sd = np.sqrt((treat_sd**2 + control_sd**2) / 2)
            std_difference = difference / pooled_sd if pooled_sd > 0 else np.nan

        results.append({
            'Variable': var,
            'Treatment Mean': treat_mean,
            'Control Mean': control_mean,
            'Difference': difference,
            'Std. Difference': std_difference,
            't-statistic': tstat,
            'p-value': pvalue,
            'Treatment N': len(treat_data),
            'Control N': len(control_data)
        })

    # Create a DataFrame with results
    result_df = pd.DataFrame(results)
    
    # Add metadata to results
    result_df.attrs['treatment_exists'] = treatment_exists
    result_df.attrs['control_exists'] = control_exists
    result_df.attrs['staggered_adoption'] = design_type == "staggered_adoption"
    result_df.attrs['design_type'] = design_type
    
    # Provide recommendations for staggered adoption design
    if result_df.attrs['staggered_adoption']:
        logger.info("====== STAGGERED ADOPTION DESIGN DETECTED ======")
        logger.info("Standard DiD estimation may be biased due to heterogeneous treatment effects.")
        logger.info("Recommended approaches (current academic standards):")
        logger.info("1. Callaway & Sant'Anna (2021): group-time average treatment effects")
        logger.info("   - Handles staggered adoption with heterogeneous effects")
        logger.info("   - Implemented in R 'did' package or Python 'econtools'")
        logger.info("2. Sun & Abraham (2021): interaction-weighted estimator")
        logger.info("   - Robust to treatment effect heterogeneity across cohorts")
        logger.info("3. de Chaisemartin & D'Haultfœuille (2020): DIDM estimator")
        logger.info("   - Accounts for dynamic treatment effects")
        logger.info("   - Implemented in Stata 'did_multiplegt' or R 'DIDmultiplegt'")
        logger.info("4. Borusyak, Jaravel, Spiess (2021): imputation estimator")
        logger.info("   - Efficient under parallel trends")
        logger.info("5. Alternative approach: robust event study")
        logger.info("   - With proper binning of endpoints (Schmidheiny & Siegloch, 2020)")
        logger.info("   - Can estimate anticipation effects and dynamic treatment effects")
        logger.info("See README for implementation details and references.")
    
    return result_df


def match_nearest_neighbor(data, treatment_var, outcome_var, matching_vars, n_neighbors=1):
    """
    Perform nearest neighbor matching

    Parameters:
    -----------
    data : pd.DataFrame
        The dataset
    treatment_var : str
        Name of the treatment variable (0/1)
    outcome_var : str
        Name of the outcome variable
    matching_vars : list
        List of variables to use for matching
    n_neighbors : int, default 1
        Number of neighbors to match

    Returns:
    --------
    tuple : (matched_data, ate, ate_se)
        matched_data: DataFrame of matched data
        ate: Average treatment effect
        ate_se: Standard error of ATE
    """
    from sklearn.neighbors import NearestNeighbors

    # Separate treated and control units
    treated = data[data[treatment_var] == 1]
    control = data[data[treatment_var] == 0]

    # Variables to match on
    X_treated = treated[matching_vars].values
    X_control = control[matching_vars].values

    # Standardize variables for matching
    X_mean = X_treated.mean(axis=0)
    X_std = X_treated.std(axis=0)
    X_treated_std = (X_treated - X_mean) / X_std
    X_control_std = (X_control - X_mean) / X_std

    # Find nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(X_control_std)
    distances, indices = nbrs.kneighbors(X_treated_std)

    # Extract matched controls
    matched_controls = []

    for i, idx_array in enumerate(indices):
        for j, idx in enumerate(idx_array):
            matched_controls.append({
                'treated_idx': treated.index[i],
                'control_idx': control.index[idx],
                'distance': distances[i][j],
                'outcome_treated': treated[outcome_var].iloc[i],
                'outcome_control': control[outcome_var].iloc[idx]
            })

    # Create matched dataset
    matched_data = pd.DataFrame(matched_controls)

    # Calculate ATE
    matched_data['effect'] = matched_data['outcome_treated'] - \
        matched_data['outcome_control']
    ate = matched_data['effect'].mean()
    ate_se = matched_data['effect'].std() / np.sqrt(len(matched_data))

    return matched_data, ate, ate_se


def prepare_event_study_data(data, event_time_var, outcome_var, window=(-5, 5)):
    """
    Prepare data for event study analysis

    Parameters:
    -----------
    data : pd.DataFrame
        The dataset
    event_time_var : str
        Name of the event time variable (relative to treatment)
    outcome_var : str
        Name of the outcome variable
    window : tuple, default (-5, 5)
        Event window (pre, post) periods to include

    Returns:
    --------
    pd.DataFrame : Data prepared for event study
    """
    # Filter to event window
    min_time, max_time = window
    event_data = data[(data[event_time_var] >= min_time) &
                      (data[event_time_var] <= max_time)].copy()

    # Create dummy variables for each event time (except the reference period)
    reference_period = -1  # t-1 is typically the reference period

    for t in range(min_time, max_time + 1):
        if t != reference_period:
            event_data[f'time_{t}'] = (
                event_data[event_time_var] == t).astype(int)

    return event_data
