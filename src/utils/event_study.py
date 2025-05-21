"""
Event study utilities for MSCI inclusion and digital transformation analysis
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import logging
from pathlib import Path

logger = logging.getLogger('digital_transformation')

def prepare_event_study_data(data, event_time_var='Event_time', outcome_var='Digital_transformationA', 
                            window=(-5, 5), min_periods=2):
    """
    Prepare data for event study analysis
    
    Parameters:
    -----------
    data : pd.DataFrame
        The dataset to analyze
    event_time_var : str
        Column name with event time (years relative to treatment)
    outcome_var : str
        Column name with outcome variable
    window : tuple
        Range of event time to include (min, max)
    min_periods : int
        Minimum number of observations required for each event time
    
    Returns:
    --------
    pd.DataFrame: Prepared data for event study
    """
    # Verify required columns exist
    required_cols = [event_time_var, outcome_var, 'Treat']
    if not all(col in data.columns for col in required_cols):
        missing = [col for col in required_cols if col not in data.columns]
        raise ValueError(f"Missing required columns: {missing}")
    
    # Filter to window
    filtered_data = data[(data[event_time_var] >= window[0]) & 
                          (data[event_time_var] <= window[1])].copy()
    
    # Check we have treated units
    if 1 not in filtered_data['Treat'].unique():
        logger.warning("No treated units in event window")
        
    # Create event time dummies
    for t in range(window[0], window[1] + 1):
        if t != -1:  # -1 is reference period
            filtered_data[f'time_{t}'] = (filtered_data[event_time_var] == t).astype(int)
            # Interact with treatment
            filtered_data[f'treat_time_{t}'] = filtered_data['Treat'] * filtered_data[f'time_{t}']
    
    # Count observations per period
    period_counts = filtered_data.groupby([event_time_var, 'Treat']).size().unstack()
    logger.info(f"Event time observations:\n{period_counts}")
    
    # Check minimum observations
    for t in range(window[0], window[1] + 1):
        treated_count = period_counts.get(1, {}).get(t, 0)
        if treated_count < min_periods:
            logger.warning(f"Few treated observations ({treated_count}) at event time {t}")
    
    return filtered_data

def run_event_study(data, outcome_var='Digital_transformationA', controls=None, 
                   window=(-5, 5), cluster_var='stkcd'):
    """
    Run event study regression
    
    Parameters:
    -----------
    data : pd.DataFrame
        Prepared event study data with time dummies
    outcome_var : str
        Column name with outcome variable
    controls : list or None
        List of control variables
    window : tuple
        Range of event time used (min, max)
    cluster_var : str
        Variable to cluster standard errors on
    
    Returns:
    --------
    statsmodels regression result
    """
    # Verify we have the necessary interaction terms
    time_vars = [f'treat_time_{t}' for t in range(window[0], window[1] + 1) if t != -1]
    missing_vars = [var for var in time_vars if var not in data.columns]
    
    if missing_vars:
        raise ValueError(f"Missing time interaction variables: {missing_vars}")
    
    # Create formula
    formula = f"{outcome_var} ~ Treat"
    
    # Add time dummies and interactions
    for t in range(window[0], window[1] + 1):
        if t != -1:  # -1 is reference period
            formula += f" + time_{t}"
            formula += f" + treat_time_{t}"
    
    # Add controls if specified
    if controls is not None:
        valid_controls = [var for var in controls if var in data.columns]
        if valid_controls:
            formula += " + " + " + ".join(valid_controls)
    
    logger.info(f"Event study formula: {formula}")
    
    # Run regression with clustered standard errors
    try:
        model = smf.ols(formula, data=data).fit(
            cov_type='cluster',
            cov_kwds={'groups': data[cluster_var].astype(str)}
        )
        
        # Extract coefficients for plotting
        coefs = []
        errors = []
        t_stats = []
        
        for t in range(window[0], window[1] + 1):
            if t != -1:
                var = f'treat_time_{t}'
                coefs.append(model.params.get(var, np.nan))
                errors.append(model.bse.get(var, np.nan))
                t_stats.append(model.tvalues.get(var, np.nan))
            else:
                # Reference period (t = -1)
                coefs.append(0)
                errors.append(0)
                t_stats.append(0)
        
        # Create results DataFrame
        event_times = list(range(window[0], window[1] + 1))
        results = pd.DataFrame({
            'event_time': event_times,
            'coefficient': coefs,
            'std_error': errors,
            't_statistic': t_stats,
            'ci_lower': [c - 1.96 * e for c, e in zip(coefs, errors)],
            'ci_upper': [c + 1.96 * e for c, e in zip(coefs, errors)]
        })
        
        # Store results in model for easy access
        model.event_study_results = results
        
        return model
        
    except Exception as e:
        logger.error(f"Error in event study regression: {e}")
        raise
        
def plot_event_study(model_or_results, title='Event Study: Dynamic Treatment Effects',
                    save_path=None):
    """
    Plot event study results
    
    Parameters:
    -----------
    model_or_results : statsmodels model or DataFrame
        Model with event_study_results attribute or DataFrame with results
    title : str
        Plot title
    save_path : str or Path or None
        Path to save the plot
    
    Returns:
    --------
    matplotlib.figure.Figure
    """
    # Extract results
    if hasattr(model_or_results, 'event_study_results'):
        results = model_or_results.event_study_results
    else:
        results = model_or_results
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot point estimates and confidence intervals
    ax.plot(results['event_time'], results['coefficient'], 'o-', color='blue', label='Coefficient')
    ax.fill_between(results['event_time'], results['ci_lower'], results['ci_upper'], 
                   alpha=0.2, color='blue')
    
    # Add reference lines
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Treatment')
    
    # Customize plot
    ax.set_xlabel('Event Time (Years Relative to Treatment)')
    ax.set_ylabel('Treatment Effect')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Annotate points
    for i, row in results.iterrows():
        if row['event_time'] != -1:  # Skip reference period
            ax.annotate(f"{row['coefficient']:.2f}",
                       (row['event_time'], row['coefficient']),
                       textcoords="offset points", 
                       xytext=(0,10), 
                       ha='center')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        logger.info(f"Event study plot saved to {save_path}")
    
    return fig
