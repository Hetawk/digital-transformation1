"""
Data diagnostic functions for MSCI inclusion and digital transformation analysis
"""

import pandas as pd
import numpy as np
import os
import logging
import sys
import traceback
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.stats.outliers_influence as smo
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import json
from datetime import datetime

# Import centralized logger
sys.path.append(str(Path(__file__).parent.parent.parent))
from log_setup import logger

# No need to set up logger here, using the centralized one

def analyze_missing_data(data, output_dir=None):
    """
    Analyze missing data patterns and export results
    
    Parameters:
    -----------
    data : pd.DataFrame
        The dataset to analyze
    output_dir : str or Path or None
        Directory to save results
        
    Returns:
    --------
    pd.DataFrame: Summary of missing data
    """
    if output_dir is None:
        from loader.config import TABLES_DIR, RESULTS_DIR
        output_dir = RESULTS_DIR / TABLES_DIR / "diagnostics"
    else:
        output_dir = Path(output_dir)
    
    # Create directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate missing values
    total_cells = np.product(data.shape)
    total_missing = data.isna().sum().sum()
    
    logger.info(f"Dataset shape: {data.shape}")
    logger.info(f"Total missing values: {total_missing} ({total_missing/total_cells:.2%} of all cells)")
    
    # Missing by column
    missing_by_col = pd.DataFrame({
        'Column': data.columns,
        'Total': data.shape[0],
        'Missing': data.isna().sum().values,
        'Missing %': data.isna().mean().values * 100,
        'Unique Values': [data[col].nunique() for col in data.columns]
    }).sort_values('Missing %', ascending=False)
    
    # Export results
    missing_by_col.to_csv(output_dir / "missing_data_by_column.csv", index=False)
    
    # Create visualization only if there are missing values
    cols_to_plot = missing_by_col[missing_by_col['Missing %'] > 0].head(20)
    
    if not cols_to_plot.empty:
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Missing %', y='Column', data=cols_to_plot)
        plt.title('Columns with Highest Missing Data %')
        plt.tight_layout()
        plt.savefig(output_dir / "missing_data_viz.png", dpi=300)
        logger.info(f"Created visualization of missing data for {len(cols_to_plot)} columns")
    else:
        # Create an informative plot when no missing data
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "No Missing Data Found in Dataset", 
                ha='center', va='center', fontsize=24, color='green',
                transform=plt.gca().transAxes)
        plt.axis('off')
        plt.savefig(output_dir / "missing_data_viz.png", dpi=300)
        logger.info("No missing data found. Created informative plot instead.")
    
    # Check for patterns in missing data only if there are missing values
    if total_missing > 0:
        key_cols = ['Digital_transformationA', 'Digital_transformationB', 'Treat', 'Post', 'MSCI']
        key_cols = [col for col in key_cols if col in data.columns]
        
        # Group by treatment and post
        group_vars = [v for v in ['Treat', 'Post'] if v in data.columns]
        if group_vars:
            try:
                # Fix for column name conflict
                # First create a copy of the data with just the columns we need
                data_subset = data[group_vars + key_cols].copy()
                
                # Add prefix to key_cols that overlap with group_vars to avoid conflicts
                key_cols_for_analysis = []
                for col in key_cols:
                    if col in group_vars:
                        # Skip columns used for grouping
                        continue
                    else:
                        key_cols_for_analysis.append(col)
                
                if key_cols_for_analysis:
                    # Group by treatment variables and calculate missing percentage
                    missing_by_group = data_subset.groupby(group_vars)[key_cols_for_analysis].apply(
                        lambda x: 100 * x.isna().mean()
                    ).reset_index()
                    
                    missing_by_group.to_csv(output_dir / "missing_by_treatment_group.csv", index=False)
                    logger.info("Exported missing data by treatment groups")
                else:
                    logger.warning("No unique key columns found after removing group variables")
                    # Create a simple file explaining the issue
                    with open(output_dir / "missing_by_treatment_group.txt", "w") as f:
                        f.write("Could not generate missing by treatment group analysis.\n")
                        f.write(f"All key columns ({key_cols}) are used as grouping variables ({group_vars}).\n")
            except Exception as e:
                logger.error(f"Error analyzing missing data by groups: {e}")
                # Fall back to using a simpler approach without reset_index
                try:
                    # Direct group by without reset_index
                    simple_missing = {}
                    
                    # Process each treatment group separately
                    for name, group in data.groupby(group_vars):
                        # Create a readable group name
                        if len(group_vars) == 1:
                            group_name = f"{group_vars[0]}_{name}"
                        else:
                            # Multiple grouping variables
                            group_name = "_".join([f"{var}_{val}" for var, val in zip(group_vars, name)])
                        
                        # Calculate missing percentage for key columns
                        missing_pct = {}
                        for col in key_cols:
                            if col not in group_vars:  # Skip grouping variables
                                missing_pct[col] = 100 * group[col].isna().mean()
                        
                        simple_missing[group_name] = missing_pct
                    
                    # Convert to DataFrame
                    simple_missing_df = pd.DataFrame(simple_missing).transpose()
                    simple_missing_df.to_csv(output_dir / "missing_by_treatment_group.csv")
                    logger.info("Used alternative method to export missing data by groups")
                except Exception as e2:
                    logger.error(f"Alternative method also failed: {e2}")
                    # Create an error file
                    with open(output_dir / "missing_by_treatment_group_error.txt", "w") as f:
                        f.write(f"Error analyzing missing data by groups: {e}\n")
                        f.write(f"Second attempt also failed: {e2}\n")
    else:
        # Create a simple file indicating no missing data
        with open(output_dir / "no_missing_data.txt", "w") as f:
            f.write("No missing data found in the dataset.\n")
            f.write(f"Total cells analyzed: {total_cells}\n")
            f.write(f"Dataset shape: {data.shape[0]} rows x {data.shape[1]} columns\n")
            
    logger.info(f"Missing data analysis exported to {output_dir}")
    return missing_by_col

def calculate_vif(data, variables=None, output_dir=None):
    """
    Calculate Variance Inflation Factor for detecting multicollinearity
    
    Parameters:
    -----------
    data : pd.DataFrame
        The dataset to analyze
    variables : list or None
        List of variables to analyze, defaults to numeric columns
    output_dir : str or Path or None
        Directory to save results
        
    Returns:
    --------
    pd.DataFrame: VIF values for each variable
    """
    if output_dir is None:
        from loader.config import TABLES_DIR
        output_dir = TABLES_DIR / "diagnostics"
    else:
        output_dir = Path(output_dir)
    
    # Create directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # If no variables specified, use numeric columns
    if variables is None:
        variables = data.select_dtypes(include=['number']).columns.tolist()
    
    # Check for mixed types and convert to numeric when needed
    numeric_data = pd.DataFrame()
    skipped_vars = []
    
    for var in variables:
        try:
            # Check if the column is already numeric
            if pd.api.types.is_numeric_dtype(data[var]):
                numeric_data[var] = data[var]
            else:
                # Try to convert to numeric
                numeric_data[var] = pd.to_numeric(data[var], errors='coerce')
                logger.info(f"Converted column '{var}' to numeric")
        except Exception as e:
            skipped_vars.append((var, str(e)))
            logger.warning(f"Skipping column '{var}' due to conversion error: {e}")
    
    # Check if we have any variables left
    if numeric_data.empty or len(numeric_data.columns) < 2:
        logger.error(f"Not enough numeric variables for VIF calculation. Found: {len(numeric_data.columns)}")
        
        # Save error report
        with open(output_dir / "vif_analysis_error.txt", "w") as f:
            f.write(f"Error calculating VIF: Not enough numeric variables\n")
            f.write(f"Variables requested: {variables}\n")
            f.write(f"Variables skipped: {skipped_vars}\n")
        
        return pd.DataFrame({"Variable": variables, "VIF": np.nan, "Error": "Not a numeric variable"})
    
    # Check for missing values
    data_subset = numeric_data.dropna()
    
    if len(data_subset) < len(data):
        logger.warning(f"Dropping {len(data) - len(data_subset)} rows with missing values for VIF calculation")
    
    # Calculate VIF
    vif_data = pd.DataFrame()
    vif_data["Variable"] = numeric_data.columns.tolist()
    vif_data["VIF"] = np.nan
    vif_data["Error"] = ""
    
    try:
        # Use sm.add_constant and handle errors
        X = sm.add_constant(data_subset)
        
        # Calculate VIF for each variable, handling errors individually
        for i, var in enumerate(numeric_data.columns):
            try:
                vif_value = variance_inflation_factor(X.values, i+1)
                vif_data.loc[i, "VIF"] = vif_value
            except Exception as e:
                vif_data.loc[i, "Error"] = str(e)
                logger.warning(f"Error calculating VIF for variable '{var}': {e}")
        
        # Sort by VIF value
        vif_data = vif_data.sort_values("VIF", ascending=False)
        
        # Export results
        vif_data.to_csv(output_dir / "vif_analysis.csv", index=False)
        
        # Log high VIF values
        high_vif = vif_data[(vif_data["VIF"] > 5) & (~vif_data["VIF"].isna())]
        if not high_vif.empty:
            logger.warning(f"High multicollinearity detected (VIF > 5):")
            for _, row in high_vif.iterrows():
                logger.warning(f"  {row['Variable']}: VIF = {row['VIF']:.2f}")
        
        logger.info(f"VIF analysis exported to {output_dir}")
        return vif_data
        
    except Exception as e:
        logger.error(f"Error calculating VIF: {e}")
        # Create detailed error report
        with open(output_dir / "vif_analysis_error.txt", "w") as f:
            f.write(f"Error calculating VIF: {e}\n\n")
            f.write("Data information:\n")
            f.write(f"- Number of rows: {len(data_subset)}\n")
            f.write(f"- Number of columns: {len(numeric_data.columns)}\n")
            f.write(f"- Variables: {numeric_data.columns.tolist()}\n\n")
            f.write("Data types:\n")
            for col in numeric_data.columns:
                f.write(f"- {col}: {numeric_data[col].dtype}\n")
            f.write("\nSample data (first 5 rows):\n")
            f.write(numeric_data.head().to_string())
            
        return pd.DataFrame({"Variable": numeric_data.columns.tolist(), "VIF": np.nan, "Error": str(e)})

def find_data_issues(data, file_format=None):
    """Identify potential data issues that could affect analysis quality."""
    issues_found = []
    
    # Check for incorrect CSV parsing (tab-delimited file loaded as single column)
    if file_format != 'dta' and data.shape[1] == 1:
        first_col = data.columns[0]
        if '\t' in first_col:
            # Tab character in column name suggests data should be tab-delimited
            potential_column_count = first_col.count('\t') + 1
            issues_found.append(
                f"CSV parsing issue detected: File appears to be tab-delimited but was loaded as a single column. "
                f"The data likely has {potential_column_count} columns instead of 1. "
                f"Try reloading with sep='\\t' and encoding='latin1' parameters."
            )
            
            # Try to extract some sample column names to show
            potential_columns = first_col.split('\t')
            if len(potential_columns) > 5:
                sample_cols = potential_columns[:5]
                issues_found.append(f"First few column names should be: {', '.join(sample_cols)}...")
        
        # Also check tab characters in the data values
        sample_row = data.iloc[0, 0] if len(data) > 0 else ""
        if isinstance(sample_row, str) and '\t' in sample_row:
            tab_count = sample_row.count('\t')
            issues_found.append(
                f"First data row contains {tab_count} tab characters, further confirming this is a tab-delimited file "
                f"loaded incorrectly."
            )
    
    # Special checks for Stata .dta files
    elif file_format == 'dta' and data.shape[1] == 1:
        # A Stata file loaded as a single column is highly unusual and indicates a problem
        issues_found.append(
            f"Stata file appears to have loaded as a single column, which is unexpected. "
            f"Try using pd.read_stata() directly or install the pyreadstat package for better .dta file support."
        )
    
    # Check for potential multi-dimensional indexing issues
    try:
        for col in data.select_dtypes(include=['object']).columns:
            # Look for columns with list-like or array-like contents
            sample = data[col].dropna().head(100)
            for val in sample:
                if isinstance(val, (list, np.ndarray)) or (isinstance(val, str) and ('[' in val and ']' in val)):
                    issues_found.append(f"Column '{col}' contains array-like values that could cause indexing issues")
                    break
    except:
        issues_found.append("Error checking for array-like values in columns")
    
    # Check for inconsistent data types within columns
    for col in data.columns:
        try:
            types = set()
            for val in data[col].dropna().head(100):
                types.add(type(val))
            if len(types) > 1:
                type_str = ", ".join(str(t) for t in types)
                issues_found.append(f"Column '{col}' has mixed types: {type_str}")
        except:
            pass
            
    return issues_found

def export_dataset_summary(data, output_dir, file_format=None):
    """
    Export a comprehensive summary of the dataset
    
    Parameters:
    -----------
    data : pd.DataFrame
        The dataset to analyze
    output_dir : str or Path
        Directory to save the summary
    file_format : str, optional
        Format of the input file ('dta' for Stata, 'csv' for CSV)
        
    Returns:
    --------
    Path: Path to the created summary file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary_file = output_dir / "dataset_summary.txt"
    
    with open(summary_file, "w") as f:
        # Write header
        f.write("=" * 80 + "\n")
        f.write(f"DATASET SUMMARY REPORT\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if file_format:
            f.write(f"File format: {file_format}\n")
        f.write("=" * 80 + "\n\n")
        
        # Basic dataset information
        f.write("DATASET INFORMATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Shape: {data.shape[0]} rows × {data.shape[1]} columns\n")
        f.write(f"Memory usage: {data.memory_usage(deep=True).sum() / (1024*1024):.2f} MB\n\n")
        
        # Data types
        f.write("DATA TYPES\n")
        f.write("-" * 80 + "\n")
        type_counts = data.dtypes.value_counts().to_dict()
        for dtype, count in type_counts.items():
            f.write(f"- {dtype}: {count} columns\n")
        f.write("\n")
        
        # Column information
        f.write("COLUMN DETAILS\n")
        f.write("-" * 80 + "\n")
        for col in data.columns:
            dtype = data[col].dtype
            unique_count = data[col].nunique()
            missing_count = data[col].isna().sum()
            missing_pct = missing_count / len(data) * 100
            
            f.write(f"Column: {col}\n")
            f.write(f"  Type: {dtype}\n")
            f.write(f"  Unique values: {unique_count}\n")
            f.write(f"  Missing: {missing_count} ({missing_pct:.2f}%)\n")
            
            # For numeric columns, show summary statistics
            if pd.api.types.is_numeric_dtype(data[col]):
                try:
                    stats = data[col].describe().to_dict()
                    f.write(f"  Min: {stats['min']:.4f}\n")
                    f.write(f"  Max: {stats['max']:.4f}\n")
                    f.write(f"  Mean: {stats['mean']:.4f}\n")
                    f.write(f"  Std Dev: {stats['std']:.4f}\n")
                except Exception as e:
                    f.write(f"  Error calculating statistics: {e}\n")
            
            # For categorical/object columns, show top values
            elif pd.api.types.is_object_dtype(data[col]) or pd.api.types.is_categorical_dtype(data[col]):
                try:
                    if unique_count <= 10:  # For columns with few unique values
                        top_values = data[col].value_counts().head(10)
                        f.write("  Values:\n")
                        for val, count in top_values.items():
                            f.write(f"    - {val}: {count} ({count/len(data):.2%})\n")
                    else:  # For columns with many unique values
                        top_values = data[col].value_counts().head(5)
                        f.write("  Top values:\n")
                        for val, count in top_values.items():
                            f.write(f"    - {val}: {count} ({count/len(data):.2%})\n")
                except Exception as e:
                    f.write(f"  Error calculating value counts: {e}\n")
            
            f.write("\n")
        
        # Potential data issues
        issues = find_data_issues(data, file_format)
        if issues:
            f.write("POTENTIAL DATA ISSUES\n")
            f.write("-" * 80 + "\n")
            for i, issue in enumerate(issues):
                f.write(f"{i+1}. {issue}\n")
        
    # Also create a JSON version for machine readability
    json_summary = {
        "dataset_info": {
            "rows": data.shape[0],
            "columns": data.shape[1],
            "memory_usage_mb": data.memory_usage(deep=True).sum() / (1024*1024),
            "data_types": {str(dtype): count for dtype, count in type_counts.items()},
            "generated_at": datetime.now().isoformat()
        },
        "columns": {}
    }
    
    for col in data.columns:
        col_info = {
            "dtype": str(data[col].dtype),
            "unique_count": int(data[col].nunique()),
            "missing_count": int(data[col].isna().sum()),
            "missing_percent": float(data[col].isna().mean() * 100)
        }
        
        # Add statistics for numeric columns
        if pd.api.types.is_numeric_dtype(data[col]):
            try:
                stats = data[col].describe().to_dict()
                col_info.update({
                    "min": float(stats["min"]),
                    "max": float(stats["max"]),
                    "mean": float(stats["mean"]),
                    "std": float(stats["std"])
                })
            except:
                pass
        
        json_summary["columns"][col] = col_info
    
    # Save JSON summary
    with open(output_dir / "dataset_summary.json", "w") as f:
        json.dump(json_summary, f, indent=2)
    
    return summary_file

def run_comprehensive_diagnostics(data, file_format=None, output_dir=None):
    """
    Run all diagnostic analyses in one function
    
    Parameters:
    -----------
    data : pd.DataFrame
        The dataset to analyze
    file_format : str, optional
        Format of the input file ('dta' for Stata, 'csv' for CSV)
    output_dir : str or Path or None
        Directory to save results
    """
    if output_dir is None:
        from loader.config import TABLES_DIR
        output_dir = TABLES_DIR / "diagnostics"
    else:
        output_dir = Path(output_dir)
    
    # Create directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting comprehensive data diagnostics...")
    
    # First check if data has at least the expected columns - show warning if not
    expected_cols = ['Digital_transformationA', 'Digital_transformationB', 'Treat', 'Post', 'MSCI']
    found_cols = [col for col in expected_cols if col in data.columns]
    
    if len(found_cols) == 0:
        logger.warning(f"Dataset appears incomplete - none of the expected columns found: {expected_cols}")
        logger.warning(f"Found columns: {data.columns.tolist()}")
        
        # Check if this might be a tab-delimiter issue
        if data.shape[1] == 1 and '\t' in str(data.columns[0]):
            logger.warning("The data appears to be tab-delimited but was loaded incorrectly.")
            logger.warning("Try loading with sep='\\t' parameter.")
            
            # Extract potential column names from the header
            potential_columns = data.columns[0].split('\t')
            expected_found = [col for col in expected_cols if any(col in pcol for pcol in potential_columns)]
            if expected_found:
                logger.info(f"Found {len(expected_found)}/{len(expected_cols)} expected columns in the raw data: {expected_found}")
        
        # Create special warning file
        with open(output_dir / "DATASET_LOADING_WARNING.txt", "w") as f:
            f.write("========== WARNING: DATASET LOADING ISSUE ==========\n\n")
            f.write(f"The data appears to be incomplete or incorrectly loaded.\n")
            f.write(f"Found {data.shape[1]} columns: {data.columns.tolist()}\n\n")
            f.write(f"Expected to find some of these columns: {expected_cols}\n\n")
            
            # Enhanced advice for tab-delimited files
            if data.shape[1] == 1 and '\t' in str(data.columns[0]):
                potential_columns = data.columns[0].split('\t')
                f.write(f"IMPORTANT: This appears to be a tab-delimited file loaded incorrectly!\n")
                f.write(f"The file likely has {len(potential_columns)} columns instead of 1.\n\n")
                f.write("Sample of potential column names from the header:\n")
                for i, col in enumerate(potential_columns[:10]):
                    f.write(f"  {i+1}. {col}\n")
                f.write("\nFIX: Load the data with the tab delimiter:\n")
                f.write("  python run_diagnostics.py --delimiter=tab\n")
                f.write("  OR in your code: pd.read_csv('file.csv', sep='\\t', encoding='latin1')\n\n")
            else:
                f.write("This may indicate a delimiter or CSV parsing issue.\n")
                f.write("Try loading the data with a different delimiter:\n")
                f.write("  python run_diagnostics.py --delimiter=';'\n")
                f.write("  python run_diagnostics.py --delimiter='\\t'\n\n")
                f.write("Or check the first few lines of the file with:\n")
                f.write("  head -n 5 dataset/msci_dt_processed_2010_2023.csv\n\n")
            
            f.write("The diagnostics will continue but may not be meaningful.\n")
    else:
        logger.info(f"Found {len(found_cols)}/{len(expected_cols)} expected columns")
    
    # Continue with diagnostics even if data seems incomplete
    try:
        # Export enhanced dataset summary
        summary_file = export_dataset_summary(data, output_dir, file_format=file_format)
        logger.info(f"Detailed dataset report generated: {summary_file}")
    except Exception as e:
        logger.error(f"Error generating dataset summary: {e}")
    
    try:
        # Analyze missing data
        missing_analysis = analyze_missing_data(data, output_dir)
    except Exception as e:
        logger.error(f"Error analyzing missing data: {e}")
    
    try:
        # Calculate VIF for model variables
        model_vars = ['Treat', 'Post', 'age', 'TFP_OP', 'SA_index', 'WW_index', 'F050501B', 'F060101B']
        model_vars = [var for var in model_vars if var in data.columns]
        
        if len(model_vars) >= 2:  # Need at least 2 variables for VIF
            vif_analysis = calculate_vif(data, model_vars, output_dir)
        else:
            logger.warning(f"Not enough model variables for VIF calculation. Found: {model_vars}")
    except Exception as e:
        logger.error(f"Error calculating VIF: {e}")
