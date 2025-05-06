"""
Simple script to run data diagnostics with robust CSV handling
"""

import os
import sys
from pathlib import Path
import pandas as pd
import argparse
import traceback
import numpy as np

# Add this directory to the path to find modules
sys.path.append(str(Path(__file__).parent))

# Import project modules
import loader.config as config
from loader.data_diagnostics import run_comprehensive_diagnostics
from loader.config import logger, configure_logging

def parse_args():
    parser = argparse.ArgumentParser(description='Run data diagnostics')
    parser.add_argument('--data-file', type=str, default=None,
                      help='Path to data file')
    parser.add_argument('--delimiter', type=str, default=None,
                      help='Delimiter for CSV files')
    parser.add_argument('--encoding', type=str, default=None,
                      help='File encoding')
    parser.add_argument('--nrows', type=int, default=None,
                      help='Number of rows to read')
    parser.add_argument('--sample', action='store_true',
                      help='Use a small sample of the data')
    return parser.parse_args()

def load_dataset(file_path, nrows=None, delimiter=None, encoding=None):
    """
    Load dataset with resilient error handling for various file formats.
    """
    logger.info(f"Loading data from: {file_path}")
    
    # Determine file format based on extension
    file_format = 'csv'  # Default
    if file_path.endswith('.dta'):
        file_format = 'dta'
    elif file_path.endswith('.pkl') or file_path.endswith('.pickle'):
        file_format = 'pickle'
        
    # Handle different file formats
    if file_format == 'dta':
        try:
            logger.info(f"Loading Stata file: {file_path}")
            # Try with pyreadstat first if available
            try:
                import pyreadstat
                df, meta = pyreadstat.read_dta(file_path)
                logger.info(f"Success! Read {len(df)} rows with {df.shape[1]} columns using pyreadstat")
                return df, file_format
            except ImportError:
                # Fall back to pandas
                df = pd.read_stata(file_path, convert_categoricals=False)
                if nrows is not None:
                    df = df.head(nrows)
                logger.info(f"Success! Read {len(df)} rows with {df.shape[1]} columns using pandas")
                return df, file_format
        except Exception as e:
            logger.warning(f"Failed to load Stata file: {e}")
            logger.warning("Attempting to load as CSV...")
            # Fall through to CSV loading
    elif file_format == 'pickle':
        try:
            logger.info(f"Loading pickle file: {file_path}")
            df = pd.read_pickle(file_path)
            if nrows is not None:
                df = df.head(nrows)
            logger.info(f"Success! Read {len(df)} rows with {df.shape[1]} columns")
            return df, file_format
        except Exception as e:
            logger.warning(f"Failed to load pickle file: {e}")
            logger.warning("Attempting to load as CSV...")
            # Fall through to CSV loading
    
    # Define loading strategies for CSV files in order of preference
    loading_strategies = []
    
    # If delimiter is specified, try it first
    if delimiter:
        if delimiter == 'tab':
            delimiter = '\t'
        loading_strategies.append({
            'sep': delimiter, 
            'engine': 'python', 
            'on_bad_lines': 'skip',
            'encoding': encoding or 'utf-8'
        })
        
        # Also try with latin1 encoding - this is often needed for tab-delimited files
        loading_strategies.append({
            'sep': delimiter, 
            'engine': 'python', 
            'on_bad_lines': 'skip',
            'encoding': 'latin1'
        })
    else:
        # Default strategies
        loading_strategies.extend([
            {'engine': 'python', 'on_bad_lines': 'skip'},
            {'sep': '\t', 'engine': 'python', 'on_bad_lines': 'skip'},
            {'engine': 'python', 'sep': None, 'on_bad_lines': 'skip'},
            {'engine': 'python', 'encoding': 'utf-8', 'on_bad_lines': 'skip'},
            {'engine': 'python', 'encoding': 'latin1', 'on_bad_lines': 'skip'},
            # Add strategy that combines tab delimiter with Latin-1 encoding
            {'sep': '\t', 'engine': 'python', 'encoding': 'latin1', 'on_bad_lines': 'skip'},
            {'sep': '\t', 'engine': 'c', 'on_bad_lines': 'skip'},
        ])
    
    loaded_single_column = False
    single_column_data = None
    
    for i, params in enumerate(loading_strategies, 1):
        try:
            logger.info(f"Attempt {i}: Reading with {params}")
            
            if nrows is not None:
                params['nrows'] = nrows
                
            df = pd.read_csv(file_path, **params)
            logger.info(f"Success! Read {len(df)} rows with {df.shape[1]} columns")
            
            # Check if we have a single column with tabs - this indicates the delimiter is wrong
            if df.shape[1] == 1 and '\t' in str(df.iloc[0, 0]):
                logger.warning("File loaded as single column with tabs. Retrying with explicit tab delimiter.")
                loaded_single_column = True
                single_column_data = df  # Save for fallback
                continue
                
            return df, 'csv'
            
        except Exception as e:
            logger.warning(f"Attempt {i} failed: {str(e)}")
    
    # If we got here, all attempts failed but we might have a single column loaded
    if loaded_single_column and single_column_data is not None:
        logger.warning("All loading attempts failed. Attempting to manually parse the tab-delimited data.")
        try:
            # Get the header row
            header = single_column_data.columns[0]
            headers = header.split('\t')
            
            # Process a small sample of rows
            rows_to_process = min(5000, len(single_column_data))
            sample_data = []
            
            for idx in range(rows_to_process):
                try:
                    row_str = single_column_data.iloc[idx, 0]
                    if isinstance(row_str, str):
                        values = row_str.split('\t')
                        # Ensure the row has the right number of columns
                        if len(values) < len(headers):
                            values.extend([''] * (len(headers) - len(values)))
                        elif len(values) > len(headers):
                            values = values[:len(headers)]
                        sample_data.append(values)
                except Exception as row_e:
                    logger.warning(f"Error processing row {idx}: {str(row_e)}")
                    continue
            
            # Create a DataFrame from the processed data
            if sample_data:
                manual_df = pd.DataFrame(sample_data, columns=headers)
                logger.info(f"Manually created DataFrame with {len(manual_df)} rows and {len(headers)} columns")
                return manual_df, 'csv'
        except Exception as parse_e:
            logger.error(f"Manual parsing failed: {str(parse_e)}")
    
    # Last resort: try reading first few lines with explicit encoding
    try:
        with open(file_path, 'r', encoding='latin1') as f:
            header = f.readline().strip()
            if '\t' in header:
                headers = header.split('\t')
                
                # Read a few more lines for sample data
                sample_data = []
                for _ in range(10):  # Read 10 lines
                    line = f.readline().strip()
                    if line:
                        values = line.split('\t')
                        # Normalize length
                        if len(values) < len(headers):
                            values.extend([''] * (len(headers) - len(values)))
                        elif len(values) > len(headers):
                            values = values[:len(headers)]
                        sample_data.append(values)
                
                # Create DataFrame from sample
                if sample_data:
                    emergency_df = pd.DataFrame(sample_data, columns=headers)
                    logger.info(f"Created emergency sample with {len(emergency_df)} rows for diagnostics")
                    return emergency_df, 'csv'
    except Exception as last_e:
        logger.error(f"Last resort parsing failed: {str(last_e)}")
    
    logger.error(f"Failed to load {file_path} after all attempts.")
    return None, None

def main():
    args = parse_args()
    
    # Setup the logger
    configure_logging()
    
    # Print header
    print("=" * 80)
    print(" " * 28 + "RUNNING DATA DIAGNOSTICS" + " " * 28)
    print("=" * 80)
    
    # Determine input file
    input_file = args.data_file
    if input_file is None:
        # Default data locations to check
        candidates = [
            Path('dataset/msci_dt_processed_2010_2023.csv'),
            Path('dataset/preprocessed_backup.csv'),
            Path('dataset/msci_dt_processed_2010_2023.dta')
        ]
        for candidate in candidates:
            if candidate.exists():
                input_file = str(candidate)
                logger.info(f"Using auto-detected data file: {input_file}")
                break
        
        if input_file is None:
            logger.error("No data file specified and no default data file found.")
            return
    
    # Load dataset with appropriate parameters
    nrows = 10000 if args.sample else args.nrows
    df, file_format = load_dataset(input_file, nrows=nrows, 
                             delimiter=args.delimiter, 
                             encoding=args.encoding)
    
    if df is None:
        logger.error("Failed to load dataset. Exiting.")
        return
    
    # Run diagnostics with file format information
    run_comprehensive_diagnostics(df, file_format=file_format)
    
    print("\nData diagnostics completed. Results available in:")
    print("  /data2/enoch/ekd_coding_env/patience/digital_transformation/results/tables/diagnostics/")

if __name__ == "__main__":
    main()
