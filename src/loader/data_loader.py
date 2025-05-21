# src/loader/data_loader.py
"""
Centralized data loading functionality for MSCI inclusion and digital transformation analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
import chardet

# Import logger
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.log_setup import logger

def load_dataset(file_path, delimiter=None, encoding=None, nrows=None, sample=False):
    """
    Load dataset with robust error handling and encoding detection
    
    Parameters:
    -----------
    file_path : str or Path
        Path to data file
    delimiter : str, optional
        Delimiter for CSV files (e.g., ',', '\t') 
    encoding : str, optional
        File encoding (e.g., 'latin1', 'utf-8')
    nrows : int, optional
        Number of rows to read
    sample : bool, default False
        Whether to load only a sample of the data
        
    Returns:
    --------
    tuple: (pd.DataFrame, str)
        DataFrame and file format ('dta' or 'csv')
    """
    # Set parameters for sample loading
    if sample and nrows is None:
        nrows = 10000
    
    file_path = Path(file_path)
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return None, None
    
    # Get file format from extension
    file_format = file_path.suffix.lower()[1:]
    
    # Handle tab delimiter string
    if delimiter == 'tab':
        delimiter = '\t'
    
    # For Stata files
    if file_format == 'dta':
        try:
            # Try with pyreadstat first for better encoding handling
            try:
                import pyreadstat
                logger.info(f"Loading Stata file with pyreadstat: {file_path}")
                df, meta = pyreadstat.read_dta(str(file_path))
                return df, 'dta'
            except Exception as e:
                logger.warning(f"Error loading with pyreadstat: {e}")
                logger.info("Falling back to pandas read_stata")
            
            # Fallback to pandas
            logger.info(f"Loading Stata file with pandas: {file_path}")
            df = pd.read_stata(file_path, convert_categoricals=False)
            return df, 'dta'
        except Exception as e:
            logger.error(f"Error loading Stata file: {e}")
            logger.warning("Will attempt to load file as CSV instead")
            # Continue to CSV loading as fallback
    
    # For very large CSV files, try chunked reading first
    logger.info(f"Attempting chunked loading for large dataset: {file_path}")
    try:
        # Set default delimiter to tab if not specified (based on previous success)
        if delimiter is None:
            delimiter = '\t'
            logger.info("No delimiter specified, using tab delimiter as default")
        
        # Use chunk-based loading for large files
        chunk_size = 25000  # Use smaller chunks to avoid buffer issues
        chunks = []
        
        # Try with Python engine first as it's more robust for problematic files
        try:
            logger.info("Trying Python engine chunked loading...")
            chunk_iter = pd.read_csv(
                file_path,
                sep=delimiter,
                encoding='latin1',
                engine='python',
                chunksize=chunk_size, 
                nrows=nrows,
                on_bad_lines='skip'
            )
            
            # Get the first few chunks
            max_chunks = 20 if nrows is None else max(1, nrows // chunk_size + 1)
            for i, chunk in enumerate(chunk_iter):
                if i >= max_chunks:
                    break
                chunks.append(chunk)
                logger.info(f"Loaded chunk {i+1} with shape {chunk.shape}")
            
            # Combine chunks
            if chunks:
                df = pd.concat(chunks, ignore_index=True)
                logger.info(f"Successfully loaded {len(df)} rows × {df.shape[1]} columns using Python engine chunks")
                return df, 'csv'
        except Exception as e:
            logger.warning(f"Python engine chunked loading failed: {e}")
            
            # Try C engine with chunked loading
            logger.info("Falling back to C engine with chunked loading")
            try:
                chunks = []
                chunk_iter = pd.read_csv(
                    file_path,
                    sep=delimiter,
                    encoding='latin1',
                    engine='c',
                    chunksize=chunk_size,
                    nrows=nrows,
                    low_memory=True,
                    on_bad_lines='warn'  # Updated parameter name
                )
                
                # Get the first few chunks
                max_chunks = 20 if nrows is None else max(1, nrows // chunk_size + 1)
                for i, chunk in enumerate(chunk_iter):
                    if i >= max_chunks:
                        break
                    chunks.append(chunk)
                    logger.info(f"Loaded chunk {i+1} with shape {chunk.shape}")
                
                # Combine chunks
                if chunks:
                    df = pd.concat(chunks, ignore_index=True)
                    logger.info(f"Successfully loaded {len(df)} rows × {df.shape[1]} columns using C engine chunks")
                    return df, 'csv'
            except Exception as e2:
                logger.warning(f"C engine chunked loading failed: {e2}")
        
    except Exception as e:
        logger.warning(f"All chunked loading approaches failed: {e}")
    
    # If chunked loading failed, try manual analysis and loading
    logger.info("Attempting extremely conservative loading approach")
    try:
        # Try to read just the first 1000 lines to get a sample
        with open(file_path, 'rb') as f:
            lines = []
            for i in range(1000):
                try:
                    line = f.readline().decode('latin1', errors='replace')
                    if not line:
                        break
                    lines.append(line)
                except:
                    break
        
        if lines:
            # Determine delimiter if not specified
            if delimiter is None:
                header = lines[0]
                delimiters = ['\t', ',', ';']
                counts = {d: header.count(d) for d in delimiters}
                delimiter = max(counts.items(), key=lambda x: x[1])[0]
                logger.info(f"Auto-detected delimiter: '{delimiter}' (found {counts[delimiter]} occurrences)")
            
            # Create a StringIO object to read the sample
            import io
            data_str = ''.join(lines)
            data_io = io.StringIO(data_str)
            
            # Try to read the sample
            try:
                sample_df = pd.read_csv(
                    data_io,
                    sep=delimiter,
                    encoding='latin1',
                    engine='python',
                    on_bad_lines='skip'
                )
                
                logger.info(f"Successfully loaded sample with {len(sample_df)} rows × {sample_df.shape[1]} columns")
                
                # If sample loaded successfully, try to read the full file
                if len(sample_df) > 0:
                    try:
                        logger.info("Trying to load full file with established parameters...")
                        df = pd.read_csv(
                            file_path,
                            sep=delimiter,
                            encoding='latin1',
                            engine='python',
                            on_bad_lines='skip',
                            nrows=nrows,
                            memory_map=True
                        )
                        logger.info(f"Successfully loaded full file: {df.shape}")
                        return df, 'csv'
                    except Exception as e:
                        logger.warning(f"Failed to load full file: {e}")
                        # Return the sample as better than nothing
                        logger.warning(f"Returning sample data ({len(sample_df)} rows) for analysis")
                        return sample_df, 'csv'
            except Exception as e:
                logger.error(f"Failed to read sample data: {e}")
    except Exception as e:
        logger.error(f"Conservative loading approach failed: {e}")
    
    # If all else fails
    logger.error(f"All attempts to load {file_path} failed")
    return None, None

def validate_dataframe(df):
    """
    Validate if the dataframe contains expected columns and structure
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to validate
    
    Returns:
    --------
    dict: Validation results with issues and warnings
    """
    result = {
        "is_valid": True,
        "issues": [],
        "warnings": [],
        "expected_columns_found": []  # Add this key to the result dictionary
    }
    
    # Check if dataframe is empty
    if df is None:
        result["is_valid"] = False
        result["issues"].append("DataFrame is None")
        return result
    
    if df.empty:
        result["is_valid"] = False
        result["issues"].append("DataFrame is empty")
        return result
    
    # Check expected columns
    expected_cols = ['Digital_transformationA', 'Digital_transformationB', 'Treat', 'Post', 'MSCI']
    missing_cols = [col for col in expected_cols if col not in df.columns]
    
    # Store found expected columns
    result["expected_columns_found"] = [col for col in expected_cols if col in df.columns]
    
    if missing_cols:
        if len(missing_cols) == len(expected_cols):
            result["is_valid"] = False
            result["issues"].append(f"Missing all expected columns: {missing_cols}")
        else:
            result["warnings"].append(f"Missing some expected columns: {missing_cols}")
    
    # Check for single-column dataframe (possible delimiter issue)
    if df.shape[1] == 1 and '\t' in str(df.columns[0]):
        result["is_valid"] = False
        result["issues"].append("DataFrame has only one column, but appears to be tab-delimited")
    
    # Check for reasonable column count
    if df.shape[1] < 5:
        result["warnings"].append(f"DataFrame has only {df.shape[1]} columns, which is fewer than expected")
    
    # Check for Chinese characters that might cause encoding issues
    for col in df.columns[:10]:  # Check just first 10 columns
        if isinstance(col, str) and any(u'\u4e00' <= c <= u'\u9fff' for c in col):
            result["warnings"].append(f"Column name '{col}' contains Chinese characters, which may cause encoding issues")
    
    # Check for high missing value percentage
    missing_pct = df.isna().mean().mean() * 100
    if missing_pct > 50:
        result["warnings"].append(f"High percentage of missing values: {missing_pct:.1f}%")
    
    return result