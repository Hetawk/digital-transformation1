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

# Add the src directory to the path to find modules
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "src"))

# Import project modules
from src.loader.data_diagnostics import run_comprehensive_diagnostics
from src.loader.config import logger, configure_logging
# Import the centralized data loader if it exists
try:
    from src.loader.data_loader import load_dataset, validate_dataframe
    USE_CENTRALIZED_LOADER = True
except ImportError:
    USE_CENTRALIZED_LOADER = False
    # Fallback to existing load_dataset function if centralized one not available

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
    Load dataset with robust error handling for different file formats
    
    Parameters:
    -----------
    file_path : str or Path
        Path to the data file
    nrows : int or None
        Number of rows to read (can improve performance)
    delimiter : str or None
        CSV delimiter, if specified (e.g., ',', '\t', ';')
    encoding : str or None
        File encoding (e.g., 'latin1', 'utf-8')
        
    Returns:
    --------
    tuple: (pd.DataFrame, str) - Loaded data and file format
    """
    file_path = Path(file_path)
    file_format = file_path.suffix.lower()[1:]  # Get file extension without dot
    
    # Map delimiter string to actual character
    if delimiter == 'tab':
        delimiter = '\t'
    
    # For Stata files
    if file_format == 'dta':
        logger.info(f"Loading Stata file: {file_path}")
        
        # Try pyreadstat first (better handling of encodings)
        try:
            import pyreadstat
            df, meta = pyreadstat.read_dta(str(file_path))
            logger.info(f"Successfully loaded Stata file with pyreadstat, shape: {df.shape}")
            return df, 'dta'
        except Exception as e:
            logger.warning(f"Error loading Stata file with pyreadstat: {e}")
            logger.warning("Falling back to pandas read_stata.")
            
        # Try pandas read_stata
        try:
            df = pd.read_stata(file_path, convert_categoricals=False)
            logger.info(f"Successfully loaded Stata file with pandas, shape: {df.shape}")
            return df, 'dta'
        except Exception as e:
            logger.error(f"Error loading Stata file with pandas: {e}")
            logger.info("Attempting to load Stata file using iterator...")
            
        # Try with iterator (for very large files)
        try:
            chunks = []
            itr = pd.read_stata(file_path, iterator=True, chunksize=10000)
            for chunk in itr:
                chunks.append(chunk)
            df = pd.concat(chunks)
            logger.info(f"Successfully loaded Stata file with iterator, shape: {df.shape}")
            return df, 'dta'
        except Exception as e:
            logger.error(f"All methods for loading Stata file failed. Last error: {e}")
            logger.warning("Failed to load Stata file. Attempting to load as CSV...")
    
    # CSV Files (or fallback for Stata files)
    logger.info(f"Loading CSV file: {file_path}")
    
    # For extremely large files, try chunked reading approach first
    logger.info("Attempting to load file in chunks for large dataset handling")
    try:
        # Try loading first chunk to detect structure
        chunk_size = 50000  # Smaller chunks to avoid buffer issues
        chunks = []
        
        # Use C engine with latin1 encoding and tab delimiter first (based on previous success)
        chunk_iter = pd.read_csv(
            file_path,
            sep='\t',
            encoding='latin1',
            engine='c',
            chunksize=chunk_size,
            low_memory=True,     # Changed to True for better memory management
            on_bad_lines='warn'  # Updated from error_bad_lines to on_bad_lines
        )
        
        # Read limited number of chunks for analysis (avoid loading entire file)
        max_chunks = 20 if nrows is None else max(1, nrows // chunk_size + 1)
        for i, chunk in enumerate(chunk_iter):
            if i >= max_chunks:
                break
            chunks.append(chunk)
            logger.info(f"Loaded chunk {i+1} with {len(chunk)} rows")
        
        # Combine chunks
        if chunks:
            df = pd.concat(chunks, ignore_index=True)
            logger.info(f"Successfully loaded data in chunks: {len(df)} rows × {df.shape[1]} columns")
            return df, 'csv'
        
    except Exception as e:
        logger.warning(f"Chunked loading approach failed: {e}")
        logger.info("Falling back to standard loading methods")
        
        # Try another chunking approach with Python engine
        try:
            logger.info("Trying alternate chunking method with Python engine...")
            chunk_iter = pd.read_csv(
                file_path,
                sep='\t',
                encoding='latin1',
                engine='python',
                chunksize=chunk_size,
                on_bad_lines='skip'
            )
            
            chunks = []
            max_chunks = 20 if nrows is None else max(1, nrows // chunk_size + 1)
            for i, chunk in enumerate(chunk_iter):
                if i >= max_chunks:
                    break
                chunks.append(chunk)
                logger.info(f"Loaded chunk {i+1} with {len(chunk)} rows")
            
            if chunks:
                df = pd.concat(chunks, ignore_index=True)
                logger.info(f"Successfully loaded data with alternate chunking method: {len(df)} rows × {df.shape[1]} columns")
                return df, 'csv'
        except Exception as e2:
            logger.warning(f"Alternate chunking method failed: {e2}")
            logger.info("Continuing to try standard loading methods")
    
    # Check encoding first - use chardet if available
    detected_encoding = None
    detected_confidence = 0
    try:
        import chardet
        with open(file_path, 'rb') as f:
            # Read a chunk of file to detect encoding (first 100k bytes)
            raw_data = f.read(100000)
            result = chardet.detect(raw_data)
            detected_encoding = result['encoding']
            detected_confidence = result['confidence']
            logger.info(f"Detected encoding: {detected_encoding} with {detected_confidence*100:.1f}% confidence")
    except (ImportError, Exception) as e:
        logger.info(f"Couldn't detect encoding automatically: {str(e)}")
    
    # If specified encoding or low confidence detection, try multiple encodings
    if encoding or detected_confidence < 0.9:
        logger.warning("Low confidence encoding detection, will try common encodings")
        
        # Define the loading attempts - optimized based on the successful pattern in logs
        # Prioritize tab delimiter with latin1 encoding which worked in the logs
        attempts = []
        
        # First try user-specified options
        if delimiter and encoding:
            attempts.append({
                'sep': delimiter, 
                'engine': 'python', 
                'on_bad_lines': 'skip',
                'encoding': encoding
                # Removed low_memory parameter when using python engine
            })
        
        # Add most common options, prioritizing latin1 with tab delimiter
        attempts.extend([
            # TAB DELIMITER WITH LATIN1 (this worked in the logs)
            {'sep': '\t', 'engine': 'python', 'on_bad_lines': 'skip', 'encoding': 'latin1'},
            
            # COMMA DELIMITER OPTIONS
            {'sep': ',', 'engine': 'python', 'on_bad_lines': 'skip', 'encoding': 'utf-8'},
            {'sep': ',', 'engine': 'python', 'on_bad_lines': 'skip', 'encoding': 'latin1'},
            {'sep': ',', 'engine': 'python', 'on_bad_lines': 'skip', 'encoding': 'gbk'},
            {'sep': ',', 'engine': 'python', 'on_bad_lines': 'skip', 'encoding': 'gb18030'},
            
            # OTHER TAB DELIMITER OPTIONS
            {'sep': '\t', 'engine': 'python', 'on_bad_lines': 'skip', 'encoding': 'utf-8'},
            {'sep': '\t', 'engine': 'python', 'on_bad_lines': 'skip', 'encoding': 'gbk'},
            {'sep': '\t', 'engine': 'python', 'on_bad_lines': 'skip', 'encoding': 'gb18030'},
            
            # SEMICOLON DELIMITER OPTIONS
            {'sep': ';', 'engine': 'python', 'on_bad_lines': 'skip', 'encoding': 'utf-8'},
            {'sep': ';', 'engine': 'python', 'on_bad_lines': 'skip', 'encoding': 'latin1'},
            {'sep': ';', 'engine': 'python', 'on_bad_lines': 'skip', 'encoding': 'gbk'},
            {'sep': ';', 'engine': 'python', 'on_bad_lines': 'skip', 'encoding': 'gb18030'},
            
            # AUTO-DETECT DELIMITER OPTIONS
            {'sep': None, 'engine': 'python', 'on_bad_lines': 'skip', 'encoding': 'utf-8'},
            {'sep': None, 'engine': 'python', 'on_bad_lines': 'skip', 'encoding': 'latin1'},
            {'sep': None, 'engine': 'python', 'on_bad_lines': 'skip', 'encoding': 'gbk'},
            {'sep': None, 'engine': 'python', 'on_bad_lines': 'skip', 'encoding': 'gb18030'},
            
            # Try default C engine with low_memory option
            {'sep': '\t', 'engine': 'c', 'encoding': 'latin1', 'low_memory': False},
            {'sep': ',', 'engine': 'c', 'encoding': 'latin1', 'low_memory': False},
            {'sep': ';', 'engine': 'c', 'encoding': 'latin1', 'low_memory': False},
        ])
        
        # Try each attempt
        expected_cols = ['Digital_transformationA', 'Digital_transformationB', 'Treat', 'Post', 'MSCI']
        best_df = None
        best_match_count = 0
        
        for i, params in enumerate(attempts):
            logger.info(f"CSV attempt {i+1}: Reading with {params}")
            try:
                # Only add nrows parameter, not low_memory with python engine
                if 'engine' in params and params['engine'] == 'python' and 'low_memory' in params:
                    del params['low_memory']  # Remove low_memory if python engine is used
                    
                df = pd.read_csv(file_path, **params, nrows=nrows)
                
                # Check if dataframe was read properly
                if df.shape[1] == 1:
                    # Check if this looks like a delimiter issue
                    first_col_name = df.columns[0]
                    if any(sep in first_col_name for sep in ['\t', ',', ';']):
                        logger.warning("File loaded as single column with separators. Current strategy not optimal.")
                        # Continue to try next strategy
                    else:
                        logger.info(f"Success! Read {len(df)} rows with {df.shape[1]} columns")
                        # Keep track of this as potential best result
                        found_cols = [col for col in expected_cols if col in df.columns]
                        if len(found_cols) > best_match_count:
                            best_df = df
                            best_match_count = len(found_cols)
                            # If we found all expected columns, no need to keep trying
                            if len(found_cols) == len(expected_cols):
                                logger.info(f"Found {len(found_cols)}/{len(expected_cols)} expected columns: {found_cols}")
                                return df, 'csv'
                        else:
                            logger.warning("Loaded data but found none of the expected columns. Continuing to search for better loading strategy.")
                else:
                    logger.info(f"Success! Read {len(df)} rows with {df.shape[1]} columns")
                    # Check if we have our key columns
                    found_cols = [col for col in expected_cols if col in df.columns]
                    if len(found_cols) > best_match_count:
                        best_df = df
                        best_match_count = len(found_cols)
                        # If we found all expected columns, no need to keep trying
                        if len(found_cols) == len(expected_cols):
                            logger.info(f"Found {len(found_cols)}/{len(expected_cols)} expected columns: {found_cols}")
                            return df, 'csv'
                    elif len(found_cols) > 0:
                        logger.info(f"Found {len(found_cols)}/{len(expected_cols)} expected columns: {found_cols}")
                        return df, 'csv'
            except Exception as e:
                logger.warning(f"CSV attempt {i+1} failed: {str(e)}")
        
        # If we tried all attempts and have a best_df with at least some matches, return it
        if best_df is not None:
            found_cols = [col for col in expected_cols if col in best_df.columns]
            logger.info(f"Found {len(found_cols)}/{len(expected_cols)} expected columns: {found_cols}")
            return best_df, 'csv'
        
        logger.error(f"Failed to load {file_path} after all attempts.")
        return None, None
    
    # If we have high confidence in detected encoding, use it directly
    else:
        try:
            # Use C engine when using low_memory parameter
            df = pd.read_csv(
                file_path, 
                sep=delimiter or ',',
                encoding=detected_encoding,
                engine='c',  # Changed to C engine to support low_memory
                on_bad_lines='skip',
                nrows=nrows,
                low_memory=False
            )
            logger.info(f"Loaded data with detected encoding {detected_encoding}. Shape: {df.shape}")
            return df, 'csv'
        except Exception as e:
            logger.error(f"Error loading with detected encoding {detected_encoding}: {e}")
            logger.warning("Falling back to Latin-1 encoding with tab delimiter...")
            
            # Fallback to Latin-1 with tab delimiter as it worked in the logs
            try:
                df = pd.read_csv(
                    file_path, 
                    sep='\t',
                    encoding='latin1',
                    engine='c',  # Changed to C engine to support low_memory
                    on_bad_lines='skip',
                    nrows=nrows,
                    low_memory=False
                )
                logger.info(f"Successfully loaded with Latin-1 encoding and tab delimiter. Shape: {df.shape}")
                return df, 'csv'
            except Exception as e2:
                # Try once more without low_memory option
                try:
                    df = pd.read_csv(
                        file_path, 
                        sep='\t',
                        encoding='latin1',
                        engine='python',
                        on_bad_lines='skip',
                        nrows=nrows
                    )
                    logger.info(f"Successfully loaded with fallback method (no low_memory). Shape: {df.shape}")
                    return df, 'csv'
                except Exception as e3:
                    logger.error(f"All fallback loading attempts failed. Last error: {e3}")
                    return None, None

    # Last resort: try reading raw file to diagnose issues
    logger.warning("All standard loading methods failed. Attempting raw file inspection.")
    try:
        # Read first few lines of the file directly to diagnose issues
        with open(file_path, 'rb') as f:
            header = f.readline().decode('latin1', errors='replace')
            logger.info(f"File header: {header[:100]}...")
            
            # Try ultra-conservative approach: read file in small pieces with manual parsing
            logger.info("Trying manual file reading approach...")
            
            # Read just the first 1000 lines to get a sample
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
                # Count columns based on delimiter guesses
                delimiters = ['\t', ',', ';']
                counts = {d: header.count(d) for d in delimiters}
                best_delimiter = max(counts.items(), key=lambda x: x[1])[0]
                logger.info(f"Detected likely delimiter: '{best_delimiter}' (count: {counts[best_delimiter]})")
                
                # Create a StringIO object to read the lines
                import io
                data_str = ''.join(lines)
                data_io = io.StringIO(data_str)
                
                # Try to read with pandas
                try:
                    df = pd.read_csv(
                        data_io,
                        sep=best_delimiter,
                        encoding='latin1',
                        engine='python',
                        on_bad_lines='skip'
                    )
                    logger.info(f"Successfully loaded sample with manual approach: {df.shape}")
                    
                    if len(df) > 0:
                        logger.info("Sample loaded successfully. Now trying to read full file with established parameters...")
                        
                        # Now try to read the full file with these parameters
                        try:
                            full_df = pd.read_csv(
                                file_path,
                                sep=best_delimiter,
                                encoding='latin1',
                                engine='python',
                                on_bad_lines='skip',
                                nrows=nrows,
                                memory_map=True
                            )
                            logger.info(f"Successfully loaded full file: {full_df.shape}")
                            return full_df, 'csv'
                        except Exception as e:
                            logger.error(f"Failed to load full file after successful sample loading: {e}")
                            # Return the sample as better than nothing
                            logger.warning(f"Returning sample data ({len(df)} rows) for analysis")
                            return df, 'csv'
                except Exception as e:
                    logger.error(f"Failed to read sample data: {e}")
    except Exception as e:
        logger.error(f"Raw file inspection failed: {e}")
    
    logger.error(f"All loading methods failed for {file_path}")
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
    if (input_file is None):
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
    
    logger.info(f"Loading data from: {input_file}")
    
    # Load dataset with appropriate parameters
    nrows = 10000 if args.sample else args.nrows
    df, file_format = load_dataset(input_file, nrows=nrows, 
                             delimiter=args.delimiter, 
                             encoding=args.encoding)
    
    if df is None:
        logger.error("Failed to load dataset. Exiting.")
        return
    
    # If we have the centralized loader, validate the data
    if USE_CENTRALIZED_LOADER:
        validation = validate_dataframe(df)
        if not validation["is_valid"]:
            logger.warning("Data validation detected issues:")
            for issue in validation["issues"]:
                logger.warning(f"- {issue}")
            
            if validation["warnings"]:
                logger.info("Data validation warnings:")
                for warning in validation["warnings"]:
                    logger.info(f"- {warning}")
    
    # Run diagnostics with file format information
    run_comprehensive_diagnostics(df, file_format=file_format)
    
    print("\nData diagnostics completed. Results available in:")
    print("  /data2/enoch/ekd_coding_env/patience/digital_transformation/results/tables/diagnostics/")

if __name__ == "__main__":
    main()
