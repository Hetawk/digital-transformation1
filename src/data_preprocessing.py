"""
Data preprocessing module for MSCI inclusion and digital transformation analysis
"""

import config
import pandas as pd
import numpy as np
import os
import sys
import logging
import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

# Set up logger
logger = logging.getLogger('digital_transformation')
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', 
                                 datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class DataPreprocessor:
    def __init__(self, data_file=None):
        """Initialize the DataPreprocessor"""
        self.data_file = data_file or config.DATA_FILE
        self.data = None

    def _load_data(self, data_file=None):
        """
        Load data from file
        
        Parameters:
        -----------
        data_file : str or Path, optional
            Path to data file
            
        Returns:
        --------
        pd.DataFrame: Loaded data
        """
        if data_file is None:
            logger.error("No data file provided. Aborting.")
            sys.exit(1)
        
        data_path = Path(data_file)
        if not data_path.exists():
            logger.error(f"Data file not found: {data_path}")
            sys.exit(1)
        
        logger.info(f"Found data file: {data_path}")
        
        # Auto-detect file type based on extension
        file_ext = data_path.suffix.lower()
        
        # Try Stata file if it has .dta extension
        if file_ext == '.dta':
            try:
                logger.info("Detected Stata file format, attempting to load...")
                # Try using pyreadstat if available (better for complex Stata files)
                try:
                    import pyreadstat
                    data, meta = pyreadstat.read_dta(str(data_path))
                    logger.info(f"Successfully loaded Stata file using pyreadstat: {len(data)} rows")
                    return data
                except ImportError:
                    # Fall back to pandas read_stata
                    data = pd.read_stata(str(data_path))
                    logger.info(f"Successfully loaded Stata file using pandas: {len(data)} rows")
                    return data
            except Exception as e:
                logger.error(f"Error loading Stata file: {e}")
                logger.info("Attempting CSV methods as fallback...")
        
        # For CSV or as fallback for failed Stata load
        # Method 1: Try with auto-detected encoding using chardet
        try:
            # Sample file to detect encoding
            import chardet
            with open(data_path, 'rb') as f:
                sample_data = f.read(10000)  # Read first 10KB
                encoding_result = chardet.detect(sample_data)
            
            detected_encoding = encoding_result['encoding']
            confidence = encoding_result['confidence']
            logger.info(f"Detected encoding: {detected_encoding} with {confidence*100:.1f}% confidence")
            
            if confidence > 0.7:  # Only use if confidence is reasonable
                logger.info(f"Attempting to load data with detected encoding: {detected_encoding}")
                # Try with different delimiters
                for sep in [',', '\t', ';']:
                    try:
                        data = pd.read_csv(data_path, encoding=detected_encoding, sep=sep, 
                                          engine='python', on_bad_lines='skip')
                        if len(data.columns) > 1:  # Successfully parsed into multiple columns
                            logger.info(f"Successfully loaded data using encoding={detected_encoding}, sep='{sep}'")
                            return data
                    except Exception as inner_e:
                        continue  # Try next delimiter
        except ImportError:
            logger.warning("chardet not installed, skipping encoding detection")
        except Exception as e:
            logger.warning(f"Error during encoding detection: {e}")
        
        # Method 2: Try chunking with python engine
        try:
            logger.info("Attempting to load data with chunking and python engine...")
            chunks = []
            for chunk in pd.read_csv(data_path, encoding='utf-8', engine='python', 
                                   on_bad_lines='skip', chunksize=10000):
                chunks.append(chunk)
            
            if chunks:
                data = pd.concat(chunks)
                logger.info(f"Successfully loaded data using chunking method: {len(data)} rows")
                return data
        except Exception as e1:
            logger.warning(f"Method 2 failed with error: {e1}")
        
        # Method 3: Try with flexible delimiter
        try:
            logger.info("Attempting to load data with flexible delimiter...")
            data = pd.read_csv(data_path, encoding='latin1', sep=None, 
                              engine='python', on_bad_lines='skip')
            logger.info(f"Successfully loaded data using flexible delimiter: {len(data)} rows")
            return data
        except Exception as e2:
            logger.warning(f"Method 3 failed with error: {e2}")
        
        # Method 4: Manual line-by-line reading as last resort
        try:
            logger.info("Attempting to read file line by line...")
            with open(data_path, 'rb') as f:
                # Read first line to get header
                header_line = f.readline()
                # Try different encodings for header
                header = None
                for enc in ['utf-8', 'latin1', 'gbk', 'gb18030']:
                    try:
                        header = header_line.decode(enc).strip().split(',')
                        if len(header) > 1:  # Valid header with multiple columns
                            break
                    except:
                        continue
                
                if not header or len(header) <= 1:
                    logger.error("Failed to decode header with any supported encoding")
                    sys.exit(1)
                
                # Read remaining lines, skipping problematic ones
                rows = []
                for i, line in enumerate(f):
                    try:
                        # Try different encodings for each line
                        for enc in ['utf-8', 'latin1', 'gbk', 'gb18030']:
                            try:
                                row = line.decode(enc).strip().split(',')
                                if len(row) == len(header):  # Only include rows with correct number of columns
                                    rows.append(row)
                                    break
                            except:
                                continue
                    except Exception as line_error:
                        if i < 10:  # Only print first few errors
                            logger.warning(f"  Skipping line {i+2}: {str(line_error)[:100]}...")
                
                # Create DataFrame
                if rows:
                    data = pd.DataFrame(rows, columns=header)
                    logger.info(f"Successfully loaded data line by line: {len(data)} rows")
                    return data
        except Exception as e3:
            logger.error(f"Method 4 failed with error: {e3}")
        
        # If all methods failed, exit with error
        logger.error("All data loading methods failed. Unable to read data file.")
        sys.exit(1)

    def load_data(self):
        """
        Load the dataset from file

        Returns:
        --------
        pd.DataFrame: Loaded DataFrame
        """
        self.data = self._load_data(self.data_file)
        return self.data

    def check_variables(self):
        """
        Check if all required variables are present in the dataset
        
        Returns:
        --------
        bool: True if all required variables are present, False otherwise
        """
        if self.data is None:
            logger.error("Data not loaded. Call load_data() first.")
            return False
        
        # Define required variables
        required_vars = ['year', 'stkcd', 'Treat', 'Post', 'MSCI']
        
        # Add digital transformation measures and control variables
        required_vars.extend(config.DT_MEASURES)
        required_vars.extend(config.CONTROL_VARS)
        
        # Check if required variables exist
        missing_vars = [var for var in required_vars if var not in self.data.columns]
        
        if missing_vars:
            logger.error(f"Missing required variables: {', '.join(missing_vars)}")
            return False
        
        # Check if we have at least one non-empty row
        if len(self.data) == 0:
            logger.error("Dataset is empty")
            return False
        
        # Check data types and convertibility for critical variables
        critical_vars = ['year', 'Treat', 'Post', 'MSCI']
        
        for var in critical_vars:
            if var in self.data.columns:
                # Check if column can be converted to numeric
                test_numeric = pd.to_numeric(self.data[var], errors='coerce')
                
                # If more than 50% values became NaN after conversion, warn
                nan_pct = test_numeric.isna().mean() * 100
                if nan_pct > 50:
                    logger.warning(f"Variable {var} has {nan_pct:.1f}% values that cannot be converted to numeric")
        
        logger.info("All required variables are present in the dataset")
        return True

    def generate_variables(self):
        """
        Generate variables needed for analysis

        Returns:
        --------
        pd.DataFrame: Updated DataFrame with new variables
        """
        if self.data is None:
            self.load_data()

        # First ensure year is numeric
        original_row_count = len(self.data)
        
        # Ensure key variables are numeric
        for col in ['year', 'Treat', 'Post', 'MSCI']:
            if col in self.data.columns:
                # Save original column if it might be needed later
                self.data[f'{col}_original'] = self.data[col]
                # Convert to numeric, coercing errors to NaN
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        
        # Handle missing or non-numeric years
        missing_years = self.data['year'].isna().sum()
        if missing_years > 0:
            logger.warning(f"Warning: {missing_years} rows with invalid year values")
            # Filter out rows with invalid years to prevent further errors
            self.data = self.data[~self.data['year'].isna()]
            logger.info(f"Removed rows with invalid year values. New shape: {self.data.shape}")
        
        # Generate TreatPost interaction term
        self.data['TreatPost'] = self.data['Treat'] * self.data['Post']

        # Create MSCI_clean variable
        self.data['MSCI_clean'] = np.where(self.data['MSCI'] == 1, 1, 0)

        # Ensure TREATMENT_YEAR is numeric
        treatment_year = int(config.TREATMENT_YEAR)

        # Generate Event_time variable
        self.data['Event_time'] = self.data['year'] - treatment_year

        # Create Large_firm indicator if A001000000 (Total Assets) exists
        if 'A001000000' in self.data.columns:
            try:
                # Convert to numeric first
                self.data['A001000000'] = pd.to_numeric(self.data['A001000000'], errors='coerce')
                # Group by year and compute median of Total Assets
                yearly_median = self.data.groupby('year')['A001000000'].transform('median')
                self.data['Large_firm'] = (self.data['A001000000'] >= yearly_median).astype(int)
            except Exception as e:
                logger.warning(f"Could not create Large_firm variable: {e}")
        
        # Attempt to convert other key numeric variables
        critical_vars_with_high_na = []
        for col in config.DT_MEASURES + config.CONTROL_VARS:
            if col in self.data.columns:
                try:
                    # Store original values
                    original_na_count = self.data[col].isna().sum()
                    
                    # Convert to numeric
                    self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
                    
                    # Check if conversion introduced many new NAs
                    new_na_count = self.data[col].isna().sum()
                    new_na_pct = (new_na_count - original_na_count) / len(self.data) * 100
                    
                    if new_na_pct > 30:  # If more than 30% new NAs introduced
                        critical_vars_with_high_na.append((col, new_na_pct))
                        logger.warning(f"Converting {col} to numeric introduced {new_na_pct:.1f}% new NA values")
                except Exception as e:
                    logger.warning(f"Could not convert {col} to numeric: {e}")
        
        # If too many critical variables have high NA after conversion, fail
        if len(critical_vars_with_high_na) > len(config.DT_MEASURES) / 2:
            critical_vars_str = ", ".join([f"{var} ({pct:.1f}%)" for var, pct in critical_vars_with_high_na])
            logger.error(f"Too many critical variables have high NA percentages after conversion: {critical_vars_str}")
            sys.exit(1)
        
        # Report on data quality after preprocessing
        rows_remaining = len(self.data)
        rows_dropped = original_row_count - rows_remaining
        logger.info(f"Data preprocessing: {rows_dropped} rows dropped, {rows_remaining} rows remaining")
        
        # Count non-NA values in key columns
        non_na_info = []
        for col in config.DT_MEASURES + config.CONTROL_VARS + ['Treat', 'Post', 'MSCI', 'MSCI_clean']:
            if col in self.data.columns:
                non_na_count = self.data[col].notna().sum()
                non_na_pct = 100 * non_na_count / len(self.data)
                non_na_info.append(f"{col}: {non_na_count} non-NA values ({non_na_pct:.1f}%)")
        
        logger.info("Non-NA count for key variables:")
        for info in non_na_info:
            logger.info(f"  {info}")

        logger.info("Successfully generated variables for analysis")
        return self.data

    def create_panel_id(self):
        """
        Set up panel structure for panel data analysis

        Returns:
        --------
        pd.DataFrame: DataFrame with panel index
        """
        if self.data is None:
            logger.error("Data not loaded. Call load_data() first.")
            sys.exit(1)

        try:
            # Ensure stkcd and year are properly formatted
            self.data['stkcd'] = self.data['stkcd'].astype(str)
            self.data['year'] = pd.to_numeric(self.data['year'], errors='coerce')

            # Check for missing years after conversion
            missing_years = self.data['year'].isna().sum()
            if missing_years > 0:
                logger.warning(f"{missing_years} rows have missing years after conversion")
                if missing_years > len(self.data) * 0.1:  # If more than 10% missing
                    logger.error("Too many missing years. Cannot create valid panel structure.")
                    sys.exit(1)
                # Filter out rows with missing years
                self.data = self.data[~self.data['year'].isna()]

            # Sort by firm ID and year
            self.data = self.data.sort_values(['stkcd', 'year'])

            # Create panel ID
            self.data = self.data.set_index(['stkcd', 'year'])

            # Reset index for easier handling in some functions
            self.data = self.data.reset_index()

            logger.info("Panel ID created for panel data analysis")
            return self.data
        
        except Exception as e:
            logger.error(f"Error creating panel ID: {e}")
            sys.exit(1)

    def create_first_differences(self):
        """
        Create first differences of key variables for first-differences estimation

        Returns:
        --------
        pd.DataFrame: Updated DataFrame with differenced variables
        """
        if self.data is None:
            logger.error("Data not loaded. Call load_data() first.")
            sys.exit(1)

        try:
            # Sort by firm and year
            self.data = self.data.sort_values(['stkcd', 'year'])

            # Ensure all relevant variables are numeric before differencing
            for var in config.DT_MEASURES:
                if var in self.data.columns:
                    # Convert to numeric, coercing errors to NaN
                    self.data[var] = pd.to_numeric(self.data[var], errors='coerce')

                    # Calculate first differences
                    self.data[f'D_{var}'] = self.data.groupby('stkcd')[var].diff()

            logger.info("First differences created for key variables")
            return self.data
        
        except Exception as e:
            logger.error(f"Error creating first differences: {e}")
            sys.exit(1)

    def run_all(self):
        """
        Run all preprocessing steps in sequence

        Returns:
        --------
        pd.DataFrame: Fully preprocessed DataFrame
        """
        try:
            self.load_data()
            
            # Check if required variables are present
            if not self.check_variables():
                logger.error("Critical variables missing from dataset. Aborting.")
                sys.exit(1)
            
            self.generate_variables()
            self.create_panel_id()
            self.create_first_differences()
            
            # Additional data validation
            # Check if we have both treatment and control groups
            if 'Treat' in self.data.columns:
                treat_values = self.data['Treat'].unique()
                if len(treat_values) < 2:
                    logger.warning(f"Only found treatment values: {treat_values}. Expected both 0 and 1.")
                    # Check if the lack of both groups is critical
                    if 0 not in treat_values:
                        logger.error("No control group (Treat=0) found in dataset. Cannot perform treatment-control comparisons.")
                        sys.exit(1)
                    if 1 not in treat_values:
                        logger.error("No treatment group (Treat=1) found in dataset. Cannot perform treatment effect analysis.")
                        sys.exit(1)
            
            # Check if we have both pre and post periods
            if 'Post' in self.data.columns:
                post_values = self.data['Post'].unique()
                if len(post_values) < 2:
                    logger.warning(f"Only found post values: {post_values}. Expected both 0 and 1.")
                    # Check if the lack of both periods is critical
                    if 0 not in post_values:
                        logger.error("No pre-treatment period (Post=0) found in dataset. Cannot perform before-after comparisons.")
                        sys.exit(1)
                    if 1 not in post_values:
                        logger.error("No post-treatment period (Post=1) found in dataset. Cannot perform treatment effect analysis.")
                        sys.exit(1)

            logger.info("All preprocessing steps completed successfully")

            # Save a backup of preprocessed data
            backup_file = config.DATA_DIR / "preprocessed_backup.csv"
            self.data.to_csv(backup_file, index=False)
            logger.info(f"Saved preprocessed data backup to {backup_file}")

            return self.data

        except Exception as e:
            logger.error(f"Error during preprocessing: {e}")
            sys.exit(1)
