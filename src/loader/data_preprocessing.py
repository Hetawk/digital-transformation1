"""
Data preprocessing module for MSCI inclusion and digital transformation analysis
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# Fix import paths
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.loader.config import *
from src.loader.data_loader import load_dataset, validate_dataframe

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
    def __init__(self, data_file=None, file_format='csv'):
        """Initialize the DataPreprocessor
        
        Parameters:
        -----------
        data_file : str or Path, optional
            Path to data file
        file_format : str, default 'csv'
            Format of the data file ('dta' for Stata, 'csv' for CSV)
        """
        self.data_file = data_file or DATA_FILE
        self.file_format = file_format
        self.data = None
        
        # If data_file is specified and exists but has no extension, add it
        if self.data_file and not Path(self.data_file).suffix:
            self.data_file = f"{self.data_file}.{self.file_format}"

    def load_data(self, delimiter=None, encoding='utf-8'):
        """Load data from file"""
        logger.info(f"Loading data from {self.data_file}")
        
        try:
            # FIX: Pass file_path as the only positional argument
            # Other arguments should be keyword arguments
            self.data, file_format = load_dataset(
                file_path=self.data_file,
                delimiter=delimiter, 
                encoding=encoding
            )
            
            # Handle empty dataframe
            if self.data is None or len(self.data) == 0:
                logger.error("Loaded data is empty")
                raise ValueError("Empty dataset loaded")
                
            # Validate the loaded data
            validation = validate_dataframe(self.data)
            if not validation["is_valid"]:
                logger.warning(f"Loaded data may have issues: {validation['issues']}")
                if validation["warnings"]:
                    for warning in validation["warnings"]:
                        logger.warning(f"Warning: {warning}")
            else:
                # Fix: Check if 'expected_columns_found' exists in validation
                expected_cols_found = validation.get('expected_columns_found', [])
                logger.info(f"Data validation passed. Found {len(expected_cols_found)} expected columns.")
                
            logger.info(f"Successfully loaded data: {len(self.data)} rows, {self.data.shape[1]} columns")
            return self.data
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            logger.info("Continuing with empty dataframe, but analysis may fail.")
            self.data = pd.DataFrame()
            return self.data
    
    def check_variables(self):
        """Check if all required variables are present in the dataset
        
        Returns:
        --------
        bool: True if all required variables are present, False otherwise
        """
        # Make this function more robust - don't exit on failure
        if self.data is None or len(self.data) == 0:
            logger.error("Data not loaded or empty. Check your data source.")
            return False
        
        # Define required variables
        required_vars = ['year', 'stkcd', 'Treat', 'Post', 'MSCI']
        
        # Add digital transformation measures and control variables
        required_vars.extend(DT_MEASURES)
        required_vars.extend(CONTROL_VARS)
        
        # Check if required variables exist
        missing_vars = [var for var in required_vars if var not in self.data.columns]
        
        if missing_vars:
            logger.warning(f"Missing variables: {', '.join(missing_vars)}")
            # Only consider critical vars as deal-breakers
            critical_vars = ['year', 'stkcd', 'Treat', 'Post']
            critical_missing = [var for var in critical_vars if var in missing_vars]
            if critical_missing:
                logger.error(f"Missing critical variables: {', '.join(critical_missing)}")
                return False
        
        # Continue with existing code for checking data types
        return True

    def generate_variables(self):
        """Generate variables for analysis"""
        if self.data is None or len(self.data) == 0:
            logger.error("No data to process")
            return self.data
            
        original_row_count = len(self.data)
        
        # Make sure year is numeric
        try:
            self.data['year'] = pd.to_numeric(self.data['year'], errors='coerce')
            self.data = self.data[~self.data['year'].isna()]
            logger.info(f"Removed rows with invalid year values. New shape: {self.data.shape}")
        except Exception as e:
            logger.warning(f"Error converting year column: {e}")
        
        # Generate treatment variables safely
        try:
            # Generate TreatPost interaction term
            if 'Treat' in self.data.columns and 'Post' in self.data.columns:
                self.data['Treat'] = pd.to_numeric(self.data['Treat'], errors='coerce').fillna(0)
                self.data['Post'] = pd.to_numeric(self.data['Post'], errors='coerce').fillna(0)
                self.data['TreatPost'] = self.data['Treat'] * self.data['Post']
                logger.info("Generated TreatPost interaction term")
            
            # Create MSCI_clean variable
            if 'MSCI' in self.data.columns:
                self.data['MSCI'] = pd.to_numeric(self.data['MSCI'], errors='coerce').fillna(0)
                self.data['MSCI_clean'] = np.where(self.data['MSCI'] == 1, 1, 0)
                logger.info("Generated MSCI_clean variable")
            
            # Generate Event_time variable
            if 'year' in self.data.columns:
                treatment_year = int(TREATMENT_YEAR)
                self.data['Event_time'] = self.data['year'] - treatment_year
                logger.info(f"Generated Event_time variable relative to treatment year {treatment_year}")
        except Exception as e:
            logger.warning(f"Error generating treatment variables: {e}")
        
        # Continue with remaining variable generation safely
        # Ensure TREATMENT_YEAR is numeric
        treatment_year = int(TREATMENT_YEAR)

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
        for col in DT_MEASURES + CONTROL_VARS:
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
        if len(critical_vars_with_high_na) > len(DT_MEASURES) / 2:
            critical_vars_str = ", ".join([f"{var} ({pct:.1f}%)" for var, pct in critical_vars_with_high_na])
            logger.error(f"Too many critical variables have high NA percentages after conversion: {critical_vars_str}")
            sys.exit(1)
        
        # Report on data quality after preprocessing
        rows_remaining = len(self.data)
        rows_dropped = original_row_count - rows_remaining
        logger.info(f"Data preprocessing: {rows_dropped} rows dropped, {rows_remaining} rows remaining")
        
        # Count non-NA values in key columns
        non_na_info = []
        for col in DT_MEASURES + CONTROL_VARS + ['Treat', 'Post', 'MSCI', 'MSCI_clean']:
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
        """Create panel ID for panel data analysis"""
        if self.data is None or len(self.data) == 0:
            logger.warning("No data available for creating panel ID")
            return self.data
            
        try:
            # Ensure year is numeric
            self.data['year'] = pd.to_numeric(self.data['year'], errors='coerce')
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
            return self.data

    def create_first_differences(self):
        """Create first differences of key variables for first-differences estimation

        Returns:
        --------
        pd.DataFrame: Updated DataFrame with differenced variables
        """
        if self.data is None or len(self.data) == 0:
            logger.warning("No data available for creating first differences")
            return self.data

        try:
            # Sort by firm and year
            self.data = self.data.sort_values(['stkcd', 'year'])

            # Ensure all relevant variables are numeric before differencing
            for var in DT_MEASURES:
                if var in self.data.columns:
                    # Convert to numeric, coercing errors to NaN
                    self.data[var] = pd.to_numeric(self.data[var], errors='coerce')

                    # Calculate first differences
                    self.data[f'D_{var}'] = self.data.groupby('stkcd')[var].diff()

            logger.info("First differences created for key variables")
            return self.data
        
        except Exception as e:
            logger.error(f"Error creating first differences: {e}")
            return self.data

    def run_all(self):
        """Run all preprocessing steps in sequence

        Returns:
        --------
        pd.DataFrame: Fully preprocessed DataFrame
        """
        try:
            self.load_data()
            
            # Check if required variables are present
            if not self.check_variables():
                logger.warning("Some critical variables missing from dataset.")
                # Continue anyway but warn user
            
            self.generate_variables()
            self.create_panel_id()
            self.create_first_differences()
            
            # Additional data validation with warnings instead of errors
            if 'Treat' in self.data.columns:
                treat_values = self.data['Treat'].unique()
                if len(treat_values) < 2:
                    logger.warning(f"Only found treatment values: {treat_values}. Expected both 0 and 1.")
            
            if 'Post' in self.data.columns:
                post_values = self.data['Post'].unique()
                if len(post_values) < 2:
                    logger.warning(f"Only found post values: {post_values}. Expected both 0 and 1.")

            logger.info("All preprocessing steps completed")

            # Save a backup of preprocessed data
            try:
                backup_file = DATA_DIR / "preprocessed_backup.csv"
                self.data.to_csv(backup_file, index=False)
                logger.info(f"Saved preprocessed data backup to {backup_file}")
            except Exception as e:
                logger.warning(f"Could not save backup: {e}")

            return self.data

        except Exception as e:
            logger.error(f"Error during preprocessing: {e}")
            # Return whatever data we have so far instead of exiting
            return self.data if self.data is not None else pd.DataFrame()
