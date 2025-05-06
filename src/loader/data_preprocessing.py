"""
Data preprocessing module for MSCI inclusion and digital transformation analysis
"""

import loader.config as config
import pandas as pd
import numpy as np
import os
import sys
import logging
import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

# Import our centralized data loading functions
from loader.data_loader import load_dataset, validate_dataframe

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
        self.data_file = data_file or config.DATA_FILE
        self.file_format = file_format
        self.data = None
        
        # If data_file is specified and exists but has no extension, add it
        if self.data_file and not Path(self.data_file).suffix:
            self.data_file = f"{self.data_file}.{self.file_format}"

    def load_data(self, delimiter=None, encoding=None):
        """
        Load the dataset from file using the centralized data loader

        Parameters:
        -----------
        delimiter : str or None
            Delimiter for CSV files (e.g., ',', '\t')
        encoding : str or None
            File encoding to use

        Returns:
        --------
        pd.DataFrame: Loaded DataFrame
        """
        if self.data_file is None:
            logger.error("No data file provided. Aborting.")
            sys.exit(1)
        
        data_path = Path(self.data_file)
        if not data_path.exists():
            logger.error(f"Data file not found: {data_path}")
            sys.exit(1)
        
        logger.info(f"Found data file: {data_path}")
        
        # Use the centralized data loader
        self.data, detected_format = load_dataset(
            data_path, 
            delimiter=delimiter, 
            encoding=encoding
        )
        
        if self.data is None:
            logger.error(f"Failed to load data from {data_path}. Aborting.")
            sys.exit(1)
        
        # Validate the loaded data
        validation = validate_dataframe(self.data)
        if not validation["is_valid"]:
            logger.warning(f"Loaded data may have issues: {validation['issues']}")
            if validation["warnings"]:
                for warning in validation["warnings"]:
                    logger.warning(f"Warning: {warning}")
        else:
            logger.info(f"Data validation passed. Found {len(validation['expected_columns_found'])} expected columns.")
            
        logger.info(f"Successfully loaded data: {len(self.data)} rows, {self.data.shape[1]} columns")
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
