"""
Descriptive statistics for MSCI inclusion and digital transformation analysis
"""

from src.utils.util import save_results_to_file, check_balance
import loader.config as config
import pandas as pd
import numpy as np
import os
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))


class DescriptiveAnalysis:
    def __init__(self, data):
        """
        Initialize the DescriptiveAnalysis

        Parameters:
        -----------
        data : pd.DataFrame
            The preprocessed dataset
        """
        self.data = data

    def summarize_variables(self, by_group=None):
        """
        Calculate summary statistics for all relevant variables
        
        Parameters:
        -----------
        by_group : str or None
            Column name to group by for separate statistics, e.g., 'Treat'
        
        Returns:
        --------
        pd.DataFrame or dict: Summary statistics (dict of DataFrames if by_group is provided)
        """
        # Define key variables we want to analyze (modify based on your analysis needs)
        numeric_vars = []
        date_vars = []
        categorical_vars = []
        
        # Identify numeric, date, and categorical columns
        for col in self.data.columns:
            if col.startswith(('Digital_transformation', 'age', 'TFP_OP', 'SA_index', 'WW_index', 'F050501B', 'F060101B', 'MSCI')):
                try:
                    pd.to_numeric(self.data[col])
                    numeric_vars.append(col)
                except:
                    # More robust date detection
                    if self.data[col].dtype == 'object':
                        # Try to determine if this could be a date column
                        # Sample a few values to check date format (avoid checking entire column for performance)
                        sample = self.data[col].dropna().head(100)
                        date_count = pd.to_datetime(sample, errors='coerce').notna().sum()
                        # If more than 50% of sample values are dates, consider it a date column
                        if date_count > len(sample) * 0.5:
                            date_vars.append(col)
                        else:
                            categorical_vars.append(col)
                    else:
                        categorical_vars.append(col)
        
        # If grouping is requested, calculate statistics for each group
        if by_group is not None:
            if by_group not in self.data.columns:
                raise ValueError(f"Group column '{by_group}' not found in the dataset")
            
            groups = self.data[by_group].unique()
            result = {}
            
            for group_val in groups:
                group_data = self.data[self.data[by_group] == group_val]
                group_summary = pd.DataFrame()
                
                # Process numeric variables
                if numeric_vars:
                    group_summary['mean'] = group_data[numeric_vars].mean()
                    group_summary['median'] = group_data[numeric_vars].median()
                    group_summary['std'] = group_data[numeric_vars].std()
                    group_summary['min'] = group_data[numeric_vars].min()
                    group_summary['max'] = group_data[numeric_vars].max()
                
                result[f"{by_group}_{group_val}"] = group_summary
            
            print(f"Generated grouped summary statistics for {len(numeric_vars)} variables")
            return result
        
        # Create summary DataFrame for all data
        summary = pd.DataFrame()
        
        # Process numeric variables
        if numeric_vars:
            summary['mean'] = self.data[numeric_vars].mean()
            summary['median'] = self.data[numeric_vars].median()
            summary['std'] = self.data[numeric_vars].std()
            summary['min'] = self.data[numeric_vars].min()
            summary['max'] = self.data[numeric_vars].max()
        
        # Handle non-numeric variables separately
        non_numeric_summary = None
        
        # Process date variables
        if date_vars:
            date_summary = pd.DataFrame(index=date_vars)
            date_summary['count'] = self.data[date_vars].count()
            
            # Safer min/max calculation for dates
            min_dates = []
            max_dates = []
            
            for var in date_vars:
                try:
                    # Convert to datetime with proper error handling
                    dates = pd.to_datetime(self.data[var], errors='coerce')
                    # Get min/max only from valid dates
                    valid_dates = dates.dropna()
                    if len(valid_dates) > 0:
                        min_dates.append(valid_dates.min())
                        max_dates.append(valid_dates.max())
                    else:
                        min_dates.append(None)
                        max_dates.append(None)
                except Exception as e:
                    print(f"Error processing dates in column {var}: {e}")
                    min_dates.append(None)
                    max_dates.append(None)
            
            date_summary['min_date'] = min_dates
            date_summary['max_date'] = max_dates
            
            if non_numeric_summary is None:
                non_numeric_summary = date_summary
            else:
                non_numeric_summary = pd.concat([non_numeric_summary, date_summary])
        
        # Process categorical variables
        if categorical_vars:
            cat_summary = pd.DataFrame(index=categorical_vars)
            cat_summary['count'] = self.data[categorical_vars].count()
            cat_summary['unique'] = [self.data[var].nunique() for var in categorical_vars]
            
            # Safer most_common and freq calculation
            most_commons = []
            freqs = []
            
            for var in categorical_vars:
                try:
                    value_counts = self.data[var].value_counts()
                    if not value_counts.empty:
                        most_commons.append(value_counts.index[0])
                        freqs.append(value_counts.iloc[0])
                    else:
                        most_commons.append(None)
                        freqs.append(0)
                except Exception as e:
                    print(f"Error processing categorical column {var}: {e}")
                    most_commons.append(None)
                    freqs.append(0)
            
            cat_summary['most_common'] = most_commons
            cat_summary['freq'] = freqs
            
            if non_numeric_summary is None:
                non_numeric_summary = cat_summary
            else:
                non_numeric_summary = pd.concat([non_numeric_summary, cat_summary])
        
        # Combine numeric and non-numeric summaries
        if non_numeric_summary is not None:
            if not summary.empty:
                # Fill NaN values for columns that don't apply to non-numeric data
                for col in summary.columns:
                    non_numeric_summary[col] = np.nan
                
                summary = pd.concat([summary, non_numeric_summary])
            else:
                summary = non_numeric_summary
        
        print(f"Generated summary statistics for {len(summary)} variables")
        return summary

    def correlation_analysis(self):
        """
        Calculate correlation matrix for main variables
        
        Returns:
        --------
        pd.DataFrame: Correlation matrix
        """
        # Focus on key numeric variables for correlation
        variables = []
        
        # Create a copy of the data for correlation analysis
        corr_data = self.data.copy()
        
        # Filter variables to include only numeric ones
        for var in ['Digital_transformationA', 'Digital_transformationB', 
                    'Digital_transformation_rapidA', 'Digital_transformation_rapidB',
                    'MSCI', 'MSCI_clean', 'Treat', 'Post', 'F050501B', 'F060101B']:
            if var in corr_data.columns:
                try:
                    # Convert to numeric, coercing errors to NaN
                    corr_data[var] = pd.to_numeric(corr_data[var], errors='coerce')
                    # Only include if we have sufficient non-NaN values
                    if corr_data[var].notna().sum() > corr_data.shape[0] * 0.1:  # At least 10% non-NaN
                        variables.append(var)
                    else:
                        print(f"Skipping variable with too many missing values: {var}")
                except Exception as e:
                    print(f"Skipping non-numeric variable in correlation: {var} - {e}")
        
        # Try to convert problematic variables explicitly
        for var in ['age', 'TFP_OP', 'SA_index', 'WW_index']:
            if var in corr_data.columns:
                try:
                    # For these variables, be more aggressive in cleaning - remove all non-digit chars
                    # This helps with strings like "year-1-2-3" that might contain embedded numbers
                    # Extract only digits and decimal points, replace rest with space
                    corr_data[var] = corr_data[var].astype(str).str.replace(r'[^\d.\-]', '', regex=True)
                    # Convert to numeric
                    corr_data[var] = pd.to_numeric(corr_data[var], errors='coerce')
                    if corr_data[var].notna().sum() > corr_data.shape[0] * 0.1:
                        variables.append(var)
                    else:
                        print(f"Skipping variable with too many missing values after conversion: {var}")
                except Exception as e:
                    print(f"Skipping non-numeric variable in correlation: {var} - {e}")        
        
        # Calculate correlation matrix for numeric variables
        if not variables:
            print("No valid numeric variables found for correlation analysis")
            return pd.DataFrame()
                
        try:
            corr_matrix = corr_data[variables].corr()
            print(f"Generated correlation matrix for {len(variables)} variables")
            return corr_matrix
        except Exception as e:
            print(f"Error in correlation analysis: {e}")
            # Return an empty DataFrame if correlation fails
            return pd.DataFrame()

    def treatment_balance(self):
        """
        Check balance between treatment and control groups
        
        For staggered adoption designs, units may only become treated after the treatment
        period begins, meaning there are no treated units in the pre-treatment period.
        In this case, we provide descriptive statistics for pre-treatment observations
        and note the limitation in our analysis.

        Returns:
        --------
        pd.DataFrame : Balance check results (may be limited if no pre-treatment treated units exist)
        """
        import logging
        logger = logging.getLogger('digital_transformation')
        
        balance_vars = config.CONTROL_VARS + config.DT_MEASURES[:2]
        
        # Filter to pre-treatment period if possible
        if 'Post' in self.data.columns:
            pre_data = self.data[self.data['Post'] == 0].copy()
            logger.info(f"Checking balance in pre-treatment period (Post=0): {len(pre_data)} observations")
        else:
            pre_data = self.data.copy()
            logger.info(f"Post variable not found, checking balance using all data: {len(pre_data)} observations")
        
        # Verify if we have treated units in pre-treatment period
        treat_status = None
        if 'Treat' in pre_data.columns:
            treat_vals = pre_data['Treat'].dropna().unique()
            treat_status = f"Found treatment values: {treat_vals} in pre-treatment period"
            
            if len(treat_vals) < 2 or 1 not in treat_vals:
                logger.warning(f"No treated units (Treat=1) found in pre-treatment period. Treatment values: {treat_vals}")
                logger.warning("This is common in staggered adoption designs where units become treated only after the policy change.")
                logger.warning("Will provide descriptive statistics for pre-treatment observations instead.")
        
        # Ensure all balance variables exist and are numeric
        valid_vars = []
        for var in balance_vars:
            if var in pre_data.columns:
                try:
                    # Use .loc to avoid SettingWithCopyWarning
                    pre_data.loc[:, var] = pd.to_numeric(pre_data[var], errors='coerce')
                    # Only include if we have sufficient non-NaN values
                    if pre_data[var].notna().sum() > pre_data.shape[0] * 0.1:  # At least 10% non-NaN
                        valid_vars.append(var)
                    else:
                        logger.warning(f"Variable {var} has too many missing values, skipping in balance check")
                except Exception as e:
                    logger.warning(f"Could not convert variable {var} to numeric: {e}")
            else:
                logger.warning(f"Variable {var} not found in dataset")
        
        if not valid_vars:
            logger.warning("No valid numeric variables found for balance check")
            # Return empty dataframe with appropriate columns
            return pd.DataFrame(columns=['Variable', 'Treatment Mean', 'Control Mean', 'Difference', 
                                       'Std. Difference', 't-statistic', 'p-value', 'Treatment N', 'Control N'])
        
        # Attempt balance check despite potential lack of treated units
        try:
            # Use allow_missing_groups=True to handle the case where only control group exists
            balance_results = check_balance(
                pre_data,
                treatment_var='Treat',
                balance_vars=valid_vars,
                standardized=True,
                allow_missing_groups=True
            )
            
            # Add note about potential limitation to the results
            if treat_status:
                balance_results.attrs['treat_status'] = treat_status
            
            # Print summary of balance check
            treated_vars = balance_results[balance_results['Treatment N'] > 0]['Variable'].count()
            control_vars = balance_results[balance_results['Control N'] > 0]['Variable'].count()
            complete_vars = balance_results[(balance_results['Treatment N'] > 0) & 
                                           (balance_results['Control N'] > 0)]['Variable'].count()
            
            if treated_vars == 0:
                logger.info("No pre-treatment treated units available. Reporting control group statistics only.")
                # Add academic note about staggered adoption limitation
                logger.info("NOTE: In difference-in-differences with staggered adoption, we typically lack pre-treatment treated units.")
                logger.info("This affects conventional balance tests but is consistent with the research design.")
            else:
                logger.info(f"Balance check completed: {len(balance_results)} variables checked, "
                           f"{treated_vars} with treatment data, {control_vars} with control data, "
                           f"{complete_vars} with both")
            
            return balance_results
        
        except Exception as e:
            logger.error(f"Error in treatment balance: {e}")
            # Return empty dataframe with appropriate columns
            return pd.DataFrame(columns=['Variable', 'Treatment Mean', 'Control Mean', 'Difference', 
                                       'Std. Difference', 't-statistic', 'p-value', 'Treatment N', 'Control N'])

    def create_treatment_distribution_table(self):
        """
        Create a table showing the distribution of treatment status over time

        Returns:
        --------
        dict: Treatment distribution tables
        """
        # Check if required variables exist
        if not all(var in self.data.columns for var in ['Treat', 'Post', 'year']):
            raise ValueError(
                "Required variables (Treat, Post, year) not found")
        
        # Create cross-tabulation
        treat_post_table = pd.crosstab(
            self.data['Post'],
            self.data['Treat'],
            rownames=['Post'],
            colnames=['Treat'],
            margins=True
        )
        
        # Create year-wise distribution
        yearly_table = pd.crosstab(
            self.data['year'],
            self.data['Treat'],
            rownames=['Year'],
            colnames=['Treat'],
            margins=True
        )
        
        # MSCI inclusion by year if available
        if 'MSCI' in self.data.columns:
            msci_yearly = pd.crosstab(
                self.data['year'],
                self.data['MSCI'],
                rownames=['Year'],
                colnames=['MSCI'],
                margins=True
            )
            return {'treat_post': treat_post_table, 'yearly': yearly_table, 'msci_yearly': msci_yearly}
        else:
            return {'treat_post': treat_post_table, 'yearly': yearly_table}

    def export_results(self, output_dir=None):
        """
        Export descriptive statistics results to files

        Parameters:
        -----------
        output_dir : str or Path or None
            Directory to save results, default to config.TABLES_DIR
        """
        if output_dir is None:
            output_dir = config.TABLES_DIR
        else:
            output_dir = Path(output_dir)
        
        # Create directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate and save summary statistics
        summary = self.summarize_variables()
        summary.to_csv(output_dir / "summary_statistics.csv")
        
        # Generate and save summary by treatment group
        treat_summary = self.summarize_variables(by_group='Treat')
        for group, df in treat_summary.items():
            df.to_csv(output_dir / f"summary_{group}.csv")

        # Generate and save correlation matrix
        corr = self.correlation_analysis()
        corr.to_csv(output_dir / "correlation_matrix.csv")

        # Generate and save treatment balance
        try:
            balance = self.treatment_balance()
            balance.to_csv(output_dir / "treatment_balance.csv")
        except Exception as e:
            print(f"Error generating treatment balance: {e}")

        # Generate and save treatment distribution
        try:
            dist_tables = self.create_treatment_distribution_table()
            for name, table in dist_tables.items():
                table.to_csv(output_dir / f"treatment_dist_{name}.csv")
        except Exception as e:
            print(f"Error generating treatment distribution: {e}")

        print(f"Descriptive statistics results saved to {output_dir}")
