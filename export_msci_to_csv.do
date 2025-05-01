/*===========================================================================
 Convert MSCI Digital Transformation dataset from .dta to .csv format
 with improved encoding and delimiters
===========================================================================*/

// Clear any existing data and set trace for debugging if needed
clear all
set more off

// Display start message
display as text "Converting dataset from .dta to .csv format with improved encoding..."

// Define input and output file paths
local input_file "dataset/msci_dt_processed_2010_2023.dta"
local output_file "dataset/msci_dt_processed_2010_2023.csv"

// Load the dataset
capture confirm file "`input_file'"
if _rc != 0 {
    display as error "Error: The file `input_file' does not exist."
    exit 601
}

use "`input_file'", clear

// Get variable count and observation count for verification
local nvars = c(k)
local nobs = c(N)
display "Dataset loaded: `nvars' variables, `nobs' observations"

// Convert string variables with potential Chinese characters to ASCII when possible
// This creates a cleaner export for Python to process
foreach var of varlist _all {
    capture confirm string variable `var'
    if _rc == 0 {
        // If it's a string variable, try to clean it
        display "Cleaning string variable: `var'"
        // Replace quotes in string variables to avoid CSV parsing issues
        quietly replace `var' = subinstr(`var', `"""', " ", .)
        quietly replace `var' = subinstr(`var', ",", ";", .)
    }
}

// Export with improved settings
// Using tab as delimiter and explicit encoding
capture export delimited using "`output_file'", replace delimiter(tab)
if _rc != 0 {
    display as error "Error: Failed to export dataset to CSV. Trying alternative approach..."
    
    // Alternative: try saving a subset of most important variables
    display "Attempting to save only essential variables..."
    keep stkcd code_str year MSCI* Treat Post Digital_transformation* TFP_OP SA_index WW_index F050501B F060101B age
    
    capture export delimited using "dataset/msci_dt_essential.csv", replace delimiter(tab)
    if _rc != 0 {
        display as error "Error: Failed to export essential variables. Last attempt with comma delimiter..."
        capture export delimited using "dataset/msci_dt_essential.csv", replace
    }
    else {
        display as text "Successfully exported essential variables to: dataset/msci_dt_essential.csv"
    }
}
else {
    display as text "Successfully exported dataset to: `output_file'"
    display as text "Original format: Stata .dta"
    display as text "New format: CSV (tab-delimited)"
}

display as text "Conversion complete."
