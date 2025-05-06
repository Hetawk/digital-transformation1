/******************************************************************************
* PREPROCESSING: Data loading, validation, and initial setup
******************************************************************************/

// Load the pre-processed dataset that contains all necessary variables
capture use "${data_in}/msci_dt_processed_2010_2023.dta", clear
if _rc != 0 {
    // Try alternative locations
    display as error "Error loading dataset from '${data_in}/msci_dt_processed_2010_2023.dta', trying alternative locations..."
    capture use "msci_dt_processed_2010_2023.dta", clear

    if _rc != 0 {
        display as error "Could not find dataset 'msci_dt_processed_2010_2023.dta'. Please check path and try again."
        exit
    }
}

display "Loaded pre-processed dataset (2010-2023) with treatment variables"

// Verify key variables exist
CheckVarsExist year stkcd Treat Post MSCI Digital_transformationA Digital_transformationB SA_index WW_index F050501B F060101B

// Set up panel structure
xtset stkcd year, yearly

// Add dataset diagnostic section
display _newline "==== DATASET DIAGNOSTICS ===="
// Check for variable completeness
codebook year Treat Post MSCI stkcd Digital_transformationA, compact
// Examine treatment variables distribution
tab Treat Post, row col
// Describe digital transformation variables
summarize $dt_measures, detail
// Check for key collinearity issues
corr Treat Post MSCI
display "==========================" _newline

// Create necessary indicators and variables for analysis
// Generate interaction term
gen TreatPost = Treat * Post
label var TreatPost "Interaction term (Treat × Post)"

// Create clean MSCI indicator
gen MSCI_clean = (MSCI==1) if !missing(MSCI)
label var MSCI_clean "MSCI inclusion (clean binary indicator)"

// Check if creation was successful
capture confirm variable MSCI_clean
if _rc != 0 {
    display as error "Failed to create MSCI_clean variable. Check MSCI variable."
    exit
} 
else {
    display "Successfully created MSCI_clean variable for analysis."
}

// Save original data state
SaveOriginalData
