/*===========================================================================
 Convert MSCI Digital Transformation dataset from .dta to .csv format
 with improved encoding and delimiters for optimal Python compatibility
===========================================================================*/

// Clear any existing data and set trace for debugging
clear all
set more off
set tracedepth 3
timer clear
timer on 1

// Display start message
display as text _newline "========================================================="
display as text "Converting MSCI dataset from .dta to .csv format for Python"
display as text "========================================================="

// Define input and output file paths
local input_file "dataset/msci_dt_processed_2010_2023.dta"
local output_dir "dataset/export_outputs"
local output_file "dataset/msci_dt_processed_2010_2023.csv"
local output_file_tab "dataset/msci_dt_processed_2010_2023_tab.csv"
local output_file_clean "dataset/msci_dt_processed_2010_2023_clean.csv"
local data_dict "dataset/msci_dt_variable_dict.csv"

// Create export output directory if it doesn't exist
capture mkdir "`output_dir'"

// Load the dataset
display as text _newline "Checking for input file..."
capture confirm file "`input_file'"
if _rc != 0 {
    display as error "Error: The file `input_file' does not exist."
    exit 601
}

display as text "Loading dataset..."
use "`input_file'", clear

// Get variable count and observation count for verification
local nvars = c(k)
local nobs = c(N)
display as text "Dataset loaded: `nvars' variables, `nobs' observations"

// Create data dictionary for reference (useful for Python data validation)
display as text _newline "Creating data dictionary for Python reference..."

// Store variable labels, types, and basic stats in a CSV
tempfile dict_file
file open dict_handle using "`data_dict'", write replace
file write dict_handle "variable,type,label,format,unique_values,missing_count,has_chinese" _n

// Collect variable information
foreach var of varlist _all {
    // Get variable type
    local type: type `var'
    
    // Get variable label
    local label: variable label `var'
    
    // Replace commas in label with semicolons to avoid CSV parsing issues
    local label = subinstr("`label'", ",", ";", .)
    local label = subinstr("`label'", `"""', "", .)
    
    // Get variable format
    local format: format `var'
    
    // For unique value counting, use a safer approach that handles large datasets
    local unique_vals = "NA"
    capture {
        // Try to determine if variable has few enough values to count safely
        quietly count if !missing(`var')
        if r(N) < 1000 {
            capture quietly tab `var', missing
            if _rc == 0 {
                local unique_vals = r(r)
            }
        }
    }
    
    // Count missing values
    quietly count if missing(`var')
    local missing_count = r(N)
    
    // Check for Chinese characters
    local has_chinese 0
    capture confirm string variable `var'
    if _rc == 0 {
        // For string variables, check if they contain Chinese chars
        quietly count if regexm(`var', "[\u4e00-\u9fff]")
        if r(N) > 0 {
            local has_chinese 1
        }
    }
    
    // Write to dictionary file
    file write dict_handle "`var',`type',`label',`format',`unique_vals',`missing_count',`has_chinese'" _n
}
file close dict_handle
display as text "Data dictionary created: `data_dict'"

// Thoroughly clean string variables for better CSV compatibility
display as text _newline "Cleaning string variables to improve CSV compatibility..."
foreach var of varlist _all {
    capture confirm string variable `var'
    if _rc == 0 {
        // If it's a string variable, apply comprehensive cleaning
        quietly count if !missing(`var')
        if r(N) > 0 {
            display as text "  - Cleaning string variable: `var'"
            
            // Replace problematic characters that could break CSV parsing
            quietly replace `var' = subinstr(`var', char(34), "", .) // Remove double quotes
            quietly replace `var' = subinstr(`var', ",", ";", .) // Replace commas with semicolons
            quietly replace `var' = subinstr(`var', char(10), " ", .) // Replace newlines with spaces
            quietly replace `var' = subinstr(`var', char(13), " ", .) // Replace carriage returns with spaces
            quietly replace `var' = subinstr(`var', char(9), " ", .) // Replace tabs with spaces
            
            // Trim leading/trailing spaces
            quietly replace `var' = strtrim(`var')
        }
    }
}

// Export with optimal settings for Python compatibility
// First, try tab-delimited export (worked best in Python loader)
display as text _newline "Exporting dataset with tab delimiter (primary format)..."
capture export delimited using "`output_file_tab'", replace delimiter(tab)

if _rc != 0 {
    display as error "Error: Failed to export tab-delimited CSV. Error code: `_rc'"
}
else {
    display as text "Successfully exported tab-delimited CSV to: `output_file_tab'"
}

// Also try standard comma-delimited for maximum compatibility
display as text "Exporting comma-delimited version as backup..."
capture export delimited using "`output_file'", replace

if _rc != 0 {
    display as error "Error: Failed to export comma-delimited CSV. Error code: `_rc'"
}
else {
    display as text "Successfully exported comma-delimited CSV to: `output_file'"
}

// Create a clean subset with only essential variables (for easier Python processing)
display as text _newline "Creating clean subset with essential variables..."
keep stkcd code_str year MSCI* Treat Post Digital_transformation* TFP_OP SA_index WW_index F050501B F060101B age

// Export this clean subset
display as text "Exporting clean subset with essential variables..."
capture export delimited using "`output_file_clean'", replace delimiter(tab)

if _rc != 0 {
    display as error "Error: Failed to export clean subset. Trying comma delimiter..."
    capture export delimited using "`output_file_clean'", replace
    
    if _rc != 0 {
        display as error "Error: All export attempts failed for clean subset."
    }
    else {
        display as text "Successfully exported clean subset with comma delimiter to: `output_file_clean'"
    }
}
else {
    display as text "Successfully exported clean subset to: `output_file_clean'"
}

// Additional export: Save a version with Chinese columns transliterated (if possible)
display as text _newline "Checking for Chinese characters in column data..."
capture which ustrto
if _rc == 0 {
    display as text "Unicode transliteration is available. Creating ASCII-friendly version..."
    
    // Loop through string variables and attempt transliteration
    foreach var of varlist _all {
        capture confirm string variable `var'
        if _rc == 0 {
            // Check if there are Chinese characters
            quietly count if regexm(`var', "[\u4e00-\u9fff]") & !missing(`var')
            if r(N) > 0 {
                display as text "  - Transliterating Chinese characters in: `var'"
                
                // Create temporary translated variable
                gen `var'_ascii = `var'
                quietly replace `var'_ascii = subinstr(`var'_ascii, "[\u4e00-\u9fff]", "[CJK]", .)
                
                // Replace original with transliterated version
                drop `var'
                rename `var'_ascii `var'
            }
        }
    }
    
    // Export transliterated version
    capture export delimited using "`output_dir'/msci_dt_ascii.csv", replace delimiter(tab)
    if _rc == 0 {
        display as text "Successfully exported ASCII-friendly version to: `output_dir'/msci_dt_ascii.csv"
    }
}

// Output completion summary
display as text _newline "=== EXPORT SUMMARY ==="
display as text "Original format: Stata .dta"
display as text "New formats:"
display as text "  - Tab-delimited CSV: `output_file_tab'"
display as text "  - Comma-delimited CSV: `output_file'"
display as text "  - Clean subset (essential variables): `output_file_clean'"
display as text "  - Data dictionary: `data_dict'"

// Record time taken
timer off 1
timer list
local secs = r(t1)
display as text _newline "Conversion complete in `secs' seconds."
display as text "Files are ready for Python processing with appropriate delimiters and encoding."






