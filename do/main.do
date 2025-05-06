/******************************************************************************
* Project: Does Capital Market Liberalization Drive Digital Transformation?
* Insights from Chinese A-Shares Inclusion in the MSCI Emerging Markets Index
*
* MAIN CONTROL FILE: Orchestrates the execution of all analysis components
******************************************************************************/

// Clear environment
clear all
set more off
capture log close
capture log close _all

// Define project directory structure (adjust as needed)
global project_dir "/Users/hetawk/Desktop/Desktop/Desktop/Others/Patience/Thesis/data/data-code/patie_preprocess"
global do_dir     "${project_dir}/do"
global data_in    "${project_dir}/dataset"
global data_out   "${project_dir}/dataset"
global results    "${project_dir}/results"
global logs       "${project_dir}/results/logs"

// Create necessary directories
foreach dir in "${results}" "${results}/tables" "${results}/figures" "${logs}" {
    capture mkdir "`dir'"
}

// Start logging
log using "${logs}/msci_digital_transformation_analysis.log", replace text

// Display basic project information
display "Project: MSCI Inclusion and Digital Transformation Analysis"
display "Date: $S_DATE"
display "Time: $S_TIME"
display "Running modular analysis scripts..."

// Load utility functions first
run "${do_dir}/utils.do"

// Execute analysis modules in sequence
do "${do_dir}/setup.do"
do "${do_dir}/prep.do"
do "${do_dir}/analysis.do"
do "${do_dir}/mechanisms.do"
do "${do_dir}/heterogeneity.do"
do "${do_dir}/visualize.do"
do "${do_dir}/report.do"

// Close log and clean up
log close
display _newline "==== MSCI Digital Transformation Analysis Complete ===="
