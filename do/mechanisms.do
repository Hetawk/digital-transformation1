/******************************************************************************
* MECHANISM ANALYSIS: Investigating channels through which MSCI affects DT
******************************************************************************/

// Reload original data to ensure clean state
ReloadOriginalData

// Make sure MSCI_clean exists
EnsureMSCIClean

display _newline
display "==========================================================================="
display "         RESEARCH QUESTION 2: MECHANISM ANALYSIS (HOW?)                    "
display "==========================================================================="
display "Analyzing potential mechanisms: Financial Access, Corporate Governance, Investor Scrutiny"
display "Model: DT = b0 + b1*MSCI_clean + b2*MSCI_clean*Mechanism + b3*Mechanism + Controls + Year FE + Firm FE (Post-Period Only)"

// --- Financial Access Mechanism ---
display _newline "--- Mechanism Analysis: Financial Access ---"

// Clear previous results
eststo clear

// Run Financial Access mechanism regressions
foreach dt_var in Digital_transformationA Digital_transformationB {
    // Create shorter name for estimation storage
    if "`dt_var'" == "Digital_transformationA" local dt_abbr "DTA"
    else if "`dt_var'" == "Digital_transformationB" local dt_abbr "DTB"
    
    foreach fin_var in SA_index WW_index {
        // Create abbreviated mechanism name
        if "`fin_var'" == "SA_index" local fin_abbr "SA"
        else if "`fin_var'" == "WW_index" local fin_abbr "WW"
        
        // Run the mechanism regression
        RunMechanismReg `dt_var', mechanism(`fin_var') store(`dt_abbr'_`fin_abbr'_post_fe) controls($control_vars)
    }
}

// Export Financial Access results (fixed to avoid wildcards)
ExportMechanismTable DTA_SA_post_fe DTA_WW_post_fe DTB_SA_post_fe DTB_WW_post_fe, ///
    title("Mechanism Analysis: Financial Access (Post-Period FE)") ///
    vars(SA_index WW_index) filename("mechanism_financial_access")

// --- Corporate Governance Mechanism ---
display _newline "--- Mechanism Analysis: Corporate Governance ---"

// Clear previous results
eststo clear

// Run Corporate Governance mechanism regressions
foreach dt_var in Digital_transformationA Digital_transformationB {
    // Create shorter name for estimation storage
    if "`dt_var'" == "Digital_transformationA" local dt_abbr "DTA"
    else if "`dt_var'" == "Digital_transformationB" local dt_abbr "DTB"
    
    // First check which corporate governance variables exist in the dataset
    local available_gov_vars ""
    foreach var in Top3DirectorSumSalary2 DirectorHoldSum2 DirectorUnpaidNo2 {
        capture confirm variable `var'
        if _rc == 0 {
            local available_gov_vars "`available_gov_vars' `var'"
        }
        else {
            display as error "Warning: Variable `var' not found in dataset. Skipping this variable."
        }
    }
    
    if "`available_gov_vars'" == "" {
        display as error "No corporate governance variables found in dataset. Skipping corporate governance analysis."
        continue
    }
    
    // Run mechanism regressions only for available variables
    foreach gov_var of local available_gov_vars {
        // Create abbreviated mechanism name
        if "`gov_var'" == "Top3DirectorSumSalary2" local gov_abbr "DS"
        else if "`gov_var'" == "DirectorHoldSum2" local gov_abbr "DH"
        else if "`gov_var'" == "DirectorUnpaidNo2" local gov_abbr "DU"
        
        // Run the mechanism regression
        RunMechanismReg `dt_var', mechanism(`gov_var') store(`dt_abbr'_`gov_abbr'_post_fe) controls($control_vars)
    }
}

// Export Corporate Governance results (only for successful estimations)
// First check which results exist
local available_results ""
local available_vars ""

foreach dt_abbr in DTA DTB {
    foreach gov_abbr in DS DH DU {
        capture est restore `dt_abbr'_`gov_abbr'_post_fe
        if _rc == 0 {
            local available_results "`available_results' `dt_abbr'_`gov_abbr'_post_fe"
            est restore `dt_abbr'_`gov_abbr'_post_fe
            
            // Extract corresponding variable name for this result
            if "`gov_abbr'" == "DS" {
                local var_name "Top3DirectorSumSalary2"
                // Add to available vars if not already included
                if !`: list var_name in available_vars' {
                    local available_vars "`available_vars' `var_name'"
                }
            }
            else if "`gov_abbr'" == "DH" {
                local var_name "DirectorHoldSum2"
                if !`: list var_name in available_vars' {
                    local available_vars "`available_vars' `var_name'"
                }
            }
            else if "`gov_abbr'" == "DU" {
                local var_name "DirectorUnpaidNo2"
                if !`: list var_name in available_vars' {
                    local available_vars "`available_vars' `var_name'"
                }
            }
        }
    }
}

if "`available_results'" != "" {
    ExportMechanismTable `available_results', ///
        title("Mechanism Analysis: Corporate Governance (Post-Period FE)") ///
        vars(`available_vars') filename("mechanism_corp_gov")
}
else {
    display as error "No corporate governance mechanism results available to export."
}

// --- Investor Scrutiny Mechanism ---
display _newline "--- Mechanism Analysis: Investor Scrutiny ---"

// Clear previous results
eststo clear

// Run Investor Scrutiny mechanism regressions
foreach dt_var in Digital_transformationA Digital_transformationB {
    // Create shorter name for estimation storage
    if "`dt_var'" == "Digital_transformationA" local dt_abbr "DTA"
    else if "`dt_var'" == "Digital_transformationB" local dt_abbr "DTB"
    
    foreach inv_var in ESG_Score_mean {
        // Create abbreviated mechanism name
        if "`inv_var'" == "ESG_Score_mean" local inv_abbr "ESG"
        
        // Run the mechanism regression
        RunMechanismReg `dt_var', mechanism(`inv_var') store(`dt_abbr'_`inv_abbr'_post_fe) controls($control_vars)
    }
}

// Export Investor Scrutiny results (fixed to avoid wildcards)
ExportMechanismTable DTA_ESG_post_fe DTB_ESG_post_fe, ///
    title("Mechanism Analysis: Investor Scrutiny (Post-Period FE)") ///
    vars(ESG_Score_mean) filename("mechanism_investor_scrutiny")

// Create mechanism visualization
// Check if program exists before running
capture program list CreateMechanismVisualization
if _rc != 0 {
    // If program doesn't exist, display a message instead of throwing an error
    display as text "Note: CreateMechanismVisualization program not found. Skipping visualization."
    display as text "This is not an error - visualization can be created separately using Python if needed."
}
else {
    // Run the program if it exists
    CreateMechanismVisualization
}
