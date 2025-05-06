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
    
    foreach fin_var in $financial_access_vars {
        // Create abbreviated mechanism name
        if "`fin_var'" == "SA_index" local fin_abbr "SA"
        else if "`fin_var'" == "WW_index" local fin_abbr "WW"
        
        // Run the mechanism regression
        RunMechanismReg `dt_var', mechanism(`fin_var') store(`dt_abbr'_`fin_abbr'_post_fe) controls($control_vars)
    }
}

// Export Financial Access results
ExportMechanismResults DTA* DTB*, title("Mechanism Analysis: Financial Access (Post-Period FE)") ///
    vars($financial_access_vars) filename("mechanism_financial_access")

// --- Corporate Governance Mechanism ---
display _newline "--- Mechanism Analysis: Corporate Governance ---"
eststo clear

foreach dt_var in Digital_transformationA Digital_transformationB {
    if "`dt_var'" == "Digital_transformationA" local dt_abbr "DTA"
    else if "`dt_var'" == "Digital_transformationB" local dt_abbr "DTB"
    
    foreach cg_var in $corp_gov_vars {
        // Skip if variable doesn't exist
        capture confirm variable `cg_var'
        if _rc != 0 {
            display as error "Warning: Mechanism variable `cg_var' not found. Skipping."
            continue
        }
        
        // Abbreviate mechanism name
        if "`cg_var'" == "Top3DirectorSumSalary2" local cg_abbr "DSal"
        else if "`cg_var'" == "DirectorHoldSum2" local cg_abbr "DHold"
        else if "`cg_var'" == "DirectorUnpaidNo2" local cg_abbr "DUnp"
        
        // Run mechanism regression
        RunMechanismReg `dt_var', mechanism(`cg_var') store(`dt_abbr'_`cg_abbr'_post_fe) controls($control_vars)
    }
}

// Export Corporate Governance results
ExportMechanismResults DTA* DTB*, title("Mechanism Analysis: Corporate Governance (Post-Period FE)") ///
    vars($corp_gov_vars) filename("mechanism_corp_gov")

// --- Investor Scrutiny Mechanism ---
display _newline "--- Mechanism Analysis: Investor Scrutiny ---"
eststo clear

foreach dt_var in Digital_transformationA Digital_transformationB {
    if "`dt_var'" == "Digital_transformationA" local dt_abbr "DTA"
    else if "`dt_var'" == "Digital_transformationB" local dt_abbr "DTB"
    
    foreach is_var in $investor_scrutiny_vars {
        // Skip if variable doesn't exist
        capture confirm variable `is_var'
        if _rc != 0 {
            display as error "Warning: Mechanism variable `is_var' not found. Skipping."
            continue
        }
        
        // Abbreviate mechanism name
        if "`is_var'" == "ESG_Score_mean" local is_abbr "ESG"
        
        // Run mechanism regression
        RunMechanismReg `dt_var', mechanism(`is_var') store(`dt_abbr'_`is_abbr'_post_fe) controls($control_vars)
    }
}

// Export Investor Scrutiny results
ExportMechanismResults DTA* DTB*, title("Mechanism Analysis: Investor Scrutiny (Post-Period FE)") ///
    vars($investor_scrutiny_vars) filename("mechanism_investor_scrutiny")

// Create mechanism visualization
CreateMechanismVisualization
