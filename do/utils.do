/******************************************************************************
* UTILITIES: Helper functions for common tasks across analysis modules
******************************************************************************/

// Function to check if variables exist
capture program drop CheckVarsExist
program define CheckVarsExist
    syntax varlist
    foreach var of varlist `varlist' {
        capture confirm variable `var'
        if _rc != 0 {
            display as error "Error: Variable `var' not found in dataset."
            exit
        }
    }
    display "All required variables confirmed present."
end

// Function to save original data state
capture program drop SaveOriginalData
program define SaveOriginalData
    display "--- Saving original data state ---"
    // Save a persistent backup copy
    capture mkdir "${results}"
    save "${results}/original_data_backup.dta", replace
    
    // Also save to a tempfile for faster access
    tempfile original_data_temp
    save `original_data_temp', replace
    global original_data "`original_data_temp'"
    display "--- Original data saved to tempfile and backup ${results}/original_data_backup.dta ---"
end

// Function to reload original data
capture program drop ReloadOriginalData
program define ReloadOriginalData
    display _newline "--- Reloading original data state ---"
    // Check if the tempfile exists and is valid
    capture confirm file `${original_data}'.dta
    if _rc == 0 {
        use `${original_data}', clear
        display "Successfully reloaded data from tempfile."
    }
    else {
        display "Warning: Tempfile not found or invalid. Reloading from backup."
        capture use "${results}/original_data_backup.dta", clear
        if _rc != 0 {
            display as error "Error: Could not reload data from backup file. Exiting."
            exit 498
        }
        else {
            display "Successfully reloaded data from backup file."
        }
    }
    xtset stkcd year, yearly // Re-apply panel settings
end

// Function to ensure MSCI_clean exists
capture program drop EnsureMSCIClean
program define EnsureMSCIClean
    capture confirm variable MSCI_clean
    if _rc != 0 {
        display "Recreating MSCI_clean variable..."
        gen MSCI_clean = (MSCI==1) if !missing(MSCI)
        label var MSCI_clean "MSCI inclusion (clean binary indicator)"
        display "MSCI_clean variable created."
    }
end

// Function to run DiD model and save results
capture program drop RunDidModel
program define RunDidModel
    syntax varname, treatment(varname) post(varname) interact(varname) [controls(varlist) yearfe(integer 0) cluster(varname) save(name)]
    
    display "Running DiD model for `varlist'..."
    
    // Construct command based on options
    local cmd "qui reg `varlist' `treatment' `post' `interact'"
    if "`controls'" != "" {
        local cmd "`cmd' `controls'"
    }
    if `yearfe' == 1 {
        local cmd "`cmd' i.year"
    }
    if "`cluster'" != "" {
        local cmd "`cmd', cluster(`cluster')"
    }
    
    // Run the command
    `cmd'
    
    // Extract and save results if requested
    if "`save'" != "" {
        local coef = _b[`interact']
        local se = _se[`interact']
        
        // Check if results are valid
        if missing(`coef') | missing(`se') | `se' == 0 {
            display as error "Warning: DiD regression failed or produced invalid results."
            local `save'_coef = .
            local `save'_se = .
            local `save'_p = .
            local `save'_t = .
            local `save'_result "Inconclusive (estimation failed)"
        } 
        else {
            local t = `coef' / `se'
            local p = 2 * ttail(e(df_r), abs(`t'))
            local result = cond(`p' < 0.05, "supported", "not supported")
            
            global `save'_coef = `coef'
            global `save'_se = `se'
            global `save'_p = `p'
            global `save'_t = `t'
            global `save'_result = "`result'"
            
            display "Effect size: `coef', Standard error: `se', t-statistic: `t', p-value: `p'"
            display "Conclusion: Hypothesis is `result' at the 5% significance level"
        }
    }
end

// Function to save hypothesis testing results
capture program drop SaveHypothesisResults
program define SaveHypothesisResults
    args h1 h2
    
    // Ensure file handle is closed
    capture file close hypo
    file open hypo using "${results}/hypothesis_testing_results.txt", write replace
    file write hypo "===========================================================" _n
    file write hypo "        FORMAL HYPOTHESIS TESTING (Pooled OLS DiD)         " _n
    file write hypo "===========================================================" _n _n
    file write hypo "H1: Capital market liberalization positively influences digital transformation" _n
    file write hypo "   Model: reg Digital_transformationA Treat Post TreatPost [controls] i.year, cluster(stkcd)" _n
    file write hypo "   Effect size (TreatPost): " %9.4f (${`h1'_coef}) ", p-value: " %9.4f (${`h1'_p}) _n
    file write hypo "   Conclusion: H1 is " "${`h1'_result}" " at the 5% significance level" _n _n
    file write hypo "H2: MSCI inclusion leads to increased adoption of digital technologies (using Digital_transformationB)" _n
    file write hypo "   Model: reg Digital_transformationB Treat Post TreatPost [controls] i.year, cluster(stkcd)" _n
    file write hypo "   Effect size (TreatPost): " %9.4f (${`h2'_coef}) ", p-value: " %9.4f (${`h2'_p}) _n
    file write hypo "   Conclusion: H2 is " "${`h2'_result}" " at the 5% significance level" _n _n
    file close hypo
    display "Hypothesis testing results saved to ${results}/hypothesis_testing_results.txt"
end

// Function to run descriptive statistics
capture program drop RunDescriptiveStats
program define RunDescriptiveStats
    syntax varlist [, by(varname)]
    
    display _newline "DESCRIPTIVE STATISTICS"
    display "======================"
    
    // Basic summary statistics
    tabstat `varlist', statistics(n mean sd min p25 median p75 max) columns(statistics) format(%9.3f)
    
    // Export summary statistics by group if requested
    if "`by'" != "" {
        eststo clear
        estpost tabstat `varlist', by(`by') statistics(mean sd min max) columns(statistics)
        esttab using "${results}/tables/descriptive_stats.rtf", replace ///
            title("Descriptive Statistics by `by' Group") ///
            cells("mean(fmt(%9.3f)) sd(fmt(%9.3f)) min(fmt(%9.3f)) max(fmt(%9.3f))") ///
            collabels("Mean" "Std. Dev." "Min" "Max") ///
            nonumbers nomtitles label ///
            note("Statistics calculated separately for each `by' group.")
        display "Descriptive statistics saved to ${results}/tables/descriptive_stats.rtf"
    }
end

// Function to create variable definitions
capture program drop CreateVarDefs
program define CreateVarDefs
    // Create variable definitions for different formats
    foreach format in tex rtf csv {
        CreateVarDef_`format'
    }
    display "Variable definitions tables created in multiple formats (TeX, RTF, CSV)."
end

// Function to export regression tables
capture program drop ExportRegTable
program define ExportRegTable
    syntax namelist [, title(string) mtitles(string) keep(string) order(string) filename(string)]
    
    // Default filename if not provided
    if "`filename'" == "" local filename "regression_results"
    
    esttab `namelist' using "${results}/tables/`filename'.rtf", replace ///
        title("`title'") ///
        mtitles(`mtitles') ///
        star(* 0.1 ** 0.05 *** 0.01) ///
        keep(`keep') /// 
        order(`order') ///
        scalars("yearfe Year FE" "controls Controls" "cluster Cluster SE") ///
        b(%9.3f) se(%9.3f) ///
        stats(N N_clust, labels("Observations" "Clusters")) ///
        note("Standard errors in parentheses. * p<0.1, ** p<0.05, *** p<0.01")
    
    display "Regression table saved to ${results}/tables/`filename'.rtf"
end

// Function to run mechanism regression
capture program drop RunMechanismReg
program define RunMechanismReg
    syntax varname, mechanism(varname) store(name) [controls(varlist)]
    
    // Create current controls excluding mechanism variable
    local current_controls `controls'
    local current_controls: list current_controls - mechanism
    
    // Generate interaction term
    tempvar MSCI_mech_interact
    gen `MSCI_mech_interact' = MSCI_clean * `mechanism' if !missing(MSCI_clean, `mechanism')
    
    // Run regression for post period only
    capture xtreg `varlist' MSCI_clean `MSCI_mech_interact' `mechanism' `current_controls' i.year if Post == 1, fe cluster(stkcd)
    
    // Store results if regression was successful
    if _rc == 0 {
        eststo `store'
        estadd local mechanism "`mechanism'"
        estadd local interaction "`MSCI_mech_interact'"
        display "Mechanism regression for `varlist' with `mechanism' stored as `store'"
    }
    else {
        display as error "Warning: Regression failed for `varlist' with `mechanism'. Skipping."
    }
end

// Function to export mechanism results
capture program drop ExportMechanismResults
program define ExportMechanismResults
    syntax namelist [, title(string) vars(string) filename(string)]
    
    // Default filename if not provided
    if "`filename'" == "" local filename "mechanism_results"
    
    capture confirm name `namelist'
    if _rc == 0 {
        esttab `namelist' using "${results}/tables/`filename'.rtf", replace ///
            title("`title'") ///
            star(* 0.1 ** 0.05 *** 0.01) substitute(_cons Constant) ///
            keep(MSCI_clean *interact `vars') ///
            order(MSCI_clean *interact `vars') ///
            b(%9.3f) se(%9.3f) nonumbers mtitles ///
            stats(N N_clust r2_w, fmt(%9.0f %9.0gc %9.3f) labels("Observations" "Clusters" "Within R-sq")) ///
            note("FE models on Post-2018 data. Interaction term is MSCI_clean * Mechanism Variable. Clustered SEs.")
        display "`title' results saved to ${results}/tables/`filename'.rtf"
    } 
    else { 
        display "No `title' results to save." 
    }
end

// Function to run alternative models
capture program drop RunAlternativeModels
program define RunAlternativeModels
    display _newline _newline "====== ALTERNATIVE MODELS FOR MSCI EFFECTS ======" _newline
    
    // Reload original data to ensure clean state
    ReloadOriginalData
    EnsureMSCIClean
    
    // --- Diagnose multicollinearity ---
    display _newline "--- Diagnosing Multicollinearity ---"
    display "Problem: High multicollinearity between Treat and MSCI, no pre-treatment for Treat=1."
    corr Treat Post MSCI
    tab Treat Post, cell column row
    tab MSCI Post, cell column row
    
    // --- Alternative Model 1: Direct MSCI Effect (FE) ---
    display _newline "--- Alternative Model 1: Direct MSCI Effect (FE) ---"
    eststo clear
    capture eststo model1: xtreg Digital_transformationA MSCI_clean $control_vars i.year, fe cluster(stkcd)
    if _rc == 0 {
        estadd local fixedeffects "Yes"
        estadd local yearfe "Yes"
        estadd local controls "Yes"
        
        ExportRegTable model1, title("Alternative Model 1: Direct MSCI Effect (FE)") ///
            keep(MSCI_clean $control_vars) ///
            filename("alt_model1")
    }
    
    // --- Alternative Model 2: Post-Period Estimation (FE) ---
    display _newline "--- Alternative Model 2: Post-Period Subsample (FE) ---"
    capture eststo model2: xtreg Digital_transformationA MSCI_clean $control_vars i.year if Post==1, fe cluster(stkcd)
    if _rc == 0 {
        estadd local fixedeffects "Yes"
        estadd local yearfe "Yes"
        estadd local controls "Yes"
        
        ExportRegTable model2, title("Alternative Model 2: Post-Period Subsample (FE)") ///
            keep(MSCI_clean $control_vars) ///
            filename("alt_model2")
    }
    
    // --- Alternative Model 3: Matched Sample Approach ---
    display _newline "--- Alternative Model 3: Matched Sample Approach ---"
    RunMatchingSample
    
    // --- Alternative Model 4: First Differences ---
    display _newline "--- Alternative Model 4: First Differences (Post-Period) ---"
    sort stkcd year
    by stkcd: gen DT_change_A = D.Digital_transformationA
    
    capture eststo model4: xtreg DT_change_A MSCI_clean $control_vars i.year if Post==1 & !missing(DT_change_A), fe cluster(stkcd)
    if _rc == 0 {
        estadd local fixedeffects "Implicit (FD)"
        estadd local yearfe "Yes"
        estadd local controls "Yes"
        
        ExportRegTable model4, title("Alternative Model 4: First Differences (Post-Period)") ///
            keep(MSCI_clean $control_vars) ///
            filename("alt_model4")
    }
    
    // Combine all models in a single table
    CombineAltModels 
    
    // Create comparison visualization
    display _newline "--- Creating Alternative Models Comparison Plot ---"
    preserve
    clear
    set obs 4
    gen model_num = _n
    gen effect = .
    gen se = .
    gen pvalue = .
    
    // Populate data from stored estimates
    PopulateAltModelsData
    
    // Create plot
    CreateAltModelsPlot, filename("alternative_models_comparison")
    restore
    
    // Save summary text file
    SaveAltModelsSummary
end

// Function to run event study
capture program drop RunEventStudy
program define RunEventStudy
    syntax varname, [controls(varlist)]
    
    display _newline "EVENT STUDY: DYNAMIC TREATMENT EFFECTS"
    display "======================================"
    
    // Check if Event_time exists
    capture confirm variable Event_time
    if _rc != 0 {
        display as error "Event_time variable not found. Cannot run event study."
        global event_study_run = 0
        exit
    }
    
    display "Running event study regression..."
    
    // Generate event time dummies, omitting the base period (t-1)
    tab Event_time, gen(ET_)
    
    // Find base period dummy (Event_time == -1)
    local base_period_dummy ""
    qui tab Event_time
    local num_periods = r(r)
    forvalues i = 1/`num_periods' {
        qui sum Event_time if ET_`i'==1
        if r(mean) == -1 {
            local base_period_dummy "ET_`i'"
            display "Base period dummy is `base_period_dummy' (Event_time = -1)"
        }
    }
    
    // Get all dummies
    ds ET_*
    local event_dummies `r(varlist)'
    
    // Remove base period dummy
    if "`base_period_dummy'" != "" {
        local event_dummies: list event_dummies - base_period_dummy
    } 
    else {
        display as warning "Base period dummy not found. Using first period as base."
        local first_dummy: word 1 of `event_dummies'
        local event_dummies: list event_dummies - first_dummy
    }
    
    // Run event study regression
    eststo clear
    eststo event_study: xtreg `varlist' `event_dummies' `controls' i.year, fe cluster(stkcd)
    estadd local fixedeffects "Yes"; estadd local yearfe "Yes"; estadd local controls "Yes"; estadd local cluster "Firm"
    
    // Export results
    esttab event_study using "${results}/tables/event_study_results.rtf", replace ///
        title("Event Study: Dynamic Effects of MSCI Inclusion") ///
        star(* 0.1 ** 0.05 *** 0.01) substitute(_cons Constant) ///
        keep(`event_dummies') order(`event_dummies') ///
        b(%9.3f) se(%9.3f) nonumbers mtitles ///
        scalars("fixedeffects Firm FE" "yearfe Year FE" "controls Controls" "cluster Cluster SE") ///
        stats(N N_clust r2_w, fmt(%9.0f %9.0gc %9.3f) labels("Observations" "Clusters" "Within R-sq")) ///
        note("Event study regression with firm and year fixed effects. Coefficients represent effect relative to t-1.")
    
    global event_study_run = 1
    display "Event study analysis complete."
end

// Function to run placebo test
capture program drop RunPlaceboTest
program define RunPlaceboTest
    syntax varname, placebo_year(integer) [controls(varlist)]
    
    display _newline "PLACEBO TEST"
    display "============="
    
    // Check required variables
    capture confirm variable year
    if _rc != 0 {
        display as error "Year variable missing. Cannot run placebo test."
        global placebo_run = 0
        exit
    }
    
    display "Running placebo test assuming treatment in `placebo_year'..."
    
    // Create placebo variables
    gen Placebo_Post = (year >= `placebo_year')
    gen Placebo_TreatPost = Treat * Placebo_Post
    
    // Run placebo regression
    qui reg `varlist' Treat Placebo_Post Placebo_TreatPost `controls' i.year, cluster(stkcd)
    
    // Store results
    local placebo_coef = _b[Placebo_TreatPost]
    local placebo_se = _se[Placebo_TreatPost]
    
    if missing(`placebo_coef') | missing(`placebo_se') | `placebo_se' == 0 {
        display as error "Placebo regression failed or produced invalid results."
        local placebo_p = .
        local placebo_t = .
    } 
    else {
        local placebo_t = `placebo_coef' / `placebo_se'
        local placebo_p = 2 * ttail(e(df_r), abs(`placebo_t'))
    }
    
    display "Placebo effect (assuming `placebo_year' treatment): `placebo_coef', p-value: `placebo_p'"
    
    // Save results for later use
    global placebo_coef = `placebo_coef'
    global placebo_se = `placebo_se'
    global placebo_p = `placebo_p'
    global placebo_year = `placebo_year'
    global placebo_run = 1
    
    // Compare with actual results if available
    capture confirm global h1_coef
    if _rc == 0 {
        // Create comparison text file
        capture file close placebo
        file open placebo using "${results}/tables/placebo_test_results.txt", write replace
        file write placebo "PLACEBO TEST RESULTS" _n
        file write placebo "===================" _n _n
        file write placebo "Real effect (2018 treatment): " %9.4f ($h1_coef) ", p-value: " %9.4f ($h1_p) _n
        file write placebo "Placebo effect (`placebo_year' treatment): " %9.4f (`placebo_coef') ", p-value: " %9.4f (`placebo_p') _n
        file close placebo
        
        display "Placebo test results saved to ${results}/tables/placebo_test_results.txt"
    }
end

// Function to run industry analysis
capture program drop RunIndustryAnalysis
program define RunIndustryAnalysis
    syntax varname, [controls(varlist)]
    
    // Check if industry variable exists
    capture confirm variable industry
    if _rc != 0 {
        display as error "Industry variable not found. Cannot run industry analysis."
        exit
    }
    
    display "Running industry-specific DiD analysis..."
    
    eststo clear
    
    // Get unique industry values
    levelsof industry, local(industry_levels)
    
    // Run DiD for each industry
    foreach ind of local industry_levels {
        capture eststo ind`ind': reg `varlist' Treat Post TreatPost `controls' i.year if industry == `ind', cluster(stkcd)
        if _rc == 0 {
            estadd local industry "Industry `ind'"
        }
    }
    
    // Export results
    capture confirm name ind*
    if _rc == 0 {
        esttab ind* using "${results}/tables/industry_effects.rtf", replace ///
            title("Industry-Specific Effects of MSCI Inclusion (Pooled OLS DiD)") ///
            star(* 0.1 ** 0.05 *** 0.01) substitute(_cons Constant) ///
            keep(TreatPost) order(TreatPost) ///
            b(%9.3f) se(%9.3f) nonumbers mtitles ///
            scalars("industry Industry Group") ///
            stats(N N_clust, labels("Observations" "Clusters")) ///
            note("Pooled OLS DiD within industry groups. Clustered SEs.")
        display "Industry analysis results saved."
    } 
    else {
        display "No industry analysis results to save."
    }
end

// Function to run sector heterogeneity analysis
capture program drop RunSectorHeterogeneity
program define RunSectorHeterogeneity
    syntax varname, [controls(varlist)]
    
    // Check if sector variable exists
    capture confirm variable sector
    if _rc != 0 {
        display as warning "Sector variable not found. Cannot run sector heterogeneity."
        exit
    }
    
    // Run sector heterogeneity
    eststo clear
    eststo het_sector: xtreg `varlist' c.MSCI_clean##i.sector `controls' i.year if Post == 1, fe cluster(stkcd)
    estadd local fixedeffects "Yes"; estadd local yearfe "Yes"; estadd local controls "Yes"; estadd local cluster "Firm"
    
    // Export results
    esttab het_sector using "${results}/tables/heterogeneity_sector.rtf", replace ///
        title("Heterogeneity by Sector (Post-Period FE)") ///
        star(* 0.1 ** 0.05 *** 0.01) substitute(_cons Constant) ///
        keep(*MSCI_clean* *sector*) order(*MSCI_clean* *sector*) ///
        b(%9.3f) se(%9.3f) nonumbers mtitles ///
        scalars("fixedeffects Firm FE" "yearfe Year FE" "controls Controls" "cluster Cluster SE") ///
        stats(N N_clust r2_w, fmt(%9.0f %9.0gc %9.3f) labels("Observations" "Clusters" "Within R-sq")) ///
        note("FE model on Post-2018 data. Interaction terms show differential effect by sector. Clustered SEs.")
    
    display "Sector heterogeneity results saved."
end

// Function to run components analysis
capture program drop RunComponentsAnalysis
program define RunComponentsAnalysis
    syntax varname, [controls(varlist)]
    
    // Define possible components
    local dt_components "cloud_adoption ai_investment digital_workforce digital_marketing"
    
    // Check if component variables exist
    local valid_components = 0
    foreach comp of local dt_components {
        capture confirm variable `comp'
        if _rc == 0 local valid_components = `valid_components' + 1
    }
    
    if `valid_components' > 0 {
        display "Running component-level analysis for `valid_components' digital transformation components..."
        eststo clear
        
        foreach comp of local dt_components {
            capture confirm variable `comp'
            if _rc == 0 {
                eststo `comp': reg `comp' Treat Post TreatPost `controls' i.year, cluster(stkcd)
            }
        }
        
        // Export results
        ExportComponentsResults, components("`dt_components'")
    }
    else {
        // Create simulated components for demonstration if requested
        display "Note: Digital transformation component variables not found."
    }
end

// Additional helper functions would be implemented below
// Such as functions for creating specific visualization types,
// handling variable definitions in different formats, etc.
