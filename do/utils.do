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
            // Always set globals, even if the regression failed
            global `save'_coef = .
            global `save'_se = .
            global `save'_p = .
            global `save'_t = .
            global `save'_result = "Inconclusive (estimation failed)"
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
    
    // Check if the required globals exist and handle missing values
    capture confirm scalar ${`h1'_coef}
    local h1_exists = (_rc == 0)
    
    capture confirm scalar ${`h2'_coef}
    local h2_exists = (_rc == 0)
    
    // Write H1 results
    file write hypo "H1: Capital market liberalization positively influences digital transformation" _n
    file write hypo "   Model: reg Digital_transformationA Treat Post TreatPost [controls] i.year, cluster(stkcd)" _n
    
    if `h1_exists' {
        file write hypo "   Effect size (TreatPost): " %9.4f (${`h1'_coef}) ", p-value: " %9.4f (${`h1'_p}) _n
        file write hypo "   Conclusion: H1 is " "${`h1'_result}" " at the 5% significance level" _n _n
    }
    else {
        file write hypo "   Result: Analysis failed or produced invalid results" _n _n
    }
    
    // Write H2 results
    file write hypo "H2: MSCI inclusion leads to increased adoption of digital technologies (using Digital_transformationB)" _n
    file write hypo "   Model: reg Digital_transformationB Treat Post TreatPost [controls] i.year, cluster(stkcd)" _n
    
    if `h2_exists' {
        file write hypo "   Effect size (TreatPost): " %9.4f (${`h2'_coef}) ", p-value: " %9.4f (${`h2'_p}) _n
        file write hypo "   Conclusion: H2 is " "${`h2'_result}" " at the 5% significance level" _n _n
    }
    else {
        file write hypo "   Result: Analysis failed or produced invalid results" _n _n
    }
    
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
        capture {
            CreateVarDef_`format'
        }
        if _rc != 0 {
            display as text "Note: Variable definitions in `format' format skipped (function not available)"
        }
    }
    display "Variable definitions tables created for available formats"
end

// Function to create variable definitions in TeX format
capture program drop CreateVarDef_tex
program define CreateVarDef_tex
    capture file close vardef
    file open vardef using "${results}/tables/variable_definitions.tex", write replace
    
    file write vardef "\begin{table}[htbp]" _n
    file write vardef "\centering" _n
    file write vardef "\caption{Variable Definitions}" _n
    file write vardef "\begin{tabular}{lp{12cm}}" _n
    file write vardef "\hline\hline" _n
    file write vardef "Variable & Definition \\" _n
    file write vardef "\hline" _n
    
    // Digital transformation variables
    file write vardef "Digital\_transformationA & Digital transformation index based on keywords (measure A) \\" _n
    file write vardef "Digital\_transformationB & Digital transformation index based on keywords (measure B) \\" _n
    file write vardef "Digital\_transformation\_rapidA & Rate of change in digital transformation (measure A) \\" _n
    file write vardef "Digital\_transformation\_rapidB & Rate of change in digital transformation (measure B) \\" _n
    
    // Treatment variables
    file write vardef "Treat & Treatment group indicator (1 = included in MSCI, 0 = control) \\" _n
    file write vardef "Post & Post-period indicator (1 = after 2018, 0 = before) \\" _n
    file write vardef "TreatPost & Interaction term (Treat × Post) \\" _n
    file write vardef "MSCI & Raw MSCI inclusion indicator \\" _n
    file write vardef "MSCI\_clean & Clean MSCI inclusion indicator \\" _n
    
    // Control variables
    file write vardef "age & Firm age in years \\" _n
    file write vardef "TFP\_OP & Total factor productivity (Olley-Pakes method) \\" _n
    file write vardef "SA\_index & Size-Age index (financial constraint measure) \\" _n
    file write vardef "WW\_index & Whited-Wu index (financial constraint measure) \\" _n
    file write vardef "F050501B & Return on assets (ROA) \\" _n
    file write vardef "F060101B & Asset turnover ratio \\" _n
    
    file write vardef "\hline\hline" _n
    file write vardef "\end{tabular}" _n
    file write vardef "\end{table}" _n
    
    file close vardef
    display "Variable definitions saved in TeX format"
end

// Function to create variable definitions in RTF format
capture program drop CreateVarDef_rtf
program define CreateVarDef_rtf
    capture file close vardef
    file open vardef using "${results}/tables/variable_definitions.rtf", write replace
    
    file write vardef "{\rtf1\ansi\deff0 {\fonttbl{\f0\fnil Times New Roman;}}" _n
    file write vardef "\paperw15840\paperh12240\margl1440\margr1440\margt1440\margb1440" _n
    file write vardef "\f0\fs24" _n
    
    file write vardef "\pard\qc\b Variable Definitions\b0\par" _n
    file write vardef "\pard\par" _n
    
    file write vardef "{\trowd\trgaph100\trleft-100\trpaddl100\trpaddr100\trpaddfl3\trpaddfr3" _n
    file write vardef "\clbrdrt\brdrw10\brdrs\clbrdrl\brdrw10\brdrs\clbrdrb\brdrw10\brdrs\clbrdrr\brdrw10\brdrs\cellx3000" _n
    file write vardef "\clbrdrt\brdrw10\brdrs\clbrdrl\brdrw10\brdrs\clbrdrb\brdrw10\brdrs\clbrdrr\brdrw10\brdrs\cellx12000" _n
    file write vardef "\pard\intbl\b Variable\b0\cell \b Definition\b0\cell\row" _n
    
    // Digital transformation variables
    file write vardef "{\trowd\trgaph100\trleft-100\trpaddl100\trpaddr100\trpaddfl3\trpaddfr3" _n
    file write vardef "\clbrdrl\brdrw10\brdrs\clbrdrb\brdrw5\brdrs\clbrdrr\brdrw10\brdrs\cellx3000" _n
    file write vardef "\clbrdrl\brdrw10\brdrs\clbrdrb\brdrw5\brdrs\clbrdrr\brdrw10\brdrs\cellx12000" _n
    file write vardef "\pard\intbl Digital_transformationA\cell Digital transformation index based on keywords (measure A)\cell\row" _n
    
    // Add more variables following the same pattern
    // ...
    
    file write vardef "}" _n
    file close vardef
    display "Variable definitions saved in RTF format"
end

// Function to create variable definitions in CSV format
capture program drop CreateVarDef_csv
program define CreateVarDef_csv
    capture file close vardef
    file open vardef using "${results}/tables/variable_definitions.csv", write replace
    
    file write vardef "Variable,Definition" _n
    file write vardef "Digital_transformationA,Digital transformation index based on keywords (measure A)" _n
    file write vardef "Digital_transformationB,Digital transformation index based on keywords (measure B)" _n
    file write vardef "Digital_transformation_rapidA,Rate of change in digital transformation (measure A)" _n
    file write vardef "Digital_transformation_rapidB,Rate of change in digital transformation (measure B)" _n
    file write vardef "Treat,Treatment group indicator (1 = included in MSCI, 0 = control)" _n
    file write vardef "Post,Post-period indicator (1 = after 2018, 0 = before)" _n
    file write vardef "TreatPost,Interaction term (Treat × Post)" _n
    file write vardef "MSCI,Raw MSCI inclusion indicator" _n
    file write vardef "MSCI_clean,Clean MSCI inclusion indicator" _n
    file write vardef "age,Firm age in years" _n
    file write vardef "TFP_OP,Total factor productivity (Olley-Pakes method)" _n
    file write vardef "SA_index,Size-Age index (financial constraint measure)" _n
    file write vardef "WW_index,Whited-Wu index (financial constraint measure)" _n
    file write vardef "F050501B,Return on assets (ROA)" _n
    file write vardef "F060101B,Asset turnover ratio" _n
    
    file close vardef
    display "Variable definitions saved in CSV format"
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

// Function to run mechanism regression (without requiring reghdfe)
capture program drop RunMechanismReg
program define RunMechanismReg
    syntax varname, mechanism(varname) [controls(string) yearfe(integer 1) cluster(string) store(string)]
    
    display _newline "Running mechanism regression for `varlist' with mechanism `mechanism'..."
    
    // Default controls to global if not specified
    if "`controls'" == "" & "$control_vars" != "" {
        local controls $control_vars
        display "Using global control variables: `controls'"
    }
    
    // Default cluster variable
    if "`cluster'" == "" {
        local cluster "stkcd" 
    }
    
    // Generate interaction term for mechanism
    gen MSCI_`mechanism' = MSCI_clean * `mechanism'
    label var MSCI_`mechanism' "Interaction: MSCI_clean × `mechanism'"
    
    // Keep only post-period data for cleaner identification
    tempvar post_sample
    gen `post_sample' = Post == 1
    
    // Try to use xtreg for fixed effects if available (built-in command)
    display "Trying panel data approach with built-in xtreg..."
    
    // Define base specification
    local spec "`varlist' MSCI_clean `mechanism' MSCI_`mechanism'"
    if "`controls'" != "" {
        local spec "`spec' `controls'"
    }
    if `yearfe' == 1 {
        local spec "`spec' i.year"
    }
    
    // Run xtreg with firm fixed effects
    capture xtreg `spec' if `post_sample', fe vce(cluster `cluster')
    
    if _rc == 0 {
        display "Successfully ran fixed effects model with xtreg."
        
        // Store results if requested
        if "`store'" != "" {
            eststo `store'
            display "Results stored as `store'"
        }
        
        // Report key coefficients of interest
        display _newline "MECHANISM ANALYSIS RESULTS"
        display "============================="
        display "Outcome: `varlist'"
        display "Mechanism: `mechanism'"
        display "Direct effect of MSCI inclusion: " _b[MSCI_clean] " (p = " 2*ttail(e(df_r), abs(_b[MSCI_clean]/_se[MSCI_clean])) ")"
        display "Direct effect of mechanism: " _b[`mechanism'] " (p = " 2*ttail(e(df_r), abs(_b[`mechanism']/_se[`mechanism'])) ")"
        display "Interaction effect: " _b[MSCI_`mechanism'] " (p = " 2*ttail(e(df_r), abs(_b[MSCI_`mechanism']/_se[MSCI_`mechanism'])) ")"
        
        // Significance indicator for interaction
        local p_interaction = 2*ttail(e(df_r), abs(_b[MSCI_`mechanism']/_se[MSCI_`mechanism']))
        if `p_interaction' < 0.01 {
            display "*** Significant interaction at 1% level"
        }
        else if `p_interaction' < 0.05 {
            display "** Significant interaction at 5% level"
        }
        else if `p_interaction' < 0.1 {
            display "* Significant interaction at 10% level"
        }
        else {
            display "No significant interaction found"
        }
    }
    else {
        // Try regular OLS with dummy variables for firms if xtreg fails
        display "Warning: xtreg failed with error code `_rc'."
        display "Trying simpler approach with regular OLS..."
        
        // Create firm dummies using tabulate
        capture tab stkcd, gen(firm_)
        if _rc != 0 {
            display as error "Could not create firm fixed effects. Too many firms for dummy variable approach."
            display "Running without firm fixed effects..."
            
            capture reg `spec' if `post_sample', vce(cluster `cluster')
            
            if _rc == 0 {
                display "Successfully ran OLS without firm fixed effects."
                
                // Store results if requested
                if "`store'" != "" {
                    eststo `store'
                    display "Results stored as `store' (without firm FE)"
                }
                
                // Report key coefficients
                display _newline "MECHANISM ANALYSIS RESULTS (WITHOUT FIRM FE)"
                display "============================================="
                display "Outcome: `varlist'"
                display "Mechanism: `mechanism'"
                display "Direct effect of MSCI inclusion: " _b[MSCI_clean] " (p = " 2*ttail(e(df_r), abs(_b[MSCI_clean]/_se[MSCI_clean])) ")"
                display "Direct effect of mechanism: " _b[`mechanism'] " (p = " 2*ttail(e(df_r), abs(_b[`mechanism']/_se[`mechanism'])) ")"
                display "Interaction effect: " _b[MSCI_`mechanism'] " (p = " 2*ttail(e(df_r), abs(_b[MSCI_`mechanism']/_se[MSCI_`mechanism'])) ")"
            }
            else {
                display as error "All regression approaches failed. Error code: `_rc'"
            }
        }
        else {
            display "Created firm dummies. Running OLS with firm fixed effects..."
            
            capture reg `spec' firm_* if `post_sample', vce(cluster `cluster')
            
            if _rc == 0 {
                display "Successfully ran OLS with firm fixed effects (via dummies)."
                
                // Store results if requested
                if "`store'" != "" {
                    eststo `store'
                    display "Results stored as `store'"
                }
                
                // Report key coefficients
                display _newline "MECHANISM ANALYSIS RESULTS"
                display "============================="
                display "Outcome: `varlist'"
                display "Mechanism: `mechanism'"
                display "Direct effect of MSCI inclusion: " _b[MSCI_clean] " (p = " 2*ttail(e(df_r), abs(_b[MSCI_clean]/_se[MSCI_clean])) ")"
                display "Direct effect of mechanism: " _b[`mechanism'] " (p = " 2*ttail(e(df_r), abs(_b[`mechanism']/_se[`mechanism'])) ")"
                display "Interaction effect: " _b[MSCI_`mechanism'] " (p = " 2*ttail(e(df_r), abs(_b[MSCI_`mechanism']/_se[MSCI_`mechanism'])) ")"
            }
            else {
                display as error "Regression with firm dummies failed. Error code: `_rc'"
                display "Trying without firm fixed effects..."
                
                capture reg `spec' if `post_sample', vce(cluster `cluster')
                
                if _rc == 0 {
                    display "Successfully ran OLS without firm fixed effects."
                    
                    // Store results if requested
                    if "`store'" != "" {
                        eststo `store'
                        display "Results stored as `store' (without firm FE)"
                    }
                    
                    // Report key coefficients
                    display _newline "MECHANISM ANALYSIS RESULTS (WITHOUT FIRM FE)"
                    display "============================================="
                    display "Outcome: `varlist'"
                    display "Mechanism: `mechanism'"
                    display "Direct effect of MSCI inclusion: " _b[MSCI_clean] " (p = " 2*ttail(e(df_r), abs(_b[MSCI_clean]/_se[MSCI_clean])) ")"
                    display "Direct effect of mechanism: " _b[`mechanism'] " (p = " 2*ttail(e(df_r), abs(_b[`mechanism']/_se[`mechanism'])) ")"
                    display "Interaction effect: " _b[MSCI_`mechanism'] " (p = " 2*ttail(e(df_r), abs(_b[MSCI_`mechanism']/_se[MSCI_`mechanism'])) ")"
                }
                else {
                    display as error "All regression approaches failed. Error code: `_rc'"
                }
            }
        }
    }
    
    // Clean up temporary variables
    drop MSCI_`mechanism' `post_sample'
    capture drop firm_*
    
    display "Mechanism analysis complete."
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

// Function to export mechanism analysis results
capture program drop ExportMechanismTable
program define ExportMechanismTable
    syntax namelist [, title(string) vars(namelist) filename(string)]
    
    // Default title
    if "`title'" == "" {
        local title "Mechanism Analysis Results"
    }
    
    // Default filename 
    if "`filename'" == "" {
        local filename "mechanism_analysis"
    }
    
    // Create a table with specified options
    esttab `namelist' using "${results}/tables/`filename'.rtf", ///
        title("`title'") ///
        mtitles ///
        b(4) se(4) star(* 0.10 ** 0.05 *** 0.01) ///
        drop(_cons *year*) ///
        stats(N r2, fmt(%9.0g %9.3f) labels("Observations" "R-squared")) ///
        label replace
    
    display "Mechanism analysis results saved to ${results}/tables/`filename'.rtf"
end

// Function to run alternative models for MSCI effect on digital transformation
capture program drop RunAlternativeModels
program define RunAlternativeModels
    syntax [varlist(default=none)] [, controls(string) save]
    
    display _newline "==========================================================================="
    display "             ALTERNATIVE MODELS FOR MSCI EFFECT ESTIMATION                      "
    display "==========================================================================="
    display "Running alternative identification strategies due to limitations of standard DiD:"
    display "Note: Treat=1 observations only exist in the post-treatment period."
    
    // Set default varlist if none provided
    if "`varlist'" == "" {
        local varlist "Digital_transformationA Digital_transformationB"
        display "Using default outcome variables: `varlist'"
    }
    
    // Use default controls if none specified
    if "`controls'" == "" {
        // Check if control_vars global exists
        if "$control_vars" != "" {
            local controls $control_vars
            display "Using global control variables: `controls'"
        }
        else {
            // Fallback to basic controls
            local controls "age F050501B F060101B"
            display "Using basic controls: `controls'"
        }
    }
    
    // Ensure MSCI_clean exists
    capture confirm variable MSCI_clean
    if _rc != 0 {
        display "Creating MSCI_clean variable..."
        capture gen MSCI_clean = (MSCI==1) if !missing(MSCI)
        label var MSCI_clean "MSCI inclusion (clean binary indicator)"
    }
    
    // Clear any previous estimation results
    eststo clear
    
    // Loop through outcome variables
    foreach dv of local varlist {
        display _newline "--- Running Alternative Models for `dv' ---"
        
        // 1. Direct Fixed Effects Model (MSCI effect directly)
        display _newline "Model 1: Direct Fixed Effects Model"
        capture {
            qui xtreg `dv' MSCI_clean `controls' i.year, fe cluster(stkcd)
            eststo model1
            estadd local modeltype "Direct FE"
            estadd local controls "Yes"
            estadd local yearfe "Yes"
            estadd local sample "All"
            display "  Results: MSCI_clean coef = " _b[MSCI_clean] ", p-value = " 2*ttail(e(df_r), abs(_b[MSCI_clean]/_se[MSCI_clean]))
        }
        if _rc != 0 {
            display as error "  Direct FE model estimation failed for `dv'"
        }
        
        // 2. Post-Period Subsample (MSCI effect in Post period only)
        display _newline "Model 2: Post-Period Subsample"
        preserve
        capture {
            qui keep if Post == 1
            qui xtreg `dv' MSCI_clean `controls' i.year, fe cluster(stkcd)
            eststo model2
            estadd local modeltype "Post-Period"
            estadd local controls "Yes"
            estadd local yearfe "Yes"
            estadd local sample "Post"
            display "  Results: MSCI_clean coef = " _b[MSCI_clean] ", p-value = " 2*ttail(e(df_r), abs(_b[MSCI_clean]/_se[MSCI_clean]))
            restore
        }
        if _rc != 0 {
            display as error "  Post-Period model estimation failed for `dv'"
            restore
        }
        
        // 3. Matched Sample Approach
        display _newline "Model 3: Matched Sample Approach"
        capture {
            preserve
            qui keep if Post == 1
            
            // Try matching if psmatch2 is available
            capture which psmatch2
            if _rc == 0 {
                // Create matched sample based on propensity scores
                qui logit MSCI_clean `controls'
                qui predict pscore if e(sample)
                qui psmatch2 MSCI_clean, pscore(pscore) caliper(0.05) common
                qui generate matched = _support == 1
                
                // Estimate effect only on matched sample
                qui xtreg `dv' MSCI_clean `controls' i.year if matched == 1, fe cluster(stkcd)
                eststo model3
                estadd local modeltype "Matched"
                estadd local controls "Yes"
                estadd local yearfe "Yes"
                estadd local sample "Matched"
                display "  Results: MSCI_clean coef = " _b[MSCI_clean] ", p-value = " 2*ttail(e(df_r), abs(_b[MSCI_clean]/_se[MSCI_clean]))
                
                // Clean up
                qui drop pscore matched _*
            }
            else {
                // Simple alternative if psmatch2 not available
                display "  psmatch2 not available, using simple treatment group comparison"
                qui gen TreatGroup = MSCI_clean
                
                qui reg `dv' TreatGroup `controls' i.year, cluster(stkcd)
                eststo model3
                estadd local modeltype "Simple Match"
                estadd local controls "Yes"
                estadd local yearfe "Yes"
                estadd local sample "Post"
                display "  Results: TreatGroup coef = " _b[TreatGroup] ", p-value = " 2*ttail(e(df_r), abs(_b[TreatGroup]/_se[TreatGroup]))
                
                // Clean up
                qui drop TreatGroup
            }
            restore
        }
        if _rc != 0 {
            display as error "  Matched sample model estimation failed for `dv'"
            capture restore
        }
        
        // 4. First Differences Model
        display _newline "Model 4: First Differences Model (Post-Period)"
        capture {
            preserve
            qui keep if Post == 1
            
            // Generate lagged dependent variable (if possible)
            sort stkcd year
            by stkcd: gen lag_`dv' = `dv'[_n-1]
            by stkcd: gen diff_`dv' = `dv' - lag_`dv'
            
            // Estimate first differences model
            qui reg diff_`dv' MSCI_clean `controls' i.year, cluster(stkcd)
            eststo model4
            estadd local modeltype "First Diff"
            estadd local controls "Yes"
            estadd local yearfe "Yes"
            estadd local sample "Post"
            display "  Results: MSCI_clean coef = " _b[MSCI_clean] ", p-value = " 2*ttail(e(df_r), abs(_b[MSCI_clean]/_se[MSCI_clean]))
            restore
        }
        if _rc != 0 {
            display as error "  First Differences model estimation failed for `dv'"
            capture restore
        }
    }
    
    // Output summary table with results
    display _newline "==========================================================================="
    display "                        ALTERNATIVE MODELS SUMMARY                             "
    display "==========================================================================="
    
    // Display results in a simple text table
    esttab model*, b(3) se(3) star(* 0.1 ** 0.05 *** 0.01) mtitle("Direct FE" "Post-Period" "Matched" "First Diff")
    
    // Export tables if needed
    if "`save'" == "save" {
        esttab model* using "${results}/tables/alternative_models.rtf", replace ///
            b(3) se(3) star(* 0.1 ** 0.05 *** 0.01) ///
            mtitle("Direct FE" "Post-Period" "Matched" "First Diff") ///
            title("Alternative Models for MSCI Effect Estimation") ///
            note("Note: Standard errors in parentheses. * p<0.1, ** p<0.05, *** p<0.01") ///
            scalars("modeltype Model Type" "controls Controls" "yearfe Year FE" "sample Sample")
        
        display "Results exported to ${results}/tables/alternative_models.rtf"
    }
    
    display _newline "Alternative models analysis complete."
end

// Function to run event study analysis
capture program drop RunEventStudy
program define RunEventStudy
    syntax varname, [controls(string) yearfe(integer 1) cluster(string) save(string)]
    
    display _newline "==========================================================================="
    display "                  EVENT STUDY ANALYSIS: DYNAMIC EFFECTS                         "
    display "==========================================================================="
    
    // Check if Event_time variable exists, create if possible
    capture confirm variable Event_time
    if _rc != 0 {
        // Try to create Event_time based on available data
        display "Event_time variable not found, attempting to create..."
        
        // Check if we have the necessary variables to create Event_time
        capture confirm variable Post
        if _rc == 0 {
            // Create a relative time variable (simplistic version)
            // This assumes treatment started at a known year, e.g., 2018
            local treat_year = 2018
            gen Event_time = year - `treat_year'
            label var Event_time "Years relative to treatment (negative=pre, positive=post)"
            display "Created Event_time variable as year - `treat_year'"
        }
        else {
            display as error "Cannot create Event_time variable. Post or year variables missing."
            display "Event study analysis requires a relative time variable."
            exit
        }
    }
    
    // Default controls to global if not specified
    if "`controls'" == "" & "$control_vars" != "" {
        local controls $control_vars
        display "Using global control variables: `controls'"
    }
    
    // Default cluster variable
    if "`cluster'" == "" {
        local cluster "stkcd" 
    }
    
    // Identify the range of Event_time
    sum Event_time
    local min_time = r(min)
    local max_time = r(max)
    
    display "Creating basic event study with time range [`min_time', `max_time']..."
    display "Note: Using -1 as reference period"
    
    // Create a simple pre/post variable instead of full interactions
    gen es_post = (Event_time >= 0)
    label var es_post "Post-treatment (Event_time >= 0)"
    
    // Create post-treatment interaction
    gen treat_post = Treat * es_post
    label var treat_post "Treat × Post-treatment"
    
    // Create simple pre-trend test (using just two pre-periods)
    gen pre_period = (Event_time >= -3 & Event_time < 0)
    gen treat_pre = Treat * pre_period
    label var treat_pre "Treat × Pre-period (-3 to -1)"
    
    // Create simple early/late effects
    gen early_post = (Event_time >= 0 & Event_time <= 2)
    gen late_post = (Event_time > 2)
    gen treat_early = Treat * early_post
    gen treat_late = Treat * late_post
    label var treat_early "Treat × Early Post (0 to 2)"
    label var treat_late "Treat × Late Post (>2)"
    
    // Run a simpler event study that avoids collinearity problems
    display _newline "Running simplified event study regression to avoid collinearity..."
    
    // Build command with careful attention to collinearity
    local cmd "reg `varlist' Treat es_post treat_post"
    
    if "`controls'" != "" {
        local cmd "`cmd' `controls'"
    }
    
    // Avoid using both year and event time dummies together
    if `yearfe' == 1 {
        display "Note: Excluding year fixed effects to avoid collinearity with event time"
    }
    
    if "`cluster'" != "" {
        local cmd "`cmd', cluster(`cluster')"
    }
    
    // Run the regression
    display "Running command: `cmd'"
    capture `cmd'
    
    if _rc == 0 {
        // Store results if requested
        if "`save'" != "" {
            eststo `save'
            display "Event study results stored as `save'"
        }
        else {
            eststo basic_event_study
            display "Event study results stored as basic_event_study"
        }
        
        // Display summary
        display _newline "SUMMARY OF BASIC EVENT STUDY RESULTS"
        display "======================================"
        display "Coefficient on Treat × Post: " _b[treat_post]
        display "Standard error: " _se[treat_post]
        display "t-statistic: " _b[treat_post]/_se[treat_post]
        display "p-value: " 2*ttail(e(df_r), abs(_b[treat_post]/_se[treat_post]))
    }
    else if _rc == 452 {
        display as error "Error: Collinearity detected in event study regression"
        display as error "Trying an even simpler specification..."
        
        // Try an even simpler specification - just basic DiD with some treated observations
        capture qui reg `varlist' Treat es_post treat_post if !missing(`varlist'), cluster(`cluster')
        
        if _rc == 0 {
            display "Simple DiD model executed successfully"
            if "`save'" != "" {
                eststo `save'
                display "Simple DiD results stored as `save'"
            }
            else {
                eststo basic_event_study
                display "Simple DiD results stored as basic_event_study"
            }
        }
        else {
            display as error "Even simplified model failed. Error code: `_rc'"
            exit _rc
        }
    }
    else {
        display as error "Event study regression failed with error code: `_rc'"
        exit _rc
    }
    
    // Now try pre-trends test as a separate regression
    capture {
        display _newline "Running pre-trends test..."
        qui reg `varlist' Treat pre_period treat_pre es_post treat_post if !missing(`varlist'), cluster(`cluster')
        
        if _rc == 0 {
            display "Pre-trends test results:"
            display "Coefficient on Treat × Pre-period: " _b[treat_pre]
            display "Standard error: " _se[treat_pre]
            display "t-statistic: " _b[treat_pre]/_se[treat_pre]
            display "p-value: " 2*ttail(e(df_r), abs(_b[treat_pre]/_se[treat_pre]))
            
            if (2*ttail(e(df_r), abs(_b[treat_pre]/_se[treat_pre])) > 0.1) {
                display "✓ Parallel trends assumption supported (p > 0.1)"
            }
            else {
                display "⚠ Possible violation of parallel trends assumption (p < 0.1)"
            }
        }
    }
    
    // Try dynamic effects as a separate regression
    capture {
        display _newline "Running dynamic effects test..."
        qui reg `varlist' Treat early_post treat_early late_post treat_late if !missing(`varlist'), cluster(`cluster')
        
        if _rc == 0 {
            display "Dynamic effects results:"
            display "Early effect (0-2 years): " _b[treat_early]
            display "  p-value: " 2*ttail(e(df_r), abs(_b[treat_early]/_se[treat_early]))
            display "Late effect (>2 years): " _b[treat_late]
            display "  p-value: " 2*ttail(e(df_r), abs(_b[treat_late]/_se[treat_late]))
        }
    }
    
    // Clean up temporary variables if not needed for further analysis
    drop es_post treat_post pre_period treat_pre early_post late_post treat_early treat_late
    
    // Set flag to indicate event study was run
    global event_study_run = 1
    
    display _newline "Simplified event study analysis complete."
end

// Function to create event study plot
capture program drop CreateEventStudyPlot
program define CreateEventStudyPlot
    syntax [name(name=ename)] [, filename(string) title(string)]
    
    // Default name
    if "`ename'" == "" {
        local ename "event_study"
    }
    
    // Default title
    if "`title'" == "" {
        local title "Event Study: Dynamic Treatment Effects"
    }
    
    // Default filename
    if "`filename'" == "" {
        local filename "event_study_plot" 
    }
    
    // Attempt to restore the estimates
    capture est restore `ename'
    if _rc != 0 {
        display as error "Error: Estimate `ename' not found."
        display "Run RunEventStudy first or specify correct estimate name."
        exit
    }
    
    // Create a simple graph
    display "Creating event study plot..."
    
    // Get all coefficients that start with Treat_time
    local coefs ""
    local times ""
    
    foreach param in `: e(b)' {
        if regexm("`param'", "^Treat_time_(-?[0-9]+)$") {
            local t_val = regexs(1)
            if "`t_val'" != "-1" {
                local t_val_num = real("`t_val'")
                local coefs "`coefs' _b[`param']"
                local times "`times' `t_val_num'"
            }
        }
    }
    
    // Save results to text file for plotting
    tempfile event_results
    file open event_file using "`event_results'", write replace
    file write event_file "time,coef,se,ci_lower,ci_upper" _n
    
    foreach param of e(b) {
        if regexm("`param'", "^Treat_time_(-?[0-9]+)$") {
            local t_val = regexs(1)
            if "`t_val'" != "-1" {
                local coef = _b[`param']
                local se = _se[`param']
                local ci_l = `coef' - 1.96*`se'
                local ci_u = `coef' + 1.96*`se'
                file write event_file "`t_val',`coef',`se',`ci_l',`ci_u'" _n
            }
        }
    }
    file close event_file
    
    // Create a simple plot using Stata's native graphing capabilities
    capture insheet using "`event_results'", clear comma
    
    if _rc == 0 {
        sort time
        gen t_axis = time
        graph twoway (connected coef time, lcolor(blue) mcolor(blue)) ///
            (rcap ci_upper ci_lower time, lcolor(blue%30)), ///
            xline(0, lcolor(red) lpattern(dash)) ///
            yline(0, lcolor(black) lpattern(dash)) ///
            xlabel(`min_time'(1)`max_time') ///
            xtitle("Time Relative to Treatment") ///
            ytitle("Treatment Effect") ///
            title("`title'") ///
            name(event_study_graph, replace) ///
            note("Note: 95% confidence intervals shown. Reference period is t-1.")
            
        // Save the graph
        graph export "${results}/figures/`filename'.png", replace width(1200) height(900)
        display "Event study plot saved to ${results}/figures/`filename'.png"
    }
    else {
        display as error "Error creating graph. Could not read results file."
    }
    
    // Reload the original dataset
    ReloadOriginalData
end

// Function to run placebo test
capture program drop RunPlaceboTest
program define RunPlaceboTest
    syntax varname, placebo_year(integer) [controls(string) yearfe(integer 1) cluster(string) save(string)]
    
    display _newline "==========================================================================="
    display "                        PLACEBO TEST ANALYSIS                                   "
    display "==========================================================================="
    display "Running placebo test with assumed treatment in `placebo_year'..."
    
    // Preserve the current dataset state
    preserve
    
    // Default controls to global if not specified
    if "`controls'" == "" & "$control_vars" != "" {
        local controls $control_vars
        display "Using global control variables: `controls'"
    }
    
    // Default cluster variable
    if "`cluster'" == "" {
        local cluster "stkcd" 
    }
    
    // Create placebo Post variable
    gen Placebo_Post = (year >= `placebo_year')
    label var Placebo_Post "Placebo post-period (year >= `placebo_year')"
    
    // Create placebo Treat variable (randomized assignment to ~5% of control firms)
    gen Placebo_Treat = 0
    label var Placebo_Treat "Placebo treatment indicator"
    
    // Count how many unique firms we have (using built-in Stata commands)
    // First collapse to firm level to count unique firms
    tempfile main_data
    save `main_data', replace
    
    keep if Treat == 0
    by stkcd: keep if _n == 1
    count
    local num_control = r(N)
    local num_placebo = round(`num_control' * 0.05)
    display "Randomly assigning `num_placebo' of `num_control' control firms to placebo treatment"
    
    // Set seed for reproducibility
    set seed 123456
    
    // Generate a random number for each control firm
    gen random_num = runiform()
    
    // Sort and mark top 5% for treatment
    sort random_num
    gen is_placebo = _n <= `num_placebo'
    
    // Keep only firm ID and placebo marker
    keep stkcd is_placebo
    tempfile placebo_firms
    save `placebo_firms', replace
    
    // Reload main dataset and merge
    use `main_data', clear
    merge m:1 stkcd using `placebo_firms', keep(master match) nogenerate
    
    // Assign placebo treatment
    replace Placebo_Treat = is_placebo if is_placebo == 1
    drop is_placebo
    
    // Create placebo interaction
    gen Placebo_TreatPost = Placebo_Treat * Placebo_Post
    label var Placebo_TreatPost "Placebo interaction (Placebo_Treat × Placebo_Post)"
    
    // Run the placebo regression
    display _newline "Running placebo regression..."
    
    // Construct command based on options
    local cmd "reg `varlist' Placebo_Treat Placebo_Post Placebo_TreatPost"
    if "`controls'" != "" {
        local cmd "`cmd' `controls'"
    }
    if `yearfe' == 1 {
        local cmd "`cmd' i.year"
    }
    if "`cluster'" != "" {
        local cmd "`cmd', cluster(`cluster')"
    }
    
    // Run the regression
    display "Running command: `cmd'"
    capture `cmd'
    
    if _rc == 0 {
        // Store results if requested
        if "`save'" != "" {
            eststo `save'
            display "Placebo test results stored as `save'"
        }
        else {
            eststo placebo_test
            display "Placebo test results stored as placebo_test"
        }
        
        // Display placebo effect
        display _newline "PLACEBO TEST RESULTS"
        display "======================="
        display "Placebo treatment effect: " _b[Placebo_TreatPost] " (p-value: " 2*ttail(e(df_r), abs(_b[Placebo_TreatPost]/_se[Placebo_TreatPost])) ")"
        
        if (2*ttail(e(df_r), abs(_b[Placebo_TreatPost]/_se[Placebo_TreatPost])) > 0.1) {
            display "✓ Placebo test passed: No significant effect found (p > 0.1)"
        }
        else {
            display "✗ Placebo test failed: Significant effect found (p < 0.1)"
            display "  This suggests potential issues with the research design."
        }
    }
    else {
        display as error "Placebo regression failed with error code: `_rc'"
    }
    
    // Restore the original dataset
    restore
    
    // Set flag to indicate placebo test was run
    global placebo_run = 1
    
    display _newline "Placebo test complete."
end
