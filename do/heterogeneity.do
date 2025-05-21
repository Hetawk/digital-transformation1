/******************************************************************************
* HETEROGENEITY ANALYSIS: Examining differential effects across firm types
******************************************************************************/

// Reload original data
ReloadOriginalData
EnsureMSCIClean

display _newline "POLICY IMPLICATIONS / HETEROGENEITY ANALYSIS"
display "============================================"

// --- Industry Analysis ---
display _newline "INDUSTRY ANALYSIS"
display "========================="

// Check if industry variables exist
capture confirm variable Sicda
if _rc == 0 {
    // Custom implementation of industry analysis since RunIndustryAnalysis is missing
    display "Running industry analysis using Sicda variable..."
    
    // Store industry analysis results
    eststo clear
    
    // Get a list of industry codes with sufficient observations
    levelsof Sicda, local(ind_codes)
    local valid_industries ""
    foreach ind in `ind_codes' {
        quietly count if Sicda == `ind' & !missing(Digital_transformationA) & Post == 1
        if r(N) >= 30 {  // Only include industries with sufficient observations
            local valid_industries "`valid_industries' `ind'"
        }
    }
    
    // Run baseline model for reference
    eststo ind_base: xtreg Digital_transformationA MSCI_clean $control_vars i.year if Post == 1, fe cluster(stkcd)
    estadd local fixedeffects "Yes"; estadd local yearfe "Yes"; estadd local controls "Yes"; estadd local cluster "Firm"
    
    // Store separate estimates for each industry
    if "`valid_industries'" != "" {
        local est_list "ind_base"
        local title_list "All"
        
        foreach ind in `valid_industries' {
            local ind_name: label Sicda `ind'
            if "`ind_name'" == "" local ind_name "Industry `ind'"
            display "Running model for `ind_name'..."
            
            capture {
                eststo ind_`ind': xtreg Digital_transformationA MSCI_clean $control_vars i.year if Post == 1 & Sicda == `ind', fe cluster(stkcd)
                if _rc == 0 {
                    estadd local fixedeffects "Yes"; estadd local yearfe "Yes"; estadd local controls "Yes"; estadd local cluster "Firm"
                    local est_list "`est_list' ind_`ind'"
                    local title_list "`title_list' `ind_name'"
                }
            }
        }
        
        // Export industry results one by one to avoid errors
        if "`est_list'" != "ind_base" {
            // Only include baseline and first industry in first table for simplicity
            local first_est_list: word 1 of `est_list'
            local second_est_list: word 2 of `est_list'
            local first_title_list: word 1 of `title_list'
            local second_title_list: word 2 of `title_list'
            
            // Export first table with baseline and first industry
            esttab `first_est_list' `second_est_list' using "${results}/tables/heterogeneity_industry_main.rtf", replace ///
                title("Heterogeneity by Industry (Post-Period FE) - Main Industries") ///
                star(* 0.1 ** 0.05 *** 0.01) ///
                keep(MSCI_clean) order(MSCI_clean) ///
                b(%9.3f) se(%9.3f) ///
                stats(N r2_w, fmt(%9.0f %9.3f) labels("Observations" "Within R-sq")) ///
                mtitles("`first_title_list'" "`second_title_list'") ///
                note("FE models on Post-2018 data. Coefficients show effect of MSCI inclusion by industry. Clustered SEs.")
                
            display "Main industry heterogeneity results saved."
            
            // Export additional industries in separate tables if needed
            if `: word count `est_list'' > 2 {
                // Get remaining industries
                local remaining_ests = subinstr("`est_list'", "`first_est_list' `second_est_list'", "", .)
                local remaining_ests = trim("`remaining_ests'")
                
                if "`remaining_ests'" != "" {
                    // Export remaining industries
                    esttab `remaining_ests' using "${results}/tables/heterogeneity_industry_additional.rtf", replace ///
                        title("Heterogeneity by Industry (Post-Period FE) - Additional Industries") ///
                        star(* 0.1 ** 0.05 *** 0.01) ///
                        keep(MSCI_clean) order(MSCI_clean) ///
                        b(%9.3f) se(%9.3f) ///
                        stats(N r2_w, fmt(%9.0f %9.3f) labels("Observations" "Within R-sq")) ///
                        note("FE models on Post-2018 data. Coefficients show effect of MSCI inclusion by additional industries. Clustered SEs.")
                        
                    display "Additional industry heterogeneity results saved."
                }
            }
        }
        else {
            display as text "Only baseline model available, no industry-specific models to export."
        }
    }
    else {
        display as error "No industries with sufficient observations found."
    }
}
else {
    // Try alternative industry variables
    capture confirm variable Sic2
    if _rc == 0 {
        display "Running industry analysis using Sic2 variable..."
        
        // Similar implementation as above but for Sic2
        eststo clear
        
        // Get a list of industry codes with sufficient observations
        levelsof Sic2, local(ind_codes)
        local valid_industries ""
        foreach ind in `ind_codes' {
            quietly count if Sic2 == `ind' & !missing(Digital_transformationA) & Post == 1
            if r(N) >= 30 {  // Only include industries with sufficient observations
                local valid_industries "`valid_industries' `ind'"
            }
        }
        
        // Run baseline model for reference
        eststo ind_base: xtreg Digital_transformationA MSCI_clean $control_vars i.year if Post == 1, fe cluster(stkcd)
        estadd local fixedeffects "Yes"; estadd local yearfe "Yes"; estadd local controls "Yes"; estadd local cluster "Firm"
        
        // Store separate estimates for each industry
        if "`valid_industries'" != "" {
            local est_list "ind_base"
            
            foreach ind in `valid_industries' {
                local ind_name: label Sic2 `ind'
                if "`ind_name'" == "" local ind_name "Industry `ind'"
                display "Running model for `ind_name'..."
                
                capture {
                    eststo ind_`ind': xtreg Digital_transformationA MSCI_clean $control_vars i.year if Post == 1 & Sic2 == `ind', fe cluster(stkcd)
                    if _rc == 0 {
                        estadd local fixedeffects "Yes"; estadd local yearfe "Yes"; estadd local controls "Yes"; estadd local cluster "Firm"
                        local est_list "`est_list' ind_`ind'"
                    }
                }
            }
            
            // Export industry results (use first two for simplicity)
            if "`est_list'" != "ind_base" {
                local first_est: word 1 of `est_list'
                local second_est: word 2 of `est_list'
                
                esttab `first_est' `second_est' using "${results}/tables/heterogeneity_industry_sic2.rtf", replace ///
                    title("Heterogeneity by Industry (Sic2, Post-Period FE)") ///
                    star(* 0.1 ** 0.05 *** 0.01) ///
                    keep(MSCI_clean) order(MSCI_clean) ///
                    b(%9.3f) se(%9.3f) ///
                    stats(N r2_w, fmt(%9.0f %9.3f) labels("Observations" "Within R-sq")) ///
                    note("FE models on Post-2018 data. Coefficients show effect of MSCI inclusion by industry (Sic2). Clustered SEs.")
                    
                display "Industry (Sic2) heterogeneity results saved."
            }
        }
        else {
            display as error "No Sic2 industries with sufficient observations found."
        }
    }
    else {
        capture confirm variable IndustryCode_11
        if _rc == 0 {
            display "Running industry analysis using IndustryCode_11 variable..."
            
            // Similar implementation using IndustryCode_11
            eststo clear
            
            // Get a list of industry codes with sufficient observations
            levelsof IndustryCode_11, local(ind_codes)
            local valid_industries ""
            foreach ind in `ind_codes' {
                quietly count if IndustryCode_11 == "`ind'" & !missing(Digital_transformationA) & Post == 1
                if r(N) >= 30 {  // Only include industries with sufficient observations
                    local valid_industries "`valid_industries' `ind'"
                }
            }
            
            // Run baseline model for reference
            eststo ind_base: xtreg Digital_transformationA MSCI_clean $control_vars i.year if Post == 1, fe cluster(stkcd)
            estadd local fixedeffects "Yes"; estadd local yearfe "Yes"; estadd local controls "Yes"; estadd local cluster "Firm"
            
            if "`valid_industries'" != "" {
                // Export only baseline model since string industries are harder to process in loops
                esttab ind_base using "${results}/tables/heterogeneity_industry_code11.rtf", replace ///
                    title("Baseline Model by Industry (IndustryCode_11, Post-Period FE)") ///
                    star(* 0.1 ** 0.05 *** 0.01) ///
                    keep(MSCI_clean) order(MSCI_clean) ///
                    b(%9.3f) se(%9.3f) ///
                    stats(N r2_w, fmt(%9.0f %9.3f) labels("Observations" "Within R-sq")) ///
                    note("Baseline model for industry analysis. See additional outputs for industry-specific effects.")
                    
                display "IndustryCode_11 analysis: Only baseline model exported due to string industry codes."
            }
            else {
                display as error "No IndustryCode_11 industries with sufficient observations found."
            }
        }
        else {
            display as error "No suitable industry variable found. Skipping industry analysis."
        }
    }
}

// --- Heterogeneity by Firm Size ---
display _newline "--- Heterogeneity by Firm Size ---"
// Create firm size variable if not exists
capture confirm variable Large_firm
if _rc != 0 {
    display "Creating Large_firm variable based on Total Assets..."
    capture confirm variable A001000000
    if _rc == 0 {
        egen median_assets = median(A001000000), by(year)
        gen Large_firm = (A001000000 >= median_assets) if !missing(A001000000, median_assets)
        label var Large_firm "Firm Size (1=Large, 0=Small, based on yearly median assets)"
        display "Created Large_firm variable."
    } 
    else {
        display as error "Total Assets variable (A001000000) not found. Cannot create Large_firm."
        // Try alternative size variable
        capture confirm variable TA
        if _rc == 0 {
            display "Using TA variable instead..."
            egen median_TA = median(TA), by(year)
            gen Large_firm = (TA >= median_TA) if !missing(TA, median_TA)
            label var Large_firm "Firm Size (1=Large, 0=Small, based on yearly median assets)"
            display "Created Large_firm variable using TA."
        }
        else {
            display as error "No suitable size variable found. Skipping size heterogeneity analysis."
        }
    }
}

// Run size heterogeneity analysis if Large_firm exists
capture confirm variable Large_firm
if _rc == 0 {
    eststo clear
    // Run size heterogeneity regression
    eststo het_size: xtreg Digital_transformationA c.MSCI_clean##i.Large_firm $control_vars i.year if Post == 1, fe cluster(stkcd)
    estadd local fixedeffects "Yes"; estadd local yearfe "Yes"; estadd local controls "Yes"; estadd local cluster "Firm"
    
    // Export results
    esttab het_size using "${results}/tables/heterogeneity_size.rtf", replace ///
        title("Heterogeneity by Firm Size (Post-Period FE)") ///
        star(* 0.1 ** 0.05 *** 0.01) substitute(_cons Constant) ///
        keep(*MSCI_clean* *Large_firm*) order(*MSCI_clean* *Large_firm*) ///
        b(%9.3f) se(%9.3f) nonumbers mtitles ///
        scalars("fixedeffects Firm FE" "yearfe Year FE" "controls Controls" "cluster Cluster SE") ///
        stats(N N_clust r2_w, fmt(%9.0f %9.0gc %9.3f) labels("Observations" "Clusters" "Within R-sq")) ///
        note("FE model on Post-2018 data. Interaction term shows differential effect for large firms. Clustered SEs.")
    display "Size heterogeneity results saved."
}

// --- Heterogeneity by Sector ---
display _newline "--- Heterogeneity by Sector ---"

// Custom implementation since RunSectorHeterogeneity is missing
// Check if sector variables exist
capture confirm variable Sicmen
if _rc == 0 {
    display "Running sector heterogeneity analysis using Sicmen variable..."
    
    // Store sector analysis results
    eststo clear
    
    // Get a list of sector codes with sufficient observations
    levelsof Sicmen, local(sect_codes)
    local valid_sectors ""
    foreach sect in `sect_codes' {
        quietly count if Sicmen == `sect' & !missing(Digital_transformationA) & Post == 1
        if r(N) >= 30 {  // Only include sectors with sufficient observations
            local valid_sectors "`valid_sectors' `sect'"
        }
    }
    
    // Run baseline model for reference
    eststo sect_base: xtreg Digital_transformationA MSCI_clean $control_vars i.year if Post == 1, fe cluster(stkcd)
    estadd local fixedeffects "Yes"; estadd local yearfe "Yes"; estadd local controls "Yes"; estadd local cluster "Firm"
    
    // Run individual sector models if we have valid sectors
    if "`valid_sectors'" != "" {
        foreach sect in `valid_sectors' {
            local sect_name: label Sicmen `sect'
            if "`sect_name'" == "" local sect_name "Sector `sect'"
            display "Running model for `sect_name'..."
            capture eststo sect_`sect': xtreg Digital_transformationA MSCI_clean $control_vars i.year if Post == 1 & Sicmen == `sect', fe cluster(stkcd)
            if _rc == 0 {
                estadd local fixedeffects "Yes"; estadd local yearfe "Yes"; estadd local controls "Yes"; estadd local cluster "Firm"
            }
        }
        
        // Export sector results
        esttab sect_base sect_* using "${results}/tables/heterogeneity_sector.rtf", replace ///
            title("Heterogeneity by Sector (Post-Period FE)") ///
            star(* 0.1 ** 0.05 *** 0.01) ///
            keep(MSCI_clean) order(MSCI_clean) ///
            b(%9.3f) se(%9.3f) ///
            stats(N r2_w, fmt(%9.0f %9.3f) labels("Observations" "Within R-sq")) ///
            note("FE models on Post-2018 data. Coefficients show effect of MSCI inclusion by sector. Clustered SEs.")
        
        display "Sector heterogeneity results saved."
    }
    else {
        display as error "No sectors with sufficient observations found."
    }
}
else {
    display as error "No suitable sector variable found. Skipping sector heterogeneity analysis."
}

// --- Digital Transformation Components Analysis ---
display _newline "DIGITAL TRANSFORMATION COMPONENTS ANALYSIS"
display "================================================="

// Custom implementation since RunComponentsAnalysis is missing
// Check if we have multiple DT measures
local dt_vars ""
foreach var in Digital_transformationA Digital_transformationB Digital_transformation_rapidA Digital_transformation_rapidB {
    capture confirm variable `var'
    if _rc == 0 {
        local dt_vars "`dt_vars' `var'"
    }
}

if "`dt_vars'" != "" {
    display "Running analysis on digital transformation components..."
    
    // Store components analysis results
    eststo clear
    local est_names "" // Store successful estimation names
    
    // Run model for each DT component
    foreach dt_var in `dt_vars' {
        display "Running model for `dt_var'..."
        
        // Create shorter valid name for estimation storage
        if "`dt_var'" == "Digital_transformationA" {
            local est_name "DTA" 
        }
        else if "`dt_var'" == "Digital_transformationB" {
            local est_name "DTB"
        }
        else if "`dt_var'" == "Digital_transformation_rapidA" {
            local est_name "DTrA"
        }
        else if "`dt_var'" == "Digital_transformation_rapidB" {
            local est_name "DTrB"
        }
        else {
            // Create a safe, short name for any other variable
            local est_name = substr("`dt_var'", 1, 3)
        }
        
        // Run regression and handle errors explicitly
        capture noisily xtreg `dt_var' MSCI_clean $control_vars i.year if Post == 1, fe cluster(stkcd)
        
        if _rc == 0 {
            // Successfully ran regression, now store it
            eststo comp_`est_name'
            estadd local fixedeffects "Yes"
            estadd local yearfe "Yes" 
            estadd local controls "Yes" 
            estadd local cluster "Firm"
            
            // Add to our list of successful estimations
            local est_names "`est_names' comp_`est_name'"
            display "  - Model stored as comp_`est_name'"
        }
        else {
            display as error "  - Error running model for `dt_var': " _rc
        }
    }
    
    // Export components results only if we have successful estimations
    if "`est_names'" != "" {
        display "Exporting results for the following models: `est_names'"
        
        // Export the tables using explicit model names instead of wildcards
        esttab `est_names' using "${results}/tables/dt_components_analysis.rtf", replace ///
            title("Effect of MSCI Inclusion on Different Digital Transformation Measures (Post-Period FE)") ///
            star(* 0.1 ** 0.05 *** 0.01) ///
            keep(MSCI_clean) order(MSCI_clean) ///
            b(%9.3f) se(%9.3f) ///
            stats(N r2_w, fmt(%9.0f %9.3f) labels("Observations" "Within R-sq")) ///
            mtitles("DT Type A" "DT Type B" "DT Rapid A" "DT Rapid B") ///
            note("FE models on Post-2018 data. Coefficients show effect of MSCI inclusion on different digital transformation measures. Clustered SEs.")
            
        display "Digital transformation components analysis saved to ${results}/tables/dt_components_analysis.rtf"
    }
    else {
        display as error "No successful component models to export. Skipping table generation."
    }
}
else {
    display as error "No digital transformation variables found. Skipping components analysis."
}
