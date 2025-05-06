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
RunIndustryAnalysis Digital_transformationA, controls($control_vars)

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
RunSectorHeterogeneity Digital_transformationA, controls($control_vars)

// --- Digital Transformation Components Analysis ---
display _newline "DIGITAL TRANSFORMATION COMPONENTS ANALYSIS"
display "================================================="
RunComponentsAnalysis Digital_transformationA, controls($control_vars)
