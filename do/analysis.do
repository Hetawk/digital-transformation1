/******************************************************************************
* MAIN ANALYSIS: Hypothesis testing and main estimations
******************************************************************************/

display _newline "FORMAL HYPOTHESIS TESTING"
display "========================="
display "H1: Capital market liberalization (MSCI inclusion) positively influences digital transformation"
display "H2: MSCI inclusion leads to increased adoption of specific digital technologies and platforms"

// Test H1: Overall effect on digital transformation (Pooled OLS DiD)
RunDidModel Digital_transformationA, treatment(Treat) post(Post) interact(TreatPost) controls($control_vars) yearfe(1) cluster(stkcd) save(h1)

// Test H2: Effect on alternative measure (Pooled OLS DiD)
RunDidModel Digital_transformationB, treatment(Treat) post(Post) interact(TreatPost) controls($control_vars) yearfe(1) cluster(stkcd) save(h2)

// Save hypothesis testing results
SaveHypothesisResults h1 h2

// Run descriptive statistics
RunDescriptiveStats $dt_measures $control_vars, by(Treat)

// Create variable definitions
CreateVarDefs

// Main DiD models (Pooled OLS DiD with caution note)
display _newline "MAIN ANALYSIS - POOLED OLS DID (CAUTION ADVISED)"
display "================================================"
eststo clear

// Basic Pooled OLS DiD
eststo m1: qui reg Digital_transformationA Treat Post TreatPost, cluster(stkcd)
estadd local fixedeffects "No"; estadd local yearfe "No"; estadd local controls "No"; estadd local cluster "Firm"

// With controls
eststo m2: qui reg Digital_transformationA Treat Post TreatPost $control_vars, cluster(stkcd)
estadd local fixedeffects "No"; estadd local yearfe "No"; estadd local controls "Yes"; estadd local cluster "Firm"

// With controls and year fixed effects
eststo m3: qui reg Digital_transformationA Treat Post TreatPost $control_vars i.year, cluster(stkcd)
estadd local fixedeffects "No"; estadd local yearfe "Yes"; estadd local controls "Yes"; estadd local cluster "Firm"

// Alternative DV
eststo m4: qui reg Digital_transformationB Treat Post TreatPost $control_vars i.year, cluster(stkcd)
estadd local fixedeffects "No"; estadd local yearfe "Yes"; estadd local controls "Yes"; estadd local cluster "Firm"

// Export results
capture est restore m1
if _rc == 0 {
    ExportRegTable m1 m2 m3 m4, title("The Effect of MSCI Inclusion on Digital Transformation (Pooled OLS DiD - CAUTION ADVISED)") ///
        mtitles("Basic DiD" "With Controls" "Controls + Year FE" "Alt DV + Controls + Year FE") ///
        keep(TreatPost Treat Post _cons $control_vars) ///
        order(TreatPost Treat Post $control_vars _cons) ///
        filename("did_models_pooled_ols")
}

// Run alternative models (more reliable approaches)
RunAlternativeModels

// Run event study if Event_time exists
RunEventStudy Digital_transformationA, controls($control_vars)

// Run placebo test
RunPlaceboTest Digital_transformationA, placebo_year(2015) controls($control_vars)
