/******************************************************************************
* VISUALIZATION: Functions for creating all plots and visualizations
******************************************************************************/

// Reload original data if needed
ReloadOriginalData

// --- Visualization Programs ---

// Define programs for visualizations before using them
capture program drop CreateTrendsPlot
program define CreateTrendsPlot
    syntax varname, [by(varname) filename(string) format(string) title(string)]
    
    // Set defaults
    if "`format'" == "" local format "png"
    if "`filename'" == "" local filename = "`varlist'_trends"
    if "`title'" == "" local title "Trends in `varlist'"
    
    // Setup
    preserve
    
    // Calculate means by year and specified group
    if "`by'" != "" {
        collapse (mean) `varlist', by(year `by')
        
        // Create the graph
        graph twoway (connected `varlist' year if `by' == 0, lcolor(navy) lwidth(medium) mcolor(navy) msymbol(circle)) ///
                    (connected `varlist' year if `by' == 1, lcolor(cranberry) lwidth(medium) mcolor(cranberry) msymbol(square)), ///
                    title("`title'") ///
                    xlabel(2010(2)2023, angle(45)) ylabel(, format(%9.2f)) ///
                    xtitle("Year") ytitle("`varlist'") ///
                    xline(2018, lcolor(red) lpattern(dash)) ///
                    legend(order(1 "Control" 2 "Treated") cols(2)) ///
                    graphregion(color(white)) bgcolor(white)
    }
    else {
        collapse (mean) `varlist', by(year)
        
        // Create the graph
        graph twoway (connected `varlist' year, lcolor(navy) lwidth(medium) mcolor(navy) msymbol(circle)), ///
                    title("`title'") ///
                    xlabel(2010(2)2023, angle(45)) ylabel(, format(%9.2f)) ///
                    xtitle("Year") ytitle("`varlist'") ///
                    xline(2018, lcolor(red) lpattern(dash)) ///
                    graphregion(color(white)) bgcolor(white)
    }
    
    // Export the graph
    graph export "${results}/figures/`filename'.`format'", replace width(1000)
    display "Trend plot for `varlist' saved to ${results}/figures/`filename'.`format'"
    
    restore
end

// Program for creating treatment effect plots
capture program drop CreateTreatmentEffectPlot
program define CreateTreatmentEffectPlot
    syntax varname, [filename(string) format(string) title(string)]
    
    // Set defaults
    if "`format'" == "" local format "png"
    if "`filename'" == "" local filename = "`varlist'_treatment_effect"
    if "`title'" == "" local title "Treatment Effect on `varlist'"
    
    // Setup
    preserve
    
    // Calculate pre- and post-treatment means for treated and control firms
    collapse (mean) `varlist', by(Treat Post)
    
    // Reshape data to wide format for easier graphing
    reshape wide `varlist', i(Post) j(Treat)
    rename `varlist'0 Control
    rename `varlist'1 Treated
    
    // Check if we have values for treated firms pre-treatment
    // (staggered adoption may mean we don't have pre-treatment treated observations)
    capture assert !missing(Treated) if Post == 0
    local has_pre_treated = (_rc == 0)
    
    // Create the graph
    if `has_pre_treated' {
        // Standard DiD visualization
        graph bar Control Treated, over(Post) ///
            title("`title'") ///
            ylabel(, format(%9.2f)) ///
            ytitle("`varlist'") ///
            bargap(10) ///
            legend(order(1 "Control" 2 "Treated") cols(2)) ///
            graphregion(color(white)) bgcolor(white) ///
            asyvars
    }
    else {
        // Post-only comparison for staggered adoption
        keep if Post == 1
        graph bar Control Treated, ///
            title("`title' (Post Period Only)") ///
            ylabel(, format(%9.2f)) ///
            ytitle("`varlist'") ///
            bargap(50) ///
            legend(order(1 "Control" 2 "Treated") cols(2)) ///
            graphregion(color(white)) bgcolor(white) ///
            asyvars
    }
    
    // Export the graph
    graph export "${results}/figures/`filename'.`format'", replace width(1000)
    display "Treatment effect plot for `varlist' saved to ${results}/figures/`filename'.`format'"
    
    restore
end

// Program for creating coefficient plots
capture program drop CreateCoefPlot
program define CreateCoefPlot
    syntax anything, [filename(string) format(string) title(string)]
    
    // Set defaults
    if "`format'" == "" local format "png"
    if "`filename'" == "" local filename = "coef_plot"
    if "`title'" == "" local title "Coefficient Plot"
    
    // Create the coefficient plot
    coefplot `anything', ///
        vertical ///
        keep(MSCI_clean 1.Large_firm#c.MSCI_clean c.MSCI_clean#c.SA_index c.MSCI_clean#c.WW_index c.MSCI_clean#c.ESG_Score_mean) ///
        coeflabels(MSCI_clean "MSCI Inclusion" ///
                  1.Large_firm#c.MSCI_clean "MSCI × Large Firm" ///
                  c.MSCI_clean#c.SA_index "MSCI × SA Index" ///
                  c.MSCI_clean#c.WW_index "MSCI × WW Index" ///
                  c.MSCI_clean#c.ESG_Score_mean "MSCI × ESG Score") ///
        xline(0, lcolor(red) lpattern(dash)) ///
        title("`title'") ///
        graphregion(color(white)) bgcolor(white) ///
        xlabel(, format(%9.2f))
    
    // Export the graph
    graph export "${results}/figures/`filename'.`format'", replace width(1000)
    display "Coefficient plot saved to ${results}/figures/`filename'.`format'"
end

// Program for creating component comparison plots
capture program drop CreateComponentPlot
program define CreateComponentPlot
    syntax [anything], [filename(string) format(string) title(string)]
    
    // Set defaults
    if "`format'" == "" local format "png"
    if "`filename'" == "" local filename = "dt_components"
    if "`title'" == "" local title "MSCI Effect on Digital Transformation Components"
    
    // Setup
    preserve
    
    // Create a temporary dataset with the effect sizes
    clear
    set obs 4
    
    // Create numeric ID for plotting (IMPORTANT: use this instead of string variable for coordinates)
    gen comp_id = _n
    
    // Store component names as strings (for labeling only)
    gen component = ""
    replace component = "Digital_transformationA" in 1
    replace component = "Digital_transformationB" in 2
    replace component = "Digital_transformation_rapidA" in 3
    replace component = "Digital_transformation_rapidB" in 4
    
    // These values come from the component analysis
    gen effect = .
    replace effect = 0.022 in 1  // DT Type A
    replace effect = 0.003 in 2  // DT Type B
    replace effect = 0.047 in 3  // DT Rapid A
    replace effect = 0.014 in 4  // DT Rapid B
    
    gen se = .
    replace se = 0.052 in 1
    replace se = 0.042 in 2
    replace se = 0.053 in 3
    replace se = 0.020 in 4
    
    // Calculate confidence intervals
    gen ci_low = effect - 1.96*se
    gen ci_high = effect + 1.96*se
    
    // Create more readable labels for display
    gen label = ""
    replace label = "DT Type A" in 1
    replace label = "DT Type B" in 2
    replace label = "DT Speed A" in 3
    replace label = "DT Speed B" in 4
    
    // Apply the labels to the comp_id variable
    label define comp_lbl 1 "DT Type A" 2 "DT Type B" 3 "DT Speed A" 4 "DT Speed B"
    label values comp_id comp_lbl
    
    // Create the graph using numeric comp_id instead of string component
    graph twoway (scatter effect comp_id, mcolor(navy) msymbol(circle)) ///
                (rcap ci_low ci_high comp_id, lcolor(navy)), ///
                title("`title'") ///
                ylabel(, format(%9.2f)) ///
                xlabel(1(1)4, valuelabel angle(45)) ///
                xtitle("") ytitle("Effect of MSCI Inclusion") ///
                graphregion(color(white)) bgcolor(white) ///
                legend(off) ///
                yline(0, lcolor(red) lpattern(dash))
                
    // Export the graph
    graph export "${results}/figures/`filename'.`format'", replace width(1000)
    display "Component plot saved to ${results}/figures/`filename'.`format'"
    
    restore
end

// --- Digital Transformation Trends Visualization ---
display _newline "Creating visualization for Digital Transformation trends..."

// Create directory for figures if it doesn't exist
capture mkdir "${results}/figures"

// Create trend plots for both key DT measures
foreach var in Digital_transformationA Digital_transformationB {
    CreateTrendsPlot `var', by(Treat) filename("`var'_trends_by_treatment")
}

// --- Treatment Effect Visualization ---
display _newline "Creating treatment effect visualizations..."

// Create treatment effect plots
foreach var in Digital_transformationA Digital_transformationB {
    CreateTreatmentEffectPlot `var', filename("`var'_treatment_effect")
}

// --- Components Visualization ---
display _newline "Creating digital transformation components visualization..."

// Create component plot
CreateComponentPlot, filename("dt_components_comparison")

// --- Heterogeneity Visualization ---
display _newline "Creating heterogeneity visualizations..."

// Load stored estimates for coefficient plot if they exist
capture est restore het_size
if _rc == 0 {
    // Create coefficient plot for size heterogeneity
    CreateCoefPlot het_size, filename("heterogeneity_size_coefplot") title("Heterogeneity by Firm Size")
}
else {
    display as error "Size heterogeneity estimates not found. Skipping coefficient plot."
}

// Look for mechanism results
local mech_found = 0
foreach model in DTA_SA_post_fe DTB_SA_post_fe DTA_WW_post_fe DTB_WW_post_fe DTA_ESG_post_fe DTB_ESG_post_fe {
    capture est restore `model'
    if _rc == 0 {
        local mech_found = 1
    }
}

if `mech_found' {
    display "Creating mechanism coefficient plot..."
    // Create coefficient plot for mechanisms
    CreateCoefPlot DTA_SA_post_fe DTA_WW_post_fe DTA_ESG_post_fe, ///
        filename("mechanism_coefplot") title("Mechanism Effects on Digital Transformation")
}
else {
    display as error "Mechanism estimates not found. Skipping mechanism coefficient plot."
}

display _newline "Visualization complete."
