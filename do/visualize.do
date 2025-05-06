/******************************************************************************
* VISUALIZATION: Functions for creating all plots and visualizations
******************************************************************************/

// Reload original data if needed
ReloadOriginalData

// --- Digital Transformation Trends Visualization ---
display _newline "Creating visualization for Digital Transformation trends..."
CreateTrendsPlot Digital_transformationA, by(Treat) filename("dt_trends_by_treatment")

// --- Event Study Visualization ---
if "`r(event_study_run)'" == "1" {
    display _newline "Creating Event Study visualization..."
    CreateEventStudyPlot event_study, filename("event_study_plot")
}

// --- Alternative Models Comparison ---
display _newline "Creating Alternative Models Comparison Plot..."
CreateAltModelsPlot, filename("alternative_models_comparison")

// --- Placebo Test Visualization ---
if "`r(placebo_run)'" == "1" {
    display _newline "Creating Placebo Test Comparison..."
    CreatePlaceboPlot, filename("placebo_comparison")
}

// --- Heterogeneity Visualizations ---
capture confirm variable Large_firm
if _rc == 0 {
    display _newline "Creating Size Heterogeneity visualization..."
    // Margins-based visualization for size heterogeneity
    capture {
        qui xtreg Digital_transformationA c.MSCI_clean##i.Large_firm $control_vars i.year if Post == 1, fe cluster(stkcd)
        margins Large_firm, at(MSCI_clean=(0 1)) vsquish
        marginsplot, recast(bar) ///
            title("Digital Transformation by Firm Size and MSCI Status", size(medium)) ///
            subtitle("(Post-Period Analysis)", size(small)) ///
            ytitle("Predicted Digital Transformation (A)", size(medium)) ///
            note("Note: Based on FE model. Bars show predicted values by firm size and MSCI status.", size(small)) ///
            graphregion(color(white)) bgcolor(white) scheme(s1color)
        graph export "${results}/figures/heterogeneity_size_plot.png", replace width(1200) height(900)
    }
}
