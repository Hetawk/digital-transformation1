/******************************************************************************
* SETUP FILE: Global settings and initialization
******************************************************************************/

// Memory and matrix settings
set mem 500m 
set matsize 800

// Define key variable groups using local macros
// Main outcome variables
global dt_measures "Digital_transformationA Digital_transformationB Digital_transformation_rapidA Digital_transformation_rapidB"

// Control variables
global control_vars "age TFP_OP SA_index WW_index F050501B F060101B"

// Firm characteristics 
global firm_chars "HHI_A_ ESG_Score_mean"

// Mechanism variables
global financial_access_vars "SA_index WW_index"
global corp_gov_vars "Top3DirectorSumSalary2 DirectorHoldSum2 DirectorUnpaidNo2"
global investor_scrutiny_vars "ESG_Score_mean"

// Additional file formats
global table_formats "rtf tex csv"

// Model specification types
global model_types "direct_fe post_period matched first_diff"

// Display variable group settings
display "Variable groups defined:"
display "- Outcome variables: $dt_measures"
display "- Control variables: $control_vars"
display "- Mechanism variables defined for financial access, corporate governance, and investor scrutiny"
