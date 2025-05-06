# Advanced DiD Methods for Staggered Adoption Designs

## The Challenge with Standard DiD

Our dataset has a staggered adoption structure, where no units are treated in the pre-treatment period. This means:

1. **No pre-treatment treated units**: We cannot conduct conventional balance tests between treatment and control in the pre-period
2. **Standard DiD can be biased**: Under treatment effect heterogeneity, the two-way fixed effects DiD estimator can be biased and even get the sign wrong (Goodman-Bacon, 2021)


## Recommended State-of-the-Art Methods

### 1. Callaway & Sant'Anna (2021) Group-Time Average Treatment Effects

**Advantages:**
- Handles staggered adoption with heterogeneous treatment effects
- Computes group-time average treatment effects for each cohort
- Allows for multiple periods and flexible control groups

**Implementation:**
- R package: 'did'
- Python implementation via 'econtools' package
- Can be manually implemented as a series of 2x2 DiD estimators

**Key Reference:**
- Callaway, B., & Sant'Anna, P. H. (2021). Difference-in-differences with multiple time periods. Journal of Econometrics, 225(2), 200-230.

### 2. Sun & Abraham (2021) Interaction-Weighted Estimator

**Advantages:**
- Robust to treatment effect heterogeneity across cohorts
- Handles dynamic treatment effects
- Can be implemented with standard regression software

**Implementation:**
- Can be implemented manually using cohort-period interactions
- Weights cohort-specific estimates by cohort size

**Key Reference:**
- Sun, L., & Abraham, S. (2021). Estimating dynamic treatment effects in event studies with heterogeneous treatment effects. Journal of Econometrics, 225(2), 175-199.

### 3. de Chaisemartin & D'Haultfœuille (2020) DIDM Estimator

**Advantages:**
- Identifies treatment effects even with heterogeneous effects
- Robust to negative weighting problems in TWFE
- Handles dynamic treatment effects

**Implementation:**
- Stata package: 'did_multiplegt'
- R package: 'DIDmultiplegt'

**Key Reference:**
- de Chaisemartin, C., & D'Haultfœuille, X. (2020). Two-way fixed effects estimators with heterogeneous treatment effects. American Economic Review, 110(9), 2964-96.

### 4. Borusyak, Jaravel & Spiess (2021) Imputation Estimator

**Advantages:**
- Efficient under parallel trends assumption
- Can handle irregular panel data
- Good power properties

**Implementation:**
- Stata package: 'did_imputation'

**Key Reference:**
- Borusyak, K., Jaravel, X., & Spiess, J. (2021). Revisiting event study designs: Robust and efficient estimation. Working Paper.

### 5. Robust Event Study Approach

**Advantages:**
- Intuitive visualization of treatment effects over time
- Can detect pre-trends, anticipation effects, and treatment dynamics
- Works with our existing event study framework with modifications

**Implementation:**
- Proper binning of endpoints (Schmidheiny & Siegloch, 2020)
- Inclusion of never-treated units as control group
- Careful interpretation of coefficients

**Key Reference:**
- Schmidheiny, K., & Siegloch, S. (2020). On event studies and distributed-lags in two-way fixed effects models: Identification, equivalence, and generalization. ZEW-Centre for European Economic Research Discussion Paper, (20-017).

## Implementation Strategy for Our Project

1. **First Step**: Use our existing post-period analysis + matching approach to establish baseline estimates
2. **Second Step**: Implement the Callaway & Sant'Anna method or robust event study for our main results
3. **Robustness**: Compare results across multiple methods to ensure findings are not driven by methodological choices

By implementing these approaches, we can provide academically rigorous evidence despite the challenges posed by our staggered treatment design.
