# Hooked on Flavor: Addiction, Present Bias, and the Consequences of E-Cigarette Flavor Policy

**William Brasic**  
Department of Economics, University of Arizona  
wbrasic@arizona.edu

---

## Overview

This repository contains the code for [*Hooked on Flavor: Addiction, Present Bias, and the Consequences of E-Cigarette Flavor Policy*](https://drive.google.com/file/d/1QVyke0nyIxgv5jUvWM3OXV3niEwzh9I5/view?usp=drive_link), examining how flavored e-cigarette regulations affect addiction and consumer behavior through a dynamic structural model using Nielsen Homescan (HMS) panel data from 2021-2023. The model features addiction and flavor stocks, heterogeneous flavor preferences by household age composition, and unobserved heterogeneity.

The counterfactual analysis evaluates the addiction and consumer surplus effects of several FDA flavor regulation policies, including comprehensive flavor bans, FDA-unauthorized product restrictions, and per-mL flavor taxes.

---

## Data

The underlying data are from the **Nielsen Homescan (HMS) Consumer Panel** (2021-2023), a household-level panel tracking retail purchases across product categories. The raw data are proprietary and not included in this repository.

---

## Languages and Dependencies

Data cleaning, first-stage estimation, and the reduced-form evidence are in R. The structural estimation, validation, counterfactuals, and model comparison are in Julia. Julia scripts are parallelized and are designed to run on a SLURM HPC cluster. Each subfolder containing a `.sb` file includes the corresponding SLURM submission script.

---

## Repository Structure

```
.
├── Data_Cleaning/
│   ├── Census/                                    # ACS microdata pull (R)
│   └── HMS/
│       └── 2021-Onward/                           # Raw HMS data cleaning (R)
└── Data_Analysis/
    └── HMS/
        ├── Summary_Stats/                         # Descriptive statistics (R)
        ├── Structural_Model_Motivation/           # Reduced-form evidence (R)
        └── Dynamic_Model/                         # Structural estimation (Julia + R)
            ├── 01_First_Stage_Estimation/
            ├── 02_Second_Stage_Estimation_Mixture/
            ├── 02_Second_Stage_Estimation_Mixture_Holdout/
            ├── 02_Second_Stage_Estimation_Mixture_LongRun/
            ├── 02_Second_Stage_Estimation_Mixture_Holdout_LongRun/
            ├── 03_Standard_Errors_Mixture/
            ├── 04_Validation_Mixture/
            ├── 04_Validation_Mixture_Holdout/
            ├── 05_CF_Mixture/
            ├── 05_CF_Mixture_Bootstrap/
            └── 06_Vuong_Test_Mixture/
```

---

## Data Cleaning

### `Data_Cleaning/HMS/2021-Onward/`

Cleans and structures raw HMS scanner data into the household-month panel used throughout the analysis.

| Script | Description |
|--------|-------------|
| `01_Aggregation_2021-Onward.R` | Aggregates raw HMS trip-level purchase records into household-month totals by product category (cigarettes, e-cigarettes by flavor/authorization status, etc.) |
| `02_Cleaning_2021-Onward.R` | Applies sample restrictions, constructs household demographic variables (TYA presence, household size, income), builds the flavored e-cigarette indicator, and merges product-level attributes |
| `03_Cleaning_Non-Tobacco_Household_Info_2021-Onward.R` | Cleans HMS household-level demographic files for panelists who do not purchase tobacco, used to construct the full household panel for the all-panelists regressions |
| `04_Monthly_Aggregation_2021-Onward.R` | Collapses cleaned purchase records to the household-month level, imputes missing prices, and outputs the final analysis-ready datasets |

### `Data_Cleaning/Census/`

| Script | Description |
|--------|-------------|
| `Census_Aggregation_2021-Onward.R` | Pulls American Community Survey microdata used to benchmark the HMS panel against the U.S. population |

---

## Summary Statistics

### `Data_Analysis/HMS/Summary_Stats/`

Produces descriptive statistics reported in the paper.

| Script | Description |
|--------|-------------|
| `01_Household_Summary_Stats.R` | Household-level summary statistics: demographic composition, TYA presence rates, tobacco purchase frequencies, flavored vs. unflavored e-cigarette shares by household type |
| `02_Product_Summary_Stats.R` | Product-level summary statistics: category market shares, price distributions, FDA authorization rates, flavored e-cigarette category breakdown |

---

## Structural Model Motivation

### `Data_Analysis/HMS/Structural_Model_Motivation/`

Reduced-form empirical evidence motivating features of the structural model.

| Script | Description |
|--------|-------------|
| `01_TYA_Regression.R` | Linear probability model regressions of flavored e-cigarette purchase on teen/young adult household presence, run across three sample cuts (all household-months, cig/e-cig purchasers, e-cig purchasers only). Also estimates the interaction between FDA-authorized e-cigarette availability and TYA presence. Motivates the TYA-specific utility shifter in the structural model. |
| `02_Transition_Matrix.R` | Estimates household-level choice persistence matrices across tobacco product categories (cigarettes, flavored e-cigs, unflavored e-cigs, outside good). |
| `04_Relapse_Analysis.R` | Analyzes purchase streak lengths and relapse patterns after cessation spells. |
| `05_Stockpiling_Test.R` | Tests for stockpiling behavior. |
| `06_CA_Synthetic_Control.R` | Synthetic control for California cigarette and unflavored e-cigarette prices around the December 2022 California flavor ban. |
| `07_Repeated_Quit_Attempts.R` | Counts distinct abstention spells per household to document repeated quit attempts, a signature of present bias. |
| `09_Flavor_Persistence_Regression.R` | Regresses current flavored e-cigarette purchase on its lag, controlling for lagged e-cigarette consumption, to document flavor-specific state dependence. |
| `10_Initial_Conditions_Time_Trend.R` | Tests for within-household time trends in consumption with household and month fixed effects, a check on the initial-conditions assumption. |
| `11_Elf_Bar_Enforcement_Event_Study.R` | Event study around the FDA's 2023 enforcement actions against Elf Bar, comparing households that bought the named brands with other unauthorized flavored e-cigarette buyers. |

---

## Dynamic Model

### `Data_Analysis/HMS/Dynamic_Model/`

The core structural estimation code. The first-stage scripts are in R; everything else is in Julia and parallelized for use on an HPC.

---

### `01_First_Stage_Estimation/`

Prepares the inputs needed for structural estimation: price processes, state transition matrices, and auxiliary data objects. All scripts are in R.

| Script | Description |
|--------|-------------|
| `01_Data_Prep.R` | Constructs the household-level panel used in estimation: product choice indicators, addiction stock state variable, price vectors, demographic controls, and TYA flags |
| `01_Data_Prep_Holdout.R` | Same as `01_Data_Prep.R` but sets aside a holdout sample for out-of-sample model validation |
| `01_Data_Prep_Holdout_Val.R` | Prepares the holdout validation dataset in the format required by the validation scripts |
| `02_Pricing_Spaces.R` | Discretizes the continuous price distributions into finite price grids for each product category; these grids define the state space over which the VFI is solved |
| `02_Pricing_Spaces_Holdout.R` | Same as `02_Pricing_Spaces.R` applied to the holdout sample |
| `03_AR_Estimation.R` | Estimates AR(1) price processes for each product category via household fixed-effects regression; the AR coefficients and shock variances parameterize the price transition matrices used in VFI |
| `03_AR_Estimation_Holdout.R` | Same as `03_AR_Estimation.R` applied to the holdout sample |
| `04_Price_State_Transitions.R` | Constructs the discrete price state transition matrices from the AR(1) estimates using Halton draws |
| `04_Price_State_Transitions_Holdout.R` | Same as `04_Price_State_Transitions.R` applied to the holdout sample |

---

### `02_Second_Stage_Estimation_Mixture/`

Estimates the structural parameters of the dynamic mixture model via maximum likelihood.

| Script | Description |
|--------|-------------|
| `01_Functions_Mixture.jl` | Core functions: VFI solver, likelihood contributions, mixture posterior weights, and others shared across estimation and counterfactual scripts |
| `02_Estimation_Mixture.jl` | Main estimation script. Solves the dynamic programming problem for each latent type and each point in the parameter search, evaluates the mixture likelihood, and optimizes over structural parameters with a multi-start Nelder-Mead optimizer. |
| `02_Estimation_Mixture_Slurm.sb` | SLURM submission script for running estimation on the HPC. |

---

### `02_Second_Stage_Estimation_Mixture_Holdout/`

Estimates the structural model on the holdout sample for out-of-sample validation. Mirrors the structure of `02_Second_Stage_Estimation_Mixture/`.

| Script | Description |
|--------|-------------|
| `01_Functions_Mixture.jl` | Same core functions as the main estimation folder |
| `02_Estimation_Mixture.jl` | Estimation on the holdout sample |
| `02_Estimation_Mixture_Slurm.sb` | SLURM submission script for holdout estimation |

---

### `02_Second_Stage_Estimation_Mixture_LongRun/` and `02_Second_Stage_Estimation_Mixture_Holdout_LongRun/`

Same scripts as the multi-start folders, but `02_Estimation_Mixture.jl` runs a single long Nelder-Mead run (`long_amoeba`) from the starting parameters, meant to polish the best inner try of the multi-start estimation. Outputs carry a `LongRun` tag.

---

### `03_Standard_Errors_Mixture/`

Computes standard errors for the MLE estimates via central finite differences on the full negative log-likelihood. Each script includes the functions file of the estimation folder it serves and writes `_SE.csv` and `_VCov.csv` into that folder's results directory.

| Script | Description |
|--------|-------------|
| `01_Standard_Errors_Mixture.jl` | Standard errors for the `02_Second_Stage_Estimation_Mixture` estimates |
| `01_Standard_Errors_Mixture_Slurm.sb` | SLURM submission script |
| `02_Standard_Errors_Mixture_LongRun.jl` | Standard errors for the `02_Second_Stage_Estimation_Mixture_LongRun` estimates |
| `02_Standard_Errors_Mixture_LongRun_Slurm.sb` | SLURM submission script |

---

### `04_Validation_Mixture/`

Forward-simulates the estimated model to evaluate in-sample fit.

| Script | Description |
|--------|-------------|
| `01_Validation_Functions_Mixture.jl` | Functions for forward simulation, market share aggregation, and fit diagnostics |
| `02_Validation_Mixture.jl` | Main validation script: forward-simulates the model under the estimated parameters and outputs in-sample fit statistics (streak persistence, quantity escalation, price elasticities) |
| `02_Validation_Mixture_Slurm.sb` | SLURM submission script for validation on the HPC |

---

### `04_Validation_Mixture_Holdout/`

Same as `04_Validation_Mixture/` but evaluated on the holdout sample, providing an out-of-sample test of model fit.

| Script | Description |
|--------|-------------|
| `01_Validation_Functions_Mixture.jl` | Same validation functions as the in-sample folder |
| `02_Validation_Mixture.jl` | Holdout validation script |
| `02_Validation_Mixture_Slurm.sb` | SLURM submission script for holdout validation |

---

### `05_CF_Mixture/`

Single-run counterfactual policy analysis at the point estimates. Re-solves the VFI under alternative regulatory regimes and forward-simulates outcomes.

Policies analyzed:

- **Comprehensive flavor ban** (`Ban_Comprehensive`): removes all flavored e-cigarettes from the choice set
- **Non-FDA flavor ban** (`Ban_Non_FDA`): bans only non-FDA-authorized flavored products
- **Per-mL flavor tax** (`Flavor_Tax`): per-mL excise tax on flavored e-cigarettes at each rate in `TAX_GRID` (default $0.50 and $2.78 per mL), with the tax rate that matches the comprehensive ban's addiction reduction interpolated from the grid

| Script | Description |
|--------|-------------|
| `01_CF_Functions_Mixture.jl` | Counterfactual-specific functions: policy utility modifiers, tax application, revenue computation, and welfare calculation |
| `02_CF_Mixture.jl` | Main counterfactual script. Runs each policy at the present-bias parameter given by the `BETA` environment variable (default 1.0), solves the VFI under each regime, forward-simulates outcomes, and saves results overall and by TYA status and household latent type |
| `02_CF_Slurm_Mixture.sb` | SLURM submission script for counterfactual analysis on the HPC |

---

### `05_CF_Mixture_Bootstrap/`

The counterfactual pipeline used in the paper, with a parametric bootstrap for inference. Each SLURM array task $b$ draws $\theta_b \sim N(\hat\theta, \hat V)$ from the estimated asymptotic distribution (task 0 uses $\hat\theta$ itself; draws violating the parameter bounds are redrawn), then solves the status quo once and runs every policy at that draw and a single fixed $\beta$ (`BETA` environment variable), using common random numbers across draws so that simulation noise does not confound parameter uncertainty.

Each task writes, per policy, category shares, addiction stocks, consumer surplus, and quantities overall and by TYA status and modal latent type, plus initiation among baseline non-users (flavored e-cigarettes, cigarettes, and unflavored e-cigarettes; non-user base defined from the observed purchase history over the prior 3, 6, or 12 months). It also traces the tax-versus-ban frontier over the rates in `FRONTIER_TAX_GRID` (default $1 to $15 per mL in whole-dollar steps) for the e-cigarette-persistent households, reporting each rate's addiction reduction as a fraction of the ban's, its consumer-surplus cost, and the rate at which the tax matches the ban. Setting `FRONTIER_ONLY=1` reruns only the frontier.

| Script | Description |
|--------|-------------|
| `01_CF_Functions_Mixture.jl` | Counterfactual functions shared by the bootstrap pipeline |
| `02_CF_Mixture_Bootstrap.jl` | Draws $\theta_b$, then solves the VFI and forward-simulates outcomes under each policy, writing output tagged with the draw index $b$ |
| `02_CF_Mixture_Bootstrap_Slurm.sb`, `_Slurm_1.sb`, `_Slurm_2.sb` | SLURM array submission scripts; identical except for the task range (0-33, 34-67, 68-100) so the 100 draws plus the point run can be split across allocations |
| `03_CF_Bootstrap_Mixture_Aggregation.jl` | Stacks the draw-specific output CSVs for every policy and writes the mean, 95% percentile confidence interval, and point-estimate value for each outcome, plus the aggregated frontier files |
| `03_CF_Bootstrap_Mixture_Aggregation_Slurm.sb` | SLURM submission script for the aggregation step |

---

### `06_Vuong_Test_Mixture/`

Compares the time-consistent ($\beta = 1$) and present-biased ($\beta = 0.7$) specifications with a Vuong (1989) non-nested test.

| Script | Description |
|--------|-------------|
| `01_Vuong_Functions_Mixture.jl` | Functions for the test statistic, its variance pre-test, AIC/BIC-corrected variants, and subgroup breakdowns |
| `02_Vuong_Test_Mixture.jl` | Loads the point estimates for the two $\beta$ values (`BETA_A`, `BETA_B` environment variables), solves the VFI for each, evaluates the mixture log-likelihood household by household, and differences the contributions to form the Vuong statistic, with breakdowns by TYA status and modal latent type |
| `02_Vuong_Test_Slurm_Mixture.sb` | SLURM submission script |
