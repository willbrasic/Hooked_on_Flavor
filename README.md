# Hookd on Flavor: Addiction, Present Bias, and the Consequences of E-Cigarette Flavor Policy

**William Brasic**  
Department of Economics, University of Arizona  
wbrasic@arizona.edu

---

## Overview

This repository contains the code for [*Hooked on Flavor: Addiction, Present Bias, and the Consequences of E-Cigarette Flavor Policy*](https://williambrasic.com/Research/JMP_20260611.pdf), studying tobacco product demand with a focus on flavored e-cigarette regulation. The paper estimates a **finite mixture dynamic discrete choice model** of household tobacco demand using Nielsen Homescan (HMS) panel data from 2021-2023. The model captures addiction dynamics, habit formation, preference heterogeneity across latent household types, and the role of teens and young adults (TYA) in driving flavored e-cigarette adoption.

The counterfactual analysis evaluates the welfare and public health effects of several FDA flavor regulation policies, including comprehensive flavor bans, FDA-authorized product restrictions, and per-mL flavor taxes.

---

## Data

The underlying data are from the **Nielsen Homescan (HMS) Consumer Panel** (2021-2023), a household-level panel tracking retail purchases across product categories. The raw data are proprietary and not included in this repository.

---

## Languages and Dependencies

| Language | Version | Key Packages |
|----------|---------|--------------|
| R | ≥ 4.2 | `data.table`, `fixest`, `plm`, `sandwich`, `stargazer`, `ggplot2`, `pacman` |
| Julia | ≥ 1.9 | `Optim`, `ForwardDiff`, `LinearAlgebra`, `Distributions`, `CSV`, `DataFrames`, `SharedArrays` |

Julia scripts are parallelized and are designed to run on a SLURM HPC cluster. Each subfolder containing a `.sb` file includes the corresponding SLURM submission script.

---

## Repository Structure

```
.
├── Data_Cleaning/
│   └── HMS/
│       └── 2021-Onward/                          # Raw HMS data cleaning pipeline (R)
└── Data_Analysis/
    └── HMS/
        ├── Summary_Stats/                         # Descriptive statistics (R)
        ├── Structural_Model_Motivation/           # Reduced-form evidence (R)
        ├── Figure_Creation/                       # Publication-ready figures (R)
        └── Dynamic_Model/                         # Structural estimation pipeline (Julia + R)
            ├── 01_First_Stage_Estimation/
            ├── 02_Second_Stage_Estimation_Mixture/
            ├── 02_Second_Stage_Estimation_Mixture_Holdout/
            ├── 03_Validation_Mixture/
            ├── 03_Validation_Mixture_Holdout/
            └── 04_CF_Mixture/
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

Reduced-form empirical evidence motivating key features of the structural model.

| Script | Description |
|--------|-------------|
| `01_TYA_Regression.R` | Linear probability model regressions of flavored e-cigarette purchase on teen/young adult household presence, run across three sample cuts (all household-months, cig/e-cig purchasers, e-cig purchasers only). Also estimates the interaction between FDA-authorized e-cigarette availability and TYA presence. Motivates the TYA-specific utility shifter in the structural model. |
| `02_Transition_Matrix.R` | Estimates household-level choice persistence matrices across tobacco product categories (cigarettes, flavored e-cigs, unflavored e-cigs, outside good). |
| `04_Relapse_Analysis.R` | Analyzes purchase streak lengths and relapse patterns after cessation spells. |
| `05_Stockpiling_Test.R` | Tests for stockpiling behavior. |
| `06_CA_Synthetic_Control.R` | Synthetic control analysis using California's flavor ban as a natural experiment. |

---

## Figure Creation

### `Data_Analysis/HMS/Figure_Creation/`

Produces all publication-ready figures in the paper and appendix.

| Script | Description |
|--------|-------------|
| `01_Cig_ECig_Consumption_Time_Plot.R` | Time series of cigarette and e-cigarette consumption levels across the sample period |
| `02_Cig_ECig_FE_Time_Plot.R` | Fixed-effects time trends for cigarette and e-cigarette purchases, netting out household-level heterogeneity |
| `03_ECig_Flavored_UnFlavored_Frequencies_Plot.R` | Purchase frequency distributions comparing flavored vs. unflavored e-cigarette household-months |
| `04_Cig_ECig_Frequencies_Plot.R` | Joint frequency plot of cigarette and e-cigarette purchase incidence |
| `05_Cig_ECig_Prices_Time_Plot.R` | Time series of average prices for cigarettes and e-cigarettes |
| `06_ECig_Flavored_Unflavored_Shares_Plot.R` | Market share evolution of flavored vs. unflavored e-cigarettes over time |
| `07_ECig_FDA_Authorized_Frequencies_Plot.R` | Purchase frequency breakdown by FDA authorization status of e-cigarette products |
| `08_Streak_Persistence_Plot.R` | Visualization of purchase streak persistence from the relapse analysis |
| `A_01_Category_Shares_Plot.R` | Category-level market share breakdown across all tobacco product types |
| `A_02_Cig_ECig_Prices_Bins_Plot.R` | Price distributions for cigarettes and e-cigarettes using binned histograms |
| `A_02_Cig_Ecig_Frequencies_TYA_Plot.R` | Purchase frequency comparison split by TYA household presence |

---

## Dynamic Model

### `Data_Analysis/HMS/Dynamic_Model/`

The core structural estimation pipeline. All scripts are written in Julia and parallelized for use on an HPC. The model is a **finite mixture dynamic discrete choice model** of tobacco demand estimated by maximum likelihood. Each latent type has its own preference parameters, and households are probabilistically assigned to types via thier posterior probabilities. Value functions are solved by value function iteration (VFI) over a discretized addiction stock state space.

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

Estimates the structural parameters of the K=3 mixture dynamic discrete choice model via maximum likelihood. 

| Script | Description |
|--------|-------------|
| `01_Functions_Mixture.jl` | Core functions: VFI solver, likelihood contributions, mixture posterior weights, and auxiliary utilities shared across estimation and counterfactual scripts |
| `02_Estimation_Mixture.jl` | Main estimation script. Solves the dynamic programming problem for each latent type and each point in the parameter search, evaluates the mixture likelihood, and optimizes over structural parameters using pseudo-Nelder-Mead optimizer. |
| `02_Estimation_Mixture_Slurm.sb` | SLURM submission script for running estimation on the HPC. |
| `03_Standard_Errors_Mixture.jl` | Computes standard errors for the MLE estimates via central finite differences. |
| `03_Standard_Errors_Mixture_Slurm.sb` | SLURM submission script for the standard errors computation. |

---

### `02_Second_Stage_Estimation_Mixture_Holdout/`

Estimates the structural model on the holdout sample for out-of-sample validation. Mirrors the structure of `02_Second_Stage_Estimation_Mixture/`.

| Script | Description |
|--------|-------------|
| `01_Functions_Mixture.jl` | Same core functions as the main estimation folder, adapted for the holdout sample |
| `02_Estimation_Mixture.jl` | Estimation on the holdout sample |
| `02_Estimation_Mixture_Slurm.sb` | SLURM submission script for holdout estimation |

---

### `03_Validation_Mixture/`

Forward-simulates the estimated model to evaluate in-sample fit.

| Script | Description |
|--------|-------------|
| `01_Validation_Functions_Mixture.jl` | Functions for forward simulation, market share aggregation, and fit diagnostics |
| `02_Validation_Mixture.jl` | Main validation script: forward-simulates the model under the estimated parameters and outputs in-sample fit statistics and figures |
| `02_Validation_Mixture_Slurm.sb` | SLURM submission script for validation on the HPC |

---

### `03_Validation_Mixture_Holdout/`

Same as `03_Validation_Mixture/` but evaluated on the holdout sample, providing a out-of-sample test of model fit.

| Script | Description |
|--------|-------------|
| `01_Validation_Functions_Mixture.jl` | Same validation functions adapted for the holdout sample |
| `02_Validation_Mixture.jl` | Holdout validation script |
| `02_Validation_Mixture_Slurm.sb` | SLURM submission script for holdout validation |

---

### `04_CF_Mixture/`

Counterfactual policy analysis. Re-solves the VFI under alternative regulatory regimes.

Policies analyzed:

- **Comprehensive flavor ban**: removes all flavored e-cigarettes from the choice set
- **FDA-authorized products only**: restricts the market to FDA-authorized e-cigarettes
- **Non-FDA flavor ban**: bans only non-FDA-authorized flavored products
- **Per-mL flavor tax**: implements per-mL excise tax on e-cigarettes

| Script | Description |
|--------|-------------|
| `01_CF_Functions_Mixture.jl` | Counterfactual-specific functions: policy utility modifiers, tax application, revenue computation, and welfare calculation |
| `02_CF_Mixture.jl` | Main counterfactual script. Loops over policies and present bias specifications ($\beta \in \{0.70, 1.0\}$), solves the VFI under each regime, forward-simulates outcomes, and saves results by policy and household latent type |
| `02_CF_Slurm_Mixture.sb` | SLURM submission script for counterfactual analysis on the HPC |
