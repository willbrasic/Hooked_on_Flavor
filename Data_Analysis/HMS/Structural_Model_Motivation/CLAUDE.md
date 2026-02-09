# Structural Model Motivation

This directory contains the **static conditional logit model** that motivates the dynamic discrete choice model. The static model estimates tobacco product demand without dynamic addiction terms, using state dependence (lagged category choice) as a reduced-form proxy for habit persistence.

## Directory Structure

```
Structural_Model_Motivation/
├── 01_TYA_Regression.R
├── 02_Transition_Matrix.R
├── 03_Static_Logit.jl
└── 03_Static_Logit_Slurm.sb
```

## Script: `03_Static_Logit.jl`

Estimates a static conditional logit model of tobacco product demand with a **finite mixture** (K=2 discrete types) to address unobserved preference heterogeneity and consumption endogeneity.

The key finding motivating the dynamic model is a large, significant state dependence parameter (ρ): past choices strongly predict current behavior, suggesting addiction/habit persistence that a static model cannot properly account for.

**Includes:** `../Dynamic_Model/02_Second_Stage_Estimation/01_Functions.jl` (local) or `../Dynamic_Model/01_Functions.jl` (HPC). Uses `hpc` flag to toggle between local and HPC paths.

**Working directory:** `cd` to `.../4th_Year_Paper_Data/HMS/2021-Onward/Dynamic_Model/Data/` for CSV reads.

**Output directory:** `.../4th_Year_Paper_Data/HMS/2021-Onward/Static_Logit_Results/`

### Data Loading

Uses shared functions from `01_Functions.jl` to load:
- Product choices (21 alternatives), choice vector y
- Consumption vectors (c_cig, c_ecig), category indices, flavored indicator
- TYA (teen/young adult) indicator
- Continuous prices via `map_prices_to_grid` (only the continuous prices are used, not the grid indices)

**Important:** `get_consumption()` returns STANDARDIZED consumption (divided by max). The static logit converts back to RAW values (`c_cig = c_cig_std * c_cig_max`) so that estimates are in **original units** (utils per pack, utils per mL, etc.). This allows the estimates to be used as starting values in the dynamic model after conversion to standardized units.

Additionally loads:
- `Lagged_Category_Choice.csv` for the state dependence term. Observations with missing lagged choice (first month per household) are dropped.
- `Household_Codes.csv` for grouping observations by household (required for household-level likelihood in finite mixture).

### Pre-computed Matrices

| Matrix | Dimension | Description |
|--------|-----------|-------------|
| `fe_T`, `fe_E`, `fe_TE` | N_J | Boolean fixed effect indicators by alternative (cig, ecig, bundle) |
| `c_bundle` | N_J | Pre-computed `c_cig[j] * c_ecig[j]` for bundle interaction |
| `E_obs[i, j]` | N_obs x N_J | Observation-level expenditure: `p_cig[i] * c_cig[j] + p_ecig[i] * c_ecig[j]` |
| `lag_match[i, j]` | N_obs x N_J | 1 if alternative j's category matches the household's lagged category choice |
| `obs_by_hh[h]` | N_HH | Vector of observation indices belonging to household h |

### Finite Mixture Model (K=2 Types)

The model uses discrete types to address consumption endogeneity (Arcidiacono & Miller, 2011; Heckman & Singer, 1984). Types capture permanent unobserved heterogeneity in propensity to purchase tobacco:

- **Type 1 (Non-buyers):** Low ξ values → prefer outside option
- **Type 2 (Buyers):** High ξ values → prefer tobacco products

**Why type-varying ξ instead of α?** With ~48% of observations choosing the outside option, the main source of heterogeneity is "buyer vs non-buyer" rather than "how much buyers value consumption." Letting ξ vary by type captures this cleanly, while keeping α common allows identification of consumption utility from within-type variation.

**Household-level likelihood:** Types are permanent household characteristics. The likelihood is computed at the household level:

```
L_h = π × [∏_t P_{ht}^1(y_{ht})] + (1-π) × [∏_t P_{ht}^2(y_{ht})]
```

where the product is over all observations t belonging to household h. This identifies types from the *sequence* of choices.

### Structural Parameters (14)

```
Common across types:
  alpha_T   = cigarette consumption utility (per pack)
  alpha_E   = e-cig consumption utility (per mL)
  alpha_TE  = bundle interaction utility (per pack * mL)
  lambda_1  = baseline flavor effect
  lambda_2  = flavor x teen/young adult interaction
  rho       = state dependence (lagged category match)
  omega     = expenditure coefficient (price sensitivity)

Type-specific:
  xi_T1, xi_E1, xi_TE1 = Type 1 category fixed effects (non-buyers)
  xi_T2, xi_E2, xi_TE2 = Type 2 category fixed effects (buyers)

Mixing:
  pi_raw    = logit-transformed mixing probability (π = exp(π_raw)/(1+exp(π_raw)))
```

### Flow Utility

```
v_k(j, i) = alpha_T * c_cig[j] + alpha_E * c_ecig[j] + alpha_TE * c_bundle[j]
           + 1[flavored] * (lambda_1 + lambda_2 * tya[i])
           + rho * lag_match[i, j]
           + omega * E_obs[i, j]
           + xi_Tk * fe_T[j] + xi_Ek * fe_E[j] + xi_TEk * fe_TE[j]
```

where k ∈ {1, 2} is the household type and `E_obs[i, j] = p_cig[i] * c_cig[j] + p_ecig[i] * c_ecig[j]` is total expenditure.

Compared to the dynamic model, the static model:
- **Drops:** addiction terms (mu * a * n[j] and gamma * a) since there is no addiction state
- **Adds:** state dependence (rho * lag_match) as a reduced-form proxy for habit persistence
- **Adds:** Finite mixture (type-varying ξ) to control for persistent unobserved heterogeneity
- **Uses:** continuous observed prices rather than discretized price grid states

### Negative Log-Likelihood

`neg_log_likelihood(θ_vec, N_HH, N_J, obs_by_hh, tya, y, c_cig, c_ecig, c_bundle, is_flavored, lag_match, E_obs, fe_T, fe_E, fe_TE)` computes the household-level mixture likelihood:

1. Loop over households (not observations)
2. For each household, compute sum of log choice probabilities under each type
3. Combine using log-sum-exp: `log L_h = logsumexp(log π + Σ_t log P^1, log(1-π) + Σ_t log P^2)`

A single-argument wrapper `nll = θ -> neg_log_likelihood(θ, ...)` is used by the optimizer and Hessian computation.

### Estimation (Two Optimizers)

**1. L-BFGS with autodiff gradients:**
- `optimize(nll, theta_start, LBFGS(), ...; autodiff = :forward)`
- JIT warmup call `nll(theta_start)` before timed optimization (pre-compiles for ForwardDiff Dual types)
- Callback logs every 10 iterations (iteration number, neg LL, gradient norm)
- Starting values: small positive α, negative ξ for Type 1, positive ξ for Type 2
- Convergence: g_tol = 1e-4, max 1000 iterations

**2. Nelder-Mead (Random Amoeba):**
- `random_amoeba(nll, starting_param_nm, add_nm, L_nm, M_nm, inner_iter_nm; log_io)`
- L=3 outer tries, M=2 inner tries, 500 iterations per inner run
- Simplex deviations scaled to expected parameter magnitudes

### Standard Errors (Two Methods)

For each optimizer's estimates, SEs are computed via:

**1. Autodiff Hessian:**
- `ForwardDiff.hessian(nll, theta_hat)` gives exact second derivatives
- Symmetrize, invert, take sqrt of diagonal
- Eigenvalue check for positive definiteness

**2. Finite Differences:**
- Central differences with step size h=1e-3
- Diagonal: `H[k,k] = (nll(theta+h*e_k) - 2*nll(theta) + nll(theta-h*e_k)) / h^2`
- Off-diagonal: `H[k,l] = (nll(theta++) - nll(theta+-) - nll(theta-+) + nll(theta--)) / (4h^2)`
- Eigenvalue check for positive definiteness

### Output

A final comparison table shows L-BFGS vs Nelder-Mead estimates side by side with SEs, neg LL, and estimation time.

**Output file:** `Static_Logit_Estimation_Log_<timestamp>.txt` in `.../Static_Logit_Results/` (timestamped to avoid overwriting previous runs)

### Logging

Uses `est_log(msg)` from `01_Functions.jl` (prints to stdout and writes to `est_log_io`). The log file is opened at the start of the script and closed at the end.

### Relationship to Dynamic Model

**Estimates are in ORIGINAL UNITS.** The dynamic model (`02_Estimation.jl`) uses standardized data, so consumption utility estimates must be converted to standardized units when used as starting values:

| Parameter | Conversion to Standardized Units |
|-----------|----------------------------------|
| α_T | α_T_std = α_T_orig × c_cig_max |
| α_E | α_E_std = α_E_orig × c_ecig_max |
| α_TE | α_TE_std = α_TE_orig × c_bundle_max |
| ω | ω_std = ω_orig × E_max |
| λ_1, λ_2, ρ | No conversion (multiply indicators) |

The type-specific fixed effects (ξ_T1, ξ_E1, ξ_TE1, ξ_T2, ξ_E2, ξ_TE2) inform the discrete type specification if implementing finite mixture in the dynamic model.

### Addressing Consumption Endogeneity

The finite mixture approach addresses consumption endogeneity by:

1. **Sorting households by unobserved taste**: The likelihood assigns households to types based on their choice patterns. Households that consistently buy tobacco get high posterior probability of being the "buyer" type.

2. **Type-specific ξ absorbs τ_i**: The category fixed effects capture "propensity to buy tobacco at all." High unobserved nicotine taste (τ_i) → buyer type. Their persistent tendency to buy is explained by type membership.

3. **α identified from within-type variation**: After conditioning on type, α is estimated from how households *within the same type* trade off between different consumption levels. This removes the cross-sectional correlation between unobserved taste and consumption.

This is analogous to a household fixed effect but:
- Uses K discrete types instead of N_HH parameters (avoids incidental parameters problem)
- Extends naturally to dynamic discrete choice models
- Allows for posterior type probabilities and counterfactual analysis by type
