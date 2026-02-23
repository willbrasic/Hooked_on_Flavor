# Structural Model Motivation

This directory contains the **static conditional logit model** that motivates the dynamic discrete choice model. The static model estimates tobacco product demand without dynamic addiction terms, using state dependence (lagged category choice) as a reduced-form proxy for habit persistence.

## Directory Structure

```
Structural_Model_Motivation/
├── 01_TYA_Regression.R
├── 02_Transition_Matrix.R
├── 03_Static_Logit.jl
├── 03_Static_Logit_Slurm.sb
├── 04_Static_Nested_Logit.jl
├── 04_Static_Nested_Logit_Slurm.sb
└── 05_Reduced_Form_Psi.R
```

## Script: `03_Static_Logit.jl`

Estimates a static conditional logit model of tobacco product demand. The key finding motivating the dynamic model is a large, significant state dependence parameter (ρ): past choices strongly predict current behavior, suggesting addiction/habit persistence that a static model cannot properly account for.

**Includes:** `../Dynamic_Model/02_Second_Stage_Estimation/01_Functions.jl` (local) or `../Dynamic_Model/01_Functions.jl` (HPC). Uses `hpc` flag to toggle between local and HPC paths.

**Working directory:** `cd` to `.../4th_Year_Paper_Data/HMS/2021-Onward/Dynamic_Model/Data/` for CSV reads.

**Output directory:** `.../4th_Year_Paper_Data/HMS/2021-Onward/Static_Logit_Results/`

### Data Loading

Uses shared functions from `01_Functions.jl` to load:
- Product choices (40 alternatives), choice vector y
- Consumption vectors (c_cig, c_ecig, c_bundle), category indices, split flavored indicators (is_non_fda_flavored, is_fda_flavored)
- TYA (teen/young adult) binary indicator
- Continuous prices via `map_prices_to_grid` (only the continuous prices are used, not the grid indices)

**Important:** `get_consumption()` returns STANDARDIZED consumption (divided by max). The static logit converts back to RAW values (`c_cig = c_cig_std * c_cig_max`) so that estimates are in **original units** (utils per pack, utils per mL, etc.). This allows the estimates to be used as starting values in the dynamic model after conversion to standardized units.

Additionally loads:
- `Lagged_Category_Choice.csv` for the state dependence term. Observations with missing lagged choice (first month per household) are dropped.

### Pre-computed Matrices

| Matrix | Dimension | Description |
|--------|-----------|-------------|
| `fe_T`, `fe_E`, `fe_TE` | N_J | Boolean fixed effect indicators by alternative (cig, ecig categories 2-4, bundle categories 5-7) |
| `c_bundle` | N_J | Pre-computed `c_cig[j] * c_ecig[j]` for bundle interaction |
| `E_obs[i, j]` | N_obs x N_J | Observation-level expenditure: `p_cig[i] * c_cig[j] + p_ecig[i] * c_ecig[j]` |
| `lag_match[i, j]` | N_obs x N_J | 1 if alternative j's category matches the household's lagged category choice |

### Structural Parameters (12)

```
α_T   = cigarette consumption utility (per pack)
α_E   = e-cig consumption utility (per mL)
α_TE  = bundle interaction utility (per pack × mL)
λ_1   = non-FDA flavor baseline effect
λ_2   = non-FDA flavor × teen/young adult interaction
λ_3   = FDA flavor baseline effect
λ_4   = FDA flavor × teen/young adult interaction
ρ     = state dependence (lagged category match)
ω     = expenditure coefficient (price sensitivity)
ξ_T   = cigarette fixed effect
ξ_E   = e-cig fixed effect
ξ_TE  = bundle fixed effect
```

### Flow Utility

```
v(j, i) = α_T * c_cig[j] + α_E * c_ecig[j] + α_TE * c_bundle[j]
         + 𝟙[non-FDA flav] * (λ_1 + λ_2 * tya[i])
         + 𝟙[FDA flav] * (λ_3 + λ_4 * tya[i])
         + ρ * lag_match[i, j]
         + ω * E_obs[i, j]
         + ξ_T * fe_T[j] + ξ_E * fe_E[j] + ξ_TE * fe_TE[j]
```

where `E_obs[i, j] = p_cig[i] * c_cig[j] + p_ecig[i] * c_ecig[j]` is total expenditure, non-FDA flav = cat ∈ {3, 6}, and FDA flav = cat ∈ {4, 7}.

Compared to the dynamic model, the static model:
- **Drops:** addiction terms (μ·a·n[j] and γ·a) since there is no addiction state
- **Adds:** state dependence (ρ·lag_match) as a reduced-form proxy for habit persistence
- **Uses:** continuous observed prices rather than discretized price grid states

### Negative Log-Likelihood

`neg_log_likelihood(θ_vec, N_obs, N_J, tya, y, c_cig, c_ecig, c_bundle, is_non_fda_flavored, is_fda_flavored, lag_match, E_obs, fe_T, fe_E, fe_TE)` computes the observation-level log-likelihood:

1. Loop over observations
2. For each observation, compute deterministic utility for all N_J alternatives
3. Compute log choice probability via logsumexp: `log P(yᵢ) = v[yᵢ] - logsumexp(v)`

A single-argument wrapper `nll = θ -> neg_log_likelihood(θ, ...)` is used by the optimizer and Hessian computation. Takes data arrays as arguments for type stability with ForwardDiff.

### Estimation (Two Optimizers)

**1. L-BFGS with autodiff gradients:**
- `optimize(nll, theta_start, LBFGS(), ...; autodiff = :forward)`
- JIT warmup call `nll(theta_start)` before timed optimization (pre-compiles for ForwardDiff Dual types)
- Callback logs every 10 iterations (iteration number, neg LL, gradient norm)
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

Uses `log_msg()` from `01_Functions.jl` (prints to stdout and writes to `log_io`). The log file is opened at the start of the script and closed at the end.

### Relationship to Dynamic Model

**Estimates are in ORIGINAL UNITS.** The dynamic model (`02_Estimation.jl`) uses standardized data, so consumption utility estimates must be converted to standardized units when used as starting values:

| Parameter | Conversion to Standardized Units |
|-----------|----------------------------------|
| α_T | α_T_std = α_T_orig × c_cig_max |
| α_E | α_E_std = α_E_orig × c_ecig_max |
| α_TE | α_TE_std = α_TE_orig × c_bundle_max |
| ω | ω_std = ω_orig × E_max |
| λ_1, λ_2, λ_3, λ_4, ρ, ξ_T, ξ_E, ξ_TE | No conversion (multiply indicators/additive) |

---

## Script: `04_Static_Nested_Logit.jl`

Estimates a **static nested logit** model of tobacco product demand. Same data and state dependence term as `03_Static_Logit.jl` but replaces the standard logit choice probabilities with nested logit to relax the IIA assumption. This serves as a **robustness check** for the flavor ban counterfactual: the nesting parameter σ controls within-nest vs across-nest substitution, which determines how much consumers substitute to original e-cigs (same nest) vs cigarettes (different nest) when flavored e-cigs are removed.

**Includes:** Same as `03_Static_Logit.jl`.

**Working directory:** Same as `03_Static_Logit.jl`.

**Output directory:** `.../4th_Year_Paper_Data/HMS/2021-Onward/Static_Nested_Logit_Results/`

### Nesting Structure (4 nests, 1 common σ)

| Nest | Alternatives | Description |
|------|-------------|-------------|
| 1 | j = 1 | Outside option (singleton, σ irrelevant) |
| 2 | j = 2:13 (12 alts) | Cigarettes |
| 3 | j = 14:34 (21 alts) | E-cigs: orig + non-FDA flav + FDA flav |
| 4 | j = 35:40 (6 alts) | Bundles: orig + non-FDA flav + FDA flav |

Original and flavored e-cigs (both non-FDA and FDA) are in the **same nest** — this is the key grouping for the flavor ban counterfactual. Removing flavored e-cigs primarily shifts consumers to original e-cigs (same nest, stronger substitution) rather than cigarettes (different nest, weaker substitution).

### σ Parameterization

- Estimate `σ_raw` (unconstrained), transform via logistic: `σ = 1/(1+exp(-σ_raw))` → σ ∈ (0,1)
- Starting value: `σ_raw = 0` → σ = 0.5
- When σ = 0: standard logit (IIA). When σ > 0: stronger within-nest substitution.
- Within-nest to across-nest substitution ratio: `1/(1-σ)`

### Nested Logit Choice Probability

```
IV_g = logsumexp(v_k/(1-σ) for k in nest g)                    [inclusive value]
log P(j) = v_j/(1-σ) - σ × IV_{nest(j)} - logsumexp((1-σ) × IV_{g'} for all g')
```

Numerically stable via logsumexp throughout. When σ → 0, reduces to standard logit.

### Structural Parameters (13)

Same 12 parameters as `03_Static_Logit.jl` plus σ_raw:

```
α_T, α_E, α_TE, λ_1, λ_2, λ_3, λ_4, ρ, ω, ξ_T, ξ_E, ξ_TE, σ_raw
```

### Pre-computed Nest Structure

| Variable | Description |
|----------|-------------|
| `nest_id[j]` | Nest assignment for each alternative (1-4) |
| `nest_alts[g]` | Vector of alternative indices belonging to nest g |
| `N_nests` | Number of nests (4) |

Computed from `cat_idx`: outside → nest 1, cig (cat 1) → nest 2, ecig orig+non-FDA flav+FDA flav (cat 2,3,4) → nest 3, bundles orig+non-FDA flav+FDA flav (cat 5,6,7) → nest 4.

### σ Standard Error (Delta Method)

After estimation, the transformed σ̂ and its SE are reported:
- `σ̂ = 1/(1+exp(-σ̂_raw))`
- `SE(σ̂) = σ̂(1-σ̂) × SE(σ̂_raw)` (delta method, since ∂σ/∂σ_raw = σ(1-σ))

### Estimation Results (February 2026)

```
σ̂ = 0.1227 (SE = 0.0168, t = 7.29)
```

**Interpretation:** σ is statistically significant but modest. Within-nest substitution is 1/(1-0.12) ≈ 14% stronger than across-nest. This indicates the IIA assumption in the dynamic model is not severely violated. The nested logit is used as a robustness check rather than being incorporated into the dynamic model, where the implementation cost (modifying VFI, likelihood, counterfactual, MC code) would be substantial relative to the modest departure from IIA.

**Parameter changes from standard logit → nested logit:**
- α_T, α_E, ω: stable (< 15% change)
- λ_1: dropped ~38% (0.852 → 0.532) — flavor effects were partly absorbing within-nest substitution
- λ_2: dropped ~29% (0.703 → 0.502)
- ξ_E: less negative (-6.500 → -5.864)
- ρ: stable (~2.77)

### Relationship to Dynamic Model

σ is **not incorporated** into the dynamic model. The modest σ̂ = 0.12 supports using standard logit in the dynamic model. If σ̂ had been large (0.5+), modifying the dynamic model VFI for nested logit would have been warranted. The static nested logit result is reported as evidence that IIA is approximately satisfied.

---

## Script: `05_Reduced_Form_Psi.R`

Estimates the addiction decay rate ψ from the reduced-form persistence of nicotine consumption via AR(1) regression. Written in R using data.table and fixest. Reads the same processed CSVs from `.../Dynamic_Model/Data/` used by the structural model.

**Main specification (Mundlak/CRE):**
```
n_it = α + ρ · n_{i,t-1} + β_1 · p_cig_it + β_2 · p_ecig_it
     + β_3 · mean_cig_i + β_4 · mean_ecig_i + ε_it
```
Estimated via OLS with `fixest::feols`, clustered SEs at household level. **ψ̂ = 1 - ρ̂**, SE(ψ̂) = SE(ρ̂) by delta method.

**Models estimated:**
1. Mundlak CRE (main specification)
2. Mundlak CRE + month fixed effects
3. Separate cig and ecig AR(1)s (product-specific persistence)
4. Household FE (Nickell-biased comparison)

**Key results:** ρ̂ ≈ 0.52 (Mundlak), implying ψ̂ ≈ 0.48. Household FE gives ρ̂ ≈ 0.25 (biased downward by Nickell bias in short panels). The main estimate ψ̂ = 0.4801 is used as the fixed value in the dynamic model.

**Why Mundlak over household FE:** With a lagged dependent variable + household FE, demeaning creates mechanical negative correlation between the demeaned lag and demeaned error (Nickell, 1981), biasing ρ̂ downward (and ψ̂ upward). The Mundlak/CRE approach (household means as controls) avoids this.

### Relationship to Dynamic Model

ψ̂ = 0.4801 is fixed in all structural estimation code via `get_fixed_parameters()` in `01_Functions.jl`. This resolves the ψ-μ identification ridge found in MC simulations where the optimizer found high ψ (~0.875) + negligible μ instead of the true values.
