# Dynamic Model

This is a **dynamic discrete choice model** for tobacco product demand (cigarettes and e-cigarettes) using HMS panel data. The estimation follows a **two-stage approach** where the first stage estimates price processes and prepares the choice/state spaces, then the second stage estimates structural parameters via value function iteration (VFI) and maximum likelihood.

## Directory Structure

```
HMS/
├── Structural_Model_Motivation/      # Static logit model motivating the dynamic model
│   ├── 03_Static_Logit.jl
│   ├── 04_Static_Nested_Logit.jl
│   └── 05_Reduced_Form_Psi.R
├── Dynamic_Model/
│   ├── 01_First_Stage_Estimation/    # Price processes and choice/state space preparation
│   │   ├── 01_Data_Prep.R
│   │   ├── 02_Pricing_Spaces.R
│   │   ├── 03_AR_Estimation.R
│   │   ├── 04_State_Transitions.R
│   │   └── 05_TYA_State_Transitions.R
│   ├── 02_Second_Stage_Estimation/   # Structural parameter estimation + SEs
│   │   ├── 01_Functions.jl           # ESTIMATE_BETA, ESTIMATE_PSI, WARM_START flags
│   │   ├── 02_Estimation.jl          # ESTIMATE_BETA, ESTIMATE_PSI, WARM_START flags
│   │   ├── 02_Estimation_Slurm.sb
│   │   └── 03_Standard_Errors.jl
│   ├── 03_Model_Validation/         # Model fit: predicted vs actual shares
│   │   ├── 01_Model_Validation_Functions.jl
│   │   └── 02_Model_Validation.jl
│   ├── 04_Counterfactual_Flavor_Ban/ # Policy counterfactual simulations
│   │   ├── 01_Counterfactual_Functions.jl
│   │   └── 02_Counterfactual_Flavor_Ban.jl
│   ├── 05_MC_Simulation/            # Monte Carlo parameter recovery (ESTIMATE_BETA, ESTIMATE_PSI, WARM_START flags)
│   │   ├── 01_MC_Simulation_Functions.jl
│   │   ├── 02_MC_Simulation_Array.jl   # ESTIMATE_BETA, ESTIMATE_PSI, WARM_START flags
│   │   ├── 02_MC_Simulation_Array_Slurm.sb
│   │   ├── 03_MC_Aggregate_Results.jl
│   │   └── 04_Two_Param_Profile.jl
```

## Data Locations

**Input data (CSVs):**
```
.../4th_Year_Paper_Data/HMS/2021-Onward/Dynamic_Model/Data/
```
All scripts `cd` to this directory for CSV reads.

**AR(1) price parameters:**
```
.../4th_Year_Paper_Data/HMS/2021-Onward/Dynamic_Model/AR_Parameters/
```
One level above `Data/`. MC simulation reads these via `../AR_Parameters/`.

**Output files by script:**
| Script | Output Directory |
|--------|-----------------|
| `02_Estimation.jl` | `.../Dynamic_Model_<psi_tag>_<beta_tag>_Results/` |
| `03_Standard_Errors.jl` | `.../Dynamic_Model_<psi_tag>_<beta_tag>_Results/` |
| `02_Model_Validation.jl` | `.../Dynamic_Model/Model_Validation_Results/` |
| `02_Counterfactual_Flavor_Ban.jl` | `.../Dynamic_Model/Counterfactual_Results/` |
| `02_MC_Simulation_Array.jl` | `.../MC_Simulation_<psi_tag>_<beta_tag>_Results/` |
| `03_Static_Logit.jl` | `.../Static_Logit_Results/` |
| `04_Static_Nested_Logit.jl` | `.../Static_Logit_Results/` |

All under `.../4th_Year_Paper_Data/HMS/2021-Onward/`.

**Output directory naming conventions:**
- `<psi_tag>`: `Psi_0.68` when ESTIMATE_PSI = false; `Psi_Estimated` when ESTIMATE_PSI = true
- `<beta_tag>`: `Beta_1.0` when ESTIMATE_BETA = false; `Beta_Estimated` when ESTIMATE_BETA = true
- Examples: `Dynamic_Model_Psi_0.68_Beta_1.0_Results/`, `MC_Simulation_Psi_Estimated_Beta_Estimated_Results/`

## First Stage Estimation (`01_First_Stage_Estimation/`)

### Script 1: `01_Data_Prep.R`
Prepares the discrete choice data from HMS panel data.

**Outputs:**
| File | Description |
|------|-------------|
| `Household_Codes.csv` | Household identifiers for each observation (household-month) |
| `Category_Choices.csv` | 4 mutually exclusive choices: outside option, cig only, ecig only, both |
| `Product_Choices.csv` | 40 alternatives (39 products + outside option) with quantity bins |
| `Consumption_Spaces.csv` | Median quantities per bin (e.g., cig_5to9 → 6 packs) |
| `Nicotine_Spaces.csv` | Median nicotine absorbed per bin (mg) |
| `Teen_Young_Adult.csv` | Youth presence indicator by household-month |
| `Prices.csv` | Per-unit prices using hierarchical imputation (actual → state-month median → month median) |
| `Lagged_Category_Choice.csv` | Lagged category choice indicators (lagged_cig, lagged_ecig, lagged_cig_ecig); NA for each household's first month |
| `Lead_Prices.csv` | Next month's median per-unit prices (lead_cig_price, lead_ecig_price) using hierarchical imputation (state-month → month); NA for last sample month |
| `Mean_Consumption.csv` | Household mean consumption (mean_cig_consumption, mean_ecig_consumption); controls for persistent unobserved heterogeneity |

**Choice alternatives (40 total):**
- Outside option (no purchase)
- 12 cigarette quantity bins: 1, 2, 3-4, 5-9, 10, 11-19, 20, 21-29, 30, 31-39, 40, 41+ packs
- 7 original e-cig bins: 0-5, 5-10, 10-15, 15-20, 20-30, 30-50, 50+ mL
- 7 non-FDA flavored e-cig bins: 0-5, 5-10, 10-15, 15-20, 20-30, 30-50, 50+ mL
- 7 FDA flavored e-cig bins: 0-5, 5-10, 10-15, 15-20, 20-30, 30-50, 50+ mL
- 6 bundles: 2 cig levels (lo/hi) × 3 ecig types (orig/non-FDA flav/FDA flav)
  - bundle_orig_lo, bundle_orig_hi
  - bundle_non_fda_flav_lo, bundle_non_fda_flav_hi
  - bundle_fda_flav_lo, bundle_fda_flav_hi

### Script 2: `02_Pricing_Spaces.R`
Creates a **10-point price grid** (5th-95th percentiles) for each product category.

**Price ranges:**
- Cigarettes: ~$3.37 - $10.09 per pack
- E-cigs: ~$0.92 - $6.76 per mL

**Output:** `Pricing_Spaces.csv`

### Script 3: `03_AR_Estimation.R`
Estimates **AR(1) price processes** from monthly median prices: `p'_k = φ_0k + φ_1k * p_k + η_k`

Note: R's `arima()` reports the process mean μ as "intercept", not the regression intercept φ₀. The saved parameters use the correct regression intercept: `φ₀ = μ * (1 - ρ)`. Standard errors for φ₀ are computed via the delta method since φ₀ is a nonlinear function of (μ, ρ).

**Estimated parameters:**
| Parameter | Cigarettes | E-cigs |
|-----------|------------|--------|
| Intercept (φ₀) | ~0.92 (SE via delta method) | ~0.34 (SE via delta method) |
| AR(1) coef (φ₁) | ~0.86 | ~0.88 |

Also estimates the **shock covariance matrix** Σ (prices are correlated across categories).

**Outputs:**
- `Median_Per-Unit_Monthly_Prices.csv`
- `AR_Parameters/AR_Parameters_Phi.csv` (AR coefficients)
- `AR_Parameters/AR_Parameters_Sigma.csv` (shock covariance matrix)

### Script 4: `04_State_Transitions.R`
Simulates **price state transitions** using Halton sequences for numerical integration.

**Process:**
1. Creates all 10² = 100 possible (cig, ecig) price combinations from the pricing grid
2. Generates R=200 Halton draws with correlated shocks via Cholesky decomposition (L where LL' = Σ)
3. For each price vector m and draw r: `p'_k = φ_0k + φ_1k * p_mk + η_rk`

**Outputs:**
- `Halton_Draw_Shocks.csv` (correlated normal shocks)
- `Halton_Draw_Transitions.csv` (100 × 200 = 20,000 simulated next-period prices)

These transitions are used to compute expected continuation values in the second stage via simulation.

### Script 5: `05_TYA_State_Transitions.R`
Computes TYA (teen/young adult) state transition probabilities from household member ages. Expands the binary TYA indicator {0, 1} to a 4-state variable that captures proximity to transitions, enabling identification of β (present bias) in the Beta estimation variant.

**Age definitions (from HMS):**
- Teen: 13-18 (inclusive)
- Young adult: 19-25 (inclusive)
- TYA: 13-25, ages out at 26

**4-state TYA classification:**
| State | Label | Condition |
|-------|-------|-----------|
| 1 | No TYA, stable | No TYA member AND oldest child ≤ 10 (or no children) |
| 2 | No TYA, approaching | No TYA member BUT oldest child is 11-12 |
| 3 | TYA present, stable | Youngest TYA member ≤ 23 |
| 4 | TYA present, ending soon | Youngest TYA member ≥ 24 |

**Process:**
1. Loads raw panel data with household member ages (heads + 7 non-head members)
2. Computes youngest TYA-age member (13-25) and oldest pre-teen child (0-12) per household-month
3. Classifies each household-month into one of 4 TYA states based on proximity to TYA transition
4. Computes a 4×4 row-stochastic monthly transition probability matrix from observed state-to-state transitions
5. ~2% of observations have a mismatch between `teen_or_young_adult_present` and member ages; the script uses `teen_or_young_adult_present` as authoritative and defaults TYA=1 with missing youngest age to state 3

**Outputs** (to `.../Dynamic_Model/Data/`):
- `TYA_States.csv` — Per household-month TYA state assignments (single column: `tya_state`)
- `TYA_Transition_Matrix.csv` — Monthly transition probabilities in long format (from, to, prob)

## Second Stage Estimation (`02_Second_Stage_Estimation/`)

The second stage is implemented in **Julia** and estimates the structural parameters via value function iteration (VFI) and maximum likelihood.

### Script 1: `01_Functions.jl`

Contains all functions and the objective function, organized by execution order. Functions are ordered to match the sequence they are called in `02_Estimation.jl`.

**Preliminaries:**

Packages (CSV, DataFrames, Optim, Statistics, ForwardDiff) are installed if missing and loaded at the top level via `using`. Also loads LinearAlgebra, Printf, Dates from the standard library. Threading is used in the VFI solver — start Julia with `julia -t auto` or set `"julia.NumThreads": "auto"` in VS Code.

| Function | Purpose |
|----------|---------|
| `set_wd(hpc)` | Sets working directory to `.../Dynamic_Model/Data/` (local). Not used by newer scripts which `cd` directly. |

**Logging:**

| Variable / Function | Purpose |
|----------|---------|
| `log_io` | Global log file handle (`IO` or `nothing`). Each calling script sets this once before logging (e.g., `log_io = open("Estimation_Log.txt", "w")`). Shared across all scripts — estimation, MC, validation, and counterfactual all use this single handle. |
| `est_eval_count` | Global objective evaluation counter. Reset to 0 before each estimation run. |
| `ra_outer_try`, `ra_inner_run` | Global optimizer phase tracking (updated by `random_amoeba`). `ra_outer_try`: current outer try (1 to L). `ra_inner_run`: current inner run (1 to M), or 0 for the long convergence run. Used by warm-start phase detection. |
| `V_warm_est`, `last_ra_phase_est` | Warm-start state for `objective()`. `V_warm_est` stores the converged V from the previous VFI solve within a NM run. `last_ra_phase_est` tracks `(outer_try, inner_run)`; V resets when phase changes. Only active when `WARM_START = true`. |
| `log_msg(msg)` | Prints `msg` to stdout and writes it to `log_io` (if open). Every write is followed by `flush`. |

**State Spaces and Choices:**
| Function | Purpose |
|----------|---------|
| `get_fixed_parameters()` | Returns `(ψ, β, δ)` where ψ is fixed at 0.68 (overridden by optimizer when ESTIMATE_PSI = true), β=1.0, δ=0.99 |
| `get_addiction_space(ψ)` | Creates normalized addiction grid: N_A=20 points from 0 to 1 |
| `get_product_choices()` | Loads product choice matrix J, returns (N_HHT, N_J, J) |
| `get_HH_choices(J)` | Converts choice matrix J to choice vector y (y[i] = chosen alternative index) |
| `get_category_choices()` | Loads category choice matrix K, returns (N_K, K) |

**Alternative-Level Vectors:**
| Function | Purpose |
|----------|---------|
| `get_consumption(N_J)` | Returns separate q_cig, q_ecig, q_bundle vectors indexed by alternative j, plus counts per category (N_cig, N_orig_ecig, N_non_fda_flav_ecig, N_fda_flav_ecig, N_bundle). For 6 bundles (2 orig + 2 non-FDA flav + 2 FDA flav), reads columns like `bundle_orig_lo_cig`, `bundle_non_fda_flav_lo_ecig`, etc. Also returns q_cig_max, q_ecig_max, q_bundle_max for rescaling. |
| `get_nicotine(N_J)` | Returns nicotine vector n[j] (mg absorbed) for each alternative; bundles sum cig + ecig nicotine using columns with `_cig_nic` and `_ecig_nic` suffixes. Also returns n_max for rescaling. |
| `get_category_index(N_J, N_cig, N_orig_ecig, N_non_fda_flav_ecig, N_fda_flav_ecig)` | Returns cat_idx[j] mapping each alternative to its category (0=outside, 1=cig, 2=orig ecig, 3=non-FDA flav ecig, 4=FDA flav ecig, 5=orig bundle, 6=non-FDA flav bundle, 7=FDA flav bundle). Handles 2 orig + 2 non-FDA flav + 2 FDA flav bundles. |
| `get_fda_flavored_indicator(cat_idx)` | Returns boolean vector; true for FDA flavored ecig (cat=4) and bundle with FDA flavored ecig (cat=7) |
| `get_flavored_indicator(cat_idx)` | Returns boolean vector; true for any flavored alternative (cat ∈ {3, 4, 6, 7}). Union of non-FDA and FDA flavored. Used by `get_flow_utility()` for λ₁, λ₂ terms. |
| `get_price_ratios(N_J, N_cig, N_orig_ecig, N_non_fda_flav_ecig, N_fda_flav_ecig, q_cig, q_ecig)` | Returns price ratio vectors for quantity discount adjustment. Price per unit varies by bin size within each category. |

**Demographics:**
| Function | Purpose |
|----------|---------|
| `get_teen_young_adult()` | Loads binary youth indicator, returns (N_HH, tya). Used by static logit only. |
| `get_tya_state(tya)` | Maps binary TYA indicator to state index (0 → 1, 1 → 2). Used by static logit only. |
| `get_tya_states()` | Loads 4-state TYA classification from `TYA_States.csv` (produced by `05_TYA_State_Transitions.R`). Returns `tya_state` vector with values 1-4. Used by dynamic estimation, SE, model validation, counterfactual, and MC. |
| `get_tya_transitions()` | Loads 4×4 monthly TYA transition matrix from `TYA_Transition_Matrix.csv`. Returns row-stochastic matrix Π where Π[s, s'] = P(TYA' = s' \| TYA = s). Used by all VFI calls. |

**Price Space:**
| Function | Purpose |
|----------|---------|
| `get_pricing_spaces()` | Loads 10-point price grid per category, returns (N_P, P) |
| `get_pricing_spaces_combination(N_K, N_P, P)` | Creates N_P^2=100 (cig x ecig) price combinations |
| `get_expenditures(N_J, N_Pcomb, q_cig, q_ecig, q_cig_max, q_ecig_max, Pcomb, ratio_cig, ratio_ecig)` | Computes E[p,j] = p_cig(p)*q_cig[j] + p_ecig(p)*q_ecig[j] for all price-alternative pairs, standardized by E_max. Also returns E_max for rescaling. |
| `get_transitions(N_K)` | Loads Halton draw price transitions: M x R x 2 array |
| `precompute_price_transitions(N_P, P, T)` | Pre-computes bilinear interpolation brackets and weights for predicted next-period prices from Halton draws. For each (price state, draw) pair, clamps predictions to grid bounds, finds brackets on each category's 1D grid via binary search. Returns 6 matrices (M × R): `p_cig_lo`, `p_cig_hi`, `p_cig_w`, `p_ecig_lo`, `p_ecig_hi`, `p_ecig_w`. Called once since price transitions don't change across VFI iterations. |

**Household State Trajectories:**
| Function | Purpose |
|----------|---------|
| `map_prices_to_grid(N_P, P, Pcomb)` | Maps observed household prices to nearest combined price grid index using median per-category prices. Reads 6 cig price columns (`cig_1to2_p`, `cig_3to10_p`, etc.) and 6 ecig price columns. Returns `(p_state, p_continuous)` where `p_state` is the nearest grid index and `p_continuous` is an N × 2 matrix of actual continuous (cig, ecig) prices for likelihood interpolation. |

**Estimation:**
| Function | Purpose |
|----------|---------|
| `addiction_evolution(ψ, a, n)` | Addiction law of motion in normalized units: ã' = (1-ψ)ã + ψ·n |
| `get_flow_utility(θ, N_J, N_A, N_Pcomb, A, q_cig, q_ecig, q_bundle, n, is_flavored, is_fda_flavored, cat_idx, E)` | Pre-computes flow utility for all (tya, alternative, addiction, price) states. Takes 13-element θ (excludes ψ). Returns 4D array U[tya_idx, j, a_idx, p_idx] of dimension 4 × N_J × N_A × N_Pcomb. TYA indicator: states 1,2 → tya=0; states 3,4 → tya=1. Flavor terms: λ₁,λ₂ apply to all flavored (is_flavored); λ₃,λ₄ are additional for FDA-authorized (is_fda_flavored). Splits base utility (consumption, addiction, expenditure, fixed effects) from TYA-dependent flavor terms for efficiency. |
| `precompute_addiction_transitions(N_J, N_A, ψ, A, n)` | Pre-computes interpolation brackets (a_lower, a_upper) and weights (a_weight) for all (alternative, addiction state) pairs using binary search |
| `get_initial_addiction_stock(ψ, A, n, y)` | Estimates initial addiction stock per household via fixed-point iteration: simulate forward from a₀, set a₀ to terminal value, repeat until convergence. Returns `(a0, max_iters)` where `max_iters` is the maximum iterations across all households. |
| `simulate_addiction_trajectories(N_A, ψ, A, n, y, a0)` | Simulates addiction forward from estimated a₀ using observed choices. Returns `(a_state, a_continuous)` where `a_state` is the nearest grid index and `a_continuous` is the actual continuous addiction level for likelihood interpolation. |

**Value Function Iteration:**
| Function | Purpose |
|----------|---------|
| `logsumexp(v)` | Numerically stable log-sum-exp: computes `log(Σ exp(v))` by factoring out the maximum to prevent overflow. Used to aggregate choice-specific values into the ex-ante value function (closed-form expected maximum from the Type I extreme value logit error assumption). |
| `solve_vfi(N_J, N_A, N_P, N_Pcomb, β, δ, U, a_lower, a_upper, a_weight, p_cig_lo, p_cig_hi, p_cig_w, p_ecig_lo, p_ecig_hi, p_ecig_w, Π; V_init, ε, max_iter, verbose)` | **Naive** VFI with 4-state TYA transitions via Π. Accepts optional `V_init` for warm-starting (defaults to `nothing` → zeros). Uses `copyto!` for type-stable initialization. Bellman uses δ only: V_choice = U + δ·EV. Continuation value integrates over TYA transitions: EV = Σ_{tya'} Π[tya, tya'] · EV_price. After convergence, computes V_decision = (1-β)·U + β·V_choice = U + βδ·EV. When β=1, V_decision = V_choice. Aggregates via logsumexp. Returns `(V, V_decision, n_iter, converged)`. Not currently called by any code — retained for reference/comparison. |
| `solve_vfi_sophisticated(...)` | **Sophisticated** VFI with 4-state TYA transitions. Same signature as `solve_vfi` (including Π and V_init). Accepts optional `V_init` for warm-starting (defaults to `nothing` → zeros). Uses `copyto!` for type-stable initialization. Maintains two 4D arrays: V_d = U + βδ·EV (decision utility) and V_e = U + δ·EV (experienced utility). Continuation value integrates over TYA transitions via Π. Aggregation: V = Σ_j softmax(V_d)_j · V_e_j + H(softmax(V_d)) where H is entropy. Post-convergence: V_decision = V_d (no transformation). When β=1, V_d = V_e and aggregation = logsumexp — numerically identical to naive. All estimation, MC, model validation, and counterfactual code calls this function. |

**Log-Likelihood:**
| Function | Purpose |
|----------|---------|
| `log_likelihood(V_choice, N_J, N_P, A, P, y, tya_state, a_continuous, p_continuous)` | Computes the sample log-likelihood ℓ(θ) = Σᵢ log P(yᵢ \| xᵢ; θ) by trilinearly interpolating V_choice at each observation's continuous state. For each observation: (1) finds linear interpolation brackets on the addiction grid, (2) finds bilinear interpolation brackets on the 2D price grid, (3) interpolates V_choice for all N_J alternatives, (4) computes log P(yᵢ) = V_choice_interp[yᵢ] - logsumexp(V_choice_interp). Returns a scalar log-likelihood. |

**Sampling:**
| Function | Purpose |
|----------|---------|
| `categorical_sample(probs)` | Draws a single sample from a categorical distribution using the CDF method. Returns sampled category index. |

**Interpolation:**
| Function | Purpose |
|----------|---------|
| `interpolate_v_choice(V_choice, tya_idx, a, obs_cig, obs_ecig, N_J, N_P, A, P)` | Trilinearly interpolates V_choice at a continuous state (addiction, cig price, ecig price) for all N_J alternatives. Same bracket/weight logic as `log_likelihood`. Used by model validation, counterfactual simulation, and MC data simulation. |

**Parameter Bounds:**
| Variable / Function | Purpose |
|----------|---------|
| `θ_lower_bound`, `θ_upper_bound` | Global constant vectors (13 elements) defining economic parameter bounds (standardized units). μ ≥ 0; γ, ω ≤ 0; α, λ, and ξ unconstrained. Used by both `objective()` and `objective_mc()`. |
| `check_parameter_bounds(θ_vec, param_names)` | Returns `(in_bounds::Bool, violations::String)`. Checks all elements of θ against the bound vectors and reports which parameters violate constraints. |

**Optimization:**
| Function / Type | Purpose |
|----------|---------|
| `SimplexWithAdd` | Custom `Optim.Simplexer` subtype. Constructs a (D+1)-vertex Nelder-Mead simplex around the starting point by adding deviation `add[d]` along each coordinate direction d, where D is the number of parameters. |
| `random_amoeba(objective, starting_param, add, L, M, inner_iter)` | Multi-start Nelder-Mead optimizer. For each of L outer tries: (1) runs M short Nelder-Mead runs (`inner_iter` iterations each, `f_abstol=1e-4`) with randomized simplex perturbations, (2) runs one long Nelder-Mead run (up to 2,500 iterations, `f_abstol=1e-3`) to fully converge, (3) updates the global best if improved, (4) randomly reinitializes parameters (20% chance original, 40% chance best-so-far, 40% chance current). Short runs may converge before hitting the iteration limit. Logs via the global `log_msg()`. Takes `starting_param` as a NamedTuple and returns `(opt_param, overall_min)`. |

**Objective Function:**
| Variable / Function | Purpose |
|----------|---------|
| `objective(θ_vec)` | Objective function for the optimizer (13-element θ_vec, or more with ESTIMATE_PSI/ESTIMATE_BETA). Increments `est_eval_count` and times each evaluation. **Box constraints:** returns `1e14` penalty if any parameter violates bounds via `check_parameter_bounds`. **ψ handling:** when `ESTIMATE_PSI = false`, uses the global `ψ` (fixed at 0.68 from `get_fixed_parameters()`); when `ESTIMATE_PSI = true`, extracts ψ from θ_vec and recomputes the addiction grid, addiction transition brackets, initial addiction stocks, and addiction trajectories at the candidate ψ each evaluation. For each candidate θ: (1) computes addiction grid from ψ (fixed or extracted), (2) computes flow utility U from the 13 structural elements, (3) computes addiction transition brackets at ψ, (4) when `WARM_START = true`, detects phase changes via `(ra_outer_try, ra_inner_run)` and resets `V_warm_est` on change, then passes it as `V_init` to VFI; when `WARM_START = false`, passes `V_init = nothing` (cold start from zeros), (5) solves VFI with Π for TYA state transitions. **Early-exit:** if VFI did not converge, logs a PENALTY message and returns `1e14`. Otherwise: (6) stores converged V in `V_warm_est` if warm-starting, (7) computes addiction trajectories at ψ, (8) evaluates log-likelihood via trilinear interpolation, (9) logs eval number, LL, VFI iters, elapsed time, and θ vector. Returns the negative log-likelihood. Uses global data objects set in `02_Estimation.jl`. |

**Alternative Ordering (j = 1, ..., 40):**
| Index | Alternative |
|-------|-------------|
| j = 1 | Outside option (zero consumption) |
| j = 2:13 | 12 cigarette quantity bins |
| j = 14:20 | 7 original e-cigarette bins |
| j = 21:27 | 7 non-FDA flavored e-cigarette bins |
| j = 28:34 | 7 FDA flavored e-cigarette bins |
| j = 35:36 | 2 original bundles (lo/hi cig) |
| j = 37:38 | 2 non-FDA flavored bundles (lo/hi cig) |
| j = 39:40 | 2 FDA flavored bundles (lo/hi cig) |

**Structural Parameters (θ) - 13 estimated parameters:**
```
α_C   = cigarette consumption utility
α_E   = e-cig consumption utility
α_CE  = bundle consumption utility
λ_1   = flavor baseline effect (all flavored products)
λ_2   = flavor × teen/young adult interaction (all flavored products)
λ_3   = FDA flavor baseline effect (additional for FDA-authorized)
λ_4   = FDA flavor × teen/young adult interaction (additional for FDA-authorized)
μ     = reinforcement effect (addiction × nicotine intake)
γ     = addiction level effect (withdrawal cost)
ω     = expenditure coefficient (price sensitivity)
ξ_C   = cigarette fixed effect
ξ_E   = e-cig fixed effect
ξ_CE  = bundle fixed effect
```

**Fixed Parameters (not estimated):**
```
ψ     = 0.68   (addiction decay rate; estimated when ESTIMATE_PSI = true)
β     = 1.0     (present bias; 1 = standard exponential discounting)
δ     = 0.99    (monthly discount factor)
```

**Flow Utility Specification:**
```
u(j,a,p,tya) = α_C·q_cig[j] + α_E·q_ecig[j] + α_CE·q_cig[j]·q_ecig[j]
             + γ·a + μ·a·n[j]
             + ω·E[p,j]
             + ξ_k
             + 𝟙[flavored]·(λ_1 + λ_2·𝟙[tya])
             + 𝟙[FDA flav]·(λ_3 + λ_4·𝟙[tya])
```
Where:
- q_cig[j], q_ecig[j] = cigarette and e-cigarette consumption for alternative j
- n[j] = nicotine intake for alternative j (standardized)
- a = addiction state
- E[p,j] = expenditure (p_cig·q_cig[j] + p_ecig·q_ecig[j])
- ξ_k = fixed effect (ξ_C for cig, ξ_E for ecig categories 2-4, ξ_CE for bundle categories 5-7)
- tya = teen/young adult indicator (TYA states 3,4 → tya=1; states 1,2 → tya=0)
- flavored = any flavored ecig or bundle (cat ∈ {3, 4, 6, 7}); λ_1,λ_2 apply to all flavored
- FDA flav = FDA flavored ecig (cat=4) or bundle with FDA flavored ecig (cat=7); λ_3,λ_4 are additional

The reinforcement term μ·a·n[j] captures the interaction between addiction stock and current nicotine intake — higher addiction increases the marginal utility of nicotine-delivering alternatives. For the outside option (j = 1), n[j] = 0 so the reinforcement term is zero.

**State Space:**
- 4 teen/young adult states (1=no TYA stable, 2=no TYA approaching, 3=TYA stable, 4=TYA ending)
- 20 addiction states (normalized grid from 0 to 1)
- 100 price states (10x10 cig x ecig grid)
- 40 product alternatives

**Value Function — Naive Agent (`solve_vfi`, VFI uses δ only):**

A naive agent has present bias (β < 1) but doesn't realize their future self will also be present-biased. They believe their future self discounts exponentially with δ, so VFI solves:
```
V_choice[tya, j, a, p] = u(j,a,p,tya) + δ·E[V(tya', a', p') | a, p, j]
V[tya, a, p] = log( Σ_j exp(V_choice[tya, j, a, p]) )   (logsumexp)
```
After convergence, β is applied post-hoc to get decision utility:
```
V_decision = (1 - β)·U + β·V_choice = U + βδ·EV
```
When β = 1 (standard exponential), V_decision = V_choice.

**Value Function — Sophisticated Agent (`solve_vfi_sophisticated`):**

A sophisticated agent knows their future self will also be present-biased. The continuation value must account for the fact that the future self chooses according to decision utility V_d (with present bias), not experienced utility V_e.

Two choice-specific arrays are maintained each VFI iteration:
```
V_d[tya, j, a, p] = U + β·δ·EV   (decision utility — what the future self uses to choose)
V_e[tya, j, a, p] = U + δ·EV     (experienced utility — what actually accrues)
```

The future self chooses with probabilities p_j = softmax(V_d)_j, but the actual payoffs are V_e. So the continuation value is:
```
V[tya, a, p] = Σ_j p_j · V_e_j + H(p)
```
where H(p) = −Σ_j p_j·log(p_j) is the entropy from the T1EV shocks (the option value of randomness). After convergence, V_decision = V_d directly (no post-hoc transformation).

**Why sophisticated = naive when β = 1:** When β = 1, V_d = V_e, so p_j = softmax(V_e)_j, and Σ_j p_j·V_e_j + H(p) = logsumexp(V_e). The aggregation collapses to the naive logsumexp, and V_decision = V_d = V_choice. All outputs are numerically identical.

**Key economic difference:** The naive agent's continuation value is too optimistic — they think their future self will be a patient δ-discounter. The sophisticated agent correctly predicts their future self's present bias, so the continuation value is lower (future self's impatience destroys some value). Choice probabilities and the log-likelihood use V_decision in both cases.

### ESTIMATE_BETA Flag (replaces former `01_Functions_Beta.jl`)

The former `01_Functions_Beta.jl` has been **merged into `01_Functions.jl`** via the `ESTIMATE_BETA` flag. Set `ESTIMATE_BETA = true` in the calling script before `include("01_Functions.jl")` to estimate β (present bias) as the last structural parameter. When `ESTIMATE_BETA = false` (default), β is fixed at 1.0.

**What ESTIMATE_BETA controls:**
| Area | `ESTIMATE_BETA = false` (Base) | `ESTIMATE_BETA = true` (Beta) |
|------|-------------------------------|-------------------------------|
| `get_fixed_parameters()` | Returns (ψ=0.68, β=1.0, δ) | Returns (ψ=0.68, β=1.0, δ) — β ignored by optimizer |
| β in objective | Uses global β from `get_fixed_parameters()` | Extracts β = θ_vec[end]; passes θ_vec[1:end-1] to `get_flow_utility()` |
| Parameter bounds | 13 elements | 14 elements (β ∈ [0.01, 1.00]) |
| Starting values | 13 parameters | 14 parameters (β = 0.90 appended) |
| Output file naming | `Dynamic_Model_*` | `Dynamic_Model_Beta_*` |

### ESTIMATE_PSI Flag

Controls whether ψ (addiction decay rate) is estimated or fixed at 0.68. Set `ESTIMATE_PSI = true` in the calling script before `include("01_Functions.jl")`. When `true`, ψ is estimated as a structural parameter. When `false` (default), ψ is fixed at 0.68.

**Parameter vector ordering:**
- Base: `[13 structural]`
- PSI only: `[13 structural, ψ]`
- BETA only: `[13 structural, β]`
- Both: `[13 structural, ψ, β]` (β always last when estimated)

**What ESTIMATE_PSI controls:**
| Area | `ESTIMATE_PSI = false` (Base) | `ESTIMATE_PSI = true` (Psi) |
|------|-------------------------------|-------------------------------|
| `get_fixed_parameters()` | Returns (ψ=0.68, β, δ) | Returns (ψ=0.68, β, δ) — ψ ignored by optimizer |
| ψ in objective | Uses global ψ from `get_fixed_parameters()` | Extracts ψ from θ_vec; recomputes addiction transitions and trajectories |
| Parameter bounds | 13 elements (or 14 with β) | +1 element (ψ ∈ [0.01, 1.00]) |
| Starting values | 13 parameters (or 14 with β) | +1 parameter (ψ = 0.50 appended) |
| Output file naming | `Psi_0.68` in tag | `Psi_Estimated` in tag |

### WARM_START Flag

Controls whether VFI reuses the previous evaluation's converged V as the initial guess within a Nelder-Mead run. Set `WARM_START = true` in the calling script before `include("01_Functions.jl")` to enable. Default is `false` (cold start from zeros each evaluation).

**How it works:**
- Within a single NM run (fixed outer try L and inner run M), θ changes only slightly between evaluations, so the previous V is nearly correct
- V is reset to `nothing` (→ zeros) whenever `(ra_outer_try, ra_inner_run)` changes — i.e., at the start of each new outer try, inner run, or long convergence run
- Only converged V is stored; unconverged V (penalty case) is not saved
- Uses `copyto!` (not `copy`) inside VFI to guarantee Julia type stability — using `copy(V_init)` with a `Union{Array, Nothing}` type caused ~100x slowdown due to type instability in the VFI inner loop

**Performance impact:** Reduces VFI from ~948 iterations (cold start) to ~7-50 iterations (warm start) for sequential evaluations within a NM run. First evaluation of each run still does ~948 iterations.

**Correctness:** The contraction mapping has a unique fixed point regardless of starting V, so warm-start only affects iteration count, not the result. NLL at the same θ is identical with warm-start on vs off.

**What WARM_START controls:**
| Area | `WARM_START = false` (Cold) | `WARM_START = true` (Warm) |
|------|---------------------------|---------------------------|
| `objective()` | `V_init = nothing` (zeros each eval) | `V_init = V_warm_est` (previous converged V) |
| `objective_mc()` | `V_init = nothing` (zeros each eval) | `V_init = V_warm` (previous converged V) |
| Globals in `01_Functions.jl` | `V_warm_est`, `last_ra_phase_est` unused | Phase detection resets V on `(L, M)` change |
| Globals in `01_MC_Simulation_Functions.jl` | `V_warm`, `last_ra_phase` unused | Phase detection resets V on `(L, M)` change |

**VFI with TYA transitions:** The continuation value in `solve_vfi` and `solve_vfi_sophisticated` integrates over both price transitions (via Halton draws) and TYA state transitions (via Π):
```
EV[tya, a', p'] = Σ_{tya'} Π[tya, tya'] · Σ_r (1/R) · V[tya', a', p'_r]
```
This enables the model to distinguish present bias from exponential discounting: anticipated changes in TYA status (e.g., a child approaching age 13) affect continuation values differently under β < 1 vs β = 1.

### Script 1c: `02_Estimation_Slurm.sb`

Slurm batch script for running `02_Estimation.jl` on the University of Arizona HPC.

**Resources:** 1 node, 1 task, 16 CPUs (for VFI threading), 5 GB per CPU (80 GB total), 16 hour time limit, standard partition.

**Output:** Slurm log written to `.../Dynamic_Model_Results/Dynamic_Model_Estimation_Slurm_Log_<job_id>.out`.

### Script 2: `02_Estimation.jl`

Main estimation script. Sets `ESTIMATE_BETA` flag (default `false`) and `WARM_START` flag (default `true`), includes `01_Functions.jl`, loads all data, sets fixed parameters (ψ, β=1.0, δ=0.99 via `get_fixed_parameters()`), and runs the multi-start Nelder-Mead optimizer. When `ESTIMATE_BETA = true`, β is appended as the 14th parameter with starting value 0.90, and output files are prefixed with "Beta_".

**Starting values:**
- Static logit estimates from `03_Static_Logit.jl` provide starting values for shared parameters
- Since static logit used unstandardized data, estimates must be **converted to standardized units**:
  - `α_C_std = α_C_orig × q_cig_max`
  - `α_E_std = α_E_orig × q_ecig_max`
  - `α_CE_std = α_CE_orig × q_bundle_max`
  - `ω_std = ω_orig × E_max`
  - `λ_1, λ_2, λ_3, λ_4, ξ_C, ξ_E, ξ_CE`: no conversion (indicators/additive)
- μ and γ have no static counterpart; initialized at 0.10 and -0.10 (scaled for ã ∈ [0,1])
- λ_3 and λ_4 (FDA flavor parameters) initialized from static logit estimates
- ψ is fixed at 0.68 (overridden when ESTIMATE_PSI = true)

**Estimation settings:**
- Simplex deviations scaled to ~50-100% of starting parameter magnitudes; μ and γ get 100% deviation
- Optimizer: L=50 outer tries, M=20 inner tries, 100 iterations per inner run

**Rescaling estimates to original units:**
After estimation, divide by max values to get interpretable units:
- `α_C_orig = α_C_std / q_cig_max` → utils per pack
- `α_E_orig = α_E_std / q_ecig_max` → utils per mL
- `α_CE_orig = α_CE_std / q_bundle_max` → utils per (pack × mL)
- `ω_orig = ω_std / E_max` → utils per dollar
- `μ_orig = μ_std × ψ / n_max²` → utils per (mg addiction × mg nicotine). μ multiplies ã and n_std, both standardized. ψ is fixed at 0.68.
- `γ_orig = γ_std × ψ / n_max` → utils per mg of addiction stock. γ multiplies only ã. ψ is fixed at 0.68.
- `λ_1, λ_2, λ_3, λ_4, ξ_C, ξ_E, ξ_CE`: no rescaling needed

**Output files** (written to `.../Dynamic_Model_Results/`):
- `Dynamic_Model_Estimation_Log_<timestamp>.txt` — Full estimation progress: VFI convergence info, eval-by-eval LL and θ, optimizer restarts, timing
- `Dynamic_Model_Estimates.csv` — CSV estimated parameters (read by `03_Standard_Errors.jl`)
- `Dynamic_Model_Outer_Try_Params_<timestamp>.csv` — Best parameters from each outer try
- `Dynamic_Model_Inner_Try_Params_<timestamp>.csv` — Best parameters from each inner run

### Script 3: `03_Standard_Errors.jl`

Computes standard errors for the dynamic model parameter estimates via finite differences on the full objective function. Loads ψ from `get_fixed_parameters()` and uses it for the addiction grid. θ_hat contains 13 structural parameters (ψ is not estimated).

**Process:**
1. Loads the same data as `02_Estimation.jl`; loads ψ from `get_fixed_parameters()`
2. Reads θ_hat (13 parameters) from `Dynamic_Model_Estimates.csv`
3. Evaluates the objective once at θ_hat (to get `nll_center` for diagonal Hessian entries)
4. Computes the Hessian via central finite differences (h=1e-3), each evaluation re-solves VFI from scratch
5. Inverts the Hessian to get variance-covariance matrix and SEs

**Hessian computation:**
- Diagonal: `H[k,k] ≈ (f(θ+h·e_k) - 2·f(θ) + f(θ-h·e_k)) / h^2` (2 evals per parameter)
- Off-diagonal: `H[k,l] ≈ (f(θ++) - f(θ+-) - f(θ-+) + f(θ--)) / (4h^2)` (4 evals per pair)
- Total: 469 objective evaluations for 13 parameters (each requires full VFI solve)
- Progress logged for every parameter pair with timing

**Output files** (written to `.../Dynamic_Model_Results/`):
- `SE_Log.txt` — Full progress: per-pair Hessian values, eigenvalue diagnostics, results table with SEs and t-statistics
- `Dynamic_Model_Standard_Errors.csv` — CSV estimates and SEs

## MC Simulation (`05_MC_Simulation/`)

### MC Functions: `05_MC_Simulation/01_MC_Simulation_Functions.jl`

Contains functions specific to the Monte Carlo simulation. These handle data simulation from a known DGP and MC-specific versions of estimation functions that work with simulated (in-memory) data rather than CSV files.

**Logging:**

Uses `log_io` and `log_msg()` from `01_Functions.jl` (no separate MC logging).

| Variable | Purpose |
|----------|---------|
| `eval_count` | Global MC evaluation counter. Reset before each replication. |
| `V_warm`, `last_ra_phase` | Warm-start state for `objective_mc()`. `V_warm` stores the converged V from the previous VFI solve within a NM run. `last_ra_phase` tracks `(outer_try, inner_run)`; V resets when phase changes. Only active when `WARM_START = true`. |

**Data Simulation:**
| Function | Purpose |
|----------|---------|
| `simulate_data(V_decision_true, ψ, N_J, N_P, A, P, n, real_p_continuous, real_tya_state, real_hh_codes)` | Design-based MC simulation: conditions on real observables (prices, TYA, panel structure) and only simulates choices. Uses per-observation TYA state (varies by month within households). Two-pass approach: Pass 1 simulates from a₀=0, Pass 2 re-simulates from fixed-point corrected a₀. Returns `(y_sim, tya_state_sim, p_continuous_sim, hh_codes_sim)`. |

**MC Objective Function:**
| Variable / Function | Purpose |
|----------|---------|
| `y_sim`, `tya_state_sim`, `p_continuous_sim`, `hh_codes_sim` | Global variables holding simulated data. Set by the MC loop in `02_MC_Simulation_Array.jl` before each replication's estimation. |
| `objective_mc(θ_vec)` | MC-specific objective function (13-element θ_vec, or more with ESTIMATE_PSI/ESTIMATE_BETA). Same as `objective()` but uses simulated data globals and the base addiction functions (`get_initial_addiction_stock`, `simulate_addiction_trajectories`) from `01_Functions.jl`. When `ESTIMATE_PSI = false`, uses the global `ψ` (fixed at 0.68 from `get_fixed_parameters()`); when `ESTIMATE_PSI = true`, extracts ψ from θ_vec and recomputes addiction grid, transitions, and trajectories at the candidate ψ each eval. When `WARM_START = true`, detects phase changes via `(ra_outer_try, ra_inner_run)` and resets `V_warm` on change, then passes it as `V_init` to VFI; stores converged V after successful solve. Passes Π to `solve_vfi_sophisticated` for TYA state transitions. Increments `eval_count` and times each evaluation. **Box constraints:** returns `1e14` penalty via `check_parameter_bounds` (matching `objective()`). **Early-exit:** if VFI did not converge, logs a PENALTY message and returns `1e14`. Logs eval number, LL, VFI iters, elapsed time, and θ vector. |

### Script 2: `05_MC_Simulation/02_MC_Simulation_Array.jl`

Parallel version of the MC simulation using Slurm job arrays. Each array task runs a **single replication**, reading its replication number `s` from `SLURM_ARRAY_TASK_ID` (or command-line argument for local testing). Seeds RNG with `Random.seed!(s)` for reproducibility. Sets `WARM_START = true` (warm-start VFI within NM runs). Loads ψ from `get_fixed_parameters()`. θ_true (13 parameters) is computed at runtime from static logit estimates converted to standardized units (α_C = α_C_orig × q_cig_max, etc.) with λ_1=0.1554, λ_2=0.6616, λ_3=-0.0975, λ_4=-0.3359, ξ_C=-3.2001, ξ_E=-6.0294, ξ_CE=-5.2096, μ=0.05, γ=-0.10 (scaled for ã ∈ [0,1]). DGP section uses fixed ψ from `get_fixed_parameters()` to create the addiction grid and solve the true VFI with Π. True parameters and log output are rounded to 4 decimal places.

**Starting Values:** Set to 50% of θ_true (`starting_param = map(x -> 0.5x, θ_true)`) with simplex deviations ~50% of absolute value; γ and μ get 100% deviation.

**Usage:**
- HPC: `sbatch 02_MC_Simulation_Array_Slurm.sb` (launches 100 independent tasks)
- Local: `julia 02_MC_Simulation_Array.jl 1` (runs replication 1)

**Output files** (one set per replication, written to `.../MC_Simulation_Results/`):
- `<ss>_MC_Rep_Results_<timestamp>.csv`: Single-row CSV result with NLL and estimated parameters
- `<ss>_MC_Rep_Log_<timestamp>.txt`: Full log for this replication
- `<ss>_MC_Rep_Parameters_<timestamp>.csv`: CSV parameter trace for this replication
- `MC_True_Parameters.csv`: True parameter values in transposed format (parameter names as columns, single row of values rounded to 4 decimal places). Overwritten by each rep, but identical.

### Script 3: `05_MC_Simulation/03_MC_Aggregate_Results.jl`

Aggregates per-replication MC results from job array. Reads all `*_MC_Rep_Results_*.csv` files, checks for missing replications, computes summary statistics (Mean, Bias, Std Dev, RMSE).

**Output files** (written to `.../MC_Simulation_Results/`):
- `MC_Results_Combined_<timestamp>.csv`: All replications combined
- `MC_Summary_<timestamp>.csv`: Summary statistics table

### Script 4: `05_MC_Simulation/04_Two_Param_Profile.jl`

Diagnostic tool for visualizing the likelihood surface. Evaluates NLL over a 2D grid of two parameters while holding all others at truth. Default profiles (α_C, γ) to diagnose the ridge between consumption utility and withdrawal cost. Change `param_idx_1` and `param_idx_2` to profile other pairs. ψ is fixed at 0.68 from `get_fixed_parameters()`. Grid evaluation uses fixed ψ for the addiction grid.

**Output files** (written to `.../MC_Simulation_Results/`):
- `Profile_<name1>_<name2>.csv`: Long-format grid (param1, param2, NLL) for plotting

## MC Simulation Beta / Psi Variants

The `05_MC_Simulation_Beta/` directory has been deleted. MC simulation with `ESTIMATE_BETA = true` and/or `ESTIMATE_PSI = true` is now handled entirely by the flags in `05_MC_Simulation/02_MC_Simulation_Array.jl`. Set the desired flags at the top of the script before running.

**To run MC with ESTIMATE_BETA and/or ESTIMATE_PSI:** Set the flags at the top of `05_MC_Simulation/02_MC_Simulation_Array.jl`, then:
- HPC: `sbatch 05_MC_Simulation/02_MC_Simulation_Array_Slurm.sb`
- Local: `julia 05_MC_Simulation/02_MC_Simulation_Array.jl 1`

**When ESTIMATE_BETA = true:**
- θ_true has 14 parameters (β = 0.95 as present-bias DGP)
- starting_param has 14 parameters (β = 0.90 as offset starting value)
- `objective_mc()` extracts β from θ_vec[end], passes θ_vec[1:end-1] to `get_flow_utility()`

**When ESTIMATE_PSI = true:**
- θ_true includes ψ as a parameter
- `objective_mc()` extracts ψ from θ_vec; recomputes addiction transitions and trajectories

Output directories are named using `<psi_tag>_<beta_tag>` conventions (see Output Directory Naming below).

## Static Logit (`../Structural_Model_Motivation/03_Static_Logit.jl`)

Static conditional logit model motivating the dynamic model. Drops the dynamic addiction terms (μ, γ) from the flow utility and does not require VFI. Prices enter as continuous observed values rather than discretized grid states. The state dependence parameter ρ motivates the dynamic model: significant ρ means past choices predict current behavior (addiction/habit persistence) which a static model cannot properly account for.

**Includes:** `../Dynamic_Model/02_Second_Stage_Estimation/01_Functions.jl` (local) or `../Dynamic_Model/01_Functions.jl` (HPC). Uses `hpc` flag to toggle paths.

**Data Loading:**
- Same data as `02_Estimation.jl` (product choices, consumption, category index, split flavored indicators, TYA, continuous prices) but does NOT load addiction grid, nicotine, or price transitions
- Additionally loads `Lagged_Category_Choice.csv` for state dependence term
- Sample restricted to observations with non-missing lagged choice (drops first month per household)

**Pre-computed Matrices:**
| Matrix | Dimension | Description |
|--------|-----------|-------------|
| `E_obs[i, j]` | N_obs × N_J | Current expenditure: `p_cig[i] · q_cig[j] + p_ecig[i] · q_ecig[j]` |
| `lag_match[i, j]` | N_obs × N_J | 1 if alternative j's category matches the household's lagged category choice |

**Structural Parameters (θ) — 12 parameters:**
```
α_C   = cigarette consumption utility (per pack)
α_E   = e-cig consumption utility (per mL)
α_CE  = bundle interaction utility (per pack·mL)
λ_1   = flavor baseline effect (all flavored products)
λ_2   = flavor × teen/young adult interaction (all flavored products)
λ_3   = FDA flavor baseline effect (additional for FDA-authorized)
λ_4   = FDA flavor × teen/young adult interaction (additional for FDA-authorized)
ρ     = state dependence (lagged category match)
ω     = expenditure coefficient (price sensitivity)
ξ_C   = cigarette fixed effect
ξ_E   = e-cig fixed effect
ξ_CE  = bundle fixed effect
```

**Negative Log-Likelihood:**

`neg_log_likelihood(θ_vec, N_obs, N_J, tya, y, q_cig, q_ecig, q_bundle, is_flavored, is_fda_flavored, lag_match, E_obs, fe_C, fe_E, fe_CE)` — Takes data arrays as arguments for type stability with ForwardDiff. A single-argument wrapper `nll = θ -> neg_log_likelihood(θ, ...)` is used by the optimizer and Hessian computation.

**Estimation (three optimizers for comparison):**
1. **L-BFGS with autodiff** — `optimize(nll, ...; autodiff = :forward)` with ForwardDiff. JIT warmup call before timed optimization. Callback logs every 10 iterations. Converges in ~94 iterations, ~36s.
2. **Nelder-Mead (Random Amoeba)** — `random_amoeba(nll, ...)` with L=3, M=2, inner_iter=500.

**Standard Errors (three methods for comparison):**
1. **Autodiff Hessian** — `ForwardDiff.hessian(nll, θ_hat)` gives exact second derivatives.
2. **Finite differences** — Central differences on `nll` with h=1e-3. Diagonal and off-diagonal entries computed separately.
3. **Comparison table** — Shows SE differences between autodiff and finite differences.

**Output** (written to `.../Static_Logit_Results/`):
- `Static_Logit_Estimation_Log_<timestamp>.txt` — L-BFGS iteration progress, Nelder-Mead progress, estimates with SEs and t-statistics from all methods, comparison tables (timestamped to avoid overwriting previous runs)

## Static Nested Logit (`../Structural_Model_Motivation/04_Static_Nested_Logit.jl`)

Static nested logit model that relaxes IIA as a **robustness check** for the flavor ban counterfactual. Identical to `03_Static_Logit.jl` except choice probabilities use nested logit with 4 nests and 1 common nesting parameter σ. Adds 1 parameter (σ_raw) for a total of 13.

**Nesting structure:**
| Nest | Alternatives | Description |
|------|-------------|-------------|
| 1 | j = 1 | Outside option (singleton) |
| 2 | j = 2:13 | Cigarettes (12 alts) |
| 3 | j = 14:34 | E-cigs: orig + non-FDA flav + FDA flav (21 alts) |
| 4 | j = 35:40 | Bundles: orig + non-FDA flav + FDA flav (6 alts) |

**σ parameterization:** `σ = 1/(1+exp(-σ_raw))` ∈ (0,1). When σ = 0: standard logit. When σ > 0: within-nest substitution ratio = 1/(1-σ).

**Key result (February 2026):** σ̂ = 0.12 (SE = 0.017, t = 7.29). Statistically significant but economically modest — within-nest substitution is only ~14% stronger than across-nest. This supports using standard logit in the dynamic model. The flavor parameters (λ_1, λ_2) absorbed some within-nest substitution in the standard logit and dropped ~30-38% in the nested logit.

**Decision:** σ̂ = 0.12 is too small to warrant modifying the dynamic model (VFI, likelihood, counterfactual, MC code). The static nested logit is reported as evidence that IIA is approximately satisfied.

**Output** (written to `.../Static_Logit_Results/`):
- `Static_Nested_Logit_Estimation_Log_<timestamp>.txt`

## Reduced-Form AR(1) Estimation of ψ (`../Structural_Model_Motivation/05_Reduced_Form_Psi.R`)

Estimates the addiction decay rate ψ from the reduced-form persistence of nicotine consumption via AR(1) regression. Written in R using data.table and fixest. Reads the same processed CSVs from `.../Dynamic_Model/Data/` used by the structural model.

**Main specification (household FE + Nickell correction):**
```
nicotine ~ lag_nicotine | household_code + purchase_month
```
Estimated via `fixest::feols` with household and month fixed effects, clustered SEs at household level. The raw FE estimate ρ̂_FE is biased downward by Nickell bias in short panels, so a Nickell correction is applied: `ρ_corrected = (ρ_FE × (T_bar - 1) + 1) / (T_bar - 2)` where T_bar is the average number of time periods per household. **ψ̂ = 1 - ρ̂_corrected**.

**Models estimated:**
1. Mundlak CRE
2. Mundlak CRE + month fixed effects
3. Separate cig and ecig AR(1)s (product-specific persistence)
4. Household FE + month FE with Nickell bias correction (main specification)

**Key results:** ρ̂_corrected ≈ 0.3216 (household FE + Nickell correction), implying ψ̂ ≈ 0.6884. The Mundlak/CRE specification gives ρ̂ ≈ 0.52 (ψ̂ ≈ 0.48), but the household FE + Nickell correction is preferred because it directly controls for time-invariant unobserved heterogeneity.

**Why household FE + Nickell correction:** With a lagged dependent variable + household FE, demeaning creates mechanical negative correlation between the demeaned lag and demeaned error (Nickell, 1981), biasing ρ̂ downward. The Nickell correction formula adjusts for this bias analytically, recovering a consistent estimate of ρ. This approach is preferred over Mundlak/CRE because it directly absorbs all time-invariant household heterogeneity via fixed effects rather than relying on household means as proxies.

## Data Standardization

**All continuous variables entering the utility function are STANDARDIZED by dividing by their maximum value.**

This is done automatically in the data-loading functions:
- `get_consumption()` → returns standardized q_cig, q_ecig and their raw max values
- `get_nicotine()` → returns standardized n and raw n_max
- `get_addiction_space(ψ)` → normalized addiction grid A ∈ [0, 1] (ã = a·ψ, steady state ã = n ∈ [0, 1])
- `get_expenditures()` → computes E from raw consumption, then standardizes; returns E_max

**Standardization factors** (logged during estimation, needed to rescale parameter estimates):
| Variable | Raw Range | Standardized Range | Max Value |
|----------|-----------|-------------------|-----------|
| q_cig | 0-60 packs | [0, 1] | q_cig_max |
| q_ecig | 0-51 mL | [0, 1] | q_ecig_max |
| n | 0-1500 mg | [0, 1] | n_max |
| a | 0-1596 | [0, 1] | (normalized: ã = a·ψ) |
| E | 0-605 $ | [0, 1] | E_max |

**Why standardize?**
Without standardization, the μ·a·n[j] term and addiction stock can reach extreme values, causing:
- Degenerate choice probabilities (one alternative has ~100% probability)
- Value function explosion (V reaches 10^7)
- VFI non-convergence

With normalization, μ·ã·n ≤ μ × 1 × 1 = μ (reasonable for small μ), since ã ∈ [0, 1] and n ∈ [0, 1].

**Rescaling parameter estimates:**
After estimation, divide coefficients by the corresponding max values to get original-unit interpretation:

| Parameter | Rescaling Formula | Original Units |
|-----------|-------------------|----------------|
| α_C | α_C_orig = α_C_std / q_cig_max | utils per pack |
| α_E | α_E_orig = α_E_std / q_ecig_max | utils per mL |
| α_CE | α_CE_orig = α_CE_std / q_bundle_max | utils per (pack × mL) |
| ω | ω_orig = ω_std / E_max | utils per dollar |
| μ | μ_orig = μ_std × ψ / n_max² | utils per (mg addiction × mg nicotine). Reinforcement term μ·ã·n_std[j] involves two standardized variables: ã = ψ·a_raw/n_max and n_std = n_raw/n_max, giving n_max² in denominator. ψ is fixed at 0.68. |
| γ | γ_orig = γ_std × ψ / n_max | utils per mg of addiction stock. Withdrawal term γ·ã involves one standardized variable: ã = ψ·a_raw/n_max, giving n_max in denominator. ψ is fixed at 0.68. |
| λ_1, λ_2, λ_3, λ_4 | No rescaling | utils (indicator) |
| ξ_C, ξ_E, ξ_CE | No rescaling | utils (additive) |

Note: The addiction grid is normalized to [0, 1] via ã = ψ·a_raw/n_max. The reinforcement term μ·ã·n_std involves two standardized variables (ã and n_std = n_raw/n_max), so μ_orig = μ_std × ψ / n_max². The withdrawal term γ·ã involves only ã, so γ_orig = γ_std × ψ / n_max. ψ is fixed at 0.68; overridden when ESTIMATE_PSI = true.

**VFI normalization:**
Currently disabled (commented out). When enabled, the solve_vfi function normalizes the value function each iteration by subtracting a reference value to prevent unbounded growth and preserve numerical precision for small effects (TYA, price).

## Key Modeling Choices

**First Stage (R):**
- **Monthly frequency**: Household-month level observations
- **Quantity discretization**: Bins chosen based on purchase frequency patterns
- **Price imputation**: Hierarchical (actual → state-month median → month median)
- **Price dynamics**: AR(1) process estimated on monthly median prices
- **Integration**: Halton sequences (quasi-Monte Carlo) with 200 draws for smooth approximation

**Second Stage (Julia):**
- **Addiction dynamics**: ã' = (1-ψ)ã + ψ·n in normalized units (ã = a·ψ ∈ [0, 1]) with ψ fixed at 0.68 (overridden when ESTIMATE_PSI = true). Normalization keeps addiction on the same [0, 1] scale as consumption, nicotine, and expenditure.
- **Initial conditions**: Household-specific ã₀ estimated via fixed-point iteration on the terminal value mapping; convergence guaranteed by Banach fixed-point theorem (contraction rate (1-ψ)^T)
- **Discounting**: β-δ quasi-hyperbolic discounting. δ=0.99 (monthly exponential). β=1.0 in baseline (standard exponential, β<1 is present-biased). Uses sophisticated agent VFI (`solve_vfi_sophisticated`): maintains decision utility V_d = U + βδ·EV and experienced utility V_e = U + δ·EV, aggregates via softmax(V_d)-weighted V_e + entropy. When β=1, reduces to standard logsumexp (identical to naive). The naive `solve_vfi` is retained but not called.
- **State space discretization**: 20 addiction states × 100 price states × 4 TYA states (both base and Beta)
- **VFI convergence**: Sup-norm tolerance ε=1e-4
- **Interpolation**: Linear interpolation over addiction grid; bilinear interpolation over 2D price grid (cig × ecig) for continuation values. Out-of-grid predicted prices clamped to grid bounds.
- **Flavor effects**: λ_1 (baseline) and λ_2 (× TYA) apply to all flavored products; λ_3 (baseline) and λ_4 (× TYA) are additional effects for FDA-authorized flavored products
- **Parallelization**: VFI parallelizes over price states via `Threads.@threads`
- **Optimization**: Multi-start Nelder-Mead (`random_amoeba`) with random restarts, minimizing the negative log-likelihood
- **Standard errors**: Finite differences on the full objective (each perturbation re-solves VFI)
- **VFI warm-start**: Controlled by `WARM_START` flag. When enabled (`true`), reuses the previous evaluation's converged V as the initial guess within a NM run, reducing iterations from ~948 to ~7-50. V is reset to zeros at each new outer try (L), inner run (M), and long run. Uses `copyto!` (not `copy`) for type-stable initialization. When disabled (`false`), each evaluation starts fresh from zeros.

## Recent Changes (February 2026)

### Data Structure Update
The product choice structure has been updated to 40 alternatives:
- 1 outside option + 12 cig bins + 7 orig ecig + 7 non-FDA flav ecig + 7 FDA flav ecig + 6 bundles
- Bundles: 2 cig levels (lo/hi) × 3 ecig types (orig/non-FDA flav/FDA flav)

### Files Updated

**`01_First_Stage_Estimation/01_Data_Prep.R`:**
- Consumption section: Updated for 12 cig bins, 7 orig ecig bins, 7 flav ecig bins, 4 bundles
- Nicotine section: Updated for 4 bundles with `_cig_nic` and `_ecig_nic` suffixes
- Prices section: Updated state-month medians, monthly medians, and final price columns

**`02_Second_Stage_Estimation/01_Functions.jl`:**
- `get_consumption()`: Handles 4 bundles with column names like `bundle_orig_lo_cig`
- `get_nicotine()`: Handles 4 bundles with nicotine suffix columns
- `get_category_index()`: Assigns cat=4 to 2 orig bundles, cat=5 to 2 flav bundles
- `map_prices_to_grid()`: Uses 12 cig bin column names (`cig_1_p`, `cig_2_p`, etc.)

### VFI Warm-Start Re-enabled via WARM_START Flag
VFI warm-starting was previously disabled entirely after a bug where VFI converged in 1 iteration (the warm-start was never reset between optimizer phases). This has been re-implemented correctly with the `WARM_START` flag:

**Mechanism:** V is reset to `nothing` (→ zeros) whenever `(ra_outer_try, ra_inner_run)` changes — i.e., at each new outer try (L), inner run (M), or long convergence run. Within a single NM run, θ barely changes between evaluations, so the previous V is nearly correct and VFI converges in ~7-50 iterations instead of ~948.

**Type stability fix:** Initial implementation used `copy(V_init)` with `V_init::Union{Array{Float64,3}, Nothing}`, which caused Julia type instability — the compiler failed to narrow the Union type through the conditional branch, making every `V_now` access in the VFI inner loop use dynamic dispatch (~100x slowdown). Fixed by always allocating `V_now = zeros(...)` first, then using `copyto!(V_now, V_init)` in-place if warm-starting.

**Files updated:**
- `01_Functions.jl`: Added `WARM_START` flag default (`false`); added `V_warm_est`/`last_ra_phase_est` globals; added `V_init` keyword to both `solve_vfi` and `solve_vfi_sophisticated`; added warm-start phase detection and V storage in `objective()`; `random_amoeba` starting params logged with `@sprintf("%.4f")`
- `01_MC_Simulation_Functions.jl`: Added `V_warm`/`last_ra_phase` globals; added warm-start phase detection, `V_init` pass-through, and V storage in `objective_mc()`
- `02_Estimation.jl`: Sets `WARM_START = true`
- `02_MC_Simulation_Array.jl`: Sets `WARM_START = true`

### MC True Parameters Updated
Updated θ_true in `02_MC_Simulation_Array.jl`: λ_1=0.1554, λ_2=0.6616, λ_3=-0.0975, λ_4=-0.3359, ξ_C=-3.2001 (was -2.007), ξ_E=-6.0294, ξ_CE=-5.2096, μ=0.05 (was 0.10). Starting values changed from fixed offsets to 50% of θ_true (`starting_param = map(x -> 0.5x, θ_true)`). True parameters CSV now transposed (parameter names as columns) with values rounded to 4 decimal places. All log output for true/starting parameters rounded to 4 decimal places via `@sprintf("%.4f")`.

### Reinforcement Term: μ·a·n[j] (considered but not changed)
Considered changing the reinforcement term from `μ·a·n[j]` to `μ·a·𝟙[j ≠ outside]` to break collinearity between n[j] and c[j]. The current code still uses `μ·a·n[j]` (nicotine intake × addiction stock). The MC simulation with ψ=0.50 improved identification sufficiently without this change.

### Static Logit Starting Values Updated
Updated `static_logit_orig` in `02_Estimation.jl` to match current static logit estimates:
```
α_C=0.0187, α_E=0.0096, α_CE=-0.0021, λ_1=0.852, λ_2=0.703,
λ_3=0.5, λ_4=0.5, ω=-0.0055, ξ_C=-3.573, ξ_E=-6.500, ξ_CE=-5.288
```
MC simulation (`02_MC_Simulation_Array.jl`) θ_true uses original-unit consumption/expenditure values converted to standardized units at runtime, with updated flavor/fixed-effect values (see "MC True Parameters Updated" below). MC starting values are set to 50% of θ_true.

### Addiction Grid Normalization (ã = ψ·a/n_max)
The addiction grid was changed from [0, 1/ψ] (ψ-dependent) to [0, 1] (fixed) by normalizing addiction via ã = ψ·a_raw/n_max. This keeps the grid on the same [0, 1] scale as all other standardized variables regardless of ψ.

**Law of motion change:**
- Old: `a' = (1-ψ)·a + n_std` with grid [0, 1/ψ] — grid blows up when ψ is small (e.g., ψ=0.04 → [0, 25])
- New: `ã' = (1-ψ)·ã + ψ·n_std` with grid [0, 1] — always stable

The ψ·n_std term is the only difference. It scales the nicotine input so the steady state is ã_ss = n_std ∈ [0, 1]. All variables named `a` or `a_h` in the code are now ã (normalized), not raw addiction.

**Parameter adjustments for the new scale:**
- μ and γ were doubled (0.05 → 0.10, -0.05 → -0.10) because ã ∈ [0,1] is half the old a ∈ [0,2] at ψ=0.50. Same utility contribution requires 2× larger coefficients.
- `get_addiction_space(ψ)` accepts ψ for API compatibility but returns [0, 1] regardless.

**Rescaling formulas updated:**
- μ_orig = μ_std × ψ / n_max² (μ multiplies both ã and n_std — two standardized variables; ψ fixed at 0.68)
- γ_orig = γ_std × ψ / n_max (γ multiplies only ã — one standardized variable; ψ fixed at 0.68)

**Files updated:** `01_Functions.jl` (grid, law of motion, docstrings), `01_MC_Simulation_Functions.jl` (inline law of motion), `01_Counterfactual_Functions.jl` (inline law of motion), `02_Estimation.jl` (starting values, rescaling notes), `02_MC_Simulation_Array.jl` (θ_true μ/γ values).

### MC Choice Share Diagnostic
Added choice share logging to `02_MC_Simulation_Array.jl` after data simulation. Logs the fraction of simulated choices in each category (outside, cig, orig_ecig, non_fda_flav_ecig, fda_flav_ecig, orig_bundle, non_fda_flav_bundle, fda_flav_bundle) to diagnose whether the DGP produces reasonable variation or is dominated by the outside option. High outside-option share (>85%) indicates weak identification of inside-alternative parameters.

### Counterfactual and Model Validation Scripts
Both `02_Counterfactual_Flavor_Ban.jl` and `02_Model_Validation.jl` use `ψ` from `get_fixed_parameters()`, which returns ψ fixed at 0.68. These scripts pass θ_hat directly to `get_flow_utility()` which expects 13 elements.

### Static Nested Logit Added (`04_Static_Nested_Logit.jl`)
New file implementing nested logit with 4 nests (outside, cigarettes, e-cigs, bundles) and 1 common σ parameter. Estimated σ̂ = 0.12 (t = 7.29) — statistically significant but modest departure from IIA. Decision: not worth incorporating into the dynamic model. Used as a robustness check.

### Separate Expenditure Terms Considered and Rejected
Tested splitting ω into ω_C (cigarette expenditure) and ω_E (e-cig expenditure) in the static logit. Results: |ω_C| ≈ 3×|ω_E|, but ω_E only borderline significant (t = -2.14) and α_E halved due to collinearity between e-cig consumption and e-cig expenditure. Decision: reverted to single ω for both static logit and nested logit.

### Beta Estimation Variant Added (then merged via ESTIMATE_BETA)
Created β estimation support using a 4-state TYA classification (from `05_TYA_State_Transitions.R`) with anticipated transitions for identification. Originally implemented as parallel Beta files; later merged into the base files via the `ESTIMATE_BETA` flag (see below).

**New file:**
- `01_First_Stage_Estimation/05_TYA_State_Transitions.R` — Computes 4-state TYA classification and monthly transition matrix from household member ages
- `05_MC_Simulation_Beta/02_MC_Simulation_Array_Beta_Slurm.sb` — Slurm script for Beta MC (since deleted; MC with ESTIMATE_BETA/ESTIMATE_PSI now handled by flags in `05_MC_Simulation/02_MC_Simulation_Array.jl`)

### Beta/Non-Beta File Merge via ESTIMATE_BETA Flag
Merged parallel Beta estimation files into the base files using a single `ESTIMATE_BETA` flag. The flag is set at the top of each calling script before `include()`. When `true`, β is estimated as the last element of θ_vec; when `false` (default), β is fixed at 1.0.

**Files merged (Beta files deleted):**
- `01_Functions_Beta.jl` → merged into `01_Functions.jl`
- `02_Estimation_Beta.jl` → merged into `02_Estimation.jl`
- `01_MC_Simulation_Functions_Beta.jl` → merged into `01_MC_Simulation_Functions.jl`
- `02_MC_Simulation_Array_Beta.jl` → merged into `02_MC_Simulation_Array.jl`

**Calling scripts updated with `ESTIMATE_BETA = false`:**
- `03_Standard_Errors.jl`, `02_Model_Validation.jl`, `02_Counterfactual_Flavor_Ban.jl`
- `03_Static_Logit.jl`, `04_Static_Nested_Logit.jl`

### Output Format Standardization (TSV → CSV)
All structured data output files across the codebase have been migrated from tab-separated `.txt` to comma-separated `.csv`:
- `Dynamic_Model_Estimates.csv` (was `.txt` TSV)
- `Dynamic_Model_Standard_Errors.csv` (was `.txt` TSV)
- MC replication results: `<ss>_MC_Rep_Results_<timestamp>.csv` (was `.txt`)
- MC parameter traces: `<ss>_MC_Rep_Parameters_<timestamp>.csv` (was `.txt`)
- MC aggregate outputs: `MC_Results_Combined_<timestamp>.csv`, `MC_Summary_<timestamp>.csv` (were `.txt`)
- Model validation: `Model_Validation_Results.csv`
- Counterfactual: `Counterfactual_Pointwise_Results.csv`, `Counterfactual_Simulation_Results.csv`

Log files remain `.txt`: estimation logs, SE logs, MC replication logs, model validation logs, counterfactual logs, and `Counterfactual_Summary.txt`.

### Comment Style Standardization
All second-stage Julia scripts have been updated to use a consistent explicit comment style:
- Descriptive comments before every function call explaining what it does and what it returns
- `# Print and log <description>` before log message blocks
- Semicolons on all function call assignments
- `# Close the log file handle` before `close(log_io)`
- Standardized section headers with `#############################`

Files updated: `02_Estimation.jl`, `03_Standard_Errors.jl`, `02_Model_Validation.jl`, `02_Counterfactual_Flavor_Ban.jl`, `02_MC_Simulation_Array.jl`, `03_MC_Aggregate_Results.jl`.

### Sequential MC Simulation Removed
`02_MC_Simulation.jl` (sequential, single-process version) has been removed. All MC simulation now uses `02_MC_Simulation_Array.jl` (Slurm job arrays) for parallelism.

### FDA vs Non-FDA Flavored E-Cigarette Split
Split the single "flavored e-cigarette" category into non-FDA flavored and FDA flavored, expanding from 6 to 8 product categories and from 31 to 40 alternatives. Added λ_3 (FDA flavor baseline) and λ_4 (FDA flavor × TYA interaction) to the parameter vector (positions 6-7), expanding from 11 to 13 parameters (base) and 12 to 14 parameters (Beta).

**8 product categories:**
| Cat | Label | Count |
|-----|-------|-------|
| 0 | Outside option | 1 |
| 1 | Cigarettes | 12 |
| 2 | Original e-cig | 7 |
| 3 | Non-FDA flavored e-cig | 7 |
| 4 | FDA flavored e-cig | 7 |
| 5 | Original bundle (cig + orig ecig) | 2 |
| 6 | Non-FDA flavored bundle (cig + non-FDA flav ecig) | 2 |
| 7 | FDA flavored bundle (cig + FDA flav ecig) | 2 |

**Files updated (dynamic model):** `01_Functions.jl` (get_flow_utility, bounds), `02_Estimation.jl`, `03_Standard_Errors.jl`, `02_Model_Validation.jl`, `01_Counterfactual_Functions.jl`, `02_Counterfactual_Flavor_Ban.jl`, `01_MC_Simulation_Functions.jl`, `02_MC_Simulation_Array.jl`.

**Files updated (Beta):** Merged into base files via `ESTIMATE_BETA` flag (see "Beta/Non-Beta File Merge" above).

**Files updated (static):** `03_Static_Logit.jl` (12 params), `04_Static_Nested_Logit.jl` (13 params).

### 4-State TYA Migrated to Base Estimation
The 4-state TYA classification with Π transition matrix integration in VFI was originally only in Beta files. It has now been migrated to the base `01_Functions.jl` and all calling scripts, so both base and Beta use identical TYA handling. The only remaining difference between base and Beta is whether β is fixed at 1.0 or estimated.

**Key changes:**
- `get_tya_states()` and `get_tya_transitions()` added to `01_Functions.jl`
- `get_flow_utility()`: N_TYA = 4 (was 2); states 1,2 → tya=0; states 3,4 → tya=1
- `recompute_choice_values_2tya!` replaced by `recompute_choice_values!` (8 EV accumulators for 4 TYA states, TYA transition integration via Π)
- `solve_vfi()` and `solve_vfi_sophisticated()`: added Π parameter, expanded to 4 TYA states
- `objective()`: passes Π to solve_vfi_sophisticated
- All calling scripts updated: `02_Estimation.jl`, `03_Standard_Errors.jl`, `02_Model_Validation.jl`, `02_Counterfactual_Flavor_Ban.jl`, `01_MC_Simulation_Functions.jl`, `02_MC_Simulation_Array.jl`
- Old `get_teen_young_adult()` and `get_tya_state()` kept in `01_Functions.jl` for backward compatibility with static logit scripts
