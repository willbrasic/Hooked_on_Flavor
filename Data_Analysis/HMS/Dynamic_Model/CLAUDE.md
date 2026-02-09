# Dynamic Model

This is a **dynamic discrete choice model** for tobacco product demand (cigarettes and e-cigarettes) using HMS panel data. The estimation follows a **two-stage approach** where the first stage estimates price processes and prepares the choice/state spaces, then the second stage estimates structural parameters via value function iteration (VFI) and maximum likelihood.

## Directory Structure

```
HMS/
├── Structural_Model_Motivation/      # Static logit model motivating the dynamic model
│   └── 03_Static_Logit.jl
├── Dynamic_Model/
│   ├── 01_First_Stage_Estimation/    # Price processes and choice/state space preparation
│   ├── 02_Second_Stage_Estimation/   # Structural parameter estimation + SEs
│   │   ├── 01_Functions.jl
│   │   ├── 02_Estimation.jl
│   │   └── 03_Standard_Errors.jl
│   ├── 03_Model_Validation/         # Model fit: predicted vs actual shares
│   │   ├── 01_Model_Validation_Functions.jl
│   │   └── 02_Model_Validation.jl
│   ├── 04_Counterfactual_Flavor_Ban/ # Policy counterfactual simulations
│   │   ├── 01_Counterfactual_Functions.jl
│   │   └── 02_Counterfactual_Flavor_Ban.jl
│   └── 05_MC_Simulation/            # Monte Carlo parameter recovery
│       ├── 01_MC_Simulation_Functions.jl
│       ├── 02_MC_Simulation.jl
│       ├── 02_MC_Simulation_Array.jl
│       ├── 02_MC_Simulation_Array_Slurm.sb
│       ├── 03_MC_Aggregate_Results.jl
│       └── 04_Two_Param_Profile.jl
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
| `02_Estimation.jl` | `.../Dynamic_Model_Results/` |
| `03_Standard_Errors.jl` | `.../Dynamic_Model_Results/` |
| `02_Model_Validation.jl` | `.../Dynamic_Model/Model_Validation_Results/` |
| `02_Counterfactual_Flavor_Ban.jl` | `.../Dynamic_Model/Counterfactual_Results/` |
| `02_MC_Simulation.jl` | `.../MC_Simulation_Results/` |
| `03_Static_Logit.jl` | `.../Static_Logit_Results/` |

All under `.../4th_Year_Paper_Data/HMS/2021-Onward/`.

## First Stage Estimation (`01_First_Stage_Estimation/`)

### Script 1: `01_Data_Prep.R`
Prepares the discrete choice data from HMS panel data.

**Outputs:**
| File | Description |
|------|-------------|
| `Household_Codes.csv` | Household identifiers for each observation (household-month) |
| `Category_Choices.csv` | 4 mutually exclusive choices: outside option, cig only, ecig only, both |
| `Product_Choices.csv` | 21 alternatives (20 products + outside option) with quantity bins |
| `Consumption_Spaces.csv` | Median quantities per bin (e.g., cig_5to9 → 6 packs) |
| `Nicotine_Spaces.csv` | Median nicotine absorbed per bin (mg) |
| `Teen_Young_Adult.csv` | Youth presence indicator by household-month |
| `Prices.csv` | Per-unit prices using hierarchical imputation (actual → state-month median → month median) |
| `Lagged_Category_Choice.csv` | Lagged category choice indicators (lagged_cig, lagged_ecig, lagged_cig_ecig); NA for each household's first month |
| `Lead_Prices.csv` | Next month's median per-unit prices (lead_cig_price, lead_ecig_price) using hierarchical imputation (state-month → month); NA for last sample month |
| `Mean_Consumption.csv` | Household mean consumption (mean_cig_consumption, mean_ecig_consumption) for Mundlak correction; controls for persistent unobserved heterogeneity |

**Choice alternatives (21 total):**
- Outside option (no purchase)
- 6 cigarette quantity bins: 1-2, 3-10, 11-20, 21-30, 31-40, 41+ packs
- 3 original e-cig bins: 1-10, 10-30, 30+ mL
- 3 flavored e-cig bins: 0-10, 10-30, 30+ mL
- 8 bundles: 2 cig levels (lo/hi) × 2 ecig levels (lo/hi) × 2 ecig types (orig/flav)
  - bundle_orig_lo_lo, bundle_orig_lo_hi, bundle_orig_hi_lo, bundle_orig_hi_hi
  - bundle_flav_lo_lo, bundle_flav_lo_hi, bundle_flav_hi_lo, bundle_flav_hi_hi

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
| `est_log_io` | Global log file handle (`IO` or `nothing`). Set by the calling script before estimation begins (e.g., `02_Estimation.jl` opens `Estimation_Log.txt`). |
| `est_eval_count` | Global objective evaluation counter. Reset to 0 before each estimation run. |
| `est_log(msg)` | Writes `msg` to stdout AND the active log file. Checks `est_log_io` first; falls back to `mc_log_io` (from `01_MC_Simulation_Functions.jl`) so VFI messages are captured regardless of which script is running. Every write is followed by `flush`. |

**State Spaces and Choices:**
| Function | Purpose |
|----------|---------|
| `get_addiction_space()` | Creates addiction grid: N_A=20 points from 0 to n_max/ψ |
| `get_product_choices()` | Loads product choice matrix J, returns (N_HHT, N_J, J) |
| `get_HH_choices(J)` | Converts choice matrix J to choice vector y (y[i] = chosen alternative index) |
| `get_category_choices()` | Loads category choice matrix K, returns (N_K, K) |

**Alternative-Level Vectors:**
| Function | Purpose |
|----------|---------|
| `get_consumption(N_J)` | Returns separate c_cig and c_ecig vectors indexed by alternative j, plus counts per category (N_cig, N_orig_ecig, N_flav_ecig, N_bundle). For 8 bundles, reads columns like `bundle_orig_lo_lo_cig`, `bundle_orig_lo_lo_ecig`, etc. |
| `get_nicotine(N_J)` | Returns nicotine vector n[j] (mg absorbed) for each alternative; bundles sum cig + ecig nicotine using columns with `_cig_nic` and `_ecig_nic` suffixes |
| `get_category_index(N_J, N_cig, N_orig_ecig, N_flav_ecig)` | Returns cat_idx[j] mapping each alternative to its category (0=outside, 1=cig, 2=orig ecig, 3=flav ecig, 4=orig bundles, 5=flav bundles). Handles 4 orig + 4 flav bundles. |
| `get_flavored_indicator(cat_idx)` | Returns boolean vector; true for flavored ecig (cat=3) and bundle with flavored ecig (cat=5) |

**Demographics:**
| Function | Purpose |
|----------|---------|
| `get_teen_young_adult()` | Loads youth indicator, returns (N_HH, tya) |
| `get_tya_state(tya)` | Maps binary TYA indicator to state index (0 → 1, 1 → 2) |

**Price Space:**
| Function | Purpose |
|----------|---------|
| `get_pricing_spaces()` | Loads 10-point price grid per category, returns (N_P, P) |
| `get_pricing_spaces_combination(N_K, N_P, P)` | Creates N_P^2=100 (cig x ecig) price combinations |
| `get_expenditures(N_J, N_Pcomb, c_cig, c_ecig, Pcomb)` | Computes E[p,j] = p_cig(p)*c_cig[j] + p_ecig(p)*c_ecig[j] for all price-alternative pairs |
| `get_transitions(N_K)` | Loads Halton draw price transitions: M x R x 2 array |
| `precompute_price_transitions(N_P, P, T)` | Pre-computes bilinear interpolation brackets and weights for predicted next-period prices from Halton draws. For each (price state, draw) pair, clamps predictions to grid bounds, finds brackets on each category's 1D grid via binary search. Returns 6 matrices (M × R): `p_cig_lo`, `p_cig_hi`, `p_cig_w`, `p_ecig_lo`, `p_ecig_hi`, `p_ecig_w`. Called once since price transitions don't change across VFI iterations. |

**Household State Trajectories:**
| Function | Purpose |
|----------|---------|
| `map_prices_to_grid(N_P, P, Pcomb)` | Maps observed household prices to nearest combined price grid index using median per-category prices. Reads 6 cig price columns (`cig_1to2_p`, `cig_3to10_p`, etc.) and 6 ecig price columns. Returns `(p_state, p_continuous)` where `p_state` is the nearest grid index and `p_continuous` is an N × 2 matrix of actual continuous (cig, ecig) prices for likelihood interpolation. |

**Estimation:**
| Function | Purpose |
|----------|---------|
| `addiction_evolution(ψ, a, n)` | Addiction law of motion: a' = (1-ψ)a + n |
| `get_flow_utility(θ, N_J, N_A, N_Pcomb, A, c_cig, c_ecig, c_bundle, n, is_flavored, cat_idx, E)` | Pre-computes flow utility for all (tya, alternative, addiction, price) states. Takes 11-element θ (excludes ψ). Returns 4D array U[tya_idx, j, a_idx, p_idx] of dimension 2 × N_J × N_A × N_Pcomb. Reinforcement term: μ·a·𝟙[j≠outside] (implemented as `cat_idx[j] > 0 ? 1.0 : 0.0`). Splits base utility (consumption, addiction, expenditure, fixed effects) from TYA-dependent flavor terms for efficiency. |
| `precompute_addiction_transitions(N_J, N_A, ψ, A, n)` | Pre-computes interpolation brackets (a_lower, a_upper) and weights (a_weight) for all (alternative, addiction state) pairs using binary search |
| `get_initial_addiction_stock(ψ, A, n, y)` | Estimates initial addiction stock per household via fixed-point iteration: simulate forward from a₀, set a₀ to terminal value, repeat until convergence. Returns `(a0, max_iters)` where `max_iters` is the maximum iterations across all households. |
| `simulate_addiction_trajectories(N_A, ψ, A, n, y, a0)` | Simulates addiction forward from estimated a₀ using observed choices. Returns `(a_state, a_continuous)` where `a_state` is the nearest grid index and `a_continuous` is the actual continuous addiction level for likelihood interpolation. |

**Value Function Iteration:**
| Function | Purpose |
|----------|---------|
| `logsumexp(v)` | Numerically stable log-sum-exp: computes `log(Σ exp(v))` by factoring out the maximum to prevent overflow. Used to aggregate choice-specific values into the ex-ante value function (closed-form expected maximum from the Type I extreme value logit error assumption). |
| `solve_vfi(N_J, N_A, N_P, N_Pcomb, β, δ, U, a_lower, a_upper, a_weight, p_cig_lo, p_cig_hi, p_cig_w, p_ecig_lo, p_ecig_hi, p_ecig_w; ε, max_iter, V_init)` | Solves the value function via VFI. The Bellman iteration uses δ only (exponential discounting): V_choice = U + δ·EV. After convergence, computes V_decision for quasi-hyperbolic (β-δ) discounting: V_decision = (1-β)·U + β·V_choice = U + βδ·EV. When β=1, V_decision = V_choice. When β<1, the naive agent over-discounts the future relative to the present. For each state (tya, a, p) and alternative j, computes expected continuation value by: (1) bilinear price interpolation at each Halton draw's predicted prices, (2) averaging over R draws, (3) linear interpolation across pre-computed addiction brackets. Aggregates via log-sum-exp. The price state loop is parallelized via `Threads.@threads`. Accepts optional `V_init` for warm starting across parameter updates (currently disabled in objective functions). Returns `(V, V_decision, n_iter, converged)` where V is 3D `[tya, a, p]` (δ only), V_decision is 4D `[tya, j, a, p]` (βδ, used for choice probabilities), n_iter is the iteration count, and `converged::Bool` indicates whether the sup-norm fell below ε. Convergence/non-convergence messages are written via `est_log`. |

**Log-Likelihood:**
| Function | Purpose |
|----------|---------|
| `log_likelihood(V_choice, N_J, N_P, A, P, y, tya_state, a_continuous, p_continuous)` | Computes the sample log-likelihood ℓ(θ) = Σᵢ log P(yᵢ \| xᵢ; θ) by trilinearly interpolating V_choice at each observation's continuous state. For each observation: (1) finds linear interpolation brackets on the addiction grid, (2) finds bilinear interpolation brackets on the 2D price grid, (3) interpolates V_choice for all 21 alternatives, (4) computes log P(yᵢ) = V_choice_interp[yᵢ] - logsumexp(V_choice_interp). Returns a scalar log-likelihood. |

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
| `θ_lower_bound`, `θ_upper_bound` | Global constant vectors (12 elements) defining economic parameter bounds (standardized units). α_T, α_E, α_TE, μ ≥ 0; γ, ω ≤ 0; λ and ξ unconstrained; ψ ∈ (0.01, 0.99). Used by both `objective()` and `objective_mc()`. |
| `check_parameter_bounds(θ_vec, param_names)` | Returns `(in_bounds::Bool, violations::String)`. Checks all elements of θ against the bound vectors and reports which parameters violate constraints. |

**Optimization:**
| Function / Type | Purpose |
|----------|---------|
| `SimplexWithAdd` | Custom `Optim.Simplexer` subtype. Constructs a (D+1)-vertex Nelder-Mead simplex around the starting point by adding deviation `add[d]` along each coordinate direction d, where D is the number of parameters. |
| `random_amoeba(objective, starting_param, add, L, M, inner_iter; log_io)` | Multi-start Nelder-Mead optimizer. For each of L outer tries: (1) runs M short Nelder-Mead runs (`inner_iter` iterations each, `f_abstol=1e-4`) with randomized simplex perturbations, (2) runs one long Nelder-Mead run (up to 5,000 iterations, `f_abstol=1e-3`) to fully converge, (3) updates the global best if improved, (4) randomly reinitializes parameters (20% chance original, 40% chance best-so-far, 40% chance current). Short runs may converge before hitting the iteration limit. Logs iteration count, convergence status, parameter values, and timing after each short run, long run, and outer try. Takes `starting_param` as a NamedTuple and returns `(opt_param, overall_min)`. |

**Objective Function:**
| Variable / Function | Purpose |
|----------|---------|
| `objective(θ_vec)` | Objective function for the optimizer (12-element θ_vec). Increments `est_eval_count` and times each evaluation. **Box constraints:** returns `1e14` penalty if any parameter violates bounds via `check_parameter_bounds`. **ψ handling:** extracts `ψ_current = θ_vec[end]`, recomputes `N_A_current, A_current = get_addiction_space(ψ_current)`, and passes `θ_vec[1:end-1]` (11 elements) to `get_flow_utility()`. For each candidate θ: (1) recomputes addiction grid from ψ, (2) recomputes flow utility U, (3) recomputes addiction transition brackets at ψ_current, (4) solves VFI from scratch (no warm-start). **Early-exit:** if VFI did not converge, logs a PENALTY message and returns `1e14`. Otherwise: (5) recomputes addiction trajectories at ψ_current, (6) evaluates log-likelihood via trilinear interpolation, (7) logs eval number, LL, VFI iters, elapsed time, and θ vector via `est_log`. Returns the negative log-likelihood. Uses global data objects set in `02_Estimation.jl`. |

**Alternative Ordering (j = 1, ..., 21):**
| Index | Alternative |
|-------|-------------|
| j = 1 | Outside option (zero consumption) |
| j = 2:7 | 6 cigarette quantity bins |
| j = 8:10 | 3 original e-cigarette bins |
| j = 11:13 | 3 flavored e-cigarette bins |
| j = 14:17 | 4 original bundles (lo/hi cig × lo/hi ecig) |
| j = 18:21 | 4 flavored bundles (lo/hi cig × lo/hi ecig) |

**Structural Parameters (θ) - 12 parameters:**
```
α_T   = cigarette consumption utility
α_E   = e-cig consumption utility
α_TE  = bundle consumption utility
λ_1   = baseline flavor effect
λ_2   = flavor × teen/young adult interaction
μ     = reinforcement effect (addiction × any-tobacco indicator)
γ     = addiction level effect (withdrawal cost)
ω     = expenditure coefficient (price sensitivity)
ξ_T   = cigarette fixed effect
ξ_E   = e-cig fixed effect
ξ_TE  = bundle fixed effect
ψ     = addiction decay rate (estimated jointly)
```

**Flow Utility Specification:**
```
u(j,a,p,tya) = α_T·c_cig[j] + α_E·c_ecig[j] + α_TE·c_cig[j]·c_ecig[j]
             + μ·a·𝟙[j ≠ outside] + γ·a
             + ω·E[p,j]
             + ξ_k
             + 𝟙[flavored]·(λ_1 + λ_2·𝟙[tya])
```
Where:
- c_cig[j], c_ecig[j] = cigarette and e-cigarette consumption for alternative j
- 𝟙[j ≠ outside] = 1 if alternative j involves any tobacco use, 0 for outside option
- a = addiction state
- E[p,j] = expenditure (p_cig·c_cig[j] + p_ecig·c_ecig[j])
- ξ_k = fixed effect (ξ_T for cig, ξ_E for ecig, ξ_TE for bundles)
- tya = teen/young adult indicator

Note: The reinforcement term was changed from `μ·a·n[j]` to `μ·a·𝟙[j ≠ outside]` to break collinearity between nicotine n[j] and consumption c[j] vectors. The old specification created a near-linear dependence μ·a·n[j] ≈ f(α·c[j]) that prevented separate identification of μ from the consumption parameters.

**State Space:**
- 2 teen/young adult states (0, 1)
- 20 addiction states (grid from 0 to n_max/ψ)
- 100 price states (10x10 cig x ecig grid)
- 21 product alternatives

**Value Function (VFI uses δ only):**
```
V_choice[tya, j, a, p] = u(j,a,p,tya) + δ·E[V(tya', a', p') | a, p, j]
V[tya, a, p] = log( Σ_j exp(V_choice[tya, j, a, p]) )
```

**Decision Utility (quasi-hyperbolic β-δ discounting):**
```
V_decision[tya, j, a, p] = u(j,a,p,tya) + β·δ·E[V(tya', a', p') | a, p, j]
                         = (1 - β)·U + β·V_choice
```
When β = 1 (standard exponential), V_decision = V_choice. When β < 1 (present bias), the naive agent over-discounts the future. Choice probabilities and the log-likelihood use V_decision.

### Script 2: `02_Estimation.jl`

Main estimation script. Includes `01_Functions.jl`, loads all data, sets fixed parameters (β=1.0, δ=0.99), and runs the multi-start Nelder-Mead optimizer. ψ is now estimated jointly as the 12th parameter (no longer fixed).

**Starting values:**
- Static logit estimates from `03_Static_Logit.jl` provide starting values for shared parameters
- Since static logit used unstandardized data, estimates must be **converted to standardized units**:
  - `α_T_std = α_T_orig × c_cig_max`
  - `α_E_std = α_E_orig × c_ecig_max`
  - `α_TE_std = α_TE_orig × c_bundle_max`
  - `ω_std = ω_orig × E_max`
  - `λ_1, λ_2, ξ_T, ξ_E, ξ_TE`: no conversion (indicators/additive)
- μ and γ have no static counterpart; initialized at 0.05 and -0.05
- ψ starts at 0.50 (midpoint of (0, 1))

**Estimation settings:**
- Simplex deviations scaled to ~50-100% of starting parameter magnitudes; ψ deviation = 0.20
- Optimizer: L=50 outer tries, M=20 inner tries, 100 iterations per inner run

**Rescaling estimates to original units:**
After estimation, divide by max values to get interpretable units:
- `α_T_orig = α_T_std / c_cig_max` → utils per pack
- `α_E_orig = α_E_std / c_ecig_max` → utils per mL
- `α_TE_orig = α_TE_std / c_bundle_max` → utils per (pack × mL)
- `ω_orig = ω_std / E_max` → utils per dollar
- `μ_orig = μ_std / n_max` → utils per mg addiction (reinforcement: μ·a·𝟙[j≠outside], a standardized by n_max)
- `γ_orig = γ_std / n_max` → utils per mg addiction
- `λ_1, λ_2, ξ_T, ξ_E, ξ_TE, ψ`: no rescaling needed

**Output files** (written to `.../Dynamic_Model_Results/`):
- `Dynamic_Model_Estimation_Log_<timestamp>.txt` — Full estimation progress: VFI convergence info, eval-by-eval LL and θ, optimizer restarts, timing
- `Dynamic_Model_Estimates.txt` — Tab-separated estimated parameters (read by `03_Standard_Errors.jl`)

### Script 3: `03_Standard_Errors.jl`

Computes standard errors for the dynamic model parameter estimates via finite differences on the full objective function. ψ is now part of θ_hat (12th parameter); the initial `N_A, A = get_addiction_space(0.94)` is a placeholder — `objective()` recomputes the addiction grid from the ψ in θ_vec at each evaluation.

**Process:**
1. Loads the same data as `02_Estimation.jl`
2. Reads θ_hat (12 parameters including ψ) from `Dynamic_Model_Estimates.txt`
3. Evaluates the objective once at θ_hat (to get `nll_center` for diagonal Hessian entries)
4. Computes the Hessian via central finite differences (h=1e-3), each evaluation re-solves VFI from scratch
5. Inverts the Hessian to get variance-covariance matrix and SEs

**Hessian computation:**
- Diagonal: `H[k,k] ≈ (f(θ+h·e_k) - 2·f(θ) + f(θ-h·e_k)) / h^2` (2 evals per parameter)
- Off-diagonal: `H[k,l] ≈ (f(θ++) - f(θ+-) - f(θ-+) + f(θ--)) / (4h^2)` (4 evals per pair)
- Total: 300 objective evaluations for 12 parameters (each requires full VFI solve)
- Progress logged for every parameter pair with timing

**Output files** (written to `.../Dynamic_Model_Results/`):
- `SE_Log.txt` — Full progress: per-pair Hessian values, eigenvalue diagnostics, results table with SEs and t-statistics
- `Dynamic_Model_Standard_Errors.txt` — Tab-separated estimates and SEs

## MC Simulation (`05_MC_Simulation/`)

### MC Functions: `05_MC_Simulation/01_MC_Simulation_Functions.jl`

Contains functions specific to the Monte Carlo simulation. These handle data simulation from a known DGP and MC-specific versions of estimation functions that work with simulated (in-memory) data rather than CSV files.

**Logging:**
| Variable / Function | Purpose |
|----------|---------|
| `mc_log_io` | Global log file handle for MC logging. Set by `02_MC_Simulation.jl` before any logging. |
| `eval_count` | Global MC evaluation counter. Reset before each replication. |
| `mc_log(msg)` | Writes `msg` to stdout AND `mc_log_io` (if open). Every write is followed by `flush`. |

**Data Simulation:**
| Function | Purpose |
|----------|---------|
| `simulate_data(V_decision_true, ψ, N_J, N_P, A, P, n, real_p_continuous, real_tya_state, real_hh_codes)` | Design-based MC simulation: conditions on real observables (prices, TYA, panel structure) and only simulates choices. Uses per-observation TYA state (varies by month within households). Two-pass approach: Pass 1 simulates from a₀=0, Pass 2 re-simulates from fixed-point corrected a₀. Returns `(y_sim, tya_state_sim, p_continuous_sim, hh_codes_sim)`. |

**MC-Specific Addiction Functions:**
| Function | Purpose |
|----------|---------|
| `get_initial_addiction_stock_mc(ψ, A, n, y, household_codes)` | Same as `get_initial_addiction_stock` but takes `household_codes` as a vector argument instead of reading CSV. For simulated data. |
| `simulate_addiction_trajectories_mc(N_A, ψ, A, n, y, a0, household_codes)` | Same as `simulate_addiction_trajectories` but takes `household_codes` as a vector argument instead of reading CSV. For simulated data. |

**MC Objective Function:**
| Variable / Function | Purpose |
|----------|---------|
| `y_sim`, `tya_state_sim`, `p_continuous_sim`, `household_codes_sim` | Global variables holding simulated data. Set by the MC loop in `02_MC_Simulation.jl` before each replication's estimation. |
| `objective_mc(θ_vec)` | MC-specific objective function (12-element θ_vec). Same as `objective()` but uses simulated data globals and MC-specific addiction functions. Extracts `ψ_current = θ_vec[end]`, recomputes addiction grid and transitions at each eval (same ψ handling as `objective()`). Increments `eval_count` and times each evaluation. **Box constraints:** returns `1e14` penalty via `check_parameter_bounds` (matching `objective()`). **Early-exit:** if VFI did not converge, logs a PENALTY message via `mc_log` and returns `1e14`. Logs eval number, LL, VFI iters, elapsed time, and θ vector. |

### Script 2: `05_MC_Simulation/02_MC_Simulation.jl`

Monte Carlo simulation to verify parameter recovery under a known DGP. Sets `Random.seed!(42)` for reproducibility. For each of S replications: (1) simulates household panel data from the true V_decision, (2) estimates θ via multi-start Nelder-Mead (`objective_mc`), (3) stores estimated parameters.

**Includes:** `../02_Second_Stage_Estimation/01_Functions.jl` and `01_MC_Simulation_Functions.jl`

**True DGP Parameters:**
```
θ_true = (α_T=0.46, α_E=0.37, α_TE=0.50, λ_1=0.67, λ_2=0.41,
          μ=0.05, γ=-0.05, ω=-1.94, ξ_T=-3.61, ξ_E=-5.46, ξ_TE=-6.05, ψ=0.94)
```
Close to the static logit estimates converted to standardized units, ensuring that starting values are in the right ballpark. α_TE is set to 0.50 (not 9.57 from naive conversion) to avoid bundle dominance. ψ=0.94 is the true addiction decay rate.

**Starting Values:** Deliberately offset from truth (α_T=0.1, α_E=0.1, α_TE=0.1, λ_1=0.1, λ_2=0.1, μ=0.1, γ=-0.1, ω=-1.0, ξ_T=-1.0, ξ_E=-2.0, ξ_TE=-3.0, ψ=0.30) with simplex deviations ~50% of absolute value; ψ deviation = 0.20.

**MC Settings (full/HPC):**
- S=100 replications (via Slurm job arrays), uses real N_HH and N_obs from data
- L=2 outer tries, M=2 inner tries, inner_iter=100
- Design-based: conditions on real observables (prices, TYA, panel structure), only simulates choices

**Process:**
1. Loads data/state spaces (same as estimation)
2. Loads AR(1) parameters (φ₀, φ₁, Σ) and computes Cholesky decomposition L_chol
3. Solves VFI at true parameters to get `V_choice_true`
4. For each replication s=1,...,S:
   - Simulates data via `simulate_data(V_choice_true, ...)`
   - Resets `V_warm = nothing` and `eval_count = 0`
   - Estimates θ via `random_amoeba(objective_mc, ...)`
   - Logs True vs Estimated parameter table
   - Logs running summary (mean, bias) across completed replications
   - Incrementally saves results to `MC_Results.txt` after each replication (survives job kills)
5. Final summary: True, Mean, Bias, Std Dev, RMSE for each parameter

**Output files** (written to `.../MC_Simulation_Results/`):
- `MC_Results.txt`: Tab-separated, columns = parameter names, rows = replications
- `MC_Log.txt`: Full progress log with timing, parameter values, LL values

### Script 3: `05_MC_Simulation/02_MC_Simulation_Array.jl`

Parallel version of the MC simulation using Slurm job arrays. Each array task runs a **single replication**, reading its replication number `s` from `SLURM_ARRAY_TASK_ID` (or command-line argument for local testing). Seeds RNG with `Random.seed!(s)` for reproducibility. θ_true includes ψ=0.94 as the 12th parameter. DGP section uses `ψ_true = θ_true.ψ` to create the addiction grid and solve the true VFI. `simulate_data()` uses `ψ_true`.

**Usage:**
- HPC: `sbatch 02_MC_Simulation_Array_Slurm.sb` (launches 100 independent tasks)
- Local: `julia 02_MC_Simulation_Array.jl 1` (runs replication 1)

**Output files** (one set per replication, written to `.../MC_Simulation_Results/`):
- `MC_Rep_<s>.txt`: Single-row result with NLL and estimated parameters
- `MC_Rep_<s>_Log.txt`: Full log for this replication
- `MC_Rep_<s>_Trace.txt`: Parameter trace for this replication

### Script 4: `05_MC_Simulation/03_MC_Aggregate_Results.jl`

Aggregates per-replication MC results from job array. Reads all `MC_Rep_<s>.txt` files, checks for missing replications, computes summary statistics (Mean, Bias, Std Dev, RMSE).

**Output files** (written to `.../MC_Simulation_Results/`):
- `MC_Results_Combined.txt`: All replications combined
- `MC_Summary.txt`: Summary statistics table

### Script 5: `05_MC_Simulation/04_Two_Param_Profile.jl`

Diagnostic tool for visualizing the likelihood surface. Evaluates NLL over a 2D grid of two parameters while holding all others at truth. Default profiles (α_T, γ) to diagnose the ridge between consumption utility and withdrawal cost. Change `param_idx_1` and `param_idx_2` to profile other pairs. θ_true includes ψ=0.94 as the 12th parameter. Grid evaluation recomputes the addiction grid from ψ at each grid point (handles profiling over ψ correctly).

**Output files** (written to `.../MC_Simulation_Results/`):
- `Profile_<name1>_<name2>.csv`: Long-format grid (param1, param2, NLL) for plotting

## Static Logit (`../Structural_Model_Motivation/03_Static_Logit.jl`)

Static conditional logit model motivating the dynamic model. Drops the dynamic addiction terms (μ, γ) from the flow utility and does not require VFI. Prices enter as continuous observed values rather than discretized grid states. The state dependence parameter ρ motivates the dynamic model: significant ρ means past choices predict current behavior (addiction/habit persistence) which a static model cannot properly account for.

**Includes:** `../Dynamic_Model/02_Second_Stage_Estimation/01_Functions.jl` (local) or `../Dynamic_Model/01_Functions.jl` (HPC). Uses `hpc` flag to toggle paths.

**Data Loading:**
- Same data as `02_Estimation.jl` (product choices, consumption, category index, flavored indicator, TYA, continuous prices) but does NOT load addiction grid, nicotine, or price transitions
- Additionally loads `Lagged_Category_Choice.csv` for state dependence term
- Sample restricted to observations with non-missing lagged choice (drops first month per household)

**Pre-computed Matrices:**
| Matrix | Dimension | Description |
|--------|-----------|-------------|
| `E_obs[i, j]` | N_obs × N_J | Current expenditure: `p_cig[i] · c_cig[j] + p_ecig[i] · c_ecig[j]` |
| `lag_match[i, j]` | N_obs × N_J | 1 if alternative j's category matches the household's lagged category choice |

**Structural Parameters (θ) — 10 parameters:**
```
α_T   = cigarette consumption utility (per pack)
α_E   = e-cig consumption utility (per mL)
α_TE  = bundle interaction utility (per pack·mL)
λ_1   = baseline flavor effect
λ_2   = flavor × teen/young adult interaction
ρ     = state dependence (lagged category match)
ω     = expenditure coefficient (price sensitivity)
ξ_T   = cigarette fixed effect
ξ_E   = e-cig fixed effect
ξ_TE  = bundle fixed effect
```

**Negative Log-Likelihood:**

`neg_log_likelihood(θ_vec, N_obs, N_J, tya, y, c_cig, c_ecig, c_bundle, is_flavored, lag_match, E_obs, fe_T, fe_E, fe_TE)` — Takes data arrays as arguments for type stability with ForwardDiff. A single-argument wrapper `nll = θ -> neg_log_likelihood(θ, ...)` is used by the optimizer and Hessian computation.

**Estimation (three optimizers for comparison):**
1. **L-BFGS with autodiff** — `optimize(nll, ...; autodiff = :forward)` with ForwardDiff. JIT warmup call before timed optimization. Callback logs every 10 iterations. Converges in ~94 iterations, ~36s.
2. **Nelder-Mead (Random Amoeba)** — `random_amoeba(nll, ...)` with L=3, M=2, inner_iter=500.

**Standard Errors (three methods for comparison):**
1. **Autodiff Hessian** — `ForwardDiff.hessian(nll, θ_hat)` gives exact second derivatives.
2. **Finite differences** — Central differences on `nll` with h=1e-3. Diagonal and off-diagonal entries computed separately.
3. **Comparison table** — Shows SE differences between autodiff and finite differences.

**Output** (written to `.../Static_Logit_Results/`):
- `Static_Logit_Estimation_Log_<timestamp>.txt` — L-BFGS iteration progress, Nelder-Mead progress, estimates with SEs and t-statistics from all methods, comparison tables (timestamped to avoid overwriting previous runs)

## Data Standardization

**All continuous variables entering the utility function are STANDARDIZED by dividing by their maximum value.**

This is done automatically in the data-loading functions:
- `get_consumption()` → returns standardized c_cig, c_ecig and their raw max values
- `get_nicotine()` → returns standardized n and raw n_max
- `get_addiction_space()` → uses standardized n_max=1.0, so A ∈ [0, 1/ψ] ≈ [0, 1.06]
- `get_expenditures()` → computes E from raw consumption, then standardizes; returns E_max

**Standardization factors** (logged during estimation, needed to rescale parameter estimates):
| Variable | Raw Range | Standardized Range | Max Value |
|----------|-----------|-------------------|-----------|
| c_cig | 0-60 packs | [0, 1] | c_cig_max |
| c_ecig | 0-51 mL | [0, 1] | c_ecig_max |
| n | 0-1500 mg | [0, 1] | n_max |
| a | 0-1596 | [0, 1.06] | (derived from n_max/ψ) |
| E | 0-605 $ | [0, 1] | E_max |

**Why standardize?**
Without standardization, the μ·a·𝟙[j≠outside] term and addiction stock can reach extreme values, causing:
- Degenerate choice probabilities (one alternative has ~100% probability)
- Value function explosion (V reaches 10^7)
- VFI non-convergence

With standardization, μ·a·𝟙 ≤ μ × 1/ψ × 1 (reasonable for small μ).

**Rescaling parameter estimates:**
After estimation, divide coefficients by the corresponding max values to get original-unit interpretation:

| Parameter | Rescaling Formula | Original Units |
|-----------|-------------------|----------------|
| α_T | α_T_orig = α_T_std / c_cig_max | utils per pack |
| α_E | α_E_orig = α_E_std / c_ecig_max | utils per mL |
| α_TE | α_TE_orig = α_TE_std / c_bundle_max | utils per (pack × mL) |
| ω | ω_orig = ω_std / E_max | utils per dollar |
| μ | μ_orig = μ_std / n_max | utils per mg addiction (reinforcement: μ·a·𝟙[j≠outside], a standardized by n_max) |
| γ | γ_orig = γ_std / n_max | utils per mg addiction |
| λ_1, λ_2 | No rescaling | utils (indicator) |
| ξ_T, ξ_E, ξ_TE | No rescaling | utils (additive) |
| ψ | No rescaling | dimensionless decay rate |

Note: The reinforcement term is μ·a·𝟙[j ≠ outside] where a (addiction stock) is standardized by n_max. Therefore μ_orig = μ_std / n_max. The addiction component a is scaled by n_max (a_max = n_max/ψ in raw units, but the grid uses standardized n_max=1, so a ∈ [0, 1/ψ]). γ divides by n_max because it only multiplies a. ψ is dimensionless and requires no rescaling.

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
- **Addiction dynamics**: a' = (1-ψ)a + n with ψ estimated jointly (bounded to (0.01, 0.99))
- **Initial conditions**: Household-specific a₀ estimated via fixed-point iteration on the terminal value mapping; convergence guaranteed by Banach fixed-point theorem (contraction rate (1-ψ)^T)
- **Discounting**: β-δ quasi-hyperbolic discounting. δ=0.99 (monthly exponential). β=1.0 (present bias; β=1 is standard exponential, β<1 is present-biased). VFI is solved with δ only (naive agent); choice probabilities use βδ via V_decision.
- **State space discretization**: 20 addiction states × 100 price states × 2 TYA states
- **VFI convergence**: Sup-norm tolerance ε=1e-4
- **Interpolation**: Linear interpolation over addiction grid; bilinear interpolation over 2D price grid (cig × ecig) for continuation values. Out-of-grid predicted prices clamped to grid bounds.
- **Flavor effects**: Separate baseline (λ_1) and youth interaction (λ_2) effects
- **Parallelization**: VFI parallelizes over price states via `Threads.@threads`
- **Optimization**: Multi-start Nelder-Mead (`random_amoeba`) with random restarts, minimizing the negative log-likelihood
- **Standard errors**: Finite differences on the full objective (each perturbation re-solves VFI)
- **VFI warm-start**: Disabled (each evaluation starts fresh) to prevent false 1-iteration convergence

## Recent Changes (February 2026)

### Data Structure Update
The product choice structure was updated to better capture bundle heterogeneity:

**Old structure (21 alternatives):**
- 1 outside option + 12 cig bins + 3 orig ecig + 3 flav ecig + 2 bundles

**New structure (21 alternatives):**
- 1 outside option + 6 cig bins + 3 orig ecig + 3 flav ecig + 8 bundles
- Bundles: 2 cig levels (lo/hi) × 2 ecig levels (lo/hi) × 2 ecig types (orig/flav)

### Files Updated

**`01_First_Stage_Estimation/01_Data_Prep.R`:**
- Consumption section: Updated for 6 cig bins and 8 bundles
- Nicotine section: Updated for 8 bundles with `_cig_nic` and `_ecig_nic` suffixes
- Prices section: Updated state-month medians, monthly medians, and final price columns

**`02_Second_Stage_Estimation/01_Functions.jl`:**
- `get_consumption()`: Handles 8 bundles with column names like `bundle_orig_lo_lo_cig`
- `get_nicotine()`: Handles 8 bundles with nicotine suffix columns
- `get_category_index()`: Assigns cat=4 to 4 orig bundles, cat=5 to 4 flav bundles
- `map_prices_to_grid()`: Uses 6 cig bin column names (`cig_1to2_p`, etc.)
- `objective()`: Disabled warm-start (V_init = nothing)

**`04_MC_Simulation/01_MC_Simulation_Functions.jl`:**
- `objective_mc()`: Disabled warm-start to match main estimation

### VFI Warm-Start Issue
Around eval 2800, VFI was converging in only 1 iteration due to warm-start from previous θ. This was fixed by disabling warm-start entirely (setting `V_Init = nothing`) so each evaluation starts fresh from zeros.

### Reinforcement Term Change (μ·a·n[j] → μ·a·𝟙[j ≠ outside])
Changed the reinforcement term from `μ·a·n[j]` to `μ·a·𝟙[j ≠ outside]` where 𝟙[j ≠ outside] = `cat_idx[j] > 0 ? 1.0 : 0.0`. This breaks the near-linear dependence between n[j] and c[j] that prevented separate identification of μ from consumption parameters (α_T, α_E). The μ rescaling changed from `μ_std / n_max²` to `μ_std / n_max`.

**Files updated:**
- `01_Functions.jl`: `get_flow_utility()` — reinforcement line, docstring, parameter unpacking
- `02_Estimation.jl`: rescaling comments for μ

### ψ Added as 12th Estimated Parameter
Addiction decay rate ψ was previously fixed at 0.94 and is now estimated jointly with structural parameters. Bounded to (0.01, 0.99) to avoid degenerate grids. Both `objective()` and `objective_mc()` extract `ψ_current = θ_vec[end]`, recompute the addiction grid, transitions, and trajectories at each evaluation.

**Files updated:**
- `01_Functions.jl`: bounds vectors (12 elements), `objective()` ψ extraction and recomputation
- `01_MC_Simulation_Functions.jl`: `objective_mc()` same ψ handling
- `02_Estimation.jl`: `starting_param` with ψ=0.50, add vector with 0.20, rescaling comments
- `02_MC_Simulation_Array.jl`: θ_true with ψ=0.94, starting ψ=0.30, DGP uses ψ_true
- `03_MC_Aggregate_Results.jl`: θ_true with ψ=0.94
- `04_Two_Param_Profile.jl`: θ_true with ψ=0.94, grid eval recomputes A at each point
- `03_Standard_Errors.jl`: removed hardcoded ψ=0.94

### MC Identification Finding
MC simulations revealed that ψ=0.94 (rapid addiction decay, only 6% persists) compresses addiction stock variation so severely that μ, γ, and the consumption parameters (α_T, α_E) are poorly identified. With ψ=0.5 (50% persistence), parameter recovery improves dramatically — even without the reinforcement term change.
