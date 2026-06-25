################################################################################
# William Brasic
# The University of Arizona
# wbrasic97@gmail.com
# June 2026
#
# This script simulates the effect of a flavored tobacco product ban on
# consumer choices, addiction, and welfare using the K=3 mixture model.
#
# Three ban types are evaluated:
#   - Comprehensive Ban: removes all flavored e-cigs and bundles
#     (cat_idx in {3, 4, 6, 7}) via apply_flavor_ban!
#   - FDA-Only Ban: removes only FDA-authorized flavored e-cigs and bundles
#     (cat_idx in {4, 7}) via apply_fda_flavor_ban!
#   - Non-FDA Ban: removes only non-FDA-authorized flavored e-cigs and bundles
#     (cat_idx in {3, 6}) via apply_non_fda_ban!; models FDA enforcement that
#     clears unauthorized products while leaving PMTA-approved ones intact
#
# The script:
#   1. Loads estimated K=3 mixture parameters θ_hat from
#      Dynamic_Model_Mixture_<psi_tag>_Beta_1_Estimates.csv
#   2. For each ban type (Comprehensive, FDA-Only, Non-FDA):
#      a. Applies the ban to copies of the flow utility arrays
#      b. For each β in BETA_GRID:
#         i.   Solves VFI for all three types under SQ and ban
#         ii.  Computes posterior type probabilities from observed data
#         iii. Computes pointwise choice probs and welfare
#         iv.  Computes category shares and welfare
#         v.   Forward-simulates households via Monte Carlo integration:
#              - Draw N_draws = 100 independent paths per household.
#              - Each path: sample a latent type from the household's posterior
#                P(type=k | data_h), then simulate T_sim = 36 periods by (a)
#                interpolating the type's value function at the current state,
#                (b) drawing a choice via inverse CDF on the softmax probabilities,
#                and (c) advancing addiction stocks and AR(1) prices.
#              - Status quo and counterfactual paths share pre-drawn common random
#                numbers (CRN): the same uniform determines the choice and the same
#                normal draws drive the price shocks in both scenarios, so
#                SQ-vs-policy differences reflect the policy rather than simulation noise. 
#              - Average over all N_HH * N_draws paths for period-by-period
#                category shares, mean addiction, and mean welfare.
#         vi.  Aggregates results by subgroup (TYA status, latent type)
################################################################################


#############################
# 1. Preliminaries
#############################

# Mixture model: β is fixed at 1.0 (not estimated); we sweep over BETA_GRID instead
ESTIMATE_BETA = false

# Flavored habit decay rate ψ_3
# If ESTIMATE_PSI_3=true, ψ_3 was estimated jointly as the 27th parameter.
# If false (default), ψ_3 is fixed at PSI_3 (read from ENV, default 0.50).
ESTIMATE_PSI_3 = parse(Bool, get(ENV, "ESTIMATE_PSI_3", "false"))
PSI_3 = parse(Float64, get(ENV, "PSI_3", "0.50"))

# Detect whether we are running on the HPC (any non-Windows system)
HPC = !Sys.iswindows()

# Load Random for common random number seeding in forward simulations
using Random

# Set output path and working directory
if HPC

    # Load mixture estimation functions and packages (must come first; provides Printf, CSV, etc.)
    # HPC folder: /home/u2/wbrasic/4th_Year_Paper/Dynamic_Model/04_CF_Mixture_V2/
    # Use absolute paths for robustness (unaffected by cd() calls)
    include("/home/u2/wbrasic/4th_Year_Paper/Dynamic_Model/02_Second_Stage_Estimation_V2/01_Functions_Mixture.jl")

    # Load counterfactual-specific functions
    include("/home/u2/wbrasic/4th_Year_Paper/Dynamic_Model/04_CF_Mixture_V2/01_CF_Functions_Mixture.jl")

    # Construct psi, beta, and psi_3 tags for directory and file naming (matches estimation script)
    psi_tag   = "Psi2_09_Psi1_01"
    psi_3_tag = ESTIMATE_PSI_3 ? "Psi3_Est" : "Psi3_$(numeric_tag(PSI_3))"
    _, _, β_naming, _ = get_fixed_parameters()

    # Output path for results (absolute path so it's unaffected by later cd)
    output_dir = "/home/u2/wbrasic/4th_Year_Paper/Dynamic_Model/04_CF_Mixture_V2/CF_Mixture_$(psi_tag)_$(psi_3_tag)_Results"

    # Create output directory if it doesn't exist
    mkpath(output_dir)

    # Set working directory to where the data CSVs live (absolute path)
    cd("/home/u2/wbrasic/4th_Year_Paper/Data")
else

    # Load mixture estimation functions and packages (must come first; provides Printf, CSV, etc.)
    include("../02_Second_Stage_Estimation_Mixture/01_Functions_Mixture.jl")

    # Load counterfactual-specific functions
    include("01_CF_Functions_Mixture.jl")

    # Construct psi, beta, and psi_3 tags for directory and file naming (matches estimation script)
    psi_tag   = "Psi2_09_Psi1_01"
    psi_3_tag = ESTIMATE_PSI_3 ? "Psi3_Est" : "Psi3_$(numeric_tag(PSI_3))"
    _, _, β_naming, _ = get_fixed_parameters()

    # Output path for results (includes psi and psi_3 tags in directory name)
    output_dir = "C:/Users/wbras/OneDrive/Documents/Desktop/UA/4th_Year_Paper/4th_Year_Paper_Data/HMS/2021-Onward/Dynamic_Model/CF_Mixture_$(psi_tag)_$(psi_3_tag)_Results"

    # Create output directory if it doesn't exist
    mkpath(output_dir)

    # Set working directory to where the data CSVs live
    cd("C:/Users/wbras/OneDrive/Documents/Desktop/UA/4th_Year_Paper/4th_Year_Paper_Data/HMS/2021-Onward/Dynamic_Model/Data")
end


#############################
# 2. Output Paths
#############################

# Set log file path
log_path = joinpath(output_dir, "CF_Mixture_Log.txt")

# Open log file for writing (log_io is defined as a global in 01_Functions.jl)
log_io = open(log_path, "w")

# Print and log counterfactual simulation start time
log_msg("Counterfactual mixture simulation started at $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")


#############################
# 3. Initialize Fixed Parameters
#############################

# Load fixed parameters:
#   ψ_2 = fast addiction decay rate (always fixed at 0.90)
#   ψ_1 = slow addiction decay rate (always fixed at 0.10)
#   β = present bias (fixed at 1.0; overridden by BETA_GRID in loop)
#   δ = monthly discount factor (fixed at 0.99)
ψ_2, ψ_1, β, δ = get_fixed_parameters();


#############################
# 4. State Spaces and Choices
#############################

# Start timer for data prep
t_setup = time();

# Get fast addiction grid (N_A_f = 5 points, "craving" stock with ψ_2 = 0.90)
N_A_f, A_f = get_addiction_space(ψ_2; N_A=5);

# Get slow addiction grid (N_A_s = 10 points, "dependence" stock with ψ_1 = 0.10)
N_A_s, A_s = get_addiction_space(ψ_1; N_A=10);

# Get number of observations (N_HHT), number of alternatives (N_J), and choice matrix J
_, N_J, J = get_product_choices();

# Convert choice matrix J to choice vector y where y[i] = chosen alternative index for observation i
y = get_hh_choices(J);

# Get household identifiers (pre-loaded to avoid repeated CSV reads)
hh_codes = get_hh_codes();

# Pre-compute contiguous household index ranges for mixture posterior computation
hh_ranges = precompute_hh_ranges(hh_codes);

# Get number of product categories excluding the outside option
N_K, _ = get_category_choices();


#############################
# 5. Alternative-Level Vectors
#############################

# Get consumption vectors by alternative (STANDARDIZED by max)
# q_bundle is standardized by its own max (not q_cig_max x q_ecig_max) for reasonable α_CE scaling
# Max values are needed for rescaling parameter estimates to original units
N_cig, N_orig_ecig, N_non_fda_flav_ecig, N_fda_flav_ecig, _, q_cig, q_ecig, q_bundle, q_cig_max, q_ecig_max, q_bundle_max = get_consumption(N_J);

# Get nicotine vector by alternative (STANDARDIZED by max)
# n_max is the raw max value for rescaling estimates
n, n_max = get_nicotine(N_J);

# Get category index by alternative
cat_idx = get_category_index(N_J, N_cig, N_orig_ecig, N_non_fda_flav_ecig, N_fda_flav_ecig);

# Get flavored indicator by alternative: is_flavored[j] in {true, false} (any flavored: non-FDA or FDA)
is_flavored = get_flavored_indicator(cat_idx);

# Get FDA flavored indicator by alternative: is_fda_flavored[j] in {0, 1}
is_fda_flavored = get_fda_flavored_indicator(cat_idx);

# Get flavor lock-in indicators (split by product type)
# is_nonflavored_ecig: orig ecig (cat 2) and orig bundle (cat 5), ecig-side lock-in (γ_3)
is_nonflavored_ecig = [cat_idx[j] in (2, 5) for j in 1:N_J]

# Get indicator for alternatives containing cigarettes (cat 1 = cig, cat 5-7 = bundles with cig)
has_cig = [(cat_idx[j] == 1 || cat_idx[j] >= 5) for j in 1:N_J]

# Get indicator for alternatives containing e-cigarettes (cat 2-7 = any ecig or bundle)
has_ecig = [cat_idx[j] >= 2 for j in 1:N_J]

# Outside option indicator: cat 0 = outside option (j = 1)
is_outside = [cat_idx[j] == 0 for j in 1:N_J]


#############################
# 6. Demographics
#############################

# TYA states: load binary data (0 = no TYA, 1 = TYA present) and shift to 1-indexed
# This matches the estimation script: get_tya_states() returns 0/1, shift to 1/2.
tya_state = [s + 1 for s in get_tya_states()];

# Household-level TYA share (fraction of months with TYA present) for mixture weights
tya_share_hh = get_tya_share();


#############################
# 7. Price Space
#############################

# Get pricing grid: N_P points per category, P is N_P × 2 (cig, ecig)
N_P, P = get_pricing_spaces();

# Get all price combinations
N_Pcomb, Pcomb = get_pricing_spaces_combination(N_K, N_P, P);

# Get price ratios for quantity discount adjustment (price per unit varies by bin size)
ratio_cig, ratio_ecig = get_price_ratios(N_J, N_cig, N_orig_ecig, N_non_fda_flav_ecig, N_fda_flav_ecig, q_cig, q_ecig);

# Get Halton draw price transitions: T[m, r, k] where m = price state, r = draw, k = category
T = get_transitions(N_K);

# Pre-compute bilinear interpolation brackets and weights for price transitions
# Returns 6 matrices (M × R): lo/hi grid indices and weights for each category
p_cig_lo, p_cig_hi, p_cig_w, p_ecig_lo, p_ecig_hi, p_ecig_w = precompute_price_transitions(N_P, P, T);


#############################
# 8. Household Price Trajectories
#############################

# Map observed household prices to continuous values for interpolation
# p_continuous is N × 2 (cig price, ecig price): actual per-unit prices, not grid indices
# P_obs_cig / P_obs_ecig are N × N_J matrices of bin-specific prices (not used in CF, discarded)
_, p_continuous, _, _ = map_prices_to_grid(N_P, P, Pcomb, N_J);

# Log data setup completion time and sample size
setup_elapsed = time() - t_setup;
log_msg("Data loading complete in $(round(setup_elapsed, digits=1))s")
log_msg("Observations: $(length(y)), Alternatives: $N_J, Fast addiction states: $N_A_f, Slow addiction states: $N_A_s, Price states: $N_Pcomb")


#############################
# 9. AR(1) Price Parameters
#############################

# Load AR(1) coefficients: phi_0 (intercept) and phi_1 (AR coefficient)
AR_Phi = CSV.read("../AR_Parameters/AR_Parameters_Phi.csv", DataFrame);
φ_0 = [AR_Phi[1, :intercept], AR_Phi[2, :intercept]];
φ_1 = [AR_Phi[1, :ar1], AR_Phi[2, :ar1]];

# Load AR(1) shock covariance matrix Sigma
AR_Sigma = CSV.read("../AR_Parameters/AR_Parameters_Sigma.csv", DataFrame);
Σ = [AR_Sigma[1, :cig] AR_Sigma[1, :ecig];
     AR_Sigma[2, :cig] AR_Sigma[2, :ecig]];

# Cholesky decomposition: L such that LL' = Sigma
L_chol = cholesky(Σ).L;

# Print and log AR(1) parameter loading confirmation
log_msg("AR(1) parameters loaded")


#############################
# 10. Parameter Loading
#     (Per β — See Section 13)
#############################

# Structural parameters, PSI_3, and all β-dependent objects differ between
# β=0.70 and β=1.0 because each was estimated under its own β. These are
# loaded and pre-computed per β value in section 13 (below), after household
# state computation provides the context needed (N_HH, unique_hh, hh_obs).
# Each counterfactual loop unpacks the appropriate set for its β iteration.


#############################
# 11. Compute Household States
#############################

# This section reconstructs each household's continuous addiction stocks and
# prices at the END of the estimation sample. These terminal states are the
# initial conditions for the forward simulation in the counterfactual: each
# Monte Carlo path begins from where the household actually was at the last
# observed period, so the simulation continues the household's history rather
# than starting from an arbitrary baseline.
#
# Two nicotine addiction stocks are computed here for each household:
#   a_f: fast (craving) stock, decay ψ_2 = 0.90; responds within weeks
#   a_s: slow (dependence) stock, decay ψ_1 = 0.10; accumulates over months
# The flavored habit stock (ã_flav) depends on PSI_3, which differs by β, so it
# is built per β inside the pre-loading loop in section 13.
#
# For each stock, two steps are needed:
#   1. Fixed-point iteration for the initial stock at the FIRST observed period.
#      Because the panel is left-truncated (we don't observe pre-sample choices),
#      start at a_0 = 0, simulate forward with observed y,
#      treat the ending value as the new a_0, and repeat until convergence.
#   2. Forward simulation from a_0 through all T observed choices to recover
#      the continuous stock at every observation date.
#
# The terminal nicotine stocks (after the household's last observed choice) are
# the starting addiction for the forward simulation; the flavored habit stock
# is initialized per β in section 13.

# Print and log household state computation header
log_msg("\n===================================")
log_msg("Computing household addiction states...")
log_msg("===================================")

t_states = time();

# --- Fast addiction (craving) stock ---
# Fixed-point initial stock: iterate a_f0 until the stock implied by observed
# choices converges. ψ_2 = 0.90 means craving dissipates quickly (~1 month),
# so convergence is fast (typically < 10 iterations).
af0, max_fp_iters_f = get_initial_addiction_stock(ψ_2, A_f, n, y, hh_codes);
log_msg("Initial fast addiction stocks: max fixed-point iterations = $max_fp_iters_f")

# Forward simulation: apply a_f' = (1 - ψ_2) * a_f + ψ_2 * n_std[y_t] at
# each observed period to get af_continuous[i] = fast stock at the START of
# observation i (before the period-i choice is made).
af_continuous = simulate_addiction_trajectories(N_A_f, ψ_2, A_f, n, y, hh_codes, af0);

# --- Slow addiction (dependence) stock ---
# Fixed-point initial stock: same procedure, but ψ_1 = 0.10 means dependence
# builds slowly; the stock has a long memory (~10 months to half-decay), so
# more observations are needed before the initial condition matters less.
as0, max_fp_iters_s = get_initial_addiction_stock(ψ_1, A_s, n, y, hh_codes);
log_msg("Initial slow addiction stocks: max fixed-point iterations = $max_fp_iters_s")

# Forward simulation: apply a_s' = (1 - ψ_1) * a_s + ψ_1 * n_std[y_t].
as_continuous = simulate_addiction_trajectories(N_A_s, ψ_1, A_s, n, y, hh_codes, as0);

# n_flav is the binary input to the flavored habit stock: 1 if the chosen
# alternative is flavored, 0 otherwise. The habit grid, transitions, terminal
# stocks, and flow utilities all depend on PSI_3 which differs by β, so those
# objects are built per β in section 13.
n_flav = Float64.(is_flavored) # binary indicator: 1 if flavored, 0 otherwise

# --- Terminal states ---
# For each household, extract the state AFTER their last observed choice.
# af_continuous[last_obs] is the stock at the START of the last period; we
# apply one more evolution step with the last observed choice to get the stock
# the household carries into the first counterfactual period.
unique_hh = unique(hh_codes);
N_HH = length(unique_hh);
N_obs = length(y);

# Build a map from household code to its observation row indices so we can
# look up the last observation for each household in O(1).
hh_obs = Dict{eltype(hh_codes), Vector{Int}}()
for i in 1:N_obs
    hh = hh_codes[i]
    if !haskey(hh_obs, hh)
        hh_obs[hh] = Int[]
    end
    push!(hh_obs[hh], i)
end

# Allocate terminal-state vectors: one entry per unique household.
# hh_aflav0 (flavored habit) is β-dependent and allocated per β in section 13.
hh_tya = Vector{Int}(undef, N_HH)          # TYA status at last observation
hh_af0 = Vector{Float64}(undef, N_HH)      # fast addiction entering period T+1
hh_as0 = Vector{Float64}(undef, N_HH)      # slow addiction entering period T+1
hh_p0  = Matrix{Float64}(undef, N_HH, 2)   # [cig price, ecig price] at last obs

for (h_idx, hh) in enumerate(unique_hh)

    obs_indices = hh_obs[hh]
    last_obs = obs_indices[end]

    # TYA state carries forward unchanged (no TYA transitions in the model)
    hh_tya[h_idx] = tya_state[last_obs]

    # Terminal prices: use the last observed continuous price pair as the
    # starting price state for the AR(1) price process in the simulation.
    hh_p0[h_idx, 1] = p_continuous[last_obs, 1]
    hh_p0[h_idx, 2] = p_continuous[last_obs, 2]

    # Fast addiction after final choice: evolve one step from the stock at the
    # START of the last period using the last observed choice.
    hh_af0[h_idx] = addiction_evolution(ψ_2, af_continuous[last_obs], n[y[last_obs]])
    hh_af0[h_idx] = clamp(hh_af0[h_idx], A_f[1], A_f[end])

    # Slow addiction after final choice: same procedure with ψ_1.
    hh_as0[h_idx] = addiction_evolution(ψ_1, as_continuous[last_obs], n[y[last_obs]])
    hh_as0[h_idx] = clamp(hh_as0[h_idx], A_s[1], A_s[end])

end

# Print and log household state computation results
states_elapsed = time() - t_states;
log_msg("Household states computed in $(round(states_elapsed, digits=1))s")
log_msg("Unique households: $N_HH")
log_msg("Mean terminal fast addiction: $(round(mean(hh_af0), digits=4))")
log_msg("Mean terminal slow addiction: $(round(mean(hh_as0), digits=4))")
log_msg("Mean terminal flavored habit: logged per β in section 13")


#############################
# 12. Pre-compute Addiction
#     Transitions
#############################

# Fast stock transitions (always pre-computed, ψ_2 is never estimated)
af_lower, af_upper, af_weight = precompute_addiction_transitions(N_J, N_A_f, ψ_2, A_f, n);

# Slow stock transitions (ψ_1 = 0.10 always fixed)
as_lower, as_upper, as_weight = precompute_addiction_transitions(N_J, N_A_s, ψ_1, A_s, n);

# β grid for present-bias sweep (shared by all counterfactuals; must be defined here
# so the pre-loading loop in section 13 can iterate over it before the ban/tax loops)
BETA_GRID = [0.70, 1.0]


#############################
# 13. β-Specific Parameter
#     Pre-loading
#############################

# Structural parameters (θ_struct_k), mixing weights (π), and PSI_3 all differ
# between β=0.70 and β=1.0 because each set was estimated under its own β.
# Flow utilities (U_k) and the flavored habit objects (A_flav grid, transitions,
# trajectories, terminal stocks) also differ because they depend on these estimates.
#
# This block loads estimates and pre-computes all β-specific objects for every β
# value in BETA_GRID, storing results in beta_params[β]. Each counterfactual
# loop then unpacks the appropriate set at the start of its β iteration, ensuring
# that VFI, posterior computation, and forward simulation all use the correct
# estimates.

log_msg("\n===================================")
log_msg("Pre-loading β-specific parameter sets...")
log_msg("===================================")

beta_params = Dict{Float64, Any}()

for _bv in BETA_GRID

    _beta_tag = "Beta_$(numeric_tag(_bv))"

    if HPC
        _est_dir = "/home/u2/wbrasic/4th_Year_Paper/Dynamic_Model/02_Second_Stage_Estimation_V2/Dynamic_Model_Mixture_V2_$(psi_tag)_$(_beta_tag)_$(psi_3_tag)_Results"
    else
        _est_dir = "C:/Users/wbras/OneDrive/Documents/Desktop/UA/4th_Year_Paper/4th_Year_Paper_Data/HMS/2021-Onward/Dynamic_Model/Dynamic_Model_Mixture_$(psi_tag)_$(_beta_tag)_$(psi_3_tag)_Results"
    end

    _est_path = joinpath(
        _est_dir,
        "Dynamic_Model_Mixture_V2_$(psi_tag)_$(_beta_tag)_$(psi_3_tag)_Estimates",
        "Dynamic_Model_Mixture_V2_$(psi_tag)_$(_beta_tag)_$(psi_3_tag)_Estimates.csv"
    )

    _df_est = CSV.read(_est_path, DataFrame)
    if "NLL" in names(_df_est)
        select!(_df_est, Not(:NLL))
    end
    _param_names = names(_df_est)
    _N_params    = length(_param_names)
    _θ_hat       = Float64.(collect(_df_est[1, :]))

    log_msg(@sprintf("\nβ = %.2f — loaded from: %s", _bv, _est_path))
    log_msg("Parameters:")
    for k in 1:_N_params
        log_msg("  $(_param_names[k]) = $(_θ_hat[k])")
    end

    _common  = _θ_hat[1:13]
    _ξ_1     = _θ_hat[14:16]
    _ξ_2     = _θ_hat[17:19]
    _ξ_3     = _θ_hat[20:22]
    _π_0_2   = _θ_hat[23]
    _π_TYA_2 = _θ_hat[24]
    _π_0_3   = _θ_hat[25]
    _π_TYA_3 = _θ_hat[26]

    _PSI_3 = ESTIMATE_PSI_3 ? _θ_hat[27] : PSI_3
    if ESTIMATE_PSI_3
        log_msg(@sprintf("  ψ_3 = %.6f (estimated)", _PSI_3))
    else
        log_msg(@sprintf("  ψ_3 = %.6f (fixed from ENV)", _PSI_3))
    end

    _θ_struct_1 = vcat(_common, _ξ_1)
    _θ_struct_2 = vcat(_common, _ξ_2)
    _θ_struct_3 = vcat(_common, _ξ_3)

    log_msg(@sprintf("Mixing weights (β=%.2f): π_0_2=%.4f  π_TYA_2=%.4f  π_0_3=%.4f  π_TYA_3=%.4f",
        _bv, _π_0_2, _π_TYA_2, _π_0_3, _π_TYA_3))
    for tya_val in (0.0, 1.0)
        e1 = 1.0; e2 = exp(_π_0_2 + _π_TYA_2 * tya_val); e3 = exp(_π_0_3 + _π_TYA_3 * tya_val)
        denom = e1 + e2 + e3
        log_msg(@sprintf("  tya=%.0f: P(k=1)=%.4f  P(k=2)=%.4f  P(k=3)=%.4f",
            tya_val, e1/denom, e2/denom, e3/denom))
    end

    # Flavored habit objects (PSI_3-dependent; differ across β when ESTIMATE_PSI_3=true)
    _N_A_flav, _A_flav = get_addiction_space(_PSI_3; N_A=10)
    _aflav_lower, _aflav_upper, _aflav_weight = precompute_addiction_transitions(
        N_J, _N_A_flav, _PSI_3, _A_flav, n_flav)
    _aflav0, _ = get_initial_addiction_stock(_PSI_3, _A_flav, n_flav, y, hh_codes)
    _aflav_continuous = simulate_addiction_trajectories(
        _N_A_flav, _PSI_3, _A_flav, n_flav, y, hh_codes, _aflav0)

    _hh_aflav0 = Vector{Float64}(undef, N_HH)
    for (h_idx, hh) in enumerate(unique_hh)
        last_obs = hh_obs[hh][end]
        _hh_aflav0[h_idx] = addiction_evolution(_PSI_3, _aflav_continuous[last_obs], n_flav[y[last_obs]])
        _hh_aflav0[h_idx] = clamp(_hh_aflav0[h_idx], _A_flav[1], _A_flav[end])
    end
    log_msg(@sprintf("Mean terminal flavored habit (β=%.2f): %.4f", _bv, mean(_hh_aflav0)))

    # Flow utilities (depend on θ_struct_k which differs by β)
    log_msg(@sprintf("Computing flow utilities for β=%.2f...", _bv))
    _U_1 = get_flow_utility(
        _θ_struct_1, N_J, N_A_f, N_A_s, _N_A_flav, N_Pcomb, A_f, A_s, _A_flav,
        q_cig, q_ecig, q_bundle, is_flavored, is_fda_flavored, is_nonflavored_ecig,
        is_outside, cat_idx, Pcomb, has_cig, has_ecig
    )
    _U_2 = get_flow_utility(
        _θ_struct_2, N_J, N_A_f, N_A_s, _N_A_flav, N_Pcomb, A_f, A_s, _A_flav,
        q_cig, q_ecig, q_bundle, is_flavored, is_fda_flavored, is_nonflavored_ecig,
        is_outside, cat_idx, Pcomb, has_cig, has_ecig
    )
    _U_3 = get_flow_utility(
        _θ_struct_3, N_J, N_A_f, N_A_s, _N_A_flav, N_Pcomb, A_f, A_s, _A_flav,
        q_cig, q_ecig, q_bundle, is_flavored, is_fda_flavored, is_nonflavored_ecig,
        is_outside, cat_idx, Pcomb, has_cig, has_ecig
    )
    log_msg(@sprintf("Flow utilities computed for β=%.2f", _bv))

    beta_params[_bv] = (
        common           = _common,
        θ_struct_1       = _θ_struct_1,
        θ_struct_2       = _θ_struct_2,
        θ_struct_3       = _θ_struct_3,
        PSI_3            = _PSI_3,
        N_A_flav         = _N_A_flav,
        A_flav           = _A_flav,
        aflav_lower      = _aflav_lower,
        aflav_upper      = _aflav_upper,
        aflav_weight     = _aflav_weight,
        aflav_continuous = _aflav_continuous,
        hh_aflav0        = _hh_aflav0,
        U_1              = _U_1,
        U_2              = _U_2,
        U_3              = _U_3,
        π_0_2            = _π_0_2,
        π_TYA_2          = _π_TYA_2,
        π_0_3            = _π_0_3,
        π_TYA_3          = _π_TYA_3,
        omega_E_est      = _common[13]
    )

end  # end β pre-loading loop

log_msg("\nAll β-specific parameter sets pre-loaded.")


#############################
# 14. Pre-draw CRN Variates
#     (Once, Before Any Loop)
#############################

# Every (household, period, draw) triple in the status quo simulation
# uses the exact same uniform and normal draws as the corresponding triple in
# the counterfactual simulation. Because the only thing that differs between the
# two runs is the policy (ban or tax applied to flow utilities), the difference
# in outcomes is a pure policy effect
#
# Without CRN, SQ and CF paths would draw independent randoms. That introduces
# simulation noise into the SQ-minus-CF difference, requiring many more draws
# to achieve the same precision. CRN is a standard variance-reduction technique
#
# All variates are drawn ONCE here, before any counterfactual loop, and reused
# across every β value and every policy scenario. This guarantees that all
# results in the output are comparable: differences across β and across policies
# cannot be attributed to different random seeds.
#
# Three arrays are pre-drawn:
#   crn_choice[h, t, d]    - U(0,1) used for inverse-CDF choice sampling at
#                            period t of draw d for household h.
#   crn_price[h, t, d, k]  - N(0,1) used as the raw shock for price dimension k
#                            (k=1: cigarette, k=2: e-cig) in the AR(1) price
#                            process. The Cholesky factor converts these
#                            independent normals into correlated price shocks.
#   crn_type[h, d]          - U(0,1) used once per (household, draw) to sample
#                            a latent type from the K=3 posterior via inverse CDF
#                            on the cumulative posterior weights.

# Simulation settings
T_sim    = 36     # months to simulate forward (3 years)
N_draws  = 100    # Monte Carlo draws per household
crn_seed = 12345  # fixed seed for reproducibility across runs

log_msg("\nSimulation settings: T_sim = $T_sim, N_draws = $N_draws, N_HH = $N_HH, CRN seed = $crn_seed")

Random.seed!(crn_seed)

# --- Choice draws ---
# crn_choice[h, t, d]: one U(0,1) per (household, period, draw).
#
# At simulation time, the softmax probabilities (p_1, ..., p_J) are accumulated
# into a CDF: CDF[j] = p_1 + ... + p_j. Drawing u ~ U(0,1) and returning the
# first j with CDF[j] >= u is the inverse-CDF (quantile) transform for a
# discrete distribution. By the probability integral transform, the chosen
# alternative j* satisfies P(j* = k) = P(CDF[k-1] < u <= CDF[k]) = p_k, so
# the draw is exactly distributed according to the softmax probabilities.
# Using the same u for SQ and CF means both scenarios make the same choice
# whenever their softmax probabilities agree, isolating the policy's effect.
crn_choice = Array{Float64}(undef, N_HH, T_sim, N_draws)
for d in 1:N_draws
    for t in 1:T_sim
        for h in 1:N_HH
            crn_choice[h, t, d] = rand()
        end
    end
end

# --- Price shock draws ---
# crn_price[h, t, d, k]: one N(0,1) per (household, period, draw, price dimension).
# The AR(1) price process advances as:
#   p_t = φ_0 + φ_1 * p_{t-1} + L_chol * z_t,   z_t ~ N(0, I_2)
# where φ_0 and φ_1 are the regression intercept and slope estimated in
# 03_AR_Estimation.R and L_chol is the Cholesky factor of the residual
# covariance matrix Σ estimated from the AR(1) residuals. The two independent
# standard normals stored here become correlated price shocks after
# multiplication by L_chol inside the simulation loop.
crn_price = Array{Float64}(undef, N_HH, T_sim, N_draws, 2)
for d in 1:N_draws
    for t in 1:T_sim
        for h in 1:N_HH
            crn_price[h, t, d, 1] = randn()   # cigarette price shock (pre-Cholesky)
            crn_price[h, t, d, 2] = randn()   # e-cig price shock (pre-Cholesky)
        end
    end
end

# --- Type assignment draws ---
# crn_type[h, d]: one U(0,1) per (household, draw), used to sample a latent
# type from the household's K=3 posterior at the START of each path (before
# any period-t choice is made). Sampling is via inverse CDF on the cumulative
# posterior weights [w_1, w_1+w_2, 1]: u <= w_1 → type 1;
# u ∈ (w_1, w_1+w_2] → type 2; else type 3. Using the same draw for SQ and CF ensures that the
# same type is simulated in both scenarios for each (h, d) pair.
crn_type = Array{Float64}(undef, N_HH, N_draws)
for d in 1:N_draws
    for h in 1:N_HH
        crn_type[h, d] = rand()
    end
end

log_msg("CRN variates pre-drawn: choice $(size(crn_choice)), price $(size(crn_price)), type $(size(crn_type))")


#############################
# 15. Category Labels
#############################

cat_labels = ["Outside", "Cig", "Orig Ecig", "Non-FDA Flav Ecig", "FDA Flav Ecig", "Orig Bundle", "Non-FDA Flav Bundle", "FDA Flav Bundle"];


#############################
# 16. Ban Type Loop
#############################

# Tuple of ban types and ban functions
BAN_TYPES = [
    ("Ban_Comprehensive", apply_flavor_ban!),
    ("Ban_FDA_Only", apply_fda_flavor_ban!),
    ("Ban_Non_FDA", apply_non_fda_ban!)
]

log_msg("\n===================================")
log_msg("Starting ban type loop: $(length(BAN_TYPES)) ban types")
log_msg("===================================")

for (ban_label, ban_fn!) in BAN_TYPES

    log_msg("\n\n###################################")
    log_msg("Ban type: $ban_label")
    log_msg("###################################")

    # Create ban-specific subdirectory
    ban_subdir = joinpath(output_dir, ban_label)
    mkpath(ban_subdir)

    # Log which alternatives are banned (ban pattern is β-independent; use first β's U_1).
    # Defined here (not inside a let block) so banned_alts is accessible in the β loop below.
    _U_log_tmp = copy(beta_params[first(BETA_GRID)].U_1)
    ban_fn!(_U_log_tmp, cat_idx)
    banned_alts = findall(j -> _U_log_tmp[1, j, 1, 1, 1, 1] == -Inf, 1:N_J)
    log_msg("Banned alternatives: j = $(banned_alts)")
    _U_log_tmp = nothing  # free memory


    #############################
    # β Grid Loop
    #############################

    log_msg("\n===================================")
    log_msg("Starting β grid loop: $(length(BETA_GRID)) values")
    log_msg("===================================")

    for beta_val in BETA_GRID

        # Declare all loop-local variables to avoid Julia warnings.
        local beta_numeric, beta_subdir
        local t_vfi_sq, t_vfi_ban, vfi_sq_elapsed, vfi_ban_elapsed
        local t_post, post_elapsed, t_pw, pw_elapsed
        local t_div, div_elapsed, t_sim_fwd, sim_fwd_elapsed
        local probs_sq, probs_ban, welfare_sq, welfare_ban, welfare_diff
        local hh_posterior_type1, hh_posterior_type2, hh_posterior_type3
        local obs_posterior_type1, obs_posterior_type2, obs_posterior_type3
        local sim_choices_sq, sim_addiction_f_sq, sim_addiction_s_sq, sim_aflav_sq, sim_welfare_sq_arr
        local sim_choices_ban, sim_addiction_f_ban, sim_addiction_s_ban, sim_aflav_ban, sim_welfare_ban_arr
        local sim_addiction_sq, sim_addiction_ban
        local agg_sq, agg_ban, agg_sq_tya, agg_sq_no_tya, agg_ban_tya, agg_ban_no_tya
        local agg_sq_type1, agg_sq_type2, agg_sq_type3, agg_ban_type1, agg_ban_type2, agg_ban_type3

        # Unpack β-specific objects pre-loaded in section 13
        local _bp = beta_params[beta_val]
        local U_1 = _bp.U_1; local U_2 = _bp.U_2; local U_3 = _bp.U_3
        local PSI_3 = _bp.PSI_3
        local N_A_flav = _bp.N_A_flav; local A_flav = _bp.A_flav
        local aflav_lower = _bp.aflav_lower; local aflav_upper = _bp.aflav_upper
        local aflav_weight = _bp.aflav_weight
        local aflav_continuous = _bp.aflav_continuous
        local hh_aflav0 = _bp.hh_aflav0
        local π_0_2 = _bp.π_0_2; local π_TYA_2 = _bp.π_TYA_2
        local π_0_3 = _bp.π_0_3; local π_TYA_3 = _bp.π_TYA_3
        local omega_E_est = _bp.omega_E_est

        # Apply ban to this β's base flow utilities
        local U_1_ban = copy(U_1); ban_fn!(U_1_ban, cat_idx)
        local U_2_ban = copy(U_2); ban_fn!(U_2_ban, cat_idx)
        local U_3_ban = copy(U_3); ban_fn!(U_3_ban, cat_idx)

        # Numeric tag for directory naming: 0.75 -> "075", 0.85 -> "085", etc.
        beta_numeric = numeric_tag(beta_val)
        beta_subdir = joinpath(ban_subdir, "Beta_$(beta_numeric)")
        mkpath(beta_subdir)

        log_msg("\n\n###################################")
        log_msg("β = $beta_val (tag: $beta_numeric)")
        log_msg("###################################")


        #############################
        # Solve VFI: Status Quo
        #############################

        log_msg("\n===================================")
        log_msg("Solving VFI: Status Quo (β = $beta_val)")
        log_msg("===================================")

        t_vfi_sq = time();

        # Solve VFI for both types in parallel via Threads.@spawn
        vfi_task_1_sq = Threads.@spawn solve_vfi_sophisticated(
            N_J, N_A_f, N_A_s, N_A_flav, N_P, N_Pcomb, beta_val, δ, U_1,
            af_lower, af_upper, af_weight,
            as_lower, as_upper, as_weight,
            aflav_lower, aflav_upper, aflav_weight,
            p_cig_lo, p_cig_hi, p_cig_w,
            p_ecig_lo, p_ecig_hi, p_ecig_w;
            V_init = nothing,
            verbose = true
        )

        vfi_task_2_sq = Threads.@spawn solve_vfi_sophisticated(
            N_J, N_A_f, N_A_s, N_A_flav, N_P, N_Pcomb, beta_val, δ, U_2,
            af_lower, af_upper, af_weight,
            as_lower, as_upper, as_weight,
            aflav_lower, aflav_upper, aflav_weight,
            p_cig_lo, p_cig_hi, p_cig_w,
            p_ecig_lo, p_ecig_hi, p_ecig_w;
            V_init = nothing,
            verbose = true
        )

        vfi_task_3_sq = Threads.@spawn solve_vfi_sophisticated(
            N_J, N_A_f, N_A_s, N_A_flav, N_P, N_Pcomb, beta_val, δ, U_3,
            af_lower, af_upper, af_weight,
            as_lower, as_upper, as_weight,
            aflav_lower, aflav_upper, aflav_weight,
            p_cig_lo, p_cig_hi, p_cig_w,
            p_ecig_lo, p_ecig_hi, p_ecig_w;
            V_init = nothing,
            verbose = true
        )

        # Wait for all VFI tasks to complete
        _, V_decision_1_sq, vfi_iters_1_sq, vfi_converged_1_sq = fetch(vfi_task_1_sq)
        _, V_decision_2_sq, vfi_iters_2_sq, vfi_converged_2_sq = fetch(vfi_task_2_sq)
        _, V_decision_3_sq, vfi_iters_3_sq, vfi_converged_3_sq = fetch(vfi_task_3_sq)

        vfi_sq_elapsed = time() - t_vfi_sq;
        log_msg("Status quo VFI: type1=$(vfi_iters_1_sq) iters ($(vfi_converged_1_sq)), type2=$(vfi_iters_2_sq) iters ($(vfi_converged_2_sq)), type3=$(vfi_iters_3_sq) iters ($(vfi_converged_3_sq)), $(round(vfi_sq_elapsed, digits=1))s")


        #############################
        # Solve VFI: Flavor Ban
        #############################

        log_msg("\n===================================")
        log_msg("Solving VFI: Flavor Ban (β = $beta_val)")
        log_msg("===================================")

        t_vfi_ban = time();

        # Solve VFI for both types under ban in parallel
        vfi_task_1_ban = Threads.@spawn solve_vfi_sophisticated(
            N_J, N_A_f, N_A_s, N_A_flav, N_P, N_Pcomb, beta_val, δ, U_1_ban,
            af_lower, af_upper, af_weight,
            as_lower, as_upper, as_weight,
            aflav_lower, aflav_upper, aflav_weight,
            p_cig_lo, p_cig_hi, p_cig_w,
            p_ecig_lo, p_ecig_hi, p_ecig_w;
            V_init = nothing,
            verbose = true
        )

        vfi_task_2_ban = Threads.@spawn solve_vfi_sophisticated(
            N_J, N_A_f, N_A_s, N_A_flav, N_P, N_Pcomb, beta_val, δ, U_2_ban,
            af_lower, af_upper, af_weight,
            as_lower, as_upper, as_weight,
            aflav_lower, aflav_upper, aflav_weight,
            p_cig_lo, p_cig_hi, p_cig_w,
            p_ecig_lo, p_ecig_hi, p_ecig_w;
            V_init = nothing,
            verbose = true
        )

        vfi_task_3_ban = Threads.@spawn solve_vfi_sophisticated(
            N_J, N_A_f, N_A_s, N_A_flav, N_P, N_Pcomb, beta_val, δ, U_3_ban,
            af_lower, af_upper, af_weight,
            as_lower, as_upper, as_weight,
            aflav_lower, aflav_upper, aflav_weight,
            p_cig_lo, p_cig_hi, p_cig_w,
            p_ecig_lo, p_ecig_hi, p_ecig_w;
            V_init = nothing,
            verbose = true
        )

        # Wait for all VFI tasks to complete
        _, V_decision_1_ban, vfi_iters_1_ban, vfi_converged_1_ban = fetch(vfi_task_1_ban)
        _, V_decision_2_ban, vfi_iters_2_ban, vfi_converged_2_ban = fetch(vfi_task_2_ban)
        _, V_decision_3_ban, vfi_iters_3_ban, vfi_converged_3_ban = fetch(vfi_task_3_ban)

        vfi_ban_elapsed = time() - t_vfi_ban;
        log_msg("Flavor ban VFI: type1=$(vfi_iters_1_ban) iters ($(vfi_converged_1_ban)), type2=$(vfi_iters_2_ban) iters ($(vfi_converged_2_ban)), type3=$(vfi_iters_3_ban) iters ($(vfi_converged_3_ban)), $(round(vfi_ban_elapsed, digits=1))s")


        #############################
        # Compute Posterior Type
        # Weights
        #############################

        log_msg("\n===================================")
        log_msg("Computing posterior type weights...")
        log_msg("===================================")

        t_post = time();

        # Compute per-type choice probabilities for all observations under status quo
        probs_1_sq, welfare_1_sq = compute_pointwise_outcomes(
            V_decision_1_sq, tya_state, af_continuous, as_continuous, aflav_continuous, p_continuous, N_J, N_P, A_f, A_s, A_flav, P
        )
        probs_2_sq, welfare_2_sq = compute_pointwise_outcomes(
            V_decision_2_sq, tya_state, af_continuous, as_continuous, aflav_continuous, p_continuous, N_J, N_P, A_f, A_s, A_flav, P
        )
        probs_3_sq, welfare_3_sq = compute_pointwise_outcomes(
            V_decision_3_sq, tya_state, af_continuous, as_continuous, aflav_continuous, p_continuous, N_J, N_P, A_f, A_s, A_flav, P
        )

        # Compute per-type choice probabilities for all observations under ban
        probs_1_ban, welfare_1_ban = compute_pointwise_outcomes(
            V_decision_1_ban, tya_state, af_continuous, as_continuous, aflav_continuous, p_continuous, N_J, N_P, A_f, A_s, A_flav, P
        )
        probs_2_ban, welfare_2_ban = compute_pointwise_outcomes(
            V_decision_2_ban, tya_state, af_continuous, as_continuous, aflav_continuous, p_continuous, N_J, N_P, A_f, A_s, A_flav, P
        )
        probs_3_ban, welfare_3_ban = compute_pointwise_outcomes(
            V_decision_3_ban, tya_state, af_continuous, as_continuous, aflav_continuous, p_continuous, N_J, N_P, A_f, A_s, A_flav, P
        )

        # Compute household-level posterior type weights using status quo probabilities
        # For each household h: P(type=k | data_h) proportional to pi_k_h * prod_t P(y_ht | x_ht, type=k)
        hh_posterior_type1 = Vector{Float64}(undef, N_HH)
        hh_posterior_type2 = Vector{Float64}(undef, N_HH)
        hh_posterior_type3 = Vector{Float64}(undef, N_HH)

        for h in 1:N_HH
            start_idx, stop_idx = hh_ranges[h]

            # Household-specific K=3 softmax mixing weights
            η_2_h = π_0_2 + π_TYA_2 * tya_share_hh[h]
            η_3_h = π_0_3 + π_TYA_3 * tya_share_hh[h]
            log_sum_exp_h = log(1.0 + exp(η_2_h) + exp(η_3_h))
            log_π_1_h = -log_sum_exp_h
            log_π_2_h = η_2_h - log_sum_exp_h
            log_π_3_h = η_3_h - log_sum_exp_h

            # Accumulate per-type log-likelihood for this household
            ll_1 = 0.0
            ll_2 = 0.0
            ll_3 = 0.0
            for i in start_idx:stop_idx
                ll_1 += log(max(probs_1_sq[i, y[i]], 1e-300))
                ll_2 += log(max(probs_2_sq[i, y[i]], 1e-300))
                ll_3 += log(max(probs_3_sq[i, y[i]], 1e-300))
            end

            # Posterior via logsumexp normalization
            a_term = log_π_1_h + ll_1
            b_term = log_π_2_h + ll_2
            c_term = log_π_3_h + ll_3
            log_max = max(a_term, b_term, c_term)
            log_denom = log_max + log(exp(a_term - log_max) + exp(b_term - log_max) + exp(c_term - log_max))

            hh_posterior_type1[h] = exp(a_term - log_denom)
            hh_posterior_type2[h] = exp(b_term - log_denom)
            hh_posterior_type3[h] = exp(c_term - log_denom)
        end

        post_elapsed = time() - t_post;
        mean_w1 = mean(hh_posterior_type1)
        mean_w2 = mean(hh_posterior_type2)
        mean_w3 = mean(hh_posterior_type3)
        log_msg("Posterior type weights computed in $(round(post_elapsed, digits=1))s")
        log_msg(@sprintf("  Mean P(type=1) = %.4f, Mean P(type=2) = %.4f, Mean P(type=3) = %.4f", mean_w1, mean_w2, mean_w3))


        #############################
        # Pointwise Outcomes
        # (Mixture-Weighted)
        #############################

        log_msg("\n===================================")
        log_msg("Computing mixture-weighted pointwise outcomes...")
        log_msg("===================================")

        t_pw = time();

        # Expand household posterior to observation level for weighting
        obs_posterior_type1 = Vector{Float64}(undef, N_obs)
        obs_posterior_type2 = Vector{Float64}(undef, N_obs)
        obs_posterior_type3 = Vector{Float64}(undef, N_obs)
        for h in 1:N_HH
            start_idx, stop_idx = hh_ranges[h]
            for i in start_idx:stop_idx
                obs_posterior_type1[i] = hh_posterior_type1[h]
                obs_posterior_type2[i] = hh_posterior_type2[h]
                obs_posterior_type3[i] = hh_posterior_type3[h]
            end
        end

        # Mixture-weighted pointwise outcomes — law of total probability over the latent type:
        #   P̂(j|i) = Σ_l P̂(l|y_h) · P(j|l, x_i)
        # Integrates out type uncertainty; hard-assigning each HH to its modal type would
        # discard posterior mass on non-modal types and bias predicted in-sample market shares.
        probs_sq   = obs_posterior_type1 .* probs_1_sq  .+ obs_posterior_type2 .* probs_2_sq  .+ obs_posterior_type3 .* probs_3_sq
        welfare_sq = obs_posterior_type1 .* welfare_1_sq .+ obs_posterior_type2 .* welfare_2_sq .+ obs_posterior_type3 .* welfare_3_sq
        probs_ban   = obs_posterior_type1 .* probs_1_ban .+ obs_posterior_type2 .* probs_2_ban .+ obs_posterior_type3 .* probs_3_ban
        welfare_ban = obs_posterior_type1 .* welfare_1_ban .+ obs_posterior_type2 .* welfare_2_ban .+ obs_posterior_type3 .* welfare_3_ban
        pw_elapsed = time() - t_pw;
        log_msg("Pointwise outcomes computed in $(round(pw_elapsed, digits=1))s")

        # Verification: banned alternatives should have exactly zero probability under ban
        max_banned_prob = maximum(probs_ban[:, banned_alts]);
        log_msg("Max probability of banned alternatives under ban: $max_banned_prob")

        # Verification: welfare under ban should be <= welfare under status quo
        welfare_diff = welfare_ban .- welfare_sq;
        max_welfare_increase = maximum(welfare_diff);
        log_msg("Max welfare increase under ban (should be <= 0): $max_welfare_increase")

        # Print and log pointwise summary statistics
        log_msg("\nPointwise summary (means across all observations):")
        log_msg(@sprintf("  %-22s  %12s  %12s  %12s", "Category", "SQ Share", "Ban Share", "Difference"))
        log_msg("  " * repeat("-", 62))

        for (c, label) in enumerate(cat_labels)
            cat_val = c - 1  # cat_idx values are 0-7
            alt_indices = findall(j -> cat_idx[j] == cat_val, 1:N_J)
            sq_share  = mean(sum(probs_sq[:, alt_indices], dims=2))
            ban_share = mean(sum(probs_ban[:, alt_indices], dims=2))
            log_msg(@sprintf("  %-22s  %12.6f  %12.6f  %12.6f", label, sq_share, ban_share, ban_share - sq_share))
        end

        log_msg(@sprintf("\n  Mean welfare SQ:   %.6f", mean(welfare_sq)))
        log_msg(@sprintf("  Mean welfare Ban:  %.6f", mean(welfare_ban)))
        log_msg(@sprintf("  Mean welfare loss: %.6f", mean(welfare_diff)))




        #############################
        # Forward Simulation
        # (Mixture with CRN)
        # (Does not depend on pricing
        #  scenario — uses AR(1)
        #  prices for both SQ and ban)
        #############################

        log_msg("\n===================================")
        log_msg("Forward simulation (β = $beta_val)...")
        log_msg("===================================")

        log_msg("T_sim = $T_sim, N_draws = $N_draws, N_HH = $N_HH")

        # Forward simulation overview:
        #   - For each household, assign to the latent type with higher posterior probability.
        #   - Simulate T_sim periods under both SQ and ban using the assigned type's V_decision.
        #   - Both scenarios use identical pre-drawn CRN variates (common random numbers) to
        #     reduce variance of the SQ-vs-ban difference. The same uniform draw determines the
        #     choice in both scenarios, and the same normal draws drive AR(1) price shocks.
        #   - Each draw starts from the household's terminal addiction state (a_T), which is the
        #     addiction level after the household's final observed choice.
        #   - Prices evolve via AR(1) in both scenarios (no supply-side re-equilibration in the
        #     forward simulation; supply-side effects are captured in the pointwise analysis above).
        #   - TYA state is held fixed at the household's last observed TYA state.

        # Initialize simulation output arrays (N_HH x T_sim x N_draws)
        sim_choices_sq      = Array{Int}(undef, N_HH, T_sim, N_draws)
        sim_addiction_f_sq  = Array{Float64}(undef, N_HH, T_sim, N_draws)
        sim_addiction_s_sq  = Array{Float64}(undef, N_HH, T_sim, N_draws)
        sim_aflav_sq        = Array{Float64}(undef, N_HH, T_sim, N_draws)
        sim_welfare_sq_arr  = Array{Float64}(undef, N_HH, T_sim, N_draws)

        sim_choices_ban     = Array{Int}(undef, N_HH, T_sim, N_draws)
        sim_addiction_f_ban = Array{Float64}(undef, N_HH, T_sim, N_draws)
        sim_addiction_s_ban = Array{Float64}(undef, N_HH, T_sim, N_draws)
        sim_aflav_ban       = Array{Float64}(undef, N_HH, T_sim, N_draws)
        sim_welfare_ban_arr = Array{Float64}(undef, N_HH, T_sim, N_draws)

        # Price grid bounds for clamping AR(1) price draws
        P_cig  = P[:, 1]
        P_ecig = P[:, 2]

        t_sim_fwd = time();

        # Main simulation loop: interleaves SQ and ban within the same (h, d, t) iteration
        # to guarantee perfect CRN alignment (both scenarios consume the same random draws
        # in the same order). Parallelized over households since each h is independent.
        Threads.@threads for h in 1:N_HH

            # TYA state at household h's final observation (used as initial TYA for simulation;
            # held fixed at binary value throughout simulation)
            tya_idx_h = hh_tya[h]

            # Posterior type probability for this household (conditions on full purchase history)
            w1 = hh_posterior_type1[h]
            w2 = hh_posterior_type2[h]

            for d in 1:N_draws

                # Probabilistic type assignment: draw type from posterior using pre-drawn uniform.
                # If u <= P(type=1) → type 1; elif u <= P(type=1)+P(type=2) → type 2; else type 3.
                # This properly marginalizes over type uncertainty across Monte Carlo draws.
                u_type = crn_type[h, d]
                type_k = u_type <= w1 ? 1 : (u_type <= w1 + w2 ? 2 : 3)
                V_sq  = type_k == 1 ? V_decision_1_sq  : (type_k == 2 ? V_decision_2_sq  : V_decision_3_sq)
                V_ban = type_k == 1 ? V_decision_1_ban : (type_k == 2 ? V_decision_2_ban : V_decision_3_ban)

                # Reset to initial (terminal) state for each draw.
                # hh_af0/hh_as0 are the addiction levels AFTER the household's final observed
                # choice, computed via addiction_evolution(psi, a_last, n[y_last]).
                # hh_p0 are the prices at the household's last observed period.

                # Status quo initial state
                a_f_sq      = hh_af0[h]
                a_s_sq      = hh_as0[h]
                a_flav_sq   = hh_aflav0[h]
                p_cig_sq    = hh_p0[h, 1]
                p_ecig_sq   = hh_p0[h, 2]

                # Ban initial state (same starting point as SQ)
                a_f_ban     = hh_af0[h]
                a_s_ban     = hh_as0[h]
                a_flav_ban  = hh_aflav0[h]
                p_cig_ban   = hh_p0[h, 1]
                p_ecig_ban  = hh_p0[h, 2]

                # TYA state initialized from last observation (held fixed, binary, no transitions)
                tya_sq  = tya_idx_h
                tya_ban = tya_idx_h

                for t in 1:T_sim

                    # --- STATUS QUO ---

                    # 5-linear interpolation of V_decision at the household's continuous state
                    # (fast addiction, slow addiction, flavored habit, cig price, ecig price)
                    # for all J alternatives.
                    # NaN can arise from 0.0 * (-Inf) when an interpolation weight is exactly 0
                    # and the grid point has V = -Inf (banned alternative); replace with -Inf so
                    # exp(-Inf) = 0 in the softmax.
                    v_interp_sq = interpolate_v_choice(
                        V_sq, tya_sq, a_f_sq, a_s_sq, a_flav_sq, p_cig_sq, p_ecig_sq,
                        N_J, N_P, A_f, A_s, A_flav, P
                    )
                    replace!(v_interp_sq, NaN => -Inf)

                    # Welfare = logsumexp(V_decision) = expected maximum utility (inclusive value)
                    sim_welfare_sq_arr[h, t, d] = logsumexp(v_interp_sq)

                    # Softmax (logit) choice probabilities: P(j) = exp(V_j) / sum_j' exp(V_j')
                    v_max_sq = maximum(v_interp_sq)
                    v_shifted_sq = v_interp_sq .- v_max_sq
                    exp_v_sq = exp.(v_shifted_sq)
                    probs_h_sq = exp_v_sq ./ sum(exp_v_sq)

                    # Inverse CDF sampling: draw choice using the pre-drawn uniform u ~ U(0,1).
                    # The SAME u_draw is used for both SQ and ban to implement CRN.
                    u_draw = crn_choice[h, t, d]
                    j_sq = 1
                    cum_prob = probs_h_sq[1]
                    while cum_prob < u_draw && j_sq < N_J
                        j_sq += 1
                        cum_prob += probs_h_sq[j_sq]
                    end
                    sim_choices_sq[h, t, d] = j_sq

                    # Update fast addiction: a_f' = (1 - ψ_2) * a_f + ψ_2 * n_std[j]
                    # n[j] is the standardized nicotine content of chosen alternative j
                    a_f_sq = addiction_evolution(ψ_2, a_f_sq, n[j_sq])
                    a_f_sq = clamp(a_f_sq, A_f[1], A_f[end])
                    sim_addiction_f_sq[h, t, d] = a_f_sq

                    # Update slow addiction: a_s' = (1 - ψ_1) * a_s + ψ_1 * n_std[j]
                    a_s_sq = addiction_evolution(ψ_1, a_s_sq, n[j_sq])
                    a_s_sq = clamp(a_s_sq, A_s[1], A_s[end])
                    sim_addiction_s_sq[h, t, d] = a_s_sq

                    # Update flavored habit stock
                    a_flav_sq = addiction_evolution(PSI_3, a_flav_sq, n_flav[j_sq])
                    a_flav_sq = clamp(a_flav_sq, A_flav[1], A_flav[end])
                    sim_aflav_sq[h, t, d] = a_flav_sq

                    # Update prices via AR(1): p' = phi_0 + phi_1 * p + L_chol * z
                    # where z ~ N(0, I_2) are pre-drawn standard normals. L_chol is the
                    # Cholesky factor of the AR(1) innovation covariance matrix Sigma,
                    # so L_chol * z has the correct correlation structure.
                    # Same price draws used for both SQ and ban (CRN).
                    ε_sq = L_chol * [crn_price[h, t, d, 1], crn_price[h, t, d, 2]]
                    p_cig_sq  = clamp(φ_0[1] + φ_1[1] * p_cig_sq  + ε_sq[1], P_cig[1], P_cig[end])
                    p_ecig_sq = clamp(φ_0[2] + φ_1[2] * p_ecig_sq + ε_sq[2], P_ecig[1], P_ecig[end])

                    # TYA state held fixed (binary, no transitions; consistent with estimation VFI)

                    # --- FLAVOR BAN ---
                    # Same simulation steps as SQ but using the ban's V_decision (which has
                    # -Inf for flavored alternatives, forcing zero choice probability for those).

                    # Interpolate ban V_decision at current ban state
                    v_interp_ban = interpolate_v_choice(
                        V_ban, tya_ban, a_f_ban, a_s_ban, a_flav_ban, p_cig_ban, p_ecig_ban,
                        N_J, N_P, A_f, A_s, A_flav, P
                    )
                    replace!(v_interp_ban, NaN => -Inf)

                    # Welfare at current state (before choice)
                    sim_welfare_ban_arr[h, t, d] = logsumexp(v_interp_ban)

                    # Softmax choice probabilities
                    v_max_ban = maximum(v_interp_ban)
                    v_shifted_ban = v_interp_ban .- v_max_ban
                    exp_v_ban = exp.(v_shifted_ban)
                    probs_h_ban = exp_v_ban ./ sum(exp_v_ban)

                    # Draw choice using same pre-drawn uniform (CRN)
                    j_ban = 1
                    cum_prob_ban = probs_h_ban[1]
                    while cum_prob_ban < u_draw && j_ban < N_J
                        j_ban += 1
                        cum_prob_ban += probs_h_ban[j_ban]
                    end
                    sim_choices_ban[h, t, d] = j_ban

                    # Update fast addiction
                    a_f_ban = addiction_evolution(ψ_2, a_f_ban, n[j_ban])
                    a_f_ban = clamp(a_f_ban, A_f[1], A_f[end])
                    sim_addiction_f_ban[h, t, d] = a_f_ban

                    # Update slow addiction
                    a_s_ban = addiction_evolution(ψ_1, a_s_ban, n[j_ban])
                    a_s_ban = clamp(a_s_ban, A_s[1], A_s[end])
                    sim_addiction_s_ban[h, t, d] = a_s_ban

                    # Update flavored habit stock
                    a_flav_ban = addiction_evolution(PSI_3, a_flav_ban, n_flav[j_ban])
                    a_flav_ban = clamp(a_flav_ban, A_flav[1], A_flav[end])
                    sim_aflav_ban[h, t, d] = a_flav_ban

                    # Update prices via AR(1) with correlated shocks (same CRN draws)
                    ε_ban = L_chol * [crn_price[h, t, d, 1], crn_price[h, t, d, 2]]
                    p_cig_ban  = clamp(φ_0[1] + φ_1[1] * p_cig_ban  + ε_ban[1], P_cig[1], P_cig[end])
                    p_ecig_ban = clamp(φ_0[2] + φ_1[2] * p_ecig_ban + ε_ban[2], P_ecig[1], P_ecig[end])

                    # TYA state held fixed (binary, no transitions; consistent with estimation VFI)
                end
            end
        end

        sim_fwd_elapsed = time() - t_sim_fwd;
        log_msg("Forward simulation complete in $(round(sim_fwd_elapsed, digits=1))s")


        #############################
        # Aggregate and Save
        #############################

        log_msg("\n===================================")
        log_msg("Aggregating results...")
        log_msg("===================================")

        # Compute average addiction (fast + slow) / 2, consistent with flow utility specification
        sim_addiction_sq  = (sim_addiction_f_sq  .+ sim_addiction_s_sq)  ./ 2.0;
        sim_addiction_ban = (sim_addiction_f_ban .+ sim_addiction_s_ban) ./ 2.0;

        # Aggregate status quo
        agg_sq = aggregate_simulation(sim_choices_sq, sim_addiction_sq, sim_aflav_sq, sim_welfare_sq_arr, cat_idx, N_J, T_sim);

        # Aggregate flavor ban
        agg_ban = aggregate_simulation(sim_choices_ban, sim_addiction_ban, sim_aflav_ban, sim_welfare_ban_arr, cat_idx, N_J, T_sim);

        # Aggregate by TYA status
        agg_sq_tya, agg_sq_no_tya = aggregate_simulation_by_tya(
            sim_choices_sq, sim_addiction_sq, sim_aflav_sq, sim_welfare_sq_arr, cat_idx, N_J, T_sim, hh_tya, N_draws
        )
        agg_ban_tya, agg_ban_no_tya = aggregate_simulation_by_tya(
            sim_choices_ban, sim_addiction_ban, sim_aflav_ban, sim_welfare_ban_arr, cat_idx, N_J, T_sim, hh_tya, N_draws
        )

        # Aggregate by modal latent type (K=3 modal assignment)
        agg_sq_type1, agg_sq_type2, agg_sq_type3 = aggregate_simulation_by_type_k3(
            sim_choices_sq, sim_addiction_sq, sim_aflav_sq, sim_welfare_sq_arr, cat_idx, N_J, T_sim, hh_posterior_type1, hh_posterior_type2, hh_posterior_type3, N_draws
        )
        agg_ban_type1, agg_ban_type2, agg_ban_type3 = aggregate_simulation_by_type_k3(
            sim_choices_ban, sim_addiction_ban, sim_aflav_ban, sim_welfare_ban_arr, cat_idx, N_J, T_sim, hh_posterior_type1, hh_posterior_type2, hh_posterior_type3, N_draws
        )

        # --- Save Overall Simulation Results ---
        sim_results_path = joinpath(beta_subdir, "Simulation_Overall.csv");
        open(sim_results_path, "w") do io

            # Header
            header = ["period",
                "sq_outside", "sq_cig", "sq_orig_ecig", "sq_non_fda_flav_ecig", "sq_fda_flav_ecig",
                "sq_orig_bundle", "sq_non_fda_flav_bundle", "sq_fda_flav_bundle",
                "sq_addiction", "sq_aflav", "sq_welfare",
                "ban_outside", "ban_cig", "ban_orig_ecig", "ban_non_fda_flav_ecig", "ban_fda_flav_ecig",
                "ban_orig_bundle", "ban_non_fda_flav_bundle", "ban_fda_flav_bundle",
                "ban_addiction", "ban_aflav", "ban_welfare"]
            println(io, join(header, ","))

            # Data rows
            for t in 1:T_sim
                row = [@sprintf("%d", t),
                    @sprintf("%.10f", agg_sq.share_outside[t]),
                    @sprintf("%.10f", agg_sq.share_cig[t]),
                    @sprintf("%.10f", agg_sq.share_orig_ecig[t]),
                    @sprintf("%.10f", agg_sq.share_non_fda_flav_ecig[t]),
                    @sprintf("%.10f", agg_sq.share_fda_flav_ecig[t]),
                    @sprintf("%.10f", agg_sq.share_orig_bundle[t]),
                    @sprintf("%.10f", agg_sq.share_non_fda_flav_bundle[t]),
                    @sprintf("%.10f", agg_sq.share_fda_flav_bundle[t]),
                    @sprintf("%.10f", agg_sq.mean_addiction[t]),
                    @sprintf("%.10f", agg_sq.mean_aflav[t]),
                    @sprintf("%.10f", agg_sq.mean_welfare[t]),
                    @sprintf("%.10f", agg_ban.share_outside[t]),
                    @sprintf("%.10f", agg_ban.share_cig[t]),
                    @sprintf("%.10f", agg_ban.share_orig_ecig[t]),
                    @sprintf("%.10f", agg_ban.share_non_fda_flav_ecig[t]),
                    @sprintf("%.10f", agg_ban.share_fda_flav_ecig[t]),
                    @sprintf("%.10f", agg_ban.share_orig_bundle[t]),
                    @sprintf("%.10f", agg_ban.share_non_fda_flav_bundle[t]),
                    @sprintf("%.10f", agg_ban.share_fda_flav_bundle[t]),
                    @sprintf("%.10f", agg_ban.mean_addiction[t]),
                    @sprintf("%.10f", agg_ban.mean_aflav[t]),
                    @sprintf("%.10f", agg_ban.mean_welfare[t])]
                println(io, join(row, ","))
            end
        end

        log_msg("Overall simulation results saved to: $sim_results_path")

        # --- Save Simulation by TYA Status ---
        sim_tya_path = joinpath(beta_subdir, "Simulation_by_TYA.csv");
        open(sim_tya_path, "w") do io

            # Header
            header = ["group", "period",
                "sq_outside", "sq_cig", "sq_orig_ecig", "sq_non_fda_flav_ecig", "sq_fda_flav_ecig",
                "sq_orig_bundle", "sq_non_fda_flav_bundle", "sq_fda_flav_bundle",
                "sq_addiction", "sq_aflav", "sq_welfare",
                "ban_outside", "ban_cig", "ban_orig_ecig", "ban_non_fda_flav_ecig", "ban_fda_flav_ecig",
                "ban_orig_bundle", "ban_non_fda_flav_bundle", "ban_fda_flav_bundle",
                "ban_addiction", "ban_aflav", "ban_welfare"]
            println(io, join(header, ","))

            # TYA group rows
            for (group_label, agg_sq_g, agg_ban_g) in [("tya", agg_sq_tya, agg_ban_tya), ("no_tya", agg_sq_no_tya, agg_ban_no_tya)]
                for t in 1:T_sim
                    row = [group_label, @sprintf("%d", t),
                        @sprintf("%.10f", agg_sq_g.share_outside[t]),
                        @sprintf("%.10f", agg_sq_g.share_cig[t]),
                        @sprintf("%.10f", agg_sq_g.share_orig_ecig[t]),
                        @sprintf("%.10f", agg_sq_g.share_non_fda_flav_ecig[t]),
                        @sprintf("%.10f", agg_sq_g.share_fda_flav_ecig[t]),
                        @sprintf("%.10f", agg_sq_g.share_orig_bundle[t]),
                        @sprintf("%.10f", agg_sq_g.share_non_fda_flav_bundle[t]),
                        @sprintf("%.10f", agg_sq_g.share_fda_flav_bundle[t]),
                        @sprintf("%.10f", agg_sq_g.mean_addiction[t]),
                        @sprintf("%.10f", agg_sq_g.mean_aflav[t]),
                        @sprintf("%.10f", agg_sq_g.mean_welfare[t]),
                        @sprintf("%.10f", agg_ban_g.share_outside[t]),
                        @sprintf("%.10f", agg_ban_g.share_cig[t]),
                        @sprintf("%.10f", agg_ban_g.share_orig_ecig[t]),
                        @sprintf("%.10f", agg_ban_g.share_non_fda_flav_ecig[t]),
                        @sprintf("%.10f", agg_ban_g.share_fda_flav_ecig[t]),
                        @sprintf("%.10f", agg_ban_g.share_orig_bundle[t]),
                        @sprintf("%.10f", agg_ban_g.share_non_fda_flav_bundle[t]),
                        @sprintf("%.10f", agg_ban_g.share_fda_flav_bundle[t]),
                        @sprintf("%.10f", agg_ban_g.mean_addiction[t]),
                        @sprintf("%.10f", agg_ban_g.mean_aflav[t]),
                        @sprintf("%.10f", agg_ban_g.mean_welfare[t])]
                    println(io, join(row, ","))
                end
            end
        end

        log_msg("TYA simulation results saved to: $sim_tya_path")

        # --- Save Simulation by Latent Type ---
        sim_type_path = joinpath(beta_subdir, "Simulation_by_Type.csv");
        open(sim_type_path, "w") do io

            # Header
            header = ["type", "period",
                "sq_outside", "sq_cig", "sq_orig_ecig", "sq_non_fda_flav_ecig", "sq_fda_flav_ecig",
                "sq_orig_bundle", "sq_non_fda_flav_bundle", "sq_fda_flav_bundle",
                "sq_addiction", "sq_aflav", "sq_welfare",
                "ban_outside", "ban_cig", "ban_orig_ecig", "ban_non_fda_flav_ecig", "ban_fda_flav_ecig",
                "ban_orig_bundle", "ban_non_fda_flav_bundle", "ban_fda_flav_bundle",
                "ban_addiction", "ban_aflav", "ban_welfare"]
            println(io, join(header, ","))

            for (type_label, agg_sq_g, agg_ban_g) in [("type1", agg_sq_type1, agg_ban_type1), ("type2", agg_sq_type2, agg_ban_type2), ("type3", agg_sq_type3, agg_ban_type3)]
                for t in 1:T_sim
                    row = [type_label, @sprintf("%d", t),
                        @sprintf("%.10f", agg_sq_g.share_outside[t]),
                        @sprintf("%.10f", agg_sq_g.share_cig[t]),
                        @sprintf("%.10f", agg_sq_g.share_orig_ecig[t]),
                        @sprintf("%.10f", agg_sq_g.share_non_fda_flav_ecig[t]),
                        @sprintf("%.10f", agg_sq_g.share_fda_flav_ecig[t]),
                        @sprintf("%.10f", agg_sq_g.share_orig_bundle[t]),
                        @sprintf("%.10f", agg_sq_g.share_non_fda_flav_bundle[t]),
                        @sprintf("%.10f", agg_sq_g.share_fda_flav_bundle[t]),
                        @sprintf("%.10f", agg_sq_g.mean_addiction[t]),
                        @sprintf("%.10f", agg_sq_g.mean_aflav[t]),
                        @sprintf("%.10f", agg_sq_g.mean_welfare[t]),
                        @sprintf("%.10f", agg_ban_g.share_outside[t]),
                        @sprintf("%.10f", agg_ban_g.share_cig[t]),
                        @sprintf("%.10f", agg_ban_g.share_orig_ecig[t]),
                        @sprintf("%.10f", agg_ban_g.share_non_fda_flav_ecig[t]),
                        @sprintf("%.10f", agg_ban_g.share_fda_flav_ecig[t]),
                        @sprintf("%.10f", agg_ban_g.share_orig_bundle[t]),
                        @sprintf("%.10f", agg_ban_g.share_non_fda_flav_bundle[t]),
                        @sprintf("%.10f", agg_ban_g.share_fda_flav_bundle[t]),
                        @sprintf("%.10f", agg_ban_g.mean_addiction[t]),
                        @sprintf("%.10f", agg_ban_g.mean_aflav[t]),
                        @sprintf("%.10f", agg_ban_g.mean_welfare[t])]
                    println(io, join(row, ","))
                end
            end
        end

        log_msg("Type simulation results saved to: $sim_type_path")

        # --- Save Extensive Margin Results (main + threshold robustness) ---
        for (thresh, suffix) in [(0.05, "_thresh005"), (0.10, ""), (0.20, "_thresh020")]
            ext_tya_t, ext_no_tya_t = aggregate_extensive_margin_by_tya(
                sim_choices_sq, sim_choices_ban, hh_aflav0, cat_idx, T_sim, hh_tya, N_draws;
                aflav_threshold = thresh
            )
            ext_path_t = joinpath(beta_subdir, "Extensive_Margin_by_TYA$(suffix).csv")
            open(ext_path_t, "w") do io
                header = ["group", "period", "n_non_users",
                          "sq_ever_initiated", "cf_ever_initiated", "prevention_rate"]
                println(io, join(header, ","))
                for (group_label, df_g) in [("tya", ext_tya_t), ("no_tya", ext_no_tya_t)]
                    for t in 1:T_sim
                        row = [group_label,
                               @sprintf("%d",    t),
                               @sprintf("%d",    df_g.n_non_users[t]),
                               @sprintf("%.10f", df_g.sq_ever_initiated[t]),
                               @sprintf("%.10f", df_g.cf_ever_initiated[t]),
                               @sprintf("%.10f", df_g.prevention_rate[t])]
                        println(io, join(row, ","))
                    end
                end
            end
            log_msg("Extensive margin (thresh=$(thresh)) saved to: $ext_path_t")
        end


        #############################
        # Log Simulation Summary
        #############################

        log_msg("\n--- Forward Simulation Summary (averaged over $T_sim periods) ---")
        log_msg(@sprintf("  %-22s  %12s  %12s  %12s", "Category", "SQ Share", "Ban Share", "Difference"))
        log_msg("  " * repeat("-", 62))

        sim_cat_cols = [:share_outside, :share_cig, :share_orig_ecig, :share_non_fda_flav_ecig, :share_fda_flav_ecig, :share_orig_bundle, :share_non_fda_flav_bundle, :share_fda_flav_bundle]
        for (c, label) in enumerate(cat_labels)
            col = sim_cat_cols[c]
            sq_share  = mean(agg_sq[!, col])
            ban_share = mean(agg_ban[!, col])
            log_msg(@sprintf("  %-22s  %12.6f  %12.6f  %12.6f", label, sq_share, ban_share, ban_share - sq_share))
        end

        log_msg(@sprintf("\n  Mean addiction (SQ):  %.6f", mean(agg_sq.mean_addiction)))
        log_msg(@sprintf("  Mean addiction (Ban): %.6f", mean(agg_ban.mean_addiction)))
        log_msg(@sprintf("  Addiction change:     %.6f", mean(agg_ban.mean_addiction) - mean(agg_sq.mean_addiction)))
        log_msg(@sprintf("\n  Mean welfare (SQ):   %.6f", mean(agg_sq.mean_welfare)))
        log_msg(@sprintf("  Mean welfare (Ban):  %.6f", mean(agg_ban.mean_welfare)))
        log_msg(@sprintf("  Welfare change:      %.6f", mean(agg_ban.mean_welfare) - mean(agg_sq.mean_welfare)))

        # Free per-β memory
        probs_1_sq = nothing; probs_2_sq = nothing; probs_3_sq = nothing
        probs_1_ban = nothing; probs_2_ban = nothing; probs_3_ban = nothing
        welfare_1_sq = nothing; welfare_2_sq = nothing; welfare_3_sq = nothing
        welfare_1_ban = nothing; welfare_2_ban = nothing; welfare_3_ban = nothing
        GC.gc()

    end  # end BETA_GRID loop

end  # end BAN_TYPES loop


#############################
# Flavor Tax Counterfactual
#############################

# Grid of per-mL tax levels (in real dollars) on flavored e-cigarette alternatives.
TAX_GRID = [0.10, 0.25, 0.50]

log_msg("\n\n===================================")
log_msg("Starting flavor tax counterfactual")
log_msg("===================================")
log_msg("Tax levels (per mL): $TAX_GRID")
log_msg(@sprintf("q_ecig_max = %.4f", q_ecig_max))
log_msg("ω_E is β-specific; logged per β below.")

tax_subdir = joinpath(output_dir, "Flavor_Tax")
mkpath(tax_subdir)

for tau in TAX_GRID

    tau_tag = replace(@sprintf("%.2f", tau), "." => "p")  # e.g., "1p00"
    tau_subdir_name = "Tax_$(tau_tag)"

    log_msg("\n\n###################################")
    log_msg("Flavor tax: \$$(tau) per mL (tag: $tau_tag)")
    log_msg("###################################")

    for beta_val in BETA_GRID

        beta_tag = numeric_tag(beta_val)
        beta_subdir = joinpath(tax_subdir, tau_subdir_name, "Beta_$beta_tag")
        mkpath(beta_subdir)

        log_msg("\n###################################")
        log_msg("β = $beta_val (tag: $beta_tag)")
        log_msg("###################################")

        # Declare loop-local variables
        local V_decision_1_sq, vfi_iters_1_sq, vfi_converged_1_sq
        local V_decision_2_sq, vfi_iters_2_sq, vfi_converged_2_sq
        local V_decision_3_sq, vfi_iters_3_sq, vfi_converged_3_sq
        local V_decision_1_tax, vfi_iters_1_tax, vfi_converged_1_tax
        local V_decision_2_tax, vfi_iters_2_tax, vfi_converged_2_tax
        local V_decision_3_tax, vfi_iters_3_tax, vfi_converged_3_tax
        local probs_1_sq, welfare_1_sq, probs_2_sq, welfare_2_sq, probs_3_sq, welfare_3_sq
        local probs_1_tax, welfare_1_tax, probs_2_tax, welfare_2_tax, probs_3_tax, welfare_3_tax
        local hh_posterior_type1, hh_posterior_type2, hh_posterior_type3, mean_w1, mean_w2, mean_w3
        local obs_posterior_type1, obs_posterior_type2, obs_posterior_type3
        local probs_sq, welfare_sq, probs_tax, welfare_tax

        # Unpack β-specific objects pre-loaded in section 13
        local _bp = beta_params[beta_val]
        local U_1 = _bp.U_1; local U_2 = _bp.U_2; local U_3 = _bp.U_3
        local PSI_3 = _bp.PSI_3
        local N_A_flav = _bp.N_A_flav; local A_flav = _bp.A_flav
        local aflav_lower = _bp.aflav_lower; local aflav_upper = _bp.aflav_upper
        local aflav_weight = _bp.aflav_weight
        local aflav_continuous = _bp.aflav_continuous
        local hh_aflav0 = _bp.hh_aflav0
        local π_0_2 = _bp.π_0_2; local π_TYA_2 = _bp.π_TYA_2
        local π_0_3 = _bp.π_0_3; local π_TYA_3 = _bp.π_TYA_3
        local omega_E_est = _bp.omega_E_est

        # Apply tax to this β's base flow utilities
        local U_1_tax = copy(U_1); apply_flavor_tax!(U_1_tax, cat_idx, omega_E_est, q_ecig, q_ecig_max, tau)
        local U_2_tax = copy(U_2); apply_flavor_tax!(U_2_tax, cat_idx, omega_E_est, q_ecig, q_ecig_max, tau)
        local U_3_tax = copy(U_3); apply_flavor_tax!(U_3_tax, cat_idx, omega_E_est, q_ecig, q_ecig_max, tau)

        # Log the utility shift for the median flavored alternative
        local flav_alts = findall(j -> cat_idx[j] in (3, 4, 6, 7), 1:N_J)
        local median_q = median(q_ecig[flav_alts] .* q_ecig_max)
        log_msg(@sprintf("ω_E (β=%.2f) = %.10f; median flavored alt: q_ecig_raw = %.2f mL, utility shift = %.6f",
            beta_val, omega_E_est, median_q, omega_E_est * tau * median_q))

        # --- Solve VFI: Status Quo ---
        log_msg("\n===================================")
        log_msg("Solving VFI: Status Quo (β = $beta_val)")
        log_msg("===================================")

        t_vfi = time();
        task_sq_1 = Threads.@spawn solve_vfi_sophisticated(
            N_J, N_A_f, N_A_s, N_A_flav, N_P, N_Pcomb, beta_val, δ, U_1,
            af_lower, af_upper, af_weight,
            as_lower, as_upper, as_weight,
            aflav_lower, aflav_upper, aflav_weight,
            p_cig_lo, p_cig_hi, p_cig_w, p_ecig_lo, p_ecig_hi, p_ecig_w;
            V_init = nothing,
            verbose = true
        )
        task_sq_2 = Threads.@spawn solve_vfi_sophisticated(
            N_J, N_A_f, N_A_s, N_A_flav, N_P, N_Pcomb, beta_val, δ, U_2,
            af_lower, af_upper, af_weight,
            as_lower, as_upper, as_weight,
            aflav_lower, aflav_upper, aflav_weight,
            p_cig_lo, p_cig_hi, p_cig_w, p_ecig_lo, p_ecig_hi, p_ecig_w;
            V_init = nothing,
            verbose = true
        )
        task_sq_3 = Threads.@spawn solve_vfi_sophisticated(
            N_J, N_A_f, N_A_s, N_A_flav, N_P, N_Pcomb, beta_val, δ, U_3,
            af_lower, af_upper, af_weight,
            as_lower, as_upper, as_weight,
            aflav_lower, aflav_upper, aflav_weight,
            p_cig_lo, p_cig_hi, p_cig_w, p_ecig_lo, p_ecig_hi, p_ecig_w;
            V_init = nothing,
            verbose = true
        )
        _, V_decision_1_sq, vfi_iters_1_sq, vfi_converged_1_sq = fetch(task_sq_1)
        _, V_decision_2_sq, vfi_iters_2_sq, vfi_converged_2_sq = fetch(task_sq_2)
        _, V_decision_3_sq, vfi_iters_3_sq, vfi_converged_3_sq = fetch(task_sq_3)
        log_msg("Status quo VFI: type1=$(vfi_iters_1_sq) iters ($(vfi_converged_1_sq)), type2=$(vfi_iters_2_sq) iters ($(vfi_converged_2_sq)), type3=$(vfi_iters_3_sq) iters ($(vfi_converged_3_sq)), $(round(time() - t_vfi, digits=1))s")

        # --- Solve VFI: Flavor Tax ---
        log_msg("\n===================================")
        log_msg("Solving VFI: Flavor Tax (β = $beta_val)")
        log_msg("===================================")

        t_vfi = time();
        task_tax_1 = Threads.@spawn solve_vfi_sophisticated(
            N_J, N_A_f, N_A_s, N_A_flav, N_P, N_Pcomb, beta_val, δ, U_1_tax,
            af_lower, af_upper, af_weight,
            as_lower, as_upper, as_weight,
            aflav_lower, aflav_upper, aflav_weight,
            p_cig_lo, p_cig_hi, p_cig_w, p_ecig_lo, p_ecig_hi, p_ecig_w;
            V_init = nothing,
            verbose = true
        )
        task_tax_2 = Threads.@spawn solve_vfi_sophisticated(
            N_J, N_A_f, N_A_s, N_A_flav, N_P, N_Pcomb, beta_val, δ, U_2_tax,
            af_lower, af_upper, af_weight,
            as_lower, as_upper, as_weight,
            aflav_lower, aflav_upper, aflav_weight,
            p_cig_lo, p_cig_hi, p_cig_w, p_ecig_lo, p_ecig_hi, p_ecig_w;
            V_init = nothing,
            verbose = true
        )
        task_tax_3 = Threads.@spawn solve_vfi_sophisticated(
            N_J, N_A_f, N_A_s, N_A_flav, N_P, N_Pcomb, beta_val, δ, U_3_tax,
            af_lower, af_upper, af_weight,
            as_lower, as_upper, as_weight,
            aflav_lower, aflav_upper, aflav_weight,
            p_cig_lo, p_cig_hi, p_cig_w, p_ecig_lo, p_ecig_hi, p_ecig_w;
            V_init = nothing,
            verbose = true
        )
        _, V_decision_1_tax, vfi_iters_1_tax, vfi_converged_1_tax = fetch(task_tax_1)
        _, V_decision_2_tax, vfi_iters_2_tax, vfi_converged_2_tax = fetch(task_tax_2)
        _, V_decision_3_tax, vfi_iters_3_tax, vfi_converged_3_tax = fetch(task_tax_3)
        log_msg("Tax VFI: type1=$(vfi_iters_1_tax) iters ($(vfi_converged_1_tax)), type2=$(vfi_iters_2_tax) iters ($(vfi_converged_2_tax)), type3=$(vfi_iters_3_tax) iters ($(vfi_converged_3_tax)), $(round(time() - t_vfi, digits=1))s")

        # --- Compute posterior type weights ---
        log_msg("\n===================================")
        log_msg("Computing posterior type weights...")
        log_msg("===================================")

        t_post = time();
        probs_1_sq, welfare_1_sq = compute_pointwise_outcomes(
            V_decision_1_sq, tya_state, af_continuous, as_continuous, aflav_continuous, p_continuous, N_J, N_P, A_f, A_s, A_flav, P
        )
        probs_2_sq, welfare_2_sq = compute_pointwise_outcomes(
            V_decision_2_sq, tya_state, af_continuous, as_continuous, aflav_continuous, p_continuous, N_J, N_P, A_f, A_s, A_flav, P
        )
        probs_3_sq, welfare_3_sq = compute_pointwise_outcomes(
            V_decision_3_sq, tya_state, af_continuous, as_continuous, aflav_continuous, p_continuous, N_J, N_P, A_f, A_s, A_flav, P
        )
        probs_1_tax, welfare_1_tax = compute_pointwise_outcomes(
            V_decision_1_tax, tya_state, af_continuous, as_continuous, aflav_continuous, p_continuous, N_J, N_P, A_f, A_s, A_flav, P
        )
        probs_2_tax, welfare_2_tax = compute_pointwise_outcomes(
            V_decision_2_tax, tya_state, af_continuous, as_continuous, aflav_continuous, p_continuous, N_J, N_P, A_f, A_s, A_flav, P
        )
        probs_3_tax, welfare_3_tax = compute_pointwise_outcomes(
            V_decision_3_tax, tya_state, af_continuous, as_continuous, aflav_continuous, p_continuous, N_J, N_P, A_f, A_s, A_flav, P
        )

        # Compute household-level posterior type probabilities (K=3 softmax)
        hh_posterior_type1 = Vector{Float64}(undef, N_HH)
        hh_posterior_type2 = Vector{Float64}(undef, N_HH)
        hh_posterior_type3 = Vector{Float64}(undef, N_HH)

        for h in 1:N_HH
            start_idx, stop_idx = hh_ranges[h]

            # Household-specific K=3 softmax log mixing weights
            η_2_h = π_0_2 + π_TYA_2 * tya_share_hh[h]
            η_3_h = π_0_3 + π_TYA_3 * tya_share_hh[h]
            log_sum_exp_h = log(1.0 + exp(η_2_h) + exp(η_3_h))
            log_pi_1_h = -log_sum_exp_h
            log_pi_2_h = η_2_h - log_sum_exp_h
            log_pi_3_h = η_3_h - log_sum_exp_h

            # Accumulate per-type log-likelihood for this household
            log_ll_1 = 0.0
            log_ll_2 = 0.0
            log_ll_3 = 0.0
            for i in start_idx:stop_idx
                log_ll_1 += log(max(probs_1_sq[i, y[i]], 1e-300))
                log_ll_2 += log(max(probs_2_sq[i, y[i]], 1e-300))
                log_ll_3 += log(max(probs_3_sq[i, y[i]], 1e-300))
            end

            # Posterior via logsumexp normalization: P(k|y_h) = exp(log_pi_k + log_ll_k - log_denom)
            a_term = log_pi_1_h + log_ll_1
            b_term = log_pi_2_h + log_ll_2
            c_term = log_pi_3_h + log_ll_3
            log_max = max(a_term, b_term, c_term)
            log_denom = log_max + log(exp(a_term - log_max) + exp(b_term - log_max) + exp(c_term - log_max))
            hh_posterior_type1[h] = exp(a_term - log_denom)
            hh_posterior_type2[h] = exp(b_term - log_denom)
            hh_posterior_type3[h] = exp(c_term - log_denom)
        end

        mean_w1 = mean(hh_posterior_type1)
        mean_w2 = mean(hh_posterior_type2)
        mean_w3 = mean(hh_posterior_type3)
        log_msg("Posterior type weights computed in $(round(time() - t_post, digits=1))s")
        log_msg(@sprintf("  Mean P(type=1) = %.4f, Mean P(type=2) = %.4f, Mean P(type=3) = %.4f", mean_w1, mean_w2, mean_w3))

        # Map posterior to observation level
        obs_posterior_type1 = Vector{Float64}(undef, N_obs)
        obs_posterior_type2 = Vector{Float64}(undef, N_obs)
        obs_posterior_type3 = Vector{Float64}(undef, N_obs)
        for h in 1:N_HH
            start_idx, stop_idx = hh_ranges[h]
            for i in start_idx:stop_idx
                obs_posterior_type1[i] = hh_posterior_type1[h]
                obs_posterior_type2[i] = hh_posterior_type2[h]
                obs_posterior_type3[i] = hh_posterior_type3[h]
            end
        end

        # Mixture-weighted pointwise outcomes — law of total probability over the latent type:
        #   P̂(j|i) = Σ_l P̂(l|y_h) · P(j|l, x_i)
        # Integrates out type uncertainty; hard-assigning each HH to its modal type would
        # discard posterior mass on non-modal types and bias predicted in-sample market shares.
        probs_sq    = obs_posterior_type1 .* probs_1_sq   .+ obs_posterior_type2 .* probs_2_sq   .+ obs_posterior_type3 .* probs_3_sq
        welfare_sq  = obs_posterior_type1 .* welfare_1_sq .+ obs_posterior_type2 .* welfare_2_sq .+ obs_posterior_type3 .* welfare_3_sq
        probs_tax   = obs_posterior_type1 .* probs_1_tax  .+ obs_posterior_type2 .* probs_2_tax  .+ obs_posterior_type3 .* probs_3_tax
        welfare_tax = obs_posterior_type1 .* welfare_1_tax .+ obs_posterior_type2 .* welfare_2_tax .+ obs_posterior_type3 .* welfare_3_tax

        # --- Log pointwise summary ---
        t_pw = time();
        log_msg("\nPointwise summary (means across all observations):")
        log_msg(@sprintf("  %-22s  %12s  %12s  %12s", "Category", "SQ Share", "Tax Share", "Difference"))
        log_msg("  " * repeat("-", 62))

        for (c, label) in enumerate(cat_labels)
            cat_val = c - 1
            alt_indices = findall(j -> cat_idx[j] == cat_val, 1:N_J)
            sq_share  = mean(sum(probs_sq[:, alt_indices], dims=2))
            tax_share = mean(sum(probs_tax[:, alt_indices], dims=2))
            log_msg(@sprintf("  %-22s  %12.6f  %12.6f  %12.6f", label, sq_share, tax_share, tax_share - sq_share))
        end

        welfare_diff = welfare_tax .- welfare_sq
        log_msg(@sprintf("\n  Mean welfare SQ:   %.6f", mean(welfare_sq)))
        log_msg(@sprintf("  Mean welfare Tax:  %.6f", mean(welfare_tax)))
        log_msg(@sprintf("  Mean welfare loss: %.6f", mean(welfare_diff)))

        # Compute expected tax revenue per household-month
        # Revenue = τ × Σ_j [P(j) × q_ecig_raw(j)] for j in flavored categories
        flav_revenue = 0.0
        for i in 1:N_obs
            for j in flav_alts
                flav_revenue += probs_tax[i, j] * q_ecig[j] * q_ecig_max * tau
            end
        end
        mean_revenue = flav_revenue / N_obs
        log_msg(@sprintf("  Mean tax revenue per HH-month: \$%.4f", mean_revenue))

        # --- Forward Simulation ---
        log_msg("\n===================================")
        log_msg("Forward simulation (τ = \$$tau, β = $beta_val)...")
        log_msg("===================================")
        log_msg("T_sim = $T_sim, N_draws = $N_draws, N_HH = $N_HH")

        # Forward simulation overview:
        #   - For each household, assign to the latent type via posterior probability draw.
        #   - Simulate T_sim periods under both SQ and tax using the assigned type's V_decision.
        #   - Both scenarios use identical pre-drawn CRN variates to reduce variance of the
        #     SQ-vs-tax difference. The same uniform draw determines the choice in both
        #     scenarios, and the same normal draws drive AR(1) price shocks.
        #   - Each draw starts from the household's terminal addiction state (a_T), which is
        #     the addiction level after the household's final observed choice.
        #   - Prices evolve via AR(1) in both scenarios (no supply-side re-equilibration in the
        #     forward simulation; supply-side effects are captured in the pointwise analysis).
        #   - TYA state is held fixed at the household's last observed TYA state.

        sim_choices_sq      = Array{Int}(undef, N_HH, T_sim, N_draws)
        sim_addiction_f_sq  = Array{Float64}(undef, N_HH, T_sim, N_draws)
        sim_addiction_s_sq  = Array{Float64}(undef, N_HH, T_sim, N_draws)
        sim_aflav_sq        = Array{Float64}(undef, N_HH, T_sim, N_draws)
        sim_welfare_sq_arr  = Array{Float64}(undef, N_HH, T_sim, N_draws)

        sim_choices_tax     = Array{Int}(undef, N_HH, T_sim, N_draws)
        sim_addiction_f_tax = Array{Float64}(undef, N_HH, T_sim, N_draws)
        sim_addiction_s_tax = Array{Float64}(undef, N_HH, T_sim, N_draws)
        sim_aflav_tax       = Array{Float64}(undef, N_HH, T_sim, N_draws)
        sim_welfare_tax_arr = Array{Float64}(undef, N_HH, T_sim, N_draws)

        P_cig  = P[:, 1]
        P_ecig = P[:, 2]

        t_sim_fwd = time();

        Threads.@threads for h in 1:N_HH

            tya_idx_h = hh_tya[h]
            w1 = hh_posterior_type1[h]
            w2 = hh_posterior_type2[h]

            for d in 1:N_draws

                # Probabilistic type assignment: draw type from posterior using pre-drawn uniform.
                # If u <= P(type=1) → type 1; elif u <= P(type=1)+P(type=2) → type 2; else type 3.
                u_type = crn_type[h, d]
                type_k = u_type <= w1 ? 1 : (u_type <= w1 + w2 ? 2 : 3)
                V_sq  = type_k == 1 ? V_decision_1_sq  : (type_k == 2 ? V_decision_2_sq  : V_decision_3_sq)
                V_tax = type_k == 1 ? V_decision_1_tax : (type_k == 2 ? V_decision_2_tax : V_decision_3_tax)

                # Initialize both scenarios from the household's terminal observed state
                a_f_sq = hh_af0[h]; a_s_sq = hh_as0[h]; a_flav_sq = hh_aflav0[h]
                p_cig_sq = hh_p0[h, 1]; p_ecig_sq = hh_p0[h, 2]
                a_f_tax = hh_af0[h]; a_s_tax = hh_as0[h]; a_flav_tax = hh_aflav0[h]
                p_cig_tax = hh_p0[h, 1]; p_ecig_tax = hh_p0[h, 2]
                tya_sq = tya_idx_h; tya_tax = tya_idx_h

                for t in 1:T_sim

                    # --- STATUS QUO ---

                    # 5-linear interpolation of V_decision at the household's continuous state;
                    # NaN from 0.0 * (-Inf) at exact grid boundaries replaced with -Inf.
                    v_interp_sq = interpolate_v_choice(V_sq, tya_sq, a_f_sq, a_s_sq, a_flav_sq, p_cig_sq, p_ecig_sq, N_J, N_P, A_f, A_s, A_flav, P)
                    replace!(v_interp_sq, NaN => -Inf)

                    # Welfare = logsumexp(V_decision) = expected maximum utility (inclusive value)
                    sim_welfare_sq_arr[h, t, d] = logsumexp(v_interp_sq)

                    # Softmax choice probabilities: P(j) = exp(V_j - V_max) / sum_j' exp(V_j' - V_max)
                    v_max_sq = maximum(v_interp_sq)
                    exp_v_sq = exp.(v_interp_sq .- v_max_sq)
                    probs_h_sq = exp_v_sq ./ sum(exp_v_sq)

                    # Inverse CDF sampling using pre-drawn uniform (same draw reused for tax: CRN)
                    u_draw = crn_choice[h, t, d]
                    j_sq = 1; cum_prob = probs_h_sq[1]
                    while cum_prob < u_draw && j_sq < N_J; j_sq += 1; cum_prob += probs_h_sq[j_sq]; end
                    sim_choices_sq[h, t, d] = j_sq

                    # Update fast addiction: a_f' = (1 - ψ_2) * a_f + ψ_2 * n_std[j_sq]
                    # Update slow addiction: a_s' = (1 - ψ_1) * a_s + ψ_1 * n_std[j_sq]
                    # Update flavored habit: a_flav' = (1 - ψ_3) * a_flav + ψ_3 * 1[flavored[j_sq]]
                    a_f_sq = clamp(addiction_evolution(ψ_2, a_f_sq, n[j_sq]), A_f[1], A_f[end])
                    a_s_sq = clamp(addiction_evolution(ψ_1, a_s_sq, n[j_sq]), A_s[1], A_s[end])
                    a_flav_sq = clamp(addiction_evolution(PSI_3, a_flav_sq, n_flav[j_sq]), A_flav[1], A_flav[end])
                    sim_addiction_f_sq[h, t, d] = a_f_sq
                    sim_addiction_s_sq[h, t, d] = a_s_sq
                    sim_aflav_sq[h, t, d] = a_flav_sq

                    # Update prices via AR(1) with Cholesky-correlated shocks (same draws as tax: CRN)
                    ε_sq = L_chol * [crn_price[h, t, d, 1], crn_price[h, t, d, 2]]
                    p_cig_sq  = clamp(φ_0[1] + φ_1[1] * p_cig_sq  + ε_sq[1], P_cig[1], P_cig[end])
                    p_ecig_sq = clamp(φ_0[2] + φ_1[2] * p_ecig_sq + ε_sq[2], P_ecig[1], P_ecig[end])

                    # TYA state held fixed (binary, no transitions)

                    # --- FLAVOR TAX ---
                    # Same steps as SQ but using V_decision from the taxed flow utility. The tax
                    # shifts flavored alternatives' utility downward, raising the relative
                    # attractiveness of untaxed substitutes.

                    # 5-linear interpolation and NaN fix (same as SQ above)
                    v_interp_tax = interpolate_v_choice(V_tax, tya_tax, a_f_tax, a_s_tax, a_flav_tax, p_cig_tax, p_ecig_tax, N_J, N_P, A_f, A_s, A_flav, P)
                    replace!(v_interp_tax, NaN => -Inf)

                    # Welfare and softmax choice probabilities
                    sim_welfare_tax_arr[h, t, d] = logsumexp(v_interp_tax)
                    v_max_tax = maximum(v_interp_tax)
                    exp_v_tax = exp.(v_interp_tax .- v_max_tax)
                    probs_h_tax = exp_v_tax ./ sum(exp_v_tax)

                    # Draw choice using same uniform as SQ (CRN)
                    j_tax = 1; cum_prob_tax = probs_h_tax[1]
                    while cum_prob_tax < u_draw && j_tax < N_J; j_tax += 1; cum_prob_tax += probs_h_tax[j_tax]; end
                    sim_choices_tax[h, t, d] = j_tax

                    # Update fast addiction, slow addiction, and flavored habit stock
                    a_f_tax = clamp(addiction_evolution(ψ_2, a_f_tax, n[j_tax]), A_f[1], A_f[end])
                    a_s_tax = clamp(addiction_evolution(ψ_1, a_s_tax, n[j_tax]), A_s[1], A_s[end])
                    a_flav_tax = clamp(addiction_evolution(PSI_3, a_flav_tax, n_flav[j_tax]), A_flav[1], A_flav[end])
                    sim_addiction_f_tax[h, t, d] = a_f_tax
                    sim_addiction_s_tax[h, t, d] = a_s_tax
                    sim_aflav_tax[h, t, d] = a_flav_tax

                    # Update prices via AR(1) with same CRN draws as SQ
                    ε_tax = L_chol * [crn_price[h, t, d, 1], crn_price[h, t, d, 2]]
                    p_cig_tax  = clamp(φ_0[1] + φ_1[1] * p_cig_tax  + ε_tax[1], P_cig[1], P_cig[end])
                    p_ecig_tax = clamp(φ_0[2] + φ_1[2] * p_ecig_tax + ε_tax[2], P_ecig[1], P_ecig[end])

                    # TYA state held fixed (binary, no transitions)
                end
            end
        end

        sim_fwd_elapsed = time() - t_sim_fwd;
        log_msg("Forward simulation complete in $(round(sim_fwd_elapsed, digits=1))s")

        # --- Aggregate and Save ---
        log_msg("\n===================================")
        log_msg("Aggregating results...")
        log_msg("===================================")

        sim_addiction_sq  = (sim_addiction_f_sq  .+ sim_addiction_s_sq)  ./ 2.0
        sim_addiction_tax = (sim_addiction_f_tax .+ sim_addiction_s_tax) ./ 2.0

        agg_sq  = aggregate_simulation(sim_choices_sq,  sim_addiction_sq,  sim_aflav_sq,  sim_welfare_sq_arr,  cat_idx, N_J, T_sim)
        agg_tax = aggregate_simulation(sim_choices_tax, sim_addiction_tax, sim_aflav_tax, sim_welfare_tax_arr, cat_idx, N_J, T_sim)

        agg_sq_tya, agg_sq_no_tya = aggregate_simulation_by_tya(
            sim_choices_sq, sim_addiction_sq, sim_aflav_sq, sim_welfare_sq_arr, cat_idx, N_J, T_sim, hh_tya, N_draws)
        agg_tax_tya, agg_tax_no_tya = aggregate_simulation_by_tya(
            sim_choices_tax, sim_addiction_tax, sim_aflav_tax, sim_welfare_tax_arr, cat_idx, N_J, T_sim, hh_tya, N_draws)

        agg_sq_type1, agg_sq_type2, agg_sq_type3 = aggregate_simulation_by_type_k3(
            sim_choices_sq,  sim_addiction_sq,  sim_aflav_sq,  sim_welfare_sq_arr,  cat_idx, N_J, T_sim, hh_posterior_type1, hh_posterior_type2, hh_posterior_type3, N_draws)
        agg_tax_type1, agg_tax_type2, agg_tax_type3 = aggregate_simulation_by_type_k3(
            sim_choices_tax, sim_addiction_tax, sim_aflav_tax, sim_welfare_tax_arr, cat_idx, N_J, T_sim, hh_posterior_type1, hh_posterior_type2, hh_posterior_type3, N_draws)

        # Save overall simulation results
        sim_results_path = joinpath(beta_subdir, "Simulation_Overall.csv")
        open(sim_results_path, "w") do io
            header = ["period",
                "sq_outside", "sq_cig", "sq_orig_ecig", "sq_non_fda_flav_ecig", "sq_fda_flav_ecig",
                "sq_orig_bundle", "sq_non_fda_flav_bundle", "sq_fda_flav_bundle",
                "sq_addiction", "sq_aflav", "sq_welfare",
                "tax_outside", "tax_cig", "tax_orig_ecig", "tax_non_fda_flav_ecig", "tax_fda_flav_ecig",
                "tax_orig_bundle", "tax_non_fda_flav_bundle", "tax_fda_flav_bundle",
                "tax_addiction", "tax_aflav", "tax_welfare"]
            println(io, join(header, ","))
            for t in 1:T_sim
                row = [@sprintf("%d", t),
                    @sprintf("%.10f", agg_sq.share_outside[t]), @sprintf("%.10f", agg_sq.share_cig[t]),
                    @sprintf("%.10f", agg_sq.share_orig_ecig[t]), @sprintf("%.10f", agg_sq.share_non_fda_flav_ecig[t]),
                    @sprintf("%.10f", agg_sq.share_fda_flav_ecig[t]), @sprintf("%.10f", agg_sq.share_orig_bundle[t]),
                    @sprintf("%.10f", agg_sq.share_non_fda_flav_bundle[t]), @sprintf("%.10f", agg_sq.share_fda_flav_bundle[t]),
                    @sprintf("%.10f", agg_sq.mean_addiction[t]), @sprintf("%.10f", agg_sq.mean_aflav[t]),
                    @sprintf("%.10f", agg_sq.mean_welfare[t]),
                    @sprintf("%.10f", agg_tax.share_outside[t]), @sprintf("%.10f", agg_tax.share_cig[t]),
                    @sprintf("%.10f", agg_tax.share_orig_ecig[t]), @sprintf("%.10f", agg_tax.share_non_fda_flav_ecig[t]),
                    @sprintf("%.10f", agg_tax.share_fda_flav_ecig[t]), @sprintf("%.10f", agg_tax.share_orig_bundle[t]),
                    @sprintf("%.10f", agg_tax.share_non_fda_flav_bundle[t]), @sprintf("%.10f", agg_tax.share_fda_flav_bundle[t]),
                    @sprintf("%.10f", agg_tax.mean_addiction[t]), @sprintf("%.10f", agg_tax.mean_aflav[t]),
                    @sprintf("%.10f", agg_tax.mean_welfare[t])]
                println(io, join(row, ","))
            end
        end
        log_msg("Overall simulation results saved to: $sim_results_path")

        # Save simulation by TYA
        sim_tya_path = joinpath(beta_subdir, "Simulation_by_TYA.csv")
        open(sim_tya_path, "w") do io
            header = ["group", "period",
                "sq_outside", "sq_cig", "sq_orig_ecig", "sq_non_fda_flav_ecig", "sq_fda_flav_ecig",
                "sq_orig_bundle", "sq_non_fda_flav_bundle", "sq_fda_flav_bundle",
                "sq_addiction", "sq_aflav", "sq_welfare",
                "tax_outside", "tax_cig", "tax_orig_ecig", "tax_non_fda_flav_ecig", "tax_fda_flav_ecig",
                "tax_orig_bundle", "tax_non_fda_flav_bundle", "tax_fda_flav_bundle",
                "tax_addiction", "tax_aflav", "tax_welfare"]
            println(io, join(header, ","))
            for (group_label, agg_sq_g, agg_tax_g) in [("tya", agg_sq_tya, agg_tax_tya), ("no_tya", agg_sq_no_tya, agg_tax_no_tya)]
                for t in 1:T_sim
                    row = [group_label, @sprintf("%d", t),
                        @sprintf("%.10f", agg_sq_g.share_outside[t]), @sprintf("%.10f", agg_sq_g.share_cig[t]),
                        @sprintf("%.10f", agg_sq_g.share_orig_ecig[t]), @sprintf("%.10f", agg_sq_g.share_non_fda_flav_ecig[t]),
                        @sprintf("%.10f", agg_sq_g.share_fda_flav_ecig[t]), @sprintf("%.10f", agg_sq_g.share_orig_bundle[t]),
                        @sprintf("%.10f", agg_sq_g.share_non_fda_flav_bundle[t]), @sprintf("%.10f", agg_sq_g.share_fda_flav_bundle[t]),
                        @sprintf("%.10f", agg_sq_g.mean_addiction[t]), @sprintf("%.10f", agg_sq_g.mean_aflav[t]),
                        @sprintf("%.10f", agg_sq_g.mean_welfare[t]),
                        @sprintf("%.10f", agg_tax_g.share_outside[t]), @sprintf("%.10f", agg_tax_g.share_cig[t]),
                        @sprintf("%.10f", agg_tax_g.share_orig_ecig[t]), @sprintf("%.10f", agg_tax_g.share_non_fda_flav_ecig[t]),
                        @sprintf("%.10f", agg_tax_g.share_fda_flav_ecig[t]), @sprintf("%.10f", agg_tax_g.share_orig_bundle[t]),
                        @sprintf("%.10f", agg_tax_g.share_non_fda_flav_bundle[t]), @sprintf("%.10f", agg_tax_g.share_fda_flav_bundle[t]),
                        @sprintf("%.10f", agg_tax_g.mean_addiction[t]), @sprintf("%.10f", agg_tax_g.mean_aflav[t]),
                        @sprintf("%.10f", agg_tax_g.mean_welfare[t])]
                    println(io, join(row, ","))
                end
            end
        end
        log_msg("TYA simulation results saved to: $sim_tya_path")

        # Save simulation by latent type
        sim_type_path = joinpath(beta_subdir, "Simulation_by_Type.csv")
        open(sim_type_path, "w") do io
            header = ["type", "period",
                "sq_outside", "sq_cig", "sq_orig_ecig", "sq_non_fda_flav_ecig", "sq_fda_flav_ecig",
                "sq_orig_bundle", "sq_non_fda_flav_bundle", "sq_fda_flav_bundle",
                "sq_addiction", "sq_aflav", "sq_welfare",
                "tax_outside", "tax_cig", "tax_orig_ecig", "tax_non_fda_flav_ecig", "tax_fda_flav_ecig",
                "tax_orig_bundle", "tax_non_fda_flav_bundle", "tax_fda_flav_bundle",
                "tax_addiction", "tax_aflav", "tax_welfare"]
            println(io, join(header, ","))
            for (type_label, agg_sq_g, agg_tax_g) in [("type1", agg_sq_type1, agg_tax_type1), ("type2", agg_sq_type2, agg_tax_type2), ("type3", agg_sq_type3, agg_tax_type3)]
                for t in 1:T_sim
                    row = [type_label, @sprintf("%d", t),
                        @sprintf("%.10f", agg_sq_g.share_outside[t]), @sprintf("%.10f", agg_sq_g.share_cig[t]),
                        @sprintf("%.10f", agg_sq_g.share_orig_ecig[t]), @sprintf("%.10f", agg_sq_g.share_non_fda_flav_ecig[t]),
                        @sprintf("%.10f", agg_sq_g.share_fda_flav_ecig[t]), @sprintf("%.10f", agg_sq_g.share_orig_bundle[t]),
                        @sprintf("%.10f", agg_sq_g.share_non_fda_flav_bundle[t]), @sprintf("%.10f", agg_sq_g.share_fda_flav_bundle[t]),
                        @sprintf("%.10f", agg_sq_g.mean_addiction[t]), @sprintf("%.10f", agg_sq_g.mean_aflav[t]),
                        @sprintf("%.10f", agg_sq_g.mean_welfare[t]),
                        @sprintf("%.10f", agg_tax_g.share_outside[t]), @sprintf("%.10f", agg_tax_g.share_cig[t]),
                        @sprintf("%.10f", agg_tax_g.share_orig_ecig[t]), @sprintf("%.10f", agg_tax_g.share_non_fda_flav_ecig[t]),
                        @sprintf("%.10f", agg_tax_g.share_fda_flav_ecig[t]), @sprintf("%.10f", agg_tax_g.share_orig_bundle[t]),
                        @sprintf("%.10f", agg_tax_g.share_non_fda_flav_bundle[t]), @sprintf("%.10f", agg_tax_g.share_fda_flav_bundle[t]),
                        @sprintf("%.10f", agg_tax_g.mean_addiction[t]), @sprintf("%.10f", agg_tax_g.mean_aflav[t]),
                        @sprintf("%.10f", agg_tax_g.mean_welfare[t])]
                    println(io, join(row, ","))
                end
            end
        end
        log_msg("Type simulation results saved to: $sim_type_path")

        # --- Save Extensive Margin Results (main + threshold robustness) ---
        for (thresh, suffix) in [(0.05, "_thresh005"), (0.10, ""), (0.20, "_thresh020")]
            ext_tya_t, ext_no_tya_t = aggregate_extensive_margin_by_tya(
                sim_choices_sq, sim_choices_tax, hh_aflav0, cat_idx, T_sim, hh_tya, N_draws;
                aflav_threshold = thresh
            )
            ext_path_t = joinpath(beta_subdir, "Extensive_Margin_by_TYA$(suffix).csv")
            open(ext_path_t, "w") do io
                header = ["group", "period", "n_non_users",
                          "sq_ever_initiated", "cf_ever_initiated", "prevention_rate"]
                println(io, join(header, ","))
                for (group_label, df_g) in [("tya", ext_tya_t), ("no_tya", ext_no_tya_t)]
                    for t in 1:T_sim
                        row = [group_label,
                               @sprintf("%d",    t),
                               @sprintf("%d",    df_g.n_non_users[t]),
                               @sprintf("%.10f", df_g.sq_ever_initiated[t]),
                               @sprintf("%.10f", df_g.cf_ever_initiated[t]),
                               @sprintf("%.10f", df_g.prevention_rate[t])]
                        println(io, join(row, ","))
                    end
                end
            end
            log_msg("Extensive margin (thresh=$(thresh)) saved to: $ext_path_t")
        end

        # Log simulation summary
        log_msg("\n--- Forward Simulation Summary (averaged over $T_sim periods) ---")
        log_msg(@sprintf("  Mean addiction (SQ):  %.6f", mean(agg_sq.mean_addiction)))
        log_msg(@sprintf("  Mean addiction (Tax): %.6f", mean(agg_tax.mean_addiction)))
        log_msg(@sprintf("  Addiction change:     %.6f", mean(agg_tax.mean_addiction) - mean(agg_sq.mean_addiction)))
        log_msg(@sprintf("  Mean welfare (SQ):   %.6f", mean(agg_sq.mean_welfare)))
        log_msg(@sprintf("  Mean welfare (Tax):  %.6f", mean(agg_tax.mean_welfare)))
        log_msg(@sprintf("  Welfare change:      %.6f", mean(agg_tax.mean_welfare) - mean(agg_sq.mean_welfare)))

        # Free memory
        GC.gc()

    end  # end BETA_GRID loop
end  # end TAX_GRID loop


#############################
# FDA-Only Flavor Tax
# (τ applied to cats 4, 7 only;
#  unauthorized cats 3, 6 untaxed)
#############################

log_msg("\n\n===================================")
log_msg("FDA-Only Flavor Tax Counterfactual")
log_msg("===================================")

fda_tax_subdir = joinpath(output_dir, "Flavor_Tax_FDA_Only")
mkpath(fda_tax_subdir)

for tau in TAX_GRID

    tau_tag = replace(@sprintf("%.2f", tau), "." => "p")
    tau_subdir_name = "Tax_$(tau_tag)"

    log_msg("\n\n###################################")
    log_msg("FDA-Only Flavor Tax: \$$(tau) per mL (tag: $tau_tag)")
    log_msg("###################################")

    for beta_val in BETA_GRID

        beta_tag = numeric_tag(beta_val)
        beta_subdir = joinpath(fda_tax_subdir, tau_subdir_name, "Beta_$beta_tag")
        mkpath(beta_subdir)

        log_msg("\n###################################")
        log_msg("β = $beta_val (tag: $beta_tag)")
        log_msg("###################################")

        # Declare loop-local variables
        local V_decision_1_sq, vfi_iters_1_sq, vfi_converged_1_sq
        local V_decision_2_sq, vfi_iters_2_sq, vfi_converged_2_sq
        local V_decision_3_sq, vfi_iters_3_sq, vfi_converged_3_sq
        local V_decision_1_fda_tax, vfi_iters_1_fda_tax, vfi_converged_1_fda_tax
        local V_decision_2_fda_tax, vfi_iters_2_fda_tax, vfi_converged_2_fda_tax
        local V_decision_3_fda_tax, vfi_iters_3_fda_tax, vfi_converged_3_fda_tax
        local probs_1_sq, welfare_1_sq, probs_2_sq, welfare_2_sq, probs_3_sq, welfare_3_sq
        local probs_1_fda_tax, welfare_1_fda_tax, probs_2_fda_tax, welfare_2_fda_tax, probs_3_fda_tax, welfare_3_fda_tax
        local hh_posterior_type1, hh_posterior_type2, hh_posterior_type3, mean_w1
        local obs_posterior_type1, obs_posterior_type2, obs_posterior_type3
        local agg_sq_type1, agg_sq_type2, agg_sq_type3
        local agg_fda_tax_type1, agg_fda_tax_type2, agg_fda_tax_type3
        local probs_sq, welfare_sq, probs_fda_tax, welfare_fda_tax

        # Unpack β-specific objects pre-loaded in section 13
        local _bp = beta_params[beta_val]
        local U_1 = _bp.U_1; local U_2 = _bp.U_2; local U_3 = _bp.U_3
        local PSI_3 = _bp.PSI_3
        local N_A_flav = _bp.N_A_flav; local A_flav = _bp.A_flav
        local aflav_lower = _bp.aflav_lower; local aflav_upper = _bp.aflav_upper
        local aflav_weight = _bp.aflav_weight
        local aflav_continuous = _bp.aflav_continuous
        local hh_aflav0 = _bp.hh_aflav0
        local π_0_2 = _bp.π_0_2; local π_TYA_2 = _bp.π_TYA_2
        local π_0_3 = _bp.π_0_3; local π_TYA_3 = _bp.π_TYA_3
        local omega_E_est = _bp.omega_E_est

        # Apply FDA-only tax (cats 4, 7) to this β's base flow utilities
        local U_1_fda_tax = copy(U_1); apply_fda_flavor_tax!(U_1_fda_tax, cat_idx, omega_E_est, q_ecig, q_ecig_max, tau)
        local U_2_fda_tax = copy(U_2); apply_fda_flavor_tax!(U_2_fda_tax, cat_idx, omega_E_est, q_ecig, q_ecig_max, tau)
        local U_3_fda_tax = copy(U_3); apply_fda_flavor_tax!(U_3_fda_tax, cat_idx, omega_E_est, q_ecig, q_ecig_max, tau)

        # Log utility shift for median FDA-authorized flavored alternative
        local fda_flav_alts = findall(j -> cat_idx[j] in (4, 7), 1:N_J)
        local median_q_fda = median(q_ecig[fda_flav_alts] .* q_ecig_max)
        log_msg(@sprintf("ω_E (β=%.2f) = %.10f; median FDA-auth alt: q_ecig_raw = %.2f mL, utility shift = %.6f",
            beta_val, omega_E_est, median_q_fda, omega_E_est * tau * median_q_fda))

        # --- Solve VFI: Status Quo ---
        log_msg("\n===================================")
        log_msg("Solving VFI: Status Quo (β = $beta_val)")
        log_msg("===================================")

        t_vfi = time();
        task_sq_1 = Threads.@spawn solve_vfi_sophisticated(
            N_J, N_A_f, N_A_s, N_A_flav, N_P, N_Pcomb, beta_val, δ, U_1,
            af_lower, af_upper, af_weight,
            as_lower, as_upper, as_weight,
            aflav_lower, aflav_upper, aflav_weight,
            p_cig_lo, p_cig_hi, p_cig_w, p_ecig_lo, p_ecig_hi, p_ecig_w;
            V_init = nothing,
            verbose = true
        )
        task_sq_2 = Threads.@spawn solve_vfi_sophisticated(
            N_J, N_A_f, N_A_s, N_A_flav, N_P, N_Pcomb, beta_val, δ, U_2,
            af_lower, af_upper, af_weight,
            as_lower, as_upper, as_weight,
            aflav_lower, aflav_upper, aflav_weight,
            p_cig_lo, p_cig_hi, p_cig_w, p_ecig_lo, p_ecig_hi, p_ecig_w;
            V_init = nothing,
            verbose = true
        )
        task_sq_3 = Threads.@spawn solve_vfi_sophisticated(
            N_J, N_A_f, N_A_s, N_A_flav, N_P, N_Pcomb, beta_val, δ, U_3,
            af_lower, af_upper, af_weight,
            as_lower, as_upper, as_weight,
            aflav_lower, aflav_upper, aflav_weight,
            p_cig_lo, p_cig_hi, p_cig_w, p_ecig_lo, p_ecig_hi, p_ecig_w;
            V_init = nothing,
            verbose = true
        )
        _, V_decision_1_sq, vfi_iters_1_sq, vfi_converged_1_sq = fetch(task_sq_1)
        _, V_decision_2_sq, vfi_iters_2_sq, vfi_converged_2_sq = fetch(task_sq_2)
        _, V_decision_3_sq, vfi_iters_3_sq, vfi_converged_3_sq = fetch(task_sq_3)
        log_msg("Status quo VFI: type1=$(vfi_iters_1_sq) iters ($(vfi_converged_1_sq)), type2=$(vfi_iters_2_sq) iters ($(vfi_converged_2_sq)), type3=$(vfi_iters_3_sq) iters ($(vfi_converged_3_sq)), $(round(time() - t_vfi, digits=1))s")

        # --- Solve VFI: FDA-Only Flavor Tax ---
        log_msg("\n===================================")
        log_msg("Solving VFI: FDA-Only Flavor Tax (β = $beta_val)")
        log_msg("===================================")

        t_vfi = time();
        task_fda_tax_1 = Threads.@spawn solve_vfi_sophisticated(
            N_J, N_A_f, N_A_s, N_A_flav, N_P, N_Pcomb, beta_val, δ, U_1_fda_tax,
            af_lower, af_upper, af_weight,
            as_lower, as_upper, as_weight,
            aflav_lower, aflav_upper, aflav_weight,
            p_cig_lo, p_cig_hi, p_cig_w, p_ecig_lo, p_ecig_hi, p_ecig_w;
            V_init = nothing,
            verbose = true
        )
        task_fda_tax_2 = Threads.@spawn solve_vfi_sophisticated(
            N_J, N_A_f, N_A_s, N_A_flav, N_P, N_Pcomb, beta_val, δ, U_2_fda_tax,
            af_lower, af_upper, af_weight,
            as_lower, as_upper, as_weight,
            aflav_lower, aflav_upper, aflav_weight,
            p_cig_lo, p_cig_hi, p_cig_w, p_ecig_lo, p_ecig_hi, p_ecig_w;
            V_init = nothing,
            verbose = true
        )
        task_fda_tax_3 = Threads.@spawn solve_vfi_sophisticated(
            N_J, N_A_f, N_A_s, N_A_flav, N_P, N_Pcomb, beta_val, δ, U_3_fda_tax,
            af_lower, af_upper, af_weight,
            as_lower, as_upper, as_weight,
            aflav_lower, aflav_upper, aflav_weight,
            p_cig_lo, p_cig_hi, p_cig_w, p_ecig_lo, p_ecig_hi, p_ecig_w;
            V_init = nothing,
            verbose = true
        )
        _, V_decision_1_fda_tax, vfi_iters_1_fda_tax, vfi_converged_1_fda_tax = fetch(task_fda_tax_1)
        _, V_decision_2_fda_tax, vfi_iters_2_fda_tax, vfi_converged_2_fda_tax = fetch(task_fda_tax_2)
        _, V_decision_3_fda_tax, vfi_iters_3_fda_tax, vfi_converged_3_fda_tax = fetch(task_fda_tax_3)
        log_msg("FDA-Only Tax VFI: type1=$(vfi_iters_1_fda_tax) iters ($(vfi_converged_1_fda_tax)), type2=$(vfi_iters_2_fda_tax) iters ($(vfi_converged_2_fda_tax)), type3=$(vfi_iters_3_fda_tax) iters ($(vfi_converged_3_fda_tax)), $(round(time() - t_vfi, digits=1))s")

        # --- Compute posterior type weights ---
        log_msg("\n===================================")
        log_msg("Computing posterior type weights...")
        log_msg("===================================")

        t_post = time();
        probs_1_sq, welfare_1_sq = compute_pointwise_outcomes(
            V_decision_1_sq, tya_state, af_continuous, as_continuous, aflav_continuous, p_continuous, N_J, N_P, A_f, A_s, A_flav, P
        )
        probs_2_sq, welfare_2_sq = compute_pointwise_outcomes(
            V_decision_2_sq, tya_state, af_continuous, as_continuous, aflav_continuous, p_continuous, N_J, N_P, A_f, A_s, A_flav, P
        )
        probs_3_sq, welfare_3_sq = compute_pointwise_outcomes(
            V_decision_3_sq, tya_state, af_continuous, as_continuous, aflav_continuous, p_continuous, N_J, N_P, A_f, A_s, A_flav, P
        )
        probs_1_fda_tax, welfare_1_fda_tax = compute_pointwise_outcomes(
            V_decision_1_fda_tax, tya_state, af_continuous, as_continuous, aflav_continuous, p_continuous, N_J, N_P, A_f, A_s, A_flav, P
        )
        probs_2_fda_tax, welfare_2_fda_tax = compute_pointwise_outcomes(
            V_decision_2_fda_tax, tya_state, af_continuous, as_continuous, aflav_continuous, p_continuous, N_J, N_P, A_f, A_s, A_flav, P
        )
        probs_3_fda_tax, welfare_3_fda_tax = compute_pointwise_outcomes(
            V_decision_3_fda_tax, tya_state, af_continuous, as_continuous, aflav_continuous, p_continuous, N_J, N_P, A_f, A_s, A_flav, P
        )

        # Compute household-level posterior type probabilities (K=3 softmax)
        hh_posterior_type1 = Vector{Float64}(undef, N_HH)
        hh_posterior_type2 = Vector{Float64}(undef, N_HH)
        hh_posterior_type3 = Vector{Float64}(undef, N_HH)

        for h in 1:N_HH
            start_idx, stop_idx = hh_ranges[h]

            # Household-specific K=3 softmax log mixing weights
            η_2_h = π_0_2 + π_TYA_2 * tya_share_hh[h]
            η_3_h = π_0_3 + π_TYA_3 * tya_share_hh[h]
            log_sum_exp_h = log(1.0 + exp(η_2_h) + exp(η_3_h))
            log_π_1_h = -log_sum_exp_h
            log_π_2_h = η_2_h - log_sum_exp_h
            log_π_3_h = η_3_h - log_sum_exp_h

            # Accumulate per-type log-likelihood for this household
            ll_1 = 0.0; ll_2 = 0.0; ll_3 = 0.0
            for i in start_idx:stop_idx
                ll_1 += log(max(probs_1_sq[i, y[i]], 1e-300))
                ll_2 += log(max(probs_2_sq[i, y[i]], 1e-300))
                ll_3 += log(max(probs_3_sq[i, y[i]], 1e-300))
            end

            # Posterior via logsumexp normalization: P(k|y_h) = exp(log_pi_k + log_ll_k - log_denom)
            a_term = log_π_1_h + ll_1
            b_term = log_π_2_h + ll_2
            c_term = log_π_3_h + ll_3
            log_max = max(a_term, b_term, c_term)
            log_denom = log_max + log(exp(a_term - log_max) + exp(b_term - log_max) + exp(c_term - log_max))
            hh_posterior_type1[h] = exp(a_term - log_denom)
            hh_posterior_type2[h] = exp(b_term - log_denom)
            hh_posterior_type3[h] = exp(c_term - log_denom)
        end

        mean_w1 = mean(hh_posterior_type1)
        log_msg("Posterior type weights computed in $(round(time() - t_post, digits=1))s")
        log_msg(@sprintf("  Mean P(type=1) = %.4f, Mean P(type=2) = %.4f, Mean P(type=3) = %.4f",
            mean_w1, mean(hh_posterior_type2), mean(hh_posterior_type3)))

        # Map posterior to observation level
        obs_posterior_type1 = Vector{Float64}(undef, N_obs)
        obs_posterior_type2 = Vector{Float64}(undef, N_obs)
        obs_posterior_type3 = Vector{Float64}(undef, N_obs)
        for h in 1:N_HH
            start_idx, stop_idx = hh_ranges[h]
            for i in start_idx:stop_idx
                obs_posterior_type1[i] = hh_posterior_type1[h]
                obs_posterior_type2[i] = hh_posterior_type2[h]
                obs_posterior_type3[i] = hh_posterior_type3[h]
            end
        end

        # Mixture-weighted pointwise outcomes — law of total probability over the latent type:
        #   P̂(j|i) = Σ_l P̂(l|y_h) · P(j|l, x_i)
        # Integrates out type uncertainty; hard-assigning each HH to its modal type would
        # discard posterior mass on non-modal types and bias predicted in-sample market shares.
        probs_sq      = obs_posterior_type1 .* probs_1_sq      .+ obs_posterior_type2 .* probs_2_sq      .+ obs_posterior_type3 .* probs_3_sq
        welfare_sq    = obs_posterior_type1 .* welfare_1_sq    .+ obs_posterior_type2 .* welfare_2_sq    .+ obs_posterior_type3 .* welfare_3_sq
        probs_fda_tax = obs_posterior_type1 .* probs_1_fda_tax .+ obs_posterior_type2 .* probs_2_fda_tax .+ obs_posterior_type3 .* probs_3_fda_tax
        welfare_fda_tax = obs_posterior_type1 .* welfare_1_fda_tax .+ obs_posterior_type2 .* welfare_2_fda_tax .+ obs_posterior_type3 .* welfare_3_fda_tax

        # --- Log pointwise summary ---
        t_pw = time();
        log_msg("\nPointwise summary (means across all observations):")
        log_msg(@sprintf("  %-22s  %12s  %12s  %12s", "Category", "SQ Share", "FDA-Tax Share", "Difference"))
        log_msg("  " * repeat("-", 62))

        for (c, label) in enumerate(cat_labels)
            cat_val = c - 1
            alt_indices = findall(j -> cat_idx[j] == cat_val, 1:N_J)
            sq_share      = mean(sum(probs_sq[:, alt_indices], dims=2))
            fda_tax_share = mean(sum(probs_fda_tax[:, alt_indices], dims=2))
            log_msg(@sprintf("  %-22s  %12.6f  %12.6f  %12.6f", label, sq_share, fda_tax_share, fda_tax_share - sq_share))
        end

        welfare_diff = welfare_fda_tax .- welfare_sq
        log_msg(@sprintf("\n  Mean welfare SQ:          %.6f", mean(welfare_sq)))
        log_msg(@sprintf("  Mean welfare FDA-Only Tax: %.6f", mean(welfare_fda_tax)))
        log_msg(@sprintf("  Mean welfare loss:         %.6f", mean(welfare_diff)))

        # Compute expected tax revenue per household-month (FDA-authorized alts only)
        fda_revenue = 0.0
        for i in 1:N_obs
            for j in fda_flav_alts
                fda_revenue += probs_fda_tax[i, j] * q_ecig[j] * q_ecig_max * tau
            end
        end
        mean_fda_revenue = fda_revenue / N_obs
        log_msg(@sprintf("  Mean FDA-only tax revenue per HH-month: \$%.4f", mean_fda_revenue))

        # --- Forward Simulation ---
        log_msg("\n===================================")
        log_msg("Forward simulation (FDA-only τ = \$$tau, β = $beta_val)...")
        log_msg("===================================")
        log_msg("T_sim = $T_sim, N_draws = $N_draws, N_HH = $N_HH")

        # Forward simulation overview:
        #   - For each household, assign to the latent type via posterior probability draw.
        #   - Simulate T_sim periods under both SQ and FDA-only tax using the assigned type's
        #     V_decision. The FDA-only tax targets only FDA-authorized flavored e-cigarettes,
        #     leaving non-FDA flavored alternatives (unauthorized market) untaxed.
        #   - Both scenarios use identical pre-drawn CRN variates to reduce variance of the
        #     SQ-vs-tax difference. The same uniform draw determines the choice in both
        #     scenarios, and the same normal draws drive AR(1) price shocks.
        #   - Each draw starts from the household's terminal addiction state (a_T), which is
        #     the addiction level after the household's final observed choice.
        #   - Prices evolve via AR(1) in both scenarios (no supply-side re-equilibration).
        #   - TYA state is held fixed at the household's last observed TYA state.

        sim_choices_sq          = Array{Int}(undef, N_HH, T_sim, N_draws)
        sim_addiction_f_sq      = Array{Float64}(undef, N_HH, T_sim, N_draws)
        sim_addiction_s_sq      = Array{Float64}(undef, N_HH, T_sim, N_draws)
        sim_aflav_sq            = Array{Float64}(undef, N_HH, T_sim, N_draws)
        sim_welfare_sq_arr      = Array{Float64}(undef, N_HH, T_sim, N_draws)

        sim_choices_fda_tax     = Array{Int}(undef, N_HH, T_sim, N_draws)
        sim_addiction_f_fda_tax = Array{Float64}(undef, N_HH, T_sim, N_draws)
        sim_addiction_s_fda_tax = Array{Float64}(undef, N_HH, T_sim, N_draws)
        sim_aflav_fda_tax       = Array{Float64}(undef, N_HH, T_sim, N_draws)
        sim_welfare_fda_tax_arr = Array{Float64}(undef, N_HH, T_sim, N_draws)

        P_cig  = P[:, 1]
        P_ecig = P[:, 2]

        t_sim_fwd = time();

        Threads.@threads for h in 1:N_HH

            tya_idx_h = hh_tya[h]
            w1 = hh_posterior_type1[h]
            w2 = hh_posterior_type2[h]

            for d in 1:N_draws

                # Probabilistic type assignment: draw type from posterior using pre-drawn uniform.
                # If u <= P(type=1) → type 1; elif u <= P(type=1)+P(type=2) → type 2; else type 3.
                u_type = crn_type[h, d]
                type_k = u_type <= w1 ? 1 : (u_type <= w1 + w2 ? 2 : 3)
                V_sq      = type_k == 1 ? V_decision_1_sq      : (type_k == 2 ? V_decision_2_sq      : V_decision_3_sq)
                V_fda_tax = type_k == 1 ? V_decision_1_fda_tax : (type_k == 2 ? V_decision_2_fda_tax : V_decision_3_fda_tax)

                # Initialize both scenarios from the household's terminal observed state
                a_f_sq = hh_af0[h]; a_s_sq = hh_as0[h]; a_flav_sq = hh_aflav0[h]
                p_cig_sq = hh_p0[h, 1]; p_ecig_sq = hh_p0[h, 2]
                a_f_fda_tax = hh_af0[h]; a_s_fda_tax = hh_as0[h]; a_flav_fda_tax = hh_aflav0[h]
                p_cig_fda_tax = hh_p0[h, 1]; p_ecig_fda_tax = hh_p0[h, 2]
                tya_sq = tya_idx_h; tya_fda_tax = tya_idx_h

                for t in 1:T_sim

                    # --- STATUS QUO ---

                    # 5-linear interpolation of V_decision at the household's continuous state;
                    # NaN from 0.0 * (-Inf) at exact grid boundaries replaced with -Inf.
                    v_interp_sq = interpolate_v_choice(V_sq, tya_sq, a_f_sq, a_s_sq, a_flav_sq, p_cig_sq, p_ecig_sq, N_J, N_P, A_f, A_s, A_flav, P)
                    replace!(v_interp_sq, NaN => -Inf)

                    # Welfare = logsumexp(V_decision) = expected maximum utility (inclusive value)
                    sim_welfare_sq_arr[h, t, d] = logsumexp(v_interp_sq)

                    # Softmax choice probabilities: P(j) = exp(V_j - V_max) / sum_j' exp(V_j' - V_max)
                    v_max_sq = maximum(v_interp_sq)
                    exp_v_sq = exp.(v_interp_sq .- v_max_sq)
                    probs_h_sq = exp_v_sq ./ sum(exp_v_sq)

                    # Inverse CDF sampling using pre-drawn uniform (same draw reused for tax: CRN)
                    u_draw = crn_choice[h, t, d]
                    j_sq = 1; cum_prob = probs_h_sq[1]
                    while cum_prob < u_draw && j_sq < N_J; j_sq += 1; cum_prob += probs_h_sq[j_sq]; end
                    sim_choices_sq[h, t, d] = j_sq

                    # Update fast addiction: a_f' = (1 - ψ_2) * a_f + ψ_2 * n_std[j_sq]
                    # Update slow addiction: a_s' = (1 - ψ_1) * a_s + ψ_1 * n_std[j_sq]
                    # Update flavored habit: a_flav' = (1 - ψ_3) * a_flav + ψ_3 * 1[flavored[j_sq]]
                    a_f_sq = clamp(addiction_evolution(ψ_2, a_f_sq, n[j_sq]), A_f[1], A_f[end])
                    a_s_sq = clamp(addiction_evolution(ψ_1, a_s_sq, n[j_sq]), A_s[1], A_s[end])
                    a_flav_sq = clamp(addiction_evolution(PSI_3, a_flav_sq, n_flav[j_sq]), A_flav[1], A_flav[end])
                    sim_addiction_f_sq[h, t, d] = a_f_sq
                    sim_addiction_s_sq[h, t, d] = a_s_sq
                    sim_aflav_sq[h, t, d] = a_flav_sq

                    # Update prices via AR(1) with Cholesky-correlated shocks (same draws as tax: CRN)
                    ε_sq = L_chol * [crn_price[h, t, d, 1], crn_price[h, t, d, 2]]
                    p_cig_sq  = clamp(φ_0[1] + φ_1[1] * p_cig_sq  + ε_sq[1], P_cig[1], P_cig[end])
                    p_ecig_sq = clamp(φ_0[2] + φ_1[2] * p_ecig_sq + ε_sq[2], P_ecig[1], P_ecig[end])

                    # --- FDA-ONLY FLAVOR TAX ---
                    # Same steps as SQ but using V_decision from the FDA-only taxed flow utility.
                    # Only FDA-authorized flavored e-cigarettes face the per-mL tax; non-FDA
                    # flavored alternatives are unaffected.

                    # 5-linear interpolation and NaN fix (same as SQ above)
                    v_interp_fda_tax = interpolate_v_choice(V_fda_tax, tya_fda_tax, a_f_fda_tax, a_s_fda_tax, a_flav_fda_tax, p_cig_fda_tax, p_ecig_fda_tax, N_J, N_P, A_f, A_s, A_flav, P)
                    replace!(v_interp_fda_tax, NaN => -Inf)

                    # Welfare and softmax choice probabilities
                    sim_welfare_fda_tax_arr[h, t, d] = logsumexp(v_interp_fda_tax)
                    v_max_fda_tax = maximum(v_interp_fda_tax)
                    exp_v_fda_tax = exp.(v_interp_fda_tax .- v_max_fda_tax)
                    probs_h_fda_tax = exp_v_fda_tax ./ sum(exp_v_fda_tax)

                    # Draw choice using same uniform as SQ (CRN)
                    j_fda_tax = 1; cum_prob_fda_tax = probs_h_fda_tax[1]
                    while cum_prob_fda_tax < u_draw && j_fda_tax < N_J; j_fda_tax += 1; cum_prob_fda_tax += probs_h_fda_tax[j_fda_tax]; end
                    sim_choices_fda_tax[h, t, d] = j_fda_tax

                    # Update fast addiction, slow addiction, and flavored habit stock
                    a_f_fda_tax = clamp(addiction_evolution(ψ_2, a_f_fda_tax, n[j_fda_tax]), A_f[1], A_f[end])
                    a_s_fda_tax = clamp(addiction_evolution(ψ_1, a_s_fda_tax, n[j_fda_tax]), A_s[1], A_s[end])
                    a_flav_fda_tax = clamp(addiction_evolution(PSI_3, a_flav_fda_tax, n_flav[j_fda_tax]), A_flav[1], A_flav[end])
                    sim_addiction_f_fda_tax[h, t, d] = a_f_fda_tax
                    sim_addiction_s_fda_tax[h, t, d] = a_s_fda_tax
                    sim_aflav_fda_tax[h, t, d] = a_flav_fda_tax

                    # Update prices via AR(1) with same CRN draws as SQ
                    ε_fda_tax = L_chol * [crn_price[h, t, d, 1], crn_price[h, t, d, 2]]
                    p_cig_fda_tax  = clamp(φ_0[1] + φ_1[1] * p_cig_fda_tax  + ε_fda_tax[1], P_cig[1], P_cig[end])
                    p_ecig_fda_tax = clamp(φ_0[2] + φ_1[2] * p_ecig_fda_tax + ε_fda_tax[2], P_ecig[1], P_ecig[end])

                    # TYA state held fixed (binary, no transitions)
                end
            end
        end

        sim_fwd_elapsed = time() - t_sim_fwd;
        log_msg("Forward simulation complete in $(round(sim_fwd_elapsed, digits=1))s")

        # --- Aggregate and Save ---
        log_msg("\n===================================")
        log_msg("Aggregating results...")
        log_msg("===================================")

        sim_addiction_sq      = (sim_addiction_f_sq      .+ sim_addiction_s_sq)      ./ 2.0
        sim_addiction_fda_tax = (sim_addiction_f_fda_tax .+ sim_addiction_s_fda_tax) ./ 2.0

        agg_sq      = aggregate_simulation(sim_choices_sq,      sim_addiction_sq,      sim_aflav_sq,      sim_welfare_sq_arr,      cat_idx, N_J, T_sim)
        agg_fda_tax = aggregate_simulation(sim_choices_fda_tax, sim_addiction_fda_tax, sim_aflav_fda_tax, sim_welfare_fda_tax_arr, cat_idx, N_J, T_sim)

        agg_sq_tya, agg_sq_no_tya = aggregate_simulation_by_tya(
            sim_choices_sq, sim_addiction_sq, sim_aflav_sq, sim_welfare_sq_arr, cat_idx, N_J, T_sim, hh_tya, N_draws)
        agg_fda_tax_tya, agg_fda_tax_no_tya = aggregate_simulation_by_tya(
            sim_choices_fda_tax, sim_addiction_fda_tax, sim_aflav_fda_tax, sim_welfare_fda_tax_arr, cat_idx, N_J, T_sim, hh_tya, N_draws)

        agg_sq_type1, agg_sq_type2, agg_sq_type3 = aggregate_simulation_by_type_k3(
            sim_choices_sq,      sim_addiction_sq,      sim_aflav_sq,      sim_welfare_sq_arr,      cat_idx, N_J, T_sim, hh_posterior_type1, hh_posterior_type2, hh_posterior_type3, N_draws)
        agg_fda_tax_type1, agg_fda_tax_type2, agg_fda_tax_type3 = aggregate_simulation_by_type_k3(
            sim_choices_fda_tax, sim_addiction_fda_tax, sim_aflav_fda_tax, sim_welfare_fda_tax_arr, cat_idx, N_J, T_sim, hh_posterior_type1, hh_posterior_type2, hh_posterior_type3, N_draws)

        # Save overall simulation results
        sim_results_path = joinpath(beta_subdir, "Simulation_Overall.csv")
        open(sim_results_path, "w") do io
            header = ["period",
                "sq_outside", "sq_cig", "sq_orig_ecig", "sq_non_fda_flav_ecig", "sq_fda_flav_ecig",
                "sq_orig_bundle", "sq_non_fda_flav_bundle", "sq_fda_flav_bundle",
                "sq_addiction", "sq_aflav", "sq_welfare",
                "fda_tax_outside", "fda_tax_cig", "fda_tax_orig_ecig", "fda_tax_non_fda_flav_ecig", "fda_tax_fda_flav_ecig",
                "fda_tax_orig_bundle", "fda_tax_non_fda_flav_bundle", "fda_tax_fda_flav_bundle",
                "fda_tax_addiction", "fda_tax_aflav", "fda_tax_welfare"]
            println(io, join(header, ","))
            for t in 1:T_sim
                row = [@sprintf("%d", t),
                    @sprintf("%.10f", agg_sq.share_outside[t]), @sprintf("%.10f", agg_sq.share_cig[t]),
                    @sprintf("%.10f", agg_sq.share_orig_ecig[t]), @sprintf("%.10f", agg_sq.share_non_fda_flav_ecig[t]),
                    @sprintf("%.10f", agg_sq.share_fda_flav_ecig[t]), @sprintf("%.10f", agg_sq.share_orig_bundle[t]),
                    @sprintf("%.10f", agg_sq.share_non_fda_flav_bundle[t]), @sprintf("%.10f", agg_sq.share_fda_flav_bundle[t]),
                    @sprintf("%.10f", agg_sq.mean_addiction[t]), @sprintf("%.10f", agg_sq.mean_aflav[t]),
                    @sprintf("%.10f", agg_sq.mean_welfare[t]),
                    @sprintf("%.10f", agg_fda_tax.share_outside[t]), @sprintf("%.10f", agg_fda_tax.share_cig[t]),
                    @sprintf("%.10f", agg_fda_tax.share_orig_ecig[t]), @sprintf("%.10f", agg_fda_tax.share_non_fda_flav_ecig[t]),
                    @sprintf("%.10f", agg_fda_tax.share_fda_flav_ecig[t]), @sprintf("%.10f", agg_fda_tax.share_orig_bundle[t]),
                    @sprintf("%.10f", agg_fda_tax.share_non_fda_flav_bundle[t]), @sprintf("%.10f", agg_fda_tax.share_fda_flav_bundle[t]),
                    @sprintf("%.10f", agg_fda_tax.mean_addiction[t]), @sprintf("%.10f", agg_fda_tax.mean_aflav[t]),
                    @sprintf("%.10f", agg_fda_tax.mean_welfare[t])]
                println(io, join(row, ","))
            end
        end
        log_msg("Overall simulation results saved to: $sim_results_path")

        # Save simulation by TYA
        sim_tya_path = joinpath(beta_subdir, "Simulation_by_TYA.csv")
        open(sim_tya_path, "w") do io
            header = ["group", "period",
                "sq_outside", "sq_cig", "sq_orig_ecig", "sq_non_fda_flav_ecig", "sq_fda_flav_ecig",
                "sq_orig_bundle", "sq_non_fda_flav_bundle", "sq_fda_flav_bundle",
                "sq_addiction", "sq_aflav", "sq_welfare",
                "fda_tax_outside", "fda_tax_cig", "fda_tax_orig_ecig", "fda_tax_non_fda_flav_ecig", "fda_tax_fda_flav_ecig",
                "fda_tax_orig_bundle", "fda_tax_non_fda_flav_bundle", "fda_tax_fda_flav_bundle",
                "fda_tax_addiction", "fda_tax_aflav", "fda_tax_welfare"]
            println(io, join(header, ","))
            for (group_label, agg_sq_g, agg_fda_tax_g) in [("tya", agg_sq_tya, agg_fda_tax_tya), ("no_tya", agg_sq_no_tya, agg_fda_tax_no_tya)]
                for t in 1:T_sim
                    row = [group_label, @sprintf("%d", t),
                        @sprintf("%.10f", agg_sq_g.share_outside[t]), @sprintf("%.10f", agg_sq_g.share_cig[t]),
                        @sprintf("%.10f", agg_sq_g.share_orig_ecig[t]), @sprintf("%.10f", agg_sq_g.share_non_fda_flav_ecig[t]),
                        @sprintf("%.10f", agg_sq_g.share_fda_flav_ecig[t]), @sprintf("%.10f", agg_sq_g.share_orig_bundle[t]),
                        @sprintf("%.10f", agg_sq_g.share_non_fda_flav_bundle[t]), @sprintf("%.10f", agg_sq_g.share_fda_flav_bundle[t]),
                        @sprintf("%.10f", agg_sq_g.mean_addiction[t]), @sprintf("%.10f", agg_sq_g.mean_aflav[t]),
                        @sprintf("%.10f", agg_sq_g.mean_welfare[t]),
                        @sprintf("%.10f", agg_fda_tax_g.share_outside[t]), @sprintf("%.10f", agg_fda_tax_g.share_cig[t]),
                        @sprintf("%.10f", agg_fda_tax_g.share_orig_ecig[t]), @sprintf("%.10f", agg_fda_tax_g.share_non_fda_flav_ecig[t]),
                        @sprintf("%.10f", agg_fda_tax_g.share_fda_flav_ecig[t]), @sprintf("%.10f", agg_fda_tax_g.share_orig_bundle[t]),
                        @sprintf("%.10f", agg_fda_tax_g.share_non_fda_flav_bundle[t]), @sprintf("%.10f", agg_fda_tax_g.share_fda_flav_bundle[t]),
                        @sprintf("%.10f", agg_fda_tax_g.mean_addiction[t]), @sprintf("%.10f", agg_fda_tax_g.mean_aflav[t]),
                        @sprintf("%.10f", agg_fda_tax_g.mean_welfare[t])]
                    println(io, join(row, ","))
                end
            end
        end
        log_msg("TYA simulation results saved to: $sim_tya_path")

        # Save simulation by latent type
        sim_type_path = joinpath(beta_subdir, "Simulation_by_Type.csv")
        open(sim_type_path, "w") do io
            header = ["type", "period",
                "sq_outside", "sq_cig", "sq_orig_ecig", "sq_non_fda_flav_ecig", "sq_fda_flav_ecig",
                "sq_orig_bundle", "sq_non_fda_flav_bundle", "sq_fda_flav_bundle",
                "sq_addiction", "sq_aflav", "sq_welfare",
                "fda_tax_outside", "fda_tax_cig", "fda_tax_orig_ecig", "fda_tax_non_fda_flav_ecig", "fda_tax_fda_flav_ecig",
                "fda_tax_orig_bundle", "fda_tax_non_fda_flav_bundle", "fda_tax_fda_flav_bundle",
                "fda_tax_addiction", "fda_tax_aflav", "fda_tax_welfare"]
            println(io, join(header, ","))
            for (type_label, agg_sq_g, agg_fda_tax_g) in [("type1", agg_sq_type1, agg_fda_tax_type1), ("type2", agg_sq_type2, agg_fda_tax_type2), ("type3", agg_sq_type3, agg_fda_tax_type3)]
                for t in 1:T_sim
                    row = [type_label, @sprintf("%d", t),
                        @sprintf("%.10f", agg_sq_g.share_outside[t]), @sprintf("%.10f", agg_sq_g.share_cig[t]),
                        @sprintf("%.10f", agg_sq_g.share_orig_ecig[t]), @sprintf("%.10f", agg_sq_g.share_non_fda_flav_ecig[t]),
                        @sprintf("%.10f", agg_sq_g.share_fda_flav_ecig[t]), @sprintf("%.10f", agg_sq_g.share_orig_bundle[t]),
                        @sprintf("%.10f", agg_sq_g.share_non_fda_flav_bundle[t]), @sprintf("%.10f", agg_sq_g.share_fda_flav_bundle[t]),
                        @sprintf("%.10f", agg_sq_g.mean_addiction[t]), @sprintf("%.10f", agg_sq_g.mean_aflav[t]),
                        @sprintf("%.10f", agg_sq_g.mean_welfare[t]),
                        @sprintf("%.10f", agg_fda_tax_g.share_outside[t]), @sprintf("%.10f", agg_fda_tax_g.share_cig[t]),
                        @sprintf("%.10f", agg_fda_tax_g.share_orig_ecig[t]), @sprintf("%.10f", agg_fda_tax_g.share_non_fda_flav_ecig[t]),
                        @sprintf("%.10f", agg_fda_tax_g.share_fda_flav_ecig[t]), @sprintf("%.10f", agg_fda_tax_g.share_orig_bundle[t]),
                        @sprintf("%.10f", agg_fda_tax_g.share_non_fda_flav_bundle[t]), @sprintf("%.10f", agg_fda_tax_g.share_fda_flav_bundle[t]),
                        @sprintf("%.10f", agg_fda_tax_g.mean_addiction[t]), @sprintf("%.10f", agg_fda_tax_g.mean_aflav[t]),
                        @sprintf("%.10f", agg_fda_tax_g.mean_welfare[t])]
                    println(io, join(row, ","))
                end
            end
        end
        log_msg("Type simulation results saved to: $sim_type_path")

        # --- Save Extensive Margin Results (main + threshold robustness) ---
        for (thresh, suffix) in [(0.05, "_thresh005"), (0.10, ""), (0.20, "_thresh020")]
            ext_tya_t, ext_no_tya_t = aggregate_extensive_margin_by_tya(
                sim_choices_sq, sim_choices_fda_tax, hh_aflav0, cat_idx, T_sim, hh_tya, N_draws;
                aflav_threshold = thresh
            )
            ext_path_t = joinpath(beta_subdir, "Extensive_Margin_by_TYA$(suffix).csv")
            open(ext_path_t, "w") do io
                header = ["group", "period", "n_non_users",
                          "sq_ever_initiated", "cf_ever_initiated", "prevention_rate"]
                println(io, join(header, ","))
                for (group_label, df_g) in [("tya", ext_tya_t), ("no_tya", ext_no_tya_t)]
                    for t in 1:T_sim
                        row = [group_label,
                               @sprintf("%d",    t),
                               @sprintf("%d",    df_g.n_non_users[t]),
                               @sprintf("%.10f", df_g.sq_ever_initiated[t]),
                               @sprintf("%.10f", df_g.cf_ever_initiated[t]),
                               @sprintf("%.10f", df_g.prevention_rate[t])]
                        println(io, join(row, ","))
                    end
                end
            end
            log_msg("Extensive margin (thresh=$(thresh)) saved to: $ext_path_t")
        end

        # Log simulation summary
        log_msg("\n--- Forward Simulation Summary (averaged over $T_sim periods) ---")
        log_msg(@sprintf("  Mean addiction (SQ):           %.6f", mean(agg_sq.mean_addiction)))
        log_msg(@sprintf("  Mean addiction (FDA-Only Tax): %.6f", mean(agg_fda_tax.mean_addiction)))
        log_msg(@sprintf("  Addiction change:              %.6f", mean(agg_fda_tax.mean_addiction) - mean(agg_sq.mean_addiction)))
        log_msg(@sprintf("  Mean welfare (SQ):             %.6f", mean(agg_sq.mean_welfare)))
        log_msg(@sprintf("  Mean welfare (FDA-Only Tax):   %.6f", mean(agg_fda_tax.mean_welfare)))
        log_msg(@sprintf("  Welfare change:                %.6f", mean(agg_fda_tax.mean_welfare) - mean(agg_sq.mean_welfare)))

        # Free memory
        GC.gc()

    end  # end BETA_GRID loop
end  # end FDA-only TAX_GRID loop


#############################
# Non-FDA Flavor Tax
# (τ applied to cats 3, 6 only;
#  FDA-authorized cats 4, 7 untaxed)
#############################

log_msg("\n\n===================================")
log_msg("Non-FDA Flavor Tax Counterfactual")
log_msg("===================================")

non_fda_tax_subdir = joinpath(output_dir, "Flavor_Tax_Non_FDA")
mkpath(non_fda_tax_subdir)

for tau in TAX_GRID

    tau_tag = replace(@sprintf("%.2f", tau), "." => "p")
    tau_subdir_name = "Tax_$(tau_tag)"

    log_msg("\n\n###################################")
    log_msg("Non-FDA Flavor Tax: \$$(tau) per mL (tag: $tau_tag)")
    log_msg("###################################")

    for beta_val in BETA_GRID

        beta_tag = numeric_tag(beta_val)
        beta_subdir = joinpath(non_fda_tax_subdir, tau_subdir_name, "Beta_$beta_tag")
        mkpath(beta_subdir)

        log_msg("\n###################################")
        log_msg("β = $beta_val (tag: $beta_tag)")
        log_msg("###################################")

        # Declare loop-local variables
        local V_decision_1_sq, vfi_iters_1_sq, vfi_converged_1_sq
        local V_decision_2_sq, vfi_iters_2_sq, vfi_converged_2_sq
        local V_decision_3_sq, vfi_iters_3_sq, vfi_converged_3_sq
        local V_decision_1_non_fda_tax, vfi_iters_1_non_fda_tax, vfi_converged_1_non_fda_tax
        local V_decision_2_non_fda_tax, vfi_iters_2_non_fda_tax, vfi_converged_2_non_fda_tax
        local V_decision_3_non_fda_tax, vfi_iters_3_non_fda_tax, vfi_converged_3_non_fda_tax
        local probs_1_sq, welfare_1_sq, probs_2_sq, welfare_2_sq, probs_3_sq, welfare_3_sq
        local probs_1_non_fda_tax, welfare_1_non_fda_tax, probs_2_non_fda_tax, welfare_2_non_fda_tax, probs_3_non_fda_tax, welfare_3_non_fda_tax
        local hh_posterior_type1, hh_posterior_type2, hh_posterior_type3, mean_w1
        local obs_posterior_type1, obs_posterior_type2, obs_posterior_type3
        local agg_sq_type1, agg_sq_type2, agg_sq_type3
        local agg_non_fda_tax_type1, agg_non_fda_tax_type2, agg_non_fda_tax_type3
        local probs_sq, welfare_sq, probs_non_fda_tax, welfare_non_fda_tax

        # Unpack β-specific objects pre-loaded in section 13
        local _bp = beta_params[beta_val]
        local U_1 = _bp.U_1; local U_2 = _bp.U_2; local U_3 = _bp.U_3
        local PSI_3 = _bp.PSI_3
        local N_A_flav = _bp.N_A_flav; local A_flav = _bp.A_flav
        local aflav_lower = _bp.aflav_lower; local aflav_upper = _bp.aflav_upper
        local aflav_weight = _bp.aflav_weight
        local aflav_continuous = _bp.aflav_continuous
        local hh_aflav0 = _bp.hh_aflav0
        local π_0_2 = _bp.π_0_2; local π_TYA_2 = _bp.π_TYA_2
        local π_0_3 = _bp.π_0_3; local π_TYA_3 = _bp.π_TYA_3
        local omega_E_est = _bp.omega_E_est

        # Apply non-FDA tax (cats 3, 6) to this β's base flow utilities
        local U_1_non_fda_tax = copy(U_1); apply_non_fda_flavor_tax!(U_1_non_fda_tax, cat_idx, omega_E_est, q_ecig, q_ecig_max, tau)
        local U_2_non_fda_tax = copy(U_2); apply_non_fda_flavor_tax!(U_2_non_fda_tax, cat_idx, omega_E_est, q_ecig, q_ecig_max, tau)
        local U_3_non_fda_tax = copy(U_3); apply_non_fda_flavor_tax!(U_3_non_fda_tax, cat_idx, omega_E_est, q_ecig, q_ecig_max, tau)

        # Log utility shift for median non-FDA flavored alternative
        local non_fda_alts = findall(j -> cat_idx[j] in (3, 6), 1:N_J)
        local median_q_non_fda = median(q_ecig[non_fda_alts] .* q_ecig_max)
        log_msg(@sprintf("ω_E (β=%.2f) = %.10f; median non-FDA-auth alt: q_ecig_raw = %.2f mL, utility shift = %.6f",
            beta_val, omega_E_est, median_q_non_fda, omega_E_est * tau * median_q_non_fda))

        # --- Solve VFI: Status Quo ---
        log_msg("\n===================================")
        log_msg("Solving VFI: Status Quo (β = $beta_val)")
        log_msg("===================================")

        t_vfi = time();
        task_sq_1 = Threads.@spawn solve_vfi_sophisticated(
            N_J, N_A_f, N_A_s, N_A_flav, N_P, N_Pcomb, beta_val, δ, U_1,
            af_lower, af_upper, af_weight,
            as_lower, as_upper, as_weight,
            aflav_lower, aflav_upper, aflav_weight,
            p_cig_lo, p_cig_hi, p_cig_w, p_ecig_lo, p_ecig_hi, p_ecig_w;
            V_init = nothing,
            verbose = true
        )
        task_sq_2 = Threads.@spawn solve_vfi_sophisticated(
            N_J, N_A_f, N_A_s, N_A_flav, N_P, N_Pcomb, beta_val, δ, U_2,
            af_lower, af_upper, af_weight,
            as_lower, as_upper, as_weight,
            aflav_lower, aflav_upper, aflav_weight,
            p_cig_lo, p_cig_hi, p_cig_w, p_ecig_lo, p_ecig_hi, p_ecig_w;
            V_init = nothing,
            verbose = true
        )
        task_sq_3 = Threads.@spawn solve_vfi_sophisticated(
            N_J, N_A_f, N_A_s, N_A_flav, N_P, N_Pcomb, beta_val, δ, U_3,
            af_lower, af_upper, af_weight,
            as_lower, as_upper, as_weight,
            aflav_lower, aflav_upper, aflav_weight,
            p_cig_lo, p_cig_hi, p_cig_w, p_ecig_lo, p_ecig_hi, p_ecig_w;
            V_init = nothing,
            verbose = true
        )
        _, V_decision_1_sq, vfi_iters_1_sq, vfi_converged_1_sq = fetch(task_sq_1)
        _, V_decision_2_sq, vfi_iters_2_sq, vfi_converged_2_sq = fetch(task_sq_2)
        _, V_decision_3_sq, vfi_iters_3_sq, vfi_converged_3_sq = fetch(task_sq_3)
        log_msg("Status quo VFI: type1=$(vfi_iters_1_sq) iters ($(vfi_converged_1_sq)), type2=$(vfi_iters_2_sq) iters ($(vfi_converged_2_sq)), type3=$(vfi_iters_3_sq) iters ($(vfi_converged_3_sq)), $(round(time() - t_vfi, digits=1))s")

        # --- Solve VFI: Non-FDA Flavor Tax ---
        log_msg("\n===================================")
        log_msg("Solving VFI: Non-FDA Flavor Tax (β = $beta_val)")
        log_msg("===================================")

        t_vfi = time();
        task_non_fda_tax_1 = Threads.@spawn solve_vfi_sophisticated(
            N_J, N_A_f, N_A_s, N_A_flav, N_P, N_Pcomb, beta_val, δ, U_1_non_fda_tax,
            af_lower, af_upper, af_weight,
            as_lower, as_upper, as_weight,
            aflav_lower, aflav_upper, aflav_weight,
            p_cig_lo, p_cig_hi, p_cig_w, p_ecig_lo, p_ecig_hi, p_ecig_w;
            V_init = nothing,
            verbose = true
        )
        task_non_fda_tax_2 = Threads.@spawn solve_vfi_sophisticated(
            N_J, N_A_f, N_A_s, N_A_flav, N_P, N_Pcomb, beta_val, δ, U_2_non_fda_tax,
            af_lower, af_upper, af_weight,
            as_lower, as_upper, as_weight,
            aflav_lower, aflav_upper, aflav_weight,
            p_cig_lo, p_cig_hi, p_cig_w, p_ecig_lo, p_ecig_hi, p_ecig_w;
            V_init = nothing,
            verbose = true
        )
        task_non_fda_tax_3 = Threads.@spawn solve_vfi_sophisticated(
            N_J, N_A_f, N_A_s, N_A_flav, N_P, N_Pcomb, beta_val, δ, U_3_non_fda_tax,
            af_lower, af_upper, af_weight,
            as_lower, as_upper, as_weight,
            aflav_lower, aflav_upper, aflav_weight,
            p_cig_lo, p_cig_hi, p_cig_w, p_ecig_lo, p_ecig_hi, p_ecig_w;
            V_init = nothing,
            verbose = true
        )
        _, V_decision_1_non_fda_tax, vfi_iters_1_non_fda_tax, vfi_converged_1_non_fda_tax = fetch(task_non_fda_tax_1)
        _, V_decision_2_non_fda_tax, vfi_iters_2_non_fda_tax, vfi_converged_2_non_fda_tax = fetch(task_non_fda_tax_2)
        _, V_decision_3_non_fda_tax, vfi_iters_3_non_fda_tax, vfi_converged_3_non_fda_tax = fetch(task_non_fda_tax_3)
        log_msg("Non-FDA Tax VFI: type1=$(vfi_iters_1_non_fda_tax) iters ($(vfi_converged_1_non_fda_tax)), type2=$(vfi_iters_2_non_fda_tax) iters ($(vfi_converged_2_non_fda_tax)), type3=$(vfi_iters_3_non_fda_tax) iters ($(vfi_converged_3_non_fda_tax)), $(round(time() - t_vfi, digits=1))s")

        # --- Compute posterior type weights ---
        log_msg("\n===================================")
        log_msg("Computing posterior type weights...")
        log_msg("===================================")

        t_post = time();
        probs_1_sq, welfare_1_sq = compute_pointwise_outcomes(
            V_decision_1_sq, tya_state, af_continuous, as_continuous, aflav_continuous, p_continuous, N_J, N_P, A_f, A_s, A_flav, P
        )
        probs_2_sq, welfare_2_sq = compute_pointwise_outcomes(
            V_decision_2_sq, tya_state, af_continuous, as_continuous, aflav_continuous, p_continuous, N_J, N_P, A_f, A_s, A_flav, P
        )
        probs_3_sq, welfare_3_sq = compute_pointwise_outcomes(
            V_decision_3_sq, tya_state, af_continuous, as_continuous, aflav_continuous, p_continuous, N_J, N_P, A_f, A_s, A_flav, P
        )
        probs_1_non_fda_tax, welfare_1_non_fda_tax = compute_pointwise_outcomes(
            V_decision_1_non_fda_tax, tya_state, af_continuous, as_continuous, aflav_continuous, p_continuous, N_J, N_P, A_f, A_s, A_flav, P
        )
        probs_2_non_fda_tax, welfare_2_non_fda_tax = compute_pointwise_outcomes(
            V_decision_2_non_fda_tax, tya_state, af_continuous, as_continuous, aflav_continuous, p_continuous, N_J, N_P, A_f, A_s, A_flav, P
        )
        probs_3_non_fda_tax, welfare_3_non_fda_tax = compute_pointwise_outcomes(
            V_decision_3_non_fda_tax, tya_state, af_continuous, as_continuous, aflav_continuous, p_continuous, N_J, N_P, A_f, A_s, A_flav, P
        )

        # Compute household-level posterior type probabilities (K=3 softmax)
        hh_posterior_type1 = Vector{Float64}(undef, N_HH)
        hh_posterior_type2 = Vector{Float64}(undef, N_HH)
        hh_posterior_type3 = Vector{Float64}(undef, N_HH)

        for h in 1:N_HH
            start_idx, stop_idx = hh_ranges[h]

            # Household-specific K=3 softmax log mixing weights
            η_2_h = π_0_2 + π_TYA_2 * tya_share_hh[h]
            η_3_h = π_0_3 + π_TYA_3 * tya_share_hh[h]
            log_sum_exp_h = log(1.0 + exp(η_2_h) + exp(η_3_h))
            log_π_1_h = -log_sum_exp_h
            log_π_2_h = η_2_h - log_sum_exp_h
            log_π_3_h = η_3_h - log_sum_exp_h

            # Accumulate per-type log-likelihood for this household
            ll_1 = 0.0; ll_2 = 0.0; ll_3 = 0.0
            for i in start_idx:stop_idx
                ll_1 += log(max(probs_1_sq[i, y[i]], 1e-300))
                ll_2 += log(max(probs_2_sq[i, y[i]], 1e-300))
                ll_3 += log(max(probs_3_sq[i, y[i]], 1e-300))
            end

            # Posterior via logsumexp normalization: P(k|y_h) = exp(log_pi_k + log_ll_k - log_denom)
            a_term = log_π_1_h + ll_1
            b_term = log_π_2_h + ll_2
            c_term = log_π_3_h + ll_3
            log_max = max(a_term, b_term, c_term)
            log_denom = log_max + log(exp(a_term - log_max) + exp(b_term - log_max) + exp(c_term - log_max))
            hh_posterior_type1[h] = exp(a_term - log_denom)
            hh_posterior_type2[h] = exp(b_term - log_denom)
            hh_posterior_type3[h] = exp(c_term - log_denom)
        end

        mean_w1 = mean(hh_posterior_type1)
        log_msg("Posterior type weights computed in $(round(time() - t_post, digits=1))s")
        log_msg(@sprintf("  Mean P(type=1) = %.4f, Mean P(type=2) = %.4f, Mean P(type=3) = %.4f",
            mean_w1, mean(hh_posterior_type2), mean(hh_posterior_type3)))

        # Map posterior to observation level
        obs_posterior_type1 = Vector{Float64}(undef, N_obs)
        obs_posterior_type2 = Vector{Float64}(undef, N_obs)
        obs_posterior_type3 = Vector{Float64}(undef, N_obs)
        for h in 1:N_HH
            start_idx, stop_idx = hh_ranges[h]
            for i in start_idx:stop_idx
                obs_posterior_type1[i] = hh_posterior_type1[h]
                obs_posterior_type2[i] = hh_posterior_type2[h]
                obs_posterior_type3[i] = hh_posterior_type3[h]
            end
        end

        # Mixture-weighted pointwise outcomes — law of total probability over the latent type:
        #   P̂(j|i) = Σ_l P̂(l|y_h) · P(j|l, x_i)
        # Integrates out type uncertainty; hard-assigning each HH to its modal type would
        # discard posterior mass on non-modal types and bias predicted in-sample market shares.
        probs_sq          = obs_posterior_type1 .* probs_1_sq          .+ obs_posterior_type2 .* probs_2_sq          .+ obs_posterior_type3 .* probs_3_sq
        welfare_sq        = obs_posterior_type1 .* welfare_1_sq        .+ obs_posterior_type2 .* welfare_2_sq        .+ obs_posterior_type3 .* welfare_3_sq
        probs_non_fda_tax = obs_posterior_type1 .* probs_1_non_fda_tax .+ obs_posterior_type2 .* probs_2_non_fda_tax .+ obs_posterior_type3 .* probs_3_non_fda_tax
        welfare_non_fda_tax = obs_posterior_type1 .* welfare_1_non_fda_tax .+ obs_posterior_type2 .* welfare_2_non_fda_tax .+ obs_posterior_type3 .* welfare_3_non_fda_tax

        # --- Log pointwise summary ---
        t_pw = time();
        log_msg("\nPointwise summary (means across all observations):")
        log_msg(@sprintf("  %-22s  %12s  %12s  %12s", "Category", "SQ Share", "Non-FDA-Tax Share", "Difference"))
        log_msg("  " * repeat("-", 62))

        for (c, label) in enumerate(cat_labels)
            cat_val = c - 1
            alt_indices = findall(j -> cat_idx[j] == cat_val, 1:N_J)
            sq_share          = mean(sum(probs_sq[:, alt_indices], dims=2))
            non_fda_tax_share = mean(sum(probs_non_fda_tax[:, alt_indices], dims=2))
            log_msg(@sprintf("  %-22s  %12.6f  %12.6f  %12.6f", label, sq_share, non_fda_tax_share, non_fda_tax_share - sq_share))
        end

        welfare_diff = welfare_non_fda_tax .- welfare_sq
        log_msg(@sprintf("\n  Mean welfare SQ:           %.6f", mean(welfare_sq)))
        log_msg(@sprintf("  Mean welfare Non-FDA Tax:   %.6f", mean(welfare_non_fda_tax)))
        log_msg(@sprintf("  Mean welfare loss:          %.6f", mean(welfare_diff)))

        # Compute expected tax revenue per household-month (non-FDA flavored alts only)
        non_fda_revenue = 0.0
        for i in 1:N_obs
            for j in non_fda_alts
                non_fda_revenue += probs_non_fda_tax[i, j] * q_ecig[j] * q_ecig_max * tau
            end
        end
        mean_non_fda_revenue = non_fda_revenue / N_obs
        log_msg(@sprintf("  Mean non-FDA tax revenue per HH-month: \$%.4f", mean_non_fda_revenue))

        # --- Forward Simulation ---
        log_msg("\n===================================")
        log_msg("Forward simulation (non-FDA τ = \$$tau, β = $beta_val)...")
        log_msg("===================================")
        log_msg("T_sim = $T_sim, N_draws = $N_draws, N_HH = $N_HH")

        # Forward simulation overview:
        #   - For each household, assign to the latent type via posterior probability draw.
        #   - Simulate T_sim periods under both SQ and non-FDA tax using the assigned type's
        #     V_decision. The non-FDA tax targets only non-FDA flavored e-cigarettes (cats 3, 6),
        #     leaving FDA-authorized flavored alternatives (cats 4, 7) untaxed.
        #   - Both scenarios use identical pre-drawn CRN variates to reduce variance of the
        #     SQ-vs-tax difference. The same uniform draw determines the choice in both
        #     scenarios, and the same normal draws drive AR(1) price shocks.
        #   - Each draw starts from the household's terminal addiction state (a_T), which is
        #     the addiction level after the household's final observed choice.
        #   - Prices evolve via AR(1) in both scenarios (no supply-side re-equilibration).
        #   - TYA state is held fixed at the household's last observed TYA state.

        sim_choices_sq              = Array{Int}(undef, N_HH, T_sim, N_draws)
        sim_addiction_f_sq          = Array{Float64}(undef, N_HH, T_sim, N_draws)
        sim_addiction_s_sq          = Array{Float64}(undef, N_HH, T_sim, N_draws)
        sim_aflav_sq                = Array{Float64}(undef, N_HH, T_sim, N_draws)
        sim_welfare_sq_arr          = Array{Float64}(undef, N_HH, T_sim, N_draws)

        sim_choices_non_fda_tax     = Array{Int}(undef, N_HH, T_sim, N_draws)
        sim_addiction_f_non_fda_tax = Array{Float64}(undef, N_HH, T_sim, N_draws)
        sim_addiction_s_non_fda_tax = Array{Float64}(undef, N_HH, T_sim, N_draws)
        sim_aflav_non_fda_tax       = Array{Float64}(undef, N_HH, T_sim, N_draws)
        sim_welfare_non_fda_tax_arr = Array{Float64}(undef, N_HH, T_sim, N_draws)

        P_cig  = P[:, 1]
        P_ecig = P[:, 2]

        t_sim_fwd = time();

        Threads.@threads for h in 1:N_HH

            tya_idx_h = hh_tya[h]
            w1 = hh_posterior_type1[h]
            w2 = hh_posterior_type2[h]

            for d in 1:N_draws

                # Probabilistic type assignment: draw type from posterior using pre-drawn uniform.
                # If u <= P(type=1) → type 1; elif u <= P(type=1)+P(type=2) → type 2; else type 3.
                u_type = crn_type[h, d]
                type_k = u_type <= w1 ? 1 : (u_type <= w1 + w2 ? 2 : 3)
                V_sq          = type_k == 1 ? V_decision_1_sq          : (type_k == 2 ? V_decision_2_sq          : V_decision_3_sq)
                V_non_fda_tax = type_k == 1 ? V_decision_1_non_fda_tax : (type_k == 2 ? V_decision_2_non_fda_tax : V_decision_3_non_fda_tax)

                # Initialize both scenarios from the household's terminal observed state
                a_f_sq = hh_af0[h]; a_s_sq = hh_as0[h]; a_flav_sq = hh_aflav0[h]
                p_cig_sq = hh_p0[h, 1]; p_ecig_sq = hh_p0[h, 2]
                a_f_non_fda_tax = hh_af0[h]; a_s_non_fda_tax = hh_as0[h]; a_flav_non_fda_tax = hh_aflav0[h]
                p_cig_non_fda_tax = hh_p0[h, 1]; p_ecig_non_fda_tax = hh_p0[h, 2]
                tya_sq = tya_idx_h; tya_non_fda_tax = tya_idx_h

                for t in 1:T_sim

                    # --- STATUS QUO ---

                    # 5-linear interpolation of V_decision at the household's continuous state;
                    # NaN from 0.0 * (-Inf) at exact grid boundaries replaced with -Inf.
                    v_interp_sq = interpolate_v_choice(V_sq, tya_sq, a_f_sq, a_s_sq, a_flav_sq, p_cig_sq, p_ecig_sq, N_J, N_P, A_f, A_s, A_flav, P)
                    replace!(v_interp_sq, NaN => -Inf)

                    # Welfare = logsumexp(V_decision) = expected maximum utility (inclusive value)
                    sim_welfare_sq_arr[h, t, d] = logsumexp(v_interp_sq)

                    # Softmax choice probabilities: P(j) = exp(V_j - V_max) / sum_j' exp(V_j' - V_max)
                    v_max_sq = maximum(v_interp_sq)
                    exp_v_sq = exp.(v_interp_sq .- v_max_sq)
                    probs_h_sq = exp_v_sq ./ sum(exp_v_sq)

                    # Inverse CDF sampling using pre-drawn uniform (same draw reused for tax: CRN)
                    u_draw = crn_choice[h, t, d]
                    j_sq = 1; cum_prob = probs_h_sq[1]
                    while cum_prob < u_draw && j_sq < N_J; j_sq += 1; cum_prob += probs_h_sq[j_sq]; end
                    sim_choices_sq[h, t, d] = j_sq

                    # Update fast addiction: a_f' = (1 - ψ_2) * a_f + ψ_2 * n_std[j_sq]
                    # Update slow addiction: a_s' = (1 - ψ_1) * a_s + ψ_1 * n_std[j_sq]
                    # Update flavored habit: a_flav' = (1 - ψ_3) * a_flav + ψ_3 * 1[flavored[j_sq]]
                    a_f_sq = clamp(addiction_evolution(ψ_2, a_f_sq, n[j_sq]), A_f[1], A_f[end])
                    a_s_sq = clamp(addiction_evolution(ψ_1, a_s_sq, n[j_sq]), A_s[1], A_s[end])
                    a_flav_sq = clamp(addiction_evolution(PSI_3, a_flav_sq, n_flav[j_sq]), A_flav[1], A_flav[end])
                    sim_addiction_f_sq[h, t, d] = a_f_sq
                    sim_addiction_s_sq[h, t, d] = a_s_sq
                    sim_aflav_sq[h, t, d] = a_flav_sq

                    # Update prices via AR(1) with Cholesky-correlated shocks (same draws as tax: CRN)
                    ε_sq = L_chol * [crn_price[h, t, d, 1], crn_price[h, t, d, 2]]
                    p_cig_sq  = clamp(φ_0[1] + φ_1[1] * p_cig_sq  + ε_sq[1], P_cig[1], P_cig[end])
                    p_ecig_sq = clamp(φ_0[2] + φ_1[2] * p_ecig_sq + ε_sq[2], P_ecig[1], P_ecig[end])

                    # --- NON-FDA FLAVOR TAX ---
                    # Same steps as SQ but using V_decision from the non-FDA taxed flow utility.
                    # Only non-FDA flavored e-cigarettes (cats 3, 6) face the per-mL tax;
                    # FDA-authorized flavored alternatives (cats 4, 7) are unaffected.

                    # 5-linear interpolation and NaN fix (same as SQ above)
                    v_interp_non_fda_tax = interpolate_v_choice(V_non_fda_tax, tya_non_fda_tax, a_f_non_fda_tax, a_s_non_fda_tax, a_flav_non_fda_tax, p_cig_non_fda_tax, p_ecig_non_fda_tax, N_J, N_P, A_f, A_s, A_flav, P)
                    replace!(v_interp_non_fda_tax, NaN => -Inf)

                    # Welfare and softmax choice probabilities
                    sim_welfare_non_fda_tax_arr[h, t, d] = logsumexp(v_interp_non_fda_tax)
                    v_max_non_fda_tax = maximum(v_interp_non_fda_tax)
                    exp_v_non_fda_tax = exp.(v_interp_non_fda_tax .- v_max_non_fda_tax)
                    probs_h_non_fda_tax = exp_v_non_fda_tax ./ sum(exp_v_non_fda_tax)

                    # Draw choice using same uniform as SQ (CRN)
                    j_non_fda_tax = 1; cum_prob_non_fda_tax = probs_h_non_fda_tax[1]
                    while cum_prob_non_fda_tax < u_draw && j_non_fda_tax < N_J; j_non_fda_tax += 1; cum_prob_non_fda_tax += probs_h_non_fda_tax[j_non_fda_tax]; end
                    sim_choices_non_fda_tax[h, t, d] = j_non_fda_tax

                    # Update fast addiction, slow addiction, and flavored habit stock
                    a_f_non_fda_tax = clamp(addiction_evolution(ψ_2, a_f_non_fda_tax, n[j_non_fda_tax]), A_f[1], A_f[end])
                    a_s_non_fda_tax = clamp(addiction_evolution(ψ_1, a_s_non_fda_tax, n[j_non_fda_tax]), A_s[1], A_s[end])
                    a_flav_non_fda_tax = clamp(addiction_evolution(PSI_3, a_flav_non_fda_tax, n_flav[j_non_fda_tax]), A_flav[1], A_flav[end])
                    sim_addiction_f_non_fda_tax[h, t, d] = a_f_non_fda_tax
                    sim_addiction_s_non_fda_tax[h, t, d] = a_s_non_fda_tax
                    sim_aflav_non_fda_tax[h, t, d] = a_flav_non_fda_tax

                    # Update prices via AR(1) with same CRN draws as SQ
                    ε_non_fda_tax = L_chol * [crn_price[h, t, d, 1], crn_price[h, t, d, 2]]
                    p_cig_non_fda_tax  = clamp(φ_0[1] + φ_1[1] * p_cig_non_fda_tax  + ε_non_fda_tax[1], P_cig[1], P_cig[end])
                    p_ecig_non_fda_tax = clamp(φ_0[2] + φ_1[2] * p_ecig_non_fda_tax + ε_non_fda_tax[2], P_ecig[1], P_ecig[end])

                    # TYA state held fixed (binary, no transitions)
                end
            end
        end

        sim_fwd_elapsed = time() - t_sim_fwd;
        log_msg("Forward simulation complete in $(round(sim_fwd_elapsed, digits=1))s")

        # --- Aggregate and Save ---
        log_msg("\n===================================")
        log_msg("Aggregating results...")
        log_msg("===================================")

        sim_addiction_sq          = (sim_addiction_f_sq          .+ sim_addiction_s_sq)          ./ 2.0
        sim_addiction_non_fda_tax = (sim_addiction_f_non_fda_tax .+ sim_addiction_s_non_fda_tax) ./ 2.0

        agg_sq          = aggregate_simulation(sim_choices_sq,          sim_addiction_sq,          sim_aflav_sq,          sim_welfare_sq_arr,          cat_idx, N_J, T_sim)
        agg_non_fda_tax = aggregate_simulation(sim_choices_non_fda_tax, sim_addiction_non_fda_tax, sim_aflav_non_fda_tax, sim_welfare_non_fda_tax_arr, cat_idx, N_J, T_sim)

        agg_sq_tya, agg_sq_no_tya = aggregate_simulation_by_tya(
            sim_choices_sq, sim_addiction_sq, sim_aflav_sq, sim_welfare_sq_arr, cat_idx, N_J, T_sim, hh_tya, N_draws)
        agg_non_fda_tax_tya, agg_non_fda_tax_no_tya = aggregate_simulation_by_tya(
            sim_choices_non_fda_tax, sim_addiction_non_fda_tax, sim_aflav_non_fda_tax, sim_welfare_non_fda_tax_arr, cat_idx, N_J, T_sim, hh_tya, N_draws)

        agg_sq_type1, agg_sq_type2, agg_sq_type3 = aggregate_simulation_by_type_k3(
            sim_choices_sq,          sim_addiction_sq,          sim_aflav_sq,          sim_welfare_sq_arr,          cat_idx, N_J, T_sim, hh_posterior_type1, hh_posterior_type2, hh_posterior_type3, N_draws)
        agg_non_fda_tax_type1, agg_non_fda_tax_type2, agg_non_fda_tax_type3 = aggregate_simulation_by_type_k3(
            sim_choices_non_fda_tax, sim_addiction_non_fda_tax, sim_aflav_non_fda_tax, sim_welfare_non_fda_tax_arr, cat_idx, N_J, T_sim, hh_posterior_type1, hh_posterior_type2, hh_posterior_type3, N_draws)

        # Save overall simulation results
        sim_results_path = joinpath(beta_subdir, "Simulation_Overall.csv")
        open(sim_results_path, "w") do io
            header = ["period",
                "sq_outside", "sq_cig", "sq_orig_ecig", "sq_non_fda_flav_ecig", "sq_fda_flav_ecig",
                "sq_orig_bundle", "sq_non_fda_flav_bundle", "sq_fda_flav_bundle",
                "sq_addiction", "sq_aflav", "sq_welfare",
                "non_fda_tax_outside", "non_fda_tax_cig", "non_fda_tax_orig_ecig", "non_fda_tax_non_fda_flav_ecig", "non_fda_tax_fda_flav_ecig",
                "non_fda_tax_orig_bundle", "non_fda_tax_non_fda_flav_bundle", "non_fda_tax_fda_flav_bundle",
                "non_fda_tax_addiction", "non_fda_tax_aflav", "non_fda_tax_welfare"]
            println(io, join(header, ","))
            for t in 1:T_sim
                row = [@sprintf("%d", t),
                    @sprintf("%.10f", agg_sq.share_outside[t]), @sprintf("%.10f", agg_sq.share_cig[t]),
                    @sprintf("%.10f", agg_sq.share_orig_ecig[t]), @sprintf("%.10f", agg_sq.share_non_fda_flav_ecig[t]),
                    @sprintf("%.10f", agg_sq.share_fda_flav_ecig[t]), @sprintf("%.10f", agg_sq.share_orig_bundle[t]),
                    @sprintf("%.10f", agg_sq.share_non_fda_flav_bundle[t]), @sprintf("%.10f", agg_sq.share_fda_flav_bundle[t]),
                    @sprintf("%.10f", agg_sq.mean_addiction[t]), @sprintf("%.10f", agg_sq.mean_aflav[t]),
                    @sprintf("%.10f", agg_sq.mean_welfare[t]),
                    @sprintf("%.10f", agg_non_fda_tax.share_outside[t]), @sprintf("%.10f", agg_non_fda_tax.share_cig[t]),
                    @sprintf("%.10f", agg_non_fda_tax.share_orig_ecig[t]), @sprintf("%.10f", agg_non_fda_tax.share_non_fda_flav_ecig[t]),
                    @sprintf("%.10f", agg_non_fda_tax.share_fda_flav_ecig[t]), @sprintf("%.10f", agg_non_fda_tax.share_orig_bundle[t]),
                    @sprintf("%.10f", agg_non_fda_tax.share_non_fda_flav_bundle[t]), @sprintf("%.10f", agg_non_fda_tax.share_fda_flav_bundle[t]),
                    @sprintf("%.10f", agg_non_fda_tax.mean_addiction[t]), @sprintf("%.10f", agg_non_fda_tax.mean_aflav[t]),
                    @sprintf("%.10f", agg_non_fda_tax.mean_welfare[t])]
                println(io, join(row, ","))
            end
        end
        log_msg("Overall simulation results saved to: $sim_results_path")

        # Save simulation by TYA
        sim_tya_path = joinpath(beta_subdir, "Simulation_by_TYA.csv")
        open(sim_tya_path, "w") do io
            header = ["group", "period",
                "sq_outside", "sq_cig", "sq_orig_ecig", "sq_non_fda_flav_ecig", "sq_fda_flav_ecig",
                "sq_orig_bundle", "sq_non_fda_flav_bundle", "sq_fda_flav_bundle",
                "sq_addiction", "sq_aflav", "sq_welfare",
                "non_fda_tax_outside", "non_fda_tax_cig", "non_fda_tax_orig_ecig", "non_fda_tax_non_fda_flav_ecig", "non_fda_tax_fda_flav_ecig",
                "non_fda_tax_orig_bundle", "non_fda_tax_non_fda_flav_bundle", "non_fda_tax_fda_flav_bundle",
                "non_fda_tax_addiction", "non_fda_tax_aflav", "non_fda_tax_welfare"]
            println(io, join(header, ","))
            for (group_label, agg_sq_g, agg_non_fda_tax_g) in [("tya", agg_sq_tya, agg_non_fda_tax_tya), ("no_tya", agg_sq_no_tya, agg_non_fda_tax_no_tya)]
                for t in 1:T_sim
                    row = [group_label, @sprintf("%d", t),
                        @sprintf("%.10f", agg_sq_g.share_outside[t]), @sprintf("%.10f", agg_sq_g.share_cig[t]),
                        @sprintf("%.10f", agg_sq_g.share_orig_ecig[t]), @sprintf("%.10f", agg_sq_g.share_non_fda_flav_ecig[t]),
                        @sprintf("%.10f", agg_sq_g.share_fda_flav_ecig[t]), @sprintf("%.10f", agg_sq_g.share_orig_bundle[t]),
                        @sprintf("%.10f", agg_sq_g.share_non_fda_flav_bundle[t]), @sprintf("%.10f", agg_sq_g.share_fda_flav_bundle[t]),
                        @sprintf("%.10f", agg_sq_g.mean_addiction[t]), @sprintf("%.10f", agg_sq_g.mean_aflav[t]),
                        @sprintf("%.10f", agg_sq_g.mean_welfare[t]),
                        @sprintf("%.10f", agg_non_fda_tax_g.share_outside[t]), @sprintf("%.10f", agg_non_fda_tax_g.share_cig[t]),
                        @sprintf("%.10f", agg_non_fda_tax_g.share_orig_ecig[t]), @sprintf("%.10f", agg_non_fda_tax_g.share_non_fda_flav_ecig[t]),
                        @sprintf("%.10f", agg_non_fda_tax_g.share_fda_flav_ecig[t]), @sprintf("%.10f", agg_non_fda_tax_g.share_orig_bundle[t]),
                        @sprintf("%.10f", agg_non_fda_tax_g.share_non_fda_flav_bundle[t]), @sprintf("%.10f", agg_non_fda_tax_g.share_fda_flav_bundle[t]),
                        @sprintf("%.10f", agg_non_fda_tax_g.mean_addiction[t]), @sprintf("%.10f", agg_non_fda_tax_g.mean_aflav[t]),
                        @sprintf("%.10f", agg_non_fda_tax_g.mean_welfare[t])]
                    println(io, join(row, ","))
                end
            end
        end
        log_msg("TYA simulation results saved to: $sim_tya_path")

        # Save simulation by latent type
        sim_type_path = joinpath(beta_subdir, "Simulation_by_Type.csv")
        open(sim_type_path, "w") do io
            header = ["type", "period",
                "sq_outside", "sq_cig", "sq_orig_ecig", "sq_non_fda_flav_ecig", "sq_fda_flav_ecig",
                "sq_orig_bundle", "sq_non_fda_flav_bundle", "sq_fda_flav_bundle",
                "sq_addiction", "sq_aflav", "sq_welfare",
                "non_fda_tax_outside", "non_fda_tax_cig", "non_fda_tax_orig_ecig", "non_fda_tax_non_fda_flav_ecig", "non_fda_tax_fda_flav_ecig",
                "non_fda_tax_orig_bundle", "non_fda_tax_non_fda_flav_bundle", "non_fda_tax_fda_flav_bundle",
                "non_fda_tax_addiction", "non_fda_tax_aflav", "non_fda_tax_welfare"]
            println(io, join(header, ","))
            for (type_label, agg_sq_g, agg_non_fda_tax_g) in [("type1", agg_sq_type1, agg_non_fda_tax_type1), ("type2", agg_sq_type2, agg_non_fda_tax_type2), ("type3", agg_sq_type3, agg_non_fda_tax_type3)]
                for t in 1:T_sim
                    row = [type_label, @sprintf("%d", t),
                        @sprintf("%.10f", agg_sq_g.share_outside[t]), @sprintf("%.10f", agg_sq_g.share_cig[t]),
                        @sprintf("%.10f", agg_sq_g.share_orig_ecig[t]), @sprintf("%.10f", agg_sq_g.share_non_fda_flav_ecig[t]),
                        @sprintf("%.10f", agg_sq_g.share_fda_flav_ecig[t]), @sprintf("%.10f", agg_sq_g.share_orig_bundle[t]),
                        @sprintf("%.10f", agg_sq_g.share_non_fda_flav_bundle[t]), @sprintf("%.10f", agg_sq_g.share_fda_flav_bundle[t]),
                        @sprintf("%.10f", agg_sq_g.mean_addiction[t]), @sprintf("%.10f", agg_sq_g.mean_aflav[t]),
                        @sprintf("%.10f", agg_sq_g.mean_welfare[t]),
                        @sprintf("%.10f", agg_non_fda_tax_g.share_outside[t]), @sprintf("%.10f", agg_non_fda_tax_g.share_cig[t]),
                        @sprintf("%.10f", agg_non_fda_tax_g.share_orig_ecig[t]), @sprintf("%.10f", agg_non_fda_tax_g.share_non_fda_flav_ecig[t]),
                        @sprintf("%.10f", agg_non_fda_tax_g.share_fda_flav_ecig[t]), @sprintf("%.10f", agg_non_fda_tax_g.share_orig_bundle[t]),
                        @sprintf("%.10f", agg_non_fda_tax_g.share_non_fda_flav_bundle[t]), @sprintf("%.10f", agg_non_fda_tax_g.share_fda_flav_bundle[t]),
                        @sprintf("%.10f", agg_non_fda_tax_g.mean_addiction[t]), @sprintf("%.10f", agg_non_fda_tax_g.mean_aflav[t]),
                        @sprintf("%.10f", agg_non_fda_tax_g.mean_welfare[t])]
                    println(io, join(row, ","))
                end
            end
        end
        log_msg("Type simulation results saved to: $sim_type_path")

        # --- Save Extensive Margin Results (main + threshold robustness) ---
        for (thresh, suffix) in [(0.05, "_thresh005"), (0.10, ""), (0.20, "_thresh020")]
            ext_tya_t, ext_no_tya_t = aggregate_extensive_margin_by_tya(
                sim_choices_sq, sim_choices_non_fda_tax, hh_aflav0, cat_idx, T_sim, hh_tya, N_draws;
                aflav_threshold = thresh
            )
            ext_path_t = joinpath(beta_subdir, "Extensive_Margin_by_TYA$(suffix).csv")
            open(ext_path_t, "w") do io
                header = ["group", "period", "n_non_users",
                          "sq_ever_initiated", "cf_ever_initiated", "prevention_rate"]
                println(io, join(header, ","))
                for (group_label, df_g) in [("tya", ext_tya_t), ("no_tya", ext_no_tya_t)]
                    for t in 1:T_sim
                        row = [group_label,
                               @sprintf("%d",    t),
                               @sprintf("%d",    df_g.n_non_users[t]),
                               @sprintf("%.10f", df_g.sq_ever_initiated[t]),
                               @sprintf("%.10f", df_g.cf_ever_initiated[t]),
                               @sprintf("%.10f", df_g.prevention_rate[t])]
                        println(io, join(row, ","))
                    end
                end
            end
            log_msg("Extensive margin (thresh=$(thresh)) saved to: $ext_path_t")
        end

        # Log simulation summary
        log_msg("\n--- Forward Simulation Summary (averaged over $T_sim periods) ---")
        log_msg(@sprintf("  Mean addiction (SQ):          %.6f", mean(agg_sq.mean_addiction)))
        log_msg(@sprintf("  Mean addiction (Non-FDA Tax): %.6f", mean(agg_non_fda_tax.mean_addiction)))
        log_msg(@sprintf("  Addiction change:             %.6f", mean(agg_non_fda_tax.mean_addiction) - mean(agg_sq.mean_addiction)))
        log_msg(@sprintf("  Mean welfare (SQ):            %.6f", mean(agg_sq.mean_welfare)))
        log_msg(@sprintf("  Mean welfare (Non-FDA Tax):   %.6f", mean(agg_non_fda_tax.mean_welfare)))
        log_msg(@sprintf("  Welfare change:               %.6f", mean(agg_non_fda_tax.mean_welfare) - mean(agg_sq.mean_welfare)))

        # Free memory
        GC.gc()

    end  # end BETA_GRID loop
end  # end Non-FDA TAX_GRID loop


#############################
# Find τ* (Tax That Matches
# Comprehensive Ban Effect)
#############################

# For each β, find the per-mL tax τ* on flavored e-cigarettes that achieves
# the same mean addiction reduction (averaged over 36 months) as the
# comprehensive flavor ban across all households. Then compare welfare
# costs. Both policies apply universally, so the aggregate addiction effect
# is the natural equivalence benchmark.

log_msg("\n\n===================================")
log_msg("Finding τ* (tax matching ban's addiction reduction)")
log_msg("===================================")

# Collect results across β for summary table
tau_star_results = Vector{Tuple{Float64, Float64, Float64, Float64}}()  # (β, τ*, tax_welfare, ban_welfare)

for beta_val in BETA_GRID

    beta_tag = numeric_tag(beta_val)

    # --- Load comprehensive ban overall simulation (all households) ---
    ban_overall_path = joinpath(output_dir, "Ban_Comprehensive", "Beta_$beta_tag", "Simulation_Overall.csv")
    if !isfile(ban_overall_path)
        log_msg("WARNING: Ban results not found at $ban_overall_path, skipping β = $beta_val")
        continue
    end
    df_ban_all = CSV.read(ban_overall_path, DataFrame)

    # Target: mean addiction reduction (ban vs SQ) across all HH over all periods
    ban_mean_addiction = mean(df_ban_all.ban_addiction)
    sq_mean_addiction  = mean(df_ban_all.sq_addiction)
    ban_addiction_target = ban_mean_addiction - sq_mean_addiction  # negative = reduction

    log_msg(@sprintf("\nβ = %.2f:", beta_val))
    log_msg(@sprintf("  Ban addiction change (all HH, 36-month avg): %.6f (%.2f%%)",
        ban_addiction_target, 100.0 * ban_addiction_target / sq_mean_addiction))

    # --- Load tax overall simulations for each τ (all households) ---
    tau_vals     = Float64[]
    tax_effects  = Float64[]  # mean addiction change (tax - SQ), all HH
    tax_welfares = Float64[]  # mean welfare change (tax - SQ), all HH

    for tau in TAX_GRID
        tau_tag = replace(@sprintf("%.2f", tau), "." => "p")
        tax_overall_path = joinpath(output_dir, "Flavor_Tax", "Tax_$tau_tag", "Beta_$beta_tag", "Simulation_Overall.csv")
        if !isfile(tax_overall_path)
            log_msg("  WARNING: Tax results not found at $tax_overall_path")
            continue
        end
        df_tax_all = CSV.read(tax_overall_path, DataFrame)

        tax_mean_addiction = mean(df_tax_all.tax_addiction)
        tax_addiction_effect = tax_mean_addiction - sq_mean_addiction

        tax_mean_welfare = mean(df_tax_all.tax_welfare)
        sq_mean_welfare  = mean(df_tax_all.sq_welfare)
        tax_welfare_effect = tax_mean_welfare - sq_mean_welfare

        push!(tau_vals, tau)
        push!(tax_effects, tax_addiction_effect)
        push!(tax_welfares, tax_welfare_effect)

        log_msg(@sprintf("  τ = \$%.2f: addiction change = %.6f (%.2f%%), welfare change = %.6f",
            tau, tax_addiction_effect, 100.0 * tax_addiction_effect / sq_mean_addiction, tax_welfare_effect))
    end

    if length(tau_vals) < 2
        log_msg("  Not enough tax results to interpolate τ*")
        continue
    end

    # --- Interpolate to find τ* ---
    # Find the interval [τ_lo, τ_hi] where the tax effect crosses the ban target
    tau_star = NaN
    welfare_at_tau_star = NaN

    for k in 1:(length(tau_vals) - 1)
        # Check if the ban target falls between tax_effects[k] and tax_effects[k+1]
        if (tax_effects[k] - ban_addiction_target) * (tax_effects[k+1] - ban_addiction_target) <= 0
            # Linear interpolation
            w = (ban_addiction_target - tax_effects[k]) / (tax_effects[k+1] - tax_effects[k])
            tau_star = tau_vals[k] + w * (tau_vals[k+1] - tau_vals[k])
            welfare_at_tau_star = tax_welfares[k] + w * (tax_welfares[k+1] - tax_welfares[k])
            break
        end
    end

    # Also get ban welfare for comparison (all HH)
    ban_mean_welfare = mean(df_ban_all.ban_welfare)
    ban_welfare_effect = ban_mean_welfare - sq_mean_welfare

    if isnan(tau_star)
        # Check if ban target is beyond the tax grid
        if ban_addiction_target < minimum(tax_effects)
            log_msg("  τ* > \$$(maximum(TAX_GRID))/mL (ban effect exceeds largest tax)")
        elseif ban_addiction_target > maximum(tax_effects)
            log_msg("  τ* < \$$(minimum(TAX_GRID))/mL (ban effect smaller than smallest tax)")
        else
            log_msg("  Could not interpolate τ* (non-monotonic tax effects)")
        end
        log_msg(@sprintf("  Ban welfare change (all HH): %.6f", ban_welfare_effect))
    else
        log_msg(@sprintf("\n  >>> τ* = \$%.4f per mL", tau_star))
        log_msg(@sprintf("  >>> Tax welfare change at τ* (all HH): %.6f", welfare_at_tau_star))
        log_msg(@sprintf("  >>> Ban welfare change (all HH):        %.6f", ban_welfare_effect))
        log_msg(@sprintf("  >>> Welfare advantage of tax:           %.6f (%.2f%%)",
            welfare_at_tau_star - ban_welfare_effect,
            100.0 * (welfare_at_tau_star - ban_welfare_effect) / abs(ban_welfare_effect)))

        log_msg(@sprintf("  >>> Approximate revenue: τ* × flavored mL consumed"))
        push!(tau_star_results, (beta_val, tau_star, welfare_at_tau_star, ban_welfare_effect))
    end
end

# --- Summary Table: τ* across β ---
log_msg("\n===================================")
log_msg("τ* Summary Across β Values")
log_msg("===================================")
log_msg(@sprintf("  %-8s  %10s  %16s  %16s  %16s", "β", "τ* (\$/mL)", "Tax Welfare Δ", "Ban Welfare Δ", "Tax Advantage"))
log_msg("  " * repeat("-", 70))
for (β_val, τ_star, tax_w, ban_w) in tau_star_results
    advantage = tax_w - ban_w
    log_msg(@sprintf("  %-8.2f  %10.4f  %16.6f  %16.6f  %16.6f", β_val, τ_star, tax_w, ban_w, advantage))
end


#############################
# Generate Figure Data Files
#############################

# Only write figure txt files when running locally (paper directory not on HPC)
if !HPC

    fig_dir = "C:/Users/wbras/OneDrive/Documents/Desktop/UA/4th_Year_Paper/4th_Year_Paper/4th_Year_Paper_Figures"
    mkpath(fig_dir)

    blabels  = [numeric_tag(b) for b in BETA_GRID]   # ["07", "1"]
    ref_tag  = numeric_tag(1.0)                        # "1"  (reference beta for single-beta figures)
    T_fig    = 36

    # Load Simulation_by_TYA.csv for a given policy/beta, split into tya/no_tya DataFrames
    function load_tya_split(path)
        df     = CSV.read(path, DataFrame)
        tya    = sort(filter(r -> r.group == "tya",    df), :period)
        no_tya = sort(filter(r -> r.group == "no_tya", df), :period)
        return tya, no_tya
    end

    # Write a space-separated figure txt file
    function write_fig_txt(path, header, rows)
        open(path, "w") do io
            println(io, header)
            for r in rows
                println(io, join([@sprintf("%.6f", v) for v in r], " "))
            end
        end
        log_msg("  Written: $(basename(path))")
    end

    # Column helpers
    cig_total(df, t, pre) = df[t, "$(pre)_cig"] + df[t, "$(pre)_orig_bundle"] + df[t, "$(pre)_non_fda_flav_bundle"] + df[t, "$(pre)_fda_flav_bundle"]
    orig_ecig_total(df, t, pre)   = df[t, "$(pre)_orig_ecig"] + df[t, "$(pre)_orig_bundle"]
    non_fda_flav_total(df, t, pre) = df[t, "$(pre)_non_fda_flav_ecig"] + df[t, "$(pre)_non_fda_flav_bundle"]

    # Load all simulation data
    # comp[bl][grp], fda[bl][grp], tax[tau_tag][bl][grp]  where grp ∈ {"tya","no_tya"}
    comp = Dict{String, Dict{String, DataFrame}}()
    fda  = Dict{String, Dict{String, DataFrame}}()
    tax  = Dict{String, Dict{String, Dict{String, DataFrame}}}()

    for bl in blabels
        p_comp = joinpath(output_dir, "Ban_Comprehensive", "Beta_$bl", "Simulation_by_TYA.csv")
        p_fda  = joinpath(output_dir, "Ban_FDA_Only",     "Beta_$bl", "Simulation_by_TYA.csv")
        if isfile(p_comp)
            t1, t2 = load_tya_split(p_comp)
            comp[bl] = Dict("tya" => t1, "no_tya" => t2)
        end
        if isfile(p_fda)
            t1, t2 = load_tya_split(p_fda)
            fda[bl] = Dict("tya" => t1, "no_tya" => t2)
        end
    end

    for tau in TAX_GRID
        tau_tag = replace(@sprintf("%.2f", tau), "." => "p")
        tax[tau_tag] = Dict{String, Dict{String, DataFrame}}()
        for bl in blabels
            p_tax = joinpath(output_dir, "Flavor_Tax", "Tax_$tau_tag", "Beta_$bl", "Simulation_by_TYA.csv")
            if isfile(p_tax)
                t1, t2 = load_tya_split(p_tax)
                tax[tau_tag][bl] = Dict("tya" => t1, "no_tya" => t2)
            end
        end
    end

    log_msg("\n===================================")
    log_msg("Generating figure data files")
    log_msg("===================================")

    for (grp, suffix) in [("tya", "TYA"), ("no_tya", "No_TYA")]

        comp_ok = all(haskey(comp, bl) for bl in blabels)
        fda_ok  = all(haskey(fda,  bl) for bl in blabels)

        sq_hdr  = join(["sq_$bl"  for bl in blabels], " ")
        ban_hdr = join(["ban_$bl" for bl in blabels], " ")
        tax_hdr = join(["tax_$bl" for bl in blabels], " ")

        if comp_ok

            # ---- Comprehensive ban (addiction, cig share, orig ecig share) ----
            rows = [[float(t); [comp[bl][grp][t, :sq_addiction]  for bl in blabels];
                               [comp[bl][grp][t, :ban_addiction] for bl in blabels]]
                    for t in 1:T_fig]
            write_fig_txt(joinpath(fig_dir, "10_Addiction_Trajectory_$(suffix).txt"),
                          "period $sq_hdr $ban_hdr", rows)

            rows = [[float(t); [cig_total(comp[bl][grp], t, "sq")  for bl in blabels];
                               [cig_total(comp[bl][grp], t, "ban") for bl in blabels]]
                    for t in 1:T_fig]
            write_fig_txt(joinpath(fig_dir, "10_Total_Cig_Share_Trajectory_$(suffix).txt"),
                          "period $sq_hdr $ban_hdr", rows)

            rows = [[float(t); [orig_ecig_total(comp[bl][grp], t, "sq")  for bl in blabels];
                               [orig_ecig_total(comp[bl][grp], t, "ban") for bl in blabels]]
                    for t in 1:T_fig]
            write_fig_txt(joinpath(fig_dir, "10_Total_Orig_Ecig_Share_Trajectory_$(suffix).txt"),
                          "period $sq_hdr $ban_hdr", rows)

        end

        if comp_ok && fda_ok

            # ---- Enforcement gap (pre-computed pcts, per-beta sq) ----
            pct_hdr = join(["comp_$bl fda_$bl" for bl in blabels], " ")
            rows = [[float(t); vcat([[100.0 * (comp[bl][grp][t, :ban_addiction] / comp[bl][grp][t, :sq_addiction] - 1),
                                      100.0 * (fda[bl][grp][t,  :ban_addiction] / comp[bl][grp][t, :sq_addiction] - 1)]
                                     for bl in blabels]...)]
                    for t in 1:T_fig]
            write_fig_txt(joinpath(fig_dir, "11_Enforcement_Gap_Addiction_$(suffix).txt"),
                          "period $pct_hdr", rows)

            # ---- Figure 12: FDA-only ban ----
            rows = [[float(t); [fda[bl][grp][t, :sq_addiction]  for bl in blabels];
                               [fda[bl][grp][t, :ban_addiction] for bl in blabels]]
                    for t in 1:T_fig]
            write_fig_txt(joinpath(fig_dir, "12_FDA_Addiction_Trajectory_$(suffix).txt"),
                          "period $sq_hdr $ban_hdr", rows)

            rows = [[float(t); [cig_total(fda[bl][grp], t, "sq")  for bl in blabels];
                               [cig_total(fda[bl][grp], t, "ban") for bl in blabels]]
                    for t in 1:T_fig]
            write_fig_txt(joinpath(fig_dir, "12_FDA_Total_Cig_Share_Trajectory_$(suffix).txt"),
                          "period $sq_hdr $ban_hdr", rows)

            rows = [[float(t); [orig_ecig_total(fda[bl][grp], t, "sq")  for bl in blabels];
                               [orig_ecig_total(fda[bl][grp], t, "ban") for bl in blabels]]
                    for t in 1:T_fig]
            write_fig_txt(joinpath(fig_dir, "12_FDA_Total_Orig_Ecig_Share_Trajectory_$(suffix).txt"),
                          "period $sq_hdr $ban_hdr", rows)

            # ---- Unauthorized flavor share under FDA-only ban ----
            rows = [[float(t); [non_fda_flav_total(fda[bl][grp], t, "sq")  for bl in blabels];
                               [non_fda_flav_total(fda[bl][grp], t, "ban") for bl in blabels]]
                    for t in 1:T_fig]
            write_fig_txt(joinpath(fig_dir, "13_Unauthorized_Flav_Share_$(suffix).txt"),
                          "period $sq_hdr $ban_hdr", rows)

        end

        # ---- Tax trajectory at ref beta=1 (single sq, 3 tax levels + ban) ----
        tau_tags_sorted = [replace(@sprintf("%.2f", tau), "." => "p") for tau in sort(TAX_GRID)]
        tax_col_hdr     = join(["tax_$tt" for tt in tau_tags_sorted], " ")
        ref_ok          = comp_ok && haskey(comp[ref_tag], grp) &&
                          all(haskey(get(tax, tt, Dict()), ref_tag) for tt in tau_tags_sorted)
        if ref_ok
            rows = [[float(t), comp[ref_tag][grp][t, :sq_addiction],
                     [tax[tt][ref_tag][grp][t, :tax_addiction] for tt in tau_tags_sorted]...,
                     comp[ref_tag][grp][t, :ban_addiction]]
                    for t in 1:T_fig]
            write_fig_txt(joinpath(fig_dir, "14_Tax_Addiction_Trajectory_$(suffix).txt"),
                          "period sq $tax_col_hdr ban", rows)
        end

        # ---- Tax $0.50/mL vs Ban across all beta values ----
        top_tag  = replace(@sprintf("%.2f", 0.50), "." => "p")
        tax50_ok = comp_ok && haskey(tax, top_tag) && all(haskey(tax[top_tag], bl) for bl in blabels)
        if tax50_ok
            full_hdr = "period $sq_hdr $tax_hdr $ban_hdr"

            rows = [[float(t); [comp[bl][grp][t, :sq_addiction]       for bl in blabels];
                               [tax[top_tag][bl][grp][t, :tax_addiction] for bl in blabels];
                               [comp[bl][grp][t, :ban_addiction]       for bl in blabels]]
                    for t in 1:T_fig]
            write_fig_txt(joinpath(fig_dir, "14_Tax_vs_Ban_Addiction_$(suffix).txt"),
                          full_hdr, rows)

            rows = [[float(t); [cig_total(comp[bl][grp], t, "sq")           for bl in blabels];
                               [cig_total(tax[top_tag][bl][grp], t, "tax")  for bl in blabels];
                               [cig_total(comp[bl][grp], t, "ban")          for bl in blabels]]
                    for t in 1:T_fig]
            write_fig_txt(joinpath(fig_dir, "14_Tax_vs_Ban_Cig_Share_$(suffix).txt"),
                          full_hdr, rows)
        end

    end  # end group loop

    log_msg("Figure data generation complete.")

end 


#############################
# Log Final Timing
#############################

# Print and log final timing and completion message
total_elapsed = time() - t_setup;
log_msg("\n===================================")
log_msg("Counterfactual mixture simulation complete")
log_msg(@sprintf("Total time: %.1fs", total_elapsed))
log_msg("β values evaluated: $(BETA_GRID)")
log_msg("Ban types evaluated: $(first.(BAN_TYPES))")
log_msg("===================================")
log_msg("Finished at $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")

# Close the log file handle
close(log_io)
