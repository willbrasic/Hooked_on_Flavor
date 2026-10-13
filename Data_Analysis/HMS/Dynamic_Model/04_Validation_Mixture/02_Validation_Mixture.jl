################################################################################
# William Brasic
# The University of Arizona
# wbrasic97@gmail.com
# June 2026
#
# This script validates the model using simulation-based
# methods. Instead of comparing observed choices to predicted choice
# probabilities, it simulates S forward choice sequences from the estimated
# model, letting the addiction stocks (fast, slow, and flavored habit) evolve
# endogenously based on simulated (not observed) choices.
#
# Observed prices and TYA states are used at each period. Only the addiction
# stocks are simulated forward.
#
# Validation exercises:
#   1. Price elasticities: permanent 1% shock, VFI re-solved; own- and
#      cross-price for cigarettes and e-cigarettes. Participation (extensive
#      margin) and quantity-weighted (both margins) at the observed addiction
#      stocks, plus forward-simulated versions in which the stocks respond,
#      pooled and by months since the price change
#   2. Streak persistence (category persistence): probability of continuing
#      a purchase streak for cig and ecig, for all households, TYA-present
#      households, and modal latent type 1 and type 2 households
#   3. Quantity escalation (quantity persistence): mean and median packs or
#      mL purchased by streak length, in original units, for the same four
#      groups, plus the purchase tallies by quantity bin
################################################################################


#############################
# Preliminaries
#############################

# Present-bias β (read from ENV to match estimation run; never estimated)
BETA = parse(Float64, get(ENV, "BETA", "1.0"))

# Whether ψ_3 was estimated jointly as the 28th parameter (must match the estimation run)
ESTIMATE_PSI_3 = true   # ψ_3 is always estimated jointly as parameter 28

# Fixed flavored habit decay rate (grid seed; θ_hat[28] is the estimate)
PSI_3 = parse(Float64, get(ENV, "PSI_3", "0.50"))

# Fixed parameters (ψ_1, ψ_2 always fixed; β fixed per run but can vary across runs)
ψ_1 = 0.10
ψ_2 = 0.90
β   = BETA

# VFI convergence tolerance 
VFI_TOL = 1e-6

# Number of simulation draws
S = 100

# Detect whether I am running on the HPC (any non-Windows system)
HPC = !Sys.iswindows()

# Load all functions and packages from the estimation functions file
if HPC
    include("../02_Second_Stage_Estimation_Mixture/01_Functions_Mixture.jl")
else
    include("../02_Second_Stage_Estimation_Mixture/01_Functions_Mixture.jl")
end

# Additional imports for simulation
using Random

# Load simulation validation functions
include("01_Validation_Functions_Mixture.jl")


# Construct psi, beta, and psi_3 tags for output directory and file naming.
# These must exactly match the tags used by 02_Estimation_Mixture.jl to locate estimates.
psi_tag   = "Psi2_09_Psi1_01"
beta_tag  = "Beta_$(numeric_tag(BETA))"

# File paths if on HPC or not
if HPC

    # Output path for results (use absolute path so it's unaffected by later cd)
    output_dir = abspath("./Validation_Mixture_$(psi_tag)_$(beta_tag)_Results")
    mkpath(output_dir)

    # Set working directory to where the data CSVs live
    cd("/home/u2/wbrasic/4th_Year_Paper/Data")
else

    # Output path for results (local Windows path)
    output_dir = "C:/Users/wbras/OneDrive/Documents/Desktop/UA/4th_Year_Paper/4th_Year_Paper_Data/HMS/2021-Onward/Dynamic_Model/Validation_Mixture_$(psi_tag)_$(beta_tag)_Results"
    mkpath(output_dir)

    # Set working directory to where the data CSVs live
    cd("C:/Users/wbras/OneDrive/Documents/Desktop/UA/4th_Year_Paper/4th_Year_Paper_Data/HMS/2021-Onward/Dynamic_Model/Data")
end

# Set log file path
log_path = joinpath(output_dir, "Validation_Mixture_Log.txt")

# Open log file for writing (log_io is defined as a global in ../02_Second_Stage_Estimation_Mixture/01_Functions.jl)
log_io = open(log_path, "w")

# Print and log the start time and number of Julia threads available for VFI parallelization
log_msg("Simulation-based K=3 mixture model validation ($(psi_tag)_$(beta_tag)) started at $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
log_msg("")
log_msg("Number of threads: $(Threads.nthreads())")
log_msg("Number of simulation draws: $S")
log_msg("PSI_3          = $PSI_3 (grid seed; overridden by the estimate θ_hat[28])")
log_msg("")

# Get household identifiers 
hh_codes = get_hh_codes();

# Pre-compute contiguous household index ranges for mixture log-likelihood
# this is used to compute posterior type weights
hh_ranges = precompute_hh_ranges(hh_codes);


#############################
# Initialize Fixed Parameters
#############################

# Load fixed parameters: only δ (the per-period discount factor) is extracted
# here. get_fixed_parameters() returns (ψ_1, ψ_2, β, δ), but ψ_1, ψ_2, and β
# are already set in the fixed-parameter block above and from the ENV-read BETA,
# so I discard the first three return values with underscore placeholders.
_, _, _, δ = get_fixed_parameters();


#############################
# State Spaces and Choices
#############################

# Start timer for data prep
t_setup = time();

# Get fast addiction grid (N_A_f = 5 points, fast stock with ψ_2 = 0.90)
N_A_f, A_f = get_addiction_space(ψ_2; N_A=5);

# Get slow addiction grid (N_A_s = 10 points, slow stock with ψ_1 = 0.10)
N_A_s, A_s = get_addiction_space(ψ_1; N_A=10);

# Get number of observations (N_HHT), number of alternatives (N_J), and choice matrix J
_, N_J, J = get_product_choices();

# Convert choice matrix J to choice vector y where y[i] = chosen alternative index for observation i
y = get_hh_choices(J);

# Get number of product categories excluding the outside option
N_K, _ = get_category_choices();


#############################
# Alternative-Level Vectors
#############################

# Get consumption vectors by alternative (STANDARDIZED by max)
# Max values are needed for rescaling parameter estimates to original units
N_cig, N_orig_ecig, N_non_fda_flav_ecig, N_fda_flav_ecig, _, q_cig, q_ecig, q_bundle, q_cig_max, q_ecig_max, q_bundle_max = get_consumption(N_J);

# Get nicotine vector by alternative (STANDARDIZED by max)
# n_max is the raw max value for rescaling estimates
n, n_max = get_nicotine(N_J);

# Get category index by alternative
cat_idx = get_category_index(N_J, N_cig, N_orig_ecig, N_non_fda_flav_ecig, N_fda_flav_ecig);

# Get flavored indicator by alternative
is_flavored = get_flavored_indicator(cat_idx);

# Get FDA flavored indicator by alternative
is_fda_flavored = get_fda_flavored_indicator(cat_idx);

# Orig ecig/bundle indicator: cat 2 = orig ecig, cat 5 = orig bundle (γ_2 lock-in)
is_nonflavored_ecig = [cat_idx[j] in (2, 5) for j in 1:N_J]

# Outside option indicator: cat 0
is_outside = [cat_idx[j] == 0 for j in 1:N_J]

# Cigarette quantity indicator: cat 1 = cig; cat 5, 6, 7 = bundles with a cig component
has_cig = [(cat_idx[j] == 1 || cat_idx[j] >= 5) for j in 1:N_J]

# E-cigarette quantity indicator: cat 2-7 = any alternative with an e-cig component
has_ecig = [cat_idx[j] >= 2 for j in 1:N_J]


#############################
# Demographics
#############################

# TYA states: load binary data and shift to 1-indexed (0 = no TYA → 1, 1 = TYA present → 2)
tya_state = [s + 1 for s in get_tya_states()]

# Household-level TYA share (fraction of months with TYA present) for mixture weights
tya_share_hh = get_tya_share()


#############################
# Price Space
#############################

# Get pricing grid: N_P points per category, P is N_P × 2 (cig, ecig)
N_P, P = get_pricing_spaces();

# Get all price combinations
N_Pcomb, Pcomb = get_pricing_spaces_combination(N_K, N_P, P);

# Get Halton draw price transitions: T[m, r, k] where m = price state, r = draw, k = category
T = get_transitions(N_K);

# Pre-compute bilinear interpolation brackets and weights for price transitions
p_cig_lo, p_cig_hi, p_cig_w, p_ecig_lo, p_ecig_hi, p_ecig_w = precompute_price_transitions(N_P, P, T);


#############################
# Household Price
# Trajectories
#############################

# Map observed household prices to continuous values for likelihood interpolation
# p_continuous is N × 2 (cig price, ecig price) - actual per-unit prices, not grid indices
_, p_continuous = map_prices_to_grid(N_P, P, Pcomb, N_J);

# Get period indices for time-series analysis
period_idx, period_labels, N_periods = get_period_indices();

# Log data setup completion time and sample size
N_obs = length(y)
N_HH = length(hh_ranges)
setup_elapsed = time() - t_setup;
log_msg("Data loading complete in $(round(setup_elapsed, digits=1))s")
log_msg("Observations: $N_obs, Households: $N_HH, Alternatives: $N_J, Periods: $N_periods")


#############################
# Load Estimated Parameters
#############################

# Read θ_hat from the estimates CSV produced by 02_Estimation_Mixture.jl.
if HPC
    estimates_path = "/home/u2/wbrasic/4th_Year_Paper/Dynamic_Model/02_Second_Stage_Estimation_Mixture/Dynamic_Model_Mixture_$(psi_tag)_$(beta_tag)_Results/Dynamic_Model_Mixture_$(psi_tag)_$(beta_tag)_Estimates/Dynamic_Model_Mixture_$(psi_tag)_$(beta_tag)_Estimates.csv"
else
    estimates_path = "C:/Users/wbras/OneDrive/Documents/Desktop/UA/4th_Year_Paper/4th_Year_Paper_Data/HMS/2021-Onward/Dynamic_Model/Dynamic_Model_Mixture_$(psi_tag)_$(beta_tag)_Results/Dynamic_Model_Mixture_$(psi_tag)_$(beta_tag)_Estimates/Dynamic_Model_Mixture_$(psi_tag)_$(beta_tag)_Estimates.csv"
end
df_est = CSV.read(estimates_path, DataFrame)

# Drop the NLL column if present (not a structural parameter)
if "NLL" in names(df_est)
    select!(df_est, Not(:NLL))
end

# Extract parameter names and values
param_names = names(df_est)
N_params = length(param_names)
θ_hat = Float64.(collect(df_est[1, :]))

# Print and log loaded parameters
log_msg("\nLoaded estimates from: $estimates_path")
log_msg("Parameters:")
for k in 1:N_params
    log_msg("  $(param_names[k]) = $(θ_hat[k])")
end

# Isolate structural parameters (ψ_1 is always fixed; no ESTIMATE_PSI_S)
θ_struct_all = θ_hat

# Extract parameters 
common   = θ_struct_all[1:14]   # α_C, α_E, α_E2, α_CE, λ_1, λ_2, λ_3, λ_4, γ_1, γ_2, γ_3, γ_4, ω_C, ω_E
ξ_1      = θ_struct_all[15:17]  # ξ_C_1, ξ_E_1, ξ_CE_1
ξ_2      = θ_struct_all[18:20]  # ξ_C_2, ξ_E_2, ξ_CE_2
ξ_3      = θ_struct_all[21:23]  # ξ_C_3, ξ_E_3, ξ_CE_3
π_0_2    = θ_struct_all[24]     # type 2 baseline logit intercept
π_TYA_2  = θ_struct_all[25]     # type 2 TYA share shifter
π_0_3    = θ_struct_all[26]     # type 3 baseline logit intercept
π_TYA_3  = θ_struct_all[27]     # type 3 TYA share shifter

# Override PSI_3 from estimated position 28 when ESTIMATE_PSI_3 is true
if ESTIMATE_PSI_3
    PSI_3 = θ_struct_all[28]
    log_msg(@sprintf("ψ_3 = %.6f (estimated, extracted from θ_hat[28])", PSI_3))
else
    log_msg(@sprintf("ψ_3 grid seed = %.6f (from ENV; the estimate itself is θ_hat[28])", PSI_3))
end

# Construct per-type structural parameter vectors 
θ_struct_1 = vcat(common, ξ_1)
θ_struct_2 = vcat(common, ξ_2)
θ_struct_3 = vcat(common, ξ_3)

# Log mixing weight summary (type 1 normalized)
log_msg("\nMixing weight parameters:")
log_msg(@sprintf("  π_0_2   = %.6f  π_TYA_2 = %.6f", π_0_2, π_TYA_2))
log_msg(@sprintf("  π_0_3   = %.6f  π_TYA_3 = %.6f", π_0_3, π_TYA_3))

# Compute mixing weights at tya_share = 0 and tya_share = 1 for reference 
for tya_ref in [0.0, 1.0]
    l2 = π_0_2 + π_TYA_2 * tya_ref
    l3 = π_0_3 + π_TYA_3 * tya_ref
    denom = 1.0 + exp(l2) + exp(l3)
    log_msg(@sprintf("  tya_share=%.0f: P(Type 1)=%.4f, P(Type 2)=%.4f, P(Type 3)=%.4f",
        tya_ref, 1.0/denom, exp(l2)/denom, exp(l3)/denom))
end


#############################
# Compute Household Addiction
# Trajectories
#############################

# Print and log household addiction state computation header
log_msg("\n===================================")
log_msg("Computing household addiction states...")
log_msg("===================================")

t_states = time()

# Fast stock: estimate initial stocks and simulate trajectories
af0, max_fp_iters_f = get_initial_addiction_stock(ψ_2, A_f, n, y, hh_codes)
log_msg("Fast stock initial addiction stocks: max fixed-point iterations = $max_fp_iters_f")
af_continuous = simulate_addiction_trajectories(N_A_f, ψ_2, A_f, n, y, hh_codes, af0)

# Slow stock: estimate initial stocks and simulate trajectories
as0, max_fp_iters_s = get_initial_addiction_stock(ψ_1, A_s, n, y, hh_codes)
log_msg("Slow stock initial addiction stocks: max fixed-point iterations = $max_fp_iters_s")
as_continuous = simulate_addiction_trajectories(N_A_s, ψ_1, A_s, n, y, hh_codes, as0)

# Flavored habit stock: estimate initial stocks and simulate trajectories
n_flav     = Float64.(is_flavored)             # ∈ {0.0, 1.0}
n_flav_max = 1.0                               # binary max; rescale γ_2, γ_3, γ_4 by × ψ_3
N_A_flav, A_flav = get_addiction_space(PSI_3; N_A=10)
aflav0, max_fp_iters_flav = get_initial_addiction_stock(PSI_3, A_flav, n_flav, y, hh_codes)
log_msg("Flavored habit stock initial: max fixed-point iterations = $max_fp_iters_flav")
aflav_continuous = simulate_addiction_trajectories(N_A_flav, PSI_3, A_flav, n_flav, y, hh_codes, aflav0)

# Print and log addiction state computation time
states_elapsed = time() - t_states
log_msg("Addiction states computed in $(round(states_elapsed, digits=1))s")


#############################
# Pre-compute Addiction
# Objects (Three Stocks)
#############################

# Fast stock (always pre-computed, ψ_2 is never estimated)
af_lower, af_upper, af_weight = precompute_addiction_transitions(N_J, N_A_f, ψ_2, A_f, n)

# Slow stock (pre-computed at fixed ψ_1 = 0.10)
as_lower, as_upper, as_weight = precompute_addiction_transitions(N_J, N_A_s, ψ_1, A_s, n)

# Flavored habit stock (pre-computed; ψ_3 may be estimated when ESTIMATE_PSI_3 = true)
aflav_lower, aflav_upper, aflav_weight = precompute_addiction_transitions(N_J, N_A_flav, PSI_3, A_flav, n_flav)


#############################
# Pre-compute Flow Utility
#############################

log_msg("\n===================================")
log_msg("Computing flow utility for all three types...")
log_msg("===================================")

t_flow = time()

U_1 = get_flow_utility(
    θ_struct_1, N_J, N_A_f, N_A_s, N_A_flav, N_Pcomb, A_f, A_s, A_flav,
    q_cig, q_ecig, q_bundle, is_flavored, is_fda_flavored, is_nonflavored_ecig, is_outside, cat_idx, Pcomb, has_cig, has_ecig
)
U_2 = get_flow_utility(
    θ_struct_2, N_J, N_A_f, N_A_s, N_A_flav, N_Pcomb, A_f, A_s, A_flav,
    q_cig, q_ecig, q_bundle, is_flavored, is_fda_flavored, is_nonflavored_ecig, is_outside, cat_idx, Pcomb, has_cig, has_ecig
)
U_3 = get_flow_utility(
    θ_struct_3, N_J, N_A_f, N_A_s, N_A_flav, N_Pcomb, A_f, A_s, A_flav,
    q_cig, q_ecig, q_bundle, is_flavored, is_fda_flavored, is_nonflavored_ecig, is_outside, cat_idx, Pcomb, has_cig, has_ecig
)

log_msg("Flow utility computed in $(round(time() - t_flow, digits=1))s")


#############################
# Maximum Streak Length
#############################

max_streak = 12


#############################
# Solve VFI for All Three Types
#############################

log_msg("\n===================================")
log_msg("Solving VFI at β = $β...")
log_msg("===================================")

# Solve VFI for all three types in parallel via Threads.@spawn.
# Each type has a distinct flow utility array (U_1, U_2, U_3) because the
# type-specific ξ_k parameters (baseline utilities for cig, ecig, bundle)
# enter the flow utility differently. I spawn three tasks so the VFI
# iterations run concurrently on separate threads, reducing wall time from
# 3 × (single VFI time) to roughly 1 × (single VFI time).
t_vfi = time()

vfi_task_1 = Threads.@spawn solve_vfi_sophisticated(
    N_J, N_A_f, N_A_s, N_A_flav, N_P, N_Pcomb, β, δ, U_1,
    af_lower, af_upper, af_weight,
    as_lower, as_upper, as_weight,
    aflav_lower, aflav_upper, aflav_weight,
    p_cig_lo, p_cig_hi, p_cig_w,
    p_ecig_lo, p_ecig_hi, p_ecig_w;
    V_init = nothing,
    ε = VFI_TOL,
    verbose = true
)

vfi_task_2 = Threads.@spawn solve_vfi_sophisticated(
    N_J, N_A_f, N_A_s, N_A_flav, N_P, N_Pcomb, β, δ, U_2,
    af_lower, af_upper, af_weight,
    as_lower, as_upper, as_weight,
    aflav_lower, aflav_upper, aflav_weight,
    p_cig_lo, p_cig_hi, p_cig_w,
    p_ecig_lo, p_ecig_hi, p_ecig_w;
    V_init = nothing,
    ε = VFI_TOL,
    verbose = true
)

vfi_task_3 = Threads.@spawn solve_vfi_sophisticated(
    N_J, N_A_f, N_A_s, N_A_flav, N_P, N_Pcomb, β, δ, U_3,
    af_lower, af_upper, af_weight,
    as_lower, as_upper, as_weight,
    aflav_lower, aflav_upper, aflav_weight,
    p_cig_lo, p_cig_hi, p_cig_w,
    p_ecig_lo, p_ecig_hi, p_ecig_w;
    V_init = nothing,
    ε = VFI_TOL,
    verbose = true
)

# Wait for all three VFI tasks to complete
_, V_choice_1, vfi_iters_1, vfi_converged_1 = fetch(vfi_task_1)
_, V_choice_2, vfi_iters_2, vfi_converged_2 = fetch(vfi_task_2)
_, V_choice_3, vfi_iters_3, vfi_converged_3 = fetch(vfi_task_3)

# Print and log VFI convergence result
vfi_elapsed = time() - t_vfi
log_msg("VFI: type1=$(vfi_iters_1) iters ($(vfi_converged_1)), type2=$(vfi_iters_2) iters ($(vfi_converged_2)), type3=$(vfi_iters_3) iters ($(vfi_converged_3)), $(round(vfi_elapsed, digits=1))s")


#############################
# Compute Posterior Type
# Weights
#############################

log_msg("\n===================================")
log_msg("Computing posterior type weights...")
log_msg("===================================")

t_pred = time()

# Step 1: Compute per-type choice probabilities at all OBSERVED states.
# probs_k is an (N_obs × N_J) array where:
#   - Each ROW is one household-month observation
#   - Each COLUMN is one product alternative
#   - Entry [i, j] is the softmax choice probability for alternative j at
#     observation i's state, under type k's value function V_k:
#
#       probs_k[i, j] = exp(V_k(j | state_i)) / Σ_{j'=1}^{N_J} exp(V_k(j' | state_i))
#
#     where state_i = (a_f_i, a_s_i, a_flav_i, tya_i, p_cig_i, p_ecig_i).
#
# I compute three separate arrays because each type has different ξ_k
# parameters and therefore a different V_k solution. This means the same
# observed state produces different choice probabilities under each type:
#
#       probs_1[i, j] ≠ probs_2[i, j] ≠ probs_3[i, j]  (in general)
#
# These arrays are needed ONLY for the posterior type weight computation in
# Step 2. compute_predicted_probs fills every cell (the full softmax over all
# N_J alternatives at every observation). Step 2 then reaches into each row and
# pulls out only the column for the alternative the household actually chose,
# probs_k[i, y[i]], where y[i] is the index of the observed choice at row i.
# V_k differs across types, so this extracted probability also differs:
#
#       probs_1[i, y[i]] ≠ probs_2[i, y[i]] ≠ probs_3[i, y[i]]
#
# The type-k log-likelihood for household h is the product of these extracted
# probabilities across every row i belonging to h:
#
#       L_k(y_h) = ∏_{i ∈ h} probs_k[i, y[i]]
#
# In log space (to avoid underflow):
#
#       log L_k(y_h) = Σ_{i ∈ h} log( probs_k[i, y[i]] )
#
# Step 2 then applies Bayes' rule to produce the posterior probability over
# types for household h. The full equation is:
#
#                          π_k(tya_share_h) × L_k(y_h)
#       P(k | y_h) = ─────────────────────────────────────────────────────
#                    Σ_{k'=1}^{3}  π_{k'}(tya_share_h) × L_{k'}(y_h)
#
# Mapping each term to the code:
#   π_k(tya_share_h)  →  exp(log_π_k_h), the prior mixing weight for type k,
#                         computed from the K=3 softmax on tya_share_hh[h]
#   L_k(y_h)          →  exp(ll_k), where ll_k = Σ_{i ∈ h} log(probs_k[i, y[i]])
#                         is the accumulated log-likelihood from the loop above
#   numerator term     →  a_k = log_π_k_h + ll_k  (sum of the two log quantities)
#   denominator        →  exp(logsumexp([a_1, a_2, a_3])), computed via logsumexp
#   P(k | y_h)         →  exp(a_k - logsumexp([a_1, a_2, a_3])) = hh_posterior[h, k]
#
# The denominator makes the three posteriors sum to 1 across types.
# Everything stays in log space until the final exp() call so that multiplying
# many small probabilities together never underflows to zero.
#
# Compute each type's choice probs based on OBSERVED addiction states to build the likelihood
# needed for Bayes' rule
probs_1 = compute_predicted_probs(
    V_choice_1, tya_state, af_continuous, as_continuous, aflav_continuous, p_continuous,
    N_J, N_P, A_f, A_s, A_flav, P
)
probs_2 = compute_predicted_probs(
    V_choice_2, tya_state, af_continuous, as_continuous, aflav_continuous, p_continuous,
    N_J, N_P, A_f, A_s, A_flav, P
)
probs_3 = compute_predicted_probs(
    V_choice_3, tya_state, af_continuous, as_continuous, aflav_continuous, p_continuous,
    N_J, N_P, A_f, A_s, A_flav, P
)

# Step 2: Apply Bayes' rule to compute the posterior probability that each
# household belongs to each of the K=3 types. hh_posterior is an (N_HH × 3)
# array where entry [h, k] = P(type=k | y_h), the probability that household
# h is type k given its entire observed choice sequence y_h.
#
# The forward simulation (simulate_household_sequences_mixture) uses hh_posterior
# to draw a type for each household in each of the S draws.
hh_posterior = Matrix{Float64}(undef, N_HH, 3)

for h in 1:N_HH
    start_idx, stop_idx = hh_ranges[h]

    # Compute log prior mixing weights for this household via K=3 softmax.
    # Type 1 is the base category (log-weight = 0 before normalization).
    # logit_2_h and logit_3_h are household-specific because the softmax
    # depends on the household's TYA share (fraction of months with TYA present).
    logit_2_h   = π_0_2 + π_TYA_2 * tya_share_hh[h]
    logit_3_h   = π_0_3 + π_TYA_3 * tya_share_hh[h]
    log_denom_h = logsumexp([0.0, logit_2_h, logit_3_h])  # log(1 + exp(l2) + exp(l3))
    log_π_1_h   = -log_denom_h              # log P(type=1 | tya_share_h) = log(1 / denom)
    log_π_2_h   = logit_2_h - log_denom_h   # log P(type=2 | tya_share_h) = log(exp(l2) / denom)
    log_π_3_h   = logit_3_h - log_denom_h   # log P(type=3 | tya_share_h) = log(exp(l3) / denom)

    # Accumulate the log-likelihood for each type by looping over the household's
    # rows in the probs_k arrays (start_idx:stop_idx). At each row i, probs_k[i, y[i]]
    # is the probability type k assigns to the actually chosen alternative: the
    # single number extracted from the full N_J-column softmax computed in Step 1.
    # Summing log(probs_k[i, y[i]]) across rows gives log L_k(y_h).
    ll_1 = 0.0
    ll_2 = 0.0
    ll_3 = 0.0
    for i in start_idx:stop_idx
        ll_1 += log(max(probs_1[i, y[i]], 1e-300))
        ll_2 += log(max(probs_2[i, y[i]], 1e-300))
        ll_3 += log(max(probs_3[i, y[i]], 1e-300))
    end

    # Form the log numerator for each type: a_k = log π_k_h + ll_k.
    # logsumexp([a_1, a_2, a_3]) gives the log denominator (log of the sum of
    # the three numerators). exp(a_k - log_denom_post) = P(k | y_h), stored in
    # hh_posterior[h, k], which is used by the forward simulation to draw a type.
    a_1 = log_π_1_h + ll_1
    a_2 = log_π_2_h + ll_2
    a_3 = log_π_3_h + ll_3
    log_denom_post = logsumexp([a_1, a_2, a_3])
    hh_posterior[h, 1] = exp(a_1 - log_denom_post)
    hh_posterior[h, 2] = exp(a_2 - log_denom_post)
    hh_posterior[h, 3] = exp(a_3 - log_denom_post)
end

# Print and log prediction computation time
pred_elapsed = time() - t_pred
log_msg("Posterior type weights computed in $(round(pred_elapsed, digits=1))s")

# Report posterior type distribution across households
log_msg(@sprintf("\nPosterior type weights: mean P(type=1) = %.4f, mean P(type=2) = %.4f, mean P(type=3) = %.4f",
    mean(hh_posterior[:, 1]), mean(hh_posterior[:, 2]), mean(hh_posterior[:, 3])))

# Free memory from probability matrices (no longer needed for simulation)
probs_1 = nothing
probs_2 = nothing
probs_3 = nothing
GC.gc()


#############################
# Price Elasticities
#############################

log_msg("\n===================================")
log_msg("Computing price elasticities...")
log_msg("===================================")

t_elas = time()

# 1% permanent price increase
PRICE_SHOCK = 0.01  

# For each observation i and alternative j, the mixture-integrated choice probability is:
#   P(j | i) = w1_h * probs_1[i,j] + w2_h * probs_2[i,j] + w3_h * probs_3[i,j]
# where w1_h, w2_h, w3_h are household h's posterior type weights computed above.
# Multiplying by the indicator has_cig[j] (or has_ecig[j]) and summing over j gives
# the marginal cig (or ecig) purchase probability for observation i. Averaging over
# all N observations gives the predicted aggregate market share among all households in the data.
# has_cig[j]: true if alternative j contains cigarettes (cat 1 or bundles 5-7)
# has_ecig[j]: true if alternative j contains e-cigarettes (cats 2-7)
function compute_mixture_shares(
    probs_1, probs_2, probs_3,
    hh_posterior, hh_ranges,
    has_cig, has_ecig
)
    N = size(probs_1, 1)
    s_cig = 0.0; s_ecig = 0.0
    for h in eachindex(hh_ranges)
        start_i, stop_i = hh_ranges[h]
        w1, w2, w3 = hh_posterior[h, 1], hh_posterior[h, 2], hh_posterior[h, 3]  # posterior weights
        for i in start_i:stop_i
            for j in eachindex(has_cig)
                p_ij    = w1 * probs_1[i, j] + w2 * probs_2[i, j] + w3 * probs_3[i, j]  # mixture prob
                s_cig  += has_cig[j]  * p_ij  # cig indicator × mixture prob
                s_ecig += has_ecig[j] * p_ij  # ecig indicator × mixture prob
            end
        end
    end
    return s_cig / N, s_ecig / N  # average shares across all N observations
end

# --- Baseline probs 
log_msg("  Re-computing baseline choice probabilities...")
probs_1_b = compute_predicted_probs(V_choice_1, tya_state, af_continuous, as_continuous, aflav_continuous, p_continuous, N_J, N_P, A_f, A_s, A_flav, P)
probs_2_b = compute_predicted_probs(V_choice_2, tya_state, af_continuous, as_continuous, aflav_continuous, p_continuous, N_J, N_P, A_f, A_s, A_flav, P)
probs_3_b = compute_predicted_probs(V_choice_3, tya_state, af_continuous, as_continuous, aflav_continuous, p_continuous, N_J, N_P, A_f, A_s, A_flav, P)
s_cig_b, s_ecig_b = compute_mixture_shares(probs_1_b, probs_2_b, probs_3_b, hh_posterior, hh_ranges, has_cig, has_ecig)
log_msg(@sprintf("  Baseline:  cig share = %.4f,  ecig share = %.4f", s_cig_b, s_ecig_b))

# ---------------------------------------------------------------------------
# Quantity-weighted shares (intensive and extensive margins)
#
# compute_mixture_shares gives the probability that a household-month contains
# a cig (or ecig) purchase: an extensive-margin, participation measure. A price
# change also moves households across quantity bins, and that intensive-margin
# response is lost once every purchasing alternative counts as one. The
# quantity-weighted version replaces the indicator with the alternative's
# quantity in original units, so the object averaged over observations is
# expected packs (or mL) per household-month:
#   Q_cig(i) = Σ_j P(j | i) q_cig_raw[j],   Q_ecig(i) = Σ_j P(j | i) q_ecig_raw[j]
# Both margins enter: a household can respond by buying less often or by
# buying fewer packs or mL when it does.
# ---------------------------------------------------------------------------

# Quantities in original units (packs for cigarettes, mL of e-liquid for
# e-cigarettes), undoing the max-standardization used in estimation.
q_cig_raw  = q_cig  .* q_cig_max
q_ecig_raw = q_ecig .* q_ecig_max

function compute_mixture_quantities(
    probs_1, probs_2, probs_3,
    hh_posterior, hh_ranges,
    q_cig_raw, q_ecig_raw
)
    N = size(probs_1, 1)
    Q_cig = 0.0; Q_ecig = 0.0
    for h in eachindex(hh_ranges)
        start_i, stop_i = hh_ranges[h]
        w1, w2, w3 = hh_posterior[h, 1], hh_posterior[h, 2], hh_posterior[h, 3]  # posterior weights
        for i in start_i:stop_i
            for j in eachindex(q_cig_raw)
                p_ij    = w1 * probs_1[i, j] + w2 * probs_2[i, j] + w3 * probs_3[i, j]  # mixture prob
                Q_cig  += q_cig_raw[j]  * p_ij  # packs × mixture prob
                Q_ecig += q_ecig_raw[j] * p_ij  # mL × mixture prob
            end
        end
    end
    return Q_cig / N, Q_ecig / N  # expected packs and mL per household-month
end

Q_cig_b, Q_ecig_b = compute_mixture_quantities(probs_1_b, probs_2_b, probs_3_b, hh_posterior, hh_ranges, q_cig_raw, q_ecig_raw)
log_msg(@sprintf("  Baseline:  cig quantity = %.4f packs,  ecig quantity = %.4f mL  (per household-month)", Q_cig_b, Q_ecig_b))


# ===========================================================================
# Dynamic Elasticities
#
# A static elasticity would re-evaluate softmax probabilities at a higher
# price while holding V_choice fixed. That captures only the within-period
# substitution effect. In a dynamic model, a permanent price increase also
# revises forward-looking behavior: households anticipate higher prices in
# every future period, which reduces the value of building up addiction
# stocks (since quitting becomes relatively cheaper). This feedback is only
# captured by re-solving the full Bellman equation under the shocked price
# grid. The resulting V_choice_shocked gives a long-run
# dynamic elasticity rather than a static one.
#
#
# My approach is to solve VFI on the shocked grid, evaluate at original p_continuous.
# Shifting Pcomb[:, 1] .*= (1.0 + PRICE_SHOCK):
# grid index m now represents a price with a 1% increase of its original value.
# The Bellman equation encodes optimal behavior assuming permanently higher prices
# at every grid point. V_shocked[m] is the lifetime value at grid index m in a
# world where that index is 1% more expensive than before. Evaluating at the
# original p_continuous then asks: "how does a household with observed price
# p_continuous[i] behave when expecting prices to be permanently 1% higher?"
# ===========================================================================

log_msg("  Re-solving VFI under permanently shocked price grids...")

# --- Cig price shock ---
Pcomb_cig = copy(Pcomb); Pcomb_cig[:, 1] .*= (1.0 + PRICE_SHOCK)

# Recompute flow utility for all three types under the shocked cig prices.
U_1_cg = get_flow_utility(
    θ_struct_1, N_J, N_A_f, N_A_s, N_A_flav, N_Pcomb, A_f, A_s, A_flav,
    q_cig, q_ecig, q_bundle, is_flavored, is_fda_flavored, is_nonflavored_ecig, is_outside, cat_idx, Pcomb_cig, has_cig, has_ecig
)
U_2_cg = get_flow_utility(
    θ_struct_2, N_J, N_A_f, N_A_s, N_A_flav, N_Pcomb, A_f, A_s, A_flav,
    q_cig, q_ecig, q_bundle, is_flavored, is_fda_flavored, is_nonflavored_ecig, is_outside, cat_idx, Pcomb_cig, has_cig, has_ecig
)
U_3_cg = get_flow_utility(
    θ_struct_3, N_J, N_A_f, N_A_s, N_A_flav, N_Pcomb, A_f, A_s, A_flav,
    q_cig, q_ecig, q_bundle, is_flavored, is_fda_flavored, is_nonflavored_ecig, is_outside, cat_idx, Pcomb_cig, has_cig, has_ecig
)

# Re-solve VFI for all three types concurrently.
cg_task_1 = Threads.@spawn solve_vfi_sophisticated(
    N_J, N_A_f, N_A_s, N_A_flav, N_P, N_Pcomb, β, δ, U_1_cg,
    af_lower, af_upper, af_weight, as_lower, as_upper, as_weight,
    aflav_lower, aflav_upper, aflav_weight,
    p_cig_lo, p_cig_hi, p_cig_w, p_ecig_lo, p_ecig_hi, p_ecig_w;
    V_init = nothing, ε = VFI_TOL, verbose = false
)
cg_task_2 = Threads.@spawn solve_vfi_sophisticated(
    N_J, N_A_f, N_A_s, N_A_flav, N_P, N_Pcomb, β, δ, U_2_cg,
    af_lower, af_upper, af_weight, as_lower, as_upper, as_weight,
    aflav_lower, aflav_upper, aflav_weight,
    p_cig_lo, p_cig_hi, p_cig_w, p_ecig_lo, p_ecig_hi, p_ecig_w;
    V_init = nothing, ε = VFI_TOL, verbose = false
)
cg_task_3 = Threads.@spawn solve_vfi_sophisticated(
    N_J, N_A_f, N_A_s, N_A_flav, N_P, N_Pcomb, β, δ, U_3_cg,
    af_lower, af_upper, af_weight, as_lower, as_upper, as_weight,
    aflav_lower, aflav_upper, aflav_weight,
    p_cig_lo, p_cig_hi, p_cig_w, p_ecig_lo, p_ecig_hi, p_ecig_w;
    V_init = nothing, ε = VFI_TOL, verbose = false
)

# fetch() blocks the main thread until each task finishes
_, V_cg_1, _, _ = fetch(cg_task_1)
_, V_cg_2, _, _ = fetch(cg_task_2)
_, V_cg_3, _, _ = fetch(cg_task_3)

# Evaluate the shocked V_choice_k at the ORIGINAL observed prices p_continuous.
probs_1_cg = compute_predicted_probs(V_cg_1, tya_state, af_continuous, as_continuous, aflav_continuous, p_continuous, N_J, N_P, A_f, A_s, A_flav, P)
probs_2_cg = compute_predicted_probs(V_cg_2, tya_state, af_continuous, as_continuous, aflav_continuous, p_continuous, N_J, N_P, A_f, A_s, A_flav, P)
probs_3_cg = compute_predicted_probs(V_cg_3, tya_state, af_continuous, as_continuous, aflav_continuous, p_continuous, N_J, N_P, A_f, A_s, A_flav, P)
s_cig_cg, s_ecig_cg = compute_mixture_shares(probs_1_cg, probs_2_cg, probs_3_cg, hh_posterior, hh_ranges, has_cig, has_ecig)

# Long-run arc elasticity: ε = (Δs / s_baseline) / (Δp / p) = (Δs / s_baseline) / PRICE_SHOCK
ε_cig_own_lr   = (s_cig_cg  - s_cig_b)  / s_cig_b  / PRICE_SHOCK
ε_ecig_xcig_lr = (s_ecig_cg - s_ecig_b) / s_ecig_b / PRICE_SHOCK

# Quantity-weighted elasticities (both margins) at the observed addiction stocks
Q_cig_cg, Q_ecig_cg = compute_mixture_quantities(probs_1_cg, probs_2_cg, probs_3_cg, hh_posterior, hh_ranges, q_cig_raw, q_ecig_raw)
ε_cig_own_q   = (Q_cig_cg  - Q_cig_b)  / Q_cig_b  / PRICE_SHOCK
ε_ecig_xcig_q = (Q_ecig_cg - Q_ecig_b) / Q_ecig_b / PRICE_SHOCK

# --- Ecig price shock ---
Pcomb_ecig = copy(Pcomb); Pcomb_ecig[:, 2] .*= (1.0 + PRICE_SHOCK)

U_1_eg = get_flow_utility(
    θ_struct_1, N_J, N_A_f, N_A_s, N_A_flav, N_Pcomb, A_f, A_s, A_flav,
    q_cig, q_ecig, q_bundle, is_flavored, is_fda_flavored, is_nonflavored_ecig, is_outside, cat_idx, Pcomb_ecig, has_cig, has_ecig
)
U_2_eg = get_flow_utility(
    θ_struct_2, N_J, N_A_f, N_A_s, N_A_flav, N_Pcomb, A_f, A_s, A_flav,
    q_cig, q_ecig, q_bundle, is_flavored, is_fda_flavored, is_nonflavored_ecig, is_outside, cat_idx, Pcomb_ecig, has_cig, has_ecig
)
U_3_eg = get_flow_utility(
    θ_struct_3, N_J, N_A_f, N_A_s, N_A_flav, N_Pcomb, A_f, A_s, A_flav,
    q_cig, q_ecig, q_bundle, is_flavored, is_fda_flavored, is_nonflavored_ecig, is_outside, cat_idx, Pcomb_ecig, has_cig, has_ecig
)

# Same parallel VFI pattern as cig shock; each type's U_k_eg differs via θ_struct_k.
eg_task_1 = Threads.@spawn solve_vfi_sophisticated(
    N_J, N_A_f, N_A_s, N_A_flav, N_P, N_Pcomb, β, δ, U_1_eg,
    af_lower, af_upper, af_weight, as_lower, as_upper, as_weight,
    aflav_lower, aflav_upper, aflav_weight,
    p_cig_lo, p_cig_hi, p_cig_w, p_ecig_lo, p_ecig_hi, p_ecig_w;
    V_init = nothing, ε = VFI_TOL, verbose = false
)
eg_task_2 = Threads.@spawn solve_vfi_sophisticated(
    N_J, N_A_f, N_A_s, N_A_flav, N_P, N_Pcomb, β, δ, U_2_eg,
    af_lower, af_upper, af_weight, as_lower, as_upper, as_weight,
    aflav_lower, aflav_upper, aflav_weight,
    p_cig_lo, p_cig_hi, p_cig_w, p_ecig_lo, p_ecig_hi, p_ecig_w;
    V_init = nothing, ε = VFI_TOL, verbose = false
)
eg_task_3 = Threads.@spawn solve_vfi_sophisticated(
    N_J, N_A_f, N_A_s, N_A_flav, N_P, N_Pcomb, β, δ, U_3_eg,
    af_lower, af_upper, af_weight, as_lower, as_upper, as_weight,
    aflav_lower, aflav_upper, aflav_weight,
    p_cig_lo, p_cig_hi, p_cig_w, p_ecig_lo, p_ecig_hi, p_ecig_w;
    V_init = nothing, ε = VFI_TOL, verbose = false
)

_, V_eg_1, _, _ = fetch(eg_task_1)
_, V_eg_2, _, _ = fetch(eg_task_2)
_, V_eg_3, _, _ = fetch(eg_task_3)

# Evaluate shocked V_choice_k at original p_continuous
probs_1_eg = compute_predicted_probs(V_eg_1, tya_state, af_continuous, as_continuous, aflav_continuous, p_continuous, N_J, N_P, A_f, A_s, A_flav, P)
probs_2_eg = compute_predicted_probs(V_eg_2, tya_state, af_continuous, as_continuous, aflav_continuous, p_continuous, N_J, N_P, A_f, A_s, A_flav, P)
probs_3_eg = compute_predicted_probs(V_eg_3, tya_state, af_continuous, as_continuous, aflav_continuous, p_continuous, N_J, N_P, A_f, A_s, A_flav, P)
s_cig_eg, s_ecig_eg = compute_mixture_shares(probs_1_eg, probs_2_eg, probs_3_eg, hh_posterior, hh_ranges, has_cig, has_ecig)

# Evaluate elasticities
ε_ecig_own_lr  = (s_ecig_eg - s_ecig_b) / s_ecig_b / PRICE_SHOCK
ε_cig_xecig_lr = (s_cig_eg  - s_cig_b)  / s_cig_b  / PRICE_SHOCK

# Quantity-weighted elasticities (both margins) at the observed addiction stocks
Q_cig_eg, Q_ecig_eg = compute_mixture_quantities(probs_1_eg, probs_2_eg, probs_3_eg, hh_posterior, hh_ranges, q_cig_raw, q_ecig_raw)
ε_ecig_own_q  = (Q_ecig_eg - Q_ecig_b) / Q_ecig_b / PRICE_SHOCK
ε_cig_xecig_q = (Q_cig_eg  - Q_cig_b)  / Q_cig_b  / PRICE_SHOCK


# ---------------------------------------------------------------------------
# Forward-simulated elasticities (addiction stocks respond)
#
# The elasticities above evaluate the shocked value functions at the observed
# addiction stocks, so the state is held fixed and only the choice
# probabilities move. Under a permanent price increase the stocks themselves
# fall over time as households purchase less, and lower stocks feed back into
# lower purchase probabilities through the withdrawal and lock-in terms. The
# simulation below lets that channel operate: it forward-simulates every
# household's sequence from its initial stocks under the baseline and the two
# shocked value functions, with addiction evolving from the simulated choices,
# and compares mean quantities and purchase shares across the three worlds.
# The same random numbers are used in all three worlds (common random
# numbers), so simulation noise largely cancels in the differences.
# ---------------------------------------------------------------------------

log_msg("  Forward-simulating elasticities ($S draws per price world)...")

ELAS_BASE_SEED = 22222

# Mean quantity (packs, mL) and purchase shares (cig, ecig) per household-month
# implied by one simulated choice vector.
function summarize_sim_choices(sim_y, q_cig_raw, q_ecig_raw, has_cig, has_ecig)
    N = length(sim_y)
    Q_cig = 0.0; Q_ecig = 0.0; s_cig = 0.0; s_ecig = 0.0
    for i in 1:N
        j = sim_y[i]
        Q_cig  += q_cig_raw[j];  Q_ecig += q_ecig_raw[j]
        s_cig  += has_cig[j];    s_ecig += has_ecig[j]
    end
    return (Q_cig / N, Q_ecig / N, s_cig / N, s_ecig / N)
end

# Accumulator: rows = (baseline, cig shock, ecig shock); cols = (Q_cig, Q_ecig, s_cig, s_ecig)
sim_elas_acc = zeros(Float64, 3, 4)

# Months since the price change for each observation: the shocked value functions apply
# from each household's first simulated month, so tenure within the household is the
# time elapsed since the permanent price increase. Accumulating the same four statistics
# by tenure traces the elasticity from its short-run value (stocks unchanged) to its
# long-run value (stocks adjusted).
tenure = zeros(Int, N_obs)
for h in eachindex(hh_ranges)
    start_i, stop_i = hh_ranges[h]
    for i in start_i:stop_i
        tenure[i] = period_idx[i] - period_idx[start_i] + 1
    end
end
T_ten = maximum(tenure)
n_ten = zeros(Float64, T_ten)
for i in 1:N_obs
    n_ten[tenure[i]] += 1.0
end
sim_elas_ten = zeros(Float64, 3, 4, T_ten)   # world × statistic × tenure, summed over draws and observations
sim_elas_V   = ((V_choice_1, V_choice_2, V_choice_3), (V_cg_1, V_cg_2, V_cg_3), (V_eg_1, V_eg_2, V_eg_3))
for s in 1:S
    if s == 1 || s % 25 == 0
        log_msg("  Elasticity simulation draw $s / $S...")
    end
    for w in 1:3
        rng_w   = MersenneTwister(ELAS_BASE_SEED + s)   # same seed in every world
        sim_y_w = simulate_household_sequences_mixture(
            sim_elas_V[w][1], sim_elas_V[w][2], sim_elas_V[w][3], hh_posterior, hh_ranges,
            tya_state, p_continuous, hh_codes, af0, as0, aflav0,
            n, n_flav, ψ_2, ψ_1, PSI_3, N_J, N_P, A_f, A_s, A_flav, P, rng_w
        )
        vals_w = summarize_sim_choices(sim_y_w, q_cig_raw, q_ecig_raw, has_cig, has_ecig)
        for c in 1:4
            sim_elas_acc[w, c] += vals_w[c]
        end
        for i in 1:N_obs
            j = sim_y_w[i]; τ = tenure[i]
            sim_elas_ten[w, 1, τ] += q_cig_raw[j]
            sim_elas_ten[w, 2, τ] += q_ecig_raw[j]
            sim_elas_ten[w, 3, τ] += has_cig[j]
            sim_elas_ten[w, 4, τ] += has_ecig[j]
        end
    end
end
sim_elas_acc ./= S
for τ in 1:T_ten
    sim_elas_ten[:, :, τ] ./= (S * n_ten[τ])
end

# Elasticity by months since the price change: statistic c (1 Q_cig, 2 Q_ecig, 3 s_cig, 4 s_ecig),
# shocked world w (2 cig price, 3 ecig price), relative to the baseline world 1
elas_ten(c, w) = [(sim_elas_ten[w, c, τ] - sim_elas_ten[1, c, τ]) / sim_elas_ten[1, c, τ] / PRICE_SHOCK for τ in 1:T_ten]
ε_ten_cig_own_sq   = elas_ten(1, 2); ε_ten_ecig_xcig_sq = elas_ten(2, 2)
ε_ten_ecig_own_sq  = elas_ten(2, 3); ε_ten_cig_xecig_sq = elas_ten(1, 3)
ε_ten_cig_own_ss   = elas_ten(3, 2); ε_ten_ecig_xcig_ss = elas_ten(4, 2)
ε_ten_ecig_own_ss  = elas_ten(4, 3); ε_ten_cig_xecig_ss = elas_ten(3, 3)

Q_cig_sim_b,  Q_ecig_sim_b,  s_cig_sim_b,  s_ecig_sim_b  = sim_elas_acc[1, :]
Q_cig_sim_cg, Q_ecig_sim_cg, s_cig_sim_cg, s_ecig_sim_cg = sim_elas_acc[2, :]
Q_cig_sim_eg, Q_ecig_sim_eg, s_cig_sim_eg, s_ecig_sim_eg = sim_elas_acc[3, :]
log_msg(@sprintf("  Simulated baseline:  cig quantity = %.4f packs,  ecig quantity = %.4f mL,  cig share = %.4f,  ecig share = %.4f",
    Q_cig_sim_b, Q_ecig_sim_b, s_cig_sim_b, s_ecig_sim_b))

# Forward-simulated quantity elasticities (both margins; addiction stocks respond)
ε_cig_own_sq   = (Q_cig_sim_cg  - Q_cig_sim_b)  / Q_cig_sim_b  / PRICE_SHOCK
ε_ecig_xcig_sq = (Q_ecig_sim_cg - Q_ecig_sim_b) / Q_ecig_sim_b / PRICE_SHOCK
ε_ecig_own_sq  = (Q_ecig_sim_eg - Q_ecig_sim_b) / Q_ecig_sim_b / PRICE_SHOCK
ε_cig_xecig_sq = (Q_cig_sim_eg  - Q_cig_sim_b)  / Q_cig_sim_b  / PRICE_SHOCK

# Forward-simulated participation elasticities (extensive margin; addiction stocks respond)
ε_cig_own_ss   = (s_cig_sim_cg  - s_cig_sim_b)  / s_cig_sim_b  / PRICE_SHOCK
ε_ecig_xcig_ss = (s_ecig_sim_cg - s_ecig_sim_b) / s_ecig_sim_b / PRICE_SHOCK
ε_ecig_own_ss  = (s_ecig_sim_eg - s_ecig_sim_b) / s_ecig_sim_b / PRICE_SHOCK
ε_cig_xecig_ss = (s_cig_sim_eg  - s_cig_sim_b)  / s_cig_sim_b  / PRICE_SHOCK


# --- Log results ---
elas_elapsed = time() - t_elas
log_msg(@sprintf("\nPrice elasticities computed in %.1fs", elas_elapsed))

log_msg("\n  All elasticities: permanent 1% price increase, V_choice re-solved")
log_msg("\n  --- Participation Elasticities (extensive margin; observed addiction stocks) ---")
log_msg(rpad("  ε(cig,  p_cig)",  28) * @sprintf("= %+.4f  (own-price)",   ε_cig_own_lr))
log_msg(rpad("  ε(ecig, p_cig)",  28) * @sprintf("= %+.4f  (cross-price)", ε_ecig_xcig_lr))
log_msg(rpad("  ε(ecig, p_ecig)", 28) * @sprintf("= %+.4f  (own-price)",   ε_ecig_own_lr))
log_msg(rpad("  ε(cig,  p_ecig)", 28) * @sprintf("= %+.4f  (cross-price)", ε_cig_xecig_lr))

log_msg("\n  --- Quantity-Weighted Elasticities (both margins; observed addiction stocks) ---")
log_msg(rpad("  ε(cig,  p_cig)",  28) * @sprintf("= %+.4f  (own-price)",   ε_cig_own_q))
log_msg(rpad("  ε(ecig, p_cig)",  28) * @sprintf("= %+.4f  (cross-price)", ε_ecig_xcig_q))
log_msg(rpad("  ε(ecig, p_ecig)", 28) * @sprintf("= %+.4f  (own-price)",   ε_ecig_own_q))
log_msg(rpad("  ε(cig,  p_ecig)", 28) * @sprintf("= %+.4f  (cross-price)", ε_cig_xecig_q))

log_msg("\n  --- Forward-Simulated Quantity Elasticities (both margins; addiction stocks respond) ---")
log_msg(rpad("  ε(cig,  p_cig)",  28) * @sprintf("= %+.4f  (own-price)",   ε_cig_own_sq))
log_msg(rpad("  ε(ecig, p_cig)",  28) * @sprintf("= %+.4f  (cross-price)", ε_ecig_xcig_sq))
log_msg(rpad("  ε(ecig, p_ecig)", 28) * @sprintf("= %+.4f  (own-price)",   ε_ecig_own_sq))
log_msg(rpad("  ε(cig,  p_ecig)", 28) * @sprintf("= %+.4f  (cross-price)", ε_cig_xecig_sq))

log_msg("\n  --- Forward-Simulated Participation Elasticities (extensive margin; addiction stocks respond) ---")
log_msg(rpad("  ε(cig,  p_cig)",  28) * @sprintf("= %+.4f  (own-price)",   ε_cig_own_ss))
log_msg(rpad("  ε(ecig, p_cig)",  28) * @sprintf("= %+.4f  (cross-price)", ε_ecig_xcig_ss))
log_msg(rpad("  ε(ecig, p_ecig)", 28) * @sprintf("= %+.4f  (own-price)",   ε_ecig_own_ss))
log_msg(rpad("  ε(cig,  p_ecig)", 28) * @sprintf("= %+.4f  (cross-price)", ε_cig_xecig_ss))

# --- Save to CSV ---
# Price_Elasticities.csv: mixture-weighted, one row per measure × demand × price.
#   participation           purchase incidence, observed addiction stocks (extensive margin)
#   quantity                expected packs or mL, observed addiction stocks (both margins)
#   simulated_quantity      mean packs or mL, forward simulation (both margins; stocks respond)
#   simulated_participation purchase incidence, forward simulation (stocks respond)
path_elas = joinpath(output_dir, "Price_Elasticities.csv")
open(path_elas, "w") do io
    println(io, "measure,demand,price,elasticity")
    for (meas, vals) in [
        ("participation",           (ε_cig_own_lr, ε_ecig_xcig_lr, ε_ecig_own_lr, ε_cig_xecig_lr)),
        ("quantity",                (ε_cig_own_q,  ε_ecig_xcig_q,  ε_ecig_own_q,  ε_cig_xecig_q)),
        ("simulated_quantity",      (ε_cig_own_sq, ε_ecig_xcig_sq, ε_ecig_own_sq, ε_cig_xecig_sq)),
        ("simulated_participation", (ε_cig_own_ss, ε_ecig_xcig_ss, ε_ecig_own_ss, ε_cig_xecig_ss)),
    ]
        for ((dem, pr), val) in zip([("cig", "p_cig"), ("ecig", "p_cig"), ("ecig", "p_ecig"), ("cig", "p_ecig")], vals)
            println(io, "$meas,$dem,$pr,$(@sprintf("%.8f", val))")
        end
    end
end
log_msg("Elasticities saved to: $path_elas")

for (title, e_cc, e_ec, e_ee, e_ce) in [
        ("Forward-Simulated Quantity Elasticities by Months Since the Price Change (both margins; addiction stocks respond)",
         ε_ten_cig_own_sq, ε_ten_ecig_xcig_sq, ε_ten_ecig_own_sq, ε_ten_cig_xecig_sq),
        ("Forward-Simulated Participation Elasticities by Months Since the Price Change (extensive margin; addiction stocks respond)",
         ε_ten_cig_own_ss, ε_ten_ecig_xcig_ss, ε_ten_ecig_own_ss, ε_ten_cig_xecig_ss)]
    log_msg("\n  --- $title ---")
    log_msg(rpad("  month", 8) * rpad("n_obs", 10) * rpad("ε(cig,p_cig)", 16) * rpad("ε(ecig,p_cig)", 16) * rpad("ε(ecig,p_ecig)", 16) * "ε(cig,p_ecig)")
    for τ in 1:T_ten
        log_msg(rpad("  $τ", 8) * rpad(string(Int(n_ten[τ])), 10) * rpad(@sprintf("%+.4f", e_cc[τ]), 16) * rpad(@sprintf("%+.4f", e_ec[τ]), 16) * rpad(@sprintf("%+.4f", e_ee[τ]), 16) * @sprintf("%+.4f", e_ce[τ]))
    end
end

# Price_Elasticities_By_Tenure.csv: one row per measure × demand × price × month since the
# price change, with the number of observations at that tenure and the baseline level
path_elas_ten = joinpath(output_dir, "Price_Elasticities_By_Tenure.csv")
open(path_elas_ten, "w") do io
    println(io, "measure,demand,price,tenure,n_obs,baseline,elasticity")
    for (meas, c_cig, c_ecig) in [("simulated_quantity", 1, 2), ("simulated_participation", 3, 4)]
        for (dem, pr, c, w) in [("cig", "p_cig", c_cig, 2), ("ecig", "p_cig", c_ecig, 2), ("ecig", "p_ecig", c_ecig, 3), ("cig", "p_ecig", c_cig, 3)]
            e = elas_ten(c, w)
            for τ in 1:T_ten
                println(io, "$meas,$dem,$pr,$τ,$(Int(n_ten[τ])),$(@sprintf("%.8f", sim_elas_ten[1, c, τ])),$(@sprintf("%.8f", e[τ]))")
            end
        end
    end
end
log_msg("Elasticities by tenure saved to: $path_elas_ten")

# Free the six probability matrices allocated for the elasticity exercise
probs_1_b = probs_2_b = probs_3_b = nothing
probs_1_cg = probs_2_cg = probs_3_cg = nothing
probs_1_eg = probs_2_eg = probs_3_eg = nothing
GC.gc()


#############################
# Modal Latent Type Masks
#############################

# Assign each household to its modal latent type: the argmax of the
# posterior computed above (ties go to the lower index, matching the
# counterfactual code). The masks below are observation-
# level and are used ONLY to restrict which observations are counted in the
# by-type validation statistics. The forward simulation is unchanged: each
# draw still samples every household's type from its full posterior.
#
# Type 1 households are the cigarette-persistent group, so their cigarette
# statistics are the informative ones; type 2 households are the e-cigarette-
# persistent group, so their e-cigarette statistics are the informative ones.
# Both products are computed for both masks (it is free); the figure scripts
# pick the relevant product for each type.
modal_type = [argmax(hh_posterior[h, :]) for h in 1:N_HH]

mask_type1 = falses(N_obs)
mask_type2 = falses(N_obs)
for h in 1:N_HH
    start_idx, stop_idx = hh_ranges[h]
    if modal_type[h] == 1
        mask_type1[start_idx:stop_idx] .= true
    elseif modal_type[h] == 2
        mask_type2[start_idx:stop_idx] .= true
    end
end

# label => obs_mask pairs passed to the validation functions via extra_masks
type_masks = ["Type1" => mask_type1, "Type2" => mask_type2]

log_msg(@sprintf("Modal type counts: type 1 = %d, type 2 = %d, type 3 = %d",
    count(==(1), modal_type), count(==(2), modal_type), count(==(3), modal_type)))
log_msg(@sprintf("Observation counts: type 1 = %d, type 2 = %d", count(mask_type1), count(mask_type2)))


#############################
# Run Simulation Validation
#############################

log_msg("\n===================================")
log_msg("Running $S simulation draws...")
log_msg("===================================\n")

t_sim = time()

# Run S simulation draws and accumulate streak persistence statistics.
# sim_results is a tuple containing:
#   streak_all / streak_tya:    Dict of (max_streak × 3) matrices
#     [streak_length, mean_continuation_rate, avg_N] for each product type
#   streak_all_ci / streak_tya_ci: Dict of (max_streak × 2) matrices
#     [p025, p975] confidence bands across S draws
#   streak_extra / streak_extra_ci: Dict{"Type1"/"Type2" => same layout}
# These are compared to actual_streaks_* computed from observed y below.
sim_results = run_simulation_validation(
    V_choice_1, V_choice_2, V_choice_3, hh_posterior, hh_ranges,
    tya_state, p_continuous, hh_codes, af0, as0, aflav0,
    n, n_flav, ψ_2, ψ_1, PSI_3, N_J, N_P, A_f, A_s, A_flav, P,
    cat_idx, period_idx, S;
    base_seed=12345, max_streak=max_streak, extra_masks=type_masks
)

sim_elapsed = time() - t_sim
log_msg("\nSimulation completed in $(round(sim_elapsed, digits=1))s")


#############################
# Compute Actual Streak
# Statistics by TYA Status
#############################

# Compute the empirical streak persistence curves from the observed choice
# sequence y. Comparing actual_streaks_* to
# sim_results.streak_* is the validation test.

# TYA observation mask (same mask used in run_simulation_validation above)
mask_tya    = [tya_state[i] == 2 for i in eachindex(tya_state)]

# Actual streak persistence for all households
actual_streaks_all = compute_sim_streak_continuation(
    y, cat_idx, hh_codes, period_idx;
    max_streak=max_streak
)

# Actual streak persistence for TYA households
actual_streaks_tya = compute_sim_streak_continuation(
    y, cat_idx, hh_codes, period_idx;
    obs_mask=mask_tya, max_streak=max_streak
)

# Actual streak persistence by modal type (Dict: "Type1"/"Type2" => streak Dict)
actual_streaks_type = Dict(el => compute_sim_streak_continuation(
    y, cat_idx, hh_codes, period_idx;
    obs_mask=mask_el, max_streak=max_streak
) for (el, mask_el) in type_masks)

# Ordered list of subgroups used by every print/save loop below:
# (print label, file suffix, actual, simulated, simulated CI)
streak_groups = [
    ("All",          "All",   actual_streaks_all,           sim_results.streak_all,             sim_results.streak_all_ci),
    ("TYA Present",  "TYA",   actual_streaks_tya,           sim_results.streak_tya,             sim_results.streak_tya_ci),
    ("Type 1",       "Type1", actual_streaks_type["Type1"], sim_results.streak_extra["Type1"],  sim_results.streak_extra_ci["Type1"]),
    ("Type 2",       "Type2", actual_streaks_type["Type2"], sim_results.streak_extra["Type2"],  sim_results.streak_extra_ci["Type2"])
]


#############################
# Print Streak Persistence
# Results
#############################

log_msg("\n===================================")
log_msg("Streak Persistence Curves")
log_msg("===================================")

# For each subgroup (All, TYA, modal type 1, modal type 2) and product type, print
# the empirical continuation rate (Actual) vs. the mean simulated rate
# (Simulated), their difference (sim − actual; positive = model overpredicts
# persistence), and the simulation's [2.5%, 97.5%] confidence band across S draws.
for (tya_label, _, actual_streaks, sim_streaks, sim_ci) in streak_groups

    log_msg("\n  --- $tya_label ---")

    for (label, full_label) in [("cig", "Cigarettes"), ("ecig", "E-Cigarettes"), ("flav_ecig", "Flavored E-Cigarettes"), ("orig_ecig", "Original E-Cigarettes")]

        # actual_res: max_streak × 3; col 1 = streak length k, col 2 = empirical
        #             continuation rate, col 3 = N (observations at streak length k)
        # sim_res:    max_streak × 3; col 2 = mean simulated continuation rate
        # ci_res:     max_streak × 2; col 1 = p025, col 2 = p975 across S draws
        actual_res = actual_streaks[label]
        sim_res    = sim_streaks[label]
        ci_res     = sim_ci[label]

        log_msg("\n  $full_label ($tya_label):")
        log_msg(@sprintf("  %-8s  %8s  %10s  %10s  %10s  %10s  %10s", "Streak", "N", "Actual", "Simulated", "Difference", "CI_025", "CI_975"))
        log_msg("  " * repeat("-", 79))

        for k in 1:size(actual_res, 1)
            streak_k = Int(actual_res[k, 1])
            n_k      = Int(actual_res[k, 3])
            actual_k = actual_res[k, 2]
            sim_k    = sim_res[k, 2]
            ci_lo    = ci_res[k, 1]
            ci_hi    = ci_res[k, 2]

            if n_k > 0
                diff_k = isnan(sim_k) ? NaN : sim_k - actual_k
                # The last streak row gets a "+" suffix (e.g. "10+") indicating
                # streaks of that length or longer are pooled in one row.
                k_label = streak_k == size(actual_res, 1) ? "$(streak_k)+" : string(streak_k)
                # sim_k is NaN when no simulated sequences reached streak k;
                # print "N/A" rather than a numeric value in that case.
                if isnan(diff_k)
                    log_msg(@sprintf("  %-8s  %8d  %10.4f  %10s  %10s  %10s  %10s", k_label, n_k, actual_k, "N/A", "N/A", "N/A", "N/A"))
                else
                    ci_lo_str = isnan(ci_lo) ? "N/A" : @sprintf("%.4f", ci_lo)
                    ci_hi_str = isnan(ci_hi) ? "N/A" : @sprintf("%.4f", ci_hi)
                    log_msg(@sprintf("  %-8s  %8d  %10.4f  %10.4f  %+10.4f  %10s  %10s", k_label, n_k, actual_k, sim_k, diff_k, ci_lo_str, ci_hi_str))
                end
            end
        end
    end
end


#############################
# Save Results
#############################

# Write Streak_Persistence_{All,TYA,Type1,Type2}.csv. The Type1 file's
# cigarette rows and the Type2 file's e-cigarette rows are the ones the paper
# uses; the other product rows are written for completeness.
for (tya_label, tya_suffix, actual_streaks, sim_streaks, sim_ci) in streak_groups

    path_streak = joinpath(output_dir, "Streak_Persistence_$(tya_suffix).csv")
    open(path_streak, "w") do io
        println(io, join(["product", "streak_length", "N", "actual_rate", "simulated_rate", "ci_025", "ci_975"], ","))
        for (label, full_label) in [("cig", "Cigarettes"), ("ecig", "E-Cigarettes"), ("flav_ecig", "Flavored E-Cigarettes"), ("orig_ecig", "Original E-Cigarettes")]
            actual_res = actual_streaks[label]
            sim_res = sim_streaks[label]
            ci_res  = sim_ci[label]
            for k in 1:size(actual_res, 1)
                n_k = Int(actual_res[k, 3])
                if n_k > 0
                    println(io, join([full_label,
                        string(Int(actual_res[k, 1])),
                        string(n_k),
                        @sprintf("%.10f", actual_res[k, 2]),
                        @sprintf("%.10f", sim_res[k, 2]),
                        @sprintf("%.10f", ci_res[k, 1]),
                        @sprintf("%.10f", ci_res[k, 2])], ","))
                end
            end
        end
    end
end

# Save posterior type distribution per household (K=3: columns for all three types)
path_posterior = joinpath(output_dir, "Posterior_Type_Distribution.csv")
open(path_posterior, "w") do io
    println(io, "household_idx,posterior_type1,posterior_type2,posterior_type3,tya_share")
    for h in 1:N_HH
        println(io, join([string(h),
            @sprintf("%.10f", hh_posterior[h, 1]),
            @sprintf("%.10f", hh_posterior[h, 2]),
            @sprintf("%.10f", hh_posterior[h, 3]),
            @sprintf("%.10f", tya_share_hh[h])], ","))
    end
end

log_msg("\nStreak results saved to: $output_dir")


#############################
# Quantity Escalation
#############################

# Quantities in original units (packs for cigarettes, mL of e-liquid for
# e-cigarettes), undoing the max-standardization used in estimation.
q_cig_raw  = q_cig  .* q_cig_max
q_ecig_raw = q_ecig .* q_ecig_max


# Streak definition: mirrors compute_sim_streak_continuation exactly.
# Cig streak uses has_cig (cat 1 and 5,6,7); ecig streak uses has_ecig
# (cat 2,3,4,5,6,7). Bundles are included because addiction builds from any
# cig-containing (or ecig-containing) purchase, regardless of bundling.
# A third track restricts the e-cigarette streak to flavored alternatives
# (is_flavored: cat 3,4,6,7), so a streak counts consecutive months with a
# flavored purchase and the quantity is the mL of e-liquid in those months.
#
# Runs inside an S-draw loop
log_msg("\n===================================")
log_msg("Quantity Escalation ($S draws)...")
log_msg("===================================\n")

t_qty = time()

# --- Group definitions (All / TYA present / modal type 1 / modal type 2) ---
# The Type1 cigarette file and the Type2 e-cigarette file are the ones the paper
# uses; the cross products are written for completeness.
qty_group_labels   = ["All", "TYA Present", "Type 1", "Type 2"]
qty_group_suffixes = ["All", "TYA", "Type1", "Type2"]
qty_group_masks    = [trues(N_obs), mask_tya, mask_type1, mask_type2]
N_qty_groups       = length(qty_group_labels)

# --- Empirical streak lengths from observed y ---
# One pass over all observations to build per-observation streak lengths for
# cig and ecig. New streak resets to 1 at the first period of a household
# (i == 1 or household/period boundary) or after a non-purchasing period.
streak_cig_data  = zeros(Int, N_obs)
streak_ecig_data = zeros(Int, N_obs)
streak_flav_data = zeros(Int, N_obs)
for i in 1:N_obs
    if !has_cig[y[i]]
        streak_cig_data[i] = 0
    elseif i == 1 || hh_codes[i] != hh_codes[i-1] || period_idx[i] != period_idx[i-1] + 1
        streak_cig_data[i] = 1
    else
        streak_cig_data[i] = has_cig[y[i-1]] ? streak_cig_data[i-1] + 1 : 1
    end
    if !has_ecig[y[i]]
        streak_ecig_data[i] = 0
    elseif i == 1 || hh_codes[i] != hh_codes[i-1] || period_idx[i] != period_idx[i-1] + 1
        streak_ecig_data[i] = 1
    else
        streak_ecig_data[i] = has_ecig[y[i-1]] ? streak_ecig_data[i-1] + 1 : 1
    end
    if !is_flavored[y[i]]
        streak_flav_data[i] = 0
    elseif i == 1 || hh_codes[i] != hh_codes[i-1] || period_idx[i] != period_idx[i-1] + 1
        streak_flav_data[i] = 1
    else
        streak_flav_data[i] = is_flavored[y[i-1]] ? streak_flav_data[i-1] + 1 : 1
    end
end

# --- Empirical mean quantity by streak length, by TYA group ---
mean_cig_qty_data  = [zeros(Float64, max_streak) for _ in 1:N_qty_groups]
mean_ecig_qty_data = [zeros(Float64, max_streak) for _ in 1:N_qty_groups]
cig_qty_n_data     = [zeros(Float64, max_streak) for _ in 1:N_qty_groups]
ecig_qty_n_data    = [zeros(Float64, max_streak) for _ in 1:N_qty_groups]
mean_flav_qty_data = [zeros(Float64, max_streak) for _ in 1:N_qty_groups]
flav_qty_n_data    = [zeros(Float64, max_streak) for _ in 1:N_qty_groups]
for g in 1:N_qty_groups
    mask = qty_group_masks[g]
    for i in 1:N_obs
        !mask[i] && continue
        j  = y[i]
        kc = min(streak_cig_data[i],  max_streak)
        if kc >= 1 && has_cig[j]
            mean_cig_qty_data[g][kc]  += q_cig_raw[j]
            cig_qty_n_data[g][kc]     += 1.0
        end
        ke = min(streak_ecig_data[i], max_streak)
        if ke >= 1 && has_ecig[j]
            mean_ecig_qty_data[g][ke] += q_ecig_raw[j]
            ecig_qty_n_data[g][ke]    += 1.0
        end
        kf = min(streak_flav_data[i], max_streak)
        if kf >= 1 && is_flavored[j]
            mean_flav_qty_data[g][kf] += q_ecig_raw[j]
            flav_qty_n_data[g][kf]    += 1.0
        end
    end
    # Convert accumulated sums to means
    for k in 1:max_streak
        mean_cig_qty_data[g][k]  = cig_qty_n_data[g][k]  > 0 ? mean_cig_qty_data[g][k]  / cig_qty_n_data[g][k]  : NaN
        mean_ecig_qty_data[g][k] = ecig_qty_n_data[g][k] > 0 ? mean_ecig_qty_data[g][k] / ecig_qty_n_data[g][k] : NaN
        mean_flav_qty_data[g][k] = flav_qty_n_data[g][k] > 0 ? mean_flav_qty_data[g][k] / flav_qty_n_data[g][k] : NaN
    end
end

# --- Median quantity by streak length ---
# The mean is a noisy statistic for e-liquid because the quantity bins are wide
# (a single top-bin purchase moves a thin cell's mean by several mL). The median
# is not. It is computed from a tally of purchases by alternative j at
# each streak length: tally[k, j] = number of qualifying purchases of alternative
# j at streak length k. The median is then the weighted median of q_cig_raw[j] (or
# q_ecig_raw[j]) with those counts as weights. The same tallies are written to CSV
# so the full simulated vs. observed bin distribution can be inspected.

# Weighted median of `qvals` with non-negative integer-valued `counts`.
# Uses the midpoint of the two middle order statistics for even totals.
function weighted_median(counts::AbstractVector{<:Real}, qvals::AbstractVector{<:Real})
    total = sum(counts)
    total <= 0 && return NaN
    ord = sortperm(qvals)
    n   = Int(round(total))
    # positions of the middle order statistic(s), 1-indexed
    p_lo = (n + 1) ÷ 2
    p_hi = (n + 2) ÷ 2
    cum = 0.0; v_lo = NaN; v_hi = NaN
    for idx in ord
        c = counts[idx]
        c <= 0 && continue
        cum += c
        if isnan(v_lo) && cum >= p_lo
            v_lo = qvals[idx]
        end
        if cum >= p_hi
            v_hi = qvals[idx]
            break
        end
    end
    return (v_lo + v_hi) / 2
end

# Empirical tallies (max_streak × N_J) and medians, by group
cig_qty_tally_data  = [zeros(Float64, max_streak, N_J) for _ in 1:N_qty_groups]
ecig_qty_tally_data = [zeros(Float64, max_streak, N_J) for _ in 1:N_qty_groups]
flav_qty_tally_data = [zeros(Float64, max_streak, N_J) for _ in 1:N_qty_groups]
for g in 1:N_qty_groups
    mask = qty_group_masks[g]
    for i in 1:N_obs
        !mask[i] && continue
        j  = y[i]
        kc = min(streak_cig_data[i],  max_streak)
        if kc >= 1 && has_cig[j]
            cig_qty_tally_data[g][kc, j] += 1.0
        end
        ke = min(streak_ecig_data[i], max_streak)
        if ke >= 1 && has_ecig[j]
            ecig_qty_tally_data[g][ke, j] += 1.0
        end
        kf = min(streak_flav_data[i], max_streak)
        if kf >= 1 && is_flavored[j]
            flav_qty_tally_data[g][kf, j] += 1.0
        end
    end
end
med_cig_qty_data  = [[weighted_median(view(cig_qty_tally_data[g],  k, :), q_cig_raw)  for k in 1:max_streak] for g in 1:N_qty_groups]
med_ecig_qty_data = [[weighted_median(view(ecig_qty_tally_data[g], k, :), q_ecig_raw) for k in 1:max_streak] for g in 1:N_qty_groups]
med_flav_qty_data = [[weighted_median(view(flav_qty_tally_data[g], k, :), q_ecig_raw) for k in 1:max_streak] for g in 1:N_qty_groups]

# --- Simulation draw accumulators ---
cig_qty_draws         = [fill(NaN, max_streak, S)  for _ in 1:N_qty_groups]
ecig_qty_draws        = [fill(NaN, max_streak, S)  for _ in 1:N_qty_groups]
acc_cig_qty_sum        = [zeros(Float64, max_streak) for _ in 1:N_qty_groups]
acc_cig_qty_n          = [zeros(Float64, max_streak) for _ in 1:N_qty_groups]
acc_ecig_qty_sum       = [zeros(Float64, max_streak) for _ in 1:N_qty_groups]
acc_ecig_qty_n         = [zeros(Float64, max_streak) for _ in 1:N_qty_groups]
flav_qty_draws         = [fill(NaN, max_streak, S)  for _ in 1:N_qty_groups]
acc_flav_qty_sum       = [zeros(Float64, max_streak) for _ in 1:N_qty_groups]
acc_flav_qty_n         = [zeros(Float64, max_streak) for _ in 1:N_qty_groups]

# Median accumulators: per-draw medians (for CI bands) and pooled tallies across
# draws (for the point estimate, mirroring how the mean pools across draws)
cig_med_draws          = [fill(NaN, max_streak, S)  for _ in 1:N_qty_groups]
ecig_med_draws         = [fill(NaN, max_streak, S)  for _ in 1:N_qty_groups]
flav_med_draws         = [fill(NaN, max_streak, S)  for _ in 1:N_qty_groups]
acc_cig_qty_tally      = [zeros(Float64, max_streak, N_J) for _ in 1:N_qty_groups]
acc_ecig_qty_tally     = [zeros(Float64, max_streak, N_J) for _ in 1:N_qty_groups]
acc_flav_qty_tally     = [zeros(Float64, max_streak, N_J) for _ in 1:N_qty_groups]

QUANT_BASE_SEED = 11111

for s in 1:S
    if s == 1 || s % 25 == 0
        log_msg("  Quantity validation draw $s / $S...")
    end

    # Simulate one complete forward sequence for all N_obs observations.
    # Addiction stocks evolve endogenously from simulated choices.
    rng     = MersenneTwister(QUANT_BASE_SEED + s)
    sim_y_q = simulate_household_sequences_mixture(
        V_choice_1, V_choice_2, V_choice_3, hh_posterior, hh_ranges,
        tya_state, p_continuous, hh_codes, af0, as0, aflav0,
        n, n_flav, ψ_2, ψ_1, PSI_3, N_J, N_P, A_f, A_s, A_flav, P, rng
    )

    # Simulated streak lengths (identical logic to empirical computation above)
    streak_cig_s  = zeros(Int, N_obs)
    streak_ecig_s = zeros(Int, N_obs)
    streak_flav_s = zeros(Int, N_obs)
    for i in 1:N_obs
        j = sim_y_q[i]
        if !has_cig[j]
            streak_cig_s[i] = 0
        elseif i == 1 || hh_codes[i] != hh_codes[i-1] || period_idx[i] != period_idx[i-1] + 1
            streak_cig_s[i] = 1
        else
            streak_cig_s[i] = has_cig[sim_y_q[i-1]] ? streak_cig_s[i-1] + 1 : 1
        end
        if !has_ecig[j]
            streak_ecig_s[i] = 0
        elseif i == 1 || hh_codes[i] != hh_codes[i-1] || period_idx[i] != period_idx[i-1] + 1
            streak_ecig_s[i] = 1
        else
            streak_ecig_s[i] = has_ecig[sim_y_q[i-1]] ? streak_ecig_s[i-1] + 1 : 1
        end
        if !is_flavored[j]
            streak_flav_s[i] = 0
        elseif i == 1 || hh_codes[i] != hh_codes[i-1] || period_idx[i] != period_idx[i-1] + 1
            streak_flav_s[i] = 1
        else
            streak_flav_s[i] = is_flavored[sim_y_q[i-1]] ? streak_flav_s[i-1] + 1 : 1
        end
    end

    # Accumulate quantity escalation for each TYA group in a single pass over observations
    for g in 1:N_qty_groups
        mask = qty_group_masks[g]
        qty_cig_sum  = zeros(Float64, max_streak)
        qty_cig_n    = zeros(Float64, max_streak)
        qty_ecig_sum = zeros(Float64, max_streak)
        qty_ecig_n   = zeros(Float64, max_streak)
        qty_flav_sum = zeros(Float64, max_streak)
        qty_flav_n   = zeros(Float64, max_streak)
        tally_cig_s  = zeros(Float64, max_streak, N_J)
        tally_ecig_s = zeros(Float64, max_streak, N_J)
        tally_flav_s = zeros(Float64, max_streak, N_J)
        for i in 1:N_obs
            !mask[i] && continue
            j  = sim_y_q[i]
            # Quantity escalation: accumulate q at each streak length
            kc = min(streak_cig_s[i],  max_streak)
            if kc >= 1 && has_cig[j]
                qty_cig_sum[kc] += q_cig_raw[j];   qty_cig_n[kc]  += 1.0
                tally_cig_s[kc, j] += 1.0
            end
            ke = min(streak_ecig_s[i], max_streak)
            if ke >= 1 && has_ecig[j]
                qty_ecig_sum[ke] += q_ecig_raw[j]; qty_ecig_n[ke] += 1.0
                tally_ecig_s[ke, j] += 1.0
            end
            kf = min(streak_flav_s[i], max_streak)
            if kf >= 1 && is_flavored[j]
                qty_flav_sum[kf] += q_ecig_raw[j]; qty_flav_n[kf] += 1.0
                tally_flav_s[kf, j] += 1.0
            end
        end
        for k in 1:max_streak
            cig_qty_draws[g][k, s]  = qty_cig_n[k]  > 0 ? qty_cig_sum[k]  / qty_cig_n[k]  : NaN
            ecig_qty_draws[g][k, s] = qty_ecig_n[k] > 0 ? qty_ecig_sum[k] / qty_ecig_n[k] : NaN
            acc_cig_qty_sum[g][k]   += qty_cig_sum[k];   acc_cig_qty_n[g][k]  += qty_cig_n[k]
            acc_ecig_qty_sum[g][k]  += qty_ecig_sum[k];  acc_ecig_qty_n[g][k] += qty_ecig_n[k]
            flav_qty_draws[g][k, s] = qty_flav_n[k] > 0 ? qty_flav_sum[k] / qty_flav_n[k] : NaN
            acc_flav_qty_sum[g][k]  += qty_flav_sum[k];  acc_flav_qty_n[g][k] += qty_flav_n[k]
            # Per-draw medians from this draw's tallies
            cig_med_draws[g][k, s]  = weighted_median(view(tally_cig_s,  k, :), q_cig_raw)
            ecig_med_draws[g][k, s] = weighted_median(view(tally_ecig_s, k, :), q_ecig_raw)
            flav_med_draws[g][k, s] = weighted_median(view(tally_flav_s, k, :), q_ecig_raw)
        end
        acc_cig_qty_tally[g]  .+= tally_cig_s
        acc_ecig_qty_tally[g] .+= tally_ecig_s
        acc_flav_qty_tally[g] .+= tally_flav_s
    end
end

qty_elapsed = time() - t_qty
log_msg("\nQuantity validation completed in $(round(qty_elapsed, digits=1))s")

# Quantity escalation point estimates: pooled ratio, i.e. (total quantity across all
# S draws) / (total qualifying household-months across all S draws).
# Pooling weights each draw by its own event count (a draw with 500
# qualifying household-months at streak k contributes more than one with 5), and a
# draw with zero events at streak k contributes 0 to both sums rather than producing
# an undefined per-draw ratio that has to be dropped.
mean_cig_qty_sim    = [[acc_cig_qty_n[g][k]  > 0 ? acc_cig_qty_sum[g][k]  / acc_cig_qty_n[g][k]  : NaN for k in 1:max_streak] for g in 1:N_qty_groups]
mean_ecig_qty_sim   = [[acc_ecig_qty_n[g][k] > 0 ? acc_ecig_qty_sum[g][k] / acc_ecig_qty_n[g][k] : NaN for k in 1:max_streak] for g in 1:N_qty_groups]
mean_flav_qty_sim   = [[acc_flav_qty_n[g][k] > 0 ? acc_flav_qty_sum[g][k] / acc_flav_qty_n[g][k] : NaN for k in 1:max_streak] for g in 1:N_qty_groups]

# [2.5%, 97.5%] CI bands across S draws
cig_qty_ci  = [hcat(
    [let v = filter(!isnan, vec(cig_qty_draws[g][k, :]));    isempty(v) ? NaN : quantile(v, 0.025) end for k in 1:max_streak],
    [let v = filter(!isnan, vec(cig_qty_draws[g][k, :]));    isempty(v) ? NaN : quantile(v, 0.975) end for k in 1:max_streak]
) for g in 1:N_qty_groups]
ecig_qty_ci = [hcat(
    [let v = filter(!isnan, vec(ecig_qty_draws[g][k, :]));   isempty(v) ? NaN : quantile(v, 0.025) end for k in 1:max_streak],
    [let v = filter(!isnan, vec(ecig_qty_draws[g][k, :]));   isempty(v) ? NaN : quantile(v, 0.975) end for k in 1:max_streak]
) for g in 1:N_qty_groups]
flav_qty_ci = [hcat(
    [let v = filter(!isnan, vec(flav_qty_draws[g][k, :]));   isempty(v) ? NaN : quantile(v, 0.025) end for k in 1:max_streak],
    [let v = filter(!isnan, vec(flav_qty_draws[g][k, :]));   isempty(v) ? NaN : quantile(v, 0.975) end for k in 1:max_streak]
) for g in 1:N_qty_groups]

# Median point estimates: weighted median of the tallies pooled across all S draws
med_cig_qty_sim  = [[weighted_median(view(acc_cig_qty_tally[g],  k, :), q_cig_raw)  for k in 1:max_streak] for g in 1:N_qty_groups]
med_ecig_qty_sim = [[weighted_median(view(acc_ecig_qty_tally[g], k, :), q_ecig_raw) for k in 1:max_streak] for g in 1:N_qty_groups]
med_flav_qty_sim = [[weighted_median(view(acc_flav_qty_tally[g], k, :), q_ecig_raw) for k in 1:max_streak] for g in 1:N_qty_groups]

# [2.5%, 97.5%] bands for the median across S draws
cig_med_ci  = [hcat(
    [let v = filter(!isnan, vec(cig_med_draws[g][k, :]));    isempty(v) ? NaN : quantile(v, 0.025) end for k in 1:max_streak],
    [let v = filter(!isnan, vec(cig_med_draws[g][k, :]));    isempty(v) ? NaN : quantile(v, 0.975) end for k in 1:max_streak]
) for g in 1:N_qty_groups]
ecig_med_ci = [hcat(
    [let v = filter(!isnan, vec(ecig_med_draws[g][k, :]));   isempty(v) ? NaN : quantile(v, 0.025) end for k in 1:max_streak],
    [let v = filter(!isnan, vec(ecig_med_draws[g][k, :]));   isempty(v) ? NaN : quantile(v, 0.975) end for k in 1:max_streak]
) for g in 1:N_qty_groups]
flav_med_ci = [hcat(
    [let v = filter(!isnan, vec(flav_med_draws[g][k, :]));   isempty(v) ? NaN : quantile(v, 0.025) end for k in 1:max_streak],
    [let v = filter(!isnan, vec(flav_med_draws[g][k, :]));   isempty(v) ? NaN : quantile(v, 0.975) end for k in 1:max_streak]
) for g in 1:N_qty_groups]


#############################
# Print Quantity Escalation
# Results
#############################

log_msg("\n===================================")
log_msg("Quantity Escalation by Streak Length")
log_msg("===================================")

# For each TYA group and product, print mean quantity in original units (packs or mL) at each
# streak length k. A rising Data Mean with k confirms that households shift
# toward higher-quantity bins as addiction builds. The Simulated column shows
# whether the model captures that slope; Difference = sim − data.
for (tya_label, g) in zip(qty_group_labels, 1:N_qty_groups)
    log_msg("\n  --- $tya_label ---")

    # Print one product's table: mean columns followed by median columns
    function print_qty_table(label, n_data, mean_data, mean_sim, mean_ci, med_data, med_sim, med_ci)
        log_msg("\n  $label:")
        log_msg(@sprintf("  %-8s  %8s  %10s  %10s  %10s  %10s  %10s  |  %10s  %10s  %10s  %10s",
            "Streak", "N_data", "Data Mean", "Sim Mean", "Difference", "CI_025", "CI_975",
            "Data Med", "Sim Med", "MedCI_025", "MedCI_975"))
        log_msg("  " * repeat("-", 127))
        fmt(x) = isnan(x) ? "N/A" : @sprintf("%.4f", x)
        for k in 1:max_streak
            n_k  = Int(n_data[k])
            act  = mean_data[k]
            (n_k == 0 || isnan(act)) && continue
            sim   = mean_sim[k]
            diff  = isnan(sim) ? NaN : sim - act
            k_lbl = k == max_streak ? "$(k)+" : string(k)
            diff_str = isnan(diff) ? "N/A" : @sprintf("%+.4f", diff)
            log_msg(@sprintf("  %-8s  %8d  %10.4f  %10s  %10s  %10s  %10s  |  %10s  %10s  %10s  %10s",
                k_lbl, n_k, act, fmt(sim), diff_str, fmt(mean_ci[k, 1]), fmt(mean_ci[k, 2]),
                fmt(med_data[k]), fmt(med_sim[k]), fmt(med_ci[k, 1]), fmt(med_ci[k, 2])))
        end
    end

    print_qty_table("Cigarettes",   cig_qty_n_data[g],  mean_cig_qty_data[g],  mean_cig_qty_sim[g],  cig_qty_ci[g],
                    med_cig_qty_data[g],  med_cig_qty_sim[g],  cig_med_ci[g])
    print_qty_table("E-Cigarettes", ecig_qty_n_data[g], mean_ecig_qty_data[g], mean_ecig_qty_sim[g], ecig_qty_ci[g],
                    med_ecig_qty_data[g], med_ecig_qty_sim[g], ecig_med_ci[g])
    print_qty_table("Flavored E-Cigarettes", flav_qty_n_data[g], mean_flav_qty_data[g], mean_flav_qty_sim[g], flav_qty_ci[g],
                    med_flav_qty_data[g], med_flav_qty_sim[g], flav_med_ci[g])
end


#############################
# Save Quantity Escalation Results
#############################

# Writes 3 × 4 = 12 CSV files (three products × four subgroups):
#   Quantity_Escalation_Cig_{All,TYA,Type1,Type2}.csv
#   Quantity_Escalation_Ecig_{All,TYA,Type1,Type2}.csv
#   Quantity_Escalation_FlavEcig_{All,TYA,Type1,Type2}.csv   (flavored e-cigarette streaks only)
# Columns: the original mean columns, followed by median columns
# (data_median_qty, sim_median_qty, med_ci_025, med_ci_975). Existing
# readers that select the mean columns by name are unaffected.
# Also writes the purchase tallies by quantity bin at each streak length:
#   Quantity_Bin_Distribution_{Cig,Ecig,FlavEcig}_{All,TYA,Type1,Type2}.csv
# with one row per (streak_length, alternative j): observed count and the
# simulated count averaged over draws, so the full simulated vs. observed bin
# distribution can be compared.
function write_qty_csv(path, n_data, mean_data, mean_sim, mean_ci, med_data, med_sim, med_ci)
    open(path, "w") do io
        println(io, join(["streak_length", "n_data", "data_mean_qty", "sim_mean_qty", "ci_025", "ci_975",
                          "data_median_qty", "sim_median_qty", "med_ci_025", "med_ci_975"], ","))
        for k in 1:max_streak
            n_k = Int(n_data[k])
            n_k == 0 && continue
            println(io, join([
                k == max_streak ? "$(k)+" : string(k),
                string(n_k),
                @sprintf("%.10f", mean_data[k]),
                @sprintf("%.10f", mean_sim[k]),
                @sprintf("%.10f", mean_ci[k, 1]),
                @sprintf("%.10f", mean_ci[k, 2]),
                @sprintf("%.10f", med_data[k]),
                @sprintf("%.10f", med_sim[k]),
                @sprintf("%.10f", med_ci[k, 1]),
                @sprintf("%.10f", med_ci[k, 2])
            ], ","))
        end
    end
end

function write_bin_csv(path, tally_data, tally_sim_acc, qvals)
    open(path, "w") do io
        println(io, join(["streak_length", "alternative", "qty", "data_count", "sim_count_per_draw"], ","))
        for k in 1:max_streak, j in 1:N_J
            (tally_data[k, j] == 0 && tally_sim_acc[k, j] == 0) && continue
            println(io, join([
                k == max_streak ? "$(k)+" : string(k),
                string(j),
                @sprintf("%.10f", qvals[j]),
                string(Int(tally_data[k, j])),
                @sprintf("%.6f", tally_sim_acc[k, j] / S)
            ], ","))
        end
    end
end

for (tya_suffix, g) in zip(qty_group_suffixes, 1:N_qty_groups)
    write_qty_csv(joinpath(output_dir, "Quantity_Escalation_Cig_$(tya_suffix).csv"),
                  cig_qty_n_data[g],  mean_cig_qty_data[g],  mean_cig_qty_sim[g],  cig_qty_ci[g],
                  med_cig_qty_data[g],  med_cig_qty_sim[g],  cig_med_ci[g])
    write_qty_csv(joinpath(output_dir, "Quantity_Escalation_Ecig_$(tya_suffix).csv"),
                  ecig_qty_n_data[g], mean_ecig_qty_data[g], mean_ecig_qty_sim[g], ecig_qty_ci[g],
                  med_ecig_qty_data[g], med_ecig_qty_sim[g], ecig_med_ci[g])
    write_bin_csv(joinpath(output_dir, "Quantity_Bin_Distribution_Cig_$(tya_suffix).csv"),
                  cig_qty_tally_data[g],  acc_cig_qty_tally[g],  q_cig_raw)
    write_bin_csv(joinpath(output_dir, "Quantity_Bin_Distribution_Ecig_$(tya_suffix).csv"),
                  ecig_qty_tally_data[g], acc_ecig_qty_tally[g], q_ecig_raw)
    write_qty_csv(joinpath(output_dir, "Quantity_Escalation_FlavEcig_$(tya_suffix).csv"),
                  flav_qty_n_data[g], mean_flav_qty_data[g], mean_flav_qty_sim[g], flav_qty_ci[g],
                  med_flav_qty_data[g], med_flav_qty_sim[g], flav_med_ci[g])
    write_bin_csv(joinpath(output_dir, "Quantity_Bin_Distribution_FlavEcig_$(tya_suffix).csv"),
                  flav_qty_tally_data[g], acc_flav_qty_tally[g], q_ecig_raw)
end

log_msg("\nQuantity escalation results saved to: $output_dir")


#############################
# Log Final Timing
#############################

# Print and log final timing and completion message
total_elapsed = time() - t_setup
log_msg("\n===================================")
log_msg("Simulation-based mixture model validation complete")
log_msg(@sprintf("Total time: %.1fs", total_elapsed))
log_msg("===================================")
log_msg("Finished at $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")

# Close the log file handle
close(log_io)
