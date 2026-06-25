################################################################################
# William Brasic
# The University of Arizona
# wbrasic97@gmail.com
# June 2026
#
# This script validates the K=3 mixture dynamic model (holdout sample) using
# simulation-based methods. Instead of comparing observed choices to predicted
# choice probabilities, it simulates S forward choice sequences from the
# estimated model, letting the addiction stocks (fast, slow, and flavored
# habit) evolve endogenously based on simulated (not observed) choices.
#
# Observed prices and TYA states are used at each period. Only the addiction
# stocks are simulated forward.
#
# WHY SIMULATION RATHER THAN PREDICTED PROBABILITIES?
# Evaluating predicted probabilities at observed states would only confirm
# in-sample fit at the exact states where the model was estimated. Because
# estimation maximizes the likelihood at those states, matching probabilities
# there is a necessary but weak criterion. The stricter test is to ask whether
# the model's own dynamics are internally consistent. In the forward simulation,
# the addiction stock at month t is determined by what the model itself chose in
# months 1 through t-1, not by what the household actually chose. If the model
# slightly overestimates purchase rates, stocks compound upward, which further
# raises purchase probabilities in subsequent periods. A misspecified addiction
# process amplifies this feedback over the horizon. The streak persistence curves
# and post-purchase destination paths test exactly this property: if the model
# correctly captures both the flow utility and the law of motion, simulated
# statistics conditional on addiction state should match observed patterns without
# being given the observed stocks to lean on.
#
# This holdout version is an even stricter test than the in-sample version.
# The households here were withheld from estimation entirely, so the model
# cannot have overfit their specific purchase histories. Matching holdout streak
# persistence curves and post-purchase paths confirms genuine out-of-sample
# predictive validity, not just internal consistency on the training data.
#
# Validation exercises:
#   1. Price elasticities: long-run (VFI re-solved under shocked prices)
#      for cigarettes and e-cigarettes, own- and cross-price
#   2. Streak persistence curves for cig and ecig, separately by TYA status
#   3. Post-purchase destination paths after a flavored e-cig purchase
#   4. Post-purchase destination paths after a cigarette purchase
#
# Results are saved to a log file and CSV files in the Results directory.
# Set ESTIMATE_PSI_3=true and/or BETA=0.7 etc. at Slurm submission time.
################################################################################


#############################
# Preliminaries
#############################

# Present-bias β (read from ENV to match estimation run; never estimated)
BETA = parse(Float64, get(ENV, "BETA", "1.0"))

# Whether ψ_3 was estimated jointly as the 27th parameter (must match the estimation run)
ESTIMATE_PSI_3 = parse(Bool, get(ENV, "ESTIMATE_PSI_3", "false"))

# Fixed flavored habit decay rate (ignored when ESTIMATE_PSI_3=true; θ_hat[27] is used instead)
PSI_3 = parse(Float64, get(ENV, "PSI_3", "0.75"))

# Fixed parameters (ψ_1, ψ_2 always fixed; β fixed per run but can vary across runs)
ψ_1 = 0.10
ψ_2 = 0.90
β   = BETA

# VFI convergence tolerance (sup-norm); use 1e-6 for accuracy in simulation
VFI_TOL = 1e-6

# Number of simulation draws
S = 100

# Detect whether we are running on the HPC (any non-Windows system)
HPC = !Sys.iswindows()

# Load all functions and packages from the mixture functions file
if HPC
    include("../02_Second_Stage_Estimation_V2_Holdout/01_Functions_Mixture.jl")
else
    include("../02_Second_Stage_Estimation_Holdout_Mixture/01_Functions_Mixture.jl")
end

# Additional imports for simulation (must come before include to provide AbstractRNG)
using Random

# Load simulation validation functions
include("01_Validation_Functions_Mixture.jl")


# Construct psi, beta, and psi_3 tags for output directory and file naming.
# These must match the tags used by 02_Estimation_Mixture.jl to locate estimates.
psi_tag   = "Psi2_09_Psi1_01"
beta_tag  = "Beta_$(numeric_tag(BETA))"
psi_3_tag = ESTIMATE_PSI_3 ? "Psi3_Est" : "Psi3_$(numeric_tag(PSI_3))"

# File paths if on HPC or not
if HPC

    # Output path for results (use absolute path so it's unaffected by later cd)
    output_dir = abspath("./Validation_Mixture_Holdout_V2_$(psi_tag)_$(beta_tag)_$(psi_3_tag)_Results")
    mkpath(output_dir)

    # Set working directory to where the 2021-2022 estimation data CSVs live
    cd("/home/u2/wbrasic/4th_Year_Paper/Data_Holdout")

    # 2023 holdout validation data directory (loaded after posteriors are computed)
    dir_val = "/home/u2/wbrasic/4th_Year_Paper/Data_Holdout_Val"
else

    # Output path for results (local Windows path)
    output_dir = "C:/Users/wbras/OneDrive/Documents/Desktop/UA/4th_Year_Paper/4th_Year_Paper_Data/HMS/2021-Onward/Dynamic_Model/Validation_Mixture_Holdout_V2_$(psi_tag)_$(beta_tag)_$(psi_3_tag)_Results"
    mkpath(output_dir)

    # Set working directory to where the 2021-2022 estimation data CSVs live
    cd("C:/Users/wbras/OneDrive/Documents/Desktop/UA/4th_Year_Paper/4th_Year_Paper_Data/HMS/2021-Onward/Dynamic_Model/Data_Holdout")

    # 2023 holdout validation data directory (loaded after posteriors are computed)
    dir_val = "C:/Users/wbras/OneDrive/Documents/Desktop/UA/4th_Year_Paper/4th_Year_Paper_Data/HMS/2021-Onward/Dynamic_Model/Data_Holdout_Val"
end

# Set log file path
log_path = joinpath(output_dir, "Validation_Mixture_Log.txt")

# Open log file for writing (log_io is defined as a global in 01_Functions.jl)
log_io = open(log_path, "w")

# Print and log the start time and number of Julia threads available for VFI parallelization
log_msg("Simulation-based K=3 mixture model validation ($(psi_tag)_$(beta_tag)_$(psi_3_tag)) started at $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
log_msg("")
log_msg("Number of threads: $(Threads.nthreads())")
log_msg("Number of simulation draws: $S")
log_msg("ESTIMATE_PSI_3 = $ESTIMATE_PSI_3")
log_msg("PSI_3          = $PSI_3" * (ESTIMATE_PSI_3 ? " (will be overridden by estimate)" : " (fixed)"))
log_msg("")

# Get household identifiers (pre-loaded to avoid repeated CSV reads in objective)
hh_codes = get_hh_codes();

# Pre-compute contiguous household index ranges for mixture log-likelihood
# (called once; result is used to compute posterior type weights)
hh_ranges = precompute_hh_ranges(hh_codes);


#############################
# Initialize Fixed Parameters
#############################

# Load fixed parameters: only δ (the per-period discount factor) is extracted
# here. get_fixed_parameters() returns (ψ_1, ψ_2, β, δ), but ψ_1, ψ_2, and β
# are already set from the hardcoded block above and from the ENV-read BETA,
# so we discard the first three return values with underscore placeholders.
_, _, _, δ = get_fixed_parameters();


#############################
# State Spaces and Choices
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
# Returns 6 matrices (M × R): lo/hi grid indices and weights for each category
p_cig_lo, p_cig_hi, p_cig_w, p_ecig_lo, p_ecig_hi, p_ecig_w = precompute_price_transitions(N_P, P, T);


#############################
# Household Price
# Trajectories
#############################

# Map observed household prices to continuous values for likelihood interpolation
# p_continuous is N × 2 (cig price, ecig price) - actual per-unit prices, not grid indices
# P_obs_cig / P_obs_ecig (N × N_J bin-specific prices) are not used in simulation
_, p_continuous, _, _ = map_prices_to_grid(N_P, P, Pcomb, N_J);

# Get period indices for time-series analysis
period_idx, period_labels, N_periods = get_period_indices();

# Log data setup completion time and sample size
N_obs = length(y)
N_HH = length(hh_ranges)
setup_elapsed = time() - t_setup;
log_msg("2021-2022 estimation data loaded in $(round(setup_elapsed, digits=1))s")
log_msg("Estimation sample: $N_obs observations, $N_HH households, $N_J alternatives, $N_periods periods")


#############################
# Load Estimated Parameters
#############################

# Read θ̂ from the holdout estimates CSV.
# HPC: nested _Results/_Estimates/ subdirectory in 02_Second_Stage_Estimation_V2_Holdout; local: flat _Results/ folder.
# Uses absolute path because cd() has already changed the working directory to Data_Holdout/.
if HPC
    estimates_path = "/home/u2/wbrasic/4th_Year_Paper/Dynamic_Model/02_Second_Stage_Estimation_V2_Holdout/Dynamic_Model_Mixture_Holdout_V2_$(psi_tag)_$(beta_tag)_$(psi_3_tag)_Results/Dynamic_Model_Mixture_Holdout_V2_$(psi_tag)_$(beta_tag)_$(psi_3_tag)_Estimates/Dynamic_Model_Mixture_Holdout_V2_$(psi_tag)_$(beta_tag)_$(psi_3_tag)_Estimates.csv"
else
    estimates_path = "C:/Users/wbras/OneDrive/Documents/Desktop/UA/4th_Year_Paper/4th_Year_Paper_Data/HMS/2021-Onward/Dynamic_Model/Dynamic_Model_Mixture_Holdout_V2_$(psi_tag)_$(beta_tag)_$(psi_3_tag)_Results/Dynamic_Model_Mixture_Holdout_V2_$(psi_tag)_$(beta_tag)_$(psi_3_tag)_Estimates/Dynamic_Model_Mixture_Holdout_V2_$(psi_tag)_$(beta_tag)_$(psi_3_tag)_Estimates.csv"
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

# Extract K=3 mixture parameters from positions 1-26 (or 1-27 when ESTIMATE_PSI_3)
# 13 common params: α_C, α_E, α_CE, λ_1, λ_2, λ_3, λ_4, γ_1, γ_2, γ_3, γ_4, ω_C, ω_E
common   = θ_struct_all[1:13]   # α_C, α_E, α_CE, λ_1, λ_2, λ_3, λ_4, γ_1, γ_2, γ_3, γ_4, ω_C, ω_E
ξ_1      = θ_struct_all[14:16]  # ξ_C_1, ξ_E_1, ξ_CE_1
ξ_2      = θ_struct_all[17:19]  # ξ_C_2, ξ_E_2, ξ_CE_2
ξ_3      = θ_struct_all[20:22]  # ξ_C_3, ξ_E_3, ξ_CE_3
π_0_2    = θ_struct_all[23]     # type 2 baseline logit intercept
π_TYA_2  = θ_struct_all[24]     # type 2 TYA share shifter
π_0_3    = θ_struct_all[25]     # type 3 baseline logit intercept
π_TYA_3  = θ_struct_all[26]     # type 3 TYA share shifter

# Override PSI_3 from estimated position 27 when ESTIMATE_PSI_3 is true
if ESTIMATE_PSI_3
    PSI_3 = θ_struct_all[27]
    log_msg(@sprintf("ψ_3 = %.6f (estimated, extracted from θ_hat[27])", PSI_3))
else
    log_msg(@sprintf("ψ_3 = %.6f (fixed, from ENV)", PSI_3))
end

# Construct per-type structural parameter vectors (16 elements each: 13 common + 3 ξ_k)
θ_struct_1 = vcat(common, ξ_1)
θ_struct_2 = vcat(common, ξ_2)
θ_struct_3 = vcat(common, ξ_3)

# Log mixing weight summary (K=3 softmax, type 1 normalized)
log_msg("\nMixing weight parameters:")
log_msg(@sprintf("  π_0_2   = %.6f  π_TYA_2 = %.6f", π_0_2, π_TYA_2))
log_msg(@sprintf("  π_0_3   = %.6f  π_TYA_3 = %.6f", π_0_3, π_TYA_3))

# Compute mixing weights at tya_share = 0 and tya_share = 1 for reference (K=3 softmax)
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
# Law of motion: ã_flav' = (1-ψ_3)·ã_flav + ψ_3·𝟙[flavored[j]]
# Binary indicator - habit builds by one unit any time a flavored alternative is chosen.
n_flav     = Float64.(is_flavored)             # ∈ {0.0, 1.0}
n_flav_max = 1.0                               # binary max; rescale γ_2, γ_3, γ_4 by × ψ_3
N_A_flav, A_flav = get_addiction_space(PSI_3; N_A=10)
aflav0, max_fp_iters_flav = get_initial_addiction_stock(PSI_3, A_flav, n_flav, y, hh_codes)
log_msg("Flavored habit stock initial: max fixed-point iterations = $max_fp_iters_flav")
aflav_continuous = simulate_addiction_trajectories(N_A_flav, PSI_3, A_flav, n_flav, y, hh_codes, aflav0)

# Print and log addiction state computation time
states_elapsed = time() - t_states
log_msg("Addiction states computed in $(round(states_elapsed, digits=1))s")

# Extract terminal 2022 addiction stocks: the last observation of each household's
# 2021-2022 sequence. These become the starting stocks for the 2023 simulation.
af_terminal    = [af_continuous[hh_ranges[h][2]]    for h in 1:N_HH]
as_terminal    = [as_continuous[hh_ranges[h][2]]    for h in 1:N_HH]
aflav_terminal = [aflav_continuous[hh_ranges[h][2]] for h in 1:N_HH]

# Map household code → estimation household index (for aligning with 2023 data)
est_hh_code_to_idx = Dict(hh_codes[hh_ranges[h][1]] => h for h in 1:N_HH)


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
# enter the flow utility differently. We spawn three tasks so the VFI
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
# We compute three separate arrays because each type has different ξ_k
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
# Because V_k differs across types, this extracted probability also differs:
#
#       probs_1[i, y[i]] ≠ probs_2[i, y[i]] ≠ probs_3[i, y[i]]
#
# The type-k likelihood for household h is the product of these extracted
# probabilities across every row i belonging to h:
#
#       L_k(y_h) = ∏_{i ∈ h} probs_k[i, y[i]]
#
# In log space (to avoid underflow when the panel is long):
#
#       log L_k(y_h) = Σ_{i ∈ h} log( probs_k[i, y[i]] )
#
# Step 2 then applies Bayes' rule to produce the posterior probability over
# types for household h. The full equation with the normalizing denominator is:
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
# The denominator ensures the three posteriors sum to 1 across k = 1, 2, 3.
# Everything stays in log space until the final exp() call so that multiplying
# many small probabilities together never underflows to zero.
#
# The forward simulation (simulate_household_sequences_mixture) does NOT use
# these arrays. It calls interpolate_v_choice directly at the evolving simulated
# addiction state, so probs at observed states are never needed there.
# These arrays are freed after the posterior is computed (see probs_1 = nothing
# below) to reclaim memory before the S-draw simulation loop.
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
# This is NOT the forward simulation. The probs_k arrays computed in Step 1
# are evaluated at the observed addiction states. The posterior here tells us
# how much weight to place on each type when we later simulate forward. The
# forward simulation (simulate_household_sequences_mixture) uses hh_posterior
# to draw a type for each household in each of the S draws.
#
# The full Bayes' rule equation (from the Step 1 comment) is:
#
#                          π_k(tya_share_h) × L_k(y_h)
#       P(k | y_h) = ─────────────────────────────────────────────────────
#                    Σ_{k'=1}^{3}  π_{k'}(tya_share_h) × L_{k'}(y_h)
#
# Everything is computed in log space to avoid underflow. The per-type
# log-likelihood ll_k = Σ_{i ∈ h} log(probs_k[i, y[i]]) is the log of
# L_k(y_h). Adding log_π_k_h gives the log numerator a_k = log π_k + ll_k.
# logsumexp([a_1, a_2, a_3]) gives the log denominator. exp(a_k - log_denom)
# recovers P(k | y_h) without ever forming the raw products L_k(y_h).
#
# The max(..., 1e-300) guard in the ll_k loop prevents log(0) = -Inf when a
# type assigns essentially zero probability to an observed choice. Without it,
# a single near-impossible choice would make the entire log-likelihood -Inf,
# collapsing the posterior to 0/0. 1e-300 is negligible for any type that
# actually fits the data.
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

PRICE_SHOCK = 0.01  # 1% permanent price increase

# compute_mixture_shares is defined locally here rather than in
# 01_Validation_Functions_Mixture.jl because it is only used for the price
# elasticity exercise. The streak persistence and post-purchase path exercises
# work with simulated choice sequences, not with predicted probability matrices.
# The elasticity exercise is the only place we need mixture-weighted aggregate
# shares evaluated at observed states under both baseline and price-shocked
# V_choice solutions, so a local definition avoids cluttering the shared
# functions file with exercise-specific code.
#
# For each observation i and alternative j, the mixture-integrated choice probability is:
#   P(j | i) = w1_h * probs_1[i,j] + w2_h * probs_2[i,j] + w3_h * probs_3[i,j]
# where w1_h, w2_h, w3_h are household h's posterior type weights.
# Multiplying by the indicator has_cig[j] (or has_ecig[j]) and summing over j gives
# the marginal cig (or ecig) purchase probability for observation i. Averaging over
# all N observations gives the predicted aggregate market share.
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

# --- Baseline probs (freed after posterior weights; recomputed here) ---
log_msg("  Re-computing baseline choice probabilities...")
probs_1_b = compute_predicted_probs(V_choice_1, tya_state, af_continuous, as_continuous, aflav_continuous, p_continuous, N_J, N_P, A_f, A_s, A_flav, P)
probs_2_b = compute_predicted_probs(V_choice_2, tya_state, af_continuous, as_continuous, aflav_continuous, p_continuous, N_J, N_P, A_f, A_s, A_flav, P)
probs_3_b = compute_predicted_probs(V_choice_3, tya_state, af_continuous, as_continuous, aflav_continuous, p_continuous, N_J, N_P, A_f, A_s, A_flav, P)
s_cig_b, s_ecig_b = compute_mixture_shares(probs_1_b, probs_2_b, probs_3_b, hh_posterior, hh_ranges, has_cig, has_ecig)
log_msg(@sprintf("  Baseline:  cig share = %.4f,  ecig share = %.4f", s_cig_b, s_ecig_b))


# ===========================================================================
# Dynamic Elasticities
#
# WHY RE-SOLVE VFI?
# A static elasticity would re-evaluate softmax probabilities at a higher
# price while holding V_choice fixed. That captures only the within-period
# substitution effect. In a dynamic model, a permanent price increase also
# revises forward-looking behavior: households anticipate higher prices in
# every future period, which reduces the option value of building up addiction
# stocks (since quitting becomes relatively cheaper). This feedback is only
# captured by re-solving the full Bellman equation under the shocked price
# grid. The resulting V_choice_shocked reflects both the direct (within-period)
# price increase AND the reduced option value of addiction, giving a long-run
# dynamic elasticity rather than a static one.
#
# WHY EVALUATE AT THE ORIGINAL OBSERVED PRICES p_continuous?
# Keep two things separate: what the VFI was solved OVER, and what price we
# EVALUATE the resulting V_choice at.
#
# Our approach: solve VFI on the shocked grid, evaluate at original p_continuous.
# Shifting Pcomb[:, 1] .*= (1.0 + PRICE_SHOCK) changes what the VFI "believes":
# grid index m now represents economic price (1.0 + PRICE_SHOCK) × P_cig[m].
# The Bellman equation encodes optimal behavior assuming permanently higher prices
# at every grid point. V_shocked[m] is the lifetime value at grid index m in a
# world where that index is 1% more expensive than before. Evaluating at the
# original p_continuous then asks: "how does a household with observed price
# p_continuous[i] behave when expecting prices to be permanently 1% higher?"
# That is exactly the long-run dynamic counterfactual we want.
#
# The opposite approach would be: solve VFI on the original grid, evaluate at
# (1.0 + PRICE_SHOCK) × p_continuous. This only captures the static
# (within-period) substitution effect: households see a higher price today
# and substitute, but the value function still assumes the original price
# distribution in all future periods. They never re-optimize their forward-
# looking addiction trajectory. That is equivalent to a static elasticity.
#
# Passing (1.0 + PRICE_SHOCK) × p_continuous WITH the shocked V_choice
# would compound the shock twice: once inside V_shocked (via Pcomb) and once
# more at the evaluation price, producing an effective ~2% price increase.
# ===========================================================================

log_msg("  Re-solving VFI under permanently shocked price grids...")

# --- Cig price shock ---
# copy(Pcomb) creates an independent copy so the original Pcomb is not modified.
# Only column 1 (cig price) is scaled up; column 2 (ecig price) remains unchanged.
Pcomb_cig = copy(Pcomb); Pcomb_cig[:, 1] .*= (1.0 + PRICE_SHOCK)

# Recompute flow utility for all three types under the shocked cig prices.
# Price enters utility as ω_C × P_cig[m] × q_cig_std[j] at each grid state m,
# so the shifted Pcomb_cig propagates permanently higher cig costs into every
# state's instantaneous payoff. Each type uses its own θ_struct_k (different
# ξ_k parameters) but the same Pcomb_cig, producing three separate U_k_cg arrays.
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

# Re-solve VFI for all three types concurrently. Each type k has different ξ_k
# parameters (via θ_struct_k), making the three Bellman problems independent.
# Spawning all three with Threads.@spawn reduces wall time to ~1× a single VFI
# solve instead of ~3×.
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

# fetch() blocks the main thread until each task finishes and returns the result
# tuple. We extract only V_choice (second element), which is the shocked value
# function used to re-evaluate choice probabilities below.
_, V_cg_1, _, _ = fetch(cg_task_1)
_, V_cg_2, _, _ = fetch(cg_task_2)
_, V_cg_3, _, _ = fetch(cg_task_3)

# Evaluate the shocked V_choice_k at the ORIGINAL observed prices p_continuous.
# See "WHY EVALUATE AT THE ORIGINAL OBSERVED PRICES" in the header above.
probs_1_cg = compute_predicted_probs(V_cg_1, tya_state, af_continuous, as_continuous, aflav_continuous, p_continuous, N_J, N_P, A_f, A_s, A_flav, P)
probs_2_cg = compute_predicted_probs(V_cg_2, tya_state, af_continuous, as_continuous, aflav_continuous, p_continuous, N_J, N_P, A_f, A_s, A_flav, P)
probs_3_cg = compute_predicted_probs(V_cg_3, tya_state, af_continuous, as_continuous, aflav_continuous, p_continuous, N_J, N_P, A_f, A_s, A_flav, P)
s_cig_cg, s_ecig_cg = compute_mixture_shares(probs_1_cg, probs_2_cg, probs_3_cg, hh_posterior, hh_ranges, has_cig, has_ecig)

# Long-run arc elasticity: ε = (Δs / s_baseline) / (Δp / p) = (Δs / s_baseline) / PRICE_SHOCK
# ε_cig_own_lr  : own-price elasticity of cig demand    wrt cig price   (expected sign: < 0)
# ε_ecig_xcig_lr: cross-price elasticity of ecig demand  wrt cig price   (expected sign: > 0, substitutes)
ε_cig_own_lr   = (s_cig_cg  - s_cig_b)  / s_cig_b  / PRICE_SHOCK
ε_ecig_xcig_lr = (s_ecig_cg - s_ecig_b) / s_ecig_b / PRICE_SHOCK

# --- Ecig price shock ---
# Identical structure to the cig shock above. Only column 2 (ecig price) is
# shifted; column 1 (cig price) remains unchanged.
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

# Evaluate shocked V_choice_k at original p_continuous (same logic as cig shock).
probs_1_eg = compute_predicted_probs(V_eg_1, tya_state, af_continuous, as_continuous, aflav_continuous, p_continuous, N_J, N_P, A_f, A_s, A_flav, P)
probs_2_eg = compute_predicted_probs(V_eg_2, tya_state, af_continuous, as_continuous, aflav_continuous, p_continuous, N_J, N_P, A_f, A_s, A_flav, P)
probs_3_eg = compute_predicted_probs(V_eg_3, tya_state, af_continuous, as_continuous, aflav_continuous, p_continuous, N_J, N_P, A_f, A_s, A_flav, P)
s_cig_eg, s_ecig_eg = compute_mixture_shares(probs_1_eg, probs_2_eg, probs_3_eg, hh_posterior, hh_ranges, has_cig, has_ecig)

# ε_ecig_own_lr  : own-price elasticity of ecig demand    wrt ecig price  (expected sign: < 0)
# ε_cig_xecig_lr : cross-price elasticity of cig demand    wrt ecig price  (expected sign: > 0, substitutes)
ε_ecig_own_lr  = (s_ecig_eg - s_ecig_b) / s_ecig_b / PRICE_SHOCK
ε_cig_xecig_lr = (s_cig_eg  - s_cig_b)  / s_cig_b  / PRICE_SHOCK


# --- Log results ---
elas_elapsed = time() - t_elas
log_msg(@sprintf("\nPrice elasticities computed in %.1fs", elas_elapsed))

log_msg("\n  --- Dynamic Price Elasticities (permanent shock; V_choice re-solved) ---")
log_msg(rpad("  ε(cig,  p_cig)",  28) * @sprintf("= %+.4f  (own-price)",   ε_cig_own_lr))
log_msg(rpad("  ε(ecig, p_cig)",  28) * @sprintf("= %+.4f  (cross-price)", ε_ecig_xcig_lr))
log_msg(rpad("  ε(ecig, p_ecig)", 28) * @sprintf("= %+.4f  (own-price)",   ε_ecig_own_lr))
log_msg(rpad("  ε(cig,  p_ecig)", 28) * @sprintf("= %+.4f  (cross-price)", ε_cig_xecig_lr))

# --- Save to CSV ---
path_elas = joinpath(output_dir, "Price_Elasticities.csv")
open(path_elas, "w") do io
    println(io, "demand,price,elasticity")
    for (dem, pr, val) in [
        ("cig",  "p_cig",  ε_cig_own_lr),
        ("ecig", "p_cig",  ε_ecig_xcig_lr),
        ("ecig", "p_ecig", ε_ecig_own_lr),
        ("cig",  "p_ecig", ε_cig_xecig_lr),
    ]
        println(io, "$dem,$pr,$(@sprintf("%.8f", val))")
    end
end
log_msg("Elasticities saved to: $path_elas")

# Free the six probability matrices allocated for the elasticity exercise
# (baseline + cig-shocked + ecig-shocked, three types each). GC.gc() immediately
# reclaims this memory before the S-draw simulation loop begins.
probs_1_b = probs_2_b = probs_3_b = nothing
probs_1_cg = probs_2_cg = probs_3_cg = nothing
probs_1_eg = probs_2_eg = probs_3_eg = nothing
GC.gc()


#############################
# Load 2023 Validation Data
#############################

# Switch working directory to the 2023 holdout data.
# Household_Codes, Product_Choices, Category_Choices, TYA_States, and Prices
# are all 2023-specific. The alternative-level files (Consumption_Spaces,
# Nicotine_Spaces, Pricing_Spaces, Halton draws) were copied from Data_Holdout/
# unchanged and are not reloaded here — they are already in memory.
log_msg("\n===================================")
log_msg("Loading 2023 holdout validation data from Data_Holdout_Val/...")
log_msg("===================================")

cd(dir_val)

# 2023 observation structure
hh_codes  = get_hh_codes()
hh_ranges = precompute_hh_ranges(hh_codes)
N_HH      = length(hh_ranges)

# 2023 observed choices
_, _, J_val = get_product_choices()
y           = get_hh_choices(J_val)
N_obs       = length(y)

# 2023 TYA states (same 1-indexing convention: no-TYA=1, TYA=2)
tya_state = [s + 1 for s in get_tya_states()]

# 2023 observed prices (mapped to the same price grid estimated on 2021-2022 data)
_, p_continuous, _, _ = map_prices_to_grid(N_P, P, Pcomb, N_J)

# 2023 period indices (used for streak boundary detection in simulation comparisons)
period_idx, period_labels, N_periods = get_period_indices()

log_msg("2023 validation data loaded: $N_obs observations, $N_HH households, $N_periods periods")

# Build Dict{household_code, terminal_stock} for the 2023 validation sample.
# The simulation functions (simulate_household_sequences_mixture and its callers)
# type af0/as0/aflav0 as AbstractDict and access stocks via af0[hh_code], so
# these must be Dicts keyed by household code rather than position-indexed vectors.
# We look up each 2023 household's code, find its 2021-2022 position in
# est_hh_code_to_idx, and pull the corresponding terminal stock.
af_val    = Dict{eltype(hh_codes), Float64}()
as_val    = Dict{eltype(hh_codes), Float64}()
aflav_val = Dict{eltype(hh_codes), Float64}()
for h_val in 1:N_HH
    code  = hh_codes[hh_ranges[h_val][1]]
    h_est = est_hh_code_to_idx[code]
    af_val[code]    = af_terminal[h_est]
    as_val[code]    = as_terminal[h_est]
    aflav_val[code] = aflav_terminal[h_est]
end
log_msg("Terminal 2022 addiction stocks mapped to 2023 households.")
log_msg(@sprintf("  Mean af_terminal: %.6f   Mean as_terminal: %.6f   Mean aflav_terminal: %.6f",
    mean(values(af_val)), mean(values(as_val)), mean(values(aflav_val))))

# Remap posterior type weights to the 2023 household ordering.
# hh_posterior (N_HH_est × 3) rows correspond to the 2021-2022 all-HH estimation
# ordering. After the data switch, hh_ranges has N_HH rows in the 2023 overlap-HH
# ordering. Build hh_posterior_val and tya_share_val aligned with the 2023 ordering
# so simulation calls receive posteriors for the correct household at each row.
hh_posterior_val = Matrix{Float64}(undef, N_HH, 3)
tya_share_val    = Vector{Float64}(undef, N_HH)
for h_val in 1:N_HH
    code  = hh_codes[hh_ranges[h_val][1]]
    h_est = est_hh_code_to_idx[code]
    hh_posterior_val[h_val, :] = hh_posterior[h_est, :]
    tya_share_val[h_val]       = tya_share_hh[h_est]
end
log_msg("Posterior type weights remapped to 2023 household ordering.")


#############################
# Run Simulation Validation
#############################

log_msg("\n===================================")
log_msg("Running $S simulation draws...")
log_msg("===================================\n")

t_sim = time()

# Run S simulation draws and accumulate streak persistence statistics.
# sim_results is a NamedTuple containing:
#   streak_all / streak_tya / streak_no_tya:    Dict of (max_streak × 3) matrices
#     [streak_length, mean_continuation_rate, avg_N] for each product type
#   streak_all_ci / streak_tya_ci / streak_no_tya_ci: Dict of (max_streak × 2) matrices
#     [p025, p975] confidence bands across S draws
# These are compared to actual_streaks_* computed from observed y below.
sim_results = run_simulation_validation(
    V_choice_1, V_choice_2, V_choice_3, hh_posterior_val, hh_ranges,
    tya_state, p_continuous, hh_codes, af_val, as_val, aflav_val,
    n, n_flav, ψ_2, ψ_1, PSI_3, N_J, N_P, A_f, A_s, A_flav, P,
    cat_idx, period_idx, S;
    base_seed=12345, max_streak=max_streak
)

sim_elapsed = time() - t_sim
log_msg("\nSimulation completed in $(round(sim_elapsed, digits=1))s")


#############################
# Run Post-Purchase Path
# Validation
#############################

log_msg("\n===================================")
log_msg("Running post-purchase path validation ($S draws)...")
log_msg("===================================\n")

t_ppp = time()

# Source categories: flavored e-cig purchases (cat 3, 4, 6, 7)
source_cats_flav = (3, 4, 6, 7)

# Simulated post-purchase paths (averaged over S draws)
ppp_sim_results = run_post_purchase_path_validation(
    V_choice_1, V_choice_2, V_choice_3, hh_posterior_val, hh_ranges,
    tya_state, p_continuous, hh_codes, af_val, as_val, aflav_val,
    n, n_flav, ψ_2, ψ_1, PSI_3, N_J, N_P, A_f, A_s, A_flav, P,
    cat_idx, period_idx, source_cats_flav, S;
    base_seed=12345, max_horizon=6
)

# Observed post-purchase paths (from actual choice data)
# TYA observation masks (reused below for streaks as well)
mask_tya_ppp    = [tya_state[i] == 2 for i in eachindex(tya_state)]
mask_no_tya_ppp = [tya_state[i] == 1 for i in eachindex(tya_state)]

actual_ppp_all, actual_ppp_n_all = compute_post_purchase_paths(
    y, cat_idx, hh_codes, period_idx, source_cats_flav;
    max_horizon=6
)

actual_ppp_tya, actual_ppp_n_tya = compute_post_purchase_paths(
    y, cat_idx, hh_codes, period_idx, source_cats_flav;
    obs_mask=mask_tya_ppp, max_horizon=6
)

actual_ppp_no_tya, actual_ppp_n_no_tya = compute_post_purchase_paths(
    y, cat_idx, hh_codes, period_idx, source_cats_flav;
    obs_mask=mask_no_tya_ppp, max_horizon=6
)

ppp_flav_elapsed = time() - t_ppp
log_msg("\nPost-purchase path validation (flavored) completed in $(round(ppp_flav_elapsed, digits=1))s")
log_msg("  Events (observed): All=$actual_ppp_n_all, TYA=$actual_ppp_n_tya, No TYA=$actual_ppp_n_no_tya")
log_msg("  Events (sim draw 1): All=$(ppp_sim_results.n_events_all), TYA=$(ppp_sim_results.n_events_tya), No TYA=$(ppp_sim_results.n_events_no_tya)")


#############################
# Run Post-Purchase Path
# Validation (Cigarettes)
#############################

log_msg("\n===================================")
log_msg("Running post-purchase path validation - cigarette purchases ($S draws)...")
log_msg("===================================\n")

t_ppp_cig = time()

# Source categories: cigarette purchases (cat 1)
source_cats_cig = (1,)

# Simulated post-purchase paths (averaged over S draws)
ppp_cig_sim_results = run_post_purchase_path_validation(
    V_choice_1, V_choice_2, V_choice_3, hh_posterior_val, hh_ranges,
    tya_state, p_continuous, hh_codes, af_val, as_val, aflav_val,
    n, n_flav, ψ_2, ψ_1, PSI_3, N_J, N_P, A_f, A_s, A_flav, P,
    cat_idx, period_idx, source_cats_cig, S;
    base_seed=54321, max_horizon=6
)

# Observed post-purchase paths (from actual choice data)
actual_ppp_cig_all, actual_ppp_cig_n_all = compute_post_purchase_paths(
    y, cat_idx, hh_codes, period_idx, source_cats_cig;
    max_horizon=6
)

actual_ppp_cig_tya, actual_ppp_cig_n_tya = compute_post_purchase_paths(
    y, cat_idx, hh_codes, period_idx, source_cats_cig;
    obs_mask=mask_tya_ppp, max_horizon=6
)

actual_ppp_cig_no_tya, actual_ppp_cig_n_no_tya = compute_post_purchase_paths(
    y, cat_idx, hh_codes, period_idx, source_cats_cig;
    obs_mask=mask_no_tya_ppp, max_horizon=6
)

ppp_cig_elapsed = time() - t_ppp_cig
log_msg("\nPost-purchase path validation (cigarettes) completed in $(round(ppp_cig_elapsed, digits=1))s")
log_msg("  Events (observed): All=$actual_ppp_cig_n_all, TYA=$actual_ppp_cig_n_tya, No TYA=$actual_ppp_cig_n_no_tya")
log_msg("  Events (sim draw 1): All=$(ppp_cig_sim_results.n_events_all), TYA=$(ppp_cig_sim_results.n_events_tya), No TYA=$(ppp_cig_sim_results.n_events_no_tya)")


#############################
# Compute Actual Streak
# Statistics by TYA Status
#############################

# Compute the empirical streak persistence curves from the observed holdout
# choice sequence y. These are the DATA SIDE of the validation comparison:
# the same compute_sim_streak_continuation function is called here with the
# observed choices y as input, and it was called inside run_simulation_validation
# with the simulated choices sim_y as input. Comparing actual_streaks_* to
# sim_results.streak_* is the validation test.
#
# We use the same obs_mask logic here as in the simulation: streak lengths are
# always computed on the full observed sequence, and the TYA/non-TYA filter is
# applied only at the accumulation step. This ensures simulated and observed
# rates are computed on identical subgroups and are directly comparable.

# TYA observation masks (same masks used in run_simulation_validation above)
mask_tya    = [tya_state[i] == 2 for i in eachindex(tya_state)]
mask_no_tya = [tya_state[i] == 1 for i in eachindex(tya_state)]

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

# Actual streak persistence for non-TYA households
actual_streaks_no_tya = compute_sim_streak_continuation(
    y, cat_idx, hh_codes, period_idx;
    obs_mask=mask_no_tya, max_streak=max_streak
)


#############################
# Print Streak Persistence
# Results
#############################

log_msg("\n===================================")
log_msg("Streak Persistence Curves")
log_msg("===================================")

# For each TYA group and product type, print the empirical continuation rate
# (Actual) vs. the mean simulated rate (Simulated), their difference
# (sim − actual; positive = model overpredicts persistence), and the
# simulation's [2.5%, 97.5%] confidence band across S draws.
for (tya_label, actual_streaks, sim_streaks, sim_ci) in [
    ("All",         actual_streaks_all,    sim_results.streak_all,    sim_results.streak_all_ci),
    ("TYA Present", actual_streaks_tya,    sim_results.streak_tya,    sim_results.streak_tya_ci),
    ("No TYA",      actual_streaks_no_tya, sim_results.streak_no_tya, sim_results.streak_no_tya_ci)
]

    log_msg("\n  --- $tya_label ---")

    for (label, full_label) in [("cig", "Cigarettes"), ("ecig", "E-Cigarettes"), ("flav_ecig", "Flavored E-Cigarettes"), ("orig_ecig", "Original E-Cigarettes")]

        # actual_res: max_streak × 3 — col 1 = streak length k, col 2 = empirical
        #             continuation rate, col 3 = N (observations at streak length k)
        # sim_res:    max_streak × 3 — col 2 = mean simulated continuation rate
        # ci_res:     max_streak × 2 — col 1 = p025, col 2 = p975 across S draws
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

# Write Streak_Persistence_{All,TYA,No_TYA}.csv. Each file has one row per
# (product type, streak_length) pair where N > 0. Columns: product,
# streak_length, N, actual_rate, simulated_rate, ci_025, ci_975.
for (tya_label, tya_suffix, actual_streaks, sim_streaks, sim_ci) in [
    ("All",         "All",    actual_streaks_all,    sim_results.streak_all,    sim_results.streak_all_ci),
    ("TYA Present", "TYA",    actual_streaks_tya,    sim_results.streak_tya,    sim_results.streak_tya_ci),
    ("No TYA",      "No_TYA", actual_streaks_no_tya, sim_results.streak_no_tya, sim_results.streak_no_tya_ci)
]

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
            @sprintf("%.10f", hh_posterior_val[h, 1]),
            @sprintf("%.10f", hh_posterior_val[h, 2]),
            @sprintf("%.10f", hh_posterior_val[h, 3]),
            @sprintf("%.10f", tya_share_val[h])], ","))
    end
end

log_msg("\nStreak results saved to: $output_dir")


#############################
# Print Post-Purchase Path
# Results (Flavored E-Cig)
#############################

log_msg("\n===================================")
log_msg("Post-Purchase Paths (after Flavored E-Cig Purchase)")
log_msg("===================================")

# For each TYA group, print post-purchase paths after a flavored e-cig purchase.
# Each row is one horizon month h (1–12). Columns show the actual and simulated
# share of households in each product category at horizon h, plus the [2.5%, 97.5%]
# confidence band. Four categories: flavored e-cig, original e-cig, cigarettes,
# outside option.
# actual_paths/sim_paths columns: col 1 = flav_ecig, 2 = orig_ecig, 3 = cig,
#   4 = outside, 5 = ecig combined.
# ci_paths columns (interleaved pairs): [p025_flav, p975_flav, p025_orig, p975_orig,
#   p025_cig, p975_cig, p025_out, p975_out, p025_ecig, p975_ecig].
for (tya_label, actual_paths, sim_paths, ci_paths) in [
    ("All",         actual_ppp_all,    ppp_sim_results.paths_all,    ppp_sim_results.paths_all_ci),
    ("TYA Present", actual_ppp_tya,    ppp_sim_results.paths_tya,    ppp_sim_results.paths_tya_ci),
    ("No TYA",      actual_ppp_no_tya, ppp_sim_results.paths_no_tya, ppp_sim_results.paths_no_tya_ci)
]

    log_msg("\n  --- $tya_label ---")
    log_msg(@sprintf("  %-8s  %10s  %10s  %10s  %10s  %10s  %10s  %10s  %10s  %10s  %10s  %10s  %10s  %10s  %10s  %10s  %10s",
        "Horizon",
        "Act Flav", "Sim Flav", "CI025 Flav", "CI975 Flav",
        "Act Orig", "Sim Orig", "CI025 Orig", "CI975 Orig",
        "Act Cig",  "Sim Cig",  "CI025 Cig",  "CI975 Cig",
        "Act Out",  "Sim Out",  "CI025 Out",  "CI975 Out"))
    log_msg("  " * repeat("-", 179))

    for h in 1:size(actual_paths, 1)
        log_msg(@sprintf("  %-8d  %10.4f  %10.4f  %10.4f  %10.4f  %10.4f  %10.4f  %10.4f  %10.4f  %10.4f  %10.4f  %10.4f  %10.4f  %10.4f  %10.4f  %10.4f  %10.4f",
            h,
            actual_paths[h, 1], sim_paths[h, 1], ci_paths[h, 1], ci_paths[h, 2],
            actual_paths[h, 2], sim_paths[h, 2], ci_paths[h, 3], ci_paths[h, 4],
            actual_paths[h, 3], sim_paths[h, 3], ci_paths[h, 5], ci_paths[h, 6],
            actual_paths[h, 4], sim_paths[h, 4], ci_paths[h, 7], ci_paths[h, 8]))
    end
end


#############################
# Print Post-Purchase Path
# Results (Cigarettes)
#############################

log_msg("\n===================================")
log_msg("Post-Purchase Paths (after Cigarette Purchase)")
log_msg("===================================")

# Same layout as the flavored e-cig post-purchase path table above, but
# households are tracked starting from a cigarette purchase (source_cats_cig).
for (tya_label, actual_paths, sim_paths, ci_paths) in [
    ("All",         actual_ppp_cig_all,    ppp_cig_sim_results.paths_all,    ppp_cig_sim_results.paths_all_ci),
    ("TYA Present", actual_ppp_cig_tya,    ppp_cig_sim_results.paths_tya,    ppp_cig_sim_results.paths_tya_ci),
    ("No TYA",      actual_ppp_cig_no_tya, ppp_cig_sim_results.paths_no_tya, ppp_cig_sim_results.paths_no_tya_ci)
]

    log_msg("\n  --- $tya_label ---")
    log_msg(@sprintf("  %-8s  %10s  %10s  %10s  %10s  %10s  %10s  %10s  %10s  %10s  %10s  %10s  %10s  %10s  %10s  %10s  %10s",
        "Horizon",
        "Act Flav", "Sim Flav", "CI025 Flav", "CI975 Flav",
        "Act Orig", "Sim Orig", "CI025 Orig", "CI975 Orig",
        "Act Cig",  "Sim Cig",  "CI025 Cig",  "CI975 Cig",
        "Act Out",  "Sim Out",  "CI025 Out",  "CI975 Out"))
    log_msg("  " * repeat("-", 179))

    for h in 1:size(actual_paths, 1)
        log_msg(@sprintf("  %-8d  %10.4f  %10.4f  %10.4f  %10.4f  %10.4f  %10.4f  %10.4f  %10.4f  %10.4f  %10.4f  %10.4f  %10.4f  %10.4f  %10.4f  %10.4f  %10.4f",
            h,
            actual_paths[h, 1], sim_paths[h, 1], ci_paths[h, 1], ci_paths[h, 2],
            actual_paths[h, 2], sim_paths[h, 2], ci_paths[h, 3], ci_paths[h, 4],
            actual_paths[h, 3], sim_paths[h, 3], ci_paths[h, 5], ci_paths[h, 6],
            actual_paths[h, 4], sim_paths[h, 4], ci_paths[h, 7], ci_paths[h, 8]))
    end
end


#############################
# Save Post-Purchase Path
# Results (Flavored E-Cig)
#############################

# write_ppp_csv is defined locally here because it is used for both the
# flavored e-cig and cigarette source categories, keeping the file-writing
# logic in one place rather than duplicating it in each save block below.
#
# ci_paths has 10 columns in interleaved layout: [p025_c1, p975_c1, p025_c2,
# p975_c2, ..., p025_c5, p975_c5] where c1=flav_ecig, c2=orig_ecig,
# c3=cig, c4=outside, c5=ecig (combined).
function write_ppp_csv(path, actual_paths, sim_paths, ci_paths)
    open(path, "w") do io
        # ci_paths columns: [p025_flav, p975_flav, p025_orig, p975_orig, p025_cig, p975_cig, p025_out, p975_out, p025_ecig, p975_ecig]
        println(io, join(["horizon",
            "actual_flav_ecig", "actual_orig_ecig", "actual_ecig", "actual_cig", "actual_outside",
            "sim_flav_ecig", "sim_orig_ecig", "sim_ecig", "sim_cig", "sim_outside",
            "ci_025_flav_ecig", "ci_975_flav_ecig",
            "ci_025_orig_ecig", "ci_975_orig_ecig",
            "ci_025_ecig", "ci_975_ecig",
            "ci_025_cig", "ci_975_cig",
            "ci_025_outside", "ci_975_outside"], ","))
        for h in 1:size(actual_paths, 1)
            println(io, join([
                string(h),
                @sprintf("%.10f", actual_paths[h, 1]),   # flav_ecig
                @sprintf("%.10f", actual_paths[h, 2]),   # orig_ecig
                @sprintf("%.10f", actual_paths[h, 5]),   # ecig (combined)
                @sprintf("%.10f", actual_paths[h, 3]),   # cig
                @sprintf("%.10f", actual_paths[h, 4]),   # outside
                @sprintf("%.10f", sim_paths[h, 1]),      # sim flav_ecig
                @sprintf("%.10f", sim_paths[h, 2]),      # sim orig_ecig
                @sprintf("%.10f", sim_paths[h, 5]),      # sim ecig (combined)
                @sprintf("%.10f", sim_paths[h, 3]),      # sim cig
                @sprintf("%.10f", sim_paths[h, 4]),      # sim outside
                @sprintf("%.10f", ci_paths[h, 1]),       # ci_025 flav_ecig
                @sprintf("%.10f", ci_paths[h, 2]),       # ci_975 flav_ecig
                @sprintf("%.10f", ci_paths[h, 3]),       # ci_025 orig_ecig
                @sprintf("%.10f", ci_paths[h, 4]),       # ci_975 orig_ecig
                @sprintf("%.10f", ci_paths[h, 9]),       # ci_025 ecig (combined)
                @sprintf("%.10f", ci_paths[h, 10]),      # ci_975 ecig (combined)
                @sprintf("%.10f", ci_paths[h, 5]),       # ci_025 cig
                @sprintf("%.10f", ci_paths[h, 6]),       # ci_975 cig
                @sprintf("%.10f", ci_paths[h, 7]),       # ci_025 outside
                @sprintf("%.10f", ci_paths[h, 8])        # ci_975 outside
            ], ","))
        end
    end
end

# Write Post_Purchase_Paths_Flav_{All,TYA,No_TYA}.csv via write_ppp_csv above.
for (tya_suffix, actual_paths, sim_paths, ci_paths) in [
    ("All",    actual_ppp_all,    ppp_sim_results.paths_all,    ppp_sim_results.paths_all_ci),
    ("TYA",    actual_ppp_tya,    ppp_sim_results.paths_tya,    ppp_sim_results.paths_tya_ci),
    ("No_TYA", actual_ppp_no_tya, ppp_sim_results.paths_no_tya, ppp_sim_results.paths_no_tya_ci)
]
    write_ppp_csv(joinpath(output_dir, "Post_Purchase_Paths_Flav_$(tya_suffix).csv"), actual_paths, sim_paths, ci_paths)
end

log_msg("\nFlavored e-cig post-purchase path results saved to: $output_dir")


#############################
# Save Post-Purchase Path
# Results (Cigarettes)
#############################

# Write Post_Purchase_Paths_Cig_{All,TYA,No_TYA}.csv via write_ppp_csv above.
for (tya_suffix, actual_paths, sim_paths, ci_paths) in [
    ("All",    actual_ppp_cig_all,    ppp_cig_sim_results.paths_all,    ppp_cig_sim_results.paths_all_ci),
    ("TYA",    actual_ppp_cig_tya,    ppp_cig_sim_results.paths_tya,    ppp_cig_sim_results.paths_tya_ci),
    ("No_TYA", actual_ppp_cig_no_tya, ppp_cig_sim_results.paths_no_tya, ppp_cig_sim_results.paths_no_tya_ci)
]
    write_ppp_csv(joinpath(output_dir, "Post_Purchase_Paths_Cig_$(tya_suffix).csv"), actual_paths, sim_paths, ci_paths)
end

log_msg("\nCigarette post-purchase path results saved to: $output_dir")


#############################
# Conditional Purchase
# Quantity Distribution
# and Quantity Escalation
#############################

# TEST 1 — Conditional purchase quantity distribution: given that a household
# buys cigarettes (or e-cigarettes), which of the discrete quantity bins does
# it choose? The model has 12 cigarette alternatives (cat 1, j=2:13) and 21
# ecig alternatives (cat 2/3/4, j=14:34), each representing a distinct
# quantity level. Comparing simulated vs. empirical bin shares tests whether
# the model replicates the intensive margin, not just the participation rate.
# Systematic underrepresentation of high-quantity bins would indicate that the
# price-sensitivity or addiction parameters underestimate the intensity of
# consumption conditional on buying.
#
# Bundle alternatives (cat 5,6,7) are excluded from the distribution because
# they are a joint cig+ecig product; their inclusion would conflate quantity
# choice with product-type choice.
#
# TEST 2 — Quantity escalation by streak length: does mean quantity purchased
# increase with consecutive-purchasing streak length? The γ_1 coefficient on
# the fast addiction stock implies that mid-streak households have higher stocks,
# raising the marginal utility of high-quantity alternatives and pushing optimal
# choices toward larger bins. If the simulated mean quantity at streak k does
# not rise with k when the data shows a slope, or rises faster than the data,
# that is direct evidence of γ_1 misspecification.
#
# Streak definition: mirrors compute_sim_streak_continuation exactly.
# Cig streak uses has_cig (cat 1 and 5,6,7); ecig streak uses has_ecig
# (cat 2,3,4,5,6,7). Bundles are included because addiction builds from any
# cig-containing (or ecig-containing) purchase, regardless of bundling.
#
# Both tests run inside a SINGLE S-draw loop (seed QUANT_BASE_SEED = 11111,
# distinct from streak=12345 and PPP=12345/54321) so the draws are independent.
# Computing both tests per draw halves simulation time vs. two separate loops.
# Results are computed for All / TYA-present / No-TYA subgroups.

log_msg("\n===================================")
log_msg("Conditional Quantity Distribution and Escalation ($S draws)...")
log_msg("===================================\n")

t_qty = time()

# --- Group definitions (All / TYA present / No TYA) ---
# mask_tya and mask_no_tya are already defined above for the streak section.
qty_group_labels   = ["All",     "TYA Present", "No TYA"]
qty_group_suffixes = ["All",     "TYA",         "No_TYA"]
qty_group_masks    = [trues(N_obs), mask_tya,   mask_no_tya]
N_qty_groups       = length(qty_group_labels)

# --- Identify pure cigarette and pure ecig alternatives ---
# cig_only_alts: j values where cat_idx[j] == 1 — the 12 cig quantity bins
# ecig_only_alts: j values where cat_idx[j] ∈ {2,3,4} — the 21 ecig quantity bins
cig_only_alts  = findall(j -> cat_idx[j] == 1,         1:N_J)
ecig_only_alts = findall(j -> cat_idx[j] ∈ (2, 3, 4), 1:N_J)

N_cig_bins  = length(cig_only_alts)   # 12
N_ecig_bins = length(ecig_only_alts)  # 21

# Map each alternative index to its within-set bin position (1-indexed)
cig_alt_to_bin  = Dict(j => b for (b, j) in enumerate(cig_only_alts))
ecig_alt_to_bin = Dict(j => b for (b, j) in enumerate(ecig_only_alts))

# --- Empirical bin shares (from observed y), by TYA group ---
# For each group, count how many cig purchases landed in each of the 12 bins;
# normalize to a share distribution. Same for 21 ecig bins.
cig_bin_shares_data  = [zeros(Float64, N_cig_bins)  for _ in 1:N_qty_groups]
ecig_bin_shares_data = [zeros(Float64, N_ecig_bins) for _ in 1:N_qty_groups]
for g in 1:N_qty_groups
    mask = qty_group_masks[g]
    for i in 1:N_obs
        !mask[i] && continue
        j = y[i]
        if cat_idx[j] == 1
            cig_bin_shares_data[g][cig_alt_to_bin[j]] += 1.0
        elseif cat_idx[j] ∈ (2, 3, 4)
            ecig_bin_shares_data[g][ecig_alt_to_bin[j]] += 1.0
        end
    end
    n_c = sum(cig_bin_shares_data[g])
    n_e = sum(ecig_bin_shares_data[g])
    cig_bin_shares_data[g]  = n_c > 0 ? cig_bin_shares_data[g]  ./ n_c : fill(NaN, N_cig_bins)
    ecig_bin_shares_data[g] = n_e > 0 ? ecig_bin_shares_data[g] ./ n_e : fill(NaN, N_ecig_bins)
end

# --- Empirical streak lengths from observed y ---
# One pass over all observations to build per-observation streak lengths for
# cig and ecig. New streak resets to 1 at the first period of a household
# (i == 1 or household/period boundary) or after a non-purchasing period.
streak_cig_data  = zeros(Int, N_obs)
streak_ecig_data = zeros(Int, N_obs)
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
end

# --- Empirical mean quantity by streak length, by TYA group ---
# q_cig[j] and q_ecig[j] are standardized quantities ∈ [0,1] (divided by
# q_cig_max and q_ecig_max). Multiply by those maxima in post-processing
# to recover raw packs / mL for publication figures.
mean_cig_qty_data  = [zeros(Float64, max_streak) for _ in 1:N_qty_groups]
mean_ecig_qty_data = [zeros(Float64, max_streak) for _ in 1:N_qty_groups]
cig_qty_n_data     = [zeros(Float64, max_streak) for _ in 1:N_qty_groups]
ecig_qty_n_data    = [zeros(Float64, max_streak) for _ in 1:N_qty_groups]
for g in 1:N_qty_groups
    mask = qty_group_masks[g]
    for i in 1:N_obs
        !mask[i] && continue
        j  = y[i]
        kc = min(streak_cig_data[i],  max_streak)
        if kc >= 1 && has_cig[j]
            mean_cig_qty_data[g][kc]  += q_cig[j]
            cig_qty_n_data[g][kc]     += 1.0
        end
        ke = min(streak_ecig_data[i], max_streak)
        if ke >= 1 && has_ecig[j]
            mean_ecig_qty_data[g][ke] += q_ecig[j]
            ecig_qty_n_data[g][ke]    += 1.0
        end
    end
    # Convert accumulated sums to means
    for k in 1:max_streak
        mean_cig_qty_data[g][k]  = cig_qty_n_data[g][k]  > 0 ? mean_cig_qty_data[g][k]  / cig_qty_n_data[g][k]  : NaN
        mean_ecig_qty_data[g][k] = ecig_qty_n_data[g][k] > 0 ? mean_ecig_qty_data[g][k] / ecig_qty_n_data[g][k] : NaN
    end
end

# --- Simulation draw accumulators ---
# cig_bin_shares_draws[g]: S × N_cig_bins — per-draw conditional cig bin shares for group g
# cig_qty_draws[g]: max_streak × S — per-draw mean standardized quantity at each streak k
cig_bin_shares_draws  = [fill(NaN, S, N_cig_bins)  for _ in 1:N_qty_groups]
ecig_bin_shares_draws = [fill(NaN, S, N_ecig_bins) for _ in 1:N_qty_groups]
cig_qty_draws         = [fill(NaN, max_streak, S)  for _ in 1:N_qty_groups]
ecig_qty_draws        = [fill(NaN, max_streak, S)  for _ in 1:N_qty_groups]

QUANT_BASE_SEED = 11111

for s in 1:S
    if s == 1 || s % 25 == 0
        log_msg("  Quantity validation draw $s / $S...")
    end

    # Simulate one complete forward sequence for all N_obs observations.
    # Addiction stocks evolve endogenously from simulated choices, so the
    # quantity escalation test captures addiction-driven selection into
    # high-quantity bins rather than static price-quantity tradeoffs.
    rng     = MersenneTwister(QUANT_BASE_SEED + s)
    sim_y_q = simulate_household_sequences_mixture(
        V_choice_1, V_choice_2, V_choice_3, hh_posterior_val, hh_ranges,
        tya_state, p_continuous, hh_codes, af_val, as_val, aflav_val,
        n, n_flav, ψ_2, ψ_1, PSI_3, N_J, N_P, A_f, A_s, A_flav, P, rng
    )

    # Simulated streak lengths (identical logic to empirical computation above)
    streak_cig_s  = zeros(Int, N_obs)
    streak_ecig_s = zeros(Int, N_obs)
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
    end

    # Accumulate both tests for each TYA group in a single pass over observations
    for g in 1:N_qty_groups
        mask = qty_group_masks[g]
        counts_cig   = zeros(Float64, N_cig_bins)
        counts_ecig  = zeros(Float64, N_ecig_bins)
        qty_cig_sum  = zeros(Float64, max_streak)
        qty_cig_n    = zeros(Float64, max_streak)
        qty_ecig_sum = zeros(Float64, max_streak)
        qty_ecig_n   = zeros(Float64, max_streak)
        for i in 1:N_obs
            !mask[i] && continue
            j  = sim_y_q[i]
            # Conditional quantity distribution: tally which bin was chosen
            if cat_idx[j] == 1
                counts_cig[cig_alt_to_bin[j]] += 1.0
            elseif cat_idx[j] ∈ (2, 3, 4)
                counts_ecig[ecig_alt_to_bin[j]] += 1.0
            end
            # Quantity escalation: accumulate q at each streak length
            kc = min(streak_cig_s[i],  max_streak)
            if kc >= 1 && has_cig[j]
                qty_cig_sum[kc] += q_cig[j];   qty_cig_n[kc]  += 1.0
            end
            ke = min(streak_ecig_s[i], max_streak)
            if ke >= 1 && has_ecig[j]
                qty_ecig_sum[ke] += q_ecig[j]; qty_ecig_n[ke] += 1.0
            end
        end
        n_c = sum(counts_cig);  n_e = sum(counts_ecig)
        if n_c > 0; cig_bin_shares_draws[g][s, :]  = counts_cig  ./ n_c; end
        if n_e > 0; ecig_bin_shares_draws[g][s, :] = counts_ecig ./ n_e; end
        for k in 1:max_streak
            cig_qty_draws[g][k, s]  = qty_cig_n[k]  > 0 ? qty_cig_sum[k]  / qty_cig_n[k]  : NaN
            ecig_qty_draws[g][k, s] = qty_ecig_n[k] > 0 ? qty_ecig_sum[k] / qty_ecig_n[k] : NaN
        end
    end
end

qty_elapsed = time() - t_qty
log_msg("\nQuantity validation completed in $(round(qty_elapsed, digits=1))s")

# Point estimates: mean over non-NaN draws (NaN arises when a group had zero
# purchases of that type in a given draw, which can happen for small subgroups)
sim_cig_bin_shares  = [[let v = filter(!isnan, cig_bin_shares_draws[g][:, b]);  isempty(v) ? NaN : mean(v) end for b in 1:N_cig_bins]  for g in 1:N_qty_groups]
sim_ecig_bin_shares = [[let v = filter(!isnan, ecig_bin_shares_draws[g][:, b]); isempty(v) ? NaN : mean(v) end for b in 1:N_ecig_bins] for g in 1:N_qty_groups]
mean_cig_qty_sim    = [[let v = filter(!isnan, vec(cig_qty_draws[g][k, :]));    isempty(v) ? NaN : mean(v) end for k in 1:max_streak]   for g in 1:N_qty_groups]
mean_ecig_qty_sim   = [[let v = filter(!isnan, vec(ecig_qty_draws[g][k, :]));   isempty(v) ? NaN : mean(v) end for k in 1:max_streak]   for g in 1:N_qty_groups]

# [2.5%, 97.5%] CI bands across S draws
cig_bin_ci  = [hcat(
    [let v = filter(!isnan, cig_bin_shares_draws[g][:, b]);  isempty(v) ? NaN : quantile(v, 0.025) end for b in 1:N_cig_bins],
    [let v = filter(!isnan, cig_bin_shares_draws[g][:, b]);  isempty(v) ? NaN : quantile(v, 0.975) end for b in 1:N_cig_bins]
) for g in 1:N_qty_groups]
ecig_bin_ci = [hcat(
    [let v = filter(!isnan, ecig_bin_shares_draws[g][:, b]); isempty(v) ? NaN : quantile(v, 0.025) end for b in 1:N_ecig_bins],
    [let v = filter(!isnan, ecig_bin_shares_draws[g][:, b]); isempty(v) ? NaN : quantile(v, 0.975) end for b in 1:N_ecig_bins]
) for g in 1:N_qty_groups]
cig_qty_ci  = [hcat(
    [let v = filter(!isnan, vec(cig_qty_draws[g][k, :]));    isempty(v) ? NaN : quantile(v, 0.025) end for k in 1:max_streak],
    [let v = filter(!isnan, vec(cig_qty_draws[g][k, :]));    isempty(v) ? NaN : quantile(v, 0.975) end for k in 1:max_streak]
) for g in 1:N_qty_groups]
ecig_qty_ci = [hcat(
    [let v = filter(!isnan, vec(ecig_qty_draws[g][k, :]));   isempty(v) ? NaN : quantile(v, 0.025) end for k in 1:max_streak],
    [let v = filter(!isnan, vec(ecig_qty_draws[g][k, :]));   isempty(v) ? NaN : quantile(v, 0.975) end for k in 1:max_streak]
) for g in 1:N_qty_groups]


#############################
# Print Conditional Quantity
# Distribution Results
#############################

log_msg("\n===================================")
log_msg("Conditional Purchase Quantity Distribution")
log_msg("===================================")
log_msg(@sprintf("\n  (Standardized quantities: multiply by q_cig_max=%.4f / q_ecig_max=%.4f for raw units)", q_cig_max, q_ecig_max))

# For each TYA group, print the empirical and simulated conditional bin shares
# and their difference. Column q_cig / q_ecig shows the standardized quantity
# level for that bin. Positive Difference = model overshoots that bin's share.
for (tya_label, g) in zip(qty_group_labels, 1:N_qty_groups)
    log_msg("\n  --- $tya_label ---")

    log_msg("\n  Cigarettes (pure cig purchases, cat 1):")
    log_msg(@sprintf("  %-6s  %8s  %10s  %10s  %10s  %10s  %10s",
        "Bin", "q_cig", "Data", "Simulated", "Difference", "CI_025", "CI_975"))
    log_msg("  " * repeat("-", 74))
    for b in 1:N_cig_bins
        j    = cig_only_alts[b]
        act  = cig_bin_shares_data[g][b]
        sim  = sim_cig_bin_shares[g][b]
        diff = isnan(act) || isnan(sim) ? NaN : sim - act
        lo   = cig_bin_ci[g][b, 1]
        hi   = cig_bin_ci[g][b, 2]
        if isnan(diff)
            log_msg(@sprintf("  %-6d  %8.4f  %10.4f  %10s  %10s  %10s  %10s",
                b, q_cig[j], act, "N/A", "N/A", "N/A", "N/A"))
        else
            lo_str = isnan(lo) ? "N/A" : @sprintf("%.4f", lo)
            hi_str = isnan(hi) ? "N/A" : @sprintf("%.4f", hi)
            log_msg(@sprintf("  %-6d  %8.4f  %10.4f  %10.4f  %+10.4f  %10s  %10s",
                b, q_cig[j], act, sim, diff, lo_str, hi_str))
        end
    end

    log_msg("\n  E-Cigarettes (pure ecig purchases, cat 2/3/4):")
    log_msg(@sprintf("  %-6s  %8s  %10s  %10s  %10s  %10s  %10s",
        "Bin", "q_ecig", "Data", "Simulated", "Difference", "CI_025", "CI_975"))
    log_msg("  " * repeat("-", 74))
    for b in 1:N_ecig_bins
        j    = ecig_only_alts[b]
        act  = ecig_bin_shares_data[g][b]
        sim  = sim_ecig_bin_shares[g][b]
        diff = isnan(act) || isnan(sim) ? NaN : sim - act
        lo   = ecig_bin_ci[g][b, 1]
        hi   = ecig_bin_ci[g][b, 2]
        if isnan(diff)
            log_msg(@sprintf("  %-6d  %8.4f  %10.4f  %10s  %10s  %10s  %10s",
                b, q_ecig[j], act, "N/A", "N/A", "N/A", "N/A"))
        else
            lo_str = isnan(lo) ? "N/A" : @sprintf("%.4f", lo)
            hi_str = isnan(hi) ? "N/A" : @sprintf("%.4f", hi)
            log_msg(@sprintf("  %-6d  %8.4f  %10.4f  %10.4f  %+10.4f  %10s  %10s",
                b, q_ecig[j], act, sim, diff, lo_str, hi_str))
        end
    end
end


#############################
# Print Quantity Escalation
# Results
#############################

log_msg("\n===================================")
log_msg("Quantity Escalation by Streak Length")
log_msg("===================================")

# For each TYA group and product, print mean standardized quantity at each
# streak length k. A rising Data Mean with k confirms that households shift
# toward higher-quantity bins as addiction builds. The Simulated column shows
# whether the model captures that slope; Difference = sim − data.
for (tya_label, g) in zip(qty_group_labels, 1:N_qty_groups)
    log_msg("\n  --- $tya_label ---")

    log_msg("\n  Cigarettes:")
    log_msg(@sprintf("  %-8s  %8s  %10s  %10s  %10s  %10s  %10s",
        "Streak", "N_data", "Data Mean", "Sim Mean", "Difference", "CI_025", "CI_975"))
    log_msg("  " * repeat("-", 79))
    for k in 1:max_streak
        n_k  = Int(cig_qty_n_data[g][k])
        act  = mean_cig_qty_data[g][k]
        (n_k == 0 || isnan(act)) && continue
        sim   = mean_cig_qty_sim[g][k]
        diff  = isnan(sim) ? NaN : sim - act
        lo    = cig_qty_ci[g][k, 1]
        hi    = cig_qty_ci[g][k, 2]
        k_lbl = k == max_streak ? "$(k)+" : string(k)
        if isnan(diff)
            log_msg(@sprintf("  %-8s  %8d  %10.4f  %10s  %10s  %10s  %10s",
                k_lbl, n_k, act, "N/A", "N/A", "N/A", "N/A"))
        else
            lo_str = isnan(lo) ? "N/A" : @sprintf("%.4f", lo)
            hi_str = isnan(hi) ? "N/A" : @sprintf("%.4f", hi)
            log_msg(@sprintf("  %-8s  %8d  %10.4f  %10.4f  %+10.4f  %10s  %10s",
                k_lbl, n_k, act, sim, diff, lo_str, hi_str))
        end
    end

    log_msg("\n  E-Cigarettes:")
    log_msg(@sprintf("  %-8s  %8s  %10s  %10s  %10s  %10s  %10s",
        "Streak", "N_data", "Data Mean", "Sim Mean", "Difference", "CI_025", "CI_975"))
    log_msg("  " * repeat("-", 79))
    for k in 1:max_streak
        n_k  = Int(ecig_qty_n_data[g][k])
        act  = mean_ecig_qty_data[g][k]
        (n_k == 0 || isnan(act)) && continue
        sim   = mean_ecig_qty_sim[g][k]
        diff  = isnan(sim) ? NaN : sim - act
        lo    = ecig_qty_ci[g][k, 1]
        hi    = ecig_qty_ci[g][k, 2]
        k_lbl = k == max_streak ? "$(k)+" : string(k)
        if isnan(diff)
            log_msg(@sprintf("  %-8s  %8d  %10.4f  %10s  %10s  %10s  %10s",
                k_lbl, n_k, act, "N/A", "N/A", "N/A", "N/A"))
        else
            lo_str = isnan(lo) ? "N/A" : @sprintf("%.4f", lo)
            hi_str = isnan(hi) ? "N/A" : @sprintf("%.4f", hi)
            log_msg(@sprintf("  %-8s  %8d  %10.4f  %10.4f  %+10.4f  %10s  %10s",
                k_lbl, n_k, act, sim, diff, lo_str, hi_str))
        end
    end
end


#############################
# Save Quantity Distribution
# and Escalation Results
#############################

# Writes 4 × 3 = 12 CSV files (four metrics × three TYA groups):
#   Conditional_Quantity_Distribution_Cig_{All,TYA,No_TYA}.csv
#     columns: bin_index, q_cig_std, data_share, sim_share, ci_025, ci_975
#   Conditional_Quantity_Distribution_Ecig_{All,TYA,No_TYA}.csv
#     columns: bin_index, q_ecig_std, data_share, sim_share, ci_025, ci_975
#   Quantity_Escalation_Cig_{All,TYA,No_TYA}.csv
#     columns: streak_length, n_data, data_mean_qty_std, sim_mean_qty_std, ci_025, ci_975
#   Quantity_Escalation_Ecig_{All,TYA,No_TYA}.csv
#     columns: streak_length, n_data, data_mean_qty_std, sim_mean_qty_std, ci_025, ci_975
# _std suffix indicates standardized quantities (÷ q_cig_max or q_ecig_max).
for (tya_suffix, g) in zip(qty_group_suffixes, 1:N_qty_groups)

    open(joinpath(output_dir, "Conditional_Quantity_Distribution_Cig_$(tya_suffix).csv"), "w") do io
        println(io, join(["bin_index", "q_cig_std", "data_share", "sim_share", "ci_025", "ci_975"], ","))
        for b in 1:N_cig_bins
            println(io, join([
                string(b),
                @sprintf("%.10f", q_cig[cig_only_alts[b]]),
                @sprintf("%.10f", cig_bin_shares_data[g][b]),
                @sprintf("%.10f", sim_cig_bin_shares[g][b]),
                @sprintf("%.10f", cig_bin_ci[g][b, 1]),
                @sprintf("%.10f", cig_bin_ci[g][b, 2])
            ], ","))
        end
    end

    open(joinpath(output_dir, "Conditional_Quantity_Distribution_Ecig_$(tya_suffix).csv"), "w") do io
        println(io, join(["bin_index", "q_ecig_std", "data_share", "sim_share", "ci_025", "ci_975"], ","))
        for b in 1:N_ecig_bins
            println(io, join([
                string(b),
                @sprintf("%.10f", q_ecig[ecig_only_alts[b]]),
                @sprintf("%.10f", ecig_bin_shares_data[g][b]),
                @sprintf("%.10f", sim_ecig_bin_shares[g][b]),
                @sprintf("%.10f", ecig_bin_ci[g][b, 1]),
                @sprintf("%.10f", ecig_bin_ci[g][b, 2])
            ], ","))
        end
    end

    open(joinpath(output_dir, "Quantity_Escalation_Cig_$(tya_suffix).csv"), "w") do io
        println(io, join(["streak_length", "n_data", "data_mean_qty_std", "sim_mean_qty_std", "ci_025", "ci_975"], ","))
        for k in 1:max_streak
            n_k = Int(cig_qty_n_data[g][k])
            n_k == 0 && continue
            println(io, join([
                k == max_streak ? "$(k)+" : string(k),
                string(n_k),
                @sprintf("%.10f", mean_cig_qty_data[g][k]),
                @sprintf("%.10f", mean_cig_qty_sim[g][k]),
                @sprintf("%.10f", cig_qty_ci[g][k, 1]),
                @sprintf("%.10f", cig_qty_ci[g][k, 2])
            ], ","))
        end
    end

    open(joinpath(output_dir, "Quantity_Escalation_Ecig_$(tya_suffix).csv"), "w") do io
        println(io, join(["streak_length", "n_data", "data_mean_qty_std", "sim_mean_qty_std", "ci_025", "ci_975"], ","))
        for k in 1:max_streak
            n_k = Int(ecig_qty_n_data[g][k])
            n_k == 0 && continue
            println(io, join([
                k == max_streak ? "$(k)+" : string(k),
                string(n_k),
                @sprintf("%.10f", mean_ecig_qty_data[g][k]),
                @sprintf("%.10f", mean_ecig_qty_sim[g][k]),
                @sprintf("%.10f", ecig_qty_ci[g][k, 1]),
                @sprintf("%.10f", ecig_qty_ci[g][k, 2])
            ], ","))
        end
    end
end

log_msg("\nQuantity distribution and escalation results saved to: $output_dir")


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
