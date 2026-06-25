################################################################################
# William Brasic
# The University of Arizona
# wbrasic97@gmail.com
# June 2026
#
# This script estimates the structural parameters of the dynamic model by
# maximizing the sample log-likelihood via multi-start Nelder-Mead.
#
# The objective function for each candidate θ:
#   1. Recomputes flow utility U given θ
#   2. Solves the value function via VFI
#   3. Computes the log-likelihood by interpolating V_decision
#      at each observation's continuous  state
#   4. Returns the negative log-likelihood (since we minimize)
#
# Progress is logged to a log file in the output directory (named by Slurm job ID on HPC, timestamp locally).
################################################################################


#############################
# Preliminaries
#############################

# Flavored habit decay rate ψ_3
# If ESTIMATE_PSI_3=true, ψ_3 is added to the parameter vector and estimated jointly.
# If false (default), ψ_3 is fixed at PSI_3 (read from ENV, default 0.75).
ESTIMATE_PSI_3 = parse(Bool, get(ENV, "ESTIMATE_PSI_3", "false"))
PSI_3 = parse(Float64, get(ENV, "PSI_3", "0.75"))

# Discount factor β (read from ENV, default 1.0)
BETA = parse(Float64, get(ENV, "BETA", "1.0"))

# Hardcoded fixed parameters
ψ_1 = 0.10
ψ_2 = 0.90
β   = BETA

# Set to true to warm-start VFI from the previous evaluation's converged V
WARM_START = true

# VFI convergence tolerance (sup-norm)
VFI_TOL = 1e-4

# Load all functions and packages from the functions file
include("./01_Functions_Mixture.jl")

# Detect whether we are running on the HPC (any non-Windows system)
HPC = !Sys.iswindows()

# Unique file suffix: date + Slurm job ID on HPC, date + HHMMSS locally
date_str = Dates.format(today(), "yyyy-mm-dd")
job_id = get(ENV, "SLURM_JOB_ID", Dates.format(now(), "HHMMSS"))
run_id = "$(date_str)_$(job_id)"

# Construct psi, beta, and psi_3 tags for output directory and file naming
psi_tag   = "Psi2_09_Psi1_01"
beta_tag  = "Beta_$(numeric_tag(BETA))"
psi_3_tag = ESTIMATE_PSI_3 ? "Psi3_Est" : "Psi3_$(numeric_tag(PSI_3))"

# File paths if on HPC or not
if HPC

    # Output path for results (resolve relative to script dir so it's unaffected by later cd)
    output_dir = abspath(joinpath(@__DIR__, "Dynamic_Model_Mixture_V2_$(psi_tag)_$(beta_tag)_$(psi_3_tag)_Results"))
    mkpath(output_dir)

    # Set working directory to where the data CSVs live
    cd("/home/u2/wbrasic/4th_Year_Paper/Data")
else

    # Output path for results (local Windows path)
    output_dir = "C:/Users/wbras/OneDrive/Documents/Desktop/UA/4th_Year_Paper/4th_Year_Paper_Data/HMS/2021-Onward/Dynamic_Model/Dynamic_Model_Mixture_V2_$(psi_tag)_$(beta_tag)_$(psi_3_tag)_Results"
    mkpath(output_dir)

    # Set working directory to where the data CSVs live
    cd("C:/Users/wbras/OneDrive/Documents/Desktop/UA/4th_Year_Paper/4th_Year_Paper_Data/HMS/2021-Onward/Dynamic_Model/Data")
end

# Create subdirectories for log, outer try, inner try, and estimates files.
# Folder names match file naming convention (no replication number prefix since this is not a job array).
log_dir          = joinpath(output_dir, "Dynamic_Model_Mixture_V2_$(psi_tag)_$(beta_tag)_$(psi_3_tag)_Log")
outer_try_dir    = joinpath(output_dir, "Dynamic_Model_Mixture_V2_$(psi_tag)_$(beta_tag)_$(psi_3_tag)_Outer_Try_Params")
inner_try_dir    = joinpath(output_dir, "Dynamic_Model_Mixture_V2_$(psi_tag)_$(beta_tag)_$(psi_3_tag)_Inner_Try_Params")
estimates_subdir = joinpath(output_dir, "Dynamic_Model_Mixture_V2_$(psi_tag)_$(beta_tag)_$(psi_3_tag)_Estimates")
mkpath(log_dir)
mkpath(outer_try_dir)
mkpath(inner_try_dir)
mkpath(estimates_subdir)

# Per-run output file paths (all go in subdirectories)
log_path       = joinpath(log_dir, "Dynamic_Model_Mixture_V2_$(psi_tag)_$(beta_tag)_$(psi_3_tag)_Log_$(run_id).txt")
outer_try_path = joinpath(outer_try_dir, "Dynamic_Model_Mixture_V2_$(psi_tag)_$(beta_tag)_$(psi_3_tag)_Outer_Try_Params_$(run_id).csv")
inner_try_path = joinpath(inner_try_dir, "Dynamic_Model_Mixture_V2_$(psi_tag)_$(beta_tag)_$(psi_3_tag)_Inner_Try_Params_$(run_id).csv")

# Open log file for writing (log_io is defined as a global in 01_Functions.jl)
log_io = open(log_path, "w")

# Print and log the start time and number of Julia threads available for VFI parallelization
log_msg("Dynamic estimation ($(psi_tag), $(beta_tag), $(psi_3_tag)) started at $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
log_msg("")
log_msg("Number of threads: $(Threads.nthreads())")
log_msg("")

# Get household identifiers (pre-loaded to avoid repeated CSV reads in objective)
hh_codes = get_hh_codes();

# Pre-compute contiguous household index ranges for mixture log-likelihood
# (called once; result is used by log_likelihood_mixture in every objective evaluation)
hh_ranges = precompute_hh_ranges(hh_codes);


#############################
# Initialize fixed parameters
#############################

# Load fixed parameters: only δ is taken from get_fixed_parameters();
# ψ_1, ψ_2, and β are hardcoded above
_, _, _, δ = get_fixed_parameters();


#############################
# State Spaces and Choices
#############################

# Get fast addiction grid (N_A_f = 5 points, stock with ψ_2 = 0.90)
N_A_f, A_f = get_addiction_space(ψ_2; N_A=5);

# Get slow addiction grid (N_A_s = 10 points, stock with ψ_1 = 0.10)
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

# Get flavor lock-in indicator: orig ecig (cat 2) and orig bundle (cat 5), ecig-side lock-in for γ_2
is_nonflavored_ecig = [cat_idx[j] in (2, 5) for j in 1:N_J]

# Outside option indicator: cat 0 = outside option (j = 1)
is_outside = [cat_idx[j] == 0 for j in 1:N_J]

# Get indicator for alternatives containing cigarettes (cat 1 = cig, cat 5-7 = bundles with cig)
has_cig = [(cat_idx[j] == 1 || cat_idx[j] >= 5) for j in 1:N_J]

# Get indicator for alternatives containing e-cigarettes (cat 2-7 = any ecig or bundle)
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
# p_continuous is N × 2 (cig price, ecig price): representative (median-across-bins) prices for VFI interpolation
# P_obs_cig / P_obs_ecig are N × N_J matrices of actual bin-specific prices for the P_obs correction
_, p_continuous, P_obs_cig, P_obs_ecig = map_prices_to_grid(N_P, P, Pcomb, N_J);


#############################
# Pre-compute Addiction
# Objects (Two Stocks)
#############################

# Fast stock (always pre-computed, ψ_2 is never estimated)
af_lower_current, af_upper_current, af_weight_current = precompute_addiction_transitions(N_J, N_A_f, ψ_2, A_f, n)
af0_current, _ = get_initial_addiction_stock(ψ_2, A_f, n, y, hh_codes)
af_continuous_current = simulate_addiction_trajectories(N_A_f, ψ_2, A_f, n, y, hh_codes, af0_current)

# Slow stock (pre-computed at fixed ψ_1 = 0.10)
as_lower_current, as_upper_current, as_weight_current = precompute_addiction_transitions(N_J, N_A_s, ψ_1, A_s, n)
as0_current, _ = get_initial_addiction_stock(ψ_1, A_s, n, y, hh_codes)
as_continuous_current = simulate_addiction_trajectories(N_A_s, ψ_1, A_s, n, y, hh_codes, as0_current)

# Flavored habit stock
# n_flav[j] = 1[flavored[j]] ∈ {0,1}: binary indicator; habit builds by one unit any time
# a flavored alternative is chosen, regardless of quantity.
n_flav     = Float64.(is_flavored)             # ∈ {0.0, 1.0}
n_flav_max = 1.0                               # binary max; rescale γ_2, γ_3, γ_4 by × ψ_3
N_A_flav, A_flav = get_addiction_space(PSI_3; N_A=10)

# When ψ_3 is fixed, pre-compute transitions and trajectories once.
# When ψ_3 is estimated, these are recomputed inside the objective function
# at each candidate ψ_3 value, so we only set initial values here.
aflav_lower_current, aflav_upper_current, aflav_weight_current = precompute_addiction_transitions(N_J, N_A_flav, PSI_3, A_flav, n_flav)
aflav0_current, _ = get_initial_addiction_stock(PSI_3, A_flav, n_flav, y, hh_codes)
aflav_continuous_current = simulate_addiction_trajectories(N_A_flav, PSI_3, A_flav, n_flav, y, hh_codes, aflav0_current)


#############################
# Estimation
#############################

# Print and log standardization factors (needed to rescale parameter estimates)
log_msg("Standardization factors (divide estimates by these to get original units):")
log_msg("")
log_msg(rpad("  q_cig_max    = $q_cig_max (packs)",        44) * "— standardizes q_cig; rescale α_C, ω_C by ÷ q_cig_max")
log_msg(rpad("  q_ecig_max   = $q_ecig_max (mL)",          44) * "— standardizes q_ecig; rescale α_E, ω_E by ÷ q_ecig_max")
log_msg(rpad("  q_bundle_max = $q_bundle_max (packs × mL)", 44) * "— standardizes q_bundle; rescale α_CE by ÷ q_bundle_max")
log_msg(rpad("  n_max        = $n_max (mg)",                44) * "— standardizes nicotine; rescale γ_1 by ÷ n_max")
log_msg(rpad("  n_flav_max   = $n_flav_max (mL)",           44) * "— max flavored ecig quantity; rescale γ_2, γ_3, γ_4 by × ψ_3 ÷ n_flav_max")
log_msg("  A_f_max      = $(maximum(A_f)) (fast addiction grid max, normalized to [0, 1])")
log_msg("  A_s_max      = $(maximum(A_s)) (slow addiction grid max, normalized to [0, 1])")
log_msg("  A_flav_max   = $(maximum(A_flav)) (flavored habit grid max, normalized to [0, 1])")
log_msg("  PSI_3        = $PSI_3 (flavored habit decay rate, $(ESTIMATE_PSI_3 ? "ESTIMATED" : "fixed"))")
log_msg("  ESTIMATE_PSI_3 = $ESTIMATE_PSI_3")

# Standardized starting values for the K=3 mixture model
# Common params include γ_2 (orig ecig/bundle lock-in), γ_3 (cig/orig bundle flavor lock-in), γ_4 (outside flavored withdrawal)
# When ESTIMATE_PSI_3=true, ψ_3 is appended as the last parameter 
if ESTIMATE_PSI_3
    starting_param = (
        α_C     =   1.7539962989,
        α_E     =   2.6528275992,
        α_CE    =  -0.5793447834,
        λ_1     =   0.1458019323,  # flavor baseline (common)
        λ_2     =   0.2106063714,  # flavor × TYA (common)
        λ_3     =  -0.0456078917,
        λ_4     =  -0.4563471644,
        γ_1     = -12.3435966720,
        γ_2     = -11.9087621394,  # flavor lock-in penalty on orig ecig/bundle (cat 2, 5)
        γ_3     =  -2.4146527666,  # flavor lock-in penalty on cig/orig bundle (cat 1, 5)
        γ_4     =  -0.5518799376,  # flavored withdrawal cost on outside option
        ω_C     =  -0.30,
        ω_E     =  -1.10,
        ξ_C_1   =  -2.8416876225,
        ξ_E_1   =  -9.3446139588,
        ξ_CE_1  =  -7.2523098636,
        ξ_C_2   =  -4.8294614255,
        ξ_E_2   =  -3.8008342716,
        ξ_CE_2  =  -4.1974220102,
        ξ_C_3   = -15.0000,        # type 3 starting values (non-purchasers: all low utility)
        ξ_E_3   = -15.0000,
        ξ_CE_3  = -15.0000,
        π_0_2   =  -0.69,           # type 2 baseline logit intercept (~5% ecig at tya=0)
        π_TYA_2 =   0.50,          # type 2 TYA share shifter (TYA more likely ecig)
        π_0_3   =   2.14,          # type 3 baseline logit intercept (~85% non-purchasers at tya=0)
        π_TYA_3 =  -0.25,          # type 3 TYA share shifter (TYA slightly less likely non-purchaser)
        ψ_3     =   0.75           # default starting value
    )
else
    starting_param = (
        α_C     =   1.7539962989,
        α_E     =   2.6528275992,
        α_CE    =  -0.5793447834,
        λ_1     =   0.1458019323,  # flavor baseline (common)
        λ_2     =   0.2106063714,  # flavor × TYA (common)
        λ_3     =  -0.0456078917,
        λ_4     =  -0.4563471644,
        γ_1     = -12.3435966720,
        γ_2     = -11.9087621394,  # flavor lock-in penalty on orig ecig/bundle (cat 2, 5)
        γ_3     =  -2.4146527666,  # flavor lock-in penalty on cig/orig bundle (cat 1, 5)
        γ_4     =  -0.5518799376,  # flavored withdrawal cost on outside option
        ω_C     =  -0.30,
        ω_E     =  -1.10,
        ξ_C_1   =  -2.8416876225,
        ξ_E_1   =  -9.3446139588,
        ξ_CE_1  =  -7.2523098636,
        ξ_C_2   =  -4.8294614255,
        ξ_E_2   =  -3.8008342716,
        ξ_CE_2  =  -4.1974220102,
        ξ_C_3   = -15.0000,        # type 3 starting values (non-purchasers: all low utility)
        ξ_E_3   = -15.0000,
        ξ_CE_3  = -15.0000,
        π_0_2   =  -0.69,           # type 2 baseline logit intercept (~5% ecig at tya=0)
        π_TYA_2 =   0.50,          # type 2 TYA share shifter (TYA more likely ecig)
        π_0_3   =   2.14,          # type 3 baseline logit intercept (~85% non-purchasers at tya=0)
        π_TYA_3 =  -0.25           # type 3 TYA share shifter (TYA slightly less likely non-purchaser)
    )
end

# Initial simplex deviations for Nelder-Mead
add = [
    abs(starting_param.α_C)     * 0.50,   # α_C
    abs(starting_param.α_E)     * 0.50,   # α_E
    abs(starting_param.α_CE)    * 0.50,   # α_CE
    abs(starting_param.λ_1)     * 0.50,   # λ_1: flavor baseline (common)
    abs(starting_param.λ_2)     * 0.50,   # λ_2: flavor × TYA (common)
    abs(starting_param.λ_3)     * 0.50,   # λ_3
    abs(starting_param.λ_4)     * 0.50,   # λ_4
    abs(starting_param.γ_1)     * 1.00,   # γ_1: larger for dynamic parameters
    abs(starting_param.γ_2)     * 1.00,   # γ_2: larger for dynamic parameters
    abs(starting_param.γ_3)     * 1.00,   # γ_3: larger for dynamic parameters
    abs(starting_param.γ_4)     * 1.00,   # γ_4: larger for dynamic parameters
    abs(starting_param.ω_C)     * 0.50,   # ω_C
    abs(starting_param.ω_E)     * 0.50,   # ω_E
    abs(starting_param.ξ_C_1)   * 0.50,   # ξ_C_1
    abs(starting_param.ξ_E_1)   * 0.50,   # ξ_E_1
    abs(starting_param.ξ_CE_1)  * 0.50,   # ξ_CE_1
    abs(starting_param.ξ_C_2)   * 0.50,   # ξ_C_2
    abs(starting_param.ξ_E_2)   * 0.50,   # ξ_E_2
    abs(starting_param.ξ_CE_2)  * 0.50,   # ξ_CE_2
    abs(starting_param.ξ_C_3)   * 0.50,   # ξ_C_3
    abs(starting_param.ξ_E_3)   * 0.50,   # ξ_E_3
    abs(starting_param.ξ_CE_3)  * 0.50,   # ξ_CE_3
    abs(starting_param.π_0_2)   * 0.50,   # π_0_2
    abs(starting_param.π_TYA_2) * 0.50,   # π_TYA_2
    abs(starting_param.π_0_3)   * 0.50,   # π_0_3
    abs(starting_param.π_TYA_3) * 0.50   # π_TYA_3
]
if ESTIMATE_PSI_3
    push!(add, 0.20)                       # ψ_3: deviation of 0.20
end

# Optimizer settings for random_amoeba multi-start Nelder-Mead
L          = 20    # Number of random restarts (outer tries)
M          = 5     # Short Nelder-Mead runs per outer try (inner tries)
inner_iter = 250   # Max iterations per short run

# Print and log estimation settings
log_msg("")
log_msg("Estimation Settings: Observations = $(length(y)), Outer Tries = $L, Inner Tries = $M, Inner Try Iterations = $inner_iter")
log_msg("")

# Reset the global evaluation counter before estimation begins
est_eval_count = 0

# Set global parameter names so objective() can log parameter names alongside values
est_param_names = replace.(collect(String, string.(keys(starting_param))),
    "α" => "alpha", "λ" => "lambda", "γ" => "gamma",
    "ω" => "omega", "ξ" => "xi", "π" => "pi", "ψ" => "psi")

# Print and log estimation header
log_msg("")
log_msg("===================================")
log_msg("Running Estimation")
log_msg("===================================")
log_msg("")

# Run multi-start Nelder-Mead optimization
t_est = time()
opt_param, opt_value = random_amoeba(
    objective, starting_param, add, L, M;
    inner_iter = inner_iter,
    outer_try_file = outer_try_path,
    inner_try_file = inner_try_path
)
est_elapsed = time() - t_est

# Print and log estimation completion
log_msg("Estimation complete: $est_eval_count evaluations in $(round(est_elapsed, digits=1))s")

# Print and log estimation results
log_msg("")
log_msg("===================================")
log_msg("Estimation Results")
log_msg("===================================")
log_msg("  Negative log-likelihood: $(round(opt_value, digits=4))")
log_msg(@sprintf("  %-8s  %12s", "Param", "Estimated"))
log_msg("  " * repeat("-", 22))
for (i, val) in enumerate(values(opt_param))
    log_msg(@sprintf("  %-8s  %12.6f", est_param_names[i], val))
end

# Save estimated parameters to _Estimates.csv (header row + estimate row; SE row added by 03_Standard_Errors_Mixture.jl)
estimates_path = joinpath(estimates_subdir, "Dynamic_Model_Mixture_V2_$(psi_tag)_$(beta_tag)_$(psi_3_tag)_Estimates.csv")
open(estimates_path, "w") do io
    println(io, join(replace.(collect(String, string.(keys(opt_param))),
        "α" => "alpha", "λ" => "lambda", "γ" => "gamma",
        "ω" => "omega", "ξ" => "xi", "π" => "pi", "ψ" => "psi"), ","))
    println(io, join([@sprintf("%.10f", v) for v in values(opt_param)], ","))
end
log_msg("")
log_msg("Estimates saved to: $estimates_path")
log_msg("Outer try parameters saved to: $outer_try_path")
log_msg("Inner try parameters saved to: $inner_try_path")


#############################
# Post-Estimation:
# Statistics and Per-HH LL
#############################

# Number of estimated parameters and NLL
nll_val = opt_value
N_params = length(opt_param)

# Save _Statistics.csv: LL (positive = -NLL), AIC, BIC
unique_hh = unique(hh_codes)
N_hh = length(unique_hh)
ll_val = -nll_val
aic_val = 2.0 * nll_val + 2.0 * N_params
bic_val = 2.0 * nll_val + N_params * log(N_hh)

stats_path = joinpath(estimates_subdir, "Dynamic_Model_Mixture_V2_$(psi_tag)_$(beta_tag)_$(psi_3_tag)_Statistics.csv")
open(stats_path, "w") do io
    println(io, "LL,AIC,BIC")
    println(io, join([@sprintf("%.10f", v) for v in [ll_val, aic_val, bic_val]], ","))
end
log_msg("Statistics saved to: $stats_path")

# Re-solve VFI at ̂θ_hat to obtain V_decision for per-household log-likelihood decomposition.
# This mirrors the objective function's parameter extraction and VFI calls exactly.
log_msg("")
log_msg("===================================")
log_msg("Re-solving VFI at θ̂ for per-HH LL")
log_msg("===================================")

# Convert opt_param NamedTuple to vector for extraction
θ_hat_vec = collect(Float64, values(opt_param))

# Extract K=3 mixture parameters from positions 1-26
common_post  = θ_hat_vec[1:13]
type_1_post  = θ_hat_vec[14:16]
type_2_post  = θ_hat_vec[17:19]
type_3_post  = θ_hat_vec[20:22]
π_0_2_post   = θ_hat_vec[23]
π_TYA_2_post = θ_hat_vec[24]
π_0_3_post   = θ_hat_vec[25]
π_TYA_3_post = θ_hat_vec[26]

# When ψ_3 was estimated, the pre-computed flavored addiction objects use ENV PSI_3, which
# may differ from the estimated ψ_3 = θ_hat_vec[27]. Recompute at the estimated value so
# the VFI re-solve and per-HH LL are consistent with what the objective maximized.
if ESTIMATE_PSI_3
    ψ_3_post = θ_hat_vec[27]
    N_A_flav, A_flav = get_addiction_space(ψ_3_post; N_A=10)
    aflav_lower_current, aflav_upper_current, aflav_weight_current = precompute_addiction_transitions(N_J, N_A_flav, ψ_3_post, A_flav, n_flav)
    aflav0_current, _ = get_initial_addiction_stock(ψ_3_post, A_flav, n_flav, y, hh_codes)
    aflav_continuous_current = simulate_addiction_trajectories(N_A_flav, ψ_3_post, A_flav, n_flav, y, hh_codes, aflav0_current)
    log_msg("Recomputed flavored addiction objects at estimated ψ_3 = $ψ_3_post for per-HH LL re-solve")
end

# Construct per-type structural parameter vectors (16 elements each)
θ_struct_1_post = vcat(common_post, type_1_post)
θ_struct_2_post = vcat(common_post, type_2_post)
θ_struct_3_post = vcat(common_post, type_3_post)

# Compute flow utility for each type
U_1_post = get_flow_utility(
    θ_struct_1_post, N_J, N_A_f, N_A_s, N_A_flav, N_Pcomb, A_f, A_s, A_flav,
    q_cig, q_ecig, q_bundle, is_flavored, is_fda_flavored, is_nonflavored_ecig, is_outside, cat_idx, Pcomb, has_cig, has_ecig
)
U_2_post = get_flow_utility(
    θ_struct_2_post, N_J, N_A_f, N_A_s, N_A_flav, N_Pcomb, A_f, A_s, A_flav,
    q_cig, q_ecig, q_bundle, is_flavored, is_fda_flavored, is_nonflavored_ecig, is_outside, cat_idx, Pcomb, has_cig, has_ecig
)
U_3_post = get_flow_utility(
    θ_struct_3_post, N_J, N_A_f, N_A_s, N_A_flav, N_Pcomb, A_f, A_s, A_flav,
    q_cig, q_ecig, q_bundle, is_flavored, is_fda_flavored, is_nonflavored_ecig, is_outside, cat_idx, Pcomb, has_cig, has_ecig
)

# Solve VFI for all three types in parallel (1 extra solve per type)
t_vfi_post = time()
vfi_task_1_post = Threads.@spawn solve_vfi_sophisticated(
    N_J, N_A_f, N_A_s, N_A_flav, N_P, N_Pcomb, β, δ, U_1_post,
    af_lower_current, af_upper_current, af_weight_current,
    as_lower_current, as_upper_current, as_weight_current,
    aflav_lower_current, aflav_upper_current, aflav_weight_current,
    p_cig_lo, p_cig_hi, p_cig_w,
    p_ecig_lo, p_ecig_hi, p_ecig_w;
    V_init = nothing,
    ε = VFI_TOL,
    verbose = true
)
vfi_task_2_post = Threads.@spawn solve_vfi_sophisticated(
    N_J, N_A_f, N_A_s, N_A_flav, N_P, N_Pcomb, β, δ, U_2_post,
    af_lower_current, af_upper_current, af_weight_current,
    as_lower_current, as_upper_current, as_weight_current,
    aflav_lower_current, aflav_upper_current, aflav_weight_current,
    p_cig_lo, p_cig_hi, p_cig_w,
    p_ecig_lo, p_ecig_hi, p_ecig_w;
    V_init = nothing,
    ε = VFI_TOL,
    verbose = true
)
vfi_task_3_post = Threads.@spawn solve_vfi_sophisticated(
    N_J, N_A_f, N_A_s, N_A_flav, N_P, N_Pcomb, β, δ, U_3_post,
    af_lower_current, af_upper_current, af_weight_current,
    as_lower_current, as_upper_current, as_weight_current,
    aflav_lower_current, aflav_upper_current, aflav_weight_current,
    p_cig_lo, p_cig_hi, p_cig_w,
    p_ecig_lo, p_ecig_hi, p_ecig_w;
    V_init = nothing,
    ε = VFI_TOL,
    verbose = true
)

# Wait for all three VFI tasks to complete
_, V_decision_1_post, vfi_iters_1_post, vfi_converged_1_post = fetch(vfi_task_1_post)
_, V_decision_2_post, vfi_iters_2_post, vfi_converged_2_post = fetch(vfi_task_2_post)
_, V_decision_3_post, vfi_iters_3_post, vfi_converged_3_post = fetch(vfi_task_3_post)
log_msg("Post-estimation VFI: type1=$(vfi_iters_1_post) iters ($(vfi_converged_1_post)), type2=$(vfi_iters_2_post) iters ($(vfi_converged_2_post)), type3=$(vfi_iters_3_post) iters ($(vfi_converged_3_post)), $(round(time() - t_vfi_post, digits=1))s")

# Compute per-household log-likelihood using mixture logsumexp
hh_ll = Vector{Float64}(undef, N_hh)
ω_C_post = opt_param.ω_C
ω_E_post = opt_param.ω_E
for h in 1:N_hh
    start_idx, stop_idx = hh_ranges[h]

    # Household-specific log mixing weights (K=3 softmax, type 1 normalized)
    logit_2_h   = π_0_2_post + π_TYA_2_post * tya_share_hh[h]
    logit_3_h   = π_0_3_post + π_TYA_3_post * tya_share_hh[h]
    log_denom_h = logsumexp([0.0, logit_2_h, logit_3_h])
    log_π_1_h   = -log_denom_h
    log_π_2_h   = logit_2_h - log_denom_h
    log_π_3_h   = logit_3_h - log_denom_h

    # Per-type log-likelihood for this household
    log_ll_1 = 0.0
    log_ll_2 = 0.0
    log_ll_3 = 0.0
    for i in start_idx:stop_idx
        v_interp_1 = interpolate_v_choice(
            V_decision_1_post, tya_state[i], af_continuous_current[i], as_continuous_current[i],
            aflav_continuous_current[i], p_continuous[i, 1], p_continuous[i, 2],
            N_J, N_P, A_f, A_s, A_flav, P
        )
        for j in 1:N_J
            v_interp_1[j] += ω_C_post * (P_obs_cig[i, j]  - p_continuous[i, 1]) * q_cig[j] +
                              ω_E_post * (P_obs_ecig[i, j] - p_continuous[i, 2]) * q_ecig[j]
        end
        log_ll_1 += v_interp_1[y[i]] - logsumexp(v_interp_1)

        v_interp_2 = interpolate_v_choice(
            V_decision_2_post, tya_state[i], af_continuous_current[i], as_continuous_current[i],
            aflav_continuous_current[i], p_continuous[i, 1], p_continuous[i, 2],
            N_J, N_P, A_f, A_s, A_flav, P
        )
        for j in 1:N_J
            v_interp_2[j] += ω_C_post * (P_obs_cig[i, j]  - p_continuous[i, 1]) * q_cig[j] +
                              ω_E_post * (P_obs_ecig[i, j] - p_continuous[i, 2]) * q_ecig[j]
        end
        log_ll_2 += v_interp_2[y[i]] - logsumexp(v_interp_2)

        v_interp_3 = interpolate_v_choice(
            V_decision_3_post, tya_state[i], af_continuous_current[i], as_continuous_current[i],
            aflav_continuous_current[i], p_continuous[i, 1], p_continuous[i, 2],
            N_J, N_P, A_f, A_s, A_flav, P
        )
        for j in 1:N_J
            v_interp_3[j] += ω_C_post * (P_obs_cig[i, j]  - p_continuous[i, 1]) * q_cig[j] +
                              ω_E_post * (P_obs_ecig[i, j] - p_continuous[i, 2]) * q_ecig[j]
        end
        log_ll_3 += v_interp_3[y[i]] - logsumexp(v_interp_3)
    end

    # Mixture logsumexp across all three types
    hh_ll[h] = logsumexp([log_π_1_h + log_ll_1, log_π_2_h + log_ll_2, log_π_3_h + log_ll_3])
end

# Save _HH_LL.csv: household code and per-household log-likelihood
hh_ll_path = joinpath(estimates_subdir, "Dynamic_Model_Mixture_V2_$(psi_tag)_$(beta_tag)_$(psi_3_tag)_HH_LL.csv")
open(hh_ll_path, "w") do io
    println(io, "hh_code,ll")
    for h in 1:N_hh
        println(io, "$(unique_hh[h]),$(@sprintf("%.10f", hh_ll[h]))")
    end
end
log_msg("Per-household LL saved to: $hh_ll_path")


#-----------------------------------------------------------------------
# Rescaling Estimates to Original Units
#-----------------------------------------------------------------------
# The estimated parameters (opt_param) are in STANDARDIZED units because
# the data entering the utility function was standardized. To interpret
# the estimates in original units (utils per pack, utils per dollar,
# etc.), rescale as follows:
#
# Two-stock addiction: a = (ã_f + ã_s) / 2. At steady state, ã_f = ã_s = n_std
# = n_raw/n_max regardless of ψ, so a = n_raw/n_max. Rescaling γ_1 by ÷ n_max
# gives utils per mg of nicotine consumed per month (assumption-free at SS).
# Flavored habit: ã_flav = ψ_3 × a_flav_raw / n_flav_max, so rescaling γ_2, γ_3, γ_4
# by × ψ_3 ÷ n_flav_max gives utils per mL of raw flavored ecig stock.
#
# Price enters as ω · P · q̃ where q̃ is the standardized quantity. Rescaling
# ω_C, ω_E by ÷ q_max converts to utils per dollar × raw unit.
#
#   α_C_orig  = α_C_std  / q_cig_max           → utils per pack
#   α_E_orig  = α_E_std  / q_ecig_max          → utils per mL
#   α_CE_orig = α_CE_std / q_bundle_max        → utils per (pack × mL)
#   ω_C_orig  = ω_C_std  / q_cig_max           → utils per ($ × pack)
#   ω_E_orig  = ω_E_std  / q_ecig_max          → utils per ($ × mL)
#
#   γ_1_orig  = γ_1_std / n_max                 (utils per mg nicotine consumed at SS)
#   γ_2_orig  = γ_2_std × ψ_3 / n_flav_max  (utils per mL raw flavored ecig stock, ecig/bundle lock-in)
#   γ_3_orig  = γ_3_std × ψ_3 / n_flav_max  (utils per mL raw flavored ecig stock, cig/orig bundle lock-in)
#   γ_4_orig  = γ_4_std × ψ_3 / n_flav_max  (utils per mL raw flavored ecig stock, outside flavored withdrawal)
#
#   λ_1, λ_2, λ_3, λ_4 = no rescaling needed (multiply binary indicators)
#   ξ_C_k, ξ_E_k, ξ_CE_k = no rescaling needed (type-specific additive fixed effects, k=1,2,3)
#   π_0_2, π_TYA_2 = no rescaling needed (type 2 logit mixing weight params)
#   π_0_3, π_TYA_3 = no rescaling needed (type 3 logit mixing weight params)
#-----------------------------------------------------------------------

# Print and log rescaling
log_msg("")
log_msg("===================================")
log_msg("Rescaling to Original Units")
log_msg("===================================")
log_msg("Use these formulas to convert estimates to interpretable units:")
log_msg("")
log_msg("  α_C_orig  = α_C  / $q_cig_max")
log_msg("  α_E_orig  = α_E  / $q_ecig_max")
log_msg("  α_CE_orig = α_CE / $q_bundle_max")
log_msg("  ω_C_orig  = ω_C  / $q_cig_max                        (utils per dollar × pack)")
log_msg("  ω_E_orig  = ω_E  / $q_ecig_max                       (utils per dollar × mL)")
log_msg("  γ_1_orig  = γ_1 / n_max                             (utils per mg nicotine consumed at SS)")
log_msg("  γ_2_orig  = γ_2 × ψ_3 / $n_flav_max               (utils per mL raw flavored ecig stock, ecig/bundle lock-in)")
log_msg("  γ_3_orig  = γ_3 × ψ_3 / $n_flav_max               (utils per mL raw flavored ecig stock, cig/orig bundle lock-in)")
log_msg("  γ_4_orig  = γ_4 × ψ_3 / $n_flav_max               (utils per mL raw flavored ecig stock, outside flavored withdrawal)")
log_msg("  λ_1, λ_2, λ_3, λ_4: no rescaling needed")
log_msg("  ξ_C_1, ξ_E_1, ξ_CE_1: no rescaling needed (type 1 fixed effects)")
log_msg("  ξ_C_2, ξ_E_2, ξ_CE_2: no rescaling needed (type 2 fixed effects)")
log_msg("  ξ_C_3, ξ_E_3, ξ_CE_3: no rescaling needed (type 3 fixed effects)")
log_msg("  π_0_2, π_TYA_2: no rescaling needed (type 2 softmax logit params)")
log_msg("  π_0_3, π_TYA_3: no rescaling needed (type 3 softmax logit params)")

# Print and log estimation finished message
log_msg("")
log_msg("Dynamic estimation finished at $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
log_msg("Log saved to: $log_path")

# Close the log file handle
close(log_io)
