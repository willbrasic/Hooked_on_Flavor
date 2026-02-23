################################################################################
# William Brasic
# The University of Arizona
# wbrasic97@gmail.com
# February 2026
#
# This script estimates the structural parameters of the dynamic model by
# maximizing the sample log-likelihood via multi-start Nelder-Mead.
#
# Set ESTIMATE_BETA = true to estimate β (present bias) as a structural
# parameter. Set ESTIMATE_BETA = false for the base model
# with β fixed at 1.0.
#
# The objective function for each candidate θ:
#   1. Recomputes flow utility U given θ (or θ[1:end-1] when ESTIMATE_BETA)
#   2. Solves the value function via VFI (with β from θ[end] or fixed)
#   3. Computes the log-likelihood by trilinearly interpolating V_decision
#      at each observation's continuous (addiction, price) state
#   4. Returns the negative log-likelihood (since we minimize)
#
# Progress is logged to a timestamped log file in the output directory.
################################################################################


#############################
# Preliminaries
#############################

# Set to true to estimate β (present bias) as a structural parameter
ESTIMATE_BETA = false

# Set to true to estimate ψ (addiction decay rate) as a structural parameter
ESTIMATE_PSI = false

# Set to true to warm-start VFI from the previous evaluation's converged V
WARM_START = true

# Load all functions and packages from the functions file
include("./01_Functions.jl")

# Detect whether we are running on the HPC (any non-Windows system)
HPC = !Sys.iswindows()

# Create a timestamp for uniquely naming output files (log, estimates, outer try params)
timestamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")

# Construct psi and beta tags for output directory and file naming.
ψ_naming, β_naming, _ = get_fixed_parameters()
psi_tag = ESTIMATE_PSI ? "Psi_Estimated" : "Psi_$(ψ_naming)"
beta_tag = ESTIMATE_BETA ? "Beta_Estimated" : "Beta_$(β_naming)"

# Name for the estimation log file (includes psi and beta tags for identification)
est_log_name = "Dynamic_Model_$(psi_tag)_$(beta_tag)_Estimation_Log_$(timestamp).txt"

# File paths if on HPC or not
if HPC

    # Output path for results (use absolute path so it's unaffected by later cd)
    output_dir = abspath("./Dynamic_Model_$(psi_tag)_$(beta_tag)_Results")
    mkpath(output_dir)
    log_path = joinpath(output_dir, est_log_name)

    # Set working directory to where the data CSVs live
    cd("../Data")
else

    # Output path for results (local Windows path, includes beta tag in directory name)
    output_dir = "C:/Users/wbras/OneDrive/Documents/Desktop/UA/4th_Year_Paper/4th_Year_Paper_Data/HMS/2021-Onward/Dynamic_Model/Dynamic_Model_$(psi_tag)_$(beta_tag)_Results"
    mkpath(output_dir)
    log_path = joinpath(output_dir, est_log_name)

    # Set working directory to where the data CSVs live
    cd("C:/Users/wbras/OneDrive/Documents/Desktop/UA/4th_Year_Paper/4th_Year_Paper_Data/HMS/2021-Onward/Dynamic_Model/Data")
end

# Open log file for writing (log_io is defined as a global in 01_Functions.jl)
log_io = open(log_path, "w")

# Print and log the start time and number of Julia threads available for VFI parallelization
log_msg("Dynamic estimation ($(psi_tag), $(beta_tag)) started at $(timestamp)")
log_msg("")
log_msg("Number of threads: $(Threads.nthreads())")
log_msg("")

# Get household identifiers (pre-loaded to avoid repeated CSV reads in objective)
hh_codes = get_hh_codes();


#############################
# Initialize fixed parameters
#############################

# Load fixed parameters:
#   ψ = addiction decay rate (fixed at 0.68; overridden by optimizer when ESTIMATE_PSI = true)
#   β = present bias (fixed at 1.0; overridden by optimizer when ESTIMATE_BETA = true)
#   δ = monthly discount factor (fixed at 0.99)
ψ, β, δ = get_fixed_parameters();


#############################
# State Spaces and Choices
#############################

# Start timer for data prep
t_setup = time()

# Get number of addiction states (N_A = 20) and the normalized addiction grid A 
N_A, A = get_addiction_space(ψ);

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


#############################
# Demographics
#############################

# TYA states
tya_state = get_tya_states()

# TYA state transition matrix 
Π = get_tya_transitions()


#############################
# Price Space
#############################

# Get pricing grid: N_P points per category, P is N_P × 2 (cig, ecig)
N_P, P = get_pricing_spaces();

# Get all price combinations
N_Pcomb, Pcomb = get_pricing_spaces_combination(N_K, N_P, P);

# Get price ratios for quantity discount adjustment (price per unit varies by bin size)
ratio_cig, ratio_ecig = get_price_ratios(N_J, N_cig, N_orig_ecig, N_non_fda_flav_ecig, N_fda_flav_ecig, q_cig, q_ecig);

# Get expenditure matrix E[p, j] = p_cig(p) * q_cig[j] + p_ecig(p) * q_ecig[j]
# STANDARDIZED by E_max; E_max is the raw max value for rescaling estimates
E, E_max = get_expenditures(N_J, N_Pcomb, q_cig, q_ecig, q_cig_max, q_ecig_max, Pcomb, ratio_cig, ratio_ecig);

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
# p_continuous is N × 2 (cig price, ecig price) — actual per-unit prices, not grid indices
_, p_continuous = map_prices_to_grid(N_P, P, Pcomb);


#############################
# Pre-compute Addiction
# Objects (at Fixed ψ)
#############################

# Pre-compute addiction transition brackets for all (alternative, addiction state) pairs.
# When ESTIMATE_PSI = false, these are used by every objective evaluation (fixed ψ).
# When ESTIMATE_PSI = true, these are needed for setup but objective() recomputes
# them at each candidate ψ.
a_lower_fixed, a_upper_fixed, a_weight_fixed = precompute_addiction_transitions(N_J, N_A, ψ, A, n)

# Pre-compute initial addiction stocks via fixed-point iteration and simulate
# addiction trajectories. Same note as above: used directly when ψ is fixed,
# recomputed inside objective() when ESTIMATE_PSI = true.
a0_fixed, _ = get_initial_addiction_stock(ψ, A, n, y, hh_codes)
_, a_continuous_fixed = simulate_addiction_trajectories(N_A, ψ, A, n, y, hh_codes, a0_fixed)


# Log data setup completion time and sample size
setup_elapsed = time() - t_setup
log_msg("Data loading complete in $(round(setup_elapsed, digits=1))s")
log_msg("Observations: $(length(y))")


#############################
# Estimation
#############################

# Print and log standardization factors (needed to rescale parameter estimates)
log_msg("\nStandardization factors (divide estimates by these to get original units):")
log_msg("  q_cig_max    = $q_cig_max (packs)")
log_msg("  q_ecig_max   = $q_ecig_max (mL)")
log_msg("  q_bundle_max = $q_bundle_max (packs × mL) — actual max, not q_cig_max × q_ecig_max")
log_msg("  n_max        = $n_max (mg)")
log_msg("  E_max        = $E_max (\$)")
log_msg("  A_max        = $(maximum(A)) (normalized to [0, 1])")

# Starting values come from static logit estimates (ORIGINAL/UNSTANDARDIZED units)
static_logit_orig = (
    α_C  =  0.0180,     # utils per pack
    α_E  =  0.0089,     # utils per mL
    α_CE = -0.0016,     # utils per (pack × mL)
    λ_1  =  0.1778,     # flavor baseline — all flavored (multiplies indicator, no conversion)
    λ_2  =  0.5555,     # flavor × TYA — all flavored (multiplies indicator, no conversion)
    λ_3  =  -0.0976,    # FDA flavor baseline — additional for FDA-authorized (no conversion)
    λ_4  =  -0.3358,    # FDA flavor × TYA — additional for FDA-authorized (no conversion)
    ω    = -0.0055,     # utils per dollar
    ξ_C  = -3.573,      # cigarette fixed effect (additive, no conversion)
    ξ_E  = -6.500,      # e-cig fixed effect (additive, no conversion)
    ξ_CE = -5.288       # bundle fixed effect (additive, no conversion)
)

# Standardized starting values for the dynamic model
# γ and μ have no static logit counterpart
starting_param = (
    α_C  = static_logit_orig.α_C  * q_cig_max,
    α_E  = static_logit_orig.α_E  * q_ecig_max,
    α_CE = static_logit_orig.α_CE * q_bundle_max,
    λ_1  = static_logit_orig.λ_1,
    λ_2  = static_logit_orig.λ_2,
    λ_3  = static_logit_orig.λ_3,
    λ_4  = static_logit_orig.λ_4,
    γ    = -0.10,      # Withdrawal cost (negative); no static counterpart (ã ∈ [0,1])
    μ    =  0.10,      # Reinforcement effect; no static counterpart (ã ∈ [0,1])
    ω    = static_logit_orig.ω * E_max,
    ξ_C  = static_logit_orig.ξ_C,
    ξ_E  = static_logit_orig.ξ_E,
    ξ_CE = static_logit_orig.ξ_CE
)

# When estimating ψ, append it after the 13 structural parameters
if ESTIMATE_PSI
    starting_param = merge(starting_param, (ψ = 0.50,))
end

# When estimating β, append it as the last parameter (after ψ if both)
if ESTIMATE_BETA
    starting_param = merge(starting_param, (β = 0.50,))
end

# Initial simplex deviations for Nelder-Mead
add = [
    abs(starting_param.α_C)  * 0.50,   # α_C
    abs(starting_param.α_E)  * 0.50,   # α_E
    abs(starting_param.α_CE) * 0.50,   # α_CE
    abs(starting_param.λ_1)  * 0.50,   # λ_1
    abs(starting_param.λ_2)  * 0.50,   # λ_2
    abs(starting_param.λ_3)  * 0.50,   # λ_3
    abs(starting_param.λ_4)  * 0.50,   # λ_4
    abs(starting_param.γ)    * 1.00,   # γ:  larger since no informed start
    abs(starting_param.μ)    * 1.00,   # μ:  larger since no informed start
    abs(starting_param.ω)    * 0.50,   # ω
    abs(starting_param.ξ_C)  * 0.50,   # ξ_C
    abs(starting_param.ξ_E)  * 0.50,   # ξ_E
    abs(starting_param.ξ_CE) * 0.50    # ξ_CE
]

# When estimating ψ, append its simplex deviation (after 13 structural)
if ESTIMATE_PSI
    push!(add, 0.20)  # ψ: deviation of 0.20 around starting value of 0.50
end

# When estimating β, append its simplex deviation (always last)
if ESTIMATE_BETA
    push!(add, 0.20)  # β: deviation of 0.20 around starting value of 0.50
end

# Optimizer settings for random_amoeba multi-start Nelder-Mead
L          = 50    # Number of random restarts (outer tries)
M          = 20    # Short Nelder-Mead runs per outer try (inner tries)
inner_iter = 100   # Max iterations per short run

# Print and log optimizer settings
log_msg("")
log_msg("Optimizer settings: L=$L, M=$M, inner_iter=$inner_iter")

# Print and log starting values
log_msg("")
log_msg("Starting parameters:")
for (name, val) in pairs(starting_param)
    log_msg("  $name = $val")
end

# Print and log simplex deviations
log_msg("")
log_msg("Simplex deviations:")
pnames = replace.(collect(String, string.(keys(starting_param))),
    "α" => "alpha", "λ" => "lambda", "γ" => "gamma",
    "μ" => "mu", "ω" => "omega", "ξ" => "xi")
for (i, dev) in enumerate(add)
    log_msg("  $(pnames[i]) = $dev")
end

# Reset the global evaluation counter before estimation begins
est_eval_count = 0

# Set global parameter names so objective() can log parameter names alongside values
est_param_names = replace.(collect(String, string.(keys(starting_param))),
    "α" => "alpha", "λ" => "lambda", "γ" => "gamma",
    "μ" => "mu", "ω" => "omega", "ξ" => "xi")

# File paths for writing parameters after each outer try and inner try
outer_try_path = joinpath(output_dir, "Dynamic_Model_$(psi_tag)_$(beta_tag)_Outer_Try_Params_$(timestamp).csv")
inner_try_path = joinpath(output_dir, "Dynamic_Model_$(psi_tag)_$(beta_tag)_Inner_Try_Params_$(timestamp).csv")

# Run multi-start Nelder-Mead optimization
t_est = time()
opt_param, opt_value = random_amoeba(
    objective, starting_param, add, L, M;
    inner_iter = inner_iter,
    outer_try_file = outer_try_path,
    inner_try_file = inner_try_path
)
est_elapsed = time() - t_est

# Print and log estimation results
log_msg("\n\n")
log_msg("===================================")
log_msg("Estimation Results")
log_msg("===================================")
log_msg("Negative log-likelihood: $opt_value")
log_msg("Total evaluations: $est_eval_count")
log_msg("Total estimation time: $(round(est_elapsed, digits=1))s")
log_msg("Parameters:")
for (name, val) in pairs(opt_param)
    log_msg("  $name = $val")
end

# Save estimated parameters to a CSV file
estimates_path = joinpath(output_dir, "Dynamic_Model_$(psi_tag)_$(beta_tag)_Estimates.csv")
open(estimates_path, "w") do io
    println(io, join(replace.(collect(String, string.(keys(opt_param))),
        "α" => "alpha", "λ" => "lambda", "γ" => "gamma",
        "μ" => "mu", "ω" => "omega", "ξ" => "xi"), ","))
    println(io, join([@sprintf("%.10f", v) for v in values(opt_param)], ","))
end
log_msg("Estimates saved to: $estimates_path")
log_msg("Outer try parameters saved to: $outer_try_path")
log_msg("Inner try parameters saved to: $inner_try_path")


#-----------------------------------------------------------------------
# Rescaling Estimates to Original Units
#-----------------------------------------------------------------------
# The estimated parameters (opt_param) are in STANDARDIZED units because 
# the data entering the utility function was standardized. To interpret 
# the estimates in original units (utils per pack, utils per dollar, 
# etc.), rescale as follows:
#
#   α_C_orig  = α_C_std  / q_cig_max           → utils per pack
#   α_E_orig  = α_E_std  / q_ecig_max          → utils per mL
#   α_CE_orig = α_CE_std / q_bundle_max        → utils per (pack × mL)
#   ω_orig    = ω_std    / E_max               → utils per dollar
#
#   μ_orig    = μ_std × ψ / n_max²             → utils per (mg addiction × mg nicotine)
#       Note: The reinforcement term is μ·ã·n_std[j] where ã ∈ [0,1] is normalized
#       addiction (ã = ψ·a_raw/n_max) and n_std = n_raw/n_max. Since μ multiplies
#       BOTH ã and n_std, undoing both standardizations gives n_max² in denominator.
#       ψ enters because ã = ψ · a_raw / n_max.
#
#       Example: μ_std = 0.10, ψ = 0.48, n_max = 1500
#         μ_orig = 0.10 × 0.48 / 1500² ≈ 2.13e-8 utils per (mg × mg)
#         This looks tiny, but at raw values a_raw = 1500, n_raw = 1500:
#         μ_orig × a_raw × n_raw = 2.13e-8 × 1500 × 1500 ≈ 0.048
#         which equals μ_std × ã × n_std = 0.10 × 0.48 × 1.0 = 0.048
#
#   γ_orig    = γ_std × ψ / n_max              → utils per mg of addiction stock
#       Note: The withdrawal term is γ·ã where ã = ψ·a_raw/n_max. Since γ multiplies
#       only ã (not n_std), undoing the single standardization gives n_max in denominator.
#       ψ enters because ã = ψ · a_raw / n_max.
#
#       Example: γ_std = -0.10, ψ = 0.48, n_max = 1500
#         γ_orig = -0.10 × 0.48 / 1500 ≈ -3.20e-5 utils per mg
#         At raw value a_raw = 1500:
#         γ_orig × a_raw = -3.20e-5 × 1500 ≈ -0.048
#         which equals γ_std × ã = -0.10 × 0.48 = -0.048
#
#   λ_1, λ_2, λ_3, λ_4 = no rescaling needed (multiply binary indicators)
#   ξ_C, ξ_E, ξ_CE = no rescaling needed (additive constants)
#
# STANDARDIZATION FACTORS:
#   q_cig_max    = $(q_cig_max) packs
#   q_ecig_max   = $(q_ecig_max) mL
#   q_bundle_max = $(q_bundle_max) packs × mL (actual max, not q_cig_max × q_ecig_max)
#   n_max        = $(n_max) mg
#   E_max        = $(E_max) dollars
#-----------------------------------------------------------------------

# Print and log rescalling 
log_msg("\n")
log_msg("===================================")
log_msg("Rescaling to Original Units")
log_msg("===================================")
log_msg("Use these formulas to convert estimates to interpretable units:")
log_msg("  α_C_orig  = α_C  / $q_cig_max")
log_msg("  α_E_orig  = α_E  / $q_ecig_max")
log_msg("  α_CE_orig = α_CE / $q_bundle_max")
log_msg("  ω_orig    = ω    / $E_max")
log_msg("  μ_orig    = μ × ψ̂ / n_max² (reinforcement: μ·ã·n_std[j]; ã = ψ·a_raw/n_max, n_std = n_raw/n_max)")
log_msg("  γ_orig    = γ × ψ̂ / n_max  (withdrawal: γ·ã; ã = ψ·a_raw/n_max)")
log_msg("  λ_1, λ_2, λ_3, λ_4, ξ_C, ξ_E, ξ_CE: no rescaling needed")
if ESTIMATE_PSI
    log_msg("  ψ = no rescaling needed")
end
if ESTIMATE_BETA
    log_msg("  β = no rescaling needed")
end

# Print and log estimation finished message 
log_msg("\nEstimation finished at $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
log_msg("Log saved to: $log_path")

# Close the log file handle
close(log_io)

