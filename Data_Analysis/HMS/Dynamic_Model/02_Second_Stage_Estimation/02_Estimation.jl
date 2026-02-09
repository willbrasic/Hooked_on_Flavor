################################################################################
# William Brasic
# The University of Arizona
# wbrasic97@gmail.com
# December 2025
#
# This script estimates the structural parameters of the dynamic model by
# maximizing the sample log-likelihood via multi-start Nelder-Mead.
#
# The objective function for each candidate θ:
#   1. Recomputes flow utility U given θ
#   2. Solves the value function via VFI (starts fresh from zeros each evaluation)
#   3. Computes the log-likelihood by trilinearly interpolating V_decision
#      at each observation's continuous (addiction, price) state
#   4. Returns the negative log-likelihood (since we minimize)
#
# Progress is logged to Estimation_Log.txt.
################################################################################


#############################
# Preliminaries
#############################

# Load all functions and packages
include("./01_Functions.jl")

# Auto-detect HPC: Windows = local, Linux = HPC
hpc = !Sys.iswindows()

# Load all functions and packages, set ouput path, and set working directory
using Dates
timestamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
est_log_name = "Dynamic_Model_Estimation_Log_$(timestamp).txt"
if hpc

    # Output path for results (use absolute path so it's unaffected by later cd)
    output_dir = abspath("./Dynamic_Model_Results")
    log_path = joinpath(output_dir, est_log_name)

    # Set working directory to where the data CSVs live
    cd("../Data")
else

    # Output path for results
    output_dir = "C:/Users/wbras/OneDrive/Documents/Desktop/UA/4th_Year_Paper/4th_Year_Paper_Data/HMS/2021-Onward/Dynamic_Model/Dynamic_Model_Results"
    log_path = joinpath(output_dir, est_log_name)

    # Set working directory to where the data CSVs live
    cd("C:/Users/wbras/OneDrive/Documents/Desktop/UA/4th_Year_Paper/4th_Year_Paper_Data/HMS/2021-Onward/Dynamic_Model/Data")
end

# Open log file and set global handle
est_log_io = open(log_path, "w")
est_log("Dynamic estimation started at $(timestamp)")

# Get household identifiers (pre-loaded to avoid repeated CSV reads in objective)
hh_codes = get_hh_codes();

# Print number of threads 
est_log("Number of threads is: $(Threads.nthreads())");


#############################
# Initialize fixed parameters
#############################

# Load fixed parameters (ψ is now estimated; only β and δ are fixed)
_, β, δ = get_fixed_parameters();


#############################
# State Spaces and Choices
#############################

# Start timer
t_setup = time()

# Get number of addiction states and the addiction grid
# Initial grid uses ψ=0.94; objective() recomputes A at each eval using ψ from θ_vec
N_A, A = get_addiction_space(0.94);

# Get number of alternatives (N_J) and choice matrix (J)
_, N_J, J = get_product_choices();

# Get choice vector (y[i] = chosen alternative index for observation i)
y = get_hh_choices(J);

# Get number of categories excluding outside option (N_K)
N_K, _ = get_category_choices();


#############################
# Alternative-Level Vectors
#############################

# Get consumption vectors by alternative (STANDARDIZED by max)
# c_bundle is standardized by its own max (not c_cig_max × c_ecig_max) for reasonable α_TE scaling
# Max values are needed for rescaling parameter estimates to original units
N_cig, N_orig_ecig, N_flav_ecig, _, c_cig, c_ecig, c_bundle, c_cig_max, c_ecig_max, c_bundle_max = get_consumption(N_J);

# Get nicotine vector by alternative (STANDARDIZED by max)
# n_max is the raw max value for rescaling estimates
n, n_max = get_nicotine(N_J);

# Get category index by alternative: cat_idx[j] ∈ {0, 1, 2, 3, 4, 5}
cat_idx = get_category_index(N_J, N_cig, N_orig_ecig, N_flav_ecig);

# Get flavored indicator by alternative: is_flavored[j] ∈ {true, false}
is_flavored = get_flavored_indicator(cat_idx);


#############################
# Demographics
#############################

# Get TYA binary indicator for each observation
_, tya = get_teen_young_adult()

# Get TYA state index for each observation (1 = no TYA, 2 = TYA present)
tya_state = get_tya_state(tya)


#############################
# Price Space
#############################

# Get pricing grid (N_P points per category)
N_P, P = get_pricing_spaces();

# Get all price combinations: N_Pcomb = N_P^2, Pcomb is N_Pcomb × 2
N_Pcomb, Pcomb = get_pricing_spaces_combination(N_K, N_P, P);

# Get expenditure matrix (STANDARDIZED by max)
# E_max is the raw max value for rescaling estimates
E, E_max = get_expenditures(N_J, N_Pcomb, c_cig, c_ecig, c_cig_max, c_ecig_max, Pcomb);

# Get price transitions from Halton draws: T[m, r, k]
T = get_transitions(N_K);

# Pre-compute price transition brackets and interpolation weights
p_cig_lo, p_cig_hi, p_cig_w, p_ecig_lo, p_ecig_hi, p_ecig_w = precompute_price_transitions(N_P, P, T);


#############################
# Household Price Trajectories
#############################

# Map observed prices to continuous values for likelihood interpolation
# p_continuous: actual continuous prices N × 2 (cig, ecig)
_, p_continuous = map_prices_to_grid(N_P, P, Pcomb)

setup_elapsed = time() - t_setup
est_log("Data loading complete in $(round(setup_elapsed, digits=1))s")
est_log("Observations: $(length(y))")

# Log standardization factors (needed to rescale parameter estimates)
est_log("\nStandardization factors (divide estimates by these to get original units):")
est_log("  c_cig_max    = $c_cig_max (packs)")
est_log("  c_ecig_max   = $c_ecig_max (mL)")
est_log("  c_bundle_max = $c_bundle_max (packs × mL) — actual max, not c_cig_max × c_ecig_max")
est_log("  n_max        = $n_max (mg)")
est_log("  E_max        = $E_max (\$)")
est_log("  A_max        = $(maximum(A)) (standardized, raw would be $(n_max / ψ))")


#############################
# Estimation
#############################

#-----------------------------------------------------------------------
# Starting Parameter Values
#-----------------------------------------------------------------------
# The starting values come from a static logit model (03_Static_Logit.jl)
# estimated with UNSTANDARDIZED data. Since the dynamic model uses STANDARDIZED
# data (all continuous variables divided by their max), we must convert the
# static logit estimates to "standardized units" before using them here.
#
# Why conversion is needed:
#   With unstandardized data: utility contribution = α_orig × x_raw
#   With standardized data:   utility contribution = α_std × (x_raw / x_max)
#   For the same x_raw to give the same utility: α_std = α_orig × x_max
#
# Example: If static logit gave α_T = 0.0077 (utils per pack) and c_cig_max = 60:
#   - Unstandardized: 60 packs → 0.0077 × 60 = 0.462 utils
#   - Standardized:   c_cig = 1.0 → α_T_std × 1.0 should also = 0.462 utils
#   - Therefore:      α_T_std = 0.0077 × 60 = 0.462
#-----------------------------------------------------------------------

# Static logit estimates from 03_Static_Logit.jl (ORIGINAL/UNSTANDARDIZED units)
# These were estimated using raw consumption (packs, mL) and raw expenditure ($)
static_logit_orig = (
    α_T  =  0.0077,   # utils per pack
    α_E  =  0.0072,   # utils per mL
    α_TE =  0.0094,   # utils per (pack × mL)
    λ_1  =  0.6704,   # baseline flavor effect (multiplies indicator, no conversion)
    λ_2  =  0.4067,   # flavor × TYA interaction (multiplies indicator, no conversion)
    ω    = -0.0032,   # utils per dollar
    ξ_T  = -3.6104,   # cigarette fixed effect (additive, no conversion)
    ξ_E  = -5.4592,   # e-cig fixed effect (additive, no conversion)
    ξ_TE = -6.0470    # bundle fixed effect (additive, no conversion)
)

# Convert static logit estimates to STANDARDIZED units for the dynamic model
# Conversion formulas:
#   α_T_std  = α_T_orig  × c_cig_max           (consumption term)
#   α_E_std  = α_E_orig  × c_ecig_max          (consumption term)
#   α_TE_std = α_TE_orig × c_bundle_max        (interaction term; c_bundle_max is actual max, not c_cig_max × c_ecig_max)
#   ω_std    = ω_orig    × E_max               (expenditure term)
#   λ, ξ     = no conversion (multiply indicators or are additive constants)
#
# μ and γ have no static logit counterpart (they are dynamic-only parameters).
# Initialized at reasonable magnitudes for standardized data where a ∈ [0, ~1].
# μ > 0: higher addiction increases utility of any tobacco use (reinforcement)
# γ < 0: higher addiction decreases utility when not consuming (withdrawal cost)
# ψ is the addiction decay rate, estimated jointly with structural parameters.
starting_param = (
    α_T  = static_logit_orig.α_T  * c_cig_max,
    α_E  = static_logit_orig.α_E  * c_ecig_max,
    α_TE = static_logit_orig.α_TE * c_bundle_max,
    λ_1  = static_logit_orig.λ_1,
    λ_2  = static_logit_orig.λ_2,
    μ    =  0.05,      # Reinforcement effect; no static counterpart
    γ    = -0.05,      # Withdrawal cost (negative); no static counterpart
    ω    = static_logit_orig.ω * E_max,
    ξ_T  = static_logit_orig.ξ_T,
    ξ_E  = static_logit_orig.ξ_E,
    ξ_TE = static_logit_orig.ξ_TE,
    ψ    =  0.50       # Addiction decay rate; start at midpoint of (0, 1)
)

# Initial simplex deviations for Nelder-Mead
# Scaled to ~50-100% of the absolute value of each starting parameter
add = [
    abs(starting_param.α_T)  * 0.50,   # α_T
    abs(starting_param.α_E)  * 0.50,   # α_E
    abs(starting_param.α_TE) * 0.50,   # α_TE
    abs(starting_param.λ_1)  * 0.50,   # λ_1
    abs(starting_param.λ_2)  * 0.50,   # λ_2
    abs(starting_param.μ)    * 1.00,   # μ:  larger since no informed start
    abs(starting_param.γ)    * 1.00,   # γ:  larger since no informed start
    abs(starting_param.ω)    * 0.50,   # ω
    abs(starting_param.ξ_T)  * 0.50,   # ξ_T
    abs(starting_param.ξ_E)  * 0.50,   # ξ_E
    abs(starting_param.ξ_TE) * 0.50,   # ξ_TE
    0.20                                # ψ: ±0.10 around starting value
]

# Optimizer settings
L          = 50    # Number of random restarts (outer tries)
M          = 20    # Short Nelder-Mead runs per outer try (inner tries)
inner_iter = 100  # Max iterations per short run

# Log optimizer settings
est_log("")
est_log("Optimizer settings: L=$L, M=$M, inner_iter=$inner_iter")

est_log("")
est_log("Starting parameters:")
for (name, val) in pairs(starting_param)
    est_log("  $name = $val")
end

est_log("")
est_log("Simplex deviations:")
pnames = collect(String, string.(keys(starting_param)))
for (i, dev) in enumerate(add)
    est_log("  $(pnames[i]) = $dev")
end

# Reset evaluation counter
est_eval_count = 0

# Set global parameter names for objective function logging
est_param_names = collect(String, string.(keys(starting_param)))

# File path for writing parameters after each outer try
outer_try_path = joinpath(output_dir, "Dynamic_Model_Outer_Try_Params_$(timestamp).txt")

# Run multi-start Nelder-Mead
t_est = time()
opt_param, opt_value = random_amoeba(
    objective, starting_param, add, L, M, inner_iter;
    log_io = est_log_io,
    outer_try_file = outer_try_path
)
est_elapsed = time() - t_est

# Report results
est_log("\n\n")
est_log("===================================")
est_log("Estimation Results")
est_log("===================================")
est_log("Negative log-likelihood: $opt_value")
est_log("Total evaluations: $est_eval_count")
est_log("Total estimation time: $(round(est_elapsed, digits=1))s")
est_log("Parameters:")
for (name, val) in pairs(opt_param)
    est_log("  $name = $val")
end

# Save estimated parameters to file for SE computation in 03_Standard_Errors.jl
estimates_path = joinpath(output_dir, "Dynamic_Model_Estimates.txt")
open(estimates_path, "w") do io
    println(io, join(collect(String, string.(keys(opt_param))), "\t"))
    println(io, join([@sprintf("%.10f", v) for v in values(opt_param)], "\t"))
end
est_log("Estimates saved to: $estimates_path")
est_log("Outer try parameters saved to: $outer_try_path")


#-----------------------------------------------------------------------
# Rescaling Estimates to Original Units
#-----------------------------------------------------------------------
# The estimated parameters (opt_param) are in STANDARDIZED units because the
# data entering the utility function was standardized. To interpret the estimates
# in original units (utils per pack, utils per dollar, etc.), rescale as follows:
#
# RESCALING FORMULAS (divide by max to convert standardized → original):
#
#   α_T_orig  = α_T_std  / c_cig_max           → utils per pack
#   α_E_orig  = α_E_std  / c_ecig_max          → utils per mL
#   α_TE_orig = α_TE_std / c_bundle_max        → utils per (pack × mL)
#   ω_orig    = ω_std    / E_max               → utils per dollar
#
#   μ_orig    = μ_std    / n_max               → utils per mg addiction
#       Note: The reinforcement term is μ·a·𝟙[j ≠ outside] where a is
#       standardized by n_max (since a_max = n_max/ψ and A grid uses n_max=1).
#
#   γ_orig    = γ_std    / n_max               → utils per mg addiction
#       Note: a is scaled by n_max (since a_max = n_max/ψ and A grid uses n_max=1)
#
#   λ_1, λ_2  = no rescaling needed (multiply binary indicators)
#   ξ_T, ξ_E, ξ_TE = no rescaling needed (additive constants)
#   ψ = no rescaling needed (dimensionless decay rate)
#
# STANDARDIZATION FACTORS (logged above, repeated here for convenience):
#   c_cig_max    = $(c_cig_max) packs
#   c_ecig_max   = $(c_ecig_max) mL
#   c_bundle_max = $(c_bundle_max) packs × mL (actual max, not c_cig_max × c_ecig_max)
#   n_max        = $(n_max) mg
#   E_max        = $(E_max) dollars
#-----------------------------------------------------------------------

est_log("\n")
est_log("===================================")
est_log("Rescaling to Original Units")
est_log("===================================")
est_log("Use these formulas to convert estimates to interpretable units:")
est_log("  α_T_orig  = α_T  / $c_cig_max")
est_log("  α_E_orig  = α_E  / $c_ecig_max")
est_log("  α_TE_orig = α_TE / $c_bundle_max")
est_log("  ω_orig    = ω    / $E_max")
est_log("  μ_orig    = μ    / $n_max (reinforcement: μ·a·𝟙[j≠outside], a standardized by n_max)")
est_log("  γ_orig    = γ    / $n_max")
est_log("  λ_1, λ_2, ξ_T, ξ_E, ξ_TE, ψ: no rescaling needed")

est_log("\nEstimation finished at $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
est_log("Log saved to: $log_path")

# Close log file
close(est_log_io)







