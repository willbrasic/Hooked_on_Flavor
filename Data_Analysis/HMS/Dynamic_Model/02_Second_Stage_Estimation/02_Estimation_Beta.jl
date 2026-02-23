################################################################################
# William Brasic
# The University of Arizona
# wbrasic97@gmail.com
# February 2026
#
# Beta estimation variant of 02_Estimation.jl. Estimates β (present bias) as
# the 14th structural parameter and uses the 4-state TYA classification with
# transition matrix Π_tya.
#
# Changes from 02_Estimation.jl:
#   - Includes 01_Functions_Beta.jl instead of 01_Functions.jl
#   - get_fixed_parameters() returns (ψ, δ) only (no β; β is estimated)
#   - Uses get_tya_states() for 4-state TYA instead of binary get_tya_state()
#   - Loads Π_tya via get_tya_transitions()
#   - θ has 14 parameters (13 structural + β as the 14th)
#   - Output files named with "Beta" prefix
#
# The objective function for each candidate θ:
#   1. Recomputes flow utility U given θ[1:13]
#   2. Solves the value function via VFI with β = θ[14] and Π_tya
#   3. Computes the log-likelihood by trilinearly interpolating V_decision
#      at each observation's continuous (addiction, price) state
#   4. Returns the negative log-likelihood (since we minimize)
#
# Progress is logged to a timestamped log file in the output directory.
################################################################################


#############################
# Preliminaries
#############################

# Load all functions and packages from the Beta functions file
include("./01_Functions_Beta.jl")

# Detect whether we are running on the HPC (any non-Windows system)
hpc = !Sys.iswindows()

# Create a timestamp for uniquely naming output files (log, estimates, outer try params)
timestamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")

# Name for the estimation log file
est_log_name = "Dynamic_Model_Beta_Estimation_Log_$(timestamp).txt"

if hpc

    # Output path for results (use absolute path so it's unaffected by later cd)
    output_dir = abspath("./Dynamic_Model_Results")
    log_path = joinpath(output_dir, est_log_name)

    # Set working directory to where the data CSVs live
    cd("../Data")
else

    # Output path for results (local Windows path)
    output_dir = "C:/Users/wbras/OneDrive/Documents/Desktop/UA/4th_Year_Paper/4th_Year_Paper_Data/HMS/2021-Onward/Dynamic_Model/Dynamic_Model_Results"
    log_path = joinpath(output_dir, est_log_name)

    # Set working directory to where the data CSVs live
    cd("C:/Users/wbras/OneDrive/Documents/Desktop/UA/4th_Year_Paper/4th_Year_Paper_Data/HMS/2021-Onward/Dynamic_Model/Data")
end

# Open log file for writing (log_io is defined as a global in 01_Functions_Beta.jl)
log_io = open(log_path, "w")

# Print and log the start time and number of Julia threads available for VFI parallelization
log_msg("Dynamic Beta estimation started at $(timestamp)")
log_msg("Number of threads: $(Threads.nthreads())")

# Get household identifiers (pre-loaded to avoid repeated CSV reads in objective)
hh_codes = get_hh_codes();


#############################
# Initialize fixed parameters
#############################

# Load fixed parameters:
#   ψ = addiction decay rate (fixed from reduced-form AR(1) estimate in 05_Reduced_Form_Psi.R)
#   δ = monthly discount factor (fixed at 0.99)
# NOTE: β is NOT returned here — it is estimated as the 14th parameter
ψ, δ = get_fixed_parameters();


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
# c_bundle is standardized by its own max (not c_cig_max × c_ecig_max) for reasonable α_TE scaling
# Max values are needed for rescaling parameter estimates to original units
N_cig, N_orig_ecig, N_non_fda_flav_ecig, N_fda_flav_ecig, _, c_cig, c_ecig, c_bundle, c_cig_max, c_ecig_max, c_bundle_max = get_consumption(N_J);

# Get nicotine vector by alternative (STANDARDIZED by max)
# n_max is the raw max value for rescaling estimates
n, n_max = get_nicotine(N_J);

# Get category index by alternative
cat_idx = get_category_index(N_J, N_cig, N_orig_ecig, N_non_fda_flav_ecig, N_fda_flav_ecig);

# Get non-FDA flavored indicator by alternative: is_non_fda_flavored[j] ∈ {0, 1}
is_non_fda_flavored = get_non_fda_flavored_indicator(cat_idx);

# Get FDA flavored indicator by alternative: is_fda_flavored[j] ∈ {0, 1}
is_fda_flavored = get_fda_flavored_indicator(cat_idx);


#############################
# Demographics
#############################

# Get 4-state TYA classification for each observation (states 1-4)
# State 1: No TYA, stable (oldest child ≤ 10 or no children)
# State 2: No TYA, approaching (oldest child 11-12)
# State 3: TYA present, stable (youngest TYA member ≤ 23)
# State 4: TYA present, ending soon (youngest TYA member ≥ 24)
tya_state = get_tya_states();

# Load 4×4 monthly TYA transition matrix Π_tya where Π_tya[s, s'] = P(TYA' = s' | TYA = s)
# Used in VFI to integrate over anticipated TYA state changes (key for β identification)
Π_tya = get_tya_transitions();


#############################
# Price Space
#############################

# Get pricing grid: N_P points per category, P is N_P × 2 (cig, ecig)
N_P, P = get_pricing_spaces();

# Get all price combinations
N_Pcomb, Pcomb = get_pricing_spaces_combination(N_K, N_P, P);

# Get price ratios for quantity discount adjustment (price per unit varies by bin size)
ratio_cig, ratio_ecig = get_price_ratios(N_J, N_cig, N_orig_ecig, N_non_fda_flav_ecig, N_fda_flav_ecig, c_cig, c_ecig);

# Get expenditure matrix E[p, j] = p_cig(p) * c_cig[j] + p_ecig(p) * c_ecig[j]
# STANDARDIZED by E_max; E_max is the raw max value for rescaling estimates
E, E_max = get_expenditures(N_J, N_Pcomb, c_cig, c_ecig, c_cig_max, c_ecig_max, Pcomb, ratio_cig, ratio_ecig);

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

# Log data setup completion time and sample size
setup_elapsed = time() - t_setup;
log_msg("Data loading complete in $(round(setup_elapsed, digits=1))s")
log_msg("Observations: $(length(y))")


#############################
# Estimation
#############################

# Print and log standardization factors (needed to rescale parameter estimates)
log_msg("\nStandardization factors (divide estimates by these to get original units):")
log_msg("  c_cig_max    = $c_cig_max (packs)")
log_msg("  c_ecig_max   = $c_ecig_max (mL)")
log_msg("  c_bundle_max = $c_bundle_max (packs × mL) — actual max, not c_cig_max × c_ecig_max")
log_msg("  n_max        = $n_max (mg)")
log_msg("  E_max        = $E_max (\$)")
log_msg("  A_max        = $(maximum(A)) (normalized to [0, 1])")

# Starting values come from static logit estimates (ORIGINAL/UNSTANDARDIZED units)
static_logit_orig = (
    α_T  =  0.0187,   # utils per pack
    α_E  =  0.0096,   # utils per mL
    α_TE = -0.0021,   # utils per (pack × mL)
    λ_1  =  0.852,    # non-FDA baseline flavor effect (multiplies indicator, no conversion)
    λ_2  =  0.703,    # non-FDA flavor × TYA interaction (multiplies indicator, no conversion)
    λ_3  =  0.5,      # FDA baseline flavor effect (multiplies indicator, no conversion)
    λ_4  =  0.5,      # FDA flavor × TYA interaction (multiplies indicator, no conversion)
    ω    = -0.0055,   # utils per dollar
    ξ_T  = -3.573,    # cigarette fixed effect (additive, no conversion)
    ξ_E  = -6.500,    # e-cig fixed effect (additive, no conversion)
    ξ_TE = -5.288     # bundle fixed effect (additive, no conversion)
)

# Standardized starting values for the dynamic model
# μ and γ have no static logit counterpart (they are dynamic-only parameters).
# β is the 14th parameter: present bias, initialized at 0.90 (mild present bias).
starting_param = (
    α_T  = static_logit_orig.α_T  * c_cig_max,
    α_E  = static_logit_orig.α_E  * c_ecig_max,
    α_TE = static_logit_orig.α_TE * c_bundle_max,
    λ_1  = static_logit_orig.λ_1,
    λ_2  = static_logit_orig.λ_2,
    λ_3  = static_logit_orig.λ_3,
    λ_4  = static_logit_orig.λ_4,
    μ    =  0.10,      # Reinforcement effect; no static counterpart (ã ∈ [0,1])
    γ    = -0.10,      # Withdrawal cost (negative); no static counterpart (ã ∈ [0,1])
    ω    = static_logit_orig.ω * E_max,
    ξ_T  = static_logit_orig.ξ_T,
    ξ_E  = static_logit_orig.ξ_E,
    ξ_TE = static_logit_orig.ξ_TE,
    β    =  0.90       # Present bias; 1.0 = exponential, < 1.0 = present-biased
)

# Initial simplex deviations for Nelder-Mead
add = [
    abs(starting_param.α_T)  * 0.50,   # α_T
    abs(starting_param.α_E)  * 0.50,   # α_E
    abs(starting_param.α_TE) * 0.50,   # α_TE
    abs(starting_param.λ_1)  * 0.50,   # λ_1
    abs(starting_param.λ_2)  * 0.50,   # λ_2
    abs(starting_param.λ_3)  * 0.50,   # λ_3
    abs(starting_param.λ_4)  * 0.50,   # λ_4
    abs(starting_param.μ)    * 1.00,   # μ:  larger since no informed start
    abs(starting_param.γ)    * 1.00,   # γ:  larger since no informed start
    abs(starting_param.ω)    * 0.50,   # ω
    abs(starting_param.ξ_T)  * 0.50,   # ξ_T
    abs(starting_param.ξ_E)  * 0.50,   # ξ_E
    abs(starting_param.ξ_TE) * 0.50,   # ξ_TE
    0.10                               # β:  deviation of 0.10 around starting value of 0.90
]

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
pnames = collect(String, string.(keys(starting_param)))
for (i, dev) in enumerate(add)
    log_msg("  $(pnames[i]) = $dev")
end

# Reset the global evaluation counter before estimation begins
est_eval_count = 0

# Set global parameter names so objective() can log parameter names alongside values
est_param_names = collect(String, string.(keys(starting_param)))

# File paths for writing parameters after each outer try and inner try
outer_try_path = joinpath(output_dir, "Dynamic_Model_Beta_Outer_Try_Params_$(timestamp).csv")
inner_try_path = joinpath(output_dir, "Dynamic_Model_Beta_Inner_Try_Params_$(timestamp).csv")

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
estimates_path = joinpath(output_dir, "Dynamic_Model_Beta_Estimates.csv")
open(estimates_path, "w") do io
    println(io, join(collect(String, string.(keys(opt_param))), ","))
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
#   α_T_orig  = α_T_std  / c_cig_max           → utils per pack
#   α_E_orig  = α_E_std  / c_ecig_max          → utils per mL
#   α_TE_orig = α_TE_std / c_bundle_max        → utils per (pack × mL)
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
#   ξ_T, ξ_E, ξ_TE = no rescaling needed (additive constants)
#   β = no rescaling needed (dimensionless discount parameter ∈ [0.01, 1.00])
#   ψ = fixed from reduced-form AR(1) estimate (not estimated)
#
# STANDARDIZATION FACTORS (logged above, repeated here for convenience):
#   c_cig_max    = $(c_cig_max) packs
#   c_ecig_max   = $(c_ecig_max) mL
#   c_bundle_max = $(c_bundle_max) packs × mL (actual max, not c_cig_max × c_ecig_max)
#   n_max        = $(n_max) mg
#   E_max        = $(E_max) dollars
#-----------------------------------------------------------------------

# Print and log rescalling
log_msg("\n")
log_msg("===================================")
log_msg("Rescaling to Original Units")
log_msg("===================================")
log_msg("Use these formulas to convert estimates to interpretable units:")
log_msg("  α_T_orig  = α_T  / $c_cig_max")
log_msg("  α_E_orig  = α_E  / $c_ecig_max")
log_msg("  α_TE_orig = α_TE / $c_bundle_max")
log_msg("  ω_orig    = ω    / $E_max")
log_msg("  μ_orig    = μ × ψ̂ / n_max² (reinforcement: μ·ã·n_std[j]; ã = ψ·a_raw/n_max, n_std = n_raw/n_max)")
log_msg("  γ_orig    = γ × ψ̂ / n_max  (withdrawal: γ·ã; ã = ψ·a_raw/n_max)")
log_msg("  λ_1, λ_2, λ_3, λ_4, ξ_T, ξ_E, ξ_TE: no rescaling needed")
log_msg("  β = no rescaling needed (dimensionless)")
log_msg("  ψ = $ψ (fixed from reduced-form AR(1) estimate)")

# Print and log estimation finished message
log_msg("\nEstimation finished at $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
log_msg("Log saved to: $log_path")

# Close the log file handle
close(log_io)
