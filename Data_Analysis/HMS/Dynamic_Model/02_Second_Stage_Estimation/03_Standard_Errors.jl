################################################################################
# William Brasic
# The University of Arizona
# wbrasic97@gmail.com
# January 2026
#
# This script computes standard errors for the dynamic model parameter
# estimates via finite differences on the full objective function.
#
# The Hessian H of the negative log-likelihood is approximated numerically
# using parameter-relative step sizes h_k = max(|θ_hat[k]| * 1e-4, 1e-5):
#   - Diagonal:     H[k,k] ≈ (f(θ+h_k·e_k) - 2·f(θ) + f(θ-h_k·e_k)) / h_k^2
#   - Off-diagonal: H[k,l] ≈ (f(θ+h_k·e_k+h_l·e_l) - f(θ+h_k·e_k-h_l·e_l)
#                            -  f(θ-h_k·e_k+h_l·e_l) + f(θ-h_k·e_k-h_l·e_l)) / (4·h_k·h_l)
#
# Each evaluation of f(θ) requires solving VFI from scratch, so this is slow
# but gives correct SEs that account for how V_choice changes with θ.
#
# Variance-covariance matrix: H^{-1}
# Standard errors: SE[k] = sqrt(H^{-1}[k,k])
#
# Reads estimated parameters from Dynamic_Model_<beta_tag>_Estimates.csv
# (produced by 02_Estimation.jl). Progress is logged to
# Dynamic_Model_<beta_tag>_SE_Log.txt.
################################################################################


#############################
# Preliminaries
#############################

# Base model: β is fixed at 1.0 (not estimated)
ESTIMATE_BETA = false

# Base model: ψ is fixed at 0.68 (not estimated)
ESTIMATE_PSI = false

# Enable warm-start VFI: reuse the previous evaluation's converged V as the
# initial guess. The small θ perturbations (h ~ 1e-4) mean the previous V is
# an excellent starting point, reducing VFI iterations from ~948 to ~7-50.
WARM_START = true

# Load all functions and packages from the functions file
include("01_Functions.jl")
using LinearAlgebra

# Initialize warm-start globals for SE computation. Since we are not using
# random_amoeba, the phase never changes, so V_warm_est is reused across
# all objective evaluations. The small θ perturbations (h ~ 1e-4) mean
# the previous V is an excellent warm-start.
ra_outer_try = 1
ra_inner_run = 1
V_warm_est = nothing
last_ra_phase_est = (1, 1)

# Construct psi and beta tags for output directory and file naming.
ψ_naming, β_naming, _ = get_fixed_parameters()
psi_tag = ESTIMATE_PSI ? "Psi_Estimated" : "Psi_$(ψ_naming)"
beta_tag = ESTIMATE_BETA ? "Beta_Estimated" : "Beta_$(β_naming)"

# Set working directory to where the data CSVs live
cd("C:/Users/wbras/OneDrive/Documents/Desktop/UA/4th_Year_Paper/4th_Year_Paper_Data/HMS/2021-Onward/Dynamic_Model/Data")


#############################
# Output Paths
#############################

# Output path for results (local Windows path, includes beta tag in directory name)
output_dir = "C:/Users/wbras/OneDrive/Documents/Desktop/UA/4th_Year_Paper/4th_Year_Paper_Data/HMS/2021-Onward/Dynamic_Model/Dynamic_Model_$(psi_tag)_$(beta_tag)_Results"
mkpath(output_dir)
log_path = joinpath(output_dir, "Dynamic_Model_$(psi_tag)_$(beta_tag)_SE_Log.txt")

# Open log file for writing (log_io is defined as a global in 01_Functions.jl)
log_io = open(log_path, "w")

# Print and log SE computation start time
log_msg("SE computation ($(psi_tag), $(beta_tag)) started at $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")


#############################
# State Spaces and Choices
#############################

# Start timer for data prep
t_setup = time()

# Load fixed parameters:
#   ψ = addiction decay rate (fixed at 0.68; overridden when ESTIMATE_PSI = true)
#   β = present bias (fixed at 1.0; overridden by optimizer when ESTIMATE_BETA = true)
#   δ = monthly discount factor (fixed at 0.99)
ψ, β, δ = get_fixed_parameters();

# Get number of addiction states (N_A = 20) and the normalized addiction grid A
N_A, A = get_addiction_space(ψ);

# Get number of observations (N_HHT), number of alternatives (N_J), and choice matrix J
_, N_J, J = get_product_choices();

# Convert choice matrix J to choice vector y where y[i] = chosen alternative index for observation i
y = get_hh_choices(J);

# Get household identifiers (pre-loaded to avoid repeated CSV reads in objective)
hh_codes = get_hh_codes();

# Get number of product categories excluding the outside option
N_K, _ = get_category_choices();


#############################
# Alternative-Level Vectors
#############################

# Get consumption vectors by alternative (STANDARDIZED by max)
# q_bundle is standardized by its own max (not q_cig_max × q_ecig_max) for reasonable α_CE scaling
# Max values are needed for rescaling parameter estimates to original units
N_cig, N_orig_ecig, N_non_fda_flav_ecig, N_fda_flav_ecig, _, q_cig, q_ecig, q_bundle, q_cig_max, q_ecig_max, q_bundle_max = get_consumption(N_J);

# Get nicotine vector by alternative (STANDARDIZED by max)
# n_max is the raw max value for rescaling estimates
n, n_max = get_nicotine(N_J);

# Get category index by alternative
cat_idx = get_category_index(N_J, N_cig, N_orig_ecig, N_non_fda_flav_ecig, N_fda_flav_ecig);

# Get flavored indicator by alternative: is_flavored[j] ∈ {true, false} (any flavored: non-FDA or FDA)
is_flavored = get_flavored_indicator(cat_idx);

# Get FDA flavored indicator by alternative: is_fda_flavored[j] ∈ {0, 1}
is_fda_flavored = get_fda_flavored_indicator(cat_idx);


#############################
# Demographics
#############################

# Get 4-state TYA classification for each observation (states 1-4)
# State 1: No TYA, stable; State 2: No TYA, approaching
# State 3: TYA present, stable; State 4: TYA present, ending soon
tya_state = get_tya_states()

# Load 4×4 monthly TYA transition matrix Π[s, s'] = P(TYA' = s' | TYA = s)
# Used in VFI to integrate over anticipated TYA state changes
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
# Household Price Trajectories
#############################

# Map observed household prices to continuous values for likelihood interpolation
# p_continuous is N × 2 (cig price, ecig price) — actual per-unit prices, not grid indices
_, p_continuous = map_prices_to_grid(N_P, P, Pcomb);

# Log data setup completion time and sample size
setup_elapsed = time() - t_setup
log_msg("Data loading complete in $(round(setup_elapsed, digits=1))s")
log_msg("Observations: $(length(y))")


#############################
# Fixed Parameters
#############################

# ψ, β, δ are all loaded above via get_fixed_parameters()
# ψ = addiction decay rate (fixed at 0.68; overridden when ESTIMATE_PSI = true)
# β = 1.0 (standard exponential discounting; present bias when ESTIMATE_BETA = true)
# δ = 0.99 (monthly discount factor)


#############################
# Load Estimated Parameters
#############################

# Read θ_hat from the estimates file produced by 02_Estimation.jl (includes beta tag in filename)
estimates_path = joinpath(output_dir, "Dynamic_Model_$(psi_tag)_$(beta_tag)_Estimates.csv")
df_est = CSV.read(estimates_path, DataFrame)

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

# Set global parameter names for objective function logging
est_param_names = collect(String, param_names)


#############################
# Evaluate Objective at θ_hat
#############################

# Reset the global evaluation counter before SE computation
global est_eval_count = 0

# Print and log the objective evaluation at θ_hat
log_msg("\nEvaluating objective at θ_hat...")
t_eval = time()
nll_center = objective(θ_hat)
eval_elapsed = time() - t_eval
log_msg("neg LL at θ_hat = $(round(nll_center, digits=4)) ($(round(eval_elapsed, digits=1))s)")


#############################
# Standard Errors via
# Finite Differences
#############################

# Approximate each second derivative of the neg LL numerically.
# The Hessian H is a matrix of second derivatives:
#   H[k, l] = ∂^2 nll / ∂θ_k ∂θ_l
#
# We approximate this using central differences with parameter-relative step
# sizes h_k = max(|θ_hat[k]| * 1e-4, 1e-5). This scales with parameter
# magnitude and has a floor for near-zero parameters.
#
# Off-diagonal (k ≠ l):
#   H[k, l] ≈ (nll(θ + h_k·e_k + h_l·e_l) - nll(θ + h_k·e_k - h_l·e_l)
#            -  nll(θ - h_k·e_k + h_l·e_l) + nll(θ - h_k·e_k - h_l·e_l)) / (4·h_k·h_l)
#
# where e_k is a unit vector in direction k.
#
# For diagonal entries (k == l), this simplifies to:
#   H[k, k] ≈ (nll(θ + h_k·e_k) - 2·nll(θ) + nll(θ - h_k·e_k)) / h_k^2
#
# Then: Var-Cov = H^{-1}, and SE[k] = sqrt(H^{-1}[k, k])
#
# Each evaluation calls the full objective (VFI + LL), so this is slow.
# Total evaluations: 1 center + N_params diagonal (2 each) + N_params*(N_params-1)/2
# off-diagonal (4 each) = 1 + 2*13 + 4*78 = 339 for 13 parameters.

# Print and log Hessian computation header
log_msg("\n==============================================")
log_msg("Computing Hessian via finite differences")
log_msg("==============================================")

t_hessian = time()

# Parameter-relative step sizes for finite differences
# Each h_k scales with the magnitude of θ_hat[k] (relative step of 1e-4) with a
# floor of 1e-5 for near-zero parameters. This avoids under-stepping large
# parameters and over-stepping small ones.
h_vec = [max(abs(θ_hat[k]) * 1e-4, 1e-5) for k in 1:N_params]

# Allocate Hessian matrix
H = zeros(N_params, N_params)

# Total number of pairs to compute
n_diagonal = N_params
n_off_diagonal = N_params * (N_params - 1) ÷ 2
n_total = n_diagonal + n_off_diagonal
# Print and log number of pairs to compute
log_msg("Pairs to compute: $n_diagonal diagonal + $n_off_diagonal off-diagonal = $n_total total")

# Counter for progress logging
pair_count = 0

# Loop over all parameter pairs (k, l)
for k in 1:N_params
    for l in k:N_params

        pair_count += 1
        t_pair = time()

        # Parameter-specific step sizes for this pair
        h_k = h_vec[k]

        if k == l

            # Diagonal entry: perturb in one direction only
            # H[k, k] ≈ (f(θ + h_k·e_k) - 2·f(θ) + f(θ - h_k·e_k)) / h_k^2
            θ_plus  = copy(θ_hat)
            θ_minus = copy(θ_hat)
            θ_plus[k]  += h_k
            θ_minus[k] -= h_k

            f_plus = objective(θ_plus)
            f_minus = objective(θ_minus)

            H[k, k] = (f_plus - 2.0 * nll_center + f_minus) / h_k^2

            pair_elapsed = time() - t_pair
            log_msg("  Pair $pair_count/$n_total | H[$k,$l] (diagonal) = $(round(H[k,k], digits=4)) | h_k=$(round(h_k, sigdigits=3)) | $(round(pair_elapsed, digits=1))s")

        else

            h_l = h_vec[l]

            # Off-diagonal entry: perturb in both directions
            # H[k, l] ≈ (f(θ++) - f(θ+-) - f(θ-+) + f(θ--)) / (4·h_k·h_l)
            θ_pp = copy(θ_hat)  # Perturb upward in both dimensions
            θ_pm = copy(θ_hat)  # Perturb upward for k, downward for l
            θ_mp = copy(θ_hat)  # Perturb downward for k, upward for l
            θ_mm = copy(θ_hat)  # Perturb downward in both dimensions

            # Perturb in direction k
            θ_pp[k] += h_k
            θ_pm[k] += h_k
            θ_mp[k] -= h_k
            θ_mm[k] -= h_k

            # Perturb in direction l
            θ_pp[l] += h_l
            θ_pm[l] -= h_l
            θ_mp[l] += h_l
            θ_mm[l] -= h_l

            # Evaluate objective at all four perturbed points
            f_pp = objective(θ_pp)
            f_pm = objective(θ_pm)
            f_mp = objective(θ_mp)
            f_mm = objective(θ_mm)

            # Central difference formula for cross-partial
            H[k, l] = (f_pp - f_pm - f_mp + f_mm) / (4.0 * h_k * h_l)

            # Hessian is symmetric
            H[l, k] = H[k, l]

            pair_elapsed = time() - t_pair
            log_msg("  Pair $pair_count/$n_total | H[$k,$l] (off-diag) = $(round(H[k,l], digits=4)) | h_k=$(round(h_k, sigdigits=3)), h_l=$(round(h_l, sigdigits=3)) | $(round(pair_elapsed, digits=1))s")

        end
    end
end

hessian_elapsed = time() - t_hessian
# Print and log Hessian computation time
log_msg("\nHessian computed in $(round(hessian_elapsed, digits=1))s")


#############################
# Variance-Covariance Matrix
#############################

# Invert Hessian to get variance-covariance matrix
V = inv(H)

# Standard errors = sqrt of diagonal
se = sqrt.(abs.(diag(V)))


#############################
# Diagnostics
#############################

# Print and log Hessian eigenvalue diagnostics
eigvals_H = eigvals(H)
log_msg("Hessian eigenvalues: min = $(round(minimum(eigvals_H), digits=2)), max = $(round(maximum(eigvals_H), digits=2))")
if all(eigvals_H .> 0)
    log_msg("Hessian is positive definite (valid MLE)")
else
    log_msg("WARNING: Hessian is NOT positive definite — SEs may be unreliable")
end


#############################
# Results Table
#############################

# Print and log standard error results
log_msg("\n\n==============================================")
log_msg("Standard Error Results")
log_msg("==============================================")

# Print and log results table header
log_msg("")
log_msg(@sprintf("%-8s  %12s  %10s  %10s", "Param", "Estimate", "Std Err", "t-stat"))
log_msg(repeat("-", 55))

# Print and log results table
for k in 1:N_params
    t_stat = θ_hat[k] / se[k]
    log_msg(@sprintf("%-8s  %12.6f  %10.6f  %10.4f", param_names[k], θ_hat[k], se[k], t_stat))
end


#############################
# Save Results
#############################

# Save standard errors to a CSV file (includes beta tag in filename)
se_path = joinpath(output_dir, "Dynamic_Model_$(psi_tag)_$(beta_tag)_Standard_Errors.csv")
open(se_path, "w") do io
    println(io, join(param_names, ","))
    println(io, join([@sprintf("%.10f", θ_hat[k]) for k in 1:N_params], ","))
    println(io, join([@sprintf("%.10f", se[k]) for k in 1:N_params], ","))
end
# Print and log SE save location
log_msg("\nSEs saved to: $se_path")

# Print and log SE computation finished message
log_msg("\nSE computation finished at $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
log_msg("Log saved to: $log_path")

# Close the log file handle
close(log_io)
