################################################################################
# William Brasic
# The University of Arizona
# wbrasic97@gmail.com
# January 2026
#
# This script computes standard errors for the dynamic model parameter
# estimates via finite differences on the full objective function.
#
# The Hessian H of the negative log-likelihood is approximated numerically:
#   - Diagonal:     H[k,k] ≈ (f(θ+h·e_k) - 2·f(θ) + f(θ-h·e_k)) / h^2
#   - Off-diagonal: H[k,l] ≈ (f(θ+h·e_k+h·e_l) - f(θ+h·e_k-h·e_l)
#                            -  f(θ-h·e_k+h·e_l) + f(θ-h·e_k-h·e_l)) / (4h^2)
#
# Each evaluation of f(θ) requires solving VFI from scratch, so this is slow
# but gives correct SEs that account for how V_choice changes with θ.
#
# Variance-covariance matrix: H^{-1}
# Standard errors: SE[k] = sqrt(H^{-1}[k,k])
#
# Reads estimated parameters from Dynamic_Model_Estimates.txt (produced by
# 02_Estimation.jl). Progress is logged to SE_Log.txt.
################################################################################


#############################
# Preliminaries
#############################

# Load all functions and packages
include("01_Functions.jl")
using LinearAlgebra

# Set working directory to where the data CSVs live
cd("C:/Users/wbras/OneDrive/Documents/Desktop/UA/4th_Year_Paper/4th_Year_Paper_Data/HMS/2021-Onward/Dynamic_Model/Data")


#############################
# Output Paths
#############################

# Output path for results
output_dir = "C:/Users/wbras/OneDrive/Documents/Desktop/UA/4th_Year_Paper/4th_Year_Paper_Data/HMS/2021-Onward/Dynamic_Model/Dynamic_Model_Results"
log_path = joinpath(output_dir, "SE_Log.txt")

# Open log file and set global handle
est_log_io = open(log_path, "w")
est_log("SE computation started at $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")


#############################
# State Spaces and Choices
#############################

t_setup = time()

# Initial addiction grid (ψ is now estimated; objective() recomputes A from θ_vec)
N_A, A = get_addiction_space(0.94)

# Get number of alternatives (N_J) and choice matrix (J)
_, N_J, J = get_product_choices()

# Get choice vector (y[i] = chosen alternative index for observation i)
y = get_hh_choices(J)

# Get household identifiers (pre-loaded to avoid repeated CSV reads in objective)
# Note: variable must be named hh_codes to match what objective() expects
hh_codes = get_hh_codes()

# Get number of categories excluding outside option (N_K)
N_K, _ = get_category_choices()


#############################
# Alternative-Level Vectors
#############################

# Get consumption vectors by alternative (STANDARDIZED by max)
# c_bundle is standardized by its own max (not c_cig_max × c_ecig_max)
N_cig, N_orig_ecig, N_flav_ecig, _, c_cig, c_ecig, c_bundle, c_cig_max, c_ecig_max, c_bundle_max = get_consumption(N_J)

# Get nicotine vector by alternative (STANDARDIZED by max)
n, n_max = get_nicotine(N_J)

# Get category index by alternative: cat_idx[j] ∈ {0, 1, 2, 3, 4, 5}
cat_idx = get_category_index(N_J, N_cig, N_orig_ecig, N_flav_ecig)

# Get flavored indicator by alternative: is_flavored[j] ∈ {true, false}
is_flavored = get_flavored_indicator(cat_idx)


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
N_P, P = get_pricing_spaces()

# Get all price combinations: N_Pcomb = N_P^2, Pcomb is N_Pcomb × 2
N_Pcomb, Pcomb = get_pricing_spaces_combination(N_K, N_P, P)

# Get expenditure matrix (STANDARDIZED by max)
E, E_max = get_expenditures(N_J, N_Pcomb, c_cig, c_ecig, c_cig_max, c_ecig_max, Pcomb)

# Get price transitions from Halton draws: T[m, r, k]
T = get_transitions(N_K)

# Pre-compute price transition brackets and interpolation weights
p_cig_lo, p_cig_hi, p_cig_w, p_ecig_lo, p_ecig_hi, p_ecig_w = precompute_price_transitions(N_P, P, T)


#############################
# Household Price Trajectories
#############################

# Map observed prices to continuous values for likelihood interpolation
# p_continuous: actual continuous prices N × 2 (cig, ecig)
_, p_continuous = map_prices_to_grid(N_P, P, Pcomb)

setup_elapsed = time() - t_setup
est_log("Data loading complete in $(round(setup_elapsed, digits=1))s")
est_log("Observations: $(length(y))")


#############################
# Fixed Parameters
#############################

# ψ is now estimated (part of θ_vec); objective() recomputes A at each eval

# Present bias parameter (β-δ discounting; β = 1.0 is standard exponential)
β = 1.0

# Monthly discount factor
δ = 0.99


#############################
# Load Estimated Parameters
#############################

# Read θ_hat from the estimates file produced by 02_Estimation.jl
estimates_path = joinpath(output_dir, "Dynamic_Model_Estimates.txt")
df_est = CSV.read(estimates_path, DataFrame)

# Extract parameter names and values
param_names = names(df_est)
N_params = length(param_names)
θ_hat = Float64.(collect(df_est[1, :]))

# Print loaded parameters
est_log("\nLoaded estimates from: $estimates_path")
est_log("Parameters:")
for k in 1:N_params
    est_log("  $(param_names[k]) = $(θ_hat[k])")
end

# Set global parameter names for objective function logging
est_param_names = collect(String, param_names)


#############################
# Evaluate Objective at θ_hat
#############################

# Reset evaluation counter
global est_eval_count = 0

# Evaluate the objective at θ_hat to get the neg LL
est_log("\nEvaluating objective at θ_hat...")
t_eval = time()
nll_center = objective(θ_hat)
eval_elapsed = time() - t_eval
est_log("neg LL at θ_hat = $(round(nll_center, digits=4)) ($(round(eval_elapsed, digits=1))s)")


#############################
# Standard Errors via
# Finite Differences
#############################

# Approximate each second derivative of the neg LL numerically.
# The Hessian H is a matrix of second derivatives:
#   H[k, l] = ∂^2 nll / ∂θ_k ∂θ_l
#
# We approximate this using central differences:
#   H[k, l] ≈ (nll(θ + h·e_k + h·e_l) - nll(θ + h·e_k - h·e_l)
#            -  nll(θ - h·e_k + h·e_l) + nll(θ - h·e_k - h·e_l)) / (4h^2)
#
# where e_k is a unit vector in direction k and h is a small step size.
#
# For diagonal entries (k == l), this simplifies to:
#   H[k, k] ≈ (nll(θ + h·e_k) - 2·nll(θ) + nll(θ - h·e_k)) / h^2
#
# Then: Var-Cov = H^{-1}, and SE[k] = sqrt(H^{-1}[k, k])
#
# Each evaluation calls the full objective (VFI + LL), so this is slow.
# Total evaluations: 1 center + N_params diagonal (2 each) + N_params*(N_params-1)/2
# off-diagonal (4 each) = 1 + 2*11 + 4*55 = 243 for 11 parameters.

est_log("\n==============================================")
est_log("Computing Hessian via finite differences")
est_log("==============================================")

t_hessian = time()

# Step size for finite differences
h = 1e-3

# Allocate Hessian matrix
H = zeros(N_params, N_params)

# Total number of pairs to compute
n_diagonal = N_params
n_off_diagonal = N_params * (N_params - 1) ÷ 2
n_total = n_diagonal + n_off_diagonal
est_log("Pairs to compute: $n_diagonal diagonal + $n_off_diagonal off-diagonal = $n_total total")

# Counter for progress logging
pair_count = 0

# Loop over all parameter pairs (k, l)
for k in 1:N_params
    for l in k:N_params

        pair_count += 1
        t_pair = time()

        if k == l

            # Diagonal entry: perturb in one direction only
            # H[k, k] ≈ (f(θ + h·e_k) - 2·f(θ) + f(θ - h·e_k)) / h^2
            θ_plus  = copy(θ_hat)
            θ_minus = copy(θ_hat)
            θ_plus[k]  += h
            θ_minus[k] -= h

            f_plus = objective(θ_plus)
            f_minus = objective(θ_minus)

            H[k, k] = (f_plus - 2.0 * nll_center + f_minus) / h^2

            pair_elapsed = time() - t_pair
            est_log("  Pair $pair_count/$n_total | H[$k,$l] (diagonal) = $(round(H[k,k], digits=4)) | $(round(pair_elapsed, digits=1))s")

        else

            # Off-diagonal entry: perturb in both directions
            # H[k, l] ≈ (f(θ++) - f(θ+-) - f(θ-+) + f(θ--)) / (4h^2)
            θ_pp = copy(θ_hat)  # Perturb upward in both dimensions
            θ_pm = copy(θ_hat)  # Perturb upward for k, downward for l
            θ_mp = copy(θ_hat)  # Perturb downward for k, upward for l
            θ_mm = copy(θ_hat)  # Perturb downward in both dimensions

            # Perturb in direction k
            θ_pp[k] += h
            θ_pm[k] += h
            θ_mp[k] -= h
            θ_mm[k] -= h

            # Perturb in direction l
            θ_pp[l] += h
            θ_pm[l] -= h
            θ_mp[l] += h
            θ_mm[l] -= h

            # Evaluate objective at all four perturbed points
            f_pp = objective(θ_pp)
            f_pm = objective(θ_pm)
            f_mp = objective(θ_mp)
            f_mm = objective(θ_mm)

            # Central difference formula for cross-partial
            H[k, l] = (f_pp - f_pm - f_mp + f_mm) / (4.0 * h^2)

            # Hessian is symmetric
            H[l, k] = H[k, l]

            pair_elapsed = time() - t_pair
            est_log("  Pair $pair_count/$n_total | H[$k,$l] (off-diag) = $(round(H[k,l], digits=4)) | $(round(pair_elapsed, digits=1))s")

        end
    end
end

hessian_elapsed = time() - t_hessian
est_log("\nHessian computed in $(round(hessian_elapsed, digits=1))s")


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

# Check that H is positive definite (eigenvalues > 0)
eigvals_H = eigvals(H)
est_log("Hessian eigenvalues: min = $(round(minimum(eigvals_H), digits=2)), max = $(round(maximum(eigvals_H), digits=2))")
if all(eigvals_H .> 0)
    est_log("Hessian is positive definite (valid MLE)")
else
    est_log("WARNING: Hessian is NOT positive definite — SEs may be unreliable")
end


#############################
# Results Table
#############################

est_log("\n\n==============================================")
est_log("Standard Error Results")
est_log("==============================================")

# Table header
est_log("")
est_log(@sprintf("%-8s  %12s  %10s  %10s", "Param", "Estimate", "Std Err", "t-stat"))
est_log(repeat("-", 55))

# Print results to table
for k in 1:N_params
    t_stat = θ_hat[k] / se[k]
    est_log(@sprintf("%-8s  %12.6f  %10.6f  %10.4f", param_names[k], θ_hat[k], se[k], t_stat))
end


#############################
# Save Results
#############################

# Save SEs to file
se_path = joinpath(output_dir, "Dynamic_Model_Standard_Errors.txt")
open(se_path, "w") do io
    println(io, join(param_names, "\t"))
    println(io, join([@sprintf("%.10f", θ_hat[k]) for k in 1:N_params], "\t"))
    println(io, join([@sprintf("%.10f", se[k]) for k in 1:N_params], "\t"))
end
est_log("\nSEs saved to: $se_path")

est_log("\nSE computation finished at $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
est_log("Log saved to: $log_path")

# Close log file
close(est_log_io)
