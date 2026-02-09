################################################################################
# William Brasic
# The University of Arizona
# wbrasic97@gmail.com
# January 2026
#
# This script estimates the structural parameters of the static logit model
# by maximizing the sample log-likelihood via L-BFGS with analytical gradients.
#
# The static model drops the dynamic addiction terms (μ, γ) from the flow
# utility and does not require value function iteration. Prices enter as
# continuous observed values rather than discretized grid states.
#
# Parameters (10): α_T, α_E, α_TE, λ_1, λ_2, ρ, ω, ξ_T, ξ_E, ξ_TE
#
# ρ captures state dependence (lagged category choice), motivating the
# dynamic model: significant ρ → past choices predict current behavior
# (addiction/habit persistence) which a static model cannot properly
# account for.
#
# IMPORTANT: Estimates are in ORIGINAL UNITS (utils per pack, utils per mL, etc.)
# The dynamic model (02_Estimation.jl) uses STANDARDIZED data, so it converts
# these estimates to standardized units when using them as starting values:
#   α_T_std  = α_T_orig  × c_cig_max
#   α_E_std  = α_E_orig  × c_ecig_max
#   α_TE_std = α_TE_orig × c_bundle_max  (actual max of c_cig×c_ecig, not c_cig_max×c_ecig_max)
#   ω_std    = ω_orig    × E_max
#
# Progress is logged to Static_Logit_Estimation_Log_<timestamp>.txt.
################################################################################


#############################
# Preliminaries
#############################

# Whether we are running on the HPC or not
hpc = false

# Load all functions and packages, set ouput path, and set working directory
using Dates
timestamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
est_log_name = "Static_Logit_Estimation_Log_$(timestamp).txt"
if hpc
    # Include functions file
    include("../Dynamic_Model/01_Functions.jl")

    # Output path for results
    output_dir = "./Static_Logit_Results"
    log_path = joinpath(output_dir, est_log_name)

    # Open log file and set global handle
    est_log_io = open(log_path, "w")
    est_log("Static estimation started at $(timestamp)")

    # Set working directory to where the data CSVs live
    cd("../Data")
else
    # Include functions file
    include("../Dynamic_Model/02_Second_Stage_Estimation/01_Functions.jl")

    # Output path for results
    output_dir = "C:/Users/wbras/OneDrive/Documents/Desktop/UA/4th_Year_Paper/4th_Year_Paper_Data/HMS/2021-Onward/Static_Logit_Results"
    log_path = joinpath(output_dir, est_log_name)

    # Open log file and set global handle
    est_log_io = open(log_path, "w")
    est_log("Static estimation started at $(timestamp)")

    # Set working directory to where the data CSVs live
    cd("C:/Users/wbras/OneDrive/Documents/Desktop/UA/4th_Year_Paper/4th_Year_Paper_Data/HMS/2021-Onward/Dynamic_Model/Data")
end


#############################
# Data Loading
#############################

# Get current time
t_setup = time()

# Get number of alternatives (N_J) and choice matrix (J)
_, N_J, J = get_product_choices()

# Get choice vector (y[i] = chosen alternative index for observation i)
y = get_hh_choices(J)

# Get number of categories excluding outside option (N_K)
N_K, _ = get_category_choices()

# Get consumption vectors by alternative (STANDARDIZED by max in 01_Functions.jl)
# We need RAW consumption for the static logit so estimates are in original units
# (utils per pack, utils per mL, etc.) that can be used as starting values in
# 02_Estimation.jl (which converts them to standardized units).
N_cig, N_orig_ecig, N_flav_ecig, _, c_cig_std, c_ecig_std, c_bundle_std, c_cig_max, c_ecig_max, c_bundle_max = get_consumption(N_J)

# Convert back to raw consumption (packs, mL, packs×mL)
c_cig    = c_cig_std    .* c_cig_max
c_ecig   = c_ecig_std   .* c_ecig_max
c_bundle = c_bundle_std .* c_bundle_max

# Get category index by alternative: cat_idx[j] ∈ {0, 1, 2, 3, 4, 5}
cat_idx = get_category_index(N_J, N_cig, N_orig_ecig, N_flav_ecig)

# Get flavored indicator by alternative: is_flavored[j] ∈ {true, false}
is_flavored = get_flavored_indicator(cat_idx)

# Get TYA binary indicator for each observation
_, tya = get_teen_young_adult()

# Get TYA state index for each observation (1 = no TYA, 2 = TYA present)
tya_state = get_tya_state(tya)

# Get continuous prices for each observation
# (Loads price grid objects to call map_prices_to_grid, but only uses continuous prices)
N_P, P = get_pricing_spaces()
_, Pcomb = get_pricing_spaces_combination(N_K, N_P, P)
_, p_continuous = map_prices_to_grid(N_P, P, Pcomb)

# Get lagged category choice indicators (NA for first observation per household)
df_lag = CSV.read("./Lagged_Category_Choice.csv", DataFrame)
lag_cig_raw      = df_lag.lagged_cig
lag_ecig_raw     = df_lag.lagged_ecig
lag_cig_ecig_raw = df_lag.lagged_cig_ecig

# Keep observations with non-missing lagged choice
valid = .!ismissing.(lag_cig_raw)
valid_idx = findall(valid)
N_full = length(y)

# Subset all observation-level data to valid observations
y            = y[valid_idx]
tya_state    = tya_state[valid_idx]
p_continuous = p_continuous[valid_idx, :]

# Get indicators for lag purchases
lag_cig      = Int64.(collect(skipmissing(lag_cig_raw[valid_idx])))
lag_ecig     = Int64.(collect(skipmissing(lag_ecig_raw[valid_idx])))
lag_cig_ecig = Int64.(collect(skipmissing(lag_cig_ecig_raw[valid_idx])))

# Number of observations after sample restriction
N_obs = length(y)

# Time elapsed in data prep
setup_elapsed = time() - t_setup
est_log("Data loading complete in $(round(setup_elapsed, digits=1))s")
est_log("Full sample: $N_full")
est_log("Valid sample (non-missing lag): $N_obs")
est_log("Alternatives: $N_J")

# Log consumption ranges (raw values used in estimation)
est_log("\nConsumption ranges (RAW, used in estimation):")
est_log("  c_cig:    [$(minimum(c_cig)), $(maximum(c_cig))] packs")
est_log("  c_ecig:   [$(minimum(c_ecig)), $(maximum(c_ecig))] mL")
est_log("  c_bundle: [$(minimum(c_bundle)), $(maximum(c_bundle))] packs×mL")

# Log max values (needed if converting estimates to standardized units)
est_log("\nMax values (for converting to standardized units if needed):")
est_log("  c_cig_max    = $c_cig_max packs")
est_log("  c_ecig_max   = $c_ecig_max mL")
est_log("  c_bundle_max = $c_bundle_max packs×mL (actual max, not c_cig_max × c_ecig_max)")


#############################
# Pre-compute Alternative
# Characteristics
#############################

# Fixed effect category for each alternative:
#   0 = outside option (no FE)
#   1 = cigarettes (ξ_T)
#   2 = e-cigarettes (ξ_E)
#   3 = bundles (ξ_TE)
fe_idx = zeros(Int, N_J)
for j in 1:N_J
    if cat_idx[j] == 1
        fe_idx[j] = 1
    elseif cat_idx[j] == 2 || cat_idx[j] == 3
        fe_idx[j] = 2
    elseif cat_idx[j] == 4 || cat_idx[j] == 5
        fe_idx[j] = 3
    end
end

# Fixed effect indicators
fe_T  = fe_idx .== 1
fe_E  = fe_idx .== 2
fe_TE = fe_idx .== 3

# TYA indicator for each observation
tya = [s == 2 ? 1 : 0 for s in tya_state]

# Expenditure matrix: E_obs[i, j] = p_cig[i] * c_cig[j] + p_ecig[i] * c_ecig[j]
E_obs = p_continuous[:, 1] * c_cig' + p_continuous[:, 2] * c_ecig'

# Log price ranges
est_log("\nPrice ranges:")
est_log("  p_cig:  [$(round(minimum(p_continuous[:, 1]), digits=2)), $(round(maximum(p_continuous[:, 1]), digits=2))] per pack")
est_log("  p_ecig: [$(round(minimum(p_continuous[:, 2]), digits=2)), $(round(maximum(p_continuous[:, 2]), digits=2))] per mL")

# Log expenditure range
est_log("\nExpenditure range:")
est_log("  E_obs:  [$(round(minimum(E_obs), digits=2)), $(round(maximum(E_obs), digits=2))]")

# Lagged category match indicator: lag_match[i, j] = 1 if alternative j belongs
# to the same category the household chose last period.
# Category mapping: fe_idx 1 = cig, 2 = ecig, 3 = bundle (cig_ecig)
lag_match = zeros(Int64, N_obs, N_J)
for i in 1:N_obs
    for j in 1:N_J
        if (fe_idx[j] == 1 && lag_cig[i] == 1) ||
           (fe_idx[j] == 2 && lag_ecig[i] == 1) ||
           (fe_idx[j] == 3 && lag_cig_ecig[i] == 1)
            lag_match[i, j] = 1
        end
    end
end


#############################
# Negative Log-Likelihood
#############################

"""
Negative log-likelihood for the static logit model.
Used with L-BFGS via automatic differentiation (autodiff = :forward).

Parameters (10):
  α_T, α_E, α_TE = consumption utility
  λ_1, λ_2       = flavor effects
  ρ              = state dependence (lagged category match)
  ω              = expenditure coefficient
  ξ_T, ξ_E, ξ_TE = category fixed effects

Data arrays are passed as arguments so they are typed locals inside the
function, avoiding global-variable type instability with ForwardDiff.
"""
function neg_log_likelihood(θ_vec, N_obs, N_J, tya, y,
                            c_cig, c_ecig, c_bundle,
                            is_flavored, lag_match, E_obs,
                            fe_T, fe_E, fe_TE)

    # Unpack parameters (10 total)
    α_T  = θ_vec[1]   # Cigarette utility (per pack)
    α_E  = θ_vec[2]   # E-cig utility (per mL)
    α_TE = θ_vec[3]   # Bundle interaction
    λ_1  = θ_vec[4]   # Flavor effect
    λ_2  = θ_vec[5]   # Flavor × TYA interaction
    ρ    = θ_vec[6]   # State dependence
    ω    = θ_vec[7]   # Expenditure coefficient
    ξ_T  = θ_vec[8]   # Cigarette fixed effect
    ξ_E  = θ_vec[9]   # E-cig fixed effect
    ξ_TE = θ_vec[10]  # Bundle fixed effect

    # Initialize
    neg_LL = zero(eltype(θ_vec))
    v = Vector{eltype(θ_vec)}(undef, N_J)

    # Loop over observations
    for i in 1:N_obs
        tya_i = tya[i]
        y_i = y[i]

        # Compute utilities for all alternatives
        for j in 1:N_J
            v[j] = (α_T * c_cig[j] + α_E * c_ecig[j] + α_TE * c_bundle[j]
                   + is_flavored[j] * (λ_1 + λ_2 * tya_i)
                   + ρ * lag_match[i, j]
                   + ω * E_obs[i, j]
                   + ξ_T * fe_T[j] + ξ_E * fe_E[j] + ξ_TE * fe_TE[j])
        end

        # Log choice probability: log P(y_i) = v[y_i] - log(Σ_j exp(v[j]))
        m = maximum(v)
        s = zero(eltype(v))
        for j in 1:N_J
            s += exp(v[j] - m)
        end
        log_P = v[y_i] - m - log(s)

        neg_LL -= log_P
    end

    return neg_LL
end

# Wrapper function for optimizer (captures all data in closure)
nll = θ -> neg_log_likelihood(θ, N_obs, N_J, tya, y,
                              c_cig, c_ecig, c_bundle,
                              is_flavored, lag_match, E_obs,
                              fe_T, fe_E, fe_TE)


#############################
# Estimation: L-BFGS
# (autodiff gradient)
#############################

# Parameter names (10 parameters)
param_names = ["α_T", "α_E", "α_TE", "λ_1", "λ_2", "ρ", "ω", "ξ_T", "ξ_E", "ξ_TE"]
N_params = length(param_names)

# Starting values (all zeros)
θ_start = zeros(N_params)

# Output header
est_log("\n==============================================")
est_log("Starting L-BFGS optimization (autodiff)")
est_log("==============================================")

# JIT warmup: first ForwardDiff evaluation triggers compilation for Dual types
est_log("JIT warmup (compiling ForwardDiff)...")
t_warmup = time()

# Compile function
nll(θ_start)
est_log("JIT warmup complete in $(round(time() - t_warmup, digits=1))s")

# Function to log progress every iteration
function autodiff_callback(state)

    # Print results every 10 iterations
    if state.iteration % 10 == 0
        est_log(@sprintf("  Iter %4d | neg LL = %12.4f | |∇| = %.2e",
            state.iteration, state.value, state.g_norm))
    end

    # Returning false tells Optim to keep optimizing
    return false
end

# Run L-BFGS with automatic differentiation for gradients
t_est_ad = time()
result_ad = optimize(
    nll,
    θ_start,
    LBFGS(),
    Optim.Options(
        iterations = 1000,
        f_reltol = 1e-6,
        g_tol = 1e-4,
        callback = autodiff_callback
    );
    # Use forward-mode automatic differentiation
    autodiff = :forward
)
est_ad_elapsed = time() - t_est_ad

# Log final iteration
est_log(@sprintf("  Iter %4d | neg LL = %12.4f | |∇| = %.2e (final)",
      Optim.iterations(result_ad), Optim.minimum(result_ad), Optim.g_residual(result_ad)))


#############################
# Autodiff Results
#############################

# Estimation from autodiff
θ_hat_ad = Optim.minimizer(result_ad)

# Print results
est_log("\n\n==============================================")
est_log("Autodiff L-BFGS Results")
est_log("==============================================")
est_log("Converged: $(Optim.converged(result_ad))")
est_log("Negative log-likelihood: $(round(Optim.minimum(result_ad), digits=4))")
est_log("Log-likelihood: $(round(-Optim.minimum(result_ad), digits=4))")
est_log("Iterations: $(Optim.iterations(result_ad))")
est_log("Total estimation time: $(round(est_ad_elapsed, digits=1))s")


#############################
# Standard Errors via
# Inverse Hessian (Autodiff)
#############################

# SE output file header
est_log("\nComputing standard errors via ForwardDiff Hessian...")
t_se_ad = time()

# Compute exact Hessian via automatic differentiation
H_ad = ForwardDiff.hessian(nll, θ_hat_ad)

# Symmetrize (should already be symmetric, but ensure numerical precision)
H_ad = 0.5 * (H_ad + H_ad')

# Invert to get variance-covariance matrix
V_ad = inv(H_ad)

# Standard errors = sqrt of diagonal
se_ad = sqrt.(abs.(diag(V_ad)))

# Show time
se_ad_elapsed = time() - t_se_ad
est_log("Standard errors computed in $(round(se_ad_elapsed, digits=1))s")

# Check that H is positive definite (eigenvalues > 0)
eigvals_H_ad = eigvals(H_ad)
est_log("Hessian eigenvalues: min = $(round(minimum(eigvals_H_ad), digits=2)), max = $(round(maximum(eigvals_H_ad), digits=2))")
if all(eigvals_H_ad .> 0)
    est_log("Hessian is positive definite (valid MLE)")
else
    est_log("Hessian is NOT positive definite")
end


#############################
# Autodiff Results Table
#############################

# Table header
est_log("")
est_log(@sprintf("%-8s  %12s  %10s  %10s", "Parameter", "Estimate", "Std Err", "t-stat"))
est_log(repeat("-", 55))

# Print results to table
for k in 1:N_params
    t_stat = θ_hat_ad[k] / se_ad[k]
    est_log(@sprintf("%-8s  %12.6f  %10.6f  %10.4f", param_names[k], θ_hat_ad[k], se_ad[k], t_stat))
end


#############################
# Standard Errors via
# Finite Differences
#############################

est_log("\nComputing standard errors via finite differences...")
t_se_fd = time()

# Step size for finite differences
h = 1e-3

# Value of neg LL at the optimum (used for diagonal entries)
nll_center = nll(θ_hat_ad)

# Allocate Hessian matrix
H_fd = zeros(N_params, N_params)

# Loop over all parameter pairs (k, l)
for k in 1:N_params
    for l in k:N_params

        # Create perturbed parameter vectors
        θ_pp = copy(θ_hat_ad)
        θ_pm = copy(θ_hat_ad)
        θ_mp = copy(θ_hat_ad)
        θ_mm = copy(θ_hat_ad)

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

        if k == l
            θ_plus  = copy(θ_hat_ad)
            θ_minus = copy(θ_hat_ad)
            θ_plus[k]  += h
            θ_minus[k] -= h
            H_fd[k, k] = (nll(θ_plus) - 2.0 * nll_center + nll(θ_minus)) / h^2
        else
            H_fd[k, l] = (nll(θ_pp) - nll(θ_pm) - nll(θ_mp) + nll(θ_mm)) / (4.0 * h^2)
            H_fd[l, k] = H_fd[k, l]
        end
    end
end

# Invert to get variance-covariance matrix
V_fd = inv(H_fd)

# Standard errors = sqrt of diagonal
se_fd = sqrt.(abs.(diag(V_fd)))

# Show time
se_fd_elapsed = time() - t_se_fd
est_log("Standard errors computed in $(round(se_fd_elapsed, digits=1))s")

# Check that H is positive definite (eigenvalues > 0)
eigvals_H_fd = eigvals(H_fd)
est_log("Hessian eigenvalues: min = $(round(minimum(eigvals_H_fd), digits=2)), max = $(round(maximum(eigvals_H_fd), digits=2))")
if all(eigvals_H_fd .> 0)
    est_log("Hessian is positive definite (valid MLE)")
else
    est_log("Hessian is NOT positive definite")
end


#############################
# Finite Differences
# Results Table
#############################

# Table header
est_log("")
est_log(@sprintf("%-8s  %12s  %10s  %10s", "Parameter", "Estimate", "Std Err", "t-stat"))
est_log(repeat("-", 55))

# Print results to table
for k in 1:N_params
    t_stat = θ_hat_ad[k] / se_fd[k]
    est_log(@sprintf("%-8s  %12.6f  %10.6f  %10.4f", param_names[k], θ_hat_ad[k], se_fd[k], t_stat))
end


#############################
# SE Comparison:
# Autodiff vs Finite Diff
#############################

# Output header
est_log("\n\n==============================================")
est_log("SE Comparison: Autodiff vs Finite Differences")
est_log("==============================================")
est_log(@sprintf("%-8s  %12s  %12s  %12s", "Param", "SE (auto)", "SE (fd)", "Difference"))
est_log(repeat("-", 55))

# Print results
for k in 1:N_params
    diff = se_ad[k] - se_fd[k]
    est_log(@sprintf("%-8s  %12.6f  %12.6f  %12.2e", param_names[k], se_ad[k], se_fd[k], diff))
end


#############################
# Nelder-Mead (Random Amoeba)
#############################

# Print output file header
est_log("\n\n==============================================")
est_log("Starting Random Amoeba optimization")
est_log("==============================================")

# Starting parameter values as a NamedTuple (all zeros)
starting_param_nm = (
    α_T    = 0.0,
    α_E    = 0.0,
    α_TE   = 0.0,
    λ_1    = 0.0,
    λ_2    = 0.0,
    ρ      = 0.0,
    ω      = 0.0,
    ξ_T    = 0.0,
    ξ_E    = 0.0,
    ξ_TE   = 0.0
)

# Initial simplex deviations for each parameter dimension
add_nm = [
    0.01,   # α_T
    0.01,   # α_E
    0.01,   # α_TE
    0.5,    # λ_1
    0.5,    # λ_2
    1.0,    # ρ
    0.005,  # ω (expenditure coefficient)
    1.0,    # ξ_T
    1.0,    # ξ_E
    1.0     # ξ_TE
]

# Optimizer settings
L_nm          = 3    # Outer tries
M_nm          = 2    # Short runs per outer try
inner_iter_nm = 500  # Iterations per short run

# Print settings
est_log("Optimizer settings: L=$L_nm, M=$M_nm, inner_iter=$inner_iter_nm")

# Run multi-start Nelder-Mead
t_nm = time()
opt_param_nm, opt_value_nm = random_amoeba(
    nll, starting_param_nm, add_nm, L_nm, M_nm, inner_iter_nm;
    log_io = est_log_io
)
nm_elapsed = time() - t_nm

# Extract parameter vector from NamedTuple
θ_hat_nm = collect(Float64, values(opt_param_nm))

# Print results header
est_log("\n\n==============================================")
est_log("Random Amoeba Results")
est_log("==============================================")
est_log("Negative log-likelihood: $(round(opt_value_nm, digits=4))")
est_log("Total estimation time: $(round(nm_elapsed, digits=1))s")


#############################
# Standard Errors via
# Inverse Hessian (Autodiff)
#############################

# SE output file header
est_log("\nComputing standard errors via ForwardDiff Hessian...")
t_se_nm_ad = time()

# Compute exact Hessian via automatic differentiation
H_nm_ad = ForwardDiff.hessian(nll, θ_hat_nm)

# Symmetrize (should already be symmetric, but ensure numerical precision)
H_nm_ad = 0.5 * (H_nm_ad + H_nm_ad')

# Invert to get variance-covariance matrix
V_nm_ad = inv(H_nm_ad)

# Standard errors = sqrt of diagonal
se_nm_ad = sqrt.(abs.(diag(V_nm_ad)))

# Show time
se_nm_ad_elapsed = time() - t_se_nm_ad
est_log("Standard errors computed in $(round(se_nm_ad_elapsed, digits=1))s")

# Check that H is positive definite (eigenvalues > 0)
eigvals_H_nm_ad = eigvals(H_nm_ad)
est_log("Hessian eigenvalues: min = $(round(minimum(eigvals_H_nm_ad), digits=2)), max = $(round(maximum(eigvals_H_nm_ad), digits=2))")
if all(eigvals_H_nm_ad .> 0)
    est_log("Hessian is positive definite (valid MLE)")
else
    est_log("Hessian is NOT positive definite")
end

# Table header
est_log("")
est_log(@sprintf("%-8s  %12s  %10s  %10s", "Parameter", "Estimate", "Std Err", "t-stat"))
est_log(repeat("-", 50))

# Print results to table
for k in 1:N_params
    t_stat = θ_hat_nm[k] / se_nm_ad[k]
    est_log(@sprintf("%-8s  %12.6f  %10.6f  %10.4f", param_names[k], θ_hat_nm[k], se_nm_ad[k], t_stat))
end


#############################
# Standard Errors via
# Finite Differences
#############################

# Ouput file header
est_log("\nComputing standard errors via finite differences...")
t_se_nm = time()

# Step size for finite differences
h = 1e-3

# Value of neg LL at the optimum (used for diagonal entries)
nll_center_nm = nll(θ_hat_nm)

# Allocate Hessian matrix
H_nm = zeros(N_params, N_params)

# Loop over all parameter pairs (k, l)
for k in 1:N_params
    for l in k:N_params

        # Create perturbed parameter vectors
        θ_pp = copy(θ_hat_nm)
        θ_pm = copy(θ_hat_nm)
        θ_mp = copy(θ_hat_nm)
        θ_mm = copy(θ_hat_nm)

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

        if k == l
            θ_plus  = copy(θ_hat_nm)
            θ_minus = copy(θ_hat_nm)
            θ_plus[k]  += h
            θ_minus[k] -= h
            H_nm[k, k] = (nll(θ_plus) - 2.0 * nll_center_nm + nll(θ_minus)) / h^2
        else
            H_nm[k, l] = (nll(θ_pp) - nll(θ_pm) - nll(θ_mp) + nll(θ_mm)) / (4.0 * h^2)
            H_nm[l, k] = H_nm[k, l]
        end
    end
end

# Invert to get variance-covariance matrix
V_nm = inv(H_nm)

# Standard errors = sqrt of diagonal
se_nm = sqrt.(abs.(diag(V_nm)))

# Show time
se_nm_elapsed = time() - t_se_nm
est_log("Standard errors computed in $(round(se_nm_elapsed, digits=1))s")

# Check that H is positive definite (eigenvalues > 0)
eigvals_H_nm = eigvals(H_nm)
est_log("Hessian eigenvalues: min = $(round(minimum(eigvals_H_nm), digits=2)), max = $(round(maximum(eigvals_H_nm), digits=2))")
if all(eigvals_H_nm .> 0)
    est_log("Hessian is positive definite (valid MLE)")
else
    est_log("Hessian is NOT positive definite")
end

# Table header
est_log("")
est_log(@sprintf("%-8s  %12s  %10s  %10s", "Parameter", "Estimate", "Std Err", "t-stat"))
est_log(repeat("-", 50))

# Print results to table
for k in 1:N_params
    t_stat = θ_hat_nm[k] / se_nm[k]
    est_log(@sprintf("%-8s  %12.6f  %10.6f  %10.4f", param_names[k], θ_hat_nm[k], se_nm[k], t_stat))
end


#############################
# Comparison Table
#############################

# Print comparison header
est_log("\n\n==============================================")
est_log("L-BFGS vs Random Amoeba Comparison")
est_log("==============================================")
est_log(@sprintf("%-8s  %12s  %10s  %12s  %10s", "Param", "L-BFGS", "SE", "Amoeba", "SE"))
est_log(repeat("-", 65))

# Print comparison table
for k in 1:N_params
    est_log(@sprintf("%-8s  %12.6f  %10.6f  %12.6f  %10.6f",
        param_names[k], θ_hat_ad[k], se_ad[k], θ_hat_nm[k], se_nm[k]))
end

# Print neg LL and time comparison
est_log(@sprintf("\n%-8s  %12.4f  %10s  %12.4f", "neg LL", Optim.minimum(result_ad), "", opt_value_nm))
est_log(@sprintf("%-8s  %12.1f  %10s  %12.1f", "Time(s)", est_ad_elapsed, "", nm_elapsed))


#############################
# Dynamic Model Starting
# Values Conversion
#############################

est_log("\n\n==============================================")
est_log("Dynamic Model Starting Values")
est_log("==============================================")
est_log("These estimates are in ORIGINAL units. To use as starting values")
est_log("in the dynamic model (which uses standardized data), multiply by max:")
est_log("")
est_log("  Standardization factors:")
est_log("    c_cig_max    = $c_cig_max packs")
est_log("    c_ecig_max   = $c_ecig_max mL")
est_log("    c_bundle_max = $c_bundle_max packs×mL (actual max, not c_cig_max × c_ecig_max)")
est_log("")
est_log("  Conversion formulas (using L-BFGS estimates):")
est_log(@sprintf("    α_T_std  = %.4f × %.1f = %.4f", θ_hat_ad[1], c_cig_max, θ_hat_ad[1] * c_cig_max))
est_log(@sprintf("    α_E_std  = %.4f × %.1f = %.4f", θ_hat_ad[2], c_ecig_max, θ_hat_ad[2] * c_ecig_max))
est_log(@sprintf("    α_TE_std = %.4f × %.1f = %.4f", θ_hat_ad[3], c_bundle_max, θ_hat_ad[3] * c_bundle_max))
est_log("    λ_1, λ_2, ρ, ξ_T, ξ_E, ξ_TE: no conversion needed")
est_log("")
est_log("  Note: ω requires E_max from the dynamic model's get_expenditures().")
est_log("  The dynamic model computes: ω_std = ω_orig × E_max")


# Print message indicating completion
est_log("\nStatic estimation finished at $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
est_log("Log saved to: $log_path")

# Close log file
close(est_log_io)
