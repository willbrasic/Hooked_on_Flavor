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
# Parameters (12): α_C, α_E, α_CE, λ_1, λ_2, λ_3, λ_4, ρ, ω, ξ_C, ξ_E, ξ_CE
#
# ρ captures state dependence (lagged category choice), motivating the
# dynamic model: significant ρ → past choices predict current behavior
# (addiction/habit persistence) which a static model cannot properly
# account for.
#
# IMPORTANT: Estimates are in ORIGINAL UNITS (utils per pack, utils per mL, etc.)
# The dynamic model (02_Estimation.jl) uses STANDARDIZED data, so it converts
# these estimates to standardized units when using them as starting values:
#   α_C_std  = α_C_orig  × q_cig_max
#   α_E_std  = α_E_orig  × q_ecig_max
#   α_CE_std = α_CE_orig × q_bundle_max  (actual max of q_cig×q_ecig, not q_cig_max×q_ecig_max)
#   ω_std    = ω_orig    × E_max
#
# Progress is logged to Static_Logit_Estimation_Log_<timestamp>.txt.
################################################################################


#############################
# Preliminaries
#############################

# Whether we are running on the HPC or not
hpc = !Sys.iswindows()

# Static logit does not estimate β
ESTIMATE_BETA = false

# Load all functions and packages, set output path, and set working directory
using Dates
timestamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
est_log_name = "Static_Logit_Estimation_Log_$(timestamp).txt"
if hpc
    # Include functions file
    include("../Dynamic_Model/01_Functions.jl")

    # Output path for results
    output_dir = "./Static_Logit_Results"
    mkpath(output_dir)
    log_path = joinpath(output_dir, est_log_name)

    # Open log file and set global log_io handle (defined in 01_Functions.jl)
    global log_io = open(log_path, "w")
    log_msg("Static estimation started at $(timestamp)")

    # Set working directory to where the data CSVs live
    cd("../Data")
else
    # Include functions file
    include("../Dynamic_Model/02_Second_Stage_Estimation/01_Functions.jl")

    # Output path for results
    output_dir = "C:/Users/wbras/OneDrive/Documents/Desktop/UA/4th_Year_Paper/4th_Year_Paper_Data/HMS/2021-Onward/Static_Logit_Results"
    mkpath(output_dir)
    log_path = joinpath(output_dir, est_log_name)

    # Open log file and set global log_io handle (defined in 01_Functions.jl)
    global log_io = open(log_path, "w")
    log_msg("Static estimation started at $(timestamp)")

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
N_cig, N_orig_ecig, N_non_fda_flav_ecig, N_fda_flav_ecig, _, q_cig_std, q_ecig_std, q_bundle_std, q_cig_max, q_ecig_max, q_bundle_max = get_consumption(N_J)

# Convert back to raw consumption (packs, mL, packs×mL)
q_cig    = q_cig_std    .* q_cig_max
q_ecig   = q_ecig_std   .* q_ecig_max
q_bundle = q_bundle_std .* q_bundle_max

# Get category index by alternative: cat_idx[j] ∈ {0, 1, 2, 3, 4, 5, 6, 7}
cat_idx = get_category_index(N_J, N_cig, N_orig_ecig, N_non_fda_flav_ecig, N_fda_flav_ecig)

# Get flavored indicator by alternative: is_flavored[j] ∈ {true, false} (any flavored: non-FDA or FDA)
is_flavored = get_flavored_indicator(cat_idx)

# Get FDA flavored indicator by alternative: is_fda_flavored[j] ∈ {true, false}
is_fda_flavored = get_fda_flavored_indicator(cat_idx)

# Get price ratios for quantity discounts (ratio = bin median / category median)
# Bundles inherit the ratio of the closest standalone bin
ratio_cig, ratio_ecig = get_price_ratios(N_J, N_cig, N_orig_ecig, N_non_fda_flav_ecig, N_fda_flav_ecig, q_cig_std, q_ecig_std)

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
valid     = .!ismissing.(lag_cig_raw)
valid_idx = findall(valid)
N_full    = length(y)

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
log_msg("Data loading complete in $(round(setup_elapsed, digits=1))s")
log_msg("Full sample: $N_full")
log_msg("Valid sample (non-missing lag): $N_obs")
log_msg("Alternatives: $N_J")

# Log consumption ranges (raw values used in estimation)
log_msg("\nConsumption ranges (RAW, used in estimation):")
log_msg("  q_cig:    [$(minimum(q_cig)), $(maximum(q_cig))] packs")
log_msg("  q_ecig:   [$(minimum(q_ecig)), $(maximum(q_ecig))] mL")
log_msg("  q_bundle: [$(minimum(q_bundle)), $(maximum(q_bundle))] packs×mL")

# Log max values (needed if converting estimates to standardized units)
log_msg("\nMax values (for converting to standardized units if needed):")
log_msg("  q_cig_max    = $q_cig_max packs")
log_msg("  q_ecig_max   = $q_ecig_max mL")
log_msg("  q_bundle_max = $q_bundle_max packs×mL (actual max, not q_cig_max × q_ecig_max)")


#############################
# Pre-compute Alternative
# Characteristics
#############################

# Fixed effect category for each alternative:
#   0 = outside option (no FE)
#   1 = cigarettes (ξ_C)
#   2 = any e-cigarettes: orig, non-FDA flav, FDA flav (ξ_E)
#   3 = any bundles: orig, non-FDA flav, FDA flav (ξ_CE)
fe_idx = zeros(Int, N_J)
for j in 1:N_J
    if cat_idx[j] == 1
        fe_idx[j] = 1
    elseif cat_idx[j] in (2, 3, 4)
        fe_idx[j] = 2
    elseif cat_idx[j] in (5, 6, 7)
        fe_idx[j] = 3
    end
end

# Fixed effect indicators
fe_C  = fe_idx .== 1
fe_E  = fe_idx .== 2
fe_CE = fe_idx .== 3

# TYA indicator for each observation
tya = [s == 2 ? 1 : 0 for s in tya_state]

# Expenditure matrix: E_obs[i, j] = p_cig[i] * ratio_cig[j] * q_cig[j] + p_ecig[i] * ratio_ecig[j] * q_ecig[j]
# Price ratios capture quantity discounts (lower per-unit prices for larger quantities)
E_obs = p_continuous[:, 1] * (ratio_cig .* q_cig)' + p_continuous[:, 2] * (ratio_ecig .* q_ecig)'

# Log price ranges
log_msg("\nPrice ranges:")
log_msg("  p_cig:  [$(round(minimum(p_continuous[:, 1]), digits=2)), $(round(maximum(p_continuous[:, 1]), digits=2))] per pack")
log_msg("  p_ecig: [$(round(minimum(p_continuous[:, 2]), digits=2)), $(round(maximum(p_continuous[:, 2]), digits=2))] per mL")

# Log expenditure range
log_msg("\nExpenditure range:")
log_msg("  E_obs:  [$(round(minimum(E_obs), digits=2)), $(round(maximum(E_obs), digits=2))]")

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

Parameters (12):
  α_C, α_E, α_CE     = consumption utility
  λ_1, λ_2           = flavor effects (baseline, × TYA) — all flavored products
  λ_3, λ_4           = FDA flavor effects (baseline, × TYA) — additional for FDA-authorized
  ρ                   = state dependence (lagged category match)
  ω                   = expenditure coefficient
  ξ_C, ξ_E, ξ_CE     = category fixed effects

Data arrays are passed as arguments so they are typed locals inside the
function, avoiding global-variable type instability with ForwardDiff.
"""
function neg_log_likelihood(θ_vec, N_obs, N_J, tya, y,
                            q_cig, q_ecig, q_bundle,
                            is_flavored, is_fda_flavored,
                            lag_match, E_obs,
                            fe_C, fe_E, fe_CE)

    # Unpack parameters (12 total)
    α_C  = θ_vec[1]   # Cigarette utility (per pack)
    α_E  = θ_vec[2]   # E-cig utility (per mL)
    α_CE = θ_vec[3]   # Bundle interaction
    λ_1  = θ_vec[4]   # Flavor baseline (all flavored)
    λ_2  = θ_vec[5]   # Flavor × TYA interaction (all flavored)
    λ_3  = θ_vec[6]   # FDA flavor baseline (additional for FDA-authorized)
    λ_4  = θ_vec[7]   # FDA flavor × TYA interaction (additional for FDA-authorized)
    ρ    = θ_vec[8]   # State dependence
    ω    = θ_vec[9]   # Expenditure coefficient
    ξ_C  = θ_vec[10]  # Cigarette fixed effect
    ξ_E  = θ_vec[11]  # E-cig fixed effect
    ξ_CE = θ_vec[12]  # Bundle fixed effect

    # Initialize
    neg_LL = zero(eltype(θ_vec))
    v = Vector{eltype(θ_vec)}(undef, N_J)

    # Loop over observations
    for i in 1:N_obs
        tya_i = tya[i]
        y_i = y[i]

        # Compute utilities for all alternatives
        for j in 1:N_J
            v[j] = (α_C * q_cig[j] + α_E * q_ecig[j] + α_CE * q_bundle[j]
                   + is_flavored[j] * (λ_1 + λ_2 * tya_i)
                   + is_fda_flavored[j] * (λ_3 + λ_4 * tya_i)
                   + ρ * lag_match[i, j]
                   + ω * E_obs[i, j]
                   + ξ_C * fe_C[j] + ξ_E * fe_E[j] + ξ_CE * fe_CE[j])
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
                              q_cig, q_ecig, q_bundle,
                              is_flavored, is_fda_flavored,
                              lag_match, E_obs,
                              fe_C, fe_E, fe_CE)


#############################
# Estimation: L-BFGS
# (autodiff gradient)
#############################

# Parameter names (12 parameters)
param_names = ["α_C", "α_E", "α_CE", "λ_1", "λ_2", "λ_3", "λ_4", "ρ", "ω", "ξ_C", "ξ_E", "ξ_CE"]
N_params = length(param_names)

# Starting values set to all zeros
θ_start = zeros(N_params)

# Output header
log_msg("\n==============================================")
log_msg("Starting L-BFGS optimization (autodiff)")
log_msg("==============================================")

# JIT warmup: first ForwardDiff evaluation triggers compilation for Dual types
log_msg("JIT warmup (compiling ForwardDiff)...")
t_warmup = time()

# Compile function
nll(θ_start)
log_msg("JIT warmup complete in $(round(time() - t_warmup, digits=1))s")

# Function to log progress every iteration
function autodiff_callback(state)

    # Print results every 10 iterations
    if state.iteration % 10 == 0
        log_msg(@sprintf("  Iter %4d | neg LL = %12.4f | |∇| = %.2e",
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
        f_reltol = 0.0,
        g_tol = 1e-4,
        callback = autodiff_callback
    );
    # Use forward-mode automatic differentiation
    autodiff = :forward
)
est_ad_elapsed = time() - t_est_ad

# Log final iteration
log_msg(@sprintf("  Iter %4d | neg LL = %12.4f | |∇| = %.2e (final)",
      Optim.iterations(result_ad), Optim.minimum(result_ad), Optim.g_residual(result_ad)))


#############################
# Autodiff Results
#############################

# Estimation from autodiff
θ_hat_ad = Optim.minimizer(result_ad)

# Print results
log_msg("\n\n==============================================")
log_msg("Autodiff L-BFGS Results")
log_msg("==============================================")
log_msg("Converged: $(Optim.converged(result_ad))")
log_msg("Negative log-likelihood: $(round(Optim.minimum(result_ad), digits=4))")
log_msg("Log-likelihood: $(round(-Optim.minimum(result_ad), digits=4))")
log_msg("Iterations: $(Optim.iterations(result_ad))")
log_msg("Total estimation time: $(round(est_ad_elapsed, digits=1))s")


#############################
# Standard Errors via
# Inverse Hessian (Autodiff)
#############################

# SE output file header
log_msg("\nComputing standard errors via ForwardDiff Hessian...")
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
log_msg("Standard errors computed in $(round(se_ad_elapsed, digits=1))s")

# Check that H is positive definite (eigenvalues > 0)
eigvals_H_ad = eigvals(H_ad)
log_msg("Hessian eigenvalues: min = $(round(minimum(eigvals_H_ad), digits=2)), max = $(round(maximum(eigvals_H_ad), digits=2))")
if all(eigvals_H_ad .> 0)
    log_msg("Hessian is positive definite (valid MLE)")
else
    log_msg("Hessian is NOT positive definite")
end


#############################
# Autodiff Results Table
#############################

# Table header
log_msg("")
log_msg(@sprintf("%-8s  %12s  %10s  %10s", "Parameter", "Estimate", "Std Err", "t-stat"))
log_msg(repeat("-", 55))

# Print results to table
for k in 1:N_params
    t_stat = θ_hat_ad[k] / se_ad[k]
    log_msg(@sprintf("%-8s  %12.6f  %10.6f  %10.4f", param_names[k], θ_hat_ad[k], se_ad[k], t_stat))
end


#############################
# Standard Errors via
# Finite Differences
#############################

log_msg("\nComputing standard errors via finite differences...")
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
log_msg("Standard errors computed in $(round(se_fd_elapsed, digits=1))s")

# Check that H is positive definite (eigenvalues > 0)
eigvals_H_fd = eigvals(H_fd)
log_msg("Hessian eigenvalues: min = $(round(minimum(eigvals_H_fd), digits=2)), max = $(round(maximum(eigvals_H_fd), digits=2))")
if all(eigvals_H_fd .> 0)
    log_msg("Hessian is positive definite (valid MLE)")
else
    log_msg("Hessian is NOT positive definite")
end


#############################
# Finite Differences
# Results Table
#############################

# Table header
log_msg("")
log_msg(@sprintf("%-8s  %12s  %10s  %10s", "Parameter", "Estimate", "Std Err", "t-stat"))
log_msg(repeat("-", 55))

# Print results to table
for k in 1:N_params
    t_stat = θ_hat_ad[k] / se_fd[k]
    log_msg(@sprintf("%-8s  %12.6f  %10.6f  %10.4f", param_names[k], θ_hat_ad[k], se_fd[k], t_stat))
end


#############################
# SE Comparison:
# Autodiff vs Finite Diff
#############################

# Output header
log_msg("\n\n==============================================")
log_msg("SE Comparison: Autodiff vs Finite Differences")
log_msg("==============================================")
log_msg(@sprintf("%-8s  %12s  %12s  %12s", "Param", "SE (auto)", "SE (fd)", "Difference"))
log_msg(repeat("-", 55))

# Print results
for k in 1:N_params
    diff = se_ad[k] - se_fd[k]
    log_msg(@sprintf("%-8s  %12.6f  %12.6f  %12.2e", param_names[k], se_ad[k], se_fd[k], diff))
end


#############################
# Nelder-Mead (Random Amoeba)
#############################

# Print output file header
log_msg("\n\n==============================================")
log_msg("Starting Random Amoeba optimization")
log_msg("==============================================")

# Starting values from auto-diff converged estimates
starting_param_nm = NamedTuple{(:α_C, :α_E, :α_CE, :λ_1, :λ_2, :λ_3, :λ_4, :ρ, :ω, :ξ_C, :ξ_E, :ξ_CE)}(Tuple(θ_hat_ad))

# Initial simplex deviations for Nelder-Mead (~50% of each parameter's magnitude)
add_nm = [abs(v) * 0.50 for v in θ_hat_ad]

# Optimizer settings
L_nm          = 2    # Outer tries
M_nm          = 2    # Short runs per outer try
inner_iter_nm = 100  # Iterations per short run

# Print settings
log_msg("Optimizer settings: L=$L_nm, M=$M_nm, inner_iter=$inner_iter_nm")

# Run multi-start Nelder-Mead
t_nm = time()
opt_param_nm, opt_value_nm = random_amoeba(
    nll, starting_param_nm, add_nm, L_nm, M_nm;
    inner_iter = inner_iter_nm
)
nm_elapsed = time() - t_nm

# Extract parameter vector from NamedTuple
θ_hat_nm = collect(Float64, values(opt_param_nm))

# Print results header
log_msg("\n\n==============================================")
log_msg("Random Amoeba Results")
log_msg("==============================================")
log_msg("Negative log-likelihood: $(round(opt_value_nm, digits=4))")
log_msg("Total estimation time: $(round(nm_elapsed, digits=1))s")


#############################
# Standard Errors via
# Inverse Hessian (Autodiff)
#############################

# SE output file header
log_msg("\nComputing standard errors via ForwardDiff Hessian...")
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
log_msg("Standard errors computed in $(round(se_nm_ad_elapsed, digits=1))s")

# Check that H is positive definite (eigenvalues > 0)
eigvals_H_nm_ad = eigvals(H_nm_ad)
log_msg("Hessian eigenvalues: min = $(round(minimum(eigvals_H_nm_ad), digits=2)), max = $(round(maximum(eigvals_H_nm_ad), digits=2))")
if all(eigvals_H_nm_ad .> 0)
    log_msg("Hessian is positive definite (valid MLE)")
else
    log_msg("Hessian is NOT positive definite")
end

# Table header
log_msg("")
log_msg(@sprintf("%-8s  %12s  %10s  %10s", "Parameter", "Estimate", "Std Err", "t-stat"))
log_msg(repeat("-", 50))

# Print results to table
for k in 1:N_params
    t_stat = θ_hat_nm[k] / se_nm_ad[k]
    log_msg(@sprintf("%-8s  %12.6f  %10.6f  %10.4f", param_names[k], θ_hat_nm[k], se_nm_ad[k], t_stat))
end


#############################
# Standard Errors via
# Finite Differences
#############################

# Output file header
log_msg("\nComputing standard errors via finite differences...")
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
log_msg("Standard errors computed in $(round(se_nm_elapsed, digits=1))s")

# Check that H is positive definite (eigenvalues > 0)
eigvals_H_nm = eigvals(H_nm)
log_msg("Hessian eigenvalues: min = $(round(minimum(eigvals_H_nm), digits=2)), max = $(round(maximum(eigvals_H_nm), digits=2))")
if all(eigvals_H_nm .> 0)
    log_msg("Hessian is positive definite (valid MLE)")
else
    log_msg("Hessian is NOT positive definite")
end

# Table header
log_msg("")
log_msg(@sprintf("%-8s  %12s  %10s  %10s", "Parameter", "Estimate", "Std Err", "t-stat"))
log_msg(repeat("-", 50))

# Print results to table
for k in 1:N_params
    t_stat = θ_hat_nm[k] / se_nm[k]
    log_msg(@sprintf("%-8s  %12.6f  %10.6f  %10.4f", param_names[k], θ_hat_nm[k], se_nm[k], t_stat))
end


#############################
# Comparison Table
#############################

# Print comparison header
log_msg("\n\n==============================================")
log_msg("L-BFGS vs Random Amoeba Comparison")
log_msg("==============================================")
log_msg(@sprintf("%-8s  %12s  %10s  %12s  %10s", "Param", "L-BFGS", "SE", "Amoeba", "SE"))
log_msg(repeat("-", 65))

# Print comparison table
for k in 1:N_params
    log_msg(@sprintf("%-8s  %12.6f  %10.6f  %12.6f  %10.6f",
        param_names[k], θ_hat_ad[k], se_ad[k], θ_hat_nm[k], se_nm[k]))
end

# Print neg LL and time comparison
log_msg(@sprintf("\n%-8s  %12.4f  %10s  %12.4f", "neg LL", Optim.minimum(result_ad), "", opt_value_nm))
log_msg(@sprintf("%-8s  %12.1f  %10s  %12.1f", "Time(s)", est_ad_elapsed, "", nm_elapsed))


#############################
# Dynamic Model Starting
# Values Conversion
#############################

log_msg("\n\n==============================================")
log_msg("Dynamic Model Starting Values")
log_msg("==============================================")
log_msg("These estimates are in ORIGINAL units. To use as starting values")
log_msg("in the dynamic model (which uses standardized data), multiply by max:")
log_msg("")
log_msg("  Standardization factors:")
log_msg("    q_cig_max    = $q_cig_max packs")
log_msg("    q_ecig_max   = $q_ecig_max mL")
log_msg("    q_bundle_max = $q_bundle_max packs×mL (actual max, not q_cig_max × q_ecig_max)")
log_msg("")
log_msg("  Conversion formulas (using L-BFGS estimates):")
log_msg(@sprintf("    α_C_std  = %.4f × %.1f = %.4f", θ_hat_ad[1], q_cig_max, θ_hat_ad[1] * q_cig_max))
log_msg(@sprintf("    α_E_std  = %.4f × %.1f = %.4f", θ_hat_ad[2], q_ecig_max, θ_hat_ad[2] * q_ecig_max))
log_msg(@sprintf("    α_CE_std = %.4f × %.1f = %.4f", θ_hat_ad[3], q_bundle_max, θ_hat_ad[3] * q_bundle_max))
log_msg("    λ_1, λ_2, λ_3, λ_4, ρ, ξ_C, ξ_E, ξ_CE: no conversion needed")
log_msg("")
log_msg("  Note: ω requires E_max from the dynamic model's get_expenditures().")
log_msg("  The dynamic model computes: ω_std = ω_orig × E_max")


################################################################################
# Static Logit WITHOUT State Dependence (ρ)
#
# Re-estimates the model dropping ρ so the fixed effects absorb all
# category-level demand. These estimates provide θ_true values for the
# MC simulation, where the dynamic addiction channel replaces ρ.
#
# Parameters (11): α_C, α_E, α_CE, λ_1, λ_2, λ_3, λ_4, ω, ξ_C, ξ_E, ξ_CE
################################################################################


#############################
# Negative Log-Likelihood
# (No State Dependence)
#############################

"""
Negative log-likelihood for the static logit model WITHOUT state dependence.
Same as neg_log_likelihood() but drops the ρ·lag_match term.

Parameters (11):
  α_C, α_E, α_CE     = consumption utility
  λ_1, λ_2           = flavor effects (baseline, × TYA) — all flavored products
  λ_3, λ_4           = FDA flavor effects (baseline, × TYA) — additional for FDA-authorized
  ω                   = expenditure coefficient
  ξ_C, ξ_E, ξ_CE     = category fixed effects
"""
function neg_log_likelihood_no_rho(θ_vec, N_obs, N_J, tya, y,
                                   q_cig, q_ecig, q_bundle,
                                   is_flavored, is_fda_flavored,
                                   E_obs, fe_C, fe_E, fe_CE)

    # Unpack parameters (11 total — no ρ)
    α_C  = θ_vec[1]   # Cigarette utility (per pack)
    α_E  = θ_vec[2]   # E-cig utility (per mL)
    α_CE = θ_vec[3]   # Bundle interaction
    λ_1  = θ_vec[4]   # Flavor baseline (all flavored)
    λ_2  = θ_vec[5]   # Flavor × TYA interaction (all flavored)
    λ_3  = θ_vec[6]   # FDA flavor baseline (additional for FDA-authorized)
    λ_4  = θ_vec[7]   # FDA flavor × TYA interaction (additional for FDA-authorized)
    ω    = θ_vec[8]   # Expenditure coefficient
    ξ_C  = θ_vec[9]   # Cigarette fixed effect
    ξ_E  = θ_vec[10]  # E-cig fixed effect
    ξ_CE = θ_vec[11]  # Bundle fixed effect

    # Initialize
    neg_LL = zero(eltype(θ_vec))
    v = Vector{eltype(θ_vec)}(undef, N_J)

    # Loop over observations
    for i in 1:N_obs
        tya_i = tya[i]
        y_i = y[i]

        # Compute utilities for all alternatives
        for j in 1:N_J
            v[j] = (α_C * q_cig[j] + α_E * q_ecig[j] + α_CE * q_bundle[j]
                   + is_flavored[j] * (λ_1 + λ_2 * tya_i)
                   + is_fda_flavored[j] * (λ_3 + λ_4 * tya_i)
                   + ω * E_obs[i, j]
                   + ξ_C * fe_C[j] + ξ_E * fe_E[j] + ξ_CE * fe_CE[j])
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
nll_no_rho = θ -> neg_log_likelihood_no_rho(θ, N_obs, N_J, tya, y,
                                             q_cig, q_ecig, q_bundle,
                                             is_flavored, is_fda_flavored,
                                             E_obs, fe_C, fe_E, fe_CE)


#############################
# Estimation: L-BFGS
# (No State Dependence)
#############################

# Parameter names (11 parameters — no ρ)
param_names_no_rho = ["α_C", "α_E", "α_CE", "λ_1", "λ_2", "λ_3", "λ_4", "ω", "ξ_C", "ξ_E", "ξ_CE"]
N_params_no_rho = length(param_names_no_rho)

# Starting values set to all zeros
θ_start_no_rho = zeros(N_params_no_rho)

# Output header
log_msg("\n\n==============================================")
log_msg("Starting L-BFGS optimization (no ρ)")
log_msg("==============================================")

# JIT warmup: first ForwardDiff evaluation triggers compilation for Dual types
log_msg("JIT warmup (compiling ForwardDiff)...")
t_warmup_no_rho = time()
nll_no_rho(θ_start_no_rho)
log_msg("JIT warmup complete in $(round(time() - t_warmup_no_rho, digits=1))s")

# Function to log progress every iteration
function autodiff_callback_no_rho(state)
    if state.iteration % 10 == 0
        log_msg(@sprintf("  Iter %4d | neg LL = %12.4f | |∇| = %.2e",
            state.iteration, state.value, state.g_norm))
    end
    
    return false
end

# Run L-BFGS with automatic differentiation for gradients
t_est_no_rho = time()
result_no_rho = optimize(
    nll_no_rho,
    θ_start_no_rho,
    LBFGS(),
    Optim.Options(
        iterations = 1000,
        f_reltol = 0.0,
        g_tol = 1e-4,
        callback = autodiff_callback_no_rho
    );
    autodiff = :forward
)
est_no_rho_elapsed = time() - t_est_no_rho

# Log final iteration
log_msg(@sprintf("  Iter %4d | neg LL = %12.4f | |∇| = %.2e (final)",
      Optim.iterations(result_no_rho), Optim.minimum(result_no_rho), Optim.g_residual(result_no_rho)))


#############################
# Results (No ρ)
#############################

# Estimates
θ_hat_no_rho = Optim.minimizer(result_no_rho)

# Print results
log_msg("\n\n==============================================")
log_msg("L-BFGS Results (no ρ)")
log_msg("==============================================")
log_msg("Converged: $(Optim.converged(result_no_rho))")
log_msg("Negative log-likelihood: $(round(Optim.minimum(result_no_rho), digits=4))")
log_msg("Log-likelihood: $(round(-Optim.minimum(result_no_rho), digits=4))")
log_msg("Iterations: $(Optim.iterations(result_no_rho))")
log_msg("Total estimation time: $(round(est_no_rho_elapsed, digits=1))s")


#############################
# Standard Errors (No ρ)
# via Autodiff Hessian
#############################

log_msg("\nComputing standard errors via ForwardDiff Hessian...")
t_se_no_rho = time()

# Compute exact Hessian via automatic differentiation
H_no_rho = ForwardDiff.hessian(nll_no_rho, θ_hat_no_rho)

# Symmetrize
H_no_rho = 0.5 * (H_no_rho + H_no_rho')

# Invert to get variance-covariance matrix
V_no_rho = inv(H_no_rho)

# Standard errors = sqrt of diagonal
se_no_rho = sqrt.(abs.(diag(V_no_rho)))

# Show time
se_no_rho_elapsed = time() - t_se_no_rho
log_msg("Standard errors computed in $(round(se_no_rho_elapsed, digits=1))s")

# Check that H is positive definite (eigenvalues > 0)
eigvals_H_no_rho = eigvals(H_no_rho)
log_msg("Hessian eigenvalues: min = $(round(minimum(eigvals_H_no_rho), digits=2)), max = $(round(maximum(eigvals_H_no_rho), digits=2))")
if all(eigvals_H_no_rho .> 0)
    log_msg("Hessian is positive definite (valid MLE)")
else
    log_msg("Hessian is NOT positive definite")
end


#############################
# Results Table (No ρ)
#############################

# Table header
log_msg("")
log_msg(@sprintf("%-8s  %12s  %10s  %10s", "Parameter", "Estimate", "Std Err", "t-stat"))
log_msg(repeat("-", 55))

# Print results
for k in 1:N_params_no_rho
    t_stat = θ_hat_no_rho[k] / se_no_rho[k]
    log_msg(@sprintf("%-8s  %12.6f  %10.6f  %10.4f", param_names_no_rho[k], θ_hat_no_rho[k], se_no_rho[k], t_stat))
end


#############################
# Comparison: With ρ vs
# Without ρ
#############################

log_msg("\n\n==============================================")
log_msg("Comparison: With ρ vs Without ρ (L-BFGS)")
log_msg("==============================================")
log_msg(@sprintf("%-8s  %12s  %10s  %12s  %10s", "Param", "With ρ", "SE", "No ρ", "SE"))
log_msg(repeat("-", 65))

# Map from no-ρ parameter index to with-ρ parameter index
# With ρ:  [α_C, α_E, α_CE, λ_1, λ_2, λ_3, λ_4, ρ, ω, ξ_C, ξ_E, ξ_CE]
# No ρ:    [α_C, α_E, α_CE, λ_1, λ_2, λ_3, λ_4, ω, ξ_C, ξ_E, ξ_CE]
idx_map = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12]  # with-ρ indices for each no-ρ param

for k in 1:N_params_no_rho
    k_rho = idx_map[k]
    log_msg(@sprintf("%-8s  %12.6f  %10.6f  %12.6f  %10.6f",
        param_names_no_rho[k], θ_hat_ad[k_rho], se_ad[k_rho], θ_hat_no_rho[k], se_no_rho[k]))
end

# Print ρ row (only in with-ρ model)
log_msg(@sprintf("%-8s  %12.6f  %10.6f  %12s  %10s", "ρ", θ_hat_ad[8], se_ad[8], "—", "—"))

# Print neg LL comparison
log_msg(@sprintf("\n%-8s  %12.4f  %10s  %12.4f", "neg LL", Optim.minimum(result_ad), "", Optim.minimum(result_no_rho)))


#############################
# Dynamic Model Starting
# Values (No ρ)
#############################

log_msg("\n\n==============================================")
log_msg("Dynamic Model / MC Starting Values (no ρ)")
log_msg("==============================================")
log_msg("These no-ρ estimates can be used as θ_true in the MC simulation.")
log_msg("The fixed effects absorb category-level demand without state dependence.")
log_msg("")
log_msg("  Original-unit estimates (for θ_true in 02_MC_Simulation_Array.jl):")
for k in 1:N_params_no_rho
    log_msg(@sprintf("    %-8s = %.4f", param_names_no_rho[k], θ_hat_no_rho[k]))
end
log_msg("")
log_msg("  Standardized-unit conversion:")
log_msg(@sprintf("    α_C_std  = %.4f × %.1f = %.4f", θ_hat_no_rho[1], q_cig_max, θ_hat_no_rho[1] * q_cig_max))
log_msg(@sprintf("    α_E_std  = %.4f × %.1f = %.4f", θ_hat_no_rho[2], q_ecig_max, θ_hat_no_rho[2] * q_ecig_max))
log_msg(@sprintf("    α_CE_std = %.4f × %.1f = %.4f", θ_hat_no_rho[3], q_bundle_max, θ_hat_no_rho[3] * q_bundle_max))
log_msg("    ω requires E_max from get_expenditures(): ω_std = ω_orig × E_max")
log_msg("    λ_1, λ_2, λ_3, λ_4, ξ_C, ξ_E, ξ_CE: no conversion needed")


# Print message indicating completion
log_msg("\nStatic estimation finished at $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
log_msg("Log saved to: $log_path")

# Close log file
close(log_io)
