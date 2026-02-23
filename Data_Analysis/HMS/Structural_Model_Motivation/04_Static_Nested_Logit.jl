################################################################################
# William Brasic
# The University of Arizona
# wbrasic97@gmail.com
# February 2026
#
# This script estimates the structural parameters of the static NESTED logit
# model by maximizing the sample log-likelihood via L-BFGS with analytical
# gradients (ForwardDiff).
#
# The nested logit relaxes IIA by grouping alternatives into nests with
# stronger within-nest substitution. This is critical for the flavor ban
# counterfactual: when flavored e-cigs are removed, consumers substitute
# primarily toward original e-cigs (same nest) rather than cigarettes.
#
# Nesting structure (4 nests, 1 common σ):
#   Nest 1: Outside option (j=1)                               — 1 alternative
#   Nest 2: Cigarettes (j=2:13)                                — 12 alternatives
#   Nest 3: E-cigs orig+non-FDA flav+FDA flav (j=14:34)       — 21 alternatives
#   Nest 4: Bundles orig+non-FDA flav+FDA flav (j=35:40)      — 6 alternatives
#
# Parameters (13): α_C, α_E, α_CE, λ_1, λ_2, λ_3, λ_4, ρ, ω, ξ_C, ξ_E, ξ_CE, σ
#
# σ is estimated directly on [0, 1) via box-constrained L-BFGS (Fminbox).
# When σ = 0: standard logit. When σ > 0: stronger within-nest substitution.
# The box constraint ensures σ stays in the valid range without reparameterization.
#
# IMPORTANT: Estimates are in ORIGINAL UNITS (utils per pack, utils per mL, etc.)
# The dynamic model (02_Estimation.jl) uses STANDARDIZED data, so it converts
# these estimates to standardized units when using them as starting values:
#   α_C_std  = α_C_orig  × q_cig_max
#   α_E_std  = α_E_orig  × q_ecig_max
#   α_CE_std = α_CE_orig × q_bundle_max
#   ω_std    = ω_orig    × E_max
#   σ: no conversion needed (dimensionless, plugged into dynamic model as fixed)
#
# Progress is logged to Static_Nested_Logit_Estimation_Log_<timestamp>.txt.
################################################################################


#############################
# Preliminaries
#############################

# Whether we are running on the HPC or not
hpc = !Sys.iswindows()

# Static nested logit does not estimate β
ESTIMATE_BETA = false

# Load all functions and packages, set output path, and set working directory
using Dates
timestamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
est_log_name = "Static_Nested_Logit_Estimation_Log_$(timestamp).txt"
if hpc
    # Include functions file
    include("../Dynamic_Model/01_Functions.jl")

    # Output path for results
    output_dir = "./Static_Logit_Results"
    mkpath(output_dir)
    log_path = joinpath(output_dir, est_log_name)

    # Open log file and set global handle
    global log_io = open(log_path, "w")
    log_msg("Static nested logit estimation started at $(timestamp)")

    # Set working directory to where the data CSVs live
    cd("../Data")
else
    # Include functions file
    include("../Dynamic_Model/02_Second_Stage_Estimation/01_Functions.jl")

    # Output path for results
    output_dir = "C:/Users/wbras/OneDrive/Documents/Desktop/UA/4th_Year_Paper/4th_Year_Paper_Data/HMS/2021-Onward/Static_Logit_Results"
    mkpath(output_dir)
    log_path = joinpath(output_dir, est_log_name)

    # Open log file and set global handle
    global log_io = open(log_path, "w")
    log_msg("Static nested logit estimation started at $(timestamp)")

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

# Get FDA flavored indicator by alternative (cat ∈ {4, 7})
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
#   2 = e-cigarettes (ξ_E)
#   3 = bundles (ξ_CE)
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
# Nest Structure
#############################

# Map each alternative to its nest (1-indexed):
#   Nest 1: Outside option (cat_idx = 0)
#   Nest 2: Cigarettes (cat_idx = 1)
#   Nest 3: E-cigs, orig + non-FDA flav + FDA flav (cat_idx = 2, 3, or 4)
#   Nest 4: Bundles, orig + non-FDA flav + FDA flav (cat_idx = 5, 6, or 7)
N_nests = 4
nest_id = zeros(Int, N_J)
for j in 1:N_J
    if cat_idx[j] == 0
        nest_id[j] = 1
    elseif cat_idx[j] == 1
        nest_id[j] = 2
    elseif cat_idx[j] in (2, 3, 4)
        nest_id[j] = 3
    elseif cat_idx[j] in (5, 6, 7)
        nest_id[j] = 4
    end
end

# Alternatives in each nest (vector of index vectors)
nest_alts = [findall(nest_id .== g) for g in 1:N_nests]

# Log nest structure
log_msg("\nNest structure:")
nest_labels = ["Outside", "Cigarettes", "E-cigs", "Bundles"]
for g in 1:N_nests
    log_msg("  Nest $g ($(nest_labels[g])): $(length(nest_alts[g])) alternatives (j = $(nest_alts[g][1]):$(nest_alts[g][end]))")
end


#############################
# Negative Log-Likelihood
#############################

"""
Negative log-likelihood for the static NESTED logit model.
Used with L-BFGS via automatic differentiation (autodiff = :forward).

Parameters (13):
  α_C, α_E, α_CE     = consumption utility
  λ_1, λ_2           = flavor effects (baseline + TYA interaction) — all flavored products
  λ_3, λ_4           = FDA flavor effects (baseline + TYA interaction) — additional for FDA-authorized
  ρ                   = state dependence (lagged category match)
  ω                   = expenditure coefficient
  ξ_C, ξ_E, ξ_CE     = category fixed effects
  σ                   = nesting parameter (box-constrained to [0, 1))

Nested logit choice probability:
  IV_g = logsumexp(v_k/(1-σ) for k in nest g)
  log P(j) = v_j/(1-σ) - σ × IV_{nest(j)} - logsumexp((1-σ) × IV_{g'} for all g')

Data arrays are passed as arguments so they are typed locals inside the
function, avoiding global-variable type instability with ForwardDiff.
"""
function neg_log_likelihood(θ_vec, N_obs, N_J, tya, y,
                            q_cig, q_ecig, q_bundle,
                            is_flavored, is_fda_flavored,
                            lag_match, E_obs,
                            fe_C, fe_E, fe_CE,
                            N_nests, nest_id, nest_alts)

    # Unpack parameters (13 total)
    α_C   = θ_vec[1]    # Cigarette utility (per pack)
    α_E   = θ_vec[2]    # E-cig utility (per mL)
    α_CE  = θ_vec[3]    # Bundle interaction
    λ_1   = θ_vec[4]    # Flavor baseline (all flavored)
    λ_2   = θ_vec[5]    # Flavor × TYA interaction (all flavored)
    λ_3   = θ_vec[6]    # FDA flavor baseline (additional for FDA-authorized)
    λ_4   = θ_vec[7]    # FDA flavor × TYA interaction (additional for FDA-authorized)
    ρ     = θ_vec[8]    # State dependence
    ω     = θ_vec[9]    # Expenditure coefficient
    ξ_C   = θ_vec[10]   # Cigarette fixed effect
    ξ_E   = θ_vec[11]   # E-cig fixed effect
    ξ_CE  = θ_vec[12]   # Bundle fixed effect
    σ = θ_vec[13]        # Nesting parameter (box-constrained to [0, 1))
    one_minus_σ = one(σ) - σ

    # Initialize
    T = eltype(θ_vec)
    neg_LL = zero(T)
    v = Vector{T}(undef, N_J)
    IV = Vector{T}(undef, N_nests)

    # Loop over observations
    for i in 1:N_obs
        tya_i = tya[i]
        y_i = y[i]

        # Compute base utilities for all alternatives
        for j in 1:N_J
            v[j] = (α_C * q_cig[j] + α_E * q_ecig[j] + α_CE * q_bundle[j]
                   + is_flavored[j] * (λ_1 + λ_2 * tya_i)
                   + is_fda_flavored[j] * (λ_3 + λ_4 * tya_i)
                   + ρ * lag_match[i, j]
                   + ω * E_obs[i, j]
                   + ξ_C * fe_C[j] + ξ_E * fe_E[j] + ξ_CE * fe_CE[j])
        end

        # Compute inclusive value for each nest: IV_g = logsumexp(v_k/(1-σ) for k in g)
        for g in 1:N_nests
            alts_g = nest_alts[g]
            K_g = length(alts_g)

            # Find max scaled utility for numerical stability
            max_w = v[alts_g[1]] / one_minus_σ
            for k in 2:K_g
                w_k = v[alts_g[k]] / one_minus_σ
                if w_k > max_w
                    max_w = w_k
                end
            end

            # Compute logsumexp
            s = zero(T)
            for k in 1:K_g
                s += exp(v[alts_g[k]] / one_minus_σ - max_w)
            end
            IV[g] = max_w + log(s)
        end

        # Compute log denominator: logsumexp((1-σ) × IV_g for all g)
        max_nest = one_minus_σ * IV[1]
        for g in 2:N_nests
            val = one_minus_σ * IV[g]
            if val > max_nest
                max_nest = val
            end
        end
        denom_sum = zero(T)
        for g in 1:N_nests
            denom_sum += exp(one_minus_σ * IV[g] - max_nest)
        end
        log_denom = max_nest + log(denom_sum)

        # Log choice probability: log P(j) = v_j/(1-σ) - σ × IV_{nest(j)} - log_denom
        g_y = nest_id[y_i]
        log_P = v[y_i] / one_minus_σ - σ * IV[g_y] - log_denom

        neg_LL -= log_P
    end

    return neg_LL
end

# Wrapper function for optimizer (captures all data in closure)
nll = θ -> neg_log_likelihood(θ, N_obs, N_J, tya, y,
                              q_cig, q_ecig, q_bundle,
                              is_flavored, is_fda_flavored,
                              lag_match, E_obs,
                              fe_C, fe_E, fe_CE,
                              N_nests, nest_id, nest_alts)


#############################
# Estimation: L-BFGS
# (autodiff gradient)
#############################

# Parameter names (13 parameters)
param_names = ["α_C", "α_E", "α_CE", "λ_1", "λ_2", "λ_3", "λ_4", "ρ", "ω", "ξ_C", "ξ_E", "ξ_CE", "σ"]
N_params = length(param_names)

# Starting values: zeros for utility parameters, 0.01 for σ (near standard logit)
θ_start = zeros(N_params)
θ_start[13] = 0.01

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

# Box constraints: σ ∈ [0, 0.99], all other parameters unconstrained
lower_bounds = fill(-Inf, N_params)
upper_bounds = fill(Inf, N_params)
lower_bounds[13] = 0.0    # σ ≥ 0
upper_bounds[13] = 0.99   # σ < 1

# Run box-constrained L-BFGS (Fminbox) with automatic differentiation for gradients
t_est_ad = time()
result_ad = optimize(
    nll,
    lower_bounds,
    upper_bounds,
    θ_start,
    Fminbox(LBFGS()),
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

# Report σ estimate (directly estimated, no transformation needed)
σ_hat_ad = θ_hat_ad[13]
se_σ_ad = se_ad[13]
log_msg(@sprintf("\nσ (nesting) = %.4f  (SE = %.4f, t = %.4f)", σ_hat_ad, se_σ_ad, σ_hat_ad / se_σ_ad))


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

# Report σ estimate with finite-difference SE
se_σ_fd = se_fd[13]
log_msg(@sprintf("\nσ (nesting) = %.4f  (SE = %.4f, t = %.4f)", σ_hat_ad, se_σ_fd, σ_hat_ad / se_σ_fd))


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
starting_param_nm = NamedTuple{(:α_C, :α_E, :α_CE, :λ_1, :λ_2, :λ_3, :λ_4, :ρ, :ω, :ξ_C, :ξ_E, :ξ_CE, :σ)}(Tuple(θ_hat_ad))

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

# Report σ estimate (directly estimated, no transformation needed)
σ_hat_nm = θ_hat_nm[13]
se_σ_nm_ad = se_nm_ad[13]
log_msg(@sprintf("\nσ (nesting) = %.4f  (SE = %.4f, t = %.4f)", σ_hat_nm, se_σ_nm_ad, σ_hat_nm / se_σ_nm_ad))


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

# Report σ estimate with finite-difference SE
se_σ_nm = se_nm[13]
log_msg(@sprintf("\nσ (nesting) = %.4f  (SE = %.4f, t = %.4f)", σ_hat_nm, se_σ_nm, σ_hat_nm / se_σ_nm))


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

# Print σ comparison (directly estimated, appears in the parameter table above as well)
log_msg(@sprintf("\n%-8s  %12.4f  %10.4f  %12.4f  %10.4f", "σ", σ_hat_ad, se_σ_ad, σ_hat_nm, se_σ_nm))


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
log_msg("")
log_msg(@sprintf("  σ (nesting parameter) = %.4f — plug directly into dynamic model (no conversion)", σ_hat_ad))


# Print message indicating completion
log_msg("\nStatic nested logit estimation finished at $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
log_msg("Log saved to: $log_path")

# Close log file
close(log_io)
