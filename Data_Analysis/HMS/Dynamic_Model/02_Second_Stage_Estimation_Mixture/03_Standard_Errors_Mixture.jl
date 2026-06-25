################################################################################
# William Brasic
# The University of Arizona
# wbrasic97@gmail.com
# June 2026
#
# This script computes standard errors for the dynamic model parameter
# estimates via finite differences on the full objective function.
#
# The Hessian H of the negative log-likelihood is approximated numerically
# using parameter-relative step sizes h_k = max(|θ_hat[k]| * 1e-3, 1e-4):
#   - Diagonal:     H[k,k] ≈ (f(θ+h_k·e_k) - 2·f(θ) + f(θ-h_k·e_k)) / h_k^2
#   - Off-diagonal: H[k,l] ≈ (f(θ+h_k·e_k+h_l·e_l) - f(θ+h_k·e_k-h_l·e_l)
#                            -  f(θ-h_k·e_k+h_l·e_l) + f(θ-h_k·e_k-h_l·e_l)) / (4·h_k·h_l)
#
# Each evaluation of f(θ) requires solving VFI from scratch
#
# Variance-covariance matrix: H^{-1}
# Standard errors: SE[k] = sqrt(H^{-1}[k,k])
################################################################################


#############################
# Preliminaries
#############################

# β can vary across runs but is never estimated - read from ENV to match 02_Estimation_Mixture.jl
BETA = parse(Float64, get(ENV, "BETA", "1.0"))

# Fixed parameters (ψ_1, ψ_2 always fixed; β fixed per run but can vary across runs)
ψ_1 = 0.10
ψ_2 = 0.90
β   = BETA

# Fixed flavored habit decay rate ψ_3 (never estimated, read from ENV for grid search)
PSI_3 = parse(Float64, get(ENV, "PSI_3", "0.75"))

# Whether ψ_3 was estimated jointly (22nd parameter) - must match the estimation run
ESTIMATE_PSI_3 = parse(Bool, get(ENV, "ESTIMATE_PSI_3", "false"))

# Enable warm-start VFI: reuse the previous evaluation's converged V as the
# initial guess. The small θ perturbations (h ~ 1e-3) mean the previous V is
# an excellent starting point, reducing VFI iterations for each perturbation.
WARM_START = true

# Tighten VFI convergence tolerance for SE computation relative to actual estimation
VFI_TOL = 1e-6

# Load all functions and packages from the functions file
include("01_Functions_Mixture.jl")
using LinearAlgebra

# Detect whether we are running on the HPC (any non-Windows system)
HPC = !Sys.iswindows()

# Initialize warm-start globals for SE computation
ra_outer_try = 1
ra_inner_run = 1
V_warm_est_1 = nothing
V_warm_est_2 = nothing
V_warm_est_3 = nothing
last_ra_phase_est = (1, 1)

# Construct psi and beta tags for output directory and file naming.
psi_tag = "Psi2_09_Psi1_01"
beta_tag = "Beta_$(numeric_tag(BETA))"
psi_3_tag = ESTIMATE_PSI_3 ? "Psi3_Est" : "Psi3_$(numeric_tag(PSI_3))"


#############################
# Output Paths
#############################

# File paths if on HPC or not
if HPC

    # Output path for results (use absolute path so it's unaffected by later cd)
    output_dir = abspath("./Dynamic_Model_Mixture_V2_$(psi_tag)_$(beta_tag)_$(psi_3_tag)_Results")
    mkpath(output_dir)

    # Set working directory to where the data CSVs live
    cd("/home/u2/wbrasic/4th_Year_Paper/Data")
else

    # Output path for results (local Windows path, includes beta tag in directory name)
    output_dir = "C:/Users/wbras/OneDrive/Documents/Desktop/UA/4th_Year_Paper/4th_Year_Paper_Data/HMS/2021-Onward/Dynamic_Model/Dynamic_Model_Mixture_V2_$(psi_tag)_$(beta_tag)_$(psi_3_tag)_Results"
    mkpath(output_dir)

    # Set working directory to where the data CSVs live
    cd("C:/Users/wbras/OneDrive/Documents/Desktop/UA/4th_Year_Paper/4th_Year_Paper_Data/HMS/2021-Onward/Dynamic_Model/Data")
end

log_path = joinpath(output_dir, "Dynamic_Model_Mixture_V2_$(psi_tag)_$(beta_tag)_$(psi_3_tag)_SE_Log.txt")

# Open log file for writing (log_io is defined as a global in 01_Functions.jl)
log_io = open(log_path, "w")

# Print and log SE computation start time
log_msg("SE computation ($(psi_tag), $(beta_tag)) started at $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")


#############################
# State Spaces and Choices
#############################

# Start timer for data prep
t_setup = time()

# Load fixed parameters: only δ is taken from get_fixed_parameters();
# ψ_1, ψ_2, and β are hardcoded above
_, _, _, δ = get_fixed_parameters();

# Get number of addiction states and normalized addiction grids for fast and slow stocks
N_A_f, A_f = get_addiction_space(ψ_2; N_A=5);
N_A_s, A_s = get_addiction_space(ψ_1; N_A=10);

# Get number of observations (N_HHT), number of alternatives (N_J), and choice matrix J
_, N_J, J = get_product_choices();

# Convert choice matrix J to choice vector y where y[i] = chosen alternative index for observation i
y = get_hh_choices(J);

# Get household identifiers (pre-loaded to avoid repeated CSV reads in objective)
hh_codes = get_hh_codes();

# Pre-compute contiguous household index ranges for mixture log-likelihood
hh_ranges = precompute_hh_ranges(hh_codes);

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

# Get flavor lock-in indicator: orig ecig (cat 2) and orig bundle (cat 5) - ecig-side lock-in for γ_2
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
# Household Price Trajectories
#############################

# Map observed household prices to continuous values for likelihood interpolation
# p_continuous is N × 2 (cig price, ecig price) - representative (median-across-bins) prices for VFI interpolation
# P_obs_cig / P_obs_ecig are N × N_J matrices of actual bin-specific prices 
_, p_continuous, P_obs_cig, P_obs_ecig = map_prices_to_grid(N_P, P, Pcomb, N_J);


#############################
# Pre-compute Addiction
# Objects (at Fixed ψ_2, ψ_1)
#############################

# Pre-compute addiction transition brackets for all (alternative, addiction state) pairs.
# Fast stock (pre-computed at fixed ψ_2)
af_lower_current, af_upper_current, af_weight_current = precompute_addiction_transitions(N_J, N_A_f, ψ_2, A_f, n)

# Slow stock (pre-computed at fixed ψ_1)
as_lower_current, as_upper_current, as_weight_current = precompute_addiction_transitions(N_J, N_A_s, ψ_1, A_s, n)

# Pre-compute initial addiction stocks via fixed-point iteration and simulate
# addiction trajectories for both fast and slow stocks.
# Fast stock
af0_current, _ = get_initial_addiction_stock(ψ_2, A_f, n, y, hh_codes)
af_continuous_current = simulate_addiction_trajectories(N_A_f, ψ_2, A_f, n, y, hh_codes, af0_current)

# Slow stock
as0_current, _ = get_initial_addiction_stock(ψ_1, A_s, n, y, hh_codes)
as_continuous_current = simulate_addiction_trajectories(N_A_s, ψ_1, A_s, n, y, hh_codes, as0_current)

# Flavored habit stock (pre-computed at ENV PSI_3; objective recomputes at each candidate ψ_3 when ESTIMATE_PSI_3=true)
n_flav     = Float64.(is_flavored)             # ∈ {0.0, 1.0}
n_flav_max = 1.0                               # binary max; rescale γ_2, γ_3, γ_4 by × ψ_3
N_A_flav, A_flav = get_addiction_space(PSI_3; N_A=10)
aflav_lower_current, aflav_upper_current, aflav_weight_current = precompute_addiction_transitions(N_J, N_A_flav, PSI_3, A_flav, n_flav)
aflav0_current, _ = get_initial_addiction_stock(PSI_3, A_flav, n_flav, y, hh_codes)
aflav_continuous_current = simulate_addiction_trajectories(N_A_flav, PSI_3, A_flav, n_flav, y, hh_codes, aflav0_current)

# Log data setup completion time and sample size
setup_elapsed = time() - t_setup
log_msg("Data loading complete in $(round(setup_elapsed, digits=1))s")
log_msg("Observations: $(length(y))")


#############################
# Load Estimated Parameters
#############################

# Read θ_hat from the estimates file produced by 02_Estimation_Mixture.jl (in Estimates subdirectory)
estimates_subdir = joinpath(output_dir, "Dynamic_Model_Mixture_V2_$(psi_tag)_$(beta_tag)_$(psi_3_tag)_Estimates")
estimates_path = joinpath(estimates_subdir, "Dynamic_Model_Mixture_V2_$(psi_tag)_$(beta_tag)_$(psi_3_tag)_Estimates.csv")
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
# I approximate this using central differences with parameter-relative step
# sizes h_k = max(|θ_hat[k]| * 1e-3, 1e-4). This scales with parameter
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
# Each evaluation calls the full objective, so this is slow.
# Total evaluations: 1 center + N_params diagonal (2 each) + N_params*(N_params-1)/2
# off-diagonal (4 each) = 1 + 2*D + 4*D*(D-1)/2 where D = N_params.

# Print and log Hessian computation header
log_msg("\n==============================================")
log_msg("Computing Hessian via finite differences")
log_msg("==============================================")

# Start Hessian computation timer
t_hessian = time()

# Parameter-relative step sizes for finite differences
# Each h_k scales with the magnitude of θ_hat[k] (relative step of 1e-3) with a
# floor of 1e-4 for near-zero parameters. 
h_vec = [max(abs(θ_hat[k]) * 1e-3, 1e-4) for k in 1:N_params]

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

        global pair_count += 1
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
v_diag = diag(V)
neg_idx = findall(v_diag .< 0)
if !isempty(neg_idx)
    log_msg("WARNING: $(length(neg_idx)) negative variance(s) on diagonal of V - SEs for these params are unreliable:")
    for k in neg_idx
        log_msg("  $(param_names[k]) [k=$k]: V[$k,$k] = $(round(v_diag[k], sigdigits=4))")
    end
end
se = sqrt.(abs.(v_diag))


#############################
# Diagnostics
#############################

# Print and log Hessian eigenvalue diagnostics
eigvals_H = eigvals(H)
log_msg("Hessian eigenvalues: min = $(round(minimum(eigvals_H), digits=2)), max = $(round(maximum(eigvals_H), digits=2))")
if all(eigvals_H .> 0)
    log_msg("Hessian is positive definite (valid MLE)")
else
    log_msg("WARNING: Hessian is NOT positive definite - SEs may be unreliable")
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

# Save _SE.csv with 2 rows: parameter names, standard errors
se_path = joinpath(estimates_subdir, "Dynamic_Model_Mixture_V2_$(psi_tag)_$(beta_tag)_$(psi_3_tag)_SE.csv")
open(se_path, "w") do io
    println(io, join(param_names, ","))
    println(io, join([@sprintf("%.10f", se[k]) for k in 1:N_params], ","))
end
# Print and log SE save location
log_msg("\nSEs saved to: $se_path")

# Save full variance-covariance matrix (needed for delta method SEs)
vcov_path = joinpath(estimates_subdir, "Dynamic_Model_Mixture_V2_$(psi_tag)_$(beta_tag)_$(psi_3_tag)_VCov.csv")
vcov_df = DataFrame(V, param_names)
CSV.write(vcov_path, vcov_df)
log_msg("Variance-covariance matrix saved to: $vcov_path")


#############################
# Delta Method SEs for
# Mixing Probabilities
#############################

# --- Why the delta method is needed here ---
#
# The optimizer estimates four mixing-weight parameters directly:
#   π_0_2, π_TYA_2, π_0_3, π_TYA_3
#
# But the economically meaningful quantities are the implied TYPE PROBABILITIES:
#   P(Type 1 | s), P(Type 2 | s), P(Type 3 | s)
# where s = household TYA share (fraction of months with TYA present).
#
# These probabilities are NONLINEAR functions of the estimated parameters
# (they involve exp() and a denominator), so we cannot simply scale a standard
# error. Instead, we apply the delta method: approximate the variance of a
# nonlinear function g(θ) around the MLE θ_hat using a first-order Taylor
# expansion,
#
#   Var(g(θ_hat)) ≈ ∇g(θ_hat)' · V · ∇g(θ_hat)
#   SE(g(θ_hat))  = sqrt(Var(g(θ_hat)))
#
# where V = H⁻¹ is the estimated variance-covariance matrix from above, and
# ∇g(θ_hat) is the gradient of g evaluated at the estimates.
#
# --- Model for mixing weights ---
#
# Each household h with TYA share s_h belongs to one of K=3 latent types.
# The mixing probabilities are a K=3 softmax over two logit indices:
#
#   l2(s) = π_0_2 + π_TYA_2 · s      (log-odds of Type 2 vs. Type 1)
#   l3(s) = π_0_3 + π_TYA_3 · s      (log-odds of Type 3 vs. Type 1)
#
#   P(Type 1 | s) = 1 / (1 + exp(l2) + exp(l3))
#   P(Type 2 | s) = exp(l2) / (1 + exp(l2) + exp(l3))
#   P(Type 3 | s) = exp(l3) / (1 + exp(l2) + exp(l3))
#
# We report mixing probabilities at two representative TYA-share values:
#   s = 0: household with TYA never present  → logits = (π_0_2, π_0_3)
#   s = 1: household with TYA always present → logits = (π_0_2+π_TYA_2, π_0_3+π_TYA_3)

# Locate the four mixing parameters in the θ vector.
# Their positions depend on whether ψ_3 was estimated as a 27th parameter:
#   ESTIMATE_PSI_3=true  (27 params): [..., π_0_2(23), π_TYA_2(24), π_0_3(25), π_TYA_3(26), ψ_3(27)]
#   ESTIMATE_PSI_3=false (26 params): [..., π_0_2(23), π_TYA_2(24), π_0_3(25), π_TYA_3(26)]
# In both cases the mixing params are the last 4 non-ψ_3 entries, so we index
# from the end of the parameter vector, shifting by 1 when ψ_3 occupies slot N_params.
idx_pi_0_2   = ESTIMATE_PSI_3 ? N_params - 4 : N_params - 3
idx_pi_TYA_2 = ESTIMATE_PSI_3 ? N_params - 3 : N_params - 2
idx_pi_0_3   = ESTIMATE_PSI_3 ? N_params - 2 : N_params - 1
idx_pi_TYA_3 = ESTIMATE_PSI_3 ? N_params - 1 : N_params

# softmax3_se: compute K=3 mixing probabilities and their delta-method SEs.
#
# Arguments:
#   l2, l3  - evaluated logit indices at the chosen TYA share s
#               l2 = π_0_2 + π_TYA_2 · s,  l3 = π_0_3 + π_TYA_3 · s
#   idx2    - [idx_pi_0_2, idx_pi_TYA_2]: positions of Type 2 mixing params in θ
#   idx3    - [idx_pi_0_3, idx_pi_TYA_3]: positions of Type 3 mixing params in θ
#   x2, x3 - covariate vectors [1, s] encoding how l2 and l3 depend on (π_0, π_TYA)
#               x2 = x3 = [1.0, 0.0] for s=0 (TYA never present)
#               x2 = x3 = [1.0, 1.0] for s=1 (TYA always present)
#   V       - full D×D variance-covariance matrix from H⁻¹
#
# Returns: (p1, p2, p3, se1, se2, se3)
function softmax3_se(l2, l3, idx2, idx3, x2, x3, V)

    # Compute the softmax denominator (shared across all three types)
    denom = 1.0 + exp(l2) + exp(l3)

    # Compute the three mixing probabilities from the softmax formula
    p1 = 1.0 / denom          # P(Type 1 | s) - base category, no logit
    p2 = exp(l2) / denom      # P(Type 2 | s) - driven by l2
    p3 = exp(l3) / denom      # P(Type 3 | s) - driven by l3

    # D = total number of parameters (length of θ, = size of the VCov matrix)
    D = size(V, 1)

    # --- Gradient for Type 1: ∇p_1 ---
    # Type 1 is the base category with no logit of its own. Its probability falls
    # whenever l2 or l3 rises, so all four partial derivatives are negative.
    # ∂p_1/∂π_0_2   = -p_1·p_2        (Type 1 is not Type 2, so no own-logit term)
    # ∂p_1/∂π_TYA_2 = -p_1·p_2·s      (same, scaled by TYA share s)
    # ∂p_1/∂π_0_3   = -p_1·p_3        (Type 1 is not Type 3)
    # ∂p_1/∂π_TYA_3 = -p_1·p_3·s
    # All other entries are 0.
    g1 = zeros(D)
    for (d, coeff) in zip(idx2, x2 .* (p1 * (0.0 - p2)))    # contributions from π_0_2, π_TYA_2
        g1[d] += coeff
    end
    for (d, coeff) in zip(idx3, x3 .* (p1 * (0.0 - p3)))    # contributions from π_0_3, π_TYA_3
        g1[d] += coeff
    end

    # --- Gradient for Type 2: ∇p_2 ---
    # Type 2 has its own logit l2, so its own-logit derivative has a +1 term.
    # ∂p_2/∂π_0_2   = p_2·(1 - p_2)   (own logit: positive, like a logit slope)
    # ∂p_2/∂π_TYA_2 = p_2·(1 - p_2)·s (same, scaled by TYA share s)
    # ∂p_2/∂π_0_3   = -p_2·p_3        (cross logit: Type 2 loses share when l3 rises)
    # ∂p_2/∂π_TYA_3 = -p_2·p_3·s
    # All other entries are 0.
    g2 = zeros(D)
    for (d, coeff) in zip(idx2, x2 .* (p2 * (1.0 - p2)))    # contributions from π_0_2, π_TYA_2
        g2[d] += coeff
    end
    for (d, coeff) in zip(idx3, x3 .* (p2 * (0.0 - p3)))    # contributions from π_0_3, π_TYA_3
        g2[d] += coeff
    end

    # --- Gradient for Type 3: ∇p_3 ---
    # Type 3 has its own logit l3, so its own-logit derivative has a +1 term.
    # ∂p_3/∂π_0_2   = -p_3·p_2        (cross logit: Type 3 loses share when l2 rises)
    # ∂p_3/∂π_TYA_2 = -p_3·p_2·s
    # ∂p_3/∂π_0_3   = p_3·(1 - p_3)   (own logit: positive)
    # ∂p_3/∂π_TYA_3 = p_3·(1 - p_3)·s (same, scaled by TYA share s)
    # All other entries are 0.
    g3 = zeros(D)
    for (d, coeff) in zip(idx2, x2 .* (p3 * (0.0 - p2)))    # contributions from π_0_2, π_TYA_2
        g3[d] += coeff
    end
    for (d, coeff) in zip(idx3, x3 .* (p3 * (1.0 - p3)))    # contributions from π_0_3, π_TYA_3
        g3[d] += coeff
    end

    # Apply the delta method: SE(p_k) = sqrt(∇p_k' · V · ∇p_k)
    # dot(g, V*g) computes g'Vg as a matrix-vector product followed by a dot product.
    # abs() guards against tiny negative values from numerical imprecision in H⁻¹.
    se1 = sqrt(abs(dot(g1, V * g1)))
    se2 = sqrt(abs(dot(g2, V * g2)))
    se3 = sqrt(abs(dot(g3, V * g3)))

    return p1, p2, p3, se1, se2, se3
end

# Build covariate vectors x = [1, s] for the two representative TYA shares.
# x encodes how the logit l = π_0 + π_TYA · s depends on each mixing parameter:
#   ∂l/∂π_0   = 1  (the first element of x)
#   ∂l/∂π_TYA = s  (the second element of x)
x2_notya = [1.0, 0.0]    # s=0: TYA never present → l2 = π_0_2
x3_notya = [1.0, 0.0]    # s=0: TYA never present → l3 = π_0_3
x2_tya   = [1.0, 1.0]    # s=1: TYA always present → l2 = π_0_2 + π_TYA_2
x3_tya   = [1.0, 1.0]    # s=1: TYA always present → l3 = π_0_3 + π_TYA_3

# Evaluate the logit indices at s=0 (TYA never present)
l2_notya = θ_hat[idx_pi_0_2]                              # π_0_2 + π_TYA_2 · 0 = π_0_2
l3_notya = θ_hat[idx_pi_0_3]                              # π_0_3 + π_TYA_3 · 0 = π_0_3

# Evaluate the logit indices at s=1 (TYA always present)
l2_tya   = θ_hat[idx_pi_0_2] + θ_hat[idx_pi_TYA_2]      # π_0_2 + π_TYA_2 · 1
l3_tya   = θ_hat[idx_pi_0_3] + θ_hat[idx_pi_TYA_3]      # π_0_3 + π_TYA_3 · 1

# Compute mixing probabilities and delta-method SEs for s=0 (TYA never present)
p1_notya, p2_notya, p3_notya, se1_notya, se2_notya, se3_notya =
    softmax3_se(l2_notya, l3_notya, [idx_pi_0_2, idx_pi_TYA_2], [idx_pi_0_3, idx_pi_TYA_3], x2_notya, x3_notya, V)

# Compute mixing probabilities and delta-method SEs for s=1 (TYA always present)
p1_tya, p2_tya, p3_tya, se1_tya, se2_tya, se3_tya =
    softmax3_se(l2_tya, l3_tya, [idx_pi_0_2, idx_pi_TYA_2], [idx_pi_0_3, idx_pi_TYA_3], x2_tya, x3_tya, V)

# Print and log delta method mixing probability results
log_msg("\n\n==============================================")
log_msg("Delta Method SEs for Mixing Probabilities (K=3 Softmax)")
log_msg("==============================================")
log_msg("")
log_msg(@sprintf("  P(Type 1 | TYA Never Present)  = %.4f  (SE = %.4f)", p1_notya, se1_notya))
log_msg(@sprintf("  P(Type 2 | TYA Never Present)  = %.4f  (SE = %.4f)", p2_notya, se2_notya))
log_msg(@sprintf("  P(Type 3 | TYA Never Present)  = %.4f  (SE = %.4f)", p3_notya, se3_notya))
log_msg("")
log_msg(@sprintf("  P(Type 1 | TYA Always Present) = %.4f  (SE = %.4f)", p1_tya, se1_tya))
log_msg(@sprintf("  P(Type 2 | TYA Always Present) = %.4f  (SE = %.4f)", p2_tya, se2_tya))
log_msg(@sprintf("  P(Type 3 | TYA Always Present) = %.4f  (SE = %.4f)", p3_tya, se3_tya))


#############################
# Rescaling SEs to
# Original (Unstandardized)
# Units
#############################

# All structural parameters were estimated in STANDARDIZED units because
# quantities and nicotine were divided by their sample maxima before entering
# the objective. To report interpretable estimates and SEs, we rescale back to
# original units using the same transformations applied during estimation.
#
# There are three groups, depending on what type of rescaling is required:
#
# -----------------------------------------------------------------------
# GROUP 1 - Divide by a fixed data constant (q_max, n_max)
# -----------------------------------------------------------------------
#
#   During estimation, quantity q_raw was replaced by q_std = q_raw / q_max,
#   so the estimated coefficient absorbs the factor (1/q_max). To recover
#   the original-unit coefficient, divide by q_max:
#
#       α_C_orig  = α_C_std  / q_cig_max
#       α_E_orig  = α_E_std  / q_ecig_max
#       α_CE_orig = α_CE_std / q_bundle_max
#       ω_C_orig  = ω_C_std  / q_cig_max
#       ω_E_orig  = ω_E_std  / q_ecig_max
#       γ_1_orig  = γ_1_std  / n_max
#
#   Because q_max and n_max are FIXED DATA CONSTANTS (not estimated), the SE
#   transforms by the same factor:
#
#       SE(α_C_orig) = SE(α_C_std) / q_cig_max
#
#   No delta method is needed for Group 1 - scaling by a constant is linear.
#
# -----------------------------------------------------------------------
# GROUP 2 - Multiply by estimated ψ_3 (requires the delta method)
# -----------------------------------------------------------------------
#
#   The flavored habit stock entering utility is the NORMALIZED stock:
#       ã_flav = ψ_3 · a_raw
#   where a_raw is the raw (unnormalized) habit stock and ψ_3 is the decay
#   rate. The utility contribution of γ_k in standardized units is:
#       γ_k_std · ã_flav = γ_k_std · ψ_3 · a_raw
#   So the coefficient on a_raw (original units) is:
#       γ_k_orig = γ_k_std · ψ_3 / n_flav_max
#   where n_flav_max = 1.0 (the binary indicator input is already in [0,1]).
#
#   WHY THE DELTA METHOD IS NEEDED:
#   When ψ_3 is estimated jointly (ESTIMATE_PSI_3=true), γ_k_orig is a
#   PRODUCT of two estimated quantities. The SE of a product is not simply
#   the product of the individual SEs as we must account for (a) the variance
#   in γ_k, (b) the variance in ψ_3, and (c) the covariance between them.
#
#   DERIVATION:
#   Let f(γ, ψ) = γ · ψ / C  where C = n_flav_max (a data constant).
#   The gradient of f with respect to the two parameters is:
#       ∂f/∂γ = ψ / C
#       ∂f/∂ψ = γ / C
#   All other partial derivatives are zero.
#
#   By the delta method:
#       Var(f) = (∂f/∂γ)² Var(γ) + (∂f/∂ψ)² Var(ψ) + 2·(∂f/∂γ)·(∂f/∂ψ)·Cov(γ,ψ)
#              = (ψ/C)² · V[γ,γ]  +  (γ/C)² · V[ψ,ψ]  +  2·(ψ/C)·(γ/C)·V[γ,ψ]
#              = [ψ² · V[γ,γ]  +  γ² · V[ψ,ψ]  +  2·γ·ψ · V[γ,ψ]] / C²
#
#   Therefore:
#       SE(γ_k_orig) = sqrt(ψ² · V[γ,γ] + γ² · V[ψ,ψ] + 2·γ·ψ · V[γ,ψ]) / C
#
#   where V[γ,γ], V[ψ,ψ], V[γ,ψ] are the relevant entries from the full
#   variance-covariance matrix V = H⁻¹ computed above.
#
#   When ψ_3 is NOT estimated (ESTIMATE_PSI_3=false), it is a fixed constant,
#   so γ_k_orig = γ_k_std · PSI_3 / C is linear in γ_k alone, and:
#       SE(γ_k_orig) = SE(γ_k_std) · PSI_3 / C   (no delta method needed)
#
# -----------------------------------------------------------------------
# GROUP 3 - No rescaling needed
# -----------------------------------------------------------------------
#
#   λ_1–λ_4 multiply binary (0/1) flavor indicators - no standardization.
#   ξ_{kC}, ξ_{kE}, ξ_{kCE} are additive category fixed effects - no standardization.
#   π_0_2, π_TYA_2, π_0_3, π_TYA_3 enter the softmax directly - no standardization.
#   ψ_3 is a decay rate in (0,1) - no standardization.
#   Estimates and SEs for these parameters are already in interpretable units.

log_msg("\n\n==============================================")
log_msg("Rescaling to Original Units")
log_msg("==============================================")

# --- Group 1: divide by fixed data constant ---
# SE scales by the same factor as the estimate - no delta method required.
log_msg("")
log_msg("Group 1 - divide by data constant (SE scales identically):")
log_msg(@sprintf("  α_C_orig  = α_C  / q_cig_max    = %.10f / %.1f = %.10f   SE = %.10f", θ_hat[1],  q_cig_max,    θ_hat[1]/q_cig_max,    se[1]/q_cig_max))
log_msg(@sprintf("  α_E_orig  = α_E  / q_ecig_max   = %.10f / %.1f = %.10f   SE = %.10f", θ_hat[2],  q_ecig_max,   θ_hat[2]/q_ecig_max,   se[2]/q_ecig_max))
log_msg(@sprintf("  α_CE_orig = α_CE / q_bundle_max = %.10f / %.1f = %.10f   SE = %.10f", θ_hat[3],  q_bundle_max, θ_hat[3]/q_bundle_max, se[3]/q_bundle_max))
log_msg(@sprintf("  ω_C_orig  = ω_C  / q_cig_max    = %.10f / %.1f = %.10f   SE = %.10f", θ_hat[12], q_cig_max,    θ_hat[12]/q_cig_max,   se[12]/q_cig_max))
log_msg(@sprintf("  ω_E_orig  = ω_E  / q_ecig_max   = %.10f / %.1f = %.10f   SE = %.10f", θ_hat[13], q_ecig_max,   θ_hat[13]/q_ecig_max,  se[13]/q_ecig_max))
log_msg(@sprintf("  γ_1_orig  = γ_1  / n_max        = %.10f / %.1f = %.10f   SE = %.10f", θ_hat[8],  n_max,        θ_hat[8]/n_max,        se[8]/n_max))

# --- Group 2: multiply by ψ_3 ---
log_msg("")
log_msg("Group 2 - multiply by ψ_3 (delta method SEs when ψ_3 estimated, simple scaling when fixed):")

if ESTIMATE_PSI_3

    # ψ_3 occupies the last position in θ when estimated
    idx_psi_3 = N_params          # position 27 in the 27-parameter model
    ψ_3_hat   = θ_hat[idx_psi_3] # point estimate of ψ_3

    # Loop over γ_2 (pos 9), γ_3 (pos 10), γ_4 (pos 11)
    for (name, idx) in [("γ_2", 9), ("γ_3", 10), ("γ_4", 11)]

        g = θ_hat[idx]     # point estimate of γ_k (standardized)
        p = ψ_3_hat        # point estimate of ψ_3

        # Original-unit estimate: γ_k_orig = γ_k · ψ_3 / n_flav_max
        orig = g * p / n_flav_max

        # Delta-method variance: [ψ² · Var(γ) + γ² · Var(ψ) + 2·γ·ψ · Cov(γ,ψ)] / C²
        # V[idx, idx]           = Var(γ_k)      from the VCov matrix
        # V[idx_psi_3, idx_psi_3] = Var(ψ_3)   from the VCov matrix
        # V[idx, idx_psi_3]     = Cov(γ_k, ψ_3) from the VCov matrix
        var_prod = p^2 * V[idx, idx] + g^2 * V[idx_psi_3, idx_psi_3] + 2 * g * p * V[idx, idx_psi_3]

        # SE = sqrt(Var) / C - abs() guards against tiny negative variance from numerical noise
        se_orig = sqrt(abs(var_prod)) / n_flav_max

        log_msg(@sprintf("  %s_orig = %s × ψ_3 / n_flav_max = %.10f × %.10f / %.4f = %.10f   SE(delta) = %.10f", name, name, g, p, n_flav_max, orig, se_orig))
    end

else

    # ψ_3 is a fixed constant (not estimated), so γ_k_orig = γ_k · PSI_3 / C
    # is linear in γ_k → SE scales by the same constant, no delta method needed.
    for (name, idx) in [("γ_2", 9), ("γ_3", 10), ("γ_4", 11)]
        orig    = θ_hat[idx] * PSI_3 / n_flav_max    # point estimate in original units
        se_orig = se[idx]    * PSI_3 / n_flav_max    # SE scales linearly
        log_msg(@sprintf("  %s_orig = %s × ψ_3 / n_flav_max = %.10f × %.4f / %.4f = %.10f   SE = %.10f  (ψ_3 fixed, no delta method needed)", name, name, θ_hat[idx], PSI_3, n_flav_max, orig, se_orig))
    end

end

# --- Group 3: no rescaling ---
log_msg("")
log_msg("Group 3 - no rescaling needed (parameters already in interpretable units):")
log_msg("  λ_1 through λ_4, ξ's (k=1,2,3), π_0_2, π_TYA_2, π_0_3, π_TYA_3, ψ_3")

# Print and log SE computation finished message
log_msg("\nSE computation finished at $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
log_msg("Log saved to: $log_path")

# Close the log file handle
close(log_io)
