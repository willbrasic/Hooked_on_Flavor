################################################################################
# William Brasic
# The University of Arizona
# wbrasic97@gmail.com
# June 2026
#
# Bootstrap counterfactuals: each Slurm array task draws θ_b ~ N(θ̂, V̂Cov)
# (task 0 uses θ̂ itself), then runs every policy at a single fixed β, tagging
# output with the draw index b. 03_CF_Bootstrap_Mixture_Aggregation.jl pools
# the draws into bootstrap confidence intervals.
#
# One run does everything:
#   1. Status quo: solve the three type-specific value functions once, compute
#      posterior type weights, and forward-simulate every household with common
#      random numbers.
#   2. Comprehensive ban, non-FDA ban, and the flavor tax at the TAX_GRID rates:
#      solve the policy value functions, simulate, and write category shares,
#      addiction stocks, welfare, and quantities overall and by TYA status and
#      modal latent type, plus initiation among baseline non-users for three
#      outcomes (flavored e-cigarettes, cigarettes, unflavored e-cigarettes),
#      with the non-user base read off the observed purchase history.
#   3. Tax-versus-ban frontier: the remaining FRONTIER_TAX_GRID rates are solved
#      and simulated for the e-cigarette-persistent households (modal type 2),
#      and the frontier files report each rate's addiction reduction as a
#      fraction of the ban's, the consumer surplus cost in dollars, and the
#      rate τ* at which the tax matches the ban.
################################################################################


#############################
# 1. Preliminaries
#############################

# Flavored habit decay rate ψ_3 is parameter 28 of the estimates. The ENV value
# only seeds the habit grid and is replaced by the draw's ψ_3 in section 13.
PSI_3 = parse(Float64, get(ENV, "PSI_3", "0.50"))

# β for this run (ENV, default 1.0)
BETA_VAL = parse(Float64, get(ENV, "BETA", "1.0"))

# Bootstrap draw index (one per Slurm array task); task 0 is the point-estimate run
b = parse(Int, get(ENV, "SLURM_ARRAY_TASK_ID", "1"))

# Detect whether I am running on the HPC (any non-Windows system)
HPC = !Sys.iswindows()

# CRN seeding for forward simulations
using Random

# Set output path and working directory
if HPC

    # Load mixture estimation functions and packages
    include("/home/u2/wbrasic/4th_Year_Paper/Dynamic_Model/02_Second_Stage_Estimation_Mixture/01_Functions_Mixture.jl")

    # Load counterfactual-specific functions
    include("/home/u2/wbrasic/4th_Year_Paper/Dynamic_Model/05_CF_Mixture_Bootstrap/01_CF_Functions_Mixture.jl")

    # Construct psi and psi_3 tags for directory and file naming
    psi_tag   = "Psi2_09_Psi1_01"
    _, _, β_naming, _ = get_fixed_parameters()
    beta_tag  = numeric_tag(BETA_VAL)

    # Output path is β-specific; each β job writes to its own top-level directory.
    output_dir = "/home/u2/wbrasic/4th_Year_Paper/Dynamic_Model/05_CF_Mixture_Bootstrap/CF_Bootstrap_Mixture_$(psi_tag)_Beta_$(beta_tag)_Results"

    # Create output directory if it doesn't exist
    mkpath(output_dir)

    # Set working directory to where the data CSVs live (absolute path)
    cd("/home/u2/wbrasic/4th_Year_Paper/Data")
else

    # Load mixture estimation functions and packages
    include("../02_Second_Stage_Estimation_Mixture/01_Functions_Mixture.jl")

    # Load counterfactual-specific functions
    include("01_CF_Functions_Mixture.jl")

    # Construct psi and psi_3 tags
    psi_tag   = "Psi2_09_Psi1_01"
    _, _, β_naming, _ = get_fixed_parameters()
    beta_tag  = numeric_tag(BETA_VAL)

    # Output path is β-specific; each β job writes to its own top-level directory.
    output_dir = "C:/Users/wbras/OneDrive/Documents/Desktop/UA/4th_Year_Paper/4th_Year_Paper_Data/HMS/2021-Onward/Dynamic_Model/CF_Bootstrap_Mixture_$(psi_tag)_Beta_$(beta_tag)_Results"

    # Create output directory if it doesn't exist
    mkpath(output_dir)

    # Set working directory to where the data CSVs live
    cd("C:/Users/wbras/OneDrive/Documents/Desktop/UA/4th_Year_Paper/4th_Year_Paper_Data/HMS/2021-Onward/Dynamic_Model/Data")
end

b_str = lpad(b, 3, '0')


#############################
# 2. Output Paths
#############################

# Set log file path (bootstrap draw specific; a frontier-only rerun keeps its
# own log so the full run's log is not overwritten)
log_path = joinpath(output_dir, get(ENV, "FRONTIER_ONLY", "0") == "1" ? "CF_Bootstrap_Frontier_b$(b_str)_Log.txt" : "CF_Bootstrap_b$(b_str)_Log.txt")

# Open log file for writing (log_io is defined as a global in 01_Functions_Mixture.jl)
log_io = open(log_path, "w")

# Start overall timer
t_start = time()

# Print and log counterfactual simulation start time
log_msg("Bootstrap counterfactual simulation started at $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
log_msg("Bootstrap draw b = $b (b_str = $b_str)")
log_msg("β = $BETA_VAL (tag: $beta_tag)")
log_msg("PSI_3 (grid seed) = $PSI_3; ψ_3 is parameter 28 of the estimates")


#############################
# 3. Initialize Fixed Parameters
#############################

# Load fixed parameters:
#   ψ_2 = fast addiction decay rate (always fixed at 0.90)
#   ψ_1 = slow addiction decay rate (always fixed at 0.10)
#   β = present bias (not used directly here; BETA_VAL from ENV overrides)
#   δ = monthly discount factor (fixed at 0.99)
ψ_2, ψ_1, β, δ = get_fixed_parameters();


#############################
# 4. State Spaces and Choices
#############################

# Start timer for data prep
t_setup = time();

# Get fast addiction grid
N_A_f, A_f = get_addiction_space(ψ_2; N_A=5);

# Get slow addiction grid
N_A_s, A_s = get_addiction_space(ψ_1; N_A=10);

# Get number of observations (N_HHT), number of alternatives (N_J), and choice matrix J
_, N_J, J = get_product_choices();

# Convert choice matrix J to choice vector y where y[i] = chosen alternative index for observation i
y = get_hh_choices(J);

# Get household identifiers
hh_codes = get_hh_codes();

# Pre-compute contiguous household index ranges for mixture posterior computation
hh_ranges = precompute_hh_ranges(hh_codes);

# Get number of product categories excluding the outside option
N_K, _ = get_category_choices();


#############################
# 5. Alternative-Level Vectors
#############################

# Get consumption vectors by alternative (STANDARDIZED by max)
N_cig, N_orig_ecig, N_non_fda_flav_ecig, N_fda_flav_ecig, _, q_cig, q_ecig, q_bundle, q_cig_max, q_ecig_max, q_bundle_max = get_consumption(N_J);

# Raw (de-standardized) quantities for recording simulated purchases
q_cig_raw  = q_cig  .* q_cig_max;
q_ecig_raw = q_ecig .* q_ecig_max;

# Get nicotine vector by alternative (STANDARDIZED by max)
# n_max is the raw max value for rescaling estimates
n, n_max = get_nicotine(N_J);

# Get category index by alternative
cat_idx = get_category_index(N_J, N_cig, N_orig_ecig, N_non_fda_flav_ecig, N_fda_flav_ecig);

# Get flavored indicator by alternative: is_flavored[j] in {true, false} (any flavored: non-FDA or FDA)
is_flavored = get_flavored_indicator(cat_idx);

# Get FDA flavored indicator by alternative: is_fda_flavored[j] in {0, 1}
is_fda_flavored = get_fda_flavored_indicator(cat_idx);

# Flavor lock-in indicator: orig ecig (cat 2) and orig bundle (cat 5)
is_nonflavored_ecig = [cat_idx[j] in (2, 5) for j in 1:N_J]

# Get indicator for alternatives containing cigarettes (cat 1 = cig, cat 5-7 = bundles with cig)
has_cig = [(cat_idx[j] == 1 || cat_idx[j] >= 5) for j in 1:N_J]

# Get indicator for alternatives containing e-cigarettes (cat 2-7 = any ecig or bundle)
has_ecig = [cat_idx[j] >= 2 for j in 1:N_J]

# Outside option indicator: cat 0 = outside option (j = 1)
is_outside = [cat_idx[j] == 0 for j in 1:N_J]

# Raw e-cig quantity split: original vs. flavored
q_orig_ecig_raw = [is_nonflavored_ecig[j] ? q_ecig_raw[j] : 0.0 for j in 1:N_J]
q_flav_ecig_raw = [is_flavored[j]         ? q_ecig_raw[j] : 0.0 for j in 1:N_J]


#############################
# 6. Demographics
#############################

# TYA states: load binary data (0 = no TYA, 1 = TYA present) and shift to 1-indexed
tya_state = [s + 1 for s in get_tya_states()];

# Household-level TYA share (fraction of months with TYA present) for mixture weights
tya_share_hh = get_tya_share();


#############################
# 7. Price Space
#############################

# Get pricing grid: N_P points per category, P is N_P × 2 (cig, ecig)
N_P, P = get_pricing_spaces();

# Get all price combinations
N_Pcomb, Pcomb = get_pricing_spaces_combination(N_K, N_P, P);

# Get price ratios for quantity discount adjustment (price per unit varies by bin size)
ratio_cig, ratio_ecig = get_price_ratios(N_J, N_cig, N_orig_ecig, N_non_fda_flav_ecig, N_fda_flav_ecig, q_cig, q_ecig);

# Get Halton draw price transitions: T[m, r, k] where m = price state, r = draw, k = category
T = get_transitions(N_K);

# Pre-compute bilinear interpolation brackets and weights for price transitions
# Returns 6 matrices (M x R): lo/hi grid indices and weights for each category
p_cig_lo, p_cig_hi, p_cig_w, p_ecig_lo, p_ecig_hi, p_ecig_w = precompute_price_transitions(N_P, P, T);


#############################
# 8. Household Price Trajectories
#############################

# Map observed household prices to continuous values for interpolation
# p_continuous is N x 2 (cig price, ecig price): actual per-unit prices, not grid indices
_, p_continuous = map_prices_to_grid(N_P, P, Pcomb, N_J);

# Log data setup completion time and sample size
setup_elapsed = time() - t_setup;
log_msg("Data loading complete in $(round(setup_elapsed, digits=1))s")
log_msg("Observations: $(length(y)), Alternatives: $N_J, Fast addiction states: $N_A_f, Slow addiction states: $N_A_s, Price states: $N_Pcomb")


#############################
# 9. AR(1) Price Parameters
#############################

# Load AR(1) coefficients
AR_Phi = CSV.read("../AR_Parameters/AR_Parameters_Phi.csv", DataFrame);
φ_0 = [AR_Phi[1, :intercept], AR_Phi[2, :intercept]];
φ_1 = [AR_Phi[1, :ar1], AR_Phi[2, :ar1]];

# Load AR(1) shock covariance matrix Sigma
AR_Sigma = CSV.read("../AR_Parameters/AR_Parameters_Sigma.csv", DataFrame);
Σ = [AR_Sigma[1, :cig] AR_Sigma[1, :ecig];
     AR_Sigma[2, :cig] AR_Sigma[2, :ecig]];

# Cholesky decomposition: L such that LL' = Sigma
L_chol = cholesky(Σ).L;

# Print and log AR(1) parameter loading confirmation
log_msg("AR(1) parameters loaded")


#############################
# 10. Parameter Loading
#     (Section 13, Bootstrap)
#############################

# Structural parameters are loaded and sampled in section 13 below,
# after household state computation. θ_b is drawn from N(θ̂, V̂Cov).


#############################
# 11. Compute Household States
#############################

# Reconstructs each household's addiction stocks and prices at the END of the
# estimation sample; these terminal states seed the forward simulation. The
# panel is left-truncated, so the initial stock at each household's first
# observation is found via fixed-point iteration rather than assumed known.
# The flavored habit stock depends on PSI_3 (from θ_b), so it is built after
# the bootstrap parameter draw in section 13.

log_msg("\n===================================")
log_msg("Computing household addiction states...")
log_msg("===================================")

t_states = time();

# --- Fast addiction stock ---
af0, max_fp_iters_f = get_initial_addiction_stock(ψ_2, A_f, n, y, hh_codes);
log_msg("Initial fast addiction stocks: max fixed-point iterations = $max_fp_iters_f")
af_continuous = simulate_addiction_trajectories(N_A_f, ψ_2, A_f, n, y, hh_codes, af0);

# --- Slow addiction  stock ---
as0, max_fp_iters_s = get_initial_addiction_stock(ψ_1, A_s, n, y, hh_codes);
log_msg("Initial slow addiction stocks: max fixed-point iterations = $max_fp_iters_s")
as_continuous = simulate_addiction_trajectories(N_A_s, ψ_1, A_s, n, y, hh_codes, as0);

# Flavored habit stock depends on PSI_3 (from θ_b); built after the parameter draw in section 13
n_flav = Float64.(is_flavored)

# Terminal state = one evolution step past each household's last observed choice
unique_hh = unique(hh_codes);
N_HH = length(unique_hh);
N_obs = length(y);

# Map household code to its observation row indices (last obs = terminal state)
hh_obs = Dict{eltype(hh_codes), Vector{Int}}()
for i in 1:N_obs
    hh = hh_codes[i]
    if !haskey(hh_obs, hh)
        hh_obs[hh] = Int[]
    end
    push!(hh_obs[hh], i)
end

# hh_aflav0 (flavored habit) is θ_b-dependent and allocated per draw in section 13.
hh_tya = Vector{Int}(undef, N_HH)          # TYA status at last observation
hh_af0 = Vector{Float64}(undef, N_HH)      # fast addiction entering period T+1
hh_as0 = Vector{Float64}(undef, N_HH)      # slow addiction entering period T+1
hh_p0  = Matrix{Float64}(undef, N_HH, 2)   # [cig price, ecig price] at last obs

for (h_idx, hh) in enumerate(unique_hh)

    obs_indices = hh_obs[hh]
    last_obs = obs_indices[end]

    hh_tya[h_idx] = tya_state[last_obs]

    hh_p0[h_idx, 1] = p_continuous[last_obs, 1]
    hh_p0[h_idx, 2] = p_continuous[last_obs, 2]

    hh_af0[h_idx] = addiction_evolution(ψ_2, af_continuous[last_obs], n[y[last_obs]])
    hh_af0[h_idx] = clamp(hh_af0[h_idx], A_f[1], A_f[end])

    hh_as0[h_idx] = addiction_evolution(ψ_1, as_continuous[last_obs], n[y[last_obs]])
    hh_as0[h_idx] = clamp(hh_as0[h_idx], A_s[1], A_s[end])

end

states_elapsed = time() - t_states;
log_msg("Household states computed in $(round(states_elapsed, digits=1))s")
log_msg("Unique households: $N_HH")
log_msg("Mean terminal fast addiction: $(round(mean(hh_af0), digits=4))")
log_msg("Mean terminal slow addiction: $(round(mean(hh_as0), digits=4))")
log_msg("Mean terminal flavored habit: logged per θ_b in section 13")


#############################
# 12. Pre-compute Addiction
#     Transitions
#############################

# Fast stock transitions (always pre-computed, ψ_2 is never estimated)
af_lower, af_upper, af_weight = precompute_addiction_transitions(N_J, N_A_f, ψ_2, A_f, n);

# Slow stock transitions (ψ_1 = 0.10 always fixed)
as_lower, as_upper, as_weight = precompute_addiction_transitions(N_J, N_A_s, ψ_1, A_s, n);



#############################
# 13. Bootstrap Parameter
#     Sampling
#############################

log_msg("\n===================================")
log_msg("Bootstrap parameter sampling (b = $b, β = $BETA_VAL)")
log_msg("===================================")

# Construct estimates and VCov paths (β-specific)
_beta_tag_full = "Beta_$(beta_tag)"
_run_tag = "Dynamic_Model_Mixture_$(psi_tag)_$(_beta_tag_full)"

if HPC
    _est_base = "/home/u2/wbrasic/4th_Year_Paper/Dynamic_Model/02_Second_Stage_Estimation_Mixture"
else
    _est_base = "C:/Users/wbras/OneDrive/Documents/Desktop/UA/4th_Year_Paper/4th_Year_Paper_Data/HMS/2021-Onward/Dynamic_Model"
end

_est_dir       = joinpath(_est_base, "$(_run_tag)_Results", "$(_run_tag)_Estimates")
estimates_path = joinpath(_est_dir, "$(_run_tag)_Estimates.csv")
vcov_path      = joinpath(_est_dir, "$(_run_tag)_VCov.csv")

# Load θ̂ (28 columns: 14 common, 9 ξ, 4 π, ψ_3; an NLL column is dropped if present)
_df_est = CSV.read(estimates_path, DataFrame)
if "NLL" in names(_df_est)
    select!(_df_est, Not(:NLL))
end
param_names = names(_df_est)
D_params    = length(param_names)
θ_hat       = Float64.(collect(_df_est[1, :]))
D_params == 28 || error("Expected the 28-parameter estimates layout; got $(D_params) parameters from $estimates_path")
log_msg("Loaded θ̂: $(D_params) parameters from $estimates_path")

if b == 0

    # Task 0 is the point-estimate run: θ_b = θ̂ with no draw. The aggregation
    # script appends it as the _point columns next to the bootstrap bands.
    θ_b = copy(θ_hat)
    n_redraws = 0
    log_msg("Task 0: point estimates, no bootstrap draw")

else

    # Load V̂Cov = H⁻¹ from the standard-error script
    _df_vcov = CSV.read(vcov_path, DataFrame)
    VCov     = Matrix{Float64}(_df_vcov)
    size(VCov) == (D_params, D_params) || error("VCov is $(size(VCov)); expected $(D_params)×$(D_params)")
    log_msg("Loaded V̂Cov: $(size(VCov,1))×$(size(VCov,2)) from $vcov_path")

    # Cholesky factor for sampling: θ_b = θ̂ + L_vcov * z, z ~ N(0,I).
    # Clip negative eigenvalues to 1e-10 before Cholesky.
    _eig     = eigen(Symmetric(VCov))
    _vals_pd = max.(_eig.values, 1e-10)
    VCov_pd  = Symmetric(_eig.vectors * Diagonal(_vals_pd) * _eig.vectors')
    L_vcov   = cholesky(VCov_pd).L

    # Seed RNG with b before sampling θ_b (CRN seed applied later in section 14)
    Random.seed!(b)
    log_msg("RNG seeded with b = $b for parameter draw")

    # Re-draw until the draw satisfies the estimation bounds (α_C, α_E ≥ 0;
    # α_E2, γ, ω ≤ 0; ψ_3 ∈ (0.01, 0.99)), using the same check as estimation
    MAX_REDRAWS = 10_000
    θ_b = copy(θ_hat)
    n_redraws = 0
    while true
        θ_b .= θ_hat .+ L_vcov * randn(D_params)
        _ok, _ = check_parameter_bounds(θ_b, param_names)
        _ok && break
        global n_redraws += 1
        n_redraws > MAX_REDRAWS && error("Bootstrap draw b = $b: no admissible draw after $MAX_REDRAWS attempts")
    end
    log_msg("")
    log_msg(@sprintf("θ_b drawn: %d re-draw(s) needed", n_redraws))

end

log_msg("")
log_msg(@sprintf("  %-14s   %-20s   %-20s   %s", "Parameter", "θ̂ (MLE)", "θ_b (draw)", "Δ"))
log_msg("  " * "-"^74)
for k in 1:D_params
    delta = θ_b[k] - θ_hat[k]
    log_msg(@sprintf("  %-14s = %20.10f   %20.10f   %+.6f",
        param_names[k], θ_hat[k], θ_b[k], delta))
end
log_msg("")

# Unpack θ_b (28-parameter layout, as in 02_Second_Stage_Estimation_Mixture)
common_b    = θ_b[1:14]    # α_C, α_E, α_E2, α_CE, λ_1-λ_4, γ_1-γ_4, ω_C, ω_E
ξ_1_b       = θ_b[15:17]   # ξ_C_1, ξ_E_1, ξ_CE_1
ξ_2_b       = θ_b[18:20]   # ξ_C_2, ξ_E_2, ξ_CE_2
ξ_3_b       = θ_b[21:23]   # ξ_C_3, ξ_E_3, ξ_CE_3
π_0_2       = θ_b[24]
π_TYA_2     = θ_b[25]
π_0_3       = θ_b[26]
π_TYA_3     = θ_b[27]
omega_E_est = common_b[14]

# ψ_3 is parameter 28; the ENV value only seeded the grid
PSI_3 = θ_b[28]
log_msg(@sprintf("ψ_3 = %.6f (estimated, θ_b[28])", PSI_3))

# Marginal utility of income in original units: utility carries
# ω_E · P · ratio · q_std = (ω_E / q_ecig_max) · (dollars spent), so dividing a
# utility difference by omega_E_orig converts it to dollars
omega_E_orig = abs(omega_E_est) / q_ecig_max
log_msg(@sprintf("ω_E = %.10f (standardized); %.10f utils per dollar", omega_E_est, omega_E_orig))
log_msg("")

θ_struct_1 = vcat(common_b, ξ_1_b)
θ_struct_2 = vcat(common_b, ξ_2_b)
θ_struct_3 = vcat(common_b, ξ_3_b)

log_msg(@sprintf("Mixing weights: π_0_2=%.4f  π_TYA_2=%.4f  π_0_3=%.4f  π_TYA_3=%.4f",
    π_0_2, π_TYA_2, π_0_3, π_TYA_3))
for tya_val in (0.0, 1.0)
    e1 = 1.0; e2 = exp(π_0_2 + π_TYA_2 * tya_val); e3 = exp(π_0_3 + π_TYA_3 * tya_val)
    denom = e1 + e2 + e3
    log_msg(@sprintf("  tya=%.0f: P(k=1)=%.4f  P(k=2)=%.4f  P(k=3)=%.4f",
        tya_val, e1/denom, e2/denom, e3/denom))
end
log_msg("")

# Flavored habit objects (depend on PSI_3 from θ_b)
N_A_flav, A_flav = get_addiction_space(PSI_3; N_A=10)
aflav_lower, aflav_upper, aflav_weight = precompute_addiction_transitions(N_J, N_A_flav, PSI_3, A_flav, n_flav)
_aflav0, _ = get_initial_addiction_stock(PSI_3, A_flav, n_flav, y, hh_codes)
aflav_continuous = simulate_addiction_trajectories(N_A_flav, PSI_3, A_flav, n_flav, y, hh_codes, _aflav0)

hh_aflav0 = Vector{Float64}(undef, N_HH)
for (h_idx, hh) in enumerate(unique_hh)
    last_obs = hh_obs[hh][end]
    hh_aflav0[h_idx] = addiction_evolution(PSI_3, aflav_continuous[last_obs], n_flav[y[last_obs]])
    hh_aflav0[h_idx] = clamp(hh_aflav0[h_idx], A_flav[1], A_flav[end])
end
log_msg(@sprintf("Mean terminal flavored habit: %.4f", mean(hh_aflav0)))

# Flow utilities from θ_b
log_msg("Computing flow utilities from θ_b...")
U_1 = get_flow_utility(
    θ_struct_1, N_J, N_A_f, N_A_s, N_A_flav, N_Pcomb, A_f, A_s, A_flav,
    q_cig, q_ecig, q_bundle, is_flavored, is_fda_flavored, is_nonflavored_ecig,
    is_outside, cat_idx, Pcomb, has_cig, has_ecig;
    ratio_cig = ratio_cig, ratio_ecig = ratio_ecig
)
U_2 = get_flow_utility(
    θ_struct_2, N_J, N_A_f, N_A_s, N_A_flav, N_Pcomb, A_f, A_s, A_flav,
    q_cig, q_ecig, q_bundle, is_flavored, is_fda_flavored, is_nonflavored_ecig,
    is_outside, cat_idx, Pcomb, has_cig, has_ecig;
    ratio_cig = ratio_cig, ratio_ecig = ratio_ecig
)
U_3 = get_flow_utility(
    θ_struct_3, N_J, N_A_f, N_A_s, N_A_flav, N_Pcomb, A_f, A_s, A_flav,
    q_cig, q_ecig, q_bundle, is_flavored, is_fda_flavored, is_nonflavored_ecig,
    is_outside, cat_idx, Pcomb, has_cig, has_ecig;
    ratio_cig = ratio_cig, ratio_ecig = ratio_ecig
)
log_msg("Flow utilities computed from θ_b.")

log_msg("\nAll bootstrap parameter objects ready.")


#############################
# 14. Pre-draw CRN Variates
#     (Once, Before Any Loop)
#############################

# All variates are drawn ONCE, before any policy loop, and reused across every
# ban/tax scenario, so SQ-vs-policy differences reflect the policy itself, not
# simulation noise. The CRN seed does not depend on b: every bootstrap draw
# shares the same simulation noise, so the bands reflect parameter uncertainty.

# Simulation settings
T_sim    = 36     # months to simulate forward (3 years)
N_draws  = 100    # Monte Carlo draws per household
crn_seed = 12345  # fixed seed for reproducibility across all bootstrap draws

log_msg("\nSimulation settings: T_sim = $T_sim, N_draws = $N_draws, N_HH = $N_HH, CRN seed = $crn_seed")

Random.seed!(crn_seed)

# crn_choice[h, t, d]: U(0,1) for inverse-CDF choice sampling
crn_choice = Array{Float64}(undef, N_HH, T_sim, N_draws)
for d in 1:N_draws
    for t in 1:T_sim
        for h in 1:N_HH
            crn_choice[h, t, d] = rand()
        end
    end
end

# crn_price[h, t, d, k]: N(0,1) shock for price dim k (1=cig, 2=e-cig); the
# AR(1) process converts these via the Cholesky factor L_chol
crn_price = Array{Float64}(undef, N_HH, T_sim, N_draws, 2)
for d in 1:N_draws
    for t in 1:T_sim
        for h in 1:N_HH
            crn_price[h, t, d, 1] = randn()
            crn_price[h, t, d, 2] = randn()
        end
    end
end

# crn_type[h, d]: U(0,1) for sampling a latent type from the K=3 posterior
crn_type = Array{Float64}(undef, N_HH, N_draws)
for d in 1:N_draws
    for h in 1:N_HH
        crn_type[h, d] = rand()
    end
end

log_msg("CRN variates pre-drawn: choice $(size(crn_choice)), price $(size(crn_price)), type $(size(crn_type))")


#############################
# 15. Policy Settings and
#     Functions
#############################

cat_labels = ["Outside", "Cig", "Orig Ecig", "Non-FDA Flav Ecig", "FDA Flav Ecig", "Orig Bundle", "Non-FDA Flav Bundle", "FDA Flav Bundle"]

# Ban types: label and the function that zeroes out the banned alternatives
BAN_TYPES = [
    ("Ban_Comprehensive", apply_flavor_ban!),
    ("Ban_Non_FDA", apply_non_fda_ban!)
]

# Per-mL flavor tax rates with the full set of outputs (shares, addiction,
# welfare, initiation, by subgroup). Comma-separated ENV override.
TAX_GRID = sort(unique(parse.(Float64, split(get(ENV, "TAX_GRID", "0.50,2.78"), ","))))

# Frontier grid: tax rates evaluated only for the e-cigarette-persistent
# subset (modal type 2), to trace the tax-versus-ban frontier. Default: whole
# dollars from $1 to $15 per mL. Rates already in TAX_GRID are taken from the
# full run rather than solved twice.
FRONTIER_TAX_GRID = sort(unique(parse.(Float64,
    split(get(ENV, "FRONTIER_TAX_GRID", join(string.(1:15), ",")), ","))))

# FRONTIER_ONLY=1 redoes only the frontier (sections 19 and 20): the full
# policies of section 18 are skipped, and the comprehensive ban and every rate
# in TAX_GRID are solved and simulated on the type-2 subset like the frontier
# rates, so the Frontier_* files are rewritten while the Ban_* and Flavor_Tax
# folders from the full run are left untouched. Use it to change the frontier
# grid without re-running the full policies.
FRONTIER_ONLY       = get(ENV, "FRONTIER_ONLY", "0") == "1"
ALL_TAXES           = sort(union(TAX_GRID, FRONTIER_TAX_GRID))
FRONTIER_ONLY_TAXES = FRONTIER_ONLY ? ALL_TAXES : setdiff(FRONTIER_TAX_GRID, TAX_GRID)

# Initiation (extensive-margin) horizons: a household is a baseline non-user at
# horizon k if none of its last k observed months contains a purchase of the
# outcome's alternatives or of any flavored alternative. Built from the observed
# choices, so the base is the same in every bootstrap draw.
INITIATION_HORIZONS = [(3, "_thresh3mo"), (6, "_thresh6mo"), (12, "_thresh12mo")]

# Initiation outcomes: file-name infix and the alternatives that count as a
# first purchase. Each outcome has its own base: flavored non-users for the
# flavored outcome, and flavored non-users who also were not buying the
# outcome's product for the cigarette and unflavored e-cigarette outcomes.
INITIATION_OUTCOMES = [
    ("",          nothing),              # flavored e-cigarettes (cat 3, 4, 6, 7)
    ("_Cig",      has_cig),              # cigarettes, alone or in a bundle (cat 1, 5, 6, 7)
    ("_OrigEcig", is_nonflavored_ecig)   # unflavored e-cigarettes, alone or in a bundle (cat 2, 5)
]

"""
Baseline non-users for one initiation outcome and horizon, from the observed
purchase history. Household h is eligible if none of its last `k` observed
months (all of them, for a household observed fewer than k months) contains a
flavored purchase or a purchase of an alternative flagged in `alt_mask`
(`nothing` means the flavored alternatives only). Returns a length-N_HH Bool
vector in the `unique_hh` ordering used by the simulation arrays.
"""
function build_initiation_base(k::Integer, alt_mask)
    is_target = alt_mask === nothing ? is_flavored : (is_flavored .| alt_mask)
    eligible = Vector{Bool}(undef, N_HH)
    for (h_idx, hh) in enumerate(unique_hh)
        rows = hh_obs[hh]
        recent = rows[max(1, length(rows) - k + 1):end]
        eligible[h_idx] = !any(is_target[y[i]] for i in recent)
    end
    return eligible
end

for (infix, alt_mask) in INITIATION_OUTCOMES, (k, suffix) in INITIATION_HORIZONS
    log_msg(@sprintf("Initiation base%s, %2d months: %d of %d households", infix, k, count(build_initiation_base(k, alt_mask)), N_HH))
end

tau_tag(tau) = replace(@sprintf("%.2f", tau), "." => "p")

log_msg("\nBan types: $(first.(BAN_TYPES))")
log_msg("Tax rates (full outputs): $TAX_GRID")
log_msg("Frontier-only tax rates (type 2 subset): $FRONTIER_ONLY_TAXES")
FRONTIER_ONLY && log_msg("FRONTIER_ONLY mode: full policies skipped; only the Frontier_* files are rewritten")

"""
Solve the three type-specific value functions in parallel for one set of flow
utilities and return `(V_now, V_decision)` as 3-tuples indexed by type.
`V_init` is an optional 3-tuple of ex-ante value functions used as the
starting point (a nearby policy's solution cuts the iteration count).
"""
function solve_three_types(U_a, U_b, U_c, label::AbstractString; V_init = (nothing, nothing, nothing))

    t0 = time()

    tasks = [Threads.@spawn solve_vfi_sophisticated(
        N_J, N_A_f, N_A_s, N_A_flav, N_P, N_Pcomb, BETA_VAL, δ, U_k,
        af_lower, af_upper, af_weight,
        as_lower, as_upper, as_weight,
        aflav_lower, aflav_upper, aflav_weight,
        p_cig_lo, p_cig_hi, p_cig_w,
        p_ecig_lo, p_ecig_hi, p_ecig_w;
        V_init = V0,
        verbose = false
    ) for (U_k, V0) in zip((U_a, U_b, U_c), V_init)]

    out = fetch.(tasks)
    iters = join(["type$(k)=$(out[k][3]) ($(out[k][4]))" for k in 1:3], ", ")
    log_msg(@sprintf("VFI %s: %s, %.1fs", label, iters, time() - t0))

    V_now      = (out[1][1], out[2][1], out[3][1])
    V_decision = (out[1][2], out[2][2], out[3][2])
    return V_now, V_decision
end

"""
Mixture-weighted choice probabilities and welfare at every observed state for
one scenario's decision values, using the household posterior type weights.
Returns `(probs, welfare)` with probs N_obs × N_J and welfare length N_obs.
"""
function pointwise_mixture(V_decision::NTuple{3, Array{Float64, 6}})

    probs   = zeros(Float64, N_obs, N_J)
    welfare = zeros(Float64, N_obs)

    for k in 1:3
        probs_k, welfare_k = compute_pointwise_outcomes(
            V_decision[k], tya_state, af_continuous, as_continuous, aflav_continuous, p_continuous,
            N_J, N_P, A_f, A_s, A_flav, P
        )
        probs   .+= obs_posterior[:, k] .* probs_k
        welfare .+= obs_posterior[:, k] .* welfare_k
    end

    return probs, welfare
end

"""
Forward-simulate one scenario for the households in `idx_hh` with the shared
common random numbers. Thin wrapper over `simulate_scenario` so every call
site passes the same data objects.
"""
function simulate_policy(idx_hh, V_decision, V_now)
    return simulate_scenario(
        idx_hh, V_decision, V_now,
        hh_posterior_type1, hh_posterior_type2, hh_tya,
        hh_af0, hh_as0, hh_aflav0, hh_p0,
        crn_choice, crn_price, crn_type,
        T_sim, N_draws, N_J, N_P, A_f, A_s, A_flav, P,
        n, n_flav, q_cig_raw, q_orig_ecig_raw, q_flav_ecig_raw,
        ψ_1, ψ_2, PSI_3, φ_0, φ_1, L_chol
    )
end

"""
Period-by-period aggregates of one full-sample simulation, overall and by
subgroup. Returns a NamedTuple of DataFrames keyed by the labels used in the
output files.
"""
function aggregate_full(sim)
    agg_all     = aggregate_simulation(sim.choices, sim.addiction, sim.aflav, sim.welfare, sim.q_cig, sim.q_orig_ecig, sim.q_flav_ecig, cat_idx, N_J, T_sim)
    agg_tya     = aggregate_simulation_by_tya(sim.choices, sim.addiction, sim.aflav, sim.welfare, sim.q_cig, sim.q_orig_ecig, sim.q_flav_ecig, cat_idx, N_J, T_sim, hh_tya, N_draws)
    agg_non_tya = aggregate_simulation_by_non_tya(sim.choices, sim.addiction, sim.aflav, sim.welfare, sim.q_cig, sim.q_orig_ecig, sim.q_flav_ecig, cat_idx, N_J, T_sim, hh_tya, N_draws)
    agg_type    = aggregate_simulation_by_type_k3(sim.choices, sim.addiction, sim.aflav, sim.welfare, sim.q_cig, sim.q_orig_ecig, sim.q_flav_ecig, cat_idx, N_J, T_sim, hh_posterior_type1, hh_posterior_type2, hh_posterior_type3, N_draws)
    agg_tya_type     = aggregate_simulation_by_tya_type_k3(sim.choices, sim.addiction, sim.aflav, sim.welfare, sim.q_cig, sim.q_orig_ecig, sim.q_flav_ecig, cat_idx, N_J, T_sim, hh_tya, hh_posterior_type1, hh_posterior_type2, hh_posterior_type3, N_draws)
    agg_non_tya_type = aggregate_simulation_by_non_tya_type_k3(sim.choices, sim.addiction, sim.aflav, sim.welfare, sim.q_cig, sim.q_orig_ecig, sim.q_flav_ecig, cat_idx, N_J, T_sim, hh_tya, hh_posterior_type1, hh_posterior_type2, hh_posterior_type3, N_draws)
    return (
        overall      = agg_all,
        tya          = [("tya", agg_tya), ("non_tya", agg_non_tya)],
        type         = [("type1", agg_type[1]), ("type2", agg_type[2]), ("type3", agg_type[3])],
        tya_type     = [("tya_type1", agg_tya_type[1]), ("tya_type2", agg_tya_type[2]), ("tya_type3", agg_tya_type[3])],
        non_tya_type = [("non_tya_type1", agg_non_tya_type[1]), ("non_tya_type2", agg_non_tya_type[2]), ("non_tya_type3", agg_non_tya_type[3])],
    )
end

# Columns written for each scenario in the Simulation_* files, in order
SIM_COLS = [:share_outside, :share_cig, :share_orig_ecig, :share_non_fda_flav_ecig, :share_fda_flav_ecig,
            :share_orig_bundle, :share_non_fda_flav_bundle, :share_fda_flav_bundle,
            :mean_addiction, :mean_aflav, :mean_welfare, :mean_q_cig, :mean_q_orig_ecig, :mean_q_flav_ecig]
SIM_NAMES = ["outside", "cig", "orig_ecig", "non_fda_flav_ecig", "fda_flav_ecig",
             "orig_bundle", "non_fda_flav_bundle", "fda_flav_bundle",
             "addiction", "aflav", "welfare", "q_cig", "q_orig_ecig", "q_flav_ecig"]

"""
Write one Simulation_* file: status quo columns (sq_*) next to the policy
columns (<cf_prefix>_*), one row per period, with an optional leading group
column. `groups` is a vector of (label, agg_sq, agg_cf) triples.
"""
function write_simulation_csv(path, group_col, groups, cf_prefix)
    open(path, "w") do io
        header = vcat(group_col === nothing ? String[] : [group_col], ["period"],
                      ["sq_$(c)" for c in SIM_NAMES], ["$(cf_prefix)_$(c)" for c in SIM_NAMES])
        println(io, join(header, ","))
        for (label, agg_sq_g, agg_cf_g) in groups
            for t in 1:T_sim
                row = vcat(group_col === nothing ? String[] : [label], [@sprintf("%d", t)],
                           [@sprintf("%.10f", agg_sq_g[t, c]) for c in SIM_COLS],
                           [@sprintf("%.10f", agg_cf_g[t, c]) for c in SIM_COLS])
                println(io, join(row, ","))
            end
        end
    end
    log_msg("Saved: $path")
end

"""
Write one Extensive_Margin_* file: one row per (group, period) with the
baseline non-user count, cumulative initiation under the status quo and the
policy, and the prevention rate. `groups` is a vector of (label, DataFrame).
"""
function write_extensive_csv(path, group_col, groups)
    open(path, "w") do io
        println(io, join([group_col, "period", "n_non_users", "sq_ever_initiated", "cf_ever_initiated", "prevention_rate"], ","))
        for (label, df_g) in groups
            for t in 1:T_sim
                row = [label, @sprintf("%d", t), @sprintf("%d", df_g.n_non_users[t]),
                       @sprintf("%.10f", df_g.sq_ever_initiated[t]), @sprintf("%.10f", df_g.cf_ever_initiated[t]),
                       @sprintf("%.10f", df_g.prevention_rate[t])]
                println(io, join(row, ","))
            end
        end
    end
    log_msg("Saved: $path")
end

"""
Log the status-quo-versus-policy category share table and mean welfare from
observation-level mixture probabilities.
"""
function log_pointwise_table(cf_label, probs_cf, welfare_cf)
    log_msg("\nPointwise summary (means across all observations):")
    log_msg(@sprintf("  %-22s  %12s  %12s  %12s", "Category", "SQ Share", "$cf_label Share", "Difference"))
    log_msg("  " * repeat("-", 62))
    for (c, label) in enumerate(cat_labels)
        alt_indices = findall(j -> cat_idx[j] == c - 1, 1:N_J)
        sq_share = mean(sum(probs_sq[:, alt_indices], dims=2))
        cf_share = mean(sum(probs_cf[:, alt_indices], dims=2))
        log_msg(@sprintf("  %-22s  %12.6f  %12.6f  %12.6f", label, sq_share, cf_share, cf_share - sq_share))
    end
    log_msg(@sprintf("\n  Mean welfare SQ:   %.6f", mean(welfare_sq)))
    log_msg(@sprintf("  Mean welfare %s: %.6f", cf_label, mean(welfare_cf)))
    log_msg(@sprintf("  Mean welfare loss: %.6f", mean(welfare_cf .- welfare_sq)))
end


#############################
# 16. Status Quo: VFI,
#     Posteriors, Simulation
#############################

# The status quo does not depend on the policy, so its value functions,
# posterior type weights, and forward simulation are computed once and reused
# against every ban and tax.

log_msg("\n\n###################################")
log_msg("Status quo (β = $BETA_VAL, b = $b)")
log_msg("###################################")

V_now_sq, V_dec_sq = solve_three_types(U_1, U_2, U_3, "status quo")

# Posterior type weights from the status quo likelihood (K=3 softmax prior on
# the household TYA share, updated with each type's choice probabilities)
log_msg("Computing posterior type weights...")
t_post = time()

probs_1_sq, welfare_1_sq = compute_pointwise_outcomes(V_dec_sq[1], tya_state, af_continuous, as_continuous, aflav_continuous, p_continuous, N_J, N_P, A_f, A_s, A_flav, P)
probs_2_sq, welfare_2_sq = compute_pointwise_outcomes(V_dec_sq[2], tya_state, af_continuous, as_continuous, aflav_continuous, p_continuous, N_J, N_P, A_f, A_s, A_flav, P)
probs_3_sq, welfare_3_sq = compute_pointwise_outcomes(V_dec_sq[3], tya_state, af_continuous, as_continuous, aflav_continuous, p_continuous, N_J, N_P, A_f, A_s, A_flav, P)

hh_posterior_type1 = Vector{Float64}(undef, N_HH)
hh_posterior_type2 = Vector{Float64}(undef, N_HH)
hh_posterior_type3 = Vector{Float64}(undef, N_HH)

for h in 1:N_HH
    start_idx, stop_idx = hh_ranges[h]

    η_2_h = π_0_2 + π_TYA_2 * tya_share_hh[h]
    η_3_h = π_0_3 + π_TYA_3 * tya_share_hh[h]
    log_sum_exp_h = log(1.0 + exp(η_2_h) + exp(η_3_h))
    log_π_1_h = -log_sum_exp_h
    log_π_2_h = η_2_h - log_sum_exp_h
    log_π_3_h = η_3_h - log_sum_exp_h

    ll_1 = 0.0; ll_2 = 0.0; ll_3 = 0.0
    for i in start_idx:stop_idx
        ll_1 += log(max(probs_1_sq[i, y[i]], 1e-300))
        ll_2 += log(max(probs_2_sq[i, y[i]], 1e-300))
        ll_3 += log(max(probs_3_sq[i, y[i]], 1e-300))
    end

    a_term = log_π_1_h + ll_1
    b_term = log_π_2_h + ll_2
    c_term = log_π_3_h + ll_3
    log_max = max(a_term, b_term, c_term)
    log_denom = log_max + log(exp(a_term - log_max) + exp(b_term - log_max) + exp(c_term - log_max))

    hh_posterior_type1[h] = exp(a_term - log_denom)
    hh_posterior_type2[h] = exp(b_term - log_denom)
    hh_posterior_type3[h] = exp(c_term - log_denom)
end

log_msg(@sprintf("Posterior type weights computed in %.1fs", time() - t_post))
log_msg(@sprintf("  Mean P(type=1) = %.4f, Mean P(type=2) = %.4f, Mean P(type=3) = %.4f",
    mean(hh_posterior_type1), mean(hh_posterior_type2), mean(hh_posterior_type3)))

# Observation-level posterior weights (N_obs × 3) for the pointwise tables
obs_posterior = Matrix{Float64}(undef, N_obs, 3)
for h in 1:N_HH
    start_idx, stop_idx = hh_ranges[h]
    for i in start_idx:stop_idx
        obs_posterior[i, 1] = hh_posterior_type1[h]
        obs_posterior[i, 2] = hh_posterior_type2[h]
        obs_posterior[i, 3] = hh_posterior_type3[h]
    end
end

probs_sq   = obs_posterior[:, 1] .* probs_1_sq   .+ obs_posterior[:, 2] .* probs_2_sq   .+ obs_posterior[:, 3] .* probs_3_sq
welfare_sq = obs_posterior[:, 1] .* welfare_1_sq .+ obs_posterior[:, 2] .* welfare_2_sq .+ obs_posterior[:, 3] .* welfare_3_sq
probs_1_sq = probs_2_sq = probs_3_sq = nothing
welfare_1_sq = welfare_2_sq = welfare_3_sq = nothing

# E-cigarette-persistent households (modal type 2), overall and with a TYA
# present: the frontier is drawn for these two groups. The classification uses
# this draw's posteriors, so the group differs slightly across bootstrap draws;
# the pooled bands treat the classification as part of the estimated object.
idx_type2 = select_type2_households(hh_posterior_type1, hh_posterior_type2, hh_posterior_type3, hh_tya)
log_msg(@sprintf("E-cigarette-persistent households: %d overall, %d with TYA, %d without",
    length(idx_type2.all), length(idx_type2.tya), length(idx_type2.non_tya)))

# Frontier groups as household indices (full-sample arrays) and as positions
# within the type-2 subset (frontier-only simulations)
FRONTIER_GROUPS      = [("type2_all", idx_type2.all), ("type2_tya", idx_type2.tya)]
FRONTIER_GROUPS_SUB  = [("type2_all", collect(1:length(idx_type2.all))), ("type2_tya", findall(h -> hh_tya[h] == 2, idx_type2.all))]

# Forward simulation of the status quo for all households
log_msg("\nForward simulation: status quo (T_sim = $T_sim, N_draws = $N_draws, N_HH = $N_HH)...")
t_sim_sq = time()
sim_sq = simulate_policy(1:N_HH, V_dec_sq, V_now_sq)
log_msg(@sprintf("Status quo simulation complete in %.1fs", time() - t_sim_sq))

V_now_sq = nothing; V_dec_sq = nothing
GC.gc()

# Status quo aggregates, reused by every policy
aggs_sq     = aggregate_full(sim_sq)
frontier_sq = Dict(g => aggregate_subset_paths(sim_sq, rows, cat_idx, N_J, T_sim) for (g, rows) in FRONTIER_GROUPS)

log_msg(@sprintf("Status quo: mean addiction %.6f, mean welfare %.6f (averaged over %d periods)",
    mean(aggs_sq.overall.mean_addiction), mean(aggs_sq.overall.mean_welfare), T_sim))


#############################
# 17. Policy Engine
#############################

# Frontier objects, filled by the full policies (comprehensive ban and the
# TAX_GRID rates) and by the frontier-only rates in section 19
frontier_traj = Dict{Tuple{String, Float64, String}, NamedTuple}()
frontier_summ = Dict{Tuple{String, Float64, String}, NamedTuple}()

"""
Run one policy with the full set of outputs: solve the policy value
functions, simulate every household against the shared status quo paths,
and write the Simulation_* and Extensive_Margin_* files to `subdir`.
`apply_cf!` maps a copy of the base flow utility to the policy's; `cf_prefix`
is "ban" or "tax" in the column names; `tau` is the per-mL rate (Inf for a
ban). The type-2 frontier aggregates are stored when `frontier_key` is given.
"""
function run_full_policy(policy::String, tau::Float64, apply_cf!::Function, subdir::String, cf_prefix::String; frontier_key = nothing)

    cf_label = cf_prefix == "ban" ? "Ban" : "Tax"
    mkpath(subdir)

    log_msg("\n\n###################################")
    log_msg(isfinite(tau) ? @sprintf("%s: \$%.2f per mL (tag: %s)", policy, tau, tau_tag(tau)) : "Policy: $policy")
    log_msg("###################################")

    # Policy flow utilities
    U_1_cf = copy(U_1); apply_cf!(U_1_cf)
    U_2_cf = copy(U_2); apply_cf!(U_2_cf)
    U_3_cf = copy(U_3); apply_cf!(U_3_cf)

    banned_alts = findall(j -> U_1_cf[1, j, 1, 1, 1, 1] == -Inf, 1:N_J)
    isempty(banned_alts) || log_msg("Banned alternatives: j = $(banned_alts)")

    # Value functions
    V_now_cf, V_dec_cf = solve_three_types(U_1_cf, U_2_cf, U_3_cf, policy)
    U_1_cf = nothing; U_2_cf = nothing; U_3_cf = nothing

    # Pointwise outcomes at the observed states (log only)
    probs_cf, welfare_cf = pointwise_mixture(V_dec_cf)
    log_pointwise_table(cf_label, probs_cf, welfare_cf)
    if !isempty(banned_alts)
        log_msg(@sprintf("  Max probability of banned alternatives under the ban: %.3e", maximum(view(probs_cf, :, banned_alts))))
    end
    log_msg(@sprintf("  Max welfare increase (should be <= 0): %.6f", maximum(welfare_cf .- welfare_sq)))
    if isfinite(tau)
        flav_alts = findall(j -> cat_idx[j] in (3, 4, 6, 7), 1:N_J)
        flav_revenue = sum(probs_cf[i, j] * q_ecig[j] * q_ecig_max * tau for i in 1:N_obs, j in flav_alts)
        log_msg(@sprintf("  Mean tax revenue per HH-month at observed states: \$%.4f", flav_revenue / N_obs))
    end
    probs_cf = nothing; welfare_cf = nothing

    # Forward simulation
    log_msg("\nForward simulation: $policy...")
    t_sim_cf = time()
    sim_cf = simulate_policy(1:N_HH, V_dec_cf, V_now_cf)
    log_msg(@sprintf("Simulation complete in %.1fs", time() - t_sim_cf))
    V_now_cf = nothing; V_dec_cf = nothing
    GC.gc()

    # Aggregates and the five Simulation_* files
    aggs_cf = aggregate_full(sim_cf)
    pair(groups_sq, groups_cf) = [(lab, a_sq, a_cf) for ((lab, a_sq), (_, a_cf)) in zip(groups_sq, groups_cf)]
    write_simulation_csv(joinpath(subdir, "Simulation_Overall_b$(b_str).csv"),       nothing, [("", aggs_sq.overall, aggs_cf.overall)], cf_prefix)
    write_simulation_csv(joinpath(subdir, "Simulation_by_TYA_b$(b_str).csv"),        "group", pair(aggs_sq.tya, aggs_cf.tya), cf_prefix)
    write_simulation_csv(joinpath(subdir, "Simulation_by_Type_b$(b_str).csv"),       "type",  pair(aggs_sq.type, aggs_cf.type), cf_prefix)
    write_simulation_csv(joinpath(subdir, "Simulation_by_TYA_Type_b$(b_str).csv"),   "type",  pair(aggs_sq.tya_type, aggs_cf.tya_type), cf_prefix)
    write_simulation_csv(joinpath(subdir, "Simulation_by_NonTYA_Type_b$(b_str).csv"), "type", pair(aggs_sq.non_tya_type, aggs_cf.non_tya_type), cf_prefix)

    # Initiation (extensive margin) among baseline non-users: three outcomes ×
    # three horizons × four groupings
    for (infix, alt_mask) in INITIATION_OUTCOMES, (k, suffix) in INITIATION_HORIZONS

        hh_eligible = build_initiation_base(k, alt_mask)

        ext_tya     = aggregate_extensive_margin_by_tya(sim_sq.choices, sim_cf.choices, hh_eligible, cat_idx, T_sim, hh_tya, N_draws; alt_mask = alt_mask)
        ext_non_tya = aggregate_extensive_margin_by_non_tya(sim_sq.choices, sim_cf.choices, hh_eligible, cat_idx, T_sim, hh_tya, N_draws; alt_mask = alt_mask)
        ext_type    = aggregate_extensive_margin_by_type_k3(sim_sq.choices, sim_cf.choices, hh_eligible, cat_idx, T_sim, hh_posterior_type1, hh_posterior_type2, hh_posterior_type3, N_draws; alt_mask = alt_mask)
        ext_tya_type     = aggregate_extensive_margin_by_tya_type_k3(sim_sq.choices, sim_cf.choices, hh_eligible, cat_idx, T_sim, hh_tya, hh_posterior_type1, hh_posterior_type2, hh_posterior_type3, N_draws; alt_mask = alt_mask)
        ext_non_tya_type = aggregate_extensive_margin_by_non_tya_type_k3(sim_sq.choices, sim_cf.choices, hh_eligible, cat_idx, T_sim, hh_tya, hh_posterior_type1, hh_posterior_type2, hh_posterior_type3, N_draws; alt_mask = alt_mask)

        write_extensive_csv(joinpath(subdir, "Extensive_Margin$(infix)_by_TYA$(suffix)_b$(b_str).csv"),         "group", [("tya", ext_tya), ("non_tya", ext_non_tya)])
        write_extensive_csv(joinpath(subdir, "Extensive_Margin$(infix)_by_Type$(suffix)_b$(b_str).csv"),        "type",  [("type1", ext_type[1]), ("type2", ext_type[2]), ("type3", ext_type[3])])
        write_extensive_csv(joinpath(subdir, "Extensive_Margin$(infix)_by_TYA_Type$(suffix)_b$(b_str).csv"),    "type",  [("tya_type1", ext_tya_type[1]), ("tya_type2", ext_tya_type[2]), ("tya_type3", ext_tya_type[3])])
        write_extensive_csv(joinpath(subdir, "Extensive_Margin$(infix)_by_NonTYA_Type$(suffix)_b$(b_str).csv"), "type",  [("non_tya_type1", ext_non_tya_type[1]), ("non_tya_type2", ext_non_tya_type[2]), ("non_tya_type3", ext_non_tya_type[3])])
    end

    # Frontier aggregates for the type-2 groups, from the same simulation
    if frontier_key !== nothing
        for (g, rows) in FRONTIER_GROUPS
            agg_cf = aggregate_subset_paths(sim_cf, rows, cat_idx, N_J, T_sim)
            frontier_traj[(frontier_key, tau, g)] = agg_cf
            frontier_summ[(frontier_key, tau, g)] = frontier_summary(frontier_sq[g], agg_cf, tau, omega_E_orig, T_sim)
        end
    end

    # Summary
    log_msg("\n--- Forward Simulation Summary (averaged over $T_sim periods) ---")
    log_msg(@sprintf("  %-22s  %12s  %12s  %12s", "Category", "SQ Share", "$cf_label Share", "Difference"))
    log_msg("  " * repeat("-", 62))
    for (c, label) in enumerate(cat_labels)
        col = SIM_COLS[c]
        sq_share = mean(aggs_sq.overall[!, col])
        cf_share = mean(aggs_cf.overall[!, col])
        log_msg(@sprintf("  %-22s  %12.6f  %12.6f  %12.6f", label, sq_share, cf_share, cf_share - sq_share))
    end
    log_msg(@sprintf("\n  Mean addiction (SQ):  %.6f", mean(aggs_sq.overall.mean_addiction)))
    log_msg(@sprintf("  Mean addiction (%s): %.6f", cf_label, mean(aggs_cf.overall.mean_addiction)))
    log_msg(@sprintf("  Addiction change:     %.6f", mean(aggs_cf.overall.mean_addiction) - mean(aggs_sq.overall.mean_addiction)))
    log_msg(@sprintf("\n  Mean welfare (SQ):   %.6f", mean(aggs_sq.overall.mean_welfare)))
    log_msg(@sprintf("  Mean welfare (%s):  %.6f", cf_label, mean(aggs_cf.overall.mean_welfare)))
    log_msg(@sprintf("  Welfare change:      %.6f", mean(aggs_cf.overall.mean_welfare) - mean(aggs_sq.overall.mean_welfare)))

    sim_cf = nothing; aggs_cf = nothing
    GC.gc()
    return nothing
end


#############################
# 18. Bans and Taxes
#     (Full Outputs)
#############################

if FRONTIER_ONLY

    # Frontier-only mode: the comprehensive ban is solved and simulated on the
    # type-2 subset as the frontier's reference point; the TAX_GRID rates are
    # handled by the section 19 loop (FRONTIER_ONLY_TAXES = ALL_TAXES)
    log_msg("\n\n###################################")
    log_msg("Ban_Comprehensive (frontier reference, type-2 subset)")
    log_msg("###################################")

    U_1_cf = copy(U_1); apply_flavor_ban!(U_1_cf, cat_idx)
    U_2_cf = copy(U_2); apply_flavor_ban!(U_2_cf, cat_idx)
    U_3_cf = copy(U_3); apply_flavor_ban!(U_3_cf, cat_idx)

    V_now_cf, V_dec_cf = solve_three_types(U_1_cf, U_2_cf, U_3_cf, "Ban_Comprehensive")
    U_1_cf = nothing; U_2_cf = nothing; U_3_cf = nothing

    t0 = time()
    sim_cf = simulate_policy(idx_type2.all, V_dec_cf, V_now_cf)
    log_msg(@sprintf("Subset simulation complete in %.1fs", time() - t0))
    V_now_cf = nothing; V_dec_cf = nothing

    for (g, rows) in FRONTIER_GROUPS_SUB
        agg_cf = aggregate_subset_paths(sim_cf, rows, cat_idx, N_J, T_sim)
        frontier_traj[("Ban_Comprehensive", Inf, g)] = agg_cf
        frontier_summ[("Ban_Comprehensive", Inf, g)] = frontier_summary(frontier_sq[g], agg_cf, Inf, omega_E_orig, T_sim)
        s = frontier_summ[("Ban_Comprehensive", Inf, g)]
        log_msg(@sprintf("  %-10s addiction p36 %+7.2f%%  avg %+7.2f%%   CS p1 \$%+9.2f  p36 %+7.2f%%   flav share p36 %.4f",
            g, s.addiction_pct_p36, s.addiction_pct_avg, s.welfare_dollar_p1, s.welfare_pct_p36, s.flav_share_p36))
    end

    sim_cf = nothing
    GC.gc()

else

    for (ban_label, ban_fn!) in BAN_TYPES
        run_full_policy(
            ban_label, Inf, U -> ban_fn!(U, cat_idx), joinpath(output_dir, ban_label), "ban";
            frontier_key = ban_label == "Ban_Comprehensive" ? "Ban_Comprehensive" : nothing
        )
    end

    for tau in TAX_GRID
        run_full_policy(
            "Flavor_Tax", tau, U -> apply_flavor_tax!(U, cat_idx, omega_E_est, q_ecig, q_ecig_max, tau),
            joinpath(output_dir, "Flavor_Tax", "Tax_$(tau_tag(tau))"), "tax";
            frontier_key = "Flavor_Tax"
        )
    end

end


#############################
# 19. Frontier-Only Tax
#     Rates (Type 2 Subset)
#############################

# The remaining frontier rates are solved and simulated only for the
# e-cigarette-persistent subset, since the frontier is drawn for that group
# and the value function solves are the whole cost.

log_msg("\n\n###################################")
log_msg("Frontier-only tax rates: $(length(FRONTIER_ONLY_TAXES)) rates on $(length(idx_type2.all)) type-2 households")
log_msg("###################################")

# The rates are ascending, so each solve starts from the previous rate's
# ex-ante value functions
V_warm = (nothing, nothing, nothing)

for tau in FRONTIER_ONLY_TAXES

    label = @sprintf("Flavor_Tax τ = \$%.2f/mL (frontier)", tau)
    log_msg("\n--- $label ---")

    U_1_cf = copy(U_1); apply_flavor_tax!(U_1_cf, cat_idx, omega_E_est, q_ecig, q_ecig_max, tau)
    U_2_cf = copy(U_2); apply_flavor_tax!(U_2_cf, cat_idx, omega_E_est, q_ecig, q_ecig_max, tau)
    U_3_cf = copy(U_3); apply_flavor_tax!(U_3_cf, cat_idx, omega_E_est, q_ecig, q_ecig_max, tau)

    V_now_cf, V_dec_cf = solve_three_types(U_1_cf, U_2_cf, U_3_cf, label; V_init = V_warm)
    U_1_cf = nothing; U_2_cf = nothing; U_3_cf = nothing
    global V_warm = V_now_cf

    t0 = time()
    sim_cf = simulate_policy(idx_type2.all, V_dec_cf, V_now_cf)
    log_msg(@sprintf("Subset simulation complete in %.1fs", time() - t0))
    V_now_cf = nothing; V_dec_cf = nothing

    for (g, rows) in FRONTIER_GROUPS_SUB
        agg_cf = aggregate_subset_paths(sim_cf, rows, cat_idx, N_J, T_sim)
        frontier_traj[("Flavor_Tax", tau, g)] = agg_cf
        frontier_summ[("Flavor_Tax", tau, g)] = frontier_summary(frontier_sq[g], agg_cf, tau, omega_E_orig, T_sim)
        s = frontier_summ[("Flavor_Tax", tau, g)]
        log_msg(@sprintf("  %-10s addiction p36 %+7.2f%%  avg %+7.2f%%   CS p1 \$%+9.2f  p36 %+7.2f%%   flav share p36 %.4f",
            g, s.addiction_pct_p36, s.addiction_pct_avg, s.welfare_dollar_p1, s.welfare_pct_p36, s.flav_share_p36))
    end

    sim_cf = nothing
    GC.gc()
end


#############################
# 20. Frontier: Fraction of
#     Ban, τ*, and Files
#############################

# Each tax rate's addiction reduction as a fraction of the comprehensive ban's,
# and the rate at which the tax matches the ban, by interpolation over the grid.

log_msg("\n===================================")
log_msg("Frontier: fraction of ban and matching tax rate τ*")
log_msg("===================================")

FRONTIER_POLICIES = vcat([("Ban_Comprehensive", Inf)], [("Flavor_Tax", tau) for tau in ALL_TAXES])

tau_star_rows = NamedTuple[]

for (g, _) in FRONTIER_GROUPS

    ban  = frontier_summ[("Ban_Comprehensive", Inf, g)]
    taus = collect(ALL_TAXES)

    eff_p36 = [frontier_summ[("Flavor_Tax", tau, g)].addiction_diff_p36 for tau in taus]
    eff_avg = [frontier_summ[("Flavor_Tax", tau, g)].addiction_diff_avg for tau in taus]
    cs_p1   = [frontier_summ[("Flavor_Tax", tau, g)].welfare_dollar_p1  for tau in taus]
    rev     = [frontier_summ[("Flavor_Tax", tau, g)].cum_revenue_per_hh for tau in taus]

    tau_star_p36 = interpolate_tau_star(taus, eff_p36, ban.addiction_diff_p36)
    tau_star_avg = interpolate_tau_star(taus, eff_avg, ban.addiction_diff_avg)

    push!(tau_star_rows, (
        group                      = g,
        tau_star_p36               = tau_star_p36,
        tau_star_avg               = tau_star_avg,
        welfare_dollar_p1_at_star  = interpolate_at_tau(taus, cs_p1, tau_star_p36),
        ban_welfare_dollar_p1      = ban.welfare_dollar_p1,
        cum_revenue_per_hh_at_star = interpolate_at_tau(taus, rev, tau_star_p36),
        ban_addiction_pct_p36      = ban.addiction_pct_p36,
        ban_addiction_pct_avg      = ban.addiction_pct_avg,
    ))

    log_msg(@sprintf("%-10s ban addiction p36 %+7.2f%%   τ* (p36) = %s   τ* (avg) = %s   CS p1 at τ*: \$%s vs ban \$%.2f",
        g, ban.addiction_pct_p36,
        isnan(tau_star_p36) ? "beyond grid" : @sprintf("\$%.2f", tau_star_p36),
        isnan(tau_star_avg) ? "beyond grid" : @sprintf("\$%.2f", tau_star_avg),
        isnan(tau_star_p36) ? "n/a" : @sprintf("%.2f", tau_star_rows[end].welfare_dollar_p1_at_star),
        ban.welfare_dollar_p1))
end

# --- Frontier_Summary: one row per (policy, τ, group) ---
summary_path = joinpath(output_dir, "Frontier_Summary_b$(b_str).csv")
open(summary_path, "w") do io
    header = ["policy", "tau", "group", "n_hh",
        "addiction_pct_p36", "addiction_pct_avg", "addiction_diff_p36", "addiction_diff_avg",
        "frac_of_ban_p36", "frac_of_ban_avg",
        "welfare_pct_p1", "welfare_pct_p36", "welfare_dollar_p1", "welfare_dollar_p36",
        "flav_share_p36", "cig_share_p36", "outside_share_p36",
        "cum_revenue_per_hh", "omega_E_est", "omega_E_orig"]
    println(io, join(header, ","))
    for (policy, tau) in FRONTIER_POLICIES, (g, rows) in FRONTIER_GROUPS
        s   = frontier_summ[(policy, tau, g)]
        ban = frontier_summ[("Ban_Comprehensive", Inf, g)]
        frac_p36 = ban.addiction_diff_p36 == 0 ? NaN : s.addiction_diff_p36 / ban.addiction_diff_p36
        frac_avg = ban.addiction_diff_avg == 0 ? NaN : s.addiction_diff_avg / ban.addiction_diff_avg
        row = [policy, isfinite(tau) ? @sprintf("%.4f", tau) : "Inf", g, @sprintf("%d", length(rows)),
            @sprintf("%.10f", s.addiction_pct_p36), @sprintf("%.10f", s.addiction_pct_avg),
            @sprintf("%.10f", s.addiction_diff_p36), @sprintf("%.10f", s.addiction_diff_avg),
            @sprintf("%.10f", frac_p36), @sprintf("%.10f", frac_avg),
            @sprintf("%.10f", s.welfare_pct_p1), @sprintf("%.10f", s.welfare_pct_p36),
            @sprintf("%.10f", s.welfare_dollar_p1), @sprintf("%.10f", s.welfare_dollar_p36),
            @sprintf("%.10f", s.flav_share_p36), @sprintf("%.10f", s.cig_share_p36), @sprintf("%.10f", s.outside_share_p36),
            @sprintf("%.10f", s.cum_revenue_per_hh), @sprintf("%.10f", omega_E_est), @sprintf("%.10f", omega_E_orig)]
        println(io, join(row, ","))
    end
end
log_msg("\nFrontier summary saved to: $summary_path")

# --- Frontier_Trajectory: one row per (policy, τ, group, period), status quo alongside ---
traj_path = joinpath(output_dir, "Frontier_Trajectory_b$(b_str).csv")
open(traj_path, "w") do io
    header = ["policy", "tau", "group", "period",
        "sq_addiction", "cf_addiction", "sq_welfare", "cf_welfare",
        "sq_q_flav_ecig", "cf_q_flav_ecig", "sq_q_cig", "cf_q_cig",
        "sq_share_flav", "cf_share_flav", "sq_share_cig", "cf_share_cig", "sq_share_outside", "cf_share_outside"]
    println(io, join(header, ","))
    for (policy, tau) in FRONTIER_POLICIES, (g, _) in FRONTIER_GROUPS
        a_sq = frontier_sq[g]
        a_cf = frontier_traj[(policy, tau, g)]
        for t in 1:T_sim
            row = [policy, isfinite(tau) ? @sprintf("%.4f", tau) : "Inf", g, @sprintf("%d", t),
                @sprintf("%.10f", a_sq.mean_addiction[t]),   @sprintf("%.10f", a_cf.mean_addiction[t]),
                @sprintf("%.10f", a_sq.mean_welfare[t]),     @sprintf("%.10f", a_cf.mean_welfare[t]),
                @sprintf("%.10f", a_sq.mean_q_flav_ecig[t]), @sprintf("%.10f", a_cf.mean_q_flav_ecig[t]),
                @sprintf("%.10f", a_sq.mean_q_cig[t]),       @sprintf("%.10f", a_cf.mean_q_cig[t]),
                @sprintf("%.10f", a_sq.share_flav[t]),       @sprintf("%.10f", a_cf.share_flav[t]),
                @sprintf("%.10f", a_sq.share_cig[t]),        @sprintf("%.10f", a_cf.share_cig[t]),
                @sprintf("%.10f", a_sq.share_outside[t]),    @sprintf("%.10f", a_cf.share_outside[t])]
            println(io, join(row, ","))
        end
    end
end
log_msg("Frontier trajectories saved to: $traj_path")

# --- Frontier_TauStar: one row per group ---
tau_star_path = joinpath(output_dir, "Frontier_TauStar_b$(b_str).csv")
open(tau_star_path, "w") do io
    header = ["group", "tau_star_p36", "tau_star_avg", "welfare_dollar_p1_at_star",
        "ban_welfare_dollar_p1", "cum_revenue_per_hh_at_star", "ban_addiction_pct_p36", "ban_addiction_pct_avg"]
    println(io, join(header, ","))
    for r in tau_star_rows
        row = [r.group,
            @sprintf("%.10f", r.tau_star_p36), @sprintf("%.10f", r.tau_star_avg),
            @sprintf("%.10f", r.welfare_dollar_p1_at_star), @sprintf("%.10f", r.ban_welfare_dollar_p1),
            @sprintf("%.10f", r.cum_revenue_per_hh_at_star),
            @sprintf("%.10f", r.ban_addiction_pct_p36), @sprintf("%.10f", r.ban_addiction_pct_avg)]
        println(io, join(row, ","))
    end
end
log_msg("τ* results saved to: $tau_star_path")


#############################
# 21. Final Timing and
#     Log Close
#############################

total_elapsed = time() - t_start
log_msg("\n\n===================================")
log_msg("Bootstrap draw b = $b COMPLETE")
log_msg("===================================")
log_msg(@sprintf("Total elapsed time: %.2f minutes (%.2f hours)", total_elapsed / 60, total_elapsed / 3600))
log_msg("All results written to: $output_dir")
log_msg("Bootstrap run complete at $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")

close(log_io)
