################################################################################
# William Brasic
# The University of Arizona
# wbrasic97@gmail.com
# June 2026
#
# Bootstrap version of 02_CF_Mixture.jl: each Slurm array task draws
# θ_b ~ N(θ̂, V̂Cov), then runs the same ban/tax pipeline at a single fixed β,
# tagging output with the draw index b. 03_CF_Bootstrap_Mixture_Aggregation.jl
# pools the draws into bootstrap confidence intervals.
#
# Three ban types plus a flavor tax are evaluated (as in 02_CF_Mixture.jl).
# For each: solve VFI under SQ and the policy, compute posterior type weights,
# then forward-simulate households via Monte Carlo using common random numbers.
# Results are aggregated overall and by TYA status / latent type.
################################################################################


#############################
# 1. Preliminaries
#############################

# β is never estimated in this pipeline.
ESTIMATE_BETA = false

# Flavored habit decay rate ψ_3
# If ESTIMATE_PSI_3=true, ψ_3 was estimated jointly as the 27th parameter.
# If false (default), ψ_3 is fixed at PSI_3 (read from ENV, default 0.50).
ESTIMATE_PSI_3 = parse(Bool, get(ENV, "ESTIMATE_PSI_3", "false"))
PSI_3 = parse(Float64, get(ENV, "PSI_3", "0.50"))

# β for this run (ENV, default 1.0)
BETA_VAL = parse(Float64, get(ENV, "BETA", "1.0"))

# Bootstrap draw index (one per Slurm array task)
b = parse(Int, get(ENV, "SLURM_ARRAY_TASK_ID", "1"))

# Detect whether I am running on the HPC (any non-Windows system)
HPC = !Sys.iswindows()

# CRN seeding for forward simulations
using Random

# Set output path and working directory
if HPC

    # Load mixture estimation functions and packages
    include("/home/u2/wbrasic/4th_Year_Paper/Dynamic_Model/02_Second_Stage_Estimation_V2/01_Functions_Mixture.jl")

    # Load counterfactual-specific functions
    include("/home/u2/wbrasic/4th_Year_Paper/Dynamic_Model/04_CF_Mixture_V2_Bootstrap/01_CF_Functions_Mixture.jl")

    # Construct psi and psi_3 tags for directory and file naming
    psi_tag   = "Psi2_09_Psi1_01"
    psi_3_tag = ESTIMATE_PSI_3 ? "Psi3_Est" : "Psi3_$(numeric_tag(PSI_3))"
    _, _, β_naming, _ = get_fixed_parameters()
    beta_tag  = numeric_tag(BETA_VAL)

    # Output path is β-specific; each β job writes to its own top-level directory.
    output_dir = "/home/u2/wbrasic/4th_Year_Paper/Dynamic_Model/04_CF_Mixture_V2_Bootstrap/CF_Bootstrap_Mixture_$(psi_tag)_$(psi_3_tag)_Beta_$(beta_tag)_Results"

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
    psi_3_tag = ESTIMATE_PSI_3 ? "Psi3_Est" : "Psi3_$(numeric_tag(PSI_3))"
    _, _, β_naming, _ = get_fixed_parameters()
    beta_tag  = numeric_tag(BETA_VAL)

    # Output path is β-specific; each β job writes to its own top-level directory.
    output_dir = "C:/Users/wbras/OneDrive/Documents/Desktop/UA/4th_Year_Paper/4th_Year_Paper_Data/HMS/2021-Onward/Dynamic_Model/CF_Bootstrap_Mixture_$(psi_tag)_$(psi_3_tag)_Beta_$(beta_tag)_Results"

    # Create output directory if it doesn't exist
    mkpath(output_dir)

    # Set working directory to where the data CSVs live
    cd("C:/Users/wbras/OneDrive/Documents/Desktop/UA/4th_Year_Paper/4th_Year_Paper_Data/HMS/2021-Onward/Dynamic_Model/Data")
end

b_str = lpad(b, 3, '0')


#############################
# 2. Output Paths
#############################

# Set log file path (bootstrap draw specific)
log_path = joinpath(output_dir, "CF_Bootstrap_b$(b_str)_Log.txt")

# Open log file for writing (log_io is defined as a global in 01_Functions_Mixture.jl)
log_io = open(log_path, "w")

# Start overall timer
t_start = time()

# Print and log counterfactual simulation start time
log_msg("Bootstrap counterfactual simulation started at $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
log_msg("Bootstrap draw b = $b (b_str = $b_str)")
log_msg("β = $BETA_VAL (tag: $beta_tag)")
log_msg("ESTIMATE_PSI_3 = $ESTIMATE_PSI_3,  PSI_3 (fixed) = $PSI_3")


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
# P_obs_cig / P_obs_ecig are N x N_J matrices of bin-specific prices (not used in CF, discarded)
_, p_continuous, _, _ = map_prices_to_grid(N_P, P, Pcomb, N_J);

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
#     (Section 13 — Bootstrap)
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
_run_tag = "Dynamic_Model_Mixture_V2_$(psi_tag)_$(_beta_tag_full)_$(psi_3_tag)"

if HPC
    _est_base = "/home/u2/wbrasic/4th_Year_Paper/Dynamic_Model/02_Second_Stage_Estimation_V2"
else
    _est_base = "C:/Users/wbras/OneDrive/Documents/Desktop/UA/4th_Year_Paper/4th_Year_Paper_Data/HMS/2021-Onward/Dynamic_Model"
end

_est_dir       = joinpath(_est_base, "$(_run_tag)_Results", "$(_run_tag)_Estimates")
estimates_path = joinpath(_est_dir, "$(_run_tag)_Estimates.csv")
vcov_path      = joinpath(_est_dir, "$(_run_tag)_VCov.csv")

# Load θ̂
_df_est = CSV.read(estimates_path, DataFrame)
if "NLL" in names(_df_est)
    select!(_df_est, Not(:NLL))
end
param_names = names(_df_est)
D_params    = length(param_names)
θ_hat       = Float64.(collect(_df_est[1, :]))
log_msg("Loaded θ̂: $(D_params) parameters from $estimates_path")

# Load V̂Cov = H⁻¹ 
_df_vcov = CSV.read(vcov_path, DataFrame)
VCov     = Matrix{Float64}(_df_vcov)
log_msg("Loaded V̂Cov: $(size(VCov,1))×$(size(VCov,2)) from $vcov_path")

# Cholesky factor for sampling: θ_b = θ̂ + L_vcov * z, z ~ N(0,I)
# Clip negative eigenvalues to 1e-10 before Cholesky.
_eig     = eigen(Symmetric(VCov))
_vals_pd = max.(_eig.values, 1e-10)
VCov_pd  = Symmetric(_eig.vectors * Diagonal(_vals_pd) * _eig.vectors')
L_vcov   = cholesky(VCov_pd).L

# Seed RNG with b before sampling θ_b (CRN seed applied later in section 14)
Random.seed!(b)
log_msg("RNG seeded with b = $b for parameter draw")

# Parameter bound check (re-draw until satisfied)
function _valid_draw(θ, est_psi_3)
    θ[1] >= 0.0  &&   # alpha_C ≥ 0
    θ[2] >= 0.0  &&   # alpha_E ≥ 0
    θ[8] <= 0.0  &&   # gamma_1 ≤ 0
    θ[9] <= 0.0  &&   # gamma_2 ≤ 0
    θ[10] <= 0.0 &&   # gamma_3 ≤ 0
    θ[11] <= 0.0 &&   # gamma_4 ≤ 0
    θ[12] <= 0.0 &&   # omega_C ≤ 0
    θ[13] <= 0.0 &&   # omega_E ≤ 0
    (!est_psi_3 || (length(θ) >= 27 && 0.01 < θ[27] < 0.99))
end

θ_b = copy(θ_hat)
n_redraws = 0
while true
    θ_b .= θ_hat .+ L_vcov * randn(D_params)
    _valid_draw(θ_b, ESTIMATE_PSI_3) && break
    global n_redraws += 1
end
log_msg("")
log_msg(@sprintf("θ_b drawn: %d re-draw(s) needed", n_redraws))
log_msg("")
log_msg(@sprintf("  %-14s   %-20s   %-20s   %s", "Parameter", "θ̂ (MLE)", "θ_b (draw)", "Δ"))
log_msg("  " * "-"^74)
for k in 1:D_params
    delta = θ_b[k] - θ_hat[k]
    log_msg(@sprintf("  %-14s = %20.10f   %20.10f   %+.6f",
        param_names[k], θ_hat[k], θ_b[k], delta))
end
log_msg("")

# Unpack θ_b
common_b    = θ_b[1:13]
ξ_1_b       = θ_b[14:16]
ξ_2_b       = θ_b[17:19]
ξ_3_b       = θ_b[20:22]
π_0_2       = θ_b[23]
π_TYA_2     = θ_b[24]
π_0_3       = θ_b[25]
π_TYA_3     = θ_b[26]
omega_E_est = common_b[13]

PSI_3 = ESTIMATE_PSI_3 ? θ_b[27] : parse(Float64, get(ENV, "PSI_3", "0.50"))
log_msg(@sprintf("ψ_3 = %.6f (%s)", PSI_3, ESTIMATE_PSI_3 ? "estimated (θ_b[27])" : "fixed from ENV"))
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
    is_outside, cat_idx, Pcomb, has_cig, has_ecig
)
U_2 = get_flow_utility(
    θ_struct_2, N_J, N_A_f, N_A_s, N_A_flav, N_Pcomb, A_f, A_s, A_flav,
    q_cig, q_ecig, q_bundle, is_flavored, is_fda_flavored, is_nonflavored_ecig,
    is_outside, cat_idx, Pcomb, has_cig, has_ecig
)
U_3 = get_flow_utility(
    θ_struct_3, N_J, N_A_f, N_A_s, N_A_flav, N_Pcomb, A_f, A_s, A_flav,
    q_cig, q_ecig, q_bundle, is_flavored, is_fda_flavored, is_nonflavored_ecig,
    is_outside, cat_idx, Pcomb, has_cig, has_ecig
)
log_msg("Flow utilities computed from θ_b.")

log_msg("\nAll bootstrap parameter objects ready.")


#############################
# 14. Pre-draw CRN Variates
#     (Once, Before Any Loop)
#############################

# All variates are drawn ONCE, before any policy loop, and reused across every
# ban/tax scenario, so SQ-vs-policy differences reflect the policy itself, not
# simulation noise. 

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
# 15. Category Labels
#############################

cat_labels = ["Outside", "Cig", "Orig Ecig", "Non-FDA Flav Ecig", "FDA Flav Ecig", "Orig Bundle", "Non-FDA Flav Bundle", "FDA Flav Bundle"];

#############################
# 16. Ban Type Loop
#############################

# Tuple of ban types and ban functions
BAN_TYPES = [
    ("Ban_Comprehensive", apply_flavor_ban!),
    ("Ban_FDA_Only", apply_fda_flavor_ban!),
    ("Ban_Non_FDA", apply_non_fda_ban!)
]

log_msg("\n===================================")
log_msg("Starting ban type loop: $(length(BAN_TYPES)) ban types")
log_msg("===================================")

for (ban_label, ban_fn!) in BAN_TYPES

    log_msg("\n\n###################################")
    log_msg("Ban type: $ban_label")
    log_msg("###################################")

    # Create ban-specific subdirectory
    ban_subdir = joinpath(output_dir, ban_label)
    mkpath(ban_subdir)

    # Log which alternatives are banned. Defined here (not inside a nested block)
    # so banned_alts is accessible in the posterior/pointwise sections below.
    local _U_log_tmp = copy(U_1)
    ban_fn!(_U_log_tmp, cat_idx)
    local banned_alts = findall(j -> _U_log_tmp[1, j, 1, 1, 1, 1] == -Inf, 1:N_J)
    log_msg("Banned alternatives: j = $(banned_alts)")
    _U_log_tmp = nothing

    local beta_subdir = ban_subdir  # β is encoded in output_dir; no separate Beta_* subdir
    mkpath(beta_subdir)

    log_msg("\n===================================")
    log_msg("Running β = $BETA_VAL (tag: $beta_tag), bootstrap draw b = $b")
    log_msg("===================================")

    # Declare loop variables
    local t_vfi_sq, t_vfi_ban, vfi_sq_elapsed, vfi_ban_elapsed
    local t_post, post_elapsed, t_pw, pw_elapsed
    local t_sim_fwd, sim_fwd_elapsed
    local probs_sq, probs_ban, welfare_sq, welfare_ban, welfare_diff
    local hh_posterior_type1, hh_posterior_type2, hh_posterior_type3
    local obs_posterior_type1, obs_posterior_type2, obs_posterior_type3
    local sim_choices_sq, sim_addiction_f_sq, sim_addiction_s_sq, sim_aflav_sq, sim_welfare_sq_arr
    local sim_choices_ban, sim_addiction_f_ban, sim_addiction_s_ban, sim_aflav_ban, sim_welfare_ban_arr
    local sim_addiction_sq, sim_addiction_ban
    local agg_sq, agg_ban, agg_sq_tya, agg_ban_tya
    local agg_sq_type1, agg_sq_type2, agg_sq_type3, agg_ban_type1, agg_ban_type2, agg_ban_type3

    # Apply ban to base flow utilities (script-level U_1, U_2, U_3 from θ_b)
    local U_1_ban = copy(U_1); ban_fn!(U_1_ban, cat_idx)
    local U_2_ban = copy(U_2); ban_fn!(U_2_ban, cat_idx)
    local U_3_ban = copy(U_3); ban_fn!(U_3_ban, cat_idx)


    #############################
    # Solve VFI: Status Quo
    #############################

    log_msg("\n===================================")
    log_msg("Solving VFI: Status Quo (β = $BETA_VAL)")
    log_msg("===================================")

    t_vfi_sq = time();

    vfi_task_1_sq = Threads.@spawn solve_vfi_sophisticated(
        N_J, N_A_f, N_A_s, N_A_flav, N_P, N_Pcomb, BETA_VAL, δ, U_1,
        af_lower, af_upper, af_weight,
        as_lower, as_upper, as_weight,
        aflav_lower, aflav_upper, aflav_weight,
        p_cig_lo, p_cig_hi, p_cig_w,
        p_ecig_lo, p_ecig_hi, p_ecig_w;
        V_init = nothing,
        verbose = true
    )

    vfi_task_2_sq = Threads.@spawn solve_vfi_sophisticated(
        N_J, N_A_f, N_A_s, N_A_flav, N_P, N_Pcomb, BETA_VAL, δ, U_2,
        af_lower, af_upper, af_weight,
        as_lower, as_upper, as_weight,
        aflav_lower, aflav_upper, aflav_weight,
        p_cig_lo, p_cig_hi, p_cig_w,
        p_ecig_lo, p_ecig_hi, p_ecig_w;
        V_init = nothing,
        verbose = true
    )

    vfi_task_3_sq = Threads.@spawn solve_vfi_sophisticated(
        N_J, N_A_f, N_A_s, N_A_flav, N_P, N_Pcomb, BETA_VAL, δ, U_3,
        af_lower, af_upper, af_weight,
        as_lower, as_upper, as_weight,
        aflav_lower, aflav_upper, aflav_weight,
        p_cig_lo, p_cig_hi, p_cig_w,
        p_ecig_lo, p_ecig_hi, p_ecig_w;
        V_init = nothing,
        verbose = true
    )

    # Wait for all three VFI tasks to complete
    V_now_1_sq, V_decision_1_sq, vfi_iters_1_sq, vfi_converged_1_sq = fetch(vfi_task_1_sq)
    V_now_2_sq, V_decision_2_sq, vfi_iters_2_sq, vfi_converged_2_sq = fetch(vfi_task_2_sq)
    V_now_3_sq, V_decision_3_sq, vfi_iters_3_sq, vfi_converged_3_sq = fetch(vfi_task_3_sq)

    vfi_sq_elapsed = time() - t_vfi_sq;
    log_msg("Status quo VFI: type1=$(vfi_iters_1_sq) iters ($(vfi_converged_1_sq)), type2=$(vfi_iters_2_sq) iters ($(vfi_converged_2_sq)), type3=$(vfi_iters_3_sq) iters ($(vfi_converged_3_sq)), $(round(vfi_sq_elapsed, digits=1))s")


    #############################
    # Solve VFI: Flavor Ban
    #############################

    log_msg("\n===================================")
    log_msg("Solving VFI: Flavor Ban (β = $BETA_VAL)")
    log_msg("===================================")

    t_vfi_ban = time();

    vfi_task_1_ban = Threads.@spawn solve_vfi_sophisticated(
        N_J, N_A_f, N_A_s, N_A_flav, N_P, N_Pcomb, BETA_VAL, δ, U_1_ban,
        af_lower, af_upper, af_weight,
        as_lower, as_upper, as_weight,
        aflav_lower, aflav_upper, aflav_weight,
        p_cig_lo, p_cig_hi, p_cig_w,
        p_ecig_lo, p_ecig_hi, p_ecig_w;
        V_init = nothing,
        verbose = true
    )

    vfi_task_2_ban = Threads.@spawn solve_vfi_sophisticated(
        N_J, N_A_f, N_A_s, N_A_flav, N_P, N_Pcomb, BETA_VAL, δ, U_2_ban,
        af_lower, af_upper, af_weight,
        as_lower, as_upper, as_weight,
        aflav_lower, aflav_upper, aflav_weight,
        p_cig_lo, p_cig_hi, p_cig_w,
        p_ecig_lo, p_ecig_hi, p_ecig_w;
        V_init = nothing,
        verbose = true
    )

    vfi_task_3_ban = Threads.@spawn solve_vfi_sophisticated(
        N_J, N_A_f, N_A_s, N_A_flav, N_P, N_Pcomb, BETA_VAL, δ, U_3_ban,
        af_lower, af_upper, af_weight,
        as_lower, as_upper, as_weight,
        aflav_lower, aflav_upper, aflav_weight,
        p_cig_lo, p_cig_hi, p_cig_w,
        p_ecig_lo, p_ecig_hi, p_ecig_w;
        V_init = nothing,
        verbose = true
    )

    # Wait for all three VFI tasks to complete
    V_now_1_ban, V_decision_1_ban, vfi_iters_1_ban, vfi_converged_1_ban = fetch(vfi_task_1_ban)
    V_now_2_ban, V_decision_2_ban, vfi_iters_2_ban, vfi_converged_2_ban = fetch(vfi_task_2_ban)
    V_now_3_ban, V_decision_3_ban, vfi_iters_3_ban, vfi_converged_3_ban = fetch(vfi_task_3_ban)

    vfi_ban_elapsed = time() - t_vfi_ban;
    log_msg("Flavor ban VFI: type1=$(vfi_iters_1_ban) iters ($(vfi_converged_1_ban)), type2=$(vfi_iters_2_ban) iters ($(vfi_converged_2_ban)), type3=$(vfi_iters_3_ban) iters ($(vfi_converged_3_ban)), $(round(vfi_ban_elapsed, digits=1))s")


    #############################
    # Compute Posterior Type
    # Weights
    #############################

    log_msg("\n===================================")
    log_msg("Computing posterior type weights...")
    log_msg("===================================")

    t_post = time();

    probs_1_sq, welfare_1_sq = compute_pointwise_outcomes(
        V_decision_1_sq, tya_state, af_continuous, as_continuous, aflav_continuous, p_continuous, N_J, N_P, A_f, A_s, A_flav, P
    )
    probs_2_sq, welfare_2_sq = compute_pointwise_outcomes(
        V_decision_2_sq, tya_state, af_continuous, as_continuous, aflav_continuous, p_continuous, N_J, N_P, A_f, A_s, A_flav, P
    )
    probs_3_sq, welfare_3_sq = compute_pointwise_outcomes(
        V_decision_3_sq, tya_state, af_continuous, as_continuous, aflav_continuous, p_continuous, N_J, N_P, A_f, A_s, A_flav, P
    )

    probs_1_ban, welfare_1_ban = compute_pointwise_outcomes(
        V_decision_1_ban, tya_state, af_continuous, as_continuous, aflav_continuous, p_continuous, N_J, N_P, A_f, A_s, A_flav, P
    )
    probs_2_ban, welfare_2_ban = compute_pointwise_outcomes(
        V_decision_2_ban, tya_state, af_continuous, as_continuous, aflav_continuous, p_continuous, N_J, N_P, A_f, A_s, A_flav, P
    )
    probs_3_ban, welfare_3_ban = compute_pointwise_outcomes(
        V_decision_3_ban, tya_state, af_continuous, as_continuous, aflav_continuous, p_continuous, N_J, N_P, A_f, A_s, A_flav, P
    )

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

        ll_1 = 0.0
        ll_2 = 0.0
        ll_3 = 0.0
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

    post_elapsed = time() - t_post;
    mean_w1 = mean(hh_posterior_type1)
    mean_w2 = mean(hh_posterior_type2)
    mean_w3 = mean(hh_posterior_type3)
    log_msg("Posterior type weights computed in $(round(post_elapsed, digits=1))s")
    log_msg(@sprintf("  Mean P(type=1) = %.4f, Mean P(type=2) = %.4f, Mean P(type=3) = %.4f", mean_w1, mean_w2, mean_w3))


    #############################
    # Pointwise Outcomes
    # (Mixture-Weighted)
    #############################

    log_msg("\n===================================")
    log_msg("Computing mixture-weighted pointwise outcomes...")
    log_msg("===================================")

    t_pw = time();

    obs_posterior_type1 = Vector{Float64}(undef, N_obs)
    obs_posterior_type2 = Vector{Float64}(undef, N_obs)
    obs_posterior_type3 = Vector{Float64}(undef, N_obs)
    for h in 1:N_HH
        start_idx, stop_idx = hh_ranges[h]
        for i in start_idx:stop_idx
            obs_posterior_type1[i] = hh_posterior_type1[h]
            obs_posterior_type2[i] = hh_posterior_type2[h]
            obs_posterior_type3[i] = hh_posterior_type3[h]
        end
    end

    probs_sq   = obs_posterior_type1 .* probs_1_sq  .+ obs_posterior_type2 .* probs_2_sq  .+ obs_posterior_type3 .* probs_3_sq
    welfare_sq = obs_posterior_type1 .* welfare_1_sq .+ obs_posterior_type2 .* welfare_2_sq .+ obs_posterior_type3 .* welfare_3_sq
    probs_ban   = obs_posterior_type1 .* probs_1_ban .+ obs_posterior_type2 .* probs_2_ban .+ obs_posterior_type3 .* probs_3_ban
    welfare_ban = obs_posterior_type1 .* welfare_1_ban .+ obs_posterior_type2 .* welfare_2_ban .+ obs_posterior_type3 .* welfare_3_ban
    pw_elapsed = time() - t_pw;
    log_msg("Pointwise outcomes computed in $(round(pw_elapsed, digits=1))s")

    max_banned_prob = maximum(probs_ban[:, banned_alts]);
    log_msg("Max probability of banned alternatives under ban: $max_banned_prob")

    welfare_diff = welfare_ban .- welfare_sq;
    max_welfare_increase = maximum(welfare_diff);
    log_msg("Max welfare increase under ban (should be <= 0): $max_welfare_increase")

    log_msg("\nPointwise summary (means across all observations):")
    log_msg(@sprintf("  %-22s  %12s  %12s  %12s", "Category", "SQ Share", "Ban Share", "Difference"))
    log_msg("  " * repeat("-", 62))

    for (c, label) in enumerate(cat_labels)
        cat_val = c - 1
        alt_indices = findall(j -> cat_idx[j] == cat_val, 1:N_J)
        sq_share  = mean(sum(probs_sq[:, alt_indices], dims=2))
        ban_share = mean(sum(probs_ban[:, alt_indices], dims=2))
        log_msg(@sprintf("  %-22s  %12.6f  %12.6f  %12.6f", label, sq_share, ban_share, ban_share - sq_share))
    end

    log_msg(@sprintf("\n  Mean welfare SQ:   %.6f", mean(welfare_sq)))
    log_msg(@sprintf("  Mean welfare Ban:  %.6f", mean(welfare_ban)))
    log_msg(@sprintf("  Mean welfare loss: %.6f", mean(welfare_diff)))


    #############################
    # Forward Simulation
    # (Mixture with CRN)
    #############################

    log_msg("\n===================================")
    log_msg("Forward simulation (β = $BETA_VAL, b = $b)...")
    log_msg("===================================")
    log_msg("T_sim = $T_sim, N_draws = $N_draws, N_HH = $N_HH")

    sim_choices_sq      = Array{Int}(undef, N_HH, T_sim, N_draws)
    sim_addiction_f_sq  = Array{Float64}(undef, N_HH, T_sim, N_draws)
    sim_addiction_s_sq  = Array{Float64}(undef, N_HH, T_sim, N_draws)
    sim_aflav_sq        = Array{Float64}(undef, N_HH, T_sim, N_draws)
    sim_welfare_sq_arr  = Array{Float64}(undef, N_HH, T_sim, N_draws)
    sim_q_cig_sq        = Array{Float64}(undef, N_HH, T_sim, N_draws)
    sim_q_orig_ecig_sq  = Array{Float64}(undef, N_HH, T_sim, N_draws)
    sim_q_flav_ecig_sq  = Array{Float64}(undef, N_HH, T_sim, N_draws)

    sim_choices_ban     = Array{Int}(undef, N_HH, T_sim, N_draws)
    sim_addiction_f_ban = Array{Float64}(undef, N_HH, T_sim, N_draws)
    sim_addiction_s_ban = Array{Float64}(undef, N_HH, T_sim, N_draws)
    sim_aflav_ban       = Array{Float64}(undef, N_HH, T_sim, N_draws)
    sim_welfare_ban_arr = Array{Float64}(undef, N_HH, T_sim, N_draws)
    sim_q_cig_ban       = Array{Float64}(undef, N_HH, T_sim, N_draws)
    sim_q_orig_ecig_ban = Array{Float64}(undef, N_HH, T_sim, N_draws)
    sim_q_flav_ecig_ban = Array{Float64}(undef, N_HH, T_sim, N_draws)

    local P_cig  = P[:, 1]
    local P_ecig = P[:, 2]

    t_sim_fwd = time();

    Threads.@threads for h in 1:N_HH

        tya_idx_h = hh_tya[h]
        w1 = hh_posterior_type1[h]
        w2 = hh_posterior_type2[h]

        for d in 1:N_draws

            u_type = crn_type[h, d]
            type_k = u_type <= w1 ? 1 : (u_type <= w1 + w2 ? 2 : 3)
            V_sq  = type_k == 1 ? V_decision_1_sq  : (type_k == 2 ? V_decision_2_sq  : V_decision_3_sq)
            V_ban = type_k == 1 ? V_decision_1_ban : (type_k == 2 ? V_decision_2_ban : V_decision_3_ban)
            V_now_sq  = type_k == 1 ? V_now_1_sq  : (type_k == 2 ? V_now_2_sq  : V_now_3_sq)
            V_now_ban = type_k == 1 ? V_now_1_ban : (type_k == 2 ? V_now_2_ban : V_now_3_ban)

            a_f_sq      = hh_af0[h]
            a_s_sq      = hh_as0[h]
            a_flav_sq   = hh_aflav0[h]
            p_cig_sq    = hh_p0[h, 1]
            p_ecig_sq   = hh_p0[h, 2]

            a_f_ban     = hh_af0[h]
            a_s_ban     = hh_as0[h]
            a_flav_ban  = hh_aflav0[h]
            p_cig_ban   = hh_p0[h, 1]
            p_ecig_ban  = hh_p0[h, 2]

            tya_sq  = tya_idx_h
            tya_ban = tya_idx_h

            for t in 1:T_sim

                # --- STATUS QUO ---
                v_interp_sq = interpolate_v_choice(
                    V_sq, tya_sq, a_f_sq, a_s_sq, a_flav_sq, p_cig_sq, p_ecig_sq,
                    N_J, N_P, A_f, A_s, A_flav, P
                )
                replace!(v_interp_sq, NaN => -Inf)

                sim_welfare_sq_arr[h, t, d] = interpolate_v_now(
                    V_now_sq, tya_sq, a_f_sq, a_s_sq, a_flav_sq, p_cig_sq, p_ecig_sq,
                    N_P, A_f, A_s, A_flav, P
                )

                v_max_sq = maximum(v_interp_sq)
                v_shifted_sq = v_interp_sq .- v_max_sq
                exp_v_sq = exp.(v_shifted_sq)
                probs_h_sq = exp_v_sq ./ sum(exp_v_sq)

                u_draw = crn_choice[h, t, d]
                j_sq = 1
                cum_prob = probs_h_sq[1]
                while cum_prob < u_draw && j_sq < N_J
                    j_sq += 1
                    cum_prob += probs_h_sq[j_sq]
                end
                sim_choices_sq[h, t, d] = j_sq
                sim_q_cig_sq[h, t, d]       = q_cig_raw[j_sq]
                sim_q_orig_ecig_sq[h, t, d] = q_orig_ecig_raw[j_sq]
                sim_q_flav_ecig_sq[h, t, d] = q_flav_ecig_raw[j_sq]

                a_f_sq = addiction_evolution(ψ_2, a_f_sq, n[j_sq])
                a_f_sq = clamp(a_f_sq, A_f[1], A_f[end])
                sim_addiction_f_sq[h, t, d] = a_f_sq

                a_s_sq = addiction_evolution(ψ_1, a_s_sq, n[j_sq])
                a_s_sq = clamp(a_s_sq, A_s[1], A_s[end])
                sim_addiction_s_sq[h, t, d] = a_s_sq

                a_flav_sq = addiction_evolution(PSI_3, a_flav_sq, n_flav[j_sq])
                a_flav_sq = clamp(a_flav_sq, A_flav[1], A_flav[end])
                sim_aflav_sq[h, t, d] = a_flav_sq

                ε_sq = L_chol * [crn_price[h, t, d, 1], crn_price[h, t, d, 2]]
                p_cig_sq  = clamp(φ_0[1] + φ_1[1] * p_cig_sq  + ε_sq[1], P_cig[1], P_cig[end])
                p_ecig_sq = clamp(φ_0[2] + φ_1[2] * p_ecig_sq + ε_sq[2], P_ecig[1], P_ecig[end])

                # --- FLAVOR BAN ---
                v_interp_ban = interpolate_v_choice(
                    V_ban, tya_ban, a_f_ban, a_s_ban, a_flav_ban, p_cig_ban, p_ecig_ban,
                    N_J, N_P, A_f, A_s, A_flav, P
                )
                replace!(v_interp_ban, NaN => -Inf)

                sim_welfare_ban_arr[h, t, d] = interpolate_v_now(
                    V_now_ban, tya_ban, a_f_ban, a_s_ban, a_flav_ban, p_cig_ban, p_ecig_ban,
                    N_P, A_f, A_s, A_flav, P
                )

                v_max_ban = maximum(v_interp_ban)
                v_shifted_ban = v_interp_ban .- v_max_ban
                exp_v_ban = exp.(v_shifted_ban)
                probs_h_ban = exp_v_ban ./ sum(exp_v_ban)

                j_ban = 1
                cum_prob_ban = probs_h_ban[1]
                while cum_prob_ban < u_draw && j_ban < N_J
                    j_ban += 1
                    cum_prob_ban += probs_h_ban[j_ban]
                end
                sim_choices_ban[h, t, d] = j_ban
                sim_q_cig_ban[h, t, d]       = q_cig_raw[j_ban]
                sim_q_orig_ecig_ban[h, t, d] = q_orig_ecig_raw[j_ban]
                sim_q_flav_ecig_ban[h, t, d] = q_flav_ecig_raw[j_ban]

                a_f_ban = addiction_evolution(ψ_2, a_f_ban, n[j_ban])
                a_f_ban = clamp(a_f_ban, A_f[1], A_f[end])
                sim_addiction_f_ban[h, t, d] = a_f_ban

                a_s_ban = addiction_evolution(ψ_1, a_s_ban, n[j_ban])
                a_s_ban = clamp(a_s_ban, A_s[1], A_s[end])
                sim_addiction_s_ban[h, t, d] = a_s_ban

                a_flav_ban = addiction_evolution(PSI_3, a_flav_ban, n_flav[j_ban])
                a_flav_ban = clamp(a_flav_ban, A_flav[1], A_flav[end])
                sim_aflav_ban[h, t, d] = a_flav_ban

                ε_ban = L_chol * [crn_price[h, t, d, 1], crn_price[h, t, d, 2]]
                p_cig_ban  = clamp(φ_0[1] + φ_1[1] * p_cig_ban  + ε_ban[1], P_cig[1], P_cig[end])
                p_ecig_ban = clamp(φ_0[2] + φ_1[2] * p_ecig_ban + ε_ban[2], P_ecig[1], P_ecig[end])

            end
        end
    end

    sim_fwd_elapsed = time() - t_sim_fwd;
    log_msg("Forward simulation complete in $(round(sim_fwd_elapsed, digits=1))s")


    #############################
    # Aggregate and Save
    #############################

    log_msg("\n===================================")
    log_msg("Aggregating results...")
    log_msg("===================================")

    sim_addiction_sq  = (sim_addiction_f_sq  .+ sim_addiction_s_sq)  ./ 2.0;
    sim_addiction_ban = (sim_addiction_f_ban .+ sim_addiction_s_ban) ./ 2.0;

    agg_sq = aggregate_simulation(sim_choices_sq, sim_addiction_sq, sim_aflav_sq, sim_welfare_sq_arr, sim_q_cig_sq, sim_q_orig_ecig_sq, sim_q_flav_ecig_sq, cat_idx, N_J, T_sim);
    agg_ban = aggregate_simulation(sim_choices_ban, sim_addiction_ban, sim_aflav_ban, sim_welfare_ban_arr, sim_q_cig_ban, sim_q_orig_ecig_ban, sim_q_flav_ecig_ban, cat_idx, N_J, T_sim);

    agg_sq_tya = aggregate_simulation_by_tya(
        sim_choices_sq, sim_addiction_sq, sim_aflav_sq, sim_welfare_sq_arr, sim_q_cig_sq, sim_q_orig_ecig_sq, sim_q_flav_ecig_sq, cat_idx, N_J, T_sim, hh_tya, N_draws
    )
    agg_ban_tya = aggregate_simulation_by_tya(
        sim_choices_ban, sim_addiction_ban, sim_aflav_ban, sim_welfare_ban_arr, sim_q_cig_ban, sim_q_orig_ecig_ban, sim_q_flav_ecig_ban, cat_idx, N_J, T_sim, hh_tya, N_draws
    )

    agg_sq_type1, agg_sq_type2, agg_sq_type3 = aggregate_simulation_by_type_k3(
        sim_choices_sq, sim_addiction_sq, sim_aflav_sq, sim_welfare_sq_arr, sim_q_cig_sq, sim_q_orig_ecig_sq, sim_q_flav_ecig_sq, cat_idx, N_J, T_sim, hh_posterior_type1, hh_posterior_type2, hh_posterior_type3, N_draws
    )
    agg_ban_type1, agg_ban_type2, agg_ban_type3 = aggregate_simulation_by_type_k3(
        sim_choices_ban, sim_addiction_ban, sim_aflav_ban, sim_welfare_ban_arr, sim_q_cig_ban, sim_q_orig_ecig_ban, sim_q_flav_ecig_ban, cat_idx, N_J, T_sim, hh_posterior_type1, hh_posterior_type2, hh_posterior_type3, N_draws
    )

    agg_sq_tya_type1, agg_sq_tya_type2, agg_sq_tya_type3 = aggregate_simulation_by_tya_type_k3(
        sim_choices_sq, sim_addiction_sq, sim_aflav_sq, sim_welfare_sq_arr, sim_q_cig_sq, sim_q_orig_ecig_sq, sim_q_flav_ecig_sq, cat_idx, N_J, T_sim, hh_tya, hh_posterior_type1, hh_posterior_type2, hh_posterior_type3, N_draws
    )
    agg_ban_tya_type1, agg_ban_tya_type2, agg_ban_tya_type3 = aggregate_simulation_by_tya_type_k3(
        sim_choices_ban, sim_addiction_ban, sim_aflav_ban, sim_welfare_ban_arr, sim_q_cig_ban, sim_q_orig_ecig_ban, sim_q_flav_ecig_ban, cat_idx, N_J, T_sim, hh_tya, hh_posterior_type1, hh_posterior_type2, hh_posterior_type3, N_draws
    )

    # --- Save Overall Simulation Results ---
    sim_results_path = joinpath(beta_subdir, "Simulation_Overall_b$(b_str).csv");
    open(sim_results_path, "w") do io
        header = ["period",
            "sq_outside", "sq_cig", "sq_orig_ecig", "sq_non_fda_flav_ecig", "sq_fda_flav_ecig",
            "sq_orig_bundle", "sq_non_fda_flav_bundle", "sq_fda_flav_bundle",
            "sq_addiction", "sq_aflav", "sq_welfare", "sq_q_cig", "sq_q_orig_ecig", "sq_q_flav_ecig",
            "ban_outside", "ban_cig", "ban_orig_ecig", "ban_non_fda_flav_ecig", "ban_fda_flav_ecig",
            "ban_orig_bundle", "ban_non_fda_flav_bundle", "ban_fda_flav_bundle",
            "ban_addiction", "ban_aflav", "ban_welfare", "ban_q_cig", "ban_q_orig_ecig", "ban_q_flav_ecig"]
        println(io, join(header, ","))
        for t in 1:T_sim
            row = [@sprintf("%d", t),
                @sprintf("%.10f", agg_sq.share_outside[t]), @sprintf("%.10f", agg_sq.share_cig[t]),
                @sprintf("%.10f", agg_sq.share_orig_ecig[t]), @sprintf("%.10f", agg_sq.share_non_fda_flav_ecig[t]),
                @sprintf("%.10f", agg_sq.share_fda_flav_ecig[t]), @sprintf("%.10f", agg_sq.share_orig_bundle[t]),
                @sprintf("%.10f", agg_sq.share_non_fda_flav_bundle[t]), @sprintf("%.10f", agg_sq.share_fda_flav_bundle[t]),
                @sprintf("%.10f", agg_sq.mean_addiction[t]), @sprintf("%.10f", agg_sq.mean_aflav[t]),
                @sprintf("%.10f", agg_sq.mean_welfare[t]), @sprintf("%.10f", agg_sq.mean_q_cig[t]), @sprintf("%.10f", agg_sq.mean_q_orig_ecig[t]), @sprintf("%.10f", agg_sq.mean_q_flav_ecig[t]),
                @sprintf("%.10f", agg_ban.share_outside[t]), @sprintf("%.10f", agg_ban.share_cig[t]),
                @sprintf("%.10f", agg_ban.share_orig_ecig[t]), @sprintf("%.10f", agg_ban.share_non_fda_flav_ecig[t]),
                @sprintf("%.10f", agg_ban.share_fda_flav_ecig[t]), @sprintf("%.10f", agg_ban.share_orig_bundle[t]),
                @sprintf("%.10f", agg_ban.share_non_fda_flav_bundle[t]), @sprintf("%.10f", agg_ban.share_fda_flav_bundle[t]),
                @sprintf("%.10f", agg_ban.mean_addiction[t]), @sprintf("%.10f", agg_ban.mean_aflav[t]),
                @sprintf("%.10f", agg_ban.mean_welfare[t]), @sprintf("%.10f", agg_ban.mean_q_cig[t]), @sprintf("%.10f", agg_ban.mean_q_orig_ecig[t]), @sprintf("%.10f", agg_ban.mean_q_flav_ecig[t])]
            println(io, join(row, ","))
        end
    end
    log_msg("Overall simulation results saved to: $sim_results_path")

    # --- Save Simulation by TYA Status ---
    sim_tya_path = joinpath(beta_subdir, "Simulation_by_TYA_b$(b_str).csv");
    open(sim_tya_path, "w") do io
        header = ["group", "period",
            "sq_outside", "sq_cig", "sq_orig_ecig", "sq_non_fda_flav_ecig", "sq_fda_flav_ecig",
            "sq_orig_bundle", "sq_non_fda_flav_bundle", "sq_fda_flav_bundle",
            "sq_addiction", "sq_aflav", "sq_welfare", "sq_q_cig", "sq_q_orig_ecig", "sq_q_flav_ecig",
            "ban_outside", "ban_cig", "ban_orig_ecig", "ban_non_fda_flav_ecig", "ban_fda_flav_ecig",
            "ban_orig_bundle", "ban_non_fda_flav_bundle", "ban_fda_flav_bundle",
            "ban_addiction", "ban_aflav", "ban_welfare", "ban_q_cig", "ban_q_orig_ecig", "ban_q_flav_ecig"]
        println(io, join(header, ","))
        for (group_label, agg_sq_g, agg_ban_g) in [("tya", agg_sq_tya, agg_ban_tya)]
            for t in 1:T_sim
                row = [group_label, @sprintf("%d", t),
                    @sprintf("%.10f", agg_sq_g.share_outside[t]), @sprintf("%.10f", agg_sq_g.share_cig[t]),
                    @sprintf("%.10f", agg_sq_g.share_orig_ecig[t]), @sprintf("%.10f", agg_sq_g.share_non_fda_flav_ecig[t]),
                    @sprintf("%.10f", agg_sq_g.share_fda_flav_ecig[t]), @sprintf("%.10f", agg_sq_g.share_orig_bundle[t]),
                    @sprintf("%.10f", agg_sq_g.share_non_fda_flav_bundle[t]), @sprintf("%.10f", agg_sq_g.share_fda_flav_bundle[t]),
                    @sprintf("%.10f", agg_sq_g.mean_addiction[t]), @sprintf("%.10f", agg_sq_g.mean_aflav[t]),
                    @sprintf("%.10f", agg_sq_g.mean_welfare[t]), @sprintf("%.10f", agg_sq_g.mean_q_cig[t]), @sprintf("%.10f", agg_sq_g.mean_q_orig_ecig[t]), @sprintf("%.10f", agg_sq_g.mean_q_flav_ecig[t]),
                    @sprintf("%.10f", agg_ban_g.share_outside[t]), @sprintf("%.10f", agg_ban_g.share_cig[t]),
                    @sprintf("%.10f", agg_ban_g.share_orig_ecig[t]), @sprintf("%.10f", agg_ban_g.share_non_fda_flav_ecig[t]),
                    @sprintf("%.10f", agg_ban_g.share_fda_flav_ecig[t]), @sprintf("%.10f", agg_ban_g.share_orig_bundle[t]),
                    @sprintf("%.10f", agg_ban_g.share_non_fda_flav_bundle[t]), @sprintf("%.10f", agg_ban_g.share_fda_flav_bundle[t]),
                    @sprintf("%.10f", agg_ban_g.mean_addiction[t]), @sprintf("%.10f", agg_ban_g.mean_aflav[t]),
                    @sprintf("%.10f", agg_ban_g.mean_welfare[t]), @sprintf("%.10f", agg_ban_g.mean_q_cig[t]), @sprintf("%.10f", agg_ban_g.mean_q_orig_ecig[t]), @sprintf("%.10f", agg_ban_g.mean_q_flav_ecig[t])]
                println(io, join(row, ","))
            end
        end
    end
    log_msg("TYA simulation results saved to: $sim_tya_path")

    # --- Save Simulation by Latent Type ---
    sim_type_path = joinpath(beta_subdir, "Simulation_by_Type_b$(b_str).csv");
    open(sim_type_path, "w") do io
        header = ["type", "period",
            "sq_outside", "sq_cig", "sq_orig_ecig", "sq_non_fda_flav_ecig", "sq_fda_flav_ecig",
            "sq_orig_bundle", "sq_non_fda_flav_bundle", "sq_fda_flav_bundle",
            "sq_addiction", "sq_aflav", "sq_welfare", "sq_q_cig", "sq_q_orig_ecig", "sq_q_flav_ecig",
            "ban_outside", "ban_cig", "ban_orig_ecig", "ban_non_fda_flav_ecig", "ban_fda_flav_ecig",
            "ban_orig_bundle", "ban_non_fda_flav_bundle", "ban_fda_flav_bundle",
            "ban_addiction", "ban_aflav", "ban_welfare", "ban_q_cig", "ban_q_orig_ecig", "ban_q_flav_ecig"]
        println(io, join(header, ","))
        for (type_label, agg_sq_g, agg_ban_g) in [("type1", agg_sq_type1, agg_ban_type1), ("type2", agg_sq_type2, agg_ban_type2), ("type3", agg_sq_type3, agg_ban_type3)]
            for t in 1:T_sim
                row = [type_label, @sprintf("%d", t),
                    @sprintf("%.10f", agg_sq_g.share_outside[t]), @sprintf("%.10f", agg_sq_g.share_cig[t]),
                    @sprintf("%.10f", agg_sq_g.share_orig_ecig[t]), @sprintf("%.10f", agg_sq_g.share_non_fda_flav_ecig[t]),
                    @sprintf("%.10f", agg_sq_g.share_fda_flav_ecig[t]), @sprintf("%.10f", agg_sq_g.share_orig_bundle[t]),
                    @sprintf("%.10f", agg_sq_g.share_non_fda_flav_bundle[t]), @sprintf("%.10f", agg_sq_g.share_fda_flav_bundle[t]),
                    @sprintf("%.10f", agg_sq_g.mean_addiction[t]), @sprintf("%.10f", agg_sq_g.mean_aflav[t]),
                    @sprintf("%.10f", agg_sq_g.mean_welfare[t]), @sprintf("%.10f", agg_sq_g.mean_q_cig[t]), @sprintf("%.10f", agg_sq_g.mean_q_orig_ecig[t]), @sprintf("%.10f", agg_sq_g.mean_q_flav_ecig[t]),
                    @sprintf("%.10f", agg_ban_g.share_outside[t]), @sprintf("%.10f", agg_ban_g.share_cig[t]),
                    @sprintf("%.10f", agg_ban_g.share_orig_ecig[t]), @sprintf("%.10f", agg_ban_g.share_non_fda_flav_ecig[t]),
                    @sprintf("%.10f", agg_ban_g.share_fda_flav_ecig[t]), @sprintf("%.10f", agg_ban_g.share_orig_bundle[t]),
                    @sprintf("%.10f", agg_ban_g.share_non_fda_flav_bundle[t]), @sprintf("%.10f", agg_ban_g.share_fda_flav_bundle[t]),
                    @sprintf("%.10f", agg_ban_g.mean_addiction[t]), @sprintf("%.10f", agg_ban_g.mean_aflav[t]),
                    @sprintf("%.10f", agg_ban_g.mean_welfare[t]), @sprintf("%.10f", agg_ban_g.mean_q_cig[t]), @sprintf("%.10f", agg_ban_g.mean_q_orig_ecig[t]), @sprintf("%.10f", agg_ban_g.mean_q_flav_ecig[t])]
                println(io, join(row, ","))
            end
        end
    end
    log_msg("Type simulation results saved to: $sim_type_path")

    # --- Save Simulation by TYA Status and Latent Type ---
    sim_tya_type_path = joinpath(beta_subdir, "Simulation_by_TYA_Type_b$(b_str).csv");
    open(sim_tya_type_path, "w") do io
        header = ["type", "period",
            "sq_outside", "sq_cig", "sq_orig_ecig", "sq_non_fda_flav_ecig", "sq_fda_flav_ecig",
            "sq_orig_bundle", "sq_non_fda_flav_bundle", "sq_fda_flav_bundle",
            "sq_addiction", "sq_aflav", "sq_welfare", "sq_q_cig", "sq_q_orig_ecig", "sq_q_flav_ecig",
            "ban_outside", "ban_cig", "ban_orig_ecig", "ban_non_fda_flav_ecig", "ban_fda_flav_ecig",
            "ban_orig_bundle", "ban_non_fda_flav_bundle", "ban_fda_flav_bundle",
            "ban_addiction", "ban_aflav", "ban_welfare", "ban_q_cig", "ban_q_orig_ecig", "ban_q_flav_ecig"]
        println(io, join(header, ","))
        for (type_label, agg_sq_g, agg_ban_g) in [("tya_type1", agg_sq_tya_type1, agg_ban_tya_type1), ("tya_type2", agg_sq_tya_type2, agg_ban_tya_type2), ("tya_type3", agg_sq_tya_type3, agg_ban_tya_type3)]
            for t in 1:T_sim
                row = [type_label, @sprintf("%d", t),
                    @sprintf("%.10f", agg_sq_g.share_outside[t]), @sprintf("%.10f", agg_sq_g.share_cig[t]),
                    @sprintf("%.10f", agg_sq_g.share_orig_ecig[t]), @sprintf("%.10f", agg_sq_g.share_non_fda_flav_ecig[t]),
                    @sprintf("%.10f", agg_sq_g.share_fda_flav_ecig[t]), @sprintf("%.10f", agg_sq_g.share_orig_bundle[t]),
                    @sprintf("%.10f", agg_sq_g.share_non_fda_flav_bundle[t]), @sprintf("%.10f", agg_sq_g.share_fda_flav_bundle[t]),
                    @sprintf("%.10f", agg_sq_g.mean_addiction[t]), @sprintf("%.10f", agg_sq_g.mean_aflav[t]),
                    @sprintf("%.10f", agg_sq_g.mean_welfare[t]), @sprintf("%.10f", agg_sq_g.mean_q_cig[t]), @sprintf("%.10f", agg_sq_g.mean_q_orig_ecig[t]), @sprintf("%.10f", agg_sq_g.mean_q_flav_ecig[t]),
                    @sprintf("%.10f", agg_ban_g.share_outside[t]), @sprintf("%.10f", agg_ban_g.share_cig[t]),
                    @sprintf("%.10f", agg_ban_g.share_orig_ecig[t]), @sprintf("%.10f", agg_ban_g.share_non_fda_flav_ecig[t]),
                    @sprintf("%.10f", agg_ban_g.share_fda_flav_ecig[t]), @sprintf("%.10f", agg_ban_g.share_orig_bundle[t]),
                    @sprintf("%.10f", agg_ban_g.share_non_fda_flav_bundle[t]), @sprintf("%.10f", agg_ban_g.share_fda_flav_bundle[t]),
                    @sprintf("%.10f", agg_ban_g.mean_addiction[t]), @sprintf("%.10f", agg_ban_g.mean_aflav[t]),
                    @sprintf("%.10f", agg_ban_g.mean_welfare[t]), @sprintf("%.10f", agg_ban_g.mean_q_cig[t]), @sprintf("%.10f", agg_ban_g.mean_q_orig_ecig[t]), @sprintf("%.10f", agg_ban_g.mean_q_flav_ecig[t])]
                println(io, join(row, ","))
            end
        end
    end
    log_msg("TYA-by-type simulation results saved to: $sim_tya_type_path")

    # --- Save Extensive Margin Results (main + threshold robustness) ---
    for (thresh, suffix) in [(0.05, "_thresh005"), (0.10, ""), (0.20, "_thresh020")]
        ext_tya_t = aggregate_extensive_margin_by_tya(
            sim_choices_sq, sim_choices_ban, hh_aflav0, cat_idx, T_sim, hh_tya, N_draws;
            aflav_threshold = thresh
        )
        ext_path_t = joinpath(beta_subdir, "Extensive_Margin_by_TYA$(suffix)_b$(b_str).csv")
        open(ext_path_t, "w") do io
            header = ["group", "period", "n_non_users",
                      "sq_ever_initiated", "cf_ever_initiated", "prevention_rate"]
            println(io, join(header, ","))
            for (group_label, df_g) in [("tya", ext_tya_t)]
                for t in 1:T_sim
                    row = [group_label,
                           @sprintf("%d",    t),
                           @sprintf("%d",    df_g.n_non_users[t]),
                           @sprintf("%.10f", df_g.sq_ever_initiated[t]),
                           @sprintf("%.10f", df_g.cf_ever_initiated[t]),
                           @sprintf("%.10f", df_g.prevention_rate[t])]
                    println(io, join(row, ","))
                end
            end
        end
        log_msg("Extensive margin (thresh=$(thresh)) saved to: $ext_path_t")
    end

    # --- Save Extensive Margin by Type Results ---
    for (thresh, suffix) in [(0.05, "_thresh005"), (0.10, ""), (0.20, "_thresh020")]
        ext_type1_t, ext_type2_t, ext_type3_t = aggregate_extensive_margin_by_type_k3(
            sim_choices_sq, sim_choices_ban, hh_aflav0, cat_idx, T_sim,
            hh_posterior_type1, hh_posterior_type2, hh_posterior_type3, N_draws;
            aflav_threshold = thresh
        )
        ext_path_t = joinpath(beta_subdir, "Extensive_Margin_by_Type$(suffix)_b$(b_str).csv")
        open(ext_path_t, "w") do io
            header = ["type", "period", "n_non_users",
                      "sq_ever_initiated", "cf_ever_initiated", "prevention_rate"]
            println(io, join(header, ","))
            for (type_label, df_g) in [("type1", ext_type1_t), ("type2", ext_type2_t), ("type3", ext_type3_t)]
                for t in 1:T_sim
                    row = [type_label,
                           @sprintf("%d",    t),
                           @sprintf("%d",    df_g.n_non_users[t]),
                           @sprintf("%.10f", df_g.sq_ever_initiated[t]),
                           @sprintf("%.10f", df_g.cf_ever_initiated[t]),
                           @sprintf("%.10f", df_g.prevention_rate[t])]
                    println(io, join(row, ","))
                end
            end
        end
        log_msg("Extensive margin by type (thresh=$(thresh)) saved to: $ext_path_t")
    end

    # --- Save Extensive Margin by TYA Status and Latent Type Results ---
    for (thresh, suffix) in [(0.05, "_thresh005"), (0.10, ""), (0.20, "_thresh020")]
        ext_tya_type1_t, ext_tya_type2_t, ext_tya_type3_t = aggregate_extensive_margin_by_tya_type_k3(
            sim_choices_sq, sim_choices_ban, hh_aflav0, cat_idx, T_sim, hh_tya,
            hh_posterior_type1, hh_posterior_type2, hh_posterior_type3, N_draws;
            aflav_threshold = thresh
        )
        ext_path_t = joinpath(beta_subdir, "Extensive_Margin_by_TYA_Type$(suffix)_b$(b_str).csv")
        open(ext_path_t, "w") do io
            header = ["type", "period", "n_non_users",
                      "sq_ever_initiated", "cf_ever_initiated", "prevention_rate"]
            println(io, join(header, ","))
            for (type_label, df_g) in [("tya_type1", ext_tya_type1_t), ("tya_type2", ext_tya_type2_t), ("tya_type3", ext_tya_type3_t)]
                for t in 1:T_sim
                    row = [type_label,
                           @sprintf("%d",    t),
                           @sprintf("%d",    df_g.n_non_users[t]),
                           @sprintf("%.10f", df_g.sq_ever_initiated[t]),
                           @sprintf("%.10f", df_g.cf_ever_initiated[t]),
                           @sprintf("%.10f", df_g.prevention_rate[t])]
                    println(io, join(row, ","))
                end
            end
        end
        log_msg("Extensive margin by TYA-type (thresh=$(thresh)) saved to: $ext_path_t")
    end


    #############################
    # Log Simulation Summary
    #############################

    log_msg("\n--- Forward Simulation Summary (averaged over $T_sim periods) ---")
    log_msg(@sprintf("  %-22s  %12s  %12s  %12s", "Category", "SQ Share", "Ban Share", "Difference"))
    log_msg("  " * repeat("-", 62))

    sim_cat_cols = [:share_outside, :share_cig, :share_orig_ecig, :share_non_fda_flav_ecig, :share_fda_flav_ecig, :share_orig_bundle, :share_non_fda_flav_bundle, :share_fda_flav_bundle]
    for (c, label) in enumerate(cat_labels)
        col = sim_cat_cols[c]
        sq_share  = mean(agg_sq[!, col])
        ban_share = mean(agg_ban[!, col])
        log_msg(@sprintf("  %-22s  %12.6f  %12.6f  %12.6f", label, sq_share, ban_share, ban_share - sq_share))
    end

    log_msg(@sprintf("\n  Mean addiction (SQ):  %.6f", mean(agg_sq.mean_addiction)))
    log_msg(@sprintf("  Mean addiction (Ban): %.6f", mean(agg_ban.mean_addiction)))
    log_msg(@sprintf("  Addiction change:     %.6f", mean(agg_ban.mean_addiction) - mean(agg_sq.mean_addiction)))
    log_msg(@sprintf("\n  Mean welfare (SQ):   %.6f", mean(agg_sq.mean_welfare)))
    log_msg(@sprintf("  Mean welfare (Ban):  %.6f", mean(agg_ban.mean_welfare)))
    log_msg(@sprintf("  Welfare change:      %.6f", mean(agg_ban.mean_welfare) - mean(agg_sq.mean_welfare)))

    probs_1_sq = nothing; probs_2_sq = nothing; probs_3_sq = nothing
    probs_1_ban = nothing; probs_2_ban = nothing; probs_3_ban = nothing
    welfare_1_sq = nothing; welfare_2_sq = nothing; welfare_3_sq = nothing
    welfare_1_ban = nothing; welfare_2_ban = nothing; welfare_3_ban = nothing
    GC.gc()

end  # end BAN_TYPES loop

#############################
# Flavor Tax Counterfactual
#############################

TAX_GRID = [0.50, 2.78]

log_msg("\n\n===================================")
log_msg("Starting flavor tax counterfactual")
log_msg("===================================")
log_msg("Tax levels (per mL): $TAX_GRID")
log_msg(@sprintf("q_ecig_max = %.4f", q_ecig_max))
log_msg(@sprintf("omega_E_est (from theta_b) = %.10f", omega_E_est))

tax_subdir = joinpath(output_dir, "Flavor_Tax")
mkpath(tax_subdir)

for tau in TAX_GRID

    tau_tag = replace(@sprintf("%.2f", tau), "." => "p")
    tau_subdir_name = "Tax_$(tau_tag)"

    log_msg("\n\n###################################")
    log_msg("Flavor tax: \$$(tau) per mL (tag: $tau_tag)")
    log_msg("###################################")

    local beta_subdir = joinpath(tax_subdir, tau_subdir_name)  # β encoded in output_dir
    mkpath(beta_subdir)

    log_msg("\n###################################")
    log_msg("β = $BETA_VAL (tag: $beta_tag), b = $b")
    log_msg("###################################")

    local V_decision_1_sq, vfi_iters_1_sq, vfi_converged_1_sq
    local V_decision_2_sq, vfi_iters_2_sq, vfi_converged_2_sq
    local V_decision_3_sq, vfi_iters_3_sq, vfi_converged_3_sq
    local V_decision_1_tax, vfi_iters_1_tax, vfi_converged_1_tax
    local V_decision_2_tax, vfi_iters_2_tax, vfi_converged_2_tax
    local V_decision_3_tax, vfi_iters_3_tax, vfi_converged_3_tax
    local probs_1_sq, welfare_1_sq, probs_2_sq, welfare_2_sq, probs_3_sq, welfare_3_sq
    local probs_1_tax, welfare_1_tax, probs_2_tax, welfare_2_tax, probs_3_tax, welfare_3_tax
    local hh_posterior_type1, hh_posterior_type2, hh_posterior_type3, mean_w1, mean_w2, mean_w3
    local obs_posterior_type1, obs_posterior_type2, obs_posterior_type3
    local probs_sq, welfare_sq, probs_tax, welfare_tax

    local U_1_tax = copy(U_1); apply_flavor_tax!(U_1_tax, cat_idx, omega_E_est, q_ecig, q_ecig_max, tau)
    local U_2_tax = copy(U_2); apply_flavor_tax!(U_2_tax, cat_idx, omega_E_est, q_ecig, q_ecig_max, tau)
    local U_3_tax = copy(U_3); apply_flavor_tax!(U_3_tax, cat_idx, omega_E_est, q_ecig, q_ecig_max, tau)

    local flav_alts = findall(j -> cat_idx[j] in (3, 4, 6, 7), 1:N_J)
    local median_q_raw = median(q_ecig[flav_alts] .* q_ecig_max)
    local median_q_std = median(q_ecig[flav_alts])
    log_msg(@sprintf("omega_E (beta=%.2f) = %.10f; median flavored alt: q_ecig_raw = %.2f mL, utility shift = %.6f",
        BETA_VAL, omega_E_est, median_q_raw, omega_E_est * tau * median_q_std))

    # --- Solve VFI: Status Quo ---
    log_msg("\n===================================")
    log_msg("Solving VFI: Status Quo (β = $BETA_VAL)")
    log_msg("===================================")

    t_vfi = time();
    task_sq_1 = Threads.@spawn solve_vfi_sophisticated(
        N_J, N_A_f, N_A_s, N_A_flav, N_P, N_Pcomb, BETA_VAL, δ, U_1,
        af_lower, af_upper, af_weight,
        as_lower, as_upper, as_weight,
        aflav_lower, aflav_upper, aflav_weight,
        p_cig_lo, p_cig_hi, p_cig_w, p_ecig_lo, p_ecig_hi, p_ecig_w;
        V_init = nothing, verbose = true
    )
    task_sq_2 = Threads.@spawn solve_vfi_sophisticated(
        N_J, N_A_f, N_A_s, N_A_flav, N_P, N_Pcomb, BETA_VAL, δ, U_2,
        af_lower, af_upper, af_weight,
        as_lower, as_upper, as_weight,
        aflav_lower, aflav_upper, aflav_weight,
        p_cig_lo, p_cig_hi, p_cig_w, p_ecig_lo, p_ecig_hi, p_ecig_w;
        V_init = nothing, verbose = true
    )
    task_sq_3 = Threads.@spawn solve_vfi_sophisticated(
        N_J, N_A_f, N_A_s, N_A_flav, N_P, N_Pcomb, BETA_VAL, δ, U_3,
        af_lower, af_upper, af_weight,
        as_lower, as_upper, as_weight,
        aflav_lower, aflav_upper, aflav_weight,
        p_cig_lo, p_cig_hi, p_cig_w, p_ecig_lo, p_ecig_hi, p_ecig_w;
        V_init = nothing, verbose = true
    )
    V_now_1_sq, V_decision_1_sq, vfi_iters_1_sq, vfi_converged_1_sq = fetch(task_sq_1)
    V_now_2_sq, V_decision_2_sq, vfi_iters_2_sq, vfi_converged_2_sq = fetch(task_sq_2)
    V_now_3_sq, V_decision_3_sq, vfi_iters_3_sq, vfi_converged_3_sq = fetch(task_sq_3)
    log_msg("Status quo VFI: type1=$(vfi_iters_1_sq) iters ($(vfi_converged_1_sq)), type2=$(vfi_iters_2_sq) iters ($(vfi_converged_2_sq)), type3=$(vfi_iters_3_sq) iters ($(vfi_converged_3_sq)), $(round(time() - t_vfi, digits=1))s")

    # --- Solve VFI: Flavor Tax ---
    log_msg("\n===================================")
    log_msg("Solving VFI: Flavor Tax (β = $BETA_VAL)")
    log_msg("===================================")

    t_vfi = time();
    task_tax_1 = Threads.@spawn solve_vfi_sophisticated(
        N_J, N_A_f, N_A_s, N_A_flav, N_P, N_Pcomb, BETA_VAL, δ, U_1_tax,
        af_lower, af_upper, af_weight,
        as_lower, as_upper, as_weight,
        aflav_lower, aflav_upper, aflav_weight,
        p_cig_lo, p_cig_hi, p_cig_w, p_ecig_lo, p_ecig_hi, p_ecig_w;
        V_init = nothing, verbose = true
    )
    task_tax_2 = Threads.@spawn solve_vfi_sophisticated(
        N_J, N_A_f, N_A_s, N_A_flav, N_P, N_Pcomb, BETA_VAL, δ, U_2_tax,
        af_lower, af_upper, af_weight,
        as_lower, as_upper, as_weight,
        aflav_lower, aflav_upper, aflav_weight,
        p_cig_lo, p_cig_hi, p_cig_w, p_ecig_lo, p_ecig_hi, p_ecig_w;
        V_init = nothing, verbose = true
    )
    task_tax_3 = Threads.@spawn solve_vfi_sophisticated(
        N_J, N_A_f, N_A_s, N_A_flav, N_P, N_Pcomb, BETA_VAL, δ, U_3_tax,
        af_lower, af_upper, af_weight,
        as_lower, as_upper, as_weight,
        aflav_lower, aflav_upper, aflav_weight,
        p_cig_lo, p_cig_hi, p_cig_w, p_ecig_lo, p_ecig_hi, p_ecig_w;
        V_init = nothing, verbose = true
    )
    V_now_1_tax, V_decision_1_tax, vfi_iters_1_tax, vfi_converged_1_tax = fetch(task_tax_1)
    V_now_2_tax, V_decision_2_tax, vfi_iters_2_tax, vfi_converged_2_tax = fetch(task_tax_2)
    V_now_3_tax, V_decision_3_tax, vfi_iters_3_tax, vfi_converged_3_tax = fetch(task_tax_3)
    log_msg("Tax VFI: type1=$(vfi_iters_1_tax) iters ($(vfi_converged_1_tax)), type2=$(vfi_iters_2_tax) iters ($(vfi_converged_2_tax)), type3=$(vfi_iters_3_tax) iters ($(vfi_converged_3_tax)), $(round(time() - t_vfi, digits=1))s")

    # --- Compute posterior type weights ---
    log_msg("\n===================================")
    log_msg("Computing posterior type weights...")
    log_msg("===================================")

    t_post = time();
    probs_1_sq, welfare_1_sq = compute_pointwise_outcomes(
        V_decision_1_sq, tya_state, af_continuous, as_continuous, aflav_continuous, p_continuous, N_J, N_P, A_f, A_s, A_flav, P
    )
    probs_2_sq, welfare_2_sq = compute_pointwise_outcomes(
        V_decision_2_sq, tya_state, af_continuous, as_continuous, aflav_continuous, p_continuous, N_J, N_P, A_f, A_s, A_flav, P
    )
    probs_3_sq, welfare_3_sq = compute_pointwise_outcomes(
        V_decision_3_sq, tya_state, af_continuous, as_continuous, aflav_continuous, p_continuous, N_J, N_P, A_f, A_s, A_flav, P
    )
    probs_1_tax, welfare_1_tax = compute_pointwise_outcomes(
        V_decision_1_tax, tya_state, af_continuous, as_continuous, aflav_continuous, p_continuous, N_J, N_P, A_f, A_s, A_flav, P
    )
    probs_2_tax, welfare_2_tax = compute_pointwise_outcomes(
        V_decision_2_tax, tya_state, af_continuous, as_continuous, aflav_continuous, p_continuous, N_J, N_P, A_f, A_s, A_flav, P
    )
    probs_3_tax, welfare_3_tax = compute_pointwise_outcomes(
        V_decision_3_tax, tya_state, af_continuous, as_continuous, aflav_continuous, p_continuous, N_J, N_P, A_f, A_s, A_flav, P
    )

    hh_posterior_type1 = Vector{Float64}(undef, N_HH)
    hh_posterior_type2 = Vector{Float64}(undef, N_HH)
    hh_posterior_type3 = Vector{Float64}(undef, N_HH)

    for h in 1:N_HH
        start_idx, stop_idx = hh_ranges[h]
        η_2_h = π_0_2 + π_TYA_2 * tya_share_hh[h]
        η_3_h = π_0_3 + π_TYA_3 * tya_share_hh[h]
        log_sum_exp_h = log(1.0 + exp(η_2_h) + exp(η_3_h))
        log_pi_1_h = -log_sum_exp_h
        log_pi_2_h = η_2_h - log_sum_exp_h
        log_pi_3_h = η_3_h - log_sum_exp_h
        log_ll_1 = 0.0; log_ll_2 = 0.0; log_ll_3 = 0.0
        for i in start_idx:stop_idx
            log_ll_1 += log(max(probs_1_sq[i, y[i]], 1e-300))
            log_ll_2 += log(max(probs_2_sq[i, y[i]], 1e-300))
            log_ll_3 += log(max(probs_3_sq[i, y[i]], 1e-300))
        end
        a_term = log_pi_1_h + log_ll_1
        b_term = log_pi_2_h + log_ll_2
        c_term = log_pi_3_h + log_ll_3
        log_max = max(a_term, b_term, c_term)
        log_denom = log_max + log(exp(a_term - log_max) + exp(b_term - log_max) + exp(c_term - log_max))
        hh_posterior_type1[h] = exp(a_term - log_denom)
        hh_posterior_type2[h] = exp(b_term - log_denom)
        hh_posterior_type3[h] = exp(c_term - log_denom)
    end

    mean_w1 = mean(hh_posterior_type1); mean_w2 = mean(hh_posterior_type2); mean_w3 = mean(hh_posterior_type3)
    log_msg("Posterior type weights computed in $(round(time() - t_post, digits=1))s")
    log_msg(@sprintf("  Mean P(type=1) = %.4f, Mean P(type=2) = %.4f, Mean P(type=3) = %.4f", mean_w1, mean_w2, mean_w3))

    obs_posterior_type1 = Vector{Float64}(undef, N_obs)
    obs_posterior_type2 = Vector{Float64}(undef, N_obs)
    obs_posterior_type3 = Vector{Float64}(undef, N_obs)
    for h in 1:N_HH
        start_idx, stop_idx = hh_ranges[h]
        for i in start_idx:stop_idx
            obs_posterior_type1[i] = hh_posterior_type1[h]
            obs_posterior_type2[i] = hh_posterior_type2[h]
            obs_posterior_type3[i] = hh_posterior_type3[h]
        end
    end

    probs_sq    = obs_posterior_type1 .* probs_1_sq   .+ obs_posterior_type2 .* probs_2_sq   .+ obs_posterior_type3 .* probs_3_sq
    welfare_sq  = obs_posterior_type1 .* welfare_1_sq .+ obs_posterior_type2 .* welfare_2_sq .+ obs_posterior_type3 .* welfare_3_sq
    probs_tax   = obs_posterior_type1 .* probs_1_tax  .+ obs_posterior_type2 .* probs_2_tax  .+ obs_posterior_type3 .* probs_3_tax
    welfare_tax = obs_posterior_type1 .* welfare_1_tax .+ obs_posterior_type2 .* welfare_2_tax .+ obs_posterior_type3 .* welfare_3_tax

    t_pw = time();
    log_msg("\nPointwise summary (means across all observations):")
    log_msg(@sprintf("  %-22s  %12s  %12s  %12s", "Category", "SQ Share", "Tax Share", "Difference"))
    log_msg("  " * repeat("-", 62))
    for (c, label) in enumerate(cat_labels)
        cat_val = c - 1
        alt_indices = findall(j -> cat_idx[j] == cat_val, 1:N_J)
        sq_share  = mean(sum(probs_sq[:, alt_indices], dims=2))
        tax_share = mean(sum(probs_tax[:, alt_indices], dims=2))
        log_msg(@sprintf("  %-22s  %12.6f  %12.6f  %12.6f", label, sq_share, tax_share, tax_share - sq_share))
    end
    welfare_diff = welfare_tax .- welfare_sq
    log_msg(@sprintf("\n  Mean welfare SQ:   %.6f", mean(welfare_sq)))
    log_msg(@sprintf("  Mean welfare Tax:  %.6f", mean(welfare_tax)))
    log_msg(@sprintf("  Mean welfare loss: %.6f", mean(welfare_diff)))

    flav_revenue = 0.0
    for i in 1:N_obs
        for j in flav_alts
            flav_revenue += probs_tax[i, j] * q_ecig[j] * q_ecig_max * tau
        end
    end
    log_msg(@sprintf("  Mean tax revenue per HH-month: \$%.4f", flav_revenue / N_obs))

    # --- Forward Simulation ---
    log_msg("\n===================================")
    log_msg("Forward simulation (tau = \$$tau, beta = $BETA_VAL, b = $b)...")
    log_msg("===================================")
    log_msg("T_sim = $T_sim, N_draws = $N_draws, N_HH = $N_HH")

    sim_choices_sq      = Array{Int}(undef, N_HH, T_sim, N_draws)
    sim_addiction_f_sq  = Array{Float64}(undef, N_HH, T_sim, N_draws)
    sim_addiction_s_sq  = Array{Float64}(undef, N_HH, T_sim, N_draws)
    sim_aflav_sq        = Array{Float64}(undef, N_HH, T_sim, N_draws)
    sim_welfare_sq_arr  = Array{Float64}(undef, N_HH, T_sim, N_draws)
    sim_q_cig_sq        = Array{Float64}(undef, N_HH, T_sim, N_draws)
    sim_q_orig_ecig_sq  = Array{Float64}(undef, N_HH, T_sim, N_draws)
    sim_q_flav_ecig_sq  = Array{Float64}(undef, N_HH, T_sim, N_draws)

    sim_choices_tax     = Array{Int}(undef, N_HH, T_sim, N_draws)
    sim_addiction_f_tax = Array{Float64}(undef, N_HH, T_sim, N_draws)
    sim_addiction_s_tax = Array{Float64}(undef, N_HH, T_sim, N_draws)
    sim_aflav_tax       = Array{Float64}(undef, N_HH, T_sim, N_draws)
    sim_welfare_tax_arr = Array{Float64}(undef, N_HH, T_sim, N_draws)
    sim_q_cig_tax       = Array{Float64}(undef, N_HH, T_sim, N_draws)
    sim_q_orig_ecig_tax = Array{Float64}(undef, N_HH, T_sim, N_draws)
    sim_q_flav_ecig_tax = Array{Float64}(undef, N_HH, T_sim, N_draws)

    local P_cig  = P[:, 1]
    local P_ecig = P[:, 2]

    t_sim_fwd = time();

    Threads.@threads for h in 1:N_HH
        tya_idx_h = hh_tya[h]
        w1 = hh_posterior_type1[h]; w2 = hh_posterior_type2[h]
        for d in 1:N_draws
            u_type = crn_type[h, d]
            type_k = u_type <= w1 ? 1 : (u_type <= w1 + w2 ? 2 : 3)
            V_sq  = type_k == 1 ? V_decision_1_sq  : (type_k == 2 ? V_decision_2_sq  : V_decision_3_sq)
            V_tax = type_k == 1 ? V_decision_1_tax : (type_k == 2 ? V_decision_2_tax : V_decision_3_tax)
            V_now_sq  = type_k == 1 ? V_now_1_sq  : (type_k == 2 ? V_now_2_sq  : V_now_3_sq)
            V_now_tax = type_k == 1 ? V_now_1_tax : (type_k == 2 ? V_now_2_tax : V_now_3_tax)
            a_f_sq = hh_af0[h]; a_s_sq = hh_as0[h]; a_flav_sq = hh_aflav0[h]
            p_cig_sq = hh_p0[h, 1]; p_ecig_sq = hh_p0[h, 2]
            a_f_tax = hh_af0[h]; a_s_tax = hh_as0[h]; a_flav_tax = hh_aflav0[h]
            p_cig_tax = hh_p0[h, 1]; p_ecig_tax = hh_p0[h, 2]
            tya_sq = tya_idx_h; tya_tax = tya_idx_h
            for t in 1:T_sim
                v_interp_sq = interpolate_v_choice(V_sq, tya_sq, a_f_sq, a_s_sq, a_flav_sq, p_cig_sq, p_ecig_sq, N_J, N_P, A_f, A_s, A_flav, P)
                replace!(v_interp_sq, NaN => -Inf)
                sim_welfare_sq_arr[h, t, d] = interpolate_v_now(V_now_sq, tya_sq, a_f_sq, a_s_sq, a_flav_sq, p_cig_sq, p_ecig_sq, N_P, A_f, A_s, A_flav, P)
                v_max_sq = maximum(v_interp_sq)
                exp_v_sq = exp.(v_interp_sq .- v_max_sq)
                probs_h_sq = exp_v_sq ./ sum(exp_v_sq)
                u_draw = crn_choice[h, t, d]
                j_sq = 1; cum_prob = probs_h_sq[1]
                while cum_prob < u_draw && j_sq < N_J; j_sq += 1; cum_prob += probs_h_sq[j_sq]; end
                sim_choices_sq[h, t, d] = j_sq
                sim_q_cig_sq[h, t, d]       = q_cig_raw[j_sq]
                sim_q_orig_ecig_sq[h, t, d] = q_orig_ecig_raw[j_sq]
                sim_q_flav_ecig_sq[h, t, d] = q_flav_ecig_raw[j_sq]
                a_f_sq = clamp(addiction_evolution(ψ_2, a_f_sq, n[j_sq]), A_f[1], A_f[end])
                a_s_sq = clamp(addiction_evolution(ψ_1, a_s_sq, n[j_sq]), A_s[1], A_s[end])
                a_flav_sq = clamp(addiction_evolution(PSI_3, a_flav_sq, n_flav[j_sq]), A_flav[1], A_flav[end])
                sim_addiction_f_sq[h, t, d] = a_f_sq; sim_addiction_s_sq[h, t, d] = a_s_sq; sim_aflav_sq[h, t, d] = a_flav_sq
                ε_sq = L_chol * [crn_price[h, t, d, 1], crn_price[h, t, d, 2]]
                p_cig_sq  = clamp(φ_0[1] + φ_1[1] * p_cig_sq  + ε_sq[1], P_cig[1], P_cig[end])
                p_ecig_sq = clamp(φ_0[2] + φ_1[2] * p_ecig_sq + ε_sq[2], P_ecig[1], P_ecig[end])
                v_interp_tax = interpolate_v_choice(V_tax, tya_tax, a_f_tax, a_s_tax, a_flav_tax, p_cig_tax, p_ecig_tax, N_J, N_P, A_f, A_s, A_flav, P)
                replace!(v_interp_tax, NaN => -Inf)
                sim_welfare_tax_arr[h, t, d] = interpolate_v_now(V_now_tax, tya_tax, a_f_tax, a_s_tax, a_flav_tax, p_cig_tax, p_ecig_tax, N_P, A_f, A_s, A_flav, P)
                v_max_tax = maximum(v_interp_tax)
                exp_v_tax = exp.(v_interp_tax .- v_max_tax)
                probs_h_tax = exp_v_tax ./ sum(exp_v_tax)
                j_tax = 1; cum_prob_tax = probs_h_tax[1]
                while cum_prob_tax < u_draw && j_tax < N_J; j_tax += 1; cum_prob_tax += probs_h_tax[j_tax]; end
                sim_choices_tax[h, t, d] = j_tax
                sim_q_cig_tax[h, t, d]       = q_cig_raw[j_tax]
                sim_q_orig_ecig_tax[h, t, d] = q_orig_ecig_raw[j_tax]
                sim_q_flav_ecig_tax[h, t, d] = q_flav_ecig_raw[j_tax]
                a_f_tax = clamp(addiction_evolution(ψ_2, a_f_tax, n[j_tax]), A_f[1], A_f[end])
                a_s_tax = clamp(addiction_evolution(ψ_1, a_s_tax, n[j_tax]), A_s[1], A_s[end])
                a_flav_tax = clamp(addiction_evolution(PSI_3, a_flav_tax, n_flav[j_tax]), A_flav[1], A_flav[end])
                sim_addiction_f_tax[h, t, d] = a_f_tax; sim_addiction_s_tax[h, t, d] = a_s_tax; sim_aflav_tax[h, t, d] = a_flav_tax
                ε_tax = L_chol * [crn_price[h, t, d, 1], crn_price[h, t, d, 2]]
                p_cig_tax  = clamp(φ_0[1] + φ_1[1] * p_cig_tax  + ε_tax[1], P_cig[1], P_cig[end])
                p_ecig_tax = clamp(φ_0[2] + φ_1[2] * p_ecig_tax + ε_tax[2], P_ecig[1], P_ecig[end])
            end
        end
    end

    sim_fwd_elapsed = time() - t_sim_fwd;
    log_msg("Forward simulation complete in $(round(sim_fwd_elapsed, digits=1))s")

    sim_addiction_sq  = (sim_addiction_f_sq  .+ sim_addiction_s_sq)  ./ 2.0
    sim_addiction_tax = (sim_addiction_f_tax .+ sim_addiction_s_tax) ./ 2.0

    agg_sq  = aggregate_simulation(sim_choices_sq,  sim_addiction_sq,  sim_aflav_sq,  sim_welfare_sq_arr,  sim_q_cig_sq, sim_q_orig_ecig_sq, sim_q_flav_ecig_sq, cat_idx, N_J, T_sim)
    agg_tax = aggregate_simulation(sim_choices_tax, sim_addiction_tax, sim_aflav_tax, sim_welfare_tax_arr, sim_q_cig_tax, sim_q_orig_ecig_tax, sim_q_flav_ecig_tax, cat_idx, N_J, T_sim)

    agg_sq_tya = aggregate_simulation_by_tya(
        sim_choices_sq, sim_addiction_sq, sim_aflav_sq, sim_welfare_sq_arr, sim_q_cig_sq, sim_q_orig_ecig_sq, sim_q_flav_ecig_sq, cat_idx, N_J, T_sim, hh_tya, N_draws)
    agg_tax_tya = aggregate_simulation_by_tya(
        sim_choices_tax, sim_addiction_tax, sim_aflav_tax, sim_welfare_tax_arr, sim_q_cig_tax, sim_q_orig_ecig_tax, sim_q_flav_ecig_tax, cat_idx, N_J, T_sim, hh_tya, N_draws)

    agg_sq_type1, agg_sq_type2, agg_sq_type3 = aggregate_simulation_by_type_k3(
        sim_choices_sq,  sim_addiction_sq,  sim_aflav_sq,  sim_welfare_sq_arr,  sim_q_cig_sq, sim_q_orig_ecig_sq, sim_q_flav_ecig_sq, cat_idx, N_J, T_sim, hh_posterior_type1, hh_posterior_type2, hh_posterior_type3, N_draws)
    agg_tax_type1, agg_tax_type2, agg_tax_type3 = aggregate_simulation_by_type_k3(
        sim_choices_tax, sim_addiction_tax, sim_aflav_tax, sim_welfare_tax_arr, sim_q_cig_tax, sim_q_orig_ecig_tax, sim_q_flav_ecig_tax, cat_idx, N_J, T_sim, hh_posterior_type1, hh_posterior_type2, hh_posterior_type3, N_draws)

    agg_sq_tya_type1, agg_sq_tya_type2, agg_sq_tya_type3 = aggregate_simulation_by_tya_type_k3(
        sim_choices_sq,  sim_addiction_sq,  sim_aflav_sq,  sim_welfare_sq_arr,  sim_q_cig_sq, sim_q_orig_ecig_sq, sim_q_flav_ecig_sq, cat_idx, N_J, T_sim, hh_tya, hh_posterior_type1, hh_posterior_type2, hh_posterior_type3, N_draws)
    agg_tax_tya_type1, agg_tax_tya_type2, agg_tax_tya_type3 = aggregate_simulation_by_tya_type_k3(
        sim_choices_tax, sim_addiction_tax, sim_aflav_tax, sim_welfare_tax_arr, sim_q_cig_tax, sim_q_orig_ecig_tax, sim_q_flav_ecig_tax, cat_idx, N_J, T_sim, hh_tya, hh_posterior_type1, hh_posterior_type2, hh_posterior_type3, N_draws)

    sim_results_path = joinpath(beta_subdir, "Simulation_Overall_b$(b_str).csv")
    open(sim_results_path, "w") do io
        header = ["period",
            "sq_outside", "sq_cig", "sq_orig_ecig", "sq_non_fda_flav_ecig", "sq_fda_flav_ecig",
            "sq_orig_bundle", "sq_non_fda_flav_bundle", "sq_fda_flav_bundle",
            "sq_addiction", "sq_aflav", "sq_welfare", "sq_q_cig", "sq_q_orig_ecig", "sq_q_flav_ecig",
            "tax_outside", "tax_cig", "tax_orig_ecig", "tax_non_fda_flav_ecig", "tax_fda_flav_ecig",
            "tax_orig_bundle", "tax_non_fda_flav_bundle", "tax_fda_flav_bundle",
            "tax_addiction", "tax_aflav", "tax_welfare", "tax_q_cig", "tax_q_orig_ecig", "tax_q_flav_ecig"]
        println(io, join(header, ","))
        for t in 1:T_sim
            row = [@sprintf("%d", t),
                @sprintf("%.10f", agg_sq.share_outside[t]), @sprintf("%.10f", agg_sq.share_cig[t]),
                @sprintf("%.10f", agg_sq.share_orig_ecig[t]), @sprintf("%.10f", agg_sq.share_non_fda_flav_ecig[t]),
                @sprintf("%.10f", agg_sq.share_fda_flav_ecig[t]), @sprintf("%.10f", agg_sq.share_orig_bundle[t]),
                @sprintf("%.10f", agg_sq.share_non_fda_flav_bundle[t]), @sprintf("%.10f", agg_sq.share_fda_flav_bundle[t]),
                @sprintf("%.10f", agg_sq.mean_addiction[t]), @sprintf("%.10f", agg_sq.mean_aflav[t]),
                @sprintf("%.10f", agg_sq.mean_welfare[t]), @sprintf("%.10f", agg_sq.mean_q_cig[t]), @sprintf("%.10f", agg_sq.mean_q_orig_ecig[t]), @sprintf("%.10f", agg_sq.mean_q_flav_ecig[t]),
                @sprintf("%.10f", agg_tax.share_outside[t]), @sprintf("%.10f", agg_tax.share_cig[t]),
                @sprintf("%.10f", agg_tax.share_orig_ecig[t]), @sprintf("%.10f", agg_tax.share_non_fda_flav_ecig[t]),
                @sprintf("%.10f", agg_tax.share_fda_flav_ecig[t]), @sprintf("%.10f", agg_tax.share_orig_bundle[t]),
                @sprintf("%.10f", agg_tax.share_non_fda_flav_bundle[t]), @sprintf("%.10f", agg_tax.share_fda_flav_bundle[t]),
                @sprintf("%.10f", agg_tax.mean_addiction[t]), @sprintf("%.10f", agg_tax.mean_aflav[t]),
                @sprintf("%.10f", agg_tax.mean_welfare[t]), @sprintf("%.10f", agg_tax.mean_q_cig[t]), @sprintf("%.10f", agg_tax.mean_q_orig_ecig[t]), @sprintf("%.10f", agg_tax.mean_q_flav_ecig[t])]
            println(io, join(row, ","))
        end
    end
    log_msg("Overall simulation results saved to: $sim_results_path")

    sim_tya_path = joinpath(beta_subdir, "Simulation_by_TYA_b$(b_str).csv")
    open(sim_tya_path, "w") do io
        header = ["group", "period",
            "sq_outside", "sq_cig", "sq_orig_ecig", "sq_non_fda_flav_ecig", "sq_fda_flav_ecig",
            "sq_orig_bundle", "sq_non_fda_flav_bundle", "sq_fda_flav_bundle",
            "sq_addiction", "sq_aflav", "sq_welfare", "sq_q_cig", "sq_q_orig_ecig", "sq_q_flav_ecig",
            "tax_outside", "tax_cig", "tax_orig_ecig", "tax_non_fda_flav_ecig", "tax_fda_flav_ecig",
            "tax_orig_bundle", "tax_non_fda_flav_bundle", "tax_fda_flav_bundle",
            "tax_addiction", "tax_aflav", "tax_welfare", "tax_q_cig", "tax_q_orig_ecig", "tax_q_flav_ecig"]
        println(io, join(header, ","))
        for (group_label, agg_sq_g, agg_tax_g) in [("tya", agg_sq_tya, agg_tax_tya)]
            for t in 1:T_sim
                row = [group_label, @sprintf("%d", t),
                    @sprintf("%.10f", agg_sq_g.share_outside[t]), @sprintf("%.10f", agg_sq_g.share_cig[t]),
                    @sprintf("%.10f", agg_sq_g.share_orig_ecig[t]), @sprintf("%.10f", agg_sq_g.share_non_fda_flav_ecig[t]),
                    @sprintf("%.10f", agg_sq_g.share_fda_flav_ecig[t]), @sprintf("%.10f", agg_sq_g.share_orig_bundle[t]),
                    @sprintf("%.10f", agg_sq_g.share_non_fda_flav_bundle[t]), @sprintf("%.10f", agg_sq_g.share_fda_flav_bundle[t]),
                    @sprintf("%.10f", agg_sq_g.mean_addiction[t]), @sprintf("%.10f", agg_sq_g.mean_aflav[t]),
                    @sprintf("%.10f", agg_sq_g.mean_welfare[t]), @sprintf("%.10f", agg_sq_g.mean_q_cig[t]), @sprintf("%.10f", agg_sq_g.mean_q_orig_ecig[t]), @sprintf("%.10f", agg_sq_g.mean_q_flav_ecig[t]),
                    @sprintf("%.10f", agg_tax_g.share_outside[t]), @sprintf("%.10f", agg_tax_g.share_cig[t]),
                    @sprintf("%.10f", agg_tax_g.share_orig_ecig[t]), @sprintf("%.10f", agg_tax_g.share_non_fda_flav_ecig[t]),
                    @sprintf("%.10f", agg_tax_g.share_fda_flav_ecig[t]), @sprintf("%.10f", agg_tax_g.share_orig_bundle[t]),
                    @sprintf("%.10f", agg_tax_g.share_non_fda_flav_bundle[t]), @sprintf("%.10f", agg_tax_g.share_fda_flav_bundle[t]),
                    @sprintf("%.10f", agg_tax_g.mean_addiction[t]), @sprintf("%.10f", agg_tax_g.mean_aflav[t]),
                    @sprintf("%.10f", agg_tax_g.mean_welfare[t]), @sprintf("%.10f", agg_tax_g.mean_q_cig[t]), @sprintf("%.10f", agg_tax_g.mean_q_orig_ecig[t]), @sprintf("%.10f", agg_tax_g.mean_q_flav_ecig[t])]
                println(io, join(row, ","))
            end
        end
    end
    log_msg("TYA simulation results saved to: $sim_tya_path")

    sim_type_path = joinpath(beta_subdir, "Simulation_by_Type_b$(b_str).csv")
    open(sim_type_path, "w") do io
        header = ["type", "period",
            "sq_outside", "sq_cig", "sq_orig_ecig", "sq_non_fda_flav_ecig", "sq_fda_flav_ecig",
            "sq_orig_bundle", "sq_non_fda_flav_bundle", "sq_fda_flav_bundle",
            "sq_addiction", "sq_aflav", "sq_welfare", "sq_q_cig", "sq_q_orig_ecig", "sq_q_flav_ecig",
            "tax_outside", "tax_cig", "tax_orig_ecig", "tax_non_fda_flav_ecig", "tax_fda_flav_ecig",
            "tax_orig_bundle", "tax_non_fda_flav_bundle", "tax_fda_flav_bundle",
            "tax_addiction", "tax_aflav", "tax_welfare", "tax_q_cig", "tax_q_orig_ecig", "tax_q_flav_ecig"]
        println(io, join(header, ","))
        for (type_label, agg_sq_g, agg_tax_g) in [("type1", agg_sq_type1, agg_tax_type1), ("type2", agg_sq_type2, agg_tax_type2), ("type3", agg_sq_type3, agg_tax_type3)]
            for t in 1:T_sim
                row = [type_label, @sprintf("%d", t),
                    @sprintf("%.10f", agg_sq_g.share_outside[t]), @sprintf("%.10f", agg_sq_g.share_cig[t]),
                    @sprintf("%.10f", agg_sq_g.share_orig_ecig[t]), @sprintf("%.10f", agg_sq_g.share_non_fda_flav_ecig[t]),
                    @sprintf("%.10f", agg_sq_g.share_fda_flav_ecig[t]), @sprintf("%.10f", agg_sq_g.share_orig_bundle[t]),
                    @sprintf("%.10f", agg_sq_g.share_non_fda_flav_bundle[t]), @sprintf("%.10f", agg_sq_g.share_fda_flav_bundle[t]),
                    @sprintf("%.10f", agg_sq_g.mean_addiction[t]), @sprintf("%.10f", agg_sq_g.mean_aflav[t]),
                    @sprintf("%.10f", agg_sq_g.mean_welfare[t]), @sprintf("%.10f", agg_sq_g.mean_q_cig[t]), @sprintf("%.10f", agg_sq_g.mean_q_orig_ecig[t]), @sprintf("%.10f", agg_sq_g.mean_q_flav_ecig[t]),
                    @sprintf("%.10f", agg_tax_g.share_outside[t]), @sprintf("%.10f", agg_tax_g.share_cig[t]),
                    @sprintf("%.10f", agg_tax_g.share_orig_ecig[t]), @sprintf("%.10f", agg_tax_g.share_non_fda_flav_ecig[t]),
                    @sprintf("%.10f", agg_tax_g.share_fda_flav_ecig[t]), @sprintf("%.10f", agg_tax_g.share_orig_bundle[t]),
                    @sprintf("%.10f", agg_tax_g.share_non_fda_flav_bundle[t]), @sprintf("%.10f", agg_tax_g.share_fda_flav_bundle[t]),
                    @sprintf("%.10f", agg_tax_g.mean_addiction[t]), @sprintf("%.10f", agg_tax_g.mean_aflav[t]),
                    @sprintf("%.10f", agg_tax_g.mean_welfare[t]), @sprintf("%.10f", agg_tax_g.mean_q_cig[t]), @sprintf("%.10f", agg_tax_g.mean_q_orig_ecig[t]), @sprintf("%.10f", agg_tax_g.mean_q_flav_ecig[t])]
                println(io, join(row, ","))
            end
        end
    end
    log_msg("Type simulation results saved to: $sim_type_path")

    sim_tya_type_path = joinpath(beta_subdir, "Simulation_by_TYA_Type_b$(b_str).csv")
    open(sim_tya_type_path, "w") do io
        header = ["type", "period",
            "sq_outside", "sq_cig", "sq_orig_ecig", "sq_non_fda_flav_ecig", "sq_fda_flav_ecig",
            "sq_orig_bundle", "sq_non_fda_flav_bundle", "sq_fda_flav_bundle",
            "sq_addiction", "sq_aflav", "sq_welfare", "sq_q_cig", "sq_q_orig_ecig", "sq_q_flav_ecig",
            "tax_outside", "tax_cig", "tax_orig_ecig", "tax_non_fda_flav_ecig", "tax_fda_flav_ecig",
            "tax_orig_bundle", "tax_non_fda_flav_bundle", "tax_fda_flav_bundle",
            "tax_addiction", "tax_aflav", "tax_welfare", "tax_q_cig", "tax_q_orig_ecig", "tax_q_flav_ecig"]
        println(io, join(header, ","))
        for (type_label, agg_sq_g, agg_tax_g) in [("tya_type1", agg_sq_tya_type1, agg_tax_tya_type1), ("tya_type2", agg_sq_tya_type2, agg_tax_tya_type2), ("tya_type3", agg_sq_tya_type3, agg_tax_tya_type3)]
            for t in 1:T_sim
                row = [type_label, @sprintf("%d", t),
                    @sprintf("%.10f", agg_sq_g.share_outside[t]), @sprintf("%.10f", agg_sq_g.share_cig[t]),
                    @sprintf("%.10f", agg_sq_g.share_orig_ecig[t]), @sprintf("%.10f", agg_sq_g.share_non_fda_flav_ecig[t]),
                    @sprintf("%.10f", agg_sq_g.share_fda_flav_ecig[t]), @sprintf("%.10f", agg_sq_g.share_orig_bundle[t]),
                    @sprintf("%.10f", agg_sq_g.share_non_fda_flav_bundle[t]), @sprintf("%.10f", agg_sq_g.share_fda_flav_bundle[t]),
                    @sprintf("%.10f", agg_sq_g.mean_addiction[t]), @sprintf("%.10f", agg_sq_g.mean_aflav[t]),
                    @sprintf("%.10f", agg_sq_g.mean_welfare[t]), @sprintf("%.10f", agg_sq_g.mean_q_cig[t]), @sprintf("%.10f", agg_sq_g.mean_q_orig_ecig[t]), @sprintf("%.10f", agg_sq_g.mean_q_flav_ecig[t]),
                    @sprintf("%.10f", agg_tax_g.share_outside[t]), @sprintf("%.10f", agg_tax_g.share_cig[t]),
                    @sprintf("%.10f", agg_tax_g.share_orig_ecig[t]), @sprintf("%.10f", agg_tax_g.share_non_fda_flav_ecig[t]),
                    @sprintf("%.10f", agg_tax_g.share_fda_flav_ecig[t]), @sprintf("%.10f", agg_tax_g.share_orig_bundle[t]),
                    @sprintf("%.10f", agg_tax_g.share_non_fda_flav_bundle[t]), @sprintf("%.10f", agg_tax_g.share_fda_flav_bundle[t]),
                    @sprintf("%.10f", agg_tax_g.mean_addiction[t]), @sprintf("%.10f", agg_tax_g.mean_aflav[t]),
                    @sprintf("%.10f", agg_tax_g.mean_welfare[t]), @sprintf("%.10f", agg_tax_g.mean_q_cig[t]), @sprintf("%.10f", agg_tax_g.mean_q_orig_ecig[t]), @sprintf("%.10f", agg_tax_g.mean_q_flav_ecig[t])]
                println(io, join(row, ","))
            end
        end
    end
    log_msg("TYA-by-type simulation results saved to: $sim_tya_type_path")

    for (thresh, suffix) in [(0.05, "_thresh005"), (0.10, ""), (0.20, "_thresh020")]
        ext_tya_t = aggregate_extensive_margin_by_tya(
            sim_choices_sq, sim_choices_tax, hh_aflav0, cat_idx, T_sim, hh_tya, N_draws;
            aflav_threshold = thresh
        )
        ext_path_t = joinpath(beta_subdir, "Extensive_Margin_by_TYA$(suffix)_b$(b_str).csv")
        open(ext_path_t, "w") do io
            header = ["group", "period", "n_non_users",
                      "sq_ever_initiated", "cf_ever_initiated", "prevention_rate"]
            println(io, join(header, ","))
            for (group_label, df_g) in [("tya", ext_tya_t)]
                for t in 1:T_sim
                    row = [group_label, @sprintf("%d", t), @sprintf("%d", df_g.n_non_users[t]),
                           @sprintf("%.10f", df_g.sq_ever_initiated[t]),
                           @sprintf("%.10f", df_g.cf_ever_initiated[t]),
                           @sprintf("%.10f", df_g.prevention_rate[t])]
                    println(io, join(row, ","))
                end
            end
        end
        log_msg("Extensive margin (thresh=$(thresh)) saved to: $ext_path_t")
    end

    # --- Save Extensive Margin by Type Results ---
    for (thresh, suffix) in [(0.05, "_thresh005"), (0.10, ""), (0.20, "_thresh020")]
        ext_type1_t, ext_type2_t, ext_type3_t = aggregate_extensive_margin_by_type_k3(
            sim_choices_sq, sim_choices_tax, hh_aflav0, cat_idx, T_sim,
            hh_posterior_type1, hh_posterior_type2, hh_posterior_type3, N_draws;
            aflav_threshold = thresh
        )
        ext_path_t = joinpath(beta_subdir, "Extensive_Margin_by_Type$(suffix)_b$(b_str).csv")
        open(ext_path_t, "w") do io
            header = ["type", "period", "n_non_users",
                      "sq_ever_initiated", "cf_ever_initiated", "prevention_rate"]
            println(io, join(header, ","))
            for (type_label, df_g) in [("type1", ext_type1_t), ("type2", ext_type2_t), ("type3", ext_type3_t)]
                for t in 1:T_sim
                    row = [type_label,
                           @sprintf("%d",    t),
                           @sprintf("%d",    df_g.n_non_users[t]),
                           @sprintf("%.10f", df_g.sq_ever_initiated[t]),
                           @sprintf("%.10f", df_g.cf_ever_initiated[t]),
                           @sprintf("%.10f", df_g.prevention_rate[t])]
                    println(io, join(row, ","))
                end
            end
        end
        log_msg("Extensive margin by type (thresh=$(thresh)) saved to: $ext_path_t")
    end

    # --- Save Extensive Margin by TYA Status and Latent Type Results ---
    for (thresh, suffix) in [(0.05, "_thresh005"), (0.10, ""), (0.20, "_thresh020")]
        ext_tya_type1_t, ext_tya_type2_t, ext_tya_type3_t = aggregate_extensive_margin_by_tya_type_k3(
            sim_choices_sq, sim_choices_tax, hh_aflav0, cat_idx, T_sim, hh_tya,
            hh_posterior_type1, hh_posterior_type2, hh_posterior_type3, N_draws;
            aflav_threshold = thresh
        )
        ext_path_t = joinpath(beta_subdir, "Extensive_Margin_by_TYA_Type$(suffix)_b$(b_str).csv")
        open(ext_path_t, "w") do io
            header = ["type", "period", "n_non_users",
                      "sq_ever_initiated", "cf_ever_initiated", "prevention_rate"]
            println(io, join(header, ","))
            for (type_label, df_g) in [("tya_type1", ext_tya_type1_t), ("tya_type2", ext_tya_type2_t), ("tya_type3", ext_tya_type3_t)]
                for t in 1:T_sim
                    row = [type_label,
                           @sprintf("%d",    t),
                           @sprintf("%d",    df_g.n_non_users[t]),
                           @sprintf("%.10f", df_g.sq_ever_initiated[t]),
                           @sprintf("%.10f", df_g.cf_ever_initiated[t]),
                           @sprintf("%.10f", df_g.prevention_rate[t])]
                    println(io, join(row, ","))
                end
            end
        end
        log_msg("Extensive margin by TYA-type (thresh=$(thresh)) saved to: $ext_path_t")
    end

    log_msg("\n--- Forward Simulation Summary (averaged over $T_sim periods) ---")
    log_msg(@sprintf("  Mean addiction (SQ):  %.6f", mean(agg_sq.mean_addiction)))
    log_msg(@sprintf("  Mean addiction (Tax): %.6f", mean(agg_tax.mean_addiction)))
    log_msg(@sprintf("  Addiction change:     %.6f", mean(agg_tax.mean_addiction) - mean(agg_sq.mean_addiction)))
    log_msg(@sprintf("  Mean welfare (SQ):   %.6f", mean(agg_sq.mean_welfare)))
    log_msg(@sprintf("  Mean welfare (Tax):  %.6f", mean(agg_tax.mean_welfare)))
    log_msg(@sprintf("  Welfare change:      %.6f", mean(agg_tax.mean_welfare) - mean(agg_sq.mean_welfare)))

    GC.gc()

end  # end TAX_GRID loop

#############################
# Final Timing and Log Close
#############################

total_elapsed = time() - t_start
log_msg("\n\n===================================")
log_msg("Bootstrap draw b = $b COMPLETE")
log_msg("===================================")
log_msg(@sprintf("Total elapsed time: %.2f minutes (%.2f hours)", total_elapsed / 60, total_elapsed / 3600))
log_msg("All results written to: $output_dir")
log_msg("Bootstrap run complete at $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")

close(log_io)