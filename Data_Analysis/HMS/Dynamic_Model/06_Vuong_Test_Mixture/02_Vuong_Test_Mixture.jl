################################################################################
# William Brasic
# The University of Arizona
# wbrasic97@gmail.com
# September 2026
#
# This script compares the fit of the time-consistent (β = 1) and present-
# biased (β < 1) specifications of the K=3 mixture model using the Vuong
# (1989) test for non-nested models.
#
# β is fixed rather than estimated in each specification, so neither model
# nests the other: they are two parameterizations of the same likelihood with
# the same number of free parameters, evaluated on the same households. A
# likelihood ratio test does not apply, but the Vuong test does.
#
# For each β in BETA_GRID: load that β's point estimates, rebuild the ψ_3-
# dependent flavored habit objects, solve VFI for all three types, and
# evaluate the mixture log-likelihood household by household. The resulting
# per-household contributions are differenced across the two β values and the
# Vuong statistic is computed from their mean and standard deviation.
################################################################################


#############################
# 1. Preliminaries
#############################

# The two β values being compared, read from ENV.
# BETA_A is model A (favored by a positive Vuong statistic) and BETA_B is model B.
BETA_A = parse(Float64, get(ENV, "BETA_A", "1.0"))
BETA_B = parse(Float64, get(ENV, "BETA_B", "0.7"))
BETA_A != BETA_B || error("BETA_A and BETA_B must differ (both are $BETA_A)")

# Flavored habit decay rate ψ_3
# ψ_3 is estimated jointly as parameter 28. PSI_3 only seeds the flavored habit grid.
ESTIMATE_PSI_3 = true   # ψ_3 is always estimated jointly as parameter 28
PSI_3 = parse(Float64, get(ENV, "PSI_3", "0.50"))

# VFI convergence tolerance, set before the functions file is included so its
# default does not apply. Tighter than the 1e-4 used in estimation, as in the
# validation and standard-error scripts.
VFI_TOL = 1e-6

# Tag for the β pair, so runs comparing different pairs do not overwrite each
# other. numeric_tag from the functions file is not loaded yet; same convention
# (1.0 -> "1", 0.7 -> "07").
numeric_tag_pair(val) = replace(rstrip(rstrip(string(val), '0'), '.'), "." => "")
pair_tag = "Beta_$(numeric_tag_pair(BETA_A))_vs_Beta_$(numeric_tag_pair(BETA_B))"

# Detect whether I am running on the HPC (any non-Windows system)
HPC = !Sys.iswindows()

# Set output path and working directory
if HPC

    # Load mixture estimation functions and packages
    include("/home/u2/wbrasic/4th_Year_Paper/Dynamic_Model/02_Second_Stage_Estimation_Mixture/01_Functions_Mixture.jl")

    # Load Vuong-specific functions
    include("/home/u2/wbrasic/4th_Year_Paper/Dynamic_Model/06_Vuong_Test_Mixture/01_Vuong_Functions_Mixture.jl")

    # Construct psi, beta, and psi_3 tags for directory and file naming
    psi_tag   = "Psi2_09_Psi1_01"
    _, _, β_naming, _ = get_fixed_parameters()

    # Root directory holding each β's estimation results
    est_root = "/home/u2/wbrasic/4th_Year_Paper/Dynamic_Model/02_Second_Stage_Estimation_Mixture"

    # Output path is shared across the two β values since the test compares them
    output_dir = "/home/u2/wbrasic/4th_Year_Paper/Dynamic_Model/06_Vuong_Test_Mixture/Vuong_Test_Mixture_$(psi_tag)_$(pair_tag)_Results"

    # Create output directory if it doesn't exist
    mkpath(output_dir)

    # Set working directory to where the data CSVs live (absolute path)
    cd("/home/u2/wbrasic/4th_Year_Paper/Data")
else

    # Load mixture estimation functions and packages
    include("../02_Second_Stage_Estimation_Mixture/01_Functions_Mixture.jl")

    # Load Vuong-specific functions
    include("01_Vuong_Functions_Mixture.jl")

    # Construct psi, beta, and psi_3 tags for directory and file naming
    psi_tag   = "Psi2_09_Psi1_01"
    _, _, β_naming, _ = get_fixed_parameters()

    # Root directory holding each β's estimation results
    est_root = "C:/Users/wbras/OneDrive/Documents/Desktop/UA/4th_Year_Paper/4th_Year_Paper_Data/HMS/2021-Onward/Dynamic_Model"

    # Output path is shared across the two β values since the test compares them
    output_dir = "C:/Users/wbras/OneDrive/Documents/Desktop/UA/4th_Year_Paper/4th_Year_Paper_Data/HMS/2021-Onward/Dynamic_Model/Vuong_Test_Mixture_$(psi_tag)_$(pair_tag)_Results"

    # Create output directory if it doesn't exist
    mkpath(output_dir)

    # Set working directory to where the data CSVs live
    cd("C:/Users/wbras/OneDrive/Documents/Desktop/UA/4th_Year_Paper/4th_Year_Paper_Data/HMS/2021-Onward/Dynamic_Model/Data")
end


#############################
# 2. Output Paths
#############################

# Set log file path
log_path = joinpath(output_dir, "Vuong_Test_Mixture_Log.txt")

# Open log file for writing (log_io is defined as a global in 01_Functions.jl)
log_io = open(log_path, "w")

# Print and log Vuong test start time
log_msg("Vuong test started at $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
log_msg("Comparing β = $BETA_A (model A) against β = $BETA_B (model B)")


#############################
# 3. Initialize Fixed Parameters
#############################

# Load fixed parameters:
#   ψ_2 = fast addiction decay rate (always fixed at 0.90)
#   ψ_1 = slow addiction decay rate (always fixed at 0.10)
#   β = present bias (fixed at 1.0; overridden per β in the loop below)
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

# Pre-compute contiguous household index ranges for the mixture log-likelihood
hh_ranges = precompute_hh_ranges(hh_codes);

# Get number of product categories excluding the outside option
N_K, _ = get_category_choices();


#############################
# 5. Alternative-Level Vectors
#############################

# Get consumption vectors by alternative (STANDARDIZED by max)
N_cig, N_orig_ecig, N_non_fda_flav_ecig, N_fda_flav_ecig, _, q_cig, q_ecig, q_bundle, q_cig_max, q_ecig_max, q_bundle_max = get_consumption(N_J);

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

# Flavored habit stock inflow: a flavored purchase adds one unit regardless of quantity
n_flav = Float64.(is_flavored)


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

# Get Halton draw price transitions: T[m, r, k] where m = price state, r = draw, k = category
T = get_transitions(N_K);

# Pre-compute bilinear interpolation brackets and weights for price transitions
# Returns 6 matrices (M × R): lo/hi grid indices and weights for each category
p_cig_lo, p_cig_hi, p_cig_w, p_ecig_lo, p_ecig_hi, p_ecig_w = precompute_price_transitions(N_P, P, T);


#############################
# 8. Household Price Trajectories
#############################

# Map observed household prices to continuous values for interpolation
# p_continuous is N × 2 (cig price, ecig price): actual per-unit prices, not grid indices
_, p_continuous = map_prices_to_grid(N_P, P, Pcomb, N_J);

# Log data setup completion time and sample size
setup_elapsed = time() - t_setup;
log_msg("Data loading complete in $(round(setup_elapsed, digits=1))s")
log_msg("Observations: $(length(y)), Alternatives: $N_J, Fast addiction states: $N_A_f, Slow addiction states: $N_A_s, Price states: $N_Pcomb")


#############################
# 9. Addiction Stocks
#############################

# Reconstructs each household's addiction stocks across the estimation sample.
# The panel is left-truncated, so the initial stock at each household's first
# observation is found via fixed-point iteration rather than assumed known.
# The flavored habit stock depends on ψ_3, which differs by β when ψ_3 is
# estimated, so it is built per β in section 11.

log_msg("\n===================================")
log_msg("Computing household addiction stocks...")
log_msg("===================================")

t_states = time();

# --- Fast addiction stock, ψ_2 = 0.90 ---
af0, max_fp_iters_f = get_initial_addiction_stock(ψ_2, A_f, n, y, hh_codes);
log_msg("Initial fast addiction stocks: max fixed-point iterations = $max_fp_iters_f")
af_continuous = simulate_addiction_trajectories(N_A_f, ψ_2, A_f, n, y, hh_codes, af0);

# --- Slow addiction stock, ψ_1 = 0.10 ---
as0, max_fp_iters_s = get_initial_addiction_stock(ψ_1, A_s, n, y, hh_codes);
log_msg("Initial slow addiction stocks: max fixed-point iterations = $max_fp_iters_s")
as_continuous = simulate_addiction_trajectories(N_A_s, ψ_1, A_s, n, y, hh_codes, as0);

# Household identifiers and counts
unique_hh = unique(hh_codes);
N_HH = length(unique_hh);
N_obs = length(y);

states_elapsed = time() - t_states;
log_msg("Household addiction stocks computed in $(round(states_elapsed, digits=1))s")
log_msg("Unique households: $N_HH")


#############################
# 10. Pre-compute Addiction
#     Transitions
#############################

# Fast stock transitions (always pre-computed, ψ_2 is never estimated)
af_lower, af_upper, af_weight = precompute_addiction_transitions(N_J, N_A_f, ψ_2, A_f, n);

# Slow stock transitions (ψ_1 = 0.10 always fixed)
as_lower, as_upper, as_weight = precompute_addiction_transitions(N_J, N_A_s, ψ_1, A_s, n);

# The two β values being compared, in the order (model A, model B)
BETA_GRID = [BETA_A, BETA_B]


#############################
# 11. Per-β Log-Likelihood
#############################

# For each β: load that β's estimates, rebuild the ψ_3-dependent flavored
# habit objects, compute flow utilities, solve VFI for all three types, and
# evaluate the mixture log-likelihood household by household.

log_msg("\n===================================")
log_msg("Computing per-household log-likelihoods...")
log_msg("===================================")

# Per-household log-likelihood vectors, keyed by β
hh_ll_by_beta = Dict{Float64, Vector{Float64}}()

# Posterior type probabilities (N_HH x 3), keyed by β
hh_post_by_beta = Dict{Float64, Matrix{Float64}}()

# Number of free parameters per specification, keyed by β
n_params_by_beta = Dict{Float64, Int}()

for _bv in BETA_GRID

    _beta_tag = "Beta_$(numeric_tag(_bv))"

    # Locate and load this β's point estimates
    _est_path = resolve_estimates_path(est_root, psi_tag, _beta_tag)
    _θ_hat, _param_names = load_beta_estimates(_est_path)
    _N_params = length(_param_names)

    log_msg(@sprintf("\nβ = %.2f, loaded from: %s", _bv, _est_path))
    log_msg("Parameters:")
    for k in 1:_N_params
        log_msg("  $(_param_names[k]) = $(_θ_hat[k])")
    end

    # Extract common, type-specific, and mixing weight parameters
    # (28-parameter layout, as in 02_Second_Stage_Estimation_Mixture)
    _N_params == 28 || error("Expected 28 parameters; got $(_N_params) from $_est_path")
    _common  = _θ_hat[1:14]    # α_C, α_E, α_E2, α_CE, λ_1-λ_4, γ_1-γ_4, ω_C, ω_E
    _ξ_1     = _θ_hat[15:17]
    _ξ_2     = _θ_hat[18:20]
    _ξ_3     = _θ_hat[21:23]
    _π_0_2   = _θ_hat[24]
    _π_TYA_2 = _θ_hat[25]
    _π_0_3   = _θ_hat[26]
    _π_TYA_3 = _θ_hat[27]

    # Flavored habit decay rate: estimated as the 28th parameter or fixed from ENV
    _PSI_3 = ESTIMATE_PSI_3 ? _θ_hat[28] : PSI_3
    if ESTIMATE_PSI_3
        log_msg(@sprintf("  ψ_3 = %.6f (estimated)", _PSI_3))
    else
        log_msg(@sprintf("  ψ_3 = %.6f (fixed from ENV)", _PSI_3))
    end

    # Per-type structural parameter vectors
    _θ_struct_1 = vcat(_common, _ξ_1)
    _θ_struct_2 = vcat(_common, _ξ_2)
    _θ_struct_3 = vcat(_common, _ξ_3)

    # Flavored habit objects (ψ_3-dependent; differ across β when ESTIMATE_PSI_3=true)
    _N_A_flav, _A_flav = get_addiction_space(_PSI_3; N_A=10)
    _aflav_lower, _aflav_upper, _aflav_weight = precompute_addiction_transitions(
        N_J, _N_A_flav, _PSI_3, _A_flav, n_flav)
    _aflav0, _ = get_initial_addiction_stock(_PSI_3, _A_flav, n_flav, y, hh_codes)
    _aflav_continuous = simulate_addiction_trajectories(
        _N_A_flav, _PSI_3, _A_flav, n_flav, y, hh_codes, _aflav0)

    # Flow utilities (depend on θ_struct_k which differs by β)
    log_msg(@sprintf("Computing flow utilities for β=%.2f...", _bv))
    _U_1 = get_flow_utility(
        _θ_struct_1, N_J, N_A_f, N_A_s, _N_A_flav, N_Pcomb, A_f, A_s, _A_flav,
        q_cig, q_ecig, q_bundle, is_flavored, is_fda_flavored, is_nonflavored_ecig,
        is_outside, cat_idx, Pcomb, has_cig, has_ecig
    )
    _U_2 = get_flow_utility(
        _θ_struct_2, N_J, N_A_f, N_A_s, _N_A_flav, N_Pcomb, A_f, A_s, _A_flav,
        q_cig, q_ecig, q_bundle, is_flavored, is_fda_flavored, is_nonflavored_ecig,
        is_outside, cat_idx, Pcomb, has_cig, has_ecig
    )
    _U_3 = get_flow_utility(
        _θ_struct_3, N_J, N_A_f, N_A_s, _N_A_flav, N_Pcomb, A_f, A_s, _A_flav,
        q_cig, q_ecig, q_bundle, is_flavored, is_fda_flavored, is_nonflavored_ecig,
        is_outside, cat_idx, Pcomb, has_cig, has_ecig
    )

    # Solve VFI for all three types in parallel at this β
    log_msg(@sprintf("Solving VFI for β=%.2f...", _bv))
    t_vfi = time()

    _vfi_task_1 = Threads.@spawn solve_vfi_sophisticated(
        N_J, N_A_f, N_A_s, _N_A_flav, N_P, N_Pcomb, _bv, δ, _U_1,
        af_lower, af_upper, af_weight,
        as_lower, as_upper, as_weight,
        _aflav_lower, _aflav_upper, _aflav_weight,
        p_cig_lo, p_cig_hi, p_cig_w,
        p_ecig_lo, p_ecig_hi, p_ecig_w;
        V_init = nothing,
        ε = VFI_TOL,
        verbose = true
    )

    _vfi_task_2 = Threads.@spawn solve_vfi_sophisticated(
        N_J, N_A_f, N_A_s, _N_A_flav, N_P, N_Pcomb, _bv, δ, _U_2,
        af_lower, af_upper, af_weight,
        as_lower, as_upper, as_weight,
        _aflav_lower, _aflav_upper, _aflav_weight,
        p_cig_lo, p_cig_hi, p_cig_w,
        p_ecig_lo, p_ecig_hi, p_ecig_w;
        V_init = nothing,
        ε = VFI_TOL,
        verbose = true
    )

    _vfi_task_3 = Threads.@spawn solve_vfi_sophisticated(
        N_J, N_A_f, N_A_s, _N_A_flav, N_P, N_Pcomb, _bv, δ, _U_3,
        af_lower, af_upper, af_weight,
        as_lower, as_upper, as_weight,
        _aflav_lower, _aflav_upper, _aflav_weight,
        p_cig_lo, p_cig_hi, p_cig_w,
        p_ecig_lo, p_ecig_hi, p_ecig_w;
        V_init = nothing,
        ε = VFI_TOL,
        verbose = true
    )

    # Wait for all three VFI tasks to complete
    _, _V_decision_1, _vfi_iters_1, _vfi_converged_1 = fetch(_vfi_task_1)
    _, _V_decision_2, _vfi_iters_2, _vfi_converged_2 = fetch(_vfi_task_2)
    _, _V_decision_3, _vfi_iters_3, _vfi_converged_3 = fetch(_vfi_task_3)

    log_msg(@sprintf("VFI (β=%.2f): type1=%d iters (%s), type2=%d iters (%s), type3=%d iters (%s), %.1fs",
        _bv, _vfi_iters_1, _vfi_converged_1, _vfi_iters_2, _vfi_converged_2,
        _vfi_iters_3, _vfi_converged_3, time() - t_vfi))

    # Evaluate the mixture log-likelihood household by household
    t_ll = time()
    _hh_ll, _hh_post = compute_hh_loglik(
        _V_decision_1, _V_decision_2, _V_decision_3,
        hh_ranges, tya_share_hh, tya_state,
        af_continuous, as_continuous, _aflav_continuous,
        p_continuous, y,
        N_J, N_P, A_f, A_s, _A_flav, P,
        _π_0_2, _π_TYA_2, _π_0_3, _π_TYA_3
    )

    # Store for the test and report the implied totals
    hh_ll_by_beta[_bv]    = _hh_ll
    hh_post_by_beta[_bv]  = _hh_post
    n_params_by_beta[_bv] = _N_params

    log_msg(@sprintf("Per-household log-likelihood computed in %.1fs", time() - t_ll))
    log_msg(@sprintf("β = %.2f: total LL = %.4f (NLL = %.4f)", _bv, sum(_hh_ll), -sum(_hh_ll)))

    # Save this β's per-household contributions for reuse
    _hh_ll_path = joinpath(output_dir, "Vuong_HH_LL_$(_beta_tag).csv")
    open(_hh_ll_path, "w") do io
        println(io, "hh_code,ll")
        for h in 1:N_HH
            println(io, "$(unique_hh[h]),$(@sprintf("%.10f", _hh_ll[h]))")
        end
    end
    log_msg("Per-household LL saved to: $_hh_ll_path")

end

log_msg("\nAll per-household log-likelihoods computed.")


#############################
# 12. Vuong Test
#############################

# Both specifications are evaluated on the same households in the same order,
# so the per-household contributions line up by index and can be differenced
# directly.

log_msg("\n===================================")
log_msg("Vuong (1989) non-nested model comparison")
log_msg("===================================")

ll_A = hh_ll_by_beta[BETA_A]
ll_B = hh_ll_by_beta[BETA_B]

label_A = @sprintf("β = %.2f", BETA_A)
label_B = @sprintf("β = %.2f", BETA_B)

vuong = vuong_test(
    ll_A, ll_B;
    k_a     = n_params_by_beta[BETA_A],
    k_b     = n_params_by_beta[BETA_B],
    n_obs   = N_HH,   # the household is the independent unit, so the BIC penalty uses N_HH (as in estimation)
    label_a = label_A,
    label_b = label_B
)

log_msg(@sprintf("Households: %d, Observations: %d", vuong.N_hh, vuong.N_obs))
log_msg(@sprintf("Free parameters: %s = %d, %s = %d", label_A, vuong.k_a, label_B, vuong.k_b))
log_msg("")
log_msg(@sprintf("Total log-likelihood, %s: %.4f", label_A, vuong.ll_total_a))
log_msg(@sprintf("Total log-likelihood, %s: %.4f", label_B, vuong.ll_total_b))
log_msg(@sprintf("Log-likelihood ratio (A - B): %.4f", vuong.LR))
log_msg("")
log_msg(@sprintf("Mean per-household difference: %.6f", vuong.mean_diff))
log_msg(@sprintf("Std. dev. of differences (ω): %.6f", vuong.omega))
log_msg(@sprintf("Share of households favoring %s: %.4f", label_A, vuong.share_a_wins))
log_msg("")

# Variance pre-test. Fixing β at two values leaves the two specifications
# overlapping rather than strictly non-nested: they coincide wherever the model
# collapses to a static one, since the continuation value is then common across
# alternatives and β drops out. The directional statistic below is only
# interpretable once ω² is distinguishable from zero.
log_msg("Variance pre-test (H0: ω² = 0, models observationally equivalent):")
log_msg(@sprintf("  ω² = %.6f  (SE = %.6f, z = %.4f)", vuong.omega_sq, vuong.omega_sq_se, vuong.omega_sq_z))
log_msg(@sprintf("  N · ω² = %.4f", vuong.pretest_stat))
if vuong.overlap_rejected
    log_msg("  Observational equivalence rejected; the directional test below applies.")
else
    log_msg("  Cannot reject observational equivalence; the directional test below is NOT interpretable.")
end
log_msg("")

log_msg(@sprintf("Vuong statistic (uncorrected): %.4f  (p = %.6f)", vuong.V_raw, vuong.p_raw))
log_msg(@sprintf("Vuong statistic (AIC-corrected): %.4f  (p = %.6f)", vuong.V_aic, vuong.p_aic))
log_msg(@sprintf("Vuong statistic (BIC-corrected): %.4f  (p = %.6f)", vuong.V_bic, vuong.p_bic))
log_msg("")
log_msg("Positive statistic favors $(label_A); negative favors $(label_B).")
log_msg("At the 5% level the test favors: $(vuong.favored)")

# Both specifications fix β rather than estimating it, so they carry the same
# number of free parameters and the AIC and BIC corrections are exactly zero.
if vuong.k_a == vuong.k_b
    log_msg("")
    log_msg("Note: both specifications have the same number of free parameters, so the")
    log_msg("AIC and BIC corrections are zero and all three statistics coincide.")
end


#############################
# 13. Subgroup Breakdowns
#############################

# The headline statistic is an average over households, so it can be carried by
# a minority with large differences. These breakdowns show where the advantage
# actually sits, reporting the median alongside the mean so that any skew is
# visible. Households are split by TYA status and by latent type, the two
# dimensions the paper's counterfactuals report.

log_msg("\n===================================")
log_msg("Subgroup breakdowns")
log_msg("===================================")
log_msg("Positive V favors $(label_A); negative favors $(label_B).")
log_msg("")

# --- TYA status ---
# tya_share_hh[h] is the fraction of household h's months with a teen or young
# adult present. Panelist demographics refresh each January, so the share is not
# always degenerate and households are split three ways as well as two.
idx_all       = collect(1:N_HH)
idx_tya_ever  = findall(>(0.0), tya_share_hh)
idx_tya_never = findall(==(0.0), tya_share_hh)
idx_tya_always    = findall(==(1.0), tya_share_hh)
idx_tya_sometimes = findall(s -> 0.0 < s < 1.0, tya_share_hh)

# --- Latent type ---
# Each household is assigned its modal type under model A's posterior, which is
# the baseline specification. Type 1 is cigarette-persistent, type 2 is
# e-cigarette-persistent, and type 3 is rare users.
post_A = hh_post_by_beta[BETA_A]
modal_type = [argmax(@view post_A[h, :]) for h in 1:N_HH]

type_labels = ("Type 1 (cig-persistent)", "Type 2 (e-cig-persistent)", "Type 3 (rare users)")

log_msg("Modal type counts under $(label_A):")
for k in 1:3
    log_msg(@sprintf("  %-28s %6d (%5.1f%%)", type_labels[k],
        count(==(k), modal_type), 100 * count(==(k), modal_type) / N_HH))
end
log_msg("")

# Build every subgroup row, then log and save them together
sub_rows = Any[]

push!(sub_rows, subgroup_vuong("All households", idx_all, ll_A, ll_B;
    label_a = label_A, label_b = label_B, N_total = N_HH))

push!(sub_rows, subgroup_vuong("TYA present (ever)", idx_tya_ever, ll_A, ll_B;
    label_a = label_A, label_b = label_B, N_total = N_HH))
push!(sub_rows, subgroup_vuong("No TYA (never)", idx_tya_never, ll_A, ll_B;
    label_a = label_A, label_b = label_B, N_total = N_HH))
push!(sub_rows, subgroup_vuong("Always TYA", idx_tya_always, ll_A, ll_B;
    label_a = label_A, label_b = label_B, N_total = N_HH))
push!(sub_rows, subgroup_vuong("Sometimes TYA", idx_tya_sometimes, ll_A, ll_B;
    label_a = label_A, label_b = label_B, N_total = N_HH))

for k in 1:3
    push!(sub_rows, subgroup_vuong(type_labels[k], findall(==(k), modal_type), ll_A, ll_B;
        label_a = label_A, label_b = label_B, N_total = N_HH))
end

# Type crossed with TYA presence, since the policy-relevant cell is
# e-cigarette-persistent households that contain a teen or young adult.
# TYA presence is a simple boolean per household, so the cells are built from
# that directly rather than by searching the index vectors.
has_tya = tya_share_hh .> 0.0

for k in 1:3
    for (tya_lab, want_tya) in (("TYA", true), ("no TYA", false))
        idx_cell = findall(h -> modal_type[h] == k && has_tya[h] == want_tya, 1:N_HH)
        push!(sub_rows, subgroup_vuong("$(type_labels[k]) x $(tya_lab)", idx_cell, ll_A, ll_B;
            label_a = label_A, label_b = label_B, N_total = N_HH))
    end
end

log_subgroup_table(sub_rows, label_A, label_B)


#############################
# 14. Save Results
#############################

# Test results as a one-row CSV
results_path = joinpath(output_dir, "Vuong_Test_Results.csv")
CSV.write(results_path, DataFrame([vuong]))
log_msg("\nTest results saved to: $results_path")

# Subgroup breakdown table
breakdown_path = joinpath(output_dir, "Vuong_Subgroup_Breakdown.csv")
CSV.write(breakdown_path, DataFrame(sub_rows))
log_msg("Subgroup breakdown saved to: $breakdown_path")

# Per-household differences with the subgroup keys attached, so any further cut
# can be made without re-running the value function solves
diff_path = joinpath(output_dir, "Vuong_HH_LL_Differences.csv")
CSV.write(diff_path, DataFrame(
    hh_code     = unique_hh,
    ll_a        = ll_A,
    ll_b        = ll_B,
    ll_diff     = ll_A .- ll_B,
    tya_share   = tya_share_hh,
    tya_present = tya_share_hh .> 0.0,
    modal_type  = modal_type,
    post_type_1 = post_A[:, 1],
    post_type_2 = post_A[:, 2],
    post_type_3 = post_A[:, 3]
))
log_msg("Per-household differences saved to: $diff_path")

# Print and log Vuong test completion time
log_msg("\nVuong test completed at $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")

# Close the log file
close(log_io)
