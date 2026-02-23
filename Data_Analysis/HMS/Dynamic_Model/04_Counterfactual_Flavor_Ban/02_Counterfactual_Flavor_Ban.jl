################################################################################
# William Brasic
# The University of Arizona
# wbrasic97@gmail.com
# February 2026
#
# This script simulates the effect of a flavored tobacco product ban on
# consumer choices, addiction, and welfare. The ban removes flavored
# e-cigarettes (cat_idx in {3,4}) and flavored bundles (cat_idx in {6,7}) from the
# choice set by setting their flow utility to -Inf.
#
# The script:
#   1. Loads estimated parameters θ_hat from Dynamic_Model_<beta_tag>_Estimates.csv
#   2. Solves VFI under the status quo (all 40 alternatives available)
#   3. Solves VFI under the flavor ban (flavored alternatives removed)
#   4. Computes pointwise choice probabilities and welfare at all observed
#      states under both scenarios
#   5. Forward-simulates households from their last observed states under
#      both scenarios
#   6. Aggregates and saves period-by-period category shares, addiction,
#      and welfare
#
# Results are saved to Counterfactual_Pointwise_Results.csv,
# Counterfactual_Simulation_Results.csv, and Counterfactual_Summary.txt.
################################################################################


#############################
# Preliminaries
#############################

# Base model: β is fixed at 1.0 (not estimated)
ESTIMATE_BETA = false

# Base model: ψ is fixed at 0.68 (not estimated)
ESTIMATE_PSI = false

# Whether we are running on the HPC or not
hpc = true

# Load Random for common random number seeding in forward simulations
using Random

# Set output path and working directory
if hpc

    # Load estimation functions and packages (must come first — provides Printf, CSV, etc.)
    include("../01_Functions.jl")

    # Load counterfactual-specific functions
    include("01_Counterfactual_Functions.jl")

    # Construct psi and beta tags for directory and file naming
    ψ_naming, β_naming, _ = get_fixed_parameters()
    psi_tag = ESTIMATE_PSI ? "Psi_Estimated" : "Psi_$(ψ_naming)"
    beta_tag = ESTIMATE_BETA ? "Beta_Estimated" : "Beta_$(β_naming)"

    # Output path for results (use absolute path so it's unaffected by later cd)
    output_dir = abspath("./Counterfactual_$(psi_tag)_$(beta_tag)_Results")

    # Create output directory if it doesn't exist
    mkpath(output_dir)

    # Set working directory to where the data CSVs live
    cd("../../Data")
else

    # Load estimation functions and packages (must come first — provides Printf, CSV, etc.)
    include("../02_Second_Stage_Estimation/01_Functions.jl")

    # Load counterfactual-specific functions
    include("01_Counterfactual_Functions.jl")

    # Construct psi and beta tags for directory and file naming
    ψ_naming, β_naming, _ = get_fixed_parameters()
    psi_tag = ESTIMATE_PSI ? "Psi_Estimated" : "Psi_$(ψ_naming)"
    beta_tag = ESTIMATE_BETA ? "Beta_Estimated" : "Beta_$(β_naming)"

    # Output path for results (includes psi and beta tags in directory name)
    output_dir = "C:/Users/wbras/OneDrive/Documents/Desktop/UA/4th_Year_Paper/4th_Year_Paper_Data/HMS/2021-Onward/Dynamic_Model/Counterfactual_$(psi_tag)_$(beta_tag)_Results"

    # Create output directory if it doesn't exist
    mkpath(output_dir)

    # Set working directory to where the data CSVs live
    cd("C:/Users/wbras/OneDrive/Documents/Desktop/UA/4th_Year_Paper/4th_Year_Paper_Data/HMS/2021-Onward/Dynamic_Model/Data")
end


#############################
# Output Paths
#############################

# Set log file path
log_path = joinpath(output_dir, "Counterfactual_Log.txt")

# Open log file for writing (log_io is defined as a global in 01_Functions.jl)
log_io = open(log_path, "w")

# Print and log counterfactual simulation start time
log_msg("Counterfactual simulation started at $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")


#############################
# Initialize Fixed Parameters
#############################

# Load fixed parameters:
#   ψ = addiction decay rate (fixed at 0.68; overridden when ESTIMATE_PSI = true)
#   β = present bias (fixed at 1.0; exponential discounting in the base model)
#   δ = monthly discount factor (fixed at 0.99)
ψ, β, δ = get_fixed_parameters();


#############################
# State Spaces and Choices
#############################

# Start timer for data prep
t_setup = time();

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
tya_state = get_tya_states();

# Load 4×4 monthly TYA transition matrix Π[s, s'] = P(TYA' = s' | TYA = s)
# Used in VFI to integrate over anticipated TYA state changes
Π = get_tya_transitions();


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
setup_elapsed = time() - t_setup;
log_msg("Data loading complete in $(round(setup_elapsed, digits=1))s")
log_msg("Observations: $(length(y)), Alternatives: $N_J, Addiction states: $N_A, Price states: $N_Pcomb")


#############################
# AR(1) Price Parameters
#############################

# Load AR(1) coefficients: φ₀ (intercept) and φ₁ (AR coefficient)
AR_Phi = CSV.read("../AR_Parameters/AR_Parameters_Phi.csv", DataFrame);
φ_0 = [AR_Phi[1, :intercept], AR_Phi[2, :intercept]];
φ_1 = [AR_Phi[1, :ar1], AR_Phi[2, :ar1]];

# Load AR(1) shock covariance matrix Σ
AR_Sigma = CSV.read("../AR_Parameters/AR_Parameters_Sigma.csv", DataFrame);
Σ = [AR_Sigma[1, :cig] AR_Sigma[1, :ecig];
     AR_Sigma[2, :cig] AR_Sigma[2, :ecig]];

# Cholesky decomposition: L such that LL' = Σ
L_chol = cholesky(Σ).L;

# Print and log AR(1) parameter loading confirmation
log_msg("AR(1) parameters loaded")


#############################
# Load Estimated Parameters
#############################

# Results directory where estimated parameters are stored (includes psi and beta tags in directory name)
if hpc
    estimates_dir = abspath("../Dynamic_Model_$(psi_tag)_$(beta_tag)_Results")
else
    estimates_dir = "C:/Users/wbras/OneDrive/Documents/Desktop/UA/4th_Year_Paper/4th_Year_Paper_Data/HMS/2021-Onward/Dynamic_Model/Dynamic_Model_$(psi_tag)_$(beta_tag)_Results"
end

# Read θ_hat from the estimates file produced by 02_Estimation.jl (includes psi and beta tags in filename)
estimates_path = joinpath(estimates_dir, "Dynamic_Model_$(psi_tag)_$(beta_tag)_Estimates.csv");
df_est = CSV.read(estimates_path, DataFrame);

# Extract parameter names and values
param_names = names(df_est);
N_params = length(param_names);
θ_hat = Float64.(collect(df_est[1, :]));

# Print and log loaded parameters
log_msg("\nLoaded estimates from: $estimates_path")
log_msg("Parameters:")
for k in 1:N_params
    log_msg("  $(param_names[k]) = $(θ_hat[k])")
end


#############################
# Compute Household States
#############################

# Print and log household state computation header
log_msg("\n===================================")
log_msg("Computing household addiction states...")
log_msg("===================================")

t_states = time();

# Estimate initial addiction stocks via fixed-point iteration
a0, max_fp_iters = get_initial_addiction_stock(ψ, A, n, y, hh_codes);
log_msg("Initial addiction stocks: max fixed-point iterations = $max_fp_iters")

# Simulate addiction trajectories forward using observed choices
_, a_continuous = simulate_addiction_trajectories(N_A, ψ, A, n, y, hh_codes, a0);

# Extract each household's last observed state
# We need: last TYA state, addiction after final choice, final prices
unique_hh = unique(hh_codes);
N_HH = length(unique_hh);
N_obs = length(y);

# Map household codes to their observation indices
hh_obs = Dict{eltype(hh_codes), Vector{Int}}()
for i in 1:N_obs
    hh = hh_codes[i]
    if !haskey(hh_obs, hh)
        hh_obs[hh] = Int[]
    end
    push!(hh_obs[hh], i)
end

# Extract terminal states for each household
hh_tya = Vector{Int}(undef, N_HH)
hh_a0  = Vector{Float64}(undef, N_HH)
hh_p0  = Matrix{Float64}(undef, N_HH, 2)

for (h_idx, hh) in enumerate(unique_hh)

    obs_indices = hh_obs[hh]
    last_obs = obs_indices[end]

    # TYA state at last observation
    hh_tya[h_idx] = tya_state[last_obs]

    # Last observed prices
    hh_p0[h_idx, 1] = p_continuous[last_obs, 1]
    hh_p0[h_idx, 2] = p_continuous[last_obs, 2]

    # Addiction after final choice: evolve one step from last observed addiction
    # a_continuous[last_obs] is the addiction at the START of the last period
    # After choosing y[last_obs], addiction evolves to:
    hh_a0[h_idx] = addiction_evolution(ψ, a_continuous[last_obs], n[y[last_obs]])
    hh_a0[h_idx] = clamp(hh_a0[h_idx], A[1], A[end])
end

# Print and log household state computation results
states_elapsed = time() - t_states;
log_msg("Household states computed in $(round(states_elapsed, digits=1))s")
log_msg("Unique households: $N_HH")
log_msg("Mean terminal addiction: $(round(mean(hh_a0), digits=4))")


#############################
# Pre-compute Addiction
# Transitions
#############################

# Computed once and reused for both VFI solves (depends on ψ and n, not U)
a_lower, a_upper, a_weight = precompute_addiction_transitions(N_J, N_A, ψ, A, n);


#############################
# Solve VFI: Status Quo
#############################

# Print and log status quo VFI header
log_msg("\n===================================")
log_msg("Solving VFI: Status Quo")
log_msg("===================================")

t_vfi_sq = time();

# Compute flow utility at θ_hat
U_sq = get_flow_utility(
    θ_hat, N_J, N_A, N_Pcomb, A, q_cig, q_ecig, q_bundle, n, is_flavored, is_fda_flavored, cat_idx, E
)

# Solve VFI
_, V_choice_sq, vfi_iters_sq, vfi_converged_sq = solve_vfi_sophisticated(
    N_J, N_A, N_P, N_Pcomb, β, δ, U_sq,
    a_lower, a_upper, a_weight,
    p_cig_lo, p_cig_hi, p_cig_w,
    p_ecig_lo, p_ecig_hi, p_ecig_w,
    Π
)

# Print and log status quo VFI result
vfi_sq_elapsed = time() - t_vfi_sq;
log_msg("Status quo VFI: $vfi_iters_sq iterations, converged = $vfi_converged_sq ($(round(vfi_sq_elapsed, digits=1))s)")


#############################
# Solve VFI: Flavor Ban
#############################

# Print and log flavor ban VFI header
log_msg("\n===================================")
log_msg("Solving VFI: Flavor Ban")
log_msg("===================================")

t_vfi_ban = time();

# Copy flow utility and apply ban
U_ban = copy(U_sq);
apply_flavor_ban!(U_ban, cat_idx);

# Log which alternatives are banned
banned_alts = findall(j -> cat_idx[j] in (3, 4, 6, 7), 1:N_J);
log_msg("Banned alternatives (cat_idx ∈ {3, 4, 6, 7}): j = $(banned_alts)")

# Solve VFI under ban
_, V_choice_ban, vfi_iters_ban, vfi_converged_ban = solve_vfi_sophisticated(
    N_J, N_A, N_P, N_Pcomb, β, δ, U_ban,
    a_lower, a_upper, a_weight,
    p_cig_lo, p_cig_hi, p_cig_w,
    p_ecig_lo, p_ecig_hi, p_ecig_w,
    Π
)

# Print and log flavor ban VFI result
vfi_ban_elapsed = time() - t_vfi_ban;
log_msg("Flavor ban VFI: $vfi_iters_ban iterations, converged = $vfi_converged_ban ($(round(vfi_ban_elapsed, digits=1))s)")


#############################
# Pointwise Outcomes
#############################

# Print and log pointwise outcomes computation header
log_msg("\n===================================")
log_msg("Computing pointwise outcomes...")
log_msg("===================================")

t_pw = time();

# Compute choice probabilities and welfare under status quo
probs_sq, welfare_sq = compute_pointwise_outcomes(
    V_choice_sq, tya_state, a_continuous, p_continuous, N_J, N_P, A, P
)

# Compute choice probabilities and welfare under ban
probs_ban, welfare_ban = compute_pointwise_outcomes(
    V_choice_ban, tya_state, a_continuous, p_continuous, N_J, N_P, A, P
)

# Print and log pointwise computation time
pw_elapsed = time() - t_pw;
log_msg("Pointwise outcomes computed in $(round(pw_elapsed, digits=1))s")

# Verification: banned alternatives should have exactly zero probability under ban
max_banned_prob = maximum(probs_ban[:, banned_alts]);
log_msg("Max probability of banned alternatives under ban: $max_banned_prob")

# Verification: welfare under ban should be ≤ welfare under status quo
welfare_diff = welfare_ban .- welfare_sq;
max_welfare_increase = maximum(welfare_diff);
log_msg("Max welfare increase under ban (should be ≤ 0): $max_welfare_increase")

# Save pointwise results
pw_results_path = joinpath(output_dir, "Counterfactual_Pointwise_Results.csv");
open(pw_results_path, "w") do io

    # Header
    header_parts = ["obs"]
    for j in 1:N_J
        push!(header_parts, "prob_sq_$j")
    end
    for j in 1:N_J
        push!(header_parts, "prob_ban_$j")
    end
    push!(header_parts, "welfare_sq", "welfare_ban", "welfare_diff")
    println(io, join(header_parts, ","))

    # Data rows
    for i in 1:N_obs
        row_parts = [@sprintf("%d", i)]
        for j in 1:N_J
            push!(row_parts, @sprintf("%.10f", probs_sq[i, j]))
        end
        for j in 1:N_J
            push!(row_parts, @sprintf("%.10f", probs_ban[i, j]))
        end
        push!(row_parts, @sprintf("%.10f", welfare_sq[i]))
        push!(row_parts, @sprintf("%.10f", welfare_ban[i]))
        push!(row_parts, @sprintf("%.10f", welfare_diff[i]))
        println(io, join(row_parts, ","))
    end
end

# Print and log pointwise results save location
log_msg("Pointwise results saved to: $pw_results_path")

# Print and log pointwise summary statistics
log_msg("\nPointwise summary (means across all observations):")
cat_labels = ["Outside", "Cig", "Orig Ecig", "Non-FDA Flav Ecig", "FDA Flav Ecig", "Orig Bundle", "Non-FDA Flav Bundle", "FDA Flav Bundle"];
log_msg(@sprintf("  %-22s  %12s  %12s  %12s", "Category", "SQ Share", "Ban Share", "Difference"))
log_msg("  " * repeat("-", 62))

for (c, label) in enumerate(cat_labels)
    cat_val = c - 1  # cat_idx values are 0-7
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
#############################

# Print and log forward simulation header
log_msg("\n===================================")
log_msg("Forward simulation...")
log_msg("===================================")

# Simulation settings
T_sim   = 24    # Months to simulate forward
N_draws = 100   # Monte Carlo draws per household

log_msg("T_sim = $T_sim, N_draws = $N_draws, N_HH = $N_HH")

# Use common random numbers (CRN) for status quo and ban simulations.
# By seeding the RNG identically before each simulation, both scenarios
# see the same price shock sequences and choice draws, reducing variance
# of the welfare difference.
crn_seed = 12345

# Simulate under status quo
log_msg("\nSimulating status quo trajectories...")
Random.seed!(crn_seed)
t_sim_sq = time();
sim_choices_sq, sim_addiction_sq, sim_welfare_sq = simulate_trajectories(
    V_choice_sq, hh_tya, hh_a0, hh_p0, T_sim, N_draws, ψ,
    N_J, N_P, A, P, n, φ_0, φ_1, L_chol
)
sim_sq_elapsed = time() - t_sim_sq;
log_msg("Status quo simulation: $(round(sim_sq_elapsed, digits=1))s")

# Simulate under flavor ban (reset seed for CRN — same draws as status quo)
log_msg("Simulating flavor ban trajectories...")
Random.seed!(crn_seed)
t_sim_ban = time();
sim_choices_ban, sim_addiction_ban, sim_welfare_ban = simulate_trajectories(
    V_choice_ban, hh_tya, hh_a0, hh_p0, T_sim, N_draws, ψ,
    N_J, N_P, A, P, n, φ_0, φ_1, L_chol
)
sim_ban_elapsed = time() - t_sim_ban;
log_msg("Flavor ban simulation: $(round(sim_ban_elapsed, digits=1))s")


#############################
# Aggregate and Save
#############################

# Print and log aggregation header
log_msg("\n===================================")
log_msg("Aggregating results...")
log_msg("===================================")

# Aggregate status quo
agg_sq = aggregate_simulation(sim_choices_sq, sim_addiction_sq, sim_welfare_sq, cat_idx, N_J, T_sim);

# Aggregate flavor ban
agg_ban = aggregate_simulation(sim_choices_ban, sim_addiction_ban, sim_welfare_ban, cat_idx, N_J, T_sim);

# Save simulation results
sim_results_path = joinpath(output_dir, "Counterfactual_Simulation_Results.csv");
open(sim_results_path, "w") do io

    # Header
    header = ["period",
        "sq_outside", "sq_cig", "sq_orig_ecig", "sq_non_fda_flav_ecig", "sq_fda_flav_ecig",
        "sq_orig_bundle", "sq_non_fda_flav_bundle", "sq_fda_flav_bundle",
        "sq_addiction", "sq_welfare",
        "ban_outside", "ban_cig", "ban_orig_ecig", "ban_non_fda_flav_ecig", "ban_fda_flav_ecig",
        "ban_orig_bundle", "ban_non_fda_flav_bundle", "ban_fda_flav_bundle",
        "ban_addiction", "ban_welfare"]
    println(io, join(header, ","))

    # Data rows
    for t in 1:T_sim
        row = [@sprintf("%d", t),
            @sprintf("%.10f", agg_sq.share_outside[t]),
            @sprintf("%.10f", agg_sq.share_cig[t]),
            @sprintf("%.10f", agg_sq.share_orig_ecig[t]),
            @sprintf("%.10f", agg_sq.share_non_fda_flav_ecig[t]),
            @sprintf("%.10f", agg_sq.share_fda_flav_ecig[t]),
            @sprintf("%.10f", agg_sq.share_orig_bundle[t]),
            @sprintf("%.10f", agg_sq.share_non_fda_flav_bundle[t]),
            @sprintf("%.10f", agg_sq.share_fda_flav_bundle[t]),
            @sprintf("%.10f", agg_sq.mean_addiction[t]),
            @sprintf("%.10f", agg_sq.mean_welfare[t]),
            @sprintf("%.10f", agg_ban.share_outside[t]),
            @sprintf("%.10f", agg_ban.share_cig[t]),
            @sprintf("%.10f", agg_ban.share_orig_ecig[t]),
            @sprintf("%.10f", agg_ban.share_non_fda_flav_ecig[t]),
            @sprintf("%.10f", agg_ban.share_fda_flav_ecig[t]),
            @sprintf("%.10f", agg_ban.share_orig_bundle[t]),
            @sprintf("%.10f", agg_ban.share_non_fda_flav_bundle[t]),
            @sprintf("%.10f", agg_ban.share_fda_flav_bundle[t]),
            @sprintf("%.10f", agg_ban.mean_addiction[t]),
            @sprintf("%.10f", agg_ban.mean_welfare[t])]
        println(io, join(row, ","))
    end
end

# Print and log simulation results save location
log_msg("Simulation results saved to: $sim_results_path")


#############################
# Summary Statistics
#############################

# Save counterfactual summary to a text file
summary_path = joinpath(output_dir, "Counterfactual_Summary.txt");
open(summary_path, "w") do io

    println(io, "===================================")
    println(io, "Counterfactual Flavor Ban Summary")
    println(io, "===================================\n")
    println(io, "Date: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
    println(io, "Households: $N_HH")
    println(io, "Simulation periods: $T_sim")
    println(io, "Monte Carlo draws: $N_draws")
    println(io, "")

    # Parameter estimates used
    println(io, "Estimated parameters (θ_hat):")
    for k in 1:N_params
        println(io, @sprintf("  %-8s = %.10f", param_names[k], θ_hat[k]))
    end

    # VFI convergence
    println(io, "\nVFI convergence:")
    println(io, "  Status quo: $vfi_iters_sq iterations, converged = $vfi_converged_sq")
    println(io, "  Flavor ban: $vfi_iters_ban iterations, converged = $vfi_converged_ban")

    # Pointwise summary
    println(io, "\n--- Pointwise Analysis (at observed states) ---\n")
    println(io, @sprintf("%-22s  %12s  %12s  %12s", "Category", "SQ Share", "Ban Share", "Difference"))
    println(io, repeat("-", 62))
    for (c, label) in enumerate(cat_labels)
        cat_val = c - 1
        alt_indices = findall(j -> cat_idx[j] == cat_val, 1:N_J)
        sq_share  = mean(sum(probs_sq[:, alt_indices], dims=2))
        ban_share = mean(sum(probs_ban[:, alt_indices], dims=2))
        println(io, @sprintf("%-22s  %12.6f  %12.6f  %12.6f", label, sq_share, ban_share, ban_share - sq_share))
    end
    println(io, @sprintf("\nMean welfare (SQ):       %.6f", mean(welfare_sq)))
    println(io, @sprintf("Mean welfare (Ban):      %.6f", mean(welfare_ban)))
    println(io, @sprintf("Mean welfare change:     %.6f", mean(welfare_diff)))

    # Simulation summary: average across all periods
    println(io, "\n--- Forward Simulation (averaged over $T_sim periods) ---\n")
    println(io, @sprintf("%-22s  %12s  %12s  %12s", "Category", "SQ Share", "Ban Share", "Difference"))
    println(io, repeat("-", 62))

    sim_cat_cols_sq  = [:share_outside, :share_cig, :share_orig_ecig, :share_non_fda_flav_ecig, :share_fda_flav_ecig, :share_orig_bundle, :share_non_fda_flav_bundle, :share_fda_flav_bundle]
    sim_cat_cols_ban = [:share_outside, :share_cig, :share_orig_ecig, :share_non_fda_flav_ecig, :share_fda_flav_ecig, :share_orig_bundle, :share_non_fda_flav_bundle, :share_fda_flav_bundle]
    for (c, label) in enumerate(cat_labels)
        col = sim_cat_cols_sq[c]
        sq_share  = mean(agg_sq[!, col])
        ban_share = mean(agg_ban[!, col])
        println(io, @sprintf("%-22s  %12.6f  %12.6f  %12.6f", label, sq_share, ban_share, ban_share - sq_share))
    end

    println(io, @sprintf("\nMean addiction (SQ):     %.6f", mean(agg_sq.mean_addiction)))
    println(io, @sprintf("Mean addiction (Ban):    %.6f", mean(agg_ban.mean_addiction)))
    println(io, @sprintf("Addiction change:        %.6f", mean(agg_ban.mean_addiction) - mean(agg_sq.mean_addiction)))
    println(io, @sprintf("\nMean welfare (SQ):      %.6f", mean(agg_sq.mean_welfare)))
    println(io, @sprintf("Mean welfare (Ban):     %.6f", mean(agg_ban.mean_welfare)))
    println(io, @sprintf("Welfare change:         %.6f", mean(agg_ban.mean_welfare) - mean(agg_sq.mean_welfare)))

    # Period-by-period welfare trajectory
    println(io, "\n--- Period-by-Period Welfare ---\n")
    println(io, @sprintf("%-8s  %12s  %12s  %12s", "Period", "SQ Welfare", "Ban Welfare", "Difference"))
    println(io, repeat("-", 50))
    for t in 1:T_sim
        diff = agg_ban.mean_welfare[t] - agg_sq.mean_welfare[t]
        println(io, @sprintf("%-8d  %12.6f  %12.6f  %12.6f", t, agg_sq.mean_welfare[t], agg_ban.mean_welfare[t], diff))
    end
end

# Print and log summary save location
log_msg("\nSummary saved to: $summary_path")


#############################
# Log Final Timing
#############################

# Print and log final timing and completion message
total_elapsed = time() - t_setup;
log_msg("\n===================================")
log_msg("Counterfactual simulation complete")
log_msg(@sprintf("Total time: %.1fs", total_elapsed))
log_msg("===================================")
log_msg("Finished at $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")

# Close the log file handle
close(log_io)
