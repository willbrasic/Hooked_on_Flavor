################################################################################
# William Brasic
# The University of Arizona
# wbrasic97@gmail.com
# February 2026
#
# This script simulates the effect of a flavored tobacco product ban on
# consumer choices, addiction, and welfare. The ban removes flavored
# e-cigarettes (cat_idx = 3) and flavored bundles (cat_idx = 5) from the
# choice set by setting their flow utility to -Inf.
#
# The script:
#   1. Loads estimated parameters θ_hat from Dynamic_Model_Estimates.txt
#   2. Solves VFI under the status quo (all 21 alternatives available)
#   3. Solves VFI under the flavor ban (flavored alternatives removed)
#   4. Computes pointwise choice probabilities and welfare at all observed
#      states under both scenarios
#   5. Forward-simulates households from their last observed states under
#      both scenarios
#   6. Aggregates and saves period-by-period category shares, addiction,
#      and welfare
#
# Results are saved to Counterfactual_Pointwise_Results.txt,
# Counterfactual_Simulation_Results.txt, and Counterfactual_Summary.txt.
################################################################################


#############################
# Preliminaries
#############################

# Whether we are running on the HPC or not
hpc = true

# Set output path and working directory
if hpc

    # Load estimation functions and packages (must come first — provides Printf, CSV, etc.)
    include("../01_Functions.jl")

    # Load counterfactual-specific functions
    include("01_Counterfactual_Functions.jl")

    # Output path for results (use absolute path so it's unaffected by later cd)
    output_dir = abspath("./Counterfactual_Results")

    # Create output directory if it doesn't exist
    mkpath(output_dir)

    # Set working directory to where the data CSVs live
    cd("../../Data")
else

    # Load estimation functions and packages (must come first — provides Printf, CSV, etc.)
    include("../02_Second_Stage_Estimation/01_Functions.jl")

    # Load counterfactual-specific functions
    include("01_Counterfactual_Functions.jl")

    # Output path for results
    output_dir = "C:/Users/wbras/OneDrive/Documents/Desktop/UA/4th_Year_Paper/4th_Year_Paper_Data/HMS/2021-Onward/Dynamic_Model/Counterfactual_Results"

    # Create output directory if it doesn't exist
    mkpath(output_dir)

    # Set working directory to where the data CSVs live
    cd("C:/Users/wbras/OneDrive/Documents/Desktop/UA/4th_Year_Paper/4th_Year_Paper_Data/HMS/2021-Onward/Dynamic_Model/Data")
end


#############################
# Output Paths
#############################

# Set paths
log_path = joinpath(output_dir, "Counterfactual_Log.txt")

# Open log file and set global handle for counterfactual logging
global cf_log_io = open(log_path, "w")
cf_log("Counterfactual simulation started at $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")

# Route VFI progress messages (est_log) to the counterfactual log
global est_log_io = cf_log_io


#############################
# Initialize Fixed Parameters
#############################

# Load fixed parameters
ψ, β, δ = get_fixed_parameters();


#############################
# State Spaces and Choices
#############################

# Start timing
t_setup = time();

# Get number of addiction states and the addiction grid
N_A, A = get_addiction_space(ψ);

# Get number of alternatives (N_J) and choice matrix (J)
_, N_J, J = get_product_choices();

# Get choice vector (y[i] = chosen alternative index for observation i)
y = get_hh_choices(J);

# Get household identifiers
hh_codes = get_hh_codes();

# Get number of categories excluding outside option (N_K)
N_K, _ = get_category_choices();


#############################
# Alternative-Level Vectors
#############################

# Get consumption vectors by alternative (STANDARDIZED by max)
N_cig, N_orig_ecig, N_flav_ecig, _, c_cig, c_ecig, c_bundle, c_cig_max, c_ecig_max, c_bundle_max = get_consumption(N_J);

# Get nicotine vector by alternative (STANDARDIZED by max)
n, n_max = get_nicotine(N_J);

# Get category index by alternative: cat_idx[j] ∈ {0, 1, 2, 3, 4, 5}
cat_idx = get_category_index(N_J, N_cig, N_orig_ecig, N_flav_ecig);

# Get flavored indicator by alternative: is_flavored[j] ∈ {true, false}
is_flavored = get_flavored_indicator(cat_idx);


#############################
# Demographics
#############################

# Get TYA binary indicator for each observation
_, tya = get_teen_young_adult();

# Get TYA state index for each observation (1 = no TYA, 2 = TYA present)
tya_state = get_tya_state(tya);


#############################
# Price Space
#############################

# Get pricing grid (N_P points per category)
N_P, P = get_pricing_spaces();

# Get all price combinations: N_Pcomb = N_P^2, Pcomb is N_Pcomb × 2
N_Pcomb, Pcomb = get_pricing_spaces_combination(N_K, N_P, P);

# Get expenditure matrix (STANDARDIZED by max)
E, E_max = get_expenditures(N_J, N_Pcomb, c_cig, c_ecig, c_cig_max, c_ecig_max, Pcomb);

# Get price transitions from Halton draws: T[m, r, k]
T = get_transitions(N_K);

# Pre-compute price transition brackets and interpolation weights
p_cig_lo, p_cig_hi, p_cig_w, p_ecig_lo, p_ecig_hi, p_ecig_w = precompute_price_transitions(N_P, P, T);


#############################
# Household Price Trajectories
#############################

# Map observed prices to continuous values for interpolation
_, p_continuous = map_prices_to_grid(N_P, P, Pcomb);

setup_elapsed = time() - t_setup;
cf_log("Data loading complete in $(round(setup_elapsed, digits=1))s")
cf_log("Observations: $(length(y)), Alternatives: $N_J, Addiction states: $N_A, Price states: $N_Pcomb")


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

cf_log("AR(1) parameters loaded")


#############################
# Load Estimated Parameters
#############################

# Results directory where Dynamic_Model_Estimates.txt is stored
# After cd to Data/, the results directory is one level up at ../Dynamic_Model_Results
if hpc
    estimates_dir = abspath("../Dynamic_Model_Results")
else
    estimates_dir = "C:/Users/wbras/OneDrive/Documents/Desktop/UA/4th_Year_Paper/4th_Year_Paper_Data/HMS/2021-Onward/Dynamic_Model/Dynamic_Model_Results"
end

# Read θ_hat from the estimates file produced by 02_Estimation.jl
estimates_path = joinpath(estimates_dir, "Dynamic_Model_Estimates.txt")
df_est = CSV.read(estimates_path, DataFrame)

# Extract parameter names and values
param_names = names(df_est)
N_params = length(param_names)
θ_hat = Float64.(collect(df_est[1, :]))

# Print loaded parameters
cf_log("\nLoaded estimates from: $estimates_path")
cf_log("Parameters:")
for k in 1:N_params
    cf_log("  $(param_names[k]) = $(θ_hat[k])")
end


#############################
# Compute Household States
#############################

cf_log("\n===================================")
cf_log("Computing household addiction states...")
cf_log("===================================")

t_states = time()

# Estimate initial addiction stocks via fixed-point iteration
a0, max_fp_iters = get_initial_addiction_stock(ψ, A, n, y, hh_codes)
cf_log("Initial addiction stocks: max fixed-point iterations = $max_fp_iters")

# Simulate addiction trajectories forward using observed choices
_, a_continuous = simulate_addiction_trajectories(N_A, ψ, A, n, y, hh_codes, a0)

# Extract each household's last observed state
# We need: last TYA state, addiction after final choice, final prices
unique_hh = unique(hh_codes)
N_HH = length(unique_hh)
N_obs = length(y)

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

states_elapsed = time() - t_states
cf_log("Household states computed in $(round(states_elapsed, digits=1))s")
cf_log("Unique households: $N_HH")
cf_log("Mean terminal addiction: $(round(mean(hh_a0), digits=4))")


#############################
# Pre-compute Addiction
# Transitions
#############################

# Computed once and reused for both VFI solves (depends on ψ and n, not U)
a_lower, a_upper, a_weight = precompute_addiction_transitions(N_J, N_A, ψ, A, n)


#############################
# Solve VFI: Status Quo
#############################

cf_log("\n===================================")
cf_log("Solving VFI: Status Quo")
cf_log("===================================")

t_vfi_sq = time()

# Compute flow utility at θ_hat
U_sq = get_flow_utility(
    θ_hat, N_J, N_A, N_Pcomb, A, c_cig, c_ecig, c_bundle, n, is_flavored, cat_idx, E
)

# Solve VFI
_, V_choice_sq, vfi_iters_sq, vfi_converged_sq = solve_vfi(
    N_J, N_A, N_P, N_Pcomb, β, δ, U_sq,
    a_lower, a_upper, a_weight,
    p_cig_lo, p_cig_hi, p_cig_w,
    p_ecig_lo, p_ecig_hi, p_ecig_w
)

vfi_sq_elapsed = time() - t_vfi_sq
cf_log("Status quo VFI: $vfi_iters_sq iterations, converged = $vfi_converged_sq ($(round(vfi_sq_elapsed, digits=1))s)")


#############################
# Solve VFI: Flavor Ban
#############################

cf_log("\n===================================")
cf_log("Solving VFI: Flavor Ban")
cf_log("===================================")

t_vfi_ban = time()

# Copy flow utility and apply ban
U_ban = copy(U_sq)
apply_flavor_ban!(U_ban, cat_idx)

# Log which alternatives are banned
banned_alts = findall(j -> cat_idx[j] == 3 || cat_idx[j] == 5, 1:N_J)
cf_log("Banned alternatives (cat_idx ∈ {3, 5}): j = $(banned_alts)")

# Solve VFI under ban
_, V_choice_ban, vfi_iters_ban, vfi_converged_ban = solve_vfi(
    N_J, N_A, N_P, N_Pcomb, β, δ, U_ban,
    a_lower, a_upper, a_weight,
    p_cig_lo, p_cig_hi, p_cig_w,
    p_ecig_lo, p_ecig_hi, p_ecig_w
)

vfi_ban_elapsed = time() - t_vfi_ban
cf_log("Flavor ban VFI: $vfi_iters_ban iterations, converged = $vfi_converged_ban ($(round(vfi_ban_elapsed, digits=1))s)")


#############################
# Pointwise Outcomes
#############################

cf_log("\n===================================")
cf_log("Computing pointwise outcomes...")
cf_log("===================================")

t_pw = time()

# Compute choice probabilities and welfare under status quo
probs_sq, welfare_sq = compute_pointwise_outcomes(
    V_choice_sq, tya_state, a_continuous, p_continuous, N_J, N_P, A, P
)

# Compute choice probabilities and welfare under ban
probs_ban, welfare_ban = compute_pointwise_outcomes(
    V_choice_ban, tya_state, a_continuous, p_continuous, N_J, N_P, A, P
)

pw_elapsed = time() - t_pw
cf_log("Pointwise outcomes computed in $(round(pw_elapsed, digits=1))s")

# Verification: banned alternatives should have exactly zero probability under ban
max_banned_prob = maximum(probs_ban[:, banned_alts])
cf_log("Max probability of banned alternatives under ban: $max_banned_prob")

# Verification: welfare under ban should be ≤ welfare under status quo
welfare_diff = welfare_ban .- welfare_sq
max_welfare_increase = maximum(welfare_diff)
cf_log("Max welfare increase under ban (should be ≤ 0): $max_welfare_increase")

# Save pointwise results
pw_results_path = joinpath(output_dir, "Counterfactual_Pointwise_Results.txt")
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
    println(io, join(header_parts, "\t"))

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
        println(io, join(row_parts, "\t"))
    end
end
cf_log("Pointwise results saved to: $pw_results_path")

# Log pointwise summary statistics
cf_log("\nPointwise summary (means across all observations):")
cat_labels = ["Outside", "Cig", "Orig Ecig", "Flav Ecig", "Orig Bundle", "Flav Bundle"]
cf_log(@sprintf("  %-15s  %12s  %12s  %12s", "Category", "SQ Share", "Ban Share", "Difference"))
cf_log("  " * repeat("-", 55))

for (c, label) in enumerate(cat_labels)
    cat_val = c - 1  # cat_idx values are 0-5
    alt_indices = findall(j -> cat_idx[j] == cat_val, 1:N_J)
    sq_share  = mean(sum(probs_sq[:, alt_indices], dims=2))
    ban_share = mean(sum(probs_ban[:, alt_indices], dims=2))
    cf_log(@sprintf("  %-15s  %12.6f  %12.6f  %12.6f", label, sq_share, ban_share, ban_share - sq_share))
end

cf_log(@sprintf("\n  Mean welfare SQ:   %.6f", mean(welfare_sq)))
cf_log(@sprintf("  Mean welfare Ban:  %.6f", mean(welfare_ban)))
cf_log(@sprintf("  Mean welfare loss: %.6f", mean(welfare_diff)))


#############################
# Forward Simulation
#############################

cf_log("\n===================================")
cf_log("Forward simulation...")
cf_log("===================================")

# Simulation settings
T_sim   = 24    # Months to simulate forward
N_draws = 100   # Monte Carlo draws per household

cf_log("T_sim = $T_sim, N_draws = $N_draws, N_HH = $N_HH")

# Simulate under status quo
cf_log("\nSimulating status quo trajectories...")
t_sim_sq = time()
sim_choices_sq, sim_addiction_sq, sim_welfare_sq = simulate_trajectories(
    V_choice_sq, hh_tya, hh_a0, hh_p0, T_sim, N_draws, ψ,
    N_J, N_P, A, P, n, φ_0, φ_1, L_chol
)
sim_sq_elapsed = time() - t_sim_sq
cf_log("Status quo simulation: $(round(sim_sq_elapsed, digits=1))s")

# Simulate under flavor ban
cf_log("Simulating flavor ban trajectories...")
t_sim_ban = time()
sim_choices_ban, sim_addiction_ban, sim_welfare_ban = simulate_trajectories(
    V_choice_ban, hh_tya, hh_a0, hh_p0, T_sim, N_draws, ψ,
    N_J, N_P, A, P, n, φ_0, φ_1, L_chol
)
sim_ban_elapsed = time() - t_sim_ban
cf_log("Flavor ban simulation: $(round(sim_ban_elapsed, digits=1))s")


#############################
# Aggregate and Save
#############################

cf_log("\n===================================")
cf_log("Aggregating results...")
cf_log("===================================")

# Aggregate status quo
agg_sq = aggregate_simulation(sim_choices_sq, sim_addiction_sq, sim_welfare_sq, cat_idx, N_J, T_sim)

# Aggregate flavor ban
agg_ban = aggregate_simulation(sim_choices_ban, sim_addiction_ban, sim_welfare_ban, cat_idx, N_J, T_sim)

# Save simulation results
sim_results_path = joinpath(output_dir, "Counterfactual_Simulation_Results.txt")
open(sim_results_path, "w") do io

    # Header
    header = ["period",
        "sq_outside", "sq_cig", "sq_orig_ecig", "sq_flav_ecig", "sq_orig_bundle", "sq_flav_bundle",
        "sq_addiction", "sq_welfare",
        "ban_outside", "ban_cig", "ban_orig_ecig", "ban_flav_ecig", "ban_orig_bundle", "ban_flav_bundle",
        "ban_addiction", "ban_welfare"]
    println(io, join(header, "\t"))

    # Data rows
    for t in 1:T_sim
        row = [@sprintf("%d", t),
            @sprintf("%.10f", agg_sq.share_outside[t]),
            @sprintf("%.10f", agg_sq.share_cig[t]),
            @sprintf("%.10f", agg_sq.share_orig_ecig[t]),
            @sprintf("%.10f", agg_sq.share_flav_ecig[t]),
            @sprintf("%.10f", agg_sq.share_orig_bundle[t]),
            @sprintf("%.10f", agg_sq.share_flav_bundle[t]),
            @sprintf("%.10f", agg_sq.mean_addiction[t]),
            @sprintf("%.10f", agg_sq.mean_welfare[t]),
            @sprintf("%.10f", agg_ban.share_outside[t]),
            @sprintf("%.10f", agg_ban.share_cig[t]),
            @sprintf("%.10f", agg_ban.share_orig_ecig[t]),
            @sprintf("%.10f", agg_ban.share_flav_ecig[t]),
            @sprintf("%.10f", agg_ban.share_orig_bundle[t]),
            @sprintf("%.10f", agg_ban.share_flav_bundle[t]),
            @sprintf("%.10f", agg_ban.mean_addiction[t]),
            @sprintf("%.10f", agg_ban.mean_welfare[t])]
        println(io, join(row, "\t"))
    end
end
cf_log("Simulation results saved to: $sim_results_path")


#############################
# Summary Statistics
#############################

summary_path = joinpath(output_dir, "Counterfactual_Summary.txt")
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
    println(io, @sprintf("%-15s  %12s  %12s  %12s", "Category", "SQ Share", "Ban Share", "Difference"))
    println(io, repeat("-", 55))
    for (c, label) in enumerate(cat_labels)
        cat_val = c - 1
        alt_indices = findall(j -> cat_idx[j] == cat_val, 1:N_J)
        sq_share  = mean(sum(probs_sq[:, alt_indices], dims=2))
        ban_share = mean(sum(probs_ban[:, alt_indices], dims=2))
        println(io, @sprintf("%-15s  %12.6f  %12.6f  %12.6f", label, sq_share, ban_share, ban_share - sq_share))
    end
    println(io, @sprintf("\nMean welfare (SQ):       %.6f", mean(welfare_sq)))
    println(io, @sprintf("Mean welfare (Ban):      %.6f", mean(welfare_ban)))
    println(io, @sprintf("Mean welfare change:     %.6f", mean(welfare_diff)))

    # Simulation summary: average across all periods
    println(io, "\n--- Forward Simulation (averaged over $T_sim periods) ---\n")
    println(io, @sprintf("%-15s  %12s  %12s  %12s", "Category", "SQ Share", "Ban Share", "Difference"))
    println(io, repeat("-", 55))

    sim_cat_cols_sq  = [:share_outside, :share_cig, :share_orig_ecig, :share_flav_ecig, :share_orig_bundle, :share_flav_bundle]
    sim_cat_cols_ban = [:share_outside, :share_cig, :share_orig_ecig, :share_flav_ecig, :share_orig_bundle, :share_flav_bundle]
    for (c, label) in enumerate(cat_labels)
        col = sim_cat_cols_sq[c]
        sq_share  = mean(agg_sq[!, col])
        ban_share = mean(agg_ban[!, col])
        println(io, @sprintf("%-15s  %12.6f  %12.6f  %12.6f", label, sq_share, ban_share, ban_share - sq_share))
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

cf_log("\nSummary saved to: $summary_path")


#############################
# Log Final Timing
#############################

total_elapsed = time() - t_setup
cf_log("\n===================================")
cf_log("Counterfactual simulation complete")
cf_log(@sprintf("Total time: %.1fs", total_elapsed))
cf_log("===================================")
cf_log("Finished at $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")

# Close log file
close(cf_log_io)
