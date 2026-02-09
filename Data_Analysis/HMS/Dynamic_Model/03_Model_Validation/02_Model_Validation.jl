################################################################################
# William Brasic
# The University of Arizona
# wbrasic97@gmail.com
# February 2026
#
# This script validates the dynamic model by comparing predicted choice
# probabilities (from the estimated model) to actual observed choices.
#
# Validation exercises:
#   1. Overall category shares: predicted vs actual (all observations pooled)
#   2. Category shares by TYA status: predicted vs actual
#   3. Category shares by calendar month: predicted vs actual time series
#   4. Alternative-level shares: predicted vs actual for all 21 products
#
# The predicted probabilities are computed by solving VFI at θ_hat and
# interpolating V_choice at each observation's continuous state.
#
# Results are saved to Model_Validation_Results.txt and
# Model_Validation_Log.txt.
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

    # Load validation-specific functions
    include("01_Model_Validation_Functions.jl")

    # Output path for results (use absolute path so it's unaffected by later cd)
    output_dir = abspath("./Model_Validation_Results")

    # Create output directory if it doesn't exist
    mkpath(output_dir)

    # Set working directory to where the data CSVs live
    cd("../../Data")
else

    # Load estimation functions and packages (must come first — provides Printf, CSV, etc.)
    include("../02_Second_Stage_Estimation/01_Functions.jl")

    # Load validation-specific functions
    include("01_Model_Validation_Functions.jl")

    # Output path for results
    output_dir = "C:/Users/wbras/OneDrive/Documents/Desktop/UA/4th_Year_Paper/4th_Year_Paper_Data/HMS/2021-Onward/Dynamic_Model/Model_Validation_Results"

    # Create output directory if it doesn't exist
    mkpath(output_dir)

    # Set working directory to where the data CSVs live
    cd("C:/Users/wbras/OneDrive/Documents/Desktop/UA/4th_Year_Paper/4th_Year_Paper_Data/HMS/2021-Onward/Dynamic_Model/Data")
end


#############################
# Output Paths
#############################

# Set paths
log_path = joinpath(output_dir, "Model_Validation_Log.txt")

# Open log file and set global handle for validation logging
global val_log_io = open(log_path, "w")
val_log("Model validation started at $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")

# Route VFI progress messages (est_log) to the validation log
global est_log_io = val_log_io


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

# Get period indices for time-series analysis
period_idx, period_labels, N_periods = get_period_indices();

N_obs = length(y)
setup_elapsed = time() - t_setup;
val_log("Data loading complete in $(round(setup_elapsed, digits=1))s")
val_log("Observations: $N_obs, Alternatives: $N_J, Periods: $N_periods")


#############################
# Load Estimated Parameters
#############################

# Results directory where Dynamic_Model_Estimates.txt is stored
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
val_log("\nLoaded estimates from: $estimates_path")
val_log("Parameters:")
for k in 1:N_params
    val_log("  $(param_names[k]) = $(θ_hat[k])")
end


#############################
# Compute Household Addiction
# Trajectories
#############################

val_log("\n===================================")
val_log("Computing household addiction states...")
val_log("===================================")

t_states = time()

# Estimate initial addiction stocks via fixed-point iteration
a0, max_fp_iters = get_initial_addiction_stock(ψ, A, n, y, hh_codes)
val_log("Initial addiction stocks: max fixed-point iterations = $max_fp_iters")

# Simulate addiction trajectories forward using observed choices
_, a_continuous = simulate_addiction_trajectories(N_A, ψ, A, n, y, hh_codes, a0)

states_elapsed = time() - t_states
val_log("Addiction states computed in $(round(states_elapsed, digits=1))s")


#############################
# Solve VFI at θ_hat
#############################

val_log("\n===================================")
val_log("Solving VFI at estimated parameters...")
val_log("===================================")

t_vfi = time()

# Pre-compute addiction transitions
a_lower, a_upper, a_weight = precompute_addiction_transitions(N_J, N_A, ψ, A, n)

# Compute flow utility at θ_hat
U = get_flow_utility(
    θ_hat, N_J, N_A, N_Pcomb, A, c_cig, c_ecig, c_bundle, n, is_flavored, cat_idx, E
)

# Solve VFI
_, V_choice, vfi_iters, vfi_converged = solve_vfi(
    N_J, N_A, N_P, N_Pcomb, β, δ, U,
    a_lower, a_upper, a_weight,
    p_cig_lo, p_cig_hi, p_cig_w,
    p_ecig_lo, p_ecig_hi, p_ecig_w
)

vfi_elapsed = time() - t_vfi
val_log("VFI: $vfi_iters iterations, converged = $vfi_converged ($(round(vfi_elapsed, digits=1))s)")


#############################
# Compute Predicted
# Choice Probabilities
#############################

val_log("\n===================================")
val_log("Computing predicted choice probabilities...")
val_log("===================================")

t_pred = time()

probs = compute_predicted_probs(
    V_choice, tya_state, a_continuous, p_continuous, N_J, N_P, A, P
)

pred_elapsed = time() - t_pred
val_log("Predicted probabilities computed in $(round(pred_elapsed, digits=1))s")


#############################
# Category Labels
#############################

cat_labels = ["Outside", "Cig", "Orig Ecig", "Flav Ecig", "Orig Bundle", "Flav Bundle"]


#############################
# Validation 1: Overall
# Category Shares
#############################

val_log("\n===================================")
val_log("Validation 1: Overall Category Shares")
val_log("===================================\n")

all_mask = trues(N_obs)
actual_all, predicted_all = compute_category_shares(y, probs, cat_idx, all_mask)

val_log(@sprintf("%-15s  %12s  %12s  %12s", "Category", "Actual", "Predicted", "Difference"))
val_log(repeat("-", 55))
for (c, label) in enumerate(cat_labels)
    diff = predicted_all[c] - actual_all[c]
    val_log(@sprintf("%-15s  %12.6f  %12.6f  %+12.6f", label, actual_all[c], predicted_all[c], diff))
end


#############################
# Validation 2: Category
# Shares by TYA Status
#############################

val_log("\n===================================")
val_log("Validation 2: Category Shares by TYA Status")
val_log("===================================")

for (tya_val, tya_label) in [(1, "No TYA"), (2, "TYA Present")]

    mask = tya_state .== tya_val
    N_sub = sum(mask)
    actual_tya, predicted_tya = compute_category_shares(y, probs, cat_idx, mask)

    val_log("\n  $tya_label (N = $N_sub):")
    val_log(@sprintf("  %-15s  %12s  %12s  %12s", "Category", "Actual", "Predicted", "Difference"))
    val_log("  " * repeat("-", 55))
    for (c, label) in enumerate(cat_labels)
        diff = predicted_tya[c] - actual_tya[c]
        val_log(@sprintf("  %-15s  %12.6f  %12.6f  %+12.6f", label, actual_tya[c], predicted_tya[c], diff))
    end
end


#############################
# Validation 3: Category
# Shares by Calendar Month
#############################

val_log("\n===================================")
val_log("Validation 3: Category Shares by Calendar Month")
val_log("===================================\n")

# Print header
val_log(@sprintf("%-8s  %6s  %8s %8s  %8s %8s  %8s %8s  %8s %8s  %8s %8s  %8s %8s",
    "Month", "N",
    "Out_A", "Out_P", "Cig_A", "Cig_P",
    "OE_A", "OE_P", "FE_A", "FE_P",
    "OB_A", "OB_P", "FB_A", "FB_P"))
val_log(repeat("-", 120))

# Store for output file
monthly_actual    = Matrix{Float64}(undef, N_periods, 6)
monthly_predicted = Matrix{Float64}(undef, N_periods, 6)
monthly_N         = Vector{Int}(undef, N_periods)

for t in 1:N_periods

    mask = period_idx .== t
    N_sub = sum(mask)
    monthly_N[t] = N_sub
    actual_t, predicted_t = compute_category_shares(y, probs, cat_idx, mask)
    monthly_actual[t, :]    = actual_t
    monthly_predicted[t, :] = predicted_t

    val_log(@sprintf("%-8s  %6d  %8.4f %8.4f  %8.4f %8.4f  %8.4f %8.4f  %8.4f %8.4f  %8.4f %8.4f  %8.4f %8.4f",
        period_labels[t], N_sub,
        actual_t[1], predicted_t[1],
        actual_t[2], predicted_t[2],
        actual_t[3], predicted_t[3],
        actual_t[4], predicted_t[4],
        actual_t[5], predicted_t[5],
        actual_t[6], predicted_t[6]))
end


#############################
# Validation 4: Alternative-
# Level Shares
#############################

val_log("\n===================================")
val_log("Validation 4: Alternative-Level Shares")
val_log("===================================\n")

actual_alt, predicted_alt = compute_alternative_shares(y, probs, N_J, all_mask)

# Alternative labels based on the ordering from CLAUDE.md
alt_labels = [
    "Outside option",
    "Cig 1-2 pks", "Cig 3-10 pks", "Cig 11-20 pks",
    "Cig 21-30 pks", "Cig 31-40 pks", "Cig 41+ pks",
    "Orig ecig 1-10", "Orig ecig 10-30", "Orig ecig 30+",
    "Flav ecig 0-10", "Flav ecig 10-30", "Flav ecig 30+",
    "Bndl orig lo-lo", "Bndl orig lo-hi", "Bndl orig hi-lo", "Bndl orig hi-hi",
    "Bndl flav lo-lo", "Bndl flav lo-hi", "Bndl flav hi-lo", "Bndl flav hi-hi"
]

val_log(@sprintf("%-4s  %-18s  %12s  %12s  %12s", "j", "Alternative", "Actual", "Predicted", "Difference"))
val_log(repeat("-", 65))
for j in 1:N_J
    diff = predicted_alt[j] - actual_alt[j]
    val_log(@sprintf("%-4d  %-18s  %12.6f  %12.6f  %+12.6f", j, alt_labels[j], actual_alt[j], predicted_alt[j], diff))
end


#############################
# Goodness-of-Fit Statistics
#############################

val_log("\n===================================")
val_log("Goodness-of-Fit Summary")
val_log("===================================\n")

# Category-level fit
cat_rmse = sqrt(mean((predicted_all .- actual_all).^2))
cat_mae  = mean(abs.(predicted_all .- actual_all))
cat_max  = maximum(abs.(predicted_all .- actual_all))
val_log(@sprintf("Category-level (6 categories):"))
val_log(@sprintf("  RMSE:          %.6f", cat_rmse))
val_log(@sprintf("  MAE:           %.6f", cat_mae))
val_log(@sprintf("  Max |diff|:    %.6f", cat_max))

# Alternative-level fit
alt_rmse = sqrt(mean((predicted_alt .- actual_alt).^2))
alt_mae  = mean(abs.(predicted_alt .- actual_alt))
alt_max  = maximum(abs.(predicted_alt .- actual_alt))
val_log(@sprintf("\nAlternative-level (21 alternatives):"))
val_log(@sprintf("  RMSE:          %.6f", alt_rmse))
val_log(@sprintf("  MAE:           %.6f", alt_mae))
val_log(@sprintf("  Max |diff|:    %.6f", alt_max))

# Monthly fit: average RMSE across months for each category
monthly_diffs = monthly_predicted .- monthly_actual
monthly_cat_rmse = sqrt.(mean(monthly_diffs.^2, dims=1))
val_log(@sprintf("\nMonthly category RMSE (averaged over %d months):", N_periods))
for (c, label) in enumerate(cat_labels)
    val_log(@sprintf("  %-15s  %.6f", label, monthly_cat_rmse[c]))
end

# Log-likelihood at θ_hat (for reference)
val_log("\n--- Log-Likelihood ---")
LL = 0.0
for i in 1:N_obs
    LL += log(max(probs[i, y[i]], 1e-300))
end
val_log(@sprintf("Log-likelihood:     %.4f", LL))
val_log(@sprintf("Avg log-likelihood: %.6f", LL / N_obs))


#############################
# Save Results
#############################

# Save detailed results to file
results_path = joinpath(output_dir, "Model_Validation_Results.txt")
open(results_path, "w") do io

    # Overall category shares
    println(io, "=== Overall Category Shares ===")
    println(io, join(["category", "actual", "predicted", "difference"], "\t"))
    for (c, label) in enumerate(cat_labels)
        println(io, join([label,
            @sprintf("%.10f", actual_all[c]),
            @sprintf("%.10f", predicted_all[c]),
            @sprintf("%.10f", predicted_all[c] - actual_all[c])], "\t"))
    end

    # Monthly time series
    println(io, "\n=== Monthly Category Shares ===")
    header = ["month", "N"]
    for label in cat_labels
        push!(header, "actual_$(label)", "predicted_$(label)")
    end
    println(io, join(header, "\t"))

    for t in 1:N_periods
        row = [period_labels[t], string(monthly_N[t])]
        for c in 1:6
            push!(row, @sprintf("%.10f", monthly_actual[t, c]))
            push!(row, @sprintf("%.10f", monthly_predicted[t, c]))
        end
        println(io, join(row, "\t"))
    end

    # Alternative-level shares
    println(io, "\n=== Alternative-Level Shares ===")
    println(io, join(["j", "alternative", "actual", "predicted", "difference"], "\t"))
    for j in 1:N_J
        println(io, join([string(j), alt_labels[j],
            @sprintf("%.10f", actual_alt[j]),
            @sprintf("%.10f", predicted_alt[j]),
            @sprintf("%.10f", predicted_alt[j] - actual_alt[j])], "\t"))
    end
end

val_log("\nResults saved to: $results_path")


#############################
# Log Final Timing
#############################

total_elapsed = time() - t_setup
val_log("\n===================================")
val_log("Model validation complete")
val_log(@sprintf("Total time: %.1fs", total_elapsed))
val_log("===================================")
val_log("Finished at $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")

# Close log file
close(val_log_io)
