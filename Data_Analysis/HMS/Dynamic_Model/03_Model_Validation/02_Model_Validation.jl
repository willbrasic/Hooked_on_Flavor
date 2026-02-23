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
#   4. Alternative-level shares: predicted vs actual for all 40 alternatives
#
# The predicted probabilities are computed by solving VFI at θ_hat and
# interpolating V_choice at each observation's continuous state.
#
# Results are saved to Model_Validation_Results.csv and
# Model_Validation_Log.txt.
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

# Set output path and working directory
if hpc

    # Load estimation functions and packages (must come first — provides Printf, CSV, etc.)
    include("../01_Functions.jl")

    # Load validation-specific functions
    include("01_Model_Validation_Functions.jl")

    # Construct psi and beta tags for directory and file naming
    ψ_naming, β_naming, _ = get_fixed_parameters()
    psi_tag = ESTIMATE_PSI ? "Psi_Estimated" : "Psi_$(ψ_naming)"
    beta_tag = ESTIMATE_BETA ? "Beta_Estimated" : "Beta_$(β_naming)"

    # Output path for results (use absolute path so it's unaffected by later cd)
    output_dir = abspath("./Model_Validation_$(psi_tag)_$(beta_tag)_Results")

    # Create output directory if it doesn't exist
    mkpath(output_dir)

    # Set working directory to where the data CSVs live
    cd("../../Data")
else

    # Load estimation functions and packages (must come first — provides Printf, CSV, etc.)
    include("../02_Second_Stage_Estimation/01_Functions.jl")

    # Load validation-specific functions
    include("01_Model_Validation_Functions.jl")

    # Construct psi and beta tags for directory and file naming
    ψ_naming, β_naming, _ = get_fixed_parameters()
    psi_tag = ESTIMATE_PSI ? "Psi_Estimated" : "Psi_$(ψ_naming)"
    beta_tag = ESTIMATE_BETA ? "Beta_Estimated" : "Beta_$(β_naming)"

    # Output path for results (includes psi and beta tags in directory name)
    output_dir = "C:/Users/wbras/OneDrive/Documents/Desktop/UA/4th_Year_Paper/4th_Year_Paper_Data/HMS/2021-Onward/Dynamic_Model/Model_Validation_$(psi_tag)_$(beta_tag)_Results"

    # Create output directory if it doesn't exist
    mkpath(output_dir)

    # Set working directory to where the data CSVs live
    cd("C:/Users/wbras/OneDrive/Documents/Desktop/UA/4th_Year_Paper/4th_Year_Paper_Data/HMS/2021-Onward/Dynamic_Model/Data")
end


#############################
# Output Paths
#############################

# Set log file path
log_path = joinpath(output_dir, "Model_Validation_Log.txt")

# Open log file for writing (log_io is defined as a global in 01_Functions.jl)
log_io = open(log_path, "w")

# Print and log model validation start time
log_msg("Model validation started at $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")


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

# Get period indices for time-series analysis
period_idx, period_labels, N_periods = get_period_indices();

# Log data setup completion time and sample size
N_obs = length(y)
setup_elapsed = time() - t_setup;
log_msg("Data loading complete in $(round(setup_elapsed, digits=1))s")
log_msg("Observations: $N_obs, Alternatives: $N_J, Periods: $N_periods")


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
estimates_path = joinpath(estimates_dir, "Dynamic_Model_$(psi_tag)_$(beta_tag)_Estimates.csv")
df_est = CSV.read(estimates_path, DataFrame)

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


#############################
# Compute Household Addiction
# Trajectories
#############################

# Print and log household addiction state computation header
log_msg("\n===================================")
log_msg("Computing household addiction states...")
log_msg("===================================")

t_states = time()

# Estimate initial addiction stocks via fixed-point iteration
a0, max_fp_iters = get_initial_addiction_stock(ψ, A, n, y, hh_codes)

# Print and log initial addiction stock result
log_msg("Initial addiction stocks: max fixed-point iterations = $max_fp_iters")

# Simulate addiction trajectories forward using observed choices
_, a_continuous = simulate_addiction_trajectories(N_A, ψ, A, n, y, hh_codes, a0)

# Print and log addiction state computation time
states_elapsed = time() - t_states
log_msg("Addiction states computed in $(round(states_elapsed, digits=1))s")


#############################
# Solve VFI at θ_hat
#############################

# Print and log VFI computation header
log_msg("\n===================================")
log_msg("Solving VFI at estimated parameters...")
log_msg("===================================")

t_vfi = time()

# Pre-compute addiction transitions
a_lower, a_upper, a_weight = precompute_addiction_transitions(N_J, N_A, ψ, A, n)

# Compute flow utility at θ_hat
U = get_flow_utility(
    θ_hat, N_J, N_A, N_Pcomb, A, q_cig, q_ecig, q_bundle, n, is_flavored, is_fda_flavored, cat_idx, E
)

# Solve VFI
_, V_choice, vfi_iters, vfi_converged = solve_vfi_sophisticated(
    N_J, N_A, N_P, N_Pcomb, β, δ, U,
    a_lower, a_upper, a_weight,
    p_cig_lo, p_cig_hi, p_cig_w,
    p_ecig_lo, p_ecig_hi, p_ecig_w,
    Π
)

# Print and log VFI convergence result
vfi_elapsed = time() - t_vfi
log_msg("VFI: $vfi_iters iterations, converged = $vfi_converged ($(round(vfi_elapsed, digits=1))s)")


#############################
# Compute Predicted
# Choice Probabilities
#############################

# Print and log predicted probability computation header
log_msg("\n===================================")
log_msg("Computing predicted choice probabilities...")
log_msg("===================================")

t_pred = time()

probs = compute_predicted_probs(
    V_choice, tya_state, a_continuous, p_continuous, N_J, N_P, A, P
)

# Print and log prediction computation time
pred_elapsed = time() - t_pred
log_msg("Predicted probabilities computed in $(round(pred_elapsed, digits=1))s")


#############################
# Category Labels
#############################

cat_labels = ["Outside", "Cig", "Orig Ecig", "Non-FDA Flav Ecig", "FDA Flav Ecig",
              "Orig Bundle", "Non-FDA Flav Bundle", "FDA Flav Bundle"]


#############################
# Validation 1: Overall
# Category Shares
#############################

# Print and log overall category shares header
log_msg("\n===================================")
log_msg("Validation 1: Overall Category Shares")
log_msg("===================================\n")

all_mask = trues(N_obs)
actual_all, predicted_all = compute_category_shares(y, probs, cat_idx, all_mask)

log_msg(@sprintf("%-15s  %12s  %12s  %12s", "Category", "Actual", "Predicted", "Difference"))
log_msg(repeat("-", 55))
for (c, label) in enumerate(cat_labels)
    diff = predicted_all[c] - actual_all[c]
    log_msg(@sprintf("%-15s  %12.6f  %12.6f  %+12.6f", label, actual_all[c], predicted_all[c], diff))
end


#############################
# Validation 2: Category
# Shares by TYA Status
#############################

# Print and log category shares by TYA status header
log_msg("\n===================================")
log_msg("Validation 2: Category Shares by TYA Status")
log_msg("===================================")

for (tya_vals, tya_label) in [([1, 2], "No TYA"), ([3, 4], "TYA Present")]

    mask = [tya_state[i] in tya_vals for i in eachindex(tya_state)]
    N_sub = sum(mask)
    actual_tya, predicted_tya = compute_category_shares(y, probs, cat_idx, mask)

    log_msg("\n  $tya_label (N = $N_sub):")
    log_msg(@sprintf("  %-15s  %12s  %12s  %12s", "Category", "Actual", "Predicted", "Difference"))
    log_msg("  " * repeat("-", 55))
    for (c, label) in enumerate(cat_labels)
        diff = predicted_tya[c] - actual_tya[c]
        log_msg(@sprintf("  %-15s  %12.6f  %12.6f  %+12.6f", label, actual_tya[c], predicted_tya[c], diff))
    end
end


#############################
# Validation 3: Category
# Shares by Calendar Month
#############################

# Print and log monthly category shares header
log_msg("\n===================================")
log_msg("Validation 3: Category Shares by Calendar Month")
log_msg("===================================\n")

# Print header
log_msg(@sprintf("%-8s  %6s  %8s %8s  %8s %8s  %8s %8s  %8s %8s  %8s %8s  %8s %8s  %8s %8s  %8s %8s",
    "Month", "N",
    "Out_A", "Out_P", "Cig_A", "Cig_P",
    "OE_A", "OE_P", "NFFE_A", "NFFE_P", "FFE_A", "FFE_P",
    "OB_A", "OB_P", "NFFB_A", "NFFB_P", "FFB_A", "FFB_P"))
log_msg(repeat("-", 160))

# Store for output file
monthly_actual    = Matrix{Float64}(undef, N_periods, 8)
monthly_predicted = Matrix{Float64}(undef, N_periods, 8)
monthly_N         = Vector{Int}(undef, N_periods)

for t in 1:N_periods

    mask = period_idx .== t
    N_sub = sum(mask)
    monthly_N[t] = N_sub
    actual_t, predicted_t = compute_category_shares(y, probs, cat_idx, mask)
    monthly_actual[t, :]    = actual_t
    monthly_predicted[t, :] = predicted_t

    log_msg(@sprintf("%-8s  %6d  %8.4f %8.4f  %8.4f %8.4f  %8.4f %8.4f  %8.4f %8.4f  %8.4f %8.4f  %8.4f %8.4f  %8.4f %8.4f  %8.4f %8.4f",
        period_labels[t], N_sub,
        actual_t[1], predicted_t[1],
        actual_t[2], predicted_t[2],
        actual_t[3], predicted_t[3],
        actual_t[4], predicted_t[4],
        actual_t[5], predicted_t[5],
        actual_t[6], predicted_t[6],
        actual_t[7], predicted_t[7],
        actual_t[8], predicted_t[8]))
end


#############################
# Validation 4: Alternative-
# Level Shares
#############################

# Print and log alternative-level shares header
log_msg("\n===================================")
log_msg("Validation 4: Alternative-Level Shares")
log_msg("===================================\n")

actual_alt, predicted_alt = compute_alternative_shares(y, probs, N_J, all_mask)

# Alternative labels matching the ordering: outside, 12 cig, 7 orig ecig,
# 7 non-FDA flav ecig, 7 FDA flav ecig, 2 orig bundle, 2 non-FDA flav bundle, 2 FDA flav bundle
alt_labels = [
    "Outside",
    "Cig 1 pk", "Cig 2 pks", "Cig 3-4 pks", "Cig 5-9 pks",
    "Cig 10 pks", "Cig 11-19 pks", "Cig 20 pks", "Cig 21-29 pks",
    "Cig 30 pks", "Cig 31-39 pks", "Cig 40 pks", "Cig 41+ pks",
    "OE 0-5", "OE 5-10", "OE 10-15", "OE 15-20", "OE 20-30", "OE 30-50", "OE 50+",
    "NFFE 0-5", "NFFE 5-10", "NFFE 10-15", "NFFE 15-20", "NFFE 20-30", "NFFE 30-50", "NFFE 50+",
    "FFE 0-5", "FFE 5-10", "FFE 10-15", "FFE 15-20", "FFE 20-30", "FFE 30-50", "FFE 50+",
    "Bndl orig lo", "Bndl orig hi",
    "Bndl NFF lo", "Bndl NFF hi",
    "Bndl FF lo", "Bndl FF hi"
]

log_msg(@sprintf("%-4s  %-18s  %12s  %12s  %12s", "j", "Alternative", "Actual", "Predicted", "Difference"))
log_msg(repeat("-", 65))
for j in 1:N_J
    diff = predicted_alt[j] - actual_alt[j]
    log_msg(@sprintf("%-4d  %-18s  %12.6f  %12.6f  %+12.6f", j, alt_labels[j], actual_alt[j], predicted_alt[j], diff))
end


#############################
# Goodness-of-Fit Statistics
#############################

# Print and log goodness-of-fit statistics
log_msg("\n===================================")
log_msg("Goodness-of-Fit Summary")
log_msg("===================================\n")

# Category-level fit
cat_rmse = sqrt(mean((predicted_all .- actual_all).^2))
cat_mae  = mean(abs.(predicted_all .- actual_all))
cat_max  = maximum(abs.(predicted_all .- actual_all))
log_msg(@sprintf("Category-level (8 categories):"))
log_msg(@sprintf("  RMSE:          %.6f", cat_rmse))
log_msg(@sprintf("  MAE:           %.6f", cat_mae))
log_msg(@sprintf("  Max |diff|:    %.6f", cat_max))

# Alternative-level fit
alt_rmse = sqrt(mean((predicted_alt .- actual_alt).^2))
alt_mae  = mean(abs.(predicted_alt .- actual_alt))
alt_max  = maximum(abs.(predicted_alt .- actual_alt))
log_msg(@sprintf("\nAlternative-level (40 alternatives):"))
log_msg(@sprintf("  RMSE:          %.6f", alt_rmse))
log_msg(@sprintf("  MAE:           %.6f", alt_mae))
log_msg(@sprintf("  Max |diff|:    %.6f", alt_max))

# Monthly fit: average RMSE across months for each category
monthly_diffs = monthly_predicted .- monthly_actual
monthly_cat_rmse = sqrt.(mean(monthly_diffs.^2, dims=1))
log_msg(@sprintf("\nMonthly category RMSE (averaged over %d months):", N_periods))
for (c, label) in enumerate(cat_labels)
    log_msg(@sprintf("  %-15s  %.6f", label, monthly_cat_rmse[c]))
end

# Log-likelihood at θ_hat (for reference)
log_msg("\n--- Log-Likelihood ---")
LL = 0.0
for i in 1:N_obs
    LL += log(max(probs[i, y[i]], 1e-300))
end
log_msg(@sprintf("Log-likelihood:     %.4f", LL))
log_msg(@sprintf("Avg log-likelihood: %.6f", LL / N_obs))


#############################
# Save Results
#############################

# Save detailed validation results to a CSV file
results_path = joinpath(output_dir, "Model_Validation_Results.csv")
open(results_path, "w") do io

    # Overall category shares
    println(io, "=== Overall Category Shares ===")
    println(io, join(["category", "actual", "predicted", "difference"], ","))
    for (c, label) in enumerate(cat_labels)
        println(io, join([label,
            @sprintf("%.10f", actual_all[c]),
            @sprintf("%.10f", predicted_all[c]),
            @sprintf("%.10f", predicted_all[c] - actual_all[c])], ","))
    end

    # Monthly time series
    println(io, "\n=== Monthly Category Shares ===")
    header = ["month", "N"]
    for label in cat_labels
        push!(header, "actual_$(label)", "predicted_$(label)")
    end
    println(io, join(header, ","))

    for t in 1:N_periods
        row = [period_labels[t], string(monthly_N[t])]
        for c in 1:8
            push!(row, @sprintf("%.10f", monthly_actual[t, c]))
            push!(row, @sprintf("%.10f", monthly_predicted[t, c]))
        end
        println(io, join(row, ","))
    end

    # Alternative-level shares
    println(io, "\n=== Alternative-Level Shares ===")
    println(io, join(["j", "alternative", "actual", "predicted", "difference"], ","))
    for j in 1:N_J
        println(io, join([string(j), alt_labels[j],
            @sprintf("%.10f", actual_alt[j]),
            @sprintf("%.10f", predicted_alt[j]),
            @sprintf("%.10f", predicted_alt[j] - actual_alt[j])], ","))
    end
end

# Print and log results save location
log_msg("\nResults saved to: $results_path")


#############################
# Log Final Timing
#############################

# Print and log final timing and completion message
total_elapsed = time() - t_setup
log_msg("\n===================================")
log_msg("Model validation complete")
log_msg(@sprintf("Total time: %.1fs", total_elapsed))
log_msg("===================================")
log_msg("Finished at $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")

# Close the log file handle
close(log_io)
