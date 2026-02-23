################################################################################
# William Brasic
# The University of Arizona
# wbrasic97@gmail.com
# February 2026
#
# Parallel Monte Carlo simulation via Slurm job arrays with β estimation.
#
# Each array task runs a SINGLE replication:
#   1. Reads replication number s from SLURM_ARRAY_TASK_ID (or command line)
#   2. Seeds RNG with s for reproducibility
#   3. Simulates household panel data from the true V_decision
#   4. Estimates θ via multi-start Nelder-Mead
#   5. Writes one result file: MC_Rep_<s>.csv
#
# Changes from 02_MC_Simulation_Array.jl:
#   - Includes 01_Functions_Beta.jl (4-state TYA transitions, β estimated)
#   - get_fixed_parameters() returns (ψ, δ) only (no β)
#   - Loads 4-state TYA via get_tya_states() instead of get_teen_young_adult()
#   - Loads Π_tya via get_tya_transitions()
#   - θ_true has 14 params (13 structural + β = 0.95 as present-bias DGP)
#   - starting_param has 14 params (13 structural + β = 0.90 as offset starting value)
#   - Passes Π_tya to solve_vfi in the DGP section
#
# After all tasks complete, run 03_MC_Aggregate_Results.jl to combine.
#
# Usage:
#   HPC:   sbatch 02_MC_Simulation_Array_Beta_Slurm.sb
#   Local: julia 02_MC_Simulation_Array_Beta.jl <replication_number>
################################################################################


#############################
# Preliminaries
#############################

# Whether we are running on the HPC or not
hpc = !Sys.iswindows()

# Set output path and working directory
if hpc

    # Load estimation functions and packages (must come first — provides Printf, CSV, etc.)
    include("../01_Functions_Beta.jl")

    # Load MC-specific functions
    include("01_MC_Simulation_Functions_Beta.jl")

    # Output path for results (use absolute path so it's unaffected by later cd)
    output_dir = abspath("./MC_Simulation_Beta_Results")

    # Set working directory to where the data CSVs live
    cd("../../Data")
else

    # Load estimation functions and packages (must come first — provides Printf, CSV, etc.)
    include("../02_Second_Stage_Estimation/01_Functions_Beta.jl")

    # Load MC-specific functions
    include("01_MC_Simulation_Functions_Beta.jl")

    # Output path for results
    output_dir = "C:/Users/wbras/OneDrive/Documents/Desktop/UA/4th_Year_Paper/4th_Year_Paper_Data/HMS/2021-Onward/Dynamic_Model/MC_Simulation_Beta_Results"

    # Set working directory to where the data CSVs live
    cd("C:/Users/wbras/OneDrive/Documents/Desktop/UA/4th_Year_Paper/4th_Year_Paper_Data/HMS/2021-Onward/Dynamic_Model/Data")
end


#############################
# Replication Number
#############################

# Get replication number from Slurm array task ID or command line argument
using Random

if haskey(ENV, "SLURM_ARRAY_TASK_ID")
    s = parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
elseif length(ARGS) >= 1
    s = parse(Int, ARGS[1])
else
    error("No replication number provided. Set SLURM_ARRAY_TASK_ID or pass as command line argument.")
end

# Seed RNG with replication number for reproducibility
Random.seed!(s)

println("Replication $s | seed = $s | PID = $(getpid())")


#############################
# Output Paths
#############################

# Create output directory if it doesn't exist
mkpath(output_dir)

# Timestamp for output files
timestamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")

# Zero-padded replication number (e.g., 01, 02, ..., 100)
s_str = lpad(s, ndigits(100), '0')

# Per-replication output files
results_path = joinpath(output_dir, "$(s_str)_MC_Rep_Results_$timestamp.csv")
log_path     = joinpath(output_dir, "$(s_str)_MC_Rep_Log_$timestamp.txt")
trace_path   = joinpath(output_dir, "$(s_str)_MC_Rep_Parameters_$timestamp.csv")

# Open log file for writing (log_io is defined as a global in 01_Functions_Beta.jl)
log_io = open(log_path, "w")

# Print and log MC replication start time
log_msg("MC Replication $s (β estimation) started at $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
log_msg("Random seed: $s")


#############################
# Initialize fixed parameters
#############################

# Load fixed parameters:
#   ψ = addiction decay rate (fixed from reduced-form AR(1) estimate)
#   δ = monthly discount factor (fixed at 0.99)
ψ, δ = get_fixed_parameters();


#############################
# State Spaces and Choices
#############################

# Start timer for data prep
t_setup = time();

# Addiction grid is created later using the fixed ψ from get_fixed_parameters.
# N_J, prices, consumption, etc. do not depend on ψ, so load them first.

# Get number of observations (N_HHT), number of alternatives (N_J), and choice matrix J
_, N_J, J = get_product_choices();

# Get number of product categories excluding the outside option
N_K, _ = get_category_choices();


#############################
# Alternative-Level Vectors
#############################

# Get consumption vectors by alternative (STANDARDIZED by max)
# c_bundle is standardized by its own max (not c_cig_max × c_ecig_max) for reasonable α_TE scaling
# Max values are needed for rescaling parameter estimates to original units
N_cig, N_orig_ecig, N_non_fda_flav_ecig, N_fda_flav_ecig, _, c_cig, c_ecig, c_bundle, c_cig_max, c_ecig_max, c_bundle_max = get_consumption(N_J);

# Get nicotine vector by alternative (STANDARDIZED by max)
# n_max is the raw max value for rescaling estimates
n, n_max = get_nicotine(N_J);

# Get category index by alternative
cat_idx = get_category_index(N_J, N_cig, N_orig_ecig, N_non_fda_flav_ecig, N_fda_flav_ecig);

# Get non-FDA flavored indicator by alternative: is_non_fda_flavored[j] ∈ {0, 1}
is_non_fda_flavored = get_non_fda_flavored_indicator(cat_idx);

# Get FDA flavored indicator by alternative: is_fda_flavored[j] ∈ {0, 1}
is_fda_flavored = get_fda_flavored_indicator(cat_idx);


#############################
# Price Space
#############################

# Get pricing grid: N_P points per category, P is N_P × 2 (cig, ecig)
N_P, P = get_pricing_spaces();

# Get all price combinations
N_Pcomb, Pcomb = get_pricing_spaces_combination(N_K, N_P, P);

# Get price ratios for quantity discount adjustment (price per unit varies by bin size)
ratio_cig, ratio_ecig = get_price_ratios(N_J, N_cig, N_orig_ecig, N_non_fda_flav_ecig, N_fda_flav_ecig, c_cig, c_ecig);

# Get expenditure matrix E[p, j] = p_cig(p) * c_cig[j] + p_ecig(p) * c_ecig[j]
# STANDARDIZED by E_max; E_max is the raw max value for rescaling estimates
E, E_max = get_expenditures(N_J, N_Pcomb, c_cig, c_ecig, c_cig_max, c_ecig_max, Pcomb, ratio_cig, ratio_ecig);

# Get Halton draw price transitions: T[m, r, k] where m = price state, r = draw, k = category
T = get_transitions(N_K);

# Pre-compute bilinear interpolation brackets and weights for price transitions
# Returns 6 matrices (M × R): lo/hi grid indices and weights for each category
p_cig_lo, p_cig_hi, p_cig_w, p_ecig_lo, p_ecig_hi, p_ecig_w = precompute_price_transitions(N_P, P, T);

# Log data setup completion time
setup_elapsed = time() - t_setup;
log_msg("Data loading complete in $(round(setup_elapsed, digits=1))s");


#############################
# TYA States and Transitions
#############################

# Load 4-state TYA assignments from first-stage estimation
# States: 1=no TYA stable, 2=no TYA approaching, 3=TYA stable, 4=TYA ending
real_tya_state = get_tya_states();

# Load 4×4 monthly TYA transition matrix from first-stage estimation
Π_tya = get_tya_transitions();

# Print and log TYA state and transition matrix information
log_msg("TYA states loaded: $(length(unique(real_tya_state))) unique states")
log_msg("TYA transition matrix:")
for row in 1:4
    log_msg("  State $row → " * join([@sprintf("%.4f", Π_tya[row, col]) for col in 1:4], "  "))
end


#############################
# Real Household Data
# (for design-based MC)
#############################

# Load real household identifiers for design-based MC simulation
# The MC simulation conditions on real observables (prices, TYA, panel structure)
# and only simulates choices from the model. This provides realistic
# cross-sectional price variation needed to identify all parameters.
real_hh_codes = get_hh_codes();

# Map observed household prices to continuous values for likelihood interpolation
_, real_p_continuous = map_prices_to_grid(N_P, P, Pcomb);

N_HH_real = length(unique(real_hh_codes));
N_obs_real = length(real_hh_codes);

# Print and log real data summary
log_msg("Real data loaded: $N_HH_real households, $N_obs_real observations")


#############################
# True Parameters
#############################

# Static logit estimates (ORIGINAL units — utils per pack, per mL, etc.)
# These are converted to standardized units using the max values from the data.
# μ, γ have no static counterpart and are set to reasonable starting values.
# ψ is fixed from the reduced-form AR(1) estimate (not estimated).
# β = 0.95 is the true present-bias parameter to be recovered.
α_T_orig  =  0.0187
α_E_orig  =  0.0096
α_TE_orig = -0.0021
ω_orig    = -0.0055

θ_true = (
    α_T  = α_T_orig * c_cig_max,
    α_E  = α_E_orig * c_ecig_max,
    α_TE = α_TE_orig * c_bundle_max,
    λ_1  =  0.852,
    λ_2  =  0.703,
    λ_3  =  0.5,       # FDA flavor baseline (no conversion)
    λ_4  =  0.5,       # FDA flavor × TYA (no conversion)
    μ    =  0.10,      # no static counterpart
    γ    = -0.10,      # no static counterpart
    ω    = ω_orig * E_max,
    ξ_T  = -3.573,
    ξ_E  = -6.500,
    ξ_TE = -5.288,
    β    =  0.95       # present bias (1 = exponential, <1 = present-biased)
);

# Convert to vector for VFI
θ_true_vec = collect(Float64, values(θ_true));

# Parameter names for output
param_names = collect(String, string.(keys(θ_true)))

# Number of parameters
N_params = length(θ_true)

# Save θ_true for aggregate script (each rep overwrites, but values are identical)
θ_true_df = DataFrame(parameter = param_names, value = θ_true_vec)
CSV.write(joinpath(output_dir, "MC_True_Parameters.csv"), θ_true_df)

# Print and log true parameters
log_msg("θ_true (standardized): $θ_true")


#############################
# MC Settings
#############################

# Optimizer settings (same across all replications)
L          = 2        # Number of random restarts (outer tries)
M          = 2        # Short Nelder-Mead runs per outer try (inner tries)
inner_iter = 100      # Max iterations per short run

# Print and log MC settings and true parameters
log_msg("MC settings: s=$s, N_HH=$N_HH_real, N_obs=$N_obs_real, L=$L, M=$M, inner_iter=$inner_iter")
log_msg("True parameters:")
for k in 1:N_params
    log_msg("  $(param_names[k]) = $(θ_true_vec[k])")
end


#############################
# Solve DGP Value Function
#############################

# Print and log DGP VFI header
log_msg("\n===================================")
log_msg("Solving VFI at true parameters...")
log_msg("===================================")

t_vfi = time()

# Get number of addiction states (N_A = 20) and the normalized addiction grid A
N_A, A = get_addiction_space(ψ);

# Compute flow utility at true θ (first 13 elements = structural parameters)
U_true = get_flow_utility(
    θ_true_vec[1:end-1], N_J, N_A, N_Pcomb, A, c_cig, c_ecig, c_bundle, n, is_non_fda_flavored, is_fda_flavored, cat_idx, E
);

# Compute addiction transition brackets at fixed ψ
a_lower_true, a_upper_true, a_weight_true = precompute_addiction_transitions(
    N_J, N_A, ψ, A, n
);

# Solve VFI at true parameters with true β and TYA transitions (cold start)
β_true = θ_true.β
_, V_decision_true, vfi_iters_true, _ = solve_vfi_sophisticated(
    N_J, N_A, N_P, N_Pcomb, β_true, δ, U_true,
    a_lower_true, a_upper_true, a_weight_true,
    p_cig_lo, p_cig_hi, p_cig_w,
    p_ecig_lo, p_ecig_hi, p_ecig_w,
    Π_tya
);

vfi_elapsed = time() - t_vfi

# Print and log DGP VFI result
log_msg("DGP VFI converged in $vfi_iters_true iterations ($(round(vfi_elapsed, digits=1))s)")


#############################
# Starting Values for
# Estimation
#############################

# Starting values for estimation (STANDARDIZED units)
# Deliberately offset from θ_true to test optimizer recovery.
# ψ is fixed and not estimated.
# β starts at 0.90 (offset from true β = 0.95).
starting_param = (
    α_T  =  0.5,
    α_E  =  0.3,
    α_TE =  0.1,
    λ_1  =  0.4,
    λ_2  =  0.3,
    λ_3  =  0.3,      # FDA flavor baseline (offset from truth)
    λ_4  =  0.2,      # FDA flavor × TYA (offset from truth)
    μ    =  0.1,      # no static counterpart
    γ    = -0.1,      # no static counterpart
    ω    = -1.0,
    ξ_T  = -2.0,
    ξ_E  = -4.0,
    ξ_TE = -3.0,
    β    =  0.90      # present bias starting value (offset from 0.95)
);

# Initial simplex deviations for Nelder-Mead
# Scaled to ~50% of the absolute value of each starting parameter;
# μ and γ get larger deviations since they have no informed starting values.
# β gets a deviation of 0.10 to explore the [0.01, 1.00] range.
# ψ is fixed and not included.
add = [
    abs(starting_param.α_T)  * 0.50,   # α_T
    abs(starting_param.α_E)  * 0.50,   # α_E
    abs(starting_param.α_TE) * 0.50,   # α_TE
    abs(starting_param.λ_1)  * 0.50,   # λ_1
    abs(starting_param.λ_2)  * 0.50,   # λ_2
    abs(starting_param.λ_3)  * 0.50,   # λ_3
    abs(starting_param.λ_4)  * 0.50,   # λ_4
    abs(starting_param.μ)    * 1.00,   # μ
    abs(starting_param.γ)    * 1.00,   # γ
    abs(starting_param.ω)    * 0.50,   # ω
    abs(starting_param.ξ_T)  * 0.50,   # ξ_T
    abs(starting_param.ξ_E)  * 0.50,   # ξ_E
    abs(starting_param.ξ_TE) * 0.50,   # ξ_TE
    0.10                                # β
];


#############################
# Run Single Replication
#############################

# Print and log replication header
log_msg("\n===================================")
log_msg("Running Replication $s")
log_msg("===================================\n")

# Open parameter trace file and write header
global param_trace_io = open(trace_path, "w")
trace_header = "sim,outer_try,inner_run,eval,NLL,VFI_iters," * join(param_names, ",")
println(param_trace_io, trace_header)
flush(param_trace_io)

t_rep = time()

# Simulate choices from the true DGP using real observables
t_sim = time()
global y_sim, tya_state_sim, p_continuous_sim, hh_codes_sim
y_sim, tya_state_sim, p_continuous_sim, hh_codes_sim = simulate_data(
    V_decision_true, ψ, N_J, N_P, A, P, n,
    real_p_continuous, real_tya_state, real_hh_codes
)
sim_elapsed = time() - t_sim

# Print and log data simulation result
log_msg("Data simulation: $(length(y_sim)) obs in $(round(sim_elapsed, digits=2))s")

# Print and log simulated choice shares by category
N_sim = length(y_sim)
cat_counts = zeros(Int, 8)  # cats 0-7
for i in 1:N_sim
    cat_counts[cat_idx[y_sim[i]] + 1] += 1
end
log_msg("Simulated choice shares:")
cat_labels = ["outside", "cig", "orig_ecig", "non_fda_flav_ecig", "fda_flav_ecig", "orig_bundle", "non_fda_flav_bundle", "fda_flav_bundle"]
for c in 1:8
    log_msg(@sprintf("  %-22s  %6d / %d  (%5.2f%%)", cat_labels[c], cat_counts[c], N_sim, 100.0 * cat_counts[c] / N_sim))
end

# Reset evaluation counter and set replication number
global eval_count = 0
global current_replication = s

# Estimate θ via multi-start Nelder-Mead
t_est = time()
opt_param, opt_value = random_amoeba(
    objective_mc, starting_param, add, L, M;
    inner_iter = inner_iter
)
est_elapsed = time() - t_est

# Print and log estimation result
log_msg("Estimation complete: $eval_count evaluations in $(round(est_elapsed, digits=1))s")

# Get estimated parameters as vector
θ_hat = collect(Float64, values(opt_param))

rep_elapsed = time() - t_rep


#############################
# Save Results
#############################

# Save per-replication results to a CSV file
open(results_path, "w") do io
    println(io, "S,NLL," * join(param_names, ","))
    println(io, "$s,$(@sprintf("%.10f", opt_value))," * join([@sprintf("%.10f", θ_hat[k]) for k in 1:N_params], ","))
end

# Print and log replication results table
log_msg("\nReplication $s results ($(round(rep_elapsed, digits=1))s total):")
log_msg("  Negative log-likelihood: $(round(opt_value, digits=4))")
log_msg(@sprintf("  %-8s  %12s  %12s  %12s", "Param", "True", "Estimated", "Diff"))
log_msg("  " * repeat("-", 50))
for k in 1:N_params
    diff = θ_hat[k] - θ_true_vec[k]
    log_msg(@sprintf("  %-8s  %12.6f  %12.6f  %12.6f", param_names[k], θ_true_vec[k], θ_hat[k], diff))
end

# Print and log results save location
log_msg("\nResults saved to: $results_path")

# Print and log replication finished message
log_msg("MC Replication $s finished at $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")

# Close the log and trace file handles
close(param_trace_io)
close(log_io)
