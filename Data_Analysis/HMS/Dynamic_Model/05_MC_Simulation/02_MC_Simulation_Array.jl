################################################################################
# William Brasic
# The University of Arizona
# wbrasic97@gmail.com
# February 2026
#
# Parallel Monte Carlo simulation via Slurm job arrays.
#
# Set ESTIMATE_BETA = true to estimate β (present bias) as a structural
# parameter. Set ESTIMATE_BETA = false for the base model with β fixed at 1.0.
#
# Each array task runs a SINGLE replication:
#   1. Reads replication number s from SLURM_ARRAY_TASK_ID (or command line)
#   2. Seeds random number generator with s for reproducibility
#   3. Simulates household panel data from the true V_decision
#   4. Estimates θ 
#   5. Writes one result file: MC_Rep_<s>.csv
#
# After all tasks complete, run 03_MC_Aggregate_Results.jl to combine.
#
# Usage:
#   HPC:   sbatch 02_MC_Simulation_Array_Slurm.sb
#   Local: julia 02_MC_Simulation_Array.jl <replication_number>
################################################################################


#############################
# Preliminaries
#############################

# Set to true to estimate β (present bias) as a structural parameter
ESTIMATE_BETA = false

# Set to true to estimate ψ (addiction decay rate) as a structural parameter
ESTIMATE_PSI = false

# Set to true to warm-start VFI from the previous evaluation's converged V
WARM_START = true

# Detect whether we are running on the HPC (any non-Windows system)
HPC = !Sys.iswindows()

# Set output path and working directory
if HPC

    # Load estimation functions and packages (must come first — provides Printf, CSV, etc.)
    include("../01_Functions.jl")

    # Load MC-specific functions
    include("01_MC_Simulation_Functions.jl")

    # Construct psi and beta tags for output directory and file naming.
    ψ_naming, β_naming, _ = get_fixed_parameters()
    psi_tag = ESTIMATE_PSI ? "Psi_Estimated" : "Psi_$(ψ_naming)"
    beta_tag = ESTIMATE_BETA ? "Beta_Estimated" : "Beta_$(β_naming)"

    # Output path for results (use absolute path so it's unaffected by later cd)
    output_dir = abspath("./MC_Simulation_$(psi_tag)_$(beta_tag)_Results")

    # Set working directory to where the data CSVs live
    cd("../../Data")
else

    # Load estimation functions and packages (must come first — provides Printf, CSV, etc.)
    include("../02_Second_Stage_Estimation/01_Functions.jl")

    # Load MC-specific functions
    include("01_MC_Simulation_Functions.jl")

    # Construct psi and beta tags for output directory and file naming.
    ψ_naming, β_naming, _ = get_fixed_parameters()
    psi_tag = ESTIMATE_PSI ? "Psi_Estimated" : "Psi_$(ψ_naming)"
    beta_tag = ESTIMATE_BETA ? "Beta_Estimated" : "Beta_$(β_naming)"

    # Output path for results (local Windows path, includes psi and beta tags in directory name)
    output_dir = "C:/Users/wbras/OneDrive/Documents/Desktop/UA/4th_Year_Paper/4th_Year_Paper_Data/HMS/2021-Onward/Dynamic_Model/MC_Simulation_$(psi_tag)_$(beta_tag)_Results"

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


#############################
# Output Paths
#############################

# Create output directory if it doesn't exist
mkpath(output_dir)

# Timestamp for output files
timestamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")

# Zero-padded replication number (e.g., 01, 02, ..., 100)
s_str = lpad(s, ndigits(100), '0')

# Per-replication output files (include beta tag after "MC_" for identification)
results_path = joinpath(output_dir, "$(s_str)_MC_$(psi_tag)_$(beta_tag)_Rep_Results_$timestamp.csv")
log_path     = joinpath(output_dir, "$(s_str)_MC_$(psi_tag)_$(beta_tag)_Rep_Log_$timestamp.txt")
trace_path   = joinpath(output_dir, "$(s_str)_MC_$(psi_tag)_$(beta_tag)_Rep_Parameters_$timestamp.csv")

# Open log file for writing (log_io is defined as a global in 01_Functions.jl)
log_io = open(log_path, "w")

# Print and log MC replication start time
log_msg("MC Replication $s ($(psi_tag), $(beta_tag)) started at $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
log_msg("")
log_msg("Random seed: $s")
log_msg("")


#############################
# Initialize fixed parameters
#############################

# Load fixed parameters:
#   ψ = addiction decay rate (fixed at 0.68; overridden by optimizer when ESTIMATE_PSI = true)
#   β = present bias (fixed at 1.0; overridden by optimizer when ESTIMATE_BETA = true)
#   δ = monthly discount factor (fixed at 0.99)
ψ, β, δ = get_fixed_parameters();


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
# Max values are needed for rescaling parameter estimates to original units
N_cig, N_orig_ecig, N_non_fda_flav_ecig, N_fda_flav_ecig, _, q_cig, q_ecig, q_bundle, q_cig_max, q_ecig_max, q_bundle_max = get_consumption(N_J);

# Get nicotine vector by alternative (STANDARDIZED by max)
# n_max is the raw max value for rescaling estimates
n, n_max = get_nicotine(N_J);

# Get category index by alternative
cat_idx = get_category_index(N_J, N_cig, N_orig_ecig, N_non_fda_flav_ecig, N_fda_flav_ecig);

# Get flavored indicator by alternative
is_flavored = get_flavored_indicator(cat_idx);

# Get FDA flavored indicator by alternative
is_fda_flavored = get_fda_flavored_indicator(cat_idx);


#############################
# Price Space
#############################

# Get pricing grid: N_P points per category
N_P, P = get_pricing_spaces();

# Get all price combinations
N_Pcomb, Pcomb = get_pricing_spaces_combination(N_K, N_P, P);

# Get price ratios for quantity discount adjustment (price per unit varies by bin size)
ratio_cig, ratio_ecig = get_price_ratios(N_J, N_cig, N_orig_ecig, N_non_fda_flav_ecig, N_fda_flav_ecig, q_cig, q_ecig);

# Get expenditure matrix E[p, j] = p_cig(p) × q_cig[j] + p_ecig(p) × q_ecig[j]
# STANDARDIZED by E_max; E_max is the raw max value for rescaling estimates
E, E_max = get_expenditures(N_J, N_Pcomb, q_cig, q_ecig, q_cig_max, q_ecig_max, Pcomb, ratio_cig, ratio_ecig);

# Get Halton draw price transitions: T[m, r, k] where m = price state, r = draw, k = category
T = get_transitions(N_K);

# Pre-compute bilinear interpolation brackets and weights for price transitions
p_cig_lo, p_cig_hi, p_cig_w, p_ecig_lo, p_ecig_hi, p_ecig_w = precompute_price_transitions(N_P, P, T);

# Print and log data setup completion time
setup_elapsed = time() - t_setup;
log_msg("Data loading complete in $(round(setup_elapsed, digits = 1))s")
log_msg("")


#############################
# Real Household Data
#############################

# Load real household identifiers for design-based MC simulation
real_hh_codes = get_hh_codes();

# TYA states
real_tya_state = get_tya_states();

# TYA transition matrix 
Π = get_tya_transitions();

# Map observed household prices to continuous values for likelihood interpolation
_, real_p_continuous = map_prices_to_grid(N_P, P, Pcomb);

# Number of unique households and number of observations
N_HH_real = length(unique(real_hh_codes));
N_obs_real = length(real_hh_codes);

# Print and log real data summary
log_msg("Real data loaded: $N_HH_real households, $N_obs_real observations");
log_msg("");


#############################
# True Parameters
#############################

# Static logit estimates in ORIGINAL units
# μ, γ have no static counterpart and are set to reasonable starting values
α_C_orig  =  0.0186
α_E_orig  =  0.0091
α_CE_orig = -0.0016
ω_orig    = -0.0055

θ_true = (
    α_C  = α_C_orig * q_cig_max,
    α_E  = α_E_orig * q_ecig_max,
    α_CE = α_CE_orig * q_bundle_max,
    λ_1  =  0.1554,
    λ_2  =  0.6616,
    λ_3  =  -0.0975,    
    λ_4  =  -0.3359,    
    γ    = -0.10,          
    μ    =  0.05,           
    ω    = ω_orig * E_max,
    ξ_C  = -2.007,
    ξ_E  = -6.0294,
    ξ_CE = -5.2096
)

# When estimating ψ, append the true addiction decay rate
if ESTIMATE_PSI
    θ_true = merge(θ_true, (ψ = 0.68,))
end

# When estimating β, append the true present-bias parameter (always last)
if ESTIMATE_BETA
    θ_true = merge(θ_true, (β = 0.75,))
end

# Convert to vector for VFI
θ_true_vec = collect(Float64, values(θ_true));

# Parameter names for output (ASCII for Excel CSV compatibility)
param_names = replace.(collect(String, string.(keys(θ_true))),
    "α" => "alpha", "λ" => "lambda", "γ" => "gamma",
    "μ" => "mu", "ω" => "omega", "ξ" => "xi")

# Number of parameters
N_params = length(θ_true)

# Write θ_true to csv (transposed: parameter names as columns, values as a single row)
θ_true_df = DataFrame([name => [round(val, digits=4)] for (name, val) in zip(param_names, θ_true_vec)]...)
CSV.write(joinpath(output_dir, "MC_$(psi_tag)_$(beta_tag)_True_Parameters.csv"), θ_true_df)

# Print and log true parameters
log_msg("θ_true (standardized): (" * join(["$(k) = $(@sprintf("%.4f", v))" for (k, v) in pairs(θ_true)], ", ") * ")")
log_msg("")


#############################
# MC Settings
#############################

# Optimizer settings (same across all replications)
L          = 50        # Number of random restarts (outer tries)
M          = 20        # Short Nelder-Mead runs per outer try (inner tries)
inner_iter = 100       # Max iterations per short run

# Print and log MC settings and true parameters
log_msg("MC settings: s=$s, N_HH=$N_HH_real, N_obs=$N_obs_real, L=$L, M=$M, inner_iter=$inner_iter")
log_msg("")
log_msg("True parameters:")
log_msg("")
for k in 1:N_params
    log_msg(@sprintf("  %s = %.4f", param_names[k], θ_true_vec[k]))
end
log_msg("")


#############################
# Solve DGP Value Function
#############################

# Print and log DGP VFI header
log_msg("\n===================================")
log_msg("Solving VFI at true parameters...")
log_msg("===================================")
log_msg("")

# Start vfi timer 
t_vfi = time()

# Extract ψ_true: from θ_true when ESTIMATE_PSI, otherwise from get_fixed_parameters()
ψ_true = ESTIMATE_PSI ? θ_true.ψ : ψ

# Get number of addiction states (N_A = 20) and the normalized addiction grid A
N_A, A = get_addiction_space(ψ_true);

# Determine which elements of θ_true_vec are the structural parameters (excludes ψ and β)
if ESTIMATE_BETA && ESTIMATE_PSI
    θ_struct_true = θ_true_vec[1:end-2]
    β_true = θ_true.β
elseif ESTIMATE_BETA
    θ_struct_true = θ_true_vec[1:end-1]
    β_true = θ_true.β
elseif ESTIMATE_PSI
    θ_struct_true = θ_true_vec[1:end-1]
    β_true = β
else
    θ_struct_true = θ_true_vec
    β_true = β
end

# Compute flow utility at true θ (structural parameters only, excludes β and ψ)
U_true = get_flow_utility(
    θ_struct_true, N_J, N_A, N_Pcomb, A, q_cig, q_ecig, q_bundle, n, is_flavored, is_fda_flavored, cat_idx, E
);

# Compute addiction transition brackets at ψ_true
a_lower_true, a_upper_true, a_weight_true = precompute_addiction_transitions(
    N_J, N_A, ψ_true, A, n
);

# Solve VFI at true parameters (cold start)
_, V_decision_true, vfi_iters_true, _ = solve_vfi_sophisticated(
    N_J, N_A, N_P, N_Pcomb, β_true, δ, U_true,
    a_lower_true, a_upper_true, a_weight_true,
    p_cig_lo, p_cig_hi, p_cig_w,
    p_ecig_lo, p_ecig_hi, p_ecig_w,
    Π
);

# End vfi timer 
vfi_elapsed = time() - t_vfi

# Print and log DGP VFI result
log_msg("DGP VFI converged in $vfi_iters_true iterations ($(round(vfi_elapsed, digits = 1))s)")


#############################
# Pre-compute Addiction
# Transitions (at Fixed ψ)
#############################

# Pre-compute addiction transition brackets for all (alternative, addiction state) pairs.
# When ESTIMATE_PSI = false, these are used by every objective_mc evaluation (fixed ψ).
# When ESTIMATE_PSI = true, these are needed for DGP but objective_mc() recomputes
# them at each candidate ψ.
mc_a_lower_fixed, mc_a_upper_fixed, mc_a_weight_fixed = precompute_addiction_transitions(N_J, N_A, ψ_true, A, n)


#############################
# Starting Values for
# Estimation
#############################

# Starting values for estimation (STANDARDIZED units)
starting_param = map(x -> 0.5x, θ_true)

# Note: starting_param = map(x -> 0.5x, θ_true) already includes ψ and β
# at 50% of their true values when ESTIMATE_PSI / ESTIMATE_BETA are true,
# since they are part of θ_true.

# Initial simplex deviations for Nelder-Mead
# Scaled to ~50% of the absolute value of each starting parameter
# γ and μ get larger deviations since they have no informed starting values.
add = [
    abs(starting_param.α_C)  * 0.50,   # α_C
    abs(starting_param.α_E)  * 0.50,   # α_E
    abs(starting_param.α_CE) * 0.50,   # α_CE
    abs(starting_param.λ_1)  * 0.50,   # λ_1
    abs(starting_param.λ_2)  * 0.50,   # λ_2
    abs(starting_param.λ_3)  * 0.50,   # λ_3
    abs(starting_param.λ_4)  * 0.50,   # λ_4
    abs(starting_param.γ)    * 1.00,   # γ
    abs(starting_param.μ)    * 1.00,   # μ
    abs(starting_param.ω)    * 0.50,   # ω
    abs(starting_param.ξ_C)  * 0.50,   # ξ_C
    abs(starting_param.ξ_E)  * 0.50,   # ξ_E
    abs(starting_param.ξ_CE) * 0.50    # ξ_CE
]

# When estimating ψ, append its simplex deviation (after 13 structural)
if ESTIMATE_PSI
    push!(add, 0.20)  # ψ: deviation of 0.20 around starting value
end

# When estimating β, append its simplex deviation (always last)
if ESTIMATE_BETA
    push!(add, 0.20)  # β: deviation of 0.20 around starting value
end


#############################
# Run Single Replication
#############################

# Print and log replication header
log_msg("\n===================================")
log_msg("Running Replication $s")
log_msg("===================================\n")

# Open parameter trace file and write header
param_trace_io = open(trace_path, "w")
trace_header = "sim,outer_try,inner_run,eval,NLL,VFI_iters," * join(param_names, ",")
println(param_trace_io, trace_header)
flush(param_trace_io)

# Start timer for MC simulation 
t_rep = time()

# Simulate choices from the true DGP using real observables
t_sim = time()
global y_sim, tya_state_sim, p_continuous_sim, hh_codes_sim
y_sim, tya_state_sim, p_continuous_sim, hh_codes_sim = simulate_data(
    V_decision_true, ψ_true, N_J, N_P, A, P, n,
    real_p_continuous, real_tya_state, real_hh_codes
)
sim_elapsed = time() - t_sim

# Print and log data used in simulation 
log_msg("Data simulation: $(length(y_sim)) obs in $(round(sim_elapsed, digits=2))s")
log_msg("")

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

# Pre-compute addiction trajectories for this replication's simulated choices.
# When ESTIMATE_PSI = false, these are used by every objective_mc evaluation.
# When ESTIMATE_PSI = true, objective_mc() recomputes them at each candidate ψ.
mc_a0_fixed, _ = get_initial_addiction_stock(ψ_true, A, n, y_sim, hh_codes_sim)
_, mc_a_continuous_fixed = simulate_addiction_trajectories(N_A, ψ_true, A, n, y_sim, hh_codes_sim, mc_a0_fixed)

# Reset evaluation counter and set replication number
eval_count = 0
current_replication = s

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

# End MC simulation timer 
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

# Close the parameter trace file handle
close(param_trace_io)

# Close the log file handle
close(log_io)
