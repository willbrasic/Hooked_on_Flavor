################################################################################
# William Brasic
# The University of Arizona
# wbrasic97@gmail.com
# February 2026
#
# Parallel Monte Carlo simulation via Slurm job arrays.
#
# Each array task runs a SINGLE replication:
#   1. Reads replication number s from SLURM_ARRAY_TASK_ID (or command line)
#   2. Seeds RNG with s for reproducibility
#   3. Simulates household panel data from the true V_decision
#   4. Estimates θ via multi-start Nelder-Mead (objective_mc)
#   5. Writes one result file: MC_Rep_<s>.txt
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

# Whether we are running on the HPC or not
hpc = !Sys.iswindows()

# Set output path and working directory
if hpc

    # Load estimation functions and packages (must come first — provides Printf, CSV, etc.)
    include("../01_Functions.jl")

    # Load MC-specific functions
    include("01_MC_Simulation_Functions.jl")

    # Output path for results (use absolute path so it's unaffected by later cd)
    output_dir = abspath("./MC_Simulation_Results")

    # Set working directory to where the data CSVs live
    cd("../../Data")
else

    # Load estimation functions and packages (must come first — provides Printf, CSV, etc.)
    include("../02_Second_Stage_Estimation/01_Functions.jl")

    # Load MC-specific functions
    include("01_MC_Simulation_Functions.jl")

    # Output path for results
    output_dir = "C:/Users/wbras/OneDrive/Documents/Desktop/UA/4th_Year_Paper/4th_Year_Paper_Data/HMS/2021-Onward/Dynamic_Model/MC_Simulation_Results"

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

# Timestamp for output files
timestamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")

# Zero-padded replication number (e.g., 01, 02, ..., 100)
s_str = lpad(s, ndigits(100), '0')

# Per-replication output files
results_path = joinpath(output_dir, "$(s_str)_MC_Rep_Results_$timestamp.txt")
log_path     = joinpath(output_dir, "$(s_str)_MC_Rep_Log_$timestamp.txt")
trace_path   = joinpath(output_dir, "$(s_str)_MC_Rep_Parameters_$timestamp.txt")

# Open log file and set global handle
global mc_log_io = open(log_path, "w")
mc_log("MC Replication $s started at $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
mc_log("Random seed: $s")


#############################
# Initialize fixed parameters
#############################

# Load fixed parameters (ψ is now estimated; only β and δ are fixed)
_, β, δ = get_fixed_parameters();


#############################
# State Spaces and Choices
#############################

# Start timing
t_setup = time();

# Addiction grid is created later using the true ψ from θ_true (defined below).
# N_J, prices, consumption, etc. do not depend on ψ, so load them first.

# Get number of alternatives (N_J) and choice matrix (J)
_, N_J, J = get_product_choices();

# Get number of categories excluding outside option (N_K)
N_K, _ = get_category_choices();


#############################
# Alternative-Level Vectors
#############################

# Get consumption vectors by alternative (STANDARDIZED by max)
# c_bundle is standardized by its own max (not c_cig_max × c_ecig_max)
N_cig, N_orig_ecig, N_flav_ecig, _, c_cig, c_ecig, c_bundle, c_cig_max, c_ecig_max, c_bundle_max = get_consumption(N_J);

# Get nicotine vector by alternative (STANDARDIZED by max)
n, n_max = get_nicotine(N_J);

# Get category index by alternative: cat_idx[j] ∈ {0, 1, 2, 3, 4, 5}
cat_idx = get_category_index(N_J, N_cig, N_orig_ecig, N_flav_ecig);

# Get flavored indicator by alternative: is_flavored[j] ∈ {true, false}
is_flavored = get_flavored_indicator(cat_idx);


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

setup_elapsed = time() - t_setup;
mc_log("Data loading complete in $(round(setup_elapsed, digits=1))s");


#############################
# Real Household Data
# (for design-based MC)
#############################

# Load real prices, TYA, and household codes from the actual data.
# The MC simulation conditions on real observables (prices, TYA, panel structure)
# and only simulates choices from the model. This provides realistic
# cross-sectional price variation needed to identify all parameters.
real_hh_codes = get_hh_codes()
_, real_tya = get_teen_young_adult()
real_tya_state = get_tya_state(real_tya)
_, real_p_continuous = map_prices_to_grid(N_P, P, Pcomb)

N_HH_real = length(unique(real_hh_codes))
N_obs_real = length(real_hh_codes)
mc_log("Real data loaded: $N_HH_real households, $N_obs_real observations")


#############################
# True Parameters
#############################

# True structural parameters for the DGP (STANDARDIZED units)
#
# The MC simulation uses real observed prices (from Prices.csv) rather than
# simulating from the AR(1) process. This provides realistic cross-sectional
# price variation (σ_between ≈ $1.22 for cig), which is necessary to identify
# ω separately from α_T and α_E.
#
# μ and γ are dynamic-only parameters (no static logit counterpart).
# γ < 0 represents withdrawal cost (standard addiction model).
# ψ is the addiction decay rate (estimated jointly with structural parameters).
θ_true = (
    α_T  =  0.46,     # ≈ 0.0077 × c_cig_max
    α_E  =  0.37,     # ≈ 0.0072 × c_ecig_max
    α_TE =  0.50,     # scaled down from 0.0094 × c_bundle_max = 9.57
    λ_1  =  0.67,
    λ_2  =  0.41,
    μ    =  0.05,
    γ    = -0.05,
    ω    = -1.94,     # ≈ -0.0032 × E_max
    ξ_T  = -3.61,
    ξ_E  = -5.46,
    ξ_TE = -6.05,
    ψ    =  0.5      # addiction decay rate (94% per month)
);

# Convert to vector for VFI
θ_true_vec = collect(Float64, values(θ_true));

# Parameter names for output
param_names = collect(String, string.(keys(θ_true)))

# Number of parameters
N_params = length(θ_true)


#############################
# MC Settings
#############################

# Optimizer settings (same across all replications)
L          = 2       # Number of random restarts (outer tries)
M          = 2        # Short Nelder-Mead runs per outer try (inner tries)
inner_iter = 100      # Max iterations per short run

mc_log("MC settings: s=$s, N_HH=$N_HH_real, N_obs=$N_obs_real, L=$L, M=$M, inner_iter=$inner_iter")
mc_log("True parameters:")
for k in 1:N_params
    mc_log("  $(param_names[k]) = $(θ_true_vec[k])")
end


#############################
# Solve DGP Value Function
#############################

mc_log("\n===================================")
mc_log("Solving VFI at true parameters...")
mc_log("===================================")

t_vfi = time()

# Create addiction grid using the true ψ from θ_true
ψ_true = θ_true.ψ
N_A, A = get_addiction_space(ψ_true)

# Compute flow utility at true θ (first 11 elements, excluding ψ)
U_true = get_flow_utility(
    θ_true_vec[1:end-1], N_J, N_A, N_Pcomb, A, c_cig, c_ecig, c_bundle, n, is_flavored, cat_idx, E
)

# Compute addiction transition brackets at true ψ
a_lower_true, a_upper_true, a_weight_true = precompute_addiction_transitions(
    N_J, N_A, ψ_true, A, n
)

# Solve VFI at true parameters (cold start)
_, V_decision_true, vfi_iters_true, _ = solve_vfi(
    N_J, N_A, N_P, N_Pcomb, β, δ, U_true,
    a_lower_true, a_upper_true, a_weight_true,
    p_cig_lo, p_cig_hi, p_cig_w,
    p_ecig_lo, p_ecig_hi, p_ecig_w
)

vfi_elapsed = time() - t_vfi
mc_log("DGP VFI converged in $vfi_iters_true iterations ($(round(vfi_elapsed, digits=1))s)")


#############################
# Starting Values for
# Estimation
#############################

# Starting values for estimation (STANDARDIZED units)
starting_param = (
    α_T  =  0.1,
    α_E  =  0.1,
    α_TE =  0.1,
    λ_1  =  0.1,
    λ_2  =  0.1,
    μ    =  0.1,      # no static counterpart
    γ    = -0.1,      # no static counterpart
    ω    = -1.0,
    ξ_T  = -1.0,
    ξ_E  = -2.0,
    ξ_TE = -3.0,
    ψ    =  0.3      
);

# Initial simplex deviations for Nelder-Mead
# Scaled to ~50% of the absolute value of each starting parameter;
# μ and γ get larger deviations since they have no informed starting values.
add = [
    abs(starting_param.α_T)  * 0.50,   # α_T
    abs(starting_param.α_E)  * 0.50,   # α_E
    abs(starting_param.α_TE) * 0.50,   # α_TE
    abs(starting_param.λ_1)  * 0.50,   # λ_1
    abs(starting_param.λ_2)  * 0.50,   # λ_2
    abs(starting_param.μ)    * 1.00,   # μ
    abs(starting_param.γ)    * 1.00,   # γ
    abs(starting_param.ω)    * 0.50,   # ω
    abs(starting_param.ξ_T)  * 0.50,   # ξ_T
    abs(starting_param.ξ_E)  * 0.50,   # ξ_E
    abs(starting_param.ξ_TE) * 0.50,   # ξ_TE
    0.20                                # ψ: ±0.10 around starting value
];


#############################
# Run Single Replication
#############################

mc_log("\n===================================")
mc_log("Running Replication $s")
mc_log("===================================\n")

# Open parameter trace file and write header
global param_trace_io = open(trace_path, "w")
trace_header = "sim\touter_try\tinner_run\teval\tNLL\tVFI_iters\t" * join(param_names, "\t")
println(param_trace_io, trace_header)
flush(param_trace_io)

t_rep = time()

# Simulate choices from the true DGP using real observables
t_sim = time()
global y_sim, tya_state_sim, p_continuous_sim, hh_codes_sim
y_sim, tya_state_sim, p_continuous_sim, hh_codes_sim = simulate_data(
    V_decision_true, ψ_true, N_J, N_P, A, P, n,
    real_p_continuous, real_tya_state, real_hh_codes
)
sim_elapsed = time() - t_sim
mc_log("Data simulation: $(length(y_sim)) obs in $(round(sim_elapsed, digits=2))s")

# Reset evaluation counter and set replication number
global eval_count = 0
global current_replication = s

# Estimate θ via multi-start Nelder-Mead
t_est = time()
opt_param, opt_value = random_amoeba(
    objective_mc, starting_param, add, L, M, inner_iter;
    log_io = mc_log_io
)
est_elapsed = time() - t_est
mc_log("Estimation complete: $eval_count evaluations in $(round(est_elapsed, digits=1))s")

# Get estimated parameters as vector
θ_hat = collect(Float64, values(opt_param))

rep_elapsed = time() - t_rep


#############################
# Save Results
#############################

# Write per-replication result file
open(results_path, "w") do io
    println(io, "S\tNLL\t" * join(param_names, "\t"))
    println(io, "$s\t$(@sprintf("%.10f", opt_value))\t" * join([@sprintf("%.10f", θ_hat[k]) for k in 1:N_params], "\t"))
end

mc_log("\nReplication $s results ($(round(rep_elapsed, digits=1))s total):")
mc_log("  Negative log-likelihood: $(round(opt_value, digits=4))")
mc_log(@sprintf("  %-8s  %12s  %12s  %12s", "Param", "True", "Estimated", "Diff"))
mc_log("  " * repeat("-", 50))
for k in 1:N_params
    diff = θ_hat[k] - θ_true_vec[k]
    mc_log(@sprintf("  %-8s  %12.6f  %12.6f  %12.6f", param_names[k], θ_true_vec[k], θ_hat[k], diff))
end

mc_log("\nResults saved to: $results_path")
mc_log("MC Replication $s finished at $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")

# Close log and trace files
close(param_trace_io)
close(mc_log_io)
