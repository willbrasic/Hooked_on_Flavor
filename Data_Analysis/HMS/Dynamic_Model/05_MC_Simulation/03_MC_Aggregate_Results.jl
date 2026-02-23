################################################################################
# William Brasic
# The University of Arizona
# wbrasic97@gmail.com
# February 2026
#
# Aggregates per-replication MC results from job array into summary statistics.
#
# Reads all MC_Rep_<s>.csv files from the results directory, combines them,
# and computes: Mean, Bias, Std Dev, RMSE for each parameter.
#
# Usage:
#   julia 03_MC_Aggregate_Results.jl
################################################################################

# Load required packages
using CSV, DataFrames, Statistics, Printf, Dates


#############################
# Settings
#############################

# Results directory (platform-dependent path)
if Sys.iswindows()
    results_dir = "C:/Users/wbras/OneDrive/Documents/Desktop/UA/4th_Year_Paper/4th_Year_Paper_Data/HMS/2021-Onward/Dynamic_Model/MC_Simulation_Results"
else
    results_dir = abspath("./MC_Simulation_Results")
end

# Read true parameters (written by 02_MC_Simulation_Array.jl during simulation)
θ_true_df = CSV.read(joinpath(results_dir, "MC_True_Parameters.csv"), DataFrame);

# Extract parameter names, true values, and number of parameters
param_names = String.(θ_true_df.parameter);
θ_true_vec = Float64.(θ_true_df.value);
N_params = length(θ_true_vec);


#############################
# Collect Results
#############################

# Find all per-replication result files (match pattern: ##_MC_Rep_Results_*.csv)
rep_files = filter(f -> contains(f, "_MC_Rep_Results_") && endswith(f, ".csv"), readdir(results_dir));

# Print number of replication files found
println("Found $(length(rep_files)) replication result files in $results_dir")

# Error if no result files exist
if isempty(rep_files)
    error("No *_MC_Rep_Results_*.csv files found. Run 02_MC_Simulation_Array.jl first.")
end

# Read each per-replication CSV and combine into a single DataFrame
all_results = DataFrame[];
for f in rep_files
    filepath = joinpath(results_dir, f);
    df = CSV.read(filepath, DataFrame);
    push!(all_results, df);
end

# Combine all replications and sort by replication number
results_df = vcat(all_results...);
sort!(results_df, :S);

# Total number of successfully completed replications
S = nrow(results_df);

# Print number of loaded replications and range
println("Loaded $S replications (s = $(minimum(results_df.S)) to $(maximum(results_df.S)))")


#############################
# Check for Missing Reps
#############################

# Compute expected vs actual replication numbers to detect missing jobs
expected_reps = Set(1:maximum(results_df.S));
actual_reps = Set(results_df.S);
missing_reps = setdiff(expected_reps, actual_reps);

# Print warning if any replications are missing
if !isempty(missing_reps)
    missing_sorted = sort(collect(missing_reps));
    println("\nWARNING: $(length(missing_reps)) missing replications: $(missing_sorted)")
end


#############################
# Summary Statistics
#############################

# Print MC summary statistics header
println("\n" * "="^70)
println("Monte Carlo Summary ($S replications)")
println("="^70 * "\n")

# Print summary table: True, Mean, Bias, Std Dev, RMSE for each parameter
header = @sprintf("%-8s  %12s  %12s  %12s  %12s  %12s", "Param", "True", "Mean", "Bias", "Std Dev", "RMSE")
println(header)
println(repeat("-", length(header)))

for k in 1:N_params
    pname = param_names[k];
    true_val = θ_true_vec[k];
    estimates = results_df[!, Symbol(pname)];
    mean_est = mean(estimates);
    bias = mean_est - true_val;
    std_est = std(estimates);
    rmse = sqrt(bias^2 + std_est^2);

    println(@sprintf("%-8s  %12.6f  %12.6f  %12.6f  %12.6f  %12.6f",
        pname, true_val, mean_est, bias, std_est, rmse))
end


#############################
# NLL Statistics
#############################

# Print NLL statistics if available
if hasproperty(results_df, :NLL)
    println(@sprintf("\nNLL: mean = %.4f, std = %.4f, min = %.4f, max = %.4f",
        mean(results_df.NLL), std(results_df.NLL),
        minimum(results_df.NLL), maximum(results_df.NLL)))
end


#############################
# Save Combined Results
#############################

# Create a timestamp for uniquely naming output files
timestamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")

# Save combined per-replication results to a CSV file
combined_path = joinpath(results_dir, "MC_Results_Combined_$timestamp.csv")
open(combined_path, "w") do io
    println(io, "S,NLL," * join(param_names, ","))
    for row in eachrow(results_df)
        nll_val = hasproperty(results_df, :NLL) ? @sprintf("%.10f", row.NLL) : "NA"
        θ_str = join([@sprintf("%.10f", row[Symbol(pname)]) for pname in param_names], ",")
        println(io, "$(row.S),$nll_val,$θ_str")
    end
end

# Print combined results save location
println("\nCombined results saved to: $combined_path")

# Save summary statistics (True, Mean, Bias, Std Dev, RMSE) to a CSV file
summary_path = joinpath(results_dir, "MC_Summary_$timestamp.csv")
open(summary_path, "w") do io
    println(io, "Param,True,Mean,Bias,Std_Dev,RMSE")
    for k in 1:N_params
        pname = param_names[k];
        true_val = θ_true_vec[k];
        estimates = results_df[!, Symbol(pname)];
        mean_est = mean(estimates);
        bias = mean_est - true_val;
        std_est = std(estimates);
        rmse = sqrt(bias^2 + std_est^2);
        println(io, @sprintf("%s,%.6f,%.6f,%.6f,%.6f,%.6f",
            pname, true_val, mean_est, bias, std_est, rmse))
    end
end

# Print summary statistics save location
println("Summary statistics saved to: $summary_path")
