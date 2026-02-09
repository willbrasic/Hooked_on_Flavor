################################################################################
# William Brasic
# The University of Arizona
# wbrasic97@gmail.com
# February 2026
#
# Aggregates per-replication MC results from job array into summary statistics.
#
# Reads all MC_Rep_<s>.txt files from the results directory, combines them,
# and computes: Mean, Bias, Std Dev, RMSE for each parameter.
#
# Usage:
#   julia 03_MC_Aggregate_Results.jl
################################################################################

using CSV, DataFrames, Statistics, Printf, Dates


#############################
# Settings
#############################

# Results directory
if Sys.iswindows()
    results_dir = "C:/Users/wbras/OneDrive/Documents/Desktop/UA/4th_Year_Paper/4th_Year_Paper_Data/HMS/2021-Onward/Dynamic_Model/MC_Simulation_Results"
else
    results_dir = abspath("./MC_Simulation_Results")
end

# True parameters (must match 02_MC_Simulation_Array.jl)
θ_true = (
    α_T  =  0.46,
    α_E  =  0.37,
    α_TE =  0.50,
    λ_1  =  0.67,
    λ_2  =  0.41,
    μ    =  0.05,
    γ    = -0.05,
    ω    = -1.94,
    ξ_T  = -3.61,
    ξ_E  = -5.46,
    ξ_TE = -6.05,
    ψ    =  0.94
)
θ_true_vec = collect(Float64, values(θ_true))
param_names = collect(String, string.(keys(θ_true)))
N_params = length(θ_true)


#############################
# Collect Results
#############################

# Find all per-replication result files (match pattern: ##_MC_Rep_Results_*.txt)
rep_files = filter(f -> contains(f, "_MC_Rep_Results_") && endswith(f, ".txt"), readdir(results_dir))

println("Found $(length(rep_files)) replication result files in $results_dir")

if isempty(rep_files)
    error("No *_MC_Rep_Results_*.txt files found. Run 02_MC_Simulation_Array.jl first.")
end

# Read and combine all results
all_results = DataFrame[]
for f in rep_files
    filepath = joinpath(results_dir, f)
    df = CSV.read(filepath, DataFrame; delim='\t')
    push!(all_results, df)
end

results_df = vcat(all_results...)
sort!(results_df, :S)

S = nrow(results_df)
println("Loaded $S replications (s = $(minimum(results_df.S)) to $(maximum(results_df.S)))")


#############################
# Check for Missing Reps
#############################

expected_reps = Set(1:maximum(results_df.S))
actual_reps = Set(results_df.S)
missing_reps = setdiff(expected_reps, actual_reps)

if !isempty(missing_reps)
    missing_sorted = sort(collect(missing_reps))
    println("\nWARNING: $(length(missing_reps)) missing replications: $(missing_sorted)")
end


#############################
# Summary Statistics
#############################

println("\n" * "="^70)
println("Monte Carlo Summary ($S replications)")
println("="^70 * "\n")

header = @sprintf("%-8s  %12s  %12s  %12s  %12s  %12s", "Param", "True", "Mean", "Bias", "Std Dev", "RMSE")
println(header)
println(repeat("-", length(header)))

for k in 1:N_params
    pname = param_names[k]
    true_val = θ_true_vec[k]
    estimates = results_df[!, Symbol(pname)]
    mean_est = mean(estimates)
    bias = mean_est - true_val
    std_est = std(estimates)
    rmse = sqrt(bias^2 + std_est^2)

    println(@sprintf("%-8s  %12.6f  %12.6f  %12.6f  %12.6f  %12.6f",
        pname, true_val, mean_est, bias, std_est, rmse))
end


#############################
# NLL Statistics
#############################

if hasproperty(results_df, :NLL)
    println(@sprintf("\nNLL: mean = %.4f, std = %.4f, min = %.4f, max = %.4f",
        mean(results_df.NLL), std(results_df.NLL),
        minimum(results_df.NLL), maximum(results_df.NLL)))
end


#############################
# Save Combined Results
#############################

timestamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")

combined_path = joinpath(results_dir, "MC_Results_Combined_$timestamp.txt")
open(combined_path, "w") do io
    # Header
    println(io, "S\tNLL\t" * join(param_names, "\t"))
    for row in eachrow(results_df)
        nll_val = hasproperty(results_df, :NLL) ? @sprintf("%.10f", row.NLL) : "NA"
        θ_str = join([@sprintf("%.10f", row[Symbol(pname)]) for pname in param_names], "\t")
        println(io, "$(row.S)\t$nll_val\t$θ_str")
    end
end
println("\nCombined results saved to: $combined_path")

# Save summary statistics
summary_path = joinpath(results_dir, "MC_Summary_$timestamp.txt")
open(summary_path, "w") do io
    println(io, @sprintf("%-8s\t%12s\t%12s\t%12s\t%12s\t%12s", "Param", "True", "Mean", "Bias", "Std_Dev", "RMSE"))
    for k in 1:N_params
        pname = param_names[k]
        true_val = θ_true_vec[k]
        estimates = results_df[!, Symbol(pname)]
        mean_est = mean(estimates)
        bias = mean_est - true_val
        std_est = std(estimates)
        rmse = sqrt(bias^2 + std_est^2)
        println(io, @sprintf("%-8s\t%12.6f\t%12.6f\t%12.6f\t%12.6f\t%12.6f",
            pname, true_val, mean_est, bias, std_est, rmse))
    end
end
println("Summary statistics saved to: $summary_path")
