################################################################################
# William Brasic
# The University of Arizona
# wbrasic97@gmail.com
# June 2026
#
# This script aggregates the B=100 bootstrap counterfactual draws produced by
# 02_CF_Mixture_Bootstrap.jl into mean and 95% confidence interval outputs.
#
# For each counterfactual type (ban and tax) and each output file stem, this
# script:
#   1. Loads B draw-specific CSVs tagged with b001 through b100.
#   2. Stacks them into a (N_rows x B) matrix per numeric column.
#   3. Computes column-wise mean, 2.5th percentile, and 97.5th percentile.
#   4. Writes one aggregated CSV per counterfactual per output file stem.
#
# Six file stems are aggregated per counterfactual:
#   - Simulation_Overall
#   - Simulation_by_TYA
#   - Simulation_by_Type
#   - Extensive_Margin_by_TYA (three threshold variants: 005, 010, 020)
#
# Flags (must match those used when running 02_CF_Mixture_Bootstrap_Slurm.sb):
#   ESTIMATE_PSI_3 (default: false)
#   PSI_3          (default: 0.75)
#   BETA           (default: 1.0)
#   B              (default: 100)    Number of bootstrap draws to aggregate
################################################################################


#############################
# Section 1: Setup
#############################

using CSV, DataFrames, Statistics, Printf, Dates

HPC = !Sys.iswindows()

ESTIMATE_PSI_3 = parse(Bool, get(ENV, "ESTIMATE_PSI_3", "false"))
PSI_3          = parse(Float64, get(ENV, "PSI_3", "0.75"))
BETA_VAL       = parse(Float64, get(ENV, "BETA", "1.0"))
B              = parse(Int,     get(ENV, "B", "100"))

# Mirrors numeric_tag from 01_Functions_Mixture.jl: 1.0 -> "1", 0.50 -> "05"
function numeric_tag(val)
    s = rstrip(rstrip(string(val), '0'), '.')
    return replace(s, "." => "")
end

psi_tag = "Psi2_09_Psi1_01"
psi_3_tag = ESTIMATE_PSI_3 ? "Psi3_Est" : "Psi3_$(numeric_tag(PSI_3))"
beta_tag  = numeric_tag(BETA_VAL)

if HPC
    results_base = "/home/u2/wbrasic/4th_Year_Paper/Dynamic_Model/04_CF_Mixture_V2_Bootstrap"
else
    results_base = "C:/Users/wbras/OneDrive/Documents/Desktop/UA/4th_Year_Paper/4th_Year_Paper_Data/HMS/2021-Onward/Dynamic_Model"
end

results_dir = joinpath(results_base, "CF_Bootstrap_Mixture_$(psi_tag)_$(psi_3_tag)_Beta_$(beta_tag)_Results")
output_dir  = joinpath(results_base, "CF_Bootstrap_Mixture_$(psi_tag)_$(psi_3_tag)_Beta_$(beta_tag)_Aggregated")
mkpath(output_dir)

t_start = time()

println("===================================")
println("Bootstrap Aggregation")
println("===================================")
println("HPC = $HPC")
println("ESTIMATE_PSI_3 = $ESTIMATE_PSI_3")
println("PSI_3 = $PSI_3 (tag: $psi_3_tag)")
println("BETA  = $BETA_VAL (tag: $beta_tag)")
println("B     = $B")
println("results_dir = $results_dir")
println("output_dir  = $output_dir")
println()


#############################
# Section 2: Helper Functions
#############################

"""
Load B bootstrap CSVs from `cf_dir`. `filename_fn(b_str)` returns the filename
for bootstrap draw b (b_str is zero-padded to 3 digits).
Returns only the draws for which the file exists; warns about missing files.
"""
function load_bootstrap_csvs(cf_dir::String, filename_fn::Function, B::Int)
    dfs = DataFrame[]
    missing_draws = Int[]
    for b in 1:B
        b_str = lpad(b, 3, '0')
        fname = filename_fn(b_str)
        fpath = joinpath(cf_dir, fname)
        if isfile(fpath)
            push!(dfs, CSV.read(fpath, DataFrame))
        else
            push!(missing_draws, b)
        end
    end
    if !isempty(missing_draws)
        @warn "Missing $(length(missing_draws)) bootstrap draws: $(missing_draws)"
    end
    println("  Loaded $(length(dfs)) / $B draws from: $cf_dir")
    return dfs
end


"""
Given a vector of DataFrames (one per bootstrap draw), compute column-wise
mean, p2.5, and p97.5 across draws for all numeric columns,
grouped by `group_cols` (e.g., ["period"] or ["group", "period"] or ["type", "period"]).

Returns a DataFrame with columns: group_cols..., <col>_mean, <col>_p025, <col>_p975
for each numeric column not in group_cols.
"""
function aggregate_draws(dfs::Vector{DataFrame}, group_cols::Vector{String})
    isempty(dfs) && return DataFrame()

    # Identify numeric columns to aggregate
    ref = dfs[1]
    num_cols = [c for c in names(ref) if !(string(c) in group_cols) && eltype(ref[!, c]) <: Number]

    # Stack all draws
    n_rows = nrow(ref)
    B_actual = length(dfs)

    # Build result from first draw's group columns
    result = select(ref, group_cols)

    for col in num_cols
        col_mat = Matrix{Float64}(undef, n_rows, B_actual)
        for (b_idx, df) in enumerate(dfs)
            col_mat[:, b_idx] = Float64.(df[!, col])
        end
        result[!, "$(col)_mean"] = vec(mean(col_mat, dims=2))
        result[!, "$(col)_p025"] = vec(mapslices(x -> quantile(x, 0.025), col_mat, dims=2))
        result[!, "$(col)_p975"] = vec(mapslices(x -> quantile(x, 0.975), col_mat, dims=2))
    end

    return result
end


"""
    write_aggregated(df, path)

Write an aggregated DataFrame to CSV using open/println style.
"""
function write_aggregated(df::DataFrame, path::String)
    open(path, "w") do io
        println(io, join(names(df), ","))
        for row in eachrow(df)
            vals = [isa(v, AbstractFloat) ? @sprintf("%.10f", v) :
                    isa(v, Integer)        ? @sprintf("%d", v)    : string(v)
                    for v in values(row)]
            println(io, join(vals, ","))
        end
    end
    println("  Written: $path")
end


#############################
# Section 3: Aggregate Ban CFs
#############################

BAN_TYPES = ["Ban_Comprehensive", "Ban_FDA_Only", "Ban_Non_FDA"]

for ban_label in BAN_TYPES

    println("\n===================================")
    println("Aggregating: $ban_label (beta = $beta_tag)")
    println("===================================")

    beta_subdir = joinpath(results_dir, ban_label)  # β encoded in results_dir name

    # --- Overall ---
    dfs_overall = load_bootstrap_csvs(beta_subdir, b_str -> "Simulation_Overall_b$(b_str).csv", B)
    if !isempty(dfs_overall)
        agg = aggregate_draws(dfs_overall, ["period"])
        out_path = joinpath(output_dir, "$(ban_label)_Overall_Aggregated.csv")
        write_aggregated(agg, out_path)
    end

    # --- By TYA ---
    dfs_tya = load_bootstrap_csvs(beta_subdir, b_str -> "Simulation_by_TYA_b$(b_str).csv", B)
    if !isempty(dfs_tya)
        agg = aggregate_draws(dfs_tya, ["group", "period"])
        out_path = joinpath(output_dir, "$(ban_label)_by_TYA_Aggregated.csv")
        write_aggregated(agg, out_path)
    end

    # --- By Type ---
    dfs_type = load_bootstrap_csvs(beta_subdir, b_str -> "Simulation_by_Type_b$(b_str).csv", B)
    if !isempty(dfs_type)
        agg = aggregate_draws(dfs_type, ["type", "period"])
        out_path = joinpath(output_dir, "$(ban_label)_by_Type_Aggregated.csv")
        write_aggregated(agg, out_path)
    end

    # --- By TYA and Type (TYA-present households, split by modal type) ---
    dfs_tya_type = load_bootstrap_csvs(beta_subdir, b_str -> "Simulation_by_TYA_Type_b$(b_str).csv", B)
    if !isempty(dfs_tya_type)
        agg = aggregate_draws(dfs_tya_type, ["type", "period"])
        out_path = joinpath(output_dir, "$(ban_label)_by_TYA_Type_Aggregated.csv")
        write_aggregated(agg, out_path)
    end

    # --- Extensive Margin by TYA (3 thresholds) ---
    for (thresh_tag, suffix) in [("005", "_thresh005"), ("010", ""), ("020", "_thresh020")]
        dfs_ext = load_bootstrap_csvs(beta_subdir, b_str -> "Extensive_Margin_by_TYA$(suffix)_b$(b_str).csv", B)
        if !isempty(dfs_ext)
            agg = aggregate_draws(dfs_ext, ["group", "period"])
            out_path = joinpath(output_dir, "$(ban_label)_Extensive_Margin_TYA_thresh$(thresh_tag)_Aggregated.csv")
            write_aggregated(agg, out_path)
        end
    end

    # --- Extensive Margin by Type (3 thresholds) ---
    for (thresh_tag, suffix) in [("005", "_thresh005"), ("010", ""), ("020", "_thresh020")]
        dfs_ext_type = load_bootstrap_csvs(beta_subdir, b_str -> "Extensive_Margin_by_Type$(suffix)_b$(b_str).csv", B)
        if !isempty(dfs_ext_type)
            agg = aggregate_draws(dfs_ext_type, ["type", "period"])
            out_path = joinpath(output_dir, "$(ban_label)_Extensive_Margin_Type_thresh$(thresh_tag)_Aggregated.csv")
            write_aggregated(agg, out_path)
        end
    end

    # --- Extensive Margin by TYA and Type (TYA-present households, split by
    # modal type; 3 thresholds) ---
    for (thresh_tag, suffix) in [("005", "_thresh005"), ("010", ""), ("020", "_thresh020")]
        dfs_ext_tya_type = load_bootstrap_csvs(beta_subdir, b_str -> "Extensive_Margin_by_TYA_Type$(suffix)_b$(b_str).csv", B)
        if !isempty(dfs_ext_tya_type)
            agg = aggregate_draws(dfs_ext_tya_type, ["type", "period"])
            out_path = joinpath(output_dir, "$(ban_label)_Extensive_Margin_TYA_Type_thresh$(thresh_tag)_Aggregated.csv")
            write_aggregated(agg, out_path)
        end
    end

end


#############################
# Section 4: Aggregate Tax CFs
#############################

TAX_TYPES = ["Flavor_Tax"]
TAX_LEVELS = [0.50, 2.78]

for tax_label in TAX_TYPES, tau in TAX_LEVELS

    tau_tag = replace(@sprintf("%.2f", tau), "." => "p")

    println("\n===================================")
    println("Aggregating: $tax_label / Tax_$(tau_tag) (beta = $beta_tag)")
    println("===================================")

    beta_subdir = joinpath(results_dir, tax_label, "Tax_$(tau_tag)")  # β encoded in results_dir name

    # --- Overall ---
    dfs_overall = load_bootstrap_csvs(beta_subdir, b_str -> "Simulation_Overall_b$(b_str).csv", B)
    if !isempty(dfs_overall)
        agg = aggregate_draws(dfs_overall, ["period"])
        out_path = joinpath(output_dir, "$(tax_label)_Tax_$(tau_tag)_Overall_Aggregated.csv")
        write_aggregated(agg, out_path)
    end

    # --- By TYA ---
    dfs_tya = load_bootstrap_csvs(beta_subdir, b_str -> "Simulation_by_TYA_b$(b_str).csv", B)
    if !isempty(dfs_tya)
        agg = aggregate_draws(dfs_tya, ["group", "period"])
        out_path = joinpath(output_dir, "$(tax_label)_Tax_$(tau_tag)_by_TYA_Aggregated.csv")
        write_aggregated(agg, out_path)
    end

    # --- By Type ---
    dfs_type = load_bootstrap_csvs(beta_subdir, b_str -> "Simulation_by_Type_b$(b_str).csv", B)
    if !isempty(dfs_type)
        agg = aggregate_draws(dfs_type, ["type", "period"])
        out_path = joinpath(output_dir, "$(tax_label)_Tax_$(tau_tag)_by_Type_Aggregated.csv")
        write_aggregated(agg, out_path)
    end

    # --- By TYA and Type (TYA-present households, split by modal type) ---
    dfs_tya_type = load_bootstrap_csvs(beta_subdir, b_str -> "Simulation_by_TYA_Type_b$(b_str).csv", B)
    if !isempty(dfs_tya_type)
        agg = aggregate_draws(dfs_tya_type, ["type", "period"])
        out_path = joinpath(output_dir, "$(tax_label)_Tax_$(tau_tag)_by_TYA_Type_Aggregated.csv")
        write_aggregated(agg, out_path)
    end

    # --- Extensive Margin by TYA (3 thresholds) ---
    for (thresh_tag, suffix) in [("005", "_thresh005"), ("010", ""), ("020", "_thresh020")]
        dfs_ext = load_bootstrap_csvs(beta_subdir, b_str -> "Extensive_Margin_by_TYA$(suffix)_b$(b_str).csv", B)
        if !isempty(dfs_ext)
            agg = aggregate_draws(dfs_ext, ["group", "period"])
            out_path = joinpath(output_dir, "$(tax_label)_Tax_$(tau_tag)_Extensive_Margin_TYA_thresh$(thresh_tag)_Aggregated.csv")
            write_aggregated(agg, out_path)
        end
    end

    # --- Extensive Margin by Type (3 thresholds) ---
    for (thresh_tag, suffix) in [("005", "_thresh005"), ("010", ""), ("020", "_thresh020")]
        dfs_ext_type = load_bootstrap_csvs(beta_subdir, b_str -> "Extensive_Margin_by_Type$(suffix)_b$(b_str).csv", B)
        if !isempty(dfs_ext_type)
            agg = aggregate_draws(dfs_ext_type, ["type", "period"])
            out_path = joinpath(output_dir, "$(tax_label)_Tax_$(tau_tag)_Extensive_Margin_Type_thresh$(thresh_tag)_Aggregated.csv")
            write_aggregated(agg, out_path)
        end
    end

    # --- Extensive Margin by TYA and Type (TYA-present households, split by
    # modal type; 3 thresholds) ---
    for (thresh_tag, suffix) in [("005", "_thresh005"), ("010", ""), ("020", "_thresh020")]
        dfs_ext_tya_type = load_bootstrap_csvs(beta_subdir, b_str -> "Extensive_Margin_by_TYA_Type$(suffix)_b$(b_str).csv", B)
        if !isempty(dfs_ext_tya_type)
            agg = aggregate_draws(dfs_ext_tya_type, ["type", "period"])
            out_path = joinpath(output_dir, "$(tax_label)_Tax_$(tau_tag)_Extensive_Margin_TYA_Type_thresh$(thresh_tag)_Aggregated.csv")
            write_aggregated(agg, out_path)
        end
    end

end


#############################
# Section 5: Final Timing
#############################

total_elapsed = time() - t_start
println("\n===================================")
println("Aggregation COMPLETE")
println("===================================")
@printf("Total elapsed time: %.2f seconds (%.2f minutes)\n", total_elapsed, total_elapsed / 60)
println("Aggregated results written to: $output_dir")
println("Completed at $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")