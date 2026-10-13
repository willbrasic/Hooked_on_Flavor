################################################################################
# William Brasic
# The University of Arizona
# wbrasic97@gmail.com
# October 2026
#
# This script aggregates the bootstrap counterfactual draws produced by
# 02_CF_Mixture_Bootstrap.jl into mean and 95% confidence interval outputs.
#
# For each policy and each output file stem, this script:
#   1. Loads the draw-specific CSVs tagged b001 through b<B>.
#   2. Stacks them into a (N_rows x B) matrix per numeric column.
#   3. Computes the column-wise mean, 2.5th percentile, and 97.5th percentile,
#      ignoring NaN entries (a τ* beyond the grid is NaN in that draw) and
#      recording how many draws entered each cell.
#   4. Appends the point-estimate run (b000) as <col>_point when it exists.
#   5. Writes one aggregated CSV per policy per stem.
#
# Policies: Ban_Comprehensive, Ban_Non_FDA, and every Flavor_Tax/Tax_* folder
# found in the results directory. Stems per policy:
#   - Simulation_{Overall, by_TYA, by_Type, by_TYA_Type, by_NonTYA_Type}
#   - Extensive_Margin{,_Cig,_OrigEcig}_by_{TYA, Type, TYA_Type, NonTYA_Type}
#     at the 3, 6, and 12 month non-user horizons (no purchase of the
#     outcome's product or of flavored products in the last k observed months)
# Frontier stems at the results root: Frontier_Summary, Frontier_Trajectory,
# Frontier_TauStar.
#
# Flags (must match those used when running 02_CF_Mixture_Bootstrap_Slurm.sb):
#   PSI_3          (default: 0.50)
#   BETA           (default: 1.0)
#   B              (default: 100)    Number of bootstrap draws to aggregate
################################################################################


#############################
# Section 1: Setup
#############################

using CSV, DataFrames, Statistics, Printf, Dates

HPC = !Sys.iswindows()

PSI_3    = parse(Float64, get(ENV, "PSI_3", "0.50"))
BETA_VAL = parse(Float64, get(ENV, "BETA", "1.0"))
B        = parse(Int,     get(ENV, "B", "100"))

# Mirrors numeric_tag from 01_Functions_Mixture.jl: 1.0 -> "1", 0.50 -> "05"
function numeric_tag(val)
    s = rstrip(rstrip(string(val), '0'), '.')
    return replace(s, "." => "")
end

psi_tag  = "Psi2_09_Psi1_01"
beta_tag = numeric_tag(BETA_VAL)

if HPC
    results_base = "/home/u2/wbrasic/4th_Year_Paper/Dynamic_Model/05_CF_Mixture_Bootstrap"
else
    results_base = "C:/Users/wbras/OneDrive/Documents/Desktop/UA/4th_Year_Paper/4th_Year_Paper_Data/HMS/2021-Onward/Dynamic_Model"
end

results_dir = joinpath(results_base, "CF_Bootstrap_Mixture_$(psi_tag)_Beta_$(beta_tag)_Results")
output_dir  = joinpath(results_base, "CF_Bootstrap_Mixture_$(psi_tag)_Beta_$(beta_tag)_Aggregated")
mkpath(output_dir)

t_start = time()

println("===================================")
println("Bootstrap Aggregation")
println("===================================")
println("HPC = $HPC")
println("PSI_3 = $PSI_3")
println("BETA  = $BETA_VAL (tag: $beta_tag)")
println("B     = $B")
println("results_dir = $results_dir")
println("output_dir  = $output_dir")
println()


#############################
# Section 2: Functions
#############################

"""
Load the bootstrap CSVs `<stem>_b001.csv` through `<stem>_b<B>.csv` from
`cf_dir`. Returns only the draws for which the file exists and warns about
missing draws.
"""
function load_bootstrap_csvs(cf_dir::String, stem::String, B::Int)
    dfs = DataFrame[]
    missing_draws = Int[]
    for b in 1:B
        fpath = joinpath(cf_dir, "$(stem)_b$(lpad(b, 3, '0')).csv")
        if isfile(fpath)
            push!(dfs, CSV.read(fpath, DataFrame))
        else
            push!(missing_draws, b)
        end
    end
    if !isempty(missing_draws) && length(missing_draws) < B
        @warn "$stem: missing $(length(missing_draws)) bootstrap draws: $(missing_draws)"
    end
    return dfs
end


"""
Load the point-estimate run `<stem>_b000.csv` from `cf_dir`, or `nothing` if absent.
"""
function load_point_csv(cf_dir::String, stem::String)
    fpath = joinpath(cf_dir, "$(stem)_b000.csv")
    isfile(fpath) || return nothing
    return CSV.read(fpath, DataFrame)
end


"""
Given a vector of DataFrames (one per draw), compute the column-wise mean,
p2.5, and p97.5 across draws for every numeric column not in `group_cols`,
ignoring NaN entries, plus the number of draws entering each cell. Rows are
matched by position, which is valid because every draw writes the same rows
in the same order. When `point` is given, its values are appended as
`<col>_point`.
"""
function aggregate_draws(dfs::Vector{DataFrame}, group_cols::Vector{String}; point = nothing)
    isempty(dfs) && return DataFrame()

    ref = dfs[1]
    num_cols = [c for c in names(ref) if !(string(c) in group_cols) && eltype(ref[!, c]) <: Union{Missing, Number}]

    n_rows   = nrow(ref)
    B_actual = length(dfs)

    result = select(ref, group_cols)

    # NaN-aware row statistic; a row with no finite draws is NaN rather than an error
    stat(f, v) = isempty(v) ? NaN : f(v)

    if point !== nothing
        nrow(point) == n_rows || error("point run has $(nrow(point)) rows, expected $(n_rows)")
    end

    for col in num_cols
        col_mat = Matrix{Float64}(undef, n_rows, B_actual)
        for (b_idx, df) in enumerate(dfs)
            nrow(df) == n_rows || error("$(col): draw $(b_idx) has $(nrow(df)) rows, expected $(n_rows)")
            col_mat[:, b_idx] = Float64.(coalesce.(df[!, col], NaN))
        end
        finite_rows = [filter(!isnan, col_mat[r, :]) for r in 1:n_rows]
        result[!, "$(col)_mean"]    = [stat(mean, v)                  for v in finite_rows]
        result[!, "$(col)_p025"]    = [stat(x -> quantile(x, 0.025), v) for v in finite_rows]
        result[!, "$(col)_p975"]    = [stat(x -> quantile(x, 0.975), v) for v in finite_rows]
        result[!, "$(col)_n_draws"] = [length(v) for v in finite_rows]
        if point !== nothing
            result[!, "$(col)_point"] = Float64.(coalesce.(point[!, col], NaN))
        end
    end

    return result
end


"""
Write an aggregated DataFrame to CSV.
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


"""
Pool one stem from `cf_dir` and write `<out_prefix>_<out_stem>_Aggregated.csv`.
"""
function pool_stem(cf_dir::String, stem::String, group_cols::Vector{String}, out_name::String)
    dfs = load_bootstrap_csvs(cf_dir, stem, B)
    if isempty(dfs)
        println("  No draws found for $stem in $cf_dir; skipped")
        return nothing
    end
    point = load_point_csv(cf_dir, stem)
    println("  $stem: $(length(dfs)) / $B draws" * (point === nothing ? "" : " + point run"))
    agg = aggregate_draws(dfs, group_cols; point = point)
    write_aggregated(agg, joinpath(output_dir, "$(out_name)_Aggregated.csv"))
    return nothing
end


#############################
# Section 3: Policies
#############################

# Ban folders plus every tax folder present in the results directory
BAN_TYPES = ["Ban_Comprehensive", "Ban_Non_FDA"]
tax_root  = joinpath(results_dir, "Flavor_Tax")
TAX_DIRS  = isdir(tax_root) ? sort(filter(d -> startswith(d, "Tax_") && isdir(joinpath(tax_root, d)), readdir(tax_root))) : String[]

# (label used in output names, folder holding the draws)
POLICIES = vcat(
    [(ban, joinpath(results_dir, ban)) for ban in BAN_TYPES],
    [("Flavor_Tax_$(d)", joinpath(tax_root, d)) for d in TAX_DIRS]
)

THRESH_TAGS = [("3mo", "_thresh3mo"), ("6mo", "_thresh6mo"), ("12mo", "_thresh12mo")]

# (file-name infix in the draw files, infix in the aggregated names)
INITIATION_OUTCOMES = [("", ""), ("_Cig", "_Cig"), ("_OrigEcig", "_OrigEcig")]

for (label, cf_dir) in POLICIES

    println("\n===================================")
    println("Aggregating: $label (beta = $beta_tag)")
    println("===================================")

    if !isdir(cf_dir)
        println("  Directory not found: $cf_dir; skipped")
        continue
    end

    # Simulation paths: overall and by subgroup
    pool_stem(cf_dir, "Simulation_Overall",        ["period"],          "$(label)_Overall")
    pool_stem(cf_dir, "Simulation_by_TYA",         ["group", "period"], "$(label)_by_TYA")
    pool_stem(cf_dir, "Simulation_by_Type",        ["type", "period"],  "$(label)_by_Type")
    pool_stem(cf_dir, "Simulation_by_TYA_Type",    ["type", "period"],  "$(label)_by_TYA_Type")
    pool_stem(cf_dir, "Simulation_by_NonTYA_Type", ["type", "period"],  "$(label)_by_NonTYA_Type")

    # Initiation among baseline non-users: three outcomes × four groupings × three thresholds
    for (infix_in, infix_out) in INITIATION_OUTCOMES, (thresh_tag, suffix) in THRESH_TAGS
        pool_stem(cf_dir, "Extensive_Margin$(infix_in)_by_TYA$(suffix)",         ["group", "period"], "$(label)_Extensive_Margin$(infix_out)_TYA_thresh$(thresh_tag)")
        pool_stem(cf_dir, "Extensive_Margin$(infix_in)_by_Type$(suffix)",        ["type", "period"],  "$(label)_Extensive_Margin$(infix_out)_Type_thresh$(thresh_tag)")
        pool_stem(cf_dir, "Extensive_Margin$(infix_in)_by_TYA_Type$(suffix)",    ["type", "period"],  "$(label)_Extensive_Margin$(infix_out)_TYA_Type_thresh$(thresh_tag)")
        pool_stem(cf_dir, "Extensive_Margin$(infix_in)_by_NonTYA_Type$(suffix)", ["type", "period"],  "$(label)_Extensive_Margin$(infix_out)_NonTYA_Type_thresh$(thresh_tag)")
    end

end


#############################
# Section 4: Frontier
#############################

println("\n===================================")
println("Aggregating: tax-versus-ban frontier (beta = $beta_tag)")
println("===================================")

pool_stem(results_dir, "Frontier_Summary",    ["policy", "tau", "group"],           "Frontier_Summary")
pool_stem(results_dir, "Frontier_Trajectory", ["policy", "tau", "group", "period"], "Frontier_Trajectory")
pool_stem(results_dir, "Frontier_TauStar",    ["group"],                            "Frontier_TauStar")


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
