################################################################################
# William Brasic
# The University of Arizona
# wbrasic97@gmail.com
# June 2026
#
# This script creates the necessary functions to estimate the dynamic model.
# Functions are ordered by their execution sequence in 02_Estimation.jl.
#
# K = 3 Finite Mixture 
#
# Common params (13): α_C, α_E, α_CE, λ_1, λ_2, λ_3, λ_4, γ_1, γ_2, γ_3, γ_4, ω_C, ω_E
# Type k (3 each):    ξ_C_k, ξ_E_k, ξ_CE_k  (k = 1, 2, 3)
# Mixing (4):         π_0_2 (type 2 baseline logit), π_TYA_2 (type 2 TYA shifter),
#                     π_0_3 (type 3 baseline logit), π_TYA_3 (type 3 TYA shifter)
#
# For each type k, get_flow_utility receives a 16-element vector:
#   [common_params, type_k] = [α_C, α_E, α_CE, λ_1, λ_2, λ_3, λ_4, γ_1, γ_2, γ_3, γ_4, ω_C, ω_E, ξ_C_k, ξ_E_k, ξ_CE_k]
################################################################################


################################################################################
# Table of Contents
#
#  1. Preliminaries           
#  2. Logging                
#  3. Data Loading: Choices                                  
#  4. Data Loading: Alternative Attributes                                                                       
#  5. Data Loading: Demographics                           
#  6. Data Loading: Prices                                                                                                 
#  7. Addiction Dynamics                                      
#  8. Value Function Iteration   
#  9. Likelihood & Prediction                                                              
# 10. Optimization & Objective                               
################################################################################


#############################
# 1. Preliminaries
#############################

# Install any missing packages, then load
import Pkg
for pkg in ["CSV", "DataFrames", "Optim", "Statistics"]
    if Base.find_package(pkg) === nothing
        Pkg.add(pkg)
    end
end
using CSV, DataFrames, Optim, Statistics, LinearAlgebra, Printf, Dates

# WARM_START flag: controls whether VFI reuses the previous evaluation's converged V
# as the initial guess within a Nelder-Mead run. V is reset to zeros at the start
# of each outer try (L) and inner run (M).
if !@isdefined(WARM_START)
    WARM_START = false
end

# VFI_TOL: convergence tolerance (sup-norm) for value function iteration.
# Default 1e-6 for both estimation and SE computation.
if !@isdefined(VFI_TOL)
    VFI_TOL = 1e-6
end

"""
Convert a numeric value to a string suitable for folder/file naming.
Strips trailing zeros and the decimal point, then removes any remaining periods.
Examples: 1.0 → "1", 0.68 → "068", 0.50 → "05"
"""
function numeric_tag(val)
    # Convert to string, strip trailing zeros and trailing decimal point
    s = rstrip(rstrip(string(val), '0'), '.')

    # Remove any remaining periods
    return replace(s, "." => "")
end


#############################
# 2. Logging
#############################

# Global log file handle. Each script sets this once before logging:
#   log_io = open("My_Log.txt", "w")
# All logging across estimation, MC, validation, and counterfactual uses this single handle.
log_io = nothing

# Counter for how many times the objective was called
# Each Nelder-Mead iteration can call the objective many times.
# See paper appendix for details
est_eval_count = 0

# Global optimizer phase tracking (updated by random_amoeba)
# ra_outer_try: current outer try (1 to L)
# ra_inner_run: current inner run (1 to M)
ra_outer_try = 0
ra_inner_run = 0

# Global parameter names (set by calling script before estimation)
# Used to print parameter names in objective function logging
est_param_names = String[]

# Warm-start state for the objective function (K = 3 mixture: one V per type).
# V_warm_est_1, V_warm_est_2, and V_warm_est_3 store the converged V from the previous VFI solve
# within a NM run for type 1, type 2, and type 3 respectively.
# last_ra_phase_est tracks (outer_try, inner_run); all V arrays reset when phase changes.
V_warm_est_1 = nothing
V_warm_est_2 = nothing
V_warm_est_3 = nothing
last_ra_phase_est = (0, 0)

"""
Print a message to stdout and write it to the active log file (if open).
I need "flush" b/c print/write right away and may take a minute. By
using "flush" I force Julia print to terminal and write to the file immediately.
Also, no need to pass "log_io" because it is defined at the top of this script as
a global variable.
"""
function log_msg(msg::String)
    println(msg)                # Print the message to the terminal
    flush(stdout)               # Force the terminal output to display immediately
    if log_io !== nothing       # Only write to file if a log file is open
        println(log_io, msg)    # Write the message to the log file
        flush(log_io)           # Force the log file to save to disk immediately
    end
end


#############################
# 3. Data Loading: Choices
#############################

# --- Household Codes ---

"""
Get household codes

Returns:
- Vector of household codes correspond to each observation
"""
function get_hh_codes(
    file_name::AbstractString = "./Household_Codes.csv"
)

    # Load in household codes
    df_hh_codes = CSV.read(file_name, DataFrame)

    # Get household codes from the dataframe
    hh_codes = df_hh_codes.household_code

    return hh_codes
end


# --- Product Choices ---

"""
Get household product choices in each time period

Returns:
- Number of household-month observations (N_HHT)
- Number of alternatives (N_J)
- Matrix of ones and zeros indicating the chosen alternative for all households in all time periods
"""
function get_product_choices(
    file_name::AbstractString = "./Product_Choices.csv"
)

    # Load in product choices
    J = Matrix(CSV.read(file_name, DataFrame))

    return size(J, 1), size(J, 2), J
end


"""
Get vector of choices for all households in all time periods

Returns:
- Vector of household choices (y[i] = chosen alternative index for observation i)
"""
function get_hh_choices(
    J::AbstractMatrix{<:Real}
)

    # Initialize vector to store chosen alternative index for each observation
    y = Vector{Int}(undef, size(J, 1))

    # Loop through observations
    for i in 1:size(J, 1)

        # Loop through choices
        for j in 1:size(J, 2)

            # If observation chose alternative j
            if J[i, j] == 1

                # Assign index j to the choice vector y at position i
                y[i] = j
                break
            end
        end
    end

    return y
end


# --- Category Choices ---

"""
Get household category choices in each time period

Returns:
- Number of categories (exclusive of the outside option; groups all ecig types and all bundle types for pricing)
- Matrix of ones and zeros indicating the chosen category for all households in all time periods
"""
function get_category_choices(
    file_name::AbstractString = "./Category_Choices.csv"
)

    # Load in category choices
    K = Matrix(CSV.read(file_name, DataFrame))

    return size(K, 2) - 1, K
end


#############################
# 4. Data Loading:
#    Alternative Attributes
#############################

"""
Get consumption vectors indexed by alternative j ∈ {1, ..., N_J}
Separates cigarette and e-cigarette consumption components so expenditures
and bundle consumption can be computed correctly.

Consumption is STANDARDIZED by dividing by the maximum value of each category.
This keeps utility terms at reasonable magnitudes for numerical stability.

Alternative ordering:
  j = 1:           outside option (zero consumption)
  j = 2:13:        12 cigarette quantity bins
  j = 14:20:       7 original e-cigarette bins
  j = 21:27:       7 non-FDA flavored e-cigarette bins
  j = 28:34:       7 FDA flavored e-cigarette bins
  j = 35:38:       4 original bundles (ll/lh/hl/hh, lo/hi cig x lo/hi ecig)
  j = 39:42:       4 non-FDA flavored bundles (ll/lh/hl/hh, lo/hi cig x lo/hi ecig)
  j = 43:46:       4 FDA flavored bundles (ll/lh/hl/hh, lo/hi cig x lo/hi ecig)

Returns:
- N_cig:               Number of cigarette alternatives
- N_orig_ecig:         Number of original e-cigarette alternatives
- N_non_fda_flav_ecig: Number of non-FDA flavored e-cigarette alternatives
- N_fda_flav_ecig:     Number of FDA flavored e-cigarette alternatives
- N_bundle:            Number of bundle alternatives
- q_cig:               Vector of standardized cigarette consumption for each alternative
- q_ecig:              Vector of standardized e-cigarette consumption for each alternative
- q_bundle:            Vector of standardized bundle consumption for each alternative
- q_cig_max:           Raw maximum cigarette consumption (for rescaling estimates)
- q_ecig_max:          Raw maximum e-cigarette consumption (for rescaling estimates)
- q_bundle_max:        Raw maximum bundle consumption (for rescaling estimates)
"""
function get_consumption(
    N_J::Integer;
    file_name::AbstractString = "./Consumption_Spaces.csv"
)

    # Load in consumption spaces
    df = CSV.read(file_name, DataFrame)

    # Single-good consumption values (use startswith to match any suffix)
    # Filter out bundle alternatives which also start with "cig_" in their bundle names
    cig               = df.consumption[startswith.(df.alternative, "cig_") .& .!occursin.("bundle", df.alternative)]
    orig_ecig         = df.consumption[startswith.(df.alternative, "orig_ecig_")]
    non_fda_flav_ecig = df.consumption[startswith.(df.alternative, "non_fda_flav_ecig_")]
    fda_flav_ecig     = df.consumption[startswith.(df.alternative, "fda_flav_ecig_")]

    # Bundle names in order: 4 original, 4 non-FDA flavored, 4 FDA flavored
    bundle_names = [
        "bundle_orig_ll", "bundle_orig_lh", "bundle_orig_hl", "bundle_orig_hh",
        "bundle_nfda_ll", "bundle_nfda_lh", "bundle_nfda_hl", "bundle_nfda_hh",
        "bundle_fda_ll",  "bundle_fda_lh",  "bundle_fda_hl",  "bundle_fda_hh"
    ]

    # Helper to extract single consumption value by alternative name
    get_consumption_value(alt_name) = only(df.consumption[df.alternative .== alt_name])

    # Counts
    N_cig               = length(cig)
    N_orig_ecig         = length(orig_ecig)
    N_non_fda_flav_ecig = length(non_fda_flav_ecig)
    N_fda_flav_ecig     = length(fda_flav_ecig)
    N_bundle_orig         = 4   # 4 original e-cig bundles (ll/lh/hl/hh)
    N_bundle_non_fda_flav = 4   # 4 non-FDA flavored e-cig bundles (ll/lh/hl/hh)
    N_bundle_fda_flav     = 4   # 4 FDA flavored e-cig bundles (ll/lh/hl/hh)
    N_bundle = N_bundle_orig + N_bundle_non_fda_flav + N_bundle_fda_flav

    # Initialize consumption vectors (j = 1 is outside option with zero consumption)
    q_cig  = zeros(Float64, N_J)
    q_ecig = zeros(Float64, N_J)

    # Fill vector in order: outside, cig, orig_ecig, non_fda_flav_ecig, fda_flav_ecig,
    # bundle_orig (4), bundle_non_fda_flav (4), bundle_fda_flav (4)
    idx = 2

    # Cigarette consumption
    for consumption in cig
        q_cig[idx] = consumption
        idx += 1
    end

    # Original e-cig alternatives
    for consumption in orig_ecig
        q_ecig[idx] = consumption
        idx += 1
    end

    # Non-FDA flavored e-cig alternatives
    for consumption in non_fda_flav_ecig
        q_ecig[idx] = consumption
        idx += 1
    end

    # FDA flavored e-cig alternatives
    for consumption in fda_flav_ecig
        q_ecig[idx] = consumption
        idx += 1
    end

    # Bundle consumption (12 bundles total: 4 orig + 4 non-FDA flav + 4 FDA flav)
    for bundle_name in bundle_names
        q_cig[idx]  = get_consumption_value(bundle_name * "_cig")
        q_ecig[idx] = get_consumption_value(bundle_name * "_ecig")
        idx += 1
    end

    # Compute bundle interaction BEFORE standardizing individual consumption
    # q_bundle[j] = q_cig[j] × q_ecig[j] (only non-zero for bundle alternatives)
    q_bundle_raw = q_cig .* q_ecig
    q_bundle_max = maximum(q_bundle_raw)

    # Standardize bundle INTERACTION by its own max
    # This keeps α_CE at a reasonable magnitude since bundles don't have max consumption
    # of both products simultaneously
    q_bundle = q_bundle_raw ./ q_bundle_max

    # Standardize by maximum (store raw max for rescaling estimates later)
    # Each variable is standardized by its own max to keep coefficients reasonably scaled
    q_cig_max  = maximum(q_cig)
    q_ecig_max = maximum(q_ecig)
    q_cig  ./= q_cig_max
    q_ecig ./= q_ecig_max

    return N_cig, N_orig_ecig, N_non_fda_flav_ecig, N_fda_flav_ecig, N_bundle, q_cig, q_ecig, q_bundle, q_cig_max, q_ecig_max, q_bundle_max
end


"""
Get nicotine vector indexed by alternative j ∈ {1, ..., N_J}
For bundle alternatives, total nicotine is the sum of the cigarette and e-cigarette components.

Nicotine is STANDARDIZED by dividing by the maximum value. This keeps the addiction
dynamics at reasonable magnitudes for numerical stability.

Returns:
- n: Vector of standardized nicotine (divided by max) for each alternative
- n_max: The raw maximum nicotine value (for rescaling parameter estimates)
"""
function get_nicotine(
    N_J::Integer;
    file_name::AbstractString = "./Nicotine_Spaces.csv"
)

    # Load nicotine spaces
    df = CSV.read(file_name, DataFrame)

    # Initialize nicotine vector (j = 1 is outside option with zero nicotine)
    n = zeros(Float64, N_J)

    # Single-good nicotine values (filter out bundle alternatives)
    cig               = df.nicotine[startswith.(df.alternative, "cig_") .& .!occursin.("bundle", df.alternative)]
    orig_ecig         = df.nicotine[startswith.(df.alternative, "orig_ecig_")]
    non_fda_flav_ecig = df.nicotine[startswith.(df.alternative, "non_fda_flav_ecig_")]
    fda_flav_ecig     = df.nicotine[startswith.(df.alternative, "fda_flav_ecig_")]

    # Bundle nicotine components (12 bundles: 4 orig + 4 non-FDA flav + 4 FDA flav,
    # each with cig and ecig components)
    # Bundle pairs: (cig_nicotine, ecig_nicotine) for each bundle alternative
    # Order: 4 original, then 4 non-FDA flavored, then 4 FDA flavored
    bundle_pair_names = [
        ("bundle_orig_ll_cig_nic", "bundle_orig_ll_ecig_nic"),
        ("bundle_orig_lh_cig_nic", "bundle_orig_lh_ecig_nic"),
        ("bundle_orig_hl_cig_nic", "bundle_orig_hl_ecig_nic"),
        ("bundle_orig_hh_cig_nic", "bundle_orig_hh_ecig_nic"),
        ("bundle_nfda_ll_cig_nic", "bundle_nfda_ll_ecig_nic"),
        ("bundle_nfda_lh_cig_nic", "bundle_nfda_lh_ecig_nic"),
        ("bundle_nfda_hl_cig_nic", "bundle_nfda_hl_ecig_nic"),
        ("bundle_nfda_hh_cig_nic", "bundle_nfda_hh_ecig_nic"),
        ("bundle_fda_ll_cig_nic",  "bundle_fda_ll_ecig_nic"),
        ("bundle_fda_lh_cig_nic",  "bundle_fda_lh_ecig_nic"),
        ("bundle_fda_hl_cig_nic",  "bundle_fda_hl_ecig_nic"),
        ("bundle_fda_hh_cig_nic",  "bundle_fda_hh_ecig_nic"),
    ]
    nic_lookup = Dict(row.alternative => row.nicotine for row in eachrow(df))
    bundle_pairs = Tuple((nic_lookup[c], nic_lookup[e]) for (c, e) in bundle_pair_names)

    # Fill vector in order: outside, cig, orig_ecig, non_fda_flav_ecig, fda_flav_ecig,
    # bundle_orig (4), bundle_non_fda_flav (4), bundle_fda_flav (4)
    idx = 2

    # Cigarette nicotine
    for nicotine in cig
        n[idx] = nicotine
        idx += 1
    end

    # Original e-cig nicotine
    for nicotine in orig_ecig
        n[idx] = nicotine
        idx += 1
    end

    # Non-FDA flavored e-cig nicotine
    for nicotine in non_fda_flav_ecig
        n[idx] = nicotine
        idx += 1
    end

    # FDA flavored e-cig nicotine
    for nicotine in fda_flav_ecig
        n[idx] = nicotine
        idx += 1
    end

    # Bundle nicotine (sum of cig + ecig components for each bundle)
    for (cig_nic, ecig_nic) in bundle_pairs
        n[idx] = cig_nic + ecig_nic
        idx += 1
    end

    # Standardize by maximum (store raw max for rescaling estimates later)
    n_max = maximum(n)
    n ./= n_max

    return n, n_max
end


"""
Get category index for each alternative j ∈ {1, ..., N_J}.

Category mapping (8 categories, 0-indexed):
  0 = outside option
  1 = cigarettes
  2 = original e-cigarettes
  3 = non-FDA flavored e-cigarettes
  4 = FDA flavored e-cigarettes
  5 = bundle (cig + original ecig)
  6 = bundle (cig + non-FDA flavored ecig)
  7 = bundle (cig + FDA flavored ecig)

Returns:
- cat_idx: Vector of category indices for each alternative
"""
function get_category_index(
    N_J::Integer,
    N_cig::Integer,
    N_orig_ecig::Integer,
    N_non_fda_flav_ecig::Integer,
    N_fda_flav_ecig::Integer
)

    # Number of bundle alternatives (4 orig + 4 non-FDA flav + 4 FDA flav = 12 total)
    N_bundle_orig         = 4
    N_bundle_non_fda_flav = 4
    N_bundle_fda_flav     = 4

    # Initialize category index vector (j = 1 is outside option with cat = 0)
    cat_idx = zeros(Int, N_J)

    # Fill vector in order: outside, cig, orig_ecig, non_fda_flav_ecig, fda_flav_ecig,
    # bundle_orig (4), bundle_non_fda_flav (4), bundle_fda_flav (4)
    idx = 2

    # Cigarette alternatives (cat = 1)
    for _ in 1:N_cig
        cat_idx[idx] = 1
        idx += 1
    end

    # Original e-cig alternatives (cat = 2)
    for _ in 1:N_orig_ecig
        cat_idx[idx] = 2
        idx += 1
    end

    # Non-FDA flavored e-cig alternatives (cat = 3)
    for _ in 1:N_non_fda_flav_ecig
        cat_idx[idx] = 3
        idx += 1
    end

    # FDA flavored e-cig alternatives (cat = 4)
    for _ in 1:N_fda_flav_ecig
        cat_idx[idx] = 4
        idx += 1
    end

    # Bundle with original e-cig (cat = 5) - 4 alternatives
    for _ in 1:N_bundle_orig
        cat_idx[idx] = 5
        idx += 1
    end

    # Bundle with non-FDA flavored e-cig (cat = 6) - 4 alternatives
    for _ in 1:N_bundle_non_fda_flav
        cat_idx[idx] = 6
        idx += 1
    end

    # Bundle with FDA flavored e-cig (cat = 7) - 4 alternatives
    for _ in 1:N_bundle_fda_flav
        cat_idx[idx] = 7
        idx += 1
    end

    return cat_idx
end


"""
Get FDA flavored indicator for each alternative j = 1, ..., N_J.
FDA flavored alternatives are: FDA flavored e-cigarette bins (cat = 4)
and the cig + FDA flavored ecig bundle (cat = 7).

Returns:
- is_fda_flavored: Vector where true indicates a FDA flavored alternative
"""
function get_fda_flavored_indicator(
    cat_idx::AbstractVector{<:Integer}
)

    # FDA flavored categories are 4 (FDA flav ecig) and 7 (bundle with FDA flav ecig)
    return (cat_idx .== 4) .| (cat_idx .== 7)
end


"""
Get flavored indicator for each alternative j = 1, ..., N_J.
Flavored alternatives are: any flavored e-cigarette bins (cat ∈ {3, 4}) and
any cig + flavored ecig bundle (cat ∈ {6, 7}). Union of non-FDA and FDA flavored.

Returns:
- is_flavored: Vector where true indicates a flavored alternative (non-FDA or FDA)
"""
function get_flavored_indicator(
    cat_idx::AbstractVector{<:Integer}
)

    # Flavored categories are 3, 4 (non-FDA/FDA flav ecig) and 6, 7 (bundles with flav ecig)
    return (cat_idx .== 3) .| (cat_idx .== 4) .| (cat_idx .== 6) .| (cat_idx .== 7)
end


#############################
# 5. Data Loading:
#    Demographics
#############################

"""
Get indicators for whether teen or young adult is present in the household.

Returns:
- Number of unique households
- Vector of ones and zeros indicating whether a given household has a teen or young adult present
"""
function get_teen_young_adult(
    file_name::AbstractString = "./Teen_Young_Adult.csv"
)

    # Load in teen or young adult indicator for each household-month
    df_tya = CSV.read(file_name, DataFrame)

    # Vector of indicators for each household for each month they are in the panel
    tya = Vector(df_tya[:, 3])

    # Unique households
    N_HH = length(unique(df_tya.household_code))

    return N_HH, tya
end


"""
Get TYA classification for each observation.

Loads pre-computed TYA state assignments 

Returns:
- tya_state: Vector{Int} of binary state values (0 = no TYA, 1 = TYA present) for each observation
"""
function get_tya_states(
    file_name::AbstractString = "./TYA_States.csv"
)

    df = CSV.read(file_name, DataFrame)
    return Vector{Int}(df.tya_state)
end


"""
Compute household-level TYA share (fraction of months with TYA present).

Reads TYA_States.csv which has columns (household_code, purchase_month,
tya_state). 

For each household, computes share_h = (# of months with TYA present) / (# of total months in panel)

Returns a per-household vector of TYA shares, one entry per household
(in the same order as unique household codes appear in the data).
"""
function get_tya_share(
    file_name::AbstractString = "./TYA_States.csv"
)

    # Load TYA data (household_code, purchase_month, tya_state)
    df = CSV.read(file_name, DataFrame)

    # Get unique household codes in order of first appearance
    hh_codes = df.household_code
    tya_vals = df.tya_state

    # Compute per-household TYA share by iterating through contiguous household blocks
    tya_share_hh = Float64[]
    N = length(hh_codes)
    i = 1
    while i <= N

        # Current household code
        hh = hh_codes[i]

        # Mark the start of this household's observation block
        start_idx = i

        # Accumulate TYA indicator across this household's months
        tya_sum = 0.0
        while i <= N && hh_codes[i] == hh
            tya_sum += tya_vals[i]
            i += 1
        end

        # Compute share = (months with TYA) / (total months in panel)
        n_months = i - start_idx
        push!(tya_share_hh, tya_sum / n_months)
    end

    return tya_share_hh
end


#############################
# 6. Data Loading: Prices
#############################

# --- Fixed Parameters ---

"""
Set fixed parameters

Returns:
- Fast addiction decay rate ψ_2 (fixed at 0.90)
- Slow addiction decay rate ψ_1 (fixed at 0.10)
- Present bias term β (fixed at 1.0, standard exponential discounting)
- Monthly discount factor δ (fixed at 0.99)
"""
function get_fixed_parameters()

    # Fast addiction decay rate ("craving" stock)
    ψ_2 = 0.90

    # Slow addiction decay rate ("dependence" stock)
    ψ_1 = 0.10

    # Present bias parameter (β-δ discounting; β = 1.0 is standard exponential)
    β = 1.0

    # Monthly discount factor
    δ = 0.99

    return ψ_2, ψ_1, β, δ
end


# --- Pricing Spaces ---

"""
Get pricing space for each category

Returns:
- Number of price grid points per category
- Matrix of price vectors for each category (N_P × 2, columns = cig, ecig)
"""
function get_pricing_spaces(
    file_name::AbstractString = "./Pricing_Spaces.csv"
)

    # Load in pricing spaces
    df_prices = CSV.read(file_name, DataFrame)

    # Extract pricing space for all categories
    P = Matrix(df_prices[:, 2:end])

    return size(P, 1), P
end


"""
Get combination of all points in the pricing space

Returns:
- Number of combined price vectors (N_P²)
- Matrix of price vectors (N_Pcomb × 2, columns = cig, ecig)
"""
function get_pricing_spaces_combination(
    N_K::Integer,
    N_P::Integer,
    P::AbstractMatrix{<:Real}
)

    # Combination of all possible prices across cig and ecig
    Pcomb = zeros(eltype(P), N_P^(N_K - 1), N_K - 1)
    idx = 1
    for i in 1:N_P
        for j in 1:N_P
            Pcomb[idx, 1] = P[i, 1]   # cig
            Pcomb[idx, 2] = P[j, 2]   # ecig
            idx += 1
        end
    end

    return size(Pcomb, 1), Pcomb
end


# --- Price Ratios ---

"""
Get price ratios for quantity discount adjustment.

Price ratios capture that per-unit prices vary systematically across quantity bins:
small-quantity bins pay more per unit (ratio > 1), large-quantity bins pay less (ratio < 1).

Reads Price_Ratios.csv which contains alternative, median_price, overall_median, ratio

Returns two vectors indexed by alternative j ∈ {1, ..., N_J}:
- ratio_cig:  Cig price ratio for each alternative (1.0 for non-cig components)
- ratio_ecig: Ecig price ratio for each alternative (1.0 for non-ecig components)
"""
function get_price_ratios(
    N_J::Integer,
    N_cig::Integer,
    N_orig_ecig::Integer,
    N_non_fda_flav_ecig::Integer,
    N_fda_flav_ecig::Integer,
    q_cig::AbstractVector{<:Real},
    q_ecig::AbstractVector{<:Real};
    file_name::AbstractString = "./Price_Ratios.csv"
)

    # Load price ratios CSV (contains only standalone cig and ecig bin ratios)
    df = CSV.read(file_name, DataFrame)

    # Build lookup: alternative name → ratio
    ratio_lookup = Dict{String, Float64}()
    for row in eachrow(df)
        ratio_lookup[row.alternative] = row.ratio
    end

    # Initialize ratio vectors (default = 1.0, so zero-consumption terms are unaffected)
    ratio_cig  = ones(Float64, N_J)
    ratio_ecig = ones(Float64, N_J)

    # Cig bin names 
    cig_names = ["cig_1", "cig_2", "cig_3to4", "cig_5to9", "cig_10", "cig_11to19",
                 "cig_20", "cig_21to29", "cig_30", "cig_31to39", "cig_40", "cig_41plus"]

    # Ecig bin names (pooled orig/flav; single ecig price state)
    ecig_names = ["ecig_0to5", "ecig_5to10", "ecig_10to15", "ecig_15to20",
                  "ecig_20to30", "ecig_30to50", "ecig_50plus"]

    # Fill ratio vectors following the alternative ordering:
    # j=1: outside option (ratios stay 1.0)
    idx = 2

    # j=2:13: cigarette bins
    for name in cig_names
        ratio_cig[idx] = ratio_lookup[name]
        idx += 1
    end

    # j=14:20: original e-cig bins (use pooled ecig ratios)
    for name in ecig_names
        ratio_ecig[idx] = ratio_lookup[name]
        idx += 1
    end

    # j=21:27: non-FDA flavored e-cig bins (same pooled ecig ratios)
    for name in ecig_names
        ratio_ecig[idx] = ratio_lookup[name]
        idx += 1
    end

    # j=28:34: FDA flavored e-cig bins (same pooled ecig ratios)
    for name in ecig_names
        ratio_ecig[idx] = ratio_lookup[name]
        idx += 1
    end

    # Standalone bin index ranges (for matching bundle consumption to closest bin)
    cig_range  = 2:(1 + N_cig)                           # j=2:13
    ecig_range = (2 + N_cig):(1 + N_cig + N_orig_ecig)   # j=14:20 (same ratios as all ecig bins)

    # j=35:46: bundles; map each bundle's consumption to the closest standalone bin's ratio
    for j in idx:N_J
        # Cig component: find standalone cig bin with closest consumption
        if q_cig[j] > 0.0
            best_cig = argmin(abs(q_cig[k] - q_cig[j]) for k in cig_range)
            ratio_cig[j] = ratio_cig[cig_range[best_cig]]
        end

        # Ecig component: find closest ecig bin (ratios are pooled across orig/flav)
        if q_ecig[j] > 0.0
            best_ecig = argmin(abs(q_ecig[k] - q_ecig[j]) for k in ecig_range)
            ratio_ecig[j] = ratio_ecig[ecig_range[best_ecig]]
        end
    end

    return ratio_cig, ratio_ecig
end


# --- Expenditures ---

"""
Get expenditure matrix indexed by combined price state p ∈ {1, ..., N_Pcomb}
and alternative j ∈ {1, ..., N_J}.

Expenditure for alternative j at combined price state p is:
  E[p, j] = p_cig(p) × ratio_cig(j) × q_cig(j) + p_ecig(p) × ratio_ecig(j) × q_ecig(j)

Price ratios capture quantity discounts: small-quantity bins have ratio > 1 (more expensive
per unit) and large-quantity bins have ratio < 1 (cheaper per unit).

This correctly handles all cases:
- Outside option: 0 (both consumption vectors are zero)
- Cig-only: p_cig × ratio_cig × q_cig (q_ecig is zero)
- Ecig-only: p_ecig × ratio_ecig × q_ecig (q_cig is zero)
- Bundles: p_cig × ratio_cig × q_cig + p_ecig × ratio_ecig × q_ecig

Expenditure is computed using RAW consumption (unstandardized) so that E is in
actual dollars, then STANDARDIZED by dividing by the maximum expenditure.

Returns:
- E: N_Pcomb × N_J matrix of standardized expenditures
- E_max: Raw maximum expenditure (for rescaling parameter estimates)
"""
function get_expenditures(
    N_J::Integer,
    N_Pcomb::Integer,
    q_cig::AbstractVector{<:Real},
    q_ecig::AbstractVector{<:Real},
    q_cig_max::Real,
    q_ecig_max::Real,
    Pcomb::AbstractMatrix{<:Real},
    ratio_cig::AbstractVector{<:Real},
    ratio_ecig::AbstractVector{<:Real}
)

    # Reconstruct raw consumption from standardized values
    q_cig_raw  = q_cig .* q_cig_max
    q_ecig_raw = q_ecig .* q_ecig_max

    # Initialize expenditure matrix
    E = zeros(Float64, N_Pcomb, N_J)

    # Fill expenditure matrix using raw consumption and price ratios (actual dollars)
    for p in 1:N_Pcomb
        p_cig  = Pcomb[p, 1]
        p_ecig = Pcomb[p, 2]
        for j in 1:N_J
            E[p, j] = p_cig * ratio_cig[j] * q_cig_raw[j] + p_ecig * ratio_ecig[j] * q_ecig_raw[j]
        end
    end

    # Standardize by maximum (store raw max for rescaling estimates later)
    E_max = maximum(E)
    E ./= E_max

    return E, E_max
end


# --- Price Transitions ---

"""
Get price transitions from Halton draws

Returns:
- Array of transitions (M × R × 2, dimensions = price vector, Halton draw, category)
"""
function get_transitions(
    N_K::Integer;
    file_name::AbstractString = "./Halton_Draw_Transitions.csv"
)

    # Load in transitions
    df_transitions = CSV.read(file_name, DataFrame)

    # Get number of price vectors (M) and Halton draws (R)
    M = maximum(df_transitions.m)
    R = maximum(df_transitions.r)

    # Get category-level transitions
    T = Array{Float64}(undef, M, R, N_K - 1)
    for row in eachrow(df_transitions)
        m = row.m
        r = row.r
        T[m, r, 1] = row.cig
        T[m, r, 2] = row.ecig
    end

    return T
end


"""
Pre-compute price transition interpolation brackets and weights for bilinear
interpolation of the value function over predicted next-period prices.

The interpolation weights are for the upper term.

For each current combined price state m and Halton draw r, the predicted
next-period prices T[m, r, :] are bracketed on each category's 1D price grid.
Out-of-bounds predictions are clamped to the grid endpoints. The function works as follows:
- If T[m, r, k] < lower bound, assign predicted price to the lower bound
- If T[m, r, k] is in the grid, no change
- If T[m, r, k] > upper bound, assign predicted price to the upper bound

The searchsortedfirst function takes in the grid for the first argument and the point
in consideration for the second argument. It finds the grid point directly above the point
in consideration and returns the index of that point in the grid.

@view looks at the matrix in P currently stored in memory, rather than
creating a new copy of it. Simply more efficient

Returns:
- p_cig_lo:  M × R matrix of lower bracket indices on cigarette price grid
- p_cig_hi:  M × R matrix of upper bracket indices on cigarette price grid
- p_cig_w:   M × R matrix of interpolation weights for upper bracket (cigarette)
- p_ecig_lo: M × R matrix of lower bracket indices on e-cigarette price grid
- p_ecig_hi: M × R matrix of upper bracket indices on e-cigarette price grid
- p_ecig_w:  M × R matrix of interpolation weights for upper bracket (e-cigarette)
"""
function precompute_price_transitions(
    N_P::Integer,
    P::AbstractMatrix{<:Real},
    T::Array{Float64, 3}
)

    # Dimensions
    M = size(T, 1)
    R = size(T, 2)

    # 1D price grids
    P_cig  = @view P[:, 1]
    P_ecig = @view P[:, 2]

    # Initialize output arrays
    p_cig_lo  = Matrix{Int}(undef, M, R)
    p_cig_hi  = Matrix{Int}(undef, M, R)
    p_cig_w   = Matrix{Float64}(undef, M, R)
    p_ecig_lo = Matrix{Int}(undef, M, R)
    p_ecig_hi = Matrix{Int}(undef, M, R)
    p_ecig_w  = Matrix{Float64}(undef, M, R)

    # Loop over combined price states
    for m in 1:M

        # Loop over Halton draws
        for r in 1:R

            # Clamp predicted prices to grid bounds
            pred_cig  = clamp(T[m, r, 1], P_cig[1], P_cig[end])
            pred_ecig = clamp(T[m, r, 2], P_ecig[1], P_ecig[end])

            # Cigarette price brackets via binary search
            hi_c = clamp(searchsortedfirst(P_cig, pred_cig), 1, N_P)
            lo_c = clamp(hi_c - 1, 1, N_P)
            p_cig_lo[m, r] = lo_c
            p_cig_hi[m, r] = hi_c
            p_cig_w[m, r]  = (lo_c == hi_c) ? 0.0 : (pred_cig - P_cig[lo_c]) / (P_cig[hi_c] - P_cig[lo_c])

            # E-cigarette price brackets via binary search
            hi_e = clamp(searchsortedfirst(P_ecig, pred_ecig), 1, N_P)
            lo_e = clamp(hi_e - 1, 1, N_P)
            p_ecig_lo[m, r] = lo_e
            p_ecig_hi[m, r] = hi_e
            p_ecig_w[m, r]  = (lo_e == hi_e) ? 0.0 : (pred_ecig - P_ecig[lo_e]) / (P_ecig[hi_e] - P_ecig[lo_e])
        end
    end

    return p_cig_lo, p_cig_hi, p_cig_w, p_ecig_lo, p_ecig_hi, p_ecig_w
end


# --- Map Prices to Grid ---

"""
Map observed household prices to the nearest combined price grid index to
get the observed price state for each observation for likelihood computation.

For the dynamic model's price state, I compute a representative price per
category by taking the median across bins.

Combined grid ordering follows 04_State_Transitions.R:
  combined_idx = (cig_idx - 1) * N_P + ecig_idx
where cig_idx varies slowly and ecig_idx varies fast.

Returns:
- p_state:      Vector{Int} of combined price grid indices for each observation
- p_continuous: N × 2 matrix of representative (cig, ecig) prices for likelihood interpolation
- P_obs_cig:    N × N_J matrix of actual cig price for each obs-alternative pair (0 if no cig component)
- P_obs_ecig:   N × N_J matrix of actual ecig price for each obs-alternative pair (0 if no ecig component)
"""
function map_prices_to_grid(
    N_P::Integer,
    P::AbstractMatrix{<:Real},
    Pcomb::AbstractMatrix{<:Real},
    N_J::Integer;
    file_name::AbstractString = "./Prices.csv"
)

    # Load bin-specific prices
    df_prices = CSV.read(file_name, DataFrame)
    N = nrow(df_prices)

    # Pricing grids for each category
    P_cig  = @view P[:, 1]
    P_ecig = @view P[:, 2]

    # Column names for each category's bins
    cig_cols = [:cig_1_p, :cig_2_p, :cig_3to4_p, :cig_5to9_p, :cig_10_p, :cig_11to19_p,
                :cig_20_p, :cig_21to29_p, :cig_30_p, :cig_31to39_p, :cig_40_p, :cig_41plus_p]
    ecig_cols = [:ecig_0to5_p, :ecig_5to10_p, :ecig_10to15_p, :ecig_15to20_p,
                 :ecig_20to30_p, :ecig_30to50_p, :ecig_50plus_p]

    # Initialize outputs
    p_state      = Vector{Int}(undef, N)
    p_continuous = Matrix{Float64}(undef, N, 2)

    # Map each observation to nearest grid point
    for i in 1:N

        # Representative per-category price: median across bins
        obs_cig  = median([df_prices[i, c] for c in cig_cols])
        obs_ecig = median([df_prices[i, c] for c in ecig_cols])

        # Store continuous prices (for interpolation in likelihood)
        p_continuous[i, 1] = obs_cig
        p_continuous[i, 2] = obs_ecig

        # Nearest grid index for each category
        cig_idx  = argmin(abs.(P_cig  .- obs_cig))
        ecig_idx = argmin(abs.(P_ecig .- obs_ecig))

        # Map (cig_idx, ecig_idx) to combined index
        p_state[i] = (cig_idx - 1) * N_P + ecig_idx
    end

    # Build full observation × alternative price matrices for the P_obs correction in the likelihood.
    # The VFI uses p_continuous[i,1] as the effective cig price for all alternatives at obs i; the correction
    # replaces this with the actual bin-specific price P_obs[i,j] from Prices.csv.
    # P_obs_cig[i,j]  = actual cig price for obs i, alternative j (0 for non-cig alternatives)
    # P_obs_ecig[i,j] = actual ecig price for obs i, alternative j (0 for non-ecig alternatives)
    P_obs_cig  = zeros(Float64, N, N_J)
    P_obs_ecig = zeros(Float64, N, N_J)

    # j=2:13: standalone cig bins
    for (b, col) in enumerate(cig_cols)
        j = 1 + b   # j=2 through j=13
        for i in 1:N
            P_obs_cig[i, j] = df_prices[i, col]
        end
    end

    # j=14:34: standalone ecig bins (orig, non-FDA flav, FDA flav all share pooled ecig prices)
    for (b, col) in enumerate(ecig_cols)
        for i in 1:N
            P_obs_ecig[i, 13 + b] = df_prices[i, col]   # j=14:20 orig ecig
            P_obs_ecig[i, 20 + b] = df_prices[i, col]   # j=21:27 non-FDA flav ecig
            P_obs_ecig[i, 27 + b] = df_prices[i, col]   # j=28:34 FDA flav ecig
        end
    end

    # j=35:46: bundle alternatives (separate cig and ecig component prices)
    bundle_cig_cols  = [:bundle_orig_ll_cig_p, :bundle_orig_lh_cig_p, :bundle_orig_hl_cig_p, :bundle_orig_hh_cig_p,
                        :bundle_nfda_ll_cig_p, :bundle_nfda_lh_cig_p, :bundle_nfda_hl_cig_p, :bundle_nfda_hh_cig_p,
                        :bundle_fda_ll_cig_p,  :bundle_fda_lh_cig_p,  :bundle_fda_hl_cig_p,  :bundle_fda_hh_cig_p]
    bundle_ecig_cols = [:bundle_orig_ll_ecig_p, :bundle_orig_lh_ecig_p, :bundle_orig_hl_ecig_p, :bundle_orig_hh_ecig_p,
                        :bundle_nfda_ll_ecig_p, :bundle_nfda_lh_ecig_p, :bundle_nfda_hl_ecig_p, :bundle_nfda_hh_ecig_p,
                        :bundle_fda_ll_ecig_p,  :bundle_fda_lh_ecig_p,  :bundle_fda_hl_ecig_p,  :bundle_fda_hh_ecig_p]
    for (b, (ccol, ecol)) in enumerate(zip(bundle_cig_cols, bundle_ecig_cols))
        j = 34 + b   # j=35 through j=46
        for i in 1:N
            P_obs_cig[i, j]  = df_prices[i, ccol]
            P_obs_ecig[i, j] = df_prices[i, ecol]
        end
    end

    return p_state, p_continuous, P_obs_cig, P_obs_ecig
end


#############################
# 7. Addiction Dynamics
#############################

"""
Get addiction space

The addiction grid is normalized to [0, 1] regardless of ψ. This is achieved
by measuring addiction in units of ã = a·ψ, where a is the raw standardized
addiction stock. The steady state of ã' = (1-ψ)ã + ψ·n is ã = n, so when
n ∈ [0, 1], ã ∈ [0, 1]. This keeps the addiction state on the same scale as
all other standardized variables (consumption, nicotine, expenditure).

Returns:
- Number of addiction states
- Vector of addiction states (normalized to [0, 1])
"""
function get_addiction_space(
    ψ::Real;
    N_A::Integer = 20
)

    # Addiction grid is always [0, 1] in normalized units (ã = a·ψ)
    left_endpoint  = 0.0
    right_endpoint = 1.0

    # Create the addiction grid
    A_grid = collect(range(left_endpoint, right_endpoint, N_A))

    return N_A, A_grid
end


"""
Addiction law of motion in normalized units.

All addiction values in the code are ã (normalized), NOT raw a.
The normalization ã = ψ·a_raw/n_max maps addiction to [0, 1].

Raw law of motion:    a_raw' = (1-ψ)·a_raw + n_raw
Divide by n_max:      a_std' = (1-ψ)·a_std + n_std        where a_std = a_raw/n_max, grid [0, 1/ψ]
Multiply by ψ:        ã'     = (1-ψ)·ã     + ψ·n_std      where ã = ψ·a_std,        grid [0, 1]

The ψ·n term is the ONLY difference from the raw law of motion. It scales the
nicotine input so that the steady state is ã_ss = n_std ∈ [0, 1] instead of
a_std_ss = n_std/ψ ∈ [0, 1/ψ].

Arguments:
- ψ: addiction decay rate
- a: current NORMALIZED addiction ã ∈ [0, 1] (not raw a)
- n: STANDARDIZED nicotine n_std = n_raw/n_max ∈ [0, 1]

Returns:
- ã' = (1 - ψ) * ã + ψ * n_std
"""
function addiction_evolution(
    ψ::Real,
    a::Real,    # this is ã (normalized addiction), NOT raw a
    n::Real     # this is n_std (standardized nicotine), NOT raw n
)

    # ã' = (1-ψ)·ã + ψ·n_std
    # `a` here is already ̃a, so no additional ψ multiplication on the decay term.
    return (1 - ψ) * a + ψ * n
end


"""
Pre-compute flow utility for all (tya, alternative, addiction, flavored habit, price) states.

The flow utility for alternative j given addiction state a, flavored habit a_flav,
price state p, and TYA state is:

  u(j,a,a_flav,p,tya) = α_C·q_cig[j] + α_E·q_ecig[j] + α_CE·q_cig[j]·q_ecig[j]
                       + 1[flavored]·(λ_1 + λ_2·𝟙[tya]) 
                       + 1[FDA flavored]·(λ_3 + λ_4·𝟙[tya])
                       + γ_1·a·1[outside option]
                       + γ_2·a_flav·1[orig ecig/bundle inside]
                       + γ_3·a_flav·1[cig or orig bundle]
                       + γ_4·a_flav·1[outside option]
                       + ω_C·P_cig[p]·q_cig[j] + ω_E·P_ecig[p]·q_ecig[j]
                       + ξ_vec[j]


γ_1·a is withdrawal cost (outside option only).
γ_2·a_flav·1[orig ecig/bundle inside] is a flavor lock-in penalty on orig ecig and orig bundle.
γ_3·a_flav·1[cig or orig bundle] is a flavor lock-in penalty on standalone cigs and orig bundles
γ_4·a_flav·1[outside option] is a flavored withdrawal cost on the outside option.
ω_C·P_cig[p]·q_cig[j] is cigarette expenditure disutility.
ω_E·P_ecig[p]·q_ecig[j] is e-cigarette expenditure disutility.
Bundles (cat 5-7) get BOTH price terms.
For the outside option (j = 1), all terms except γ_1·a and γ_4·a_flav are zero.

θ is a 16-element vector: [α_C, α_E, α_CE, λ_1, λ_2, λ_3, λ_4, γ_1, γ_2, γ_3, γ_4, ω_C, ω_E, ξ_C, ξ_E, ξ_CE].


Two-stock addiction model: the addiction level entering utility is the unweighted
average of the fast and slow stocks: a = (A_f[af_idx] + A_s[as_idx]) / 2.
Flavored habit stock a_flav is a separate state variable.

Returns:
- U: 6D array of flow utilities, U[tya_idx, j, af_idx, as_idx, aflav_idx, p_idx]
      Dimensions: 2 × N_J × N_A_f × N_A_s × N_A_flav × N_Pcomb
"""
function get_flow_utility(
    θ::AbstractVector{<:Real},
    N_J::Integer,
    N_A_f::Integer,
    N_A_s::Integer,
    N_A_flav::Integer,
    N_Pcomb::Integer,
    A_f::AbstractVector{<:Real},
    A_s::AbstractVector{<:Real},
    A_flav::AbstractVector{<:Real},
    q_cig::AbstractVector{<:Real},
    q_ecig::AbstractVector{<:Real},
    q_bundle::AbstractVector{<:Real},
    is_flavored::AbstractVector{Bool},
    is_fda_flavored::AbstractVector{Bool},
    is_nonflavored_ecig::AbstractVector{Bool},
    is_outside::AbstractVector{Bool},
    cat_idx::AbstractVector{<:Integer},
    Pcomb::AbstractMatrix{<:Real},
    has_cig::AbstractVector{Bool},
    has_ecig::AbstractVector{Bool}
)

    # Unpack parameters (16-element θ: 13 common + 3 type-specific)
    α_C, α_E, α_CE, λ_1, λ_2, λ_3, λ_4, γ_1, γ_2, γ_3, γ_4, ω_C, ω_E, ξ_C, ξ_E, ξ_CE = θ

    # Construct per-alternative fixed effect vector from category-level ξ parameters
    ξ_vec = zeros(Float64, N_J)
    for j in 1:N_J
        if cat_idx[j] == 1
            ξ_vec[j] = ξ_C
        elseif cat_idx[j] in (2, 3, 4)
            ξ_vec[j] = ξ_E
        elseif cat_idx[j] in (5, 6, 7)
            ξ_vec[j] = ξ_CE
        end
    end

    # Number of TYA states (binary: 1 = no TYA, 2 = TYA present)
    N_TYA = 2

    # Initialize flow utility array (6D: TYA × alternatives × fast addiction × slow addiction × flavored habit × price)
    U = zeros(Float64, N_TYA, N_J, N_A_f, N_A_s, N_A_flav, N_Pcomb)

    # Fill flow utility array starting with price state
    for p_idx in 1:N_Pcomb

        # Loop over slow addiction states
        for as_idx in 1:N_A_s

            # Loop over flavored habit states
            for aflav_idx in 1:N_A_flav

                # Loop over fast addiction states
                for af_idx in 1:N_A_f

                    # Addiction level is the unweighted average of fast and slow stocks
                    a = (A_f[af_idx] + A_s[as_idx]) / 2.0

                    # Loop over alternatives
                    for j_idx in 1:N_J

                        # Components that do not depend on TYA state
                        u_base = (α_C * q_cig[j_idx] + α_E * q_ecig[j_idx] + α_CE * q_bundle[j_idx]
                                 + ω_C * Pcomb[p_idx, 1] * q_cig[j_idx]
                                 + ω_E * Pcomb[p_idx, 2] * q_ecig[j_idx]
                                 + ξ_vec[j_idx])

                        # Withdrawal cost: γ_1·a applies only to the outside option.
                        if j_idx == 1
                            u_base += γ_1 * a
                        end

                        # Flavor lock-in penalty for orig ecig/bundle
                        u_base += γ_2 * A_flav[aflav_idx] * is_nonflavored_ecig[j_idx]

                        # Flavor lock-in penalty for cig/orig bundle
                        u_base += γ_3 * A_flav[aflav_idx] * (has_cig[j_idx] && !is_flavored[j_idx])

                        # Flavored withdrawal cost: γ_4·a_flav applies to outside option 
                        u_base += γ_4 * A_flav[aflav_idx] * is_outside[j_idx]

                        # Loop over binary TYA states (1 = no TYA, 2 = TYA present)
                        for tya_idx in 1:N_TYA

                            # TYA indicator (state 1 → tya=0, state 2 → tya=1)
                            tya = tya_idx - 1

                            # Start from base utility
                            u_tya = u_base

                            # Flavor effects: λ₁,λ₂ for all flavored; λ₃,λ₄ additional for FDA-authorized
                            u_tya += is_flavored[j_idx] * (λ_1 + λ_2 * tya) + is_fda_flavored[j_idx] * (λ_3 + λ_4 * tya)

                            # Assign flow utility
                            U[tya_idx, j_idx, af_idx, as_idx, aflav_idx, p_idx] = u_tya
                        end
                    end
                end
            end
        end
    end

    return U
end


"""
Pre-compute addiction transition grid indices and interpolation weights
for each (alternative, addiction state) pair. Used by the VFI to look up
continuation values at the next-period addiction state, which generally
falls between grid points.

For alternative j and addiction state a_idx:
  ã' = (1 - ψ) * A[a_idx] + ψ * n[j]

The result is bracketed between A[lo] and A[hi] with interpolation weight w:
  V(a') ≈ (1 - w) * V(A[lo]) + w * V(A[hi])

Returns:
- a_lower:  N_J × N_A matrix of lower bracket grid indices
- a_upper:  N_J × N_A matrix of upper bracket grid indices
- a_weight: N_J × N_A matrix of interpolation weights for upper bracket
"""
function precompute_addiction_transitions(
    N_J::Integer,
    N_A::Integer,
    ψ::Real,
    A::AbstractVector{<:Real},
    n::AbstractVector{<:Real}
)

    # Initialize output matrices
    a_lower  = zeros(Int, N_J, N_A)
    a_upper  = zeros(Int, N_J, N_A)
    a_weight = zeros(Float64, N_J, N_A)

    # Loop over alternatives
    for j_idx in 1:N_J

        # Loop over addiction states
        for a_idx in 1:N_A

            # Next-period addiction level given current addiction and current choice
            a_prime = addiction_evolution(ψ, A[a_idx], n[j_idx])

            # Find upper bracket index via binary search
            hi_raw = searchsortedfirst(A, a_prime)

            # Get the index right below and right above where a′ lands in the addiction grid
            lo = clamp(hi_raw - 1, 1, N_A)
            hi = clamp(hi_raw, 1, N_A)

            # Assign these indices to the lower and upper addiction matrices
            a_lower[j_idx, a_idx] = lo
            a_upper[j_idx, a_idx] = hi

            # Interpolation weight for upper bracket
            a_weight[j_idx, a_idx] = (lo == hi) ? 0.0 : (a_prime - A[lo]) / (A[hi] - A[lo])
        end
    end

    return a_lower, a_upper, a_weight
end


"""
Estimate initial addiction stock for each household via fixed-point iteration.

Starting from a₀ = 0, simulates the addiction trajectory forward using observed
choices, then sets a₀ to the terminal addiction level. Repeats until convergence.

Convergence is guaranteed because the law of motion ã' = (1-ψ)ã + ψn is a
contraction in a₀: the influence of a₀ on a_t decays b/c (1-ψ)^t.
Look at paper appendix for further details

Returns:
- a₀: Dict mapping household_code → estimated initial addiction stock
"""
function get_initial_addiction_stock(
    ψ::Real,
    A::AbstractVector{<:Real},
    n::AbstractVector{<:Real},
    y::AbstractVector{<:Integer},
    hh_codes::AbstractVector{<:Integer};
    max_iter::Integer = 500,
    tol::Real = 1e-6,
)

    # Number of observations
    N = length(y)

    # Create a dictionary of vectors
    # Keys are household codes
    # Values are observation indices corresponding to that household
    hh_obs = Dict{eltype(hh_codes), Vector{Int}}()
    for i in 1:N

        # Get household code
        hh = hh_codes[i]

        # If household code has not appeared as dictionary key yet (b/c hh codes show up more than once)
        if !haskey(hh_obs, hh)

            # Create empty vector as the value associated with this new key
            hh_obs[hh] = Int[]
        end

        # Add the current observation index to current household's vector of observation indices
        # The ! modifies the dictionary in place
        push!(hh_obs[hh], i)
    end

    # Initialize a₀ = 0 for all households
    a₀ = Dict{eltype(hh_codes), Float64}(
        hh => 0.0 for hh in keys(hh_obs)
    )

    # Track iterations until convergence for each household
    hh_idx = 1
    iters_to_convergence = zeros(Int, length(hh_obs))
    n_not_converged = 0

    # Loop over households
    for (hh, obs_indices) in hh_obs

        # Iterate until convergence for the current household
        for iter in 1:max_iter

            # Simulate forward from current a₀
            a = a₀[hh]
            for i in obs_indices
                a = addiction_evolution(ψ, a, n[y[i]])
            end

            # Update a₀ to terminal addiction level
            change = abs(a - a₀[hh])
            a₀[hh] = a

            # Check convergence
            if change < tol
                iters_to_convergence[hh_idx] = iter
                break
            end

            # Track non-convergence
            if iter == max_iter
                iters_to_convergence[hh_idx] = max_iter
                n_not_converged += 1
            end
        end

        # Update household index
        hh_idx += 1
    end

    # Print and log single summary if any households did not converge
    if n_not_converged > 0
        log_msg("WARNING: Initial addiction fixed-point did not converge for $n_not_converged / $(length(hh_obs)) households (max_iter=$max_iter, tol=$tol)")
    end

    return a₀, maximum(iters_to_convergence)
end


"""
Simulate household addiction trajectories and map to nearest grid indices.
Used by the likelihood to get each household's continuous addiction state
at each observation, which the log-likelihood then uses to interpolate
V_choice at the household's actual state.

For each household, starts with a₀ from the pre-estimated initial addiction
stock and evolves addiction forward using the observed choices:
  ã_{t+1} = (1 - ψ) * ã_t + ψ * n[y_t]

where y_t is the chosen alternative at time t and n[j] is its nicotine content.

Returns:
- a_continuous: Vector of actual continuous addiction levels for likelihood interpolation
"""
function simulate_addiction_trajectories(
    N_A::Integer,
    ψ::Real,
    A::AbstractVector{<:Real},
    n::AbstractVector{<:Real},
    y::AbstractVector{<:Integer},
    hh_codes::AbstractVector{<:Integer},
    a₀::AbstractDict
)

    # Number of observations
    N = length(y)

    # Initialize continuous addiction level for each observation
    a_continuous = Vector{Float64}(undef, N)

    # Track current addiction level per household
    a_current = Dict{eltype(hh_codes), Float64}()

    # Loop over observations
    for i in 1:N

        # Get current household code
        hh = hh_codes[i]

        # For each observation, this checks if the household already has an addiction level
        # from a previous period in the dictionary a_current. If yes, it uses
        # that. If no, meaning this is the household's first observation,
        # it uses the pre-estimated initial stock from a₀[hh].
        a = get(a_current, hh, a₀[hh])

        # Store continuous addiction level (for interpolation in likelihood)
        a_continuous[i] = a

        # Evolve addiction based on chosen alternative's nicotine
        a_prime = addiction_evolution(ψ, a, n[y[i]])

        # Clamp to grid bounds
        a_prime = clamp(a_prime, A[1], A[end])

        # Store for next period
        a_current[hh] = a_prime
    end

    return a_continuous
end


#############################
# 8. Value Function Iteration
#############################

"""
Log-sum-exp for numerical stability.

Computes log(∑_j exp(v[j])) without overflow. Directly computing exp(v[j])
can overflow for large v[j] (e.g., exp(800) = Inf in Float64). The trick is
to factor out the maximum m = max_j v[j]:

  log(∑_j exp(v[j])) = m + log(∑_j exp(v[j] - m))

Since v[j] - m ≤ 0 for all j, exp(v[j] - m) ∈ (0, 1], so the sum never
overflows. The largest term contributes exp(0) = 1, ensuring log(∑) ≥ 0.

Used in VFI to aggregate choice-specific value functions into the
ex-ante value function: V[tya, a, p] = logsumexp(V_choice[tya, :, a, p]),
which is the closed-form expected maximum from the Type I extreme value
(logit) error assumption.

Returns:
- log(∑_j exp(v[j]))
"""
function logsumexp(
    v::AbstractVector{<:Real}
)

    # Factor out maximum to prevent overflow
    m = maximum(v)

    # Guard against all-(-Inf) input: exp(-Inf - (-Inf)) = exp(NaN) = NaN
    if m == -Inf
        return -Inf
    end

    # Sum exp(v[j] - m)
    s = 0.0
    for x in v
        s += exp(x - m)
    end

    # Add back the maximum: m + log(∑ exp(v - m)) = log(∑ exp(v))
    return m + log(s)
end


"""
Recompute choice-specific values from a converged state value function.

During VFI, the last iteration's V_choice values are computed using V_now from the
PREVIOUS iteration. After convergence, V_now has been updated to
the final converged values, but V_choice was built from the second-to-last
V_now. This function recomputes V_choice one final time using the converged V_now so
that the returned choice-specific values are exactly consistent with the converged
state value function.

"""
function recompute_choice_values!(
    V_out::Array{Float64, 6},
    V_now::Array{Float64, 5},
    U::Array{Float64, 6},
    discount_scale::Float64,
    N_J::Integer,
    N_A_f::Integer,
    N_A_s::Integer,
    N_A_flav::Integer,
    N_P::Integer,
    N_Pcomb::Integer,
    inv_R::Float64,
    p_cig_lo::Matrix{Int},
    p_cig_hi::Matrix{Int},
    p_cig_w::Matrix{Float64},
    p_ecig_lo::Matrix{Int},
    p_ecig_hi::Matrix{Int},
    p_ecig_w::Matrix{Float64},
    af_lower::Matrix{Int},
    af_upper::Matrix{Int},
    af_weight::Matrix{Float64},
    as_lower::Matrix{Int},
    as_upper::Matrix{Int},
    as_weight::Matrix{Float64},
    aflav_lower::Matrix{Int},
    aflav_upper::Matrix{Int},
    aflav_weight::Matrix{Float64}
)

    # Number of Halton draws and TYA states (binary: 1 = no TYA, 2 = TYA present)
    R = size(p_cig_lo, 2)
    N_TYA = 2

    # Parallelize over (price state, slow addiction state, flavored habit state) triples.
    Threads.@threads for flat_idx in 1:(N_Pcomb * N_A_s * N_A_flav)

        # Recover 3D indices from flat index
        p_idx     = div(flat_idx - 1, N_A_s * N_A_flav) + 1
        remainder = mod(flat_idx - 1, N_A_s * N_A_flav)
        as_idx    = div(remainder, N_A_flav) + 1
        aflav_idx = mod(remainder, N_A_flav) + 1

        # Loop over fast addiction states
        @inbounds for af_idx in 1:N_A_f

            # Loop over alternatives
            @inbounds for j_idx in 1:N_J

                # Pre-computed fast addiction transition brackets for (j_idx, af_idx)
                lo_af = af_lower[j_idx, af_idx]
                hi_af = af_upper[j_idx, af_idx]
                w_af  = af_weight[j_idx, af_idx]

                # Pre-computed slow addiction transition brackets for (j_idx, as_idx)
                lo_as = as_lower[j_idx, as_idx]
                hi_as = as_upper[j_idx, as_idx]
                w_as  = as_weight[j_idx, as_idx]

                # Pre-computed flavored habit transition brackets for (j_idx, aflav_idx)
                lo_aflav = aflav_lower[j_idx, aflav_idx]
                hi_aflav = aflav_upper[j_idx, aflav_idx]
                w_aflav  = aflav_weight[j_idx, aflav_idx]

                # Initialize 16 accumulators: 2 TYA states × 8 addiction corners
                # Naming: EV_XYZ_T where X=af(f/s), Y=as(f/s), Z=aflav(f/s), T=tya(1-2)
                EV_fff_1 = 0.0; EV_ffs_1 = 0.0; EV_fsf_1 = 0.0; EV_fss_1 = 0.0; EV_sff_1 = 0.0; EV_sfs_1 = 0.0; EV_ssf_1 = 0.0; EV_sss_1 = 0.0
                EV_fff_2 = 0.0; EV_ffs_2 = 0.0; EV_fsf_2 = 0.0; EV_fss_2 = 0.0; EV_sff_2 = 0.0; EV_sfs_2 = 0.0; EV_ssf_2 = 0.0; EV_sss_2 = 0.0

                # Loop over Halton draws to approximate expected continuation values
                @inbounds for r in 1:R

                    # Pre-computed price transition brackets for draw r from price state p_idx
                    c_lo = p_cig_lo[p_idx, r]
                    c_hi = p_cig_hi[p_idx, r]
                    w_c  = p_cig_w[p_idx, r]
                    e_lo = p_ecig_lo[p_idx, r]
                    e_hi = p_ecig_hi[p_idx, r]
                    w_e  = p_ecig_w[p_idx, r]

                    # Convert 2D price grid indices to 1D combined index
                    p_ll = (c_lo - 1) * N_P + e_lo
                    p_lh = (c_lo - 1) * N_P + e_hi
                    p_hl = (c_hi - 1) * N_P + e_lo
                    p_hh = (c_hi - 1) * N_P + e_hi

                    # Bilinear interpolation weights for the 4 corners of the price grid
                    w_ll = (1 - w_c) * (1 - w_e)
                    w_lh = (1 - w_c) * w_e
                    w_hl = w_c * (1 - w_e)
                    w_hh = w_c * w_e

                    # Accumulate bilinearly-interpolated V_now at each of the 2 TYA states
                    # and 8 addiction corners (af × as × aflav), for a total of 16 accumulators.
                    # No TYA transitions; each TYA state's continuation depends only on itself.

                    # TYA state 1 (no TYA)
                    EV_fff_1 += w_ll * V_now[1, lo_af, lo_as, lo_aflav, p_ll] + w_lh * V_now[1, lo_af, lo_as, lo_aflav, p_lh] + w_hl * V_now[1, lo_af, lo_as, lo_aflav, p_hl] + w_hh * V_now[1, lo_af, lo_as, lo_aflav, p_hh]
                    EV_ffs_1 += w_ll * V_now[1, lo_af, lo_as, hi_aflav, p_ll] + w_lh * V_now[1, lo_af, lo_as, hi_aflav, p_lh] + w_hl * V_now[1, lo_af, lo_as, hi_aflav, p_hl] + w_hh * V_now[1, lo_af, lo_as, hi_aflav, p_hh]
                    EV_fsf_1 += w_ll * V_now[1, lo_af, hi_as, lo_aflav, p_ll] + w_lh * V_now[1, lo_af, hi_as, lo_aflav, p_lh] + w_hl * V_now[1, lo_af, hi_as, lo_aflav, p_hl] + w_hh * V_now[1, lo_af, hi_as, lo_aflav, p_hh]
                    EV_fss_1 += w_ll * V_now[1, lo_af, hi_as, hi_aflav, p_ll] + w_lh * V_now[1, lo_af, hi_as, hi_aflav, p_lh] + w_hl * V_now[1, lo_af, hi_as, hi_aflav, p_hl] + w_hh * V_now[1, lo_af, hi_as, hi_aflav, p_hh]
                    EV_sff_1 += w_ll * V_now[1, hi_af, lo_as, lo_aflav, p_ll] + w_lh * V_now[1, hi_af, lo_as, lo_aflav, p_lh] + w_hl * V_now[1, hi_af, lo_as, lo_aflav, p_hl] + w_hh * V_now[1, hi_af, lo_as, lo_aflav, p_hh]
                    EV_sfs_1 += w_ll * V_now[1, hi_af, lo_as, hi_aflav, p_ll] + w_lh * V_now[1, hi_af, lo_as, hi_aflav, p_lh] + w_hl * V_now[1, hi_af, lo_as, hi_aflav, p_hl] + w_hh * V_now[1, hi_af, lo_as, hi_aflav, p_hh]
                    EV_ssf_1 += w_ll * V_now[1, hi_af, hi_as, lo_aflav, p_ll] + w_lh * V_now[1, hi_af, hi_as, lo_aflav, p_lh] + w_hl * V_now[1, hi_af, hi_as, lo_aflav, p_hl] + w_hh * V_now[1, hi_af, hi_as, lo_aflav, p_hh]
                    EV_sss_1 += w_ll * V_now[1, hi_af, hi_as, hi_aflav, p_ll] + w_lh * V_now[1, hi_af, hi_as, hi_aflav, p_lh] + w_hl * V_now[1, hi_af, hi_as, hi_aflav, p_hl] + w_hh * V_now[1, hi_af, hi_as, hi_aflav, p_hh]

                    # TYA state 2 (TYA present)
                    EV_fff_2 += w_ll * V_now[2, lo_af, lo_as, lo_aflav, p_ll] + w_lh * V_now[2, lo_af, lo_as, lo_aflav, p_lh] + w_hl * V_now[2, lo_af, lo_as, lo_aflav, p_hl] + w_hh * V_now[2, lo_af, lo_as, lo_aflav, p_hh]
                    EV_ffs_2 += w_ll * V_now[2, lo_af, lo_as, hi_aflav, p_ll] + w_lh * V_now[2, lo_af, lo_as, hi_aflav, p_lh] + w_hl * V_now[2, lo_af, lo_as, hi_aflav, p_hl] + w_hh * V_now[2, lo_af, lo_as, hi_aflav, p_hh]
                    EV_fsf_2 += w_ll * V_now[2, lo_af, hi_as, lo_aflav, p_ll] + w_lh * V_now[2, lo_af, hi_as, lo_aflav, p_lh] + w_hl * V_now[2, lo_af, hi_as, lo_aflav, p_hl] + w_hh * V_now[2, lo_af, hi_as, lo_aflav, p_hh]
                    EV_fss_2 += w_ll * V_now[2, lo_af, hi_as, hi_aflav, p_ll] + w_lh * V_now[2, lo_af, hi_as, hi_aflav, p_lh] + w_hl * V_now[2, lo_af, hi_as, hi_aflav, p_hl] + w_hh * V_now[2, lo_af, hi_as, hi_aflav, p_hh]
                    EV_sff_2 += w_ll * V_now[2, hi_af, lo_as, lo_aflav, p_ll] + w_lh * V_now[2, hi_af, lo_as, lo_aflav, p_lh] + w_hl * V_now[2, hi_af, lo_as, lo_aflav, p_hl] + w_hh * V_now[2, hi_af, lo_as, lo_aflav, p_hh]
                    EV_sfs_2 += w_ll * V_now[2, hi_af, lo_as, hi_aflav, p_ll] + w_lh * V_now[2, hi_af, lo_as, hi_aflav, p_lh] + w_hl * V_now[2, hi_af, lo_as, hi_aflav, p_hl] + w_hh * V_now[2, hi_af, lo_as, hi_aflav, p_hh]
                    EV_ssf_2 += w_ll * V_now[2, hi_af, hi_as, lo_aflav, p_ll] + w_lh * V_now[2, hi_af, hi_as, lo_aflav, p_lh] + w_hl * V_now[2, hi_af, hi_as, lo_aflav, p_hl] + w_hh * V_now[2, hi_af, hi_as, lo_aflav, p_hh]
                    EV_sss_2 += w_ll * V_now[2, hi_af, hi_as, hi_aflav, p_ll] + w_lh * V_now[2, hi_af, hi_as, hi_aflav, p_lh] + w_hl * V_now[2, hi_af, hi_as, hi_aflav, p_hl] + w_hh * V_now[2, hi_af, hi_as, hi_aflav, p_hh]
                end

                # Average over Halton draws (multiply by 1/R) and trilinearly interpolate
                # across the 8 addiction bracket corners using weights w_af, w_as, and w_aflav.
                # Result: E[V(tya, af', as', aflav', p') | af, as, aflav, p, j] for each TYA state
                # No TYA transition integration; each TYA state uses its own continuation value.
                EV_1 = (1-w_af)*(1-w_as)*(1-w_aflav)*(EV_fff_1*inv_R) + (1-w_af)*(1-w_as)*w_aflav*(EV_ffs_1*inv_R) + (1-w_af)*w_as*(1-w_aflav)*(EV_fsf_1*inv_R) + (1-w_af)*w_as*w_aflav*(EV_fss_1*inv_R) + w_af*(1-w_as)*(1-w_aflav)*(EV_sff_1*inv_R) + w_af*(1-w_as)*w_aflav*(EV_sfs_1*inv_R) + w_af*w_as*(1-w_aflav)*(EV_ssf_1*inv_R) + w_af*w_as*w_aflav*(EV_sss_1*inv_R)
                EV_2 = (1-w_af)*(1-w_as)*(1-w_aflav)*(EV_fff_2*inv_R) + (1-w_af)*(1-w_as)*w_aflav*(EV_ffs_2*inv_R) + (1-w_af)*w_as*(1-w_aflav)*(EV_fsf_2*inv_R) + (1-w_af)*w_as*w_aflav*(EV_fss_2*inv_R) + w_af*(1-w_as)*(1-w_aflav)*(EV_sff_2*inv_R) + w_af*(1-w_as)*w_aflav*(EV_sfs_2*inv_R) + w_af*w_as*(1-w_aflav)*(EV_ssf_2*inv_R) + w_af*w_as*w_aflav*(EV_sss_2*inv_R)

                # Store: V_out = U + discount_scale * EV (no TYA transition integration)
                V_out[1, j_idx, af_idx, as_idx, aflav_idx, p_idx] = U[1, j_idx, af_idx, as_idx, aflav_idx, p_idx] + discount_scale * EV_1
                V_out[2, j_idx, af_idx, as_idx, aflav_idx, p_idx] = U[2, j_idx, af_idx, as_idx, aflav_idx, p_idx] + discount_scale * EV_2
            end
        end
    end
end


"""
Solve the value function via value function iteration (VFI) for the sophisticated
quasi-hyperbolic discounting case.

# Model
A sophisticated agent has present bias β ∈ (0, 1] and knows their future selves
will also be present-biased. This creates a distinction between:
  - Decision utility V_d: what the agent *uses* to choose (discounts future at β·δ)
  - Experienced utility V_e: what the agent *actually receives* (discounts future at δ)

The agent today knows that tomorrow's self will choose according to V_d (not V_e),
so the continuation value must reflect the *experienced* payoffs from *decision*-
driven choices.

# Bellman Equations
Each iteration computes two choice-specific value functions:

  V_d[tya, j, a, p] = U[tya, j, a, p] + β·δ · EV[tya, a'(j,a), p]
  V_e[tya, j, a, p] = U[tya, j, a, p] +   δ · EV[tya, a'(j,a), p]

where:
  - U[tya, j, a, p] is the deterministic flow utility of choosing alternative j
    in state (tya, a, p), precomputed outside VFI
  - a'(j, a) is the next-period addiction state from choosing j at addiction a,
    given by ã' = (1-ψ)·ã + ψ·n[j], interpolated onto the addiction grid
  - EV is the expected continuation value, integrating over stochastic price
    transitions using R Halton draws 

V_d and V_e share the *same* EV term. The only difference is the discount
factor applied to it (β·δ vs δ). The continuation value V is what *will actually
happen*, and the sophisticated agent correctly predicts this.

# Continuation Value (Sophisticated Aggregation)
The ex-ante state value V aggregates over alternatives using decision-utility
choice probabilities applied to experienced-utility payoffs:

  V[tya, a, p] = Σ_j p_j · V_e[tya, j, a, p] + H(p)

where:
  - p_j = softmax(V_d)_j = exp(V_d_j) / Σ_k exp(V_d_k) are the choice
    probabilities from the T1EV logit assumption, computed from decision utility
  - H(p) = -Σ_j p_j · log(p_j) is the entropy bonus from the T1EV error
    distribution (the option value of randomness)

This says: the future self chooses with probs p_j (based on present-biased V_d),
but the actual payoff from each choice is V_e (not V_d). The entropy term accounts
for the fact that the T1EV shocks create randomness that has positive expected value.

# Why V = Σ p_j V_e_j + H(p) and not logsumexp(V_e)?
Under standard exponential discounting (β = 1), the agent chooses to maximize
V_e, so the expected max is logsumexp(V_e) by the T1EV formula. But when β < 1,
the agent chooses to maximize V_d ≠ V_e. The choice probabilities come from V_d,
not V_e, so the standard logsumexp formula does not apply. Instead we must
manually compute the expected payoff: probability-weighted V_e plus entropy.

# When β = 1 (Standard Exponential Discounting)
V_d = V_e (since β·δ = δ), so p_j = softmax(V_e)_j. In this case:
  Σ_j p_j · V_e_j + H(p) = logsumexp(V_e)
This is an identity of the T1EV distribution. So the sophisticated aggregator
collapses exactly to the naive logsumexp, and all outputs are numerically
identical to naive VFI.

# Expected Continuation Value Computation (EV)
For a given (j, a, p) triple, the expected continuation value integrates over
stochastic price transitions:

  EV[tya, a', p] = (1/R) · Σ_{r=1}^{R} V[tya, a', p'_r]

where p'_r is the realized next-period price vector from Halton draw r, starting
from current price state p. TYA is a binary observable indicator (no transitions),
so the continuation value for TYA=t only depends on V[t, ...] (same TYA state).
Since a' and p' are generally off-grid, we interpolate:
  - Addiction dimension: trilinear interpolation between grid brackets (af, as, aflav)
  - Price dimensions: bilinear interpolation over the 2D (cig × ecig) price grid
    using four corner points (p_ll, p_lh, p_hl, p_hh)

The interpolation brackets and weights are precomputed outside VFI:
  - (a_lower, a_upper, a_weight): from precompute_addiction_transitions()
  - (p_cig_lo/hi/w, p_ecig_lo/hi/w): from precompute_price_transitions()

# Iteration Scheme
Uses Jacobi-style iteration: the entire V_next array is computed from V_now,
then V_now is replaced with V_next. 

# Post-Convergence Recomputation
After convergence, V_d is recomputed one final time from the converged V_now.
This ensures the returned V_decision is exactly consistent with the fixed point,
rather than being one iteration stale (since V_d was last computed *before*
the final V_now update).

Returns
    - V:          Converged ex-ante value function, V[tya_idx, af_idx, as_idx, aflav_idx, p_idx]
    - V_decision: Decision-utility choice values (βδ discounting), V_decision[tya_idx, j, af_idx, as_idx, aflav_idx, p_idx]
    - n_iter:     Number of iterations to convergence
    - converged:  Whether VFI converged within max_iter
"""
function solve_vfi_sophisticated(
    N_J::Integer,
    N_A_f::Integer,
    N_A_s::Integer,
    N_A_flav::Integer,
    N_P::Integer,
    N_Pcomb::Integer,
    β::Real,
    δ::Real,
    U::Array{Float64, 6},
    af_lower::Matrix{Int},
    af_upper::Matrix{Int},
    af_weight::Matrix{Float64},
    as_lower::Matrix{Int},
    as_upper::Matrix{Int},
    as_weight::Matrix{Float64},
    aflav_lower::Matrix{Int},
    aflav_upper::Matrix{Int},
    aflav_weight::Matrix{Float64},
    p_cig_lo::Matrix{Int},
    p_cig_hi::Matrix{Int},
    p_cig_w::Matrix{Float64},
    p_ecig_lo::Matrix{Int},
    p_ecig_hi::Matrix{Int},
    p_ecig_w::Matrix{Float64};
    V_init::Union{Array{Float64, 5}, Nothing} = nothing,
    ε::Real = 1e-4,
    max_iter::Integer = 3000,
    verbose::Bool = true
)

    # Number of TYA states (binary: 1 = no TYA, 2 = TYA present; no transitions) and Halton draws
    N_TYA = 2
    R = size(p_cig_lo, 2)

    # Compute 1.0/R
    inv_R = 1.0 / R

    # Pre-compute β·δ, the discount factor applied in decision utility.
    βδ = β * δ

    #############################
    # VFI
    #############################

    # Initialize V_now from V_init (warm-start) or zeros (cold start).
    # V_now:  current iterate of the ex-ante value function V[tya, af, as, aflav, p]
    # V_next: next iterate of V
    # V_d:    decision utility V_d[tya, j, af, as, aflav, p] = U + βδ·EV (populated only after convergence)
    V_now  = zeros(Float64, N_TYA, N_A_f, N_A_s, N_A_flav, N_Pcomb)
    if V_init !== nothing
        copyto!(V_now, V_init)
    end
    V_next = zeros(Float64, N_TYA, N_A_f, N_A_s, N_A_flav, N_Pcomb)
    V_d    = zeros(Float64, N_TYA, N_J, N_A_f, N_A_s, N_A_flav, N_Pcomb)
    V_e    = zeros(Float64, N_TYA, N_J, N_A_f, N_A_s, N_A_flav, N_Pcomb)

    # Initialize number of iterations
    n_iter = 0

    # Initialize convergence
    converged = false

    # Previous iterations sup-norm across all states
    prev_diff = Inf

    # Time VFI
    t_vfi = time()
    for iter in 1:max_iter

        # Update iteration counter
        n_iter = iter

        # Parallelize over (price state, slow addiction state, flavored habit state) triples.
        Threads.@threads for flat_idx in 1:(N_Pcomb * N_A_s * N_A_flav)

            # Recover 3D indices from flat index
            p_idx     = div(flat_idx - 1, N_A_s * N_A_flav) + 1
            remainder = mod(flat_idx - 1, N_A_s * N_A_flav)
            as_idx    = div(remainder, N_A_flav) + 1
            aflav_idx = mod(remainder, N_A_flav) + 1

            # Loop over fast addiction states
            @inbounds for af_idx in 1:N_A_f

                # Loop over alternatives
                @inbounds for j_idx in 1:N_J

                    # Pre-computed fast addiction transition brackets for (j_idx, af_idx)
                    lo_af = af_lower[j_idx, af_idx]
                    hi_af = af_upper[j_idx, af_idx]
                    w_af  = af_weight[j_idx, af_idx]

                    # Pre-computed slow addiction transition brackets for (j_idx, as_idx)
                    lo_as = as_lower[j_idx, as_idx]
                    hi_as = as_upper[j_idx, as_idx]
                    w_as  = as_weight[j_idx, as_idx]

                    # Pre-computed flavored habit transition brackets for (j_idx, aflav_idx)
                    lo_aflav = aflav_lower[j_idx, aflav_idx]
                    hi_aflav = aflav_upper[j_idx, aflav_idx]
                    w_aflav  = aflav_weight[j_idx, aflav_idx]

                    # Initialize 16 accumulators: 2 TYA states × 8 addiction corners
                    # Naming: EV_XYZ_T where X=af(f/s), Y=as(f/s), Z=aflav, T=tya(1-2)
                    EV_fff_1 = 0.0; EV_ffs_1 = 0.0; EV_fsf_1 = 0.0; EV_fss_1 = 0.0; EV_sff_1 = 0.0; EV_sfs_1 = 0.0; EV_ssf_1 = 0.0; EV_sss_1 = 0.0
                    EV_fff_2 = 0.0; EV_ffs_2 = 0.0; EV_fsf_2 = 0.0; EV_fss_2 = 0.0; EV_sff_2 = 0.0; EV_sfs_2 = 0.0; EV_ssf_2 = 0.0; EV_sss_2 = 0.0

                    # Loop over Halton draws to approximate expected continuation values
                    @inbounds for r in 1:R

                        # Pre-computed price transition brackets for draw r from price state p_idx
                        c_lo = p_cig_lo[p_idx, r]
                        c_hi = p_cig_hi[p_idx, r]
                        w_c  = p_cig_w[p_idx, r]
                        e_lo = p_ecig_lo[p_idx, r]
                        e_hi = p_ecig_hi[p_idx, r]
                        w_e  = p_ecig_w[p_idx, r]

                        # Convert 2D price grid indices to 1D combined index
                        p_ll = (c_lo - 1) * N_P + e_lo
                        p_lh = (c_lo - 1) * N_P + e_hi
                        p_hl = (c_hi - 1) * N_P + e_lo
                        p_hh = (c_hi - 1) * N_P + e_hi

                        # Bilinear interpolation weights for the 4 corners of the price grid
                        w_ll = (1 - w_c) * (1 - w_e)
                        w_lh = (1 - w_c) * w_e
                        w_hl = w_c * (1 - w_e)
                        w_hh = w_c * w_e

                        # TYA state 1 (no TYA)
                        EV_fff_1 += w_ll * V_now[1, lo_af, lo_as, lo_aflav, p_ll] + w_lh * V_now[1, lo_af, lo_as, lo_aflav, p_lh] + w_hl * V_now[1, lo_af, lo_as, lo_aflav, p_hl] + w_hh * V_now[1, lo_af, lo_as, lo_aflav, p_hh]
                        EV_ffs_1 += w_ll * V_now[1, lo_af, lo_as, hi_aflav, p_ll] + w_lh * V_now[1, lo_af, lo_as, hi_aflav, p_lh] + w_hl * V_now[1, lo_af, lo_as, hi_aflav, p_hl] + w_hh * V_now[1, lo_af, lo_as, hi_aflav, p_hh]
                        EV_fsf_1 += w_ll * V_now[1, lo_af, hi_as, lo_aflav, p_ll] + w_lh * V_now[1, lo_af, hi_as, lo_aflav, p_lh] + w_hl * V_now[1, lo_af, hi_as, lo_aflav, p_hl] + w_hh * V_now[1, lo_af, hi_as, lo_aflav, p_hh]
                        EV_fss_1 += w_ll * V_now[1, lo_af, hi_as, hi_aflav, p_ll] + w_lh * V_now[1, lo_af, hi_as, hi_aflav, p_lh] + w_hl * V_now[1, lo_af, hi_as, hi_aflav, p_hl] + w_hh * V_now[1, lo_af, hi_as, hi_aflav, p_hh]
                        EV_sff_1 += w_ll * V_now[1, hi_af, lo_as, lo_aflav, p_ll] + w_lh * V_now[1, hi_af, lo_as, lo_aflav, p_lh] + w_hl * V_now[1, hi_af, lo_as, lo_aflav, p_hl] + w_hh * V_now[1, hi_af, lo_as, lo_aflav, p_hh]
                        EV_sfs_1 += w_ll * V_now[1, hi_af, lo_as, hi_aflav, p_ll] + w_lh * V_now[1, hi_af, lo_as, hi_aflav, p_lh] + w_hl * V_now[1, hi_af, lo_as, hi_aflav, p_hl] + w_hh * V_now[1, hi_af, lo_as, hi_aflav, p_hh]
                        EV_ssf_1 += w_ll * V_now[1, hi_af, hi_as, lo_aflav, p_ll] + w_lh * V_now[1, hi_af, hi_as, lo_aflav, p_lh] + w_hl * V_now[1, hi_af, hi_as, lo_aflav, p_hl] + w_hh * V_now[1, hi_af, hi_as, lo_aflav, p_hh]
                        EV_sss_1 += w_ll * V_now[1, hi_af, hi_as, hi_aflav, p_ll] + w_lh * V_now[1, hi_af, hi_as, hi_aflav, p_lh] + w_hl * V_now[1, hi_af, hi_as, hi_aflav, p_hl] + w_hh * V_now[1, hi_af, hi_as, hi_aflav, p_hh]

                        # TYA state 2 (TYA present)
                        EV_fff_2 += w_ll * V_now[2, lo_af, lo_as, lo_aflav, p_ll] + w_lh * V_now[2, lo_af, lo_as, lo_aflav, p_lh] + w_hl * V_now[2, lo_af, lo_as, lo_aflav, p_hl] + w_hh * V_now[2, lo_af, lo_as, lo_aflav, p_hh]
                        EV_ffs_2 += w_ll * V_now[2, lo_af, lo_as, hi_aflav, p_ll] + w_lh * V_now[2, lo_af, lo_as, hi_aflav, p_lh] + w_hl * V_now[2, lo_af, lo_as, hi_aflav, p_hl] + w_hh * V_now[2, lo_af, lo_as, hi_aflav, p_hh]
                        EV_fsf_2 += w_ll * V_now[2, lo_af, hi_as, lo_aflav, p_ll] + w_lh * V_now[2, lo_af, hi_as, lo_aflav, p_lh] + w_hl * V_now[2, lo_af, hi_as, lo_aflav, p_hl] + w_hh * V_now[2, lo_af, hi_as, lo_aflav, p_hh]
                        EV_fss_2 += w_ll * V_now[2, lo_af, hi_as, hi_aflav, p_ll] + w_lh * V_now[2, lo_af, hi_as, hi_aflav, p_lh] + w_hl * V_now[2, lo_af, hi_as, hi_aflav, p_hl] + w_hh * V_now[2, lo_af, hi_as, hi_aflav, p_hh]
                        EV_sff_2 += w_ll * V_now[2, hi_af, lo_as, lo_aflav, p_ll] + w_lh * V_now[2, hi_af, lo_as, lo_aflav, p_lh] + w_hl * V_now[2, hi_af, lo_as, lo_aflav, p_hl] + w_hh * V_now[2, hi_af, lo_as, lo_aflav, p_hh]
                        EV_sfs_2 += w_ll * V_now[2, hi_af, lo_as, hi_aflav, p_ll] + w_lh * V_now[2, hi_af, lo_as, hi_aflav, p_lh] + w_hl * V_now[2, hi_af, lo_as, hi_aflav, p_hl] + w_hh * V_now[2, hi_af, lo_as, hi_aflav, p_hh]
                        EV_ssf_2 += w_ll * V_now[2, hi_af, hi_as, lo_aflav, p_ll] + w_lh * V_now[2, hi_af, hi_as, lo_aflav, p_lh] + w_hl * V_now[2, hi_af, hi_as, lo_aflav, p_hl] + w_hh * V_now[2, hi_af, hi_as, lo_aflav, p_hh]
                        EV_sss_2 += w_ll * V_now[2, hi_af, hi_as, hi_aflav, p_ll] + w_lh * V_now[2, hi_af, hi_as, hi_aflav, p_lh] + w_hl * V_now[2, hi_af, hi_as, hi_aflav, p_hl] + w_hh * V_now[2, hi_af, hi_as, hi_aflav, p_hh]
                    end

                    # Average over Halton draws and trilinearly interpolate across 8 addiction corners
                    EV_1 = (1-w_af)*(1-w_as)*(1-w_aflav)*(EV_fff_1*inv_R) + (1-w_af)*(1-w_as)*w_aflav*(EV_ffs_1*inv_R) + (1-w_af)*w_as*(1-w_aflav)*(EV_fsf_1*inv_R) + (1-w_af)*w_as*w_aflav*(EV_fss_1*inv_R) + w_af*(1-w_as)*(1-w_aflav)*(EV_sff_1*inv_R) + w_af*(1-w_as)*w_aflav*(EV_sfs_1*inv_R) + w_af*w_as*(1-w_aflav)*(EV_ssf_1*inv_R) + w_af*w_as*w_aflav*(EV_sss_1*inv_R)
                    EV_2 = (1-w_af)*(1-w_as)*(1-w_aflav)*(EV_fff_2*inv_R) + (1-w_af)*(1-w_as)*w_aflav*(EV_ffs_2*inv_R) + (1-w_af)*w_as*(1-w_aflav)*(EV_fsf_2*inv_R) + (1-w_af)*w_as*w_aflav*(EV_fss_2*inv_R) + w_af*(1-w_as)*(1-w_aflav)*(EV_sff_2*inv_R) + w_af*(1-w_as)*w_aflav*(EV_sfs_2*inv_R) + w_af*w_as*(1-w_aflav)*(EV_ssf_2*inv_R) + w_af*w_as*w_aflav*(EV_sss_2*inv_R)

                    # Store decision and experienced choice-specific value functions
                    V_d[1, j_idx, af_idx, as_idx, aflav_idx, p_idx] = U[1, j_idx, af_idx, as_idx, aflav_idx, p_idx] + βδ * EV_1
                    V_d[2, j_idx, af_idx, as_idx, aflav_idx, p_idx] = U[2, j_idx, af_idx, as_idx, aflav_idx, p_idx] + βδ * EV_2
                    V_e[1, j_idx, af_idx, as_idx, aflav_idx, p_idx] = U[1, j_idx, af_idx, as_idx, aflav_idx, p_idx] + δ  * EV_1
                    V_e[2, j_idx, af_idx, as_idx, aflav_idx, p_idx] = U[2, j_idx, af_idx, as_idx, aflav_idx, p_idx] + δ  * EV_2
                end

                # Sophisticated aggregation: V_next = Σ_j p_j * V_e_j + H(p) where p = softmax(V_d)
                @inbounds for tya_idx in 1:N_TYA
                    vd_max = V_d[tya_idx, 1, af_idx, as_idx, aflav_idx, p_idx]
                    @inbounds for j in 2:N_J
                        if V_d[tya_idx, j, af_idx, as_idx, aflav_idx, p_idx] > vd_max
                            vd_max = V_d[tya_idx, j, af_idx, as_idx, aflav_idx, p_idx]
                        end
                    end

                    sum_exp = 0.0
                    @inbounds for j in 1:N_J
                        sum_exp += exp(V_d[tya_idx, j, af_idx, as_idx, aflav_idx, p_idx] - vd_max)
                    end
                    log_denom = log(sum_exp)

                    agg = 0.0
                    @inbounds for j in 1:N_J
                        log_pj = V_d[tya_idx, j, af_idx, as_idx, aflav_idx, p_idx] - vd_max - log_denom
                        if log_pj == -Inf; continue; end
                        pj = exp(log_pj)
                        agg += pj * V_e[tya_idx, j, af_idx, as_idx, aflav_idx, p_idx] - pj * log_pj
                    end
                    V_next[tya_idx, af_idx, as_idx, aflav_idx, p_idx] = agg
                end

            end
        end

        # Sup-norm: max absolute difference between V_next and V_now across all states
        current_diff = maximum(abs.(V_next .- V_now))

        # Jacobi update: replace V_now with V_next for the next iteration
        V_now .= V_next

        # Print progress at iterations 1, 10, and every 100th iteration
        if verbose && (iter % 100 == 0 || iter == 1 || iter == 10)
            elapsed = time() - t_vfi
            ratio = prev_diff < Inf ? current_diff / prev_diff : NaN
            log_msg(@sprintf("      VFI iter %6d | sup-norm = %.6e | ratio = %.4f | elapsed = %.1fs",
                iter, current_diff, ratio, elapsed))
        end

        # Update sup-norm
        prev_diff = current_diff

        # Check if sup-norm is below tolerance ε
        if current_diff < ε
            if verbose
                elapsed = time() - t_vfi
                log_msg("")
                log_msg(@sprintf("      VFI converged in %d iterations (sup-norm = %.6e, %.1fs):", iter, current_diff, elapsed))
            end
            converged = true
            break
        end

        # Report if we hit the maximum number of iterations without converging
        if iter == max_iter
            elapsed = time() - t_vfi
            log_msg("")
            log_msg(@sprintf("VFI did not converge after %d iterations (sup-norm = %.6e, %.1fs)", max_iter, current_diff, elapsed))
        end
    end

    # During the last VFI iteration, V_d was computed using V_now from the
    # *previous* iteration. But then V_now was updated to V_next. So V_d is
    # one iteration stale relative to the converged V_now. We recompute V_d
    # one final time using the converged V_now to ensure exact consistency.
    # This uses the same Bellman equation: V_d = U + βδ · EV(V_now_converged).
    recompute_choice_values!(
        V_d, V_now, U, βδ,
        N_J, N_A_f, N_A_s, N_A_flav, N_P, N_Pcomb, inv_R,
        p_cig_lo, p_cig_hi, p_cig_w,
        p_ecig_lo, p_ecig_hi, p_ecig_w,
        af_lower, af_upper, af_weight,
        as_lower, as_upper, as_weight,
        aflav_lower, aflav_upper, aflav_weight
    )

    # V_decision = V_d is what enters the log-likelihood.
    # Choice probabilities are P(j | state) = softmax(V_decision)_j.
    V_decision = V_d
    return V_now, V_decision, n_iter, converged
end


#############################
# 9. Likelihood & Prediction
#############################

"""
5-linearly interpolate V_choice at continuous states (af, as, aflav, p_cig, p_ecig)
for all alternatives j = 1, ..., N_J.

Uses the same bracket/weight logic as the log_likelihood function:
  1. Trilinear interpolation over the three addiction/habit grids (fast, slow, flavored)
  2. Bilinear interpolation over the 2D price grid (cig × ecig)

Returns:
- v_interp: Vector of interpolated V_choice values for all N_J alternatives
"""
function interpolate_v_choice(
    V_choice::Array{Float64, 6},
    tya_idx::Integer,
    af::Real,
    as::Real,
    aflav::Real,
    obs_cig::Real,
    obs_ecig::Real,
    N_J::Integer,
    N_P::Integer,
    A_f::AbstractVector{<:Real},
    A_s::AbstractVector{<:Real},
    A_flav::AbstractVector{<:Real},
    P::AbstractMatrix{<:Real}
)

    # Number of addiction grid points
    N_A_f = length(A_f)
    N_A_s = length(A_s)
    N_A_flav = length(A_flav)

    # 1D price grids
    P_cig  = @view P[:, 1]
    P_ecig = @view P[:, 2]

    # Clamp continuous fast addiction to grid bounds
    af_i = clamp(af, A_f[1], A_f[end])
    hi_af = clamp(searchsortedfirst(A_f, af_i), 1, N_A_f)
    lo_af = clamp(hi_af - 1, 1, N_A_f)
    w_af = (lo_af == hi_af) ? 0.0 : (af_i - A_f[lo_af]) / (A_f[hi_af] - A_f[lo_af])

    # Clamp continuous slow addiction to grid bounds
    as_i = clamp(as, A_s[1], A_s[end])
    hi_as = clamp(searchsortedfirst(A_s, as_i), 1, N_A_s)
    lo_as = clamp(hi_as - 1, 1, N_A_s)
    w_as = (lo_as == hi_as) ? 0.0 : (as_i - A_s[lo_as]) / (A_s[hi_as] - A_s[lo_as])

    # Clamp continuous flavored habit to grid bounds
    aflav_i = clamp(aflav, A_flav[1], A_flav[end])
    hi_aflav = clamp(searchsortedfirst(A_flav, aflav_i), 1, N_A_flav)
    lo_aflav = clamp(hi_aflav - 1, 1, N_A_flav)
    w_aflav = (lo_aflav == hi_aflav) ? 0.0 : (aflav_i - A_flav[lo_aflav]) / (A_flav[hi_aflav] - A_flav[lo_aflav])

    # Clamp continuous prices to grid bounds
    cig_clamped  = clamp(obs_cig, P_cig[1], P_cig[end])
    ecig_clamped = clamp(obs_ecig, P_ecig[1], P_ecig[end])

    # Cigarette price brackets
    hi_c = clamp(searchsortedfirst(P_cig, cig_clamped), 1, N_P)
    lo_c = clamp(hi_c - 1, 1, N_P)
    w_c  = (lo_c == hi_c) ? 0.0 : (cig_clamped - P_cig[lo_c]) / (P_cig[hi_c] - P_cig[lo_c])

    # E-cigarette price brackets
    hi_e = clamp(searchsortedfirst(P_ecig, ecig_clamped), 1, N_P)
    lo_e = clamp(hi_e - 1, 1, N_P)
    w_e  = (lo_e == hi_e) ? 0.0 : (ecig_clamped - P_ecig[lo_e]) / (P_ecig[hi_e] - P_ecig[lo_e])

    # Combined price grid indices for bilinear interpolation (4 corners)
    p_ll = (lo_c - 1) * N_P + lo_e
    p_lh = (lo_c - 1) * N_P + hi_e
    p_hl = (hi_c - 1) * N_P + lo_e
    p_hh = (hi_c - 1) * N_P + hi_e

    # Bilinear price weights
    w_ll = (1 - w_c) * (1 - w_e)
    w_lh = (1 - w_c) * w_e
    w_hl = w_c * (1 - w_e)
    w_hh = w_c * w_e

    # 8 addiction corner weights (trilinear over fast × slow × flavored)
    w_fff = (1 - w_af) * (1 - w_as) * (1 - w_aflav)
    w_ffs = (1 - w_af) * (1 - w_as) * w_aflav
    w_fsf = (1 - w_af) * w_as * (1 - w_aflav)
    w_fss = (1 - w_af) * w_as * w_aflav
    w_sff = w_af * (1 - w_as) * (1 - w_aflav)
    w_sfs = w_af * (1 - w_as) * w_aflav
    w_ssf = w_af * w_as * (1 - w_aflav)
    w_sss = w_af * w_as * w_aflav

    # 5-linear interpolation for all alternatives
    v_interp = Vector{Float64}(undef, N_J)

    for j in 1:N_J

        # Bilinear price interpolation at each of the 8 addiction corners
        v_fff = w_ll * V_choice[tya_idx, j, lo_af, lo_as, lo_aflav, p_ll] + w_lh * V_choice[tya_idx, j, lo_af, lo_as, lo_aflav, p_lh] + w_hl * V_choice[tya_idx, j, lo_af, lo_as, lo_aflav, p_hl] + w_hh * V_choice[tya_idx, j, lo_af, lo_as, lo_aflav, p_hh]
        v_ffs = w_ll * V_choice[tya_idx, j, lo_af, lo_as, hi_aflav, p_ll] + w_lh * V_choice[tya_idx, j, lo_af, lo_as, hi_aflav, p_lh] + w_hl * V_choice[tya_idx, j, lo_af, lo_as, hi_aflav, p_hl] + w_hh * V_choice[tya_idx, j, lo_af, lo_as, hi_aflav, p_hh]
        v_fsf = w_ll * V_choice[tya_idx, j, lo_af, hi_as, lo_aflav, p_ll] + w_lh * V_choice[tya_idx, j, lo_af, hi_as, lo_aflav, p_lh] + w_hl * V_choice[tya_idx, j, lo_af, hi_as, lo_aflav, p_hl] + w_hh * V_choice[tya_idx, j, lo_af, hi_as, lo_aflav, p_hh]
        v_fss = w_ll * V_choice[tya_idx, j, lo_af, hi_as, hi_aflav, p_ll] + w_lh * V_choice[tya_idx, j, lo_af, hi_as, hi_aflav, p_lh] + w_hl * V_choice[tya_idx, j, lo_af, hi_as, hi_aflav, p_hl] + w_hh * V_choice[tya_idx, j, lo_af, hi_as, hi_aflav, p_hh]
        v_sff = w_ll * V_choice[tya_idx, j, hi_af, lo_as, lo_aflav, p_ll] + w_lh * V_choice[tya_idx, j, hi_af, lo_as, lo_aflav, p_lh] + w_hl * V_choice[tya_idx, j, hi_af, lo_as, lo_aflav, p_hl] + w_hh * V_choice[tya_idx, j, hi_af, lo_as, lo_aflav, p_hh]
        v_sfs = w_ll * V_choice[tya_idx, j, hi_af, lo_as, hi_aflav, p_ll] + w_lh * V_choice[tya_idx, j, hi_af, lo_as, hi_aflav, p_lh] + w_hl * V_choice[tya_idx, j, hi_af, lo_as, hi_aflav, p_hl] + w_hh * V_choice[tya_idx, j, hi_af, lo_as, hi_aflav, p_hh]
        v_ssf = w_ll * V_choice[tya_idx, j, hi_af, hi_as, lo_aflav, p_ll] + w_lh * V_choice[tya_idx, j, hi_af, hi_as, lo_aflav, p_lh] + w_hl * V_choice[tya_idx, j, hi_af, hi_as, lo_aflav, p_hl] + w_hh * V_choice[tya_idx, j, hi_af, hi_as, lo_aflav, p_hh]
        v_sss = w_ll * V_choice[tya_idx, j, hi_af, hi_as, hi_aflav, p_ll] + w_lh * V_choice[tya_idx, j, hi_af, hi_as, hi_aflav, p_lh] + w_hl * V_choice[tya_idx, j, hi_af, hi_as, hi_aflav, p_hl] + w_hh * V_choice[tya_idx, j, hi_af, hi_as, hi_aflav, p_hh]

        # Trilinear interpolation over addiction
        v_interp[j] = w_fff * v_fff + w_ffs * v_ffs + w_fsf * v_fsf + w_fss * v_fss + w_sff * v_sff + w_sfs * v_sfs + w_ssf * v_ssf + w_sss * v_sss
    end

    return v_interp
end



"""
Pre-compute contiguous household index ranges from a sorted household code vector.

Assumes hh_codes are sorted so that all observations for the same household
are contiguous. Returns a vector of (start, stop) pairs, one per household.
Called once before estimation; the result is passed to log_likelihood_mixture()
to avoid recomputing every evaluation.

Returns:
- Vector{Tuple{Int, Int}} of (start_index, stop_index) pairs, one per unique household
"""
function precompute_hh_ranges(
    hh_codes::AbstractVector{<:Integer}
)

    # Initialize vector of (start, stop) index pairs
    ranges = Tuple{Int, Int}[]
    N = length(hh_codes)
    i = 1

    # Iterate through contiguous household blocks
    while i <= N

        # Current household code
        hh = hh_codes[i]

        # Mark the start of this household's observation block
        start_idx = i

        # Advance past all observations belonging to this household
        while i <= N && hh_codes[i] == hh
            i += 1
        end

        # Store the (start, stop) range for this household
        push!(ranges, (start_idx, i - 1))
    end

    return ranges
end


"""
Compute the mixture log-likelihood for a K=3 finite mixture model with
type-varying category fixed effects (ξ) and household-specific mixing
weights based on TYA share.

For each household i:
  π_logit_2_i = π_0_2 + π_TYA_2 · tya_share_i
  π_logit_3_i = π_0_3 + π_TYA_3 · tya_share_i
  log L_i = logsumexp_k( log π_k_i + Σ_t log P(y_it | x_it; θ_k) )

where θ_k = [common_params, ξ_k] and each type k has its own V_choice array.
The outer logsumexp aggregates across types, and the inner sum is over the
household's observations (time periods).
"""
function log_likelihood_mixture(
    V_choices::Vector{Array{Float64, 6}},
    π_0_2::Float64,
    π_TYA_2::Float64,
    π_0_3::Float64,
    π_TYA_3::Float64,
    tya_share_hh::Vector{Float64},
    N_J::Integer,
    N_P::Integer,
    A_f::AbstractVector{<:Real},
    A_s::AbstractVector{<:Real},
    A_flav::AbstractVector{<:Real},
    P::AbstractMatrix{<:Real},
    y::AbstractVector{<:Integer},
    tya_state::AbstractVector{<:Integer},
    af_continuous::AbstractVector{<:Real},
    as_continuous::AbstractVector{<:Real},
    aflav_continuous::AbstractVector{<:Real},
    p_continuous::AbstractMatrix{<:Real},
    ω_C::Float64,
    ω_E::Float64,
    P_obs_cig::AbstractMatrix{<:Real},
    P_obs_ecig::AbstractMatrix{<:Real},
    q_cig::AbstractVector{<:Real},
    q_ecig::AbstractVector{<:Real},
    hh_ranges::Vector{Tuple{Int, Int}}
)

    # Number of types and households
    K = length(V_choices)
    N_HH = length(hh_ranges)

    # Accumulate total log-likelihood across all households
    LL = 0.0

    # Loop over households
    for h in 1:N_HH

        # Get contiguous observation index range for this household
        start_idx, stop_idx = hh_ranges[h]

        # Compute household-specific log mixing weights via K=3 softmax.
        # Type 1 is the reference. Types 2 and 3 have
        # household-specific logit indices that shift their mixing probabilities.
        #
        # Step 1: Compute logit indices for types 2 and 3.
        logit_2_h = π_0_2 + π_TYA_2 * tya_share_hh[h]
        logit_3_h = π_0_3 + π_TYA_3 * tya_share_hh[h]

        # Step 2: Convert logit indices to log mixing probabilities via softmax.
        # logsumexp([0.0, logit_2_h, logit_3_h]) is numerically stable and avoids
        # overflow/underflow.
        #   log π_1_h = 0 - log_denom_h = -log_denom_h
        #   log π_2_h = logit_2_h - log_denom_h
        #   log π_3_h = logit_3_h - log_denom_h
        log_denom_h = logsumexp([0.0, logit_2_h, logit_3_h])
        log_π_1_h   = -log_denom_h
        log_π_2_h   = logit_2_h - log_denom_h
        log_π_3_h   = logit_3_h - log_denom_h

        # Step 3: Compute per-type log-likelihood for this household.
        # For each type k, we ask: "How well does type k's value function explain
        # this household's entire purchase history?" We sum the log choice probability
        # across all of the household's observed months:
        #   log L_k(h) = Σ_t log P(y_ht | state_ht; θ_k)
        # A type whose ξ values better match this household's purchasing pattern
        # produces higher choice probabilities, so a larger (less negative) log L_k.
        log_ll_k = zeros(K)

        # Loop over types
        for k in 1:K

            # Loop over this household's observations (time periods)
            for i in start_idx:stop_idx

                # 5-linear interpolation of V_choice_k at this observation's continuous state
                # (fast addiction, slow addiction, flavored habit, cig price, ecig price).
                #
                # Addiction stocks (af_continuous, as_continuous, aflav_continuous) are the
                # actual continuous values simulated from the household's observed purchase history
                # via simulate_addiction_trajectories, not grid approximations. No correction is
                # needed for addiction because the flow utility terms involving addiction are 
                # linear in the addiction stock, and linear interpolation recovers
                # them exactly:
                #   γ_1 * (w_lo * A_f[lo] + w_hi * A_f[hi]) = γ_1 * af_continuous[i]
                # So the interpolation at the actual continuous addiction stock is exact.
                #
                # Prices (p_continuous[i,1], p_continuous[i,2]) are the household's median cig
                # and ecig prices used as the interpolation coordinates. The correction
                # below is needed because the flow utility baked into V uses p_continuous[i,1]
                # as the effective cig price for all cig/bundle alternatives. This is a grid-based
                # approximation that differs from the actual observed bin price P_obs_cig[i,j].
                # Unlike addiction, P_obs_cig[i,j] varies across bins j and across households
                # within the same price state in a way that the aggregate grid price cannot
                # recover. P_obs_cig[i,j] from Prices.csv provides the actual price household
                # i faces in bin j, restoring the full cross-sectional and time-series price
                # variation needed to identify ω_C and ω_E.
                v_interp = interpolate_v_choice(
                    V_choices[k], tya_state[i], af_continuous[i], as_continuous[i],
                    aflav_continuous[i], p_continuous[i, 1], p_continuous[i, 2],
                    N_J, N_P, A_f, A_s, A_flav, P
                )

                # P_obs correction: nets out the baked-in grid price p_continuous[i] and
                # replaces it with the actual bin-specific price P_obs_cig[i,j] from Prices.csv.
                # Under the expenditure spec (ω*p*q), the net price term entering
                # the likelihood becomes ω_C * P_obs_cig[i,j] * q_cig[j]. The correction
                # is exact because price enters the flow utility linearly. The continuation
                # value is unchanged: future price dynamics follow from the aggregate price
                # state p_continuous[i], not from bin-specific prices.
                # v_corrected[j] = v_interp[j]
                #   + ω_C * (P_obs_cig[i,j]  - p_continuous[i,1]) * q_cig[j]
                #   + ω_E * (P_obs_ecig[i,j] - p_continuous[i,2]) * q_ecig[j]
                for j in 1:N_J
                    v_interp[j] += ω_C * (P_obs_cig[i, j]  - p_continuous[i, 1]) * q_cig[j] +
                                   ω_E * (P_obs_ecig[i, j] - p_continuous[i, 2]) * q_ecig[j]
                end

                # Log choice probability: log P(y_i) = V_choice(y_i) - logsumexp(V_choice)
                # This is the standard logit formula from the T1EV error assumption
                log_ll_k[k] += v_interp[y[i]] - logsumexp(v_interp)
            end
        end

        # Step 4: Integrate over types via logsumexp. This marginalizes over the
        # unobserved type. The model never commits a household to one type.
        # Conceptually: L_h = π_1_h·L_1(h) + π_2_h·L_2(h) + π_3_h·L_3(h), i.e.,
        # the weighted average likelihood across all three types. We compute this in
        # log space to avoid underflow when L_k(h) is a product of many small probs.
        mix_terms = [log_π_1_h + log_ll_k[1], log_π_2_h + log_ll_k[2], log_π_3_h + log_ll_k[3]]
        LL += logsumexp(mix_terms)
    end

    return LL
end


"""
Draw a single sample from a categorical distribution using the inverse CDF
(quantile function) method.

Given a probability vector probs where probs[j] = P(J = j), draws a
realization by comparing a uniform draw to the cumulative distribution.
This is the standard method for sampling from a discrete distribution:
  1. Draw u ~ Uniform(0, 1)
  2. Return the smallest j such that F(j) = Σ_{k=1}^{j} probs[k] ≥ u

For instance, say I have 3 alternatives with probabilities [0.2, 0.5, 0.3]. The CDF
partitions the unit interval into:

  |---0.2---|------0.5------|---0.3---|
  0        0.2             0.7        1.0
    j = 1        j = 2        j = 3

The function draws a uniform random number u and walks left to right,
accumulating probabilities. Whichever interval u lands in determines the choice:

  - u = 0.15 → cumulative hits 0.2 at j=1, and 0.15 ≤ 0.2, so return j=1
  - u = 0.45 → cumulative passes 0.2 (skip), hits 0.7 at j=2, and 0.45 ≤ 0.7, so return j=2
  - u = 0.85 → cumulative passes 0.2 and 0.7 (skip both), hits 1.0 at j=3, so return j=3

Each alternative is chosen with probability equal to its interval width, which
is exactly probs[j]. The fallback `return length(probs)` at the end handles
the edge case where floating-point rounding makes the probabilities sum to
0.9999... instead of exactly 1.0, so u could slightly exceed the accumulated sum.

Returns:
- Sampled category index j ∈ {1, ..., length(probs)}
"""
function categorical_sample(
    probs::AbstractVector{<:Real}
)

    # Draw a uniform random number on [0, 1)
    u = rand()

    # Walk through the CDF: accumulate probabilities until I exceed u.
    # This partitions [0, 1) into intervals [F(j-1), F(j)) for each j,
    # where F(0) = 0 and F(j) = Σ_{k=1}^{j} probs[k]. The drawn j is
    # whichever interval u falls into.
    cumulative = 0.0
    for j in eachindex(probs)
        cumulative += probs[j]
        if u <= cumulative
            return j
        end
    end

    # If floating-point rounding causes Σ probs < 1 so that
    # u exceeds the accumulated sum, return the last category. This
    # can happen when probs doesn't sum to exactly 1.0 due to
    # finite-precision arithmetic in the softmax computation.
    return length(probs)
end


#############################
# 10. Optimization &
#     Objective
#############################


# Economic parameter bounds (standardized units).
# K=3 mixture ordering: [13 common, 3 type1, 3 type2, 3 type3, π_0_2, π_TYA_2, π_0_3, π_TYA_3] = 26 params
# When ESTIMATE_PSI_3=true, ψ_3 is appended as param 27: bounds (0.01, 0.99)
# Common: α_C, α_E, α_CE, λ_1, λ_2, λ_3, λ_4, γ_1, γ_2, γ_3, γ_4, ω_C, ω_E
# Type-specific (3 each): ξ_C_k, ξ_E_k, ξ_CE_k (k=1,2 unconstrained; k=3 lower bound -30 to prevent degenerate non-purchaser type)
# Mixing: π_0_2, π_TYA_2 (type 2 logit intercept and TYA shifter, unconstrained)
#         π_0_3, π_TYA_3 (type 3 logit intercept and TYA shifter, unconstrained)
# Economic sign restrictions: α_C ≥ 0, α_E ≥ 0 (non-negative consumption utility),
#                             γ_1 ≤ 0 (withdrawal cost),
#                             γ_2 ≤ 0 (orig ecig/bundle lock-in penalty),
#                             γ_3 ≤ 0 (cig/orig bundle flavor lock-in penalty),
#                             γ_4 ≤ 0 (flavored withdrawal cost on outside option),
#                             ω_C ≤ 0, ω_E ≤ 0 (price disutility).
#                             α_C  α_E  α_CE   λ_1    λ_2    λ_3    λ_4    γ_1   γ_2   γ_3   γ_4  ω_C   ω_E   ξ_C_1 ξ_E_1 ξ_CE_1 ξ_C_2 ξ_E_2 ξ_CE_2  ξ_C_3  ξ_E_3 ξ_CE_3 π_0_2 π_TYA_2 π_0_3 π_TYA_3
θ_lower_bound = Float64[      0.0,  0.0, -Inf, -Inf,  -Inf,  -Inf,  -Inf,  -Inf, -Inf, -Inf, -Inf, -Inf, -Inf, -Inf, -Inf,  -Inf,  -Inf, -Inf,  -Inf, -Inf, -Inf, -Inf, -Inf, -Inf, -Inf, -Inf]
θ_upper_bound = Float64[      Inf,  Inf,  Inf,  Inf,   Inf,   Inf,   Inf,   0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  Inf,  Inf,   Inf,   Inf,  Inf,   Inf,  Inf,  Inf,  Inf,  Inf,  Inf,  Inf,  Inf]

# Append ψ_3 bounds when estimating it (length guard prevents double-push on re-include)
if @isdefined(ESTIMATE_PSI_3) && ESTIMATE_PSI_3 && length(θ_lower_bound) == 26
    push!(θ_lower_bound, 0.01)   # ψ_3 > 0
    push!(θ_upper_bound, 0.99)   # ψ_3 < 1
end

"""
Check whether θ_vec satisfies the economic parameter bounds.

Returns 
- Bool indicating whether there are violations or not
- Vector of strings indicating the potential violations
"""
function check_parameter_bounds(
    θ_vec::AbstractVector{<:Real}, 
    param_names_vec::AbstractVector{<:AbstractString}
)
    # Initialize vector of strings to store violation parameters 
    violated = String[]

    # Loop over indices of θ
    for i in eachindex(θ_vec)

        # Extract parameter name
        pname = param_names_vec[i]

        # Check parameter violations 
        if θ_vec[i] < θ_lower_bound[i]

            # Store violation string in place 
            push!(violated, "$pname=$(round(θ_vec[i], digits=4))<$(θ_lower_bound[i])")

        # Check parameter violations 
        elseif θ_vec[i] > θ_upper_bound[i]

            # Store violation string in place 
            push!(violated, "$pname=$(round(θ_vec[i], digits=4))>$(θ_upper_bound[i])")
        end
    end

    return isempty(violated), join(violated, ", ")
end


"""
Nelder-Mead operates on a simplex in D-dimensional parameter space.
A simplex in D dimensions has (D+1) vertices (e.g., a triangle in 2D,
a tetrahedron in 3D). This struct implements the Optim.Simplexer interface so that
Optim.jl's NelderMead can construct its initial simplex.

The simplex is built around a starting point x0 by perturbing each
coordinate direction d independently:
  Vertex 0: x0                      (the starting point itself)
  Vertex d: x0 + add[d] * e_d       (shifted along dimension d)
where e_d is the d-th standard basis vector. I need the adding
otherwise all three vertices would be the same and Nelder-Mead would not
be able to form conditions to move towards the optimum.

The add vector controls the initial simplex size along each dimension
and should be set proportional to the expected magnitude of each parameter.

Returns:
- A vector of (D+1) vertices forming the initial simplex
"""

# Define an object that Optim.Simplexer can handle
# This object contains the add vector
# SimplexWithAdd <: Optim.Simplexer means
# SimplexWithAdd is a subtype of Optim.Simplexer
struct SimplexWithAdd <: Optim.Simplexer
    add::Vector{Float64}
end

function Optim.simplexer(
    S::SimplexWithAdd,
    x0::AbstractVector{<:Real}
)

    # Number of parameters
    D = length(x0)

    # Allocate (D+1) vertices
    simplex = Vector{Vector{Float64}}(undef, D + 1)

    # First vertex is the starting point
    simplex[1] = copy(x0)

    # Remaining D vertices: shift x0 along each coordinate direction corresponding to S.add
    for d in 1:D
        
        # Copy of original vertex (the actual parameter vector)
        vertex = copy(x0)

        # Perturb d'th dimension of the vertex
        vertex[d] += S.add[d]

        # Simplex at index d + 1 is the perturbed vertex
        # d + 1 b/c the simplex at position 1 is simply the actual parameter vector 
        simplex[d + 1] = vertex
    end

    return simplex
end


"""
Modified Nelder-Mead optimizer with random restarts.

Addresses the key weakness of Nelder-Mead (sensitivity to initialization)
by running multiple outer tries with random reinitializations.

Outer loop (L outer tries):
  1. Inner loop (M inner tries): short Nelder-Mead runs with randomized
     simplex deviations to cheaply explore the parameter space
  2. Best update: track the global best across all outer tries
  3. Random reinitialization: stochastically choose starting point for
     the next outer try to balance exploration and exploitation

Returns:
- Optimal parameters as a NamedTuple matching the keys of starting_param
- The minimum objective value
"""
function random_amoeba(
    objective::Function,
    starting_param::NamedTuple,
    add::Vector{Float64},
    L::Integer,
    M::Integer;
    inner_iter::Integer = 250,
    outer_try_file::Union{AbstractString, Nothing} = nothing,
    inner_try_file::Union{AbstractString, Nothing} = nothing
)

    # Starting values of the parameters and never modified
    base_param = collect(Float64, values(starting_param))

    # Working vector θ which is updated in each iteration of the inner try
    # and then at the end of each iteration of the outer try
    param = base_param

    # Stores best parameters from each outer try
    best_param = base_param

    # Number of parameters
    N_params = length(param)

    # Names of the parameters
    pnames = replace.(collect(String, string.(keys(starting_param))),
        "α" => "alpha", "λ" => "lambda", "γ" => "gamma",
        "ω" => "omega", "ξ" => "xi",
        "ψ" => "psi", "β" => "beta")

    # Initialize global best objective to a very large value
    overall_min = Inf

    # Initialize file for writing outer try results (if path provided)
    outer_try_io = nothing
    if outer_try_file !== nothing

        # Open the outer try file
        outer_try_io = open(outer_try_file, "w")

        # Write header row with NLL and parameter names
        println(outer_try_io, "outer_try,NLL," * join(pnames, ","))
        flush(outer_try_io)
    end

    # Initialize file for writing inner try results (if path provided)
    inner_try_io = nothing
    if inner_try_file !== nothing

        # Open the inner try file
        inner_try_io = open(inner_try_file, "w")

        # Write header row with NLL and parameter names
        println(inner_try_io, "outer_try,inner_try,NLL," * join(pnames, ","))
        flush(inner_try_io)
    end


    #############################
    # Outer Tries
    #############################

    # Print and log message indicating full optimization routine is starting
    log_msg("")
    log_msg("="^60)
    log_msg("RANDOM AMOEBA OPTIMIZATION")
    log_msg("="^60)
    log_msg("")
    log_msg("Starting Parameters:")
    log_msg("")
    for d in 1:N_params
        log_msg(@sprintf("  %s = %.4f", pnames[d], base_param[d]))
    end
    log_msg("")

    # Print and log simplex deviations
    log_msg("Simplex Deviations:")
    log_msg("")
    for (i, dev) in enumerate(add)
        log_msg(@sprintf("  %s = %.4f", pnames[i], dev))
    end
    log_msg("")

    # Time all outer tries
    t_total = time()

    # Loop over outer tries
    for l in 1:L

        # Time the current outer try
        t_outer = time()

        # Reset simplex deviations to a copy of the base values at the start of each
        # outer try. Using copy() prevents in-place mutations of this_add from
        # accidentally corrupting the original add vector.
        this_add = copy(add)

        # Print and log message indicating current outer try
        log_msg("")
        log_msg("-"^60)
        log_msg("OUTER TRY $l / $L")
        log_msg("-"^60)


        #############################
        # Inner Tries
        #############################

        # Track best parameters across inner tries
        best_inner_val = Inf
        best_inner_param = param

        # Simplex deviations decay exponentially across inner tries so that early
        # inner tries explore broadly (full add) and later ones do fine-grained
        # search near the minimum (add_min_frac × add at m = M).
        #
        # Decay schedule: this_add ∝ add × inner_shrink^(m-1)
        #   m = 1:  1.00 × add  (full exploration)
        #   m = M:  0.10 × add  (fine-grained search)
        #
        # inner_shrink is computed so that inner_shrink^(M-1) = add_min_frac.
        # Combined with U(0.5, 2.0) random scaling for simplex shape diversity.
        add_min_frac = 0.10
        inner_shrink = add_min_frac^(1 / max(M - 1, 1))

        # Loop over inner tries
        for m in 1:M

            # Update global phase tracking
            global ra_outer_try = l
            global ra_inner_run = m

            # Time the current inner try
            t_inner = time()

            # Print and log message regarding current minimization run
            log_msg("")
            log_msg("")
            log_msg("  Minimization run $l.$m")
            log_msg("  " * "-"^22)
            log_msg("")

            # Run Nelder-Mead for a limited number of iterations (inner_iter iterations)
            # Each iteration can call the objective many times.
            # See paper appendix for details on Nelder-Mead
            result = optimize(
                objective, param,
                NelderMead(initial_simplex = SimplexWithAdd(this_add)),
                Optim.Options(iterations = inner_iter, f_abstol = 1e-2)
            )

            # Store unperturbed minimizer (used for best tracking, logging, and CSV)
            minimizer = Optim.minimizer(result)

            # Update best inner result if this run improved (uses unperturbed minimizer)
            if Optim.minimum(result) < best_inner_val
                best_inner_val = Optim.minimum(result)
                best_inner_param = minimizer
            end

            # Time the current inner try
            inner_elapsed = time() - t_inner

            # Print and log short run results (unperturbed minimizer, matches NLL)
            log_msg("")
            log_msg("    Run $l.$m Complete:")
            log_msg("")
            log_msg("      Time: $(round(inner_elapsed, digits=1))s | " *
                    "Iterations: $(Optim.iterations(result))/$inner_iter | " *
                    "Converged: $(Optim.converged(result)) | " *
                    "Objective: $(round(Optim.minimum(result), digits=6))")
            log_msg("")
            log_msg("      Parameters:")
            log_msg("")
            for d in 1:N_params
                log_msg("        $(pnames[d]) = $(round(minimizer[d], digits=8))")
            end

            # Write this inner try's NLL and parameters to file (unperturbed minimizer)
            if inner_try_io !== nothing

                # Join minimizer vector into a single string separated by commas
                param_str = join([@sprintf("%.10f", x) for x in minimizer], ",")

                # Write current outer run, inner run, NLL, and parameters for the current inner run to a file
                println(inner_try_io, "$l,$m,$(@sprintf("%.10f", Optim.minimum(result))),$param_str")
                flush(inner_try_io)
            end

            # Decay simplex deviations for next inner try.
            # Exponential decay: add × inner_shrink^m, with U(0.5, 2.0) random
            # scaling for simplex shape diversity. At m=1 (after first inner try),
            # deviations are ~inner_shrink × add; at m=M, ~add_min_frac × add.
            this_add = add .* inner_shrink^m .* (0.5 .+ 1.5 .* rand(N_params))

            # Perturb starting point for next inner try using the decayed simplex
            # deviations. The perturbation is ±25% of this_add, so it shrinks
            # naturally as the simplex deviations decay across inner tries:
            #   m = 1:  moderate perturbation (explore nearby)
            #   m = M:  tiny perturbation (fine-tune near minimum)
            param = minimizer .+ 0.25 .* this_add .* (2.0 .* rand(N_params) .- 1.0)
        end


        # Update global best θ* if this outer try produced a lower minimum
        if best_inner_val < overall_min
            overall_min = best_inner_val
            best_param  = best_inner_param
        end

        # Time the current outer try
        outer_elapsed = time() - t_outer

        # Print and log message from this outer try
        log_msg("")
        log_msg("")
        log_msg("    OUTER TRY $l SUMMARY:")
        log_msg("    " * "-"^22)
        log_msg("")
        log_msg("      Time: $(round(outer_elapsed, digits=1))s")
        log_msg("      Overall Best Objective: $(round(overall_min, digits=6))")
        log_msg("      Best Parameters:")
        for d in 1:N_params
            log_msg("        $(pnames[d]) = $(round(best_param[d], digits=8))")
        end

        # Write this outer try's best NLL and parameters to file
        if outer_try_io !== nothing

            # Join best_inner_param vector into a single string separated by commas
            param_str = join([@sprintf("%.10f", x) for x in best_inner_param], ",")

            # Write current outer run, NLL, and parameters for the current outer run to a file
            println(outer_try_io, "$l,$(@sprintf("%.10f", best_inner_val)),$param_str")
            flush(outer_try_io)
        end

        # ε-greedy reinitialization with exponential decay for next outer try.
        #
        # Early outer tries favor exploration (perturbed starting parameters) to
        # discover different basins. Later outer tries shift toward exploitation
        # (perturbed best / current) to refine the best solution found.
        #
        # Exploration probability decays exponentially from ε_start to ε_end:
        #   ε(l) = ε_end + (ε_start - ε_end) · exp(-λ · (l - 1))
        #
        # With ε_start = 0.75, ε_end = 0.25, λ = 0.077 (chosen so ε(10) ≈ 0.50):
        #   l =  1 → ε = 0.75  (75% explore, 12.5% best, 12.5% current)
        #   l = 10 → ε = 0.50  (50% explore, 25% best, 25% current)
        #   l = 25 → ε = 0.32  (32% explore, 34% best, 34% current)
        #   l = 50 → ε = 0.26  (26% explore, 37% best, 37% current)
        #
        # All three branches perturb by ±add to ensure every restart explores
        # new territory (no restart ever uses the exact same parameter vector).
        ε_start = 0.75
        ε_end   = 0.25
        λ_decay = 0.077
        ε = ε_end + (ε_start - ε_end) * exp(-λ_decay * (l - 1))

        u = rand()
        if u < ε
            log_msg("")
            log_msg("-"^50)
            log_msg(@sprintf("Reinitializing: Perturbed Starting Parameters (ε = %.2f, u = %.2f)", ε, u))
            log_msg("-"^50)
            param = base_param .+ add .* (2.0 .* rand(N_params) .- 1.0)
        elseif u < ε + (1 - ε) / 2
            log_msg("")
            log_msg("-"^50)
            log_msg(@sprintf("Reinitializing: Perturbed Best Parameters (ε = %.2f, u = %.2f)", ε, u))
            log_msg("-"^50)
            param = best_param .+ add .* (2.0 .* rand(N_params) .- 1.0)
        else
            log_msg("")
            log_msg("-"^50)
            log_msg(@sprintf("Reinitializing: Perturbed Current Parameters (ε = %.2f, u = %.2f)", ε, u))
            log_msg("-"^50)
            param = best_inner_param .+ add .* (2.0 .* rand(N_params) .- 1.0)
        end
    end

    # Close inner try results file
    if inner_try_io !== nothing
        close(inner_try_io)
    end

    # Close outer try results file
    if outer_try_io !== nothing
        close(outer_try_io)
    end

    # Time all outer tries
    total_elapsed = time() - t_total

    # Print and log message indicating results from full optimization run
    log_msg("")
    log_msg("="^60)
    log_msg("RANDOM AMOEBA COMPLETE")
    log_msg("="^60)
    log_msg("Total time: $(round(total_elapsed, digits=1))s")
    log_msg("Final objective: $(round(overall_min, digits=6))")
    log_msg("Final parameters:")
    for d in 1:N_params
        log_msg("  $(pnames[d]) = $(round(best_param[d], digits=8))")
    end
    log_msg("")

    # Reconstruct named tuple with the same keys as starting_param
    # E.g., (α_C = 0.5, α_E = 0.2, ...)
    opt_param = (; zip(keys(starting_param), best_param)...)

    return opt_param, overall_min
end


# Objective Function 

"""
Objective function for the optimizer.

Takes a parameter vector θ_vec (26 structural params for K=3 mixture, 27 when ESTIMATE_PSI_3=true),
recomputes flow utility and value function, evaluates the log-likelihood,
and returns the negative log-likelihood.

Increments `est_eval_count` and times each evaluation. **Box constraints:**
returns `1e14` penalty if any parameter violates bounds via `check_parameter_bounds`.

For each candidate θ:
  (1) computes flow utility U (6D: TYA × alternatives × fast addiction × slow addiction × flavored habit × price)
  (2) solves VFI using pre-computed addiction transitions (fast, slow, flavored habit)
  (3) evaluates log-likelihood 
  (4) returns the negative log-likelihood (since we minimize)
 
K=3 mixture objective: extracts common params (positions 1-13), type-specific ξ
(14-16, 17-19, and 20-22), π_0_2 (23), π_TYA_2 (24), π_0_3 (25), and π_TYA_3 (26).
Solves 3 VFI problems in parallel via Threads.@spawn (one per type, each internally
using Threads.@threads over N_Pcomb×N_A_s×N_A_flav state triples), then evaluates
the mixture log-likelihood with household-specific mixing weights:
π_logit_k_h = π_0_k + π_TYA_k · tya_share_h.

Fixed parameters: ψ_2 = 0.90, ψ_1 = 0.10, β = 1.0, δ = 0.99.

Accesses global data loaded by 02_Estimation.jl (e.g., N_J, y, hh_ranges, etc.)
so each data does not need to be passed as arguments.
"""

# Determine whether to print verbose output for this evaluation.
# Prints evals 1-10 and every 50th eval thereafter. Always prints penalties.
function should_print_eval(eval_num::Integer)

    return eval_num <= 10 || eval_num % 50 == 0
end


function objective(θ_vec::AbstractVector{<:Real})

    # Use global evaluation counter
    global est_eval_count

    # Start evaluation time
    t_eval = time()

    # Update evaluation count
    est_eval_count += 1

    # Determine whether to print verbose output for this eval
    print_this = should_print_eval(est_eval_count)

    # Print evaluation header
    if print_this
        log_msg("")
        log_msg(@sprintf("    Objective Evaluation %d:", est_eval_count))
        log_msg("")
    end

    # Economic parameter bounds check
    # If any parameter falls outside their respective bounds, return penalty (very large number for LL)
    in_bounds, violations = check_parameter_bounds(θ_vec, est_param_names)
    if !in_bounds

        # Always print bounds violations regardless of eval number
        log_msg("")
        log_msg(@sprintf("      PENALTY (bounds: %s) | time = %.1fs", violations, time() - t_eval))

        return 1e14
    end

    # Extract K=3 mixture parameters from positions 1-26
    common    = θ_vec[1:13]      # α_C, α_E, α_CE, λ_1, λ_2, λ_3, λ_4, γ_1, γ_2, γ_3, γ_4, ω_C, ω_E
    type_1    = θ_vec[14:16]     # ξ_C_1, ξ_E_1, ξ_CE_1
    type_2    = θ_vec[17:19]     # ξ_C_2, ξ_E_2, ξ_CE_2
    type_3    = θ_vec[20:22]     # ξ_C_3, ξ_E_3, ξ_CE_3
    π_0_2     = θ_vec[23]        # type 2 baseline logit intercept
    π_TYA_2   = θ_vec[24]        # type 2 TYA share shifter
    π_0_3     = θ_vec[25]        # type 3 baseline logit intercept
    π_TYA_3   = θ_vec[26]        # type 3 TYA share shifter

    # Extract price coefficients (common params 12-13) for the P_obs correction in the likelihood
    ω_C = Float64(common[12])
    ω_E = Float64(common[13])

    # When estimating ψ_3, extract it from position 27 and recompute flavored habit objects
    if ESTIMATE_PSI_3
        ψ_3_val = θ_vec[27]

        # Recompute flavored habit transitions, initial stocks, and trajectories at candidate ψ_3
        aflav_lower_eval, aflav_upper_eval, aflav_weight_eval = precompute_addiction_transitions(N_J, N_A_flav, ψ_3_val, A_flav, n_flav)
        aflav0_eval, _ = get_initial_addiction_stock(ψ_3_val, A_flav, n_flav, y, hh_codes)
        aflav_continuous_eval = simulate_addiction_trajectories(N_A_flav, ψ_3_val, A_flav, n_flav, y, hh_codes, aflav0_eval)
    else
        # Use pre-computed globals
        aflav_lower_eval = aflav_lower_current
        aflav_upper_eval = aflav_upper_current
        aflav_weight_eval = aflav_weight_current
        aflav_continuous_eval = aflav_continuous_current
    end

    # Construct per-type structural parameter vectors (16 elements each, matching get_flow_utility format)
    θ_struct_1 = vcat(common, type_1)
    θ_struct_2 = vcat(common, type_2)
    θ_struct_3 = vcat(common, type_3)

    # Compute flow utility for each type (each receives a 16-element θ_struct_k)
    U_1 = get_flow_utility(
        θ_struct_1, N_J, N_A_f, N_A_s, N_A_flav, N_Pcomb, A_f, A_s, A_flav,
        q_cig, q_ecig, q_bundle, is_flavored, is_fda_flavored, is_nonflavored_ecig, is_outside, cat_idx, Pcomb, has_cig, has_ecig
    )
    U_2 = get_flow_utility(
        θ_struct_2, N_J, N_A_f, N_A_s, N_A_flav, N_Pcomb, A_f, A_s, A_flav,
        q_cig, q_ecig, q_bundle, is_flavored, is_fda_flavored, is_nonflavored_ecig, is_outside, cat_idx, Pcomb, has_cig, has_ecig
    )
    U_3 = get_flow_utility(
        θ_struct_3, N_J, N_A_f, N_A_s, N_A_flav, N_Pcomb, A_f, A_s, A_flav,
        q_cig, q_ecig, q_bundle, is_flavored, is_fda_flavored, is_nonflavored_ecig, is_outside, cat_idx, Pcomb, has_cig, has_ecig
    )

    # Warm-start: reuse the previous V as initial guess within a NM run (K=3 warm-start).
    # Reset all V_warm arrays when the optimizer phase changes (new outer try, inner run, or long run).
    if WARM_START

        # Access global value functions for all three types
        global V_warm_est_1, V_warm_est_2, V_warm_est_3, last_ra_phase_est

        # Get current outer and inner try
        current_phase = (ra_outer_try, ra_inner_run)

        # If the outer and inner runs are different than before
        if current_phase != last_ra_phase_est

            # Reset all three value functions
            V_warm_est_1 = nothing
            V_warm_est_2 = nothing
            V_warm_est_3 = nothing

            # Update the outer and inner run
            last_ra_phase_est = current_phase
        end

        # Set initial V for each type
        V_init_1 = V_warm_est_1
        V_init_2 = V_warm_est_2
        V_init_3 = V_warm_est_3
    else

        # If not doing warm start, then set all to nothing
        V_init_1 = nothing
        V_init_2 = nothing
        V_init_3 = nothing
    end

    # Solve VFI for all three types in parallel. The three VFI problems are independent
    # (same addiction/price transitions, different flow utilities U_1, U_2, U_3), so
    # Threads.@spawn launches them concurrently. Each VFI internally uses
    # Threads.@threads over N_Pcomb×N_A_s×N_A_flav state triples; Julia's task scheduler distributes
    # all tasks across the available thread pool.

    # Launch VFI for type 1 (returns immediately, runs on available threads)
    vfi_task_1 = Threads.@spawn solve_vfi_sophisticated(
        N_J, N_A_f, N_A_s, N_A_flav, N_P, N_Pcomb, β, δ, U_1,
        af_lower_current, af_upper_current, af_weight_current,
        as_lower_current, as_upper_current, as_weight_current,
        aflav_lower_eval, aflav_upper_eval, aflav_weight_eval,
        p_cig_lo, p_cig_hi, p_cig_w,
        p_ecig_lo, p_ecig_hi, p_ecig_w;
        V_init = V_init_1,
        ε = VFI_TOL,
        verbose = print_this
    )

    # Launch VFI for type 2 (returns immediately, runs on available threads)
    vfi_task_2 = Threads.@spawn solve_vfi_sophisticated(
        N_J, N_A_f, N_A_s, N_A_flav, N_P, N_Pcomb, β, δ, U_2,
        af_lower_current, af_upper_current, af_weight_current,
        as_lower_current, as_upper_current, as_weight_current,
        aflav_lower_eval, aflav_upper_eval, aflav_weight_eval,
        p_cig_lo, p_cig_hi, p_cig_w,
        p_ecig_lo, p_ecig_hi, p_ecig_w;
        V_init = V_init_2,
        ε = VFI_TOL,
        verbose = print_this
    )

    # Launch VFI for type 3 (returns immediately, runs on available threads)
    vfi_task_3 = Threads.@spawn solve_vfi_sophisticated(
        N_J, N_A_f, N_A_s, N_A_flav, N_P, N_Pcomb, β, δ, U_3,
        af_lower_current, af_upper_current, af_weight_current,
        as_lower_current, as_upper_current, as_weight_current,
        aflav_lower_eval, aflav_upper_eval, aflav_weight_eval,
        p_cig_lo, p_cig_hi, p_cig_w,
        p_ecig_lo, p_ecig_hi, p_ecig_w;
        V_init = V_init_3,
        ε = VFI_TOL,
        verbose = print_this
    )

    # Wait for all three VFI tasks to complete and collect results
    V_1, V_decision_1, vfi_iters_1, vfi_converged_1 = fetch(vfi_task_1)
    V_2, V_decision_2, vfi_iters_2, vfi_converged_2 = fetch(vfi_task_2)
    V_3, V_decision_3, vfi_iters_3, vfi_converged_3 = fetch(vfi_task_3)

    # If any VFI did not converge, skip LL and return penalty
    if !vfi_converged_1 || !vfi_converged_2 || !vfi_converged_3

        # Get elapsed time
        elapsed = time() - t_eval
        nll = 1e14

        # Print and log message
        log_msg("")
        θ_str = join([@sprintf("%.6f", x) for x in θ_vec], ", ")
        log_msg(@sprintf("      PENALTY (VFI not converged: type1=%s, type2=%s, type3=%s) | VFI iters = %d/%d/%d | time = %.1fs | θ = [%s]",
            vfi_converged_1 ? "ok" : "FAIL", vfi_converged_2 ? "ok" : "FAIL", vfi_converged_3 ? "ok" : "FAIL",
            vfi_iters_1, vfi_iters_2, vfi_iters_3, elapsed, θ_str))

        return nll
    end

    # Store converged V for warm-starting the next evaluation (all three types).
    # Only store when all VFIs converged; unconverged V (penalty case) is not stored.
    if WARM_START
        V_warm_est_1 = V_1
        V_warm_est_2 = V_2
        V_warm_est_3 = V_3
    end

    # Compute mixture log-likelihood via 5-linear interpolation at continuous states
    # Mixing weights are household-specific K=3 softmax: logit_k_h = π_0_k + π_TYA_k · tya_share_h
    LL = log_likelihood_mixture(
        [V_decision_1, V_decision_2, V_decision_3], π_0_2, π_TYA_2, π_0_3, π_TYA_3, tya_share_hh,
        N_J, N_P, A_f, A_s, A_flav, P,
        y, tya_state, af_continuous_current, as_continuous_current, aflav_continuous_eval, p_continuous,
        ω_C, ω_E, P_obs_cig, P_obs_ecig, q_cig, q_ecig,
        hh_ranges
    )

    nll = -LL

    # Get elapsed time
    elapsed = time() - t_eval

    # Print and log message for current objective evaluation
    if print_this
        log_msg("")
        θ_str = join([@sprintf("%.6f", x) for x in θ_vec], ", ")
        # Compute average mixing weights across households for logging (K=3 softmax)
        avg_π = [0.0, 0.0, 0.0]
        for s in tya_share_hh
            l2 = π_0_2 + π_TYA_2 * s
            l3 = π_0_3 + π_TYA_3 * s
            denom = 1.0 + exp(l2) + exp(l3)
            avg_π[1] += 1.0 / denom
            avg_π[2] += exp(l2) / denom
            avg_π[3] += exp(l3) / denom
        end
        avg_π ./= length(tya_share_hh)
        log_msg(@sprintf("      NLL = %.4f | VFI iters = %d/%d/%d | avg π = (%.3f, %.3f, %.3f) | time = %.1fs | θ = [%s]",
            nll, vfi_iters_1, vfi_iters_2, vfi_iters_3, avg_π[1], avg_π[2], avg_π[3], elapsed, θ_str))
    end

    # Return negative log-likelihood (optimizer minimizes)
    return nll
end


