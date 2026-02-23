################################################################################
# William Brasic
# The University of Arizona
# wbrasic97@gmail.com
# February 2026
#
# Beta estimation variant of 01_Functions.jl.
#
# Changes from 01_Functions.jl:
#   - get_fixed_parameters() returns (ψ, δ) only; β is now estimated
#   - New get_tya_states() loads 4-state TYA from TYA_States.csv
#   - New get_tya_transitions() loads 4×4 transition matrix
#   - get_flow_utility() uses N_TYA = 4 (states 1,2 → tya=0; states 3,4 → tya=1)
#   - solve_vfi() takes Π_tya and integrates over TYA transitions
#   - Parameter bounds extended to 14 elements (β is 14th)
#   - objective() takes 14-element θ_vec (β = θ_vec[14])
################################################################################


################################################################################
# Table of Contents
#
#  1. Preliminaries              — Package loading
#  2. Logging                    — log_io, est_eval_count, log_msg
#  3. Data Loading: Choices      — get_hh_codes, get_product_choices,
#                                  get_hh_choices, get_category_choices
#  4. Data Loading: Alternative  — get_consumption, get_nicotine,
#     Attributes                   get_category_index, get_flavored_indicator
#  5. Data Loading: Demographics — get_teen_young_adult, get_tya_states,
#                                  get_tya_transitions
#  6. Data Loading: Prices       — get_fixed_parameters, get_pricing_spaces,
#                                  get_pricing_spaces_combination,
#                                  get_price_ratios, get_expenditures,
#                                  get_transitions, precompute_price_transitions,
#                                  map_prices_to_grid
#  7. Addiction Dynamics          — get_addiction_space, addiction_evolution,
#                                  get_flow_utility,
#                                  precompute_addiction_transitions,
#                                  get_initial_addiction_stock,
#                                  simulate_addiction_trajectories
#  8. Value Function Iteration   — logsumexp, solve_vfi, solve_vfi_sophisticated
#  9. Likelihood & Prediction    — log_likelihood, interpolate_v_choice,
#                                  categorical_sample
# 10. Optimization & Objective   — θ_lower_bound, θ_upper_bound,
#                                  check_parameter_bounds, SimplexWithAdd,
#                                  random_amoeba, objective
################################################################################


#############################
# 1. Preliminaries
#############################

# Install any missing packages, then load
import Pkg
for pkg in ["CSV", "DataFrames", "Optim", "Statistics", "ForwardDiff"]
    if Base.find_package(pkg) === nothing
        Pkg.add(pkg)
    end
end
using CSV, DataFrames, Optim, Statistics, ForwardDiff, LinearAlgebra, Printf, Dates


#############################
# 2. Logging
#############################

# Global log file handle. Each calling script sets this once before logging:
#   log_io = open("My_Log.txt", "w")
# All logging across estimation, MC, validation, and counterfactual uses this single handle.
log_io = nothing

# Counter for how many times the objective was called
# Note, each Nelder-Mead iteration can call the objective many times
# See paper appendix for details 
est_eval_count = 0

# Global optimizer phase tracking (updated by random_amoeba)
# ra_outer_try: current outer try (1 to L)
# ra_inner_run: current inner run (1 to M), or 0 for the long convergence run
ra_outer_try = 0
ra_inner_run = 0

# Global parameter names (set by calling script before estimation)
# Used to print parameter names in objective function logging
est_param_names = String[]

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
  j = 35:36:       2 original bundles (lo/hi cig)
  j = 37:38:       2 non-FDA flavored bundles (lo/hi cig)
  j = 39:40:       2 FDA flavored bundles (lo/hi cig)

Returns:
- N_cig:               Number of cigarette alternatives
- N_orig_ecig:         Number of original e-cigarette alternatives
- N_non_fda_flav_ecig: Number of non-FDA flavored e-cigarette alternatives
- N_fda_flav_ecig:     Number of FDA flavored e-cigarette alternatives
- N_bundle:            Number of bundle alternatives
- c_cig:               Vector of standardized cigarette consumption for each alternative
- c_ecig:              Vector of standardized e-cigarette consumption for each alternative
- c_bundle:            Vector of standardized bundle consumption for each alternative
- c_cig_max:           Raw maximum cigarette consumption (for rescaling estimates)
- c_ecig_max:          Raw maximum e-cigarette consumption (for rescaling estimates)
- c_bundle_max:        Raw maximum bundle consumption (for rescaling estimates)
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

    # Bundle names in order: 2 original, 2 non-FDA flavored, 2 FDA flavored
    bundle_names = [
        "bundle_orig_lo", "bundle_orig_hi",
        "bundle_non_fda_flav_lo", "bundle_non_fda_flav_hi",
        "bundle_fda_flav_lo", "bundle_fda_flav_hi"
    ]

    # Helper to extract single consumption value by alternative name
    get_consumption_value(alt_name) = only(df.consumption[df.alternative .== alt_name])

    # Counts
    N_cig               = length(cig)
    N_orig_ecig         = length(orig_ecig)
    N_non_fda_flav_ecig = length(non_fda_flav_ecig)
    N_fda_flav_ecig     = length(fda_flav_ecig)
    N_bundle_orig         = 2  # 2 original e-cig bundles (lo/hi cig)
    N_bundle_non_fda_flav = 2  # 2 non-FDA flavored e-cig bundles (lo/hi cig)
    N_bundle_fda_flav     = 2  # 2 FDA flavored e-cig bundles (lo/hi cig)
    N_bundle = N_bundle_orig + N_bundle_non_fda_flav + N_bundle_fda_flav

    # Initialize consumption vectors (j = 1 is outside option with zero consumption)
    c_cig  = zeros(Float64, N_J)
    c_ecig = zeros(Float64, N_J)

    # Fill vector in order: outside, cig, orig_ecig, non_fda_flav_ecig, fda_flav_ecig,
    # bundle_orig (2), bundle_non_fda_flav (2), bundle_fda_flav (2)
    idx = 2

    # Cigarette consumption
    for consumption in cig
        c_cig[idx] = consumption
        idx += 1
    end

    # Original e-cig alternatives
    for consumption in orig_ecig
        c_ecig[idx] = consumption
        idx += 1
    end

    # Non-FDA flavored e-cig alternatives
    for consumption in non_fda_flav_ecig
        c_ecig[idx] = consumption
        idx += 1
    end

    # FDA flavored e-cig alternatives
    for consumption in fda_flav_ecig
        c_ecig[idx] = consumption
        idx += 1
    end

    # Bundle consumption (6 bundles total)
    for bundle_name in bundle_names
        c_cig[idx]  = get_consumption_value(bundle_name * "_cig")
        c_ecig[idx] = get_consumption_value(bundle_name * "_ecig")
        idx += 1
    end

    # Compute bundle interaction BEFORE standardizing individual consumption
    # c_bundle[j] = c_cig[j] × c_ecig[j] (only non-zero for bundle alternatives)
    c_bundle_raw = c_cig .* c_ecig
    c_bundle_max = maximum(c_bundle_raw)

    # Standardize by maximum (store raw max for rescaling estimates later)
    # Each variable is standardized by its own max to keep coefficients reasonably scaled
    c_cig_max  = maximum(c_cig)
    c_ecig_max = maximum(c_ecig)
    c_cig  ./= c_cig_max
    c_ecig ./= c_ecig_max

    # Standardize bundle INTERACTION by its own max
    # This keeps α_TE at a reasonable magnitude since bundles don't have max consumption
    # of both products simultaneously
    c_bundle = c_bundle_raw ./ c_bundle_max

    return N_cig, N_orig_ecig, N_non_fda_flav_ecig, N_fda_flav_ecig, N_bundle, c_cig, c_ecig, c_bundle, c_cig_max, c_ecig_max, c_bundle_max
end


"""
Get nicotine vector indexed by alternative j ∈ {1, ..., N_J}
For bundle alternatives, total nicotine is the sum of the cigarette and e-cigarette components.

Nicotine is STANDARDIZED by dividing by the maximum value. This keeps the addiction
dynamics (ã' = (1-ψ)ã + ψ·n[j]) at reasonable magnitudes for numerical stability.

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

    # Bundle nicotine components (6 bundles: 2 orig + 2 non-FDA flav + 2 FDA flav, each with cig and ecig components)
    # Note: nicotine columns have _cig_nic and _ecig_nic suffixes
    # Original e-cig bundles
    bundle_orig_lo_cig_nic  = df.nicotine[df.alternative .== "bundle_orig_lo_cig_nic"]
    bundle_orig_lo_ecig_nic = df.nicotine[df.alternative .== "bundle_orig_lo_ecig_nic"]
    bundle_orig_hi_cig_nic  = df.nicotine[df.alternative .== "bundle_orig_hi_cig_nic"]
    bundle_orig_hi_ecig_nic = df.nicotine[df.alternative .== "bundle_orig_hi_ecig_nic"]
    # Non-FDA flavored e-cig bundles
    bundle_non_fda_flav_lo_cig_nic  = df.nicotine[df.alternative .== "bundle_non_fda_flav_lo_cig_nic"]
    bundle_non_fda_flav_lo_ecig_nic = df.nicotine[df.alternative .== "bundle_non_fda_flav_lo_ecig_nic"]
    bundle_non_fda_flav_hi_cig_nic  = df.nicotine[df.alternative .== "bundle_non_fda_flav_hi_cig_nic"]
    bundle_non_fda_flav_hi_ecig_nic = df.nicotine[df.alternative .== "bundle_non_fda_flav_hi_ecig_nic"]
    # FDA flavored e-cig bundles
    bundle_fda_flav_lo_cig_nic  = df.nicotine[df.alternative .== "bundle_fda_flav_lo_cig_nic"]
    bundle_fda_flav_lo_ecig_nic = df.nicotine[df.alternative .== "bundle_fda_flav_lo_ecig_nic"]
    bundle_fda_flav_hi_cig_nic  = df.nicotine[df.alternative .== "bundle_fda_flav_hi_cig_nic"]
    bundle_fda_flav_hi_ecig_nic = df.nicotine[df.alternative .== "bundle_fda_flav_hi_ecig_nic"]

    # Bundle pairs: (cig_nicotine, ecig_nicotine) for each bundle alternative
    # Order: 2 original, then 2 non-FDA flavored, then 2 FDA flavored
    # Extract scalars with only() since each filter should match exactly one row
    bundle_pairs = (
        (only(bundle_orig_lo_cig_nic), only(bundle_orig_lo_ecig_nic)),
        (only(bundle_orig_hi_cig_nic), only(bundle_orig_hi_ecig_nic)),
        (only(bundle_non_fda_flav_lo_cig_nic), only(bundle_non_fda_flav_lo_ecig_nic)),
        (only(bundle_non_fda_flav_hi_cig_nic), only(bundle_non_fda_flav_hi_ecig_nic)),
        (only(bundle_fda_flav_lo_cig_nic), only(bundle_fda_flav_lo_ecig_nic)),
        (only(bundle_fda_flav_hi_cig_nic), only(bundle_fda_flav_hi_ecig_nic)),
    )

    # Fill vector in order: outside, cig, orig_ecig, non_fda_flav_ecig, fda_flav_ecig,
    # bundle_orig (2), bundle_non_fda_flav (2), bundle_fda_flav (2)
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

    # Number of bundle alternatives (2 orig + 2 non-FDA flav + 2 FDA flav = 6 total)
    N_bundle_orig         = 2
    N_bundle_non_fda_flav = 2
    N_bundle_fda_flav     = 2

    # Initialize category index vector (j = 1 is outside option with cat = 0)
    cat_idx = zeros(Int, N_J)

    # Fill vector in order: outside, cig, orig_ecig, non_fda_flav_ecig, fda_flav_ecig,
    # bundle_orig (2), bundle_non_fda_flav (2), bundle_fda_flav (2)
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

    # Bundle with original e-cig (cat = 5) - 2 alternatives
    for _ in 1:N_bundle_orig
        cat_idx[idx] = 5
        idx += 1
    end

    # Bundle with non-FDA flavored e-cig (cat = 6) - 2 alternatives
    for _ in 1:N_bundle_non_fda_flav
        cat_idx[idx] = 6
        idx += 1
    end

    # Bundle with FDA flavored e-cig (cat = 7) - 2 alternatives
    for _ in 1:N_bundle_fda_flav
        cat_idx[idx] = 7
        idx += 1
    end

    return cat_idx
end


"""
Get non-FDA flavored indicator for each alternative j = 1, ..., N_J.
Non-FDA flavored alternatives are: non-FDA flavored e-cigarette bins (cat = 3)
and the cig + non-FDA flavored ecig bundle (cat = 6).

Returns:
- is_non_fda_flavored: Vector where true indicates a non-FDA flavored alternative
"""
function get_non_fda_flavored_indicator(
    cat_idx::AbstractVector{<:Integer}
)

    # Non-FDA flavored categories are 3 (non-FDA flav ecig) and 6 (bundle with non-FDA flav ecig)
    return (cat_idx .== 3) .| (cat_idx .== 6)
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

NOTE: This function loads the BINARY TYA indicator from Teen_Young_Adult.csv.
For the 4-state TYA used in β estimation, use get_tya_states() instead.

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
Get 4-state TYA classification for each observation.

Loads pre-computed TYA states from 05_TYA_State_Transitions.R:
  State 1: No TYA, stable (oldest child <= 10 or no children)
  State 2: No TYA, approaching (oldest child is 11-12)
  State 3: TYA present, stable (youngest TYA member <= 23)
  State 4: TYA present, ending soon (youngest TYA member >= 24)

Returns:
- tya_state: Vector{Int} of state indices (1-4) for each observation
"""
function get_tya_states(;
    file_name::AbstractString = "./TYA_States.csv"
)

    df = CSV.read(file_name, DataFrame)
    return Vector{Int}(df.tya_state)
end


"""
Get 4×4 TYA transition probability matrix.

Loads pre-computed monthly transition probabilities from
05_TYA_State_Transitions.R. Π[s, s'] = P(TYA state next month = s' | current = s).

Returns:
- Π: 4×4 row-stochastic transition matrix
"""
function get_tya_transitions(;
    file_name::AbstractString = "./TYA_Transition_Matrix.csv"
)

    df = CSV.read(file_name, DataFrame)

    Π = zeros(Float64, 4, 4)
    for row in eachrow(df)
        Π[row.from, row.to] = row.prob
    end

    return Π
end


#############################
# 6. Data Loading: Prices
#############################

# --- Fixed Parameters ---

"""
Set fixed parameters

In the Beta variant, β (present bias) is ESTIMATED as the 14th
parameter rather than being fixed. So this function only returns (ψ, δ).

Returns:
- Addiction decay rate ψ (fixed from reduced-form AR(1) estimate)
- Monthly discount factor δ
"""
function get_fixed_parameters()

    # Addiction decay rate (fixed from reduced-form AR(1) estimate)
    ψ = 0.6784

    # Monthly discount factor
    δ = 0.99

    return ψ, δ
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

Reads Price_Ratios.csv (generated by 01_Data_Prep.R) which contains:
  alternative, median_price, overall_median, ratio

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
    c_cig::AbstractVector{<:Real},
    c_ecig::AbstractVector{<:Real};
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

    # Cig bin names (must match 01_Data_Prep.R ordering)
    cig_names = ["cig_1", "cig_2", "cig_3to4", "cig_5to9", "cig_10", "cig_11to19",
                 "cig_20", "cig_21to29", "cig_30", "cig_31to39", "cig_40", "cig_41plus"]

    # Ecig bin names (pooled orig/flav — single ecig price state)
    ecig_names = ["ecig_0to5", "ecig_5to10", "ecig_10to15", "ecig_15to20",
                  "ecig_20to30", "ecig_30to50", "ecig_50plus"]

    # Fill ratio vectors following the alternative ordering:
    # j=1: outside option (ratios stay 1.0)
    idx = 2

    # Cigarette bins
    for name in cig_names
        ratio_cig[idx] = ratio_lookup[name]
        idx += 1
    end

    # Original e-cig bins (use pooled ecig ratios)
    for name in ecig_names
        ratio_ecig[idx] = ratio_lookup[name]
        idx += 1
    end

    # Non-FDA flavored e-cig bins (same pooled ecig ratios)
    for name in ecig_names
        ratio_ecig[idx] = ratio_lookup[name]
        idx += 1
    end

    # FDA flavored e-cig bins (same pooled ecig ratios)
    for name in ecig_names
        ratio_ecig[idx] = ratio_lookup[name]
        idx += 1
    end

    # Standalone bin index ranges (for matching bundle consumption to closest bin)
    cig_range  = 2:(1 + N_cig)                           # cigarette bins
    ecig_range = (2 + N_cig):(1 + N_cig + N_orig_ecig)   # orig ecig bins (same ratios as flav)

    # Bundles — map each bundle's consumption to the closest standalone bin's ratio
    for j in idx:N_J
        # Cig component: find standalone cig bin with closest consumption
        if c_cig[j] > 0.0
            best_cig = argmin(abs(c_cig[k] - c_cig[j]) for k in cig_range)
            ratio_cig[j] = ratio_cig[cig_range[best_cig]]
        end

        # Ecig component: find closest ecig bin (ratios are pooled across orig/flav)
        if c_ecig[j] > 0.0
            best_ecig = argmin(abs(c_ecig[k] - c_ecig[j]) for k in ecig_range)
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
  E[p, j] = p_cig(p) × ratio_cig(j) × c_cig(j) + p_ecig(p) × ratio_ecig(j) × c_ecig(j)

Price ratios capture quantity discounts: small-quantity bins have ratio > 1 (more expensive
per unit) and large-quantity bins have ratio < 1 (cheaper per unit).

This correctly handles all cases:
- Outside option: 0 (both consumption vectors are zero)
- Cig-only: p_cig × ratio_cig × c_cig (c_ecig is zero)
- Ecig-only: p_ecig × ratio_ecig × c_ecig (c_cig is zero)
- Bundles: p_cig × ratio_cig × c_cig + p_ecig × ratio_ecig × c_ecig

Expenditure is computed using RAW consumption (unstandardized) so that E is in
actual dollars, then STANDARDIZED by dividing by the maximum expenditure.

Returns:
- E: N_Pcomb × N_J matrix of standardized expenditures
- E_max: Raw maximum expenditure (for rescaling parameter estimates)
"""
function get_expenditures(
    N_J::Integer,
    N_Pcomb::Integer,
    c_cig::AbstractVector{<:Real},
    c_ecig::AbstractVector{<:Real},
    c_cig_max::Real,
    c_ecig_max::Real,
    Pcomb::AbstractMatrix{<:Real},
    ratio_cig::AbstractVector{<:Real},
    ratio_ecig::AbstractVector{<:Real}
)

    # Reconstruct raw consumption from standardized values
    c_cig_raw  = c_cig .* c_cig_max
    c_ecig_raw = c_ecig .* c_ecig_max

    # Initialize expenditure matrix
    E = zeros(Float64, N_Pcomb, N_J)

    # Fill expenditure matrix using raw consumption and price ratios (actual dollars)
    for p in 1:N_Pcomb
        p_cig  = Pcomb[p, 1]
        p_ecig = Pcomb[p, 2]
        for j in 1:N_J
            E[p, j] = p_cig * ratio_cig[j] * c_cig_raw[j] + p_ecig * ratio_ecig[j] * c_ecig_raw[j]
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

    # Loop over combined price states (m) and Halton draws (r)
    for m in 1:M
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

Prices.csv contains bin-specific prices (capturing quantity discounts).
For the dynamic model's price state, I compute a representative price per
category by taking the median across bins.

Combined grid ordering follows 04_State_Transitions.R:
  combined_idx = (cig_idx - 1) * N_P + ecig_idx
where cig_idx varies slowly and ecig_idx varies fast.

Returns:
- p_state: Vector{Int} of combined price grid indices for each observation
- p_continuous: N × 2 matrix of representative (cig, ecig) prices for likelihood interpolation
"""
function map_prices_to_grid(
    N_P::Integer,
    P::AbstractMatrix{<:Real},
    Pcomb::AbstractMatrix{<:Real};
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

    return p_state, p_continuous
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
    # Note: `a` here is already ̃a, so no additional ψ multiplication on the decay term.
    return (1 - ψ) * a + ψ * n
end


"""
Pre-compute flow utility for all (tya, alternative, addiction, price) states.

The flow utility for alternative j given addiction state a, price state p, and
TYA state is:

  u(j,a,p,tya) = α_T·c_cig[j] + α_E·c_ecig[j] + α_TE·c_cig[j]·c_ecig[j]
               + 𝟙[non-FDA flavored]·(λ_1 + λ_2·𝟙[tya])
               + 𝟙[FDA flavored]·(λ_3 + λ_4·𝟙[tya])
               + μ·a·n[j] + γ·a
               + ω·E[p,j]
               + ξ_k

The reinforcement term μ·a·n[j] captures the interaction between addiction
stock and current nicotine intake — higher addiction increases the marginal
utility of nicotine-delivering alternatives.

For the outside option (j = 1), all consumption, expenditure, fixed effect,
flavored, and reinforcement terms are zero, so u = γ·a. No special case needed.

Fixed effects: ξ_T for k = 1, ξ_E for k ∈ {2, 3, 4}, ξ_TE for k ∈ {5, 6, 7}.

4-state TYA mapping for the flavor × TYA interaction:
  States 1, 2 (no TYA present) → tya indicator = 0
  States 3, 4 (TYA present)    → tya indicator = 1

Returns:
- U: 4D array of flow utilities, U[tya_idx, j, a_idx, p_idx]
      Dimensions: 4 × N_J × N_A × N_Pcomb
"""
function get_flow_utility(
    θ::AbstractVector{<:Real},
    N_J::Integer,
    N_A::Integer,
    N_Pcomb::Integer,
    A::AbstractVector{<:Real},
    c_cig::AbstractVector{<:Real},
    c_ecig::AbstractVector{<:Real},
    c_bundle::AbstractVector{<:Real},
    n::AbstractVector{<:Real},
    is_non_fda_flavored::AbstractVector{Bool},
    is_fda_flavored::AbstractVector{Bool},
    cat_idx::AbstractVector{<:Integer},
    E::AbstractMatrix{<:Real}
)

    # Unpack structural parameters (first 13 elements; β is the 14th and not used here)
    α_T, α_E, α_TE, λ_1, λ_2, λ_3, λ_4, μ, γ, ω, ξ_T, ξ_E, ξ_TE = θ

    # Number of TYA states (expanded from 2 to 4)
    N_TYA = 4

    # Initialize flow utility array
    U = zeros(Float64, N_TYA, N_J, N_A, N_Pcomb)

    # Pre-compute fixed effect for each alternative
    ξ = zeros(Float64, N_J)
    for j in 1:N_J
        if cat_idx[j] == 1
            ξ[j] = ξ_T
        elseif cat_idx[j] in (2, 3, 4)
            ξ[j] = ξ_E
        elseif cat_idx[j] in (5, 6, 7)
            ξ[j] = ξ_TE
        end
    end

    # Fill flow utility array starting with price state
    for p_idx in 1:N_Pcomb

        # Loop over addiction states
        for a_idx in 1:N_A
            a = A[a_idx]

            # Loop over alternatives
            for j_idx in 1:N_J

                # Components that do not depend on TYA state
                u_base = (α_T * c_cig[j_idx] + α_E * c_ecig[j_idx] + α_TE * c_bundle[j_idx]
                         + μ * a * n[j_idx] + γ * a
                         + ω * E[p_idx, j_idx]
                         + ξ[j_idx])

                # Loop over 4 TYA states
                for tya_idx in 1:N_TYA

                    # TYA indicator: states 3,4 have TYA present (tya=1); states 1,2 do not (tya=0)
                    tya = (tya_idx >= 3) ? 1 : 0

                    # Flow utility including non-FDA and FDA flavor effects
                    U[tya_idx, j_idx, a_idx, p_idx] = u_base + is_non_fda_flavored[j_idx] * (λ_1 + λ_2 * tya) + is_fda_flavored[j_idx] * (λ_3 + λ_4 * tya)
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

            # Next-period addiction level
            a_prime = addiction_evolution(ψ, A[a_idx], n[j_idx])

            # Find upper bracket index via binary search
            hi_raw = searchsortedfirst(A, a_prime)
            lo = clamp(hi_raw - 1, 1, N_A)
            hi = clamp(hi_raw, 1, N_A)

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

        # If household code has not appeared as dictionary key yet
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

NOTE: Assumes observations are ordered chronologically within each household
(i.e., sorted by household_code then purchase_month), matching the order from
the R data prep.

Returns:
- a_state: Vector of addiction grid indices for each observation
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

    # Initialize addiction state index and continuous addiction level for each observation
    a_state      = Vector{Int}(undef, N)
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

        # Map to nearest grid index
        a_state[i] = argmin(abs.(A .- a))

        # Evolve addiction based on chosen alternative's nicotine
        a_prime = addiction_evolution(ψ, a, n[y[i]])

        # Clamp to grid bounds
        a_prime = clamp(a_prime, A[1], A[end])

        # Store for next period
        a_current[hh] = a_prime
    end

    return a_state, a_continuous
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

This is used after VFI convergence so returned choice arrays are exactly
consistent with the returned V_now.
"""
function recompute_choice_values_4tya!(
    V_out::Array{Float64, 4},
    V_now::Array{Float64, 3},
    U::Array{Float64, 4},
    discount_scale::Float64,
    Π_tya::Matrix{Float64},
    N_J::Integer,
    N_A::Integer,
    N_P::Integer,
    N_Pcomb::Integer,
    inv_R::Float64,
    p_cig_lo::Matrix{Int},
    p_cig_hi::Matrix{Int},
    p_cig_w::Matrix{Float64},
    p_ecig_lo::Matrix{Int},
    p_ecig_hi::Matrix{Int},
    p_ecig_w::Matrix{Float64},
    a_lower::Matrix{Int},
    a_upper::Matrix{Int},
    a_weight::Matrix{Float64}
)

    R = size(p_cig_lo, 2)
    N_TYA = 4

    Threads.@threads for p_idx in 1:N_Pcomb
        @inbounds for a_idx in 1:N_A
            for j_idx in 1:N_J
                lo_a = a_lower[j_idx, a_idx]
                hi_a = a_upper[j_idx, a_idx]
                w_a  = a_weight[j_idx, a_idx]

                EV_lo_1 = 0.0; EV_hi_1 = 0.0
                EV_lo_2 = 0.0; EV_hi_2 = 0.0
                EV_lo_3 = 0.0; EV_hi_3 = 0.0
                EV_lo_4 = 0.0; EV_hi_4 = 0.0

                for r in 1:R
                    c_lo = p_cig_lo[p_idx, r]
                    c_hi = p_cig_hi[p_idx, r]
                    w_c  = p_cig_w[p_idx, r]
                    e_lo = p_ecig_lo[p_idx, r]
                    e_hi = p_ecig_hi[p_idx, r]
                    w_e  = p_ecig_w[p_idx, r]

                    p_ll = (c_lo - 1) * N_P + e_lo
                    p_lh = (c_lo - 1) * N_P + e_hi
                    p_hl = (c_hi - 1) * N_P + e_lo
                    p_hh = (c_hi - 1) * N_P + e_hi

                    w_ll = (1 - w_c) * (1 - w_e)
                    w_lh = (1 - w_c) * w_e
                    w_hl = w_c * (1 - w_e)
                    w_hh = w_c * w_e

                    EV_lo_1 += w_ll * V_now[1, lo_a, p_ll] +
                               w_lh * V_now[1, lo_a, p_lh] +
                               w_hl * V_now[1, lo_a, p_hl] +
                               w_hh * V_now[1, lo_a, p_hh]

                    EV_hi_1 += w_ll * V_now[1, hi_a, p_ll] +
                               w_lh * V_now[1, hi_a, p_lh] +
                               w_hl * V_now[1, hi_a, p_hl] +
                               w_hh * V_now[1, hi_a, p_hh]

                    EV_lo_2 += w_ll * V_now[2, lo_a, p_ll] +
                               w_lh * V_now[2, lo_a, p_lh] +
                               w_hl * V_now[2, lo_a, p_hl] +
                               w_hh * V_now[2, lo_a, p_hh]

                    EV_hi_2 += w_ll * V_now[2, hi_a, p_ll] +
                               w_lh * V_now[2, hi_a, p_lh] +
                               w_hl * V_now[2, hi_a, p_hl] +
                               w_hh * V_now[2, hi_a, p_hh]

                    EV_lo_3 += w_ll * V_now[3, lo_a, p_ll] +
                               w_lh * V_now[3, lo_a, p_lh] +
                               w_hl * V_now[3, lo_a, p_hl] +
                               w_hh * V_now[3, lo_a, p_hh]

                    EV_hi_3 += w_ll * V_now[3, hi_a, p_ll] +
                               w_lh * V_now[3, hi_a, p_lh] +
                               w_hl * V_now[3, hi_a, p_hl] +
                               w_hh * V_now[3, hi_a, p_hh]

                    EV_lo_4 += w_ll * V_now[4, lo_a, p_ll] +
                               w_lh * V_now[4, lo_a, p_lh] +
                               w_hl * V_now[4, lo_a, p_hl] +
                               w_hh * V_now[4, lo_a, p_hh]

                    EV_hi_4 += w_ll * V_now[4, hi_a, p_ll] +
                               w_lh * V_now[4, hi_a, p_lh] +
                               w_hl * V_now[4, hi_a, p_hl] +
                               w_hh * V_now[4, hi_a, p_hh]
                end

                EV_next_1 = (1 - w_a) * (EV_lo_1 * inv_R) + w_a * (EV_hi_1 * inv_R)
                EV_next_2 = (1 - w_a) * (EV_lo_2 * inv_R) + w_a * (EV_hi_2 * inv_R)
                EV_next_3 = (1 - w_a) * (EV_lo_3 * inv_R) + w_a * (EV_hi_3 * inv_R)
                EV_next_4 = (1 - w_a) * (EV_lo_4 * inv_R) + w_a * (EV_hi_4 * inv_R)

                for tya_idx in 1:N_TYA
                    EV = Π_tya[tya_idx, 1] * EV_next_1 +
                         Π_tya[tya_idx, 2] * EV_next_2 +
                         Π_tya[tya_idx, 3] * EV_next_3 +
                         Π_tya[tya_idx, 4] * EV_next_4
                    V_out[tya_idx, j_idx, a_idx, p_idx] = U[tya_idx, j_idx, a_idx, p_idx] + discount_scale * EV
                end
            end
        end
    end
end


"""
Solve the value function via value function iteration (VFI) for the exponential or naive
quasi hyperbolic discounting case. 

To show how this works, initially I have V⁰_now = 0. This implies V¹_next = U since
E[V⁰_now] = 0. Then, I update so V¹_now ⟵ V¹_next, meaning V¹_now = U. Denoting EV as the
expectation of U over the states, which is just the expectation of the current V_now over all states,
I have V²_next = U + δ EV = U + δ E[V¹_now] = U + δ EV. Then, I update V²_now ⟵ V²_next so
V²_now = U + δ EV. Again, EV is the expectation over the current V_now for all states. So,
V³_next = U + δ[U + δ EV] = U + δ U + δ² EV. Then, I update so V³_now ⟵ V³_next. Note, this
uses Jacobi iteration so I can parallelize over prices.

The infinite horizon assumption is standard for addiction models because there's no natural end date
as people don't know when they'll stop making tobacco purchase decisions.

δ < 1 ensures the infinite sum converges to a finite value.

CHANGES FROM 01_Functions.jl:
  - N_TYA = 4 (expanded from 2)
  - Takes Π_tya (4×4 TYA transition matrix) as a new parameter
  - Expected continuation value integrates over TYA transitions:
      EV[tya, j, a, p] = Σ_{tya'=1}^{4} Π[tya, tya'] × E_price[V(tya', a', p')]
  - This is exact (no simulation needed) since TYA transitions are discrete

VFI uses δ only (exponential discounting) until the sup-norm difference
between successive value function iterates falls below tolerance ε:

  V_choice[tya, j, a, p] = u(tya, j, a, p) + δ · E[V(tya', a', p') | tya, a, p, j]
  V[tya, a, p] = log( Σ_j exp(V_choice[tya, j, a, p]) )

After convergence, computes V_decision for quasi-hyperbolic (β-δ) discounting:

  V_decision = (1 - β) · U + β · V_choice = U + βδ · E[V | a, p, j]

When β = 1 (standard exponential), V_decision = V_choice. When β < 1
(present bias), the agent over-discounts the future relative to the present.
VFI is solved with δ only because a naive agent believes their future self
will discount exponentially. 

The expected continuation value E[V(tya', a', p') | tya, a, p, j] is computed by:
  1. Pre-computed addiction transition brackets for ã' = (1-ψ)ã + ψ·n[j]
  2. Bilinear price interpolation at each Halton draw's predicted prices
  3. Averaging over R Halton draws
  4. Linear interpolation across the two addiction bracket points
  5. Weighting by TYA transition probabilities Π[tya, tya']

When running on the HPC, need to make sure the thread count is the same as the number of cores I request

Returns:
- V:          Converged value function (δ only), V[tya_idx, a_idx, p_idx]
- V_decision: Decision-utility value function (βδ), V_decision[tya_idx, j, a_idx, p_idx]
- n_iter:     Number of iterations to convergence
- converged:  Whether VFI converged within max_iter
"""
function solve_vfi(
    N_J::Integer,
    N_A::Integer,
    N_P::Integer,
    N_Pcomb::Integer,
    β::Real,
    δ::Real,
    U::Array{Float64, 4},
    a_lower::Matrix{Int},
    a_upper::Matrix{Int},
    a_weight::Matrix{Float64},
    p_cig_lo::Matrix{Int},
    p_cig_hi::Matrix{Int},
    p_cig_w::Matrix{Float64},
    p_ecig_lo::Matrix{Int},
    p_ecig_hi::Matrix{Int},
    p_ecig_w::Matrix{Float64},
    Π_tya::Matrix{Float64};
    ε::Real = 1e-4,
    max_iter::Integer = 3000,
    verbose::Bool = true
)

    # Number of TYA states and Halton draws
    N_TYA = 4
    R     = size(p_cig_lo, 2)

    # Compute 1.0/R
    # I need this because I need to average the continuation values over the R Halton draws
    # and EV / R is more computationally expensive that EV * inv_R
    inv_R = 1.0 / R


    #############################
    # VFI
    #############################

    # Initialize value function arrays (start from zeros each time)
    V_now    = zeros(Float64, N_TYA, N_A, N_Pcomb)
    V_next   = zeros(Float64, N_TYA, N_A, N_Pcomb)
    V_choice = zeros(Float64, N_TYA, N_J, N_A, N_Pcomb)

    # Initialize number of iterations
    n_iter = 0

    # Initialize convergence
    converged = false

    # Previous iterations sup-norm across all states
    prev_diff = Inf

    # VFI timer
    t_vfi = time()
    for iter in 1:max_iter
        n_iter = iter

        # Compute V_choice and V_next for all states
        # Parallelize over price states (each p_idx writes to distinct memory)
        Threads.@threads for p_idx in 1:N_Pcomb

            # Inbounds just tells Julia that there is no need to check that A[i] actually exists
            @inbounds for a_idx in 1:N_A

                # Loop over alternatives
                for j_idx in 1:N_J

                    # Pre-computed addiction transition brackets for (j_idx, a_idx)
                    # When I consume alternative j while having addiction level a, I 
                    # get some continuous addiction a′. lo_a is the index of the grid immediately
                    # preceeding a′ and hi_a is the index of the grid immediately
                    # proceeding a′. w_a is the associated interpolation weight
                    # depending on which grid point a′ is closest to.
                    lo_a = a_lower[j_idx, a_idx]
                    hi_a = a_upper[j_idx, a_idx]
                    w_a  = a_weight[j_idx, a_idx]

                    # Accumulate expected continuation value over Halton draws
                    # Separate accumulators for each NEXT-PERIOD TYA state and addiction bracket.
                    # I first compute EV at all 4 next-period TYA states,
                    # then combine using the transition matrix.
                    # EV_lo_1 corresponds to the low addiction grid point and TYA state 1
                    # EV_hi_1 corresponds to the high addiction grid point and TYA state 1
                    # etc. for TYA states 2, 3, 4
                    # So, for each low and high addiction grid points and TYA state
                    # I will do a bilinear interpolation over the two price categories
                    # Then, I will combine EV_lo and EV_hi using a
                    # single interpolation over the addiction grid.
                    # Finally, I weight by TYA transition probabilities.
                    EV_lo_1 = 0.0; EV_hi_1 = 0.0
                    EV_lo_2 = 0.0; EV_hi_2 = 0.0
                    EV_lo_3 = 0.0; EV_hi_3 = 0.0
                    EV_lo_4 = 0.0; EV_hi_4 = 0.0

                    # Loop over Halton draws
                    for r in 1:R

                        # Pre-computed price transition brackets for (p_idx, r)
                        c_lo = p_cig_lo[p_idx, r]
                        c_hi = p_cig_hi[p_idx, r]
                        w_c  = p_cig_w[p_idx, r]
                        e_lo = p_ecig_lo[p_idx, r]
                        e_hi = p_ecig_hi[p_idx, r]
                        w_e  = p_ecig_w[p_idx, r]

                        # Combined price indices for bilinear interpolation
                        p_ll = (c_lo - 1) * N_P + e_lo
                        p_lh = (c_lo - 1) * N_P + e_hi
                        p_hl = (c_hi - 1) * N_P + e_lo
                        p_hh = (c_hi - 1) * N_P + e_hi

                        # Bilinear weights
                        w_ll = (1 - w_c) * (1 - w_e)
                        w_lh = (1 - w_c) * w_e
                        w_hl = w_c * (1 - w_e)
                        w_hh = w_c * w_e

                        # Bilinear price interpolation at each addiction bracket
                        # for all 4 next-period TYA states
                        EV_lo_1 += w_ll * V_now[1, lo_a, p_ll] +
                                   w_lh * V_now[1, lo_a, p_lh] +
                                   w_hl * V_now[1, lo_a, p_hl] +
                                   w_hh * V_now[1, lo_a, p_hh]

                        EV_hi_1 += w_ll * V_now[1, hi_a, p_ll] +
                                   w_lh * V_now[1, hi_a, p_lh] +
                                   w_hl * V_now[1, hi_a, p_hl] +
                                   w_hh * V_now[1, hi_a, p_hh]

                        EV_lo_2 += w_ll * V_now[2, lo_a, p_ll] +
                                   w_lh * V_now[2, lo_a, p_lh] +
                                   w_hl * V_now[2, lo_a, p_hl] +
                                   w_hh * V_now[2, lo_a, p_hh]

                        EV_hi_2 += w_ll * V_now[2, hi_a, p_ll] +
                                   w_lh * V_now[2, hi_a, p_lh] +
                                   w_hl * V_now[2, hi_a, p_hl] +
                                   w_hh * V_now[2, hi_a, p_hh]

                        EV_lo_3 += w_ll * V_now[3, lo_a, p_ll] +
                                   w_lh * V_now[3, lo_a, p_lh] +
                                   w_hl * V_now[3, lo_a, p_hl] +
                                   w_hh * V_now[3, lo_a, p_hh]

                        EV_hi_3 += w_ll * V_now[3, hi_a, p_ll] +
                                   w_lh * V_now[3, hi_a, p_lh] +
                                   w_hl * V_now[3, hi_a, p_hl] +
                                   w_hh * V_now[3, hi_a, p_hh]

                        EV_lo_4 += w_ll * V_now[4, lo_a, p_ll] +
                                   w_lh * V_now[4, lo_a, p_lh] +
                                   w_hl * V_now[4, lo_a, p_hl] +
                                   w_hh * V_now[4, lo_a, p_hh]

                        EV_hi_4 += w_ll * V_now[4, hi_a, p_ll] +
                                   w_lh * V_now[4, hi_a, p_lh] +
                                   w_hl * V_now[4, hi_a, p_hl] +
                                   w_hh * V_now[4, hi_a, p_hh]
                    end

                    # Average over Halton draws and interpolate over addiction
                    # for each next-period TYA state
                    EV_next_1 = (1 - w_a) * (EV_lo_1 * inv_R) + w_a * (EV_hi_1 * inv_R)
                    EV_next_2 = (1 - w_a) * (EV_lo_2 * inv_R) + w_a * (EV_hi_2 * inv_R)
                    EV_next_3 = (1 - w_a) * (EV_lo_3 * inv_R) + w_a * (EV_hi_3 * inv_R)
                    EV_next_4 = (1 - w_a) * (EV_lo_4 * inv_R) + w_a * (EV_hi_4 * inv_R)

                    # Weight by TYA transition probabilities for each current TYA state
                    # V_choice[tya, j, a, p] = U[tya, j, a, p] + δ * Σ_{tya'} Π[tya, tya'] * EV_next[tya']
                    for tya_idx in 1:N_TYA
                        EV = Π_tya[tya_idx, 1] * EV_next_1 +
                             Π_tya[tya_idx, 2] * EV_next_2 +
                             Π_tya[tya_idx, 3] * EV_next_3 +
                             Π_tya[tya_idx, 4] * EV_next_4
                        V_choice[tya_idx, j_idx, a_idx, p_idx] = U[tya_idx, j_idx, a_idx, p_idx] + δ * EV
                    end
                end

                # Aggregate over alternatives via log-sum-exp
                # @view tells Julia to not make a copy, but rather use the V_choice currently
                # in memory
                for tya_idx in 1:N_TYA
                    V_next[tya_idx, a_idx, p_idx] = logsumexp(@view V_choice[tya_idx, :, a_idx, p_idx])
                end
            end
        end

        # # Normalize V_next to prevent value function levels from growing unboundedly.
        # # Choice probabilities only depend on V differences, so the level is irrelevant.
        # # This normalization is applied BEFORE the convergence check so that I measure
        # # the sup-norm of normalized values, avoiding false non-convergence.
        # V_ref = V_next[1, 1, 1]
        # V_next .-= V_ref

        # Convergence check: sup norm across all states
        current_diff = maximum(abs.(V_next .- V_now))

        # Update V_now for next iteration
        V_now .= V_next

        # Log progress every 100 iterations (only when verbose)
        if verbose && (iter % 100 == 0 || iter == 1 || iter == 10)

            # Get elapsed time for the current vfi 
            elapsed = time() - t_vfi

            # Ratio gives the convergence rate. A stable ratio below 1.0
            # means it's contracting. The closer to 0, the faster it's converging. If it's near 1.0 or above,
            # convergence is stalling.
            ratio = prev_diff < Inf ? current_diff / prev_diff : NaN

            # Print and log vfi message 
            # sprintf stands for string print formatted. Essentially, just a formatted string. The @
            # transforms code at compile time before running, making things faster 
            # VFI iter can handle up to 9,999,999 without misaligned printing
            # Sup-norm prints as scientific notation like 0.00034 as 3.4e-04
            # Ratio prints up to four decimals
            # Elapsed prints up to one decimal with a literal "s" afterwards for seconds 
            log_msg(@sprintf("    VFI iter %6d | sup-norm = %.6e | ratio = %.4f | elapsed = %.1fs",
                iter, current_diff, ratio, elapsed))
        end

        # Update the sup-norm
        prev_diff = current_diff

        # Check convergence
        if current_diff < ε
            if verbose

                # Get elapsed time for the current vfi 
                elapsed = time() - t_vfi

                # Print and log vfi message
                log_msg(@sprintf("    VFI converged in %d iterations (sup-norm = %.6e, %.1fs)", iter, current_diff, elapsed))
            end

            # If converged, then break out of vfi loop 
            converged = true
            break
        end

        # Report non-convergence
        if iter == max_iter

            # Get elapsed time for the current vfi
            elapsed = time() - t_vfi

            # Print and log vfi message
            log_msg(@sprintf("VFI did not converge after %d iterations (sup-norm = %.6e, %.1fs)", max_iter, current_diff, elapsed))
        end
    end

    # Final policy recomputation at converged V_now.
    recompute_choice_values_4tya!(
        V_choice, V_now, U, Float64(δ), Π_tya,
        N_J, N_A, N_P, N_Pcomb, inv_R,
        p_cig_lo, p_cig_hi, p_cig_w,
        p_ecig_lo, p_ecig_hi, p_ecig_w,
        a_lower, a_upper, a_weight
    )

    # Compute V_decision for choice probabilities (quasi-hyperbolic discounting)
    if β == 1.0
        V_decision = V_choice
    else
        V_decision = (1 .- β) .* U .+ β .* V_choice
    end

    return V_now, V_decision, n_iter, converged
end


"""
Solve the value function via value function iteration (VFI) for the sophisticated
quasi-hyperbolic discounting case with 4-state TYA transitions.

# Differences from 01_Functions.jl solve_vfi_sophisticated (baseline)
  - N_TYA = 4 instead of 2
  - Takes Π_tya (4×4 row-stochastic TYA transition matrix) as a new parameter
  - Expected continuation value integrates over both price AND TYA transitions:
      EV[tya, a', p] = Σ_{tya'} Π_tya[tya, tya'] · (1/R) · Σ_r V[tya', a', p'_r]

This is the key difference enabling identification of β. In the baseline (2-state
TYA, no transition matrix), TYA is absorbing — a household in TYA state 1 stays
in state 1 forever. Here, TYA states evolve stochastically (from the model's
perspective — see note below), so anticipated TYA transitions affect continuation
values differently under β < 1 vs β = 1.

# Why TYA transitions are stochastic in the model
Aging is deterministic at the household level (a 25-year-old will turn 26). But
the model does not track exact child ages — only the 4-state TYA classification.
Within each state, households have different exact ages, so the transition
probability Π_tya[s, s'] reflects the empirical distribution of exact ages within
state s. For example, state 4 ("ending soon", ages 24-25) has ~50% probability
of transitioning to state 1 (the 25-year-olds who age out) and ~50% of staying
in state 4 (the 24-year-olds who remain). This is a standard dimensionality
reduction: we approximate deterministic individual-level aging with average
transition rates within coarse states.

# 4-state TYA classification
  State 1: No TYA member, stable (oldest child ≤ 10 or no children)
  State 2: No TYA member, approaching (oldest child age 11-12)
  State 3: TYA member present, stable (youngest TYA member age ≤ 23)
  State 4: TYA member present, ending soon (youngest TYA member age ≥ 24)

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
    in state (tya, a, p) — precomputed outside VFI. States 1,2 share tya=0
    (no TYA member) and states 3,4 share tya=1 (TYA member present) for the
    flavor interaction term λ_2·1[flavored]·1[tya].
  - a'(j, a) is the next-period addiction state from choosing j at addiction a,
    given by ã' = (1-ψ)·ã + ψ·n[j], interpolated onto the addiction grid
  - EV integrates over BOTH stochastic price transitions AND discrete TYA
    transitions (this is the key difference from the baseline)

Note: V_d and V_e share the *same* EV term. The only difference is the discount
factor applied to it (β·δ vs δ). This is because the continuation value V is
what *will actually happen* — the sophisticated agent correctly predicts this.

# Expected Continuation Value (with TYA integration)
The EV computation has two layers of integration:

  1. For each next-period TYA state tya' = 1,...,4, compute the expected value
     over price transitions (Halton draws):
       E_price[V(tya', a', p)] = (1/R) · Σ_r V[tya', a', p'_r]

  2. Then integrate over TYA transitions using the Markov transition matrix:
       EV[tya, a', p] = Σ_{tya'} Π_tya[tya, tya'] · E_price[V(tya', a', p)]

This two-layer structure means the continuation value depends on the CURRENT
TYA state through Π_tya: a household in state 4 (ending soon) weights the
no-TYA future (state 1) more heavily than a household in state 3 (stable).

# How β is identified
The flavor interaction λ_2·1[flavored]·1[tya] makes flavored e-cig utility
higher in TYA states 3,4 (TYA present) than states 1,2 (no TYA). A household
in state 4 (ending soon) anticipates losing this flavor bonus. Under β = 1,
they smoothly reduce flavored e-cig consumption well in advance. Under β < 1,
they discount future losses more steeply, so adjustment is compressed closer
to the actual transition — consistent with the reduced-form finding that the
behavioral response concentrates at ages 23-25.

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

# Iteration Scheme
Uses Jacobi-style iteration: the entire V_next array is computed from V_now,
then V_now is replaced with V_next. This is safe for multithreading because
all reads come from V_now (read-only) and all writes go to V_next/V_d/V_e
at disjoint (p_idx) slices.

# Post-Convergence Recomputation
After convergence, V_d is recomputed one final time from the converged V_now.
This ensures the returned V_decision is exactly consistent with the fixed point,
rather than being one iteration stale (since V_d was last computed *before*
the final V_now update).

# Arguments
- N_J:      Number of product alternatives (31)
- N_A:      Number of addiction grid points (20)
- N_P:      Number of price grid points per category (10)
- N_Pcomb:  Number of combined price states, N_P² (100)
- β:        Present bias parameter, β ∈ (0, 1]. β = 1 is standard exponential.
            This is estimated as the 14th parameter (θ_vec[14]).
- δ:        Monthly exponential discount factor (0.99)
- U:        Pre-computed flow utility array, U[tya_idx, j, a_idx, p_idx].
            Dimension 1 has 4 TYA states; states 1,2 share tya=0 flow utility,
            states 3,4 share tya=1 flow utility (only the flavor interaction differs).
- a_lower:  Lower addiction bracket indices, a_lower[j, a_idx]
- a_upper:  Upper addiction bracket indices, a_upper[j, a_idx]
- a_weight: Addiction interpolation weights, a_weight[j, a_idx]
- p_cig_lo/hi/w:  Cig price transition brackets/weights, [p_idx, r]
- p_ecig_lo/hi/w: Ecig price transition brackets/weights, [p_idx, r]
- Π_tya:    4×4 row-stochastic TYA transition matrix. Π_tya[s, s'] = P(TYA' = s' | TYA = s).
            Estimated empirically from observed month-to-month state transitions.
- ε:        Convergence tolerance for sup-norm (default 1e-4)
- max_iter: Maximum VFI iterations (default 3000)
- verbose:  Whether to print convergence diagnostics

# Returns
- V:          Converged ex-ante value function, V[tya_idx, a_idx, p_idx]
- V_decision: Decision-utility choice values (βδ discounting), V_decision[tya_idx, j, a_idx, p_idx]
- n_iter:     Number of iterations to convergence
- converged:  Whether VFI converged within max_iter
"""
function solve_vfi_sophisticated(
    N_J::Integer,
    N_A::Integer,
    N_P::Integer,
    N_Pcomb::Integer,
    β::Real,
    δ::Real,
    U::Array{Float64, 4},
    a_lower::Matrix{Int},
    a_upper::Matrix{Int},
    a_weight::Matrix{Float64},
    p_cig_lo::Matrix{Int},
    p_cig_hi::Matrix{Int},
    p_cig_w::Matrix{Float64},
    p_ecig_lo::Matrix{Int},
    p_ecig_hi::Matrix{Int},
    p_ecig_w::Matrix{Float64},
    Π_tya::Matrix{Float64};
    ε::Real = 1e-4,
    max_iter::Integer = 3000,
    verbose::Bool = true
)

    # Beta variant has 4 TYA states (vs 2 in baseline):
    #   1 = no TYA, stable;  2 = no TYA, approaching
    #   3 = TYA present, stable;  4 = TYA present, ending soon
    N_TYA = 4

    # Number of Halton draws for quasi-Monte Carlo integration over price shocks.
    # R draws are pre-simulated from the AR(1) price process with correlated shocks.
    R = size(p_cig_lo, 2)

    # Pre-compute 1/R so we multiply instead of divide inside the hot loop
    inv_R = 1.0 / R

    # Pre-compute β·δ — this is the discount factor applied in decision utility.
    # When β = 1, βδ = δ and V_d = V_e (sophisticated collapses to naive).
    βδ = β * δ

    # Initialize all arrays to zero (fresh each solve — no warm-starting).
    # V_now:  current iterate of the ex-ante value function V[tya, a, p]
    # V_next: next iterate of V, computed entirely from V_now each iteration
    # V_d:    decision utility V_d[tya, j, a, p] = U + βδ·EV (drives choice probs)
    # V_e:    experienced utility V_e[tya, j, a, p] = U + δ·EV (actual payoffs)
    V_now  = zeros(Float64, N_TYA, N_A, N_Pcomb)
    V_next = zeros(Float64, N_TYA, N_A, N_Pcomb)
    V_d    = zeros(Float64, N_TYA, N_J, N_A, N_Pcomb)
    V_e    = zeros(Float64, N_TYA, N_J, N_A, N_Pcomb)

    n_iter = 0
    converged = false
    prev_diff = Inf

    # ==================== MAIN VFI LOOP ====================
    # Jacobi iteration: compute V_next entirely from V_now, then swap.
    # Safe for multithreading because V_now is read-only within each iteration.
    t_vfi = time()
    for iter in 1:max_iter
        n_iter = iter

        # Parallelize over the N_Pcomb = 100 combined price states.
        # Each thread writes to disjoint p_idx slices of V_next, V_d, V_e.
        Threads.@threads for p_idx in 1:N_Pcomb
            @inbounds for a_idx in 1:N_A
                for j_idx in 1:N_J

                    # ---- Step 1: Get addiction transition brackets for (j, a) ----
                    # When the agent in addiction state a_idx chooses alternative j_idx,
                    # next-period addiction is ã' = (1-ψ)·ã + ψ·n[j]. This generally
                    # falls between two grid points. lo_a and hi_a are the bracketing
                    # grid indices, w_a is the weight on hi_a (linear interpolation).
                    lo_a = a_lower[j_idx, a_idx]
                    hi_a = a_upper[j_idx, a_idx]
                    w_a  = a_weight[j_idx, a_idx]

                    # ---- Step 2: Accumulate EV over Halton draws at each addiction bracket ----
                    # We accumulate separately for each of the 4 next-period TYA states.
                    # This is because we need E_price[V(tya', a', p)] for each tya'
                    # BEFORE integrating over TYA transitions with Π_tya.
                    # EV_lo_k = Σ_r V_now[k, lo_a, p'_r]  (value at lower addiction bracket)
                    # EV_hi_k = Σ_r V_now[k, hi_a, p'_r]  (value at upper addiction bracket)
                    EV_lo_1 = 0.0; EV_hi_1 = 0.0
                    EV_lo_2 = 0.0; EV_hi_2 = 0.0
                    EV_lo_3 = 0.0; EV_hi_3 = 0.0
                    EV_lo_4 = 0.0; EV_hi_4 = 0.0

                    for r in 1:R
                        # ---- Step 2a: Get price transition brackets for draw r ----
                        # Each Halton draw r gives a predicted next-period price vector
                        # p'_r from the AR(1) process: p'_k = φ_0k + φ_1k·p_k + η_rk.
                        # This predicted price is generally between two grid points on
                        # each category's 1D price grid. c_lo/c_hi bracket the cig price,
                        # e_lo/e_hi bracket the ecig price, w_c/w_e are interp weights.
                        c_lo = p_cig_lo[p_idx, r]
                        c_hi = p_cig_hi[p_idx, r]
                        w_c  = p_cig_w[p_idx, r]
                        e_lo = p_ecig_lo[p_idx, r]
                        e_hi = p_ecig_hi[p_idx, r]
                        w_e  = p_ecig_w[p_idx, r]

                        # ---- Step 2b: Map 2D price brackets to combined price state indices ----
                        # The combined price state is stored as a single index where
                        # p_combined = (cig_idx - 1) * N_P + ecig_idx. The four corners
                        # of the bilinear interpolation are (cig_lo, ecig_lo), (cig_lo, ecig_hi),
                        # (cig_hi, ecig_lo), (cig_hi, ecig_hi).
                        p_ll = (c_lo - 1) * N_P + e_lo
                        p_lh = (c_lo - 1) * N_P + e_hi
                        p_hl = (c_hi - 1) * N_P + e_lo
                        p_hh = (c_hi - 1) * N_P + e_hi

                        # ---- Step 2c: Compute bilinear interpolation weights in price space ----
                        # Standard bilinear weights: (1-w_c)(1-w_e) for lower-lower corner, etc.
                        w_ll = (1 - w_c) * (1 - w_e)
                        w_lh = (1 - w_c) * w_e
                        w_hl = w_c * (1 - w_e)
                        w_hh = w_c * w_e

                        # ---- Step 2d: Accumulate bilinearly interpolated V_now for this draw ----
                        # For each of the 4 next-period TYA states, at both addiction brackets.
                        # After the r-loop, EV_lo_k / EV_hi_k hold the unnormalized sum over draws.

                        # Next-period TYA state 1 (no TYA, stable):
                        EV_lo_1 += w_ll * V_now[1, lo_a, p_ll] +
                                   w_lh * V_now[1, lo_a, p_lh] +
                                   w_hl * V_now[1, lo_a, p_hl] +
                                   w_hh * V_now[1, lo_a, p_hh]

                        EV_hi_1 += w_ll * V_now[1, hi_a, p_ll] +
                                   w_lh * V_now[1, hi_a, p_lh] +
                                   w_hl * V_now[1, hi_a, p_hl] +
                                   w_hh * V_now[1, hi_a, p_hh]

                        # Next-period TYA state 2 (no TYA, approaching):
                        EV_lo_2 += w_ll * V_now[2, lo_a, p_ll] +
                                   w_lh * V_now[2, lo_a, p_lh] +
                                   w_hl * V_now[2, lo_a, p_hl] +
                                   w_hh * V_now[2, lo_a, p_hh]

                        EV_hi_2 += w_ll * V_now[2, hi_a, p_ll] +
                                   w_lh * V_now[2, hi_a, p_lh] +
                                   w_hl * V_now[2, hi_a, p_hl] +
                                   w_hh * V_now[2, hi_a, p_hh]

                        # Next-period TYA state 3 (TYA present, stable):
                        EV_lo_3 += w_ll * V_now[3, lo_a, p_ll] +
                                   w_lh * V_now[3, lo_a, p_lh] +
                                   w_hl * V_now[3, lo_a, p_hl] +
                                   w_hh * V_now[3, lo_a, p_hh]

                        EV_hi_3 += w_ll * V_now[3, hi_a, p_ll] +
                                   w_lh * V_now[3, hi_a, p_lh] +
                                   w_hl * V_now[3, hi_a, p_hl] +
                                   w_hh * V_now[3, hi_a, p_hh]

                        # Next-period TYA state 4 (TYA present, ending soon):
                        EV_lo_4 += w_ll * V_now[4, lo_a, p_ll] +
                                   w_lh * V_now[4, lo_a, p_lh] +
                                   w_hl * V_now[4, lo_a, p_hl] +
                                   w_hh * V_now[4, lo_a, p_hh]

                        EV_hi_4 += w_ll * V_now[4, hi_a, p_ll] +
                                   w_lh * V_now[4, hi_a, p_lh] +
                                   w_hl * V_now[4, hi_a, p_hl] +
                                   w_hh * V_now[4, hi_a, p_hh]
                    end

                    # ---- Step 3: Combine Halton draws and addiction brackets ----
                    # For each next-period TYA state tya', compute:
                    #   E_price[V(tya', a', p)] by averaging over R draws and interpolating
                    #   between the low and high addiction brackets.
                    # EV_next_k = (1 - w_a) · (1/R · Σ_r V_now[k, lo_a, p'_r])
                    #           +      w_a  · (1/R · Σ_r V_now[k, hi_a, p'_r])
                    EV_next_1 = (1 - w_a) * (EV_lo_1 * inv_R) + w_a * (EV_hi_1 * inv_R)
                    EV_next_2 = (1 - w_a) * (EV_lo_2 * inv_R) + w_a * (EV_hi_2 * inv_R)
                    EV_next_3 = (1 - w_a) * (EV_lo_3 * inv_R) + w_a * (EV_hi_3 * inv_R)
                    EV_next_4 = (1 - w_a) * (EV_lo_4 * inv_R) + w_a * (EV_hi_4 * inv_R)

                    # ---- Step 4: Integrate over TYA transitions, then compute V_d and V_e ----
                    # For each CURRENT TYA state tya_idx, the expected continuation value
                    # weights each next-period TYA state by the transition probability:
                    #   EV = Σ_{tya'} Π_tya[tya_idx, tya'] · EV_next_{tya'}
                    #
                    # This is where β identification comes from: a household in state 4
                    # (ending soon) puts high weight on EV_next_1 (no TYA, where the
                    # flavor bonus λ_2 disappears). Under β < 1, this future loss is
                    # discounted more steeply, reducing the anticipatory behavioral response.
                    for tya_idx in 1:N_TYA
                        EV = Π_tya[tya_idx, 1] * EV_next_1 +
                             Π_tya[tya_idx, 2] * EV_next_2 +
                             Π_tya[tya_idx, 3] * EV_next_3 +
                             Π_tya[tya_idx, 4] * EV_next_4

                        # Decision utility (what the agent uses to choose):
                        #   V_d[tya, j, a, p] = U[tya, j, a, p] + β·δ · EV
                        # Experienced utility (what the agent actually receives):
                        #   V_e[tya, j, a, p] = U[tya, j, a, p] +   δ · EV
                        # Both use the SAME EV (the sophisticated agent correctly predicts
                        # future behavior). The only difference is the discount factor.
                        V_d[tya_idx, j_idx, a_idx, p_idx] = U[tya_idx, j_idx, a_idx, p_idx] + βδ * EV
                        V_e[tya_idx, j_idx, a_idx, p_idx] = U[tya_idx, j_idx, a_idx, p_idx] + δ * EV
                    end
                end

                # ---- Step 5: Aggregate into ex-ante value V_next ----
                # For each TYA state at this (a, p), compute the sophisticated continuation
                # value from the choice-specific V_d and V_e computed above.
                for tya_idx in 1:N_TYA

                    # ---- Step 5a: Compute choice probabilities from V_d (stable softmax) ----
                    # Find max of V_d across alternatives for numerical stability.
                    # Without subtracting the max, exp(V_d) can overflow for large values.
                    vd_max = V_d[tya_idx, 1, a_idx, p_idx]
                    for j in 2:N_J
                        val = V_d[tya_idx, j, a_idx, p_idx]
                        if val > vd_max
                            vd_max = val
                        end
                    end

                    # Compute Σ_j exp(V_d_j - vd_max) for the denominator of softmax.
                    # This is the unnormalized sum; dividing gives choice probabilities.
                    sum_exp = 0.0
                    for j in 1:N_J
                        sum_exp += exp(V_d[tya_idx, j, a_idx, p_idx] - vd_max)
                    end
                    log_denom = log(sum_exp)

                    # ---- Step 5b: Compute the sophisticated aggregator ----
                    # V_next[tya, a, p] = Σ_j p_j · V_e_j + H(p)
                    # where p_j = softmax(V_d)_j and H(p) = -Σ_j p_j · log(p_j).
                    #
                    # Expanding: = Σ_j p_j · V_e_j - Σ_j p_j · log(p_j)
                    #            = Σ_j p_j · (V_e_j - log(p_j))
                    #
                    # We compute log(p_j) = V_d_j - vd_max - log_denom for stability,
                    # then p_j = exp(log(p_j)).
                    #
                    # When β = 1: V_d = V_e, so this becomes Σ_j p_j·(V_e_j - log(p_j))
                    # = Σ_j p_j·V_e_j + H(p) = logsumexp(V_e). The standard T1EV formula.
                    agg = 0.0
                    for j in 1:N_J
                        log_pj = V_d[tya_idx, j, a_idx, p_idx] - vd_max - log_denom
                        pj = exp(log_pj)
                        # p_j · V_e_j + p_j · (-log(p_j))  =  p_j · V_e_j - p_j · log(p_j)
                        agg += pj * V_e[tya_idx, j, a_idx, p_idx] - pj * log_pj
                    end
                    V_next[tya_idx, a_idx, p_idx] = agg
                end
            end
        end

        # ---- Convergence check ----
        # Sup-norm: max absolute difference between V_next and V_now across all states
        current_diff = maximum(abs.(V_next .- V_now))

        # Jacobi update: replace V_now with V_next for the next iteration
        V_now .= V_next

        # Print progress at iterations 1, 10, and every 100th iteration
        if verbose && (iter % 100 == 0 || iter == 1 || iter == 10)
            elapsed = time() - t_vfi
            ratio = prev_diff < Inf ? current_diff / prev_diff : NaN
            log_msg(@sprintf("    VFI iter %6d | sup-norm = %.6e | ratio = %.4f | elapsed = %.1fs",
                iter, current_diff, ratio, elapsed))
        end

        prev_diff = current_diff

        # Check if sup-norm is below tolerance ε
        if current_diff < ε
            if verbose
                elapsed = time() - t_vfi
                log_msg(@sprintf("    VFI converged in %d iterations (sup-norm = %.6e, %.1fs)", iter, current_diff, elapsed))
            end
            converged = true
            break
        end

        # Report if we hit the maximum number of iterations without converging
        if iter == max_iter
            elapsed = time() - t_vfi
            log_msg(@sprintf("VFI did not converge after %d iterations (sup-norm = %.6e, %.1fs)", max_iter, current_diff, elapsed))
        end
    end

    # ---- Post-convergence: recompute V_d at the fixed point ----
    # During the last VFI iteration, V_d was computed using V_now from the
    # *previous* iteration. But then V_now was updated to V_next. So V_d is
    # one iteration stale relative to the converged V_now. We recompute V_d
    # one final time using the converged V_now to ensure exact consistency.
    # This uses the same Bellman equation: V_d = U + βδ · EV(V_now_converged),
    # with EV integrating over both price transitions and Π_tya.
    recompute_choice_values_4tya!(
        V_d, V_now, U, Float64(βδ), Π_tya,
        N_J, N_A, N_P, N_Pcomb, inv_R,
        p_cig_lo, p_cig_hi, p_cig_w,
        p_ecig_lo, p_ecig_hi, p_ecig_w,
        a_lower, a_upper, a_weight
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
Trilinearly interpolate V_choice at a continuous state (a, p_cig, p_ecig)
for all alternatives j = 1, ..., N_J.

Uses the same bracket/weight logic as the log_likelihood function:
  1. Linear interpolation over the addiction grid
  2. Bilinear interpolation over the 2D price grid (cig × ecig)

Returns:
- v_interp: Vector of interpolated V_choice values for all N_J alternatives
"""
function interpolate_v_choice(
    V_choice::Array{Float64, 4},
    tya_idx::Integer,
    a::Real,
    obs_cig::Real,
    obs_ecig::Real,
    N_J::Integer,
    N_P::Integer,
    A::AbstractVector{<:Real},
    P::AbstractMatrix{<:Real}
)

    N_A = length(A)

    # 1D price grids
    P_cig  = @view P[:, 1]
    P_ecig = @view P[:, 2]

    # --- Addiction interpolation brackets ---
    # Clamp continuous addiction to grid bounds
    a_i = clamp(a, A[1], A[end])

    # Find upper bracket via binary search
    hi_a = clamp(searchsortedfirst(A, a_i), 1, N_A)
    lo_a = clamp(hi_a - 1, 1, N_A)

    # Interpolation weight for addiction
    w_a = (lo_a == hi_a) ? 0.0 : (a_i - A[lo_a]) / (A[hi_a] - A[lo_a])

    # --- Price interpolation brackets ---
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

    # --- Trilinear interpolation for all alternatives ---
    v_interp = Vector{Float64}(undef, N_J)

    for j in 1:N_J

        # Bilinear price interpolation at lower addiction bracket
        v_lo_a = w_ll * V_choice[tya_idx, j, lo_a, p_ll] +
                 w_lh * V_choice[tya_idx, j, lo_a, p_lh] +
                 w_hl * V_choice[tya_idx, j, lo_a, p_hl] +
                 w_hh * V_choice[tya_idx, j, lo_a, p_hh]

        # Bilinear price interpolation at upper addiction bracket
        v_hi_a = w_ll * V_choice[tya_idx, j, hi_a, p_ll] +
                 w_lh * V_choice[tya_idx, j, hi_a, p_lh] +
                 w_hl * V_choice[tya_idx, j, hi_a, p_hl] +
                 w_hh * V_choice[tya_idx, j, hi_a, p_hh]

        # Linear interpolation over addiction
        v_interp[j] = (1 - w_a) * v_lo_a + w_a * v_hi_a
    end

    return v_interp
end


"""
Compute the sample log-likelihood by interpolating V_choice at each
observation's continuous addiction level and observed prices.

Under the Type I extreme value (logit) assumption, the log conditional
choice probability for observation i choosing alternative y_i is:

  log P(y_i | x_i; θ) = log(exp(V_choice[tya_i, y_i, a_i, p_i]) / Σ_j exp(V_choice[tya_i, j, a_i, p_i]))
                      = V_choice[tya_i, y_i, a_i, p_i]
                        - log(Σ_j exp(V_choice[tya_i, j, a_i, p_i]))

where V_choice is on the discretized state grid. Since the observed
addiction level and prices are continuous, I interpolate V_choice:
  - Linear interpolation over the addiction grid
  - Bilinear interpolation over the 2D price grid (cig × ecig)

This gives trilinear interpolation of V_choice at each observation's
actual state (a_continuous_i, p_cig_i, p_ecig_i), from which the
log choice probability is computed as the interpolated V_choice for
the chosen alternative minus the logsumexp over all alternatives.

No structural change from 01_Functions.jl — tya_state values are now 1-4
instead of 1-2, which works because tya_state[i] is used as a direct index
into V_choice's first dimension (now size 4).

Returns:
- Log-likelihood (scalar): ℓ(θ) = Σ_i log P(y_i | x_i; θ)
"""
function log_likelihood(
    V_choice::Array{Float64, 4},
    N_J::Integer,
    N_P::Integer,
    A::AbstractVector{<:Real},
    P::AbstractMatrix{<:Real},
    y::AbstractVector{<:Integer},
    tya_state::AbstractVector{<:Integer},
    a_continuous::AbstractVector{<:Real},
    p_continuous::AbstractMatrix{<:Real}
)

    # Number of observations
    N_HHT = length(y)

    # Accumulate log-likelihood
    LL = 0.0

    # Loop over observations
    for i in 1:N_HHT

        # Trilinear interpolation of V_choice at this observation's continuous state
        v_interp = interpolate_v_choice(
            V_choice, tya_state[i], a_continuous[i],
            p_continuous[i, 1], p_continuous[i, 2],
            N_J, N_P, A, P
        )

        # Log choice probability: V_choice(y_i) - logsumexp(V_choice)
        LL += v_interp[y[i]] - logsumexp(v_interp)
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

Used by:
  - MC simulation (simulate_data): after computing logit choice probabilities
    from interpolated V_decision_true, draws each household's simulated
    product choice j ∈ {1, ..., N_J}
  - Counterfactual simulation: draws counterfactual choices under the
    flavor ban policy to compute predicted market shares

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

    # Fallback: if floating-point rounding causes Σ probs < 1 so that
    # u exceeds the accumulated sum, return the last category. This
    # can happen when probs doesn't sum to exactly 1.0 due to
    # finite-precision arithmetic in the softmax computation.
    return length(probs)
end


#############################
# 10. Optimization &
#     Objective
#############################

# --- Parameter Bounds ---

# Economic parameter bounds (standardized units).
# Ordering: α_T, α_E, α_TE, λ_1, λ_2, λ_3, λ_4, μ, γ, ω, ξ_T, ξ_E, ξ_TE, β
const θ_lower_bound = Float64[-Inf, -Inf, -Inf, -Inf, -Inf, -Inf, -Inf,    0, -Inf, -Inf, -Inf, -Inf, -Inf, 0.01]
const θ_upper_bound = Float64[ Inf,  Inf,  Inf,  Inf,  Inf,  Inf,  Inf,  Inf,    0,    0,  Inf,  Inf,  Inf, 1.00]

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


# --- Nelder-Mead Simplex ---

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

    # Remaining D vertices: shift x0 along each coordinate direction corresping to S.add
    for d in 1:D
        
        # Copy of original vertex (the actual parameter vector)
        vertex = copy(x0)

        # Pertrub d'th dimension of the vertex 
        vertex[d] += S.add[d]

        # Simplex at index d + 1 is the pertrubed vertex 
        # d + 1 b/c the simplex at position 1 is simply the actual parameter vector 
        simplex[d + 1] = vertex
    end

    return simplex
end


# --- Random Amoeba Optimizer ---

"""
Modified Nelder-Mead optimizer with random restarts.

Addresses the key weakness of Nelder-Mead (sensitivity to initialization)
by running multiple outer tries with random reinitializations.

Outer loop (L outer tries):
  1. Inner loop (M inner tries): short Nelder-Mead runs with randomized
     simplex deviations to cheaply explore the parameter space
  2. Full convergence run: long Nelder-Mead from the best inner result
  3. Best update: track the global best across all outer tries
  4. Random reinitialization: stochastically choose starting point for
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
    inner_iter::Integer = 100,
    long_run_iter::Integer = 1_000,
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
    pnames = collect(String, string.(keys(starting_param)))

    # Initialize global best objective to a very large value
    overall_min = Inf

    # Initialize file for writing outer try results (if path provided)
    outer_try_io = nothing
    if outer_try_file !== nothing

        # Open the outer try file 
        outer_try_io = open(outer_try_file, "w")
        
        # Write header row with parameter names
        println(outer_try_io, "outer_try," * join(pnames, ","))
        flush(outer_try_io)
    end

    # Initialize file for writing inner try results (if path provided)
    inner_try_io = nothing
    if inner_try_file !== nothing

        # Open the inner try file
        inner_try_io = open(inner_try_file, "w")

        # Write header row with parameter names
        println(inner_try_io, "outer_try,inner_try," * join(pnames, ","))
        flush(inner_try_io)
    end


    #############################
    # Outer Tries
    #############################

    # Print and log message indicating full optimization routine is starting
    log_msg("\n" * "="^60)
    log_msg("RANDOM AMOEBA OPTIMIZATION")
    log_msg("="^60)
    log_msg("Starting parameters:")
    for d in 1:N_params
        log_msg("  $(pnames[d]) = $(base_param[d])")
    end
    log_msg("")

    # Time all outer tries
    t_total = time()

    # Loop over outer tries
    for l in 1:L

        # Time the current outer try
        t_outer = time()

        # Reset simplex deviations to the base values at the start of each outer try
        this_add = add

        # Print and log message indicating current outer try 
        log_msg("\n" * "-"^60)
        log_msg("OUTER TRY $l / $L")
        log_msg("-"^60)


        #############################
        # Inner Tries
        #############################

        # Track best parameters across inner tries
        best_inner_val = Inf
        best_inner_param = param

        # Loop over inner tries
        for m in 1:M

            # Update global phase tracking
            global ra_outer_try = l
            global ra_inner_run = m

            # Time the current inner try
            t_inner = time()

            # Print and log message regarding current minimization run 
            log_msg("\n  Minimization run $l.$m")
            log_msg("  " * "-"^22)

            # Run Nelder-Mead for a limited number of iterations (inner_iter iterations)
            # Note, each iteration can call the objective many times. 
            # See paper appendix for details on Nelder-Mead 
            result = optimize(
                objective, param,
                NelderMead(initial_simplex = SimplexWithAdd(this_add)),
                Optim.Options(iterations = inner_iter, f_abstol = 1e-4)
            )

            # Update θ to the minimizer from this short run
            param = Optim.minimizer(result)

            # Update best inner result if this run improved
            if Optim.minimum(result) < best_inner_val
                best_inner_val = Optim.minimum(result)
                best_inner_param = param
            end

            # Time the current inner try
            inner_elapsed = time() - t_inner

            # Print and log short run results
            log_msg("  Run $l.$m complete:")
            log_msg("    Time: $(round(inner_elapsed, digits=1))s | " *
                    "Iters: $(Optim.iterations(result))/$inner_iter | " *
                    "Converged: $(Optim.converged(result))")
            log_msg("    Objective: $(round(Optim.minimum(result), digits=6))")
            log_msg("    Parameters:")
            for d in 1:N_params
                log_msg("      $(pnames[d]) = $(round(param[d], digits=8))")
            end

            # Write this inner try's parameters to file
            if inner_try_io !== nothing

                # Join best_param vector into a single string separated by commas
                param_str = join([@sprintf("%.10f", x) for x in param], ",")

                # Write current outer run, inner run, and parameters for the current inner run to a file 
                println(inner_try_io, "$l,$m,$param_str")
                flush(inner_try_io)
            end

            # Randomize simplex deviations: v_d ← v_d * u_d, u_d ~ Uniform(0.5, 2.0)
            # Multiplicative scaling preserves relative magnitude across parameters
            # rand(N_params) draws N_params numbers from the standard uniform U(0, 1)
            # Multiplying this distribution by 1.5 gives a U(0, 1.5) distribution
            # Adding 0.5 gives a U(0.5, 2.0) distibution
            # This means on the next short run (m + 1), the parameter vector recovered by Nelder-Mead
            # at iteration m, which I called param, gets scaled by a random number between 0.5x and 2.0x of
            # it's current value
            this_add = add .* (0.5 .+ 1.5 .* rand(N_params))
        end


        #############################
        # Full Convergence Run
        #############################

        # Time the full convergence run
        t_long = time()

        # Update global phase tracking (0 = long convergence run)
        global ra_outer_try = l
        global ra_inner_run = 0

        # Print and log message indicating starting full convergence run
        log_msg("\n  Full convergence run...")

        # Start full convergence from the best inner run (not necessarily the last one)
        param = best_inner_param

        # Apply Nelder-Mead until convergence or long_run_iter iterations, whichever occurs first
        result = optimize(
            objective, param,
            NelderMead(initial_simplex = SimplexWithAdd(this_add)),
            Optim.Options(iterations = long_run_iter, f_abstol = 1e-3)
        )

        # Time the full convergence run
        long_elapsed = time() - t_long

        # Print and log message indicating results from long convergence run
        log_msg("  Full run complete:")
        log_msg("    Time: $(round(long_elapsed, digits=1))s | " *
                "Iters: $(Optim.iterations(result))/$(long_run_iter) | " *
                "Converged: $(Optim.converged(result))")
        log_msg("    Objective: $(round(Optim.minimum(result), digits=6))")

        # Update global best θ* if this outer try produced a lower minimum
        if Optim.minimum(result) < overall_min
            overall_min = Optim.minimum(result)
            best_param  = Optim.minimizer(result)
        end

        # Time the current outer try
        outer_elapsed = time() - t_outer

        # Print and log message from this outer try
        log_msg("\n  OUTER TRY $l SUMMARY:")
        log_msg("    Time: $(round(outer_elapsed, digits=1))s")
        log_msg("    Overall best objective: $(round(overall_min, digits=6))")
        log_msg("    Best parameters:")
        for d in 1:N_params
            log_msg("      $(pnames[d]) = $(round(best_param[d], digits=8))")
        end

        # Write this outer try's best parameters to file
        if outer_try_io !== nothing

            # Join best_param vector into a single string separated by commas
            param_str = join([@sprintf("%.10f", x) for x in best_param], ",")

            # Write current outer run and parameters for the current outer run to a file 
            println(outer_try_io, "$l,$param_str")
            flush(outer_try_io)
        end

        # Random reinitialization for next outer try:
        #   P(0.2): reset to θ₀ (exploration — escape local minima)
        #   P(0.4): reset to θ*  (exploitation — refine best found)
        #   P(0.4): continue from current minimizer (exploitation — keep searching nearby)
        u = rand()
        if u < 0.2
            log_msg("\n  Reinitializing: original starting parameters")
            param = base_param
        elseif u < 0.6
            log_msg("\n  Reinitializing: best parameters to date")
            param = best_param
        else
            log_msg("\n  Reinitializing: current result parameters")
            param = Optim.minimizer(result)
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
    log_msg("\n" * "="^60)
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
    # E.g., (α_T = 0.5, α_E = 0.2, ...)
    opt_param = (; zip(keys(starting_param), best_param)...)

    return opt_param, overall_min
end


# --- Objective Function ---

"""
Objective function for the optimizer.

Takes a parameter vector θ_vec,
recomputes flow utility and value function, evaluates the log-likelihood,
and returns the negative log-likelihood.

Increments `est_eval_count` and times each evaluation. **Box constraints:**
returns `1e14` penalty if any parameter violates bounds via `check_parameter_bounds`.

For each candidate θ:
  (1) computes addiction grid 
  (2) computes flow utility U 
  (3) computes addiction transition brackets 
  (4) solves VFI from scratch
        **Early-exit:** if VFI did not converge, logs a PENALTY message and returns `1e14`.
        Otherwise:
  (5) computes addiction trajectories at fixed ψ
  (6) evaluates log-likelihood via trilinear interpolation
  (7) logs eval number, LL, VFI iters, elapsed time, and θ vector

Accesses global data loaded by 02_Estimation.jl (e.g., N_J, y, etc.)
so each data does not need to be passed as arguments.
"""
function objective(θ_vec::AbstractVector{<:Real})
    
    # Use global evaluation counter 
    global est_eval_count

    # Start evaluation time
    t_eval = time()

    # Update evaluation count
    est_eval_count += 1

    # Economic parameter bounds check (all 14 parameters)
    in_bounds, violations = check_parameter_bounds(θ_vec, est_param_names)
    if !in_bounds
        elapsed = time() - t_eval
        log_msg("")
        log_msg(@sprintf("  Eval %d | PENALTY (bounds: %s) | time = %.1fs",
            est_eval_count, violations, elapsed))
        log_msg("")
        return 1e14
    end

    # ψ is global from get_fixed_parameters()
    N_A_current, A_current = get_addiction_space(ψ)

    # Extract β from θ_vec
    β_current = θ_vec[end]

    # Structural parameters for flow utility 
    θ_struct = θ_vec[1:end-1]

    # Compute flow utility for the current θ
    U_current = get_flow_utility(
        θ_struct, N_J, N_A_current, N_Pcomb, A_current, c_cig, c_ecig, c_bundle, n, is_non_fda_flavored, is_fda_flavored, cat_idx, E
    )

    # Compute addiction transition brackets 
    a_lower_current, a_upper_current, a_weight_current = precompute_addiction_transitions(
        N_J, N_A_current, ψ, A_current, n
    )

    # Solve VFI 
    V, V_decision_current, vfi_iters, vfi_converged = solve_vfi_sophisticated(
        N_J, N_A_current, N_P, N_Pcomb, β_current, δ, U_current,
        a_lower_current, a_upper_current, a_weight_current,
        p_cig_lo, p_cig_hi, p_cig_w,
        p_ecig_lo, p_ecig_hi, p_ecig_w,
        Π_tya
    )

    # If VFI did not converge, skip LL and return penalty (very large number for LL)
    if !vfi_converged

        # Get elapsed time 
        elapsed = time() - t_eval

        # Print and log message about VFI not converging 
        log_msg("")
        log_msg(@sprintf("  Eval %d | PENALTY (VFI not converged) | VFI iters = %d | time = %.1fs",
            est_eval_count, vfi_iters, elapsed))
        log_msg(@sprintf("    θ for Eval %d:", est_eval_count))

        # Print and log message of parameter name and parameter for which the VFI did not converge 
        for (i, x) in enumerate(θ_vec)
            pname = est_param_names[i]
            log_msg(@sprintf("      %s = %.6f", pname, x))
        end
        log_msg("")


        return 1e14
    end

    # Compute addiction trajectories 
    a0_current, _ = get_initial_addiction_stock(ψ, A_current, n, y, hh_codes)
    _, a_continuous_current = simulate_addiction_trajectories(N_A_current, ψ, A_current, n, y, hh_codes, a0_current)

    # Compute log-likelihood via trilinear interpolation at continuous states
    LL = log_likelihood(
        V_decision_current, N_J, N_P, A_current, P, y, tya_state, a_continuous_current, p_continuous
    )

    # Get elapsed time 
    elapsed = time() - t_eval

    # Print and log message for current objective evaluation 
    log_msg("")
    log_msg(@sprintf("  Eval %d | LL = %.4f | VFI iters = %d | time = %.1fs",
        est_eval_count, LL, vfi_iters, elapsed))
    log_msg(@sprintf("    θ for Eval %d:", est_eval_count))

    # Print and log message of parameter name and parameter 
    for (i, x) in enumerate(θ_vec)
        pname = est_param_names[i]
        log_msg(@sprintf("      %s = %.6f", pname, x))
    end
    log_msg("")

    # Return negative log-likelihood (optimizer minimizes)
    return -LL
end


