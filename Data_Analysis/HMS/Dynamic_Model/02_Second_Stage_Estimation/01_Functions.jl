################################################################################
# William Brasic
# The University of Arizona
# wbrasic97@gmail.com
# December 2025
#
# This script creates the necessary functions to estimate the dynamic model.
# Functions are ordered by their execution sequence in 02_Estimation.jl.
################################################################################


#############################
# Preliminaries
#############################

# Install any missing packages, then load
import Pkg
for pkg in ["CSV", "DataFrames", "Optim", "Statistics", "ForwardDiff"]
    if Base.find_package(pkg) === nothing
        Pkg.add(pkg)
    end
end
using CSV, DataFrames, Optim, Statistics, LinearAlgebra, ForwardDiff, Printf, Dates


#############################
# Logging
#############################

# Global log file handle (set by calling script before estimation)
# Initialize to nothing because each separate script will explicitly update this one single time
est_log_io = nothing

# Global evaluation counter (reset before each estimation run)
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
Write a message to the active log file and flush immediately.
Also prints to stdout. Checks est_log_io first, then falls back to
mc_log_io (defined in 01_MC_Sim_Functions.jl) so VFI messages are
captured regardless of which script is running.
"""
function est_log(msg::String)

    # Grab the global est_log_io
    # I use global because once this is updated from "nothing" in the script using it, it never changes 
    # so there isn't much of a need of continually passing est_log_io each separate time
    global est_log_io

    # Print the message to the terminal
    println(msg)

    # Log the message if est_log_io is assigned 
    if est_log_io !== nothing
        println(est_log_io, msg)    # Write to memory buffer
        flush(est_log_io)           # Forces buffer to write to disk
    
    # If we are running the MC simulation, use mc_log_io instead to log messages
    elseif isdefined(Main, :mc_log_io) && Main.mc_log_io !== nothing
        println(Main.mc_log_io, msg)    # Write to memory buffer
        flush(Main.mc_log_io)           # Forces buffer to write to disk
    end
end


#############################
# Household Codes
#############################

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


#############################
# Fixed Parameters
#############################

"""
Set fixed parameters

Returns:
- Addiction decay rate ψ
- Present bias term β
- Monthly discount factor δ
"""
function get_fixed_parameters()

    # Addiction decay rate (to be estimated jointly in the future)
    ψ = 0.50

    # Present bias parameter (β-δ discounting; β = 1.0 is standard exponential)
    β = 1.0

    # Monthly discount factor
    δ = 0.99

    return ψ, β, δ
end


#############################
# State Spaces and Choices
#############################


"""
Get addiction space

The addiction grid is constructed assuming STANDARDIZED nicotine (n_max = 1.0).
This ensures the addiction state space is also in standardized units [0, 1/ψ].

Returns:
- Number of addiction states
- Vector of addiction states (standardized)
"""
function get_addiction_space(
    ψ::Real;
    N_A::Integer = 20
)

    # With standardized nicotine, n_max = 1.0
    # Addiction upper bound: a_max = n_max / ψ = 1.0 / ψ
    n_max_standardized = 1.0
    left_endpoint  = 0.0
    right_endpoint = n_max_standardized / ψ

    # Create the addiction grid
    A_grid = collect(range(left_endpoint, right_endpoint, N_A))

    return N_A, A_grid
end


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

    # Initialize vetor to store product_
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


"""
Get household category choices in each time period

Returns:
- Number of categories (exclusive of the outside option, and combining flavored and original ecigs into one)
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
# Alternative-Level Vectors
#############################

"""
Get consumption vectors indexed by alternative j ∈ {1, ..., N_J}
Separates cigarette and e-cigarette consumption components so expenditures
and bundle consumption can be computed correctly.

Consumption is STANDARDIZED by dividing by the maximum value of each category.
This keeps utility terms at reasonable magnitudes for numerical stability.

Alternative ordering:
  j = 1:           outside option (zero consumption)
  j = 2:7:         6 cigarette quantity bins
  j = 8:10:        3 original e-cigarette bins
  j = 11:13:       3 flavored e-cigarette bins
  j = 14:17:       4 original bundles (lo/hi cig × lo/hi ecig)
  j = 18:21:       4 flavored bundles (lo/hi cig × lo/hi ecig)

Returns:
- N_cig:         Number of cigarette alternatives
- N_orig_ecig:   Number of original e-cigarette alternatives
- N_flav_ecig:   Number of flavored e-cigarette alternatives
- N_bundle:      Number of bundle alternatives
- c_cig:         Vector of standardized cigarette consumption for each alternative
- c_ecig:        Vector of standardized e-cigarette consumption for each alternative
- c_bundle:      Vector of standardized bundle consumption for each alternative
- c_cig_max:     Raw maximum cigarette consumption (for rescaling estimates)
- c_ecig_max:    Raw maximum e-cigarette consumption (for rescaling estimates)
- c_bunlde_max:  Raw maximum bundle consumption (for rescaling estimates)
"""
function get_consumption(
    N_J::Integer;
    file_name::AbstractString = "./Consumption_Spaces.csv"
)

    # Load in consumption spaces
    df = CSV.read(file_name, DataFrame)

    # Single-good consumption values (use startswith to match any suffix)
    # Filter out bundle alternatives which also start with "cig_" in their bundle names
    cig       = df.consumption[startswith.(df.alternative, "cig_") .& .!occursin.("bundle", df.alternative)]
    orig_ecig = df.consumption[startswith.(df.alternative, "orig_ecig_")]
    flav_ecig = df.consumption[startswith.(df.alternative, "flav_ecig_")]

    # Bundle names in order: 4 original bundles, then 4 flavored bundles
    bundle_names = [
        "bundle_orig_lo_lo", "bundle_orig_lo_hi", "bundle_orig_hi_lo", "bundle_orig_hi_hi",
        "bundle_flav_lo_lo", "bundle_flav_lo_hi", "bundle_flav_hi_lo", "bundle_flav_hi_hi"
    ]

    # Helper to extract single consumption value by alternative name
    get_consumption_value(alt_name) = only(df.consumption[df.alternative .== alt_name])

    # Counts
    N_cig       = length(cig)
    N_orig_ecig = length(orig_ecig)
    N_flav_ecig = length(flav_ecig)
    N_bundle_orig = 4  # 4 original e-cig bundles
    N_bundle_flav = 4  # 4 flavored e-cig bundles
    N_bundle = N_bundle_orig + N_bundle_flav

    # Initialize consumption vectors (j = 1 is outside option with zero consumption)
    c_cig  = zeros(Float64, N_J)
    c_ecig = zeros(Float64, N_J)

    # Fill vector in order: outside, cig, orig_ecig, flav_ecig, bundle_orig (4), bundle_flav (4)
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

    # Flavored e-cig alternatives
    for consumption in flav_ecig
        c_ecig[idx] = consumption
        idx += 1
    end

    # Bundle consumption (8 bundles total)
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

    return N_cig, N_orig_ecig, N_flav_ecig, N_bundle, c_cig, c_ecig, c_bundle, c_cig_max, c_ecig_max, c_bundle_max
end


"""
Get nicotine vector indexed by alternative j ∈ {1, ..., N_J}
For bundle alternatives, total nicotine is the sum of the cigarette and e-cigarette components.

Nicotine is STANDARDIZED by dividing by the maximum value. This keeps the addiction
dynamics (a' = (1-ψ)a + n[j]) at reasonable magnitudes for numerical stability.

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
    cig       = df.nicotine[startswith.(df.alternative, "cig_") .& .!occursin.("bundle", df.alternative)]
    orig_ecig = df.nicotine[startswith.(df.alternative, "orig_ecig_")]
    flav_ecig = df.nicotine[startswith.(df.alternative, "flav_ecig_")]

    # Bundle nicotine components (8 bundles: 4 orig + 4 flav, each with cig and ecig components)
    # Note: nicotine columns have _cig_nic and _ecig_nic suffixes
    # Original e-cig bundles
    bundle_orig_lo_lo_cig_nic  = df.nicotine[df.alternative .== "bundle_orig_lo_lo_cig_nic"]
    bundle_orig_lo_lo_ecig_nic = df.nicotine[df.alternative .== "bundle_orig_lo_lo_ecig_nic"]
    bundle_orig_lo_hi_cig_nic  = df.nicotine[df.alternative .== "bundle_orig_lo_hi_cig_nic"]
    bundle_orig_lo_hi_ecig_nic = df.nicotine[df.alternative .== "bundle_orig_lo_hi_ecig_nic"]
    bundle_orig_hi_lo_cig_nic  = df.nicotine[df.alternative .== "bundle_orig_hi_lo_cig_nic"]
    bundle_orig_hi_lo_ecig_nic = df.nicotine[df.alternative .== "bundle_orig_hi_lo_ecig_nic"]
    bundle_orig_hi_hi_cig_nic  = df.nicotine[df.alternative .== "bundle_orig_hi_hi_cig_nic"]
    bundle_orig_hi_hi_ecig_nic = df.nicotine[df.alternative .== "bundle_orig_hi_hi_ecig_nic"]
    # Flavored e-cig bundles
    bundle_flav_lo_lo_cig_nic  = df.nicotine[df.alternative .== "bundle_flav_lo_lo_cig_nic"]
    bundle_flav_lo_lo_ecig_nic = df.nicotine[df.alternative .== "bundle_flav_lo_lo_ecig_nic"]
    bundle_flav_lo_hi_cig_nic  = df.nicotine[df.alternative .== "bundle_flav_lo_hi_cig_nic"]
    bundle_flav_lo_hi_ecig_nic = df.nicotine[df.alternative .== "bundle_flav_lo_hi_ecig_nic"]
    bundle_flav_hi_lo_cig_nic  = df.nicotine[df.alternative .== "bundle_flav_hi_lo_cig_nic"]
    bundle_flav_hi_lo_ecig_nic = df.nicotine[df.alternative .== "bundle_flav_hi_lo_ecig_nic"]
    bundle_flav_hi_hi_cig_nic  = df.nicotine[df.alternative .== "bundle_flav_hi_hi_cig_nic"]
    bundle_flav_hi_hi_ecig_nic = df.nicotine[df.alternative .== "bundle_flav_hi_hi_ecig_nic"]

    # Bundle pairs: (cig_nicotine, ecig_nicotine) for each bundle alternative
    # Order: 4 original bundles, then 4 flavored bundles
    bundle_pairs = (
        (bundle_orig_lo_lo_cig_nic, bundle_orig_lo_lo_ecig_nic),
        (bundle_orig_lo_hi_cig_nic, bundle_orig_lo_hi_ecig_nic),
        (bundle_orig_hi_lo_cig_nic, bundle_orig_hi_lo_ecig_nic),
        (bundle_orig_hi_hi_cig_nic, bundle_orig_hi_hi_ecig_nic),
        (bundle_flav_lo_lo_cig_nic, bundle_flav_lo_lo_ecig_nic),
        (bundle_flav_lo_hi_cig_nic, bundle_flav_lo_hi_ecig_nic),
        (bundle_flav_hi_lo_cig_nic, bundle_flav_hi_lo_ecig_nic),
        (bundle_flav_hi_hi_cig_nic, bundle_flav_hi_hi_ecig_nic),
    )

    # Fill vector in order: outside, cig, orig_ecig, flav_ecig, bundle_orig (4), bundle_flav (4)
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

    # Flavored e-cig nicotine
    for nicotine in flav_ecig
        n[idx] = nicotine
        idx += 1
    end

    # Bundle nicotine (sum of cig + ecig components for each bundle)
    for (cig_nic, ecig_nic) in bundle_pairs
        @assert length(cig_nic) == length(ecig_nic) == 1 "Each bundle should have exactly one nicotine value"
        n[idx] = cig_nic[1] + ecig_nic[1]
        idx += 1
    end

    # Standardize by maximum (store raw max for rescaling estimates later)
    n_max = maximum(n)
    n ./= n_max

    return n, n_max
end


"""
Get category index for each alternative j ∈ {1, ..., N_J}.

Category mapping:
  0 = outside option
  1 = cigarettes
  2 = original e-cigarettes
  3 = flavored e-cigarettes
  4 = bundle (cig + original ecig)
  5 = bundle (cig + flavored ecig)

Returns:
- cat_idx: Vector of category indices for each alternative
"""
function get_category_index(
    N_J::Integer,
    N_cig::Integer,
    N_orig_ecig::Integer,
    N_flav_ecig::Integer
)

    # Number of bundle alternatives (4 orig + 4 flav = 8 total)
    N_bundle_orig = 4
    N_bundle_flav = 4

    # Initialize category index vector (j = 1 is outside option with cat = 0)
    cat_idx = zeros(Int, N_J)

    # Fill vector in order: outside, cig, orig_ecig, flav_ecig, bundle_orig (4), bundle_flav (4)
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

    # Flavored e-cig alternatives (cat = 3)
    for _ in 1:N_flav_ecig
        cat_idx[idx] = 3
        idx += 1
    end

    # Bundle with original e-cig (cat = 4) - 4 alternatives
    for _ in 1:N_bundle_orig
        cat_idx[idx] = 4
        idx += 1
    end

    # Bundle with flavored e-cig (cat = 5) - 4 alternatives
    for _ in 1:N_bundle_flav
        cat_idx[idx] = 5
        idx += 1
    end

    return cat_idx
end


"""
Get flavored indicator for each alternative j = 1, ..., N_J.
Flavored alternatives are: flavored e-cigarette bins (cat = 3) and
the cig + flavored ecig bundle (cat = 5).

Returns:
- is_flavored: Vector where true indicates a flavored alternative
"""
function get_flavored_indicator(
    cat_idx::AbstractVector{<:Integer}
)

    # Flavored categories are 3 (flavored ecig) and 5 (bundle with flavored ecig)
    return (cat_idx .== 3) .| (cat_idx .== 5)
end


#############################
# Demographics
#############################

"""
Get indicators for whether teen or young adult is present in the household

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
Get teen or young adult state index for each observation.

Maps the binary indicator to an array index:
  0 (no TYA present) ⟹ state 1
  1 (TYA present)    ⟹ state 2

Returns:
- tya_state: Vector of state indices for each observation
"""
function get_tya_state(
    tya::AbstractVector{<:Real}
)

    # Get teen or young adult state
    tya_state = [t == 1 ? 2 : 1 for t in tya]

    return tya_state
end


#############################
# Price Space
#############################

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
- Number of combined price vectors (N_P^2)
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


"""
Get expenditure matrix indexed by combined price state p ∈ {1, ..., N_Pcomb}
and alternative j ∈ {1, ..., N_J}.

Expenditure for alternative j at combined price state p is:
  E[p, j] = p_cig(p) × c_cig(j) + p_ecig(p) × c_ecig(j)

This correctly handles all cases:
- Outside option: 0 (both consumption vectors are zero)
- Cig-only: p_cig × c_cig (c_ecig is zero)
- Ecig-only: p_ecig × c_ecig (c_cig is zero)
- Bundles: p_cig × c_cig + p_ecig × c_ecig

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
    Pcomb::AbstractMatrix{<:Real}
)

    # Reconstruct raw consumption from standardized values
    c_cig_raw  = c_cig .* c_cig_max
    c_ecig_raw = c_ecig .* c_ecig_max

    # Initialize expenditure matrix
    E = zeros(Float64, N_Pcomb, N_J)

    # Fill expenditure matrix using raw consumption (actual dollars)
    for p in 1:N_Pcomb
        p_cig  = Pcomb[p, 1]
        p_ecig = Pcomb[p, 2]
        for j in 1:N_J
            E[p, j] = p_cig * c_cig_raw[j] + p_ecig * c_ecig_raw[j]
        end
    end

    # Standardize by maximum (store raw max for rescaling estimates later)
    E_max = maximum(E)
    E ./= E_max

    return E, E_max
end


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


#############################
# Household State 
# Trajectories
#############################

"""
Map observed household prices to the nearest combined price grid index to
get the observed price state for each observation for likelihood computation.

Prices.csv contains bin-specific prices (capturing quantity discounts).
For the dynamic model's price state, we compute a representative price per
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
    cig_cols = [:cig_1to2_p, :cig_3to10_p, :cig_11to20_p, :cig_21to30_p, :cig_31to40_p, :cig_41plus_p]
    ecig_cols = [:orig_ecig_1to10_p, :orig_ecig_10to30_p, :orig_ecig_30plus_p,
                 :flav_ecig_0to10_p, :flav_ecig_10to30_p, :flav_ecig_30plus_p]

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
# Estimation
#############################

"""
Addiction law of motion governed by parameter ψ

Returns:
- Updated addiction level: a' = (1 - ψ) * a + n
"""
function addiction_evolution(
    ψ::Real,
    a::Real,
    n::Real
)

    return (1 - ψ) * a + n
end


"""
Pre-compute flow utility for all (tya, alternative, addiction, price) states.

The flow utility for alternative j given addiction state a, price state p, and
TYA state is:

  u(j,a,p,tya) = α_T·c_cig[j] + α_E·c_ecig[j] + α_TE·c_cig[j]·c_ecig[j]
               + 𝟙[flavored]·(λ_1 + λ_2·𝟙[tya])
               + μ·a·𝟙[j ≠ outside] + γ·a
               + ω·E[p,j]
               + ξ_k

The reinforcement term μ·a·𝟙[j ≠ outside] operates at the extensive margin
(use vs. not use), breaking the collinearity with α·c[j] that arises when
using μ·a·n[j] (since n[j] ∝ c[j]).

For the outside option (j = 1), all consumption, expenditure, fixed effect,
flavored, and reinforcement terms are zero, so u = γ·a. No special case needed.

Fixed effects: ξ_T for k = 1, ξ_E for k ∈ {2, 3}, ξ_TE for k ∈ {4, 5}.

Returns:
- U: 4D array of flow utilities, U[tya_idx, j, a_idx, p_idx]
      Dimensions: 2 × N_J × N_A × N_Pcomb
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
    is_flavored::AbstractVector{Bool},
    cat_idx::AbstractVector{<:Integer},
    E::AbstractMatrix{<:Real}
)

    # Unpack parameters
    α_T, α_E, α_TE, λ_1, λ_2, μ, γ, ω, ξ_T, ξ_E, ξ_TE = θ

    # Number of TYA states
    N_TYA = 2

    # Initialize flow utility array
    U = zeros(Float64, N_TYA, N_J, N_A, N_Pcomb)

    # Pre-compute fixed effect for each alternative
    ξ = zeros(Float64, N_J)
    for j in 1:N_J
        if cat_idx[j] == 1
            ξ[j] = ξ_T
        elseif cat_idx[j] == 2 || cat_idx[j] == 3
            ξ[j] = ξ_E
        elseif cat_idx[j] == 4 || cat_idx[j] == 5
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
                # Note: c_bundle is pre-computed and standardized by its own max (not c_cig_max × c_ecig_max)
                # Reinforcement: μ·a·𝟙[j ≠ outside] (cat_idx > 0 for all non-outside alternatives)
                u_base = (α_T * c_cig[j_idx] + α_E * c_ecig[j_idx] + α_TE * c_bundle[j_idx]
                         + μ * a * (cat_idx[j_idx] > 0 ? 1.0 : 0.0) + γ * a
                         + ω * E[p_idx, j_idx]
                         + ξ[j_idx])

                # Loop over teen or young adult state
                for tya_idx in 1:N_TYA

                    # TYA indicator (tya_idx = 1 → no TYA, tya_idx = 2 → TYA present)
                    tya = (tya_idx == 2) ? 1 : 0

                    # Flow utility including flavor effect
                    U[tya_idx, j_idx, a_idx, p_idx] = u_base + is_flavored[j_idx] * (λ_1 + λ_2 * tya)
                end
            end
        end
    end

    return U
end


"""
Pre-compute addiction transition grid indices and interpolation weights
for each (alternative, addiction state) pair.

For alternative j and addiction state a_idx:
  a' = (1 - ψ) * A[a_idx] + n[j]

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

Starting from a0 = 0, simulates the addiction trajectory forward using observed
choices, then sets a0 to the terminal addiction level. Repeats until convergence.

Convergence is guaranteed because the law of motion a' = (1-ψ)a + n is a
contraction in a0: the influence of a0 on a_t decays b/c (1-ψ)^t.
Look at paper appendix for further details

Returns:
- a0: Dict mapping household_code → estimated initial addiction stock
"""
function get_initial_addiction_stock(
    ψ::Real,
    A::AbstractVector{<:Real},
    n::AbstractVector{<:Real},
    y::AbstractVector{<:Integer},
    hh_codes::AbstractVector{<:Integer};
    max_iter::Integer = 50,
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

    # Initialize a0 = 0 for all households
    a0 = Dict{eltype(hh_codes), Float64}(
        hh => 0.0 for hh in keys(hh_obs)
    )

    # Track iterations until convergence for each household
    hh_idx = 1
    iters_to_convergence = zeros(Int, length(hh_obs))

    # Loop over households
    for (hh, obs_indices) in hh_obs

        # Iterate until convergence for the current household
        for iter in 1:max_iter

            # Simulate forward from current a0
            a = a0[hh]
            for i in obs_indices
                a = addiction_evolution(ψ, a, n[y[i]])
            end

            # Update a0 to terminal addiction level
            change = abs(a - a0[hh])
            a0[hh] = a

            # Check convergence
            if change < tol
                iters_to_convergence[hh_idx] = iter
                break
            end

            # Safety net: warn if fixed-point iteration did not converge
            if iter == max_iter
                println("WARNING: Fixed-point iteration did not converge for household $hh after $max_iter iterations (change = $change)")
                iters_to_convergence[hh_idx] = max_iter
            end
        end

        # Update household index
        hh_idx += 1
    end

    return a0, maximum(iters_to_convergence)

end


"""
Simulate household addiction trajectories and map to nearest grid indices.

For each household, starts with a₀ from the pre-estimated initial addiction
stock and evolves addiction forward using the observed choices:
  a_{t+1} = (1 - ψ) * a_t + n[y_t]

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
    a0::AbstractDict
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
        # it uses the pre-estimated initial stock from a0[hh].
        a = get(a_current, hh, a0[hh])

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
# Value Function Iteration
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
Solve the value function via value function iteration (VFI).

To show how this works, initially we have V⁰_now = 0. This implies V¹_next = U since
E[V⁰_now] = 0. Then, we update so V¹_now ⟵ V¹_next, meaning V¹_now = U. Denoting EV as the 
expectation of U over the states, which is just the expectation of the current V_now over all states, 
we have V²_next = U + δ EV = U + δ E[V¹_now] = U + δ EV. Then, we update V²_now ⟵ V²_next so
V²_now = U + δ EV. Again, EV is the expectation over the current V_now for all states. So,
V³_next = U + δ[U + δ EV] = U + δ U + δ² EV. Then, we update so V³_now ⟵ V³_next

The infinite horizon assumption is standard for addiction models because there's no natural end date
as people don't know when they'll stop making tobacco purchase decisions. 

δ < 1 ensures the infinite sum converges to a finite value.

VFI uses δ only (exponential discounting) until the sup-norm difference
between successive value function iterates falls below tolerance ε:

  V_choice[tya, j, a, p] = u(tya, j, a, p) + δ · E[V(tya, a', p') | a, p, j]
  V[tya, a, p] = log( Σ_j exp(V_choice[tya, j, a, p]) )

After convergence, computes V_decision for quasi-hyperbolic (β-δ) discounting:

  V_decision = (1 - β) · U + β · V_choice = U + βδ · E[V | a, p, j]

When β = 1 (standard exponential), V_decision = V_choice. When β < 1
(present bias), the agent over-discounts the future relative to the present.
VFI is solved with δ only because a naive agent believes their future self
will discount exponentially.

The expected continuation value E[V(tya, a', p') | a, p, j] is computed by:
  1. Pre-computed addiction transition brackets for a' = (1-ψ)a + n[j]
  2. Bilinear price interpolation at each Halton draw's predicted prices
  3. Averaging over R Halton draws
  4. Linear interpolation across the two addiction bracket points

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
    p_ecig_w::Matrix{Float64};
    ε::Real = 1e-4,
    max_iter::Integer = 3000,
    verbose::Bool = true
)

    # Number of TYA states and Halton draws
    N_TYA = 2
    R     = size(p_cig_lo, 2)

    # Compute 1.0/R
    # We need this because we need to average the continuation values over the R Halton draws
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
                    # When we consume alternative j while having addiction level a, we 
                    # some continuous addiction a′. lo_a is the index of the grid immediately
                    # preceeding a′ and hi_a is the end of the grid immediately
                    # procedding a^′. w_a the associated interpolation weight 
                    # depending on which grid point a′ is closest to. 
                    lo_a = a_lower[j_idx, a_idx]
                    hi_a = a_upper[j_idx, a_idx]
                    w_a  = a_weight[j_idx, a_idx]

                    # Accumulate expected continuation value over Halton draws
                    # Separate accumulators for each TYA state and addiction bracket
                    # EV_lo_1 corresponds to the low addiction grid point and TYA state of 1
                    # EV_hi_1 corresponds to the high addiction grid point and TYA state of 1
                    # EV_lo_2 corresponds to the low addiction grid point and TYA state of 2
                    # EV_hi_2 corresponds to the high addiction grid point and TYA state of 2
                    # So, for each low and high addiction grid points and TYA state 
                    # we will do a bilinear interpolation over the two price categories
                    # Then, we will combine EV_lo_1 and EV_hi_1, and EV_lo_2 and EV_hi_2 using a 
                    # single interpolation over the addiction grid. 
                    EV_lo_1 = 0.0; EV_hi_1 = 0.0
                    EV_lo_2 = 0.0; EV_hi_2 = 0.0

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

                        # Bilinear interpolation of V_now over prices at each addiction bracket
                        # for TYA state 1 (_1) and TYA state 2 (_2)
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
                    end

                    # Average over Halton draws and interpolate over addiction brackets
                    EV_1 = (1 - w_a) * (EV_lo_1 * inv_R) + w_a * (EV_hi_1 * inv_R)
                    EV_2 = (1 - w_a) * (EV_lo_2 * inv_R) + w_a * (EV_hi_2 * inv_R)

                    # Store choice-specific value function
                    V_choice[1, j_idx, a_idx, p_idx] = U[1, j_idx, a_idx, p_idx] + δ * EV_1
                    V_choice[2, j_idx, a_idx, p_idx] = U[2, j_idx, a_idx, p_idx] + δ * EV_2
                end

                # Aggregate over alternatives via log-sum-exp
                # @view tells Julia to not make a copy, but rather use the V_choice currently
                # in memory
                V_next[1, a_idx, p_idx] = logsumexp(@view V_choice[1, :, a_idx, p_idx])
                V_next[2, a_idx, p_idx] = logsumexp(@view V_choice[2, :, a_idx, p_idx])
            end
        end

        # # Normalize V_next to prevent value function levels from growing unboundedly.
        # # Choice probabilities only depend on V differences, so the level is irrelevant.
        # # This normalization is applied BEFORE the convergence check so that we measure
        # # the sup-norm of normalized values, avoiding false non-convergence.
        # V_ref = V_next[1, 1, 1]
        # V_next .-= V_ref

        # Convergence check: sup norm across all states
        current_diff = maximum(abs.(V_next .- V_now))

        # Update V_now for next iteration 
        V_now .= V_next

        # Log progress every 100 iterations (only when verbose)
        if verbose && (iter % 100 == 0 || iter == 1 || iter == 10)
            elapsed = time() - t_vfi
            # ratio gives the convergence rate. A stable ratio below 1.0
            # means it's contracting. The closer to 0, the faster it's converging. If it's near 1.0 or above,
            # convergence is stalling.
            ratio = prev_diff < Inf ? current_diff / prev_diff : NaN
            est_log(@sprintf("    VFI iter %4d | sup-norm = %.6e | ratio = %.4f | elapsed = %.1fs",
                iter, current_diff, ratio, elapsed))
        end

        # Update the sup-norm
        prev_diff = current_diff

        # Check convergence
        if current_diff < ε
            if verbose
                elapsed = time() - t_vfi
                est_log(@sprintf("    VFI converged in %d iterations (sup-norm = %.6e, %.1fs)", iter, current_diff, elapsed))
            end
            converged = true
            break
        end

        # Report non-convergence
        if iter == max_iter
            elapsed = time() - t_vfi
            est_log(@sprintf("VFI did not converge after %d iterations (sup-norm = %.6e, %.1fs)", max_iter, current_diff, elapsed))
        end
    end

    # Compute V_decision for choice probabilities (quasi-hyperbolic discounting)
    # We want V_decision = U + βδ·EV
    # VFI gives V_choice = U + δ·EV, so δ·EV = V_choice - U
    # Substituting results in V_decision = U + β·(V_choice - U) = (1 - β)·U + β·V_choice
    # When β = 1 (standard exponential), V_decision = V_choice
    if β == 1.0
        V_decision = V_choice
    else
        V_decision = (1 .- β) .* U .+ β .* V_choice
    end

    return V_now, V_decision, n_iter, converged
end


#############################
# Log-Likelihood
#############################

"""
Compute the sample log-likelihood by interpolating V_choice at each
observation's continuous addiction level and observed prices.

Under the Type I extreme value (logit) assumption, the log conditional
choice probability for observation i choosing alternative y_i is:

  log P(y_i | x_i; θ) = log(exp(V_choice[tya_i, y_i, a_i, p_i]) / Σ_j exp(V_choice[tya_i, j, a_i, p_i]))
                      = V_choice[tya_i, y_i, a_i, p_i]
                        - log(Σ_j exp(V_choice[tya_i, j, a_i, p_i]))

where V_choice is on the discretized state grid. Since the observed
addiction level and prices are continuous, we interpolate V_choice:
  - Linear interpolation over the addiction grid
  - Bilinear interpolation over the 2D price grid (cig × ecig)

This gives trilinear interpolation of V_choice at each observation's
actual state (a_continuous_i, p_cig_i, p_ecig_i), from which the
log choice probability is computed as the interpolated V_choice for
the chosen alternative minus the logsumexp over all alternatives.

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

    # Number of observations and addiction grid size
    N_HHT = length(y)
    N_A   = length(A)

    # 1D price grids
    P_cig  = @view P[:, 1]
    P_ecig = @view P[:, 2]

    # Accumulate log-likelihood
    LL = 0.0

    # Buffer to hold interpolated V_choice for all alternatives at one observation
    v_interp = Vector{Float64}(undef, N_J)

    # Loop over observations
    for i in 1:N_HHT

        # TYA state for this observation
        tya_idx = tya_state[i]

        #############################
        # Addiction
        ############################# 

        # Clamp continuous addiction to grid bounds
        a_i = clamp(a_continuous[i], A[1], A[end])

        # Find upper bracket via binary search
        hi_a = clamp(searchsortedfirst(A, a_i), 1, N_A)
        lo_a = clamp(hi_a - 1, 1, N_A)

        # Interpolation weight for addiction
        w_a = (lo_a == hi_a) ? 0.0 : (a_i - A[lo_a]) / (A[hi_a] - A[lo_a])


        #############################
        # Price  
        ############################# 

        # Clamp continuous prices to grid bounds
        obs_cig  = clamp(p_continuous[i, 1], P_cig[1], P_cig[end])
        obs_ecig = clamp(p_continuous[i, 2], P_ecig[1], P_ecig[end])

        # Cigarette price brackets
        hi_c = clamp(searchsortedfirst(P_cig, obs_cig), 1, N_P)
        lo_c = clamp(hi_c - 1, 1, N_P)
        w_c  = (lo_c == hi_c) ? 0.0 : (obs_cig - P_cig[lo_c]) / (P_cig[hi_c] - P_cig[lo_c])

        # E-cigarette price brackets
        hi_e = clamp(searchsortedfirst(P_ecig, obs_ecig), 1, N_P)
        lo_e = clamp(hi_e - 1, 1, N_P)
        w_e  = (lo_e == hi_e) ? 0.0 : (obs_ecig - P_ecig[lo_e]) / (P_ecig[hi_e] - P_ecig[lo_e])

        # Combined price grid indices for bilinear interpolation 
        p_ll = (lo_c - 1) * N_P + lo_e
        p_lh = (lo_c - 1) * N_P + hi_e
        p_hl = (hi_c - 1) * N_P + lo_e
        p_hh = (hi_c - 1) * N_P + hi_e

        # Bilinear interpolation weights for price
        w_ll = (1 - w_c) * (1 - w_e)
        w_lh = (1 - w_c) * w_e
        w_hl = w_c * (1 - w_e)
        w_hh = w_c * w_e

        # Trilinear interpolation over price states and addiction state
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

        # Log choice probability: V_choice(y_i) - logsumexp(V_choice)
        LL += v_interp[y[i]] - logsumexp(v_interp)
    end

    return LL
end


#############################
# Sampling
#############################

"""
Draw a single sample from a categorical distribution using the CDF method.

Given a probability vector probs where probs[j] = P(J = j), draws a
realization by comparing a uniform draw to the cumulative distribution.

Returns:
- Sampled category index j ∈ {1, ..., length(probs)}
"""
function categorical_sample(
    probs::AbstractVector{<:Real}
)

    u = rand()
    cumulative = 0.0
    for j in eachindex(probs)
        cumulative += probs[j]
        if u <= cumulative
            return j
        end
    end

    # Fallback for numerical imprecision
    return length(probs)
end


#############################
# Interpolation
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


#############################
# Optimization
#############################

"""
Nelder-Mead operates on a simplex in D-dimensional parameter space.
A simplex in D dimensions has (D+1) vertices (e.g., a triangle in 2D,
a tetrahedron in 3D). This struct implements the Optim.Simplexer interface so that
Optim.jl's NelderMead can construct its initial simplex.

The simplex is built around a starting point x0 by perturbing each
coordinate direction d independently:
  Vertex 0: x0                      (the starting point itself)
  Vertex d: x0 + add[d] * e_d       (shifted along dimension d)
where e_d is the d-th standard basis vector. We need the adding
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
        vertex = copy(x0)
        vertex[d] += S.add[d]
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
    M::Integer,
    inner_iter::Integer;
    log_io::Union{IO, Nothing} = nothing,
    outer_try_file::Union{AbstractString, Nothing} = nothing
)

    # Write message to both terminal and log file (if provided)
    # I define this as a closure (function inside a function)
    # so I don't have to pass log_io into log_msg each time b/c evertime random_amoeba is called
    # the log_msg function recognizes that input once its passed into random_amoeba
    function log_msg(msg)

        # Print message to terminal and flush immediately
        println(msg)
        flush(stdout)

        # Log message if log_io is provided to random_amoeba
        if log_io !== nothing
            println(log_io, msg)
            flush(log_io)
        end
    end

    # Starting values of the parameters and never modified 
    base_param = collect(Float64, values(starting_param))

    # Working vector θ which is updated in each iteration of the inner try
    # and then at the end of each iteration of the outer try 
    param      = base_param

    # Stores best parameters from each outer try 
    best_param = base_param

    # Number of parameters
    N_params   = length(param)

    # Names of the parameters 
    pnames     = collect(String, string.(keys(starting_param)))

    # Initialize global best objective to a very large value
    overall_min = Inf

    # # Initialize file for writing outer try results (if path provided)
    # outer_try_io = nothing
    # if outer_try_file !== nothing
    #     outer_try_io = open(outer_try_file, "w")
    #     # Write header row with parameter names
    #     println(outer_try_io, "outer_try\t" * join(pnames, "\t"))
    #     flush(outer_try_io)
    # end


    #############################
    # Outer Tries
    #############################

    # Print message indicating full optimization routine is starting
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
            log_msg("\n  Minimization run $l.$m")
            log_msg("  " * "-"^22)

            # Run Nelder-Mead for a limited number of iterations (inner_iter iterations)
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

            # Log short run results
            log_msg("  Run $l.$m complete:")
            log_msg("    Time: $(round(inner_elapsed, digits=1))s | " *
                    "Iters: $(Optim.iterations(result))/$inner_iter | " *
                    "Converged: $(Optim.converged(result))")
            log_msg("    Objective: $(round(Optim.minimum(result), digits=6))")
            log_msg("    Parameters:")
            for d in 1:N_params
                log_msg("      $(pnames[d]) = $(round(param[d], digits=8))")
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

        # Output message indicating starting full convergence run
        log_msg("\n  Full convergence run...")

        # Start full convergence from the best inner run (not necessarily the last one)
        param = best_inner_param

        # Apply Nelder-Mead until convergence or 2,500 iterations, whichever occurs first
        result = optimize(
            objective, param,
            NelderMead(initial_simplex = SimplexWithAdd(this_add)),
            Optim.Options(iterations = 2_500, f_abstol = 1e-3)
        )

        # Time the full convergence run
        long_elapsed = time() - t_long

        # Output message indicating results from long convergence run
        log_msg("  Full run complete:")
        log_msg("    Time: $(round(long_elapsed, digits=1))s | " *
                "Iters: $(Optim.iterations(result))/5000 | " *
                "Converged: $(Optim.converged(result))")
        log_msg("    Objective: $(round(Optim.minimum(result), digits=6))")

        # Update global best θ* if this outer try produced a lower minimum
        if Optim.minimum(result) < overall_min
            overall_min = Optim.minimum(result)
            best_param  = Optim.minimizer(result)
        end

        # Time the current outer try
        outer_elapsed = time() - t_outer

        # Output message from this outer try
        log_msg("\n  OUTER TRY $l SUMMARY:")
        log_msg("    Time: $(round(outer_elapsed, digits=1))s")
        log_msg("    Overall best objective: $(round(overall_min, digits=6))")
        log_msg("    Best parameters:")
        for d in 1:N_params
            log_msg("      $(pnames[d]) = $(round(best_param[d], digits=8))")
        end

        # # Write this outer try's best parameters to file
        # if outer_try_io !== nothing
        #     param_str = join([@sprintf("%.10f", x) for x in best_param], "\t")
        #     println(outer_try_io, "$l\t$param_str")
        #     flush(outer_try_io)
        # end

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

    # # Close outer try results file
    # if outer_try_io !== nothing
    #     close(outer_try_io)
    # end

    # Time all outer tries
    total_elapsed = time() - t_total

    # Print message indicating results from full optimization run
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
    opt_param = (; zip(keys(starting_param), best_param)...)

    return opt_param, overall_min
end


#############################
# Parameter Bounds
#############################

# Economic parameter bounds (standardized units).
# Ordering: α_T, α_E, α_TE, λ_1, λ_2, μ, γ, ω, ξ_T, ξ_E, ξ_TE, ψ
#   α_T, α_E, α_TE ≥ 0  (positive consumption utility)
#   μ ≥ 0                (reinforcement is positive)
#   γ ≤ 0                (withdrawal cost is negative)
#   ω ≤ 0                (expenditure reduces utility)
#   ψ ∈ (0.01, 0.99)     (addiction decay rate)
#   λ, ξ unconstrained
const θ_lower_bound = Float64[  0,   0,   0, -Inf, -Inf,   0, -Inf, -Inf, -Inf, -Inf, -Inf, 0.01]
const θ_upper_bound = Float64[Inf, Inf, Inf,  Inf,  Inf, Inf,    0,    0,  Inf,  Inf,  Inf, 0.99]

"""
Check whether θ_vec satisfies the economic parameter bounds.
Returns (in_bounds::Bool, violations::String).
"""
function check_parameter_bounds(θ_vec::AbstractVector{<:Real}, param_names_vec::AbstractVector{<:AbstractString})
    violated = String[]
    for i in 1:length(θ_vec)
        pname = length(param_names_vec) >= i ? param_names_vec[i] : "[$i]"
        if θ_vec[i] < θ_lower_bound[i]
            push!(violated, "$pname=$(round(θ_vec[i], digits=4))<$(θ_lower_bound[i])")
        elseif θ_vec[i] > θ_upper_bound[i]
            push!(violated, "$pname=$(round(θ_vec[i], digits=4))>$(θ_upper_bound[i])")
        end
    end
    return isempty(violated), join(violated, ", ")
end


#############################
# Objective Function
#############################

"""
Objective function for the optimizer.

Takes a parameter vector θ_vec (12 elements: 11 structural + ψ), recomputes
the addiction grid, flow utility, and value function, evaluates the
log-likelihood, and returns the negative log-likelihood.

ψ is the last element of θ_vec. The first 11 elements are passed to
get_flow_utility (which does not use ψ directly).

Accesses global data loaded by 02_Estimation.jl (e.g., N_J, y, etc.)
so each data does not need to be passed as arguments.
"""
function objective(θ_vec::AbstractVector{<:Real})
    global est_eval_count

    # Start evaluation time
    t_eval = time()

    # Update evaluation count
    est_eval_count += 1

    # Economic parameter bounds: α_T,α_E,α_TE,μ ≥ 0; γ,ω ≤ 0; ψ ∈ (0.01, 0.99).
    # Return penalty without solving VFI to save time.
    in_bounds, violations = check_parameter_bounds(θ_vec, est_param_names)
    if !in_bounds
        elapsed = time() - t_eval
        est_log("")
        est_log(@sprintf("  Eval %d | PENALTY (bounds: %s) | time = %.1fs",
            est_eval_count, violations, elapsed))
        est_log("")
        return 1e14
    end

    # Extract ψ from the parameter vector (last element)
    ψ_current = θ_vec[end]

    # Recompute addiction grid for the current ψ (grid spans [0, 1/ψ])
    N_A_current, A_current = get_addiction_space(ψ_current)

    # Recompute flow utility for the current θ (first 11 elements)
    U_current = get_flow_utility(
        θ_vec[1:end-1], N_J, N_A_current, N_Pcomb, A_current, c_cig, c_ecig, c_bundle, n, is_flavored, cat_idx, E
    )

    # Recompute addiction transition brackets for the current ψ and A
    a_lower_current, a_upper_current, a_weight_current = precompute_addiction_transitions(
        N_J, N_A_current, ψ_current, A_current, n
    )

    # Solve VFI (each evaluation starts fresh from zeros)
    V, V_decision_current, vfi_iters, vfi_converged = solve_vfi(
        N_J, N_A_current, N_P, N_Pcomb, β, δ, U_current,
        a_lower_current, a_upper_current, a_weight_current,
        p_cig_lo, p_cig_hi, p_cig_w,
        p_ecig_lo, p_ecig_hi, p_ecig_w
    )

    # Early-exit: if VFI did not converge, skip LL and return penalty
    if !vfi_converged
        elapsed = time() - t_eval
        est_log("")
        est_log(@sprintf("  Eval %d | PENALTY (VFI not converged) | VFI iters = %d | time = %.1fs",
            est_eval_count, vfi_iters, elapsed))
        est_log(@sprintf("    θ for Eval %d:", est_eval_count))
        for (i, x) in enumerate(θ_vec)
            pname = length(est_param_names) >= i ? est_param_names[i] : "[$i]"
            est_log(@sprintf("      %s = %.6f", pname, x))
        end
        est_log("")
        return 1e14
    end

    # Recompute addiction trajectories for the current ψ and A
    a0_current, _ = get_initial_addiction_stock(ψ_current, A_current, n, y, hh_codes)
    _, a_continuous_current = simulate_addiction_trajectories(N_A_current, ψ_current, A_current, n, y, hh_codes, a0_current)

    # Compute log-likelihood via trilinear interpolation at continuous states
    LL = log_likelihood(
        V_decision_current, N_J, N_P, A_current, P, y, tya_state, a_continuous_current, p_continuous
    )

    elapsed = time() - t_eval
    est_log("")
    est_log(@sprintf("  Eval %d | LL = %.4f | VFI iters = %d | time = %.1fs",
        est_eval_count, LL, vfi_iters, elapsed))
    est_log(@sprintf("    θ for Eval %d:", est_eval_count))
    for (i, x) in enumerate(θ_vec)
        pname = length(est_param_names) >= i ? est_param_names[i] : "[$i]"
        est_log(@sprintf("      %s = %.6f", pname, x))
    end
    est_log("")

    # Return negative log-likelihood (optimizer minimizes)
    return -LL
end
