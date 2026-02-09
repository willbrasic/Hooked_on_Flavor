################################################################################
# William Brasic
# The University of Arizona
# wbrasic97@gmail.com
# February 2026
#
# This script creates functions specific to the model validation exercise.
# These functions compute predicted category-level shares from model choice
# probabilities and compare them to actual observed shares in the data.
################################################################################


#############################
# Logging
#############################

# Global log file handle (set by 02_Model_Validation.jl before any logging)
val_log_io = nothing

"""
Write a message to the validation log file and flush immediately.
Also prints to stdout as a fallback.
"""
function val_log(msg::String)
    global val_log_io
    println(msg)
    if val_log_io !== nothing
        println(val_log_io, msg)
        flush(val_log_io)
    end
end


#############################
# Predicted Choice
# Probabilities
#############################

"""
Compute predicted choice probabilities at all observed states under a given
V_choice solution.

For each observation i with state (tya_i, a_i, p_cig_i, p_ecig_i):
  1. Interpolate V_choice at the continuous state for all N_J alternatives
  2. Compute softmax choice probabilities

Returns:
- probs: Matrix{Float64} of size (N_obs × N_J), choice probabilities
"""
function compute_predicted_probs(
    V_choice::Array{Float64, 4},
    tya_state::AbstractVector{<:Integer},
    a_continuous::AbstractVector{<:Real},
    p_continuous::AbstractMatrix{<:Real},
    N_J::Integer,
    N_P::Integer,
    A::AbstractVector{<:Real},
    P::AbstractMatrix{<:Real}
)

    N_obs = length(tya_state)

    # Initialize output
    probs = Matrix{Float64}(undef, N_obs, N_J)

    for i in 1:N_obs

        # Interpolate V_choice at continuous state
        v_interp = interpolate_v_choice(
            V_choice, tya_state[i], a_continuous[i],
            p_continuous[i, 1], p_continuous[i, 2],
            N_J, N_P, A, P
        )

        # Fix NaN from 0.0 * (-Inf) in IEEE 754 arithmetic.
        # Banned alternatives have V_choice = -Inf at all grid points;
        # when an interpolation weight is exactly 0, 0.0 * (-Inf) = NaN.
        # Replace with -Inf so exp(-Inf) = 0 in subsequent softmax/logsumexp.
        replace!(v_interp, NaN => -Inf)

        # Softmax choice probabilities
        v_max = maximum(v_interp)
        v_shifted = v_interp .- v_max
        exp_v = exp.(v_shifted)
        sum_exp_v = sum(exp_v)
        for j in 1:N_J
            probs[i, j] = exp_v[j] / sum_exp_v
        end
    end

    return probs
end


#############################
# Category-Level Aggregation
#############################

"""
Aggregate alternative-level predicted probabilities into category-level shares.

Category mapping:
  0 = outside option, 1 = cigarettes, 2 = original e-cig,
  3 = flavored e-cig, 4 = original bundle, 5 = flavored bundle

For actual shares, computes the fraction of observations choosing each category.
For predicted shares, computes the mean predicted probability for each category.

Returns:
- actual_shares:    Vector{Float64} of length 6 (categories 0-5)
- predicted_shares: Vector{Float64} of length 6 (categories 0-5)
"""
function compute_category_shares(
    y::AbstractVector{<:Integer},
    probs::AbstractMatrix{<:Real},
    cat_idx::AbstractVector{<:Integer},
    obs_mask::AbstractVector{Bool}
)

    N_cats = 6  # categories 0-5

    # Subset observations
    y_sub = y[obs_mask]
    probs_sub = probs[obs_mask, :]
    N_sub = length(y_sub)

    # Actual shares: fraction of observations choosing each category
    actual_shares = zeros(Float64, N_cats)
    for i in 1:N_sub
        cat = cat_idx[y_sub[i]]
        actual_shares[cat + 1] += 1.0
    end
    actual_shares ./= N_sub

    # Predicted shares: mean predicted probability for each category
    predicted_shares = zeros(Float64, N_cats)
    N_J = size(probs_sub, 2)
    for j in 1:N_J
        cat = cat_idx[j]
        predicted_shares[cat + 1] += mean(probs_sub[:, j])
    end

    return actual_shares, predicted_shares
end


#############################
# Alternative-Level
# Aggregation
#############################

"""
Compute actual and predicted shares at the alternative level (all 21 alternatives).

Returns:
- actual_shares:    Vector{Float64} of length N_J
- predicted_shares: Vector{Float64} of length N_J
"""
function compute_alternative_shares(
    y::AbstractVector{<:Integer},
    probs::AbstractMatrix{<:Real},
    N_J::Integer,
    obs_mask::AbstractVector{Bool}
)

    y_sub = y[obs_mask]
    probs_sub = probs[obs_mask, :]
    N_sub = length(y_sub)

    # Actual shares
    actual_shares = zeros(Float64, N_J)
    for i in 1:N_sub
        actual_shares[y_sub[i]] += 1.0
    end
    actual_shares ./= N_sub

    # Predicted shares
    predicted_shares = vec(mean(probs_sub, dims=1))

    return actual_shares, predicted_shares
end


#############################
# Time Period Extraction
#############################

"""
Extract purchase months from Teen_Young_Adult.csv and convert to integer
period indices (1, 2, 3, ...) for time-series aggregation.

Returns:
- period_idx:    Vector{Int} of period indices for each observation
- period_labels: Vector{String} of calendar month labels (e.g., "2021-01")
- N_periods:     Number of unique periods
"""
function get_period_indices(;
    file_name::AbstractString = "./Teen_Young_Adult.csv"
)

    df = CSV.read(file_name, DataFrame)

    # Extract purchase month strings
    months_raw = string.(df.purchase_month)

    # Get unique months in sorted order (YYYY-MM-DD format sorts correctly)
    unique_months = sort(unique(months_raw))
    N_periods = length(unique_months)

    # Create month → index mapping
    month_to_idx = Dict{String, Int}()
    for (idx, m) in enumerate(unique_months)
        month_to_idx[m] = idx
    end

    # Map each observation to its period index
    period_idx = [month_to_idx[m] for m in months_raw]

    # Clean labels (YYYY-MM format)
    period_labels = [m[1:7] for m in unique_months]

    return period_idx, period_labels, N_periods
end
