################################################################################
# William Brasic
# The University of Arizona
# wbrasic97@gmail.com
# February 2026
#
# This script creates functions specific to the counterfactual flavor ban
# simulation. These functions handle applying the ban to flow utilities,
# computing pointwise outcomes (choice probabilities and welfare) at observed
# states, forward-simulating households under status quo and ban scenarios,
# and aggregating simulation results.
################################################################################


#############################
# Logging
#############################

# Uses log_io and log_msg() from 01_Functions.jl (included before this file).


#############################
# Flavor Ban
#############################

"""
Apply a flavor ban by setting flow utility to -Inf for all flavored alternatives.

Flavored alternatives are those with cat_idx[j] ∈ {3, 4, 6, 7}:
  - cat_idx = 3: non-FDA flavored e-cigarettes
  - cat_idx = 4: FDA flavored e-cigarettes
  - cat_idx = 6: bundles with non-FDA flavored e-cig
  - cat_idx = 7: bundles with FDA flavored e-cig

Setting U = -Inf means exp(U) = 0 in the logsumexp, so banned alternatives
get zero choice probability. VFI contraction property is maintained because
the effective choice set simply shrinks.

Operates in-place on U_ban (which should be a copy of the status quo U).
"""
function apply_flavor_ban!(
    U_ban::Array{Float64, 4},
    cat_idx::AbstractVector{<:Integer}
)

    N_J = size(U_ban, 2)

    for j in 1:N_J
        if cat_idx[j] in (3, 4, 6, 7)
            U_ban[:, j, :, :] .= -Inf
        end
    end

    return nothing
end


#############################
# Pointwise Outcomes
#############################

"""
Compute choice probabilities and welfare at all observed states under a given
V_choice solution.

For each observation i with state (tya_i, a_i, p_cig_i, p_ecig_i):
  1. Interpolate V_choice at the continuous state for all N_J alternatives
  2. Compute softmax choice probabilities
  3. Compute welfare as logsumexp(V_choice_interp) (expected maximum utility)

Returns:
- probs:   Matrix{Float64} of size (N_obs × N_J), choice probabilities
- welfare: Vector{Float64} of length N_obs, welfare (inclusive value)
"""
function compute_pointwise_outcomes(
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

    # Initialize output arrays
    probs   = Matrix{Float64}(undef, N_obs, N_J)
    welfare = Vector{Float64}(undef, N_obs)

    # Compute outcomes for each observation
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

        # Welfare: logsumexp gives the expected maximum utility (inclusive value)
        welfare[i] = logsumexp(v_interp)

        # Softmax choice probabilities
        v_max = maximum(v_interp)
        v_shifted = v_interp .- v_max
        exp_v = exp.(v_shifted)
        sum_exp_v = sum(exp_v)
        for j in 1:N_J
            probs[i, j] = exp_v[j] / sum_exp_v
        end
    end

    return probs, welfare
end


#############################
# Forward Simulation
#############################

"""
Forward-simulate households from their last observed states under a given
V_choice solution.

For each household h with initial state (tya_h, a_h, p_cig_h, p_ecig_h):
  For each Monte Carlo draw d = 1, ..., N_draws:
    For each period t = 1, ..., T_sim:
      1. Interpolate V_choice at continuous state
      2. Compute softmax choice probabilities
      3. Draw choice from categorical distribution
      4. Update addiction: ã' = (1-ψ)ã + ψ·n[j]
      5. Update prices: p' = φ₀ + φ₁·p + L_chol·ε

Returns:
- sim_choices:   Array{Int, 3} of size (N_HH, T_sim, N_draws)
- sim_addiction: Array{Float64, 3} of size (N_HH, T_sim, N_draws), addiction AFTER choice
- sim_welfare:   Array{Float64, 3} of size (N_HH, T_sim, N_draws), welfare at state BEFORE choice
"""
function simulate_trajectories(
    V_choice::Array{Float64, 4},
    hh_tya::AbstractVector{<:Integer},
    hh_a0::AbstractVector{<:Real},
    hh_p0::AbstractMatrix{<:Real},
    T_sim::Integer,
    N_draws::Integer,
    ψ::Real,
    N_J::Integer,
    N_P::Integer,
    A::AbstractVector{<:Real},
    P::AbstractMatrix{<:Real},
    n::AbstractVector{<:Real},
    φ_0::AbstractVector{<:Real},
    φ_1::AbstractVector{<:Real},
    L_chol::AbstractMatrix{<:Real}
)

    N_HH = length(hh_tya)

    # Price grid bounds for clamping
    P_cig  = P[:, 1]
    P_ecig = P[:, 2]

    # Initialize output arrays
    sim_choices   = Array{Int}(undef, N_HH, T_sim, N_draws)
    sim_addiction = Array{Float64}(undef, N_HH, T_sim, N_draws)
    sim_welfare   = Array{Float64}(undef, N_HH, T_sim, N_draws)

    # Simulate each household
    for h in 1:N_HH

        tya_idx_h = hh_tya[h]

        for d in 1:N_draws

            # Reset to initial state for each draw
            a_h      = hh_a0[h]
            p_cig_h  = hh_p0[h, 1]
            p_ecig_h = hh_p0[h, 2]

            for t in 1:T_sim

                # Interpolate V_choice at continuous state
                v_interp = interpolate_v_choice(
                    V_choice, tya_idx_h, a_h, p_cig_h, p_ecig_h,
                    N_J, N_P, A, P
                )

                # Fix NaN from 0.0 * (-Inf) in IEEE 754 arithmetic (see comment
                # in compute_pointwise_outcomes for full explanation)
                replace!(v_interp, NaN => -Inf)

                # Welfare at current state (before choice)
                sim_welfare[h, t, d] = logsumexp(v_interp)

                # Softmax choice probabilities
                v_max = maximum(v_interp)
                v_shifted = v_interp .- v_max
                exp_v = exp.(v_shifted)
                probs = exp_v ./ sum(exp_v)

                # Draw choice
                j = categorical_sample(probs)
                sim_choices[h, t, d] = j

                # Update addiction: ã' = (1-ψ)ã + ψ·n[j]
                # a_h is ã (normalized addiction); n[j] is n_std (standardized nicotine)
                a_h = (1 - ψ) * a_h + ψ * n[j]
                a_h = clamp(a_h, A[1], A[end])
                sim_addiction[h, t, d] = a_h

                # Update prices via AR(1) with correlated shocks
                ε = L_chol * randn(2)
                p_cig_h  = clamp(φ_0[1] + φ_1[1] * p_cig_h  + ε[1], P_cig[1], P_cig[end])
                p_ecig_h = clamp(φ_0[2] + φ_1[2] * p_ecig_h + ε[2], P_ecig[1], P_ecig[end])
            end
        end
    end

    return sim_choices, sim_addiction, sim_welfare
end


#############################
# Aggregation
#############################

"""
Aggregate simulation results into period-by-period category shares, mean
addiction, and mean welfare.

Category mapping:
  0 = outside option, 1 = cigarettes, 2 = original e-cig,
  3 = non-FDA flavored e-cig, 4 = FDA flavored e-cig,
  5 = original bundle, 6 = non-FDA flavored bundle, 7 = FDA flavored bundle

Returns:
- DataFrame with columns: period, share_outside, share_cig, share_orig_ecig,
  share_non_fda_flav_ecig, share_fda_flav_ecig, share_orig_bundle,
  share_non_fda_flav_bundle, share_fda_flav_bundle, mean_addiction, mean_welfare
"""
function aggregate_simulation(
    sim_choices::Array{Int, 3},
    sim_addiction::Array{Float64, 3},
    sim_welfare::Array{Float64, 3},
    cat_idx::AbstractVector{<:Integer},
    N_J::Integer,
    T_sim::Integer
)

    N_HH, _, N_draws = size(sim_choices)
    N_total = N_HH * N_draws

    # Category names for output
    cat_names = ["share_outside", "share_cig", "share_orig_ecig",
                 "share_non_fda_flav_ecig", "share_fda_flav_ecig",
                 "share_orig_bundle", "share_non_fda_flav_bundle", "share_fda_flav_bundle"]

    # Initialize result storage
    results = DataFrame(
        period                    = 1:T_sim,
        share_outside             = zeros(Float64, T_sim),
        share_cig                 = zeros(Float64, T_sim),
        share_orig_ecig           = zeros(Float64, T_sim),
        share_non_fda_flav_ecig   = zeros(Float64, T_sim),
        share_fda_flav_ecig       = zeros(Float64, T_sim),
        share_orig_bundle         = zeros(Float64, T_sim),
        share_non_fda_flav_bundle = zeros(Float64, T_sim),
        share_fda_flav_bundle     = zeros(Float64, T_sim),
        mean_addiction            = zeros(Float64, T_sim),
        mean_welfare              = zeros(Float64, T_sim)
    )

    for t in 1:T_sim

        # Count category shares across all households and draws
        cat_counts = zeros(Float64, 8)  # cats 0-7
        total_addiction = 0.0
        total_welfare   = 0.0

        for h in 1:N_HH
            for d in 1:N_draws
                j = sim_choices[h, t, d]
                cat = cat_idx[j]
                cat_counts[cat + 1] += 1.0  # +1 because cat=0 maps to index 1

                total_addiction += sim_addiction[h, t, d]
                total_welfare   += sim_welfare[h, t, d]
            end
        end

        # Compute shares and means
        results.share_outside[t]             = cat_counts[1] / N_total
        results.share_cig[t]                 = cat_counts[2] / N_total
        results.share_orig_ecig[t]           = cat_counts[3] / N_total
        results.share_non_fda_flav_ecig[t]   = cat_counts[4] / N_total
        results.share_fda_flav_ecig[t]       = cat_counts[5] / N_total
        results.share_orig_bundle[t]         = cat_counts[6] / N_total
        results.share_non_fda_flav_bundle[t] = cat_counts[7] / N_total
        results.share_fda_flav_bundle[t]     = cat_counts[8] / N_total
        results.mean_addiction[t]            = total_addiction / N_total
        results.mean_welfare[t]              = total_welfare / N_total
    end

    return results
end
