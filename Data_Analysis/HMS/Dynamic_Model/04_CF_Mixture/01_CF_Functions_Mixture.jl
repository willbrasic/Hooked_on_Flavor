################################################################################
# William Brasic
# The University of Arizona
# wbrasic97@gmail.com
# June 2026
#
# This script creates functions specific to the counterfactual flavor ban
# simulation. Functions are ordered by their execution sequence in
# 02_CF_Mixture.jl and handle:
#   - Applying bans and taxes to flow utilities
#   - Interpolating V_choice at continuous states
#   - Computing pointwise choice probabilities and welfare at observed states
#   - Aggregating forward-simulation results
################################################################################


################################################################################
# Table of Contents
#
#  1. Flavor Bans         — Zero out flow utility for banned categories.
#                           Comprehensive, FDA-only, and non-FDA-only variants.
#
#  2. Flavor Tax          — Shift flow utility down for taxed categories.
#                           Same three variants as the bans.
#
#  3. Pointwise Outcomes  — Interpolate a single type's V_choice at each
#                           observed state, then compute choice probabilities
#                           and welfare. 
#
#  4. Aggregation         — Average Monte Carlo paths into period-by-period
#                           category shares, mean addiction, and mean welfare.
#                           Subgroup variants stratify by TYA status and modal
#                           latent type. A separate
#                           extensive-margin function tracks cumulative
#                           flavored initiation among baseline non-users.
################################################################################


#############################
# 1. Flavor Bans
#############################

# --- Comprehensive Ban ---

"""
Comprehensive flavor ban: set flow utility to -Inf for all flavored
alternatives (cat_idx in {3, 4, 6, 7}), so they get zero choice probability.
"""
function apply_flavor_ban!(
    U_ban::Array{Float64, 6},
    cat_idx::AbstractVector{<:Integer}
)

    N_J = size(U_ban, 2)

    for j in 1:N_J
        if cat_idx[j] in (3, 4, 6, 7)
            U_ban[:, j, :, :, :, :] .= -Inf
        end
    end

    return nothing
end


# --- FDA-Only Ban ---

"""
FDA-only flavor ban: set flow utility to -Inf for FDA-authorized flavored
alternatives only (cat_idx in {4, 7}). Non-FDA flavored alternatives
(cat_idx 3, 6) remain available.
"""
function apply_fda_flavor_ban!(
    U_ban::Array{Float64, 6},
    cat_idx::AbstractVector{<:Integer}
)

    N_J = size(U_ban, 2)

    for j in 1:N_J
        if cat_idx[j] in (4, 7)
            U_ban[:, j, :, :, :, :] .= -Inf
        end
    end

    return nothing
end


# --- Non-FDA Ban ---

"""
Non-FDA ban: set flow utility to -Inf for non-FDA-authorized flavored
alternatives only (cat_idx in {3, 6}, no PMTA approval). FDA-authorized
flavored alternatives (cat_idx 4, 7) remain available. Models FDA enforcement
against unauthorized products while leaving PMTA-approved products intact.
"""
function apply_non_fda_ban!(
    U_ban::Array{Float64, 6},
    cat_idx::AbstractVector{<:Integer}
)

    N_J = size(U_ban, 2)

    for j in 1:N_J
        if cat_idx[j] in (3, 6)
            U_ban[:, j, :, :, :, :] .= -Inf
        end
    end

    return nothing
end


#############################
# 2. Flavor Tax
#############################

# --- All Flavored Products ---

"""
Apply a per-unit tax on flavored e-cig alternatives by shifting flow utility
downward: Δu[j] = ω_E * τ * q_ecig[j] for cat_idx[j] in {3, 4, 6, 7}. Since
ω_E < 0 and τ > 0, this makes flavored alternatives less attractive relative
to unflavored products and the outside option.

This is algebraically equivalent to raising the market price P_ecig[p] by τ
inside the existing price term ω_E * P_ecig[p] * q_ecig[j], so that term
doesn't need to be touched separately. Applied uniformly across all price
states since it's a per-unit excise tax, independent of the realized retail
price.
"""
function apply_flavor_tax!(
    U_tax::Array{Float64, 6},
    cat_idx::AbstractVector{<:Integer},
    omega_E::Real,
    q_ecig::AbstractVector{<:Real},
    q_ecig_max::Real,
    tau::Real
)

    N_J = size(U_tax, 2)

    for j in 1:N_J
        if cat_idx[j] in (3, 4, 6, 7)
            delta_u = omega_E * tau * q_ecig[j]
            U_tax[:, j, :, :, :, :] .+= delta_u
        end
    end

    return nothing
end


#############################
# 3. Pointwise Outcomes
#############################

"""
Compute choice probabilities and welfare at all observed states under a single
type's V_choice solution. Called six times per counterfactual loop (3 types ×
SQ/CF); 02_CF_Mixture.jl then combines the per-type results into the
mixture-weighted outcome using posterior type weights from Bayes' rule.
"""
function compute_pointwise_outcomes(
    V_choice::Array{Float64, 6},
    tya_state::AbstractVector{<:Integer},
    af_continuous::AbstractVector{<:Real},
    as_continuous::AbstractVector{<:Real},
    aflav_continuous::AbstractVector{<:Real},
    p_continuous::AbstractMatrix{<:Real},
    N_J::Integer,
    N_P::Integer,
    A_f::AbstractVector{<:Real},
    A_s::AbstractVector{<:Real},
    A_flav::AbstractVector{<:Real},
    P::AbstractMatrix{<:Real}
)

    N_obs = length(tya_state)

    probs   = Matrix{Float64}(undef, N_obs, N_J)
    welfare = Vector{Float64}(undef, N_obs)

    for i in 1:N_obs

        v_interp = interpolate_v_choice(
            V_choice, tya_state[i], af_continuous[i], as_continuous[i],
            aflav_continuous[i], p_continuous[i, 1], p_continuous[i, 2],
            N_J, N_P, A_f, A_s, A_flav, P
        )

        # Banned alternatives have V_choice = -Inf everywhere; if an interpolation
        # weight is exactly 0, 0.0 * (-Inf) = NaN. Replace with -Inf.
        replace!(v_interp, NaN => -Inf)

        # logsumexp gives the expected maximum utility (inclusive value)
        welfare[i] = logsumexp(v_interp)

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
# 4. Aggregation
#############################

# --- Overall ---

"""
Aggregate simulation results into period-by-period category shares, mean
addiction, and mean welfare.

Category mapping: 0 = outside, 1 = cig, 2 = orig ecig, 3 = non-FDA flav ecig,
4 = FDA flav ecig, 5 = orig bundle, 6 = non-FDA flav bundle, 7 = FDA flav bundle
"""
function aggregate_simulation(
    sim_choices::Array{Int, 3},
    sim_addiction::Array{Float64, 3},
    sim_aflav::Array{Float64, 3},
    sim_welfare::Array{Float64, 3},
    sim_q_cig::Array{Float64, 3},
    sim_q_orig_ecig::Array{Float64, 3},
    sim_q_flav_ecig::Array{Float64, 3},
    cat_idx::AbstractVector{<:Integer},
    N_J::Integer,
    T_sim::Integer
)

    N_HH, _, N_draws = size(sim_choices)

    # each household contributes N_draws Monte Carlo paths
    N_total = N_HH * N_draws

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
        mean_aflav                = zeros(Float64, T_sim),
        mean_welfare              = zeros(Float64, T_sim),
        mean_q_cig                = zeros(Float64, T_sim),
        mean_q_orig_ecig          = zeros(Float64, T_sim),
        mean_q_flav_ecig          = zeros(Float64, T_sim)
    )

    for t in 1:T_sim

        cat_counts = zeros(Float64, 8)
        total_addiction    = 0.0
        total_aflav        = 0.0
        total_welfare      = 0.0
        total_q_cig        = 0.0
        total_q_orig_ecig  = 0.0
        total_q_flav_ecig  = 0.0

        for h in 1:N_HH
            for d in 1:N_draws
                j   = sim_choices[h, t, d]   # chosen alternative index 
                cat = cat_idx[j]              # product category 
                cat_counts[cat + 1] += 1.0

                total_addiction   += sim_addiction[h, t, d]
                total_aflav       += sim_aflav[h, t, d]
                total_welfare     += sim_welfare[h, t, d]
                total_q_cig       += sim_q_cig[h, t, d]
                total_q_orig_ecig += sim_q_orig_ecig[h, t, d]
                total_q_flav_ecig += sim_q_flav_ecig[h, t, d]
            end
        end

        results.share_outside[t]             = cat_counts[1] / N_total
        results.share_cig[t]                 = cat_counts[2] / N_total
        results.share_orig_ecig[t]           = cat_counts[3] / N_total
        results.share_non_fda_flav_ecig[t]   = cat_counts[4] / N_total
        results.share_fda_flav_ecig[t]       = cat_counts[5] / N_total
        results.share_orig_bundle[t]         = cat_counts[6] / N_total
        results.share_non_fda_flav_bundle[t] = cat_counts[7] / N_total
        results.share_fda_flav_bundle[t]     = cat_counts[8] / N_total
        results.mean_addiction[t]            = total_addiction   / N_total
        results.mean_aflav[t]               = total_aflav        / N_total
        results.mean_welfare[t]              = total_welfare      / N_total
        results.mean_q_cig[t]                = total_q_cig        / N_total
        results.mean_q_orig_ecig[t]           = total_q_orig_ecig / N_total
        results.mean_q_flav_ecig[t]           = total_q_flav_ecig / N_total
    end

    return results
end


# --- By TYA Status ---

"""
Aggregate simulation results for TYA-present households (hh_tya_terminal == 2).
Same column structure as aggregate_simulation.
"""
function aggregate_simulation_by_tya(
    sim_choices::Array{Int, 3},
    sim_addiction::Array{Float64, 3},
    sim_aflav::Array{Float64, 3},
    sim_welfare::Array{Float64, 3},
    sim_q_cig::Array{Float64, 3},
    sim_q_orig_ecig::Array{Float64, 3},
    sim_q_flav_ecig::Array{Float64, 3},
    cat_idx::AbstractVector{<:Integer},
    N_J::Integer,
    T_sim::Integer,
    hh_tya_terminal::Vector{Int},
    N_draws::Integer
)

    N_HH = size(sim_choices, 1)

    idx_tya = findall(h -> hh_tya_terminal[h] == 2, 1:N_HH)

    function _aggregate_subgroup(hh_indices)

        N_sub = length(hh_indices)
        # each household contributes N_draws Monte Carlo paths
        N_total = N_sub * N_draws

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
            mean_aflav                = zeros(Float64, T_sim),
            mean_welfare              = zeros(Float64, T_sim),
            mean_q_cig                = zeros(Float64, T_sim),
            mean_q_orig_ecig          = zeros(Float64, T_sim),
            mean_q_flav_ecig          = zeros(Float64, T_sim)
        )

        for t in 1:T_sim

            cat_counts = zeros(Float64, 8)  # cats 0-7
            total_addiction    = 0.0
            total_aflav        = 0.0
            total_welfare      = 0.0
            total_q_cig        = 0.0
            total_q_orig_ecig  = 0.0
            total_q_flav_ecig  = 0.0

            for h in hh_indices
                for d in 1:N_draws
                    j = sim_choices[h, t, d]
                    cat = cat_idx[j]
                    cat_counts[cat + 1] += 1.0  # +1 because cat=0 maps to index 1

                    total_addiction   += sim_addiction[h, t, d]
                    total_aflav       += sim_aflav[h, t, d]
                    total_welfare     += sim_welfare[h, t, d]
                    total_q_cig       += sim_q_cig[h, t, d]
                    total_q_orig_ecig += sim_q_orig_ecig[h, t, d]
                    total_q_flav_ecig += sim_q_flav_ecig[h, t, d]
                end
            end

            results.share_outside[t]             = cat_counts[1] / N_total
            results.share_cig[t]                 = cat_counts[2] / N_total
            results.share_orig_ecig[t]           = cat_counts[3] / N_total
            results.share_non_fda_flav_ecig[t]   = cat_counts[4] / N_total
            results.share_fda_flav_ecig[t]       = cat_counts[5] / N_total
            results.share_orig_bundle[t]         = cat_counts[6] / N_total
            results.share_non_fda_flav_bundle[t] = cat_counts[7] / N_total
            results.share_fda_flav_bundle[t]     = cat_counts[8] / N_total
            results.mean_addiction[t]            = total_addiction   / N_total
            results.mean_aflav[t]               = total_aflav        / N_total
            results.mean_welfare[t]              = total_welfare      / N_total
            results.mean_q_cig[t]                = total_q_cig        / N_total
            results.mean_q_orig_ecig[t]           = total_q_orig_ecig / N_total
            results.mean_q_flav_ecig[t]           = total_q_flav_ecig / N_total
        end

        return results
    end

    df_tya = _aggregate_subgroup(idx_tya)

    return df_tya
end


# --- By Latent Type ---

"""
Aggregate simulation results by modal latent type (K=3 mixture), households
assigned via argmax of posterior type probabilities. Same column structure
as aggregate_simulation.
"""
function aggregate_simulation_by_type_k3(
    sim_choices::Array{Int, 3},
    sim_addiction::Array{Float64, 3},
    sim_aflav::Array{Float64, 3},
    sim_welfare::Array{Float64, 3},
    sim_q_cig::Array{Float64, 3},
    sim_q_orig_ecig::Array{Float64, 3},
    sim_q_flav_ecig::Array{Float64, 3},
    cat_idx::AbstractVector{<:Integer},
    N_J::Integer,
    T_sim::Integer,
    hh_posterior_type1::Vector{Float64},
    hh_posterior_type2::Vector{Float64},
    hh_posterior_type3::Vector{Float64},
    N_draws::Integer
)

    N_HH = size(sim_choices, 1)

    # Modal type assignment via argmax of posterior probabilities
    idx_type1 = findall(h -> hh_posterior_type1[h] >= hh_posterior_type2[h] && hh_posterior_type1[h] >= hh_posterior_type3[h], 1:N_HH)
    idx_type2 = findall(h -> hh_posterior_type2[h] >  hh_posterior_type1[h] && hh_posterior_type2[h] >= hh_posterior_type3[h], 1:N_HH)
    idx_type3 = findall(h -> hh_posterior_type3[h] >  hh_posterior_type1[h] && hh_posterior_type3[h] >  hh_posterior_type2[h], 1:N_HH)

    function _aggregate_subgroup(hh_indices)

        N_sub = length(hh_indices)
        # max(..., 1) guards against division by zero when a type has no households
        N_total = max(N_sub * N_draws, 1)

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
            mean_aflav                = zeros(Float64, T_sim),
            mean_welfare              = zeros(Float64, T_sim),
            mean_q_cig                = zeros(Float64, T_sim),
            mean_q_orig_ecig          = zeros(Float64, T_sim),
            mean_q_flav_ecig          = zeros(Float64, T_sim)
        )

        # Return the zero-filled DataFrame without entering the period loop.
        if N_sub == 0
            return results
        end

        for t in 1:T_sim

            cat_counts = zeros(Float64, 8)
            total_addiction    = 0.0
            total_aflav        = 0.0
            total_welfare      = 0.0
            total_q_cig        = 0.0
            total_q_orig_ecig  = 0.0
            total_q_flav_ecig  = 0.0

            for h in hh_indices
                for d in 1:N_draws
                    j   = sim_choices[h, t, d]   # chosen alternative index 
                    cat = cat_idx[j]              # product category 
                    cat_counts[cat + 1] += 1.0

                    total_addiction   += sim_addiction[h, t, d]
                    total_aflav       += sim_aflav[h, t, d]
                    total_welfare     += sim_welfare[h, t, d]
                    total_q_cig       += sim_q_cig[h, t, d]
                    total_q_orig_ecig += sim_q_orig_ecig[h, t, d]
                    total_q_flav_ecig += sim_q_flav_ecig[h, t, d]
                end
            end

            results.share_outside[t]             = cat_counts[1] / N_total
            results.share_cig[t]                 = cat_counts[2] / N_total
            results.share_orig_ecig[t]           = cat_counts[3] / N_total
            results.share_non_fda_flav_ecig[t]   = cat_counts[4] / N_total
            results.share_fda_flav_ecig[t]       = cat_counts[5] / N_total
            results.share_orig_bundle[t]         = cat_counts[6] / N_total
            results.share_non_fda_flav_bundle[t] = cat_counts[7] / N_total
            results.share_fda_flav_bundle[t]     = cat_counts[8] / N_total
            results.mean_addiction[t]            = total_addiction   / N_total
            results.mean_aflav[t]               = total_aflav        / N_total
            results.mean_welfare[t]              = total_welfare      / N_total
            results.mean_q_cig[t]                = total_q_cig        / N_total
            results.mean_q_orig_ecig[t]           = total_q_orig_ecig / N_total
            results.mean_q_flav_ecig[t]           = total_q_flav_ecig / N_total
        end

        return results
    end

    df_type1 = _aggregate_subgroup(idx_type1)
    df_type2 = _aggregate_subgroup(idx_type2)
    df_type3 = _aggregate_subgroup(idx_type3)

    return df_type1, df_type2, df_type3
end


# --- By TYA Status and Latent Type ---

"""
Aggregate simulation results for TYA-present households, further split by
modal latent type. Same column structure as aggregate_simulation.
"""
function aggregate_simulation_by_tya_type_k3(
    sim_choices::Array{Int, 3},
    sim_addiction::Array{Float64, 3},
    sim_aflav::Array{Float64, 3},
    sim_welfare::Array{Float64, 3},
    sim_q_cig::Array{Float64, 3},
    sim_q_orig_ecig::Array{Float64, 3},
    sim_q_flav_ecig::Array{Float64, 3},
    cat_idx::AbstractVector{<:Integer},
    N_J::Integer,
    T_sim::Integer,
    hh_tya_terminal::Vector{Int},
    hh_posterior_type1::Vector{Float64},
    hh_posterior_type2::Vector{Float64},
    hh_posterior_type3::Vector{Float64},
    N_draws::Integer
)

    N_HH = size(sim_choices, 1)

    # Restrict to TYA-present households, then assign modal type via argmax of posterior probabilities
    idx_tya_type1 = findall(h -> hh_tya_terminal[h] == 2 && hh_posterior_type1[h] >= hh_posterior_type2[h] && hh_posterior_type1[h] >= hh_posterior_type3[h], 1:N_HH)
    idx_tya_type2 = findall(h -> hh_tya_terminal[h] == 2 && hh_posterior_type2[h] >  hh_posterior_type1[h] && hh_posterior_type2[h] >= hh_posterior_type3[h], 1:N_HH)
    idx_tya_type3 = findall(h -> hh_tya_terminal[h] == 2 && hh_posterior_type3[h] >  hh_posterior_type1[h] && hh_posterior_type3[h] >  hh_posterior_type2[h], 1:N_HH)

    function _aggregate_subgroup(hh_indices)

        N_sub = length(hh_indices)
        # max(..., 1) guards against division by zero when a type has no households
        N_total = max(N_sub * N_draws, 1)

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
            mean_aflav                = zeros(Float64, T_sim),
            mean_welfare              = zeros(Float64, T_sim),
            mean_q_cig                = zeros(Float64, T_sim),
            mean_q_orig_ecig          = zeros(Float64, T_sim),
            mean_q_flav_ecig          = zeros(Float64, T_sim)
        )

        # Return the zero-filled DataFrame without entering the period loop.
        if N_sub == 0
            return results
        end

        for t in 1:T_sim

            cat_counts = zeros(Float64, 8)
            total_addiction    = 0.0
            total_aflav        = 0.0
            total_welfare      = 0.0
            total_q_cig        = 0.0
            total_q_orig_ecig  = 0.0
            total_q_flav_ecig  = 0.0

            for h in hh_indices
                for d in 1:N_draws
                    j   = sim_choices[h, t, d]   # chosen alternative index 
                    cat = cat_idx[j]              # product category 
                    cat_counts[cat + 1] += 1.0

                    total_addiction   += sim_addiction[h, t, d]
                    total_aflav       += sim_aflav[h, t, d]
                    total_welfare     += sim_welfare[h, t, d]
                    total_q_cig       += sim_q_cig[h, t, d]
                    total_q_orig_ecig += sim_q_orig_ecig[h, t, d]
                    total_q_flav_ecig += sim_q_flav_ecig[h, t, d]
                end
            end

            results.share_outside[t]             = cat_counts[1] / N_total
            results.share_cig[t]                 = cat_counts[2] / N_total
            results.share_orig_ecig[t]           = cat_counts[3] / N_total
            results.share_non_fda_flav_ecig[t]   = cat_counts[4] / N_total
            results.share_fda_flav_ecig[t]       = cat_counts[5] / N_total
            results.share_orig_bundle[t]         = cat_counts[6] / N_total
            results.share_non_fda_flav_bundle[t] = cat_counts[7] / N_total
            results.share_fda_flav_bundle[t]     = cat_counts[8] / N_total
            results.mean_addiction[t]            = total_addiction   / N_total
            results.mean_aflav[t]               = total_aflav        / N_total
            results.mean_welfare[t]              = total_welfare      / N_total
            results.mean_q_cig[t]                = total_q_cig        / N_total
            results.mean_q_orig_ecig[t]           = total_q_orig_ecig / N_total
            results.mean_q_flav_ecig[t]           = total_q_flav_ecig / N_total
        end

        return results
    end

    df_tya_type1 = _aggregate_subgroup(idx_tya_type1)
    df_tya_type2 = _aggregate_subgroup(idx_tya_type2)
    df_tya_type3 = _aggregate_subgroup(idx_tya_type3)

    return df_tya_type1, df_tya_type2, df_tya_type3
end


# --- By Addiction Tercile ---

"""
Aggregate simulation results by addiction tercile, split by quantile cutpoints
on terminal addiction (hh_addiction_terminal = (a_f + a_s) / 2 at the last
period). Same column structure as aggregate_simulation.
"""
function aggregate_simulation_by_addiction(
    sim_choices::Array{Int, 3},
    sim_addiction::Array{Float64, 3},
    sim_aflav::Array{Float64, 3},
    sim_welfare::Array{Float64, 3},
    cat_idx::AbstractVector{<:Integer},
    N_J::Integer,
    T_sim::Integer,
    hh_addiction_terminal::Vector{Float64},
    N_draws::Integer
)

    N_HH = size(sim_choices, 1)

    # Compute tercile cutpoints
    q_33, q_67 = quantile(hh_addiction_terminal, [1/3, 2/3])

    # Identify subgroup household indices
    idx_low  = findall(h -> hh_addiction_terminal[h] <= q_33,                                    1:N_HH)
    idx_med  = findall(h -> hh_addiction_terminal[h] > q_33 && hh_addiction_terminal[h] <= q_67, 1:N_HH)
    idx_high = findall(h -> hh_addiction_terminal[h] > q_67,                                     1:N_HH)

    function _aggregate_subgroup(hh_indices)

        N_sub = length(hh_indices)
        # each household contributes N_draws Monte Carlo paths
        N_total = N_sub * N_draws

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
            mean_aflav                = zeros(Float64, T_sim),
            mean_welfare              = zeros(Float64, T_sim)
        )

        for t in 1:T_sim

            cat_counts = zeros(Float64, 8)  # cats 0-7
            total_addiction = 0.0
            total_aflav     = 0.0
            total_welfare   = 0.0

            for h in hh_indices
                for d in 1:N_draws
                    j = sim_choices[h, t, d]
                    cat = cat_idx[j]
                    cat_counts[cat + 1] += 1.0  # +1 because cat=0 maps to index 1

                    total_addiction += sim_addiction[h, t, d]
                    total_aflav     += sim_aflav[h, t, d]
                    total_welfare   += sim_welfare[h, t, d]
                end
            end

            results.share_outside[t]             = cat_counts[1] / N_total
            results.share_cig[t]                 = cat_counts[2] / N_total
            results.share_orig_ecig[t]           = cat_counts[3] / N_total
            results.share_non_fda_flav_ecig[t]   = cat_counts[4] / N_total
            results.share_fda_flav_ecig[t]       = cat_counts[5] / N_total
            results.share_orig_bundle[t]         = cat_counts[6] / N_total
            results.share_non_fda_flav_bundle[t] = cat_counts[7] / N_total
            results.share_fda_flav_bundle[t]     = cat_counts[8] / N_total
            results.mean_addiction[t]            = total_addiction / N_total
            results.mean_aflav[t]               = total_aflav     / N_total
            results.mean_welfare[t]              = total_welfare   / N_total
        end

        return results
    end

    df_low  = _aggregate_subgroup(idx_low)
    df_med  = _aggregate_subgroup(idx_med)
    df_high = _aggregate_subgroup(idx_high)

    return df_low, df_med, df_high
end


# --- By TYA x Addiction Tercile ---

"""
Aggregate simulation results by the cross of TYA status and addiction tercile
(6 subgroups). Same column structure as aggregate_simulation.

Returns a Dict{String, DataFrame} keyed "tya_low", "tya_med", "tya_high",
"no_tya_low", "no_tya_med", "no_tya_high".
"""
function aggregate_simulation_by_tya_addiction(
    sim_choices::Array{Int, 3},
    sim_addiction::Array{Float64, 3},
    sim_aflav::Array{Float64, 3},
    sim_welfare::Array{Float64, 3},
    cat_idx::AbstractVector{<:Integer},
    N_J::Integer,
    T_sim::Integer,
    hh_tya_terminal::Vector{Int},
    hh_addiction_terminal::Vector{Float64},
    N_draws::Integer
)

    N_HH = size(sim_choices, 1)

    # Compute addiction tercile cutpoints
    q_33, q_67 = quantile(hh_addiction_terminal, [1/3, 2/3])

    # TYA masks
    is_tya    = [hh_tya_terminal[h] == 2 for h in 1:N_HH]
    is_no_tya = [hh_tya_terminal[h] == 1 for h in 1:N_HH]

    # Addiction masks
    is_low  = [hh_addiction_terminal[h] <= q_33                                    for h in 1:N_HH]
    is_med  = [hh_addiction_terminal[h] > q_33 && hh_addiction_terminal[h] <= q_67 for h in 1:N_HH]
    is_high = [hh_addiction_terminal[h] > q_67                                     for h in 1:N_HH]

    # Cross subgroup indices
    idx_tya_low     = findall(h -> is_tya[h]    && is_low[h],  1:N_HH)
    idx_tya_med     = findall(h -> is_tya[h]    && is_med[h],  1:N_HH)
    idx_tya_high    = findall(h -> is_tya[h]    && is_high[h], 1:N_HH)
    idx_no_tya_low  = findall(h -> is_no_tya[h] && is_low[h],  1:N_HH)
    idx_no_tya_med  = findall(h -> is_no_tya[h] && is_med[h],  1:N_HH)
    idx_no_tya_high = findall(h -> is_no_tya[h] && is_high[h], 1:N_HH)

    function _aggregate_subgroup(hh_indices)

        N_sub = length(hh_indices)
        # each household contributes N_draws Monte Carlo paths
        N_total = N_sub * N_draws

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
            mean_aflav                = zeros(Float64, T_sim),
            mean_welfare              = zeros(Float64, T_sim)
        )

        for t in 1:T_sim

            cat_counts = zeros(Float64, 8)  # cats 0-7
            total_addiction = 0.0
            total_aflav     = 0.0
            total_welfare   = 0.0

            for h in hh_indices
                for d in 1:N_draws
                    j = sim_choices[h, t, d]
                    cat = cat_idx[j]
                    cat_counts[cat + 1] += 1.0  # +1 because cat=0 maps to index 1

                    total_addiction += sim_addiction[h, t, d]
                    total_aflav     += sim_aflav[h, t, d]
                    total_welfare   += sim_welfare[h, t, d]
                end
            end

            results.share_outside[t]             = cat_counts[1] / N_total
            results.share_cig[t]                 = cat_counts[2] / N_total
            results.share_orig_ecig[t]           = cat_counts[3] / N_total
            results.share_non_fda_flav_ecig[t]   = cat_counts[4] / N_total
            results.share_fda_flav_ecig[t]       = cat_counts[5] / N_total
            results.share_orig_bundle[t]         = cat_counts[6] / N_total
            results.share_non_fda_flav_bundle[t] = cat_counts[7] / N_total
            results.share_fda_flav_bundle[t]     = cat_counts[8] / N_total
            results.mean_addiction[t]            = total_addiction / N_total
            results.mean_aflav[t]               = total_aflav     / N_total
            results.mean_welfare[t]              = total_welfare   / N_total
        end

        return results
    end

    result = Dict{String, DataFrame}(
        "tya_low"     => _aggregate_subgroup(idx_tya_low),
        "tya_med"     => _aggregate_subgroup(idx_tya_med),
        "tya_high"    => _aggregate_subgroup(idx_tya_high),
        "no_tya_low"  => _aggregate_subgroup(idx_no_tya_low),
        "no_tya_med"  => _aggregate_subgroup(idx_no_tya_med),
        "no_tya_high" => _aggregate_subgroup(idx_no_tya_high)
    )

    return result
end


# --- Extensive Margin ---

"""
Compute extensive margin initiation rates for TYA-present households.

Identifies households with initial flavored habit stock below `aflav_threshold`
("non-users at baseline") and tracks the cumulative probability that such a
household makes at least one flavored e-cigarette purchase by each horizon t,
under both the status quo and counterfactual simulation.

prevention_rate = sq_ever_initiated - cf_ever_initiated: the fraction of
would-be initiators whose first flavored purchase is prevented by the policy.
"""
function aggregate_extensive_margin_by_tya(
    sim_choices_sq::Array{Int, 3},
    sim_choices_cf::Array{Int, 3},
    hh_aflav0::Vector{Float64},
    cat_idx::AbstractVector{<:Integer},
    T_sim::Integer,
    hh_tya_terminal::Vector{Int},
    N_draws::Integer;
    aflav_threshold::Float64 = 0.10
)

    N_HH = size(sim_choices_sq, 1)

    # is alternative j a flavored product? (cats 3,4,6,7)
    is_flavored_alt = [cat_idx[j] in (3, 4, 6, 7) for j in 1:length(cat_idx)]

    # Non-user households: initial flavored habit below threshold
    non_user_mask = hh_aflav0 .< aflav_threshold

    # TYA-present non-users
    idx_tya = findall(h -> hh_tya_terminal[h] == 2 && non_user_mask[h], 1:N_HH)

    function _compute_subgroup(hh_indices)

        N_sub   = length(hh_indices)
        # max(..., 1) guards against division by zero when there are no non-users
        N_paths = max(N_sub * N_draws, 1)

        sq_cumulative = zeros(Float64, T_sim)
        cf_cumulative = zeros(Float64, T_sim)

        for h in hh_indices
            for d in 1:N_draws
                sq_initiated = false
                cf_initiated = false
                for t in 1:T_sim
                    # Update initiation flags on first flavored choice
                    if !sq_initiated && is_flavored_alt[sim_choices_sq[h, t, d]]
                        sq_initiated = true
                    end
                    if !cf_initiated && is_flavored_alt[sim_choices_cf[h, t, d]]
                        cf_initiated = true
                    end
                    # Accumulate: 1 if initiated by period t, 0 otherwise
                    sq_cumulative[t] += sq_initiated ? 1.0 : 0.0
                    cf_cumulative[t] += cf_initiated ? 1.0 : 0.0
                end
            end
        end

        return DataFrame(
            period            = 1:T_sim,
            n_non_users       = fill(N_sub, T_sim),
            sq_ever_initiated = sq_cumulative ./ N_paths,
            cf_ever_initiated = cf_cumulative ./ N_paths,
            prevention_rate   = (sq_cumulative .- cf_cumulative) ./ N_paths
        )
    end

    df_tya = _compute_subgroup(idx_tya)

    return df_tya
end


# --- Extensive Margin by Latent Type (K=3 Mixture) ---

"""
Compute extensive margin initiation rates by modal latent type (K=3 mixture),
among non-user households within each type group (initial flavored habit
below `aflav_threshold`). Same column structure as aggregate_extensive_margin_by_tya.
"""
function aggregate_extensive_margin_by_type_k3(
    sim_choices_sq::Array{Int, 3},
    sim_choices_cf::Array{Int, 3},
    hh_aflav0::Vector{Float64},
    cat_idx::AbstractVector{<:Integer},
    T_sim::Integer,
    hh_posterior_type1::Vector{Float64},
    hh_posterior_type2::Vector{Float64},
    hh_posterior_type3::Vector{Float64},
    N_draws::Integer;
    aflav_threshold::Float64 = 0.10
)

    N_HH = size(sim_choices_sq, 1)

    # is alternative j a flavored product? (cats 3,4,6,7)
    is_flavored_alt = [cat_idx[j] in (3, 4, 6, 7) for j in 1:length(cat_idx)]

    # Non-user households: initial flavored habit below threshold
    non_user_mask = hh_aflav0 .< aflav_threshold

    # Modal type assignment via argmax of posterior, restricted to non-users at baseline.
    # Ties broken in favor of lower type index (type1 >= type2 takes type1).
    idx_type1 = findall(h -> hh_posterior_type1[h] >= hh_posterior_type2[h] && hh_posterior_type1[h] >= hh_posterior_type3[h] && non_user_mask[h], 1:N_HH)
    idx_type2 = findall(h -> hh_posterior_type2[h] >  hh_posterior_type1[h] && hh_posterior_type2[h] >= hh_posterior_type3[h] && non_user_mask[h], 1:N_HH)
    idx_type3 = findall(h -> hh_posterior_type3[h] >  hh_posterior_type1[h] && hh_posterior_type3[h] >  hh_posterior_type2[h] && non_user_mask[h], 1:N_HH)

    function _compute_subgroup(hh_indices)

        N_sub   = length(hh_indices)
        # max(..., 1) guards against division by zero when a type has no non-users
        N_paths = max(N_sub * N_draws, 1)

        sq_cumulative = zeros(Float64, T_sim)
        cf_cumulative = zeros(Float64, T_sim)

        for h in hh_indices
            for d in 1:N_draws
                sq_initiated = false
                cf_initiated = false
                for t in 1:T_sim
                    # Update initiation flags on first flavored choice
                    if !sq_initiated && is_flavored_alt[sim_choices_sq[h, t, d]]
                        sq_initiated = true
                    end
                    if !cf_initiated && is_flavored_alt[sim_choices_cf[h, t, d]]
                        cf_initiated = true
                    end
                    # Accumulate: 1 if initiated by period t, 0 otherwise
                    sq_cumulative[t] += sq_initiated ? 1.0 : 0.0
                    cf_cumulative[t] += cf_initiated ? 1.0 : 0.0
                end
            end
        end

        return DataFrame(
            period            = 1:T_sim,
            n_non_users       = fill(N_sub, T_sim),
            sq_ever_initiated = sq_cumulative ./ N_paths,
            cf_ever_initiated = cf_cumulative ./ N_paths,
            prevention_rate   = (sq_cumulative .- cf_cumulative) ./ N_paths
        )
    end

    df_type1 = _compute_subgroup(idx_type1)
    df_type2 = _compute_subgroup(idx_type2)
    df_type3 = _compute_subgroup(idx_type3)

    return df_type1, df_type2, df_type3
end


# --- Extensive Margin by TYA-Presence and Latent Type (K=3 Mixture) ---

"""
Compute extensive margin initiation rates among TYA-present households only,
split by modal latent type (K=3 mixture). Mirrors aggregate_extensive_margin_by_type_k3,
but restricts to TYA-present households before assigning modal type.
"""
function aggregate_extensive_margin_by_tya_type_k3(
    sim_choices_sq::Array{Int, 3},
    sim_choices_cf::Array{Int, 3},
    hh_aflav0::Vector{Float64},
    cat_idx::AbstractVector{<:Integer},
    T_sim::Integer,
    hh_tya_terminal::Vector{Int},
    hh_posterior_type1::Vector{Float64},
    hh_posterior_type2::Vector{Float64},
    hh_posterior_type3::Vector{Float64},
    N_draws::Integer;
    aflav_threshold::Float64 = 0.10
)

    N_HH = size(sim_choices_sq, 1)

    # is alternative j a flavored product? (cats 3,4,6,7)
    is_flavored_alt = [cat_idx[j] in (3, 4, 6, 7) for j in 1:length(cat_idx)]

    # Non-user households: initial flavored habit below threshold
    non_user_mask = hh_aflav0 .< aflav_threshold

    # Restrict to TYA-present households, then assign modal type via argmax of
    # posterior probabilities, among non-users at baseline. 
    idx_tya_type1 = findall(h -> hh_tya_terminal[h] == 2 && hh_posterior_type1[h] >= hh_posterior_type2[h] && hh_posterior_type1[h] >= hh_posterior_type3[h] && non_user_mask[h], 1:N_HH)
    idx_tya_type2 = findall(h -> hh_tya_terminal[h] == 2 && hh_posterior_type2[h] >  hh_posterior_type1[h] && hh_posterior_type2[h] >= hh_posterior_type3[h] && non_user_mask[h], 1:N_HH)
    idx_tya_type3 = findall(h -> hh_tya_terminal[h] == 2 && hh_posterior_type3[h] >  hh_posterior_type1[h] && hh_posterior_type3[h] >  hh_posterior_type2[h] && non_user_mask[h], 1:N_HH)

    function _compute_subgroup(hh_indices)

        N_sub   = length(hh_indices)
        # max(..., 1) guards against division by zero when a type has no non-users
        N_paths = max(N_sub * N_draws, 1)

        sq_cumulative = zeros(Float64, T_sim)
        cf_cumulative = zeros(Float64, T_sim)

        for h in hh_indices
            for d in 1:N_draws
                sq_initiated = false
                cf_initiated = false
                for t in 1:T_sim
                    # Update initiation flags on first flavored choice
                    if !sq_initiated && is_flavored_alt[sim_choices_sq[h, t, d]]
                        sq_initiated = true
                    end
                    if !cf_initiated && is_flavored_alt[sim_choices_cf[h, t, d]]
                        cf_initiated = true
                    end
                    # Accumulate: 1 if initiated by period t, 0 otherwise
                    sq_cumulative[t] += sq_initiated ? 1.0 : 0.0
                    cf_cumulative[t] += cf_initiated ? 1.0 : 0.0
                end
            end
        end

        return DataFrame(
            period            = 1:T_sim,
            n_non_users       = fill(N_sub, T_sim),
            sq_ever_initiated = sq_cumulative ./ N_paths,
            cf_ever_initiated = cf_cumulative ./ N_paths,
            prevention_rate   = (sq_cumulative .- cf_cumulative) ./ N_paths
        )
    end

    df_tya_type1 = _compute_subgroup(idx_tya_type1)
    df_tya_type2 = _compute_subgroup(idx_tya_type2)
    df_tya_type3 = _compute_subgroup(idx_tya_type3)

    return df_tya_type1, df_tya_type2, df_tya_type3
end
