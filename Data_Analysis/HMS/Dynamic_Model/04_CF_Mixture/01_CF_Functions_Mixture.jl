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
#  1. Flavor Bans         — Zero out flow utility for banned categories so they
#                           receive zero choice probability in VFI and simulation.
#                           Three variants: comprehensive (cats 3,4,6,7),
#                           FDA-only (cats 4,7), and non-FDA-only (cats 3,6).
#
#  2. Flavor Tax          — Shift flow utility downward by ω_E · τ · q_ecig_raw[j]
#                           for taxed categories. Three variants matching the ban
#                           taxonomy: all flavored, FDA-only, and non-FDA-only.
#
#  3. Pointwise Outcomes  — Interpolate a single type's V_choice at each observed
#                           continuous state, then compute softmax choice
#                           probabilities and logsumexp welfare. Called six times
#                           per counterfactual loop (3 types × SQ + CF); mixture
#                           weighting is done inline in 02_CF_Mixture.jl.
#
#  4. Aggregation         — Average N_HH × N_draws Monte Carlo paths into
#                           period-by-period category shares, mean addiction, and
#                           mean welfare. Subgroup variants stratify by TYA status,
#                           modal latent type, addiction tercile, and their combinations.
#                           A separate extensive-margin function tracks cumulative
#                           flavored initiation rates among baseline non-users.
################################################################################


#############################
# 1. Flavor Bans
#############################

# --- Comprehensive Ban ---

"""
Apply a comprehensive flavor ban by setting flow utility to -Inf for all
flavored alternatives.

Flavored alternatives are those with cat_idx[j] in {3, 4, 6, 7}:
  - cat_idx = 3: non-FDA flavored e-cigarettes
  - cat_idx = 4: FDA flavored e-cigarettes
  - cat_idx = 6: bundles with non-FDA flavored e-cig
  - cat_idx = 7: bundles with FDA flavored e-cig

Setting U = -Inf means exp(U) = 0 in the logsumexp, so banned alternatives
get zero choice probability.
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
Apply an FDA-only flavor ban by setting flow utility to -Inf for FDA-authorized
flavored alternatives only.

FDA-authorized flavored alternatives are those with cat_idx[j] in {4, 7}:
  - cat_idx = 4: FDA flavored e-cigarettes
  - cat_idx = 7: FDA flavored bundles

Non-FDA flavored alternatives (cat_idx = 3, 6) remain available.

Setting U = -Inf means exp(U) = 0 in the logsumexp, so banned alternatives
get zero choice probability.
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
Apply a non-FDA-authorized product ban by setting flow utility to -Inf for
non-FDA-authorized flavored alternatives only.

Non-FDA-authorized flavored alternatives are those with cat_idx[j] in {3, 6}:
  - cat_idx = 3: non-FDA flavored e-cigarettes (no PMTA approval)
  - cat_idx = 6: bundles with non-FDA flavored e-cig

FDA-authorized flavored alternatives (cat_idx = 4, 7) and all other products
remain available. This scenario models FDA enforcement that removes unauthorized
products from the market while leaving PMTA-approved flavored products intact.

Setting U = -Inf means exp(U) = 0 in the logsumexp, so banned alternatives
get zero choice probability.
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
Apply a per-unit tax on flavored e-cigarette alternatives by shifting their
flow utility downward.

For each flavored alternative j (cat_idx[j] in {3, 4, 6, 7}), the tax adds:

    Δu[j] = ω_E * τ * q_ecig_raw[j]

to the flow utility at every state point. Since ω_E < 0 and τ > 0, this reduces
the utility of flavored alternatives, making them less attractive relative to
unflavored products and the outside option.

The tax is applied uniformly across all price states because it is a per-unit
excise tax (independent of the retail price). With the separate price coefficient
specification (ω_E * P_ecig * has_ecig[j]), a per-mL tax of τ adds τ * q_ecig_raw
to expenditure, so the utility shift is ω_E * τ * q_ecig_raw[j].

Arguments:
- U_tax:      Array{Float64, 6} to modify in-place (should be a copy of SQ U)
- cat_idx:    Vector{Int} (N_J), category index for each alternative
- omega_E:    Float64, estimated e-cig price coefficient (negative)
- q_ecig:     Vector{Float64} (N_J), standardized e-cig mL quantity per alternative
- q_ecig_max: Float64, max raw e-cig quantity (for de-standardizing)
- tau:        Float64, tax per mL of e-liquid (in real dollars)
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
            # Tax shifts utility: ω_E · τ · q_ecig_raw[j]
            delta_u = omega_E * tau * q_ecig[j] * q_ecig_max
            U_tax[:, j, :, :, :, :] .+= delta_u
        end
    end

    return nothing
end


# --- FDA-Authorized Only ---

"""
Apply a per-unit tax on FDA-authorized flavored e-cigarette alternatives only
(cat_idx in {4, 7}), leaving unauthorized flavored alternatives (cat_idx in
{3, 6}) untaxed. Mirrors apply_flavor_tax! but restricted to authorized brands.
"""
function apply_fda_flavor_tax!(
    U_tax::Array{Float64, 6},
    cat_idx::AbstractVector{<:Integer},
    omega_E::Real,
    q_ecig::AbstractVector{<:Real},
    q_ecig_max::Real,
    tau::Real
)

    N_J = size(U_tax, 2)

    for j in 1:N_J
        if cat_idx[j] in (4, 7)
            # Tax shifts utility: ω_E · τ · q_ecig_raw[j]
            delta_u = omega_E * tau * q_ecig[j] * q_ecig_max
            U_tax[:, j, :, :, :, :] .+= delta_u
        end
    end

    return nothing
end


# --- Non-FDA-Authorized Only ---

"""
Apply a per-unit tax on non-FDA-authorized flavored e-cigarette alternatives only
(cat_idx in {3, 6}), leaving FDA-authorized flavored alternatives (cat_idx in
{4, 7}) untaxed. Mirrors apply_flavor_tax! but restricted to unauthorized brands.
"""
function apply_non_fda_flavor_tax!(
    U_tax::Array{Float64, 6},
    cat_idx::AbstractVector{<:Integer},
    omega_E::Real,
    q_ecig::AbstractVector{<:Real},
    q_ecig_max::Real,
    tau::Real
)

    N_J = size(U_tax, 2)

    for j in 1:N_J
        if cat_idx[j] in (3, 6)
            # Tax shifts utility: ω_E · τ · q_ecig_raw[j]
            delta_u = omega_E * tau * q_ecig[j] * q_ecig_max
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
type's V_choice solution. Called once per latent type (k = 1, 2, 3) and once
per policy (SQ, ban/tax), producing six matrices per counterfactual loop.

The K=3 mixture result is assembled in 02_CF_Mixture.jl in three steps:
  1. Call this function six times to get per-type probabilities and welfare
     under both SQ and counterfactual at every observed state.
  2. Use the SQ per-type probabilities to compute the household posterior
     P(type=k | data_h) via Bayes' rule with the K=3 softmax prior.
  3. Form the mixture-weighted outcome as a posterior-weighted average:
       probs_sq   = w1 * probs_1_sq + w2 * probs_2_sq + w3 * probs_3_sq
       welfare_sq = w1 * welfare_1_sq + ...

For each observation i with state (tya_i, af_i, as_i, aflav_i, p_cig_i, p_ecig_i):
  1. Interpolate V_choice at the continuous state for all N_J alternatives.
  2. Replace NaN (from 0.0 * (-Inf) at grid boundaries) with -Inf.
  3. Compute welfare as logsumexp(V_choice_interp) (expected maximum utility).
  4. Compute softmax choice probabilities.

Returns:
- probs:   Matrix{Float64} (N_obs x N_J), choice probabilities
- welfare: Vector{Float64} (N_obs), welfare (inclusive value)
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

    # Initialize output arrays
    probs   = Matrix{Float64}(undef, N_obs, N_J)
    welfare = Vector{Float64}(undef, N_obs)

    # Compute outcomes for each observation
    for i in 1:N_obs

        # Interpolate V_choice at continuous state
        v_interp = interpolate_v_choice(
            V_choice, tya_state[i], af_continuous[i], as_continuous[i],
            aflav_continuous[i], p_continuous[i, 1], p_continuous[i, 2],
            N_J, N_P, A_f, A_s, A_flav, P
        )

        # Fix NaN from 0.0 * (-Inf).
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
# 4. Aggregation
#############################

# --- Overall ---

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
    sim_aflav::Array{Float64, 3},
    sim_welfare::Array{Float64, 3},
    cat_idx::AbstractVector{<:Integer},
    N_J::Integer,
    T_sim::Integer
)

    N_HH, _, N_draws = size(sim_choices)

    # N_total is the effective sample size: each household contributes N_draws
    # Monte Carlo paths, so all averages divide by N_HH * N_draws.
    N_total = N_HH * N_draws

    # Pre-allocate result DataFrame with one row per simulated period
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

        # cat_counts[c+1] accumulates the number of (h, d) paths in category c
        # at period t. Adding 1 converts the 0-based category index to a
        # 1-based Julia array index (cat 0 = outside → index 1, ..., cat 7 → index 8).
        cat_counts = zeros(Float64, 8)
        total_addiction = 0.0
        total_aflav     = 0.0
        total_welfare   = 0.0

        for h in 1:N_HH
            for d in 1:N_draws
                j   = sim_choices[h, t, d]   # chosen alternative index (1-based)
                cat = cat_idx[j]              # product category (0-based)
                cat_counts[cat + 1] += 1.0

                total_addiction += sim_addiction[h, t, d]
                total_aflav     += sim_aflav[h, t, d]
                total_welfare   += sim_welfare[h, t, d]
            end
        end

        # Divide counts by N_total to get category market shares;
        # divide totals by N_total to get means across all (h, d) paths.
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


# --- By TYA Status ---

"""
Aggregate simulation results separately for TYA-present and TYA-absent households.

TYA-present households have hh_tya_terminal in {3, 4} (TYA states indicating
teen/young adult presence). TYA-absent households have hh_tya_terminal in {1, 2}.

Each subgroup DataFrame has the same column structure as aggregate_simulation:
  period, share_outside, share_cig, share_orig_ecig, share_non_fda_flav_ecig,
  share_fda_flav_ecig, share_orig_bundle, share_non_fda_flav_bundle,
  share_fda_flav_bundle, mean_addiction, mean_welfare

Returns:
- df_tya:    DataFrame for TYA-present households
- df_no_tya: DataFrame for TYA-absent households
"""
function aggregate_simulation_by_tya(
    sim_choices::Array{Int, 3},
    sim_addiction::Array{Float64, 3},
    sim_aflav::Array{Float64, 3},
    sim_welfare::Array{Float64, 3},
    cat_idx::AbstractVector{<:Integer},
    N_J::Integer,
    T_sim::Integer,
    hh_tya_terminal::Vector{Int},
    N_draws::Integer
)

    N_HH = size(sim_choices, 1)

    # Identify subgroup household indices
    # TYA state is binary: 1 = no TYA, 2 = TYA present
    idx_tya    = findall(h -> hh_tya_terminal[h] == 2, 1:N_HH)
    idx_no_tya = findall(h -> hh_tya_terminal[h] == 1, 1:N_HH)

    # Helper: aggregate over a subset of households
    function _aggregate_subgroup(hh_indices)

        N_sub = length(hh_indices)
        # N_total is the effective sample size for this subgroup: each household
        # contributes N_draws Monte Carlo paths, so all averages divide by N_sub * N_draws.
        N_total = N_sub * N_draws

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
            mean_aflav                = zeros(Float64, T_sim),
            mean_welfare              = zeros(Float64, T_sim)
        )

        for t in 1:T_sim

            # Count category shares across subgroup households and draws
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
            results.mean_aflav[t]               = total_aflav     / N_total
            results.mean_welfare[t]              = total_welfare   / N_total
        end

        return results
    end

    df_tya    = _aggregate_subgroup(idx_tya)
    df_no_tya = _aggregate_subgroup(idx_no_tya)

    return df_tya, df_no_tya
end


# --- By Latent Type (K=3 Mixture) ---

"""
Aggregate simulation results by modal latent type for K=3 mixture.

Households are assigned to their modal type via argmax of posterior probabilities:
  argmax(hh_posterior_type1[h], hh_posterior_type2[h], hh_posterior_type3[h])

Each subgroup DataFrame has the same column structure as aggregate_simulation.

Returns:
- df_type1: DataFrame for modal Type 1 households
- df_type2: DataFrame for modal Type 2 households
- df_type3: DataFrame for modal Type 3 households
"""
function aggregate_simulation_by_type_k3(
    sim_choices::Array{Int, 3},
    sim_addiction::Array{Float64, 3},
    sim_aflav::Array{Float64, 3},
    sim_welfare::Array{Float64, 3},
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
        # N_total is the effective sample size for this subgroup; max(..., 1) guards
        # against division by zero when a modal type has zero assigned households.
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
            mean_welfare              = zeros(Float64, T_sim)
        )

        # Return the zero-filled DataFrame without entering the period loop.
        if N_sub == 0
            return results
        end

        for t in 1:T_sim

            # cat_counts[c+1] accumulates the number of (h, d) paths in category c
            # at period t. Adding 1 converts the 0-based category index to a
            # 1-based Julia array index (cat 0 = outside → index 1, ..., cat 7 → index 8).
            cat_counts = zeros(Float64, 8)
            total_addiction = 0.0
            total_aflav     = 0.0
            total_welfare   = 0.0

            for h in hh_indices
                for d in 1:N_draws
                    j   = sim_choices[h, t, d]   # chosen alternative index (1-based)
                    cat = cat_idx[j]              # product category (0-based)
                    cat_counts[cat + 1] += 1.0

                    total_addiction += sim_addiction[h, t, d]
                    total_aflav     += sim_aflav[h, t, d]
                    total_welfare   += sim_welfare[h, t, d]
                end
            end

            # Divide counts by N_total to get category market shares;
            # divide totals by N_total to get means across all (h, d) paths.
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

    df_type1 = _aggregate_subgroup(idx_type1)
    df_type2 = _aggregate_subgroup(idx_type2)
    df_type3 = _aggregate_subgroup(idx_type3)

    return df_type1, df_type2, df_type3
end


# --- By Addiction Tercile ---

"""
Aggregate simulation results by addiction tercile based on terminal addiction
levels.

Addiction terciles are defined by quantile cutpoints on hh_addiction_terminal,
where each element is (a_f + a_s) / 2 at the terminal period:
  - Low:    hh_addiction_terminal <= q_33
  - Medium: q_33 < hh_addiction_terminal <= q_67
  - High:   hh_addiction_terminal > q_67

Each subgroup DataFrame has the same column structure as aggregate_simulation.

Returns:
- df_low:  DataFrame for bottom-tercile addiction households
- df_med:  DataFrame for middle-tercile addiction households
- df_high: DataFrame for top-tercile addiction households
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

    # Helper: aggregate over a subset of households
    function _aggregate_subgroup(hh_indices)

        N_sub = length(hh_indices)
        # N_total is the effective sample size for this subgroup: each household
        # contributes N_draws Monte Carlo paths, so all averages divide by N_sub * N_draws.
        N_total = N_sub * N_draws

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
            mean_aflav                = zeros(Float64, T_sim),
            mean_welfare              = zeros(Float64, T_sim)
        )

        for t in 1:T_sim

            # Count category shares across subgroup households and draws
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
Aggregate simulation results by the cross of TYA status and addiction tercile,
yielding 6 subgroups.

Combines the TYA split (hh_tya_terminal in {3,4} vs {1,2}) with the addiction
tercile split (low/med/high based on quantile cutpoints of hh_addiction_terminal).

Each subgroup DataFrame has the same column structure as aggregate_simulation.

Returns:
- Dict{String, DataFrame} with keys:
    "tya_low", "tya_med", "tya_high",
    "no_tya_low", "no_tya_med", "no_tya_high"
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

    # Helper: aggregate over a subset of households
    function _aggregate_subgroup(hh_indices)

        N_sub = length(hh_indices)
        # N_total is the effective sample size for this subgroup: each household
        # contributes N_draws Monte Carlo paths, so all averages divide by N_sub * N_draws.
        N_total = N_sub * N_draws

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
            mean_aflav                = zeros(Float64, T_sim),
            mean_welfare              = zeros(Float64, T_sim)
        )

        for t in 1:T_sim

            # Count category shares across subgroup households and draws
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
            results.mean_aflav[t]               = total_aflav     / N_total
            results.mean_welfare[t]              = total_welfare   / N_total
        end

        return results
    end

    # Build result dictionary
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
Compute extensive margin initiation rates by TYA status.

Identifies households with initial flavored habit stock below `aflav_threshold`
("non-users at baseline") and tracks the cumulative probability that such a
household makes at least one flavored e-cigarette purchase by each horizon t
under both the status quo and counterfactual simulation.

The prevention rate is sq_ever_initiated - cf_ever_initiated: the fraction of
would-be initiators whose first flavored purchase is prevented by the policy.

Arguments:
- sim_choices_sq:   N_HH × T_sim × N_draws array of SQ choice indices
- sim_choices_cf:   N_HH × T_sim × N_draws array of CF choice indices
- hh_aflav0:        Vector{Float64} of length N_HH, terminal flavored habit stocks
- cat_idx:          Alternative-level category index vector (0-indexed categories)
- T_sim:            Number of simulation periods
- hh_tya_terminal:  Vector{Int} of length N_HH, TYA state (1 = no TYA, 2 = TYA)
- N_draws:          Number of Monte Carlo draws per household
- aflav_threshold:  Households with hh_aflav0[h] < this are treated as non-users (default 0.10)

Returns:
- df_tya:    DataFrame for TYA-present non-user households
- df_no_tya: DataFrame for TYA-absent non-user households

Each DataFrame has columns:
  period, n_non_users, sq_ever_initiated, cf_ever_initiated, prevention_rate
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

    # Boolean vector: is alternative j a flavored product?
    # Flavored categories: 3 = non-FDA flav ecig, 4 = FDA flav ecig,
    #                      6 = non-FDA flav bundle, 7 = FDA flav bundle
    is_flavored_alt = [cat_idx[j] in (3, 4, 6, 7) for j in 1:length(cat_idx)]

    # Non-user households: initial flavored habit below threshold
    non_user_mask = hh_aflav0 .< aflav_threshold

    # Split non-users by TYA status
    idx_tya    = findall(h -> hh_tya_terminal[h] == 2 && non_user_mask[h], 1:N_HH)
    idx_no_tya = findall(h -> hh_tya_terminal[h] == 1 && non_user_mask[h], 1:N_HH)

    # Helper: compute cumulative initiation rates for a subgroup
    function _compute_subgroup(hh_indices)

        N_sub   = length(hh_indices)
        # N_paths is the total number of (household, draw) paths in this subgroup;
        # cumulative initiation counts are divided by N_paths to get initiation rates.
        N_paths = N_sub * N_draws

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

    df_tya    = _compute_subgroup(idx_tya)
    df_no_tya = _compute_subgroup(idx_no_tya)

    return df_tya, df_no_tya
end
