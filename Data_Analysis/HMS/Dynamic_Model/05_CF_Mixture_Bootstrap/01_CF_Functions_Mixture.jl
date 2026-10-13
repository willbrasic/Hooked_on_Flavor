################################################################################
# William Brasic
# The University of Arizona
# wbrasic97@gmail.com
# October 2026
#
# Functions specific to the counterfactual simulations in
# 02_CF_Mixture_Bootstrap.jl. They handle:
#   - Applying bans and taxes to flow utilities
#   - Interpolating the ex-ante value function at continuous states
#   - Computing pointwise choice probabilities and welfare at observed states
#   - Forward-simulating households with common random numbers
#   - Aggregating forward-simulation results, overall and by subgroup
#   - The tax-versus-ban frontier for e-cigarette-persistent households
#
# Depends on 02_Second_Stage_Estimation_Mixture/01_Functions_Mixture.jl, which
# must be included first.
################################################################################


################################################################################
# Table of Contents
#
#  1. Flavor Bans         - Zero out flow utility for banned categories.
#                           Comprehensive (cat 3, 4, 6, 7) and non-FDA (cat 3, 6).
#
#  2. Flavor Tax          - Per-mL excise on flavored e-liquid, entered as a
#                           flow-utility shift.
#
#  3. Pointwise Outcomes  - Interpolate the ex-ante value function at a
#                           continuous state, and a single type's V_choice at
#                           each observed state for choice probabilities and
#                           welfare.
#
#  4. Forward Simulation  - Simulate one scenario for a set of households with
#                           the shared common random numbers.
#
#  5. Aggregation         - Average Monte Carlo paths into period-by-period
#                           category shares, mean addiction, and mean welfare.
#                           Subgroup variants stratify by TYA status and modal
#                           latent type. Extensive-margin functions track
#                           cumulative initiation among baseline non-users for
#                           a chosen set of alternatives.
#
#  6. Frontier            - Type-2 household selection, subset aggregates,
#                           scalar summaries, and the matching tax rate τ*.
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

The price term in utility is ω_E · P_ecig · ratio_ecig[j] · q_ecig[j], with
q_ecig standardized by q_ecig_max, so ω_E / q_ecig_max is the marginal utility
of a dollar and a per-mL excise τ adds ω_E · τ · q_ecig[j] in standardized
units. The shift is the same in every price state because the excise is a
fixed amount per mL, independent of the realized retail price and of the bin
price ratio.
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

# --- Ex-Ante Value at a Continuous State ---

"""
5-linearly interpolate the sophisticated-agent value function V_now (no alternative
dimension) at a continuous state (af, as, aflav, p_cig, p_ecig).

Returns:
- v_interp: interpolated V_now scalar at the given continuous state
"""
function interpolate_v_now(
    V_now::Array{Float64, 5},
    tya_idx::Integer,
    af::Real,
    as::Real,
    aflav::Real,
    obs_cig::Real,
    obs_ecig::Real,
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

    # Bilinear price interpolation at each of the 8 addiction corners
    v_fff = w_ll * V_now[tya_idx, lo_af, lo_as, lo_aflav, p_ll] + w_lh * V_now[tya_idx, lo_af, lo_as, lo_aflav, p_lh] + w_hl * V_now[tya_idx, lo_af, lo_as, lo_aflav, p_hl] + w_hh * V_now[tya_idx, lo_af, lo_as, lo_aflav, p_hh]
    v_ffs = w_ll * V_now[tya_idx, lo_af, lo_as, hi_aflav, p_ll] + w_lh * V_now[tya_idx, lo_af, lo_as, hi_aflav, p_lh] + w_hl * V_now[tya_idx, lo_af, lo_as, hi_aflav, p_hl] + w_hh * V_now[tya_idx, lo_af, lo_as, hi_aflav, p_hh]
    v_fsf = w_ll * V_now[tya_idx, lo_af, hi_as, lo_aflav, p_ll] + w_lh * V_now[tya_idx, lo_af, hi_as, lo_aflav, p_lh] + w_hl * V_now[tya_idx, lo_af, hi_as, lo_aflav, p_hl] + w_hh * V_now[tya_idx, lo_af, hi_as, lo_aflav, p_hh]
    v_fss = w_ll * V_now[tya_idx, lo_af, hi_as, hi_aflav, p_ll] + w_lh * V_now[tya_idx, lo_af, hi_as, hi_aflav, p_lh] + w_hl * V_now[tya_idx, lo_af, hi_as, hi_aflav, p_hl] + w_hh * V_now[tya_idx, lo_af, hi_as, hi_aflav, p_hh]
    v_sff = w_ll * V_now[tya_idx, hi_af, lo_as, lo_aflav, p_ll] + w_lh * V_now[tya_idx, hi_af, lo_as, lo_aflav, p_lh] + w_hl * V_now[tya_idx, hi_af, lo_as, lo_aflav, p_hl] + w_hh * V_now[tya_idx, hi_af, lo_as, lo_aflav, p_hh]
    v_sfs = w_ll * V_now[tya_idx, hi_af, lo_as, hi_aflav, p_ll] + w_lh * V_now[tya_idx, hi_af, lo_as, hi_aflav, p_lh] + w_hl * V_now[tya_idx, hi_af, lo_as, hi_aflav, p_hl] + w_hh * V_now[tya_idx, hi_af, lo_as, hi_aflav, p_hh]
    v_ssf = w_ll * V_now[tya_idx, hi_af, hi_as, lo_aflav, p_ll] + w_lh * V_now[tya_idx, hi_af, hi_as, lo_aflav, p_lh] + w_hl * V_now[tya_idx, hi_af, hi_as, lo_aflav, p_hl] + w_hh * V_now[tya_idx, hi_af, hi_as, lo_aflav, p_hh]
    v_sss = w_ll * V_now[tya_idx, hi_af, hi_as, hi_aflav, p_ll] + w_lh * V_now[tya_idx, hi_af, hi_as, hi_aflav, p_lh] + w_hl * V_now[tya_idx, hi_af, hi_as, hi_aflav, p_hl] + w_hh * V_now[tya_idx, hi_af, hi_as, hi_aflav, p_hh]

    # Trilinear interpolation over addiction
    v_interp = w_fff * v_fff + w_ffs * v_ffs + w_fsf * v_fsf + w_fss * v_fss + w_sff * v_sff + w_sfs * v_sfs + w_ssf * v_ssf + w_sss * v_sss

    return v_interp
end


# --- Choice Probabilities and Welfare at Observed States ---

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
# 4. Forward Simulation
#############################

"""
    simulate_scenario(idx_hh, V_decision, V_now, ...)

Forward-simulate one scenario for the households in `idx_hh` and return the
per-(household, period, draw) paths.

Each household's latent type is sampled once per draw from its posterior
(`crn_type`), choices are drawn by inverse CDF from the logit probabilities
(`crn_choice`), and prices follow the AR(1) process with the shared shocks
(`crn_price`), so a policy's paths differ from the status quo's only through
the policy itself. `V_decision` and `V_now` are 3-tuples of the type-specific
decision and ex-ante value functions for the scenario being simulated.

Returns a NamedTuple of arrays sized (length(idx_hh), T_sim, N_draws):
`choices`, `addiction_f`, `addiction_s`, `addiction` (their average), `aflav`,
`welfare` (the ex-ante value at the period's state), `q_cig` (packs),
`q_orig_ecig` and `q_flav_ecig` (mL), all in original units.
"""
function simulate_scenario(
    idx_hh::AbstractVector{<:Integer},
    V_decision::NTuple{3, Array{Float64, 6}},
    V_now::NTuple{3, Array{Float64, 5}},
    hh_posterior_type1::AbstractVector{<:Real},
    hh_posterior_type2::AbstractVector{<:Real},
    hh_tya::AbstractVector{<:Integer},
    hh_af0::AbstractVector{<:Real},
    hh_as0::AbstractVector{<:Real},
    hh_aflav0::AbstractVector{<:Real},
    hh_p0::AbstractMatrix{<:Real},
    crn_choice::AbstractArray{<:Real, 3},
    crn_price::AbstractArray{<:Real, 4},
    crn_type::AbstractMatrix{<:Real},
    T_sim::Integer,
    N_draws::Integer,
    N_J::Integer,
    N_P::Integer,
    A_f::AbstractVector{<:Real},
    A_s::AbstractVector{<:Real},
    A_flav::AbstractVector{<:Real},
    P::AbstractMatrix{<:Real},
    n::AbstractVector{<:Real},
    n_flav::AbstractVector{<:Real},
    q_cig_raw::AbstractVector{<:Real},
    q_orig_ecig_raw::AbstractVector{<:Real},
    q_flav_ecig_raw::AbstractVector{<:Real},
    ψ_1::Real,
    ψ_2::Real,
    PSI_3::Real,
    φ_0::AbstractVector{<:Real},
    φ_1::AbstractVector{<:Real},
    L_chol::AbstractMatrix{<:Real}
)

    N_sub = length(idx_hh)

    sim_choices     = Array{Int}(undef, N_sub, T_sim, N_draws)
    sim_addiction_f = Array{Float64}(undef, N_sub, T_sim, N_draws)
    sim_addiction_s = Array{Float64}(undef, N_sub, T_sim, N_draws)
    sim_addiction   = Array{Float64}(undef, N_sub, T_sim, N_draws)
    sim_aflav       = Array{Float64}(undef, N_sub, T_sim, N_draws)
    sim_welfare     = Array{Float64}(undef, N_sub, T_sim, N_draws)
    sim_q_cig       = Array{Float64}(undef, N_sub, T_sim, N_draws)
    sim_q_orig_ecig = Array{Float64}(undef, N_sub, T_sim, N_draws)
    sim_q_flav_ecig = Array{Float64}(undef, N_sub, T_sim, N_draws)

    P_cig  = P[:, 1]
    P_ecig = P[:, 2]

    Threads.@threads for s in 1:N_sub

        h = idx_hh[s]
        tya_h = hh_tya[h]
        w1 = hh_posterior_type1[h]
        w2 = hh_posterior_type2[h]

        for d in 1:N_draws

            # Latent type for this draw, from the household's posterior
            u_type = crn_type[h, d]
            type_k = u_type <= w1 ? 1 : (u_type <= w1 + w2 ? 2 : 3)
            V_d = V_decision[type_k]
            V_n = V_now[type_k]

            # Start from the household's end-of-sample state
            a_f    = hh_af0[h]
            a_s    = hh_as0[h]
            a_flav = hh_aflav0[h]
            p_cig  = hh_p0[h, 1]
            p_ecig = hh_p0[h, 2]

            for t in 1:T_sim

                # Choice-specific values at the current state
                v_interp = interpolate_v_choice(
                    V_d, tya_h, a_f, a_s, a_flav, p_cig, p_ecig,
                    N_J, N_P, A_f, A_s, A_flav, P
                )
                replace!(v_interp, NaN => -Inf)

                # Ex-ante value at the current state (the welfare measure)
                sim_welfare[s, t, d] = interpolate_v_now(
                    V_n, tya_h, a_f, a_s, a_flav, p_cig, p_ecig,
                    N_P, A_f, A_s, A_flav, P
                )

                # Logit choice probabilities and inverse-CDF sampling
                v_max = maximum(v_interp)
                exp_v = exp.(v_interp .- v_max)
                probs = exp_v ./ sum(exp_v)

                u_draw = crn_choice[h, t, d]
                j = 1
                cum_prob = probs[1]
                while cum_prob < u_draw && j < N_J
                    j += 1
                    cum_prob += probs[j]
                end

                sim_choices[s, t, d]     = j
                sim_q_cig[s, t, d]       = q_cig_raw[j]
                sim_q_orig_ecig[s, t, d] = q_orig_ecig_raw[j]
                sim_q_flav_ecig[s, t, d] = q_flav_ecig_raw[j]

                # Update the three stocks
                a_f    = clamp(addiction_evolution(ψ_2, a_f, n[j]), A_f[1], A_f[end])
                a_s    = clamp(addiction_evolution(ψ_1, a_s, n[j]), A_s[1], A_s[end])
                a_flav = clamp(addiction_evolution(PSI_3, a_flav, n_flav[j]), A_flav[1], A_flav[end])
                sim_addiction_f[s, t, d] = a_f
                sim_addiction_s[s, t, d] = a_s
                sim_addiction[s, t, d]   = (a_f + a_s) / 2.0
                sim_aflav[s, t, d]       = a_flav

                # Update prices with the shared AR(1) shocks
                ε = L_chol * [crn_price[h, t, d, 1], crn_price[h, t, d, 2]]
                p_cig  = clamp(φ_0[1] + φ_1[1] * p_cig  + ε[1], P_cig[1], P_cig[end])
                p_ecig = clamp(φ_0[2] + φ_1[2] * p_ecig + ε[2], P_ecig[1], P_ecig[end])

            end
        end
    end

    return (
        choices     = sim_choices,
        addiction_f = sim_addiction_f,
        addiction_s = sim_addiction_s,
        addiction   = sim_addiction,
        aflav       = sim_aflav,
        welfare     = sim_welfare,
        q_cig       = sim_q_cig,
        q_orig_ecig = sim_q_orig_ecig,
        q_flav_ecig = sim_q_flav_ecig,
    )
end


#############################
# 5. Aggregation
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


# --- By Non-TYA Status ---

"""
Aggregate simulation results for non-TYA households (hh_tya_terminal == 1).
Same column structure as aggregate_simulation.
"""
function aggregate_simulation_by_non_tya(
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

    idx_non_tya = findall(h -> hh_tya_terminal[h] == 1, 1:N_HH)

    function _aggregate_subgroup(hh_indices)

        N_sub = length(hh_indices)
        # each household contributes N_draws Monte Carlo paths
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

    df_non_tya = _aggregate_subgroup(idx_non_tya)

    return df_non_tya
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
        # max(..., 1) avoids division by zero when a type has no households
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
        # max(..., 1) avoids division by zero when a type has no households
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


# --- By Non-TYA Status and Latent Type ---

"""
Aggregate simulation results for non-TYA households, further split by
modal latent type. Same column structure as aggregate_simulation.
"""
function aggregate_simulation_by_non_tya_type_k3(
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

    # Restrict to non-TYA households, then assign modal type via argmax of posterior probabilities
    idx_non_tya_type1 = findall(h -> hh_tya_terminal[h] == 1 && hh_posterior_type1[h] >= hh_posterior_type2[h] && hh_posterior_type1[h] >= hh_posterior_type3[h], 1:N_HH)
    idx_non_tya_type2 = findall(h -> hh_tya_terminal[h] == 1 && hh_posterior_type2[h] >  hh_posterior_type1[h] && hh_posterior_type2[h] >= hh_posterior_type3[h], 1:N_HH)
    idx_non_tya_type3 = findall(h -> hh_tya_terminal[h] == 1 && hh_posterior_type3[h] >  hh_posterior_type1[h] && hh_posterior_type3[h] >  hh_posterior_type2[h], 1:N_HH)

    function _aggregate_subgroup(hh_indices)

        N_sub = length(hh_indices)
        # max(..., 1) avoids division by zero when a type has no households
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

    df_non_tya_type1 = _aggregate_subgroup(idx_non_tya_type1)
    df_non_tya_type2 = _aggregate_subgroup(idx_non_tya_type2)
    df_non_tya_type3 = _aggregate_subgroup(idx_non_tya_type3)

    return df_non_tya_type1, df_non_tya_type2, df_non_tya_type3
end


# --- Extensive Margin ---

"""
Compute extensive margin initiation rates for TYA-present households.

`hh_eligible[h]` flags the baseline non-users for this outcome and horizon,
built by `build_initiation_base` in 02_CF_Mixture_Bootstrap.jl from the
observed purchase history (no purchase of the outcome's alternatives, and no
flavored purchase, in the household's last k observed months). The function
tracks the cumulative probability that such a household makes at least one
purchase of the outcome's alternatives by each horizon t, under both the
status quo and counterfactual simulation.

prevention_rate = sq_ever_initiated - cf_ever_initiated: the fraction of
would-be initiators whose first flavored purchase is prevented by the policy.

`alt_mask` (optional) is a length-N_J Bool vector naming the alternatives that
count as initiation. The default is the flavored e-cigarette alternatives
(cats 3,4,6,7). Passing `has_cig` or `is_nonflavored_ecig` tracks cigarette or
unflavored e-cigarette initiation, with `hh_eligible` built for that outcome.
The same keyword is accepted by every extensive-margin function below.
"""
function aggregate_extensive_margin_by_tya(
    sim_choices_sq::Array{Int, 3},
    sim_choices_cf::Array{Int, 3},
    hh_eligible::AbstractVector{Bool},
    cat_idx::AbstractVector{<:Integer},
    T_sim::Integer,
    hh_tya_terminal::Vector{Int},
    N_draws::Integer;
    alt_mask::Union{Nothing, AbstractVector{Bool}} = nothing
)

    N_HH = size(sim_choices_sq, 1)

    # Alternatives that count as initiation: flavored e-cigarettes (cats 3,4,6,7)
    # unless alt_mask names another set
    is_target_alt = alt_mask === nothing ? [cat_idx[j] in (3, 4, 6, 7) for j in 1:length(cat_idx)] : collect(Bool, alt_mask)

    # Baseline non-users for this outcome and horizon, from the observed purchase history
    non_user_mask = collect(Bool, hh_eligible)

    # TYA-present non-users
    idx_tya = findall(h -> hh_tya_terminal[h] == 2 && non_user_mask[h], 1:N_HH)

    function _compute_subgroup(hh_indices)

        N_sub   = length(hh_indices)
        # max(..., 1) avoids division by zero when there are no non-users
        N_paths = max(N_sub * N_draws, 1)

        sq_cumulative = zeros(Float64, T_sim)
        cf_cumulative = zeros(Float64, T_sim)

        for h in hh_indices
            for d in 1:N_draws
                sq_initiated = false
                cf_initiated = false
                for t in 1:T_sim
                    # Update initiation flags on the first target-alternative choice
                    if !sq_initiated && is_target_alt[sim_choices_sq[h, t, d]]
                        sq_initiated = true
                    end
                    if !cf_initiated && is_target_alt[sim_choices_cf[h, t, d]]
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


# --- Extensive Margin by Non-TYA Status ---

"""
Compute extensive margin initiation rates for non-TYA households
(hh_tya_terminal == 1). Same column structure as aggregate_extensive_margin_by_tya.
"""
function aggregate_extensive_margin_by_non_tya(
    sim_choices_sq::Array{Int, 3},
    sim_choices_cf::Array{Int, 3},
    hh_eligible::AbstractVector{Bool},
    cat_idx::AbstractVector{<:Integer},
    T_sim::Integer,
    hh_tya_terminal::Vector{Int},
    N_draws::Integer;
    alt_mask::Union{Nothing, AbstractVector{Bool}} = nothing
)

    N_HH = size(sim_choices_sq, 1)

    # Alternatives that count as initiation: flavored e-cigarettes (cats 3,4,6,7)
    # unless alt_mask names another set
    is_target_alt = alt_mask === nothing ? [cat_idx[j] in (3, 4, 6, 7) for j in 1:length(cat_idx)] : collect(Bool, alt_mask)

    # Baseline non-users for this outcome and horizon, from the observed purchase history
    non_user_mask = collect(Bool, hh_eligible)

    # Non-TYA non-users
    idx_non_tya = findall(h -> hh_tya_terminal[h] == 1 && non_user_mask[h], 1:N_HH)

    function _compute_subgroup(hh_indices)

        N_sub   = length(hh_indices)
        # max(..., 1) avoids division by zero when there are no non-users
        N_paths = max(N_sub * N_draws, 1)

        sq_cumulative = zeros(Float64, T_sim)
        cf_cumulative = zeros(Float64, T_sim)

        for h in hh_indices
            for d in 1:N_draws
                sq_initiated = false
                cf_initiated = false
                for t in 1:T_sim
                    # Update initiation flags on the first target-alternative choice
                    if !sq_initiated && is_target_alt[sim_choices_sq[h, t, d]]
                        sq_initiated = true
                    end
                    if !cf_initiated && is_target_alt[sim_choices_cf[h, t, d]]
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

    df_non_tya = _compute_subgroup(idx_non_tya)

    return df_non_tya
end


# --- Extensive Margin by Latent Type (K=3 Mixture) ---

"""
Compute extensive margin initiation rates by modal latent type (K=3 mixture),
among non-user households within each type group (`hh_eligible`, see
aggregate_extensive_margin_by_tya). Same column structure as that function.
"""
function aggregate_extensive_margin_by_type_k3(
    sim_choices_sq::Array{Int, 3},
    sim_choices_cf::Array{Int, 3},
    hh_eligible::AbstractVector{Bool},
    cat_idx::AbstractVector{<:Integer},
    T_sim::Integer,
    hh_posterior_type1::Vector{Float64},
    hh_posterior_type2::Vector{Float64},
    hh_posterior_type3::Vector{Float64},
    N_draws::Integer;
    alt_mask::Union{Nothing, AbstractVector{Bool}} = nothing
)

    N_HH = size(sim_choices_sq, 1)

    # Alternatives that count as initiation: flavored e-cigarettes (cats 3,4,6,7)
    # unless alt_mask names another set
    is_target_alt = alt_mask === nothing ? [cat_idx[j] in (3, 4, 6, 7) for j in 1:length(cat_idx)] : collect(Bool, alt_mask)

    # Baseline non-users for this outcome and horizon, from the observed purchase history
    non_user_mask = collect(Bool, hh_eligible)

    # Modal type assignment via argmax of posterior, restricted to non-users at baseline.
    # Ties broken in favor of lower type index (type1 >= type2 takes type1).
    idx_type1 = findall(h -> hh_posterior_type1[h] >= hh_posterior_type2[h] && hh_posterior_type1[h] >= hh_posterior_type3[h] && non_user_mask[h], 1:N_HH)
    idx_type2 = findall(h -> hh_posterior_type2[h] >  hh_posterior_type1[h] && hh_posterior_type2[h] >= hh_posterior_type3[h] && non_user_mask[h], 1:N_HH)
    idx_type3 = findall(h -> hh_posterior_type3[h] >  hh_posterior_type1[h] && hh_posterior_type3[h] >  hh_posterior_type2[h] && non_user_mask[h], 1:N_HH)

    function _compute_subgroup(hh_indices)

        N_sub   = length(hh_indices)
        # max(..., 1) avoids division by zero when a type has no non-users
        N_paths = max(N_sub * N_draws, 1)

        sq_cumulative = zeros(Float64, T_sim)
        cf_cumulative = zeros(Float64, T_sim)

        for h in hh_indices
            for d in 1:N_draws
                sq_initiated = false
                cf_initiated = false
                for t in 1:T_sim
                    # Update initiation flags on the first target-alternative choice
                    if !sq_initiated && is_target_alt[sim_choices_sq[h, t, d]]
                        sq_initiated = true
                    end
                    if !cf_initiated && is_target_alt[sim_choices_cf[h, t, d]]
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
    hh_eligible::AbstractVector{Bool},
    cat_idx::AbstractVector{<:Integer},
    T_sim::Integer,
    hh_tya_terminal::Vector{Int},
    hh_posterior_type1::Vector{Float64},
    hh_posterior_type2::Vector{Float64},
    hh_posterior_type3::Vector{Float64},
    N_draws::Integer;
    alt_mask::Union{Nothing, AbstractVector{Bool}} = nothing
)

    N_HH = size(sim_choices_sq, 1)

    # Alternatives that count as initiation: flavored e-cigarettes (cats 3,4,6,7)
    # unless alt_mask names another set
    is_target_alt = alt_mask === nothing ? [cat_idx[j] in (3, 4, 6, 7) for j in 1:length(cat_idx)] : collect(Bool, alt_mask)

    # Baseline non-users for this outcome and horizon, from the observed purchase history
    non_user_mask = collect(Bool, hh_eligible)

    # Restrict to TYA-present households, then assign modal type via argmax of
    # posterior probabilities, among non-users at baseline.
    idx_tya_type1 = findall(h -> hh_tya_terminal[h] == 2 && hh_posterior_type1[h] >= hh_posterior_type2[h] && hh_posterior_type1[h] >= hh_posterior_type3[h] && non_user_mask[h], 1:N_HH)
    idx_tya_type2 = findall(h -> hh_tya_terminal[h] == 2 && hh_posterior_type2[h] >  hh_posterior_type1[h] && hh_posterior_type2[h] >= hh_posterior_type3[h] && non_user_mask[h], 1:N_HH)
    idx_tya_type3 = findall(h -> hh_tya_terminal[h] == 2 && hh_posterior_type3[h] >  hh_posterior_type1[h] && hh_posterior_type3[h] >  hh_posterior_type2[h] && non_user_mask[h], 1:N_HH)

    function _compute_subgroup(hh_indices)

        N_sub   = length(hh_indices)
        # max(..., 1) avoids division by zero when a type has no non-users
        N_paths = max(N_sub * N_draws, 1)

        sq_cumulative = zeros(Float64, T_sim)
        cf_cumulative = zeros(Float64, T_sim)

        for h in hh_indices
            for d in 1:N_draws
                sq_initiated = false
                cf_initiated = false
                for t in 1:T_sim
                    # Update initiation flags on the first target-alternative choice
                    if !sq_initiated && is_target_alt[sim_choices_sq[h, t, d]]
                        sq_initiated = true
                    end
                    if !cf_initiated && is_target_alt[sim_choices_cf[h, t, d]]
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


# --- Extensive Margin by Non-TYA Status and Latent Type (K=3 Mixture) ---

"""
Compute extensive margin initiation rates among non-TYA households only,
split by modal latent type (K=3 mixture). Mirrors aggregate_extensive_margin_by_tya_type_k3,
but restricts to non-TYA households before assigning modal type.
"""
function aggregate_extensive_margin_by_non_tya_type_k3(
    sim_choices_sq::Array{Int, 3},
    sim_choices_cf::Array{Int, 3},
    hh_eligible::AbstractVector{Bool},
    cat_idx::AbstractVector{<:Integer},
    T_sim::Integer,
    hh_tya_terminal::Vector{Int},
    hh_posterior_type1::Vector{Float64},
    hh_posterior_type2::Vector{Float64},
    hh_posterior_type3::Vector{Float64},
    N_draws::Integer;
    alt_mask::Union{Nothing, AbstractVector{Bool}} = nothing
)

    N_HH = size(sim_choices_sq, 1)

    # Alternatives that count as initiation: flavored e-cigarettes (cats 3,4,6,7)
    # unless alt_mask names another set
    is_target_alt = alt_mask === nothing ? [cat_idx[j] in (3, 4, 6, 7) for j in 1:length(cat_idx)] : collect(Bool, alt_mask)

    # Baseline non-users for this outcome and horizon, from the observed purchase history
    non_user_mask = collect(Bool, hh_eligible)

    # Restrict to non-TYA households, then assign modal type via argmax of
    # posterior probabilities, among non-users at baseline.
    idx_non_tya_type1 = findall(h -> hh_tya_terminal[h] == 1 && hh_posterior_type1[h] >= hh_posterior_type2[h] && hh_posterior_type1[h] >= hh_posterior_type3[h] && non_user_mask[h], 1:N_HH)
    idx_non_tya_type2 = findall(h -> hh_tya_terminal[h] == 1 && hh_posterior_type2[h] >  hh_posterior_type1[h] && hh_posterior_type2[h] >= hh_posterior_type3[h] && non_user_mask[h], 1:N_HH)
    idx_non_tya_type3 = findall(h -> hh_tya_terminal[h] == 1 && hh_posterior_type3[h] >  hh_posterior_type1[h] && hh_posterior_type3[h] >  hh_posterior_type2[h] && non_user_mask[h], 1:N_HH)

    function _compute_subgroup(hh_indices)

        N_sub   = length(hh_indices)
        # max(..., 1) avoids division by zero when a type has no non-users
        N_paths = max(N_sub * N_draws, 1)

        sq_cumulative = zeros(Float64, T_sim)
        cf_cumulative = zeros(Float64, T_sim)

        for h in hh_indices
            for d in 1:N_draws
                sq_initiated = false
                cf_initiated = false
                for t in 1:T_sim
                    # Update initiation flags on the first target-alternative choice
                    if !sq_initiated && is_target_alt[sim_choices_sq[h, t, d]]
                        sq_initiated = true
                    end
                    if !cf_initiated && is_target_alt[sim_choices_cf[h, t, d]]
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

    df_non_tya_type1 = _compute_subgroup(idx_non_tya_type1)
    df_non_tya_type2 = _compute_subgroup(idx_non_tya_type2)
    df_non_tya_type3 = _compute_subgroup(idx_non_tya_type3)

    return df_non_tya_type1, df_non_tya_type2, df_non_tya_type3
end


#############################
# 6. Frontier
#############################

"""
    select_type2_households(hh_posterior_type1, hh_posterior_type2, hh_posterior_type3, hh_tya)

Return the indices of e-cigarette-persistent households, overall and split by
terminal TYA status.

A household is e-cigarette-persistent when type 2 has the largest posterior
probability, using the same tie-breaking as the by-type aggregators above
so that the household sets match those results.
`hh_tya` is the terminal TYA state, 1 = absent and 2 = present.

Returns a NamedTuple `(all, tya, non_tya)` of index vectors into 1:N_HH.
"""
function select_type2_households(
    hh_posterior_type1::AbstractVector{<:Real},
    hh_posterior_type2::AbstractVector{<:Real},
    hh_posterior_type3::AbstractVector{<:Real},
    hh_tya::AbstractVector{<:Integer}
)

    N_HH = length(hh_posterior_type2)

    is_type2 = [hh_posterior_type2[h] >  hh_posterior_type1[h] &&
                hh_posterior_type2[h] >= hh_posterior_type3[h] for h in 1:N_HH]

    idx_all     = findall(is_type2)
    idx_tya     = findall(h -> is_type2[h] && hh_tya[h] == 2, 1:N_HH)
    idx_non_tya = findall(h -> is_type2[h] && hh_tya[h] == 1, 1:N_HH)

    return (all = idx_all, tya = idx_tya, non_tya = idx_non_tya)
end


"""
    aggregate_subset_paths(sim, rows, cat_idx, N_J, T_sim)

Average one scenario's simulated paths over a subgroup of the simulated
households and over Monte Carlo draws, period by period.

`sim` is the NamedTuple returned by `simulate_scenario`, and `rows` indexes
its first dimension (household indices for a full-sample simulation, or
positions within the subset for a subset simulation). Returns a NamedTuple of length-`T_sim` vectors: mean
addiction, mean welfare, mean flavored mL, mean packs, and the shares of
household-months choosing the outside option, cigarettes, and any flavored
alternative.
"""
function aggregate_subset_paths(
    sim::NamedTuple,
    rows::AbstractVector{<:Integer},
    cat_idx::AbstractVector{<:Integer},
    N_J::Integer,
    T_sim::Integer
)

    N_draws = size(sim.choices, 3)

    # Floor at 1 so a draw with no households in the subgroup still writes a row
    N_total = max(length(rows) * N_draws, 1)

    is_flav = [cat_idx[j] in (3, 4, 6, 7) for j in 1:N_J]
    is_cig  = [cat_idx[j] == 1 for j in 1:N_J]
    is_out  = [cat_idx[j] == 0 for j in 1:N_J]

    mean_addiction   = zeros(Float64, T_sim)
    mean_welfare     = zeros(Float64, T_sim)
    mean_q_flav_ecig = zeros(Float64, T_sim)
    mean_q_cig       = zeros(Float64, T_sim)
    share_outside    = zeros(Float64, T_sim)
    share_cig        = zeros(Float64, T_sim)
    share_flav       = zeros(Float64, T_sim)

    for t in 1:T_sim
        for d in 1:N_draws
            for s in rows
                j = sim.choices[s, t, d]
                mean_addiction[t]   += sim.addiction[s, t, d]
                mean_welfare[t]     += sim.welfare[s, t, d]
                mean_q_flav_ecig[t] += sim.q_flav_ecig[s, t, d]
                mean_q_cig[t]       += sim.q_cig[s, t, d]
                share_outside[t]    += is_out[j]
                share_cig[t]        += is_cig[j]
                share_flav[t]       += is_flav[j]
            end
        end
        mean_addiction[t]   /= N_total
        mean_welfare[t]     /= N_total
        mean_q_flav_ecig[t] /= N_total
        mean_q_cig[t]       /= N_total
        share_outside[t]    /= N_total
        share_cig[t]        /= N_total
        share_flav[t]       /= N_total
    end

    return (
        mean_addiction   = mean_addiction,
        mean_welfare     = mean_welfare,
        mean_q_flav_ecig = mean_q_flav_ecig,
        mean_q_cig       = mean_q_cig,
        share_outside    = share_outside,
        share_cig        = share_cig,
        share_flav       = share_flav,
    )
end


"""
    frontier_summary(agg_sq, agg_cf, tau, omega_E_orig, T_sim)

Collapse a scenario's period-by-period aggregates into the scalar quantities
the frontier is drawn from.

Addiction effects are reported as percentage changes relative to the status
quo, both at the final period and averaged over the horizon. Welfare effects
are reported at period 1, which is the ex-ante comparison from the shared
end-of-sample state and the theoretically clean measure of a policy's
lifetime effect, and at the final period. Utils convert to dollars by
dividing by `omega_E_orig`, the marginal utility of income in original units,
matching the convention used for the paper's dollar figures. Cumulative
revenue is the tax rate times flavored mL summed over the horizon, per
household; it is zero for the ban and reported without any national scaling.
"""
function frontier_summary(
    agg_sq::NamedTuple,
    agg_cf::NamedTuple,
    tau::Real,
    omega_E_orig::Real,
    T_sim::Integer
)

    pct(cf, sq) = sq == 0 ? NaN : 100.0 * (cf - sq) / abs(sq)

    add_sq_36  = agg_sq.mean_addiction[T_sim]
    add_cf_36  = agg_cf.mean_addiction[T_sim]
    add_sq_avg = mean(agg_sq.mean_addiction)
    add_cf_avg = mean(agg_cf.mean_addiction)

    w_sq_1  = agg_sq.mean_welfare[1]
    w_cf_1  = agg_cf.mean_welfare[1]
    w_sq_36 = agg_sq.mean_welfare[T_sim]
    w_cf_36 = agg_cf.mean_welfare[T_sim]

    # Revenue only accrues under a finite tax
    cum_revenue = isfinite(tau) ? tau * sum(agg_cf.mean_q_flav_ecig) : 0.0

    return (
        addiction_pct_p36   = pct(add_cf_36, add_sq_36),
        addiction_pct_avg   = pct(add_cf_avg, add_sq_avg),
        addiction_diff_p36  = add_cf_36 - add_sq_36,
        addiction_diff_avg  = add_cf_avg - add_sq_avg,
        welfare_pct_p1      = pct(w_cf_1, w_sq_1),
        welfare_pct_p36     = pct(w_cf_36, w_sq_36),
        welfare_dollar_p1   = (w_cf_1 - w_sq_1) / omega_E_orig,
        welfare_dollar_p36  = (w_cf_36 - w_sq_36) / omega_E_orig,
        flav_share_p36      = agg_cf.share_flav[T_sim],
        cig_share_p36       = agg_cf.share_cig[T_sim],
        outside_share_p36   = agg_cf.share_outside[T_sim],
        cum_revenue_per_hh  = cum_revenue,
    )
end


"""
    interpolate_tau_star(taus, tax_effects, ban_effect)

Find the tax rate at which the tax's addiction effect equals the ban's, by
linear interpolation over the sorted tax grid.

`tax_effects[k]` is the tax's addiction change at rate `taus[k]` and
`ban_effect` the ban's, both negative for a reduction. The grid is scanned
from the highest rate down, so with simulation noise in the tax curve τ* is
the largest rate at which the tax first matches the ban rather than a spurious
early crossing. Returns `NaN` when the ban's effect lies outside the range the
grid spans, which means the grid does not reach far enough for the tax to
match the ban.
"""
function interpolate_tau_star(
    taus::AbstractVector{<:Real},
    tax_effects::AbstractVector{<:Real},
    ban_effect::Real
)

    order = sortperm(taus)
    τ = collect(Float64, taus[order])
    e = collect(Float64, tax_effects[order])

    for k in (length(τ) - 1):-1:1
        if (e[k] - ban_effect) * (e[k+1] - ban_effect) <= 0 && e[k+1] != e[k]
            w = (ban_effect - e[k]) / (e[k+1] - e[k])
            return τ[k] + w * (τ[k+1] - τ[k])
        end
    end

    return NaN
end


"""
    interpolate_at_tau(taus, values, tau_star)

Linearly interpolate any per-rate quantity at `tau_star`. Used to read off
the welfare cost and revenue at the rate that matches the ban.
"""
function interpolate_at_tau(
    taus::AbstractVector{<:Real},
    values::AbstractVector{<:Real},
    tau_star::Real
)

    isnan(tau_star) && return NaN

    order = sortperm(taus)
    τ = collect(Float64, taus[order])
    v = collect(Float64, values[order])

    for k in 1:(length(τ) - 1)
        if τ[k] <= tau_star <= τ[k+1]
            w = (tau_star - τ[k]) / (τ[k+1] - τ[k])
            return v[k] + w * (v[k+1] - v[k])
        end
    end

    return NaN
end
